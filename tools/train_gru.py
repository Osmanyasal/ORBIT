#!/usr/bin/env python3
"""Train models that predict the best OpenMP configuration of a code region.

Two model types share the whole pipeline (data, cross-validation, ``predict``, ``generate``);
choose with ``train --model``: ``gru`` (default, recurrent over the observations) or ``mlp``
(feed-forward DNN: the same dense layers encode every observation, the encodings are averaged).
Observation order carries no information, so both reach similar accuracy; see the README.

Data layout (``--data-dir``): every folder below it that contains both
``all_configs.csv`` and ``best_per_region.json`` is treated as one application.

Problem formulation
-------------------
* One training sample = one region.
* Input  = a set of profiling observations of that region. Each one holds
  the performance metrics (default: ipc, l2_hit_ratio, l3_mpki,
  branch_mispr_ratio, plus time_s and power_w - restrict with --features) and
  the configuration they were measured under (threads, sched, chunk,
  frequency), since the metrics depend on it.
* Output = four heads: threads, sched, chunk, frequency. ``sched`` is a plain classification.
  ``threads``, ``chunk`` and ``frequency`` are classifications over *positions on a normalised
  axis*, not over the values seen in training: threads / max threads of the machine, chunk on a log2
  axis up to the largest trained chunk, frequency as the fraction of the machine's frequency range.
  The bins of these axes are fixed (--thread-bins, --freq-bins), so the model never depends on which
  exact settings were swept. At prediction time the bin distribution is turned into a continuous
  position (--readout, default: the most likely bin; "median" interpolates between bins), de-normalised with the *target* machine's core count
  and frequency range and snapped to the nearest valid setting there. So the model can return
  settings that were never in the training sweep and can be used on another machine.
* Label  = the "fastest" / "best_edp" entry of best_per_region.json. One model
  is trained per target. Because the optimum of 540 configs is often a
  near-tie, the label is blended (--hard-weight) with soft targets derived from
  all_configs.csv, where every config is weighted by how close it is to the
  optimum (--soft-tau).
* The observations of a region are the rows of all_configs.csv. Every epoch a
  fresh random subset (random length 1..max-len, random order) is drawn per
  region, which gives a lot of augmentation and makes the model usable with
  any number of observations at inference time.

Evaluation
----------
Leave-one-application-out cross-validation (rows of one region share a label,
so a random row split would leak). Besides exact accuracy it reports how much
slower (fastest) / higher EDP (best_edp) the predicted config is relative to
the true optimum, looked up in all_configs.csv, next to two simple baselines.

Usage
-----
    python train_gru.py train    --data-dir ../../P10-data --out-dir models [--model gru|mlp]
    python train_gru.py generate --model-dir models --snapshots <snapshot run or folder of runs> \\
                                 --output predicted_best_per_region.json \\
                                 [--max-threads N] [--freq-range MIN_KHZ MAX_KHZ] [--freq-step KHZ]
    python train_gru.py predict  --model-dir models --input some_all_configs.csv

``generate`` is the application-execution step: run the application once in ORBIT snapshot
mode (or a few times with different settings), point ``--snapshots`` at the resulting
execution folder(s) and it writes a best_per_region.json-style file (``fastest`` and
``best_edp`` entries holding threads/sched/chunk/frequency per region) that ORBIT loads via
ORBIT_OPTIMIZED_CONF.

The machine description used for ``generate`` / ``predict`` defaults to what ``scan.sh`` sweeps on the
current machine (nproc threads, the cpufreq range in 200 MHz steps); override it with
--max-threads / --freq-range / --freq-step when predicting for another machine.

Requires: numpy, pandas, torch (and ORBIT/tools/best_settings.py for ``generate``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HEADS = ("threads", "sched", "chunk", "frequency")
UNIT_HEADS = ("threads", "chunk", "frequency")  # heads over positions on a normalised [0, 1] axis
READOUTS = ("median", "mean", "argmax")
CPUFREQ_DIR = Path("/sys/devices/system/cpu/cpu0/cpufreq")
SWEEP_MIN_KHZ = 1_000_000  # scan.sh never sweeps below 1 GHz
TARGETS = ("fastest", "best_edp")
TARGET_METRIC = {"fastest": "time_s", "best_edp": "edp"}
DEFAULT_FEATURES = ("ipc", "l2_hit_ratio", "l3_mpki", "branch_mispr_ratio", "time_s", "power_w")
CLIP = 5.0
REQUIRED_COLUMNS = ("region", *HEADS, "time_s", "edp")
BASELINES = ("baseline: majority label", "baseline: best global config")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Machine:
    """Valid values of the tunable settings on one machine. The model works on positions within these
    ranges (see to_unit), never on absolute values, which is what lets it transfer between machines."""
    threads: tuple
    chunk: tuple
    frequency: tuple
    chunk_max: int  # chunk axis scale, fixed by the training data so it means the same everywhere


@dataclass
class Region:
    key: str
    app: str
    feats: np.ndarray  # (n, n_features) raw metric values
    ctx: np.ndarray  # (n, 4) normalised config the metrics were measured under
    machine: Machine
    labels: dict = field(default_factory=dict)  # target -> (4,) class indices
    perf: dict = field(default_factory=dict)  # target -> dense (T,S,C,F) metric grid


def to_unit(head: str, x, m: Machine) -> np.ndarray:
    """Position of a setting on its normalised axis (0..1 inside the machine's range)."""
    x = np.asarray(x, dtype=float)
    if head == "threads":
        return x / max(m.threads)
    if head == "frequency":
        lo, hi = min(m.frequency), max(m.frequency)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return np.log2(x) / math.log2(m.chunk_max) if m.chunk_max > 1 else np.zeros_like(x)


def snap(head: str, u: float, m: Machine) -> int:
    """Valid setting of the machine whose position is closest to u."""
    cands = np.asarray(getattr(m, head))
    return int(cands[np.abs(to_unit(head, cands, m) - u).argmin()])


def default_axes(vocab: dict, args) -> dict:
    chunk_max = max(vocab["chunk"])
    return {"chunk_max": int(chunk_max),
            "bins": {"threads": args.thread_bins, "frequency": args.freq_bins,
                     "chunk": max(2, round(math.log2(chunk_max)) + 1)}}  # one bin per power of two


def head_sizes(vocab: dict, axes: dict) -> dict:
    return {h: len(vocab[h]) if h == "sched" else axes["bins"][h] for h in HEADS}


def unit_to_dist(u, k: int) -> np.ndarray:
    """Positions in [0, 1] -> distributions over k evenly spaced bins (mass split between the two
    nearest bins, so no information is lost when a value falls between bins)."""
    pos = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), 0.0, 1.0) * (k - 1)
    lo = np.minimum(np.floor(pos).astype(int), k - 2)
    frac = pos - lo
    dist = np.zeros((pos.size, k))
    rows = np.arange(pos.size)
    dist[rows, lo] = 1.0 - frac
    dist[rows, lo + 1] += frac
    return dist


def bin_matrix(head: str, vocab: dict, m: Machine, k: int) -> np.ndarray:
    """(len(vocab[head]), k): the bin distribution of each vocabulary value on machine m."""
    return unit_to_dist(to_unit(head, vocab[head], m), k)


def readout_unit(p: torch.Tensor, readout: str) -> torch.Tensor:
    """Bin probabilities (..., k) -> continuous position in [0, 1]. The median treats each bin's mass
    as spread uniformly over the bin, so it can land between bins (and between trained settings)."""
    k = p.shape[-1]
    pos = torch.linspace(0.0, 1.0, k)
    if readout == "argmax":
        return pos[p.argmax(-1)]
    if readout == "mean":
        return (p * pos).sum(-1) / p.sum(-1)
    edges = ((torch.arange(k + 1, dtype=torch.float32) - 0.5) / (k - 1)).clamp(0.0, 1.0)
    cdf = p.cumsum(-1) / p.sum(-1, keepdim=True)
    j = (cdf < 0.5).sum(-1).clamp(max=k - 1)
    before = cdf.gather(-1, (j - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1) * (j > 0)
    mass = p.gather(-1, j.unsqueeze(-1)).squeeze(-1) / p.sum(-1)
    frac = ((0.5 - before) / mass.clamp(min=1e-12)).clamp(0.0, 1.0)
    return edges[j] + frac * (edges[j + 1] - edges[j])


def decode_raw(probs: dict, machines: list, vocab: dict, readout: str) -> list:
    """probs[h]: (n_regions, draws, n_bins) -> [region][draw] -> {head: setting valid on that machine}."""
    units = {h: readout_unit(probs[h], readout).numpy() for h in UNIT_HEADS}
    sched = probs["sched"].argmax(-1).numpy()
    n, draws = sched.shape
    return [[{h: vocab["sched"][sched[i, d]] if h == "sched" else snap(h, units[h][i, d], machines[i])
              for h in HEADS} for d in range(draws)] for i in range(n)]


def decode_configs(probs: dict, machines: list, vocab: dict, readout: str) -> np.ndarray:
    """Like decode_raw, as (n_regions, draws, 4) indices into the vocabulary (for scoring)."""
    return np.array([[[vocab[h].index(cfg[h]) for h in HEADS] for cfg in draws]
                     for draws in decode_raw(probs, machines, vocab, readout)])


def sweep_frequencies(fmin: int, fmax: int, step: int) -> tuple:
    """The frequencies scan.sh sweeps for a hardware range: the exact maximum (possibly a turbo
    marker), then every step below it down to the rounded minimum (never below 1 GHz)."""
    start = max((fmin // step) * step, SWEEP_MIN_KHZ)
    top = (fmax // step) * step
    return tuple(sorted({fmax, *range(top, start - 1, -step)}))


def inference_machine(opts: dict, df: pd.DataFrame, chunk_max: int) -> Machine:
    """The machine to predict for: explicit options, else this machine (like scan.sh), else the
    range spanned by the observations."""
    if opts.get("freq_range"):
        fmin, fmax = opts["freq_range"]
    else:
        try:
            fmin = int((CPUFREQ_DIR / "cpuinfo_min_freq").read_text())
            fmax = int((CPUFREQ_DIR / "cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            seen = sorted(int(f) for f in df["frequency"].unique()) if "frequency" in df.columns else []
            if len(seen) < 2:
                sys.exit("cannot determine the frequency range of the target machine (no cpufreq in sysfs "
                         "and the observations use one frequency); pass --freq-range MIN_KHZ MAX_KHZ")
            fmin, fmax = seen[0], seen[-1]
    if fmax <= fmin:
        sys.exit(f"invalid frequency range {fmin}..{fmax} kHz")
    tmax = int(opts.get("max_threads") or os.cpu_count())
    pow2 = lambda top: sorted({2 ** k for k in range(int(math.log2(top)) + 1)} | {top})
    return Machine(threads=tuple(pow2(tmax)), chunk=tuple(pow2(chunk_max)),
                   frequency=sweep_frequencies(fmin, fmax, opts.get("freq_step") or 200000),
                   chunk_max=int(chunk_max))


def training_machines(frames: list, vocab: dict) -> list:
    """One Machine per application, from the settings its sweep covers."""
    return [Machine(threads=tuple(sorted(int(v) for v in f["threads"].unique())),
                    chunk=tuple(sorted(int(v) for v in f["chunk"].unique())),
                    frequency=tuple(sorted(int(v) for v in f["frequency"].unique())),
                    chunk_max=int(max(vocab["chunk"])))
            for f in frames]


def discover_apps(root: Path) -> list[Path]:
    dirs = sorted(
        p.parent
        for p in root.rglob("all_configs.csv")
        if (p.parent / "best_per_region.json").is_file()
    )
    if not dirs:
        sys.exit(f"No folder with all_configs.csv + best_per_region.json under {root}")
    return dirs


def app_name(path: Path) -> str:
    name = path.name
    return name[len("best_settings_"):] if name.startswith("best_settings_") else name


def build_vocab(frames: list[pd.DataFrame]) -> dict:
    cat = pd.concat([f[list(HEADS)] for f in frames], ignore_index=True)
    return {
        "threads": sorted(int(v) for v in cat["threads"].unique()),
        "sched": sorted(str(v) for v in cat["sched"].unique()),
        "chunk": sorted(int(v) for v in cat["chunk"].unique()),
        "frequency": sorted(int(v) for v in cat["frequency"].unique()),
    }


def context_features(threads, sched, chunk, freq, vocab: dict, machine: Machine) -> np.ndarray:
    """Encode the measured-under configuration as 4 values in roughly [-1, 1] (relative to the machine)."""
    unknown = set(map(str, sched)) - set(vocab["sched"])
    if unknown:
        raise ValueError(f"unknown sched value(s) {sorted(unknown)}; model knows {vocab['sched']}")
    sched_idx = np.array([vocab["sched"].index(str(s)) for s in sched], dtype=float)
    n_sched = len(vocab["sched"])
    cols = [
        2.0 * to_unit("threads", np.asarray(threads), machine) - 1.0,
        2.0 * sched_idx / (n_sched - 1) - 1.0 if n_sched > 1 else np.zeros_like(sched_idx),
        2.0 * to_unit("chunk", np.asarray(chunk), machine) - 1.0,
        2.0 * to_unit("frequency", np.asarray(freq), machine) - 1.0,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def _dense_grid(df: pd.DataFrame, metric: str, vocab: dict) -> np.ndarray:
    grid = np.full(tuple(len(vocab[h]) for h in HEADS), np.nan)
    maps = {h: {v: i for i, v in enumerate(vocab[h])} for h in HEADS}
    mean = df.groupby(list(HEADS), sort=False)[metric].mean()
    for (t, s, c, f), val in mean.items():
        grid[maps["threads"][int(t)], maps["sched"][str(s)], maps["chunk"][int(c)],
             maps["frequency"][int(f)]] = val
    return grid


def load_regions(data_dir: Path, features: tuple) -> tuple[list[Region], dict]:
    apps = discover_apps(data_dir)
    frames = []
    need = list(REQUIRED_COLUMNS) + list(features)
    for d in apps:
        df = pd.read_csv(d / "all_configs.csv")
        missing = [c for c in need if c not in df.columns]
        if missing:
            sys.exit(f"{d / 'all_configs.csv'} is missing columns: {missing}")
        before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
        if len(df) < before:
            print(f"[warn] {d.name}: dropped {before - len(df)} rows with missing/inf values")
        frames.append(df)
    vocab = build_vocab(frames)
    machines = training_machines(frames, vocab)

    regions: list[Region] = []
    for d, df, machine in zip(apps, frames, machines):
        with open(d / "best_per_region.json") as fh:
            best = json.load(fh)
        app = app_name(d)
        for name, g in df.groupby("region", sort=False):
            entry = best.get(name)
            if not isinstance(entry, dict):
                print(f"[warn] {app}: region {name} has no entry in best_per_region.json, skipped")
                continue
            reg = Region(
                key=f"{app}/{name}",
                app=app,
                feats=g[list(features)].to_numpy(dtype=np.float64),
                ctx=context_features(g["threads"], g["sched"], g["chunk"], g["frequency"], vocab, machine),
                machine=machine,
            )
            for target in TARGETS:
                lab = entry.get(target)
                try:
                    idx = [vocab[h].index(type(vocab[h][0])(lab[h])) for h in HEADS]
                except (TypeError, KeyError, ValueError):
                    print(f"[warn] {reg.key}: unusable '{target}' label, skipped for that target")
                    continue
                reg.labels[target] = np.array(idx, dtype=np.int64)
                reg.perf[target] = _dense_grid(g, TARGET_METRIC[target], vocab)
            if reg.labels:
                regions.append(reg)
    return regions, vocab


# --------------------------------------------------------------------------- #
# Feature preprocessing and tensors
# --------------------------------------------------------------------------- #
def _slog(x, scale):
    return np.sign(x) * np.log1p(np.abs(x) / scale)


class Preproc:
    """signed log1p(x/scale) -> standardise -> clip. Fitted on training rows only."""

    def __init__(self, scale, mean, std):
        self.scale, self.mean, self.std = (np.asarray(a, dtype=np.float64) for a in (scale, mean, std))

    @classmethod
    def fit(cls, raw: np.ndarray) -> "Preproc":
        scale = []
        for col in raw.T:
            nz = np.abs(col[np.abs(col) > 0])
            scale.append(float(np.median(nz)) if nz.size else 1.0)
        scale = np.array(scale)
        t = _slog(raw, scale)
        return cls(scale, t.mean(0), np.maximum(t.std(0), 1e-6))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        z = (_slog(raw, self.scale) - self.mean) / self.std
        return np.clip(z, -CLIP, CLIP).astype(np.float32)

    def state(self) -> dict:
        return {"scale": self.scale.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "Preproc":
        return cls(s["scale"], s["mean"], s["std"])


def build_tensors(regions: list[Region], pre: Preproc, use_ctx: bool):
    seqs = []
    for r in regions:
        x = pre(r.feats)
        seqs.append(np.concatenate([x, r.ctx], axis=1) if use_ctx else x)
    n_max = max(len(s) for s in seqs)
    xpad = np.zeros((len(seqs), n_max, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        xpad[i, : len(s)] = s
    nrows = np.array([len(s) for s in seqs])
    return torch.from_numpy(xpad), torch.from_numpy(nrows)


def sample_batch(xpad, nrows, reg, max_len, lens, gen):
    """Random observation subsets (random order) of the given regions."""
    n_max = xpad.shape[1]
    max_len = min(max_len, n_max)
    rnd = torch.rand(reg.numel(), n_max, generator=gen)
    rnd.masked_fill_(torch.arange(n_max)[None, :] >= nrows[reg][:, None], 2.0)
    idx = rnd.argsort(dim=1)[:, :max_len]
    x = xpad[reg[:, None], idx]
    return x, torch.minimum(lens, nrows[reg]).clamp(min=1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConfigGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, head_sizes: dict):
        super().__init__()
        self.in_dim = in_dim
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, n) for h, n in head_sizes.items()})

    def forward(self, x, lens):
        out, _ = self.gru(self.embed(x))
        h = self.drop(out[torch.arange(x.size(0)), lens - 1])
        return {k: head(h) for k, head in self.heads.items()}


class ConfigMLP(nn.Module):
    """Feed-forward DNN. The same dense stack encodes every observation, the encodings are averaged
    over the observations of the sample (so their order is irrelevant) and linear heads predict the
    four settings. With a single observation this is a plain MLP classifier."""

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, head_sizes: dict):
        super().__init__()
        self.in_dim = in_dim
        blocks, width = [], in_dim
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.encoder = nn.Sequential(*blocks)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, n) for h, n in head_sizes.items()})

    def forward(self, x, lens):
        valid = (torch.arange(x.size(1))[None, :] < lens[:, None]).unsqueeze(-1).to(x.dtype)
        h = (self.encoder(x) * valid).sum(1) / lens[:, None].to(x.dtype)
        return {k: head(h) for k, head in self.heads.items()}


MODELS = {"gru": ConfigGRU, "mlp": ConfigMLP}
DEFAULT_LAYERS = {"gru": 1, "mlp": 3}


def soft_label_matrix(n: int, sigma: float) -> torch.Tensor:
    """Row i = target distribution when the true class is i (Gaussian over class distance)."""
    if sigma <= 0:
        return torch.eye(n)
    d = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    m = torch.exp(-(d.float() ** 2) / (2 * sigma ** 2))
    return m / m.sum(1, keepdim=True)


def soft_marginals(grid: np.ndarray, tau: float) -> list[np.ndarray]:
    """Per-head target distributions from the measured grid: weight exp(-(ratio-1)/tau) per config."""
    w = np.exp(-(grid / np.nanmin(grid) - 1.0) / tau)
    w = np.nan_to_num(w, nan=0.0)
    w /= w.sum()
    return [w.sum(axis=tuple(a for a in range(w.ndim) if a != j)) for j in range(w.ndim)]


def build_targets(train: list[Region], target: str, vocab: dict, axes: dict, args) -> dict:
    """Per-head target distributions: the JSON label placed on the head's axis (smoothed over
    neighbouring bins), blended with distributions derived from the measured grid."""
    bins = axes["bins"]
    sigma = {"threads": args.thread_sigma, "chunk": args.chunk_sigma, "frequency": args.freq_sigma}
    tgt = {}
    for j, h in enumerate(HEADS):
        idx = [r.labels[target][j] for r in train]
        if h == "sched":
            tgt[h] = torch.eye(len(vocab[h]))[torch.tensor(idx)]
            continue
        u = [float(to_unit(h, vocab[h][i], r.machine)) for i, r in zip(idx, train)]
        tgt[h] = torch.from_numpy(unit_to_dist(u, bins[h])).float() @ soft_label_matrix(bins[h], sigma[h])
    if args.soft_tau > 0:
        per_region = [soft_marginals(r.perf[target], args.soft_tau) for r in train]
        for j, h in enumerate(HEADS):
            if h == "sched":
                grid = np.stack([m[j] for m in per_region])
            else:
                grid = np.stack([m[j] @ bin_matrix(h, vocab, r.machine, bins[h])
                                 for m, r in zip(per_region, train)])
            tgt[h] = args.hard_weight * tgt[h] + (1.0 - args.hard_weight) * torch.tensor(grid, dtype=torch.float32)
    return tgt


def train_model(train: list[Region], target: str, vocab: dict, axes: dict, args, tag: str = ""):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train = [r for r in train if target in r.labels]
    pre = Preproc.fit(np.concatenate([r.feats for r in train]))
    xpad, nrows = build_tensors(train, pre, not args.no_config_context)
    model = MODELS[args.model](xpad.shape[2], args.hidden, args.layers, args.dropout, head_sizes(vocab, axes))
    steps_per_epoch = math.ceil(len(train) * args.samples_per_region / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    tgt = build_targets(train, target, vocab, axes, args)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.arange(len(train)).repeat(args.samples_per_region)
        order = order[torch.randperm(order.numel(), generator=gen)]
        total = 0.0
        for start in range(0, order.numel(), args.batch_size):
            reg = order[start:start + args.batch_size]
            lens = torch.randint(1, args.max_len + 1, (reg.numel(),), generator=gen)
            x, lens = sample_batch(xpad, nrows, reg, args.max_len, lens, gen)
            if args.noise > 0:
                x = x + args.noise * torch.randn(x.shape, generator=gen)
            logits = model(x, lens)
            loss = sum(-(tgt[h][reg] * torch.log_softmax(logits[h], -1)).sum(1).mean() for h in HEADS)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        history.append(total / steps_per_epoch)
        if args.verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    {tag} epoch {epoch:3d}/{args.epochs}  loss {history[-1]:.4f}")
    model.eval()
    return model, pre, history


# --------------------------------------------------------------------------- #
# Prediction / scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_proba(model, xpad, nrows, length, draws, seed):
    """Per-head probabilities, shape (n_regions, draws, n_classes)."""
    gen = torch.Generator().manual_seed(seed)
    reg = torch.arange(xpad.shape[0])
    outs = {h: [] for h in HEADS}
    for _ in range(draws):
        x, lens = sample_batch(xpad, nrows, reg, length, torch.full((reg.numel(),), length), gen)
        for h, lg in model(x, lens).items():
            outs[h].append(torch.softmax(lg, -1))
    return {h: torch.stack(v, 1) for h, v in outs.items()}


def ratio_to_oracle(region: Region, target: str, cfg) -> float:
    grid = region.perf[target]
    return float(grid[tuple(cfg)] / np.nanmin(grid))


def global_best_config(train: list[Region], target: str) -> np.ndarray:
    """Single config with the lowest geometric-mean ratio-to-oracle over training regions."""
    logr = np.stack([np.log(r.perf[target] / np.nanmin(r.perf[target])) for r in train])
    mean = np.where(np.isnan(logr).any(0), np.inf, logr.mean(0))
    return np.array(np.unravel_index(np.argmin(mean), mean.shape))


def majority_config(train: list[Region], target: str) -> np.ndarray:
    labs = np.stack([r.labels[target] for r in train])
    return np.array([np.bincount(labs[:, j]).argmax() for j in range(len(HEADS))])


def score_rows(method, regions, target, preds) -> list[dict]:
    """preds: (n_regions, n_draws, 4) -> one row per (region, draw)."""
    rows = []
    for r, p in zip(regions, preds):
        for cfg in p:
            hit = cfg == r.labels[target]
            rows.append({"method": method, "app": r.app, "region": r.key,
                         **{f"acc_{h}": bool(hit[j]) for j, h in enumerate(HEADS)},
                         "acc_all": bool(hit.all()),
                         "ratio": ratio_to_oracle(r, target, cfg)})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for method, g in df.groupby("method", sort=False):
        ratio = g["ratio"].to_numpy()
        rows[method] = {
            **{f"{h}%": 100 * g[f"acc_{h}"].mean() for h in HEADS},
            "exact%": 100 * g["acc_all"].mean(),
            "geomean_loss%": 100 * (np.exp(np.nanmean(np.log(ratio))) - 1),
            "median_loss%": 100 * (np.nanmedian(ratio) - 1),
            "p90_loss%": 100 * (np.nanpercentile(ratio, 90) - 1),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def make_folds(regions: list[Region], mode: str, seed: int):
    if mode == "app":
        apps = sorted({r.app for r in regions})
        return [(a, [i for i, r in enumerate(regions) if r.app == a]) for a in apps]
    perm = np.random.default_rng(seed).permutation(len(regions))
    k = min(5, len(regions))
    return [(f"fold{i}", sorted(perm[i::k].tolist())) for i in range(k)]


def cross_validate(regions, target, vocab, axes, args) -> pd.DataFrame:
    regions = [r for r in regions if target in r.labels]
    rows = []
    for name, test_idx in make_folds(regions, args.cv_mode, args.seed):
        test_set = set(test_idx)
        train = [r for i, r in enumerate(regions) if i not in test_set]
        test = [regions[i] for i in test_idx]
        if not train:
            continue
        t0 = time.time()
        model, pre, _ = train_model(train, target, vocab, axes, args, tag=f"[{target}/{name}]")
        xpad, nrows = build_tensors(test, pre, not args.no_config_context)
        machines = [r.machine for r in test]

        def decode(probs, readout=args.readout):
            return decode_configs(probs, machines, vocab, readout)

        for length in sorted({1, args.max_len}):
            probs = predict_proba(model, xpad, nrows, length, args.eval_draws, args.seed + 1)
            rows += score_rows(f"{args.model.upper()} L={length}", test, target, decode(probs))
            if length == args.max_len:
                mean_probs = {h: p.mean(1, keepdim=True) for h, p in probs.items()}
                rows += score_rows(avg_method_name(args), test, target, decode(mean_probs))
                if args.readout != "median":
                    rows += score_rows(avg_method_name(args, "median"), test, target,
                                       decode(mean_probs, "median"))
        for label, cfg in zip(BASELINES, (majority_config(train, target), global_best_config(train, target))):
            rows += score_rows(label, test, target, np.tile(cfg, (len(test), 1, 1)))
        rows += score_rows("oracle", test, target, np.stack([r.labels[target] for r in test])[:, None, :])
        print(f"  fold {name:<6} train={len(train):3d} test={len(test):3d} regions  ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def avg_method_name(args, readout: str | None = None) -> str:
    return f"{args.model.upper()} L={args.max_len}, avg of {args.eval_draws} subsets, {readout or args.readout} readout"


def report_cv(df: pd.DataFrame, target: str, n_regions: int, args):
    unit = "time" if target == "fastest" else "EDP"
    print(f"\n=== {target}: cross-validation ({args.cv_mode}), {n_regions} regions ===")
    print(f"<head>%/exact% = exact-match accuracy; *_loss% = extra {unit} of the predicted config "
          f"vs. the true optimum (lower is better)")
    print(summarize(df).round(1).to_string())
    methods = [avg_method_name(args), BASELINES[1]]
    sub = df[df["method"].isin(methods)]
    per_app = sub.groupby(["app", "method"])["ratio"].apply(
        lambda s: 100 * (np.exp(np.nanmean(np.log(s))) - 1)).unstack()[methods]
    print(f"\ngeomean {unit} loss% per held-out group:")
    print(per_app.round(1).to_string())


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, pre, vocab, axes, target, args, n_regions, features):
    torch.save({
        "state_dict": model.state_dict(),
        "model": args.model,
        "arch": {"in_dim": model.in_dim, "hidden": args.hidden, "layers": args.layers,
                 "dropout": args.dropout},
        "vocab": vocab,
        "axes": axes,
        "readout": args.readout,
        "features": list(features),
        "use_config_context": not args.no_config_context,
        "max_len": args.max_len,
        "preproc": pre.state(),
        "target": target,
        "n_train_regions": n_regions,
    }, path)


def load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "axes" not in ck:
        sys.exit(f"{path} was trained with per-value classes for threads/chunk/frequency and cannot "
                 f"transfer to other settings or machines; retrain with 'train'")
    a = ck["arch"]
    model = MODELS[ck.get("model", "gru")](a["in_dim"], a["hidden"], a["layers"], a["dropout"],
                                           head_sizes(ck["vocab"], ck["axes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_train(args):
    if args.layers is None:
        args.layers = DEFAULT_LAYERS[args.model]
    features = tuple(f.strip() for f in args.features.split(",") if f.strip())
    out_dir = Path(args.out_dir)
    regions, vocab = load_regions(Path(args.data_dir), features)
    apps = sorted({r.app for r in regions})
    print(f"Loaded {len(regions)} regions from {len(apps)} applications: {', '.join(apps)}")
    if args.cv_mode == "app" and len(apps) < 2:
        print("[warn] only one application found, falling back to 5-fold CV by region")
        args.cv_mode = "region"
    print(f"Features: {', '.join(features)}"
          f"{'' if args.no_config_context else ' + measured-under config (threads, sched, chunk, frequency)'}")
    print("Config space: " + ", ".join(f"{h}={len(vocab[h])}" for h in HEADS))
    axes = default_axes(vocab, args)
    print("Output heads: " + ", ".join(f"{h}={n} {'classes' if h == 'sched' else 'bins'}"
                                       for h, n in head_sizes(vocab, axes).items())
          + f"  (readout: {args.readout})")
    print(f"Model: {args.model} ({args.layers} layer{'s' if args.layers != 1 else ''}, hidden {args.hidden}, "
          f"{args.max_len} observation{'s' if args.max_len != 1 else ''} per sample)")

    summary = {}
    for target in args.targets:
        n = sum(target in r.labels for r in regions)
        if args.cv_mode != "none":
            print(f"\n--- {target}: cross-validation ---")
            cv = cross_validate(regions, target, vocab, axes, args)
            report_cv(cv, target, n, args)
            summary[target] = json.loads(summarize(cv).round(3).to_json(orient="index"))
        if args.cv_only:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {target}: training final model on all {n} regions ---")
        model, pre, hist = train_model([r for r in regions if target in r.labels], target, vocab, axes, args,
                                       tag=f"[{target}/final]")
        path = out_dir / f"model_{target}.pt"
        save_checkpoint(path, model, pre, vocab, axes, target, args, n, features)
        print(f"  final loss {hist[-1]:.4f}  ->  saved {path}")
    if summary and not args.cv_only:
        with open(out_dir / "cv_metrics.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nCV metrics written to {out_dir / 'cv_metrics.json'}")


def predict_configs(df: pd.DataFrame, model_dir, targets, draws: int, seed: int, source: str,
                    opts: dict | None = None) -> list[dict]:
    """Best config per region and target. df has one row per observation (see 'predict' help).
    opts describes the target machine (max_threads, freq_range, freq_step) and the readout."""
    opts = opts or {}
    out_rows = []
    loaded = 0
    for target in targets:
        path = Path(model_dir) / f"model_{target}.pt"
        if not path.is_file():
            print(f"[warn] {path} not found, skipping {target}")
            continue
        loaded += 1
        model, ck = load_checkpoint(path)
        vocab, features, max_len = ck["vocab"], ck["features"], ck["max_len"]
        need = list(features) + (list(HEADS) if ck["use_config_context"] else [])
        missing = [c for c in need if c not in df.columns]
        if missing:
            sys.exit(f"{source} is missing columns required by the {target} model: {missing}")
        machine = inference_machine(opts, df, ck["axes"]["chunk_max"])
        readout = opts.get("readout") or ck["readout"]
        if loaded == 1:
            print(f"Target machine: up to {max(machine.threads)} threads, "
                  f"{min(machine.frequency) / 1e6:.2f}-{max(machine.frequency) / 1e6:.2f} GHz "
                  f"({len(machine.frequency)} steps); readout: {readout}")
        pre = Preproc.from_state(ck["preproc"])
        gen = torch.Generator().manual_seed(seed)
        for region, g in df.dropna(subset=need).groupby("region", sort=False):
            x = pre(g[features].to_numpy(dtype=np.float64))
            if ck["use_config_context"]:
                x = np.concatenate([x, context_features(g["threads"], g["sched"], g["chunk"],
                                                        g["frequency"], vocab, machine)], axis=1)
            x = torch.from_numpy(x)
            with torch.no_grad():
                if len(x) <= max_len:
                    batches = [x[None]]
                else:  # more observations than trained on: average over random subsets
                    batches = [x[torch.randperm(len(x), generator=gen)[:max_len]][None]
                               for _ in range(draws)]
                probs = [{h: torch.softmax(lg, -1) for h, lg in
                          model(b, torch.tensor([b.shape[1]])).items()} for b in batches]
            mean = {h: torch.stack([p[h] for p in probs]).mean(0)[:, None] for h in HEADS}
            cfg = decode_raw(mean, [machine], vocab, readout)[0][0]
            out_rows.append({"region": region, "target": target, "n_observations": len(x), **cfg})
    if not loaded:
        sys.exit(f"no model_<target>.pt for {', '.join(targets)} in {model_dir}; run 'train' first")
    return out_rows


def machine_opts(args) -> dict:
    return {"max_threads": args.max_threads, "freq_range": args.freq_range,
            "freq_step": args.freq_step, "readout": args.readout}


def cmd_predict(args):
    df = pd.read_csv(args.input)
    if "region" not in df.columns:
        df["region"] = "region"
    result = pd.DataFrame(predict_configs(df, args.model_dir, args.targets, args.draws, args.seed, args.input,
                                          machine_opts(args)))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} predictions to {args.output}")
    else:
        print(result.to_string(index=False))


# --------------------------------------------------------------------------- #
# Snapshot executions -> configuration file for ORBIT
# --------------------------------------------------------------------------- #
def import_best_settings():
    """best_settings.py (ORBIT/tools) turns executions into all_configs.csv rows; reuse it so the
    snapshot features are computed exactly like the training data."""
    here = Path(__file__).resolve().parent
    for d in (here, here.parent):
        if (d / "best_settings.py").is_file():
            sys.path.insert(0, str(d))
            import best_settings
            return best_settings
    sys.exit(f"best_settings.py not found next to or above {here}")


def find_runs(paths) -> list[Path]:
    """Each path is either one execution folder (has snapshot.conf) or a folder of executions."""
    runs = []
    for p in map(Path, paths):
        if (p / "snapshot.conf").is_file():
            runs.append(p)
        elif p.is_dir():
            runs += sorted(d for d in p.iterdir() if (d / "snapshot.conf").is_file())
    if not runs:
        sys.exit("no snapshot execution (folder containing snapshot.conf) found in: " + ", ".join(map(str, paths)))
    return runs


def load_snapshots(paths, stat: str, skip_first: int, time_source: str) -> pd.DataFrame:
    """One row per (region, settings) with the all_configs.csv columns, from snapshot executions."""
    bs = import_best_settings()
    runs = find_runs(paths)
    rows, recorded = [], set()
    for run in runs:
        recorded.update(item["name"] for item in bs.read_concatenated_json(run / "snapshot.conf"))
        rows += bs.process_run((str(run), stat, skip_first, time_source))
    if not rows:
        sys.exit("the snapshot executions contain no usable measurements "
                 "(each region needs <region>__cpu_energy.json and <region>__cpu_pmu.json)")
    df = pd.DataFrame(rows)
    missing = sorted(recorded - set(df["region"]))
    if missing:
        print(f"[warn] {len(missing)} region(s) in snapshot.conf have no energy measurements, skipped: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")

    # Merge executions with identical settings, like best_settings.py does for all_configs.csv.
    metrics = ["time_s", "energy_j", *bs.PMU_METRICS]
    df = df.groupby(["region", *HEADS], sort=False, as_index=False).agg(
        {"caller": "first", "function": "first", "calls": lambda s: int(s.sum() // len(s)),
         **{m: stat for m in metrics}})
    df["edp"] = df["time_s"] * df["energy_j"]
    df["power_w"] = (df["energy_j"] / df["time_s"]).where(df["time_s"] > 0)
    print(f"Read {len(runs)} snapshot execution(s): {df['region'].nunique()} regions, {len(df)} observations")
    return df


def build_config(obs: pd.DataFrame, preds: list[dict], meta: dict) -> dict:
    """best_per_region.json-style document: region -> {caller, function, calls, fastest, best_edp}."""
    by_region = {}
    for p in preds:
        by_region.setdefault(p["region"], {})[p["target"]] = {h: p[h] for h in HEADS}
    config = {}
    for region, g in obs.groupby("region", sort=False):
        if region not in by_region:
            continue
        calls = int(g["calls"].iloc[0])
        # ORBIT skips regions whose total_time_s is below its threshold; the best time seen in the
        # snapshot executions stands in for it (the time under the predicted settings is unknown).
        total_time_s = calls * float(g["time_s"].min())
        entry = {"caller": g["caller"].iloc[0], "function": g["function"].iloc[0], "calls": calls}
        for target in TARGETS:
            if target in by_region[region]:
                entry[target] = {**by_region[region][target], "total_time_s": total_time_s}
        config[region] = entry
    config["_prediction"] = meta
    return config


def cmd_generate(args):
    obs = load_snapshots(args.snapshots, args.stat, args.skip_first, args.time_source)
    preds = predict_configs(obs, args.model_dir, args.targets, args.draws, args.seed, "snapshot executions",
                            machine_opts(args))
    skipped = sorted(set(obs["region"]) - {p["region"] for p in preds})
    if skipped:
        print(f"[warn] no complete PMU metrics for {len(skipped)} region(s), left out: "
              f"{', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}")
    meta = {
        "generated_by": "train_gru.py generate",
        "snapshots": [str(Path(p).resolve()) for p in args.snapshots],
        "targets": sorted({p["target"] for p in preds}),
        "note": "threads/sched/chunk/frequency are model predictions; total_time_s is calls x the best "
                "per-call time measured in the snapshot executions, not a prediction",
    }
    config = build_config(obs, preds, meta)
    regions = {k: v for k, v in config.items() if not k.startswith("_")}
    if not regions:
        sys.exit("no region could be predicted, nothing written")
    with open(args.output, "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    def fmt(c):
        return f"{c['threads']}t {c['sched']},{c['chunk']} @{c['frequency'] / 1e6:.2f}GHz" if c else "-"

    print(f"\n{'region':<18}{'calls':>6}{'total_s':>9}   {'fastest':<30}{'best_edp'}")
    for name, e in regions.items():
        total_s = (e.get("fastest") or e["best_edp"])["total_time_s"]
        print(f"{name:<18}{e['calls']:>6}{total_s:>9.3f}   {fmt(e.get('fastest')):<30}{fmt(e.get('best_edp'))}")
    print(f"\nWrote {len(regions)} regions to {args.output}\n"
          f"Apply with: ORBIT_OPTIMIZED_CONF={args.output} ORBIT_OPTIMIZE_POLICY=<fastest|edp> "
          f"LD_PRELOAD=<liborbit.so> ./app\n"
          f"(ORBIT only tunes regions with total_time_s above ORBIT_TIME_THRESHOLD, default 100ms)")


def add_machine_args(p: argparse.ArgumentParser):
    p.add_argument("--max-threads", type=int, default=None,
                   help="threads of the target machine (default: nproc of this machine, as in scan.sh)")
    p.add_argument("--freq-range", type=int, nargs=2, metavar=("MIN_KHZ", "MAX_KHZ"), default=None,
                   help="hardware frequency range of the target machine in kHz (default: this machine's "
                        "cpufreq range, as in scan.sh)")
    p.add_argument("--freq-step", type=int, default=200000, help="frequency step in kHz (scan.sh: 200000)")
    p.add_argument("--readout", choices=READOUTS, default=None,
                   help="override the readout stored in the model (median, mean or argmax)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="cross-validate and train one model (GRU or MLP) per target")
    t.add_argument("--data-dir", required=True, help="folder holding per-application sub-folders")
    t.add_argument("--out-dir", default="models")
    t.add_argument("--model", choices=tuple(MODELS), default="gru",
                   help="gru = recurrent over the observations, mlp = feed-forward DNN on the mean of "
                        "per-observation encodings (use --max-len 1 for one observation per sample)")
    t.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    t.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                   help="comma-separated metric columns of all_configs.csv used as model input")
    t.add_argument("--no-config-context", action="store_true",
                   help="do not feed the config each observation was measured under")
    t.add_argument("--cv-mode", choices=("app", "region", "none"), default="app",
                   help="app = leave-one-application-out, region = 5-fold by region")
    t.add_argument("--cv-only", action="store_true", help="run cross-validation but do not save final models")
    t.add_argument("--eval-draws", type=int, default=32, help="random observation subsets per test region")
    t.add_argument("--max-len", type=int, default=8, help="max observations per sequence")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=None,
                   help="recurrent layers (gru, default 1) or dense layers (mlp, default 3)")
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--samples-per-region", type=int, default=64, help="sequences drawn per region per epoch")
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--weight-decay", type=float, default=1e-2)
    t.add_argument("--noise", type=float, default=0.05, help="gaussian input noise (std, standardised units)")
    t.add_argument("--freq-sigma", type=float, default=1.0,
                   help="label smoothing width (in bins) for the frequency head; 0 disables")
    t.add_argument("--thread-sigma", type=float, default=0.0, help="same for the threads head")
    t.add_argument("--chunk-sigma", type=float, default=0.0, help="same for the chunk head")
    t.add_argument("--freq-bins", type=int, default=21,
                   help="bins of the frequency axis (fraction of the machine's range); 21 = 5%% steps")
    t.add_argument("--thread-bins", type=int, default=17,
                   help="bins of the threads axis (fraction of the machine's cores); 17 = 1/16 steps")
    t.add_argument("--readout", choices=READOUTS, default="argmax",
                   help="how a bin distribution becomes a setting: argmax = most likely bin, median/mean = "
                        "continuous position that can fall between bins; the result is then snapped to a "
                        "valid setting. Used for cross-validation and stored in the model as default")
    t.add_argument("--soft-tau", type=float, default=0.3,
                   help="if > 0, also use soft per-head targets built from all_configs.csv (every config "
                        "weighted by exp(-(ratio_to_best-1)/tau)); 0 = JSON labels only")
    t.add_argument("--hard-weight", type=float, default=0.25,
                   help="weight of the best_per_region.json label vs. the soft targets (1 = JSON only)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="predict best configs from profiling observations")
    q.add_argument("--model-dir", required=True)
    q.add_argument("--input", required=True,
                   help="CSV with 'region', the model's metric columns and (if used) threads,sched,chunk,frequency")
    q.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    q.add_argument("--draws", type=int, default=32, help="random subsets averaged when a region has > max-len rows")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--output", help="write predictions to this CSV instead of stdout")
    add_machine_args(q)
    q.set_defaults(func=cmd_predict)

    g = sub.add_parser("generate", help="read snapshot executions, write a best_per_region.json-style "
                                        "configuration file for ORBIT")
    g.add_argument("--model-dir", required=True, help="folder with model_fastest.pt / model_best_edp.pt")
    g.add_argument("--snapshots", nargs="+", required=True, metavar="PATH",
                   help="snapshot execution folder(s) (containing snapshot.conf and the per-region "
                        "*__cpu_pmu.json / *__cpu_energy.json) or folder(s) of such executions")
    g.add_argument("--output", default="predicted_best_per_region.json")
    g.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    g.add_argument("--stat", choices=("mean", "median"), default="mean",
                   help="how repeated calls of a region are compressed (as in best_settings.py)")
    g.add_argument("--skip-first", type=int, default=0, help="ignore the first N calls of each region")
    g.add_argument("--time-source", choices=("energy", "pmu"), default="energy")
    g.add_argument("--draws", type=int, default=32, help="random subsets averaged when a region has > max-len rows")
    g.add_argument("--seed", type=int, default=0)
    add_machine_args(g)
    g.set_defaults(func=cmd_generate)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
