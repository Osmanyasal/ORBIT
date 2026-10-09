#!/usr/bin/env python3
"""Per-region heatmaps of execution time and K-EDP.

Reads all_configs.csv produced by best_settings.py. For every region two
heatmaps are written: time and K-EDP, i.e. EDP / 1000 in kJ*s (per-call mean,
or --total for the whole-region totals). Y axis is the frequency, X axis is
sched/threads/chunk. Both heatmaps mark the fastest (star) and lowest-K-EDP
(diamond) configuration; the legend lists per-call and total (per-call x number
of calls) time, energy and K-EDP.

Usage:
    plot_heatmaps.py best_settings_bt.C.x [-o out_dir] [--total] [--format png]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm
import numpy as np


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["region"]].append({
                "sched": r["sched"], "threads": int(r["threads"]), "chunk": int(r["chunk"]),
                "frequency": int(r["frequency"]), "calls": float(r["calls"]),
                "time_s": float(r["time_s"]), "energy_j": float(r["energy_j"]),
                "edp": float(r["edp"]),
            })
    return rows


def num(v):
    """3 significant digits, never in scientific notation."""
    return np.format_float_positional(float("%.3g" % v), trim="-")


def draw(ax_fig, grid, xlabels, ylabels, group_edges, title, cbar_label, marks, fmt, log=True):
    fig, ax = ax_fig
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                   cmap="RdYlGn_r", norm=LogNorm() if log else None)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("sched / threads / chunk")
    ax.set_ylabel("frequency [GHz]")
    ax.set_title(title, fontsize=10)
    for e in group_edges:
        ax.axvline(e - 0.5, color="white", lw=1.2)
    for (x, y, marker, color, label) in marks:
        ax.scatter([x], [y], marker=marker, s=300 if marker == "*" else 130, facecolor=color,
                   edgecolor="black" if color == "white" else "white", linewidths=1.5, label=label, zorder=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure,
              ncol=2, fontsize=7.5, frameon=True)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    if log:
        lo, hi = im.norm.vmin, im.norm.vmax
        ticks = np.geomspace(lo, hi, 7)
        cb.set_ticks(ticks)
        cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.set_ticklabels([num(t) for t in ticks])
    cb.set_label(cbar_label)
    fig.subplots_adjust(top=0.85)
    fig.savefig(fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="folder with all_configs.csv (best_settings.py output)")
    ap.add_argument("-o", "--out", default=None, help="output dir (default: <results>/heatmaps)")
    ap.add_argument("--total", action="store_true",
                    help="colour the heatmaps by totals over all calls instead of per-call values")
    ap.add_argument("--linear", action="store_true", help="linear colour scale (default: log)")
    ap.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = ap.parse_args()

    csv_path = os.path.join(args.results, "all_configs.csv")
    if not os.path.isfile(csv_path):
        sys.exit(f"{csv_path} not found; run best_settings.py first")
    out = args.out or os.path.join(args.results, "heatmaps")
    os.makedirs(out, exist_ok=True)

    for region, rs in sorted(load(csv_path).items()):
        for r in rs:
            # Totals = per-call mean x number of calls of the region.
            r["tt"] = r["time_s"] * r["calls"]
            r["te"] = r["energy_j"] * r["calls"]
            r["tp"] = r["tt"] * r["te"]
            r["p"] = r["time_s"] * r["energy_j"]

        cols = sorted({(r["sched"], r["threads"], r["chunk"]) for r in rs})
        freqs = sorted({r["frequency"] for r in rs})
        xi = {c: i for i, c in enumerate(cols)}
        yi = {f: i for i, f in enumerate(freqs)}
        xlabels = [f"{s[:3]}/{t}/{c}" for s, t, c in cols]
        ylabels = [f"{f / 1e6:.2f}" for f in freqs]
        edges = [i for i in range(1, len(cols)) if cols[i][:2] != cols[i - 1][:2]]

        # Selection is identical for per-call and total values (calls is constant per region).
        fast = min(rs, key=lambda r: r["time_s"])
        edp = min(rs, key=lambda r: r["p"])

        def pos(r):
            return xi[(r["sched"], r["threads"], r["chunk"])], yi[r["frequency"]]

        def desc(r):
            return (f"{r['sched']}/{r['threads']}t/chunk {r['chunk']}/"
                    f"{r['frequency'] / 1e6:.2f} GHz")

        def stats(r):
            return (f"per call: {r['time_s'] * 1e3:.2f} ms, {num(r['energy_j'])} J, "
                    f"K-EDP {num(r['p'] / 1e3)} kJ*s\n"
                    f"total ({r['calls']:g} calls): {r['tt'] * 1e3:.2f} ms, {num(r['te'])} J, "
                    f"K-EDP {num(r['tp'] / 1e3)} kJ*s")

        marks = [
            (*pos(fast), "*", "white", f"fastest: {desc(fast)}\n{stats(fast)}"),
            (*pos(edp), "D", "white", f"lowest K-EDP: {desc(edp)}\n{stats(edp)}"),
        ]

        # (grid key, file label, display name, scale to display unit, unit)
        metrics = (("tt", "time", "time", 1e3, "ms"), ("tp", "edp", "K-EDP", 1e-3, "kJ*s")) if args.total else \
                  (("time_s", "time", "time", 1e3, "ms"), ("p", "edp", "K-EDP", 1e-3, "kJ*s"))
        scope = "total" if args.total else "per call"
        for key, label, name, factor, unit in metrics:
            grid = np.full((len(freqs), len(cols)), np.nan)
            for r in rs:
                x, y = pos(r)
                grid[y, x] = r[key] * factor
            title = f"{region}: {'execution time' if label == 'time' else name} ({scope})"
            fig, ax = plt.subplots(figsize=(11, 6.5))
            path = os.path.join(out, f"{region}_{label}.{args.format}")
            draw((fig, ax), grid, xlabels, ylabels, edges, title,
                 f"{name} [{unit}]" + ("" if args.linear else " (log scale)"),
                 marks, path, log=not args.linear)
        print(f"{region}: fastest {desc(fast)}, lowest K-EDP {desc(edp)}")

    print(f"Wrote heatmaps to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
