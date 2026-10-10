#!/usr/bin/env python3
"""Find the best OpenMP settings per parallel region (fastest and lowest EDP).

Expected layout (one sub-folder per execution):
    <root>/<run>/snapshot.conf                     concatenated JSON objects, one per region and
                                                   set of settings (a repeated call with the same
                                                   settings is not recorded again)
    <root>/<run>/<region>__cpu_energy.json         list of readings, one per call of <region>
    <root>/<run>/<region>__cpu_pmu.json            list of readings, one per call of <region>

All calls of a region inside one execution are compressed into a single
representation (mean or median), giving one row per (region, settings).
For every region the fastest and the lowest-EDP settings are reported.

EDP = energy [J] * time [s], using the compressed energy and time.

Usage:
    best_settings.py bt.C.x [-o out_dir] [--stat mean|median] [--skip-first N]
                            [--time-source energy|pmu] [--top K]
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool

SETTING_KEYS = ("threads", "sched", "chunk", "frequency")


def read_concatenated_json(path):
    dec = json.JSONDecoder()
    text = open(path).read()
    pos, items = 0, []
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return items
        obj, pos = dec.raw_decode(text, pos)
        items.append(obj)


def load_readings(path):
    """Return a list (one per call) of (duration_ms, {measurement_name: value})."""
    with open(path) as f:
        data = json.load(f)
    calls = []
    for entry in data:
        readings = entry.get("readings", [])
        if not readings:
            calls.append(None)
            continue
        # Multiple readings per call (e.g. one per socket): sum the measurements,
        # average the duration.
        meas = defaultdict(float)
        for r in readings:
            for m in r.get("measurements", []):
                try:
                    meas[m["name"]] += float(m["value"])
                except (ValueError, KeyError):
                    pass
        dur = sum(r["duration"] for r in readings) / len(readings)
        calls.append((dur, dict(meas)))
    return calls


def process_run(args):
    run_dir, stat, skip_first, time_source = args
    conf_path = os.path.join(run_dir, "snapshot.conf")
    if not os.path.isfile(conf_path):
        return []
    conf = read_concatenated_json(conf_path)
    agg = stat_fn(stat)

    # Group call indices per region, preserving call order.
    per_region = defaultdict(list)
    for item in conf:
        per_region[item["name"]].append(item)

    rows = []
    for region, items in per_region.items():
        energy_path = os.path.join(run_dir, f"{region}__cpu_energy.json")
        pmu_path = os.path.join(run_dir, f"{region}__cpu_pmu.json")
        if not os.path.isfile(energy_path):
            continue
        energy = load_readings(energy_path)
        pmu = load_readings(pmu_path) if os.path.isfile(pmu_path) else []
        # The energy file holds one reading per call. snapshot.conf only has a new record when the
        # settings of the region change, so its record count says nothing about the number of calls.
        n = len(energy)
        settings = {k: items[0][k] for k in SETTING_KEYS}
        for it in items:
            if any(it[k] != settings[k] for k in SETTING_KEYS):
                print(f"warning: settings change inside {run_dir} region {region}",
                      file=sys.stderr)
                break

        times, joules, ipcs = [], [], []
        for i in range(skip_first, n):
            e = energy[i]
            if e is None:
                continue
            e_dur, e_meas = e
            e_j = sum(v for k, v in e_meas.items() if k.startswith("energy-"))
            t = e_dur
            p = pmu[i] if i < len(pmu) else None
            if time_source == "pmu" and p is not None:
                t = p[0]
            times.append(t / 1000.0)
            joules.append(e_j)
            if p is not None and "ipc" in p[1]:
                ipcs.append(p[1]["ipc"])
        if not times:
            continue
        t, e = agg(times), agg(joules)
        rows.append({
            "region": region,
            "caller": items[0].get("caller"),
            "function": items[0].get("function"),
            "run": os.path.basename(run_dir),
            **settings,
            "calls": len(times),
            "time_s": t,
            "energy_j": e,
            "edp": t * e,
            "power_w": e / t if t > 0 else float("nan"),
            "ipc": agg(ipcs) if ipcs else float("nan"),
        })
    return rows


def stat_fn(name):
    return statistics.median if name == "median" else statistics.mean


def fmt_settings(r):
    return (f"threads={r['threads']:<3} sched={r['sched']:<7} "
            f"chunk={r['chunk']:<5} freq={r['frequency'] / 1e6:.2f}GHz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder holding one sub-folder per execution")
    ap.add_argument("-o", "--out", default=None,
                    help="output dir for CSV/JSON (default: <root>/../best_settings_<root>)")
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="how to compress repeated calls of a region (default: mean)")
    ap.add_argument("--skip-first", type=int, default=0, metavar="N",
                    help="ignore the first N calls of each region (warm-up)")
    ap.add_argument("--time-source", choices=("energy", "pmu"), default="energy",
                    help="which measurement's duration to use as time (default: energy)")
    ap.add_argument("--top", type=int, default=3, help="show top-K settings per region")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    runs = sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "snapshot.conf")))
    if not runs:
        sys.exit(f"no executions found in {root}")
    out = args.out or os.path.join(os.path.dirname(root),
                                   f"best_settings_{os.path.basename(root)}")
    os.makedirs(out, exist_ok=True)

    print(f"Processing {len(runs)} executions with {args.jobs} workers...", file=sys.stderr)
    jobs = [(r, args.stat, args.skip_first, args.time_source) for r in runs]
    with Pool(args.jobs) as pool:
        rows = [row for res in pool.imap_unordered(process_run, jobs) for row in res]

    # Merge executions that share identical settings (repeated runs) as well.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["region"],) + tuple(r[k] for k in SETTING_KEYS)].append(r)
    agg = stat_fn(args.stat)
    merged = []
    for rs in grouped.values():
        m = dict(rs[0])
        for k in ("time_s", "energy_j", "ipc"):
            m[k] = agg([x[k] for x in rs])
        m["edp"] = m["time_s"] * m["energy_j"]
        m["power_w"] = m["energy_j"] / m["time_s"] if m["time_s"] > 0 else float("nan")
        m["calls"] = sum(x["calls"] for x in rs) // len(rs)
        m["run"] = ",".join(x["run"] for x in rs)
        merged.append(m)

    by_region = defaultdict(list)
    for r in merged:
        by_region[r["region"]].append(r)

    fields = ["region", "caller", "function", *SETTING_KEYS, "calls",
              "time_s", "energy_j", "edp", "power_w", "ipc", "run"]
    with open(os.path.join(out, "all_configs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(merged, key=lambda r: (r["region"], r["time_s"])):
            w.writerow({k: r[k] for k in fields})

    summary = {}
    for region in sorted(by_region):
        rs = by_region[region]
        fastest = sorted(rs, key=lambda r: r["time_s"])
        edp = sorted(rs, key=lambda r: r["edp"])
        calls = rs[0]["calls"]

        def pick(r):
            d = {k: r[k] for k in (*SETTING_KEYS, "time_s", "energy_j", "edp")}
            # Totals over all calls of the region: per-call mean x number of calls.
            d["total_time_s"] = calls * r["time_s"]
            d["total_energy_j"] = calls * r["energy_j"]
            d["total_edp"] = d["total_time_s"] * d["total_energy_j"]
            return d

        summary[region] = {
            "caller": rs[0]["caller"], "function": rs[0]["function"],
            "calls": calls, "configs_evaluated": len(rs),
            "fastest": pick(fastest[0]),
            "best_edp": pick(edp[0]),
        }
        print(f"\n=== {region}  (caller {rs[0]['caller']}, ~{rs[0]['calls']} calls/run, "
              f"{len(rs)} configs) ===")
        for title, ranked in (("fastest", fastest), ("lowest EDP", edp)):
            print(f"  {title}:")
            for i, r in enumerate(ranked[:args.top]):
                print(f"    {i + 1}. {fmt_settings(r)}  time={r['time_s'] * 1e3:8.3f} ms  "
                      f"energy={r['energy_j']:7.3f} J  EDP={r['edp']:.3e} J*s")

    # Program-level estimate: per-call values scaled by calls per region.
    def totals(pick):
        t = sum(s["calls"] * pick(s)["time_s"] for s in summary.values())
        e = sum(s["calls"] * pick(s)["energy_j"] for s in summary.values())
        return t, e

    print("\n=== Estimated totals over all regions (calls x per-call value) ===")
    for label, pick in (("per-region fastest", lambda s: s["fastest"]),
                        ("per-region best EDP", lambda s: s["best_edp"])):
        t, e = totals(pick)
        print(f"  {label:<20} time={t:9.3f} s  energy={e:9.2f} J  EDP={t * e:.3e} J*s")

    # Best single configuration applied to every region, for comparison.
    cfg_tot = defaultdict(lambda: [0.0, 0.0, 0])
    for region, rs in by_region.items():
        for r in rs:
            c = cfg_tot[tuple(r[k] for k in SETTING_KEYS)]
            c[0] += r["calls"] * r["time_s"]
            c[1] += r["calls"] * r["energy_j"]
            c[2] += 1
    full = [(k, v) for k, v in cfg_tot.items() if v[2] == len(by_region)]
    if full:
        kt = min(full, key=lambda kv: kv[1][0])
        ke = min(full, key=lambda kv: kv[1][0] * kv[1][1])
        for label, (k, v) in (("best uniform fastest", kt), ("best uniform EDP", ke)):
            d = dict(zip(SETTING_KEYS, k))
            print(f"  {label:<20} time={v[0]:9.3f} s  energy={v[1]:9.2f} J  "
                  f"EDP={v[0] * v[1]:.3e} J*s  [{fmt_settings(d)}]")

    summary["_total"] = {}
    for key, label in (("fastest", "per-region fastest"), ("best_edp", "per-region best EDP")):
        t = sum(s[key]["total_time_s"] for s in summary.values() if key in s)
        e = sum(s[key]["total_energy_j"] for s in summary.values() if key in s)
        summary["_total"][key] = {"total_time_s": t, "total_energy_j": e, "total_edp": t * e}

    with open(os.path.join(out, "best_per_region.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}/best_per_region.json and {out}/all_configs.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
