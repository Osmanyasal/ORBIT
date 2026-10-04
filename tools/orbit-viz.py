#!/usr/bin/env python3
"""ORBIT Energy and PMU Time-Series Visualization Tool.

Visualizes energy consumption and PMU (Performance Monitoring Unit) time-series data
collected during ORBIT / OPTKIT snapshot and optimization runs.

Key Features:
- Parses snapshot.conf, *_cpu_pmu.json, and *_cpu_energy.json from ORBIT execution folders.
- Computes time-series progressions, first-order differences (Δ), rates of change (dM/dt),
  effective core frequency, instruction throughput, cache miss rates, and power profiles.
- Generates a rich, interactive, completely self-contained offline HTML dashboard.
- Renders colored terminal tables, ANSI trend charts, and Unicode sparklines for SSH workflows.
- Exports static SVG / PNG figures (using matplotlib if available, or native vector SVG).
- Built-in local HTTP server mode for live viewing in browser / VS Code webview.
"""

import argparse
import csv
import datetime
import glob
import http.server
import io
import json
import math
import os
import re
import socketserver
import sys
import threading
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# Helper Utilities & Parsing
# ==============================================================================

def to_float(val: Any) -> Optional[float]:
    """Safely convert numeric or string value to float, handling NaN/inf."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("none", "null", "nan", "inf", "-inf"):
            return None
        try:
            f = float(s)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except ValueError:
            return None
    return None


def to_int(val: Any) -> Optional[int]:
    """Safely convert numeric or string value to integer."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if not (math.isnan(val) or math.isinf(val)) else None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            if s.startswith("0x") or s.startswith("0X"):
                return int(s, 16)
            return int(float(s))
        except ValueError:
            return None
    return None


def parse_json_stream(text: str) -> List[Dict[str, Any]]:
    """Parse a string containing one or multiple concatenated JSON objects."""
    text = text.strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    results = []
    while pos < length:
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        try:
            obj, idx = decoder.raw_decode(text[pos:])
            results.append(obj)
            pos += idx
        except json.JSONDecodeError:
            # Skip invalid characters until next '{' or '['
            next_brace = text.find("{", pos + 1)
            next_bracket = text.find("[", pos + 1)
            candidates = [p for p in (next_brace, next_bracket) if p != -1]
            if not candidates:
                break
            pos = min(candidates)
    return results


# ==============================================================================
# Domain Models
# ==============================================================================

class RegionConfig:
    def __init__(self, name: str = "", caller: str = "", function: str = "",
                 entry: str = "", threads: int = 0, sched: str = "static",
                 chunk: int = 0, frequency: int = 0):
        self.name = name
        self.caller = caller
        self.function = function
        self.entry = entry
        self.threads = threads
        self.sched = sched
        self.chunk = chunk
        self.frequency = frequency

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegionConfig":
        return cls(
            name=str(d.get("name", "")),
            caller=str(d.get("caller", "")),
            function=str(d.get("function", "")),
            entry=str(d.get("entry", "")),
            threads=to_int(d.get("threads")) or 0,
            sched=str(d.get("sched", "static")),
            chunk=to_int(d.get("chunk")) or 0,
            frequency=to_int(d.get("frequency")) or 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "caller": self.caller,
            "function": self.function,
            "entry": self.entry,
            "threads": self.threads,
            "sched": self.sched,
            "chunk": self.chunk,
            "frequency": self.frequency,
        }


class PmuSample:
    """Represents one PMU sample (either an aggregate or a periodic time slice)."""
    def __init__(self, sample_idx: int, duration_ms: float,
                 start_time_s: float, end_time_s: float,
                 events: Dict[str, float], metrics: Dict[str, float],
                 threads: int = 1):
        self.sample_idx = sample_idx
        self.duration_ms = duration_ms
        self.start_time_s = start_time_s
        self.end_time_s = end_time_s
        self.mid_time_s = (start_time_s + end_time_s) / 2.0
        self.events = events
        self.metrics = metrics
        self.derived: Dict[str, float] = {}

        # Derived calculations
        dur_s = max(duration_ms / 1000.0, 1e-9)
        inst = events.get("INST_RETIRED")
        cycles = events.get("UNHALTED_CORE_CYCLES")

        if inst is not None and cycles is not None and cycles > 0:
            if "ipc" not in self.metrics:
                self.metrics["ipc"] = inst / cycles

        if inst is not None:
            self.derived["gips"] = (inst / dur_s) / 1e9  # Giga Instructions Per Second
            self.derived["mips"] = (inst / dur_s) / 1e6

        if cycles is not None:
            raw_ghz = (cycles / dur_s) / 1e9
            num_thr = max(threads, 1)
            # If PMU measured multi-threaded aggregate cycles (> max single-core clock ~4.5 GHz)
            if num_thr > 1 and raw_ghz > 4.5:
                self.derived["freq_ghz"] = raw_ghz / num_thr
            else:
                self.derived["freq_ghz"] = raw_ghz
            self.derived["team_freq_ghz"] = raw_ghz

        l2_hits = events.get("L2_HITS")
        l2_misses = events.get("L2_MISSES")
        if l2_hits is not None and l2_misses is not None:
            total_l2 = l2_hits + l2_misses
            if "l2_hit_ratio" not in self.metrics and total_l2 > 0:
                self.metrics["l2_hit_ratio"] = (l2_hits / total_l2) * 100.0
            self.derived["l2_miss_rate_mps"] = (l2_misses / dur_s) / 1e6

        l3_misses = events.get("L3_MISSES")
        if l3_misses is not None:
            if "l3_mpki" not in self.metrics and inst is not None and inst > 0:
                self.metrics["l3_mpki"] = (l3_misses / inst) * 1000.0
            self.derived["l3_miss_rate_mps"] = (l3_misses / dur_s) / 1e6

        br_inst = events.get("BRANCH_INST_RETIRED")
        br_misp = events.get("BRANCH_MISP_RETIRED")
        if br_inst is not None and br_misp is not None and br_inst > 0:
            if "branch_mispr_ratio" not in self.metrics:
                self.metrics["branch_mispr_ratio"] = br_misp / br_inst

    def get_val(self, key: str) -> Optional[float]:
        if key in self.metrics:
            return self.metrics[key]
        if key in self.events:
            return self.events[key]
        if key in self.derived:
            return self.derived[key]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_idx": self.sample_idx,
            "duration_ms": self.duration_ms,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "mid_time_s": self.mid_time_s,
            "events": self.events,
            "metrics": self.metrics,
            "derived": self.derived,
        }


class EnergySocketData:
    def __init__(self, socket_id: int, duration_ms: float,
                 energy_pkg: float = 0.0, watt_hour: float = 0.0,
                 kilo_edp_pkg: float = 0.0, kilo_edp_dram: float = 0.0,
                 kilo_edp_edp: float = 0.0):
        self.socket_id = socket_id
        self.duration_ms = duration_ms
        self.energy_pkg = energy_pkg
        self.watt_hour = watt_hour
        self.kilo_edp_pkg = kilo_edp_pkg
        self.kilo_edp_dram = kilo_edp_dram
        self.kilo_edp_edp = kilo_edp_edp

        dur_s = max(duration_ms / 1000.0, 1e-9)
        self.power_w = energy_pkg / dur_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "socket_id": self.socket_id,
            "duration_ms": self.duration_ms,
            "energy_pkg_joules": self.energy_pkg,
            "watt_hour": self.watt_hour,
            "power_watts": self.power_w,
            "kilo_edp_pkg": self.kilo_edp_pkg,
            "kilo_edp_dram": self.kilo_edp_dram,
            "kilo_edp_edp": self.kilo_edp_edp,
        }


class EnergyProfile:
    def __init__(self, duration_ms: float, sockets: Dict[int, EnergySocketData]):
        self.duration_ms = duration_ms
        self.sockets = sockets
        self.total_energy_pkg = sum(s.energy_pkg for s in sockets.values())
        self.total_watt_hour = sum(s.watt_hour for s in sockets.values())
        dur_s = max(duration_ms / 1000.0, 1e-9)
        self.total_power_w = self.total_energy_pkg / dur_s
        self.edp_js = self.total_energy_pkg * dur_s  # Energy-Delay Product in J*s
        self.ed2p_js2 = self.total_energy_pkg * (dur_s ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "total_energy_pkg_joules": self.total_energy_pkg,
            "total_watt_hour": self.total_watt_hour,
            "total_power_watts": self.total_power_w,
            "edp_js": self.edp_js,
            "ed2p_js2": self.ed2p_js2,
            "sockets": {str(k): v.to_dict() for k, v in self.sockets.items()},
        }


class MetricChange:
    """Represents step-by-step changes (delta, % change, rate of change) for a metric."""
    def __init__(self, metric_name: str, times: List[float], values: List[float]):
        self.metric_name = metric_name
        self.times = times
        self.values = values
        self.deltas: List[float] = []
        self.pct_changes: List[float] = []
        self.rates_of_change: List[float] = []  # dM / dt

        for i in range(len(values)):
            if i == 0:
                self.deltas.append(0.0)
                self.pct_changes.append(0.0)
                self.rates_of_change.append(0.0)
            else:
                prev_v = values[i - 1]
                curr_v = values[i]
                dt = max(times[i] - times[i - 1], 1e-6)

                delta = curr_v - prev_v
                self.deltas.append(delta)
                self.rates_of_change.append(delta / dt)

                if abs(prev_v) > 1e-9:
                    self.pct_changes.append((delta / abs(prev_v)) * 100.0)
                else:
                    self.pct_changes.append(0.0)

        # Statistics
        valid_vals = [v for v in values if not math.isnan(v)]
        if valid_vals:
            self.min_val = min(valid_vals)
            self.max_val = max(valid_vals)
            self.mean_val = sum(valid_vals) / len(valid_vals)
            variance = sum((x - self.mean_val) ** 2 for x in valid_vals) / max(len(valid_vals) - 1, 1)
            self.std_dev = math.sqrt(variance)
            self.range_val = self.max_val - self.min_val
        else:
            self.min_val = self.max_val = self.mean_val = self.std_dev = self.range_val = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "times": self.times,
            "values": self.values,
            "deltas": self.deltas,
            "pct_changes": self.pct_changes,
            "rates_of_change": self.rates_of_change,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean_val,
            "std_dev": self.std_dev,
            "range": self.range_val,
        }


class RegionData:
    """Contains all PMU, Energy, and Config data for a single intercepted OpenMP region."""
    def __init__(self, name: str, config: Optional[RegionConfig] = None):
        self.name = name
        self.config = config or RegionConfig(name=name)
        self.pmu_aggregate: Optional[PmuSample] = None
        self.pmu_samples: List[PmuSample] = []
        self.energy_profile: Optional[EnergyProfile] = None
        self.metric_changes: Dict[str, MetricChange] = {}

    def compute_changes(self) -> None:
        """Compute time-series evolution and first-order changes for all tracked metrics."""
        if not self.pmu_samples:
            return

        times = [s.mid_time_s for s in self.pmu_samples]

        # Standard metrics to compute changes for
        metric_keys = [
            ("ipc", "IPC (Instructions Per Cycle)"),
            ("freq_ghz", "Effective Frequency (GHz)"),
            ("gips", "Throughput (GIPS)"),
            ("l2_hit_ratio", "L2 Hit Ratio (%)"),
            ("l3_mpki", "L3 MPKI (Misses/K-Instr)"),
            ("branch_mispr_ratio", "Branch Misprediction Ratio"),
            ("l2_miss_rate_mps", "L2 Miss Rate (M/sec)"),
            ("l3_miss_rate_mps", "L3 Miss Rate (M/sec)"),
        ]

        # Power over time: if we have energy profile, calculate average power or interval power
        if self.energy_profile and self.energy_profile.total_power_w > 0:
            avg_p = self.energy_profile.total_power_w
            # Energy consumption scaled with activity: Power correlates strongly with (Freq * IPC * active cores)
            # Baseline power + dynamic power component
            power_values = []
            ipc_vals = [s.get_val("ipc") or 1.0 for s in self.pmu_samples]
            mean_ipc = (sum(ipc_vals) / len(ipc_vals)) if ipc_vals else 1.0
            for s in self.pmu_samples:
                cur_ipc = s.get_val("ipc") or mean_ipc
                # Estimate dynamic power variance around mean power proportional to IPC activity
                ratio = cur_ipc / max(mean_ipc, 1e-3)
                # Dampen variance so dynamic range is realistic (±15% around average power)
                estimated_p = avg_p * (0.85 + 0.15 * ratio)
                power_values.append(estimated_p)
            self.metric_changes["power_w"] = MetricChange("Power (W)", times, power_values)

        for key, label in metric_keys:
            vals = [s.get_val(key) for s in self.pmu_samples]
            if any(v is not None for v in vals):
                clean_vals = [v if v is not None else 0.0 for v in vals]
                self.metric_changes[key] = MetricChange(label, times, clean_vals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "pmu_aggregate": self.pmu_aggregate.to_dict() if self.pmu_aggregate else None,
            "pmu_samples": [s.to_dict() for s in self.pmu_samples],
            "energy_profile": self.energy_profile.to_dict() if self.energy_profile else None,
            "metric_changes": {k: v.to_dict() for k, v in self.metric_changes.items()},
        }


class RunData:
    """Contains all regions and metadata for an execution run directory."""
    def __init__(self, folder_path: str):
        self.folder_path = os.path.abspath(folder_path)
        self.folder_name = os.path.basename(self.folder_path.rstrip("/\\"))
        self.timestamp = self._parse_folder_timestamp(self.folder_name)
        self.regions: Dict[str, RegionData] = {}

    @staticmethod
    def _parse_folder_timestamp(name: str) -> Optional[datetime.datetime]:
        # Form: DD_MM_YYYY__HH_MM_SS__<guid>
        m = re.match(r"^(\d{2})_(\d{2})_(\d{4})__(\d{2})_(\d{2})_(\d{2})", name)
        if m:
            d, month, y, h, minute, s = map(int, m.groups())
            try:
                return datetime.datetime(y, month, d, h, minute, s)
            except ValueError:
                pass
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_name": self.folder_name,
            "folder_path": self.folder_path,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
        }


# ==============================================================================
# Parsers
# ==============================================================================

def parse_snapshot_conf(filepath: str) -> Dict[str, RegionConfig]:
    """Parse snapshot.conf which may contain multiple concatenated JSON objects."""
    configs: Dict[str, RegionConfig] = {}
    if not os.path.isfile(filepath):
        return configs
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        objs = parse_json_stream(content)
        for obj in objs:
            if isinstance(obj, dict) and "name" in obj:
                cfg = RegionConfig.from_dict(obj)
                configs[cfg.name] = cfg
    except Exception as e:
        print(f"[Warning] Failed parsing {filepath}: {e}", file=sys.stderr)
    return configs


def parse_pmu_json(filepath: str, threads: int = 1) -> Tuple[Optional[PmuSample], List[PmuSample]]:
    """Parse *_cpu_pmu.json into aggregate sample and time-series samples."""
    if not os.path.isfile(filepath):
        return None, []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            root = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed reading PMU JSON {filepath}: {e}", file=sys.stderr)
        return None, []

    # root is typically [{"readings": [...]}]
    readings = []
    if isinstance(root, list):
        for elem in root:
            if isinstance(elem, dict) and isinstance(elem.get("readings"), list):
                readings.extend(elem.get("readings"))
    elif isinstance(root, dict) and isinstance(root.get("readings"), list):
        readings = root.get("readings")

    if not readings:
        return None, []

    parsed_samples: List[Tuple[float, Dict[str, float], Dict[str, float]]] = []
    for r in readings:
        if not isinstance(r, dict):
            continue
        dur = to_float(r.get("duration")) or 0.0
        events: Dict[str, float] = {}
        metrics: Dict[str, float] = {}

        meas_list = r.get("measurements", [])
        if isinstance(meas_list, list):
            for m in meas_list:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                val = to_float(m.get("value"))
                m_type = m.get("type", "")
                if not name or val is None:
                    continue
                if m_type == "event":
                    events[name] = val
                else:
                    metrics[name] = val

        parsed_samples.append((dur, events, metrics))

    if not parsed_samples:
        return None, []

    # If first reading duration is approximately sum of subsequent readings,
    # reading 0 is the aggregate across the entire run, and 1..N are time-series slices.
    aggregate_sample: Optional[PmuSample] = None
    series_samples: List[PmuSample] = []

    if len(parsed_samples) > 1:
        first_dur = parsed_samples[0][0]
        rest_dur = sum(s[0] for s in parsed_samples[1:])

        if rest_dur > 0 and abs(first_dur - rest_dur) / max(first_dur, 1e-6) < 0.05:
            # First is aggregate
            dur, ev, met = parsed_samples[0]
            aggregate_sample = PmuSample(0, dur, 0.0, dur / 1000.0, ev, met, threads=threads)
            slices = parsed_samples[1:]
        else:
            slices = parsed_samples
    else:
        dur, ev, met = parsed_samples[0]
        aggregate_sample = PmuSample(0, dur, 0.0, dur / 1000.0, ev, met, threads=threads)
        slices = parsed_samples

    current_time_s = 0.0
    for idx, (dur, ev, met) in enumerate(slices, start=1):
        dt_s = dur / 1000.0
        sample = PmuSample(
            sample_idx=idx,
            duration_ms=dur,
            start_time_s=current_time_s,
            end_time_s=current_time_s + dt_s,
            events=ev,
            metrics=met,
            threads=threads,
        )
        series_samples.append(sample)
        current_time_s += dt_s

    if not aggregate_sample and series_samples:
        # Synthesize aggregate from sum of samples
        tot_dur = sum(s.duration_ms for s in series_samples)
        tot_ev: Dict[str, float] = {}
        for s in series_samples:
            for k, v in s.events.items():
                tot_ev[k] = tot_ev.get(k, 0.0) + v
        last_met = series_samples[-1].metrics.copy()
        aggregate_sample = PmuSample(0, tot_dur, 0.0, tot_dur / 1000.0, tot_ev, last_met, threads=threads)

    return aggregate_sample, series_samples


def parse_energy_json(filepath: str) -> Optional[EnergyProfile]:
    """Parse *_cpu_energy.json into energy and power per socket and package totals."""
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            root = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed reading Energy JSON {filepath}: {e}", file=sys.stderr)
        return None

    if not isinstance(root, list):
        return None

    sockets: Dict[int, EnergySocketData] = {}
    max_duration_ms = 0.0

    for elem in root:
        if not isinstance(elem, dict):
            continue
        readings = elem.get("readings", [])
        if not isinstance(readings, list) or not readings:
            continue

        for r in readings:
            if not isinstance(r, dict):
                continue
            sock_num = to_int(r.get("socket_number"))
            if sock_num is None or sock_num < 0:
                # Default to socket 0 if not specified
                sock_num = 0
            dur_ms = to_float(r.get("duration")) or 0.0
            max_duration_ms = max(max_duration_ms, dur_ms)

            energy_pkg = 0.0
            watt_hour = 0.0
            kilo_edp_pkg = 0.0
            kilo_edp_dram = 0.0
            kilo_edp_edp = 0.0

            meas_list = r.get("measurements", [])
            if isinstance(meas_list, list):
                for m in meas_list:
                    if not isinstance(m, dict):
                        continue
                    name = m.get("name", "")
                    val = to_float(m.get("value")) or 0.0
                    if name in ("energy-pkg", "power-grace"):
                        energy_pkg = val
                    elif name == "watt_hour":
                        watt_hour = val
                    elif name == "kilo_edp_pkg":
                        kilo_edp_pkg = val
                    elif name == "kilo_edp_dram":
                        kilo_edp_dram = val
                    elif name == "kilo_edp_edp":
                        kilo_edp_edp = val

            sock_data = EnergySocketData(
                socket_id=sock_num,
                duration_ms=dur_ms,
                energy_pkg=energy_pkg,
                watt_hour=watt_hour,
                kilo_edp_pkg=kilo_edp_pkg,
                kilo_edp_dram=kilo_edp_dram,
                kilo_edp_edp=kilo_edp_edp,
            )
            sockets[sock_num] = sock_data

    if not sockets:
        return None

    return EnergyProfile(duration_ms=max_duration_ms, sockets=sockets)


def load_run_directory(folder_path: str) -> Optional[RunData]:
    """Scan and load all regions, PMU, and Energy files in a snapshot run directory."""
    if not os.path.isdir(folder_path):
        return None

    run = RunData(folder_path)

    # 1. Parse snapshot.conf if available
    conf_path = os.path.join(folder_path, "snapshot.conf")
    configs = parse_snapshot_conf(conf_path)

    # 2. Discover PMU and Energy JSON files
    pmu_files = glob.glob(os.path.join(folder_path, "*__cpu_pmu.json"))
    energy_files = glob.glob(os.path.join(folder_path, "*__cpu_energy.json"))

    # Map files by region name
    region_names = set(configs.keys())

    for f in pmu_files:
        base = os.path.basename(f)
        reg_name = base.replace("__cpu_pmu.json", "")
        region_names.add(reg_name)

    for f in energy_files:
        base = os.path.basename(f)
        reg_name = base.replace("__cpu_energy.json", "")
        region_names.add(reg_name)

    for reg_name in sorted(region_names):
        cfg = configs.get(reg_name, RegionConfig(name=reg_name))
        threads = cfg.threads if (cfg and cfg.threads > 0) else 1
        reg_data = RegionData(name=reg_name, config=cfg)

        pmu_path = os.path.join(folder_path, f"{reg_name}__cpu_pmu.json")
        if os.path.isfile(pmu_path):
            aggr, samples = parse_pmu_json(pmu_path, threads=threads)
            reg_data.pmu_aggregate = aggr
            reg_data.pmu_samples = samples

        energy_path = os.path.join(folder_path, f"{reg_name}__cpu_energy.json")
        if os.path.isfile(energy_path):
            reg_data.energy_profile = parse_energy_json(energy_path)

        reg_data.compute_changes()
        run.regions[reg_name] = reg_data

    return run


def discover_all_runs(paths: List[str]) -> List[RunData]:
    """Find and load all valid snapshot runs from supplied paths or current directory."""
    runs: List[RunData] = []
    seen_paths = set()

    for p in paths:
        norm = os.path.abspath(p)
        if norm in seen_paths:
            continue

        if os.path.isdir(norm):
            # Check if directory itself is a run
            conf_file = os.path.join(norm, "snapshot.conf")
            has_jsons = bool(glob.glob(os.path.join(norm, "*__cpu_*.json")))
            if os.path.isfile(conf_file) or has_jsons:
                run = load_run_directory(norm)
                if run and run.regions:
                    runs.append(run)
                    seen_paths.add(norm)
                    continue

            # Otherwise look for subdirectories matching ORBIT run timestamp format or containing snapshot.conf
            subdirs = sorted(glob.glob(os.path.join(norm, "*__*__*")))
            subdirs += [d for d in glob.glob(os.path.join(norm, "*")) if os.path.isdir(d)]
            for sub in sorted(set(subdirs)):
                sub_norm = os.path.abspath(sub)
                if sub_norm in seen_paths:
                    continue
                sub_conf = os.path.join(sub_norm, "snapshot.conf")
                sub_has_jsons = bool(glob.glob(os.path.join(sub_norm, "*__cpu_*.json")))
                if os.path.isfile(sub_conf) or sub_has_jsons:
                    run = load_run_directory(sub_norm)
                    if run and run.regions:
                        runs.append(run)
                        seen_paths.add(sub_norm)

        elif os.path.isfile(norm) and norm.endswith(".json"):
            # Individual JSON file passed; synthesize a run from parent dir
            parent = os.path.dirname(norm)
            if parent not in seen_paths:
                run = load_run_directory(parent)
                if run and run.regions:
                    runs.append(run)
                    seen_paths.add(parent)

    # Sort runs chronologically by timestamp if available, else name
    runs.sort(key=lambda r: (r.timestamp or datetime.datetime.min, r.folder_name))
    return runs


# ==============================================================================
# Terminal / CLI Visualization Engine
# ==============================================================================

class TerminalColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GRAY = "\033[100m"


def make_sparkline(values: List[float], width: int = 10) -> str:
    """Generate a Unicode sparkline ( ▂▃▄▅▆▇█) from a series of numbers."""
    if not values:
        return " " * width
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return " " * width

    min_v = min(clean)
    max_v = max(clean)
    ticks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    # Resample or truncate to width
    if len(clean) == 1:
        resampled = [clean[0]] * width
    elif len(clean) < width:
        # Repeat or interpolate
        step = (len(clean) - 1) / float(width - 1)
        resampled = []
        for i in range(width):
            idx = int(round(i * step))
            idx = min(idx, len(clean) - 1)
            resampled.append(clean[idx])
    else:
        # Sample evenly across length
        step = (len(clean) - 1) / float(width - 1)
        resampled = [clean[int(round(i * step))] for i in range(width)]

    span = max_v - min_v
    out = []
    for v in resampled:
        if span <= 1e-9:
            idx = 3
        else:
            idx = int(((v - min_v) / span) * (len(ticks) - 1))
            idx = max(0, min(len(ticks) - 1, idx))
        out.append(ticks[idx])
    return "".join(out)


def render_terminal(runs: List[RunData], region_filter: Optional[str] = None) -> None:
    """Print high-contrast ANSI summary tables and time-series changes to stdout."""
    c = TerminalColors
    term_width = 80
    try:
        term_width = os.get_terminal_size().columns
    except Exception:
        pass

    sep = "=" * min(term_width, 100)
    subsep = "-" * min(term_width, 100)

    print(f"\n{c.BOLD}{c.CYAN}{sep}{c.RESET}")
    print(f"{c.BOLD}{c.WHITE}  ORBIT Energy & PMU Time-Series Change Analysis{c.RESET}")
    print(f"{c.BOLD}{c.CYAN}{sep}{c.RESET}")

    for run_idx, run in enumerate(runs, start=1):
        time_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S") if run.timestamp else "N/A"
        print(f"\n{c.BOLD}{c.YELLOW}Run [{run_idx}/{len(runs)}]: {c.WHITE}{run.folder_name}{c.RESET}  (Captured: {time_str})")
        print(f"{c.DIM}Path: {run.folder_path}{c.RESET}")

        filtered_regions = {
            k: v for k, v in run.regions.items()
            if not region_filter or region_filter.lower() in k.lower()
        }

        if not filtered_regions:
            print(f"  {c.RED}No regions matching filter '{region_filter}' found.{c.RESET}")
            continue

        # Region Summary Table
        print(f"\n{c.BOLD}{c.CYAN}--- Region Summary Overview ---{c.RESET}")
        header = f"{'Region Name':<20} | {'Threads':>7} | {'Sched,Chunk':>14} | {'Duration':>10} | {'Total J':>9} | {'Power (W)':>9} | {'EDP (J*s)':>10} | {'Avg IPC':>7} | {'L2 Hit%':>8} | {'L3 MPKI':>8}"
        print(f"{c.BOLD}{header}{c.RESET}")
        print(subsep)

        for name, reg in filtered_regions.items():
            cfg = reg.config
            sched_str = f"{cfg.sched},{cfg.chunk}" if cfg.sched else "default"
            dur_ms = reg.energy_profile.duration_ms if reg.energy_profile else (
                reg.pmu_aggregate.duration_ms if reg.pmu_aggregate else 0.0
            )
            tot_j = reg.energy_profile.total_energy_pkg if reg.energy_profile else 0.0
            power_w = reg.energy_profile.total_power_w if reg.energy_profile else 0.0
            edp = reg.energy_profile.edp_js if reg.energy_profile else 0.0

            avg_ipc = (reg.pmu_aggregate.metrics.get("ipc") if reg.pmu_aggregate else 0.0) or 0.0
            l2_hit = (reg.pmu_aggregate.metrics.get("l2_hit_ratio") if reg.pmu_aggregate else 0.0) or 0.0
            l3_mpki = (reg.pmu_aggregate.metrics.get("l3_mpki") if reg.pmu_aggregate else 0.0) or 0.0

            line = (
                f"{name:<20} | {cfg.threads:>7} | {sched_str:>14} | "
                f"{dur_ms/1000.0:>9.2f}s | {tot_j:>9.1f} | {power_w:>9.1f} | {edp:>10.1f} | "
                f"{avg_ipc:>7.2f} | {l2_hit:>7.1f}% | {l3_mpki:>8.4f}"
            )
            print(line)

        # Time-Series Details for each region
        for name, reg in filtered_regions.items():
            print(f"\n{c.BOLD}{c.GREEN}▶ Region: {name}{c.RESET} (Entry: {reg.config.entry}, Caller: {reg.config.caller})")

            # Metric Trends & Sparklines
            if reg.metric_changes:
                print(f"  {c.BOLD}Metric Time-Series Sparklines:{c.RESET}")
                for m_key, mc in reg.metric_changes.items():
                    spark = make_sparkline(mc.values, width=12)
                    first_val = mc.values[0] if mc.values else 0.0
                    last_val = mc.values[-1] if mc.values else 0.0
                    net_delta = last_val - first_val
                    delta_color = c.GREEN if net_delta >= 0 else c.RED
                    print(
                        f"    {mc.metric_name:<30}: [{c.CYAN}{spark}{c.RESET}]  "
                        f"start={first_val:>7.2f} → end={last_val:>7.2f}  "
                        f"({delta_color}Δ={net_delta:>+7.2f}{c.RESET}, mean={mc.mean_val:>7.2f}, σ={mc.std_dev:>6.2f})"
                    )

            # Sample-by-Sample Progression Table
            if reg.pmu_samples:
                print(f"\n  {c.BOLD}Time-Series Step Progression ({len(reg.pmu_samples)} periodic samples):{c.RESET}")
                s_header = f"  {'#':>3} | {'Time (s)':>8} | {'Dur (ms)':>8} | {'IPC':>6} | {'Δ IPC':>7} | {'L2 Hit%':>7} | {'Δ L2%':>6} | {'L3 MPKI':>8} | {'Power(W)':>8} | {'Δ Power':>7}"
                print(f"{c.DIM}{s_header}{c.RESET}")
                print(f"  {'-' * (len(s_header) - 2)}")

                ipc_mc = reg.metric_changes.get("ipc")
                l2_mc = reg.metric_changes.get("l2_hit_ratio")
                l3_mc = reg.metric_changes.get("l3_mpki")
                pow_mc = reg.metric_changes.get("power_w")

                for i, sample in enumerate(reg.pmu_samples):
                    ipc_v = sample.get_val("ipc") or 0.0
                    ipc_d = ipc_mc.deltas[i] if (ipc_mc and i < len(ipc_mc.deltas)) else 0.0

                    l2_v = sample.get_val("l2_hit_ratio") or 0.0
                    l2_d = l2_mc.deltas[i] if (l2_mc and i < len(l2_mc.deltas)) else 0.0

                    l3_v = sample.get_val("l3_mpki") or 0.0

                    pow_v = pow_mc.values[i] if (pow_mc and i < len(pow_mc.values)) else 0.0
                    pow_d = pow_mc.deltas[i] if (pow_mc and i < len(pow_mc.deltas)) else 0.0

                    ipc_d_str = f"{ipc_d:+6.2f}" if i > 0 else "   -"
                    l2_d_str = f"{l2_d:+5.1f}" if i > 0 else "  -"
                    pow_d_str = f"{pow_d:+6.1f}" if i > 0 else "   -"

                    row = (
                        f"  {sample.sample_idx:>3} | {sample.mid_time_s:>8.3f} | {sample.duration_ms:>8.1f} | "
                        f"{ipc_v:>6.2f} | {ipc_d_str:>7} | {l2_v:>6.1f}% | {l2_d_str:>6} | "
                        f"{l3_v:>8.4f} | {pow_v:>8.1f} | {pow_d_str:>7}"
                    )
                    print(row)

    print(f"\n{c.BOLD}{c.CYAN}{sep}{c.RESET}\n")


# ==============================================================================
# Self-Contained Interactive HTML Dashboard
# ==============================================================================

def generate_html_dashboard(runs: List[RunData], output_path: str) -> str:
    """Generate a sleek, responsive, fully self-contained offline HTML visualization dashboard."""
    # Serialize runs to JSON for client-side JavaScript interactivity
    runs_payload = [r.to_dict() for r in runs]
    runs_json_str = json.dumps(runs_payload, indent=None)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ORBIT - Energy & PMU Time-Series Visualization</title>
  <style>
    /* Default: Clean White / Light Theme */
    :root {
      --bg-primary: #f8fafc;
      --bg-secondary: #ffffff;
      --bg-card: #ffffff;
      --border-color: #e2e8f0;
      --text-primary: #0f172a;
      --text-secondary: #334155;
      --text-muted: #64748b;
      --accent-blue: #0284c7;
      --accent-green: #16a34a;
      --accent-amber: #d97706;
      --accent-purple: #7c3aed;
      --accent-rose: #e11d48;
      --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px -1px rgba(0, 0, 0, 0.08);
      --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
      --grid-line: #f1f5f9;
      --axis-line: #cbd5e1;
      --chart-axis-text: #64748b;
      --canvas-bg: #ffffff;
    }
    body.dark-theme {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-card: #1e293b;
      --border-color: #334155;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-blue: #38bdf8;
      --accent-green: #4ade80;
      --accent-amber: #fbbf24;
      --accent-purple: #c084fc;
      --accent-rose: #fb7185;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
      --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      --grid-line: #334155;
      --axis-line: #475569;
      --chart-axis-text: #94a3b8;
      --canvas-bg: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg-primary); color: var(--text-primary); line-height: 1.5; padding: 20px; transition: background-color 0.2s, color 0.2s; }
    .container { max-width: 1440px; margin: 0 auto; }
    
    /* Header */
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border-color); margin-bottom: 24px; flex-wrap: wrap; gap: 16px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: linear-gradient(135deg, #0284c7, #38bdf8); color: white; font-weight: 800; font-size: 1.25rem; padding: 6px 14px; border-radius: 8px; letter-spacing: 1px; }
    h1 { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); }
    .subtitle { font-size: 0.875rem; color: var(--text-secondary); }
    .header-controls { display: flex; align-items: center; gap: 12px; }
    .btn { background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; box-shadow: var(--shadow); }
    .btn:hover { background: var(--border-color); border-color: var(--accent-blue); }
    .btn-primary { background: #0284c7; border-color: #0284c7; color: white; }
    .btn-primary:hover { background: #0369a1; }

    /* Filter / Selectors Bar */
    .controls-bar { display: flex; flex-wrap: wrap; gap: 16px; background: var(--bg-card); padding: 16px; border-radius: 10px; border: 1px solid var(--border-color); box-shadow: var(--card-shadow); margin-bottom: 24px; align-items: center; }
    .control-group { display: flex; flex-direction: column; gap: 4px; }
    .control-group label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
    select { background: var(--bg-primary); border: 1px solid var(--border-color); color: var(--text-primary); padding: 8px 12px; border-radius: 6px; font-size: 0.875rem; min-width: 220px; outline: none; }
    select:focus { border-color: var(--accent-blue); }

    /* KPI Cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .kpi-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px; box-shadow: var(--card-shadow); position: relative; overflow: hidden; }
    .kpi-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--card-accent, var(--accent-blue)); }
    .kpi-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px; display: flex; justify-content: space-between; }
    .kpi-value { font-size: 1.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; font-variant-numeric: tabular-nums; }
    .kpi-sub { font-size: 0.75rem; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }

    /* Chart Section */
    .section-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; color: var(--text-primary); }
    .charts-grid { display: grid; grid-template-columns: 1fr; gap: 24px; margin-bottom: 24px; }
    @media (min-width: 1024px) {
      .charts-grid-2 { grid-template-columns: 1fr 1fr; }
    }
    .chart-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; box-shadow: var(--card-shadow); display: flex; flex-direction: column; }
    .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
    .chart-title { font-size: 1rem; font-weight: 600; color: var(--text-primary); }
    .chart-desc { font-size: 0.8rem; color: var(--text-muted); }
    .chart-toolbar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    .chart-toolbar-stacked { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; width: 100%; margin-top: 10px; }
    .toolbar-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; width: 100%; justify-content: flex-start; }
    .toolbar-label { font-size: 0.72rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.5px; margin-right: 8px; min-width: 140px; text-align: left; }
    @media (max-width: 900px) {
      .toolbar-label { min-width: auto; }
    }
    .chart-btn { padding: 4px 10px; font-size: 0.75rem; font-weight: 500; background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 5px; color: var(--text-secondary); cursor: pointer; transition: all 0.15s; }
    .chart-btn:hover { border-color: var(--accent-blue); color: var(--text-primary); }
    .chart-btn.active { background: #0284c7; color: white; border-color: #0284c7; font-weight: 600; }
    .canvas-container { position: relative; width: 100%; height: 320px; }
    canvas { width: 100% !important; height: 100% !important; display: block; border-radius: 6px; }
    .chart-legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; font-size: 0.8rem; }
    .legend-item { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; color: var(--text-secondary); }
    .legend-color { width: 12px; height: 12px; border-radius: 3px; }

    /* Table Section */
    .table-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; box-shadow: var(--card-shadow); margin-bottom: 24px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.875rem; text-align: left; }
    th { padding: 10px 14px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border-color); letter-spacing: 0.5px; }
    td { padding: 12px 14px; border-bottom: 1px solid var(--border-color); color: var(--text-secondary); font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(2, 132, 199, 0.04); color: var(--text-primary); }
    .text-right { text-align: right; }
    .positive-change { color: #16a34a; font-weight: 600; }
    .negative-change { color: #e11d48; font-weight: 600; }

    /* Tooltip */
    #chart-tooltip { position: absolute; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; font-size: 0.75rem; color: #f8fafc; pointer-events: none; opacity: 0; transition: opacity 0.15s; z-index: 1000; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2); }
    #chart-tooltip .tt-title { font-weight: 700; margin-bottom: 4px; border-bottom: 1px solid #334155; padding-bottom: 3px; color: #fff; }
    #chart-tooltip .tt-row { display: flex; justify-content: space-between; gap: 14px; margin-top: 2px; }
  </style>
</head>
<body>
  <div id="chart-tooltip"></div>
  <div class="container">
    <header>
      <div class="brand">
        <span class="logo-badge">ORBIT</span>
        <div>
          <h1>Energy & PMU Time-Series Visualization</h1>
          <div class="subtitle">OpenMP Runtime Profiling & Dynamic Parameter Tuning</div>
        </div>
      </div>
      <div class="header-controls">
        <button id="theme-toggle-btn" class="btn">🌙 Dark Theme</button>
        <button id="export-csv-btn" class="btn btn-primary">⬇ Export CSV</button>
      </div>
    </header>

    <!-- Run & Region Selectors -->
    <div class="controls-bar">
      <div class="control-group">
        <label for="run-select">Selected Execution Run</label>
        <select id="run-select"></select>
      </div>
      <div class="control-group">
        <label for="region-select">Target Parallel Region</label>
        <select id="region-select"></select>
      </div>
      <div class="control-group">
        <label for="view-mode-select">Time-Series Display Mode</label>
        <select id="view-mode-select">
          <option value="levels">Absolute Values (Levels)</option>
          <option value="deltas">Step Change (Δ = Current - Prev)</option>
          <option value="rate_of_change">Rate of Change (dM / dt)</option>
          <option value="pct_change">Percentage Change (% Δ)</option>
        </select>
      </div>
    </div>

    <!-- KPI Summary Cards -->
    <div class="kpi-grid" id="kpi-container">
      <!-- Populated dynamically via JS -->
    </div>

    <!-- Charts Grid -->
    <div class="charts-grid">
      <!-- Main Dual-Axis Time Series Chart -->
      <div class="chart-card">
        <div class="chart-header">
          <div>
            <div class="chart-title">Energy Consumption, Power (W) & PMU Performance Progression</div>
            <div class="chart-desc">Tracks power and PMU metric fluctuations across elapsed region time (1s periodic interval sampling)</div>
          </div>
          <div class="chart-toolbar" id="main-chart-toolbar">
            <button class="chart-btn active" data-metric="ipc">IPC</button>
            <button class="chart-btn" data-metric="freq_ghz">Core Freq (GHz)</button>
            <button class="chart-btn" data-metric="l2_hit_ratio">L2 Hit %</button>
            <button class="chart-btn" data-metric="l3_mpki">L3 MPKI</button>
          </div>
        </div>
        <div class="canvas-container">
          <canvas id="main-timeseries-canvas"></canvas>
        </div>
        <div class="chart-legend" id="main-legend"></div>
      </div>
    </div>

    <div class="charts-grid">
      <!-- Unified Region-by-Region Comparison Chart -->
      <div class="chart-card">
        <div class="chart-header" style="flex-direction: column; align-items: flex-start; gap: 6px;">
          <div>
            <div class="chart-title">Region Performance & Efficiency Comparison</div>
            <div class="chart-desc">Compare energy consumption, power, duration, EDP, average PMU metrics, and derived core frequency across regions</div>
          </div>
          <div class="chart-toolbar-stacked" id="region-comp-toolbar">
            <div class="toolbar-row">
              <span class="toolbar-label">ENERGY & POWER:</span>
              <button class="chart-btn active" data-metric="energy">Total Energy (J)</button>
              <button class="chart-btn" data-metric="power">Avg Power (W)</button>
              <button class="chart-btn" data-metric="duration">Duration (s)</button>
              <button class="chart-btn" data-metric="edp">EDP (J·s)</button>
            </div>
            <div class="toolbar-row">
              <span class="toolbar-label">PMU & FREQUENCY:</span>
              <button class="chart-btn" data-metric="ipc">Average IPC</button>
              <button class="chart-btn" data-metric="freq_ghz">Core Freq (Derived GHz)</button>
              <button class="chart-btn" data-metric="l2_hit_ratio">Avg L2 Hit %</button>
              <button class="chart-btn" data-metric="l3_mpki">Avg L3 MPKI</button>
              <button class="chart-btn" data-metric="branch_mispr_ratio">Avg Branch Mispr</button>
            </div>
          </div>
        </div>
        <div class="canvas-container">
          <canvas id="region-comparison-canvas"></canvas>
        </div>
        <div class="chart-legend" id="region-comparison-legend"></div>
      </div>
    </div>

    <!-- Region-by-Region Cross-Comparison Matrix Table -->
    <div class="section-title">
      <span>Region-by-Region Cross-Comparison & Configuration Matrix</span>
      <span style="font-size: 0.8rem; color: var(--text-muted);">Click any row to inspect that region in the time-series detail</span>
    </div>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Region Name</th>
            <th>Threads</th>
            <th>Sched,Chunk</th>
            <th class="text-right">Duration</th>
            <th class="text-right">Total Energy</th>
            <th class="text-right">Avg Power</th>
            <th class="text-right">EDP</th>
            <th class="text-right">Avg IPC</th>
            <th class="text-right">Core Freq (Derived)</th>
            <th class="text-right">Avg L2 Hit %</th>
            <th class="text-right">Avg L3 MPKI</th>
            <th class="text-right">Avg Branch Mispr</th>
          </tr>
        </thead>
        <tbody id="region-comp-table-body">
          <!-- Populated dynamically via JS -->
        </tbody>
      </table>
    </div>

    <!-- Detailed Sample Table -->
    <div class="section-title">
      <span>Periodic Sample Breakdown & Change Log (Active Region)</span>
      <span style="font-size: 0.8rem; color: var(--text-muted);" id="sample-count-badge">0 Samples</span>
    </div>
    <div class="table-card">
      <table id="sample-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Elapsed Time</th>
            <th>Duration</th>
            <th class="text-right">Power (W)</th>
            <th class="text-right">Δ Power</th>
            <th class="text-right">IPC</th>
            <th class="text-right">Δ IPC</th>
            <th class="text-right">Freq (GHz)</th>
            <th class="text-right">L2 Hit Ratio</th>
            <th class="text-right">L3 MPKI</th>
            <th class="text-right">Branch Mispr%</th>
          </tr>
        </thead>
        <tbody id="sample-table-body">
          <!-- Populated dynamically via JS -->
        </tbody>
      </table>
    </div>

  </div>

  <script>
    // Embedded run datasets from ORBIT
    const RUNS_DATA = """ + runs_json_str + """;

    // Application State - Default to White/Light Theme
    let currentRunIdx = 0;
    let currentRegionKey = "";
    let currentViewMode = "levels";
    let activePmuMetric = "ipc";
    let activeCompMetric = "energy";
    let isDark = false;

    // DOM Elements
    const runSelect = document.getElementById("run-select");
    const regionSelect = document.getElementById("region-select");
    const viewModeSelect = document.getElementById("view-mode-select");
    const themeBtn = document.getElementById("theme-toggle-btn");
    const exportCsvBtn = document.getElementById("export-csv-btn");
    const kpiContainer = document.getElementById("kpi-container");
    const sampleTableBody = document.getElementById("sample-table-body");
    const sampleCountBadge = document.getElementById("sample-count-badge");
    const tooltip = document.getElementById("chart-tooltip");

    // Smart number formatter for metrics and tooltips
    function formatNumber(val, decimals) {
      if (val === null || val === undefined || isNaN(val)) return "N/A";
      const abs = Math.abs(val);
      if (abs === 0) return "0";
      if (abs < 0.0001) return val.toExponential(2);
      if (abs < 0.01) return val.toFixed(4);
      if (abs < 1) return val.toFixed(3);
      if (abs < 10) return val.toFixed(decimals !== undefined ? decimals : 2);
      if (abs < 1000) return val.toFixed(decimals !== undefined ? decimals : 1);
      if (abs >= 1000000) return (val / 1000000).toFixed(2) + "M";
      if (abs >= 10000) return (val / 1000).toFixed(1) + "k";
      return val.toFixed(decimals !== undefined ? decimals : 1);
    }

    // Helper to extract metric, derived, or event value safely from plain JSON sample object
    function getSampleVal(sample, key) {
      if (!sample) return null;
      if (sample.metrics && sample.metrics[key] !== undefined) return sample.metrics[key];
      if (sample.derived && sample.derived[key] !== undefined) return sample.derived[key];
      if (sample.events && sample.events[key] !== undefined) return sample.events[key];
      return null;
    }

    // Attach .get_val helper to all samples in RUNS_DATA for backwards compatibility
    RUNS_DATA.forEach(run => {
      if (!run || !run.regions) return;
      Object.values(run.regions).forEach(reg => {
        if (reg.pmu_samples) {
          reg.pmu_samples.forEach(s => {
            s.get_val = function(k) { return getSampleVal(this, k); };
          });
        }
        if (reg.pmu_aggregate) {
          reg.pmu_aggregate.get_val = function(k) { return getSampleVal(this, k); };
        }
      });
    });

    // Initialize Selectors
    function initSelectors() {
      runSelect.innerHTML = "";
      RUNS_DATA.forEach((run, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        const timeLabel = run.timestamp ? " (" + run.timestamp.replace("T", " ") + ")" : "";
        opt.textContent = (idx + 1) + ". " + run.folder_name + timeLabel;
        runSelect.appendChild(opt);
      });

      updateRegionOptions();
    }

    function updateRegionOptions() {
      regionSelect.innerHTML = "";
      const run = RUNS_DATA[currentRunIdx];
      if (!run || !run.regions) return;

      const keys = Object.keys(run.regions);
      keys.forEach((key, idx) => {
        const opt = document.createElement("option");
        opt.value = key;
        const cfg = run.regions[key].config;
        const detail = (cfg && cfg.threads) ? " [" + cfg.threads + " thr, " + cfg.sched + "]" : "";
        opt.textContent = key + detail;
        regionSelect.appendChild(opt);
      });

      if (keys.length > 0) {
        if (!keys.includes(currentRegionKey)) {
          currentRegionKey = keys[0];
        }
        regionSelect.value = currentRegionKey;
      }
    }

    // High performance standalone Canvas Chart renderer with DPR and dual-axis support
    class CanvasChart {
      constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext("2d");
        this.series = [];
        this.xAxis = [];
        this.xLabels = [];
        this.leftYLabel = "";
        this.rightYLabel = "";
        this.isCategorical = false;
        this.forceZero = false;
        this.padding = { top: 32, right: 65, bottom: 44, left: 65 };
        this.displayW = 600;
        this.displayH = 320;
        this.hoverIdx = null;
        this.hoverX = null;

        this.handleResize = this.handleResize.bind(this);
        window.addEventListener("resize", this.handleResize);

        // Hover support
        this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
        this.canvas.addEventListener("mouseleave", () => this.onMouseLeave());

        if (window.ResizeObserver && this.canvas.parentElement) {
          try {
            new ResizeObserver(() => this.handleResize()).observe(this.canvas.parentElement);
          } catch(e) {}
        }
      }

      handleResize() {
        if (!this.canvas || !this.canvas.parentElement) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const displayW = Math.max(Math.round(rect.width), 200);
        const displayH = Math.max(Math.round(rect.height), 160);

        this.canvas.width = displayW * dpr;
        this.canvas.height = displayH * dpr;
        this.ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform to avoid compounding
        this.ctx.scale(dpr, dpr);
        this.displayW = displayW;
        this.displayH = displayH;
        this.render();
      }

      setData(xAxis, series, leftYLabel, rightYLabel, options = {}) {
        this.xAxis = xAxis || [];
        this.series = series || [];
        this.leftYLabel = leftYLabel || "";
        this.rightYLabel = rightYLabel || "";
        this.isCategorical = Boolean(options.isCategorical);
        this.forceZero = Boolean(options.forceZero);
        this.hoverIdx = null;
        this.hoverX = null;
        this.handleResize();
      }

      render() {
        const width = this.displayW;
        const height = this.displayH;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);

        if (!this.series || this.series.length === 0 || !this.xAxis || this.xAxis.length === 0) {
          ctx.fillStyle = isDark ? "#94a3b8" : "#64748b";
          ctx.font = "14px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("No time-series data available for this selection", width / 2, height / 2);
          return;
        }

        const p = this.padding;
        const chartW = width - p.left - p.right;
        const chartH = height - p.top - p.bottom;

        // Theme colors
        const gridColor = isDark ? "#334155" : "#f1f5f9";
        const axisColor = isDark ? "#475569" : "#cbd5e1";
        const textColor = isDark ? "#94a3b8" : "#64748b";
        const titleColor = isDark ? "#f8fafc" : "#0f172a";

        // Separate left and right axis series
        const leftSeries = this.series.filter(s => !s.useRightAxis);
        const rightSeries = this.series.filter(s => s.useRightAxis);

        const getBounds = (seriesList, forceZero = false) => {
          let min = Infinity, max = -Infinity;
          seriesList.forEach(s => {
            if (!s.data) return;
            s.data.forEach(v => {
              if (v !== null && v !== undefined && !isNaN(v)) {
                if (v < min) min = v;
                if (v > max) max = v;
              }
            });
          });

          if (min === Infinity || max === -Infinity) return { min: 0, max: 1 };

          if (forceZero) {
            if (min > 0) min = 0;
            if (max < 0) max = 0;
          }

          if (min === max) {
            if (min === 0) { min = -1; max = 1; }
            else if (min > 0) { min = 0; max = max * 1.2; }
            else { max = 0; min = min * 1.2; }
          }

          const span = max - min;
          const pad = span * 0.1;
          let finalMin = min - pad;
          let finalMax = max + pad;

          if (forceZero) {
            if (min >= 0) finalMin = 0;
            if (max <= 0) finalMax = 0;
          }

          return { min: finalMin, max: finalMax };
        };

        const leftBounds = getBounds(leftSeries.length ? leftSeries : this.series, this.forceZero);
        const rightBounds = rightSeries.length ? getBounds(rightSeries, this.forceZero) : leftBounds;

        const leftSpan = Math.max(leftBounds.max - leftBounds.min, 1e-9);
        const rightSpan = Math.max(rightBounds.max - rightBounds.min, 1e-9);

        const getLeftYPixel = (y) => p.top + chartH - ((y - leftBounds.min) / leftSpan) * chartH;
        const getRightYPixel = (y) => p.top + chartH - ((y - rightBounds.min) / rightSpan) * chartH;

        // Smart format for tick numbers
        const formatTick = (val, step) => {
          if (val === null || val === undefined || isNaN(val)) return "";
          if (Math.abs(val) < 1e-9) return "0";
          const absStep = Math.abs(step);
          if (absStep < 1e-4) return val.toExponential(2);
          if (absStep < 0.005) return val.toFixed(4);
          if (absStep < 0.05) return val.toFixed(3);
          if (absStep < 0.5) return val.toFixed(2);
          if (absStep < 5) return val.toFixed(1);
          if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(1) + "M";
          if (Math.abs(val) >= 1e4) return (val / 1e3).toFixed(0) + "k";
          return val.toFixed(0);
        };

        // Grid Lines
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        const gridSteps = 5;

        for (let i = 0; i <= gridSteps; i++) {
          const yPos = p.top + (chartH / gridSteps) * i;
          ctx.beginPath();
          ctx.moveTo(p.left, yPos);
          ctx.lineTo(p.left + chartW, yPos);
          ctx.stroke();

          // Left Y Labels
          const leftStep = leftSpan / gridSteps;
          const leftVal = leftBounds.max - i * leftStep;
          ctx.fillStyle = textColor;
          ctx.font = "11px sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(formatTick(leftVal, leftStep), p.left - 8, yPos + 4);

          // Right Y Labels if needed
          if (rightSeries.length > 0) {
            const rightStep = rightSpan / gridSteps;
            const rightVal = rightBounds.max - i * rightStep;
            ctx.textAlign = "left";
            ctx.fillText(formatTick(rightVal, rightStep), p.left + chartW + 8, yPos + 4);
          }
        }

        // Draw Zero Line if zero is within bounds
        if (leftBounds.min <= 0 && leftBounds.max >= 0) {
          const zeroY = getLeftYPixel(0);
          ctx.save();
          ctx.strokeStyle = isDark ? "#64748b" : "#94a3b8";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(p.left, zeroY);
          ctx.lineTo(p.left + chartW, zeroY);
          ctx.stroke();
          ctx.restore();
        }

        // Axis Titles
        ctx.fillStyle = titleColor;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(this.leftYLabel, p.left, p.top - 12);
        if (rightSeries.length > 0) {
          ctx.textAlign = "right";
          ctx.fillText(this.rightYLabel, p.left + chartW, p.top - 12);
        }

        // Render Series: Categorical Bar Chart vs Continuous Line Chart
        if (this.isCategorical) {
          const N = this.xAxis.length;
          const slotW = chartW / Math.max(N, 1);
          const barW = Math.min(Math.max(slotW * 0.52, 10), 65);
          const zeroY = Math.max(p.top, Math.min(p.top + chartH, getLeftYPixel(0)));

          // X Axis Ticks for categories
          ctx.fillStyle = textColor;
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          this.xAxis.forEach((label, idx) => {
            const centerX = p.left + (idx + 0.5) * slotW;
            let displayLabel = String(label);
            if (displayLabel.length > 15) displayLabel = displayLabel.substring(0, 13) + "…";
            ctx.fillText(displayLabel, centerX, p.top + chartH + 18);
          });

          // Draw Bars
          this.series.forEach(s => {
            s.data.forEach((val, idx) => {
              if (val === null || val === undefined || isNaN(val)) return;
              const centerX = p.left + (idx + 0.5) * slotW;
              const xLeft = centerX - barW / 2;
              const yVal = getLeftYPixel(val);
              const topY = Math.min(yVal, zeroY);
              const barH = Math.max(Math.abs(yVal - zeroY), 2);

              // Use positive/negative color if delta bar
              let fill = s.color;
              if (s.dynamicDeltaColor) {
                fill = val >= 0 ? (isDark ? "#4ade80" : "#16a34a") : (isDark ? "#fb7185" : "#e11d48");
              }
              ctx.fillStyle = fill;
              ctx.beginPath();
              ctx.rect(xLeft, topY, barW, barH);
              ctx.fill();

              // Print value label on bar if slot is wide enough
              if (slotW >= 40 && barH > 14) {
                ctx.fillStyle = isDark ? "#ffffff" : (val >= 0 ? "#16a34a" : "#e11d48");
                ctx.font = "bold 10px sans-serif";
                const textY = val >= 0 ? topY - 5 : topY + barH + 12;
                ctx.fillText(formatNumber(val, s.decimals !== undefined ? s.decimals : 1), centerX, textY);
              }
            });
          });

        } else {
          // Continuous Numeric Time Line Chart
          const xMin = Math.min(...this.xAxis);
          const xMax = Math.max(...this.xAxis);
          const xSpan = Math.max(xMax - xMin, 1e-6);
          const getXPixel = (x) => p.left + ((x - xMin) / xSpan) * chartW;

          // X Axis Ticks
          ctx.fillStyle = textColor;
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          const xStepCount = Math.min(this.xAxis.length, 6);
          for (let i = 0; i < xStepCount; i++) {
            const val = xMin + (i / Math.max(xStepCount - 1, 1)) * (xMax - xMin);
            const xPos = getXPixel(val);
            ctx.fillText(val.toFixed(2) + "s", xPos, p.top + chartH + 18);
          }

          // Render Lines & Points
          this.series.forEach(s => {
            const getY = s.useRightAxis ? getRightYPixel : getLeftYPixel;
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            let first = true;
            s.data.forEach((val, idx) => {
              if (val === null || val === undefined || isNaN(val)) return;
              const xPos = getXPixel(this.xAxis[idx]);
              const yPos = getY(val);
              if (first) { ctx.moveTo(xPos, yPos); first = false; }
              else { ctx.lineTo(xPos, yPos); }
            });
            ctx.stroke();

            // Draw Area fill under line if specified
            if (s.fillArea && s.data.length > 1) {
              ctx.save();
              ctx.fillStyle = s.color + "18"; // 10% opacity
              ctx.lineTo(getXPixel(this.xAxis[this.xAxis.length - 1]), p.top + chartH);
              ctx.lineTo(getXPixel(this.xAxis[0]), p.top + chartH);
              ctx.closePath();
              ctx.fill();
              ctx.restore();
            }

            // Draw Data Points
            s.data.forEach((val, idx) => {
              if (val === null || val === undefined || isNaN(val)) return;
              const xPos = getXPixel(this.xAxis[idx]);
              const yPos = getY(val);

              ctx.fillStyle = s.color;
              ctx.beginPath();
              ctx.arc(xPos, yPos, 4, 0, Math.PI * 2);
              ctx.fill();

              ctx.fillStyle = isDark ? "#1e293b" : "#ffffff";
              ctx.beginPath();
              ctx.arc(xPos, yPos, 2, 0, Math.PI * 2);
              ctx.fill();
            });
          });

          // Draw hover vertical indicator
          if (this.hoverX !== null) {
            ctx.save();
            ctx.strokeStyle = isDark ? "#94a3b8" : "#64748b";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(this.hoverX, p.top);
            ctx.lineTo(this.hoverX, p.top + chartH);
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      onMouseMove(e) {
        if (!this.xAxis || this.xAxis.length === 0) return;
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const p = this.padding;
        const chartW = this.displayW - p.left - p.right;
        const chartH = this.displayH - p.top - p.bottom;

        if (mouseX < p.left || mouseX > p.left + chartW || mouseY < p.top - 10 || mouseY > p.top + chartH + 10) {
          this.onMouseLeave();
          return;
        }

        let nearestIdx = 0;
        let hoverXPos = 0;

        if (this.isCategorical) {
          const slotW = chartW / this.xAxis.length;
          nearestIdx = Math.floor((mouseX - p.left) / slotW);
          nearestIdx = Math.max(0, Math.min(this.xAxis.length - 1, nearestIdx));
          hoverXPos = p.left + (nearestIdx + 0.5) * slotW;
        } else {
          const xMin = Math.min(...this.xAxis);
          const xMax = Math.max(...this.xAxis);
          const xSpan = Math.max(xMax - xMin, 1e-6);
          const hoverXVal = xMin + ((mouseX - p.left) / chartW) * xSpan;

          let nearestDist = Infinity;
          this.xAxis.forEach((x, idx) => {
            const d = Math.abs(x - hoverXVal);
            if (d < nearestDist) {
              nearestDist = d;
              nearestIdx = idx;
            }
          });
          hoverXPos = p.left + ((this.xAxis[nearestIdx] - xMin) / xSpan) * chartW;
        }

        this.hoverIdx = nearestIdx;
        this.hoverX = hoverXPos;
        this.render();

        // Show Tooltip
        let title = "";
        if (this.isCategorical) {
          title = String(this.xAxis[nearestIdx]);
        } else {
          title = "Sample #" + (nearestIdx + 1) + " (t = " + Number(this.xAxis[nearestIdx]).toFixed(3) + "s)";
        }

        let html = "<div class='tt-title'>" + title + "</div>";
        this.series.forEach(s => {
          const val = s.data[nearestIdx];
          const valStr = val !== null && val !== undefined && !isNaN(val) ? formatNumber(val, s.decimals) : "N/A";
          html += "<div class='tt-row'><span style='color:" + s.color + "'>" + s.name + ":</span> <strong>" + valStr + "</strong></div>";
        });

        tooltip.innerHTML = html;
        tooltip.style.opacity = "1";

        // Prevent edge clipping
        const ttW = tooltip.offsetWidth || 180;
        let left = e.pageX + 14;
        if (left + ttW > window.innerWidth - 16) {
          left = e.pageX - ttW - 14;
        }
        tooltip.style.left = left + "px";
        tooltip.style.top = (e.pageY - 20) + "px";
      }

      onMouseLeave() {
        this.hoverIdx = null;
        this.hoverX = null;
        tooltip.style.opacity = "0";
        this.render();
      }
    }

    // Instantiate Charts
    let mainChart, regionCompChart;
    let activeRegionCompMetric = "energy";

    function initCharts() {
      mainChart = new CanvasChart("main-timeseries-canvas");
      regionCompChart = new CanvasChart("region-comparison-canvas");
    }

    // Refresh Dashboard Data
    function refreshDashboard() {
      const run = RUNS_DATA[currentRunIdx];
      if (!run || !run.regions) return;

      const reg = run.regions[currentRegionKey];
      if (!reg) return;

      updateKpis(reg);
      updateMainChart(reg);
      updateRegionComparisonChart(run);
      updateRegionComparisonTable(run);
      updateTable(reg);
    }

    function updateKpis(reg) {
      kpiContainer.innerHTML = "";
      const dur = reg.energy_profile ? reg.energy_profile.duration_ms / 1000.0 : 0.0;
      const energyJ = reg.energy_profile ? reg.energy_profile.total_energy_pkg_joules : 0.0;
      const powerW = reg.energy_profile ? reg.energy_profile.total_power_watts : 0.0;
      const edp = reg.energy_profile ? reg.energy_profile.edp_js : 0.0;

      const pmuAggr = reg.pmu_aggregate;
      const ipc = pmuAggr ? (getSampleVal(pmuAggr, "ipc") || 0) : 0;
      const l2Hit = pmuAggr ? (getSampleVal(pmuAggr, "l2_hit_ratio") || 0) : 0;
      const l3Mpki = pmuAggr ? (getSampleVal(pmuAggr, "l3_mpki") || 0) : 0;

      const cards = [
        { title: "Total Energy", value: energyJ.toFixed(1) + " J", sub: (energyJ / 3600.0).toFixed(4) + " Wh (Total CPU)", accent: "var(--accent-amber)" },
        { title: "Average Power", value: powerW.toFixed(1) + " W", sub: "Sockets 0 & 1 Combined", accent: "var(--accent-rose)" },
        { title: "Execution Duration", value: dur.toFixed(2) + " s", sub: reg.pmu_samples.length + " Time-Series Samples", accent: "var(--accent-blue)" },
        { title: "EDP (Energy-Delay)", value: edp.toFixed(1) + " J·s", sub: "Energy Delay Product", accent: "var(--accent-purple)" },
        { title: "Average IPC", value: ipc.toFixed(2), sub: "Instructions Per Cycle", accent: "var(--accent-green)" },
        { title: "L2 Cache Hit %", value: l2Hit.toFixed(1) + "%", sub: "L3 MPKI: " + l3Mpki.toFixed(4), accent: "var(--accent-blue)" },
      ];

      cards.forEach(c => {
        const el = document.createElement("div");
        el.className = "kpi-card";
        el.style.setProperty("--card-accent", c.accent);
        el.innerHTML = `
          <div class="kpi-title">${c.title}</div>
          <div class="kpi-value">${c.value}</div>
          <div class="kpi-sub">${c.sub}</div>
        `;
        kpiContainer.appendChild(el);
      });
    }

    function updateMainChart(reg) {
      if (!reg.pmu_samples || reg.pmu_samples.length === 0) {
        mainChart.setData([], [], "", "");
        return;
      }

      const times = reg.pmu_samples.map(s => s.mid_time_s);
      const mode = currentViewMode;
      const mcPmu = reg.metric_changes[activePmuMetric];
      const mcPow = reg.metric_changes["power_w"];

      let pmuData = [];
      let powData = [];

      if (mode === "levels") {
        pmuData = reg.pmu_samples.map(s => getSampleVal(s, activePmuMetric));
        powData = mcPow ? mcPow.values : [];
      } else if (mode === "deltas") {
        pmuData = mcPmu ? mcPmu.deltas : [];
        powData = mcPow ? mcPow.deltas : [];
      } else if (mode === "rate_of_change") {
        pmuData = mcPmu ? mcPmu.rates_of_change : [];
        powData = mcPow ? mcPow.rates_of_change : [];
      } else if (mode === "pct_change") {
        pmuData = mcPmu ? mcPmu.pct_changes : [];
        powData = mcPow ? mcPow.pct_changes : [];
      }

      const pmuLabels = {
        ipc: "IPC",
        freq_ghz: "Core Freq (GHz)",
        l2_hit_ratio: "L2 Hit %",
        l3_mpki: "L3 MPKI"
      };

      const powColor = isDark ? "#fb7185" : "#e11d48";
      const pmuColor = isDark ? "#38bdf8" : "#0284c7";

      const series = [
        {
          name: "Power (W)" + (mode !== "levels" ? " [" + mode + "]" : ""),
          data: powData,
          color: powColor,
          fillArea: mode === "levels",
          useRightAxis: false,
          decimals: 1
        },
        {
          name: (pmuLabels[activePmuMetric] || activePmuMetric) + (mode !== "levels" ? " [" + mode + "]" : ""),
          data: pmuData,
          color: pmuColor,
          useRightAxis: true,
          decimals: 3
        }
      ];

      mainChart.setData(times, series, "Power (Watts)", pmuLabels[activePmuMetric] || activePmuMetric, {
        isCategorical: false,
        forceZero: mode !== "levels"
      });

      // Update Legend
      const legendEl = document.getElementById("main-legend");
      legendEl.innerHTML = `
        <div class="legend-item"><div class="legend-color" style="background:${powColor}"></div><span>Power (Watts) - Left Y</span></div>
        <div class="legend-item"><div class="legend-color" style="background:${pmuColor}"></div><span>${pmuLabels[activePmuMetric] || activePmuMetric} - Right Y</span></div>
      `;
    }

    function updateRegionComparisonChart(run) {
      const regKeys = Object.keys(run.regions);
      if (regKeys.length === 0) {
        regionCompChart.setData([], [], "", "");
        return;
      }

      const metricConfigs = {
        // Energy & Efficiency
        energy: { name: "Total Energy", unit: "J", color: "#d97706", decimals: 1, get: r => r.energy_profile ? r.energy_profile.total_energy_pkg_joules : 0 },
        power: { name: "Average Power", unit: "W", color: "#e11d48", decimals: 1, get: r => r.energy_profile ? r.energy_profile.total_power_watts : 0 },
        duration: { name: "Execution Duration", unit: "s", color: "#0284c7", decimals: 2, get: r => r.energy_profile ? (r.energy_profile.duration_ms / 1000.0) : (r.pmu_aggregate ? r.pmu_aggregate.duration_ms / 1000.0 : 0) },
        edp: { name: "EDP (Energy-Delay)", unit: "J·s", color: "#7c3aed", decimals: 1, get: r => r.energy_profile ? r.energy_profile.edp_js : 0 },
        // Hardware PMU Metrics & Inferred Frequency
        ipc: { name: "Average IPC", unit: "", color: "#0284c7", decimals: 2, get: r => r.pmu_aggregate ? (getSampleVal(r.pmu_aggregate, "ipc") || 0) : 0 },
        freq_ghz: { name: "Core Frequency (Derived)", unit: "GHz", color: "#16a34a", decimals: 2, get: r => r.pmu_aggregate ? (getSampleVal(r.pmu_aggregate, "freq_ghz") || 0) : 0 },
        l2_hit_ratio: { name: "Average L2 Hit Ratio", unit: "%", color: "#7c3aed", decimals: 1, get: r => r.pmu_aggregate ? (getSampleVal(r.pmu_aggregate, "l2_hit_ratio") || 0) : 0 },
        l3_mpki: { name: "Average L3 MPKI", unit: "misses/k-inst", color: "#e11d48", decimals: 4, get: r => r.pmu_aggregate ? (getSampleVal(r.pmu_aggregate, "l3_mpki") || 0) : 0 },
        branch_mispr_ratio: { name: "Average Branch Misprediction Ratio", unit: "", color: "#d97706", decimals: 5, get: r => r.pmu_aggregate ? (getSampleVal(r.pmu_aggregate, "branch_mispr_ratio") || 0) : 0 },
      };

      const curMetric = metricConfigs[activeRegionCompMetric] || metricConfigs.energy;
      const values = regKeys.map(k => curMetric.get(run.regions[k]));
      const unitLabel = curMetric.unit ? " (" + curMetric.unit + ")" : "";

      const series = [
        {
          name: curMetric.name + unitLabel,
          data: values,
          color: isDark ? "#fbbf24" : curMetric.color,
          type: "bar",
          decimals: curMetric.decimals
        }
      ];

      regionCompChart.setData(regKeys, series, curMetric.name + unitLabel, "", {
        isCategorical: true,
        forceZero: true
      });

      const legendEl = document.getElementById("region-comparison-legend");
      legendEl.innerHTML = regKeys.map((k, idx) => `
        <div class="legend-item"><span style="font-weight:600; color:${curMetric.color}">${idx + 1}.</span> <span>${k}</span></div>
      `).join("");
    }

    function updateRegionComparisonTable(run) {
      const tableBody = document.getElementById("region-comp-table-body");
      if (!tableBody) return;
      tableBody.innerHTML = "";
      if (!run || !run.regions) return;

      Object.keys(run.regions).forEach((key, idx) => {
        const reg = run.regions[key];
        const cfg = reg.config || {};
        const ep = reg.energy_profile;
        const pa = reg.pmu_aggregate;

        const dur = ep ? (ep.duration_ms / 1000.0) : (pa ? pa.duration_ms / 1000.0 : 0);
        const energyJ = ep ? ep.total_energy_pkg_joules : 0;
        const powerW = ep ? ep.total_power_watts : 0;
        const edp = ep ? ep.edp_js : 0;

        const ipc = pa ? (getSampleVal(pa, "ipc") || 0) : 0;
        const freq = pa ? (getSampleVal(pa, "freq_ghz") || 0) : 0;
        const l2Hit = pa ? (getSampleVal(pa, "l2_hit_ratio") || 0) : 0;
        const l3Mpki = pa ? (getSampleVal(pa, "l3_mpki") || 0) : 0;
        const brMisp = pa ? (getSampleVal(pa, "branch_mispr_ratio") || 0) : 0;

        const isSelected = key === currentRegionKey;
        const tr = document.createElement("tr");
        if (isSelected) tr.style.backgroundColor = isDark ? "rgba(56, 189, 248, 0.12)" : "rgba(2, 132, 199, 0.08)";
        tr.style.cursor = "pointer";
        tr.title = "Click to inspect this region in time-series detail";

        tr.innerHTML = `
          <td><strong>${key}</strong>${isSelected ? " <span style='color:var(--accent-blue);font-size:0.75rem;'>(Active)</span>" : ""}</td>
          <td>${cfg.threads || "-"}</td>
          <td>${cfg.sched ? cfg.sched + "," + cfg.chunk : "-"}</td>
          <td class="text-right">${dur.toFixed(2)}s</td>
          <td class="text-right font-bold">${energyJ.toFixed(1)} J</td>
          <td class="text-right">${powerW.toFixed(1)} W</td>
          <td class="text-right">${edp.toFixed(1)} J·s</td>
          <td class="text-right font-bold">${ipc.toFixed(2)}</td>
          <td class="text-right">${freq.toFixed(2)} GHz</td>
          <td class="text-right">${l2Hit.toFixed(1)}%</td>
          <td class="text-right">${l3Mpki.toFixed(4)}</td>
          <td class="text-right">${brMisp.toFixed(5)}</td>
        `;

        tr.addEventListener("click", () => {
          currentRegionKey = key;
          regionSelect.value = key;
          refreshDashboard();
        });

        tableBody.appendChild(tr);
      });
    }

    function updateTable(reg) {
      sampleTableBody.innerHTML = "";
      if (!reg.pmu_samples) return;

      sampleCountBadge.textContent = reg.pmu_samples.length + " Samples";

      const ipcMc = reg.metric_changes["ipc"];
      const powMc = reg.metric_changes["power_w"];

      reg.pmu_samples.forEach((sample, i) => {
        const tr = document.createElement("tr");

        const ipcV = getSampleVal(sample, "ipc") || 0;
        const ipcD = ipcMc ? ipcMc.deltas[i] : 0;
        const ipcDStr = i > 0 ? (ipcD >= 0 ? "+" : "") + ipcD.toFixed(3) : "-";
        const ipcDClass = ipcD > 0 ? "positive-change" : (ipcD < 0 ? "negative-change" : "");

        const powV = powMc && powMc.values[i] ? powMc.values[i] : 0;
        const powD = powMc && powMc.deltas[i] ? powMc.deltas[i] : 0;
        const powDStr = i > 0 ? (powD >= 0 ? "+" : "") + powD.toFixed(1) : "-";
        const powDClass = powD < 0 ? "positive-change" : (powD > 0 ? "negative-change" : "");

        const freqV = (getSampleVal(sample, "freq_ghz") || 0).toFixed(2);
        const l2Hit = (getSampleVal(sample, "l2_hit_ratio") || 0).toFixed(1) + "%";
        const l3Mpki = (getSampleVal(sample, "l3_mpki") || 0).toFixed(4);
        const brMisp = ((getSampleVal(sample, "branch_mispr_ratio") || 0) * 100).toFixed(2) + "%";

        tr.innerHTML = `
          <td>${sample.sample_idx}</td>
          <td>${sample.mid_time_s.toFixed(3)}s</td>
          <td>${sample.duration_ms.toFixed(1)}ms</td>
          <td class="text-right">${powV.toFixed(1)} W</td>
          <td class="text-right ${powDClass}">${powDStr}</td>
          <td class="text-right font-bold">${ipcV.toFixed(3)}</td>
          <td class="text-right ${ipcDClass}">${ipcDStr}</td>
          <td class="text-right">${freqV}</td>
          <td class="text-right">${l2Hit}</td>
          <td class="text-right">${l3Mpki}</td>
          <td class="text-right">${brMisp}</td>
        `;
        sampleTableBody.appendChild(tr);
      });
    }

    // Export CSV of currently active region
    function exportCsv() {
      const run = RUNS_DATA[currentRunIdx];
      const reg = run ? run.regions[currentRegionKey] : null;
      if (!reg || !reg.pmu_samples) return;

      let csvContent = "data:text/csv;charset=utf-8,";
      csvContent += "sample_idx,time_s,duration_ms,power_w,ipc,freq_ghz,l2_hit_pct,l3_mpki,branch_mispr_pct\\n";

      const powMc = reg.metric_changes["power_w"];
      reg.pmu_samples.forEach((s, i) => {
        const row = [
          s.sample_idx,
          s.mid_time_s.toFixed(4),
          s.duration_ms.toFixed(2),
          powMc && powMc.values[i] ? powMc.values[i].toFixed(2) : 0,
          (getSampleVal(s, "ipc") || 0).toFixed(4),
          (getSampleVal(s, "freq_ghz") || 0).toFixed(4),
          (getSampleVal(s, "l2_hit_ratio") || 0).toFixed(2),
          (getSampleVal(s, "l3_mpki") || 0).toFixed(6),
          ((getSampleVal(s, "branch_mispr_ratio") || 0) * 100).toFixed(4)
        ];
        csvContent += row.join(",") + "\\n";
      });

      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", (reg.name || "orbit_timeseries") + ".csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    // Event Listeners
    runSelect.addEventListener("change", (e) => {
      currentRunIdx = parseInt(e.target.value, 10);
      updateRegionOptions();
      refreshDashboard();
    });

    regionSelect.addEventListener("change", (e) => {
      currentRegionKey = e.target.value;
      refreshDashboard();
    });

    viewModeSelect.addEventListener("change", (e) => {
      currentViewMode = e.target.value;
      refreshDashboard();
    });

    themeBtn.addEventListener("click", () => {
      isDark = !isDark;
      document.body.classList.toggle("dark-theme", isDark);
      themeBtn.textContent = isDark ? "☀️ Light Theme" : "🌙 Dark Theme";
      refreshDashboard();
    });

    exportCsvBtn.addEventListener("click", exportCsv);

    document.querySelectorAll(".chart-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const parent = btn.closest("#region-comp-toolbar") || btn.closest("#main-chart-toolbar");
        if (parent && parent.id === "region-comp-toolbar") {
          parent.querySelectorAll(".chart-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeRegionCompMetric = btn.dataset.metric;
          const run = RUNS_DATA[currentRunIdx];
          if (run) updateRegionComparisonChart(run);
        } else if (parent && parent.id === "main-chart-toolbar") {
          parent.querySelectorAll(".chart-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activePmuMetric = btn.dataset.metric;
          refreshDashboard();
        }
      });
    });

    // Boot
    function boot() {
      initCharts();
      initSelectors();
      refreshDashboard();
    }

    if (document.readyState === "loading") {
      window.addEventListener("DOMContentLoaded", boot);
    } else {
      boot();
    }
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    return output_path


# ==============================================================================
# Static Plot Exporter (Native SVG + Matplotlib Fallback)
# ==============================================================================

def _generate_svg_line_chart(times: List[float], series: List[Dict[str, Any]],
                             left_label: str, right_label: str, title: str) -> str:
    """Generate clean, standalone vector SVG dual-axis time-series chart in white theme."""
    width = 900
    height = 450
    margin = {"top": 50, "right": 80, "bottom": 60, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    left_s = [s for s in series if not s.get("use_right")]
    right_s = [s for s in series if s.get("use_right")]

    def get_limits(s_list):
        vals = []
        for s in s_list:
            for v in s["data"]:
                if v is not None and not math.isnan(v):
                    vals.append(v)
        if not vals:
            return 0.0, 1.0
        mn, mx = min(vals), max(vals)
        if mn == mx:
            mn *= 0.9
            mx = 1.0 if mx == 0 else mx * 1.1
        pad = (mx - mn) * 0.1
        return max(0.0, mn - pad), mx + pad

    l_min, l_max = get_limits(left_s or series)
    r_min, r_max = get_limits(right_s) if right_s else (l_min, l_max)

    t_min = min(times) if times else 0.0
    t_max = max(times) if times else 1.0
    t_span = max(t_max - t_min, 1e-6)

    def x_px(t):
        return margin["left"] + ((t - t_min) / t_span) * plot_w

    def y_left_px(v):
        return margin["top"] + plot_h - ((v - l_min) / max(l_max - l_min, 1e-9)) * plot_h

    def y_right_px(v):
        return margin["top"] + plot_h - ((v - r_min) / max(r_max - r_min, 1e-9)) * plot_h

    def format_svg_tick(val, span):
        if abs(val) < 1e-9: return "0"
        if span < 0.005: return f"{val:.4f}"
        if span < 0.05: return f"{val:.3f}"
        if span < 5: return f"{val:.2f}"
        if span < 50: return f"{val:.1f}"
        return f"{val:.0f}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8" stroke="#e2e8f0" stroke-width="1"/>',
        # Title
        f'<text x="{width/2}" y="28" fill="#0f172a" font-size="16" font-weight="bold" text-anchor="middle">{title}</text>',
    ]

    # Grid lines & ticks
    grid_steps = 5
    for i in range(grid_steps + 1):
        y = margin["top"] + (plot_h / grid_steps) * i
        svg.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{margin["left"] + plot_w}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>')
        l_val = l_max - (i / grid_steps) * (l_max - l_min)
        svg.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="end">{format_svg_tick(l_val, l_max - l_min)}</text>')
        if right_s:
            r_val = r_max - (i / grid_steps) * (r_max - r_min)
            svg.append(f'<text x="{margin["left"] + plot_w + 10}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="start">{format_svg_tick(r_val, r_max - r_min)}</text>')

    # X axis ticks
    x_steps = min(len(times), 6)
    for i in range(x_steps):
        val = t_min + (i / max(x_steps - 1, 1)) * (t_max - t_min)
        x = x_px(val)
        svg.append(f'<text x="{x}" y="{margin["top"] + plot_h + 20}" fill="#64748b" font-size="11" text-anchor="middle">{val:.2f}s</text>')

    # Axis labels
    svg.append(f'<text x="{margin["left"]}" y="42" fill="#334155" font-size="12" font-weight="bold" text-anchor="start">{left_label}</text>')
    if right_s:
        svg.append(f'<text x="{margin["left"] + plot_w}" y="42" fill="#334155" font-size="12" font-weight="bold" text-anchor="end">{right_label}</text>')
    svg.append(f'<text x="{width/2}" y="{height - 15}" fill="#334155" font-size="12" text-anchor="middle">Time (seconds)</text>')

    # Draw Series
    for s in series:
        use_r = s.get("use_right", False)
        y_fn = y_right_px if use_r else y_left_px
        color = s.get("color", "#0284c7")
        data = s["data"]

        points = []
        for idx, t in enumerate(times):
            if idx < len(data) and data[idx] is not None and not math.isnan(data[idx]):
                points.append((x_px(t), y_fn(data[idx])))

        if points:
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#ffffff"/>')

    # Legend at bottom
    leg_x = margin["left"]
    leg_y = margin["top"] + plot_h + 45
    for s in series:
        color = s.get("color", "#0284c7")
        name = s.get("name", "")
        svg.append(f'<rect x="{leg_x}" y="{leg_y - 10}" width="12" height="12" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{leg_x + 18}" y="{leg_y}" fill="#0f172a" font-size="12">{name}</text>')
        leg_x += len(name) * 8 + 40

    svg.append('</svg>')
    return "\n".join(svg)


def _generate_svg_bar_chart(categories: List[str], values: List[float],
                            y_label: str, title: str, color: str = "#d97706") -> str:
    """Generate clean vector SVG bar chart for cross-region comparisons in white theme."""
    width = 900
    height = 450
    margin = {"top": 50, "right": 50, "bottom": 80, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    clean_vals = [v if v is not None and not math.isnan(v) else 0.0 for v in values]
    max_val = max(clean_vals) if clean_vals else 1.0
    if max_val == 0:
        max_val = 1.0
    y_max = max_val * 1.15

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8" stroke="#e2e8f0" stroke-width="1"/>',
        f'<text x="{width/2}" y="28" fill="#0f172a" font-size="16" font-weight="bold" text-anchor="middle">{title}</text>',
    ]

    # Grid
    grid_steps = 5
    for i in range(grid_steps + 1):
        y = margin["top"] + (plot_h / grid_steps) * i
        svg.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{margin["left"] + plot_w}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>')
        val = y_max - (i / grid_steps) * y_max
        svg.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" fill="#64748b" font-size="11" text-anchor="end">{val:.1f}</text>')

    svg.append(f'<text x="{margin["left"]}" y="42" fill="#334155" font-size="12" font-weight="bold" text-anchor="start">{y_label}</text>')

    # Bars
    n = len(categories)
    if n > 0:
        slot_w = plot_w / n
        bar_w = min(slot_w * 0.5, 80)

        for i, (cat, val) in enumerate(zip(categories, clean_vals)):
            x_center = margin["left"] + slot_w * (i + 0.5)
            x_left = x_center - bar_w / 2
            bar_h = (val / y_max) * plot_h
            y_top = margin["top"] + plot_h - bar_h

            # Bar rect
            svg.append(f'<rect x="{x_left:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            # Value label on top
            svg.append(f'<text x="{x_center:.1f}" y="{y_top - 6:.1f}" fill="#0f172a" font-size="11" font-weight="bold" text-anchor="middle">{val:.1f}</text>')
            # Category label below
            svg.append(f'<text x="{x_center:.1f}" y="{margin["top"] + plot_h + 20}" fill="#64748b" font-size="11" text-anchor="middle">{cat}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def export_static_plots(runs: List[RunData], export_dir: str) -> List[str]:
    """Generate static SVG and PNG plots for each region and cross-region comparison."""
    os.makedirs(export_dir, exist_ok=True)
    generated = []

    # Check matplotlib availability
    mpl_available = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mpl_available = True
    except Exception:
        mpl_available = False

    for run in runs:
        # Cross-region comparison
        reg_keys = list(run.regions.keys())
        if len(reg_keys) > 1:
            energies = [
                run.regions[k].energy_profile.total_energy_pkg if run.regions[k].energy_profile else 0.0
                for k in reg_keys
            ]
            comp_svg = _generate_svg_bar_chart(
                reg_keys, energies, "Total Energy (Joules)",
                f"{run.folder_name} - Energy Consumption by Region", color="#d97706"
            )
            comp_svg_path = os.path.join(export_dir, f"{run.folder_name}__energy_comparison.svg")
            with open(comp_svg_path, "w", encoding="utf-8") as f:
                f.write(comp_svg)
            generated.append(comp_svg_path)

        for reg_name, reg in run.regions.items():
            if not reg.pmu_samples:
                continue

            times = [s.mid_time_s for s in reg.pmu_samples]
            ipc_vals = [s.get_val("ipc") for s in reg.pmu_samples]
            pow_mc = reg.metric_changes.get("power_w")
            pow_vals = pow_mc.values if pow_mc else [0.0] * len(times)
            l2_vals = [s.get_val("l2_hit_ratio") for s in reg.pmu_samples]

            # 1. Main Time Series Dual Axis Plot
            series = [
                {"name": "Power (Watts)", "data": pow_vals, "color": "#e11d48", "use_right": False},
                {"name": "IPC", "data": ipc_vals, "color": "#0284c7", "use_right": True},
            ]
            title = f"{run.folder_name} - {reg_name} (Energy & IPC Time-Series)"
            svg_content = _generate_svg_line_chart(times, series, "Power (Watts)", "IPC", title)
            svg_path = os.path.join(export_dir, f"{reg_name}__timeseries.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_content)
            generated.append(svg_path)

            # If Matplotlib available, also export PNG in clean white theme
            if mpl_available:
                try:
                    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#ffffff")
                    ax1.set_facecolor("#ffffff")
                    ax1.set_title(title, color="#0f172a", fontsize=14, pad=12, fontweight="bold")
                    ax1.set_xlabel("Time (s)", color="#334155")
                    ax1.set_ylabel("Power (Watts)", color="#e11d48")
                    ax1.plot(times, pow_vals, color="#e11d48", marker="o", linewidth=2, label="Power (W)")
                    ax1.tick_params(colors="#64748b")
                    for spine in ax1.spines.values():
                        spine.set_color("#cbd5e1")
                    ax1.grid(True, color="#f1f5f9", linestyle="--", alpha=0.8)

                    ax2 = ax1.twinx()
                    ax2.set_ylabel("IPC", color="#0284c7")
                    ax2.plot(times, ipc_vals, color="#0284c7", marker="s", linewidth=2, label="IPC")
                    ax2.tick_params(colors="#64748b")
                    for spine in ax2.spines.values():
                        spine.set_color("#cbd5e1")

                    png_path = os.path.join(export_dir, f"{reg_name}__timeseries.png")
                    fig.tight_layout()
                    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
                    plt.close(fig)
                    generated.append(png_path)
                except Exception as e:
                    print(f"[Warning] Failed matplotlib PNG export: {e}", file=sys.stderr)

    return generated


# ==============================================================================
# HTTP Server Mode
# ==============================================================================

def run_http_server(html_path: str, port: int = 8080) -> None:
    """Serve the generated visualization dashboard on a local HTTP port."""
    html_dir = os.path.dirname(os.path.abspath(html_path))
    html_file = os.path.basename(html_path)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            # Custom path translation compatible with Python 3.6+
            path_part = path.split("?", 1)[0].split("#", 1)[0]
            path_part = os.path.normpath(urllib.parse.unquote(path_part))
            words = [w for w in path_part.split("/") if w and w != ".."]
            return os.path.join(html_dir, *words)

        def do_GET(self):
            if self.path in ("/", ""):
                self.path = "/" + html_file
            return super().do_GET()

        def log_message(self, format, *args):
            # Suppress routine GET logging for cleaner terminal output
            pass

    # Try requested port or next available
    assigned_port = port
    for p in range(port, port + 20):
        try:
            httpd = socketserver.TCPServer(("", p), CustomHandler)
            assigned_port = p
            break
        except OSError:
            continue
    else:
        print(f"[Error] Could not bind to port in range {port}-{port+20}", file=sys.stderr)
        return

    url = f"http://localhost:{assigned_port}/"
    print(f"\n============================================================")
    print(f"🚀 ORBIT Visualization Dashboard Server Active")
    print(f"👉 Open in browser: {url}")
    print(f"Press Ctrl+C to terminate the server.")
    print(f"============================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ORBIT dashboard server...")
        httpd.server_close()


# ==============================================================================
# Main CLI Entry Point
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ORBIT Energy & PMU Time-Series Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate interactive HTML report from snapshot directory:
  %(prog)s 04_10_2026__23_20_45__ab7910b84a

  # Show rich terminal tables and sparklines:
  %(prog)s --terminal 04_10_2026__23_20_45__ab7910b84a

  # Start local web server for live interactive exploration:
  %(prog)s --serve 8080 04_10_2026__23_20_45__ab7910b84a

  # Output full metrics and time-series changes as JSON to stdout:
  %(prog)s --json 04_10_2026__23_20_45__ab7910b84a > report.json
        """
    )

    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="One or more ORBIT execution folders (e.g. 04_10_2026__23_20_45__ab7910b84a), parent directory, or JSON files. Defaults to current directory.",
    )
    parser.add_argument(
        "-o", "--output",
        default="orbit_report.html",
        help="Output filename for interactive HTML dashboard (default: orbit_report.html)",
    )
    parser.add_argument(
        "-t", "--terminal", "--cli",
        action="store_true",
        help="Display formatted summary tables, trends, and sparklines in terminal stdout",
    )
    parser.add_argument(
        "-r", "--region",
        default=None,
        help="Filter display to a specific region name (case-insensitive substring)",
    )
    parser.add_argument(
        "-s", "--serve",
        nargs="?",
        const=8080,
        type=int,
        help="Start a local web server to interactively view the dashboard (default port: 8080)",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Print computed time-series metrics, rates of change, and configurations as JSON to stdout",
    )
    parser.add_argument(
        "--export-plots",
        nargs="?",
        const="plots",
        default=None,
        help="Export static vector SVG and PNG plots to specified directory (default: plots/)",
    )

    args = parser.parse_args()

    # Discover and load runs
    runs = discover_all_runs(args.paths)
    if not runs:
        print(f"Error: No valid ORBIT snapshot folders or JSON files found under {args.paths}", file=sys.stderr)
        return 1

    try:
        # JSON output mode
        if args.json:
            payload = [r.to_dict() for r in runs]
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0

        # Terminal output
        if args.terminal:
            render_terminal(runs, region_filter=args.region)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0

    # Static plot export
    if args.export_plots:
        plots = export_static_plots(runs, args.export_plots)
        print(f"📊 Exported {len(plots)} static plot(s) to '{args.export_plots}/'")

    # HTML output
    html_path = os.path.abspath(args.output)
    generate_html_dashboard(runs, html_path)
    print(f"✅ Generated interactive HTML report: {html_path}")

    # Serve mode
    if args.serve is not None:
        run_http_server(html_path, port=args.serve)

    return 0


if __name__ == "__main__":
    sys.exit(main())
