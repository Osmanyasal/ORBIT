import itertools, json, os, re, subprocess, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(TOOLS))  # folder holding ORBIT/ and the data
MODELS_OUT = os.path.join(ROOT, "models_best")
LOG_DIR = os.path.join(TOOLS, "predictor_sweep_logs")
PY = sys.executable
DATA = os.environ.get("ORBIT_DATA_DIR", os.path.join(ROOT, "P12-data"))
os.makedirs(LOG_DIR, exist_ok=True)
TARGET = sys.argv[1]
CV = sys.argv[2]
SEEDS = [0, 1]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

# network / optimisation settings around the default (hidden 64, 3 layers, dropout 0.2, 40 epochs, lr 2e-3)
VARIANTS = {
    "base": [],
    "h32": ["--hidden", "32"],
    "h128": ["--hidden", "128"],
    "l2": ["--layers", "2"],
    "l4": ["--layers", "4"],
    "drop0.1": ["--dropout", "0.1"],
    "drop0.4": ["--dropout", "0.4"],
    "ep20": ["--epochs", "20"],
    "ep80": ["--epochs", "80"],
    "lr1e-3": ["--lr", "1e-3"],
}
if len(sys.argv) > 4:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in sys.argv[4].split(",")}


def run(job):
    name, extra, seed = job
    cv_dir = f"{LOG_DIR}/{TARGET}_{CV}_{name}_s{seed}"
    cmd = [PY, "-u", f"{TOOLS}/orbit-predictor.py", "train", "--data-dir", DATA, "--cv-only", "--cv-mode", CV,
           "--targets", TARGET, "--seed", str(seed), "--out-dir", cv_dir, *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    open(f"{LOG_DIR}/{TARGET}_{CV}_{name}_s{seed}.log", "w").write(out)
    rows = {}
    for line in out.splitlines():
        m = re.match(r"^(MLP \(1 observation\)|global fixed config.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m and m.group(1)[:3] not in rows:
            rows[m.group(1)[:3]] = tuple(map(float, m.groups()[1:]))
    res = rows.get("MLP")
    print(name, seed, res, "fixed:", rows.get("glo"), flush=True)
    return name, seed, res, rows.get("glo")


jobs = [(n, e, s) for (n, e), s in itertools.product(VARIANTS.items(), SEEDS)]
with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(run, jobs))

print(f"\n{TARGET}, {CV} CV: extra cost vs. the best config of each region (%), mean over seeds {SEEDS}")
print(f"{'variant':10s} {'geomean':>8s} {'median':>8s} {'p90':>8s}   geomean per seed")
for n in VARIANTS:
    r = [x[2] for x in results if x[0] == n and x[2]]
    if r:
        print(f"{n:10s} " + " ".join(f"{sum(c)/len(c):8.2f}" for c in zip(*r)) + "   " + " ".join(f"{x[0]:.2f}" for x in r))
fixed = [x[3] for x in results if x[3]]
if fixed:
    print(f"{'fixed cfg':10s} " + " ".join(f"{fixed[0][i]:8.2f}" for i in range(3)) + "   (best single config, reference)")

# Train the final model of this target on all regions with the variant of the lowest mean geomean loss
# ("base" is the default setting, so it wins when no variant is better).
means = {n: sum(x[2][0] for x in results if x[0] == n and x[2]) / max(1, sum(1 for x in results if x[0] == n and x[2]))
         for n in VARIANTS if any(x[0] == n and x[2] for x in results)}
if not means:
    sys.exit("no usable results, final model not trained")
best = min(means, key=means.get)
print(f"\nbest variant for {TARGET}: {best} (geomean loss {means[best]:.2f}%)  ->  training the final model in {MODELS_OUT}")
cmd = [PY, "-u", f"{TOOLS}/orbit-predictor.py", "train", "--data-dir", DATA, "--cv-mode", "none",
       "--targets", TARGET, "--out-dir", MODELS_OUT, "--seed", "0", *VARIANTS[best]]
subprocess.run(cmd, check=True)

# Per-application table of the chosen variant, averaged over the seeds (for the paper)
tables = [pd.read_csv(f"{LOG_DIR}/{TARGET}_{CV}_{best}_s{s}/cv_per_app_{TARGET}.csv", index_col="application")
          for s in SEEDS if os.path.exists(f"{LOG_DIR}/{TARGET}_{CV}_{best}_s{s}/cv_per_app_{TARGET}.csv")]
if tables:
    pd.concat(tables).groupby("application", sort=False).mean().round(1).to_csv(
        f"{MODELS_OUT}/cv_per_app_{TARGET}.csv")
    print(f"per-application table (mean of {len(tables)} seeds) -> {MODELS_OUT}/cv_per_app_{TARGET}.csv")
with open(f"{MODELS_OUT}/best_{TARGET}.json", "w") as fh:
    json.dump({"target": TARGET, "variant": best, "args": VARIANTS[best], "cv": CV, "seeds": SEEDS,
               "geomean_loss_pct_by_variant": means}, fh, indent=2)
