# Populace: system requirements for local builds

**Status:** measured 2026-06-15. Numbers are wall-clock and peak-RSS readings
from the benchmark in the appendix, run against the workspace at this revision.
Re-run it on a new machine to refresh the certificate — these are operating
figures, not guarantees.

This document answers one question: **what does a machine need to develop and
build populace locally?** It separates the cheap paths (edit the library, run
the contract suite, consume a published population) from the expensive ones
(impute donor stages, calibrate a national pool), because they have very
different footprints and a single "recommended spec" hides that.

## TL;DR

The binding resource is **RAM**, and there are two distinct memory ceilings:
the **calibration compile** (scales with `targets × records`) and the **SCF
wealth imputation** (~15 GB, single-threaded). Disk is ~18 GB for a full build
setup. CPU core count barely matters — the slowest stage (QRF imputation) is
single-threaded, so single-core speed dominates.

| Machine RAM | What it can do |
|---|---|
| **16 GB** | Edit the library, run the full contract suite (11 GB peak), load and analyze a published populace population (~9 GB). **Cannot** run a national calibration build. Fine for a *consumer/contributor*, not a *builder*. |
| **32 GB** | Floor for building. Imputation stages and calibration up to ~300k records × ~3,700 targets (17 GB). Full-county calibration (6,288 targets) will swap. |
| **48 GB** | Comfortable end-to-end local build at the matched-N / small generate-big scale (≤300k pool, full county targets, 29 GB peak) with headroom. **Recommended.** |
| **64 GB+** | Headroom for bigger pools, but the 1M→3M generate-big rungs need cloud regardless (see [the cliffs](#where-the-cliffs-are)). |

**Recommended laptop:** MacBook Pro 14", M-series **Pro** (or Max), **48 GB**
unified memory, **1 TB** SSD. Rationale in [Hardware notes](#hardware-notes).

## What "building populace" involves

A US build is two heavy phases over a sampling Frame, plus loaders:

1. **Imputation** (`populace.fit`) — donor stages attach variables onto the pool
   with regime-gated, sequentially-chained, weighted-bootstrap quantile-regression
   forests: SCF wealth (~27 targets, the heavy one), SIPP tips/vehicles, CPS-ORG
   labor inputs, MEPS-IC premiums, prior-year ASEC income, ACS rent. Each stage
   is a QRF fit on a donor (tens of thousands of rows) then a predict onto the
   national pool. The US stages are declared in `US_SOURCE_MANIFEST`
   (`populace.build.us`); the heavy primitive each one drives is `populace.fit`,
   which is what the imputation benchmark below measures directly.
2. **Calibration** (`populace.calibrate`) — compile targets (national + county
   control totals) into a sparse constraint matrix over the pool's weight
   vector, then optimize log-weights with torch Adam (the bounded relative-error
   loss), optionally with L0 generate-big-then-prune. See
   `packages/populace-calibrate/src/populace/calibrate/solve.py`.
3. **Loaders** (`populace.data`) — pull a published population artifact from the
   Hugging Face Hub and return it as a policyengine engine dataset.

The heavy primitives (`populace.fit`, `populace.calibrate`) are live and measured
below at build scale; the US source-stage wiring is currently a declared manifest
being ported in, so the end-to-end build footprint is the projection of these
primitive measurements onto the survey source artifacts above.

## Measured footprint

All figures on the reference machine (Apple M5 Max, 128 GB, macOS 26.5.1;
Python 3.14.4, torch 2.12.0, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3,
scikit-learn 1.8.0; BLAS = Accelerate). Peak RSS is `ru_maxrss`, each run in its
own process.

### Disk

| Item | Size |
|---|---|
| Workspace `.venv` (core stack: torch, sklearn, scipy, pandas, policyengine-us) | 1.0 GB |
| US survey source artifacts (ASEC/PUF/SCF/ACS h5 — external build inputs, cached locally) | 14 GB |
| Hugging Face hub cache (published artifacts + donors) | 3.3 GB |
| **Total for a full local build setup** | **~18 GB** |

512 GB is sufficient; 1 TB is comfortable once multiple build outputs and dataset
vintages accumulate.

### Dev / test loop — `uv sync --all-packages && uv run pytest`

| Suite | Peak RSS | Notes |
|---|---|---|
| Full contract suite | **11.3 GB**, 82 s | dominated by the two below |
| `populace-data` only | 9.1 GB | loads a published US population (national h5 + policyengine-us) |
| `populace-frame` only | 3.8 GB | the policyengine-us adapter test |
| `populace-calibrate` / `-fit` / `-build` | < 0.5 GB each | pure-library contract tests |

Editing only the kernel/fit/calibrate logic and running *those* suites is a
sub-gigabyte workflow. The 11 GB peak is real but comes entirely from the data
loader and the rules-engine adapter, which a pure-library change can skip.

### Imputation — `populace.fit.RegimeGatedQRF` (single-threaded)

Donor 30k rows, 8 predictors, predicting onto a 75k pool:

| Targets (stage analogue) | Peak RSS | Wall |
|---|---|---|
| 1 (SIPP tips, ACS rent) | 1.1 GB | 15 s |
| 5 (CPS-ORG labor) | 3.6 GB | 83 s |
| 27 (SCF wealth) | **~15 GB** | **> 30 min** |

Cost grows **super-linearly in chained targets** — each target joins the
predictor set for the next, and the quantile forests retain training samples for
quantile prediction. The SCF wealth stage is the single most demanding step of a
build, on both memory and wall-clock, and it runs on **one core** (it does not
parallelize across the machine).

### Calibration — `populace.calibrate.calibrate`

County-style sparse system (per-county count + income-sum control totals plus
national anchors), `epochs=256`, `max_weight_ratio=10`:

| Pool × targets | Matrix | Peak RSS | Wall | Within 10% |
|---|---|---|---|---|
| 75k × 3,704 | sparse CSR | 4.5 GB | 5 s | 100% |
| 300k × 3,704 | sparse CSR | 17 GB | 20 s | 100% |
| 300k × 6,288 | sparse CSR | 29 GB | 23 s | 100% |
| 75k × 3,704 (forced dense) | dense | 5.6 GB | 8 s | 100% |
| 150k × 3,704 (L0 prune) | sparse CSR | 8.7 GB | 52 s | 91% |

The solver is sparse and fast. But peak RAM is set by the **compile** step, not
the solve: `build_constraint_matrix` materializes one dense row per target into a
Python list and `np.vstack`s them before compressing to CSR, so

> **peak RAM ≈ 2 × n_targets × n_records × 8 bytes** (the row list + the vstack copy)

which the table matches (300k × 6,288 → 2·6288·3e5·8 ≈ 28 GB). The forced-dense
run's +1 GB over sparse confirms the solver docstring's own estimate (a
3,704 × 75k dense float32 tensor ≈ 1.1 GB). The L0 generate-big-then-prune path
costs **~5–6× the wall time** of a single solve (it searches the penalty across
several full optimizations) and trades a few points of target fit for the record
budget.

## Where the cliffs are

- **Calibration past ~300k records locally.** Because compile RAM is
  `2 × targets × records × 8 B`, a full-county (6,288-target) calibration extrapolates
  to **~94 GB at 1M records** and **~280 GB at 3M** — the charter's generate-big
  rungs (300k → 3M → 30M) are **not feasible on any laptop** with today's
  compiler. They belong on a large cloud box, *or* the compiler should build the
  CSR incrementally (see [Follow-up](#follow-up)). Local builds realistically
  mean a pool of **≤300k records**, which 48 GB handles with the full county
  target set.
- **Dense calibration at scale.** Forcing the dense path is fine nationally
  (+1 GB) but fatal at 3M records (~44 GB just for the matrix tensor); the
  automatic sparse switch (`_SPARSE_MIN_CELLS`, `_SPARSE_DENSITY_CUTOFF`) keeps
  this from biting in normal use.
- **SCF imputation memory.** ~15 GB for the 27-target stage is independent of the
  calibration ceiling and lands on a single core; it is the reason 16 GB cannot
  run a build even though no calibration is running yet.

Since the phases run **sequentially**, the end-to-end build peak is the *max* of
the stage peaks (~29 GB at 300k × 6,288), not their sum — which is why 48 GB is
the comfortable target rather than something larger.

## Hardware notes

- **RAM (48 GB).** Clears the 29 GB full-county calibration peak and the 15 GB
  SCF fit with headroom, and never swaps during a ≤300k build. 32 GB is a usable
  floor (it covers everything except full-county calibration, which swaps); 16 GB
  is consumer-only. Unified memory is soldered on Apple Silicon — this is the one
  spec that cannot be fixed later, so it is where to spend.
- **Chip (M-Pro or M-Max).** The wall-clock bottleneck (SCF QRF) is
  single-threaded, so high **single-core** performance matters more than core
  count; the Pro/Max tiers also bring the memory bandwidth the calibration SpMM
  uses (torch took 6 threads here). The base-M **Air is fanless and will thermal-
  throttle** on the 30-minute SCF fit — the Pro chassis's active cooling is worth
  it for anyone running builds, not just tests.
- **Storage (1 TB).** 14 GB of source data + caches + build outputs and multiple
  dataset vintages; 512 GB works, 1 TB is comfortable.
- **Cloud for the asymptote.** Generate-big beyond ~300k records is a cloud
  workload by design; a laptop is for development and small/matched-N builds.

## Follow-up

The calibration compile densifying to `n_targets × n_records` before compressing
to sparse is the one avoidable ceiling here: building the CSR incrementally
(accumulate `data`/`indices`/`indptr` per row, or `scipy.sparse.vstack` of
per-row sparse rows) would cut compile RAM by ~`n_records×` and move the 1M–3M
rungs within reach of a large box. Tracked as a `populace-calibrate` improvement;
the existing sparse-path tests cover the equivalence this must preserve.

## Appendix: reproduction

Environment certificate: Apple M5 Max, 128 GB, macOS 26.5.1 (arm64); Python
3.14.4; torch 2.12.0, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, scikit-learn
1.8.0. Measure peak RSS with `ru_maxrss` (bytes on macOS, KB on Linux), one
config per process.

**Disk / dev loop:**

```bash
du -sh .venv                              # core venv
du -sh <survey-source-cache>/storage      # external ASEC/PUF/SCF/ACS h5 inputs (~14 GB)
/usr/bin/time -l .venv/bin/python -m pytest -q                # full suite peak RSS
for p in packages/*/; do /usr/bin/time -l .venv/bin/python -m pytest "$p" -q; done
```

**Calibration** — drives the real `calibrate()` over a county-style sparse
system; `MODE` ∈ `sparse|dense|l0`:

```python
# bench_calibrate.py  N_HOUSEHOLDS N_TARGETS MODE [EPOCHS]
import json, resource, sys, time
import numpy as np, pandas as pd
from populace.calibrate import Target, TargetSet, calibrate
from populace.calibrate import solve as _solve
from populace.frame import EntitySchema, Frame, WeightKind, Weights

def peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / 1024**3

n_hh, n_t, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 256
n_cty = max(1, n_t // 2)
if mode == "dense":
    _solve._SPARSE_MIN_CELLS = 10**18
rng = np.random.default_rng(0)
income = rng.lognormal(10.7, 0.8, n_hh); county = rng.integers(0, n_cty, n_hh)
ids = np.arange(n_hh, dtype="int64")
frame = Frame(
    {"person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
     "household": pd.DataFrame({"household_id": ids, "income": income, "county": county})},
    EntitySchema(group_entities=("household",)),
    {"household": Weights(values=np.full(n_hh, 150.0), kind=WeightKind.DESIGN)})
pop, inc = n_hh * 150.0, float(income.sum()) * 150.0
mask = lambda c: (lambda f, c=c: f.table("household")["county"].to_numpy() == c)
ts = [Target("pop", "household", aggregation="count", value=pop * 1.05),
      Target("income", "household", measure="income", aggregation="sum", value=inc * 1.05)]
for c in np.unique(county):
    if len(ts) >= n_t: break
    ts.append(Target(f"pop_c{c}", "household", aggregation="count",
                     value=pop / n_cty * 1.05, filter=mask(c)))
    if len(ts) >= n_t: break
    ts.append(Target(f"inc_c{c}", "household", measure="income", aggregation="sum",
                     value=inc / n_cty * 1.05, filter=mask(c)))
kw = dict(epochs=epochs, max_weight_ratio=10.0, seed=0)
if mode == "l0": kw.update(target_records=int(n_hh * 0.6), budget_iters=6)
t = time.time(); r = calibrate(frame, TargetSet(ts), **kw); dt = time.time() - t
print(json.dumps({"pool": n_hh, "targets": r.problem.n_targets, "mode": mode,
                  "format": r.options["matrix_format"], "wall_s": round(dt, 1),
                  "peak_gb": round(peak_gb(), 2), "within10": round(r.fraction_within_10pct, 3)}))
```

**Imputation** — drives the real `RegimeGatedQRF` (donor → fit → predict):

```python
# bench_fit.py  N_DONOR N_RECEIVER N_TARGETS [N_PREDICTORS]
import json, resource, sys, time
import numpy as np, pandas as pd
from populace.fit import RegimeGatedQRF
from populace.frame import EntitySchema, Frame, WeightKind, Weights

def peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / 1024**3

nd, nr, nt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
npred = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rng = np.random.default_rng(0)
preds = [f"x{i}" for i in range(npred)]; tgts = [f"y{j}" for j in range(nt)]
X = rng.normal(size=(nd, npred)); Y = X @ rng.normal(size=(npred, nt)) + rng.normal(scale=0.3, size=(nd, nt))
dp = pd.DataFrame({p: X[:, i] for i, p in enumerate(preds)})
for j, tt in enumerate(tgts): dp[tt] = Y[:, j]
dp.insert(0, "person_id", np.arange(nd, dtype=np.int64)); dp["person_record_id"] = dp["person_id"]
donor = Frame({"person": dp, "record": pd.DataFrame({"record_id": dp["person_id"]})},
              EntitySchema(group_entities=("record",)),
              {"person": Weights(values=rng.uniform(50, 5000, nd), kind=WeightKind.DESIGN)})
recv = pd.DataFrame({p: rng.normal(size=nr) for p in preds})
t = time.time(); fit = RegimeGatedQRF(seed=0).fit(donor, preds, tgts); tf = time.time() - t
t = time.time(); _ = fit.predict(recv); tp = time.time() - t
print(json.dumps({"donor": nd, "recv": nr, "targets": nt, "fit_s": round(tf, 1),
                  "predict_s": round(tp, 1), "peak_gb": round(peak_gb(), 2)}))
```
