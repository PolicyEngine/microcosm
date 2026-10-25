"""Compare a 100-household exact-k set with PolicyBench's sampled 100.

Evidence record only: nothing here runs PolicyBench, evaluates a model,
publishes, uploads or notifies. Every number in this directory's README is
produced by this script and bundled in one of its JSON outputs.

Run from the repository root using the existing environment::

    OMP_NUM_THREADS=4 .venv/bin/python experiments/policybench-exact-k-100-vs-sampled-20260922/compare_designs.py run

Atomic aggregate JSON receipts are written after each design and sampling
seed; progress.log records progress alongside the redirected nohup run.log.
Household arrays remain in memory. Only households.csv and household_ids.json
contain household rows, for the frozen sample and national exact-k support.
An interrupted run preserves completed aggregate results, but must recompute
upstream household arrays. No external cache or population-row pickle is made.

Use --smoke-households N --n-seeds 2 --out-dir PATH for a reduced pipeline
check. PATH must be inside this worktree, distinct from the record directory,
and contain a .gitignore with '*'. Smoke results are not evidence.

Inputs (all hash-verified before use):

* the public release ``populace-us-2024-5da5a95-20260611`` artifact
  ``populace_us_2024.h5`` (the file PolicyBench's frozen US run drew from);
* the labelled Chronicle consumer-facts feed (b571381);
* PolicyBench's frozen run directory (read-only: ``scenarios.csv`` and its
  ``.meta.json``).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = Path(__file__).resolve().parent
SCRIPT_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

RELEASE_ID = "populace-us-2024-5da5a95-20260611"
H5_SHA256 = "f32c2e5e9098bc6540724fdd5debf963af495da4c29b3a7a63fb53c2a4bb5a34"
DEFAULT_H5 = (
    Path.home()
    / ".cache/huggingface/hub/datasets--policyengine--populace-us/blobs"
    / H5_SHA256
)
FEED_SHA256 = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
FEED_ROW_COUNT = 39_158
DEFAULT_FEED = Path(
    "/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/chronicle_us_b571381/artifact"
)
DEFAULT_POLICYBENCH_RUN = Path(
    "/Users/maxghenis/PolicyEngine/policybench/paper/snapshot/20260501/runs/"
    "us_full_run_20260612_policyengine_4_16_1_populace"
)
PERIOD = 2024
K = 100

# Settings recorded in every output (see README "Settings").
SAMPLE_REFIT = {
    # Applied identically to both 100-household sets once their members are
    # fixed: the post-stratified PPS sample (design 2) and the exact-k refit
    # (design 3). max_weight_ratio is relative to each set's starting weights
    # (sum of eligible design weights / n for the sample; the normalized
    # Horvitz-Thompson w/q baseline for exact-k).
    "method": "adam",
    "mass": "free",
    "max_weight_ratio": 10.0,
    "epochs": 1024,
    "learning_rate": 0.02,
    "seed": 0,
}
EXACT_K_SELECTION = {
    # L0 selection over the full household file. The survivors must carry
    # roughly 750x their design weights, so the hard per-record bound is off.
    # The budget measure is the maintained default, the count of records whose
    # returned weight survives pruning. The draw's certainty threshold is
    # passed to select_exact_k directly
    # (calibrate accepts feasible_draw_pi_hi only with the mass basis).
    "method": "adam",
    "mass": "conserve",
    "max_weight_ratio": None,
    "target_records": K,
    "budget_basis": "nonzero_count",
    "epochs": 1024,
    "learning_rate": 0.02,
    "init_mean": 0.5,
    "temperature": 0.25,
    "budget_iters": 10,
    "seed": 0,
}
EXACT_K_DRAW_PI_HI = 0.95
EXACT_K_DRAW_SEED = 0
SAMPLING_DISTRIBUTION = {"n_seeds": 200, "seed_generator_seed": 20260922}
HEADLINE_MAX = 40
DEFAULT_N_BATCHES = 6
INVARIANCE_WINDOW = 200  # households each side of the batch-0/1 boundary
TARGET_LOSS_CAP = 10.0  # microcosm.calibrate's default cap (solve.py _DEFAULT_TARGET_LOSS_CAP)

#: PolicyBench's ``MONETARY_INCOME_FIELDS`` (policybench/scenarios.py:187-221
#: at c7aaffe), the person fields ``Scenario.total_income`` sums.
POLICYBENCH_MONETARY_INCOME_FIELDS = (
    "alimony_income", "child_support_received", "disability_benefits",
    "employment_income", "estate_income", "farm_income", "farm_operations_income",
    "farm_rent_income", "miscellaneous_income", "non_qualified_dividend_income",
    "partnership_s_corp_income", "partnership_se_income", "qualified_dividend_income",
    "rental_income", "salt_refund_income", "self_employment_income",
    "short_term_capital_gains", "social_security_dependents", "social_security_disability",
    "social_security_retirement", "social_security_survivors", "ssi_reported",
    "tax_exempt_interest_income", "taxable_401k_distributions", "taxable_403b_distributions",
    "taxable_interest_income", "taxable_ira_distributions", "taxable_private_pension_income",
    "taxable_sep_distributions", "unemployment_compensation", "veterans_benefits",
    "workers_compensation", "long_term_capital_gains",
)
POLICYBENCH_FILING_STATUSES = {"SINGLE": "single", "JOINT": "joint", "HEAD_OF_HOUSEHOLD": "head_of_household"}


PROGRESS_LOG: Path | None = None
ACTIVE_CHILD: subprocess.Popen | None = None
MAX_RSS_BYTES = 31_000_000_000  # stop with headroom below the authorized 32 GB


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    if PROGRESS_LOG is not None:
        with PROGRESS_LOG.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()


def _check_disk() -> None:
    disk = Path("/System/Volumes/Data")
    if not disk.exists():
        disk = REPO_ROOT
    if shutil.disk_usage(disk).free < 10 * 1024**3:
        raise RuntimeError("Stopping writes: less than 10 GiB free")


def _peak_rss_gb_self() -> float:
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw * 1024 / 1e9


class ResourceMeter:
    """Wall time and sampled peak RSS (this process plus live children)."""

    def __init__(self, interval: float = 0.25, label: str | None = None) -> None:
        self.interval = interval
        self.label = label
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self) -> int:
        import psutil

        proc = psutil.Process()
        total = proc.memory_info().rss
        if ACTIVE_CHILD is not None:
            try:
                total += psutil.Process(ACTIVE_CHILD.pid).memory_info().rss
            except psutil.Error:
                pass
        return total

    def _run(self) -> None:
        next_heartbeat = time.time() + 45
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._sample())
                if self.peak >= MAX_RSS_BYTES:
                    _log("Stopping: sampled process-tree RSS reached 31 GB")
                    if ACTIVE_CHILD is not None:
                        ACTIVE_CHILD.kill()
                    os._exit(137)
                if self.label and time.time() >= next_heartbeat:
                    _log(f"{self.label} alive: elapsed {time.time() - self.t0:.0f}s; sampled peak RSS {self.peak / 1e9:.2f} GB")
                    next_heartbeat = time.time() + 45
            except Exception:  # pragma: no cover - diagnostic only
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> ResourceMeter:
        self.t0 = time.time()
        self.peak = self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._sample())
        self.seconds = time.time() - self.t0

    def record(self) -> dict:
        return {"seconds": round(self.seconds, 2), "peak_rss_gb": round(self.peak / 1e9, 3)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: object) -> None:
    _check_disk()
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _module_sha256(module: str) -> str:
    import importlib  # noqa: PLC0415

    return _sha256(Path(importlib.import_module(module).__file__))


def _digest_of(payload: object) -> str:
    return hashlib.sha256(json.dumps(_clean(payload), sort_keys=True).encode()).hexdigest()[:16]


def source_provenance() -> dict:
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "script_sha256": SCRIPT_SHA256_AT_IMPORT,
        "modules_sha256": {name: _module_sha256(name) for name in (
            "microcosm.calibrate.solve", "microcosm.calibrate.gates",
            "microcosm.calibrate.exact_k", "microcosm.calibrate.matrix",
        )},
    }


class Checkpoint:
    """Durable aggregate receipts only; household arrays are never cached.

    Completed design summaries survive interruption. Recomputing upstream
    materialization is necessary after a crash because row caches are forbidden.
    """

    def __init__(self, root: Path, fingerprint: dict) -> None:
        self.dir = root
        _write_json(self.dir / "run_fingerprint.json", fingerprint)

    def load(self, name: str):
        return None

    def save(self, name: str, payload: object) -> None:
        # Some stage payloads contain full-population rows. Keep them in memory.
        pass

    def save_summary(self, name: str, payload: object) -> None:
        """Aggregate-only JSON, readable while the run goes on."""
        _write_json(self.dir / f"{name}.json", payload)


def _tool_module():
    """Import the maintained US release builder (tools/ is not a package)."""
    tools_dir = str(REPO_ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import build_us_fiscal_refresh_release as tool  # noqa: PLC0415

    return tool


def _version(package: str) -> str:
    from importlib.metadata import version

    try:
        return version(package)
    except Exception:  # pragma: no cover - diagnostic only
        return "unavailable"


# ---------------------------------------------------------------------------
# Frame loading (maintained loader) and the formula-owned-column adaptation
# ---------------------------------------------------------------------------


def load_measurement_frame(h5: Path, household_ids=None):
    """Load the release with the maintained legacy loader, minus engine outputs.

    ``load_legacy_calibrated_us_h5`` loads the file as-is. The maintained
    target materializer then refuses it (``_assert_no_formula_owned_columns``),
    because the June 11 file carries columns that the installed PolicyEngine-US
    computes by formula. Those columns are dropped here so the engine computes
    them from the file's leaf inputs; their stored-versus-engine aggregate
    deltas are recorded in ``inputs.json``. ``household_ids`` optionally
    restricts both frames to those households (smoke runs and batches).
    """
    from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
    from microcosm.frame import Frame
    from microcosm.frame.units import US_SCHEMA

    tool = _tool_module()
    frame = load_legacy_calibrated_us_h5(h5)
    if household_ids is not None:
        person = frame.table("person")
        frame = frame.select(np.isin(person["person_household_id"].to_numpy(), household_ids))
    tables = {entity: frame.table(entity) for entity in frame.entities}
    adapter = tool._formula_owned_gate_adapter()
    formula_owned = sorted(adapter._engine_computed_columns(tables, period=tool.PERIOD))
    dropped: dict[str, list[str]] = {}
    new_tables = {}
    for entity, table in tables.items():
        cols = [c for c in formula_owned if c in table.columns]
        if cols:
            dropped[entity] = cols
        new_tables[entity] = table.drop(columns=cols).copy()
    measured = Frame(new_tables, US_SCHEMA, {"household": frame.weights_for("household")})
    return frame, measured, dropped


def _entity_weights(frame, entity: str) -> np.ndarray:
    """Household weight carried to each row of ``entity`` (via persons)."""
    household = frame.table("household")
    hh_weight = pd.Series(frame.weights_for("household").values, index=household["household_id"].to_numpy())
    person = frame.table("person")
    person_w = hh_weight.loc[person["person_household_id"].to_numpy()].to_numpy()
    if entity == "person":
        return person_w
    if entity == "household":
        return frame.weights_for("household").values
    membership = person[f"person_{entity}_id"].to_numpy()
    first = pd.Series(person_w).groupby(membership).first()
    ids = frame.table(entity)[f"{entity}_id"].to_numpy()
    return first.loc[ids].to_numpy()


# ---------------------------------------------------------------------------
# Child processes (spawned one at a time; results return through a pipe)
# ---------------------------------------------------------------------------


#: Pipeline-test switch (``--pipeline-test-synthetic-measures``, smoke only):
#: replaces the engine materializer with deterministic synthetic household
#: values so the calibration and report code can be exercised quickly. Its
#: output is a code check, never evidence, and is refused inside the repo.
SYNTHETIC_MEASURES_FOR_PIPELINE_TEST = False


def _synthetic_materialize(household_ids, specs) -> dict:
    columns = sorted({s.measure for s in specs if s.measure} | {s.filter for s in specs if getattr(s, "filter", None)})
    values = np.zeros((len(household_ids), len(columns)))
    for i, hid in enumerate(household_ids):
        rng = np.random.default_rng(int(hid))
        values[i] = rng.lognormal(8.0, 1.5, len(columns)) * (rng.random(len(columns)) < 0.3)
    return {
        "household_id": np.asarray(household_ids),
        "columns": {c: values[:, j] for j, c in enumerate(columns)},
        "compiled_names": sorted(s.name for s in specs),
        "dropped_target_names": [],
        "households": len(household_ids),
        "persons": 0,
        "seconds": 0.0,
        "child_peak_rss_gb": 0.0,
    }


def _run_child(target, *args):
    global ACTIVE_CHILD
    if SYNTHETIC_MEASURES_FOR_PIPELINE_TEST and target is _child_materialize:
        return _synthetic_materialize(args[1], args[2])
    from multiprocessing.connection import Connection

    # subprocess avoids multiprocessing.spawn's extra resource-tracker process.
    read_fd, write_fd = os.pipe()
    receiver = Connection(read_fd, readable=True, writable=False)
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_engine_child", target.__name__, str(write_fd)],
        stdin=subprocess.PIPE,
        pass_fds=(write_fd,),
    )
    ACTIVE_CHILD = proc
    os.close(write_fd)
    pickle.dump(args, proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.close()
    try:
        payload = receiver.recv()
    except EOFError:
        payload = None
    finally:
        receiver.close()
    proc.wait()
    ACTIVE_CHILD = None
    if proc.returncode != 0 or payload is None:
        raise RuntimeError(f"child {target.__name__} failed with exit code {proc.returncode}")
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"child {target.__name__} raised:\n{payload['error']}")
    return payload


def _child_engine_facts(conn, h5: str, household_ids) -> None:
    """Formula-owned column deltas and filing status, both frames."""
    import traceback

    try:
        from policyengine_us import Microsimulation

        from microcosm.build.us_runtime.engine_lifecycle import (
            release_engine_simulation,
        )

        tool = _tool_module()
        t0 = time.time()
        original, measured, dropped = load_measurement_frame(Path(h5), household_ids)
        sim = Microsimulation(dataset=tool._dataset_from_frame(measured))
        deltas = []
        for entity, columns in sorted(dropped.items()):
            weights = _entity_weights(original, entity)
            stored_table = original.table(entity)
            for column in columns:
                stored = stored_table[column].to_numpy()
                try:
                    engine = np.asarray(sim.calculate(column, PERIOD))
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    deltas.append({"entity": entity, "column": column, "error": repr(exc)[:300]})
                    continue
                if engine.shape != stored.shape:
                    deltas.append({"entity": entity, "column": column, "note": "shape mismatch"})
                    continue
                try:
                    stored_f = stored.astype(np.float64)
                    engine_f = engine.astype(np.float64)
                except (TypeError, ValueError):
                    deltas.append(
                        {
                            "entity": entity,
                            "column": column,
                            "kind": "categorical",
                            "share_rows_equal": float(np.mean(stored.astype(str) == engine.astype(str))),
                        }
                    )
                    continue
                s_total = float(np.dot(stored_f, weights))
                e_total = float(np.dot(engine_f, weights))
                deltas.append(
                    {
                        "entity": entity,
                        "column": column,
                        "stored_weighted_total": s_total,
                        "engine_weighted_total": e_total,
                        "relative_delta": None if s_total == 0 else (e_total - s_total) / abs(s_total),
                        "share_rows_equal": float(np.mean(np.isclose(stored_f, engine_f))),
                    }
                )
        fs_measured = tool._filing_status_names(np.asarray(sim.calculate("filing_status", PERIOD)))
        release_engine_simulation(sim)
        del sim
        # The file as shipped (stored columns kept), as PolicyBench's runtime saw it.
        sim = Microsimulation(
            dataset=tool._dataset_from_frame(original, assert_no_formula_owned_columns=False)
        )
        fs_original = tool._filing_status_names(np.asarray(sim.calculate("filing_status", PERIOD)))
        release_engine_simulation(sim)
        del sim
        conn.send(
            {
                "dropped": dropped,
                "deltas": deltas,
                "tax_unit_ids": original.table("tax_unit")["tax_unit_id"].to_numpy(),
                "filing_status_measured": np.asarray(fs_measured, dtype=str),
                "filing_status_as_shipped": np.asarray(fs_original, dtype=str),
                "seconds": time.time() - t0,
                "child_peak_rss_gb": _peak_rss_gb_self(),
            }
        )
    except Exception:  # noqa: BLE001
        conn.send({"error": traceback.format_exc()})
    finally:
        conn.close()


def _child_materialize(conn, h5: str, household_ids, specs) -> None:
    """Run the maintained materializer on one household batch."""
    import traceback

    try:
        tool = _tool_module()
        t0 = time.time()
        _original, frame, _dropped = load_measurement_frame(Path(h5), household_ids)
        del _original
        target_frame, compiled, compilation = tool._materialize_target_frame(
            frame, specs, gate_congressional_district_targets=False
        )
        columns = sorted(
            {spec.measure for spec in compiled.specs if spec.measure}
            | {spec.filter for spec in compiled.specs if getattr(spec, "filter", None)}
        )
        hh = target_frame.table("household")
        conn.send(
            {
                "household_id": hh["household_id"].to_numpy(),
                "columns": {c: hh[c].to_numpy(dtype=np.float64) for c in columns},
                "compiled_names": sorted(spec.name for spec in compiled.specs),
                "dropped_target_names": list(compilation.get("dropped_target_names", [])),
                "households": int(frame.n("household")),
                "persons": int(frame.n("person")),
                "seconds": time.time() - t0,
                "child_peak_rss_gb": _peak_rss_gb_self(),
            }
        )
    except Exception:  # noqa: BLE001
        conn.send({"error": traceback.format_exc()})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def compile_national_specs(feed_dir: Path):
    """Maintained registry compile, then keep national-scope targets only."""
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime.congressional_district_vintage import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )
    from microcosm.build.us_runtime.medicaid_take_up import (
        apply_us_medicaid_enrollment_substitutions,
    )

    artifact = load_ledger_consumer_artifact(feed_dir, expected_facts_sha256=FEED_SHA256)
    if len(artifact.facts) != FEED_ROW_COUNT:
        raise RuntimeError(f"feed has {len(artifact.facts)} rows, expected {FEED_ROW_COUNT}")
    crosswalk_path = default_congressional_district_vintage_crosswalk_path()
    # The same call and substitution tools/build_us_fiscal_refresh_release.py
    # main() makes (age_targets defaults True; unaged dollar targets refused).
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=load_congressional_district_vintage_crosswalk(crosswalk_path),
        age_targets=True,
        allow_unaged_dollar_targets=False,
    )
    registry, substitutions = apply_us_medicaid_enrollment_substitutions(registry)
    levels = Counter(
        (str(spec.metadata.get("ledger_geography_level")), str(spec.metadata.get("geography_scope")))
        for spec in registry.specs
    )
    national = tuple(
        spec
        for spec in registry.specs
        if spec.metadata.get("ledger_geography_level") == "country"
        or spec.metadata.get("geography_scope") == "national"
    )
    receipt = {
        "feed_dir": str(feed_dir),
        "feed_facts_sha256": artifact.facts_sha256,
        "feed_manifest_sha256": artifact.manifest_sha256,
        "feed_rows": len(artifact.facts),
        "crosswalk": str(crosswalk_path.name),
        "crosswalk_sha256": _sha256(crosswalk_path),
        "age_targets": True,
        "allow_unaged_dollar_targets": False,
        "medicaid_enrollment_substitutions": len(substitutions),
        "compiled_specs_all_geographies": len(registry.specs),
        "specs_by_geography_level_and_scope": {
            f"{level}|{scope}": count for (level, scope), count in sorted(levels.items())
        },
        "national_rule": (
            "metadata.ledger_geography_level == 'country' or metadata.geography_scope == 'national'"
        ),
        "national_specs_declared": len(national),
        "national_families_declared": dict(sorted(Counter(spec.family for spec in national).items())),
    }
    return national, receipt


#: Labelled-feed filter keys that restate a compiled constraint the maintained
#: materializer already applies, with that compiled counterpart. The recorded
#: checkout's ``_unsupported_ledger_filter_metadata`` refuses these keys.
#: The age restatement is the same rule applied
#: to ``age_lower_bound`` / ``age_upper_bound``, which the ``population_age``
#: materializer and the age-banded ``policyengine_variable`` path both read.
RESTATED_BOUND_KEYS = {
    "ledger_filter_us:statutes/26/62#adjusted_gross_income_lower_bound": "agi_lower_bound",
    "ledger_filter_us:statutes/26/62#adjusted_gross_income_upper_bound": "agi_upper_bound",
    "ledger_filter_age_lower_bound": "age_lower_bound",
    "ledger_filter_age_upper_bound": "age_upper_bound",
}
RESTATED_EITC_CHILD_LOWER = "ledger_filter_us.tax.earned_income_credit_qualifying_children_lower_bound"


def strip_verified_restatements(specs):
    """Drop labelled filter keys only where they restate an applied constraint.

    For every key the maintained guard refuses, accept it (remove it from the
    spec's metadata) only if it equals its compiled counterpart: numerically
    equal bounds for AGI and age, and the same qualifying-child population over
    counts 0..16 for the EITC lower bound. Any
    other refused key leaves the spec refused; such specs are excluded and
    named in the receipt.
    """
    import dataclasses

    tool = _tool_module()
    refused = tool._unsupported_ledger_filter_metadata(specs)
    kept, excluded = [], {}
    accepted_keys: Counter = Counter()
    for spec in specs:
        keys = refused.get(spec.name)
        if not keys:
            kept.append(spec)
            continue
        metadata = dict(spec.metadata)
        problems = []
        for key in keys:
            value = metadata[key]
            if key in RESTATED_BOUND_KEYS:
                compiled_key = RESTATED_BOUND_KEYS[key]
                compiled = metadata.get(compiled_key)
                if compiled is None or tool._as_bound(str(value).strip()) != tool._as_bound(str(compiled).strip()):
                    problems.append(f"{key}={value} vs {compiled_key}={compiled}")
            elif key == RESTATED_EITC_CHILD_LOWER:
                compiled = tool._soi_eitc_child_count_filter(metadata)
                counts = np.arange(17, dtype=np.float64)
                try:
                    same = compiled is not None and np.array_equal(
                        counts >= float(value), tool._eitc_child_count_mask(counts, compiled)
                    )
                except ValueError:
                    same = False
                if not same:
                    problems.append(f"{key}={value} vs child-count filter {compiled}")
            else:
                problems.append(f"{key}={value} has no compiled counterpart")
        if problems:
            excluded[spec.name] = problems
            continue
        for key in keys:
            metadata.pop(key)
            accepted_keys[key] += 1
        kept.append(dataclasses.replace(spec, metadata=metadata))
    if tool._unsupported_ledger_filter_metadata(kept):
        raise RuntimeError("restatement stripping left refused keys")
    receipt = {
        "refused_by_origin_main_guard": len(refused),
        "accepted_as_verified_restatements": len(refused) - len(excluded),
        "accepted_keys": dict(sorted(accepted_keys.items())),
        "excluded_specs": excluded,
    }
    return tuple(kept), receipt


HEADLINE_RULES = [
    ("population by age band (Census PEP)", "census_pep.cy2024.national_resident_population_age."),
    ("wages and salaries (BEA NIPA)", "bea_nipa.cy2024.total_wages_salaries."),
    ("wages and salaries (CBO)", "cbo.revenue_projection.ty2024.income_by_source.wages_and_salaries."),
    ("SNAP benefits (USDA)", "usda_snap.fy2024.national_benefits.national_total.total_benefits"),
    ("SNAP average monthly households (USDA)", "usda_snap.fy2024.national_average_monthly_households."),
    ("Medicaid enrollment (CMS, Dec 2024)", "cms_medicaid.month2024_12.state_enrollment.us.total_medicaid_enrollment"),
    ("SSI payments (SSA)", "ssa_supplement.cy2024.oasdi_ssi_payments.ssi_payments."),
    ("SSI recipients by age (SSA)", "ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients."),
    ("EITC amount and returns (IRS SOI TY2024 filing season)", "irs_soi.ty2024.filing_season_week47.eitc_all_returns."),
    ("AGI and return count (IRS SOI Table 1.1)", "irs_soi.ty2023.table_1_1.all."),
    ("AGI (CBO)", "cbo.revenue_projection.ty2024.income_by_source.adjusted_gross_income."),
]


def headline_names(names: list[str]) -> tuple[list[str], dict[str, int]]:
    """The <=40-target headline subset, chosen by name prefix (HEADLINE_RULES)."""
    chosen: list[str] = []
    per_rule: dict[str, int] = {}
    for label, prefix in HEADLINE_RULES:
        hits = sorted(n for n in names if n.startswith(prefix))
        per_rule[label] = len(hits)
        chosen.extend(hits)
    if len(chosen) > HEADLINE_MAX:
        raise RuntimeError(f"headline subset has {len(chosen)} > {HEADLINE_MAX}")
    return chosen, per_rule


def _national_agi_band_rows(specs) -> list[str]:
    """Report AGI measures with a finite bound; unbounded totals are not bands."""
    return sorted(
        s.name for s in specs
        if any("adjusted_gross_income" in str(s.metadata.get(key, "")) for key in ("base_variable", "source_variable", "variable"))
        and any(
            math.isfinite(float(s.metadata[key]))
            for key in ("agi_lower_bound", "agi_upper_bound")
            if s.metadata.get(key) is not None
        )
    )


# ---------------------------------------------------------------------------
# PolicyBench draw (reimplemented read-only from policybench/scenarios.py)
# ---------------------------------------------------------------------------


def policybench_eligible(desc: pd.DataFrame, filing_status_column: str) -> pd.DataFrame:
    """``policybench/scenarios.py::_eligible_households`` on this file.

    One tax unit, one SPM unit, one family, at least one adult (age >= 18),
    and a first-person filing status in SINGLE / JOINT / HEAD_OF_HOUSEHOLD.
    Households sort by id (pandas groupby order), as in PolicyBench.
    """
    keep = (
        (desc["tax_units"] == 1)
        & (desc["spm_units"] == 1)
        & (desc["families"] == 1)
        & (desc["adults"] >= 1)
        & desc[filing_status_column].isin(list(POLICYBENCH_FILING_STATUSES))
    )
    return desc.loc[keep].sort_values("household_id").reset_index(drop=True)


def policybench_draw(eligible: pd.DataFrame, *, seed: int, requested: int, private_fraction: float, split_seed: int) -> list[int]:
    """PPS-without-replacement draw plus the id-hash public split.

    ``_sample_household_ids``: ``numpy.random.default_rng(seed).choice(ids,
    size=requested, replace=False, p=w / w.sum())``; scenarios are numbered
    ``scenario_000`` ... in draw order; ``split_scenarios`` ranks ids by
    ``sha256(f"{split_seed}:{scenario_id}")`` and holds the first
    ``round(requested * private_fraction)`` private. Returns the public
    household ids in draw order.
    """
    rng = np.random.default_rng(seed)
    ids = eligible["household_id"].to_numpy()
    weights = eligible["design_weight"].to_numpy(dtype=float)
    weights = np.where(weights > 0, weights, 0.0)
    drawn = rng.choice(ids, size=requested, replace=False, p=weights / weights.sum())
    scenario_ids = [f"scenario_{i:03d}" for i in range(requested)]
    private_count = round(requested * private_fraction)
    ranked = sorted(scenario_ids, key=lambda sid: hashlib.sha256(f"{split_seed}:{sid}".encode()).hexdigest())
    private = set(ranked[:private_count])
    return [int(h) for sid, h in zip(scenario_ids, drawn, strict=True) if sid not in private]


def load_frozen_policybench(run_dir: Path):
    meta = json.loads((run_dir / "scenarios.csv.meta.json").read_text())
    scenarios = pd.read_csv(run_dir / "scenarios.csv")
    rows = []
    for _, row in scenarios.iterrows():
        payload = json.loads(row["scenario_json"])
        people = payload.get("adults", []) + payload.get("children", [])
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "household_id": int(payload["metadata"]["household_id"]),
                "state": row["state"],
                "filing_status": row["filing_status"],
                "num_adults": int(row["num_adults"]),
                "num_children": int(row["num_children"]),
                "total_income": float(row["total_income"]),
                "employment_income_sum": float(sum(float(p.get("employment_income", 0.0)) for p in people)),
            }
        )
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def subset_frame(frame, household_ids, weights: np.ndarray):
    """Select households by id and set their starting weights (id order)."""
    from microcosm.frame import MassChange, WeightKind

    person = frame.table("person")
    mask = np.isin(person["person_household_id"].to_numpy(), household_ids)
    sub = frame.select(mask)
    sub_ids = sub.table("household")["household_id"].to_numpy()
    order = {int(h): i for i, h in enumerate(household_ids)}
    values = np.asarray([weights[order[int(h)]] for h in sub_ids], dtype=np.float64)
    current = sub.resolve_weights("household")
    return sub.with_weights(
        "household",
        current.with_values(values, kind=WeightKind.CALIBRATED),
        mass=MassChange(
            factor=float(values.sum() / current.total),
            reason="design weights of a 100-household sample (sum of eligible design weights / n)",
        ),
    )


def fit_summary(names, estimates, targets) -> dict:
    """Relative-error fit on the calibrator's own scale, s = max(|b|, 1)."""
    from microcosm.calibrate import default_target_loss_scales

    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    rel = (estimates - targets) / default_target_loss_scales(targets)
    abs_rel = np.abs(rel)
    nonzero = np.abs(targets) > 0
    order = np.argsort(-abs_rel, kind="stable")
    worst = order[:5]
    worst_nonzero = [i for i in order if nonzero[i]][:5]
    return {
        "n_targets": int(len(targets)),
        "n_zero_valued_targets": int((~nonzero).sum()),
        "capped_mape": float(np.minimum(abs_rel, TARGET_LOSS_CAP).mean()),
        "uncapped_mape": float(abs_rel.mean()),
        "median_abs_rel_error": float(np.median(abs_rel)),
        "share_within_10pct": float(np.mean(abs_rel <= 0.10)),
        "share_within_25pct": float(np.mean(abs_rel <= 0.25)),
        "worst_five": [{"target": str(names[i]), "relative_error": float(rel[i])} for i in worst],
        "worst_five_nonzero_targets": [
            {"target": str(names[i]), "relative_error": float(rel[i])} for i in worst_nonzero
        ],
    }


def evaluate_weights(problem, weights) -> dict:
    names = [t.name for t in problem.targets]
    return fit_summary(names, problem.estimates(np.asarray(weights)), problem.target_vector)


def weight_concentration(weights, equal_share: float | None = None) -> dict:
    from microcosm.calibrate import effective_sample_size

    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0]
    median = float(np.median(w))
    out = {
        "n": int(len(w)),
        "total": float(w.sum()),
        "max_over_mean": float(w.max() / w.mean()),
        "min_over_mean": float(w.min() / w.mean()),
        "top10_share": float(np.sort(w)[::-1][:10].sum() / w.sum()),
        "kish_ess": float(effective_sample_size(w)),
        "n_above_5x_own_median": int(np.sum(w > 5 * median)),
        "n_above_10x_own_median": int(np.sum(w > 10 * median)),
    }
    if equal_share is not None:
        out["equal_share"] = float(equal_share)
        out["n_above_5x_equal_share"] = int(np.sum(w > 5 * equal_share))
        out["n_above_10x_equal_share"] = int(np.sum(w > 10 * equal_share))
    return out


def _scatter(n: int, positions, values) -> np.ndarray:
    out = np.zeros(n)
    out[positions] = values
    return out


def _quantiles(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _share_at_or_below(values, x) -> float:
    return float(np.mean(np.asarray(values) <= x))


def _weighted_quantiles(values, weights, qs) -> list[float]:
    order = np.argsort(values, kind="stable")
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cum = np.cumsum(w) / w.sum()
    return [float(v[min(np.searchsorted(cum, q, side="left"), len(v) - 1)]) for q in qs]


# ---------------------------------------------------------------------------
# Household descriptors
# ---------------------------------------------------------------------------

#: households.csv column -> (target measure, expected base variable(s), test)
DESCRIPTOR_MEASURES = {
    "employment_income": (
        "cbo.revenue_projection.ty2024.income_by_source.wages_and_salaries.projected_amount",
        "value",
    ),
    "agi_or_total_income": (
        "cbo.revenue_projection.ty2024.income_by_source.adjusted_gross_income.projected_amount",
        "value",
    ),
    "receives_snap": ("usda_snap.fy2024.national_benefits.national_total.total_benefits", "positive"),
    "receives_medicaid": ("cms_medicaid.month2024_12.state_enrollment.us.total_medicaid_enrollment", "positive"),
    "receives_ssi": ("ssa_supplement.cy2024.oasdi_ssi_payments.ssi_payments.payment_amount", "positive"),
    "eitc_positive": (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns.earned_income_credit.total_earned_income_credit_amount",
        "positive",
    ),
    "social_security_positive": (
        "ssa_supplement.cy2024.oasdi_ssi_payments.social_security_benefits.payment_amount",
        "positive",
    ),
}
DESCRIPTOR_METADATA_KEYS = (
    "materializer", "base_variable", "base_variables", "source_variable", "variable",
    "measure_mode", "indicator_map_to", "indicator_filter_variable", "filing_status",
    "agi_lower_bound", "agi_upper_bound", "ledger_domain",
)


def build_descriptors(original, filing_status: dict[str, np.ndarray]) -> pd.DataFrame:
    """Household descriptors from the file's own columns."""
    from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL

    person = original.table("person")
    household = original.table("household")
    hh_ids = household["household_id"].to_numpy()
    age = person["age"].to_numpy(dtype=np.float64)
    p_hh = person["person_household_id"].to_numpy()
    by = pd.DataFrame(
        {
            "household_id": p_hh,
            "tax_unit_id": person["person_tax_unit_id"].to_numpy(),
            "spm_unit_id": person["person_spm_unit_id"].to_numpy(),
            "family_id": person["person_family_id"].to_numpy(),
            "adult": age >= 18,
            "age": age,
            "is_household_head": person["is_household_head"].to_numpy(dtype=bool),
            "employment_income_stored": person["employment_income"].to_numpy(dtype=np.float64),
        }
    )
    present_fields = [f for f in POLICYBENCH_MONETARY_INCOME_FIELDS if f in person.columns]
    by["policybench_total_income_stored"] = (
        person[present_fields].to_numpy(dtype=np.float64).sum(axis=1) if present_fields else 0.0
    )
    for label, values in filing_status.items():
        by[label] = pd.Series(values["values"], index=values["tax_unit_ids"]).loc[by["tax_unit_id"].to_numpy()].to_numpy()
    grouped = by.groupby("household_id", sort=True)
    head_age = by[by["is_household_head"]].groupby("household_id")["age"].max()
    oldest = grouped["age"].max()
    desc = pd.DataFrame(
        {
            "household_size": grouped.size(),
            "adults": grouped["adult"].sum().astype(int),
            "tax_units": grouped["tax_unit_id"].nunique(),
            "spm_units": grouped["spm_unit_id"].nunique(),
            "families": grouped["family_id"].nunique(),
            "employment_income_stored": grouped["employment_income_stored"].sum(),
            "policybench_total_income_stored": grouped["policybench_total_income_stored"].sum(),
            **{f"{label}_first": grouped[label].first() for label in filing_status},
        }
    )
    desc["children"] = desc["household_size"] - desc["adults"]
    desc["head_age"] = head_age.reindex(desc.index).fillna(oldest)
    desc["head_age_from_head_flag"] = head_age.reindex(desc.index).notna()
    desc = desc.reindex(hh_ids)
    desc["state_fips"] = household["state_fips"].to_numpy()
    desc["state"] = [US_STATE_NUMERIC_FIPS_TO_POSTAL[int(s)] for s in desc["state_fips"]]
    desc["design_weight"] = original.weights_for("household").values
    desc.index.name = "household_id"
    return desc.reset_index(), present_fields


def _composition(table: pd.DataFrame, weights: np.ndarray, cuts: list[float]) -> dict:
    keep = np.asarray(weights) > 0
    t = table.loc[keep]
    w = np.asarray(weights, dtype=np.float64)[keep]
    total = w.sum()

    def share(mask) -> float:
        return float(w[np.asarray(mask)].sum() / total)

    size = t["household_size"].to_numpy()
    kids = t["children"].to_numpy()
    age = t["head_age"].to_numpy()
    bins = np.digitize(t["employment_income"].to_numpy(), cuts, right=True)  # 0..4 -> Q1..Q5
    return {
        "households": int(len(t)),
        "household_size": {"1": share(size == 1), "2": share(size == 2), "3": share(size == 3), "4+": share(size >= 4)},
        "children": {"0": share(kids == 0), "1": share(kids == 1), "2+": share(kids >= 2)},
        "head_age": {
            "<30": share(age < 30),
            "30-49": share((age >= 30) & (age < 50)),
            "50-64": share((age >= 50) & (age < 65)),
            "65+": share(age >= 65),
        },
        "employment_income_quintile": {f"Q{i + 1}": share(bins == i) for i in range(5)},
        "receipt": {
            col: share(t[col].to_numpy())
            for col in ("receives_snap", "receives_medicaid", "receives_ssi", "eitc_positive", "social_security_positive")
        },
        "distinct_states": int(t["state"].nunique()),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args) -> None:
    import torch

    torch.set_num_threads(4)
    from microcosm.calibrate import (
        TargetRegistry,
        assert_exact_k_support,
        build_constraint_matrix,
        calibrate,
        exact_k_design_feasibility,
        refit_l0_selection,
        select_exact_k,
    )
    from microcosm.frame import Frame
    from microcosm.frame.units import US_SCHEMA

    global SYNTHETIC_MEASURES_FOR_PIPELINE_TEST, PROGRESS_LOG
    smoke = args.smoke_households is not None
    if args.pipeline_test_synthetic_measures:
        if not smoke:
            raise SystemExit("--pipeline-test-synthetic-measures requires --smoke-households")
        SYNTHETIC_MEASURES_FOR_PIPELINE_TEST = True
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RECORD_DIR
    if REPO_ROOT not in out_dir.parents:
        raise SystemExit("Outputs must stay inside the assigned workspace")
    if smoke and (out_dir == RECORD_DIR or not (out_dir / ".gitignore").exists()):
        raise SystemExit("Smoke output requires a separate directory with a .gitignore containing '*'")
    if smoke and "*" not in (out_dir / ".gitignore").read_text().splitlines():
        raise SystemExit("Smoke output .gitignore must contain '*'")
    _check_disk()
    out_dir.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG = out_dir / "progress.log"
    selection_settings = dict(EXACT_K_SELECTION)
    refit_settings = dict(SAMPLE_REFIT)
    n_seeds = args.n_seeds
    if smoke:
        selection_settings.update(epochs=64, budget_iters=4)
        refit_settings.update(epochs=64)
        n_seeds = min(n_seeds, 3)
    _write_json(out_dir / "run_status.json", {"status": "running", "smoke": smoke, "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    runtime: dict[str, object] = {"notes": {
        "peak_rss_gb": "sampled every 0.25 s: this process plus any live child (psutil RSS sum)",
        "child_peak_rss_gb": "the child's own getrusage high-water mark",
    }}
    t_run = time.time()

    # -- inputs -------------------------------------------------------------
    h5 = Path(args.h5)
    digest = _sha256(h5)
    if digest != H5_SHA256:
        raise RuntimeError(f"{h5} sha256 {digest} != {H5_SHA256}")
    run_dir = Path(args.policybench_run)
    frozen, meta = load_frozen_policybench(run_dir)
    frozen_ids = frozen["household_id"].tolist()
    if meta["policyengine_bundles"]["us"]["certified_data_artifact_sha256"] != H5_SHA256:
        raise RuntimeError("PolicyBench's frozen run names a different data artifact")
    _log(f"h5 verified; PolicyBench frozen run has {len(frozen_ids)} public scenarios")
    n_batches = 2 if smoke else args.n_batches
    ckpt = Checkpoint(
        out_dir,
        {
            "h5_sha256": digest,
            "feed_facts_sha256": FEED_SHA256,
            "smoke_households": args.smoke_households,
            "pipeline_test_synthetic_measures": bool(args.pipeline_test_synthetic_measures),
            "n_batches": n_batches,
            "invariance_window": 25 if smoke else INVARIANCE_WINDOW,
            "policyengine_us": _version("policyengine-us"),
            "policyengine_core": _version("policyengine-core"),
            "release_tool_sha256": _sha256(REPO_ROOT / "tools" / "build_us_fiscal_refresh_release.py"),
            "fiscal_targets_module_sha256": _module_sha256("microcosm.build.us_runtime.fiscal_targets"),
            "h5_io_module_sha256": _module_sha256("microcosm.build.us_runtime.h5_io"),
            "source": source_provenance(),
        },
    )
    if ckpt.dir is not None:
        _log(f"checkpoint directory {ckpt.dir}")

    subset_ids = None
    if smoke:
        from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5

        all_ids = load_legacy_calibrated_us_h5(h5).table("household")["household_id"].to_numpy()
        others = np.setdiff1d(all_ids, frozen_ids)
        rng = np.random.default_rng(7)
        subset_ids = np.sort(np.concatenate([frozen_ids, rng.choice(others, args.smoke_households, replace=False)]))
    original, measured, dropped = load_measurement_frame(h5, subset_ids)
    entity_rows = {e: int(original.n(e)) for e in original.entities}
    _log(f"frame loaded {entity_rows}; formula-owned columns dropped {dropped}")

    cached = ckpt.load("engine_facts")
    if cached is None:
        with ResourceMeter() as meter:
            facts = _run_child(_child_engine_facts, str(h5), subset_ids)
        cached = {"facts": facts, "runtime": {**meter.record(), "child_peak_rss_gb": facts["child_peak_rss_gb"]}}
        ckpt.save("engine_facts", cached)
    facts = cached["facts"]
    runtime["engine_facts_child"] = cached["runtime"]
    _log(f"engine facts: {runtime['engine_facts_child']}")
    fs = {
        "filing_status_as_shipped": {"values": facts["filing_status_as_shipped"], "tax_unit_ids": facts["tax_unit_ids"]},
        "filing_status_measured": {"values": facts["filing_status_measured"], "tax_unit_ids": facts["tax_unit_ids"]},
    }
    desc, income_fields_present = build_descriptors(original, fs)
    pool_ids = desc["household_id"].to_numpy()
    design_w = desc["design_weight"].to_numpy(dtype=np.float64)
    w_all = float(design_w.sum())

    # -- targets ------------------------------------------------------------
    with ResourceMeter() as meter:
        national, registry_receipt = compile_national_specs(Path(args.feed_dir))
        national, restatement_receipt = strip_verified_restatements(national)
    runtime["target_compile"] = meter.record()
    registry_receipt["labelled_filter_restatements"] = restatement_receipt
    ckpt.save_summary("target_registry", registry_receipt)
    _log(f"national specs: {len(national)}; restatements {restatement_receipt['accepted_as_verified_restatements']}")
    flagged = [s.name for s in national if "poverty" in s.name.lower() or "spm" in s.name.lower()]
    if flagged:
        raise RuntimeError(f"poverty/SPM target names present: {flagged}")

    # -- materialization in child batches -------------------------------------
    batches = np.array_split(pool_ids, n_batches)
    payloads = []
    batch_records = []
    with ResourceMeter() as meter:
        for b, ids in enumerate(batches):
            payload = ckpt.load(f"materialize_batch_{b + 1}_of_{n_batches}")
            if payload is None or not np.array_equal(np.sort(payload["household_id"]), np.sort(ids)):
                with ResourceMeter() as batch_meter:
                    payload = _run_child(_child_materialize, str(h5), ids, national)
                payload["meter"] = batch_meter.record()
                ckpt.save(f"materialize_batch_{b + 1}_of_{n_batches}", payload)
            payloads.append(payload)
            batch_records.append(
                {
                    "batch": b + 1,
                    "households": int(payload["households"]),
                    "persons": int(payload["persons"]),
                    **payload["meter"],
                    "child_peak_rss_gb": round(float(payload["child_peak_rss_gb"]), 3),
                }
            )
            ckpt.save_summary("materialization_progress", {"batches": batch_records})
            _log(
                f"batch {b + 1}/{n_batches}: {payload['households']} households, "
                f"{payload['meter']}, child peak {payload['child_peak_rss_gb']:.1f} GB"
            )
    # Sum of the batch children's wall times and the largest batch peak (a
    # batch reloaded from a checkpoint keeps the time it took when computed).
    runtime["materialize_all_batches"] = {
        "seconds": round(sum(r["seconds"] for r in batch_records), 2),
        "peak_rss_gb": max(r["peak_rss_gb"] for r in batch_records),
    }
    names0 = set(payloads[0]["compiled_names"])
    if any(set(p["compiled_names"]) != names0 for p in payloads):
        raise RuntimeError("batches compiled different target sets")
    columns = sorted(payloads[0]["columns"])
    measure_ids = np.concatenate([p["household_id"] for p in payloads])
    measure_values = {c: np.concatenate([p["columns"][c] for p in payloads]) for c in columns}
    order = pd.Index(measure_ids).get_indexer(pool_ids)
    if (order < 0).any() or len(set(measure_ids.tolist())) != len(pool_ids):
        raise RuntimeError("materialized households do not match the pool")
    measures = pd.DataFrame({c: v[order] for c, v in measure_values.items()})

    # Batch invariance: re-materialize a window straddling the batch 0/1
    # boundary in one child and compare every household value.
    half = 25 if smoke else INVARIANCE_WINDOW
    window = np.concatenate([batches[0][-half:], batches[1][:half]])
    cached = ckpt.load("batch_invariance")
    if cached is None:
        with ResourceMeter() as meter:
            check = _run_child(_child_materialize, str(h5), window, national)
        cached = {"check": check, "runtime": {**meter.record(), "child_peak_rss_gb": check["child_peak_rss_gb"]}}
        ckpt.save("batch_invariance", cached)
    check = cached["check"]
    runtime["batch_invariance_child"] = cached["runtime"]
    del cached
    pos = pd.Index(pool_ids).get_indexer(check["household_id"])
    max_abs = 0.0
    n_diff_cells = 0
    for c in columns:
        a = measures[c].to_numpy()[pos]
        b_ = check["columns"][c]
        diff = np.abs(a - b_)
        max_abs = max(max_abs, float(diff.max()))
        n_diff_cells += int(np.sum(~np.isclose(a, b_, rtol=1e-9, atol=1e-6)))
    invariance = {
        "window_households": int(len(window)),
        "window": f"last {half} households of batch 1 and first {half} of batch 2, re-materialized together in one child",
        "measure_columns_compared": len(columns),
        "cells_differing": n_diff_cells,
        "max_abs_difference": max_abs,
    }
    _log(f"batch invariance: {invariance}")
    ckpt.save_summary("batch_invariance", invariance)
    if n_diff_cells:
        raise RuntimeError("Target values depend on household batch boundaries")
    del check
    if args.stop_after == "materialize":
        _write_json(out_dir / "run_status.json", {"status": "stopped", "stage": "materialize", "smoke": smoke})
        _log("stopping after materialization (--stop-after materialize); checkpoints kept")
        return

    compiled_specs = tuple(s for s in national if s.name in names0)
    household = measured.table("household").reset_index(drop=True)
    overwritten = sorted(set(household.columns) & set(measures.columns))
    household = household.drop(columns=overwritten)
    tables = {entity: measured.table(entity) for entity in measured.entities}
    tables["household"] = pd.concat([household, measures], axis=1)
    if tables["household"].columns.duplicated().any():
        raise RuntimeError("duplicate household columns after adding target measures")
    frame = Frame(tables, US_SCHEMA, {"household": measured.weights_for("household")})
    del measure_values, payloads

    # No poverty or SPM measure may be a target: exclude any compiled spec whose
    # name or measured variable mentions either (conservative: an SPM-unit
    # benefit field counts), before any target set is built.
    base_keys = ("base_variable", "base_variables", "source_variable", "variable", "indicator_filter_variable")
    spm_excluded = {
        s.name: sorted(
            f"{k}={v}" for k, v in s.metadata.items()
            if k in base_keys and ("poverty" in str(v).lower() or "spm" in str(v).lower())
        )
        for s in compiled_specs
    }
    spm_excluded = {
        n: v for n, v in spm_excluded.items() if v or "poverty" in n.lower() or "spm" in n.lower()
    }
    compiled_specs = tuple(s for s in compiled_specs if s.name not in spm_excluded)
    _log(f"excluded under the no-poverty/SPM rule: {spm_excluded}")
    all_names = [s.name for s in compiled_specs]
    headline, headline_per_rule = headline_names(all_names)
    head_set = set(headline)
    registries = {
        "national_all": TargetRegistry(compiled_specs, country="us"),
        "headline": TargetRegistry([s for s in compiled_specs if s.name in head_set], country="us"),
    }
    sets = {key: reg.to_target_set() for key, reg in registries.items()}
    problems = {key: build_constraint_matrix(frame, ts, "household") for key, ts in sets.items()}
    spec_by_name = {s.name: s for s in compiled_specs}
    flagged_base = sorted(
        f"{s.name}: {v}"
        for s in compiled_specs
        for k, v in s.metadata.items()
        if k in base_keys and ("poverty" in str(v).lower() or "spm" in str(v).lower())
    )
    if flagged_base or any("poverty" in n.lower() or "spm" in n.lower() for n in all_names):
        raise RuntimeError(f"poverty/SPM target survived the exclusion: {flagged_base}")

    # Household descriptor columns from target measures, with the spec metadata
    # that defines each measure recorded (not asserted).
    descriptor_sources = {}
    for col, (name, kind) in DESCRIPTOR_MEASURES.items():
        spec = spec_by_name[name]
        values = frame.table("household")[spec.measure].to_numpy(dtype=np.float64)
        desc[col] = values if kind == "value" else values > 0
        descriptor_sources[col] = {
            "target": name,
            "measure_column": spec.measure,
            "household_value": "household value of the target's measure" if kind == "value" else "measure > 0",
            "spec_metadata": {k: spec.metadata.get(k) for k in DESCRIPTOR_METADATA_KEYS if spec.metadata.get(k) is not None},
        }

    # -- frozen draw --------------------------------------------------------
    eligible = {
        basis: policybench_eligible(desc, f"{basis}_first")
        for basis in ("filing_status_as_shipped", "filing_status_measured")
    }
    requested = int(meta["requested_num_scenarios"])
    draw_kwargs = dict(
        requested=requested,
        private_fraction=float(meta["private_fraction"]),
        split_seed=int(meta["split_seed"]),
    )
    repro = {basis: policybench_draw(el, seed=int(meta["seed"]), **draw_kwargs) for basis, el in eligible.items()}
    basis = "filing_status_as_shipped"
    elig = eligible[basis]
    w_elig = float(elig["design_weight"].sum())
    fd = desc.set_index("household_id").loc[frozen_ids]
    frozen_draw = {
        "route_to_ids": "scenarios.csv scenario_json.metadata.household_id (PolicyBench writes the source household id there)",
        "reproduction": (
            "policybench/scenarios.py draw re-run here: eligibility filter, "
            "numpy.random.default_rng(seed).choice(ids, requested, replace=False, p=w/sum(w)), "
            "then split_scenarios' sha256 id-hash public split"
        ),
        "meta": {k: meta[k] for k in ("seed", "requested_num_scenarios", "num_scenarios", "private_fraction", "split_seed", "split")},
        "certified_data_build_id": meta["policyengine_bundles"]["us"]["certified_data_build_id"],
        "certified_data_artifact_sha256": meta["policyengine_bundles"]["us"]["certified_data_artifact_sha256"],
        "policybench_engine": {
            "policyengine_us": meta["policyengine_bundles"]["us"]["model_version"],
            "policyengine": meta["policyengine_bundles"]["us"]["policyengine_version"],
        },
        "eligibility_basis_used": basis,
        "by_filing_status_basis": {
            b: {
                "eligible_households": int(len(eligible[b])),
                "eligible_weight_total": float(eligible[b]["design_weight"].sum()),
                "reproduced_ids_equal_in_order": repro[b] == frozen_ids,
                "reproduced_overlap_with_frozen": len(set(repro[b]) & set(frozen_ids)),
            }
            for b in eligible
        },
        "filing_status_bases_agree_share_of_households": float(
            np.mean(desc["filing_status_as_shipped_first"] == desc["filing_status_measured_first"])
        ),
        "all_frozen_ids_in_file": bool(np.isin(frozen_ids, pool_ids).all()),
        "all_frozen_ids_eligible": bool(np.isin(frozen_ids, elig["household_id"]).all()),
        "row_checks_file_vs_scenarios_csv": {
            "of": len(frozen_ids),
            "state_equal": int(np.sum(fd["state"].to_numpy() == frozen["state"].to_numpy())),
            "filing_status_equal": int(
                np.sum(
                    np.asarray([POLICYBENCH_FILING_STATUSES.get(v, v) for v in fd[f"{basis}_first"]])
                    == frozen["filing_status"].to_numpy()
                )
            ),
            "adults_equal": int(np.sum(fd["adults"].to_numpy() == frozen["num_adults"].to_numpy())),
            "children_equal": int(np.sum(fd["children"].to_numpy() == frozen["num_children"].to_numpy())),
            "employment_income_equal_within_1_dollar": int(
                np.sum(np.abs(fd["employment_income_stored"].to_numpy() - frozen["employment_income_sum"].to_numpy()) <= 1.0)
            ),
            "employment_income_equal_exactly": int(np.sum(fd["employment_income_stored"].to_numpy() == frozen["employment_income_sum"].to_numpy())),
            "total_income_equal_exactly": int(np.sum(fd["policybench_total_income_stored"].to_numpy() == frozen["total_income"].to_numpy())),
            "total_income_equal_within_1_dollar": int(
                np.sum(np.abs(fd["policybench_total_income_stored"].to_numpy() - frozen["total_income"].to_numpy()) <= 1.0)
            ),
            "total_income_fields_present_in_file": income_fields_present,
            "total_income_fields_absent_from_file": [
                f for f in POLICYBENCH_MONETARY_INCOME_FIELDS if f not in income_fields_present
            ],
        },
        "pps_first_order_inclusion": {
            "exact_k_module_offers_successive_sampling_probabilities": False,
            "weight_used": "sum of eligible design weights / 100 (self-weighting approximation)",
            "max_requested_times_p": float(requested * elig["design_weight"].max() / w_elig),
            "max_public_n_times_p": float(K * elig["design_weight"].max() / w_elig),
            "note": (
                "microcosm.calibrate.exact_k supplies inclusion probabilities only for its own "
                "Sampford draw (select_exact_k); it has no function for numpy "
                "Generator.choice(replace=False, p=...), so the self-weighting approximation is used "
                "and checked empirically against the seed draws (sampling_distribution.json "
                "self_weighting_check)"
            ),
        },
    }
    _log(f"frozen draw reproduction: { {b: v['reproduced_ids_equal_in_order'] for b, v in frozen_draw['by_filing_status_basis'].items()} }")
    ckpt.save_summary("frozen_draw", frozen_draw)
    if not smoke and repro[basis] != frozen_ids:
        raise RuntimeError("Frozen PolicyBench draw did not reproduce exactly")

    # -- designs 1-3 --------------------------------------------------------
    idx = pd.Index(pool_ids)
    frozen_pos = idx.get_indexer(frozen_ids)
    n_pool = len(pool_ids)
    results: dict[str, dict] = {}
    weights_by: dict[str, np.ndarray] = {}
    exact_support_ids: dict[str, np.ndarray] = {}
    calib_runtime: dict[str, dict] = {}
    from microcosm.calibrate.gates import hard_concrete_open_probability_threshold

    # A record's returned L0 weight is positive exactly when its open
    # probability exceeds this cutoff (gates.py hard_concrete_open_probability_threshold).
    gate_threshold = hard_concrete_open_probability_threshold(selection_settings["temperature"])
    for key, target_set in sets.items():
        problem = problems[key]
        stage = "designs_{}_{}".format(
            key,
            _digest_of(
                {
                    "selection": selection_settings,
                    "refit": refit_settings,
                    "draw_pi_hi": EXACT_K_DRAW_PI_HI,
                    "draw_seed": EXACT_K_DRAW_SEED,
                    "targets": [t.name for t in problem.targets],
                    "frozen_ids": [int(h) for h in frozen_ids],
                }
            ),
        )
        cached = ckpt.load(stage)
        if cached is not None:
            results[key] = cached["result"]
            weights_by.update(cached["weights_by"])
            if cached["exact_support_ids"] is not None:
                exact_support_ids[key] = cached["exact_support_ids"]
            calib_runtime.update(cached["calib_runtime"])
            continue
        key_weights: dict[str, np.ndarray] = {}
        key_runtime: dict[str, dict] = {}
        r: dict[str, object] = {
            "targets_compiled": len(problem.targets),
            "targets_skipped": [s.target.name for s in problem.skipped],
            "target_names": [t.name for t in problem.targets],
            "full_file": {"fit": evaluate_weights(problem, design_w), "weights": weight_concentration(design_w)},
        }
        # Design 1: self-weighted frozen sample.
        w1 = _scatter(n_pool, frozen_pos, w_elig / K)
        r["design1_self_weighted"] = {
            "weight_each": w_elig / K,
            "fit": evaluate_weights(problem, w1),
            "weights": weight_concentration(w1[frozen_pos], w_elig / K),
        }
        r["design1b_self_weighted_file_total"] = {
            "weight_each": w_all / K,
            "fit": evaluate_weights(problem, _scatter(n_pool, frozen_pos, w_all / K)),
        }
        ckpt.save_summary(f"{key}_design1", {"settings": {"target_set": key}, "result": r["design1_self_weighted"], "file_total_sensitivity": r["design1b_self_weighted_file_total"]})
        # Design 2: post-stratified frozen sample.
        with ResourceMeter() as meter:
            sub = subset_frame(frame, frozen_ids, np.full(K, w_elig / K))
            res2 = calibrate(sub, target_set, **refit_settings)
        sub_ids = sub.table("household")["household_id"].to_numpy()
        w2 = _scatter(n_pool, idx.get_indexer(sub_ids), res2.weights)
        r["design2_post_stratified"] = {
            "fit": evaluate_weights(problem, w2),
            "weights": weight_concentration(res2.weights, w_elig / K),
            "skipped_on_subset": [s.target.name for s in res2.skipped],
        }
        key_runtime[f"{key}_design2_post_stratified"] = meter.record()
        key_weights[f"{key}_design2"] = w2
        ckpt.save_summary(f"{key}_design2", {"settings": refit_settings, "result": r["design2_post_stratified"], "runtime": meter.record()})
        if key == "national_all":
            write_households(out_dir, desc, pool_ids, frozen_ids, w_elig, key_weights, {})
        _log(f"[{key}] design 2 {meter.record()}")

        # Design 3: exact-k (L0 selection, Sampford boundary draw, frozen-support refit).
        probes: list[dict] = []

        def _progress(event, _probes=probes, _key=key):
            if event.get("kind") in ("budget_probe", "budget_search_done"):
                _probes.append(dict(event))
                ckpt.save_summary(f"{_key}_selection_progress", {"settings": selection_settings, "probes": _probes})
                _log(f"[{_key}] {event}")

        d3: dict[str, object] = {}
        selection = None
        with ResourceMeter() as meter:
            try:
                selection = calibrate(frame, target_set, progress_callback=_progress, **selection_settings)
            except (ValueError, RuntimeError) as exc:
                # Recorded, not hidden: the brief asks for the closest counts and
                # penalties when the calibrator cannot reach the budget.
                d3["selection_failed"] = repr(exc)
        key_runtime[f"{key}_design3_l0_selection"] = meter.record()
        support_ids = None
        if selection is None:
            d3["selection"] = {
                "probes": [p for p in probes if p.get("kind") == "budget_probe"],
                "search_done": next((p for p in probes if p.get("kind") == "budget_search_done"), None),
            }
            _log(f"[{key}] exact-k L0 selection FAILED: {d3['selection_failed']}")
        else:
            pi = np.asarray(selection.gate_open_probabilities, dtype=np.float64)
            pi_sorted = np.sort(pi)[::-1]
            d3["selection"] = {
                "settled_l0_lambda": float(selection.l0_lambda),
                "budget_basis": selection_settings["budget_basis"],
                "budget_measure_at_settle": int(selection.n_nonzero),
                "open_probability_mass": float(pi.sum()),
                "gated_weights_nonzero_count": int(np.sum(selection.weights > 1e-6 * float(np.mean(design_w)))),
                "deterministic_gate_open_threshold": float(gate_threshold),
                "gates_above_deterministic_threshold": int(np.sum(pi > gate_threshold)),
                "open_probability_mass_at_or_below_deterministic_threshold": float(pi[pi <= gate_threshold].sum()),
                "gates_open_prob_ge_0_5": int(np.sum(pi >= 0.5)),
                "gates_open_prob_ge_pi_hi": int(np.sum(pi >= EXACT_K_DRAW_PI_HI)),
                "open_probability_rank_100": float(pi_sorted[min(K - 1, len(pi) - 1)]),
                "open_probability_rank_101": float(pi_sorted[min(K, len(pi) - 1)]),
                "budget_search": selection.options.get("budget_search"),
                "gated_fit_full_pool": evaluate_weights(problem, selection.weights),
            }
            d3["feasibility"] = exact_k_design_feasibility(pi, K, EXACT_K_DRAW_PI_HI)
            try:
                support, receipt, q = select_exact_k(pi, K, EXACT_K_DRAW_PI_HI, EXACT_K_DRAW_SEED)
                support = assert_exact_k_support(support, K, pool_size=n_pool)
            except (ValueError, RuntimeError) as exc:
                d3["draw_failed"] = repr(exc)
                _log(f"[{key}] exact-k draw FAILED: {exc!r}")
            else:
                ckpt.save_summary(f"{key}_design3", {"settings": selection_settings, "result": d3, "draw_receipt": receipt, "status": "refit_pending", "runtime": key_runtime})
                with ResourceMeter() as meter:
                    refit = refit_l0_selection(
                        frame,
                        target_set,
                        selection,
                        support=support,
                        k=K,
                        support_inclusion_probabilities=q,
                        epochs=refit_settings["epochs"],
                        learning_rate=refit_settings["learning_rate"],
                        mass=refit_settings["mass"],
                        max_weight_ratio=refit_settings["max_weight_ratio"],
                        seed=refit_settings["seed"],
                    )
                key_runtime[f"{key}_design3_refit"] = meter.record()
                refit_ids = refit.refit.frame.table("household")["household_id"].to_numpy()
                if set(refit_ids.tolist()) != set(pool_ids[support].tolist()) or len(refit_ids) != K:
                    raise RuntimeError("refit households differ from the exact-k support")
                w3 = _scatter(n_pool, idx.get_indexer(refit_ids), refit.refit.weights)
                ht = _scatter(n_pool, idx.get_indexer(refit_ids), refit.refit.initial_weights)
                pi_support = pi[support]
                certain = pi_support >= EXACT_K_DRAW_PI_HI
                d3["draw"] = {
                    "receipt": receipt,
                    "pi_hi": EXACT_K_DRAW_PI_HI,
                    "seed": EXACT_K_DRAW_SEED,
                    "boundary_draw_needed": int(receipt["certainty_count"]) < K,
                    "certainties": int(receipt["certainty_count"]),
                    "boundary_draws": int(K - int(receipt["certainty_count"])),
                    "boundary_draws_at_or_below_deterministic_threshold": int(
                        np.sum(~certain & (pi_support <= gate_threshold))
                    ),
                    "support_open_probability": _quantiles(pi_support),
                    "inclusion_probability_min": float(np.min(q)),
                    "inclusion_probability_max": float(np.max(q)),
                    "support_in_top_100_open_probability": int(
                        np.isin(support, np.argsort(-pi, kind="stable")[:K]).sum()
                    ),
                }
                d3["refit"] = {
                    "fit": evaluate_weights(problem, w3),
                    "ht_baseline_fit": evaluate_weights(problem, ht),
                    "weights": weight_concentration(refit.refit.weights, w_all / K),
                    "ht_baseline_weights": weight_concentration(refit.refit.initial_weights, w_all / K),
                    "skipped_on_subset": [s.target.name for s in refit.refit.skipped],
                }
                key_weights[f"{key}_design3"] = w3
                support_ids = np.asarray(refit_ids)
                _log(
                    f"[{key}] exact-k: lambda {selection.l0_lambda:.3g}, certainties {receipt['certainty_count']}, "
                    f"refit capped MAPE {d3['refit']['fit']['capped_mape']:.3f}"
                )
        results[key] = {**r, "design3_exact_k": d3}
        ckpt.save_summary(f"{key}_design3", {"settings": selection_settings, "result": d3, "runtime": key_runtime})
        if support_ids is None:
            ckpt.save_summary("exact_k_failure", {"target_set": key, "result": d3, "reason": "Maintained path failed; no replacement or hand-selected households; stopping before further designs."})
            raise RuntimeError(f"Maintained exact-k path failed for {key}; see exact_k_failure.json")
        weights_by.update(key_weights)
        calib_runtime.update(key_runtime)
        if support_ids is not None:
            exact_support_ids[key] = support_ids
        if key == "national_all":
            write_households(out_dir, desc, pool_ids, frozen_ids, w_elig, weights_by, exact_support_ids)
        ckpt.save(
            stage,
            {
                "result": results[key],
                "weights_by": key_weights,
                "exact_support_ids": support_ids,
                "calib_runtime": key_runtime,
            },
        )
        ckpt.save_summary(
            f"summary_{key}",
            {
                "stage": stage,
                "design1_fit": r["design1_self_weighted"]["fit"],
                "design2_fit": r["design2_post_stratified"]["fit"],
                "design3": {k: v for k, v in d3.items() if k != "selection"}
                | {"selection": {k: v for k, v in d3.get("selection", {}).items() if k != "budget_search"}},
                "runtime": key_runtime,
            },
        )
    if args.stop_after == "designs":
        _write_json(out_dir / "run_status.json", {"status": "stopped", "stage": "designs", "smoke": smoke})
        _log("stopping after designs 1-3 (--stop-after designs); checkpoints kept")
        return


    # -- design 4: sampling distribution ------------------------------------
    seed_rng = np.random.default_rng(SAMPLING_DISTRIBUTION["seed_generator_seed"])
    seeds: list[int] = []
    while len(seeds) < n_seeds:
        s = int(seed_rng.integers(0, 2**31 - 1))
        if s != int(meta["seed"]) and s not in seeds:
            seeds.append(s)
    sampling_stage = "sampling_{}".format(
        _digest_of(
            {
                "refit": refit_settings,
                "seeds": seeds,
                "targets": {k: [t.name for t in problems[k].targets] for k in sets},
                "draw": {k: v for k, v in draw_kwargs.items()},
                "eligible": int(len(elig)),
            }
        )
    )
    state = ckpt.load(sampling_stage) or {
        "draws": [],
        "dist": {k: {"d1": [], "d2": []} for k in sets},
        "elapsed_seconds": 0.0,
        "peak_rss_gb": 0.0,
    }
    draws = state["draws"]
    dist: dict[str, dict[str, list]] = state["dist"]
    prior_elapsed = float(state["elapsed_seconds"])
    if draws:
        _log(f"sampling distribution: resuming after {len(draws)} checkpointed seeds ({prior_elapsed:.0f}s)")
    # Wall-clock budget: seeds run in list order; before each seed, stop if the
    # elapsed time plus the mean time per completed seed would exceed the
    # budget. The completed count is recorded next to the requested one.
    # Elapsed time includes checkpointed seeds from an interrupted run.
    budget_seconds = max(0.0, min(float(args.sampling_budget_minutes) * 60.0, args.total_budget_minutes * 60.0 - (time.time() - t_run)))
    stopped_on_budget = False
    with ResourceMeter() as meter:
        for i, s in enumerate(seeds):
            if i < len(draws):
                continue
            elapsed = prior_elapsed + time.time() - meter.t0
            if i > 0 and elapsed + elapsed / i > budget_seconds:
                stopped_on_budget = True
                _log(f"sampling distribution: time budget reached after {i} seeds ({elapsed:.0f}s)")
                break
            ids = policybench_draw(elig, seed=s, **draw_kwargs)
            draws.append(ids)
            pos = idx.get_indexer(ids)
            for key, target_set in sets.items():
                problem = problems[key]
                f1 = evaluate_weights(problem, _scatter(n_pool, pos, w_elig / K))
                sub = subset_frame(frame, ids, np.full(K, w_elig / K))
                with ResourceMeter() as calibration_meter:
                    res = calibrate(sub, target_set, **refit_settings)
                sub_ids = sub.table("household")["household_id"].to_numpy()
                f2 = evaluate_weights(problem, _scatter(n_pool, idx.get_indexer(sub_ids), res.weights))
                conc = weight_concentration(res.weights)
                dist[key]["d1"].append({"seed": s, "capped_mape": f1["capped_mape"], "uncapped_mape": f1["uncapped_mape"], "within10": f1["share_within_10pct"], "within25": f1["share_within_25pct"]})
                dist[key]["d2"].append(
                    {
                        "seed": s,
                        "capped_mape": f2["capped_mape"],
                        "uncapped_mape": f2["uncapped_mape"],
                        "within10": f2["share_within_10pct"],
                        "within25": f2["share_within_25pct"],
                        "kish_ess": conc["kish_ess"],
                        "max_over_mean": conc["max_over_mean"],
                        "runtime": calibration_meter.record(),
                    }
                )
            state["elapsed_seconds"] = prior_elapsed + time.time() - meter.t0
            state["peak_rss_gb"] = max(float(state["peak_rss_gb"]), meter.peak / 1e9)
            ckpt.save(sampling_stage, state)
            ckpt.save_summary("sampling_progress", {"seeds_completed": i + 1, "seeds_requested": n_seeds, "elapsed_seconds": state["elapsed_seconds"], "per_seed": dist})
            if (i + 1) % 20 == 0:
                _log(f"sampling distribution {i + 1}/{len(seeds)} ({state['elapsed_seconds']:.0f}s)")
    seeds_requested = len(seeds)
    seeds = seeds[: len(draws)]
    runtime["sampling_distribution"] = {
        "seconds": round(prior_elapsed + meter.seconds, 2),
        "peak_rss_gb": round(max(float(state["peak_rss_gb"]), meter.peak / 1e9), 3),
        "calibrations": len(sets) * len(seeds),
    }
    sampling_run = {
        "seeds_requested": seeds_requested,
        "seeds_completed": len(seeds),
        "time_budget_minutes": float(args.sampling_budget_minutes),
        "stopped_on_time_budget": stopped_on_budget,
        "total_run_budget_minutes": args.total_budget_minutes,
        "effective_sampling_budget_seconds": budget_seconds,
        "reduction_reason": ("Estimated next seed would exceed sampling or whole-run time budget" if stopped_on_budget else "Explicit smoke/CLI seed count" if n_seeds < SAMPLING_DISTRIBUTION["n_seeds"] else None),
    }
    if args.stop_after == "sampling":
        _write_json(out_dir / "run_status.json", {"status": "stopped", "stage": "sampling", "smoke": smoke})
        _log("stopping after the sampling distribution (--stop-after sampling); checkpoints kept")
        return
    # Empirical check of the self-weighting approximation: across the seed
    # draws, public inclusions per design-weight decile of the eligible pool
    # versus the count the approximation implies (n * w_i / sum(w) per draw).
    elig_w = elig["design_weight"].to_numpy(dtype=np.float64)
    decile = np.minimum((pd.Series(elig_w).rank(pct=True, method="first").to_numpy() * 10).astype(int), 9)
    counts = pd.Series(np.concatenate(draws)).value_counts()
    observed = counts.reindex(elig["household_id"].to_numpy(), fill_value=0).to_numpy()
    expected = K * elig_w / elig_w.sum() * len(draws)
    self_weighting_check = {
        "draws": len(draws),
        "by_design_weight_decile": [
            {
                "decile": d + 1,
                "observed_inclusions": int(observed[decile == d].sum()),
                "expected_under_self_weighting": float(expected[decile == d].sum()),
            }
            for d in range(10)
        ],
    }
    runtime["per_calibration"] = calib_runtime
    runtime["whole_run_seconds"] = round(time.time() - t_run, 1)
    runtime["parent_getrusage_high_water_gb"] = round(_peak_rss_gb_self(), 3)
    runtime["materialization_batches"] = batch_records

    write_outputs(
        out_dir=out_dir,
        smoke=smoke,
        desc=desc,
        pool_ids=pool_ids,
        design_w=design_w,
        w_all=w_all,
        w_elig=w_elig,
        elig=elig,
        frozen_ids=frozen_ids,
        results=results,
        weights_by=weights_by,
        dist=dist,
        draws=draws,
        seeds=seeds,
        sampling_run=sampling_run,
        exact_support_ids=exact_support_ids,
        self_weighting_check=self_weighting_check,
        runtime=runtime,
        inputs_extra={
            "release_id": RELEASE_ID,
            "source": source_provenance(),
            "h5_sha256": digest,
            "loader": "microcosm.build.us_runtime.h5_io.load_legacy_calibrated_us_h5",
            "entity_rows": entity_rows,
            "household_weight_total": w_all,
            "engine": {
                "policyengine_us": _version("policyengine-us"),
                "policyengine_core": _version("policyengine-core"),
                "torch": _version("torch"),
                "numpy": _version("numpy"),
            },
            "formula_owned_columns_dropped": dropped,
            "formula_owned_column_deltas": facts["deltas"],
            "policybench_run_dir": str(run_dir.relative_to(run_dir.parents[4])) if not smoke else str(run_dir),
            "policybench_scenarios_csv_sha256": _sha256(run_dir / "scenarios.csv"),
            "policybench_scenarios_meta_sha256": _sha256(run_dir / "scenarios.csv.meta.json"),
            "registry": registry_receipt,
            "materialization": {
                "function": "tools/build_us_fiscal_refresh_release.py::_materialize_target_frame",
                "household_batches": n_batches,
                "national_specs_after_restatements": len(national),
                "national_specs_compiled": len(names0),
                "national_specs_used_after_poverty_spm_exclusion": len(compiled_specs),
                "dropped_target_names": sorted(set(s.name for s in national) - names0),
                "compiled_families": dict(sorted(Counter(s.family for s in compiled_specs).items())),
                "batch_invariance_check": invariance,
                "household_columns_replaced_by_measures": overwritten,
            },
            "national_agi_by_band_rows": _national_agi_band_rows(compiled_specs),
            "headline_rules": {label: prefix for label, prefix in HEADLINE_RULES},
            "headline_targets_per_rule": headline_per_rule,
            "poverty_spm_check": {
                "rule": "exclude any compiled target whose name, or whose base/source/filter variable, contains 'poverty' or 'spm'",
                "excluded_targets": spm_excluded,
                "target_names_matching_poverty_or_spm_after_exclusion": [n for n in all_names if "poverty" in n.lower() or "spm" in n.lower()],
                "base_variables_matching_poverty_or_spm_after_exclusion": flagged_base,
            },
            "descriptor_columns_from_target_measures": descriptor_sources,
            "frozen_draw": frozen_draw,
            "pool": {
                "households": int(n_pool),
                "household_weight_total": w_all,
                "eligible_households": int(len(elig)),
                "eligible_weight_total": w_elig,
                "eligible_weight_share": w_elig / w_all,
            },
            "settings": {
                "resource_limits": {"omp_num_threads": int(os.environ["OMP_NUM_THREADS"]), "concurrent_compute_processes": 2, "max_rss_gb": 32, "rss_guard_gb": MAX_RSS_BYTES / 1e9, "minimum_free_disk_gib": 10},
                "sample_refit": refit_settings,
                "exact_k_selection": selection_settings,
                "exact_k_draw_pi_hi": EXACT_K_DRAW_PI_HI,
                "exact_k_draw_seed": EXACT_K_DRAW_SEED,
                "sampling_distribution": {**SAMPLING_DISTRIBUTION, "n_seeds": n_seeds},
                "fit_scale": "s = max(|target|, 1) (microcosm.calibrate.default_target_loss_scales)",
                "capped_mape_cap": TARGET_LOSS_CAP,
                "target_loss_weights": "equal (calibrate default)",
            },
            "smoke_households": args.smoke_households,
            "pipeline_test_synthetic_measures": bool(args.pipeline_test_synthetic_measures),
        },
    )
    _log(f"done in {time.time() - t_run:.0f}s")
    _write_json(out_dir / "run_status.json", {"status": "complete", "smoke": smoke, "seconds": round(time.time() - t_run, 2), "seeds_completed": len(seeds)})


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_households(out_dir, desc, pool_ids, frozen_ids, w_elig, weights_by, exact_support_ids):
    """Persist the two permitted household sets immediately after national refit."""
    idx = pd.Index(pool_ids)
    frozen_pos = idx.get_indexer(frozen_ids)
    exact = {key: np.sort(idx.get_indexer(ids)) for key, ids in exact_support_ids.items()}
    exact_ids = {key: pool_ids[pos].tolist() for key, pos in exact.items()}
    # households.csv + household_ids.json (the two 100-household sets)
    columns = [
        "design", "household_id", "state", "household_size", "adults", "children", "head_age",
        "employment_income", "agi_or_total_income", "receives_snap", "receives_medicaid", "receives_ssi",
        "eitc_positive", "social_security_positive", "design_weight", "calibrated_weight",
        "post_stratified_weight",
    ]
    sampled = desc.iloc[frozen_pos].copy()
    sampled["design"] = "sampled"
    sampled["calibrated_weight"] = w_elig / K
    sampled["post_stratified_weight"] = weights_by["national_all_design2"][frozen_pos]
    frames = [sampled]
    if "national_all" in exact:
        ek = desc.iloc[exact["national_all"]].copy()
        ek["design"] = "exact_k"
        ek["calibrated_weight"] = weights_by["national_all_design3"][exact["national_all"]]
        ek["post_stratified_weight"] = np.nan
        frames.append(ek)
    rows = pd.concat(frames)[columns]
    rows = rows.sort_values(["design", "employment_income", "household_id"], ascending=[False, True, True], kind="stable")
    for col in ("employment_income", "agi_or_total_income", "head_age"):
        rows[col] = rows[col].round(0).astype(np.int64)
    rows.to_csv(out_dir / "households.csv", index=False)
    _write_json(
        out_dir / "household_ids.json",
        {
            "release_id": RELEASE_ID,
            "rows": rows.where(pd.notna(rows), None).to_dict(orient="records"),
            "sampled_frozen_policybench_public_100_in_draw_order": [int(h) for h in frozen_ids],
            "exact_k_national_all_100": sorted(int(h) for h in exact_ids.get("national_all", [])),
        },
    )


def write_outputs(*, out_dir, smoke, desc, pool_ids, design_w, w_all, w_elig, elig, frozen_ids, results,
                  weights_by, dist, draws, seeds, sampling_run, exact_support_ids, self_weighting_check, runtime,
                  inputs_extra) -> None:
    n_pool = len(pool_ids)
    idx = pd.Index(pool_ids)
    frozen_pos = idx.get_indexer(frozen_ids)
    # The exact-k sets are the refit supports themselves (not weights > 0).
    exact = {k: np.sort(idx.get_indexer(ids)) for k, ids in exact_support_ids.items()}
    if any((p < 0).any() or len(p) != K for p in exact.values()):
        raise RuntimeError("exact-k support ids do not map to 100 pool households")
    exact_ids = {k: pool_ids[p].tolist() for k, p in exact.items()}
    eligible_ids = elig["household_id"].to_numpy()

    inputs = dict(inputs_extra)
    for key, r in results.items():
        inputs.setdefault("target_sets", {})[key] = {
            "targets_compiled_on_frame": r["targets_compiled"],
            "targets_skipped_on_frame": r["targets_skipped"],
            "target_names": r["target_names"],
        }
    inputs["headline_target_names"] = results["headline"]["target_names"]
    _write_json(out_dir / "inputs.json", inputs)

    fit, conc, exact_out = {}, {}, {}
    for key, r in results.items():
        d3 = r["design3_exact_k"]
        fit[key] = {
            "targets": r["targets_compiled"],
            "full_file": r["full_file"]["fit"],
            "design1_pps_self_weighted": r["design1_self_weighted"]["fit"],
            "design1b_pps_self_weighted_scaled_to_file_total": r["design1b_self_weighted_file_total"]["fit"],
            "design2_pps_post_stratified": r["design2_post_stratified"]["fit"],
            "design3_exact_k_refit": d3.get("refit", {}).get("fit"),
            "design3_exact_k_ht_baseline_before_refit": d3.get("refit", {}).get("ht_baseline_fit"),
            "design3_l0_gated_weights_full_pool": d3["selection"].get("gated_fit_full_pool"),
        }
        conc[key] = {
            "full_file": r["full_file"]["weights"],
            "design1_pps_self_weighted": r["design1_self_weighted"]["weights"],
            "design2_pps_post_stratified": r["design2_post_stratified"]["weights"],
            "design3_exact_k_ht_baseline_before_refit": d3.get("refit", {}).get("ht_baseline_weights"),
            "design3_exact_k_refit": d3.get("refit", {}).get("weights"),
        }
        exact_out[key] = {
            "selection": {k: v for k, v in d3["selection"].items() if k != "gated_fit_full_pool"},
            "selection_failed": d3.get("selection_failed"),
            "feasibility": d3.get("feasibility"),
            "draw": d3.get("draw"),
            "draw_failed": d3.get("draw_failed"),
            "refit_skipped_targets": d3.get("refit", {}).get("skipped_on_subset"),
            "design2_skipped_targets": r["design2_post_stratified"]["skipped_on_subset"],
        }
    _write_json(out_dir / "fit.json", fit)
    _write_json(out_dir / "weight_concentration.json", conc)
    _write_json(out_dir / "exact_k.json", {"settings": inputs["settings"], "runs": exact_out})

    sd = {
        "procedure": (
            "PolicyBench's own procedure per seed: PPS without replacement of "
            f"{inputs['frozen_draw']['meta']['requested_num_scenarios']} eligible households, then the "
            f"id-hash public split (split_seed {inputs['frozen_draw']['meta']['split_seed']}), keeping the public {K}; "
            "design 1 = self-weighted; design 2 = post-stratified with the sample-refit settings"
        ),
        "seed_generator": (
            f"numpy.random.default_rng({SAMPLING_DISTRIBUTION['seed_generator_seed']}).integers(0, 2**31 - 1), "
            "skipping duplicates and PolicyBench's seed 42"
        ),
        "seeds": seeds,
        "run": sampling_run,
        "eligibility_filter": (
            "policybench/scenarios.py::_eligible_households: one tax unit, one SPM unit, one family, "
            ">=1 adult (age >= 18), first-person filing status SINGLE/JOINT/HEAD_OF_HOUSEHOLD"
        ),
        "eligible_households": int(len(elig)),
        "self_weighting_check": self_weighting_check,
    }
    for key, r in results.items():
        d1, d2 = dist[key]["d1"], dist[key]["d2"]
        f1 = r["design1_self_weighted"]["fit"]
        f2 = r["design2_post_stratified"]["fit"]
        ek = r["design3_exact_k"].get("refit", {}).get("fit")
        entry = {
            "design1_self_weighted": {m: _quantiles([x[m] for x in d1]) for m in ("capped_mape", "uncapped_mape", "within10", "within25")},
            "design2_post_stratified": {
                m: _quantiles([x[m] for x in d2]) for m in ("capped_mape", "uncapped_mape", "within10", "within25", "kish_ess", "max_over_mean")
            },
            "frozen_draw_position": {
                "design1_capped_mape": f1["capped_mape"],
                "design1_uncapped_mape": f1["uncapped_mape"],
                "design1_uncapped_share_of_seeds_at_or_below": _share_at_or_below([x["uncapped_mape"] for x in d1], f1["uncapped_mape"]),
                "design1_share_of_seeds_at_or_below": _share_at_or_below([x["capped_mape"] for x in d1], f1["capped_mape"]),
                "design1_within10": f1["share_within_10pct"],
                "design1_share_of_seeds_within10_at_or_below": _share_at_or_below([x["within10"] for x in d1], f1["share_within_10pct"]),
                "design2_capped_mape": f2["capped_mape"],
                "design2_uncapped_mape": f2["uncapped_mape"],
                "design2_uncapped_share_of_seeds_at_or_below": _share_at_or_below([x["uncapped_mape"] for x in d2], f2["uncapped_mape"]),
                "design2_share_of_seeds_at_or_below": _share_at_or_below([x["capped_mape"] for x in d2], f2["capped_mape"]),
                "design2_within10": f2["share_within_10pct"],
                "design2_share_of_seeds_within10_at_or_below": _share_at_or_below([x["within10"] for x in d2], f2["share_within_10pct"]),
            },
            "per_seed": {"design1": d1, "design2": d2},
        }
        if ek is not None:
            entry["exact_k_position"] = {
                "capped_mape": ek["capped_mape"],
                "uncapped_mape": ek["uncapped_mape"],
                "share_of_design2_seeds_uncapped_at_or_below": _share_at_or_below([x["uncapped_mape"] for x in d2], ek["uncapped_mape"]),
                "share_of_design2_seeds_at_or_below": _share_at_or_below([x["capped_mape"] for x in d2], ek["capped_mape"]),
                "within10": ek["share_within_10pct"],
                "share_of_design2_seeds_within10_at_or_below": _share_at_or_below([x["within10"] for x in d2], ek["share_within_10pct"]),
            }
        sd[key] = entry
    _write_json(out_dir / "sampling_distribution.json", sd)

    # composition.json
    cuts = _weighted_quantiles(desc["employment_income"].to_numpy(), design_w, [0.2, 0.4, 0.6, 0.8])
    by_design = {
        "full_population": design_w,
        "sampled_self_weighted": _scatter(n_pool, frozen_pos, w_elig / K),
        "sampled_post_stratified_national_all": weights_by["national_all_design2"],
        "sampled_post_stratified_headline": weights_by["headline_design2"],
    }
    for key in exact:
        by_design[f"exact_k_{key}"] = weights_by[f"{key}_design3"]
    composition = {
        "employment_income_quintile_cut_points_population": cuts,
        "columns": {
            "household_size": "count of person rows in the household",
            "children": "members under 18 (household_size - adults)",
            "head_age": "age of the is_household_head member (oldest member if none flagged)",
            "employment_income": inputs["descriptor_columns_from_target_measures"]["employment_income"]["target"],
            "receipt": {
                k: inputs["descriptor_columns_from_target_measures"][k]["target"] + " measure > 0"
                for k in ("receives_snap", "receives_medicaid", "receives_ssi", "eitc_positive", "social_security_positive")
            },
            "weights": "shares are weighted by each design's own weights",
        },
        "designs": {name: _composition(desc, w, cuts) for name, w in by_design.items()},
    }
    _write_json(out_dir / "composition.json", composition)

    # overlap.json
    union = set(np.unique(np.asarray(draws)).tolist())
    overlap = {"distinct_households_across_seed_draws": len(union), "seed_draws": len(draws),
               "frozen_100_in_any_seed_draw": len(set(frozen_ids) & union)}
    for key, ids in exact_ids.items():
        overlap[f"exact_k_{key}_in_frozen_100"] = len(set(ids) & set(frozen_ids))
        overlap[f"exact_k_{key}_in_any_seed_draw"] = len(set(ids) & union)
        overlap[f"exact_k_{key}_policybench_eligible"] = int(np.isin(ids, eligible_ids).sum())
    if len(exact_ids) == 2:
        overlap["exact_k_national_all_vs_headline_shared"] = len(set(exact_ids["national_all"]) & set(exact_ids["headline"]))
    _write_json(out_dir / "overlap.json", overlap)

    # leverage.json + household_contrast.json
    top1 = float(np.quantile(design_w, 0.99))
    emp = desc["employment_income"].to_numpy()
    emp_p90, emp_p99 = _weighted_quantiles(emp, design_w, [0.90, 0.99])
    design_pct = pd.Series(design_w).rank(pct=True).to_numpy()
    leverage = {
        "design_weight_p99_threshold": top1,
        "design_weight_median_population": float(np.median(design_w)),
        "employment_income_p90_population_weighted": emp_p90,
        "employment_income_p99_population_weighted": emp_p99,
    }
    contrast = {
        "definitions": {
            "one_person": "household_size == 1",
            "retiree_head": "head_age >= 65",
            "high_earner_p90": f"household employment income above the population's weighted 90th percentile ({emp_p90:.0f})",
            "high_earner_p99": f"household employment income above the population's weighted 99th percentile ({emp_p99:.0f})",
            "benefit_recipient": "receives SNAP, Medicaid or SSI (measure > 0)",
            "weights": "sampled: self-weighted sum(eligible w)/100 each; exact_k: refit calibrated weights",
        }
    }
    sets = {"sampled": (frozen_pos, _scatter(n_pool, frozen_pos, w_elig / K))}
    for key in exact:
        sets[f"exact_k_{key}"] = (exact[key], weights_by[f"{key}_design3"])
    for name, (pos, cal) in sets.items():
        sub = desc.iloc[pos]
        cw = cal[pos]
        entry = {
            "households": int(len(pos)),
            "in_top_1pct_design_weight": int((design_w[pos] >= top1).sum()),
            "design_weight_percentile_median": float(np.median(design_pct[pos])),
            "one_person_households": int((sub["household_size"] == 1).sum()),
            "household_size_6_plus": int((sub["household_size"] >= 6).sum()),
            "head_65_plus": int((sub["head_age"] >= 65).sum()),
            "head_under_30": int((sub["head_age"] < 30).sum()),
            "with_children": int((sub["children"] > 0).sum()),
            "zero_employment_income": int((sub["employment_income"] <= 0).sum()),
            "employment_income_above_population_p90": int((sub["employment_income"] > emp_p90).sum()),
            "employment_income_above_population_p99": int((sub["employment_income"] > emp_p99).sum()),
            "receives_snap": int(sub["receives_snap"].sum()),
            "receives_medicaid": int(sub["receives_medicaid"].sum()),
            "receives_ssi": int(sub["receives_ssi"].sum()),
            "eitc_positive": int(sub["eitc_positive"].sum()),
            "social_security_positive": int(sub["social_security_positive"].sum()),
            "receives_any_of_snap_medicaid_ssi": int((sub["receives_snap"] | sub["receives_medicaid"] | sub["receives_ssi"]).sum()),
            "receives_two_or_more_of_snap_medicaid_ssi_eitc": int(
                (sub[["receives_snap", "receives_medicaid", "receives_ssi", "eitc_positive"]].sum(axis=1).to_numpy() >= 2).sum()
            ),
            "distinct_states": int(sub["state"].nunique()),
            "policybench_eligible": int(np.isin(pool_ids[pos], eligible_ids).sum()),
            "weights": weight_concentration(cw, w_all / K),
            "calibrated_over_design_weight": _quantiles(cw / design_w[pos]),
        }
        top10 = np.argsort(-cw, kind="stable")[:10]
        t10 = sub.iloc[top10]
        entry["top10_weight_households_profile"] = {
            "one_person": int((t10["household_size"] == 1).sum()),
            "head_65_plus": int((t10["head_age"] >= 65).sum()),
            "zero_employment_income": int((t10["employment_income"] <= 0).sum()),
            "receives_any_of_snap_medicaid_ssi": int((t10["receives_snap"] | t10["receives_medicaid"] | t10["receives_ssi"]).sum()),
            "share_of_mass": float(cw[top10].sum() / cw.sum()),
        }
        contrast[name] = entry
    leverage.update({k: v for k, v in contrast.items() if k != "definitions"})
    _write_json(out_dir / "leverage.json", leverage)
    _write_json(out_dir / "household_contrast.json", contrast)
    _write_json(out_dir / "runtime.json", runtime)

    write_households(out_dir, desc, pool_ids, frozen_ids, w_elig, weights_by, exact_support_ids)
    render_readme(out_dir)
    validate_outputs(out_dir)
    _log(f"outputs written to {out_dir}")


# ---------------------------------------------------------------------------
# README rendering (generated blocks between markers; prose is hand-written)
# ---------------------------------------------------------------------------


def _pct(x, digits=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _num(x, digits=0) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def generated_blocks(out_dir: Path) -> dict[str, str]:
    load = lambda name: json.loads((out_dir / name).read_text())  # noqa: E731
    inputs, fit, conc = load("inputs.json"), load("fit.json"), load("weight_concentration.json")
    ek, sd, comp = load("exact_k.json"), load("sampling_distribution.json"), load("composition.json")
    overlap, contrast, runtime = load("overlap.json"), load("household_contrast.json"), load("runtime.json")
    leverage = load("leverage.json")
    blocks: dict[str, str] = {}
    target_keys = [key for key in ("national_all", "headline") if key in fit]
    st = inputs["settings"]
    fd = inputs["frozen_draw"]
    blocks["source"] = "\n".join([
        f"Source release: `{inputs['release_id']}`; artifact SHA-256 `{inputs['h5_sha256']}`; "
        f"maintained loader: `{inputs['loader']}`. Engine versions: `{json.dumps(inputs['engine'], sort_keys=True)}`.",
        "",
        f"The full file has {_num(inputs['pool']['households'])} households with total design weight "
        f"{_num(inputs['pool']['household_weight_total'], 1)}. PolicyBench's eligible pool has "
        f"{_num(inputs['pool']['eligible_households'])} households and {_pct(inputs['pool']['eligible_weight_share'])} "
        "of that weight. This scope difference matters: the primary self-weighted sample expands to the eligible pool, "
        "while exact-k starts from the full file. The fit tables include the sample rescaled to full-file mass as a sensitivity check.",
        "",
        f"Frozen snapshot: `{inputs['policybench_run_dir']}`; scenarios CSV SHA-256 "
        f"`{inputs['policybench_scenarios_csv_sha256']}`; metadata SHA-256 `{inputs['policybench_scenarios_meta_sha256']}`.",
        "",
        f"Producer source receipt: `{json.dumps(inputs.get('source', {}), sort_keys=True)}`.",
        "",
        "Smoke/pipeline-only output: " + str(inputs.get("smoke_households") is not None or inputs.get("pipeline_test_synthetic_measures", False)) + ".",
    ])

    reg = inputs["registry"]
    mat = inputs["materialization"]
    fam = ", ".join(f"{k} {v}" for k, v in mat["compiled_families"].items())
    blocks["targets"] = "\n".join(
        [
            f"- Registry compiled from the labelled feed: {reg['compiled_specs_all_geographies']:,} specs at all geographies; "
            f"{reg['national_specs_declared']} national ({reg['national_rule']}).",
            f"- Labelled filter keys that the recorded checkout's guard refuses: {reg['labelled_filter_restatements']['refused_by_origin_main_guard']} specs; "
            f"{reg['labelled_filter_restatements']['accepted_as_verified_restatements']} accepted as verified restatements, "
            f"{len(reg['labelled_filter_restatements']['excluded_specs'])} excluded.",
            f"- Materialized on the file: {mat['national_specs_compiled']} national targets "
            f"({len(mat['dropped_target_names'])} not materialized); used after the poverty/SPM exclusion: "
            f"{inputs['target_sets']['national_all']['targets_compiled_on_frame']}. Families used: {fam}.",
            f"- Headline subset: {len(inputs['headline_target_names'])} targets "
            f"({'; '.join(f'{k}: {v}' for k, v in inputs['headline_targets_per_rule'].items())}). "
            f"National AGI-by-band rows in the compiled set: {len(inputs['national_agi_by_band_rows'])}.",
            f"- Excluded under the no-poverty/SPM rule: {len(inputs['poverty_spm_check']['excluded_targets'])} "
            f"({'; '.join(f'`{n}` ({', '.join(v)})' for n, v in inputs['poverty_spm_check']['excluded_targets'].items()) or 'none'}). "
            f"After exclusion, targets naming poverty or SPM: {len(inputs['poverty_spm_check']['target_names_matching_poverty_or_spm_after_exclusion'])}; "
            f"measuring a poverty or SPM variable: {len(inputs['poverty_spm_check']['base_variables_matching_poverty_or_spm_after_exclusion'])}.",
            f"- Batch invariance: {mat['batch_invariance_check']['window_households']} households re-materialized across a batch boundary; "
            f"{mat['batch_invariance_check']['cells_differing']} of their values differ "
            f"(max absolute difference {mat['batch_invariance_check']['max_abs_difference']:g}).",
        ]
    )

    rc = fd["row_checks_file_vs_scenarios_csv"]
    rows = [[b, _num(v["eligible_households"]), str(v["reproduced_ids_equal_in_order"]), str(v["reproduced_overlap_with_frozen"])]
            for b, v in fd["by_filing_status_basis"].items()]
    blocks["frozen_draw"] = "\n".join(
        [
            f"IDs recovered through `{fd['route_to_ids']}`. The frozen public set came from "
            f"{fd['meta']['requested_num_scenarios']} requested draws with seed {fd['meta']['seed']}, "
            f"private fraction {fd['meta']['private_fraction']:g}, split seed {fd['meta']['split_seed']}, "
            f"leaving {fd['meta']['num_scenarios']} public scenarios. Reproduction: {fd['reproduction']}.",
            "",
            _table(["filing-status basis for eligibility", "eligible households", "draw reproduced exactly (ids, order)", "overlap with frozen 100"], rows),
            "",
            f"Row checks of the file against `scenarios.csv` for the {rc['of']} frozen households: state {rc['state_equal']}, "
            f"filing status {rc['filing_status_equal']}, adults {rc['adults_equal']}, children {rc['children_equal']}, "
            f"employment income within $1 {rc['employment_income_equal_within_1_dollar']}, "
            f"total income within $1 {rc['total_income_equal_within_1_dollar']}. "
            f"Exact equality counts: employment income {rc.get('employment_income_equal_exactly', 'unavailable')}, "
            f"total income {rc.get('total_income_equal_exactly', 'unavailable')}. "
            f"Largest requested-draw-count × p over eligible households: {fd['pps_first_order_inclusion']['max_requested_times_p']:.4f}.",
            "",
            f"Weight correction: {fd['pps_first_order_inclusion']['weight_used']}. "
            f"{fd['pps_first_order_inclusion']['note']}",
        ]
    )

    def fit_rows(key):
        f, c = fit[key], conc[key]
        spec = [
            (f"Full file ({inputs['pool']['households']:,} design weights)", "full_file", "full_file"),
            ("1. PPS sample, self-weighted", "design1_pps_self_weighted", "design1_pps_self_weighted"),
            ("1b. Same sample, full-file mass sensitivity", "design1b_pps_self_weighted_scaled_to_file_total", "design1_pps_self_weighted"),
            ("2. PPS sample, post-stratified", "design2_pps_post_stratified", "design2_pps_post_stratified"),
            ("3. Exact-k 100, refit", "design3_exact_k_refit", "design3_exact_k_refit"),
            ("3a. Exact-k 100, HT weights before refit", "design3_exact_k_ht_baseline_before_refit", "design3_exact_k_ht_baseline_before_refit"),
        ]
        out = []
        for label, fk, ck in spec:
            ff, cc = f.get(fk), c.get(ck)
            if ff is None:
                continue
            out.append([
                label, _pct(ff["uncapped_mape"]), _pct(ff["capped_mape"]), _pct(ff["share_within_10pct"]),
                _pct(ff["share_within_25pct"]), _num(cc["kish_ess"], 1) if cc else "n/a",
                _num(cc["max_over_mean"], 1) if cc else "n/a", _pct(cc["top10_share"]) if cc else "n/a",
            ])
        return out

    header = ["design", "MAPE (uncapped)", "capped MAPE (secondary)", "within 10%", "within 25%", "Kish ESS", "max/mean weight", "top-10 mass"]
    blocks["fit"] = "\n\n".join(
        f"{key} ({fit[key]['targets']} targets):\n\n" + _table(header, fit_rows(key)) for key in target_keys
    )

    wrows = []
    for key in target_keys:
        for design, label in (("design1_pps_self_weighted", "1. sampled, self-weighted"), ("design2_pps_post_stratified", "2. sampled, post-stratified"), ("design3_exact_k_refit", "3. exact-k, refit")):
            fk = {"design1_pps_self_weighted": "design1_pps_self_weighted", "design2_pps_post_stratified": "design2_pps_post_stratified", "design3_exact_k_refit": "design3_exact_k_refit"}[design]
            ff = fit[key].get(fk)
            if ff is None:
                continue
            worst = "; ".join(f"`{w['target']}` {_pct(w['relative_error'], 0)}" for w in ff["worst_five"])
            wrows.append([key, label, worst])
    blocks["worst"] = _table(["target set", "design", "worst five targets (signed scaled relative error)"], wrows)

    srows = []
    for key in target_keys:
        s = sd[key]
        for d, label in (("design1_self_weighted", "1. self-weighted"), ("design2_post_stratified", "2. post-stratified")):
            q = s[d]
            uq = q.get("uncapped_mape", {})
            srows.append([key, label, f"{_pct(uq.get('p05'))} / {_pct(uq.get('p50'))} / {_pct(uq.get('p95'))}",
                          f"{_pct(q['capped_mape']['p05'])} / {_pct(q['capped_mape']['p50'])} / {_pct(q['capped_mape']['p95'])}",
                          f"{_pct(q['within10']['p05'])} / {_pct(q['within10']['p50'])} / {_pct(q['within10']['p95'])}"])
    pos_rows = []
    for key in target_keys:
        p = sd[key]["frozen_draw_position"]
        pos_rows.append([key, "frozen draw, design 1", _pct(p.get("design1_uncapped_mape")), _pct(p.get("design1_uncapped_share_of_seeds_at_or_below"), 0), _pct(p["design1_within10"]), _pct(p["design1_share_of_seeds_within10_at_or_below"], 0)])
        pos_rows.append([key, "frozen draw, design 2", _pct(p.get("design2_uncapped_mape")), _pct(p.get("design2_uncapped_share_of_seeds_at_or_below"), 0), _pct(p["design2_within10"]), _pct(p["design2_share_of_seeds_within10_at_or_below"], 0)])
        e = sd[key].get("exact_k_position")
        if e:
            pos_rows.append([key, "exact-k refit vs design 2 seeds", _pct(e.get("uncapped_mape")), _pct(e.get("share_of_design2_seeds_uncapped_at_or_below"), 0), _pct(e["within10"]), _pct(e["share_of_design2_seeds_within10_at_or_below"], 0)])
    blocks["sampling"] = "\n".join(
        [f"{sd['run']['seeds_completed']} of {sd['run']['seeds_requested']} requested fresh seeds completed "
         f"(time budget {sd['run']['time_budget_minutes']:g} minutes; stopped on budget: {sd['run']['stopped_on_time_budget']}); "
         f"seeds from {sd['seed_generator']}; eligible pool {sd['eligible_households']:,} households.", "",
         f"Seed-count / stopping explanation: {sd['run'].get('reduction_reason') or sd['run'].get('termination_reason') or ('the recorded time budget was reached' if sd['run']['stopped_on_time_budget'] else 'all requested seeds completed')}. "
         f"Eligibility: {sd['eligibility_filter']}. Procedure: {sd['procedure']}", "",
         _table(["target set", "design", "uncapped MAPE p5 / p50 / p95", "capped MAPE p5 / p50 / p95", "within 10% p5 / p50 / p95"], srows), "",
         _table(["target set", "set", "uncapped MAPE", "MAPE: share of seeds at or below", "within 10%", "within-10%: share of seeds at or below"], pos_rows),
         "",
         f"Self-weighting check over the {sd['self_weighting_check']['draws']} seed draws (public inclusions by design-weight decile of the eligible pool):",
         "",
         _table(["decile"] + [str(r["decile"]) for r in sd["self_weighting_check"]["by_design_weight_decile"]],
                [["observed"] + [str(r["observed_inclusions"]) for r in sd["self_weighting_check"]["by_design_weight_decile"]],
                 ["expected (100 w / sum w per draw)"] + [_num(r["expected_under_self_weighting"], 1) for r in sd["self_weighting_check"]["by_design_weight_decile"]]])]
    )

    crow = []
    for key in target_keys:
        for design, label in (("full_file", "full file"), ("design1_pps_self_weighted", "1. sampled, self-weighted"), ("design2_pps_post_stratified", "2. sampled, post-stratified"), ("design3_exact_k_ht_baseline_before_refit", "3a. exact-k HT before refit"), ("design3_exact_k_refit", "3. exact-k, refit")):
            c = conc[key].get(design)
            if c is None or (key == "headline" and design in ("full_file", "design1_pps_self_weighted")):
                continue
            crow.append([key, label, _num(c["max_over_mean"], 1), _pct(c["top10_share"]), _num(c["kish_ess"], 1),
                         str(c["n_above_5x_own_median"]), str(c["n_above_10x_own_median"]),
                         str(c.get("n_above_5x_equal_share", "n/a")), str(c.get("n_above_10x_equal_share", "n/a"))])
    blocks["concentration"] = _table(
        ["target set", "design", "max/mean", "top-10 share", "Kish ESS", ">5x own median", ">10x own median", ">5x equal share", ">10x equal share"], crow
    )

    names = list(comp["designs"])
    short = {"full_population": "population", "sampled_self_weighted": "sampled (1)", "sampled_post_stratified_national_all": "sampled (2, national)",
             "sampled_post_stratified_headline": "sampled (2, headline)", "exact_k_national_all": "exact-k (national)", "exact_k_headline": "exact-k (headline)"}
    rows = []
    for group, cats in (("household_size", ["1", "2", "3", "4+"]), ("children", ["0", "1", "2+"]), ("head_age", ["<30", "30-49", "50-64", "65+"]),
                        ("employment_income_quintile", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
                        ("receipt", ["receives_snap", "receives_medicaid", "receives_ssi", "eitc_positive", "social_security_positive"])):
        for cat in cats:
            rows.append([f"{group} {cat}"] + [_pct(comp["designs"][n][group][cat]) for n in names])
    rows.append(["distinct states"] + [str(comp["designs"][n]["distinct_states"]) for n in names])
    cuts = ", ".join(_num(c) for c in comp["employment_income_quintile_cut_points_population"])
    blocks["composition"] = "\n".join([_table(["weighted share"] + [short.get(n, n) for n in names], rows), "",
                                       f"Employment-income quintile cut points (population, weighted): {cuts}.", "",
                                       "Ties stay in the same income band, so the full population's quintile shares need not be equal."])
    descriptor_rows = [
        [name, spec.get("spec_metadata", {}).get("base_variable", "see target"), spec["target"], spec["household_value"]]
        for name, spec in inputs["descriptor_columns_from_target_measures"].items()
    ]
    blocks["descriptors"] = "\n\n".join([
        _table(["output column", "base variable", "materialized target column", "household value"], descriptor_rows),
        "Structural descriptors: " + "; ".join(
            f"{name}: {value}" for name, value in comp["columns"].items() if isinstance(value, str)
        ) + ". The `agi_or_total_income` column uses adjusted gross income, aggregated to the household. "
        "It is distinct from the frozen scenario's sum of monetary income fields used in the reproduction check.",
        "Receipt flags describe positive modeled amounts or enrollment in this measurement frame. "
        "The head-age retirement proxy does not establish employment or retirement status.",
    ])

    orow = [[k.replace("_", " "), str(v)] for k, v in overlap.items()]
    blocks["overlap"] = _table(["overlap", "households"], orow)

    keys = [k for k in contrast if k != "definitions"]
    metrics = [
        ("one-person households", "one_person_households"), ("households of 6+", "household_size_6_plus"),
        ("head 65+", "head_65_plus"), ("head under 30", "head_under_30"), ("with children", "with_children"),
        ("zero employment income", "zero_employment_income"),
        ("employment income above population p90", "employment_income_above_population_p90"),
        ("employment income above population p99", "employment_income_above_population_p99"),
        ("SNAP", "receives_snap"), ("Medicaid", "receives_medicaid"), ("SSI", "receives_ssi"),
        ("EITC", "eitc_positive"), ("Social Security", "social_security_positive"),
        ("any of SNAP / Medicaid / SSI", "receives_any_of_snap_medicaid_ssi"),
        ("two or more of SNAP / Medicaid / SSI / EITC", "receives_two_or_more_of_snap_medicaid_ssi_eitc"),
        ("distinct states", "distinct_states"), ("in the top 1% of design weight", "in_top_1pct_design_weight"),
        ("PolicyBench-eligible", "policybench_eligible"),
    ]
    rows = [["households"] + [str(contrast[k]["households"]) for k in keys]]
    rows += [[label] + [str(contrast[k][m]) for k in keys] for label, m in metrics]
    rows.append(["median design-weight percentile"] + [_pct(contrast[k]["design_weight_percentile_median"], 0) for k in keys])
    rows.append(["max/mean weight"] + [_num(contrast[k]["weights"]["max_over_mean"], 1) for k in keys])
    rows.append(["top-10 share of weight"] + [_pct(contrast[k]["weights"]["top10_share"]) for k in keys])
    rows.append(["Kish ESS"] + [_num(contrast[k]["weights"]["kish_ess"], 1) for k in keys])
    rows.append(["weight / design weight, median"] + [_num(contrast[k]["calibrated_over_design_weight"]["p50"], 0) for k in keys])
    rows.append(["weight / design weight, max"] + [_num(contrast[k]["calibrated_over_design_weight"]["max"], 0) for k in keys])
    blocks["contrast"] = _table(["unweighted household count / weight diagnostic"] + keys, rows)
    if "exact_k_national_all" in contrast:
        sampled, selected = contrast["sampled"], contrast["exact_k_national_all"]
        blocks["contrast"] += (
            "\n\nSampled versus national-target exact-k: "
            f"{sampled['one_person_households']} versus {selected['one_person_households']} one-person households; "
            f"{sampled['head_65_plus']} versus {selected['head_65_plus']} retirement-age heads; "
            f"{sampled['employment_income_above_population_p90']} versus {selected['employment_income_above_population_p90']} "
            "households above the population's upper-decile employment-income threshold; "
            f"{sampled['receives_any_of_snap_medicaid_ssi']} versus {selected['receives_any_of_snap_medicaid_ssi']} "
            "receiving SNAP, Medicaid or SSI; "
            f"{sampled['distinct_states']} versus {selected['distinct_states']} states. "
            f"The top-weight households carry {_pct(sampled['weights']['top10_share'])} versus "
            f"{_pct(selected['weights']['top10_share'])} of their sets' total weight."
        )

    blocks["households_sampled"] = "Sampled household rows are in [households.csv](households.csv), under `design=sampled`, sorted by employment income."
    blocks["households_exact_k"] = (
        "National-target exact-k household rows are in [households.csv](households.csv), under `design=exact_k`, "
        "sorted by employment income. Identifiers and mirrored rows for the sampled and national-target exact-k sets "
        "are in [household_ids.json](household_ids.json); the headline-target sensitivity is reported only in aggregates."
        if "exact_k_national_all" in contrast
        else "The maintained exact-k path did not produce a national-target set; no replacement households were hand-selected."
    )
    blocks["leverage"] = "\n\n".join([
        f"Population design-weight threshold for the top percentile: {_num(leverage['design_weight_p99_threshold'], 2)}; "
        f"population median design weight: {_num(leverage['design_weight_median_population'], 2)}. "
        f"Weighted employment-income upper-decile threshold: ${_num(leverage['employment_income_p90_population_weighted'])}; "
        f"upper-percentile threshold: ${_num(leverage['employment_income_p99_population_weighted'])}.",
        *[
            f"{key}: {contrast[key]['in_top_1pct_design_weight']} selected households are in the top design-weight percentile, "
            f"{contrast[key]['one_person_households']} are one-person households, "
            f"{contrast[key]['employment_income_above_population_p99']} exceed the upper-percentile employment-income threshold, "
            f"and {contrast[key]['policybench_eligible']} satisfy PolicyBench eligibility. "
            f"The largest weights carry {_pct(contrast[key]['weights']['top10_share'])} of mass with Kish ESS "
            f"{_num(contrast[key]['weights']['kish_ess'], 1)}."
            for key in keys if key != "sampled"
        ],
        "Deduction: these diagnostics indicate how far a selected support behaves as integration points for the fitted "
        "aggregates (quadrature nodes). Concentration or unusual income support alone does not establish why any individual "
        "household was selected, and does not measure model-evaluation coverage.",
    ])

    xrows, prows = [], []
    for key in target_keys:
        run = ek["runs"].get(key, {})
        sel, draw = run.get("selection") or {}, run.get("draw") or {}
        search = sel.get("budget_search") or {}
        for p in search.get("probes") or sel.get("probes") or []:
            prows.append([key, f"{p['l0_lambda']:.4g}", str(p.get("measure"))])
        if run.get("selection_failed") or not sel.get("settled_l0_lambda"):
            xrows.append([key, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                          f"selection unavailable: {run.get('selection_failed', 'no completed selection receipt')}"])
            continue
        xrows.append([key, sel["budget_basis"], f"{sel['settled_l0_lambda']:.4g}", str(sel["budget_measure_at_settle"]),
                      _num(sel["open_probability_mass"], 1), str(sel["gates_above_deterministic_threshold"]),
                      str(sel["gates_open_prob_ge_pi_hi"]), str(draw.get("certainties", "n/a")), str(draw.get("boundary_draws", "n/a")),
                      str(draw.get("boundary_draws_at_or_below_deterministic_threshold", "n/a")),
                      f"{search.get('evaluations')} ({search.get('stopped_on')})", run.get("draw_failed") or "no"])
    parts = [
        _table(
            ["target set", "budget measure", "settled l0_lambda", "measure at settle", "open-probability mass",
             "gates open at eval (pi > threshold)", "gates with pi >= 0.95", "certainties drawn", "boundary draws",
             "boundary draws with pi <= threshold", "probes (search stopped on)", "draw failed"],
            xrows,
        ),
        "",
        "Budget-search probes (penalty, budget measure):",
        "",
        _table(["target set", "l0_lambda", "measure"], prows),
    ]
    for key in target_keys:
        draw = ek["runs"].get(key, {}).get("draw") or {}
        refit_conc = conc[key].get("design3_exact_k_refit")
        if draw and refit_conc:
            parts += [
                "",
                f"{key}: final draw count {draw['receipt']['k']}; positive refit weights {refit_conc['n']}; "
                f"boundary draw needed: {draw['boundary_draw_needed']}. "
                f"The boundary draw selected {draw['boundary_draws_at_or_below_deterministic_threshold']} households "
                "whose L0 gates were closed at deterministic evaluation. The final support is therefore a maintained "
                "probability-based boundary draw, rather than simply the households with open L0 gates.",
            ]
    blocks["exact_k"] = "\n".join(parts)

    rrows = []
    for name in ("engine_facts_child", "target_compile", "materialize_all_batches", "batch_invariance_child"):
        rec = runtime.get(name)
        if rec:
            rrows.append([name.replace("_", " "), _num(rec["seconds"], 1), _num(rec["peak_rss_gb"], 1)])
    for name, rec in runtime.get("per_calibration", {}).items():
        rrows.append([name.replace("_", " "), _num(rec["seconds"], 1), _num(rec["peak_rss_gb"], 1)])
    if runtime.get("sampling_distribution"):
        rrows.append([f"sampling distribution ({runtime['sampling_distribution']['calibrations']} calibrations)", _num(runtime["sampling_distribution"]["seconds"], 0), _num(runtime["sampling_distribution"]["peak_rss_gb"], 1)])
    rrows.append(["whole run", _num(runtime.get("whole_run_seconds"), 0), _num(runtime.get("parent_getrusage_high_water_gb"), 1) + " (parent high-water)"])
    blocks["runtime"] = "\n\n".join([
        _table(["step", "seconds", "peak RSS GB"], rrows),
        "Measurement definitions: " + "; ".join(f"{k}: {v}" for k, v in runtime.get("notes", {}).items()) + ".",
        "Main calibration timings are retained in [runtime.json](runtime.json); each sampling-refit timing and memory reading "
        "is in [sampling_distribution.json](sampling_distribution.json). Parent high-water RSS is distinct from simultaneous parent-plus-child RSS.",
    ])

    st = inputs["settings"]
    blocks["settings"] = "\n".join(
        [
            f"- Design 2 and the exact-k refit: `{json.dumps(st['sample_refit'])}`.",
            f"- Exact-k L0 selection: `{json.dumps(st['exact_k_selection'])}`; Sampford boundary draw seed {st['exact_k_draw_seed']}.",
            f"- Design 4: `{json.dumps(st['sampling_distribution'])}`.",
            f"- The primary MAPE is uncapped and uses {st['target_loss_weights']} target weights and scale `{st['fit_scale']}`. "
            f"The secondary capped MAPE caps each target's absolute relative error at {st['capped_mape_cap']:g}. "
            "The unit floor also defines the error for zero-valued targets; this is a scaled relative error, not an undefined division by zero.",
            f"- Certainty threshold for the exact-k draw: {st['exact_k_draw_pi_hi']:g}. "
            "L0 selection's weight-ratio bound is disabled when its setting is null. The post-stratification bound is relative "
            "to the sample's starting equal weights; the exact-k refit bound is relative to its normalized inclusion-corrected starting weights.",
        ]
    )
    summaries = []
    for key in target_keys:
        f1 = fit[key]["design1_pps_self_weighted"]
        f2 = fit[key]["design2_pps_post_stratified"]
        f3 = fit[key].get("design3_exact_k_refit")
        sentence = (
            f"On {key}, uncapped MAPE is {_pct(f1['uncapped_mape'])} for the frozen self-weighted sample "
            f"and {_pct(f2['uncapped_mape'])} after post-stratification"
        )
        sentence += f", versus {_pct(f3['uncapped_mape'])} for exact-k." if f3 else "; exact-k did not complete."
        summaries.append(sentence)
    blocks["reading"] = "\n\n".join(summaries + [
        "Deduction: self-weighting preserves an interpretable approximately equal-weight draw from PolicyBench's eligible population. "
        "Post-stratification can improve aggregate fit on those same households, but changes their influence. Exact-k spends the household "
        "budget on fitting the chosen aggregates across the full source population; its weight concentration, eligibility and receipt "
        "coverage determine the cost of that fit. The national and headline runs answer different target-set questions and should be compared separately.",
    ])
    return blocks


README_SKELETON = """# 100-record exact-k versus PolicyBench's sampled 100

This is a local evidence record comparing the frozen PolicyBench sample with a
support chosen by Microcosm's maintained L0/exact-k calibration path. It does
not evaluate models or certify either design for a benchmark or release.
Every reported result is rendered from the bundled JSONs by
`compare_designs.py render-readme`.

The household comparison is the centerpiece. Household rows are confined to
[households.csv](households.csv) and [household_ids.json](household_ids.json),
following the task's latest rule; this README reports their aggregate contrast.
The CSV groups the sampled and national-target exact-k sets and sorts each by
employment income. Both exports include only public-dataset household rows.

## Household comparison

<!-- BEGIN generated:contrast -->
<!-- END generated:contrast -->

<!-- BEGIN generated:households_sampled -->
<!-- END generated:households_sampled -->

<!-- BEGIN generated:households_exact_k -->
<!-- END generated:households_exact_k -->

<!-- BEGIN generated:leverage -->
<!-- END generated:leverage -->

## What the designs measure

- **PPS sample, self-weighted:** the frozen public sample, with each household
  assigned eligible population weight divided by sample size. This is an
  approximation to an inclusion-corrected estimator.
- **PPS sample, post-stratified:** the same households, refitted to each target
  set with ordinary calibration and no L0 selection.
- **Exact-k:** L0 selection from the full source population, the maintained
  exact-cardinality draw, and an ordinary refit on the selected support.
  `select_exact_k` returns support-aligned inclusion probabilities
  (`packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py:527`).
  `refit_l0_selection` starts from the selected original weights divided by
  those probabilities and normalized to full-file mass
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:2382`).
- **Sampling distribution:** repeated PPS draws with self-weighting and
  post-stratification, using the frozen draw's eligibility and public split.

The primary fit measure is uncapped MAPE with equal weight per target.
The denominator is the target's absolute value with a unit floor, matching
`default_target_loss_scales`
(`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:594`).
Capped MAPE is secondary. The fit tables evaluate each design against the same
compiled targets, including targets unsupported by a particular sampled set.
Kish effective sample size measures weight concentration, not model accuracy.

## Reading the results

<!-- BEGIN generated:reading -->
<!-- END generated:reading -->

## Source and reproduction

<!-- BEGIN generated:source -->
<!-- END generated:source -->

From the repository root, use the existing environment:

```bash
OMP_NUM_THREADS=4 .venv/bin/python experiments/policybench-exact-k-100-vs-sampled-20260922/compare_designs.py run
```

Run output belongs in a local ignored log. Each completed design is also
written to its own JSON before subsequent designs run. Re-render with the same
command using `render-readme` in place of `run`. The recorded settings and seed
count below are authoritative for this run.

## Settings

<!-- BEGIN generated:settings -->
<!-- END generated:settings -->

## Targets and the no-poverty/SPM check

<!-- BEGIN generated:targets -->
<!-- END generated:targets -->

Poverty and SPM measures are excluded from fitting and selection; the audit
above covers names and source/filter variables. Structural SPM-unit counts
appear only in PolicyBench's pre-existing eligibility rule. They are not a
poverty measure. The household descriptors are post-selection diagnostics.

## Frozen sample verification

The reference sampling and split implementations were read at
`policybench/scenarios.py:1020` (eligibility),
`policybench/scenarios.py:1044` (PPS draw), and
`policybench/scenarios.py:1498` (public/private split) in the read-only
PolicyBench checkout.

<!-- BEGIN generated:frozen_draw -->
<!-- END generated:frozen_draw -->

## Exact-k selection

<!-- BEGIN generated:exact_k -->
<!-- END generated:exact_k -->

Deterministically open L0 gates and the final exact-k support are distinct:
the maintained boundary draw can select positive-probability records whose
L0 gates are closed at evaluation. No households are hand-selected to force
the requested count. A failed maintained selection or draw remains a failed
result, with its available search receipts retained.

## Fit against shared targets

<!-- BEGIN generated:fit -->
<!-- END generated:fit -->

<!-- BEGIN generated:worst -->
<!-- END generated:worst -->

## Sampling distribution

<!-- BEGIN generated:sampling -->
<!-- END generated:sampling -->

Lower MAPE is better; a lower MAPE percentile means fewer random samples
achieved equally low error. Higher within-tolerance share is better, so its
percentile has the opposite interpretation. This distribution varies the PPS
sample seed; it is not an uncertainty interval for exact-k optimization.

## Weight concentration

<!-- BEGIN generated:concentration -->
<!-- END generated:concentration -->

Own-median thresholds use each design's positive output weights. Equal-share
thresholds use eligible-pool mass divided by sample size for the sampled designs
and full-file mass divided by support size for exact-k.

## Weighted composition

<!-- BEGIN generated:composition -->
<!-- END generated:composition -->

## Household column definitions

<!-- BEGIN generated:descriptors -->
<!-- END generated:descriptors -->

## Overlap

<!-- BEGIN generated:overlap -->
<!-- END generated:overlap -->

## Runtime and memory

<!-- BEGIN generated:runtime -->
<!-- END generated:runtime -->

## What was not measured

No PolicyBench harness or language model was run. Aggregate target fit and
household coverage do not establish model-ranking stability, policy-reform
accuracy, out-of-sample performance, or release readiness. The self-weighting
approximation does not recover exact first-order inclusion probabilities for
the PPS-without-replacement draw followed by its fixed public split. Current
engine-materialized descriptors are not a claim to reproduce every frozen
engine output. Earlier unlogged failed attempts are not evidence for this
record's calibration outcomes.
"""


def render_readme(out_dir: Path) -> None:
    path = out_dir / "README.md"
    text = path.read_text() if path.exists() else README_SKELETON
    for name, body in generated_blocks(out_dir).items():
        pattern = re.compile(
            rf"(<!-- BEGIN generated:{re.escape(name)} -->\n).*?(<!-- END generated:{re.escape(name)} -->)", re.S
        )
        if not pattern.search(text):
            raise RuntimeError(f"README has no generated block {name!r}")
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    path.write_text(text)


def _verify_producer_source(out_dir: Path, expected_sha256: str) -> dict[str, bool]:
    """Verify the producer directly or reconstruct it from reporting-only edits.

    Revision offsets are zero-based line indexes and line strings retain their
    endings. Both sides of each edit must match; the reconstructed source must
    have the original numerical producer's digest recorded in inputs.json.
    """
    current_bytes = Path(__file__).read_bytes()
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    if current_sha256 == expected_sha256:
        return {"source.script_sha256": True}
    revision_path = out_dir / "source_revision.json"
    if not revision_path.exists():
        return {"source.script_sha256": False}
    revision = json.loads(revision_path.read_text())
    checks = {
        "source.revision_current_sha256": revision["current_sha256"] == current_sha256,
        "source.revision_producer_sha256": revision["producer_sha256"] == expected_sha256,
    }
    edits = revision["edits"]
    current = current_bytes.decode("utf-8").splitlines(keepends=True)
    if not isinstance(edits, list) or not edits:
        raise ValueError("Source revision must contain edits")
    for edit in edits:
        for name in ("old_start", "new_start"):
            if type(edit[name]) is not int or edit[name] < 0:
                raise ValueError("Source revision offsets must be nonnegative integers")
        for name in ("old_lines", "new_lines"):
            if not isinstance(edit[name], list) or not all(isinstance(line, str) for line in edit[name]):
                raise ValueError("Source revision lines must be string lists")
    edits = sorted(edits, key=lambda edit: edit["new_start"])
    end = 0
    patch_matches = True
    for edit in edits:
        start = edit["new_start"]
        stop = start + len(edit["new_lines"])
        patch_matches &= start >= end and stop <= len(current) and current[start:stop] == edit["new_lines"]
        end = stop
    checks["source.revision_current_hunks_match"] = patch_matches
    if not patch_matches:
        checks["source.script_sha256"] = False
        return checks
    reconstructed = current.copy()
    for edit in reversed(edits):
        start = edit["new_start"]
        reconstructed[start:start + len(edit["new_lines"])] = edit["old_lines"]
    end = 0
    original_matches = True
    for edit in sorted(edits, key=lambda edit: edit["old_start"]):
        start = edit["old_start"]
        stop = start + len(edit["old_lines"])
        original_matches &= start >= end and stop <= len(reconstructed) and reconstructed[start:stop] == edit["old_lines"]
        end = stop
    checks["source.revision_producer_hunks_match"] = original_matches
    reconstructed_sha256 = hashlib.sha256("".join(reconstructed).encode("utf-8")).hexdigest()
    checks["source.script_sha256"] = reconstructed_sha256 == expected_sha256
    return checks


def validate_outputs(out_dir: Path, *, verify_source: bool = True) -> dict:
    """Check the completed record without emitting household values or IDs.

    Only aggregate verdicts are written to validation.json. Rounded household
    exports are compared at a relative tolerance of 1e-4 for concentration;
    their JSON mirror is checked at the precision actually exported to CSV.
    """
    out_dir = Path(out_dir).resolve()
    if REPO_ROOT not in out_dir.parents:
        raise ValueError("Validation output must stay inside the assigned workspace")
    checks: dict[str, bool] = {}
    skipped: list[str] = []

    def check(label: str, condition) -> None:
        checks[label] = bool(condition)

    section = "input_files"
    try:
        data = {
            name: json.loads((out_dir / f"{name}.json").read_text())
            for name in (
                "inputs", "fit", "weight_concentration", "household_ids",
                "household_contrast", "sampling_distribution", "exact_k",
            )
        }
        inputs = data["inputs"]
        smoke = inputs.get("smoke_households") is not None
        hh = pd.read_csv(out_dir / "households.csv")
        ids = data["household_ids"]
        section = "household_exports"
        check("household_rows_total", len(hh) == 2 * K)
        check("household_design_labels", set(hh["design"]) == {"sampled", "exact_k"})
        check("household_sizes", (hh["household_size"] == hh["adults"] + hh["children"]).all())
        mirror = pd.DataFrame(ids["rows"])
        check("household_json_mirror_columns", set(mirror.columns) == set(hh.columns))
        check("household_json_mirror_count", len(mirror) == len(hh))
        if set(mirror.columns) == set(hh.columns) and len(mirror) == len(hh):
            for column in hh.columns:
                left, right = hh[column], mirror[column]
                missing_match = np.array_equal(left.isna(), right.isna())
                present = left.notna() & right.notna()
                if pd.api.types.is_numeric_dtype(left.dtype):
                    values_match = np.allclose(
                        left[present].to_numpy(dtype=float), right[present].to_numpy(dtype=float),
                        rtol=0, atol=1e-8,
                    )
                else:
                    values_match = np.array_equal(left[present].astype(str), right[present].astype(str))
                check(f"household_json_mirror.{column}", missing_match and values_match)

        receipt_columns = ("receives_snap", "receives_medicaid", "receives_ssi", "eitc_positive", "social_security_positive")
        design_keys = (
            ("sampled", "sampled_frozen_policybench_public_100_in_draw_order", "sampled", "design1_pps_self_weighted"),
            ("exact_k", "exact_k_national_all_100", "exact_k_national_all", "design3_exact_k_refit"),
        )
        for design, id_key, contrast_key, concentration_key in design_keys:
            section = f"household_exports.{design}"
            sub = hh[hh["design"] == design]
            exported_ids = sub["household_id"].tolist()
            check(f"{design}.cardinality", len(sub) == K and sub["household_id"].nunique() == K)
            check(f"{design}.identifier_list", len(ids[id_key]) == K and len(set(ids[id_key])) == K and set(exported_ids) == set(ids[id_key]))
            check(f"{design}.income_order", sub["employment_income"].is_monotonic_increasing)
            weights = sub["calibrated_weight"].to_numpy(dtype=float)
            valid_weights = np.isfinite(weights).all() and (weights > 0).all()
            check(f"{design}.positive_finite_weights", valid_weights)
            expected = {
                "households": len(sub),
                "one_person_households": int((sub["household_size"] == 1).sum()),
                "household_size_6_plus": int((sub["household_size"] >= 6).sum()),
                "head_65_plus": int((sub["head_age"] >= 65).sum()),
                "head_under_30": int((sub["head_age"] < 30).sum()),
                "with_children": int((sub["children"] > 0).sum()),
                "distinct_states": sub["state"].nunique(),
            }
            flags = {}
            for column in receipt_columns:
                flags[column] = sub[column].astype(str).str.lower()
                check(f"{design}.boolean.{column}", flags[column].isin(["true", "false"]).all())
                expected[column] = int((flags[column] == "true").sum())
            benefits = np.column_stack([(flags[c] == "true").to_numpy() for c in receipt_columns[:3]])
            expected["receives_any_of_snap_medicaid_ssi"] = int(benefits.any(axis=1).sum())
            expected["receives_two_or_more_of_snap_medicaid_ssi_eitc"] = int(
                ((benefits.sum(axis=1) + (flags["eitc_positive"] == "true").to_numpy()) >= 2).sum()
            )
            contrast = data["household_contrast"][contrast_key]
            for metric, value in expected.items():
                check(f"{design}.contrast.{metric}", value == contrast[metric])
            if valid_weights and len(weights):
                total = float(weights.sum())
                concentration = {
                    "n": len(weights), "total": total,
                    "max_over_mean": float(weights.max() / weights.mean()),
                    "min_over_mean": float(weights.min() / weights.mean()),
                    "top10_share": float(np.sort(weights)[-10:].sum() / total),
                    "kish_ess": float(total**2 / np.square(weights).sum()),
                }
                bundled = data["weight_concentration"]["national_all"][concentration_key]
                for metric, value in concentration.items():
                    check(f"{design}.concentration.{metric}", math.isclose(value, bundled[metric], rel_tol=1e-4, abs_tol=1e-4))

        section = "targets"
        prohibited = inputs["poverty_spm_check"]
        for key in ("target_names_matching_poverty_or_spm_after_exclusion", "base_variables_matching_poverty_or_spm_after_exclusion"):
            check(f"targets.{key}", prohibited[key] == [])
        for target_set in ("national_all", "headline"):
            target_receipt = inputs["target_sets"][target_set]
            target_count = target_receipt["targets_compiled_on_frame"]
            check(f"targets.{target_set}.count", target_count > 0 and target_count == data["fit"][target_set]["targets"])
            names = target_receipt.get("target_names")
            if names is None and smoke:
                skipped.append(f"targets.{target_set}.all_names_unavailable_in_earlier_smoke")
            else:
                check(f"targets.{target_set}.all_names", names is not None and len(names) == target_count and len(set(names)) == target_count)
                check(f"targets.{target_set}.no_poverty_spm_names", names is not None and all("poverty" not in n.lower() and "spm" not in n.lower() for n in names))
            for design, fit in data["fit"][target_set].items():
                if isinstance(fit, dict):
                    check(f"fit.{target_set}.{design}.target_count", fit["n_targets"] == target_count)
                    check(f"fit.{target_set}.{design}.tolerance_shares", 0 <= fit["share_within_10pct"] <= fit["share_within_25pct"] <= 1)
            check(f"exact_k.{target_set}.cardinality", data["exact_k"]["runs"][target_set]["draw"]["receipt"]["k"] == K)

        section = "sampling_distribution"
        sd = data["sampling_distribution"]
        seeds = sd["seeds"]
        count = sd["run"]["seeds_completed"]
        check("sampling.seed_count", count > 0 and count == len(seeds) == len(set(seeds)))
        check("sampling.fresh_seeds", inputs["frozen_draw"]["meta"]["seed"] not in seeds)
        for target_set in ("national_all", "headline"):
            for short, summary_key in (("design1", "design1_self_weighted"), ("design2", "design2_post_stratified")):
                samples = sd[target_set]["per_seed"][short]
                label = f"sampling.{target_set}.{short}"
                check(f"{label}.seed_alignment", [row["seed"] for row in samples] == seeds)
                for metric in ("uncapped_mape", "capped_mape", "within10", "within25"):
                    values = np.asarray([row[metric] for row in samples], dtype=float)
                    check(f"{label}.{metric}.finite", np.isfinite(values).all())
                    expected = _quantiles(values)
                    actual = sd[target_set][summary_key][metric]
                    check(f"{label}.{metric}.quantiles", all(math.isclose(value, actual[key], rel_tol=1e-12, abs_tol=1e-12) for key, value in expected.items()))
                if short == "design2":
                    records = [row.get("runtime", {}) for row in samples]
                    check(f"{label}.individual_runtime", all(
                        isinstance(rec.get("seconds"), (int, float)) and math.isfinite(rec["seconds"]) and rec["seconds"] >= 0
                        and isinstance(rec.get("peak_rss_gb"), (int, float)) and math.isfinite(rec["peak_rss_gb"]) and 0 < rec["peak_rss_gb"] < 32
                        for rec in records
                    ))

        section = "readme"
        readme = (out_dir / "README.md").read_text()
        for name, body in generated_blocks(out_dir).items():
            matches = re.findall(
                rf"<!-- BEGIN generated:{re.escape(name)} -->\n(.*?)<!-- END generated:{re.escape(name)} -->",
                readme, re.S,
            )
            check(f"readme.generated.{name}", matches == [body + "\n"])
        section = "source_provenance"
        if verify_source:
            for label, passed in _verify_producer_source(out_dir, inputs["source"]["script_sha256"]).items():
                check(label, passed)
        else:
            skipped.append("source.script_sha256_explicitly_not_checked")
    except Exception as exc:  # Never include an exception's potentially row-valued message.
        check(f"schema.{section}.{type(exc).__name__}", False)

    failures = [label for label, passed in checks.items() if not passed]
    result = {
        "status": "passed" if not failures else "failed",
        "checks_count": len(checks), "passed_count": len(checks) - len(failures),
        "failed_checks": failures, "skipped_checks": skipped, "checks": checks,
        "concentration_relative_tolerance": 1e-4,
        "concentration_absolute_tolerance": 1e-4,
        "source_verification_requested": verify_source,
    }
    _write_json(out_dir / "validation.json", result)
    if failures:
        raise RuntimeError("Output validation failed: " + ", ".join(failures)) from None
    return result


def main(argv=None) -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_engine_child":
        from multiprocessing.connection import Connection

        targets = {f.__name__: f for f in (_child_engine_facts, _child_materialize)}
        target = targets[sys.argv[2]]
        conn = Connection(int(sys.argv[3]), readable=False, writable=True)
        args = pickle.load(sys.stdin.buffer)
        with ResourceMeter():
            target(conn, *args)
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["run", "render-readme", "validate"])
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--feed-dir", default=str(DEFAULT_FEED))
    parser.add_argument("--policybench-run", default=str(DEFAULT_POLICYBENCH_RUN))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stop-after", choices=["materialize", "designs", "sampling"], default=None)
    parser.add_argument("--smoke-households", type=int, default=None)
    parser.add_argument("--pipeline-test-synthetic-measures", action="store_true")
    parser.add_argument("--sampling-budget-minutes", type=float, default=60.0)
    parser.add_argument("--total-budget-minutes", type=float, default=150.0)
    parser.add_argument("--n-seeds", type=int, default=SAMPLING_DISTRIBUTION["n_seeds"])
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else RECORD_DIR
    if REPO_ROOT not in out_dir.parents:
        raise SystemExit("Outputs must stay inside the assigned workspace")
    if args.stage == "render-readme":
        render_readme(out_dir)
    elif args.stage == "validate":
        result = validate_outputs(out_dir)
        _log(f"validation: {result['passed_count']}/{result['checks_count']} checks passed")
    else:
        if args.n_seeds < 1 or args.n_batches < 2:
            raise SystemExit("Require at least one seed and two materialization batches")
        _log(f"starting run pid={os.getpid()} out_dir={out_dir}")
        try:
            with ResourceMeter(label="run"):
                run(args)
        except BaseException as exc:
            if out_dir.exists():
                _write_json(out_dir / "run_status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "smoke": args.smoke_households is not None})
            raise


if __name__ == "__main__":
    main()
