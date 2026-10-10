"""Replay the release builder's re-runs of base stages on a base frame.

The release builder (``tools/build_us_fiscal_refresh_release.py``) loads the
base H5 and, when it is not consuming a validated pool, runs several base
stages again before calibration. This script imports the builder module from
the source tree that built the release, loads the base with the builder's own
``_load_frame``, and replays those stages under named variants. After each
variant it compares the stage outputs with the values stored in the base H5
and in the release H5, matched by entity id.

Variants (``--variant``; repeatable):

``isolated``
    Each of weeks unemployed, childcare, energy subsidy and retirement
    contributions run once on the base frame exactly as loaded. A stage that
    reproduces its own stored output here is deterministic on this platform
    for an unchanged input.
``release_order``
    The release builder's sequence: weeks unemployed, population mass repair,
    Social Security component repair (the builder's function, fed the four
    recorded SSA targets), QBI reconciliation, childcare, energy subsidy,
    retirement contributions. A stage whose output equals the release's stored
    column here is reproduced by this replay.
``mass_only``
    Population mass repair, then retirement contributions.
``ss_only``
    The recorded Social Security component factors applied to the base frame
    (no mass repair), then childcare, energy subsidy and retirement
    contributions.
``isolated_t4``
    Retirement contributions on the base frame as loaded, with BLAS/OpenMP
    pools limited to 4 threads and four draw workers (the base build's Modal
    thread environment).

Usage:
    python replay_release_reimputation.py --builder-root ROOT --base-h5 BASE \
        --release-h5 RELEASE --release-diagnostics CALIBRATION_DIAGNOSTICS \
        --asec-2023-source PPPUB23_ZIP_OR_DIR --out OUT_JSON --variant NAME
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

STAGE_OUTPUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "weeks_unemployed": ("person", ("weeks_unemployed",)),
    "childcare": ("spm_unit", ("spm_unit_pre_subsidy_childcare_expenses",)),
    "energy_subsidy": ("spm_unit", ("spm_unit_energy_subsidy",)),
    "retirement_contributions": (
        "person",
        (
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ),
    ),
}
ID_COLUMN = {"person": "person_id", "spm_unit": "spm_unit_id"}
SOCIAL_SECURITY_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_builder(root: Path):
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release",
        root / "tools" / "build_us_fiscal_refresh_release.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def channel_labels(table: pd.DataFrame, entity: str) -> np.ndarray:
    channel = table[f"{entity}_support_channel"].map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    )
    clone = table[f"{entity}_support_clone_index"].astype(int).astype(str)
    return (channel + "/clone" + clone).to_numpy()


def compare(
    replayed: np.ndarray, stored: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    changed = replayed != stored
    out: dict[str, object] = {
        "rows_differing": int(changed.sum()),
        "rows_differing_by_channel": {
            str(key): int(value)
            for key, value in pd.Series(changed).groupby(labels).sum().items()
        },
    }
    if changed.any():
        a = stored[changed]
        b = replayed[changed]
        out["max_abs_difference"] = float(np.max(np.abs(b - a)))
        out["min_abs_difference"] = float(np.min(np.abs(b - a)))
        out["direction_stored_to_replayed"] = {
            "zero_to_nonzero": int(np.count_nonzero((a == 0) & (b != 0))),
            "nonzero_to_zero": int(np.count_nonzero((a != 0) & (b == 0))),
            "up_both_nonzero": int(np.count_nonzero((a != 0) & (b != 0) & (b > a))),
            "down_both_nonzero": int(np.count_nonzero((a != 0) & (b != 0) & (b < a))),
        }
    return out


class Recorder:
    def __init__(self, frame0, release_h5: Path, out: Path, header: dict) -> None:
        self.base_tables = {
            entity: frame0.table(entity).set_index(ID_COLUMN[entity], drop=False)
            for entity in ID_COLUMN
        }
        self.labels = {
            entity: channel_labels(table, entity)
            for entity, table in self.base_tables.items()
        }
        self.release = {}
        for entity, columns in (
            (
                "person",
                [
                    c
                    for s, (e, cs) in STAGE_OUTPUTS.items()
                    if e == "person"
                    for c in cs
                ],
            ),
            (
                "spm_unit",
                [
                    c
                    for s, (e, cs) in STAGE_OUTPUTS.items()
                    if e == "spm_unit"
                    for c in cs
                ],
            ),
        ):
            table = pd.read_hdf(
                release_h5, entity, columns=[ID_COLUMN[entity], *columns]
            )
            self.release[entity] = table.set_index(ID_COLUMN[entity]).loc[
                self.base_tables[entity].index
            ]
        self.out = out
        self.receipt = dict(header)
        self.receipt["results"] = []

    def record(
        self, variant: str, step: str, stage: str, frame, seconds: float
    ) -> None:
        entity, columns = STAGE_OUTPUTS[stage]
        table = (
            frame.table(entity)
            .set_index(ID_COLUMN[entity], drop=False)
            .loc[self.base_tables[entity].index]
        )
        labels = self.labels[entity]
        result = {
            "variant": variant,
            "step": step,
            "stage": stage,
            "seconds": round(seconds, 1),
            "columns": {},
        }
        for column in columns:
            replayed = pd.to_numeric(table[column]).to_numpy(dtype=np.float64)
            base = pd.to_numeric(self.base_tables[entity][column]).to_numpy(
                dtype=np.float64
            )
            release = pd.to_numeric(self.release[entity][column]).to_numpy(
                dtype=np.float64
            )
            result["columns"][column] = {
                "replayed_vs_base": compare(replayed, base, labels),
                "replayed_vs_release": compare(replayed, release, labels),
            }
        self.receipt["results"].append(result)
        self.flush()
        summary = {
            column: (
                values["replayed_vs_base"]["rows_differing"],
                values["replayed_vs_release"]["rows_differing"],
            )
            for column, values in result["columns"].items()
        }
        print(
            f"[{variant}] {step} {stage} {seconds:.0f}s (vs base, vs release): {summary}",
            flush=True,
        )

    def note(self, key: str, value) -> None:
        self.receipt.setdefault("notes", {})[key] = value
        self.flush()

    def flush(self) -> None:
        self.out.write_text(json.dumps(self.receipt, indent=1, sort_keys=True) + "\n")


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def ss_targets_from_diagnostics(builder, diagnostics: dict) -> list[SimpleNamespace]:
    components = diagnostics["build"]["support_value_repairs"][
        "social_security_components"
    ]["components"]
    role_by_column = {
        column: role
        for role, column in builder.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.items()
    }
    return [
        SimpleNamespace(
            metadata={"target_role": role_by_column[column]}, value=payload["target"]
        )
        for column, payload in components.items()
    ]


def with_recorded_ss_factors(frame, diagnostics: dict):
    from microcosm.frame import Frame

    components = diagnostics["build"]["support_value_repairs"][
        "social_security_components"
    ]["components"]
    person = frame.table("person").copy()
    for column, payload in components.items():
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0)
        person[column] = values.to_numpy(dtype=np.float64) * float(payload["factor"])
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder-root", type=Path, required=True)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--release-h5", type=Path, required=True)
    parser.add_argument("--release-diagnostics", type=Path, required=True)
    parser.add_argument("--asec-2023-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--period", type=int, default=2024)
    args = parser.parse_args()

    builder = load_builder(args.builder_root)
    diagnostics = json.loads(args.release_diagnostics.read_text())
    base_sha = sha256(args.base_h5)
    header = {
        "builder_root": str(args.builder_root),
        "builder_head": (args.builder_root / ".git").exists()
        and os.popen(f"git -C '{args.builder_root}' rev-parse HEAD").read().strip(),
        "base_h5": str(args.base_h5),
        "base_h5_sha256": base_sha,
        "release_h5": str(args.release_h5),
        "release_h5_sha256": sha256(args.release_h5),
        "release_diagnostics": str(args.release_diagnostics),
        "release_diagnostics_sha256": sha256(args.release_diagnostics),
        "variants": args.variant,
        "seed": args.seed,
        "period": args.period,
        "platform": {
            "python": sys.version,
            "machine": platform.machine(),
            "system": platform.system(),
            "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "quantile_forest": __import__("quantile_forest").__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "POPULACE_FIT_N_JOBS",
                    "POPULACE_FIT_PREDICT_WORKERS",
                    "PYTHONHASHSEED",
                )
            },
            "cpu_count": os.cpu_count(),
        },
    }

    frame0, load_seconds = timed(
        builder._load_frame, args.base_h5, expected_sha256=base_sha
    )
    recorder = Recorder(frame0, args.release_h5, args.out, header)
    recorder.note("load_seconds", round(load_seconds, 1))
    seed, period = args.seed, args.period

    def run(stage: str, frame):
        if stage == "weeks_unemployed":
            source = builder.load_asec_2023_weeks_unemployed_source(
                args.asec_2023_source
            )
            return timed(
                builder.with_us_weeks_unemployed,
                frame,
                seed=seed,
                time_period=period,
                asec_2023_source=source,
            )
        if stage == "childcare":
            return timed(
                builder.with_us_childcare_inputs,
                frame,
                seed=seed,
                time_period=period,
                allow_existing_without_source=True,
            )
        if stage == "energy_subsidy":
            return timed(
                builder.with_us_energy_subsidy_input,
                frame,
                seed=seed,
                time_period=period,
                allow_existing_without_source=True,
            )
        if stage == "retirement_contributions":
            return timed(
                builder.with_us_retirement_contribution_inputs,
                frame,
                seed=seed,
                time_period=period,
            )
        raise ValueError(stage)

    for variant in args.variant:
        if variant == "isolated":
            for stage in (
                "weeks_unemployed",
                "childcare",
                "energy_subsidy",
                "retirement_contributions",
            ):
                frame, seconds = run(stage, frame0)
                recorder.record(
                    variant, f"base_as_loaded->{stage}", stage, frame, seconds
                )
                del frame
        elif variant == "isolated_t4":
            from threadpoolctl import threadpool_limits

            os.environ["POPULACE_FIT_PREDICT_WORKERS"] = "4"
            with threadpool_limits(limits=4):
                frame, seconds = run("retirement_contributions", frame0)
            del os.environ["POPULACE_FIT_PREDICT_WORKERS"]
            recorder.record(
                variant,
                "base_as_loaded->retirement_contributions[threads=4]",
                "retirement_contributions",
                frame,
                seconds,
            )
            del frame
        elif variant == "release_order":
            frame, seconds = run("weeks_unemployed", frame0)
            recorder.record(
                variant, "1 weeks_unemployed", "weeks_unemployed", frame, seconds
            )
            frame, mass = builder._with_base_population_mass_repair(frame)
            recorder.note("release_order_mass_repair", mass)
            frame, ss = builder._with_social_security_component_value_repair(
                frame, ss_targets_from_diagnostics(builder, diagnostics)
            )
            recorded = diagnostics["build"]["support_value_repairs"][
                "social_security_components"
            ]["components"]
            recorder.note(
                "release_order_ss_factors_match_recorded",
                {
                    column: {
                        "replayed": ss["components"][column]["factor"],
                        "recorded": recorded[column]["factor"],
                        "equal": ss["components"][column]["factor"]
                        == recorded[column]["factor"],
                    }
                    for column in recorded
                },
            )
            frame = builder.with_us_qbi_input_reconciliation(frame)
            for index, stage in enumerate(
                ("childcare", "energy_subsidy", "retirement_contributions"), start=2
            ):
                frame, seconds = run(stage, frame)
                recorder.record(variant, f"{index} {stage}", stage, frame, seconds)
            del frame
        elif variant == "mass_only":
            frame, mass = builder._with_base_population_mass_repair(frame0)
            recorder.note("mass_only_mass_repair", mass)
            out, seconds = run("retirement_contributions", frame)
            recorder.record(
                variant,
                "mass_repair->retirement_contributions",
                "retirement_contributions",
                out,
                seconds,
            )
            del frame, out
        elif variant == "ss_only":
            frame = with_recorded_ss_factors(frame0, diagnostics)
            for stage in ("childcare", "energy_subsidy", "retirement_contributions"):
                out, seconds = run(stage, frame)
                recorder.record(
                    variant, f"recorded_ss_factors->{stage}", stage, out, seconds
                )
                del out
            del frame
        else:
            raise ValueError(f"unknown variant {variant!r}")
    recorder.note("completed", True)


if __name__ == "__main__":
    main()
