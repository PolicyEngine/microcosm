"""Receipt for microcosm#297: CPS educational attainment against ACS S1501.

Runs the build's attainment stage on a published US release, compiles the ACS
2024 1-year S1501 state attainment facts through the US target compiler, and
materializes every compiled target with the fiscal-refresh tool's own
materializer. It writes:

- ``experiments/receipts/297-cps-educational-attainment-s1501.json``: every
  compiled cell's S1501 value, the release's weighted count, the support
  (records and households carrying the cell) and a standalone per-state
  entropy calibration of the release's household weights to the cells;
- ``experiments/297-cps-educational-attainment-receipt.md``: the summary;
- ``<sidecar-dir>/populace_us_2024_cps_educational_attainment.csv.gz``:
  ``person_id``, ``A_HGA`` and ``cps_educational_attainment`` for every person
  of the release, so an analysis can join the column now. The sidecar is build
  evidence, not a release, and is not committed.

The S1501 facts are a TEMPORARY draft input (PolicyEngine/chronicle#347,
unmerged); ``--s1501-raw`` is the Census response that package pins, and every
compiled value is checked against it. Nothing here publishes, stages or
certifies anything.

    python experiments/build_297_attainment_receipt.py --h5 populace_us_2024.h5 \\
        --asec-archive 2022=asecpub23csv.zip --asec-archive 2023=asecpub24csv.zip \\
        --asec-archive 2024=asecpub25csv.zip --s1501-facts consumer_facts.jsonl \\
        --s1501-raw acs_S1501_state_2024.json --sidecar-dir out --run <run id>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.ledger_targets import compile_ledger_target_references
from microcosm.build.us_runtime import educational_attainment as ea
from microcosm.build.us_runtime import fiscal_targets
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA

REPO = Path(__file__).resolve().parents[1]
RECEIPT_JSON = (
    REPO / "experiments" / "receipts" / "297-cps-educational-attainment-s1501.json"
)
RECEIPT_MD = REPO / "experiments" / "297-cps-educational-attainment-receipt.md"
HOST_RELEASE = "populace-us-2024-spm-20260915"
TARGET_PERIOD = 2024
#: S1501_C01 estimate ids of the population-25-and-over rows, by category.
S1501_25_PLUS_COLUMNS = {
    "all": "S1501_C01_006E",
    ea.LESS_THAN_9TH_GRADE: "S1501_C01_007E",
    ea.GRADE_9_TO_12_NO_DIPLOMA: "S1501_C01_008E",
    ea.HIGH_SCHOOL_GRADUATE: "S1501_C01_009E",
    ea.SOME_COLLEGE_NO_DEGREE: "S1501_C01_010E",
    ea.ASSOCIATES_DEGREE: "S1501_C01_011E",
    ea.BACHELORS_DEGREE: "S1501_C01_012E",
    ea.GRADUATE_OR_PROFESSIONAL_DEGREE: "S1501_C01_013E",
}
S1501_18_TO_24_COLUMNS = {
    "all": "S1501_C01_001E",
    "less_than_high_school_graduate": "S1501_C01_002E",
    ea.HIGH_SCHOOL_GRADUATE: "S1501_C01_003E",
    "some_college_or_associates_degree": "S1501_C01_004E",
    "bachelors_degree_or_higher": "S1501_C01_005E",
}
SHORT = {
    ea.LESS_THAN_9TH_GRADE: "< 9th grade",
    ea.GRADE_9_TO_12_NO_DIPLOMA: "9th-12th, no diploma",
    ea.HIGH_SCHOOL_GRADUATE: "High school or GED",
    ea.SOME_COLLEGE_NO_DEGREE: "Some college, no degree",
    ea.ASSOCIATES_DEGREE: "Associate's",
    ea.BACHELORS_DEGREE: "Bachelor's",
    ea.GRADUATE_OR_PROFESSIONAL_DEGREE: "Graduate or professional",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_tool():
    path = REPO / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_host(path: Path) -> Frame:
    """The release's entity tables, reduced to what the stage and receipt read."""

    person_columns = [
        "person_id",
        "person_household_id",
        "person_tax_unit_id",
        "person_spm_unit_id",
        "person_family_id",
        "person_marital_unit_id",
        "source_year",
        "PERIDNUM",
        "source_household_id",
        "P_SEQ",
        "A_LINENO",
        "A_AGE",
        "age",
        "person_support_channel",
    ]
    with pd.HDFStore(path, "r") as store:
        person = store.select("person", columns=person_columns)
        household = store.select(
            "household", columns=["household_id", "household_weight", "state_fips"]
        )
        groups = {
            entity: store.select(entity, columns=[f"{entity}_id"])
            for entity in ("tax_unit", "spm_unit", "family", "marital_unit")
        }
    person = person.reset_index(drop=True)
    household = household.reset_index(drop=True)
    weights = household["household_weight"].to_numpy(dtype=np.float64)
    return Frame(
        {
            "person": person,
            "household": household.drop(columns="household_weight"),
            **{
                entity: table.reset_index(drop=True) for entity, table in groups.items()
            },
        },
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def _raw_s1501(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["response"]["data"] if "response" in payload else payload
    raw = pd.DataFrame(rows[1:], columns=rows[0])
    raw["state_fips"] = raw["GEO_ID"].str.removeprefix("0400000US")
    return raw.set_index("state_fips")


def _facts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _entropy_calibrate(
    base: np.ndarray, matrix: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Minimum-KL household weights hitting ``matrix @ w = targets`` exactly.

    Newton on the dual with a backtracking line search; ``matrix`` is
    constraints x households.
    """

    scale = np.maximum(np.abs(targets), 1.0)
    a = matrix / scale[:, None]
    t = targets / scale
    lam = np.zeros(len(t))

    def dual(values: np.ndarray) -> float:
        return float(np.sum(base * np.exp(np.clip(a.T @ values, -50, 50))) - t @ values)

    iterations = 0
    for iterations in range(1, 201):  # noqa: B007
        w = base * np.exp(np.clip(a.T @ lam, -50, 50))
        gradient = a @ w - t
        if np.max(np.abs(gradient)) < 1e-10:
            break
        hessian = (a * w) @ a.T + 1e-12 * np.eye(len(t))
        step = np.linalg.solve(hessian, gradient)
        size, current = 1.0, dual(lam)
        while dual(lam - size * step) > current - 1e-4 * size * gradient @ step:
            size /= 2
            if size < 1e-12:
                break
        lam = lam - size * step
    w = base * np.exp(np.clip(a.T @ lam, -50, 50))
    ratio = w / base
    return w, {
        "iterations": iterations,
        "max_abs_relative_error": float(np.max(np.abs(matrix @ w - targets) / scale)),
        "weight_ratio_p01": float(np.quantile(ratio, 0.01)),
        "weight_ratio_p50": float(np.quantile(ratio, 0.50)),
        "weight_ratio_p99": float(np.quantile(ratio, 0.99)),
        "weight_ratio_min": float(ratio.min()),
        "weight_ratio_max": float(ratio.max()),
        "kish_ess_before": float(base.sum() ** 2 / np.sum(base**2)),
        "kish_ess_after": float(w.sum() ** 2 / np.sum(w**2)),
    }


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--asec-archive", action="append", default=[], required=True)
    parser.add_argument("--s1501-facts", type=Path, required=True)
    parser.add_argument("--s1501-raw", type=Path, required=True)
    parser.add_argument("--s1501-facts-source", default="")
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--run", default="")
    args = parser.parse_args()
    tool = _load_tool()

    # 1. The build stage on the published release.
    frame = _read_host(args.h5)
    archives = tool._asec_attainment_source_paths(args.asec_archive)
    sources = ea.load_asec_attainment_sources(
        frame.table("person")["source_year"].astype(int).unique(), archives
    )
    frame, record = ea.with_us_cps_educational_attainment(frame, sources=sources)
    gate = ea.us_cps_educational_attainment_signal_gate(frame)
    person = frame.table("person")
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    household_position = pd.Series(
        np.arange(len(household)), index=household["household_id"].to_numpy()
    )
    person_household = household_position.reindex(
        person["person_household_id"].to_numpy()
    ).to_numpy()
    person_weight = weights[person_household]
    person_state = household["state_fips"].to_numpy()[person_household]
    classes = person[ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN].to_numpy(dtype=object)
    age = person["age"].to_numpy(dtype=np.float64)

    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = args.sidecar_dir / "populace_us_2024_cps_educational_attainment.csv.gz"
    person[["person_id", "A_HGA", ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN]].to_csv(
        sidecar, index=False, compression={"method": "gzip", "mtime": 0}
    )

    # 2. The compile path on the draft S1501 facts.
    facts = _facts(args.s1501_facts)
    references = fiscal_targets._dynamic_us_fiscal_target_references(
        tuple(facts), target_period=TARGET_PERIOD
    )
    registry = compile_ledger_target_references(facts, references, country="us")
    specs = [
        spec
        for spec in registry.specs
        if spec.family == ea.US_EDUCATIONAL_ATTAINMENT_TARGET_FAMILY
    ]
    if len(specs) != len(registry.specs):
        raise SystemExit("S1501 facts compiled into a family other than attainment.")
    unsupported = tool._unsupported_ledger_filter_metadata(specs)
    if unsupported:
        raise SystemExit(f"The tool would refuse these targets: {unsupported}")

    # 3. Every compiled value against the raw Census response.
    raw = _raw_s1501(args.s1501_raw)
    raw_mismatches = []
    for spec in specs:
        fips = spec.metadata["state_fips"]
        category = spec.metadata["educational_attainment_category"]
        published = float(raw.loc[fips, S1501_25_PLUS_COLUMNS[category]])
        if published != float(spec.value):
            raw_mismatches.append((spec.name, published, float(spec.value)))
    partition_gaps = {
        fips: float(raw.loc[fips, S1501_25_PLUS_COLUMNS["all"]])
        - sum(
            float(raw.loc[fips, column])
            for category, column in S1501_25_PLUS_COLUMNS.items()
            if category != "all"
        )
        for fips in US_STATE_FIPS_TO_POSTAL
    }

    # 4. The tool's materializer, cell by cell.
    cells = []
    for spec in specs:
        values = tool._population_attainment_household_values(
            frame=frame, household=household, age=age, metadata=spec.metadata
        )
        indicator = ea.attainment_target_person_indicator(classes, age, spec.metadata)
        state = int(spec.metadata["state_fips"])
        in_cell = (indicator > 0) & (person_state == state)
        estimate = float(weights @ values)
        direct = float(person_weight[in_cell].sum())
        if not np.isclose(estimate, direct, rtol=1e-12):
            raise SystemExit(f"{spec.name}: materializer {estimate} != direct {direct}")
        cells.append(
            {
                "name": spec.name,
                "state_fips": spec.metadata["state_fips"],
                "state": US_STATE_FIPS_TO_POSTAL[spec.metadata["state_fips"]],
                "category": spec.metadata["educational_attainment_category"],
                "target": float(spec.value),
                "estimate": estimate,
                "relative_error": estimate / float(spec.value) - 1.0,
                "records": int(in_cell.sum()),
                "households": int(np.unique(person_household[in_cell]).size),
            }
        )
    table = pd.DataFrame(cells)
    total_target = table.groupby("state_fips")["target"].transform("sum")
    total_estimate = table.groupby("state_fips")["estimate"].transform("sum")
    table["target_share"] = table["target"] / total_target
    table["estimate_share"] = table["estimate"] / total_estimate
    table["share_error_pp"] = 100 * (table["estimate_share"] - table["target_share"])

    # 5. The 18-to-24 universe, held out (not compiled).
    held_out = []
    for fips in US_STATE_FIPS_TO_POSTAL:
        mask = (person_state == int(fips)) & (age >= 18) & (age < 25)
        total = float(person_weight[mask].sum())
        for category, column in S1501_18_TO_24_COLUMNS.items():
            if category == "all":
                continue
            members = ea.S1501_ATTAINMENT_CATEGORY_CLASSES[category]
            in_cat = mask & np.isin(classes, sorted(members))
            held_out.append(
                {
                    "state_fips": fips,
                    "category": category,
                    "target_share": float(raw.loc[fips, column])
                    / float(raw.loc[fips, S1501_18_TO_24_COLUMNS["all"]]),
                    "estimate_share": float(person_weight[in_cat].sum()) / total,
                }
            )
    held = pd.DataFrame(held_out)
    held["share_error_pp"] = 100 * (held["estimate_share"] - held["target_share"])

    # 6. A standalone per-state entropy calibration to the cells, keeping each
    # state's population under 25 fixed. Not the release solve: it shows what
    # binding these cells alone asks of the weights.
    calibration = {}
    calibrated_weights = weights.copy()
    category_order = list(ea.US_EDUCATIONAL_ATTAINMENT_CLASSES)
    for fips in US_STATE_FIPS_TO_POSTAL:
        state = int(fips)
        households = np.flatnonzero(household["state_fips"].to_numpy() == state)
        local = pd.Series(np.arange(len(households)), index=households)
        in_state = person_state == state
        rows = np.zeros((len(category_order) + 1, len(households)))
        hh_index = local.reindex(person_household[in_state]).to_numpy()
        state_classes = classes[in_state]
        state_age = age[in_state]
        for row, label in enumerate(category_order):
            np.add.at(
                rows[row],
                hh_index,
                ((state_classes == label) & (state_age >= 25)) * 1.0,
            )
        np.add.at(rows[-1], hh_index, (state_age < 25) * 1.0)
        state_cells = table.loc[table["state_fips"] == fips].set_index("category")
        targets = np.array(
            [*state_cells.loc[category_order, "target"], rows[-1] @ weights[households]]
        )
        fitted, summary = _entropy_calibrate(weights[households], rows, targets)
        calibrated_weights[households] = fitted
        before = float(rows.sum(axis=0) @ weights[households])
        after = float(rows.sum(axis=0) @ fitted)
        summary["state_population_change"] = after / before - 1.0
        calibration[fips] = summary
    calibrated_estimate = []
    for spec, cell in zip(specs, cells, strict=True):
        values = tool._population_attainment_household_values(
            frame=frame, household=household, age=age, metadata=spec.metadata
        )
        calibrated_estimate.append(
            float(calibrated_weights @ values) / cell["target"] - 1
        )
    table["calibrated_relative_error"] = calibrated_estimate

    # 7. National and ROI-class summaries.
    national = (
        table.groupby("category")[["target", "estimate"]].sum().reindex(category_order)
    )
    national["target_share"] = national["target"] / national["target"].sum()
    national["estimate_share"] = national["estimate"] / national["estimate"].sum()
    roi = {
        role: {
            "class": label,
            "share_error_pp_mean_abs": float(
                table.loc[table["category"] == label, "share_error_pp"].abs().mean()
            ),
            "share_error_pp_max_abs": float(
                table.loc[table["category"] == label, "share_error_pp"].abs().max()
            ),
            "relative_error_median_abs": float(
                table.loc[table["category"] == label, "relative_error"].abs().median()
            ),
            "records_min": int(table.loc[table["category"] == label, "records"].min()),
        }
        for role, label in ea.US_DEGREE_ROI_CLASSES.items()
    }
    level_gap = (
        table.groupby("state_fips")[["target", "estimate"]]
        .sum()
        .pipe(lambda frame_: frame_["target"] / frame_["estimate"] - 1.0)
    )
    payload = {
        "schema_version": 1,
        "issue": "PolicyEngine/microcosm#297",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "modal_run": args.run,
        "host": {
            "release": HOST_RELEASE,
            "path": args.h5.name,
            "sha256": _sha256(args.h5),
            "persons": len(person),
            "households": len(household),
        },
        "s1501": {
            "status": "TEMPORARY draft input: PolicyEngine/chronicle#347 is unmerged",
            "source": args.s1501_facts_source,
            "facts_sha256": _sha256(args.s1501_facts),
            "facts": len(facts),
            "raw_sha256": _sha256(args.s1501_raw),
            "compiled_targets": len(specs),
            "raw_value_mismatches": raw_mismatches,
            "max_abs_partition_gap": max(abs(gap) for gap in partition_gaps.values()),
        },
        "stage": {
            "derivation": record,
            "signal_gate_passed": gate.passed,
            "signal_gate_failures": list(gate.failures),
            "signal_gate": dict(gate.details),
        },
        "sidecar": {
            "file": sidecar.name,
            "sha256": _sha256(sidecar),
            "rows": len(person),
            "columns": ["person_id", "A_HGA", ea.US_CPS_EDUCATIONAL_ATTAINMENT_COLUMN],
        },
        "national_25_plus": {
            label: {
                "target": float(national.loc[label, "target"]),
                "estimate": float(national.loc[label, "estimate"]),
                "target_share": float(national.loc[label, "target_share"]),
                "estimate_share": float(national.loc[label, "estimate_share"]),
            }
            for label in category_order
        },
        "state_25_plus_level_gap": {
            "description": "S1501 25+ total / release 25+ weighted population - 1",
            "by_state": {
                US_STATE_FIPS_TO_POSTAL[k]: float(v) for k, v in level_gap.items()
            },
            "max_abs": float(level_gap.abs().max()),
            "median_abs": float(level_gap.abs().median()),
        },
        "cells": {
            "share_error_pp_mean_abs": float(table["share_error_pp"].abs().mean()),
            "share_error_pp_p90_abs": float(
                table["share_error_pp"].abs().quantile(0.9)
            ),
            "share_error_pp_max_abs": float(table["share_error_pp"].abs().max()),
            "relative_error_median_abs": float(table["relative_error"].abs().median()),
            "relative_error_p90_abs": float(
                table["relative_error"].abs().quantile(0.9)
            ),
            "relative_error_max_abs": float(table["relative_error"].abs().max()),
            "records_min": int(table["records"].min()),
            "households_min": int(table["households"].min()),
            "rows": table.to_dict(orient="records"),
        },
        "roi_classes": roi,
        "held_out_18_to_24": {
            "reason": (
                "Not compiled: the ACS counts college students in dormitories "
                "(group quarters) in the state of their college, the CPS in their "
                "parents' household."
            ),
            "share_error_pp_mean_abs": float(held["share_error_pp"].abs().mean()),
            "share_error_pp_max_abs": float(held["share_error_pp"].abs().max()),
            "national_by_category": {
                category: float(
                    held.loc[held["category"] == category, "share_error_pp"].mean()
                )
                for category in held["category"].unique()
            },
        },
        "standalone_calibration": {
            "description": (
                "Per-state minimum-KL reweighting of the release's household "
                "weights to the seven 25+ cells, holding the state's population "
                "under 25 fixed. Not the release solve, which fits every target "
                "jointly; it shows what binding these cells alone asks of the "
                "weights."
            ),
            "max_abs_relative_error_after": float(
                table["calibrated_relative_error"].abs().max()
            ),
            "weight_ratio_p01_min": min(
                v["weight_ratio_p01"] for v in calibration.values()
            ),
            "weight_ratio_p99_max": max(
                v["weight_ratio_p99"] for v in calibration.values()
            ),
            "weight_ratio_min": min(
                v["weight_ratio_min"] for v in calibration.values()
            ),
            "weight_ratio_max": max(
                v["weight_ratio_max"] for v in calibration.values()
            ),
            "kish_ess_ratio_min": min(
                v["kish_ess_after"] / v["kish_ess_before"] for v in calibration.values()
            ),
            "by_state": {US_STATE_FIPS_TO_POSTAL[k]: v for k, v in calibration.items()},
        },
    }
    RECEIPT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    RECEIPT_MD.write_text(
        _markdown(payload, table, national, held), encoding="utf-8", newline="\n"
    )
    print(f"wrote {RECEIPT_JSON.relative_to(REPO)} and {RECEIPT_MD.relative_to(REPO)}")


def _markdown(payload, table, national, held) -> str:
    cells = payload["cells"]
    gap = payload["state_25_plus_level_gap"]
    calibration = payload["standalone_calibration"]
    lines = [
        "# microcosm#297 receipt: CPS educational attainment against ACS S1501",
        "",
        f"Generated {payload['generated_at'][:10]} by "
        "`experiments/build_297_attainment_receipt.py` at commit "
        f"`{payload['commit']}`, Modal run `{payload['modal_run']}` "
        "(`experiments/modal_297.py`). Machine-readable copy: "
        "`experiments/receipts/297-cps-educational-attainment-s1501.json`.",
        "",
        f"The build stage ran on the published release `{payload['host']['release']}` "
        f"({payload['host']['persons']:,} persons, {payload['host']['households']:,} "
        "households) at its calibrated weights. Every person joined its own Census "
        "member row; the signal gate "
        f"{'passed' if payload['stage']['signal_gate_passed'] else 'FAILED'}. "
        "The S1501 facts are a **temporary draft input** "
        "(PolicyEngine/chronicle#347, unmerged): "
        f"{payload['s1501']['facts']} facts, {payload['s1501']['compiled_targets']} "
        "compiled targets (51 states and DC x 7 classes, population 25 and over), "
        f"{len(payload['s1501']['raw_value_mismatches'])} mismatches against the raw "
        "Census response, largest gap between a state's categories and its total "
        f"{payload['s1501']['max_abs_partition_gap']:.0f}.",
        "",
        "## Classes",
        "",
        "| Class | CPS ASEC `A_HGA` | ACS PUMS `SCHL` | Six-class view |",
        "|---|---|---|---|",
        "| `less_than_9th_grade` | 31-34 | 01-11 | less than high school |",
        "| `9th_to_12th_grade_no_diploma` | 35-38 | 12-15 | less than high school |",
        "| `high_school_graduate_includes_equivalency` | 39 | 16, 17 | high school or GED |",
        "| `some_college_no_degree` | 40 | 18, 19 | some college, no degree |",
        "| `associates_degree` | 41, 42 | 20 | associate's |",
        "| `bachelors_degree` | 43 | 21 | bachelor's |",
        "| `graduate_or_professional_degree` | 44, 45, 46 | 22, 23, 24 | graduate or professional |",
        "| `not_in_universe` | 0 (Children) | blank, or under 15 | |",
        "",
        "## National, population 25 and over (sum of the 51 state cells)",
        "",
        "| Class | S1501 | Release | S1501 share | Release share |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in national.iterrows():
        lines.append(
            f"| {SHORT[label]} | {row['target'] / 1e6:.2f}M | {row['estimate'] / 1e6:.2f}M "
            f"| {_pct(row['target_share'])} | {_pct(row['estimate_share'])} |"
        )
    lines += [
        "",
        "## State cells (357)",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Mean abs share error | {cells['share_error_pp_mean_abs']:.2f} pp |",
        f"| 90th percentile abs share error | {cells['share_error_pp_p90_abs']:.2f} pp |",
        f"| Max abs share error | {cells['share_error_pp_max_abs']:.2f} pp |",
        f"| Median abs level error | {_pct(cells['relative_error_median_abs'])} |",
        f"| 90th percentile abs level error | {_pct(cells['relative_error_p90_abs'])} |",
        f"| Max abs level error | {_pct(cells['relative_error_max_abs'])} |",
        f"| Fewest records in a cell | {cells['records_min']} |",
        f"| Fewest households in a cell | {cells['households_min']} |",
        f"| S1501 25+ total vs release 25+ population, median abs | {_pct(gap['median_abs'], 2)} |",
        f"| S1501 25+ total vs release 25+ population, max abs | {_pct(gap['max_abs'], 2)} |",
        "",
        "Share error is the release's share of a state's population 25 and over in a "
        "class minus S1501's; level error compares counts.",
        "",
        "### Classes the degree-ROI analysis uses",
        "",
        "Two-year degree = `associates_degree`, four-year degree = "
        "`bachelors_degree`, each against the "
        "`high_school_graduate_includes_equivalency` baseline.",
        "",
        "| Role | Class | Mean abs share error | Max abs share error | Median abs level error | Fewest records |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role, row in payload["roi_classes"].items():
        lines.append(
            f"| {role} | `{row['class']}` | {row['share_error_pp_mean_abs']:.2f} pp "
            f"| {row['share_error_pp_max_abs']:.2f} pp "
            f"| {_pct(row['relative_error_median_abs'])} | {row['records_min']} |"
        )
    worst = table.reindex(
        table["share_error_pp"].abs().sort_values(ascending=False).index
    ).head(10)
    lines += [
        "",
        "### Largest share errors",
        "",
        "| State | Class | S1501 share | Release share | Error | Records |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in worst.iterrows():
        lines.append(
            f"| {row['state']} | {SHORT[row['category']]} | {_pct(row['target_share'])} "
            f"| {_pct(row['estimate_share'])} | {row['share_error_pp']:+.2f} pp "
            f"| {row['records']} |"
        )
    merged_release = float(
        national.loc[
            [ea.SOME_COLLEGE_NO_DEGREE, ea.ASSOCIATES_DEGREE], "estimate_share"
        ].sum()
    )
    merged_s1501 = float(
        national.loc[
            [ea.SOME_COLLEGE_NO_DEGREE, ea.ASSOCIATES_DEGREE], "target_share"
        ].sum()
    )
    some = table[table["category"] == ea.SOME_COLLEGE_NO_DEGREE]
    assoc = table[table["category"] == ea.ASSOCIATES_DEGREE]
    thin = table[table["records"] < 30]
    worst_state, worst_ratio = max(
        (
            (state, values["weight_ratio_max"])
            for state, values in calibration["by_state"].items()
        ),
        key=lambda item: item[1],
    )
    lines += [
        "",
        "### Reading the comparison",
        "",
        "- **Some college against associate's is an instrument difference.** "
        f"Nationally the release has {_pct(national.loc[ea.SOME_COLLEGE_NO_DEGREE, 'estimate_share'])} "
        "some college, no degree against S1501's "
        f"{_pct(national.loc[ea.SOME_COLLEGE_NO_DEGREE, 'target_share'])}, and "
        f"{_pct(national.loc[ea.ASSOCIATES_DEGREE, 'estimate_share'])} associate's against "
        f"{_pct(national.loc[ea.ASSOCIATES_DEGREE, 'target_share'])}. The release is below "
        f"S1501 on some college in {int((some['share_error_pp'] < 0).sum())} of 51 states "
        f"and above it on associate's in {int((assoc['share_error_pp'] > 0).sum())}. The "
        "gap has one sign almost everywhere, so it reads as a difference between "
        "the instruments rather than sampling: the CPS interviews by telephone or "
        "visit and offers occupational and academic associate's answers, the ACS is "
        'mostly self-response and includes group quarters (Census, "Educational '
        'Attainment Fact Sheet", '
        "https://www.census.gov/topics/education/educational-attainment/guidance/"
        "factsheet-acs-cps.html). Binding the two cells separately would move "
        "weight to settle a definition. Their sum is "
        f"{100 * (merged_release - merged_s1501):+.1f} pp off nationally "
        f"({_pct(merged_release)} against {_pct(merged_s1501)}), less than the "
        "some-college cell alone; the merged rows below give the state spread.",
        "- **Thin cells.** "
        f"{len(thin)} cells have fewer than 30 records; "
        f"{int((thin['category'] == ea.LESS_THAN_9TH_GRADE).sum())} of them are "
        "`less_than_9th_grade`. The standalone calibration's largest weight ratio "
        f"({worst_ratio:.0f}) is in {worst_state}. Merging the two classes below a "
        "diploma (the six-class view) thickens those cells.",
    ]
    merged = (
        table.assign(
            merged=table["category"].map(
                {
                    ea.SOME_COLLEGE_NO_DEGREE: "some_college_or_associates",
                    ea.ASSOCIATES_DEGREE: "some_college_or_associates",
                    ea.LESS_THAN_9TH_GRADE: "less_than_high_school",
                    ea.GRADE_9_TO_12_NO_DIPLOMA: "less_than_high_school",
                }
            )
        )
        .dropna(subset=["merged"])
        .groupby(["merged", "state_fips"])[["target_share", "estimate_share"]]
        .sum()
    )
    merged["error"] = 100 * (merged["estimate_share"] - merged["target_share"])
    lines += [
        "",
        "| Merged row | Mean abs share error | Max abs share error |",
        "|---|---:|---:|",
    ]
    for name, group in merged.groupby(level=0):
        lines.append(
            f"| {name} | {group['error'].abs().mean():.2f} pp "
            f"| {group['error'].abs().max():.2f} pp |"
        )
    held_national = payload["held_out_18_to_24"]["national_by_category"]
    lines += [
        "",
        "## Held out: population 18 to 24",
        "",
        payload["held_out_18_to_24"]["reason"],
        f"Mean abs share error {payload['held_out_18_to_24']['share_error_pp_mean_abs']:.2f} pp, "
        f"max {payload['held_out_18_to_24']['share_error_pp_max_abs']:.2f} pp; mean "
        "signed error by category: "
        + ", ".join(f"{k} {v:+.2f} pp" for k, v in held_national.items())
        + ".",
        "",
        "## Standalone calibration to the cells",
        "",
        calibration["description"],
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Max abs level error after | {_pct(calibration['max_abs_relative_error_after'], 4)} |",
        f"| Weight ratio, lowest state 1st percentile | {calibration['weight_ratio_p01_min']:.3f} |",
        f"| Weight ratio, highest state 99th percentile | {calibration['weight_ratio_p99_max']:.3f} |",
        f"| Weight ratio, min / max | {calibration['weight_ratio_min']:.3f} / {calibration['weight_ratio_max']:.3f} |",
        f"| Kish effective sample size kept, lowest state | {_pct(calibration['kish_ess_ratio_min'])} |",
        "",
        "## SCF education-debt imputation (microcosm#1073, PR #1246)",
        "",
        "The #1073 hold-out already measured SCF `educ` as a predictor: "
        "`build_plus_education` raised the household rank correlation (0.168 to 0.207) "
        "but moved the share with debt from -0.30 pp to -0.92 pp. With this column "
        "the host side of that predictor exists: map SCF `educ` (reference person) "
        "and the host reference person's `cps_educational_attainment` to the same "
        "seven classes, and condition the QRFs on it. The share error suggests "
        "adding it with the share calibrated to FSA borrower counts "
        "(microcosm#1074) rather than as a bare predictor. Nothing here changes #1246.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
