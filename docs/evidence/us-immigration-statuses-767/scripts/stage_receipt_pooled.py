"""Pooled default-pool stage receipt for the US immigration stage (#767, #779).

Runs the US immigration source stage alone on the three Census ASEC person
files of main's default pool (income years 2023, 2024 and 2025: the 2024, 2025
and 2026 ASEC, ``asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS``), then runs the
composition gate on its output and writes a JSON receipt of aggregates.

- Each archive must match the SHA-256 that
  ``us_runtime/education_assistance_source.py`` pins for it.
- Weights (``--weighting household``, the default) follow
  ``asec_pool.pool_asec_sources`` with ``anchor_year=2024``, as
  ``build_pooled_asec_unit_frame`` calls it for target year 2024: each person
  carries the household's ``HSUP_WGT`` times 0.01, and each file is rescaled to
  one third of income year 2024's weighted person population.
  ``--weighting person`` uses ``MARSUPWT / 100 / 3`` instead.
- Draw keys use the pool's full lineage: ``source_year`` (income year),
  ``source_household_id`` (``PH_SEQ``) and ``source_person_id`` (``PERIDNUM``).
- Seed 0 and period 2024, Route A's ``--target-year``.

A refusal is recorded as the receipt's ``error`` instead of raised. Main's tree
has no humanitarian controls, Pew universe or composition gate, so its receipt
carries the composition only.

The archives are Census's public-use ASEC CSV releases, at the URLs
``education_assistance_source.py`` names:

    https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip
    https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip
    https://www2.census.gov/programs-surveys/cps/datasets/2026/march/asecpub26csv.zip

Run it with any interpreter that has the workspace's dependencies, pointing at
the tree to measure (a checkout, or ``git archive <commit> packages`` extracted):

    .venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/stage_receipt_pooled.py \\
        <tree> --archive-dir <dir with the three zips> --out receipt.json

``--labels-out`` also writes the row-level labels (a pickle) for
cross-tabulating two trees' runs. Those rows come from the public files but are
record-level, so they stay out of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PERIOD = 2024
ANCHOR_INCOME_YEAR = 2024
ARCHIVES = (
    (
        2023,
        "asecpub24csv.zip",
        "pppub24.csv",
        "hhpub24.csv",
        "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
    ),
    (
        2024,
        "asecpub25csv.zip",
        "pppub25.csv",
        "hhpub25.csv",
        "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
    ),
    (
        2025,
        "asecpub26csv.zip",
        "pppub26.csv",
        "hhpub26.csv",
        "fe819d0d2fc4470c282e76c247b8aa8b6054811619730307e9f2fe6186707d93",
    ),
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("src_root", type=Path, help="tree whose packages/*/src to run")
parser.add_argument(
    "--archive-dir",
    type=Path,
    action="append",
    required=True,
    help="directory holding ASEC zips (repeatable; each zip is found by name)",
)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--weighting", choices=("household", "person"), default="household")
parser.add_argument(
    "--source-commit",
    help="commit the tree came from, when src_root is not a git checkout",
)
parser.add_argument("--labels-out", type=Path)
args = parser.parse_args()

ROOT = args.src_root.resolve()
OUT = args.out.resolve()
for _src in sorted((ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.immigration as immigration
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

assert Path(immigration.__file__).resolve().is_relative_to(ROOT), immigration.__file__


def _archive_path(name: str) -> Path:
    found = [d / name for d in args.archive_dir if (d / name).is_file()]
    if not found:
        raise SystemExit(f"{name} is in none of {args.archive_dir}")
    return found[0]


def _git(*command: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *command], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


git_top = _git("rev-parse", "--show-toplevel")
if git_top is not None and Path(git_top).resolve() == ROOT:
    source_commit = _git("rev-parse", "HEAD")
    source_commit_from = "git"
    source_tree_clean = _git("status", "--porcelain", "--", "packages") == ""
    if args.source_commit and args.source_commit != source_commit:
        raise SystemExit(
            f"--source-commit {args.source_commit} is not the checkout's "
            f"HEAD {source_commit}"
        )
else:
    source_commit = args.source_commit
    source_commit_from = "argument" if source_commit else None
    source_tree_clean = None

columns = list(
    dict.fromkeys(
        [
            *immigration.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS,
            "MARSUPWT",
            "PERIDNUM",
            "PH_SEQ",
        ]
    )
)
parts = []
archives = []
raw_population = {}
for income_year, archive_name, member, household_member, pinned in ARCHIVES:
    path = _archive_path(archive_name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == pinned, (path, digest)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name = next(n for n in names if n.lower().endswith(member))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, usecols=columns)
        name = next(n for n in names if n.lower().endswith(household_member))
        with archive.open(name) as handle:
            household = pd.read_csv(handle, usecols=["H_SEQ", "HSUP_WGT"])
    household_weight = household.drop_duplicates("H_SEQ").set_index("H_SEQ")["HSUP_WGT"]
    frame["household_weight"] = frame["PH_SEQ"].map(household_weight) * 0.01
    assert not frame["household_weight"].isna().any(), income_year
    frame["source_year"] = income_year
    frame["source_household_id"] = frame["PH_SEQ"].to_numpy()
    frame["source_person_id"] = frame["PERIDNUM"].astype(str)
    raw_population[income_year] = float(frame["household_weight"].sum())
    parts.append(frame)
    archives.append(
        {
            "income_year": income_year,
            "archive": archive_name,
            "sha256": digest,
            "rows": int(len(frame)),
            "peinusyr_max": int(frame["PEINUSYR"].max()),
            "household_weighted_persons": raw_population[income_year],
            "marsupwt_persons": float(frame["MARSUPWT"].sum() / 100.0),
        }
    )
target_population = raw_population[ANCHOR_INCOME_YEAR]
for frame in parts:
    year = int(frame["source_year"].iloc[0])
    if args.weighting == "household":
        scale = target_population / len(ARCHIVES) / raw_population[year]
        frame["person_weight"] = frame["household_weight"] * scale
    else:
        frame["person_weight"] = frame["MARSUPWT"] / 100.0 / len(ARCHIVES)
    frame.drop(columns=["household_weight"], inplace=True)
person = pd.concat(parts, ignore_index=True)
person["person_id"] = np.arange(1, len(person) + 1, dtype=np.int64)
person["person_household_id"] = 0

started = time.monotonic()
error = None
try:
    output = run_source_stage(
        immigration.us_immigration_stage_spec(),
        tables={"person": person},
        operation_handlers={
            "derive_immigration_status": (
                immigration.derive_us_immigration_status_from_manifest
            )
        },
        config=SourceRuntimeConfig(seed=0, target_year=PERIOD),
    )
except Exception as exc:  # the receipt records a refusal instead of hiding it
    error = f"{type(exc).__name__}: {exc}"
stage_seconds = time.monotonic() - started

receipt = {
    "source_commit": source_commit,
    "source_commit_from": source_commit_from,
    "source_tree_clean": source_tree_clean,
    "immigration_module_sha256": hashlib.sha256(
        Path(immigration.__file__).read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "archives": archives,
    "seed": 0,
    "time_period": PERIOD,
    "weights": (
        "HSUP_WGT * 0.01 by PH_SEQ, each file rescaled to one third of income "
        "year 2024's weighted person population (asec_pool.pool_asec_sources, "
        "anchor_year=2024)"
        if args.weighting == "household"
        else "MARSUPWT / 100 / 3 per file"
    ),
    "draw_keys": "source_year:source_household_id(PH_SEQ):source_person_id(PERIDNUM)",
    "stage_seconds": round(stage_seconds, 1),
    "error": error,
}
if error is None:
    weights = output["person_weight"].to_numpy(dtype=np.float64)
    status = output["immigration_status_str"].astype(str).to_numpy()
    ssn = output["ssn_card_type"].astype(str).to_numpy()
    age = output["A_AGE"].to_numpy()
    code = output["PEINUSYR"].to_numpy()
    year = output["source_year"].to_numpy()
    composition = {}
    for value in sorted(set(status)):
        mask = status == value
        composition[value] = {
            "weighted": float(weights[mask].sum()),
            "rows": int(mask.sum()),
            "by_income_year": {
                str(y): float(weights[mask & (year == y)].sum())
                for y in (2023, 2024, 2025)
            },
            "code_29_weighted": float(weights[mask & (code == 29)].sum()),
            "under_19_weighted": float(weights[mask & (age < 19)].sum()),
        }
    receipt.update(
        {
            "rows": int(len(output)),
            "composition": composition,
            "none_ssn_iff_undocumented": bool(
                ((ssn == "NONE") == (status == "UNDOCUMENTED")).all()
            ),
        }
    )
    # This PR's controls, Pew universe and gate; main's tree has none of them.
    if hasattr(immigration, "us_immigration_controls"):
        labor_force = np.isin(output["A_LFSR"].to_numpy(), [1, 2, 3, 4]) & (age >= 16)
        controls = immigration.us_immigration_controls()
        targets: dict[str, float] = {}
        for draw in controls.humanitarian:
            targets[draw.status] = targets.get(draw.status, 0.0) + float(draw.target)
        pew = np.isin(status, immigration._PEW_UNAUTHORIZED_STATUS_VALUES) | (
            (status == "CUBAN_HAITIAN_ENTRANT") & (ssn == "NON_CITIZEN_VALID_EAD")
        )
        gate_person = (
            output.drop(columns=["person_weight"]).reset_index(drop=True).copy()
        )
        ids = np.arange(1, len(gate_person) + 1, dtype=np.int64)
        for entity in US_SCHEMA.group_entities:
            gate_person[US_SCHEMA.membership_column(entity)] = ids
        frame = Frame(
            {
                "person": gate_person,
                **{
                    entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
                    for entity in US_SCHEMA.group_entities
                },
            },
            US_SCHEMA,
            {"household": Weights(weights, WeightKind.DESIGN)},
        )
        gate = immigration.us_immigration_composition_gate(
            frame, time_period=PERIOD, controls=controls
        )
        receipt.update(
            {
                "humanitarian_targets_by_status": targets,
                "ratio_to_target": {
                    k: composition.get(k, {}).get("weighted", 0.0) / v if v else None
                    for k, v in targets.items()
                },
                "pew_universe": float(weights[pew].sum()),
                "pew_universe_anchor": float(controls.undocumented.population_anchor),
                "pew_labor_force": float(weights[pew & labor_force].sum()),
                "pew_labor_force_control": float(controls.undocumented.workers),
                "gate_passed": bool(gate.passed),
                "gate_failures": list(gate.failures),
                "humanitarian_draw_achieved": gate.details.get(
                    "humanitarian_draw_achieved"
                ),
                "humanitarian_target_relative_band": list(
                    immigration._HUMANITARIAN_TARGET_RELATIVE_BAND
                ),
            }
        )
OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
if error is None and args.labels_out is not None:
    pd.DataFrame(
        {
            "source_year": output["source_year"].to_numpy(),
            "source_person_id": output["source_person_id"].astype(str).to_numpy(),
            "person_weight": output["person_weight"].to_numpy(dtype=np.float64),
            "immigration_status_str": output["immigration_status_str"].astype(str),
            "ssn_card_type": output["ssn_card_type"].astype(str),
        }
    ).to_pickle(args.labels_out)

print("weighting:", args.weighting)
print("source commit:", source_commit, f"({source_commit_from})")
print("archives:", [(a["income_year"], a["rows"], a["peinusyr_max"]) for a in archives])
if error is not None:
    print("REFUSED:", error)
    raise SystemExit(0)
print(f"stage {stage_seconds:.0f}s on {len(output)} rows")
ratios = receipt.get("ratio_to_target", {})
for value, row in composition.items():
    ratio = ratios.get(value)
    ratio_text = f" ratio {ratio:.2f}" if ratio is not None else ""
    years = " ".join(f"{y}:{v / 1e6:.3f}" for y, v in row["by_income_year"].items())
    print(
        f"  {value:26s} {row['weighted'] / 1e6:8.3f}M{ratio_text}  [{years}]"
        f"  code29 {row['code_29_weighted'] / 1e6:.3f}M"
        f"  u19 {row['under_19_weighted'] / 1e6:.3f}M"
    )
print("NONE ssn iff UNDOCUMENTED:", receipt["none_ssn_iff_undocumented"])
if "gate_passed" in receipt:
    print(
        f"Pew universe {receipt['pew_universe'] / 1e6:.3f}M vs anchor "
        f"{receipt['pew_universe_anchor'] / 1e6:.1f}M; labor force "
        f"{receipt['pew_labor_force'] / 1e6:.3f}M vs control "
        f"{receipt['pew_labor_force_control'] / 1e6:.1f}M"
    )
    print("gate passed:", receipt["gate_passed"], "failures:", receipt["gate_failures"])
