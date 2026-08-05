"""Build the SLD per-district layer for a US local-area artifact (populace#625).

Chain: membership -> targets -> doctrine solve -> validated sidecar. The
layer is strictly downstream of the artifact build: it changes no artifact
bytes and no national calibration anywhere — it reads the packaged
artifact's household/person tables and calibrated weights, derives
2024-vintage district membership, solves every district under the reviewed
doctrine (:mod:`populace.build.us_runtime.sld_local_doctrine`), and writes
the sidecar bundle (long weights, diagnostics, achieved-vs-target,
honest-boundaries statement) with per-file sha256s.

Inputs:
- ``--artifact-h5``: the local-area artifact (buildl-shape or the #512
  rebuild — the layer binds to columns, not to a build id);
- ``--sld-facts``: a ledger facts/consumer-facts JSONL carrying the ACS
  5-year SLD facts (populace#625 ledger lane);
- ``--ladder``: the SLD membership-ladder NPZ
  (``tools/build_us_sld_membership_ladder_artifact.py``).

The run scopes itself to the states the facts cover (the pilot is one
state), and the summary JSON records input shas, gate details, and the
doctrine so the run is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.us_runtime.sld_local_doctrine import (
    US_SLD_LOCAL_SOLVE_DOCTRINE,
    solve_us_sld_chamber_under_doctrine,
)
from populace.build.us_runtime.sld_local_report import write_sld_sidecar
from populace.build.us_runtime.sld_local_targets import (
    build_sld_district_problems,
    load_sld_target_facts,
)
from populace.build.us_runtime.sld_membership import (
    assign_us_sld_membership,
    load_us_sld_membership_ladder,
    us_sld_membership_gate,
)

HOUSEHOLD_COLUMNS = (
    "household_id",
    "state_fips",
    "puma",
    "congressional_district_geoid",
    "county_fips",
    "tract_geoid",
    "household_weight",
)
PERSON_ID_COLUMNS = ("person_household_id", "age")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_column(h5file, name: str) -> np.ndarray | None:
    import h5py

    if name not in h5file:
        return None
    node = h5file[name]
    if isinstance(node, h5py.Dataset):
        return node[...]
    keys = list(node.keys())
    if len(keys) != 1:
        raise ValueError(
            f"H5 variable {name} has {len(keys)} year datasets {keys}; "
            "expected exactly one."
        )
    return node[keys[0]][...]


def _is_entity_table_layout(artifact_h5: Path) -> bool:
    """Whether the H5 is a pandas-HDF store with per-entity frames.

    Covers both pandas formats: ``table`` (buildl-era) and ``fixed`` (the
    buildo local release). The alternative is the packaged
    variable-per-year layout.
    """
    try:
        with pd.HDFStore(artifact_h5, "r") as store:
            keys = set(store.keys())
    except Exception:
        return False
    return "/household" in keys and "/person" in keys


def _select_entity_columns(store, key: str, wanted: tuple[str, ...]) -> pd.DataFrame:
    available = list(store.select(key, start=0, stop=1).columns)
    columns = [column for column in wanted if column in available]
    try:
        frame = store.select(key, columns=columns)
    except TypeError:
        # Fixed-format stores must be read in their entirety.
        frame = store.select(key)[columns]
    return frame.reset_index(drop=True)


def load_layer_frames(
    artifact_h5: Path,
    *,
    extra_person_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the household/person tables the layer needs from the H5.

    Two supported layouts: the packaged single-year layout (variable ->
    year dataset) and the pandas-HDF entity-table layout the buildl-era
    local artifact ships (one ``table`` node per entity). String columns
    decode to ``str``; missing optional geography columns (``tract_geoid``
    on ACS-spine rows) load as absent.
    """
    import h5py

    person_wanted = PERSON_ID_COLUMNS + tuple(extra_person_columns)
    if _is_entity_table_layout(artifact_h5):
        with pd.HDFStore(artifact_h5, "r") as store:
            households_frame = _select_entity_columns(
                store,
                "household",
                HOUSEHOLD_COLUMNS,
            )
            person_frame = _select_entity_columns(
                store,
                "person",
                person_wanted,
            )
    else:
        with h5py.File(artifact_h5, "r") as h5file:
            households: dict[str, np.ndarray] = {}
            for name in HOUSEHOLD_COLUMNS:
                column = _h5_column(h5file, name)
                if column is not None:
                    households[name] = column
            persons: dict[str, np.ndarray] = {}
            for name in person_wanted:
                column = _h5_column(h5file, name)
                if column is not None:
                    persons[name] = column
        households_frame = pd.DataFrame(
            {
                name: (
                    np.char.decode(values.astype("S"), "utf-8")
                    if values.dtype.kind in ("S", "O")
                    else values
                )
                for name, values in households.items()
            }
        )
        person_frame = pd.DataFrame(persons)
    for required in ("household_id", "state_fips", "household_weight"):
        if required not in households_frame.columns:
            raise ValueError(f"artifact is missing household column {required}.")
    for required in PERSON_ID_COLUMNS:
        if required not in person_frame.columns:
            raise ValueError(f"artifact is missing person column {required}.")
    return households_frame, person_frame


def normalize_puma(households: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``puma`` to the national convention (state*100000+puma5).

    ACS PUMS publishes the 5-digit within-state code; the ladders key on the
    national 7-digit integer. Values already above 100000 pass through.
    """
    if "puma" not in households.columns:
        return households
    result = households.copy()
    puma = pd.to_numeric(result["puma"], errors="coerce")
    state = pd.to_numeric(result["state_fips"], errors="coerce")
    national = np.where(
        puma.notna() & (puma > 0) & (puma < 100_000),
        state * 100_000 + puma,
        puma,
    )
    result["puma"] = national
    return result


def run_us_sld_local_layer(
    households: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    facts_path: Path,
    ladder_path: Path,
    out_dir: Path,
    epochs: int = 512,
    learning_rate: float = 0.15,
    seed: int = 0,
    input_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The pure chain over already-loaded frames; returns the summary."""
    facts = load_sld_target_facts(facts_path)
    if facts.calibration.empty:
        raise SystemExit(f"No SLD calibration facts in {facts_path}.")
    ladder = load_us_sld_membership_ladder(ladder_path)

    fact_states = sorted(set(facts.calibration["state_fips"]))
    _log(f"Facts cover state(s) {fact_states}; scoping the layer to them")
    households = normalize_puma(households)
    state_str = households["state_fips"].map(lambda value: f"{int(value):02d}")
    scope = state_str.isin(fact_states).to_numpy()
    if not scope.any():
        raise SystemExit(f"Artifact has no households in fact state(s) {fact_states}.")
    scoped_households = households.loc[scope].reset_index(drop=True)
    scoped_ids = set(scoped_households["household_id"].tolist())
    person_mask = persons["person_household_id"].isin(scoped_ids).to_numpy()
    scoped_persons = persons.loc[person_mask].reset_index(drop=True)
    _log(
        f"Scoped frame: {len(scoped_households):,} households, "
        f"{len(scoped_persons):,} persons"
    )

    assigned = assign_us_sld_membership(scoped_households, ladder, seed=seed)
    gate = us_sld_membership_gate(assigned, ladder)
    if not gate.passed:
        raise SystemExit("SLD membership gate failed: " + "; ".join(gate.failures))
    _log(f"Membership gate passed: {gate.details['method_counts']}")

    weights = assigned["household_weight"].to_numpy(dtype=np.float64)
    chambers = []
    zero_support: dict[str, tuple[str, ...]] = {}
    recipe_resolution = None
    money_income = None
    for area_type in ("sldu", "sldl"):
        chamber_facts = facts.calibration[facts.calibration["area_type"] == area_type]
        if chamber_facts.empty:
            _log(f"No {area_type} facts; skipping the chamber")
            continue
        build = build_sld_district_problems(
            assigned,
            scoped_persons,
            base_weights=weights,
            facts=facts,
            area_type=area_type,
        )
        zero_support[area_type] = build.zero_support_districts
        if recipe_resolution is None:
            recipe_resolution = build.recipe_resolution
            money_income = build.money_income
        if not build.problems:
            _log(f"No solvable {area_type} districts; skipping the chamber")
            continue
        _log(f"Solving {len(build.problems)} {area_type} district(s) under doctrine")
        chambers.append(
            solve_us_sld_chamber_under_doctrine(
                list(build.problems),
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
            )
        )
    if not chambers:
        raise SystemExit("No chamber had solvable districts.")
    assert recipe_resolution is not None and money_income is not None

    money_income_by_household_id = dict(
        zip(
            assigned["household_id"].tolist(),
            [float(value) for value in money_income],
            strict=True,
        )
    )
    shas = write_sld_sidecar(
        out_dir,
        chambers=chambers,
        facts=facts,
        recipe_resolution=recipe_resolution,
        membership_gate_details=gate.details,
        doctrine_record=US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        zero_support_districts=zero_support,
        money_income_by_household_id=money_income_by_household_id,
    )
    summary = {
        "schema_version": 1,
        "inputs": {
            "sld_facts": {
                "path": str(facts_path),
                "sha256": facts.source_sha256,
            },
            "membership_ladder": {
                "path": str(ladder_path),
                "sha256": _sha256(Path(ladder_path)),
            },
            **(dict(input_identity) if input_identity else {}),
        },
        "scope_states": fact_states,
        "membership_gate": gate.details,
        "doctrine": US_SLD_LOCAL_SOLVE_DOCTRINE.as_record(),
        "chambers": [
            {
                "area_type": chamber.area_type,
                "n_districts": int(chamber.census_rollup["n_districts"]),
                "pushed_out": int(chamber.census_rollup["pushed_out"]),
                "past_at_final": int(chamber.census_rollup["past_at_final"]),
                "median_fraction_within_10pct": float(
                    chamber.district_summary["fraction_within_10pct"].median()
                ),
                "max_realized_weight_ratio": float(
                    chamber.district_summary["realized_max_weight_ratio"].max()
                ),
            }
            for chamber in chambers
        ],
        "zero_support_districts": {
            area_type: list(codes) for area_type, codes in zero_support.items()
        },
        "sidecar_sha256": shas,
        "solver": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }
    summary_path = Path(out_dir) / "sld_layer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _log(f"Wrote sidecar + summary to {out_dir}")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SLD per-district sidecar for a local artifact."
    )
    parser.add_argument("--artifact-h5", required=True, type=Path)
    parser.add_argument("--sld-facts", required=True, type=Path)
    parser.add_argument("--ladder", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from populace.build.us_runtime.sld_local_targets import (
        SLD_ACS_MONEY_INCOME_RECIPE,
    )

    recipe_columns = tuple(
        column
        for component in SLD_ACS_MONEY_INCOME_RECIPE
        for column in component.columns
    )
    _log(f"Loading artifact {args.artifact_h5}")
    households, persons = load_layer_frames(
        args.artifact_h5,
        extra_person_columns=recipe_columns,
    )
    run_us_sld_local_layer(
        households,
        persons,
        facts_path=args.sld_facts,
        ladder_path=args.ladder,
        out_dir=args.out_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        input_identity={
            "artifact_h5": {
                "path": str(args.artifact_h5),
                "sha256": _sha256(args.artifact_h5),
            }
        },
    )


if __name__ == "__main__":
    main()
