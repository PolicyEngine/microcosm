"""Verbatim copies of the US local hold-out code before microcosm#1728 moved it.

``tools/us_acs_local_cd_surface.py``, ``tools/us_acs_local_county_surface.py``
and ``tools/build_us_acs_local_release.py`` held this code at ``f4dee4c70``. It
now lives in the shared ``microcosm.build.target_holdouts``, with the US
choices in ``us/local_target_construction.json``. The equivalence tests hold
the shared code to these copies. Only names changed: the district and county
versions of ``assign_target_roles`` and ``pro_rata_baseline`` carry a ``cd_``
or ``county_`` prefix, and ``cd_surface.`` is dropped from the county code.
"""

# ruff: noqa

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

CD_HOLDOUT_SALT = "microcosm.us_acs_local.cd_holdout.v1"

CD_HOLDOUT_UNIT = "state_x_soi_concept_family"

DEFAULT_STATE_CD_HOLDOUT_FRACTION = 0.1

MAX_CD_HOLDOUT_FRACTION = 0.5

ROLE_TRAIN = "train"

ROLE_HOLDOUT = "holdout"

GEOGRAPHY_CD = "congressional_district"

def soi_concept_family(source_measure_id: str) -> str:
    """The holdout family of an SOI measure: its concept without count/amount.

    Every EITC measure is one family, because the per-child-count rows add up
    to the EITC total and holding one out would leave it pinned by the rest.
    """

    measure = str(source_measure_id)
    if measure.startswith("eitc"):
        return "eitc"
    for suffix in ("_amount", "_returns", "_claims", "_count"):
        if measure.endswith(suffix):
            return measure[: -len(suffix)]
    return measure

def cd_holdout_unit_key(record: Mapping) -> str | None:
    """The holdout unit of a district SOI target, or None if not eligible."""

    if record.get("family") != "irs_soi":
        return None
    if record.get("geography_level") != GEOGRAPHY_CD:
        return None
    if not record.get("state_cd_parent_target_name"):
        return None
    return (
        f"{record['state_fips']}|"
        f"{soi_concept_family(str(record.get('source_measure_id')))}"
    )

def cd_assign_target_roles(
    records: Sequence[dict],
    *,
    fraction: float,
    salt: str = CD_HOLDOUT_SALT,
) -> dict:
    """Set ``role`` and ``holdout_unit`` on every record; return the receipt.

    Only district SOI targets with a bound state parent are eligible; every
    other target trains. A unit is held out by
    :func:`microcosm.build.holdout.hash_holdout_unit` on its key.
    """

    from microcosm.build.holdout import hash_holdout_unit

    if not 0.0 <= float(fraction) <= MAX_CD_HOLDOUT_FRACTION:
        raise ValueError(
            f"CD holdout fraction must be in [0, {MAX_CD_HOLDOUT_FRACTION}], "
            f"got {fraction!r}."
        )
    units: dict[str, bool] = {}
    for record in records:
        unit = cd_holdout_unit_key(record)
        record["holdout_unit"] = unit
        if unit is None:
            record["role"] = ROLE_TRAIN
            continue
        if unit not in units:
            units[unit] = hash_holdout_unit(unit, fraction=fraction, salt=salt)
        record["role"] = ROLE_HOLDOUT if units[unit] else ROLE_TRAIN
    held_units = sorted(unit for unit, held in units.items() if held)
    return {
        "unit": CD_HOLDOUT_UNIT,
        "salt": salt,
        "fraction": float(fraction),
        "eligible_units": len(units),
        "held_units": len(held_units),
        "held_unit_keys": held_units,
        "eligible_targets": sum(
            1 for record in records if record["holdout_unit"] is not None
        ),
        "held_targets": sum(1 for record in records if record["role"] == ROLE_HOLDOUT),
        "hash": "sha256(salt + 0x1f + unit)[:8] / 2**64 < fraction",
    }

def train_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_TRAIN
    ]

def holdout_rows(records: Sequence[Mapping]) -> list[int]:
    return [
        index for index, record in enumerate(records) if record["role"] == ROLE_HOLDOUT
    ]

def csr_row_measure(matrix: sparse.csr_array, row: int, n: int):
    """A callable measure returning one CSR row as a dense float64 vector.

    The calibrate kernel compiles targets through ``build_constraint_matrix``,
    which reads each target's row and keeps its nonzeros; a callable measure
    lets it read the row straight from the checkpoint CSR (the UK rowwise
    precedent, ``uk_runtime.local_rowwise._rowwise_target_set``). Only one
    dense row exists at a time. float32 values widen exactly to float64, as
    the dense checkpoint's float32 columns did.
    """

    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    data = matrix.data[start:stop]

    def measure(frame) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        values[indices] = data
        return values

    # Calibration diagnostics name a callable measure by its qualified name;
    # name the matrix row rather than this closure.
    measure.__qualname__ = f"target_matrix_row[{row}]"
    return measure

def calibration_target_set(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    n_households: int,
    *,
    specs: Sequence | None = None,
):
    """The TargetSet the calibrator sees: training targets only.

    Held-out targets are never constructed, so they cannot reach the solve.
    With ``specs`` (the checkpoint's registry specs, row-aligned with
    ``records``) each target keeps its spec's value, period, source,
    metadata and calibration hierarchy, as ``TargetSpec.to_target`` would,
    and only its measure becomes the CSR row.
    """

    from microcosm.calibrate.target import Target, TargetSet

    if matrix.shape != (len(records), n_households):
        raise ValueError(
            f"target matrix shape {matrix.shape} does not match "
            f"{len(records)} targets x {n_households} households."
        )
    if specs is not None and [spec.name for spec in specs] != [
        record["name"] for record in records
    ]:
        raise ValueError("target specs are not row-aligned with the roles.")
    targets = []
    for index in train_rows(records):
        record = records[index]
        if record["role"] != ROLE_TRAIN:  # pragma: no cover - train_rows filters
            raise AssertionError("a held-out target reached the calibration set")
        measure = csr_row_measure(matrix, index, n_households)
        if specs is None:
            targets.append(
                Target(
                    name=record["name"],
                    entity=record.get("entity", "household"),
                    measure=measure,
                    value=float(record["value"]),
                    period=record["period"],
                    source=record["source"],
                )
            )
            continue
        spec = specs[index]
        if spec.filter is not None:
            raise ValueError(
                f"{spec.name}: a filtered spec cannot take a matrix-row measure; "
                "the row already carries the filter."
            )
        targets.append(
            Target(
                name=spec.name,
                entity=spec.entity,
                measure=measure,
                value=spec.value,
                period=spec.period,
                tolerance=spec.tolerance,
                source=spec.source,
                metadata=spec.metadata,
                hierarchy=spec.hierarchy,
            )
        )
    held = {records[index]["name"] for index in holdout_rows(records)}
    leaked = held & {target.name for target in targets}
    if leaked:
        raise AssertionError(f"held-out targets reached the calibrator: {leaked}")
    return TargetSet(targets)

def _error_summary(estimates, targets, *, cap: float) -> dict:
    from microcosm.calibrate import relative_error_loss

    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    scale = np.maximum(np.abs(targets), 1.0)
    errors = np.abs(estimates - targets) / scale
    return {
        "mean_abs_rel_error": float(errors.mean()),
        "median_abs_rel_error": float(np.median(errors)),
        "p90_abs_rel_error": float(np.quantile(errors, 0.9)),
        "max_abs_rel_error": float(errors.max()),
        "fraction_within_10pct": float((errors <= 0.10).mean()),
        "capped_loss": float(
            relative_error_loss(estimates, targets, target_loss_cap=cap)
        ),
    }

def cd_pro_rata_baseline(records: Sequence[Mapping], rows: Sequence[int]) -> np.ndarray:
    """State parent value x district population share, for each row."""

    by_name = {record["name"]: record for record in records}
    values = []
    for index in rows:
        record = records[index]
        parent = by_name.get(record.get("state_cd_parent_target_name"))
        if parent is None:
            raise ValueError(
                f"{record['name']} has no bound state parent for the baseline."
            )
        cd_population = record.get("cd_population")
        state_population = record.get("state_population")
        if not cd_population or not state_population:
            raise ValueError(
                f"{record['name']} carries no district/state population for "
                "the pro-rata baseline."
            )
        values.append(
            float(parent["value"]) * float(cd_population) / float(state_population)
        )
    return np.asarray(values, dtype=np.float64)

def score_cd_holdout(
    records: Sequence[Mapping],
    matrix: sparse.csr_array,
    *,
    design_weights: np.ndarray,
    final_weights: np.ndarray,
    cap: float,
    receipt: Mapping | None = None,
) -> dict:
    """Held-out district targets: calibrated vs design vs pro-rata baseline."""

    rows = [
        row for row in holdout_rows(records)
        if records[row].get("geography_level") == GEOGRAPHY_CD
    ]
    base = {"report_only": True, **dict(receipt or {})}
    if not rows:
        return {**base, "n_targets": 0}
    held = sparse.csr_array(matrix[rows]).astype(np.float64)
    targets = np.asarray([records[i]["value"] for i in rows], dtype=np.float64)
    final = held @ np.asarray(final_weights, dtype=np.float64)
    design = held @ np.asarray(design_weights, dtype=np.float64)
    baseline = cd_pro_rata_baseline(records, rows)
    scale = np.maximum(np.abs(targets), 1.0)
    final_error = np.abs(final - targets) / scale
    baseline_error = np.abs(baseline - targets) / scale
    families: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(rows):
        families[str(records[index]["holdout_unit"]).split("|", 1)[1]].append(position)
    by_family = {}
    for family, positions in sorted(families.items()):
        idx = np.asarray(positions)
        by_family[family] = {
            "n_targets": len(positions),
            "calibrated_mean_abs_rel_error": float(final_error[idx].mean()),
            "pro_rata_mean_abs_rel_error": float(baseline_error[idx].mean()),
            "calibrated_beats_pro_rata": float(
                (final_error[idx] < baseline_error[idx]).mean()
            ),
        }
    return {
        **base,
        "n_targets": len(rows),
        "calibrated": _error_summary(final, targets, cap=cap),
        "design_weights": _error_summary(design, targets, cap=cap),
        "pro_rata_baseline": _error_summary(baseline, targets, cap=cap),
        "calibrated_beats_pro_rata_share": float((final_error < baseline_error).mean()),
        "baseline": (
            "state parent target x (district population / state population), "
            "populations from the PUMA ladder's 119th-plan district overlap"
        ),
        "by_family": by_family,
        "targets": [
            {
                "name": records[index]["name"],
                "unit": records[index]["holdout_unit"],
                "target": float(targets[position]),
                "calibrated": float(final[position]),
                "design": float(design[position]),
                "pro_rata": float(baseline[position]),
            }
            for position, index in enumerate(rows)
        ],
    }

COUNTY_HOLDOUT_SALT = "acs-local-county-v1"

DEFAULT_COUNTY_HOLDOUT_FRACTION = 0.1

def county_assign_target_roles(
    records, *, fraction=DEFAULT_COUNTY_HOLDOUT_FRACTION, salt=COUNTY_HOLDOUT_SALT
):
    """Hold out whole state/concept-family blocks to prevent sibling leakage."""
    from microcosm.build.holdout import hash_holdout_unit

    if not isinstance(salt, str) or not salt:
        raise ValueError("County holdout salt must be a non-empty string.")
    if not 0 <= fraction <= MAX_CD_HOLDOUT_FRACTION:
        raise ValueError(
            f"County holdout fraction must be in [0, {MAX_CD_HOLDOUT_FRACTION}]."
        )
    units = {}
    eligible = []
    for record in records:
        if record.get("geography_level") != "county":
            continue
        unit = None
        if record.get("state_county_parent_target_name"):
            family = (
                soi_concept_family(str(record.get("source_measure_id")))
                if record.get("family") == "irs_soi"
                else str(record.get("family"))
            )
            unit = f"{record['state_fips']}|{family}"
        record["holdout_unit"] = unit
        record["role"] = ROLE_TRAIN
        if unit is not None:
            eligible.append(record)
            units.setdefault(
                unit, hash_holdout_unit(unit, fraction=fraction, salt=salt)
            )
            if units[unit]:
                record["role"] = ROLE_HOLDOUT
    return {
        "unit": "state x county concept family",
        "salt": salt,
        "fraction": float(fraction),
        "eligible_units": len(units),
        "eligible_targets": len(eligible),
        "held_units": sum(units.values()),
        "held_unit_keys": sorted(key for key, value in units.items() if value),
        "held_targets": sum(
            record["role"] == ROLE_HOLDOUT for record in eligible
        ),
    }

def county_pro_rata_baseline(records, rows):
    by_name = {record["name"]: record for record in records}
    values = []
    for index in rows:
        record = records[index]
        parent = by_name.get(record.get("state_county_parent_target_name"))
        county, state = (
            float(record.get("county_population", 0)),
            float(record.get("state_population", 0)),
        )
        if parent is None or not (
            math.isfinite(county) and math.isfinite(state) and county > 0 and state > 0
        ):
            raise ValueError(
                f"{record['name']} has no state parent or positive county/state population for pro-rata baseline."
            )
        values.append(float(parent["value"]) * county / state)
    return np.asarray(values, dtype=float)

def score_county_holdout(
    records, matrix, *, design_weights, final_weights, cap, receipt=None
):
    rows = [
        index
        for index, record in enumerate(records)
        if record.get("geography_level") == "county"
        and record.get("role") == ROLE_HOLDOUT
    ]
    base = {"report_only": True, **dict(receipt or {}), "n_targets": len(rows)}
    if not rows:
        return base
    held = sparse.csr_array(matrix[rows]).astype(float)
    targets = np.asarray([records[index]["value"] for index in rows], float)
    final, design = held @ np.asarray(final_weights), held @ np.asarray(design_weights)
    baseline = county_pro_rata_baseline(records, rows)
    scale = np.maximum(np.abs(targets), 1.0)
    return {
        **base,
        "calibrated": _error_summary(final, targets, cap=cap),
        "design_weights": _error_summary(design, targets, cap=cap),
        "pro_rata_baseline": _error_summary(baseline, targets, cap=cap),
        "calibrated_beats_pro_rata_share": float(
            (
                np.abs(final - targets) / scale < np.abs(baseline - targets) / scale
            ).mean()
        ),
        "baseline": "state parent target x county population / state population",
        "targets": [
            {
                "name": records[index]["name"],
                "county_fips": records[index]["county_fips"],
                "target": float(targets[position]),
                "calibrated": float(final[position]),
                "design": float(design[position]),
                "pro_rata": float(baseline[position]),
            }
            for position, index in enumerate(rows)
        ],
    }

def _attach_pro_rata_populations(targets: list[dict], cd_populations: dict) -> None:
    """Give each district SOI target its district and state populations.

    The pro-rata baseline allocates the state parent by these; both come
    from the PUMA ladder's 119th-plan district overlap populations, so a
    state's district shares sum to one.
    """

    state_population: dict[int, float] = {}
    for district, population in cd_populations.items():
        state = int(district) // 100
        state_population[state] = state_population.get(state, 0.0) + float(population)
    for target in targets:
        geoid = target.get("congressional_district_geoid")
        if target.get("family") != "irs_soi" or not geoid:
            continue
        district = int(geoid)
        target["cd_population"] = float(cd_populations.get(district, 0.0))
        target["state_population"] = float(state_population.get(district // 100, 0.0))

def _require_pro_rata_populations(targets: list[dict]) -> None:
    """Refuse, before any solve, a district row the baseline cannot score."""

    unpopulated = [
        target["name"]
        for target in targets
        if target.get("family") == "irs_soi"
        and target.get("congressional_district_geoid")
        and not (
            target.get("cd_population", 0) > 0 and target.get("state_population", 0) > 0
        )
    ]
    if unpopulated:
        raise SystemExit(
            f"{len(unpopulated)} district SOI target(s) have no positive "
            "ladder district or state population for the pro-rata "
            f"baseline (e.g. {unpopulated[:3]}); refusing before the solve."
        )
