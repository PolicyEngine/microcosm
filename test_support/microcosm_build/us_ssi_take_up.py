"""Reporter-anchored, Bernoulli-at-documented-prior SSI take-up tests."""

# ruff: noqa: F401

from __future__ import annotations

import copy
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.fiscal_targets import (
    SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE,
)
from microcosm.build.us_runtime.ssi_take_up import (
    SSI_TAKE_UP_ARCHIVED_DERIVATION_URL,
    SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL,
    SSI_TAKE_UP_ARCHIVED_TARGETS_URL,
    SSI_TAKE_UP_SSA_SOURCE_URL,
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_ANCHOR,
    US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE,
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    US_SSI_TAKE_UP_PRIOR_BASIS_CURRENT_FRAME,
    US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT,
    US_SSI_TAKE_UP_STAGE_NAME,
    SSITakeUpBandPriorBasis,
    SSITakeUpPriorBasis,
    _band_prior,
    _stable_source_draw,
    ssi_take_up_prior_basis_from_artifact,
    ssi_take_up_prior_basis_from_diagnostics,
    us_ssi_take_up_delivery_gate,
    us_ssi_take_up_diagnostics,
    us_ssi_take_up_gate,
    us_ssi_take_up_reporter_source_ids,
    us_ssi_take_up_stage_spec,
    with_us_ssi_take_up,
    write_us_ssi_take_up_diagnostics,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]
_TARGETS = {target.key: 50.0 for target in US_SSI_TAKE_UP_AGE_TARGETS}
_AGES = {"under_18": 12.0, "18_64": 40.0, "65_plus": 72.0}
# Fixture arithmetic per band: six dual-channel candidates weigh 20.0 each
# and the PUF-only candidate weighs 10.0 (capacity 130.0); the sole anchored
# candidate is source 0 (reporter floor 20.0).
_BAND_CAPACITY = 130.0
_REPORTER_FLOOR = 20.0
_ANCHORED_SOURCE_NUMBERS = {"0", "6"}


def _expected_prior(
    target: float,
    capacity: float = _BAND_CAPACITY,
    floor: float = _REPORTER_FLOOR,
) -> float:
    """The anchored-mass-corrected count-truthful threshold (#507/#508).

    Anchors are always selected, so expected delivered mass at non-anchor
    threshold p is floor + p*(capacity - floor); solving for the target
    gives (target - floor) / (capacity - floor). Saturated bands fall back
    to the reporter rate; an anchor floor at/above the target draws zero.
    """

    if capacity <= target:
        return min(floor / capacity, 1.0)
    if floor >= target:
        return 0.0
    return (target - floor) / (capacity - floor)


def _expected_bernoulli_flag(source_id: str, prior: float, *, seed: int = 17) -> bool:
    """The selection law: anchors unconditionally, else draw below prior."""

    if source_id.split(":")[1] in _ANCHORED_SOURCE_NUMBERS:
        return True
    return _stable_source_draw(source_id, seed=seed) < prior


def _frame(*, stale_output: bool = False) -> tuple[Frame, np.ndarray]:
    """Build three age bands with ASEC/PUF clones and PUF-only support."""

    rows: list[dict[str, object]] = []
    potential: list[float] = []
    person_id = 0
    for band, age in _AGES.items():
        for source_number in range(9):
            if source_number <= 5 or source_number == 8:
                channels = ("asec", "puf_tax_detail")
            elif source_number == 6:
                channels = ("asec",)
            else:
                channels = ("puf_tax_detail",)
            candidate = source_number <= 5 or source_number == 7
            reporter = source_number in {0, 6}
            for channel in channels:
                rows.append(
                    {
                        "person_id": person_id,
                        "person_household_id": person_id,
                        "person_tax_unit_id": person_id,
                        "person_spm_unit_id": person_id,
                        "person_family_id": person_id,
                        "person_marital_unit_id": person_id,
                        "age": age,
                        # PUF copies can carry SSI_VAL, but only direct ASEC
                        # rows are independent reporter anchors.
                        US_SSI_TAKE_UP_ANCHOR: 1_200.0 if reporter else 0.0,
                        "person_source_id": f"{band}:{source_number}",
                        "person_support_channel": channel,
                        _OUTPUT: bool((person_id % 2) == 0) if stale_output else False,
                    }
                )
                potential.append(100.0 if candidate else 0.0)
                person_id += 1

    person = pd.DataFrame(rows)
    ids = person["person_id"].to_numpy()
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(len(person), 10.0),
                kind=WeightKind.DESIGN,
            )
        },
    )
    return frame, np.asarray(potential, dtype=np.float64)


def _stacked_frame() -> tuple[Frame, np.ndarray]:
    """Convert the legacy operator fixture to physical ASEC/ACS channels."""

    frame, potential = _frame()
    person = frame.table("person").copy()
    source_number = person["person_source_id"].str.rsplit(":").str[-1].astype(int)
    legacy_roles = person["person_support_channel"].copy()
    person["person_spine_source_id"] = person["person_source_id"]
    person["person_support_clone_index"] = np.where(
        legacy_roles.eq("asec"),
        0,
        1,
    )
    person["person_support_channel"] = np.where(
        source_number.le(6),
        "asec",
        "acs",
    )
    acs_source = person["person_support_channel"].eq("acs")
    person.loc[acs_source, US_SSI_TAKE_UP_ANCHOR] = np.nan
    return _replace_person(frame, person), potential


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _assigned(
    *,
    seed: int = 17,
    targets: dict[str, float] | None = None,
    stale_output: bool = False,
) -> tuple[Frame, Frame, np.ndarray, dict[str, object]]:
    frame, potential = _frame(stale_output=stale_output)
    result, diagnostics = with_us_ssi_take_up(
        frame,
        uncapped_ssi=potential,
        seed=seed,
        targets=targets or _TARGETS,
    )
    return frame, result, potential, diagnostics


# --- Delivered-weight prior basis + hard delivery gate (microcosm#507/#508) ---

_BASIS_SHA = "ab" * 32


def _artifact_basis(
    *,
    capacities: dict[str, float],
    floors: dict[str, float] | None = None,
) -> SSITakeUpPriorBasis:
    resolved_floors = floors or {key: _REPORTER_FLOOR for key in capacities}
    return SSITakeUpPriorBasis(
        kind=US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT,
        bands=tuple(
            SSITakeUpBandPriorBasis(
                key=band.key,
                candidate_capacity=float(capacities[band.key]),
                reporter_candidate_floor=float(resolved_floors[band.key]),
            )
            for band in US_SSI_TAKE_UP_AGE_TARGETS
        ),
        source_sha256=_BASIS_SHA,
        source_schema_version=2,
    )


def _release_final_diagnostics() -> dict[str, object]:
    """Genuine final-measurement diagnostics — what us_ssi_take_up.json holds."""

    _, result, potential, stage_diagnostics = _assigned()
    stage_priors = {
        band["age_band"]: band["assignment_prior"]
        for band in stage_diagnostics["age_bands"]
    }
    return us_ssi_take_up_diagnostics(
        result,
        uncapped_ssi=potential,
        seed=17,
        targets=_TARGETS,
        assignment_priors=stage_priors,
        prior_basis=ssi_take_up_prior_basis_from_diagnostics(stage_diagnostics),
    )


def _delivered(diagnostics: dict[str, object], **selected: float) -> dict[str, object]:
    delivered = copy.deepcopy(diagnostics)
    # Delivery is judged on the final release-weight measurement only; the
    # tampered fixture emulates that payload.
    delivered["measurement_phase"] = "release_final"
    for band in delivered["age_bands"]:
        key = str(band["age_band"])
        if key in selected:
            band["selected_recipient_weight"] = float(selected[key])
    return delivered


__all__ = [name for name in globals() if not name.startswith("__")]
