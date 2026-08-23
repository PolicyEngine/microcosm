"""Orchestrate optional ACS augmentation of an ASEC-by-PUF base pool.

This is the narrow integration seam between the independently testable ACS
stages.  It deliberately does not own acquisition, native mappings, transfer
models, or pool-mass policy: it invokes those stages in order and gathers
their provenance for a build manifest.

The disabled path is an identity contract.  When ``source`` is ``None`` no
option is validated and no ACS stage is called; the result carries the exact
same base :class:`~microcosm.frame.Frame` object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.gates import FitWeightRecord
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_pums import (
    DEFAULT_CHUNKSIZE,
    AcsPumsSource,
    build_acs_pums_unit_frame,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    ASEC_PUF_DONOR_SPINE,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    TargetFamilies,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.base_pool import (
    preflight_pooled_ladder_geography,
    with_optional_acs_spine,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from microcosm.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    us_puma_ladder_assignment_summary,
)
from microcosm.frame import Frame

__all__ = ["AcsMultispineResult", "build_optional_acs_multispine"]


@dataclass(frozen=True)
class AcsMultispineResult:
    """A base-pool frame plus ACS build and fit provenance.

    ``fit_records`` remains typed so the caller can pass it directly to the
    build-level weights audit. ``provenance`` contains the same fit evidence,
    along with loader, native-input, and imputed-input records, converted to
    values accepted by strict ``json.dumps(..., allow_nan=False)``.
    """

    frame: Frame
    fit_records: tuple[FitWeightRecord, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


def build_optional_acs_multispine(
    base: Frame,
    source: AcsPumsSource | None = None,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    acs_share: float = 0.5,
    target_families: TargetFamilies | None = None,
    donor_spine: str = ASEC_PUF_DONOR_SPINE,
    donor_channel: str | None = ACS_DONOR_CHANNEL_AUTO,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    puma_ladder: UsPumaLadder | None = None,
    geography_seed: int = 0,
    expected_congressional_district_vintage: str | None = (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ),
) -> AcsMultispineResult:
    """Optionally build, transfer, and append the ACS 2024 1-year spine.

    The already-built ``base`` is both the ASEC-by-PUF transfer donor and the
    pool being augmented.  Calibration is intentionally outside this seam.

    Large intermediate frames are released as soon as the next stage has
    materialized its own frame.  This cannot make the final dense pool small,
    but it avoids retaining the raw, mapped, transferred, and pooled versions
    together for the rest of the build.
    """

    if source is None:
        return AcsMultispineResult(
            frame=base,
            provenance={"enabled": False},
        )

    raw_acs, loader_metadata = build_acs_pums_unit_frame(
        source,
        chunksize=chunksize,
    )
    loader_provenance = _json_ready_mapping(loader_metadata)

    mapped = map_acs_native_inputs(raw_acs)
    del raw_acs
    native_provenance = _json_ready_mapping(mapped.native_inputs)
    mapped_frame = mapped.frame
    del mapped

    # Geography preflight BEFORE the expensive transfer: an unmapped donor
    # tract, incoherent preserved geography, or an unknown ACS PUMA must
    # fail here, not after the QRF fits have run.
    donor_geography: str | None = None
    if puma_ladder is not None:
        donor_geography = preflight_pooled_ladder_geography(
            base.table("household"),
            mapped_frame.table("household"),
            puma_ladder,
        )

    transferred = transfer_acs_inputs(
        mapped_frame,
        base,
        target_families=target_families,
        donor_spine=donor_spine,
        donor_channel=donor_channel,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
    )
    del mapped_frame
    adult_care_gate = _require_recipient_adult_care_structure(transferred.frame)
    fit_records = tuple(transferred.fit_records)
    imputed_provenance = _json_ready_sequence(transferred.imputed_inputs)
    deferred_inputs = tuple(transferred.deferred_inputs)
    if puma_ladder is not None:
        deferred_inputs = tuple(
            column
            for column in deferred_inputs
            if column not in {"congressional_district_geoid", "county_fips"}
        )
    deferred_provenance = _json_ready_sequence(deferred_inputs)
    resolved_donor_channel = transferred.resolved_donor_channel
    fit_provenance = _json_ready_sequence(fit_records)
    transferred_frame = transferred.frame
    del transferred

    pool_options: dict[str, Any] = {"acs_share": acs_share}
    if puma_ladder is not None:
        pool_options.update(
            {
                "puma_ladder": puma_ladder,
                "geography_seed": geography_seed,
                "expected_congressional_district_vintage": (
                    expected_congressional_district_vintage
                ),
            }
        )
    pooled = with_optional_acs_spine(base, transferred_frame, **pool_options)
    del transferred_frame

    provenance = {
        "enabled": True,
        "acs_share": _json_ready(acs_share),
        "loader": loader_provenance,
        "native_inputs": native_provenance,
        "imputed_inputs": imputed_provenance,
        "deferred_inputs": deferred_provenance,
        "adult_care_recipient_gate": _json_ready(adult_care_gate)
        if adult_care_gate is not None
        else None,
        "fit_records": fit_provenance,
        "fit_configuration": {
            "donor_spine": donor_spine,
            "requested_donor_channel": donor_channel,
            "resolved_donor_channel": resolved_donor_channel,
            "seed": seed,
            "n_estimators": n_estimators,
            "max_targets_per_fit": max_targets_per_fit,
        },
    }
    if puma_ladder is not None:
        geography = us_puma_ladder_assignment_summary(
            pooled.table("household"),
            puma_ladder,
            weight_values=pooled.weights_for("household").values,
        )
        geography.update(
            {
                "seed": geography_seed,
                "expected_congressional_district_vintage": (
                    expected_congressional_district_vintage
                ),
                # How ASEC-by-PUF rows got their geography: a base-O-line
                # donor keeps its certified block-ladder assignment
                # ("preserved_assigned"); a buildl-line donor draws within
                # state ("ladder_drawn"). ACS rows always anchor on their
                # observed PUMA.
                "donor_geography": donor_geography,
                "resolved_model_inputs": [
                    "congressional_district_geoid",
                    "county_fips",
                ],
                # Sub-PUMA geography stays unresolved for the ACS spine;
                # a preserved donor carries its own block/tract columns.
                "unresolved_sub_puma_inputs": [
                    "block_geoid",
                    "tract_geoid",
                ],
            }
        )
        provenance["geography_ladder"] = _json_ready_mapping(geography)
    return AcsMultispineResult(
        frame=pooled,
        fit_records=fit_records,
        provenance=provenance,
    )


def _require_recipient_adult_care_structure(
    frame: Frame,
) -> dict[str, Any] | None:
    """Hard-gate the transferred ACS adult-care surface when it exists.

    Runs on the recipient (ACS-only) frame, where the raw ASEC source column
    is absent by construction so the gate's measured-identity check scopes
    itself out; the statute-structure certificate and plausibility bands
    bind. A transfer that never produced the adult-care columns (custom test
    plans) is not gated.
    """

    from microcosm.build.us_runtime.adult_care import (
        US_ADULT_CARE_OUTPUT_COLUMNS,
        us_adult_care_signal_gate,
    )

    person = frame.table("person")
    if any(column not in person.columns for column in US_ADULT_CARE_OUTPUT_COLUMNS):
        return None
    gate = us_adult_care_signal_gate(frame)
    if not gate.passed:
        raise ValueError(
            "Transferred ACS adult-care surface failed the statute-structure "
            "gate:\n  " + "\n  ".join(gate.failures)
        )
    return {"passed": True, "details": dict(gate.details)}


def _json_ready_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    converted = _json_ready(value)
    if not isinstance(converted, dict):  # pragma: no cover - type contract
        raise TypeError("Expected JSON-ready mapping conversion to return a dict.")
    return converted


def _json_ready_sequence(value: Sequence[Any]) -> list[Any]:
    converted = _json_ready(value)
    if not isinstance(converted, list):  # pragma: no cover - type contract
        raise TypeError("Expected JSON-ready sequence conversion to return a list.")
    return converted


def _json_ready(value: Any) -> Any:
    """Recursively detach and normalize one manifest value."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_ready(getattr(value, item.name))
            for item in fields(value)
            if item.name != "target_regimes" or getattr(value, item.name)
        }
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "ACS multispine provenance cannot contain NaN or infinity."
            )
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "ACS multispine provenance mapping keys must be strings; "
                    f"got {type(key).__name__}."
                )
            if key == "target_regimes" and not item:
                # The opt-in stacked audit field did not exist on legacy ACS
                # multispine pattern provenance. Keep default callers' JSON
                # schema byte-compatible when the audit selection is empty.
                continue
            converted[key] = _json_ready(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    raise TypeError(
        "ACS multispine provenance contains a non-JSON value of type "
        f"{type(value).__name__}."
    )
