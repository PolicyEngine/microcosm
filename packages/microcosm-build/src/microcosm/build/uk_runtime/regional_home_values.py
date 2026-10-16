"""Home values from the region's Land Registry distribution, on the spine
(microcosm#1112, microcosm#1095).

The dense line reads each England-and-Wales owner clone's home value off its
assigned constituency's uprated latest-sale distribution once it has an area
(``uk.full.property.value``, microcosm#1240). The national line solves the
bound spine, which has no area, so its ``main_residence_value`` stayed the
WAS-imputed value after the regional uprating, not a Land Registry value.

This stage, after the high-value home support copies, gives every
England-and-Wales owner household with a value the one its region's
distribution holds at the household's weighted mid-rank among the owner
households of its FRS region, at the spine's weights, with the same ranking
the dense node uses (``property_localisation.owner_uniforms``: ties broken by
an identity-keyed draw, support copies spread over the top stratum by their
own draws). The region knots are the dense node's resource's region level
(``uk/high_value_home_value_knots.json``), from the same owner-weighted
dwelling table as the seats. ``property_wealth`` moves by the value's change.
Scotland, Northern Ireland, renters and owners without a value keep theirs,
and the council-tax band is untouched (the national band is the FRS's).

The value it replaces is kept in ``main_residence_value_rank_basis``, which the
dense node ranks on, so the dense line's values, wealth and bands do not
change; the release boundary drops the column on both lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.household_lineage import HIGH_VALUE_SUPPORT_COPY_COLUMN
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.property_localisation import (
    RANK_BASIS_COLUMN,
    SUPPORT_STRATUM_LOWER,
    ValueKnots,
    load_value_knots,
    main_residence_rank_basis,
    owner_mask,
    owner_uniforms,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "FRS_REGION_TO_KNOTS_REGION",
    "RANK_BASIS_COLUMN",
    "REGIONAL_HOME_VALUES_OPERATION_KIND",
    "REGIONAL_HOME_VALUES_STAGE_NAME",
    "REGIONAL_HOME_VALUES_STREAM",
    "UKRegionalHomeValuesResult",
    "UKRegionalHomeValuesStageTransform",
    "assign_regional_home_values",
    "main_residence_rank_basis",
    "regional_home_values_operation_parameters",
]

REGIONAL_HOME_VALUES_STAGE_NAME = "regional_home_values"
REGIONAL_HOME_VALUES_OPERATION_KIND = "assign_regional_home_values"
KNOTS_RESOURCE = "high_value_home_value_knots.json"
#: The stage's own keyed stream, apart from the dense node's.
REGIONAL_HOME_VALUES_STREAM = (
    "sha256-u53-v1",
    "uk-spine-regional-home-values-v1",
    0,
    0,
)
#: The knots' region codes for the FRS regions they cover. Wales is the
#: country code the knots use, not the placeholder region code the area
#: assignment keeps for it (``W99999999``).
FRS_REGION_TO_KNOTS_REGION: Mapping[str, str] = {
    **{
        region: code
        for region, code in FRS_REGION_TO_REGION_CODE.items()
        if code.startswith("E12")
    },
    "WALES": "W92000004",
}
REGIONAL_HOME_VALUES_MASS_CHANGE_REASON = (
    "Regional home values assigned on the source spine: household weights pass "
    "through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


def load_region_knots(knots: bytes | None = None) -> tuple[ValueKnots, dict]:
    """The packaged knots and their region level, checked like the seats'."""

    import json

    payload = (
        knots
        if knots is not None
        else files("microcosm.build.uk").joinpath(KNOTS_RESOURCE).read_bytes()
    )
    parsed = load_value_knots(payload)
    data = json.loads(payload.decode("utf-8"))
    regions = (data.get("levels") or {}).get("region") or {}
    quantiles: dict[str, np.ndarray] = {}
    for code in sorted(set(FRS_REGION_TO_KNOTS_REGION.values())):
        if code not in regions:
            raise ValueError(f"The value knots carry no region {code}.")
        values = np.asarray(regions[code]["quantiles"], dtype=float)
        if values.shape != parsed.grid.shape:
            raise ValueError(f"Region {code} knots do not match the grid.")
        if (
            np.any(~np.isfinite(values))
            or values[0] <= 0
            or np.any(values > parsed.cap)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(
                f"Region {code} knots must be positive, below the cap and "
                "strictly increasing, so a positive value stays positive and "
                "the dense node's ranks do not tie anew."
            )
        quantiles[code] = values
    return parsed, quantiles


@dataclass(frozen=True)
class UKRegionalHomeValuesResult:
    """The frame with the regional values, and the stage's aggregate receipt."""

    frame: Frame
    receipt: dict[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": REGIONAL_HOME_VALUES_STAGE_NAME, **self.receipt}


def _support_band(count: int) -> str:
    """A count as a band, never itself (UKDS disclosure control)."""

    if count < 10:
        return "<10"
    if count < 30:
        return "10-29"
    return "30+"


def assign_regional_home_values(
    frame: Frame, *, knots: bytes | None = None
) -> UKRegionalHomeValuesResult:
    """Read England-and-Wales owners' values off their region's knots."""

    validate_uk_national_frame(frame)
    household = frame.table("household").copy()
    if RANK_BASIS_COLUMN in household.columns:
        raise ValueError(
            f"{RANK_BASIS_COLUMN!r} is already stored: this stage is its only writer."
        )
    for column in ("region", "tenure_type", "main_residence_value", "property_wealth"):
        if column not in household.columns:
            raise ValueError(f"Regional home values need {column!r}.")
    parsed, region_knots = load_region_knots(knots)
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    region = household["region"].astype(str).to_numpy()
    tenure = household["tenure_type"].to_numpy()
    value = household["main_residence_value"].to_numpy(dtype=float)
    wealth = household["property_wealth"].to_numpy(dtype=float)
    support = (
        household[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy().astype(bool)
        if HIGH_VALUE_SUPPORT_COPY_COLUMN in household
        else None
    )
    u, uniform_receipt = owner_uniforms(
        spine_key=household["household_id"].astype(str).to_numpy(dtype=object),
        region=region,
        tenure_type=tenure,
        value=value,
        weight=weights,
        stream=REGIONAL_HOME_VALUES_STREAM,
        support_copy=support,
        support_stratum_lower=SUPPORT_STRATUM_LOWER,
    )
    knots_region = np.asarray(
        [FRS_REGION_TO_KNOTS_REGION.get(code, "") for code in region], dtype=object
    )
    target = owner_mask(tenure) & (value > 0) & (knots_region != "")
    new_value = value.copy()
    rewritten_by_region: dict[str, int] = {}
    for code in sorted(set(knots_region[target])):
        rows = np.flatnonzero(target & (knots_region == code))
        new_value[rows] = np.interp(u[rows], parsed.grid, region_knots[code])
        rewritten_by_region[str(code)] = int(len(rows))
    if np.any(~np.isfinite(new_value)) or np.any(new_value[target] <= 0):
        raise ValueError("Regional home values must be finite and positive.")
    new_wealth = np.maximum(wealth + (new_value - value), 0.0)
    household[RANK_BASIS_COLUMN] = value
    household["main_residence_value"] = new_value
    household["property_wealth"] = new_wealth
    household_weights = frame.weights_for("household")
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=REGIONAL_HOME_VALUES_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=frame.table("person").copy(),
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    at_cap = int((target & (new_value >= parsed.cap)).sum())
    receipt = {
        "knots_basis_year": parsed.basis_year,
        "knots_inputs_sha256": parsed.inputs_sha256,
        "rows": int(len(household)),
        "rows_rewritten_by_region": rewritten_by_region,
        "rows_at_cap": _support_band(at_cap) if at_cap else "0",
        "weighted_value_before": float(np.dot(value[target], weights[target])),
        "weighted_value_after": float(np.dot(new_value[target], weights[target])),
        "ranking": {
            key: uniform_receipt[key] for key in ("rank_basis", "support_stratum_lower")
        },
    }
    return UKRegionalHomeValuesResult(frame=result, receipt=receipt)


def regional_home_values_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            REGIONAL_HOME_VALUES_OPERATION_KIND,
            {
                "knots": KNOTS_RESOURCE,
                "knots_level": "region",
                "region_map": dict(sorted(FRS_REGION_TO_KNOTS_REGION.items())),
                "rows": (
                    "owner households (tenure owned outright or with a mortgage) "
                    "with main_residence_value > 0 in the mapped regions"
                ),
                "rank": (
                    "weighted mid-rank of main_residence_value among the owner "
                    "households of the FRS region at the spine's weights, ties "
                    "broken by a keyed draw; high-value support copies take "
                    "support_stratum_lower + (1 - support_stratum_lower) * draw"
                ),
                "support_stratum_lower": SUPPORT_STRATUM_LOWER,
                "stream": list(REGIONAL_HOME_VALUES_STREAM),
                "value": "np.interp(rank, grid, region knots)",
                "property_wealth": "moves by the value's change, floored at 0",
                "rank_basis_column": RANK_BASIS_COLUMN,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": REGIONAL_HOME_VALUES_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    _assert_closed_world_operations(stage, regional_home_values_operation_parameters())
    if stage.stage != REGIONAL_HOME_VALUES_STAGE_NAME:
        raise ValueError(
            f"Expected stage {REGIONAL_HOME_VALUES_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if tuple(stage.outputs) != (RANK_BASIS_COLUMN,) or tuple(stage.rewrites) != (
        "main_residence_value",
        "property_wealth",
    ):
        raise ValueError(
            "The regional home values stage writes the rank basis and rewrites "
            "main_residence_value and property_wealth only."
        )
    if stage.grain != "household":
        raise ValueError(
            f"The regional home values stage is household grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKRegionalHomeValuesStageTransform:
    """Whole-stage callable for the regional home values."""

    stage: SourceStageSpec
    last_result: UKRegionalHomeValuesResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = assign_regional_home_values(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (RANK_BASIS_COLUMN,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
