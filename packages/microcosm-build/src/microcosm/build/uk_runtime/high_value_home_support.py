"""UK high-value home support: light copies of the top value stratum of owner
households (microcosm#1240, fix 1).

The dense graph reads every owner clone's home value off its assigned
constituency's Land Registry sale distribution at the household's within-region
rank (``uk.full.property.value``), so where the rank lands decides whether a
clone is a £2m-plus home. The top of that rank is thin: a few hundred FRS owner
households carry the top five per cent of each region, and each seat's £2m-plus
cell then rests on the handful of them whose clones landed there. This stage
copies every owner household in the top stratum of its FRS region ``m`` times
at equal weight, each copy carrying its index in
``high_value_support_copy_index`` (0 on roots and unsplit households), the
explicit lineage the geography identity keys on; the value node then gives each
copy its own keyed rank inside the stratum instead of the household's single
rank, so the copies spread across the tail and across seats.

Deterministic and mass-conserving: no draw, no seed, no salt. Every
household's mass and the total household mass are conserved exactly; the
weighted rank that selects the stratum is the one the value node uses, so the
two agree by construction. No value moves here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.cgt_support import _exact_support_weights
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.property_localisation import (
    SUPPORT_STRATUM_LOWER,
    owner_mask,
)
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from microcosm.frame import Frame, MassChangeRecord, WeightKind

__all__ = [
    "HIGH_VALUE_SUPPORT_COPIES",
    "HIGH_VALUE_SUPPORT_COPIES_COLUMN",
    "HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON",
    "HIGH_VALUE_SUPPORT_OPERATION_KIND",
    "HIGH_VALUE_SUPPORT_STAGE_NAME",
    "HIGH_VALUE_SUPPORT_STRATUM_LOWER",
    "UKHighValueHomeSupportResult",
    "UKHighValueHomeSupportStageTransform",
    "high_value_support_operation_parameters",
    "select_high_value_support_households",
    "split_high_value_support_households",
]

HIGH_VALUE_SUPPORT_STAGE_NAME = "high_value_home_support"
HIGH_VALUE_SUPPORT_OPERATION_KIND = "split_top_value_owner_households"
#: The family's copy count on every member of a split household, root
#: included; 1 on every household the stage left alone.
HIGH_VALUE_SUPPORT_COPIES_COLUMN = "high_value_support_copies"
#: Owner households at or above this weighted within-region rank of
#: ``main_residence_value`` are the stratum (ruled top 5 per cent).
HIGH_VALUE_SUPPORT_STRATUM_LOWER = SUPPORT_STRATUM_LOWER
#: Copies per stratum household, root included (ruled six).
HIGH_VALUE_SUPPORT_COPIES = 6
HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON = (
    "High-value home support divides every owner household in the top value "
    "stratum of its region into light copies at equal weight; every household's "
    "mass and the total household mass are conserved."
)
_PERSON_ID_COLUMNS = ("person_id", "person_household_id", "person_benunit_id")
_VALUE_COLUMN = "main_residence_value"
_TENURE_COLUMN = "tenure_type"
_REGION_COLUMN = "region"


@dataclass(frozen=True)
class UKHighValueHomeSupportResult:
    """Split frame and the executed-effect receipt of the support stage."""

    frame: Frame
    region_rows: tuple[dict[str, object], ...]
    households_selected: int
    copies_created: int
    selected_mass: float
    households_before: int
    households_after: int
    owner_households: int
    old_total: float
    new_total: float
    stratum_lower: float = HIGH_VALUE_SUPPORT_STRATUM_LOWER
    copies: int = HIGH_VALUE_SUPPORT_COPIES

    def evidence(self) -> dict[str, object]:
        return {
            "stage": HIGH_VALUE_SUPPORT_STAGE_NAME,
            "regions": [dict(row) for row in self.region_rows],
            "totals": {
                "owner_households": int(self.owner_households),
                "households_selected": int(self.households_selected),
                "copies_created": int(self.copies_created),
                "selected_mass": float(self.selected_mass),
                "households_before": int(self.households_before),
                "households_after": int(self.households_after),
            },
            "mass": {
                "old_total": float(self.old_total),
                "new_total": float(self.new_total),
            },
            "parameters": {
                "stratum_lower": float(self.stratum_lower),
                "copies": int(self.copies),
            },
        }


def high_value_support_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operation the manifests must carry verbatim."""

    return (
        (
            HIGH_VALUE_SUPPORT_OPERATION_KIND,
            {
                "stratum_lower": HIGH_VALUE_SUPPORT_STRATUM_LOWER,
                "copies": HIGH_VALUE_SUPPORT_COPIES,
                "stratum_rule": (
                    "within each FRS region, owner households (tenure owned outright "
                    "or with a mortgage, main_residence_value > 0, positive weight) "
                    "ordered by main_residence_value ascending then household_id "
                    "ascending; a household whose weighted mid-rank "
                    "(cumulative weight before it plus half its own, over the "
                    "region's owner weight) is at or above stratum_lower is selected"
                ),
                "copy_rule": "every selected household becomes copies rows at weight / copies",
                "id_remapping": (
                    "id_multiplier_for_values; copy k takes every entity id + k x multiplier"
                ),
                "flag_column": HIGH_VALUE_SUPPORT_COPY_COLUMN,
                "copies_column": HIGH_VALUE_SUPPORT_COPIES_COLUMN,
                "copy_index_column": HIGH_VALUE_SUPPORT_INDEX_COLUMN,
                "values": "every column copied unchanged; no draw, no seed, no salt",
                "weight_kind_out": "importance",
                "conservation": "exact_total",
                "declared_factor": 1.0,
                "reason": HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON,
            },
        ),
    )


def select_high_value_support_households(
    *,
    household_ids: np.ndarray,
    weights: np.ndarray,
    region: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    stratum_lower: float = HIGH_VALUE_SUPPORT_STRATUM_LOWER,
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """The owner households in the top value stratum of each region.

    Returns the selection mask aligned to the inputs and one receipt row per
    region (owner households, owner mass, households selected, selected mass
    and the value rank the stratum starts at). A region without owner
    households selects nothing and says so.
    """
    ids = np.asarray(household_ids)
    weight = np.asarray(weights, dtype=float)
    region = np.asarray(region).astype(str)
    values = np.asarray(value, dtype=float)
    if not (len(ids) == len(weight) == len(region) == len(values)):
        raise ValueError("High-value support selection inputs must align.")
    if not 0 < stratum_lower < 1:
        raise ValueError("stratum_lower must be in (0, 1).")
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError(
            "High-value support selection weights must be finite and non-negative."
        )
    if not np.isfinite(values).all():
        raise ValueError("High-value support selection values must be finite.")
    owner = owner_mask(tenure_type) & (values > 0) & (weight > 0)
    selected = np.zeros(len(ids), dtype=bool)
    rows: list[dict[str, object]] = []
    for code in sorted(set(region)):
        pool = np.flatnonzero(owner & (region == code))
        if pool.size == 0:
            rows.append(
                {
                    "region": code,
                    "owner_households": 0,
                    "owner_mass": 0.0,
                    "households_selected": 0,
                    "selected_mass": 0.0,
                }
            )
            continue
        order = pool[np.lexsort((ids[pool], values[pool]))]
        pool_weight = weight[order]
        total = float(pool_weight.sum())
        cumulative = np.cumsum(pool_weight)
        mid_rank = (cumulative - pool_weight / 2.0) / total
        chosen = order[mid_rank >= stratum_lower]
        selected[chosen] = True
        rows.append(
            {
                "region": code,
                "owner_households": int(pool.size),
                "owner_mass": total,
                "households_selected": int(chosen.size),
                "selected_mass": float(weight[chosen].sum()),
            }
        )
    return selected, tuple(rows)


def split_high_value_support_households(
    frame: Frame,
    *,
    stratum_lower: float = HIGH_VALUE_SUPPORT_STRATUM_LOWER,
    copies: int = HIGH_VALUE_SUPPORT_COPIES,
) -> UKHighValueHomeSupportResult:
    """Split the top value stratum of each region's owners into light copies.

    Each selected household of weight ``w`` becomes ``copies`` rows at
    ``w / copies``; the root keeps its ids, copy ``k`` takes every entity id
    plus ``k`` times the frame's id multiplier, records ``k`` in
    ``high_value_support_copy_index`` and carries a full copy of its person
    and benefit-unit rows. The exact-total correction lands on the heaviest
    unselected incumbent so every copy weight stays bitwise ``w / copies``.
    Refuses a frame this stage has already split.
    """
    if isinstance(copies, bool) or copies < 2:
        raise ValueError("High-value support copies must be an integer of at least 2.")
    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    for column in (
        HIGH_VALUE_SUPPORT_COPY_COLUMN,
        HIGH_VALUE_SUPPORT_COPIES_COLUMN,
        HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    ):
        if column in household.columns:
            raise ValueError(
                f"The high-value home support stage already ran: the household "
                f"table carries {column!r}."
            )
    for column in (_VALUE_COLUMN, _TENURE_COLUMN, _REGION_COLUMN):
        if column not in household.columns:
            raise ValueError(
                f"The high-value home support stage needs household.{column}."
            )
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    old_total = frame.weights_for("household").total
    household_ids = household["household_id"].to_numpy()
    selected, region_rows = select_high_value_support_households(
        household_ids=household_ids,
        weights=weights,
        region=household[_REGION_COLUMN].to_numpy(),
        tenure_type=household[_TENURE_COLUMN].to_numpy(),
        value=household[_VALUE_COLUMN].to_numpy(),
        stratum_lower=stratum_lower,
    )
    copy_counts = np.ones(len(household), dtype="int64")
    copy_counts[selected] = int(copies)
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    household[HIGH_VALUE_SUPPORT_COPY_COLUMN] = False
    household[HIGH_VALUE_SUPPORT_COPIES_COLUMN] = copy_counts
    household[HIGH_VALUE_SUPPORT_INDEX_COLUMN] = np.zeros(len(household), dtype="int64")
    incumbent_weights = weights / copy_counts
    copy_person: list[pd.DataFrame] = []
    copy_benunit: list[pd.DataFrame] = []
    copy_household: list[pd.DataFrame] = []
    copy_weights: list[np.ndarray] = []
    for k in range(1, int(copy_counts.max())):
        family = copy_counts > k
        family_ids = set(household_ids[family].tolist())
        members = person["person_household_id"].isin(family_ids).to_numpy()
        family_person = person.loc[members].copy()
        family_benunit = benunit.loc[
            benunit["benunit_id"].isin(set(family_person["person_benunit_id"]))
        ].copy()
        family_household = household.loc[family].copy()
        offset = k * multiplier
        for column in _PERSON_ID_COLUMNS:
            family_person[column] = family_person[column].astype("int64") + offset
        family_benunit["benunit_id"] = (
            family_benunit["benunit_id"].astype("int64") + offset
        )
        family_household["household_id"] = (
            family_household["household_id"].astype("int64") + offset
        )
        family_household[HIGH_VALUE_SUPPORT_COPY_COLUMN] = True
        family_household[HIGH_VALUE_SUPPORT_INDEX_COLUMN] = np.full(
            int(family.sum()), k, dtype="int64"
        )
        copy_person.append(family_person)
        copy_benunit.append(family_benunit)
        copy_household.append(family_household)
        copy_weights.append(incumbent_weights[family])
    final_person = pd.concat([person, *copy_person], ignore_index=True)
    final_benunit = pd.concat([benunit, *copy_benunit], ignore_index=True)
    final_household = pd.concat([household, *copy_household], ignore_index=True)
    final_weights = np.concatenate([incumbent_weights, *copy_weights])
    exact = _exact_support_weights(
        final_weights,
        target=old_total,
        unselected=np.flatnonzero(~selected & (incumbent_weights > 0.0)),
    )
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=exact.total,
        declared_factor=1.0,
        reason=HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=final_person,
        benunit=final_benunit,
        household=final_household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=exact.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    return UKHighValueHomeSupportResult(
        frame=result,
        region_rows=tuple(
            {**row, "copies_created": int(row["households_selected"]) * (copies - 1)}
            for row in region_rows
        ),
        households_selected=int(selected.sum()),
        copies_created=int((copy_counts - 1).sum()),
        selected_mass=float(weights[selected].sum()),
        households_before=int(len(household)),
        households_after=int(len(final_household)),
        owner_households=int(sum(int(row["owner_households"]) for row in region_rows)),
        old_total=float(old_total),
        new_total=float(exact.total),
        stratum_lower=float(stratum_lower),
        copies=int(copies),
    )


def _assert_high_value_support_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage manifest to the reviewed code constants (closed world)."""
    _assert_closed_world_operations(stage, high_value_support_operation_parameters())
    if stage.stage != HIGH_VALUE_SUPPORT_STAGE_NAME:
        raise ValueError(
            f"Expected stage {HIGH_VALUE_SUPPORT_STAGE_NAME!r}, got {stage.stage!r}."
        )
    expected_outputs = (
        HIGH_VALUE_SUPPORT_COPY_COLUMN,
        HIGH_VALUE_SUPPORT_COPIES_COLUMN,
        HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    )
    if tuple(stage.outputs) != expected_outputs or tuple(stage.rewrites):
        raise ValueError(
            "The high-value home support stage writes exactly its flag, copy-count "
            f"and copy-index columns; the manifest must declare outputs "
            f"{expected_outputs} and no rewrites, got {tuple(stage.outputs)} / "
            f"{tuple(stage.rewrites)}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The high-value home support stage moves household mass; grain must be "
            f"'household', got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKHighValueHomeSupportStageTransform:
    """Whole-stage transform for the high-value home support split."""

    stage: SourceStageSpec
    last_result: UKHighValueHomeSupportResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_high_value_support_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_high_value_support_stage_parameters(self.stage)
        result = split_high_value_support_households(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (
            HIGH_VALUE_SUPPORT_COPY_COLUMN,
            HIGH_VALUE_SUPPORT_COPIES_COLUMN,
            HIGH_VALUE_SUPPORT_INDEX_COLUMN,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
