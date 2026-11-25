"""UK high-value home support: light copies of the top value stratum of owner
households (microcosm#1240, fix 1).

This stage copies every owner household in the top stratum of its FRS region
``m`` times at equal weight, each copy carrying its index in
``high_value_support_copy_index`` (0 on roots and unsplit households), the
explicit lineage the copy readers read (``household_lineage.UK_COPY_LINEAGE``).
Since microcosm#931 the
stratum ranks England and Wales owners on the values ``was_wealth`` reads off
their constituency's knots, which the regional property uprating returns them
to. Each copy then takes a place of its own, as another clone of its
household would (María's ruling of 2026-11-25): its own geography key, an
area drawn from its parent's draw inputs, so at its parent's level, its local
authority, BRMA and fuel-relief flag
(:func:`~microcosm.build.uk_runtime.spine_geography.assign_uk_high_value_copy_areas`),
and, in Great Britain, its council tax rescaled to its new authority's Band D
(:func:`~microcosm.build.uk_runtime.spine_council_tax.relocalise_uk_council_tax`).
Its value and every other column stay its parent's
(``uk_copy_area_agreement`` checks its region and clone index).

Deterministic and mass-conserving: every household's mass and the total
household mass are conserved exactly. The split's one draw is the stratum's
tie-break, a uniform keyed on each household's own id that orders
households tied on value (microcosm#931); the copies' areas are drawn on
their own keys, and Northern Ireland's BRMAs on their own ids. No value is
assigned here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import AtomicSupport
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import keyed_tie_order
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    UK_SPINE_ATOMIC_AREA_SEED,
    uk_conditioned_assignment_definition,
    uk_conditioned_supports,
)
from microcosm.build.uk_runtime.brma_area_layer import BRMAAreaLayer
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_EXACT_TOTAL_EXCEPTION,
    _exact_support_weights,
)
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_ARTIFACT_ROLE,
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_ARTIFACT_ROLE,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_BAND_D_FISCAL_YEAR,
    UKCouncilTaxBandD,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
)
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
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
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    record_source_households,
    rounded_count,
)
from microcosm.build.uk_runtime.rfdr_area_layer import RFDRAreaLayer
from microcosm.build.uk_runtime.spine_council_tax import (
    UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS,
    UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES,
    UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES,
    relocalise_uk_council_tax,
    uk_scotland_band_d_verification,
)
from microcosm.build.uk_runtime.spine_geography import (
    HIGH_VALUE_COPY_AREAS_OPERATION_KIND,
    UK_HIGH_VALUE_COPY_AREAS_REWRITE_COLUMNS,
    UKHighValueCopyAreasResult,
    assign_uk_high_value_copy_areas,
    high_value_copy_area_parameters,
    load_committed_brma_counts,
    load_committed_brma_layer,
    load_committed_rfdr_layer,
    uk_copy_area_artifacts,
    uk_copy_area_predictor_operation,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights
from microcosm.frame.rules import assert_rules_engine_country

__all__ = [
    "HIGH_VALUE_SUPPORT_COPIES",
    "HIGH_VALUE_SUPPORT_COPIES_COLUMN",
    "HIGH_VALUE_SUPPORT_COPY_ID_FIELD",
    "HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON",
    "HIGH_VALUE_SUPPORT_OPERATION_KIND",
    "HIGH_VALUE_SUPPORT_STAGE_NAME",
    "HIGH_VALUE_SUPPORT_STRATUM_LOWER",
    "HIGH_VALUE_SUPPORT_TIE_SALT",
    "HIGH_VALUE_SUPPORT_TIE_SEED",
    "UK_HIGH_VALUE_SUPPORT_REWRITES",
    "UKHighValueHomeSupportResult",
    "UKHighValueHomeSupportStageTransform",
    "high_value_support_operation_parameters",
    "high_value_support_tie_break",
    "select_high_value_support_households",
    "split_high_value_support_households",
    "support_uk_high_value_homes",
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
#: The copy-id field copy ``k`` takes ``k`` in (microcosm#931).
HIGH_VALUE_SUPPORT_COPY_ID_FIELD = "high_value_support"
#: The seed and salt of the stratum's tie-break: owners tied on value take
#: the order of a uniform keyed on their own household_id, then of their ids,
#: not of their ids alone, which always ranked a household's highest copy or
#: clone ordinal highest (microcosm#931).
HIGH_VALUE_SUPPORT_TIE_SEED = 0
HIGH_VALUE_SUPPORT_TIE_SALT = "high_value_home_support:value_tie"
HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON = (
    "High-value home support divides every owner household in the top value "
    "stratum of its region into light copies at equal weight; every household's "
    "mass and the total household mass are conserved."
)
_PERSON_ID_COLUMNS = ("person_id", "person_household_id", "person_benunit_id")
_VALUE_COLUMN = "main_residence_value"
_TENURE_COLUMN = "tenure_type"
_REGION_COLUMN = "region"
#: The columns the stage rewrites on its copies once they are made: their
#: place (the geography key, the area and ladder, the local authority, the
#: BRMA and the fuel-relief flag), then the household bill and rebate and the
#: person council tax reduction G6 wrote at the parent's authority
#: (microcosm#931).
UK_HIGH_VALUE_SUPPORT_REWRITES = (
    *UK_HIGH_VALUE_COPY_AREAS_REWRITE_COLUMNS,
    *UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES,
    *UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES,
)
#: Every artifact the copies' placement and council tax read, by (role,
#: resource): the FRS release, the names roster, the BRMA layer and counts and
#: the RFDR layer, then the two Band D resources G6 prices at.
_ARTIFACTS = (
    *uk_copy_area_artifacts(),
    (UK_BAND_D_ARTIFACT_ROLE, UK_BAND_D_ENGLAND_WALES_RESOURCE),
    (SCOTLAND_BAND_D_ARTIFACT_ROLE, SCOTLAND_BAND_D_RESOURCE),
)
_VERIFY_KIND, _VERIFY_PARAMETERS = uk_scotland_band_d_verification()


@dataclass(frozen=True)
class UKHighValueHomeSupportResult:
    """Split frame and the executed-effect receipt of the support stage.

    The split alone (:func:`split_high_value_support_households`) leaves
    ``copy_areas``, ``band_d`` and ``moved_sources`` unset; the stage
    (:func:`support_uk_high_value_homes`) sets them once its copies are
    placed and their council tax rescaled. ``moved_sources`` is exact, prints
    only rounded and is never in the repr.
    """

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
    copy_areas: UKHighValueCopyAreasResult | None = field(default=None, repr=False)
    band_d: UKCouncilTaxBandD | None = field(default=None, repr=False)
    moved_sources: int | None = field(default=None, repr=False)

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
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
        if self.copy_areas is not None:
            evidence["copy_areas"] = self.copy_areas.evidence()
        if self.band_d is not None and self.moved_sources is not None:
            evidence["council_tax"] = {
                "band_d": {
                    "england_wales": {
                        "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
                        "sha256": self.band_d.england_wales_sha256,
                    },
                    "scotland": {
                        "resource": SCOTLAND_BAND_D_RESOURCE,
                        "sha256": self.band_d.scotland_sha256,
                        "source_fact_feed": dict(self.band_d.scotland_source_fact_feed),
                    },
                },
                "counted_among": "clone_0",
                "moved_copy_source_households": rounded_count(self.moved_sources),
                "excluded_regions": list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS),
            }
        if self.copy_areas is not None or self.band_d is not None:
            evidence["rewrites"] = list(UK_HIGH_VALUE_SUPPORT_REWRITES)
        return evidence


def high_value_support_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifests must carry verbatim, in order.

    The split; the Scottish Band D resource's feed check, as G6 declares it;
    the engine predictor Northern Ireland's copies consume for their BRMA;
    then the copies' placement (:func:`high_value_copy_area_parameters`) and
    their council tax rescaled to their new authority (microcosm#931).
    """

    return (
        (
            HIGH_VALUE_SUPPORT_OPERATION_KIND,
            {
                "stratum_lower": HIGH_VALUE_SUPPORT_STRATUM_LOWER,
                "copies": HIGH_VALUE_SUPPORT_COPIES,
                "stratum_rule": (
                    "within each FRS region, owner households (tenure owned outright "
                    "or with a mortgage, main_residence_value > 0, positive weight) "
                    "ordered by main_residence_value ascending, households tied on "
                    "it in ascending order of a uniform keyed on their own "
                    "household_id under seed and salt, then household_id ascending; "
                    "a household whose weighted mid-rank (cumulative weight before "
                    "it plus half its own, over the region's owner weight) is at or "
                    "above stratum_lower is selected"
                ),
                "seed": HIGH_VALUE_SUPPORT_TIE_SEED,
                "salt": HIGH_VALUE_SUPPORT_TIE_SALT,
                "copy_rule": (
                    "every selected household becomes copies rows at weight / "
                    f"copies, {CGT_SUPPORT_EXACT_TOTAL_EXCEPTION}"
                ),
                "id_remapping": (
                    "copy-id table field high_value_support; copy k takes every "
                    "entity id + k x the field's radix (10**12)"
                ),
                "flag_column": HIGH_VALUE_SUPPORT_COPY_COLUMN,
                "copies_column": HIGH_VALUE_SUPPORT_COPIES_COLUMN,
                "copy_index_column": HIGH_VALUE_SUPPORT_INDEX_COLUMN,
                "values": (
                    "every column copied unchanged, then each copy placed and its "
                    "council tax rescaled by assign_high_value_copy_areas; its "
                    "value and every other column stay its parent's"
                ),
                "weight_kind_out": "importance",
                "conservation": "exact_total",
                "declared_factor": 1.0,
                "reason": HIGH_VALUE_SUPPORT_MASS_CHANGE_REASON,
            },
        ),
        (_VERIFY_KIND, dict(_VERIFY_PARAMETERS)),
        uk_copy_area_predictor_operation(),
        (
            HIGH_VALUE_COPY_AREAS_OPERATION_KIND,
            {
                **high_value_copy_area_parameters(),
                "council_tax": {
                    "authority_column": UK_LOCAL_AUTHORITY_CODE_COLUMN,
                    "band_d_resources": {
                        "england_wales": UK_BAND_D_ENGLAND_WALES_RESOURCE,
                        "scotland": SCOTLAND_BAND_D_RESOURCE,
                    },
                    "band_d_year": UK_BAND_D_FISCAL_YEAR,
                    "scaled_columns": {
                        "household": list(UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES),
                        "person": list(UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES),
                    },
                    "rule": (
                        "on each copy whose local_authority_code changed, every "
                        "scaled column times the new authority's Band D over the "
                        "former one's, one correctly rounded product of exact "
                        "rationals: council_tax_localise's amount at the new "
                        "authority, since the copy keeps its parent's draw cell"
                    ),
                    "excluded_regions": list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS),
                },
                "rewrites": list(UK_HIGH_VALUE_SUPPORT_REWRITES),
            },
        ),
    )


def high_value_support_tie_break() -> tuple[int, str]:
    """The declared seed and salt of the stratum's tie-break.

    Read from the reviewed operation the manifests restate verbatim
    (:func:`high_value_support_operation_parameters`), so the stage keys its
    ties on the declared pair (microcosm#931).
    """

    operation = dict(high_value_support_operation_parameters())[
        HIGH_VALUE_SUPPORT_OPERATION_KIND
    ]
    return int(operation["seed"]), str(operation["salt"])


def select_high_value_support_households(
    *,
    household_ids: np.ndarray,
    weights: np.ndarray,
    region: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    stratum_lower: float = HIGH_VALUE_SUPPORT_STRATUM_LOWER,
    seed: int,
    salt: str,
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """The owner households in the top value stratum of each region.

    Owners are ranked by value ascending, owners tied on value by a uniform
    keyed on their own household id under ``seed`` and ``salt``, then by id
    (:func:`microcosm.build.stochastic_assignment.keyed_tie_order`); ``seed``
    and ``salt`` have no default (the stage passes
    :func:`high_value_support_tie_break`). Returns the selection mask aligned
    to the inputs and one receipt row per region (owner households, owner
    mass, households selected, selected mass and the value rank the stratum
    starts at). A region without owner households selects nothing and says
    so.
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
        order = pool[keyed_tie_order(values[pool], ids[pool], seed=seed, salt=salt)]
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
    plus ``k`` times the declared radix of the copy-id table's
    ``high_value_support`` field (so every incoming id must sit below it),
    records ``k`` in ``high_value_support_copy_index`` and carries a full copy
    of its person and benefit-unit rows. The exact-total correction lands on
    an unselected household, the first of the 64 heaviest that can carry it
    (:data:`microcosm.build.uk_runtime.cgt_support.CGT_SUPPORT_CORRECTION_CANDIDATES`),
    so every root and copy weighs bitwise ``w / copies``. Only where none of
    them can does spi_support's shared fallback place it, and that row can be
    a root or copy, then off ``w / copies`` by about one ulp of the total
    (:func:`microcosm.build.uk_runtime.cgt_support._exact_support_weights`).
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
    tie_seed, tie_salt = high_value_support_tie_break()
    selected, region_rows = select_high_value_support_households(
        household_ids=household_ids,
        weights=weights,
        region=household[_REGION_COLUMN].to_numpy(),
        tenure_type=household[_TENURE_COLUMN].to_numpy(),
        value=household[_VALUE_COLUMN].to_numpy(),
        stratum_lower=stratum_lower,
        seed=tie_seed,
        salt=tie_salt,
    )
    copy_counts = np.ones(len(household), dtype="int64")
    copy_counts[selected] = int(copies)
    for table, columns in (
        (person, _PERSON_ID_COLUMNS),
        (benunit, ("benunit_id",)),
        (household, ("household_id",)),
    ):
        for column in columns:
            copy_id_table.require_below(
                table[column],
                HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
                what=f"incoming {column} values",
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
        for column in _PERSON_ID_COLUMNS:
            family_person[column] = copy_id_table.mint(
                family_person[column], HIGH_VALUE_SUPPORT_COPY_ID_FIELD, k
            )
        family_benunit["benunit_id"] = copy_id_table.mint(
            family_benunit["benunit_id"], HIGH_VALUE_SUPPORT_COPY_ID_FIELD, k
        )
        family_household["household_id"] = copy_id_table.mint(
            family_household["household_id"], HIGH_VALUE_SUPPORT_COPY_ID_FIELD, k
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
        whole=np.flatnonzero(~selected & (incumbent_weights > 0.0)),
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


def _with_tables(frame: Frame, household: pd.DataFrame, person: pd.DataFrame) -> Frame:
    """``frame`` with ``household`` and ``person`` in place of its own tables;
    ids, weights, strata, the mass log and the metadata are the frame's own."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    tables["person"] = person
    # The household weights pass through unchanged. Annotated so the stage's
    # static walk meets microcosm.frame's Weights module.
    weights: Weights = frame.weights_for("household")
    result = Frame(
        tables,
        frame.schema,
        {"household": weights},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    validate_uk_national_frame(result)
    return result


def support_uk_high_value_homes(
    frame: Frame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    engine: object,
    brma_layer: BRMAAreaLayer,
    brma_layer_sha256: str,
    count_resource: Mapping[str, Any],
    count_resource_sha256: str,
    rfdr_layer: RFDRAreaLayer,
    rfdr_layer_sha256: str,
    band_d: UKCouncilTaxBandD,
    stratum_lower: float = HIGH_VALUE_SUPPORT_STRATUM_LOWER,
    copies: int = HIGH_VALUE_SUPPORT_COPIES,
) -> UKHighValueHomeSupportResult:
    """The whole stage: the split, each copy's place, and its council tax.

    :func:`split_high_value_support_households` first, at ``stratum_lower``
    and ``copies`` (the ruled ones unless a test passes others). Then every
    copy it made takes a place of its own
    (:func:`~microcosm.build.uk_runtime.spine_geography.assign_uk_high_value_copy_areas`):
    its own key and an area drawn from its parent's draw inputs, as another
    clone of its household would be (María's ruling of 2026-11-25). Then, on
    each Great Britain copy whose authority changed, the bill, the rebate and
    the person reduction G6 wrote at its parent's authority are rescaled to
    its new authority's Band D
    (:func:`~microcosm.build.uk_runtime.spine_council_tax.relocalise_uk_council_tax`).
    Every root and every other column keeps its values. The evidence adds the
    placement's and, among clone-0 rows, the distinct source households
    whose copies changed authority, rounded.
    """

    split = split_high_value_support_households(
        frame, stratum_lower=stratum_lower, copies=copies
    )
    before = split.frame.table("household")[UK_LOCAL_AUTHORITY_CODE_COLUMN].to_numpy(
        dtype=object
    )
    placed = assign_uk_high_value_copy_areas(
        split.frame,
        stage=HIGH_VALUE_SUPPORT_STAGE_NAME,
        definition=definition,
        supports=supports,
        engine=engine,
        brma_layer=brma_layer,
        brma_layer_sha256=brma_layer_sha256,
        count_resource=count_resource,
        count_resource_sha256=count_resource_sha256,
        rfdr_layer=rfdr_layer,
        rfdr_layer_sha256=rfdr_layer_sha256,
    )
    household = placed.frame.table("household").copy()
    person = placed.frame.table("person").copy()
    rescaled = relocalise_uk_council_tax(
        household,
        person,
        rows=placed.copies,
        authorities_before=before,
        band_d=band_d,
    )
    for column, values in (
        (rescaled.council_tax.name, rescaled.council_tax),
        (rescaled.council_tax_rebate.name, rescaled.council_tax_rebate),
    ):
        household[column] = values
    person[rescaled.council_tax_benefit_reported.name] = (
        rescaled.council_tax_benefit_reported
    )
    result = _with_tables(placed.frame, household, person)
    sources = record_source_households(
        result, "household", stage=HIGH_VALUE_SUPPORT_STAGE_NAME
    )
    clone_zero = _clone_zero(household)
    return replace(
        split,
        frame=result,
        copy_areas=placed,
        band_d=band_d,
        moved_sources=distinct_count(sources, rescaled.moved & clone_zero),
    )


def _clone_zero(household: pd.DataFrame) -> np.ndarray:
    """Rows of geographic clone 0: every row of a table without the index."""

    column = ladder_clone_index_column("household")
    if column not in household.columns:
        return np.ones(len(household), dtype=bool)
    return household[column].to_numpy(dtype=np.int64) == 0


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
    if (
        tuple(stage.outputs) != expected_outputs
        or tuple(stage.rewrites) != UK_HIGH_VALUE_SUPPORT_REWRITES
    ):
        raise ValueError(
            "The high-value home support stage writes exactly its flag, copy-count "
            "and copy-index columns, and rewrites its copies' place and council "
            f"tax; the manifest must declare outputs {expected_outputs} and "
            f"rewrites {UK_HIGH_VALUE_SUPPORT_REWRITES}, got "
            f"{tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The high-value home support stage moves household mass; grain must be "
            f"'household', got {stage.grain!r}."
        )
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
    )
    if len(stage.artifacts) != len(_ARTIFACTS) or declared != sorted(_ARTIFACTS):
        raise ValueError(
            "The high-value home support stage reads the artifacts "
            f"{list(_ARTIFACTS)} (role, resource), each with "
            "runtime_sha256_required: true, and no other; the stage declares "
            f"{len(stage.artifacts)} artifact(s)."
        )


@dataclass(frozen=True)
class UKHighValueHomeSupportStageTransform:
    """Whole-stage transform for the high-value home support stage.

    ``engine`` is the UK rules engine and ``supports`` the three conditioned
    supports' bytes, keyed by system, each held to canonical bytes, as the
    SPI-copy area stage's transform takes them. ``band_d`` is the Band D the
    copies' council tax is rescaled at: the committed resources
    (``load_uk_council_tax_band_d``) unless a test supplies one. Constructing
    the transform checks the stage it runs as, builds G2's definition with
    G2's seed, decodes the supports and loads the committed BRMA layer and
    counts and the RFDR layer; running it runs the Scottish resource's
    runtime arm with the frame's period, then
    :func:`support_uk_high_value_homes`.
    """

    stage: SourceStageSpec
    engine: object = field(repr=False)
    supports: Mapping[str, bytes] = field(repr=False)
    band_d: UKCouncilTaxBandD | None = field(default=None, repr=False, compare=False)
    definition: Mapping[str, object] = field(init=False, repr=False, compare=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    brma_layer: BRMAAreaLayer = field(init=False, repr=False, compare=False)
    brma_layer_sha256: str = field(init=False, compare=False)
    count_resource: Mapping[str, Any] = field(init=False, repr=False, compare=False)
    count_resource_sha256: str = field(init=False, compare=False)
    rfdr_layer: RFDRAreaLayer = field(init=False, repr=False, compare=False)
    rfdr_layer_sha256: str = field(init=False, compare=False)
    last_result: UKHighValueHomeSupportResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_high_value_support_stage_parameters(self.stage)
        if self.band_d is not None and not isinstance(self.band_d, UKCouncilTaxBandD):
            raise TypeError(
                "The high-value home support stage rescales council tax at a "
                "UKCouncilTaxBandD."
            )
        # The definition holds every support to canonical bytes.
        definition = uk_conditioned_assignment_definition(
            self.supports, seed=UK_SPINE_ATOMIC_AREA_SEED
        )
        decoded = uk_conditioned_supports(self.supports)
        brma_layer, brma_layer_sha256 = load_committed_brma_layer()
        counts, counts_sha256 = load_committed_brma_counts()
        rfdr_layer, rfdr_layer_sha256 = load_committed_rfdr_layer()
        band_d = (
            self.band_d if self.band_d is not None else load_uk_council_tax_band_d()
        )
        for name, value in (
            ("definition", definition),
            ("decoded_supports", MappingProxyType(decoded)),
            ("brma_layer", brma_layer),
            ("brma_layer_sha256", brma_layer_sha256),
            ("count_resource", MappingProxyType(dict(counts))),
            ("count_resource_sha256", counts_sha256),
            ("rfdr_layer", rfdr_layer),
            ("rfdr_layer_sha256", rfdr_layer_sha256),
            ("band_d", band_d),
        ):
            object.__setattr__(self, name, value)

    def __call__(self, frame: Frame) -> Frame:
        _assert_high_value_support_stage_parameters(self.stage)
        assert_rules_engine_country(self.engine, "uk")
        verify = next(
            operation
            for operation in self.stage.operations
            if operation.kind == _VERIFY_KIND
        )
        check_scotland_band_d_declaration(
            verify.parameters,
            stage=HIGH_VALUE_SUPPORT_STAGE_NAME,
            period=int(uk_time_period(frame)),
        )
        result = support_uk_high_value_homes(
            frame,
            definition=self.definition,
            supports=self.decoded_supports,
            engine=self.engine,
            brma_layer=self.brma_layer,
            brma_layer_sha256=self.brma_layer_sha256,
            count_resource=self.count_resource,
            count_resource_sha256=self.count_resource_sha256,
            rfdr_layer=self.rfdr_layer,
            rfdr_layer_sha256=self.rfdr_layer_sha256,
            band_d=self.band_d,
        )
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
