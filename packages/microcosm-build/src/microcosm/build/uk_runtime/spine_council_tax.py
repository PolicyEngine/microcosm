"""The clone-first spine's council-tax localisation, G6 (microcosm#931).

The spine's ``council_tax_localise`` stage
(:class:`UKCouncilTaxLocaliseStageTransform`) prices each Great Britain
household's council tax at the local authority the spine drew for it, by
María's rulings of 2026-11-06 (rulings 1 and 2), 2026-11-13 (its place in the
spine: after ``private_rent`` and just before ``was_wealth``) and 2026-11-14
(the statute where no gross bill was observed and the band was; reductions
capped at the bill outside Northern Ireland):

- **The kept bill.** Before it rewrites anything, the stage writes
  ``council_tax_frs_bill`` on every row: ``max(council_tax -
  council_tax_rebate, 0)``, the expression
  ``was_wealth.recipient_predictors`` computed its council-tax predictor with
  before it read this column. The WAS chain's bill predictor reads it, so
  the area does not enter that predictor (ruling 2). It is float64 and
  nonnegative.
- **The reference.** Each household's draw cell is recomputed from the
  inputs the area draw (G2, ``atomic_area_assign``) read:
  :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_draw_inputs`,
  :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_draw_levels`
  and :func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_draw_frame`
  over the three conditioned supports, with G2's definition built at seed 0,
  since a level reads no stream. A Great Britain household's reference is
  the expected Band D of that cell under the draw's own weights
  (:func:`~microcosm.build.atomic_support_expectations.draw_cell_expectations`,
  exact, with each system's constraints, weight rule and default as
  ``uk_draw_levels`` passes them to the chooser): ``E = sum(w D) / sum(w)``
  over the cell's areas (the household's region, and its IMD decile at L0
  and L1), ``w`` the households in the household's band where its level
  keeps the band (L0 and L2) and all households otherwise (L1 and L3), ``D``
  each area's local authority's 2024-25 Band D. The ratio is ``r =
  D(a) / E``, ``a`` the household's own ``local_authority_code``: one
  correctly rounded division of the two exact rationals, made once for each
  distinct expected Band D and authority. Rows the SPI support channel
  copied (SPI copies and income band donors) have both draw inputs missing
  by rule, so their cell is their region by households, the L3 cell the
  SPI-copy area stage (``spi_copy_areas``) drew them from.
- **The draw-cell guard.** Every Great Britain row's drawn
  ``atomic_area_code`` must be an area of its recomputed cell (in its own
  system, with the support's codes matching every input its level keeps)
  with positive mass in the weight it samples by, and its
  ``atomic_area_system`` and ``local_authority_code`` must be that area's.
  The draw never places a household elsewhere, so the guard fails only where
  a later stage changed a draw input or the ladder changed.
- **The paths**, by nation first and channel second. Northern Ireland, which
  pays domestic rates, keeps every row as it is, with no cap. A Great
  Britain row the SPI support channel copied takes the ratio at its L3 cell,
  whatever bill provenance it inherited. A Great Britain row of the FRS
  channel follows stage 5's ``council_tax_bill_provenance``:
  ``observed_gross`` takes the ratio; ``missing`` and ``unknown_reduction``
  take the 2024-25 statute where ``council_tax_band_observed``
  (:func:`~microcosm.build.uk_runtime.council_tax_band_d.uk_statutory_council_tax`
  at the household's authority, nation and band, with the single-adult
  discount where ``council_tax_single_adult_raw`` is 1), and the ratio on
  stage 5's bill where the band was not observed; ``not_levied`` is refused.
  The ratio multiplies the bill in float64.
- **Reductions.** On every Great Britain row the rebate and the one person's
  reported council tax reduction are scaled by the same ratio and then capped
  at the new bill: ``min(r x amount, bill)``. Since the cap repeats one
  operation on equal inputs, the rebate equals the reduction after the stage
  wherever it did before.

The household weights pass through, and the stage records its household mass
receipt under its own reason. It computes with exact rationals, one correctly
rounded division for each distinct expected Band D and authority, and float64
products and minima; it calls no engine and draws nothing. After writing, it
asserts that Great Britain's reductions and rebates lie within their bills,
finite and nonnegative, that the rebate still equals the reduction wherever
it did, and that Northern Ireland is unchanged.

The evidence (:meth:`UKCouncilTaxLocaliseResult.evidence`) starts with the
stage and counts distinct true source households through ``receipt_counts``:
each FRS-channel path over every clone, as ``rounded_count`` gives it; the
SPI-channel rows' two paths among the rows of geographic clone 0; the Great
Britain households whose reductions were capped, on a ratio or a statute
path, and the Northern Ireland households whose reductions exceed their
bill, both among clone-0 rows, as ``extreme_count`` gives them. With them
come the Band D resources' digests, the supports' digests, the checks as
booleans (whether Northern Ireland's reductions lie within their bills read
among clone-0 rows, as its count is), the outputs and the rewrites. Every
field but ``clone_count`` is the same at every clone count. No ratio, bill,
extreme value, authority, cell or level is printed (G2 already counts
levels).

This module's refusals name columns, dtypes and declared constants only,
never a key, an id, a code or band of a household, a value or a count of
households.

The transform checks the stage it runs as against the reviewed declaration
this module states (name, household grain, the kept bill as its one output,
nonnegative, the three rewrites, the reviewed operations verbatim and its two
Band D artifacts), and runs the Scottish resource's runtime arm with the
frame's period before it reads a row. sources.yaml declares the stage between
``private_rent`` and ``was_wealth``, and the spine build runs it on the
conditioned supports the area stages read, held to the spine's support pins.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import AtomicSupport, validate_assignment_spec
from microcosm.build.atomic_support_expectations import draw_cell_expectations
from microcosm.build.bounded_amounts import FINITE_NONNEGATIVE, amounts_within_bound
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    COUNCIL_TAX_BAND_COLUMN,
    COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    IDENTITY_COLUMN,
    REGION_COLUMN,
    SUPPORT_CHANNEL_BRANCH,
    _system_rules,
    uk_conditioned_assignment_definition,
    uk_conditioned_supports,
    uk_draw_frame,
    uk_draw_inputs,
    uk_draw_levels,
    uk_draw_systems,
    uk_support_channel_copies,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    SYSTEMS,
)
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_ARTIFACT_ROLE,
    SCOTLAND_BAND_D_RECORD_SETS,
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_ARTIFACT_ROLE,
    UK_BAND_D_BUILD_PERIOD,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UK_BAND_D_FISCAL_YEAR,
    UK_COUNCIL_TAX_BAND_RATIOS,
    UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT,
    UKCouncilTaxBandD,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
    uk_statutory_council_tax,
)
from microcosm.build.uk_runtime.council_tax_constants import UK_NATION_BY_REGION
from microcosm.build.uk_runtime.frs_council_tax import (
    FRS_COUNCIL_TAX_BILL_MISSING,
    FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
    FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES,
    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
)
from microcosm.build.uk_runtime.geographic_clones import (
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geography_first_roster import (
    ATOMIC_AREA_ASSIGN_STAGE,
    COUNCIL_TAX_LOCALISE_STAGE,
    SPI_COPY_AREAS_STAGE,
)
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.build.uk_runtime.local_authority_input import (
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    extreme_count,
    record_source_households,
    rounded_count,
)
from microcosm.frame import Frame, Weights

__all__ = [
    "COUNCIL_TAX_FRS_BILL_COLUMN",
    "COUNCIL_TAX_LOCALISE_METHOD",
    "COUNCIL_TAX_LOCALISE_OPERATION_KIND",
    "COUNCIL_TAX_LOCALISE_STAGE_NAME",
    "UK_COUNCIL_TAX_LOCALISE_CHECKS",
    "UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS",
    "UK_COUNCIL_TAX_LOCALISE_FRS_PATHS",
    "UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_INPUTS",
    "UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES",
    "UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON",
    "UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS",
    "UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES",
    "UK_COUNCIL_TAX_LOCALISE_REWRITES",
    "UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS",
    "UKCouncilTaxLocaliseResult",
    "UKCouncilTaxLocaliseStageTransform",
    "UKCouncilTaxReferences",
    "UKSpineCouncilTax",
    "council_tax_localise_operation_parameters",
    "localise_uk_council_tax",
    "localise_uk_spine_council_tax",
    "uk_council_tax_band_ratio_text",
    "uk_council_tax_references",
]

#: The stage's name, as the geography-first roster leaf states it.
COUNCIL_TAX_LOCALISE_STAGE_NAME = COUNCIL_TAX_LOCALISE_STAGE
COUNCIL_TAX_LOCALISE_OPERATION_KIND = "localise_council_tax"
COUNCIL_TAX_LOCALISE_METHOD = "uk-council-tax-localise-v1"
#: The paid bill from before the stage rewrites the bill and the rebate: the
#: stage's one output, nonnegative.
COUNCIL_TAX_FRS_BILL_COLUMN = "council_tax_frs_bill"
UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS = (COUNCIL_TAX_FRS_BILL_COLUMN,)
_BILL_COLUMN = "council_tax"
_REBATE_COLUMN = "council_tax_rebate"
_REDUCTION_COLUMN = "council_tax_benefit_reported"
_SINGLE_ADULT_COLUMN = "council_tax_single_adult_raw"
_HOUSEHOLD_ID_COLUMN = "household_id"
_PERSON_HOUSEHOLD_COLUMN = "person_household_id"
#: The columns the stage rewrites: the bill and the rebate on the household
#: table, the reported council tax reduction on the person table.
UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES = (_BILL_COLUMN, _REBATE_COLUMN)
UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES = (_REDUCTION_COLUMN,)
UK_COUNCIL_TAX_LOCALISE_REWRITES = (
    *UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES,
    *UK_COUNCIL_TAX_LOCALISE_PERSON_REWRITES,
)
#: The household-mass receipt this stage records (the reviewed
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON = (
    "Council tax localised by each household's local authority on the source "
    "spine: household weights pass through unchanged and total household mass "
    "is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)
#: The two Band D artifacts the stage declares, (role, resource): England's
#: and Wales's, and Scotland's vendored facts.
_ARTIFACTS = (
    (UK_BAND_D_ARTIFACT_ROLE, UK_BAND_D_ENGLAND_WALES_RESOURCE),
    (SCOTLAND_BAND_D_ARTIFACT_ROLE, SCOTLAND_BAND_D_RESOURCE),
)
_VERIFY_VENDORED_FACT_RESOURCE_KIND = "verify_vendored_fact_resource"
_DERIVE_KIND = "derive"
_RECORD_MASS_KIND = "record_mass_conservation_receipt"
#: The regions whose nation levies no council tax, so the statute prices no
#: band there: Northern Ireland, which pays domestic rates. The stage leaves
#: their rows as they are.
UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS: tuple[str, ...] = tuple(
    sorted(
        region
        for region, nation in UK_NATION_BY_REGION.items()
        if nation not in UK_COUNCIL_TAX_BAND_RATIOS
    )
)
#: The seed G6 builds G2's definition at: a level reads no stream, so no seed
#: moves a reference.
_LEVEL_SEED = 0
#: How far above its bill a reduction may sit in the stage's own checks: the
#: cap makes every Great Britain reduction exactly at most its bill.
_WITHIN_BILL_TOLERANCE = 0.0
#: The support channel's lineage columns: the stage reads which rows it
#: copied.
_SUPPORT_CHANNEL_COLUMNS = next(
    branch.columns
    for branch in UK_COPY_LINEAGE.branches
    if branch.name == SUPPORT_CHANNEL_BRANCH
)
#: The household columns the stage reads, besides ``source_household_id``,
#: the true source the receipts' reader takes, and the clone index the
#: evidence counts clone 0 by: the draw's inputs, the drawn area, its system
#: and authority, the bill's provenance and the statute's single-adult flag,
#: the bill and rebate it rewrites, and the support channel's lineage.
UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_INPUTS = (
    IDENTITY_COLUMN,
    REGION_COLUMN,
    IMD_DECILE_COLUMN,
    COUNCIL_TAX_BAND_COLUMN,
    COUNCIL_TAX_BAND_OBSERVED_COLUMN,
    ASSIGNMENT_COLUMNS[0],
    ASSIGNMENT_COLUMNS[1],
    UK_LOCAL_AUTHORITY_CODE_COLUMN,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
    _SINGLE_ADULT_COLUMN,
    _BILL_COLUMN,
    _REBATE_COLUMN,
    *_SUPPORT_CHANNEL_COLUMNS,
)
_CLONE_INDEX_COLUMN = ladder_clone_index_column("household")
_FLOAT64 = np.dtype("float64")
_WHAT = "The council-tax localisation"

#: Each row's path through the stage. FRS-channel rows take one of the six
#: in :data:`UK_COUNCIL_TAX_LOCALISE_FRS_PATHS`; rows the support channel
#: copied one of the two in :data:`UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS`.
_RATIO_OBSERVED_GROSS = "ratio_observed_gross"
_RATIO_MISSING_UNBANDED = "ratio_missing_unbanded"
_RATIO_UNKNOWN_REDUCTION_UNBANDED = "ratio_unknown_reduction_unbanded"
_FORMULA_MISSING = "formula_missing"
_FORMULA_UNKNOWN_REDUCTION = "formula_unknown_reduction"
_RATIO_SUPPORT_CHANNEL = "ratio_support_channel"
_UNCHANGED_NORTHERN_IRELAND = "unchanged_northern_ireland"
UK_COUNCIL_TAX_LOCALISE_FRS_PATHS = (
    _RATIO_OBSERVED_GROSS,
    _RATIO_MISSING_UNBANDED,
    _RATIO_UNKNOWN_REDUCTION_UNBANDED,
    _FORMULA_MISSING,
    _FORMULA_UNKNOWN_REDUCTION,
    _UNCHANGED_NORTHERN_IRELAND,
)
UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS = (
    _RATIO_SUPPORT_CHANNEL,
    _UNCHANGED_NORTHERN_IRELAND,
)
_RATIO_PATHS = (
    _RATIO_OBSERVED_GROSS,
    _RATIO_MISSING_UNBANDED,
    _RATIO_UNKNOWN_REDUCTION_UNBANDED,
    _RATIO_SUPPORT_CHANNEL,
)
_FORMULA_PATHS = (_FORMULA_MISSING, _FORMULA_UNKNOWN_REDUCTION)
#: Each FRS-channel bill provenance and whether the band was observed, by the
#: path it takes in Great Britain.
_FRS_PATH_BY_PROVENANCE = MappingProxyType(
    {
        (FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS, True): _RATIO_OBSERVED_GROSS,
        (FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS, False): _RATIO_OBSERVED_GROSS,
        (FRS_COUNCIL_TAX_BILL_MISSING, True): _FORMULA_MISSING,
        (FRS_COUNCIL_TAX_BILL_MISSING, False): _RATIO_MISSING_UNBANDED,
        (FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION, True): _FORMULA_UNKNOWN_REDUCTION,
        (FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION, False): (
            _RATIO_UNKNOWN_REDUCTION_UNBANDED
        ),
    }
)
#: The stage's checks, in the evidence's order. The first four must hold; the
#: last reports whether Northern Ireland's reductions, which the stage leaves
#: as they are, lie within their bills among the rows of geographic clone 0.
UK_COUNCIL_TAX_LOCALISE_CHECKS = (
    "reduction_within_bill",
    "rebate_within_bill",
    "rebate_equals_reduction",
    "northern_ireland_unchanged",
    "northern_ireland_reductions_within_bill",
)
_REQUIRED_CHECKS = UK_COUNCIL_TAX_LOCALISE_CHECKS[:4]


# --- The declaration ----------------------------------------------------------------


def uk_council_tax_band_ratio_text() -> dict[str, dict[str, str]]:
    """The statute's band proportions as the declaration states them.

    For each nation that levies council tax, each band's proportion to Band D
    from ``council_tax_band_d.UK_COUNCIL_TAX_BAND_RATIOS``, written
    ``"numerator/denominator"`` over the least common denominator of the
    nation's proportions, the statute's own: ninths in England and Wales, and
    three-hundred-and-sixtieths in Scotland.
    """

    text = {}
    for nation, ratios in UK_COUNCIL_TAX_BAND_RATIOS.items():
        denominator = math.lcm(*(ratio.denominator for ratio in ratios.values()))
        text[nation] = {
            band: f"{ratio.numerator * denominator // ratio.denominator}/{denominator}"
            for band, ratio in ratios.items()
        }
    return text


def council_tax_localise_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations G6's stage must declare verbatim, in order.

    The Scottish Band D resource's feed check; the kept paid bill, derived
    before the stage rewrites the bill and the rebate; the localisation (its
    method, Band D resources and year, the authority column, the reference and
    the stages whose draws it recomputes, the supports, the bill provenance
    classes that take the ratio or the statute and what the statute needs,
    the band proportions and single-adult discount, the scaled columns, the
    cap, the excluded regions and the support channel's rows); then the
    household mass receipt.
    """

    return (
        (
            _VERIFY_VENDORED_FACT_RESOURCE_KIND,
            {
                "artifact_role": SCOTLAND_BAND_D_ARTIFACT_ROLE,
                "resource": SCOTLAND_BAND_D_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(SCOTLAND_BAND_D_RECORD_SETS),
                "source_vintage": "2024-25 (ScotGov CTAS 2026)",
                "mapped_build_period": UK_BAND_D_BUILD_PERIOD,
                "period_mapping": "fiscal_year_2024_25_is_the_build_period",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            _DERIVE_KIND,
            {
                "output": COUNCIL_TAX_FRS_BILL_COLUMN,
                "formula": (
                    "max(council_tax - council_tax_rebate, 0) on every row, from "
                    "the bill and rebate before this stage rewrites them: the paid "
                    "bill was_wealth's council-tax predictor reads, so the area "
                    "does not enter that predictor (ruling 2)"
                ),
            },
        ),
        (
            COUNCIL_TAX_LOCALISE_OPERATION_KIND,
            {
                "method": COUNCIL_TAX_LOCALISE_METHOD,
                "band_d_resources": {
                    "england_wales": UK_BAND_D_ENGLAND_WALES_RESOURCE,
                    "scotland": SCOTLAND_BAND_D_RESOURCE,
                },
                "band_d_year": UK_BAND_D_FISCAL_YEAR,
                "authority_column": UK_LOCAL_AUTHORITY_CODE_COLUMN,
                "reference": "expected_band_d_of_draw_cell",
                "draw_stage": ATOMIC_AREA_ASSIGN_STAGE,
                "copy_draw_stage": SPI_COPY_AREAS_STAGE,
                "support_sources": {
                    system: CONDITIONED_SOURCES[system] for system in SYSTEMS
                },
                "provenance_column": FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
                "ratio_classes": [FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS],
                "formula_classes": [
                    FRS_COUNCIL_TAX_BILL_MISSING,
                    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
                ],
                "formula_needs": COUNCIL_TAX_BAND_OBSERVED_COLUMN,
                "band_ratios": uk_council_tax_band_ratio_text(),
                "single_adult_column": _SINGLE_ADULT_COLUMN,
                "single_adult_discount": float(UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT),
                "scaled_columns": {
                    "household": [_REBATE_COLUMN],
                    "person": [_REDUCTION_COLUMN],
                },
                "cap": "reduction_and_rebate_at_bill",
                "excluded_regions": list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS),
                "support_channel_rows": "ratio_at_copy_draw_level",
            },
        ),
        (
            _RECORD_MASS_KIND,
            {
                "entity": "household",
                "reason": UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def _assert_council_tax_localise_stage(stage: SourceStageSpec) -> None:
    """The stage the transform runs as: its name, household grain, the kept
    bill as its one output and its one nonnegative output, the three
    rewrites, the reviewed operations verbatim, and exactly its two Band D
    artifacts, each with ``runtime_sha256_required``."""

    if not isinstance(stage, SourceStageSpec):
        raise TypeError(f"{_WHAT} runs as a SourceStageSpec stage.")
    if stage.stage != COUNCIL_TAX_LOCALISE_STAGE_NAME:
        raise ValueError(
            f"Expected stage {COUNCIL_TAX_LOCALISE_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if stage.grain != "household":
        raise ValueError(
            f"{_WHAT} rewrites households; grain must be 'household', got "
            f"{stage.grain!r}."
        )
    if (
        tuple(stage.outputs) != UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS
        or tuple(stage.nonnegative_outputs) != UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS
        or tuple(stage.rewrites) != UK_COUNCIL_TAX_LOCALISE_REWRITES
    ):
        raise ValueError(
            f"{_WHAT} writes exactly its own columns; the stage must declare the "
            f"outputs and nonnegative outputs {UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS} "
            f"and the rewrites {UK_COUNCIL_TAX_LOCALISE_REWRITES}, got "
            f"{tuple(stage.outputs)} / {tuple(stage.nonnegative_outputs)} / "
            f"{tuple(stage.rewrites)}."
        )
    _assert_closed_world_operations(stage, council_tax_localise_operation_parameters())
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
    )
    if len(stage.artifacts) != len(_ARTIFACTS) or declared != sorted(_ARTIFACTS):
        raise ValueError(
            f"{_WHAT} reads the artifacts {list(_ARTIFACTS)} (role, resource), each "
            "with runtime_sha256_required: true, and no other; the stage declares "
            "another, or lacks one of them or its flag."
        )


def _operation(stage: SourceStageSpec, kind: str) -> Mapping[str, Any]:
    """The parameters of the stage's one operation of ``kind``."""

    (operation,) = (op for op in stage.operations if op.kind == kind)
    return operation.parameters


# --- Inputs -------------------------------------------------------------------------


def _require_columns(table: pd.DataFrame, columns: Sequence[str], entity: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"{_WHAT} reads {entity} {missing}, which earlier stages write; the "
            "frame lacks them (microcosm#931)."
        )


def _amounts(table: pd.DataFrame, column: str, entity: str) -> np.ndarray:
    """A rewritten column as float64, every value finite and nonnegative."""

    dtype = table[column].dtype
    if dtype != _FLOAT64:
        raise ValueError(
            f"{_WHAT} rewrites a float64 {entity} {column!r}, bit for bit where it "
            f"keeps it; the column holds {dtype}."
        )
    values = table[column].to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(values) & (values >= 0.0))):
        raise ValueError(
            f"{_WHAT} reads {entity} {column!r} as a finite nonnegative amount on "
            "every row; some value is NaN, infinite or negative."
        )
    return values


def _codes(table: pd.DataFrame, column: str, rows: np.ndarray) -> np.ndarray:
    """A code column's values on ``rows``, each a nonempty string."""

    values = table[column].to_numpy(dtype=object)[rows]
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(
            f"{_WHAT} reads a nonempty {column!r} on every Great Britain "
            "household; some household lacks one."
        )
    return values


def _texts(values: object) -> np.ndarray:
    """``values`` as an object array with every value that is not a string,
    a missing one among them, as ``None``, so an elementwise comparison with
    strings reads it as unequal."""

    texts = np.asarray(values, dtype=object).copy()
    texts[[not isinstance(value, str) for value in texts]] = None
    return texts


def _provenance(household: pd.DataFrame) -> np.ndarray:
    """Stage 5's bill provenance, one of its four values on every row."""

    values = household[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN].to_numpy(dtype=object)
    allowed = set(FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES)
    if not all(isinstance(value, str) and value in allowed for value in values):
        raise ValueError(
            f"{_WHAT} reads {FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN!r} as one of "
            f"{list(FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES)} on every household; "
            "some household holds another value or none."
        )
    return values.astype(str)


def _single_adults(household: pd.DataFrame) -> np.ndarray:
    """Where ``council_tax_single_adult_raw`` is 1, as the statute reads it."""

    column = household[_SINGLE_ADULT_COLUMN]
    if pd.api.types.is_bool_dtype(column.dtype) or not pd.api.types.is_numeric_dtype(
        column.dtype
    ):
        raise ValueError(
            f"{_WHAT} reads a numeric {_SINGLE_ADULT_COLUMN!r}; the column holds "
            f"{column.dtype}."
        )
    return column.to_numpy(dtype=float) == 1.0


def _great_britain(household: pd.DataFrame) -> np.ndarray:
    """Rows outside :data:`UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS`."""

    return ~np.isin(
        household[REGION_COLUMN].to_numpy(dtype=object),
        list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS),
    )


def _clone_zero(household: pd.DataFrame) -> np.ndarray:
    """Rows of geographic clone 0: every row of a table without the index."""

    if _CLONE_INDEX_COLUMN not in household.columns:
        return np.ones(len(household), dtype=bool)
    return household[_CLONE_INDEX_COLUMN].to_numpy(dtype=np.int64) == 0


@dataclass(frozen=True)
class _Members:
    """Each person's household position, and each household's reduction and
    count of persons carrying a positive one."""

    owner: np.ndarray
    reduction: np.ndarray
    household_reduction: np.ndarray
    carriers: np.ndarray


def _members(household: pd.DataFrame, person: pd.DataFrame) -> _Members:
    ids = pd.Index(household[_HOUSEHOLD_ID_COLUMN].to_numpy())
    owner = ids.get_indexer(person[_PERSON_HOUSEHOLD_COLUMN].to_numpy())
    if not bool(np.all(owner >= 0)):
        raise ValueError(
            f"{_WHAT} sums each person's {_REDUCTION_COLUMN!r} to the household its "
            f"{_PERSON_HOUSEHOLD_COLUMN!r} names; some person names none."
        )
    reduction = _amounts(person, _REDUCTION_COLUMN, "person")
    carriers = np.bincount(owner, weights=reduction > 0.0, minlength=len(household))
    if bool(np.any(carriers > 1)):
        raise ValueError(
            f"{_WHAT} scales one carrier of {_REDUCTION_COLUMN!r} per household, the "
            "household reference person the root and the housing shell write it "
            "on; some household holds more than one positive value."
        )
    return _Members(
        owner=owner,
        reduction=reduction,
        household_reduction=np.bincount(
            owner, weights=reduction, minlength=len(household)
        ),
        carriers=carriers,
    )


# --- The references -----------------------------------------------------------------


@dataclass(frozen=True)
class UKCouncilTaxReferences:
    """Each household row's draw cell reference, in row order.

    ``systems`` and ``levels`` are the area system and fallback level G2's
    chooser gives the row; ``great_britain`` marks the rows outside
    :data:`UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS`. On those rows,
    ``expected_band_d`` is the exact expected Band D of the row's draw cell,
    ``authority_band_d`` the exact Band D of its own authority and ``ratios``
    their quotient, correctly rounded to float64; elsewhere they are ``None``
    and NaN. Nothing here prints in the repr.
    """

    systems: np.ndarray = field(repr=False)
    levels: np.ndarray = field(repr=False)
    great_britain: np.ndarray = field(repr=False)
    expected_band_d: tuple[Fraction | None, ...] = field(repr=False)
    authority_band_d: tuple[Fraction | None, ...] = field(repr=False)
    ratios: np.ndarray = field(repr=False)


def _draw_weights(
    at_level: pd.DataFrame, rows: np.ndarray, system: Mapping[str, Any]
) -> np.ndarray:
    """The weight column each row samples by at its level: the band's where
    its band input is present, the default otherwise."""

    weight_by, default = _system_rules(system)
    weights = np.full(len(rows), default, dtype=object)
    if weight_by is not None:
        bands = at_level[weight_by["input"]].iloc[rows]
        present = ~bands.isna().to_numpy(dtype=bool)
        codes = bands.to_numpy(dtype=object)
        weights[present] = [weight_by["map"][code] for code in codes[present]]
    return weights


def _check_draw_cells(
    household: pd.DataFrame,
    at_level: pd.DataFrame,
    systems: np.ndarray,
    great_britain: np.ndarray,
    spec: Mapping[str, Any],
    supports: Mapping[str, AtomicSupport],
) -> None:
    """The draw-cell guard: every Great Britain row's drawn area is an area of
    its recomputed cell with positive mass in the weight it samples by, and
    its drawn system and authority are that area's."""

    for system in spec["systems"]:
        rows = np.flatnonzero(great_britain & (systems == system["id"]))
        if not len(rows):
            continue
        support = supports[system["id"]]
        area = _codes(household, ASSIGNMENT_COLUMNS[0], rows)
        position = pd.Index(
            np.asarray(support.arrays["area"], dtype=object)
        ).get_indexer(area)
        held = position >= 0
        at = np.where(held, position, 0)
        held &= (
            _texts(household[ASSIGNMENT_COLUMNS[1]].to_numpy(dtype=object)[rows])
            == (system["id"])
        )
        for constraint in system["constraints"]:
            inputs = at_level[constraint["input"]].iloc[rows]
            present = ~inputs.isna().to_numpy(dtype=bool)
            codes = np.asarray(support.arrays[constraint["support"]], dtype=object)[at]
            held &= ~present | (codes == _texts(inputs.to_numpy(dtype=object)))
        weights = _draw_weights(at_level, rows, system)
        mass = np.zeros(len(rows), dtype=bool)
        for weight in dict.fromkeys(weights.tolist()):
            sampled = weights == weight
            mass[sampled] = (
                np.asarray(support.arrays[weight], dtype=np.int64)[at[sampled]] > 0
            )
        held &= mass
        authorities = np.asarray(
            support.arrays[UK_LOCAL_AUTHORITY_CODE_COLUMN], dtype=object
        )[at]
        held &= authorities == _codes(household, UK_LOCAL_AUTHORITY_CODE_COLUMN, rows)
        if not bool(held.all()):
            raise ValueError(
                f"{_WHAT} prices each Great Britain household against the draw cell "
                f"its draw inputs give it at its level; a household's drawn "
                f"{ASSIGNMENT_COLUMNS[0]!r} is not an area of that cell with mass in "
                f"the weight it samples by, or its {ASSIGNMENT_COLUMNS[1]!r} or "
                f"{UK_LOCAL_AUTHORITY_CODE_COLUMN!r} is not that area's: a stage "
                "after the draw changed a draw input, or the ladder changed "
                "(microcosm#931)."
            )


def uk_council_tax_references(
    household: pd.DataFrame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    band_d: UKCouncilTaxBandD,
) -> UKCouncilTaxReferences:
    """Each household row's expected Band D under its draw cell, and its ratio.

    ``definition`` is G2's
    (:func:`~microcosm.build.uk_runtime.atomic_area_conditioning.uk_conditioned_assignment_definition`)
    and ``supports`` the three decoded conditioned supports it declares. The
    row's level is recomputed as G2 chose it (``uk_draw_inputs``,
    ``uk_draw_levels``, ``uk_draw_frame``); a row the SPI support channel
    copied has both inputs missing, so it takes L3. For every Great Britain
    row, the draw-cell guard runs, then
    :func:`~microcosm.build.atomic_support_expectations.draw_cell_expectations`
    gives the exact mean of its cell's areas' authorities' Band D, weighted by
    the column the row samples by, with each system's constraints, weight
    rule and default as ``uk_draw_levels`` passes them to the chooser. Its
    ratio is its own authority's Band D over that mean: one correctly rounded
    division of the two exact rationals, made once for each distinct expected
    Band D and authority.

    Refused, naming columns only: a ``band_d`` that is not a
    :class:`~microcosm.build.uk_runtime.council_tax_band_d.UKCouncilTaxBandD`;
    what the draw's adapter and chooser refuse; a Great Britain row without
    an authority, or whose authority has no 2024-25 Band D; the draw-cell
    guard; and a cell area with mass whose authority has no Band D.
    """

    if not isinstance(band_d, UKCouncilTaxBandD):
        raise TypeError(f"{_WHAT} prices bills at a UKCouncilTaxBandD.")
    spec = validate_assignment_spec(definition)
    inputs = uk_draw_inputs(household)
    levels = uk_draw_levels(inputs, spec, supports)
    at_level = uk_draw_frame(inputs, levels)
    systems = uk_draw_systems(inputs, spec)
    rows = len(household)
    great_britain = _great_britain(household)
    gb_rows = np.flatnonzero(great_britain)
    authorities = np.full(rows, None, dtype=object)
    authorities[gb_rows] = _codes(household, UK_LOCAL_AUTHORITY_CODE_COLUMN, gb_rows)
    values = band_d.band_d
    if not all(code in values for code in authorities[gb_rows]):
        raise ValueError(
            f"{_WHAT} prices each Great Britain household at its authority's "
            f"{UK_BAND_D_FISCAL_YEAR} Band D; some household's "
            f"{UK_LOCAL_AUTHORITY_CODE_COLUMN!r} has none."
        )
    _check_draw_cells(household, at_level, systems, great_britain, spec, supports)
    expected = np.full(rows, None, dtype=object)
    for system in spec["systems"]:
        members = np.flatnonzero(great_britain & (systems == system["id"]))
        if not len(members):
            continue
        weight_by, default = _system_rules(system)
        try:
            means = draw_cell_expectations(
                supports[system["id"]],
                at_level.iloc[members],
                constraint_columns={
                    constraint["input"]: constraint["support"]
                    for constraint in system["constraints"]
                },
                weight_by=weight_by,
                default_weight=default,
                value_column=UK_LOCAL_AUTHORITY_CODE_COLUMN,
                values=values,
            )
        except ValueError as error:
            raise ValueError(
                f"{_WHAT} takes the expected {UK_BAND_D_FISCAL_YEAR} Band D of each "
                f"Great Britain household's draw cell: {error}"
            ) from error
        table = np.empty(len(means), dtype=object)
        table[:] = means
        expected[members] = table
    own = np.full(rows, None, dtype=object)
    ratios = np.full(rows, np.nan, dtype=np.float64)
    quotients: dict[tuple[Fraction, str], float] = {}
    for row in gb_rows.tolist():
        code = authorities[row]
        own[row] = values[code]
        key = (expected[row], code)
        if key not in quotients:
            # One correctly rounded division of exact rationals.
            quotients[key] = float(values[code] / expected[row])
        ratios[row] = quotients[key]
    return UKCouncilTaxReferences(
        systems=systems,
        levels=np.asarray(levels),
        great_britain=great_britain,
        expected_band_d=tuple(expected.tolist()),
        authority_band_d=tuple(own.tolist()),
        ratios=ratios,
    )


# --- The localisation ---------------------------------------------------------------


@dataclass(frozen=True)
class UKSpineCouncilTax:
    """G6 on one household table and its persons: the rewritten columns and
    the kept bill, each row's path, whether the support channel copied it,
    and whether its reductions were capped, with the references. Nothing
    here prints in the repr."""

    council_tax: pd.Series = field(repr=False)
    council_tax_rebate: pd.Series = field(repr=False)
    council_tax_benefit_reported: pd.Series = field(repr=False)
    council_tax_frs_bill: pd.Series = field(repr=False)
    paths: np.ndarray = field(repr=False)
    support_channel: np.ndarray = field(repr=False)
    capped: np.ndarray = field(repr=False)
    references: UKCouncilTaxReferences = field(repr=False)


def _paths(
    household: pd.DataFrame,
    *,
    great_britain: np.ndarray,
    support_channel: np.ndarray,
) -> np.ndarray:
    """Each row's path: Northern Ireland first, then the support channel's
    rows, then the FRS channel by its bill provenance and observed band."""

    provenance = _provenance(household)
    not_levied = provenance == FRS_COUNCIL_TAX_BILL_NOT_LEVIED
    if bool(np.any(great_britain & not_levied)):
        raise ValueError(
            f"{_WHAT} localises council tax outside "
            f"{list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS)}; a household there "
            f"reads {FRS_COUNCIL_TAX_BILL_NOT_LEVIED!r} in "
            f"{FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN!r}."
        )
    if bool(np.any(~great_britain & ~support_channel & ~not_levied)):
        raise ValueError(
            f"{_WHAT} leaves {list(UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS)} as it "
            f"is; an FRS-channel household there does not read "
            f"{FRS_COUNCIL_TAX_BILL_NOT_LEVIED!r} in "
            f"{FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN!r}."
        )
    banded = household[COUNCIL_TAX_BAND_OBSERVED_COLUMN].to_numpy(dtype=bool)
    paths = np.full(len(household), _UNCHANGED_NORTHERN_IRELAND, dtype=object)
    paths[great_britain & support_channel] = _RATIO_SUPPORT_CHANNEL
    frs = np.flatnonzero(great_britain & ~support_channel)
    paths[frs] = [
        _FRS_PATH_BY_PROVENANCE[(value, bool(flag))]
        for value, flag in zip(provenance[frs], banded[frs], strict=True)
    ]
    return paths


def _statutory_bills(
    household: pd.DataFrame,
    rows: np.ndarray,
    authority_band_d: Sequence[Fraction | None],
) -> np.ndarray:
    """The 2024-25 statute's bill on ``rows``, as float64: each row's
    authority's Band D at its nation's proportion for its band, less the
    single-adult discount where it applies."""

    regions = household[REGION_COLUMN].to_numpy(dtype=object)[rows]
    nations = [UK_NATION_BY_REGION.get(region) for region in regions]
    bands = household[COUNCIL_TAX_BAND_COLUMN].to_numpy(dtype=object)[rows]
    if not all(
        nation in UK_COUNCIL_TAX_BAND_RATIOS
        and band in UK_COUNCIL_TAX_BAND_RATIOS[nation]
        for nation, band in zip(nations, bands, strict=True)
    ):
        raise ValueError(
            f"{_WHAT} prices a household by the statute at its observed "
            f"{COUNCIL_TAX_BAND_COLUMN!r}; some such band is not one its region's "
            "nation prices."
        )
    single = _single_adults(household)[rows]
    bills: dict[tuple[Fraction, str, str, bool], float] = {}
    result = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows.tolist()):
        key = (
            authority_band_d[row],
            nations[position],
            str(bands[position]),
            bool(single[position]),
        )
        if key not in bills:
            bills[key] = float(
                uk_statutory_council_tax(key[0], key[1], key[2], single_adult=key[3])
            )
        result[position] = bills[key]
    return result


def localise_uk_council_tax(
    household: pd.DataFrame,
    person: pd.DataFrame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    band_d: UKCouncilTaxBandD,
) -> UKSpineCouncilTax:
    """Every household row's localised bill, rebate and reduction, and the
    kept paid bill (G6).

    The kept bill first, from the columns as they arrive: ``max(council_tax
    - council_tax_rebate, 0)`` on every row, the paid bill
    ``was_wealth.recipient_predictors`` reads as its council-tax predictor. Then
    :func:`uk_council_tax_references` and each row's path: Northern Ireland's
    rows keep every column bit for bit; a Great Britain row of the support
    channel, or of the FRS channel whose bill provenance is
    ``observed_gross``, or ``missing`` or ``unknown_reduction`` without an
    observed band, takes ``r x council_tax``; a Great Britain FRS row whose
    provenance is ``missing`` or ``unknown_reduction`` with an observed band
    takes the 2024-25 statute. On every Great Britain row the rebate and the
    reduction take ``min(r x amount, bill)``. Returns the columns on their
    tables' indexes and in their row order, float64; a row's values depend
    on its own inputs and its cell alone, never on the tables' row order.

    Refused, naming columns and declared constants only: a missing column; a
    bill, rebate or reduction that is not float64, or not finite and
    nonnegative on every row; a household with more than one positive
    reduction; a person whose household is absent; a provenance outside
    stage 5's four values, ``not_levied`` in Great Britain or another value
    on a Northern Ireland FRS row; a statute row whose band its nation does
    not price; and every refusal of :func:`uk_council_tax_references`.
    """

    _require_columns(household, UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_INPUTS, "household")
    _require_columns(person, (_PERSON_HOUSEHOLD_COLUMN, _REDUCTION_COLUMN), "person")
    _amounts(household, _BILL_COLUMN, "household")
    _amounts(household, _REBATE_COLUMN, "household")
    # The paid bill from before any rewrite, the one was_wealth's council-tax
    # predictor reads.
    kept = np.maximum(
        household[_BILL_COLUMN].to_numpy(dtype=float)
        - household[_REBATE_COLUMN].to_numpy(dtype=float),
        0.0,
    )
    members = _members(household, person)
    support_channel = uk_support_channel_copies(household)
    great_britain = _great_britain(household)
    paths = _paths(
        household, great_britain=great_britain, support_channel=support_channel
    )
    references = uk_council_tax_references(
        household, definition=definition, supports=supports, band_d=band_d
    )
    bill = household[_BILL_COLUMN].to_numpy(dtype=float)
    rebate = household[_REBATE_COLUMN].to_numpy(dtype=float)
    ratio = references.ratios
    new_bill = bill.copy()
    ratio_rows = np.flatnonzero(np.isin(paths, _RATIO_PATHS))
    new_bill[ratio_rows] = ratio[ratio_rows] * bill[ratio_rows]
    formula_rows = np.flatnonzero(np.isin(paths, _FORMULA_PATHS))
    new_bill[formula_rows] = _statutory_bills(
        household, formula_rows, references.authority_band_d
    )
    gb_rows = np.flatnonzero(great_britain)
    new_rebate = rebate.copy()
    scaled_rebate = ratio[gb_rows] * rebate[gb_rows]
    new_rebate[gb_rows] = np.minimum(scaled_rebate, new_bill[gb_rows])
    capped = np.zeros(len(household), dtype=bool)
    capped[gb_rows] = scaled_rebate > new_bill[gb_rows]
    reduction = members.reduction
    new_reduction = reduction.copy()
    carriers = np.flatnonzero((reduction > 0.0) & great_britain[members.owner])
    owners = members.owner[carriers]
    scaled_reduction = ratio[owners] * reduction[carriers]
    new_reduction[carriers] = np.minimum(scaled_reduction, new_bill[owners])
    capped[owners[scaled_reduction > new_bill[owners]]] = True
    return UKSpineCouncilTax(
        council_tax=pd.Series(new_bill, index=household.index, name=_BILL_COLUMN),
        council_tax_rebate=pd.Series(
            new_rebate, index=household.index, name=_REBATE_COLUMN
        ),
        council_tax_benefit_reported=pd.Series(
            new_reduction, index=person.index, name=_REDUCTION_COLUMN
        ),
        council_tax_frs_bill=pd.Series(
            kept, index=household.index, name=COUNCIL_TAX_FRS_BILL_COLUMN
        ),
        paths=paths,
        support_channel=support_channel,
        capped=capped,
        references=references,
    )


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            np.ascontiguousarray(left, dtype=np.float64).view(np.uint64),
            np.ascontiguousarray(right, dtype=np.float64).view(np.uint64),
        )
    )


def _within_bills(
    household: pd.DataFrame, person: pd.DataFrame, rows: np.ndarray
) -> dict[str, bool]:
    """``amounts_within_bound`` over the households at ``rows`` and their
    persons: the rebate and the summed reduction against the bill."""

    kept = household.iloc[rows]
    members = person[_PERSON_HOUSEHOLD_COLUMN].isin(kept[_HOUSEHOLD_ID_COLUMN])
    return amounts_within_bound(
        {"household": kept, "person": person.loc[members.to_numpy(dtype=bool)]},
        entity="household",
        id_column=_HOUSEHOLD_ID_COLUMN,
        bound_column=_BILL_COLUMN,
        same_entity_columns=[_REBATE_COLUMN],
        member_columns={"person": [_REDUCTION_COLUMN]},
        membership_columns={"person": _PERSON_HOUSEHOLD_COLUMN},
        absolute_tolerance=_WITHIN_BILL_TOLERANCE,
    )


def _checks(
    before: tuple[pd.DataFrame, pd.DataFrame],
    after: tuple[pd.DataFrame, pd.DataFrame],
    *,
    great_britain: np.ndarray,
    owner: np.ndarray,
) -> dict[str, bool]:
    """The stage's checks on the tables before and after it ran, as booleans.

    ``owner`` is each person's household position, the same in both: the
    stage rewrites values, never rows. Northern Ireland's within-bill report
    reads the rows of geographic clone 0, as its count in the evidence does,
    so it is the same at every clone count; the other checks read every row.
    """

    (household, person), (localised, localised_person) = before, after
    gb_rows = np.flatnonzero(great_britain)
    ni_rows = np.flatnonzero(~great_britain)
    gb = _within_bills(localised, localised_person, gb_rows)
    ni = _within_bills(
        localised,
        localised_person,
        np.flatnonzero(~great_britain & _clone_zero(household)),
    )
    reduction = person[_REDUCTION_COLUMN].to_numpy(dtype=float)
    new_reduction = localised_person[_REDUCTION_COLUMN].to_numpy(dtype=float)
    summed = np.bincount(owner, weights=reduction, minlength=len(household))
    new_summed = np.bincount(owner, weights=new_reduction, minlength=len(household))
    rebate = household[_REBATE_COLUMN].to_numpy(dtype=float)
    new_rebate = localised[_REBATE_COLUMN].to_numpy(dtype=float)
    held = rebate == summed
    ni_people = ~great_britain[owner]
    return {
        "reduction_within_bill": gb[_REDUCTION_COLUMN] and gb[FINITE_NONNEGATIVE],
        "rebate_within_bill": gb[_REBATE_COLUMN] and gb[FINITE_NONNEGATIVE],
        "rebate_equals_reduction": bool(np.all(new_rebate[held] == new_summed[held])),
        "northern_ireland_unchanged": all(
            _bitwise_equal(
                household[column].to_numpy(dtype=float)[ni_rows],
                localised[column].to_numpy(dtype=float)[ni_rows],
            )
            for column in UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_REWRITES
        )
        and _bitwise_equal(reduction[ni_people], new_reduction[ni_people]),
        "northern_ireland_reductions_within_bill": (
            ni[_REBATE_COLUMN] and ni[_REDUCTION_COLUMN] and ni[FINITE_NONNEGATIVE]
        ),
    }


@dataclass(frozen=True)
class UKCouncilTaxLocaliseResult:
    """G6's frame and what its evidence reads of it.

    The source counts are exact distinct true source households: each
    FRS-channel path's over every clone, the support channel's paths' and the
    capped and Northern Ireland counts among clone-0 rows. The evidence gives
    them through ``receipt_counts``, and none prints in the repr.
    """

    frame: Frame = field(repr=False)
    clone_count: int
    band_d: UKCouncilTaxBandD = field(repr=False)
    support_sha256: Mapping[str, str]
    sources_by_path: Mapping[str, int] = field(repr=False)
    support_channel_sources_by_path: Mapping[str, int] = field(repr=False)
    capped_sources: Mapping[str, int] = field(repr=False)
    northern_ireland_reduction_above_bill_sources: int = field(repr=False)
    checks: Mapping[str, bool]

    def evidence(self) -> dict[str, object]:
        # A result exists only where the stage's first four checks held.
        return {
            "stage": COUNCIL_TAX_LOCALISE_STAGE_NAME,
            "method": COUNCIL_TAX_LOCALISE_METHOD,
            "clone_count": int(self.clone_count),
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
            "support_sha256": dict(self.support_sha256),
            "sources_by_path": {
                path: rounded_count(count)
                for path, count in self.sources_by_path.items()
            },
            "support_channel_sources_by_path": {
                path: rounded_count(count)
                for path, count in self.support_channel_sources_by_path.items()
            },
            "capped_sources": {
                kind: extreme_count(count)
                for kind, count in self.capped_sources.items()
            },
            "northern_ireland_reduction_above_bill_sources": extreme_count(
                self.northern_ireland_reduction_above_bill_sources
            ),
            "checks": dict(self.checks),
            "outputs": list(UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS),
            "rewrites": list(UK_COUNCIL_TAX_LOCALISE_REWRITES),
        }


def localise_uk_spine_council_tax(
    frame: Frame,
    *,
    definition: Mapping[str, object],
    supports: Mapping[str, AtomicSupport],
    band_d: UKCouncilTaxBandD,
) -> UKCouncilTaxLocaliseResult:
    """``frame`` with Great Britain's council tax localised and the paid bill
    kept (G6).

    :func:`localise_uk_council_tax` on the frame's households and persons,
    whose results replace ``council_tax``, ``council_tax_rebate`` and
    ``council_tax_benefit_reported`` and add ``council_tax_frs_bill``. Rows,
    ids, weights, every other column and the metadata are the frame's own;
    the mass log gains the stage's receipt. Then the in-stage assertion:
    Great Britain's reductions and rebates within their bills, finite and
    nonnegative, the rebate equal to the reduction wherever it was, and
    Northern Ireland unchanged.

    Refused, by name: a frame that is not a valid UK frame, whose period is
    not the Band D's build period, or that already carries
    ``council_tax_frs_bill``; every refusal of
    :func:`localise_uk_council_tax` and of the receipts' source reader; and
    a failed in-stage assertion, naming the checks.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"{_WHAT} rewrites the households of a Frame.")
    validate_uk_national_frame(frame)
    if int(uk_time_period(frame)) != UK_BAND_D_BUILD_PERIOD:
        raise ValueError(
            f"{_WHAT} prices bills at the {UK_BAND_D_FISCAL_YEAR} Band D, build "
            f"period {UK_BAND_D_BUILD_PERIOD}; the frame's period is another."
        )
    household = frame.table("household")
    person = frame.table("person")
    if COUNCIL_TAX_FRS_BILL_COLUMN in household.columns:
        raise ValueError(
            f"The frame already carries {COUNCIL_TAX_FRS_BILL_COLUMN!r}; {_WHAT} "
            "writes each household's once (microcosm#931)."
        )
    localised = localise_uk_council_tax(
        household, person, definition=definition, supports=supports, band_d=band_d
    )
    rewritten = household.copy()
    rewritten[_BILL_COLUMN] = localised.council_tax
    rewritten[_REBATE_COLUMN] = localised.council_tax_rebate
    rewritten[COUNCIL_TAX_FRS_BILL_COLUMN] = localised.council_tax_frs_bill
    rewritten_person = person.copy()
    rewritten_person[_REDUCTION_COLUMN] = localised.council_tax_benefit_reported
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = rewritten
    tables["person"] = rewritten_person
    # The household weights pass through unchanged. Annotated so the stage's
    # static walk meets microcosm.frame's Weights module.
    weights: Weights = frame.weights_for("household")
    result = Frame(
        tables,
        frame.schema,
        {"household": weights},
        frame.strata,
        mass_log=(
            *frame.mass_log,
            uk_household_mass_conservation_receipt(
                frame, UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON
            ),
        ),
        metadata=frame.metadata,
    )
    validate_uk_national_frame(result)
    great_britain = localised.references.great_britain
    members = _members(household, person)
    checks = _checks(
        (household, person),
        (rewritten, rewritten_person),
        great_britain=great_britain,
        owner=members.owner,
    )
    failed = [check for check in _REQUIRED_CHECKS if not checks[check]]
    if failed:
        raise ValueError(
            f"{_WHAT}'s in-stage assertion failed on {failed} (microcosm#931)."
        )
    sources = record_source_households(
        result, "household", stage=COUNCIL_TAX_LOCALISE_STAGE_NAME
    )
    paths = localised.paths
    frs = ~localised.support_channel
    clone_zero = _clone_zero(household)
    bill = household[_BILL_COLUMN].to_numpy(dtype=float)
    above = (household[_REBATE_COLUMN].to_numpy(dtype=float) > bill) | (
        members.household_reduction > bill
    )
    return UKCouncilTaxLocaliseResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        band_d=band_d,
        support_sha256=MappingProxyType(
            {system: supports[system].sha256 for system in sorted(supports)}
        ),
        sources_by_path=MappingProxyType(
            {
                path: distinct_count(sources, frs & (paths == path))
                for path in UK_COUNCIL_TAX_LOCALISE_FRS_PATHS
            }
        ),
        support_channel_sources_by_path=MappingProxyType(
            {
                path: distinct_count(sources, ~frs & clone_zero & (paths == path))
                for path in UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS
            }
        ),
        capped_sources=MappingProxyType(
            {
                kind: distinct_count(
                    sources,
                    clone_zero & localised.capped & np.isin(paths, kind_paths),
                )
                for kind, kind_paths in (
                    ("ratio", _RATIO_PATHS),
                    ("formula", _FORMULA_PATHS),
                )
            }
        ),
        northern_ireland_reduction_above_bill_sources=distinct_count(
            sources, clone_zero & ~great_britain & above
        ),
        checks=MappingProxyType(
            {check: bool(checks[check]) for check in UK_COUNCIL_TAX_LOCALISE_CHECKS}
        ),
    )


# --- The transform ------------------------------------------------------------------


@dataclass(frozen=True)
class UKCouncilTaxLocaliseStageTransform:
    """Whole-stage transform for the council-tax localisation (G6).

    ``supports`` holds the three conditioned supports' bytes, keyed by
    system, each held to canonical bytes, as G2's and G5's transforms take
    them. ``band_d`` is the Band D the stage prices at: the committed
    resources (``load_uk_council_tax_band_d``, which checks the Scottish
    resource's feed against the pin) unless a test supplies one.
    Constructing the transform checks the stage it runs as, builds G2's
    definition at seed 0 and decodes the supports; running it runs the
    Scottish resource's runtime arm with the frame's period, then
    :func:`localise_uk_spine_council_tax`. It calls no engine and draws
    nothing.
    """

    stage: SourceStageSpec
    supports: Mapping[str, bytes] = field(repr=False)
    band_d: UKCouncilTaxBandD | None = field(default=None, repr=False, compare=False)
    definition: Mapping[str, object] = field(init=False, repr=False, compare=False)
    decoded_supports: Mapping[str, AtomicSupport] = field(
        init=False, repr=False, compare=False
    )
    last_result: UKCouncilTaxLocaliseResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_council_tax_localise_stage(self.stage)
        if self.band_d is not None and not isinstance(self.band_d, UKCouncilTaxBandD):
            raise TypeError(f"{_WHAT} prices bills at a UKCouncilTaxBandD.")
        # Building the definition holds every support to canonical bytes.
        definition = uk_conditioned_assignment_definition(
            self.supports, seed=_LEVEL_SEED
        )
        decoded = uk_conditioned_supports(self.supports)
        band_d = (
            self.band_d if self.band_d is not None else load_uk_council_tax_band_d()
        )
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "decoded_supports", MappingProxyType(decoded))
        object.__setattr__(self, "band_d", band_d)

    def __call__(self, frame: Frame) -> Frame:
        _assert_council_tax_localise_stage(self.stage)
        check_scotland_band_d_declaration(
            _operation(self.stage, _VERIFY_VENDORED_FACT_RESOURCE_KIND),
            stage=COUNCIL_TAX_LOCALISE_STAGE_NAME,
            period=int(uk_time_period(frame)),
        )
        result = localise_uk_spine_council_tax(
            frame,
            definition=self.definition,
            supports=self.decoded_supports,
            band_d=self.band_d,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
