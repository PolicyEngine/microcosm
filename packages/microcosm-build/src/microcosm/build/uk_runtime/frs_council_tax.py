"""FRS council-tax cell-mean imputation, with Scotland's bills levelled to its published liability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
    SCOTTISH_WATER_CHARGES_REDUCTION_RECIPIENT_SHARE,
    WEEKS_IN_YEAR,
    normalize_ids,
    read_pinned_tab,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import support_band
from microcosm.frame import Frame

FRS_COUNCIL_TAX_OUTPUT_COLUMNS = ("council_tax",)
SCOTLAND_GVTREGNO = 12
#: Raw household cells the derivation reads; the pinned tab carries them all.
FRS_COUNCIL_TAX_RAW_COLUMNS = (
    "ctannual",
    "ctpaid",
    "ctreb",
    "ctrebamt",
    "gvtregno",
    "ctband",
    "adulth",
    "cwatamt1",
    "csewamt1",
    "cwatamtd",
)
#: FRS 2024-25 question CTPaid, "Does anyone living here pay Council Tax?":
#: 1 yes, 2 no because the council pays (a rebate), 3 no because the household
#: does not have to. Northern Ireland, which pays domestic rates, is not asked.
FRS_CTPAID_PAYS = 1
#: Share of the gross Scottish water and sewerage charges that DWP's CTANNUAL
#: carries for a council tax reduction recipient. Netting 65% of the gross
#: charges brings Scottish recipients' bills to the non-recipient cells (ratio
#: 1.00 overall, 0.98-1.01 in bands A-F; uk-data#499), the Water Charges
#: Reduction Scheme's 35% maximum; non-recipients carry the gross charges in
#: full, status discount or not: netted that way, discounted bills sit at
#: 0.75 to 0.76 of undiscounted ones in the same band, the 25% discount.
SCOTTISH_CHARGES_IN_RECIPIENT_BILL = SCOTTISH_WATER_CHARGES_REDUCTION_RECIPIENT_SHARE
FRS_COUNCIL_TAX_STAGE_NAME = "frs_council_tax"
FRS_COUNCIL_TAX_IMPUTATION_KIND = "impute_cell_means"
FRS_COUNCIL_TAX_BUILD_PERIOD = 2024
VERIFY_VENDORED_FACT_RESOURCE_KIND = "verify_vendored_fact_resource"
#: S2(a) of the November event-data plan (microcosm#1095): the FRS's Scottish
#: bills sum to about 9% below the Scottish Government's own measure of the
#: 2024-25 liability, while the bound Scottish band stock already gives the
#: household mix, so the stage levels the bills' amounts, not the weights.
FRS_COUNCIL_TAX_LEVELLING_KIND = "level_scottish_council_tax"
SCOTTISH_BAND_D_LIABILITY_ROLE = "scotgov_band_d_liability_facts"
SCOTTISH_BAND_D_LIABILITY_RESOURCE = "scotgov_band_d_liability_facts.json"
#: CTAXBASE 2025's Band D equivalent dwellings before council tax reduction at
#: September 2024, the middle of 2024-25, for each council area.
SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET = "scotgov.ctaxbase2025.band_d_equivalents.y2024"
SCOTTISH_BAND_D_EQUIVALENTS_CONCEPT = "scotgov.council_tax.band_d_equivalents"
#: CTAS 2026's average Band D council tax for 2024-25, for each council area.
SCOTTISH_BAND_D_RATES_RECORD_SET = "scotgov.ctas2026.band_d.fy2024"
SCOTTISH_BAND_D_RATE_CONCEPT = "scotgov.council_tax.average_band_d"
SCOTTISH_COUNCIL_AREA_COUNT = 32


def frs_council_tax_operation_parameters() -> dict[str, Any]:
    """The ``impute_cell_means`` declaration ``derive_council_tax`` implements."""

    return {
        "kind": FRS_COUNCIL_TAX_IMPUTATION_KIND,
        "cells": ["gvtregno", "ctband", "adulth == 1"],
        "donor_filter": "raw ctreb != 1 and ctannual > 0",
        "missing": (
            "raw ctannual < 0 or NaN, or ctannual == 0 with ctpaid == 1 and ctreb != 1"
        ),
        "value": (
            "council tax before council tax reduction: CTANNUAL net of the "
            "Scottish water and sewerage charges ((CWATAMT1 + CSEWAMT1) x "
            "52, at 0.65 where CTREB is 1; CWATAMTD where CWATAMT1 is not "
            "positive), plus CTREBAMT x 52 where CTREB is 1 and CTREBAMT "
            "> 0; a CTREB 1 household without a positive CTREBAMT takes the "
            "larger of its netted CTANNUAL and its cell mean"
        ),
    }


def frs_council_tax_resource_operation_parameters() -> dict[str, Any]:
    """The ``verify_vendored_fact_resource`` declaration the Scottish level reads."""

    return {
        "kind": VERIFY_VENDORED_FACT_RESOURCE_KIND,
        "artifact_role": SCOTTISH_BAND_D_LIABILITY_ROLE,
        "resource": SCOTTISH_BAND_D_LIABILITY_RESOURCE,
        "feed_pin": "chronicle_feed.json",
        "record_sets": [
            SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET,
            SCOTTISH_BAND_D_RATES_RECORD_SET,
        ],
        "source_vintage": (
            "CTAXBASE 2025 Band D equivalents at September 2024; CTAS 2026 "
            "average Band D for 2024-25"
        ),
        "mapped_build_period": FRS_COUNCIL_TAX_BUILD_PERIOD,
        "period_mapping": "september_base_within_and_band_d_of_the_build_tax_year",
        "require_before_source_read": True,
        "runtime_sha256_required": True,
        "fail_on_mismatch": True,
    }


def frs_council_tax_levelling_operation_parameters() -> dict[str, Any]:
    """The ``level_scottish_council_tax`` declaration ``level_scottish_council_tax`` implements."""

    return {
        "kind": FRS_COUNCIL_TAX_LEVELLING_KIND,
        "output": "council_tax",
        "households": "raw gvtregno == 12 (Scotland)",
        "level": (
            "the sum over Scotland's 32 council areas of CTAXBASE 2025's Band D "
            "equivalent dwellings before council tax reduction at September 2024 "
            "times CTAS 2026's average Band D council tax for 2024-25"
        ),
        "stored": (
            "the Scottish households' council_tax after impute_cell_means, at the "
            "spine design weights"
        ),
        "factor": (
            "level / stored, one factor on every Scottish household's council_tax; "
            "a frame with no Scottish bill to level keeps its bills as derived"
        ),
        "weights": "spine_design_weights",
    }


def frs_council_tax_operations() -> tuple[dict[str, Any], ...]:
    """Every declaration after ``read_tables``, in the order the stage runs them."""

    return (
        frs_council_tax_operation_parameters(),
        frs_council_tax_resource_operation_parameters(),
        frs_council_tax_levelling_operation_parameters(),
    )


def frs_council_tax_resource_artifact() -> dict[str, Any]:
    """The stage's artifact for the vendored Scottish Band D facts."""

    return {
        "role": SCOTTISH_BAND_D_LIABILITY_ROLE,
        "kind": "public_aggregate_reference",
        "resource": SCOTTISH_BAND_D_LIABILITY_RESOURCE,
        "format": "json",
        "runtime_sha256_required": True,
    }


def assert_frs_council_tax_stage_parameters(stage: SourceStageSpec) -> None:
    """Closed-world drift assert: the manifest declares the implemented rule."""

    if stage.stage != FRS_COUNCIL_TAX_STAGE_NAME:
        raise ValueError(f"frs_council_tax transform received stage {stage.stage!r}.")
    operations = list(stage.operations)
    kinds = [operation.kind for operation in operations]
    expected_operations = frs_council_tax_operations()
    expected_kinds = ["read_tables", *(str(op["kind"]) for op in expected_operations)]
    if kinds != expected_kinds:
        raise ValueError(
            f"frs_council_tax stage must declare {expected_kinds}, got {kinds}."
        )
    for operation, expected in zip(operations[1:], expected_operations, strict=True):
        declared = {"kind": operation.kind, **dict(operation.parameters)}
        drifted = sorted(
            key
            for key in {*declared, *expected}
            if _plain(declared.get(key)) != _plain(expected.get(key))
        )
        if drifted:
            raise ValueError(
                f"frs_council_tax {operation.kind} declaration drifted from the "
                f"rule the runtime implements on parameter(s) {drifted} "
                "(uk-data#496/#499, microcosm#1095)."
            )
    expected_artifact = frs_council_tax_resource_artifact()
    resources = [
        {key: artifact.get(key) for key in expected_artifact}
        for artifact in stage.artifacts
        if artifact.get("role") == SCOTTISH_BAND_D_LIABILITY_ROLE
    ]
    if resources != [expected_artifact]:
        raise ValueError(
            "frs_council_tax must declare exactly one "
            f"{SCOTTISH_BAND_D_LIABILITY_ROLE} artifact, {expected_artifact}; "
            f"got {resources}."
        )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ScottishCouncilTaxLevel:
    """Scotland's 2024-25 council tax before council tax reduction, from its councils.

    ``total`` sums, over the 32 council areas, the area's Band D equivalent
    dwellings before council tax reduction times its average Band D council
    tax: the liability after discounts and exemptions that council tax
    reduction is then taken from, as the stage's bills are.
    """

    total: float
    receipt: Mapping[str, Any]


def scottish_council_tax_level_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> ScottishCouncilTaxLevel:
    """The level from the vendored rows; refused unless both tables cover the same 32 areas."""

    def by_area(record_set: str, concept: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in rows:
            layout = row.get("layout") or {}
            geography = row.get("geography") or {}
            if layout.get("record_set_id") != record_set:
                continue
            if geography.get("level") != "local_authority":
                continue
            if row.get("concept") != concept:
                raise ValueError(
                    f"{record_set} carries concept {row.get('concept')!r}, not {concept!r}."
                )
            area = str(geography.get("id"))
            if not area.startswith("S12"):
                raise ValueError(f"{record_set} carries a non-Scottish area {area!r}.")
            if area in values:
                raise ValueError(f"{record_set} carries {area} twice.")
            value = float(row["value"])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{record_set} gives {area} a non-positive value.")
            values[area] = value
        return values

    equivalents = by_area(
        SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET, SCOTTISH_BAND_D_EQUIVALENTS_CONCEPT
    )
    rates = by_area(SCOTTISH_BAND_D_RATES_RECORD_SET, SCOTTISH_BAND_D_RATE_CONCEPT)
    if len(equivalents) != SCOTTISH_COUNCIL_AREA_COUNT or set(equivalents) != set(
        rates
    ):
        raise ValueError(
            f"The Scottish level needs Band D equivalents and Band D rates for the "
            f"same {SCOTTISH_COUNCIL_AREA_COUNT} council areas; got "
            f"{len(equivalents)} and {len(rates)}, "
            f"{len(set(equivalents) ^ set(rates))} unmatched."
        )
    total = float(sum(equivalents[area] * rates[area] for area in sorted(equivalents)))
    return ScottishCouncilTaxLevel(
        total=total,
        receipt={
            "record_sets": [
                SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET,
                SCOTTISH_BAND_D_RATES_RECORD_SET,
            ],
            "council_areas": len(equivalents),
            "band_d_equivalents": float(sum(equivalents.values())),
            "level_gbp": total,
        },
    )


def load_scottish_council_tax_level() -> ScottishCouncilTaxLevel:
    """The level from the committed vendored resource, checked against the feed pin."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return scottish_council_tax_level_from_rows(
        vendored_rows(SCOTTISH_BAND_D_LIABILITY_RESOURCE)
    )


def level_scottish_council_tax(
    council_tax: np.ndarray,
    *,
    scottish: np.ndarray,
    weights: np.ndarray,
    level: ScottishCouncilTaxLevel,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale the Scottish bills by the one factor that takes their total to the level.

    The bills elsewhere and the weights are unchanged. A frame with no Scottish
    bill to level (a synthetic fixture's) keeps its bills, and the receipt says so.
    """

    bills = np.asarray(council_tax, dtype=float)
    scottish = np.asarray(scottish, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    stored = float(np.dot(weights[scottish], bills[scottish]))
    if not np.isfinite(stored) or stored < 0:
        raise ValueError("The Scottish council tax bills sum to no finite total.")
    receipt: dict[str, Any] = {
        **dict(level.receipt),
        "households": support_band(int(scottish.sum())),
        "stored_gbp": stored,
    }
    if stored == 0:
        return bills.copy(), {**receipt, "levelled": False, "factor": None}
    factor = level.total / stored
    levelled = bills.copy()
    levelled[scottish] = bills[scottish] * factor
    return levelled, {**receipt, "levelled": True, "factor": factor}


class UKFRSCouncilTaxStageTransform:
    """Whole-stage callable for FRS council-tax imputation."""

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        stage: SourceStageSpec,
        level: ScottishCouncilTaxLevel | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        #: The Scottish level; read from the vendored resource when not given.
        self.level = level
        self.last_receipt: dict[str, Any] | None = None

    def __call__(self, frame: Frame) -> Frame:
        result, receipt = frs_council_tax_with_receipt(
            frame, self.raw_dir, stage=self.stage, level=self.level
        )
        self.last_receipt = receipt
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_COUNCIL_TAX_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            return {}
        return {
            "evidence": {
                "stage": FRS_COUNCIL_TAX_STAGE_NAME,
                "scottish_level": dict(self.last_receipt),
            }
        }


def add_frs_council_tax(
    frame: Frame,
    raw_dir: str | Path,
    *,
    stage: SourceStageSpec,
    level: ScottishCouncilTaxLevel | None = None,
) -> Frame:
    return frs_council_tax_with_receipt(frame, raw_dir, stage=stage, level=level)[0]


def frs_council_tax_with_receipt(
    frame: Frame,
    raw_dir: str | Path,
    *,
    stage: SourceStageSpec,
    level: ScottishCouncilTaxLevel | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """The stage's frame and its Scottish level receipt."""

    assert_frs_council_tax_stage_parameters(stage)
    # The vendored facts are checked against the feed pin before the raw tab is
    # read (the declaration's require_before_source_read).
    level = level if level is not None else load_scottish_council_tax_level()
    artifacts = _artifact_by_table(stage)
    household_raw = normalize_ids(
        read_pinned_tab(
            Path(raw_dir) / str(artifacts["househol"]["locator"]),
            artifacts["househol"],
        )
    )
    household = frame.table("household").copy()
    derived = derive_council_tax(household, household_raw)
    bills = derived.reindex(household["household_id"]).to_numpy(dtype=float)
    gvtregno = pd.to_numeric(
        household_raw.set_index("household_id")["gvtregno"].reindex(
            household["household_id"]
        ),
        errors="coerce",
    )
    levelled, receipt = level_scottish_council_tax(
        bills,
        scottish=(gvtregno == SCOTLAND_GVTREGNO).to_numpy(),
        weights=frame.weights_for("household").values,
        level=level,
    )
    household["council_tax"] = levelled
    result = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result, receipt


def derive_council_tax(
    household: pd.DataFrame, household_raw: pd.DataFrame
) -> pd.Series:
    """The council tax bill after discounts and before council tax reduction.

    FRS 2024-25 ``CTANNUAL`` is the "Annual CT amount after discounts/reduction"
    (DV summary, UKDS SN 9563), derived with the reported reduction ``CTREB`` /
    ``CTREBAMT``; the engine reads ``council_tax`` as the liability before
    council tax reduction. A household reporting a reduction (``CTREB`` 1)
    therefore gets its weekly ``CTREBAMT`` added back, a full reduction
    (``CTANNUAL`` 0) included: on the raw tab that restores the recipients'
    bills to the non-recipients' in the same region, band and single-adult
    cell (ratio 0.99 in England, 1.01 in Wales; uk-data#496/#499,
    microcosm#1095). A recipient with no positive amount has an unknown
    reduction and takes the larger of its bill and its cell's mean. Missing
    or negative bills take the cell mean. So does a zero bill where someone
    pays council tax (``CTPAID`` 1) and no reduction is reported, since such
    a household owes a bill (microcosm#1278). From April 2024 the band and
    "some other Council Tax block questions" left the FRS field interview:
    editing teams input the band from Valuation Office Agency and Scottish
    Assessors data, and the other answers "are then imputed during DWP's
    validation processes" (FRS 2024-25 background information and
    methodology, section 1.3). The methodology does not say whether
    ``CTPAID`` itself is still asked or is imputed. A
    non-recipient's zero bill otherwise stays zero: the household does not
    have to pay (``CTPAID`` 3), or says the council pays (``CTPAID`` 2)
    without reporting a reduction. The cells pool non-recipients with a
    positive bill only, since recipients' ``CTANNUAL`` is net of their
    reduction.

    Scottish bills carry the water and sewerage charges, which council tax
    reduction does not cover. They are netted at the gross cells
    ``CWATAMT1`` + ``CSEWAMT1``, in full for non-recipients and at
    ``SCOTTISH_CHARGES_IN_RECIPIENT_BILL`` for recipients, the shares DWP's
    derivation embeds (uk-data#499); a household with no gross water cell
    nets its recorded ``CWATAMTD``. The charge a household pays is a separate
    quantity, ``frs_spine.scottish_water_and_sewerage_weekly``.
    """

    missing_columns = sorted(set(FRS_COUNCIL_TAX_RAW_COLUMNS) - set(household_raw))
    if missing_columns:
        raise KeyError(f"FRS council tax needs raw household cells {missing_columns}.")
    raw = household_raw.set_index("household_id")
    aligned = raw.reindex(household["household_id"])

    def number(column: str) -> pd.Series:
        return pd.to_numeric(aligned[column], errors="coerce")

    def positive(column: str) -> pd.Series:
        return number(column).fillna(0.0).clip(lower=0.0)

    ctannual = number("ctannual")
    gvtregno = number("gvtregno")
    ctband = number("ctband")
    single_adult = number("adulth") == 1
    recipient = number("ctreb") == 1
    reduction = positive("ctrebamt") * WEEKS_IN_YEAR
    known_reduction = recipient & (reduction > 0)

    gross_charges = positive("cwatamt1") + positive("csewamt1")
    netted_weekly = np.where(
        positive("cwatamt1") > 0,
        gross_charges * np.where(recipient, SCOTTISH_CHARGES_IN_RECIPIENT_BILL, 1.0),
        positive("cwatamtd"),
    )
    netted = np.where(gvtregno == SCOTLAND_GVTREGNO, netted_weekly * WEEKS_IN_YEAR, 0.0)
    tax_only = pd.Series(np.maximum(ctannual - netted, 0), index=aligned.index)

    donors = ~recipient & (ctannual > 0)
    cell_mean = (
        tax_only[donors]
        .groupby([gvtregno[donors], ctband[donors], single_adult[donors]], dropna=False)
        .mean()
    )
    keys = pd.MultiIndex.from_arrays([gvtregno, ctband, single_adult])
    imputed = pd.Series(keys.map(cell_mean).to_numpy(dtype=float), index=aligned.index)
    imputed = imputed.fillna(0.0).clip(lower=0)

    pays = number("ctpaid") == FRS_CTPAID_PAYS
    paying_zero = (ctannual == 0) & pays & ~recipient
    missing = (ctannual < 0) | ctannual.isna() | paying_zero
    gross = np.where(
        missing,
        imputed,
        np.where(
            known_reduction,
            np.maximum(ctannual - netted + reduction, 0),
            np.where(recipient, np.maximum(tax_only, imputed), tax_only),
        ),
    )
    return pd.Series(gross, index=aligned.index).fillna(0.0).clip(lower=0)


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "househol" not in by_table:
        raise ValueError("frs_council_tax manifest is missing househol.tab.")
    return by_table
