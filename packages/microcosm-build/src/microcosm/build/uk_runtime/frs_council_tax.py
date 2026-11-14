"""FRS council-tax cell-mean imputation."""

from __future__ import annotations

from collections.abc import Mapping
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
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    record_source_households,
    rounded_count,
)
from microcosm.frame import Frame

SCOTLAND_GVTREGNO = 12
NORTHERN_IRELAND_GVTREGNO = 13
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
#: The household column naming the branch each stage-5 bill takes, or
#: ``not_levied`` in Northern Ireland (``derive_council_tax_provenance``):
#: one of the four values below.
FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN = "council_tax_bill_provenance"
#: The bill is the observed CTANNUAL, netted of Scotland's water and sewerage
#: charges, with any reported reduction added back: an observed gross bill. A
#: household without a reported reduction whose CTANNUAL is zero and whose
#: CTPAID is not 1 (2, 3 or blank) keeps its zero bill under this value too.
FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS = "observed_gross"
#: CTANNUAL is negative or blank, or zero where someone pays and no reduction
#: is reported: the bill is the cell mean.
FRS_COUNCIL_TAX_BILL_MISSING = "missing"
#: A reduction is reported (CTREB 1) without a positive CTREBAMT: the bill is
#: the larger of the netted CTANNUAL and the cell mean.
FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION = "unknown_reduction"
#: Northern Ireland, which pays domestic rates, not council tax.
FRS_COUNCIL_TAX_BILL_NOT_LEVIED = "not_levied"
FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES = (
    FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    FRS_COUNCIL_TAX_BILL_MISSING,
    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
    FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
)
#: The graph's string dtype, which the provenance column is stored in.
_PROVENANCE_DTYPE = pd.StringDtype(storage="python")
#: The household columns the stage writes: the bill, then its provenance,
#: which the release boundary drops and council_tax_localise alone reads.
FRS_COUNCIL_TAX_OUTPUT_COLUMNS = ("council_tax", FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN)


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


def frs_council_tax_provenance_operation_parameters() -> dict[str, Any]:
    """The ``derive`` declaration ``derive_council_tax_provenance`` implements."""

    return {
        "kind": "derive",
        "output": FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
        "formula": (
            "first match: not_levied where raw gvtregno == 13 (Northern "
            "Ireland, which pays domestic rates), whatever branch its bill "
            "takes; else the branch derive_council_tax takes: missing where "
            "the bill is missing (raw ctannual < 0 or NaN, or ctannual == 0 "
            "with ctpaid == 1 and ctreb != 1); observed_gross where ctreb == 1 "
            "and ctrebamt > 0 (the reported reduction added back); "
            "unknown_reduction where ctreb == 1 (the larger of the netted "
            "ctannual and the cell mean); else observed_gross (the netted "
            "ctannual, kept at zero where ctannual == 0 and ctpaid != 1, as "
            "with ctpaid 2 and 3)"
        ),
    }


def assert_frs_council_tax_stage_parameters(stage: SourceStageSpec) -> None:
    """Closed-world drift assert: the manifest declares the implemented rules.

    The stage reads its table, imputes the bill
    (:func:`frs_council_tax_operation_parameters`) and derives the bill's
    provenance (:func:`frs_council_tax_provenance_operation_parameters`), in
    that order, each declaration exactly as those functions state it.
    """

    if stage.stage != FRS_COUNCIL_TAX_STAGE_NAME:
        raise ValueError(f"frs_council_tax transform received stage {stage.stage!r}.")
    operations = list(stage.operations)
    expected_operations = (
        frs_council_tax_operation_parameters(),
        frs_council_tax_provenance_operation_parameters(),
    )
    kinds = [operation.kind for operation in operations]
    expected_kinds = ["read_tables", *(op["kind"] for op in expected_operations)]
    if kinds != expected_kinds:
        raise ValueError(
            f"frs_council_tax stage must declare {', '.join(expected_kinds)}, in "
            f"that order, got {kinds}."
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
                "(uk-data#496/#499, microcosm#1095, microcosm#931)."
            )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class UKFRSCouncilTaxStageTransform:
    """Whole-stage callable for FRS council-tax imputation.

    Running it writes each household's bill and the bill's provenance
    (:func:`add_frs_council_tax`), then keeps the stage's receipt of the
    provenance (:func:`council_tax_provenance_evidence`), counted on the frame
    it wrote, for :meth:`checkpoint_metadata`.
    """

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        self.last_evidence: dict[str, Any] | None = None

    def __call__(self, frame: Frame) -> Frame:
        result = add_frs_council_tax(frame, self.raw_dir, stage=self.stage)
        self.last_evidence = council_tax_provenance_evidence(
            result,
            result.table("household")[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN],
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_COUNCIL_TAX_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_evidence is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_evidence}


def add_frs_council_tax(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> Frame:
    """Each household's bill and the branch the bill takes, on the frame.

    ``council_tax`` is :func:`derive_council_tax`'s bill and
    ``FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN`` is
    :func:`derive_council_tax_provenance`'s flag, both from one set of masks;
    the flag is in the graph's string dtype (``StringDtype("python")``).
    """

    assert_frs_council_tax_stage_parameters(stage)
    artifacts = _artifact_by_table(stage)
    household_raw = normalize_ids(
        read_pinned_tab(
            Path(raw_dir) / str(artifacts["househol"]["locator"]),
            artifacts["househol"],
        )
    )
    household = frame.table("household").copy()
    bill, provenance = _council_tax_branches(household, household_raw)
    household_ids = household["household_id"]
    household["council_tax"] = bill.reindex(household_ids).to_numpy()
    household[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN] = pd.Series(
        provenance.reindex(household_ids).to_numpy(dtype=object),
        index=household.index,
        dtype=_PROVENANCE_DTYPE,
    )
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
    return result


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

    return _council_tax_branches(household, household_raw)[0]


def _council_tax_branches(
    household: pd.DataFrame, household_raw: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """``derive_council_tax``'s bill, and the branch each bill takes.

    One set of masks chooses both, so the provenance names the branch the
    bill took on every row outside Northern Ireland, where it reads
    ``not_levied`` whatever the branch.
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
    # The bill's own branches in the bill's order, after Northern Ireland: the
    # first condition that holds names the row.
    provenance = np.select(
        [
            (gvtregno == NORTHERN_IRELAND_GVTREGNO).to_numpy(dtype=bool),
            missing.to_numpy(dtype=bool),
            known_reduction.to_numpy(dtype=bool),
            recipient.to_numpy(dtype=bool),
        ],
        [
            FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
            FRS_COUNCIL_TAX_BILL_MISSING,
            FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
            FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
        ],
        default=FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    )
    return (
        pd.Series(gross, index=aligned.index).fillna(0.0).clip(lower=0),
        pd.Series(provenance, index=aligned.index, dtype=_PROVENANCE_DTYPE),
    )


def derive_council_tax_provenance(
    household: pd.DataFrame, household_raw: pd.DataFrame
) -> pd.Series:
    """The branch each household's ``derive_council_tax`` bill takes.

    One of ``FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES`` per household, on the
    bill's index, in the graph's string dtype (``StringDtype("python")``).
    The first that holds: ``not_levied`` in Northern Ireland (raw GVTREGNO
    ``NORTHERN_IRELAND_GVTREGNO``); ``missing`` where the bill takes the cell
    mean as a missing one; ``observed_gross`` where a reported reduction is
    added back; ``unknown_reduction`` for any other household reporting a
    reduction; else ``observed_gross``. The masks are the bill's own, so
    outside Northern Ireland the value is the branch the bill took.
    """

    return _council_tax_branches(household, household_raw)[1]


def council_tax_provenance_evidence(
    frame: Frame | Mapping[str, pd.DataFrame], provenance: Any
) -> dict[str, Any]:
    """Stage 5's receipt of its bill provenance, by distinct true sources.

    ``provenance`` holds one of ``FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES`` for
    each household row of ``frame`` (a frame, or its tables by entity), in
    row order. Each value's carriers are counted in distinct true source
    households through ``receipt_counts.record_source_households`` (before
    the first copying step each household is its own source; a copy or clone
    counts as its source), UK-wide, and recorded as
    ``receipt_counts.rounded_count`` gives them, every value present, zero
    included. Refusals name the column (and the four values), never a value
    read or a count.
    """

    sources = record_source_households(
        frame, "household", stage=FRS_COUNCIL_TAX_STAGE_NAME, before_copies=True
    )
    values = np.asarray(provenance, dtype=object)
    if values.ndim != 1 or len(values) != len(sources):
        raise ValueError(
            f"{FRS_COUNCIL_TAX_STAGE_NAME}'s bill provenance receipt needs one "
            f"{FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN} value per household row."
        )
    flags = pd.Series(values)
    if not flags.isin(FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES).all():
        raise ValueError(
            f"{FRS_COUNCIL_TAX_STAGE_NAME}'s "
            f"{FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN} holds a value outside "
            f"{list(FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES)}."
        )
    return {
        "stage": FRS_COUNCIL_TAX_STAGE_NAME,
        "bill_provenance_sources": {
            value: rounded_count(
                distinct_count(sources, (flags == value).to_numpy(dtype=bool))
            )
            for value in FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES
        },
    }


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "househol" not in by_table:
        raise ValueError("frs_council_tax manifest is missing househol.tab.")
    return by_table
