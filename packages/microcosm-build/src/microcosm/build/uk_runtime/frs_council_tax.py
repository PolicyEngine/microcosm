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
from microcosm.frame import Frame

FRS_COUNCIL_TAX_OUTPUT_COLUMNS = ("council_tax",)
SCOTLAND_GVTREGNO = 12
#: Raw household cells the derivation reads; the pinned tab carries them all.
FRS_COUNCIL_TAX_RAW_COLUMNS = (
    "ctannual",
    "ctreb",
    "ctrebamt",
    "gvtregno",
    "ctband",
    "adulth",
    "cwatamt1",
    "csewamt1",
    "cwatamtd",
)
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


def frs_council_tax_operation_parameters() -> dict[str, Any]:
    """The ``impute_cell_means`` declaration ``derive_council_tax`` implements."""

    return {
        "kind": FRS_COUNCIL_TAX_IMPUTATION_KIND,
        "cells": ["gvtregno", "ctband", "adulth == 1"],
        "donor_filter": "raw ctreb != 1 and ctannual > 0",
        "missing": "raw ctannual < 0 or NaN",
        "value": (
            "council tax before council tax reduction: CTANNUAL net of the "
            "Scottish water and sewerage charges ((CWATAMT1 + CSEWAMT1) x "
            "365.25/7, at 0.65 where CTREB is 1; CWATAMTD where CWATAMT1 is not "
            "positive), plus CTREBAMT x 365.25/7 where CTREB is 1 and CTREBAMT "
            "> 0; a CTREB 1 household without a positive CTREBAMT takes the "
            "larger of its netted CTANNUAL and its cell mean"
        ),
    }


def assert_frs_council_tax_stage_parameters(stage: SourceStageSpec) -> None:
    """Closed-world drift assert: the manifest declares the implemented rule."""

    if stage.stage != FRS_COUNCIL_TAX_STAGE_NAME:
        raise ValueError(f"frs_council_tax transform received stage {stage.stage!r}.")
    operations = list(stage.operations)
    kinds = [operation.kind for operation in operations]
    if kinds != ["read_tables", FRS_COUNCIL_TAX_IMPUTATION_KIND]:
        raise ValueError(
            "frs_council_tax stage must declare read_tables followed by "
            f"{FRS_COUNCIL_TAX_IMPUTATION_KIND}, got {kinds}."
        )
    declared = {"kind": operations[1].kind, **dict(operations[1].parameters)}
    expected = frs_council_tax_operation_parameters()
    drifted = sorted(
        key
        for key in {*declared, *expected}
        if _plain(declared.get(key)) != _plain(expected.get(key))
    )
    if drifted:
        raise ValueError(
            "frs_council_tax impute_cell_means declaration drifted from the "
            f"rule the runtime implements on parameter(s) {drifted} "
            "(uk-data#496/#499, microcosm#1095)."
        )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class UKFRSCouncilTaxStageTransform:
    """Whole-stage callable for FRS council-tax imputation."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_council_tax(frame, self.raw_dir, stage=self.stage)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_COUNCIL_TAX_OUTPUT_COLUMNS


def add_frs_council_tax(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> Frame:
    assert_frs_council_tax_stage_parameters(stage)
    artifacts = _artifact_by_table(stage)
    household_raw = normalize_ids(
        read_pinned_tab(
            Path(raw_dir) / str(artifacts["househol"]["locator"]),
            artifacts["househol"],
        )
    )
    household = frame.table("household").copy()
    derived = derive_council_tax(household, household_raw)
    household["council_tax"] = derived.reindex(household["household_id"]).to_numpy()
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
    or negative bills take the cell mean. The cells pool non-recipients with
    a positive bill only, since recipients' ``CTANNUAL`` is net of their
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
    cell_mean = tax_only[donors].groupby(
        [gvtregno[donors], ctband[donors], single_adult[donors]], dropna=False
    ).mean()
    keys = pd.MultiIndex.from_arrays([gvtregno, ctband, single_adult])
    imputed = pd.Series(keys.map(cell_mean).to_numpy(dtype=float), index=aligned.index)
    imputed = imputed.fillna(0.0).clip(lower=0)

    missing = (ctannual < 0) | ctannual.isna()
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
