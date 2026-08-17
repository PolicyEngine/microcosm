"""FRS council-tax cell-mean imputation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import (
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
    """Impute raw missing council tax from raw region/band/adult cells."""

    raw = household_raw.set_index("household_id")
    aligned = raw.reindex(household["household_id"])
    ctannual = pd.to_numeric(aligned["ctannual"], errors="coerce")
    gvtregno = pd.to_numeric(aligned["gvtregno"], errors="coerce")
    ctband = pd.to_numeric(aligned["ctband"], errors="coerce")
    single_adult = pd.to_numeric(aligned["adulth"], errors="coerce") == 1
    scottish_water = np.where(
        gvtregno == SCOTLAND_GVTREGNO,
        (
            np.maximum(pd.to_numeric(aligned["csewamt"], errors="coerce").fillna(0), 0)
            + np.maximum(
                pd.to_numeric(aligned["cwatamtd"], errors="coerce").fillna(0), 0
            )
        )
        * WEEKS_IN_YEAR,
        0.0,
    )
    tax_only = pd.Series(np.maximum(ctannual - scottish_water, 0), index=aligned.index)
    donors = ctannual > 0
    cell_mean = tax_only[donors].groupby(
        [gvtregno[donors], ctband[donors], single_adult[donors]], dropna=False
    ).mean()
    keys = pd.MultiIndex.from_arrays([gvtregno, ctband, single_adult])
    imputed = pd.Series(keys.map(cell_mean).to_numpy(dtype=float), index=aligned.index)
    imputed = imputed.fillna(0.0).clip(lower=0)
    missing = (ctannual < 0) | ctannual.isna()
    result = pd.Series(np.where(missing, imputed, tax_only), index=aligned.index)
    return result.fillna(0.0).clip(lower=0)


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    if "househol" not in by_table:
        raise ValueError("frs_council_tax manifest is missing househol.tab.")
    return by_table
