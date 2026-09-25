"""Contracts for the retired eCPS Section 199A input family."""

# ruff: noqa: F401

from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime import qbi_inputs as qbi_inputs_module
from microcosm.build.us_runtime.qbi_inputs import (
    QBI_ARCHIVED_ASSUMPTIONS_URL,
    QBI_ARCHIVED_CLONE_URL,
    QBI_ARCHIVED_DERIVATION_URL,
    QBI_ARCHIVED_EXPORT_URL,
    QBI_ARCHIVED_IMPUTATION_URL,
    QBI_ARCHIVED_PUF_ARTIFACT_URL,
    QBI_ARCHIVED_SIMULATION_URL,
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    us_qbi_inputs_signal_gate,
    us_qbi_inputs_stage_spec,
    us_qbi_inputs_summary,
    us_qbi_reconciliation_change_receipt,
    us_qbi_reconciliation_universe_receipt,
    validate_us_qbi_reconciliation_live_output,
    validate_us_qbi_reconciliation_receipt,
    with_us_qbi_input_reconciliation,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.schema import EntitySchema


def _frame(person: pd.DataFrame) -> Frame:
    person = person.copy(deep=True).reset_index(drop=True)
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person.insert(0, "person_id", ids)
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _qbi_person(n: int = 200) -> pd.DataFrame:
    person = pd.DataFrame(
        {
            column: np.zeros(n, dtype=bool)
            if column in US_QBI_BOOLEAN_OUTPUT_COLUMNS
            else np.zeros(n, dtype=np.float64)
            for column in US_QBI_OUTPUT_COLUMNS
        }
    )
    person["person_support_channel"] = np.where(
        np.arange(n) < n // 2, "asec", "puf_tax_detail"
    )
    person["self_employment_income_before_lsr"] = 0.0
    person["partnership_income"] = 0.0
    person["s_corp_income"] = 0.0
    person["estate_income"] = 0.0
    person["rental_income"] = 0.0
    person["non_qualified_dividend_income"] = 0.0

    puf = np.arange(n // 2, n)
    for index, column in enumerate(US_QBI_BOOLEAN_OUTPUT_COLUMNS):
        if column == "business_is_sstb":
            continue
        person.loc[puf[index % 5 :: 5], column] = False
        person.loc[puf, column] |= np.arange(len(puf)) % 5 != index % 5

    sstb = puf[:10]
    person.loc[sstb, "business_is_sstb"] = True
    person.loc[sstb, "self_employment_income_would_be_qualified"] = True
    person.loc[sstb, "self_employment_income_before_lsr"] = 10_000.0
    person.loc[sstb, "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[sstb, "unadjusted_basis_qualified_property"] = 5_000.0

    person.loc[puf[:2], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:2], "qualified_bdc_income"] = 20.0
    person.loc[puf[:20], "non_qualified_dividend_income"] = 1_000.0
    person.loc[puf[:20], "qualified_reit_and_ptp_income"] = 40.0
    person.loc[puf[:25], "w2_wages_from_qualified_business"] = 2_000.0
    person.loc[puf[:30], "unadjusted_basis_qualified_property"] = 5_000.0
    return person


def _stacked_qbi_universe_frame(*, child_age: float = 12.0) -> Frame:
    person = _qbi_person(20)
    person["person_support_channel"] = np.where(
        np.arange(len(person)) < 10,
        "asec",
        "acs",
    )
    person["person_support_clone_index"] = 0
    person["person_spine_source_id"] = np.arange(len(person), dtype=np.int64)
    # Spine assembly writes the assembly-unique source ID beside the raw one.
    person["person_source_id"] = person["person_spine_source_id"]
    person["age"] = 40.0
    person["SEMP"] = person["self_employment_income_before_lsr"]
    child = 10
    person.loc[child, "age"] = child_age
    person.loc[child, "self_employment_income_before_lsr"] = 0.0
    person.loc[child, "SEMP"] = np.nan
    person.loc[child, "business_is_sstb"] = True
    person.loc[child, "qualified_bdc_income"] = 321.0
    return _frame(person)


def _authorized_qbi_output(frame: Frame) -> tuple[Frame, dict[str, object]]:
    reconciled = with_us_qbi_input_reconciliation(frame)
    receipt = us_qbi_reconciliation_change_receipt(frame, reconciled)
    authorized = qbi_inputs_module.bind_us_qbi_reconciliation_transition_authority(
        reconciled,
        receipt,
    )
    return authorized, receipt


def _rehash_forged_stacked_universe_receipt(
    receipt: dict[str, object],
) -> None:
    universe = receipt["recipient_source_universe"]
    assert isinstance(universe, dict)
    source_receipt = {
        key: value
        for key, value in universe.items()
        if key
        not in {
            "source_universe_sha256",
            "source_universe_resolution_mutated_raw_pums_cells",
            "operation",
            "rows_excluded_from_base_self_employment_rewrite",
            "rows_included_in_other_qbi_reconciliation",
            "structurally_absent_base_source_cells_mutated",
            "sha256",
        }
    }
    source_receipt["raw_pums_source_cells_mutated"] = universe[
        "source_universe_resolution_mutated_raw_pums_cells"
    ]
    universe["source_universe_sha256"] = qbi_inputs_module._qbi_receipt_sha256(
        source_receipt
    )
    unsigned_universe = dict(universe)
    unsigned_universe.pop("sha256")
    universe["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned_universe)
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("sha256")
    receipt["sha256"] = qbi_inputs_module._qbi_receipt_sha256(unsigned_receipt)


__all__ = [name for name in globals() if not name.startswith("__")]
