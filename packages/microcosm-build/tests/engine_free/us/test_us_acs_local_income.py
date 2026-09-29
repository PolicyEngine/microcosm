"""The separate ASEC-channel income transfer for the ACS local lane
(microcosm#1022)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_local_income
from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_INCOME_DONOR_CHANNEL,
    ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS,
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
    ACS_LOCAL_INCOME_TRANSFER_FAMILIES,
    ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
    ACS_LOCAL_INCOME_TRANSFER_ISSUE,
    acs_local_income_transfer_signal_gate,
    acs_local_income_transfer_target_families,
    record_acs_local_income_transfer,
    require_acs_local_income_donor,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    declared_acs_transfer_target_families,
    required_acs_transfer_inputs,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")


def _pooled(
    *,
    spines: tuple[str, ...] = (
        ASEC_PUF_DONOR_SPINE,
        ASEC_PUF_DONOR_SPINE,
        ACS_2024_1YR_SPINE,
        ACS_2024_1YR_SPINE,
    ),
    amounts: dict[str, list[float]] | None = None,
) -> Frame:
    """Two donor and two ACS persons, one household each."""

    n = len(spines)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "person_household_id": np.arange(1, n + 1),
            "person_tax_unit_id": np.arange(1, n + 1),
            "person_spm_unit_id": np.arange(1, n + 1),
            "person_family_id": np.arange(1, n + 1),
            "person_marital_unit_id": np.arange(1, n + 1),
            TAG: list(spines),
        }
    )
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        person[column] = [0.0, 1_200.0, 0.0, 900.0][:n]
    for column, values in (amounts or {}).items():
        person[column] = values
    ids = np.arange(1, n + 1)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
    )


def _receipt(**overrides: Any) -> dict[str, Any]:
    receipt = record_acs_local_income_transfer(
        {"channel": ACS_LOCAL_INCOME_DONOR_CHANNEL},
        [
            {
                "column": column,
                "donor_channel": ACS_LOCAL_INCOME_DONOR_CHANNEL,
                "imputed_recipient_rows": 2,
                "unmodeled_recipient_rows": 0,
                "predictors": ["age", "state_fips"],
            }
            for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS
        ],
        acs_persons=2,
    )
    receipt.update(overrides)
    return receipt


def test_target_families_are_person_only_and_disjoint_from_the_shared_plan():
    families = acs_local_income_transfer_target_families()
    assert set(families) == {"person"}
    assert families["person"] == dict(ACS_LOCAL_INCOME_TRANSFER_FAMILIES)
    columns = {c for targets in families["person"].values() for c in targets}
    assert columns == set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)
    assert not columns & required_acs_transfer_inputs()
    assert not columns & set(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)


def test_the_shared_declared_plan_does_not_carry_the_local_income_columns():
    shared = {
        column
        for families in declared_acs_transfer_target_families().values()
        for targets in families.values()
        for column in targets
    }
    assert not shared & set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)


def test_target_families_refuse_overlap_with_the_shared_plan(monkeypatch):
    monkeypatch.setattr(
        acs_local_income,
        "required_acs_transfer_inputs",
        lambda: frozenset({"workers_compensation"}),
    )
    with pytest.raises(ValueError, match="double counting"):
        acs_local_income_transfer_target_families()


def test_receipt_counts_each_column_from_the_income_pass_provenance():
    receipt = _receipt()
    assert receipt["issue"] == ACS_LOCAL_INCOME_TRANSFER_ISSUE
    assert receipt["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert receipt["acs_persons"] == 2
    assert receipt["not_transferred_shared_plan_components"] == list(
        ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS
    )
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        entry = receipt["columns"][column]
        assert entry["imputed_rows"] == 2
        assert entry["unmodeled_rows"] == 0
        assert entry["donor_channel"] == [ACS_LOCAL_INCOME_DONOR_CHANNEL]
        assert entry["predictors"] == ["age", "state_fips"]


def test_donor_without_an_asec_role_is_refused():
    with pytest.raises(ValueError, match="ASEC observation role"):
        require_acs_local_income_donor(_pooled())


def test_donor_summary_requires_complete_non_negative_amounts(monkeypatch):
    donor = _pooled(spines=(ASEC_PUF_DONOR_SPINE,) * 4)
    monkeypatch.setattr(
        acs_local_income,
        "resolve_acs_donor_channel",
        lambda frame, channel: (frame, channel),
    )
    summary = require_acs_local_income_donor(donor)
    assert summary["channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert summary["person_rows"] == 4
    assert summary["columns"]["workers_compensation"]["positive_rows"] == 2
    negative = _pooled(
        spines=(ASEC_PUF_DONOR_SPINE,) * 4,
        amounts={"child_support_received": [0.0, -5.0, 0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="negative"):
        require_acs_local_income_donor(negative)
    absent = _pooled(spines=(ASEC_PUF_DONOR_SPINE,) * 4)
    absent = Frame(
        {
            **{e: absent.table(e) for e in absent.entities if e != "person"},
            "person": absent.table("person").drop(columns=["keogh_distributions"]),
        },
        absent.schema,
        {"household": absent.weights_for("household")},
    )
    with pytest.raises(ValueError, match="lacks"):
        require_acs_local_income_donor(absent)


def test_gate_passes_on_a_complete_transfer_with_a_clean_receipt():
    gate = acs_local_income_transfer_signal_gate(_pooled(), receipt=_receipt())
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_INCOME_TRANSFER_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]
    assert acs["rows"] == 2


@pytest.mark.parametrize(
    ("amounts", "receipt_overrides", "expected"),
    [
        (
            {"child_support_received": [0.0, 10.0, np.nan, 5.0]},
            {},
            "missing",
        ),
        ({"workers_compensation": [0.0, 10.0, -1.0, 5.0]}, {}, "negative"),
        (
            {"disability_benefits": [0.0, 10.0, 0.0, 0.0]},
            {},
            "all zero although the donor",
        ),
        ({}, {"donor_channel": "puf"}, "donor channel"),
        ({}, {"acs_persons": 5}, "ACS person"),
    ],
    ids=["missing-acs-cell", "negative", "acs-all-zero", "wrong-channel", "count"],
)
def test_gate_fails(amounts, receipt_overrides, expected):
    gate = acs_local_income_transfer_signal_gate(
        _pooled(amounts=amounts), receipt=_receipt(**receipt_overrides)
    )
    assert not gate.passed
    assert expected in " ".join(gate.failures)


def test_gate_fails_without_a_receipt_or_with_unmodeled_rows():
    missing = acs_local_income_transfer_signal_gate(_pooled(), receipt=None)
    assert not missing.passed
    assert "No acs_local_income_transfer staging receipt" in " ".join(missing.failures)
    receipt = _receipt()
    receipt["columns"]["keogh_distributions"]["unmodeled_rows"] = 1
    unmodeled = acs_local_income_transfer_signal_gate(_pooled(), receipt=receipt)
    assert not unmodeled.passed
    assert "keogh_distributions left 1" in " ".join(unmodeled.failures)


def test_gate_fails_on_missing_tags_or_an_unknown_spine():
    frame = _pooled()
    person = frame.table("person").copy()
    person[TAG] = person[TAG].where(person.index > 0)
    untagged = Frame(
        {
            **{e: frame.table(e) for e in frame.entities if e != "person"},
            "person": person,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    assert "Missing person origin tags" in " ".join(
        acs_local_income_transfer_signal_gate(untagged, receipt=_receipt()).failures
    )
    unknown = _pooled(
        spines=(
            ASEC_PUF_DONOR_SPINE,
            ASEC_PUF_DONOR_SPINE,
            ACS_2024_1YR_SPINE,
            "other",
        )
    )
    gate = acs_local_income_transfer_signal_gate(unknown, receipt=_receipt())
    assert "unsupported spine" in " ".join(gate.failures)
