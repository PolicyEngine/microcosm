"""The separate ASEC-channel income transfer for the ACS local lane
(microcosm#1022): its plan, receipt and gate, and a real QRF fit through the
multispine seam."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_local_income, acs_multispine
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
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    ASEC_PUF_DONOR_SPINE,
    declared_acs_transfer_target_families,
    required_acs_transfer_inputs,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
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


# ---------------------------------------------------------------------------
# A real QRF fit through the multispine seam (review of microcosm#1056)
# ---------------------------------------------------------------------------

#: A shared-plan retirement leaf: the shared transfer, not this pass, owns it.
_SHARED_LEAF = "taxable_ira_distributions"
#: The PUF clone role's amounts sit far outside the ASEC role's support, so a
#: fit on the wrong role cannot go unnoticed.
_PUF_ROLE_AMOUNT = 5_000_000.0


def _support_donor(n_asec: int = 40) -> Frame:
    """An ASEC-by-PUF donor: an ASEC observation role and its PUF clones.

    Every person is a household. The ASEC role carries measured income (half
    of each leaf positive); each PUF clone repeats its source person's
    demographics with out-of-support amounts. Only the PUF role carries the
    shared-plan leaf.
    """

    rng = np.random.default_rng(1056)
    n = 2 * n_asec
    ids = np.arange(1, n + 1)
    roles = np.asarray(["asec"] * n_asec + ["puf_tax_detail"] * n_asec, dtype=object)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "age": np.tile(rng.integers(18, 85, n_asec), 2).astype(float),
            "is_female": np.tile(rng.integers(0, 2, n_asec).astype(bool), 2),
        }
    )
    for entity in US_SCHEMA.entities:
        if entity != "person":
            person[f"person_{entity}_id"] = ids
    for offset, column in enumerate(ACS_LOCAL_INCOME_TRANSFER_COLUMNS):
        asec = np.where(
            rng.random(n_asec) < 0.5, rng.integers(100, 20_000, n_asec), 0
        ).astype(float)
        puf = np.where(asec > 0, _PUF_ROLE_AMOUNT + offset, 0.0)
        person[column] = np.concatenate([asec, puf])
    person[_SHARED_LEAF] = np.concatenate(
        [np.zeros(n_asec), np.where(rng.random(n_asec) < 0.5, 3_000.0, 0.0)]
    )
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity != "person":
            tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"]["state_fips"] = np.resize([6, 36, 48], n)
    for entity, table in tables.items():
        table[support_channel_column(entity)] = roles
        table[support_source_id_column(entity)] = np.tile(np.arange(1, n_asec + 1), 2)
        table[support_clone_index_column(entity)] = (roles == "puf_tax_detail").astype(
            int
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
        pd.Series(ASEC_PUF_DONOR_SPINE, index=person.index, dtype=object),
    )


def _raw_acs(n: int = 24, **person_columns: Any) -> Frame:
    """A raw ACS PUMS spine: adults in two-person households, one PUMA."""

    rng = np.random.default_rng(2024)
    households = 500 + np.arange(n) // 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(500, 500 + n),
            "person_household_id": households,
            "person_tax_unit_id": households,
            "person_spm_unit_id": households,
            "person_family_id": households,
            "person_marital_unit_id": households,
            "AGEP": rng.integers(18, 85, n),
            "SEX": rng.integers(1, 3, n),
            "RELSHIPP": np.resize([20, 21], n),
            "ADJINC": 1_100_000,
            "WAGP": rng.integers(0, 60_000, n).astype(float),
            "SEMP": 0.0,
            "SSP": 0.0,
            "SSIP": 0.0,
            "RETP": np.where(rng.random(n) < 0.3, 12_000.0, 0.0),
            "INTP": 0.0,
            **person_columns,
        }
    )
    unique = np.unique(households)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": unique,
                    "state_fips": np.resize([6, 36, 48], len(unique)),
                    "puma": "0600100",
                    "ADJHSG": 1_000_000,
                    "TEN": 3,
                    "RNTP": 1_000.0,
                    "GRNTP": 1_200.0,
                    "TAXAMT": np.nan,
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": unique}),
            "spm_unit": pd.DataFrame({"spm_unit_id": unique}),
            "family": pd.DataFrame({"family_id": unique}),
            "marital_unit": pd.DataFrame({"marital_unit_id": unique}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(len(unique), 50.0), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def _multispine(monkeypatch, tmp_path, base: Frame, raw: Frame, calls: list):
    """Run the local lane's multispine seam with the real QRF.

    Only the ACS loader is replaced (by ``raw``); every
    ``transfer_acs_inputs`` call is recorded and delegated to the real one.
    """

    real_transfer = acs_multispine.transfer_acs_inputs

    def recorded(recipient, donor, **kwargs):
        calls.append(kwargs)
        return real_transfer(recipient, donor, **kwargs)

    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *_a, **_k: (raw, {})
    )
    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", recorded)
    return acs_multispine.build_optional_acs_multispine(
        base,
        AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
        target_families={"person": {"shared_retirement": (_SHARED_LEAF,)}},
        income_transfer=True,
        seed=5,
        n_estimators=5,
    )


def test_a_real_fit_fills_missing_acs_income_from_the_asec_role_only(
    monkeypatch, tmp_path
):
    base = _support_donor()
    donor_before = base.table("person").copy()
    stored = np.full(24, np.nan)
    stored[[0, 5]] = [777.0, 0.0]
    raw = _raw_acs(child_support_received=stored)
    calls: list = []

    result = _multispine(monkeypatch, tmp_path, base, raw, calls)

    person = result.frame.table("person")
    acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    donor = person.loc[person[TAG].eq(ASEC_PUF_DONOR_SPINE)].reset_index(drop=True)
    asec_role = donor_before[support_channel_column("person")].eq("asec")
    assert len(acs) == 24
    # Every missing ACS cell of all nine leaves is filled finite and >= 0,
    # inside the ASEC role's measured support, never the PUF clones' values.
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        values = acs[column].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all(), column
        assert (values >= 0).all(), column
        assert values.max() <= donor_before.loc[asec_role, column].max(), column
        assert (acs[column] > 0).any(), column
    # Stored ACS values survive the null-only merge, a stored 0 included.
    assert acs.loc[0, "child_support_received"] == 777.0
    assert acs.loc[5, "child_support_received"] == 0.0
    # Donor rows, and the donor frame itself, are unchanged.
    for column in (*ACS_LOCAL_INCOME_TRANSFER_COLUMNS, _SHARED_LEAF):
        np.testing.assert_array_equal(
            donor[column].to_numpy(), donor_before[column].to_numpy()
        )
    pd.testing.assert_frame_equal(base.table("person"), donor_before)

    # Two real fits: this pass on the ASEC role, then the shared plan on the
    # PUF role; neither transfers the other's leaves.
    assert [call["donor_channel"] for call in calls] == [
        ACS_LOCAL_INCOME_DONOR_CHANNEL,
        ACS_DONOR_CHANNEL_AUTO,
    ]
    assert calls[0]["target_families"] == acs_local_income_transfer_target_families()
    imputed = result.provenance["imputed_inputs"]
    local = {item["column"]: item for item in imputed if item["column"] != _SHARED_LEAF}
    assert set(local) == set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)
    assert len(imputed) == len(ACS_LOCAL_INCOME_TRANSFER_COLUMNS) + 1
    for item in local.values():
        assert item["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    (shared,) = [item for item in imputed if item["column"] == _SHARED_LEAF]
    assert shared["donor_channel"] == "puf_tax_detail"
    assert result.provenance["fit_configuration"]["income_donor_channel"] == (
        ACS_LOCAL_INCOME_DONOR_CHANNEL
    )

    # The receipt comes from the fit and the gate accepts it.
    receipt = result.provenance["acs_local_income_transfer"]
    assert receipt["issue"] == ACS_LOCAL_INCOME_TRANSFER_ISSUE
    assert receipt["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert receipt["acs_persons"] == 24
    assert receipt["donor"]["person_rows"] == 40
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        entry = receipt["columns"][column]
        assert entry["imputed_rows"] == (
            22 if column == "child_support_received" else 24
        )
        assert entry["unmodeled_rows"] == 0
        assert entry["donor_channel"] == [ACS_LOCAL_INCOME_DONOR_CHANNEL]
        assert entry["predictors"]
    gate = acs_local_income_transfer_signal_gate(result.frame, receipt=receipt)
    assert gate.passed, gate.failures


def test_a_real_fit_refuses_a_plan_that_overlaps_the_shared_plan(monkeypatch, tmp_path):
    shared_leaves = set(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)
    assert shared_leaves <= required_acs_transfer_inputs()
    overlapping = {
        **ACS_LOCAL_INCOME_TRANSFER_FAMILIES,
        "acs_local_retirement_distributions": (
            *ACS_LOCAL_INCOME_TRANSFER_FAMILIES["acs_local_retirement_distributions"],
            _SHARED_LEAF,
        ),
    }
    monkeypatch.setattr(
        acs_local_income, "ACS_LOCAL_INCOME_TRANSFER_FAMILIES", overlapping
    )
    monkeypatch.setattr(
        acs_local_income,
        "ACS_LOCAL_INCOME_TRANSFER_COLUMNS",
        (*ACS_LOCAL_INCOME_TRANSFER_COLUMNS, _SHARED_LEAF),
    )
    calls: list = []
    with pytest.raises(ValueError, match="double counting"):
        _multispine(monkeypatch, tmp_path, _support_donor(), _raw_acs(), calls)
    # Refused before any fit: no transfer ran, so no leaf was written twice.
    assert calls == []
