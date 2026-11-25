"""ACS local lane: weeks unemployed and retirement deferrals (microcosm#1720)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_local_weeks_deferrals, acs_multispine
from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
)
from microcosm.build.us_runtime.acs_local_weeks_deferrals import (
    ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS,
    ACS_LOCAL_RETIREMENT_DEFERRAL_FAMILY,
    ACS_LOCAL_UNEMPLOYMENT_COMPENSATION_FEATURE,
    ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS,
    ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL,
    ACS_LOCAL_WEEKS_DEFERRALS_ISSUE,
    ACS_LOCAL_WEEKS_DEFERRALS_METHOD,
    ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY,
    acs_local_weeks_deferrals_target_families,
    require_acs_local_weeks_deferrals_donor,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    required_acs_transfer_inputs,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    postprocess_us_retirement_contribution_predictions,
)
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    postprocess_us_weeks_unemployed_predictions,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")
_WEEKS = "weeks_unemployed"
_UC = "unemployment_compensation"
_TRADITIONAL_401K, _ROTH_401K, _ROTH_IRA = ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS
#: The PUF clone role's values sit outside the ASEC role's support, so a fit on
#: the wrong role cannot go unnoticed.
_PUF_WEEKS = 52.0
_PUF_AMOUNT = 5_000_000.0


# ---------------------------------------------------------------------------
# The plan and the donor's own post-processing
# ---------------------------------------------------------------------------


def test_the_plan_is_person_only_and_disjoint_from_every_other_transfer() -> None:
    plan = acs_local_weeks_deferrals_target_families()
    assert plan == {
        "person": {
            ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY: (_WEEKS,),
            ACS_LOCAL_RETIREMENT_DEFERRAL_FAMILY: ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS,
        }
    }
    columns = set(ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS)
    assert not columns & required_acs_transfer_inputs()
    assert not columns & set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)
    # The shared plan draws the other two contribution leaves.
    others = set(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS) - columns
    assert others == {
        "traditional_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
    }
    assert others <= required_acs_transfer_inputs()


def test_the_plan_refuses_an_overlap_with_the_shared_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        acs_local_weeks_deferrals,
        "ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS",
        (*ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS, "traditional_ira_contributions_desired"),
    )
    with pytest.raises(ValueError, match="double counting"):
        acs_local_weeks_deferrals_target_families()


def test_the_weeks_post_processing_is_the_archived_rule() -> None:
    predicted = np.asarray([-3.0, 4.4, 4.6, 60.0, 12.0])
    uc = np.asarray([100.0, 100.0, 100.0, 100.0, 0.0])
    assert postprocess_us_weeks_unemployed_predictions(
        predicted, unemployment_compensation=uc
    ).tolist() == [0.0, 4.0, 5.0, 52.0, 0.0]
    assert postprocess_us_weeks_unemployed_predictions(
        predicted, unemployment_compensation=None, clip=(0, 10)
    ).tolist() == [0.0, 4.0, 5.0, 10.0, 10.0]


def test_the_deferral_post_processing_is_the_archived_rule() -> None:
    predictions = pd.DataFrame(
        {
            _TRADITIONAL_401K: [100.0, 200.0, -5.0],
            _ROTH_401K: [10.0, 20.0, 30.0],
            _ROTH_IRA: [1.0, -2.0, 3.0],
        }
    )
    result = postprocess_us_retirement_contribution_predictions(
        predictions, employment_income=np.asarray([0.0, 5.0, 5.0])
    )
    assert result[_TRADITIONAL_401K].tolist() == [0.0, 200.0, 0.0]
    assert result[_ROTH_401K].tolist() == [0.0, 20.0, 30.0]
    # The Roth IRA has no earnings gate in the donor's rule; it is clipped.
    assert result[_ROTH_IRA].tolist() == [1.0, 0.0, 3.0]
    pension = predictions.assign(self_employed_pension_contributions_desired=[5.0] * 3)
    with pytest.raises(ValueError, match="self_employment_income"):
        postprocess_us_retirement_contribution_predictions(
            pension, employment_income=np.ones(3)
        )
    gated = postprocess_us_retirement_contribution_predictions(
        pension,
        employment_income=np.ones(3),
        self_employment_income=np.asarray([0.0, 1.0, 0.0]),
    )
    assert gated["self_employed_pension_contributions_desired"].tolist() == [
        0.0,
        5.0,
        0.0,
    ]


# ---------------------------------------------------------------------------
# A real fit through the multispine seam
# ---------------------------------------------------------------------------


def _support_donor(n_asec: int = 60) -> Frame:
    """An ASEC-by-PUF donor: an ASEC observation role and its PUF clones.

    Every person is a household. On the ASEC role a third receive UC, weeks
    are 1-40 for most of them (and positive for two non-recipients looking
    for work), and the deferrals follow the archived split: 401(k) only with
    wages. Each PUF clone repeats its source person's demographics, wages and
    UC with out-of-support weeks and deferrals.
    """

    rng = np.random.default_rng(1720)
    n = 2 * n_asec
    ids = np.arange(1, n + 1)
    roles = np.asarray(["asec"] * n_asec + ["puf_tax_detail"] * n_asec, dtype=object)
    wages = np.where(rng.random(n_asec) < 0.7, rng.integers(5_000, 90_000, n_asec), 0)
    uc = np.where(rng.random(n_asec) < 0.35, rng.integers(500, 9_000, n_asec), 0)
    weeks = np.where(uc > 0, rng.integers(1, 41, n_asec), 0).astype(float)
    weeks[np.flatnonzero(uc == 0)[:2]] = 6.0
    contributing = (wages > 0) & (rng.random(n_asec) < 0.6)
    traditional = np.where(contributing, rng.integers(500, 9_000, n_asec), 0)
    roth_ira = np.where(rng.random(n_asec) < 0.2, rng.integers(200, 6_000, n_asec), 0)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "age": np.tile(rng.integers(18, 80, n_asec), 2).astype(float),
            "is_female": np.tile(rng.integers(0, 2, n_asec).astype(bool), 2),
            "employment_income_before_lsr": np.tile(wages.astype(float), 2),
            _UC: np.tile(uc.astype(float), 2),
            _WEEKS: np.concatenate([weeks, np.where(uc > 0, _PUF_WEEKS, 0.0)]),
            _TRADITIONAL_401K: np.concatenate(
                [traditional.astype(float), np.where(wages > 0, _PUF_AMOUNT, 0.0)]
            ),
            _ROTH_401K: np.concatenate(
                [0.1 * traditional, np.where(wages > 0, _PUF_AMOUNT + 1, 0.0)]
            ),
            _ROTH_IRA: np.concatenate(
                [roth_ira.astype(float), np.full(n_asec, _PUF_AMOUNT + 2)]
            ),
        }
    )
    for entity in US_SCHEMA.entities:
        if entity != "person":
            person[f"person_{entity}_id"] = ids
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


def _raw_acs(n: int = 24, children: int = 2) -> Frame:
    """A raw ACS PUMS spine of two-person households; the last ``children``
    persons are 10, with WAGP blank."""

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
            "ADJINC": 1_000_000,
            "WAGP": np.where(
                rng.random(n) < 0.7, rng.integers(5_000, 90_000, n), 0
            ).astype(float),
            "SEMP": 0.0,
            "SSP": 0.0,
            "SSIP": 0.0,
            "RETP": 0.0,
            "INTP": 0.0,
        }
    )
    person.loc[n - children :, "AGEP"] = 10
    person.loc[n - children :, ["WAGP", "SEMP"]] = np.nan
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


def _multispine(monkeypatch, tmp_path, *, weeks_deferrals: bool, calls: list):
    """The local lane's seam with the real QRF; only the loader is replaced.

    The shared plan transfers ``unemployment_compensation`` (on the PUF role,
    as the declared plan does), which the weeks family then reads.
    """

    real_transfer = acs_local_weeks_deferrals.transfer_acs_inputs

    def recorded(recipient, donor, **kwargs):
        calls.append(kwargs)
        return real_transfer(recipient, donor, **kwargs)

    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *_a, **_k: (_raw_acs(), {})
    )
    monkeypatch.setattr(acs_local_weeks_deferrals, "transfer_acs_inputs", recorded)
    shared_transfer = acs_multispine.transfer_acs_inputs

    def shared_recorded(recipient, donor, **kwargs):
        calls.append(kwargs)
        return shared_transfer(recipient, donor, **kwargs)

    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", shared_recorded)
    try:
        return acs_multispine.build_optional_acs_multispine(
            _support_donor(),
            AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
            target_families={"person": {"shared_uc": (_UC,)}},
            weeks_deferrals=weeks_deferrals,
            seed=5,
            n_estimators=5,
        )
    finally:
        monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", shared_transfer)
        monkeypatch.setattr(
            acs_local_weeks_deferrals, "transfer_acs_inputs", real_transfer
        )


def _rows(result, spine: str) -> pd.DataFrame:
    person = result.frame.table("person")
    return person.loc[person[TAG].eq(spine)].reset_index(drop=True)


def test_a_real_fit_fills_the_acs_rows_from_the_asec_role_only(
    monkeypatch, tmp_path
) -> None:
    donor_before = _support_donor().table("person")
    asec = donor_before[support_channel_column("person")].eq("asec")
    calls: list = []
    result = _multispine(monkeypatch, tmp_path, weeks_deferrals=True, calls=calls)
    acs = _rows(result, ACS_2024_1YR_SPINE)
    donor = _rows(result, ASEC_PUF_DONOR_SPINE)
    assert len(acs) == 24

    # Two fits: the shared plan on the PUF role, then this pass on the ASEC
    # role with its UC predictor extension.
    assert [call["donor_channel"] for call in calls] == [
        "auto",
        ACS_LOCAL_WEEKS_DEFERRALS_DONOR_CHANNEL,
    ]
    assert calls[1]["target_families"] == acs_local_weeks_deferrals_target_families()
    assert "person_predictor_extensions" not in calls[0]

    uc = acs[_UC].to_numpy(dtype=np.float64)
    weeks = acs[_WEEKS].to_numpy(dtype=np.float64)
    assert np.isfinite(weeks).all()
    assert ((weeks >= 0) & (weeks <= 52) & (weeks == np.floor(weeks))).all()
    # The archived zero rule, and never the PUF clones' 52.
    assert (weeks[uc <= 0] == 0).all()
    assert (weeks < _PUF_WEEKS).all()
    assert (weeks[uc > 0] > 0).any()
    wages = acs["employment_income_before_lsr"].fillna(0.0).to_numpy()
    for column in ACS_LOCAL_RETIREMENT_DEFERRAL_COLUMNS:
        values = acs[column].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all(), column
        assert (values >= 0).all(), column
        assert values.max() <= donor_before.loc[asec, column].max(), column
    # The 401(k) deferrals need wages; the children (WAGP blank) have none.
    for column in (_TRADITIONAL_401K, _ROTH_401K):
        assert (acs[column].to_numpy()[wages == 0] == 0).all(), column
    assert (acs[_TRADITIONAL_401K] > 0).any()
    assert acs[_TRADITIONAL_401K].iloc[-2:].tolist() == [0.0, 0.0]

    # Donor rows keep the donor release's values.
    for column in (*ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS, _UC):
        np.testing.assert_array_equal(
            donor[column].to_numpy(dtype=np.float64),
            donor_before[column].to_numpy(dtype=np.float64),
        )

    imputed = {item["column"]: item for item in result.provenance["imputed_inputs"]}
    for column in ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS:
        assert imputed[column]["donor_channel"] == "asec", column
        used = set(imputed[column]["predictors"])
        assert (ACS_LOCAL_UNEMPLOYMENT_COMPENSATION_FEATURE in used) is (
            column == _WEEKS
        ), column
    assert imputed[_UC]["donor_channel"] == "puf_tax_detail"
    names = [record.fit_name for record in result.fit_records]
    assert any(ACS_LOCAL_WEEKS_UNEMPLOYED_FAMILY in name for name in names)

    receipt = result.provenance["acs_local_weeks_deferrals"]
    assert receipt["issue"] == ACS_LOCAL_WEEKS_DEFERRALS_ISSUE == "microcosm#1720"
    assert receipt["method"] == ACS_LOCAL_WEEKS_DEFERRALS_METHOD
    assert receipt["donor_channel"] == receipt["resolved_donor_channel"] == "asec"
    assert receipt["donor"]["person_rows"] == 60
    assert receipt["donor"]["weeks_positive_without_uc_rows"] == 2
    assert receipt["acs_persons"] == 24
    assert receipt["acs_uc_recipient_rows"] == int((uc > 0).sum())
    for column in ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS:
        entry = receipt["columns"][column]
        assert entry["imputed_rows"] == entry["filled_rows"] == 24, column
        assert entry["unmodeled_rows"] == 0, column
        assert entry["donor_channel"] == ["asec"], column
    assert receipt["columns"][_WEEKS]["acs"]["positive_rows"] == int((weeks > 0).sum())
    json.dumps(result.provenance, allow_nan=False)


def test_the_pass_leaves_every_earlier_draw_unchanged(monkeypatch, tmp_path) -> None:
    """The proof microcosm#1720 asks for: with and without the pass, every
    other column of both spines, the earlier provenance and the earlier fits
    are identical, because the pass runs after them and its families draw
    with seeds of their own."""

    with_calls: list = []
    without_calls: list = []
    with_pass = _multispine(
        monkeypatch, tmp_path, weeks_deferrals=True, calls=with_calls
    )
    without_pass = _multispine(
        monkeypatch, tmp_path, weeks_deferrals=False, calls=without_calls
    )
    assert len(with_calls) == 2 and len(without_calls) == 1
    assert with_calls[0] == without_calls[0]

    new = list(ACS_LOCAL_WEEKS_DEFERRALS_COLUMNS)
    with_person = with_pass.frame.table("person")
    without_person = without_pass.frame.table("person")
    acs = without_person[TAG].eq(ACS_2024_1YR_SPINE)
    # Without the pass the ACS rows have no value for the four columns.
    assert without_person.loc[acs, new].isna().all().all()
    pd.testing.assert_frame_equal(
        with_person.drop(columns=new),
        without_person.drop(columns=new),
        check_like=True,
    )
    earlier = [
        item
        for item in with_pass.provenance["imputed_inputs"]
        if item["column"] not in new
    ]
    assert earlier == without_pass.provenance["imputed_inputs"]
    assert with_pass.fit_records[: len(without_pass.fit_records)] == (
        without_pass.fit_records
    )
    assert "acs_local_weeks_deferrals" not in without_pass.provenance


# ---------------------------------------------------------------------------
# The donor
# ---------------------------------------------------------------------------


def test_a_donor_without_a_column_or_the_asec_role_is_refused() -> None:
    donor = _support_donor()
    summary = require_acs_local_weeks_deferrals_donor(donor)
    assert summary["channel"] == "asec"
    assert summary["columns"][_WEEKS]["positive_rows"] > 0
    tables = {entity: donor.table(entity) for entity in donor.entities}
    tables["person"] = tables["person"].drop(columns=[_ROTH_IRA])
    reduced = Frame(
        tables, US_SCHEMA, {"household": donor.weights_for("household")}, donor.strata
    )
    with pytest.raises(ValueError, match="roth_ira_contributions_desired"):
        require_acs_local_weeks_deferrals_donor(reduced)
    plain: dict[str, Any] = {
        entity: donor.table(entity).drop(
            columns=[
                support_channel_column(entity),
                support_source_id_column(entity),
                support_clone_index_column(entity),
            ]
        )
        for entity in donor.entities
    }
    with pytest.raises(ValueError, match="ASEC observation role"):
        require_acs_local_weeks_deferrals_donor(
            Frame(
                plain,
                US_SCHEMA,
                {"household": donor.weights_for("household")},
                donor.strata,
            )
        )
