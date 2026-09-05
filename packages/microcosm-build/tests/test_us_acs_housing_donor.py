from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_housing_donor import (
    load_acs_2024_housing_donor,
    load_synthetic_acs_2024_housing_donor,
    prepare_acs_2024_reported_rent,
    project_synthetic_acs_2024_housing_donor,
)
from microcosm.build.us_runtime.acs_pums import AcsPumsSource

TARGET = "acs_annual_reported_contract_rent"


def _tables():
    # Invented rows. Non-unit factors deliberately expose missing/double adjustment.
    h = pd.DataFrame(
        [
            dict(
                SERIALNO="r1",
                STATE="06",
                WGTP=10,
                NP=2,
                TYPEHUGQ=1,
                TEN=3,
                ADJHSG=1_200_000,
                RNTP=500,
                TAXAMT=None,
                FRNTP=0,
                FTAXP=0,
            ),
            dict(
                SERIALNO="r2",
                STATE="36",
                WGTP=20,
                NP=1,
                TYPEHUGQ=1,
                TEN=3,
                ADJHSG=1_000_000,
                RNTP=1000,
                TAXAMT=None,
                FRNTP=0,
                FTAXP=0,
            ),
            dict(
                SERIALNO="r3",
                STATE="06",
                WGTP=30,
                NP=1,
                TYPEHUGQ=1,
                TEN=3,
                ADJHSG=1_000_000,
                RNTP=9000,
                TAXAMT=None,
                FRNTP=1,
                FTAXP=0,
            ),
            dict(
                SERIALNO="o1",
                STATE="06",
                WGTP=40,
                NP=1,
                TYPEHUGQ=1,
                TEN=2,
                ADJHSG=1_200_000,
                RNTP=None,
                TAXAMT=1000,
                FRNTP=0,
                FTAXP=0,
            ),
            dict(
                SERIALNO="o2",
                STATE="36",
                WGTP=50,
                NP=1,
                TYPEHUGQ=1,
                TEN=1,
                ADJHSG=1_000_000,
                RNTP=None,
                TAXAMT=0,
                FRNTP=0,
                FTAXP=1,
            ),
            dict(
                SERIALNO="n1",
                STATE="06",
                WGTP=60,
                NP=1,
                TYPEHUGQ=1,
                TEN=4,
                ADJHSG=1_000_000,
                RNTP=None,
                TAXAMT=None,
                FRNTP=0,
                FTAXP=0,
            ),
            dict(
                SERIALNO="gq",
                STATE="06",
                WGTP=0,
                NP=1,
                TYPEHUGQ=2,
                TEN=None,
                ADJHSG=1_000_000,
                RNTP=None,
                TAXAMT=None,
                FRNTP=None,
                FTAXP=None,
            ),
            dict(
                SERIALNO="vacant",
                STATE="06",
                WGTP=70,
                NP=0,
                TYPEHUGQ=1,
                TEN=None,
                ADJHSG=1_000_000,
                RNTP=800,
                TAXAMT=None,
                FRNTP=0,
                FTAXP=0,
            ),
        ]
    )
    p = pd.DataFrame(
        [
            dict(
                SERIALNO=row.SERIALNO,
                SPORDER=1,
                RELSHIPP=37 if row.SERIALNO == "gq" else 20,
                AGEP=40,
                SEX=1,
                ADJINC=1_100_000,
                WAGP=100,
                SEMP=-10,
                SSP=20,
                RETP=30,
            )
            for row in h.itertuples()
            if row.NP
        ]
    )
    child = dict(
        p.iloc[0],
        SPORDER=2,
        RELSHIPP=25,
        AGEP=10,
        WAGP=None,
        SEMP=None,
        SSP=None,
        RETP=None,
    )
    p = pd.concat([p, pd.DataFrame([child])], ignore_index=True)
    return h, p


def _source(tmp_path: Path, h: pd.DataFrame, p: pd.DataFrame) -> AcsPumsSource:
    for role, table, prefix in (("hus", h, "h"), ("pus", p, "p")):
        with ZipFile(tmp_path / f"csv_{role}.zip", "w") as z:
            # Multi-member/chunk ordering must not become a household alignment rule.
            z.writestr(f"psam_{prefix}usa.csv", table.iloc[::2].to_csv(index=False))
            z.writestr(f"psam_{prefix}usb.csv", table.iloc[1::2].to_csv(index=False))
    return AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")


def test_scalar_money_oracle_linked_heads_weights_and_honest_universes():
    h, p = _tables()
    original_h, original_p = h.copy(deep=True), p.copy(deep=True)
    result = project_synthetic_acs_2024_housing_donor(h, p)
    t = result.table
    assert t.index.tolist() == ["n1", "o1", "o2", "r1", "r2", "r3"]
    assert t.loc["r1", TARGET] == 7200  # 500 * 1.2 * 12, exactly once.
    assert t.loc["o1", "acs_annual_property_tax"] == 1200  # Already annual.
    assert t.loc["o2", "acs_annual_property_tax"] == 0
    assert t.loc["r1", "employment_income_before_lsr"] == pytest.approx(110)
    assert t.loc["r1", "self_employment_income_before_lsr"] == pytest.approx(-11)
    assert t.loc["r1", "acs_social_security_income"] == pytest.approx(22)
    assert t.loc["r1", "acs_retirement_income"] == pytest.approx(33)
    assert t.loc["r1", "household_size"] == 2
    assert t.loc["r1", "household_weight"] == 10
    assert t.loc["o1", "TEN"] == 2
    assert t.loc["o1", "tenure_type"] == "OWNED_WITH_MORTGAGE"
    assert t.loc[["n1", "o1", "o2"], TARGET].isna().all()
    assert not t.loc["r3", "rent_observed"]
    assert not t.loc["o2", "property_tax_observed"]  # allocated observed zero
    assert not {"pre_subsidy_rent", "rent", "taxable_private_pension_income"} & set(t)
    assert result.provenance["price_year"] == result.provenance["vintage"] == 2024
    assert result.provenance["source_evidence"] == "synthetic_fixture_tables"
    assert (
        result.provenance["gq_rows_excluded"]
        == result.provenance["vacant_rows_excluded"]
        == 1
    )
    pd.testing.assert_frame_equal(h, original_h)
    pd.testing.assert_frame_equal(p, original_p)


def test_alignment_uses_reference_person_identity_not_row_position():
    h, p = _tables()
    p.loc[(p.SERIALNO == "r1") & (p.RELSHIPP == 20), "SPORDER"] = 2
    p.loc[(p.SERIALNO == "r1") & (p.RELSHIPP == 25), "SPORDER"] = 1
    a = project_synthetic_acs_2024_housing_donor(h, p)
    b = project_synthetic_acs_2024_housing_donor(
        h.sample(frac=1, random_state=7), p.iloc[::-1]
    )
    pd.testing.assert_frame_equal(a.table, b.table)
    assert a.table.loc["r1", "source_head_person_number"] == 2
    assert a.table.loc["r1", "employment_income_before_lsr"] == pytest.approx(110)


def test_synthetic_archive_path_uses_genuine_projection_but_cannot_claim_real_pins(
    tmp_path,
):
    h, p = _tables()
    source = _source(tmp_path, h, p)
    loaded = load_synthetic_acs_2024_housing_donor(source, chunksize=1)
    direct = project_synthetic_acs_2024_housing_donor(h, p)
    pd.testing.assert_frame_equal(loaded.table, direct.table, check_dtype=False)
    assert loaded.provenance["source_evidence"] == "synthetic_fixture_archives"
    assert len(loaded.provenance["csv_members"]["household"]) == 2
    assert len(loaded.provenance["archives"]["household"]["sha256"]) == 64
    with pytest.raises(ValueError, match="sha-256 verification failed"):
        load_acs_2024_housing_donor(source, chunksize=1)
    with pytest.raises(ValueError, match="independent full"):
        load_synthetic_acs_2024_housing_donor(replace(source, max_households=1))


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("FRNTP", None, "FRNTP"),
        ("FRNTP", 2, "FRNTP"),
        ("FTAXP", -1, "FTAXP"),
        ("FRNTP", 0.5, "FRNTP"),
        ("TEN", 0, "TEN"),
        ("RNTP", None, "RNTP.*universe"),
        ("RNTP", 0, "RNTP.*range"),
        ("RNTP", -1, "RNTP.*range"),
        ("RNTP", 1.5, "RNTP.*range"),
        ("RNTP", 100000, "RNTP.*range"),
        ("RNTP", "broken", "parse"),
        ("TAXAMT", 100, "TAXAMT.*universe"),
        ("NP", 3, "row-count"),
        ("WGTP", 0, "WGTP"),
        ("WGTP", -1, "WGTP"),
        ("WGTP", np.inf, "nonfinite"),
        ("WGTP", 10000, "9999"),
        ("STATE", "72", "non-US"),
        ("ADJHSG", 0, "ADJHSG"),
        ("ADJHSG", np.nan, "ADJHSG"),
    ],
)
def test_rejects_invalid_household_fields(column, value, match):
    h, p = _tables()
    h[column] = h[column].astype(object)
    h.loc[h.SERIALNO == "r1", column] = value
    with pytest.raises(ValueError, match=match):
        project_synthetic_acs_2024_housing_donor(h, p)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("ADJINC", 0, "ADJINC"),
        ("ADJINC", np.inf, "ADJINC"),
        ("WAGP", np.nan, "WAGP"),
        ("WAGP", -1, "WAGP"),
        ("SEMP", np.inf, "SEMP"),
        ("SEX", 3, "SEX"),
        ("SPORDER", 0, "SPORDER"),
        ("SPORDER", 1.5, "SPORDER"),
        ("RELSHIPP", 21, "exactly one"),
        ("RELSHIPP", 37, "RELSHIPP"),
    ],
)
def test_rejects_invalid_reference_person_fields(column, value, match):
    h, p = _tables()
    p[column] = p[column].astype(object)
    p.loc[(p.SERIALNO == "r1") & (p.RELSHIPP == 20), column] = value
    with pytest.raises(ValueError, match=match):
        project_synthetic_acs_2024_housing_donor(h, p)


def test_rejects_duplicate_or_orphan_members_and_missing_projection_flags(tmp_path):
    h, p = _tables()
    for bad_h, bad_p, match in (
        (pd.concat([h, h.iloc[:1]]), p, "SERIALNO must be unique"),
        (h, pd.concat([p, p.iloc[:1]]), "SPORDER must be unique"),
        (h, p.assign(SERIALNO="orphan"), "missing household"),
        (h.drop(columns="FRNTP"), p, "missing source columns"),
    ):
        with pytest.raises(ValueError, match=match):
            project_synthetic_acs_2024_housing_donor(bad_h, bad_p)
    source = _source(tmp_path, h.drop(columns="FRNTP"), p)
    with pytest.raises(ValueError, match="missing required.*FRNTP"):
        load_synthetic_acs_2024_housing_donor(source)
    h.loc[h.SERIALNO == "gq", "FRNTP"] = 0
    with pytest.raises(ValueError, match="blank for GQ"):
        project_synthetic_acs_2024_housing_donor(h, p)


def test_joint_sampler_keeps_tax_only_rows_and_rent_fit_uses_its_own_support():
    h, p = _tables()
    result = prepare_acs_2024_reported_rent(
        project_synthetic_acs_2024_housing_donor(h, p)
    )
    assert result.provenance["sample_rows"] == 3  # r1, r2, unallocated o1 tax.
    assert result.provenance["fit_rows"] == 2  # only unallocated reported rents.
    assert result.training.index.tolist() == ["r1", "r2"]
    assert result.training[TARGET].tolist() == [7200, 12000]
    assert result.training.household_weight.tolist() == [10, 20]
    assert result.provenance["category_levels"]["tenure_type"] == ["RENTED"]
    assert result.provenance["weight_kind"] == "design"
    assert TARGET not in result.predictors
    assert "state_code_str__36" in result.predictors
    assert "source_head_person_number" not in result.predictors
    assert not any("pre_subsidy" in c for c in result.training)
    schema = result.provenance["feature_schema"]
    assert schema["ordered_predictors"] == list(result.predictors)
    assert schema["price_year"] == 2024
    assert (
        result.provenance["feature_schema_sha256"]
        == hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_feature_identity_changes_when_donor_category_omission_changes():
    h, p = _tables()
    before = prepare_acs_2024_reported_rent(
        project_synthetic_acs_2024_housing_donor(h, p)
    )
    h["STATE"] = "06"
    after = prepare_acs_2024_reported_rent(
        project_synthetic_acs_2024_housing_donor(h, p)
    )
    assert "state_code_str__36" in before.predictors
    assert "state_code_str__36" not in after.predictors
    assert (
        before.provenance["feature_schema_sha256"]
        != after.provenance["feature_schema_sha256"]
    )


def test_state_aliases_must_agree_by_value_and_property_tax_has_its_own_domain():
    h, p = _tables()
    h["ST"] = h["STATE"].astype(float)
    project_synthetic_acs_2024_housing_donor(h, p)
    h.loc[h.SERIALNO == "r1", "ST"] = 36
    with pytest.raises(ValueError, match="ST and STATE disagree"):
        project_synthetic_acs_2024_housing_donor(h, p)
    h, p = _tables()
    for value in (-1, 57501, np.nan):
        h.loc[h.SERIALNO == "o1", "TAXAMT"] = value
        with pytest.raises(ValueError, match="TAXAMT"):
            project_synthetic_acs_2024_housing_donor(h, p)


def test_missing_observed_target_fails_and_prepared_flags_are_revalidated():
    h, p = _tables()
    h.loc[h.TEN.isin([1, 2]), "FTAXP"] = 1
    donor = project_synthetic_acs_2024_housing_donor(h, p)
    with pytest.raises(ValueError, match="no observed rows for real_estate_taxes"):
        prepare_acs_2024_reported_rent(donor)
    h, p = _tables()
    donor = project_synthetic_acs_2024_housing_donor(h, p)
    donor.table.loc["r1", "FRNTP"] = 2
    with pytest.raises(ValueError, match="content identity"):
        prepare_acs_2024_reported_rent(donor)


def test_tiny_actual_qrf_consumes_reported_contract_rent_interface(monkeypatch):
    # All rows are invented. This exercises the real projection, sampler,
    # encoding and weighted QRF; it is not an independent validation dataset.
    from microcosm.fit import QRF

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    donor = project_synthetic_acs_2024_housing_donor(*_tables())
    prepared = prepare_acs_2024_reported_rent(donor)
    training = prepared.verified_training()
    fitted = QRF(n_estimators=2, seed=17).fit(
        training,
        predictors=list(prepared.predictors),
        targets=[TARGET],
        weights="household_weight",
    )
    result = fitted.predict(training.loc[:, prepared.predictors])
    assert result.index.equals(prepared.training.index)
    assert result.columns.tolist() == [TARGET]
    assert np.isfinite(result[TARGET]).all()
    assert result[TARGET].between(7200, 12000).all()
