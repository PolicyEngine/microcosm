import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import l0_refit_export
from populace.build.us_runtime.l0_refit_export import (
    assert_required_us_release_source_columns,
    attach_l0_refit_entity_weights,
    attach_l0_refit_weights,
    export_us_l0_refit_h5,
    load_l0_refit_npz,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _us_frame(**person_extra: object) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 20], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 201], dtype="int64"),
            "person_spm_unit_id": np.asarray([1000, 2000, 2000], dtype="int64"),
            "person_family_id": np.asarray([10000, 20000, 20000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [100000, 200000, 200001], dtype="int64"
            ),
            "ssn_card_type": ["CITIZEN", "CITIZEN", "NONE"],
            "immigration_status_str": ["CITIZEN", "CITIZEN", "UNDOCUMENTED"],
            "weekly_hours_worked_before_lsr": [40.0, 20.0, 0.0],
            "hours_worked_last_week": [40.0, 18.0, 0.0],
            "weeks_worked": [52.0, 26.0, 0.0],
            "is_household_head": [True, True, False],
            "is_separated": [False, False, True],
            "is_surviving_spouse": [False, True, False],
            "is_disabled": [False, True, False],
            "is_blind": [False, False, True],
            "is_full_time_college_student": [False, False, True],
            "own_children_in_household": [1.0, 0.0, 2.0],
            "veterans_benefits": [0.0, 8_000.0, 3_000.0],
            "is_pregnant": [True, False, True],
            "is_snap_abawd_discretionary_exempt": [True, False, True],
            "bank_account_assets": [5_000.0, 0.0, 1_200.0],
            "stock_assets": [0.0, 30_000.0, 0.0],
            "bond_assets": [0.0, 0.0, 800.0],
            "tip_income": [2_400.0, 0.0, 600.0],
            "treasury_tipped_occupation_code": [101, 0, 304],
            "alimony_income": [5_000.0, 0.0, 1_000.0],
            "alimony_expense": [0.0, 2_500.0, 0.0],
            "child_support_received": [3_600.0, 0.0, 1_200.0],
            "child_support_expense": [0.0, 2_400.0, 600.0],
            "disability_benefits": [0.0, 5_000.0, 1_500.0],
            "educator_expense": [0.0, 300.0, 150.0],
            "other_health_insurance_premiums": [1_200.0, 0.0, 600.0],
            # Keep both signs without making the fixture's signed weighted
            # total nearly cancel; the mass-parity tests reweight household 20.
            "farm_operations_income": [4_000.0, 2_500.0, -600.0],
            "farm_rent_income": [0.0, 1_500.0, -600.0],
            "casualty_loss": [0.0, 2_500.0, 0.0],
            "unreimbursed_business_employee_expenses": [1_200.0, 0.0, 800.0],
            "investment_income_elected_form_4952": [0.0, 500.0, 250.0],
            "salt_refund_income": [0.0, 1_200.0, 400.0],
            "long_term_capital_gains_on_collectibles": [0.0, 2_500.0, 1_000.0],
            "qualified_tuition_expenses": [1_000.0, 0.0, 2_500.0],
            "educational_assistance": [0.0, 500.0, 0.0],
            "traditional_401k_contributions_desired": [1_000.0, 0.0, 500.0],
            "roth_401k_contributions_desired": [200.0, 0.0, 100.0],
            "traditional_ira_contributions_desired": [300.0, 0.0, 150.0],
            "roth_ira_contributions_desired": [400.0, 0.0, 200.0],
            "self_employed_pension_contributions_desired": [0.0, 800.0, 0.0],
            "taxable_401k_distributions": [1_000.0, 0.0, 500.0],
            "taxable_403b_distributions": [0.0, 600.0, 0.0],
            "tax_exempt_ira_distributions": [300.0, 0.0, 100.0],
            "taxable_ira_distributions": [400.0, 0.0, 200.0],
            "keogh_distributions": [0.0, 700.0, 0.0],
            "taxable_sep_distributions": [0.0, 0.0, 250.0],
            "estate_income_would_be_qualified": [True, False, True],
            "farm_operations_income_would_be_qualified": [True, True, False],
            "farm_rent_income_would_be_qualified": [True, False, True],
            "partnership_s_corp_income_would_be_qualified": [True, False, True],
            "rental_income_would_be_qualified": [True, False, True],
            "self_employment_income_would_be_qualified": [True, True, False],
            "sstb_self_employment_income_would_be_qualified": [False, True, False],
            "business_is_sstb": [False, True, False],
            "qualified_bdc_income": [0.0, 20.0, 0.0],
            "qualified_reit_and_ptp_income": [100.0, 0.0, 50.0],
            "sstb_self_employment_income_before_lsr": [0.0, 1_000.0, 0.0],
            "sstb_unadjusted_basis_qualified_property": [0.0, 5_000.0, 0.0],
            "sstb_w2_wages_from_qualified_business": [0.0, 2_000.0, 0.0],
            "unadjusted_basis_qualified_property": [1_000.0, 5_000.0, 0.0],
            "w2_wages_from_qualified_business": [500.0, 2_000.0, 0.0],
            "is_pursuing_credential_for_american_opportunity_credit": [
                True,
                False,
                True,
            ],
            "attends_eligible_educational_institution_for_american_opportunity_credit": [
                True,
                False,
                True,
            ],
            "is_enrolled_at_least_half_time_for_american_opportunity_credit": [
                True,
                False,
                True,
            ],
            "has_american_opportunity_credit_1098_t_or_exception": [
                True,
                False,
                True,
            ],
            "has_american_opportunity_credit_institution_ein": [
                True,
                False,
                True,
            ],
            "cps_race": [1, 2, 4],
            "is_hispanic": [False, True, False],
            "detailed_occupation_recode": [1, 20, 53],
            "has_never_worked": [False, False, True],
            "is_military": [False, True, False],
            "is_computer_scientist": [False, True, False],
            "is_executive_administrative_professional": [False, True, False],
            "is_farmer_fisher": [False, True, False],
            "hourly_wage": [25.0, 18.0, 0.0],
            "is_paid_hourly": [False, True, False],
            "is_union_member_or_covered": [False, True, False],
            "fsla_overtime_premium": [0.0, 1_000.0, 0.0],
            "pre_subsidy_rent": [0.0, 12_000.0, 0.0],
            "self_employment_income_last_year": [0.0, 8_000.0, -1_000.0],
            "previous_year_income_available": [False, True, False],
            **person_extra,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([10, 20], dtype="int64"),
                    "state_fips": np.asarray([36, 6], dtype="int64"),
                    "congressional_district_geoid": np.asarray(
                        [3612, 653], dtype="int64"
                    ),
                    "block_geoid": ["360610001001000", "060370001001000"],
                    "tract_geoid": ["36061000100", "06037000100"],
                    "county_fips": ["36061", "06037"],
                    "place_fips": ["51000", "44000"],
                    "sldu": ["027", "024"],
                    "sldl": ["075", "051"],
                    "cbsa_code": ["35620", "31080"],
                    "auto_loan_balance": [0.0, 24_000.0],
                    "auto_loan_interest": [0.0, 1_200.0],
                    "qualified_passenger_vehicle_loan_interest": [0.0, 240.0],
                    "net_worth": [-50_000.0, 350_000.0],
                    "household_vehicles_owned": [0, 2],
                    "household_vehicles_value": [0.0, 30_000.0],
                    "tenure_type": ["OWNED_WITH_MORTGAGE", "RENTED"],
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([100, 200, 201], dtype="int64"),
                    "takes_up_aca_if_eligible": np.asarray(
                        [False, True, False], dtype=bool
                    ),
                    "selected_marketplace_plan_benchmark_ratio": np.asarray(
                        [1.0, 0.8, 1.2], dtype="float64"
                    ),
                    "domestic_production_ald": [0.0, 10_000.0, 5_000.0],
                    "unrecaptured_section_1250_gain": [0.0, 2_000.0, 500.0],
                }
            ),
            "spm_unit": pd.DataFrame(
                {
                    "spm_unit_id": np.asarray([1000, 2000], dtype="int64"),
                    "spm_unit_pre_subsidy_childcare_expenses": [0.0, 1_200.0],
                    "spm_unit_energy_subsidy": [0.0, 600.0],
                    "receives_housing_assistance": [False, True],
                    "spm_unit_tenure_type": [
                        "OWNER_WITH_MORTGAGE",
                        "RENTER",
                    ],
                }
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([10000, 20000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100000, 200000, 200001], dtype="int64")}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 200.0]), WeightKind.CALIBRATED)},
    )


def test_load_l0_refit_npz_validates_full_candidate_vector(tmp_path: Path) -> None:
    path = tmp_path / "weights.npz"
    metadata = {"candidate_records": 3, "n_selected": 2, "weight_entity": "household"}
    np.savez_compressed(
        path,
        weights=np.asarray([0.0, 4.0, 5.0]),
        metadata_json=json.dumps(metadata),
    )

    solution = load_l0_refit_npz(path, expected_candidate_records=3)

    assert solution.selected_mask.tolist() == [False, True, True]
    np.testing.assert_allclose(solution.selected_weights, np.asarray([4.0, 5.0]))
    assert solution.metadata == metadata


def test_load_l0_refit_npz_rejects_metadata_candidate_count_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.npz"
    np.savez_compressed(
        path,
        weights=np.asarray([0.0, 4.0, 5.0]),
        metadata_json=json.dumps({"candidate_records": 4, "n_selected": 2}),
    )

    with pytest.raises(ValueError, match="candidate_records"):
        load_l0_refit_npz(path, expected_candidate_records=3)


def test_attach_l0_refit_weights_subsets_clean_base_support() -> None:
    frame = _us_frame()
    solution = l0_refit_export.L0RefitWeights(
        weights=np.asarray([0.0, 333.0]),
        selected_mask=np.asarray([False, True]),
        metadata={},
    )

    exported = attach_l0_refit_weights(frame, solution)

    assert exported.table("household")["household_id"].to_list() == [20]
    assert exported.table("person")["person_id"].to_list() == [2, 3]
    assert exported.table("tax_unit")["tax_unit_id"].to_list() == [200, 201]
    assert exported.table("spm_unit")["spm_unit_id"].to_list() == [2000]
    np.testing.assert_allclose(
        exported.weights_for("household").values,
        np.asarray([333.0]),
    )
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED


def test_attach_l0_refit_entity_weights_rejects_misaligned_weights() -> None:
    frame = _us_frame()

    with pytest.raises(ValueError, match="same shape"):
        attach_l0_refit_entity_weights(
            frame,
            weight_entity="household",
            selected_entity_ids=np.asarray([10, 20]),
            selected_weights=np.asarray([333.0]),
            reason="test",
        )


def test_required_us_release_source_columns_rejects_missing_source_stage() -> None:
    frame = _us_frame()
    raw_tax_units = frame.table("tax_unit").drop(
        columns=[
            "takes_up_aca_if_eligible",
            "selected_marketplace_plan_benchmark_ratio",
        ]
    )
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "tax_unit": raw_tax_units,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match="takes_up_aca_if_eligible: missing"):
        assert_required_us_release_source_columns(raw_frame)


def test_export_us_l0_refit_h5_uses_existing_policyengine_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _us_frame()
    npz = tmp_path / "weights.npz"
    np.savez_compressed(
        npz,
        weights=np.asarray([0.0, 333.0]),
        metadata_json=json.dumps({"candidate_records": 2, "n_selected": 1}),
    )
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    attrs_h5 = tmp_path / "attrs.h5"
    attrs_h5.write_text("attrs")
    output = tmp_path / "populace_us_2024.h5"
    copied_from: list[Path] = []

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "copy_populace_root_attrs",
        lambda source, destination: (
            copied_from.append(Path(source)) or ("populace_test_attr",)
        ),
    )

    class FakeEngine:
        def write_dataset(self, bundle, path, period):
            assert Path(path) == output
            assert period == 2024
            assert bundle.table("household")["household_id"].to_list() == [20]
            np.testing.assert_allclose(
                bundle.weights_for("household").values,
                np.asarray([333.0]),
            )
            Path(path).write_text("sentinel")

    monkeypatch.setattr(l0_refit_export, "PolicyEngineUSEngine", FakeEngine)

    summary = export_us_l0_refit_h5(
        base_h5=base_h5,
        weights_npz=npz,
        output_h5=output,
        root_attrs_h5=attrs_h5,
        # The two-household fixture reweights its single kept household from
        # 200 to 333, a +66% drift on every kept column.
        input_mass_relative_tolerance=1.0,
        # A single kept household cannot satisfy national NYC-share bounds.
        require_geography_ladder=False,
    )

    assert output.read_text() == "sentinel"
    assert copied_from == [attrs_h5]
    manifest_path = output.with_suffix(".l0_refit_export_summary.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "us_l0_refit_h5_export"
    assert manifest["summary_json_path"] == str(manifest_path)
    assert manifest["required_source_columns_checked"] is True
    assert manifest["input_mass_parity_enforced"] is True
    assert manifest["input_mass_parity"]["passed"] is True
    assert manifest["input_mass_reference_h5"] is None
    assert manifest["base_h5"]["sha256"] == sha256(b"base").hexdigest()
    assert manifest["root_attrs_h5"]["sha256"] == sha256(b"attrs").hexdigest()
    assert manifest["weights_npz"]["sha256"] == sha256(npz.read_bytes()).hexdigest()
    assert manifest["output_h5"]["sha256"] == sha256(b"sentinel").hexdigest()
    assert summary["candidate_households"] == 2
    assert summary["selected_households"] == 1
    assert summary["selected_weight_sum"] == pytest.approx(333.0)
    assert summary["copied_root_attrs"] == ["populace_test_attr"]
    assert (
        summary["summary_json"]["sha256"]
        == sha256(manifest_path.read_bytes()).hexdigest()
    )


def _l0_refit_npz(tmp_path: Path) -> Path:
    npz = tmp_path / "weights.npz"
    np.savez_compressed(
        npz,
        weights=np.asarray([0.0, 333.0]),
        metadata_json=json.dumps({"candidate_records": 2, "n_selected": 1}),
    )
    return npz


def test_export_us_l0_refit_h5_fails_when_selection_zeroes_input_mass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The IRA mass lives entirely in household 10, which the saved weights
    # deselect — the issue #278 failure shape.
    frame = _us_frame(traditional_ira_contributions=[700.0, 0.0, 0.0])
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    output = tmp_path / "populace_us_2024.h5"

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "PolicyEngineUSEngine",
        lambda: pytest.fail("a gated export must not reach the writer"),
    )

    with pytest.raises(ValueError, match="traditional_ira_contributions"):
        export_us_l0_refit_h5(
            base_h5=base_h5,
            weights_npz=_l0_refit_npz(tmp_path),
            output_h5=output,
            input_mass_relative_tolerance=1.0,
            require_geography_ladder=False,
        )
    assert not output.exists()


def test_export_us_l0_refit_h5_records_input_mass_drift_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _us_frame(traditional_ira_contributions=[700.0, 0.0, 0.0])
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    output = tmp_path / "populace_us_2024.h5"

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "copy_populace_root_attrs",
        lambda source, destination: (),
    )

    class FakeEngine:
        def write_dataset(self, bundle, path, period):
            Path(path).write_text("sentinel")

    monkeypatch.setattr(l0_refit_export, "PolicyEngineUSEngine", FakeEngine)

    summary = export_us_l0_refit_h5(
        base_h5=base_h5,
        weights_npz=_l0_refit_npz(tmp_path),
        output_h5=output,
        input_mass_relative_tolerance=1.0,
        require_geography_ladder=False,
        require_input_mass_parity=False,
    )

    assert output.read_text() == "sentinel"
    assert summary["input_mass_parity_enforced"] is False
    assert summary["input_mass_parity"]["passed"] is False
    assert any(
        "traditional_ira_contributions" in failure
        for failure in summary["input_mass_parity"]["failures"]
    )


def test_export_us_l0_refit_h5_gates_against_external_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The base frame never carried the IRA input the certified reference
    # release populates — the wrong-base variant of issue #278. Comparing
    # export-vs-base alone would pass vacuously; the reference catches it.
    base_frame = _us_frame()
    reference_frame = _us_frame(traditional_ira_contributions=[700.0, 0.0, 0.0])
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    reference_h5 = tmp_path / "reference.h5"
    reference_h5.write_text("reference")
    output = tmp_path / "populace_us_2024.h5"

    monkeypatch.setattr(
        l0_refit_export,
        "load_us_frame",
        lambda path: reference_frame if Path(path) == reference_h5 else base_frame,
    )
    monkeypatch.setattr(
        l0_refit_export,
        "PolicyEngineUSEngine",
        lambda: pytest.fail("a gated export must not reach the writer"),
    )

    with pytest.raises(ValueError, match="absent from l0_refit_export"):
        export_us_l0_refit_h5(
            base_h5=base_h5,
            weights_npz=_l0_refit_npz(tmp_path),
            output_h5=output,
            reference_h5=reference_h5,
            input_mass_relative_tolerance=1.0,
            require_geography_ladder=False,
        )
    assert not output.exists()


def test_required_us_release_source_columns_rejects_missing_geography_spine() -> None:
    frame = _us_frame()
    raw_households = frame.table("household").drop(columns=["county_fips"])
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "household": raw_households,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match="household.county_fips: missing"):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize(
    "column",
    [
        "auto_loan_balance",
        "auto_loan_interest",
        "qualified_passenger_vehicle_loan_interest",
        "net_worth",
        "household_vehicles_owned",
        "household_vehicles_value",
    ],
)
def test_required_us_release_source_columns_rejects_missing_auto_input(
    column: str,
) -> None:
    frame = _us_frame()
    raw_households = frame.table("household").drop(columns=[column])
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "household": raw_households,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match=rf"household.{column}: missing"):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_rejects_constant_auto_input() -> None:
    frame = _us_frame()
    raw_households = frame.table("household").copy()
    raw_households["qualified_passenger_vehicle_loan_interest"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "household": raw_households,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="household.qualified_passenger_vehicle_loan_interest: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_rejects_constant_net_worth() -> None:
    frame = _us_frame()
    raw_households = frame.table("household").copy()
    raw_households["net_worth"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "household": raw_households,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="household.net_worth: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize(
    "column",
    ["household_vehicles_owned", "household_vehicles_value"],
)
def test_required_us_release_source_columns_rejects_constant_vehicle_input(
    column: str,
) -> None:
    frame = _us_frame()
    raw_households = frame.table("household").copy()
    raw_households[column] = 0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "household": raw_households,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match=rf"household.{column}: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_casualty_loss_signal() -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["casualty_loss"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="person.casualty_loss: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_form_4952_signal() -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["investment_income_elected_form_4952"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="person.investment_income_elected_form_4952: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize(
    "column",
    [
        "taxable_401k_distributions",
        "taxable_403b_distributions",
        "tax_exempt_ira_distributions",
        "taxable_ira_distributions",
        "keogh_distributions",
        "taxable_sep_distributions",
    ],
)
def test_required_us_release_source_columns_enforces_retirement_distribution_signal(
    column: str,
) -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people[column] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match=rf"person.{column}: not nonconstant"):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_domestic_production_signal() -> (
    None
):
    frame = _us_frame()
    raw_tax_units = frame.table("tax_unit").copy()
    raw_tax_units["domestic_production_ald"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "tax_unit": raw_tax_units,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="tax_unit.domestic_production_ald: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize("column", ["alimony_income", "alimony_expense"])
def test_required_us_release_source_columns_enforces_alimony_signal(
    column: str,
) -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people[column] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match=rf"person.{column}: not nonconstant"):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize(
    "column",
    ["child_support_received", "child_support_expense"],
)
def test_required_us_release_source_columns_enforces_child_support_signal(
    column: str,
) -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people[column] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match=rf"person.{column}: not nonconstant"):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_disability_benefits_signal() -> (
    None
):
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["disability_benefits"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="person.disability_benefits: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_educator_expense_signal() -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["educator_expense"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="person.educator_expense: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_other_premium_signal() -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["other_health_insurance_premiums"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="person.other_health_insurance_premiums: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


@pytest.mark.parametrize(
    "column",
    ["farm_operations_income", "farm_rent_income"],
)
def test_required_us_release_source_columns_enforces_farm_business_signal(
    column: str,
) -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people[column] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match=rf"person.{column}: not nonconstant"):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_misc_itemized_signal() -> None:
    frame = _us_frame()
    raw_people = frame.table("person").copy()
    raw_people["unreimbursed_business_employee_expenses"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "person": raw_people,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match=("person.unreimbursed_business_employee_expenses: not nonconstant"),
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_childcare_signal() -> None:
    frame = _us_frame()
    raw_spm_units = frame.table("spm_unit").copy()
    raw_spm_units["spm_unit_pre_subsidy_childcare_expenses"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "spm_unit": raw_spm_units,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match=("spm_unit.spm_unit_pre_subsidy_childcare_expenses: not nonconstant"),
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_required_us_release_source_columns_enforces_energy_subsidy_signal() -> None:
    frame = _us_frame()
    raw_spm_units = frame.table("spm_unit").copy()
    raw_spm_units["spm_unit_energy_subsidy"] = 0.0
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "spm_unit": raw_spm_units,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError,
        match="spm_unit.spm_unit_energy_subsidy: not nonconstant",
    ):
        assert_required_us_release_source_columns(raw_frame)


def test_export_us_l0_refit_h5_fails_geography_ladder_gate_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A single kept household concentrates all weight outside the national
    # NYC-share bounds — the gate must stop the export before the writer.
    frame = _us_frame()
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    output = tmp_path / "populace_us_2024.h5"

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "PolicyEngineUSEngine",
        lambda: pytest.fail("a gated export must not reach the writer"),
    )

    with pytest.raises(ValueError, match="geography-ladder gate"):
        export_us_l0_refit_h5(
            base_h5=base_h5,
            weights_npz=_l0_refit_npz(tmp_path),
            output_h5=output,
            input_mass_relative_tolerance=1.0,
        )
    assert not output.exists()


def test_export_us_l0_refit_h5_records_geography_ladder_gate_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _us_frame()
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    output = tmp_path / "populace_us_2024.h5"

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "copy_populace_root_attrs",
        lambda source, destination: (),
    )

    class FakeEngine:
        def write_dataset(self, bundle, path, period):
            Path(path).write_text("sentinel")

    monkeypatch.setattr(l0_refit_export, "PolicyEngineUSEngine", FakeEngine)

    summary = export_us_l0_refit_h5(
        base_h5=base_h5,
        weights_npz=_l0_refit_npz(tmp_path),
        output_h5=output,
        input_mass_relative_tolerance=1.0,
        require_geography_ladder=False,
    )

    assert output.read_text() == "sentinel"
    assert summary["geography_ladder_gate_enforced"] is False
    assert summary["geography_ladder_gate"]["passed"] is False
    assert summary["required_household_source_columns"][0] == "state_fips"
    assert "alimony_income" in summary["required_person_source_columns"]
    assert "alimony_expense" in summary["required_person_source_columns"]
    assert "casualty_loss" in summary["required_person_source_columns"]
    assert "child_support_received" in summary["required_person_source_columns"]
    assert "child_support_expense" in summary["required_person_source_columns"]
    assert "disability_benefits" in summary["required_person_source_columns"]
    assert "educator_expense" in summary["required_person_source_columns"]
    assert (
        "investment_income_elected_form_4952"
        in summary["required_person_source_columns"]
    )
    assert (
        "other_health_insurance_premiums" in summary["required_person_source_columns"]
    )
    assert "farm_operations_income" in summary["required_person_source_columns"]
    assert "farm_rent_income" in summary["required_person_source_columns"]
    assert "keogh_distributions" in summary["required_person_source_columns"]
    assert "business_is_sstb" in summary["required_person_source_columns"]
    assert "qualified_reit_and_ptp_income" in summary["required_person_source_columns"]
    assert "domestic_production_ald" in summary["required_source_columns"]
    assert (
        "unreimbursed_business_employee_expenses"
        in summary["required_person_source_columns"]
    )
    assert summary["required_spm_unit_source_columns"] == [
        "spm_unit_pre_subsidy_childcare_expenses",
        "spm_unit_energy_subsidy",
        "receives_housing_assistance",
        "spm_unit_tenure_type",
    ]
    assert summary["required_household_nonconstant_source_columns"] == [
        "auto_loan_balance",
        "auto_loan_interest",
        "qualified_passenger_vehicle_loan_interest",
        "net_worth",
        "household_vehicles_owned",
        "household_vehicles_value",
        "tenure_type",
    ]
