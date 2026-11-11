"""Tests split from packages/microcosm-build/tests/test_uk_uc_matrix_diagnostics.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_matrix_diagnostics import *


def test_complete_joint_export_on_released_uk_engine(diagnostic_tool, tmp_path):
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 20],
            "person_household_id": [100, 100, 200, 200],
            "age": [35, 8, 40, 38],
            "is_benunit_head": [True, False, True, False],
            "is_parent": [True, False, False, False],
            "is_uc_claimant": [True, False, True, True],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20],
            "benunit_source_id": [10, 20],
            "dependent_children": [1, 0],
            "is_married": [False, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_source_id": [100, 200],
            "region": ["WALES", "NORTHERN_IRELAND"],
            "council_tax": [1200.0, 0.0],
            "rent": [6000.0, 6000.0],
            "tenure_type": ["RENT_PRIVATELY", "RENT_PRIVATELY"],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        household_weights=np.ones(2),
        weight_kind=WeightKind.IMPORTANCE,
    )
    resolver = diagnostic_tool.UKMeasureResolver(
        simulation_source=None, frame=frame, year=2025, scratch_dir=tmp_path / "engine"
    )
    diagnostic_tool.write_uc_support_diagnostics(
        frame,
        resolver,
        {"prior": np.ones(2), "calibrated": np.array([2.0, 3.0])},
        tmp_path,
    )
    report = json.loads((tmp_path / "uc_joint_diagnostics.json").read_text())
    assert len(report["rows"]) == 39
    assert report["unavailable_optional_components"] == [
        "uc_net_earned_income",
        "uc_capital_income",
    ]
    assert "uc_tariff_income" in report["available_component_variables"]
    private = pd.read_pickle(tmp_path / "benefit_units_private.pkl")
    assert private["great_britain"].tolist() == [True, False]
    assert len(json.loads((tmp_path / "protected_outcomes.json").read_text())) == 6
