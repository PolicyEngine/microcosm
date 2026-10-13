"""Tests split from packages/microcosm-build/tests/test_us_prior_year_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_prior_year_income import *


def test_policyengine_17646_input_and_dependency_contract() -> None:
    engine = PolicyEngineUSEngine()
    assert set(US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS) <= set(engine.variables())
    assert engine.formula_owned_outputs(US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS) == {
        "employment_income_last_year"
    }
    system = engine._tax_benefit_system()
    assert system.variables["earned_income_last_year"].adds == [
        "employment_income_last_year",
        "self_employment_income_last_year",
    ]
    assert system.variables["tax_unit_earned_income_last_year"].entity.key == "tax_unit"

def test_export_ready_support_persists_inputs_and_excludes_formula_owned_wage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module, "QRF", _QRF)
    direct = with_us_prior_year_income_inputs(_source_frame(), seed=0, time_period=2024)
    result = with_us_prior_year_income_inputs(
        clone_us_frame_for_puf_support(direct), seed=0, time_period=2024
    )
    output = tmp_path / "prior_year_income.h5"

    PolicyEngineUSEngine().write_dataset(result, output, period=2024)

    with pd.HDFStore(output, mode="r") as store:
        person = store["person"]
    assert "self_employment_income_last_year" in person
    assert "previous_year_income_available" in person
    assert "employment_income_last_year" not in person
