"""Tests split from packages/microcosm-build/tests/test_us_medicare_take_up.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_medicare_take_up import *


def test_all_sha_locked_asec_sources_have_exact_measured_signal() -> None:
    from policyengine_us.data import USSingleYearDataset

    expected = {
        "census_cps_2022.h5": {
            "sha256": "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            "positive": 26495,
            "weighted_share": 0.18617172806991097,
        },
        "census_cps_2023.h5": {
            "sha256": "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            "positive": 26466,
            "weighted_share": 0.187909801755554,
        },
        "census_cps_2024.h5": {
            "sha256": "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            "positive": 26448,
            "weighted_share": 0.19082998028041492,
        },
    }
    summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in summary["base_source"]["sources"]
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("SHA-locked ASEC artifacts are not mounted")

    assert set(paths) == set(expected)
    for filename, facts in expected.items():
        path = paths[filename]
        assert _sha256(path) == facts["sha256"]
        person = USSingleYearDataset(file_path=str(path)).person
        codes = pd.to_numeric(person[_SOURCE], errors="coerce")
        weights = pd.to_numeric(person["A_FNLWGT"], errors="coerce") / 100.0
        enrolled = codes == 1
        assert set(codes.unique()) == {0, 1, 2}
        assert int(enrolled.sum()) == facts["positive"]
        assert float(weights[enrolled].sum() / weights.sum()) == pytest.approx(
            facts["weighted_share"]
        )
        derived = _derive(person)
        np.testing.assert_array_equal(derived[_OUTPUT].to_numpy(), enrolled.to_numpy())


def test_policyengine_contract_and_live_neutralization() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    variable = system.variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert bool(variable.default_value) is True
    assert str(variable.definition_period).lower() == "year"
    assert system.variables["medicare_enrolled"].adds == [_OUTPUT]

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 70},
                _OUTPUT: {"2024": True},
            }
        },
        "tax_units": {"tax_unit": {"members": ["adult"]}},
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
        "marital_units": {"marital_unit": {"members": ["adult"]}},
    }

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "medicare_take_up_neutralization"
    )
    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))
    assert baseline.calculate("medicare_enrolled", 2024)[0]
    assert not neutralized.calculate("medicare_enrolled", 2024)[0]
    assert baseline.calculate("medicare_cost", 2024)[0] > 0
    assert neutralized.calculate("medicare_cost", 2024)[0] == 0
