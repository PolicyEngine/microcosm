"""Tests split from packages/microcosm-build/tests/test_us_workers_compensation.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_workers_compensation import *


def test_all_sha_locked_asec_sources_have_exact_wc_val_signal() -> None:
    from policyengine_us.data import USSingleYearDataset

    expected = {
        "census_cps_2022.h5": {
            "sha256": "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            "positive": 328,
            "raw_sum": 3_821_784.0,
            "maximum": 99_999.0,
            "weighted_share": 0.0021741514,
            "weighted_total": 7_785_631_296.0,
        },
        "census_cps_2023.h5": {
            "sha256": "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            "positive": 379,
            "raw_sum": 4_710_504.0,
            "maximum": 99_999.0,
            "weighted_share": 0.0026142319,
            "weighted_total": 10_850_596_150.0,
        },
        "census_cps_2024.h5": {
            "sha256": "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            "positive": 391,
            "raw_sum": 4_160_758.0,
            "maximum": 72_000.0,
            "weighted_share": 0.0028173010,
            "weighted_total": 9_146_153_896.0,
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
        values = pd.to_numeric(person["WC_VAL"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        weights = (
            pd.to_numeric(person["A_FNLWGT"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            / 100.0
        )
        positive = values > 0.0
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert np.array_equal(values, np.floor(values))
        assert int(np.count_nonzero(positive)) == facts["positive"]
        assert float(values.sum()) == facts["raw_sum"]
        assert float(values.max()) == facts["maximum"]
        assert float(weights[positive].sum() / weights.sum()) == pytest.approx(
            facts["weighted_share"], abs=1e-9
        )
        assert float((values * weights).sum()) == pytest.approx(
            facts["weighted_total"], rel=1e-8
        )


def test_policyengine_us_2_2_1_contract_and_positive_annual_behavior() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                # SNAP 1.769.0+ applies this person's countable-income share
                # to unearned income; make the graph fixture work-eligible.
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    simulation = Simulation(situation=situation)

    assert simulation.calculate(_OUTPUT, 2024)[0] == pytest.approx(6_000.0)
    assert simulation.calculate(_OUTPUT, "2024-01")[0] == pytest.approx(500.0)
    assert simulation.calculate("snap_unearned_income", "2024-01")[0] == pytest.approx(
        500.0
    )


def test_shipped_snap_exclusion_probe_binds_with_positive_sign() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "workers_compensation_snap_exclusion"
    )
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 12_000.0},
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect > 1_000.0
