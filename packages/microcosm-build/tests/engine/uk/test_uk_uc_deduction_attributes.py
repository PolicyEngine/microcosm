"""Tests split from packages/microcosm-build/tests/test_uk_uc_deduction_attributes.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_deduction_attributes import *


def test_resource_values_lockstep_with_engine_parameter_tree() -> None:
    import importlib

    system_module = importlib.import_module("policyengine_uk.system")
    latent_module = pytest.importorskip(
        "policyengine_uk.variables.gov.dwp.universal_credit.deductions."
        "uc_latent_deduction_rate"
    )
    region_module = pytest.importorskip(
        "policyengine_uk.variables.household.demographic.geography"
    )
    resource = load_uc_deduction_distributions()
    parameters = system_module.system.parameters.gov.simulation.uc_deductions
    bands = resource["latent_rate_distribution"]["bands"]

    np.testing.assert_array_equal(
        np.asarray([row["lower"] for row in bands]), latent_module.BAND_LOWER
    )
    np.testing.assert_array_equal(
        np.asarray([row["upper"] for row in bands]), latent_module.BAND_UPPER
    )
    assert [
        float(getattr(parameters.latent_rate_distribution, row["name"])("2024"))
        for row in bands
    ] == [row["share"] for row in bands]
    assert (
        float(parameters.calibration_cap("2024"))
        == (resource["latent_rate_distribution"]["calibration_cap"]["value"])
    )
    combinations = resource["type_combination"]["shares"]
    assert [
        float(getattr(parameters.type_combination, row["name"])("2024"))
        for row in combinations
    ] == [row["share"] for row in combinations]
    factors = resource["region_incidence_factor"]["factors"]
    assert {
        name: float(getattr(parameters.region_incidence_factor, name)("2024"))
        for name in factors
    } == factors
    assert set(region_module.Region.__members__) == UC_DEDUCTION_REGIONS


def test_engine_golden_mirror_on_held_float32_draws() -> None:
    import importlib

    policyengine_uk = importlib.import_module("policyengine_uk")
    resource = load_uc_deduction_distributions()
    # Every real region factor is exercised against the engine: 1,024 units
    # cycle through the twelve UK regions (~85 each). UNKNOWN is left to the
    # hermetic mapping test: the engine cannot uprate rents for a household
    # without a region (no private_rental_prices.UNKNOWN parameter), so it
    # cannot host a simulation, while its factor of 1.0 is exercised in the
    # stage's own mapping.
    engine_regions = tuple(sorted(UC_DEDUCTION_REGIONS - {"UNKNOWN"}))
    assigned = UKUCDeductionAttributesStageTransform(stage=_stage(), resource=resource)(
        _frame(1024, region_names=engine_regions)
    )
    assigned_benunit = assigned.table("benunit")
    assert set(
        assigned.table("household")["region"].map(lambda v: str(getattr(v, "name", v)))
    ) == set(engine_regions)
    fallback_benunit = assigned_benunit.drop(
        columns=["uc_latent_deduction_rate", "uc_deduction_combination"]
    )
    fallback = uk_national_frame(
        person=assigned.table("person").copy(),
        benunit=fallback_benunit,
        household=assigned.table("household").copy(),
        household_weights=assigned.weights_for("household").values,
        time_period="2024",
    )

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    adapter = PolicyEngineUKEngine()
    simulation = policyengine_uk.Microsimulation(
        dataset=adapter._build_dataset(fallback, 2024)
    )
    engine_has = np.asarray(simulation.calculate("uc_has_deduction", 2024))
    engine_rate = np.asarray(simulation.calculate("uc_latent_deduction_rate", 2024))
    engine_combination = (
        simulation.calculate("uc_deduction_combination", 2024)
        .astype(str)
        .to_numpy(dtype=object)
    )

    expected_rate = assigned_benunit["uc_latent_deduction_rate"].to_numpy()
    np.testing.assert_array_equal(engine_has, expected_rate > 0.0)
    np.testing.assert_allclose(engine_rate, expected_rate, atol=1e-6)
    np.testing.assert_array_equal(
        engine_combination,
        assigned_benunit["uc_deduction_combination"].to_numpy(),
    )
