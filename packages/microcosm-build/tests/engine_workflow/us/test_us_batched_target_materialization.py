"""Tests split from packages/microcosm-build/tests/test_us_batched_target_materialization.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_batched_target_materialization import *


def test_real_engine_batched_base_simulation_matches_unbatched(monkeypatch) -> None:
    """Compare these household-local targets at batch sizes 1, 2 and unbatched."""

    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())
    frame = _real_engine_frame()
    targets = (
        _soi("agi_amount", "adjusted_gross_income"),
        _soi(
            "agi_under_50k_count",
            "count",
            agi_lower_bound="-inf",
            agi_upper_bound="50000",
        ),
        _soi("single_returns", "count", filing_status="Single"),
        _soi("wages", "employment_income"),
        _soi("filer_individuals", "tax_filer_individual_count"),
        _population_age("pop_18_to_65", 18, 65),
        _population_age("ca_pop_65_plus", 65, "inf", state_fips="06"),
        _variable("snap_amount", base_variable="snap"),
        _variable("ssi_recipients", base_variable="ssi", measure_mode="indicator_sum"),
        _spec("ca_state_income_tax", "state_income_tax", {"state_fips": "06"}),
        _spec("md_state_income_tax", "state_income_tax", {"state_fips": "24"}),
        _spec("eitc", "fixture"),
    )

    results = {}
    for batch_size in (None, 2, 1):
        target_frame, registry, compilation = builder._materialize_target_frame(
            frame,
            targets,
            maximum_microsim_batch_size=batch_size,
        )
        assert compilation["dropped_target_names"] == []
        results[batch_size] = (target_frame.table("household"), registry.version)

    unbatched, unbatched_version = results[None]
    # The fixture reaches real engine output: taxes, credits and transfers.
    for column in (
        "income_tax",
        "eitc",
        "snap_amount",
        "ssi_recipients",
        "ca_state_income_tax",
        "md_state_income_tax",
    ):
        assert unbatched[column].to_numpy().any(), column
    for batch_size in (2, 1):
        household, version = results[batch_size]
        pd.testing.assert_frame_equal(household, unbatched, check_exact=True)
        assert version == unbatched_version


def test_real_engine_base_batches_build_no_system_and_release_engines(
    monkeypatch,
) -> None:
    """Check variable-file loads and live engines across the base batch loop."""

    import gc

    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    variable_file_loads = [0]
    load_variable_file = TaxBenefitSystem.add_variables_from_file

    def counting_load(self, file_path):
        variable_file_loads[0] += 1
        return load_variable_file(self, file_path)

    monkeypatch.setattr(TaxBenefitSystem, "add_variables_from_file", counting_load)

    def alive_microsimulations() -> int:
        gc.collect()
        return sum(1 for obj in gc.get_objects() if isinstance(obj, Microsimulation))

    frame = _real_engine_frame()
    targets = (
        _soi("agi_amount", "adjusted_gross_income"),
        _variable("snap_amount", base_variable="snap"),
    )
    system = CountryTaxBenefitSystem()
    # Sanity: the counter sees a real build.
    assert variable_file_loads[0] > 1_000
    unbatched, _ = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=None,
    )
    alive_before = alive_microsimulations()
    loads_before_batches = variable_file_loads[0]
    batched, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=1,
    )

    assert receipt["batches"] == frame.n("household")
    assert variable_file_loads[0] == loads_before_batches, (
        f"{receipt['batches']} base batches loaded "
        f"{variable_file_loads[0] - loads_before_batches} variable files: a "
        "tax-benefit system was built per batch (microcosm#456)"
    )
    assert alive_microsimulations() <= alive_before
    assert list(batched) == list(unbatched)
    for column, values in unbatched.items():
        np.testing.assert_array_equal(batched[column], values)


def test_real_engine_refuses_a_batched_medicaid_cost_target() -> None:
    """Compare guarded materialization with unguarded one-household engines."""

    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    builder = _load_builder_module()
    frame = _medicaid_frame()
    targets = (_variable("medicaid_cost_total", base_variable="medicaid_cost"),)
    system = CountryTaxBenefitSystem()
    whole, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=None,
    )
    assert receipt["batches"] == 1
    assert receipt["population_aggregate_variables_checked"] == []
    pool_cost = whole["medicaid_cost_total"]
    single_batch, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=frame.n("household"),
    )
    np.testing.assert_array_equal(single_batch["medicaid_cost_total"], pool_cost)
    assert receipt["batches"] == 1

    unguarded_cost = []
    enrolled = 0
    aggregate_periods = []
    for position in range(frame.n("household")):
        batch_frame = builder.select_households_by_position(
            frame, np.asarray([position], dtype=np.int64)
        )
        simulation = Microsimulation(
            dataset=builder._dataset_from_frame(
                batch_frame, assert_no_formula_owned_columns=False
            )
        )
        try:
            batch_columns = builder._base_simulation_household_columns(
                batch_frame, targets, simulation=simulation, system=system
            )
            unguarded_cost.extend(batch_columns["medicaid_cost_total"])
            enrolled += int(
                np.asarray(
                    simulation.calculate("medicaid_enrolled", builder.PERIOD)
                ).sum()
            )
            aggregate_periods.append(
                builder._engine_known_periods(
                    simulation, builder.US_POPULATION_AGGREGATE_VARIABLES
                )
            )
        finally:
            builder.release_engine_simulation(simulation)
    unguarded_cost = np.asarray(unguarded_cost)
    weights = frame.weights_for("household").values
    assert enrolled > 0
    assert aggregate_periods[0] == set()
    assert (
        "medicaid_slcsp_state_denominator",
        str(builder.PERIOD),
    ) in aggregate_periods[1]
    assert pool_cost[0] == 0
    assert np.all(pool_cost[1:] > 0)
    assert np.any(unguarded_cost != pool_cost)
    assert np.dot(unguarded_cost, weights) > np.dot(pool_cost, weights)
    print(
        f"Medicaid control: enrollees={enrolled}, "
        f"whole_pool_weighted={np.dot(pool_cost, weights):.9g}, "
        f"unguarded_batch_size_1_weighted={np.dot(unguarded_cost, weights):.9g}, "
        f"changed_households={np.count_nonzero(unguarded_cost != pool_cost)}"
    )

    for batch_size in (1, 2):
        first_aggregate_batch = 2 if batch_size == 1 else 1
        with pytest.raises(
            ValueError,
            match=(
                rf"household batch {first_aggregate_batch}/"
                rf"{frame.n('household') // batch_size} computed .*"
                rf"medicaid_slcsp_state_denominator@{builder.PERIOD}"
            ),
        ):
            builder._materialize_base_simulation_columns(
                frame,
                targets,
                system=system,
                microsimulation_cls=Microsimulation,
                maximum_microsim_batch_size=batch_size,
            )


def test_population_aggregate_list_matches_installed_engine_sources() -> None:
    """Pin the guard list to weight reads and aggregation markers in US sources.

    The scan covers the package except tests, including module helpers beside
    Variable classes and in tools. It is not a proof of household locality.
    """
    builder = _load_builder_module()
    source_digests = {}
    found = _engine_population_aggregate_sources(source_digests=source_digests)
    triaged = {name: {"np.isin("} for name in _CONSTANT_MEMBERSHIP_FORMULAS}
    # These search sorted parameter brackets or literal earnings thresholds.
    triaged.update(
        {
            name: {"np.searchsorted("}
            for name in (
                "aca_required_contribution_percentage",
                "ca_premium_subsidy_applicable_percentage",
                "md_premium_assistance_target_contribution_percentage",
                "nm_premium_assistance_target_contribution_percentage",
                "substitution_elasticity",
            )
        }
    )
    triaged.update(
        {
            # np.unique groups the loop by indexing year; each person still
            # reads their own earnings and the year's common wage index.
            "ss_aime": {"_compute_aime()"},
            # The helper groups a packaged county schedule by effective year,
            # then each household joins its own county and bedroom count.
            "hud_utility_allowance": {"utility_allowance_schedule()"},
            # A lexical collision: ndarray.reshape matches the converter's
            # module helper named reshape. Actual sums use axis=1 on brackets.
            "ny_supplemental_tax": {"get_previous_threshold()"},
            # These cross-person joins are permitted only after the frame
            # precondition proves positive claiming IDs stay in the household.
            "medicaid_has_known_claiming_tax_unit": {"np.isin("},
            "medicaid_household_income": {
                "medicaid_claiming_tax_unit_value()",
                "medicaid_external_claimed_sum()",
            },
            "medicaid_household_size": {
                "medicaid_claiming_tax_unit_value()",
                "medicaid_external_claimed_sum()",
            },
        }
    )
    assert set(found) == set(builder.US_POPULATION_AGGREGATE_VARIABLES) | set(
        triaged
    ), found
    for name, evidence in triaged.items():
        assert set(found[name]) == evidence, (name, found[name])
    # Pin the reviewed exceptions' class and reachable helper source bodies;
    # adding another aggregate to an allowed name still requires a fresh review.
    triaged_digest = hashlib.sha256(
        "\n".join(f"{name}:{source_digests[name]}" for name in sorted(triaged)).encode()
    ).hexdigest()
    assert triaged_digest == (
        "f26eb560e474207fc1fa1d8828b62ba4791fb763089ea2ce7a85b57a259f2755"
    ), triaged_digest


def test_forward_reads_match_the_reviewed_engine_sources() -> None:
    """Pin the formulas that may read a later period than the one asked.

    In-build simulations hold PERIOD alone, and the adapter holds the read
    year alone, so a formula that read a later year's stored data would get
    nothing there. The scan marks forward period offsets and holder lookups in
    US Variable classes and the helpers they call. Each one found is reviewed
    here and its source is pinned, so an engine that adds or changes one fails
    until it is reviewed. It is a tripwire, not a proof.
    """
    source_digests = {}
    found = _engine_forward_read_sources(source_digests=source_digests)
    reviewed = {
        # The offset dates a parameter lookup, not a data read.
        "sc_cdcc_potential": {".offset(max_decoupled_year_offset"},
        "md_dependent_care_subtraction": {".offset(max_decoupled_year_offset"},
        # The twelve months of the year asked.
        "tx_ceap_eligible": {".offset(month_offset"},
        # The holder is read only at the period asked.
        "has_tin": {"get_holder(", "get_known_periods(", "get_array("},
        # The holder gives the filing-status enum, not a stored value.
        "gross_medicare_part_b_premium": {"get_holder("},
        # The holder sets an input at the period asked, on a branch.
        "tob_revenue_oasdi": {"get_holder("},
        "tob_revenue_medicare_hi": {"get_holder("},
        # Over a dataset, county returns its last stored year's array when the
        # year asked holds none. A dataset that holds county holds it at the
        # year the build reads, so that read never reaches the lookup.
        "county": {"get_holder(", "get_known_periods(", "get_array("},
    }
    assert {name: set(evidence) for name, evidence in found.items()} == reviewed
    reviewed_digest = hashlib.sha256(
        "\n".join(
            f"{name}:{source_digests[name]}" for name in sorted(reviewed)
        ).encode()
    ).hexdigest()
    assert reviewed_digest == (
        "4c3a61a320e3d7d57f887ba3a364293d59b9cc363e2e38e7aecd8fac13409d5e"
    ), reviewed_digest


def test_core_fills_a_missing_period_only_from_earlier_ones() -> None:
    """policyengine-core fills an input missing at the period asked from the
    latest earlier stored period (uprated), or carries the last stored period
    over only when it is not later than the period asked; a later one gives
    the default. So a dataset cut at the year read and the engine's projection
    agree at that year. A new digest means core changed the rule (in
    ``Simulation._calculate``): re-check the cut, then re-pin."""
    import inspect

    from policyengine_core.simulations import Simulation

    source = inspect.getsource(Simulation._calculate)
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "3a36c798f58eced1d61812cd374997ac1732fa1a1beaff74ba6b7e6588aa3269"
    )


def test_in_build_datasets_hold_period_alone_and_match_the_projection(
    monkeypatch,
) -> None:
    """In-build simulations read only PERIOD, so they ask for datasets that
    stop there; by default a dataset keeps policyengine-us's projection.

    Values must match policyengine-us's own projection to later years: the
    target frame with one JCT reform family (a reform system on the cut
    dataset), the ACA source inputs, Medicaid and SNAP eligibility, and
    December SSI amounts.
    """
    from policyengine_us.data import USMultiYearDataset, USSingleYearDataset

    builder = _load_builder_module()
    # One real family stands for the others: each neutralizes one deduction
    # on the same reform path.
    (salt,) = (
        reform
        for reform in builder.US_JCT_TAX_EXPENDITURE_REFORMS
        if reform.neutralized_variable == "salt_deduction"
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", (salt,))
    frame = _real_engine_frame()
    cut = builder._dataset_from_frame(
        frame, assert_no_formula_owned_columns=False, period_only=True
    )
    projected = builder._dataset_from_frame(
        frame, assert_no_formula_owned_columns=False
    )
    assert isinstance(cut, USMultiYearDataset)
    assert cut.years == [builder.PERIOD]
    assert isinstance(projected, USSingleYearDataset)

    targets = (
        _soi("agi_amount", "adjusted_gross_income"),
        _soi("wages", "employment_income"),
        _population_age("pop_18_to_65", 18, 65),
        _variable("snap_amount", base_variable="snap"),
        _variable("ssi_recipients", base_variable="ssi", measure_mode="indicator_sum"),
        _spec("ca_state_income_tax", "state_income_tax", {"state_fips": "06"}),
        _spec("md_state_income_tax", "state_income_tax", {"state_fips": "24"}),
        _spec("eitc", "fixture"),
        _spec("jct_salt", "jct", measure=salt.measure),
    )

    def run() -> dict[str, object]:
        target_frame, _, compilation = builder._materialize_target_frame(frame, targets)
        batching = compilation["target_materialization_batching"]
        assert batching["jct_reform_families_simulated"] == 1
        return {
            "targets": target_frame.table("household"),
            "aca": builder._aca_source_tax_unit_table(frame, {}),
            "medicaid": builder._medicaid_person_eligibility(frame),
            "snap": builder._snap_spm_unit_eligibility(frame),
            "ssi": builder._ssi_person_uncapped_amount(frame),
        }

    dataset_from_frame = builder._dataset_from_frame
    asked: list[bool] = []

    def recording(*args, **kwargs):
        asked.append(kwargs.get("period_only", False))
        return dataset_from_frame(*args, **kwargs)

    def projecting(*args, **kwargs):
        # The in-build sites ask for PERIOD alone; give them the projection.
        return dataset_from_frame(*args, **{**kwargs, "period_only": False})

    monkeypatch.setattr(builder, "_dataset_from_frame", recording)
    with_cut = run()
    # Every site run here asked for PERIOD alone: the target frame, the JCT
    # reform, the ACA source inputs, Medicaid, SNAP and December SSI.
    assert len(asked) >= 6 and all(asked), asked
    monkeypatch.setattr(builder, "_dataset_from_frame", projecting)
    with_projection = run()

    assert with_cut["targets"]["income_tax"].to_numpy().any()
    assert salt.measure in with_cut["targets"].columns
    for name in ("targets", "aca"):
        pd.testing.assert_frame_equal(
            with_cut[name], with_projection[name], check_exact=True, obj=name
        )
    for name in ("medicaid", "snap", "ssi"):
        np.testing.assert_array_equal(
            with_cut[name], with_projection[name], err_msg=name
        )
