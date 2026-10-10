"""Tests split from packages/microcosm-build/tests/test_uk_uc_capital_coherence.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_capital_coherence import *

RECEIPTS = ("coarsening_levels", "capital_against_investment_income")


def test_manifest_declares_exact_redraw_seed_and_output() -> None:
    stage = _stage()
    redraw = next(
        operation
        for operation in stage.operations
        if operation.kind == "redraw_spi_reporter_capital"
    )

    assert redraw.parameters["output"] == UC_CAPITAL_REDRAW_OUTPUT
    assert redraw.parameters["seed"] == UC_CAPITAL_REDRAW_SEED
    assert redraw.parameters["salt"] == UC_CAPITAL_REDRAW_SALT
    assert redraw.parameters["couple_status"] == "is_uc_couple"
    assert redraw.parameters["rows"] == "spi_channel"
    assert redraw.parameters["minimum_cell_donors"] == UC_CAPITAL_MINIMUM_CELL_DONORS
    assert redraw.parameters["coarsening"] == list(UC_CAPITAL_COARSENING)
    assert redraw.parameters["investment_income_band_edges"] == list(
        UC_CAPITAL_INCOME_BAND_EDGES
    )
    assert stage.outputs == ("uc_reported_capital", "pension_credit_reported_capital")
    assert stage.rewrites == ("frs_benunit_capital", "would_claim_uc")


def test_stage_refuses_a_stale_marriage_based_donor_contract():
    stage = _stage()
    operations = tuple(
        replace(
            operation,
            parameters={**operation.parameters, "couple_status": "is_married"},
        )
        if operation.kind == "redraw_spi_reporter_capital"
        else operation
        for operation in stage.operations
    )
    transform = UKUCCapitalCoherenceStageTransform(
        stage=replace(stage, operations=operations)
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        transform(_frame())


def test_stage_orders_after_every_universal_credit_report_writer() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stages = spec.sources.stages
    coherence_index = next(
        index
        for index, stage in enumerate(stages)
        if stage.stage == "uc_capital_coherence"
    )
    reporter_writers = [
        (index, stage.stage)
        for index, stage in enumerate(stages)
        if "universal_credit_reported" in (*stage.outputs, *stage.rewrites)
    ]

    assert reporter_writers
    assert all(index < coherence_index for index, _ in reporter_writers)
    # The Pension Credit take-up redraw (microcosm#1069 c7) and the Child
    # Benefit redraw (microcosm#1063 c8) sit between the coherence stage and
    # the deduction attributes; neither writes a UC report.
    assert [
        stage.stage for stage in stages[coherence_index + 1 : coherence_index + 4]
    ] == [
        "pension_credit_take_up",
        "child_benefit_take_up",
        "uc_deduction_attributes",
    ]


def test_or_refresh_truth_table_and_same_capital_source() -> None:
    result = cohere_uc_capital(_frame())
    benunit = result.frame.table("benunit").set_index("benunit_id")

    assert benunit.loc[101, "would_claim_uc"]
    assert benunit.loc[201, "would_claim_uc"]
    assert benunit.loc[401, "would_claim_uc"]
    assert not benunit.loc[501, "would_claim_uc"]
    assert benunit.loc[1005, "would_claim_uc"]
    assert benunit.loc[1007, "would_claim_uc"]
    np.testing.assert_array_equal(
        benunit["uc_reported_capital"], benunit["frs_benunit_capital"]
    )
    # Pension Credit reads the same recorded capital, the unavailable
    # sentinel included (pe-uk#2018, uk-data#513).
    np.testing.assert_array_equal(
        benunit["pension_credit_reported_capital"], benunit["frs_benunit_capital"]
    )
    assert result.refreshed_would_claim_count == 4


def test_redraw_is_reporter_conditioned_cell_exact_and_household_weighted() -> None:
    result = cohere_uc_capital(_frame(), minimum_cell_donors=1)
    benunit = result.frame.table("benunit").set_index("benunit_id")

    # benunit 1005's identity draw is 0.364. The weighted donor CDF is
    # [0.1, 1.0], so it selects 200; an unweighted draw would select 100.
    assert benunit.loc[1005, "frs_benunit_capital"] == 200.0
    assert benunit.loc[1007, "frs_benunit_capital"] == 3_000.0
    # The SPI non-reporter is redrawn too, from the base non-reporters of its
    # cell (777,777 and 999,999 at equal weight): its copied answer is gone.
    draw = stable_identity_uniforms(
        np.asarray([1006]), seed=UC_CAPITAL_REDRAW_SEED, salt=UC_CAPITAL_REDRAW_SALT
    )[0]
    expected = 777_777.0 if draw < 0.5 else 999_999.0
    assert benunit.loc[1006, "frs_benunit_capital"] == expected
    assert benunit.loc[401, "frs_benunit_capital"] == 999_999.0
    assert result.redrawn_spi_reporter_count == 2
    assert result.redrawn_spi_benefit_units == 3
    assert result.coarsening_levels["exact_cell"] == 3


def test_capital_donor_cells_use_claimant_partnership_and_preserve_marital_status():
    frame = _frame()
    benunit = frame.table("benunit")
    # Cohabiting couple and a married claimant whose spouse is not in this
    # unit must use the couple and single donor cells respectively.
    benunit.loc[benunit["benunit_id"] == 1007, "is_married"] = False
    benunit.loc[benunit["benunit_id"] == 1005, "is_married"] = True
    marital_before = benunit["is_married"].copy()

    result = cohere_uc_capital(frame, minimum_cell_donors=1).frame.table("benunit")

    by_id = result.set_index("benunit_id")
    assert by_id.loc[1007, "frs_benunit_capital"] == 3_000.0
    assert by_id.loc[1005, "frs_benunit_capital"] == 200.0
    pd.testing.assert_series_equal(result["is_married"], marital_before)


def test_graph_scoped_capital_redraw_receives_claimant_roles() -> None:
    """The real node must retain role flags through executor input pruning."""

    frame = _frame()
    person = frame.table("person")
    benunit = frame.table("benunit")
    frame.table("household")["region"] = "LONDON"
    person["age"] = np.where(person["is_benunit_head"] | person["is_parent"], 40, 5)
    person[support_channel_column("person")] = person["person_benunit_id"].map(
        benunit.set_index("benunit_id")[support_channel_column("benunit")]
    )
    # Legal marriage disagrees with the intended donor cells in both directions.
    benunit.loc[benunit["benunit_id"] == 1007, "is_married"] = False
    benunit.loc[benunit["benunit_id"] == 1005, "is_married"] = True
    node = uk_spine_graph(source_mode="split").node("uc_capital_coherence")
    context = _project_context(
        node,
        Population.from_frame(frame, "capital-input"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    transform = UKUCCapitalCoherenceStageTransform(
        stage=_stage(), minimum_cell_donors=1
    )
    result = UKStageKernel("uc_capital_coherence", transform).run(context)

    capital = result.columns[("benunit", "frs_benunit_capital")]
    assert capital.loc[1005] == 200.0
    assert capital.loc[1007] == 3_000.0
    # The non-reporter draws from base non-reporters, never reporters.
    assert capital.loc[1006] in (777_777.0, 999_999.0)
    assert result.columns[("benunit", "would_claim_uc")].loc[1007]
    assert (
        transform.checkpoint_metadata()["evidence"]["redrawn_spi_reporter_count"] == 2
    )


def test_transform_is_deterministic_and_idempotent() -> None:
    transform = UKUCCapitalCoherenceStageTransform(stage=_stage())

    first = transform(_frame())
    twin = UKUCCapitalCoherenceStageTransform(stage=_stage())(_frame())
    repeated = UKUCCapitalCoherenceStageTransform(stage=_stage())(first)

    for candidate in (twin, repeated):
        for entity in ("person", "benunit", "household"):
            pd.testing.assert_frame_equal(
                first.table(entity), candidate.table(entity), check_exact=True
            )
    evidence = transform.checkpoint_metadata()["evidence"]
    assert {key: evidence[key] for key in evidence if key not in RECEIPTS} == {
        "stage": "uc_capital_coherence",
        "post_fill_reporter_count": 5,
        "redrawn_spi_reporter_count": 2,
        "redrawn_spi_benefit_units": 3,
        "refreshed_would_claim_count": 4,
        "redraw_seed": UC_CAPITAL_REDRAW_SEED,
        "redraw_salt": UC_CAPITAL_REDRAW_SALT,
        "minimum_cell_donors": UC_CAPITAL_MINIMUM_CELL_DONORS,
    }
    # Two base reporters and two base non-reporters cannot fill a 20-donor
    # cell, so every SPI unit draws from its reporter-status pool.
    assert evidence["coarsening_levels"][UC_CAPITAL_COARSENING[-1]] == 3


def test_redraw_is_stable_under_input_row_permutation() -> None:
    frame = _frame()
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    weights = frame.weights_for("household").values.copy()

    def redraw_tables(
        person_table: pd.DataFrame,
        benunit_table: pd.DataFrame,
        household_table: pd.DataFrame,
        household_weight_values: np.ndarray,
    ) -> pd.Series:
        reporter_ids = person_table.loc[
            person_table["universal_credit_reported"] > 0,
            "person_benunit_id",
        ]
        reporter = benunit_table["benunit_id"].isin(reporter_ids).to_numpy()
        base = benunit_table[support_channel_column("benunit")].eq("frs").to_numpy()
        redraw = benunit_table[support_channel_column("benunit")].eq("spi").to_numpy()
        capital = benunit_table["frs_benunit_capital"].to_numpy(dtype=float).copy()
        _redraw_spi_capital(
            benunit_table,
            person=person_table,
            weights=_household_to_benunit_weights(
                benunit_table,
                person=person_table,
                household=household_table,
                household_weights=household_weight_values,
            ),
            reporter=reporter,
            base=base,
            redraw=redraw,
            capital=capital,
            investment=benunit_financial_investment_income(person_table, benunit_table),
            minimum_cell_donors=1,
        )
        return pd.Series(capital, index=benunit_table["benunit_id"]).sort_index()

    expected = redraw_tables(person, benunit, household, weights)
    order = np.asarray([7, 2, 5, 0, 6, 1, 4, 3])
    actual = redraw_tables(
        person.sample(frac=1.0, random_state=91).reset_index(drop=True),
        benunit.iloc[order].reset_index(drop=True),
        household.iloc[order].reset_index(drop=True),
        weights[order],
    )

    pd.testing.assert_series_equal(expected, actual)


def test_stage_refuses_the_undefined_negative_interval() -> None:
    # Round-2 residual (a), stage-time arm: the -1 contract has exactly two
    # regions. A carrier value of -0.5 (finite, above the old bare floor,
    # not the sentinel) must refuse at the stage boundary, not flow on.
    frame = _frame()
    frame.table("benunit")["frs_benunit_capital"] = -0.5
    with pytest.raises(ValueError, match="sentinel or nonnegative"):
        cohere_uc_capital(frame)


def test_stage_refuses_near_sentinel_values_exactly() -> None:
    # #833: np.isclose admitted ~[-1.00001, -0.99999] as "exactly the
    # sentinel", silently reclassifying a corrupted value as a declared
    # absence. Sentinel equality is exact; the near-sentinel sliver refuses.
    frame = _frame()
    frame.table("benunit")["frs_benunit_capital"] = -1.000005
    with pytest.raises(ValueError, match="sentinel or nonnegative"):
        cohere_uc_capital(frame)


def test_children_band_caps_at_three_plus_and_boolean_helper_is_strict() -> None:
    np.testing.assert_array_equal(
        _dependent_children_band(pd.Series([0, 1, 2, 3, 8])),
        np.asarray([0, 1, 2, 3, 3], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        _boolean_values(pd.Series([False, True], name="flag")),
        np.asarray([False, True]),
    )


def _with_investment_income(frame, income_by_benunit):
    person = frame.table("person")
    heads = person["is_benunit_head"].astype(bool)
    for benunit_id, amount in income_by_benunit.items():
        rows = heads & person["person_benunit_id"].eq(benunit_id)
        person.loc[rows, "dividend_income"] = amount
    return frame


def test_investment_income_bands_follow_the_implied_capital_edges() -> None:
    income = np.asarray([0.0, 0.01, 240.0, 240.01, 640.0, 2_000.0, 8_000.0, 9e6])
    np.testing.assert_array_equal(
        _investment_income_band(income), [0, 1, 1, 2, 2, 3, 4, 5]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        _investment_income_band(np.asarray([-1.0]))


def test_spi_capital_follows_the_units_own_investment_income() -> None:
    # Base reporters 101 and 201 hold 100 and 200 with no investment income;
    # give 201 GBP 1,000 of dividends (band 3) and the SPI reporter 1005 the
    # same: in exact cells it can only draw 201's capital.
    frame = _with_investment_income(_frame(), {201: 1_000.0, 1005: 1_000.0})
    result = cohere_uc_capital(frame, minimum_cell_donors=1)
    benunit = result.frame.table("benunit").set_index("benunit_id")
    assert benunit.loc[1005, "frs_benunit_capital"] == 200.0
    # Property income is not financial capital's income.
    frame = _frame()
    frame.table("person")["property_income"] = 50_000.0
    np.testing.assert_array_equal(
        benunit_financial_investment_income(
            frame.table("person"), frame.table("benunit")
        ),
        np.zeros(len(frame.table("benunit"))),
    )


def test_coarsening_widens_in_the_declared_order_and_keeps_reporter_status() -> None:
    # A two-donor floor leaves the 1-child couple reporter cell (one donor)
    # short, so 1007 widens; its pool never includes a non-reporter's capital.
    result = cohere_uc_capital(_frame(), minimum_cell_donors=2)
    benunit = result.frame.table("benunit").set_index("benunit_id")
    assert benunit.loc[1007, "frs_benunit_capital"] in (100.0, 200.0, 3_000.0)
    assert benunit.loc[1005, "frs_benunit_capital"] in (100.0, 200.0, 3_000.0)
    assert benunit.loc[1006, "frs_benunit_capital"] in (777_777.0, 999_999.0)
    levels = result.coarsening_levels
    assert list(levels) == ["exact_cell", *UC_CAPITAL_COARSENING]
    assert sum(levels.values()) == 3
    assert levels["exact_cell"] == 2  # 1005 and 1006 have two donors each
    # Without any donor of its reporter status the unit refuses.
    frame = _frame()
    benunit_table = frame.table("benunit")
    benunit_table.loc[
        benunit_table["benunit_id"].isin([401, 501]), "frs_benunit_capital"
    ] = -1.0
    with pytest.raises(ValueError, match="reporter status False"):
        cohere_uc_capital(frame, minimum_cell_donors=1)


def test_receipt_counts_capital_below_what_the_income_implies() -> None:
    # 1005 reports GBP 1,000 of dividends (GBP 25,000 at 4%) and holds
    # 999,999 before; base 101 holds 100 against GBP 1,000.
    frame = _with_investment_income(_frame(), {101: 1_000.0, 1005: 1_000.0})
    result = cohere_uc_capital(frame, minimum_cell_donors=1)
    receipt = result.capital_against_investment_income
    before = receipt["before"]["implied_above_16000_capital_at_or_below"]
    after = receipt["after"]["implied_above_16000_capital_at_or_below"]
    assert before["frs"]["benefit_units"] == 1  # 101 is a base row: unchanged
    assert after["frs"]["benefit_units"] == 1
    assert before["spi"]["benefit_units"] == 0
    # 1005 now draws from its income band's donor, 101 (GBP 100).
    assert after["spi"]["benefit_units"] == 1
    assert receipt["after"]["spi_reporters_capital_above_16000"] == 0
