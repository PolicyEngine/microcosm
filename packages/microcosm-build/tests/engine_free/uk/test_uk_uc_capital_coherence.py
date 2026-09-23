"""Tests split from packages/microcosm-build/tests/test_uk_uc_capital_coherence.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_uc_capital_coherence import *


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
    assert stage.outputs == ("uc_reported_capital",)
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
    assert stages[coherence_index + 1].stage == "uc_deduction_attributes"


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
    assert result.refreshed_would_claim_count == 4


def test_redraw_is_reporter_conditioned_cell_exact_and_household_weighted() -> None:
    result = cohere_uc_capital(_frame())
    benunit = result.frame.table("benunit").set_index("benunit_id")

    # benunit 1005's identity draw is 0.364. The weighted donor CDF is
    # [0.1, 1.0], so it selects 200; an unweighted draw would select 100.
    assert benunit.loc[1005, "frs_benunit_capital"] == 200.0
    assert benunit.loc[1007, "frs_benunit_capital"] == 3_000.0
    assert benunit.loc[1006, "frs_benunit_capital"] == 888_888.0
    assert benunit.loc[401, "frs_benunit_capital"] == 999_999.0
    assert result.redrawn_spi_reporter_count == 2


def test_capital_donor_cells_use_claimant_partnership_and_preserve_marital_status():
    frame = _frame()
    benunit = frame.table("benunit")
    # Cohabiting couple and a married claimant whose spouse is not in this
    # unit must use the couple and single donor cells respectively.
    benunit.loc[benunit["benunit_id"] == 1007, "is_married"] = False
    benunit.loc[benunit["benunit_id"] == 1005, "is_married"] = True
    marital_before = benunit["is_married"].copy()

    result = cohere_uc_capital(frame).frame.table("benunit")

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
    transform = UKUCCapitalCoherenceStageTransform(stage=_stage())
    result = UKStageKernel("uc_capital_coherence", transform).run(context)

    capital = result.columns[("benunit", "frs_benunit_capital")]
    assert capital.loc[1005] == 200.0
    assert capital.loc[1007] == 3_000.0
    assert capital.loc[1006] == 888_888.0  # A non-reporter is unchanged.
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
    assert transform.checkpoint_metadata()["evidence"] == {
        "stage": "uc_capital_coherence",
        "post_fill_reporter_count": 5,
        "redrawn_spi_reporter_count": 2,
        "refreshed_would_claim_count": 4,
        "redraw_seed": UC_CAPITAL_REDRAW_SEED,
        "redraw_salt": UC_CAPITAL_REDRAW_SALT,
    }


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
        redraw = (
            benunit_table[support_channel_column("benunit")].eq("spi").to_numpy()
            & reporter
        )
        capital = benunit_table["frs_benunit_capital"].to_numpy(dtype=float).copy()
        _redraw_spi_reporter_capital(
            benunit_table,
            person=person_table,
            household=household_table,
            household_weights=household_weight_values,
            reporter=reporter,
            base=base,
            redraw=redraw,
            capital=capital,
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
