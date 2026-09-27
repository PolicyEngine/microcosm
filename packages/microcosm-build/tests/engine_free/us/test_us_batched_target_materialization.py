"""Tests split from packages/microcosm-build/tests/test_us_batched_target_materialization.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_batched_target_materialization import *


@pytest.mark.parametrize("batch_size", [1, 2, 3, 0])
def test_batched_base_simulation_matches_unbatched_target_frame(
    monkeypatch, batch_size
) -> None:
    """Identical target columns, column order and dtypes at every batch size."""

    builder = _load_builder_module()
    frame = _nested_frame()
    unbatched, unbatched_registry, unbatched_compilation, _ = _materialize(
        builder, monkeypatch, frame, None
    )
    batched, batched_registry, batched_compilation, _ = _materialize(
        builder, monkeypatch, frame, batch_size
    )

    # Every declared target materialized, so the comparison covers every
    # branch of the base simulation rather than an empty surface.
    assert unbatched_compilation["dropped_target_names"] == []
    unbatched_household = unbatched.table("household")
    assert set(spec.measure for spec in _TARGETS) <= set(unbatched_household)
    # The fixture exercises non-trivial values, not columns of zeros.
    for spec in _TARGETS:
        assert unbatched_household[spec.measure].to_numpy().any(), spec.measure

    pd.testing.assert_frame_equal(
        batched.table("household"), unbatched_household, check_exact=True
    )
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            batched.table(entity), unbatched.table(entity), check_exact=True
        )
    np.testing.assert_array_equal(
        batched.weights_for("household").values,
        unbatched.weights_for("household").values,
    )
    assert batched_registry.version == unbatched_registry.version
    assert (
        batched_compilation["dropped_target_names"]
        == unbatched_compilation["dropped_target_names"]
    )

def test_batched_base_simulation_reproduces_the_hand_computed_columns(
    monkeypatch,
) -> None:
    """Spot-check that the batched columns are the right numbers, not only
    equal to the unbatched ones. Household 4 has three tax units (a single, a
    joint and a head-of-household filer) and two SPM units."""

    builder = _load_builder_module()
    target_frame, _, _, _ = _materialize(builder, monkeypatch, _nested_frame(), 1)
    household = target_frame.table("household").set_index("household_id")

    tax_units = (40, 41, 42)
    expected_income_tax = sum(_record_value("income_tax", id_) for id_ in tax_units)
    assert household.loc[4, "income_tax"] == expected_income_tax
    assert household.loc[4, "jct_mock_credit"] == sum(
        10.0 * (id_ % 7) for id_ in tax_units
    )
    expected_agi = sum(_record_value("adjusted_gross_income", id_) for id_ in tax_units)
    assert household.loc[4, "agi_amount"] == expected_agi
    # MD is neither CA nor NY: the state-sliced rows are zero there.
    assert household.loc[4, "ca_state_income_tax"] == 0.0
    assert household.loc[4, "ny_snap"] == 0.0
    assert household.loc[2, "ny_snap"] == _record_value("snap", 200)
    assert household.loc[1, "ca_state_income_tax"] == sum(
        _record_value("state_income_tax", id_) for id_ in (10, 11)
    )
    persons = [row for row in _PERSONS if row[1] == 4]
    assert household.loc[4, "pop_under_18"] == sum(
        _record_value("age", row[0]) < 18 for row in persons
    )

@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batched_base_engines_partition_the_pool_and_are_released(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    household_ids = tuple(
        int(value) for value in frame.table("household")["household_id"]
    )
    _, _, compilation, ledger = _materialize(builder, monkeypatch, frame, batch_size)

    base = [entry for entry in ledger.constructions if entry["reform"] is None]
    reform = [entry for entry in ledger.constructions if entry["reform"] is not None]
    expected_batches = -(-len(household_ids) // batch_size)
    # The base simulation and the JCT family use the same partition: disjoint,
    # contiguous, in pool order, never larger than the batch size.
    for constructions in (base, reform):
        assert len(constructions) == expected_batches
        assert all(len(entry["households"]) <= batch_size for entry in constructions)
        assert (
            tuple(id_ for entry in constructions for id_ in entry["households"])
            == household_ids
        )
    # microcosm#456: each engine is released before the next is built, and
    # none survives the materializer.
    assert [entry["alive_before"] for entry in ledger.constructions] == [0] * len(
        ledger.constructions
    )
    assert all(simulation.dataset is None for simulation in ledger.simulations)
    # One metadata system plus one reform system for the family, never one
    # per batch.
    assert [system.reform for system in ledger.reform_systems] == [
        None,
        "mock_credit",
    ]

    assert compilation["target_materialization_batching"] == {
        "method": "household_position_batches",
        "maximum_microsim_batch_size": batch_size,
        "households": len(household_ids),
        "batches": expected_batches,
        "largest_batch_households": min(batch_size, len(household_ids)),
        "base_household_columns": compilation["target_materialization_batching"][
            "base_household_columns"
        ],
        "group_nesting_verified": True,
        "population_aggregate_guard_armed": True,
        "population_aggregate_variables_checked": list(
            builder.US_POPULATION_AGGREGATE_VARIABLES
        ),
        "jct_reform_families_simulated": 1,
    }
    # Every target column but the JCT row and the base-table input, plus the
    # two formula helpers the JCT subtraction and state rows read.
    base_columns = {spec.measure for spec in _TARGETS} - {
        "jct_mock_credit",
        "household_input",
    }
    base_columns |= {"income_tax", "state_income_tax"}
    assert compilation["target_materialization_batching"][
        "base_household_columns"
    ] == len(base_columns)

@pytest.mark.parametrize("batch_size", [None, 0, 5, 50])
def test_unbatched_base_simulation_runs_once_over_the_whole_frame(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    _, _, compilation, ledger = _materialize(builder, monkeypatch, frame, batch_size)

    base = [entry for entry in ledger.constructions if entry["reform"] is None]
    assert len(base) == 1
    assert len(base[0]["households"]) == frame.n("household")
    receipt = compilation["target_materialization_batching"]
    assert receipt["batches"] == 1
    assert receipt["largest_batch_households"] == frame.n("household")
    assert receipt["group_nesting_verified"] is False
    assert receipt["population_aggregate_guard_armed"] is False
    assert receipt["population_aggregate_variables_checked"] == []

@pytest.mark.parametrize(
    ("person_index", "crossing_person", "crossing_unit"),
    [
        # Person 5 (household 3, second batch) joins household 2's marital
        # unit 20000 (first batch): the group straddles the batch boundary,
        # and each batch on its own still nests. Only a pool-wide check
        # refuses it.
        pytest.param(4, (5, 3, 30, 300, 3000, 20000), 20000, id="across-batches"),
        # Person 4 (household 2) joins household 1's marital unit 10001;
        # both households sit in the first batch.
        pytest.param(3, (4, 2, 20, 200, 2000, 10001), 10001, id="within-a-batch"),
    ],
)
def test_batched_base_simulation_refuses_groups_that_cross_households(
    monkeypatch, person_index, crossing_person, crossing_unit
) -> None:
    """A group split across two batches would be simulated twice, each time
    with part of its members. Batching refuses it before building an engine."""

    builder = _load_builder_module()
    crossing = list(_PERSONS)
    crossing[person_index] = crossing_person
    frame = _nested_frame(tuple(crossing))
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)

    with pytest.raises(
        ValueError,
        match=rf"marital_unit units must be nested .*\[{crossing_unit}\]",
    ):
        builder._materialize_target_frame(
            frame, _TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.constructions == []

@pytest.mark.parametrize("claiming_tax_unit_id", [20, 50, 999])
def test_batched_base_simulation_refuses_nonlocal_claiming_tax_units(
    monkeypatch, claiming_tax_unit_id
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person").copy()
    person["medicaid_claiming_tax_unit_id"] = 0
    person.loc[0, "medicaid_claiming_tax_unit_id"] = claiming_tax_unit_id
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=())
    with pytest.raises(ValueError, match="medicaid_claiming_tax_unit_id.*household"):
        builder._materialize_target_frame(
            frame, _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.constructions == []

@pytest.mark.parametrize("claiming_tax_unit_id", [0, 10, 11])
def test_batched_base_simulation_allows_household_local_claiming_tax_units(
    monkeypatch, claiming_tax_unit_id
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person").copy()
    person["medicaid_claiming_tax_unit_id"] = 0
    person.loc[0, "medicaid_claiming_tax_unit_id"] = claiming_tax_unit_id
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=())
    _, _, compilation = builder._materialize_target_frame(
        frame, _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
    )
    assert len(ledger.constructions) == 3
    assert compilation["target_materialization_population_aggregate_guard"] == {
        "armed": True,
        "population_aggregate_variables_checked": list(
            builder.US_POPULATION_AGGREGATE_VARIABLES
        ),
    }

def test_batched_base_simulation_refuses_each_population_aggregate(
    monkeypatch,
) -> None:
    """A batch engine that computed any listed population aggregate is
    refused, and released, at the first batch; one simulation over the whole
    pool computes the same formulas without refusal."""

    builder = _load_builder_module()
    assert builder.US_POPULATION_AGGREGATE_VARIABLES == (
        "household_income_decile",
        "medicaid_slcsp_state_average_cost_index",
        "medicaid_slcsp_state_denominator",
        "spm_unit_income_decile",
    )
    for aggregate in builder.US_POPULATION_AGGREGATE_VARIABLES:
        ledger = _install_fake_engine(
            builder,
            monkeypatch,
            reform_specs=(),
            aggregate_reads={"aggregate_probe": (aggregate,)},
        )
        with pytest.raises(
            ValueError,
            match=(
                "Target materialization is not batch-invariant: the "
                rf"engine for household batch 1/3 computed {aggregate}@"
                rf"{builder.PERIOD}\. .*unbatched.*one slice or chunk"
            ),
        ):
            builder._materialize_target_frame(
                _nested_frame(),
                _AGGREGATE_PROBE_TARGETS,
                maximum_microsim_batch_size=2,
            )
        assert len(ledger.constructions) == 1, aggregate
        assert all(simulation.dataset is None for simulation in ledger.simulations)

        target_frame, _, compilation = builder._materialize_target_frame(
            _nested_frame(),
            _AGGREGATE_PROBE_TARGETS,
            maximum_microsim_batch_size=None,
        )
        assert compilation["dropped_target_names"] == []
        assert "aggregate_probe_total" in target_frame.table("household")
        assert ledger.simulations[-1].known[aggregate] == [str(builder.PERIOD)]
        receipt = compilation["target_materialization_batching"]
        assert receipt["batches"] == 1
        assert receipt["population_aggregate_guard_armed"] is False
        assert receipt["population_aggregate_variables_checked"] == []

def test_batched_base_simulation_refuses_an_aggregate_first_reached_in_last_batch(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("medicaid_slcsp_state_denominator",)},
        aggregate_household=5,
    )
    with pytest.raises(
        ValueError,
        match=r"household batch 3/3 computed medicaid_slcsp_state_denominator@",
    ):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert len(ledger.constructions) == 3
    assert all(simulation.dataset is None for simulation in ledger.simulations)

@pytest.mark.parametrize("branch_depth", [1, 2])
@pytest.mark.parametrize("clone_branches", [False, True])
def test_batched_base_simulation_reads_aggregates_held_by_live_branches(
    monkeypatch, branch_depth, clone_branches
) -> None:
    builder = _load_builder_module()
    _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("spm_unit_income_decile",)},
        aggregate_branch="/".join(f"branch_{depth}" for depth in range(branch_depth)),
        clone_branches=clone_branches,
    )
    with pytest.raises(ValueError, match=r"computed spm_unit_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )

@pytest.mark.parametrize("clone_branches", [False, True])
def test_batched_base_simulation_reads_aggregates_held_by_deleted_branches(
    monkeypatch, clone_branches
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("spm_unit_income_decile",)},
        aggregate_branch="temporary_parent/temporary_child",
        aggregate_delete_branch=True,
        clone_branches=clone_branches,
    )
    with pytest.raises(ValueError, match=r"computed spm_unit_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.simulations[0].branches == {}
    assert ledger.simulations[0].dataset is None

def test_batched_base_simulation_refuses_a_stored_aggregate_input(
    monkeypatch,
) -> None:
    """Any known period is refused, including values present at construction."""

    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        stored_inputs={"household_income_decile": (str(builder.PERIOD),)},
    )
    with pytest.raises(ValueError, match=r"computed household_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert len(ledger.constructions) == 1
    assert all(simulation.dataset is None for simulation in ledger.simulations)

@pytest.mark.parametrize("aggregate_in_baseline", [False, True])
def test_batched_reform_simulation_refuses_population_aggregates(
    monkeypatch, aggregate_in_baseline
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=_REFORMS,
        aggregate_reads={"income_tax": ("medicaid_slcsp_state_denominator",)},
        aggregate_branch="baseline/temporary",
        aggregate_delete_branch=True,
        aggregate_reform_only=True,
        aggregate_in_baseline=aggregate_in_baseline,
        clone_branches=True,
    )
    with pytest.raises(ValueError, match=r"computed medicaid_slcsp_state_denominator@"):
        builder._materialize_target_frame(
            _nested_frame(), _TARGETS, maximum_microsim_batch_size=2
        )
    assert sum(entry["reform"] is None for entry in ledger.constructions) == 3
    assert sum(entry["reform"] is not None for entry in ledger.constructions) == 1
    assert all(simulation.dataset is None for simulation in ledger.simulations)

@pytest.mark.parametrize(
    ("target", "wrap", "message"),
    [
        (
            "_select_households_by_position",
            # Each batch is handed the next batch's households.
            lambda real: lambda frame, positions: real(frame, positions + 1),
            "does not carry exactly its households in pool order",
        ),
        (
            "_base_simulation_household_columns",
            _second_batch_drops_a_column,
            "different column sets or orders",
        ),
        (
            "_base_simulation_household_columns",
            _one_value_short,
            "has shape",
        ),
    ],
)
def test_batched_base_simulation_refuses_batches_it_cannot_place(
    monkeypatch, target, wrap, message
) -> None:
    """The loop places batch values by pool position, so it refuses a batch
    whose households, columns or lengths do not line up with that position."""

    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    monkeypatch.setattr(builder, target, wrap(getattr(builder, target)))

    with pytest.raises(RuntimeError, match=message):
        builder._materialize_target_frame(
            _nested_frame(), _TARGETS, maximum_microsim_batch_size=2
        )

def test_checkpoint_round_trip_keeps_the_batching_receipt(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    identity = {
        "kind": "fixture",
        "materializer_version": builder.TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
    }
    path = tmp_path / "target_frame_checkpoint.h5"

    target_frame, _, compilation = builder._load_or_materialize_target_frame(
        frame,
        _TARGETS,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
        target_frame_checkpoint_build_commit="a" * 40,
        maximum_microsim_batch_size=2,
    )
    assert compilation["target_frame_checkpoint"]["status"] == "miss_written"
    receipt = compilation["target_materialization_batching"]
    assert receipt["batches"] == 3
    engines_after_miss = len(ledger.constructions)

    reloaded, _, reloaded_compilation = builder._load_or_materialize_target_frame(
        frame,
        _TARGETS,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
        target_frame_checkpoint_build_commit="b" * 40,
        maximum_microsim_batch_size=2,
    )
    checkpoint = reloaded_compilation["target_frame_checkpoint"]
    assert checkpoint["status"] == "hit"
    # microcosm#1018: the hit names the commit that wrote the checkpoint,
    # beside the batching receipt it restores.
    assert checkpoint["source_build_commit"] == "a" * 40
    # The hit runs no engine and says so; the writing run's receipt survives
    # in the stored compilation.
    assert len(ledger.constructions) == engines_after_miss
    assert reloaded_compilation["target_materialization_batching"] == {
        "status": "skipped_target_frame_checkpoint_hit"
    }
    assert checkpoint["stored_compilation"]["target_materialization_batching"] == (
        receipt
    )
    pd.testing.assert_frame_equal(
        reloaded.table("household")[list(target_frame.table("household"))],
        target_frame.table("household"),
        check_dtype=False,
    )

def test_population_aggregate_scan_follows_helpers_and_cross_record_markers(
    tmp_path,
) -> None:
    (tmp_path / "variables").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "variables" / "allocation.py").write_text(
        "def _state_total(values, state):\n"
        "    return np.sum(values[state == 6])\n"
        "def _axis_total(values):\n"
        "    return np.sum(values, axis=1)\n"
        "class helper_share(Variable):\n"
        "    def formula(person, period):\n"
        "        return _state_total(values, state)\n"
        "class direct_share(Variable):\n"
        "    def formula(person, period):\n"
        "        return values / np.sum(values)\n"
        "class axis_local(Variable):\n"
        "    def formula(person, period):\n"
        "        return _axis_total(values)\n"
        "class tools_rank(Variable):\n"
        "    def formula(person, period):\n"
        "        return _rank(values)\n"
    )
    (tmp_path / "tools" / "rank.py").write_text(
        "def _rank(values):\n"
        "    return _ordered(values)\n"
        "def _ordered(values):\n"
        "    return np.argsort(values)\n"
    )
    (tmp_path / "tests" / "fixture.py").write_text(
        "class ignored_test_fixture(Variable):\n"
        "    def formula(person, period):\n"
        "        return values.sum()\n"
    )
    markers = {
        "method_sum": "values.sum()",
        "method_mean": "values.mean()",
        "method_max": "values.max()",
        "method_min": "values.min()",
        "method_median": "values.median()",
        "join_isin": "np.isin(values, ids)",
        "join_in1d": "np.in1d(values, ids)",
        "join_unique": "np.unique(values)",
        "join_bincount": "np.bincount(values)",
        "join_searchsorted": "np.searchsorted(values, ids)",
    }
    (tmp_path / "variables" / "markers.py").write_text(
        "\n".join(
            f"class {name}(Variable):\n"
            "    def formula(person, period):\n"
            f"        return {expression}\n"
            for name, expression in markers.items()
        )
    )
    found = _engine_population_aggregate_sources(tmp_path)
    assert set(found) == {"direct_share", "helper_share", "tools_rank", *markers}
    assert found["helper_share"] == ["_state_total()"]
    assert found["tools_rank"] == ["_rank()"]

def test_refusal_reads_a_value_held_only_on_baseline() -> None:
    """Target materialization and post-export scoring share one walker. It
    also reads a ``reform=`` engine's ``baseline`` simulation, which holds its
    own values and is not among the engine's branches."""
    builder = _load_builder_module()

    def engine(known, **extra):
        return SimpleNamespace(
            get_holder=lambda name: SimpleNamespace(
                get_known_periods=lambda: known.get(name, [])
            ),
            branches={},
            **extra,
        )

    root = engine({}, baseline=engine({"medicaid_slcsp_state_denominator": ["2024"]}))
    with pytest.raises(
        ValueError, match=r"computed medicaid_slcsp_state_denominator@2024"
    ):
        builder._refuse_batch_population_aggregates(
            root, batch=1, batches=3, n_households=6
        )
