"""Tests split from packages/microcosm-frame/tests/test_policyengine_uk_adapter.py."""

# ruff: noqa: F403, F405
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters import policyengine_uk as adapter_module
from microcosm.frame.adapters.engine_population import SPINE_BATCHED_PASS
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
)
from test_support.microcosm_frame.policyengine_uk_adapter import *


def test_policyengine_uk_adapter_satisfies_rules_protocol_without_importing_engine() -> (
    None
):
    adapter = PolicyEngineUKEngine()

    assert isinstance(adapter, RulesEngine)
    assert adapter.country == "uk"
    assert adapter.entity_schema() == UK_SCHEMA


def test_policyengine_uk_adapter_export_side_is_not_implemented() -> None:
    adapter = PolicyEngineUKEngine()

    with pytest.raises(NotImplementedError, match="write_uk_national_frame"):
        adapter.write_dataset(object(), "unused.h5", period=2023)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_explicit_uc_roles(monkeypatch, defect):
    definition = SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )
    variables = {
        "is_uc_claimant": definition,
        "universal_credit": SimpleNamespace(
            entity=SimpleNamespace(key="benunit"), value_type=float
        ),
    }
    if defect == "missing":
        del variables["is_uc_claimant"]
    elif defect == "entity":
        definition.entity.key = "benunit"
    elif defect == "dtype":
        definition.value_type = float
    elif defect == "period":
        definition.definition_period = "month"
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.array([False if defect == "changed" else True])

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=[] if defect == "not_loaded" else ["is_uc_claimant"],
        calculate=calculate,
    )
    engine = PolicyEngineUKEngine()
    engine._system = simulation.tax_benefit_system
    monkeypatch.setattr(engine, "_build_dataset", lambda *args: object())
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: simulation),
    )
    # The engine's year window needs policyengine-uk itself; the stand-in
    # engine is built from the dataset as it is.
    monkeypatch.setattr(
        engine,
        "_read_year_simulation",
        lambda microsimulation_class, dataset, period: microsimulation_class(
            dataset=dataset
        ),
    )
    bundle = SimpleNamespace(
        table=lambda entity: pd.DataFrame({"is_uc_claimant": [True]}),
        n=lambda entity: 1,
    )
    with pytest.raises(ValueError, match="is_uc_claimant"):
        engine.materialize(bundle, ["universal_credit"], 2025)
    assert "universal_credit" not in calls


def _person_role():
    return SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )


_ROLES = ("is_uc_claimant", "is_claimant_or_partner", "is_hbai_dependent_child")
_SUPPLIED = {
    "is_uc_claimant": np.array([True, False]),
    "is_claimant_or_partner": np.array([True, False]),
    "is_hbai_dependent_child": np.array([False, True]),
}


def _role_engine(monkeypatch, variables, *, loaded, values):
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.asarray(values[name]) if name in values else np.zeros(2)

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=loaded,
        calculate=calculate,
    )
    engine = PolicyEngineUKEngine()
    engine._system = simulation.tax_benefit_system
    monkeypatch.setattr(engine, "_build_dataset", lambda *args: object())
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: simulation),
    )
    # The engine's year window needs policyengine-uk itself; the stand-in
    # engine is built from the dataset as it is.
    monkeypatch.setattr(
        engine,
        "_read_year_simulation",
        lambda microsimulation_class, dataset, period: microsimulation_class(
            dataset=dataset
        ),
    )
    bundle = SimpleNamespace(
        table=lambda entity: pd.DataFrame(_SUPPLIED), n=lambda entity: 2
    )
    return engine, bundle, calls


_OUTPUT = SimpleNamespace(entity=SimpleNamespace(key="person"), value_type=float)


def test_the_validated_roles_are_the_supplied_person_types() -> None:
    assert tuple(name for name, _ in SUPPLIED_ROLE_INPUTS) == _ROLES


@pytest.mark.parametrize("role", ["is_claimant_or_partner", "is_hbai_dependent_child"])
@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_person_type_roles(monkeypatch, role, defect):
    # policyengine-uk 2.107.0 reads both person types across its programmes;
    # a supplied role must reach the model as an input, unchanged.
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    if defect == "missing":
        del variables[role]
    elif defect == "entity":
        variables[role].entity.key = "benunit"
    elif defect == "dtype":
        variables[role].value_type = float
    elif defect == "period":
        variables[role].definition_period = "month"
    loaded = [name for name in _ROLES if not (defect == "not_loaded" and name == role)]
    values = dict(_SUPPLIED)
    if defect == "changed":
        values[role] = ~values[role]
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=loaded, values=values
    )
    with pytest.raises(ValueError, match=role):
        engine.materialize(bundle, ["income_tax"], 2025)
    assert "income_tax" not in calls


def test_adapter_accepts_person_type_roles_the_model_reads(monkeypatch):
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=list(_ROLES), values=_SUPPLIED
    )
    engine.materialize(bundle, ["income_tax"], 2025)
    assert calls == [*_ROLES, "income_tax"]


def _stand_in_engine_modules(monkeypatch, path: str, **attributes) -> None:
    """Install stand-in policyengine-uk modules down to ``path``."""

    import sys
    from types import ModuleType

    parts = path.split(".")
    parent = None
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, parts[depth - 1], module)
        parent = module
    for key, value in attributes.items():
        setattr(parent, key, value)


def test_uk_dataset_through_asks_policyengine_uk_for_the_years_to_end_year(
    monkeypatch,
):
    from microcosm.frame.adapters.policyengine_uk import uk_dataset_through

    calls = []

    def extend_single_year_dataset(dataset, parameters, end_year=None):
        calls.append((dataset, parameters, end_year))
        return list(range(int(dataset.time_period), end_year + 1))

    _stand_in_engine_modules(
        monkeypatch,
        "policyengine_uk.data.economic_assumptions",
        extend_single_year_dataset=extend_single_year_dataset,
    )
    dataset = SimpleNamespace(
        time_period="2024", validate=lambda: calls.append("validate")
    )
    parameters = object()

    assert uk_dataset_through(dataset, "2025", parameters) == [2024, 2025]
    assert uk_dataset_through(dataset, 2024, parameters) == [2024]
    # The dataset is checked before every projection, even one of no year.
    assert calls == [
        "validate",
        (dataset, parameters, 2025),
        "validate",
        (dataset, parameters, 2024),
    ]
    with pytest.raises(ValueError, match="forward only"):
        uk_dataset_through(dataset, 2023, parameters)
    assert len(calls) == 4


def test_uk_dataset_through_refuses_a_dataset_its_check_refuses(monkeypatch):
    """policyengine-uk checks a dataset only on the years it projects, so a
    dataset read at its own year is checked here; a refusal stops the
    projection."""

    from microcosm.frame.adapters.policyengine_uk import uk_dataset_through

    projected = []
    _stand_in_engine_modules(
        monkeypatch,
        "policyengine_uk.data.economic_assumptions",
        extend_single_year_dataset=lambda *args, **kwargs: projected.append(args),
    )

    def validate():
        raise ValueError("Column 'note' contains NaN values.")

    dataset = SimpleNamespace(time_period="2024", validate=validate)
    for end_year in (2024, 2025):
        with pytest.raises(ValueError, match="'note' contains NaN"):
            uk_dataset_through(dataset, end_year, object())
    assert projected == []


def _stand_in_rebalancing(monkeypatch, years, **constants):
    _stand_in_engine_modules(
        monkeypatch,
        "policyengine_uk.scenarios.uc_reform",
        REBALANCING_YEARS=years,
        **constants,
    )
    import sys

    return sys.modules["policyengine_uk.scenarios.uc_reform"]


#: The years policyengine-uk builds its ``REBALANCING_YEARS`` from.
_ENGINE_YEARS = {"FIRST_REBALANCING_YEAR": 2026, "PROJECTION_END_YEAR": 2032}


@pytest.mark.parametrize(
    ("end_year", "inside"),
    [
        (2024, range(2026, 2026)),
        (2025, range(2026, 2026)),
        (2027, range(2026, 2028)),
        (2040, range(2026, 2033)),
    ],
)
def test_rebalancing_years_stop_at_end_year_inside_the_block(
    monkeypatch, end_year, inside
):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    with uk_rebalancing_years_through(end_year) as years:
        assert uc_reform.REBALANCING_YEARS == inside
        assert years == inside

    assert uc_reform.REBALANCING_YEARS is installed


def test_rebalancing_years_are_restored_when_the_block_raises(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    with pytest.raises(RuntimeError, match="construction failed"):
        with uk_rebalancing_years_through(2025):
            assert uc_reform.REBALANCING_YEARS == range(2026, 2026)
            raise RuntimeError("construction failed")

    assert uc_reform.REBALANCING_YEARS is installed


@pytest.mark.parametrize("installed", [(2026, 2027), range(2026, 2033, 2)])
def test_rebalancing_years_refuse_a_changed_engine(monkeypatch, installed):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    with pytest.raises(TypeError, match="REBALANCING_YEARS"):
        with uk_rebalancing_years_through(2025):
            pass

    assert uc_reform.REBALANCING_YEARS is installed


def test_rebalancing_years_are_restored_when_blocks_exit_out_of_order(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    first = uk_rebalancing_years_through(2027)
    second = uk_rebalancing_years_through(2027)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    # The second block is still open, so its simulation still gets the cut.
    assert uc_reform.REBALANCING_YEARS == range(2026, 2028)
    second.__exit__(None, None, None)

    assert uc_reform.REBALANCING_YEARS is installed


def test_a_block_for_another_year_is_refused_inside_an_open_block(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    with uk_rebalancing_years_through(2027):
        for other_year in (2025, 2030):
            with pytest.raises(RuntimeError, match="cut at 2027"):
                with uk_rebalancing_years_through(other_year):
                    pass
            # The refusal leaves the open block's cut in force.
            assert uc_reform.REBALANCING_YEARS == range(2026, 2028)
        with uk_rebalancing_years_through("2027") as years:
            assert years == range(2026, 2028)
        assert uc_reform.REBALANCING_YEARS == range(2026, 2028)

    assert uc_reform.REBALANCING_YEARS is installed


def test_a_block_on_another_thread_waits_for_the_open_block(monkeypatch):
    import threading

    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)
    entered = threading.Event()
    seen = []

    def other_thread():
        with uk_rebalancing_years_through(2027) as years:
            entered.set()
            seen.append((years, uc_reform.REBALANCING_YEARS))

    thread = threading.Thread(target=other_thread)
    with uk_rebalancing_years_through(2025):
        thread.start()
        # The other thread cannot cut the range while this block is open.
        assert not entered.wait(0.2)
        assert uc_reform.REBALANCING_YEARS == range(2026, 2026)
    thread.join(timeout=10)

    assert not thread.is_alive()
    # It cut the engine's own range, not this block's cut of it.
    assert seen == [(range(2026, 2028), range(2026, 2028))]
    assert uc_reform.REBALANCING_YEARS is installed


def test_a_range_changed_inside_the_block_is_refused_and_restored(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import uk_rebalancing_years_through

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed)

    with pytest.raises(RuntimeError, match="changed inside"):
        with uk_rebalancing_years_through(2025):
            uc_reform.REBALANCING_YEARS = range(2026, 2030)

    assert uc_reform.REBALANCING_YEARS is installed
    with uk_rebalancing_years_through(2027) as years:
        assert years == range(2026, 2028)
    assert uc_reform.REBALANCING_YEARS is installed


def test_the_whole_loop_block_accepts_the_engines_own_range(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import uk_whole_rebalancing_years

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed, **_ENGINE_YEARS)

    with uk_whole_rebalancing_years():
        assert uc_reform.REBALANCING_YEARS is installed

    assert uc_reform.REBALANCING_YEARS is installed


def test_the_whole_loop_block_accepts_an_engine_that_was_never_imported(monkeypatch):
    import sys

    from microcosm.frame.adapters.policyengine_uk import uk_whole_rebalancing_years

    monkeypatch.delitem(
        sys.modules, "policyengine_uk.scenarios.uc_reform", raising=False
    )

    with uk_whole_rebalancing_years():
        assert "policyengine_uk.scenarios.uc_reform" not in sys.modules


@pytest.mark.parametrize("left", [range(2026, 2026), range(2026, 2028)])
def test_the_whole_loop_block_refuses_a_range_left_cut(monkeypatch, left):
    from microcosm.frame.adapters.policyengine_uk import uk_whole_rebalancing_years

    uc_reform = _stand_in_rebalancing(monkeypatch, left, **_ENGINE_YEARS)

    with pytest.raises(RuntimeError, match="a cut was left"):
        with uk_whole_rebalancing_years():
            pass

    assert uc_reform.REBALANCING_YEARS is left


def test_the_whole_loop_block_is_refused_inside_an_open_cut(monkeypatch):
    from microcosm.frame.adapters.policyengine_uk import (
        uk_rebalancing_years_through,
        uk_whole_rebalancing_years,
    )

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed, **_ENGINE_YEARS)

    with uk_rebalancing_years_through(2025):
        with pytest.raises(RuntimeError, match="needs the whole loop"):
            with uk_whole_rebalancing_years():
                pass
        assert uc_reform.REBALANCING_YEARS == range(2026, 2026)
    with uk_whole_rebalancing_years():
        assert uc_reform.REBALANCING_YEARS is installed


def test_the_whole_loop_block_waits_for_a_cut_on_another_thread(monkeypatch):
    import threading

    from microcosm.frame.adapters.policyengine_uk import (
        uk_rebalancing_years_through,
        uk_whole_rebalancing_years,
    )

    installed = range(2026, 2033)
    uc_reform = _stand_in_rebalancing(monkeypatch, installed, **_ENGINE_YEARS)
    entered = threading.Event()
    seen = []

    def other_thread():
        with uk_whole_rebalancing_years():
            entered.set()
            seen.append(uc_reform.REBALANCING_YEARS)

    thread = threading.Thread(target=other_thread)
    with uk_rebalancing_years_through(2025):
        thread.start()
        # A simulation that needs the whole loop is not built during the cut.
        assert not entered.wait(0.2)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert seen == [installed]


@pytest.mark.parametrize(
    "constants",
    [
        {},
        {"FIRST_REBALANCING_YEAR": 2026},
        {"FIRST_REBALANCING_YEAR": 2026, "PROJECTION_END_YEAR": "2032"},
    ],
)
def test_the_whole_loop_block_refuses_a_changed_engine(monkeypatch, constants):
    from microcosm.frame.adapters.policyengine_uk import uk_whole_rebalancing_years

    _stand_in_rebalancing(monkeypatch, range(2026, 2033), **constants)

    with pytest.raises(TypeError, match="FIRST_REBALANCING_YEAR"):
        with uk_whole_rebalancing_years():
            pass


# ---------------------------------------------------------------------------
# Calls over more than one clone run clone by clone (microcosm#931)
# ---------------------------------------------------------------------------


def _clone_frame(clones=(0, 1, 2), *, column=True, missing=False):
    """Two households per clone, one person each, clones interleaved by id.
    Everything is made up."""

    households = len(clones) * 2
    ids = np.arange(1, households + 1, dtype="int64")
    household = pd.DataFrame({"household_id": ids})
    if column:
        index = np.array([clones[i % len(clones)] for i in range(households)])
        household[UK_SPINE_ENGINE_BATCH_COLUMN] = (
            pd.array(index, dtype="Int64") if missing else index
        )
        if missing:
            household.loc[0, UK_SPINE_ENGINE_BATCH_COLUMN] = pd.NA
    person = pd.DataFrame(
        {
            "person_id": ids[::-1] * 10,
            "person_benunit_id": ids[::-1],
            "person_household_id": ids[::-1],
            "age": np.full(households, 40.0),
        }
    )
    return Frame(
        tables={
            "person": person,
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "household": household,
        },
        schema=EntitySchema(group_entities=("benunit", "household")),
        weights={
            "household": Weights(
                values=np.full(households, 100.0), kind=WeightKind.DESIGN
            )
        },
    )


def _no_engine(monkeypatch, engine):
    def refuse():
        raise AssertionError("no engine is built")

    monkeypatch.setattr(engine, "_import_policyengine_uk", refuse)


def _record(batches):
    return {
        "by": UK_SPINE_ENGINE_BATCH_COLUMN,
        "batches": batches,
        "pass": SPINE_BATCHED_PASS.name,
        "declaration_sha256": UK_ENGINE_POPULATION.sha256,
    }


def test_whether_a_call_batches_is_decided_from_the_frame(monkeypatch):
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    assert engine.simulation_batches(_clone_frame(column=False)) == 1
    assert engine.simulation_batches(_clone_frame((0,))) == 1
    assert engine.simulation_batches(_clone_frame((0, 1, 2))) == 3
    # The record is the frame's, with no call run: names and the clone count.
    assert engine.batch_record(_clone_frame(column=False)) is None
    assert engine.batch_record(_clone_frame((0,))) is None
    assert engine.batch_record(_clone_frame((0, 1, 2))) == _record(3)


@pytest.mark.parametrize(
    ("expected", "clones", "batches"),
    [
        (3, (0, 1, 2), 3),
        (3, (0,), None),
        (3, (0, 1), None),
        (1, (0,), 1),
        (1, (0, 1), None),
    ],
)
def test_the_spines_clone_count_refuses_any_other_frame(
    monkeypatch, expected, clones, batches
):
    # Part of the pool (one clone, or fewer than the spine's) or another
    # spine's frame is refused before any engine is built; a frame without
    # the column (a stage before the clones exist) runs as it always has.
    engine = PolicyEngineUKEngine(clones=expected)
    assert engine.simulation_batches(_clone_frame(column=False)) == 1
    frame = _clone_frame(clones)
    if batches is not None:
        assert engine.simulation_batches(frame) == batches
        return
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.simulation_batches(frame)
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: pytest.fail()),
    )
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.materialize(frame, ["age"], 2025)


@pytest.mark.parametrize("clones", [0, -1, True, np.bool_(True), 2.0, "3"])
def test_the_spines_clone_count_is_a_positive_whole_number(clones):
    with pytest.raises(ValueError, match="positive whole number"):
        PolicyEngineUKEngine(clones=clones)


@pytest.mark.parametrize("clones", [3, np.int64(3), np.int32(3)])
def test_the_spines_clone_count_may_be_any_whole_number_type(monkeypatch, clones):
    engine = PolicyEngineUKEngine(clones=clones)
    _no_engine(monkeypatch, engine)
    assert engine.simulation_batches(_clone_frame((0, 1, 2))) == 3
    with pytest.raises(ValueError, match="no engine call runs per clone"):
        engine.simulation_batches(_clone_frame((0, 1)))


def test_a_clone_index_with_missing_households_is_refused(monkeypatch):
    engine = PolicyEngineUKEngine()
    _no_engine(monkeypatch, engine)
    with pytest.raises(ValueError, match="clone each belongs to"):
        engine.simulation_batches(_clone_frame((0, 1), missing=True))


def _one_simulation_engine(monkeypatch):
    """An adapter whose engine is a stub: one call builds one simulation."""

    built = []

    class Simulation:
        tax_benefit_system = SimpleNamespace(variables={})
        input_variables = ()

        def __init__(self, dataset):
            built.append(dataset)

        def calculate(self, name, period=None):
            return np.arange(len(built[-1]), dtype=np.float32)

    engine = PolicyEngineUKEngine()
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=Simulation),
    )
    # The engine's year window needs policyengine-uk itself; the stand-in
    # engine is built from the dataset as it is.
    monkeypatch.setattr(
        engine,
        "_read_year_simulation",
        lambda microsimulation_class, dataset, period: microsimulation_class(
            dataset=dataset
        ),
    )
    monkeypatch.setattr(
        engine, "_build_dataset", lambda bundle, period: list(range(bundle.n("person")))
    )
    monkeypatch.setattr(
        engine,
        "variable_metadata",
        lambda name: SimpleNamespace(entity="person"),
    )
    return engine, built


@pytest.mark.parametrize("clones", [None, (0,)])
def test_a_frame_with_one_clone_takes_one_simulation(monkeypatch, clones):
    engine, built = _one_simulation_engine(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("one clone never runs in batches")

    monkeypatch.setattr(adapter_module, "materialize_in_batches", refuse)
    frame = _clone_frame(column=False) if clones is None else _clone_frame(clones)
    values = engine.materialize(frame, ["age"], 2025)
    assert len(built) == 1 and len(built[0]) == frame.n("person")
    np.testing.assert_array_equal(values["age"], np.arange(frame.n("person")))
    assert engine.batch_record(frame) is None


def test_a_frame_with_clones_runs_through_the_shared_batches(monkeypatch):
    engine, built = _one_simulation_engine(monkeypatch)
    calls = []

    def batched(bundle, variables, period, **kwargs):
        calls.append((bundle, variables, period, kwargs))
        return {"age": np.zeros(bundle.n("person"))}

    monkeypatch.setattr(adapter_module, "materialize_in_batches", batched)
    roles = []
    monkeypatch.setattr(
        adapter_module,
        "validate_uc_claimant_input",
        lambda simulation, person, period: roles.append((len(person), period)),
    )
    frame = _clone_frame((0, 1, 2))
    engine.materialize(frame, ["age"], 2025, dataset_period=2023)
    ((bundle, variables, period, kwargs),) = calls
    assert bundle is frame and list(variables) == ["age"] and period == 2025
    assert kwargs["declaration"] is UK_ENGINE_POPULATION
    assert kwargs["policy"] is SPINE_BATCHED_PASS
    assert kwargs["dataset_year"] == 2023
    assert [batch.key for batch in kwargs["batches"]] == [0, 1, 2]
    assert kwargs["entity_of"]("age") == "person"
    # Every batch is held to the adapter's own parameters wherever the UK
    # declaration's restatements read them.
    assert kwargs["reference_parameters"] == engine.parameters
    # Each batch's engine is built from the batch's own frame, at the
    # dataset's year, and its roles are checked against the batch's people.
    batch = frame.select(kwargs["batches"][1].persons)
    kwargs["build"](batch)
    assert built == [list(range(batch.n("person")))]
    assert roles == [(batch.n("person"), 2025)]
    engine.materialize(frame, ["age"], 2025)
    assert calls[-1][3]["dataset_year"] == 2025
    # The calls this adapter ran leave no trace in the record: it is the
    # frame's, as a fresh adapter reads it.
    assert engine.batch_record(frame) == _record(3)
    assert PolicyEngineUKEngine().batch_record(frame) == _record(3)
    with pytest.raises(ValueError, match="projects a dataset forward only"):
        engine.materialize(frame, ["age"], 2023, dataset_period=2025)


def test_the_parameters_are_the_engines_own():
    engine = PolicyEngineUKEngine()
    engine._system = SimpleNamespace(parameters=lambda period: ("tree at", period))
    assert engine.parameters(2025) == ("tree at", 2025)
