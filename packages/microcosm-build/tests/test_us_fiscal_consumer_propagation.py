"""Record consumer construction without calculating invented policy results."""

import builtins
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_us_fiscal_refresh_builder import _load_builder_module, _multi_reform_frame

from microcosm.build.us_runtime import reform_validation


class StopBeforeCalculationError(Exception):
    pass


class Metadata:
    def __init__(self, denied=()):
        self.denied = set(denied)
        self.calls = []

    def _engine_computed_columns(self, tables, *, period):
        self.calls.append((tables, period))
        return self.denied


@pytest.fixture
def builder(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name == "policyengine_us" or name.startswith("policyengine_us.")
        ) and not isinstance(sys.modules.get(name), SimpleNamespace):
            raise AssertionError("Actual country imports are forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return _load_builder_module()


class _NoParameters:
    """Invented parameter tree without behavioral-response leaves."""

    def get_child(self, path):
        return SimpleNamespace(get_descendants=lambda: ())


def _invented_engine_type(builder, log, *, stop_when):
    """Record each written-H5 batch construction; stop where ``stop_when`` says.

    A construction that does not stop scores every key as one invented weighted
    zero per household batch and computes no watched aggregate. These are
    array-routing stand-ins, not policy results.
    """

    class Simulation:
        default_tax_benefit_system_instance = SimpleNamespace(
            parameters=_NoParameters()
        )

        @staticmethod
        def default_tax_benefit_system(**kwargs):
            log.systems.append(kwargs)
            return SimpleNamespace(parameters=_NoParameters())

        def __init__(self, **kwargs):
            log.simulations.append(kwargs)
            if stop_when(kwargs):
                raise StopBeforeCalculationError

        def calculate(self, variable, period, map_to=None):
            return builder._PostExportValues(np.zeros(1), np.ones(1))

        def get_holder(self, variable):
            return SimpleNamespace(get_known_periods=lambda: [])

    return Simulation


def _written_h5(tmp_path):
    path = tmp_path / "invented.h5"
    path.write_bytes(b"invented written release bytes")
    return path


def _frame_reader(builder, frame, constructors):
    """An injected dataset class that reads ``frame`` back from a file path."""

    def dataset(**kwargs):
        constructors.datasets.append(kwargs)
        if set(kwargs) != {"file_path"}:
            return SimpleNamespace(**kwargs)
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["household_weight"] = frame.weights_for("household").values
        return SimpleNamespace(**tables)

    return dataset


class Constructors:
    def __init__(self):
        self.datasets = []
        self.simulations = []
        self.systems = []

    def dataset(self, **kwargs):
        self.datasets.append(kwargs)
        return SimpleNamespace(**kwargs)

    def system(self, **kwargs):
        self.systems.append(kwargs)
        return SimpleNamespace(variables={})

    def simulation(self, **kwargs):
        self.simulations.append(kwargs)
        return SimpleNamespace()


def test_materializer_uses_explicit_constructors_and_spm(builder, monkeypatch):
    frame = _multi_reform_frame(builder)
    supplied = {"geography_kind": "national"}
    constructors = Constructors()
    provider = Metadata()

    def stop(*args, **kwargs):
        raise StopBeforeCalculationError

    monkeypatch.setattr(builder, "_calculate_array", stop)
    with pytest.raises(StopBeforeCalculationError):
        builder._materialize_target_frame(
            frame,
            (),
            formula_metadata=provider,
            dataset_cls=constructors.dataset,
            microsimulation_cls=constructors.simulation,
            system_factory=constructors.system,
            spm=supplied,
        )
    assert len(constructors.datasets) == len(constructors.simulations) == 1
    assert constructors.systems == [{"spm": supplied}]
    assert constructors.simulations[0]["spm"] == supplied
    assert constructors.simulations[0]["spm"] is not supplied
    assert constructors.systems[0]["spm"] is not supplied
    assert constructors.systems[0]["spm"] is not constructors.simulations[0]["spm"]
    assert len(provider.calls) == 1


def test_explicit_jct_factory_materializes_without_default_country_import(
    builder, monkeypatch
):
    from test_us_fiscal_refresh_builder import _install_multi_reform_fakes

    frame = _multi_reform_frame(builder)
    calls = []
    _, targets = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=(("jct_test_credit", "test_credit"),),
        reform_income_tax_by_id={"test_credit": {10: 90.0, 20: 25.0, 30: 40.0}},
        reform_sim_calls=[],
    )
    constructors = sys.modules["policyengine_us"]
    monkeypatch.delitem(sys.modules, "policyengine_us")

    def forbidden(*args, **kwargs):
        raise AssertionError("Default reform factory must not be used")

    monkeypatch.setattr(builder, "_make_zero_variable_reform", forbidden)

    def make_reform(system, variable):
        assert type(system) is constructors.CountryTaxBenefitSystem
        calls.append(variable)
        return variable

    result, registry, compilation = builder._materialize_target_frame(
        frame,
        targets,
        maximum_microsim_batch_size=1,
        formula_metadata=Metadata(),
        dataset_cls=Constructors().dataset,
        microsimulation_cls=constructors.Microsimulation,
        system_factory=constructors.CountryTaxBenefitSystem,
        zero_variable_reform_factory=make_reform,
    )
    assert calls == ["test_credit"]
    assert compilation["dropped_target_names"] == []
    assert len(registry.specs) == 1
    # Invented array routing through three tax units and two household batches;
    # these are not policy results from a substitute engine.
    np.testing.assert_array_equal(
        result.table("household").jct_test_credit, [-15.0, -30.0]
    )


def test_zero_variable_reform_uses_explicit_base_classes(builder):
    class Variable:
        pass

    class Reform:
        def replace_variable(self, value):
            self.replaced = value

    original = SimpleNamespace(
        value_type=float, entity=object(), definition_period="year", unit="currency"
    )
    reform = builder._make_zero_variable_reform(
        SimpleNamespace(variables={"invented_credit": original}),
        "invented_credit",
        reform_cls=Reform,
        variable_cls=Variable,
    )
    assert issubclass(reform, Reform)
    instance = reform()
    instance.apply()
    variable = instance.replaced
    assert issubclass(variable, Variable)
    assert variable.__name__ == "invented_credit"
    assert variable.entity is original.entity
    assert variable.value_type is float
    assert variable.definition_period == "year"
    assert variable.formula(None, None, None) == 0
    assert variable.adds is variable.subtracts is variable.uprating is None


@pytest.mark.parametrize(
    "entry,options",
    [
        ("_materialize_target_frame", {"target_materialization_cache_dir": "/unused"}),
        (
            "_load_or_materialize_target_frame",
            {"target_frame_checkpoint_path": "/unused"},
        ),
    ],
)
def test_explicit_reform_factory_refuses_unbound_disk_caches(builder, entry, options):
    with pytest.raises(ValueError, match="target caches disabled"):
        getattr(builder, entry)(
            None, (), zero_variable_reform_factory=lambda *_: None, **options
        )


def test_materializer_refuses_before_consumer_constructors(builder):
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    with pytest.raises(ValueError, match="Formula-owned"):
        builder._materialize_target_frame(
            frame,
            (),
            formula_metadata=Metadata({"invented_output"}),
            dataset_cls=constructors.dataset,
            microsimulation_cls=constructors.simulation,
            system_factory=constructors.system,
            spm={"geography_kind": "national"},
        )
    assert (
        not constructors.datasets
        and not constructors.simulations
        and not constructors.systems
    )


def test_written_file_factory_copies_spm_for_baseline_and_reform(tmp_path):
    constructors = Constructors()
    supplied = {"geography_kind": "national"}
    simulate = reform_validation.default_simulate_factory(
        tmp_path / "invented.h5",
        dataset_cls=constructors.dataset,
        microsimulation_cls=constructors.simulation,
        spm=supplied,
    )
    supplied["geography_kind"] = "changed-after-binding"
    simulate(None)
    constructors.simulations[0]["spm"]["geography_kind"] = "changed-by-first-consumer"
    reform = object()
    simulate(reform)
    assert constructors.simulations[1]["spm"] == {"geography_kind": "national"}
    assert constructors.simulations[1]["reform"] is reform
    assert constructors.datasets == [{"file_path": str(tmp_path / "invented.h5")}] * 2


def test_written_file_factory_omission_preserves_county_default(tmp_path):
    constructors = Constructors()
    simulate = reform_validation.default_simulate_factory(
        tmp_path / "invented.h5",
        dataset_cls=constructors.dataset,
        microsimulation_cls=constructors.simulation,
    )
    simulate(None)
    assert constructors.simulations[0]["spm"] == {"geography_kind": "county"}


def test_written_file_factory_preserves_explicit_empty_selection(tmp_path):
    constructors = Constructors()
    simulate = reform_validation.default_simulate_factory(
        tmp_path / "invented.h5",
        dataset_cls=constructors.dataset,
        microsimulation_cls=constructors.simulation,
        spm={},
    )
    simulate(None)
    assert constructors.simulations[0]["spm"] == {}


@pytest.mark.parametrize("changed", [False, True])
def test_readback_injected_dataset_preserves_file_identity_checks(
    builder, tmp_path, changed
):
    frame = _multi_reform_frame(builder)
    dataset_path = tmp_path / "invented.h5"
    dataset_path.write_bytes(b"invented dataset identity")
    expected = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    constructed = []

    def dataset(**kwargs):
        constructed.append(kwargs)
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["household_weight"] = frame.weights_for("household").values
        if changed:
            dataset_path.write_bytes(b"changed by invented reader")
        return SimpleNamespace(**tables)

    if changed:
        with pytest.raises(ValueError, match="changed"):
            builder._load_frame(
                dataset_path, expected_sha256=expected, dataset_cls=dataset
            )
    else:
        restored = builder._load_frame(
            dataset_path, expected_sha256=expected, dataset_cls=dataset
        )
        assert restored.table("household").equals(frame.table("household"))
        np.testing.assert_array_equal(
            restored.weights_for("household").values,
            frame.weights_for("household").values,
        )
    assert constructed == [{"file_path": str(dataset_path)}]


def test_readback_identity_mismatch_refuses_before_injected_constructor(
    builder, tmp_path
):
    dataset_path = tmp_path / "invented.h5"
    dataset_path.write_bytes(b"invented dataset identity")
    constructors = Constructors()
    with pytest.raises(ValueError, match="not the base dataset"):
        builder._load_frame(
            dataset_path, expected_sha256="0" * 64, dataset_cls=constructors.dataset
        )
    assert not constructors.datasets


@pytest.mark.parametrize("spm", [None, {}, {"geography_kind": "national"}])
@pytest.mark.parametrize("household_position", [0, 1])
def test_each_reform_batch_receives_explicit_consumer(
    builder, monkeypatch, spm, household_position
):
    """Visit each original household batch separately, stopping at construction."""
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    reform = object()
    reform_system = object()

    class Simulation:
        @staticmethod
        def default_tax_benefit_system(**kwargs):
            constructors.systems.append(kwargs)
            return reform_system

        def __init__(self, **kwargs):
            constructors.simulations.append(kwargs)
            raise StopBeforeCalculationError

    monkeypatch.setattr(builder, "_make_zero_variable_reform", lambda *args: reform)
    monkeypatch.setattr(
        builder,
        "_household_position_batches",
        lambda *args: (np.asarray([household_position]),),
    )
    with pytest.raises(StopBeforeCalculationError):
        builder._reform_household_income_tax(
            base_frame=frame,
            reform_spec=SimpleNamespace(neutralized_variable="invented"),
            system=SimpleNamespace(variables={}),
            microsimulation_cls=Simulation,
            n_households=2,
            batch_size=1,
            formula_metadata=Metadata(),
            dataset_cls=constructors.dataset,
            spm=spm,
        )
    assert constructors.systems == [
        {"reform": reform, **({} if spm is None else {"spm": spm})}
    ]
    call = constructors.simulations[0]
    assert call["reform"] is reform and call["tax_benefit_system"] is reform_system
    assert constructors.datasets[0]["household"]["household_id"].tolist() == [
        household_position + 1
    ]
    assert ("spm" in call) is (spm is not None)
    if spm is not None:
        assert call["spm"] == spm and call["spm"] is not spm
        assert constructors.systems[0]["spm"] is not spm
        assert constructors.systems[0]["spm"] is not call["spm"]


@pytest.mark.parametrize("reformed", [False, True])
@pytest.mark.parametrize("household_position", [0, 1])
def test_validation_batches_capture_independent_spm_selection(
    builder, tmp_path, reformed, household_position
):
    """The written-H5 scorer gives each batch engine the explicit consumer.

    Route A (microcosm#956) replaced the frame-based reform-validation factory
    with the household-batched post-export scorer, so the explicit
    constructors and SPM selection now reach that scorer's batch engines. The
    scorer must partition the pool, so each original household batch is
    visited by stopping at its construction; the other batch runs an invented
    engine. A reform engine receives the reform's system alone (no
    ``reform=``), which carries the explicit selection too.
    """
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    supplied = {"geography_kind": "national"}
    reform = object() if reformed else None
    target = household_position + 1
    simulation_type = _invented_engine_type(
        builder,
        constructors,
        stop_when=lambda kwargs: (
            kwargs["dataset"].household["household_id"].tolist() == [target]
        ),
    )
    scorer = builder._HouseholdBatchedPostExportScorer(
        _written_h5(tmp_path),
        maximum_microsim_batch_size=1,
        load_frame=lambda path, *, expected_sha256: frame,
        formula_metadata=Metadata(),
        dataset_cls=constructors.dataset,
        microsimulation_cls=simulation_type,
        spm=supplied,
    )
    supplied["geography_kind"] = "changed-after-binding"
    key = ("never_calculated", 2024, None)
    if reformed:
        wrapper = scorer.open_consumer("reform_validation", ()).simulate(reform)

        def attempt():
            wrapper.calculate("never_calculated", 2024)

    else:

        def attempt():
            scorer.open_consumer("reform_validation", (key,))

    with pytest.raises(StopBeforeCalculationError):
        attempt()
    call = constructors.simulations[-1]
    assert len(constructors.simulations) == target
    assert call["spm"] == {"geography_kind": "national"}
    assert "reform" not in call
    if reformed:
        assert constructors.systems == [
            {"reform": reform, "spm": {"geography_kind": "national"}}
        ]
        assert constructors.systems[0]["spm"] is not call["spm"]
        assert call["tax_benefit_system"].parameters is not None
    else:
        assert not constructors.systems
        assert "tax_benefit_system" not in call
    assert constructors.datasets[-1]["household"]["household_id"].tolist() == [target]
    # A consumer may mutate its own options even when construction fails. The
    # same scorer must retain its captured selection for the next attempt.
    call["spm"]["geography_kind"] = "changed-by-first-consumer"
    with pytest.raises(StopBeforeCalculationError):
        attempt()
    retried = constructors.simulations[-1]
    assert retried["spm"] == {"geography_kind": "national"}
    assert retried["spm"] is not call["spm"]
    if reformed:
        assert len(constructors.systems) == 1
        assert constructors.systems[0]["spm"] == {"geography_kind": "national"}
        assert constructors.systems[0]["spm"] is not retried["spm"]
    record = scorer.manifest_record()
    assert record["spm"] == {"geography_kind": "national"}


@pytest.mark.parametrize("helper", ["aca", "ssi"])
def test_source_amount_helpers_forward_before_any_calculation(builder, helper):
    constructors = Constructors()
    frame = _multi_reform_frame(builder)
    spm = {"geography_kind": "county"}

    def simulation(**kwargs):
        constructors.simulations.append(kwargs)
        raise StopBeforeCalculationError

    common = dict(
        formula_metadata=Metadata(),
        dataset_cls=constructors.dataset,
        microsimulation_cls=simulation,
        maximum_microsim_batch_size=1,
        spm=spm,
    )
    with pytest.raises(StopBeforeCalculationError):
        if helper == "aca":
            builder._aca_source_tax_unit_table_batched(frame, {}, **common)
        else:
            builder._ssi_person_uncapped_amount(frame, **common)
    assert len(constructors.datasets) == 1
    assert constructors.simulations[0]["spm"] == spm
    assert constructors.simulations[0]["spm"] is not spm


@pytest.mark.parametrize("helper", ["aca", "ssi", "validation", "cached"])
def test_provider_refusal_precedes_injected_execution(builder, tmp_path, helper):
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    common = dict(
        formula_metadata=Metadata({"invented_output"}),
        dataset_cls=constructors.dataset,
        microsimulation_cls=constructors.simulation,
        spm={"geography_kind": "national"},
    )
    with pytest.raises(ValueError, match="Formula-owned"):
        if helper == "aca":
            builder._aca_source_tax_unit_table_batched(
                frame, {}, maximum_microsim_batch_size=1, **common
            )
        elif helper == "ssi":
            builder._ssi_person_uncapped_amount(frame, **common)
        elif helper == "validation":
            # The written-H5 scorer that replaced the frame-based factory.
            builder._HouseholdBatchedPostExportScorer(
                _written_h5(tmp_path),
                maximum_microsim_batch_size=1,
                load_frame=lambda path, *, expected_sha256: frame,
                **common,
            )
        else:
            builder._load_or_materialize_target_frame(
                frame, (), system_factory=constructors.system, **common
            )
    assert (
        not constructors.datasets
        and not constructors.simulations
        and not constructors.systems
    )


@pytest.mark.parametrize("readback", [False, True])
def test_target_wrappers_forward_captured_configuration(
    builder, monkeypatch, tmp_path, readback
):
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    provider = Metadata()
    spm = {"geography_kind": "national"}
    seen = []

    def load(path, *, dataset_cls):
        assert dataset_cls == constructors.dataset
        spm["geography_kind"] = "changed-during-readback"
        return frame

    def materialize(frame_arg, specs, **kwargs):
        seen.append(kwargs)
        assert frame_arg is frame
        raise StopBeforeCalculationError

    monkeypatch.setattr(builder, "_load_frame", load)
    monkeypatch.setattr(builder, "_materialize_target_frame", materialize)
    kwargs = dict(
        dataset_cls=constructors.dataset,
        microsimulation_cls=constructors.simulation,
        system_factory=constructors.system,
        formula_metadata=provider,
        spm=spm,
    )
    with pytest.raises(StopBeforeCalculationError):
        if readback:
            builder._assert_export_matches_calibration(
                tmp_path / "invented.h5", None, (), **kwargs
            )
        else:
            builder._load_or_materialize_target_frame(frame, (), **kwargs)
    assert seen[0]["spm"] == {"geography_kind": "national"}
    assert seen[0]["spm"] is not spm
    for name in (
        "dataset_cls",
        "microsimulation_cls",
        "system_factory",
        "formula_metadata",
    ):
        assert seen[0][name] == kwargs[name]


def test_demographics_forwards_selection_without_computing(
    builder, monkeypatch, tmp_path
):
    """Demographics reads and scores the written H5 with the explicit consumer.

    The written file is read back through the supplied dataset class, then
    each household batch's engine receives an independent copy of the
    explicit selection; construction stops before any calculation.
    """
    constructors = Constructors()
    spm = {"geography_kind": "national"}
    dataset_path = _written_h5(tmp_path)
    frame = _multi_reform_frame(builder)
    simulation_type = _invented_engine_type(
        builder, constructors, stop_when=lambda kwargs: True
    )
    with pytest.raises(StopBeforeCalculationError):
        builder._write_demographics(
            release_dir=tmp_path,
            dataset_path=dataset_path,
            release_id="invented",
            dataset_cls=_frame_reader(builder, frame, constructors),
            microsimulation_cls=simulation_type,
            spm=spm,
        )
    assert constructors.simulations[0]["spm"] == spm
    assert constructors.simulations[0]["spm"] is not spm
    assert constructors.datasets[0] == {"file_path": str(dataset_path)}
    assert constructors.datasets[1]["household"]["household_id"].tolist() == [1, 2]
    assert len(constructors.simulations) == 1


def test_reform_report_forwards_written_file_consumer_without_scores(
    builder, monkeypatch, tmp_path
):
    """Reform validation scores the written H5 with the explicit consumer.

    The selection is captured before spec loading. The baseline engine and
    the reform's system and engine each receive it; the reform engine gets
    the reform's system alone, as the household-batched scorer builds it.
    """
    constructors = Constructors()
    spm = {"geography_kind": "national"}
    dataset_path = _written_h5(tmp_path)
    frame = _multi_reform_frame(builder)

    def load_specs(**kwargs):
        spm["geography_kind"] = "changed-during-spec-loading"
        return ()

    def payload(specs, *, simulate, **kwargs):
        # The same requests on the engine-free dry run and the scored run.
        simulate(None).calculate("invented_measure", builder.PERIOD)
        simulate("invented reform marker").calculate("invented_measure", builder.PERIOD)
        return {}

    simulation_type = _invented_engine_type(
        builder,
        constructors,
        stop_when=lambda kwargs: "tax_benefit_system" in kwargs,
    )
    monkeypatch.setattr(builder, "load_default_reform_specs", load_specs)
    monkeypatch.setattr(builder, "default_baseline_level_specs", lambda: ())
    monkeypatch.setattr(builder, "_in_sample_estimates", lambda result: {})
    monkeypatch.setattr(builder, "_in_sample_targets", lambda result: {})
    monkeypatch.setattr(builder, "reform_validation_payload", payload)
    with pytest.raises(StopBeforeCalculationError):
        builder._write_reform_validation(
            release_dir=tmp_path,
            dataset_path=dataset_path,
            release_id="invented",
            result=None,
            registry=None,
            simulate_out_of_sample=True,
            dataset_cls=_frame_reader(builder, frame, constructors),
            microsimulation_cls=simulation_type,
            spm=spm,
        )
    assert [call["spm"] for call in constructors.simulations] == [
        {"geography_kind": "national"}
    ] * 2
    assert constructors.systems == [
        {"reform": "invented reform marker", "spm": {"geography_kind": "national"}}
    ]
    reformed = constructors.simulations[1]
    assert "reform" not in reformed
    assert reformed["tax_benefit_system"].parameters is not None
    assert constructors.datasets[0] == {"file_path": str(dataset_path)}


def test_materializer_omission_preserves_constructor_call_shape(builder, monkeypatch):
    constructors = Constructors()
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=constructors.system,
            Microsimulation=constructors.simulation,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.data",
        SimpleNamespace(USSingleYearDataset=constructors.dataset),
    )

    def stop(*args, **kwargs):
        raise StopBeforeCalculationError

    monkeypatch.setattr(builder, "_calculate_array", stop)
    with pytest.raises(StopBeforeCalculationError):
        builder._materialize_target_frame(
            _multi_reform_frame(builder), (), formula_metadata=Metadata()
        )
    assert set(constructors.simulations[0]) == {"dataset"}
    assert constructors.systems == [{}]


def test_existing_ssi_simulation_refuses_unapplied_overrides(builder):
    with pytest.raises(ValueError, match="existing SSI simulation"):
        builder._ssi_person_uncapped_amount(
            None, simulation=object(), spm={"geography_kind": "national"}
        )


def test_custom_dataset_callback_refuses_competing_class(builder, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        builder._HouseholdBatchedPostExportScorer(
            tmp_path / "never-read.h5",
            maximum_microsim_batch_size=None,
            dataset_from_frame=lambda frame: frame,
            dataset_cls=object,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"dataset_cls": object},
        {"microsimulation_cls": object},
        {"spm": {}},
    ],
)
def test_existing_post_export_scorer_refuses_unapplied_overrides(builder, override):
    """A shared scorer was already constructed; a writer cannot re-bind it."""

    def forbidden(simulate):
        raise AssertionError("an override must refuse before any scoring")

    with pytest.raises(ValueError, match="existing post-export scorer"):
        builder._score_post_export_consumer(
            "invented",
            forbidden,
            dataset_path=None,
            post_export_scorer=object(),
            baseline_plan=(),
            maximum_microsim_batch_size=None,
            **override,
        )


def test_scorer_omission_preserves_release_selection_and_system_call(builder, tmp_path):
    """Without explicit seams the scorer's calls are exactly the release's."""
    frame = _multi_reform_frame(builder)
    constructors = Constructors()
    simulation_type = _invented_engine_type(
        builder,
        constructors,
        stop_when=lambda kwargs: "tax_benefit_system" in kwargs,
    )
    scorer = builder._HouseholdBatchedPostExportScorer(
        _written_h5(tmp_path),
        maximum_microsim_batch_size=None,
        load_frame=lambda path, *, expected_sha256: frame,
        dataset_from_frame=lambda batch_frame: batch_frame,
        microsimulation_cls=simulation_type,
    )
    consumer = scorer.open_consumer("invented", (("invented_measure", 2024, None),))
    with pytest.raises(StopBeforeCalculationError):
        consumer.simulate("invented reform").calculate("invented_measure", 2024)
    assert [call["spm"] for call in constructors.simulations] == [
        dict(builder.US_RELEASE_SPM_SELECTION)
    ] * 2
    assert constructors.systems == [{"reform": "invented reform"}]
    assert scorer.manifest_record()["spm"] == dict(builder.US_RELEASE_SPM_SELECTION)
