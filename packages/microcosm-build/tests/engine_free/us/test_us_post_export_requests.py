"""Post-export requests recorded as picklable descriptors.

The engine-free dry run that learns each post-export consumer's baseline plan
also returns every request the consumer makes as a ``PostExportRequest``, so
that later work can send the requests to spawned worker processes and rebuild
each consumer's output from their results. These tests check that:

- the dry run returns one descriptor per request of every shipped consumer,
  with no engine importable, and builds no reform: each reform is recorded as
  the description its consumer builds it from;
- the descriptors list the requests in the order the serial scorer runs them
  (the baseline plan, then each reform pass), name the reformed simulation
  that asked for each, and rebuild the reform that simulation was given;
- a ``prefetch`` of several keys is one reform pass, and its keys not asked
  before follow in ascending period order, as the scorer computes them;
- every descriptor survives a pickle round trip through a spawned process;
- the dry run refuses a reform it cannot describe or pickle;
- descriptors cannot be hashed and keep their own copy of a reform's
  parameter changes;
- the recorded outputs of the release stay byte-identical on the fixture.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import multiprocessing
import os
import pickle
import re
import threading
import types
import typing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from types import MappingProxyType

import microcosm.build.us_runtime.release_input_coverage as coverage_module
from microcosm.build.us_runtime.post_export_requests import (
    RECORDS_REQUESTS_ONLY,
    ParameterChangesReform,
    PostExportReform,
    PostExportRequest,
    build_post_export_reform,
)
from test_support.microcosm_build.post_export_request_worker import (
    requests_as_a_spawned_worker_receives_them,
)
from test_support.microcosm_build.us_post_export_scoring import *

_RELEASE_ID = "fixture-release"


def _shipped_consumers(builder):
    """The five post-export consumers a release runs, in run order."""
    return builder._post_export_consumers(
        _plan_args(),
        result=_calibration_result_with_in_sample_fit(builder.PERIOD),
        release_id=_RELEASE_ID,
    )


@pytest.fixture
def no_engine(monkeypatch):
    """No engine package importable, no list edit resolvable, no reform built."""
    for module in ("policyengine_us", "policyengine_core", "policyengine_core.reforms"):
        monkeypatch.setitem(sys.modules, module, None)

    def refused(*args, **kwargs):
        raise AssertionError("the engine-free dry run built a reform")

    # The installed-engine reader is cached per process, so blocking the
    # imports alone would not catch a resolution served from an earlier test.
    monkeypatch.setattr(coverage_module, "_installed_parameter_stages", refused)
    monkeypatch.setattr(smoke_module, "_build_reform", refused)
    monkeypatch.setattr(reform_validation_module, "_build_parameter_reform", refused)
    monkeypatch.setattr(ReformValidationSpec, "build_reform", refused)


def test_the_dry_run_returns_every_request_as_a_descriptor(builder, no_engine) -> None:
    probes = us_release_reform_coverage_probes()
    specs = reform_validation_module.load_default_reform_specs(period=builder.PERIOD)
    fitted = {
        spec.id
        for spec in reform_validation_module.in_sample_reform_specs(
            period=builder.PERIOD
        )[:2]
    }
    obbba_groups = {
        (spec.budget_measure, spec.period)
        for spec in specs
        if reform_validation_module._is_obbba_spec(spec)
    }
    recorded: dict[str, tuple[PostExportRequest, ...]] = {}
    for name, consumer in _shipped_consumers(builder).items():
        requests = builder._record_post_export_request_descriptors(name, consumer)
        assert all(isinstance(request, PostExportRequest) for request in requests)
        assert {request.consumer for request in requests} <= {name}
        assert [request.position for request in requests] == list(range(len(requests)))
        baseline = tuple(request for request in requests if request.reform is None)
        reformed = tuple(request for request in requests if request.reform is not None)
        # The baseline plan first, as the serial scorer scores it.
        assert requests == (*baseline, *reformed)
        assert all(request.reformed_simulation is None for request in baseline)
        assert all(isinstance(request.reformed_simulation, int) for request in reformed)
        # The plan and the pass count the release records are the same dry
        # run's.
        plan, passes = builder._record_post_export_requests(consumer)
        assert tuple(request.key for request in baseline) == plan
        # One pass per reform request, except reform validation's pre-OBBBA
        # state: one prefetch reads it for every OBBBA group in one pass
        # (microcosm#1595).
        shared_pass_requests = (
            len(obbba_groups) - 1 if name == "reform_validation" else 0
        )
        assert len(reformed) == passes + shared_pass_requests
        recorded[name] = requests
    assert list(recorded) == [
        "reform_coverage_smoke",
        "reform_validation",
        "demographics",
        "ssi_social_security_holdout",
        "medicare_irmaa_holdout",
    ]

    smoke = [
        request
        for request in recorded["reform_coverage_smoke"]
        if request.reform is not None
    ]
    assert [request.reform for request in smoke] == list(probes)
    assert [request.reformed_simulation for request in smoke] == list(
        range(len(probes))
    )
    assert [request.key for request in smoke] == [
        (probe.budget_measure, int(probe.period or builder.PERIOD), None)
        for probe in probes
    ]

    validation = [
        request
        for request in recorded["reform_validation"]
        if request.reform is not None
    ]
    assert {type(request.reform) for request in validation} == {
        ReformValidationSpec,
        ParameterChangesReform,
    }
    # Every spec simulated on its own: the out-of-sample ones outside OBBBA's
    # stacks, and the in-sample ones the calibration fit does not cover.
    assert [
        request.reform.id
        for request in validation
        if isinstance(request.reform, ReformValidationSpec)
    ] == [
        spec.id
        for spec in specs
        if not reform_validation_module._is_obbba_spec(spec)
        and not (spec.in_sample and spec.id in fitted)
    ]
    # OBBBA: every (measure, period) group reads the pre-OBBBA state, then one
    # stacked state per provision.
    obbba = [spec for spec in specs if reform_validation_module._is_obbba_spec(spec)]
    states = [
        request
        for request in validation
        if isinstance(request.reform, ParameterChangesReform)
    ]
    groups = {(spec.budget_measure, spec.period) for spec in obbba}
    assert len(states) == len(obbba) + len(groups)
    pre_obbba = ParameterChangesReform(
        reform_validation_module._merged_parameter_changes(obbba)
    )
    assert [request.key[:2] for request in states if request.reform == pre_obbba] == [
        (measure, period)
        for measure, period in dict.fromkeys(
            (spec.budget_measure, spec.period) for spec in obbba
        )
    ]
    # One reformed simulation per request, except the pre-OBBBA state: every
    # group's key comes from one simulation of it (microcosm#1595).
    assert (
        len(
            {
                request.reformed_simulation
                for request in states
                if request.reform == pre_obbba
            }
        )
        == 1
    )
    assert len({request.reformed_simulation for request in validation}) == (
        len(validation) - len(groups) + 1
    )

    for name in (
        "demographics",
        "ssi_social_security_holdout",
        "medicare_irmaa_holdout",
    ):
        assert all(request.reform is None for request in recorded[name]), name


def test_every_request_round_trips_through_a_spawned_process(
    builder, no_engine
) -> None:
    """A worker started with the ``spawn`` method, which has never run the
    release tool, unpickles every shipped request and sends it back equal."""
    requests = tuple(
        request
        for name, consumer in _shipped_consumers(builder).items()
        for request in builder._record_post_export_request_descriptors(name, consumer)
    )
    assert any(request.reform is not None for request in requests)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        received = executor.submit(
            requests_as_a_spawned_worker_receives_them, requests
        ).result()
    assert received["start_method"] == "spawn"
    assert received["process_id"] != os.getpid()
    assert received["requests"] == requests
    assert received["type_modules"] == [
        "microcosm.build.us_runtime.post_export_requests",
        "microcosm.build.us_runtime.reform_validation",
        "microcosm.build.us_runtime.release_input_coverage",
    ]


def _watched_simulate(builder, simulate, asked: list):
    """``simulate``, logging each reformed simulation's first ask per key.

    Each entry is (the reformed simulation's number among the consumer's
    ``simulate(reform)`` calls, the reform's label, the key).
    """
    numbers = itertools.count()

    def watched(reform):
        simulation = simulate(reform)
        if reform is None:
            return simulation
        number = next(numbers)

        class Watched:
            def calculate(self, variable, period=None, map_to=None):
                entry = (
                    number,
                    reform.label,
                    builder._post_export_key(variable, period, map_to),
                )
                if entry not in asked:
                    asked.append(entry)
                return simulation.calculate(variable, period, map_to)

        return Watched()

    return watched


def test_the_descriptors_list_the_passes_the_serial_scorer_runs(
    builder, fake_reforms, tmp_path
) -> None:
    """Each consumer, scored as ``_main`` scores it, runs its baseline plan and
    then one reform pass per reform descriptor, in descriptor order, on the
    reformed simulation each descriptor names, with the reform it rebuilds."""
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    try:
        for name, consumer in _shipped_consumers(builder).items():
            requests = builder._record_post_export_request_descriptors(name, consumer)
            baseline = tuple(r.key for r in requests if r.reform is None)
            reformed = [r for r in requests if r.reform is not None]
            scoring = scorer.open_consumer(
                name, baseline, planned_reform_passes=len(reformed)
            )
            assert scoring.baseline_plan == baseline
            asked: list = []
            consumer(_watched_simulate(builder, scoring.simulate, asked))
            assert asked == [
                (request.reformed_simulation, request.build_reform().label, request.key)
                for request in reformed
            ], name
            assert scorer.finish_consumer(scoring)["reform_passes"] == len(reformed)
    finally:
        scorer.close()


def test_reform_requests_follow_the_order_of_first_asks(
    builder, fake_reforms, tmp_path
) -> None:
    """A reform pass runs when a reformed simulation is first asked for a key,
    so the descriptors follow those first asks. The scorer finishes a
    reformed simulation when the consumer opens the next one and refuses it
    further keys, so a consumer asks each reformed simulation for all its
    keys before opening the next; a simulation never asked makes no request
    but keeps its number."""
    first = ParameterChangesReform({"gov.fixture.first": {"2024-01-01": 1}})
    second = ParameterChangesReform({"gov.fixture.second": {"2024-01-01": 2}})
    unasked = ParameterChangesReform({"gov.fixture.unasked": {"2024-01-01": 3}})

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True

        def reformed(source):
            return simulate(
                source if records_only else build_post_export_reform(source)
            )

        first_simulation = reformed(first)
        simulate(None).calculate("income_tax", 2025)
        first_simulation.calculate("income_tax", 2024)
        first_simulation.calculate("income_tax", 2024)
        first_simulation.calculate("age", 2024, map_to="household")
        reformed(unasked)
        second_simulation = reformed(second)
        simulate(None).calculate("age", 2024, map_to="household")
        second_simulation.calculate("income_tax", 2024)

    requests = builder._record_post_export_request_descriptors("fixture", consumer)
    assert [(r.position, r.reformed_simulation, r.reform, r.key) for r in requests] == [
        (0, None, None, ("age", 2024, "household")),
        (1, None, None, ("income_tax", 2025, None)),
        (2, 0, first, ("income_tax", 2024, None)),
        (3, 0, first, ("age", 2024, "household")),
        (4, 2, second, ("income_tax", 2024, None)),
    ]
    assert builder._record_post_export_requests(consumer) == (
        (("age", 2024, "household"), ("income_tax", 2025, None)),
        3,
    )

    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    try:
        baseline = tuple(r.key for r in requests if r.reform is None)
        scoring = scorer.open_consumer("fixture", baseline)
        asked: list = []
        consumer(_watched_simulate(builder, scoring.simulate, asked))
        assert asked == [
            (r.reformed_simulation, r.build_reform().label, r.key)
            for r in requests
            if r.reform is not None
        ]
    finally:
        scorer.close()


@pytest.mark.parametrize(
    "late_ask",
    [("income_tax", 2024), ("age", 2024)],
    ids=["a key it already asked", "a key it did not ask"],
)
def test_the_dry_run_refuses_a_reformed_simulation_asked_after_the_next_opens(
    builder, fake_reforms, tmp_path, late_ask
) -> None:
    """The scorer finishes a reformed simulation when the consumer opens the
    next one and refuses it any further key, even one it already scored. The
    dry run refuses the same consumer, so the release's plan fails before the
    export instead of certification's scoring failing after it."""
    first = ParameterChangesReform({"gov.fixture.first": {"2024-01-01": 1}})
    second = ParameterChangesReform({"gov.fixture.second": {"2024-01-01": 2}})

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True

        def reformed(source):
            return simulate(
                source if records_only else build_post_export_reform(source)
            )

        first_simulation = reformed(first)
        first_simulation.calculate("income_tax", 2024)
        simulate(None).calculate("income_tax", 2024)
        reformed(second).calculate("income_tax", 2024)
        first_simulation.calculate(*late_ask)

    refusal = r"reformed simulation 0 for .* after opening reformed simulation 1\."
    with pytest.raises(RuntimeError, match=refusal):
        builder._record_post_export_request_descriptors("fixture", consumer)
    with pytest.raises(RuntimeError, match=refusal):
        builder._record_post_export_requests(consumer)

    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    try:
        scoring = scorer.open_consumer("fixture", (("income_tax", 2024, None),))
        with pytest.raises(RuntimeError, match="asked a finished reform"):
            consumer(scoring.simulate)
    finally:
        scorer.close()


def _prefetching_consumer(*, prefetch_after_the_next_opens: bool):
    """A consumer that asks its first reformed simulation for one key, then
    prefetches three (one asked already, two new in descending period order),
    prefetches an asked key again, and asks a second reformed simulation for
    one key; optionally it then prefetches from the first again."""
    first = ParameterChangesReform({"gov.fixture.first": {"2024-01-01": 1}})
    second = ParameterChangesReform({"gov.fixture.second": {"2024-01-01": 2}})

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True

        def reformed(source):
            return simulate(
                source if records_only else build_post_export_reform(source)
            )

        first_simulation = reformed(first)
        first_simulation.calculate("income_tax", 2026)
        first_simulation.prefetch(
            (("estate_tax", 2027), ("income_tax", 2026), ("estate_tax", 2026))
        )
        first_simulation.prefetch((("estate_tax", 2027),))
        reformed(second).calculate("income_tax", 2026)
        if prefetch_after_the_next_opens:
            first_simulation.prefetch((("income_tax", 2027),))

    return consumer, first, second


def test_a_prefetch_is_one_pass_with_its_new_keys_in_ascending_period_order(
    builder, fake_reforms, tmp_path
) -> None:
    """``_BatchedPostExportReform.prefetch`` scores the keys a reformed
    simulation was not asked before in one pass, in ascending period order.
    The dry run records those keys in that order, counts that prefetch as one
    pass and a prefetch of asked keys as none, and refuses a prefetch of a
    reformed simulation once the next one has opened, as the scorer does."""
    consumer, first, second = _prefetching_consumer(prefetch_after_the_next_opens=False)
    requests = builder._record_post_export_request_descriptors("fixture", consumer)
    assert [(r.position, r.reformed_simulation, r.reform, r.key) for r in requests] == [
        (0, 0, first, ("income_tax", 2026, None)),
        (1, 0, first, ("estate_tax", 2026, None)),
        (2, 0, first, ("estate_tax", 2027, None)),
        (3, 1, second, ("income_tax", 2026, None)),
    ]
    assert builder._record_post_export_requests(consumer) == ((), 3)

    late_consumer, _, _ = _prefetching_consumer(prefetch_after_the_next_opens=True)
    refusal = r"reformed simulation 0 for .* after opening reformed simulation 1\."
    with pytest.raises(RuntimeError, match=refusal):
        builder._record_post_export_request_descriptors("fixture", late_consumer)

    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    try:
        scoring = scorer.open_consumer("fixture", (), planned_reform_passes=3)
        consumer(scoring.simulate)
        assert scoring.record()["reform_passes"] == 3
        scorer.finish_consumer(scoring)
        late_scoring = scorer.open_consumer("late fixture", ())
        with pytest.raises(RuntimeError, match="asked a finished reform"):
            late_consumer(late_scoring.simulate)
    finally:
        scorer.close()


class _Unreducible:
    """A value whose pickling raises ``ValueError``, not ``PicklingError``."""

    def __reduce__(self):
        raise ValueError("fixture value refuses to be pickled")


def _undescribable_reform(kind: str, monkeypatch):
    """(a reform the dry run must refuse, the message it must give)."""

    class BuiltReform:
        pass

    spec = next(
        spec
        for spec in reform_validation_module.load_default_reform_specs(period=2024)
        if spec.parameter_changes
    )
    if kind == "built reform":
        return BuiltReform(), "got a .*BuiltReform"
    if kind == "subclass registered only in this process":
        # It pickles here, by its module's name; a spawned worker, which has
        # no such module, could not unpickle it.
        module = types.ModuleType("registered_only_in_this_process")
        exec(
            "from microcosm.build.us_runtime.post_export_requests import "
            "ParameterChangesReform\n"
            "class RegisteredChanges(ParameterChangesReform):\n"
            "    pass\n",
            module.__dict__,
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)
        reform = module.RegisteredChanges({"gov.fixture": {"2024-01-01": 1}})
        pickle.dumps(reform)
        return (
            reform,
            r"got a registered_only_in_this_process\.RegisteredChanges\. .* exactly",
        )
    if kind == "read-only mapping":
        proxied = dataclasses.replace(
            spec, parameter_changes=MappingProxyType(dict(spec.parameter_changes))
        )
        return proxied, "ReformValidationSpec reform description that cannot be pickled"
    unreducible = dataclasses.replace(
        spec, parameter_changes={"gov.fixture": {"2024-01-01": _Unreducible()}}
    )
    return unreducible, "cannot be pickled .*ValueError: fixture value refuses"


@pytest.mark.parametrize(
    "kind",
    [
        "built reform",
        "subclass registered only in this process",
        "read-only mapping",
        "value whose pickling raises ValueError",
    ],
)
def test_the_dry_run_refuses_a_reform_it_cannot_describe(
    builder, no_engine, monkeypatch, kind
) -> None:
    """A worker rebuilds each reform from its description, so a consumer that
    hands the dry run a built reform, or a description a worker might not
    unpickle, fails the plan, which the release records and refuses before
    the export, instead of recording a request no worker could run."""
    reform, message = _undescribable_reform(kind, monkeypatch)

    def consumer(simulate):
        simulate(reform).calculate("income_tax", 2024)

    with pytest.raises(TypeError, match=message):
        builder._record_post_export_requests(consumer)
    with pytest.raises(TypeError, match=message):
        builder._record_post_export_request_descriptors("fixture", consumer)
    monkeypatch.setattr(
        builder,
        "_post_export_consumers",
        lambda args, *, result, release_id: {"fixture": consumer},
    )
    plan = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=7,
        result=_empty_calibration_result(),
        release_id=_RELEASE_ID,
    )
    assert plan.error is not None
    assert plan.error.startswith("TypeError") and re.search(message, plan.error)
    assert plan.terminal_failures()


def test_a_request_rebuilds_its_reform_with_its_consumer_s_builder(
    monkeypatch,
) -> None:
    built: list[tuple[str, object]] = []
    monkeypatch.setattr(
        smoke_module, "_build_reform", lambda probe: built.append(("probe", probe))
    )
    monkeypatch.setattr(
        ReformValidationSpec,
        "build_reform",
        lambda spec: built.append(("spec", spec)),
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: built.append(("changes", changes)),
    )
    probe = us_release_reform_coverage_probes()[0]
    spec = reform_validation_module.load_default_reform_specs(period=2024)[0]
    changes = {"gov.fixture.parameter": {"2026-01-01": 0}}
    for reform in (probe, spec, ParameterChangesReform(changes)):
        PostExportRequest(
            consumer="fixture",
            position=0,
            variable="income_tax",
            period=2026,
            reform=reform,
            reformed_simulation=0,
        ).build_reform()
    assert built == [("probe", probe), ("spec", spec), ("changes", changes)]
    baseline = PostExportRequest(
        consumer="fixture", position=0, variable="age", period=2024
    )
    assert baseline.build_reform() is None
    assert baseline.key == ("age", 2024, None)


@pytest.mark.parametrize(
    "fields, error, message",
    [
        ({"reform": None, "reformed_simulation": 0}, ValueError, "reformed simulation"),
        ({"reformed_simulation": None}, ValueError, "reformed simulation"),
        ({"reformed_simulation": -1}, ValueError, "reformed_simulation"),
        ({"reformed_simulation": True}, TypeError, "reformed_simulation"),
        ({"reformed_simulation": np.int64(0)}, TypeError, "reformed_simulation"),
        ({"reform": object()}, TypeError, "object"),
        ({"position": -1}, ValueError, "position"),
        ({"position": np.int64(0)}, TypeError, "position"),
        ({"period": "2024"}, TypeError, "period"),
        ({"period": True}, TypeError, "period"),
        ({"map_to": ""}, ValueError, "map_to"),
        ({"map_to": 5}, TypeError, "map_to"),
        ({"consumer": ""}, ValueError, "consumer"),
        ({"consumer": 5}, TypeError, "consumer"),
        ({"variable": ""}, ValueError, "variable"),
        ({"variable": None}, TypeError, "variable"),
    ],
)
def test_a_request_refuses_inconsistent_fields(fields, error, message) -> None:
    valid = {
        "consumer": "fixture",
        "position": 0,
        "variable": "income_tax",
        "period": 2026,
        "map_to": None,
        "reform": ParameterChangesReform({"gov.fixture": {"2026-01-01": 0}}),
        "reformed_simulation": 0,
    }
    PostExportRequest(**valid)
    with pytest.raises(error, match=message):
        PostExportRequest(**{**valid, **fields})


@pytest.mark.parametrize("changes", [{}, None, [("gov.fixture", 1)]])
def test_parameter_changes_must_name_a_reform(changes) -> None:
    """No changes is the baseline, which consumers ask for as ``None``."""
    with pytest.raises((TypeError, ValueError), match="parameter changes"):
        ParameterChangesReform(changes)


def test_parameter_changes_that_cannot_be_copied_are_refused() -> None:
    with pytest.raises(TypeError, match="parameter changes cannot be copied"):
        ParameterChangesReform({"gov.fixture": {"2024-01-01": threading.Lock()}})


def test_descriptors_cannot_be_hashed_and_keep_their_own_changes() -> None:
    """Most reform descriptions hold dictionaries, so no descriptor hashes
    (results are keyed by consumer and position), and a cumulative OBBBA
    state keeps a copy of the payload the consumer handed the dry run."""
    changes = {"gov.fixture.parameter": {"2026-01-01": 0}}
    reform = ParameterChangesReform(changes)
    changes["gov.fixture.parameter"]["2026-01-01"] = 1
    changes["gov.fixture.other"] = {"2026-01-01": 2}
    assert reform.parameter_changes == {"gov.fixture.parameter": {"2026-01-01": 0}}
    baseline = PostExportRequest(
        consumer="fixture", position=0, variable="age", period=2024
    )
    reformed = PostExportRequest(
        consumer="fixture",
        position=1,
        variable="income_tax",
        period=2026,
        reform=reform,
        reformed_simulation=0,
    )
    for value in (baseline, reformed, reform):
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    # A read-only mapping cannot be pickled; the copy is a dict, which can.
    proxied = ParameterChangesReform(
        MappingProxyType({"gov.fixture.parameter": {"2026-01-01": 0}})
    )
    assert type(proxied.parameter_changes) is dict
    assert pickle.loads(pickle.dumps(proxied)) == proxied == reform


def test_the_reform_alias_names_the_three_descriptions() -> None:
    assert typing.get_args(PostExportReform.__value__) == (
        ReformCoverageProbe,
        ReformValidationSpec,
        ParameterChangesReform,
    )


#: SHA-256 of each output the post-export stages record, on the fixture frame
#: and fake engine below, with a calibration fit covering two in-sample JCT
#: reforms. Pinned from origin/main at fcabe4e65, before the dry run returned
#: descriptors. The digests also cover the shipped probes, reform specs,
#: baseline levels and holdout benchmarks, so a change to any of those moves
#: them too. Re-pin only when such an input or the fixture changed on purpose,
#: in the same pull request, by copying the digests this test's failure
#: prints. The manifests' ``post_export_scoring`` block, which holds the
#: method label and the batching, is not pinned here.
_RECORDED_OUTPUT_SHA256 = {
    "calibration_diagnostics.build.post_export_scoring": (
        "7ed2a0dd088cf2d5a65e27e00368819cdf357160a5bdc2e42f079233d1f2f7b1"
    ),
    "reform_coverage_smoke": (
        "c2e6cbf63b02fc5dbbc1ddd5507edd5e9cfb21a41143469cc737c69279da9842"
    ),
    "reform_validation": (
        "e84481d7051f60570f7ba7bfafd0c631c4de5f19fb27ec2d50e8d7802689910e"
    ),
    "demographics": (
        "081adcc03922e4e3a7cf288f337fa28bd32511d4ca6626b3afed896edfa0d4c2"
    ),
    "ssi_social_security_holdout": (
        "b3d2db968bb3b10d004783ab7c080d6b4e376e4441e6753c05712bbd35c1a4b4"
    ),
    "medicare_irmaa_holdout": (
        "1962831f0fb368fdadd222e42310ca45a70effe214fe520c11a7f1f60d75337f"
    ),
}


#: The reform passes the plan counts for each stage (its progress reporting
#: reads them), on the same fixture; as on origin/main at fcabe4e65, except
#: reform validation's: it scores the pre-OBBBA state once for both OBBBA
#: groups (microcosm#1595), so it counts 63 passes, one fewer, for the same
#: 64 reform requests.
_PLANNED_REFORM_PASSES = {
    "reform_coverage_smoke": 43,
    "reform_validation": 63,
    "demographics": 0,
    "ssi_social_security_holdout": 0,
    "medicare_irmaa_holdout": 0,
}


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_of(value) -> str:
    encoded = json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _recorded_post_export_outputs(builder, tmp_path) -> dict[str, object]:
    """What ``_main`` records: the plan, then each consumer through one scorer."""
    fit = _calibration_result_with_in_sample_fit(builder.PERIOD)
    args = _plan_args()
    plan = builder._record_post_export_scoring_plan(
        args, n_households=7, result=fit, release_id=_RELEASE_ID
    )
    assert plan.error is None
    assert {
        name: plan.planned_reform_passes(name) for name in plan.baseline_plans
    } == _PLANNED_REFORM_PASSES
    outputs: dict[str, object] = {
        "calibration_diagnostics.build.post_export_scoring": plan.record()
    }
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    try:
        for name, consumer in builder._post_export_consumers(
            args, result=fit, release_id=_RELEASE_ID
        ).items():
            outputs[name] = builder._score_post_export_consumer(
                name,
                consumer,
                dataset_path=scorer.dataset_path,
                post_export_scorer=scorer,
                baseline_plan=plan.baseline_plan(name),
                maximum_microsim_batch_size=args.maximum_microsim_batch_size,
                planned_reform_passes=plan.planned_reform_passes(name),
            )
    finally:
        scorer.close()
    smoke = outputs["reform_coverage_smoke"]
    outputs["reform_coverage_smoke"] = {
        "name": smoke.name,
        "passed": smoke.passed,
        "failures": list(smoke.failures),
        "details": smoke.details,
    }
    ages, weights = outputs["demographics"]
    outputs["demographics"] = demographics_payload(
        ages, weights, period=builder.PERIOD, release_id=_RELEASE_ID
    )
    return outputs


def test_the_release_s_recorded_post_export_outputs_are_unchanged(
    builder, fake_reforms, tmp_path
) -> None:
    """The plan in ``calibration_diagnostics.json`` and every consumer's output
    keep their bytes: the descriptors are returned, never recorded."""
    outputs = _recorded_post_export_outputs(builder, tmp_path)
    assert list(outputs) == list(_RECORDED_OUTPUT_SHA256)
    assert {
        name: _sha256_of(output) for name, output in outputs.items()
    } == _RECORDED_OUTPUT_SHA256
