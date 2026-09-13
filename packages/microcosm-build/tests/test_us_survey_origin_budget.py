"""Invented sources only; no calibration solver, targets or genuine inputs."""

import copy
import json
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime import survey_origin_budget as owner
from microcosm.frame import WeightKind, Weights
from microcosm.graph import KernelResult, Node, StructuralDelta, WeightTransition
from microcosm.graph.population import PopulationError, patch


@pytest.mark.parametrize("d", [Fraction(0), Fraction(1, 10), Fraction(63803, 25)])
@pytest.mark.parametrize("p", [Fraction(1), Fraction(1, 2), Fraction(3, 7)])
@pytest.mark.parametrize("s", [Fraction(1), Fraction(1, 2)])
def test_reference_exact_and_float_conventions(d, p, s):
    a, b = d * s / p, d / p
    value = owner._reference(d, p, s, float(a))
    assert value["d"] == [d.numerator, d.denominator]
    assert value["b"] == [b.numerator, b.denominator]
    assert value["a"] == [a.numerator, a.denominator]
    assert float.fromhex(value["actual_incoming_float64_hex"]) == float(a)
    assert float.fromhex(value["upper_float64_hex"]) == 4.0 * float(a)
    if d == 0:
        assert float.fromhex(value["upper_float64_hex"]) == 0


def test_reference_refuses_staged_float_product_ulp():
    d, p, s = Fraction(1, 10), Fraction(3, 7), Fraction(1, 2)
    staged = float(d) * float(s / p)
    assert staged != float(d * s / p)
    with pytest.raises(
        owner.SurveyOriginBudgetError, match="ALLOCATION_OPERATION_ORDER"
    ):
        owner._reference(d, p, s, staged)


@pytest.mark.parametrize("bad", [True, 1, 0.5, "1", None])
def test_reference_requires_exact_fraction(bad):
    with pytest.raises(owner.SurveyOriginBudgetError, match="EXACT_FRACTION_REQUIRED"):
        owner._reference(bad, Fraction(1), Fraction(1), 1.0)


@pytest.mark.parametrize(
    "p,s",
    [
        (Fraction(0), Fraction(1)),
        (Fraction(2), Fraction(1)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(3, 4)),
    ],
)
def test_reference_refuses_outside_declaration(p, s):
    with pytest.raises(owner.SurveyOriginBudgetError, match="REFERENCE_DOMAIN"):
        owner._reference(Fraction(1), p, s, 1.0)


@pytest.mark.parametrize(
    "cls", [owner.SamplingOriginBudget, owner.SamplingOriginSuccessor]
)
def test_no_public_capsule_constructor(cls):
    with pytest.raises(owner.SurveyOriginBudgetError, match="NO_PUBLIC"):
        cls(b"{}")


def test_transport_exact_boundary_before_oversize_encoding(monkeypatch):
    value = {"a": "x" * 32}
    encoded = owner._json(value)
    monkeypatch.setattr(owner, "MAX_PAYLOAD_BYTES", len(encoded))
    assert owner._json(value) == encoded
    monkeypatch.setattr(owner, "MAX_PAYLOAD_BYTES", len(encoded) - 1)
    with pytest.raises(graph.SurveyPopulationGraphError, match="TRANSPORT_LIMIT"):
        owner._json(value)


def test_nullable_storage_distinguishes_hidden_values():
    left = pd.Series([1, pd.NA], dtype="Int64")
    right = left.copy(deep=True)
    right.array._data[1] += 19
    pd.testing.assert_series_equal(left, right)
    assert not owner.storage_equal(left, right)


def test_refinement_refuses_before_source_borrow():
    with pytest.raises(owner.SurveyOriginBudgetError, match="UNSUPPORTED_REFINEMENT"):
        owner.admit_survey_weight_only_population(
            None, previous=None, current=None, previous_binding=object()
        )


@pytest.mark.parametrize("kind", ["prescription", "callable"])
def test_producer_refuses_changed_loaded_contract(monkeypatch, kind):
    if kind == "prescription":
        monkeypatch.setattr(owner, "PRESCRIPTION", (*owner.PRESCRIPTION, "changed"))
    else:
        monkeypatch.setattr(owner, "_reference", lambda *args: {})
    with pytest.raises(owner.SurveyOriginBudgetError, match="PRODUCER_CHANGED"):
        owner._producer()


def _actual(tmp_path, monkeypatch, *, fraction=Fraction(1), zero=True):
    from test_us_survey_population_preparation import fixture

    arguments = fixture(tmp_path, monkeypatch, fraction=fraction, zero=zero)
    return _run_actual(arguments, store_root=tmp_path / "store")[0]


def _run_actual(arguments, *, store_root, resume="auto"):
    captured = {}
    codes = {
        id(graph._checked_preparation.__code__): "preparation",
        id(graph._check_population_state.__code__): "population",
    }

    def profile(frame, event, arg):
        if event != "call":
            return
        kind = codes.get(id(frame.f_code))
        if kind == "preparation":
            value = frame.f_locals["preparation"]
            assert captured.get("preparation", value) is value
            captured["preparation"] = value
        elif kind == "population":
            value = frame.f_locals["population"]
            if value.version == graph.ALLOCATION_NODE:
                captured["allocated_population"] = value
            elif (
                value.version == owner.clone.COMBINED_CLONE_NODE
                and owner.clone.COMBINED_CLONE_CLAIM_NODE in value.owners.values()
            ):
                captured["clone_population"] = value

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        manifest = graph.run_authenticated_survey_population(
            **arguments, store_root=store_root, resume=resume
        )
    finally:
        sys.setprofile(previous)
    assert set(captured) == {"preparation", "allocated_population", "clone_population"}
    assert len(manifest.nodes) == 4
    return captured, manifest


def _advance(population, values, version):
    # Apply the real generic REWEIGHT contract, including the unchanged original
    # DESIGN scalar guard. No solver, target or replacement source authority.
    node = Node(
        version,
        "invented.weight-only@1",
        base=population.version,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        params={"max_weight_ratio": 8.0, "weight_anchor": "design"},
    )
    return patch(
        population,
        node,
        KernelResult(
            weights=Weights(np.asarray(values, dtype=np.float64), WeightKind.CALIBRATED)
        ),
    )


def test_actual_initial_budget_and_zero(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    view = budget.checked_view()
    assert view.preparation is arguments["preparation"]
    assert view.initial_population is arguments["clone_population"]
    doc = view.document
    assert doc["group_count"] == 6 and len(doc["household_ids"]) == 12
    assert {row["source"] for row in doc["origins"]} == {"acs", "asec"}
    assert all(len(row["members"]) == 2 for row in doc["origins"])
    zeros = [row for row in doc["origins"] if row["d"] == [0, 1]]
    assert len(zeros) == 1 and zeros[0]["a"] == zeros[0]["b"] == [0, 1]
    assert float.fromhex(zeros[0]["upper_float64_hex"]) == 0
    for row in doc["origins"]:
        assert row["p"] == [1, 1]
        assert row["upper_float64_hex"] == row["incoming_bound_float64_hex"]
    values = arguments["clone_population"].frame.weights_for("household").values
    view.grouped_bounds.check(values)
    with pytest.raises(owner.SurveyOriginBudgetError, match="UNISSUED"):
        owner.verify_survey_origin_budget(copy.copy(budget))
    doc["origins"][0]["a"] = [999, 1]
    assert json.loads(budget.payload)["origins"][0]["a"] != [999, 1]
    (tmp_path / "budget-metrics.json").write_text(
        json.dumps(
            {
                "budget_bytes": len(budget.payload),
                "groups": 6,
                "households": 12,
                "persons": 18,
                "release_eligible": False,
                "source_mode": "invented_only",
                "calibration": "not_run",
            }
        )
    )


def test_actual_fraction_is_catalogue_n_over_n(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch, fraction=Fraction(2, 3), zero=False)
    budget = owner.freeze_survey_origin_budget(**arguments)
    doc = json.loads(budget.payload)
    asec = [row for row in doc["origins"] if row["source"] == "asec"]
    assert len(asec) == 1 and asec[0]["p"] == [1, 2]
    assert Fraction(*asec[0]["b"]) == 2 * Fraction(*asec[0]["d"])
    assert asec[0]["source_year"] == 2024 and asec[0]["survey_year"] == 2025


def test_actual_rehashed_candidate_refuses(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    for field in ("p", "s", "b"):
        doc = json.loads(budget.payload)
        doc["origins"][0][field] = [1, 9]
        changed = owner._json(doc)
        with pytest.raises(
            owner.SurveyOriginBudgetError, match="CANDIDATE_RECONSTRUCTION"
        ):
            owner.freeze_survey_origin_budget(**arguments, candidate=changed)


def test_actual_warm_candidate_reconstructs_fresh_source(tmp_path, monkeypatch):
    from test_us_survey_population_preparation import fixture

    arguments = fixture(tmp_path, monkeypatch)
    cold, cold_manifest = _run_actual(arguments, store_root=tmp_path / "store")
    budget = owner.freeze_survey_origin_budget(**cold)
    warm, warm_manifest = _run_actual(
        arguments, store_root=tmp_path / "store", resume="require"
    )
    assert warm["preparation"] is not cold["preparation"]
    assert warm["clone_population"] is not cold["clone_population"]
    assert warm_manifest.key == cold_manifest.key
    assert len(warm_manifest.nodes) == 4 and all(
        row.store_hit for row in warm_manifest.nodes.values()
    )
    reconstructed = owner.freeze_survey_origin_budget(**warm, candidate=budget.payload)
    assert reconstructed is not budget and reconstructed.payload == budget.payload


def test_actual_successor_keeps_zero_and_unequal_pair_weights(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    values = initial.frame.weights_for("household").values.copy()
    values[:6] *= 1.25
    values[6:] *= 0.75
    current = _advance(initial, values, "invented.calibration.one")
    binding = owner.admit_survey_weight_only_population(
        budget, previous=initial, current=current
    )
    checked = binding.checked_view()
    assert checked.current is current and checked.previous is initial
    assert checked.payload == binding.payload and checked.previous_binding is None
    with pytest.raises(owner.SurveyOriginBudgetError, match="UNISSUED"):
        owner.verify_survey_weight_only_successor(copy.copy(binding))
    assert json.loads(binding.payload)["budget_sha256"] == owner._sha(budget.payload)
    assert np.count_nonzero(current.frame.weights_for("household").values == 0) == 2
    with pytest.raises(PopulationError, match="must move forward"):
        _advance(current, values * 1.1, "invented.unsupported.refinement")
    with pytest.raises(owner.SurveyOriginBudgetError, match="UNSUPPORTED_REFINEMENT"):
        owner.admit_survey_weight_only_population(
            budget, previous=current, current=current
        )
    with pytest.raises(owner.SurveyOriginBudgetError, match="UNSUPPORTED_REFINEMENT"):
        owner.admit_survey_weight_only_population(
            budget, previous=initial, current=current, previous_binding=binding
        )
    # Independent first-calibration branches from the same issued initial clone
    # are supported by the actual graph and retain the same frozen budget.
    next_population = _advance(initial, values * 1.1, "invented.calibration.two")
    second = owner.admit_survey_weight_only_population(
        budget, previous=initial, current=next_population
    )
    assert json.loads(second.payload)["previous_binding_sha256"] is None
    assert (
        json.loads(second.payload)["budget_sha256"]
        == json.loads(binding.payload)["budget_sha256"]
    )


def test_actual_strict_group_bound_with_real_patch(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    # Six times the initial pair total exceeds U=4a, but each member remains
    # below the unchanged 8d scalar graph guard. The real patch therefore runs.
    values = initial.frame.weights_for("household").values * 6
    current = _advance(initial, values, "invented.over.group.bound")
    with pytest.raises(ValueError, match="group weights exceed frozen absolute bounds"):
        owner.admit_survey_weight_only_population(
            budget, previous=initial, current=current
        )
    zeros = np.flatnonzero(initial.frame.weights_for("household").values == 0)
    assert len(zeros) == 2
    positive_zero = initial.frame.weights_for("household").values.copy()
    positive_zero[zeros[0]] = 1
    with pytest.raises(PopulationError, match="design"):
        _advance(initial, positive_zero, "invented.positive.zero")


def test_actual_view_decode_mutation_is_checked_before_return(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    current = _advance(
        initial, initial.frame.weights_for("household").values.copy(), "invented.view"
    )
    binding = owner.admit_survey_weight_only_population(
        budget, previous=initial, current=current
    )
    cases = ((budget, initial), (binding, current))
    for handle, population in cases:
        table = population.frame.table("household")
        column = table.columns.get_loc("household_id")
        original = table.iloc[0, column]
        injected = []
        checked_code = type(handle).checked_view.__code__
        decode_code = json.loads.__code__

        def profile(
            frame,
            event,
            arg,
            *,
            decode_code=decode_code,
            checked_code=checked_code,
            table=table,
            column=column,
            original=original,
            population=population,
            injected=injected,
        ):
            if (
                event == "return"
                and frame.f_code is decode_code
                and frame.f_back.f_code is checked_code
            ):
                table.iloc[0, column] = int(original) + 1234
                assert (
                    population.frame.table("household").iloc[0, column]
                    == int(original) + 1234
                )
                injected.append(True)

        prior = sys.getprofile()
        sys.setprofile(profile)
        try:
            with pytest.raises(owner.SurveyOriginBudgetError, match="POPULATION"):
                handle.checked_view()
        finally:
            sys.setprofile(prior)
            table.iloc[0, column] = original
        assert injected == [True]


def _masked_numeric(frame):
    for entity in frame.entities:
        for column in frame.table(entity):
            series = frame.table(entity)[column]
            data, mask = (
                getattr(series.array, "_data", None),
                getattr(series.array, "_mask", None),
            )
            if (
                isinstance(data, np.ndarray)
                and isinstance(mask, np.ndarray)
                and mask.any()
                and data.dtype.kind in "iuf"
            ):
                return series, int(np.flatnonzero(mask)[0])
    raise AssertionError("Invented fixture lacks a numeric nullable storage control")


def test_actual_nullable_backing_refuses_before_admission(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    current = _advance(
        initial, initial.frame.weights_for("household").values.copy(), "invented.masked"
    )
    series, position = _masked_numeric(current.frame)
    before = series.copy(deep=True)
    series.array._data[position] += 1
    pd.testing.assert_series_equal(before, series)
    with pytest.raises(owner.SurveyOriginBudgetError, match="SUCCESSOR_NONWEIGHT"):
        owner.admit_survey_weight_only_population(
            budget, previous=initial, current=current
        )


def test_actual_retained_population_mutations_refuse(tmp_path, monkeypatch):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    for entity, column in (
        ("household", "household_id"),
        ("person", "person_household_id"),
    ):
        table = initial.frame.table(entity)
        column_position = table.columns.get_loc(column)
        original = table.iloc[0, column_position]
        table.iloc[0, column_position] = int(original) + 98765
        assert (
            initial.frame.table(entity).iloc[0, column_position]
            == int(original) + 98765
        )
        with pytest.raises(owner.SurveyOriginBudgetError, match="POPULATION"):
            owner.verify_survey_origin_budget(budget)
        table.iloc[0, column_position] = original
        assert initial.frame.table(entity).iloc[0, column_position] == original
    weights = initial.frame.weights_for("household")
    original_values = weights.values
    changed_values = original_values.copy()
    changed_values[0] += 1
    object.__setattr__(weights, "values", changed_values)
    assert initial.frame.weights_for("household").values[0] == original_values[0] + 1
    with pytest.raises(owner.SurveyOriginBudgetError, match="POPULATION"):
        owner.verify_survey_origin_budget(budget)
    object.__setattr__(weights, "values", original_values)
    original_design = initial.design_weights
    with pytest.raises(ValueError, match="WRITEABLE"):
        original_design["household"].flags.writeable = True
    changed_design = original_design["household"].copy()
    changed_design[0] += 1
    object.__setattr__(initial, "design_weights", {"household": changed_design})
    assert initial.design_weights["household"][0] == original_design["household"][0] + 1
    with pytest.raises(owner.SurveyOriginBudgetError, match="POPULATION"):
        owner.verify_survey_origin_budget(budget)
    object.__setattr__(initial, "design_weights", original_design)
    for attribute, changed in (("owners", {}), ("mass_ledger", ())):
        original = getattr(initial, attribute)
        object.__setattr__(initial, attribute, changed)
        with pytest.raises(owner.SurveyOriginBudgetError, match="POPULATION"):
            owner.verify_survey_origin_budget(budget)
        object.__setattr__(initial, attribute, original)


def _assert_final_mutation_refused(helper, operation, mutate, restore):
    calls = []

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is helper.__code__:
            calls.append(True)
            if len(calls) == 2:
                mutate()

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            (
                owner.SurveyOriginBudgetError,
                owner.source.SurveyPopulationPreparationError,
            )
        ):
            operation()
    finally:
        sys.setprofile(previous)
        restore()
    assert len(calls) == 2
    assert operation() is not None


@pytest.mark.parametrize("operation_kind", ["issuance", "borrow"])
@pytest.mark.parametrize("changed_owner", ["preparation", "attached"])
def test_actual_final_preparation_owner_mutation_refuses(
    tmp_path, monkeypatch, operation_kind, changed_owner
):
    arguments = _actual(tmp_path, monkeypatch)
    preparation = arguments["preparation"]
    budget = owner.freeze_survey_origin_budget(**arguments)
    value = (
        preparation
        if changed_owner == "preparation"
        else owner.source._ISSUED[id(preparation)][2].native[1]
    )
    original = value.payload
    changed = original + b"\n"

    def mutate():
        object.__setattr__(value, "payload", changed)
        assert object.__getattribute__(value, "payload") == changed

    def restore():
        object.__setattr__(value, "payload", original)
        assert value.payload == original

    operation = (
        (lambda: owner.freeze_survey_origin_budget(**arguments))
        if operation_kind == "issuance"
        else budget.checked_view
    )
    _assert_final_mutation_refused(owner._initial, operation, mutate, restore)


@pytest.mark.parametrize("operation_kind", ["admission", "borrow"])
def test_actual_final_budget_mutation_refuses_successor(
    tmp_path, monkeypatch, operation_kind
):
    arguments = _actual(tmp_path, monkeypatch)
    budget = owner.freeze_survey_origin_budget(**arguments)
    initial = arguments["clone_population"]
    current = _advance(
        initial, initial.frame.weights_for("household").values.copy(), "invented.final"
    )

    def admit():
        return owner.admit_survey_weight_only_population(
            budget, previous=initial, current=current
        )

    binding = admit()
    original = budget.payload
    changed = original + b"\n"

    def mutate():
        object.__setattr__(budget, "payload", changed)
        assert object.__getattribute__(budget, "payload") == changed

    def restore():
        object.__setattr__(budget, "payload", original)
        assert budget.payload == original

    _assert_final_mutation_refused(
        owner._checked_successor_document,
        admit if operation_kind == "admission" else binding.checked_view,
        mutate,
        restore,
    )


def test_actual_replaced_preparation_entry_refuses_current_budget_borrow(
    tmp_path, monkeypatch
):
    arguments = _actual(tmp_path, monkeypatch)
    preparation = arguments["preparation"]
    budget = owner.freeze_survey_origin_budget(**arguments)
    original = owner.source._ISSUED[id(preparation)]
    copied = tuple(list(original))
    assert copied == original and copied is not original

    def mutate():
        owner.source._ISSUED[id(preparation)] = copied
        assert owner.source._ISSUED[id(preparation)] is copied

    def restore():
        owner.source._ISSUED[id(preparation)] = original
        assert owner.source._ISSUED[id(preparation)] is original

    _assert_final_mutation_refused(owner._initial, budget.checked_view, mutate, restore)
