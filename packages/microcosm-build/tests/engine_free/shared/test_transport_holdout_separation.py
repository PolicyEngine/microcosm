"""Declared holdout facts stay out of calibration, for every country.

A ``holdout_references`` row tests the calibrated model against a fact it
never saw. Composition (``validate_transport_activation`` in
``microcosm.build.transport.compose``) refuses any row of another reference
document, mass included, whose ``measure`` equals a holdout row's measure on
any entity, and ``_precalibration_stages`` refuses a pre-calibration stage
output with a holdout's measure on any entity.

Invariants:

- Examples: a calibration row on the holdout's column, the same measure on
  another entity, a pre-calibration row, a mass row and a pre-calibration
  stage output on another entity each refuse with both rows named.
- No packaged spec (AM, BE, NZ, UK, US) trips the guard, over every packaged
  reference document, and composition of each gives the outcome it gave
  before the guard.
- Row-level holdouts stay: a spec that declares no ``holdout_references``
  rows composes whatever its rows measure.
- Property: on random reference documents drawn from a small pool of
  measures and entities, activation refuses for holdout separation exactly
  when a non-holdout row shares a measure with a holdout row.
- Differential: Belgium's package-test helper ``_separation_violations``
  (``test_be_target_references.py``) and the guard agree, on BE's packaged
  documents and on generated documents.
"""

from __future__ import annotations

import copy
import importlib.util
import re
import sys
from importlib.resources import files

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.holdout import rotated_folds
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport.compose import (
    compose_transport_graph,
    load_transport_spec,
    validate_transport_activation,
)
from test_support.microcosm_build.transport_composed import (
    PROGRAMS,
    make_composed_fixture,
)
from test_support.microcosm_build.transport_graph import reference_row
from test_support.paths import paths_for

PACKAGED_COUNTRIES = ("am", "be", "nz", "uk", "us")
SEPARATION = "holdout separation"
#: Belgium's three reference sets and the documents that hold them.
BE_SETS = {
    "calibration": "target_references",
    "precal": "precal_references",
    "holdout": "holdout_references",
}
_BE_TEST = (
    paths_for("microcosm-build").package
    / "tests/engine_free/shared/test_be_target_references.py"
)


def _be_test_module():
    """Belgium's package test, imported by path so its helper is the original."""

    name = "_holdout_separation_be_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _BE_TEST)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """The composed toy package, without unrelated gate binding identities."""
    from microcosm.build.transport import registry as registry_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry_module, "register_gate_kernel", lambda *args: None)
        return make_composed_fixture(tmp_path_factory.mktemp("holdout-separation"))


def _rows(spec, resource):
    return spec["resources"][resource]["target_references"]


def _activation_error(spec):
    try:
        validate_transport_activation(spec)
    except ValueError as error:
        return str(error)
    return None


def _separation_gaps(message):
    if message is None:
        return []
    return [
        gap.strip()
        for gap in message.removeprefix("Transport spec is not activated: ")
        .removesuffix(".")
        .split("; ")
        if gap.strip().startswith(SEPARATION)
    ]


def test_toy_composes_with_its_holdout(toy):
    holdout = _rows(toy.spec, "holdout_references")
    assert [(row["entity"], row["measure"]) for row in holdout] == [
        ("person", "receives_main")
    ]
    assert len(compose_transport_graph(toy.spec, toy.config).nodes) == 27


@pytest.mark.parametrize(
    ("resource", "entity"),
    [
        ("target_references", "person"),
        ("target_references", "household"),
        ("precal_references", "person"),
        ("precal_references", "household"),
        ("mass_references", "person"),
        ("mass_references", "household"),
    ],
)
def test_row_measuring_a_holdout_refuses_naming_both_rows(toy, resource, entity):
    """The triage's reproductions, plus pre-calibration and mass variants."""

    spec = copy.deepcopy(toy.spec)
    _rows(spec, resource).append(
        reference_row("toy_leak", entity=entity, measure="receives_main")
    )
    with pytest.raises(ValueError) as refused:
        compose_transport_graph(spec, toy.config)
    assert _separation_gaps(str(refused.value)) == [
        f"{SEPARATION}: {resource}.toy_leak ({entity}, receives_main) measures "
        "the concept of holdout_references.toy_holdout_receipts "
        "(person, receives_main)"
    ]


def _family_stage(spec):
    binding = next(
        row
        for row in spec["resources"]["axiom_rules_bindings"]["bindings"]
        if row["id"] == "family_housing"
    )
    assert binding["entity"] == "family"
    assert "toy_family_weekly_amount" in PROGRAMS["family_housing"]["outputs"]
    spec["resources"]["precalibration_stages"] = {
        "schema_version": 1,
        "stages": [
            {
                "id": "toy.precal.family_housing",
                "rules_binding": "family_housing",
                "rulespec_path": binding["rulespec_path"],
                "outputs": [
                    {
                        "entity": "family",
                        "column": "toy_family_weekly_amount",
                        "purpose": "calibration_measure",
                    }
                ],
                "citation": "Synthetic pre-calibration method card.",
            }
        ],
    }
    return spec


@pytest.mark.parametrize("holdout_entity", ["family", "person", "household"])
def test_stage_output_measuring_a_holdout_refuses_on_any_entity(toy, holdout_entity):
    spec = _family_stage(copy.deepcopy(toy.spec))
    assert set(compose_module._precalibration_stages(spec)) == {"family_housing"}
    holdout = _rows(spec, "holdout_references")[0]
    holdout.update(entity=holdout_entity, measure="toy_family_weekly_amount")
    with pytest.raises(ValueError) as refused:
        compose_module._precalibration_stages(spec)
    assert str(refused.value) == (
        "precalibration_stages output is a holdout measure: stage "
        "'toy.precal.family_housing' output (family, toy_family_weekly_amount) "
        "measures the concept of holdout_references.toy_holdout_receipts "
        f"({holdout_entity}, toy_family_weekly_amount)."
    )


def _packaged_reference_documents(country):
    spec = load_transport_spec(country)
    return spec, [
        (name, document)
        for name, document in sorted(spec["resources"].items())
        if isinstance(document, dict) and "target_references" in document
    ]


def test_packaged_country_list_is_complete():
    root = files("microcosm.build")
    packaged = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and any(child.name.endswith("references.json") for child in entry.iterdir())
    )
    assert packaged == list(PACKAGED_COUNTRIES)


@pytest.mark.parametrize("country", PACKAGED_COUNTRIES)
def test_packaged_reference_documents_pass_the_guard(country):
    """The triage's scan, over every packaged reference document."""

    spec, documents = _packaged_reference_documents(country)
    assert documents
    assert list(compose_module._holdout_separation_gaps(documents)) == []
    assert SEPARATION not in (_activation_error(spec) or "")
    declares = "holdout_references" in spec["resources"]
    assert declares == (country in {"be", "nz"})
    if declares:
        holdout = {row["measure"] for row in _rows(spec, "holdout_references")}
        others = {
            row["measure"]
            for name, document in documents
            if name != "holdout_references"
            for row in document["target_references"]
        }
        assert holdout and not holdout & others


@pytest.mark.parametrize("country", PACKAGED_COUNTRIES)
def test_packaged_activation_outcome_is_unchanged(country, monkeypatch):
    """Each packaged spec activates, or refuses, exactly as without the guard."""

    spec, _ = _packaged_reference_documents(country)
    guarded = _activation_error(spec)
    monkeypatch.setattr(
        compose_module, "_holdout_separation_gaps", lambda documents: iter(())
    )
    assert _activation_error(spec) == guarded
    assert _separation_gaps(guarded) == []


@pytest.mark.parametrize("measure", [["receives_main"], {"a": 1}], ids=["list", "dict"])
@pytest.mark.parametrize(
    "resource", ["target_references", "precal_references", "holdout_references"]
)
def test_non_text_measure_keeps_activation_outcome(toy, monkeypatch, resource, measure):
    """A list or dict ``measure`` raises no ``TypeError`` from the guard.

    On a non-holdout row the guard skips the row, as it skips a non-text
    holdout measure; activation's outcome equals its outcome without the
    guard. ``_activation_error`` catches only ``ValueError``, so a
    ``TypeError`` fails the test.
    """

    spec = copy.deepcopy(toy.spec)
    row = reference_row("toy_odd", entity="household", measure="receives_main")
    row["measure"] = copy.deepcopy(measure)
    _rows(spec, resource).append(row)
    guarded = _activation_error(spec)
    monkeypatch.setattr(
        compose_module, "_holdout_separation_gaps", lambda documents: iter(())
    )
    assert _activation_error(spec) == guarded
    assert _separation_gaps(guarded) == []


def test_specs_without_holdout_references_keep_row_level_holdouts(toy):
    """The UK's rotated folds and the US's district blocks stay allowed.

    Neither country declares ``holdout_references``; each holds out rows of
    measures it calibrates elsewhere. A fold document under any other name,
    and an empty ``holdout_references``, leave every row free.
    """

    spec = copy.deepcopy(toy.spec)
    calibration = _rows(spec, "target_references")
    calibration.append(
        reference_row("toy_main_calibrated", entity="person", measure="receives_main")
    )
    rows = [
        reference_row(f"toy_fold_{index}", entity="person", measure="receives_main")
        for index in range(5)
    ]
    folds = rotated_folds(len(rows), n_folds=2)
    held = [rows[int(index)] for index in folds[0]]
    spec["resources"]["holdout_fold_references"] = {
        **spec["resources"]["holdout_references"],
        "target_references": held,
    }
    spec["resources"]["holdout_references"]["target_references"] = []
    assert _separation_gaps(_activation_error(spec)) == []
    assert len(compose_transport_graph(spec, toy.config).nodes) == 27
    # The same rows under the declared name refuse.
    spec["resources"]["holdout_references"]["target_references"] = held
    assert len(_separation_gaps(_activation_error(spec))) == len(held)


_MEASURES = ("receives_main", "age", "person_count", "rent")
_ENTITIES = ("person", "household", "family")
_RESOURCES = (
    "target_references",
    "precal_references",
    "mass_references",
    "holdout_references",
)


@st.composite
def _documents(draw, resources=_RESOURCES):
    documents = {}
    index = 0
    for resource in resources:
        rows = []
        for _ in range(draw(st.integers(0, 3))):
            rows.append(
                reference_row(
                    f"toy_row_{index}",
                    entity=draw(st.sampled_from(_ENTITIES)),
                    measure=draw(st.sampled_from(_MEASURES)),
                )
            )
            index += 1
        documents[resource] = rows
    return documents


def _expected_gaps(documents):
    """An independent oracle: every (other row, holdout row) measure pair."""

    return sorted(
        (resource, row["name"], holdout["name"])
        for resource, rows in documents.items()
        if resource != "holdout_references"
        for row in rows
        for holdout in documents["holdout_references"]
        if row["measure"] == holdout["measure"]
    )


_GAP = re.compile(
    rf"{SEPARATION}: (\w+)\.(\w+) \(\w+, \w+\) measures the concept of "
    r"holdout_references\.(\w+) \(\w+, \w+\)"
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@example(
    documents={
        "target_references": [
            reference_row("toy_row_0", entity="household", measure="receives_main")
        ],
        "precal_references": [],
        "mass_references": [],
        "holdout_references": [
            reference_row("toy_row_1", entity="person", measure="receives_main")
        ],
    }
)
@given(documents=_documents())
def test_activation_refuses_exactly_on_a_shared_measure(toy, documents):
    spec = copy.deepcopy(toy.spec)
    for resource, rows in documents.items():
        spec["resources"][resource]["target_references"] = rows
    gaps = _separation_gaps(_activation_error(spec))
    found = sorted(_GAP.fullmatch(gap).groups() for gap in gaps)
    assert found == _expected_gaps(documents)


def _be_documents():
    spec = load_transport_spec("be")
    return {
        reference_set: copy.deepcopy(spec["resources"][resource])
        for reference_set, resource in BE_SETS.items()
    }


def _belgian_measures(documents):
    return sorted(
        {
            measure
            for kind, measure in _be_test_module()._separation_violations(documents)
            if kind == "holdout_measure"
        }
    )


_GAP_MEASURE = re.compile(rf"{SEPARATION}: \S+ \(\w+, (\w+)\) measures ")


def _guard_measures(documents):
    views = [(BE_SETS[name], document) for name, document in documents.items()]
    return sorted(
        {
            _GAP_MEASURE.match(gap).group(1)
            for gap in compose_module._holdout_separation_gaps(views)
        }
    )


def test_belgium_helper_and_guard_agree_on_packaged_documents():
    documents = _be_documents()
    assert _belgian_measures(documents) == _guard_measures(documents) == []
    assert _be_test_module()._separation_violations(documents) == []


@settings(max_examples=60, deadline=None)
@given(documents=_documents(tuple(BE_SETS.values())))
def test_belgium_helper_and_guard_agree_on_generated_documents(documents):
    """Generated rows reuse one packaged BE row for every field but the key."""

    packaged = _be_documents()
    template = packaged["calibration"]["target_references"][0]
    generated = {}
    for reference_set, resource in BE_SETS.items():
        rows = []
        for row in documents[resource]:
            built = copy.deepcopy(template)
            built.update(name=row["name"], entity=row["entity"], measure=row["measure"])
            built["metadata"]["reference_set"] = reference_set
            rows.append(built)
        generated[reference_set] = {
            **packaged[reference_set],
            "target_references": rows,
        }
    assert _belgian_measures(generated) == _guard_measures(generated)


def test_belgium_helper_is_measure_keyed():
    """Pin the premise: Belgium's helper keys holdout separation on the measure."""

    source = _BE_TEST.read_text()
    assert 'measures.setdefault(row["measure"], set()).add(reference_set)' in source
