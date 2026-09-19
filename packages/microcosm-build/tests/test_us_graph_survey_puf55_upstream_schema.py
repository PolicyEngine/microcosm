"""Cheap schema/replay controls; these invented manifests issue no owner."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_survey_puf55 as graph
from microcosm.frame import (
    US_SCHEMA,
    EntitySchema,
    Frame,
    LinkSpec,
    WeightKind,
    Weights,
)
from microcosm.graph.population import MassRecord


def _frame(schema):
    person = {"person_id": [1, 2], "invented_value": [3.0, 4.0]}
    tables = {}
    for group in schema.group_entities:
        person[f"person_{group}_id"] = [10, 20]
        tables[group] = pd.DataFrame({f"{group}_id": [10, 20]})
    tables["person"] = pd.DataFrame(person)
    for link in schema.links:
        tables[link.name] = pd.DataFrame(
            {"person_id": [1, 2], "household_id": [10, 20]}
        )
    return Frame(
        tables,
        schema,
        {"household": Weights(np.array([1.0, 2.0]), WeightKind.DESIGN)},
    )


def _manifest(frames):
    ledgers = {version: () for version in frames}
    return SimpleNamespace(
        country="us",
        nodes=dict.fromkeys(frames),
        populations=frames,
        mass_ledgers=ledgers,
        population=frames.__getitem__,
        mass_ledger=ledgers.__getitem__,
    )


def _case():
    schemas = {
        "survey_population.create": US_SCHEMA,
        "child_property.donors": EntitySchema(group_entities=("household",)),
        "child_property.recipients": EntitySchema(group_entities=("household",)),
    }
    compiled = SimpleNamespace(
        order=tuple(schemas),
        versions={version: version for version in schemas},
        graph=SimpleNamespace(country="us"),
    )
    return (
        _manifest({v: _frame(s) for v, s in schemas.items()}),
        _manifest({v: _frame(s) for v, s in schemas.items()}),
        compiled,
    )


def test_exact_household_support_versions_replay():
    expected, actual, compiled = _case()
    graph._check_replayed_survey_manifest(expected, actual, compiled)


@pytest.mark.parametrize(
    "version", ["child_property.donors", "child_property.recipients"]
)
def test_changed_support_schema_is_refused(version):
    expected, actual, compiled = _case()
    actual.populations[version] = _frame(US_SCHEMA)
    with pytest.raises(ValueError, match="SURVEY_PUF55_UPSTREAM_SURVEY_SCHEMA"):
        graph._check_replayed_survey_manifest(expected, actual, compiled)


@pytest.mark.parametrize("field", ["value", "strata", "weight", "mass", "roster"])
def test_heterogeneous_replay_preserves_physical_controls(field):
    expected, actual, compiled = _case()
    version = "child_property.donors"
    frame = actual.populations[version]
    if field == "value":
        frame.person.loc[0, "invented_value"] += 1
    elif field == "strata":
        frame.strata.iloc[0] = "changed"
    elif field == "weight":
        frame._weights["household"] = Weights(np.array([2.0, 2.0]), WeightKind.DESIGN)
    elif field == "mass":
        actual.mass_ledgers[version] = (
            MassRecord(
                "invented", "filter", "declared", 3.0, 2.0, (), (), entity="household"
            ),
        )
    else:
        actual.populations["invented.extra"] = frame
        actual.mass_ledgers["invented.extra"] = ()
        # Extra downstream versions are allowed, but extra expected prefix
        # versions cannot become a source of upstream authority.
        expected.populations["invented.extra"] = frame
    with pytest.raises(ValueError, match="SURVEY_(POPULATION_REPLAY|PUF55)_"):
        graph._check_replayed_survey_manifest(expected, actual, compiled)


def _preflight(monkeypatch, manifest, compiled, check):
    run = SimpleNamespace(manifest=manifest, compiled=compiled)
    monkeypatch.setattr(graph.financial, "check_atomic_survey_financial_run", check)

    def stop_before_construction(*args, **kwargs):
        raise RuntimeError("REACHED_RECIPIENT_QUALIFICATION")

    monkeypatch.setattr(
        graph.recipient.values,
        "qualify_puf55_survey_recipients",
        stop_before_construction,
    )
    return lambda: graph._construct(
        run, {}, fixture_definition=None, seed=578, n_estimators=2, zero_atol=1e-8
    )


def test_preflight_brackets_manifest_reads_with_owner_checks(monkeypatch):
    expected, _, compiled = _case()
    checks = []
    construct = _preflight(
        monkeypatch, expected, compiled, lambda run: checks.append(run)
    )
    with pytest.raises(RuntimeError, match="REACHED_RECIPIENT_QUALIFICATION"):
        construct()
    assert len(checks) == 2
    assert checks[0] is checks[1]


@pytest.mark.parametrize("refuse_on", [1, 2])
def test_owner_refusal_precedes_any_puf_construction(monkeypatch, refuse_on):
    expected, _, compiled = _case()
    checks = []

    def check(run):
        checks.append(run)
        if len(checks) == refuse_on:
            raise ValueError("INVENTED_OWNER_REFUSAL")

    construct = _preflight(monkeypatch, expected, compiled, check)
    with pytest.raises(ValueError, match="INVENTED_OWNER_REFUSAL"):
        construct()
    assert len(checks) == refuse_on


def test_links_refuse_in_preflight_before_any_puf_construction(monkeypatch):
    expected, _, compiled = _case()
    expected.populations["child_property.donors"] = _frame(
        EntitySchema(
            group_entities=("household",),
            links=(LinkSpec("invented_link", "person", "household"),),
        )
    )
    checks = []
    construct = _preflight(
        monkeypatch, expected, compiled, lambda run: checks.append(run)
    )
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"):
        construct()
    assert len(checks) == 1
