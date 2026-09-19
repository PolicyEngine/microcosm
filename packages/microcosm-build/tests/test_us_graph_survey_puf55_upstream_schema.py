"""Tiny real graph replay plus refusal controls; no financial owner is issued."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_survey_puf55 as graph
from microcosm.build.us_runtime.survey_population_replay import (
    same_replayed_population,
)
from microcosm.frame import (
    US_SCHEMA,
    EntitySchema,
    Frame,
    LinkSpec,
    WeightKind,
    Weights,
)
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source_bytes,
    run_graph,
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


class _InventedCreate(KernelBase):
    ref = "invented.heterogeneous-create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def __init__(self):
        self.calls = 0

    def run(self, context):
        self.calls += 1
        assert (
            load_source_bytes("raw-bytes-v1", context.sources["invented"])
            == b"invented two-person support\n"
        )
        schema = (
            US_SCHEMA
            if context.params["full_survey"]
            else EntitySchema(group_entities=("household",))
        )
        return KernelResult(frame=_frame(schema))


def test_household_support_create_store_and_required_replay(tmp_path):
    source = tmp_path / "invented.txt"
    source.write_bytes(b"invented two-person support\n")
    versions = (
        "survey_population.create",
        "child_property.donors",
        "child_property.recipients",
    )
    compiled = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("invented", "raw-bytes-v1"),),
            nodes=tuple(
                Node(
                    version,
                    _InventedCreate.ref,
                    sources=("invented",),
                    structural=StructuralDelta.CREATE,
                    outputs=(Owned("person", "invented_value", "float64"),),
                    params={"full_survey": version == versions[0]},
                )
                for version in versions
            ),
        )
    )
    manifests, observations = [], []
    for resume in ("forbid", "require"):
        kernel, registry, observed = _InventedCreate(), KernelRegistry(), {}
        registry.register(kernel)
        manifests.append(
            run_graph(
                compiled,
                sources={"invented": source},
                store=ContentStore(tmp_path / "store"),
                kernels=registry,
                resume=resume,
                _population_observer=observed.__setitem__,
            )
        )
        observations.append(observed)
        assert kernel.calls == (3 if resume == "forbid" else 0)
        assert all(
            receipt.hit is (resume == "require")
            for receipt in manifests[-1].nodes.values()
        )
    graph._check_replayed_survey_manifest(*manifests, compiled)
    assert set(observations[0]) == set(observations[1]) == set(versions)
    for version in versions:
        expected, replayed = (observation[version] for observation in observations)
        assert expected.frame is not replayed.frame
        assert replayed.frame.schema == (
            US_SCHEMA
            if version == versions[0]
            else EntitySchema(group_entities=("household",))
        )
        same_replayed_population(expected, replayed)


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


def _preflight(monkeypatch, manifest, compiled, check, events=None):
    run = SimpleNamespace(manifest=manifest, compiled=compiled)
    monkeypatch.setattr(graph.financial, "check_atomic_survey_financial_run", check)

    def unexpected_access(*args, **kwargs):
        pytest.fail("Source state or PUF construction reached before refusal")

    monkeypatch.setattr(graph.financial, "_run_entry", unexpected_access)
    monkeypatch.setattr(graph.canonical, "CanonicalPuf55DonorKernel", unexpected_access)
    monkeypatch.setattr(graph.attach, "Boundary", unexpected_access)

    def stop_before_property_check(*args, **kwargs):
        if events is not None:
            events.append("recipient_property_check")
        raise RuntimeError("REACHED_RECIPIENT_PROPERTY_CHECK")

    # Keep the real recipient qualifier: its first owner check must close the
    # bracket before property/source access. These are refusal/sequencing tests,
    # not a substitute issuer or positive financial-owner proof.
    monkeypatch.setattr(
        graph.financial, "require_complete_property_taxes", stop_before_property_check
    )
    return lambda: graph._construct(
        run, {}, fixture_definition=None, seed=578, n_estimators=2, zero_atol=1e-8
    )


def test_preflight_brackets_manifest_reads_with_owner_checks(monkeypatch):
    expected, _, compiled = _case()
    events = []
    population = expected.population

    def read(version):
        events.append("manifest_access")
        return population(version)

    expected.population = read
    construct = _preflight(
        monkeypatch,
        expected,
        compiled,
        lambda run: events.append("owner_check"),
        events,
    )
    with pytest.raises(RuntimeError, match="REACHED_RECIPIENT_PROPERTY_CHECK"):
        construct()
    assert events == (
        ["owner_check"]
        + ["manifest_access"] * 6
        + ["owner_check", "recipient_property_check"]
    )


def test_manifest_accessor_mutation_is_refused_by_real_qualifier_first_check(
    monkeypatch,
):
    expected, _, compiled = _case()
    events, retained = [], []
    population = expected.population

    def read(version):
        events.append("manifest_access")
        retained[0].invented_owner_seal = "changed_during_access"
        return population(version)

    def check(run):
        events.append("owner_check")
        if not retained:
            retained.append(run)
            run.invented_owner_seal = "retained"
        elif run.invented_owner_seal != "retained":
            raise ValueError("INVENTED_OWNER_REFUSAL")

    expected.population = read
    construct = _preflight(monkeypatch, expected, compiled, check, events)
    with pytest.raises(ValueError, match="INVENTED_OWNER_REFUSAL"):
        construct()
    assert events == ["owner_check"] + ["manifest_access"] * 6 + ["owner_check"]


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
