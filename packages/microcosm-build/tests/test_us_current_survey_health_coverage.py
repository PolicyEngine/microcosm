"""Invented health source values, exact source joins and two-clone attachment."""

import hashlib
import io
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_health_coverage as health
from microcosm.build.us_runtime import current_survey_health_source as source


def _health_acs_person(original):
    """Decorate only an invented fixture's source-row constructor, before pins."""

    def person(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update({c: "2" for c in health.ACS_VALUE_COLUMNS})
        row.update(
            {c: "0" for c in (*health.ACS_ALLOCATION_COLUMNS, *health.ACS_EDIT_COLUMNS)}
        )
        row["HINS1"] = "1"
        row["HINS7"] = "" if int(row["SPORDER"]) == 2 else "2"
        return row

    return person


def _health_asec_table(table):
    """Add source fields without changing existing money or coverage literals."""
    table = table.copy(deep=True)
    for c in health.ASEC_VALUE_COLUMNS:
        if c not in table:
            table[c] = "2"
    for c in health.ASEC_ALLOCATION_COLUMNS:
        if c not in table:
            table[c] = "0"
    return table


def _health_source_arguments(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as preparation_fixture
    from test_us_current_asec_demographics import _demographic_arguments

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographic

    monkeypatch.setattr(
        preparation_fixture, "_person", _health_acs_person(preparation_fixture._person)
    )
    arguments = _demographic_arguments(tmp_path, monkeypatch, zero=False)
    source_dir = arguments["source_dir"] / "asec"
    members, pins = {}, []
    for year, member, archive, *_ in source.asec._MEMBER_PINS:
        path = source_dir / member
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        table = _health_asec_table(original)
        pd.testing.assert_frame_equal(table.loc[:, original.columns], original)
        table.to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(table),
                len(data),
            )
        )
        members[year] = path
    for module in (source.asec, restoration, demographic.demographic):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "health-restored-money"
    restoration.restore_asec_person_income_source(
        source_dir / "parent.h5",
        source_dir / "household-attachment.h5",
        member_paths=members,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME,
        source_dir / "person-income-attachment.h5",
    )
    return arguments


def test_actual_health_qualifier_uses_retained_original_source_issuers(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import graph_current_survey_health as graph

    arguments = _health_source_arguments(tmp_path, monkeypatch)
    prepared = source.source.prepare_authenticated_survey_population(**arguments)
    original = prepared.checked_view().frame
    retained = {e: original.table(e).copy(deep=True) for e in original.entities}
    qualified = graph.qualify_health_coverage(prepared)
    assert qualified.source_frame is original
    assert qualified.origins.index.tolist() == original.person.person_id.tolist()
    assert set(qualified.raw.source) == {"asec", "acs"}
    assert qualified.evidence["asec_coverage_observation_year"] == 2025
    assert qualified.evidence["acs_coverage_observation_year"] == 2024
    assert not qualified.evidence["source_admission_issued"]
    assert not qualified.evidence["unallocated_observation_claim"]
    assert not qualified.evidence["release_eligible"]
    for field in health.FIELDS:
        result = health.recode_field(qualified.raw, field.output)
        asec = qualified.raw.source.eq("asec")
        assert result.loc[asec, field.output + "__known"].all()
        if field.acs is None:
            assert result.loc[~asec, field.output].isna().all()
    acs = qualified.raw.source.eq("acs")
    assert qualified.raw.loc[acs, "NOW_CAID"].map(lambda v: v is None).all()
    assert qualified.raw.loc[acs, "HINS7"].eq("").any()
    for entity, expected in retained.items():
        pd.testing.assert_frame_equal(
            expected, original.table(entity), check_exact=True
        )
    seal = graph.health_coverage_seal(qualified)
    fresh = graph.qualify_health_coverage(prepared)
    assert graph.health_coverage_seal(fresh) == seal
    qualified.raw.loc[qualified.raw.source.eq("asec"), "NOW_CAID"] = "1"
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        graph.health_coverage_seal(qualified)
    assert graph.health_coverage_seal(fresh) == seal


def invented_raw():
    index = pd.Index([10, 20], name="person_id")
    raw = pd.DataFrame(
        {
            "source": ["asec", "acs"],
            "source_year": [2024, 2024],
            "survey_year": [2025, 2024],
        },
        index=index,
    )
    for c in health.RAW_COLUMNS:
        raw[c] = pd.Series([None] * len(index), index=index, dtype=object)
    for c in health.ASEC_VALUE_COLUMNS:
        raw.loc[10, c] = "2"
    for c in health.ASEC_ALLOCATION_COLUMNS:
        raw.loc[10, c] = "0"
    for c in health.ACS_VALUE_COLUMNS:
        raw.loc[20, c] = "2"
    for c in (*health.ACS_ALLOCATION_COLUMNS, *health.ACS_EDIT_COLUMNS):
        raw.loc[20, c] = "0"
    raw.loc[10, "NOW_GRP"] = raw.loc[20, "HINS1"] = "1"
    raw.loc[10, "NOW_MCAID"] = "1"  # Broad aggregate yes, exact Medicaid no.
    raw.loc[20, "HINS7"] = ""
    return raw


def invented_origins():
    return pd.DataFrame(
        {"source": ["asec", "acs"], "native_person_id": [2, 3]},
        index=pd.Index([10, 20], name="person_id"),
    )


def invented_receiving():
    p = health.provenance
    return SimpleNamespace(
        person=pd.DataFrame(
            {
                "person_id": np.array([400, 100, 300, 200], dtype="int64"),
                p.support_source_id_column("person"): np.array(
                    [20, 10, 20, 10], dtype="int64"
                ),
                p.support_clone_index_column("person"): np.array(
                    [1, 0, 0, 1], dtype="int64"
                ),
                p.spine_source_id_column("person"): np.array(
                    [3, 2, 3, 2], dtype="int64"
                ),
                p.support_channel_column("person"): ["acs", "asec", "acs", "asec"],
                "unemployment_compensation": np.array(
                    [np.nan, 0, np.nan, 0], dtype="float64"
                ),
            }
        )
    )


@pytest.mark.parametrize(
    "survey,flags,labels",
    [
        (
            "asec",
            ["0", "1", "2", "3", "", "9"],
            [
                "reported",
                "hotdeck",
                "logical",
                "whole_unit",
                "allocation_unknown",
                "allocation_unknown",
            ],
        ),
        (
            "acs",
            ["0", "1", "", "2", "0", "1"],
            [
                "not_allocated",
                "allocated",
                "allocation_unknown",
                "allocation_unknown",
                "not_allocated",
                "allocated",
            ],
        ),
    ],
)
def test_yes_no_and_allocation_are_separate(survey, flags, labels):
    result = health.recode(["1", "2", "1", "2", "1", "2"], flags, survey=survey)
    assert result.value.tolist() == [True, False, True, False, True, False]
    assert result.known.all()
    assert result.status.tolist() == [
        ("source_yes_" if i % 2 == 0 else "source_no_") + label
        for i, label in enumerate(labels)
    ]


@pytest.mark.parametrize(
    "token,status",
    [
        ("", "missing_source_value"),
        ("0", "unrecognized_source_value"),
        ("9", "unrecognized_source_value"),
        (" 1", "unrecognized_source_value"),
        ("1.0", "unrecognized_source_value"),
        ("-1", "unrecognized_source_value"),
    ],
)
def test_unknown_codes_never_turn_into_false(token, status):
    result = health.recode([token], ["0"], survey="asec")
    assert pd.isna(result.value.iloc[0])
    assert not result.known.iloc[0]
    assert result.status.iloc[0] == status


@pytest.mark.parametrize("token", [True, 1, None, np.int64(1)])
def test_recode_does_not_coerce_nonliteral_inputs(token):
    with pytest.raises(ValueError, match="LITERAL_CONTRACT"):
        health.recode([token], ["0"], survey="acs")


def test_medicaid_specific_recode_and_closed_profile_roster():
    raw = invented_raw()
    medicaid = health.recode_field(raw, "has_medicaid_health_coverage_at_interview")
    assert medicaid.iloc[0, 0] == False  # noqa: E712
    assert raw.loc[10, "NOW_MCAID"] == "1"
    assert set(f.output for f in health.FIELDS) == set(
        health.US_REPORTED_COVERAGE_PERSON_INPUTS
    )
    with pytest.raises(ValueError, match="FIELD"):
        health.recode_field(raw, "medicaid_eligible")


def test_acs_exact_values_and_all_semantic_gaps():
    raw = invented_raw()
    for field in health.FIELDS:
        result = health.recode_field(raw, field.output)
        assert not pd.isna(result.loc[10, field.output])
        if field.acs is None:
            assert pd.isna(result.loc[20, field.output])
            assert not result.loc[20, field.output + "__known"]
            assert result.loc[20, field.output + "__source_status"].startswith(
                "semantic_gap:"
            )
    assert health.recode_field(raw, "has_esi").loc[20, "has_esi"]
    ihs = health.recode_field(raw, health.FIELDS[-1].output)
    assert pd.isna(ihs.iloc[1, 0])
    assert ihs.iloc[1, 2] == "missing_source_value"
    literal = health.source_columns(raw)
    assert pd.isna(literal.loc[10, "health_source_HINS1"])
    assert literal.loc[20, "health_source_HINS7"] == ""
    assert literal.loc[20, "health_source_HINS4"] == "2"
    assert literal.health_source_survey_year.tolist() == [2025, 2024]


def test_attachment_uses_original_identity_and_preserves_unknowns():
    receiving, origins = invented_receiving(), invented_origins()
    before = receiving.person.copy(deep=True)
    columns = health.recode_field(invented_raw(), health.FIELDS[-1].output)
    result = health.attach_columns(origins, receiving, columns)
    value = result["person", health.FIELDS[-1].output]
    assert value.index.tolist() == [400, 100, 300, 200]
    assert pd.isna(value.loc[400]) and pd.isna(value.loc[300])
    assert value.loc[100] == value.loc[200] == False  # noqa: E712
    pd.testing.assert_frame_equal(receiving.person, before, check_exact=True)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("duplicate_id", "CLONE_SOURCE_ROSTER"),
        ("duplicate_clone", "CLONE_PAIR"),
        ("foreign_origin", "CLONE_SOURCE_ROSTER"),
        ("wrong_native", "CLONE_SOURCE_IDENTITY"),
        ("wrong_channel", "CLONE_SOURCE_IDENTITY"),
        ("collision", "OWNERSHIP_COLLISION"),
    ],
)
def test_attachment_refuses_invalid_clone_identity_or_existing_output(mutation, reason):
    receiving = invented_receiving()
    p = health.provenance
    name, value = {
        "duplicate_id": ("person_id", 100),
        "duplicate_clone": (p.support_clone_index_column("person"), 0),
        "foreign_origin": (p.support_source_id_column("person"), 99),
        "wrong_native": (p.spine_source_id_column("person"), 2),
        "wrong_channel": (p.support_channel_column("person"), "asec"),
        "collision": ("has_esi", False),
    }[mutation]
    receiving.person.loc[0, name] = value
    with pytest.raises(ValueError, match=reason):
        health.attach_columns(
            invented_origins(),
            receiving,
            health.recode_field(invented_raw(), "has_esi"),
        )


def _csv(survey, *, second=None):
    columns = source.ASEC_COLUMNS if survey == "asec" else source.ACS_COLUMNS
    row = {c: "0" if c.startswith(("I_", "F")) else "2" for c in columns}
    row.update(
        {"PERIDNUM": "0123456789012345678901", "PH_SEQ": "7", "A_LINENO": "1"}
        if survey == "asec"
        else {"SERIALNO": "2024HU0000001", "SPORDER": "1"}
    )
    rows = [row] + ([] if second is None else [{**row, **second}])
    body = (
        ",".join(columns)
        + "\n"
        + "\n".join(",".join(r[c] for c in columns) for r in rows)
        + "\n"
    )
    return body.encode(), row


@pytest.mark.parametrize("survey", ["asec", "acs"])
def test_literal_scan_preserves_values_and_exhausts_tail(survey):
    body, row = _csv(survey)
    key = source._key(row, survey)
    selected = {}
    assert (
        source._scan(
            io.BytesIO(body), survey=survey, wanted={key}, selected=selected, maximum=1
        )
        == 1
    )
    assert selected == {key: row}
    with pytest.raises(ValueError, match="SOURCE_ROW_SHAPE"):
        source._scan(
            io.BytesIO(body + b"malformed\n"),
            survey=survey,
            wanted={key},
            selected={},
            maximum=2,
        )
    body, _ = _csv(survey, second={})
    with pytest.raises(ValueError, match="DUPLICATE_SELECTED"):
        source._scan(
            io.BytesIO(body), survey=survey, wanted={key}, selected={}, maximum=2
        )


def test_combine_refuses_missing_and_changed_source_keys():
    body, row = _csv("asec")
    del body
    key = source._key(row, "asec")
    origins = invented_origins().iloc[:1]
    origins = origins.assign(source_year=2024, survey_year=2025)
    keys = {("asec", key): 10}
    selected = {"asec": {key: row}, "acs": {}}
    result = source._combine(origins, keys, selected)
    assert result.loc[10, "NOW_CAID"] == "2"
    with pytest.raises(ValueError, match="SELECTED_SOURCE_ROSTER"):
        source._combine(origins, keys, {"asec": {}, "acs": {}})
    selected["asec"][key] = {**row, "PH_SEQ": "8"}
    with pytest.raises(ValueError, match="SOURCE_COORDINATE_CHANGED"):
        source._combine(origins, keys, selected)


def test_public_values_do_not_authorize_native_projection():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        source.qualify_current_survey_health(SimpleNamespace(payload=b"{}"))


def _frame(people):
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    people = people.copy(deep=True)
    tables = {"person": people}
    for entity in US_SCHEMA.group_entities:
        people[US_SCHEMA.membership_column(entity)] = people.person_id.to_numpy()
        tables[entity] = pd.DataFrame(
            {US_SCHEMA.entity_id_column(entity): np.sort(people.person_id.to_numpy())}
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(people)), WeightKind.DESIGN)},
    )


def _qualified():
    import hashlib

    from microcosm.build.us_runtime import graph_current_survey_health as graph

    raw = invented_raw()
    origins = invented_origins()
    frame = _frame(pd.DataFrame({"person_id": raw.index.to_numpy()}))
    projection = raw.reset_index().to_json(orient="table", index=False).encode()
    q = source.QualifiedSurveyHealthCoverage(
        frame,
        origins,
        raw,
        projection,
        {
            "projection_sha256": hashlib.sha256(projection).hexdigest(),
            "fixture": "invented; no source owner or data admission claimed",
        },
    )
    graph.health_coverage_seal(q)
    return q


def _graph_case(tmp_path):
    from microcosm.build.us_runtime import graph_current_survey_health as graph
    from microcosm.graph import (
        ArtifactInput,
        ArtifactOutput,
        ArtifactType,
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
        run_graph,
    )

    qualified = _qualified()
    people = invented_receiving().person
    people[health.provenance.support_channel_column("person")] = pd.array(
        people[health.provenance.support_channel_column("person")], dtype="string"
    )
    receiving = _frame(people)
    attachment_type = ArtifactType("invented.amount_attachment", 1)

    class InventedSource(KernelBase):
        ref = "invented.health_receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "a" * 64  # Test-only fixed producer for the tiny invented Frame.

        def run(self, context):
            return KernelResult(
                frame=receiving, artifacts={"attachment": b"invented amount input"}
            )

    create = Node(
        "invented.amounts",
        InventedSource.ref,
        sources=(graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("person", c, graph._dtype(people[c]))
            for c in people
            if c != "person_id"
        ),
        artifact_outputs=(ArtifactOutput("attachment", attachment_type),),
    )
    after = ArtifactInput("amount_attachment", create.id, "attachment", attachment_type)
    nodes = graph.health_coverage_nodes(
        qualified, receiving_version=create.id, after=after
    )
    compiled = compile_graph(
        Graph("us", (SourceRef(graph.SOURCE_NAME, "raw-bytes-v1"),), (create, *nodes))
    )
    expected_seal = graph.health_coverage_seal(qualified)
    calls = []

    def current():
        assert graph.health_coverage_seal(qualified) == expected_seal
        calls.append(True)

    kernels = KernelRegistry()
    kernels.register(InventedSource())
    for kernel in graph.health_coverage_kernels(
        qualified, receiving_version=create.id, after=after, require_current=current
    ):
        kernels.register(kernel)
    path = tmp_path / "invented-source.txt"
    path.write_bytes(b"invented health fixture only")
    store = ContentStore(tmp_path / "store")
    args = {"sources": {graph.SOURCE_NAME: path}, "store": store, "kernels": kernels}
    cold = run_graph(compiled, **args)
    observed = {}

    def capture(_node_id, population):
        observed[population.version] = population

    warm = run_graph(compiled, **args, resume="require", _population_observer=capture)
    return SimpleNamespace(
        graph=graph,
        qualified=qualified,
        receiving=receiving,
        nodes=nodes,
        compiled=compiled,
        cold=cold,
        warm=warm,
        store=store,
        calls=calls,
        observed=observed,
    )


def test_actual_graph_cold_required_replay_preserves_amount_columns(tmp_path):
    case = _graph_case(tmp_path)
    assert len(case.compiled.order) == 13
    assert case.cold.key == case.warm.key
    assert all(n.hit for n in case.warm.nodes.values())
    assert (
        len(case.calls) == 24
    )  # Before/after 12 cold fragment nodes; replay invokes none.
    output = case.warm.population("invented.amounts")
    pd.testing.assert_frame_equal(
        output.person[case.receiving.person.columns],
        case.receiving.person,
        check_exact=True,
    )
    assert output.person.has_esi.tolist() == [True, True, True, True]
    assert output.person.has_medicaid_health_coverage_at_interview.isna().tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_fragment_dependency_edges_and_explicit_receiving_version():
    from microcosm.build.us_runtime import graph_current_survey_health as graph
    from microcosm.graph import ArtifactInput, ArtifactType

    after = ArtifactInput(
        "amount_attachment",
        "country.amounts",
        "attachment",
        ArtifactType("invented", 1),
    )
    nodes = graph.health_coverage_nodes(
        _qualified(), receiving_version="country.receiving", after=after
    )
    assert nodes[-1].population == "country.receiving"
    assert after in nodes[-1].artifact_inputs
    assert len(nodes[-1].artifact_inputs) == 12
    assert sum(n.id.startswith(graph.RECODE_PREFIX) for n in nodes) == 9
    assert all(n.structural.value == "none" for n in nodes[1:])


def test_qualified_source_and_raw_mutation_changes_full_seal():
    from microcosm.build.us_runtime import graph_current_survey_health as graph

    qualified = _qualified()
    before = graph.health_coverage_seal(qualified)
    qualified.origins.loc[10, "native_person_id"] = 99
    assert graph.health_coverage_seal(qualified) != before
    qualified.raw.loc[10, "NOW_GRP"] = "2"
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        graph.health_coverage_seal(qualified)


def test_actual_graph_independent_complete_reconstruction(tmp_path):
    from microcosm.graph import ArtifactValue, NumericScope
    from microcosm.graph import population as populations

    case = _graph_case(tmp_path)
    expected = {
        "invented.amounts": populations.Population.from_frame(
            case.receiving, "invented.amounts"
        )
    }
    for node_id in case.compiled.order:
        if node_id == "invented.amounts":
            continue
        node = case.compiled.graph.node(node_id)
        artifacts = {}
        for edge in node.artifact_inputs:
            producer = case.warm.node(edge.producer)
            key = producer.opaque_artifacts[edge.artifact]
            artifacts[edge.name] = ArtifactValue(
                case.store.load_bytes(key), edge.type, key, producer.key, NumericScope()
            )
        version = case.compiled.versions[node_id]
        incoming = expected.get(version)
        expected[version] = case.graph.expected_health_population(
            node_id, incoming, qualified=case.qualified, node=node, artifacts=artifacts
        )
    for version, population in expected.items():
        case.graph.physical.replay.same_replayed_population(
            population, case.observed[version]
        )


def test_source_artifact_values_cannot_be_replaced():
    from microcosm.build.us_runtime import graph_current_survey_health as graph
    from microcosm.graph import ArtifactInput, ArtifactType, ArtifactValue, NumericScope

    after = ArtifactInput(
        "amount_attachment", "amounts", "attachment", ArtifactType("invented", 1)
    )
    q = _qualified()
    node = graph.health_coverage_nodes(q, receiving_version="receiving", after=after)[1]
    artifact = ArtifactValue(
        b"{}", graph.PROJECTION_TYPE, "0" * 64, "1" * 64, NumericScope()
    )
    with pytest.raises(ValueError, match="ARTIFACT_PAYLOAD"):
        graph._check_artifacts(node, {"health_projection": artifact}, q)
