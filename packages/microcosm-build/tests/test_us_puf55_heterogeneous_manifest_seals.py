"""Real tiny CREATE/store/replay controls; no financial fixture or fitting.

These compare attached manifest contents. They issue no survey, donor,
calibration or release authority, and never call the full PUF55 host.
"""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_survey_puf55 as host
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
    run_graph,
)
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.population import token_for_dtype

DONOR_VERSION = host.canonical.CANONICAL_DONOR_NODE


class InventedHeterogeneousSource(KernelBase):
    """An actual CREATE loading only a supplied, owned frame-store source."""

    ref = "test.puf55.heterogeneous_frame_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        frame = load_frame_store(context.sources[context.node.sources[0]])
        if context.node.params.get("noncanonical_null_backing", False):
            assert frame.schema == US_SCHEMA
            flag = frame.person["observed_flag"].array
            assert bool(flag._mask[1])
            # A real CREATE result, before graph persistence: no production
            # monkeypatch or normalized-away source-only probe.
            flag._data[1] = True
            assert bool(flag._data[1]) and pd.isna(
                frame.person["observed_flag"].iloc[1]
            )
        return KernelResult(frame=frame)


def _survey_frame():
    groups = np.array([1, 2], dtype=np.int64)
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): groups.copy()})
        for entity in US_SCHEMA.group_entities
    }
    tables["person"] = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3, 4], dtype=np.int64),
            **{
                US_SCHEMA.membership_column(entity): np.repeat(groups, 2)
                for entity in US_SCHEMA.group_entities
            },
            "age": np.array([40, 12, 70, 20], dtype=np.int64),
            "observed_flag": pd.array([True, pd.NA, False, True], dtype="boolean"),
        }
    )
    # Deliberately retain nonzero backing under a missing cell. The ordinary
    # source codec may canonicalize it; the separate CREATE-result probe below
    # reintroduces it only after that source decode, before graph persistence.
    tables["person"]["observed_flag"].array._data[1] = True
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 0.0]), WeightKind.DESIGN)},
        metadata={},
    )


def _donor_frame():
    ids = np.array([1, 2, 3], dtype=np.int64)
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": ids.copy(), "person_tax_unit_id": ids.copy()}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": ids.copy(),
                    "amount": np.array([10.0, -2.0, 0.0], dtype=np.float64),
                    "observed_count": pd.array([1, 0, 3], dtype="Int64"),
                }
            ),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(np.array([3.0, 1.0, 0.0]), WeightKind.DESIGN)},
        metadata={"fixture": "invented_donor", "revision": 1},
    )


def _outputs(frame):
    return tuple(
        Owned(entity, column, token_for_dtype(frame.table(entity)[column].dtype))
        for entity in frame.entities
        for column in frame.table(entity)
        if column != frame.schema.entity_id_column(entity)
        and not (
            entity == frame.schema.person_entity
            and column
            in {frame.schema.membership_column(g) for g in frame.schema.group_entities}
        )
    )


def _run_pair(root, donor, *, noncanonical_null_backing=False):
    store = ContentStore(root, codecs={"frame-store": load_frame_store})
    frames = {"survey": _survey_frame(), DONOR_VERSION: donor}
    sources, nodes, references = {}, [], []
    for name, frame in frames.items():
        source_name = "invented." + name
        sources[source_name] = store.put_frame(
            hashlib.sha256(("heterogeneous-source:" + name).encode()).hexdigest(),
            frame,
        )
        references.append(SourceRef(source_name, "frame-store"))
        nodes.append(
            Node(
                name,
                InventedHeterogeneousSource.ref,
                sources=(source_name,),
                structural=StructuralDelta.CREATE,
                outputs=_outputs(frame),
                params=(
                    {"noncanonical_null_backing": True}
                    if name == "survey" and noncanonical_null_backing
                    else {}
                ),
            )
        )
    graph = Graph("us", tuple(references), tuple(nodes))
    compiled = compile_graph(graph)
    survey_compiled = compile_graph(Graph("us", (references[0],), (nodes[0],)))
    registry = KernelRegistry()
    registry.register(InventedHeterogeneousSource())
    cold = run_graph(compiled, sources=sources, store=store, kernels=registry)
    warm = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert set(compiled.order) == {"survey", DONOR_VERSION}
    assert all(not receipt.hit for receipt in cold.nodes.values())
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert cold.key == warm.key
    return SimpleNamespace(
        compiled=compiled,
        survey_compiled=survey_compiled,
        cold=cold,
        warm=warm,
    )


@pytest.fixture(scope="module")
def mixed_manifest(tmp_path_factory):
    return _run_pair(tmp_path_factory.mktemp("heterogeneous-manifest"), _donor_frame())


def _survey_manifest(manifest):
    """Scope actual tiny CREATE receipts/attachments; no issuer is fabricated."""
    return replace(
        manifest,
        nodes={"survey": manifest.node("survey")},
        populations={"survey": manifest.population("survey")},
        mass_ledgers={"survey": manifest.mass_ledger("survey")},
    )


def _copy_survey(frame):
    return Frame(
        {entity: frame.table(entity).copy(deep=True) for entity in frame.entities},
        frame.schema,
        {
            entity: Weights(
                frame.weights_for(entity).values.copy(), frame.weights_for(entity).kind
            )
            for entity in frame.weighted_entities
        },
        frame.strata.copy(deep=True),
        mass_log=frame.mass_log,
        metadata=dict(frame.metadata),
    )


def test_real_two_create_graph_has_distinct_schemas_and_required_hits(mixed_manifest):
    case = mixed_manifest
    for manifest in (case.cold, case.warm):
        survey, donor = (
            manifest.population("survey"),
            manifest.population(DONOR_VERSION),
        )
        assert survey.schema == US_SCHEMA and len(survey.entities) == 6
        assert donor.schema == EntitySchema(group_entities=("tax_unit",))
        assert donor.entities == ("person", "tax_unit")
        assert survey.n("person") == 4 and donor.n("person") == 3
        assert (
            manifest.mass_ledger("survey") == manifest.mass_ledger(DONOR_VERSION) == ()
        )
        assert survey.links == donor.links == ()
        assert survey.person["observed_flag"].isna().tolist() == [
            False,
            True,
            False,
            False,
        ]


def test_survey_only_helper_refuses_the_actual_mixed_roster(mixed_manifest):
    for manifest in (mixed_manifest.cold, mixed_manifest.warm):
        with pytest.raises(ValueError, match="^FRAME_TYPE$"):
            host.financial._manifest_population_seals(manifest, mixed_manifest.compiled)


def test_heterogeneous_seal_includes_donor_and_is_store_stable(mixed_manifest):
    case = mixed_manifest
    seals = host._heterogeneous_manifest_seals(case.cold, case.compiled)
    assert type(seals) is tuple
    assert tuple(name for name, _ in seals) == tuple(sorted((DONOR_VERSION, "survey")))
    assert len(dict(seals)) == len(set(case.compiled.versions.values())) == 2
    # This independent existing donor helper is equivalent only for empty ledgers.
    assert case.cold.mass_ledger(DONOR_VERSION) == ()
    assert dict(seals)[DONOR_VERSION] == host.canonical._frame_seal(
        case.cold.population(DONOR_VERSION)
    )
    assert seals == host._heterogeneous_manifest_seals(case.warm, case.compiled)
    assert all(type(row) is tuple and all(type(v) is str for v in row) for row in seals)
    with pytest.raises(TypeError):
        seals[0][1] = "replacement"
    # Both missing-content failures remain closed without a full host invocation.
    without_donor = replace(
        case.cold, populations={"survey": case.cold.population("survey")}
    )
    with pytest.raises(KeyError, match="Population .* is not attached"):
        host._heterogeneous_manifest_seals(without_donor, case.compiled)
    without_ledger = replace(case.cold, mass_ledgers={"survey": ()})
    with pytest.raises(KeyError, match="Mass ledger .* is not attached"):
        host._heterogeneous_manifest_seals(without_ledger, case.compiled)


def test_upstream_survey_roster_comparison_is_store_stable(mixed_manifest):
    case = mixed_manifest
    expected = _survey_manifest(case.cold)
    assert tuple(expected.populations) == ("survey",)
    assert (
        host._check_replayed_survey_manifest(expected, case.cold, case.survey_compiled)
        is None
    )
    assert (
        host._check_replayed_survey_manifest(expected, case.warm, case.survey_compiled)
        is None
    )
    full = dict(host._heterogeneous_manifest_seals(case.cold, case.compiled))
    subset = dict(host._heterogeneous_manifest_seals(case.cold, case.survey_compiled))
    assert tuple(subset) == ("survey",)
    assert subset["survey"] == full["survey"]
    # The two different seal algorithms are intentionally not cross-compared.


def test_survey_seal_replay_probe_retains_nonzero_missing_backing(tmp_path):
    """A failure here is a real remaining cross-store seal blocker, not an xfail."""
    case = _run_pair(
        tmp_path / "noncanonical-backing",
        _donor_frame(),
        noncanonical_null_backing=True,
    )
    cold_flag = case.cold.population("survey").person["observed_flag"]
    warm_flag = case.warm.population("survey").person["observed_flag"]
    assert pd.isna(cold_flag.iloc[1]) and pd.isna(warm_flag.iloc[1])
    # The actual kernel asserted nonzero backing just before KernelResult.
    # Canonicalization before cold-manifest retention is valid lifecycle behavior.
    try:
        host._check_replayed_survey_manifest(
            _survey_manifest(case.cold), case.warm, case.survey_compiled
        )
    except ValueError as error:
        raise AssertionError(
            "Upstream replay comparison failed for this valid nullable US CREATE; "
            f"cold_missing_backing={bool(cold_flag.array._data[1])}, "
            f"warm_missing_backing={bool(warm_flag.array._data[1])}"
        ) from error


@pytest.mark.parametrize(
    "change", ("value", "mask", "dtype", "column_order", "weight", "metadata")
)
def test_each_donor_change_alters_only_its_manifest_seal(
    mixed_manifest, tmp_path, change
):
    original = _donor_frame()
    tables = {e: original.table(e).copy(deep=True) for e in original.entities}
    weights = original.weights_for("tax_unit").values.copy()
    metadata = dict(original.metadata)
    table = tables["tax_unit"]
    if change == "value":
        table.loc[0, "amount"] = 11.0
        assert table.loc[0, "amount"] != original.table("tax_unit").loc[0, "amount"]
    elif change == "mask":
        table.loc[0, "observed_count"] = pd.NA
        assert table["observed_count"].isna().tolist() == [True, False, False]
        assert not original.table("tax_unit")["observed_count"].isna().any()
    elif change == "dtype":
        table["amount"] = table["amount"].astype("float32")
        assert table["amount"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            table["amount"], original.table("tax_unit")["amount"]
        )
    elif change == "column_order":
        tables["tax_unit"] = table.loc[:, ["tax_unit_id", "observed_count", "amount"]]
        assert tuple(tables["tax_unit"]) != tuple(original.table("tax_unit"))
    elif change == "weight":
        weights[0] = 4.0
        assert weights[0] != original.weights_for("tax_unit").values[0]
    else:
        assert change == "metadata"
        metadata["revision"] = 2
        assert metadata != dict(original.metadata)
    changed = Frame(
        tables,
        original.schema,
        {"tax_unit": Weights(weights, WeightKind.DESIGN)},
        original.strata.copy(deep=True),
        metadata=metadata,
    )
    # Every variation goes through actual source encoding, CREATE and required replay.
    rerun = _run_pair(tmp_path / "changed-store", changed)
    before = dict(
        host._heterogeneous_manifest_seals(mixed_manifest.cold, mixed_manifest.compiled)
    )
    after = dict(host._heterogeneous_manifest_seals(rerun.cold, rerun.compiled))
    assert set(before) == set(after) == {DONOR_VERSION, "survey"}
    assert before[DONOR_VERSION] != after[DONOR_VERSION]
    assert before["survey"] == after["survey"]
    assert tuple(after.items()) == host._heterogeneous_manifest_seals(
        rerun.warm, rerun.compiled
    )
    for actual in (rerun.cold, rerun.warm):
        assert (
            host._check_replayed_survey_manifest(
                _survey_manifest(mixed_manifest.cold),
                actual,
                mixed_manifest.survey_compiled,
            )
            is None
        )


def test_explicit_link_body_is_refused_before_content_sealing(mixed_manifest):
    base = mixed_manifest.cold.population(DONOR_VERSION)
    schema = EntitySchema(
        group_entities=("tax_unit",),
        links=(LinkSpec("claims", "person", "tax_unit"),),
    )
    linked = Frame(
        {
            **{e: base.table(e).copy(deep=True) for e in base.entities},
            "claims": pd.DataFrame(
                {
                    "person_id": np.array([1, 2], dtype=np.int64),
                    "tax_unit_id": np.array([2, 3], dtype=np.int64),
                    "share": np.array([0.25, 0.75], dtype=np.float64),
                }
            ),
        },
        schema,
        {
            "tax_unit": Weights(
                base.weights_for("tax_unit").values.copy(), WeightKind.DESIGN
            )
        },
        base.strata.copy(deep=True),
        metadata=dict(base.metadata),
    )
    assert linked.links == ("claims",) and len(linked.link("claims")) == 2
    # A public descriptive manifest copy is sufficient for a negative helper input;
    # its copied node receipts confer no new source or graph admission.
    manifest = replace(
        mixed_manifest.cold,
        populations={
            "survey": mixed_manifest.cold.population("survey"),
            DONOR_VERSION: linked,
        },
    )
    with pytest.raises(ValueError, match="^SURVEY_PUF55_MANIFEST_SEAL_LINK_TABLES$"):
        host._heterogeneous_manifest_seals(manifest, mixed_manifest.compiled)


@pytest.mark.parametrize(
    "fault,reason",
    (
        ("known_value", "SURVEY_POPULATION_REPLAY_NATIVE_BITS"),
        ("mask", "SURVEY_POPULATION_REPLAY_MASKED_STORAGE"),
        ("dtype", "SURVEY_POPULATION_REPLAY_SERIES_DTYPE_OR_LENGTH"),
        ("column_order", "SURVEY_POPULATION_REPLAY_AXIS"),
        ("column_roster", "SURVEY_POPULATION_REPLAY_AXIS"),
        ("entity_roster", "SURVEY_PUF55_UPSTREAM_SURVEY_SCHEMA"),
        ("weight", "SURVEY_POPULATION_REPLAY_WEIGHT_BYTES"),
        ("metadata", "SURVEY_POPULATION_REPLAY_FRAME_CONTEXT"),
        (
            "noncanonical_null_backing",
            "SURVEY_POPULATION_REPLAY_NONCANONICAL_NULL_BACKING",
        ),
    ),
)
def test_replayed_survey_refuses_changed_actual_content(mixed_manifest, fault, reason):
    # Start from actual required-store content. Only a detached descriptive
    # negative input changes; the shared source/fixture manifest is untouched.
    expected = _survey_manifest(mixed_manifest.warm)
    original = expected.population("survey")
    changed = _copy_survey(original)
    tables = {entity: changed.table(entity) for entity in changed.entities}
    weights = {
        entity: changed.weights_for(entity) for entity in changed.weighted_entities
    }
    metadata = dict(changed.metadata)
    person = tables["person"]
    if fault == "known_value":
        person.loc[0, "age"] += 1
        assert person.loc[0, "age"] != original.person.loc[0, "age"]
    elif fault == "mask":
        person.loc[0, "observed_flag"] = pd.NA
        assert pd.isna(person.loc[0, "observed_flag"])
        assert not pd.isna(original.person.loc[0, "observed_flag"])
    elif fault == "dtype":
        person["age"] = person["age"].astype("int32")
        assert person["age"].dtype != original.person["age"].dtype
        np.testing.assert_array_equal(person["age"], original.person["age"])
    elif fault == "column_order":
        columns = [*person.columns[:-2], *reversed(person.columns[-2:])]
        tables["person"] = person.loc[:, columns]
        assert tuple(tables["person"]) != tuple(original.person)
    elif fault == "column_roster":
        tables["person"] = person.drop(columns=["age"])
        assert "age" not in tables["person"] and "age" in original.person
    elif fault == "entity_roster":
        changed = _donor_frame()
        assert changed.schema != original.schema
    elif fault == "weight":
        values = original.weights_for("household").values.copy()
        values[0] += 1
        weights["household"] = Weights(values, WeightKind.DESIGN)
        assert values[0] != original.weights_for("household").values[0]
    elif fault == "metadata":
        metadata["invented_changed_metadata"] = True
        assert metadata != dict(original.metadata)
    else:
        assert fault == "noncanonical_null_backing"
        # The required-store side is the expected side here; a newly nonzero
        # missing buffer in actual cannot use the one-way canonical-zero rule.
        assert bool(original.person["observed_flag"].array._mask[1])
        assert not bool(original.person["observed_flag"].array._data[1])
        person["observed_flag"].array._data[1] = True
        assert bool(person["observed_flag"].array._data[1])
        assert pd.isna(person["observed_flag"].iloc[1])
    if fault != "entity_roster":
        changed = Frame(
            tables,
            changed.schema,
            weights,
            changed.strata.copy(deep=True),
            mass_log=changed.mass_log,
            metadata=metadata,
        )
    actual = replace(expected, populations={"survey": changed})
    with pytest.raises(ValueError, match="^" + reason + "$"):
        host._check_replayed_survey_manifest(
            expected, actual, mixed_manifest.survey_compiled
        )


@pytest.mark.parametrize(
    "fault",
    (
        "expected_population",
        "expected_ledger",
        "actual_population",
        "actual_ledger",
        "compiled_roster",
    ),
)
def test_replayed_survey_requires_exact_upstream_roster(mixed_manifest, fault):
    expected = _survey_manifest(mixed_manifest.cold)
    actual = mixed_manifest.warm  # The legitimate extra donor version is allowed.
    compiled = mixed_manifest.survey_compiled
    if fault == "expected_population":
        expected = replace(expected, populations={})
    elif fault == "expected_ledger":
        expected = replace(expected, mass_ledgers={})
    elif fault == "actual_population":
        actual = replace(
            actual, populations={DONOR_VERSION: actual.population(DONOR_VERSION)}
        )
    elif fault == "actual_ledger":
        actual = replace(
            actual, mass_ledgers={DONOR_VERSION: actual.mass_ledger(DONOR_VERSION)}
        )
    else:
        assert fault == "compiled_roster"
        compiled = mixed_manifest.compiled
    with pytest.raises(ValueError, match="^SURVEY_PUF55_UPSTREAM_SURVEY_ROSTER$"):
        host._check_replayed_survey_manifest(expected, actual, compiled)


def test_replayed_survey_refuses_changed_mass_ledger(mixed_manifest):
    expected = _survey_manifest(mixed_manifest.warm)
    assert expected.mass_ledger("survey") == ()
    record = host.population_ops.MassRecord(
        node_id="invented.extra",
        operation="filter",
        policy="preserve",
        before_total=2.0,
        after_total=2.0,
        before_by_stratum=(),
        after_by_stratum=(),
        entity="household",
    )
    actual = replace(expected, mass_ledgers={"survey": (record,)})
    assert actual.mass_ledger("survey") != expected.mass_ledger("survey")
    with pytest.raises(
        ValueError, match="^SURVEY_POPULATION_REPLAY_POPULATION_CONTEXT$"
    ):
        host._check_replayed_survey_manifest(
            expected, actual, mixed_manifest.survey_compiled
        )


def test_full_manifest_physical_seal_still_detects_late_masked_bits(mixed_manifest):
    expected = _survey_manifest(mixed_manifest.warm)
    detached = _copy_survey(expected.population("survey"))
    actual = replace(
        mixed_manifest.warm,
        populations={
            "survey": detached,
            DONOR_VERSION: mixed_manifest.warm.population(DONOR_VERSION),
        },
    )
    assert (
        host._check_replayed_survey_manifest(
            expected, actual, mixed_manifest.survey_compiled
        )
        is None
    )
    before = host._heterogeneous_manifest_seals(actual, mixed_manifest.compiled)
    flag = actual.population("survey").person["observed_flag"].array
    assert bool(flag._mask[1]) and not bool(flag._data[1])
    flag._data[1] = True
    assert bool(flag._data[1]) and bool(flag._mask[1])
    assert host._heterogeneous_manifest_seals(actual, mixed_manifest.compiled) != before
