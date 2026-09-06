"""The combined-survey PUF-support clone stage, executed through the graph.

Every byte here is invented. The host is a hand-written mixed ASEC/ACS pool
assembled by the real :func:`assemble_spines`: two arms whose raw record ids
deliberately collide, five households, ten people, seven tax units, several
group memberships per household, one zero-weight household, unequal weights,
non-contiguous ids and a person order that is not the id order. No microdata is
read, no donor exists, no model is fitted, no draw is taken.

The stage under test does one thing: attach exactly one whole-household
PUF-detail clone to that host and split every household weight across the pair.
The tests compare the graph result against the direct operator through the
declared ``(source channel, source id, clone role)`` key rather than through raw
integers, so the comparison stays honest about which identity is contractual.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_combined_clone as clone_stage
from microcosm.build.us_runtime.graph_combined_clone import (
    COMBINED_CLONE_CLAIM_NODE,
    COMBINED_CLONE_NODE,
    COMBINED_CLONE_ROLES,
    CombinedCloneError,
    combined_clone_provenance_columns,
    register_us_combined_survey_clone_kernels,
    us_combined_survey_clone_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    PUF_TAX_DETAIL_CLONE_INDEX,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)
from microcosm.graph.codecs import SourceCodecRegistry, load_frame_store
from microcosm.graph.errors import NodeRejectedError
from microcosm.graph.population import dtype_for_token

SOURCE_CHANNELS = ("acs", "asec")

# --------------------------------------------------------------------------
# Invented host
# --------------------------------------------------------------------------


def _string(values) -> pd.Series:
    return pd.Series(values, dtype=dtype_for_token("string"))


def _asec_arm() -> Frame:
    """Three ASEC households: two tax units in one, a zero-weight household."""

    person = pd.DataFrame(
        {
            # Neither sorted nor contiguous, and not grouped by household.
            "person_id": [77, 5, 40, 206, 41, 9, 205],
            "person_household_id": [11, 11, 47, 900, 47, 47, 900],
            "person_tax_unit_id": [3101, 3101, 3140, 3900, 3140, 3155, 3900],
            "person_spm_unit_id": [61, 61, 62, 63, 62, 62, 63],
            "person_family_id": [510, 510, 540, 590, 540, 541, 590],
            "person_marital_unit_id": [8001, 8002, 8040, 8091, 8041, 8055, 8090],
            "age": [44, 9, 61, 30, 58, 22, 31],
            "asec_wage": [51_000.0, 0.0, 12_000.0, 25_000.0, 0.0, 7_000.0, 40_000.0],
            "source_year": _string(
                ["2023", "2023", "2022", "2024", "2022", "2022", "2024"]
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": [11, 47, 900], "state_fips": [6, 36, 48]}
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [3101, 3140, 3155, 3900],
                "filing_status_input": _string(
                    ["HEAD_OF_HOUSEHOLD", "JOINT", "SINGLE", "JOINT"]
                ),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [61, 62, 63]}),
        "family": pd.DataFrame({"family_id": [510, 540, 541, 590]}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": [8001, 8002, 8040, 8041, 8055, 8090, 8091]}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        # One zero-weight household and two unequal positive ones.
        {"household": Weights(np.array([1200.0, 0.0, 730.0]), WeightKind.DESIGN)},
        _string(["asec-a", "asec-a", "asec-b", "asec-c", "asec-b", "asec-b", "asec-c"]),
    )


def _acs_arm() -> Frame:
    """Two ACS households whose raw ids collide with the ASEC arm's."""

    person = pd.DataFrame(
        {
            # 5 and 77 also exist in the ASEC arm, so assembly must remap this
            # arm and the spine-source ids must stay distinguishable from the
            # assembly-unique source ids.
            "person_id": [5, 77, 71],
            "person_household_id": [11, 11, 88],
            "person_tax_unit_id": [700, 701, 780],
            "person_spm_unit_id": [20, 20, 21],
            "person_family_id": [30, 30, 31],
            "person_marital_unit_id": [400, 401, 480],
            "age": [67, 65, 19],
            "acs_income": [8_000.0, 1_500.0, 0.0],
            "source_year": _string(["2024", "2024", "2024"]),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": [11, 88], "state_fips": [12, 24]}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [700, 701, 780],
                "filing_status_input": _string(["SINGLE", "SINGLE", "SINGLE"]),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [20, 21]}),
        "family": pd.DataFrame({"family_id": [30, 31]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [400, 401, 480]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([310.0, 940.0]), WeightKind.DESIGN)},
        _string(["acs-x", "acs-x", "acs-y"]),
    )


def _canonical_strings(frame: Frame) -> Frame:
    """Move every text column onto the store's canonical string storage."""

    tables = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        for column in table.columns:
            if not pd.api.types.is_numeric_dtype(table[column]):
                table[column] = table[column].astype(dtype_for_token("string"))
        tables[entity] = table
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.astype(dtype_for_token("string")),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _pool(asec: Frame | None = None, acs: Frame | None = None) -> Frame:
    """The combined native host: IMPORTANCE household weights, clone index 0."""

    return _canonical_strings(
        assemble_spines(
            {"asec": asec or _asec_arm(), "acs": acs or _acs_arm()},
            household_mass_shares={"asec": 0.6, "acs": 0.4},
        )
    )


# --------------------------------------------------------------------------
# Graph harness
# --------------------------------------------------------------------------


class _Create(KernelBase):
    """Load the invented host from a local frame store. No source is read."""

    ref = "test.combined_clone.host@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["population"])
        )


class _Counter:
    """Counts kernel invocations so a warm replay can be proven, not assumed."""

    def __init__(self) -> None:
        self.expand = 0
        self.claim = 0


def _counting_registry(counter: _Counter) -> KernelRegistry:
    expand_base = clone_stage.USCombinedSurveyCloneExpandKernel
    claim_base = clone_stage.USCombinedSurveyCloneClaimKernel

    class _Expand(expand_base):  # type: ignore[misc, valid-type]
        def run(self, context):
            counter.expand += 1
            return super().run(context)

    class _Claim(claim_base):  # type: ignore[misc, valid-type]
        def run(self, context):
            counter.claim += 1
            return super().run(context)

    registry = KernelRegistry()
    registry.register(_Create())
    registry.register(_Expand())
    registry.register(_Claim())
    return registry


def _store_key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _nodes(pool: Frame, *, source_channels=SOURCE_CHANNELS, mutate=None):
    columns = frame_column_declarations(pool)
    produced = us_combined_survey_clone_nodes(
        columns, base="source", source_channels=source_channels
    )
    if mutate is not None:
        produced = mutate(produced)
    return (
        Node(
            "source",
            _Create.ref,
            sources=("population",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
        ),
        *produced,
    )


def _run(
    tmp_path: Path,
    pool: Frame,
    *,
    resume: str = "forbid",
    source_channels=SOURCE_CHANNELS,
    mutate=None,
    counter: _Counter | None = None,
    label: str = "host",
    store: ContentStore | None = None,
):
    nodes = _nodes(pool, source_channels=source_channels, mutate=mutate)
    codecs = SourceCodecRegistry()
    codecs.register("frame-store", load_frame_store)
    kernels = _counting_registry(counter or _Counter())
    store = store or ContentStore(tmp_path / "store", codecs=codecs)
    population = ContentStore(tmp_path / "inputs").put_frame(_store_key(label), pool)
    compiled = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("population", "frame-store"),),
            nodes=nodes,
        )
    )
    manifest = run_graph(
        compiled,
        sources={"population": population},
        store=store,
        kernels=kernels,
        resume=resume,
    )
    return manifest, store, codecs


def _expanded(manifest) -> Frame:
    return manifest.population(COMBINED_CLONE_NODE)


def _direct(pool: Frame) -> Frame:
    return clone_us_frame_for_puf_support(
        pool, clone_attachment_fraction=1.0, clone_attachment_seed=0
    )


def _role_key(frame: Frame, entity: str) -> pd.DataFrame:
    """The declared identity of every row: source channel, source id, role."""

    table = frame.table(entity)
    return pd.DataFrame(
        {
            "channel": table[support_channel_column(entity)].astype(str).to_numpy(),
            "source_id": table[support_source_id_column(entity)].to_numpy(),
            "role": table[support_clone_index_column(entity)].to_numpy(),
        }
    )


def _keyed(frame: Frame, entity: str) -> pd.DataFrame:
    """One entity's rows indexed by the declared key, ids dropped from values.

    Raw ids are excluded from the compared payload and carried in a separate
    column, so a comparison states plainly whether the two paths agree on the
    contractual identity and, separately, on the integers.
    """

    table = frame.table(entity).reset_index(drop=True)
    key = _role_key(frame, entity)
    keyed = pd.concat([key, table], axis=1)
    index = pd.MultiIndex.from_frame(key)
    keyed.index = index
    assert index.is_unique, f"{entity}: declared key is not a bijection"
    return keyed.drop(columns=list(key.columns)).sort_index()


# --------------------------------------------------------------------------
# Declaration surface
# --------------------------------------------------------------------------


def test_registration_binds_both_kernels_of_the_stage():
    registry = KernelRegistry()
    register_us_combined_survey_clone_kernels(registry)
    assert set(registry.refs()) == {
        clone_stage.USCombinedSurveyCloneExpandKernel.ref,
        clone_stage.USCombinedSurveyCloneClaimKernel.ref,
    }
    expand = registry.get(clone_stage.USCombinedSurveyCloneExpandKernel.ref)
    assert expand.capabilities.structural is StructuralDelta.EXPAND
    assert expand.capabilities.determinism is Determinism.DETERMINISTIC
    assert expand.capabilities.numeric is Numeric.BITWISE
    assert expand.capabilities.dependencies == ("numpy", "pandas")


def test_declared_inputs_are_exactly_the_four_provenance_columns():
    pool = _pool()
    expand, claim = us_combined_survey_clone_nodes(
        frame_column_declarations(pool), base="source", source_channels=SOURCE_CHANNELS
    )
    assert {slice_.entity for slice_ in expand.inputs} == set(US_SCHEMA.entities)
    for slice_ in expand.inputs:
        assert slice_.columns == combined_clone_provenance_columns(slice_.entity)
        assert len(slice_.columns) == 4
    assert expand.structural is StructuralDelta.EXPAND
    assert expand.entrants is False
    assert expand.mass == "conserve"
    assert expand.base == "source"
    assert claim.population == expand.id
    assert all(output.rewrite for output in claim.outputs)
    assert {(output.entity, output.column) for output in claim.outputs} == {
        (entity, support_clone_index_column(entity)) for entity in US_SCHEMA.entities
    }


def test_the_stage_reads_no_amount_it_is_not_declared_to_read():
    """The kernel's visible surface is provenance and structure, never money."""

    pool = _pool()
    expand, _ = us_combined_survey_clone_nodes(
        frame_column_declarations(pool), base="source", source_channels=SOURCE_CHANNELS
    )
    declared = {column for slice_ in expand.inputs for column in slice_.columns}
    for amount in ("asec_wage", "acs_income", "age", "state_fips"):
        assert amount not in declared


# --------------------------------------------------------------------------
# The host itself is what the scope says it is
# --------------------------------------------------------------------------


def test_invented_host_is_a_combined_native_importance_pool():
    pool = _pool()
    assert pool.weights_for("household").kind is WeightKind.IMPORTANCE
    assert pool.weighted_entities == ("household",)
    for entity in US_SCHEMA.entities:
        table = pool.table(entity)
        assert set(table[support_clone_index_column(entity)]) == {0}
        assert set(table[support_channel_column(entity)].astype(str)) == set(
            SOURCE_CHANNELS
        )
    # Both arms carry a colliding raw record id, so assembly had to remap one
    # of them: the spine-source id and the assembly-unique source id differ.
    person = pool.table("person")
    remapped = (
        person[spine_source_id_column("person")].to_numpy()
        != person[support_source_id_column("person")].to_numpy()
    )
    assert remapped.any(), "fixture must exercise a remapped raw record id"
    # A zero weight and unequal weights are both present.
    values = pool.weights_for("household").values
    assert (values == 0.0).any() and len(set(values.tolist())) > 2
    # Multiple tax units inside one household.
    counts = person.groupby("person_household_id")["person_tax_unit_id"].nunique()
    assert counts.max() >= 2


# --------------------------------------------------------------------------
# Graph vs direct operator
# --------------------------------------------------------------------------


def test_graph_result_equals_direct_operator_by_declared_key(tmp_path):
    manifest, _, _ = _run(tmp_path, _pool())
    graphed, direct = _expanded(manifest), _direct(_pool())
    for entity in US_SCHEMA.entities:
        left, right = _keyed(graphed, entity), _keyed(direct, entity)
        assert list(left.index) == list(right.index), entity
        assert list(left.columns) == list(right.columns), entity
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_graph_and_direct_agree_on_raw_ids_too(tmp_path):
    """The bijection is the contract; the integers agreeing is a fact we state."""

    manifest, _, _ = _run(tmp_path, _pool())
    graphed, direct = _expanded(manifest), _direct(_pool())
    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        graph_map = dict(
            zip(
                pd.MultiIndex.from_frame(_role_key(graphed, entity)),
                graphed.table(entity)[id_column],
                strict=True,
            )
        )
        direct_map = dict(
            zip(
                pd.MultiIndex.from_frame(_role_key(direct, entity)),
                direct.table(entity)[id_column],
                strict=True,
            )
        )
        assert graph_map == direct_map, entity


def test_graph_weights_and_strata_equal_the_direct_operator(tmp_path):
    manifest, _, _ = _run(tmp_path, _pool())
    graphed, direct = _expanded(manifest), _direct(_pool())
    assert graphed.weighted_entities == direct.weighted_entities == ("household",)
    left = graphed.weights_for("household")
    right = direct.weights_for("household")
    assert left.kind is right.kind is WeightKind.IMPORTANCE
    assert np.array_equal(left.values, right.values)
    assert list(graphed.strata) == list(direct.strata)
    assert graphed.strata.dtype == direct.strata.dtype


# --------------------------------------------------------------------------
# What the stage preserves
# --------------------------------------------------------------------------


def test_native_rows_are_untouched_and_the_detail_role_is_the_only_addition(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    for entity in US_SCHEMA.entities:
        base = pool.table(entity).reset_index(drop=True)
        table = graphed.table(entity).reset_index(drop=True)
        assert list(table.columns) == list(base.columns), entity
        native = table[table[support_clone_index_column(entity)].eq(0)].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(native, base, check_exact=True)
        roles = set(table[support_clone_index_column(entity)])
        assert roles == {0, PUF_TAX_DETAIL_CLONE_INDEX}


def test_both_source_channels_survive_in_both_roles(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    for entity in US_SCHEMA.entities:
        base = pool.table(entity)
        table = graphed.table(entity)
        channel = support_channel_column(entity)
        role = support_clone_index_column(entity)
        for name in SOURCE_CHANNELS:
            incoming = int(base[channel].astype(str).eq(name).sum())
            assert incoming > 0, (entity, name)
            for value in (0, PUF_TAX_DETAIL_CLONE_INDEX):
                assert (
                    int(
                        (
                            table[channel].astype(str).eq(name) & table[role].eq(value)
                        ).sum()
                    )
                    == incoming
                ), (entity, name, value)
        assert graphed.n(entity) == 2 * pool.n(entity)


def test_every_pair_of_weights_sums_to_its_parent_including_the_zero(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    incoming = pd.Series(
        pool.weights_for("household").values,
        index=pool.table("household")[support_source_id_column("household")].to_numpy(),
    )
    table = graphed.table("household")
    weights = pd.DataFrame(
        {
            "source_id": table[support_source_id_column("household")].to_numpy(),
            "role": table[support_clone_index_column("household")].to_numpy(),
            "weight": graphed.weights_for("household").values,
        }
    )
    pairs = weights.pivot(index="source_id", columns="role", values="weight")
    assert list(pairs.columns) == [0, PUF_TAX_DETAIL_CLONE_INDEX]
    parent = incoming.reindex(pairs.index).to_numpy()
    # Halving a finite double is exact, so the pair sum reproduces its parent
    # bit for bit rather than to a tolerance.
    assert np.array_equal(pairs[0].to_numpy() + pairs[1].to_numpy(), parent)
    assert np.array_equal(pairs[0].to_numpy(), pairs[1].to_numpy())
    assert (pairs[0].to_numpy() == 0.0).any(), "the zero-weight pair must be exercised"
    assert float(graphed.weights_for("household").values.sum()) == float(
        pool.weights_for("household").values.sum()
    )


def test_no_identifier_collides_within_or_across_entities(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        ids = graphed.table(entity)[id_column]
        assert ids.is_unique, entity
        base_ids = set(pool.table(entity)[id_column])
        detail = graphed.table(entity)
        detail_ids = set(
            detail.loc[
                detail[support_clone_index_column(entity)].eq(
                    PUF_TAX_DETAIL_CLONE_INDEX
                ),
                id_column,
            ]
        )
        assert not (detail_ids & base_ids), entity


def test_copied_persons_follow_their_copied_groups(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    person = graphed.table("person")
    detail_person = person[
        person[support_clone_index_column("person")].eq(PUF_TAX_DETAIL_CLONE_INDEX)
    ]
    for group in US_SCHEMA.group_entities:
        table = graphed.table(group)
        role_by_id = pd.Series(
            table[support_clone_index_column(group)].to_numpy(),
            index=table[US_SCHEMA.entity_id_column(group)].to_numpy(),
        )
        source_by_id = pd.Series(
            table[support_source_id_column(group)].to_numpy(),
            index=table[US_SCHEMA.entity_id_column(group)].to_numpy(),
        )
        membership = detail_person[US_SCHEMA.membership_column(group)].to_numpy()
        assert (role_by_id.reindex(membership).to_numpy() == 1).all(), group
        # The copied person joins the copy of the same source group its source
        # person belonged to.
        base_person = pool.table("person")
        base_membership = base_person[US_SCHEMA.membership_column(group)].to_numpy()
        base_source = pd.Series(
            pool.table(group)[support_source_id_column(group)].to_numpy(),
            index=pool.table(group)[US_SCHEMA.entity_id_column(group)].to_numpy(),
        )
        assert np.array_equal(
            source_by_id.reindex(membership).to_numpy(),
            base_source.reindex(base_membership).to_numpy(),
        ), group


def test_stage_creates_no_column_draw_or_tax_benefit_value(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    graphed = _expanded(manifest)
    for entity in US_SCHEMA.entities:
        assert set(graphed.table(entity).columns) == set(pool.table(entity).columns)
    # The clone node declares no source at all, so no donor artifact can reach
    # it, and the donor sampling weight is not a column of the host.
    expand = next(node for node in _nodes(pool) if node.id == COMBINED_CLONE_NODE)
    assert expand.sources == ()
    assert expand.artifact_inputs == () and expand.artifact_outputs == ()
    every_column = {
        column
        for entity in graphed.entities
        for column in graphed.table(entity).columns
    }
    assert not {"S006", "s006"} & every_column


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


def test_receipt_publishes_counts_and_digests_only(tmp_path):
    pool = _pool()
    manifest, _, _ = _run(tmp_path, pool)
    receipt = manifest.nodes[COMBINED_CLONE_NODE].receipt
    assert receipt["profile"] == "development"
    assert receipt["release_eligible"] is False
    assert receipt["clone_attachment"]["fraction"] == 1.0
    assert receipt["clone_attachment"]["seed"] == 0
    assert (
        receipt["clone_attachment"]["authority_form"]
        == "full_clone_identity_no_manifest"
    )
    assert list(receipt["source_channels"]) == list(SOURCE_CHANNELS)
    assert list(receipt["clone_roles"]) == list(COMBINED_CLONE_ROLES)
    assert dict(receipt["donor_inputs"]) == {
        "puf_donor_read": False,
        "donor_sampling_weight_in_host_weights": False,
        "draws_or_tax_benefit_values_written": False,
        "prior_wage_stage": False,
    }
    weights = receipt["household_weights"]
    assert weights["kind"] == "importance" and weights["split_denominator"] == 2
    assert weights["base_total"] == weights["expanded_total"]
    assert dict(weights["base_by_source_channel"]) == dict(
        weights["expanded_by_source_channel"]
    )
    for entity in US_SCHEMA.entities:
        facts = receipt["entities"][entity]
        assert facts["base_rows"] == pool.n(entity)
        assert facts["expanded_rows"] == 2 * pool.n(entity)
        assert facts["native_rows"] == facts["detail_rows"] == pool.n(entity)
        assert len(facts["lineage_sha256"]) == 64
        assert len(facts["native_source_ids_sha256"]) == 64
        counts = receipt["provenance_counts"]["expanded"][entity]
        assert counts["rows"] == 2 * pool.n(entity)
        assert dict(counts["by_clone_index"]) == {
            "0": pool.n(entity),
            "1": pool.n(entity),
        }
    # No row-level identity is published. The stage's own receipt carries only
    # declared labels, counts, mass totals and 64-hex digests, and no sequence
    # long enough to be a row, id or amount vector. The executor's own "mass"
    # ledger is its record, not this stage's.
    stage_keys = {
        "phase",
        "profile",
        "release_eligible",
        "operator",
        "source_channels",
        "clone_roles",
        "clone_attachment",
        "entities",
        "provenance_counts",
        "household_weights",
        "donor_inputs",
    }
    # The executor authors the rest of the node receipt itself, including the
    # per-entity ``expand`` lineage it needs to restore a cached population.
    # That row-level record is the executor's machinery, not this stage's
    # published evidence; the stage's own keys are exactly these.
    assert stage_keys <= set(receipt)
    assert set(receipt) - stage_keys == {
        "capabilities",
        "expand",
        "expand_declared",
        "expand_writes",
        "mass",
    }
    published = {key: receipt[key] for key in stage_keys}
    allowed_labels = {
        "us_combined_survey_puf_support_clone",
        "development",
        "importance",
        "full_clone_identity_no_manifest",
        "full native/detail identity",
        "microcosm.build.us_runtime.puf_support",
        "clone_us_frame_for_puf_support",
        "validate_puf_clone_attachment",
        *SOURCE_CHANNELS,
        *COMBINED_CLONE_ROLES,
    }
    for leaf in _leaves(published):
        if isinstance(leaf, str):
            assert leaf in allowed_labels or _is_digest(leaf), leaf
        else:
            assert isinstance(leaf, (bool, int, float)) or leaf is None, leaf
    for sequence in _sequences(published):
        assert len(sequence) <= len(SOURCE_CHANNELS) + len(COMBINED_CLONE_ROLES)


def _leaves(value):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _sequences(value):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _sequences(item)
    elif isinstance(value, (list, tuple)):
        yield value
        for item in value:
            yield from _sequences(item)


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


# --------------------------------------------------------------------------
# Cache: cold and required warm replay
# --------------------------------------------------------------------------


def test_required_warm_replay_recomputes_nothing_and_agrees(tmp_path):
    pool = _pool()
    codecs = SourceCodecRegistry()
    codecs.register("frame-store", load_frame_store)
    store = ContentStore(tmp_path / "store", codecs=codecs)
    cold_counter = _Counter()
    cold, _, _ = _run(tmp_path, pool, resume="auto", counter=cold_counter, store=store)
    assert (cold_counter.expand, cold_counter.claim) == (1, 1)
    assert manifest_hits(cold) == {
        COMBINED_CLONE_NODE: False,
        COMBINED_CLONE_CLAIM_NODE: False,
    }

    warm_counter = _Counter()
    warm, _, _ = _run(
        tmp_path, pool, resume="require", counter=warm_counter, store=store
    )
    assert (warm_counter.expand, warm_counter.claim) == (0, 0)
    assert manifest_hits(warm) == {
        COMBINED_CLONE_NODE: True,
        COMBINED_CLONE_CLAIM_NODE: True,
    }
    assert warm.nodes[COMBINED_CLONE_NODE].key == cold.nodes[COMBINED_CLONE_NODE].key

    cold_frame, warm_frame = _expanded(cold), _expanded(warm)
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(
            cold_frame.table(entity).reset_index(drop=True),
            warm_frame.table(entity).reset_index(drop=True),
            check_exact=True,
        )
        assert set(
            warm_frame.table(entity)[support_channel_column(entity)].astype(str)
        ) == set(SOURCE_CHANNELS)
    assert np.array_equal(
        cold_frame.weights_for("household").values,
        warm_frame.weights_for("household").values,
    )
    assert (
        warm_frame.weights_for("household").kind
        is cold_frame.weights_for("household").kind
    )
    assert list(warm_frame.strata) == list(cold_frame.strata)


def manifest_hits(manifest) -> dict[str, bool]:
    return {
        node_id: manifest.nodes[node_id].hit
        for node_id in (COMBINED_CLONE_NODE, COMBINED_CLONE_CLAIM_NODE)
    }


# --------------------------------------------------------------------------
# What the node key binds
# --------------------------------------------------------------------------


def _key_for(tmp_path: Path, pool: Frame, subdirectory: str) -> str:
    """Run one pool in its own directory under one fixed store key.

    The store key is deliberately constant: the frame-store source identity
    hashes the object directory's bytes, including the key its metadata
    records, so varying the key would move every node key for reasons that
    have nothing to do with the population.
    """

    manifest, _, _ = _run(tmp_path / subdirectory, pool, label="host")
    return manifest.nodes[COMBINED_CLONE_NODE].key


def _with_person_change(column: str, position: int, value) -> Frame:
    arm = _asec_arm()
    person = arm.table("person").copy()
    person.loc[position, column] = value
    tables = {entity: arm.table(entity).copy() for entity in arm.entities}
    tables["person"] = person
    changed = Frame(
        tables,
        arm.schema,
        {"household": arm.weights_for("household")},
        arm.strata,
    )
    return _pool(asec=changed)


def test_node_key_binds_amounts_ids_order_provenance_and_weights(tmp_path):
    baseline = _key_for(tmp_path, _pool(), "baseline")
    assert _key_for(tmp_path, _pool(), "baseline-again") == baseline

    # An amount the kernel never reads still enters the key, because the node
    # binds its whole base population version, not merely its declared slices.
    assert _key_for(tmp_path, _with_person_change("asec_wage", 0, 99.0), "amount") != (
        baseline
    )

    # A structural id.
    renumbered = _asec_arm()
    person = renumbered.table("person").copy()
    person["person_id"] = person["person_id"] + 1
    tables = {e: renumbered.table(e).copy() for e in renumbered.entities}
    tables["person"] = person
    assert (
        _key_for(
            tmp_path,
            _pool(
                asec=Frame(
                    tables,
                    renumbered.schema,
                    {"household": renumbered.weights_for("household")},
                    renumbered.strata,
                )
            ),
            "ids",
        )
        != baseline
    )

    # Row order.
    reordered = _asec_arm()
    order = [1, 0, 2, 3, 4, 5, 6]
    person = reordered.table("person").iloc[order].reset_index(drop=True)
    tables = {e: reordered.table(e).copy() for e in reordered.entities}
    tables["person"] = person
    assert (
        _key_for(
            tmp_path,
            _pool(
                asec=Frame(
                    tables,
                    reordered.schema,
                    {"household": reordered.weights_for("household")},
                    reordered.strata.iloc[order].reset_index(drop=True),
                )
            ),
            "order",
        )
        != baseline
    )

    # Incoming weights.
    reweighted = _asec_arm()
    assert (
        _key_for(
            tmp_path,
            _pool(
                asec=Frame(
                    {e: reweighted.table(e).copy() for e in reweighted.entities},
                    reweighted.schema,
                    {
                        "household": Weights(
                            np.array([1201.0, 0.0, 730.0]), WeightKind.DESIGN
                        )
                    },
                    reweighted.strata,
                )
            ),
            "weights",
        )
        != baseline
    )

    # Provenance: a different declared source-channel roster is a different node.
    pool = _pool()
    with pytest.raises(CombinedCloneError):
        us_combined_survey_clone_nodes(
            frame_column_declarations(pool),
            base="source",
            source_channels=("asec",),
        )


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_refuses_a_preoperated_host(tmp_path):
    already = _direct(_pool())
    with pytest.raises(NodeRejectedError, match="exactly once"):
        _run(tmp_path, already, label="preoperated")


def test_refuses_a_partial_provenance_inventory():
    pool = _pool()
    columns = tuple(
        owned
        for owned in frame_column_declarations(pool)
        if owned.column != support_channel_column("family")
    )
    with pytest.raises(CombinedCloneError, match="MISSING_PROVENANCE_COLUMN"):
        us_combined_survey_clone_nodes(
            columns, base="source", source_channels=SOURCE_CHANNELS
        )


@pytest.mark.parametrize(
    "channels",
    [("asec",), ("asec", "acs"), ("acs", "acs"), ("acs", ""), ()],
)
def test_refuses_a_malformed_channel_declaration(channels):
    pool = _pool()
    with pytest.raises(CombinedCloneError):
        us_combined_survey_clone_nodes(
            frame_column_declarations(pool),
            base="source",
            source_channels=channels,
        )


def test_refuses_a_declaration_that_does_not_name_a_live_channel(tmp_path):
    with pytest.raises(NodeRejectedError, match="SOURCE_CHANNELS_LIVE"):
        _run(tmp_path, _pool(), source_channels=("acs", "asec", "sipp"))


def test_refuses_a_single_arm_host(tmp_path):
    """A host that lost one survey arm is not the combined pool this stage takes."""

    pool = _pool()
    keep = pool.table("person")[support_channel_column("person")].astype(str).eq("asec")
    single = pool.select(keep.to_numpy())
    with pytest.raises(NodeRejectedError, match="SOURCE_CHANNELS_LIVE"):
        _run(tmp_path, single, label="single-arm")


def test_refuses_a_design_weighted_host(tmp_path):
    pool = _pool()
    design = Frame(
        {entity: pool.table(entity).copy() for entity in pool.entities},
        pool.schema,
        {"household": Weights(pool.weights_for("household").values, WeightKind.DESIGN)},
        pool.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )
    with pytest.raises(NodeRejectedError, match="INCOMING_WEIGHT_KIND"):
        _run(tmp_path, design, label="design")


def test_refuses_an_explicitly_weighted_second_entity(tmp_path):
    pool = _pool()
    person_weights = pd.Series(
        pool.weights_for("household").values,
        index=pool.table("household")["household_id"].to_numpy(),
    ).reindex(pool.table("person")["person_household_id"].to_numpy())
    weighted = Frame(
        {entity: pool.table(entity).copy() for entity in pool.entities},
        pool.schema,
        {
            "household": pool.weights_for("household"),
            "person": Weights(
                person_weights.to_numpy(dtype=float) + 1.0, WeightKind.IMPORTANCE
            ),
        },
        pool.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )
    with pytest.raises(NodeRejectedError, match="NON_INHERITED_WEIGHTS"):
        _run(tmp_path, weighted, label="two-weights")


@pytest.mark.parametrize(
    ("param", "value", "reason"),
    [
        ("clone_attachment_fraction", 0.5, "NODE_ATTACHMENT_FRACTION"),
        ("clone_attachment_seed", 7, "NODE_ATTACHMENT_SEED"),
        ("clone_roles", ("asec", "other"), "NODE_ROLES"),
        ("profile", "release", "NODE_PROFILE"),
        ("release_eligible", True, "NODE_RELEASE_ELIGIBLE"),
        ("weight_split_denominator", 3, "NODE_WEIGHT_SPLIT"),
        ("phase", "something_else", "NODE_PHASE"),
    ],
)
def test_refuses_tampered_parameters(tmp_path, param, value, reason):
    def mutate(nodes):
        expand, claim = nodes
        return (
            Node(
                id=expand.id,
                kernel=expand.kernel,
                inputs=expand.inputs,
                params={**dict(expand.params), param: value},
                structural=expand.structural,
                base=expand.base,
                mass=expand.mass,
            ),
            claim,
        )

    with pytest.raises(NodeRejectedError, match=reason):
        _run(tmp_path, _pool(), mutate=mutate, label=f"param-{param}")


def test_refuses_an_extra_or_missing_parameter(tmp_path):
    def mutate(nodes):
        expand, claim = nodes
        params = {**dict(expand.params), "unreviewed": 1}
        return (
            Node(
                id=expand.id,
                kernel=expand.kernel,
                inputs=expand.inputs,
                params=params,
                structural=expand.structural,
                base=expand.base,
                mass=expand.mass,
            ),
            claim,
        )

    with pytest.raises(NodeRejectedError, match="NODE_PARAMS"):
        _run(tmp_path, _pool(), mutate=mutate, label="extra-param")


def test_refuses_a_relaxed_mass_policy(tmp_path):
    """Conservation is the stage's contract, not a default it may drop."""

    def mutate(nodes):
        expand, claim = nodes
        return (
            Node(
                id=expand.id,
                kernel=expand.kernel,
                inputs=expand.inputs,
                params=dict(expand.params),
                structural=expand.structural,
                base=expand.base,
                mass="free",
            ),
            claim,
        )

    with pytest.raises(NodeRejectedError, match="NODE_MASS"):
        _run(tmp_path, _pool(), mutate=mutate, label="mass-free")


def test_refuses_a_node_that_admits_entrants(tmp_path):
    """This stage copies households; it never invents one."""

    def mutate(nodes):
        expand, claim = nodes
        return (
            Node(
                id=expand.id,
                kernel=expand.kernel,
                inputs=expand.inputs,
                params=dict(expand.params),
                structural=expand.structural,
                base=expand.base,
                mass="free",
                entrants=True,
            ),
            claim,
        )

    with pytest.raises(NodeRejectedError, match="NODE_ENTRANTS"):
        _run(tmp_path, _pool(), mutate=mutate, label="entrants")


def test_refuses_an_undeclared_input_slice(tmp_path):
    def mutate(nodes):
        expand, claim = nodes
        return (
            Node(
                id=expand.id,
                kernel=expand.kernel,
                inputs=expand.inputs[:-1],
                params=dict(expand.params),
                structural=expand.structural,
                base=expand.base,
                mass=expand.mass,
            ),
            claim,
        )

    with pytest.raises(NodeRejectedError, match="NODE_INPUTS"):
        _run(tmp_path, _pool(), mutate=mutate, label="inputs")


def _mutated_clone(mutate):
    """Wrap the real operator so a specific post-hoc corruption can be tested."""

    real = clone_stage.clone_us_frame_for_puf_support

    def broken(frame, **kwargs):
        result = real(frame, **kwargs)
        tables = {entity: result.table(entity).copy() for entity in result.entities}
        strata = mutate(tables, result)
        return Frame(
            tables,
            result.schema,
            {e: result.weights_for(e) for e in result.weighted_entities},
            result.strata if strata is None else strata,
            mass_log=result.mass_log,
            metadata=result.metadata,
        )

    return broken


def test_refuses_a_clone_output_whose_household_pairing_is_wrong(tmp_path, monkeypatch):
    """The reused validator, not a private re-derivation, is what refuses."""

    def mutate(tables, result):
        column = support_source_id_column("household")
        values = tables["household"][column].to_numpy(copy=True)
        half = len(values) // 2
        values[half:] = values[half:][::-1]
        tables["household"][column] = values
        return None

    monkeypatch.setattr(
        clone_stage, "clone_us_frame_for_puf_support", _mutated_clone(mutate)
    )
    with pytest.raises(NodeRejectedError, match="full-clone identity failed"):
        _run(tmp_path, _pool(), label="mispaired-household")


def test_refuses_a_clone_output_whose_person_layout_disagrees_with_its_lineage(
    tmp_path, monkeypatch
):
    """The bijection and the physical layout must describe the same pairing.

    The household validator cannot see this: it only inspects households. The
    stage refuses rather than quietly preferring one derivation over the other,
    because a layout change is a review question, not a run-time choice.
    """

    def mutate(tables, result):
        person = tables["person"]
        half = len(person) // 2
        order = list(range(half)) + list(range(len(person) - 1, half - 1, -1))
        tables["person"] = person.iloc[order].reset_index(drop=True)
        return result.strata.iloc[order].reset_index(drop=True)

    monkeypatch.setattr(
        clone_stage, "clone_us_frame_for_puf_support", _mutated_clone(mutate)
    )
    with pytest.raises(NodeRejectedError, match="LINEAGE_POSITION_DISAGREES:person"):
        _run(tmp_path, _pool(), label="mispaired-person")


def test_refuses_a_clone_output_with_an_unauthorized_person_role(tmp_path, monkeypatch):
    def mutate(tables, result):
        column = support_clone_index_column("person")
        values = tables["person"][column].to_numpy(copy=True)
        values[-1] = 2
        tables["person"][column] = values
        return None

    monkeypatch.setattr(
        clone_stage, "clone_us_frame_for_puf_support", _mutated_clone(mutate)
    )
    with pytest.raises(NodeRejectedError, match="CLONE_ROLE_UNEXPECTED:person"):
        _run(tmp_path, _pool(), label="extra-person-role")


def test_refuses_a_clone_output_with_an_unauthorized_household_role(
    tmp_path, monkeypatch
):
    def mutate(tables, result):
        column = support_clone_index_column("household")
        values = tables["household"][column].to_numpy(copy=True)
        values[-1] = 2
        tables["household"][column] = values
        return None

    monkeypatch.setattr(
        clone_stage, "clone_us_frame_for_puf_support", _mutated_clone(mutate)
    )
    with pytest.raises(NodeRejectedError, match="full-clone identity failed"):
        _run(tmp_path, _pool(), label="extra-household-role")
