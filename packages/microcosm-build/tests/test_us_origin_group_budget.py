"""Full materialized origin budgets and successors, invented sources only."""

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_native_household_origin as origins
from microcosm.build.us_runtime import native_household_origin as native
from microcosm.build.us_runtime import origin_group_budget as budget_module
from microcosm.build.us_runtime.graph_combined_clone import (
    USCombinedSurveyCloneClaimKernel,
    USCombinedSurveyCloneExpandKernel,
    us_combined_survey_clone_nodes,
)
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.frame import Frame, WeightKind, Weights
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
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.population import Population


def _peer(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Importance(KernelBase):
    ref = "test.origin_budget.importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.REWEIGHT,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        values = (
            np.asarray(context.params["incoming_weights"], dtype=np.float64)
            if "incoming_weights" in context.params
            else context.weights["household"].values * 0.5
        )
        return KernelResult(weights=Weights(values, WeightKind.IMPORTANCE))


def invented_budget_graph(
    tmp_path,
    monkeypatch,
    *,
    positive=False,
    omit_column=False,
    duplicate_root=False,
    positive_incoming_zero_design=False,
    return_execution_args=False,
):
    """Real source producer, CREATE/IMPORTANCE/EXPAND and typed origin graph."""
    peer = _peer("test_us_graph_native_household_origin")
    root = peer._fixture_graph(tmp_path, monkeypatch)
    meta = peer._restore(root)
    source = load_frame_store(Path(meta["population"]))
    if duplicate_root:
        table = source.table("household")
        acs_rows = table.household_support_channel == "acs"
        table.loc[acs_rows, "SERIALNO"] = table.loc[acs_rows, "SERIALNO"].iloc[0]
    weights = source.weights_for("household").values.copy()
    if positive:
        weights[weights == 0] = 16.0
    source.person["invented_unknown_boolean"] = pd.Series(
        [pd.NA if i % 3 == 0 else i % 2 == 0 for i in range(source.n("person"))],
        dtype="boolean",
    )
    design_weights = weights * 2
    if positive_incoming_zero_design:
        design_weights[0] = 0.0
    source = Frame(
        {e: source.table(e) for e in source.entities},
        source.schema,
        {"household": Weights(design_weights, WeightKind.DESIGN)},
        source.strata,
    )
    source_path = ContentStore(root / "design-input").put_frame("d" * 64, source)
    columns = frame_column_declarations(source)
    binding_columns = tuple(
        c
        for c in columns
        if not (omit_column and c.column == "invented_unknown_boolean")
    )
    create = _peer("test_us_graph_combined_clone")._Create
    clones = us_combined_survey_clone_nodes(
        columns, base="importance", source_channels=("acs", "asec")
    )
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"), *origins.SOURCES),
        nodes=(
            Node(
                "source",
                create.ref,
                sources=("population",),
                structural=StructuralDelta.CREATE,
                outputs=columns,
            ),
            Node(
                "importance",
                _Importance.ref,
                base="source",
                inputs=(Slice("household", ("state_fips",)),),
                structural=StructuralDelta.REWEIGHT,
                weights=WeightTransition("household", "importance", mass="free"),
                mass="free",
                params={"incoming_weights": tuple(weights.tolist())}
                if positive_incoming_zero_design
                else {},
            ),
            *origins.native_origin_source_nodes(population="importance"),
            origins.native_origin_binding_node(
                binding_columns, population="importance", node_id="native_binding"
            ),
            *clones,
            origins.native_origin_binding_node(
                binding_columns,
                population=clones[0].id,
                node_id="clone_binding",
                parent_binding_node="native_binding",
            ),
        ),
    )
    registry = KernelRegistry()
    for kernel in (
        create(),
        _Importance(),
        origins.NativeACSOriginKernel(),
        origins.NativeASECOriginKernel(),
        origins.PopulationOriginBindingKernel(),
        USCombinedSurveyCloneExpandKernel(),
        USCombinedSurveyCloneClaimKernel(),
    ):
        registry.register(kernel)
    codecs = us_source_codecs()
    codecs.register("frame-store", load_frame_store)
    store = ContentStore(root / "budget-store", codecs=codecs)
    sources = {"population": source_path, origins.ACS_SOURCE: Path(meta["acs_source"])}
    sources.update(
        {
            name: Path(meta["asec_paths"][str(year)])
            for year, name in zip((2022, 2023, 2024), origins.ASEC_SOURCES, strict=True)
        }
    )
    observed = {}
    manifest = run_graph(
        compile_graph(graph),
        store=store,
        sources=sources,
        kernels=registry,
        resume="forbid",
        _population_observer=lambda node, population: observed.__setitem__(
            node, population
        ),
    )
    kwargs = dict(
        native_population=observed["native_binding"],
        expanded_population=observed["clone_binding"],
        native_binding_node="native_binding",
        expanded_binding_node="clone_binding",
        design_multiplier=2.0,
        incoming_multiplier=3.0,
    )
    if return_execution_args:
        return (
            manifest,
            store,
            kwargs,
            dict(prefix_graph=graph, sources=sources, kernels=registry),
        )
    return manifest, store, kwargs


@pytest.fixture(scope="module")
def materialized(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield invented_budget_graph(
            tmp_path_factory.mktemp("materialized-budget"), monkeypatch
        )


def _budget(materialized):
    manifest, store, kwargs = materialized
    return budget_module.freeze_origin_group_budget(manifest, store, **kwargs)


def _changed(
    population, values=None, *, transform=None, design=None, independent=False
):
    frame = population.frame
    tables = {e: frame.table(e).copy() for e in frame.entities}
    if transform:
        transform(tables)
    weights = {
        "household": Weights(
            frame.weights_for("household").values if values is None else values,
            WeightKind.CALIBRATED,
        )
    }
    if independent:
        weights["person"] = frame.resolve_weights("person")
    replacement = Frame(
        tables, frame.schema, weights, frame.strata, metadata=frame.metadata
    )
    return Population.from_frame(
        replacement,
        "invented_successor",
        owners=population.owners,
        design_weights=population.design_weights if design is None else design,
    )


def test_materialized_budget_freezes_distinct_design_and_incoming(materialized):
    budget = _budget(materialized)
    assert budget.document["group_count"] == 5
    assert budget.document["zero_weight_households_with_lineage"] == 2
    assert set(budget.document["source_edges"]) == {"acs", "asec"}
    for group in budget.document["origins"]:
        design = float.fromhex(group["prepared_design_hex"])
        incoming = float.fromhex(group["frozen_incoming_hex"])
        assert design == 2 * incoming
        assert float.fromhex(group["absolute_bound_hex"]) == min(
            2 * design, 3 * incoming
        )
        assert {r["clone_index"] for r in group["members"]} == {0, 1}
    decoded = budget_module.FrozenOriginBudget.from_payload(budget.payload)
    assert (
        budget_module.verify_origin_group_budget(
            decoded, materialized[0], materialized[1], **materialized[2]
        ).payload
        == budget.payload
    )
    budget.grouped_bounds.check(
        materialized[2]["expanded_population"].frame.weights_for("household").values
    )
    with pytest.raises(ValueError, match="strictly positive"):
        budget.grouped_bounds.check(
            materialized[2]["expanded_population"]
            .frame.weights_for("household")
            .values,
            positive=True,
        )


def test_constructor_is_not_admission_and_copied_anchors_refuse(materialized):
    with pytest.raises(budget_module.OriginBudgetError, match="BUDGET_CONSTRUCTOR"):
        budget_module.FrozenOriginBudget(b"{}")
    manifest, store, kwargs = materialized
    pop = kwargs["expanded_population"]
    changed = replace(
        pop, design_weights={"household": pop.design_weights["household"] * 0.5}
    )
    with pytest.raises(budget_module.OriginBudgetError, match="COPIED_DESIGN_ANCHOR"):
        budget_module.freeze_origin_group_budget(
            manifest, store, **{**kwargs, "expanded_population": changed}
        )
    lost = replace(kwargs["native_population"], design_weights={})
    with pytest.raises(
        budget_module.OriginBudgetError, match="ORIGINAL_DESIGN_REQUIRED"
    ):
        budget_module.freeze_origin_group_budget(
            manifest, store, **{**kwargs, "native_population": lost}
        )


@pytest.mark.parametrize(
    "field",
    [
        "source_payload_sha256",
        "native_binding_sha256",
        "expanded_artifact_key",
        "native_population_content_sha256",
        "origins",
    ],
)
def test_well_shaped_fabricated_budget_is_not_authority(materialized, field):
    budget = _budget(materialized)
    document = budget.document
    if field == "origins":
        document[field][0]["frozen_incoming_hex"] = (1e12).hex()
    elif isinstance(document[field], list):
        document[field][0] = "f" * 64
    else:
        document[field] = "f" * 64
    fabricated = budget_module.FrozenOriginBudget.from_payload(canonical_json(document))
    with pytest.raises(
        budget_module.OriginBudgetError, match="BUDGET_MATERIALIZED_IDENTITY"
    ):
        budget_module.verify_origin_group_budget(
            fabricated, materialized[0], materialized[1], **materialized[2]
        )


def test_group_map_only_mutation_is_reconstructed_not_trusted(materialized):
    budget = _budget(materialized)
    doc = budget.document
    groups = doc["group_indices"]
    i = next(i for i in range(1, len(groups)) if groups[i] != groups[0])
    groups[0], groups[i] = groups[i], groups[0]
    from microcosm.calibrate.group_bounds import GroupedUpperBounds

    changed = GroupedUpperBounds(
        doc["household_ids"],
        groups,
        [float.fromhex(v) for v in doc["absolute_bound_hex"]],
    )
    doc["constraint_digest"] = changed.constraint_digest
    fabricated = budget_module.FrozenOriginBudget.from_payload(canonical_json(doc))
    assert fabricated.digest != budget.digest
    with pytest.raises(
        budget_module.OriginBudgetError, match="BUDGET_MATERIALIZED_IDENTITY"
    ):
        budget_module.verify_origin_group_budget(
            fabricated, materialized[0], materialized[1], **materialized[2]
        )


def test_complete_materialization_refuses_dropped_unknown_column(tmp_path, monkeypatch):
    fixture = invented_budget_graph(tmp_path, monkeypatch, omit_column=True)
    with pytest.raises(budget_module.OriginBudgetError, match="MATERIALIZED_CONTRACT"):
        _budget(fixture)


def test_duplicate_native_root_cannot_mint_two_budgets(tmp_path, monkeypatch):
    fixture = invented_budget_graph(tmp_path, monkeypatch, duplicate_root=True)
    with pytest.raises(
        budget_module.OriginBudgetError, match="ONE_NATIVE_ROOT_PER_ORIGIN"
    ):
        _budget(fixture)


def test_weight_only_successor_preserves_unknowns_multitaxunit_and_all_entities(
    materialized,
):
    budget = _budget(materialized)
    previous = materialized[2]["expanded_population"]
    values = previous.frame.weights_for("household").values.copy()
    groups = budget.grouped_bounds.group_indices
    for group in range(budget.document["group_count"]):
        members = np.flatnonzero(groups == group)
        total = math.fsum(values[members])
        values[members] = total * np.array([0.875, 0.125])
    current = _changed(previous, values)
    binding = budget_module.bind_weight_only_successor(
        budget, previous=previous, current=current
    )
    assert (
        binding.document["new_population_content_sha256"]
        != binding.document["old_population_content_sha256"]
    )
    assert set(binding.document["derived_entity_weight_sha256"]) == set(
        previous.frame.entities
    )
    assert current.frame.weighted_entities == ("household",)
    assert current.frame.person.invented_unknown_boolean.isna().any()
    assert (
        current.frame.person.groupby("person_household_id")
        .person_tax_unit_id.nunique()
        .max()
        == 2
    )
    assert (
        budget_module.verify_weight_only_successor(
            binding, budget, previous=previous, current=current
        ).payload
        == binding.payload
    )
    with pytest.raises(native.NativeOriginError, match="POPULATION_CONTENT"):
        native.verify_population_origin_binding(
            current.frame,
            origins.verify_materialized_population_origins(
                materialized[0],
                materialized[1],
                population_node=previous.version,
                binding_node="clone_binding",
                parent_binding_node="native_binding",
            ),
        )


def test_refits_retain_first_budget_and_cannot_ratchet(materialized):
    budget = _budget(materialized)
    before = materialized[2]["expanded_population"]
    first = _changed(before, before.frame.weights_for("household").values * 2)
    binding = budget_module.bind_weight_only_successor(
        budget, previous=before, current=first
    )
    second = _changed(first, first.frame.weights_for("household").values * 1.25)
    successor = budget_module.bind_weight_only_successor(
        budget, previous=first, current=second, previous_binding=binding
    )
    assert (
        successor.document["budget_sha256"]
        == binding.document["budget_sha256"]
        == budget.digest
    )
    over = _changed(second, second.frame.weights_for("household").values * 1.5)
    with pytest.raises(ValueError, match="exceed"):
        budget_module.bind_weight_only_successor(
            budget, previous=second, current=over, previous_binding=successor
        )
    with pytest.raises(
        budget_module.OriginBudgetError, match="SUCCESSOR_PREVIOUS_POPULATION"
    ):
        budget_module.bind_weight_only_successor(
            budget, previous=second, current=second
        )


@pytest.mark.parametrize(
    "mutation", ["cell", "column", "membership", "source", "anchor", "independent"]
)
def test_successor_rejects_every_nonweight_change(materialized, mutation):
    budget = _budget(materialized)
    previous = materialized[2]["expanded_population"]

    def transform(tables):
        people = tables["person"]
        if mutation == "cell":
            people["invented_unknown_boolean"] = pd.Series(
                [True] * len(people), dtype="boolean"
            )
        elif mutation == "column":
            people.drop(columns="invented_unknown_boolean", inplace=True)
        elif mutation == "membership":
            people["person_marital_unit_id"] = people.person_marital_unit_id.iloc[
                ::-1
            ].to_numpy()
        elif mutation == "source":
            people["source_record_note"] = "changed"

    with pytest.raises((ValueError, budget_module.OriginBudgetError)):
        current = _changed(
            previous,
            transform=transform,
            design={"household": previous.design_weights["household"] * 2}
            if mutation == "anchor"
            else None,
            independent=mutation == "independent",
        )
        budget_module.bind_weight_only_successor(
            budget, previous=previous, current=current
        )


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, 1e308])
def test_invalid_multiplier_or_overflow_refuses(materialized, value):
    manifest, store, kwargs = materialized
    with pytest.raises(budget_module.OriginBudgetError):
        budget_module.freeze_origin_group_budget(
            manifest, store, **{**kwargs, "design_multiplier": value}
        )


def test_diagnostics_report_direct_inequalities_and_separate_counts(materialized):
    budget = _budget(materialized)
    values = (
        materialized[2]["expanded_population"].frame.weights_for("household").values
    )
    result = budget.diagnostics(values)
    assert result["actual_household_rows"] == 10
    assert result["positive_weight_household_rows"] == 8
    assert result["authenticated_original_origins"] == 5
    assert result["positive_household_contribution_origins"] == 4
    assert result["max_origin_design_ratio"] == 0.5
    assert result["max_origin_incoming_ratio"] == 1.0
    assert result["row_kish_ess_computational"] > 0
    assert result["budget_sha256"] == budget.digest


def test_origin_support_neutral_clone_role_labels_and_person_contributions(
    materialized,
):
    from decimal import Decimal

    from microcosm.build.cd_benchmark.protocol import canonical_protocol
    from microcosm.build.cd_benchmark.support import (
        SourceRecord,
        SupportMeasure,
        aggregate_origins,
        denominator_support,
    )

    keys = [r["origin_key"] for r in _budget(materialized).document["origins"][:2]]
    native_records = [SourceRecord(key, 100, 100) for key in keys]
    clones = [SourceRecord(key, 50, 50) for key in keys for _ in ("native", "detail")]
    redistributed = [
        SourceRecord(key, value, value) for key in keys for value in (90, 10)
    ]

    def reading(records, measure):
        return denominator_support(
            canonical_protocol().support,
            records,
            block_id="invented-origin-cap-support",
            geography_id="invented",
            measure_kind=measure,
        )

    results = [
        reading(records, SupportMeasure.OCCUPIED_HOUSING_UNITS)
        for records in (native_records, clones, redistributed)
    ]
    assert all(result.effective_sample_size == Decimal(2) for result in results)
    assert all(result.largest_origin_share == Decimal("0.5") for result in results)
    assert [result.counts.actual_records for result in results] == [2, 4, 4]
    assert all(result.counts.authenticated_original_origins == 2 for result in results)
    assert dict(aggregate_origins(clones).totals) == dict(
        aggregate_origins(reversed(redistributed)).totals
    )
    row_ess = [
        sum(r.weight for r in records) ** 2 / sum(r.weight**2 for r in records)
        for records in (native_records, clones, redistributed)
    ]
    assert row_ess[:2] == [2, 4]
    assert row_ess[2] == pytest.approx(2.4390243902439024)
    # An eligible-person contribution multiplies by the eligible persons per
    # household; substituting the HH weight would give the wrong person mass.
    people = [
        SourceRecord(keys[0], 50, 150),
        SourceRecord(keys[0], 50, 150),
        SourceRecord(keys[1], 50, 50),
        SourceRecord(keys[1], 50, 50),
    ]
    person_result = reading(people, SupportMeasure.PERSONS)
    assert person_result.contribution_total == 400
    assert person_result.effective_sample_size == Decimal("1.6")
    assert person_result.largest_origin_share == Decimal("0.75")
    assert person_result.measure is SupportMeasure.PERSONS


def test_v4_source_number_cohort_and_lexical_identity_are_not_role_ids():
    from microcosm.build.cd_benchmark.origin import (
        AcsHouseholdOrigin,
        AsecHouseholdOrigin,
        SourceMember,
        origin_key,
    )

    member = SourceMember("invented/original/member", "a" * 64)
    acs_leading = origin_key(AcsHouseholdOrigin(member, 2024, "0011"))
    acs_plain = origin_key(AcsHouseholdOrigin(member, 2024, "11"))
    asec_2023 = origin_key(AsecHouseholdOrigin(member, 2023, 11))
    asec_2024 = origin_key(AsecHouseholdOrigin(member, 2024, 11))
    assert len({acs_leading, acs_plain, asec_2023, asec_2024}) == 4
    assert acs_leading == origin_key(AcsHouseholdOrigin(member, 2024, "0011"))


def test_successor_accepts_exact_real_reweight_ownership_and_rejects_mixed(
    materialized,
):
    from microcosm.graph.population import patch

    budget = _budget(materialized)
    previous = materialized[2]["expanded_population"]
    node = Node(
        "invented_complete_reweight",
        "test.origin_budget.reweight@1",
        base=previous.version,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        params={"max_weight_ratio": 2.0, "weight_anchor": "design"},
    )
    current = patch(
        previous,
        node,
        KernelResult(
            weights=Weights(
                previous.frame.weights_for("household").values.copy(),
                WeightKind.CALIBRATED,
            )
        ),
    )
    assert set(current.owners.values()) == {current.version}
    assert current.owners != previous.owners
    binding = budget_module.bind_weight_only_successor(
        budget, previous=previous, current=current
    )
    assert (
        binding.document["old_computational_ownership_sha256"]
        != binding.document["new_computational_ownership_sha256"]
    )
    mixed = dict(current.owners)
    mixed[next(iter(mixed))] = "fabricated_mixed_owner"
    with pytest.raises(
        budget_module.OriginBudgetError, match="SUCCESSOR_COMPUTATIONAL_OWNERSHIP"
    ):
        budget_module.bind_weight_only_successor(
            budget, previous=previous, current=replace(current, owners=mixed)
        )
