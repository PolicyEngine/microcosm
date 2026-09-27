"""Invented structural proofs through the real executor; no money placement."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_native_puf_tail_expand as tail
from microcosm.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.survey_population_domains import Domain, Source
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
    NodeRejectedError,
    Numeric,
    Owned,
    SourceRef,
    StoreCorruptError,
    StoreMissError,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from microcosm.graph.population import token_for_dtype


def invented_frame(case="normal"):
    """Literal origin/role fixture; no old pool or survey source is consulted."""
    native_hh = [10, 10, 20, 30, 40]
    native_people = [1, 2, 3, 4, 5]
    person = pd.DataFrame(
        {
            "person_id": native_people + [i + 100 for i in native_people],
            **{
                US_SCHEMA.membership_column(g): native_hh + [i + 100 for i in native_hh]
                for g in US_SCHEMA.group_entities
                if g != "marital_unit"
            },
            "person_marital_unit_id": [i + 1000 for i in native_people]
            + [i + 1100 for i in native_people],
            "payload": np.array([-0.0, 7.5, -19.0, 0.0, 3.0] * 2),
            "nullable": pd.Series([1, pd.NA, 2, 0, 8] * 2, dtype="Int64"),
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        ids = list(dict.fromkeys(person[US_SCHEMA.membership_column(group)]))
        tables[group] = pd.DataFrame({US_SCHEMA.entity_id_column(group): ids})
        tables[group][f"{group}_retained_payload"] = pd.Series(
            [f"{group}-{i % (len(ids) // 2)}" for i in range(len(ids))], dtype="string"
        )
    for entity, table in tables.items():
        count = len(table) // 2
        ids = table[US_SCHEMA.entity_id_column(entity)].to_numpy()
        original = ids[:count]
        table[support_source_id_column(entity)] = np.tile(original, 2)
        table[spine_source_id_column(entity)] = np.tile(original + 2000, 2)
        if entity in ("person", "marital_unit"):
            channels = ["asec", "asec", "acs", "acs", "acs"]
        else:
            channels = ["asec", "acs", "acs", "acs"]
        table[support_channel_column(entity)] = pd.Series(channels * 2, dtype="string")
        table[support_clone_index_column(entity)] = np.array(
            [0] * count + [1] * count, dtype=np.int64
        )
    if case == "cross_group":
        # Both rows are ACS clone1, so source/clone checks pass first. Remove
        # the now-unreferenced fixture group to isolate household closure.
        person.loc[person.person_id == 104, "person_family_id"] = 120
        tables["family"] = (
            tables["family"]
            .loc[tables["family"].family_id != 130]
            .reset_index(drop=True)
        )
    elif case == "multiple_tax_units":
        person.loc[person.person_id == 102, "person_tax_unit_id"] = 999
        row = tables["tax_unit"].loc[tables["tax_unit"].tax_unit_id == 110].copy()
        row["tax_unit_id"] = 999
        row[support_source_id_column("tax_unit")] = 999
        row[spine_source_id_column("tax_unit")] = 2999
        tables["tax_unit"] = pd.concat([tables["tax_unit"], row], ignore_index=True)
    elif case == "clone2":
        person.loc[person.person_id == 101, support_clone_index_column("person")] = 2
    elif case == "membership_channel":
        tables["marital_unit"].loc[5, support_channel_column("marital_unit")] = "acs"
    elif case == "id_exhaustion":
        person.loc[person.person_id == 105, "person_id"] = np.iinfo(np.int64).max
        assert person.person_id.dtype == np.dtype("int64")
    elif case == "changed_payload":
        person.loc[0, "payload"] = 12.0
    elif case == "changed_membership":
        person.loc[person.person_id == 101, "person_marital_unit_id"] = 1102
        person.loc[person.person_id == 102, "person_marital_unit_id"] = 1101
    weights = {
        "household": Weights(np.array([16.0, 80.0, 0.0, 12.0] * 2), WeightKind.DESIGN)
    }
    if case == "extra_stored":
        # At the expansion boundary this is exactly the household-inherited
        # vector, including kind. Effective-weight comparison cannot detect it.
        weights["person"] = Weights(
            np.array([8.0, 8.0, 40.0, 0.0, 6.0] * 2), WeightKind.IMPORTANCE
        )
    return Frame(
        tables,
        US_SCHEMA,
        weights,
        pd.Series(["adult", "child", "adult", "zero", "gq"] * 2, dtype="string"),
    )


def domain_rows():
    rows = []
    for clone in (0, 1):
        for native in (10, 20, 30, 40):
            rows.append(
                tail.InventedHouseholdDomain(
                    native + clone * 100,
                    Source.ASEC if native == 10 else Source.ACS,
                    native,
                    native + 2000,
                    clone,
                    Domain.NONINSTITUTIONAL_GQ
                    if native == 40
                    else Domain.SHARED_HOUSING,
                    "person" if native == 40 else "occupied_housing_unit",
                )
            )
    return tuple(rows)


def declaration(*, domains=None, assignments=None):
    return tail.declare_invented_tail_expansion(
        domain_rows() if domains is None else domains,
        (
            tail.InventedTailAssignment(901, 110, 100.0),
            tail.InventedTailAssignment(902, 120, 10_000.0),
        )
        if assignments is None
        else assignments,
    )


def inventory(frame):
    structural = {US_SCHEMA.membership_column(g) for g in US_SCHEMA.group_entities}
    return tuple(
        Owned(entity, name, token_for_dtype(table[name].dtype))
        for entity in US_SCHEMA.entities
        for table in (frame.table(entity),)
        for name in table
        if name != US_SCHEMA.entity_id_column(entity)
        and not (entity == "person" and name in structural)
    )


class InventedHost(KernelBase):
    ref = "test.native_tail.literal_host@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        structural=StructuralDelta.CREATE,
    )

    def run(self, context):
        assert context.sources["host"].read_bytes() == b"invented tail structure v1"
        return KernelResult(frame=invented_frame(context.params["case"]))


class InventedImportance(KernelBase):
    ref = "test.native_tail.literal_importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        structural=StructuralDelta.REWEIGHT,
    )

    def run(self, context):
        values = np.array([8.0, 40.0, 0.0, 6.0] * 2)
        case = context.params["case"]
        if case == "zero_parent":
            values[4] = 0.0
        elif case == "underflow":
            values[4] = float.fromhex("0x0.0000000000001p-1022")
        elif case == "odd_subnormal":
            values[4] = float.fromhex("0x0.0000000000003p-1022")
        elif case == "even_subnormal":
            values[4] = float.fromhex("0x0.0000000000002p-1022")
        return KernelResult(weights=Weights(values, WeightKind.IMPORTANCE))


def graph_fixture(*, case="normal", declared=None):
    columns = inventory(invented_frame(case))
    nodes = tail.native_puf_tail_expand_nodes(
        columns, base="importance", declaration=declared or declaration()
    )
    graph = compile_graph(
        Graph(
            "us",
            (SourceRef("host", "raw-bytes-v1"),),
            (
                Node(
                    "host",
                    InventedHost.ref,
                    outputs=columns,
                    sources=("host",),
                    structural=StructuralDelta.CREATE,
                    params={"case": case},
                ),
                Node(
                    "importance",
                    InventedImportance.ref,
                    structural=StructuralDelta.REWEIGHT,
                    base="host",
                    weights=WeightTransition("household", "importance", "free"),
                    mass="free",
                    params={"case": case},
                ),
                *nodes,
            ),
        )
    )
    registry = KernelRegistry()
    registry.register(InventedHost())
    registry.register(InventedImportance())
    tail.register_native_puf_tail_expand_kernels(registry)
    return graph, registry


def execute(root, *, case="normal", declared=None, store=None, resume="auto"):
    root.mkdir(exist_ok=True)
    source = root / "host.txt"
    source.write_bytes(b"invented tail structure v1")
    graph, registry = graph_fixture(case=case, declared=declared)
    observed = {}
    result = run_graph(
        graph,
        sources={"host": source},
        store=store or ContentStore(root / "store"),
        kernels=registry,
        resume=resume,
        _population_observer=lambda name, pop: observed.update({name: pop}),
    )
    return result, observed, graph


EXPAND = "native_tail_fixture_expand"


def test_real_executor_preserves_six_lineages_mass_anchors_and_required_replay(
    tmp_path,
):
    store = ContentStore(tmp_path / "store")
    cold, cold_populations, graph = execute(tmp_path, store=store)
    warm, warm_populations, _ = execute(tmp_path, store=store, resume="require")
    assert all(not cold.nodes[n].hit and warm.nodes[n].hit for n in graph.order)
    assert cold.key == warm.key
    for result, populations in ((cold, cold_populations), (warm, warm_populations)):
        base = populations["importance"]
        expanded = populations[EXPAND]
        before, after = base.frame, expanded.frame
        receipt = result.nodes[EXPAND].receipt
        assert receipt["policy"] == tail.POLICY
        assert receipt["source_admission_issued"] is False
        assert receipt["monetary_placement"] is False
        assert receipt["matching_qualified"] is False
        assert receipt["release_eligible"] is False
        assert set(receipt["expand"]) == set(US_SCHEMA.entities)
        np.testing.assert_array_equal(
            after.weights_for("household").values,
            [8.0, 40.0, 0.0, 6.0, 4.0, 20.0, 0.0, 6.0, 4.0, 20.0],
        )
        for entity in US_SCHEMA.entities:
            old, new = before.table(entity), after.table(entity)
            pd.testing.assert_frame_equal(
                old.reset_index(drop=True),
                new.iloc[: len(old)].reset_index(drop=True),
                check_exact=True,
            )
            idcol = US_SCHEMA.entity_id_column(entity)
            lineage = dict(receipt["expand"][entity])
            assert len(new) == len(old) + len(lineage)
            structural = {idcol}
            if entity == "person":
                structural.update(
                    US_SCHEMA.membership_column(g) for g in US_SCHEMA.group_entities
                )
            carried = [
                c
                for c in old
                if c not in structural and c != support_clone_index_column(entity)
            ]
            copied = new.set_index(idcol).loc[list(lineage)]
            parent = old.set_index(idcol).loc[list(lineage.values())]
            pd.testing.assert_frame_equal(
                copied[carried].reset_index(drop=True),
                parent[carried].reset_index(drop=True),
                check_exact=True,
            )
            if entity == "person":
                # Numeric equality alone treats -0.0 and +0.0 as equal.
                assert (
                    old.payload.to_numpy().tobytes()
                    == new.payload.iloc[: len(old)].to_numpy().tobytes()
                )
                assert (
                    copied.payload.to_numpy().tobytes()
                    == parent.payload.to_numpy().tobytes()
                )
            assert copied[support_clone_index_column(entity)].eq(2).all()
            assert (
                before.resolve_weights(entity).total
                == after.resolve_weights(entity).total
            )
        anchors = base.design_weights["household"]
        np.testing.assert_array_equal(
            expanded.design_weights["household"],
            np.concatenate([anchors, anchors[[4, 5]]]),
        )
        before_person_mass = before.resolve_weights("person").values
        after_person_mass = after.resolve_weights("person").values
        for stratum in before.strata.unique():
            assert (
                before_person_mass[before.strata.eq(stratum)].sum()
                == after_person_mass[after.strata.eq(stratum)].sum()
            )
        person_lineage = dict(receipt["expand"]["person"])
        for group in US_SCHEMA.group_entities:
            membership = US_SCHEMA.membership_column(group)
            group_lineage = {
                parent: target for target, parent in receipt["expand"][group]
            }
            old_people = before.person.set_index("person_id")
            new_people = after.person.set_index("person_id")
            for target, source in person_lineage.items():
                assert (
                    new_people.loc[target, membership]
                    == group_lineage[old_people.loc[source, membership]]
                )
        ledger = result.mass_ledger(EXPAND)[-1]
        assert ledger.policy == "conserve"
        assert ledger.before_total == ledger.after_total


def test_declared_readset_only_exposes_provenance_and_structural_columns():
    graph, _ = graph_fixture()
    node = next(node for node in graph.graph.nodes if node.id == EXPAND)
    assert {(s.entity, c) for s in node.inputs for c in s.columns} == {
        (e, c) for e in US_SCHEMA.entities for c in tail.provenance_columns(e)
    }
    assert not any("payload" in s.columns for s in node.inputs)
    assert node.params["expand_require_sole_weight_entity"] is True
    assert EXPAND in graph.predecessors[EXPAND + ".owned"]


def test_fixed_assignment_donor_weights_do_not_enter_host_mass(tmp_path):
    first, _, _ = execute(tmp_path / "first")
    original = declaration()
    changed = declaration(
        assignments=tuple(
            replace(pair, donor_support_weight=pair.donor_support_weight * 17)
            for pair in original.assignments
        )
    )
    second, _, _ = execute(tmp_path / "second", declared=changed)
    assert first.nodes[EXPAND].key != second.nodes[EXPAND].key
    np.testing.assert_array_equal(
        first.population(EXPAND).weights_for("household").values,
        second.population(EXPAND).weights_for("household").values,
    )
    assert original.sha256 != changed.sha256


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("cross_group", "GROUP_NOT_HOUSEHOLD_CLOSED"),
        ("multiple_tax_units", "PARENT_TAX_UNITS"),
        ("clone2", "BASE_CLONE"),
        ("membership_channel", "MEMBERSHIP_CHANNEL"),
        ("id_exhaustion", "ID_EXHAUSTION"),
        ("zero_parent", "POSITIVE_HALF"),
        ("underflow", "POSITIVE_HALF"),
        ("odd_subnormal", "EXACT_HALF"),
        ("extra_stored", "sole stored weight entity"),
    ],
)
def test_live_executor_refusals(tmp_path, case, error):
    with pytest.raises((ValueError, NodeRejectedError), match=error):
        execute(tmp_path, case=case)


def test_even_subnormal_half_remains_positive_and_exact(tmp_path):
    result, _, _ = execute(tmp_path, case="even_subnormal")
    values = result.population(EXPAND).weights_for("household").values
    assert values[4] > 0 and values[4] == values[8]
    assert values[4] + values[8] == float.fromhex("0x0.0000000000002p-1022")


@pytest.mark.parametrize(
    "field", ["source", "support_source_id", "spine_source_id", "clone_index"]
)
def test_complete_fixture_domain_origin_must_equal_live_host(tmp_path, field):
    rows = list(domain_rows())
    row = rows[0]
    changes = {
        "source": Source.ACS,
        "support_source_id": 666,
        "spine_source_id": 667,
        "clone_index": 1,
    }
    rows[0] = replace(row, **{field: changes[field]})
    with pytest.raises((ValueError, NodeRejectedError), match="DOMAIN_ORIGIN"):
        execute(tmp_path, declared=declaration(domains=rows))


def test_missing_unselected_domain_is_not_silently_assumed_housing(tmp_path):
    with pytest.raises((ValueError, NodeRejectedError), match="DOMAIN_COVERAGE"):
        execute(tmp_path, declared=declaration(domains=domain_rows()[:-1]))


@pytest.mark.parametrize(
    ("parent", "error"),
    [(140, "PARENT_HOUSING"), (10, "PARENT_CLONE"), (999, "PARENT_MISSING")],
)
def test_ineligible_parent_declarations_refuse(parent, error):
    with pytest.raises(ValueError, match=error):
        declaration(assignments=(tail.InventedTailAssignment(901, parent, 1.0),))


@pytest.mark.parametrize(
    "change",
    [
        "unknown_domain",
        "unknown_source",
        "contradictory_unit",
        "duplicate_domain",
        "duplicate_parent",
        "duplicate_donor",
    ],
)
def test_declaration_refuses_unknown_or_nonunique_claims(change):
    rows, pairs = list(domain_rows()), list(declaration().assignments)
    if change == "unknown_domain":
        rows[0] = replace(rows[0], domain=None)
    elif change == "unknown_source":
        rows[0] = replace(rows[0], source="asec")
    elif change == "contradictory_unit":
        rows[0] = replace(rows[0], statistical_unit="person")
    elif change == "duplicate_domain":
        rows.append(rows[0])
    elif change == "duplicate_parent":
        pairs[1] = replace(pairs[1], parent_household_id=110)
    else:
        pairs[1] = replace(pairs[1], donor_id=901)
    with pytest.raises(ValueError):
        declaration(domains=rows, assignments=pairs)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), True, 1])
def test_donor_metadata_must_be_known_positive_float(weight):
    with pytest.raises(ValueError, match="DONOR_SUPPORT_WEIGHT"):
        declaration(assignments=(tail.InventedTailAssignment(901, 110, weight),))


@pytest.mark.parametrize("case", ["changed_payload", "changed_membership"])
def test_changed_base_cannot_reuse_required_cache(tmp_path, case):
    store = ContentStore(tmp_path / "store")
    before, _, _ = execute(tmp_path, store=store)
    with pytest.raises(StoreMissError):
        execute(tmp_path, store=store, case=case, resume="require")
    after, _, _ = execute(tmp_path, store=store, case=case)
    assert before.nodes[EXPAND].key != after.nodes[EXPAND].key


def test_changed_assignment_cannot_reuse_required_cache(tmp_path):
    store = ContentStore(tmp_path / "store")
    execute(tmp_path, store=store)
    changed = declaration(
        assignments=(
            tail.InventedTailAssignment(902, 110, 10_000.0),
            tail.InventedTailAssignment(901, 120, 100.0),
        )
    )
    with pytest.raises(StoreMissError):
        execute(tmp_path, store=store, declared=changed, resume="require")


def test_missing_cached_frame_refuses_without_kernel_fallback(tmp_path):
    store = ContentStore(tmp_path / "store")
    result, _, _ = execute(tmp_path, store=store)
    # Only an invented object produced by this test is removed, never a retained store.
    path = store.object_path(result.nodes[EXPAND].frame_key)
    metadata = path / "meta.json"
    assert metadata.is_file()
    metadata.unlink()
    with pytest.raises(StoreCorruptError):
        execute(tmp_path, store=store, resume="require")


def test_declaration_snapshots_order_and_does_not_issue_authority():
    rows, pairs = list(domain_rows()), list(declaration().assignments)
    first = tail.declare_invented_tail_expansion(rows, pairs)
    second = tail.declare_invented_tail_expansion(reversed(rows), reversed(pairs))
    rows.clear()
    pairs.clear()
    assert first == second and first.sha256 == second.sha256
    assert not first.source_admission_issued and not first.release_eligible


@pytest.mark.parametrize("change", ["missing", "wrong_dtype", "duplicate"])
def test_factory_requires_complete_provenance_inventory(change):
    columns = list(inventory(invented_frame()))
    position = next(
        i
        for i, c in enumerate(columns)
        if c.column == support_source_id_column("person")
    )
    if change == "missing":
        columns.pop(position)
    elif change == "wrong_dtype":
        columns[position] = replace(columns[position], dtype="float64")
    else:
        columns.append(columns[position])
    with pytest.raises(ValueError, match="INVENTORY"):
        tail.native_puf_tail_expand_nodes(
            columns, base="importance", declaration=declaration()
        )
