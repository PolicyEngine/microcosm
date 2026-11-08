"""Tests split from packages/microcosm-build/tests/test_uk_graph.py."""

# ruff: noqa: F403, F405
import functools
import hashlib
from typing import NamedTuple

from microcosm.graph import graph_schema, orrery_document
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.uk_graph import *


def test_uk_expand_contract_carries_cells_links_weights_and_design_lineage() -> None:
    expanded = patch(_expand_population(), _expand_node(), _expand_result())

    person = expanded.frame.table("person")
    assert person["person_id"].tolist() == [1, 2, 3]
    assert person["person_household_id"].tolist() == [10, 20, 30]
    assert person["hidden_payload"].tolist() == [1.25, 9.5, 1.25]
    assert expanded.frame.table("household")["is_clone"].tolist() == [
        False,
        False,
        True,
    ]
    assert expanded.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(
        expanded.frame.weights_for("household").values,
        np.array([0.5, 2.0, 0.5]),
    )
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0, 1.0])
    )
    assert expanded.frame.mass_log[-1].reason == "test clone mass is conserved"
    assert expanded.mass_ledger[-1].operation == "expand"


def test_uk_expand_contract_rejects_unknown_source_ids() -> None:
    with pytest.raises(PopulationError, match="unknown 'person' source ids"):
        patch(_expand_population(), _expand_node(), _expand_result(bad_source=True))


def test_uk_spine_graph_contains_every_manifest_stage() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(stage.stage for stage in spec.sources.stages)
    graph = uk_spine_graph(spec)
    ids = {node.id for node in graph.nodes}

    # 35 with the #832 uc_reporter_redraw, #685 uc_deduction_attributes,
    # #791 frs_relationships, #725 hmrc_cgt_asset_type_spine, #970
    # cgt_incidence_anchor, #930 nts_bus_travel and the income-anchor lane's
    # (PolicyEngine/chronicle#280) spi_income_band_donors stages, the #1012
    # spi_housing_shell, the #1003 was_lisa, the microcosm#1069
    # pension_credit_take_up and the microcosm#1063 child_benefit_take_up; the
    # frs_hmrc_retained_leaves / hmrc_spi_income certified-pair alternatives
    # are retired (#901), so the manifest roster is the graph roster.
    # microcosm#1063 also adds cgt_residential_split after the anchor,
    # microcosm#1095 the spi_benefit_coherence pass after uc_reporter_redraw,
    # microcosm#1106 property_components after the regional uprating, and
    # microcosm#1240 high_value_home_support between those two; microcosm#1095
    # adds property_purchases and uc_legacy_closure_take_up, microcosm#1112
    # regional_home_values, microcosm#1319 carers_allowance_take_up, the
    # copies' childcare re-draw childcare_copy_take_up, and microcosm#1334
    # uc_take_up and frs_annual_income_spells; microcosm#1428 the last stage,
    # trading_allowance_receipts; and company_cars after hmrc_spi_income_spine
    # (PolicyEngine/uk-autumn-budget-dashboard-2026#316); and microcosm#1490
    # spi_employment_shell after company_cars, then aps_occupation.
    assert len(expected) == 54
    assert {"frs_hmrc_retained_leaves", "hmrc_spi_income"}.isdisjoint(expected)
    assert set(expected) <= ids
    assert {"frs_hmrc_retained_leaves", "hmrc_spi_income"}.isdisjoint(ids)
    root_dtypes = {
        (owned.entity, owned.column): owned.dtype
        for owned in graph.node("create_uk_frs").outputs
    }
    assert root_dtypes[("person", "is_uc_claimant")] == "bool"
    assert {
        node.id for node in graph.nodes if node.structural is StructuralDelta.EXPAND
    } == UK_SPINE_STRUCTURAL_STAGES


def test_uk_spine_compile_order_is_derived_from_declared_inputs() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(stage.stage for stage in spec.sources.stages)
    compiled = compile_graph(uk_spine_graph(spec))
    stage_order = tuple(node_id for node_id in compiled.order if node_id in expected)

    assert set(stage_order) == set(expected)
    assert len(stage_order) == len(expected)
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )
    assert all(
        compiled.graph.node(node_id).inputs
        for node_id in expected[1:]
        if node_id not in UK_SPINE_STRUCTURAL_STAGES
    )
    graph = compiled.graph
    reversed_declaration = Graph(
        graph.country, graph.sources, tuple(reversed(graph.nodes))
    )
    assert compile_graph(reversed_declaration).order == compiled.order
    assert "frs_employment" in compiled.predecessors["frs_legacy_proxies"]
    assert "was_wealth" in compiled.predecessors["regional_property_uprating.boundary"]
    assert (
        "regional_property_uprating" in compiled.predecessors["high_value_home_support"]
    )
    assert "high_value_home_support" in compiled.predecessors["property_components"]
    assert "property_components" in compiled.predecessors["lcfs_consumption"]
    assert "spi_support_channel" in compiled.predecessors["spi_income_band_donors"]
    assert "spi_income_band_donors" in compiled.predecessors["hmrc_spi_income_spine"]


def test_uk_production_graph_binds_split_donor_sources_and_runtime_config() -> None:
    graph = uk_spine_graph(
        source_mode="split",
        sample_fraction=0.1,
        sample_seed=999,
    )

    assert {source.name for source in graph.sources} == {
        "frs",
        "was",
        "was_person",
        "nts_household",
        "nts_individual",
        "nts_trip",
        "nts_stage",
        "nts_ticket",
        "nts_vehicle",
        "lcfs_household",
        "lcfs_person",
        "etb",
        "aps",
        "spi",
        "hmrc_income",
    }
    # The NTS household and vehicle tabs feed the zero-emission share by income
    # fifth and the household-paid road-fuel mix (microcosm#1095 B3).
    assert graph.node("lcfs_consumption").sources == (
        "lcfs_household",
        "lcfs_person",
        "nts_household",
        "nts_vehicle",
    )
    assert graph.node("was_lisa").sources == ("was", "was_person")
    assert graph.node("aps_occupation").sources == ("aps",)
    assert graph.node("hmrc_spi_income_spine").sources == (
        "spi",
        "hmrc_income",
    )
    assert graph.node("hmrc_cgt_gains_spine").sources == ()
    create = graph.node("create_uk_frs")
    assert create.params["sample_fraction"] == 0.1
    assert create.params["sample_seed"] == 999
    assert len(str(create.params["stage_contract_sha256"])) == 64
    assert all(
        len(str(node.params["stage_contract_sha256"])) == 64
        for node in graph.nodes
        if "stage" in node.params
    )


def test_uk_registry_covers_every_kernel_ref_and_hashes_stage_modules() -> None:
    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)

    assert set(registry.refs()) == {node.kernel for node in graph.nodes}
    assert registry.implementation_hash(
        "uk.stage.frs_employment@1"
    ) != registry.implementation_hash("uk.stage.frs_council_tax@1")


def test_uc_relationship_helper_changes_affected_kernel_hashes(monkeypatch) -> None:
    from microcosm.build.uk_runtime import uc_relationships

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected_refs = (
        graph.node("create_uk_frs").kernel,
        graph.node("uc_reporter_redraw").kernel,
        graph.node("uc_capital_coherence").kernel,
    )
    control_ref = graph.node("frs_council_tax").kernel
    before = {
        ref: registry.implementation_hash(ref) for ref in (*affected_refs, control_ref)
    }
    helper_path = Path(uc_relationships.__file__).resolve()
    original_read_bytes = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)

    for ref in affected_refs:
        assert registry.implementation_hash(ref) != before[ref]
    assert registry.implementation_hash(control_ref) == before[control_ref]


def test_spi_income_source_changes_invalidate_the_stages_that_run_it(monkeypatch):
    # The SPI income stage's module only imports the imputation, the band-donor
    # resample and its receipt from spi_income, so spi_income is bound as the
    # stage's helper; the landlords' stage already binds it (microcosm#1278).
    from microcosm.build.uk_runtime import spi_income

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected = [
        graph.node(stage).kernel
        for stage in ("hmrc_spi_income_spine", "property_components")
    ]
    control = graph.node("frs_council_tax").kernel
    before = {ref: registry.implementation_hash(ref) for ref in (*affected, control)}
    helper_path = Path(spi_income.__file__).resolve()
    original = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref]
    assert registry.implementation_hash(control) == before[control]


def _imported_module_names(module) -> set[str]:
    import ast
    import inspect

    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return names


def test_every_stage_that_counts_by_lineage_binds_the_counting_modules() -> None:
    """An edit to how a floor counts moves the hash of every stage that counts.

    ``source_hash`` does not follow imports, so the receipts' counting rules,
    the shared lineage reader and the UK's lineage declaration are bound into
    each stage whose module or helpers import one of them (microcosm#931).
    """
    from microcosm.build.uk_runtime import graph_kernels

    counting = {"receipt_counts", "household_lineage", "source_lineage"}
    expected = {
        stage
        for stage in graph_kernels._STAGE_MODULES
        if any(
            counting & _imported_module_names(module)
            for module in (
                graph_kernels._stage_module(stage),
                *graph_kernels._STAGE_HELPER_MODULES.get(stage, ()),
            )
        )
    }
    assert set(graph_kernels._COUNTING_STAGES) == expected


def test_counting_module_changes_invalidate_every_stage_that_counts(monkeypatch):
    from microcosm.build import source_lineage
    from microcosm.build.uk_runtime import graph_kernels, receipt_counts

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected = [
        graph.node(stage).kernel for stage in sorted(graph_kernels._COUNTING_STAGES)
    ]
    control = graph.node("frs_council_tax").kernel
    for module in (receipt_counts, source_lineage):
        before = {
            ref: registry.implementation_hash(ref) for ref in (*affected, control)
        }
        helper_path = Path(module.__file__).resolve()
        original = Path.read_bytes

        def changed_helper_bytes(path: Path, helper=helper_path, read=original):
            content = read(path)
            if path.resolve() == helper:
                return content + b"\n# A helper-only source change.\n"
            return content

        with monkeypatch.context() as patched:
            patched.setattr(Path, "read_bytes", changed_helper_bytes)
            for ref in affected:
                assert registry.implementation_hash(ref) != before[ref], ref
            assert registry.implementation_hash(control) == before[control]


#: The two SPI support stages mint their copies' ids through the support
#: channel's clone helper (``spi_support``), which reads the copy-id table;
#: every other minting stage's module reads the table itself (microcosm#931).
_SPI_COPY_ID_STAGES = frozenset({"spi_support_channel", "spi_income_band_donors"})
#: Stages that bind ``spi_support`` or ``frs_spine``, which read the copy-id
#: table, for names that never mint, decode or bound a copy id: constants,
#: the support channel's column names and source fill, and frs_spine's tab
#: reader and self-employment rule. They need not bind the table.
_COPY_ID_TABLE_NAME_ONLY_STAGES = frozenset(
    {"company_cars", "property_components", "spi_benefit_coherence"}
)
#: The functions those stages may import from the two helpers.
_COPY_ID_TABLE_FREE_HELPER_FUNCTIONS = frozenset(
    {
        "fill_support_channel_from_source",
        "read_pinned_tab",
        "support_channel_column",
        "support_source_id_column",
        "uc_gainful_self_employment",
    }
)


def _names_imported_from(module, helpers: set[str]) -> set[str]:
    import ast
    import inspect

    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.rsplit(".", 1)[-1] in helpers
        for alias in node.names
    }


def test_every_stage_that_mints_or_bounds_copy_ids_binds_the_copy_id_table() -> None:
    """``source_hash`` does not follow imports, so every stage whose module or
    bound helpers read the copy-id table binds it, save the stages that bind
    a reading helper for table-free names alone (microcosm#931)."""
    from microcosm.build.uk_runtime import copy_id_table, graph_kernels, spi_support

    reading = {
        stage
        for stage in graph_kernels._STAGE_MODULES
        if any(
            "copy_id_table" in _imported_module_names(module)
            for module in (
                graph_kernels._stage_module(stage),
                *graph_kernels._STAGE_HELPER_MODULES.get(stage, ()),
            )
        )
    }
    binding = {
        stage
        for stage, helpers in graph_kernels._STAGE_HELPER_MODULES.items()
        if copy_id_table in helpers
    }
    assert _COPY_ID_TABLE_NAME_ONLY_STAGES <= reading
    assert binding == reading - _COPY_ID_TABLE_NAME_ONLY_STAGES
    assert _SPI_COPY_ID_STAGES <= binding
    for stage in _SPI_COPY_ID_STAGES:
        assert spi_support in graph_kernels._STAGE_HELPER_MODULES[stage]
    for stage in _COPY_ID_TABLE_NAME_ONLY_STAGES:
        module = graph_kernels._stage_module(stage)
        assert "copy_id_table" not in _imported_module_names(module), stage
        functions = {
            name
            for name in _names_imported_from(module, {"spi_support", "frs_spine"})
            if not name.isupper()
        }
        assert functions <= _COPY_ID_TABLE_FREE_HELPER_FUNCTIONS, stage


def test_copy_id_table_changes_invalidate_every_stage_that_reads_it(monkeypatch):
    from microcosm.build.uk_runtime import copy_id_table, graph_kernels

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    stages = sorted(
        stage
        for stage, helpers in graph_kernels._STAGE_HELPER_MODULES.items()
        if copy_id_table in helpers
    )
    # The root stage's node is the CREATE kernel.
    affected = [
        graph.node("create_uk_frs" if stage == "frs_spine" else stage).kernel
        for stage in stages
    ]
    controls = [
        graph.node(stage).kernel
        for stage in ("frs_council_tax", "hmrc_spi_income_spine", "isa_subscriptions")
    ]
    before = {ref: registry.implementation_hash(ref) for ref in affected + controls}
    helper_path = Path(copy_id_table.__file__).resolve()
    original = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)
    assert len(affected) == 8
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref], ref
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref], ref


_STOCHASTIC_ASSIGNMENT = "microcosm.build.stochastic_assignment"
#: Stages that import a function drawing through the keyed-draw helpers
#: (``microcosm.build.stochastic_assignment``) but never call it, so need not
#: bind them: the stage's transform, the drawing names its module imports, and
#: why the transform never calls them.
_STOCHASTIC_ASSIGNMENT_UNBOUND: dict[str, tuple[str, frozenset[str], str]] = {
    "spi_support_channel": (
        "UKSPISupportChannelStageTransform",
        frozenset(
            {
                "employer_pension_contributions.draw_employer_pension_contributions",
                "spi_income.impute_uk_spi_income_support",
            }
        ),
        "its module, spi_spine, also holds frs_hmrc_spine_leaves' employer "
        "pension draw and hmrc_spi_income_spine's SPI income imputation, both "
        "of which bind the helpers; the support channel's transform calls "
        "neither",
    ),
}


class _DrawFacts(NamedTuple):
    """What one first-party module imports and what its definitions name."""

    imports: tuple[tuple[str, str, str], ...]
    module_aliases: dict[str, str]
    references: dict[str, frozenset[str]]


@functools.cache
def _draw_facts(name: str) -> _DrawFacts | None:
    import ast
    import importlib.util

    if not name.startswith("microcosm.") or name == _STOCHASTIC_ASSIGNMENT:
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or "").endswith(".py"):
        return None
    package = name if spec.submodule_search_locations else name.rpartition(".")[0]
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imports: list[tuple[str, str, str]] = []
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level else ""
            origin = (
                ".".join(part for part in (base, node.module) if part)
                if node.level
                else node.module or ""
            )
            for alias in node.names:
                local = alias.asname or alias.name
                imports.append((local, origin, alias.name))
                module_aliases[local] = f"{origin}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    module_aliases[alias.asname] = alias.name
    references = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                    names.add(f"{sub.value.id}.{sub.attr}")
            references[node.name] = frozenset(names)
    return _DrawFacts(tuple(imports), module_aliases, references)


def _drawing_names(roots) -> dict[str, frozenset[str]]:
    """Each first-party module's names that draw through the keyed-draw helpers.

    A definition draws when it names one of their functions or another drawing
    name, of its own module or imported from (or reached by attribute on)
    another; a module's drawing names include the drawing names it imports, so
    a re-export draws too. A fixed point over every module the roots reach.
    """

    facts: dict[str, _DrawFacts] = {}
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in facts:
            continue
        found = _draw_facts(name)
        if found is None:
            continue
        facts[name] = found
        pending.extend(origin for _, origin, _ in found.imports)
        pending.extend(found.module_aliases.values())
    drawing: dict[str, set[str]] = {name: set() for name in facts}

    def draws(module: str, local: str) -> bool:
        info = facts[module]
        if local in drawing[module]:
            return True
        if "." in local:
            head, _, attribute = local.partition(".")
            target = info.module_aliases.get(head)
            return target == _STOCHASTIC_ASSIGNMENT or attribute in drawing.get(
                target, ()
            )
        return any(
            local == alias
            and (
                origin == _STOCHASTIC_ASSIGNMENT or imported in drawing.get(origin, ())
            )
            for alias, origin, imported in info.imports
        )

    changed = True
    while changed:
        changed = False
        for module, info in facts.items():
            for local, _, _ in info.imports:
                if local not in drawing[module] and draws(module, local):
                    drawing[module].add(local)
                    changed = True
            for definition, names in info.references.items():
                if definition not in drawing[module] and any(
                    draws(module, name) for name in names
                ):
                    drawing[module].add(definition)
                    changed = True
    return {module: frozenset(names) for module, names in drawing.items()}


def _stochastic_assignment_reach() -> dict[str, frozenset[str]]:
    """Every spine stage that draws through the keyed-draw helpers, and how.

    A stage draws when its module or a bound helper imports the helpers, or
    imports from another first-party module a function that draws through
    them. Each such stage maps to the drawing names it imports that way, as
    ``module.name`` (empty when it imports the helpers themselves).
    """

    from microcosm.build.uk_runtime import graph_kernels

    hashed = {
        stage: (
            graph_kernels._stage_module(stage),
            *graph_kernels._STAGE_HELPER_MODULES.get(stage, ()),
        )
        for stage in graph_kernels._STAGE_MODULES
    }
    drawing = _drawing_names(
        {module.__name__ for modules in hashed.values() for module in modules}
    )
    reach: dict[str, frozenset[str]] = {}
    for stage, modules in hashed.items():
        if any(
            "stochastic_assignment" in _imported_module_names(module)
            for module in modules
        ):
            reach[stage] = frozenset()
            continue
        found = set()
        for module in modules:
            facts = _draw_facts(module.__name__)
            if facts is None:
                continue
            for _, origin, imported in facts.imports:
                if imported in drawing.get(origin, ()):
                    found.add(f"{origin.rpartition('.')[2]}.{imported}")
            # A drawing function reached by attribute on an imported module.
            for names in facts.references.values():
                for name in names:
                    head, _, attribute = name.partition(".")
                    target = facts.module_aliases.get(head) if attribute else None
                    if target and attribute in drawing.get(target, ()):
                        found.add(f"{target.rpartition('.')[2]}.{attribute}")
        if found:
            reach[stage] = frozenset(found)
    return reach


def _stages_drawing_through_stochastic_assignment() -> set[str]:
    return set(_stochastic_assignment_reach())


def test_every_stage_that_draws_through_stochastic_assignment_binds_it() -> None:
    """``source_hash`` does not follow imports, so every stage whose module or
    bound helpers import the keyed-draw helpers, or import another module's
    function that draws through them, binds them, or an edit to how a uniform
    is keyed would replay the stage's stale output (microcosm#931), save the
    stages ``_STOCHASTIC_ASSIGNMENT_UNBOUND`` names with a reason."""
    from microcosm.build import stochastic_assignment
    from microcosm.build.uk_runtime import graph_kernels

    reach = _stochastic_assignment_reach()
    binding = {
        stage
        for stage, helpers in graph_kernels._STAGE_HELPER_MODULES.items()
        if stochastic_assignment in helpers
    }
    assert binding == set(reach) - set(_STOCHASTIC_ASSIGNMENT_UNBOUND)
    # The employer pension draw reaches stage 14 through spi_spine alone.
    assert reach["frs_hmrc_spine_leaves"] >= {
        "employer_pension_contributions.draw_employer_pension_contributions"
    }
    # An exempt stage imports exactly the drawing names its reason covers, and
    # its transform names none of them, directly or through its module.
    for stage, (transform, names, _) in _STOCHASTIC_ASSIGNMENT_UNBOUND.items():
        assert reach.get(stage) == names, stage
        module = graph_kernels._stage_module(stage).__name__
        assert transform in _draw_facts(module).references, stage
        assert transform not in _drawing_names([module])[module], stage


def test_stochastic_assignment_changes_invalidate_every_stage_that_draws(
    monkeypatch,
):
    from microcosm.build import stochastic_assignment
    from microcosm.build.uk_runtime import graph_kernels

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    stages = sorted(
        stage
        for stage, helpers in graph_kernels._STAGE_HELPER_MODULES.items()
        if stochastic_assignment in helpers
    )
    # Stage 14 draws through the employer pension helper, which binds nothing.
    assert "frs_hmrc_spine_leaves" in stages
    # The root stage's node is the CREATE kernel.
    affected = [
        graph.node("create_uk_frs" if stage == "frs_spine" else stage).kernel
        for stage in stages
    ]
    # Three stages that draw through nothing of it.
    control_stages = (
        "frs_council_tax",
        "cgt_support_split",
        "hmrc_cgt_asset_type_spine",
    )
    drawing = _stages_drawing_through_stochastic_assignment()
    assert not set(control_stages) & drawing
    controls = [graph.node(stage).kernel for stage in control_stages]
    before = {ref: registry.implementation_hash(ref) for ref in affected + controls}
    helper_path = Path(stochastic_assignment.__file__).resolve()
    original = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)
    assert len(affected) == len(drawing - set(_STOCHASTIC_ASSIGNMENT_UNBOUND))
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref], ref
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref], ref


def test_isa_statistics_source_changes_invalidate_the_stages_that_read_them(
    monkeypatch,
):
    # The ISA subscriptions and the Lifetime ISA anchor read HMRC's ISA tables
    # through the typed loader, so it is bound as both stages' helper
    # (microcosm#1415).
    from microcosm.build.uk_runtime import hmrc_isa_statistics

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected = [graph.node(stage).kernel for stage in ("isa_subscriptions", "was_lisa")]
    control = graph.node("frs_council_tax").kernel
    before = {ref: registry.implementation_hash(ref) for ref in (*affected, control)}
    helper_path = Path(hmrc_isa_statistics.__file__).resolve()
    original = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref]
    assert registry.implementation_hash(control) == before[control]


#: What calls a rules engine, whatever it is called on: ``materialize`` (the
#: RulesEngine protocol), the shared batch driver, and the predictor
#: materialization of ``microcosm.frame.rules``, which calls it.
_ENGINE_CALLS = frozenset(
    {"materialize", "materialize_in_batches", "materialize_rules_engine_predictors"}
)

#: Stages whose transform takes the engine without computing through it, each
#: with why. The ISA subscriptions only check that the engine is the UK's
#: (assert_rules_engine_country).
_ENGINE_TAKEN_FOR_ITS_COUNTRY_ONLY = frozenset({"isa_subscriptions"})


def _calls_the_engine(module) -> bool:
    """Whether a module calls a rules engine: any call named in
    ``_ENGINE_CALLS``, on any receiver (``engine``, ``self.engine``,
    ``self._engine``, ``rules_engine`` ...) or as a bare name."""
    import ast
    import inspect

    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _ENGINE_CALLS:
            return True
        if isinstance(function, ast.Name) and function.id in _ENGINE_CALLS:
            return True
    return False


def _takes_an_engine(module) -> bool:
    """Whether a class the module defines (a stage transform) takes an
    ``engine`` when it is built."""
    import inspect

    for _name, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ != module.__name__:
            continue
        try:
            parameters = inspect.signature(member).parameters
        except (TypeError, ValueError):
            continue
        if "engine" in parameters:
            return True
    return False


def test_every_stage_that_calls_the_engine_binds_the_engine_modules() -> None:
    """How an engine call runs, and clone by clone once a frame carries more
    than one clone, is the adapter's, the shared engine-block machinery's and
    the UK declaration's (microcosm#931). ``source_hash`` does not follow
    imports, so every stage whose module calls the engine, or whose transform
    takes one, binds all three, and no other stage binds any. The one
    exception takes the engine only to check its country and computes
    nothing through it."""
    from microcosm.build.uk_runtime import graph_kernels

    engine_modules = set(graph_kernels._ENGINE_MODULES)
    modules = {
        stage: graph_kernels._stage_module(stage)
        for stage in graph_kernels._STAGE_MODULES
    }
    calling = {stage for stage, module in modules.items() if _calls_the_engine(module)}
    taking = {stage for stage, module in modules.items() if _takes_an_engine(module)}
    assert {
        "frs_legacy_proxies",
        "frs_brma",
        "was_lisa",
        "uc_take_up",
        "private_school_attendance",
        "company_cars",
    } <= calling
    # The exception is still one: it takes the engine and never calls it.
    assert _ENGINE_TAKEN_FOR_ITS_COUNTRY_ONLY <= taking - calling
    computing = calling | (taking - _ENGINE_TAKEN_FOR_ITS_COUNTRY_ONLY)
    for stage in graph_kernels._STAGE_MODULES:
        bound = set(graph_kernels._STAGE_HELPER_MODULES.get(stage, ()))
        if stage in computing:
            assert engine_modules <= bound, stage
        else:
            assert not engine_modules & bound, stage


def test_the_engine_call_scan_sees_any_receiver(tmp_path) -> None:
    import types

    for source, calls in (
        ("rules_engine.materialize(frame, names, period)", True),
        ("self._engine.materialize(frame, names, period)", True),
        ("helpers.materialize_in_batches(frame, names, period)", True),
        ("materialize_rules_engine_predictors(frame)", True),
        ("frame.select(rows)", False),
    ):
        path = tmp_path / "stage.py"
        path.write_text(f"def transform(self, frame):\n    {source}\n")
        module = types.ModuleType("stage")
        module.__file__ = str(path)
        assert _calls_the_engine(module) is calls, source


@pytest.mark.parametrize(
    "module_name",
    ["policyengine_uk", "engine_population", "policyengine_uk_population"],
)
def test_engine_module_changes_invalidate_every_stage_that_calls_the_engine(
    monkeypatch, module_name
):
    import importlib

    from microcosm.build.uk_runtime import graph_kernels

    module = importlib.import_module(f"microcosm.frame.adapters.{module_name}")
    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    stages = sorted(
        stage
        for stage, helpers in graph_kernels._STAGE_HELPER_MODULES.items()
        if module in helpers
    )
    assert "frs_legacy_proxies" in stages and "private_school_attendance" in stages
    affected = [graph.node(stage).kernel for stage in stages]
    controls = [
        graph.node(stage).kernel
        for stage in ("frs_council_tax", "uc_capital_coherence", "isa_subscriptions")
    ]
    before = {ref: registry.implementation_hash(ref) for ref in affected + controls}
    module_path = Path(module.__file__).resolve()
    original = Path.read_bytes

    def changed_module_bytes(path):
        content = original(path)
        return (
            content + b"\n# engine-module-only change\n"
            if path.resolve() == module_path
            else content
        )

    monkeypatch.setattr(Path, "read_bytes", changed_module_bytes)
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref], ref
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref], ref


def test_uk_graph_json_round_trip_is_canonical() -> None:
    graph = uk_spine_graph()
    serialized = graph_to_json(graph)

    assert graph_from_json(serialized) == graph
    assert graph_to_json(graph_from_json(serialized)) == serialized


def test_uk_spine_exports_complete_orrery_document() -> None:
    compiled = compile_graph(uk_spine_graph())
    schema = graph_schema(compiled)
    materialized_claims = [
        binding
        for binding in schema["input_bindings"]
        if binding["kind"] == "materialized_expand_output"
    ]

    document = orrery_document(compiled, title="UK spine")
    nodes = {node["id"]: node for node in document["nodes"]}
    materialized_reads = [
        edge
        for edge in document["edges"]
        if edge["kind"] == "declared_read"
        and edge["data"]["read_kind"] == "materialized_expand_output"
    ]
    expected_reads = {
        (
            binding["node"],
            binding["population"],
            binding["entity"],
            binding["column"],
            binding["provider"],
            binding["declared_in"],
            binding["rows"],
        )
        for binding in materialized_claims
    }
    actual_reads = {
        (
            nodes[edge["target"]]["label"],
            nodes[edge["source"]]["data"]["population"],
            nodes[edge["source"]]["data"]["entity"],
            nodes[edge["source"]]["data"]["column"],
            nodes[edge["source"]]["data"]["provider"],
            nodes[edge["source"]]["data"]["declared_in"],
            edge["data"]["rows"],
        )
        for edge in materialized_reads
    }

    assert materialized_claims
    assert actual_reads == expected_reads
    assert document["schemaVersion"] == "graph-explorer/v1"
    summary = document["metadata"]["microcosm"]
    assert summary["graph_sha256"] == schema["graph_sha256"]
    assert (
        summary["schema_sha256"] == hashlib.sha256(canonical_json(schema)).hexdigest()
    )
    assert summary["counts"]["operations"] == len(schema["graph"]["nodes"])
    assert summary["counts"]["fields"] == len(schema["fields"])
    assert "graph" not in summary
    assert document["metadata"]["truncated"] is False


def test_conserve_rejects_a_mass_shift_across_household_sizes() -> None:
    # The executor's ledger is person mass: an expansion that conserves the
    # weight entity's mass but shifts it between households of different size
    # cannot pass ``conserve``.  This is the class that refused the SPI support
    # channel on the licensed FRS 2024-25 spine (68.25m -> 65.44m persons).
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(
            _mixed_size_population(),
            _expand_node(),
            _mass_shifting_expand_result(declared=False),
        )


def test_declared_accepts_the_same_expansion_with_the_kernel_ledger() -> None:
    node = Node(
        id="stack",
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params=_expand_node().params,
        mass="declared",
    )
    expanded = patch(
        _mixed_size_population(),
        node,
        _mass_shifting_expand_result(declared=True),
    )

    assert expanded.frame.table("person")["person_household_id"].tolist() == [
        10,
        20,
        20,
        30,
    ]
    assert expanded.frame.weights_for("household").total == pytest.approx(3.0)
    record = expanded.mass_ledger[-1]
    assert record.policy == "declared"
    assert record.before_total == pytest.approx(5.0)
    assert record.after_total == pytest.approx(4.5)


def test_spi_support_channel_declares_its_mass_change_and_cgt_clones_conserve() -> None:
    graph = uk_spine_graph(load_country_spec("uk"))

    assert graph.node("spi_support_channel").mass == "declared"
    assert graph.node("cgt_support_split").mass == "conserve"
    assert graph.node("cgt_incidence_clone").mass == "conserve"
    assert graph.node("cgt_incidence_anchor").mass == "conserve"
    assert graph.node("cgt_incidence_anchor").params["expand_cells"] == ()
    assert "cgt_incidence_anchor.owned" not in {node.id for node in graph.nodes}


def test_zero_row_expand_conserves_mass_through_the_executor() -> None:
    """An EXPAND may add no rows and only move weight between incumbent rows."""

    expanded = patch(
        _expand_population(),
        _zero_row_expand_node(),
        _zero_row_expand_result([0.25, 2.75], total=3.0),
    )

    assert expanded.version == "anchor"
    assert expanded.frame.table("person")["person_id"].tolist() == [1, 2]
    assert expanded.frame.table("household")["household_id"].tolist() == [10, 20]
    weights = expanded.frame.weights_for("household")
    assert weights.kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(weights.values, np.array([0.25, 2.75]))
    assert weights.total == 3.0
    record = expanded.mass_ledger[-1]
    assert record.operation == "expand"
    assert record.before_total == 3.0
    assert record.after_total == 3.0
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0])
    )
    assert expanded.frame.mass_log[-1].declared_factor == 1.0


def test_zero_row_expand_still_refuses_a_shift_across_household_sizes() -> None:
    # Conservation stays on person mass: a zero-row EXPAND that moves weight
    # between households of different size changes the stratum ledger and is
    # rejected, so the anchor can only move mass within same-size pairs.
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(
            _mixed_size_population(),
            _zero_row_expand_node(),
            _zero_row_expand_result([0.5, 2.5], total=3.0),
        )


def test_property_components_reads_cover_the_stage_inputs() -> None:
    """A node receives its declared cells alone, so the declaration must name
    every person column the landlord stage requires (the recipient flag joined
    them with microcosm#840; the 13 October twin build failed at the node
    before the declaration followed)."""
    from microcosm.build.uk_runtime.graph import _STAGE_CONSUMES
    from microcosm.build.uk_runtime.property_components import (
        PROPERTY_COMPONENTS_PERSON_INPUTS,
    )

    declared = _STAGE_CONSUMES["property_components"]
    assert declared is not None
    required = {("person", column) for column in PROPERTY_COMPONENTS_PERSON_INPUTS}
    assert required <= declared, sorted(required - declared)


def test_spi_benefit_coherence_reads_cover_its_work_tests() -> None:
    """The contributory awards' work tests read the copy's earnings and its
    twin's hours, and the receipt reads its pension payments and counts by
    source household: the node gets only its declared cells (the 4 November
    b5j spine failed at the node before the hours followed)."""
    from microcosm.build.uk_runtime.graph import _STAGE_CONSUMES
    from microcosm.build.uk_runtime.spi_benefit_coherence import (
        SPI_PENSION_PAYMENTS_COLUMN,
        SPI_WITHIN_LIMIT_EARNINGS_COLUMNS,
        SPI_WITHIN_LIMIT_HOURS_COLUMN,
        SPI_WITHIN_LIMIT_REPORT_COLUMNS,
    )

    declared = _STAGE_CONSUMES["spi_benefit_coherence"]
    assert declared is not None
    required = {
        ("person", column)
        for column in (
            *SPI_WITHIN_LIMIT_REPORT_COLUMNS,
            *SPI_WITHIN_LIMIT_EARNINGS_COLUMNS,
            SPI_WITHIN_LIMIT_HOURS_COLUMN,
            SPI_PENSION_PAYMENTS_COLUMN,
        )
    } | {("household", "source_household_id")}
    assert required <= declared, sorted(required - declared)
