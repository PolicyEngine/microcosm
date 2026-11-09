"""Tests split from packages/microcosm-build/tests/test_uk_graph.py."""

# ruff: noqa: F403, F405
import functools
import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from microcosm.graph import graph_schema, orrery_document
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.first_party_imports import hashed_module_names
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
    # The root stage's node is the CREATE kernel.
    affected = [
        graph.node("create_uk_frs" if stage == "frs_spine" else stage).kernel
        for stage in sorted(graph_kernels._COUNTING_STAGES)
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
#: reader, id normaliser, interview dates and self-employment rule. They need
#: not bind the table; the helper rule below holds that none of them reaches
#: the table's code another way.
_COPY_ID_TABLE_NAME_ONLY_STAGES = frozenset(
    {
        "company_cars",
        "etb_services",
        "etb_vat",
        "frs_annual_income_spells",
        "frs_council_tax",
        "frs_disability",
        "frs_education",
        "frs_employment",
        "frs_legacy_proxies",
        "frs_relationships",
        "isa_subscriptions",
        "lcfs_consumption",
        "nts_bus_travel",
        "property_components",
        "spi_benefit_coherence",
        "uc_capital_coherence",
        "was_lisa",
        "was_wealth",
    }
)
#: The functions those stages may import from the two helpers.
_COPY_ID_TABLE_FREE_HELPER_FUNCTIONS = frozenset(
    {
        "fill_support_channel_from_source",
        "frs_interview_dates",
        "normalize_ids",
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
        for stage in ("frs_council_tax", "hmrc_cgt_gains_spine", "isa_subscriptions")
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
    assert len(affected) == 10
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref], ref
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref], ref


_STOCHASTIC_ASSIGNMENT = "microcosm.build.stochastic_assignment"
#: Stages that import a function drawing through the keyed-draw helpers
#: (``microcosm.build.stochastic_assignment``) but never call it, so need not
#: bind them: the stage's transform, the drawing names its module imports, and
#: why the transform never calls them. None at present: the support channel,
#: the one stage listed before, binds frs_take_up, which imports the helpers,
#: so it must list them too, and does (microcosm#931).
_STOCHASTIC_ASSIGNMENT_UNBOUND: dict[str, tuple[str, frozenset[str], str]] = {}


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
    # Stage 14's module imports the employer pension draw, which draws through
    # the helpers; the stage binds the draw's module as well.
    spi_spine = graph_kernels._stage_module("frs_hmrc_spine_leaves").__name__
    employer = "microcosm.build.uk_runtime.employer_pension_contributions"
    drawing = _drawing_names([spi_spine])[employer]
    assert "draw_employer_pension_contributions" in drawing
    assert "frs_hmrc_spine_leaves" in reach
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
    # Stage 14 draws through the employer pension helper, which it binds too.
    assert "frs_hmrc_spine_leaves" in stages
    # The root stage's node is the CREATE kernel.
    affected = [
        graph.node("create_uk_frs" if stage == "frs_spine" else stage).kernel
        for stage in stages
    ]
    # Three stages that draw through nothing of it.
    control_stages = (
        "regional_home_values",
        "hmrc_cgt_gains_spine",
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


@functools.cache
def _engine_use() -> tuple[frozenset[str], frozenset[str]]:
    """The stages whose module calls the engine, and those whose transform
    takes one."""
    from microcosm.build.uk_runtime import graph_kernels

    modules = {
        stage: graph_kernels._stage_module(stage)
        for stage in graph_kernels._STAGE_MODULES
    }
    return (
        frozenset(
            stage for stage, module in modules.items() if _calls_the_engine(module)
        ),
        frozenset(
            stage for stage, module in modules.items() if _takes_an_engine(module)
        ),
    )


def _engine_stages() -> frozenset[str]:
    """The stages that compute through the engine: those that call it or take
    one, less those that take it for its country only."""

    calling, taking = _engine_use()
    return calling | (taking - _ENGINE_TAKEN_FOR_ITS_COUNTRY_ONLY)


def test_every_stage_that_calls_the_engine_binds_the_engine_modules() -> None:
    """How an engine call runs, and clone by clone once a frame carries more
    than one clone, is the adapter's, the shared engine-block machinery's and
    the UK declaration's (microcosm#931). ``source_hash`` does not follow
    imports, so every stage whose module calls the engine, or whose transform
    takes one, binds all three, and no other stage binds any of them in the
    stage table. Each such stage also binds what else the injected engine
    computes through for it (``_ENGINE_COMPUTE_MODULES``, which the next test
    derives); another stage binds a module of that group only where its own
    transform computes through it. The one exception takes the engine only to
    check its country and computes nothing through it. Every stage also
    hashes the adapter beside national_frame, which stamps its frames with
    the adapter's ISA income contract and savings income projection
    (test_engine_module_changes_invalidate_every_stage_that_calls_the_engine)."""
    from microcosm.build.uk_runtime import graph_kernels

    engine_modules = set(graph_kernels._ENGINE_MODULES)
    compute_modules = set(graph_kernels._ENGINE_COMPUTE_MODULES)
    calling, taking = _engine_use()
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
    computing = _engine_stages()
    transform_walks = _transform_walks()
    for stage in graph_kernels._STAGE_MODULES:
        bound = set(graph_kernels._STAGE_HELPER_MODULES.get(stage, ()))
        if stage in computing:
            assert engine_modules | compute_modules <= bound, stage
        else:
            assert not engine_modules & bound, stage
            own = {
                module
                for module in compute_modules
                if module.__name__ in transform_walks[stage].reach
            }
            assert bound & compute_modules == own, stage
    # The SPI spine's three stages and the band donors bind frame.materialize
    # for their own tables.
    assert {
        stage
        for stage in set(graph_kernels._STAGE_MODULES) - computing
        if compute_modules & set(graph_kernels._STAGE_HELPER_MODULES.get(stage, ()))
    } == {
        "frs_hmrc_spine_leaves",
        "spi_support_channel",
        "hmrc_spi_income_spine",
        "spi_income_band_donors",
    }


def test_the_engine_stages_bind_what_the_injected_engine_computes_through() -> None:
    """Every stage that calls the engine computes through the
    PolicyEngineUKEngine the builds inject, whose code the walk from the
    stage's transform does not follow (microcosm#931 ruling 5). A second walk
    is rooted at that class (``_ENGINE_CLASS``): it follows the methods the
    stage's code names on it, by attribute or through ``getattr``, with the
    constructor and every method they call on ``self``, at the first-party
    scope with the stage rule's tables (``_engine_walk``). ``_stage_walks``
    adds it to the walk from the transform, so the rule's tests hold what it
    reaches like any other module the stage computes through. Every engine
    stage names ``materialize``, the RulesEngine protocol's call. Together the
    walks reach exactly the three engine modules and
    ``_ENGINE_COMPUTE_MODULES``: a module an engine method newly computes
    through fails here until it joins the group, which every engine stage
    binds (the test above), and so does one that leaves the group."""
    from microcosm.build.uk_runtime import graph_kernels

    roots = _engine_root_walks()
    assert set(roots) == _engine_stages()
    engine_class = _engine_class()
    transform_walks = _transform_walks()
    for stage in roots:
        methods = _engine_methods(engine_class, transform_walks[stage])
        assert "materialize" in methods, stage
    engine_modules = {module.__name__ for module in graph_kernels._ENGINE_MODULES}
    compute_modules = {
        module.__name__ for module in graph_kernels._ENGINE_COMPUTE_MODULES
    }
    assert not engine_modules & compute_modules
    reached = set().union(*(set(walk.reach) for walk in roots.values()))
    assert reached == engine_modules | compute_modules, sorted(
        reached ^ (engine_modules | compute_modules)
    )


def test_the_engine_root_is_the_engine_the_builds_inject() -> None:
    """The second root is the class of the engine both builders construct
    and hand the engine stages: graph_kernels' hermetic H2 builder and the
    spine build's rules engine (microcosm#931 ruling 5)."""
    import ast

    from microcosm.build.uk_runtime import graph_kernels, spine_build

    for module, function in (
        (graph_kernels.__name__, "_fixture_implementations"),
        (spine_build.__name__, "_rules_engine"),
    ):
        node = _module_definitions(module, _HELPER_SCOPE).definitions[function]
        built = {
            _resolve(module, sub.func.id, _HELPER_SCOPE)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        assert _ENGINE_CLASS in built, (module, function)


def test_the_engine_root_follows_the_methods_a_stage_calls(
    tmp_path, monkeypatch
) -> None:
    """Control for the second root: the methods a stage names on an engine,
    by attribute on any receiver or through ``getattr``, are followed, with
    the constructor and the methods they call on ``self``; a method no stage
    names is not, nor what only it computes through."""
    import importlib
    import uuid

    scratch = f"engine_root_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "tables": "def tables(frame):\n    return frame\n",
        "tree": "def tree(year):\n    return year\n",
        "vocabulary": "def vocabulary():\n    return {'a': 1}\n",
        "count": "def count(x):\n    return int(x)\n",
        "engine": (
            "from .count import count\n"
            "from .tables import tables\n"
            "from .tree import tree\n"
            "from .vocabulary import vocabulary\n\n"
            "class Engine:\n"
            "    def __init__(self, clones=1):\n"
            "        self._clones = count(clones)\n\n"
            "    def materialize(self, frame, names, period):\n"
            "        return self._dataset(frame)\n\n"
            "    def _dataset(self, frame):\n"
            "        return tables(frame)\n\n"
            "    def parameters(self, period):\n"
            "        return tree(period)\n\n"
            "    def concept_mapping(self):\n"
            "        return vocabulary()\n"
        ),
        "stage": (
            "class Transform:\n"
            "    def __call__(self, frame, model):\n"
            "        reader = getattr(model, 'parameters', None)\n"
            "        return model.materialize(frame, ['x'], 2024), reader\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    engine_class = importlib.import_module(f"{prefix}engine").Engine
    walk = _helper_walk((f"{prefix}stage", "Transform"), scope=prefix)
    methods = _engine_methods(engine_class, walk)
    assert methods == {"materialize", "parameters"}
    reach = _engine_walk(engine_class, methods, scope=prefix).reach
    assert {module.removeprefix(prefix): names for module, names in reach.items()} == {
        "engine": {"Engine"},
        "count": {"count"},
        "tables": {"tables"},
        "tree": {"tree"},
    }


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
    ("module_name", "every_stage"),
    [
        # national_frame stamps every frame through the adapter's ISA income
        # contract and savings income projection, so every stage hashes it.
        ("microcosm.frame.adapters.policyengine_uk", True),
        ("microcosm.frame.adapters.engine_population", False),
        ("microcosm.frame.adapters.policyengine_uk_population", False),
        # The adapter builds each simulation's dataset through it.
        ("microcosm.frame.materialize", False),
    ],
)
def test_engine_module_changes_invalidate_every_stage_that_calls_the_engine(
    monkeypatch, module_name, every_stage
):
    import importlib

    from microcosm.build.uk_runtime import graph_kernels

    module = importlib.import_module(module_name)
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
        assert (registry.implementation_hash(ref) != before[ref]) is every_stage, ref


# The rule behind the bindings above, for every helper (microcosm#931):
# ``source_hash`` does not follow imports, so a node's hash must name every
# module whose code can change what the node computes. The tests below
# derive those modules from what a stage's transform, or a UK graph kernel's
# own code, names, and follow each named definition into the module that
# defines it, through re-exports and module attributes, to a fixed point.

#: The packages the rule follows: every first-party package, each under the
#: microcosm namespace (microcosm-build, -calibrate, -data, -diagnostics,
#: -fit, -frame and -graph). It binds what the stages and kernels compute
#: through there as in microcosm.build: microcosm.fit's QRF, model,
#: categorical, donor and held-out modules; microcosm.frame's materialize and
#: rules modules and the engine adapter, which national_frame stamps every
#: frame through; microcosm.graph's keyed uniforms, canonical encoder,
#: weight-update receipt and content store; and microcosm.calibrate's solver,
#: targets and region tier. It stops at the core modules every node runs
#: through (``_CORE_MODULES``).
_HELPER_SCOPE = "microcosm."
#: Where the rule stops. What a stage reads from these is its own
#: declaration, which its node key already hashes as the stage contract: its
#: operations, artifacts and resource pins (graph._stage_contract_sha256).
_DECLARATION_MODULES = (
    "microcosm.build.country_spec",
    "microcosm.build.source_manifest",
    "microcosm.build.spec_engine",
)
#: The Frame's reason, which its three modules share.
_FRAME_REASON = (
    "the spine stages' transforms compute on it, every spine stage kernel "
    "builds one, and the executor holds every population version in one. No "
    "spine key hashes it, and H2 cannot catch an edit to it, since the "
    "graph's spine and the legacy stage plan both run through it. H1, whose "
    "fixtures run four wrapped kernels through run_graph against stored "
    "bytes, catches one only on those kernels' frames, so an edit that changes "
    "a UK value is otherwise left to the Frame's own tests; the totals the UK "
    "exact-total corrections target are pinned here "
    "(test_the_frame_totals_the_uk_corrections_target_are_numpys_sums)"
)
#: The core modules every node runs through, which the rule neither follows
#: nor asks a stage to bind, each with what runs through it and what catches
#: an edit to it that would change values. Binding one would re-key every
#: node on any edit to it. The Frame's are the only ones a stage's transform
#: names; the executor and the modules that compute node keys run every node
#: from outside it. microcosm.graph.canonical also encodes every key, but
#: regional_home_values draws through it, so that stage binds it instead.
#: test_the_core_modules_are_on_every_nodes_path checks the table.
_CORE_MODULES: dict[str, str] = {
    "microcosm.frame.bundle": f"the Frame: {_FRAME_REASON}",
    "microcosm.frame.schema": f"the Frame's entity schema: {_FRAME_REASON}",
    "microcosm.frame.weights": (
        f"the Frame's typed weights and mass records: {_FRAME_REASON}"
    ),
    "microcosm.graph.executor": (
        "runs every node (run_graph): keys it, projects its declared inputs, "
        "calls its kernel and applies its result. No spine key hashes it, so "
        "an edit that changes a value on H2's fixture fails H2, which compares "
        "every cell of the spine run through it with the legacy stage plan's, "
        "which does not run through it"
    ),
    "microcosm.graph.population": (
        "the executor applies every node's result to its population version "
        "through it (patch); as for the executor, H2 catches an edit that "
        "changes a value on its fixture"
    ),
    "microcosm.graph.kernel": (
        "every node's inputs and outputs pass through its KernelContext and "
        "KernelResult, where H2 catches a changed value as for the executor; "
        "every spine stage kernel's implementation hash is computed through "
        "its source_hash, so an edit that changes what that returns changes "
        "every key"
    ),
    "microcosm.graph.keys": (
        "computes every node's key and its artifacts' keys (node_key), so an "
        "edit that changes what it returns changes every key; no spine stage "
        "kernel reads the generator the executor seeds from a key (each "
        "declares SeedSource.NONE)"
    ),
    "microcosm.graph.decl": (
        "node_key hashes every node's normative declaration and resolves its "
        "inputs and population version through the compiled graph, as the "
        "executor does, so an edit that changes how a node is declared or "
        "what it reads changes its key"
    ),
}
#: The definitions of the executor's and the key modules that the UK graph
#: kernels' own code names, each with why naming it binds nothing the kernel's
#: key misses (test_the_core_definitions_the_kernels_name_are_listed). The
#: Frame's definitions, which kernels build and read frames with, are the
#: Frame's reason's; a stage's transform meets the Frame's modules alone
#: (test_the_core_modules_are_on_every_nodes_path).
_KERNEL_CORE_NAMES: dict[tuple[str, str], str] = {
    ("microcosm.graph.population", "dtype_for_token"): (
        "maps a node's declared dtype token to the dtype its owned cells are "
        "stored at. The executor stores every node's owned cells at the dtype "
        "it returns (patch, checked), so a kernel that casts its cells through "
        "it first hands the executor what the executor stores anyway, and an "
        "edit to it moves every node's stored dtypes alike: H2 catches one on "
        "the dtypes its spine fixture carries, and graph.population's own "
        "tests the rest"
    ),
    ("microcosm.graph.decl", "ArtifactType"): (
        "a frozen record of an artifact type's name and version, which a "
        "kernel names to declare or check its artifacts' types; node_key "
        "hashes each node's declared artifact types as part of its normative "
        "declaration"
    ),
    ("microcosm.graph.decl", "ArtifactOutput"): (
        "a frozen record pairing an artifact's name with its type, which the "
        "atomic geography gate names to declare and check its outputs; "
        "node_key hashes each node's declared artifact outputs"
    ),
    ("microcosm.graph.decl", "StructuralDelta"): (
        "the enum a kernel's capabilities declare its row-set change in; the "
        "executor refuses a kernel whose declared change differs from its "
        "node's, and node_key binds the declared value with the rest of the "
        "kernel's capabilities"
    ),
    ("microcosm.graph.kernel", "KernelRole"): (
        "the enum a kernel's capabilities declare its role in, by which the "
        "executor holds a gate's outcome to the gate outcomes and derives a "
        "release's tier from its gate ancestors; node_key binds the declared "
        "value with the rest of the kernel's capabilities"
    ),
    # The kernels name the entries below through microcosm.graph's lazy
    # exports, which the walk follows since microcosm#931 ruling 8.
    ("microcosm.graph.kernel", "Capabilities"): (
        "the frozen record of a kernel's declared contract, which a kernel "
        "names to declare its capabilities; it only checks the types and "
        "consistency of its fields, and node_key binds every field it holds "
        "(keys._capabilities_projection)"
    ),
    ("microcosm.graph.kernel", "Determinism"): (
        "the enum a kernel's capabilities declare its determinism in, whether "
        "its output is a function of its inputs and seed; node_key binds the "
        "declared value with the rest of the kernel's capabilities"
    ),
    ("microcosm.graph.kernel", "Numeric"): (
        "the enum a kernel's capabilities declare its numbers' "
        "reproducibility in, from which the executor resolves the numeric "
        "scope of each column a reader is given; node_key binds the declared "
        "value with the rest of the kernel's capabilities"
    ),
    ("microcosm.graph.kernel", "SeedSource"): (
        "the enum a kernel's capabilities declare its seed source in, where a "
        "seeded kernel takes its randomness from; node_key binds the declared "
        "value with the rest of the kernel's capabilities"
    ),
    ("microcosm.graph.kernel", "KernelContext"): (
        "the frozen record of what the executor hands a kernel to read, which "
        "a kernel names to annotate it; it only checks and freezes the fields "
        "the executor fills, and every node's inputs pass through it, where "
        "H2 catches a changed value (graph.kernel's core reason)"
    ),
    ("microcosm.graph.kernel", "KernelResult"): (
        "the frozen record a kernel returns its outputs in, which holds what "
        "the kernel puts in it and computes nothing; the executor validates "
        "every field and applies it through graph.population's patch, where "
        "H2 catches a changed value as for the executor"
    ),
}
#: Checks that read values only to decide whether to refuse, each with why it
#: cannot change what its caller computes. The rule does not follow them, and
#: a definition that names nothing else of value needs no binding for them;
#: test_the_refusal_only_helpers_return_and_write_nothing checks each one. A
#: kernel's method is named ``Class.method``; the kernel rule leaves it out of
#: what the kernel runs.
_REFUSAL_ONLY_HELPERS: dict[tuple[str, str], str] = {
    (
        "microcosm.build.uk_runtime.graph_evidence",
        "require_uk_spine_gate_admission",
    ): (
        "refuses a node unless the spine gates admitted its input; the dense "
        "sample node calls it before it computes anything"
    ),
    (
        "microcosm.build.uk_runtime.graph_terminal",
        "_require_full_gate_admission",
    ): (
        "decodes a full gate report and refuses, with its caller's message, "
        "unless the report permits the artifact; the export description and "
        "the rotated holdout call it before they compute anything"
    ),
    ("microcosm.build.uk_runtime.graph_calibration", "UKDenseSolveKernel._admit"): (
        "refuses the dense solve unless its declared admission holds: the "
        "source preflight's report permits the artifact, or the bound spine's "
        "provenance binds its gate report; the solve calls it before it "
        "computes anything"
    ),
    ("microcosm.build.uk_runtime.cgt_structure", "_assert_closed_world_operations"): (
        "compares a stage's declared operations with the reviewed mapping, in "
        "order and in full, and refuses on any difference; the stages' "
        "parameter checks call it"
    ),
    ("microcosm.build.uk_runtime.spi_support", "_require_entity"): (
        "returns its argument unchanged and refuses anything but person, "
        "benunit or household; the support channel's column names call it"
    ),
    ("microcosm.frame.rules", "assert_rules_engine_country"): (
        "reads the engine's declared country and refuses one that declares "
        "none or another; the stages that take an engine call it before "
        "computing"
    ),
}
#: Names a stage reads from a module that neither its node nor any node
#: upstream of it hashes, each with why an edit to them cannot leave the
#: stage's output stale. A name hashed upstream needs no entry: node keys
#: chain on their predecessors' keys, so an edit there re-keys the stage too.
#: Only a single string, a function returning one column name, or a record
#: can be listed (``_helper_kind``); the tests hold the table to exactly what
#: they find, so a new one is bound or listed here with its reason.
_STAGE_UNBOUND_NAMES: dict[tuple[str, str], str] = {
    ("microcosm.build.uk_runtime.rowwise_dataset", "ladder_clone_index_column"): (
        "names the geographic clone index columns, which only the dense "
        "expand writes; no spine stage writes them (checked), so a spine "
        "frame never carries them and their name moves no spine value"
    ),
    ("microcosm.frame.concepts", "SOURCE_HOUSEHOLD_ID_COLUMN"): (
        "names the source household column, CopyLineage's default source, "
        "which the support channel writes by its literal name. A node is "
        "given only its declared columns, so under a renamed constant the "
        "lineage readers would find no source column, which they refuse "
        "unless counting before the copies on a frame without a copy column. "
        "A spine stage that reads it declares the column by that name, which "
        "its key hashes, and counts before the copies only beside a copy "
        "column, or declares neither, and counts as before (checked)"
    ),
}
#: The same for the UK graph kernels: names a kernel reads from a module its
#: key does not hash, each with why.
_KERNEL_UNBOUND_NAMES: dict[tuple[str, str], str] = {
    ("microcosm.build.uk_runtime.geography_sources", name): (
        "the LAD23 names lookup's pinned URL and digest: local_authority_input "
        "refuses an input whose declared source differs from them, and reads "
        "nothing through them"
    )
    for name in ("LAD23_NAMES_SHA256", "LAD23_NAMES_URL")
}
#: Where the kernel rule stops, each with why a kernel that names the
#: definition need not hash what lies beyond it;
#: test_the_kernel_rule_stops_only_where_its_reasons_hold checks each.
_KERNEL_STOPS: dict[tuple[str, str], str] = {
    ("microcosm.build.uk_runtime.graph_kernels", "_FixtureTransformResolver"): (
        "resolves the hermetic H2 fixture's transform for the node's stage, "
        "which _fixture_implementations builds and which is the stage's own "
        "transform: the stage rule holds what each transform computes "
        "through, from where graph_kernels builds it, and what the "
        "PolicyEngineUKEngine it builds and injects into the engine stages "
        "computes through for them, from the methods they call on it (its "
        "second root, _ENGINE_CLASS; microcosm#931 ruling 5)"
    ),
    ("microcosm.graph.codecs", "load_source"): (
        "decodes a declared source into a frame. The create kernel calls it "
        "only where no root transform is bound, the hermetic registry's "
        "data-only path, which loads the captured root tables of the H2 "
        "fixture. The spine build installs the root's transform, and H2 "
        "parity binds it too (fixture_stage_plan_inputs), so neither takes "
        "that path (checked). Hashing microcosm.graph.codecs, and the content "
        "store it names, would re-key the spine root, and with it every spine "
        "node and every node downstream of the spine, on edits to code no "
        "build runs; microcosm-graph's codec tests hold what it decodes "
        "(microcosm#931 ruling 8)"
    ),
    ("microcosm.build.uk_runtime.spi_spine", "uk_spi_support_mass_change_reason"): (
        "_minimal_frame rebuilds the SPI support allocation's mass record "
        "through it for the hmrc_spi_income_spine stage alone, inside the "
        "branch on that stage's name, and that stage's node hashes everything "
        "it computes through"
    ),
    ("microcosm.build.uk_runtime.battery_bindings", "UK_GATE_REGISTRY"): (
        "maps each UK gate to its evaluator. The bound spine builds a battery "
        "run through it only for the checkpoint's gate digests, which read the "
        "manifest, and the run reads the registry only to refuse a parameter "
        "its binding does not declare. A kernel that evaluates gates through "
        "it is held (_HELD_KERNELS, _HELD_KERNEL_STOPS)"
    ),
}
#: Kernels the rule holds at one listed stop only, for a ruling
#: (microcosm#931), each with why. Past the stop the rule holds them as any
#: other kernel. test_every_hold_still_guards_a_module_its_kernel_leaves_out
#: checks that the walk past each stop still meets a module the kernel's hash
#: leaves out.
_HELD_KERNEL_STOPS: dict[tuple[str, tuple[str, str]], str] = {
    (
        "uk.full.national-gates@1",
        ("microcosm.build.uk_runtime.battery_bindings", "UK_GATE_REGISTRY"),
    ): (
        "evaluates the gates its manifest declares through the registry, and "
        "how a gate kernel's key binds what each evaluator computes through is "
        "held with the gate batteries (_HELD_KERNELS)"
    ),
}
#: Kernels the rule does not yet hold, each with why, all held for a ruling
#: (microcosm#931): the gate batteries, whose phases share one key but declare
#: different gates, and the target kernels, whose keys microcosm#1497's
#: import record holds. Until the ruling, a held kernel can replay a stale
#: output after an edit to a module it computes through but does not hash;
#: test_every_hold_still_guards_a_module_its_kernel_leaves_out checks that
#: each still does, so a hold cannot outlive its need.
_TARGET_RECORD_REASON = (
    "test_uk_target_compile_modules holds this kernel's key to "
    "microcosm#1497's import record, which lists each import of a module the "
    "kernel hashes that it leaves unhashed and may only shrink. Binding what "
    "this rule finds it computing through would add to the record every such "
    "import of each module bound"
)
_HELD_KERNELS: dict[str, str] = {
    "uk.spine-gates@1": (
        "the assembled and transferred spine phases share one key, and only "
        "the transferred phase declares the gates that check the CGT asset "
        "type domain and most of the stage-health checks. Binding what those "
        "compute through into the shared key would re-key the assembled gate, "
        "which every later spine stage waits on, as the re-review of "
        "microcosm#1420 (E) kept the energy level out of it"
    ),
    "uk.full-gates.preflight@1": (
        "runs the calibrated battery's code, one class body and one code "
        "hash, for the preflight phase only. The dense solve keys on it, so "
        "binding the terminal phase's modules into the shared hash would "
        "re-solve on any terminal-gate edit, which microcosm#1301 and #1306 "
        "keyed out"
    ),
    "uk.full-gates@1": (
        "shares its code hash with the preflight, and its terminal-only "
        "publication code keys it through its terminal_code parameter "
        "(microcosm#1306)"
    ),
    **{
        ref: _TARGET_RECORD_REASON
        for ref in (
            "uk.full.target_compilation@1",
            "uk.full.target_selection@1",
            "uk.full.measures@1",
            "uk.full.problem@1",
            "uk.full.national_targets@1",
            "uk.full.national_problem@1",
        )
    },
}
#: Packages that export names lazily, through a module ``__getattr__``, each
#: with the table, of name to the submodule defining it, that ``__getattr__``
#: reads. The walk resolves such a name to the module that defines it and
#: follows it like any other (microcosm#931 ruling 8). microcosm.graph imports
#: its runtime modules, which need numpy, pandas and microcosm-frame, only on
#: first access. test_every_lazy_export_resolves_to_the_object_python_returns
#: checks the table.
_LAZY_EXPORT_TABLES: dict[str, str] = {"microcosm.graph": "_RUNTIME_EXPORTS"}
#: First-party modules that define a module ``__getattr__`` but export
#: nothing through it, each with why.
_NO_LAZY_EXPORTS: dict[str, str] = {
    "microcosm.build.us_runtime.sources": (
        "its __getattr__ raises for every name: it only words the error for "
        "the names it removed"
    ),
}
#: Method and function names that change their receiver or an argument in
#: place.
_IN_PLACE_CALLS = frozenset(
    {
        "__delitem__",
        "__setattr__",
        "__setitem__",
        "add",
        "append",
        "appendleft",
        "at",
        "clear",
        "copyto",
        "difference_update",
        "discard",
        "extend",
        "extendleft",
        "fill",
        "fill_diagonal",
        "insert",
        "intersection_update",
        "itemset",
        "place",
        "pop",
        "popitem",
        "put",
        "putmask",
        "remove",
        "resize",
        "reverse",
        "rotate",
        "setdefault",
        "setflags",
        "shuffle",
        "sort",
        "symmetric_difference_update",
        "update",
    }
)


class _ModuleDefinitions(NamedTuple):
    """What one module imports, anywhere in it, and defines at its top level."""

    imports: dict[str, tuple[str, str]]
    modules: dict[str, str]
    stars: tuple[str, ...]
    definitions: dict[str, object]


@functools.cache
def _is_module(name: str) -> bool:
    """Whether a first-party dotted name is a module. The rule never follows
    another package, so it never imports one to find out."""

    import importlib.util

    if not name.startswith("microcosm."):
        return False
    parent = name.rpartition(".")[0]
    try:
        if parent:
            spec = importlib.util.find_spec(parent)
            if spec is None or spec.submodule_search_locations is None:
                return False
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def _module_definitions(name: str, scope: str) -> _ModuleDefinitions | None:
    import ast
    import importlib.util

    if not name.startswith(scope):
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or "").endswith(".py"):
        return None
    package = name if spec.submodule_search_locations else name.rpartition(".")[0]
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imports: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    stars: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                origin = ".".join(part for part in (base, node.module) if part)
            else:
                origin = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    stars.append(origin)
                elif _is_module(f"{origin}.{alias.name}"):
                    modules[alias.asname or alias.name] = f"{origin}.{alias.name}"
                else:
                    imports[alias.asname or alias.name] = (origin, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    head = alias.name.partition(".")[0]
                    modules[head] = head
    definitions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions[node.name] = node
        elif isinstance(node, ast.TypeAlias):
            definitions[node.name.id] = node
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        definitions[sub.id] = node
    return _ModuleDefinitions(imports, modules, tuple(stars), definitions)


@functools.cache
def _lazy_exports(module: str) -> Mapping[str, str]:
    """Each name ``module`` exports lazily, to the module defining it, read
    from the table its ``__getattr__`` reads (``_LAZY_EXPORT_TABLES``)."""

    import importlib

    table = _LAZY_EXPORT_TABLES.get(module)
    if table is None:
        return MappingProxyType({})
    exports = getattr(importlib.import_module(module), table)
    return MappingProxyType(
        {name: f"{module}.{submodule}" for name, submodule in exports.items()}
    )


def _resolve(module: str, name: str, scope: str) -> tuple[str, str] | None:
    """The top-level definition ``name`` refers to in ``module``, through
    re-exports and lazy exports (``_LAZY_EXPORT_TABLES``); None for a module
    or anything outside ``scope``."""

    pending, seen = [(module, name)], set()
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        module, name = item
        facts = _module_definitions(module, scope)
        if facts is None:
            continue
        if name in facts.definitions:
            return item
        if name in facts.imports:
            pending.append(facts.imports[name])
            continue
        # As in Python, the module's __getattr__ supplies a name only where
        # none of its own attributes, its star imports' included, is one.
        lazy = _lazy_exports(module).get(name)
        if lazy is not None:
            pending.append((lazy, name))
        pending.extend((star, name) for star in facts.stars)
    return None


def _named(module: str, node, scope: str) -> set[tuple[str, str]]:
    """The top-level definitions ``node`` names, annotations included: a bare
    name, or a name read by attribute on a module, each resolved through
    re-exports."""

    import ast

    facts = _module_definitions(module, scope)
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            hit = _resolve(module, sub.id, scope)
        elif isinstance(sub, ast.Attribute):
            chain = []
            value = sub
            while isinstance(value, ast.Attribute):
                chain.insert(0, value.attr)
                value = value.value
            target = (
                facts.modules.get(value.id) if isinstance(value, ast.Name) else None
            )
            while target and chain and _is_module(f"{target}.{chain[0]}"):
                target = f"{target}.{chain.pop(0)}"
            hit = _resolve(target, chain[0], scope) if target and chain else None
        else:
            continue
        if hit is not None:
            found.add(hit)
    return found


def _writes_nothing(node) -> bool:
    """Whether a function can only refuse: it returns nothing but None, or its
    sole parameter unrebound, and writes nothing: no store through a
    subscript or attribute, no augmented assignment, no ``out=`` or
    ``inplace=`` argument, and no call by a name that changes its receiver or
    an argument in place."""

    import ast

    arguments = node.args
    parameters = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
        )
    }
    rebound = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del)
    }
    returnable = parameters - rebound if len(parameters) == 1 else set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            value = sub.value
            if not (
                (isinstance(value, ast.Constant) and value.value is None)
                or (isinstance(value, ast.Name) and value.id in returnable)
            ):
                return False
        elif isinstance(
            sub, ast.AugAssign | ast.Yield | ast.YieldFrom | ast.Global | ast.Nonlocal
        ):
            return False
        elif isinstance(sub, ast.Assign | ast.AnnAssign | ast.Delete):
            targets = (
                sub.targets
                if isinstance(sub, ast.Assign | ast.Delete)
                else [sub.target]
            )
            if any(
                isinstance(part, ast.Subscript | ast.Attribute)
                for target in targets
                for part in ast.walk(target)
            ):
                return False
        elif isinstance(sub, ast.keyword) and sub.arg in {"inplace", "out"}:
            return False
        elif isinstance(sub, ast.Call):
            function = sub.func
            if isinstance(function, ast.Attribute) and function.attr in _IN_PLACE_CALLS:
                return False
            if isinstance(function, ast.Name) and function.id in {
                *_IN_PLACE_CALLS,
                "delattr",
                "eval",
                "exec",
                "setattr",
            }:
                return False
    return True


def _string_expression(node) -> bool:
    """A string literal, an f-string of names or of one call on names, or a
    concatenation of them."""

    import ast

    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_expression(node.left) and _string_expression(node.right)
    if not isinstance(node, ast.JoinedStr):
        return False
    for part in node.values:
        value = part.value if isinstance(part, ast.FormattedValue) else part
        if isinstance(value, ast.Call):
            if not isinstance(value.func, ast.Name) or value.keywords:
                return False
            value = value.args[0] if len(value.args) == 1 else None
        if not isinstance(value, ast.Name | ast.Constant):
            return False
    return True


_HELPER_KINDS: dict[tuple[str, str, str], str | None] = {}


def _helper_kind(module: str, name: str, scope: str) -> str:
    """How a top-level definition can bear on a value.

    ``value`` unless it is a listed refusal-only ``check``
    (``_REFUSAL_ONLY_HELPERS``) or provably one of: a ``name`` (one string: a
    literal, an f-string or concatenation of names, or another name of these
    kinds); a ``column name`` (a function returning one such string built
    from its arguments); a ``record`` (a class of undefaulted fields whose
    types are of these kinds, no methods). A collection of strings is a
    value, since it selects rows, columns or codes, as is a number. A cycle
    reads as ``value``.
    """

    if (module, name) in _REFUSAL_ONLY_HELPERS:
        return "check"
    key = (module, name, scope)
    if key in _HELPER_KINDS:
        return _HELPER_KINDS[key] or "value"
    _HELPER_KINDS[key] = None
    _HELPER_KINDS[key] = _classify_helper(module, name, scope)
    return _HELPER_KINDS[key]


def _names_only_harmless(module: str, node, scope: str) -> bool:
    return all(
        found[0].startswith(_DECLARATION_MODULES)
        or _helper_kind(*found, scope) != "value"
        for found in _named(module, node, scope)
    )


def _classify_helper(module: str, name: str, scope: str) -> str:
    import ast

    node = _module_definitions(module, scope).definitions[name]
    # A type alias statement (``type X = ...``) reads as an assignment of its
    # value.
    if isinstance(node, ast.Assign | ast.AnnAssign | ast.TypeAlias):
        if node.value is None or (
            (
                isinstance(node.value, ast.Name | ast.Attribute)
                or _string_expression(node.value)
            )
            and _names_only_harmless(module, node.value, scope)
        ):
            return "name"
        return "value"
    if isinstance(node, ast.ClassDef):
        fields = [item for item in node.body if isinstance(item, ast.AnnAssign)]
        fields_only = all(
            (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
            or isinstance(item, ast.Pass)
            or (isinstance(item, ast.AnnAssign) and item.value is None)
            for item in node.body
        )
        if fields_only and all(
            _names_only_harmless(module, item, scope)
            for item in (
                *node.bases,
                *node.keywords,
                *node.decorator_list,
                *(field.annotation for field in fields),
            )
        ):
            return "record"
        return "value"
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    if (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and _string_expression(body[0].value)
        and _names_only_harmless(module, body[0].value, scope)
    ):
        return "column name"
    return "value"


class _Walk(NamedTuple):
    """What a root can compute through (``_helper_walk``)."""

    reach: dict[str, frozenset[str]]
    stops: frozenset[tuple[str, str]]
    core: frozenset[str]


def _helper_walk(
    root: tuple[str, str], scope: str = _HELPER_SCOPE, *, follow_core: bool = False
) -> _Walk:
    """What ``root`` can compute through, where the rule stops, and the core
    modules it meets.

    ``reach`` maps every module holding a definition that can change what
    ``root`` computes to those definitions: ``root`` itself, and each
    ``value`` definition it names, and they name, to a fixed point. ``stops``
    is every ``name``, ``column name`` or ``record`` those definitions name,
    which the rule does not follow. ``core`` is every listed core module
    (``_CORE_MODULES``) whose definitions they name, which the rule neither
    follows nor returns in ``reach``, unless ``follow_core``. The declaration
    modules and the listed refusal-only checks are neither followed nor
    returned."""

    module, name = root
    node = _module_definitions(module, scope).definitions[name]
    walk, _, _ = _walk(
        {root},
        [(found, root) for found in _named(module, node, scope)],
        scope,
        follow_core=follow_core,
    )
    return walk


def _walk(
    roots: set[tuple[str, str]],
    seeds: list[tuple[tuple[str, str], object]],
    scope: str,
    *,
    follow_core: bool = False,
    halts: Mapping[tuple[str, str], str] = MappingProxyType({}),
) -> tuple[_Walk, dict[tuple[str, str], frozenset]]:
    """The walk behind ``_helper_walk`` and ``_kernel_walk``: ``roots`` are
    reached as given, and each ``(definition, namer)`` seed is followed as
    ``_helper_walk`` follows what its root names. A definition in ``halts`` is
    neither followed nor returned; the second result maps each one met to
    what named it, and the third holds every core module's definition met."""

    reached: dict[str, set[str]] = {}
    for module, name in roots:
        reached.setdefault(module, set()).add(name)
    stops: set[tuple[str, str]] = set()
    core: set[str] = set()
    core_names: set[tuple[str, str]] = set()
    met: dict[tuple[str, str], set] = {}
    pending, seen = list(seeds), set(roots)
    while pending:
        item, namer = pending.pop()
        if item in halts:
            met.setdefault(item, set()).add(namer)
            continue
        if item in seen:
            continue
        seen.add(item)
        module, name = item
        if module.startswith(_DECLARATION_MODULES):
            continue
        if module in _CORE_MODULES:
            core.add(module)
            core_names.add(item)
            if not follow_core:
                continue
        kind = _helper_kind(module, name, scope)
        if kind != "value":
            if kind != "check":
                stops.add(item)
            continue
        reached.setdefault(module, set()).add(name)
        node = _module_definitions(module, scope).definitions[name]
        pending.extend((found, item) for found in _named(module, node, scope))
    walk = _Walk(
        {module: frozenset(names) for module, names in reached.items()},
        frozenset(stops),
        frozenset(core),
    )
    return (
        walk,
        {item: frozenset(namers) for item, namers in met.items()},
        frozenset(core_names),
    )


def _helper_reach(
    root: tuple[str, str], scope: str = _HELPER_SCOPE
) -> dict[str, frozenset[str]]:
    """Every module holding a definition that can change what ``root``
    computes, with those definitions (``_helper_walk``)."""

    return _helper_walk(root, scope).reach


#: The class of the engine the builds inject into the stages that compute
#: through the engine, the second root of their walk (microcosm#931 ruling 5):
#: graph_kernels' hermetic H2 builder and the spine build both construct it
#: (test_the_engine_root_is_the_engine_the_builds_inject).
_ENGINE_CLASS = ("microcosm.frame.adapters.policyengine_uk", "PolicyEngineUKEngine")


def _engine_class() -> type:
    import importlib

    module, name = _ENGINE_CLASS
    return getattr(importlib.import_module(module), name)


def _class_attributes(cls: type) -> dict[str, list[tuple[type, object]]]:
    """Each attribute ``cls`` defines, with the class and statement that
    define it, as ``_kernel_body`` chooses them by method resolution order."""

    body: dict[str, list[tuple[type, object]]] = {}
    for owner, statement, name in _kernel_body(cls):
        body.setdefault(name, []).append((owner, statement))
    return body


def _engine_methods(engine_class: type, walk: _Walk) -> frozenset[str]:
    """The attributes of ``engine_class`` that the code ``walk`` reaches
    names: read by attribute, on any receiver, or through ``getattr`` or
    ``hasattr`` with a constant name. It matches by name alone, as
    ``_calls_the_engine`` matches its calls, so it sees the engine under any
    name a stage holds it by, and can only over-count."""

    import ast

    found: set[object] = set()
    for module, names in walk.reach.items():
        definitions = _module_definitions(module, _HELPER_SCOPE).definitions
        for name in names:
            for sub in ast.walk(definitions[name]):
                if isinstance(sub, ast.Attribute):
                    found.add(sub.attr)
                elif (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in {"getattr", "hasattr"}
                    and len(sub.args) > 1
                    and isinstance(sub.args[1], ast.Constant)
                ):
                    found.add(sub.args[1].value)
    return frozenset(found & set(_class_attributes(engine_class)))


def _engine_walk(
    engine_class: type, methods: frozenset[str], scope: str = _HELPER_SCOPE
) -> _Walk:
    """What an engine of ``engine_class`` computes through for a stage that
    calls ``methods`` on it: the stage rule's walk, at ``scope`` with the
    same tables, from those methods, the constructor the builds call, and
    every method they call on ``self``, to a fixed point."""

    import ast

    body = _class_attributes(engine_class)
    pending, chosen = sorted({"__init__", *methods} & set(body)), set()
    while pending:
        name = pending.pop()
        if name in chosen:
            continue
        chosen.add(name)
        for _, statement in body[name]:
            pending.extend(
                sub.attr
                for sub in ast.walk(statement)
                if isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in {"self", "cls"}
                and sub.attr in body
            )
    seeds = [
        (found, (owner.__module__, f"{owner.__name__}.{name}"))
        for name in sorted(chosen)
        for owner, statement in body[name]
        for found in _named(owner.__module__, statement, scope)
    ]
    walk, _, _ = _walk(
        {(cls.__module__, cls.__name__) for cls in _kernel_classes(engine_class)},
        seeds,
        scope,
    )
    return walk


def _merged(first: _Walk, second: _Walk) -> _Walk:
    reach = dict(first.reach)
    for module, names in second.reach.items():
        reach[module] = reach.get(module, frozenset()) | names
    return _Walk(reach, first.stops | second.stops, first.core | second.core)


def _stage_transforms() -> dict[str, tuple[str, str]]:
    """Each spine stage's transform, as graph_kernels builds it for the
    hermetic registry and the legacy plan: ``(module, name)``."""

    import ast

    from microcosm.build.uk_runtime import graph_kernels

    tree = ast.parse(Path(graph_kernels.__file__).read_text(encoding="utf-8"))
    package = graph_kernels.__name__.rpartition(".")[0]
    transforms = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in {
            "_fixture_implementations",
            "fixture_stage_plan_inputs",
        }:
            continue
        local = {
            alias.asname or alias.name: (f"{package}.{sub.module}", alias.name)
            for sub in ast.walk(function)
            if isinstance(sub, ast.ImportFrom) and sub.level == 1
            for alias in sub.names
        }
        built = []
        for sub in ast.walk(function):
            if isinstance(sub, ast.Dict):
                built.extend(zip(sub.keys, sub.values, strict=True))
            elif isinstance(sub, ast.Assign) and isinstance(
                sub.targets[0], ast.Subscript
            ):
                built.append((sub.targets[0].slice, sub.value))
        for key, value in built:
            if (
                isinstance(key, ast.Constant)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in local
            ):
                transforms[key.value] = _resolve(*local[value.func.id], _HELPER_SCOPE)
    return transforms


@functools.cache
def _transform_walks() -> MappingProxyType:
    """The walk from each spine stage's transform (``_helper_walk``)."""

    from microcosm.build.uk_runtime import graph_kernels

    walks = {}
    for stage, transform in sorted(_stage_transforms().items()):
        assert transform[0] == graph_kernels._stage_module(stage).__name__, stage
        walks[stage] = _helper_walk(transform)
    return MappingProxyType(walks)


@functools.cache
def _engine_root_walks() -> MappingProxyType:
    """Each engine stage's second walk (``_engine_walk``): from the methods
    of the injected engine its transform's walk names."""

    engine_class = _engine_class()
    walks = _transform_walks()
    return MappingProxyType(
        {
            stage: _engine_walk(
                engine_class, _engine_methods(engine_class, walks[stage])
            )
            for stage in sorted(_engine_stages())
        }
    )


@functools.cache
def _stage_walks() -> dict[str, tuple[str, frozenset[str], tuple]]:
    """Each spine stage's node id, the modules its node hashes, and what it
    computes through: the walk from its transform and, for a stage that
    calls the engine, the walk from the engine methods it calls
    (``_engine_root_walks``)."""

    from microcosm.build.uk_runtime import graph_kernels

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    roots = _engine_root_walks()
    walks = {}
    for stage, walk in _transform_walks().items():
        # The root stage's node is the CREATE kernel.
        node = graph.node("create_uk_frs" if stage == "frs_spine" else stage)
        hashed = frozenset(
            hashed_module_names(registry.get(node.kernel), graph_kernels)
        )
        if stage in roots:
            walk = _merged(walk, roots[stage])
        walks[stage] = (node.id, hashed, walk)
    return walks


def test_every_stage_hashes_each_helper_its_transform_computes_through() -> None:
    """A stage's hash names every module holding code its transform can run
    to compute a value, so an edit to any of them re-keys the stage instead
    of replaying a stale output (microcosm#931).

    The rule starts at the transform graph_kernels builds for the stage and
    follows every definition it names, annotations included (an injected
    contract or table is called through them), into the module defining it,
    in any first-party package. It does not follow a listed refusal-only
    check, a stage declaration or a listed core module every node runs
    through, and stops at a single string, a column-name function or a record
    (see ``_helper_kind``), whose module the next test accounts for. Anything
    else is a value, a collection of strings and a number included, and
    needs its module bound. The rule starts at the transform, not at the
    kernel that runs it: the kernel's code, in graph_kernels, which every
    stage hashes, builds every stage's transform for the hermetic registry,
    so following it would bind every helper into every stage. For a stage
    that calls the engine it also starts at the methods the stage calls on
    the engine the builds inject (``_engine_root_walks``).

    What the rule cannot see: a call through any other object whose class
    the code never names, a name looked up by a computed string, and data
    files read by path.
    """
    from microcosm.build.uk_runtime import graph_kernels

    walks = _stage_walks()
    assert set(walks) == set(graph_kernels._STAGE_MODULES)
    unhashed = {
        stage: sorted(set(walk.reach) - hashed)
        for stage, (_, hashed, walk) in walks.items()
        if set(walk.reach) - hashed
    }
    assert not unhashed, unhashed


def test_every_name_a_stage_reads_is_hashed_with_it_upstream_or_listed() -> None:
    """Where the rule stops at a name, the name's module is hashed by the
    stage's node or by a node upstream of it, whose key the stage's key
    chains on, or the name is listed in ``_STAGE_UNBOUND_NAMES`` with its
    reason; the table lists nothing else (microcosm#931)."""
    from microcosm.build.uk_runtime import graph_kernels, rowwise_dataset
    from microcosm.graph.decl import compile_graph

    graph = uk_spine_graph()
    compiled = compile_graph(graph)
    walks = _stage_walks()
    hashed_by_node = {node_id: hashed for node_id, hashed, _ in walks.values()}
    registry = uk_registry(graph=graph)
    for node in graph.nodes:
        if node.id not in hashed_by_node:
            hashed_by_node[node.id] = frozenset(
                hashed_module_names(registry.get(node.kernel), graph_kernels)
            )

    def upstream(node_id: str) -> set[str]:
        seen, pending = set(), list(compiled.predecessors[node_id])
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item)
                pending.extend(compiled.predecessors[item])
        return seen

    unlisted: dict[tuple[str, str], list[str]] = {}
    for stage, (node_id, hashed, walk) in walks.items():
        covered = set(hashed).union(
            *(hashed_by_node[item] for item in upstream(node_id))
        )
        for module, name in walk.stops:
            if module not in covered:
                unlisted.setdefault((module, name), []).append(stage)
    assert set(unlisted) == set(_STAGE_UNBOUND_NAMES), sorted(
        set(unlisted) ^ set(_STAGE_UNBOUND_NAMES)
    )
    # The geographic clone index columns' reason: no spine node writes them.
    written = {
        (owned.entity, owned.column) for node in graph.nodes for owned in node.outputs
    }
    for entity in ("person", "benunit", "household"):
        assert (
            entity,
            rowwise_dataset.ladder_clone_index_column(entity),
        ) not in written
    # The source household column's reason. The support channel writes it by
    # its literal name; a stage that reads it declares it by that name and
    # counts before the copies only beside a copy column, or declares neither
    # it nor a copy column.
    from microcosm.build.uk_runtime import spi_support
    from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
    from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN

    assert f'"{SOURCE_HOUSEHOLD_ID_COLUMN}"' in Path(spi_support.__file__).read_text()
    copy_columns = set(UK_COPY_LINEAGE.copy_columns)
    readers = unlisted[("microcosm.frame.concepts", "SOURCE_HOUSEHOLD_ID_COLUMN")]
    counting = {stage: _counts_before_copies(walks[stage][2]) for stage in readers}
    assert any(counting.values())
    for stage in readers:
        read = {
            column
            for slice_ in graph.node(walks[stage][0]).inputs
            if slice_.entity == "household"
            for column in slice_.columns
        }
        if SOURCE_HOUSEHOLD_ID_COLUMN not in read:
            assert not read & copy_columns, stage
        elif not read & copy_columns:
            assert not counting[stage], (stage, sorted(counting[stage]))


def _counts_before_copies(walk: _Walk) -> set[tuple[str, str]]:
    """The definitions a walk reaches that count sources before the copies:
    each passes ``before_copies=True`` to a lineage reader, which then counts
    a household as its own source where the frame carries neither the source
    column nor a copy column."""

    import ast

    found = set()
    for module, names in walk.reach.items():
        definitions = _module_definitions(module, _HELPER_SCOPE).definitions
        for name in names:
            if any(
                isinstance(sub, ast.keyword)
                and sub.arg == "before_copies"
                and isinstance(sub.value, ast.Constant)
                and sub.value.value is True
                for sub in ast.walk(definitions[name])
            ):
                found.add((module, name))
    return found


def test_the_core_modules_are_on_every_nodes_path() -> None:
    """Each listed core module is on every node's path, as its reason says,
    and is no stage's own helper (microcosm#931). The walk from run_graph,
    which keys, runs and patches every node, meets each. The core modules a
    stage's transform meets are the Frame's, all three or none, so the table
    exempts no module that only some stages compute through; no stage's
    transform meets the executor or a key module, whose reasons rest on their
    running every node from outside it. The key module's reason holds too:
    every spine stage kernel declares SeedSource.NONE."""
    from microcosm.graph import SeedSource

    executor = _helper_walk(("microcosm.graph.executor", "run_graph"), follow_core=True)
    on_path = set(executor.reach) | set(executor.core)
    assert set(_CORE_MODULES) <= on_path, sorted(set(_CORE_MODULES) - on_path)
    frame = {
        module for module in _CORE_MODULES if module.startswith("microcosm.frame.")
    }
    walks = _stage_walks()
    met = {stage: walk.core for stage, (_, _, walk) in walks.items()}
    assert set().union(*met.values()) == frame
    uneven = {
        stage: sorted(core) for stage, core in met.items() if core and core != frame
    }
    assert not uneven, uneven
    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    for stage, (node_id, _, _) in walks.items():
        kernel = registry.get(graph.node(node_id).kernel)
        # regional_home_values draws its keyed uniforms through
        # microcosm.graph.randomness, whose contract asks a kernel drawing
        # through it to declare SeedSource.KEYED. Every spine stage kernel,
        # its own included, declares NONE through the one stage kernel class,
        # as María ruled on 9 Nov (microcosm#931 ruling 7): its stream is a
        # stage parameter, which its contract hash covers, and it reads no
        # generator the executor seeds.
        assert kernel.capabilities.seed_source is SeedSource.NONE, stage


def test_the_refusal_only_helpers_return_and_write_nothing() -> None:
    import ast

    for module, name in _REFUSAL_ONLY_HELPERS:
        owner, _, method = name.rpartition(".")
        node = _module_definitions(module, _HELPER_SCOPE).definitions[owner or name]
        if owner:
            (node,) = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == method
            ]
        assert _writes_nothing(node), (module, name)


def test_the_refusal_check_sees_each_way_a_function_writes() -> None:
    """Control for ``_writes_nothing``: a function that only refuses, or
    returns its sole argument, passes; one that writes through any of these
    shapes, or returns one of several arguments, does not."""
    import ast

    refusing = {
        "raise": "def f(frame):\n    if frame is None:\n        raise ValueError('x')\n",
        "identity": (
            "def f(entity):\n"
            "    if entity not in {'person'}:\n"
            "        raise ValueError('x')\n"
            "    return entity\n"
        ),
    }
    writing = {
        "copyto": "def f(values, mask):\n    np.copyto(values, 0.0, where=mask)\n",
        "out": "def f(values):\n    np.clip(values, 0, 1, out=values)\n",
        "add_at": "def f(w, roots, copies):\n    np.add.at(w, roots, w[copies])\n",
        "shuffle": "def f(rng, values):\n    rng.shuffle(values)\n",
        "reverse": "def f(items):\n    items.reverse()\n",
        "augmented": "def f(values):\n    values *= 2.0\n",
        "subscript": "def f(values):\n    values[0] = 1.0\n",
        "inplace": "def f(table):\n    table.drop(columns=['a'], inplace=True)\n",
        "pick": "def f(left, right):\n    return left\n",
        "computed": "def f(values):\n    return values + 1\n",
    }
    for label, source in {**refusing, **writing}.items():
        function = ast.parse(source).body[0]
        assert _writes_nothing(function) is (label in refusing), label


def test_the_helper_rule_follows_each_way_a_stage_names_a_helper(
    tmp_path, monkeypatch
) -> None:
    """Control: a helper called by name, through a re-export, a lazy export,
    by attribute on its module, through an alias or a computed constant, or
    held only in an annotation (a record's field type included) is reached,
    as are a numeric constant, a collection of strings and a function that
    writes in place or only refuses but is not listed; a listed check, a
    single string, a column name, a record of plain fields, a listed core
    module, what only a core module names, and a helper the root never names
    are not. The walk returns the names it stopped at and the core modules it
    met, and follows a core module only when asked to."""
    import uuid

    scratch = f"helper_rule_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    # A package exporting its runtime module's names lazily, as
    # microcosm.graph does.
    (package / "lazily").mkdir()
    (package / "lazily" / "__init__.py").write_text(
        "from importlib import import_module\n\n"
        "_EXPORTS = {'deferred': 'runtime', 'idle': 'runtime'}\n\n"
        "def __getattr__(name):\n"
        "    if name not in _EXPORTS:\n"
        "        raise AttributeError(name)\n"
        "    return getattr(import_module(f'{__name__}.{_EXPORTS[name]}'), name)\n"
    )
    (package / "lazily" / "runtime.py").write_text(
        "def deferred(x):\n    return x * 6\n\ndef idle(x):\n    return x * 7\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "direct": "def draw(x):\n    return x + 1\n",
        "origin": "def total(x):\n    return sum(x)\n",
        "reexport": "from .origin import total\n",
        "attribute": "def share(x):\n    return x / 2\n",
        "alias": "def _scale(x):\n    return x * 3\n\nscale = _scale\n",
        "computed": "def _build():\n    return {'a': 1}\n\nTABLE = _build()\n",
        "collaborator": (
            "class Contract:\n    def rate(self, key):\n        return 0.5\n"
        ),
        "fielded": (
            "from typing import NamedTuple\n\n"
            "class Rate:\n    def value(self):\n        return 0.1\n\n"
            "class Inputs(NamedTuple):\n    rate: Rate\n"
        ),
        "numeric": "FLOOR = 10\n",
        "codes": (
            "CODES = frozenset({'A', 'B'})\n"
            "OWNER = {'OWNED': 'owner', 'RENTED': 'renter'}\n"
        ),
        "inplace": (
            "import numpy as np\n\n"
            "def fold(weights, roots, copies):\n"
            "    np.add.at(weights, roots, weights[copies])\n"
        ),
        "guard": (
            "def guard(frame):\n"
            "    if frame is None:\n"
            "        raise ValueError('no frame')\n"
        ),
        "harmless": (
            "COLUMN = 'household_weight'\n\n"
            "def check(frame):\n"
            "    if frame is None:\n"
            "        raise ValueError('no frame')\n\n"
            "def column(entity):\n"
            "    return f'{entity}_support_channel'\n\n"
            "class Record:\n"
            "    value: float\n"
        ),
        "unused": "def never(x):\n    return x - 1\n",
        "behind": "def kept(x):\n    return x * 4\n",
        "core": (
            "from .behind import kept\n\n"
            "class Container:\n"
            "    def grow(self, x):\n"
            "        return kept(x)\n"
        ),
        "stage": (
            "from . import attribute\n"
            "from .alias import scale\n"
            "from .codes import CODES, OWNER\n"
            "from .collaborator import Contract\n"
            "from .computed import TABLE\n"
            "from .core import Container\n"
            "from .direct import draw\n"
            "from .fielded import Inputs\n"
            "from .guard import guard\n"
            "from .harmless import COLUMN, Record, check, column\n"
            "from .inplace import fold\n"
            "from .lazily import deferred\n"
            "from .numeric import FLOOR\n"
            "from .reexport import total\n"
            "from .unused import never\n\n"
            "class Transform:\n"
            "    contract: Contract\n"
            "    inputs: Inputs\n\n"
            "    def __call__(self, frame):\n"
            "        check(frame)\n"
            "        guard(frame)\n"
            "        fold(frame, [0], [1])\n"
            "        record = Record(value=draw(1) + total([1]) + deferred(1))\n"
            "        rate = self.contract.rate(column('household'))\n"
            "        rate += self.inputs.rate.value()\n"
            "        owners = [OWNER[code] for code in CODES]\n"
            "        rate += Container().grow(rate)\n"
            "        return attribute.share(scale(rate)) + TABLE['a'] + FLOOR, (\n"
            "            record, COLUMN, owners\n"
            "        )\n\n"
            "def helper():\n"
            "    return never(1)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    monkeypatch.setitem(
        _REFUSAL_ONLY_HELPERS, (f"{prefix}harmless", "check"), "control"
    )
    monkeypatch.setitem(_CORE_MODULES, f"{prefix}core", "control")
    monkeypatch.setitem(_LAZY_EXPORT_TABLES, f"{prefix}lazily", "_EXPORTS")
    reach, stops, core = _helper_walk((f"{prefix}stage", "Transform"), scope=prefix)
    assert {module.removeprefix(prefix): names for module, names in reach.items()} == {
        "stage": {"Transform"},
        "direct": {"draw"},
        "origin": {"total"},
        "lazily.runtime": {"deferred"},
        "attribute": {"share"},
        "alias": {"scale", "_scale"},
        "computed": {"TABLE", "_build"},
        "collaborator": {"Contract"},
        "fielded": {"Inputs", "Rate"},
        "numeric": {"FLOOR"},
        "codes": {"CODES", "OWNER"},
        "inplace": {"fold"},
        "guard": {"guard"},
    }
    assert {(module.removeprefix(prefix), name) for module, name in stops} == {
        ("harmless", "COLUMN"),
        ("harmless", "Record"),
        ("harmless", "column"),
    }
    assert core == {f"{prefix}core"}
    followed = _helper_walk(
        (f"{prefix}stage", "Transform"), scope=prefix, follow_core=True
    )
    assert set(followed.reach) - set(reach) == {f"{prefix}core", f"{prefix}behind"}
    assert followed.core == core


class _KernelWalk(NamedTuple):
    """What a kernel's own code can compute through (``_kernel_walk``)."""

    reach: dict[str, frozenset[str]]
    stops: frozenset[tuple[str, str]]
    core: frozenset[str]
    met: dict[tuple[str, str], frozenset]
    core_names: frozenset[tuple[str, str]]


def _kernel_classes(kernel_class: type) -> list[type]:
    """The first-party classes a kernel's code comes from, in its method
    resolution order, without the core modules' (``KernelBase``)."""

    return [
        cls
        for cls in kernel_class.__mro__
        if cls.__module__.startswith(_HELPER_SCOPE)
        and cls.__module__ not in _CORE_MODULES
    ]


def _calls_super(node) -> bool:
    import ast

    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "super"
        for sub in ast.walk(node)
    )


def _statement_names(statement) -> list[str]:
    import ast

    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
        return [statement.name]
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        return [
            sub.id
            for target in targets
            for sub in ast.walk(target)
            if isinstance(sub, ast.Name)
        ]
    return []


def _kernel_body(kernel_class: type) -> list[tuple[type, object, str]]:
    """What a kernel of ``kernel_class`` runs: each method or class attribute
    with its class and name, from the first class in the method resolution
    order that defines it, as Python resolves it, and a base's definition too
    where the one that wins calls ``super()``. ``implementation_hash`` is left
    out, since it computes the key: an edit to what it calls that changes what
    it returns changes the key. So are the listed refusal-only methods."""

    chosen: dict[str, object | None] = {}
    body = []
    for cls in _kernel_classes(kernel_class):
        definitions = _module_definitions(cls.__module__, _HELPER_SCOPE).definitions
        for statement in definitions[cls.__name__].body:
            for name in _statement_names(statement):
                if name in chosen:
                    previous = chosen[name]
                    if previous is not None and _calls_super(previous):
                        chosen[name] = statement
                        body.append((cls, statement, name))
                    continue
                if name == "implementation_hash" or (
                    (cls.__module__, f"{cls.__name__}.{name}") in _REFUSAL_ONLY_HELPERS
                ):
                    chosen[name] = None
                    continue
                chosen[name] = statement
                body.append((cls, statement, name))
    return body


def _kernel_walk(
    kernel_class: type,
    scope: str = _HELPER_SCOPE,
    *,
    halts: Mapping[tuple[str, str], str] | None = None,
) -> _KernelWalk:
    """What a kernel's own code can compute through: the rule followed from
    what ``_kernel_body`` names, as from a stage's transform, stopping also at
    ``halts`` (``_KERNEL_STOPS`` unless given). Its classes are in ``reach``;
    ``met`` maps each listed stop it meets to what names it there (a
    definition, or ``Class.attribute`` for the kernel's own code), and
    ``core_names`` holds the core modules' definitions it names."""

    classes = _kernel_classes(kernel_class)
    seeds = [
        (found, (cls.__module__, f"{cls.__name__}.{name}"))
        for cls, statement, name in _kernel_body(kernel_class)
        for found in _named(cls.__module__, statement, scope)
    ]
    walk, met, core_names = _walk(
        {(cls.__module__, cls.__name__) for cls in classes},
        seeds,
        scope,
        halts=_KERNEL_STOPS if halts is None else halts,
    )
    return _KernelWalk(walk.reach, walk.stops, walk.core, met, core_names)


def _hashed_modules(kernel) -> frozenset[str]:
    """Every module ``kernel.implementation_hash()`` hashes, recorded at each
    ``source_hash`` it reaches, in whichever module calls it: every loaded
    module's ``source_hash`` name stands in for the recorder during one call,
    after a first call has imported what the hash imports."""

    import inspect
    import sys
    from types import ModuleType

    from microcosm.graph import kernel as graph_kernel

    original = graph_kernel.source_hash
    recorded: list[object] = []

    def record(*objects, **dependencies):
        recorded.extend(objects)
        return original(*objects, **dependencies)

    kernel.implementation_hash()
    patched = [
        module
        for module in list(sys.modules.values())
        if isinstance(module, ModuleType)
        and getattr(module, "__dict__", {}).get("source_hash") is original
    ]
    for module in patched:
        module.source_hash = record
    try:
        kernel.implementation_hash()
    finally:
        for module in patched:
            module.source_hash = original
    return frozenset(
        (item if isinstance(item, ModuleType) else inspect.getmodule(item)).__name__
        for item in recorded
    )


@functools.cache
def _uk_graph_kernels() -> MappingProxyType:
    """Every kernel the UK registries register, by ref: the spine's (the
    hermetic registry, whose stage kernels the production spine also uses)
    with its gate battery, the full build's, its gate batteries and terminal,
    certification and inspection kernels, and the national line's."""

    from microcosm.build.uk_runtime.full_certification import (
        register_uk_full_certification_kernel,
    )
    from microcosm.build.uk_runtime.graph_build import register_uk_full_kernels
    from microcosm.build.uk_runtime.graph_evidence import register_spine_gate_kernel
    from microcosm.build.uk_runtime.graph_national import register_uk_national_kernels
    from microcosm.build.uk_runtime.graph_terminal import (
        register_uk_full_gate_kernels,
        register_uk_terminal_kernels,
    )
    from microcosm.build.uk_runtime.inspection_export import (
        register_uk_inspection_export_kernel,
    )

    registry = uk_registry(graph=uk_spine_graph())
    register_spine_gate_kernel(
        registry,
        spec=load_country_spec("uk"),
        engine=object(),
        engine_identity="declaration-only",
    )
    register_uk_full_kernels(registry)
    register_uk_full_gate_kernels(
        registry, coverage_engine=object(), engine_identity="declaration-only"
    )
    register_uk_terminal_kernels(registry)
    register_uk_full_certification_kernel(registry)
    register_uk_inspection_export_kernel(registry)
    register_uk_national_kernels(registry, resolver_factory=None)
    return MappingProxyType({ref: registry.get(ref) for ref in registry.refs()})


def test_the_kernel_rule_sees_every_kernel_the_uk_runtime_defines() -> None:
    """The kernels the rule enumerates are every kernel class the UK runtime
    and the shared atomic geography define, so a new one cannot escape it,
    and the held kernels are among them (microcosm#931)."""
    import ast
    import importlib

    from microcosm.build import graph_atomic_geography
    from microcosm.build.uk_runtime import graph_kernels

    files = sorted(Path(graph_kernels.__file__).parent.glob("*.py"))
    files.append(Path(graph_atomic_geography.__file__))
    defined = set()
    for path in files:
        module = (
            graph_atomic_geography.__name__
            if path.stem == "graph_atomic_geography"
            else f"{graph_kernels.__package__}.{path.stem}"
        )
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            declares_ref = any(
                "ref" in _statement_names(statement) for statement in node.body
            ) or any(
                isinstance(sub, ast.Attribute)
                and sub.attr == "ref"
                and isinstance(sub.ctx, ast.Store)
                for sub in ast.walk(node)
            )
            if declares_ref:
                defined.add(getattr(importlib.import_module(module), node.name))
    kernels = _uk_graph_kernels()
    assert {type(kernel) for kernel in kernels.values()} == defined
    assert set(_HELD_KERNELS) <= set(kernels)


def test_every_uk_graph_kernel_hashes_each_helper_it_computes_through() -> None:
    """The stage rule, for every other UK graph kernel (microcosm#931): each
    module holding code a kernel's own methods can run to compute a value is
    in what its implementation hash really hashes, recorded at the
    ``source_hash`` calls it makes, and every name the rule stops at outside
    that hash is listed in ``_KERNEL_UNBOUND_NAMES`` with its reason.

    The rule starts at what the kernel runs (``_kernel_body``: its methods by
    method resolution order, without the hash itself and the listed
    refusal-only methods), not at its class, and stops at the declaration,
    refusal-only and core-module tables as for the stages, and at
    ``_KERNEL_STOPS``. Where a class has several kernels (the spine's stage
    kernels, one per stage), each one's key is held to it. The held kernels
    (``_HELD_KERNELS``) are not. A name microcosm.graph exports lazily,
    through its module ``__getattr__``, is followed to the runtime module
    that defines it like any other (``_LAZY_EXPORT_TABLES``): the kernels'
    capability and context records are listed core names, and the create
    kernel's ``load_source``, which it calls only where no root transform is
    bound, is a listed stop."""
    from microcosm.build.uk_runtime import graph_population, rowwise_geography

    kernels = _uk_graph_kernels()
    by_class: dict[type, list[tuple[str, object]]] = {}
    for ref, kernel in sorted(kernels.items()):
        by_class.setdefault(type(kernel), []).append((ref, kernel))
    unhashed: dict[str, list[str]] = {}
    unlisted: set[tuple[str, str]] = set()
    met: set[tuple[str, str]] = set()
    for kernel_class, members in by_class.items():
        held = {ref in _HELD_KERNELS for ref, _ in members}
        assert len(held) == 1, kernel_class
        if held == {True}:
            continue
        walk = _kernel_walk(kernel_class)
        met |= set(walk.met)
        for ref, kernel in members:
            hashed = _hashed_modules(kernel)
            missing = sorted(set(walk.reach) - hashed)
            if missing:
                unhashed[ref] = missing
            unlisted.update(item for item in walk.stops if item[0] not in hashed)
    assert not unhashed, unhashed
    assert unlisted == set(_KERNEL_UNBOUND_NAMES), sorted(
        unlisted ^ set(_KERNEL_UNBOUND_NAMES)
    )
    assert met == set(_KERNEL_STOPS), sorted(met ^ set(_KERNEL_STOPS))
    # The dense population kernels that share _PopulationKernel's hash
    # (sample, normalize, expand, locations, geography mapping, pool and
    # gate) mint their clones through rowwise_geography's helpers.
    shared = [
        kernel
        for kernel in kernels.values()
        if type(kernel).implementation_hash
        is graph_population._PopulationKernel.implementation_hash
    ]
    assert {kernel.ref for kernel in shared} >= {
        "uk.full.sample@1",
        "uk.full.expand@1",
        "uk.full.geography_gate@1",
    }
    for kernel in shared:
        assert rowwise_geography.__name__ in _hashed_modules(kernel), kernel.ref


def test_the_kernel_rule_stops_only_where_its_reasons_hold() -> None:
    """Each listed kernel stop is met only where its reason says
    (microcosm#931)."""
    import ast

    from microcosm.build.uk_runtime import (
        battery_bindings,
        graph_kernels,
        release_certification,
        spi_spine,
    )

    kernels = _uk_graph_kernels()
    namers: dict[tuple[str, str], dict[str, frozenset]] = {}
    for ref, kernel in kernels.items():
        if ref in _HELD_KERNELS:
            continue
        for stop, by in _kernel_walk(type(kernel)).met.items():
            namers.setdefault(stop, {})[ref] = by

    # The fixture resolver resolves the stage's own transform: of what holds a
    # value, it names only the builder the stage rule starts from.
    resolver = (graph_kernels.__name__, "_FixtureTransformResolver")
    definitions = _module_definitions(graph_kernels.__name__, _HELPER_SCOPE).definitions
    named = {
        item
        for item in _named(
            graph_kernels.__name__, definitions[resolver[1]], _HELPER_SCOPE
        )
        if not item[0].startswith(_DECLARATION_MODULES)
        and item[0] not in _CORE_MODULES
        and _helper_kind(*item, _HELPER_SCOPE) == "value"
    }
    assert named == {(graph_kernels.__name__, "_fixture_implementations")}
    assert set(namers[resolver]) == {
        kernel.ref
        for kernel in kernels.values()
        if isinstance(
            kernel, graph_kernels.UKStageKernel | graph_kernels.UKExpandStageKernel
        )
    }

    # The SPI support allocation's mass record: _minimal_frame names it only
    # inside its branch on the SPI income stage, whose node hashes all it
    # computes through.
    stage = "hmrc_spi_income_spine"
    reason = (spi_spine.__name__, "uk_spi_support_mass_change_reason")
    assert {namer for by in namers[reason].values() for namer in by} == {
        (graph_kernels.__name__, "_minimal_frame")
    }
    minimal = definitions["_minimal_frame"]
    guarded = set()
    for branch in ast.walk(minimal):
        if (
            isinstance(branch, ast.If)
            and ast.unparse(branch.test) == f"context.params.get('stage') == '{stage}'"
        ):
            guarded |= {id(sub) for item in branch.body for sub in ast.walk(item)}
    references = [
        sub
        for sub in ast.walk(minimal)
        if isinstance(sub, ast.Name | ast.Attribute)
        and reason in _named(graph_kernels.__name__, sub, _HELPER_SCOPE)
    ]
    assert references and all(id(sub) in guarded for sub in references)
    reach = _helper_walk(reason)
    hashed = _hashed_modules(kernels[f"uk.stage.{stage}@1"])
    assert set(reach.reach) | {module for module, _ in reach.stops} <= hashed

    # The source codecs: only the create kernel's run loads a source through
    # them, and only inside its branch for an unbound root transform. The
    # spine build installs the root's transform, and fixture_stage_plan_inputs,
    # which H2 parity builds its registry from, adds it.
    loading = ("microcosm.graph.codecs", "load_source")
    assert namers[loading] == {
        "uk.create@1": frozenset({(graph_kernels.__name__, "UKCreateKernel.run")})
    }
    (run,) = [
        item
        for item in definitions["UKCreateKernel"].body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    ]
    unbound = set()
    for branch in ast.walk(run):
        if (
            isinstance(branch, ast.If)
            and ast.unparse(branch.test) == "self.transform is None"
        ):
            unbound |= {id(sub) for item in branch.body for sub in ast.walk(item)}
    references = [
        sub
        for sub in ast.walk(run)
        if isinstance(sub, ast.Name | ast.Attribute)
        and loading in _named(graph_kernels.__name__, sub, _HELPER_SCOPE)
    ]
    assert references and all(id(sub) in unbound for sub in references)
    assert "frs_spine" in {install.stage for install in _spine_build_installs()}
    plan = definitions["fixture_stage_plan_inputs"]
    assert any(
        isinstance(sub, ast.Assign)
        and ast.unparse(sub.targets[0]) == "implementations['frs_spine']"
        for sub in ast.walk(plan)
    )

    # The gate registry: the bound spine derives its gate digests through it,
    # in _manifest_digests, which returns the battery run's digests alone;
    # the national battery evaluates through it and is held there.
    registry = (battery_bindings.__name__, "UK_GATE_REGISTRY")
    held_there = {ref for ref, stop in _HELD_KERNEL_STOPS if stop == registry}
    assert held_there == {"uk.full.national-gates@1"}
    assert {
        ref: by for ref, by in namers[registry].items() if ref not in held_there
    } == {
        "uk.full.bound_spine@1": frozenset(
            {(release_certification.__name__, "_manifest_digests")}
        ),
    }
    assert namers[registry]["uk.full.national-gates@1"] == frozenset(
        {("microcosm.build.uk_runtime.graph_national", "UKNationalGateKernel.run")}
    )
    digests = _module_definitions(
        release_certification.__name__, _HELPER_SCOPE
    ).definitions["_manifest_digests"]
    (returned,) = [sub for sub in ast.walk(digests) if isinstance(sub, ast.Return)]
    assert {ast.unparse(value) for value in returned.value.values} == {
        "run.gates_manifest_sha256",
        "run.policy_sha256",
        "run.spec_fingerprint",
    }


def test_every_hold_still_guards_a_module_its_kernel_leaves_out() -> None:
    """A hold lasts only while it is needed (microcosm#931 review): each held
    kernel's walk still meets a module its hash leaves out, and so does the
    walk past each held stop, so a hold that guards nothing fails here and
    leaves its table. Until the ruling, such a kernel can replay a stale
    output after an edit to one of those modules."""
    kernels = _uk_graph_kernels()
    assert {ref for ref, _ in _HELD_KERNEL_STOPS} <= set(kernels)
    assert not {ref for ref, _ in _HELD_KERNEL_STOPS} & set(_HELD_KERNELS)
    unneeded = []
    for ref in _HELD_KERNELS:
        kernel = kernels[ref]
        if not set(_kernel_walk(type(kernel)).reach) - _hashed_modules(kernel):
            unneeded.append(ref)
    for ref, stop in _HELD_KERNEL_STOPS:
        assert stop in _KERNEL_STOPS, (ref, stop)
        kernel = kernels[ref]
        past = _kernel_walk(
            type(kernel),
            halts={item: why for item, why in _KERNEL_STOPS.items() if item != stop},
        )
        if not set(past.reach) - _hashed_modules(kernel):
            unneeded.append((ref, stop))
    assert not unneeded, unneeded


def test_the_core_definitions_the_kernels_name_are_listed() -> None:
    """The kernel version of the stage rule's core check (microcosm#931
    review): the definitions of the core modules other than the Frame's that
    the UK graph kernels' own code names, outside the held kernels, are
    exactly ``_KERNEL_CORE_NAMES``, and the executor stores every node's
    owned cells through the dtype tokens' map, as that entry's reason says."""
    kernels = _uk_graph_kernels()
    named: set[tuple[str, str]] = set()
    for ref, kernel in kernels.items():
        if ref not in _HELD_KERNELS:
            named |= {
                item
                for item in _kernel_walk(type(kernel)).core_names
                if not item[0].startswith("microcosm.frame.")
            }
    assert named == set(_KERNEL_CORE_NAMES), sorted(named ^ set(_KERNEL_CORE_NAMES))
    patching = _helper_walk(("microcosm.graph.population", "patch"), follow_core=True)
    assert "dtype_for_token" in patching.reach["microcosm.graph.population"]


def test_every_lazy_export_resolves_to_the_object_python_returns() -> None:
    """The walk follows a name a package exports lazily, through its module
    ``__getattr__``, to the module that defines it, as Python imports it
    (microcosm#931 ruling 8). Every name microcosm.graph exports that way
    resolves to a definition holding the very object the package returns,
    the table the walk reads is the one the package's ``__getattr__`` reads,
    and a name the package imports eagerly resolves as before. Every
    first-party module that defines a module ``__getattr__`` is listed, as a
    lazy exporter or as one that exports nothing through it, so a new one
    cannot hide names from the walk."""
    import ast
    import importlib

    import microcosm
    from microcosm import graph

    for module, table in _LAZY_EXPORT_TABLES.items():
        getter = _module_definitions(module, _HELPER_SCOPE).definitions["__getattr__"]
        assert table in {
            sub.id for sub in ast.walk(getter) if isinstance(sub, ast.Name)
        }, module
    exports = getattr(graph, _LAZY_EXPORT_TABLES[graph.__name__])
    assert {"load_source", "KernelContext", "keyed_uniform"} <= set(exports)
    for name in exports:
        found = _resolve(graph.__name__, name, _HELPER_SCOPE)
        assert found is not None, name
        module, defined = found
        assert getattr(importlib.import_module(module), defined) is getattr(
            graph, name
        ), name
    assert _resolve(graph.__name__, "compile_graph", _HELPER_SCOPE) == (
        "microcosm.graph.decl",
        "compile_graph",
    )
    getters = set()
    for root in map(Path, microcosm.__path__):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "def __getattr__" not in text or not any(
                isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
                for node in ast.parse(text).body
            ):
                continue
            parts = path.relative_to(root.parent).with_suffix("").parts
            getters.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    listed = set(_LAZY_EXPORT_TABLES) | set(_NO_LAZY_EXPORTS)
    assert getters == listed, sorted(getters ^ listed)
    # The modules that export nothing that way only raise from __getattr__.
    for module in _NO_LAZY_EXPORTS:
        getter = _module_definitions(module, _HELPER_SCOPE).definitions["__getattr__"]
        assert not any(isinstance(sub, ast.Return) for sub in ast.walk(getter))


def test_no_spine_or_national_line_kernel_hashes_the_source_codecs() -> None:
    """The create kernel's ``load_source`` stop keeps the spine's keys off
    microcosm.graph.codecs (microcosm#931 ruling 8): no spine kernel hashes
    it, so an edit to it re-keys no spine node, nor any node downstream of
    the spine through it, and no kernel of the national line, whose root is
    the bound spine checkpoint, hashes it either. The atomic geography
    kernels, which read their sources' bytes through it, do."""
    from microcosm.build.uk_runtime import (
        graph_build,
        graph_calibration,
        graph_national,
    )
    from microcosm.graph import codecs

    spine = uk_registry(graph=uk_spine_graph())
    for ref in spine.refs():
        assert codecs.__name__ not in _hashed_modules(spine.get(ref)), ref
    kernels = _uk_graph_kernels()
    for ref in (
        graph_build.UKBoundSpineKernel.ref,
        graph_national.UKNationalTargetKernel.ref,
        graph_national.UKNationalProblemKernel.ref,
        graph_calibration.UKDenseSolveKernel.ref,
        graph_national.UKNationalGateKernel.ref,
        graph_national.UKNationalReadbackKernel.ref,
    ):
        assert codecs.__name__ not in _hashed_modules(kernels[ref]), ref
    assert codecs.__name__ in _hashed_modules(kernels["geography.support_import@1"])


def test_the_frame_totals_the_uk_corrections_target_are_numpys_sums() -> None:
    """H2 cannot catch an edit to the Frame, which the graph's spine and the
    legacy stage plan both run through (``_FRAME_REASON``). The UK exact-total
    corrections correct a vector until NumPy's sum of it equals a frame's
    ``Weights.total``, so this pins that the total is that same sum, bit for
    bit, on vectors whose pairwise and correctly rounded sums differ
    (microcosm#931 review)."""
    import math

    rng = np.random.default_rng(11)
    differing = 0
    for size in (7, 129, 1_000, 78_188):
        values = rng.lognormal(3.0, 2.0, size)
        total = Weights(values, WeightKind.IMPORTANCE).total
        assert total == float(values.sum())
        differing += total != math.fsum(values)
    assert differing


def test_the_kernel_rule_follows_what_a_kernel_runs(tmp_path, monkeypatch) -> None:
    """Control: the kernel rule follows what a kernel's methods name,
    through its bases by method resolution order (a base method it inherits,
    and one it overrides only where it calls ``super()``), and leaves out its
    implementation hash, a listed refusal-only method, a base method it
    overrides without ``super()`` and what lies behind a listed kernel stop,
    which it reports with what named it. The recorder sees each module the
    hash passes to ``source_hash``, in whichever module calls it."""
    import importlib
    import uuid

    scratch = f"kernel_rule_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "computed": "def computed(x):\n    return x + 1\n",
        "inherited": "def base_only(x):\n    return x * 2\n",
        "assisted": "def kept(x):\n    return x * 3\n",
        "overridden": "def replaced(x):\n    return x - 2\n",
        "refused": "def check(x):\n    return x < 0\n",
        "keyonly": "def salt():\n    return 'salt'\n",
        "behind": "def beyond(x):\n    return x * 5\n",
        "table": "from .behind import beyond\n\nTABLE = {'a': beyond}\n",
        "keyed": (
            "from microcosm.graph import source_hash\n\n"
            "from . import computed, keyonly\n\n"
            "def key():\n    return source_hash(computed, keyonly)\n"
        ),
        "base": (
            "from .assisted import kept\n"
            "from .inherited import base_only\n"
            "from .overridden import replaced\n\n"
            "class Base:\n"
            "    def run(self, x):\n"
            "        return base_only(x)\n\n"
            "    def prepare(self, x):\n"
            "        return replaced(x)\n\n"
            "    def assist(self, x):\n"
            "        return kept(x)\n"
        ),
        "kernel": (
            "from .base import Base\n"
            "from .computed import computed\n"
            "from .keyed import key\n"
            "from .keyonly import salt\n"
            "from .refused import check\n"
            "from .table import TABLE\n\n"
            "class Kernel(Base):\n"
            "    def implementation_hash(self):\n"
            "        return salt() + key()\n\n"
            "    def _admit(self, x):\n"
            "        if check(x):\n"
            "            raise ValueError('refused')\n\n"
            "    def prepare(self, x):\n"
            "        return x\n\n"
            "    def run(self, x):\n"
            "        self._admit(x)\n"
            "        return computed(x) + super().run(x) + TABLE['a'](x)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    monkeypatch.setitem(
        _REFUSAL_ONLY_HELPERS, (f"{prefix}kernel", "Kernel._admit"), "control"
    )
    monkeypatch.setitem(_KERNEL_STOPS, (f"{prefix}table", "TABLE"), "control")
    kernel_class = importlib.import_module(f"{prefix}kernel").Kernel
    walk = _kernel_walk(kernel_class, scope=prefix)
    assert {
        module.removeprefix(prefix): names for module, names in walk.reach.items()
    } == {
        "kernel": {"Kernel"},
        "base": {"Base"},
        "computed": {"computed"},
        "inherited": {"base_only"},
        "assisted": {"kept"},
    }
    assert walk.met == {
        (f"{prefix}table", "TABLE"): frozenset({(f"{prefix}kernel", "Kernel.run")})
    }
    assert _hashed_modules(kernel_class()) == {f"{prefix}computed", f"{prefix}keyonly"}


#: Adapters the spine build installs around a stage's own transform, each with
#: why it computes no value.
_SPINE_BUILD_ADAPTERS: dict[tuple[str, str], str] = {
    ("microcosm.build.uk_runtime.spine_build", "_GraphSourceTransform"): (
        "builds the stage's own transform from the node's declared sources and calls it"
    ),
    ("microcosm.build.observation", "ObservedTransform"): (
        "times the wrapped transform's call and records its frames' sizes, and "
        "forwards every other attribute, the declared modules included (checked)"
    ),
}
#: Adapters whose first argument is the stage's own transform, or builds it,
#: and which do not hand the node what that transform declares: the source
#: adapter builds it only when the node runs, after the node is keyed, and the
#: rung sampler declares its own modules alone. What either wraps must live in
#: the stage's module, which the node hashes.
_SPINE_BUILD_WRAPPERS = frozenset(
    {
        ("microcosm.build.uk_runtime.spine_build", "_GraphSourceTransform"),
        ("microcosm.build.uk_runtime.national_sampling", "UKSampledSpineRootTransform"),
    }
)


class _Install(NamedTuple):
    """One class or factory the spine build installs (``_spine_build_installs``)."""

    stage: str | None
    found: tuple[str, str]
    wrapper: tuple[str, str] | None


def _spine_build_installs() -> list[_Install]:
    """Each class or factory the production spine build installs as a
    stage's transform, seen through the adapters above, the local names it
    assigns and the stage entries a wrapper reads back. ``stage`` is ``None``
    for an adapter it installs around every stage; ``wrapper`` is the
    innermost wrapping adapter (``_SPINE_BUILD_WRAPPERS``) the install sits
    inside, or ``None``."""

    import ast

    from microcosm.build.uk_runtime import spine_build

    module = spine_build.__name__
    tree = ast.parse(Path(spine_build.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_uk_spine_execution"
    )

    def entry(target) -> str | None:
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "implementations"
            and isinstance(target.slice, ast.Constant)
        ):
            return target.slice.value
        return None

    local: dict[str, list] = {}
    entries: dict[str, list] = {}
    for sub in ast.walk(function):
        if not isinstance(sub, ast.Assign) or len(sub.targets) != 1:
            continue
        target = sub.targets[0]
        if isinstance(target, ast.Name):
            local.setdefault(target.id, []).append(sub.value)
            if target.id == "implementations" and isinstance(sub.value, ast.Dict):
                for key, value in zip(sub.value.keys, sub.value.values, strict=True):
                    entries.setdefault(key.value, []).append(value)
        elif entry(target) is not None:
            entries.setdefault(entry(target), []).append(sub.value)
    installs: list[_Install] = []

    def follow(expression, stage, wrapper=None, seen=frozenset()) -> None:
        if isinstance(expression, ast.Lambda):
            follow(expression.body, stage, wrapper, seen)
        elif isinstance(expression, ast.Name) and expression.id not in seen:
            for value in local.get(expression.id, ()):
                follow(value, stage, wrapper, seen | {expression.id})
        elif entry(expression) is not None and ("entry", entry(expression)) not in seen:
            # A wrapper reads back the stage's entry: what was installed there
            # before it now sits inside it.
            for value in entries.get(entry(expression), ()):
                follow(value, stage, wrapper, seen | {("entry", entry(expression))})
        elif isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            found = _resolve(module, expression.func.id, _HELPER_SCOPE)
            if found is None:
                return
            installs.append(_Install(stage, found, wrapper))
            adapts = found in _SPINE_BUILD_ADAPTERS or found in _SPINE_BUILD_WRAPPERS
            if adapts and stage and expression.args:
                inner = found if found in _SPINE_BUILD_WRAPPERS else wrapper
                follow(expression.args[0], stage, inner, seen)

    for sub in ast.walk(function):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
            target = sub.targets[0]
            if isinstance(target, ast.Name) and target.id == "implementations":
                if isinstance(sub.value, ast.Dict):
                    for key, value in zip(
                        sub.value.keys, sub.value.values, strict=True
                    ):
                        follow(value, key.value)
                elif isinstance(sub.value, ast.DictComp):
                    follow(sub.value.value, None)
            elif entry(target) is not None:
                follow(sub.value, entry(target))
    return installs


def test_every_transform_the_spine_build_installs_is_hashed_with_its_stage() -> None:
    """The production spine builds its stages' transforms in spine_build, not
    in graph_kernels, so the rule above, which starts at graph_kernels'
    transforms, does not see them. Each class or factory spine_build
    installs for a stage is defined in the stage's module, declares the
    modules it computes through (``graph_implementation_dependencies``, which
    the stage's node hashes), or is an adapter listed with why it computes no
    value. Inside a wrapping adapter, which hands the node nothing the wrapped
    transform declares, only the stage's own module's classes and the listed
    adapters may be installed (microcosm#931)."""
    import importlib

    from microcosm.build.observation import ObservedTransform
    from microcosm.build.uk_runtime import graph_kernels, national_sampling

    installs = _spine_build_installs()
    assert {install.stage for install in installs if install.stage} == set(
        graph_kernels._STAGE_MODULES
    )
    assert {install.found for install in installs if install.stage is None} == {
        ("microcosm.build.observation", "ObservedTransform")
    }
    # Both wrappers wrap something: every source adapter wraps its stage's
    # transform, and the rung sampler wraps the root's source adapter.
    assert {install.wrapper for install in installs} - {None} == _SPINE_BUILD_WRAPPERS
    unhashed = []
    for stage, (module, name), wrapper in installs:
        own = bool(stage) and module == graph_kernels._stage_module(stage).__name__
        if own or (module, name) in _SPINE_BUILD_ADAPTERS:
            continue
        declared = getattr(
            getattr(importlib.import_module(module), name),
            "graph_implementation_dependencies",
            None,
        )
        if wrapper is not None or not callable(declared):
            unhashed.append((stage, f"{module}.{name}", wrapper))
    assert not unhashed, unhashed
    # The observation adapter hands the node the declared modules of what it
    # wraps, the rung sampler's among them.
    sampler = national_sampling.UKSampledSpineRootTransform(None, fraction=0.1, seed=1)
    observed = ObservedTransform(
        sampler, stage_id="frs_spine", produced_column_count=0, observer=None
    )
    assert hashed_module_names(
        graph_kernels.UKCreateKernel(observed), graph_kernels
    ) == hashed_module_names(graph_kernels.UKCreateKernel(sampler), graph_kernels)


def test_the_sampled_spine_root_hashes_what_it_samples_through(monkeypatch) -> None:
    """The spine build wraps the root's transform in the rung sampler, whose
    code lives in national_sampling and the shared core. The root node hashes
    national_sampling at every rung, since the sampler resets the root's
    indexes at f100 too, and the shared core only below f100, where
    sample_uk_spine_root calls it; there it hashes every module the rule
    finds the sampler computing through (microcosm#931)."""
    from microcosm.build import frame_sampling
    from microcosm.build.uk_runtime import graph_kernels, national_sampling

    plain = hashed_module_names(graph_kernels.UKCreateKernel(), graph_kernels)
    hashed = {}
    for fraction in (1.0, 0.1):
        sampler = national_sampling.UKSampledSpineRootTransform(
            None, fraction=fraction, seed=1
        )
        hashed[fraction] = hashed_module_names(
            graph_kernels.UKCreateKernel(sampler), graph_kernels
        )
    assert hashed[1.0] - plain == {national_sampling.__name__}
    assert hashed[0.1] - plain == {national_sampling.__name__, frame_sampling.__name__}
    reach = _helper_reach((national_sampling.__name__, "UKSampledSpineRootTransform"))
    assert {national_sampling.__name__, frame_sampling.__name__} <= set(reach)
    assert set(reach) <= hashed[0.1], sorted(set(reach) - hashed[0.1])

    # At f100 the root returns before the shared core samples anything.
    def refuse(*args, **kwargs):
        raise AssertionError("f100 must not sample")

    for name in ("sample_frame_households", "normalize_sampled_household_mass"):
        monkeypatch.setattr(national_sampling, name, refuse)
    frame = object()
    assert national_sampling.sample_uk_spine_root(frame, fraction=1.0, seed=1) == (
        frame,
        None,
    )


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
