"""Tests split from packages/microcosm-build/tests/test_uk_graph.py."""

# ruff: noqa: F403, F405
import functools
import hashlib
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
#: and so binds them too (microcosm#931).
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


# The rule behind the bindings above, for every helper (microcosm#931):
# ``source_hash`` does not follow imports, so a node's hash must name every
# module whose code can change what the node computes. The tests below
# derive those modules from what a stage's transform, or a dense population
# kernel, names, and follow each named definition into the module that
# defines it, through re-exports and module attributes, to a fixed point.

#: The package the rule follows: microcosm.build holds every UK stage and its
#: helpers. The workspace's other packages stay outside it. microcosm.graph
#: and microcosm.frame are the executor and the Frame that every node runs
#: on, so binding them would re-key every node of every graph on any edit to
#: them; microcosm.fit's fitters, shared by every country, are bound by hand
#: (the employment shells above) and not derived here.
_HELPER_SCOPE = "microcosm.build."
#: Where the rule stops. What a stage reads from these is its own
#: declaration, which its node key already hashes as the stage contract: its
#: operations, artifacts and resource pins (graph._stage_contract_sha256).
_DECLARATION_MODULES = (
    "microcosm.build.country_spec",
    "microcosm.build.source_manifest",
    "microcosm.build.spec_engine",
)
#: Checks that read values only to decide whether to refuse, each with why it
#: cannot change what its caller computes. The rule does not follow them; the
#: test checks that each returns nothing and writes nothing.
_REFUSAL_ONLY_HELPERS: dict[tuple[str, str], str] = {
    (
        "microcosm.build.uk_runtime.graph_evidence",
        "require_uk_spine_gate_admission",
    ): (
        "refuses a node unless the spine gates admitted its input; the dense "
        "sample node calls it before it computes anything"
    ),
}
#: Method names that change their receiver in place.
_IN_PLACE_CALLS = frozenset(
    {
        "__setattr__",
        "__setitem__",
        "add",
        "append",
        "clear",
        "discard",
        "extend",
        "fill",
        "insert",
        "pop",
        "popitem",
        "remove",
        "setdefault",
        "sort",
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
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        definitions[sub.id] = node
    return _ModuleDefinitions(imports, modules, tuple(stars), definitions)


def _resolve(module: str, name: str, scope: str) -> tuple[str, str] | None:
    """The top-level definition ``name`` refers to in ``module``, through
    re-exports; None for a module or anything outside ``scope``."""

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
        else:
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
    """Whether a function returns nothing but None or an argument it never
    rebinds, and writes through no subscript, attribute or in-place call."""

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
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            value = sub.value
            if not (
                (isinstance(value, ast.Constant) and value.value is None)
                or (isinstance(value, ast.Name) and value.id in parameters - rebound)
            ):
                return False
        elif isinstance(sub, ast.Yield | ast.YieldFrom | ast.Global | ast.Nonlocal):
            return False
        elif isinstance(sub, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete):
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
        elif isinstance(sub, ast.keyword) and sub.arg == "inplace":
            return False
        elif isinstance(sub, ast.Call):
            function = sub.func
            if isinstance(function, ast.Attribute) and function.attr in _IN_PLACE_CALLS:
                return False
            if isinstance(function, ast.Name) and function.id in {
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


def _only_strings(node) -> bool:
    """Whether an expression is built of string literals, None, and names,
    in literal collections: a name, a code, or a collection of them."""

    import ast

    collections = {"MappingProxyType", "dict", "frozenset", "list", "set", "tuple"}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant):
            if not (sub.value is None or isinstance(sub.value, str)):
                return False
        elif isinstance(sub, ast.Call):
            if not (isinstance(sub.func, ast.Name) and sub.func.id in collections):
                return False
        elif not isinstance(
            sub,
            ast.Name
            | ast.Attribute
            | ast.Tuple
            | ast.List
            | ast.Set
            | ast.Dict
            | ast.JoinedStr
            | ast.FormattedValue
            | ast.expr_context,
        ):
            return False
    return True


_HELPER_KINDS: dict[tuple[str, str, str], str | None] = {}


def _helper_kind(module: str, name: str, scope: str) -> str:
    """How a top-level definition can bear on a value.

    ``value`` unless it is provably one of: a ``check`` (a function that
    writes nothing and names only definitions of these kinds, so it can only
    refuse); a ``column name`` (a function returning one string built from
    its arguments); a ``record`` (a class of undefaulted fields, no methods);
    ``names`` (a constant of strings: a column name, a code, or a collection
    of them). A numeric constant is a value. A cycle reads as ``value``.
    """

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
    if isinstance(node, ast.Assign | ast.AnnAssign):
        if node.value is None or (
            _only_strings(node.value)
            and _names_only_harmless(module, node.value, scope)
        ):
            return "names"
        return "value"
    if isinstance(node, ast.ClassDef):
        fields_only = all(
            (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
            or isinstance(item, ast.Pass)
            or (isinstance(item, ast.AnnAssign) and item.value is None)
            for item in node.body
        )
        if fields_only and all(
            _names_only_harmless(module, item, scope)
            for item in (*node.bases, *node.keywords, *node.decorator_list)
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
    if _writes_nothing(node) and _names_only_harmless(module, node, scope):
        return "check"
    return "value"


def _helper_reach(
    root: tuple[str, str], scope: str = _HELPER_SCOPE
) -> dict[str, frozenset[str]]:
    """Every module holding a definition that can change what ``root``
    computes, with those definitions: ``root`` itself, and each ``value``
    definition it names, and they name, to a fixed point. The declaration
    modules and the refusal-only checks are not followed."""

    reached: dict[str, set[str]] = {}
    pending, seen = [root], set()
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        module, name = item
        if module.startswith(_DECLARATION_MODULES) or item in _REFUSAL_ONLY_HELPERS:
            continue
        if item != root and _helper_kind(module, name, scope) != "value":
            continue
        reached.setdefault(module, set()).add(name)
        node = _module_definitions(module, scope).definitions[name]
        pending.extend(_named(module, node, scope))
    return {module: frozenset(names) for module, names in reached.items()}


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


def test_every_stage_hashes_each_helper_its_transform_computes_through() -> None:
    """A stage's hash names every module holding code its transform can run
    to compute a value, so an edit to any of them re-keys the stage instead
    of replaying a stale output (microcosm#931).

    The rule starts at the transform graph_kernels builds for the stage and
    follows every definition it names, annotations included (an injected
    contract or table is called through them), into the module defining it.
    It needs no binding, and does not follow, a definition that cannot change
    a value: a check, a column name, a record, a constant of strings (see
    ``_helper_kind``), or a stage declaration. A numeric constant needs its
    module bound. The rule starts at the transform, not at the kernel that
    runs it: the kernel's code, in graph_kernels, which every stage hashes,
    builds every stage's transform for the hermetic registry, so following it
    would bind every helper into every stage.

    What the rule cannot see: a call through an object whose class the code
    never names, a name looked up by string, and data files read by path.
    """
    from microcosm.build.uk_runtime import graph_kernels

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    transforms = _stage_transforms()
    assert set(transforms) == set(graph_kernels._STAGE_MODULES)
    unhashed = {}
    for stage, transform in sorted(transforms.items()):
        assert transform[0] == graph_kernels._stage_module(stage).__name__, stage
        # The root stage's node is the CREATE kernel.
        node = graph.node("create_uk_frs" if stage == "frs_spine" else stage)
        hashed = hashed_module_names(registry.get(node.kernel), graph_kernels)
        missing = sorted(set(_helper_reach(transform)) - hashed)
        if missing:
            unhashed[stage] = missing
    assert not unhashed, unhashed


def test_the_refusal_only_helpers_return_and_write_nothing() -> None:
    for module, name in _REFUSAL_ONLY_HELPERS:
        node = _module_definitions(module, _HELPER_SCOPE).definitions[name]
        assert _writes_nothing(node), (module, name)


def test_the_helper_rule_follows_each_way_a_stage_names_a_helper(
    tmp_path, monkeypatch
) -> None:
    """Control: a helper called by name, through a re-export, by attribute on
    its module, through an alias or a computed constant, or held only in an
    annotation is reached, as is a numeric constant; a check, a column name,
    a record, a constant of strings, and a helper the root never names are
    not."""
    import uuid

    scratch = f"helper_rule_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
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
        "numeric": "FLOOR = 10\n",
        "harmless": (
            "COLUMN = 'household_weight'\n"
            "CODES = frozenset({'A', 'B'})\n\n"
            "def check(frame):\n"
            "    if frame is None:\n"
            "        raise ValueError('no frame')\n\n"
            "def column(entity):\n"
            "    return f'{entity}_support_channel'\n\n"
            "class Record:\n"
            "    value: float\n"
        ),
        "unused": "def never(x):\n    return x - 1\n",
        "stage": (
            "from . import attribute\n"
            "from .alias import scale\n"
            "from .collaborator import Contract\n"
            "from .computed import TABLE\n"
            "from .direct import draw\n"
            "from .harmless import CODES, COLUMN, Record, check, column\n"
            "from .numeric import FLOOR\n"
            "from .reexport import total\n"
            "from .unused import never\n\n"
            "class Transform:\n"
            "    contract: Contract\n\n"
            "    def __call__(self, frame):\n"
            "        check(frame)\n"
            "        record = Record(value=draw(1) + total([1]))\n"
            "        rate = self.contract.rate(column('household'))\n"
            "        return attribute.share(scale(rate)) + TABLE['a'] + FLOOR, (\n"
            "            record, COLUMN, CODES\n"
            "        )\n\n"
            "def helper():\n"
            "    return never(1)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    reach = _helper_reach((f"{prefix}stage", "Transform"), scope=prefix)
    assert {module.removeprefix(prefix): names for module, names in reach.items()} == {
        "stage": {"Transform"},
        "direct": {"draw"},
        "origin": {"total"},
        "attribute": {"share"},
        "alias": {"scale", "_scale"},
        "computed": {"TABLE", "_build"},
        "collaborator": {"Contract"},
        "numeric": {"FLOOR"},
    }


def test_the_dense_population_kernels_hash_each_helper_they_compute_through() -> None:
    """The same rule for the dense population kernels that share
    ``_PopulationKernel``'s hash (sample, normalize, expand, locations,
    geography mapping, pool and gate): every module holding code one of them
    can run to compute a value is in the hash they share (microcosm#931).
    The rule starts at each kernel's class; the spine gate admission check
    the sample kernel calls is refusal-only and is not followed."""
    import inspect

    from microcosm.build.uk_runtime import graph_population, rowwise_geography

    kernels = [
        member
        for _, member in inspect.getmembers(graph_population, inspect.isclass)
        if member.__module__ == graph_population.__name__
        and issubclass(member, graph_population._PopulationKernel)
        and member is not graph_population._PopulationKernel
        and member.implementation_hash
        is graph_population._PopulationKernel.implementation_hash
    ]
    assert {kernel.ref for kernel in kernels} >= {
        "uk.full.sample@1",
        "uk.full.expand@1",
        "uk.full.geography_gate@1",
    }
    unhashed = {}
    for kernel in kernels:
        hashed = hashed_module_names(kernel.__new__(kernel), graph_population)
        # The dense clones are minted by rowwise_geography's helpers.
        assert rowwise_geography.__name__ in hashed
        missing = sorted(
            set(_helper_reach((graph_population.__name__, kernel.__name__))) - hashed
        )
        if missing:
            unhashed[kernel.ref] = missing
    assert not unhashed, unhashed


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
