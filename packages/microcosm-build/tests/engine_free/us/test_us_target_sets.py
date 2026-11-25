"""The US products' declared target sets (microcosm#1728).

``us/target_sets.json`` declares what the national release and the ACS local
release calibrate to. The engine-free tests check the declarations against
the tools. The pinned-feed tests prove that the declarations select the same
rows, in the same order, as the tool code they replace: each surface's count
and ``TargetRegistry.version`` (an order-sensitive content hash) were measured
on main ``0f963ac77`` before the change. They run only when
``MICROCOSM_US_CHRONICLE_FACTS`` points at the pinned feed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from microcosm.build.us_runtime.target_sets import (
    CONGRESSIONAL_DISTRICT_PREDICATE,
    LOCAL_PRODUCT,
    NATIONAL_PRODUCT,
    STATE_CD_RULE,
    us_target_set_declaration,
    us_target_set_declarations,
)
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.target_sets import (
    select_target_set,
    target_set_census,
    undeclared_cells,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

#: The level labels of US targets: the ledger compiler's
#: ``ledger_geography_level`` and the ladder's ``geography_level``.
US_LEVELS = {"country", "state", "congressional_district", "county"}

#: Each surface on the pinned 41b2fb2 feed: (count, TargetRegistry.version).
PINNED_NATIONAL = {
    "full": (31_748, "4cc4cfb7a147"),
    "national_state": (6_102, "4dfe25b1238a"),
}
PINNED_LOCAL = {
    "state": (4_380, "0ee16ca42be5"),
    "totals": (255, "c6a0f757c0bc"),
    "full": (29_977, "b6c71f2e8505"),
    "state_cd": (26_425, "e0bb302e3add"),
    "state_county": (4_380, "0ee16ca42be5"),
}
#: The national_state surface by family and level: every compiled family's
#: country and state rows; the district file's 49 country and 2,601 state
#: rows and every district row are out.
PINNED_NATIONAL_STATE_CENSUS = {
    ("bea", "country"): 2,
    ("cbo", "country"): 5,
    ("census_population", "country"): 18,
    ("census_population", "state"): 918,
    ("cms_aca", "state"): 102,
    ("cms_medicaid", "country"): 3,
    ("cms_medicaid", "state"): 132,
    ("cms_medicare", "country"): 1,
    ("federal_reserve", "country"): 1,
    ("hhs_acf_liheap", "country"): 1,
    ("hhs_acf_tanf", "country"): 1,
    ("hhs_acf_tanf", "state"): 29,
    ("irs_soi", "country"): 392,
    ("irs_soi", "state"): 4_227,
    ("jct", "country"): 11,
    ("ssa", "country"): 9,
    ("ssa", "state"): 102,
    ("state_income_tax", "state"): 44,
    ("usda_snap", "country"): 2,
    ("usda_snap", "state"): 102,
}
#: The local default with the PUMA ladder's state and district totals
#: (--geographies state,cd): microcosm#1728's 4,867.
PINNED_LADDER_SHA256 = (
    "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8"
)
PINNED_LOCAL_DEFAULT_WITH_POPULATION = (4_867, "69b952ef181f")


def _load_tool(name: str):
    path = _TEST_PATHS.repository / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def local_tool():
    return _load_tool("build_us_acs_local_release")


@pytest.fixture(scope="module")
def national_tool():
    return _load_tool("build_us_fiscal_refresh_release")


def _pinned_feed() -> str:
    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip(
            "set MICROCOSM_US_CHRONICLE_FACTS to the pinned consumer-facts feed"
        )
    return feed


# ---------------------------------------------------------------------------
# The declarations agree with the tools
# ---------------------------------------------------------------------------


def test_the_products_and_modes_are_the_tools_options(
    national_tool, local_tool
) -> None:
    declarations = us_target_set_declarations()
    national = declarations.product(NATIONAL_PRODUCT)
    local = declarations.product(LOCAL_PRODUCT)

    assert tuple(declarations.products) == (NATIONAL_PRODUCT, LOCAL_PRODUCT)
    assert national.option == "--target-surface"
    assert national.mode_names == national_tool.TARGET_SURFACE_MODES
    assert national.default_mode == national_tool.TARGET_SURFACE_FULL
    assert local.option == "--soi-mode"
    assert local.mode_names == local_tool.SOI_MODES
    assert local.default_mode == local_tool.DEFAULT_SOI_MODE
    base = ["--ledger-facts", "facts.jsonl", "--out", "release"]
    assert national_tool._parse_args(base).target_surface == national.default_mode
    for product in (national, local):
        assert (_TEST_PATHS.repository / product.tool).is_file(), product.tool


def test_every_local_mode_declares_the_families_option_clauses(
    local_tool, tmp_path: Path
) -> None:
    """``--families`` turns the local clauses on by id; its default names all."""

    args = local_tool._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--feed",
            str(tmp_path / "facts.jsonl"),
        ]
    )
    assert args.families == "snap,medicaid,soi"
    assert args.geographies == "state,cd"
    for mode in local_tool.SOI_MODES:
        declaration = us_target_set_declaration(LOCAL_PRODUCT, mode)
        assert declaration.clause_ids == tuple(args.families.split(",")), mode
        population = declaration.built_rows("population")
        assert population.option == "--geographies"
        assert set(args.geographies.split(",")) <= set(population.levels), mode


def test_the_declared_names_are_the_ones_the_code_uses(local_tool) -> None:
    declarations = us_target_set_declarations()
    assert declarations.level_keys == ("ledger_geography_level", "geography_level")
    national_state = declarations.declaration(NATIONAL_PRODUCT, "national_state")
    assert national_state.compiled[0].exclude == (CONGRESSIONAL_DISTRICT_PREDICATE,)
    for mode in ("state", "state_county"):
        clause = us_target_set_declaration(LOCAL_PRODUCT, mode).clause("soi")
        (prefix_test,) = [t for t in clause.metadata if t.kind == "not_prefix"]
        assert (
            prefix_test.value
            == local_tool.CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX
        )
        assert prefix_test.value == local_tool.cd_surface.CD_FILE_RECORD_SET_SPEC_PREFIX
    state_cd = us_target_set_declaration(LOCAL_PRODUCT, "state_cd").clause("soi")
    assert state_cd.rule == STATE_CD_RULE
    county = us_target_set_declaration(LOCAL_PRODUCT, "state_county")
    assert county.built_rows("acs_households").family == (
        local_tool.county_surface.ACS_HOUSEHOLDS_FAMILY
    )
    for product in declarations.products.values():
        for declaration in product.modes.values():
            for clause in declaration.compiled:
                assert clause.levels is None or clause.levels <= US_LEVELS
            for built in declaration.built:
                assert set(built.levels.values()) <= US_LEVELS


def test_ladder_population_rows_have_the_declared_family_and_levels(
    local_tool,
) -> None:
    built = us_target_set_declaration(LOCAL_PRODUCT, "state").built_rows("population")
    level_keys = us_target_set_declarations().level_keys
    populations = {"state": {6: 39_000_000.0}, "cd": {601: 760_000.0}}

    specs = local_tool.ladder_population_target_specs(populations, ["state", "cd"])
    county = local_tool.county_surface.population_spec(
        "pop_county_06001", 1_600_000.0, period=local_tool.PERIOD
    )

    from microcosm.build.us_runtime.target_loss_weights import (
        target_geography_level,
    )
    from microcosm.calibrate.target_sets import target_level

    for spec in (*specs, county):
        assert target_level(spec, level_keys) == target_geography_level(spec)
    census = target_set_census((*specs, county), level_keys)
    assert census == {
        (built.family, built.levels["state"]): 1,
        (built.family, built.levels["cd"]): 1,
        (built.family, built.levels["county"]): 1,
    }


def test_the_soi_predicate_is_the_declared_clause(local_tool) -> None:
    """``soi_surface_predicate`` reads the declaration, whatever the mode."""

    from types import SimpleNamespace

    ht2 = "irs_soi.historic_table_2.state_broad_totals.v1"
    rows = {
        "state_band": {
            "state_fips": "06",
            "ledger_geography_level": "state",
            "ledger_layout_record_set_spec_id": ht2,
            "target_role": "soi_fiscal_distribution",
        },
        "district": {
            "state_fips": "06",
            "ledger_geography_level": "congressional_district",
            "ledger_layout_record_set_spec_id": "irs_soi.congressional_district_2022.x",
            "target_role": "aca_ptc_returns",
        },
        "national": {
            "ledger_geography_level": "country",
            "ledger_layout_record_set_spec_id": ht2,
        },
    }
    expected = {
        "state": {"state_band"},
        "state_county": {"state_band"},
        "totals": {"district"},
        "full": {"state_band", "district"},
        "state_cd": {"state_band", "district"},
    }
    for mode, kept in expected.items():
        predicate = local_tool.soi_surface_predicate(mode)
        assert {
            name
            for name, metadata in rows.items()
            if predicate(SimpleNamespace(metadata=metadata))
        } == kept, mode


# ---------------------------------------------------------------------------
# On the pinned feed: the same rows in the same order as before the change
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pinned_compile(local_tool):
    feed = _pinned_feed()
    registry, _substitutions = local_tool._compiled_fiscal_registry(feed)
    return feed, registry


def test_pinned_feed_levels_are_the_target_loss_levels(pinned_compile) -> None:
    """Every compiled target has the level the target-loss rows read
    (``target_loss_weights.target_geography_level``, microcosm#1729)."""

    from microcosm.build.us_runtime.target_loss_weights import (
        target_geography_level,
    )
    from microcosm.calibrate.target_sets import target_level

    _feed, registry = pinned_compile
    level_keys = us_target_set_declarations().level_keys

    assert all(
        target_level(spec, level_keys) == target_geography_level(spec)
        for spec in registry.specs
    )
    assert {target_level(spec, level_keys) for spec in registry.specs} == {
        "country",
        "state",
        "congressional_district",
    }


def test_pinned_feed_national_surfaces_are_unchanged(
    pinned_compile, national_tool
) -> None:
    _feed, registry = pinned_compile
    assert len(registry.specs) == PINNED_NATIONAL["full"][0]

    for mode, (count, version) in PINNED_NATIONAL.items():
        kept, receipt = national_tool._select_target_surface(registry.specs, mode)
        assert (len(kept), TargetRegistry(list(kept), country="us").version) == (
            count,
            version,
        ), mode
        assert receipt["calibrated_targets"] == count
        declaration = us_target_set_declaration(NATIONAL_PRODUCT, mode)
        assert undeclared_cells(declaration, kept) == {}
        if mode == "national_state":
            census = target_set_census(kept, declaration.level_keys)
            assert census == PINNED_NATIONAL_STATE_CENSUS
            assert receipt["dropped_congressional_district_targets"] == (
                len(registry.specs) - count
            )


def test_pinned_feed_local_surfaces_are_unchanged(pinned_compile, local_tool) -> None:
    feed, registry = pinned_compile
    families = ["snap", "medicaid", "soi"]

    for mode, (count, version) in PINNED_LOCAL.items():
        surface = local_tool.state_admin_surface(feed, families, soi_mode=mode)
        specs = surface.registry.specs
        assert (len(specs), surface.registry.version) == (count, version), mode
        declaration = us_target_set_declaration(LOCAL_PRODUCT, mode)
        assert undeclared_cells(declaration, specs) == {}, mode
        assert surface.soi_receipt["soi_mode"] == mode
        if mode != "state_cd":
            soi = sum(spec.family == "irs_soi" for spec in specs)
            assert surface.soi_receipt["counts"] == {"total": soi}

    # The shared selector, run directly on the compile (after the Medicaid
    # substitution), gives the same state surface as the tool.
    declaration = us_target_set_declaration(LOCAL_PRODUCT, "state")
    direct = select_target_set(declaration, registry.specs).specs
    assert (
        TargetRegistry(list(direct), country="us").version == PINNED_LOCAL["state"][1]
    )


def test_pinned_feed_local_default_with_ladder_population_is_4867(
    pinned_compile, local_tool
) -> None:
    """Runs when ``MICROCOSM_US_PUMA_LADDER_ARTIFACT`` names the pinned ladder."""

    ladder = os.environ.get("MICROCOSM_US_PUMA_LADDER_ARTIFACT", "")
    if not ladder or not Path(ladder).exists():
        pytest.skip("set MICROCOSM_US_PUMA_LADDER_ARTIFACT to the PUMA ladder")
    assert hashlib.sha256(Path(ladder).read_bytes()).hexdigest() == PINNED_LADDER_SHA256
    feed, _registry = pinned_compile
    geographies = ["state", "cd"]

    surface = local_tool.state_admin_surface(
        feed, ["snap", "medicaid", "soi"], soi_mode="state"
    )
    populations = local_tool.ladder_population(
        Path(ladder), sorted(set(geographies) | {"cd"})
    )
    joint = local_tool.require_one_label_per_geography(
        surface.registry, populations, geographies
    )

    assert (len(joint.specs), joint.version) == PINNED_LOCAL_DEFAULT_WITH_POPULATION
    declaration = us_target_set_declaration(LOCAL_PRODUCT, "state")
    assert undeclared_cells(declaration, joint.specs) == {}
