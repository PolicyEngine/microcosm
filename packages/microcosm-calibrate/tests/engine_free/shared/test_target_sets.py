"""Declared target sets: the shared reader and selector (microcosm#1728)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from microcosm.calibrate.target_sets import (
    UNSPECIFIED_LEVEL,
    TargetSetDeclarationError,
    clause_predicate,
    load_target_set_declarations,
    select_by_levels,
    select_target_set,
    target_level,
    target_set_census,
    undeclared_cells,
)


def _spec(name: str, family: str, **metadata) -> SimpleNamespace:
    return SimpleNamespace(name=name, family=family, metadata=metadata)


SPECS = (
    _spec("nation_tax", "tax", level="country"),
    _spec("ca_tax", "tax", level="state", state="06", role="band"),
    _spec("ca_tax_total", "tax", level="state", state="06", role="total"),
    _spec("ca_benefit", "benefit", level="state", state="06"),
    _spec("ca01_tax", "tax", level="district", state="06", record="district_file.ca01"),
    _spec("ca_tax_file", "tax", level="state", state="06", record="district_file.ca"),
    _spec("unlabelled", "tax", geo="state", state="06"),
)

DECLARATIONS = {
    "version": 1,
    "country": "test",
    "level_keys": ["level", "geo"],
    "products": {
        "national": {
            "tool": "tools/national.py",
            "option": "--surface",
            "default_mode": "all",
            "modes": {
                "all": {
                    "description": "Every compiled target.",
                    "compiled": [
                        {"id": "all", "families": "*", "levels": "*", "rationale": "r"}
                    ],
                },
                "no_district": {
                    "description": "Every target but the district ones.",
                    "compiled": [
                        {
                            "id": "rows",
                            "families": "*",
                            "levels": "*",
                            "exclude": ["district"],
                            "rationale": "r",
                        }
                    ],
                },
            },
        },
        "local": {
            "tool": "tools/local.py",
            "option": "--mode",
            "default_mode": "state",
            "modes": {
                "state": {
                    "description": "State benefit and state tax rows.",
                    "compiled": [
                        {
                            "id": "benefit",
                            "families": ["benefit"],
                            "levels": ["state"],
                            "metadata": {"state": {"present": True}},
                            "rationale": "r",
                        },
                        {
                            "id": "tax",
                            "families": ["tax"],
                            "levels": ["state"],
                            "metadata": {
                                "state": {"present": True},
                                "role": {"not_equals": "band"},
                            },
                            "rationale": "r",
                        },
                    ],
                    "built": [
                        {
                            "id": "population",
                            "family": "population",
                            "option": "--geographies",
                            "levels": {"state": "state", "cd": "district"},
                            "source": "a ladder",
                            "rationale": "r",
                        }
                    ],
                },
                "shares": {
                    "description": "State tax rows and district shares of them.",
                    "compiled": [
                        {
                            "id": "tax",
                            "families": ["tax"],
                            "levels": ["state", "district"],
                            "metadata": {"state": {"present": True}},
                            "rule": "shares",
                            "rationale": "r",
                        }
                    ],
                },
            },
        },
    },
}


def _declarations(**changes):
    raw = copy.deepcopy(DECLARATIONS)
    for path, value in changes.items():
        node = raw
        *parents, leaf = path.split("__")
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if value is _DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return load_target_set_declarations(raw)


_DELETE = object()


def _names(specs) -> list[str]:
    return [spec.name for spec in specs]


def test_declarations_load_from_a_file_and_name_products_and_modes(tmp_path) -> None:
    path = tmp_path / "target_sets.json"
    path.write_text(json.dumps(DECLARATIONS), encoding="utf-8")

    declarations = load_target_set_declarations(path)

    assert declarations.country == "test"
    assert declarations.level_keys == ("level", "geo")
    assert tuple(declarations.products) == ("national", "local")
    local = declarations.product("local")
    assert local.option == "--mode"
    assert local.default_mode == "state"
    assert local.mode_names == ("state", "shares")
    state = declarations.declaration("local", "state")
    assert state.clause_ids == ("benefit", "tax")
    assert state.built_rows("population").levels == {"state": "state", "cd": "district"}
    with pytest.raises(KeyError, match="no mode 'county'"):
        local.mode("county")
    with pytest.raises(KeyError, match="no product 'regional'"):
        declarations.product("regional")
    with pytest.raises(KeyError, match="no compiled clause 'snap'"):
        state.clause("snap")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": 2}, "version must be 1"),
        ({"extra": True}, "unknown keys"),
        ({"level_keys": []}, "level_keys"),
        ({"products__local__default_mode": "county"}, "not a declared mode"),
        ({"products__local__modes__state__compiled": []}, "at least one clause"),
        (
            {"products__local__modes__state__compiled__0__families": []},
            "nonempty list",
        ),
        (
            {"products__local__modes__state__compiled__0__levels": ["state", "state"]},
            "repeats a name",
        ),
        (
            {"products__local__modes__state__compiled__1__id": "benefit"},
            "repeats a clause or built id",
        ),
        ({"products__local__modes__state__compiled__0__rationale": _DELETE}, "missing"),
        (
            {
                "products__local__modes__state__compiled__0__metadata": {
                    "state": {"in": 1}
                }
            },
            "unknown test",
        ),
        (
            {
                "products__local__modes__state__compiled__0__metadata": {
                    "state": {"present": "yes"}
                }
            },
            "present takes true or false",
        ),
        (
            {"products__local__modes__state__built__0__levels": {}},
            "map each option value",
        ),
    ],
)
def test_a_malformed_declaration_is_refused(changes, message) -> None:
    with pytest.raises(TargetSetDeclarationError, match=message):
        _declarations(**changes)


def test_a_level_is_the_first_truthy_level_key() -> None:
    keys = ("level", "geo")

    assert target_level(_spec("a", "f", level="state", geo="district"), keys) == "state"
    assert target_level(_spec("b", "f", level="", geo="district"), keys) == "district"
    assert target_level(_spec("c", "f"), keys) == UNSPECIFIED_LEVEL


def test_wildcards_keep_every_spec_in_compile_order() -> None:
    declaration = _declarations().declaration("national", "all")

    selection = select_target_set(declaration, SPECS)

    assert selection.specs == SPECS
    assert dict(selection.clause_counts) == {"all": len(SPECS)}


def test_an_exclusion_predicate_drops_the_specs_it_accepts() -> None:
    declaration = _declarations().declaration("national", "no_district")

    def district(spec) -> bool:
        return spec.metadata.get("level") == "district" or "district_file" in str(
            spec.metadata.get("record", "")
        )

    selection = select_target_set(declaration, SPECS, predicates={"district": district})

    assert _names(selection.specs) == [
        "nation_tax",
        "ca_tax",
        "ca_tax_total",
        "ca_benefit",
        "unlabelled",
    ]
    with pytest.raises(ValueError, match="excludes by \\['district'\\]"):
        select_target_set(declaration, SPECS)


def test_clauses_keep_family_level_and_metadata_matches_in_clause_order() -> None:
    declaration = _declarations().declaration("local", "state")

    selection = select_target_set(declaration, SPECS)

    # Clause order (benefit, then tax), compile order inside each clause. The
    # unlabelled spec's level comes from the second level key.
    assert _names(selection.specs) == [
        "ca_benefit",
        "ca_tax_total",
        "ca_tax_file",
        "unlabelled",
    ]
    assert dict(selection.clause_counts) == {"benefit": 1, "tax": 3}


def test_clause_ids_turn_on_only_some_clauses() -> None:
    declaration = _declarations().declaration("local", "state")

    selection = select_target_set(declaration, SPECS, clause_ids=["tax"])

    assert _names(selection.specs) == ["ca_tax_total", "ca_tax_file", "unlabelled"]
    with pytest.raises(KeyError, match="no compiled clause 'snap'"):
        select_target_set(declaration, SPECS, clause_ids=["tax", "snap"])


def test_a_rule_takes_its_clause_specs_and_returns_specs_and_a_receipt() -> None:
    declaration = _declarations().declaration("local", "shares")
    seen = []

    def shares(candidates):
        seen.append(_names(candidates))
        return candidates[:2], {"rule": "shares", "in": len(candidates)}

    selection = select_target_set(declaration, SPECS, rules={"shares": shares})

    assert seen == [["ca_tax", "ca_tax_total", "ca01_tax", "ca_tax_file", "unlabelled"]]
    assert _names(selection.specs) == ["ca_tax", "ca_tax_total"]
    assert dict(selection.rule_receipts) == {"tax": {"rule": "shares", "in": 5}}
    with pytest.raises(ValueError, match="names rule 'shares'"):
        select_target_set(declaration, SPECS)


def test_a_spec_that_two_clauses_keep_is_refused() -> None:
    declarations = copy.deepcopy(DECLARATIONS)
    clauses = declarations["products"]["local"]["modes"]["state"]["compiled"]
    clauses[0]["families"] = ["benefit", "tax"]
    declaration = load_target_set_declarations(declarations).declaration(
        "local", "state"
    )

    with pytest.raises(ValueError, match="both keep 'ca_tax_total'"):
        select_target_set(declaration, SPECS)


def test_metadata_tests_match_their_definitions() -> None:
    declarations = copy.deepcopy(DECLARATIONS)
    clause = declarations["products"]["local"]["modes"]["state"]["compiled"][1]
    clause["levels"] = "*"
    clause["metadata"] = {
        "record": {"not_prefix": "district_file."},
        "role": {"equals": "total"},
    }
    declaration = load_target_set_declarations(declarations).declaration(
        "local", "state"
    )
    specs = (
        _spec("kept", "tax", record="table_2.ca", role="total"),
        _spec("district_file", "tax", record="district_file.ca", role="total"),
        _spec("empty_record", "tax", record="", role="total"),
        _spec("no_record", "tax", role="total"),
        _spec("band", "tax", record="table_2.ca", role="band"),
        _spec("no_role", "tax", record="table_2.ca"),
    )

    assert _names(select_target_set(declaration, specs, clause_ids=["tax"]).specs) == [
        "kept"
    ]


def test_a_clause_predicate_can_skip_the_family_test() -> None:
    declaration = _declarations().declaration("local", "state")
    with_family = clause_predicate(declaration, "tax")
    without_family = clause_predicate(declaration, "tax", test_family=False)
    plain = SimpleNamespace(metadata={"level": "state", "state": "06"})

    assert without_family(plain)
    assert with_family(_spec("x", "tax", level="state", state="06"))
    assert not with_family(_spec("x", "benefit", level="state", state="06"))


def test_census_and_undeclared_cells() -> None:
    declaration = _declarations().declaration("local", "state")
    specs = (
        *SPECS,
        _spec("pop_ca", "population", geo="state"),
        _spec("pop_ca01", "population", geo="district"),
        _spec("pop_county", "population", geo="county"),
    )

    census = target_set_census(specs, declaration.level_keys)

    assert census[("tax", "state")] == 4
    assert census[("population", "district")] == 1
    assert undeclared_cells(declaration, specs) == {
        ("population", "county"): 1,
        ("tax", "country"): 1,
        ("tax", "district"): 1,
    }


# ---------------------------------------------------------------------------
# Level rules and selection by level (microcosm#1759)
# ---------------------------------------------------------------------------

LEVELS = ("country", "region", "la")


def _level_declarations(**rule) -> dict:
    return {
        "version": 1,
        "country": "test",
        "level_rule": {
            "keys": ["level", "geo"],
            "values": list(LEVELS),
            "conflict": "refuse",
            "rationale": "r",
            **rule,
        },
        "products": {
            "full": {
                "tool": "tools/full.py",
                "option": "--levels",
                "default_mode": "levels",
                "modes": {
                    "levels": {
                        "description": "One clause for each level.",
                        "compiled": [
                            {
                                "id": level,
                                "families": "*",
                                "levels": [level],
                                "rationale": "r",
                            }
                            for level in LEVELS
                        ],
                    }
                },
            }
        },
    }


def _level_mode(**rule):
    return load_target_set_declarations(_level_declarations(**rule)).declaration(
        "full", "levels"
    )


def _registry(*rows):
    from microcosm.calibrate.registry import TargetRegistry, TargetSpec

    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                measure="household_weight",
                value=1.0 + index,
                period=2025,
                source="test",
                family=family,
                metadata=metadata,
            )
            for index, (name, family, metadata) in enumerate(rows)
        ],
        country="test",
    )


def test_a_level_rule_reads_the_first_key_and_refuses_what_it_cannot_place() -> None:
    rule = _level_mode().level_rule
    assert rule.keys == ("level", "geo")
    assert rule.values == frozenset(LEVELS)
    assert rule.level_of(_spec("a", "tax", level="la")) == "la"
    assert rule.level_of(_spec("b", "tax", geo="region")) == "region"
    assert rule.level_of(_spec("c", "tax", level="la", geo="la")) == "la"
    assert rule.level_of(_spec("d", "tax", level="", geo="country")) == "country"
    with pytest.raises(ValueError, match="'e' has contradictory geography levels"):
        rule.level_of(_spec("e", "tax", level="la", geo="region"))
    with pytest.raises(ValueError, match="'f' has unsupported geography level None"):
        rule.level_of(_spec("f", "tax"))
    with pytest.raises(ValueError, match="unsupported geography level ''"):
        rule.level_of(_spec("g", "tax", level=None, geo=""))
    with pytest.raises(ValueError, match="unsupported geography level 'nation'"):
        rule.level_of(_spec("h", "tax", geo="nation"))


def test_an_open_rule_and_the_first_conflict_rule() -> None:
    first = _level_mode(conflict="first").level_rule
    assert first.level_of(_spec("a", "tax", level="la", geo="region")) == "la"
    open_rule = _level_mode(values="*").level_rule
    assert open_rule.values is None
    assert open_rule.level_of(_spec("b", "tax")) == UNSPECIFIED_LEVEL
    assert open_rule.level_of(_spec("c", "tax", geo="nation")) == "nation"
    with pytest.raises(ValueError, match="contradictory"):
        open_rule.level_of(_spec("d", "tax", level="la", geo="region"))
    # The level_keys form reads exactly as target_level does.
    keys_form = _declarations().declaration("national", "all")
    assert keys_form.level_rule.values is None
    assert keys_form.level_rule.conflict == "first"
    for spec in SPECS:
        assert keys_form.level_of(spec) == target_level(spec, ("level", "geo"))


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"conflict": "maybe"}, "must be one of"),
        ({"keys": []}, "nonempty list of metadata keys"),
        ({"keys": ["level", "level"]}, "repeats a key"),
        ({"values": []}, "nonempty list"),
        ({"rationale": ""}, "nonempty string"),
        ({"extra": 1}, "unknown keys"),
    ],
)
def test_a_malformed_level_rule_is_refused(rule, message) -> None:
    with pytest.raises(TargetSetDeclarationError, match=message):
        load_target_set_declarations(_level_declarations(**rule))


def test_a_declaration_gives_exactly_one_level_form() -> None:
    both = _level_declarations()
    both["level_keys"] = ["level"]
    with pytest.raises(TargetSetDeclarationError, match="exactly one of"):
        load_target_set_declarations(both)
    neither = _level_declarations()
    del neither["level_rule"]
    with pytest.raises(TargetSetDeclarationError, match="exactly one of"):
        load_target_set_declarations(neither)


def test_select_by_levels_keeps_compile_order_and_lists_every_target() -> None:
    registry = _registry(
        ("la_1", "tax", {"level": "la"}),
        ("uk", "tax", {"geo": "country"}),
        ("north", "benefit", {"level": "region", "geo": "region"}),
        ("la_2", "benefit", {"geo": "la"}),
    )
    selection = select_by_levels(registry, _level_mode(), ["la", "country"])
    assert [spec.name for spec in selection.registry.specs] == ["la_1", "uk", "la_2"]
    receipt = selection.receipt
    assert receipt["schema"] == "microcosm.calibrate.target-selection.v1"
    assert receipt["selector"] == {
        "geography_levels": ["country", "la"],
        "explicit": True,
    }
    assert receipt["source_registry_version"] == registry.version
    assert receipt["selected_registry_version"] == selection.registry.version
    assert [(row["name"], row["reason"]) for row in receipt["included"]] == [
        ("la_1", "geography_selected"),
        ("uk", "geography_selected"),
        ("la_2", "geography_selected"),
    ]
    assert [(row["name"], row["geography_level"]) for row in receipt["excluded"]] == [
        ("north", "region")
    ]
    everything = select_by_levels(registry, _level_mode())
    assert everything.registry.version == registry.version
    assert everything.receipt["selector"] == {
        "geography_levels": None,
        "explicit": False,
    }
    assert {row["reason"] for row in everything.receipt["included"]} == {
        "all_geographies"
    }
    assert everything.sha256 != selection.sha256
    receipt["included"].clear()
    assert selection.receipt["included"]


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ("la", "nonempty sequence of levels"),
        ([], "nonempty literal names"),
        ([" la"], "nonempty literal names"),
        (["la", "la"], "must not repeat levels"),
        (["region"], r"unknown geography levels \['region'\]"),
        (["nowhere", "region"], r"unknown geography levels \['nowhere', 'region'\]"),
    ],
)
def test_select_by_levels_refuses_a_bad_request(levels, message) -> None:
    registry = _registry(("la_1", "tax", {"level": "la"}))
    with pytest.raises(ValueError, match=message):
        select_by_levels(registry, _level_mode(), levels)


def test_select_by_levels_refuses_an_empty_selection_and_a_non_registry() -> None:
    with pytest.raises(ValueError, match="nonempty selected problem"):
        select_by_levels(_registry(), _level_mode())
    with pytest.raises(TypeError, match="requires TargetRegistry"):
        select_by_levels([], _level_mode())
    with pytest.raises(ValueError, match="contradictory"):
        select_by_levels(
            _registry(("x", "tax", {"level": "la", "geo": "region"})), _level_mode()
        )


def test_select_by_levels_needs_a_level_mode() -> None:
    registry = _registry(("la_1", "tax", {"level": "la"}))
    with pytest.raises(TargetSetDeclarationError, match="closed set of values"):
        select_by_levels(registry, _level_mode(values="*"))
    family_clause = _level_declarations()
    mode = family_clause["products"]["full"]["modes"]["levels"]
    mode["compiled"][0]["families"] = ["tax"]
    with pytest.raises(TargetSetDeclarationError, match="not a level clause"):
        select_by_levels(
            registry,
            load_target_set_declarations(family_clause).declaration("full", "levels"),
        )
    missing = _level_declarations()
    missing["products"]["full"]["modes"]["levels"]["compiled"].pop()
    with pytest.raises(TargetSetDeclarationError, match="are not the levels"):
        select_by_levels(
            registry,
            load_target_set_declarations(missing).declaration("full", "levels"),
        )
