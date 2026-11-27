"""The declared target sets as a typed resource of the spec engine (microcosm#1755).

A country ships its declared target sets as ``spec/target_sets.yaml``, kind
``target_sets``, schema ``target_sets.schema.json``. The spec engine reads the
file strictly, checks its form, and separates the descriptions and rationales
(documentation) from the content a selection reads (normative). These tests
hold that split, the content hash that a cache key can use, and the narrow
loader that the country modules read through.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from microcosm.build import country_spec
from microcosm.build.country_spec import (
    load_country_target_sets,
    typed_resource_content_sha256,
)
from microcosm.build.spec_engine import load_schema_registry
from microcosm.build.spec_engine.canonical import normalize_and_project
from microcosm.build.spec_engine.errors import SpecParseError, SpecValidationError
from microcosm.build.spec_engine.model import ResourceKind, thaw_json
from microcosm.build.spec_engine.schemas import (
    SCHEMA_ID_BY_KIND,
    assert_schema_id_allowed,
)
from microcosm.calibrate.target_sets import (
    TargetSetDeclarationError,
    load_target_set_declarations,
)

SCHEMA = "target_sets.schema.json"

DECLARATIONS = {
    "version": 1,
    "country": "zz",
    "description": "What this file declares.",
    "level_rule": {
        "keys": ["level", "geo"],
        "values": ["state", "district", "country"],
        "conflict": "refuse",
        "rationale": "Why a level is read this way.",
    },
    "products": [
        {
            "id": "national",
            "tool": "tools/national.py",
            "option": "--surface",
            "default_mode": "all",
            "modes": [
                {
                    "id": "all",
                    "description": "Every compiled target.",
                    "compiled": [
                        {
                            "id": "all",
                            "families": "*",
                            "levels": "*",
                            "exclude": ["district"],
                            "rationale": "Why every row.",
                        }
                    ],
                }
            ],
        },
        {
            "id": "local",
            "tool": "tools/local.py",
            "option": "--mode",
            "default_mode": "state",
            "modes": [
                {
                    "id": "state",
                    "description": "State benefit and state tax rows.",
                    "compiled": [
                        {
                            "id": "benefit",
                            "families": ["benefit"],
                            "levels": ["state"],
                            "metadata": {"state": {"present": True}},
                            "rationale": "Why the benefit rows.",
                        },
                        {
                            "id": "tax",
                            "families": ["tax", "levy"],
                            "levels": ["state", "district"],
                            "metadata": {
                                "state": {"present": True},
                                "role": {"not_equals": "band"},
                            },
                            "rule": "shares",
                            "rationale": "Why the tax rows.",
                        },
                    ],
                    "built": [
                        {
                            "id": "population",
                            "family": "population",
                            "option": "--geographies",
                            "levels": {"state": "state", "cd": "district"},
                            "source": "a ladder",
                            "rationale": "Why the population rows.",
                        }
                    ],
                },
                {
                    "id": "totals",
                    "description": "The state totals.",
                    "compiled": [
                        {
                            "id": "tax",
                            "families": ["tax"],
                            "levels": ["state"],
                            "rationale": "Why the totals.",
                        }
                    ],
                },
            ],
        },
    ],
}

_DELETE = object()


def _edited(**changes) -> dict:
    """``DECLARATIONS`` with each ``a__b__c`` path set to its value.

    A step into a list is an index, or the ``id`` of one of its objects.
    """

    raw = copy.deepcopy(DECLARATIONS)
    for path, value in changes.items():
        node = raw
        *parents, leaf = path.split("__")
        for key in parents:
            if not isinstance(node, list):
                node = node[key]
            elif key.isdigit():
                node = node[int(key)]
            else:
                (node,) = [item for item in node if item["id"] == key]
        if value is _DELETE:
            del node[leaf]
        else:
            node[leaf] = value
    return raw


def _write(tmp_path: Path, raw: object, *, text: str | None = None) -> Path:
    """A ``target_sets.yaml`` holding ``raw`` (JSON is YAML 1.2) or ``text``."""

    path = tmp_path / "target_sets.yaml"
    path.write_text(
        json.dumps(raw, indent=1) if text is None else text, encoding="utf-8"
    )
    return path


def _content(tmp_path: Path, **changes) -> str:
    return typed_resource_content_sha256(_write(tmp_path, _edited(**changes)), SCHEMA)


def _package(tmp_path: Path, raw: object, *, text: str | None = None) -> Path:
    """A country package ``zz`` that declares only its target sets."""

    root = tmp_path / "zz"
    (root / "spec").mkdir(parents=True, exist_ok=True)
    (root / "country_package.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "country": "zz",
                "resources": [
                    {
                        "path": "spec/target_sets.yaml",
                        "kind": "target_sets",
                        "schema_id": SCHEMA,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write(root / "spec", raw, text=text)
    return root


_STATE = "products__local__modes__state"


# ---------------------------------------------------------------------------
# The kind and its schema
# ---------------------------------------------------------------------------


def test_the_target_sets_kind_has_one_schema() -> None:
    assert ResourceKind.TARGET_SETS.value == "target_sets"
    assert SCHEMA_ID_BY_KIND["target_sets"] == SCHEMA
    assert "target_sets" in country_spec.ALLOWED_COUNTRY_RESOURCE_KINDS
    assert_schema_id_allowed("target_sets", SCHEMA)
    with pytest.raises(SpecValidationError, match="cannot use schema_id"):
        assert_schema_id_allowed("target_sets", "legacy_json")
    with pytest.raises(SpecValidationError, match="cannot use schema_id"):
        assert_schema_id_allowed("legacy_json", SCHEMA)
    load_schema_registry().validate(DECLARATIONS, SCHEMA)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": 2}, "1 was expected"),
        ({"extra": True}, "Additional properties"),
        # The level rule has one form; the short form is not in the grammar.
        ({"level_keys": ["level", "geo"]}, "Additional properties"),
        ({"level_rule": _DELETE}, "'level_rule' is a required property"),
        ({"level_rule__conflict": "maybe"}, "is not one of"),
        ({"level_rule__keys": []}, "should be non-empty"),
        ({"level_rule__values": []}, "is not valid under any of the given schemas"),
        ({"level_rule__rationale": _DELETE}, "'rationale' is a required property"),
        ({"products": []}, "should be non-empty"),
        # Products and modes are lists with ids, not objects keyed by name.
        ({"products": {"national": {}}}, "is not of type 'array'"),
        ({"products__local__id": _DELETE}, "'id' is a required property"),
        ({"products__local__modes": []}, "should be non-empty"),
        ({f"{_STATE}__compiled": []}, "should be non-empty"),
        ({f"{_STATE}__compiled__0__families": []}, "is not valid under any"),
        ({f"{_STATE}__compiled__0__levels": ["state", "state"]}, "is not valid under"),
        ({f"{_STATE}__compiled__0__rationale": _DELETE}, "'rationale' is a required"),
        ({f"{_STATE}__compiled__0__metadata": {"state": {"in": 1}}}, "is not valid"),
        (
            {f"{_STATE}__compiled__0__metadata": {"state": {"present": "yes"}}},
            "is not valid",
        ),
        ({f"{_STATE}__compiled__0__left": 1}, "Additional properties"),
        ({f"{_STATE}__built__0__levels": {}}, "should be non-empty"),
        ({f"{_STATE}__built__0__source": _DELETE}, "'source' is a required property"),
    ],
)
def test_the_schema_refuses_a_malformed_declaration(changes, message) -> None:
    with pytest.raises(SpecValidationError, match=message):
        load_schema_registry().validate(_edited(**changes), SCHEMA)


# ---------------------------------------------------------------------------
# Normative content and documentation
# ---------------------------------------------------------------------------


def _leaves(node: object, path: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        found: dict[str, object] = {}
        for key, value in node.items():
            found.update(_leaves(value, f"{path}/{key}"))
        return found
    if isinstance(node, list):
        found = {}
        for index, value in enumerate(node):
            found.update(_leaves(value, f"{path}/{index}"))
        return found
    return {path: node}


def test_the_descriptions_and_rationales_are_documentation() -> None:
    registry = load_schema_registry()
    _normalized, projections = normalize_and_project(
        registry.validate_and_inject_defaults(DECLARATIONS, SCHEMA),
        schema_id=SCHEMA,
        registry=registry,
    )
    surfaces = thaw_json(projections)

    documentation = _leaves(surfaces["documentation"])
    normative = _leaves(surfaces["normative"])
    everything = _leaves(DECLARATIONS)

    assert {path.rsplit("/", 1)[1] for path in documentation} == {
        "description",
        "rationale",
    }
    # Top, level rule, three modes, four clauses and one built entry.
    assert len(documentation) == 10
    assert not any(
        path.rsplit("/", 1)[1] in {"description", "rationale"} for path in normative
    )
    assert len(normative) + len(documentation) == len(everything)
    # Only these two surfaces hold a field of a declaration.
    assert {surface for surface, fragment in surfaces.items() if _leaves(fragment)} == {
        "normative",
        "documentation",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"description": "Other words."},
        {"description": _DELETE},
        {"level_rule__rationale": "Another reason."},
        {f"{_STATE}__description": "Other words."},
        {f"{_STATE}__compiled__0__rationale": "Another reason."},
        {f"{_STATE}__built__0__rationale": "Another reason."},
        # Names in a list of families or levels are a set.
        {f"{_STATE}__compiled__1__families": ["levy", "tax"]},
        {f"{_STATE}__compiled__1__levels": ["district", "state"]},
        {"level_rule__values": ["country", "district", "state"]},
        # A clause's metadata tests are an object: every one must pass.
        {
            f"{_STATE}__compiled__1__metadata": {
                "role": {"not_equals": "band"},
                "state": {"present": True},
            }
        },
        {f"{_STATE}__built__0__levels": {"cd": "district", "state": "state"}},
    ],
)
def test_the_content_hash_does_not_read_prose_or_set_order(tmp_path, changes) -> None:
    assert _content(tmp_path, **changes) == _content(tmp_path)


def test_the_content_hash_does_not_read_the_layout_of_the_file(tmp_path) -> None:
    import yaml

    def reversed_fields(node):
        if isinstance(node, dict):
            return {key: reversed_fields(node[key]) for key in reversed(node)}
        if isinstance(node, list):
            return [reversed_fields(item) for item in node]
        return node

    as_json = _content(tmp_path)
    block_yaml = yaml.safe_dump(DECLARATIONS, sort_keys=False, width=60)
    assert block_yaml.lstrip()[0] != "{"
    flow_json = json.dumps(reversed_fields(DECLARATIONS), indent=7)
    for text in ("# A comment.\n" + block_yaml, flow_json):
        assert (
            typed_resource_content_sha256(_write(tmp_path, None, text=text), SCHEMA)
            == as_json
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"country": "yy"},
        {"level_rule__keys": ["geo", "level"]},
        {"level_rule__values": ["state", "district"]},
        {"level_rule__values": "*"},
        {"level_rule__conflict": "first"},
        {"products__local__tool": "tools/other.py"},
        {"products__local__option": "--other"},
        {"products__local__default_mode": "totals"},
        {"products__local__id": "area"},
        {f"{_STATE}__id": "states"},
        {f"{_STATE}__compiled__0__id": "benefits"},
        {f"{_STATE}__compiled__0__families": ["benefit", "tax"]},
        {f"{_STATE}__compiled__0__families": "*"},
        {f"{_STATE}__compiled__0__levels": ["state", "district"]},
        {f"{_STATE}__compiled__0__levels": "*"},
        {f"{_STATE}__compiled__0__metadata": {"state": {"present": False}}},
        {f"{_STATE}__compiled__1__metadata__role": {"equals": "band"}},
        {f"{_STATE}__compiled__0__exclude": ["a_predicate"]},
        {f"{_STATE}__compiled__0__rule": "a_rule"},
        {f"{_STATE}__compiled__1__rule": _DELETE},
        {f"{_STATE}__built__0__family": "people"},
        {f"{_STATE}__built__0__levels": {"state": "state"}},
        {f"{_STATE}__built__0__option": "--areas"},
        {f"{_STATE}__built__0__source": "another ladder"},
        {f"{_STATE}__built": _DELETE},
    ],
)
def test_the_content_hash_reads_every_field_a_selection_can_use(
    tmp_path, changes
) -> None:
    assert _content(tmp_path, **changes) != _content(tmp_path)


def test_the_content_hash_reads_the_order_of_clauses_modes_and_products(
    tmp_path,
) -> None:
    packaged = _content(tmp_path)

    raw = _edited()
    raw["products"][1]["modes"][0]["compiled"].reverse()
    assert typed_resource_content_sha256(_write(tmp_path, raw), SCHEMA) != packaged

    raw = _edited()
    raw["products"][1]["modes"].reverse()
    assert typed_resource_content_sha256(_write(tmp_path, raw), SCHEMA) != packaged

    raw = _edited()
    raw["products"].reverse()
    assert typed_resource_content_sha256(_write(tmp_path, raw), SCHEMA) != packaged


# ---------------------------------------------------------------------------
# The narrow loader
# ---------------------------------------------------------------------------


def test_the_narrow_loader_gives_the_object_the_reader_takes(tmp_path) -> None:
    document = load_country_target_sets(_package(tmp_path, DECLARATIONS))

    declarations = load_target_set_declarations(document)
    assert declarations.country == "zz"
    assert tuple(declarations.products) == ("national", "local")
    assert declarations.product("local").mode_names == ("state", "totals")
    assert declarations.declaration("local", "state").clause_ids == ("benefit", "tax")
    assert declarations.level_rule.keys == ("level", "geo")
    assert declarations.level_rule.values == {"state", "district", "country"}
    # The names of a set are in canonical order in the loaded object.
    assert document["level_rule"]["values"] == ["country", "district", "state"]


def test_a_package_without_target_sets_gives_none() -> None:
    assert load_country_target_sets("be") is None


def test_the_file_is_read_strictly(tmp_path) -> None:
    text = json.dumps(DECLARATIONS)
    with pytest.raises(SpecParseError, match="duplicate mapping key"):
        load_country_target_sets(
            _package(tmp_path, None, text='{"version": 1, ' + text[1:])
        )
    with pytest.raises(SpecParseError):
        load_country_target_sets(
            _package(tmp_path, None, text=text[:-1] + ', "extra": .nan}')
        )
    with pytest.raises(SpecValidationError, match="Additional properties"):
        load_country_target_sets(_package(tmp_path, _edited(extra=True)))


def test_the_reader_refuses_what_the_schema_cannot_state(tmp_path) -> None:
    # A list holds literal names, so ["*"] would match only a name "*".
    with pytest.raises(TargetSetDeclarationError, match="alone, not a list"):
        load_country_target_sets(
            _package(tmp_path, _edited(**{f"{_STATE}__compiled__0__families": ["*"]}))
        )
    with pytest.raises(TargetSetDeclarationError, match="not a declared mode"):
        load_country_target_sets(
            _package(tmp_path, _edited(products__local__default_mode="county"))
        )
    with pytest.raises(TargetSetDeclarationError, match="repeats a clause"):
        load_country_target_sets(
            _package(tmp_path, _edited(**{f"{_STATE}__compiled__1__id": "benefit"}))
        )
    twice = _edited()
    twice["products"][1]["modes"].append(
        copy.deepcopy(twice["products"][1]["modes"][0])
    )
    with pytest.raises(TargetSetDeclarationError, match="repeats the mode 'state'"):
        load_country_target_sets(_package(tmp_path, twice))
    with pytest.raises(ValueError, match="lives in the 'zz' package"):
        load_country_target_sets(_package(tmp_path, _edited(country="yy")))


@pytest.mark.parametrize("country", ["uk", "us"])
def test_each_packaged_file_is_a_typed_resource_that_loads(country) -> None:
    """The packaged declarations are a typed resource, and they load through
    the narrow loader. ``test_uk_solve_scope.py`` and ``test_us_spec_bundle.py``
    hold the loader to the object and the content hash of the whole bundle."""

    document = load_country_target_sets(country)
    assert document is not None
    assert load_target_set_declarations(document).country == country

    root = Path(country_spec.__file__).parent / country
    manifest = json.loads((root / "country_package.json").read_text(encoding="utf-8"))
    (row,) = [row for row in manifest["resources"] if row["kind"] == "target_sets"]
    assert row == {
        "path": "spec/target_sets.yaml",
        "kind": "target_sets",
        "schema_id": SCHEMA,
    }
    assert (root / row["path"]).is_file()
    assert not (root / "target_sets.json").exists()
