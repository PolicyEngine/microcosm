"""Exact contracts for the closed finite row-scope algebra."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from microcosm.build.spec_engine.scope_algebra import (
    CanonicalScope,
    ClosedScopeRegistry,
    ScopeAlgebraError,
)


@pytest.fixture
def registry() -> ClosedScopeRegistry:
    return ClosedScopeRegistry(
        predicate_space="fixture_rows",
        universe=("a", "b", "c", "d"),
        scopes={
            "left": ("a", "b"),
            "right": ("c", "d"),
            "outer": {"any_of": ["left", {"atoms": ["c"]}]},
            "left_without_b": {"all_of": ["left", {"not": {"atoms": ["b"]}}]},
        },
        primitives={
            "channel_is": {
                "survey": {"a", "c"},
                "synthetic": {"b", "d"},
            },
            "support_role_is": {
                "donor": {"a", "b"},
                "recipient": {"c", "d"},
            },
            "column_predicate": [
                (
                    {
                        "entity": "person",
                        "column": "status",
                        "op": "eq",
                        "value": {"eligible": [True, 1]},
                    },
                    {"a", "d"},
                ),
                (
                    {
                        "entity": "person",
                        "column": "income",
                        "op": "not_null",
                    },
                    {"a", "b", "c"},
                ),
            ],
        },
    )


def test_named_and_nested_boolean_scopes_reduce_to_canonical_atoms(
    registry: ClosedScopeRegistry,
) -> None:
    assert registry.atoms("left") == frozenset({"a", "b"})
    assert registry.atoms("outer") == frozenset({"a", "b", "c"})
    assert registry.atoms("left_without_b") == frozenset({"a"})

    first = registry.canonical(
        {"any_of": ["left", {"all_of": ["right", {"not": {"atoms": ["d"]}}]}]}
    )
    second = registry.canonical({"any_of": [{"atoms": ["c"]}, "left"]})
    assert first == second == CanonicalScope("fixture_rows", ("a", "b", "c"))


def test_compiler_provided_primitives_are_generic_and_composable(
    registry: ClosedScopeRegistry,
) -> None:
    column_scope = {
        "column_predicate": {
            "entity": "person",
            "column": "status",
            "op": "eq",
            "value": {"eligible": [True, 1]},
        }
    }
    assert registry.atoms({"channel_is": "survey"}) == frozenset({"a", "c"})
    assert registry.atoms({"support_role_is": "donor"}) == frozenset({"a", "b"})
    assert registry.atoms(column_scope) == frozenset({"a", "d"})
    assert registry.atoms(
        {
            "all_of": [
                {"channel_is": "survey"},
                {"support_role_is": "donor"},
                column_scope,
            ]
        }
    ) == frozenset({"a"})


def test_equality_overlap_and_exhaustiveness_are_exact(
    registry: ClosedScopeRegistry,
) -> None:
    assert registry.equal("left", {"not": "right"})
    assert not registry.equal("left", "right")
    assert registry.overlaps("left", {"channel_is": "survey"})
    assert not registry.overlaps("left", "right")
    assert registry.is_exhaustive(["left", "right"])
    assert registry.is_exhaustive({"any_of": ["left", "right"]})
    assert not registry.is_exhaustive(["left", {"atoms": ["c"]}])


def test_authorization_uses_stable_row_ids_and_full_atom_containment(
    registry: ClosedScopeRegistry,
) -> None:
    row_atoms = {
        "row-10": {"a"},
        ("stable", 20): {"b"},
        30: {"a", "b"},
        40: {"a", "c"},
        50: {"d"},
    }
    assert registry.authorized_row_ids("left", row_atoms) == frozenset(
        {"row-10", ("stable", 20), 30}
    )


def test_constructor_copies_inputs_and_exposes_deeply_immutable_state() -> None:
    universe = ["b", "a"]
    left_atoms = {"a"}
    channel_atoms = {"b"}
    scopes = {"left": left_atoms}
    primitives = {"channel_is": {"secondary": channel_atoms}}
    registry = ClosedScopeRegistry("rows", universe, scopes, primitives)

    universe.append("c")
    left_atoms.add("b")
    channel_atoms.add("a")
    scopes["other"] = {"b"}
    primitives["channel_is"]["secondary"] = {"a"}

    assert registry.universe == ("a", "b")
    assert registry.atoms("left") == frozenset({"a"})
    assert registry.atoms({"channel_is": "secondary"}) == frozenset({"b"})
    with pytest.raises(TypeError):
        registry.scopes["other"] = CanonicalScope("rows", ("b",))
    with pytest.raises(TypeError):
        registry.primitives["channel_is"]["secondary"] = frozenset({"a"})
    with pytest.raises(FrozenInstanceError):
        registry.predicate_space = "changed"


def test_compiler_wire_round_trip_has_a_stable_authority_identity() -> None:
    wire = {
        "predicate_space": "fixture_rows",
        "universe": ["b", "a"],
        "scopes": [
            {"id": "whole", "atoms": ["b", "a"]},
            {"id": "left", "atoms": ["a"]},
        ],
    }
    registry = ClosedScopeRegistry.from_wire(wire)
    assert registry.to_wire() == {
        "predicate_space": "fixture_rows",
        "universe": ["a", "b"],
        "scopes": [
            {"id": "left", "atoms": ["a"]},
            {"id": "whole", "atoms": ["a", "b"]},
        ],
    }
    assert (
        ClosedScopeRegistry.from_wire(registry.to_wire()).identity_sha256
        == registry.identity_sha256
    )
    broader = ClosedScopeRegistry.from_wire(
        {
            **registry.to_wire(),
            "scopes": [
                {"id": "left", "atoms": ["a", "b"]},
                {"id": "whole", "atoms": ["a", "b"]},
            ],
        }
    )
    assert broader.identity_sha256 != registry.identity_sha256


@pytest.mark.parametrize(
    "wire",
    [
        {},
        {"predicate_space": "rows", "universe": ["a"], "scopes": {}},
        {
            "predicate_space": "rows",
            "universe": ["a"],
            "scopes": [{"id": "all", "atoms": ["a"], "extra": True}],
        },
        {
            "predicate_space": "rows",
            "universe": ["a"],
            "scopes": [
                {"id": "all", "atoms": ["a"]},
                {"id": "all", "atoms": ["a"]},
            ],
        },
    ],
)
def test_compiler_wire_registry_is_closed(wire: object) -> None:
    with pytest.raises(ScopeAlgebraError):
        ClosedScopeRegistry.from_wire(wire)


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (
            lambda: ClosedScopeRegistry("Rows", ["a"], {"all": ["a"]}),
            "predicate_space",
        ),
        (
            lambda: ClosedScopeRegistry("rows", [], {"all": []}),
            "at least one finite atom",
        ),
        (
            lambda: ClosedScopeRegistry("rows", ["a", "a"], {"all": ["a"]}),
            "duplicate finite atom",
        ),
        (
            lambda: ClosedScopeRegistry("rows", ["a"], {}),
            "at least one named scope",
        ),
        (
            lambda: ClosedScopeRegistry("rows", ["a"], {"all": ["missing"]}),
            "outside universe",
        ),
        (
            lambda: ClosedScopeRegistry(
                "rows", ["a"], {"all": ["a"]}, {"country_program": {}}
            ),
            "unknown predicate spaces",
        ),
        (
            lambda: ClosedScopeRegistry("rows", ["a"], {"one": "two", "two": "one"}),
            "cyclic named scopes",
        ),
    ],
)
def test_constructor_rejects_open_or_malformed_registries(
    constructor: object,
    match: str,
) -> None:
    with pytest.raises(ScopeAlgebraError, match=match):
        constructor()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("scope", "match"),
    [
        ("unknown", "unknown named scope"),
        ({"atoms": ["unknown"]}, "outside universe"),
        ({"all_of": []}, "non-empty expression array"),
        ({"any_of": "left"}, "non-empty expression array"),
        ({"left": "right"}, "exactly one predicate form"),
        ({"not": "left", "any_of": ["right"]}, "exactly one predicate form"),
        ({"channel_is": "unregistered"}, "unknown compiler-provided primitive"),
        (
            {
                "column_predicate": {
                    "entity": "person",
                    "column": "status",
                    "op": "eq",
                }
            },
            "required for 'eq'",
        ),
        (
            {
                "column_predicate": {
                    "entity": "person",
                    "column": "status",
                    "op": "is_null",
                    "value": None,
                }
            },
            "forbidden for 'is_null'",
        ),
    ],
)
def test_scope_evaluation_rejects_unknown_or_malformed_expressions(
    registry: ClosedScopeRegistry,
    scope: object,
    match: str,
) -> None:
    with pytest.raises(ScopeAlgebraError, match=match):
        registry.atoms(scope)


def test_canonical_scope_rejects_an_unknown_predicate_space_or_atom(
    registry: ClosedScopeRegistry,
) -> None:
    with pytest.raises(ScopeAlgebraError, match="unknown predicate space"):
        registry.atoms(CanonicalScope("other_rows", ("a",)))
    with pytest.raises(ScopeAlgebraError, match="outside universe"):
        registry.atoms(CanonicalScope("fixture_rows", ("z",)))


@pytest.mark.parametrize(
    ("row_atoms", "match"),
    [
        ({"row": set()}, "non-empty atom set"),
        ({"row": {"z"}}, "outside universe"),
        ({"row": ["a"]}, "non-empty atom set"),
    ],
)
def test_authorization_refuses_unclassified_or_open_rows(
    registry: ClosedScopeRegistry,
    row_atoms: object,
    match: str,
) -> None:
    with pytest.raises(ScopeAlgebraError, match=match):
        registry.authorized_row_ids("left", row_atoms)  # type: ignore[arg-type]


def test_recursive_expression_container_is_refused(
    registry: ClosedScopeRegistry,
) -> None:
    recursive: dict[str, object] = {}
    recursive["not"] = recursive
    with pytest.raises(ScopeAlgebraError, match="cyclic predicate container"):
        registry.atoms(recursive)
