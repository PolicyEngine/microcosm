"""[BE] The skeleton admits only the settled two-key binding selector."""

from types import SimpleNamespace

import pytest

from microcosm.build.transport import entitlement
from microcosm.build.transport.compose import SKELETON_KERNELS, _param

ROW = {"id": "tax", "period": "test-period"}
REFS = {"tax": "prepared-tax"}


def _spec(rows=(ROW,)):
    return {"resources": {"axiom_rules_bindings": {"bindings": list(rows)}}}


def _outcome(select, *args):
    try:
        return "value", select(*args)
    except (KeyError, ValueError) as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize(
    "field,expected", [("engine_ref", "prepared-tax"), ("period", "test-period")]
)
def test_binding_selector_resolves_only_the_selected_field(field, expected):
    assert (
        _param(_spec(), {"binding": "tax", "field": field}, {"tax": "prepared-tax"})
        == expected
    )


@pytest.mark.parametrize(
    "selection",
    [
        {"binding": "tax", "field": "engine_ref", "extra": 1},
        {"binding": "tax"},
        {"binding": 1, "field": "engine_ref"},
        {"binding": [], "field": "engine_ref"},
        {"binding": "", "field": "engine_ref"},
        {"binding": " ", "field": "engine_ref"},
        {"binding": "tax", "field": 1},
        {"binding": "tax", "field": []},
        {"binding": "tax", "field": "variables"},
    ],
)
def test_binding_selector_refuses_every_unsettled_shape(selection):
    with pytest.raises(ValueError, match="binding selection"):
        _param(_spec(), selection, {"tax": "prepared-tax"})


@pytest.mark.parametrize("refs", [None, {}, {"other": "prepared-tax"}])
def test_binding_selector_requires_the_declared_prepared_reference(refs):
    with pytest.raises(ValueError, match="needs one prepared reference"):
        _param(_spec(), {"binding": "tax", "field": "engine_ref"}, refs)


@pytest.mark.parametrize(
    "selection,rows,refs",
    [
        ({"binding": "tax", "field": "engine_ref"}, (ROW,), REFS),
        ({"binding": "tax", "field": "period"}, (ROW,), REFS),
        ({"binding": "tax", "field": "variables"}, (ROW,), REFS),
        ({"binding": "tax", "field": "engine_ref", "x": 1}, (ROW,), REFS),
        ({"binding": "tax"}, (ROW,), REFS),
        ({"binding": "other", "field": "engine_ref"}, (ROW,), REFS),
        ({"binding": "tax", "field": "engine_ref"}, (ROW,), {}),
        ({"binding": "tax", "field": "engine_ref"}, (ROW, ROW), REFS),
        ({"binding": "tax", "field": "period"}, ({"id": "tax"},), REFS),
    ],
)
def test_skeleton_selector_matches_the_entitlement_selector(selection, rows, refs):
    skeleton = _outcome(_param, _spec(rows), selection, refs)
    extension = _outcome(
        entitlement._binding,
        _spec(rows),
        SimpleNamespace(engine_refs=refs),
        selection,
    )
    assert skeleton[0] == extension[0]
    if skeleton[0] == "value":
        assert skeleton == extension


def test_only_the_skeleton_selector_refuses_a_blank_binding_id():
    """The one intended difference between the two selectors."""
    selection = {"binding": " ", "field": "engine_ref"}
    rows = ({"id": " ", "period": "test-period"},)
    refs = {" ": "prepared-blank"}
    with pytest.raises(ValueError, match="binding selection"):
        _param(_spec(rows), selection, refs)
    assert (
        entitlement._binding(_spec(rows), SimpleNamespace(engine_refs=refs), selection)
        == "prepared-blank"
    )


def test_only_the_new_nonstructural_kernels_enter_the_skeleton():
    assert {
        "simulate.solve_zero@1",
        "money.minor_units@1",
        "money.annual_transfers@1",
    } <= SKELETON_KERNELS
    assert "simulate.counterfactual@1" not in SKELETON_KERNELS
