"""The allocator hook of the cross-grain operator (microcosm#1760).

``CrossGrainRule.allocator`` lets a country declare how one control's value is
divided among its lower rows. The UK declares none, and the UK graph hashes
``cross_grain.py``, so the hook must leave the standing rule as it was: the
same values bit for bit, the same receipts and the same refusals.

``test_support/microcosm_build/cross_grain_reference.py`` is the module as it
stood at ``e29843dc7``, before the hook; its sha256 is pinned here. The
differential test runs seeded random surfaces through both modules. A second
test runs the operator's own tests (``test_cross_grain.py``) with every call
of the operator compared with the reference copy: those tests hold the shapes
that the seeded surfaces lack (a bridge valued for each geography, rows that
bind nothing, reviewed unbound targets, partitions). When the standing rule
changes on purpose, delete the reference copy and the tests that read it:
they prove only that the hook changed nothing.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build import cross_grain
from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)
from test_support.microcosm_build import cross_grain_reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

#: sha256 of ``cross_grain.py`` at ``e29843dc7``.
REFERENCE_SHA256 = "f61f83dd4577f322cfbc5a1276c4454c36877b56b2d480480ac74155623cd7f3"

#: What this branch changed or added in the module; everything else is the
#: reference, source line for source line. The allocator is the first two of
#: each set. The rest is the general model (``test_cross_grain_model.py``):
#: the bound of a bridge, the members of a leg and the bridge selector.
CHANGED = {
    "CrossGrainRule",
    "_reconcile_cross_grain_surface_with_control_receipts",
    "CrossGrainBridge",
    "apply_cross_grain_reconciliation",
    "_validate_rule",
}
ADDED = {
    "_allocation",
    "_allocator_name",
    "CrossGrainBridgeSelector",
    "expand_bridge_selectors",
    "_selects",
    "_input_factor",
    "_median_input_factor",
    "_assert_expected_members",
    # The metadata of a reconciled row, for a caller that holds target rows.
    "cross_grain_metadata",
    # The rule of the declared members, for a leg that a caller checks itself.
    "assert_leg_members",
}

_LEGS = ("E", "W", "S", "N")


def test_the_reference_copy_is_the_operator_before_the_hook() -> None:
    path = (
        _TEST_PATHS.repository
        / "test_support"
        / "microcosm_build"
        / "cross_grain_reference.py"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256


def _definitions(module) -> dict[str, str]:
    tree = ast.parse(inspect.getsource(module))
    source = inspect.getsource(module)
    return {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }


def test_the_branch_touches_only_these_definitions() -> None:
    new, old = _definitions(cross_grain), _definitions(cross_grain_reference)
    assert set(new) - set(old) == ADDED
    assert set(old) - set(new) == set()
    assert {name for name in old if new[name] != old[name]} == CHANGED


def _imports(module) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.parse(inspect.getsource(module)).body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]


def test_the_branch_adds_no_import() -> None:
    """The UK graph hashes the modules a target module imports by name."""

    assert _imports(cross_grain) == _imports(cross_grain_reference)


def _signature(concept: str) -> dict[str, object]:
    return {
        "measurement": {
            "concept": concept,
            "entity": "household",
            "map_to": None,
            "filters": [],
        }
    }


def _random_case(seed: int):
    """A surface, its bound targets, signatures, rule arguments and licences.

    Three tiers (a country, regions as a middle control tier, two leaf
    grains), several exact-signature concepts and one bounded bridge. Values
    are sometimes signed, zero or missing on a leg, so the draws reach the
    refusals as well as the rescaling.
    """

    rng = np.random.default_rng(seed)
    legs = tuple(leg for leg in _LEGS if rng.random() < 0.85) or ("E",)
    # A middle tier on every leg, on none, or (now and then) on some: the
    # last is refused, since a partial tier would take the whole control.
    tier = rng.random()
    regions = (
        ()
        if tier < 0.45
        else legs
        if tier < 0.9
        else tuple(leg for leg in legs if rng.random() < 0.6)
    )
    rows: list[tuple[str, str, str, float]] = []
    signatures: dict[str, dict[str, object]] = {}
    bound: list[str] = []
    licences: dict[str, frozenset[str]] = {}

    def value(scale: float) -> float:
        draw = float(rng.lognormal(mean=math.log(scale), sigma=1.0))
        roll = rng.random()
        if roll < 0.02:
            return 0.0
        if roll < 0.06:
            return -draw
        return draw

    for index in range(int(rng.integers(1, 4))):
        concept = f"concept_{index}"
        national, local = f"national_{index}", f"local_{index}"
        signatures[national] = _signature(concept)
        signatures[local] = _signature(concept)
        bound.append(national)
        rows.append(("country", "UK", national, abs(value(1e6)) + 1.0))
        for region in regions:
            if rng.random() < 0.97:
                rows.append(("region", region, national, abs(value(2e5)) + 1.0))
        empty: set[str] = set()
        for leg in legs:
            if rng.random() < 0.05:
                empty.add(leg)
                continue
            for grain in ("constituency", "la"):
                for area in range(int(rng.integers(1, 5))):
                    rows.append((grain, f"{leg}{grain[0]}{area}", local, value(5e4)))
        if rng.random() < 0.85:
            licences[local] = frozenset(empty)
    bridges: tuple[CrossGrainBridge, ...] = ()
    if rng.random() < 0.6:
        shift = float(rng.choice([0.05, 0.5, 0.9]))
        bridges = (
            CrossGrainBridge(
                "bridged",
                concept="bridged_concept",
                higher_target_ids=("part_a", "part_b"),
                lower_side="contract:bridged_local",
                max_factor_shift=None if rng.random() < 0.3 else shift,
            ),
        )
        signatures["part_a"] = _signature("bridged_concept")
        signatures["part_b"] = _signature("bridged_concept")
        signatures["bridged_local"] = _signature("bridged_local_concept")
        bound += ["part_a", "part_b"]
        total = abs(value(1e6)) + 1.0
        split = float(rng.uniform(0.1, 0.9))
        rows.append(("country", "UK", "part_a", total * split))
        rows.append(("country", "UK", "part_b", total * (1.0 - split)))
        target = total * float(rng.uniform(0.7, 1.3))
        areas = [f"{leg}c{index}" for leg in legs for index in range(3)]
        shares = rng.dirichlet(np.ones(len(areas)))
        rows += [
            ("constituency", area, "bridged_local", float(target * share))
            for area, share in zip(areas, shares, strict=True)
        ]
    order = rng.permutation(len(rows))
    surface = pd.DataFrame(
        [rows[index] for index in order],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    rule_arguments = {
        "grain_precedence": ("country", "region", "constituency", "la"),
        "signature_fields": ("concept", "entity", "map_to", "filters"),
        "bridges": bridges,
        "leg_of_area": lambda area: area[0],
        "parent_geography_legs": {"UK": legs, **{leg: (leg,) for leg in regions}},
        "control_grains": ("country", "region") if regions else (),
    }
    return surface, tuple(bound), signatures, rule_arguments, licences


def _outcome(module, case, *, check_factor_bounds: bool):
    surface, bound, signatures, rule_arguments, licences = case
    bridges = tuple(
        module.CrossGrainBridge(
            bridge.bridge_id,
            concept=bridge.concept,
            higher_target_ids=bridge.higher_target_ids,
            lower_side=bridge.lower_side,
            per_geography=bridge.per_geography,
            max_factor_shift=bridge.max_factor_shift,
        )
        for bridge in rule_arguments["bridges"]
    )
    rule = module.CrossGrainRule(**{**rule_arguments, "bridges": bridges})
    try:
        reconciled, receipt = module.apply_cross_grain_reconciliation(
            surface,
            bound,
            signatures,
            rule,
            licensed_empty_legs=licences,
            check_factor_bounds=check_factor_bounds,
        )
    except ValueError as error:
        return ("refused", str(error))
    return (
        "reconciled",
        reconciled["value"].to_numpy(dtype=np.float64).tobytes(),
        json.dumps(receipt, sort_keys=False),
    )


def test_the_default_path_is_the_operator_before_the_hook_bit_for_bit() -> None:
    outcomes = {"reconciled": 0, "refused": 0}
    rescaled_legs = 0
    for seed in range(400):
        case = _random_case(seed)
        for check_factor_bounds in (True, False):
            new = _outcome(cross_grain, case, check_factor_bounds=check_factor_bounds)
            old = _outcome(
                cross_grain_reference, case, check_factor_bounds=check_factor_bounds
            )
            assert new == old, seed
            outcomes[new[0]] += 1
            if new[0] == "reconciled":
                rescaled_legs += sum(
                    len(group["legs"]) for group in json.loads(new[2])["groups"]
                )
    # The draws reach both the rescaling and the refusals.
    assert outcomes["reconciled"] >= 200
    assert outcomes["refused"] >= 100
    assert rescaled_legs >= 1_000


def _per_geography_case(seed: int):
    """A bridge whose higher targets are each the control of their own geography.

    The UK declares such bridges (each nation's estimate over its own legs).
    A geography can carry no estimate, a leg can be empty, and the bound is
    sometimes close, so the draws reach the refusals too.
    """

    rng = np.random.default_rng(10_000 + seed)
    legs = tuple(leg for leg in _LEGS if rng.random() < 0.9) or ("E",)
    rows: list[tuple[str, str, str, float]] = []
    signatures = {"bridged_local": _signature("bridged_local_concept")}
    estimates: list[str] = []
    empty: set[str] = set()
    for leg in legs:
        total = float(rng.lognormal(mean=math.log(1e5), sigma=1.0)) + 1.0
        if rng.random() < 0.9:
            estimate = f"estimate_{leg}"
            signatures[estimate] = _signature("bridged_concept")
            estimates.append(estimate)
            rows.append(("region", leg, estimate, total))
        if rng.random() < 0.06:
            empty.add(leg)
            continue
        target = total * float(rng.uniform(0.6, 1.4))
        for grain in ("constituency", "la"):
            shares = rng.dirichlet(np.ones(int(rng.integers(1, 5))))
            rows += [
                (
                    grain,
                    f"{leg}{grain[0]}{index}",
                    "bridged_local",
                    float(target * share),
                )
                for index, share in enumerate(shares)
            ]
    bridge = CrossGrainBridge(
        "estimates",
        concept="bridged_concept",
        higher_target_ids=tuple(estimates) or ("estimate_E",),
        lower_side="contract:bridged_local",
        per_geography=True,
        max_factor_shift=(
            None if rng.random() < 0.3 else float(rng.choice([0.2, 0.5, 0.9]))
        ),
    )
    order = rng.permutation(len(rows))
    surface = pd.DataFrame(
        [rows[index] for index in order],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    rule_arguments = {
        "grain_precedence": ("country", "region", "constituency", "la"),
        "signature_fields": ("concept", "entity", "map_to", "filters"),
        "bridges": (bridge,),
        "leg_of_area": lambda area: area[0],
        "parent_geography_legs": {"UK": legs, **{leg: (leg,) for leg in legs}},
        "control_grains": ("country", "region"),
    }
    licences = {"bridged_local": frozenset(empty)} if rng.random() < 0.85 else {}
    return surface, tuple(estimates), signatures, rule_arguments, licences


def test_a_bridge_for_each_geography_is_the_operator_before_bit_for_bit() -> None:
    outcomes = {"reconciled": 0, "refused": 0}
    rescaled_legs = 0
    for seed in range(200):
        case = _per_geography_case(seed)
        for check_factor_bounds in (True, False):
            new = _outcome(cross_grain, case, check_factor_bounds=check_factor_bounds)
            old = _outcome(
                cross_grain_reference, case, check_factor_bounds=check_factor_bounds
            )
            assert new == old, seed
            outcomes[new[0]] += 1
            if new[0] == "reconciled":
                rescaled_legs += sum(
                    len(group["legs"]) for group in json.loads(new[2])["groups"]
                )
    # The draws reach both the rescaling and the refusals.
    assert outcomes["reconciled"] >= 150
    assert outcomes["refused"] >= 80
    assert rescaled_legs >= 1_000


# ---------------------------------------------------------------------------
# The operator's own tests, with each call compared with the reference
# ---------------------------------------------------------------------------

#: Tests of ``test_cross_grain.py`` that change the operator itself with
#: ``monkeypatch``: the reference copy would not have the change.
_NOT_REPLAYED = {"test_off_control_reconciliation_is_refused"}


def _as_reference(value):
    """``value`` as the reference module's class, field for field."""

    for name in ("CrossGrainRule", "CrossGrainBridge", "CrossGrainPartition"):
        if isinstance(value, getattr(cross_grain, name)):
            old_class = getattr(cross_grain_reference, name)
            return old_class(
                **{
                    field.name: _as_reference(getattr(value, field.name))
                    for field in dataclasses.fields(old_class)
                }
            )
    if isinstance(value, tuple):
        return tuple(_as_reference(item) for item in value)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return value


def _uses_a_general_option(arguments) -> bool:
    """A call that the reference cannot run: it has options the branch added."""

    rules = [value for value in arguments if isinstance(value, CrossGrainRule)]
    return any(
        rule.allocator is not None
        or any(
            bridge.min_factor_shift is not None
            or bridge.shift_around != "one"
            or bridge.out_of_bound != "refuse"
            for bridge in rule.bridges
        )
        for rule in rules
    )


def _call_image(function, arguments, keywords) -> tuple:
    try:
        result = function(*arguments, **keywords)
    except Exception as error:  # the refusal is part of the result
        return ("raised", type(error).__name__, str(error)), error
    if isinstance(result, tuple) and isinstance(result[0], pd.DataFrame):
        frame, receipt = result
        image = (
            "returned",
            frame.to_json(double_precision=15),
            frame["value"].to_numpy(dtype=np.float64).tobytes(),
            json.dumps(receipt, default=str),
        )
    else:
        # Detected groups: frozen dataclasses with the same name in both
        # modules, so their text is the same when their content is.
        image = ("returned", repr(result))
    return image, result


def _compared_with_the_reference(name: str, stats: Counter):
    new_function = getattr(cross_grain, name)
    old_function = getattr(cross_grain_reference, name)

    def run(*arguments, **keywords):
        arguments = tuple(
            tuple(value) if isinstance(value, Iterator) else value
            for value in arguments
        )
        keywords = {
            key: tuple(value) if isinstance(value, Iterator) else value
            for key, value in keywords.items()
        }
        assert not _uses_a_general_option(arguments) and (
            "expected_members" not in keywords
        ), "test_cross_grain.py uses an option that the reference does not have"
        new_image, result = _call_image(new_function, arguments, keywords)
        old_image, _ = _call_image(
            old_function,
            tuple(_as_reference(value) for value in arguments),
            {key: _as_reference(value) for key, value in keywords.items()},
        )
        assert new_image == old_image, name
        stats[name] += 1
        stats[new_image[0]] += 1
        for value in arguments:
            if isinstance(value, CrossGrainRule):
                stats["a bounded bridge"] += any(
                    bridge.max_factor_shift is not None for bridge in value.bridges
                )
        for key in (
            "frame_only_positions",
            "reviewed_unbound_higher_targets",
            "licensed_empty_legs",
        ):
            stats[key] += bool(keywords.get(key))
        stats["bounds not checked"] += keywords.get("check_factor_bounds") is False
        if isinstance(result, Exception):
            raise result
        return result

    return run


def _parameter_sets(test) -> list[dict]:
    """The arguments of each case of a test: its ``parametrize`` mark, or none."""

    marks = [
        mark for mark in getattr(test, "pytestmark", []) if mark.name == "parametrize"
    ]
    if not marks:
        return [{}]
    (mark,) = marks
    names, cases = mark.args
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",")]
    return [
        dict(zip(names, case if len(names) > 1 else (case,), strict=True))
        for case in cases
    ]


def test_the_operators_own_tests_give_the_reference_results(monkeypatch) -> None:
    """Every operator call of ``test_cross_grain.py`` is the reference's result."""

    path = Path(__file__).with_name("test_cross_grain.py")
    spec = importlib.util.spec_from_file_location("cross_grain_suite_replay", path)
    suite = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(suite)
    stats: Counter = Counter()
    for name in (
        "apply_cross_grain_reconciliation",
        "apply_cross_grain_partitions",
        "detect_cross_grain_inconsistencies",
    ):
        monkeypatch.setattr(suite, name, _compared_with_the_reference(name, stats))
    tests = {
        name: test
        for name, test in vars(suite).items()
        if name.startswith("test_") and callable(test)
    }
    assert _NOT_REPLAYED <= set(tests)
    for name, test in tests.items():
        if name in _NOT_REPLAYED:
            continue
        for case in _parameter_sets(test):
            test(**case)
            stats["cases"] += 1
    # The replay reached the operator, with both results and refusals, and
    # with each shape that the seeded surfaces above do not have.
    assert stats["cases"] >= 40
    assert stats["apply_cross_grain_reconciliation"] >= 40
    assert stats["returned"] >= 15 and stats["raised"] >= 15
    for shape in (
        "apply_cross_grain_partitions",
        "detect_cross_grain_inconsistencies",
        "a bounded bridge",
        "frame_only_positions",
        "reviewed_unbound_higher_targets",
        "licensed_empty_legs",
        "bounds not checked",
    ):
        assert stats[shape] >= 1, shape


def _fsum_allocator(raw_values, parent_value):
    raw_sum = math.fsum(raw_values.tolist())
    factor = 1.0 if raw_sum == 0.0 else parent_value / raw_sum
    return raw_values * factor, factor, raw_sum


def _rule(allocator=None) -> CrossGrainRule:
    return CrossGrainRule(
        grain_precedence=("country", "constituency"),
        signature_fields=("concept", "entity", "map_to", "filters"),
        bridges=(),
        leg_of_area=lambda area: area[0],
        parent_geography_legs={"UK": ("E", "W")},
        allocator=allocator,
    )


def _surface(values) -> pd.DataFrame:
    return pd.DataFrame(
        [("country", "UK", "national", 120.0)]
        + [
            ("constituency", f"E{index}", "local", value)
            for index, value in enumerate(values)
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )


_SIGNATURES = {"national": _signature("households"), "local": _signature("households")}


def test_an_allocator_divides_the_control_and_the_receipt_names_it() -> None:
    seen = []

    def allocator(raw_values, parent_value):
        seen.append((raw_values.tolist(), parent_value))
        return _fsum_allocator(raw_values, parent_value)

    surface = _surface([40.0, 20.0])
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, ("national",), _SIGNATURES, _rule(allocator)
    )

    # The rows arrive in surface order with the control's value.
    assert seen == [([40.0, 20.0], 120.0)]
    assert reconciled["value"].tolist() == [120.0, 80.0, 40.0]
    (leg,) = receipt["groups"][0]["legs"]
    assert leg["old_total"] == 60.0
    assert leg["declared_factor"] == 2.0
    assert leg["new_total"] == 120.0
    assert leg["allocator"] == "allocator"


def test_the_standing_rule_writes_no_allocator_key() -> None:
    _reconciled, receipt = apply_cross_grain_reconciliation(
        _surface([40.0, 20.0]), ("national",), _SIGNATURES, _rule()
    )
    (leg,) = receipt["groups"][0]["legs"]
    assert "allocator" not in leg


def test_an_allocator_can_differ_from_the_numpy_sum_in_the_last_bit() -> None:
    """Why the hook exists: an exactly rounded sum is another factor."""

    rng = np.random.default_rng(1760)
    for _attempt in range(200):
        values = rng.lognormal(mean=10.0, sigma=2.0, size=37)
        if math.fsum(values.tolist()) != float(values.sum()):
            break
    else:
        pytest.fail("no draw separated fsum from the numpy sum")
    surface = _surface(values.tolist())
    standing, _ = apply_cross_grain_reconciliation(
        surface, ("national",), _SIGNATURES, _rule()
    )
    allocated, receipt = apply_cross_grain_reconciliation(
        surface, ("national",), _SIGNATURES, _rule(_fsum_allocator)
    )
    assert (
        standing["value"].to_numpy().tobytes()
        != allocated["value"].to_numpy().tobytes()
    )
    (leg,) = receipt["groups"][0]["legs"]
    assert leg["old_total"] == math.fsum(values.tolist())
    np.testing.assert_array_equal(
        allocated["value"].to_numpy()[1:],
        values * (120.0 / math.fsum(values.tolist())),
    )


@pytest.mark.parametrize(
    ("allocator", "message"),
    [
        (lambda raw, parent: (raw * 0.0, 1.0, 0.0), "vanishing lower-leg total"),
        (lambda raw, parent: (-raw, -1.0, float(raw.sum())), "opposite-signed"),
        (lambda raw, parent: (raw, 1.0, float(raw.sum())), "off its control"),
        (lambda raw, parent: (raw[:1], 2.0, float(raw.sum())), "returned"),
        (
            lambda raw, parent: (raw * math.nan, 2.0, float(raw.sum())),
            "non-finite reconciled total",
        ),
        (lambda raw, parent: (raw * 2.0, math.inf, float(raw.sum())), "non-finite"),
    ],
)
def test_the_operators_refusals_apply_to_an_allocator(allocator, message) -> None:
    with pytest.raises(ValueError, match=message):
        apply_cross_grain_reconciliation(
            _surface([40.0, 20.0]), ("national",), _SIGNATURES, _rule(allocator)
        )
