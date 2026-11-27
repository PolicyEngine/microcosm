"""The UK full build's selection by level, against the code it replaced.

Before microcosm#1759 the UK full build selected its solve surface with
``microcosm.calibrate.target_selection.select_targets`` and the UK resolver
``uk_runtime.graph_targets.target_geography``. It now selects with the shared
``microcosm.calibrate.target_sets.select_by_levels`` and the UK's declared
level rule (``uk/spec/target_sets.yaml``). ``test_support/microcosm_build/target_selection_reference.py``
holds verbatim copies of the old code. On random registries and requests the
two give the same registry, the same receipt bytes, or the same refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.target_sets import (
    uk_full_level_selection,
    uk_target_set_declarations,
)
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.calibrate.target_sets import select_by_levels
from test_support.microcosm_build import target_selection_reference as reference

UK_LEVELS = ("country", "region", "constituency", "la", "brma")

#: The full build's level mode, as uk/spec/target_sets.yaml declares it.
DECLARATION = uk_full_level_selection()


def test_the_packaged_level_rule_is_the_rule_the_uk_resolver_applied() -> None:
    rule = uk_target_set_declarations().level_rule
    assert rule.keys == ("geography_level", "ledger_geography_level")
    assert rule.values == frozenset(UK_LEVELS)
    assert rule.conflict == "refuse"
    assert DECLARATION.level_rule == rule
    assert DECLARATION.clause_ids == UK_LEVELS


#: Levels the UK rule refuses. A spec's metadata values are nonempty
#: strings, so a missing level is a missing key.
_BAD = ("nation", " la", "LA")
_FAMILIES = ("census_households", "hmrc_income", "dwp_uc", "high_value_homes")


def _level(rng) -> str:
    if rng.uniform() < 0.005:
        return str(rng.choice(_BAD))
    return str(rng.choice(UK_LEVELS))


def _registry(seed: int) -> TargetRegistry:
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(int(rng.integers(0, 30))):
        metadata = {}
        local = _level(rng)
        ledger = local if rng.uniform() < 0.97 else _level(rng)
        if rng.uniform() < 0.95:
            metadata["geography_level"] = local
        if rng.uniform() < 0.6:
            metadata["ledger_geography_level"] = ledger
        if rng.uniform() < 0.5:
            metadata["area_code"] = f"E{index:08d}"
        specs.append(
            TargetSpec(
                name=f"target_{index}",
                entity="household",
                measure="household_weight",
                value=float(rng.uniform(1.0, 1e6)),
                period=int(rng.choice([2024, 2025])),
                source=str(rng.choice(["ons", "hmrc", "dwp"])),
                family=str(rng.choice(_FAMILIES)),
                metadata=metadata,
            )
        )
    return TargetRegistry(specs, country="uk")


def _request(seed: int):
    rng = np.random.default_rng(seed + 50_000)
    draw = rng.uniform()
    if draw < 0.3:
        return None
    if draw < 0.85:
        size = int(rng.integers(1, len(UK_LEVELS) + 1))
        return list(rng.choice(UK_LEVELS, size=size, replace=False))
    return [[], ["nowhere"], ["la", "la"], [" la"], "la", ("brma", "nowhere")][
        int(rng.integers(6))
    ]


def _outcome(select):
    try:
        selection = select()
    except Exception as error:  # noqa: BLE001 - the refusal is the outcome
        return ("refused", type(error).__name__, str(error))
    return (
        "selected",
        selection.registry.version,
        [spec.name for spec in selection.registry.specs],
        selection.to_bytes(),
    )


@pytest.mark.parametrize("seed", range(400))
def test_selection_matches_the_replaced_code(seed: int) -> None:
    registry, levels = _registry(seed), _request(seed)
    old = _outcome(
        lambda: reference.select_targets(
            registry,
            geography_levels=levels,
            geography_resolver=reference.target_geography,
        )
    )
    new = _outcome(lambda: select_by_levels(registry, DECLARATION, levels))
    assert new == old


def test_the_random_cases_cover_selections_and_each_refusal() -> None:
    selected, messages = 0, []
    for seed in range(400):
        registry, levels = _registry(seed), _request(seed)
        outcome = _outcome(
            lambda registry=registry, levels=levels: select_by_levels(
                registry, DECLARATION, levels
            )
        )
        if outcome[0] == "selected":
            selected += 1
        else:
            messages.append(outcome[2])
    assert selected >= 50
    text = " ".join(messages)
    for fragment in (
        "contradictory",
        "unsupported geography level",
        "unknown geography levels",
        "nonempty literal names",
        "must not repeat",
        "nonempty sequence",
    ):
        assert fragment in text, fragment


def test_the_level_rule_reads_as_the_uk_resolver_did() -> None:
    for seed in range(200):
        for spec in _registry(seed).specs:
            old = _outcome(lambda spec=spec: _Wrapped(reference.target_geography(spec)))
            new = _outcome(lambda spec=spec: _Wrapped(DECLARATION.level_of(spec)))
            assert new == old


class _Wrapped:
    """A level dressed as a selection, so _outcome compares it."""

    def __init__(self, level: str) -> None:
        self.registry = TargetRegistry([], country="uk")
        self._level = level

    def to_bytes(self) -> bytes:
        return self._level.encode()
