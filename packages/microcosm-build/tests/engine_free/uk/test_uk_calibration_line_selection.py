"""The UK calibration lines as declared target sets, against the replaced code.

Before microcosm#1759, ``uk_runtime.ledger_targets.uk_registry_on_calibration_line``
dropped the targets that a line does not bind with its own loop. It now
selects through ``uk/target_sets.json`` (product ``calibration_line``, one mode
for each line) with the shared ``select_target_set`` and one UK predicate.
``test_support/microcosm_build/calibration_line_reference.py`` holds the old
function verbatim. On random registries the two give the same registry (the
same object when nothing drops), the same receipt, or the same refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime import ledger_targets
from microcosm.build.uk_runtime.target_sets import (
    CALIBRATION_LINE_PRODUCT,
    uk_calibration_lines,
    uk_target_set_declarations,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from test_support.microcosm_build import calibration_line_reference as reference

_CONTRACT = ledger_targets._uk_contract_targets(national_only=False)
#: The contract targets that bind on one line only, and some that bind on both.
_NARROWED = sorted(
    target_id
    for target_id, target in _CONTRACT.items()
    if ledger_targets.uk_contract_target_calibration_lines(target)
    != frozenset(ledger_targets.UK_CALIBRATION_LINES)
)
_BOTH = sorted(set(_CONTRACT) - set(_NARROWED))[:40]


def _registry(seed: int) -> TargetRegistry:
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(int(rng.integers(0, 25))):
        draw = rng.uniform()
        if draw < 0.3:
            target_id = str(rng.choice(_NARROWED))
        elif draw < 0.8:
            target_id = str(rng.choice(_BOTH))
        else:
            target_id = f"microcosm.test.unknown_{index}"
        metadata = {}
        if rng.uniform() < 0.5:
            metadata["contract_target_id"] = target_id
            name = f"row_{index}@2025"
        else:
            name = f"{target_id}@{2024 + int(rng.integers(2))}#{index}"
        # A line selection must not read levels: some rows have none.
        if rng.uniform() < 0.5:
            metadata["geography_level"] = str(
                rng.choice(["country", "region", "nation"])
            )
        specs.append(
            TargetSpec(
                name=name,
                entity="household",
                measure="household_weight",
                value=float(rng.uniform(1.0, 1e6)),
                period=2025,
                source="test",
                family="test_family",
                metadata=metadata,
            )
        )
    return TargetRegistry(specs, country="uk")


def _outcome(select, registry):
    try:
        result, receipt = select()
    except Exception as error:  # noqa: BLE001 - the refusal is the outcome
        return ("refused", type(error).__name__, str(error))
    return (
        "selected",
        result is registry,
        result.version,
        [spec.name for spec in result.specs],
        receipt,
    )


@pytest.mark.parametrize("seed", range(150))
@pytest.mark.parametrize("line", ["national", "local", "regional"])
def test_each_line_matches_the_replaced_code(seed: int, line: str) -> None:
    registry = _registry(seed)
    old = _outcome(
        lambda: reference.uk_registry_on_calibration_line(registry, line=line),
        registry,
    )
    new = _outcome(
        lambda: ledger_targets.uk_registry_on_calibration_line(registry, line=line),
        registry,
    )
    assert new == old


def test_the_cases_drop_rows_on_both_lines() -> None:
    dropped = {"national": 0, "local": 0}
    for seed in range(150):
        registry = _registry(seed)
        for line in dropped:
            result, receipt = ledger_targets.uk_registry_on_calibration_line(
                registry, line=line
            )
            dropped[line] += receipt["dropped_reference_count"]
            assert (result is registry) == (receipt["dropped_reference_count"] == 0)
    assert all(count > 0 for count in dropped.values()), dropped


def test_the_lines_are_the_declared_modes() -> None:
    product = uk_target_set_declarations().product(CALIBRATION_LINE_PRODUCT)
    assert ledger_targets.UK_CALIBRATION_LINES == uk_calibration_lines()
    assert uk_calibration_lines() == product.mode_names == ("national", "local")
    assert _NARROWED
