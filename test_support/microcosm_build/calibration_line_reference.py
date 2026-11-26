# Verbatim copy of uk_registry_on_calibration_line from
# microcosm.build.uk_runtime.ledger_targets at 508351658, before microcosm#1759
# declared the calibration lines in uk/target_sets.json. Tests hold the new
# code to it; do not edit. It reads the contract through the live helpers.
# ruff: noqa

from __future__ import annotations

from typing import Any

from microcosm.build.uk_runtime.ledger_targets import (
    _uk_contract_targets,
    uk_contract_target_calibration_lines,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

UK_CALIBRATION_LINES: tuple[str, ...] = ("national", "local")


def uk_registry_on_calibration_line(
    registry: TargetRegistry, *, line: str
) -> tuple[TargetRegistry, dict[str, Any]]:
    """The compiled registry less the references ``line`` does not bind.

    Every committed reference compiles on every line, so the compile-parity
    registries stay complete; a line's solve surface drops the references
    whose contract target's ``calibration_lines`` leave ``line`` out, right
    after the compile and before reconciliation, the measure exclusions and the
    frozen-register check. A line no declaration narrows keeps its registry
    (the same object, so the same version). The receipt names the line, the
    all-lines registry version and count, and the dropped contract targets.
    """

    if line not in UK_CALIBRATION_LINES:
        raise ValueError(
            f"unknown UK calibration line {line!r}; the lines are "
            f"{list(UK_CALIBRATION_LINES)!r}."
        )
    off_line = frozenset(
        target_id
        for target_id, target in _uk_contract_targets(national_only=False).items()
        if line not in uk_contract_target_calibration_lines(target)
    )
    kept: list[TargetSpec] = []
    dropped: list[TargetSpec] = []
    dropped_target_ids: set[str] = set()
    for spec in registry.specs:
        target_id = str(
            spec.metadata.get("contract_target_id") or spec.name.split("@", 1)[0]
        )
        if target_id in off_line:
            dropped.append(spec)
            dropped_target_ids.add(target_id)
        else:
            kept.append(spec)
    receipt: dict[str, Any] = {
        "line": line,
        "all_lines_registry_version": registry.version,
        "all_lines_reference_count": len(registry.specs),
        "dropped_reference_count": len(dropped),
        "dropped_target_ids": sorted(dropped_target_ids),
    }
    if not dropped:
        return registry, receipt
    return TargetRegistry(kept, country=registry.country), receipt
