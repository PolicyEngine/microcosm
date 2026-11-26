"""US target nesting on the shared cross-grain operator (microcosm#1760).

A target for a small area must agree with the target of the area that
contains it. This module makes the rows of a lower geography level from a
control at a higher level, for each mechanism that
``us/cross_grain_declarations.json`` declares. The arithmetic is not here:

- :func:`microcosm.build.cross_grain.apply_cross_grain_reconciliation` groups
  the rows of one quantity, finds the control of each lower leg, divides it
  among the leg's rows with the mechanism's declared allocator, checks the sum
  and writes its receipt;
- :mod:`microcosm.build.cross_grain_controls` supplies the allocators, the
  derived controls, the factor band, the completeness check and the
  self-consistency check.

What is here is the US side of each mechanism: which specs or facts are the
control and the lower rows, the refusals with their US wording, and the
metadata keys and receipts the local tools wrote before the operator made the
rows. Stage A changes no value: every surface has the bytes it had.

The module reads specs and facts only. It reads no frame and no household.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.cross_grain import (
    CrossGrainRule,
    apply_cross_grain_reconciliation,
)
from microcosm.build.cross_grain_controls import allocator_named
from microcosm.build.us_runtime.cross_grain_declarations import (
    load_us_cross_grain_declarations,
    us_cross_grain_mechanism,
)

#: The file a materialize stage writes the operator's own receipts to.
NESTING_RECEIPTS_FILENAME = "target_nesting.json"

# ---------------------------------------------------------------------------
# Blocks through the operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NestingBlock:
    """One control and the lower rows it is divided among.

    ``concept`` and ``filters`` are the measurement signature that ties the
    control to its rows. ``lower_values`` are in the order the allocator must
    see them: a residual rule gives its residual to the first of equal
    largest rows.
    """

    concept: str
    filters: tuple[tuple[str, str], ...]
    control_geography_id: str
    control_value: float
    lower_geography_ids: tuple[str, ...]
    lower_values: tuple[float, ...]


@dataclass(frozen=True)
class NestedRows:
    """A block's rows as the operator left them, with its leg receipt."""

    values: tuple[float, ...]
    #: The factor and the lower-row total the allocator reported.
    factor: float
    lower_sum: float


def geography_id(grain: str, code: str) -> str:
    """A surface geography id that cannot collide across grains."""

    return f"{grain}:{code}"


def _signature_key(block: NestingBlock) -> str:
    return "|".join(
        [block.concept, *(f"{name}={value}" for name, value in block.filters)]
    )


def reconcile_blocks(
    mechanism_id: str, blocks: Sequence[NestingBlock]
) -> tuple[list[NestedRows], dict[str, Any]]:
    """Divide each block's control among its rows on the shared operator.

    Builds one surface for the mechanism (a control row and the lower rows of
    every block), runs :func:`apply_cross_grain_reconciliation` with the
    mechanism's declared allocator, and returns each block's rows, in block
    order, with the operator's receipt. A block is one leg: its control row
    parents exactly its own lower rows.

    Raises:
        ValueError: If two blocks share a signature and a control geography,
            if a lower row sits under two controls, if a block has no lower
            row, or if the operator refuses a block (a vanishing or
            non-finite total, a negative factor, or rows off their control).
    """

    declarations = load_us_cross_grain_declarations()
    mechanism = us_cross_grain_mechanism(mechanism_id)
    higher, lower = str(mechanism["higher_grain"]), str(mechanism["lower_grain"])
    rows: list[tuple[str, str, str, float]] = []
    signatures: dict[str, dict[str, Any]] = {}
    bound: list[str] = []
    area_legs: dict[str, str] = {}
    spans: list[tuple[str, int, int]] = []
    controls_seen: set[tuple[str, str]] = set()
    for block in blocks:
        key = _signature_key(block)
        control_id = f"{mechanism_id}:{higher}:{key}"
        lower_id = f"{mechanism_id}:{lower}:{key}"
        if (control_id, block.control_geography_id) in controls_seen:
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r} has two blocks for {key!r} "
                f"at {block.control_geography_id!r}."
            )
        controls_seen.add((control_id, block.control_geography_id))
        if not block.lower_geography_ids or len(block.lower_geography_ids) != len(
            block.lower_values
        ):
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r} block {key!r} at "
                f"{block.control_geography_id!r} needs one value for each of "
                "its lower rows, and at least one row."
            )
        if control_id not in signatures:
            measurement = {
                "concept": block.concept,
                "filters": [{name: value} for name, value in block.filters],
            }
            signatures[control_id] = {"measurement": measurement}
            signatures[lower_id] = {"measurement": measurement}
            bound.append(control_id)
        rows.append(
            (higher, block.control_geography_id, control_id, block.control_value)
        )
        start = len(rows)
        for area, value in zip(
            block.lower_geography_ids, block.lower_values, strict=True
        ):
            if area_legs.setdefault(area, block.control_geography_id) != (
                block.control_geography_id
            ):
                raise ValueError(
                    f"Nesting mechanism {mechanism_id!r}: lower geography "
                    f"{area!r} sits under two controls."
                )
            rows.append((lower, area, lower_id, float(value)))
        spans.append((control_id, start, len(rows)))
    if not blocks:
        receipt: dict[str, Any] = {
            "absence": f"Mechanism {mechanism_id} had no block to reconcile."
        }
        _record(mechanism_id, 0, receipt)
        return [], receipt
    surface = pd.DataFrame(
        rows, columns=["grain", "geography_id", "target_id", "value"]
    )
    rule = CrossGrainRule(
        grain_precedence=tuple(declarations["grain_precedence"]),
        signature_fields=tuple(declarations["signature_fields"]),
        bridges=(),
        leg_of_area=area_legs.__getitem__,
        parent_geography_legs={
            control_geography: (control_geography,)
            for _control_id, control_geography in sorted(controls_seen)
        },
        control_grains=tuple(declarations["control_grains"]),
        allocator=allocator_named(str(mechanism["allocator"])),
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, bound, signatures, rule
    )
    values = reconciled["value"].to_numpy(dtype=np.float64)
    legs = {
        (leg["higher_target_ids"][0], leg["parent_geography_id"]): leg
        for group in receipt["groups"]
        for leg in group["legs"]
    }
    nested: list[NestedRows] = []
    for block, (control_id, start, stop) in zip(blocks, spans, strict=True):
        leg = legs.get((control_id, block.control_geography_id))
        if leg is None:
            raise ValueError(
                f"Nesting mechanism {mechanism_id!r}: the operator did not "
                f"reconcile the block at {block.control_geography_id!r}."
            )
        nested.append(
            NestedRows(
                values=tuple(values[start:stop].tolist()),
                factor=leg["declared_factor"],
                lower_sum=leg["old_total"],
            )
        )
    _record(mechanism_id, len(blocks), receipt)
    return nested, receipt


# ---------------------------------------------------------------------------
# The operator's own receipts
# ---------------------------------------------------------------------------

_RECEIPTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "us_target_nesting_receipts", default=None
)


def start_nesting_receipts() -> list[dict[str, Any]]:
    """Start a fresh record of the nesting passes that run from here on.

    The four tool functions keep the signatures and the receipts they had, so
    the operator's own receipt cannot travel in their results. A stage that
    wants it calls this first and writes the list with
    :func:`write_nesting_receipts` when its targets are made. Without a call,
    nothing is recorded.
    """

    receipts: list[dict[str, Any]] = []
    _RECEIPTS.set(receipts)
    return receipts


def stop_nesting_receipts() -> None:
    """End the record that :func:`start_nesting_receipts` started."""

    _RECEIPTS.set(None)


def _record(mechanism_id: str, blocks: int, receipt: Mapping[str, Any]) -> None:
    receipts = _RECEIPTS.get()
    if receipts is not None:
        mechanism = us_cross_grain_mechanism(mechanism_id)
        receipts.append(
            {
                "mechanism": mechanism_id,
                "issue_row": mechanism["issue_row"],
                "allocator": mechanism["allocator"],
                "blocks": blocks,
                "operator_receipt": dict(receipt),
            }
        )


def write_nesting_receipts(
    checkpoint_dir: Path, receipts: Sequence[Mapping[str, Any]]
) -> Path:
    """Write the recorded operator receipts beside the stage's other files.

    The file is a receipt only. No stage reads it, and it is not part of the
    run identity.
    """

    path = Path(checkpoint_dir) / NESTING_RECEIPTS_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operator": "microcosm.build.cross_grain",
                "declarations": "us/cross_grain_declarations.json",
                "passes": list(receipts),
            },
            indent=2,
        )
    )
    return path
