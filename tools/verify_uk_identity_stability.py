"""Verify E4 stochastic stage identity stability on an existing UK frame."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from microcosm.build.uk_runtime.national_build import load_uk_national_frame


def identity_stability_receipt(
    frame,
    *,
    transform: Callable[[object], object],
    columns_by_entity: dict[str, Sequence[str]],
) -> dict[str, object]:
    """Run a transform on original and permuted rows, then compare by id."""

    original = transform(frame)
    permuted_input = _reverse_rows(frame)
    permuted = transform(permuted_input)
    mismatches: dict[str, list[str]] = {}
    for entity, columns in columns_by_entity.items():
        id_column = f"{entity}_id"
        left = original.table(entity).set_index(id_column)
        right = permuted.table(entity).set_index(id_column).reindex(left.index)
        bad = [
            column
            for column in columns
            if not left[column]
            .reset_index(drop=True)
            .equals(right[column].reset_index(drop=True))
        ]
        if bad:
            mismatches[entity] = bad
    return {
        "check": "uk_e4_identity_stability",
        "identical": not mismatches,
        "mismatches": mismatches,
        "columns_by_entity": {
            key: list(value) for key, value in columns_by_entity.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, _provenance = load_uk_national_frame(args.input_h5)
    receipt = {
        "check": "uk_e4_identity_stability",
        "input_h5": str(args.input_h5),
        "status": "requires caller-supplied E4 transform in acceptance harness",
        "entity_row_counts": {
            entity: int(len(frame.table(entity))) for entity in frame.entities
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _reverse_rows(frame):
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
        uk_time_period,
    )

    return uk_national_frame(
        person=frame.table("person").iloc[::-1].reset_index(drop=True),
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


if __name__ == "__main__":
    main()
