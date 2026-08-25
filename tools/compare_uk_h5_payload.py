"""Compare two UK single-year H5 artifacts at payload level (#612 acceptance).

Byte identity is not a valid acceptance test for microcosm HDF5 artifacts —
``HDFStore`` stamps object-header write times, so two runs of an unchanged
writer already differ in bytes. What downstream readers depend on is the
**payload**: the same store keys in the same write order, the same per-table
column list in order, per-column dtypes, index type and values, row order,
values, and the same HDF5 root attributes. This tool asserts exactly that
surface between two artifacts — the #612 carrier swap's old-vs-new staging
acceptance, and thereafter the standing instrument for any writer or carrier
refactor that claims "the artifact did not change".

Disclosure control: the output is written to be publishable under the UK Data
Service EUL (CD137 §8 / CD171 §5.2.1). Differences are reported as column
names, dtype names, booleans, and threshold-guarded row counts — never as
unit-record values. Reads are ``mode="r"`` throughout.

``--structure-only`` re-verdicts the same measurements for the #686 swap
comparison, where the control and candidate artifacts are expected to share a
surface exactly and to differ in values only where a difference is signed:
every structural predicate stays strict, and each differing column or root
attribute must name an entry in the committed signed-differences register.
``payload_identical`` is still computed and reported either way, so a
structure-only receipt stays comparable with a full-mode one.

Exit code: 0 when the payloads are identical — or, under ``--structure-only``,
when the structures match and every value difference is signed; 1 when they
differ; and 2 when no verdict is possible — an unsafe CLI configuration, or an
artifact that could not be read (reported with exception text suppressed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SDC_MINIMUM_COUNT = 10
# CD171 §5.2.1: cells based on one or two cases are never reportable. Callers
# may raise the threshold, never lower it.
MINIMUM_SDC_COUNT = 3


def _validated_sdc_minimum(minimum: int) -> int:
    minimum = int(minimum)
    if minimum < MINIMUM_SDC_COUNT:
        raise ValueError(f"SDC minimum count must be at least {MINIMUM_SDC_COUNT}.")
    return minimum


def sdc_count(count: int, *, minimum: int) -> int | str:
    """Mask small nonzero counts: zero and counts >= minimum are safe."""

    minimum = _validated_sdc_minimum(minimum)
    count = int(count)
    if count == 0 or count >= minimum:
        return count
    return f"< {minimum}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_alias(left: Path, right: Path) -> bool:
    """Compare lexical/resolved identity and filesystem inode identity."""

    resolved_alias = False
    try:
        resolved_alias = left.resolve(strict=False) == right.resolve(strict=False)
    except (OSError, RuntimeError):
        pass
    filesystem_alias = False
    try:
        filesystem_alias = left.samefile(right)
    except OSError:
        pass
    return resolved_alias or filesystem_alias


def _series_equal_mask(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Row-wise equality treating aligned NaN as equal, without leaking values.

    Integer and boolean pairs compare on their raw values: a float64 cast
    would silently equate int64 values above 2**53, exactly the magnitude
    regime rowwise id multiplication grows toward.
    """

    integer_like = (
        pd.api.types.is_integer_dtype(left) or pd.api.types.is_bool_dtype(left)
    ) and (pd.api.types.is_integer_dtype(right) or pd.api.types.is_bool_dtype(right))
    if integer_like and not left.isna().any() and not right.isna().any():
        return left.to_numpy() == right.to_numpy()
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_numeric = left.to_numpy(dtype="float64", na_value=np.nan)
        right_numeric = right.to_numpy(dtype="float64", na_value=np.nan)
        both_nan = np.isnan(left_numeric) & np.isnan(right_numeric)
        return (left_numeric == right_numeric) | both_nan
    left_values = left.to_numpy()
    right_values = right.to_numpy()
    left_na = pd.isna(left).to_numpy()
    right_na = pd.isna(right).to_numpy()
    equal = np.zeros(len(left), dtype=bool)
    comparable = ~left_na & ~right_na
    equal[comparable] = left_values[comparable] == right_values[comparable]
    equal |= left_na & right_na
    return equal


def compare_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Compare one table pair; report classifications, never values."""

    minimum = _validated_sdc_minimum(minimum)
    report: dict[str, Any] = {
        "rows": {"left": int(len(left)), "right": int(len(right))},
        "row_count_equal": bool(len(left) == len(right)),
        "column_order_equal": bool(list(left.columns) == list(right.columns)),
        "columns_only_left": sorted(set(left.columns) - set(right.columns)),
        "columns_only_right": sorted(set(right.columns) - set(left.columns)),
        "index_type_equal": bool(type(left.index) is type(right.index)),
        "index_dtype_equal": bool(str(left.index.dtype) == str(right.index.dtype)),
        "index_name_equal": bool(left.index.name == right.index.name),
        "index_values_equal": bool(
            len(left) == len(right) and left.index.equals(right.index)
        ),
    }
    dtype_mismatches: dict[str, dict[str, str]] = {}
    value_mismatches: dict[str, int | str] = {}
    for column in left.columns:
        if column not in right.columns:
            continue
        left_dtype = left[column].dtype
        right_dtype = right[column].dtype
        # Compare the dtype OBJECTS: two categorical dtypes both stringify
        # to "category" while differing in categories or the ordered flag.
        try:
            dtypes_equal = bool(left_dtype == right_dtype)
        except TypeError:
            dtypes_equal = False
        if not dtypes_equal:
            mismatch = {"left": str(left_dtype), "right": str(right_dtype)}
            if mismatch["left"] == mismatch["right"]:
                mismatch["note"] = (
                    "dtypes differ beyond their string names "
                    "(e.g. categorical categories or ordered flag)"
                )
            dtype_mismatches[column] = mismatch
        if len(left) == len(right):
            equal = _series_equal_mask(left[column], right[column])
            differing = int((~equal).sum())
            if differing:
                value_mismatches[column] = sdc_count(differing, minimum=minimum)
    report["dtype_mismatches"] = dtype_mismatches
    report["value_mismatch_rows_by_column"] = value_mismatches
    report["payload_equal"] = bool(
        report["row_count_equal"]
        and report["column_order_equal"]
        and not report["columns_only_left"]
        and not report["columns_only_right"]
        and report["index_type_equal"]
        and report["index_dtype_equal"]
        and report["index_name_equal"]
        and report["index_values_equal"]
        and not dtype_mismatches
        and not value_mismatches
    )
    return report


def _read_store(path: Path) -> tuple[list[str], dict[str, Any]]:
    with pd.HDFStore(path, mode="r") as store:
        keys = [key.lstrip("/") for key in store.keys()]
        objects = {key: store[key] for key in keys}
    return keys, objects


def _read_root_attrs(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to read UK national metadata.") from exc
    with h5py.File(path, mode="r") as file:
        attrs = {}
        for name in file.attrs:
            value = file.attrs[name]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            attrs[str(name)] = value
        return attrs


def _attr_values_equal(left: Any, right: Any) -> bool:
    """Raw-value attr equality: a str() comparison masks array and typed
    differences (numpy truncates long arrays in str()), so arrays compare
    element-wise and everything else compares on value and type name."""

    if type(left).__name__ != type(right).__name__:
        return False
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    return bool(left == right)


def compare_uk_h5_payload(
    left_path: Path,
    right_path: Path,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Compare two artifacts over the full payload surface."""

    minimum = _validated_sdc_minimum(minimum)
    left_keys, left_objects = _read_store(left_path)
    right_keys, right_objects = _read_store(right_path)
    left_attrs = _read_root_attrs(left_path)
    right_attrs = _read_root_attrs(right_path)

    tables: dict[str, Any] = {}
    for key in left_keys:
        if key not in right_objects:
            continue
        left_object = left_objects[key]
        right_object = right_objects[key]
        # The stored kind is payload: a DataFrame where readers expect a
        # Series changes what loaders parse, so normalizing for comparison
        # must not hide the difference.
        stored_kinds = {
            "left": type(left_object).__name__,
            "right": type(right_object).__name__,
        }
        if isinstance(left_object, pd.Series):
            left_object = left_object.to_frame(name=key)
        if isinstance(right_object, pd.Series):
            right_object = right_object.to_frame(name=key)
        table_report = compare_tables(left_object, right_object, minimum=minimum)
        table_report["stored_kind"] = stored_kinds
        table_report["stored_kind_equal"] = bool(
            stored_kinds["left"] == stored_kinds["right"]
        )
        table_report["payload_equal"] = bool(
            table_report["payload_equal"] and table_report["stored_kind_equal"]
        )
        tables[key] = table_report

    attr_names_equal = list(left_attrs) == list(right_attrs)
    attrs_differing = sorted(
        name
        for name in set(left_attrs) & set(right_attrs)
        if not _attr_values_equal(left_attrs[name], right_attrs[name])
    )
    report: dict[str, Any] = {
        "left": str(left_path),
        # Digest-bind the verdict to the exact bytes compared, so a committed
        # receipt is verifiable offline against named artifacts.
        "left_sha256": _sha256(left_path),
        "right_sha256": _sha256(right_path),
        "right": str(right_path),
        "keys_equal": bool(left_keys == right_keys),
        "keys_only_left": sorted(set(left_keys) - set(right_keys)),
        "keys_only_right": sorted(set(right_keys) - set(left_keys)),
        "tables": tables,
        "root_attrs": {
            "names_in_order_equal": bool(attr_names_equal),
            "attrs_only_left": sorted(set(left_attrs) - set(right_attrs)),
            "attrs_only_right": sorted(set(right_attrs) - set(left_attrs)),
            # Attribute names are schema; which one differs is reportable,
            # the differing values are not necessarily, so names only.
            "attrs_with_differing_values": attrs_differing,
        },
    }
    report["payload_identical"] = bool(
        report["keys_equal"]
        and all(table["payload_equal"] for table in tables.values())
        and attr_names_equal
        and not attrs_differing
    )
    return report


def _structure_equal(report: dict[str, Any]) -> bool:
    """Whether the two artifacts have the same shape, ignoring values.

    Everything ``payload_identical`` asserts except the per-column value
    comparison: the same keys, stored kinds, row counts, column lists in
    order, dtypes, indexes, and root-attribute names.
    """

    if not report["keys_equal"]:
        return False
    if report["keys_only_left"] or report["keys_only_right"]:
        return False
    for table in report["tables"].values():
        if not (
            table["row_count_equal"]
            and table["column_order_equal"]
            and table["stored_kind_equal"]
            and table["index_type_equal"]
            and table["index_dtype_equal"]
            and table["index_name_equal"]
            and table["index_values_equal"]
        ):
            return False
        if table["columns_only_left"] or table["columns_only_right"]:
            return False
        if table["dtype_mismatches"]:
            return False
    attrs = report["root_attrs"]
    return bool(
        attrs["names_in_order_equal"]
        and not attrs["attrs_only_left"]
        and not attrs["attrs_only_right"]
    )


def apply_structure_only_verdict(
    report: dict[str, Any], register: Any
) -> dict[str, Any]:
    """Re-verdict a full-payload report as structure-only against a register.

    The swap comparison expects the control and candidate artifacts to have
    an identical surface and to differ only where a difference is signed, so
    this keeps every structural predicate strict and requires each differing
    column and root attribute to name a register entry. Lookup is
    expectation-aware: a value mismatch is covered by a ``column_differs``
    entry on ``payload_column`` or, through the register's payload bridge, on
    a value-bearing surface (``nonzero_shares``, ``weighted_totals``) — the
    share instrument and this one read the same adjudicated fact. Structural
    expectations never excuse a value difference.

    ``payload_identical`` is left exactly as computed, so a structure-only
    receipt stays comparable with a full-mode one.
    """

    matched: set[str] = set()
    unsigned_columns: list[str] = []
    for key, table in report["tables"].items():
        signed_ids: dict[str, str | None] = {}
        for column in sorted(table["value_mismatch_rows_by_column"]):
            # The store key is the entity whose table this column lives in;
            # passing it stops a household-scoped adjudication signing a
            # same-named person column.
            entry = register.matching(
                surface="payload_column",
                column=column,
                expectation="column_differs",
                entity=key,
            )
            signed_ids[column] = entry.id if entry else None
            if entry is None:
                unsigned_columns.append(f"{key}.{column}")
            else:
                matched.add(entry.id)
        table["value_mismatch_signed_ids"] = signed_ids

    unsigned_attrs: list[str] = []
    attr_signed: dict[str, str | None] = {}
    for name in report["root_attrs"]["attrs_with_differing_values"]:
        entry = register.matching(
            surface="root_attr", column=name, expectation="column_differs"
        )
        attr_signed[name] = entry.id if entry else None
        if entry is None:
            unsigned_attrs.append(name)
        else:
            matched.add(entry.id)
    report["root_attrs"]["differing_signed_ids"] = attr_signed

    structure_equal = _structure_equal(report)
    report["verdict_mode"] = "structure_only"
    report["structure_equal"] = structure_equal
    report["signed"] = {
        "resource": "spine_swap_signed_differences.json",
        "matched_ids": sorted(matched),
        "unsigned_columns": sorted(unsigned_columns),
        "unsigned_root_attrs": sorted(unsigned_attrs),
        "unused_ids": sorted(
            difference.id
            for difference in register.differences
            if difference.id not in matched
        ),
    }
    report["structure_only_ok"] = bool(
        structure_equal and not unsigned_columns and not unsigned_attrs
    )
    return report


def _load_signed_register(override: Path | None) -> Any:
    """Load the signed-differences register the structure-only verdict uses."""

    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "packages" / "microcosm-build" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from microcosm.build.uk_runtime.signed_differences import (
        load_uk_spine_swap_signed_differences,
    )

    if override is not None:
        return load_uk_spine_swap_signed_differences(str(override))
    return load_uk_spine_swap_signed_differences()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two UK single-year H5 artifacts at payload level "
            "(keys, column order, dtypes, index, values, root attrs), "
            "with SDC-safe output."
        )
    )
    parser.add_argument("left", type=Path, help="Baseline H5 (opened read-only).")
    parser.add_argument("right", type=Path, help="Candidate H5 (opened read-only).")
    parser.add_argument(
        "--sdc-minimum-count",
        type=int,
        default=DEFAULT_SDC_MINIMUM_COUNT,
        help=(
            "Differing-row counts below this (other than zero) are reported "
            f"as a threshold, never exactly (default: {DEFAULT_SDC_MINIMUM_COUNT}; "
            f"minimum: {MINIMUM_SDC_COUNT})."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the report JSON to this path.",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help=(
            "Verdict on shape rather than bytes (#686 swap acceptance): every "
            "structural predicate stays strict, and each differing column or "
            "root attribute must name an entry in the signed-differences "
            "register. Values are expected to differ where a difference is "
            "signed; anything else still fails."
        ),
    )
    parser.add_argument(
        "--signed-differences",
        type=Path,
        default=None,
        help=(
            "Override the committed signed-differences register used by "
            "--structure-only (tests only; the packaged resource is the "
            "contract)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        minimum = _validated_sdc_minimum(args.sdc_minimum_count)
    except ValueError:
        print(
            f"error: --sdc-minimum-count must be at least {MINIMUM_SDC_COUNT}.",
            file=sys.stderr,
        )
        return 2
    if _paths_alias(args.left, args.right):
        print("error: left and right must be distinct artifacts.", file=sys.stderr)
        return 2
    if args.json_out is not None and (
        _paths_alias(args.json_out, args.left)
        or _paths_alias(args.json_out, args.right)
    ):
        print("error: --json-out must not alias either H5 input.", file=sys.stderr)
        return 2

    if args.signed_differences is not None and not args.structure_only:
        print(
            "error: --signed-differences only applies to --structure-only.",
            file=sys.stderr,
        )
        return 2

    try:
        report = compare_uk_h5_payload(args.left, args.right, minimum=minimum)
        if args.structure_only:
            register = _load_signed_register(args.signed_differences)
            report = apply_structure_only_verdict(report, register)
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except Exception:
        # An unreadable or malformed artifact is not a "payloads differ"
        # verdict, and exception text from licensed data is never echoed —
        # the operator pastes this terminal to the tracking issue.
        print(
            "error: comparison could not be completed; exception text was "
            "suppressed. Verify both artifacts are readable UK single-year "
            "H5 files.",
            file=sys.stderr,
        )
        return 2

    if args.json_out is not None and (
        _paths_alias(args.json_out, args.left)
        or _paths_alias(args.json_out, args.right)
    ):
        print("error: --json-out became an alias of an H5 input.", file=sys.stderr)
        return 2
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    if args.structure_only:
        if report["structure_only_ok"]:
            return 0
        if not report["structure_equal"]:
            print(
                "DIFFER: the two artifacts do not share a structure.",
                file=sys.stderr,
            )
        else:
            signed = report["signed"]
            print(
                "UNSIGNED: value differences with no register entry: "
                + ", ".join(
                    (signed["unsigned_columns"] + signed["unsigned_root_attrs"])[:20]
                ),
                file=sys.stderr,
            )
        return 1

    if not report["payload_identical"]:
        print("DIFFER: the two artifacts are not payload-identical.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
