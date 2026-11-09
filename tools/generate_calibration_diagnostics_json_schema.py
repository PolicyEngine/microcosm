"""Regenerate or check the committed calibration-diagnostics JSON Schemas.

Without ``--schema-version`` the tool writes or checks both committed files:
schema 9, which current builds publish, and schema 8, for documents published
before it. ``--out`` writes or checks one schema at another path and needs
``--schema-version``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from microcosm.diagnostics import (
    assert_calibration_diagnostics_v8_json_schema_current,
    assert_calibration_diagnostics_v9_json_schema_current,
    committed_calibration_diagnostics_v8_json_schema_path,
    committed_calibration_diagnostics_v9_json_schema_path,
    write_calibration_diagnostics_v8_json_schema,
    write_calibration_diagnostics_v9_json_schema,
)

#: For each schema version: its committed path, its check and its writer.
_SCHEMAS: dict[
    int,
    tuple[
        Callable[[], Path],
        Callable[[str | Path | None], None],
        Callable[[str | Path | None], Path],
    ],
] = {
    9: (
        committed_calibration_diagnostics_v9_json_schema_path,
        assert_calibration_diagnostics_v9_json_schema_current,
        write_calibration_diagnostics_v9_json_schema,
    ),
    8: (
        committed_calibration_diagnostics_v8_json_schema_path,
        assert_calibration_diagnostics_v8_json_schema_current,
        write_calibration_diagnostics_v8_json_schema,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-version",
        type=int,
        choices=sorted(_SCHEMAS),
        help="the diagnostics schema to write or check (default: both)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write or check this file instead of the committed schema",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless the file equals freshly generated output",
    )
    args = parser.parse_args(argv)
    if args.out is not None and args.schema_version is None:
        parser.error("--out needs --schema-version")
    versions = [args.schema_version] if args.schema_version else list(_SCHEMAS)
    exit_code = 0
    for version in versions:
        committed_path, assert_current, write = _SCHEMAS[version]
        schema_file = args.out or committed_path()
        if args.check:
            try:
                assert_current(schema_file)
            except (OSError, ValueError) as error:
                print(f"stale: {error}", file=sys.stderr)
                exit_code = 1
                continue
            print(f"current: {schema_file}")
            continue
        print(write(schema_file))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
