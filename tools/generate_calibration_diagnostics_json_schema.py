"""Regenerate or check the committed calibration-diagnostics schema-8 JSON Schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcosm.diagnostics import (
    assert_calibration_diagnostics_v8_json_schema_current,
    committed_calibration_diagnostics_v8_json_schema_path,
    write_calibration_diagnostics_v8_json_schema,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    path = args.out or committed_calibration_diagnostics_v8_json_schema_path()
    if args.check:
        try:
            assert_calibration_diagnostics_v8_json_schema_current(path)
        except (OSError, ValueError) as error:
            print(f"stale: {error}", file=sys.stderr)
            return 1
        print(f"current: {path}")
        return 0
    print(write_calibration_diagnostics_v8_json_schema(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
