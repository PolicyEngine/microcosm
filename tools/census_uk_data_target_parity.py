"""Regenerate or check the late uk-data target-parity register."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcosm.build.uk_runtime.data_target_parity import (
    assert_uk_data_target_parity_current,
    committed_uk_data_target_parity_path,
    write_uk_data_target_parity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = args.out or committed_uk_data_target_parity_path()
    if args.check:
        try:
            assert_uk_data_target_parity_current(path)
        except (OSError, ValueError) as error:
            print(f"stale: {error}", file=sys.stderr)
            return 1
        print(f"current: {path}")
        return 0
    print(write_uk_data_target_parity(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
