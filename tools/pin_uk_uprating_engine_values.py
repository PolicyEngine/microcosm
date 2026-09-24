"""Vendor the policyengine-uk parameter values the declared UK uprating appliers read.

The compile of the UK target registry moves SPI amount rows by the ratio of an
engine parameter between 1 January of the fact's opening year and 1 January of
the calibration year (``uk_runtime.hmrc_uprating``). Reading the engine at
compile time would make every compile, and every CI tier that compiles the
fixture subset, depend on the uk extra; instead this tool writes the values the
appliers can ask for, one per declared parameter per 1 January of
``UK_ENGINE_PIN_YEARS``, into ``uk/hmrc_uprating_engine_pins.json`` with the
engine version they came from. A ``requires_uk`` test holds the file in
lockstep with the installed engine, so an engine bump re-runs this tool.

Usage (with the uk extra installed):

    uv run --no-sync python tools/pin_uk_uprating_engine_values.py [--check]

``--check`` fails instead of writing when the committed file differs from the
installed engine's values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.build.uk_runtime.hmrc_uprating import (
    UK_ENGINE_INDEX_BASIS,
    UK_ENGINE_INDEX_PARAMETERS,
    UK_ENGINE_PIN_YEARS,
    UK_ENGINE_PINS_RESOURCE,
    installed_engine_version,
    live_engine_parameter_value,
)

UK_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk"
)


def build_pins() -> dict[str, object]:
    version = installed_engine_version()
    if version == "absent":
        raise SystemExit(
            "policyengine-uk is not installed; sync the uk extra before pinning."
        )
    instants = [f"{year}-01-01" for year in UK_ENGINE_PIN_YEARS]
    values: dict[str, dict[str, float]] = {}
    for path in UK_ENGINE_INDEX_PARAMETERS:
        by_instant: dict[str, float] = {}
        for instant in instants:
            value = live_engine_parameter_value(path, instant)
            if not value > 0:
                raise SystemExit(
                    f"policyengine-uk {version}: {path!r} is {value!r} at {instant}; "
                    "an uprating index must be positive."
                )
            by_instant[instant] = value
        values[path] = by_instant
    return {
        "schema_version": 1,
        "engine": {"package": "policyengine-uk", "version": version},
        "basis": UK_ENGINE_INDEX_BASIS,
        "written_by": "tools/pin_uk_uprating_engine_values.py",
        "instants": instants,
        "values": values,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed pins differ from the engine",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=UK_PACKAGE_DIR / UK_ENGINE_PINS_RESOURCE,
    )
    args = parser.parse_args(argv)
    payload = build_pins()
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        committed = (
            args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        )
        if committed != text:
            print(
                f"{args.output} differs from policyengine-uk "
                f"{payload['engine']['version']}; re-run without --check.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} matches policyengine-uk {payload['engine']['version']}")
        return 0
    args.output.write_text(text, encoding="utf-8")
    print(
        f"wrote {args.output}: {len(UK_ENGINE_INDEX_PARAMETERS)} parameters x "
        f"{len(payload['instants'])} instants from policyengine-uk "
        f"{payload['engine']['version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
