"""Pin the bonus-sacrifice anchor from the vendored OBR row and the locked engine.

The ``salary_sacrifice`` stage converts FRS bonuses into
``pension_contributions_via_bonus_sacrifice`` until policyengine-uk's projection
of them meets the OBR's 2027-28 bonus-sacrifice base above the GBP 2,000
threshold (OBR, supplementary forecast information on salary sacrifice, 5
February 2026, Table 1.1; microcosm#1340). The OBR row is the build's vendored
copy of the pinned Chronicle feed (``obr_salary_sacrifice_costing_facts.json``),
never a hand-copied figure.

policyengine-uk projects a dataset one year at a time, each year's amount the
year before's times one plus that year's growth (``data/economic_assumptions.py``,
``apply_single_year_uprating``): both sacrifice routes by the index its
``data/uprating_indices.yaml`` lists them under, and ``household_weight`` by
population. This tool writes those factors from the data year to the base year,
the OBR row and the upper earnings limit the conversion's candidates must reach
into ``uk/bonus_sacrifice_anchor.json`` with the engine version they came from.
A ``requires_uk`` test holds the file in lockstep with the installed engine, so
an engine bump re-runs this tool.

Usage (with the uk extra installed):

    uv run --no-sync python tools/pin_uk_bonus_sacrifice_anchor.py [--check]

``--check`` fails instead of writing when the committed file differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.build.uk_runtime.hmrc_uprating import (
    installed_engine_version,
    live_engine_parameter_value,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.salary_sacrifice import (
    BONUS_SACRIFICE_ANCHOR_RESOURCE,
    BONUS_SACRIFICE_BASE_YEAR,
    BONUS_SACRIFICE_DATA_YEAR,
    BONUS_SACRIFICE_FACTS_RESOURCE,
    BONUS_SACRIFICE_OUTPUT,
    BONUS_SACRIFICE_THRESHOLD,
    BONUS_SACRIFICE_UEL_INSTANT,
    BONUS_SACRIFICE_UEL_PARAMETER,
    SALSAC_OUTPUT,
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
#: The OBR line the anchor reads: the bonus route's base above the threshold,
#: in the base year.
_OBR_RECORD_SET = (
    f"obr.salary_sacrifice_costing_2026_02.tax_base.fy{BONUS_SACRIFICE_BASE_YEAR}"
)
_OBR_LINE = "bonus_sacrifice__above_2000"


def _growth(parameter: str) -> dict[str, object]:
    years = list(range(BONUS_SACRIFICE_DATA_YEAR + 1, BONUS_SACRIFICE_BASE_YEAR + 1))
    rates = [live_engine_parameter_value(parameter, f"{year}-01-01") for year in years]
    factor = 1.0
    for rate in rates:
        if not rate > -1:
            raise SystemExit(
                f"{parameter!r} is {rate!r}; a growth rate must exceed -1."
            )
        factor *= 1.0 + rate
    return {"parameter": parameter, "years": years, "rates": rates, "factor": factor}


def build_anchor() -> dict[str, object]:
    version = installed_engine_version()
    if version == "absent":
        raise SystemExit(
            "policyengine-uk is not installed; sync the uk extra before pinning."
        )
    from policyengine_uk.data.economic_assumptions import (
        _uprating_index_map,
        household_weight_growth_path,
    )

    indices = {
        variable: index
        for index, variables in _uprating_index_map().items()
        for variable in variables
        if variable in (SALSAC_OUTPUT, BONUS_SACRIFICE_OUTPUT)
    }
    if len(indices) != 2 or len(set(indices.values())) != 1:
        raise SystemExit(
            f"policyengine-uk {version} projects the two sacrifice routes by "
            f"{indices!r}; the anchor needs them on one index."
        )
    rows = [
        row
        for row in vendored_rows(BONUS_SACRIFICE_FACTS_RESOURCE)
        if (row.get("layout") or {}).get("record_set_id") == _OBR_RECORD_SET
        and (row.get("layout") or {}).get("groupby_value_id") == _OBR_LINE
    ]
    if len(rows) != 1:
        raise SystemExit(
            f"{BONUS_SACRIFICE_FACTS_RESOURCE} holds {len(rows)} rows for "
            f"{_OBR_RECORD_SET} {_OBR_LINE}, not one."
        )
    weekly = live_engine_parameter_value(
        BONUS_SACRIFICE_UEL_PARAMETER, BONUS_SACRIFICE_UEL_INSTANT
    )
    return {
        "schema_version": 1,
        "engine": {"package": "policyengine-uk", "version": version},
        "written_by": "tools/pin_uk_bonus_sacrifice_anchor.py",
        "source": {
            "resource": BONUS_SACRIFICE_FACTS_RESOURCE,
            "record_set_id": _OBR_RECORD_SET,
            "groupby_value_id": _OBR_LINE,
        },
        "obr_base_above_threshold": float(rows[0]["value"]),
        "threshold": int(BONUS_SACRIFICE_THRESHOLD),
        "data_year": BONUS_SACRIFICE_DATA_YEAR,
        "base_year": BONUS_SACRIFICE_BASE_YEAR,
        "earnings_growth": _growth(next(iter(indices.values()))),
        "weight_growth": _growth(household_weight_growth_path()),
        "upper_earnings_limit": {
            "parameter": BONUS_SACRIFICE_UEL_PARAMETER,
            "instant": BONUS_SACRIFICE_UEL_INSTANT,
            "weekly": weekly,
            # policyengine-uk's own weeks in a year.
            "annual": weekly * 52,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed anchor differs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=UK_PACKAGE_DIR / BONUS_SACRIFICE_ANCHOR_RESOURCE,
    )
    args = parser.parse_args(argv)
    payload = build_anchor()
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
    print(f"wrote {args.output} from policyengine-uk {payload['engine']['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
