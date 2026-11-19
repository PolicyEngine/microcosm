"""Generate NZ's reviewed ageing declarations and authenticated factor receipts.

Run with ``NZ_CHRONICLE_ARTIFACT`` or ``--artifact``. The nine operational
expectations are assertions of the feed calculation, never arithmetic inputs.
The observed-only ratio is diagnostic and is not available to profiles. The
three donor factors age calendar-2024 donor money to tax year 2026-27; only
money normalization may apply them.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)
from microcosm.build.reference_factors import resolve_reference_factor
from tools.generate_nz_target_references import SPEC, _selector

_QES_JUNE_2025_HASH = "25053dcbd343320cad72e2896475eaebaf5f5bb0b5965ca788506ca15efcea30"
_QES_JUNE_2026_HASH = "1af2e7e37f1c108a2656842cf1f519c903e1a982bcdc03ee02d0ad888ebc3a97"
_QES_SEPTEMBER_2026_HASH = "d6fb9a947185551292dd6e81a527d994615ae15053b800a11e364ddb915626f3"
_TREASURY_HASH = "e2486375dde515d0722f8c3022f481804bb014beb83c81b8bfc99bc9426544d9"
_OPERANDS = (
    (12028, "2023-Q2", "cfbce1343aa646fce0411631", "1502.27"),
    (12030, "2023-Q3", "f9f4f127d080fe2f06617947", "1528.92"),
    (12032, "2023-Q4", "fb5938810b7b37ae8534e5f7", "1556.16"),
    (12034, "2024-Q1", "854e91ec1dd95e396659ad8e", "1563.54"),
    (12036, "2024-Q2", "c07fd9ce3137fa20386bafe0", "1586.32"),
    (12038, "2024-Q3", "d5556ef840710807f613f744", "1595.11"),
    (12040, "2024-Q4", "245872b74d3fa4c5d3ff76ab", "1622.73"),
    (12044, "2025-Q2", "e8b5cdf88ef225693d90ab7d", "1652.72"),
    (12046, "2025-Q3", "2cc4746a60d99a9ec89bd329", "1661.12"),
    (12048, "2025-Q4", "f6a6066ab1b1efdb6a8c0d37", "1681.06"),
    (12050, "2026-Q1", "a8640856bd04c28d42ed22fa", "1685.39"),
    (12052, "2026-Q2", "b7af729aa786dd45cfec097e", "1700.68"),
    (12054, "2026-Q3", "8eacd5e7db3db27d35aa128a", "1714.86"),
)
_TY2024 = ("2023-Q2", "2023-Q3", "2023-Q4", "2024-Q1")
_TY2026 = ("2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1")
_CY2024 = ("2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4")
_TARGET = ("2026-Q2", "2026-Q3", "2026-Q4", "2027-Q1")
_EXPECTED = {
    "ird_ty2024_to_ty2027_low": 1117460,
    "ird_ty2024_to_ty2027_central": 1118491,
    "ird_ty2024_to_ty2027_high": 1119519,
    "p1_ty2026_to_ty2027_low": 1028904,
    "p1_ty2026_to_ty2027_central": 1029853,
    "p1_ty2026_to_ty2027_high": 1030799,
    "observed_ty2026_over_ty2024": 1086069,
    "donor_cy2024_to_ty2027_low": 1079413,
    "donor_cy2024_to_ty2027_central": 1080408,
    "donor_cy2024_to_ty2027_high": 1081401,
}
_DECISIONS = [
    "d1599; IRD band ageing factors, 22 October 2026",
    "G8-r2 hub rulings 3, 23 October 2026, sections 1–2",
    "NZ hub QES September 2026 re-declaration, 4 November 2026: each quarter from the newest Stats NZ vintage that publishes it (chronicle#405)",
]
_DONOR_DECISIONS = [
    "d1952 card; NZ money year build item: donor money is calendar-2024 dollars",
    "G8-w hub rulings 5, 24 October 2026, items H1–H3",
    "G8-r2 hub rulings 3, 23 October 2026, section 1",
    "NZ hub QES September 2026 re-declaration, 4 November 2026: each quarter from the newest Stats NZ vintage that publishes it (chronicle#405)",
]


def _coverage(period):
    year, quarter = period.split("-Q")
    year, quarter = int(year), int(quarter)
    start_month = 3 * quarter - 2
    end_day = (31, 30, 30, 31)[quarter - 1]
    return {
        "start_date": f"{year:04d}-{start_month:02d}-01",
        "end_date": f"{year:04d}-{start_month + 2:02d}-{end_day}",
    }


def _qes_hash(period):
    """Each quarter comes from the newest QES vintage that publishes it."""
    if period in _TY2024:
        return _QES_JUNE_2025_HASH
    if period == "2024-Q2":
        return _QES_JUNE_2026_HASH
    return _QES_SEPTEMBER_2026_HASH


def _validation(fact):
    return {
        "unit": fact["observed_measure"]["unit"],
        "source_sha256": fact["source"]["source_sha256"],
        "source_vintage": fact["source"]["vintage"],
        "entity_role": fact["entity"]["role"],
        "provenance_class": fact["provenance_class"],
        "assertion": fact["assertion"],
        "period_coverage": {
            key: fact["period_coverage"][key] for key in ("start_date", "end_date")
        },
        "source_column_id": fact["layout"]["source_column_id"],
    }


def build_aging_declarations(artifact, *, package_dir=SPEC):
    """Author the reviewed fixed selectors, refusing a changed operand or pin."""
    feed_pin = json.loads((package_dir / "chronicle_feed.json").read_text())
    validate_chronicle_feed_pin(artifact, feed_pin)
    feed_pin = chronicle_feed_identity(feed_pin)
    facts_by_key = {fact["aggregate_fact_key"]: fact for fact in artifact.facts}
    qes = {}
    for _line, period, suffix, expected in _OPERANDS:
        fact = facts_by_key[f"ledger.aggregate_fact.v2:{suffix}"]
        if str(fact["value"]) != expected or fact["period"]["value"] != period:
            raise ValueError(f"Reviewed QES operand changed for {period}")
        expected_hash = _qes_hash(period)
        if (
            fact["source"]["source_sha256"] != expected_hash
            or fact["observed_measure"]["source_measure_id"] != "SBSZ9A"
            or fact["observed_measure"]["unit"] != "nzd_per_week"
            or fact["assertion"] != "observation"
        ):
            raise ValueError(f"Reviewed QES evidence changed for {period}")
        qes[period] = fact
    treasury = facts_by_key["ledger.aggregate_fact.v2:aba7d3d54489b068116e6a25"]
    if (
        str(treasury["value"]) != "2.7"
        or treasury["period"]["value"] != 2026
        or treasury["source"]["source_sha256"] != _TREASURY_HASH
        or treasury["layout"]["source_column_id"] != "year_ending_30_june_2027"
    ):
        raise ValueError("Reviewed PREFU operand changed")

    def schedule(periods, *, target=False):
        result = []
        for index, period in enumerate(periods):
            fact = qes.get(period, qes["2026-Q2"])
            selector = _selector(fact, pin_key=period in qes)
            selector["assertion"] = "observation"
            selector["period_value"] = period
            selector["record_set_id"] = (
                f"stats_nz.qes.awe.{period.lower().replace('-', '')}"
            )
            validation = _validation(fact)
            validation["period_coverage"] = _coverage(period)
            slot = {
                "selector": selector,
                "validation": validation,
                "weight": "1",
                "sequence": index,
            }
            if target:
                slot["projection_offset"] = index
            result.append(slot)
        return result

    growth_selector = _selector(treasury)
    growth_selector["assertion"] = "source_projection"
    growth_selector["source_table"] = treasury["observed_measure"]["source_table"]
    declarations = []
    for prefix, periods, source_period in (
        ("ird_ty2024", _TY2024, 2024),
        ("p1_ty2026", _TY2026, 2026),
    ):
        for profile, delta in (("low", "-0.5"), ("central", "0"), ("high", "0.5")):
            factor_id = f"{prefix}_to_ty2027_{profile}"
            declarations.append(
                {
                    "schema_version": 1,
                    "id": factor_id,
                    "operation": "projected_period_average_ratio",
                    "operation_version": 1,
                    "purpose": "profile_input",
                    "feed_identity": copy.deepcopy(feed_pin),
                    "source_period": source_period,
                    "target_period": 2027,
                    "source_schedule": schedule(periods),
                    "target_schedule": schedule(_TARGET, target=True),
                    "growth": {
                        "selector": copy.deepcopy(growth_selector),
                        "validation": _validation(treasury),
                        "unit_multiplier": "0.01",
                        "scenario_delta": delta,
                        "assertion_policy": "allow_source_projection",
                        "sensitivity_provenance": [
                            "d1599: symmetric ±0.5 percentage-point Microcosm sensitivity; high is not a Treasury forecast path"
                        ],
                    },
                    "subperiods_per_year": 4,
                    "projection_policy": "observed_preferred_latest_contiguous",
                    "rounding": {"digits": 6, "mode": "nearest_half_even"},
                    "expected_operational_factor": {
                        "numerator": _EXPECTED[factor_id],
                        "denominator": 1000000,
                    },
                    "validity": {
                        "declared_on": "2026-11-04",
                        "valid_from": "2026-11-04",
                        "valid_until": "2027-02-02",
                        "as_of": "2026-11-04",
                    },
                    "evidence": [
                        "Stats NZ QES Table 7; QEMQ.SBSZ9A; ordinary-time FTE, all sectors, actual",
                        "Treasury PREFU 2026 Table 4.1; year ending 30 June 2027; Chronicle period 2026",
                    ],
                    "decision_references": list(_DECISIONS),
                }
            )
    diagnostic = copy.deepcopy(declarations[1])
    diagnostic.update(
        {
            "id": "observed_ty2026_over_ty2024",
            "purpose": "diagnostic",
            "target_period": 2026,
            "target_schedule": schedule(_TY2026, target=True),
            "growth": None,
            "expected_operational_factor": {
                "numerator": _EXPECTED["observed_ty2026_over_ty2024"],
                "denominator": 1000000,
            },
        }
    )
    declarations.append(diagnostic)
    for profile in ("low", "central", "high"):
        donor = copy.deepcopy(
            next(
                row
                for row in declarations
                if row["id"] == f"ird_ty2024_to_ty2027_{profile}"
            )
        )
        factor_id = f"donor_cy2024_to_ty2027_{profile}"
        donor.update(
            {
                "id": factor_id,
                "purpose": "money_normalization_input",
                "source_period": "CY2024",
                "source_schedule": schedule(_CY2024),
                "expected_operational_factor": {
                    "numerator": _EXPECTED[factor_id],
                    "denominator": 1000000,
                },
                "evidence": [
                    *donor["evidence"],
                    "Declared calendar-2024 money convention (H1), covering "
                    "flows and asset stocks; the recorded _time_period is a "
                    "consistency check, not authentication",
                ],
                "decision_references": list(_DONOR_DECISIONS),
            }
        )
        declarations.append(donor)
    return feed_pin, declarations


def build_reference_aging(artifact, *, package_dir=SPEC):
    feed_pin, declarations = build_aging_declarations(artifact, package_dir=package_dir)
    records = []
    for declaration in declarations:
        factor, receipt = resolve_reference_factor(
            declaration, artifact.facts, feed_pin
        )
        records.append(
            {
                "declaration": declaration,
                "operational_factor": factor,
                "receipt": receipt,
            }
        )
    return {"schema_version": 1, "feed_identity": feed_pin, "factors": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("NZ_CHRONICLE_ARTIFACT"))
    parser.add_argument("--out", type=Path, default=SPEC / "reference_aging.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.artifact:
        parser.error("--artifact or NZ_CHRONICLE_ARTIFACT is required")
    artifact = load_ledger_consumer_artifact(args.artifact, lexical_floats=True)
    document = build_reference_aging(artifact)
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if args.out.read_text() != encoded:
            raise SystemExit(f"Stale generated resource: {args.out}")
    else:
        args.out.write_text(encoded)


if __name__ == "__main__":
    main()
