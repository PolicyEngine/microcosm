"""Regenerate the US release input-column coverage manifest (populace #368).

The manifest is the declared column-coverage contract the release gate enforces:
every input column the reference eCPS exports must be persisted by a populace
release as a real key with non-default signal, or carry a reviewed exclusion.

Derivation (fully from checked-in, sha-pinned facts — no transient artifact):

- Required surface = every input-variable column the pinned reference eCPS
  populates, i.e. the ``nonzero_shares`` keys of ``ecps_parity_reference.json``,
  plus explicit later inputs needed by shipped reform probes
  (computed once from the sha-verified ``enhanced_cps_2024.h5``; an input the
  incumbent exports but leaves all-zero is not a coverage requirement, the same
  rule the parity gate uses).
- Status per column:
    * ``reviewed_exclusion`` — the column is a documented incumbent-parity gap
      (an entry in ``ecps_parity_known_gaps.json``, carrying that register's
      reason and tracking issue), so the current candidate does not populate it
      yet. EXCEPT the SSI countable-resource asset inputs (below).
    * ``required`` — every other populated layer, PLUS the SSI countable-resource
      asset inputs. Per #368 ("it must be an actual gate"), the asset inputs get
      NO exclusion even though they are currently absent, so the gate ships red
      for today's artifacts and Deliverable 2 (asset restoration) turns it green.

The SSI asset inputs are ``bank_account_assets``, ``stock_assets``, and
``bond_assets`` — the exact ``adds`` list of ``ssi_countable_resources``. With
them absent, countable resources are 0 for every record, so the SSI asset-limit
reform probe scores $0; that is the failure the gate exists to surface.

Run:  uv run python tools/build_us_release_input_coverage_manifest.py
It rewrites packages/populace-build/src/populace/build/us/
release_input_coverage_manifest.json. A test asserts the committed file matches
this regeneration, so the manifest cannot silently drift from the pinned eCPS
surface or the parity register.
"""

from __future__ import annotations

import json
from pathlib import Path

US_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "populace-build"
    / "src"
    / "populace"
    / "build"
    / "us"
)
MANIFEST_PATH = US_PACKAGE_DIR / "release_input_coverage_manifest.json"

#: The ``adds`` list of PolicyEngine-US ``ssi_countable_resources``: the asset
#: leaves SSI eligibility keys on. Absent from the release, they zero countable
#: resources and the SSI asset-limit reform scores $0. Per #368 these ship as
#: hard requirements with NO reviewed exclusion.
SSI_COUNTABLE_RESOURCE_ASSETS = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)

# The pinned reference H5 predates the retired pipeline's FLSA-premium export
# and OBBBA's distinct qualifying passenger-vehicle interest leaf.  Both later
# inputs are hard requirements because the shipped OBBBA provisions must bind.
POST_REFERENCE_ECPS_REQUIRED_INPUTS = (
    "fsla_overtime_premium",
    "qualified_passenger_vehicle_loan_interest",
)

#: Pinned reform-coverage probes. Raising the SSI resource limit from
#: the 2024 statutory $2,000 individual / $3,000 couple to $10,000 / $20,000 is
#: a pure relaxation that binds only through ``ssi_countable_resources``. Nonzero
#: iff the asset inputs are restored. Dense-native reference magnitudes: +$1.6B
#: at $10k/$20k and ~+$16.1B with no limit (populace #356). ``min_abs_effect`` is
#: a floor far below the plausible effect but far above simulation noise, so a
#: structural $0 fails while a real (even conservative) score passes.
REFORM_COVERAGE_PROBES = [
    {
        "id": "ssi_asset_limit_10k_20k",
        "name": "SSI asset limits raised to $10k individual / $20k couple",
        "parameter_changes": {
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10_000
            },
            "gov.ssa.ssi.eligibility.resources.limit.couple": {
                "2024-01-01.2100-12-31": 20_000
            },
        },
        "budget_measure": "ssi",
        "period": 2024,
        "effect_direction": "reform_minus_baseline",
        "expected_sign": "positive",
        "binding_inputs": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "min_abs_effect": 1_000_000_000.0,
        "reason": (
            "Raising the SSI resource limit is a pure relaxation that only "
            "changes who passes meets_ssi_resource_test, which compares "
            "ssi_countable_resources = bank_account_assets + stock_assets + "
            "bond_assets against the limit. With those asset inputs absent, "
            "countable resources are 0 for every record, everyone already "
            "passes the resource test, and raising the limit scores exactly "
            "$0. Dense-native reference: +$1.6B at $10k/$20k, ~+$16.1B with no "
            "limit (PolicyEngine/populace#356)."
        ),
        "issue": "PolicyEngine/populace#356",
    },
    {
        "id": "obbba_no_tax_on_tips",
        "name": "OBBBA no-tax-on-tips deduction",
        "parameter_changes": {
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": [
            "tip_income",
            "treasury_tipped_occupation_code",
        ],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA tip-deduction cap to zero removes the deduction, "
            "so baseline-minus-reform income tax must be negative in 2026. "
            "With tip_income or the Treasury tipped-occupation code absent, "
            "qualified tip income is zero and the repeal scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#38",
    },
    {
        "id": "obbba_no_tax_on_overtime",
        "name": "OBBBA no-tax-on-overtime deduction",
        "parameter_changes": {
            "gov.irs.deductions.overtime_income.cap.JOINT": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.HEAD_OF_HOUSEHOLD": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SURVIVING_SPOUSE": {
                "2026-01-01.2026-12-31": 0
            },
            "gov.irs.deductions.overtime_income.cap.SEPARATE": {
                "2026-01-01.2026-12-31": 0
            },
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["fsla_overtime_premium"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting every OBBBA overtime-deduction cap to zero removes the "
            "deduction, so reform income tax rises and baseline-minus-reform "
            "must be negative in 2026. With fsla_overtime_premium absent or "
            "degenerate, qualified overtime is zero and the repeal scores $0."
        ),
        "issue": "PolicyEngine/populace#242",
    },
    {
        "id": "obbba_auto_loan_interest",
        "name": "OBBBA no-tax-on-auto-loan-interest deduction",
        "parameter_changes": {
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        "budget_measure": "income_tax",
        "period": 2026,
        "effect_direction": "baseline_minus_reform",
        "expected_sign": "negative",
        "binding_inputs": ["qualified_passenger_vehicle_loan_interest"],
        "min_abs_effect": 100_000_000.0,
        "reason": (
            "Setting the OBBBA auto-loan-interest deduction cap to zero "
            "removes the deduction, so reform income tax rises and "
            "baseline-minus-reform must be negative in 2026. With qualified "
            "passenger-vehicle loan interest absent or degenerate, the repeal "
            "scores exactly $0."
        ),
        "issue": "PolicyEngine/populace#252",
    },
]


def _load(name: str) -> dict:
    return json.loads((US_PACKAGE_DIR / name).read_text(encoding="utf-8"))


def build_manifest() -> dict:
    parity = _load("ecps_parity_reference.json")
    known_gaps = _load("ecps_parity_known_gaps.json")["known_gaps"]

    populated_layers = {
        name for name, share in parity["nonzero_shares"].items() if float(share) > 0.0
    } | set(POST_REFERENCE_ECPS_REQUIRED_INPUTS)
    ssi_assets = set(SSI_COUNTABLE_RESOURCE_ASSETS)

    missing_assets = sorted(ssi_assets - populated_layers)
    if missing_assets:
        raise ValueError(
            "SSI countable-resource asset inputs are not in the reference eCPS "
            f"populated surface, cannot pin them as required: {missing_assets}."
        )

    columns: dict[str, dict] = {}
    for name in sorted(populated_layers):
        if name in known_gaps and name not in ssi_assets:
            entry = known_gaps[name]
            columns[name] = {
                "status": "reviewed_exclusion",
                "reason": str(entry["reason"]),
                "issue": str(entry["issue"]),
            }
        else:
            column = {"status": "required"}
            if name in ssi_assets:
                # These are currently absent; #368 forbids an exclusion so the
                # gate fails until Build J restores them.
                column["note"] = (
                    "SSI countable-resource asset input; required with NO "
                    "reviewed exclusion per PolicyEngine/populace#368 so the "
                    "gate fails until the asset stage is restored (Deliverable "
                    "2). Currently absent — this is the intended red gate."
                )
            columns[name] = column

    required = sorted(n for n, c in columns.items() if c["status"] == "required")
    reviewed = sorted(
        n for n, c in columns.items() if c["status"] == "reviewed_exclusion"
    )

    # Provenance without naming the retired data package. Only the eCPS parity
    # reference, its loader, and its test are allow-listed to name the incumbent
    # (test_us_plan.test_no_incumbent_data_package_references_in_live_tree); this
    # manifest is not, so it records the sha-locked coordinates that DON'T name a
    # package (filename + content sha + revision + vintage) and points at that
    # allow-listed parity reference for the full record. Nothing reads these
    # fields — the gate and the anti-rot check derive the surface from
    # ecps_parity_reference.json directly — so this stays pure documentation.
    parity_source = dict(parity["source"])
    reference = {
        "derived_from": "ecps_parity_reference.json",
        "filename": str(parity_source.get("filename", "")),
        "revision": str(parity_source.get("revision", "")),
        "sha256": str(parity_source.get("sha256", "")),
        "vintage": str(parity_source.get("vintage", "")),
        "period": str(parity_source.get("period", "")),
        "note": (
            "Column surface derived from the populated input layers of the "
            "pinned, sha-verified reference eCPS recorded in "
            "ecps_parity_reference.json (the launch parity contract, "
            "PolicyEngine/populace#313) — the single allow-listed record of the "
            "incumbent coordinates. This manifest names no data package."
        ),
    }

    return {
        "schema_version": 1,
        "issue": "PolicyEngine/populace#368",
        "description": (
            "Declared full-coverage contract for a US release: every input "
            "column the reference eCPS exports must be persisted as a key with "
            "non-default signal, or carry a reviewed exclusion. Enforced as a "
            "hard release gate (populace.build.us_runtime.release_input_"
            "coverage) that generalizes assert_required_us_release_source_"
            "columns from 5 columns to the full eCPS input surface."
        ),
        "reference": reference,
        "derivation": (
            "Required surface = input columns in the pinned, sha-verified "
            "ecps_parity_reference.json populated layers, plus the documented "
            "documented post-reference fsla_overtime_premium and "
            "qualified_passenger_vehicle_loan_interest inputs required by "
            "the shipped OBBBA probes. "
            "status='reviewed_exclusion' for ecps_parity_known_gaps.json entries "
            "(reason+issue from that register); EXCEPT the SSI countable-resource "
            "asset inputs (bank_account_assets, stock_assets, bond_assets), which "
            "are status='required' with NO exclusion per PolicyEngine/populace#368 "
            "so the gate fails on today's artifacts and asset restoration "
            "(Deliverable 2) turns it green. All other populated layers are "
            "'required'. Regenerate with "
            "tools/build_us_release_input_coverage_manifest.py."
        ),
        "counts": {
            "required": len(required),
            "reviewed_exclusion": len(reviewed),
            "total": len(columns),
        },
        "ssi_countable_resource_assets": list(SSI_COUNTABLE_RESOURCE_ASSETS),
        "columns": columns,
        "reform_coverage_probes": REFORM_COVERAGE_PROBES,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {MANIFEST_PATH} — {manifest['counts']['required']} required, "
        f"{manifest['counts']['reviewed_exclusion']} reviewed exclusions, "
        f"{len(manifest['reform_coverage_probes'])} reform probe(s)."
    )


if __name__ == "__main__":
    main()
