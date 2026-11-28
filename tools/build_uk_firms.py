"""Build the UK firm table locally (microcosm#1705).

Runs ``microcosm.build.uk_runtime.firm_build.build_uk_firm_dataset`` on the
processed ONS/HMRC tables in ``--processed-dir`` and the ONS attribute tables
in ``--attributes-dir`` (and, with ``--paybill-dir``, the ONS ASHE section
pay table that sets ``firm_paybill``), and writes ``uk_firms.parquet`` and
``uk_firms.receipt.json`` to ``--out-dir``. Nothing is published or uploaded.

``--rules`` picks the VAT rules and has no default:

- ``policyengine-uk`` runs policyengine-uk's firm entity
  (``PolicyEngineUKFirmRules``), which needs a policyengine-uk that includes
  PolicyEngine/policyengine-uk#2940 (the ``uk`` extra's release does not);
- ``axiom`` runs a compiled Axiom UK VAT RuleSpec artifact, named by
  ``--axiom-artifact``, through ``axiom-rules-engine``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microcosm.build.uk_runtime.firm_build import build_uk_firm_dataset
from microcosm.build.uk_runtime.firm_generation import UKFirmGenerationConfig
from microcosm.build.uk_runtime.firm_rules import UKFirmRules

RULES_CHOICES = ("policyengine-uk", "axiom")
DEFAULT_DATA_VINTAGE = "2024-25"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--attributes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--paybill-dir",
        type=Path,
        help=(
            "directory holding the ONS ASHE section pay CSV; sets firm_paybill "
            "(tax year 2024-25 only, so --vintage 2024-25; omit to leave it unset)"
        ),
    )
    parser.add_argument(
        "--employment-allowance-flags",
        action="store_true",
        help=(
            "set the assumed Employment Allowance flags "
            "(firm_claims_employment_allowance, firm_is_single_director_company) "
            "from ONS legal status and employment"
        ),
    )
    parser.add_argument(
        "--public-sector-dir",
        type=Path,
        help=(
            "directory holding the ONS public sector employment CSV; scales "
            "central government and local authority headcounts to it"
        ),
    )
    parser.add_argument(
        "--profit-dir",
        type=Path,
        help=(
            "directory holding the HMRC Corporation Tax liability bands; sets "
            "the assumed firm_taxable_trading_profit"
        ),
    )
    parser.add_argument(
        "--rateable-value-dir",
        type=Path,
        help=(
            "directory holding the VOA rateable value bands; sets the assumed "
            "firm_rateable_value"
        ),
    )
    parser.add_argument(
        "--pay-distribution-dir",
        type=Path,
        help=(
            "directory holding the ONS ASHE section pay percentiles; sets the "
            "assumed count and pay of employees at or below the secondary "
            "threshold (needs --paybill-dir)"
        ),
    )
    parser.add_argument(
        "--bpe-dir",
        type=Path,
        help=(
            "directory holding the DBT Business Population Estimates 2025 "
            "tables; redraws headcount inside the ONS 0-4 band by legal status "
            "and sets firm_employees to employment less assumed working "
            "proprietors, with the labour link columns (needs --paybill-dir)"
        ),
    )
    parser.add_argument(
        "--vat-composition-dir",
        type=Path,
        help=(
            "directory holding the VAT composition fixture; adds Microcosm-only "
            "SIC section shares of standard-, reduced-, zero-rated and exempt "
            "supplies (columns only; needs --consumption-dir)"
        ),
    )
    parser.add_argument(
        "--paybill-anchor-dir",
        type=Path,
        help=(
            "directory holding the ONS D.11 and PAYE RTI paybill anchors; "
            "reports paybill over each in the receipt (needs --paybill-dir)"
        ),
    )
    parser.add_argument(
        "--consumption-dir",
        type=Path,
        help=(
            "directory holding the ONS Supply and Use Tables 2024 extract; "
            "splits turnover into household, intermediate and other final "
            "sales by SIC division"
        ),
    )
    parser.add_argument(
        "--bpe-mid-headcount",
        action="store_true",
        help=(
            "remap headcount inside the ONS 5-9 to 100-249 bands to BPE 2025 "
            "Table 1 shapes (needs --bpe-dir)"
        ),
    )
    parser.add_argument(
        "--bpe-turnover",
        action="store_true",
        help=(
            "draw the open ONS turnover band on a tail pinned to BPE 2025 total "
            "turnover and re-pair employment with turnover (needs --bpe-dir)"
        ),
    )
    parser.add_argument(
        "--bpe-large-headcount",
        action="store_true",
        help="remap 250+ headcount to BPE 2025 Table 1 large rows (needs --bpe-dir)",
    )
    parser.add_argument(
        "--rd-tax-credits-dir",
        type=Path,
        help=(
            "directory holding HMRC R&D tax credit statistics 2024-25; sets the "
            "assumed firm_qualifying_rd_expenditure and firm_is_sme"
        ),
    )
    parser.add_argument(
        "--out-of-frame-no-employees",
        action="store_true",
        help=(
            "give rows outside the ONS frame no headcount, an assumption "
            "(needs --bpe-dir)"
        ),
    )
    parser.add_argument(
        "--employment-allowance-takeup-dir",
        type=Path,
        help=(
            "directory holding HMRC Employment Allowance take-up 2024-25; "
            "draws claims at HMRC's take-up by employee size band (needs "
            "--bpe-dir and --employment-allowance-flags)"
        ),
    )
    parser.add_argument("--rules", choices=RULES_CHOICES, required=True)
    parser.add_argument(
        "--axiom-artifact",
        type=Path,
        help="compiled Axiom UK VAT RuleSpec artifact (with --rules axiom)",
    )
    parser.add_argument("--data-vintage", default=DEFAULT_DATA_VINTAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="thin the candidate draw as UKFirmGenerationConfig.fast does",
    )
    parser.add_argument("--n-iterations", type=int)
    args = parser.parse_args(argv)
    if args.rules == "axiom" and args.axiom_artifact is None:
        parser.error("--rules axiom needs --axiom-artifact")
    if args.rules != "axiom" and args.axiom_artifact is not None:
        parser.error("--axiom-artifact applies only with --rules axiom")
    return args


def _rules(args: argparse.Namespace) -> UKFirmRules:
    if args.rules == "axiom":
        from microcosm.build.uk_runtime.firm_generation import AxiomVATRuleEvaluator
        from microcosm.build.uk_runtime.firm_rules import LegacyVATEvaluatorRules

        return LegacyVATEvaluatorRules(
            AxiomVATRuleEvaluator(artifact_path=args.axiom_artifact)
        )
    from microcosm.build.uk_runtime.firm_policyengine_uk_rules import (
        PolicyEngineUKFirmRules,
    )

    return PolicyEngineUKFirmRules()


def _config(args: argparse.Namespace, rules: UKFirmRules) -> UKFirmGenerationConfig:
    overrides: dict[str, object] = {"seed": args.seed, "firm_rules": rules}
    if args.n_iterations is not None:
        overrides["n_iterations"] = args.n_iterations
    if args.fast:
        config = UKFirmGenerationConfig.fast(**overrides)
    else:
        config = UKFirmGenerationConfig(**overrides)
    return config.with_vintage(args.data_vintage)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rules = _rules(args)
    built = build_uk_firm_dataset(
        processed_dir=args.processed_dir,
        attributes_dir=args.attributes_dir,
        output_dir=args.out_dir,
        rules=rules,
        config=_config(args, rules),
        paybill_dir=args.paybill_dir,
        employment_allowance=args.employment_allowance_flags,
        public_sector_dir=args.public_sector_dir,
        profit_dir=args.profit_dir,
        rateable_value_dir=args.rateable_value_dir,
        pay_distribution_dir=args.pay_distribution_dir,
        bpe_dir=args.bpe_dir,
        employment_allowance_takeup_dir=args.employment_allowance_takeup_dir,
        consumption_dir=args.consumption_dir,
        vat_composition_dir=args.vat_composition_dir,
        paybill_anchor_dir=args.paybill_anchor_dir,
        bpe_mid_headcount=args.bpe_mid_headcount,
        out_of_frame_no_employees=args.out_of_frame_no_employees,
        bpe_turnover=args.bpe_turnover,
        bpe_large_headcount=args.bpe_large_headcount,
        rd_tax_credits_dir=args.rd_tax_credits_dir,
    )
    receipt = built["receipt"]
    print(
        f"Wrote {receipt['output']['rows']} firm rows to {built['table_path']}"
        f"\nWrote the receipt to {built['receipt_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
