"""Build the UK firm table locally (microcosm#1706).

Runs ``microcosm.build.uk_runtime.firm_build.build_uk_firm_dataset`` on the
processed ONS/HMRC tables in ``--processed-dir`` and the ONS attribute tables
in ``--attributes-dir``, and writes ``uk_firms.parquet`` and
``uk_firms.receipt.json`` to ``--out-dir``. Nothing is published or uploaded.

``--rules`` picks the VAT rules and has no default:

- ``policyengine-uk`` runs policyengine-uk's firm entity
  (``PolicyEngineUKFirmRules``), which needs the UK engine extra;
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
    )
    receipt = built["receipt"]
    print(
        f"Wrote {receipt['output']['rows']} firm rows to {built['table_path']}"
        f"\nWrote the receipt to {built['receipt_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
