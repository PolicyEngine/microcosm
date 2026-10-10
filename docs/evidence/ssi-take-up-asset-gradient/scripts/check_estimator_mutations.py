"""Run registered pytest behaviors against deliberate in-memory source mutants.

Mutants alter production-tool source in memory; the working tree stays intact.
Run from the repository root with its complete Python test environment.  Each
case must produce pytest exit code 1 from a test-body failure; collection/setup
errors and successful runs do not qualify as killed mutants.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path.cwd()
TOOL = ROOT / "tools/estimate_ssi_asset_gradient.py"
TESTS = (
    ROOT
    / "packages/microcosm-build/tests/engine_free/us/test_us_ssi_asset_estimation.py"
)
source = TOOL.read_text()


class InjectMutant:
    """Temporarily replace the registered test module's production tool."""

    def __init__(self, mutant: ModuleType):
        self.mutant = mutant
        self.body_reached = False

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        namespace = item.obj.__globals__
        original = namespace["estimator"]
        namespace["estimator"] = self.mutant
        self.body_reached = True
        try:
            yield
        finally:
            namespace["estimator"] = original


mutations = [
    (
        "treat_complete_separation_as_regular_fit",
        "if negative.max() <= positive.min() or positive.max() <= negative.min():",
        "if False:",
        "test_complete_and_quasi_separation_do_not_report_finite_estimates",
        (False,),
    ),
    (
        "treat_quasi_separation_as_regular_fit",
        "if negative.max() <= positive.min() or positive.max() <= negative.min():",
        "if False:",
        "test_complete_and_quasi_separation_do_not_report_finite_estimates",
        (True,),
    ),
    (
        "let_eligibility_cliff_enter_primary_slope",
        'band_frame["liquid_assets"].le(2000)',
        'band_frame["liquid_assets"].le(1000000)',
        "test_primary_fit_excludes_resource_ineligible_range_and_reports_it_separately",
        (),
    ),
    (
        "derived_mapping_omits_bonds",
        'tuple(DONOR_READ["targets"].values())',
        'tuple(DONOR_READ["targets"].values())[:-1]',
        "test_source_mappings_and_pin_match_manifest",
        (),
    ),
    (
        "omit_bonds",
        "frame[list(ASSET_COLUMNS)].sum(axis=1, min_count=3)",
        'frame[["TVAL_BANK", "TVAL_STMF"]].sum(axis=1, min_count=2)',
        "test_sample_uses_all_three_own_assets",
        (),
    ),
    (
        "count_ssi_as_eligibility_income",
        '(frame["TPTOTINC"] - frame["TSSI_AMT"].fillna(0))',
        'frame["TPTOTINC"]',
        "test_receipt_does_not_create_disability_and_ssi_does_not_disqualify_income",
        (),
    ),
    (
        "fill_unmeasured_assets_with_zero",
        "sum(axis=1, min_count=3)",
        "sum(axis=1, min_count=0)",
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("TVAL_BANK", np.nan),
    ),
    (
        "accept_imputed_assets",
        "observed &= frame[column].isin(OBSERVED_STATUSES)",
        "observed &= True",
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("AJSSAVVAL", 2),
    ),
    (
        "accept_imputed_receipt",
        'frame["ASSI_MNYN"].isin(OBSERVED_STATUSES)',
        "True",
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("ASSI_MNYN", 2),
    ),
    (
        "ignore_marriage",
        "& unmarried\n",
        "& True\n",
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("EMS", 1),
    ),
    (
        "accept_zero_weight",
        'frame["weight"].gt(0)',
        'frame["weight"].ge(0)',
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("WPFINWGT", 0),
    ),
    (
        "accept_unmeasured_child_universe",
        'frame["TAGE"].ge(15)',
        'frame["TAGE"].ge(0)',
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("TAGE", 14),
    ),
    (
        "ignore_income_eligibility",
        "countable_income.le(841.0)",
        "countable_income.le(1000000.0)",
        "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records",
        ("TPTOTINC", 2000),
    ),
    (
        "allow_duplicate_source_people",
        'if frame[["SSUID", "PNUM"]].duplicated().any():',
        "if False:",
        "test_duplicate_source_people_refuse",
        (),
    ),
    (
        "replace_log1p_with_levels",
        "np.log1p(assets)",
        "assets",
        "test_weighted_logit_matches_analytic_odds_and_is_order_and_weight_scale_stable",
        (),
    ),
    (
        "treat_clones_as_new_households",
        'frame["household"].astype(str), return_inverse=True',
        "np.arange(len(frame)).astype(str), return_inverse=True",
        "test_standard_errors_cluster_households",
        (),
    ),
    (
        "fit_single_class",
        "if len(frame) < 3 or frame[outcome].nunique() != 2:",
        "if len(frame) < 3:",
        "test_single_class_or_no_asset_variation_is_not_estimable",
        (),
    ),
    (
        "use_unidentified_child_slope",
        'source_band = "18_64" if band == "under_18" else band',
        "source_band = band",
        "test_packaged_summary_transfers_adult_child_slope_without_reform_tuning",
        (),
    ),
]

report = []
for name, original, replacement, test_name, arguments in mutations:
    if original not in source:
        raise ValueError(f"Mutation source fragment absent: {name}")
    mutant = ModuleType(f"mutant_{name}")
    mutant.__dict__["__file__"] = str(TOOL)
    exec(
        compile(source.replace(original, replacement), str(TOOL), "exec"),
        mutant.__dict__,
    )
    selector = f"{TESTS}::{test_name}"
    if (
        test_name
        == "test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records"
    ):
        selector += f"[{arguments[0]}-{arguments[1]}]"
    elif (
        test_name == "test_complete_and_quasi_separation_do_not_report_finite_estimates"
    ):
        selector += f"[{arguments[0]}]"
    plugin = InjectMutant(mutant)
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = pytest.main(
            [selector, "-q", "--tb=short", "--disable-warnings"], plugins=[plugin]
        )
    killed = code == pytest.ExitCode.TESTS_FAILED and plugin.body_reached
    report.append(
        {
            "mutation": name,
            "test": selector.removeprefix(str(ROOT) + "/"),
            "killed": bool(killed),
            "pytest_exit_code": int(code),
            "test_body_reached": plugin.body_reached,
            "pytest_tail": "\n".join(output.getvalue().strip().splitlines()[-12:]),
            "production_fragment": original,
            "replacement": replacement,
        }
    )

output = ROOT / "docs/evidence/ssi-take-up-asset-gradient/estimator_mutations.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {"mutations": len(report), "killed": sum(row["killed"] for row in report)}
    )
)
if not all(row["killed"] for row in report):
    raise SystemExit(1)
