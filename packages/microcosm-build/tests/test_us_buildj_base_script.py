"""Regression contracts for the Build-J reusable-base driver."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILDJ_BASE_SCRIPT = ROOT / "experiments/build_j_recert/buildj_base.sh"


def _cache_required_columns() -> dict[str, tuple[str, ...]]:
    source = BUILDJ_BASE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"required = (?P<required>\{.*?\n\})\ntry:",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "Build-J cache required-column map is missing"
    required = ast.literal_eval(match.group("required"))
    assert isinstance(required, dict)
    return required


def test_cache_rejects_bases_predating_salt_refund_and_energy_subsidy() -> None:
    required = _cache_required_columns()

    assert "salt_refund_income" in required["person"]
    assert "spm_unit_energy_subsidy" in required["spm_unit"]
    assert "takes_up_housing_assistance_if_eligible" in required["spm_unit"]


def test_cache_rejects_bases_predating_measured_medicare_take_up() -> None:
    required = _cache_required_columns()

    assert "investment_interest_expense" in required["person"]
    assert "takes_up_medicare_if_eligible" in required["person"]
    assert "workers_compensation" in required["person"]
    assert "takes_up_wic_if_eligible" in required["person"]
