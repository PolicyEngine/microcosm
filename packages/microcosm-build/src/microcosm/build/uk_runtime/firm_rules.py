"""The rules seam between the UK firm generator and a VAT rule engine.

The experimental UK firm generator (``firm_generation``) calibrates candidate
firms to HMRC net VAT liability. It does not compute VAT itself: it hands a
pre-calibration candidate table to a ``UKFirmRules`` implementation and reads
back each firm's VAT liability if it were registered. Target rows then weight
that liability by each firm's registration propensity.

The candidate table uses policyengine-uk firm variable names in pounds
(``firm_table.build_candidate_rules_table``): all turnover standard-rated, no
VAT scope and no registration. The rules return
``vat_liability_if_registered`` in pounds, keyed by ``firm_id``;
``align_rule_outputs`` puts it back in the generator's row order.

Today the only implementation in source is ``LegacyVATEvaluatorRules``, which
wraps an existing ``UKFirmVATRuleEvaluator`` such as the Axiom RuleSpec
evaluator. A future ``PolicyEngineUKFirmRules`` adapter will replace it: it
will run policyengine-uk's firm entity on the candidate table, read
``firm_vat_liability_if_registered`` (the liability ``firm_vat`` charges a
registered firm) and reconcile ``firm_vat_registered`` with the generator's
registration flags. Axiom then stays an optional cross-check. No reference VAT
evaluator lives in source; tests bring their own doubles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from microcosm.build.uk_runtime.firm_generation import UKFirmVATRuleEvaluator

#: The policyengine-uk period (a year label) each firm data vintage (a tax
#: year) is evaluated in: the tax year's start year. policyengine-uk annualises
#: its ``gov`` parameters so that period Y takes the value in force on 30 April
#: Y (``convert_to_fiscal_year_parameters``), which is the Y-to-Y+1 tax year's
#: value. Period 2024 therefore reads the £90,000 VAT registration threshold
#: that took effect on 2024-04-01, and period 2023 the £85,000 one.
UK_FIRM_RULE_PERIOD_BY_VINTAGE: Mapping[str, int] = MappingProxyType(
    {
        "2023-24": 2023,
        "2024-25": 2024,
    }
)

#: The inverse view, for adapters that need the vintage back from a period.
UK_FIRM_VINTAGE_BY_RULE_PERIOD: Mapping[int, str] = MappingProxyType(
    {period: vintage for vintage, period in UK_FIRM_RULE_PERIOD_BY_VINTAGE.items()}
)


def rule_period_for_vintage(data_vintage: str) -> int:
    """The policyengine-uk period a firm data vintage is evaluated in.

    ``"2024-25"`` gives 2024 and ``"2023-24"`` gives 2023: see
    ``UK_FIRM_RULE_PERIOD_BY_VINTAGE`` for why the period is the tax year's
    start year. Any other vintage raises ``ValueError``.
    """

    try:
        return UK_FIRM_RULE_PERIOD_BY_VINTAGE[data_vintage]
    except KeyError:
        raise ValueError(
            f"No UK firm rule period for vintage {data_vintage!r}; "
            f"choose from {sorted(UK_FIRM_RULE_PERIOD_BY_VINTAGE)}."
        ) from None


@dataclass(frozen=True)
class UKFirmRuleOutputs:
    """VAT rule results for candidate firms, keyed by ``firm_id``.

    ``vat_liability_if_registered`` is in pounds per firm per year; negative
    values are repayments. ``provenance`` names the rules that produced it.
    """

    firm_id: np.ndarray
    vat_liability_if_registered: np.ndarray
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        firm_id = np.asarray(self.firm_id)
        if firm_id.ndim != 1 or not np.issubdtype(firm_id.dtype, np.integer):
            raise ValueError("UK firm rule outputs need a 1-D integer firm_id.")
        liability = np.asarray(self.vat_liability_if_registered, dtype=np.float64)
        if liability.shape != firm_id.shape:
            raise ValueError(
                "UK firm rule outputs have "
                f"{liability.shape[0] if liability.ndim else 0} liabilities for "
                f"{firm_id.shape[0]} firm ids."
            )
        if not np.all(np.isfinite(liability)):
            raise ValueError("UK firm VAT liability must be finite.")
        object.__setattr__(self, "firm_id", firm_id.astype(np.int64))
        object.__setattr__(self, "vat_liability_if_registered", liability)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType({str(k): str(v) for k, v in self.provenance.items()}),
        )


class UKFirmRules(Protocol):
    """Evaluate VAT rules for a pre-calibration UK firm candidate table."""

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        """Return each firm's VAT liability if registered, in pounds."""


def align_rule_outputs(
    outputs: UKFirmRuleOutputs,
    firm_id: np.ndarray | pd.Series,
) -> np.ndarray:
    """``outputs``' liabilities in the order of ``firm_id``, in pounds.

    Raises ``ValueError`` when either side repeats a firm id, when a requested
    firm has no output, or when the outputs hold a firm not requested.
    """

    requested = pd.Index(np.asarray(firm_id, dtype=np.int64))
    produced = pd.Index(outputs.firm_id)
    if not requested.is_unique:
        raise ValueError("Requested UK firm ids repeat.")
    if not produced.is_unique:
        raise ValueError("UK firm rule outputs repeat a firm id.")
    positions = produced.get_indexer(requested)
    missing = requested[positions < 0]
    if len(missing):
        raise ValueError(
            f"UK firm rule outputs miss {len(missing)} firm ids, "
            f"e.g. {missing[:5].tolist()}."
        )
    extra = produced[~produced.isin(requested)]
    if len(extra):
        raise ValueError(
            f"UK firm rule outputs hold {len(extra)} unrequested firm ids, "
            f"e.g. {extra[:5].tolist()}."
        )
    return outputs.vat_liability_if_registered[positions]


#: Candidate-table columns ``LegacyVATEvaluatorRules`` reads.
LEGACY_VAT_EVALUATOR_INPUT_COLUMNS = (
    "firm_id",
    "firm_standard_rated_supplies",
    "firm_input_costs",
)


@dataclass(frozen=True)
class LegacyVATEvaluatorRules:
    """``UKFirmRules`` over an existing ``UKFirmVATRuleEvaluator``.

    The evaluator works in thousands of pounds and by data vintage; this
    adapter takes the candidate table in policyengine-uk names and pounds:

    - ``turnover_k`` is ``firm_standard_rated_supplies / 1000`` (the Axiom
      evaluator reads it as standard-rated taxable supplies);
    - ``input_cost_k`` is ``firm_input_costs / 1000`` (read as standard-rated
      deductible purchases; the evaluator applies the VAT rate itself, so the
      table's ``firm_input_vat`` is not read);
    - ``vat_registered`` is all true, so the result is the liability if
      registered;
    - ``data_vintage`` is the vintage ``period`` stands for
      (``UK_FIRM_VINTAGE_BY_RULE_PERIOD``).

    The evaluator's thousands of pounds come back multiplied by 1000.
    """

    evaluator: UKFirmVATRuleEvaluator
    name: str = "legacy_vat_rule_evaluator"

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        missing = [
            column
            for column in LEGACY_VAT_EVALUATOR_INPUT_COLUMNS
            if column not in firms.columns
        ]
        if missing:
            raise ValueError(f"UK firm candidate table misses columns {missing}.")
        try:
            data_vintage = UK_FIRM_VINTAGE_BY_RULE_PERIOD[int(period)]
        except KeyError:
            raise ValueError(
                f"No UK firm data vintage for rule period {period!r}; "
                f"choose from {sorted(UK_FIRM_VINTAGE_BY_RULE_PERIOD)}."
            ) from None
        supplies_k = firms["firm_standard_rated_supplies"].to_numpy(np.float64) / 1000
        input_k = firms["firm_input_costs"].to_numpy(np.float64) / 1000
        liability_k = np.asarray(
            self.evaluator.net_liability_k(
                turnover_k=supplies_k,
                input_cost_k=input_k,
                vat_registered=np.ones(len(firms), dtype=bool),
                data_vintage=data_vintage,
            ),
            dtype=np.float64,
        )
        if liability_k.shape != supplies_k.shape:
            raise ValueError(
                "VAT rule evaluator returned shape "
                f"{liability_k.shape}, expected {supplies_k.shape}."
            )
        return UKFirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(np.int64),
            vat_liability_if_registered=liability_k * 1000.0,
            provenance={
                "rules": self.name,
                "evaluator": type(self.evaluator).__name__,
                "data_vintage": data_vintage,
                "period": str(int(period)),
                "output": "vat_liability_if_registered",
            },
        )


__all__ = [
    "LEGACY_VAT_EVALUATOR_INPUT_COLUMNS",
    "UK_FIRM_RULE_PERIOD_BY_VINTAGE",
    "UK_FIRM_VINTAGE_BY_RULE_PERIOD",
    "LegacyVATEvaluatorRules",
    "UKFirmRuleOutputs",
    "UKFirmRules",
    "align_rule_outputs",
    "rule_period_for_vintage",
]
