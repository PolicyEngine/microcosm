"""The seam between a firm build and a rule engine.

A firm build does not compute taxes itself. It hands a firm table to a rule
engine (a country model, or a cross-check) and reads back named per-firm
outputs, keyed by firm id. The seam is the same for every country:

- a :class:`FirmRules` implementation's ``evaluate(firms, *, period)``
  returns :class:`FirmRuleOutputs`;
- :func:`align_rule_outputs` puts the outputs back in the build's row order,
  refusing a missing, repeated or unrequested firm id;
- :func:`weighted_totals` sums a per-firm output table by firm weight.

What the table's columns mean, which outputs are read and what ``period``
stands for are the country's business.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "FirmRuleOutputs",
    "FirmRules",
    "align_rule_outputs",
    "weighted_totals",
]


@dataclass(frozen=True)
class FirmRuleOutputs:
    """Named per-firm rule results, keyed by ``firm_id``.

    ``values`` maps each output's name to a float64 array aligned with
    ``firm_id``; every value must be finite. ``provenance`` names the rules
    that produced them.
    """

    firm_id: np.ndarray
    values: Mapping[str, np.ndarray]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        firm_id = np.asarray(self.firm_id)
        if firm_id.ndim != 1 or not np.issubdtype(firm_id.dtype, np.integer):
            raise ValueError("firm rule outputs need a 1-D integer firm_id")
        if not self.values:
            raise ValueError("firm rule outputs need at least one named output")
        values: dict[str, np.ndarray] = {}
        for name, raw in self.values.items():
            name = str(name)
            if not name or name == "firm_id":
                raise ValueError(f"firm rule output name {name!r} is not allowed")
            array = np.asarray(raw, dtype=np.float64)
            if array.shape != firm_id.shape:
                raise ValueError(
                    f"firm rule output {name} has shape {array.shape} for "
                    f"{firm_id.shape[0]} firm ids"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"firm rule output {name} must be finite")
            values[name] = array
        object.__setattr__(self, "firm_id", firm_id.astype(np.int64))
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType({str(k): str(v) for k, v in self.provenance.items()}),
        )


class FirmRules(Protocol):
    """Evaluate rules for a firm table in ``period``."""

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> FirmRuleOutputs:
        """Return named per-firm outputs keyed by ``firm_id``."""


def align_rule_outputs(
    outputs: FirmRuleOutputs,
    firm_id: np.ndarray | pd.Series,
) -> dict[str, np.ndarray]:
    """Every output of ``outputs``, in the order of ``firm_id``.

    Raises ``ValueError`` when either side repeats a firm id, when a requested
    firm has no output, or when the outputs hold a firm not requested.
    """

    requested = pd.Index(np.asarray(firm_id, dtype=np.int64))
    produced = pd.Index(outputs.firm_id)
    if not requested.is_unique:
        raise ValueError("requested firm ids repeat")
    if not produced.is_unique:
        raise ValueError("firm rule outputs repeat a firm id")
    positions = produced.get_indexer(requested)
    missing = requested[positions < 0]
    if len(missing):
        raise ValueError(
            f"firm rule outputs miss {len(missing)} firm ids, "
            f"e.g. {missing[:5].tolist()}"
        )
    extra = produced[~produced.isin(requested)]
    if len(extra):
        raise ValueError(
            f"firm rule outputs hold {len(extra)} unrequested firm ids, "
            f"e.g. {extra[:5].tolist()}"
        )
    return {name: values[positions] for name, values in outputs.values.items()}


def weighted_totals(
    table: pd.DataFrame,
    *,
    weight_column: str = "firm_weight",
    id_column: str = "firm_id",
) -> dict[str, float]:
    """``sum(value * weight)`` for every column but the id and weight columns.

    Each total stands alone; the caller decides which ones add up.
    """

    if weight_column not in table.columns:
        raise ValueError(f"firm output table misses column {weight_column}")
    weight = table[weight_column].to_numpy(np.float64)
    return {
        str(column): float(np.dot(table[column].to_numpy(np.float64), weight))
        for column in table.columns
        if column not in (id_column, weight_column)
    }
