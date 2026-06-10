"""microframe: the micro-stack kernel.

One datatype — the :class:`WeightedBundle` — holding entity tables with
explicit linkage, typed weights with conservation invariants, and per-person
strata; weighted accounting computed on it; the US unit-structure operator
that assembles it; and the :class:`RulesEngine` protocol rules engines adapt
to.
"""

from microframe.accounting import gini, groupby_wsum, wmean, wmedian, wquantile, wsum
from microframe.bundle import DEFAULT_STRATUM, WeightedBundle
from microframe.rules import ExportContract, RulesEngine
from microframe.schema import EntitySchema, VariableMetadata
from microframe.units import (
    MICROUNIT_REQUIRED_COLUMNS,
    TAX_UNIT_FILING_STATUS_COLUMN,
    US_GROUP_ENTITIES,
    US_SCHEMA,
    assign_us_unit_structure,
)
from microframe.weights import WeightKind, Weights, assert_kind_transition

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_STRATUM",
    "MICROUNIT_REQUIRED_COLUMNS",
    "TAX_UNIT_FILING_STATUS_COLUMN",
    "US_GROUP_ENTITIES",
    "US_SCHEMA",
    "EntitySchema",
    "ExportContract",
    "RulesEngine",
    "VariableMetadata",
    "WeightKind",
    "WeightedBundle",
    "Weights",
    "assert_kind_transition",
    "assign_us_unit_structure",
    "gini",
    "groupby_wsum",
    "wmean",
    "wmedian",
    "wquantile",
    "wsum",
]
