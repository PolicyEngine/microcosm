"""microcosm.frame: the kernel of the microcosm stack.

One datatype — the :class:`Frame`, a weighted sampling frame of entity
tables — holding explicit linkage, typed weights with conservation
invariants, and per-person strata; weighted accounting computed on it; the
US unit-structure operator that assembles it; and the :class:`RulesEngine`
protocol rules engines adapt to.
"""

from microcosm.frame.accounting import (
    gini,
    groupby_wsum,
    wmean,
    wmedian,
    wquantile,
    wsum,
)
from microcosm.frame.bundle import CONSERVE_MASS, DEFAULT_STRATUM, Frame
from microcosm.frame.materialize import (
    PyTablesBooleanMaterialization,
    engine_tables,
    materialize_nullable_booleans_for_pytables,
    nullable_boolean_values_and_mask,
    put_frame_table,
    read_frame_table,
)
from microcosm.frame.rules import ExportContract, RulesEngine
from microcosm.frame.schema import EntitySchema, LinkSpec, VariableMetadata
from microcosm.frame.units import (
    MICROUNIT_REQUIRED_COLUMNS,
    TAX_UNIT_FILING_STATUS_COLUMN,
    US_GROUP_ENTITIES,
    US_SCHEMA,
    assign_us_unit_structure,
)
from microcosm.frame.weights import (
    MassChange,
    MassChangeRecord,
    WeightKind,
    Weights,
    assert_kind_transition,
)

__version__ = "0.1.0"

__all__ = [
    "CONSERVE_MASS",
    "DEFAULT_STRATUM",
    "MICROUNIT_REQUIRED_COLUMNS",
    "TAX_UNIT_FILING_STATUS_COLUMN",
    "US_GROUP_ENTITIES",
    "US_SCHEMA",
    "EntitySchema",
    "ExportContract",
    "Frame",
    "LinkSpec",
    "MassChange",
    "MassChangeRecord",
    "PyTablesBooleanMaterialization",
    "RulesEngine",
    "VariableMetadata",
    "WeightKind",
    "Weights",
    "assert_kind_transition",
    "assign_us_unit_structure",
    "engine_tables",
    "gini",
    "groupby_wsum",
    "materialize_nullable_booleans_for_pytables",
    "nullable_boolean_values_and_mask",
    "put_frame_table",
    "read_frame_table",
    "wmean",
    "wmedian",
    "wquantile",
    "wsum",
]
