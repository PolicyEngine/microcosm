"""Disjoint household age counts, and the composition that calibrates them.

This is an ordinary source-blind population operator. It declares exactly what
it reads — each person's age and household membership, and the household ids —
and owns one int64 count column per declared age band. It opens no file, reads
no source-channel or spine column, imports no rules engine, calculates nothing
from a model, and never touches weights: these are unweighted head counts, and
the graph executor owns every weight in the build.

The graph contract splits that read surface in two, and the declaration follows
it rather than working around it. ``age`` is an owned cell, so it travels in an
input :class:`~microcosm.graph.Slice`. Entity ids and person-to-group
memberships are structural: the executor puts them in every context table and
no node owns them, so a ``Slice`` cannot name them. They are declared instead
in the node's normative parameters, where they still enter the node key, and
the kernel refuses a context whose household view carries anything beyond the
declared id column.

Every included person is counted exactly once. Group-quarters people are
included because the activated target universe is the whole resident
population, and their household membership is preserved rather than dropped.
The bands are declared, contiguous from zero, disjoint, and closed by a single
open top band, so an age outside every band is impossible by construction and a
person can never fall into two.

Refusals are explicit; nothing is silently dropped. An unknown, non-integer,
negative or non-finite age refuses the node, as does a membership that names no
household row, a duplicated household id, or a household row that no person
belongs to. The last one matches the population contract itself, which requires
each group row to be referenced by at least one person — a memberless household
is a broken population, not a zero-count household.

:func:`national_age_calibration_nodes` composes this operator with the existing
`demographic_calibration_node`, unchanged. It emits no US frame-context
artifact: a context written before reweighting must not be relabelled after it,
so consumers read the executor-owned calibrated population and the diagnostics
edge instead.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
    source_hash,
)

from .demographic_calibration_graph import demographic_calibration_node
from .national_age_activation import NATIONAL_AGE_ACTIVATION, AgeBand

__all__ = [
    "AGE_COUNT_SCHEMA_VERSION",
    "SUPPORTED_AGE_CONVENTION",
    "NationalAgeCountKernel",
    "national_age_count_node",
    "national_age_calibration_nodes",
]

#: Revision of the band-schema parameter's canonical form.
AGE_COUNT_SCHEMA_VERSION = 1

#: The one convention this operator implements: the person's ``age`` column is
#: read verbatim as completed years observed at that person's own source
#: interview. No birth date is consulted, no one is aged forward, and no claim
#: is made that the person was observed in the target period.
SUPPORTED_AGE_CONVENTION = "observed_interview_age_completed_years"
if SUPPORTED_AGE_CONVENTION != NATIONAL_AGE_ACTIVATION.age_convention_id:
    raise ValueError("The age-count operator and target activation must agree on age.")

# Operational input envelope for this development connection, not an assertion
# that every source age within it is observed or valid. Outliers refuse rather
# than silently entering the open top band.
MAX_SUPPORTED_AGE = 120

_AGE_COLUMN = "age"
_PERSON = US_SCHEMA.person_entity
_HOUSEHOLD = "household"
_HOUSEHOLD_ID = US_SCHEMA.entity_id_column(_HOUSEHOLD)
_PERSON_HOUSEHOLD_ID = US_SCHEMA.membership_column(_HOUSEHOLD)

#: Every person column the kernel reads: the owned age cell plus the structural
#: household membership. Declared normatively so the read surface is in the key.
_PERSON_COLUMNS = (_AGE_COLUMN, _PERSON_HOUSEHOLD_ID)
#: Every household column the kernel reads: the structural id, and nothing else.
_HOUSEHOLD_COLUMNS = (_HOUSEHOLD_ID,)

_PARAMS = frozenset(
    {
        "bands",
        "person_columns",
        "household_columns",
        "age_convention",
        "schema_version",
        "maximum_age",
    }
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bands_param(bands: tuple[AgeBand, ...]) -> str:
    return _canonical([[band.column, band.low, band.high] for band in bands])


def _bands_from_param(text: object) -> tuple[tuple[str, int, int | None], ...]:
    """Parse and fully re-validate the declared band schema.

    The kernel trusts nothing it did not check here: the schema must be the
    canonical form of a contiguous, disjoint, zero-based partition closed by
    one open top band, with distinct column names.
    """

    if not isinstance(text, str) or len(text.encode()) > 65_536:
        raise ValueError("A bounded canonical age-band declaration is required.")
    rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError("The age-band declaration must be a nonempty list.")
    bands: list[tuple[str, int, int | None]] = []
    expected_low = 0
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or type(row[1]) is not int
        ):
            raise ValueError("Each age band declares a column, a low and a high.")
        column, low, high = row
        last = index == len(rows) - 1
        if (high is None) is not last:
            raise ValueError("Exactly the final age band is an open interval.")
        if high is not None and (type(high) is not int or high < low):
            raise ValueError("A closed age band needs an integer high at or above low.")
        if low != expected_low:
            raise ValueError("Age bands must be contiguous and start at zero.")
        expected_low = None if high is None else high + 1
        bands.append((column, low, high))
    if len({column for column, _, _ in bands}) != len(bands):
        raise ValueError("Age bands must own distinct count columns.")
    if _canonical([[c, lo, hi] for c, lo, hi in bands]) != text:
        raise ValueError("The age-band declaration must be in canonical form.")
    return tuple(bands)


def _node_from_schema(schema: str, *, population: str, node_id: str) -> Node:
    """The single declaration site, shared by the factory and the kernel."""

    bands = _bands_from_param(schema)
    return Node(
        id=node_id,
        kernel=NationalAgeCountKernel.ref,
        population=population,
        inputs=(Slice(_PERSON, (_AGE_COLUMN,)),),
        outputs=tuple(Owned(_HOUSEHOLD, column, "int64") for column, _, _ in bands),
        params={
            "bands": schema,
            "person_columns": _PERSON_COLUMNS,
            "household_columns": _HOUSEHOLD_COLUMNS,
            "age_convention": SUPPORTED_AGE_CONVENTION,
            "schema_version": AGE_COUNT_SCHEMA_VERSION,
            "maximum_age": MAX_SUPPORTED_AGE,
        },
    )


def national_age_count_node(
    *,
    population: str,
    bands: tuple[AgeBand, ...] = NATIONAL_AGE_ACTIVATION.bands,
    node_id: str = "national.age_counts",
) -> Node:
    """Declare the household age-count operator over one population version.

    Args:
        population: The population version whose rows the counts live in.
        bands: The activated bands, in order. Their canonical form enters the
            node key, so an altered schema is a different node.
        node_id: The node's id.

    Returns:
        The node owning one int64 count column per band.
    """

    return _node_from_schema(
        _bands_param(bands), population=population, node_id=node_id
    )


class NationalAgeCountKernel(KernelBase):
    """Count each person once into their household's declared age band."""

    ref = "us.national_age_counts@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.NONE,
        consumes_se=False,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            sys.modules[__name__], dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        params = context.params
        if set(params) != _PARAMS:
            raise ValueError("The age-count node declares exactly its band schema.")
        if params["age_convention"] != SUPPORTED_AGE_CONVENTION:
            raise ValueError(
                "This operator implements only "
                f"{SUPPORTED_AGE_CONVENTION!r}; it cannot honour "
                f"{params['age_convention']!r}."
            )
        if params["schema_version"] != AGE_COUNT_SCHEMA_VERSION:
            raise ValueError("Unsupported age-band schema version.")
        if (
            params["person_columns"] != _PERSON_COLUMNS
            or params["household_columns"] != _HOUSEHOLD_COLUMNS
        ):
            raise ValueError(
                "This operator reads exactly "
                f"{_PERSON_COLUMNS} on {_PERSON!r} and {_HOUSEHOLD_COLUMNS} on "
                f"{_HOUSEHOLD!r}."
            )
        bands = _bands_from_param(params["bands"])
        expected = _node_from_schema(
            params["bands"],
            population=context.node.population or "",
            node_id=context.node.id,
        )
        if context.node.normative() != expected.normative():
            raise ValueError("Age-count node differs from its complete declaration.")

        household = context.tables[_HOUSEHOLD]
        person = context.tables[_PERSON]
        # The household view is exactly its id column: no weight, no source
        # channel, no other attribute is visible to this operator.
        if tuple(household.columns) != _HOUSEHOLD_COLUMNS:
            raise ValueError(
                f"The household view must carry exactly {_HOUSEHOLD_COLUMNS}, "
                f"got {tuple(household.columns)}."
            )
        missing = [column for column in _PERSON_COLUMNS if column not in person]
        if missing:
            raise ValueError(f"The person view is missing {missing}.")
        household_ids = _ids(household, _HOUSEHOLD_ID, what="household id")
        index = pd.Index(household_ids, name=_HOUSEHOLD_ID)
        if index.has_duplicates:
            raise ValueError("Household ids must be unique.")
        # Align by id, never by row order: neither axis is assumed sequential,
        # sorted, or shared between the two tables.
        positions = index.get_indexer(
            _ids(person, _PERSON_HOUSEHOLD_ID, what="household membership")
        )
        if np.any(positions < 0):
            missing = np.unique(person[_PERSON_HOUSEHOLD_ID].to_numpy()[positions < 0])[
                :5
            ]
            raise ValueError(
                "Every person's household membership must resolve to a household "
                f"row; unresolved ids include {missing.tolist()}."
            )

        age = _ages(person)
        band_of_person = np.full(len(person), -1, dtype=np.int64)
        for ordinal, (_, low, high) in enumerate(bands):
            inside = age >= low if high is None else (age >= low) & (age <= high)
            if np.any(band_of_person[inside] >= 0):
                raise ValueError("Declared age bands overlap.")
            band_of_person[inside] = ordinal
        if np.any(band_of_person < 0):
            raise ValueError("Declared age bands do not cover every observed age.")

        counts = np.bincount(
            band_of_person * len(index) + positions,
            minlength=len(bands) * len(index),
        ).astype(np.int64, copy=False)
        counts = counts.reshape(len(bands), len(index))
        if int(counts.sum()) != len(person):
            raise ValueError("Every person must be counted into exactly one band.")
        people = counts.sum(axis=0)
        if np.any(people == 0):
            # The population contract requires every group row to be referenced
            # by a person, so this is a broken population, not an empty house.
            raise ValueError(
                "Every household row must contain at least one person; "
                f"{int((people == 0).sum())} household(s) contain none."
            )
        return KernelResult(
            columns={
                (_HOUSEHOLD, column): pd.Series(
                    counts[ordinal], index=index, dtype="int64"
                )
                for ordinal, (column, _, _) in enumerate(bands)
            },
            receipt={
                "scope": "national_age_counts_development",
                "age_convention": SUPPORTED_AGE_CONVENTION,
                "bands": params["bands"],
                "people_counted": int(counts.sum()),
                "households": int(len(index)),
                "maximum_observed_age": int(age.max()) if len(age) else None,
                "minimum_observed_age": int(age.min()) if len(age) else None,
                "maximum_supported_age": MAX_SUPPORTED_AGE,
                "consumes_weights": False,
                "universe": "every person on the population, group quarters included",
                "release_eligible": False,
            },
        )


def _ids(table: pd.DataFrame, column: str, *, what: str) -> np.ndarray:
    series = table[column]
    if (
        is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
        or not is_numeric_dtype(series.dtype)
    ):
        raise ValueError(f"The {what} column must be an integer column.")
    if series.isna().any():
        raise ValueError(f"The {what} column contains missing values.")
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(values) & (values == np.floor(values))):
        raise ValueError(f"The {what} column must hold whole numbers.")
    # Preserve exact integer ids without a float round trip, and refuse casts
    # that would wrap or saturate into another household's identity. The open
    # float upper bound matters: float64 cannot represent int64's maximum.
    if is_integer_dtype(series.dtype):
        limits = np.iinfo(np.int64)
        outside = ((series < limits.min) | (series > limits.max)).any()
    else:
        outside = np.any(values < -(2**63)) or np.any(values >= 2**63)
    if outside:
        raise ValueError(f"The {what} column must fit signed int64 exactly.")
    return series.to_numpy(dtype="int64")


def _ages(person: pd.DataFrame) -> np.ndarray:
    """Ages as nonnegative whole years; anything else refuses the node."""

    series = person[_AGE_COLUMN]
    if (
        is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
        or not is_numeric_dtype(series.dtype)
    ):
        raise ValueError("Person age must be a real numeric column.")
    if series.isna().any():
        raise ValueError(
            f"{int(series.isna().sum())} person age(s) are unknown; an age-count "
            "node refuses rather than dropping people."
        )
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(values)):
        raise ValueError("Person age must be finite.")
    if np.any(values < 0):
        raise ValueError("Person age must not be negative.")
    if np.any(values > MAX_SUPPORTED_AGE):
        raise ValueError(
            f"Person age exceeds the declared maximum of {MAX_SUPPORTED_AGE}."
        )
    if not np.all(values == np.floor(values)):
        raise ValueError("Person age must be a whole number of completed years.")
    return values


def national_age_calibration_nodes(
    *,
    base: str,
    registry,
    epochs: int,
    learning_rate: float,
    max_weight_ratio: float,
    max_initial_weight_ratio: float,
    bands: tuple[AgeBand, ...] = NATIONAL_AGE_ACTIVATION.bands,
    count_node_id: str = "national.age_counts",
    calibration_node_id: str = "national.demographic_calibration",
) -> tuple[Node, Node]:
    """Compose the age-count operator with the existing calibration node.

    The count node lives in ``base``'s population version and owns the columns;
    the unchanged `demographic_calibration_node` reweights over ``base`` and
    reads exactly those columns, so the count node is its predecessor.

    Args:
        base: The population version carrying household importance weights.
        registry: The activated :class:`TargetRegistry`. Its measures must be
            exactly the declared band columns, in order — a registry and a band
            schema that disagree are refused here rather than half-wired.
        epochs: Solver epochs.
        learning_rate: Solver learning rate.
        max_weight_ratio: The executor's cap against original design weights.
        max_initial_weight_ratio: The solver's cap against incoming weights.
        bands: The activated bands, in order.
        count_node_id: Id of the count node.
        calibration_node_id: Id of the calibration node.

    Returns:
        ``(count_node, calibration_node)`` in dependency order.

    Raises:
        ValueError: If the registry's measures are not the band columns.
    """

    measures = tuple(spec.measure for spec in registry)
    columns = tuple(band.column for band in bands)
    if measures != columns:
        raise ValueError(
            "The registry's measures must be exactly the declared band columns "
            f"in order; got {measures} against {columns}."
        )
    return (
        national_age_count_node(population=base, bands=bands, node_id=count_node_id),
        demographic_calibration_node(
            registry,
            base=base,
            node_id=calibration_node_id,
            epochs=epochs,
            learning_rate=learning_rate,
            max_weight_ratio=max_weight_ratio,
            max_initial_weight_ratio=max_initial_weight_ratio,
        ),
    )
