"""Reuse a household-batched engine pass for households whose inputs are unchanged.

A pass computes one value per person by running an engine over batches of
whole households. When each person's value depends only on the rows of their
own household (the IDs and membership of its people and group units, the
order of those rows, and the columns the engine can read for the pass's
variable), a later pass over a changed frame can take the earlier value of
every person whose household's rows are unchanged, and run the engine only on
the other households.

The caller guarantees the conditions that make this hold: the variable reads
no weight and nothing summed over the simulated population, every group unit
belongs to one household, and the input list names every column the engine
can read for the variable. :class:`HouseholdPassRecord` keeps the values, the
person IDs and a 128-bit digest of each household's rows; it keeps no frame.

A reuse also runs the engine on some unchanged households and compares the
results byte for byte with the recorded values. If any differ (an input list
that misses a column the engine reads, say), it runs every household again,
so the values it returns are those of a full pass.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import Frame

__all__ = [
    "HouseholdPassInputs",
    "HouseholdPassRecord",
    "HouseholdPassReuse",
    "PassInputColumn",
    "household_input_digests",
    "record_household_pass",
    "reuse_household_pass",
]


@dataclass(frozen=True)
class PassInputColumn:
    """One column the engine can read for a pass's variable.

    Attributes:
        name: Column (engine variable) name.
        entity: Frame table the engine reads it from.
        default: The value the engine uses when the column is absent.
        absent_reads_default: Whether a stored ``default`` gives the engine
            the same input as no column. False for a column the engine would
            otherwise compute (a stored value replaces the formula) and for a
            default the engine would change when it is stored (uprating, or
            division into months). When False, a column that appears or
            disappears changes every household's digest.
    """

    name: str
    entity: str
    default: object
    absent_reads_default: bool


@dataclass(frozen=True)
class HouseholdPassInputs:
    """Every column one pass's engine variable can read, and where."""

    variable: str
    columns: tuple[PassInputColumn, ...]
    #: Where the list came from, for messages.
    description: str = ""
    household_entity: str = "household"


@dataclass(frozen=True)
class HouseholdPassRecord:
    """One pass's per-person values and the household digests they came from."""

    inputs: HouseholdPassInputs
    household_ids: np.ndarray
    #: ``(households, 2)`` unsigned 64-bit digest halves.
    household_digests: np.ndarray
    person_ids: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class HouseholdPassReuse:
    """The values a later pass returns, and how many households it reran."""

    values: np.ndarray
    reused_households: int
    recomputed_households: int
    checked_households: int
    #: True when the check disagreed and every household was run again.
    check_failed: bool


# Two digest halves. Numbers are digested from their bits, so the halves
# differ by salt; strings are also hashed with a different key per half.
_LANES = ("household_pass_a", "household_pass_b")


def _salt(lane: str, label: str) -> np.uint64:
    digest = hashlib.blake2b(f"{lane}\0{label}".encode(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _mixed(values: np.ndarray) -> np.ndarray:
    """A bijective scramble of unsigned 64-bit values (the splitmix64 finalizer)."""

    mixed = np.array(values, dtype=np.uint64, copy=True)
    with np.errstate(over="ignore"):
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    return mixed


def _canonical(values: np.ndarray | pd.Series) -> np.ndarray:
    """Values widened so equal numbers stored at different widths agree.

    Booleans and signed integers become int64, unsigned integers uint64 and
    floats float64; anything else becomes an object array. Widening is exact,
    so values that differ stay different (and ``-0.0`` keeps its sign bit).
    """

    if isinstance(values, pd.Series):
        numpy_values = values.to_numpy()
        if numpy_values.dtype.kind not in "biuf":
            return values.astype(object).to_numpy()
        values = numpy_values
    values = np.asarray(values)
    kind = values.dtype.kind
    if kind in "bi":
        return values.astype(np.int64)
    if kind == "u":
        return values.astype(np.uint64)
    if kind == "f":
        return values.astype(np.float64)
    return values.astype(object)


def _value_bits(values: np.ndarray, lane: str) -> np.ndarray:
    """One unsigned 64-bit value per element: exact bits for numbers."""

    if values.dtype.kind in "iuf":
        return values.view(np.uint64)
    return pd.util.hash_array(values, hash_key=lane, categorize=True)


def _differs_from_default(values: np.ndarray, default: object) -> np.ndarray:
    """Elements whose stored value is not exactly the engine's default."""

    if values.dtype.kind in "iuf":
        try:
            stored_default = np.asarray([default], dtype=values.dtype)
        except (TypeError, ValueError, OverflowError):
            return np.ones(len(values), dtype=bool)
        if isinstance(default, str) or stored_default[0] != default:
            return np.ones(len(values), dtype=bool)
        return values.view(np.uint64) != stored_default.view(np.uint64)[0]
    equal = pd.Series(values, dtype=object).eq(default)
    return ~equal.fillna(False).to_numpy(dtype=bool)


def _row_household_positions(
    frame: Frame, entity: str, household_entity: str
) -> np.ndarray:
    """Each row's position in the household table, for one entity table."""

    schema = frame.schema
    households = pd.Index(
        frame.table(household_entity)[schema.id_column(household_entity)]
    )
    if entity == household_entity:
        return np.arange(len(households), dtype=np.int64)
    person = frame.table(schema.person_entity)
    person_households = person[schema.membership_column(household_entity)].to_numpy()
    if entity == schema.person_entity:
        positions = households.get_indexer(person_households)
    else:
        membership = pd.DataFrame(
            {
                "unit": person[schema.membership_column(entity)].to_numpy(),
                "household": person_households,
            }
        ).drop_duplicates()
        if membership["unit"].duplicated().any():
            examples = membership.loc[membership["unit"].duplicated(), "unit"]
            raise ValueError(
                f"{entity} units must each belong to one {household_entity}; "
                f"examples: {examples.head(5).tolist()}."
            )
        unit_households = membership.set_index("unit")["household"].reindex(
            frame.table(entity)[schema.id_column(entity)].to_numpy()
        )
        if unit_households.isna().any():
            raise ValueError(f"{entity} units without a member person.")
        positions = households.get_indexer(unit_households.to_numpy())
    if (positions < 0).any():
        raise ValueError(f"{entity} rows reference unknown {household_entity} IDs.")
    return positions.astype(np.int64)


def _structural_columns(frame: Frame, entity: str) -> list[str]:
    """ID, membership and role columns the engine builds its entities from."""

    schema = frame.schema
    table = frame.table(entity)
    columns = [schema.entity_id_column(entity)]
    if entity == schema.person_entity:
        for group in schema.group_entities:
            columns.append(schema.membership_column(group))
            role = f"{schema.person_entity}_{group}_role"
            if role in table:
                columns.append(role)
        if "role" in table:
            columns.append("role")
    return columns


def household_input_digests(
    frame: Frame, inputs: HouseholdPassInputs
) -> tuple[np.ndarray, np.ndarray]:
    """Household IDs and a ``(households, 2)`` digest of each household's rows.

    A household's digest covers, for each of its rows in every entity table:
    the row's ID and membership (and role) columns, its place among its
    household's rows of that table, and each input column. An input whose
    absent column reads as its default contributes only where the stored
    value differs from that default, so writing the default is no change.
    Other inputs contribute on every row whenever the column is present.
    """

    schema = frame.schema
    household_entity = inputs.household_entity
    household_ids = frame.table(household_entity)[
        schema.id_column(household_entity)
    ].to_numpy()
    columns_by_entity: dict[str, list[PassInputColumn]] = {}
    for column in inputs.columns:
        if column.entity not in frame.entities:
            raise ValueError(
                f"Input {column.name!r} belongs to {column.entity!r}, which the "
                f"frame does not have ({list(frame.entities)})."
            )
        columns_by_entity.setdefault(column.entity, []).append(column)

    digests = np.zeros((len(household_ids), len(_LANES)), dtype=np.uint64)
    for entity in frame.entities:
        table = frame.table(entity)
        positions = _row_household_positions(frame, entity, household_entity)
        place = pd.Series(positions).groupby(positions).cumcount().to_numpy()
        structural = {
            f"structure:{name}": _canonical(table[name])
            for name in _structural_columns(frame, entity)
        }
        structural[f"structure:{household_entity}"] = _canonical(
            household_ids[positions]
        )
        structural["structure:place"] = place.astype(np.int64)
        for lane_index, lane in enumerate(_LANES):
            # Unsigned sums wrap around, which is what a digest wants.
            row_sums = np.zeros(len(table), dtype=np.uint64)
            with np.errstate(over="ignore"):
                for label, values in structural.items():
                    row_sums += _mixed(
                        _value_bits(values, lane) ^ _salt(lane, f"{entity}:{label}")
                    )
                for column in columns_by_entity.get(entity, ()):
                    if column.name not in table:
                        continue
                    values = _canonical(table[column.name])
                    if column.absent_reads_default:
                        rows = np.flatnonzero(
                            _differs_from_default(values, column.default)
                        )
                    else:
                        rows = np.arange(len(values))
                    if not len(rows):
                        continue
                    row_sums[rows] += _mixed(
                        _value_bits(values[rows], lane)
                        ^ _salt(lane, f"{entity}:input:{column.name}")
                    )
                row_digests = _mixed(row_sums ^ _salt(lane, f"{entity}:row"))
                lane_digests = np.zeros(len(household_ids), dtype=np.uint64)
                np.add.at(lane_digests, positions, row_digests)
                digests[:, lane_index] += lane_digests
    return household_ids.copy(), digests


def record_household_pass(
    frame: Frame, values: np.ndarray, inputs: HouseholdPassInputs
) -> HouseholdPassRecord:
    """Keep a pass's values with the digests of the households they came from."""

    values = np.asarray(values)
    person = frame.table(frame.schema.person_entity)
    if values.shape != (len(person),):
        raise ValueError(
            f"A {inputs.variable} pass record needs one value per person: "
            f"{len(person)} people, values of shape {values.shape}."
        )
    household_ids, digests = household_input_digests(frame, inputs)
    return HouseholdPassRecord(
        inputs=inputs,
        household_ids=household_ids,
        household_digests=digests,
        person_ids=person[frame.schema.person_id_column].to_numpy().copy(),
        values=values.copy(),
    )


def _run_households(
    frame: Frame,
    person_mask: np.ndarray,
    recompute: Callable[[Frame], np.ndarray],
    *,
    variable: str,
    dtype: np.dtype,
) -> np.ndarray:
    """The engine's values for the masked people, in frame order."""

    schema = frame.schema
    run_frame = frame if person_mask.all() else frame.select(person_mask)
    run_person_ids = run_frame.table(schema.person_entity)[
        schema.person_id_column
    ].to_numpy()
    run_values = np.asarray(recompute(run_frame))
    if run_values.shape != (len(run_person_ids),):
        raise ValueError(
            f"The {variable} pass must return one value per person: "
            f"{len(run_person_ids)} people, values of shape {run_values.shape}."
        )
    if run_values.dtype != dtype:
        raise ValueError(
            f"The {variable} pass returned dtype {run_values.dtype}; the "
            f"record holds {dtype}."
        )
    person_ids = frame.table(schema.person_entity)[schema.person_id_column].to_numpy()
    positions = pd.Index(run_person_ids).get_indexer(person_ids[person_mask])
    if (positions < 0).any():
        raise ValueError(f"The {variable} pass dropped people from its frame.")
    return run_values[positions]


def reuse_household_pass(
    record: HouseholdPassRecord,
    frame: Frame,
    *,
    recompute: Callable[[Frame], np.ndarray],
    checked_household_count: int,
) -> HouseholdPassReuse:
    """Values of ``record``'s pass on ``frame``, rerunning only what changed.

    ``recompute`` runs the pass on a frame (a selection of whole households)
    and returns one value per person of that frame. It runs once, on the
    households whose digest differs from the record or that the record lacks,
    together with the first ``checked_household_count`` unchanged households
    in frame order. If any checked person's value differs from the recorded
    one, ``recompute`` runs once more on every household and those values are
    returned.
    """

    schema = frame.schema
    variable = record.inputs.variable
    dtype = record.values.dtype
    household_ids, digests = household_input_digests(frame, record.inputs)
    stored = pd.Index(record.household_ids).get_indexer(household_ids)
    unchanged = stored >= 0
    unchanged[unchanged] = np.all(
        record.household_digests[stored[unchanged]] == digests[unchanged], axis=1
    )

    person = frame.table(schema.person_entity)
    person_ids = person[schema.person_id_column].to_numpy()
    person_households = pd.Index(household_ids).get_indexer(
        person[schema.membership_column(record.inputs.household_entity)].to_numpy()
    )
    person_reused = unchanged[person_households]
    stored_people = pd.Index(record.person_ids).get_indexer(person_ids)
    if (stored_people[person_reused] < 0).any():
        raise ValueError(
            f"A household whose {variable} digest matches has a person the "
            "record lacks."
        )
    values = np.empty(len(person), dtype=dtype)
    values[person_reused] = record.values[stored_people[person_reused]]

    checked = np.zeros(len(household_ids), dtype=bool)
    checked[np.flatnonzero(unchanged)[: max(int(checked_household_count), 0)]] = True
    run_households = ~unchanged | checked
    if run_households.any():
        run_people = run_households[person_households]
        run_values = np.empty(len(person), dtype=dtype)
        run_values[run_people] = _run_households(
            frame, run_people, recompute, variable=variable, dtype=dtype
        )
        checked_people = run_people & person_reused
        if values[checked_people].tobytes() != run_values[checked_people].tobytes():
            every_person = np.ones(len(person), dtype=bool)
            return HouseholdPassReuse(
                values=_run_households(
                    frame, every_person, recompute, variable=variable, dtype=dtype
                ),
                reused_households=0,
                recomputed_households=len(household_ids),
                checked_households=int(checked.sum()),
                check_failed=True,
            )
        rerun_people = run_people & ~person_reused
        values[rerun_people] = run_values[rerun_people]
    return HouseholdPassReuse(
        values=values,
        reused_households=int(unchanged.sum()),
        recomputed_households=int((~unchanged).sum()),
        checked_households=int(checked.sum()),
        check_failed=False,
    )
