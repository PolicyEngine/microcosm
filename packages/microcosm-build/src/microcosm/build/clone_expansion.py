"""Clone-major copies of a frame's records, numbered by a country's rule.

A build can turn a frame into ``K`` clones of every record. A later stage may
treat each clone as a record of its own (a clone that takes an area of its
own, say), while together the clones stand for the record once, each at an
equal share of its weight. This module makes those clones the same way for
every country (microcosm#931). The country declares how a clone's ids are
numbered (:class:`CloneIdRule`) and what the clones carry
(:class:`CloneDeclaration`). The module names no country's columns, tables or
constants: it reads every entity, id column and membership column from the
frame's :class:`~microcosm.frame.EntitySchema`.

:func:`expand_clones` returns ``K`` clones of every entity table,
clone-major. Clone 0 is the frame's own rows, ids unchanged. Clone ``k``
repeats them, with each entity's ids minted by the declared rule, each
person's membership the id of its unit's clone ``k``, and every other column
copied unchanged. Every row carries its clone ordinal, and, where the country
declares a column for it, its id before the expansion. Every weighted
entity's weights are split equally, the same float in each of a row's clones,
and the mass log gains one record per weighted entity stating the declared
reason.

Every refusal names entities, columns, dtypes, weight kinds and declared
constants only, never an id, a weight or another value of a row. The declared
rule's own refusals must keep to counts and declared constants as well.

The core copies no link table and gives every record the same number of
clones. Those wait for a caller that needs them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights

__all__ = [
    "CLONE_MASS_RTOL",
    "CloneDeclaration",
    "CloneIdRule",
    "expand_clones",
]

#: How closely the clones' total weight must equal the frame's, relative to
#: the frame's: the tolerance of the Frame's own mass-conserving replacement.
CLONE_MASS_RTOL = 1e-9


@dataclass(frozen=True)
class CloneIdRule:
    """How a country numbers its clones' ids.

    ``bound`` is the most clones the rule can number: clone ordinals run
    ``0 .. bound - 1``, so a frame expands into 1 to ``bound`` clones.
    ``mint(ids, k)`` returns clone ``k``'s ids from the originals', for
    ``1 <= k < bound``. It must return one integer per id, of the ids' own
    dtype, as a function of that id alone, keep the originals' order, and
    put every clone ``k`` id above every clone ``k - 1`` id. A mint in the
    ids' dtype keeps it at every clone count. ``require_below(ids, what)``
    refuses incoming ids the mint has no room for; ``what`` names the ids in
    its message.
    """

    bound: int
    mint: Callable[[np.ndarray, int], np.ndarray]
    require_below: Callable[[np.ndarray, str], None]

    def __post_init__(self) -> None:
        if not _is_count(self.bound) or self.bound < 1:
            raise ValueError("A clone-id rule needs an integer bound of at least 1.")
        if not callable(self.mint) or not callable(self.require_below):
            raise TypeError(
                "A clone-id rule's mint and require_below must be callable."
            )


@dataclass(frozen=True, eq=False)
class CloneDeclaration:
    """What a country's clones carry, declared once and the same at every K.

    - ``ids``: how the clones' ids are numbered.
    - ``clone_index_columns``: each entity's column for its rows' clone
      ordinal, ``0 .. K - 1`` as int64. Every entity of the frame needs one.
    - ``weight_kinds``: the weight kinds the expansion accepts, on every
      entity the frame weights. Each of a row's ``K`` clones takes ``1 / K``
      of its weight, and the clones keep each entity's kind.
    - ``reason``: the reason each mass record states. It is declared here,
      not per call, so it is the same at every clone count.
    - ``source_id_columns``: for any entity, a column for each row's id
      before the expansion (its clone-0 id), the same in all its clones.

    A declaration compares and hashes by identity, as its mappings are
    frozen views that do not hash.
    """

    ids: CloneIdRule
    clone_index_columns: Mapping[str, str]
    weight_kinds: frozenset[WeightKind]
    reason: str
    source_id_columns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ids, CloneIdRule):
            raise TypeError("A clone declaration's ids must be a CloneIdRule.")
        index = _column_names(self.clone_index_columns, "clone-index")
        source = _column_names(self.source_id_columns, "source-id")
        if not index:
            raise ValueError("A clone declaration names a clone-index column.")
        columns = [*index.values(), *source.values()]
        if len(set(columns)) != len(columns):
            raise ValueError("A clone declaration's columns must be distinct.")
        kinds = frozenset(self.weight_kinds)
        if not kinds or not all(isinstance(kind, WeightKind) for kind in kinds):
            raise ValueError("A clone declaration accepts at least one WeightKind.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("A clone declaration states its mass record's reason.")
        object.__setattr__(self, "clone_index_columns", MappingProxyType(index))
        object.__setattr__(self, "source_id_columns", MappingProxyType(source))
        object.__setattr__(self, "weight_kinds", kinds)


def expand_clones(frame: Frame, *, clones: int, declaration: CloneDeclaration) -> Frame:
    """``frame`` as ``clones`` clones of every record, clone-major.

    Each entity table becomes ``clones`` copies of itself, one after another.
    Clone 0's rows are the frame's own, with their ids. Clone ``k``'s rows
    repeat them: each entity's ids are ``declaration.ids.mint(ids, k)``, and a
    person's membership in a unit is that unit's clone ``k`` id. Every other
    column is copied unchanged. Each row gains its clone ordinal in its
    entity's clone-index column, and its id before the expansion in a
    declared source-id column, so a row and its clones share that id.

    Every weighted entity's weights become ``numpy.tile(w / clones,
    clones)``: one float per row of the frame, the same in each of its
    clones, of the entity's weight kind. The strata repeat once per clone
    and the metadata is kept. The mass log gains one record per weighted
    entity, in the frame's entity order: the entity's totals before and
    after, which agree within :data:`CLONE_MASS_RTOL`, a declared factor of
    1.0, and the declared reason. At ``clones == 1`` the rows, ids and
    weights are the frame's own; only the declared columns and the records
    are added.

    Refused, naming entities, columns, dtypes, weight kinds and declared
    constants only:

    - a clone count that is not an integer in ``[1, bound]``;
    - a declaration that names an entity the frame lacks, or gives an entity
      of the frame no clone-index column;
    - a frame that already carries a declared column or a link table;
    - weights, on any entity, of a kind the declaration does not accept;
    - an id or membership column that is not of a numpy integer dtype, or a
      membership column whose dtype is not its unit ids' dtype;
    - incoming ids the rule's ``require_below`` refuses;
    - a mint that does not return one integer per id in the ids' own dtype,
      does not keep the ids' order, or does not put every clone ``k`` id
      above every clone ``k - 1`` id;
    - clones whose total weight strays from the frame's beyond
      :data:`CLONE_MASS_RTOL`.
    """

    rule = declaration.ids
    if not _is_count(clones) or not 1 <= clones <= rule.bound:
        raise ValueError(
            f"A clone count must be an integer in [1, {rule.bound}], the declared "
            "bound."
        )
    count = int(clones)
    _check_frame(frame, declaration)
    schema = frame.schema
    ids = {entity: _clone_ids(frame, entity, count, rule) for entity in frame.entities}
    weights, records = _split_weights(frame, count, declaration.reason)
    ordinals = np.arange(count, dtype=np.int64)
    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        rows = len(table)
        out = table.take(np.tile(np.arange(rows), count)).reset_index(drop=True)
        out[schema.entity_id_column(entity)] = np.concatenate(ids[entity])
        if entity == schema.person_entity:
            for group in schema.group_entities:
                column = schema.membership_column(group)
                out[column] = _member_ids(table[column], ids[group])
        source = declaration.source_id_columns.get(entity)
        if source is not None:
            out[source] = np.tile(ids[entity][0], count)
        out[declaration.clone_index_columns[entity]] = np.repeat(ordinals, rows)
        tables[entity] = out
    return Frame(
        tables,
        schema,
        weights,
        pd.concat([frame.strata] * count, ignore_index=True),
        mass_log=(*frame.mass_log, *records),
        metadata=frame.metadata,
    )


def _is_count(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _column_names(columns: Mapping[str, str], what: str) -> dict[str, str]:
    """A declared entity-to-column mapping, each name a non-empty string."""

    if not isinstance(columns, Mapping):
        raise TypeError(f"A clone declaration's {what} columns must be a mapping.")
    names = dict(columns)
    for entity, column in names.items():
        if not (
            isinstance(entity, str) and entity and isinstance(column, str) and column
        ):
            raise ValueError(
                f"A clone declaration's {what} columns map entity names to "
                "column names."
            )
    return names


def _check_frame(frame: Frame, declaration: CloneDeclaration) -> None:
    """Refuse a frame the declaration cannot clone, before anything is minted."""

    entities = frame.entities
    named = {*declaration.clone_index_columns, *declaration.source_id_columns}
    unknown = sorted(named.difference(entities))
    if unknown:
        raise ValueError(
            f"The clone declaration names entities the frame lacks: {unknown}."
        )
    unindexed = [e for e in entities if e not in declaration.clone_index_columns]
    if unindexed:
        raise ValueError(
            f"The clone declaration gives {unindexed} no clone-index column."
        )
    declared = {
        *declaration.clone_index_columns.values(),
        *declaration.source_id_columns.values(),
    }
    carried = set().union(*(frame.table(e).columns for e in entities))
    held = sorted(declared & carried)
    if held:
        raise ValueError(f"The frame already carries the declared column(s) {held}.")
    if frame.links:
        raise ValueError("The clone expansion copies no link table; the frame has one.")
    for entity in frame.weighted_entities:
        kind = frame.weights_for(entity).kind
        if kind not in declaration.weight_kinds:
            accepted = sorted(k.value for k in declaration.weight_kinds)
            raise ValueError(
                f"The clone declaration accepts {accepted} weights, not "
                f"{kind.value!r} weights on {entity!r}."
            )
    schema = frame.schema
    person = frame.table(schema.person_entity)
    columns = [(frame.table(e), schema.entity_id_column(e)) for e in entities]
    columns += [(person, schema.membership_column(g)) for g in schema.group_entities]
    for table, column in columns:
        dtype = table[column].dtype
        if not (isinstance(dtype, np.dtype) and dtype.kind in "iu"):
            raise TypeError(
                f"The clone expansion mints numpy integer ids; {column!r} holds "
                f"{dtype}."
            )
    # A membership points at its unit's minted ids, so it must share their
    # dtype, or a clone count above one would change it.
    for group in schema.group_entities:
        membership, unit = schema.membership_column(group), schema.id_column(group)
        found, wanted = person[membership].dtype, frame.table(group)[unit].dtype
        if found != wanted:
            raise TypeError(
                f"The clone expansion needs {membership!r} in the dtype of "
                f"{unit!r}; it holds {found} against {wanted}."
            )
    for entity in entities:
        column = schema.entity_id_column(entity)
        declaration.ids.require_below(
            frame.table(entity)[column].to_numpy(copy=True), f"{column} values"
        )


def _clone_ids(
    frame: Frame, entity: str, count: int, rule: CloneIdRule
) -> list[np.ndarray]:
    """``entity``'s ids in each clone: the frame's, then each clone's minted.

    The frame's ids are distinct (and a group's ascend), so a mint that keeps
    their order strictly and puts each clone above the one before gives ids
    that are distinct across all clones and ascend in every group table.
    """

    column = frame.schema.entity_id_column(entity)
    original = frame.table(entity)[column].to_numpy(copy=True)
    original.setflags(write=False)
    order = np.argsort(original, kind="stable")
    clones = [original]
    for k in range(1, count):
        minted = np.asarray(rule.mint(original, k))
        # The frame's ids are numpy integers, so a mint in their dtype mints
        # integers, and concatenating the clones keeps the ids' dtype.
        if minted.shape != original.shape or minted.dtype != original.dtype:
            raise ValueError(
                f"The clone-id rule must mint one integer for each {entity} id, "
                "in the ids' own dtype."
            )
        ranked = minted[order]
        if (ranked[1:] <= ranked[:-1]).any():
            raise ValueError(
                f"The clone-id rule must keep the order of the {entity} ids it copies."
            )
        if minted.size and int(minted.min()) <= int(clones[-1].max()):
            raise ValueError(
                f"The clone-id rule must mint every clone's {entity} ids above "
                "the clone before's."
            )
        clones.append(minted)
    return clones


def _member_ids(members: pd.Series, unit_ids: list[np.ndarray]) -> np.ndarray:
    """A membership column in every clone: clone ``k``'s members point at
    their unit's clone ``k`` id.

    A frame's group ids ascend and are exactly the ids its members name, so
    each member's position in its unit's table is exact.
    """

    values = members.to_numpy(copy=True)
    positions = np.searchsorted(unit_ids[0], values)
    return np.concatenate([values, *(clone[positions] for clone in unit_ids[1:])])


def _equal_shares(values: np.ndarray, count: int) -> np.ndarray:
    """Each row's weight split into ``count`` equal shares, clone-major."""

    return np.tile(values / count, count)


def _split_weights(
    frame: Frame, count: int, reason: str
) -> tuple[dict[str, Weights], tuple[MassChangeRecord, ...]]:
    """Every weighted entity's split weights, and the mass record each leaves,
    in the frame's entity order."""

    weights: dict[str, Weights] = {}
    records: list[MassChangeRecord] = []
    for entity in frame.weighted_entities:
        before = frame.weights_for(entity)
        after = Weights(_equal_shares(before.values, count), before.kind)
        if not np.isclose(after.total, before.total, rtol=CLONE_MASS_RTOL, atol=0.0):
            raise ValueError(
                f"The clones' {entity} weights stray from the frame's total mass "
                f"beyond the relative tolerance {CLONE_MASS_RTOL}."
            )
        weights[entity] = after
        records.append(
            MassChangeRecord(
                entity=entity,
                old_total=before.total,
                new_total=after.total,
                declared_factor=1.0,
                reason=reason,
            )
        )
    return weights, tuple(records)
