"""Project supplied source/structural lineage to stable UK geography keys.

This is a pure descriptive adapter, not a source owner or graph admission gate.
The host must bind roots, every before/after axis, EXPAND parent receipt and
explicit ordinal to its actual retained observations before and after use.
No identifier is reconstructed from offsets, weights, financial flags or order.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from . import atomic_household_identity as identity
from .atomic_area_support import IDENTITY_COLUMN

_ROOT_COLUMNS = ("household_id", "source", "source_vintage", "source_household_id")


@dataclass(frozen=True)
class HouseholdExpansion:
    """One checked EXPAND's household axis and explicitly supplied ordinals.

    parent_pairs is the immutable household portion of the shared executor's
    ``receipt['expand']`` mapping. child_ordinals names each new household's
    declared branch ordinal; its position in the receipt is not an ordinal.
    Existing households retain their previous path without adding a zero step.
    Constructing this object does not authenticate any of these claims.
    """

    branch: str
    before_ids: tuple[int, ...]
    after_ids: tuple[int, ...]
    parent_pairs: tuple[tuple[int, int], ...]
    child_ordinals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HouseholdSelection:
    """A checked FILTER's explicit household axes; no new identity is created."""

    before_ids: tuple[int, ...]
    after_ids: tuple[int, ...]


def _require(condition, reason):
    if not condition:
        raise ValueError("UK atomic household lineage: " + reason)


def _integer(value, *, ordinal=False):
    _require(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
        "exact integer required",
    )
    value = int(value)
    _require(
        0 <= value < 2**32 if ordinal else 0 < value < 2**63,
        "ordinal or ID outside its exact declared range",
    )
    return value


def _axis(values):
    _require(type(values) is tuple, "axes must be immutable tuples")
    ids = tuple(_integer(value) for value in values)
    _require(len(ids) == len(set(ids)), "duplicate household ID")
    return ids


def _pairs(values, *, ordinals=False):
    _require(type(values) is tuple, "pairs must be an immutable tuple")
    pairs = {}
    for row in values:
        _require(type(row) is tuple and len(row) == 2, "exact immutable pair required")
        target = _integer(row[0])
        source = _integer(row[1], ordinal=ordinals)
        _require(target not in pairs, "duplicate pair target")
        pairs[target] = source
    return pairs


def _roots(roots):
    _require(type(roots) is pd.DataFrame, "a plain root description table is required")
    _require(
        roots.columns.is_unique
        and set(roots.columns) == set(_ROOT_COLUMNS)
        and len(roots) > 0,
        "exact nonempty root description columns required",
    )
    for column in ("household_id", "source_household_id"):
        _require(roots[column].dtype == np.dtype("int64"), "root IDs must be int64")
    # Only immutable validated scalar values survive this copy. pandas row
    # labels, column ordering and physical row order do not enter the key.
    copied = roots.loc[:, list(_ROOT_COLUMNS)].copy(deep=True)
    result = {}
    initial_keys = set()
    for row in copied.itertuples(index=False, name=None):
        household_id, source, vintage, source_id = row
        household_id = _integer(household_id)
        source_id = _integer(source_id)
        _require(household_id not in result, "duplicate root household ID")
        key = identity.household_draw_key(
            source=source,
            source_vintage=vintage,
            source_household_id=source_id,
            clone_path=(),
        )
        _require(key not in initial_keys, "duplicate original source identity")
        initial_keys.add(key)
        result[household_id] = (source, vintage, source_id, ())
    return result


def project_atomic_household_keys(
    roots: pd.DataFrame,
    *,
    steps: tuple[HouseholdExpansion | HouseholdSelection, ...],
    final_ids: np.ndarray,
) -> pd.DataFrame:
    """Return exact final household IDs and canonical geography keys.

    Membership of each before axis must equal the preceding result; physical
    ordering may differ because it is not identity. Every added row must have
    one known preceding parent and one explicit ordinal unique for that parent
    in this branch. Branch stages must follow the existing UK branch ordering.
    A selection may only remove IDs; unexplained arrivals or losses refuse.
    Empty selection results are valid descriptions, not viable build evidence.

    The output follows final_ids and is detached. All supplied descriptions
    remain unissued; this checks internal consistency, not source authenticity,
    complete-population ancestry, a sampling rule or release eligibility.
    """
    current = _roots(roots)
    _require(type(steps) is tuple, "steps must be an immutable tuple")
    _require(
        type(final_ids) is np.ndarray
        and final_ids.dtype == np.dtype("int64")
        and final_ids.ndim == 1,
        "final household axis must be a one-dimensional int64 ndarray",
    )
    final = _axis(tuple(final_ids))
    previous_branch = -1
    for step in steps:
        _require(
            type(step) in (HouseholdExpansion, HouseholdSelection),
            "unsupported structural description",
        )
        before = _axis(step.before_ids)
        after = _axis(step.after_ids)
        _require(
            set(before) == set(current), "before axis differs from retained history"
        )
        if type(step) is HouseholdSelection:
            _require(set(after) <= set(before), "selection contains new households")
            current = {target: current[target] for target in after}
            continue
        _require(
            type(step.branch) is str and step.branch in identity._BRANCHES,
            "unknown UK structural branch",
        )
        position = identity._BRANCHES.index(step.branch)
        _require(position > previous_branch, "repeated or reordered structural branch")
        previous_branch = position
        _require(set(before) <= set(after), "expansion removed incumbent households")
        arrivals = set(after) - set(before)
        parents = _pairs(step.parent_pairs)
        ordinals = _pairs(step.child_ordinals, ordinals=True)
        _require(
            set(parents) == arrivals == set(ordinals),
            "parents and ordinals must cover exactly the new households",
        )
        _require(set(parents.values()) <= set(before), "parent absent before expansion")
        branch_ids = [(parents[target], ordinals[target]) for target in arrivals]
        _require(
            len(branch_ids) == len(set(branch_ids)),
            "repeated ordinal for one parent in this branch",
        )
        extended = dict(current)
        for target in arrivals:
            source, vintage, source_id, path = current[parents[target]]
            new_path = (*path, (step.branch, ordinals[target]))
            # Reuse the existing complete path contract; no alternate key
            # encoding or static branch roster is introduced here.
            identity.household_draw_key(
                source=source,
                source_vintage=vintage,
                source_household_id=source_id,
                clone_path=new_path,
            )
            extended[target] = (source, vintage, source_id, new_path)
        current = {target: extended[target] for target in after}
    _require(set(final) == set(current), "final axis differs from completed history")
    keys = [
        identity.household_draw_key(
            source=current[target][0],
            source_vintage=current[target][1],
            source_household_id=current[target][2],
            clone_path=current[target][3],
        )
        for target in final
    ]
    _require(len(keys) == len(set(keys)), "final geography keys are not unique")
    return pd.DataFrame(
        {
            "household_id": np.asarray(final, dtype=np.int64),
            IDENTITY_COLUMN: pd.array(keys, dtype=pd.StringDtype(storage="python")),
        }
    )
