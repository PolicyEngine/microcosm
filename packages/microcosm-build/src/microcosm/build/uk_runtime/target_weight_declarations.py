"""The UK local weight rules, declared as the shared rules (microcosm#1729).

Each name in the UK local vocabulary
(:data:`~microcosm.build.uk_runtime.local_doctrine.UK_LOCAL_TARGET_WEIGHT_RULES`)
is one shared rule of :mod:`microcosm.calibrate.target_loss_rules` with its
parameters, and one way to say which level a UK row weighs in. That is all a
country has to declare: the arithmetic is the shared rule's.

**The UK solve does not read this module.**
:func:`~microcosm.build.uk_runtime.target_weights.uk_rule_loss_weights` still
computes its vectors with its own code. That module's bytes are part of the UK
graph's code identity (``uk.full.problem@1`` and the rotated holdout hash it),
so an edit there re-keys those nodes and every solve after them.
:func:`uk_shared_rule_loss_weights` computes the same vectors through the
shared rule, and ``test_uk_target_weight_declarations.py`` holds the two equal
bit for bit under every rule. So moving ``target_weights`` onto this function
changes which code computes the vector, not the vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_TARGET_WEIGHT_RULE,
    UK_LOCAL_TARGET_WEIGHT_RULES,
)
from microcosm.build.uk_runtime.target_weights import (
    UKTargetRows,
    _rule_grains,
    uk_target_value_basis,
)
from microcosm.calibrate.target_loss_rules import (
    TargetLossRows,
    TargetWeightRule,
    target_loss_weights_for_rule,
)

__all__ = [
    "UK_LOCAL_NATION_LEVEL_RULES",
    "UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATION",
    "UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS",
    "uk_shared_rule_loss_weights",
    "uk_target_loss_rows",
]

#: The UK count-row power inside a (level, family) cell: the square root, as
#: the US value weight (microcosm#1104, microcosm#1124).
UK_LOCAL_COUNT_POWER = 0.5

#: Each UK rule name as a shared rule with its parameters.
UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS: Mapping[str, TargetWeightRule] = (
    MappingProxyType(
        {
            "uniform": TargetWeightRule("uniform"),
            "grain_equal": TargetWeightRule("level_equal"),
            "grain_family_equal": TargetWeightRule("level_family_equal"),
            "grain_family_equal_sqrt_count": TargetWeightRule(
                "level_family_equal", {"count_power": UK_LOCAL_COUNT_POWER}
            ),
            "nation_grain_family_equal": TargetWeightRule("level_family_equal"),
            "nation_grain_family_equal_sqrt_count": TargetWeightRule(
                "level_family_equal", {"count_power": UK_LOCAL_COUNT_POWER}
            ),
        }
    )
)

#: The rules under which a national row that is not a UK-wide aggregate (a
#: nation or an English region) weighs in a level of its own, ``nation_region``.
#: Under the others every national row weighs in the national level.
UK_LOCAL_NATION_LEVEL_RULES = frozenset(
    {"nation_grain_family_equal", "nation_grain_family_equal_sqrt_count"}
)

#: The doctrine's rule, as a shared declaration.
UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATION = UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS[
    UK_LOCAL_TARGET_WEIGHT_RULE
]


def _declaration(rule: str) -> TargetWeightRule:
    if rule not in UK_LOCAL_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {UK_LOCAL_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    return UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS[rule]


def uk_target_loss_rows(rows: UKTargetRows, *, rule: str) -> TargetLossRows:
    """What each UK row is, as the shared rule ``rule`` names reads it.

    The level is the row's weighting grain (a BRMA row weighs in the
    local-authority grain; under a nation rule a nation or region row leaves
    the national grain). The basis is read only for a rule with a count
    power, so a rule without one still accepts a row whose basis is not
    declared, as the UK code does.
    """

    declaration = _declaration(rule)
    if not isinstance(rows, UKTargetRows):
        raise TypeError("UK target loss rows need a UKTargetRows carrier.")
    count_power = declaration.parameters.get("count_power")
    return TargetLossRows(
        values=rows.values,
        levels=_rule_grains(rows, nation=rule in UK_LOCAL_NATION_LEVEL_RULES),
        families=rows.family,
        bases=uk_target_value_basis(rows) if count_power is not None else None,
    )


def uk_shared_rule_loss_weights(rows: UKTargetRows, *, rule: str) -> np.ndarray:
    """The loss-weight vector of a UK rule, computed by the shared rule."""

    declaration = _declaration(rule)
    if declaration.name != "uniform" and not len(rows):
        raise ValueError("target weighting requires at least one target.")
    return target_loss_weights_for_rule(
        uk_target_loss_rows(rows, rule=rule), declaration
    )
