"""Country-neutral mechanics for synthetic firm populations.

A country's firm build draws candidate firms from published band counts,
thins them for speed, calibrates them against a business register and a
tax-registration count, and hands them to a rule engine. The pieces here know
nothing about any country: band edges, band roles, registered counts and rule
engines all arrive as arguments. The UK firm build in ``uk_runtime`` is the
first consumer.

- :mod:`.bands`: values drawn inside a band from uniforms the caller supplies;
- :mod:`.thinning`: stratified thinning with exact base weights;
- :mod:`.registration`: the two-universe treatment of a register and a
  tax-registration count, set by a :class:`RegistrationSpec`;
- :mod:`.rules`: the seam to a rule engine, with outputs keyed by firm id.
"""

from __future__ import annotations

from .bands import draw_in_band
from .registration import (
    BAND_ROLES,
    RegistrationSpec,
    assign_scope_and_registration,
    registration_propensity,
    select_to_weighted_target,
)
from .rules import FirmRuleOutputs, FirmRules, align_rule_outputs, weighted_totals
from .thinning import stratified_thin

__all__ = [
    "BAND_ROLES",
    "FirmRuleOutputs",
    "FirmRules",
    "RegistrationSpec",
    "align_rule_outputs",
    "assign_scope_and_registration",
    "draw_in_band",
    "registration_propensity",
    "select_to_weighted_target",
    "stratified_thin",
    "weighted_totals",
]
