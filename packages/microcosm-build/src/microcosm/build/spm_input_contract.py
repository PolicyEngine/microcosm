"""Names and three-state annual scope for the native SPM input boundary.

These labels describe source qualification, not poverty outcomes or release
certification. The consuming country validates its own dataset contract.
"""

ROLE_INPUT = "is_spm_independent_minor_role"
UNIVERSE_INPUT = "spm_unit_spm_universe_status"
INCLUDED = "INCLUDED"
OUTSIDE = "OUTSIDE"
UNRESOLVED = "UNRESOLVED"
UNIVERSE_STATUSES = frozenset((INCLUDED, OUTSIDE, UNRESOLVED))
