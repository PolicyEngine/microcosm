"""US alias for the shared weighted per-column input-mass helper.

The implementation was promoted to :mod:`microcosm.build.input_mass` when the
UK terminal battery adopted the same #278 input-mass parity gate (#609); the
computation is schema-driven and never carried US-specific logic. This module
remains so existing US call sites keep their import path.
"""

from __future__ import annotations

from microcosm.build.input_mass import input_mass_totals

__all__ = ["us_input_mass_totals"]

us_input_mass_totals = input_mass_totals
