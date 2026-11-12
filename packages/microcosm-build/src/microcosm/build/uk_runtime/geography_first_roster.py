"""The names that make a UK spine geography-first (microcosm#931).

A geography-first spine runs every stage ``GEOGRAPHY_FIRST_STAGES`` names and
``was_wealth``'s ``localise_main_residence_value`` operation, and its build
sidecar records how its area draw was bound in a ``geography`` block under
``SIDECAR_GEOGRAPHY_KEYS``. The dense line builds only on such a spine
(:mod:`~microcosm.build.uk_runtime.spine_posture`); the spine build's sidecar
writer is to write the block under these keys, and a test holds the stage
names to their transforms' own. Names only, importing nothing first-party.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "GEOGRAPHY_FIRST_OPERATIONS",
    "GEOGRAPHY_FIRST_STAGES",
    "SIDECAR_GEOGRAPHY_KEYS",
]

#: The stages a geography-first spine runs, in the spine's order.
GEOGRAPHY_FIRST_STAGES: tuple[str, ...] = (
    "geographic_expand",
    "geographic_identity",
    "atomic_area_assign",
    "local_authority",
    "rfdr",
    "area_brma",
    "spi_copy_areas",
    "private_rent",
)

#: The operations a geography-first spine runs inside stages that are not its
#: own, by stage: the main-residence localiser runs inside the WAS chain.
GEOGRAPHY_FIRST_OPERATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"was_wealth": ("localise_main_residence_value",)}
)

#: The keys of the build sidecar's ``geography`` block: the assignment, the
#: identity column it draws on, the definition's digest, the support pins, the
#: draw stream and the register the pins are held to.
SIDECAR_GEOGRAPHY_KEYS: tuple[str, ...] = (
    "assignment",
    "identity",
    "definition_sha256",
    "support_pins",
    "stream",
    "register",
)
