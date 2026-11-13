"""The names that make a UK spine geography-first (microcosm#931).

A geography-first spine runs every stage ``GEOGRAPHY_FIRST_STAGES`` names and
``was_wealth``'s ``localise_main_residence_value`` operation, and its build
sidecar records how its area draw was bound in a ``geography`` block under
``SIDECAR_GEOGRAPHY_KEYS``. The dense line builds only on such a spine
(:mod:`~microcosm.build.uk_runtime.spine_posture`), and the spine build's
sidecar writer writes the block under these keys. The stages' transforms read
their own names from here, so a stage and the roster cannot name it apart; a
test holds the two together too. Names only, importing nothing first-party.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "AREA_BRMA_STAGE",
    "ATOMIC_AREA_ASSIGN_STAGE",
    "GEOGRAPHIC_EXPAND_STAGE",
    "GEOGRAPHIC_IDENTITY_STAGE",
    "GEOGRAPHY_FIRST_OPERATIONS",
    "GEOGRAPHY_FIRST_STAGES",
    "LOCAL_AUTHORITY_STAGE",
    "PRIVATE_RENT_STAGE",
    "RFDR_STAGE",
    "SIDECAR_GEOGRAPHY_KEYS",
    "SPI_COPY_AREAS_STAGE",
]

#: Each geography-first stage's name, as its transform runs it: the clones,
#: their keys, the area draw, the local authority, the fuel-relief flag, the
#: BRMA, the SPI copies' areas and the private rents.
GEOGRAPHIC_EXPAND_STAGE = "geographic_expand"
GEOGRAPHIC_IDENTITY_STAGE = "geographic_identity"
ATOMIC_AREA_ASSIGN_STAGE = "atomic_area_assign"
LOCAL_AUTHORITY_STAGE = "local_authority"
RFDR_STAGE = "rfdr"
AREA_BRMA_STAGE = "area_brma"
SPI_COPY_AREAS_STAGE = "spi_copy_areas"
PRIVATE_RENT_STAGE = "private_rent"

#: The stages a geography-first spine runs, in the spine's order.
GEOGRAPHY_FIRST_STAGES: tuple[str, ...] = (
    GEOGRAPHIC_EXPAND_STAGE,
    GEOGRAPHIC_IDENTITY_STAGE,
    ATOMIC_AREA_ASSIGN_STAGE,
    LOCAL_AUTHORITY_STAGE,
    RFDR_STAGE,
    AREA_BRMA_STAGE,
    SPI_COPY_AREAS_STAGE,
    PRIVATE_RENT_STAGE,
)

#: The operations a geography-first spine runs inside stages that are not its
#: own, by stage: the main-residence localiser runs inside the WAS chain.
GEOGRAPHY_FIRST_OPERATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"was_wealth": ("localise_main_residence_value",)}
)

#: The keys of the build sidecar's ``geography`` block, in the order the
#: writer writes them: the assignment, the identity column it draws on, the
#: definition's digest, the support pins, the draw stream and the register the
#: pins are held to.
SIDECAR_GEOGRAPHY_KEYS: tuple[str, ...] = (
    "assignment",
    "identity",
    "definition_sha256",
    "support_pins",
    "stream",
    "register",
)
