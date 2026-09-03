"""Sealed deny-list of pool publications excluded from readiness, release, and
publication (microcosm#856).

It lives in ``microcosm-data`` because the publisher, which never loads an
H5, must consult it too; ``microcosm.build.us_runtime.h5_io`` re-exports it and
enforces it at every loader. Entries are source-coded and immutable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "DENIED_POOL_PUBLICATIONS",
    "DeniedPoolPublication",
    "denied_pool_publication_for",
]


@dataclass(frozen=True)
class DeniedPoolPublication:
    """One source-coded pool identity excluded from readiness and release."""

    manifest_sha256: str
    pool_h5_sha256: str
    content_identity_sha256: str
    release_id: str
    reason: str
    reference: str


DENIED_POOL_PUBLICATIONS: Mapping[str, DeniedPoolPublication] = MappingProxyType(
    {
        "2ab3f5a136bf4033be876bf150a6fbb4": DeniedPoolPublication(
            manifest_sha256=(
                "2a06fc2b1b73b006bb1bae7d13daeef813a4645c989374408eceaed0ef321cbd"
            ),
            pool_h5_sha256=(
                "45f401735d7c5dc75da78be01bec4db7bf49ef074f69cecf39a1d5b1d77d7b9b"
            ),
            content_identity_sha256=(
                "f5a5023bb9a74003d433abf04c796c96da0a34c6a7caff78b70fee421c4a7b2c"
            ),
            release_id=(
                "populace-us-2024-stacked-f025-s578-asec42213-acs382903-"
                "20260831T162338Z-e14b24e8"
            ),
            reason=(
                "candidate-26 is gate_failed on its terminal by-origin battery "
                "and is excluded from the certifiable dense line by decision"
            ),
            reference="microcosm#856; plan gate 20260902-220844-plan-532dab66",
        )
    }
)


def denied_pool_publication_for(
    *,
    publication_run_id: object = None,
    manifest_sha256: object = None,
    pool_h5_sha256: object = None,
    content_identity_sha256: object = None,
) -> tuple[str, DeniedPoolPublication, str] | None:
    """Return the denied entry an identity matches, and which identity matched."""

    for run_id, denied in DENIED_POOL_PUBLICATIONS.items():
        if publication_run_id == run_id:
            return run_id, denied, "publication_run_id"
        if manifest_sha256 == denied.manifest_sha256:
            return run_id, denied, "manifest_sha256"
        if pool_h5_sha256 == denied.pool_h5_sha256:
            return run_id, denied, "pool_h5_sha256"
        if content_identity_sha256 == denied.content_identity_sha256:
            return run_id, denied, "content_identity_sha256"
    return None
