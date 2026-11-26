"""Two universes: a business register and a tax-registration count.

A firm build often calibrates to two populations that overlap but are not the
same. One is a business register (the frame), counted by band. The other is
a tax administration's count of registered traders by the same bands, which
includes traders outside the register and leaves out registered businesses
that are not registered for the tax. Each frame firm enters the calibration
with its band's registration propensity (registered traders divided by the
band's frame mass). After calibration, a weighted selection per band marks
which firms are in the tax's scope and which are registered.

What differs by country is data, not code: the band labels and what each band
means for registration. A :class:`RegistrationSpec` declares them. Each band
has a role:

- ``"excluded"``: frame firms in the band are neither in scope nor
  registered, and their propensity is 0. Registered traders outside the frame
  sit here (for example, traders with zero or negative turnover).
- ``"registrable"``: every frame firm in the band is in scope (it could
  register), and a weighted selection reaching the band's registered count
  is registered. This is the band below a registration threshold.
- ``"selected"``: a weighted selection reaching the band's registered count
  is both in scope and registered; the rest of the band is out of scope.
  These are the bands above a threshold, where an unregistered firm is
  outside the tax (exempt, or registered for another tax only).

Rows outside the frame (``frame_mask`` False) are registered traders by
construction: in scope, registered, with propensity 1.

Registered counts arrive per call, aligned with the spec's bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ._arrays import as_1d

__all__ = [
    "BAND_ROLES",
    "RegistrationSpec",
    "assign_scope_and_registration",
    "registration_propensity",
    "select_to_weighted_target",
]

#: The roles a band can take; see the module docstring.
BAND_ROLES: Final[tuple[str, ...]] = ("excluded", "registrable", "selected")


@dataclass(frozen=True)
class RegistrationSpec:
    """A country's registration bands and the role each band plays.

    ``band_labels[b]`` names band ``b``; ``band_roles[b]`` is its role, one
    of :data:`BAND_ROLES`. Band indices passed to the functions here index
    these tuples.
    """

    band_labels: tuple[str, ...]
    band_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.band_labels)
        roles = tuple(str(role) for role in self.band_roles)
        if not labels:
            raise ValueError("a registration spec needs at least one band")
        if len(labels) != len(roles):
            raise ValueError(
                f"a registration spec has {len(labels)} band labels "
                f"and {len(roles)} band roles"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("registration band labels must be unique")
        unknown = sorted(set(roles) - set(BAND_ROLES))
        if unknown:
            raise ValueError(
                f"unknown registration band roles {unknown}; choose from {BAND_ROLES}"
            )
        object.__setattr__(self, "band_labels", labels)
        object.__setattr__(self, "band_roles", roles)

    @property
    def n_bands(self) -> int:
        return len(self.band_labels)

    def bands_with_role(self, role: str) -> tuple[int, ...]:
        """The indices of the bands with ``role``, in index order."""

        if role not in BAND_ROLES:
            raise ValueError(f"unknown registration band role {role!r}")
        return tuple(b for b, r in enumerate(self.band_roles) if r == role)


def _counts(spec: RegistrationSpec, registered_counts: object) -> np.ndarray:
    counts = as_1d("registered_counts", registered_counts, spec.n_bands).astype(
        np.float64
    )
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("registered_counts must hold finite, non-negative counts")
    return counts


def _band_indices(
    spec: RegistrationSpec, band_idx: object, length: int | None = None
) -> np.ndarray:
    bands = as_1d("band_idx", band_idx, length)
    if not np.issubdtype(bands.dtype, np.integer):
        raise TypeError("band_idx must be an integer array")
    if bands.size and (bands.min() < 0 or bands.max() >= spec.n_bands):
        raise ValueError(f"band_idx values must lie in [0, {spec.n_bands - 1}]")
    return bands.astype(np.int64)


def registration_propensity(
    spec: RegistrationSpec,
    band_idx: np.ndarray,
    frame_mask: np.ndarray,
    base_weights: np.ndarray,
    registered_counts: np.ndarray,
) -> np.ndarray:
    """Each row's probability of being a registered trader.

    For each band that is not ``"excluded"``, the propensity of its frame
    rows is the band's registered count divided by its base-weighted frame
    mass, capped at 1.0 when the frame cannot hold the registered population.
    A band with no frame mass leaves its rows at 0.0. Frame rows in an
    ``"excluded"`` band get 0.0; rows outside the frame get 1.0.

    Args:
        spec: The bands and their roles.
        band_idx: Integer array, shape ``(n,)``: each row's band.
        frame_mask: Boolean array, shape ``(n,)``: True for frame rows.
        base_weights: Float array, shape ``(n,)``: each row's base weight.
        registered_counts: Float array, shape ``(spec.n_bands,)``.

    Returns:
        Float64 array, shape ``(n,)``, of propensities in ``[0, 1]``.
    """

    bands = _band_indices(spec, band_idx)
    n = len(bands)
    frame = as_1d("frame_mask", frame_mask, n).astype(bool)
    weights = as_1d("base_weights", base_weights, n).astype(np.float64)
    counts = _counts(spec, registered_counts)

    frame_mass = np.bincount(
        bands[frame], weights=weights[frame], minlength=spec.n_bands
    )
    per_band = np.zeros(spec.n_bands, dtype=np.float64)
    positive = frame_mass > 0
    per_band[positive] = np.minimum(1.0, counts[positive] / frame_mass[positive])
    per_band[list(spec.bands_with_role("excluded"))] = 0.0

    propensity = np.where(frame, per_band[bands], 1.0)
    return propensity.astype(np.float64)


def select_to_weighted_target(
    candidates: np.ndarray,
    weights: np.ndarray,
    target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``candidates`` in a seeded random order until their weight reaches ``target``.

    The first row whose cumulative weight reaches the target is included, so
    the selected weight overshoots the target by less than one row's weight.
    If the candidates' total weight falls short, every candidate is selected.
    With no candidates, or a target of zero or less, nothing is selected and
    ``rng`` is not drawn from.

    Args:
        candidates: Integer row positions to choose from.
        weights: Float array indexed by row position.
        target: The weight to reach.
        rng: The generator that draws the permutation.

    Returns:
        The selected row positions, in selection order.
    """

    if len(candidates) == 0 or target <= 0:
        return candidates[:0]
    order = candidates[rng.permutation(len(candidates))]
    cumulative = np.cumsum(weights[order])
    cutoff = min(int(np.searchsorted(cumulative, target, side="left")), len(order) - 1)
    return order[: cutoff + 1]


def assign_scope_and_registration(
    spec: RegistrationSpec,
    band_idx: np.ndarray,
    frame_mask: np.ndarray,
    weights: np.ndarray,
    registered_counts: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign scope and registration flags that match the registered counts.

    Rows outside the frame are in scope and registered. Frame rows follow
    their band's role (module docstring). In each band that is not
    ``"excluded"``, the registered weight lies in ``[target, target + w)``
    for the weight ``w`` of the last selected row, unless the band's weight
    falls short of the target, when every row in the band is selected.

    One ``np.random.default_rng(seed)`` generator draws one permutation per
    band that is not ``"excluded"``, in band index order (none for a band
    with no candidates or a zero target).

    Args:
        spec: The bands and their roles.
        band_idx: Integer array, shape ``(n,)``: each row's band.
        frame_mask: Boolean array, shape ``(n,)``: True for frame rows.
        weights: Float array, shape ``(n,)``: calibration (or base) weights.
        registered_counts: Float array, shape ``(spec.n_bands,)``.
        seed: Seed of the selection generator.

    Returns:
        ``(scope, registered)``: boolean arrays of shape ``(n,)``.
    """

    bands = _band_indices(spec, band_idx)
    n = len(bands)
    frame = as_1d("frame_mask", frame_mask, n).astype(bool)
    row_weights = as_1d("weights", weights, n).astype(np.float64)
    counts = _counts(spec, registered_counts)
    rng = np.random.default_rng(int(seed))

    scope = np.zeros(n, dtype=bool)
    registered = np.zeros(n, dtype=bool)
    scope[~frame] = True
    registered[~frame] = True

    for band, role in enumerate(spec.band_roles):
        if role == "excluded":
            continue
        in_band = frame & (bands == band)
        if role == "registrable":
            scope[in_band] = True
        chosen = select_to_weighted_target(
            np.flatnonzero(in_band), row_weights, counts[band], rng
        )
        scope[chosen] = True
        registered[chosen] = True

    return scope, registered
