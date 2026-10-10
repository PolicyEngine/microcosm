"""Verbatim schema-4 arithmetic from a224dc110, for differential tests.

Only the two historical deterministic kernels are frozen; no engine or
resource eligibility is copied.
"""

import hashlib

_OUTPUT = "takes_up_ssi_if_eligible"


def _stable_source_draw(source_id: str, *, seed: int) -> float:
    value = int.from_bytes(
        hashlib.blake2b(
            f"{seed}:{_OUTPUT}:{source_id}".encode(),
            digest_size=8,
        ).digest(),
        byteorder="big",
        signed=False,
    )
    return value / float(2**64)


def _band_prior(target: float, capacity: float, reporter_floor: float) -> float:
    """Return the documented Bernoulli prior for one SSA age band.

    Reporter anchors are selected unconditionally, so the expected delivered
    candidate mass at non-anchor threshold ``p`` is ``floor + p*(capacity -
    floor)``. The count-truthful threshold is therefore ``(target - floor) /
    (capacity - floor)`` — a naive ``target/capacity`` overshoots by
    ``floor*(1 - target/capacity)`` whenever anchors exist (microcosm#507
    sol review finding 4; with Build-N-scale floors that first-order error
    is 15–22%, beyond the 5% delivery envelope). When the anchors alone
    meet or exceed the target, non-anchors draw at zero and the anchored
    mass ships as ``anchor_excess``.

    The threshold only subsamples while ``capacity > target``. Otherwise it
    would flag the whole band — a constant, signal-free output — so the
    prior falls back to the observed take-up rate among the basis
    candidates: reporter mass over candidate capacity. Reform-created
    eligibles then take up at the rate the basis candidates are observed
    reporting.
    """

    if capacity <= 0:
        return 0.0
    # Anchors-meet-target outranks the saturation fallback: at the
    # degenerate capacity == target == floor corner the fallback would
    # return floor/capacity == 1.0 — a constant, count-overshooting band —
    # when zero non-anchor draws is the truthful answer (sol review round
    # 2, finding 4).
    if reporter_floor >= target:
        return 0.0
    if capacity <= target:
        return min(reporter_floor / capacity, 1.0)
    return (target - reporter_floor) / (capacity - reporter_floor)
