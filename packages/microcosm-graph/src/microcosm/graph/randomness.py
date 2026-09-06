"""Stable random coordinates, independent of graph packing and cache identity.

A stream is ("sha256-u53-v1", experiment_id, replicate, base_seed). Each
nonempty coordinate tuple identifies a draw, conventionally (person_id,
process, period, draw_index). The top 53 bits of the SHA-256 digest, interpreted
big-endian, divided by 2**53 define a uniform in [0, 1). Stream parameters must
be normative node params; kernels must hash this module as implementing source
and declare SeedSource.KEYED. Repeated coordinates intentionally repeat draws.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from .canonical import canonical_json

__all__ = ["keyed_uniform"]


def _coordinate(value: object) -> list[object]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, float) and math.isfinite(value):
        return ["float", value]
    raise TypeError("Random coordinates must be non-null finite scalar identities.")


def keyed_uniform(*, stream: tuple, keys: Sequence[tuple]) -> np.ndarray:
    """Return bytes-backed, read-only float64 draws keyed by stable coordinates.

    Row order, chunk boundaries, and unrelated inserted identities cannot affect
    a draw. Integers, strings, booleans and floats have distinct canonical tags.
    The experiment name is nonempty; replicate and base seed are non-negative
    Python integers (booleans refused). No numpy RNG state is read or mutated.
    """
    if not isinstance(stream, tuple) or len(stream) != 4:
        raise TypeError(
            "stream must be (algorithm, experiment_id, replicate, base_seed)."
        )
    algorithm, experiment, replicate, base_seed = stream
    if algorithm != "sha256-u53-v1":
        raise ValueError(f"Unsupported random stream algorithm {algorithm!r}.")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("Random stream experiment_id must be non-empty.")
    if any(type(value) is not int or value < 0 for value in (replicate, base_seed)):
        raise ValueError(
            "Random stream replicate/base_seed must be non-negative integers."
        )
    prefix = b"microcosm-graph/keyed-uniform/1\0" + canonical_json(stream) + b"\0"
    values = []
    for key in keys:
        if not isinstance(key, tuple) or not key:
            raise TypeError("Each random key must be a nonempty coordinate tuple.")
        encoded = canonical_json([_coordinate(value) for value in key])
        digest = hashlib.sha256(prefix + encoded).digest()
        values.append((int.from_bytes(digest[:8], "big") >> 11) / 2**53)
    return np.frombuffer(
        np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64
    )
