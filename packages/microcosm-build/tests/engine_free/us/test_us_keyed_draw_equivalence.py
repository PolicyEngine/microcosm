"""The US keyed draws are ``stable_identity_uniforms``, bit for bit (#1585).

Eleven US modules each carried a copy of the seeded BLAKE2b loop that
:func:`microcosm.build.stochastic_assignment.stable_identity_uniforms`
implements: ``blake2b(f"{seed}:{salt}:{key}", digest_size=8)``, read as a
big-endian unsigned integer and divided by ``2**64``. Each copy is frozen
below as main had it at 90e4d94e7, with its salt written out, and each test
gives a module's helper and the frozen copy the same inputs. They must return
the same float64 bits in the same container: an array of the same shape, a
Series on the same index, or a Python float.

The helpers keep their private names, because other modules and test support
import them; the last test checks that those importers still reach them.

The shared helper formats ``int(seed)``, where three of the copies formatted
the seed as given. The two agree for every integer seed, Python or NumPy, and
every caller passes one: a ``type=int`` command-line flag, an integer
constant, or ``int(...)``.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_local_ssi_disability as acs_local_ssi_disability
import microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up as acs_local_ssi_medicaid_take_up
import microcosm.build.us_runtime.acs_local_take_up as acs_local_take_up
import microcosm.build.us_runtime.acs_local_vehicles_head_start as acs_local_vehicles_head_start
import microcosm.build.us_runtime.immigration as immigration
import microcosm.build.us_runtime.medicaid_take_up as medicaid_take_up
import microcosm.build.us_runtime.pregnancy as pregnancy
import microcosm.build.us_runtime.snap_discretionary_exemption as snap_discretionary_exemption
import microcosm.build.us_runtime.snap_state_take_up as snap_state_take_up
import microcosm.build.us_runtime.snap_take_up as snap_take_up
import microcosm.build.us_runtime.source_runtime as source_runtime
import microcosm.build.us_runtime.ssi_take_up as ssi_take_up
import microcosm.build.us_runtime.take_up as take_up
import microcosm.build.us_runtime.wic_claim as wic_claim
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime.support_provenance import support_source_id_column

#: Python and NumPy integer seeds, including both ends of the signed 64-bit
#: range and an unsigned value above it.
SEEDS = (
    0,
    578,
    -3,
    2**63 - 1,
    np.int64(578),
    np.int32(-3),
    np.uint64(2**64 - 1),
)

#: Raw keys of every kind the copies format: integers, floats, strings and
#: their NumPy forms.
RAW_KEYS = (
    [],
    [0, 1, 2, -5, 10**18],
    [1001.0, 2.5, -0.0, float("nan")],
    ["a", "2024:10:2", "", "ü"],
    np.arange(4, dtype=np.int64),
    np.asarray([1001.0, 2.5], dtype=np.float64),
    np.asarray(["x", "y"]),
)


# ---------------------------------------------------------------------------
# The copies as main had them at 90e4d94e7; only the salts are written out.
# ---------------------------------------------------------------------------


def _main_loop(ids, *, seed, salt):
    """The loop every copy shares, as one reference for the shared helper."""
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{salt}:{value}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for value in ids
        ],
        dtype=np.float64,
    )


def _main_take_up_stable_unit_draws(
    units: pd.DataFrame, *, id_column: str, seed: int, variable: str
) -> np.ndarray:
    entity = id_column.removesuffix("_id")
    source_id_column = support_source_id_column(entity)
    keys = units[id_column].astype(str)
    if source_id_column in units:
        source_ids = units[source_id_column]
        keys = keys.where(source_ids.isna(), source_ids.astype(str))
    identity = ("source_year", "source_household_id", "source_person_id")
    if set(identity) <= set(units.columns):
        complete_identity = units.loc[:, list(identity)].notna().all(axis=1)
        source_keys = (
            units["source_year"].astype(str)
            + ":"
            + units["source_household_id"].astype(str)
            + ":"
            + units["source_person_id"].astype(str)
        )
        keys = keys.where(~complete_identity, source_keys)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{variable}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _main_snap_take_up_stable_unit_draws(
    units: pd.DataFrame, *, seed: int
) -> np.ndarray:
    if {"source_year", "source_household_id", "source_person_id"} <= set(units.columns):
        keys = (
            units["source_year"].astype(str)
            + ":"
            + units["source_household_id"].astype(str)
            + ":"
            + units["source_person_id"].astype(str)
        )
    else:
        keys = units["person_spm_unit_id"].astype(str)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:snap_take_up:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _main_person_draws(persons: pd.DataFrame, *, seed: int, salt: str) -> np.ndarray:
    """Pregnancy's and the SNAP exemption's copy; they differ only in salt."""
    if {"source_year", "source_household_id", "source_person_id"} <= set(
        persons.columns
    ):
        keys = (
            persons["source_year"].astype(str)
            + ":"
            + persons["source_household_id"].astype(str)
            + ":"
            + persons["source_person_id"].astype(str)
        )
    else:
        keys = persons["person_id"].astype(str)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{salt}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _main_ssi_stable_source_draw(source_id: str, *, seed: int) -> float:
    value = int.from_bytes(
        hashlib.blake2b(
            f"{seed}:takes_up_ssi_if_eligible:{source_id}".encode(),
            digest_size=8,
        ).digest(),
        byteorder="big",
        signed=False,
    )
    return value / float(2**64)


def _main_wic_stable_person_draws(person: pd.DataFrame, *, seed: int) -> np.ndarray:
    keys = wic_claim._stable_person_keys(person)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:would_claim_wic:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _main_immigration_stable_person_draws(
    person: pd.DataFrame,
    *,
    seed: int,
    salt: str,
    keys: pd.Series | None = None,
) -> np.ndarray:
    if keys is not None:
        if len(keys) != len(person):
            raise ValueError(
                f"Explicit draw keys cover {len(keys)} row(s); the person "
                f"table has {len(person)}."
            )
        keys = pd.Series(keys).astype(str)
    elif {"source_year", "source_person_id"}.issubset(person.columns):
        keys = (
            person["source_year"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    else:
        keys = person["person_id"].astype(str)
    denominator = float(2**64)
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{salt}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ),
        dtype=np.float64,
        count=len(keys),
    )


def _main_source_runtime_stable_draws(
    frame: pd.DataFrame, *, seed: int, salt: str
) -> pd.Series:
    if "tax_unit_id" in frame:
        keys = frame["tax_unit_id"]
    else:
        keys = pd.Series(frame.index, index=frame.index)
    denominator = float(2**64)
    draws = [
        int.from_bytes(
            hashlib.blake2b(
                f"{seed}:{salt}:{key}".encode(),
                digest_size=8,
            ).digest(),
            byteorder="big",
            signed=False,
        )
        / denominator
        for key in keys
    ]
    return pd.Series(draws, index=frame.index, dtype=float)


def _main_ssi_disability_keyed_uniforms(
    keys: Sequence[str], *, seed: int, stream: str
) -> np.ndarray:
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:acs_local_ssi_disability_criteria:{stream}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _main_medicaid_keyed_uniform(key: str, *, seed: int, stream: str) -> float:
    value = int.from_bytes(
        hashlib.blake2b(f"{seed}:{stream}:{key}".encode(), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )
    return value / float(2**64)


def _main_head_start_keyed_uniforms(keys: Sequence[str], *, seed: int) -> np.ndarray:
    denominator = float(2**64)
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:acs_local_head_start_take_up:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ),
        dtype=np.float64,
        count=len(keys),
    )


# ---------------------------------------------------------------------------
# Bit-for-bit comparisons
# ---------------------------------------------------------------------------


def _assert_same_array(actual, expected: np.ndarray) -> None:
    assert type(actual) is np.ndarray
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    assert np.ascontiguousarray(actual).view(np.uint64).tolist() == (
        np.ascontiguousarray(expected).view(np.uint64).tolist()
    )


def _assert_same_series(actual, expected: pd.Series) -> None:
    assert type(actual) is pd.Series
    assert actual.dtype == np.float64
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.name == expected.name
    _assert_same_array(actual.to_numpy(), expected.to_numpy())


def _assert_same_float(actual, expected: float) -> None:
    assert type(actual) is float
    assert struct.pack(">d", actual) == struct.pack(">d", expected)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _unit_frames(entity: str) -> list[pd.DataFrame]:
    """Unit tables in every keying shape the take-up copy distinguishes."""
    id_column = f"{entity}_id"
    source_column = support_source_id_column(entity)
    return [
        pd.DataFrame({id_column: pd.Series([], dtype=np.int64)}),
        pd.DataFrame({id_column: [3, 1, 2]}),
        pd.DataFrame({id_column: [10.0, 20.0, 30.0]}),
        pd.DataFrame({id_column: ["u1", "u2"]}),
        pd.DataFrame(
            {
                id_column: [1, 2, 3, 4],
                source_column: ["s1", None, "s3", np.nan],
            }
        ),
        pd.DataFrame(
            {
                id_column: [1, 2, 3, 4],
                source_column: ["s1", "s2", None, "s4"],
                "source_year": [2023, 2024, np.nan, 2024],
                "source_household_id": [11, 12, 13, np.nan],
                "source_person_id": [101, 102, 103, 104],
            }
        ),
        pd.DataFrame(
            {
                id_column: [7, 8],
                "source_year": [2024, 2024],
                "source_household_id": [5, 5],
                "source_person_id": [1, 2],
            }
        ),
    ]


def _person_frames() -> list[pd.DataFrame]:
    """Person tables keyed by ``person_id`` or by the source triple."""
    return [
        pd.DataFrame({"person_id": pd.Series([], dtype=np.int64)}),
        pd.DataFrame({"person_id": [5, 1, 3]}),
        pd.DataFrame({"person_id": [1001.0, 2.0]}),
        pd.DataFrame({"person_id": ["p1", "p2"]}),
        pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "source_year": [2022, 2023, 2024],
                "source_household_id": [4, 4, 9],
                "source_person_id": [1, 2, 1],
            }
        ),
        pd.DataFrame(
            {
                "person_id": [1, 2],
                "source_year": [2024.0, np.nan],
                "source_household_id": [4, 5],
                "source_person_id": [1, 1],
            }
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ids", RAW_KEYS, ids=lambda ids: repr(list(ids))[:40])
@pytest.mark.parametrize(
    "salt",
    (
        "snap_take_up",
        "takes_up_tanf_if_eligible",
        "acs_local_ssi_disability_criteria:quantile",
        "",
    ),
)
def test_shared_helper_is_the_copied_loop(seed, ids, salt) -> None:
    """Integer, float and string keys; Python and NumPy seeds; empty input;
    and the composite ``{salt}:{stream}`` salt the SSI disability draw uses."""
    _assert_same_array(
        stable_identity_uniforms(ids, seed=seed, salt=salt),
        _main_loop(ids, seed=seed, salt=salt),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("entity", "variable"),
    (
        ("spm_unit", "takes_up_tanf_if_eligible"),
        ("tax_unit", "takes_up_eitc"),
        ("person", "takes_up_medicaid_if_eligible"),
        ("spm_unit", "takes_up_snap_if_eligible"),
    ),
)
def test_take_up_unit_draws(seed, entity, variable) -> None:
    for units in _unit_frames(entity):
        _assert_same_array(
            take_up._stable_unit_draws(
                units, id_column=f"{entity}_id", seed=seed, variable=variable
            ),
            _main_take_up_stable_unit_draws(
                units, id_column=f"{entity}_id", seed=seed, variable=variable
            ),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_medicaid_person_draws_are_the_take_up_draws(seed) -> None:
    for person in _unit_frames("person"):
        _assert_same_array(
            medicaid_take_up._stable_person_draws(person, seed=seed),
            _main_take_up_stable_unit_draws(
                person,
                id_column="person_id",
                seed=seed,
                variable="takes_up_medicaid_if_eligible",
            ),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_snap_take_up_unit_draws(seed) -> None:
    frames = [
        pd.DataFrame({"person_spm_unit_id": pd.Series([], dtype=np.int64)}),
        pd.DataFrame({"person_spm_unit_id": [4, 2, 9]}),
        pd.DataFrame({"person_spm_unit_id": [4.0, 2.0]}),
        pd.DataFrame(
            {
                "person_spm_unit_id": [1, 2],
                "source_year": [2023, 2024],
                "source_household_id": [10, 10],
                "source_person_id": [2, 3],
            }
        ),
    ]
    for units in frames:
        _assert_same_array(
            snap_take_up._stable_unit_draws(units, seed=seed),
            _main_snap_take_up_stable_unit_draws(units, seed=seed),
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("module", "salt"),
    (
        (snap_discretionary_exemption, "snap_abawd_discretionary_exemption"),
        (pregnancy, "pregnancy"),
    ),
    ids=("snap_discretionary_exemption", "pregnancy"),
)
def test_person_draws(seed, module, salt) -> None:
    for persons in _person_frames():
        _assert_same_array(
            module._stable_person_draws(persons, seed=seed),
            _main_person_draws(persons, seed=seed, salt=salt),
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("source_id", ("asec:2024:1:1", "", "ü", "12", "nan"))
def test_ssi_source_draw(seed, source_id) -> None:
    _assert_same_float(
        ssi_take_up._stable_source_draw(source_id, seed=seed),
        _main_ssi_stable_source_draw(source_id, seed=seed),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_wic_person_draws(seed) -> None:
    frames = [
        pd.DataFrame({"person_id": pd.Series([], dtype=np.int64)}),
        pd.DataFrame({"person_id": [3, 1]}),
        pd.DataFrame({"person_id": [1, 2], "person_support_source_id": ["a", 7]}),
        pd.DataFrame(
            {
                "person_id": [1, 2],
                "source_year": [2024, 2024],
                "source_household_id": [3, 3],
                "source_person_id": [1, 2],
            }
        ),
        pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_spine_source_id": [11, 12],
                "person_source_id": ["asec:1", "acs:1"],
            }
        ),
    ]
    for person in frames:
        _assert_same_array(
            wic_claim._stable_person_draws(person, seed=seed),
            _main_wic_stable_person_draws(person, seed=seed),
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "salt", ("immigration:ead_workers", "immigration:ead_students", "other")
)
def test_immigration_person_draws(seed, salt) -> None:
    cases = [
        (pd.DataFrame({"person_id": pd.Series([], dtype=np.int64)}), None),
        (pd.DataFrame({"person_id": [9, 4, 1]}), None),
        (pd.DataFrame({"person_id": [9.0, 4.0]}), None),
        (
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "source_year": [2024, 2023],
                    "source_person_id": [5, 5],
                }
            ),
            None,
        ),
        (pd.DataFrame({"person_id": [1, 2, 3]}), pd.Series(["k1", "k2", "k3"])),
        (pd.DataFrame({"person_id": [1, 2]}), ["acs:1:1", "acs:1:2"]),
        (pd.DataFrame({"person_id": [1, 2]}), np.asarray([7, 8], dtype=np.int64)),
    ]
    for person, keys in cases:
        _assert_same_array(
            immigration._stable_person_draws(person, seed=seed, salt=salt, keys=keys),
            _main_immigration_stable_person_draws(
                person, seed=seed, salt=salt, keys=keys
            ),
        )


def test_immigration_refuses_misaligned_keys_as_before() -> None:
    person = pd.DataFrame({"person_id": [1, 2]})
    for draws in (
        immigration._stable_person_draws,
        _main_immigration_stable_person_draws,
    ):
        with pytest.raises(
            ValueError, match=r"cover 1 row\(s\); the person table has 2"
        ):
            draws(person, seed=0, salt="immigration:ead_workers", keys=["k"])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "salt",
    (
        "aca:takes_up_aca_if_eligible",
        "calibrate:takes_up_x",
        "joint-calibrate:takes_up_y",
    ),
)
def test_source_runtime_draws_keep_raw_keys_and_index(seed, salt) -> None:
    """The ACA-family copy keys on raw ``tax_unit_id`` values, or on the
    positional index without one, and returns a Series on the frame's index."""
    frames = [
        pd.DataFrame({"tax_unit_id": pd.Series([], dtype=np.int64)}),
        pd.DataFrame({"tax_unit_id": [99, 1, 5]}),
        pd.DataFrame({"tax_unit_id": [1001.0, 2.0]}, index=[10, 20]),
        pd.DataFrame({"tax_unit_id": ["t1", "t2"]}),
        pd.DataFrame({"tax_unit_id": pd.array([1, None], dtype="Int64")}),
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"x": [1.0, 2.0]}, index=[7, 3]),
        pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="row")),
        pd.DataFrame({"x": pd.Series([], dtype=float)}),
    ]
    for frame in frames:
        _assert_same_series(
            source_runtime._stable_draws(frame, seed=seed, salt=salt),
            _main_source_runtime_stable_draws(frame, seed=seed, salt=salt),
        )


_ACS_KEYS = (
    [],
    ["acs_2024_1yr:2024HU0000001:1"],
    ["acs_2024_1yr:2024HU0000001:1", "acs_2024_1yr:2024GQ0000002:3", "ü"],
)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("keys", _ACS_KEYS)
@pytest.mark.parametrize("stream", ("quantile", "sign"))
def test_acs_ssi_disability_uniforms(seed, keys, stream) -> None:
    _assert_same_array(
        acs_local_ssi_disability._keyed_uniforms(keys, seed=seed, stream=stream),
        _main_ssi_disability_keyed_uniforms(keys, seed=seed, stream=stream),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("key", ("acs_2024_1yr:2024HU0000001:1", "", "ü"))
@pytest.mark.parametrize(
    "stream", ("acs_local_medicaid_hins4_anchor_keep", "takes_up_medicaid_if_eligible")
)
def test_acs_medicaid_keyed_uniform(seed, key, stream) -> None:
    _assert_same_float(
        acs_local_ssi_medicaid_take_up._keyed_uniform(key, seed=seed, stream=stream),
        _main_medicaid_keyed_uniform(key, seed=seed, stream=stream),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("keys", _ACS_KEYS)
def test_acs_head_start_uniforms(seed, keys) -> None:
    _assert_same_array(
        acs_local_vehicles_head_start._keyed_uniforms(keys, seed=seed),
        _main_head_start_keyed_uniforms(keys, seed=seed),
    )


def test_importers_reach_the_same_helpers() -> None:
    """The private names other modules import resolve to the helpers above."""
    assert snap_state_take_up._stable_unit_draws is take_up._stable_unit_draws
    assert acs_local_take_up._stable_unit_draws is take_up._stable_unit_draws
    assert (
        acs_local_take_up._stable_person_draws
        is snap_discretionary_exemption._stable_person_draws
    )
    assert (
        acs_local_ssi_medicaid_take_up._stable_person_draws
        is medicaid_take_up._stable_person_draws
    )
    assert (
        acs_local_ssi_medicaid_take_up._stable_source_draw
        is ssi_take_up._stable_source_draw
    )
