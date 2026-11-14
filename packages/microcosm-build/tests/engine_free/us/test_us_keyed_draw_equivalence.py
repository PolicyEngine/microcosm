"""The US keyed draws are ``stable_identity_uniforms``, bit for bit (#1585).

Eleven US modules carried a copy of ``_main_loop``, as main had it at
90e4d94e7. Each now passes the keys it builds, with unchanged code, to the
shared helper. These tests hold the helper to the loop, and each module's
draw to the loop on the module's own keys and salt, in the container the
module returned. Every caller passes an integer seed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_local_ssi_disability as acs_local_ssi_disability
import microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up as acs_local_ssi_medicaid_take_up
import microcosm.build.us_runtime.acs_local_vehicles_head_start as acs_local_vehicles_head_start
import microcosm.build.us_runtime.immigration as immigration
import microcosm.build.us_runtime.medicaid_take_up as medicaid_take_up
import microcosm.build.us_runtime.pregnancy as pregnancy
import microcosm.build.us_runtime.snap_discretionary_exemption as snap_discretionary_exemption
import microcosm.build.us_runtime.snap_take_up as snap_take_up
import microcosm.build.us_runtime.source_runtime as source_runtime
import microcosm.build.us_runtime.ssi_take_up as ssi_take_up
import microcosm.build.us_runtime.take_up as take_up
import microcosm.build.us_runtime.wic_claim as wic_claim
from microcosm.build.stochastic_assignment import stable_identity_uniforms

SEEDS = (0, 578, 2**63 - 1, np.int64(578))


def _main_loop(ids, *, seed, salt):
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


def _bits(values) -> list[int]:
    return np.asarray(values, dtype=np.float64).view(np.uint64).tolist()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "ids", ([], [0, 7, -5], [1001.0, 2.5], ["a", "2024:10:2", ""], np.arange(3))
)
def test_shared_helper_is_the_copied_loop(seed, ids) -> None:
    salt = "acs_local_ssi_disability_criteria:sign"
    assert _bits(stable_identity_uniforms(ids, seed=seed, salt=salt)) == _bits(
        _main_loop(ids, seed=seed, salt=salt)
    )


_PERSONS = pd.DataFrame({"person_id": [5, 1, 3]})
_KEYS = ["5", "1", "3"]

#: Each module's draw on simple keys, the salt main's copy used, and the keys
#: the module builds from them.
_DRAWS = {
    "take_up": (
        lambda seed: take_up._stable_unit_draws(
            pd.DataFrame({"spm_unit_id": [5, 1, 3]}),
            id_column="spm_unit_id",
            seed=seed,
            variable="takes_up_tanf_if_eligible",
        ),
        "takes_up_tanf_if_eligible",
        _KEYS,
    ),
    "medicaid_take_up": (
        lambda seed: medicaid_take_up._stable_person_draws(_PERSONS, seed=seed),
        "takes_up_medicaid_if_eligible",
        _KEYS,
    ),
    "snap_take_up": (
        lambda seed: snap_take_up._stable_unit_draws(
            pd.DataFrame({"person_spm_unit_id": [5, 1, 3]}), seed=seed
        ),
        "snap_take_up",
        _KEYS,
    ),
    "snap_discretionary_exemption": (
        lambda seed: snap_discretionary_exemption._stable_person_draws(
            _PERSONS, seed=seed
        ),
        "snap_abawd_discretionary_exemption",
        _KEYS,
    ),
    "pregnancy": (
        lambda seed: pregnancy._stable_person_draws(_PERSONS, seed=seed),
        "pregnancy",
        _KEYS,
    ),
    "wic_claim": (
        lambda seed: wic_claim._stable_person_draws(_PERSONS, seed=seed),
        "would_claim_wic",
        ["person:5", "person:1", "person:3"],
    ),
    "immigration": (
        lambda seed: immigration._stable_person_draws(
            _PERSONS, seed=seed, salt="immigration:ead_workers"
        ),
        "immigration:ead_workers",
        _KEYS,
    ),
    "acs_local_ssi_disability": (
        lambda seed: acs_local_ssi_disability._keyed_uniforms(
            _KEYS, seed=seed, stream="sign"
        ),
        "acs_local_ssi_disability_criteria:sign",
        _KEYS,
    ),
    "acs_local_vehicles_head_start": (
        lambda seed: acs_local_vehicles_head_start._keyed_uniforms(_KEYS, seed=seed),
        "acs_local_head_start_take_up",
        _KEYS,
    ),
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("module", sorted(_DRAWS))
def test_module_draws_are_the_copied_loop(seed, module) -> None:
    draw, salt, keys = _DRAWS[module]
    actual = draw(seed)
    assert type(actual) is np.ndarray
    assert _bits(actual) == _bits(_main_loop(keys, seed=seed, salt=salt))


@pytest.mark.parametrize("seed", SEEDS)
def test_one_key_draws_stay_python_floats(seed) -> None:
    for actual, salt in (
        (
            ssi_take_up._stable_source_draw("asec:1", seed=seed),
            "takes_up_ssi_if_eligible",
        ),
        (
            acs_local_ssi_medicaid_take_up._keyed_uniform(
                "asec:1", seed=seed, stream="acs_local_medicaid_hins4_anchor_keep"
            ),
            "acs_local_medicaid_hins4_anchor_keep",
        ),
    ):
        assert type(actual) is float
        assert _bits([actual]) == _bits(_main_loop(["asec:1"], seed=seed, salt=salt))


@pytest.mark.parametrize("seed", SEEDS)
def test_source_runtime_keys_raw_ids_on_the_frames_index(seed) -> None:
    """The raw ``tax_unit_id`` values (``1001.0`` formats as itself), or the
    positional index without them, and a Series on the frame's index."""
    salt = "aca:takes_up_aca_if_eligible"
    for frame, keys in (
        (pd.DataFrame({"tax_unit_id": [1001.0, 2.0]}, index=[10, 20]), [1001.0, 2.0]),
        (pd.DataFrame({"x": [1.0, 2.0]}, index=[7, 3]), [7, 3]),
    ):
        actual = source_runtime._stable_draws(frame, seed=seed, salt=salt)
        assert type(actual) is pd.Series
        pd.testing.assert_index_equal(actual.index, frame.index)
        assert _bits(actual.to_numpy()) == _bits(_main_loop(keys, seed=seed, salt=salt))
