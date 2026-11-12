"""UK spine sampling (#627): the root's rung sampler, on invented households.

The spine samples its FRS root at a rung below f100; the dense line samples
nothing of its own and takes the bound spine's sampling (microcosm#931), so
its family sampler is retired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import national_sampling
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    sample_uk_spine_root,
)
from microcosm.frame import Frame


def _root_frame(households: int = 200) -> Frame:
    """An invented FRS root of ``households`` one-person households."""

    ids = np.arange(1, households + 1, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": pd.array(
                ["LONDON" if value % 2 else "WALES" for value in ids], dtype="string"
            ),
            "household_weight": np.full(households, 10.0),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10,
            "person_benunit_id": ids * 100,
            "person_household_id": ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids * 100})
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2023"
    )


def test_the_root_is_returned_as_it_is_at_f100() -> None:
    frame = _root_frame()
    sampled, receipt = sample_uk_spine_root(frame, fraction=1.0, seed=19)
    assert sampled is frame
    assert receipt is None


@pytest.mark.parametrize("fraction,rung", [(0.01, "f001"), (0.10, "f010")])
def test_the_root_sample_keeps_whole_households_and_renormalizes_mass(
    fraction: float, rung: str
) -> None:
    frame = _root_frame()
    full_mass = float(frame.weights_for("household").total)
    sampled, receipt = sample_uk_spine_root(frame, fraction=fraction, seed=19)
    kept = set(sampled.table("household")["household_id"].tolist())
    assert kept < set(frame.table("household")["household_id"].tolist())
    # Every member of a kept household is kept with it.
    assert set(sampled.table("person")["person_household_id"].tolist()) == kept
    assert float(sampled.weights_for("household").total) == pytest.approx(full_mass)
    assert receipt["fraction"] == fraction
    assert receipt["seed"] == 19
    assert receipt["rung_token"] == rung == UK_SAMPLE_RUNG_TOKENS[fraction]
    assert receipt["pre_household_count"] == 200
    assert receipt["post_household_count"] == len(kept)
    assert receipt["normalization_factor"] > 0.0


def test_the_root_sample_is_reproducible_from_its_seed() -> None:
    first, _ = sample_uk_spine_root(_root_frame(), fraction=0.10, seed=7)
    again, _ = sample_uk_spine_root(_root_frame(), fraction=0.10, seed=7)
    pd.testing.assert_frame_equal(first.table("household"), again.table("household"))


def test_the_dense_family_sampler_is_retired() -> None:
    """The dense line takes its spine's sampling (microcosm#931): the module
    holds the root's sampler, the rung tokens and the default seed alone."""

    assert set(national_sampling.__all__) == {
        "UK_SAMPLE_RUNG_TOKENS",
        "UK_SAMPLE_SEED_DEFAULT",
        "UKSampledSpineRootTransform",
        "sample_uk_spine_root",
    }
    for retired in ("sample_uk_spine_frame", "uk_spine_source_family_units"):
        assert not hasattr(national_sampling, retired)
