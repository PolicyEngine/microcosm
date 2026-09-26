"""The eager ``microcosm.build.uk_runtime`` namespace after the HMRC tail retirement."""

from __future__ import annotations

import pytest

from microcosm.build import uk_runtime
from microcosm.build.uk_runtime import frs_hmrc_source

RETIRED_NAMES = (
    "FRS_HMRC_RETAINED_LEAVES_STAGE_NAME",
    "UKFRSHMRCRetainedLeavesResult",
    "UKFRSHMRCRetainedLeavesStageTransform",
    "retain_uk_frs_hmrc_leaves",
    "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE",
)

FRS_HMRC_CONSTANTS = (
    "FRS_HMRC_INCPBEN_COLUMN",
    "FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN",
    "FRS_HMRC_PAY_COLUMN",
    "FRS_HMRC_RETAINED_LEAF_COLUMNS",
    "FRS_HMRC_SRP_REGULAR_CODE5_COLUMN",
    "FRS_HMRC_UBISJA_COLUMN",
)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_retired_candidate_names_are_gone(name: str) -> None:
    assert name not in uk_runtime.__all__
    assert not hasattr(uk_runtime, name)


@pytest.mark.parametrize("name", FRS_HMRC_CONSTANTS)
def test_frs_hmrc_constants_come_from_the_source_module(name: str) -> None:
    assert name in uk_runtime.__all__
    assert getattr(uk_runtime, name) is getattr(frs_hmrc_source, name)


def test_public_names_are_unique_and_resolvable() -> None:
    assert len(uk_runtime.__all__) == len(set(uk_runtime.__all__))
    for name in uk_runtime.__all__:
        assert hasattr(uk_runtime, name), name
