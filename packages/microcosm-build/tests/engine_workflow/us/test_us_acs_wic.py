"""Persist ACS decisions and confirm the country engine consumes them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    load_legacy_calibrated_us_h5,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.wic_claim import (
    require_complete_us_wic_claim_h5,
    with_acs_wic_claim_input,
)
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from test_support.microcosm_build.us_wic_claim import _frame

OUTPUT = "takes_up_wic_if_eligible"


def test_generated_acs_participation_round_trips_and_controls_wic_benefits(tmp_path):
    frame = _frame([{"age": 3}, {"age": 40}] * 16)
    frame = with_acs_wic_claim_input(frame, seed=19, time_period=2024)
    path = tmp_path / "complete-acs.h5"
    write_nullable_us_h5(
        frame, path, period=2024, artifact_kind=LEGACY_NULLABLE_STAGING_ARTIFACT_KIND
    )
    assert require_complete_us_wic_claim_h5(path) == len(frame.person)
    loaded = load_legacy_calibrated_us_h5(path)
    pd.testing.assert_series_equal(frame.person[OUTPUT], loaded.person[OUTPUT])
    assert "would_claim_wic" not in loaded.person
    result = PolicyEngineUSEngine().materialize(
        loaded, variables=["wic", "is_wic_eligible"], period=2024
    )
    claimed = loaded.person[OUTPUT].to_numpy()
    adult = loaded.person.age.eq(40).to_numpy()
    child = ~adult
    assert claimed[child].any() and (~claimed[child]).any()
    assert not result["is_wic_eligible"][adult].any()
    assert not result["wic"][~claimed].any()
    assert (result["wic"][child & claimed] > 0).all()


@pytest.mark.parametrize("hdf_format", ["fixed", "table"])
def test_h5_validation_reads_every_row_in_bounded_batches(
    tmp_path, monkeypatch, hdf_format
):
    path = tmp_path / "multi-batch.h5"
    pd.DataFrame({OUTPUT: np.zeros(65_537, dtype=bool)}).to_hdf(
        path, key="person", format=hdf_format
    )
    reads = []
    original = pd.HDFStore.select

    def record_read(store, key, *args, **kwargs):
        reads.append((kwargs["start"], kwargs["stop"]))
        return original(store, key, *args, **kwargs)

    monkeypatch.setattr(pd.HDFStore, "select", record_read)
    assert require_complete_us_wic_claim_h5(path) == 65_537
    assert reads == [(0, 65_536), (65_536, 131_072)]


@pytest.mark.parametrize("hdf_format", ["fixed", "table"])
@pytest.mark.parametrize("values", [None, [1.0, np.nan], ["False", "True"], [1, 0]])
def test_h5_validation_refuses_missing_or_non_boolean_decisions(
    tmp_path, values, hdf_format
):
    path = tmp_path / "incomplete.h5"
    person = pd.DataFrame({"person_id": [1, 2]})
    if values is not None:
        person[OUTPUT] = values
    person.to_hdf(path, key="person", format=hdf_format)
    with pytest.raises(ValueError, match="WIC participation"):
        require_complete_us_wic_claim_h5(path)
