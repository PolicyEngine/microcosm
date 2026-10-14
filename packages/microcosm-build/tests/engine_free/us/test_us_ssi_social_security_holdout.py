"""Held-out SSI/Social Security overlap reporting without a country engine."""

import json

import numpy as np
import pytest

from microcosm.build.us_runtime.ssi_social_security_holdout import (
    SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9,
    SSA_SSI_SOCIAL_SECURITY_SOURCE_URL,
    ssi_social_security_holdout_payload,
    write_ssi_social_security_holdout,
)


def test_ssa_table_9_counts_verify_the_rounded_issue_comparators():
    working_age = SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9["18_64"]
    older = SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9["65_plus"]
    assert working_age.concurrent_recipients == 1_080_790
    assert working_age.share == pytest.approx(1_080_790 / 3_951_866)
    assert older.concurrent_recipients == 1_410_664
    assert older.share == pytest.approx(1_410_664 / 2_469_103)
    assert round(100 * working_age.share) == 27
    assert round(100 * older.share) == 57
    with pytest.raises(TypeError):
        SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9["new_band"] = older


def test_overlap_uses_household_person_weights_and_the_ssi_denominator():
    payload = ssi_social_security_holdout_payload(
        np.array([17, 18, 64, 65, 90, 40, 70]),
        np.array([100, 100, 100, 100, 100, 0, 0]),
        np.array([100, 100, 0, 0, 100, 100, 100]),
        np.array([999, 2, 8, 3, 7, 999, 999]),
        period=2024,
        release_id="fixture",
    )
    working_age, older = payload["age_bands"]
    assert working_age["weighted_ssi_recipients"] == 10
    assert working_age["weighted_concurrent_recipients"] == 2
    assert working_age["share"] == 0.2
    assert older["weighted_ssi_recipients"] == 10
    assert older["weighted_concurrent_recipients"] == 7
    assert older["share"] == 0.7
    assert older["difference_percentage_points"] == pytest.approx(
        100 * (0.7 - older["benchmark_share"])
    )
    assert payload["release_id"] == "fixture"
    assert payload["benchmark"]["source_url"] == SSA_SSI_SOCIAL_SECURITY_SOURCE_URL
    assert payload["benchmark"]["table"] == 9
    assert payload["benchmark"]["year"] == 2024
    assert payload["enforced"] is False
    assert payload["used_for_calibration"] is False
    assert payload["target_role"] == "validation"
    assert "December 2024" in payload["comparability"]
    assert "annual" in payload["comparability"]
    assert "gate" not in payload


def test_no_weighted_ssi_recipients_reports_null_share(tmp_path):
    payload = ssi_social_security_holdout_payload(
        np.array([20, 70]),
        np.array([0, 100]),
        np.array([100, 100]),
        np.array([1, 0]),
        period=2025,
    )
    for row in payload["age_bands"]:
        assert row["weighted_ssi_recipients"] == 0
        assert row["weighted_concurrent_recipients"] == 0
        assert row["share"] is None
        assert row["difference_percentage_points"] is None
    path = write_ssi_social_security_holdout(payload, tmp_path / "diagnostic.json")
    assert json.loads(path.read_text()) == payload


def test_misaligned_person_arrays_are_not_silently_broadcast():
    with pytest.raises(ValueError, match="aligned"):
        ssi_social_security_holdout_payload(
            np.array([20, 70]),
            np.array([100]),
            np.array([100, 100]),
            np.array([1, 1]),
            period=2024,
        )
