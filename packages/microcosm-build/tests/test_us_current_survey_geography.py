"""Current geography from actual issuers over pinned invented source members."""

import json
import sys
from fractions import Fraction

import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build.us_runtime import current_survey_geography as geography
from microcosm.build.us_runtime import survey_population_preparation as source
from microcosm.graph.population import dtype_for_token


def _prepared(tmp_path, monkeypatch, *, unknown=False, zero=True):
    arguments = _demographic_arguments(
        tmp_path, monkeypatch, unknown=unknown, zero=zero
    )
    return arguments, source.prepare_authenticated_survey_population(**arguments)


@pytest.mark.parametrize("unknown", (False, True))
def test_source_keys_geography_counts_and_unknownness(tmp_path, monkeypatch, unknown):
    _, prepared = _prepared(tmp_path, monkeypatch, unknown=unknown)
    frame = prepared.checked_view().frame
    before = source._frame_identity(frame)
    result = geography.qualify_current_survey_geography(prepared)
    table, receipt = result.household, json.loads(result.receipt)
    assert tuple(table.columns) == geography.COLUMNS
    assert table.index.tolist() == frame.table("household").household_id.tolist()
    assert all(table[c].dtype == dtype_for_token("string") for c in table)
    assert table.shape == (6, 3)
    assert receipt["households"] == 6
    assert receipt["acs_households"] == receipt["puma_observed_households"] == 4
    assert receipt["asec_households"] == 2
    assert receipt["state_unknown_households"] == int(unknown)
    assert receipt["asec_survey_year"] == 2025
    assert receipt["acs_survey_year"] == receipt["asec_income_year"] == 2024
    assert not receipt["state_is_income_year_residence_claim"]
    assert not receipt["source_admission_issued"]
    assert not receipt["population_admission_issued"]
    assert not receipt["release_eligible"]
    assert len(result.receipt) <= geography.MAX_RECEIPT_BYTES
    assert receipt["projection_sha256"] == geography._projection_digest(table)
    by_origin = {}
    for row in table.itertuples(index=False, name=None):
        key = tuple(json.loads(row[0]))
        assert row[0] == json.dumps(key, separators=(",", ":"))
        by_origin[key] = row[1:]
    assert set(by_origin) == {
        ("acs", 2024, 2024, "2024HU0000001"),
        ("acs", 2024, 2024, "2024HU0000002"),
        ("acs", 2024, 2024, "2024GQ0000001"),
        ("acs", 2024, 2024, "2024GQ0000002"),
        ("asec", 2024, 2025, "00007"),
        ("asec", 2024, 2025, "00008"),
    }
    for key, (state, puma) in by_origin.items():
        if key[0] == "acs":
            assert (state, puma) == ("06", "0612345")
        else:
            assert pd.isna(puma)
            if key[-1] == "00008" and unknown:
                assert pd.isna(state)
            else:
                assert state == ("06" if key[-1] == "00007" else "36")
    # The fixture reverses the original ASEC member rows before actual issuance;
    # the distinct 06/36 observations above therefore exercise the keyed join.
    assert source._frame_identity(frame) == before


def test_sampling_rungs_keep_the_source_draw_key(tmp_path, monkeypatch):
    # Both ASEC records need positive support so the smaller source cell remains
    # admissible regardless of which household the deterministic sample retains.
    arguments, prepared = _prepared(tmp_path, monkeypatch, zero=False)
    full = geography.qualify_current_survey_geography(prepared).household
    original = full.set_index(geography.COLUMNS[0]).sort_index()
    request_path = arguments["source_dir"] / "selection-request.json"
    request = json.loads(request_path.read_bytes())
    request["fraction"] = [2, 3]
    request_path.write_bytes(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    )
    selected = source.prepare_authenticated_survey_population(
        **{**arguments, "fraction": Fraction(2, 3)}
    )
    smaller = geography.qualify_current_survey_geography(selected).household
    assert 0 < len(smaller) < len(full)
    smaller = smaller.set_index(geography.COLUMNS[0]).sort_index()
    pd.testing.assert_frame_equal(
        smaller, original.loc[smaller.index], check_exact=True
    )


def test_detached_output_mutation_does_not_change_source_or_requalification(
    tmp_path, monkeypatch
):
    _, prepared = _prepared(tmp_path, monkeypatch)
    first = geography.qualify_current_survey_geography(prepared)
    expected = first.household.copy(deep=True)
    receipt = first.receipt
    first.household.loc[first.household.index[0], geography.COLUMNS[1]] = "99"
    fresh = geography.qualify_current_survey_geography(prepared)
    pd.testing.assert_frame_equal(fresh.household, expected, check_exact=True)
    assert fresh.receipt == receipt


@pytest.mark.parametrize(
    "change", ("asec_member", "acs_native_puma", "receiving_order", "carried_state")
)
def test_changed_original_or_receiving_population_refuses(
    tmp_path, monkeypatch, change
):
    arguments, prepared = _prepared(tmp_path, monkeypatch)
    state = source._ISSUED[id(prepared)][2]
    if change == "asec_member":
        path = arguments["source_dir"] / "asec" / "hhpub25.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "acs_native_puma":
        table = state.source_frames[0].table("household")
        table.loc[table.index[0], "PUMA"] = "00100"
    elif change == "receiving_order":
        state.frame._tables["household"] = (
            state.frame.table("household").iloc[::-1].copy(deep=True)
        )
    else:
        state.frame.table("household")["state_fips"] = 99
    with pytest.raises(ValueError):
        geography.qualify_current_survey_geography(prepared)


@pytest.mark.parametrize("change", ("source_frame", "returned_projection"))
def test_mutation_on_last_owner_return_cannot_escape(tmp_path, monkeypatch, change):
    _, prepared = _prepared(tmp_path, monkeypatch)
    state = source._ISSUED[id(prepared)][2]
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code
            is source.AuthenticatedSurveyPopulationPreparation._checked.__code__
            and frame.f_back.f_code
            is geography.qualify_current_survey_geography.__code__
            and "result" in frame.f_back.f_locals
            and not fired
        ):
            fired.append(True)
            if change == "source_frame":
                state.frame.table("household")["state_fips"] = 99
            else:
                result = frame.f_back.f_locals["result"]
                result.household.loc[
                    result.household.index[0], geography.COLUMNS[1]
                ] = "99"

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError):
            geography.qualify_current_survey_geography(prepared)
    finally:
        sys.setprofile(previous)
    assert fired == [True]


def test_detached_asec_projection_mutation_on_return_refuses(tmp_path, monkeypatch):
    _, prepared = _prepared(tmp_path, monkeypatch)
    fired = []

    def trace(frame, event, result):
        if (
            event == "return"
            and frame.f_code
            is geography.demographics.qualify_current_asec_demographics.__code__
            and not fired
        ):
            fired.append(True)
            row = result.household.index[result.household.GESTFIPS.eq("06")][0]
            # Internally consistent values must still match the upstream receipt.
            result.household.loc[row, "GESTFIPS"] = "36"
            result.household.loc[row, "state_fips"] = 36

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match="ASEC_PROJECTION_DIGEST"):
            geography.qualify_current_survey_geography(prepared)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
