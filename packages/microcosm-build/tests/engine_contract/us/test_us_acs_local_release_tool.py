"""The ACS local release tool's engine pass on a lane frame (microcosm #454).

``fill_reviewed_nulls`` applies the nullable-artifact contract with the
installed engine's input names and defaults. It refuses a frame whose ESI
employer premium is null on some rows, which is what an ACS spine pooled
beside a donor used to carry. The lane now transfers and anchors the column
at staging, so the engine pass must run on its frame.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.frame import Frame
from test_support.microcosm_build.us_acs_local_esi_premiums import (
    EMPLOYER,
    build_lane,
    dense_donor,
    donor_rows,
    raw_acs,
)
from test_support.microcosm_build.us_acs_local_release_tool import (
    load_staging_builder_module,
    load_tool_module,
)

#: An input the donor release carries and no transfer in this fixture gives
#: the ACS rows: the kind of reviewed null the engine pass exists to fill.
_REVIEWED_NULL = "pre_subsidy_rent"


def _with_person(frame: Frame, person) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


@pytest.fixture(scope="module")
def staged() -> Frame:
    """A staged lane frame with one reviewed engine-input null on ACS rows."""

    frame = build_lane(dense_donor(120), raw_acs(100), n_estimators=5).frame
    person = frame.table("person").copy()
    person[_REVIEWED_NULL] = np.where(donor_rows(frame), 9_600.0, np.nan)
    return _with_person(frame, person)


def _summary(tmp_path, frame: Frame):
    """The staging summary's null register, as the staging tool audits it."""

    audit = load_staging_builder_module()._legacy._engine_input_null_audit(frame)
    path = tmp_path / "acs_multispine_staging.summary.json"
    path.write_text(json.dumps({"reviewed_engine_input_nulls": audit}))
    return path, audit


def test_the_engine_pass_runs_on_the_lane_frame_and_leaves_the_premium(
    tmp_path, staged
) -> None:
    tool = load_tool_module()
    summary_path, audit = _summary(tmp_path, staged)
    # Every row carries the premium, so the staging audit registers no null
    # for it: there is nothing for the engine pass to fill or to refuse.
    assert EMPLOYER not in {entry["column"] for entry in audit}
    assert {entry["column"] for entry in audit} == {_REVIEWED_NULL}
    before = staged.table("person")[EMPLOYER].to_numpy(copy=True)

    projected, _dropped = tool.project_input_only(staged)
    fills, drift = tool.fill_reviewed_nulls(projected, summary_path)

    person = projected.table("person")
    assert [fill["column"] for fill in fills] == [_REVIEWED_NULL]
    assert fills[0]["filled_rows"] == 100
    assert fills[0]["missing_rows_by_spine"] == {"acs_2024_1yr": 100}
    assert drift == []
    assert person[_REVIEWED_NULL].notna().all()
    # The engine knows the premium as an input and the pass did not touch it.
    assert EMPLOYER in person.columns
    np.testing.assert_array_equal(person[EMPLOYER].to_numpy(), before)
    assert (person.loc[~donor_rows(staged), EMPLOYER] > 0).any()


def test_the_engine_pass_still_refuses_an_untransferred_acs_spine(
    tmp_path, staged
) -> None:
    # A staging H5 built before the lane transferred the column: null on ACS
    # rows, and registered as a reviewed null by that build's audit.
    tool = load_tool_module()
    person = staged.table("person").copy()
    person.loc[~donor_rows(staged), EMPLOYER] = np.nan
    stale = _with_person(staged, person)
    summary_path, audit = _summary(tmp_path, stale)
    assert EMPLOYER in {entry["column"] for entry in audit}

    with pytest.raises(SourceRuntimeError, match="never filled") as error:
        tool.fill_reviewed_nulls(stale, summary_path)

    assert f"'{EMPLOYER}': 100" in str(error.value)
    # Refused before any fill: the registered null was not defaulted to zero.
    assert stale.table("person")[EMPLOYER].isna().sum() == 100
