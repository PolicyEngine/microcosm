"""ACS local lane: housing take-up of group-quarters units (microcosm#975)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_local_group_quarters_housing import (
    ACS_GROUP_QUARTERS_KINDS,
    ACS_LOCAL_GROUP_QUARTERS_HOUSING_COLUMN,
    ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE,
    ACS_LOCAL_GROUP_QUARTERS_HOUSING_METHOD,
    with_acs_local_group_quarters_housing,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import required_acs_transfer_inputs
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_COLUMN = ACS_LOCAL_GROUP_QUARTERS_HOUSING_COLUMN


def _acs_frame(kinds: list[float], take_up: list[object], **household: Any) -> Frame:
    """One person, SPM unit and household per entry, in that order."""

    ids = np.arange(1, len(kinds) + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                f"person_{entity}_id": ids
                for entity in US_SCHEMA.entities
                if entity != "person"
            },
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.entities
            if entity != "person"
        },
    }
    tables["household"]["TYPEHUGQ"] = kinds
    for column, values in household.items():
        tables["household"][column] = values
    tables["spm_unit"][_COLUMN] = pd.Series(take_up, dtype=object)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(len(ids), 10.0), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def test_the_flag_is_still_a_shared_transfer_target() -> None:
    """The step overrides the shared draw; it does not change the plan."""

    assert _COLUMN in required_acs_transfer_inputs()
    assert ACS_GROUP_QUARTERS_KINDS == (2, 3)


def test_group_quarters_units_get_false_and_housing_units_keep_their_draw() -> None:
    frame = _acs_frame(
        kinds=[1, 1, 2, 2, 3, 3, 1],
        take_up=[True, False, True, False, True, np.nan, True],
    )
    result, receipt = with_acs_local_group_quarters_housing(frame)
    assert result.table("spm_unit")[_COLUMN].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert result.table("spm_unit")[_COLUMN].dtype == bool
    # The input frame is untouched.
    assert frame.table("spm_unit")[_COLUMN].iloc[2] is True

    assert receipt["issue"] == ACS_LOCAL_GROUP_QUARTERS_HOUSING_ISSUE == "microcosm#975"
    assert receipt["method"] == ACS_LOCAL_GROUP_QUARTERS_HOUSING_METHOD
    assert receipt["column"] == _COLUMN
    assert receipt["acs_spm_units"] == 7
    assert receipt["group_quarters_spm_units"] == 4
    assert receipt["by_typehugq"] == {
        "2": {"spm_units": 2, "set_false_from_true": 1, "set_false_from_missing": 0},
        "3": {"spm_units": 2, "set_false_from_true": 1, "set_false_from_missing": 1},
    }
    assert receipt["housing_unit_spm_units"] == 3
    assert receipt["housing_unit_take_up_true"] == 2
    assert receipt["unknown_typehugq_spm_units"] == 0
    json.dumps(receipt, allow_nan=False)


def test_an_unknown_kind_keeps_its_draw_and_is_counted() -> None:
    frame = _acs_frame(kinds=[1, np.nan, 3], take_up=[True, True, True])
    result, receipt = with_acs_local_group_quarters_housing(frame)
    assert result.table("spm_unit")[_COLUMN].tolist() == [True, True, False]
    assert receipt["unknown_typehugq_spm_units"] == 1


@pytest.mark.parametrize("missing", ["TYPEHUGQ", _COLUMN])
def test_a_frame_without_the_columns_it_reads_is_refused(missing) -> None:
    frame = _acs_frame(kinds=[1, 2], take_up=[True, True])
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for entity in ("household", "spm_unit"):
        if missing in tables[entity]:
            tables[entity] = tables[entity].drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        with_acs_local_group_quarters_housing(
            Frame(
                tables,
                US_SCHEMA,
                {"household": frame.weights_for("household")},
                frame.strata,
            )
        )


def test_the_seam_runs_the_step_after_the_transfer_and_before_pooling(
    monkeypatch, tmp_path
) -> None:
    events: list[tuple[Any, ...]] = []
    transferred = SimpleNamespace(table=lambda entity: pd.DataFrame())
    stepped = object()
    base = SimpleNamespace(table=lambda entity: pd.DataFrame())

    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda source, *, chunksize: (object(), {"spine": ACS_2024_1YR_SPINE}),
    )
    monkeypatch.setattr(
        acs_multispine,
        "map_acs_native_inputs",
        lambda frame: SimpleNamespace(frame=object(), native_inputs={}),
    )
    monkeypatch.setattr(
        acs_multispine,
        "transfer_acs_inputs",
        lambda recipient, donor, **kwargs: (
            events.append(("transfer",))
            or SimpleNamespace(
                frame=transferred,
                imputed_inputs=(),
                fit_records=(),
                deferred_inputs=(),
                resolved_donor_channel="puf_support",
            )
        ),
    )

    def fake_step(frame):
        events.append(("group_quarters", frame))
        return stepped, {"issue": "microcosm#975", "units": np.int64(2)}

    monkeypatch.setattr(
        acs_multispine, "with_acs_local_group_quarters_housing", fake_step
    )
    monkeypatch.setattr(
        acs_multispine,
        "with_optional_acs_spine",
        lambda actual_base, actual_acs, **kwargs: (
            events.append(("pool", actual_acs)) or object()
        ),
    )
    source = AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")
    result = acs_multispine.build_optional_acs_multispine(
        cast(Frame, base), source, group_quarters_housing=True
    )
    assert events == [
        ("transfer",),
        ("group_quarters", transferred),
        ("pool", stepped),
    ]
    assert result.provenance["acs_local_group_quarters_housing"] == {
        "issue": "microcosm#975",
        "units": 2,
    }

    events.clear()
    result = acs_multispine.build_optional_acs_multispine(cast(Frame, base), source)
    assert events == [("transfer",), ("pool", transferred)]
    assert "acs_local_group_quarters_housing" not in result.provenance
