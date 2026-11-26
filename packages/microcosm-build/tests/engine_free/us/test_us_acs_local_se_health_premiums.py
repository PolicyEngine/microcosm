"""ACS local lane: self-employed health insurance inputs (microcosm#1720)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_local_se_health_premiums import (
    ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS,
    ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE,
    ACS_LOCAL_SE_HEALTH_PREMIUMS_METHOD,
    ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS,
    with_acs_local_se_health_premiums,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.other_health_insurance import (
    attribute_us_se_health_premiums,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

#: Eight ACS persons after the shared transfer, one per household:
#: 0 self-employed, premium, no ESI       -> flagged, premium kept
#: 1 self-employed via the SSTB leaf only -> flagged, premium kept
#: 2 self-employed but 70 (Medicare proxy) -> flagged, premium withheld
#: 3 self-employed with SS disability     -> flagged, premium withheld
#: 4 self-employed with ESI               -> flagged, premium withheld
#: 5 Schedule C loss                      -> not flagged
#: 6 a wage earner with a premium         -> not flagged
#: 7 a child of 10 (SEMP blank)           -> not flagged
_PERSONS = {
    "age": [40.0, 35.0, 70.0, 50.0, 45.0, 30.0, 28.0, 10.0],
    "self_employment_income_before_lsr": [
        20_000.0,
        0.0,
        9_000.0,
        5_000.0,
        8_000.0,
        -3_000.0,
        0.0,
        np.nan,
    ],
    "sstb_self_employment_income_before_lsr": [0.0, 4_000.0, 0, 0, 0, 0, 0, 0],
    "health_insurance_premiums_without_medicare_part_b": [
        6_000.0,
        3_000.0,
        2_000.0,
        1_500.0,
        1_200.0,
        900.0,
        4_000.0,
        0.0,
    ],
    "social_security_disability": [0.0, 0.0, 0.0, 7_000.0, 0, 0, 0, 0],
    "has_esi": [False, False, False, False, True, False, True, False],
    "COW": [6.0, 7.0, 6.0, 6.0, 7.0, 6.0, 1.0, np.nan],
}


def _acs_frame(**overrides: Any) -> Frame:
    ids = np.arange(1, 9)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                f"person_{entity}_id": ids
                for entity in US_SCHEMA.entities
                if entity != "person"
            },
            **_PERSONS,
            **overrides,
        }
    )
    return Frame(
        {
            "person": person,
            **{
                entity: pd.DataFrame({f"{entity}_id": ids})
                for entity in US_SCHEMA.entities
                if entity != "person"
            },
        },
        US_SCHEMA,
        {"household": Weights(np.arange(1.0, 9.0) * 10.0, WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def test_the_acs_rows_get_the_donors_attribution() -> None:
    frame = _acs_frame()
    result, receipt = with_acs_local_se_health_premiums(frame)
    person = result.table("person")

    assert person["is_self_employed"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert person["health_insurance_premiums"].tolist() == [
        6_000.0,
        3_000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert person["is_self_employed"].dtype == bool
    # The input frame is untouched.
    assert "is_self_employed" not in frame.table("person")

    assert receipt["issue"] == ACS_LOCAL_SE_HEALTH_PREMIUMS_ISSUE == "microcosm#1720"
    assert receipt["method"] == ACS_LOCAL_SE_HEALTH_PREMIUMS_METHOD
    assert receipt["source_columns"] == list(
        ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS
    )
    assert receipt["acs_persons"] == 8
    assert receipt["self_employment_blank_under_15_read_as_zero_rows"] == 1
    assert receipt["columns"] == {
        "health_insurance_premiums": {"filled_rows": 8, "preserved_rows": 0},
        "is_self_employed": {"filled_rows": 8, "preserved_rows": 0},
    }
    assert receipt["self_employed_rows"] == 5
    assert receipt["self_employed_weighted_share"] == pytest.approx(150.0 / 360.0)
    assert receipt["premium_positive_rows"] == 2
    assert receipt["premium_weighted_total"] == pytest.approx(
        10 * 6_000.0 + 20 * 3_000.0
    )
    assert receipt["withheld"] == {
        "self_employed_with_reported_premium_rows": 5,
        "medicare_proxy_rows": 2,
        "employer_coverage_rows": 1,
    }
    # COW 6/7 against the Schedule C flag: the loss (COW 6) is cow-only.
    assert receipt["cow_agreement"] == {
        "loaded": True,
        "cow_codes": [6, 7],
        "both_rows": 5,
        "flag_only_rows": 0,
        "cow_only_rows": 1,
        "cow_blank_rows": 1,
    }
    json.dumps(receipt, allow_nan=False)


def test_the_stage_is_the_donors_function_on_a_complete_view() -> None:
    """One rule for both sets of rows: on complete inputs the stage's values
    are exactly the donor function's."""

    view = pd.DataFrame(
        {
            column: _PERSONS[column]
            for column in ACS_LOCAL_SE_HEALTH_PREMIUMS_SOURCE_COLUMNS
        }
    ).fillna(0.0)
    expected = attribute_us_se_health_premiums(view)
    result, _ = with_acs_local_se_health_premiums(_acs_frame())
    person = result.table("person")
    for column in ACS_LOCAL_SE_HEALTH_PREMIUM_COLUMNS:
        np.testing.assert_array_equal(
            person[column].to_numpy(), expected[column].to_numpy()
        )


def test_stored_values_stand_and_only_missing_cells_are_filled() -> None:
    stored = [np.nan, 111.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    result, receipt = with_acs_local_se_health_premiums(
        _acs_frame(health_insurance_premiums=stored)
    )
    person = result.table("person")
    assert person["health_insurance_premiums"].tolist()[:2] == [6_000.0, 111.0]
    assert receipt["columns"]["health_insurance_premiums"] == {
        "filled_rows": 7,
        "preserved_rows": 1,
    }


def test_without_cow_the_agreement_is_marked_unloaded() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=["COW"])
    _, receipt = with_acs_local_se_health_premiums(
        Frame(
            tables,
            US_SCHEMA,
            {"household": frame.weights_for("household")},
            frame.strata,
        )
    )
    assert receipt["cow_agreement"] == {"loaded": False}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"self_employment_income_before_lsr": [np.nan, 0, 0, 0, 0, 0, 0, 0]},
            r"aged 15\+ have a blank self_employment_income_before_lsr",
        ),
        (
            {"social_security_disability": [np.nan, 0, 0, 0, 0, 0, 0, 0]},
            "blank transferred social_security_disability",
        ),
        ({"has_esi": [np.nan, 0, 0, 0, 0, 0, 0, 0]}, "complete boolean"),
        (
            {
                "health_insurance_premiums_without_medicare_part_b": [
                    -1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            },
            "microcosm#1720",
        ),
    ],
)
def test_inputs_the_rule_cannot_read_are_refused(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        with_acs_local_se_health_premiums(_acs_frame(**overrides))


def test_a_missing_input_column_is_refused() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=["has_esi"])
    with pytest.raises(ValueError, match=r"\['has_esi'\]"):
        with_acs_local_se_health_premiums(
            Frame(
                tables,
                US_SCHEMA,
                {"household": frame.weights_for("household")},
                frame.strata,
            )
        )


def test_the_seam_attributes_after_the_transfer_and_before_pooling(
    monkeypatch, tmp_path
) -> None:
    events: list[tuple[Any, ...]] = []
    transferred = SimpleNamespace(table=lambda entity: pd.DataFrame())
    attributed = object()
    pooled = object()
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

    def fake_transfer(recipient, donor, **kwargs):
        events.append(("transfer",))
        return SimpleNamespace(
            frame=transferred,
            imputed_inputs=(),
            fit_records=(),
            deferred_inputs=(),
            resolved_donor_channel="puf_support",
        )

    def fake_attribute(frame):
        events.append(("se_health", frame))
        return attributed, {"issue": "microcosm#1720", "share": np.float64(0.1)}

    def fake_pool(actual_base, actual_acs, **kwargs):
        events.append(("pool", actual_acs))
        return pooled

    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", fake_transfer)
    monkeypatch.setattr(
        acs_multispine, "with_acs_local_se_health_premiums", fake_attribute
    )
    monkeypatch.setattr(acs_multispine, "with_optional_acs_spine", fake_pool)

    source = AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")
    result = acs_multispine.build_optional_acs_multispine(
        cast(Frame, base), source, se_health_premiums=True
    )
    assert events == [("transfer",), ("se_health", transferred), ("pool", attributed)]
    assert result.provenance["acs_local_se_health_premiums"] == {
        "issue": "microcosm#1720",
        "share": 0.1,
    }
    json.dumps(result.provenance, allow_nan=False)

    # Without the flag (the pool lane) the stage never runs.
    events.clear()
    result = acs_multispine.build_optional_acs_multispine(cast(Frame, base), source)
    assert events == [("transfer",), ("pool", transferred)]
    assert "acs_local_se_health_premiums" not in result.provenance
