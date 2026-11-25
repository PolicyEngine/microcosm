"""ACS local lane: voluntary filing of ACS tax units (microcosm#1720)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_local_voluntary_filing
from microcosm.build.us_runtime.acs_local_voluntary_filing import (
    ACS_LOCAL_VOLUNTARY_FILING_COLUMN,
    ACS_LOCAL_VOLUNTARY_FILING_ISSUE,
    ACS_LOCAL_VOLUNTARY_FILING_METHOD,
    load_acs_local_voluntary_filing_donor,
    with_acs_local_voluntary_filing,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    impute_us_voluntary_filing,
    impute_us_voluntary_filing_from_tables,
)
from microcosm.frame import US_SCHEMA, Frame
from test_support.microcosm_build.us_voluntary_filing import _donor, _frame

_OUTPUT = ACS_LOCAL_VOLUNTARY_FILING_COLUMN
_SEED = 31
_TREES = 8


def test_the_table_entry_point_is_the_donors_imputation() -> None:
    """The donor function now reads its two tables through
    impute_us_voluntary_filing_from_tables; the donor's values are unchanged."""

    frame = _frame(14)
    donor = _donor(120)
    from_frame = impute_us_voluntary_filing(
        frame, donor, seed=_SEED, n_estimators=_TREES
    )
    from_tables = impute_us_voluntary_filing_from_tables(
        frame.table("person"),
        frame.table("tax_unit"),
        donor,
        seed=_SEED,
        n_estimators=_TREES,
    )
    pd.testing.assert_series_equal(from_frame, from_tables)


def _pooled(*, acs_stored: dict[int, object] | None = None) -> Frame:
    """Households 1-6 are donor rows with stored values, 7-12 ACS rows."""

    frame = _frame(12)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    tax_unit = tables["tax_unit"]
    person[spine_column("person")] = np.where(
        person["person_household_id"] <= 6, ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE
    )
    acs_unit = tax_unit["tax_unit_id"] > 106
    tax_unit[spine_column("tax_unit")] = np.where(
        acs_unit, ACS_2024_1YR_SPINE, ASEC_PUF_DONOR_SPINE
    )
    values = pd.Series(
        [True, False, True, False, True, False] + [np.nan] * 6, dtype=object
    )
    for unit_id, value in (acs_stored or {}).items():
        values.iloc[int(np.flatnonzero(tax_unit["tax_unit_id"] == unit_id)[0])] = value
    tax_unit[_OUTPUT] = values
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )


def _acs_only(frame: Frame) -> Frame:
    """The ACS rows as their own frame, without origin tags."""

    tables = {}
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        if tag in table:
            table = table.loc[table[tag].eq(ACS_2024_1YR_SPINE)].drop(columns=[tag])
        elif entity == "household":
            table = table.loc[table["household_id"] > 6]
        else:
            ids = table[f"{entity}_id"]
            table = table.loc[ids % 100 > 6]
        tables[entity] = table.reset_index(drop=True)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": frame.weights_for("household").__class__(
                frame.weights_for("household").values[6:],
                frame.weights_for("household").kind,
            )
        },
    )


def test_the_acs_units_get_the_donors_model_and_the_donor_units_stand() -> None:
    frame = _pooled()
    donor = _donor(120)
    result, receipt = with_acs_local_voluntary_filing(
        frame,
        sipp_donor=donor,
        seed=_SEED,
        donor_identity={"sha256": "5" * 64},
        n_estimators=_TREES,
    )
    after = result.table("tax_unit")
    assert after[_OUTPUT].dtype == bool
    assert after[_OUTPUT].iloc[:6].tolist() == [True, False, True, False, True, False]
    # The ACS fill is exactly the donor function on the ACS rows alone.
    expected = impute_us_voluntary_filing(
        _acs_only(frame), donor, seed=_SEED, n_estimators=_TREES
    )
    assert after[_OUTPUT].iloc[6:].tolist() == expected.tolist()

    assert receipt["issue"] == ACS_LOCAL_VOLUNTARY_FILING_ISSUE == "microcosm#1720"
    assert receipt["method"] == ACS_LOCAL_VOLUNTARY_FILING_METHOD
    assert receipt["column"] == _OUTPUT
    assert receipt["acs_tax_units"] == 6
    assert receipt["filled_units"] == 6
    assert receipt["preserved_acs_units"] == 0
    assert receipt["unfilled_acs_units"] == 0
    assert receipt["model"]["fitted"] is True
    assert receipt["model"]["n_estimators"] == _TREES
    assert receipt["model"]["training_units"] == 120
    assert receipt["sipp_donor"]["sha256"] == "5" * 64
    assert receipt["outcome"]["acs_true_units"] == int(expected.sum())
    assert 0.0 <= receipt["outcome"]["acs_weighted_true_share"] <= 1.0
    donor_weights = np.linspace(1.0, 2.0, 12)[:6]
    assert receipt["outcome"]["donor_weighted_true_share"] == pytest.approx(
        donor_weights[[0, 2, 4]].sum() / donor_weights.sum()
    )
    json.dumps(receipt, allow_nan=False)


def test_stored_acs_values_stand_and_a_complete_column_is_untouched() -> None:
    frame = _pooled(acs_stored={107: True})
    result, receipt = with_acs_local_voluntary_filing(
        frame, sipp_donor=_donor(120), seed=_SEED, n_estimators=_TREES
    )
    assert result.table("tax_unit")[_OUTPUT].iloc[6] == True  # noqa: E712
    assert receipt["filled_units"] == 5
    assert receipt["preserved_acs_units"] == 1

    complete = _pooled(acs_stored={unit: False for unit in range(107, 113)})
    same, receipt = with_acs_local_voluntary_filing(
        complete, sipp_donor=_donor(120), seed=_SEED, n_estimators=_TREES
    )
    assert same is complete
    assert receipt["model"]["fitted"] is False
    assert receipt["filled_units"] == 0


def test_a_wage_blank_from_15_and_missing_inputs_are_refused() -> None:
    frame = _pooled()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    acs_adult = person.index[
        person[spine_column("person")].eq(ACS_2024_1YR_SPINE) & person["age"].ge(15)
    ][0]
    person.loc[acs_adult, "employment_income_before_lsr"] = np.nan
    blanked = Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})
    with pytest.raises(ValueError, match=r"aged 15\+ have a blank"):
        with_acs_local_voluntary_filing(blanked, sipp_donor=_donor(), seed=_SEED)

    # A blank under 15 reads as no wages.
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    acs_child = person.index[
        person[spine_column("person")].eq(ACS_2024_1YR_SPINE) & person["age"].lt(15)
    ][0]
    person.loc[acs_child, "employment_income_before_lsr"] = np.nan
    _, receipt = with_acs_local_voluntary_filing(
        Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")}),
        sipp_donor=_donor(120),
        seed=_SEED,
        n_estimators=_TREES,
    )
    assert receipt["predictor_view"]["wage_blank_under_15_rows"] == 1

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"] = tables["tax_unit"].drop(columns=["filing_status_input"])
    with pytest.raises(ValueError, match="filing_status_input"):
        with_acs_local_voluntary_filing(
            Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")}),
            sipp_donor=_donor(),
            seed=_SEED,
        )


def test_origin_tags_and_the_donor_column_are_required() -> None:
    frame = _pooled()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"] = tables["tax_unit"].drop(columns=[spine_column("tax_unit")])
    with pytest.raises(ValueError, match="origin tags"):
        with_acs_local_voluntary_filing(
            Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")}),
            sipp_donor=_donor(),
            seed=_SEED,
        )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"] = tables["tax_unit"].drop(columns=[_OUTPUT])
    with pytest.raises(ValueError, match="donor release carries it"):
        with_acs_local_voluntary_filing(
            Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")}),
            sipp_donor=_donor(),
            seed=_SEED,
        )


def test_the_loader_verifies_the_pinned_file_it_is_given(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return _donor(4)

    monkeypatch.setattr(
        acs_local_voluntary_filing, "load_sipp_2023_voluntary_filing_donor", fake_load
    )
    source = tmp_path / "pu2023.csv"
    donor, identity = load_acs_local_voluntary_filing_donor(source)
    assert len(donor) == 4
    assert calls == {
        "path": Path(source),
        "expected_sha256": SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        "expected_size_bytes": SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    }
    assert identity["path"] == str(source.resolve())
    assert identity["sha256"] == SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
