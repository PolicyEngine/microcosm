"""The pre-R2 county AGI check against R2's SOI AGI measure (microcosm#922).

``tools/check_us_acs_county_agi_support.py`` computes each household's AGI
with R2's projection, reviewed-null fill and dataset builder, in household
chunks. The engine counts only the positive income sources of a tax unit's
head and spouse (``irs_gross_income``), and only an adult can be a head or a
spouse, so income held by a dependent, or by a tax unit without an adult,
adds nothing. This test runs the check's engine pass on six households and
compares it with the production SOI materializer's AGI measure, the value
R2's state AGI row stores (#1660 review).
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
    _soi,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
#: Household AGI: a dependent's wages (1), a head's wages (2), a retiree's
#: interest beside a dependent's wages (3), a spouse's wages (4), a tax unit
#: of one 16-year-old with wages (5), and a capital loss alone (6).
_EXPECTED_AGI = [0.0, 30_000.0, 100.0, 20_000.0, 0.0, -2_000.0]


def _check():
    path = _TEST_PATHS.repository / "tools" / "check_us_acs_county_agi_support.py"
    spec = importlib.util.spec_from_file_location("check_us_county_agi_engine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(*, with_nulls: bool) -> Frame:
    household = np.asarray([1, 1, 2, 3, 3, 4, 4, 5, 6, 6], "int64")
    tax_unit = np.asarray([10, 10, 20, 30, 30, 40, 40, 50, 60, 60], "int64")
    interest = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if with_nulls:
        # A registered null: the fill step gives it the engine default, 0.
        interest[0] = np.nan
    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 11, dtype="int64"),
            "person_household_id": household,
            "person_tax_unit_id": tax_unit,
            "person_spm_unit_id": tax_unit * 10,
            "person_family_id": tax_unit * 100,
            "person_marital_unit_id": np.asarray(
                [1, 2, 3, 4, 5, 6, 6, 7, 8, 9], "int64"
            )
            * 10_000,
            "age": np.asarray(
                [40.0, 10.0, 40.0, 70.0, 16.0, 40.0, 38.0, 16.0, 50.0, 12.0]
            ),
            "employment_income_before_lsr": np.asarray(
                [0.0, 5_000.0, 30_000.0, 0.0, 8_000.0, 0.0, 20_000.0, 6_000.0, 0.0, 0.0]
            ),
            "taxable_interest_income": np.asarray(interest),
            "short_term_capital_gains": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2_000.0, 0.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    if with_nulls:
        # A runtime-owned take-up flag that R2's stages fill on ACS rows and
        # the AGI does not read: the check holds it back while its nulls are
        # on ACS rows only.
        tables["spm_unit"]["takes_up_snap_if_eligible"] = pd.Series(
            [True, None, True, True, True, True], dtype=object
        )
        tables["spm_unit"]["spm_unit_spine"] = "acs_2024_1yr"
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.arange(1, 7, dtype="int64"),
            "state_fips": np.full(6, 6, dtype="int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(6), WeightKind.DESIGN)},
    )


def test_no_column_an_r2_stage_fills_is_an_agi_input() -> None:
    """The check holds back a null stage-filled column on ACS rows because
    the AGI does not read it; an engine release whose AGI read one would make
    the check refuse a staging file R2 accepts (review of #1660)."""

    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    check = _check()
    release = check._release_tool()
    inputs = check.agi_engine_inputs(PolicyEngineUSVariableMetadataIndex())
    stage_filled = check.reviewed_fills.stage_filled_never_default(release)

    assert "employment_income_before_lsr" in inputs
    assert stage_filled
    assert inputs.isdisjoint(column for _entity, column in stage_filled)


def test_household_agi_counts_only_heads_and_spouses_as_the_soi_row_does(
    tmp_path,
) -> None:
    check = _check()
    release = check._release_tool()
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {
                        "entity": "person",
                        "column": "taxable_interest_income",
                        "missing_rows": 1,
                        "rows": 10,
                    }
                ]
            }
        )
    )

    agi, receipt = check.household_agi(
        _frame(with_nulls=True), summary, release=release, hh_chunk=2
    )

    np.testing.assert_array_equal(agi, _EXPECTED_AGI)
    assert receipt["chunks"] == 3
    assert receipt["reviewed_fill_columns"] == 1
    assert receipt["null_runtime_columns_held_back"] == {
        "spm_unit": ["takes_up_snap_if_eligible"]
    }

    builder = _load_builder_module()
    target_frame, _registry, compilation = builder._materialize_target_frame(
        _frame(with_nulls=False),
        (_soi("agi_amount", "adjusted_gross_income"),),
        maximum_microsim_batch_size=None,
    )
    assert compilation["dropped_target_names"] == []
    np.testing.assert_array_equal(
        target_frame.table("household")["agi_amount"].to_numpy(dtype=np.float64), agi
    )
