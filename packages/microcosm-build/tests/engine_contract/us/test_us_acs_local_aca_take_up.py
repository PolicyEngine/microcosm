"""The ACS local ACA take-up stage with the real engine (microcosm#1720).

The release tool's pre-pass runs the fiscal lane's batched ACA tax-unit
table under materialize's engine-pass contract, first on the donor
households, then on the ACS households, and the stage runs the fiscal lane's
``aca_marketplace_inputs`` manifest stage on the ACS tax units against what
the donor rows leave of each state count. Engine-free tests cover the
accounting on invented values; this test checks that the real engine's
values flow through it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import Frame
from test_support.microcosm_build.us_acs_local_esi_premiums import (
    build_lane,
    dense_donor,
    raw_acs,
)
from test_support.microcosm_build.us_acs_local_release_tool import (
    load_staging_builder_module,
    load_tool_module,
)

TAKE_UP = "takes_up_aca_if_eligible"
RATIO = "selected_marketplace_plan_benchmark_ratio"
CMS = "cms_aca_aptc_recipients_by_state"
RETURNS = "irs_soi_premium_tax_credit_returns_by_state"
AMOUNT = "irs_soi_premium_tax_credit_amount_by_state"
ACS = "acs_2024_1yr"


@pytest.fixture(scope="module")
def staged() -> Frame:
    """A staged lane frame with the ACA inputs on donor rows only.

    The donor stores both ACA inputs, as a national release does; the ACS
    rows carry them missing, with the ``SERIALNO``/``SPORDER`` keys PUMS
    gives them, and no person reports Marketplace coverage.
    """

    frame = build_lane(dense_donor(120), raw_acs(100), n_estimators=5).frame
    tax_unit = frame.table("tax_unit").copy()
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    acs_units = tax_unit[spine_column("tax_unit")].eq(ACS)
    acs_persons = person[spine_column("person")].eq(ACS).to_numpy()
    acs_households = household[spine_column("household")].eq(ACS)
    rng = np.random.default_rng(0)
    tax_unit[TAKE_UP] = pd.Series(rng.random(len(tax_unit)) < 0.5, dtype=object).where(
        ~acs_units, np.nan
    )
    tax_unit[RATIO] = pd.Series(
        np.where(rng.random(len(tax_unit)) < 0.3, 0.8, 1.0)
    ).where(~acs_units, np.nan)
    person["has_marketplace_health_coverage_at_interview"] = False
    person["SPORDER"] = np.where(acs_persons, 1.0, np.nan)
    household["SERIALNO"] = pd.Series(
        [f"2024HU{value:07d}" for value in household["household_id"]], dtype=object
    ).where(acs_households, np.nan)
    return Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "tax_unit": tax_unit,
            "person": person,
            "household": household,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _summary(tmp_path, frame: Frame):
    audit = load_staging_builder_module()._legacy._engine_input_null_audit(frame)
    path = tmp_path / "acs_multispine_staging.summary.json"
    path.write_text(json.dumps({"reviewed_engine_input_nulls": audit}))
    return path


def _tables(frame: Frame, values: dict[str, float]) -> dict[str, pd.DataFrame]:
    states = sorted({int(code) for code in frame.table("household")["state_fips"]})
    return {
        name: pd.DataFrame(
            {"state_fips": [f"{code:02d}" for code in states], "target": value}
        )
        for name, value in values.items()
    }


def test_the_stage_runs_on_the_real_engine_and_meets_the_soi_residuals(
    tmp_path, staged
) -> None:
    tool = load_tool_module()
    summary = _summary(tmp_path, staged)
    acs = staged.table("tax_unit")[spine_column("tax_unit")].eq(ACS).to_numpy()

    # Counts far above what any set of rows carries: every eligible ACS
    # unit takes up, which measures each state's donor contribution and ACS
    # capacity with the engine.
    huge = _tables(staged, {CMS: 1e9, RETURNS: 1e9, AMOUNT: 1e13})
    _saturated, measured = tool._with_local_aca_take_up(
        staged,
        seed=3,
        targets={"target_tables": huge},
        summary_path=summary,
        hh_chunk=60,
    )
    assert measured["engine_prepass"]["acs_eligible_tax_units"] > 0
    assert measured["engine_prepass"]["donor"]["eligible_tax_units"] > 0
    rows = {
        name: {row["state_fips"]: row for row in entry["states"]}
        for name, entry in measured["tables"].items()
    }

    # Counts that leave each state's ACS rows half their capacity to fill.
    tables = {
        name: pd.DataFrame(
            {
                "state_fips": list(by_state),
                "target": [
                    row["donor"] + 0.5 * row["acs_capacity"]
                    for row in by_state.values()
                ],
            }
        )
        for name, by_state in rows.items()
    }
    # One batch this time: the donor contributions must not depend on it.
    result, receipt = tool._with_local_aca_take_up(
        staged,
        seed=3,
        targets={"target_tables": tables},
        summary_path=summary,
        hh_chunk=1_000,
    )

    assert receipt["binding_table"] == RETURNS
    assert receipt["stop_after"] == "support_clip"
    assert receipt["gate"]["passed"], receipt["gate"]["failures"]
    assert receipt["gate"]["report_only"] is True
    for row in receipt["tables"][RETURNS]["states"]:
        if row["acs_capacity"] <= 0:
            continue
        assert row["donor"] == pytest.approx(rows[RETURNS][row["state_fips"]]["donor"])
        assert row["residual"] <= row["acs_delivered"]
        assert row["acs_delivered"] < row["residual"] + row["acs_max_unit_weight"]
    units = result.table("tax_unit")
    assert units.loc[acs, TAKE_UP].notna().all()
    assert units.loc[acs, RATIO].between(0.5, 1.5).all()
    flags = units.loc[acs, TAKE_UP].astype(bool)
    assert flags.any() and not flags.all()
    # Donor cells are the donor's own.
    for column in (TAKE_UP, RATIO):
        assert (
            units.loc[~acs, column].tolist()
            == staged.table("tax_unit").loc[~acs, column].tolist()
        )
    # The recorded assignment reproduces the stage on a fresh staging frame.
    path = tmp_path / tool.ACS_ACA_TAKE_UP_FILENAME
    receipt["assignment_file"] = {
        "name": path.name,
        "sha256": tool._write_aca_take_up(result, path),
    }
    applied = tool._with_recorded_aca_take_up(staged, receipt, path)
    for column in (TAKE_UP, RATIO):
        assert applied.table("tax_unit")[column].tolist() == units[column].tolist()
