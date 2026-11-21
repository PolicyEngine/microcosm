"""US SNAP state take-up stage tests (microcosm #372)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    US_SNAP_HOUSEHOLDS_TARGET_TABLE,
    US_SNAP_STATE_TAKE_UP_ANCHOR,
    US_SNAP_STATE_TAKE_UP_STAGE,
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SOURCE_MANIFEST,
    us_snap_state_take_up_gate,
    with_us_snap_state_take_up,
)
from microcosm.build.us_runtime.cps_carried import reported_snap_receipt_by_spm_unit
from microcosm.build.us_runtime.one_population_contract import ACS_SUPPORT_CHANNEL
from microcosm.build.us_runtime.snap_state_take_up import (
    US_SNAP_ELIGIBILITY_COLUMN,
    us_snap_source_spm_unit_table,
    with_us_snap_state_take_up_rate,
)
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_one_population_contract import (
    assembled_frame,
    generate_population,
)

TIME_PERIOD = 2024


def _person_table(rows: list[dict]) -> pd.DataFrame:
    baseline = {"SPM_SNAPSUB": 0.0}
    records = []
    for index, row in enumerate(rows):
        record = dict(baseline)
        record.update(row)
        record.setdefault("person_id", index + 1)
        record.setdefault("person_spm_unit_id", index + 1)
        records.append(record)
    return pd.DataFrame(records)


def _us_frame(
    person_rows: list[dict],
    *,
    household_weights: list[float] | None = None,
) -> Frame:
    person = _person_table(person_rows)
    n = len(person)
    person["person_household_id"] = person["person_spm_unit_id"]
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.unique(person["person_spm_unit_id"])}
        ),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    weights = household_weights or [1.0] * len(unique_households)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(weights, dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _state_targets(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_fips": state,
                "target": target,
                "source_record_id": f"usda_snap.fy2024.test.{state}",
            }
            for state, target in rows
        ]
    )


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    """Rebuild ``frame`` with ``person`` as its person table."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


@cache
def _one_population_frame() -> Frame:
    """The one-population contract's CPS-and-ACS frame for seed 0 (d2283)."""

    return assembled_frame(generate_population(0))


def _one_population_snap_frame(acs_subsidy: float) -> Frame:
    """The one-population frame with ``SPM_SNAPSUB`` on every person row.

    The CPS repeats ``SPM_SNAPSUB``, an annual SPM-unit amount, on every
    member's row. A CPS row carries 2,400 when its SPM unit's source id
    divides evenly by three and 0 otherwise, so a clone carries its native's
    amount. An ACS row carries ``acs_subsidy``: ACS records no CPS reported
    subsidy.
    """

    frame = _one_population_frame()
    person = frame.table("person").copy()
    spm_unit = frame.table("spm_unit")
    unit_source = pd.Series(
        spm_unit[support_source_id_column("spm_unit")].to_numpy(),
        index=spm_unit["spm_unit_id"].to_numpy(),
    )
    source = person["person_spm_unit_id"].map(unit_source).to_numpy()
    acs = person[support_channel_column("person")].eq(ACS_SUPPORT_CHANNEL)
    person["SPM_SNAPSUB"] = np.where(
        acs.to_numpy(), acs_subsidy, np.where(source % 3 == 0, 2_400.0, 0.0)
    )
    return _with_person(frame, person)


def _source_table(frame: Frame) -> pd.DataFrame:
    """The state stage's source table, every unit eligible in one state."""

    count = len(frame.table("spm_unit"))
    return us_snap_source_spm_unit_table(
        frame,
        is_snap_eligible=np.ones(count, dtype=bool),
        state_fips=np.full(count, 6),
        seed=0,
    )


class TestManifestDeclaration:
    def test_stage_is_declared_with_calibrated_ops(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_SNAP_STATE_TAKE_UP_STAGE]
        assert tuple(stage.outputs) == (US_SNAP_TAKE_UP_OUTPUT_COLUMN,)
        kinds = [operation.kind for operation in stage.operations]
        assert kinds == [
            "read_table",
            "assign_binary_from_rate",
            "calibrate_binary_assignment",
        ]
        calibrate = stage.operations[-1]
        assert calibrate.parameters["targets"] == [US_SNAP_HOUSEHOLDS_TARGET_TABLE]
        assert calibrate.parameters["preserve_true_anchor"] == (
            US_SNAP_STATE_TAKE_UP_ANCHOR
        )
        assert calibrate.parameters["domain"] == US_SNAP_ELIGIBILITY_COLUMN


class TestSourceTable:
    def test_anchor_derives_from_reported_subsidy(self) -> None:
        frame = _us_frame(
            [{"SPM_SNAPSUB": 2_400.0}, {"SPM_SNAPSUB": 0.0}, {"SPM_SNAPSUB": 0.0}]
        )
        table = us_snap_source_spm_unit_table(
            frame,
            is_snap_eligible=np.asarray([True, True, False]),
            state_fips=np.asarray(["6", "6", "36"]),
            seed=0,
        )
        assert table[US_SNAP_STATE_TAKE_UP_ANCHOR].tolist() == [True, False, False]
        assert table["state_fips"].tolist() == ["06", "06", "36"]
        assert table["spm_unit_weight"].tolist() == [1.0, 1.0, 1.0]

    def test_misaligned_inputs_are_refused(self) -> None:
        frame = _us_frame([{}, {}])
        with pytest.raises(ValueError, match="must align with the spm_unit table"):
            us_snap_source_spm_unit_table(
                frame,
                is_snap_eligible=np.asarray([True]),
                state_fips=np.asarray(["06", "06"]),
                seed=0,
            )

    def test_missing_raw_column_is_refused(self) -> None:
        frame = _us_frame([{}, {}])
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["person"] = tables["person"].drop(columns=["SPM_SNAPSUB"])
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(ValueError, match="SPM_SNAPSUB"):
            us_snap_source_spm_unit_table(
                stripped,
                is_snap_eligible=np.asarray([True, True]),
                state_fips=np.asarray(["06", "06"]),
                seed=0,
            )

    def test_a_null_reported_subsidy_is_not_anchored_on_a_mixed_frame(self) -> None:
        """A null ``SPM_SNAPSUB`` anchors no ACS SPM unit (d2283).

        The one-population join adds ACS rows, which carry no CPS reported
        subsidy, to Route A's CPS rows. The state stage's source table and
        the shared receipt helper both read a null subsidy as 0, so neither
        anchors an ACS SPM unit, and each returns the same output as with
        0.0 in the ACS cells.
        """

        frame = _one_population_snap_frame(np.nan)
        person = frame.table("person")
        spm_unit = frame.table("spm_unit")
        units = set(spm_unit["spm_unit_id"])
        acs = person[support_channel_column("person")].eq(ACS_SUPPORT_CHANNEL)
        acs_units = set(
            spm_unit.loc[
                spm_unit[support_channel_column("spm_unit")].eq(ACS_SUPPORT_CHANNEL),
                "spm_unit_id",
            ]
        )
        # Each ACS SPM unit holds ACS persons only, and each ACS person sits in one.
        assert acs_units
        assert set(person.loc[acs, "person_spm_unit_id"]) == acs_units
        assert acs_units.isdisjoint(person.loc[~acs, "person_spm_unit_id"])
        assert person.loc[acs, "SPM_SNAPSUB"].isna().all()
        positive = set(person.loc[person["SPM_SNAPSUB"].gt(0), "person_spm_unit_id"])
        assert positive
        assert positive < units - acs_units

        table = _source_table(frame)
        receipt = reported_snap_receipt_by_spm_unit(person)
        anchor = pd.Series(
            table[US_SNAP_STATE_TAKE_UP_ANCHOR].to_numpy(),
            index=table["spm_unit_id"].to_numpy(),
        )
        assert sorted(anchor.index) == sorted(units)
        assert sorted(receipt.index) == sorted(units)
        assert [unit for unit in sorted(acs_units) if anchor[unit]] == []
        assert [unit for unit in sorted(acs_units) if receipt[unit]] == []
        # The CPS units' anchors follow their subsidy.
        assert set(anchor.index[anchor.to_numpy()]) == positive
        assert set(receipt.index[receipt.to_numpy()]) == positive

        zero = _one_population_snap_frame(0.0)
        assert zero.table("person").loc[acs, "SPM_SNAPSUB"].eq(0.0).all()
        pd.testing.assert_frame_equal(_source_table(zero), table, check_exact=True)
        pd.testing.assert_series_equal(
            reported_snap_receipt_by_spm_unit(zero.table("person")),
            receipt,
            check_exact=True,
        )

    def test_a_mixed_frame_without_the_reported_subsidy_is_refused(self) -> None:
        """Both readers refuse the mixed frame without ``SPM_SNAPSUB``."""

        frame = _one_population_snap_frame(np.nan)
        stripped = _with_person(
            frame, frame.table("person").drop(columns=["SPM_SNAPSUB"])
        )
        with pytest.raises(ValueError, match="raw 'SPM_SNAPSUB' reported-subsidy"):
            _source_table(stripped)
        with pytest.raises(ValueError, match="Reported SNAP receipt requires"):
            reported_snap_receipt_by_spm_unit(stripped.table("person"))


class TestRatePrior:
    def test_rate_is_target_over_weighted_eligibles(self) -> None:
        table = pd.DataFrame(
            {
                "state_fips": ["06", "06", "06", "36"],
                US_SNAP_ELIGIBILITY_COLUMN: [True, True, False, True],
                "spm_unit_weight": [2.0, 2.0, 2.0, 1.0],
            }
        )
        rated = with_us_snap_state_take_up_rate(
            table, _state_targets([("06", 2.0), ("36", 5.0)])
        )
        rates = rated.set_index(rated.index)["snap_take_up_rate"]
        assert rates.iloc[0] == pytest.approx(0.5)  # 2 / (2 + 2)
        assert rates.iloc[3] == 1.0  # 5 / 1 clipped


class TestAssignment:
    def test_unsaturated_state_hits_the_fns_count(self) -> None:
        # State 06: one eligible reporter + five eligible non-reporters,
        # unit weight 1, FNS count 3 -> reporter plus exactly two fills.
        rows = [{"SPM_SNAPSUB": 1_200.0}] + [{}] * 5
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.ones(6, dtype=bool),
            state_fips=np.asarray(["06"] * 6),
            state_targets=_state_targets([("06", 3.0)]),
            seed=0,
        )
        spm_unit = result.table("spm_unit")
        takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) is True  # the reporter
        assert int(takes_up.sum()) == 3
        state_row = diagnostics["states"][0]
        assert state_row["caseload_weight"] == pytest.approx(3.0)
        assert state_row["saturated"] is False
        assert us_snap_state_take_up_gate(diagnostics).passed

    def test_saturated_state_enrolls_every_eligible_unit(self) -> None:
        rows = [{}] * 3
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([True, True, False]),
            state_fips=np.asarray(["36"] * 3),
            state_targets=_state_targets([("36", 10.0)]),
            seed=0,
        )
        spm_unit = result.table("spm_unit")
        takes_up = spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) and bool(takes_up.iloc[1])
        state_row = diagnostics["states"][0]
        assert state_row["saturated"] is True
        assert diagnostics["saturated_states"] == ["36"]
        gate = us_snap_state_take_up_gate(diagnostics)
        assert gate.passed

    def test_reported_recipients_survive_downward_calibration(self) -> None:
        # Two eligible reporters against an FNS count of one: calibration
        # cannot remove anchored units, so both keep the flag and the gate
        # treats the anchor mass as the reachable floor.
        rows = [{"SPM_SNAPSUB": 900.0}, {"SPM_SNAPSUB": 600.0}, {}, {}]
        frame = _us_frame(rows)
        result, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.ones(4, dtype=bool),
            state_fips=np.asarray(["48"] * 4),
            state_targets=_state_targets([("48", 1.0)]),
            seed=0,
        )
        takes_up = result.table("spm_unit")[US_SNAP_TAKE_UP_OUTPUT_COLUMN]
        assert bool(takes_up.iloc[0]) and bool(takes_up.iloc[1])
        assert diagnostics["national"]["anchored_not_taking_up_count"] == 0
        assert us_snap_state_take_up_gate(diagnostics).passed

    def test_empty_targets_are_refused(self) -> None:
        frame = _us_frame([{}])
        with pytest.raises(ValueError, match="non-empty FNS state household"):
            with_us_snap_state_take_up(
                frame,
                is_snap_eligible=np.asarray([True]),
                state_fips=np.asarray(["06"]),
                state_targets=pd.DataFrame(),
                seed=0,
            )


class TestGate:
    def test_state_without_target_fails(self) -> None:
        frame = _us_frame([{}, {}])
        _, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([True, True]),
            state_fips=np.asarray(["06", "36"]),
            state_targets=_state_targets([("06", 1.0)]),
            seed=0,
        )
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("36" in failure for failure in gate.failures)

    def test_positive_target_with_zero_eligibility_fails(self) -> None:
        frame = _us_frame([{}, {}])
        _, diagnostics = with_us_snap_state_take_up(
            frame,
            is_snap_eligible=np.asarray([False, False]),
            state_fips=np.asarray(["06", "06"]),
            state_targets=_state_targets([("06", 2.0)]),
            seed=0,
        )
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("eligibility feed collapsed" in f for f in gate.failures)

    def test_dropped_anchor_fails_even_when_counts_agree(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "06",
                    "target": 2.0,
                    "eligible_weight": 5.0,
                    "anchored_eligible_weight": 1.0,
                    "caseload_weight": 2.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 1,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("anchor was not preserved" in f for f in gate.failures)

    def test_unsaturated_count_miss_fails(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "17",
                    "target": 100.0,
                    "eligible_weight": 500.0,
                    "anchored_eligible_weight": 10.0,
                    "caseload_weight": 60.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 0,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("misses the FNS count" in f for f in gate.failures)

    def test_universal_take_up_landmine_fails(self) -> None:
        diagnostics = {
            "states": [
                {
                    "state_fips": "17",
                    "target": 100.0,
                    "eligible_weight": 500.0,
                    "anchored_eligible_weight": 10.0,
                    "caseload_weight": 500.0,
                    "saturated": False,
                    "anchored_not_taking_up_count": 0,
                    "max_unit_weight": 1.0,
                }
            ],
            "states_without_targets": [],
        }
        gate = us_snap_state_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("universal-take-up landmine" in f for f in gate.failures)
