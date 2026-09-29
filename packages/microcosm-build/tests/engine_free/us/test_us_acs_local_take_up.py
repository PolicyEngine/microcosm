"""ACS-row SNAP/TANF take-up for the retained ACS local lane (microcosm#1019)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_take_up import (
    ACS_LOCAL_TAKE_UP_GATE_NAME,
    US_TANF_TAKE_UP_OUTPUT_COLUMN,
    acs_local_take_up_signal_gate,
    with_acs_local_take_up_inputs,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.snap_take_up import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    us_snap_take_up_stage_spec,
)
from microcosm.build.us_runtime.take_up import _seed_program
from microcosm.build.us_runtime.take_up_contract import seeded_take_up_programs
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

SNAP = US_SNAP_TAKE_UP_OUTPUT_COLUMN
TANF = US_TANF_TAKE_UP_OUTPUT_COLUMN
TAG = spine_column("spm_unit")


def _manifest_snap_rate() -> float:
    (operation,) = [
        op
        for op in us_snap_take_up_stage_spec().operations
        if op.kind == "derive_snap_take_up"
    ]
    return float(operation.parameters["take_up_rate"]["value"])


def _frame(
    *,
    n_asec: int = 200,
    n_acs: int = 3000,
    reporter_share: float = 0.1,
    drop: tuple[str, ...] = (),
) -> Frame:
    """ASEC rows carry donor flags; ACS rows carry NaN flags and no source ids."""

    n = n_asec + n_acs
    ids = np.arange(1, n + 1)
    rng = np.random.default_rng(7)
    asec = np.arange(n) < n_asec
    person = pd.DataFrame({"person_id": ids})
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity == "person":
            continue
        person[f"person_{entity}_id"] = ids
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    person["source_year"] = np.where(asec, 2024, np.nan)
    person["source_household_id"] = np.where(asec, ids + 10_000, np.nan)
    person["source_person_id"] = np.where(asec, ids + 20_000, np.nan)
    spm_unit = tables["spm_unit"]
    spm_unit[TAG] = np.where(asec, ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE)
    position = np.arange(n)
    spm_unit[SNAP] = pd.Series(
        np.where(position < int(0.85 * n_asec), True, False), dtype=object
    ).where(asec, np.nan)
    spm_unit[TANF] = pd.Series(
        np.where(position < int(0.25 * n_asec), True, False), dtype=object
    ).where(asec, np.nan)
    reporters = np.where(asec, position < 20, rng.random(n) < reporter_share)
    spm_unit["receives_snap"] = reporters
    for column in drop:
        spm_unit.drop(columns=column, inplace=True)
    weights = rng.uniform(0.5, 2.0, n)
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def _with_spm_table(frame: Frame, spm_unit: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: spm_unit if entity == "spm_unit" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _with_spm(frame: Frame, **columns) -> Frame:
    return _with_spm_table(frame, frame.table("spm_unit").assign(**columns))


def _acs(frame: Frame) -> np.ndarray:
    return frame.table("spm_unit")[TAG].eq(ACS_2024_1YR_SPINE).to_numpy()


def test_donor_rows_stay_identical_and_acs_cells_are_filled() -> None:
    frame = _frame()
    before = frame.table("spm_unit").copy(deep=True)
    result, receipt = with_acs_local_take_up_inputs(frame, seed=0)
    after = result.table("spm_unit")
    asec = ~_acs(frame)
    for column in (SNAP, TANF):
        assert after[column].notna().all()
        assert after[column].dtype == bool
        assert np.array_equal(
            before.loc[asec, column].to_numpy(dtype=object),
            after.loc[asec, column].to_numpy(dtype=object),
        )
        assert receipt["programs"][column]["filled_rows"] == int(_acs(frame).sum())
    # The input frame is not mutated.
    assert frame.table("spm_unit")[SNAP].isna().sum() == int(_acs(frame).sum())


def test_acs_reporters_take_up_and_stored_acs_cells_are_kept() -> None:
    frame = _frame()
    acs = _acs(frame)
    spm_unit = frame.table("spm_unit")
    stored = int(np.flatnonzero(acs & spm_unit["receives_snap"].to_numpy())[0])
    snap = spm_unit[SNAP].copy()
    snap.iloc[stored] = False
    result, receipt = with_acs_local_take_up_inputs(
        _with_spm(frame, **{SNAP: snap}), seed=0
    )
    after = result.table("spm_unit")
    reporters = acs & spm_unit["receives_snap"].to_numpy()
    reporters[stored] = False
    assert after.loc[reporters, SNAP].all()
    assert not after[SNAP].iloc[stored]
    assert receipt["programs"][SNAP]["preserved_rows"] == 1


def test_weighted_acs_shares_land_on_the_manifest_and_contract_rates() -> None:
    frame = _frame()
    acs = _acs(frame)
    result, receipt = with_acs_local_take_up_inputs(frame, seed=3)
    after = result.table("spm_unit")
    weights = frame.weights_for("household").values
    snap_share = weights[acs & after[SNAP].to_numpy()].sum() / weights[acs].sum()
    rate = _manifest_snap_rate()
    assert receipt["programs"][SNAP]["rate"] == rate
    assert snap_share == pytest.approx(rate, abs=0.02)
    assert receipt["programs"][SNAP]["weighted_take_up_share"] == pytest.approx(
        snap_share
    )
    # ACS rows have no source identity; the draws must not collapse onto one key.
    non_reporters = acs & ~frame.table("spm_unit")["receives_snap"].to_numpy()
    assert after.loc[non_reporters, SNAP].nunique() == 2

    (program,) = [p for p in seeded_take_up_programs() if p.variable == TANF]
    seeded, _ = _seed_program(frame, program, seed=3)
    assert np.array_equal(after.loc[acs, TANF].to_numpy(), seeded[TANF][acs])
    tanf_share = receipt["programs"][TANF]["weighted_take_up_share"]
    assert tanf_share == pytest.approx(float(program.rate["value"]), abs=0.03)


def test_assignment_is_deterministic_in_the_seed() -> None:
    first, first_receipt = with_acs_local_take_up_inputs(_frame(), seed=11)
    again, again_receipt = with_acs_local_take_up_inputs(_frame(), seed=11)
    other, other_receipt = with_acs_local_take_up_inputs(_frame(), seed=12)
    assert first_receipt == again_receipt
    for column in (SNAP, TANF):
        assert first.table("spm_unit")[column].equals(again.table("spm_unit")[column])
    assert other_receipt["assigned_sha256"] != first_receipt["assigned_sha256"]


def test_a_filled_frame_passes_through_unchanged() -> None:
    filled, receipt = with_acs_local_take_up_inputs(_frame(), seed=0)
    again, again_receipt = with_acs_local_take_up_inputs(filled, seed=0)
    assert again is filled
    assert again_receipt["programs"][SNAP]["filled_rows"] == 0
    assert again_receipt["assigned_sha256"] == receipt["assigned_sha256"]


@pytest.mark.parametrize("column", [SNAP, TANF, "receives_snap", TAG])
def test_missing_inputs_are_refused(column: str) -> None:
    with pytest.raises(ValueError, match="ACS local take-up requires"):
        with_acs_local_take_up_inputs(_frame(drop=(column,)), seed=0)


def _filled() -> Frame:
    return with_acs_local_take_up_inputs(_frame(), seed=0)[0]


def test_gate_passes_on_the_seeded_surface_and_ignores_other_flags() -> None:
    frame = _filled()
    # Other take-up flags remain engine defaults on ACS rows (microcosm#1022).
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["tax_unit"] = tables["tax_unit"].assign(takes_up_eitc=True)
    tables["spm_unit"] = tables["spm_unit"].assign(
        takes_up_housing_assistance_if_eligible=True
    )
    frame = Frame(tables, frame.schema, {"household": frame.weights_for("household")})
    gate = acs_local_take_up_signal_gate(frame)
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_TAKE_UP_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]["columns"]
    assert acs[SNAP]["reporters_not_taking_up"] == 0
    assert set(acs) == {SNAP, TANF}


def test_gate_fails_on_the_default_filled_release_signature() -> None:
    """The 2026-09-23 release: every ACS unit default-filled to take up."""

    frame = _frame()
    acs = _acs(frame)
    spm_unit = frame.table("spm_unit")
    frame = _with_spm(
        frame,
        **{column: spm_unit[column].where(~acs, True) for column in (SNAP, TANF)},
    )
    gate = acs_local_take_up_signal_gate(frame)
    assert not gate.passed
    for column in (SNAP, TANF):
        assert f"{ACS_2024_1YR_SPINE}: {column} is constant" in " ".join(gate.failures)
    assert not any(ASEC_PUF_DONOR_SPINE in failure for failure in gate.failures)


def test_gate_fails_on_missing_cells() -> None:
    gate = acs_local_take_up_signal_gate(_frame())
    assert not gate.passed
    assert f"{ACS_2024_1YR_SPINE}: {SNAP} has missing rows" in " ".join(gate.failures)


def test_gate_fails_when_a_reporter_does_not_take_up() -> None:
    frame = _filled()
    spm_unit = frame.table("spm_unit")
    reporter = int(
        np.flatnonzero(_acs(frame) & spm_unit["receives_snap"].to_numpy())[0]
    )
    snap = spm_unit[SNAP].copy()
    snap.iloc[reporter] = False
    gate = acs_local_take_up_signal_gate(_with_spm(frame, **{SNAP: snap}))
    assert not gate.passed
    assert "1 SPM unit(s) report SNAP receipt" in " ".join(gate.failures)


def test_gate_fails_on_a_non_constant_share_outside_the_band() -> None:
    frame = _filled()
    acs = _acs(frame)
    spm_unit = frame.table("spm_unit")
    # Only reporters take up: a non-constant surface far below the FNS rate.
    low_snap = spm_unit[SNAP].where(~acs, spm_unit["receives_snap"])
    gate = acs_local_take_up_signal_gate(_with_spm(frame, **{SNAP: low_snap}))
    assert not gate.passed
    assert f"{ACS_2024_1YR_SPINE}: {SNAP} take-up share" in " ".join(gate.failures)
    assert f"{TANF}" not in " ".join(gate.failures)


def test_gate_reports_but_does_not_grade_donor_shares_or_anchors() -> None:
    """Donor-spine cells come from the donor release; only completeness fails."""

    frame = _filled()
    asec = ~_acs(frame)
    spm_unit = frame.table("spm_unit")
    # A donor surface far outside the band, with a reporter not taking up.
    donor_snap = spm_unit[SNAP].where(~asec, spm_unit["receives_snap"])
    reporter = int(np.flatnonzero(asec & spm_unit["receives_snap"].to_numpy())[0])
    donor_snap.iloc[reporter] = False
    gate = acs_local_take_up_signal_gate(_with_spm(frame, **{SNAP: donor_snap}))
    assert gate.passed, gate.failures
    donor = gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]
    assert donor["graded"] is False
    assert donor["columns"][SNAP]["reporters_not_taking_up"] == 1
    assert gate.details["per_spine"][ACS_2024_1YR_SPINE]["graded"] is True
    # Completeness still fails on the donor spine.
    constant = acs_local_take_up_signal_gate(
        _with_spm(frame, **{SNAP: spm_unit[SNAP].where(~asec, True)})
    )
    assert f"{ASEC_PUF_DONOR_SPINE}: {SNAP} is constant" in " ".join(constant.failures)


def test_gate_fails_on_missing_columns_tags_or_unknown_spines() -> None:
    frame = _filled()
    spm_unit = frame.table("spm_unit")
    missing = acs_local_take_up_signal_gate(
        _with_spm_table(frame, spm_unit.drop(columns=[TANF, "receives_snap"]))
    )
    assert f"{ACS_2024_1YR_SPINE}: missing {TANF}." in missing.failures
    assert any("reported-receipt anchor" in failure for failure in missing.failures)
    untagged = acs_local_take_up_signal_gate(
        _with_spm(frame, **{TAG: spm_unit[TAG].where(spm_unit.index > 0)})
    )
    assert untagged.failures == (f"Missing SPM-unit origin tags: {TAG}.",)
    unknown = acs_local_take_up_signal_gate(
        _with_spm(frame, **{TAG: spm_unit[TAG].where(spm_unit.index > 0, "other")})
    )
    assert "unsupported spine" in " ".join(unknown.failures)
