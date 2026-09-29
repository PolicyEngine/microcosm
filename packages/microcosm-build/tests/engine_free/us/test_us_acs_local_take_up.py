"""ACS-row SNAP/TANF take-up (microcosm#1019) and the engine-free
discretionary-exemption, housing-receipt and Medicare fills (microcosm#1022)
for the retained ACS local lane."""

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
from microcosm.build.us_runtime.snap_discretionary_exemption import (
    _stable_person_draws,
    us_snap_discretionary_exemption_stage_spec,
)
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
PERSON_TAG = spine_column("person")
DISCRETIONARY = "is_snap_abawd_discretionary_exempt"
HOUSING = "receives_housing_assistance"
HOUSING_TAKE_UP = "takes_up_housing_assistance_if_eligible"
MEDICARE = "takes_up_medicare_if_eligible"


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
    """ASEC rows carry donor flags; ACS rows carry NaN flags and no source ids.

    One person per household and unit. ACS households carry ``SERIALNO`` and
    ``TYPEHUGQ`` (every tenth one group quarters), ACS persons ``SPORDER`` 1
    and a native ``HINS3``; the housing take-up flag is transferred on both
    spines.
    """

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
    # microcosm#1022 inputs: donor cells present, ACS cells missing.
    housing_take_up = np.where(asec, position < 30, rng.random(n) < 0.1)
    spm_unit[HOUSING_TAKE_UP] = housing_take_up
    spm_unit[HOUSING] = pd.Series(housing_take_up, dtype=object).where(asec, np.nan)
    household = tables["household"]
    household[spine_column("household")] = spm_unit[TAG].to_numpy()
    household["SERIALNO"] = pd.Series(
        [f"2024HU{i:07d}" for i in ids], dtype=object
    ).where(~asec, np.nan)
    household["TYPEHUGQ"] = np.where(
        asec, np.nan, np.where(position % 10 == 3, 2.0, 1.0)
    )
    person[PERSON_TAG] = spm_unit[TAG].to_numpy()
    age = rng.integers(0, 91, n).astype(float)
    person["age"] = age
    person["SPORDER"] = np.where(asec, np.nan, 1.0)
    covered_65 = rng.random(n) < 0.94
    covered_young = rng.random(n) < 0.03
    person["HINS3"] = np.where(
        asec, np.nan, np.where(np.where(age >= 65, covered_65, covered_young), 1, 2)
    )
    person[DISCRETIONARY] = pd.Series(
        (age >= 18) & (age <= 64) & (position < 30), dtype=object
    ).where(asec, np.nan)
    person[MEDICARE] = pd.Series(position < 40, dtype=object).where(asec, np.nan)
    for column in drop:
        for table in tables.values():
            if column in table:
                table.drop(columns=column, inplace=True)
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
    tables["person"] = tables["person"].assign(takes_up_ssi_if_eligible=True)
    frame = Frame(tables, frame.schema, {"household": frame.weights_for("household")})
    gate = acs_local_take_up_signal_gate(frame)
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_TAKE_UP_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]["columns"]
    assert acs[SNAP]["reporters_not_taking_up"] == 0
    assert set(acs) == {SNAP, TANF, DISCRETIONARY, HOUSING, MEDICARE}
    assert acs[HOUSING]["disagrees_with_take_up"] == 0
    assert acs[MEDICARE]["differs_from_native_coverage"] == 0


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


# ---------------------------------------------------------------------------
# microcosm#1022: engine-free fills
# ---------------------------------------------------------------------------


def _acs_persons(frame: Frame) -> np.ndarray:
    return frame.table("person")[PERSON_TAG].eq(ACS_2024_1YR_SPINE).to_numpy()


def _with_table(frame: Frame, entity: str, table: pd.DataFrame) -> Frame:
    return Frame(
        {
            name: table if name == entity else frame.table(name)
            for name in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _manifest_exemption_rate() -> float:
    (operation,) = [
        op
        for op in us_snap_discretionary_exemption_stage_spec().operations
        if op.kind == "derive_snap_abawd_discretionary_exemption"
    ]
    return float(operation.parameters["exemption_rate"]["value"])


def _covered(frame: Frame) -> np.ndarray:
    age = frame.table("person")["age"].to_numpy()
    return (age >= 18) & (age <= 64)


def test_engine_free_fills_keep_donor_cells_and_fill_every_acs_cell() -> None:
    frame = _frame()
    result, receipt = with_acs_local_take_up_inputs(frame, seed=0)
    fills = receipt["engine_free_fills"]
    assert fills["issue"] == "microcosm#1022"
    for entity, column, acs in (
        ("person", DISCRETIONARY, _acs_persons(frame)),
        ("spm_unit", HOUSING, _acs(frame)),
        ("person", MEDICARE, _acs_persons(frame)),
    ):
        before = frame.table(entity)[column]
        after = result.table(entity)[column]
        assert after.notna().all() and after.dtype == bool
        assert np.array_equal(
            before[~acs].to_numpy(dtype=object), after[~acs].to_numpy(dtype=object)
        )
        assert fills["columns"][column]["filled_rows"] == int(acs.sum())
        assert fills["columns"][column]["preserved_rows"] == 0
        # The input frame is not mutated.
        assert frame.table(entity)[column].isna().sum() == int(acs.sum())


def test_stored_acs_cells_are_kept() -> None:
    frame = _frame()
    acs_row = int(np.flatnonzero(_acs_persons(frame))[0])
    person = frame.table("person").copy()
    person.loc[acs_row, DISCRETIONARY] = True
    person.loc[acs_row, MEDICARE] = False
    result, receipt = with_acs_local_take_up_inputs(
        _with_table(frame, "person", person), seed=0
    )
    assert bool(result.table("person")[DISCRETIONARY].iloc[acs_row])
    assert not bool(result.table("person")[MEDICARE].iloc[acs_row])
    for column in (DISCRETIONARY, MEDICARE):
        assert receipt["engine_free_fills"]["columns"][column]["preserved_rows"] == 1


def test_discretionary_share_lands_on_the_manifest_rate_among_18_to_64() -> None:
    frame = _frame()
    result, receipt = with_acs_local_take_up_inputs(frame, seed=5)
    exempt = result.table("person")[DISCRETIONARY].to_numpy(dtype=bool)
    acs = _acs_persons(frame)
    covered = _covered(frame)
    weights = frame.resolve_weights("person").values
    rate = _manifest_exemption_rate()
    entry = receipt["engine_free_fills"]["columns"][DISCRETIONARY]
    assert entry["rate"] == rate
    assert entry["share_band"] == pytest.approx([rate / 2, rate * 1.5])
    share = weights[acs & covered & exempt].sum() / weights[acs & covered].sum()
    assert share == pytest.approx(rate, abs=0.02)
    assert entry["weighted_covered_exempt_share"] == pytest.approx(share)
    assert not exempt[acs & ~covered].any()
    assert entry["covered_filled_rows"] == int((acs & covered).sum())


def test_discretionary_rate_is_read_from_the_manifest(monkeypatch) -> None:
    from microcosm.build.us_runtime import acs_local_take_up

    monkeypatch.setattr(acs_local_take_up, "_discretionary_rate", lambda: (0.3, "x"))
    frame = _frame()
    result, receipt = with_acs_local_take_up_inputs(frame, seed=5)
    exempt = result.table("person")[DISCRETIONARY].to_numpy(dtype=bool)
    rows = _acs_persons(frame) & _covered(frame)
    assert exempt[rows].mean() == pytest.approx(0.3, abs=0.04)
    assert receipt["engine_free_fills"]["columns"][DISCRETIONARY]["rate"] == 0.3


def test_discretionary_draws_mirror_the_donor_hash_on_serialno_sporder() -> None:
    frame = _frame()
    result, _ = with_acs_local_take_up_inputs(frame, seed=9)
    person = frame.table("person")
    serial = person["person_household_id"].map(
        frame.table("household").set_index("household_id")["SERIALNO"]
    )
    rows = _acs_persons(frame) & _covered(frame)
    keys = f"{ACS_2024_1YR_SPINE}:" + serial[rows] + ":1"
    draws = _stable_person_draws(pd.DataFrame({"person_id": keys.to_numpy()}), seed=9)
    exempt = result.table("person")[DISCRETIONARY].to_numpy(dtype=bool)[rows]
    assert np.array_equal(exempt, draws < _manifest_exemption_rate())


def test_discretionary_draw_keys_are_unique_across_acs_households() -> None:
    """``SPORDER`` repeats in every household; the key must not collapse."""

    frame = _frame()
    result, _ = with_acs_local_take_up_inputs(frame, seed=0)
    rows = _acs_persons(frame) & _covered(frame)
    # Every ACS person has SPORDER 1, yet the draws still vary.
    assert result.table("person").loc[rows, DISCRETIONARY].nunique() == 2
    household = frame.table("household").copy()
    first, second = np.flatnonzero(household["SERIALNO"].notna())[:2]
    household.loc[second, "SERIALNO"] = household.loc[first, "SERIALNO"]
    with pytest.raises(ValueError, match="draw key"):
        with_acs_local_take_up_inputs(
            _with_table(frame, "household", household), seed=0
        )


@pytest.mark.parametrize("column", ["SERIALNO", "SPORDER"])
def test_discretionary_draws_refuse_a_missing_key_part(column: str) -> None:
    frame = _frame()
    entity = "household" if column == "SERIALNO" else "person"
    table = frame.table(entity).copy()
    acs_row = int(np.flatnonzero(table[column].notna())[0])
    table.loc[acs_row, column] = np.nan
    with pytest.raises(ValueError, match="SERIALNO and its SPORDER"):
        with_acs_local_take_up_inputs(_with_table(frame, entity, table), seed=0)


def test_housing_receipt_copies_take_up_and_is_false_in_group_quarters() -> None:
    frame = _frame()
    result, receipt = with_acs_local_take_up_inputs(frame, seed=0)
    spm_unit = result.table("spm_unit")
    acs = _acs(frame)
    kinds = frame.table("household")["TYPEHUGQ"].to_numpy()
    gq = acs & np.isin(kinds, (2, 3))
    take_up = spm_unit[HOUSING_TAKE_UP].to_numpy(dtype=bool)
    receives = spm_unit[HOUSING].to_numpy(dtype=bool)
    assert np.array_equal(receives[acs & ~gq], take_up[acs & ~gq])
    assert not receives[gq].any()
    # A group-quarters unit can carry a transferred take-up flag (#975); the
    # receipt records it and leaves the flag alone.
    assert (gq & take_up).any()
    entry = receipt["engine_free_fills"]["columns"][HOUSING]
    assert entry["group_quarters_filled_false"] == int(gq.sum())
    assert entry["group_quarters_take_up_true"] == int((gq & take_up).sum())
    assert np.array_equal(
        spm_unit[HOUSING_TAKE_UP].to_numpy(),
        frame.table("spm_unit")[HOUSING_TAKE_UP].to_numpy(),
    )


def test_housing_receipt_refuses_an_unknown_household_kind() -> None:
    frame = _frame()
    household = frame.table("household").copy()
    household.loc[int(np.flatnonzero(_acs(frame))[0]), "TYPEHUGQ"] = np.nan
    with pytest.raises(ValueError, match="TYPEHUGQ 1/2/3"):
        with_acs_local_take_up_inputs(
            _with_table(frame, "household", household), seed=0
        )


def test_housing_receipt_refuses_a_missing_transferred_take_up() -> None:
    frame = _frame()
    spm_unit = frame.table("spm_unit").copy()
    kinds = frame.table("household")["TYPEHUGQ"].to_numpy()
    housing_unit = int(np.flatnonzero(_acs(frame) & (kinds == 1))[0])
    spm_unit[HOUSING_TAKE_UP] = spm_unit[HOUSING_TAKE_UP].astype(object)
    spm_unit.loc[housing_unit, HOUSING_TAKE_UP] = np.nan
    with pytest.raises(ValueError, match="no transferred"):
        with_acs_local_take_up_inputs(_with_spm_table(frame, spm_unit), seed=0)


def test_medicare_take_up_is_native_hins3_and_blank_reads_false() -> None:
    frame = _frame()
    person = frame.table("person").copy()
    acs = _acs_persons(frame)
    blank = int(np.flatnonzero(acs & (person["HINS3"] == 1).to_numpy())[0])
    person.loc[blank, "HINS3"] = np.nan
    result, receipt = with_acs_local_take_up_inputs(
        _with_table(frame, "person", person), seed=0
    )
    medicare = result.table("person")[MEDICARE].to_numpy(dtype=bool)
    assert np.array_equal(medicare[acs], (person["HINS3"] == 1).to_numpy()[acs])
    assert not medicare[blank]
    entry = receipt["engine_free_fills"]["columns"][MEDICARE]
    assert entry["blank_coverage_rows_as_false"] == 1
    assert entry["weighted_aged_take_up_share"] > 0.8


def test_discretionary_receipt_describes_a_cap_based_proxy() -> None:
    _, receipt = with_acs_local_take_up_inputs(_frame(), seed=0)
    entry = receipt["engine_free_fills"]["columns"][DISCRETIONARY]
    assert entry["interpretation"] == (
        "cap-based proxy: an upper-bound propensity drawn at the statutory cap "
        "across all adults 18-64, not an observed exemption assignment"
    )


def test_blank_and_invalid_hins3_are_counted_but_not_graded() -> None:
    frame = _frame()
    person = frame.table("person").copy()
    acs = _acs_persons(frame)
    age = person["age"].to_numpy()
    aged_rows = np.flatnonzero(acs & (age >= 65))
    young_rows = np.flatnonzero(acs & (age < 65))
    blank = [int(aged_rows[0]), int(young_rows[0]), int(young_rows[1])]
    invalid = [int(aged_rows[1]), int(young_rows[2])]
    person["HINS3"] = person["HINS3"].astype(object)
    person.loc[blank, "HINS3"] = np.nan
    person.loc[invalid[0], "HINS3"] = 3
    person.loc[invalid[1], "HINS3"] = 0
    edited = _with_table(frame, "person", person)
    weights = edited.resolve_weights("person").values

    def expected(rows: list[int]) -> dict[str, object]:
        aged = [row for row in rows if age[row] >= 65]
        return {
            "rows": len(rows),
            "weight": pytest.approx(float(weights[rows].sum())),
            "rows_65_plus": len(aged),
            "weight_65_plus": pytest.approx(float(weights[aged].sum())),
        }

    result, receipt = with_acs_local_take_up_inputs(edited, seed=0)
    audit = receipt["engine_free_fills"]["columns"][MEDICARE]["hins3_audit"]
    assert audit["blank"] == expected(blank)
    assert audit["invalid"] == expected(invalid)
    assert audit["persons"] == int(acs.sum())
    assert audit["persons_65_plus"] == int((acs & (age >= 65)).sum())
    # Both read as not covered.
    medicare = result.table("person")[MEDICARE].to_numpy(dtype=bool)
    assert not medicare[blank + invalid].any()
    # The gate reports the same counts on the ACS spine and still passes.
    gate = acs_local_take_up_signal_gate(result)
    assert gate.passed, gate.failures
    columns = gate.details["per_spine"][ACS_2024_1YR_SPINE]["columns"]
    assert columns[MEDICARE]["hins3_audit"]["blank"] == expected(blank)
    assert columns[MEDICARE]["hins3_audit"]["invalid"] == expected(invalid)
    donor = gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]["columns"]
    assert "hins3_audit" not in donor[MEDICARE]


def test_engine_free_fills_are_deterministic_and_digested() -> None:
    first, first_receipt = with_acs_local_take_up_inputs(_frame(), seed=11)
    again, again_receipt = with_acs_local_take_up_inputs(_frame(), seed=11)
    other, _ = with_acs_local_take_up_inputs(_frame(), seed=12)
    assert first_receipt == again_receipt
    for column in (DISCRETIONARY, MEDICARE):
        assert first.table("person")[column].equals(again.table("person")[column])
    assert not first.table("person")[DISCRETIONARY].equals(
        other.table("person")[DISCRETIONARY]
    )
    # The digest covers the #1022 cells: flipping one changes it.
    person = first.table("person").copy()
    row = int(np.flatnonzero(_acs_persons(first))[0])
    person.loc[row, MEDICARE] = not bool(person.loc[row, MEDICARE])
    _, flipped = with_acs_local_take_up_inputs(
        _with_table(first, "person", person), seed=11
    )
    assert flipped["assigned_sha256"] != first_receipt["assigned_sha256"]


@pytest.mark.parametrize(
    "column", [DISCRETIONARY, MEDICARE, "age", "HINS3", HOUSING, HOUSING_TAKE_UP]
)
def test_missing_engine_free_inputs_are_refused(column: str) -> None:
    with pytest.raises(ValueError, match="microcosm#1022"):
        with_acs_local_take_up_inputs(_frame(drop=(column,)), seed=0)


def _gate_failures(frame: Frame) -> str:
    gate = acs_local_take_up_signal_gate(frame)
    assert not gate.passed
    return " ".join(gate.failures)


def test_gate_fails_on_the_engine_default_signature_of_1022() -> None:
    """The pre-#1022 release: all three at the engine default on ACS rows."""

    frame = _filled()
    acs_persons = _acs_persons(frame)
    person = frame.table("person").copy()
    person[DISCRETIONARY] = person[DISCRETIONARY].where(~acs_persons, False)
    person[MEDICARE] = person[MEDICARE].where(~acs_persons, True)
    spm_unit = frame.table("spm_unit").copy()
    spm_unit[HOUSING] = spm_unit[HOUSING].where(~_acs(frame), False)
    failures = _gate_failures(
        _with_spm_table(_with_table(frame, "person", person), spm_unit)
    )
    for column in (DISCRETIONARY, MEDICARE):
        assert f"{ACS_2024_1YR_SPINE}: {column} is constant" in failures
    assert f"{HOUSING} differs from the transferred" in failures
    assert ASEC_PUF_DONOR_SPINE not in failures


def test_gate_fails_on_missing_engine_free_cells() -> None:
    frame = _filled()
    person = frame.table("person").copy()
    person[MEDICARE] = person[MEDICARE].astype(object)
    person.loc[int(np.flatnonzero(_acs_persons(frame))[0]), MEDICARE] = np.nan
    failures = _gate_failures(_with_table(frame, "person", person))
    assert f"{ACS_2024_1YR_SPINE}: {MEDICARE} has missing rows" in failures


def test_gate_grades_the_discretionary_band_and_age_scope() -> None:
    frame = _filled()
    acs = _acs_persons(frame)
    covered = _covered(frame)
    person = frame.table("person").copy()
    # Out of band: a third of the covered ACS adults exempt.
    high = person[DISCRETIONARY].where(~(acs & covered), person["person_id"] % 3 == 0)
    failures = _gate_failures(
        _with_table(frame, "person", person.assign(**{DISCRETIONARY: high}))
    )
    assert f"{DISCRETIONARY} share among ages 18-64" in failures
    # A flagged child is refused even when the covered share is in band.
    child = int(np.flatnonzero(acs & (person["age"] < 18).to_numpy())[0])
    person.loc[child, DISCRETIONARY] = True
    failures = _gate_failures(_with_table(frame, "person", person))
    assert "1 person(s) outside ages 18-64" in failures
    assert "share among ages 18-64" not in failures


def test_gate_grades_the_housing_copy_and_group_quarters() -> None:
    frame = _filled()
    spm_unit = frame.table("spm_unit").copy()
    kinds = frame.table("household")["TYPEHUGQ"].to_numpy()
    gq = int(np.flatnonzero(_acs(frame) & (kinds == 2))[0])
    spm_unit.loc[gq, HOUSING] = True
    failures = _gate_failures(_with_spm_table(frame, spm_unit))
    assert "1 group-quarters SPM unit(s) receive housing assistance" in failures


def test_gate_grades_medicare_against_hins3_and_the_aged_floor() -> None:
    frame = _filled()
    acs = _acs_persons(frame)
    person = frame.table("person").copy()
    row = int(np.flatnonzero(acs)[0])
    person.loc[row, MEDICARE] = not bool(person.loc[row, MEDICARE])
    failures = _gate_failures(_with_table(frame, "person", person))
    assert f"{MEDICARE} differs from ACS HINS3 == 1 on 1 person(s)" in failures
    # A miscoded item (1/2 swapped) matches its own recomputation but not the
    # aged floor.
    swapped = frame.table("person").copy()
    swapped["HINS3"] = swapped["HINS3"].map({1: 2, 2: 1})
    swapped[MEDICARE] = swapped[MEDICARE].where(~acs, swapped["HINS3"] == 1)
    failures = _gate_failures(_with_table(frame, "person", swapped))
    assert f"{MEDICARE} share among ages 65+" in failures
    assert "differs from ACS HINS3" not in failures


def test_gate_does_not_grade_donor_engine_free_cells() -> None:
    frame = _filled()
    asec = ~_acs(frame)
    spm_unit = frame.table("spm_unit").copy()
    donor_row = int(np.flatnonzero(asec & ~spm_unit[HOUSING].to_numpy(dtype=bool))[0])
    spm_unit.loc[donor_row, HOUSING] = True
    gate = acs_local_take_up_signal_gate(_with_spm_table(frame, spm_unit))
    assert gate.passed, gate.failures
    donor = gate.details["per_spine"][ASEC_PUF_DONOR_SPINE]["columns"]
    assert donor[HOUSING]["disagrees_with_take_up"] == 1


def test_gate_fails_on_missing_person_tags() -> None:
    frame = _filled()
    person = frame.table("person").copy()
    person[PERSON_TAG] = person[PERSON_TAG].where(person.index > 0)
    gate = acs_local_take_up_signal_gate(_with_table(frame, "person", person))
    assert gate.failures == (f"Missing person origin tags: {PERSON_TAG}.",)
