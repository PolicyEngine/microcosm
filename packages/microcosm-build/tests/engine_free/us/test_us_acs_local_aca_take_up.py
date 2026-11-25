"""ACS-row ACA take-up and plan ratio for the retained ACS local lane
(microcosm#1720), with stubbed engine pre-pass values."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_aca_take_up import (
    ACS_LOCAL_ACA_TAKE_UP_COLUMNS,
    ACS_LOCAL_ACA_TAKE_UP_GATE_NAME,
    ACS_LOCAL_ACA_TAKE_UP_ISSUE,
    acs_local_aca_take_up_assignment,
    acs_local_aca_take_up_signal_gate,
    with_acs_local_aca_take_up,
    with_recorded_acs_local_aca_take_up,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAKE_UP = "takes_up_aca_if_eligible"
RATIO = "selected_marketplace_plan_benchmark_ratio"
CMS = "cms_aca_aptc_recipients_by_state"
RETURNS = "irs_soi_premium_tax_credit_returns_by_state"
AMOUNT = "irs_soi_premium_tax_credit_amount_by_state"
COVERAGE = "has_marketplace_health_coverage_at_interview"
#: The manifest stage's anchor column; staging writes none today.
ANCHOR = "reported_has_subsidized_marketplace_health_coverage_at_interview"
STATES = (6, 36, 48)
#: Stub engine values ride the tax-unit table, so the pre-pass view carries
#: them: eligibility, the credit before the filing gate, the SLCSP, the
#: reported premium and whether the unit files for another reason.
STUB = ("stub_eligible", "stub_credit", "stub_slcsp", "stub_premium", "stub_filer")


def _frame(*, n_donor: int = 300, n_acs: int = 1500, seed: int = 3) -> Frame:
    """Donor persons alone in their units; ACS households of two persons.

    Even ACS households file one return for both persons and odd ones two,
    so a unit's draw key is its lowest ``SPORDER``. Donor units store both
    ACA inputs; ACS units carry them missing.
    """

    rng = np.random.default_rng(seed)
    donor_households = np.arange(1, n_donor + 1)
    acs_households = np.arange(n_donor + 1, n_donor + n_acs + 1)
    households = np.concatenate([donor_households, acs_households])
    person_household = np.concatenate([donor_households, np.repeat(acs_households, 2)])
    n = len(person_household)
    acs = np.arange(n) >= n_donor
    unit_of = np.empty(n, dtype=np.int64)
    unit_of[:n_donor] = np.arange(1, n_donor + 1)
    sporder = np.full(n, np.nan)
    next_id = n_donor + 1
    for k, household in enumerate(acs_households):
        first, second = n_donor + 2 * k, n_donor + 2 * k + 1
        sporder[first], sporder[second] = 1.0, 2.0
        unit_of[first] = next_id
        unit_of[second] = next_id if household % 2 == 0 else next_id + 1
        next_id += 1 if household % 2 == 0 else 2
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": person_household,
            "person_tax_unit_id": unit_of,
            spine_column("person"): np.where(
                acs, ACS_2024_1YR_SPINE, ASEC_PUF_DONOR_SPINE
            ),
            "SPORDER": sporder,
            COVERAGE: rng.random(n) < 0.05,
        }
    )
    tables = {"person": person}
    for entity in ("spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    unit_ids = np.arange(1, next_id)
    units = len(unit_ids)
    donor_unit = unit_ids <= n_donor
    eligible = rng.random(units) < 0.4
    stored = eligible & (rng.random(units) < 0.5)
    tables["tax_unit"] = pd.DataFrame(
        {
            "tax_unit_id": unit_ids,
            spine_column("tax_unit"): np.where(
                donor_unit, ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE
            ),
            "stub_eligible": eligible,
            "stub_credit": np.where(eligible, rng.uniform(500, 8_000, units), 0.0),
            "stub_slcsp": rng.uniform(5_000, 12_000, units),
            "stub_premium": rng.uniform(0, 6_000, units),
            "stub_filer": rng.random(units) < 0.6,
            TAKE_UP: pd.Series(stored, dtype=object).where(donor_unit, np.nan),
            RATIO: pd.Series(np.where(stored, rng.uniform(0.6, 1.3, units), 1.0)).where(
                donor_unit, np.nan
            ),
        }
    )
    household_acs = households > n_donor
    tables["household"] = pd.DataFrame(
        {
            "household_id": households,
            spine_column("household"): np.where(
                household_acs, ACS_2024_1YR_SPINE, ASEC_PUF_DONOR_SPINE
            ),
            "SERIALNO": pd.Series(
                [f"2024HU{i:07d}" for i in households], dtype=object
            ).where(household_acs, np.nan),
            "state_fips": rng.choice(STATES, len(households)),
        }
    )
    weights = rng.uniform(0.5, 2.0, len(households))
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def _with_tax_unit(frame: Frame, tax_unit: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: tax_unit if entity == "tax_unit" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _unit_values(frame: Frame, *, flags: np.ndarray, person_counts: bool = True):
    """The fiscal lane's tax-unit table, computed from the stub columns.

    The credit is gated on filing as on policyengine-us 2.145.3: a unit
    whose flag is ``True`` or that files anyway gets it.
    """

    units = frame.table("tax_unit")
    person = frame.table("person")
    household = frame.table("household").set_index("household_id")
    weight = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=np.float64),
        index=household.index,
    )
    members = person.groupby("person_tax_unit_id")
    home = members["person_household_id"].first().reindex(units["tax_unit_id"])
    size = members.size().reindex(units["tax_unit_id"]).to_numpy()
    household_weight = weight.reindex(home.to_numpy()).to_numpy()
    eligible = units["stub_eligible"].to_numpy(dtype=bool)
    gate = np.asarray(flags, dtype=bool) | units["stub_filer"].to_numpy(dtype=bool)
    credit = np.where(eligible, units["stub_credit"].to_numpy(), 0.0) * gate
    people = np.where(eligible, household_weight * size, 0.0)
    unit_weight = people if person_counts else household_weight
    return pd.DataFrame(
        {
            "tax_unit_id": units["tax_unit_id"].to_numpy(),
            "state_fips": household["state_fips"]
            .reindex(home.to_numpy())
            .astype(str)
            .str.zfill(2)
            .to_numpy(),
            "tax_unit_weight": unit_weight,
            "household_weight": household_weight,
            "is_aca_ptc_eligible": (unit_weight > 0) & (credit > 0),
            "health_insurance_premiums_without_medicare_part_b": units[
                "stub_premium"
            ].to_numpy(),
            "assigned_aca_ptc": credit,
            "weighted_assigned_aca_ptc": credit * household_weight,
            "slcsp": units["stub_slcsp"].to_numpy(),
        }
    )


def _stub(calls: list | None = None):
    """The pre-pass callable: the stub table plus the fiscal lane's rate."""

    def table(view: Frame, tables) -> pd.DataFrame:
        flags = view.table("tax_unit")[TAKE_UP]
        if calls is not None:
            calls.append(
                {
                    "units": view.table("tax_unit").copy(),
                    "tables": {name: value.copy() for name, value in tables.items()},
                }
            )
        values = _unit_values(
            view, flags=flags.astype(bool).to_numpy(), person_counts=CMS in tables
        )
        cms = tables.get(CMS)
        rate = pd.Series(0.0, index=values.index)
        if cms is not None:
            eligible = values.loc[values["is_aca_ptc_eligible"]]
            weighted = eligible.groupby("state_fips")["tax_unit_weight"].sum()
            rates = (cms.set_index("state_fips")["target"] / weighted).fillna(0.0)
            rate = values["state_fips"].map(rates).fillna(0.0).clip(0.0, 1.0)
        values["aca_take_up_rate"] = rate
        return values

    return table


def _sums(frame: Frame, *, acs: bool) -> dict[str, dict[str, float]]:
    """Per table and state: what one set of rows delivers (ACS: at most)."""

    units = frame.table("tax_unit")
    rows = units[spine_column("tax_unit")].eq(ACS_2024_1YR_SPINE).to_numpy()
    if not acs:
        rows = ~rows
    flags = (
        np.ones(len(units), dtype=bool)
        if acs
        else units[TAKE_UP]
        .where(pd.Series(rows, index=units.index), False)
        .astype(bool)
        .to_numpy()
    )
    values = _unit_values(frame, flags=flags | acs)
    chosen = rows & flags & values["is_aca_ptc_eligible"].to_numpy()
    result: dict[str, dict[str, float]] = {}
    for name, column in (
        (CMS, "tax_unit_weight"),
        (RETURNS, "household_weight"),
        (AMOUNT, "weighted_assigned_aca_ptc"),
    ):
        summed = (
            pd.Series(np.where(chosen, values[column], 0.0))
            .groupby(values["state_fips"].to_numpy())
            .sum()
        )
        result[name] = {str(state): float(value) for state, value in summed.items()}
    return result


def _targets(
    frame: Frame, *, fraction: float = 0.4, excess_state: str | None = None
) -> dict[str, pd.DataFrame]:
    """Counts whose donor residual is ``fraction`` of the ACS capacity.

    In ``excess_state`` every count sits below the donor contribution.
    """

    donor, capacity = _sums(frame, acs=False), _sums(frame, acs=True)
    tables = {}
    for name in (CMS, RETURNS, AMOUNT):
        rows = []
        for state in (f"{code:02d}" for code in STATES):
            base = donor[name].get(state, 0.0)
            target = (
                0.5 * base
                if state == excess_state
                else base + fraction * capacity[name].get(state, 0.0)
            )
            rows.append({"state_fips": state, "target": target})
        tables[name] = pd.DataFrame(rows)
    return tables


def _assign(frame: Frame, tables=None, *, seed: int = 7, calls=None, **kwargs):
    return with_acs_local_aca_take_up(
        frame,
        seed=seed,
        target_tables=_targets(frame, **kwargs) if tables is None else tables,
        aca_tax_units=_stub(calls),
    )


@pytest.fixture(scope="module")
def frame() -> Frame:
    return _frame()


@pytest.fixture(scope="module")
def assigned(frame):
    return _assign(frame, excess_state="48")


def _acs_units(frame: Frame) -> np.ndarray:
    return (
        frame.table("tax_unit")[spine_column("tax_unit")]
        .eq(ACS_2024_1YR_SPINE)
        .to_numpy()
    )


def test_each_residual_is_its_count_minus_the_donor_rows_floored_at_zero(
    frame, assigned
) -> None:
    _, receipt = assigned
    donor = _sums(frame, acs=False)
    tables = _targets(frame, excess_state="48")
    assert receipt["issue"] == ACS_LOCAL_ACA_TAKE_UP_ISSUE
    for name in (CMS, RETURNS, AMOUNT):
        rows = {row["state_fips"]: row for row in receipt["tables"][name]["states"]}
        for state, target in zip(
            tables[name]["state_fips"], tables[name]["target"], strict=True
        ):
            row = rows[state]
            assert row["target"] == pytest.approx(target)
            assert row["donor"] == pytest.approx(donor[name][state])
            assert row["residual"] == pytest.approx(
                max(target - donor[name][state], 0.0)
            )
        # The donor rows alone exceed Texas's counts: no ACS unit takes up.
        assert rows["48"]["residual"] == 0.0
        assert rows["48"]["donor_excess"] > 0
        assert rows["48"]["acs_delivered"] == 0.0


def test_the_soi_return_count_binds_and_the_cms_people_are_reported(assigned) -> None:
    _, receipt = assigned
    assert receipt["binding_table"] == RETURNS
    assert receipt["tables"][RETURNS]["binds"] is True
    assert receipt["tables"][CMS]["binds"] is False
    for row in receipt["tables"][RETURNS]["states"]:
        if row["residual"] <= 0:
            continue
        # The greedy fill stops on the first unit that crosses the residual.
        assert row["acs_capacity"] > row["residual"]
        assert row["residual"] <= row["acs_delivered"]
        assert row["acs_delivered"] < row["residual"] + row["acs_max_unit_weight"]
    composition = receipt["composition"]
    assert composition["people_table"] == CMS
    assert composition["returns_table"] == RETURNS
    assert composition["acs"] > 1.0
    assert receipt["stop_after"] == "support_clip"


def test_only_missing_acs_cells_are_filled_and_donor_cells_are_kept(
    frame, assigned
) -> None:
    result, receipt = assigned
    acs = _acs_units(frame)
    before = frame.table("tax_unit")
    after = result.table("tax_unit")
    for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        assert after.loc[~acs, column].tolist() == before.loc[~acs, column].tolist()
        assert after.loc[acs, column].notna().all()
    flags = after.loc[acs, TAKE_UP].astype(bool)
    assert flags.any() and not flags.all()
    assert receipt["take_up"]["filled_rows"] == int(acs.sum())
    assert receipt["take_up"]["unfilled_acs_rows"] == 0
    assert receipt["plan_ratio"]["unfilled_acs_rows"] == 0
    # Only units eligible with a positive credit take up.
    values = _unit_values(result, flags=np.ones(len(after), dtype=bool))
    eligible = values["is_aca_ptc_eligible"].to_numpy()
    assert not (flags.to_numpy() & ~eligible[acs]).any()


def test_takers_get_the_clipped_premium_ratio_and_others_the_benchmark(
    frame, assigned
) -> None:
    result, _ = assigned
    acs = _acs_units(frame)
    units = result.table("tax_unit").loc[acs]
    values = _unit_values(result, flags=np.ones(len(_acs_units(result)), dtype=bool))
    values = values.loc[acs]
    takers = units[TAKE_UP].astype(bool).to_numpy()
    expected = np.clip(
        (
            values["health_insurance_premiums_without_medicare_part_b"]
            + values["assigned_aca_ptc"]
        )
        / values["slcsp"],
        0.5,
        1.5,
    ).to_numpy()
    ratio = units[RATIO].to_numpy(dtype=float)
    np.testing.assert_allclose(ratio[takers], expected[takers])
    assert (ratio[~takers] == 1.0).all()


def test_the_prepass_reads_stored_donor_flags_then_acs_defaults(frame) -> None:
    calls: list = []
    tables = _targets(frame)
    _assign(frame, tables, calls=calls)
    donor_call, acs_call = calls
    donor_units = donor_call["units"]
    assert donor_units[spine_column("tax_unit")].eq(ASEC_PUF_DONOR_SPINE).all()
    stored = frame.table("tax_unit").set_index("tax_unit_id")[TAKE_UP]
    assert (
        donor_units[TAKE_UP].tolist()
        == stored.reindex(donor_units["tax_unit_id"]).tolist()
    )
    pd.testing.assert_frame_equal(donor_call["tables"][CMS], tables[CMS])
    acs_units = acs_call["units"]
    assert acs_units[spine_column("tax_unit")].eq(ACS_2024_1YR_SPINE).all()
    # A missing cell reads the engine default, so the credit is the
    # credit before the filing gate.
    assert acs_units[TAKE_UP].eq(True).all()
    assert acs_units[RATIO].eq(1.0).all()
    # And the ACS pass is given the residuals, which its rate reads.
    assert acs_call["tables"][RETURNS]["target"].sum() < tables[RETURNS]["target"].sum()


def test_a_saturated_state_takes_up_every_eligible_unit(frame) -> None:
    tables = _targets(frame, fraction=3.0)
    result, receipt = _assign(frame, tables)
    for row in receipt["tables"][RETURNS]["states"]:
        assert row["capacity_shortfall"] > 0
        assert row["acs_delivered"] == pytest.approx(row["acs_capacity"])
    acs = _acs_units(frame)
    values = _unit_values(result, flags=np.ones(len(acs), dtype=bool))
    eligible = values["is_aca_ptc_eligible"].to_numpy()[acs]
    assert result.table("tax_unit").loc[acs, TAKE_UP].astype(bool)[eligible].all()
    gate = acs_local_aca_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures
    assert {row["status"] for row in gate.details["binding_states"]} == {"saturated"}


def test_stored_acs_cells_are_kept_true_as_an_anchor_false_out_of_the_stage(
    frame,
) -> None:
    acs = _acs_units(frame)
    units = frame.table("tax_unit").copy()
    values = _unit_values(frame, flags=np.ones(len(units), dtype=bool))
    eligible = values["is_aca_ptc_eligible"].to_numpy()
    acs_eligible = np.flatnonzero(acs & eligible)
    anchor, excluded = acs_eligible[:5], acs_eligible[5:10]
    flags = units[TAKE_UP].astype(object)
    flags.iloc[anchor] = True
    flags.iloc[excluded] = False
    units[TAKE_UP] = flags
    ratio = units[RATIO].astype(float)
    ratio.iloc[excluded] = 0.9
    units[RATIO] = ratio
    stored = _with_tax_unit(frame, units)

    result, receipt = _assign(stored, _targets(frame, fraction=0.0))

    after = result.table("tax_unit")
    assert after[TAKE_UP].iloc[anchor].astype(bool).all()
    assert not after[TAKE_UP].iloc[excluded].astype(bool).any()
    assert after[RATIO].iloc[excluded].tolist() == [0.9] * 5
    assert receipt["anchor"]["stored_true_tax_units"] == 5
    assert receipt["take_up"]["preserved_acs_rows"] == 10
    assert receipt["take_up"]["stored_false_rows_kept"] == 5
    # Every residual is 0, so only the anchors take up.
    assert int(after.loc[acs, TAKE_UP].astype(bool).sum()) == 5


def test_draws_key_on_the_acs_record_not_on_the_tax_unit_id(frame) -> None:
    # The CMS table alone, where the draw decides the takers: with the SOI
    # tables the joint step orders units by their credit's distance from the
    # residual average and reads the draw only to break ties, as on the donor.
    tables = {CMS: _targets(frame)[CMS]}
    result, _ = _assign(frame, tables)
    acs = _acs_units(frame)
    shift = 10_000
    units = frame.table("tax_unit").copy()
    units.loc[acs, "tax_unit_id"] += shift
    person = frame.table("person").copy()
    moved = person[spine_column("person")].eq(ACS_2024_1YR_SPINE)
    person.loc[moved, "person_tax_unit_id"] += shift
    renumbered = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "tax_unit": units,
            "person": person,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    again, _ = _assign(renumbered, tables)
    for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        assert (
            again.table("tax_unit").loc[acs, column].tolist()
            == result.table("tax_unit").loc[acs, column].tolist()
        )
    # Another seed draws other takers.
    other, _ = _assign(frame, tables, seed=8)
    assert (
        other.table("tax_unit").loc[acs, TAKE_UP].tolist()
        != result.table("tax_unit").loc[acs, TAKE_UP].tolist()
    )


def test_a_recorded_assignment_reproduces_its_digest_or_is_refused(
    frame, assigned
) -> None:
    result, receipt = assigned
    recorded = acs_local_aca_take_up_assignment(result)
    assert recorded.columns.tolist() == ["tax_unit_id", *ACS_LOCAL_ACA_TAKE_UP_COLUMNS]
    applied = with_recorded_acs_local_aca_take_up(
        frame, recorded, assigned_sha256=receipt["assigned_sha256"]
    )
    for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        assert (
            applied.table("tax_unit")[column].tolist()
            == result.table("tax_unit")[column].tolist()
        )
    with pytest.raises(ValueError, match="digest"):
        with_recorded_acs_local_aca_take_up(frame, recorded, assigned_sha256="0" * 64)
    changed = recorded.copy()
    changed.loc[changed.index[0], RATIO] = changed[RATIO].iloc[0] + 0.01
    with pytest.raises(ValueError, match="digest"):
        with_recorded_acs_local_aca_take_up(
            frame, changed, assigned_sha256=receipt["assigned_sha256"]
        )
    with pytest.raises(ValueError, match="in order"):
        with_recorded_acs_local_aca_take_up(
            frame,
            recorded.iloc[::-1].reset_index(drop=True),
            assigned_sha256=receipt["assigned_sha256"],
        )
    flipped = result.table("tax_unit").copy()
    acs = _acs_units(result)
    first = np.flatnonzero(acs)[0]
    flipped.loc[first, TAKE_UP] = not bool(flipped.loc[first, TAKE_UP])
    with pytest.raises(ValueError, match="disagree"):
        with_recorded_acs_local_aca_take_up(
            _with_tax_unit(result, flipped),
            recorded,
            assigned_sha256=receipt["assigned_sha256"],
        )
    with pytest.raises(ValueError, match="not complete"):
        acs_local_aca_take_up_assignment(frame)


def test_the_gate_passes_the_materialize_assignment_and_reports_its_states(
    assigned,
) -> None:
    result, receipt = assigned
    gate = acs_local_aca_take_up_signal_gate(result, receipt=receipt)
    assert gate.name == ACS_LOCAL_ACA_TAKE_UP_GATE_NAME
    assert gate.passed, gate.failures
    statuses = {
        row["state_fips"]: row["status"] for row in gate.details["binding_states"]
    }
    assert statuses["48"] == "donor_meets_count"
    assert {statuses["06"], statuses["36"]} == {"within_tolerance"}


def test_the_gate_fails_a_tampered_receipt_and_incomplete_cells(
    frame, assigned
) -> None:
    result, receipt = assigned
    tampered = copy.deepcopy(receipt)
    tampered["tables"][AMOUNT]["states"][0]["residual"] += 1_000.0
    gate = acs_local_aca_take_up_signal_gate(result, receipt=tampered)
    assert not gate.passed
    assert any("not the donor residual" in line for line in gate.failures)

    missed = copy.deepcopy(receipt)
    row = next(
        row for row in missed["tables"][RETURNS]["states"] if row["residual"] > 0
    )
    row["acs_delivered"] = 0.5 * row["residual"]
    gate = acs_local_aca_take_up_signal_gate(result, receipt=missed)
    assert any("misses the donor residual" in line for line in gate.failures)

    gate = acs_local_aca_take_up_signal_gate(frame, receipt=receipt)
    assert not gate.passed
    assert any("missing take up rows" in line for line in gate.failures)

    gate = acs_local_aca_take_up_signal_gate(result, receipt=None)
    assert any("No acs_local_aca_take_up receipt" in line for line in gate.failures)


def test_without_the_soi_tables_the_cms_residual_binds(frame) -> None:
    tables = {CMS: _targets(frame)[CMS]}
    result, receipt = _assign(frame, tables)
    assert receipt["binding_table"] == CMS
    assert receipt["composition"] is None
    for row in receipt["tables"][CMS]["states"]:
        # The greedy calibration adds or removes drawn units until it
        # crosses the residual, so it lands within one unit's weight.
        assert abs(row["acs_delivered"] - row["residual"]) < row["acs_max_unit_weight"]
    gate = acs_local_aca_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures


def test_malformed_inputs_are_refused(frame) -> None:
    tables = _targets(frame)
    with pytest.raises(ValueError, match="reads target table"):
        _assign(frame, {**tables, "cms_aca_other": tables[CMS]})
    repeated = pd.concat([tables[CMS], tables[CMS]], ignore_index=True)
    with pytest.raises(ValueError, match="repeats state"):
        _assign(frame, {**tables, CMS: repeated})
    household = frame.table("household").copy()
    household.loc[household["SERIALNO"].notna().idxmax(), "SERIALNO"] = np.nan
    unkeyed = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "household": household,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(ValueError, match="SERIALNO"):
        _assign(unkeyed, tables)
    units = frame.table("tax_unit").drop(columns=[TAKE_UP])
    with pytest.raises(ValueError, match="universal take-up"):
        _assign(_with_tax_unit(frame, units), tables)


def _with_reported_anchor(frame: Frame, units: set) -> Frame:
    """The frame with the anchor column True for every member of ``units``."""

    person = frame.table("person").copy()
    person[ANCHOR] = person["person_tax_unit_id"].isin(units).to_numpy()
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )


def _eligible_acs_units(frame: Frame) -> np.ndarray:
    acs = _acs_units(frame)
    values = _unit_values(frame, flags=np.ones(len(acs), dtype=bool))
    eligible = values["is_aca_ptc_eligible"].to_numpy()
    return frame.table("tax_unit")["tax_unit_id"].to_numpy()[acs & eligible]


def _takers(result: Frame) -> set:
    units = result.table("tax_unit")
    acs = _acs_units(result)
    flags = units.loc[acs, TAKE_UP].astype(bool).to_numpy()
    return set(units.loc[acs, "tax_unit_id"].to_numpy()[flags])


def test_the_manifest_anchor_column_anchors_acs_units(frame) -> None:
    """The stage reads the manifest's own anchor column where staging writes
    it on ACS persons (as the fiscal lane does), so a staged subsidy item
    needs no new rule here. A unit stored False stays out."""

    candidates = _eligible_acs_units(frame)
    anchored = set(candidates[::3])
    excluded = candidates[1]
    units = frame.table("tax_unit").copy()
    flags = units[TAKE_UP].astype(object)
    flags[units["tax_unit_id"].eq(excluded).to_numpy()] = False
    units[TAKE_UP] = flags
    staged = _with_reported_anchor(_with_tax_unit(frame, units), anchored | {excluded})

    # Every residual is 0, so only the anchors take up.
    result, receipt = _assign(staged, _targets(frame, fraction=0.0))

    assert _takers(result) == anchored
    assert receipt["anchor"]["column"] == ANCHOR
    assert receipt["anchor"]["column_on_acs_persons"] is True
    assert receipt["anchor"]["reported_tax_units"] == len(anchored) + 1
    assert receipt["anchor"]["anchored_tax_units"] == len(anchored)
    for row in receipt["tables"][RETURNS]["states"]:
        assert row["residual"] == 0.0
        assert row["acs_delivered"] == pytest.approx(row["acs_anchored"])
        assert row["anchor_excess"] == pytest.approx(row["acs_anchored"])
    gate = acs_local_aca_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures


def test_anchors_above_a_residual_all_take_up_and_the_excess_is_reported(
    frame,
) -> None:
    """With an anchor, a state's anchored mass can exceed its residual: every
    anchor takes up (preserve_true_anchors), no other ACS unit does, and the
    receipt and the grade report the excess without failing."""

    anchored = set(_eligible_acs_units(frame)[::3])
    staged = _with_reported_anchor(frame, anchored)
    _measured, first = _assign(staged, _targets(frame, fraction=0.0))
    tables = {
        name: pd.DataFrame(
            {
                "state_fips": [row["state_fips"] for row in entry["states"]],
                "target": [
                    row["donor"] + 0.5 * row["acs_anchored"] for row in entry["states"]
                ],
            }
        )
        for name, entry in first["tables"].items()
    }

    result, receipt = _assign(staged, tables)

    assert _takers(result) == anchored
    rows = receipt["tables"][RETURNS]["states"]
    assert all(row["acs_anchored"] > 0 for row in rows)
    for row in rows:
        assert row["residual"] == pytest.approx(0.5 * row["acs_anchored"])
        assert row["anchor_excess"] == pytest.approx(0.5 * row["acs_anchored"])
    gate = acs_local_aca_take_up_signal_gate(result, receipt=receipt)
    assert gate.passed, gate.failures
    assert {row["status"] for row in gate.details["binding_states"]} == {
        "anchor_excess"
    }
