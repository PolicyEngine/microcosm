"""US take-up seeding stage tests (populace #312).

Covers the two seeded programs (TANF, EITC): deterministic seeding under a fixed
seed, the administrative rate reproduced within tolerance on a synthetic frame,
EITC's per-child rate honored, support-channel clone consistency, healing the
constant-True landmine, and the signal gate's failure modes (including a
prove-it-can-find-something check that the gate actually fails on a bad frame).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("policyengine_us")

from populace.build.us_runtime.take_up import (  # noqa: E402
    US_TAKE_UP_SHARE_BAND,
    us_take_up_participation_diagnostics,
    us_take_up_signal_gate,
    us_take_up_summary,
    with_us_take_up_inputs,
    write_us_take_up_participation_diagnostics,
)
from populace.build.us_runtime.take_up_contract import (  # noqa: E402
    TakeUpProgram,
    load_take_up_contract,
    seeded_take_up_programs,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights  # noqa: E402

TIME_PERIOD = 2024


def _program(variable: str) -> TakeUpProgram:
    return load_take_up_contract().program_map()[variable]


def _us_frame(
    *,
    n_units: int,
    children_pattern=None,
    weights: list[float] | None = None,
    source_identity: bool = False,
    spm_extra: dict | None = None,
    tax_unit_extra: dict | None = None,
) -> Frame:
    """Synthetic US frame: one adult per unit plus a pattern of children.

    ``children_pattern(u)`` returns the child count for unit ``u`` (default:
    cycle 0,1,2,3). Every unit is its own household / tax_unit / spm_unit /
    family; each person is its own marital unit.
    """
    if children_pattern is None:
        children_pattern = lambda u: u % 4  # noqa: E731
    rows: list[dict] = []
    pid = 0
    for u in range(n_units):
        rows.append({"person_id": pid, "unit": u, "age": 40})
        pid += 1
        for _ in range(children_pattern(u)):
            rows.append({"person_id": pid, "unit": u, "age": 8})
            pid += 1
    person = pd.DataFrame(rows)
    for entity in ("household", "tax_unit", "spm_unit", "family"):
        person[f"person_{entity}_id"] = person["unit"]
    person["person_marital_unit_id"] = person["person_id"]
    if source_identity:
        # All members of a unit share the unit's source identity.
        person["source_year"] = 2023
        person["source_household_id"] = person["unit"]
        person["source_person_id"] = person["person_id"]
    person = person.drop(columns=["unit"])
    unit_ids = np.unique(person["person_household_id"])
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unit_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unit_ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unit_ids}),
        "family": pd.DataFrame({"family_id": unit_ids}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.unique(person["person_marital_unit_id"])}
        ),
    }
    for column, values in (spm_extra or {}).items():
        tables["spm_unit"][column] = values
    for column, values in (tax_unit_extra or {}).items():
        tables["tax_unit"][column] = values
    w = weights if weights is not None else [1.0] * len(unit_ids)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(w, dtype=np.float64), kind=WeightKind.DESIGN
            )
        },
    )


class TestSeededProgramsAreSourced:
    def test_only_tanf_and_eitc_are_seeded(self) -> None:
        seeded = {p.variable for p in seeded_take_up_programs()}
        assert seeded == {"takes_up_tanf_if_eligible", "takes_up_eitc"}

    def test_each_seeded_program_cites_an_administrative_source(self) -> None:
        for program in seeded_take_up_programs():
            assert program.rate.get("source"), program.variable
            assert str(program.rate["status"]).startswith("sourced")


class TestSeeding:
    def test_writes_both_flags_onto_their_entities(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=100), seed=0, time_period=TIME_PERIOD
        )
        assert "takes_up_tanf_if_eligible" in frame.table("spm_unit").columns
        assert "takes_up_eitc" in frame.table("tax_unit").columns

    def test_seed_is_deterministic(self) -> None:
        base = _us_frame(n_units=300)
        first = with_us_take_up_inputs(base, seed=7, time_period=TIME_PERIOD)
        second = with_us_take_up_inputs(base, seed=7, time_period=TIME_PERIOD)
        assert (
            first.table("spm_unit")["takes_up_tanf_if_eligible"].tolist()
            == second.table("spm_unit")["takes_up_tanf_if_eligible"].tolist()
        )
        assert (
            first.table("tax_unit")["takes_up_eitc"].tolist()
            == second.table("tax_unit")["takes_up_eitc"].tolist()
        )

    def test_different_seed_changes_assignment(self) -> None:
        base = _us_frame(n_units=300)
        a = with_us_take_up_inputs(base, seed=1, time_period=TIME_PERIOD)
        b = with_us_take_up_inputs(base, seed=2, time_period=TIME_PERIOD)
        assert (
            a.table("spm_unit")["takes_up_tanf_if_eligible"].tolist()
            != b.table("spm_unit")["takes_up_tanf_if_eligible"].tolist()
        )

    def test_tanf_rate_reproduced_within_tolerance(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=8000), seed=0, time_period=TIME_PERIOD
        )
        share = us_take_up_summary(frame)["takes_up_tanf_if_eligible"]["take_up_share"]
        target = float(_program("takes_up_tanf_if_eligible").rate["value"])
        assert abs(share - target) < 0.02

    def test_eitc_overall_share_matches_irs_participation(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=8000), seed=0, time_period=TIME_PERIOD
        )
        share = us_take_up_summary(frame)["takes_up_eitc"]["take_up_share"]
        # IRS overall EITC participation is ~78%; the by-children mapping should
        # land there on an even child-count mix.
        assert 0.74 <= share <= 0.82

    def test_eitc_per_child_rate_is_honored(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=8000), seed=0, time_period=TIME_PERIOD
        )
        tax_unit = frame.table("tax_unit")
        person = frame.table("person")
        child_count = (
            person.assign(_c=(person["age"] < 19).astype(int))
            .groupby("person_tax_unit_id")["_c"]
            .sum()
        )
        tax_unit = tax_unit.assign(
            _k=tax_unit["tax_unit_id"].map(child_count).fillna(0).astype(int)
        )
        rates = _program("takes_up_eitc").rate["values_by_num_children"]
        expected = {0: rates["0"], 1: rates["1"], 2: rates["2"], 3: rates["3+"]}
        for k, want in expected.items():
            share = tax_unit.loc[tax_unit["_k"] == k, "takes_up_eitc"].mean()
            assert abs(share - want) < 0.03, f"{k}-child bin: {share} vs {want}"


class TestCloneConsistency:
    def test_source_identity_clones_agree(self) -> None:
        # Two spm units that share a source identity must draw the same flag.
        frame = _us_frame(n_units=40, source_identity=True)
        # Force units 0 and 1 to share the exact source identity (a support
        # clone of one origin unit split across two records).
        person = frame.table("person").copy()
        clone_mask = person["person_spm_unit_id"].isin([0, 1])
        person.loc[clone_mask, "source_year"] = 2023
        person.loc[clone_mask, "source_household_id"] = 999
        person.loc[clone_mask, "source_person_id"] = 7
        tables = {e: frame.table(e).copy() for e in frame.entities}
        tables["person"] = person
        clone_frame = Frame(
            tables,
            US_SCHEMA,
            {"household": frame.weights_for("household")},
        )
        seeded = with_us_take_up_inputs(clone_frame, seed=3, time_period=TIME_PERIOD)
        tanf = seeded.table("spm_unit").set_index("spm_unit_id")[
            "takes_up_tanf_if_eligible"
        ]
        assert bool(tanf.loc[0]) == bool(tanf.loc[1])


class TestIdempotenceAndHealing:
    def test_signal_column_passes_through_untouched(self) -> None:
        derived = with_us_take_up_inputs(
            _us_frame(n_units=200), seed=0, time_period=TIME_PERIOD
        )
        again = with_us_take_up_inputs(derived, seed=99, time_period=TIME_PERIOD)
        assert again is derived

    def test_constant_true_landmine_is_recomputed(self) -> None:
        # The published failure mode: takes_up constant True for every unit.
        frame = _us_frame(
            n_units=400,
            spm_extra={"takes_up_tanf_if_eligible": [True] * 400},
            tax_unit_extra={"takes_up_eitc": [True] * 400},
        )
        healed = with_us_take_up_inputs(frame, seed=0, time_period=TIME_PERIOD)
        assert healed.table("spm_unit")["takes_up_tanf_if_eligible"].nunique() == 2
        assert healed.table("tax_unit")["takes_up_eitc"].nunique() == 2

    def test_partial_signal_program_is_left_but_missing_is_filled(self) -> None:
        # TANF already carries signal; EITC is missing. Only EITC is computed.
        base = with_us_take_up_inputs(
            _us_frame(n_units=300), seed=0, time_period=TIME_PERIOD
        )
        # Drop the EITC column to simulate a partial frame.
        tables = {e: base.table(e).copy() for e in base.entities}
        tanf_before = tables["spm_unit"]["takes_up_tanf_if_eligible"].tolist()
        tables["tax_unit"] = tables["tax_unit"].drop(columns=["takes_up_eitc"])
        partial = Frame(tables, US_SCHEMA, {"household": base.weights_for("household")})
        refilled = with_us_take_up_inputs(partial, seed=0, time_period=TIME_PERIOD)
        assert "takes_up_eitc" in refilled.table("tax_unit").columns
        # TANF (already signal-carrying) is unchanged.
        assert (
            refilled.table("spm_unit")["takes_up_tanf_if_eligible"].tolist()
            == tanf_before
        )


class TestGate:
    def test_plausible_assignment_passes(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=4000), seed=0, time_period=TIME_PERIOD
        )
        gate = us_take_up_signal_gate(frame)
        assert gate.passed, gate.failures

    def test_missing_column_fails(self) -> None:
        gate = us_take_up_signal_gate(_us_frame(n_units=50))
        assert not gate.passed
        assert any("missing" in failure for failure in gate.failures)

    def test_constant_true_fails(self) -> None:
        frame = _us_frame(
            n_units=50,
            spm_extra={"takes_up_tanf_if_eligible": [True] * 50},
            tax_unit_extra={"takes_up_eitc": [True] * 50},
        )
        gate = us_take_up_signal_gate(frame)
        assert not gate.passed
        assert any("constant" in failure for failure in gate.failures)

    def test_share_outside_band_fails(self) -> None:
        # An all-False TANF column (0% take-up) is below the plausibility band.
        frame = _us_frame(
            n_units=50,
            spm_extra={
                "takes_up_tanf_if_eligible": [True] + [False] * 49,
            },
            tax_unit_extra={"takes_up_eitc": [True] * 25 + [False] * 25},
        )
        gate = us_take_up_signal_gate(frame)
        assert not gate.passed
        assert any("take-up share" in failure for failure in gate.failures)

    def test_summary_reports_administrative_provenance(self) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=400), seed=0, time_period=TIME_PERIOD
        )
        summary = us_take_up_summary(frame)
        for variable in ("takes_up_tanf_if_eligible", "takes_up_eitc"):
            assert summary[variable]["administrative_source"]
            assert summary[variable]["administrative_agency"]
            assert summary[variable]["share_band"] == list(
                US_TAKE_UP_SHARE_BAND[variable]
            )


class TestParticipationDiagnostics:
    def test_every_contract_flag_appears_with_honest_status(self) -> None:
        # The artifact must cover the WHOLE contract, not just the seeded
        # programs — the unseeded rows are the record that those flags still
        # ship at the engine's universal-take-up default.
        frame = with_us_take_up_inputs(
            _us_frame(n_units=400), seed=0, time_period=TIME_PERIOD
        )
        payload = us_take_up_participation_diagnostics(frame)
        contract = load_take_up_contract()
        by_variable = {row["variable"]: row for row in payload["programs"]}
        assert set(by_variable) == {p.variable for p in contract.programs}
        for variable in ("takes_up_tanf_if_eligible", "takes_up_eitc"):
            row = by_variable[variable]
            assert row["seeded"] is True
            rate = row["administrative_rate"]
            # TANF carries a scalar rate; EITC a by-child-count mapping.
            if isinstance(rate, dict):
                assert rate and all(0.0 < float(v) <= 1.0 for v in rate.values())
            else:
                assert 0.0 < float(rate) <= 1.0
            assert row["administrative_source"]
            assert 0.0 < row["take_up_share"] < 1.0
        unseeded = [row for row in payload["programs"] if not row["seeded"]]
        assert unseeded, "diagnostics must not declare the class fully repaired"
        assert all("ships_at_engine_default" in row for row in unseeded)
        assert payload["seeded_program_count"] == len(seeded_take_up_programs())
        assert payload["gate"]["passed"] is True

    def test_writer_round_trips_strict_json(self, tmp_path) -> None:
        frame = with_us_take_up_inputs(
            _us_frame(n_units=200), seed=0, time_period=TIME_PERIOD
        )
        payload = us_take_up_participation_diagnostics(frame)
        path = write_us_take_up_participation_diagnostics(
            payload, tmp_path / "us_take_up_participation.json"
        )
        assert path.name == "us_take_up_participation.json"
        assert json.loads(path.read_text()) == json.loads(
            json.dumps(payload, allow_nan=False)
        )
