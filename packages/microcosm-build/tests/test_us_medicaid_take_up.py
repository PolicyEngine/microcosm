"""Medicaid take-up stage tests (microcosm #331).

Covers the count_calibrated contract treatment (validation that the doctrine
cannot be bypassed), the anchored count-calibration itself (anchors always
enroll, unsaturated states hit their CMS count to unit-weight granularity,
saturated states enroll every eligible and are classified saturated),
determinism and source-identity draw keying, the off-domain propensity
carried for eligibility-expanding reforms, the gate's failure modes (with
prove-it-can-find-something checks), and the builder helpers that compile the
CMS state target table and person state codes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("policyengine_us")

from microcosm.build.us_runtime.medicaid_take_up import (  # noqa: E402
    US_MEDICAID_ENROLLMENT_SUBSTITUTIONS,
    US_MEDICAID_ENROLLMENT_TARGET_ROLE,
    US_MEDICAID_TAKE_UP_ANCHOR,
    US_MEDICAID_TAKE_UP_VARIABLE,
    _stable_person_draws,
    apply_us_medicaid_enrollment_substitutions,
    us_medicaid_take_up_diagnostics,
    us_medicaid_take_up_gate,
    with_us_medicaid_take_up,
)
from microcosm.build.us_runtime.take_up import (  # noqa: E402
    us_take_up_participation_diagnostics,
)
from microcosm.build.us_runtime.take_up_contract import (  # noqa: E402
    count_calibrated_take_up_programs,
    load_take_up_contract,
    seeded_take_up_programs,
)
from microcosm.calibrate import TargetRegistry, TargetSpec  # noqa: E402
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights  # noqa: E402

UNIT_WEIGHT = 100.0


def _frame(n: int, *, anchor: np.ndarray, source_identity: bool = True) -> Frame:
    """Synthetic single-person-household US frame with a reported anchor."""
    person = pd.DataFrame(
        {
            "person_id": range(n),
            "age": [30 + (i % 40) for i in range(n)],
            US_MEDICAID_TAKE_UP_ANCHOR: np.asarray(anchor, dtype=bool),
        }
    )
    for entity in ("household", "tax_unit", "spm_unit", "family"):
        person[f"person_{entity}_id"] = person["person_id"]
    person["person_marital_unit_id"] = person["person_id"]
    if source_identity:
        person["source_year"] = 2023
        person["source_household_id"] = person["person_id"]
        person["source_person_id"] = person["person_id"]
    ids = person["person_id"].to_numpy()
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(values=np.full(n, UNIT_WEIGHT), kind=WeightKind.DESIGN)},
    )


def _scenario(n: int = 400, seed: int = 7):
    """Two-state scenario: state 01 unsaturated (0.7x), state 02 saturated."""
    rng = np.random.default_rng(0)
    anchor = rng.random(n) < 0.30
    frame = _frame(n, anchor=anchor)
    state = np.where(np.arange(n) % 2 == 0, "01", "02")
    eligible = rng.random(n) < 0.60
    eligible_weight_01 = UNIT_WEIGHT * ((state == "01") & eligible).sum()
    eligible_weight_02 = UNIT_WEIGHT * ((state == "02") & eligible).sum()
    targets = pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "target": [0.7 * eligible_weight_01, 1.2 * eligible_weight_02],
        }
    )
    result, diagnostics = with_us_medicaid_take_up(
        frame,
        is_medicaid_eligible=eligible,
        state_fips=state,
        state_targets=targets,
        seed=seed,
    )
    return frame, result, diagnostics, anchor, eligible, state, targets


class TestContractTreatment:
    def test_medicaid_is_count_calibrated_not_seeded(self) -> None:
        assert [p.variable for p in count_calibrated_take_up_programs()] == [
            "takes_up_medicaid_if_eligible",
            "takes_up_ssi_if_eligible",
        ]
        assert "takes_up_medicaid_if_eligible" not in {
            p.variable for p in seeded_take_up_programs()
        }

    def test_calibration_block_names_anchor_and_targets(self) -> None:
        program = load_take_up_contract().program_map()["takes_up_medicaid_if_eligible"]
        calibration = program.raw["calibration"]
        assert calibration["anchor"] == US_MEDICAID_TAKE_UP_ANCHOR
        assert calibration["targets"]

    def test_calibration_block_matches_the_stage_manifest(self) -> None:
        # The contract must describe what the release actually calibrates to:
        # reconcile its calibration block against the manifest stage's ops so
        # the inventory cannot silently drift from the stage (#331 review).
        from microcosm.build.us_runtime import US_SOURCE_MANIFEST

        program = load_take_up_contract().program_map()["takes_up_medicaid_if_eligible"]
        calibration = program.raw["calibration"]
        stage = US_SOURCE_MANIFEST.stage_map()["medicaid_take_up"]
        ops = {op.kind: op.parameters for op in stage.operations}
        assert list(calibration["targets"]) == list(
            ops["calibrate_binary_assignment"]["targets"]
        )
        assert (
            calibration["anchor"]
            == ops["assign_binary_from_rate"]["reported_true_anchor"]
            == ops["calibrate_binary_assignment"]["preserve_true_anchor"]
        )

    def _reload_with(self, monkeypatch, tmp_path: Path, mutated: dict) -> None:
        load_take_up_contract.cache_clear()
        path = tmp_path / "take_up_contract.json"
        path.write_text(json.dumps(mutated))
        monkeypatch.setattr(
            "microcosm.build.us_runtime.take_up_contract._contract_path",
            lambda: path,
        )

    @pytest.fixture()
    def base_table(self):
        from microcosm.build.us_runtime.take_up_contract import _contract_path

        raw = json.loads(_contract_path().read_text())
        yield raw
        load_take_up_contract.cache_clear()

    def test_count_calibrated_without_calibration_block_fails(
        self, monkeypatch, tmp_path, base_table
    ) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_medicaid_if_eligible":
                program.pop("calibration")
        self._reload_with(monkeypatch, tmp_path, mutated)
        with pytest.raises(ValueError, match="calibration"):
            load_take_up_contract()

    def test_count_calibrated_with_sourced_rate_fails(
        self, monkeypatch, tmp_path, base_table
    ) -> None:
        # A sourced rate means the program belongs in `seed`; carrying one on a
        # count_calibrated entry is a misclassification the loader must refuse.
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_medicaid_if_eligible":
                program["rate"] = {
                    "value": 0.8,
                    "source": "https://example.gov",
                    "status": "sourced_administrative",
                }
        self._reload_with(monkeypatch, tmp_path, mutated)
        with pytest.raises(ValueError, match="sourced rate"):
            load_take_up_contract()


class TestAssignment:
    def test_anchored_reporters_always_take_up(self) -> None:
        _, result, _, anchor, _, _, _ = _scenario()
        flag = result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        assert flag[anchor].all()

    def test_anchored_but_model_ineligible_reporters_keep_the_flag(self) -> None:
        # The anchor is deliberately unmasked by eligibility: the engine's
        # is_medicaid_eligible gate hides the flag at baseline, and forcing it
        # False would erase the observation.
        _, result, _, anchor, eligible, _, _ = _scenario()
        flag = result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        ineligible_reporters = anchor & ~eligible
        assert ineligible_reporters.any()
        assert flag[ineligible_reporters].all()

    def test_unsaturated_state_hits_cms_count_to_unit_weight(self) -> None:
        _, result, _, _, eligible, state, targets = _scenario()
        flag = result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        enrolled_weight = UNIT_WEIGHT * (flag & eligible & (state == "01")).sum()
        target = float(targets.loc[targets["state_fips"] == "01", "target"].iloc[0])
        assert abs(enrolled_weight - target) <= UNIT_WEIGHT

    def test_saturated_state_enrolls_every_eligible_and_is_classified(self) -> None:
        _, result, diagnostics, _, eligible, state, _ = _scenario()
        flag = result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        assert flag[eligible & (state == "02")].all()
        assert diagnostics["saturated_states"] == ["02"]

    def test_rerun_is_bit_reproducible(self) -> None:
        frame, result, _, _, eligible, state, targets = _scenario(seed=7)
        again, _ = with_us_medicaid_take_up(
            frame,
            is_medicaid_eligible=eligible,
            state_fips=state,
            state_targets=targets,
            seed=7,
        )
        assert (
            result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
            == again.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        ).all()

    def test_draws_key_on_source_identity(self) -> None:
        # Support-channel clones of one source person share the draw, so the
        # Bernoulli phase treats them identically and they are adjacent in the
        # greedy fill order.
        table = pd.DataFrame(
            {
                "person_id": [0, 1],
                "source_year": [2023, 2023],
                "source_household_id": [5, 5],
                "source_person_id": [9, 9],
            }
        )
        draws = _stable_person_draws(table, seed=3)
        assert draws[0] == draws[1]

    def test_off_domain_persons_carry_a_propensity(self) -> None:
        # Non-anchored, model-ineligible persons keep a draw-based flag at the
        # state fill rate so eligibility-expanding reforms do not inherit a
        # hard-coded zero response.
        _, result, _, anchor, eligible, _, _ = _scenario()
        flag = result.table("person")[US_MEDICAID_TAKE_UP_VARIABLE].to_numpy()
        off_domain = ~anchor & ~eligible
        assert off_domain.any()
        assert flag[off_domain].any()
        assert not flag[off_domain].all()

    def test_missing_anchor_column_raises(self) -> None:
        frame = _frame(10, anchor=np.zeros(10, dtype=bool))
        person = frame.table("person").drop(columns=[US_MEDICAID_TAKE_UP_ANCHOR])
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = person
        stripped = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        with pytest.raises(ValueError, match=US_MEDICAID_TAKE_UP_ANCHOR):
            with_us_medicaid_take_up(
                stripped,
                is_medicaid_eligible=np.ones(10, dtype=bool),
                state_fips=np.full(10, "01"),
                state_targets=pd.DataFrame({"state_fips": ["01"], "target": [500.0]}),
                seed=0,
            )

    def test_misaligned_inputs_raise(self) -> None:
        frame = _frame(10, anchor=np.zeros(10, dtype=bool))
        with pytest.raises(ValueError, match="align"):
            with_us_medicaid_take_up(
                frame,
                is_medicaid_eligible=np.ones(4, dtype=bool),
                state_fips=np.full(10, "01"),
                state_targets=pd.DataFrame({"state_fips": ["01"], "target": [500.0]}),
                seed=0,
            )

    def test_malformed_targets_raise(self) -> None:
        frame = _frame(10, anchor=np.zeros(10, dtype=bool))
        with pytest.raises(ValueError, match="state_fips"):
            with_us_medicaid_take_up(
                frame,
                is_medicaid_eligible=np.ones(10, dtype=bool),
                state_fips=np.full(10, "01"),
                state_targets=pd.DataFrame({"state": ["01"], "count": [500.0]}),
                seed=0,
            )

    def test_empty_targets_raise_early(self) -> None:
        # An empty feed must refuse to run, not silently ship anchored-only
        # enrollment or surface as a deep missing-value-column error.
        frame = _frame(10, anchor=np.zeros(10, dtype=bool))
        with pytest.raises(ValueError, match="non-empty"):
            with_us_medicaid_take_up(
                frame,
                is_medicaid_eligible=np.ones(10, dtype=bool),
                state_fips=np.full(10, "01"),
                state_targets=pd.DataFrame(),
                seed=0,
            )


class TestGate:
    def test_passes_on_healthy_assignment(self) -> None:
        _, _, diagnostics, _, _, _, _ = _scenario()
        gate = us_medicaid_take_up_gate(diagnostics)
        assert gate.passed, gate.failures

    def _assigned_table(
        self,
        *,
        flag: np.ndarray,
        eligible: np.ndarray,
        anchor: np.ndarray,
        state: np.ndarray,
    ) -> pd.DataFrame:
        n = len(flag)
        return pd.DataFrame(
            {
                "person_id": range(n),
                US_MEDICAID_TAKE_UP_VARIABLE: flag,
                "is_medicaid_eligible": eligible,
                US_MEDICAID_TAKE_UP_ANCHOR: anchor,
                "person_weight": np.full(n, UNIT_WEIGHT),
                "state_fips": state,
            }
        )

    def test_fails_on_the_universal_take_up_landmine(self) -> None:
        # All eligibles enrolled while the CMS count sits well below eligible
        # weight: exactly the #170 degeneracy this stage exists to heal.
        n = 100
        eligible = np.ones(n, dtype=bool)
        table = self._assigned_table(
            flag=np.ones(n, dtype=bool),
            eligible=eligible,
            anchor=np.zeros(n, dtype=bool),
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame(
            {"state_fips": ["01"], "target": [0.5 * n * UNIT_WEIGHT]}
        )
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("landmine" in failure for failure in gate.failures)

    def test_fails_when_a_state_has_no_target(self) -> None:
        n = 40
        table = self._assigned_table(
            flag=np.zeros(n, dtype=bool),
            eligible=np.ones(n, dtype=bool),
            anchor=np.zeros(n, dtype=bool),
            state=np.where(np.arange(n) % 2 == 0, "01", "03"),
        )
        targets = pd.DataFrame({"state_fips": ["01"], "target": [500.0]})
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        assert diagnostics["states_without_targets"] == ["03"]
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("03" in failure for failure in gate.failures)

    def test_fails_when_a_state_has_a_count_but_zero_eligible_weight(self) -> None:
        # An eligibility-feed collapse classifies as saturated (any positive
        # count exceeds zero eligible weight), which would skip the count
        # checks and ship anchored-only enrollment silently. The gate must
        # treat that saturation as a division artifact and fail loudly.
        n = 40
        table = self._assigned_table(
            flag=np.zeros(n, dtype=bool),
            eligible=np.zeros(n, dtype=bool),
            anchor=np.zeros(n, dtype=bool),
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame({"state_fips": ["01"], "target": [500.0]})
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        assert diagnostics["saturated_states"] == ["01"]
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("eligibility feed collapsed" in failure for failure in gate.failures)

    def test_fails_when_the_anchor_is_not_preserved(self) -> None:
        n = 100
        anchor = np.zeros(n, dtype=bool)
        anchor[:30] = True
        flag = np.zeros(n, dtype=bool)
        flag[:10] = True  # fewer enrolled than anchored: anchor was dropped
        table = self._assigned_table(
            flag=flag,
            eligible=np.ones(n, dtype=bool),
            anchor=anchor,
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame({"state_fips": ["01"], "target": [1000.0]})
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("anchor" in failure for failure in gate.failures)

    def test_fully_anchored_unsaturated_state_is_not_a_landmine(self) -> None:
        # Every modeled-eligible person reports Medicaid: anchors cannot be
        # removed, so enrollment == eligibility is the correct calibrated
        # answer even though the CMS count sits below eligible weight. The
        # landmine check must not abort the build for it (#331 review).
        n = 100
        anchor = np.ones(n, dtype=bool)
        table = self._assigned_table(
            flag=np.ones(n, dtype=bool),
            eligible=np.ones(n, dtype=bool),
            anchor=anchor,
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame({"state_fips": ["01"], "target": [60 * UNIT_WEIGHT]})
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        gate = us_medicaid_take_up_gate(diagnostics)
        assert gate.passed, gate.failures

    def test_full_enrollment_within_one_unit_weight_of_the_count_passes(self) -> None:
        # Greedy calibration overshoots by up to one person: a target within
        # one unit weight of eligible weight legitimately fills every
        # eligible person without being the landmine.
        n = 100
        table = self._assigned_table(
            flag=np.ones(n, dtype=bool),
            eligible=np.ones(n, dtype=bool),
            anchor=np.zeros(n, dtype=bool),
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame(
            {"state_fips": ["01"], "target": [(n - 0.5) * UNIT_WEIGHT]}
        )
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        gate = us_medicaid_take_up_gate(diagnostics)
        assert gate.passed, gate.failures

    def test_dropped_anchor_fails_even_in_a_saturated_state(self) -> None:
        # The anchor invariant is person-level and unconditional: a saturated
        # state must not hide a regression that flips anchored reporters off.
        n = 100
        anchor = np.zeros(n, dtype=bool)
        anchor[:30] = True
        flag = np.ones(n, dtype=bool)
        flag[:5] = False  # five anchored reporters lost the flag
        table = self._assigned_table(
            flag=flag,
            eligible=np.ones(n, dtype=bool),
            anchor=anchor,
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame(
            {"state_fips": ["01"], "target": [1.5 * n * UNIT_WEIGHT]}  # saturated
        )
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        assert diagnostics["saturated_states"] == ["01"]
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("anchor" in failure for failure in gate.failures)

    def test_anchor_mass_above_the_cms_count_is_the_floor_not_a_miss(self) -> None:
        # CPS reporting above the CMS snapshot in a state must not fail the
        # gate: anchors are preserved by design, so the anchor mass is the
        # reachable floor.
        n = 100
        anchor = np.zeros(n, dtype=bool)
        anchor[:80] = True
        flag = anchor.copy()
        table = self._assigned_table(
            flag=flag,
            eligible=np.ones(n, dtype=bool),
            anchor=anchor,
            state=np.full(n, "01"),
        )
        targets = pd.DataFrame({"state_fips": ["01"], "target": [60 * UNIT_WEIGHT]})
        diagnostics = us_medicaid_take_up_diagnostics(table, targets)
        gate = us_medicaid_take_up_gate(diagnostics)
        assert gate.passed, gate.failures


class TestParticipationDiagnosticsIntegration:
    def test_materialized_column_reports_live_surface(self) -> None:
        _, result, _, _, _, _, _ = _scenario()
        payload = us_take_up_participation_diagnostics(result)
        row = next(
            row
            for row in payload["programs"]
            if row["variable"] == US_MEDICAID_TAKE_UP_VARIABLE
        )
        assert row["count_calibrated"] is True
        assert row["ships_at_engine_default"] is False
        assert 0.0 < row["take_up_share"] < 1.0

    def test_missing_column_reports_engine_default_with_owner(self) -> None:
        frame = _frame(20, anchor=np.zeros(20, dtype=bool))
        payload = us_take_up_participation_diagnostics(frame)
        row = next(
            row
            for row in payload["programs"]
            if row["variable"] == US_MEDICAID_TAKE_UP_VARIABLE
        )
        assert row["count_calibrated"] is False
        assert row["ships_at_engine_default"] is True
        # The debt-ledger pointer must survive the count_calibrated branch.
        assert "populace#331" in row["scope_owner"]

    def test_materialized_share_is_labeled_with_its_universe(self) -> None:
        # The flag deliberately carries off-domain propensity Trues, so the
        # share over all persons must be labeled as such rather than read as
        # an enrollment or participation rate.
        _, result, _, _, _, _, _ = _scenario()
        payload = us_take_up_participation_diagnostics(result)
        row = next(
            row
            for row in payload["programs"]
            if row["variable"] == US_MEDICAID_TAKE_UP_VARIABLE
        )
        assert "off_domain_propensity" in row["share_universe"]


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestBuilderHelpers:
    def test_medicaid_target_table_filters_role_and_geography(self) -> None:
        builder = _load_builder_module()
        specs = (
            SimpleNamespace(
                name="cms_medicaid.month2024_12.state_enrollment.06.total_medicaid_enrollment@2024",
                value=12_000_000.0,
                metadata={
                    "ledger_geography_level": "state",
                    "state_fips": "6",
                    "target_role": "medicaid_enrollment",
                },
            ),
            SimpleNamespace(
                name="national-row",
                value=72_000_000.0,
                metadata={
                    "ledger_geography_level": "national",
                    "target_role": "medicaid_enrollment",
                },
            ),
            SimpleNamespace(
                name="chip-row",
                value=100_000.0,
                metadata={
                    "ledger_geography_level": "state",
                    "state_fips": "06",
                    "target_role": "chip_enrollment",
                },
            ),
        )
        table = builder._medicaid_source_target_table(specs)
        assert table["state_fips"].tolist() == ["06"]
        assert table["target"].tolist() == [12_000_000.0]

    def test_duplicate_state_target_rows_refuse_to_compile(self) -> None:
        # Duplicate state rows diverge downstream (calibration is last-row-
        # wins; rate/diagnostics/gate sum), so the table must refuse them.
        builder = _load_builder_module()
        spec = {
            "ledger_geography_level": "state",
            "state_fips": "06",
            "target_role": "medicaid_enrollment",
        }
        specs = (
            SimpleNamespace(name="row-1", value=12_000_000.0, metadata=dict(spec)),
            SimpleNamespace(name="row-2", value=11_000_000.0, metadata=dict(spec)),
        )
        with pytest.raises(RuntimeError, match="duplicate state rows"):
            builder._medicaid_source_target_table(specs)

    def test_person_state_fips_maps_households(self) -> None:
        builder = _load_builder_module()
        frame = _frame(4, anchor=np.zeros(4, dtype=bool))
        household = frame.table("household").copy()
        household["state_fips"] = [6, 6, 36, 36]
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["household"] = household
        with_states = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        assert builder._person_state_fips(with_states).tolist() == [
            "06",
            "06",
            "36",
            "36",
        ]


def _medicaid_state_spec(
    state_fips: str,
    value: float,
    *,
    source_period: str = "2024-12",
    substitution: bool = False,
) -> TargetSpec:
    """A CMS state ``medicaid_enrollment`` spec shaped like a real compiled one.

    Mirrors the metadata a ledger CMS enrollment fact carries through
    :mod:`microcosm.build.ledger_targets` and the US target-measure map
    (``materializer=policyengine_variable`` over ``medicaid_enrolled``, the
    state geography keys, and a per-fact content-hash key), so the register's
    template-cloning and the builder's target-table extraction are exercised
    against a faithful surface rather than a stub.
    """
    padded = state_fips.zfill(2)
    month = source_period.replace("-", "_")
    record_set_id = f"cms_medicaid.month{month}.state_enrollment"
    record_id = f"{record_set_id}.{padded}.total_medicaid_enrollment"
    name = f"{record_id}@2024"
    metadata = {
        "materializer": "policyengine_variable",
        "base_variable": "medicaid_enrolled",
        "measure_mode": "indicator_sum",
        "target_role": US_MEDICAID_ENROLLMENT_TARGET_ROLE,
        "source_period": source_period,
        "ledger_geography_level": "state",
        "ledger_geography_id": f"0400000US{padded}",
        "ledger_layout_record_set_id": record_set_id,
        "ledger_layout_groupby_dimension": "geography_state",
        "ledger_layout_groupby_value_id": padded,
        "ledger_source_record_id": record_id,
        "ledger_fact_period": source_period,
        # A per-fact content-hash key the register must drop rather than stamp
        # onto a neighbouring state's substituted spec.
        "ledger_fact_key": f"factkey-{padded}-{month}",
        "state_fips": padded,
    }
    if substitution:
        metadata["medicaid_enrollment_substitution"] = "true"
    return TargetSpec(
        name=name,
        entity="household",
        value=value,
        measure=name,
        period=2024,
        source="CMS Medicaid & CHIP monthly enrollment (April 2026 release)",
        family="cms_medicaid",
        metadata=metadata,
    )


def _medicaid_registry(
    state_values: dict[str, float], *, extra_specs: tuple[TargetSpec, ...] = ()
) -> TargetRegistry:
    """A US target registry of natural CMS medicaid_enrollment state specs."""
    specs = [
        _medicaid_state_spec(state_fips, value)
        for state_fips, value in state_values.items()
    ]
    specs.extend(extra_specs)
    return TargetRegistry(specs, country="us")


def _record_for(records, state_fips: str) -> dict:
    padded = state_fips.zfill(2)
    return next(record for record in records if record["state_fips"] == padded)


def _healthy_assigned(
    targets: dict[str, float], *, weight: float = 100.0, slack: int = 4
) -> pd.DataFrame:
    """A stage-output table where every state cleanly hits its CMS count.

    Each state gets ``target / weight`` anchored, enrolled, eligible persons
    (so the count is met and the anchor invariant holds) plus ``slack``
    eligible non-enrolled persons (so the state is unsaturated, taking the
    count-check branch rather than the saturation skip).
    """
    rows: list[dict[str, object]] = []
    person_id = 0
    for state_fips, target in targets.items():
        n_on, remainder = divmod(round(target), round(weight))
        assert remainder == 0, f"target {target} must be divisible by weight {weight}"
        for i in range(n_on + slack):
            enrolled = i < n_on
            rows.append(
                {
                    "person_id": person_id,
                    US_MEDICAID_TAKE_UP_VARIABLE: enrolled,
                    "is_medicaid_eligible": True,
                    US_MEDICAID_TAKE_UP_ANCHOR: enrolled,
                    "person_weight": weight,
                    "state_fips": state_fips,
                }
            )
            person_id += 1
    return pd.DataFrame(rows)


class TestReviewedSubstitutionRegister:
    """The microcosm#386 reviewed CMS Medicaid enrollment substitution register.

    RI's December-2024 CMS snapshot is unreported at source, so the natural
    FIPS-44 medicaid_enrollment spec is absent and the #334 count-calibration
    gate fails closed. The register maps RI to its nearest-prior reported month
    (November 2024 = 273,400) with cannot-rot semantics.
    """

    def test_pass_with_substitution(self) -> None:
        # Without RI's target the gate fails closed on FIPS 44 (the Build J
        # failure); the register supplies the cited nearest-prior-month count,
        # RI gains a target, and the same assignment now passes.
        builder = _load_builder_module()
        targets = {"06": 600.0, "36": 400.0}
        registry = _medicaid_registry(targets)

        augmented, records = apply_us_medicaid_enrollment_substitutions(registry)

        rhode_island = _record_for(records, "44")
        assert rhode_island["applied"] is True
        assert rhode_island["stale"] is False
        assert rhode_island["substitute_value"] == 273_400.0
        assert rhode_island["issue"] == "microcosm#386"

        before = builder._medicaid_source_target_table(registry.specs)
        after = builder._medicaid_source_target_table(augmented.specs)
        assert "44" not in set(before["state_fips"])
        assert "44" in set(after["state_fips"])
        assert (
            float(after.loc[after["state_fips"] == "44", "target"].iloc[0]) == 273_400.0
        )

        assigned = _healthy_assigned({**targets, "44": 273_400.0})
        missing = us_medicaid_take_up_diagnostics(assigned, before, substitutions=())
        assert "44" in missing["states_without_targets"]
        assert not us_medicaid_take_up_gate(missing).passed

        healed = us_medicaid_take_up_diagnostics(assigned, after, substitutions=records)
        assert healed["states_without_targets"] == []
        gate = us_medicaid_take_up_gate(healed)
        assert gate.passed, gate.failures
        # The applied record rides the release diagnostics artifact.
        assert healed["medicaid_enrollment_substitutions"][0]["state_fips"] == "44"

    def test_fails_without_substitution_on_a_genuinely_missing_state(self) -> None:
        # The register heals only its reviewed hole (RI): a different state with
        # no CMS target and no register entry must still fail the gate closed,
        # so the register cannot become a silent blanket bypass.
        builder = _load_builder_module()
        registry = _medicaid_registry({"06": 600.0})

        augmented, records = apply_us_medicaid_enrollment_substitutions(registry)

        target_table = builder._medicaid_source_target_table(augmented.specs)
        # Illinois (17) is genuinely missing from the feed and absent from the
        # register; RI (44) is only supplied by the register.
        assert set(target_table["state_fips"]) == {"06", "44"}
        assigned = _healthy_assigned({"06": 600.0, "17": 500.0})
        diagnostics = us_medicaid_take_up_diagnostics(
            assigned, target_table, substitutions=records
        )
        assert diagnostics["states_without_targets"] == ["17"]
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        assert any("17" in failure for failure in gate.failures)

    def test_cannot_rot_fails_when_the_backfilled_ri_fact_appears(self) -> None:
        # The moment CMS backfills a genuine RI December-2024 count, the natural
        # FIPS-44 spec reappears: the register marks the entry stale, injects
        # nothing (no duplicate target), and the gate fails so the substitution
        # cannot outlive its justification (#286 stale-exclusion doctrine).
        backfilled_ri = _medicaid_state_spec("44", 275_000.0, source_period="2024-12")
        registry = _medicaid_registry({"06": 600.0}, extra_specs=(backfilled_ri,))

        augmented, records = apply_us_medicaid_enrollment_substitutions(registry)

        rhode_island = _record_for(records, "44")
        assert rhode_island["stale"] is True
        assert rhode_island["applied"] is False
        # Nothing injected: the backfilled natural spec is the only RI spec.
        assert len(augmented) == len(registry)
        assert not any(
            spec.metadata.get("medicaid_enrollment_substitution") == "true"
            for spec in augmented.specs
        )

        # A healthy assignment (RI hits its backfilled count) still fails the
        # gate purely on the stale register entry.
        assigned = _healthy_assigned({"06": 600.0, "44": 275_000.0})
        target_table = pd.DataFrame(
            {"state_fips": ["06", "44"], "target": [600.0, 275_000.0]}
        )
        diagnostics = us_medicaid_take_up_diagnostics(
            assigned, target_table, substitutions=records
        )
        gate = us_medicaid_take_up_gate(diagnostics)
        assert not gate.passed
        stale_failures = [failure for failure in gate.failures if "stale" in failure]
        assert stale_failures
        assert any(
            "44" in failure
            and "microcosm#386" in failure
            and "cms_medicaid.month2024_11" in failure
            for failure in stale_failures
        )

    def test_substitution_metadata_reaches_the_compiled_spec(self) -> None:
        # The substituted value ships as a first-class compiled target whose
        # metadata carries the full provenance, so the calibration target
        # surface and release manifest render the substitution rather than a
        # silent value swap.
        registry = _medicaid_registry({"06": 600.0, "36": 400.0})

        augmented, _ = apply_us_medicaid_enrollment_substitutions(registry)

        injected = [
            spec
            for spec in augmented.specs
            if spec.metadata.get("medicaid_enrollment_substitution") == "true"
        ]
        assert len(injected) == 1
        spec = injected[0]
        assert spec.value == 273_400.0
        assert spec.metadata["state_fips"] == "44"
        assert spec.metadata["target_role"] == US_MEDICAID_ENROLLMENT_TARGET_ROLE
        assert spec.metadata["substitution_issue"] == "microcosm#386"
        assert spec.metadata["substituted_for_source_period"] == "2024-12"
        assert spec.metadata["source_period"] == "2024-11"
        assert (
            spec.metadata["ledger_source_record_id"]
            == "cms_medicaid.month2024_11.state_enrollment.ri.total_medicaid_enrollment"
        )
        assert "System Limitations" in spec.metadata["substitution_reason"]
        # The clone re-derives the state geography from the substitute record id
        # and drops the template state's per-fact content-hash key.
        assert spec.metadata["ledger_layout_groupby_value_id"] == "ri"
        assert "ledger_fact_key" not in spec.metadata
        # It keeps the template's materialization shape so it calibrates like a
        # natural per-state count.
        assert spec.metadata["materializer"] == "policyengine_variable"
        assert spec.metadata["base_variable"] == "medicaid_enrolled"

        # The provenance survives compilation into the calibration TargetSet
        # (what the solver and the diagnostics target surface consume).
        compiled = augmented.to_target_set()
        target = next(
            target
            for target in compiled
            if target.metadata.get("medicaid_enrollment_substitution") == "true"
        )
        assert target.value == 273_400.0
        assert target.metadata["substitution_issue"] == "microcosm#386"

    def test_register_pins_rhode_island_to_the_ledger_value(self) -> None:
        # Guard the register's single reviewed entry against silent drift: the
        # substituted fact id and value are the audited ledger source of record
        # (RI November 2024 = 273,400).
        assert len(US_MEDICAID_ENROLLMENT_SUBSTITUTIONS) == 1
        entry = US_MEDICAID_ENROLLMENT_SUBSTITUTIONS[0]
        assert entry.state_fips == "44"
        assert entry.substitute_value == 273_400.0
        assert entry.substitute_source_period == "2024-11"
        assert entry.substituted_for_source_period == "2024-12"
        assert (
            entry.substitute_source_record_id
            == "cms_medicaid.month2024_11.state_enrollment.ri.total_medicaid_enrollment"
        )
        assert entry.issue == "microcosm#386"

    def test_registry_without_medicaid_family_is_a_no_op(self) -> None:
        # A registry that carries no medicaid_enrollment state family at all
        # (a diagnostic run whose targets omit the CMS enrollment facts) has
        # nothing to substitute into: the register injects nothing and does not
        # raise. Medicaid-absence is the take-up stage's empty-target guard's
        # job, not the register's — so main() still runs on such a registry.
        registry = TargetRegistry(
            (
                TargetSpec(
                    name="amount",
                    entity="household",
                    measure="income",
                    value=100.0,
                    source="fixture",
                    metadata={"source_measure_id": "payment_amount"},
                ),
            ),
            country="us",
        )

        augmented, records = apply_us_medicaid_enrollment_substitutions(registry)

        assert len(augmented) == len(registry)
        rhode_island = _record_for(records, "44")
        assert rhode_island["applied"] is False
        assert rhode_island["stale"] is False
        assert not any(
            spec.metadata.get("medicaid_enrollment_substitution") == "true"
            for spec in augmented.specs
        )
