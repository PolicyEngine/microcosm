"""US release input-column coverage + reform-coverage smoke, isolated from PE-US.

microcosm #368. The five acceptance cases the brief pins:

1. a full required set present with signal passes;
2. a missing required column fails, named;
3. a required column present but degenerate (every value the engine default)
   fails without a reviewed exclusion;
4. a reviewed-exclusion column that has caught up (present with signal) is stale
   and fails (#286 cannot-rot);
5. a bound reform scoring ~$0 fails the reform-coverage smoke.

The frame is a real :class:`~microcosm.frame.Frame`; most tests use an engine stub
exposing only ``default_values`` (the surface the gate uses), and the simulation is
injected (with ``_build_reform`` monkeypatched). One optional regression reproduces
the nullable-boolean failure against the real PolicyEngine-US defaults and full US
entity schema. Separate tests assert the shipped manifest keeps the #368
red-by-design guarantee: the SSI countable-resource assets stay hard requirements
with no exclusion, and demoting one is rejected.
"""

from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.reform_coverage_smoke as smoke_module
from microcosm.build.us_runtime import (
    SSI_COUNTABLE_RESOURCE_ASSETS,
    US_CGD_ROUTE_REQUIRED_INPUTS,
    US_QBI_OUTPUT_COLUMNS,
    US_RELEASE_INPUT_COVERAGE_RESOURCE,
    ReformCoverageProbe,
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    assert_release_input_coverage_manifest_current,
    load_release_input_coverage_manifest,
    us_reform_coverage_smoke_gate,
    us_release_input_coverage_gate,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.release_input_coverage import (
    REFERENCE_ECPS_LAYER_RENAMES,
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_GENERATOR = (
    _REPO_ROOT / "tools" / "build_us_release_input_coverage_manifest.py"
)


def _person_frame(columns: dict[str, np.ndarray | pd.Series]) -> Frame:
    """A real single-household Frame carrying ``columns`` on the person table."""
    n = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.ones(n, dtype="int64"),
            **columns,
        }
    )
    household = pd.DataFrame({"household_id": np.asarray([1], dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=np.asarray([1000.0]), kind=WeightKind.DESIGN)},
    )


def _us_person_frame(columns: dict[str, np.ndarray | pd.Series]) -> Frame:
    """A full six-entity US Frame preserving pandas extension dtypes."""
    n = len(next(iter(columns.values())))
    person_columns: dict[str, object] = {
        US_SCHEMA.person_id_column: np.arange(n, dtype="int64"),
        **{
            US_SCHEMA.membership_column(entity): np.ones(n, dtype="int64")
            for entity in US_SCHEMA.group_entities
        },
        **columns,
    }
    tables = {
        entity: pd.DataFrame(
            {US_SCHEMA.id_column(entity): np.asarray([1], dtype="int64")}
        )
        for entity in US_SCHEMA.group_entities
    }
    tables["person"] = pd.DataFrame(person_columns)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray([1000.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _household_weight_frame(
    typed_values: np.ndarray,
    *,
    stored_values: np.ndarray | None = None,
) -> Frame:
    """A Frame whose authoritative household weights may shadow a stale column."""

    typed_values = np.asarray(typed_values, dtype=np.float64)
    n = len(typed_values)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.arange(1, n + 1, dtype="int64"),
        }
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1, dtype="int64")})
    if stored_values is not None:
        household["household_weight"] = np.asarray(stored_values, dtype=np.float64)
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=typed_values,
                kind=WeightKind.CALIBRATED,
            )
        },
    )


class _StubEngine:
    """Only ``default_values(names)`` — the single engine surface the gate uses."""

    def __init__(self, defaults: dict[str, object]) -> None:
        self._defaults = dict(defaults)

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}


def _manifest(
    columns: tuple[ReleaseInputColumn, ...],
    probes: tuple[ReformCoverageProbe, ...] = (),
) -> ReleaseInputCoverageManifest:
    return ReleaseInputCoverageManifest(
        reference={"source": "test"}, columns=columns, probes=probes
    )


# A two-required-plus-one-excluded contract, reused across the gate cases.
_CONTRACT = _manifest(
    (
        ReleaseInputColumn("employment_income", "required"),
        ReleaseInputColumn("stock_assets", "required"),
        ReleaseInputColumn(
            "alimony_income",
            "reviewed_exclusion",
            reason="Residual income-source layer not yet sourced; tracked.",
            issue="PolicyEngine/microcosm#38",
        ),
    )
)

# Every declared column defaults to 0.0 in the stub engine, so an all-zero
# required column reads as degenerate (present but indistinguishable from absent).
_DEFAULTS = {"employment_income": 0.0, "stock_assets": 0.0, "alimony_income": 0.0}


class TestReleaseInputCoverageGate:
    def test_full_required_set_with_signal_passes(self) -> None:
        # Case 1: both required columns present and carrying signal; the excluded
        # column is absent (dormant), which is reported, not failed.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0, 12_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.failures == ()
        assert result.details["dormant_exclusions"] == ["alimony_income"]

    def test_missing_required_column_fails(self) -> None:
        # Case 2: stock_assets is absent from the export entirely — the silent
        # zero the #368 launch failure rode in on.
        frame = _person_frame({"employment_income": np.asarray([0.0, 52_000.0])})
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["missing"]
        assert any(
            "stock_assets" in failure and "absent" in failure
            for failure in result.failures
        )

    def test_degenerate_required_column_without_exclusion_fails(self) -> None:
        # Case 3: stock_assets is present but every value is the engine default,
        # so the export writer's default-broadcast makes it indistinguishable
        # from absence — and there is no reviewed exclusion to accept it.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["degenerate_required"]
        assert any(
            "stock_assets" in failure and "default" in failure
            for failure in result.failures
        )

    def test_all_nonfinite_required_asset_columns_fail(self) -> None:
        assets = tuple(sorted(SSI_COUNTABLE_RESOURCE_ASSETS))
        assert assets == ("bank_account_assets", "bond_assets", "stock_assets")
        manifest = _manifest(
            tuple(ReleaseInputColumn(name, "required") for name in assets)
        )
        frame = _person_frame(
            {name: np.asarray([np.nan, np.nan], dtype=np.float64) for name in assets}
        )

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: 0.0 for name in assets}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["missing"] == []
        assert result.details["degenerate_required"] == list(assets)
        assert result.details["no_observed_required"] == list(assets)
        assert len(result.failures) == 3
        for name, failure in zip(assets, result.failures, strict=True):
            assert name in failure
            assert "no finite/non-null observed values" in failure

    @pytest.mark.parametrize(
        ("name", "values", "default"),
        [
            (
                "business_is_sstb",
                pd.Series([pd.NA, pd.NA], dtype="boolean"),
                False,
            ),
            (
                "ssn_card_type",
                pd.Series([pd.NA, pd.NA], dtype="string"),
                "CITIZEN",
            ),
            (
                "stock_assets",
                np.asarray([pd.NA, pd.NA], dtype=object),
                0.0,
            ),
            (
                "event_date",
                pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
                pd.Timestamp("2000-01-01"),
            ),
        ],
        ids=["nullable_boolean", "nullable_string", "object_pd_na", "datetime_nat"],
    )
    def test_required_all_pandas_missing_column_has_named_no_observed_finding(
        self,
        name: str,
        values: np.ndarray | pd.Series,
        default: object,
    ) -> None:
        frame = _person_frame({name: values})
        manifest = _manifest((ReleaseInputColumn(name, "required"),))

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: default}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["missing"] == []
        assert result.details["degenerate_required"] == [name]
        assert result.details["no_observed_required"] == [name]
        assert len(result.failures) == 1
        assert result.failures[0].startswith(f"{name}: required eCPS input column")
        assert "no finite/non-null observed values" in result.failures[0]

    @pytest.mark.parametrize(
        ("name", "values", "default"),
        [
            (
                "business_is_sstb",
                pd.Series([pd.NA, True], dtype="boolean"),
                False,
            ),
            (
                "string_input",
                pd.Series([pd.NA, "x"], dtype="string"),
                "",
            ),
            (
                "stock_assets",
                np.asarray([np.nan, 125.0], dtype=np.float64),
                0.0,
            ),
        ],
        ids=["nullable_boolean", "nullable_string", "float_nan"],
    )
    def test_one_observed_nondefault_value_is_release_signal(
        self,
        name: str,
        values: np.ndarray | pd.Series,
        default: object,
    ) -> None:
        frame = _person_frame({name: values})
        manifest = _manifest((ReleaseInputColumn(name, "required"),))

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: default}),
            manifest=manifest,
        )

        assert result.passed
        assert result.failures == ()
        assert result.details["degenerate_required"] == []
        assert result.details["no_observed_required"] == []

    def test_real_us_nullable_boolean_is_no_observed_required(self) -> None:
        pytest.importorskip("policyengine_us")
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        name = "business_is_sstb"
        frame = _us_person_frame({name: pd.Series([pd.NA, pd.NA], dtype="boolean")})
        manifest = load_release_input_coverage_manifest()
        engine = PolicyEngineUSEngine()

        assert frame.table("person")[name].dtype == pd.BooleanDtype()
        assert name in manifest.required_columns
        assert engine.default_values([name]) == {name: False}

        result = us_release_input_coverage_gate(frame, engine, manifest=manifest)

        assert not result.passed
        assert name not in result.details["missing"]
        assert name in result.details["degenerate_required"]
        assert result.details["no_observed_required"] == [name]
        assert any(
            failure.startswith(f"{name}: required eCPS input column")
            and "no finite/non-null observed values" in failure
            for failure in result.failures
        )

    def test_stale_reviewed_exclusion_fails(self) -> None:
        # Case 4: alimony_income is a reviewed exclusion, but the data caught up
        # — it is now present with signal, so the exclusion is stale and must be
        # promoted to a hard requirement (#286 cannot-rot).
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 800.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["stale_exclusions"] == ["alimony_income"]
        assert any(
            "Stale reviewed exclusions" in failure for failure in result.failures
        )

    def test_absent_and_degenerate_excluded_column_passes(self) -> None:
        # The exclusion accepts both an absent column and a degenerate one: with
        # alimony_income present-but-all-default, the reviewed exclusion holds.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "alimony_income": "Residual income-source layer not yet sourced; tracked."
        }

    def test_typed_household_weights_count_as_persisted_input_signal(self) -> None:
        manifest = _manifest((ReleaseInputColumn("household_weight", "required"),))
        frame = _household_weight_frame(np.asarray([125.0, 275.0]))
        assert "household_weight" not in frame.table("household")

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({"household_weight": 1.0}),
            manifest=manifest,
        )

        assert result.passed
        assert result.failures == ()

    def test_typed_household_weights_override_stale_table_column(self) -> None:
        manifest = _manifest((ReleaseInputColumn("household_weight", "required"),))
        frame = _household_weight_frame(
            np.asarray([1.0, 1.0]),
            stored_values=np.asarray([125.0, 275.0]),
        )

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({"household_weight": 1.0}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["degenerate_required"] == ["household_weight"]


class _Series:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> float:
        return self._total


class _Sim:
    """A simulation whose weighted total for the measure is a fixed number."""

    def __init__(self, total: float) -> None:
        self._total = total

    def calculate(self, measure: str, period):  # noqa: ARG002 - stub
        return _Series(self._total)


def _probe(min_abs_effect: float = 1_000_000_000.0) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="ssi_probe",
        name="SSI asset limits raised to $10k / $20k",
        parameter_changes={
            "gov.ssa.ssi.eligibility.resources.limit.individual": {
                "2024-01-01.2100-12-31": 10000
            }
        },
        budget_measure="ssi",
        binding_inputs=("bank_account_assets", "stock_assets", "bond_assets"),
        min_abs_effect=min_abs_effect,
        reason="Assets absent → countable resources 0 → the relaxation scores $0.",
        issue="PolicyEngine/microcosm#356",
    )


def _tips_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="tips_probe",
        name="OBBBA no-tax-on-tips deduction",
        parameter_changes={
            "gov.irs.deductions.tip_income.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("tip_income", "treasury_tipped_occupation_code"),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through qualified tip income.",
        issue="PolicyEngine/microcosm#38",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _overtime_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_no_tax_on_overtime",
        name="OBBBA no-tax-on-overtime deduction",
        parameter_changes={
            "gov.irs.deductions.overtime_income.cap.SINGLE": {
                "2026-01-01.2026-12-31": 0
            }
        },
        budget_measure="income_tax",
        binding_inputs=("fsla_overtime_premium",),
        min_abs_effect=100_000_000.0,
        reason="The cap repeal must bind through the FLSA overtime premium.",
        issue="PolicyEngine/microcosm#242",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def _auto_loan_probe() -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="obbba_auto_loan_interest",
        name="OBBBA no-tax-on-auto-loan-interest deduction",
        parameter_changes={
            "gov.irs.deductions.auto_loan_interest.cap": {"2026-01-01.2026-12-31": 0}
        },
        budget_measure="income_tax",
        binding_inputs=("qualified_passenger_vehicle_loan_interest",),
        min_abs_effect=100_000_000.0,
        reason="The repeal must bind through qualifying vehicle-loan interest.",
        issue="PolicyEngine/microcosm#252",
        effect_direction="baseline_minus_reform",
        period=2026,
        expected_sign="negative",
    )


def test_reform_probe_requires_exactly_one_reform_kind() -> None:
    kwargs = {
        "id": "form_4952",
        "name": "Form 4952 neutralization",
        "budget_measure": "income_tax",
        "binding_inputs": ("investment_income_elected_form_4952",),
        "min_abs_effect": 1_000_000.0,
        "reason": "The neutralization binds only through the input.",
        "issue": "PolicyEngine/microcosm#274",
    }
    with pytest.raises(ValueError, match="exactly one"):
        ReformCoverageProbe(parameter_changes={}, **kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        ReformCoverageProbe(
            parameter_changes={"some.parameter": {"2024-01-01": 0}},
            neutralized_variable="investment_income_elected_form_4952",
            **kwargs,
        )
    with pytest.raises(ValueError, match="binding_inputs"):
        ReformCoverageProbe(
            parameter_changes={},
            neutralized_variable="different_input",
            **kwargs,
        )


class TestReformCoverageSmokeGate:
    def test_zero_bound_reform_fails(self, monkeypatch) -> None:
        # Case 5: with the asset inputs absent, everyone already passes the SSI
        # resource test, so raising the limit moves nothing — baseline and reform
        # score the same total and the bound reform reads as a coverage hole.
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):  # noqa: ARG001 - baseline == reform on purpose
            return _Sim(4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert not result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == 0.0
        assert "did not bind" in result.failures[0]
        assert "bank_account_assets" in result.failures[0]

    def test_bound_reform_with_effect_passes(self, monkeypatch) -> None:
        # The green counterpart: when the assets are carried, the same reform
        # moves SSI by ~$1.6B (the dense-native reference), clearing the floor.
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):
            return _Sim(4.16e10 if reform == "REFORM" else 4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == pytest.approx(1.6e9)

    def test_wrong_signed_effect_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        def simulate(reform):
            return _Sim(3.0e10 if reform == "REFORM" else 4.0e10)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate, probes=[_probe()], period=2024
        )
        assert not result.passed
        assert result.details["results"]["ssi_probe"]["effect"] == -1.0e10
        assert "expected a positive effect" in result.failures[0]

    def test_negative_tip_effect_uses_probe_period_and_passes(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")
        periods: list[int] = []

        class RecordingSim(_Sim):
            def calculate(self, measure: str, period):
                periods.append(period)
                return super().calculate(measure, period)

        def simulate(reform):
            return RecordingSim(10.5e9 if reform == "REFORM" else 10.0e9)

        result = us_reform_coverage_smoke_gate(
            simulate=simulate,
            probes=[_tips_probe()],
            period=2024,
        )

        assert result.passed
        assert periods == [2026, 2026]
        assert result.details["default_period"] == 2024
        tip_result = result.details["results"]["tips_probe"]
        assert tip_result["period"] == 2026
        assert tip_result["effect"] == pytest.approx(-0.5e9)
        assert tip_result["expected_sign"] == "negative"

    def test_negative_overtime_effect_passes_and_wrong_sign_fails(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        passing = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(10.5e9 if reform else 10.0e9),
            probes=[_overtime_probe()],
        )
        assert passing.passed
        assert passing.details["results"]["obbba_no_tax_on_overtime"][
            "effect"
        ] == pytest.approx(-0.5e9)

        wrong_sign = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(9.5e9 if reform else 10.0e9),
            probes=[_overtime_probe()],
        )
        assert not wrong_sign.passed
        assert "expected a negative effect" in wrong_sign.failures[0]

    def test_either_sign_probe_passes_both_directions_and_keeps_floor_teeth(
        self, monkeypatch
    ) -> None:
        # A signed, two-channel input (e.g. farm_operations_income: measured
        # ASEC leg plus donor-pinned PUF leg) proves COVERAGE by moving the
        # measure at all — the aggregate direction is a property of the frame
        # mix, not of coverage. "either" accepts the floor in both directions
        # while a structural (sub-floor) effect still fails.
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")
        probe = ReformCoverageProbe(
            id="either_probe",
            name="Signed two-channel leaf exclusion",
            parameter_changes={"gov.example.switch": {"2026-01-01.2026-12-31": 0}},
            budget_measure="income_tax",
            binding_inputs=("farm_operations_income",),
            min_abs_effect=1_000_000.0,
            reason="The exclusion binds through a signed leaf.",
            issue="PolicyEngine/microcosm#298",
            effect_direction="baseline_minus_reform",
            expected_sign="either",
        )

        positive = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(9.0e9 if reform else 10.0e9),
            probes=[probe],
        )
        assert positive.passed

        negative = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(11.0e9 if reform else 10.0e9),
            probes=[probe],
        )
        assert negative.passed

        structural_zero = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(10.0e9 + (1.0e5 if reform else 0.0)),
            probes=[probe],
        )
        assert not structural_zero.passed
        assert "an effect in either direction" in structural_zero.failures[0]

    def test_probe_rejects_unknown_expected_sign(self) -> None:
        with pytest.raises(ValueError, match="expected_sign must be"):
            ReformCoverageProbe(
                id="bad_sign",
                name="Bad sign",
                parameter_changes={"gov.example.switch": {"2026": 0}},
                budget_measure="income_tax",
                binding_inputs=("x",),
                min_abs_effect=1.0,
                reason="r.",
                issue="PolicyEngine/microcosm#1",
                expected_sign="sideways",
            )

    def test_negative_auto_loan_effect_passes_and_wrong_sign_fails(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(smoke_module, "_build_reform", lambda changes: "REFORM")

        passing = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(10.5e9 if reform else 10.0e9),
            probes=[_auto_loan_probe()],
        )
        assert passing.passed
        result = passing.details["results"]["obbba_auto_loan_interest"]
        assert result["effect"] == pytest.approx(-0.5e9)
        assert result["period"] == 2026

        wrong_sign = us_reform_coverage_smoke_gate(
            simulate=lambda reform: _Sim(9.5e9 if reform else 10.0e9),
            probes=[_auto_loan_probe()],
        )
        assert not wrong_sign.passed
        assert "expected a negative effect" in wrong_sign.failures[0]

    def test_probeless_gate_is_refused(self) -> None:
        # A probe-less smoke gate would pass vacuously — refuse it.
        with pytest.raises(ValueError, match="at least one probe"):
            us_reform_coverage_smoke_gate(simulate=lambda reform: _Sim(0.0), probes=[])


class TestShippedManifest:
    def test_manifest_is_current(self) -> None:
        # The checked-in-facts half runs everywhere (no engine in this env): the
        # declared surface must equal the reference eCPS populated columns and the
        # SSI assets must stay hard requirements.
        assert_release_input_coverage_manifest_current()

    def test_medicare_part_b_plural_names_are_deliberate_nonrequirements(
        self,
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        assert {
            "medicare_part_b_premiums",
            "medicare_part_b_premiums_reported",
        }.isdisjoint(manifest.declared_columns)

    def test_deprecated_marketplace_leaf_is_a_deliberate_nonrequirement(
        self,
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "has_marketplace_health_coverage" not in manifest.declared_columns
        assert (
            "has_marketplace_health_coverage_at_interview" in manifest.required_columns
        )

    def test_reference_wic_layer_projects_verified_successor(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert REFERENCE_ECPS_LAYER_RENAMES == {
            "would_claim_wic": "takes_up_wic_if_eligible"
        }
        assert "would_claim_wic" not in manifest.declared_columns
        assert "takes_up_wic_if_eligible" in manifest.required_columns

    def test_ssi_assets_are_required_without_exclusion(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for asset in SSI_COUNTABLE_RESOURCE_ASSETS:
            assert asset in manifest.required_columns
            assert asset not in manifest.reviewed_exclusions

    def test_restored_investment_interest_is_required_without_exclusion(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "investment_interest_expense"
        assert column in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

    def test_post_reference_ssi_disability_criterion_has_unique_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "meets_ssi_disability_criteria"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

        probe = next(
            probe
            for probe in manifest.probes
            if probe.id == "ssi_disability_criteria_neutralization"
        )
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == column
        assert probe.budget_measure == "ssi"
        assert probe.period == 2024
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == (column,)
        assert probe.min_abs_effect == 100_000_000.0

    def test_restored_ssi_take_up_has_unique_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "takes_up_ssi_if_eligible"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

        matches = [
            probe
            for probe in manifest.probes
            if probe.id == "ssi_take_up_neutralization"
        ]
        assert len(matches) == 1
        probe = matches[0]
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == column
        assert probe.budget_measure == "ssi"
        assert probe.period == 2024
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == (column,)
        assert probe.min_abs_effect == 10_000_000_000.0

    def test_restored_head_start_take_up_has_unique_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "takes_up_head_start_if_eligible"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

        matches = [
            probe
            for probe in manifest.probes
            if probe.id == "head_start_take_up_neutralization"
        ]
        assert len(matches) == 1
        probe = matches[0]
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == column
        assert probe.budget_measure == "head_start"
        assert probe.period == 2024
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == (column,)
        assert probe.min_abs_effect > 0.0

    def test_post_reference_obbba_inputs_are_hard_requirements(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "fsla_overtime_premium",
            "qualified_passenger_vehicle_loan_interest",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_legacy_auto_loan_columns_are_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("auto_loan_balance", "auto_loan_interest"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_sipp_vehicle_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("household_vehicles_owned", "household_vehicles_value"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_signed_scf_net_worth_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "net_worth" in manifest.required_columns
        assert "net_worth" not in manifest.reviewed_exclusions

    def test_typed_household_weight_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "household_weight" in manifest.required_columns
        assert "household_weight" not in manifest.reviewed_exclusions

    def test_education_input_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "qualified_tuition_expenses",
            "educational_assistance",
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_retirement_contribution_family_is_a_hard_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_casualty_loss_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "casualty_loss" in manifest.required_columns
        assert "casualty_loss" not in manifest.reviewed_exclusions

    def test_alimony_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("alimony_income", "alimony_expense"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_misc_itemized_input_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "unreimbursed_business_employee_expenses"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

    def test_childcare_input_is_a_hard_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "spm_unit_pre_subsidy_childcare_expenses"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

    def test_energy_subsidy_is_promoted_with_unique_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "spm_unit_energy_subsidy"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

        probe = next(
            probe
            for probe in manifest.probes
            if probe.id == "spm_unit_energy_subsidy_neutralization"
        )
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == column
        assert probe.budget_measure == "spm_unit_benefits"
        assert probe.period == 2024
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == (column,)
        assert probe.min_abs_effect == 100_000_000.0

    def test_voluntary_filing_is_promoted_with_aca_ptc_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "would_file_taxes_voluntarily"
        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions

        probe = next(
            probe
            for probe in manifest.probes
            if probe.id == "voluntary_filing_aca_ptc_neutralization"
        )
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == column
        assert probe.budget_measure == "aca_ptc"
        assert probe.period == 2024
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == (column,)
        assert probe.min_abs_effect == 100_000_000.0

    def test_child_support_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("child_support_received", "child_support_expense"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_disability_benefits_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "disability_benefits" in manifest.required_columns
        assert "disability_benefits" not in manifest.reviewed_exclusions

    def test_educator_expense_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "educator_expense" in manifest.required_columns
        assert "educator_expense" not in manifest.reviewed_exclusions

    def test_other_health_insurance_premiums_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "other_health_insurance_premiums" in manifest.required_columns
        assert "other_health_insurance_premiums" not in manifest.reviewed_exclusions

    def test_prior_year_income_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in (
            "self_employment_income_last_year",
            "previous_year_income_available",
        ):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

        probe = next(
            probe
            for probe in manifest.probes
            if probe.id == "prior_year_self_employment_neutralization"
        )
        assert probe.period == 2024
        assert probe.neutralized_variable == "self_employment_income_last_year"
        assert probe.parameter_changes == {}
        assert probe.binding_inputs == ("self_employment_income_last_year",)
        assert probe.budget_measure == "tax_unit_earned_income_last_year"
        assert probe.effect_direction == "baseline_minus_reform"

    def test_weeks_unemployed_is_required_without_a_self_referential_probe(
        self,
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        column = "weeks_unemployed"

        assert column in manifest.required_columns
        assert column not in manifest.reviewed_exclusions
        # PolicyEngine-US 1.819.0 consumes this leaf through the AL, NY, OK,
        # and PA unemployment-benefit formulas. A direct neutralization would
        # test the column against itself rather than an independent policy
        # path, so this family deliberately adds no such probe.
        assert all(
            column not in probe.binding_inputs and probe.neutralized_variable != column
            for probe in manifest.probes
        )

    def test_signed_farm_business_income_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in ("farm_operations_income", "farm_rent_income"):
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_qbi_input_family_is_promoted(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in US_QBI_OUTPUT_COLUMNS:
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    def test_domestic_production_ald_is_promoted_separately_from_qbi(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert "domestic_production_ald" in manifest.required_columns
        assert "domestic_production_ald" not in manifest.reviewed_exclusions

    def test_form_4952_election_is_promoted_with_unique_probe(self) -> None:
        manifest = load_release_input_coverage_manifest()
        output = "investment_income_elected_form_4952"
        assert output in manifest.required_columns
        assert output not in manifest.reviewed_exclusions

        probe = next(
            probe
            for probe in manifest.probes
            if probe.id == "form_4952_election_neutralization"
        )
        assert probe.parameter_changes == {}
        assert probe.neutralized_variable == output
        assert probe.binding_inputs == (output,)

    def test_shipped_ssi_probe_binds_through_the_assets(self) -> None:
        probes = us_release_reform_coverage_probes()
        assert probes, "the shipped manifest must pin at least one reform probe"
        ssi = next(probe for probe in probes if probe.id == "ssi_asset_limit_10k_20k")
        assert set(SSI_COUNTABLE_RESOURCE_ASSETS) <= set(ssi.binding_inputs)
        assert ssi.budget_measure == "ssi"
        assert ssi.min_abs_effect > 0
        assert ssi.expected_sign == "positive"

    def test_shipped_tip_probe_has_2026_period_sign_and_inputs(self) -> None:
        tip = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_no_tax_on_tips"
        )
        assert tip.period == 2026
        assert tip.expected_sign == "negative"
        assert tip.effect_direction == "baseline_minus_reform"
        assert tip.budget_measure == "income_tax"
        assert set(tip.binding_inputs) == {
            "tip_income",
            "treasury_tipped_occupation_code",
        }

    def test_shipped_aotc_probe_binds_through_education_inputs(self) -> None:
        aotc = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "aotc_abolition"
        )
        assert aotc.period == 2024
        assert aotc.expected_sign == "positive"
        assert aotc.effect_direction == "baseline_minus_reform"
        assert aotc.budget_measure == "american_opportunity_credit"
        assert set(aotc.binding_inputs) == {
            "qualified_tuition_expenses",
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        }
        assert aotc.min_abs_effect > 0
        assert set(aotc.parameter_changes) == {
            "gov.irs.credits.education.american_opportunity_credit.abolition"
        }

    def test_shipped_savers_credit_probe_binds_through_contributions(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "savers_credit_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "savers_credit"
        assert set(probe.binding_inputs) == {
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        }
        assert probe.min_abs_effect == 100_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.credits.retirement_saving.contributions_cap"
        }

    def test_shipped_casualty_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_casualty_loss_limit"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("casualty_loss",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.deductions.itemized.casualty.active"
        }

    def test_shipped_alimony_probe_has_sign_period_and_expense_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "alimony_expense_ald_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "negative"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("alimony_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.ald.alimony_expense.divorce_year_threshold[0].amount"
        }

    def test_shipped_misc_itemized_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_misc_itemized_deductions"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("unreimbursed_business_employee_expenses",)
        assert probe.min_abs_effect == 100_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.deductions.itemized.misc.applies"
        }

    def test_shipped_cdcc_probe_has_2026_period_sign_and_input(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_cdcc"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "negative"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("spm_unit_pre_subsidy_childcare_expenses",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.irs.credits.cdcc.phase_out.max",
            "gov.irs.credits.cdcc.phase_out.min",
            "gov.irs.credits.cdcc.phase_out.amended_structure.applies",
        }

    def test_shipped_child_support_received_probe_removes_only_snap_source(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "child_support_received_snap_exclusion"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("child_support_received",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "disability_benefits",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        }

    def test_shipped_child_support_expense_probe_removes_only_snap_deduction(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "child_support_expense_snap_deduction_abolition"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("child_support_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.deductions.allowed": {
                "2024-01-01.2024-12-31": [
                    "snap_standard_deduction",
                    "snap_earned_income_deduction",
                    "snap_dependent_care_deduction",
                    "snap_excess_medical_expense_deduction",
                    "snap_excess_shelter_expense_deduction",
                ]
            }
        }

    def test_shipped_disability_probe_removes_only_snap_unearned_source(
        self,
    ) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "disability_benefits_snap_exclusion"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "snap"
        assert probe.binding_inputs == ("disability_benefits",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.usda.snap.income.sources.unearned": {
                "2024-01-01.2024-12-31": [
                    "ssi",
                    "tanf",
                    "general_assistance",
                    "pension_income",
                    "veterans_benefits",
                    "unemployment_compensation",
                    "workers_compensation",
                    "social_security",
                    "retirement_distributions",
                    "rental_income",
                    "child_support_received",
                    "alimony_income",
                    "financial_assistance",
                    "survivor_benefits",
                    "dividend_income",
                    "interest_income",
                    "miscellaneous_income",
                ]
            }
        }

    def test_shipped_educator_expense_probe_removes_only_its_ald(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "educator_expense_ald_abolition"
        )

        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "reform_minus_baseline"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("educator_expense",)
        assert probe.min_abs_effect == 1_000_000.0
        assert probe.parameter_changes == {
            "gov.irs.ald.deductions": {
                "2024-01-01.2024-12-31": [
                    "loss_ald",
                    "self_employment_tax_ald",
                    "student_loan_interest_ald",
                    "early_withdrawal_penalty",
                    "alimony_expense_ald",
                    "health_savings_account_ald",
                    "self_employed_health_insurance_ald",
                    "self_employed_pension_contribution_ald",
                    "traditional_ira_contributions",
                    "qualified_adoption_assistance_expense",
                    "us_bonds_for_higher_ed",
                    "specified_possession_income",
                    "puerto_rico_income",
                ]
            }
        }

    def test_shipped_qbi_probes_cover_reit_and_wage_property_inputs(self) -> None:
        probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        reit = probes["qbi_reit_ptp_rate_abolition"]
        assert reit.period == 2024
        assert reit.expected_sign == "positive"
        assert reit.effect_direction == "baseline_minus_reform"
        assert reit.budget_measure == "qualified_business_income_deduction"
        assert reit.binding_inputs == ("qualified_reit_and_ptp_income",)
        assert set(reit.parameter_changes) == {
            "gov.irs.deductions.qbi.max.reit_ptp_rate"
        }

        guardrails = probes["qbi_wage_property_guardrails_zeroed"]
        assert guardrails.period == 2024
        assert guardrails.expected_sign == "positive"
        assert guardrails.effect_direction == "baseline_minus_reform"
        assert guardrails.budget_measure == "qualified_business_income_deduction"
        assert set(guardrails.binding_inputs) == {
            "w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
        }
        assert set(guardrails.parameter_changes) == {
            "gov.irs.deductions.qbi.max.w2_wages.rate",
            "gov.irs.deductions.qbi.max.w2_wages.alt_rate",
            "gov.irs.deductions.qbi.max.business_property.rate",
        }

    def test_shipped_farm_probes_each_remove_only_the_bound_qbi_leaf(self) -> None:
        probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        current_income_definition = {
            "self_employment_income",
            "partnership_s_corp_income",
            "farm_rent_income",
            "farm_operations_income",
            "rental_income",
            "estate_income",
        }
        cases = {
            # "either": the leaf is signed and two-channel (measured ASEC FRSE
            # plus donor-pinned PUF Schedule F, microcosm#435) — the aggregate
            # QBID direction is a frame property, not a coverage property.
            "qbi_farm_operations_income_exclusion": (
                "farm_operations_income",
                "either",
            ),
            "qbi_farm_rent_income_exclusion": ("farm_rent_income", "positive"),
        }

        for probe_id, (removed_input, expected_sign) in cases.items():
            probe = probes[probe_id]
            assert probe.period == 2026
            assert probe.expected_sign == expected_sign
            assert probe.effect_direction == "baseline_minus_reform"
            assert probe.budget_measure == "qualified_business_income_deduction"
            assert probe.binding_inputs == (removed_input,)
            assert probe.min_abs_effect == 1_000_000.0
            assert set(probe.parameter_changes) == {
                "gov.irs.deductions.qbi.income_definition"
            }
            definition = probe.parameter_changes[
                "gov.irs.deductions.qbi.income_definition"
            ]
            assert set(definition) == {"2026-01-01.2026-12-31"}
            assert set(definition["2026-01-01.2026-12-31"]) == (
                current_income_definition - {removed_input}
            )

    def test_shipped_domestic_production_probe_reactivates_only_its_ald(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "domestic_production_ald_reactivation"
        )
        assert probe.period == 2024
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "income_tax"
        assert probe.binding_inputs == ("domestic_production_ald",)
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {"gov.irs.ald.deductions"}
        deductions = probe.parameter_changes["gov.irs.ald.deductions"]
        assert set(deductions) == {"2024-01-01.2024-12-31"}
        assert deductions["2024-01-01.2024-12-31"].count("domestic_production_ald") == 1

    def test_shipped_overtime_probe_has_2026_period_sign_and_input(self) -> None:
        overtime = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_no_tax_on_overtime"
        )
        assert overtime.period == 2026
        assert overtime.expected_sign == "negative"
        assert overtime.effect_direction == "baseline_minus_reform"
        assert overtime.budget_measure == "income_tax"
        assert overtime.binding_inputs == ("fsla_overtime_premium",)
        assert overtime.min_abs_effect > 0
        assert set(overtime.parameter_changes) == {
            "gov.irs.deductions.overtime_income.cap.JOINT",
            "gov.irs.deductions.overtime_income.cap.SINGLE",
            "gov.irs.deductions.overtime_income.cap.HEAD_OF_HOUSEHOLD",
            "gov.irs.deductions.overtime_income.cap.SURVIVING_SPOUSE",
            "gov.irs.deductions.overtime_income.cap.SEPARATE",
        }

    def test_shipped_overtime_probes_cite_the_jct_ledger_anchor(self) -> None:
        by_id = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        for probe_id in (
            "obbba_no_tax_on_overtime",
            "fsla_overtime_premium_neutralization",
        ):
            reason = by_id[probe_id].reason
            assert "JCX-35-25" in reason
            assert (
                "jct.obbba_title_vii.fy2026.no_tax_on_overtime.revenue_effect" in reason
            )
            assert "-$32.806 billion" in reason
        neutralization = by_id["fsla_overtime_premium_neutralization"].reason
        assert "certified Build N" in neutralization
        assert "-$16.86 billion" in neutralization
        assert "$114.79 billion weighted" in neutralization
        assert "over 29 million filers" in neutralization
        assert "over $3,100" in neutralization
        assert "approximately $90 billion claimed floor" in neutralization
        assert "sb0517" in neutralization

    def test_shipped_tips_probes_cite_the_jct_ledger_anchor(self) -> None:
        by_id = {probe.id: probe for probe in us_release_reform_coverage_probes()}
        for probe_id in ("obbba_no_tax_on_tips", "tip_income_neutralization"):
            reason = by_id[probe_id].reason
            assert "JCX-35-25" in reason
            assert "jct.obbba_title_vii.fy2026.no_tax_on_tips.revenue_effect" in reason
            assert "-$10.121 billion" in reason
        neutralization = by_id["tip_income_neutralization"].reason
        assert "certified Build N" in neutralization
        assert "-$1.63 billion" in neutralization
        assert "$34.28 billion weighted" in neutralization
        assert "$26.79 billion" in neutralization
        assert "over 7.5 million filers" in neutralization
        assert "over $7,000" in neutralization
        assert "sb0517" in neutralization

    def test_shipped_auto_loan_probe_has_2026_period_sign_and_input(self) -> None:
        auto = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "obbba_auto_loan_interest"
        )
        assert auto.period == 2026
        assert auto.expected_sign == "negative"
        assert auto.effect_direction == "baseline_minus_reform"
        assert auto.budget_measure == "income_tax"
        assert auto.binding_inputs == ("qualified_passenger_vehicle_loan_interest",)
        assert set(auto.parameter_changes) == {
            "gov.irs.deductions.auto_loan_interest.cap"
        }

    def test_shipped_vehicle_asset_probe_binds_through_both_inputs(self) -> None:
        probe = next(
            probe
            for probe in us_release_reform_coverage_probes()
            if probe.id == "tx_snap_additional_vehicle_exemption_abolition"
        )
        assert probe.period == 2026
        assert probe.expected_sign == "positive"
        assert probe.effect_direction == "baseline_minus_reform"
        assert probe.budget_measure == "snap"
        assert set(probe.binding_inputs) == {
            "household_vehicles_owned",
            "household_vehicles_value",
        }
        assert probe.min_abs_effect == 1_000_000.0
        assert set(probe.parameter_changes) == {
            "gov.hhs.tanf.non_cash.tx_additional_vehicle_exemption"
        }

    def test_demoting_an_ssi_asset_to_exclusion_is_rejected(self) -> None:
        # The #368 red-gate guarantee cannot be quietly undone: turning an SSI
        # asset into a reviewed exclusion must fail the anti-rot assertion.
        manifest = load_release_input_coverage_manifest()
        tampered_columns = tuple(
            ReleaseInputColumn(
                column.name,
                "reviewed_exclusion",
                reason="pretend this gap is tracked",
                issue="PolicyEngine/microcosm#000",
            )
            if column.name == "stock_assets"
            else column
            for column in manifest.columns
        )
        tampered = ReleaseInputCoverageManifest(
            reference=manifest.reference,
            columns=tampered_columns,
            probes=manifest.probes,
        )
        with pytest.raises(ValueError, match="stock_assets"):
            assert_release_input_coverage_manifest_current(
                manifest=tampered, engine=None
            )

    def test_duplicate_probe_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate reform coverage probe id"):
            _manifest(_CONTRACT.columns, probes=(_probe(), _probe()))


def _load_manifest_generator():
    spec = importlib.util.spec_from_file_location(
        "build_us_release_input_coverage_manifest", _MANIFEST_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestManifestGeneratorSync:
    def test_committed_manifest_matches_regeneration(self) -> None:
        # The committed manifest is derivable purely from the checked-in eCPS
        # parity reference + known-gaps register, so it cannot drift from them:
        # regenerating must reproduce the committed file byte-for-value.
        generator = _load_manifest_generator()
        committed = json.loads(
            files("microcosm.build.us")
            .joinpath(US_RELEASE_INPUT_COVERAGE_RESOURCE)
            .read_text(encoding="utf-8")
        )
        assert generator.build_manifest() == committed

    def test_generated_manifest_names_no_retired_data_package(self) -> None:
        # The manifest is not on the incumbent-reference allow-list, so its
        # provenance block must not name the retired data package (the guard
        # test_us_plan.test_no_incumbent_data_package_references_in_live_tree
        # enforces on the committed file; this pins the generator too).
        generator = _load_manifest_generator()
        rendered = json.dumps(generator.build_manifest())
        # Build the needles by concatenation so this test file does not itself
        # trip the live-tree guard it mirrors (test_us_plan does the same).
        assert ("policyengine-" + "us-data") not in rendered
        assert ("policyengine_" + "us_data") not in rendered


class TestCapitalGainDistributionRouteGuarantee:
    """microcosm#462 / #361 remedy: BOTH capital-gain-distribution route legs
    are export-guarded. The Build M live default shipped the direct-route leg
    (``non_sch_d_capital_gains``) 7.3x over its SOI dollar target while the
    Schedule-D route (``schedule_d_capital_gain_distributions``) was absent
    from the export entirely — and the coverage manifest guarded neither
    against demotion."""

    def test_route_constant_names_both_legs(self) -> None:
        assert US_CGD_ROUTE_REQUIRED_INPUTS == (
            "non_sch_d_capital_gains",
            "schedule_d_capital_gain_distributions",
        )

    def test_both_route_variables_are_required_without_exclusion(self) -> None:
        manifest = load_release_input_coverage_manifest()
        for column in US_CGD_ROUTE_REQUIRED_INPUTS:
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions

    @pytest.mark.parametrize(
        "column",
        [
            "non_sch_d_capital_gains",
            "schedule_d_capital_gain_distributions",
        ],
    )
    def test_route_variable_cannot_regress_to_reviewed_exclusion(
        self, column: str
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        assert column in manifest.declared_columns
        demoted = ReleaseInputCoverageManifest(
            reference=manifest.reference,
            columns=tuple(
                ReleaseInputColumn(
                    name=entry.name,
                    status="reviewed_exclusion",
                    reason="regression attempt",
                    issue="PolicyEngine/microcosm#462",
                )
                if entry.name == column
                else entry
                for entry in manifest.columns
            ),
            probes=manifest.probes,
            schema_version=manifest.schema_version,
        )

        with pytest.raises(ValueError, match=column):
            assert_release_input_coverage_manifest_current(manifest=demoted)
