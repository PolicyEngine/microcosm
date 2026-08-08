"""Measured ASEC Medicare take-up restoration."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    MEDICARE_TAKE_UP_ARCHIVED_CLONE_URL,
    MEDICARE_TAKE_UP_ARCHIVED_DERIVATION_URL,
    MEDICARE_TAKE_UP_ARCHIVED_EXPORT_URL,
    MEDICARE_TAKE_UP_ARCHIVED_SOURCE_COLUMNS_URL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_DONORS,
    US_MEDICARE_TAKE_UP_NONCONSTANT_PERSON_COLUMNS,
    US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS,
    US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS,
    US_MEDICARE_TAKE_UP_STAGE_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    US_STAGE_NAMES,
    clone_us_frame_for_puf_support,
    derive_us_medicare_take_up_from_manifest,
    load_release_input_coverage_manifest,
    us_medicare_take_up_signal_gate,
    us_medicare_take_up_stage_spec,
    us_medicare_take_up_summary,
    us_release_reform_coverage_probes,
    with_us_medicare_take_up_input,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

ROOT = Path(__file__).resolve().parents[3]
policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_OUTPUT = "takes_up_medicare_if_eligible"
_SOURCE = "MCARE"


def _frame(
    source_codes: list[object] | None = None,
    *,
    output: list[object] | None = None,
    weights: list[float] | None = None,
) -> Frame:
    codes = source_codes or [1, 2, 2, 2, 0]
    count = len(codes)
    household_ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids,
            "person_household_id": household_ids,
            "person_tax_unit_id": household_ids + 100,
            "person_spm_unit_id": household_ids + 200,
            "person_family_id": household_ids + 300,
            "person_marital_unit_id": household_ids + 400,
            _SOURCE: codes,
            "age": [70, 66, 40, 20, 10][:count],
        }
    )
    if output is not None:
        person[_OUTPUT] = output
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 100}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 200}),
        "family": pd.DataFrame({"family_id": household_ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights or [1.0] * count, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _operation() -> SourceOperationSpec:
    return next(
        operation
        for operation in us_medicare_take_up_stage_spec().operations
        if operation.kind == "derive_medicare_take_up"
    )


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    return derive_us_medicare_take_up_from_manifest(person, _operation(), None)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestManifestAndPlan:
    def test_stage_pins_exact_measured_mapping_and_clone_semantics(self) -> None:
        spec = us_medicare_take_up_stage_spec()

        assert spec.stage == US_MEDICARE_TAKE_UP_STAGE_NAME
        assert tuple(spec.outputs) == US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS == (_OUTPUT,)
        assert US_MEDICARE_TAKE_UP_NONCONSTANT_PERSON_COLUMNS == (_OUTPUT,)
        assert US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS == (_SOURCE,)
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            "derive_medicare_take_up",
        ]
        assert _operation().parameters == {
            "source": _SOURCE,
            "enrolled_code": 1,
            "output": _OUTPUT,
        }
        assert "MCARE == 1" in spec.notes
        assert "No take-up rate or stochastic draw" in spec.notes

    def test_archived_coordinates_are_immutable(self) -> None:
        assert MEDICARE_TAKE_UP_ARCHIVED_DERIVATION_URL.endswith(
            "/datasets/cps/cps.py#L1579-L1585"
        )
        assert MEDICARE_TAKE_UP_ARCHIVED_SOURCE_COLUMNS_URL.endswith(
            "/datasets/cps/census_cps.py#L39-L58"
        )
        assert MEDICARE_TAKE_UP_ARCHIVED_CLONE_URL.endswith(
            "/calibration/puf_impute.py#L608-L629"
        )
        assert MEDICARE_TAKE_UP_ARCHIVED_EXPORT_URL.endswith(
            "/datasets/cps/extended_cps.py#L1747-L1754"
        )

    def test_handler_and_plan_are_wired_before_puf_support(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_medicare_take_up"]
            is derive_us_medicare_take_up_from_manifest
        )
        assert US_MEDICARE_TAKE_UP_STAGE_NAME in US_DONORS
        assert US_STAGE_NAMES.index(US_MEDICARE_TAKE_UP_STAGE_NAME) < (
            US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        )


class TestDerivation:
    def test_maps_only_mcare_code_one_to_true(self) -> None:
        person = _frame([0, 1, 2]).table("person")

        result = _derive(person)

        assert result[_OUTPUT].tolist() == [False, True, False]
        assert result[_OUTPUT].dtype == bool

    @pytest.mark.parametrize("invalid", [np.nan, 1.5, -1, 3, "unknown"])
    def test_invalid_source_codes_fail_closed(self, invalid: object) -> None:
        person = _frame([1, invalid, 2]).table("person")
        with pytest.raises(SourceRuntimeError, match="MCARE"):
            _derive(person)

    def test_missing_source_wrong_operation_and_parameters_are_rejected(self) -> None:
        person = _frame().table("person")
        with pytest.raises(SourceRuntimeError, match="MCARE"):
            _derive(person.drop(columns=[_SOURCE]))
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_medicare_take_up_from_manifest(
                person,
                SourceOperationSpec(kind="wrong", parameters={}),
                None,
            )
        with pytest.raises(SourceRuntimeError, match="requires the person table"):
            derive_us_medicare_take_up_from_manifest(None, _operation(), None)
        with pytest.raises(SourceRuntimeError, match="drifted"):
            derive_us_medicare_take_up_from_manifest(
                person,
                SourceOperationSpec(
                    kind="derive_medicare_take_up",
                    parameters={"source": _SOURCE},
                ),
                None,
            )


class TestFrameStageAndGate:
    def test_materializes_exact_source_and_is_idempotent(self) -> None:
        frame = _frame()
        derived = with_us_medicare_take_up_input(frame, seed=0, time_period=2024)

        assert derived.table("person")[_OUTPUT].tolist() == [
            True,
            False,
            False,
            False,
            False,
        ]
        assert (
            with_us_medicare_take_up_input(derived, seed=99, time_period=2026)
            is derived
        )
        assert us_medicare_take_up_signal_gate(derived).passed

    def test_stale_nonconstant_output_is_rederived_from_source(self) -> None:
        frame = _frame(output=[False, True, False, False, False])

        derived = with_us_medicare_take_up_input(frame, seed=0, time_period=2024)

        assert derived.table("person")[_OUTPUT].tolist() == [
            True,
            False,
            False,
            False,
            False,
        ]
        assert us_medicare_take_up_summary(derived)["source_mismatch_count"] == 0

    def test_support_cloning_preserves_measured_values_on_both_channels(self) -> None:
        derived = with_us_medicare_take_up_input(_frame(), seed=0, time_period=2024)
        cloned = clone_us_frame_for_puf_support(derived)
        person = cloned.table("person")

        assert (
            person[_OUTPUT].tolist()
            == [
                True,
                False,
                False,
                False,
                False,
            ]
            * 2
        )
        summary = us_medicare_take_up_summary(cloned)
        assert summary["source_mismatch_count"] == 0
        assert summary["channel_weighted_enrolled_shares"] == {
            "asec": pytest.approx(0.2),
            PUF_TAX_DETAIL_SUPPORT_CHANNEL: pytest.approx(0.2),
        }
        assert us_medicare_take_up_signal_gate(cloned).passed

    def test_gate_rejects_missing_constant_bad_share_and_mismatch(self) -> None:
        missing = _frame()
        assert not us_medicare_take_up_signal_gate(missing).passed

        constant = _frame(output=[True] * 5)
        assert not us_medicare_take_up_signal_gate(constant).passed

        bad_share = _frame(
            source_codes=[1, 1, 1, 1, 2],
            output=[True, True, True, True, False],
        )
        assert not us_medicare_take_up_signal_gate(bad_share).passed

        mismatch = _frame(output=[False, True, False, False, False])
        gate = us_medicare_take_up_signal_gate(mismatch)
        assert not gate.passed
        assert any("reconciliation mismatch" in failure for failure in gate.failures)

    def test_non_us_schema_is_rejected(self) -> None:
        frame = Frame(
            {
                "person": pd.DataFrame(
                    {"person_id": [1], "person_household_id": [1], _SOURCE: [1]}
                ),
                "household": pd.DataFrame({"household_id": [1]}),
            },
            EntitySchema(group_entities=("household",)),
            {"household": Weights(np.ones(1), WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="US Medicare"):
            with_us_medicare_take_up_input(frame, seed=0, time_period=2024)


@requires_us
def test_all_sha_locked_asec_sources_have_exact_measured_signal() -> None:
    from policyengine_us.data import USSingleYearDataset

    expected = {
        "census_cps_2022.h5": {
            "sha256": "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            "positive": 26495,
            "weighted_share": 0.18617172806991097,
        },
        "census_cps_2023.h5": {
            "sha256": "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            "positive": 26466,
            "weighted_share": 0.187909801755554,
        },
        "census_cps_2024.h5": {
            "sha256": "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            "positive": 26448,
            "weighted_share": 0.19082998028041492,
        },
    }
    summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in summary["base_source"]["sources"]
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("SHA-locked ASEC artifacts are not mounted")

    assert set(paths) == set(expected)
    for filename, facts in expected.items():
        path = paths[filename]
        assert _sha256(path) == facts["sha256"]
        person = USSingleYearDataset(file_path=str(path)).person
        codes = pd.to_numeric(person[_SOURCE], errors="coerce")
        weights = pd.to_numeric(person["A_FNLWGT"], errors="coerce") / 100.0
        enrolled = codes == 1
        assert set(codes.unique()) == {0, 1, 2}
        assert int(enrolled.sum()) == facts["positive"]
        assert float(weights[enrolled].sum() / weights.sum()) == pytest.approx(
            facts["weighted_share"]
        )
        derived = _derive(person)
        np.testing.assert_array_equal(derived[_OUTPUT].to_numpy(), enrolled.to_numpy())


@requires_us
def test_policyengine_contract_and_live_neutralization() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "1.764.6"
    system = CountryTaxBenefitSystem()
    variable = system.variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert variable.value_type is bool
    assert bool(variable.default_value) is True
    assert str(variable.definition_period).lower() == "year"
    assert system.variables["medicare_enrolled"].adds == [_OUTPUT]

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 70},
                _OUTPUT: {"2024": True},
            }
        },
        "tax_units": {"tax_unit": {"members": ["adult"]}},
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
        "marital_units": {"marital_unit": {"members": ["adult"]}},
    }

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "medicare_take_up_neutralization"
    )
    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))
    assert baseline.calculate("medicare_enrolled", 2024)[0]
    assert not neutralized.calculate("medicare_enrolled", 2024)[0]
    assert baseline.calculate("medicare_cost", 2024)[0] > 0
    assert neutralized.calculate("medicare_cost", 2024)[0] == 0


def test_release_contract_promotion_probe_and_take_up_inventory() -> None:
    manifest = load_release_input_coverage_manifest()
    assert _OUTPUT in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert _OUTPUT in manifest.required_columns
    assert _OUTPUT not in manifest.reviewed_exclusions
    assert _OUTPUT in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "medicare_take_up_neutralization"
    )
    assert probe.neutralized_variable == _OUTPUT
    assert probe.binding_inputs == (_OUTPUT,)
    assert probe.budget_measure == "medicare_cost"
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    assert probe.min_abs_effect == 1_000_000_000.0

    contract = load_take_up_contract().program_map()[_OUTPUT]
    assert contract.populace_treatment == "out_of_scope"
    assert contract.rate == {"status": "not_used_measured_source"}
    assert "MCARE == 1" in str(contract.raw["notes"])


def test_both_release_builders_run_stage_and_gate() -> None:
    support_builder = (ROOT / "tools/build_us_puf_support_base.py").read_text()
    fiscal_builder = (ROOT / "tools/build_us_fiscal_refresh_release.py").read_text()
    cache_driver = (ROOT / "experiments/build_j_recert/buildj_base.sh").read_text()

    assert support_builder.index("with_us_medicare_take_up_input(") < (
        support_builder.index("clone_us_frame_for_puf_support(base)")
    )
    assert support_builder.count("us_medicare_take_up_signal_gate(") == 4
    assert "with_us_medicare_take_up_input(" in fiscal_builder
    assert "us_medicare_take_up_signal_gate(" in fiscal_builder
    assert f'"{_OUTPUT}"' in cache_driver


def test_generated_manifest_drops_the_retired_generic_gap() -> None:
    known_gaps = json.loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )["known_gaps"]
    assert _OUTPUT not in known_gaps
