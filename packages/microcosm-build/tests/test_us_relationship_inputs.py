"""ASEC relationship-input restoration and partner-source exclusion."""

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

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    US_DONORS,
    US_PUF_SUPPORT_STAGE_NAME,
    US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS,
    US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_RELATIONSHIP_INPUTS_STAGE_NAME,
    US_STAGE_NAMES,
    derive_us_relationship_inputs_from_manifest,
    load_release_input_coverage_manifest,
    us_relationship_inputs_signal_gate,
    us_relationship_inputs_stage_spec,
    us_relationship_inputs_summary,
    us_release_reform_coverage_probes,
    with_us_relationship_inputs,
)
from microcosm.build.us_runtime.asec_pool import _with_relationship_recode
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

ROOT = Path(__file__).resolve().parents[3]
policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)

_HEAD = "is_household_head"
_SEPARATED = "is_separated"
_SURVIVING = "is_surviving_spouse"
_UNMARRIED_PARTNER = "is_unmarried_partner_of_household_head"
_OUTPUTS = (_HEAD, _SEPARATED, _SURVIVING)


def _person(rows: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record: dict[str, object] = {
            "person_id": index + 1,
            "PH_SEQ": 100 + index,
            "P_SEQ": 1,
            "A_MARITL": 7,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


def _frame(rows: list[dict[str, object]], weights: list[float] | None = None) -> Frame:
    person = _person(rows)
    household_ids, person_household_ids = np.unique(
        person["PH_SEQ"].to_numpy(dtype=np.int64), return_inverse=True
    )
    person_household_ids = household_ids[person_household_ids]
    person["person_household_id"] = person_household_ids
    person["person_tax_unit_id"] = person_household_ids + 1_000
    person["person_spm_unit_id"] = person_household_ids + 2_000
    person["person_family_id"] = person_household_ids + 3_000
    person["person_marital_unit_id"] = np.arange(len(person)) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
        "family": pd.DataFrame({"family_id": household_ids + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(len(person)) + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights or [1.0] * len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _operation():
    return next(
        operation
        for operation in us_relationship_inputs_stage_spec().operations
        if operation.kind == "derive_relationship_inputs"
    )


def _known_gap(name: str) -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us")
        .joinpath("ecps_parity_known_gaps.json")
        .read_text(encoding="utf-8")
    )
    return payload["known_gaps"][name]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestManifestAndPlan:
    def test_stage_pins_exact_archived_derivations(self) -> None:
        spec = us_relationship_inputs_stage_spec()

        assert spec.stage == US_RELATIONSHIP_INPUTS_STAGE_NAME == "relationship_inputs"
        assert US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS == _OUTPUTS
        assert US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS == _OUTPUTS
        assert US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS == (
            "PH_SEQ",
            "P_SEQ",
            "A_MARITL",
        )
        assert tuple(spec.outputs) == _OUTPUTS
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            "derive_relationship_inputs",
        ]
        assert "cps.py lines 1069-1075 and 1209-1221" in spec.notes
        assert "P_SEQ == 1" in spec.notes
        assert "A_MARITL == 6" in spec.notes
        assert "A_MARITL == 4" in spec.notes

    def test_handler_and_plan_are_wired_before_puf_support(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["derive_relationship_inputs"]
            is derive_us_relationship_inputs_from_manifest
        )
        assert US_RELATIONSHIP_INPUTS_STAGE_NAME in US_DONORS
        assert US_STAGE_NAMES.index(US_RELATIONSHIP_INPUTS_STAGE_NAME) < (
            US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        )


class TestDerivation:
    def test_maps_exact_asec_codes(self) -> None:
        source = _person(
            [
                {"PH_SEQ": 1, "P_SEQ": 1, "A_MARITL": 4},
                {"PH_SEQ": 1, "P_SEQ": 2, "A_MARITL": 6},
                {"PH_SEQ": 2, "P_SEQ": 1, "A_MARITL": 7},
            ]
        )

        result = derive_us_relationship_inputs_from_manifest(source, _operation(), None)

        assert result[_HEAD].tolist() == [True, False, True]
        assert result[_SEPARATED].tolist() == [False, True, False]
        assert result[_SURVIVING].tolist() == [True, False, False]
        assert all(result[column].dtype == bool for column in _OUTPUTS)

    @pytest.mark.parametrize("missing", US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS)
    def test_missing_source_is_named(self, missing: str) -> None:
        source = _person([{}]).drop(columns=[missing])
        with pytest.raises(SourceRuntimeError, match=missing):
            derive_us_relationship_inputs_from_manifest(source, _operation(), None)

    @pytest.mark.parametrize(
        ("column", "value"),
        [("PH_SEQ", np.nan), ("P_SEQ", 0), ("P_SEQ", 1.5), ("A_MARITL", 0)],
    )
    def test_invalid_source_values_fail_closed(self, column: str, value: float) -> None:
        source = _person([{}, {}])
        source[column] = source[column].astype(float)
        source.loc[0, column] = value
        with pytest.raises(SourceRuntimeError, match=column):
            derive_us_relationship_inputs_from_manifest(source, _operation(), None)

    def test_requires_exactly_one_head_per_household(self) -> None:
        source = _person(
            [
                {"PH_SEQ": 1, "P_SEQ": 1},
                {"PH_SEQ": 1, "P_SEQ": 1},
            ]
        )
        with pytest.raises(SourceRuntimeError, match="exactly one P_SEQ == 1"):
            derive_us_relationship_inputs_from_manifest(source, _operation(), None)

    def test_support_clones_group_by_frame_household_not_raw_ph_seq(self) -> None:
        source = _person(
            [
                {"PH_SEQ": 1, "person_household_id": 10, "P_SEQ": 1},
                {"PH_SEQ": 1, "person_household_id": 10, "P_SEQ": 2},
                {"PH_SEQ": 1, "person_household_id": 20, "P_SEQ": 1},
                {"PH_SEQ": 1, "person_household_id": 20, "P_SEQ": 2},
            ]
        )
        result = derive_us_relationship_inputs_from_manifest(source, _operation(), None)
        assert result[_HEAD].tolist() == [True, False, True, False]

    def test_rejects_wrong_operation_and_missing_table(self) -> None:
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_relationship_inputs_from_manifest(
                _person([{}]),
                SourceStageSpec.from_mapping(
                    {
                        "stage": "test",
                        "survey": "test",
                        "source": "https://example.com",
                        "grain": "person",
                        "operations": [{"kind": "derive"}],
                        "outputs": list(_OUTPUTS),
                    }
                ).operations[0],
                None,
            )
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_relationship_inputs_from_manifest(None, _operation(), None)


class TestFrameAndGate:
    def test_frame_integration_and_idempotence(self) -> None:
        frame = _frame(
            [
                {"PH_SEQ": 1, "P_SEQ": 1, "A_MARITL": 4},
                {"PH_SEQ": 1, "P_SEQ": 2, "A_MARITL": 6},
                {"PH_SEQ": 2, "P_SEQ": 1, "A_MARITL": 7},
            ],
            weights=[2.0, 1.0],
        )
        result = with_us_relationship_inputs(frame, seed=0, time_period=2024)

        assert result.table("person")[_HEAD].tolist() == [True, False, True]
        assert with_us_relationship_inputs(result, seed=0, time_period=2024) is result

    def test_summary_and_gate_require_plausible_signal_and_one_head(self) -> None:
        rows: list[dict[str, object]] = []
        for household in range(1, 51):
            rows.extend(
                [
                    {
                        "PH_SEQ": household,
                        "P_SEQ": 1,
                        "A_MARITL": 4 if household <= 5 else 7,
                    },
                    {
                        "PH_SEQ": household,
                        "P_SEQ": 2,
                        "A_MARITL": 6 if household <= 2 else 7,
                    },
                ]
            )
        result = with_us_relationship_inputs(_frame(rows), seed=0, time_period=2024)

        summary = us_relationship_inputs_summary(result)
        assert summary["household_head_share"] == pytest.approx(0.5)
        assert summary["separated_share"] == pytest.approx(0.02)
        assert summary["surviving_spouse_share"] == pytest.approx(0.05)
        assert summary["households_without_exactly_one_head"] == 0
        assert us_relationship_inputs_signal_gate(result).passed

        result.table("person").loc[1, _HEAD] = True
        gate = us_relationship_inputs_signal_gate(result)
        assert not gate.passed
        assert any("exactly one" in failure for failure in gate.failures)

    def test_gate_rejects_missing_output(self) -> None:
        gate = us_relationship_inputs_signal_gate(_frame([{}]))
        assert not gate.passed
        assert _HEAD in gate.details["missing"]


class TestCoverageAndExclusion:
    def test_restored_inputs_are_hard_release_requirements(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert set(_OUTPUTS) <= RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert set(_OUTPUTS) <= set(US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS)
        assert set(_OUTPUTS) <= manifest.required_columns
        assert set(_OUTPUTS).isdisjoint(manifest.reviewed_exclusions)

    def test_unmarried_partner_exclusion_is_source_unavailability(self) -> None:
        entry = _known_gap(_UNMARRIED_PARTNER)
        evidence = entry["evidence"]

        assert entry["reason"].startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
        assert evidence["classification"] == "source_unavailability"
        assert evidence["retired_derivation"] == {
            "repository_owner": "PolicyEngine",
            "repository_name_parts": ["policyengine-", "us-data"],
            "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
            "path_parts": [
                "policyengine_",
                "us_data",
                "datasets",
                "cps",
                "cps.py",
            ],
            "lines": "1214-1221",
        }
        assert evidence["required_person_columns"] == ["PERRP"]
        assert all(
            "PERRP" in item["missing_person_columns"]
            for item in evidence["hermetic_inputs"]
        )
        assert evidence["semantic_non_substitutes"]["rejection"].startswith(
            "Using only the 2024"
        )

        manifest = load_release_input_coverage_manifest()
        assert manifest.reviewed_exclusions[_UNMARRIED_PARTNER] == entry["reason"]

    @requires_us
    def test_locked_artifacts_confirm_source_presence_and_absence(self) -> None:
        summary = json.loads(
            (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
        )
        paths = {
            Path(item["path"]).name: Path(item["path"])
            for item in summary["base_source"]["sources"]
        }
        if not all(path.is_file() for path in paths.values()):
            pytest.skip("SHA-locked ASEC artifacts are not mounted")

        evidence = _known_gap(_UNMARRIED_PARTNER)["evidence"]
        expected_heads = {2022: 56_839, 2023: 56_251, 2024: 55_762}
        for item in evidence["hermetic_inputs"]:
            path = paths[item["filename"]]
            assert _sha256(path) == item["sha256"]
            with pd.HDFStore(path, mode="r") as store:
                person = store["person"]
            assert set(item["missing_person_columns"]).isdisjoint(person.columns)
            assert {"PH_SEQ", "P_SEQ", "A_MARITL"} <= set(person.columns)
            year = int(item["filename"].split("_")[-1].split(".")[0])
            assert int((person["P_SEQ"] == 1).sum()) == expected_heads[year]
            assert (
                person.groupby("PH_SEQ")["P_SEQ"].apply(lambda x: (x == 1).sum()) == 1
            ).all()
            if year < 2024:
                recoded, source = _with_relationship_recode(person)
                assert source == "derived:line_spouse_parent"
                assert int((recoded["A_EXPRRP"] == 13).sum()) == 0
            else:
                assert (
                    int((person["PECOHAB"] > 0).sum()) == item["positive_pecohab_rows"]
                )
                assert (
                    int((person["A_EXPRRP"] == 13).sum())
                    == item["a_exprrp_partner_or_roommate_rows"]
                )


@requires_us
def test_policyengine_1_764_6_contract_and_live_bindings() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "1.764.6"
    variables = CountryTaxBenefitSystem().variables
    for name in _OUTPUTS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert variable.value_type is bool
        assert variable.default_value is False
    assert str(variables[_HEAD].definition_period).lower() == "eternity"
    assert str(variables[_SEPARATED].definition_period).lower() == "year"
    assert str(variables[_SURVIVING].definition_period).lower() == "year"

    common = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 50_000},
                "is_tax_unit_head": {"2024": True},
                _SURVIVING: {"2024": True},
            },
            "child": {
                "age": {"2024": 5},
                "is_tax_unit_dependent": {"2024": True},
            },
        },
        "tax_units": {"tax_unit": {"members": ["adult", "child"]}},
        "spm_units": {"spm_unit": {"members": ["adult", "child"]}},
        "households": {
            "household": {
                "members": ["adult", "child"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    assert (
        Simulation(situation=common).calculate("filing_status", 2024).decode()[0].value
        == "Surviving spouse"
    )

    separated = json.loads(json.dumps(common))
    separated["people"]["adult"].pop(_SURVIVING)
    separated["people"]["adult"][_SEPARATED] = {"2024": True}
    assert (
        Simulation(situation=separated)
        .calculate("filing_status", 2024)
        .decode()[0]
        .value
        == "Head of household"
    )


def test_shipped_household_head_neutralization_probe() -> None:
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "household_head_childcare_cap_neutralization"
    )
    assert probe.neutralized_variable == _HEAD
    assert probe.binding_inputs == (_HEAD,)
    assert probe.budget_measure == "spm_unit_capped_work_childcare_expenses"
    assert probe.period == 2024
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "negative"
    assert probe.min_abs_effect >= 1_000_000.0
