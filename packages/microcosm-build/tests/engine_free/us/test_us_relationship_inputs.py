"""Tests split from packages/microcosm-build/tests/test_us_relationship_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_relationship_inputs import *


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
