"""Tests split from packages/microcosm-build/tests/test_us_medicare_take_up.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_medicare_take_up import *


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

    def test_stacked_gate_reconciles_only_physical_asec_source_rows(self) -> None:
        stacked = _stacked_frame()

        gate = us_medicare_take_up_signal_gate(stacked)

        assert gate.passed, gate.failures
        assert gate.details["source_rows"] == 6
        assert gate.details["source_mismatch_count"] == 0
        assert gate.details["channel_weighted_enrolled_shares"] == {
            "asec": pytest.approx(0.2),
            PUF_TAX_DETAIL_SUPPORT_CHANNEL: pytest.approx(0.2),
        }

        person = stacked.table("person")
        asec = person["person_support_channel"].eq("asec")
        person.loc[person.index[asec][0], _SOURCE] = np.nan
        failed = us_medicare_take_up_signal_gate(stacked)
        assert not failed.passed
        assert any("MCARE" in failure for failure in failed.failures)

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
