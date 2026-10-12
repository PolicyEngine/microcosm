"""Tests split from packages/microcosm-build/tests/test_us_multispine_pool.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_multispine_pool import *


def test_simulation_projection_defaults_match_pinned_engine_surface() -> None:
    __import__("policyengine_us")
    receipt = pool_engine_input_projection_receipt(PolicyEngineUSEngine())

    assert receipt == {
        "engine_version": "2.2.1",
        "input_count": 926,
        "default_count": 925,
        "defaults_sha256": (POOL_ENGINE_INPUT_PROJECTION_CONTRACT.defaults_sha256),
    }


def test_production_operator_invocations_are_total_and_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    __import__("policyengine_us")
    structural_expectations = (
        (
            multispine_pool_module.prepare_multispine_puf_predictors,
            ("derive_us_cps_carried_inputs",),
            {"_run_source_operator_chain"},
        ),
        (
            multispine_pool_module.prepare_multispine_source_inputs_for_clone,
            POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
            {"_run_source_operator_chain"},
        ),
        (
            multispine_pool_module._post_clone_source_operators,
            POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
            set(),
        ),
        (
            multispine_pool_module.derive_multispine_pool_inputs,
            POOL_DERIVE_OPERATOR_ORDER,
            {
                "PoolStageOutput",
                "_run_source_operator_chain",
                "bind_us_qbi_reconciliation_transition_authority",
                "dict",
                "list",
                "pool_remaining_stage_input_manifest_receipt",
                "us_qbi_reconciliation_change_receipt",
                "validate_us_qbi_reconciliation_live_output",
                "validate_us_qbi_reconciliation_transition",
                "with_us_qbi_input_reconciliation",
            },
        ),
    )
    for (
        entrypoint,
        expected_operators,
        expected_orchestration,
    ) in structural_expectations:
        operators, orchestration = _operator_mapping_structure(entrypoint)
        assert operators == expected_operators
        assert orchestration == expected_orchestration

    observed: list[tuple[str, tuple[str, ...]]] = []

    def fail_direct_call(_frame: Frame, **_kwargs: object) -> Frame:
        raise AssertionError("operator kernel bypassed the phase-checked runner")

    for operator_name in POOL_OPERATOR_CONTRACTS:
        monkeypatch.setattr(
            multispine_pool_module,
            operator_name,
            fail_direct_call,
        )

    def observe_guarded_chain(
        frame: Frame,
        *,
        phase: str,
        operator_names: tuple[str, ...],
        operators: dict[str, Callable[[Frame], Frame]],
        **_kwargs: object,
    ) -> PoolStageOutput:
        assert tuple(operators) == operator_names
        observed.append((phase, operator_names))
        return PoolStageOutput(
            frame,
            {
                "phase": phase,
                "operator_order": list(operator_names),
                "suboperators": [
                    {
                        "operator": name,
                        "kernel_receipt": (
                            {"sha256": "0" * 64}
                            if name == "with_us_qbi_input_reconciliation"
                            else {}
                        ),
                    }
                    for name in operator_names
                ],
            },
        )

    monkeypatch.setattr(
        multispine_pool_module,
        "_run_source_operator_chain",
        observe_guarded_chain,
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "validate_us_qbi_reconciliation_live_output",
        lambda _frame, _receipt, *, boundary, expected_transition_authority_sha256: {},
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "bind_us_qbi_reconciliation_transition_authority",
        lambda current, _receipt: current,
    )
    frame = _source_frame()
    multispine_pool_module.prepare_multispine_source_inputs_for_clone(
        frame,
        acs_rent_donor=pd.DataFrame(),
    )
    completed = multispine_pool_module.complete_multispine_source_inputs(frame)
    multispine_pool_module.derive_multispine_pool_inputs(frame)

    assert set(completed.receipt["deferred_transfer_inputs"]["inputs"]) == set(
        POOL_DEFERRED_TRANSFER_INPUTS
    )

    assert observed == [
        ("pre_clone", POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER),
        *(
            ("post_clone", (operator,))
            for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        ),
        ("post_clone", POOL_DERIVE_OPERATOR_ORDER),
    ]
    observed_placements = {
        (name, phase) for phase, operator_names in observed for name in operator_names
    }
    registered_placements = {
        (name, phase)
        for name, contract in POOL_OPERATOR_CONTRACTS.items()
        for phase in contract.phases
    }
    assert observed_placements == registered_placements
    assert len({name for name, _phase in observed_placements}) == 23
    assert len(observed_placements) == 24


def test_derive_stage_rejects_preclone_pool_before_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    __import__("policyengine_us")
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    calls: list[str] = []

    def unexpected_kernel(frame: Frame) -> Frame:
        calls.append("called")
        return frame

    monkeypatch.setattr(
        multispine_pool_module,
        "_complete_schedule_d_input",
        unexpected_kernel,
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "with_us_qbi_input_reconciliation",
        unexpected_kernel,
    )

    with pytest.raises(ValueError, match="post_clone.*incompatible clone provenance"):
        multispine_pool_module.derive_multispine_pool_inputs(assembled)
    assert not calls


def test_derive_stage_keeps_whole_pool_qbi_reconciliation() -> None:
    __import__("policyengine_us")
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    frame = clone_us_frame_for_puf_support(assembled)
    person = frame.table("person").copy()
    person["long_term_capital_gains_before_response"] = 100.0
    person["non_sch_d_capital_gains"] = 0.0
    for column in US_QBI_OUTPUT_COLUMNS:
        person[column] = 0.0
    person["self_employment_income_before_lsr"] = 10.0
    person["SEMP"] = 10.0
    person["sstb_self_employment_income_before_lsr"] = 5.0
    frame = _replace_person(frame, person)

    result = multispine_pool_module.derive_multispine_pool_inputs(frame)
    derived = result.frame.table("person")

    assert result.receipt["operator_order"] == list(POOL_DERIVE_OPERATOR_ORDER)
    assert result.receipt["remaining_stage_input_manifest"]["entry_count"] == 1059
    assert result.receipt["remaining_stage_input_manifest"]["stage_counts"] == {
        "derive": 34,
        "seed": 33,
        "simulate": 992,
    }
    assert (
        result.receipt["qbi_input_reconciliation"]["recipient_source_universe"][
            "rows_excluded_from_base_self_employment_rewrite"
        ]
        == 0
    )
    assert result.receipt["qbi_input_reconciliation"][
        "base_self_employment_changed_rows"
    ] == len(derived)
    assert (
        result.receipt["qbi_input_reconciliation"][
            "structurally_absent_base_source_changed_rows"
        ]
        == 0
    )
    assert derived["schedule_d_capital_gain_distributions"].notna().all()
    assert derived["self_employment_income_before_lsr"].eq(15.0).all()
    assert derived["sstb_self_employment_income_before_lsr"].eq(0.0).all()


def test_derive_stage_rejects_forged_qbi_kernel_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    __import__("policyengine_us")
    monkeypatch.setattr(
        multispine_pool_module,
        "us_qbi_reconciliation_change_receipt",
        lambda _before, _after: {
            "version": 2,
            "sha256": "0" * 64,
            "changed_person_rows": -1,
            "tampered": True,
        },
    )

    with pytest.raises(ValueError, match="QBI.*schema mismatch"):
        multispine_pool_module.derive_multispine_pool_inputs(_qbi_ready_derive_frame())


def test_derive_stage_rejects_mutated_qbi_output_with_fresh_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    __import__("policyengine_us")
    real_kernel = multispine_pool_module.with_us_qbi_input_reconciliation

    def mutate_kernel(frame: Frame) -> Frame:
        result = real_kernel(frame)
        result.table("person").loc[0, "qualified_bdc_income"] = 0.25
        return result

    monkeypatch.setattr(
        multispine_pool_module,
        "with_us_qbi_input_reconciliation",
        mutate_kernel,
    )

    with pytest.raises(ValueError, match="deterministic kernel"):
        multispine_pool_module.derive_multispine_pool_inputs(_qbi_ready_derive_frame())


def test_simulation_projection_refuses_missing_measured_spm_role() -> None:
    frame = _assembled_cloned_with_partial_take_up()
    person = frame.table("person").copy()
    role = "is_spm_independent_minor_role"
    person[role] = pd.Series(True, index=person.index, dtype="boolean")
    person.loc[person.index[0], role] = pd.NA
    frame = _replace_person(frame, person)

    with pytest.raises(ValueError, match=rf"{role}; the engine declares no default"):
        multispine_pool_module._simulation_projection(frame, PolicyEngineUSEngine())

    assert frame.table("person")[role].isna().sum() == 1
