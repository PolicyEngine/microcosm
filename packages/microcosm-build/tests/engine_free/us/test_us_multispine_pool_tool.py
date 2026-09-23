"""Tests split from packages/microcosm-build/tests/test_us_multispine_pool_tool.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_multispine_pool_tool import *


def test_stacked_operator_target_requires_preparation_before_activation_authority(
    pool_tool: ModuleType,
) -> None:
    stacked = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.01,
        sample_seed=578,
    ).frame
    direction = GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={
            "person": {
                "source_operator_cps_carried": ("strike_benefits",),
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            r"asec_survey_to_acs/person/source_operator_cps_carried/"
            r"strike_benefits: declared gap-fill target column is absent"
        ),
    ):
        stacked_spine_module._verify_gap_fill_activation_authority(
            stacked,
            direction=direction,
        )

    prepared = _with_fixture_pre_clone_strike_benefits(stacked)
    assert stacked_spine_module._verify_gap_fill_activation_authority(
        prepared,
        direction=direction,
    ) == {
        ("person", "strike_benefits"): {
            "authorized_null_rows": 1,
            "recipient_rows": 1,
            "donor_rows": 1,
        }
    }


def test_stacked_config_authority_defaults_to_constants_without_loading_bundle(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = pool_tool._parser().parse_args(_stacked_main_argv(tmp_path))
    monkeypatch.setattr(
        pool_tool,
        "load_bundle",
        lambda _country: (_ for _ in ()).throw(
            AssertionError("the constants mode must not load a spec bundle")
        ),
    )

    assert args.config_authority == "constants"
    assert pool_tool._stacked_run_config(args) == {"config_authority": "constants"}


@pytest.mark.parametrize(
    ("failure_stage", "expected_status"),
    [
        ("preflight", "resolution_pending"),
        ("resolution", "resolution_failed"),
    ],
)
def test_constants_adapter_failure_receipt_preserves_requested_resolution_state(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    expected_status: str,
) -> None:
    _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    if failure_stage == "preflight":
        monkeypatch.setattr(
            pool_tool,
            "_git_code_pin",
            lambda: (_ for _ in ()).throw(RuntimeError("fixture preflight failure")),
        )
        expected_error = "fixture preflight failure"
    else:
        monkeypatch.setattr(
            pool_tool,
            "_stacked_run_config",
            lambda _args: (_ for _ in ()).throw(
                RuntimeError("fixture adapter resolution failure")
            ),
        )
        expected_error = "fixture adapter resolution failure"

    with pytest.raises(RuntimeError, match=expected_error):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--config-authority",
                "constants_adapter",
            ]
        )

    error_path = next((tmp_path / "logbook-receipts").glob("*/error.json"))
    error_receipt = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_receipt["run_config"] == {
        "config_authority": "constants_adapter",
        "spec_binding_status": expected_status,
    }
    assert "spec_binding" not in error_receipt["run_config"]
    assert "spec_sha256" not in json.dumps(error_receipt["run_config"])


def test_stacked_checkpoint_emission_propagates_and_authenticates_late_authority(
    pool_tool: ModuleType,
) -> None:
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    captured: list[MultispinePoolCheckpoint] = []

    pool_tool._emit_stacked_checkpoint(
        captured.append,
        stage="transferred",
        frame=authorized,
        assembly_receipt={},
        stage_receipts={
            "geography_assignment": {"fixture": True},
            "impute": impute,
        },
        late_producer_transition_authority_sha256=(transition_authority_sha256),
    )

    assert len(captured) == 1
    assert captured[0].late_producer_transition_authority_sha256 == (
        transition_authority_sha256
    )
    with pytest.raises(
        ValueError,
        match="differs from the independently carried late-producer",
    ):
        pool_tool._emit_stacked_checkpoint(
            captured.append,
            stage="transferred",
            frame=authorized,
            assembly_receipt={},
            stage_receipts={
                "geography_assignment": {"fixture": True},
                "impute": impute,
            },
            late_producer_transition_authority_sha256="0" * 64,
        )
    assert len(captured) == 1


def test_stacked_publication_rejects_noncanonical_receipt_before_any_write(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    noncanonical = _noncanonical_post_puf_authority_receipt()
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool,
        _source_frame(),
        authority=noncanonical,
    )
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked publication entry: non-canonical stacked authority is forbidden"
        ),
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_stacked_publication_rejects_forged_late_transition_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    authorized, impute, _transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="differs from the independently carried late-producer",
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_late_dag_validator_rejects_forged_execution_row(
    pool_tool: ModuleType,
) -> None:
    receipt = _canonical_late_dag_receipt(pool_tool)
    receipt["execution"][1]["producer"] = "source:forged"

    with pytest.raises(
        ValueError,
        match=r"execution row 1 is misbound",
    ):
        pool_tool.validate_stacked_late_producer_receipt(
            receipt,
            boundary="forged execution regression",
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        (
            "stripped_group_summary",
            "stripped or misbound calibration summary evidence",
        ),
        ("stripped_target_evidence", "calibration evidence is absent"),
        ("malformed_target_evidence", "calibration receipt digest is invalid"),
        ("deleted_target_receipt", "target surface is non-canonical"),
        ("extra_target_receipt", "target surface is non-canonical"),
        ("stripped_owner_count", "reference_rows evidence is misbound"),
        ("stripped_live_output", "live-output binding is absent"),
    ),
)
def test_late_transfer_validator_rejects_stripped_calibration_evidence(
    pool_tool: ModuleType,
    mutation: str,
    error_match: str,
) -> None:
    receipt = _canonical_late_transfer_receipt(pool_tool)
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="canonical calibration evidence control",
    )
    forged = copy.deepcopy(receipt)
    group_receipt = next(
        group
        for group in forged["groups"].values()
        if group["post_transfer_calibration"]["target_count"] > 0
    )
    if mutation == "stripped_group_summary":
        group_receipt.pop("post_transfer_calibration")
    else:
        target_key = group_receipt["post_transfer_calibration"]["targets"][0]
        if mutation == "deleted_target_receipt":
            group_receipt["targets"].pop(target_key)
        elif mutation == "extra_target_receipt":
            group_receipt["targets"]["person/forged/extra_target"] = {}
        elif mutation == "stripped_owner_count":
            group_receipt["targets"][target_key]["post_transfer_calibration"].pop(
                "reference_rows"
            )
        elif mutation == "stripped_live_output":
            group_receipt["targets"][target_key]["post_transfer_calibration"][
                "context_binding"
            ].pop("live_output")
        elif mutation == "stripped_target_evidence":
            target_receipt = group_receipt["targets"][target_key]
            target_receipt.pop("post_transfer_calibration")
        else:
            target_receipt = group_receipt["targets"][target_key]
            target_receipt["post_transfer_calibration"]["calibration"]["sha256"] = (
                "0" * 64
            )

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            forged,
            boundary=f"{mutation} regression",
        )


def test_late_transfer_validator_rejects_forged_pregnancy_policy(
    pool_tool: ModuleType,
) -> None:
    receipt = _canonical_late_transfer_receipt(pool_tool)
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="canonical pregnancy policy control",
    )
    forged = copy.deepcopy(receipt)
    pregnancy = next(
        target_receipt
        for group in forged["groups"].values()
        for target_key, target_receipt in group["targets"].items()
        if target_key.endswith("/is_pregnant")
    )
    pregnancy["structural_policy"]["policy_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="pregnancy structural policy is invalid"):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            forged,
            boundary="forged pregnancy policy regression",
        )


@pytest.mark.parametrize(
    ("target", "constraint_column", "replacement", "error_match"),
    (
        (
            "weeks_unemployed",
            "unemployment_compensation",
            0.0,
            "positive weeks-unemployed carriers lack positive unemployment",
        ),
        (
            "pre_subsidy_care_expenses",
            "is_incapable_of_self_care",
            False,
            "live adult-care carriers violate qualifying-person",
        ),
    ),
)
def test_late_transfer_validator_recomputes_live_coupled_constraints(
    pool_tool: ModuleType,
    target: str,
    constraint_column: str,
    replacement: object,
    error_match: str,
) -> None:
    frame, impute, _transition = _authorized_late_impute_fixture(
        pool_tool,
        _source_frame(),
    )
    receipt = impute["stacked_post_puf_transfer"]
    stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="live coupled-constraint control",
        frame=frame,
    )
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    person = tables["person"]
    recipient = person[support_channel_column("person")].astype(str).eq("acs")
    carrier = recipient & pd.to_numeric(person[target], errors="raise").gt(0.0)
    assert carrier.any()
    person.loc[person.index[carrier][0], constraint_column] = replacement
    tables.update({name: frame.link(name) for name in frame.links})
    corrupted = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    with pytest.raises(ValueError, match=error_match):
        stacked_spine_module.validate_stacked_post_puf_transfer_receipt(
            receipt,
            boundary="live coupled-constraint mutation",
            frame=corrupted,
        )


def test_stacked_publication_rejects_forged_derived_order_before_any_write(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, _source_frame()
    )
    impute["stacked_late_producer_dag"]["post_puf_transfer"][
        "producer_execution_order"
    ] = ["forged:wrong"]
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")
    result = SimpleNamespace(
        frame=authorized,
        stage_receipts={"impute": impute},
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=r"execution order does not match the derived late-producer schedule",
    ):
        pool_tool._write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs={},
            acs_source_manifest=pool_tool.load_acs_source_manifest(),
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


@pytest.mark.parametrize(
    "failure",
    ("negative_seed", "oversized_seed", "invalid_output", "code_pin"),
)
def test_stacked_preflight_errors_emit_one_logbook_row(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    predecessor = "a" * 64
    arguments = _stacked_main_argv(tmp_path, predecessor=predecessor)
    if failure in {"negative_seed", "oversized_seed"}:
        seed_index = arguments.index("--sample-seed") + 1
        arguments[seed_index] = "-1" if failure == "negative_seed" else str(2**63)
        expected = "--sample-seed must be a non-negative signed 64-bit integer"
    elif failure == "invalid_output":
        output_index = arguments.index("--out") + 1
        arguments[output_index] = str(tmp_path / "not-an-h5.txt")
        expected = "--out must name an .h5 or .hdf5 file"
    else:
        monkeypatch.setattr(
            pool_tool,
            "_git_code_pin",
            lambda: (_ for _ in ()).throw(RuntimeError("fixture code pin failure")),
        )
        expected = "fixture code pin failure"

    with pytest.raises((RuntimeError, ValueError), match=expected):
        pool_tool.main(arguments)

    row_paths = list((tmp_path / "logbook-spool").glob("*.json"))
    assert len(row_paths) == 1
    row = load_logbook_row(row_paths[0])
    assert row.disposition == "failed"
    assert row.rung == "f001"
    assert row.prev_row_digest == predecessor
    assert row.code_pin == (
        "unresolved-local-git-code-pin" if failure == "code_pin" else "a" * 40
    )
    assert row.artifact_location is None
    assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
    error_path = _receipt_file_from_reference(
        row.gate_verdicts["pipeline_error"]["receipt"]
    )
    assert error_path.is_file()
    assert error_path.parent == tmp_path / "logbook-receipts" / row.build_id
    assert order == []


def test_pool_checkpoint_identity_binds_late_producer_schedule(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")

    def identity() -> dict[str, object]:
        return pool_tool._pool_checkpoint_base_identity(
            verified,
            policyengine_us_version="fixture-engine",
        )

    current = identity()
    expected_schedule = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    assert current["pool_code"]["late_producer_schedule"] == expected_schedule

    changed_schedule = copy.deepcopy(expected_schedule)
    changed_schedule["payload_sha256"] = "0" * 64
    monkeypatch.setattr(
        pool_tool,
        "us_late_producer_schedule_receipt",
        lambda: changed_schedule,
    )
    changed = identity()

    assert pool_tool._pool_checkpoint_identity_sha256(changed) != (
        pool_tool._pool_checkpoint_identity_sha256(current)
    )


def test_legacy_checkpoint_identity_excludes_stacked_late_producer_schedule(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    current = pool_tool._legacy_pool_checkpoint_base_identity(
        verified,
        policyengine_us_version="fixture-engine",
    )
    assert current["materializer_version"] == 3
    assert "late_producer_schedule" not in current["pool_code"]

    changed_schedule = pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    changed_schedule["payload_sha256"] = "0" * 64
    monkeypatch.setattr(
        pool_tool,
        "us_late_producer_schedule_receipt",
        lambda: changed_schedule,
    )
    changed = pool_tool._legacy_pool_checkpoint_base_identity(
        verified,
        policyengine_us_version="fixture-engine",
    )

    assert changed == current


def test_geography_assignment_receipt_rejects_divergent_clone_geography(
    pool_tool: ModuleType,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(count=2),
        _many_household_source_frame(count=2, measured_offset=1_000.0),
        sample_fraction=1.0,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    household = assigned.table("household")
    source_column = pool_tool.support_source_id_column("household")
    household.loc[household.index[1], source_column] = household.loc[
        household.index[0], source_column
    ]
    household.loc[household.index[1], "congressional_district_geoid"] = 602
    receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    receipt["target_universe"] = (
        pool_tool._target_congressional_district_universe_receipt((601, 602))
    )
    receipt["output"]["unique_congressional_district_values"] = 2

    with pytest.raises(
        ValueError,
        match="cloned household support rows disagree.*congressional_district_geoid",
    ):
        pool_tool._validate_stacked_geography_assignment_receipt(
            assigned,
            receipt,
            target_districts=(601, 602),
            boundary="fixture clone divergence",
            require_exact_assembled_rows=False,
        )


def test_geography_assignment_receipt_rejects_coherent_valid_target_rewrite(
    pool_tool: ModuleType,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(count=2),
        _many_household_source_frame(count=2, measured_offset=1_000.0),
        sample_fraction=1.0,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    receipt["target_universe"] = (
        pool_tool._target_congressional_district_universe_receipt((601, 602))
    )
    assigned.table("household")["congressional_district_geoid"] = 602

    with pytest.raises(
        ValueError,
        match="ordered native household geography differs",
    ):
        pool_tool._validate_stacked_geography_assignment_receipt(
            assigned,
            receipt,
            target_districts=(601, 602),
            boundary="fixture coherent geography rewrite",
            require_exact_assembled_rows=False,
        )


@pytest.mark.parametrize(
    ("route", "stage_receipts"),
    (
        (
            "legacy",
            {"derive": {"qbi_input_reconciliation": {"fixture": "receipt"}}},
        ),
        (
            "stacked",
            {
                "derive": {
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "receipt"}
                    }
                }
            },
        ),
    ),
)
def test_qbi_receipt_route_resolution_is_exact(
    pool_tool: ModuleType,
    route: str,
    stage_receipts: Mapping[str, Mapping[str, object]],
) -> None:
    assert pool_tool._qbi_receipt_from_stage_receipts(
        stage_receipts,
        route=route,
        boundary="fixture QBI route",
    ) == {"fixture": "receipt"}


@pytest.mark.parametrize(
    ("route", "stage_receipts", "message"),
    (
        (
            "legacy",
            {
                "derive": {
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "stacked"}
                    }
                }
            },
            "legacy QBI receipt used the stacked derive route",
        ),
        (
            "stacked",
            {"derive": {"qbi_input_reconciliation": {"fixture": "legacy"}}},
            "stacked derive receipts have no pool_derivation object",
        ),
        (
            "stacked",
            {
                "derive": {
                    "qbi_input_reconciliation": {"fixture": "legacy"},
                    "pool_derivation": {
                        "qbi_input_reconciliation": {"fixture": "stacked"}
                    },
                }
            },
            "stacked QBI receipt also appears at the legacy route",
        ),
    ),
)
def test_qbi_receipt_route_resolution_rejects_wrong_or_ambiguous_paths(
    pool_tool: ModuleType,
    route: str,
    stage_receipts: Mapping[str, Mapping[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pool_tool._qbi_receipt_from_stage_receipts(
            stage_receipts,
            route=route,
            boundary="fixture QBI route",
        )


def test_stacked_resume_rejects_noncanonical_post_puf_transfer_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )
    assigned = _with_fixture_household_geography(stack.frame)
    geography_receipt = _fixture_geography_assignment_receipt(pool_tool, assigned)
    noncanonical = _noncanonical_post_puf_authority_receipt()
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool,
        assigned,
        authority=noncanonical,
    )
    resume = pool_tool.MultispinePoolCheckpoint(
        stage="transferred",
        frame=authorized,
        assembly_receipt=stack.frame.metadata[pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY],
        stage_receipts={
            "geography_assignment": geography_receipt,
            "impute": impute,
        },
        late_producer_transition_authority_sha256=transition_authority_sha256,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked transferred checkpoint resume: non-canonical stacked "
            "authority is forbidden"
        ),
    ):
        pool_tool.build_stacked_pool(
            assigned,
            expected_stack_receipt=stack.receipt,
            geography_assignment_receipt=None,
            geography_target_districts=(601,),
            release_id=(
                "populace-us-2024-stacked-f010-s578-asec4-acs1-"
                "20260807T000000Z-deadbeef"
            ),
            puf_donor=None,
            acs_rent_donor=None,
            primary_qrf_checkpoint_dir=tmp_path / "primary-qrf",
            acs_transfer_checkpoint_dir=tmp_path / "acs-transfer",
            checkpoint_identity={},
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            resume=resume,
        )


@pytest.mark.parametrize(
    ("fraction", "token"),
    [(0.01, "f001"), (0.10, "f010"), (1.0, "f100")],
)
def test_stacked_release_id_carries_rung_seed_and_realized_counts(
    pool_tool: ModuleType,
    fraction: float,
    token: str,
) -> None:
    release_id = pool_tool._new_stacked_release_id(
        sample_fraction=fraction,
        sample_seed=578,
        realized_asec_households=123,
        realized_acs_households=456,
        timestamp=datetime(2026, 8, 5, 12, 34, 56, tzinfo=UTC),
        nonce="abcdef01",
    )

    assert release_id == (
        f"populace-us-2024-stacked-{token}-s578-asec123-acs456-"
        "20260805T123456Z-abcdef01"
    )


def test_legacy_two_spine_fixture_is_origin_main_byte_exact(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Keep this byte-level legacy-output golden on its pre-authentication
    # synthetic fixture. QBI authentication has independent generation,
    # checkpoint, resume, manifest, and publication tamper tests.
    monkeypatch.setattr(
        multispine_pool_module,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, boundary, transition_authority_sha256: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, route, boundary, transition_authority_sha256: (
            None
        ),
    )
    store = _checkpoint_fixture_store(pool_tool, tmp_path / "checkpoints")
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    result, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=store,
        authenticated_qbi=False,
    )
    artifact = tmp_path / "legacy-origin-main.canonical.h5"
    pool_tool.write_frame_checkpoint(artifact, result.frame)

    assert order == ["impute", "derive", "seed", "simulate"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        # Rebased once on the import-entry branch: frame checkpoints now
        # record string storage and NA markers explicitly (environment-
        # independent restores, needed once microcosm-build ships pyarrow),
        # which adds two metadata fields per StringDtype column. Verified
        # identical on CPython 3.13/3.14, Linux CI, and macOS.
        "12e937914d739f5bd9a1a59df7b6de7ae5458f06cdc6c3b93a09ddf4ee47ecbd"
    )


def test_legacy_entrypoint_publication_matches_origin_main_golden(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _unused_outputs, verified, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
        authenticated_qbi=False,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    build_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fixture_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return ready

    def deterministic_fixture_h5(
        frame: Frame,
        path: Path,
        *,
        period: int,
        artifact_kind: str,
        publication_run_id: str,
    ) -> None:
        assert period == pool_tool.POOL_TIME_PERIOD
        assert artifact_kind == pool_tool.POOL_H5_ARTIFACT_KIND
        assert publication_run_id == "fixture-publication-run"
        # Pandas/PyTables embeds wall-clock metadata in its H5 bytes.  Keep the
        # real publication envelope but use Microcosm's timestamp-free Frame
        # serializer so this origin/main golden is stable across processes.
        pool_tool.write_frame_checkpoint(path, frame)

    monkeypatch.setattr(
        pool_tool,
        "_verify_inputs",
        lambda _args, _outputs: (verified, source_manifest),
    )
    monkeypatch.setattr(
        pool_tool,
        "_load_inputs",
        lambda _args, *, acs_source_manifest: loaded,
    )
    monkeypatch.setattr(pool_tool, "build_multispine_pool", fixture_build)
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: "fixture-publication-run",
    )
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-policyengine-us",
    )
    monkeypatch.setattr(
        pool_tool,
        "write_nullable_us_h5",
        deterministic_fixture_h5,
    )
    # This golden deliberately preserves the pre-authentication synthetic
    # output byte-for-byte. Dedicated QBI boundary tests below exercise the
    # authenticated production contract against real frame-bound receipts.
    monkeypatch.setattr(
        pool_tool,
        "_validate_qbi_stage_receipt",
        lambda _frame, _stage_receipts, *, route, boundary, transition_authority_sha256: (
            None
        ),
    )

    output = tmp_path / "legacy-pool.h5"
    checkpoint_root = tmp_path / "checkpoints"
    argv: list[str] = []
    for option in (
        "asec-raw-stage-h5",
        "acs-household-zip",
        "acs-person-zip",
        "acs-rent-h5",
        "puf-h5",
        "puf-source-year-csv",
    ):
        argv.extend([f"--{option}", str(tmp_path / option)])
        argv.extend([f"--{option}-sha256", "1" * 64])
    argv.extend(
        [
            "--checkpoint-root",
            str(checkpoint_root),
            "--out",
            str(output),
            "--legacy-two-spine",
        ]
    )

    assert pool_tool.main(argv) == 0
    assert len(build_calls) == 1
    positional, keywords = build_calls[0]
    assert positional == (loaded.asec, loaded.acs)
    assert keywords["puf_donor"] is loaded.puf_donor
    assert keywords["acs_rent_donor"] is loaded.acs_rent_donor
    assert keywords["source_native_inputs"] == {"acs": loaded.acs_native_inputs}
    assert keywords["resume"] is None
    assert callable(keywords["checkpoint"])
    checkpoint_store = keywords["checkpoint"].__self__
    assert checkpoint_store.base_identity["materializer_version"] == 3
    assert "late_producer_schedule" not in checkpoint_store.base_identity["pool_code"]

    outputs = pool_tool._output_paths(output, checkpoint_root=checkpoint_root)
    manifest = pool_tool._read_json_object(outputs.manifest)
    diagnostics = pool_tool._read_json_object(outputs.agreement_diagnostics)
    assert pool_tool.POOL_MANIFEST_SCHEMA_VERSION == 10
    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    assert manifest["schema_version"] == 4
    assert diagnostics["schema_version"] == 4
    assert "materializer_version" not in manifest["pool_h5"]
    assert manifest["stage_checkpoints"]["materializer_version"] == 3
    assert {
        receipt["materializer_version"]
        for receipt in manifest["stage_checkpoints"]["stages"].values()
    } == {3}
    manifest_bytes = outputs.manifest.read_bytes().replace(
        str(tmp_path.resolve()).encode(),
        b"$TMP",
    )
    actual = {
        "pool_h5": hashlib.sha256(outputs.pool_h5.read_bytes()).hexdigest(),
        "agreement": hashlib.sha256(
            outputs.agreement_diagnostics.read_bytes()
        ).hexdigest(),
        "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    # Generated by executing origin/main at 188c5d9c and this branch's explicit
    # legacy entrypoint against the same fixture path and publication run ID;
    # the pool tool remains unchanged through the e6be79a7 transplant base.
    assert actual == {
        # Rebased with the fixture golden above (explicit string-storage
        # checkpoint metadata).
        "pool_h5": "ced797ecdd44a638c2a3945f07ad612098a7095ca53a5f458699bca6d6e38b3e",
        "agreement": "f39f0d918bf7ee01dddb5517d8830b8adb541273c5be084307be91397caca3cb",
        # The engine-lock move to PolicyEngine-US 2.2.1 legitimately moves the
        # pool-code checkpoint identities embedded in the otherwise legacy
        # publication: the checkpoint identity carries policyengine_us_version,
        # and the pool engine contracts it binds were re-derived for 2.2.1.
        # pool_h5 and agreement above are unchanged, so only the identity
        # surface moved, not the pool content.
        "manifest": "e4692aa45f05826eb0097a7ae76dcbc712c13886a4d23a9c6a62b53752e323f1",
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_main_dispatches_only_to_the_selected_pipeline(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    calls: list[str] = []
    namespace = SimpleNamespace(
        legacy_two_spine=legacy,
        config_authority="constants",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(pool_tool, "_parser", lambda: parser)
    monkeypatch.setattr(
        pool_tool,
        "_main_legacy",
        lambda _args: calls.append("legacy") or 17,
    )
    monkeypatch.setattr(
        pool_tool,
        "_main_stacked",
        lambda _args: calls.append("stacked") or 23,
    )

    assert pool_tool.main(["fixture"]) == (17 if legacy else 23)
    assert calls == (["legacy"] if legacy else ["stacked"])


def test_main_refuses_constants_adapter_with_legacy_two_spine(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    namespace = SimpleNamespace(
        legacy_two_spine=True,
        config_authority="constants_adapter",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(pool_tool, "_parser", lambda: parser)
    monkeypatch.setattr(
        pool_tool,
        "_main_legacy",
        lambda _args: calls.append("legacy") or 17,
    )
    monkeypatch.setattr(
        pool_tool,
        "_main_stacked",
        lambda _args: calls.append("stacked") or 23,
    )

    with pytest.raises(
        ValueError,
        match="constants_adapter.*stacked pipeline",
    ):
        pool_tool.main(["fixture"])
    assert calls == []


def test_parser_exposes_eight_pinned_inputs_out_and_checkpoint_root(
    pool_tool: ModuleType,
) -> None:
    parser = pool_tool._parser()
    actions = {
        action.dest: action for action in parser._actions if action.dest != "help"
    }
    pairs = (
        ("asec_raw_stage_h5", "asec_raw_stage_h5_sha256"),
        ("acs_household_zip", "acs_household_zip_sha256"),
        ("acs_person_zip", "acs_person_zip_sha256"),
        ("acs_rent_h5", "acs_rent_h5_sha256"),
        ("puf_h5", "puf_h5_sha256"),
        ("puf_source_year_csv", "puf_source_year_csv_sha256"),
        ("puma_ladder", "puma_ladder_sha256"),
        (
            "congressional_district_vintage_crosswalk",
            "congressional_district_vintage_crosswalk_sha256",
        ),
    )
    expected_destinations = {destination for pair in pairs for destination in pair} | {
        "config_authority",
        "logbook_prev_row_digest",
        "clone_attachment_fraction",
        "clone_attachment_seed",
        "checkpoint_root",
        "legacy_two_spine",
        "out",
        "sample_fraction",
        "sample_seed",
    }

    assert set(actions) == expected_destinations
    assert all(
        actions[destination].required
        for destination in expected_destinations
        - {
            "checkpoint_root",
            "config_authority",
            "logbook_prev_row_digest",
            "clone_attachment_fraction",
            "clone_attachment_seed",
            "puma_ladder",
            "puma_ladder_sha256",
            "congressional_district_vintage_crosswalk",
            "congressional_district_vintage_crosswalk_sha256",
            "legacy_two_spine",
            "sample_fraction",
            "sample_seed",
        }
    )
    assert not actions["checkpoint_root"].required
    assert actions["out"].option_strings == ["--out"]
    assert actions["checkpoint_root"].option_strings == ["--checkpoint-root"]
    assert actions["checkpoint_root"].type is Path
    assert actions["sample_fraction"].default == 1.0
    assert actions["sample_seed"].default == 578
    assert actions["clone_attachment_fraction"].default == 1.0
    assert actions["clone_attachment_seed"].default == 578
    assert actions["legacy_two_spine"].default is False
    assert actions["config_authority"].default == "constants"
    assert tuple(actions["config_authority"].choices) == (
        "constants",
        "constants_adapter",
    )
    assert actions["logbook_prev_row_digest"].default is None
    for path_destination, sha_destination in pairs:
        assert actions[path_destination].type is Path
        assert actions[sha_destination].type is pool_tool._sha256_argument
        assert len(actions[path_destination].option_strings) == 1
        assert len(actions[sha_destination].option_strings) == 1

    option_names = {
        option for action in actions.values() for option in action.option_strings
    }
    assert not any(
        forbidden in option
        for option in option_names
        for forbidden in ("tolerance", "target", "per-target")
    )


@pytest.mark.parametrize(
    ("destination", "option"),
    [
        ("puma_ladder", "--puma-ladder"),
        ("puma_ladder_sha256", "--puma-ladder-sha256"),
        (
            "congressional_district_vintage_crosswalk",
            "--congressional-district-vintage-crosswalk",
        ),
        (
            "congressional_district_vintage_crosswalk_sha256",
            "--congressional-district-vintage-crosswalk-sha256",
        ),
    ],
)
def test_stacked_route_requires_each_pinned_geography_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
    destination: str,
    option: str,
) -> None:
    args = SimpleNamespace(
        puma_ladder=tmp_path / "puma-ladder.npz",
        puma_ladder_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
        congressional_district_vintage_crosswalk=tmp_path / "crosswalk.csv",
        congressional_district_vintage_crosswalk_sha256=(
            pool_tool._STACKED_CD_CROSSWALK_SHA256
        ),
    )
    setattr(args, destination, None)

    with pytest.raises(ValueError, match=option):
        pool_tool._require_stacked_geography_arguments(args)


@pytest.mark.parametrize(
    ("destination", "option"),
    [
        ("puma_ladder_sha256", "--puma-ladder-sha256"),
        (
            "congressional_district_vintage_crosswalk_sha256",
            "--congressional-district-vintage-crosswalk-sha256",
        ),
    ],
)
def test_stacked_route_rejects_noncanonical_geography_authority_pins(
    pool_tool: ModuleType,
    tmp_path: Path,
    destination: str,
    option: str,
) -> None:
    args = SimpleNamespace(
        puma_ladder=tmp_path / "puma-ladder.npz",
        puma_ladder_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
        congressional_district_vintage_crosswalk=tmp_path / "crosswalk.csv",
        congressional_district_vintage_crosswalk_sha256=(
            pool_tool._STACKED_CD_CROSSWALK_SHA256
        ),
    )
    setattr(args, destination, "f" * 64)

    with pytest.raises(ValueError, match=option):
        pool_tool._require_stacked_geography_arguments(args)


def test_pool_tool_structurally_accepts_only_the_raw_stage_loader(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_asec_raw_stage_checkpoint" in imported
    assert "load_asec_raw_stage_checkpoint" in called
    assert "load_asec_pre_clone_checkpoint" not in imported
    assert "load_asec_pre_clone_checkpoint" not in called
    assert "pre_clone_enrichment" not in source


def test_checkpoint_root_defaults_alongside_out_and_accepts_override(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run" / "pool.h5"
    default = pool_tool._output_paths(output)

    assert default.checkpoint_root == output.with_suffix(".checkpoints")
    assert default.primary_qrf_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "primary-qrf"
    )
    assert default.acs_transfer_checkpoint_dir == (
        output.with_suffix(".checkpoints") / "acs-transfer"
    )

    explicit_root = tmp_path / "persistent" / "pool-checkpoints"
    explicit = pool_tool._output_paths(
        output,
        checkpoint_root=explicit_root,
    )

    assert explicit.checkpoint_root == explicit_root
    assert explicit.primary_qrf_checkpoint_dir == explicit_root / "primary-qrf"
    assert explicit.acs_transfer_checkpoint_dir == explicit_root / "acs-transfer"

    identity_sha256 = "a" * 64
    bound = pool_tool._with_checkpoint_identity(
        explicit,
        base_identity_sha256=identity_sha256,
    )
    assert bound.primary_qrf_checkpoint_dir == (
        explicit_root / "primary-qrf" / identity_sha256
    )
    assert bound.acs_transfer_checkpoint_dir == (
        explicit_root / "acs-transfer" / identity_sha256
    )


def test_identity_routed_bank_sibling_scan_is_bounded_and_ignores_junk(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bank_root = tmp_path / "primary-qrf"
    bank_root.mkdir()
    current_digest = "f" * 64
    selected = bank_root / current_digest

    class _Entry:
        def __init__(self, name: str, *, directory: bool) -> None:
            self.name = name
            self._directory = directory

        def is_dir(self, *, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            return self._directory

    entries = [
        _Entry(current_digest, directory=True),
        _Entry("not-a-digest", directory=True),
        _Entry("1" * 64, directory=False),
        _Entry("2" * 64, directory=False),
        _Entry("3" * 64, directory=True),
        *[_Entry(f"{index:064x}", directory=True) for index in range(10, 70)],
    ]

    class _Scandir:
        def __enter__(self):
            return iter(entries)

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(pool_tool.os, "scandir", lambda _root: _Scandir())

    receipt = pool_tool._identity_routed_bank_open_receipt(
        selected,
        current_base_identity_sha256=current_digest,
    )

    assert receipt["scan"] == {
        "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "entries_examined": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
        "truncated": True,
    }
    stale_digests = {
        record["stale_base_identity_sha256"]
        for record in receipt["identity_mismatches"]
    }
    assert "3" * 64 in stale_digests
    assert current_digest not in stale_digests
    assert "1" * 64 not in stale_digests
    assert "2" * 64 not in stale_digests
    assert all(
        record["load_status"] == "identity_mismatch"
        and record["disposition"] == "bypassed"
        for record in receipt["identity_mismatches"]
    )


def test_checkpoint_root_allows_safe_input_colocation_on_persistent_volume(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "persistent-volume"
    outputs = pool_tool._output_paths(
        tmp_path / "published" / "pool.h5",
        checkpoint_root=checkpoint_root,
    )

    pool_tool._validate_checkpoint_path_layout(
        outputs,
        source_paths={checkpoint_root / "inputs" / "asec.h5"},
    )


@pytest.mark.parametrize(
    "outputs",
    (
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "pool.h5",
            checkpoint_root=tmp_path / "pool.h5",
        ),
        lambda pool_tool, tmp_path: pool_tool._output_paths(
            tmp_path / "checkpoints" / "assembled.checkpoint.h5",
            checkpoint_root=tmp_path / "checkpoints",
        ),
    ),
)
def test_checkpoint_root_rejects_publication_file_collisions(
    pool_tool: ModuleType,
    tmp_path: Path,
    outputs: Callable[[ModuleType, Path], object],
) -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint paths collide with publication files",
    ):
        pool_tool._validate_checkpoint_path_layout(
            outputs(pool_tool, tmp_path),
            source_paths=set(),
        )


def test_stacked_nested_checkpoint_root_rejects_a_stage_publication_collision(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    configured_root = tmp_path / "checkpoints"
    stacked_root = configured_root / "stacked" / ("a" * 64)
    base_outputs = pool_tool._stacked_output_paths(
        stacked_root / "assembled.checkpoint.h5",
        checkpoint_root=configured_root,
    )
    nested_outputs = replace(
        base_outputs,
        checkpoint_root=stacked_root,
        primary_qrf_checkpoint_dir=stacked_root / "primary-qrf",
        acs_transfer_checkpoint_dir=stacked_root / "acs-transfer",
    )

    with pytest.raises(
        ValueError,
        match="checkpoint paths collide with publication files",
    ):
        pool_tool._validate_checkpoint_path_layout(
            nested_outputs,
            source_paths=set(),
        )


def test_atomic_json_fsyncs_parent_directory_after_rename(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(pool_tool.os, "fsync", tracked_fsync)
    monkeypatch.setattr(pool_tool.os, "replace", tracked_replace)
    output = tmp_path / "receipt.json"

    pool_tool._atomic_write_json(output, {"fixture": True})

    assert events == [
        ("fsync", "file"),
        ("replace", output.name),
        ("fsync", "directory"),
    ]


def test_json_ready_omits_empty_opt_in_transfer_pattern_regimes(
    pool_tool: ModuleType,
) -> None:
    pattern = acs_transfer_module.AcsTransferPattern(
        name="fixture",
        observed_optional_predictors=(),
        predictors=("age",),
        seed=1,
        weight_kind="source",
        donor_rows=2,
        recipient_rows=1,
    )

    legacy = pool_tool._json_ready(pattern)
    selected = pool_tool._json_ready(
        replace(
            pattern,
            target_regimes=(("fixture_target", "positive_only"),),
        )
    )

    assert isinstance(legacy, dict)
    assert "target_regimes" not in legacy
    assert isinstance(selected, dict)
    assert selected["target_regimes"] == [["fixture_target", "positive_only"]]


def test_pool_imputation_wires_post_clone_source_chain_after_primary_and_tail(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool._impute_pool)
    tree = ast.parse(source)
    expected = (
        "_initialize_or_resume_primary_qrf",
        "run_primary_puf_qrf_chain",
        "finalize_primary_puf_qrf_chain",
        "transfer_puf_capital_gains_tail",
        "complete_multispine_source_inputs",
        "pool_transfer_target_families",
        "AcsTransferTargetBankStore",
        "transfer_acs_inputs",
    )
    calls = sorted(
        (
            node.lineno,
            node.func.id,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in expected
    )

    assert tuple(name for _line, name in calls) == expected

    bank_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AcsTransferTargetBankStore"
    )
    identity_value = next(
        keyword.value for keyword in bank_call.keywords if keyword.arg == "identity"
    )
    assert isinstance(identity_value, ast.Call)
    assert isinstance(identity_value.func, ast.Name)
    assert identity_value.func.id == "_pool_checkpoint_stage_identity"
    assert isinstance(identity_value.args[0], ast.Name)
    assert identity_value.args[0].id == "checkpoint_identity"
    assert isinstance(identity_value.args[1], ast.Constant)
    assert identity_value.args[1].value == "transferred"

    transfer_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transfer_acs_inputs"
    )
    target_bank_value = next(
        keyword.value
        for keyword in transfer_call.keywords
        if keyword.arg == "target_bank"
    )
    assert isinstance(target_bank_value, ast.Name)
    assert target_bank_value.id == "target_bank"


def test_pool_imputation_binds_and_publishes_acs_target_bank(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _source_frame()
    checkpoint_identity = {
        "artifact_kind": "fixture-pool-checkpoint-identity",
        "materializer_version": 7,
        "inputs": {"fixture": {"sha256": "b" * 64}},
    }
    base_identity_sha256 = pool_tool._pool_checkpoint_identity_sha256(
        checkpoint_identity
    )
    primary_qrf_dir = tmp_path / "primary-qrf" / base_identity_sha256
    primary_qrf_dir.mkdir(parents=True)
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_manifest_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-manifest"},
    )
    pool_tool._atomic_write_json(
        pool_tool._primary_qrf_input_binding_path(primary_qrf_dir),
        {"artifact_kind": "fixture-primary-binding"},
    )
    acs_bank_dir = tmp_path / "acs-transfer" / base_identity_sha256
    observed: dict[str, object] = {}
    bank_receipt = {
        "artifact_kind": "fixture-acs-target-bank-receipt",
        "targets": {"0": {"source": "checkpoint"}},
    }

    class _RecordingBank:
        def __init__(self, root: Path, *, identity: object) -> None:
            observed["bank"] = self
            observed["root"] = root
            observed["identity"] = identity

        def receipt(self) -> dict[str, object]:
            return bank_receipt

    def fake_transfer(*args, target_bank=None, **kwargs):
        observed["transfer_target_bank"] = target_bank
        return SimpleNamespace(
            frame=args[0],
            fit_records=(),
            resolved_donor_channel="fixture",
            imputed_inputs=(),
            deferred_inputs=(),
        )

    tail_receipt = {
        "tail_distribution_receipts": {
            "frame_after_stage": {
                "positive_mass_five_x_target_exceeded": True,
            }
        }
    }
    monkeypatch.setattr(
        pool_tool,
        "_initialize_or_resume_primary_qrf",
        lambda *args, **kwargs: "resumed",
    )
    monkeypatch.setattr(pool_tool, "run_primary_puf_qrf_chain", lambda *args: None)
    monkeypatch.setattr(
        pool_tool,
        "finalize_primary_puf_qrf_chain",
        lambda *args, **kwargs: (frame, "calibrated"),
    )
    monkeypatch.setattr(
        pool_tool,
        "transfer_puf_capital_gains_tail",
        lambda *args, **kwargs: (frame, tail_receipt),
    )
    monkeypatch.setattr(
        pool_tool,
        "validate_puf_capital_gains_tail_manifest",
        lambda receipt: None,
    )
    monkeypatch.setattr(
        pool_tool,
        "complete_multispine_source_inputs",
        lambda input_frame: SimpleNamespace(frame=input_frame, receipt={}),
    )
    monkeypatch.setattr(
        pool_tool,
        "pool_transfer_target_families",
        lambda: {"person": {"fixture": ("fixture_target",)}},
    )
    monkeypatch.setattr(pool_tool, "AcsTransferTargetBankStore", _RecordingBank)
    monkeypatch.setattr(pool_tool, "transfer_acs_inputs", fake_transfer)

    result = pool_tool._impute_pool(
        frame,
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_bank_dir,
        checkpoint_identity=checkpoint_identity,
        checkpoint_input_binding={"artifact_kind": "fixture-input-binding"},
    )

    assert observed["root"] == acs_bank_dir
    assert observed["identity"] == pool_tool._pool_checkpoint_stage_identity(
        checkpoint_identity,
        "transferred",
    )
    assert observed["transfer_target_bank"] is observed["bank"]
    assert result.receipt["acs_qrf_transfer"]["target_bank"] == bank_receipt


def test_production_pool_wires_source_preparation_into_clone_stage(
    pool_tool: ModuleType,
) -> None:
    source = inspect.getsource(pool_tool.build_multispine_pool)

    assert "prepare_multispine_source_inputs_for_clone" in source
    assert "prepare_clone=prepare_clone_operator" in source


def test_direct_pool_fixtures_are_operator_free_before_assembly() -> None:
    frame = _source_frame()
    for entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.values():
        for entity, columns in entities.items():
            assert set(frame.table(entity)).isdisjoint(columns)


@pytest.mark.parametrize(
    ("family", "entity", "column"),
    [
        (family, entity, sorted(columns)[0])
        for family, entities in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items()
        for entity, columns in entities.items()
    ],
)
def test_each_operator_output_family_is_rejected_before_assembly(
    pool_tool: ModuleType,
    family: str,
    entity: str,
    column: str,
) -> None:
    frame = _source_frame()
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity][column] = 0.0
    contaminated = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(
        ValueError,
        match=rf"fixture {family}.*{column}|fixture.*{column}",
    ):
        pool_tool.assert_operator_free_source_frame(
            contaminated,
            label=f"fixture {family}",
        )


def test_sha_mismatch_refuses_before_loading_or_writing(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_paths = {
        "asec-raw-stage-h5": tmp_path / "asec.h5",
        "acs-household-zip": tmp_path / "household.zip",
        "acs-person-zip": tmp_path / "person.zip",
        "acs-rent-h5": tmp_path / "rent.h5",
        "puf-h5": tmp_path / "puf.h5",
        "puf-source-year-csv": tmp_path / "puf.csv",
    }
    for path in source_paths.values():
        path.write_bytes(b"fixture input")

    called = {"load": False, "write": False}

    def unexpected_load(*_args, **_kwargs):
        called["load"] = True
        raise AssertionError("SHA mismatch must precede source-frame loading.")

    def unexpected_write(*_args, **_kwargs):
        called["write"] = True
        raise AssertionError("SHA mismatch must precede output writing.")

    monkeypatch.setattr(pool_tool, "_load_inputs", unexpected_load)
    monkeypatch.setattr(pool_tool, "_write_outputs", unexpected_write)
    output = tmp_path / "pool.h5"
    argv: list[str] = []
    for option, path in source_paths.items():
        argv.extend([f"--{option}", str(path)])
        digest = (
            pool_tool.ACS_2022_RENT_ARTIFACT_SHA256
            if option == "acs-rent-h5"
            else "0" * 64
        )
        argv.extend([f"--{option}-sha256", digest])
    argv.extend(
        [
            "--puma-ladder",
            str(tmp_path / "puma-ladder.npz"),
            "--puma-ladder-sha256",
            pool_tool._STACKED_PUMA_LADDER_SHA256,
            "--congressional-district-vintage-crosswalk",
            str(tmp_path / "congressional-district-vintage-crosswalk.csv"),
            "--congressional-district-vintage-crosswalk-sha256",
            pool_tool._STACKED_CD_CROSSWALK_SHA256,
        ]
    )
    argv.extend(["--out", str(output)])

    with pytest.raises(ValueError, match="ASEC raw-stage.*SHA-256 mismatch"):
        pool_tool.main(argv)

    assert called == {"load": False, "write": False}
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
    assert not output.with_suffix(".agreement.json").exists()


def test_primary_qrf_resume_refuses_a_changed_input_binding(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "primary-qrf"
    checkpoint_dir.mkdir()
    (checkpoint_dir / pool_tool.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    original = {
        "artifact_kind": "fixture_binding",
        "schema_version": 1,
        "inputs": {"processed_puf": {"sha256": "a" * 64}},
    }
    pool_tool._atomic_write_json(
        checkpoint_dir / pool_tool._PRIMARY_QRF_INPUT_BINDING_FILENAME,
        original,
    )
    changed = {
        **original,
        "inputs": {"processed_puf": {"sha256": "b" * 64}},
    }

    with pytest.raises(ValueError, match="refusing to reuse stale predictions"):
        pool_tool._initialize_or_resume_primary_qrf(
            _source_frame(),
            pd.DataFrame(),
            checkpoint_dir,
            input_binding=changed,
        )


@pytest.mark.parametrize(
    ("interruption_label", "durable_target_index"),
    (
        ("j0", 0),
        ("j1", 1),
        ("j2", 2),
        ("mid", 4),
        ("last", 8),
    ),
)
def test_production_transferred_checkpoint_bytes_ignore_bank_interruption_provenance(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interruption_label: str,
    durable_target_index: int,
) -> None:
    active_bank_receipt: dict[str, Mapping[str, object]] = {}
    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
        active_bank_receipt=active_bank_receipt,
    )

    canonical_acs_receipt_keys = {
        "target_families",
        "n_estimators",
        "max_targets_per_fit",
        "resolved_donor_channel",
        "imputed_inputs",
        "fit_records",
        "deferred_inputs",
    }
    canonical_primary_receipt_keys = {
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
        "input_binding",
        "input_binding_sha256",
        "n_estimators",
        "tail_bound_diagnostics",
    }
    canonical_impute_receipt_keys = {
        "source_operator_chain",
        "primary_puf_qrf",
        "puf_capital_gains_tail_transfer",
        "acs_qrf_transfer",
        "weights_audit",
    }
    cold_receipt = _target_bank_receipt_after_interruption(None)
    cold_store = _checkpoint_fixture_store(pool_tool, tmp_path / "cold-checkpoints")
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    primary_qrf_dir = tmp_path / "primary-qrf" / cold_store.base_identity_sha256
    acs_transfer_dir = tmp_path / "acs-transfer" / cold_store.base_identity_sha256
    active_bank_receipt["value"] = cold_receipt
    cold_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=cold_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )
    cold_bytes = cold_store.checkpoint_path("transferred").read_bytes()
    cold_sidecar = pool_tool._read_json_object(
        cold_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]
    cold_primary_operational = cold_sidecar["impute"]["primary_puf_qrf"]
    assert cold_primary_operational["resume_status"] == "initialized"
    assert cold_primary_operational["identity_routing"]["identity_mismatches"] == []
    cold_target_bank = copy.deepcopy(
        cold_sidecar["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    assert cold_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert cold_target_bank == cold_receipt
    assert (
        cold_result.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "initialized"
    )

    resumed_receipt = _target_bank_receipt_after_interruption(durable_target_index)
    resumed_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    resumed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    assert resumed_store.base_identity_sha256 == cold_store.base_identity_sha256
    active_bank_receipt["value"] = resumed_receipt
    resumed_result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=resumed_store,
        primary_qrf_checkpoint_dir=primary_qrf_dir,
        acs_transfer_checkpoint_dir=acs_transfer_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    transferred_path = resumed_store.checkpoint_path("transferred")
    assert transferred_path.read_bytes() == cold_bytes
    metadata = pool_tool.load_frame_checkpoint(transferred_path).metadata
    canonical_receipts = metadata["stage_receipts"]
    assert set(canonical_receipts) == {"clone", "impute"}
    canonical_impute_receipt = canonical_receipts["impute"]
    assert set(canonical_impute_receipt) == canonical_impute_receipt_keys
    canonical_primary_receipt = canonical_impute_receipt["primary_puf_qrf"]
    assert set(canonical_primary_receipt) == canonical_primary_receipt_keys
    assert "resume_status" not in canonical_primary_receipt
    canonical_acs_receipt = canonical_impute_receipt["acs_qrf_transfer"]
    assert set(canonical_acs_receipt) == canonical_acs_receipt_keys
    assert "target_bank" not in canonical_acs_receipt

    receipts_path = resumed_store.checkpoint_receipts_path("transferred")
    sidecar = pool_tool._read_json_object(receipts_path)
    operational_impute = sidecar["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "resumed"
    assert (
        operational_impute["primary_puf_qrf"]["identity_routing"]["identity_mismatches"]
        == []
    )
    resumed_target_bank = copy.deepcopy(
        operational_impute["acs_qrf_transfer"]["target_bank"]
    )
    assert resumed_target_bank.pop("identity_routing")["identity_mismatches"] == []
    assert resumed_target_bank == resumed_receipt

    resumed_store.checkpoint_path("simulated").unlink()
    resumed_store.checkpoint_manifest_path("simulated").unlink()
    warm_store = _checkpoint_fixture_store(
        pool_tool,
        tmp_path / f"{interruption_label}-checkpoints",
    )
    checkpoint = warm_store.load_deepest()
    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert (
        checkpoint.stage_receipts["impute"]["primary_puf_qrf"]["resume_status"]
        == "resumed"
    )
    restored_target_bank = copy.deepcopy(
        checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    restored_target_bank.pop("identity_routing")
    assert restored_target_bank == resumed_receipt
    runtime_target_bank = copy.deepcopy(
        resumed_result.stage_receipts["impute"]["acs_qrf_transfer"]["target_bank"]
    )
    runtime_target_bank.pop("identity_routing")
    assert runtime_target_bank == resumed_receipt
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "loaded"
    assert receipts_provenance["path"] == str(receipts_path.resolve())


def test_transferred_checkpoint_bytes_ignore_stacked_qrf_manifest_location(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_bytes: list[bytes] = []
    manifest_locations: list[str] = []
    identity_digests: list[str] = []
    for label in ("volume-a", "volume-b"):
        (tmp_path / label).mkdir()
        store = _checkpoint_fixture_store(
            pool_tool,
            tmp_path / label / "checkpoints",
        )
        store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        manifest_path = tmp_path / label / "primary-qrf" / "manifest.json"
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=store,
            primary_qrf_manifest_path=manifest_path,
        )
        transferred_path = store.checkpoint_path("transferred")
        checkpoint_bytes.append(transferred_path.read_bytes())
        metadata = pool_tool.load_frame_checkpoint(transferred_path).metadata
        identity_digests.append(metadata["identity_sha256"])
        assert (
            "checkpoint_manifest_path"
            not in metadata["stage_receipts"]["impute"]["primary_puf_qrf"]
        )
        sidecar = pool_tool._read_json_object(
            store.checkpoint_receipts_path("transferred")
        )
        manifest_locations.append(
            sidecar["operational_stage_receipts"]["impute"]["primary_puf_qrf"][
                "checkpoint_manifest_path"
            ]
        )

    assert checkpoint_bytes[0] == checkpoint_bytes[1]
    assert identity_digests[0] == identity_digests[1]
    assert manifest_locations == [
        str((tmp_path / label / "primary-qrf" / "manifest.json").resolve())
        for label in ("volume-a", "volume-b")
    ]


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_operational_receipts_sidecar_damage_does_not_invalidate_checkpoint(
    pool_tool: ModuleType,
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / f"{damage}-checkpoints"
    target_bank_receipt = _target_bank_receipt_after_interruption(2)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=target_bank_receipt,
    )
    transferred_path = cold_store.checkpoint_path("transferred")
    canonical_bytes = transferred_path.read_bytes()
    expected_identity_sha256 = pool_tool.load_frame_checkpoint(
        transferred_path
    ).metadata["identity_sha256"]
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    if damage == "missing":
        receipts_path.unlink()
        expected_status = "missing"
    else:
        receipts_path.write_text("{not-json", encoding="utf-8")
        expected_status = "invalid_ignored"

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert transferred_path.read_bytes() == canonical_bytes
    assert (
        pool_tool.load_frame_checkpoint(transferred_path).metadata["identity_sha256"]
        == expected_identity_sha256
    )
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == expected_status


def test_same_identity_rewrite_cannot_reattach_stale_operational_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "rewrite-checkpoints"
    receipt_a = _target_bank_receipt_after_interruption(0)
    cold_store = _checkpoint_fixture_store(pool_tool, root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        target_bank_receipt=receipt_a,
    )
    receipts_path = cold_store.checkpoint_receipts_path("transferred")
    assert receipts_path.is_file()
    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    cold_store.checkpoint_receipts_path("simulated").unlink()

    real_atomic_write_json = pool_tool._atomic_write_json

    def interrupt_before_receipts_install(path: Path, payload: Mapping[str, object]):
        if Path(path) == receipts_path:
            assert not receipts_path.exists()
            raise RuntimeError("fixture crash before fresh receipts install")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        pool_tool,
        "_atomic_write_json",
        interrupt_before_receipts_install,
    )
    rewrite_store = _checkpoint_fixture_store(pool_tool, root)
    rewrite_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    with pytest.raises(RuntimeError, match="before fresh receipts install"):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=rewrite_store,
            target_bank_receipt=_target_bank_receipt_after_interruption(4),
        )
    assert not receipts_path.exists()

    warm_store = _checkpoint_fixture_store(pool_tool, root)
    checkpoint = warm_store.load_deepest()

    assert checkpoint is not None
    assert checkpoint.stage == "transferred"
    assert "target_bank" not in checkpoint.stage_receipts["impute"]["acs_qrf_transfer"]
    receipts_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf"
    )["stages"]["transferred"]["receipts_sidecar"]
    assert receipts_provenance["load_status"] == "missing"


@pytest.mark.parametrize(
    ("resume_stage", "expected_order"),
    (
        ("assembled", ["impute", "derive", "seed", "simulate"]),
        ("transferred", ["derive", "seed", "simulate"]),
        ("simulated", []),
    ),
)
def test_pool_checkpoint_round_trip_resumes_each_boundary_byte_identically(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    resume_stage: str,
    expected_order: list[str],
) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    uninterrupted, cold_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    assert cold_order == ["impute", "derive", "seed", "simulate"]
    cold_output = capsys.readouterr().out
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert cold_output.count(f"Rebuilt pool stage {stage!r}") == 1
    checkpoint_identity_sha256: dict[str, str] = {}
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stored_metadata = pool_tool.load_frame_checkpoint(
            cold_store.checkpoint_path(stage)
        ).metadata
        checkpoint_identity_sha256[stage] = stored_metadata["identity_sha256"]
        assert "agreement_gate" not in stored_metadata
        assert "simulation_ready" not in stored_metadata
        assert "agreement" not in stored_metadata
        assert "agreement" not in stored_metadata["stage_receipts"]

    resume_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(resume_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[resume_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == resume_stage
    resumed, warm_order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert warm_order == expected_order
    warm_output = capsys.readouterr().out
    cached_stages = pool_tool.POOL_CHECKPOINT_STAGE_ORDER[: resume_index + 1]
    assert (
        f"Resumed pool checkpoint {resume_stage!r} from " in warm_output
        and f"cached stages: {', '.join(cached_stages)}." in warm_output
    )
    for stage_index, stage in enumerate(pool_tool.POOL_CHECKPOINT_STAGE_ORDER):
        assert warm_output.count(f"Rebuilt pool stage {stage!r}") == (
            0 if stage_index <= resume_index else 1
        )
    assert resumed.assembly_receipt == uninterrupted.assembly_receipt
    assert resumed.provenance_counts == uninterrupted.provenance_counts
    assert resumed.stage_receipts == uninterrupted.stage_receipts
    assert pool_tool._json_ready(resumed.frame.metadata) == pool_tool._json_ready(
        uninterrupted.frame.metadata
    )
    for entity in uninterrupted.frame.entities:
        pd.testing.assert_frame_equal(
            resumed.frame.table(entity),
            uninterrupted.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
        string_columns = _semantic_string_columns(uninterrupted.frame.table(entity))
        assert string_columns
        assert all(
            uninterrupted.frame.table(entity)[column].dtype == CANONICAL_STRING_DTYPE
            for column in string_columns
        )

    uninterrupted_pool_path = tmp_path / f"{resume_stage}.uninterrupted.pool.h5"
    resumed_pool_path = tmp_path / f"{resume_stage}.resumed.pool.h5"
    for path, result in (
        (uninterrupted_pool_path, uninterrupted),
        (resumed_pool_path, resumed),
    ):
        pool_tool.write_nullable_us_h5(
            result.frame,
            path,
            period=pool_tool.POOL_TIME_PERIOD,
            artifact_kind=pool_tool.POOL_H5_ARTIFACT_KIND,
            publication_run_id="fixture-publication",
        )
    with (
        pd.HDFStore(uninterrupted_pool_path, mode="r") as uninterrupted_store,
        pd.HDFStore(resumed_pool_path, mode="r") as resumed_store,
    ):
        assert resumed_store.keys() == uninterrupted_store.keys()
        for key in uninterrupted_store.keys():
            expected = uninterrupted_store[key]
            observed = resumed_store[key]
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )
            else:
                pd.testing.assert_series_equal(
                    observed,
                    expected,
                    check_dtype=True,
                    check_exact=True,
                )

    # PyTables' published container carries nondeterministic HDF metadata. The
    # deterministic Frame serializer gives the literal byte-level assertion
    # over the exact input-pool state after the production writer is checked.
    uninterrupted_canonical = tmp_path / f"{resume_stage}.uninterrupted.canonical.h5"
    resumed_canonical = tmp_path / f"{resume_stage}.resumed.canonical.h5"
    pool_tool.write_frame_checkpoint(uninterrupted_canonical, uninterrupted.frame)
    pool_tool.write_frame_checkpoint(resumed_canonical, resumed.frame)
    assert resumed_canonical.read_bytes() == uninterrupted_canonical.read_bytes()

    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] == resume_stage
    assert provenance["base_identity_sha256"] == warm_store.base_identity_sha256
    assert (
        provenance["primary_qrf"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert (
        provenance["acs_transfer"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    assert provenance["acs_transfer"]["boundary_stage"] == "transferred"
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert (
            provenance["stages"][stage]["identity_sha256"]
            == checkpoint_identity_sha256[stage]
        )
    assert provenance["stages"][resume_stage]["source"] == "checkpoint"
    assert provenance["stages"][resume_stage]["resume_kind"] == "direct"
    for covered_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[:resume_index]:
        covered = provenance["stages"][covered_stage]
        assert covered["source"] == "checkpoint"
        assert covered["resume_kind"] == "covered_by_deeper_checkpoint"
        assert covered["source_checkpoint_stage"] == resume_stage
        assert covered["path"] == str(
            warm_store.checkpoint_path(resume_stage).resolve()
        )
        assert covered["nominal_stage_path"] == str(
            warm_store.checkpoint_path(covered_stage).resolve()
        )
    assert provenance["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }


def test_pool_checkpoint_store_round_trips_nullable_boolean_families(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "nullable-boolean-checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
        checkpoint_nullable_booleans=True,
    )

    h5py = pytest.importorskip("h5py")
    for stage, expected_schema in (
        ("assembled", 2),
        ("transferred", 3),
        ("simulated", 3),
    ):
        path = cold_store.checkpoint_path(stage)
        with h5py.File(path, mode="r") as h5:
            raw = np.asarray(h5["_populace_frame_checkpoint/metadata_json"]).tobytes()
        assert json.loads(raw)["schema_version"] == expected_schema
        manifest = pool_tool._read_json_object(
            cold_store.checkpoint_manifest_path(stage)
        )
        assert manifest["materializer_version"] == 7
        loaded = pool_tool.load_frame_checkpoint(path).frame
        if stage == "assembled":
            assert "fixture_declared_boolean" not in loaded.person
            continue
        assert loaded.person["is_female"].dtype == pd.BooleanDtype()
        assert loaded.person["fixture_declared_boolean"].dtype == pd.BooleanDtype()
        assert not loaded.person["is_female"].isna().any()
        assert loaded.person["fixture_declared_boolean"].isna().sum() == 1

    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resumed = warm_store.load_deepest()

    assert resumed is not None
    assert resumed.stage == "transferred"
    assert resumed.frame.person["is_female"].dtype == pd.BooleanDtype()
    assert resumed.frame.person["fixture_declared_boolean"].isna().sum() == 1


def test_simulated_v7_checkpoint_accepts_both_string_encodings_without_rewrite(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    """V7 authenticates both physical string encodings as one logical frame."""

    pytest.importorskip("h5py")
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    checkpoint_path = cold_store.checkpoint_path("simulated")
    loaded = pool_tool.load_frame_checkpoint(checkpoint_path)
    canonical_v2_bytes = checkpoint_path.read_bytes()
    canonical_identity = loaded.metadata["identity"]
    assert loaded.metadata["materializer_version"] == 7
    assert any(
        column["dtype"] == str(CANONICAL_STRING_DTYPE)
        for columns in loaded.metadata["frame_schema"]["entities"].values()
        for column in columns
    )

    canonical_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    canonical_resume = canonical_store.load_deepest()
    assert canonical_resume is not None
    assert canonical_resume.stage == "simulated"
    assert canonical_resume.simulation_frame is not None
    assert checkpoint_path.read_bytes() == canonical_v2_bytes

    legacy_frame = _with_object_backed_strings(loaded.frame)
    legacy_metadata = dict(loaded.metadata)
    legacy_metadata["frame_schema"] = pool_tool._frame_schema_payload(legacy_frame)
    pool_tool.write_frame_checkpoint(
        checkpoint_path,
        legacy_frame,
        metadata=legacy_metadata,
    )
    manifest_path = cold_store.checkpoint_manifest_path("simulated")
    manifest = pool_tool._read_json_object(manifest_path)
    manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(checkpoint_path)
    manifest["checkpoint"]["size_bytes"] = checkpoint_path.stat().st_size
    manifest["frame_schema"] = legacy_metadata["frame_schema"]
    pool_tool._atomic_write_json(manifest_path, manifest)
    banked_v2_bytes = checkpoint_path.read_bytes()
    assert banked_v2_bytes != canonical_v2_bytes
    assert legacy_metadata["identity"] == canonical_identity
    assert legacy_metadata["materializer_version"] == 7
    assert any(
        column["dtype"] == "object"
        for columns in legacy_metadata["frame_schema"]["entities"].values()
        for column in columns
    )

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "simulated"
    assert resume.simulation_frame is not None
    assert checkpoint_path.read_bytes() == banked_v2_bytes
    assert (
        warm_store.provenance(
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        )["stages"]["simulated"]["load_status"]
        == "resumed"
    )
    for entity in US_SCHEMA.entities:
        canonical_table = canonical_resume.frame.table(entity)
        table = resume.frame.table(entity)
        pd.testing.assert_frame_equal(table, canonical_table, check_exact=True)
        pd.testing.assert_frame_equal(
            resume.simulation_frame.table(entity),
            canonical_resume.simulation_frame.table(entity),
            check_exact=True,
        )
        string_columns = _semantic_string_columns(table)
        assert string_columns
        assert all(
            table[column].dtype == CANONICAL_STRING_DTYPE for column in string_columns
        )


def test_resumed_checkpoint_provenance_is_published_in_final_manifest(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _unused, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    base_identity = pool_tool._pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version="fixture-engine-1",
    )
    cold_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    cold_store.bind_input_receipts(pool_tool._loaded_input_receipts(loaded))
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    warm_store = pool_tool._PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=base_identity,
    )
    resume = warm_store.load_deepest()
    assert resume is not None
    assert resume.stage == "simulated"
    resumed, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )
    assert order == []
    checkpoint_provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
    )

    pool_tool._write_outputs(
        resumed,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        input_receipts=warm_store.input_receipts,
        checkpoint_provenance=checkpoint_provenance,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["stage_checkpoints"] == checkpoint_provenance
    assert (
        manifest["stage_checkpoints"]["base_identity_sha256"]
        == warm_store.base_identity_sha256
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        stage_manifest = pool_tool._read_json_object(
            warm_store.checkpoint_manifest_path(stage)
        )
        assert (
            manifest["stage_checkpoints"]["stages"][stage]["identity_sha256"]
            == stage_manifest["identity_sha256"]
        )


def test_pool_checkpoint_input_sha_mismatch_rebuilds_every_stage(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    changed_store = _checkpoint_fixture_store(
        pool_tool,
        checkpoint_root,
        changed_role="processed_puf",
    )
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    output = capsys.readouterr().out
    assert output.count("Ignored stale pool checkpoint") == 3
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_production_bank_routing_names_stale_contract_identity_siblings(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "production-routing-checkpoints"
    base_outputs = pool_tool._output_paths(
        tmp_path / "pool.h5",
        checkpoint_root=checkpoint_root,
    )
    contract = load_take_up_contract()
    current_contract_identity = take_up_contract_identity(contract)
    stale_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-stale"},
    )
    stale_contract_identity = take_up_contract_identity(stale_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: stale_contract_identity,
    )
    stale_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    stale_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=stale_store.base_identity_sha256,
    )
    stale_markers = []
    for bank_dir in (
        stale_outputs.primary_qrf_checkpoint_dir,
        stale_outputs.acs_transfer_checkpoint_dir,
    ):
        bank_dir.mkdir(parents=True)
        marker = bank_dir / "must-not-be-opened"
        marker.write_text("stale bank marker\n", encoding="utf-8")
        stale_markers.append(marker)

    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: current_contract_identity,
    )
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    current_outputs = pool_tool._with_checkpoint_identity(
        base_outputs,
        base_identity_sha256=current_store.base_identity_sha256,
    )
    assert current_store.base_identity_sha256 != stale_store.base_identity_sha256
    assert (
        current_outputs.primary_qrf_checkpoint_dir
        != stale_outputs.primary_qrf_checkpoint_dir
    )
    assert (
        current_outputs.acs_transfer_checkpoint_dir
        != stale_outputs.acs_transfer_checkpoint_dir
    )

    checkpoint_input_binding = _stub_production_impute_kernels(
        pool_tool,
        monkeypatch,
    )
    current_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    result = _run_production_impute_checkpoint_fixture(
        pool_tool,
        store=current_store,
        primary_qrf_checkpoint_dir=current_outputs.primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=current_outputs.acs_transfer_checkpoint_dir,
        checkpoint_input_binding=checkpoint_input_binding,
    )

    operational_impute = pool_tool._read_json_object(
        current_store.checkpoint_receipts_path("transferred")
    )["operational_stage_receipts"]["impute"]
    assert operational_impute["primary_puf_qrf"]["resume_status"] == "initialized"
    primary_routing = operational_impute["primary_puf_qrf"]["identity_routing"]
    target_bank = operational_impute["acs_qrf_transfer"]["target_bank"]
    acs_routing = target_bank["identity_routing"]
    assert target_bank["root"] == str(
        current_outputs.acs_transfer_checkpoint_dir.resolve()
    )
    assert target_bank["targets"] == {}

    for routing, bank_root, selected_path, stale_path in (
        (
            primary_routing,
            current_outputs.primary_qrf_checkpoint_dir.parent,
            current_outputs.primary_qrf_checkpoint_dir,
            stale_outputs.primary_qrf_checkpoint_dir,
        ),
        (
            acs_routing,
            current_outputs.acs_transfer_checkpoint_dir.parent,
            current_outputs.acs_transfer_checkpoint_dir,
            stale_outputs.acs_transfer_checkpoint_dir,
        ),
    ):
        assert routing["bank_root"] == str(bank_root.resolve())
        assert routing["selected_path"] == str(selected_path.resolve())
        assert (
            routing["current_base_identity_sha256"]
            == current_store.base_identity_sha256
        )
        assert routing["scan"] == {
            "limit": pool_tool._BANK_IDENTITY_SIBLING_SCAN_LIMIT,
            "entries_examined": 1,
            "truncated": False,
        }
        assert routing["identity_mismatches"] == [
            {
                "load_status": "identity_mismatch",
                "stale_base_identity_sha256": stale_store.base_identity_sha256,
                "current_base_identity_sha256": current_store.base_identity_sha256,
                "disposition": "bypassed",
                "path": str(stale_path.resolve()),
            }
        ]
    assert all(
        marker.read_text(encoding="utf-8") == "stale bank marker\n"
        for marker in stale_markers
    )
    assert (
        result.stage_receipts["impute"]["primary_puf_qrf"]["identity_routing"]
        == primary_routing
    )


@pytest.mark.parametrize(
    "contract_field",
    (
        pytest.param("asserted_constraint", id="asserted_constraint_changed"),
        pytest.param(
            "inventory_built_against",
            id="inventory_built_against_changed",
        ),
    ),
)
def test_take_up_contract_identity_mutation_rebuilds_every_pool_boundary(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract_field: str,
) -> None:
    checkpoint_root = tmp_path / "take-up-contract-checkpoints"
    contract = load_take_up_contract()
    original_contract_identity = take_up_contract_identity(contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: original_contract_identity,
    )
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert (
        cold_store.base_identity["pool_code"]["take_up_contract"]
        == original_contract_identity
    )
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    changed_contract = replace(
        contract,
        **{contract_field: f"{getattr(contract, contract_field)}-changed"},
    )
    changed_contract_identity = take_up_contract_identity(changed_contract)
    monkeypatch.setattr(
        pool_tool,
        "take_up_contract_identity",
        lambda: changed_contract_identity,
    )
    changed_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)

    assert (
        changed_store.base_identity["pool_code"]["take_up_contract"]
        == changed_contract_identity
    )
    assert changed_store.base_identity_sha256 != cold_store.base_identity_sha256
    assert changed_store.load_deepest() is None
    changed_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=changed_store,
    )

    assert order == ["impute", "derive", "seed", "simulate"]
    assert not rebuilt.simulation_ready
    provenance = changed_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    assert provenance["deepest_resumed_stage"] is None
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert provenance["stages"][stage]["source"] == "rebuilt"
        assert provenance["stages"][stage]["load_status"] == "identity_mismatch"


def test_tail_support_contract_identity_mutation_rebuilds_pool_checkpoints(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "tail-support-contract-checkpoints"
    original = pool_tool.puf_capital_gains_tail_support_contract_identity()
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert (
        cold_store.base_identity["pool_code"]["puf_capital_gains_tail_support_contract"]
        == original
    )
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)

    changed = copy.deepcopy(original)
    changed["insufficient_support_action"] = "silently_widen"
    monkeypatch.setattr(
        pool_tool,
        "puf_capital_gains_tail_support_contract_identity",
        lambda: changed,
    )
    changed_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)

    assert (
        changed_store.base_identity["pool_code"][
            "puf_capital_gains_tail_support_contract"
        ]
        == changed
    )
    assert changed_store.base_identity_sha256 != cold_store.base_identity_sha256
    assert changed_store.load_deepest() is None


@pytest.mark.parametrize("legacy_version", (1, 2, 3, 4, 5, 6))
def test_legacy_pool_materializer_artifacts_fail_closed_with_named_receipts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_version: int,
) -> None:
    checkpoint_root = tmp_path / "legacy-materializer-checkpoints"

    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
            legacy_version,
        )
        legacy_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
        assert legacy_store.base_identity["materializer_version"] == legacy_version
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        _run_checkpoint_fixture(pool_tool, tmp_path, store=legacy_store)
        for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
            metadata = pool_tool.load_frame_checkpoint(
                legacy_store.checkpoint_path(stage)
            ).metadata
            manifest = pool_tool._read_json_object(
                legacy_store.checkpoint_manifest_path(stage)
            )
            assert metadata["materializer_version"] == legacy_version
            assert metadata["identity"]["materializer_version"] == legacy_version
            assert manifest["materializer_version"] == legacy_version
            assert manifest["identity"]["materializer_version"] == legacy_version
    capsys.readouterr()

    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    current_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    assert current_store.base_identity["materializer_version"] == 7
    assert current_store.load_deepest() is None

    output = capsys.readouterr().out
    provenance = current_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    for stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER:
        assert f"Ignored corrupt pool checkpoint '{stage}'" in output
        receipt = provenance["stages"][stage]
        assert receipt["source"] == "rebuilt"
        assert receipt["load_status"] == "invalid_rebuild"
        invalid = receipt["invalid_checkpoint"]
        assert invalid["reason"] == "checkpoint_validation_failed"
        assert invalid["message"] == (
            f"{stage} checkpoint manifest has an unsupported binding"
        )


@pytest.mark.parametrize(
    ("corrupt_stage", "expected_resume", "expected_order"),
    (
        (
            "assembled",
            None,
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "transferred",
            "assembled",
            ["impute", "derive", "seed", "simulate"],
        ),
        (
            "simulated",
            "transferred",
            ["derive", "seed", "simulate"],
        ),
    ),
)
def test_corrupt_pool_checkpoint_names_boundary_and_rebuilds(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corrupt_stage: str,
    expected_resume: str | None,
    expected_order: list[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    baseline, _order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=cold_store,
    )
    capsys.readouterr()

    corrupt_index = pool_tool.POOL_CHECKPOINT_STAGE_ORDER.index(corrupt_stage)
    for deeper_stage in pool_tool.POOL_CHECKPOINT_STAGE_ORDER[corrupt_index + 1 :]:
        cold_store.checkpoint_path(deeper_stage).unlink()
        cold_store.checkpoint_manifest_path(deeper_stage).unlink()
    cold_store.checkpoint_path(corrupt_stage).write_bytes(b"corrupt checkpoint")

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()
    assert (None if resume is None else resume.stage) == expected_resume
    if resume is None:
        warm_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    rebuilt, order = _run_checkpoint_fixture(
        pool_tool,
        tmp_path,
        store=warm_store,
        resume=resume,
    )

    assert order == expected_order
    output = capsys.readouterr().out
    assert f"Ignored corrupt pool checkpoint '{corrupt_stage}'" in output
    assert "SHA-256 mismatch" in output
    for entity in baseline.frame.entities:
        pd.testing.assert_frame_equal(
            rebuilt.frame.table(entity),
            baseline.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
    provenance = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )
    corrupt_receipt = provenance["stages"][corrupt_stage]
    assert corrupt_receipt["source"] == "rebuilt"
    assert corrupt_receipt["load_status"] == "invalid_rebuild"
    assert corrupt_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


@pytest.mark.parametrize(
    ("drift_field", "expected_error"),
    (
        ("row_counts", "row_counts differs from its sidecar"),
        ("frame_schema", "frame_schema differs from its sidecar"),
    ),
)
def test_checkpoint_sidecar_shape_drift_names_failure_and_falls_back(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drift_field: str,
    expected_error: str,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    cold_store.checkpoint_path("simulated").unlink()
    cold_store.checkpoint_manifest_path("simulated").unlink()
    transferred_manifest_path = cold_store.checkpoint_manifest_path("transferred")
    transferred_manifest = pool_tool._read_json_object(transferred_manifest_path)
    if drift_field == "row_counts":
        transferred_manifest["row_counts"]["person"] += 1
    else:
        transferred_manifest["frame_schema"]["entities"]["person"][0]["dtype"] = (
            "corrupt-dtype"
        )
    pool_tool._atomic_write_json(transferred_manifest_path, transferred_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "assembled"
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'transferred'" in output
    assert expected_error in output
    transferred_receipt = warm_store.provenance(
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
    )["stages"]["transferred"]
    assert transferred_receipt["load_status"] == "invalid_rebuild"
    assert transferred_receipt["invalid_checkpoint"]["reason"] == (
        "checkpoint_validation_failed"
    )


def test_invalid_deep_checkpoint_receipts_do_not_poison_valid_fallback(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    simulated_path = cold_store.checkpoint_path("simulated")
    simulated = pool_tool.load_frame_checkpoint(simulated_path)
    poisoned_metadata = dict(simulated.metadata)
    poisoned_metadata["input_receipts"] = {"poisoned": True}
    poisoned_metadata["simulation_output"] = None
    pool_tool.write_frame_checkpoint(
        simulated_path,
        simulated.frame,
        metadata=poisoned_metadata,
    )
    simulated_manifest_path = cold_store.checkpoint_manifest_path("simulated")
    simulated_manifest = pool_tool._read_json_object(simulated_manifest_path)
    simulated_manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(simulated_path)
    simulated_manifest["checkpoint"]["size_bytes"] = simulated_path.stat().st_size
    pool_tool._atomic_write_json(simulated_manifest_path, simulated_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "transferred"
    assert warm_store.input_receipts == _checkpoint_fixture_input_receipts()
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'simulated'" in output
    assert "invalid SSI output binding" in output


def test_durable_checkpoint_write_rejects_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    def forge_after_emission(checkpoint: MultispinePoolCheckpoint) -> None:
        if checkpoint.stage != "simulated":
            store.write(checkpoint)
            return
        receipts = copy.deepcopy(checkpoint.stage_receipts)
        receipts["derive"]["qbi_input_reconciliation"]["sha256"] = "0" * 64
        store.write(replace(checkpoint, stage_receipts=receipts))

    with pytest.raises(
        ValueError,
        match=(
            "pool simulated durable checkpoint write: QBI reconciliation "
            "receipt SHA-256"
        ),
    ):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=SimpleNamespace(write=forge_after_emission),
        )

    assert not store.checkpoint_path("simulated").exists()


def test_durable_checkpoint_write_rejects_forged_qbi_transition_authority(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    store.bind_input_receipts(_checkpoint_fixture_input_receipts())

    def forge_after_emission(checkpoint: MultispinePoolCheckpoint) -> None:
        if checkpoint.stage != "simulated":
            store.write(checkpoint)
            return
        store.write(
            replace(
                checkpoint,
                qbi_transition_authority_sha256="0" * 64,
            )
        )

    with pytest.raises(
        ValueError,
        match="independently carried transition authority",
    ):
        _run_checkpoint_fixture(
            pool_tool,
            tmp_path,
            store=SimpleNamespace(write=forge_after_emission),
        )

    assert not store.checkpoint_path("simulated").exists()


def test_durable_checkpoint_load_rejects_forged_qbi_receipt_and_falls_back(
    pool_tool: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    cold_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    cold_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    _run_checkpoint_fixture(pool_tool, tmp_path, store=cold_store)
    capsys.readouterr()

    simulated_path = cold_store.checkpoint_path("simulated")
    simulated = pool_tool.load_frame_checkpoint(simulated_path)
    poisoned_metadata = copy.deepcopy(simulated.metadata)
    poisoned_metadata["stage_receipts"]["derive"]["qbi_input_reconciliation"][
        "sha256"
    ] = "0" * 64
    pool_tool.write_frame_checkpoint(
        simulated_path,
        simulated.frame,
        metadata=poisoned_metadata,
    )
    simulated_manifest_path = cold_store.checkpoint_manifest_path("simulated")
    simulated_manifest = pool_tool._read_json_object(simulated_manifest_path)
    simulated_manifest["checkpoint"]["sha256"] = pool_tool._file_sha256(simulated_path)
    simulated_manifest["checkpoint"]["size_bytes"] = simulated_path.stat().st_size
    pool_tool._atomic_write_json(simulated_manifest_path, simulated_manifest)

    warm_store = _checkpoint_fixture_store(pool_tool, checkpoint_root)
    resume = warm_store.load_deepest()

    assert resume is not None
    assert resume.stage == "transferred"
    output = capsys.readouterr().out
    assert "Ignored corrupt pool checkpoint 'simulated'" in output
    assert (
        "pool simulated durable checkpoint load: QBI reconciliation receipt SHA-256"
    ) in output


def test_synthetic_two_spine_path_reaches_fixed_red_terminal_gate(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result = _red_pool_result(pool_tool, tmp_path)

    assert result.assembly_receipt["channels"] == ["asec", "acs"]
    assert result.assembly_receipt["native_row_counts"]["person"] == {
        "asec": 2,
        "acs": 2,
    }
    assert result.provenance_counts["person"] == {
        "rows": 8,
        "by_source_channel": {"asec": 4, "acs": 4},
        "by_clone_index": {"0": 4, "1": 4},
        "by_source_channel_and_clone_index": {
            "asec": {"0": 2, "1": 2},
            "acs": {"0": 2, "1": 2},
        },
    }
    assert result.stage_receipts == {
        stage: {"fixture_stage": stage}
        for stage in ("impute", "derive", "seed", "simulate")
    }
    assert any(
        "person/simulated_output/ssi/acs_vs_asec" in failure
        for failure in result.agreement_gate.failures
    )


def test_wired_path_uses_real_raw_preserving_transfer_before_gate(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    transfer_receipts = []

    def impute(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["age"] = pd.to_numeric(person["A_AGE"], errors="raise")
        person["is_female"] = pd.to_numeric(person["A_SEX"], errors="raise") == 2
        prepared = _replace_person(frame, person)
        transferred = transfer_acs_inputs(
            prepared,
            prepared,
            target_families={"person": {"fixture": ("fixture_transfer",)}},
            seed=0,
            n_estimators=3,
        )
        transfer_receipts.extend(transferred.imputed_inputs)
        return PoolStageOutput(
            transferred.frame,
            {"fit_records": list(transferred.fit_records)},
        )

    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def simulate(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        person["ssi"] = person["fixture_transfer"]
        return PoolStageOutput(_replace_person(frame, person))

    result = pool_tool.build_multispine_pool(
        _transfer_source_frame([10.0, 20.0]),
        _transfer_source_frame([np.nan, np.nan]),
        puf_donor=pd.DataFrame(),
        primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
        impute=impute,
        derive=no_op,
        seed=no_op,
        simulate=simulate,
    )

    person = result.frame.table("person")
    channels = person[support_channel_column("person")]
    assert sorted(person.loc[channels.eq("asec"), "fixture_transfer"]) == [
        10.0,
        10.0,
        20.0,
        20.0,
    ]
    assert person.loc[channels.eq("acs"), "fixture_transfer"].tolist() == [
        15.0,
        15.0,
        15.0,
        15.0,
    ]
    assert sum(item.imputed_recipient_rows for item in transfer_receipts) == 4
    assert not result.agreement_gate.passed
    assert result.agreement_gate.name == "us_spine_agreement"
    assert "ssi" not in person


def test_manifest_and_publication_reject_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    receipts = copy.deepcopy(result.stage_receipts)
    receipts["derive"]["qbi_input_reconciliation"]["sha256"] = "0" * 64
    forged = replace(result, stage_receipts=receipts)

    with pytest.raises(
        ValueError,
        match=("legacy production manifest: QBI reconciliation receipt SHA-256"),
    ):
        pool_tool._manifest_payload(
            result=forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="forged-manifest",
        )

    with pytest.raises(
        ValueError,
        match="legacy publication entry: QBI reconciliation receipt SHA-256",
    ):
        pool_tool._write_outputs(
            forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_stacked_manifest_and_publication_reject_forged_qbi_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    legacy, _outputs, verified_inputs, source_manifest, _loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    receipt = copy.deepcopy(legacy.stage_receipts["derive"]["qbi_input_reconciliation"])
    receipt["sha256"] = "0" * 64
    authorized, impute, transition_authority_sha256 = _authorized_late_impute_fixture(
        pool_tool, legacy.frame
    )
    stacked = SimpleNamespace(
        frame=authorized,
        qbi_transition_authority_sha256=(legacy.qbi_transition_authority_sha256),
        late_producer_transition_authority_sha256=transition_authority_sha256,
        stage_receipts={
            "impute": impute,
            "derive": {"pool_derivation": {"qbi_input_reconciliation": receipt}},
        },
    )
    outputs = pool_tool._stacked_output_paths(tmp_path / "stacked-pool.h5")

    with pytest.raises(
        ValueError,
        match=("stacked production manifest: QBI reconciliation receipt SHA-256"),
    ):
        pool_tool._stacked_manifest_payload(
            result=stacked,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="forged-stacked-manifest",
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    with pytest.raises(
        ValueError,
        match="stacked publication entry: QBI reconciliation receipt SHA-256",
    ):
        pool_tool._write_stacked_outputs(
            stacked,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=579,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_publication_rejects_mutated_qbi_output_with_regenerated_receipt(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    person = result.frame.table("person").copy()
    person.loc[person.index[0], "non_qualified_dividend_income"] = 100.0
    person.loc[person.index[0], "qualified_bdc_income"] = 50.0
    mutated_frame = _replace_person(result.frame, person)
    receipts = copy.deepcopy(result.stage_receipts)
    receipts["derive"]["qbi_input_reconciliation"] = (
        us_qbi_reconciliation_change_receipt(mutated_frame, mutated_frame)
    )
    forged = replace(
        result,
        frame=mutated_frame,
        stage_receipts=receipts,
    )

    with pytest.raises(
        ValueError,
        match=(
            "legacy production manifest: QBI receipt differs from the "
            "independently carried transition authority"
        ),
    ):
        pool_tool._manifest_payload(
            result=forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            input_receipts={},
            checkpoint_provenance={},
            publication_run_id="reissued-qbi-manifest",
        )

    with pytest.raises(
        ValueError,
        match=(
            "legacy publication entry: QBI receipt differs from the "
            "independently carried transition authority"
        ),
    ):
        pool_tool._write_outputs(
            forged,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    assert not outputs.pool_h5.exists()
    assert not outputs.manifest.exists()
    assert not outputs.agreement_diagnostics.exists()


def test_red_outputs_preserve_receipts_and_exclude_simulation_output(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )

    pool_tool._write_outputs(
        result,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
    expected_gate = GateReport((result.agreement_gate,)).to_manifest()

    assert manifest["status"] == "agreement_failed"
    assert manifest["simulation_ready"] is False
    assert manifest["calibration_applied"] is False
    assert manifest["calibration"]["applied"] is False
    assert manifest["assembly_receipt"] == result.assembly_receipt
    assert manifest["provenance_counts"] == result.provenance_counts
    assert manifest["stage_receipts"] == result.stage_receipts
    assert manifest["stage_checkpoints"]["enabled"] is False
    assert manifest["stage_checkpoints"]["agreement"] == {
        "source": "always_fresh",
        "cached": False,
        "terminal_verdict_persisted": False,
    }
    assert manifest["agreement_gate"] == expected_gate
    assert diagnostics["agreement_gate"] == expected_gate
    assert diagnostics["simulation_ready"] is False
    assert diagnostics["publication_run_id"] == manifest["publication_run_id"]
    assert manifest["provenance_pins"] == {
        role: pin.to_manifest() for role, pin in verified_inputs.items()
    }
    assert manifest["pool_h5"]["formula_outputs_persisted"] is False
    assert manifest["pool_h5"]["input_only"] is True
    assert manifest["pool_h5"]["publication_run_id"] == manifest["publication_run_id"]
    assert (
        manifest["pool_h5"]["sha256"]
        == hashlib.sha256(outputs.pool_h5.read_bytes()).hexdigest()
    )

    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        assert "ssi" not in store["person"].columns
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == manifest["publication_run_id"]
    assert "materializer_version" not in metadata


def test_ready_reader_binds_manifest_h5_and_diagnostics_to_one_run(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )

    manifest = pool_tool.load_simulation_ready_us_multispine_pool_manifest(
        outputs.manifest
    )

    assert manifest["simulation_ready"] is True
    assert (
        manifest["agreement_diagnostics"]["publication_run_id"]
        == manifest["publication_run_id"]
    )


def test_ready_reader_rejects_manifest_h5_run_id_mismatch(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    ready = replace(
        result,
        agreement_gate=GateResult("us_spine_agreement", True),
    )
    pool_tool._write_outputs(
        ready,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=source_manifest,
        loaded=loaded,
    )
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    manifest["publication_run_id"] = "substituted-run"
    manifest["pool_h5"]["publication_run_id"] = "substituted-run"
    outputs.manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="H5.*run ID does not match"):
        pool_tool.load_simulation_ready_us_multispine_pool_manifest(outputs.manifest)


def test_h5_publication_failure_invalidates_stale_green_manifest(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "h5-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )

    def fail_h5(*_args, **_kwargs) -> None:
        _assert_publication_tombstone(
            pool_tool,
            outputs,
            publication_run_id=publication_run_id,
        )
        raise RuntimeError("injected H5 publication failure")

    monkeypatch.setattr(pool_tool, "write_nullable_us_h5", fail_h5)

    with pytest.raises(RuntimeError, match="injected H5 publication failure"):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"


def test_diagnostics_publication_failure_keeps_pool_not_ready(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "diagnostics-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    temporary_diagnostics = pool_tool._publication_temporary_path(
        outputs.agreement_diagnostics,
        publication_run_id=publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_diagnostics(path, payload) -> None:
        if Path(path) == temporary_diagnostics:
            raise RuntimeError("injected diagnostics publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_diagnostics)

    with pytest.raises(
        RuntimeError,
        match="injected diagnostics publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    assert outputs.pool_h5.read_bytes() == b"stale green h5"
    assert not pool_tool._publication_temporary_path(
        outputs.pool_h5,
        publication_run_id=publication_run_id,
    ).exists()


def test_final_manifest_failure_leaves_tombstone_as_readiness_authority(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    result, outputs, verified_inputs, source_manifest, loaded = _output_context(
        pool_tool,
        tmp_path,
    )
    _seed_stale_green_outputs(outputs)
    publication_run_id = "manifest-failure-run"
    monkeypatch.setattr(
        pool_tool,
        "_new_publication_run_id",
        lambda: publication_run_id,
    )
    atomic_write_json = pool_tool._atomic_write_json

    def fail_final_manifest(path, payload) -> None:
        if (
            Path(path) == outputs.manifest
            and payload["status"] != "publication_in_progress"
        ):
            raise RuntimeError("injected final manifest publication failure")
        atomic_write_json(path, payload)

    monkeypatch.setattr(pool_tool, "_atomic_write_json", fail_final_manifest)

    with pytest.raises(
        RuntimeError,
        match="injected final manifest publication failure",
    ):
        pool_tool._write_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=source_manifest,
            loaded=loaded,
        )

    _assert_publication_tombstone(
        pool_tool,
        outputs,
        publication_run_id=publication_run_id,
    )
    diagnostics = json.loads(outputs.agreement_diagnostics.read_text(encoding="utf-8"))
    assert diagnostics["publication_run_id"] == publication_run_id
    with pd.HDFStore(outputs.pool_h5, mode="r") as store:
        metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
    assert metadata["publication_run_id"] == publication_run_id


def test_clone_safe_id_error_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    asec = _source_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["person"].loc[0, "person_id"] = PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID + 1
    invalid = Frame(
        tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    def unreachable(_frame: Frame) -> PoolStageOutput:
        raise AssertionError("Assembly errors must precede every pool stage.")

    with pytest.raises(ValueError, match="Spine 'asec'.*clone-safe bound"):
        pool_tool.build_multispine_pool(
            invalid,
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=unreachable,
            derive=unreachable,
            seed=unreachable,
            simulate=unreachable,
        )


def test_assembly_receipt_loss_surfaces_unchanged_through_tool(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def drop_receipt(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(
            _replace_person(
                frame,
                frame.table("person").copy(),
                preserve_metadata=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="multispine pool derive output:.*no assembly manifest",
    ):
        pool_tool.build_multispine_pool(
            _source_frame(),
            _source_frame(),
            puf_donor=pd.DataFrame(),
            primary_qrf_checkpoint_dir=tmp_path / "unused-qrf",
            impute=no_op,
            derive=drop_receipt,
            seed=no_op,
            simulate=no_op,
        )


def test_local_artifact_reference_never_embeds_host_absolute_paths(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    """Exported row locations anchor to checkout, then home, never ``/``."""

    repo_file = Path(pool_tool.__file__).resolve()
    repo_reference = pool_tool._local_artifact_reference(repo_file)
    assert repo_reference == "local://tools/build_us_multispine_pool.py"

    home_path = Path.home() / "microcosm-test-unwritten" / "artifact.h5"
    home_reference = pool_tool._local_artifact_reference(home_path)
    assert home_reference == ("local://~/microcosm-test-unwritten/artifact.h5")

    outside_path = (tmp_path / "artifact.h5").resolve()
    outside_reference = pool_tool._local_artifact_reference(outside_path)
    assert outside_reference == f"local://{outside_path.as_posix().lstrip('/')}"

    for reference in (repo_reference, home_reference, outside_reference):
        assert not reference.startswith("local:///")
        assert "local://Users/" not in reference
