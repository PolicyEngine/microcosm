"""Tests split from packages/microcosm-build/tests/test_us_multispine_pool_tool.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_multispine_pool_tool import *


@pytest.mark.parametrize(
    ("terminal", "expected_code", "disposition"),
    [
        ("success", 0, "iterating"),
        ("red", 1, "failed"),
        ("error", None, "failed"),
    ],
)
def test_stacked_tool_entrypoint_fixture_e2e_emits_one_logbook_row_at_every_terminal_state(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal: str,
    expected_code: int | None,
    disposition: str,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    """Exercise the real tool, stack assembly, orchestrator, and publication shell."""
    order, full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal=terminal,
    )
    if terminal == "error":
        with pytest.raises(RuntimeError, match="fixture stacked error"):
            pool_tool.main(_stacked_main_argv(tmp_path))
    else:
        assert pool_tool.main(_stacked_main_argv(tmp_path)) == expected_code

    rows = list((tmp_path / "logbook-spool").glob("*.json"))
    assert len(rows) == 1
    row = load_logbook_row(rows[0])
    assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    assert row.disposition == disposition
    assert row.rung == "f001"
    assert row.seed == 578
    assert row.pipeline == "us-stacked-pool"
    assert len(row.input_pins_digest) == len(row.identity_digest) == 64
    assert "f001-s578-asec1-acs1" in row.build_id
    assert full_puf_rows == 7
    if terminal == "error":
        assert order == [
            "stack",
            "geography",
            "build_stacked_pool",
            "prepare",
            "gap",
            "puf",
        ]
        assert row.gate_verdicts["pipeline_error"]["verdict"] == "error"
        assert row.artifact_location is None
    else:
        assert order == [
            "stack",
            "geography",
            "build_stacked_pool",
            "prepare",
            "gap",
            "puf",
            "late_producer_dag",
            "tail_prepare",
            "derive",
            "seed",
            "simulate",
            "completeness",
            "battery",
            "publish",
        ]
        # Exported rows must never embed host-absolute paths; pytest tmp
        # directories live outside both the checkout and home on supported
        # platforms, so the reference lands on the stripped-absolute form.
        assert row.artifact_location == (
            "local://" + (tmp_path / "stacked-pool.h5").resolve().as_posix().lstrip("/")
        )
        expected_receipt_prefix = "local://" + (
            tmp_path / "logbook-receipts" / row.build_id
        ).resolve().as_posix().lstrip("/")
        assert all(
            verdict["receipt"].startswith(expected_receipt_prefix)
            for verdict in row.gate_verdicts.values()
        )
        manifest = json.loads(
            (tmp_path / "stacked-pool.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == pool_tool.POOL_MANIFEST_SCHEMA_VERSION
        assert manifest["pipeline"] == "us-stacked-pool"
        assert manifest["pool_h5"]["materializer_version"] == (
            pool_tool.US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        )
        with pd.HDFStore(manifest["pool_h5"]["path"], mode="r") as store:
            h5_metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
        assert h5_metadata["materializer_version"] == (
            pool_tool.US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        )
        published_dag = manifest["stage_receipts"]["impute"][
            "stacked_late_producer_dag"
        ]
        published_transfer = published_dag["post_puf_transfer"]
        assert published_transfer["completion"] == {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        }
        calibrated = [
            target["post_transfer_calibration"]
            for target in published_transfer["targets"].values()
            if "post_transfer_calibration" in target
        ]
        assert len(calibrated) == 7
        assert all(owner["context_binding"]["live_output"] for owner in calibrated)
        expected_late_authority_sha256 = (
            stacked_spine_module._late_producer_transition_authority_receipt(
                published_dag
            )["sha256"]
        )
        assert manifest["late_producer_transition_authority_sha256"] == (
            expected_late_authority_sha256
        )
        checkpoint_root = next(
            (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
        )
        for stage in ("transferred", "simulated"):
            checkpoint_path = checkpoint_root / f"{stage}.checkpoint.h5"
            checkpoint_manifest = json.loads(
                checkpoint_path.with_suffix(".manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint_metadata = pool_tool.load_frame_checkpoint(
                checkpoint_path
            ).metadata
            assert (
                checkpoint_manifest["late_producer_transition_authority_sha256"]
                == expected_late_authority_sha256
            )
            assert (
                checkpoint_metadata["late_producer_transition_authority_sha256"]
                == expected_late_authority_sha256
            )
        assert manifest["operator_order"] == [
            "assemble_stacked_spine",
            "assign_us_puma_ladder",
            "prepare_multispine_source_inputs_for_clone",
            "gap_fill_stacked_spine",
            "run_stacked_late_producer_dag",
            "prepare_stacked_tail_derivation",
            "derive_multispine_pool_inputs",
            "seed_multispine_pool_inputs",
            "materialize_multispine_agreement_outputs",
            "stacked_completeness_gate",
            "by_origin_battery",
        ]
        assert manifest["sampling"] == {
            **manifest["sampling"],
            "sample_fraction": 0.01,
            "fraction_token": "f001",
            "sample_seed": 578,
            "realized_households": {"asec": 1, "acs": 1},
        }


def test_stacked_pool_fixed_h5_reaches_release_cd_vintage_preflight(
    pool_tool: ModuleType,
    release_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py", exc_type=ModuleNotFoundError)
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
        real_geography_assignment=True,
    )

    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    assert order.index("stack") < order.index("geography") < order.index("prepare")

    pool_h5 = tmp_path / "stacked-pool.h5"
    with h5py.File(pool_h5, mode="r") as h5:
        assert (
            release_tool._h5_attr_text(
                h5.attrs,
                release_tool.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
            )
            == pool_tool._STACKED_CD_CROSSWALK_SHA256
        )
        assert (
            release_tool._h5_attr_text(
                h5.attrs,
                release_tool.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
            )
            == pool_tool.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
        )
    with pd.HDFStore(pool_h5, mode="r") as store:
        assert store.get_storer("household").format_type == "fixed"
        household = read_frame_table(store, "household")
    districts = pd.to_numeric(
        household["congressional_district_geoid"],
        errors="raise",
    )
    assert districts.gt(0).all()
    assert set(districts.astype(int)) == {601}

    release_tool._assert_cd_vintage_support_matches(
        pool_h5,
        {"sha256": pool_tool._STACKED_CD_CROSSWALK_SHA256},
    )
    preflight = release_tool._read_cd_vintage_support_provenance(pool_h5)
    assert preflight["household_congressional_district_geoid"] == {
        "exists": True,
        "table": "household",
        "column": "congressional_district_geoid",
        "rows": len(household),
        "positive_unique_count": 1,
    }


def test_constants_adapter_equals_live_constants_and_stays_out_of_identities(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    __import__("policyengine_us")
    args = pool_tool._parser().parse_args(
        [
            *_stacked_main_argv(tmp_path),
            "--config-authority",
            "constants_adapter",
        ]
    )
    configured_identity = pool_tool._configured_stacked_identity(args)
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.01,
        sample_seed=578,
    )
    base_identity = pool_tool._stacked_checkpoint_base_identity(
        _verified_inputs_fixture(pool_tool, tmp_path / "pins"),
        stack_receipt=stack.receipt,
        sample_fraction=0.01,
        sample_seed=578,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=579,
        policyengine_us_version="fixture-engine",
    )
    checkpoint_identity = pool_tool._pool_checkpoint_stage_identity(
        base_identity,
        "assembled",
    )
    equality_call: dict[str, object] = {}
    real_assert_equal = pool_tool.assert_legacy_payload_equal

    def capture_equality(expected: object, actual: object) -> None:
        equality_call["expected"] = expected
        equality_call["actual"] = actual
        real_assert_equal(expected, actual)

    monkeypatch.setattr(
        pool_tool,
        "assert_legacy_payload_equal",
        capture_equality,
    )

    # This call performs the real bundle load, compilation, and field-complete
    # equality assertion against the live generation-0 constructors.
    run_config = pool_tool._stacked_run_config(args)
    assert (
        set(equality_call["expected"])
        == set(equality_call["actual"])
        == {
            "battery_contract",
            "gap_fill_plan",
            "gap_fill_producer_schedule_receipt",
            "late_producer_schedule_receipt",
            "overlap_ownership",
            "publication_release",
            "source_manifest",
            "spine_assembly",
            "spine_sampling",
            "stacked_authority_receipt",
            "stacked_checkpoint_static_components",
            "support_spine",
            "take_up_contract",
            "take_up_contract_identity",
        }
    )
    expected_gate = equality_call["expected"]
    assert isinstance(expected_gate, dict)
    assert expected_gate["publication_release"] == {
        "legacy_prefixes": ["populace-us-2024"],
        "rungs": ["f001", "f004", "f010", "f025", "f100"],
        "legacy_compiled_regexes": [pool_tool._STACKED_RELEASE_ID_PATTERN.pattern],
    }
    assert expected_gate["spine_sampling"] == {
        "channels": ["asec", "acs"],
        "fraction": {
            "default": 1.0,
            "rungs": [
                {
                    "fraction": fraction,
                    "token": token,
                    "percent_basis_points": basis_points,
                }
                for fraction, token, basis_points in (
                    (0.01, "f001", 100),
                    (0.04, "f004", 400),
                    (0.10, "f010", 1_000),
                    (0.25, "f025", 2_500),
                    (1.00, "f100", 10_000),
                )
            ],
        },
        "seed": {"default": 578},
        "exact_count_rule": "floor(fraction * eligible)",
    }
    assert expected_gate["spine_assembly"] == {
        "mass_anchor_channel": "asec",
        "household_mass_shares": {"asec": 0.5, "acs": 0.5},
    }
    assert run_config == {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": {
            "attestation": "mirror-attested",
            "canonicalizer_version": 1,
            "country": "us",
            "schema_id": "country_spec",
            "schema_version": 1,
            "spec_sha256": "1c46dda9cff4e53e5a44c6502a46fa695d02d92731f23d618616a892071d4841",
        },
    }

    def contains_spec_binding(value: object) -> bool:
        if isinstance(value, Mapping):
            return "spec_binding" in value or any(
                contains_spec_binding(item) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_spec_binding(item) for item in value)
        return False

    assert not contains_spec_binding(stack.frame.metadata)
    assert not contains_spec_binding(configured_identity)
    assert not contains_spec_binding(base_identity)
    assert not contains_spec_binding(checkpoint_identity)


@pytest.mark.parametrize(
    ("mutation_path", "mutated_value", "expected_difference"),
    [
        (
            ("battery_contract", "gates", 1, "thresholds", "absolute"),
            0.5,
            "/battery_contract/gates/1/thresholds/absolute",
        ),
        (
            ("spine_assembly", "household_mass_shares", "asec"),
            0.49,
            "/spine_assembly/household_mass_shares/asec",
        ),
        (
            ("publication_release", "legacy_compiled_regexes", 0),
            "^mutated-release$",
            "/publication_release/legacy_compiled_regexes/0",
        ),
        (
            ("spine_sampling", "fraction", "default"),
            0.25,
            "/spine_sampling/fraction/default",
        ),
    ],
)
def test_constants_adapter_refuses_live_execution_surface_drift(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_path: tuple[str | int, ...],
    mutated_value: object,
    expected_difference: str,
) -> None:
    __import__("policyengine_us")
    args = pool_tool._parser().parse_args(
        [
            *_stacked_main_argv(tmp_path),
            "--config-authority",
            "constants_adapter",
        ]
    )
    real_compile = pool_tool.compile_to_legacy_payload

    def compile_with_drift(resolved: object) -> dict[str, object]:
        payload = copy.deepcopy(real_compile(resolved))
        target: object = payload
        for token in mutation_path[:-1]:
            if isinstance(token, int):
                assert isinstance(target, list)
                target = target[token]
            else:
                assert isinstance(target, dict)
                target = target[token]
        leaf = mutation_path[-1]
        if isinstance(leaf, int):
            assert isinstance(target, list)
            target[leaf] = mutated_value
        else:
            assert isinstance(target, dict)
            target[leaf] = mutated_value
        return payload

    monkeypatch.setattr(pool_tool, "compile_to_legacy_payload", compile_with_drift)

    with pytest.raises(LegacyPayloadMismatchError) as failure:
        pool_tool._stacked_run_config(args)
    assert expected_difference in {
        difference.path for difference in failure.value.differences
    }


def test_constants_adapter_post_resolution_failure_receipt_retains_binding(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="error",
    )
    binding = {
        "attestation": "mirror-attested",
        "canonicalizer_version": 1,
        "country": "us",
        "schema_id": "country_spec",
        "schema_version": 1,
        "spec_sha256": "f" * 64,
    }
    resolved_config = {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": binding,
    }
    monkeypatch.setattr(
        pool_tool,
        "_stacked_run_config",
        lambda _args: resolved_config,
    )

    with pytest.raises(RuntimeError, match="fixture stacked error"):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--config-authority",
                "constants_adapter",
            ]
        )

    error_path = next((tmp_path / "logbook-receipts").glob("*/error.json"))
    error_receipt = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_receipt["run_config"] == resolved_config


def test_constants_adapter_fixture_checkpoints_are_byte_identical_and_only_receipt_changes(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    fixed_binding = {
        "attestation": "mirror-attested",
        "canonicalizer_version": 1,
        "country": "us",
        "schema_id": "country_spec",
        "schema_version": 1,
        "spec_sha256": "5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e",
    }

    def run_fixture(root: Path, *, config_authority: str) -> dict[str, object]:
        root.mkdir()
        with monkeypatch.context() as patch:
            _install_stacked_entrypoint_stubs(
                pool_tool,
                patch,
                root,
                terminal="success",
            )
            patch.setattr(
                pool_tool,
                "_new_stacked_attempt_id",
                lambda **_kwargs: "populace-us-2024-stacked-attempt-fixture",
            )
            patch.setattr(
                pool_tool,
                "_new_stacked_release_id",
                lambda **_kwargs: (
                    "populace-us-2024-stacked-f001-s578-asec1-acs1-"
                    "20260817T000000Z-deadbeef"
                ),
            )
            patch.setattr(
                pool_tool,
                "_new_publication_run_id",
                lambda: "fixture-publication-run",
            )
            patch.setattr(
                pool_tool,
                "_stacked_run_config",
                lambda args: (
                    {"config_authority": "constants"}
                    if args.config_authority == "constants"
                    else {
                        "config_authority": "constants_adapter",
                        "spec_binding_status": "resolved",
                        "spec_binding": fixed_binding,
                    }
                ),
            )
            argv = _stacked_main_argv(root)
            if config_authority != "constants":
                argv.extend(["--config-authority", config_authority])
            assert pool_tool.main(argv) == 0

        checkpoint_files = sorted(
            (root / "stacked-pool.checkpoints").glob("stacked/*/*.checkpoint.h5")
        )
        assert [path.name for path in checkpoint_files] == [
            "assembled.checkpoint.h5",
            "simulated.checkpoint.h5",
            "transferred.checkpoint.h5",
        ]
        checkpoint_sha256 = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint_files
        }
        receipt_path = next((root / "logbook-receipts").glob("*/terminal-gates.json"))
        return {
            "checkpoint_sha256": checkpoint_sha256,
            "receipt": json.loads(receipt_path.read_text(encoding="utf-8")),
        }

    constants = run_fixture(tmp_path / "constants", config_authority="constants")
    adapter = run_fixture(
        tmp_path / "constants-adapter",
        config_authority="constants_adapter",
    )

    assert constants["checkpoint_sha256"] == adapter["checkpoint_sha256"]
    constants_receipt = dict(constants["receipt"])
    adapter_receipt = dict(adapter["receipt"])
    assert constants_receipt.pop("run_config") == {"config_authority": "constants"}
    assert adapter_receipt.pop("run_config") == {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": fixed_binding,
    }
    assert constants_receipt == adapter_receipt


def test_stacked_entrypoint_rejects_noncanonical_post_puf_transfer_receipt(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    noncanonical = _noncanonical_post_puf_authority_receipt()
    assert noncanonical["authority_form"] == "NON-CANONICAL"
    assert noncanonical["production_manifest_permitted"] is False
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
        post_puf_authority=noncanonical,
    )

    with pytest.raises(
        ValueError,
        match=(
            "stacked cold-build late-producer DAG: non-canonical stacked "
            "authority is forbidden"
        ),
    ):
        pool_tool.main(_stacked_main_argv(tmp_path))

    assert order == [
        "stack",
        "geography",
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
    ]
    assert not (tmp_path / "stacked-pool.h5").exists()
    assert not (tmp_path / "stacked-pool.manifest.json").exists()


def test_publication_error_keeps_gate_receipts_and_does_not_claim_stale_h5(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    __import__("policyengine_us")
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    stale_h5 = tmp_path / "stacked-pool.h5"
    stale_h5.write_bytes(b"prior-build-artifact")

    def fail_publication(*_args, **_kwargs) -> None:
        raise RuntimeError("fixture publication failure")

    monkeypatch.setattr(pool_tool, "_write_stacked_outputs", fail_publication)

    with pytest.raises(RuntimeError, match="fixture publication failure"):
        pool_tool.main(_stacked_main_argv(tmp_path))

    row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))
    assert row.artifact_location is None
    assert set(row.gate_verdicts) == {
        "fixture_completeness",
        "fixture_battery",
        "pipeline_error",
    }
    terminal_path = _receipt_file_from_reference(
        row.gate_verdicts["fixture_battery"]["receipt"]
    )
    assert terminal_path.is_file()
    assert stale_h5.read_bytes() == b"prior-build-artifact"


def test_logbook_gate_receipts_are_immutable_across_later_attempts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))
    first_receipt = _receipt_file_from_reference(
        first_row.gate_verdicts["fixture_battery"]["receipt"]
    )
    first_bytes = first_receipt.read_bytes()

    monkeypatch.setattr(
        pool_tool,
        "by_origin_battery",
        lambda _frame, *, tail_manifest: GateResult(
            name="fixture_battery",
            passed=False,
            failures=("later red verdict",),
        ),
    )
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))
        == 1
    )

    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    second_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )
    second_receipt = _receipt_file_from_reference(
        second_row.gate_verdicts["fixture_battery"]["receipt"]
    )
    assert second_receipt != first_receipt
    assert first_receipt.read_bytes() == first_bytes
    assert (
        json.loads(first_bytes)["terminal_gates"]["gates"]["fixture_battery"]["passed"]
        is True
    )
    assert (
        json.loads(second_receipt.read_bytes())["terminal_gates"]["gates"][
            "fixture_battery"
        ]["passed"]
        is False
    )


def test_stacked_checkpoint_identity_binds_both_scale_controls_and_manifest(
    pool_tool: ModuleType,
    tmp_path: Path,
) -> None:
    __import__("policyengine_us")
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    asec = _many_household_source_frame()
    acs = _many_household_source_frame(measured_offset=1_000.0)
    base_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity(
        *,
        stack=base_stack,
        sample_fraction: float = 0.10,
        sample_seed: int = 578,
        clone_fraction: float = 1.0,
        clone_seed: int = 578,
    ) -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            clone_attachment_fraction=clone_fraction,
            clone_attachment_seed=clone_seed,
            policyengine_us_version="fixture-engine",
        )

    fraction_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.01,
        sample_seed=578,
    )
    seed_stack = pool_tool.assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.10,
        sample_seed=579,
    )
    mutated_receipt = copy.deepcopy(dict(base_stack.receipt))
    mutated_receipt["survey_samples"]["acs"]["selected_household_ids_sha256"] = "f" * 64
    mutated_stack = SimpleNamespace(receipt=mutated_receipt)
    identities = {
        "base": identity(),
        "sample_fraction": identity(
            stack=fraction_stack,
            sample_fraction=0.01,
        ),
        "sample_seed": identity(stack=seed_stack, sample_seed=579),
        "clone_fraction": identity(clone_fraction=0.5),
        "clone_seed": identity(clone_seed=579),
        "stack_manifest": identity(stack=mutated_stack),
    }
    producer_schedule = identities["base"]["pool_code"]["gap_fill_producer_schedule"]
    assert producer_schedule["status"] == "all_producers_precede_activation"
    assert producer_schedule["direction_count"] == 2
    assert producer_schedule["target_count"] == 48
    digests = {
        name: pool_tool._pool_checkpoint_identity_sha256(value)
        for name, value in identities.items()
    }
    assert len(set(digests.values())) == len(digests)

    bank_outputs = pool_tool._output_paths(
        tmp_path / "identity-pool.h5",
        checkpoint_root=tmp_path / "identity-banks",
    )
    base_bank_outputs = pool_tool._with_checkpoint_identity(
        bank_outputs,
        base_identity_sha256=digests["base"],
    )
    for base_bank in (
        base_bank_outputs.primary_qrf_checkpoint_dir,
        base_bank_outputs.acs_transfer_checkpoint_dir,
    ):
        base_bank.mkdir(parents=True)
        (base_bank / "stale-marker").write_text("must remain unopened\n")
    for name, changed_digest in digests.items():
        if name == "base":
            continue
        changed_outputs = pool_tool._with_checkpoint_identity(
            bank_outputs,
            base_identity_sha256=changed_digest,
        )
        for selected, stale in (
            (
                changed_outputs.primary_qrf_checkpoint_dir,
                base_bank_outputs.primary_qrf_checkpoint_dir,
            ),
            (
                changed_outputs.acs_transfer_checkpoint_dir,
                base_bank_outputs.acs_transfer_checkpoint_dir,
            ),
        ):
            assert selected != stale
            assert selected.name == changed_digest
            routing = pool_tool._identity_routed_bank_open_receipt(
                selected,
                current_base_identity_sha256=changed_digest,
            )
            assert routing["selected_path"] == str(selected.resolve())
            assert routing["identity_mismatches"] == [
                {
                    "load_status": "identity_mismatch",
                    "stale_base_identity_sha256": digests["base"],
                    "current_base_identity_sha256": changed_digest,
                    "disposition": "bypassed",
                    "path": str(stale.resolve()),
                }
            ]
            assert (stale / "stale-marker").read_text() == "must remain unopened\n"

    checkpoint_root = tmp_path / "identity-checkpoints"
    original_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=identities["base"],
    )
    original_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    original_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=base_stack.frame,
            assembly_receipt=base_stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )
    for name, changed_identity in identities.items():
        if name == "base":
            continue
        changed_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=changed_identity,
        )
        assert changed_store.load_deepest() is None


def test_stacked_checkpoint_identity_binds_v13_semantic_contracts(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    __import__("policyengine_us")
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-engine",
    )
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity() -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )

    current = identity()
    pool_code = current["pool_code"]
    assert current["materializer_version"] == 13
    assert current["stacked_authority"]["version"] == 12
    assert current["geography_assignment"] == (
        pool_tool._stacked_geography_assignment_contract()
    )
    assert current["geography_assignment"]["seed"] == {
        "site": "legacy_puma_ladder",
        "stream": "geography_legacy",
        "value_source": "run_request.build_model_seed",
        "value": 0,
    }
    assert pool_code["operator_order"] == [
        "assemble_stacked_spine",
        "assign_us_puma_ladder",
        "prepare_multispine_source_inputs_for_clone",
        "gap_fill_stacked_spine",
        "run_stacked_late_producer_dag",
        "prepare_stacked_tail_derivation",
        "derive_multispine_pool_inputs",
        "seed_multispine_pool_inputs",
        "materialize_multispine_agreement_outputs",
        "stacked_completeness_gate",
        "by_origin_battery",
    ]
    assert pool_code["late_producer_schedule"] == pool_tool._json_ready(
        pool_tool.us_late_producer_schedule_receipt()
    )
    resource_semantics = pool_code["late_producer_resource_semantics"]
    unsigned_resource_semantics = dict(resource_semantics)
    resource_semantics_sha256 = unsigned_resource_semantics.pop("sha256")
    assert resource_semantics_sha256 == stacked_spine_module._canonical_sha256(
        unsigned_resource_semantics
    )
    assert resource_semantics["producer_count"] == 38
    resource_rows = {
        row["producer"]: row["resources"] for row in resource_semantics["producers"]
    }
    assert list(resource_rows) == list(
        stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
    )
    for (
        producer,
        contract,
    ) in stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY.items():
        assert set(resource_rows[producer]) == (
            stacked_spine_module._late_contract_available_input_keys(contract)
        )
    assert pool_code["primary_qrf_checkpoint_schema_version"] == 6
    assert pool_code["puf_capital_gains_tail_manifest_schema_version"] == 2
    assert pool_code["puf_capital_gains_tail_support_contract"] == (
        pool_tool.puf_capital_gains_tail_support_contract_identity()
    )
    assert pool_code["acs_pums_earnings_universe_contract"] == (
        pool_tool.acs_pums_earnings_universe_contract_identity()
    )
    assert pool_code["us_qbi_reconciliation_contract"] == (
        pool_tool.us_qbi_reconciliation_contract_identity()
    )
    assert pool_code["remaining_stage_input_manifest"] == (
        pool_tool.pool_remaining_stage_input_manifest_receipt()
    )

    with monkeypatch.context() as changed:
        changed.setattr(pool_tool, "PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION", 5)
        stale_qrf = identity()
    with monkeypatch.context() as changed:
        acs_contract = copy.deepcopy(
            pool_tool.acs_pums_earnings_universe_contract_identity()
        )
        acs_contract["minimum_age"] = 14
        acs_body = dict(acs_contract)
        acs_body.pop("sha256")
        acs_contract["sha256"] = hashlib.sha256(
            pool_tool._canonical_json_bytes(acs_body)
        ).hexdigest()
        changed.setattr(
            pool_tool,
            "acs_pums_earnings_universe_contract_identity",
            lambda: acs_contract,
        )
        stale_acs = identity()
    with monkeypatch.context() as changed:
        qbi_contract = copy.deepcopy(
            pool_tool.us_qbi_reconciliation_contract_identity()
        )
        qbi_contract["execution_scope"] = "recipient_subset"
        changed.setattr(
            pool_tool,
            "us_qbi_reconciliation_contract_identity",
            lambda: qbi_contract,
        )
        stale_qbi = identity()
    with monkeypatch.context() as changed:
        changed.setattr(pool_tool, "PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION", 1)
        stale_tail_schema = identity()
    with monkeypatch.context() as changed:
        remaining_manifest = copy.deepcopy(
            pool_tool.pool_remaining_stage_input_manifest_receipt()
        )
        remaining_manifest["manifest_sha256"] = "0" * 64
        changed.setattr(
            pool_tool,
            "pool_remaining_stage_input_manifest_receipt",
            lambda: remaining_manifest,
        )
        stale_remaining_manifest = identity()
    with monkeypatch.context() as changed:
        tail_contract = copy.deepcopy(
            pool_tool.puf_capital_gains_tail_support_contract_identity()
        )
        tail_contract["required_minimum"] = "one_recipient_per_status"
        changed.setattr(
            pool_tool,
            "puf_capital_gains_tail_support_contract_identity",
            lambda: tail_contract,
        )
        stale_tail_contract = identity()
    with monkeypatch.context() as changed:
        late_schedule = pool_tool._json_ready(
            pool_tool.us_late_producer_schedule_receipt()
        )
        late_schedule["payload_sha256"] = "0" * 64
        changed.setattr(
            pool_tool,
            "us_late_producer_schedule_receipt",
            lambda: late_schedule,
        )
        stale_late_schedule = identity()
    with monkeypatch.context() as changed:
        calibration_authority = copy.deepcopy(
            pool_tool.stacked_spine_authority_receipt()
        )
        calibration_authority["components"]["post_transfer_calibration"]["identity"][
            "scope"
        ]["reference"] = "forged_reference_scope"
        changed.setattr(
            pool_tool,
            "stacked_spine_authority_receipt",
            lambda: calibration_authority,
        )
        stale_post_transfer_calibration = identity()
    with monkeypatch.context() as changed:
        source_stage_binding = stacked_spine_module._late_source_stage_spec_binding

        def changed_source_stage_binding(
            operator: str,
            **kwargs: object,
        ) -> dict[str, object] | None:
            binding = source_stage_binding(operator, **kwargs)
            if operator != "with_us_adult_care_inputs" or binding is None:
                return binding
            mutated = copy.deepcopy(binding)
            mutated["asset_sha256"] = "0" * 64
            return mutated

        changed.setattr(
            stacked_spine_module,
            "_late_source_stage_spec_binding",
            changed_source_stage_binding,
        )
        stale_source_asset = identity()

    digests = {
        pool_tool._pool_checkpoint_identity_sha256(candidate)
        for candidate in (
            current,
            stale_qrf,
            stale_acs,
            stale_qbi,
            stale_tail_schema,
            stale_remaining_manifest,
            stale_tail_contract,
            stale_late_schedule,
            stale_post_transfer_calibration,
            stale_source_asset,
        )
    }
    assert len(digests) == 10

    # Positive control: discovery accepts the exact current semantic identity
    # under the same fixture engine version used to construct it.
    current_checkpoint_root = tmp_path / "current-semantic-checkpoints"
    current_store = pool_tool._PoolStageCheckpointStore(
        current_checkpoint_root,
        base_identity=current,
    )
    current_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    current_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            current_checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        == current
    )

    # A checkpoint produced by the current materializer with the prior QRF
    # schema is not merely identity-distinct: discovery must refuse it as stale.
    checkpoint_root = tmp_path / "mixed-qrf-version-checkpoints"
    stale_qrf_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=stale_qrf,
    )
    stale_qrf_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    stale_qrf_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )

    assert current["materializer_version"] == stale_qrf["materializer_version"] == 13
    assert stale_qrf["pool_code"]["primary_qrf_checkpoint_schema_version"] == 5
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out

    # Resource semantics are equally resume-fatal: a checkpoint whose source
    # asset/config binding differs must never be selected under current code.
    resource_checkpoint_root = tmp_path / "mixed-resource-checkpoints"
    stale_resource_store = pool_tool._PoolStageCheckpointStore(
        resource_checkpoint_root,
        base_identity=stale_source_asset,
    )
    stale_resource_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
    stale_resource_store.write(
        pool_tool.MultispinePoolCheckpoint(
            stage="assembled",
            frame=stack.frame,
            assembly_receipt=stack.frame.metadata[
                pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
            ],
            stage_receipts={},
        )
    )

    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            resource_checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out


def test_pool_envelope_v7_preserves_stacked_bank_identity_but_rejects_v6(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    __import__("policyengine_us")
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )

    def identity() -> dict[str, object]:
        return pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )

    current_identity = identity()
    current_digest = pool_tool._pool_checkpoint_identity_sha256(current_identity)
    checkpoint_root = tmp_path / "envelope-version-checkpoints"
    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
            6,
        )
        assert identity() == current_identity
        legacy_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=current_identity,
        )
        assert legacy_store.base_identity_sha256 == current_digest
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        legacy_store.write(
            pool_tool.MultispinePoolCheckpoint(
                stage="assembled",
                frame=stack.frame,
                assembly_receipt=stack.frame.metadata[
                    pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
                ],
                stage_receipts={},
            )
        )
    capsys.readouterr()

    assert pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION == 7
    assert identity() == current_identity
    current_store = pool_tool._PoolStageCheckpointStore(
        checkpoint_root,
        base_identity=current_identity,
    )
    assert current_store.base_identity_sha256 == current_digest
    assert current_store.load_deepest() is None
    assert "unsupported binding" in capsys.readouterr().out


@pytest.mark.parametrize("legacy_version", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
def test_legacy_stacked_materializer_checkpoint_is_not_discovered(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_version: int,
) -> None:
    __import__("policyengine_us")
    monkeypatch.setattr(
        pool_tool,
        "_policyengine_us_version",
        lambda: "fixture-engine",
    )
    verified = _verified_inputs_fixture(pool_tool, tmp_path / "pins")
    stack = pool_tool.assemble_stacked_spine(
        _many_household_source_frame(),
        _many_household_source_frame(measured_offset=1_000.0),
        sample_fraction=0.10,
        sample_seed=578,
    )
    checkpoint_root = tmp_path / "stacked-materializer-checkpoints"

    with monkeypatch.context() as legacy:
        legacy.setattr(
            pool_tool,
            "_STACKED_CHECKPOINT_MATERIALIZER_VERSION",
            legacy_version,
        )
        legacy_identity = pool_tool._stacked_checkpoint_base_identity(
            verified,
            stack_receipt=stack.receipt,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            policyengine_us_version="fixture-engine",
        )
        legacy_store = pool_tool._PoolStageCheckpointStore(
            checkpoint_root,
            base_identity=legacy_identity,
        )
        legacy_store.bind_input_receipts(_checkpoint_fixture_input_receipts())
        legacy_store.write(
            pool_tool.MultispinePoolCheckpoint(
                stage="assembled",
                frame=stack.frame,
                assembly_receipt=stack.frame.metadata[
                    pool_tool.SPINE_ASSEMBLY_MANIFEST_KEY
                ],
                stage_receipts={},
            )
        )

    assert pool_tool._STACKED_CHECKPOINT_MATERIALIZER_VERSION == 13
    assert (
        pool_tool._discover_stacked_checkpoint_identity(
            checkpoint_root,
            verified_inputs=verified,
            sample_fraction=0.10,
            sample_seed=578,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        is None
    )
    assert "checkpoint base identity is stale" in capsys.readouterr().out


def test_stacked_entrypoint_resumes_each_checkpoint_boundary(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    cold_order = list(order)
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))

    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))
        == 0
    )
    simulated_resume_order = list(order)
    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    second_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )

    checkpoint_root = next(
        (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
    )
    for suffix in (".h5", ".manifest.json"):
        (checkpoint_root / f"simulated.checkpoint{suffix}").unlink()
    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=second_row.row_digest))
        == 0
    )
    transferred_resume_order = list(order)
    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    third_row = next(
        row for row in rows if row.prev_row_digest == second_row.row_digest
    )

    for stage in ("transferred", "simulated"):
        for suffix in (".h5", ".manifest.json"):
            (checkpoint_root / f"{stage}.checkpoint{suffix}").unlink()
    order.clear()
    assert (
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=third_row.row_digest))
        == 0
    )
    assembled_resume_order = list(order)

    assert cold_order == [
        "stack",
        "geography",
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    assert simulated_resume_order == [
        "build_stacked_pool",
        "completeness",
        "battery",
        "publish",
    ]
    assert transferred_resume_order == [
        "build_stacked_pool",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    assert assembled_resume_order == [
        "build_stacked_pool",
        "prepare",
        "gap",
        "puf",
        "late_producer_dag",
        "tail_prepare",
        "derive",
        "seed",
        "simulate",
        "completeness",
        "battery",
        "publish",
    ]
    final_rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    assert len(final_rows) == 4
    assert any(row.prev_row_digest == third_row.row_digest for row in final_rows)


def test_stacked_resume_error_uses_realized_stack_identity(
    pool_tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables", exc_type=ModuleNotFoundError)
    _order, _full_puf_rows = _install_stacked_entrypoint_stubs(
        pool_tool,
        monkeypatch,
        tmp_path,
        terminal="success",
    )
    assert pool_tool.main(_stacked_main_argv(tmp_path)) == 0
    first_row = load_logbook_row(next((tmp_path / "logbook-spool").glob("*.json")))

    checkpoint_root = next(
        (tmp_path / "stacked-pool.checkpoints" / "stacked").iterdir()
    )
    for stage in ("transferred", "simulated"):
        for suffix in (".h5", ".manifest.json"):
            (checkpoint_root / f"{stage}.checkpoint{suffix}").unlink()
    monkeypatch.setattr(
        pool_tool,
        "_load_puf_donor",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("fixture resumed donor failure")
        ),
    )

    with pytest.raises(RuntimeError, match="fixture resumed donor failure"):
        pool_tool.main(_stacked_main_argv(tmp_path, predecessor=first_row.row_digest))

    rows = [
        load_logbook_row(path) for path in (tmp_path / "logbook-spool").glob("*.json")
    ]
    failed_row = next(
        row for row in rows if row.prev_row_digest == first_row.row_digest
    )
    assert "f001-s578-asec1-acs1" in failed_row.build_id
    assert failed_row.identity_digest == first_row.identity_digest
    assert "checkpoint_loaded" in failed_row.phases_reached
    assert "resume_donors_loaded" not in failed_row.phases_reached
    assert failed_row.gate_verdicts["pipeline_error"]["verdict"] == "error"
