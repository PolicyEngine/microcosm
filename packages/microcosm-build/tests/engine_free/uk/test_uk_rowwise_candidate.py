"""The UK rowwise command surface on the graph full-build driver (microcosm#823).

The dense role's synthetic end-to-end contract is pinned by
``test_uk_full_build_cli.py``; here are the release-role tables, the pure-CLI
refusals, the shared command helpers and the staged-dataset pre-flight, as
the retired rowwise tool pinned them, plus the tool's stub.
"""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_rowwise_candidate import *


def test_rowwise_candidate_tool_is_a_stub_over_the_graph_driver() -> None:
    """Both release roles are served by the graph driver's ``main``."""

    spec = importlib.util.spec_from_file_location(
        "build_uk_rowwise_candidate",
        _TEST_PATHS.repository / "tools" / "build_uk_rowwise_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main is _load_builder_module().main


def test_graph_driver_dry_run_prints_the_operation_inventory(
    monkeypatch, capsys, tmp_path
) -> None:
    """The graph driver's dry run plans the same request without solving.

    The tool's dry run (above) prints the fenced clone/matrix plan it computes
    in process; the graph driver prints the compiled operation inventory of
    the same request. Both refuse to write. A bound spine checkpoint stands
    in for the tool's sidecar-free households-only path, and the Ledger pins
    are the committed feed's because the graph refuses any other.
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    from microcosm.build.uk_runtime import full_build_cli as cli
    from microcosm.build.uk_runtime import spine_build
    from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
    from test_support.microcosm_build.uk_calibration_run import _bound_checkpoint

    input_h5 = tmp_path / "spine.h5"
    ladder_path = tmp_path / "ladder.npz"
    output_dir = tmp_path / "dry-run-output"
    _write_staging_h5(input_h5)
    _write_ladder(ladder_path)
    frame, _ = load_uk_national_frame(input_h5)
    sidecar_path, gates_path, sidecar = _bound_checkpoint(tmp_path, frame)
    sidecar["stages"] = ["frs_spine"]
    sidecar["sampling"] = {"fraction": 1.0, "seed": 578}
    sidecar_path.write_text(json.dumps(sidecar))
    monkeypatch.setattr(spine_build, "_rules_engine", lambda: object())
    monkeypatch.setattr(
        spine_build, "_rules_engine_provenance", lambda: {"version": "fixture"}
    )
    monkeypatch.setattr(
        cli, "run_graph", lambda *a, **k: pytest.fail("dry run executed graph")
    )
    feed = load_uk_chronicle_feed()
    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(input_h5),
        "--input-sidecar",
        str(sidecar_path),
        "--input-spine-gates",
        str(gates_path),
        "--input-sha256",
        hashlib.sha256(input_h5.read_bytes()).hexdigest(),
        "--ladder",
        str(ladder_path),
        "--ladder-sha256",
        hashlib.sha256(ladder_path.read_bytes()).hexdigest(),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        feed.facts_sha256,
        "--ledger-manifest-sha256",
        feed.manifest_sha256,
        "--out",
        str(output_dir),
        "--n-clones",
        "2",
        "--seed",
        "7",
        "--dry-run",
        "--no-staging",
    ]
    assert cli.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["default_scope"] == "all_geographies"
    assert plan["configuration"]["n_clones"] == 2
    assert plan["configuration"]["seed"] == 7
    assert plan["configuration"]["calibration"]["epochs"] == 1500
    assert plan["configuration"]["target_families"] is None
    assert {node["id"] for node in plan["nodes"]} >= {
        "uk.full.dense",
        "uk.full.target_selection",
        "uk.full.gates.calibrated",
    }
    assert not output_dir.exists()
    assert not (output_dir / "logbook-spool").exists()
    # --candidate-clone-counts plans one inventory per requested K.
    assert cli.main([*argv, "--candidate-clone-counts", "1,2"]) == 0
    inventories = json.loads(capsys.readouterr().out)
    assert [item["n_clones"] for item in inventories] == [1, 2]
    assert [item["configuration"]["n_clones"] for item in inventories] == [1, 2]
    # --households-only narrows the selection node to the census family.
    assert cli.main([*argv, "--households-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["configuration"]["target_families"] == [
        "census_households/constituency"
    ]
    assert not output_dir.exists()


def test_candidate_clone_count_planning_is_dry_run_only(tmp_path) -> None:
    builder = _load_builder_module()
    with pytest.raises(ValueError, match="only with --dry-run"):
        builder.main(
            [
                "--input-h5",
                str(tmp_path / "missing.h5"),
                "--release-role",
                "dense",
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
                "--input-sha256",
                "0" * 64,
                "--ladder-sha256",
                "0" * 64,
                "--ledger-facts",
                str(tmp_path / "ledger"),
                "--ledger-facts-sha256",
                "0" * 64,
                "--ledger-manifest-sha256",
                "0" * 64,
            ]
        )


def test_candidate_publication_rolls_back_on_interrupt(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "candidate"
    staging_dir.mkdir()
    output_paths = rowwise_cli.output_paths(
        output_dir,
        posture=builder.UK_ROWWISE_DENSE_POSTURE,
        vintage="2024_25",
    )
    staged = {key: staging_dir / path.name for key, path in output_paths.items()}
    for path in staged.values():
        path.write_text("complete staged artifact\n")

    original_replace = Path.replace

    def interrupt_support(self, target):
        if Path(target) == output_paths["support"]:
            raise KeyboardInterrupt
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_support)
    with pytest.raises(KeyboardInterrupt):
        rowwise_staging.publish_staged_files(staged, output_paths)

    assert not output_dir.exists()


def test_release_verdict_requires_single_block_engine() -> None:
    builder = rowwise_cli
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=1, release_blocking_gates_passed=True
    )
    assert releasable is True and all(posture.values())
    # A per-block engine resolution never writes a releasable artifact, even
    # with every release-blocking gate passed on the full rung (#736 erratum).
    releasable, posture = builder._release_verdict(
        sample_fraction=1.0, engine_blocks=15, release_blocking_gates_passed=True
    )
    assert releasable is False
    assert posture == {
        "full_rung": True,
        "single_block_engine": False,
        "release_blocking_gates_passed": True,
    }
    assert (
        builder._release_verdict(
            sample_fraction=0.1, engine_blocks=1, release_blocking_gates_passed=True
        )[0]
        is False
    )


def test_gate_criticality_reads_fail_closed() -> None:
    builder = rowwise_cli
    assert builder._is_release_blocking({"criticality": "release_blocking"}) is True
    assert builder._is_release_blocking({"criticality": "diagnostic"}) is False
    # Missing or unknown criticality vetoes: schema drift on one entry cannot
    # drop a failed gate out of both the blocking list and all_gates_passed.
    assert builder._is_release_blocking({}) is True
    assert builder._is_release_blocking({"criticality": "advisory"}) is True
    blocking, diagnostic = builder._gate_failures_by_criticality(
        {
            "gates": {
                "uk_local_area_support": {
                    "status": "failed",
                    "failures": ["ESS 42.3 < 50"],
                },
                "uk_local_weight_ratio": {
                    "status": "failed",
                    "criticality": "diagnostic",
                    "failures": ["ratio 578 > 100"],
                },
                "uk_local_target_fit": {
                    "status": "passed",
                    "criticality": "diagnostic",
                },
            }
        }
    )
    assert blocking == ["[uk_local_area_support] ESS 42.3 < 50"]
    assert diagnostic == ["[uk_local_weight_ratio] ratio 578 > 100"]


def test_release_candidate_refuses_non_doctrine_solve_settings(tmp_path) -> None:
    builder = _load_builder_module()
    pin = "0" * 64
    base = [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--release-role",
        "dense",
        "--input-sha256",
        pin,
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        pin,
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        pin,
        "--ledger-manifest-sha256",
        pin,
        "--out",
        str(tmp_path / "out"),
        "--release-candidate",
    ]
    # The doctrine defaults are the release posture: nothing to refuse.
    args = builder._parse_args(base)
    builder._validate_cli_args(args)
    assert args.n_clones == builder.UK_ROWWISE_DENSE_POSTURE.clone_count == 15
    assert args.epochs == builder.UK_ROWWISE_DENSE_POSTURE.epochs == 1500
    assert args.target_weight_rule == "grain_equal"

    with pytest.raises(ValueError, match=r"--epochs != doctrine 1500"):
        builder._validate_cli_args(builder._parse_args([*base, "--epochs", "512"]))
    with pytest.raises(ValueError, match=r"--n-clones != doctrine 15"):
        builder._validate_cli_args(builder._parse_args([*base, "--n-clones", "10"]))
    with pytest.raises(ValueError, match=r"--target-weight-rule"):
        builder._validate_cli_args(
            builder._parse_args([*base, "--target-weight-rule", "uniform"])
        )


def test_candidate_requires_pinned_ledger_inputs(tmp_path) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--input-sha256",
            "0" * 64,
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--ladder-sha256",
            "0" * 64,
            "--out",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="mandatory"):
        builder._validate_cli_args(args)


def test_selection_seed_requires_a_dataset_size(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            "--selection-seed",
            "11",
        ]
    )
    with pytest.raises(ValueError, match="requires --dataset-households"):
        builder._validate_cli_args(args)


@pytest.mark.parametrize(
    ("argv_tail", "message"),
    [
        (["--selection-pi-hi", "0.95"], "requires --dataset-households"),
        (["--dataset-households", "10", "--selection-pi-hi", "0"], r"in \(0, 1\]"),
        (["--dataset-households", "10", "--selection-pi-hi", "1.5"], r"in \(0, 1\]"),
        (["--baseline-pi-floor", "0.01"], "requires --dataset-households"),
        (["--dataset-households", "10", "--baseline-pi-floor", "-0.1"], r"in \[0, 1\]"),
        (["--dataset-households", "10", "--baseline-pi-floor", "1.5"], r"in \[0, 1\]"),
    ],
)
def test_selection_pi_hi_is_candidate_only_and_bounded(tmp_path, argv_tail, message):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            *argv_tail,
        ]
    )
    with pytest.raises(ValueError, match=message):
        builder._validate_cli_args(args)


def test_dense_candidate_manifest_has_no_size_sidecars(tmp_path):
    builder = rowwise_cli
    paths = builder._output_paths(
        tmp_path, posture=builder.UK_ROWWISE_DENSE_POSTURE, vintage="2024_25"
    )
    assert paths["dataset"].name == "microcosm_uk_2024_25_local.h5"
    assert paths["dense_reference"].name == builder.DENSE_REFERENCE_DIAGNOSTICS_FILENAME
    assert paths["selection"].name == builder.DATASET_SIZE_SELECTION_FILENAME
    assert builder._SIZE_RUN_ONLY_OUTPUTS == {"dense_reference", "selection"}


def test_size_cli_refuses_promotion_without_separate_certification(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(
        [
            "--input-h5",
            str(tmp_path / "spine.h5"),
            "--release-role",
            "dense",
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--out",
            str(tmp_path / "out"),
            "--dataset-households",
            "50000",
            "--release-candidate",
        ]
    )
    with pytest.raises(ValueError, match="candidate-only"):
        builder._validate_cli_args(args)


def test_staging_epoch_stride_keeps_the_forwarded_rows_bounded():
    builder = rowwise_staging

    def stride(epochs, households):
        return builder._staging_epoch_every(
            SimpleNamespace(epochs=epochs, dataset_households=households)
        )

    assert stride(2, None) == 10 and stride(2, 300) == 10
    assert stride(2000, None) == 10
    # dense + ten probes + refit at 2,000 epochs: 24,000 epochs -> 2,400 rows
    assert stride(2000, 55000) == 10
    assert stride(10000, 55000) == 50
    assert stride(100000, None) == 42
    for epochs, households in ((2000, 55000), (10000, 55000), (100000, None)):
        solves = 1 if households is None else 2 + rowwise_cli.BUDGET_ITERS
        assert (
            epochs * solves / stride(epochs, households)
            <= builder._STAGING_MAX_EPOCH_ROWS
        )


def test_remote_dataset_staging_is_refused_up_front_without_credential_or_repo(
    monkeypatch, tmp_path, capsys
):
    builder = _load_builder_module()
    input_h5, ladder_path, flags = _staging_run_setup(
        builder, monkeypatch, tmp_path, remote=True
    )
    out = tmp_path / "refused"
    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: None)
    with pytest.raises(ValueError, match="write credential"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()

    class Unreachable:
        def repo_info(self, **kwargs):
            raise RuntimeError("503 token=do-not-record")

    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: "hf_test_token")
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: Unreachable())
    with pytest.raises(ValueError, match="cannot reach") as info:
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert "do-not-record" not in str(info.value)
    assert not out.exists()

    # A read token sees the private repository but cannot upload: refused
    # before the spine is read, not after the solve (the Hub answers 403).
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: _FakeHub(role="read"))
    with pytest.raises(ValueError, match="read-only"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()

    # A fine-grained token scoped to another owner is refused the same way;
    # one scoped to the repository's organisation passes the pre-flight.
    user_scoped = [
        {"entity": {"type": "user", "name": "someone"}, "permissions": ["repo.write"]}
    ]
    monkeypatch.setattr(
        rowwise_staging,
        "_hub_api",
        lambda: _FakeHub(role="fineGrained", scopes=user_scoped),
    )
    with pytest.raises(ValueError, match="repo.write"):
        builder.main(_build_args(input_h5, ladder_path, flags, out))
    assert not out.exists()
    org_scoped = [
        {
            "entity": {"type": "org", "name": "policyengine"},
            "permissions": ["repo.write"],
        }
    ]
    org_hub = _FakeHub(role="fineGrained", scopes=org_scoped)
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: org_hub)
    status = builder.main(
        _build_args(input_h5, ladder_path, flags, tmp_path / "org-scoped")
    )
    # The pre-flight admits the org-scoped token; the synthetic spine
    # carries no bound checkpoint sidecar, so the graph driver refuses
    # later, on its inputs, never on the credential.
    assert status == 1
    assert "sidecar absent" in capsys.readouterr().err

    # Argument refusals cost nothing and come first: a missing credential is
    # never the reported reason when the arguments are wrong.
    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: None)
    with pytest.raises(ValueError, match="only with --dry-run"):
        builder.main(
            _build_args(
                input_h5, ladder_path, flags, out, "--candidate-clone-counts", "2,3"
            )
        )
    assert not out.exists()


def test_release_role_is_required(tmp_path) -> None:
    builder = _load_builder_module()
    argv = _dense_argv(tmp_path)
    argv.remove("--release-role")
    argv.remove("dense")
    with pytest.raises(SystemExit):
        builder._parse_args(argv)
    with pytest.raises(SystemExit):
        builder._parse_args([*argv, "--release-role", "local"])


def test_release_role_supplies_the_solve_defaults(tmp_path) -> None:
    builder = _load_builder_module()
    dense = builder._parse_args(_dense_argv(tmp_path))
    posture = builder.UK_ROWWISE_DENSE_POSTURE
    assert (dense.n_clones, dense.seed, dense.epochs, dense.learning_rate) == (
        posture.clone_count,
        posture.seed,
        posture.epochs,
        posture.learning_rate,
    )
    assert dense.target_weight_rule == "grain_equal"
    assert dense.expected_constituency_vintage == "2024_pcon"
    assert dense.staging_upload_interval_seconds == 300.0
    assert dense._explicit_arguments == frozenset()
    builder._validate_cli_args(dense)

    national = builder._parse_args(_role_argv(tmp_path, "national"))
    posture = builder.uk_rowwise_posture("national")
    assert national._posture is posture
    assert national.n_clones is None
    assert (national.seed, national.epochs, national.learning_rate) == (0, 1500, 0.02)
    assert national.target_weight_rule == "family_equal"
    assert national.expected_constituency_vintage is None
    builder._validate_cli_args(national)
    explicit = builder._parse_args(_role_argv(tmp_path, "national", "--epochs", "5"))
    assert explicit.epochs == 5
    assert explicit._explicit_arguments == frozenset({"epochs"})
    # The doctrine's own seed may be spelled out; only another seed is refused.
    builder._validate_cli_args(
        builder._parse_args(_role_argv(tmp_path, "national", "--seed", "0"))
    )


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--target-loss-cap", "5"], "--target-loss-cap"),
        (["--allow-unpinned-feed"], "--allow-unpinned-feed"),
        (["--target-weight-rule", "family_equal"], "--target-weight-rule family_equal"),
    ],
)
def test_dense_role_refusal_table(tmp_path, extra, needle) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(_dense_argv(tmp_path, *extra))
    with pytest.raises(ValueError, match="--release-role dense refuses") as excinfo:
        builder._validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_dense_role_requires_the_ladder(tmp_path) -> None:
    builder = _load_builder_module()
    argv = _role_argv(tmp_path, "dense", "--ladder-sha256", "3" * 64)
    with pytest.raises(ValueError, match="requires --ladder"):
        builder._validate_cli_args(builder._parse_args(argv))


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--ladder", "ladder.npz"], "--ladder"),
        (["--ladder-sha256", "3" * 64], "--ladder-sha256"),
        (["--expected-constituency-vintage", "2024_pcon"], "--expected-constituency"),
        (["--source-year", "2023"], "--source-year"),
        (["--source-lineage-modulus", "7"], "--source-lineage-modulus"),
        (["--n-clones", "15"], "--n-clones"),
        (["--candidate-clone-counts", "2,4", "--dry-run"], "--candidate-clone-counts"),
        (["--engine-blocks", "2"], "--engine-blocks"),
        (["--households-only"], "--households-only"),
        (["--skip-holdout"], "--skip-holdout"),
        (["--dataset-households", "10"], "--dataset-households"),
        (["--selection-seed", "3"], "--selection-seed"),
        (["--selection-pi-hi", "0.5"], "--selection-pi-hi"),
        (["--baseline-pi-floor", "0.1"], "--baseline-pi-floor"),
        (["--no-size-checkpoint"], "--no-size-checkpoint"),
        (["--resume-size-checkpoint", "dir"], "--resume-size-checkpoint"),
        (["--sample-fraction", "0.1"], "--sample-fraction"),
        (["--sample-seed", "9"], "--sample-seed"),
        (["--seed", "7"], "--seed != doctrine 0"),
        (["--target-weight-rule", "grain_equal"], "--target-weight-rule grain_equal"),
    ],
)
def test_national_role_refusal_table(tmp_path, extra, needle) -> None:
    builder = _load_builder_module()
    args = builder._parse_args(_role_argv(tmp_path, "national", *extra))
    with pytest.raises(ValueError, match="--release-role national refuses") as excinfo:
        builder._validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_national_role_refuses_release_candidate_with_the_seam_reason(tmp_path):
    builder = _load_builder_module()
    args = builder._parse_args(_role_argv(tmp_path, "national", "--release-candidate"))
    with pytest.raises(ValueError, match="cannot sign shippability"):
        builder._validate_cli_args(args)
