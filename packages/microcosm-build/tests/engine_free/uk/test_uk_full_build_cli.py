"""The canonical CLI restores declared files and preserves failure/scope semantics."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_full_build_cli import *


def test_every_scope_and_size_control_keeps_default_all(tmp_path):
    for extra in (
        (),
        ("--target-geographies", "all"),
        ("--n-clones", "1"),
        ("--dataset-households", "10"),
        ("--n-clones", "1", "--dataset-households", "10"),
    ):
        assert arguments(tmp_path, *extra).target_geographies is None
    assert arguments(
        tmp_path, "--target-geographies", "country"
    ).target_geographies == ("country",)
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--target-geographies", "national")


def test_release_role_is_required_and_supplies_the_dense_posture(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(["--input-h5", "x.h5", "--out", str(tmp_path)])
    args = arguments(tmp_path)
    posture = cli.UK_ROWWISE_DENSE_POSTURE
    assert args._posture is posture
    assert (args.n_clones, args.seed, args.epochs, args.learning_rate) == (
        posture.clone_count,
        posture.seed,
        posture.epochs,
        posture.learning_rate,
    )
    assert (args.n_clones, args.epochs, args.learning_rate) == (15, 1500, 0.15)
    assert args.target_weight_rule == "grain_equal"
    assert args.expected_constituency_vintage == "2024_pcon"
    assert args.sample_seed == 578
    assert args.staging_upload_interval_seconds == 300.0
    assert args._explicit_arguments == frozenset()
    cli.validate_cli_args(args)
    explicit = arguments(tmp_path, "--epochs", "8", "--n-clones", "2")
    assert (explicit.epochs, explicit.n_clones) == (8, 2)
    assert explicit._explicit_arguments == frozenset({"epochs", "n_clones"})


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--target-loss-cap", "5"], "--target-loss-cap"),
        (["--allow-unpinned-feed"], "--allow-unpinned-feed"),
        (["--incumbent-h5", "incumbent.h5"], "--incumbent-h5"),
        (["--target-weight-rule", "family_equal"], "--target-weight-rule family_equal"),
    ],
)
def test_dense_role_refuses_the_national_knobs(tmp_path, extra, needle):
    args = arguments(tmp_path, *extra)
    with pytest.raises(ValueError, match="--release-role dense refuses") as excinfo:
        cli.validate_cli_args(args)
    assert needle in str(excinfo.value)


def test_dense_role_requires_the_ladder_and_the_pins(tmp_path):
    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
    ]
    with pytest.raises(ValueError, match="requires --ladder"):
        cli.validate_cli_args(cli.parse_args(argv))
    with pytest.raises(ValueError, match="--input-sha256 and --ladder-sha256"):
        cli.validate_cli_args(
            cli.parse_args([*argv, "--ladder", str(tmp_path / "ladder.npz")])
        )
    # A spine-request build has no input H5 to pin: only the ladder pin is due.
    request = [
        "--release-role",
        "dense",
        "--spine-request",
        str(tmp_path / "request.json"),
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        PIN,
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
    ]
    cli.validate_cli_args(cli.parse_args(request))


def test_national_role_is_validated_then_refused_until_phase_four(tmp_path, capsys):
    argv = [
        "--release-role",
        "national",
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--input-sha256",
        PIN,
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        PIN,
        "--ledger-manifest-sha256",
        PIN,
        "--no-staging",
    ]
    national = cli.parse_args(argv)
    assert national._posture is cli.uk_rowwise_posture("national")
    assert national.n_clones is None
    assert (national.seed, national.epochs, national.learning_rate) == (0, 1500, 0.02)
    cli.validate_cli_args(national)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert "next commit" in capsys.readouterr().err
    # The dense refusal table still applies before the national refusal.
    with pytest.raises(ValueError, match="--release-role national refuses"):
        cli.main([*argv, "--ladder", str(tmp_path / "ladder.npz")])


def test_candidate_clone_counts_are_dry_run_only(tmp_path):
    with pytest.raises(ValueError, match="only with --dry-run"):
        cli.main(
            [
                "--release-role",
                "dense",
                "--input-h5",
                str(tmp_path / "missing.h5"),
                "--input-sha256",
                PIN,
                "--ladder",
                str(tmp_path / "missing.npz"),
                "--ladder-sha256",
                PIN,
                "--ledger-facts",
                str(tmp_path / "ledger"),
                "--ledger-facts-sha256",
                PIN,
                "--ledger-manifest-sha256",
                PIN,
                "--out",
                str(tmp_path / "out"),
                "--candidate-clone-counts",
                "1,2,4",
                "--no-staging",
            ]
        )


def test_source_sampling_cannot_be_reapplied_as_pool_sampling():
    config = UKFullBuildConfig(calibration_year=2025, source_sample_fraction=0.1)
    assert config.sample_fraction == 1.0
    assert config.effective_sample_fraction == 0.1
    with pytest.raises(ValueError, match="second time"):
        replace(config, sample_fraction=0.1)


def test_households_only_binds_the_census_family_on_the_selection_node():
    config = UKFullBuildConfig(
        calibration_year=2025, target_families=("census_households/constituency",)
    )
    assert config.target_families == ("census_households/constituency",)
    with pytest.raises(ValueError, match="target families"):
        UKFullBuildConfig(calibration_year=2025, target_families=())


def test_cli_cold_and_required_replay_recreate_dataset_and_sidecars(
    tmp_path, monkeypatch
):
    pytest.importorskip("tables")
    args = arguments(tmp_path)
    first = prepared(tmp_path)
    assert cli.execute_full_build(first, args) == 0
    out = args.out
    expected = json.loads((out / "build.json").read_text())
    assert expected["readback_passed"] is True
    assert expected["release_authorized"] is False
    assert expected["release_role"] == "dense"
    assert (out / f"{STEM}.targets.csv").read_text().startswith("name,target_name")
    for path in out.iterdir():
        if path.is_file():
            path.unlink()

    def forbidden(*args):
        raise AssertionError(
            "Required replay repeated a completed numerical/evidence node"
        )

    for kernel in (Fixture, Evidence, Holdout, Targets, Population):
        monkeypatch.setattr(kernel, "run", forbidden)
    args.resume = "require"
    assert cli.execute_full_build(prepared(tmp_path), args) == 0
    actual = json.loads((out / "build.json").read_text())
    assert actual["content_sha256"] == expected["content_sha256"]
    # The output names come from the dense posture and the FRS vintage.
    assert (out / f"{STEM}.h5").is_file()
    assert (out / f"{STEM}.holdout.json").is_file()
    assert (out / f"{STEM}.local_gates.json").is_file()
    assert not (out / "microcosm_uk_2025.h5").exists()


def test_dense_run_projects_the_rowwise_candidate_manifest(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path, "--release-candidate")
    assert cli.execute_full_build(prepared(tmp_path), args) == 0
    out = args.out
    manifest = json.loads((out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["build_kind"] == "uk_rowwise_calibrated_candidate"
    assert manifest["release_role"] == "dense"
    assert manifest["release_id"] == UK_DENSE_RELEASE_ID
    assert manifest["parameters"]["release_role"] == "dense"
    assert manifest["parameters"]["release_candidate"] is True
    assert (
        manifest["parameters"]["doctrine"]
        == cli.UK_ROWWISE_DENSE_POSTURE.doctrine_bounds()
    )
    assert (manifest["parameters"]["n_clones"], manifest["parameters"]["epochs"]) == (
        15,
        1500,
    )
    for key in (
        "solve",
        "fit",
        "weights",
        "cross_grain",
        "census_household_uprating",
        "outputs",
        "releasable",
        "identity",
        "bound_target_families",
        "binding_adjudications",
        "measure_exclusions",
        "blocked_at_f100",
        "blocking_failures",
        "diagnostic_failures",
        "release_gate_failures_not_enforced",
        "failing_gate_ids",
        "sampling",
        "rung_surface",
        "support",
        "geography",
        "vintages",
        "graph",
    ):
        assert key in manifest, key
    assert set(manifest["identity"]) >= {
        "code",
        "ladder",
        "runtime",
        "spine",
        "targets",
    }
    assert manifest["identity"]["spine"]["pin_verified"] is True
    assert manifest["identity"]["ladder"]["pin_verified"] is True
    assert manifest["identity"]["targets"]["paired_ladder_sha256"] == PIN
    assert manifest["solve"]["n_targets_by_kind"] == {
        "local": 0,
        "ladder": 1,
        "national": 0,
    }
    assert manifest["solve"]["measure_resolution"] == {"blocks": 1}
    assert manifest["solve"]["target_weight_rule_override"] == {}
    assert manifest["solve"]["n_households"] == 2
    assert manifest["solve"]["pool_households"] == 2
    assert manifest["fit"]["rotated_holdout"]["n_folds"] == 5
    assert manifest["fit"]["local_by_family"][0]["family"] == "census_households"
    assert manifest["weights"]["realized_max_weight_ratio_vs_design"] == 1.0
    assert manifest["weights"]["household_weight_kind"] == "calibrated"
    assert manifest["census_household_uprating"]["applied"] is True
    assert manifest["releasable"] is True and manifest["blocked_at_f100"] is False
    assert manifest["release_posture"]["release_blocking_gates_passed"] is True
    dataset = manifest["outputs"]["dataset"]
    assert Path(dataset["path"]) == out / f"{STEM}.h5"
    assert (
        dataset["sha256"]
        == hashlib.sha256((out / f"{STEM}.h5").read_bytes()).hexdigest()
    )
    assert Path(manifest["outputs"]["local_gate_report"]["path"]).name == (
        f"{STEM}.local_gates.json"
    )
    assert manifest["graph"]["epoch_rows"] == "dense_solve_only"
    assert "uk.full.problem" in manifest["graph"]["artifacts"]
    # The staged bundle is every registered output plus the manifest; each
    # registered file is on disk beside it with its recorded digest.
    for entry in manifest["outputs"].values():
        path = Path(entry["path"])
        assert path.parent == out and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_blocked_gate_projects_an_unreleasable_manifest(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path)
    build = prepared(tmp_path, "uk_local_target_fit")
    assert cli.execute_full_build(build, args) == 1
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["releasable"] is False
    assert manifest["blocked_at_f100"] is True
    assert manifest["blocking_failures"] == ["[uk_local_target_fit] synthetic failure"]
    assert manifest["failing_gate_ids"] == ["uk_local_target_fit"]


@pytest.mark.parametrize(
    "failure,exported",
    [
        ("uk_local_geography_ladder_post_calibration", False),
        ("uk_local_target_fit", True),
    ],
)
def test_cli_retains_failed_evidence_and_correct_status(tmp_path, failure, exported):
    if exported:
        pytest.importorskip("tables")
    args = arguments(tmp_path)
    build = prepared(tmp_path, failure)
    assert cli.execute_full_build(build, args) == 1
    assert (args.out / "uk.full.gates.calibrated.gate_report.json").is_file()
    assert (args.out / f"{STEM}.h5").exists() == exported


def test_dry_run_has_no_files_or_kernel_execution(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, "--dry-run")
    build = prepared(tmp_path)
    monkeypatch.setattr(
        cli, "run_graph", lambda *a, **k: pytest.fail("dry run executed graph")
    )
    assert cli.execute_full_build(build, args) == 0
    assert json.loads(capsys.readouterr().out)["default_scope"] == "all_geographies"
    assert not args.out.exists()


def test_rejected_output_inside_source_never_writes_failure_sidecar(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    build = replace(build, sources={"fixture": tmp_path})
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 1
    assert not args.out.exists()


def test_main_runs_the_logbook_envelope_around_a_dense_build(tmp_path, monkeypatch):
    pytest.importorskip("tables")
    from microcosm.build.logbook import load_spool_rows

    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    args = arguments(tmp_path, "--seed", "7")
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 0
    rows = load_spool_rows(args.out / "logbook-spool")
    assert len(rows) == 1
    row = rows[0]
    assert row.pipeline == "uk-local-candidate"
    assert row.build_id.startswith("uk-local-candidate-f100-s7-")
    assert row.rung == "f100" and row.seed == 7
    assert row.disposition == "iterating"
    assert row.artifact_location.endswith(f"{STEM}.h5")
    assert "published" in row.phases_reached
    assert row.gate_verdicts and all(
        item["verdict"] == "passed" and ".local_gates.json#/gates/" in item["receipt"]
        for item in row.gate_verdicts.values()
    )
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["staging_delivery"]["enabled"] is False
    assert manifest["staged_dataset"]["status"] == "skipped"


def test_main_records_a_failed_attempt_when_the_build_raises(tmp_path, monkeypatch):
    from microcosm.build.logbook import load_spool_rows

    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    monkeypatch.setattr(
        cli,
        "run_graph",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic refusal")),
    )
    assert cli.main([]) == 1
    failure = json.loads((args.out / "failure.json").read_text())
    assert failure["message"] == "synthetic refusal"
    rows = load_spool_rows(args.out / "logbook-spool")
    assert len(rows) == 1 and rows[0].disposition == "failed"
    receipts = list((args.out / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["error_type"].endswith("RuntimeError")


@pytest.mark.parametrize("resume", ["auto", "require"])
def test_source_phase_evidence_survives_later_exception(tmp_path, monkeypatch, resume):
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    assembled = Node(
        "spine.gates.assembled",
        Evidence.ref,
        population="uk.full.calibrated",
        params={"phase": "preflight", "failed": None},
        artifact_outputs=(ArtifactOutput("gate_report", FULL_GATE_REPORT_TYPE),),
    )
    transferred = replace(assembled, id="spine.gates.transferred")
    graph = replace(
        build.full.graph, nodes=(*build.full.graph.nodes, assembled, transferred)
    )
    build = replace(build, full=replace(build.full, graph=graph))
    args.out.mkdir()
    previous = b'{"kind":"previous-complete-build"}'
    (args.out / "build.json").write_bytes(previous)
    args.resume = resume
    original_run = cli.run_graph
    if resume == "require":
        original_run(
            cli.compile_graph(cli._through(graph, assembled.id)),
            sources=build.sources,
            store=cli.ContentStore(args.out / ".graph-store"),
            kernels=build.kernels,
        )
        monkeypatch.setattr(
            Fixture, "run", lambda *a: pytest.fail("Checkpoint replay reran the source")
        )
        monkeypatch.setattr(
            Evidence, "run", lambda *a: pytest.fail("Checkpoint replay reran the gate")
        )

    def refuse_transferred(compiled, **kwargs):
        if transferred.id in {node.id for node in compiled.graph.nodes}:
            raise RuntimeError("downstream source donor bin is empty")
        return original_run(compiled, **kwargs)

    monkeypatch.setattr(cli, "run_graph", refuse_transferred)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 1
    assert (args.out / "build.json").read_bytes() == previous
    failure = json.loads((args.out / "failure.json").read_bytes())
    durable = Path(failure["evidence_directory"])
    assert (durable / "spine.gates.assembled.graph.json").is_file()
    assert not (durable / "spine.gates.transferred.graph.json").exists()
    gate_bytes = (durable / "spine.gates.assembled.gate_report.json").read_bytes()
    assert gate_bytes == gate_payload("preflight")
    index = json.loads((durable / "evidence-index.json").read_bytes())
    assert (
        index["artifacts"]["spine.gates.assembled/gate_report"]["sha256"]
        == hashlib.sha256(gate_bytes).hexdigest()
    )
