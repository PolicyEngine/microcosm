"""The canonical CLI restores declared files and preserves failure/scope semantics."""

# ruff: noqa: F403, F405
from types import SimpleNamespace

import test_support.microcosm_build.uk_full_build_cli as support  # noqa: E402
from test_support.microcosm_build.uk_full_build_cli import *


def test_geography_assignment_arguments_are_closed(tmp_path):
    """Atomic is the default; the cross-flag rules live in the validator."""
    assert arguments(tmp_path).geography_assignment == "atomic"
    cli.validate_cli_args(arguments(tmp_path))
    legacy = arguments(tmp_path, "--geography-assignment", "legacy", supports=())
    assert legacy.geography_assignment == "legacy"
    cli.validate_cli_args(legacy)
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--geography-assignment", "keyed", supports=())
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--atomic-support-sha256-ew", "zz")
    with pytest.raises(ValueError, match="requires the three atomic-area supports"):
        cli.validate_cli_args(arguments(tmp_path, supports=()))
    with pytest.raises(ValueError, match="--atomic-support-ni"):
        cli.validate_cli_args(arguments(tmp_path, supports=SUPPORT_ARGUMENTS[:4]))
    with pytest.raises(ValueError, match="legacy takes no atomic-area supports"):
        cli.validate_cli_args(arguments(tmp_path, "--geography-assignment", "legacy"))
    with pytest.raises(ValueError, match="--geography-assignment legacy"):
        cli.validate_cli_args(
            arguments(
                tmp_path,
                "--release-candidate",
                "--geography-assignment",
                "legacy",
                supports=(),
                release_pins=(),
            )
        )
    with pytest.raises(ValueError, match="--atomic-support-sha256-ew"):
        cli.validate_cli_args(
            arguments(tmp_path, "--release-candidate", release_pins=())
        )
    release = arguments(tmp_path, "--release-candidate")
    assert release.release_candidate and release.geography_assignment == "atomic"
    cli.validate_cli_args(release)
    # The national role builds on the bound checkpoint with no geography
    # assignment: the supports are refused by name with the other dense-only
    # flags, and its default needs none.
    cli.validate_cli_args(cli.parse_args(_national_argv(tmp_path)))
    national = cli.parse_args(
        _national_argv(tmp_path, "--atomic-support-sha256-ni", "e" * 64)
    )
    with pytest.raises(ValueError, match="--atomic-support-sha256-ni"):
        cli.validate_cli_args(national)


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


def test_engine_blocks_must_be_positive_and_equal_the_clone_count(tmp_path):
    """The per-clone engine block count is either one or the clone count."""
    with pytest.raises(ValueError, match="must equal --n-clones"):
        cli.validate_cli_args(
            arguments(tmp_path, "--n-clones", "4", "--engine-blocks", "2")
        )
    with pytest.raises(ValueError, match="must be positive"):
        cli.validate_cli_args(arguments(tmp_path, "--engine-blocks", "0"))
    cli.validate_cli_args(
        arguments(tmp_path, "--n-clones", "2", "--engine-blocks", "2")
    )


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
        *SUPPORT_ARGUMENTS,
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
        *SUPPORT_ARGUMENTS,
    ]
    cli.validate_cli_args(cli.parse_args(request))


def test_national_role_is_validated_then_built_through_the_graph(tmp_path, monkeypatch):
    """The national line prepares the national graph, never the dense one.

    The validated request goes to ``prepare_national_build`` /
    ``execute_national_build`` after the role tables have run; the seam's
    attempt id, the Hub pre-flight, the Logbook and the staging telemetry
    live in the driver's national envelope.
    """
    argv = _national_argv(tmp_path)
    national = cli.parse_args(argv)
    assert national._posture is cli.uk_rowwise_posture("national")
    assert national.n_clones is None
    assert (national.seed, national.epochs, national.learning_rate) == (0, 1500, 0.02)
    cli.validate_cli_args(national)
    served = []
    monkeypatch.setattr(
        cli, "preflight_staged_dataset", lambda args: served.append("preflight")
    )
    monkeypatch.setattr(
        cli,
        "prepare_national_build",
        lambda args, *, telemetry=None, attempt=None: served.append(args) or "prepared",
    )
    monkeypatch.setattr(
        cli,
        "execute_national_build",
        lambda prepared, args, *, telemetry=None, attempt=None: (
            served.append(prepared) or 7
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_full_build",
        lambda *a, **k: pytest.fail("the national role prepared the dense graph"),
    )
    assert cli.main(argv) == 7
    assert served[0] == "preflight"
    args = served[1]
    assert args.release_role == "national"
    assert args._posture is national._posture
    assert served[2] == "prepared"
    # The dense refusal table still applies before anything is prepared.
    with pytest.raises(ValueError, match="--release-role national refuses"):
        cli.main([*argv, "--ladder", str(tmp_path / "ladder.npz")])
    assert len(served) == 3


def test_national_dry_run_plans_with_the_graph_inventory(tmp_path, monkeypatch, capsys):
    """``--dry-run`` on the national role plans through ``national_dry_run``
    with the compiled graph's inventory: no solve, no staged-dataset
    pre-flight, nothing written."""
    monkeypatch.setattr(
        cli.national_role,
        "national_dry_run",
        lambda args, *, operation_inventory=None: (
            print(
                json.dumps(
                    {
                        "dry_run": args.dry_run,
                        "role": args.release_role,
                        "graph": operation_inventory(),
                    }
                )
            )
            or 0
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_national_build",
        lambda args, *, telemetry=None, attempt=None: SimpleNamespace(
            national=SimpleNamespace(
                operation_inventory=lambda: {"nodes": ["uk.full.national_problem"]}
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_full_build",
        lambda *a, **k: pytest.fail("the national dry run prepared the dense graph"),
    )
    monkeypatch.setattr(
        cli,
        "preflight_staged_dataset",
        lambda args: pytest.fail("a dry run reached the Hub pre-flight"),
    )
    assert cli.main(_national_argv(tmp_path, "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "dry_run": True,
        "role": "national",
        "graph": {"nodes": ["uk.full.national_problem"]},
    }
    assert not (tmp_path / "out").exists()


def test_national_role_refuses_a_spine_request(tmp_path, monkeypatch):
    """The national line solves a bound spine; ``--spine-request`` is dense-only."""
    monkeypatch.setattr(
        cli,
        "preflight_staged_dataset",
        lambda args: pytest.fail("the refusal must precede the Hub pre-flight"),
    )
    argv = [
        "--release-role",
        "national",
        "--spine-request",
        str(tmp_path / "request.json"),
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
    with pytest.raises(ValueError, match="bound spine checkpoint.*--input-h5"):
        cli.main(argv)
    assert not (tmp_path / "out").exists()


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
                *SUPPORT_ARGUMENTS,
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
    assert manifest["graph"]["epoch_rows"] == "dense_solve,size_search,size_refit"
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


def test_dense_dry_run_starts_and_finishes_hosted_telemetry(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--dry-run")
    events = []

    class FakeEmitter:
        available = True

        def transition_stage(self, stage_id, **details):
            events.append(("stage", stage_id, details))

        def complete(self):
            events.append(("complete",))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        cli,
        "start_telemetry_emitter",
        lambda requested, *, build_id, run_kind: (
            events.append(("start", build_id, run_kind)) or FakeEmitter()
        ),
    )
    monkeypatch.setattr(
        cli, "_dry_run", lambda requested: events.append(("plan",)) or 0
    )
    monkeypatch.setattr(
        cli,
        "finalize_staging_run_bundle",
        lambda requested, telemetry: None,
    )

    assert cli.main([]) == 0
    assert [event[0] for event in events] == [
        "start",
        "stage",
        "plan",
        "complete",
        "close",
    ]
    assert events[0][2] == "dry_run"


def test_dense_preflight_failure_is_reported_by_early_emitter(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    events = []

    class FakeEmitter:
        available = True

        def transition_stage(self, stage_id, **details):
            events.append(("stage", stage_id, details))

        def fail(self, error, **classification):
            events.append(("failed", None, str(error), classification))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        cli,
        "start_telemetry_emitter",
        lambda requested, *, build_id, run_kind: (
            events.append(("start", build_id, run_kind)) or FakeEmitter()
        ),
    )

    def refuse_preflight(requested):
        events.append(("preflight",))
        raise RuntimeError("staging credential unavailable")

    monkeypatch.setattr(cli, "preflight_staged_dataset", refuse_preflight)
    monkeypatch.setattr(
        cli,
        "fail_staging_run_bundle",
        lambda telemetry, error: None,
    )
    monkeypatch.setattr(
        cli,
        "create_staging_run_bundle",
        lambda *args, **kwargs: pytest.fail("preflight failure reached staging setup"),
    )

    with pytest.raises(RuntimeError, match="staging credential unavailable"):
        cli.main([])
    assert [event[0] for event in events] == [
        "start",
        "stage",
        "preflight",
        "failed",
        "close",
    ]
    assert events[-2][1] is None
    # Raised before the attempt took over, the error is classified all the same.
    assert events[-2][3] == {"failure_class": "error", "error_code": "BUILD_FAILED"}


def test_hosted_stage_reporting_survives_staging_bundle_refusal(monkeypatch, capsys):
    from microcosm.build.staging_v2 import StagingContentError
    from microcosm.build.uk_runtime import rowwise_staging

    events = []

    class RefusingBundle:
        def stage(self, *args, **kwargs):
            raise StagingContentError("synthetic staging bundle refusal")

    class Emitter:
        def transition_stage(self, stage_id, **details):
            events.append((stage_id, details))

    monkeypatch.setattr(rowwise_staging, "_ACTIVE_EMITTER", Emitter())

    rowwise_staging.stage(
        RefusingBundle(),
        "target_compilation",
        "started",
        selected_target_count=10,
    )

    assert events == [
        (
            "target_compilation",
            {"status": "started", "message": None, "selected_target_count": 10},
        )
    ]
    assert "synthetic staging bundle refusal" in capsys.readouterr().err


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


def test_input_inside_the_output_directory_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    """An input that would collide with a published file keeps its bytes.

    The candidate tool refused a ladder named as the manifest inside the
    output directory; the graph driver refuses any input source under the
    output directory before it writes, and the colliding file is untouched.
    """
    args = arguments(tmp_path)
    args.out.mkdir()
    collision = args.out / cli.MANIFEST_FILENAME
    collision.write_bytes(b"ladder stand-in living where the manifest goes")
    build = prepared(tmp_path)
    build = replace(build, sources={"fixture": collision})
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "prepare_full_build", lambda args, **kwargs: build)
    assert cli.main([]) == 1
    assert collision.read_bytes() == b"ladder stand-in living where the manifest goes"
    assert list(args.out.iterdir()) == [collision]


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


def test_main_stages_the_bundle_locally_with_staging_local_only(
    tmp_path, monkeypatch, capsys
):
    """``--staging-local-only`` writes and validates the v2 bundle, uploads nothing.

    The second half restores the candidate tool's bundle-inventory contract on
    the graph driver: the stdout manifest is the on-disk one, the run id is
    the build id, the staged inventory names every output with its digest, the
    local sums verify the directory, and the sidecars are never outputs.
    """

    pytest.importorskip("tables")
    from microcosm.build.staging_dataset import (
        SHA256SUMS_FILENAME,
        STAGED_MANIFEST_FILENAME,
    )
    from microcosm.build.staging_v2 import validate_v2_bundle

    staging_dir = tmp_path / "staging-bundle"
    out = graph_dense_bundle(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
    )
    manifest = json.loads((out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staging_delivery"]["upload_attempts"] == 0
    assert manifest["staged_dataset"]["mode"] == "local_only"
    assert manifest["staged_dataset"]["status"] == "skipped"
    assert manifest["staged_dataset"]["repository"] is None
    # The published bundle carries its own inventory beside the manifest.
    assert (out / STAGED_MANIFEST_FILENAME).is_file()
    assert (out / SHA256SUMS_FILENAME).is_file()

    runs = sorted(path.name for path in (staging_dir / "runs").iterdir())
    assert len(runs) == 1
    bundle = validate_v2_bundle(staging_dir, runs[0])
    assert bundle["progress"]["status"] == "completed"
    assert bundle["run_manifest"]["run_kind"] == "calibration"
    transitions = [
        event["status"]
        for event in bundle["events"]
        if event["stage_id"] == "dataset_staging"
    ]
    assert transitions == ["started", "completed"]
    artifacts = staging_dir / "runs" / runs[0] / "artifacts"
    staged = json.loads((artifacts / "staged_dataset.json").read_text())
    assert staged["mode"] == "local_only" and staged["status"] == "skipped"
    assert (artifacts / "fit_summary.json").is_file()

    captured = capsys.readouterr()
    # The stdout manifest is the on-disk manifest, evidence blocks included.
    assert json.loads(captured.out)["staged_dataset"] == manifest["staged_dataset"]
    assert "staged dataset: skipped (local_only)" in captured.err
    run_id = runs[0]
    rows = spool_rows(out)
    assert rows[0].build_id == run_id
    run_manifest = bundle["run_manifest"]
    assert run_manifest["operation_id"] == "uk_rowwise_candidate"
    assert run_manifest["pipeline"]["id"] == "uk-local-candidate"
    assert run_manifest["non_release"] is True
    # The f100 rung stages the contract's one sampling statement, as the
    # candidate tool wrote after its sampling step (receipts R5, row 18).
    assert run_manifest["sample"] == {"mode": "full"}
    assert bundle["progress"]["sample"] == {"mode": "full"}
    assert run_manifest["delivery"]["mode"] == "local_only"
    assert run_manifest["delivery"]["upload_attempts"] == 0
    assert {a["logical_name"] for a in run_manifest["artifacts"]} == {
        "fit_summary",
        "staged_dataset",
    }
    fit_summary = json.loads((artifacts / "fit_summary.json").read_text())
    assert fit_summary["run_id"] == run_id
    assert fit_summary["gates"]["uk_local_target_fit"] == "passed"
    assert fit_summary["loss"]["final"] == manifest["solve"]["final_loss"]
    assert staged == manifest["staged_dataset"]
    # Every graph phase reports started then completed, in build order. The
    # synthetic preflight gate binds no selection node, so target compilation
    # only starts here; on the real graph it completes with the selected count.
    events = bundle["events"]
    completed = [e["stage_id"] for e in events if e["status"] == "completed"]
    assert completed == [
        "calibration",
        "gate_battery",
        "output_bundle",
        "dataset_staging",
        "complete",
    ]
    assert [e["status"] for e in events if e["stage_id"] == "target_compilation"] == [
        "started"
    ]
    for stage in ("calibration", "output_bundle"):
        transitions = [e["status"] for e in events if e["stage_id"] == stage]
        assert transitions == ["started", "completed"], stage
    # The manifest carries both receipts; the bundle inventory is the outputs.
    assert manifest["staging_delivery"]["run_id"] == run_id
    assert staged["prefix"] == f"staged/{run_id}"
    assert set(staged["files"]) == {
        Path(entry["path"]).name for entry in manifest["outputs"].values()
    }
    for entry in manifest["outputs"].values():
        assert staged["files"][Path(entry["path"]).name]["sha256"] == entry["sha256"]
    # The local sums verify the directory as it is, evidence blocks included.
    for line in (out / SHA256SUMS_FILENAME).read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
    inventory = json.loads((out / STAGED_MANIFEST_FILENAME).read_text())
    assert inventory["run_id"] == run_id and inventory["files"] == staged["files"]
    assert inventory["summary"]["releasable"] is True
    assert inventory["telemetry"] == {
        "repository": None,
        "prefix": f"runs/{run_id}",
        "mode": "local_only",
    }
    # The sidecars are evidence about the outputs, never outputs themselves.
    assert "sha256sums" not in manifest["outputs"]
    assert "staged_manifest" not in manifest["outputs"]


def test_sampled_run_stages_a_null_sample_block(tmp_path, monkeypatch):
    """A rung below f100 stages ``sample: null``, as the candidate tool did.

    The contract's only sampling statement is ``{"mode": "full"}``. The driver
    judges the rung on the effective fraction (pool times source spine), read
    from the prepared build's configuration, so a ``--sample-fraction 0.1``
    pool over a full spine stages no sample block while its Logbook row
    carries the f010 rung.
    """

    pytest.importorskip("tables")
    from microcosm.build.staging_v2 import validate_v2_bundle

    arguments(tmp_path)  # writes the stand-ins the prepared build pins
    build = prepared(tmp_path)
    sampled = replace(
        build,
        full=replace(
            build.full, config=replace(build.full.config, sample_fraction=0.1)
        ),
    )
    assert sampled.full.config.effective_sample_fraction == 0.1
    staging_dir = tmp_path / "staging-bundle"
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--sample-fraction",
        "0.1",
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
        build=sampled,
    )
    assert status == 0
    runs = sorted(path.name for path in (staging_dir / "runs").iterdir())
    assert len(runs) == 1
    bundle = validate_v2_bundle(staging_dir, runs[0])
    assert bundle["progress"]["status"] == "completed"
    assert bundle["run_manifest"]["sample"] is None
    assert bundle["progress"]["sample"] is None
    assert spool_rows(out)[0].rung == "f010"


# ---------------------------------------------------------------------------
# Delivery-side resilience and refusal receipts on the graph driver. These
# restore the in-process candidate tool's contracts (retired in af01b990d;
# mapping in experiments/901-uk-main-rebase-receipts.md, R5) against
# ``full_build_cli.main`` over the synthetic dense build: a delivery-side
# problem must never fail a good build, and a refusal must leave its receipt
# pointers on the Logbook row.
# ---------------------------------------------------------------------------


def _drive_epochs(args, telemetry, epochs: int = 2) -> None:
    """Feed synthetic dense-solve epochs through the driver's own observer.

    The synthetic graph has no calibration kernel, so the thinned staging rows
    the real solve would forward are produced here through the very callback
    ``prepare_full_build`` hands the kernels.
    """
    observer = cli._solve_observer(args, telemetry)
    for epoch in range(1, epochs + 1):
        observer(
            {
                "kind": "calibration_epoch",
                "epoch": epoch,
                "epochs": epochs,
                "loss": 0.5 / epoch,
            }
        )


def _remote_hub(monkeypatch, **kwargs):
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.uk_rowwise_candidate import _FakeHub

    hub = _FakeHub(**kwargs)
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: hub)
    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: "hf_test_token")
    return hub


def test_no_staging_records_both_opt_outs(tmp_path, monkeypatch):
    pytest.importorskip("tables")
    status, out = run_dense_main(tmp_path, monkeypatch, staging="--no-staging")
    assert status == 0
    assert not (out / "staging").exists()
    assert not (out / "sha256sums.txt").exists()
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "disabled"
    assert manifest["staging_delivery"]["opt_out_reason"] == "--no-staging"
    assert manifest["staged_dataset"] == {
        "contract_version": 1,
        "mode": "disabled",
        "repository": None,
        "prefix": None,
        "run_id": None,
        "revision": None,
        "status": "skipped",
        "error_code": None,
        "opt_out_reason": "--no-staging",
        "files": {},
    }
    rows = spool_rows(out)
    assert "dataset_stage_skipped" in rows[0].phases_reached


def test_invalid_local_telemetry_bundle_is_a_warning_not_the_runs_failure(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("tables")
    from microcosm.build.staging_v2 import StagingContractError
    from microcosm.build.uk_runtime import rowwise_staging

    class Invalid(rowwise_staging.StagingRunBundleWriterV2):
        def validate_local_bundle(self):
            raise StagingContractError("synthetic bundle defect")

    monkeypatch.setattr(rowwise_staging, "StagingRunBundleWriterV2", Invalid)
    status, out = run_dense_main(tmp_path, monkeypatch, staging="--staging-local-only")
    assert status == 0
    err = capsys.readouterr().err
    assert "does not validate" in err and "synthetic bundle defect" in err
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staged_dataset"]["status"] == "skipped"
    rows = spool_rows(out)
    assert rows and rows[0].disposition == "iterating"


def test_telemetry_content_refusal_never_aborts_the_solve(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("tables")
    from microcosm.build.staging_v2 import StagingContentError, validate_v2_bundle
    from microcosm.build.uk_runtime import rowwise_staging

    class Refusing(rowwise_staging.StagingRunBundleWriterV2):
        def calibration_progress(self, event):
            raise StagingContentError("Staging file exceeds the 5242880-byte limit.")

    monkeypatch.setattr(rowwise_staging, "StagingRunBundleWriterV2", Refusing)
    status, out = run_dense_main(
        tmp_path, monkeypatch, staging="--staging-local-only", on_prepare=_drive_epochs
    )
    assert status == 0
    err = capsys.readouterr().err
    assert err.count("no longer forwarded") == 1
    run_id = single_run_id(out)
    bundle = validate_v2_bundle(out / "staging", run_id)
    assert bundle["run_manifest"]["status"] == "completed"
    assert not (
        out / "staging" / "runs" / run_id / "calibration_progress.json"
    ).exists()
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert spool_rows(out)[0].disposition == "iterating"


def test_remote_staging_uploads_telemetry_and_the_bundle_in_one_commit(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("tables")
    hub = _remote_hub(monkeypatch)
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-upload-interval-seconds",
        "0",
        staging=None,
        on_prepare=_drive_epochs,
    )
    assert status == 0
    err = capsys.readouterr().err
    run_id = single_run_id(out)
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())

    # Telemetry went to runs/<run_id>/ of the staging repository, artifacts
    # included, under the fixed prefix and nothing else.
    telemetry_paths = hub.paths("policyengine/populace-uk-staging")
    assert telemetry_paths == sorted(
        f"runs/{run_id}/{name}"
        for name in (
            "run_manifest.json",
            "progress.json",
            "events.ndjson",
            "calibration_progress.json",
            "artifacts/fit_summary.json",
            "artifacts/staged_dataset.json",
        )
    )
    delivery = manifest["staging_delivery"]
    assert delivery["mode"] == "local_and_remote"
    assert delivery["configured_repository"] == "policyengine/populace-uk-staging"
    assert delivery["upload_successes"] == delivery["upload_attempts"] > 0
    remote_progress = json.loads(
        hub.files[("policyengine/populace-uk-staging", f"runs/{run_id}/progress.json")]
    )
    assert remote_progress["status"] == "completed"

    # The bundle went to staged/<run_id>/ of the private repository in one
    # commit: every output, the manifest as built, and the two sidecars.
    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert commit["repo_id"] == "policyengine/populace-uk-private"
    expected = {Path(e["path"]).name for e in manifest["outputs"].values()} | {
        cli.MANIFEST_FILENAME,
        "staged_manifest.json",
        "sha256sums.txt",
    }
    assert commit["paths"] == sorted(f"staged/{run_id}/{name}" for name in expected)
    assert hub.paths("policyengine/populace-uk-private") == commit["paths"]
    staged = manifest["staged_dataset"]
    assert staged["status"] == "uploaded"
    assert staged["repository"] == "policyengine/populace-uk-private"
    assert staged["prefix"] == f"staged/{run_id}"
    assert staged["revision"] == hub.sha
    assert (
        f"staged dataset: uploaded at policyengine/populace-uk-private/staged/{run_id}"
        in err
    )
    dataset_name = Path(manifest["outputs"]["dataset"]["path"]).name
    remote_h5 = hub.files[
        ("policyengine/populace-uk-private", f"staged/{run_id}/{dataset_name}")
    ]
    assert remote_h5 == (out / dataset_name).read_bytes()
    # The uploaded manifest is the one the bundle was built from; the local
    # copy gained the two evidence blocks afterwards.
    remote_manifest = json.loads(
        hub.files[
            (
                "policyengine/populace-uk-private",
                f"staged/{run_id}/{cli.MANIFEST_FILENAME}",
            )
        ]
    )
    assert (
        "staged_dataset" not in remote_manifest
        and "staging_delivery" not in remote_manifest
    )
    assert remote_manifest["outputs"] == manifest["outputs"]
    rows = spool_rows(out)
    assert "dataset_staged" in rows[0].phases_reached
    assert rows[0].disposition == "iterating"

    def sums_verify() -> None:
        for line in (out / "sha256sums.txt").read_text().splitlines():
            digest, name = line.split("  ")
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name

    # Re-staging a directory whose record already says these outputs are
    # uploaded touches nothing: the driver's record and revision stand, the
    # sidecars keep their bytes, and no commit is made.
    stager = load_tool("stage_uk_rowwise_candidate")
    monkeypatch.setattr(stager, "_hub_api", lambda: hub)
    sidecar_bytes = (out / "staged_manifest.json").read_bytes()
    capsys.readouterr()
    assert stager.main(["--run-dir", str(out)]) == 0
    assert "nothing to do" in capsys.readouterr().err
    restaged = json.loads((out / cli.MANIFEST_FILENAME).read_text())["staged_dataset"]
    assert restaged == staged
    assert (out / "staged_manifest.json").read_bytes() == sidecar_bytes
    sums_verify()
    assert len(hub.commits) == 1

    # A record that says the upload failed while the Hub already holds these
    # outputs: the re-stage finds the bundle and records its own commit, not
    # the repository head, which has moved on since.
    manifest_path = out / cli.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["staged_dataset"] = {
        **staged,
        "status": "failed",
        "revision": None,
        "error_code": "UPLOAD_FAILED",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    hub.sha = "e" * 40
    assert stager.main(["--run-dir", str(out)]) == 0
    recovered = json.loads(manifest_path.read_text())["staged_dataset"]
    assert recovered["status"] == "already_staged"
    assert recovered["revision"] == staged["revision"] != hub.sha
    assert len(hub.commits) == 1
    sums_verify()

    # Consumers fetch by run id and get digest-verified local files.
    fetcher = load_tool("fetch_uk_staged_dataset")
    monkeypatch.setattr(fetcher, "_hub_api", lambda: hub)
    dest = tmp_path / "fetched"
    capsys.readouterr()
    assert fetcher.main(["--run-id", run_id, "--dest", str(dest)]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert str(dest / dataset_name) in listed
    assert (dest / "sha256sums.txt").is_file()
    assert (dest / dataset_name).read_bytes() == remote_h5


def test_remote_staging_failure_is_recorded_and_the_build_still_succeeds(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("tables")
    hub = _remote_hub(monkeypatch, fail_commit=True)
    status, out = run_dense_main(tmp_path, monkeypatch, staging=None)
    assert status == 0
    err = capsys.readouterr().err
    assert "staged dataset upload failed" in err and "do-not-record" not in err
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    staged = manifest["staged_dataset"]
    assert staged["status"] == "failed" and staged["error_code"] == "UPLOAD_FAILED"
    assert staged["revision"] is None and staged["files"]
    assert "do-not-record" not in json.dumps(manifest)
    assert hub.paths("policyengine/populace-uk-private") == []
    # Telemetry still completed and recorded the outcome.
    run_id = single_run_id(out)
    progress = json.loads(
        hub.files[("policyengine/populace-uk-staging", f"runs/{run_id}/progress.json")]
    )
    assert progress["status"] == "completed"
    events = [
        json.loads(line)
        for line in hub.files[
            ("policyengine/populace-uk-staging", f"runs/{run_id}/events.ndjson")
        ]
        .decode()
        .splitlines()
        if line
    ]
    done = next(
        e
        for e in events
        if e["stage_id"] == "dataset_staging" and e["status"] == "completed"
    )
    assert done["details"]["status"] == "failed"
    assert done["details"]["error_code"] == "UPLOAD_FAILED"
    rows = spool_rows(out)
    assert rows[0].disposition == "iterating"
    assert "dataset_stage_failed" in rows[0].phases_reached
    # The sidecars are in place for a later re-stage.
    assert (out / "sha256sums.txt").is_file() and (
        out / "staged_manifest.json"
    ).is_file()


def test_no_staged_dataset_keeps_telemetry_remote_and_the_bundle_local(
    tmp_path, monkeypatch
):
    pytest.importorskip("tables")
    hub = _remote_hub(monkeypatch)
    status, out = run_dense_main(
        tmp_path, monkeypatch, "--no-staged-dataset", staging=None
    )
    assert status == 0
    assert hub.paths("policyengine/populace-uk-private") == []
    assert hub.commits == []
    assert hub.paths("policyengine/populace-uk-staging")
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["staging_delivery"]["mode"] == "local_and_remote"
    assert manifest["staged_dataset"]["mode"] == "disabled"
    assert manifest["staged_dataset"]["opt_out_reason"] == "--no-staged-dataset"
    assert not (out / "sha256sums.txt").exists()


def test_refusal_records_the_gate_and_error_receipt_pointers(tmp_path, monkeypatch):
    """A refusal leaves its receipt pointer on the failed Logbook row.

    The candidate tool re-raised a blocking gate and recorded both the gate
    verdict and a pipeline error; the graph driver returns the block as a
    non-zero status and records the failed gate's receipt pointer, and a run
    that raises records the error receipt pointer.
    """
    pytest.importorskip("tables")
    geography = "uk_local_geography_ladder_post_calibration"
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    status, out = run_dense_main(blocked, monkeypatch, failed=geography)
    assert status == 1
    gate_report_path = out / f"{STEM}.local_gates.json"
    assert gate_report_path.exists()
    rows = spool_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert row.gate_verdicts[geography] == {
        "verdict": "failed",
        "receipt": f"{local_ref(gate_report_path)}#/gates/{geography}",
    }
    assert "pipeline_error" not in row.gate_verdicts

    # A refusal after the targets are bound (the binding adjudication's
    # place on the candidate tool) and before the solve completes.
    raised = tmp_path / "raised"
    raised.mkdir()
    original_run = cli.run_graph

    def refuse_numerical(compiled, **kwargs):
        if "uk.full.gates.calibrated" in {node.id for node in compiled.graph.nodes}:
            raise ValueError("census_disclosure_control_noise is not adjudicated")
        return original_run(compiled, **kwargs)

    monkeypatch.setattr(cli, "run_graph", refuse_numerical)
    status, out = run_dense_main(raised, monkeypatch)
    assert status == 1
    rows = spool_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert "targets_bound" in row.phases_reached
    assert "solved" not in row.phases_reached
    receipts = list((out / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert row.gate_verdicts["pipeline_error"] == {
        "verdict": "error",
        "receipt": f"{local_ref(receipts[0])}#/error_type",
    }
    assert not (out / cli.MANIFEST_FILENAME).exists()

    # A pre-graph setup failure (the ladder load's place on the candidate
    # tool) still spools a failed row with its error receipt pointer.
    setup = tmp_path / "setup"
    setup.mkdir()
    monkeypatch.setattr(cli, "run_graph", original_run)

    def refuse_setup(args, telemetry):
        raise RuntimeError("ladder artifact refused to parse")

    status, out = run_dense_main(setup, monkeypatch, on_prepare=refuse_setup)
    assert status == 1
    rows = spool_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row.disposition == "failed"
    assert "targets_bound" not in row.phases_reached
    receipts = list((out / "logbook-receipts").rglob("error.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["message"] == (
        "ladder artifact refused to parse"
    )
    assert row.gate_verdicts["pipeline_error"] == {
        "verdict": "error",
        "receipt": f"{local_ref(receipts[0])}#/error_type",
    }
    assert not (out / cli.MANIFEST_FILENAME).exists()


def test_blocked_gates_partition_failures_by_criticality(tmp_path, monkeypatch, capsys):
    """Two release-blocking local failures are both enforced and none is diagnostic."""
    pytest.importorskip("tables")
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        failed=(
            ("uk_local_area_support", "ESS 42.3 < 50"),
            ("uk_local_weight_ratio", "ratio 578 > 100"),
        ),
    )
    assert status == 1
    assert "artifact unreleasable" in capsys.readouterr().err
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["failing_gate_ids"] == [
        "uk_local_area_support",
        "uk_local_weight_ratio",
    ]
    assert manifest["blocked_at_f100"] is True
    assert manifest["blocking_failures"] == [
        "[uk_local_area_support] ESS 42.3 < 50",
        "[uk_local_weight_ratio] ratio 578 > 100",
    ]
    assert manifest["diagnostic_failures"] == []
    assert manifest["releasable"] is False
    report = json.loads(
        Path(manifest["outputs"]["local_gate_report"]["path"]).read_text()
    )
    outcomes = {outcome["id"]: outcome for outcome in report["report"]["outcomes"]}
    for gate_id in ("uk_local_area_support", "uk_local_weight_ratio"):
        assert outcomes[gate_id]["criticality"] == "release_blocking"
        assert outcomes[gate_id]["status"] == "failed"
    assert spool_rows(out)[0].disposition == "failed"


def test_multi_block_engine_run_is_releasable_when_its_blocks_represent_the_pool(
    tmp_path, monkeypatch
):
    """End to end: ``--engine-blocks K`` on f100 with every block an identical
    copy scaled to the pool writes ``releasable: true``; the manifest names the
    legs (``single_block_engine`` false, ``engine_population_exact`` true).
    """
    pytest.importorskip("tables")
    status, out = run_dense_main(
        tmp_path, monkeypatch, "--n-clones", "2", "--engine-blocks", "2"
    )
    assert status == 0
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["parameters"]["engine_blocks"] == 2
    assert manifest["blocking_failures"] == []
    assert manifest["releasable"] is True
    posture = manifest["release_posture"]
    assert posture["full_rung"] is True
    assert posture["single_block_engine"] is False
    assert posture["engine_population_exact"] is True
    assert posture["release_blocking_gates_passed"] is True


def test_multi_block_engine_run_without_an_exact_representation_is_never_releasable(
    tmp_path, monkeypatch
):
    """The blocks were not identical copies: the scaling is approximate, the
    measures receipt keeps its caveat, and the posture alone withholds the
    verdict with every release-blocking gate passed (#736 erratum).
    """
    pytest.importorskip("tables")
    monkeypatch.setattr(support, "ENGINE_POPULATION_EXACT", False)
    status, out = run_dense_main(
        tmp_path, monkeypatch, "--n-clones", "2", "--engine-blocks", "2"
    )
    assert status == 0
    manifest = json.loads((out / cli.MANIFEST_FILENAME).read_text())
    assert manifest["blocking_failures"] == []
    assert manifest["releasable"] is False
    posture = manifest["release_posture"]
    assert posture["single_block_engine"] is False
    assert posture["engine_population_exact"] is False
    assert posture["release_blocking_gates_passed"] is True


def test_every_typed_edge_of_the_full_graph_has_a_compatible_consumer(tmp_path):
    """Amendment 19 refuses a typed artifact whose consumer claims a stronger
    numeric class than its producer; the first licensed graph build found the
    calibrated gate battery reading a platform-bitwise holdout report. Every
    typed edge of the composed graph is checked here, on the synthetic spec."""
    from microcosm.graph.artifact_edges import numeric_scope, require_compatible_scope

    build = prepared(tmp_path)
    graph, kernels = build.full.graph, build.kernels
    incompatible = []
    typed_edges = 0
    for node in graph.nodes:
        consumer = kernels.get(node.kernel).capabilities
        for binding in node.artifact_inputs:
            typed_edges += 1
            producer = kernels.get(graph.node(binding.producer).kernel).capabilities
            try:
                require_compatible_scope(numeric_scope(producer), consumer)
            except Exception as error:  # noqa: BLE001 - the message is the finding
                incompatible.append(
                    (node.id, binding.producer, binding.name, str(error))
                )
    producers = {b.producer for n in graph.nodes for b in n.artifact_inputs}
    assert {"uk.full.holdout", "uk.full.gates.calibrated"} <= producers
    assert typed_edges > 0
    assert incompatible == []


def test_gate_battery_recorded_exception_is_named_not_a_missing_artifact():
    """A GATE kernel keeps a failure inside its receipt and completes with no
    artifacts; the driver names the node and the exception (first licensed
    graph build: a bare KeyError on the missing diagnostics artifact)."""
    from types import SimpleNamespace

    import pytest

    failed = SimpleNamespace(
        nodes={
            "uk.full.gates.calibrated": SimpleNamespace(
                receipt={
                    "outcome": "fail",
                    "evidence": {
                        "exception_type": "ValueError",
                        "message": "registry must exactly partition",
                    },
                }
            )
        }
    )
    with pytest.raises(RuntimeError, match="uk.full.gates.calibrated failed inside"):
        cli._require_gate_kernel_completed(failed, "uk.full.gates.calibrated")
    passed = SimpleNamespace(
        nodes={
            "uk.full.gates.calibrated": SimpleNamespace(
                receipt={"outcome": "fail", "evidence": {"blocking": ["a_gate"]}}
            )
        }
    )
    cli._require_gate_kernel_completed(passed, "uk.full.gates.calibrated")


def test_dense_completion_marker_binds_the_closed_manifest(tmp_path, monkeypatch):
    """``build.json`` is re-issued after the close step appends the staging
    receipts to the manifest, so its digest is the manifest's final bytes."""
    status, out = run_dense_main(tmp_path, monkeypatch)
    assert status == 0
    completion = json.loads((out / "build.json").read_text())
    entry = completion["rowwise_candidate_manifest"]
    manifest_path = out / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "staged_dataset" in manifest and "staging_delivery" in manifest
    assert entry["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert entry["size_bytes"] == manifest_path.stat().st_size
    assert "note" not in entry


def test_national_interrupt_records_a_discarded_row_and_re_raises(
    tmp_path, monkeypatch
):
    """A Ctrl-C during the national build is a discarded attempt, not a failed
    one: the seam's terminal-disposition contract, kept by the driver (the
    failure sidecar names the interrupt, the row says ``discarded``, the
    interrupt propagates)."""
    argv = _national_argv(tmp_path)
    monkeypatch.setattr(cli, "preflight_staged_dataset", lambda args: None)
    monkeypatch.setattr(
        cli,
        "prepare_national_build",
        lambda args, *, telemetry=None, attempt=None: "prepared",
    )

    def interrupted(prepared, args, *, telemetry=None, attempt=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_national_build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(argv)
    out = tmp_path / "out"
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    rows = load_spool_rows(out / "logbook-spool")
    assert [row.disposition for row in rows] == ["discarded"]
    assert rows[0].pipeline == "uk-frs-calibration"


def test_dense_interrupt_records_a_discarded_row_and_re_raises(tmp_path, monkeypatch):
    """The dense line closes an interrupted attempt the same way."""
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    patch_certification(monkeypatch)
    args = arguments(tmp_path)
    build = prepared(tmp_path)
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(
        cli, "prepare_full_build", lambda args, *, telemetry=None, attempt=None: build
    )

    def interrupted(prepared, args, *, telemetry=None, attempt=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_full_build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main([])
    failure = json.loads((args.out / "failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    rows = load_spool_rows(args.out / "logbook-spool")
    assert [row.disposition for row in rows] == ["discarded"]


@pytest.mark.parametrize("failed", [None, "uk_local_geography_ladder_post_calibration"])
def test_dense_validation_result_matches_emitted_run_status(
    tmp_path, monkeypatch, fake_telemetry_emitters, failed
):
    status, _ = run_dense_main(tmp_path, monkeypatch, failed=failed)
    assert status == (1 if failed else 0)
    events = [
        event
        for event in fake_telemetry_emitters[-1].events
        if event["event_type"] == "run"
    ]
    # A candidate the gates refused is a blocked run, not a failed one.
    assert [event["status"] for event in events] == [
        "blocked" if failed else "completed"
    ]


@pytest.mark.parametrize("role", ["dense", "national"])
@pytest.mark.parametrize("outcome", [0, 7, "error", "interrupt"])
def test_build_lifecycle_handles_returned_errors_and_exceptions(
    tmp_path, monkeypatch, fake_telemetry_emitters, role, outcome
):
    args = (
        arguments(tmp_path)
        if role == "dense"
        else cli.parse_args(_national_argv(tmp_path))
    )
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    monkeypatch.setattr(cli, "_record_failure", lambda *args, **kwargs: None)
    prepare_name = "prepare_full_build" if role == "dense" else "prepare_national_build"
    execute_name = "execute_full_build" if role == "dense" else "execute_national_build"
    monkeypatch.setattr(cli, prepare_name, lambda *args, **kwargs: object())
    error = ValueError("synthetic build error")
    interrupt = KeyboardInterrupt("synthetic interrupt")

    def execute(*args, **kwargs):
        if outcome == "error":
            raise error
        if outcome == "interrupt":
            raise interrupt
        return outcome

    monkeypatch.setattr(cli, execute_name, execute)
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt) as caught:
            cli.main([])
        assert caught.value is interrupt
    else:
        assert cli.main([]) == (1 if outcome == "error" else outcome)
    emitter = fake_telemetry_emitters[-1]
    events = [event for event in emitter.events if event["event_type"] == "run"]
    assert [event["status"] for event in events] == [
        "completed" if outcome == 0 else "failed"
    ]
    if outcome == "error":
        assert events[0]["message"] == str(error)
    assert not emitter.available


@pytest.mark.parametrize("role", ["dense", "national"])
def test_dry_run_nonzero_result_emits_failure(
    tmp_path, monkeypatch, fake_telemetry_emitters, role
):
    args = (
        arguments(tmp_path, "--dry-run")
        if role == "dense"
        else cli.parse_args(_national_argv(tmp_path, "--dry-run"))
    )
    monkeypatch.setattr(cli, "parse_args", lambda argv: args)
    if role == "dense":
        monkeypatch.setattr(cli, "_dry_run", lambda *args, **kwargs: 7)
    else:
        monkeypatch.setattr(
            cli.national_role, "national_dry_run", lambda *args, **kwargs: 7
        )
    assert cli.main([]) == 7
    events = [
        event
        for event in fake_telemetry_emitters[-1].events
        if event["event_type"] == "run"
    ]
    assert [event["status"] for event in events] == ["failed"]


def test_staging_bundle_finalization_does_not_complete_hosted_run(
    tmp_path, monkeypatch, fake_telemetry_emitters
):
    from microcosm.build.uk_runtime import rowwise_staging

    args = arguments(tmp_path)
    emitter = rowwise_staging.start_telemetry_emitter(args, build_id="cleanup-only")
    rowwise_staging.finalize_staging_run_bundle(args, None)
    assert emitter.events == []
    assert emitter.available


def _only_staging_run(staging_dir: Path):
    from microcosm.build.staging_v2 import validate_staging_bundle

    runs = sorted(path.name for path in (staging_dir / "runs").iterdir())
    assert len(runs) == 1
    return validate_staging_bundle(staging_dir, runs[0])


def test_a_gate_block_closes_the_staging_run_as_blocked(tmp_path, monkeypatch):
    """The terminal battery's refusal is a ``blocked`` run, not a completed one,
    and the Logbook row records the same end as ``failed``."""
    pytest.importorskip("tables")
    staging_dir = tmp_path / "staging-bundle"
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
        failed=(("uk_local_area_support", "ESS 42.3 < 50"),),
    )
    assert status == 1
    bundle = _only_staging_run(staging_dir)
    assert bundle["progress"]["status"] == "blocked"
    assert bundle["progress"]["failure"] is None
    assert bundle["progress"]["block"] == {
        "phase": "terminal",
        "blocking_failure_count": 1,
        "blocking_gate_ids": ["uk_local_area_support"],
    }
    terminal = bundle["events"][-1]
    assert (terminal["stage_id"], terminal["status"]) == ("blocked", "blocked")
    assert terminal["details"]["gate_statuses"]["uk_local_area_support"] == "failed"
    assert spool_rows(out)[0].disposition == "failed"


def test_a_preflight_refusal_is_blocked_at_preflight_not_passed(tmp_path, monkeypatch):
    """Refused before solving, the run closes ``blocked`` at phase ``preflight``
    with the refusing gate, and its Logbook receipts resolve in the preflight
    gate document."""
    pytest.importorskip("tables")
    gate = "uk_target_surface_local_default_2025"
    staging_dir = tmp_path / "staging-bundle"
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
        failed=gate,
    )
    assert status == 1
    bundle = _only_staging_run(staging_dir)
    assert bundle["progress"]["status"] == "blocked"
    assert bundle["progress"]["block"]["phase"] == "preflight"
    assert bundle["progress"]["block"]["blocking_gate_ids"] == [gate]
    stages = [event["stage_id"] for event in bundle["events"]]
    assert "preflight_gates" in stages and "calibration" not in stages
    row = spool_rows(out)[0]
    assert row.disposition == "failed"
    assert "candidate_blocked_at_preflight" in row.phases_reached
    report_path = out / "uk.full.gates.preflight.gate_report.json"
    assert report_path.is_file()
    receipt = row.gate_verdicts[gate]["receipt"]
    assert receipt.startswith(local_ref(report_path))
    index = int(receipt.rsplit("/", 1)[1])
    outcome = json.loads(report_path.read_text())["report"]["outcomes"][index]
    assert outcome["id"] == gate and row.gate_verdicts[gate]["verdict"] == "failed"


def test_a_raised_error_is_classified_in_the_staging_run(tmp_path, monkeypatch):
    """A build that raises closes ``failed`` with a mapped code and class."""
    pytest.importorskip("tables")
    staging_dir = tmp_path / "staging-bundle"
    original_run = cli.run_graph

    def run_out_of_memory(compiled, **kwargs):
        if "uk.full.gates.calibrated" in {node.id for node in compiled.graph.nodes}:
            raise MemoryError("pool too large")
        return original_run(compiled, **kwargs)

    monkeypatch.setattr(cli, "run_graph", run_out_of_memory)
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
    )
    assert status == 1
    failure = _only_staging_run(staging_dir)["progress"]["failure"]
    assert (failure["error_code"], failure["failure_class"]) == (
        "OUT_OF_MEMORY",
        "out_of_memory",
    )
    assert spool_rows(out)[0].disposition == "failed"
