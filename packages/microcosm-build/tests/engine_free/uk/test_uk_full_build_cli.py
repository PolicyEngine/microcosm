"""The canonical CLI restores declared files and preserves failure/scope semantics."""

# ruff: noqa: F403, F405
import base64
import os
import signal
from types import SimpleNamespace

import test_support.microcosm_build.uk_full_build_cli as support  # noqa: E402
from test_support.microcosm_build.uk_full_build_cli import *


def test_geography_assignment_arguments_are_closed(tmp_path, monkeypatch):
    """Atomic is the default; the cross-flag rules live in the validator."""
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
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
    assert (args.n_clones, args.epochs, args.learning_rate) == (15, 2000, 0.15)
    assert args.target_weight_rule == "nation_grain_family_equal_sqrt_count"
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


def test_dense_role_admits_the_row_metadata_rules_as_candidate_overrides(tmp_path):
    """microcosm#1124: candidate-only rules; release candidates and the national role refuse them."""
    for rule in (
        "grain_family_equal",
        "grain_family_equal_sqrt_count",
        "nation_grain_family_equal",
        "nation_grain_family_equal_sqrt_count",
    ):
        args = arguments(tmp_path, "--target-weight-rule", rule)
        cli.validate_cli_args(args)
        assert args.target_weight_rule == rule
    with pytest.raises(ValueError, match="--release-candidate refuses non-release"):
        cli.validate_cli_args(
            arguments(
                tmp_path,
                "--release-candidate",
                "--target-weight-rule",
                "grain_family_equal",
            )
        )
    national = cli.parse_args(
        _national_argv(tmp_path, "--target-weight-rule", "grain_family_equal")
    )
    with pytest.raises(ValueError, match="--target-weight-rule grain_family_equal"):
        cli.validate_cli_args(national)


def test_size_stage_l2_flags_are_validated_and_recorded_only_when_on(tmp_path):
    """microcosm#1124: candidate-only, size-run-only, refused by the national role."""
    from microcosm.build.uk_runtime.rowwise_cli import rowwise_parameters

    # A sized build takes the doctrine's refit L2 unless it is turned off.
    defaulted = arguments(tmp_path, "--dataset-households", "2")
    cli.validate_cli_args(defaulted)
    assert set(rowwise_parameters(defaulted, source_year=2024)) >= {"refit_l2"}
    plain = arguments(tmp_path, "--dataset-households", "2", "--refit-l2-lambda", "0")
    cli.validate_cli_args(plain)
    assert not {"selection_l2", "refit_l2"} & set(
        rowwise_parameters(plain, source_year=2024)
    )
    on = arguments(
        tmp_path,
        "--dataset-households",
        "2",
        "--refit-l2-lambda",
        "0.03",
        "--refit-l2-anchor",
        "uniform",
        "--selection-l2-lambda",
        "1e-3",
    )
    cli.validate_cli_args(on)
    parameters = rowwise_parameters(on, source_year=2024)
    assert parameters["refit_l2"] == {
        "lambda": 0.03,
        "anchor": "uniform",
        "basis": "chi_square",
    }
    assert parameters["selection_l2"]["lambda"] == 1e-3
    with pytest.raises(
        ValueError, match="--refit-l2-lambda requires --dataset-households"
    ):
        cli.validate_cli_args(arguments(tmp_path, "--refit-l2-lambda", "0.03"))
    with pytest.raises(ValueError, match="without a positive"):
        cli.validate_cli_args(
            arguments(
                tmp_path, "--dataset-households", "2", "--refit-l2-anchor", "uniform"
            )
        )
    with pytest.raises(ValueError, match="finite number"):
        cli.validate_cli_args(
            arguments(
                tmp_path, "--dataset-households", "2", "--selection-l2-lambda", "-1"
            )
        )
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--refit-l2-basis", "record")
    national = cli.parse_args(_national_argv(tmp_path, "--refit-l2-lambda", "0.03"))
    with pytest.raises(ValueError, match="--refit-l2-lambda"):
        cli.validate_cli_args(national)


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
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
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
    from microcosm.graph import (
        ContentStore,
        collect_execution_evidence,
        load_run_evidence,
    )
    from microcosm.graph.orrery import orrery_document_from_schema

    schema = json.loads((out / "graph.schema.json").read_bytes())
    store = ContentStore(out / ".graph-store", create=False)
    overlay = collect_execution_evidence(
        schema,
        runs=load_run_evidence(out / "execution.evidence.json", store=store),
        store=store,
    )
    snapshot = orrery_document_from_schema(schema, execution=overlay)
    assert any(
        json.loads(node["id"]) == ["operation", "uk.full.export.readback"]
        for node in snapshot["nodes"]
    )
    assert [phase["phase"] for phase in overlay["phases"]][-2:] == ["export", "final"]
    assert json.loads((out / "graph.orrery.json").read_bytes()) == snapshot
    assert (out / f"{STEM}.targets.csv").read_text().startswith("name,target_name")
    cold = json.loads((out / cli.MANIFEST_FILENAME).read_text())["execution"]
    assert cold["nodes_total"] > 0 and cold["nodes_reused"] == 0
    assert cold["nodes_computed"] == cold["nodes_total"]
    assert cold["earlier_attempts"] == []
    # An attempt of another request on the same store is counted, not listed.
    attempts = Path(cold["attempt_directory"]).parent
    (attempts / "unrelated").mkdir()
    (attempts / "unrelated" / "request.json").write_text(
        json.dumps({"schema": "other-request", "attempt": {"build_id": "x"}})
    )
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
    # The replay records that it reused the cold attempt's work and names it.
    replay = json.loads((out / cli.MANIFEST_FILENAME).read_text())["execution"]
    assert replay["nodes_reused"] > 0
    assert replay["nodes_total"] == replay["nodes_reused"] + replay["nodes_computed"]
    assert [Path(item["directory"]) for item in replay["earlier_attempts"]] == [
        Path(cold["attempt_directory"])
    ]
    assert replay["earlier_attempts_on_store"] == 2
    assert replay["earlier_attempts_same_request"] == 1
    assert replay["graph_store"] == cold["graph_store"]
    # The output names come from the dense posture and the FRS vintage.
    assert (out / f"{STEM}.h5").is_file()
    assert (out / f"{STEM}.holdout.json").is_file()
    assert (out / f"{STEM}.local_gates.json").is_file()
    assert not (out / "microcosm_uk_2025.h5").exists()


def test_the_dense_gate_replay_signs_a_bound_attempt_the_contract_accepts(
    tmp_path, monkeypatch
):
    """The seven local outcomes of the graph's terminal document become a signed
    battery report bound to the attempt, which the dense contract verifies."""
    from microcosm.build.uk_runtime.graph_national import replay_uk_dense_gate_battery
    from microcosm.data.contract import _check_uk_dense_gate_report

    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    document = json.loads(gate_payload("terminal", release_candidate=True))
    path = tmp_path / f"{STEM}.local_gates.json"
    report = replay_uk_dense_gate_battery(
        document,
        report_path=path,
        release_id="uk-local-candidate-f100-s0-test",
        release_candidate=True,
    )
    assert json.loads(path.read_text()) == report
    assert set(report["gates"]) == set(UK_LOCAL_GATE_SCOPE)
    assert report["posture"] == "local_candidate"
    assert report["shippable"] is True and "signing_error" not in report["attestation"]
    failures: list[str] = []
    _check_uk_dense_gate_report(
        report, failures=failures, attempt_id="uk-local-candidate-f100-s0-test"
    )
    assert failures == []
    # A dev-posture document cannot be replayed as a release candidate.
    with pytest.raises(ValueError, match="another release posture"):
        replay_uk_dense_gate_battery(
            json.loads(gate_payload("terminal")),
            report_path=path,
            release_id="uk-local-candidate-f100-s0-test",
            release_candidate=True,
        )


def test_the_dense_gate_replay_stays_unsigned_without_a_key_or_an_attempt(
    tmp_path, monkeypatch
):
    from microcosm.build.uk_runtime.graph_national import (
        UK_DENSE_UNBOUND_RELEASE_ID,
        replay_uk_dense_gate_battery,
    )
    from microcosm.data.contract import _check_uk_dense_gate_report

    document = json.loads(gate_payload("terminal", release_candidate=True))
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    unbound = replay_uk_dense_gate_battery(
        document,
        report_path=tmp_path / "unbound.local_gates.json",
        release_id=None,
        release_candidate=True,
    )
    assert unbound["release_id"] == UK_DENSE_UNBOUND_RELEASE_ID
    assert unbound["attestation"]["signature"] is None
    assert "No Logbook attempt" in unbound["attestation"]["signing_error"]
    assert unbound["shippable"] is False
    monkeypatch.delenv(SIGNING_KEY_ENV)
    keyless = replay_uk_dense_gate_battery(
        document,
        report_path=tmp_path / "keyless.local_gates.json",
        release_id="uk-local-candidate-f100-s0-test",
        release_candidate=True,
    )
    assert keyless["attestation"]["signature"] is None
    assert SIGNING_KEY_ENV in keyless["attestation"]["signing_error"]
    assert keyless["shippable"] is False and keyless["posture"] == "local_candidate"
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    failures: list[str] = []
    _check_uk_dense_gate_report(keyless, failures=failures)
    assert any("signing error" in line for line in failures)


def test_a_filtered_build_writes_no_local_gate_report(tmp_path, monkeypatch):
    """No local targets, no local fit claim: the replay writes nothing rather
    than filling the excluded local gates in as not applicable."""
    from microcosm.build.uk_runtime.graph_national import replay_uk_dense_gate_battery

    national_only = {
        **support.SELECTION,
        "selector": {"geography_levels": ["country"], "explicit": False},
        "included": [{"name": "count", "period": 2025, "geography_level": "country"}],
    }
    monkeypatch.setattr(support, "SELECTION", national_only)
    path = tmp_path / f"{STEM}.local_gates.json"
    assert (
        replay_uk_dense_gate_battery(
            json.loads(gate_payload("terminal")),
            report_path=path,
            release_id="uk-local-candidate-f100-s0-test",
            release_candidate=False,
        )
        is None
    )
    assert not path.exists()


def test_a_release_candidate_needs_the_signing_key_and_local_targets(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    release = arguments(tmp_path, "--release-candidate")
    with pytest.raises(ValueError, match="needs the UK gate signing key"):
        cli.validate_cli_args(release)
    monkeypatch.setenv(SIGNING_KEY_ENV, base64.b64encode(b"\x05" * 16).decode())
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        cli.validate_cli_args(release)
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    cli.validate_cli_args(release)
    with pytest.raises(ValueError, match="--target-geographies without"):
        cli.validate_cli_args(
            arguments(
                tmp_path,
                "--release-candidate",
                "--target-geographies",
                "country,region",
            )
        )
    cli.validate_cli_args(
        arguments(tmp_path, "--release-candidate", "--target-geographies", "la")
    )


def test_a_withheld_brma_row_projects_over_the_published_rows(tmp_path, monkeypatch):
    """microcosm#1305 re-review: a withheld row reads blank, not NaN.

    The terminal table leaves a withheld BRMA row's figures blank. The
    manifest's error summaries and family fit cover the published rows, the
    solve block counts the withheld one, and the manifest still writes as
    strict JSON (it refuses NaN).
    """
    pytest.importorskip("tables")
    monkeypatch.setattr(support, "WITHHELD_BRMA_ROW", True)
    args = arguments(tmp_path, "--release-candidate")
    assert cli.execute_full_build(prepared(tmp_path, release_candidate=True), args) == 0
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    solve = manifest["solve"]
    assert solve["n_targets"] == 2
    assert solve["model_figures_withheld"] == {
        "targets": 1,
        "totals_basis": "published_rows",
    }
    assert solve["max_abs_relative_error"] == 0.0
    assert solve["median_abs_relative_error"] == 0.0
    assert [row["family"] for row in manifest["fit"]["local_by_family"]] == [
        "census_households"
    ]


def test_a_published_size_receipt_keeps_only_its_allow_list_while_rows_are_withheld():
    """The third review of #1305: an allow-list, not a deny-list.

    Every field a reader of the manifest uses stays; the losses, the refit's
    distances, the search's probes and any field nobody listed read null.
    """
    from microcosm.build.uk_runtime.graph_terminal import _published_size_receipt

    receipt = {
        "requested_households": 2,
        "realized_households": 2,
        "pool_households": 4,
        "protected_carriers": 1,
        "refit_epochs": 10,
        "selection_epochs": 10,
        "selection_l0_lambda": 1e-7,
        "selection_receipt": {"design": "sampford"},
        "selection_search_bracket": {"lower": 1e-10, "upper": 1e-5},
        "dense_loss": 0.2,
        "compact_loss": 0.3,
        "max_target_scaled_change": 0.4,
        "compact_loss_under_dense_weights": 0.35,
        "refit_l2": {"chi_square_distance_to_anchor": 0.01},
        "selection_budget_search": {"probes": [{"loss": 0.5}]},
        "a_field_nobody_listed": 0.6,
    }
    assert _published_size_receipt(receipt, withheld=0) == receipt
    published = _published_size_receipt(receipt, withheld=1)
    kept = {
        "requested_households",
        "realized_households",
        "pool_households",
        "protected_carriers",
        "refit_epochs",
        "selection_epochs",
        "selection_l0_lambda",
        "selection_receipt",
        "selection_search_bracket",
    }
    for key, value in receipt.items():
        assert published[key] == (value if key in kept else None), key
    assert "1 row(s)" in published["withheld_totals_note"]
    assert receipt["dense_loss"] == 0.2
    assert _published_size_receipt(None, withheld=1) is None


def test_dense_run_projects_the_rowwise_candidate_manifest(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path, "--release-candidate")
    assert cli.execute_full_build(prepared(tmp_path, release_candidate=True), args) == 0
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
        2000,
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
    # The rotated holdout is an opt-in diagnostic (microcosm#1301): a release
    # candidate that did not request it records the skip marker.
    assert manifest["parameters"]["skip_holdout"] is True
    assert manifest["fit"]["rotated_holdout"]["skipped"] is True
    assert manifest["diagnostics"]["rotated_holdout"] == {"skipped": True}
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


def test_release_candidate_holdout_is_opt_in(tmp_path, monkeypatch):
    """microcosm#1301: a release candidate validates without the holdout and
    records that it was not requested; --rotated-holdout runs it."""
    from microcosm.build.uk_runtime.rowwise_cli import rowwise_parameters

    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    release = arguments(tmp_path, "--release-candidate")
    cli.validate_cli_args(release)
    assert release.skip_holdout is True
    assert rowwise_parameters(release, source_year=2024)["skip_holdout"] is True
    rotated = arguments(tmp_path, "--release-candidate", "--rotated-holdout")
    cli.validate_cli_args(rotated)
    assert rotated.skip_holdout is False
    assert rowwise_parameters(rotated, source_year=2024)["skip_holdout"] is False
    national = cli.parse_args(_national_argv(tmp_path))
    assert national.skip_holdout is False
    assert rowwise_parameters(national, source_year=2024)["skip_holdout"] is False
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--skip-holdout")


def test_a_requested_holdout_projects_its_measured_folds(tmp_path):
    pytest.importorskip("tables")
    args = arguments(tmp_path, "--release-candidate", "--rotated-holdout")
    build = prepared(tmp_path, release_candidate=True, rotated_holdout=True)
    assert cli.execute_full_build(build, args) == 0
    manifest = json.loads((args.out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["parameters"]["skip_holdout"] is False
    assert manifest["fit"]["rotated_holdout"]["n_folds"] == 5
    assert "skipped" not in manifest["fit"]["rotated_holdout"]
    assert manifest["diagnostics"]["rotated_holdout"]["n_folds"] == 5


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
        lambda requested, *, build_id, run_kind, close_service_on_finish: (
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
        lambda requested, *, build_id, run_kind, close_service_on_finish: (
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
    # The seven local gates resolve in the signed local report; every other
    # gate in the graph's full document, by its outcome position.
    assert row.gate_verdicts and all(
        item["verdict"] == "passed" for item in row.gate_verdicts.values()
    )
    for gate_id, item in row.gate_verdicts.items():
        if gate_id in UK_LOCAL_GATE_SCOPE:
            assert item["receipt"].endswith(f".local_gates.json#/gates/{gate_id}")
        else:
            assert (
                f"{cli.FULL_GATE_REPORT_FILENAME}#/report/outcomes/"
                in (item["receipt"])
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
        "graph_execution",
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


def _staged_epoch_rows(out) -> list[dict]:
    run_id = single_run_id(out)
    path = out / "staging" / "runs" / run_id / "calibration_progress.json"
    rows: list[dict] = []

    def walk(value):
        if isinstance(value, dict):
            if "epoch" in value and "loss" in value:
                rows.append(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json.loads(path.read_text()))
    return rows


@pytest.mark.parametrize(
    "extra,withheld", [((), True), (("--target-geographies", "la"), False)]
)
def test_a_request_that_can_bind_brma_rows_stages_its_epochs_without_loss(
    tmp_path, monkeypatch, capsys, extra, withheld
):
    """The third review of #1305: an epoch's loss covers every row.

    A request that can bind BRMA rows stages its epochs, and sends them to
    the emitter, with a null loss; only the local stderr line keeps it. A
    request that binds none keeps the loss in the staged rows.
    """
    pytest.importorskip("tables")
    emitted: list[dict] = []
    monkeypatch.setattr(
        cli, "emit_calibration_progress", lambda event: emitted.append(dict(event))
    )
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        *extra,
        staging="--staging-local-only",
        on_prepare=_drive_epochs,
    )
    assert status == 0
    rows = _staged_epoch_rows(out)
    assert rows and emitted
    losses = {row["loss"] for row in rows} | {event["loss"] for event in emitted}
    assert (losses == {None}) is withheld
    # The local stderr line keeps the last epoch's loss either way.
    assert "loss 0.25" in capsys.readouterr().err


def test_the_thinned_epochs_null_the_loss_on_request():
    from microcosm.build.uk_runtime.rowwise_staging import thinned_epochs

    seen: list[dict] = []
    forward = thinned_epochs(seen.append, every=1, withhold_loss=True)
    forward({"kind": "calibration_epoch", "epoch": 1, "epochs": 1, "loss": 0.25})
    assert seen == [
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 1, "loss": None}
    ]
    seen.clear()
    thinned_epochs(seen.append, every=1)(
        {"kind": "calibration_epoch", "epoch": 1, "epochs": 1, "loss": 0.25}
    )
    assert seen[0]["loss"] == 0.25


@pytest.mark.parametrize(
    "role,localisation,levels,binds",
    [
        ("dense", "area", None, True),
        ("dense", "area", ("la", "brma"), True),
        ("dense", "area", ("la",), False),
        ("dense", "none", None, False),
        ("national", "area", None, False),
    ],
)
def test_a_request_binds_brma_rows_only_on_area_brmas_it_selects(
    role, localisation, levels, binds
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        release_role=role, brma_localisation=localisation, target_geographies=levels
    )
    assert cli._binds_brma_rows(args) is binds


def test_a_national_request_stages_its_epochs_with_their_loss(tmp_path, monkeypatch):
    """The fourth review of #1305: a national run staged a null loss.

    The national role shares the dense role's parser defaults, but binds no
    BRMA row, so its staged epochs keep their loss.
    """

    args = cli.parse_args(_national_argv(tmp_path))
    assert cli._binds_brma_rows(args) is False
    emitted: list[dict] = []
    monkeypatch.setattr(
        cli, "emit_calibration_progress", lambda event: emitted.append(dict(event))
    )
    observer = cli._solve_observer(args, None)
    observer({"kind": "calibration_epoch", "epoch": 1, "epochs": 1, "loss": 0.25})
    assert [event["loss"] for event in emitted] == [0.25]


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


def test_a_failed_refresh_after_a_passed_read_back_still_records_the_attempt(
    tmp_path, monkeypatch, capsys
):
    """A best-effort refresh that fails after read-back passed fails nothing.

    Read-back's verdict owns the delivery's error code (microcosm#1301), so
    the run closes as built: no failure receipt, the manifest gains its
    verified staging delivery, and the Logbook row is spooled.
    """

    pytest.importorskip("tables")
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.uk_rowwise_candidate import _FakeHub

    class RefreshFailsHub(_FakeHub):
        read_back_started = False

        def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
            if filename.startswith("runs/"):
                self.read_back_started = True
            return super().hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type=repo_type, **kwargs
            )

        def upload_file(self, **kwargs):
            if self.read_back_started:
                raise RuntimeError("503 token=do-not-record")
            super().upload_file(**kwargs)

    hub = RefreshFailsHub()
    monkeypatch.setattr(rowwise_staging, "_hub_api", lambda: hub)
    monkeypatch.setattr(rowwise_staging, "_hub_token", lambda: "hf_test_token")
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-upload-interval-seconds",
        "0",
        "--staging-read-back",
        staging=None,
    )

    assert status == 0
    assert hub.read_back_started
    assert not (out / "failure.json").exists()
    err = capsys.readouterr().err
    assert "staging upload failed" in err and "do-not-record" not in err
    delivery = json.loads((out / cli.MANIFEST_FILENAME).read_text())["staging_delivery"]
    assert delivery["read_back"] == "passed"
    assert delivery["last_error_code"] is None
    assert delivery["upload_attempts"] > delivery["upload_successes"] > 0
    rows = spool_rows(out)
    assert len(rows) == 1 and rows[0].disposition == "iterating"


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
    # The local battery report carries the graph's outcomes for its seven
    # gates, blocked ones included, and is never shippable.
    assert set(report["gates"]) == set(UK_LOCAL_GATE_SCOPE)
    for gate_id in ("uk_local_area_support", "uk_local_weight_ratio"):
        assert report["gates"][gate_id]["criticality"] == "release_blocking"
        assert report["gates"][gate_id]["status"] == "failed"
    assert report["shippable"] is False
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
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
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
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
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
    if outcome == 7:
        # A non-zero return with no recorded block: the close-outs' vocabulary.
        assert events[0]["details"]["error_code"] == "BUILD_REFUSED"
        assert events[0]["details"]["failure_class"] == "refused"
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


def test_an_attempt_names_itself_and_reports_its_graph_reuse(tmp_path, monkeypatch):
    """The attempt's request evidence carries its Logbook and staging ids, and
    the staging run gets the attempt's reuse counts before it closes."""
    pytest.importorskip("tables")
    staging_dir = tmp_path / "staging-bundle"
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--staging-dir",
        str(staging_dir),
        staging="--staging-local-only",
    )
    assert status == 0
    row = spool_rows(out)[0]
    execution = json.loads((out / cli.MANIFEST_FILENAME).read_text())["execution"]
    request = json.loads(
        (Path(execution["attempt_directory"]) / "request.json").read_text()
    )
    bundle = _only_staging_run(staging_dir)
    assert request["attempt"] == {
        "build_id": row.build_id,
        "run_id": bundle["run_manifest"]["run_id"],
    }
    events = [
        event
        for event in bundle["events"]
        if (event["stage_id"], event["status"]) == ("graph_execution", "completed")
    ]
    assert len(events) == 1
    assert events[0]["details"] == {
        key: execution[key] for key in ("nodes_total", "nodes_reused", "nodes_computed")
    }


def test_a_block_the_staging_contract_refuses_closes_both_destinations_failed(
    tmp_path,
):
    """A gate block whose details the staging content policy rejects (a gate id
    that reads as a sensitive key) must not leave the run ``running``: the
    staging run and the hosted emitter both close ``failed`` with the
    unrecorded-block class, and a recordable block still closes both ``blocked``
    with the gate statuses."""
    from microcosm.build.run_outcome import UNRECORDED_GATE_BLOCK, GateBlock
    from microcosm.build.staging_v2 import StagingRunBundleWriterV2
    from microcosm.build.telemetry_emitter import TelemetryRun
    from microcosm.build.uk_runtime import rowwise_staging
    from test_support.microcosm_build.telemetry import FakeTelemetryEmitter

    def bundle(name):
        return StagingRunBundleWriterV2(
            run_id=name,
            country_code="GB",
            operation_id="uk_full_build",
            pipeline_id="uk_local_candidate",
            pipeline_version="2026.10",
            candidate_id=name,
            local_dir=tmp_path / name,
            release_id=None,
            run_kind="smoke",
            delivery_mode="local_only",
            repo_id=None,
        )

    def emitter(name):
        return FakeTelemetryEmitter(
            TelemetryRun(
                run_id=name,
                country_code="GB",
                pipeline="uk_local_candidate",
                candidate_id=name,
                producer_id="producer-a",
            )
        )

    def run_events(fake):
        return [event for event in fake.events if event["event_type"] == "run"]

    refused_bundle, refused_emitter = bundle("refused"), emitter("refused")
    rowwise_staging.close_run_blocked(
        refused_bundle,
        refused_emitter,
        GateBlock.of("terminal", ["token"], gate_statuses={"token": "failed"}),
    )
    failure = refused_bundle.validate_local_bundle()["progress"]["failure"]
    assert (failure["error_code"], failure["failure_class"]) == (
        UNRECORDED_GATE_BLOCK.error_code,
        UNRECORDED_GATE_BLOCK.failure_class,
    )
    events = run_events(refused_emitter)
    assert [event["status"] for event in events] == ["failed"]
    assert events[0]["details"]["error_code"] == UNRECORDED_GATE_BLOCK.error_code

    blocked_bundle, blocked_emitter = bundle("blocked"), emitter("blocked")
    rowwise_staging.close_run_blocked(
        blocked_bundle,
        blocked_emitter,
        GateBlock.of(
            "terminal",
            ["uk_local_target_fit"],
            gate_statuses={"uk_local_target_fit": "failed"},
        ),
    )
    assert blocked_bundle.validate_local_bundle()["progress"]["status"] == "blocked"
    events = run_events(blocked_emitter)
    assert [event["status"] for event in events] == ["blocked"]
    assert events[0]["details"]["gate_statuses"] == {"uk_local_target_fit": "failed"}


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


def test_a_sigterm_closes_the_dense_run_as_terminated_and_exits_143(
    tmp_path, monkeypatch
):
    """A supervisor's SIGTERM is recorded like Ctrl-C (a discarded row, a
    failed staging run) with its own class, and the command exits 143."""
    pytest.importorskip("tables")
    staging_dir = tmp_path / "staging-bundle"
    original_run = cli.run_graph

    def terminated(compiled, **kwargs):
        if "uk.full.gates.calibrated" in {node.id for node in compiled.graph.nodes}:
            os.kill(os.getpid(), signal.SIGTERM)
        return original_run(compiled, **kwargs)

    monkeypatch.setattr(cli, "run_graph", terminated)
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as raised:
        run_dense_main(
            tmp_path,
            monkeypatch,
            "--staging-dir",
            str(staging_dir),
            staging="--staging-local-only",
        )
    assert raised.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    out = tmp_path / "out"
    failure = _only_staging_run(staging_dir)["progress"]["failure"]
    assert (failure["error_code"], failure["failure_class"]) == (
        "TERMINATED",
        "terminated",
    )
    assert json.loads((out / "failure.json").read_text())["error_type"] == (
        "BuildTerminatedError"
    )
    assert spool_rows(out)[0].disposition == "discarded"


def test_a_sigterm_during_the_national_build_records_a_discarded_row(
    tmp_path, monkeypatch
):
    """The national line routes a SIGTERM through its interrupt arm too."""
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    argv = _national_argv(tmp_path)
    monkeypatch.setattr(cli, "preflight_staged_dataset", lambda args: None)
    monkeypatch.setattr(
        cli,
        "prepare_national_build",
        lambda args, *, telemetry=None, attempt=None: "prepared",
    )

    def terminated(prepared, args, *, telemetry=None, attempt=None):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not stop the build")

    monkeypatch.setattr(cli, "execute_national_build", terminated)
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 143
    out = tmp_path / "out"
    assert json.loads((out / "failure.json").read_text())["error_type"] == (
        "BuildTerminatedError"
    )
    rows = load_spool_rows(out / "logbook-spool")
    assert [row.disposition for row in rows] == ["discarded"]


def test_inspection_export_is_candidate_only_and_dense_only(tmp_path):
    args = arguments(tmp_path, "--inspection-export")
    cli.validate_cli_args(args)
    assert args.inspection_export
    assert not arguments(tmp_path).inspection_export
    with pytest.raises(ValueError, match="--inspection-export is candidate-only"):
        cli.validate_cli_args(
            arguments(tmp_path, "--release-candidate", "--inspection-export")
        )
    national = cli.parse_args(_national_argv(tmp_path, "--inspection-export"))
    with pytest.raises(ValueError, match="--inspection-export"):
        cli.validate_cli_args(national)


def test_inspection_export_swaps_only_the_export_node():
    from dataclasses import dataclass, field

    from microcosm.build.uk_runtime import inspection_export
    from microcosm.build.uk_runtime.graph_terminal import UKExportPrepareKernel

    @dataclass(frozen=True)
    class Node:
        id: str
        kernel: str
        params: dict = field(default_factory=dict)

    @dataclass(frozen=True)
    class Graph:
        nodes: tuple

    export = Node(
        "uk.full.export.prepare", UKExportPrepareKernel.ref, {"bindings": "{}"}
    )
    other = Node("uk.full.gates.calibrated", "gate@1")
    swapped = inspection_export.use_inspection_export(Graph((other, export)))
    nodes = {node.id: node for node in swapped.nodes}
    assert nodes["uk.full.gates.calibrated"] is other
    assert nodes["uk.full.export.prepare"].kernel == (
        inspection_export.UKInspectionExportPrepareKernel.ref
    )
    assert nodes["uk.full.export.prepare"].params == {
        "bindings": "{}",
        "inspection_export": True,
    }
    assert export.params == {"bindings": "{}"}
    with pytest.raises(ValueError, match="Expected one"):
        inspection_export.use_inspection_export(Graph((other,)))


def test_inspection_kernel_describes_past_a_gate_block_only_when_requested(monkeypatch):
    from microcosm.build.uk_runtime import inspection_export

    monkeypatch.setattr(inspection_export, "context_frame", lambda context: "frame")
    monkeypatch.setattr(
        inspection_export.graph_terminal,
        "describe_uk_export",
        lambda frame, *, bindings: {"frame": frame, "bindings": bindings},
    )
    kernel = inspection_export.UKInspectionExportPrepareKernel()
    gate = SimpleNamespace(key="g" * 64)

    def context(params):
        return SimpleNamespace(params=params, artifacts={"full_gates": gate})

    result = kernel.run(
        context({"bindings": '{"inspection_export":true}', "inspection_export": True})
    )
    descriptor = json.loads(result.artifacts["export_descriptor"])
    assert descriptor["bindings"]["inspection_export"] is True
    assert descriptor["bindings"]["artifacts"] == {"full_gates": "g" * 64}
    with pytest.raises(ValueError, match="requires inspection_export"):
        kernel.run(context({"bindings": '{"inspection_export":true}'}))
    with pytest.raises(ValueError, match="must record inspection_export"):
        kernel.run(context({"bindings": "{}", "inspection_export": True}))


def test_sized_dense_builds_take_the_doctrine_size_defaults(tmp_path):
    """microcosm#1124: draw ceiling, refit floor and refit L2 default only when sized."""
    from microcosm.build.uk_runtime import local_doctrine as doctrine

    unsized = arguments(tmp_path)
    cli.validate_cli_args(unsized)
    assert (unsized.selection_pi_hi, unsized.baseline_pi_floor) == (1.0, 0.0)
    assert unsized.refit_l2_lambda is None
    assert cli.cli_size_l2(unsized, "refit") is None

    sized = arguments(
        tmp_path, "--dataset-households", str(doctrine.UK_LOCAL_DATASET_HOUSEHOLDS)
    )
    cli.validate_cli_args(sized)
    assert sized.selection_pi_hi == doctrine.UK_LOCAL_SIZE_SELECTION_PI_HI == 0.5
    assert sized.baseline_pi_floor == doctrine.UK_LOCAL_SIZE_BASELINE_PI_FLOOR == 0.5
    l2 = cli.cli_size_l2(sized, "refit")
    assert l2.l2_lambda == doctrine.UK_LOCAL_SIZE_REFIT_L2_LAMBDA
    assert (l2.anchor, l2.basis) == ("uniform", "chi_square")
    assert cli.cli_size_l2(sized, "selection") is None

    explicit = arguments(
        tmp_path,
        "--dataset-households",
        str(doctrine.UK_LOCAL_DATASET_HOUSEHOLDS),
        "--selection-pi-hi",
        "0.95",
        "--baseline-pi-floor",
        "0",
        "--refit-l2-lambda",
        "0",
    )
    cli.validate_cli_args(explicit)
    assert (explicit.selection_pi_hi, explicit.baseline_pi_floor) == (0.95, 0.0)
    assert cli.cli_size_l2(explicit, "refit") is None


def test_the_doctrine_is_k15_2000_epochs_nation_grain():
    from microcosm.build.uk_runtime import local_doctrine as doctrine

    assert doctrine.UK_LOCAL_CLONE_COUNT == 15
    assert doctrine.UK_LOCAL_SOLVE_EPOCHS == 2000
    assert doctrine.UK_LOCAL_SOLVE_DOCTRINE.target_weight_rule == (
        "nation_grain_family_equal_sqrt_count"
    )
    assert doctrine.UK_LOCAL_MAX_WEIGHT_RATIO == 10.0


# ---------------------------------------------------------------------------
# microcosm#1301: the gate clock and the solve-reuse guard of a second build
# ---------------------------------------------------------------------------


def test_the_gate_clock_lies_between_the_solve_clock_and_today(tmp_path):
    from datetime import date, timedelta

    today = date.today()
    solve = (today - timedelta(days=3)).isoformat()
    args = arguments(
        tmp_path, "--review-date", solve, "--gate-review-date", today.isoformat()
    )
    cli.validate_cli_args(args)
    assert cli._gate_review_date(args) == today
    unsplit = arguments(tmp_path, "--review-date", solve)
    cli.validate_cli_args(unsplit)
    assert cli._gate_review_date(unsplit) == date.fromisoformat(solve)
    early = (today - timedelta(days=4)).isoformat()
    with pytest.raises(ValueError, match=f"--gate-review-date {early} precedes"):
        cli.validate_cli_args(
            arguments(tmp_path, "--review-date", solve, "--gate-review-date", early)
        )
    future = (today + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="later than today"):
        cli.validate_cli_args(
            arguments(tmp_path, "--review-date", solve, "--gate-review-date", future)
        )
    # A future solve clock on its own stays admitted: a build that does not
    # split the clocks may still evaluate on a planned sign-off date.
    cli.validate_cli_args(arguments(tmp_path, "--review-date", future))


def _signed(approved_on, *lines, expires_on=None):
    """A synthetic sign-off approved on ``approved_on``, scoped to ``lines``,
    expiring 28 days later unless ``expires_on`` says otherwise."""
    from datetime import timedelta

    from microcosm.build.uk_runtime.weighted_integrity import UKReviewedExclusion

    expires_on = approved_on + timedelta(days=28) if expires_on is None else expires_on
    return UKReviewedExclusion(
        reason="Synthetic sign-off.",
        approved_by="reviewer",
        adjudication="microcosm#1301",
        approved_on=approved_on.isoformat(),
        expires_on=expires_on.isoformat(),
        release_lines=lines or None,
    )


def _measure(name, approved_on, expires_on):
    """A synthetic calibration measure exclusion, as the register loads it."""
    return {
        "name": name,
        "reason": "Synthetic measure exclusion.",
        "tracking": "microcosm#1301",
        "approved_by": "reviewer",
        "adjudication": "microcosm#1301",
        "approved_on": approved_on.isoformat(),
        "expires_on": expires_on.isoformat(),
    }


def _stand_in_registers(
    monkeypatch,
    *,
    target_fit=None,
    local=None,
    input_mass=None,
    areas=None,
    bound=None,
    degenerate=None,
    measures=(),
):
    """Stand synthetic registers in for the committed ones the solve-reuse
    guard reads, so a solve-reuse test never depends on what is signed on the
    day it runs: the gate-only registers, the degenerate register and the
    calibration measure exclusions."""
    from microcosm.build.uk_runtime import (
        measure_simulation,
        terminal_gates,
        weighted_integrity,
    )

    monkeypatch.setattr(
        terminal_gates,
        "uk_default_degenerate_reviewed_exclusions",
        lambda: dict(degenerate or {}),
    )
    monkeypatch.setattr(
        measure_simulation,
        "load_uk_calibration_measure_exclusions",
        lambda path=None: tuple(measures),
    )
    monkeypatch.setattr(
        terminal_gates,
        "uk_default_target_fit_reviewed_exclusions",
        lambda: dict(target_fit or {}),
    )
    monkeypatch.setattr(
        terminal_gates,
        "uk_default_local_target_fit_reviewed_exclusions",
        lambda: dict(local or {}),
    )
    monkeypatch.setattr(
        weighted_integrity,
        "uk_default_input_mass_reviewed_exclusions",
        lambda: dict(input_mass or {}),
    )
    monkeypatch.setattr(
        weighted_integrity,
        "load_uk_local_area_support_exclusion_register",
        lambda source, *, resource="local_area_support_exclusions.json": {
            "exclusions": dict(areas or {}),
            "bound_despite_support_floor": dict(bound or {}),
        },
    )


@pytest.fixture
def unsigned_registers(monkeypatch):
    _stand_in_registers(monkeypatch)


def _build_1_store(tmp_path) -> Path:
    """Build 1's graph store, outside its --out, as the two-build recipe names it."""
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    return store


def test_solve_reuse_is_refused_with_resume_forbid(tmp_path, unsigned_registers):
    store = str(_build_1_store(tmp_path))
    for extra in ((), ("--resume", "require")):
        args = arguments(
            tmp_path, "--require-solve-reuse", "--graph-store", store, *extra
        )
        cli.validate_cli_args(args)
        assert args.require_solve_reuse is True
    assert arguments(tmp_path).require_solve_reuse is False
    with pytest.raises(ValueError, match="--resume forbid"):
        cli.validate_cli_args(
            arguments(
                tmp_path,
                "--require-solve-reuse",
                "--graph-store",
                store,
                "--resume",
                "forbid",
            )
        )


def test_solve_reuse_names_build_1s_graph_store(tmp_path, unsigned_registers):
    """The default store lies under --out, which a second build gives a new
    directory, so it would hold no solve (microcosm#1301)."""
    with pytest.raises(ValueError, match="needs --graph-store naming build 1's"):
        cli.validate_cli_args(arguments(tmp_path, "--require-solve-reuse"))
    missing = tmp_path / "no-such-store"
    with pytest.raises(ValueError, match="is not a directory"):
        cli.validate_cli_args(
            arguments(tmp_path, "--require-solve-reuse", "--graph-store", str(missing))
        )
    assert not missing.exists()
    # Build 1 may still take the default store.
    cli.validate_cli_args(arguments(tmp_path))


def test_solve_reuse_refuses_a_gate_clock_before_an_in_line_sign_off(
    tmp_path, monkeypatch
):
    """A second build left on build 1's clock would replay the solve and then
    fail every sign-off approved since as premature. It is refused before
    anything runs, with the clock that puts them all in force (microcosm#1301)."""
    from datetime import date, timedelta

    from microcosm.build.uk_runtime import rowwise_cli

    solve = date.today() - timedelta(days=10)

    def day(offset):
        return solve + timedelta(days=offset)

    _stand_in_registers(
        monkeypatch,
        target_fit={
            # In force from before the solve clock through day 8.
            "national.in_force@2025": _signed(day(-20)),
            # Scoped to the national line, so the local gates set it aside.
            "national.scoped@2025": _signed(day(9), "national"),
        },
        local={"family/E06999999/metric": _signed(day(3), "local")},
        input_mass={"efrs-post-calibration": {"column": _signed(day(1))}},
        areas={"local_authority/E06999999": _signed(day(2), "local")},
        # The area-support gate window-checks these whatever the line.
        bound={"census_households": _signed(day(5), "national")},
    )
    signoffs = rowwise_cli.gate_only_signoffs()
    assert set(signoffs["target_fit_reviewed_exclusions.json"]) == {
        "national.in_force@2025"
    }
    windows = rowwise_cli.solve_reuse_windows()
    assert rowwise_cli.gate_clock_conflicts(solve, windows) == {
        "premature": {
            "input_mass_reviewed_exclusions.json": {
                "efrs-post-calibration/column": day(1).isoformat()
            },
            "local_area_support_exclusions.json": {
                "bound_despite_support_floor/census_households": day(5).isoformat(),
                "exclusions/local_authority/E06999999": day(2).isoformat(),
            },
            "local_target_fit_reviewed_exclusions.json": {
                "family/E06999999/metric": day(3).isoformat()
            },
        },
        "expired": {},
    }
    reuse = (
        "--require-solve-reuse",
        "--graph-store",
        str(_build_1_store(tmp_path)),
        "--review-date",
        solve.isoformat(),
    )
    with pytest.raises(ValueError) as refused:
        cli.validate_cli_args(arguments(tmp_path, *reuse))
    message = str(refused.value)
    assert f"on {solve.isoformat()} (--review-date; no --gate-review-date" in message
    assert "before 4 in-line sign-off(s) take force" in message
    assert (
        f"Pass --gate-review-date on or after {day(5).isoformat()} and on or "
        f"before {day(8).isoformat()}, the latest usable gate clock, when "
        "target_fit_reviewed_exclusions.json:national.in_force@2025 expires."
    ) in message
    with pytest.raises(ValueError, match=f"on or after {day(5).isoformat()}"):
        cli.validate_cli_args(
            arguments(tmp_path, *reuse, "--gate-review-date", day(4).isoformat())
        )
    cli.validate_cli_args(
        arguments(tmp_path, *reuse, "--gate-review-date", day(5).isoformat())
    )
    # A build that reuses no solve keeps its unsplit clock as before.
    cli.validate_cli_args(arguments(tmp_path, "--review-date", solve.isoformat()))


def _reuse_from(tmp_path, solve):
    """A second build's arguments: build 1's store and solve clock."""
    return (
        "--require-solve-reuse",
        "--graph-store",
        str(_build_1_store(tmp_path)),
        "--review-date",
        solve.isoformat(),
    )


#: One entry per register the solve-reuse guard reads: (stand-in keyword, entry
#: name, release lines, the refusal's label). The area-support gate
#: window-checks its bound families whatever the line.
_GUARDED_ENTRIES = (
    (
        "target_fit",
        "national.lapsing@2025",
        (),
        "target_fit_reviewed_exclusions.json:national.lapsing@2025",
    ),
    (
        "local",
        "family/E06999999/metric",
        ("local",),
        "local_target_fit_reviewed_exclusions.json:family/E06999999/metric",
    ),
    (
        "input_mass",
        "column",
        (),
        "input_mass_reviewed_exclusions.json:efrs-post-calibration/column",
    ),
    (
        "areas",
        "local_authority/E06999999",
        ("local",),
        "local_area_support_exclusions.json:exclusions/local_authority/E06999999",
    ),
    (
        "bound",
        "census_households",
        ("national",),
        "local_area_support_exclusions.json:"
        "bound_despite_support_floor/census_households",
    ),
    (
        "degenerate",
        "household.synthetic_constant",
        ("local",),
        "degenerate_reviewed_exclusions.json:household.synthetic_constant",
    ),
    (
        "measures",
        "obr.synthetic@2025",
        None,
        "calibration_measure_exclusions.json:obr.synthetic@2025",
    ),
)


def _stand_in_one(monkeypatch, register, name, entry):
    """Stand in registers holding ``entry`` alone, under ``name``."""
    if register == "input_mass":
        _stand_in_registers(
            monkeypatch, input_mass={"efrs-post-calibration": {name: entry}}
        )
    elif register == "measures":
        _stand_in_registers(monkeypatch, measures=[entry])
    else:
        _stand_in_registers(monkeypatch, **{register: {name: entry}})


@pytest.mark.parametrize(("register", "name", "lines", "label"), _GUARDED_ENTRIES)
def test_solve_reuse_refuses_a_gate_clock_past_an_expiry_it_depends_on(
    tmp_path, monkeypatch, register, name, lines, label
):
    """An entry in force on build 1's clock may expire before build 2's: the
    calibrated gates would fail it, or assembly refuse an expired measure
    exclusion, after the solve replay. The gate clock is refused before it,
    and the refusal names the entry and the latest usable gate clock, the end
    of the range the sign-off drafter prints (microcosm#1301)."""
    from datetime import date, timedelta

    solve = date.today() - timedelta(days=10)

    def day(offset):
        return solve + timedelta(days=offset)

    entry = (
        _measure(name, day(-5), day(3))
        if register == "measures"
        else _signed(day(-5), *lines, expires_on=day(3))
    )
    _stand_in_one(monkeypatch, register, name, entry)
    reuse = _reuse_from(tmp_path, solve)
    # On build 1's clock and on the expiry itself, the entry is in force.
    for clock in (day(0), day(3)):
        cli.validate_cli_args(
            arguments(tmp_path, *reuse, "--gate-review-date", clock.isoformat())
        )
    with pytest.raises(ValueError) as refused:
        cli.validate_cli_args(
            arguments(tmp_path, *reuse, "--gate-review-date", day(4).isoformat())
        )
    message = str(refused.value)
    assert f"on {day(4).isoformat()} (--gate-review-date)" in message
    assert (
        "after 1 sign-off(s) the release depends on expired "
        f"({label} ({day(3).isoformat()}))"
    ) in message
    assert (
        f"Pass --gate-review-date on or after {solve.isoformat()} and on or "
        f"before {day(3).isoformat()}, the latest usable gate clock, when "
        f"{label} expires."
    ) in message
    solve_scoped = register in ("degenerate", "measures")
    assert ("changes the solve and needs a third build" in message) is solve_scoped


#: The registers whose line rule the guard shares with the calibrated gates
#: (``battery_bindings._in_line``): (stand-in keyword, entry name).
_LINE_SCOPED_ENTRIES = (
    ("target_fit", "national.synthetic@2025"),
    ("local", "family/E06999999/metric"),
    ("input_mass", "column"),
    ("areas", "local_authority/E06999999"),
    ("degenerate", "household.synthetic_constant"),
)


@pytest.mark.parametrize("window", ["premature", "expired"])
@pytest.mark.parametrize(("register", "name"), _LINE_SCOPED_ENTRIES)
def test_the_solve_reuse_guard_holds_each_register_to_the_local_line(
    tmp_path, monkeypatch, register, name, window
):
    """An entry out of its window on the gate clock refuses it only where the
    calibrated gates evaluate it: an entry scoped to the national line stands
    aside, as those gates set it aside, and the same entry on the local line,
    or on every line, refuses the gate clock. Dropping the line rule for any
    register the guard reads fails this test (microcosm#1301)."""
    from datetime import date, timedelta

    solve = date.today() - timedelta(days=10)
    clock = solve + timedelta(days=5)
    approved, expires = (
        (clock + timedelta(days=1), clock + timedelta(days=29))
        if window == "premature"
        else (solve - timedelta(days=20), clock - timedelta(days=1))
    )
    reuse = (*_reuse_from(tmp_path, solve), "--gate-review-date", clock.isoformat())
    _stand_in_one(
        monkeypatch, register, name, _signed(approved, "national", expires_on=expires)
    )
    cli.validate_cli_args(arguments(tmp_path, *reuse))
    phrase = "take force" if window == "premature" else "the release depends on expired"
    for lines in (("local",), ()):
        _stand_in_one(
            monkeypatch, register, name, _signed(approved, *lines, expires_on=expires)
        )
        with pytest.raises(ValueError, match=phrase):
            cli.validate_cli_args(arguments(tmp_path, *reuse))


def test_solve_reuse_names_no_usable_gate_clock_when_the_windows_part(
    tmp_path, monkeypatch
):
    from datetime import date, timedelta

    solve = date.today() - timedelta(days=10)

    def day(offset):
        return solve + timedelta(days=offset)

    _stand_in_registers(
        monkeypatch,
        local={"family/E06999999/metric": _signed(day(5), "local")},
        degenerate={
            "household.synthetic_constant": _signed(
                day(-30), "local", expires_on=day(3)
            )
        },
    )
    with pytest.raises(ValueError) as refused:
        cli.validate_cli_args(
            arguments(
                tmp_path,
                *_reuse_from(tmp_path, solve),
                "--gate-review-date",
                day(4).isoformat(),
            )
        )
    message = str(refused.value)
    assert (
        "before 1 in-line sign-off(s) take force (local_target_fit_reviewed_"
        f"exclusions.json:family/E06999999/metric ({day(5).isoformat()})) and "
        "after 1 sign-off(s) the release depends on expired (degenerate_"
        "reviewed_exclusions.json:household.synthetic_constant "
        f"({day(3).isoformat()}))"
    ) in message
    assert (
        f"No gate clock is usable: the latest approval, {day(5).isoformat()}, "
        f"is after the earliest expiry, {day(3).isoformat()}, when "
        "degenerate_reviewed_exclusions.json:household.synthetic_constant "
        "expires."
    ) in message
    assert "changes the solve and needs a third build" in message


def test_the_solve_reuse_guard_reads_the_measure_exclusions_the_solve_applies(
    tmp_path, monkeypatch
):
    from microcosm.build.uk_runtime import measure_simulation

    _stand_in_registers(monkeypatch)
    read = []
    monkeypatch.setattr(
        measure_simulation,
        "load_uk_calibration_measure_exclusions",
        lambda path=None: read.append(path) or (),
    )
    store = str(_build_1_store(tmp_path))
    cli.validate_cli_args(
        arguments(tmp_path, "--require-solve-reuse", "--graph-store", store)
    )
    override = tmp_path / "measure_exclusions.json"
    cli.validate_cli_args(
        arguments(
            tmp_path,
            "--require-solve-reuse",
            "--graph-store",
            store,
            "--measure-exclusions",
            str(override),
        )
    )
    assert read == [None, override]


@pytest.mark.parametrize(
    "extra",
    [("--gate-review-date", "today"), ("--require-solve-reuse",)],
)
def test_national_role_refuses_the_gate_clock_and_the_reuse_guard(tmp_path, extra):
    from datetime import date

    extra = tuple(date.today().isoformat() if x == "today" else x for x in extra)
    national = cli.parse_args(_national_argv(tmp_path, *extra))
    with pytest.raises(ValueError, match=extra[0]):
        cli.validate_cli_args(national)


def _with_calibrated_params(build, **params):
    """The synthetic build with the calibrated gate node re-parameterised."""
    graph = build.full.graph
    nodes = tuple(
        replace(node, params={**node.params, **params})
        if node.id == "uk.full.gates.calibrated"
        else node
        for node in graph.nodes
    )
    return replace(build, full=replace(build.full, graph=replace(graph, nodes=nodes)))


def test_the_solve_subgraph_is_every_strict_ancestor_of_the_calibrated_gates(
    tmp_path,
):
    graph = prepared(tmp_path, size_households=2).full.graph
    solve = {node.id for node in cli._solve_subgraph(graph).nodes}
    assert solve == {
        node.id for node in cli._through(graph, "uk.full.gates.calibrated").nodes
    } - {"uk.full.gates.calibrated"}
    assert {
        "uk.full.calibrated",
        "uk.full.gates.preflight",
        "uk.full.holdout",
        "uk.full.problem",
    } <= solve
    assert not solve & {"uk.full.export.prepare", "uk.full.gates.calibrated"}


def test_required_solve_reuse_reruns_only_the_calibrated_gates(tmp_path, monkeypatch):
    """A second build at a gate-only sign-off, into a new --out on build 1's
    store: the gates re-run on a later clock while every solve node replays
    from the store, and build 1's bundle stays as it was (microcosm#1301)."""
    pytest.importorskip("tables")
    store = str(_build_1_store(tmp_path))
    first = arguments(tmp_path, "--dataset-households", "2", "--graph-store", store)
    assert cli.execute_full_build(prepared(tmp_path, size_households=2), first) == 0
    marker = (first.out / "build.json").read_bytes()

    def forbidden(*args):
        raise AssertionError("A required solve replay recomputed a solve node")

    for kernel in (Fixture, Holdout, Targets, Population, SizeReceipt):
        monkeypatch.setattr(kernel, "run", forbidden)
    evaluated = []
    original = Evidence.run

    def gates_only(self, context):
        evaluated.append(context.params["phase"])
        return original(self, context)

    monkeypatch.setattr(Evidence, "run", gates_only)
    args = arguments(
        tmp_path,
        "--dataset-households",
        "2",
        "--graph-store",
        store,
        "--require-solve-reuse",
        "--out",
        str(tmp_path / "build-2"),
    )
    second = _with_calibrated_params(
        prepared(tmp_path, size_households=2), review_date="2026-10-16"
    )
    assert cli.execute_full_build(second, args) == 0
    assert evaluated == ["terminal"]
    assert (args.out / "build.json").is_file()
    assert (first.out / "build.json").read_bytes() == marker
    execution = json.loads((args.out / cli.MANIFEST_FILENAME).read_text())["execution"]
    assert execution["nodes_reused"] > 0 and execution["nodes_computed"] > 0


def test_required_solve_reuse_fails_before_the_gates_on_a_store_miss(
    tmp_path, monkeypatch, unsigned_registers
):
    """The preflight is stored but the solve is not: the build stops with a
    SolveReuseError naming what is missing, and neither solves nor gates."""
    build = prepared(tmp_path)
    graph = build.full.graph
    store = _build_1_store(tmp_path)
    cli.run_graph(
        cli.compile_graph(cli._through(graph, "uk.full.gates.preflight")),
        sources=build.sources,
        store=cli.ContentStore(store),
        kernels=build.kernels,
    )

    def forbidden(*args):
        raise AssertionError("A required solve replay computed a node")

    for kernel in (Fixture, Evidence, Holdout, Targets, Population):
        monkeypatch.setattr(kernel, "run", forbidden)
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--require-solve-reuse",
        "--graph-store",
        str(store),
        build=build,
    )
    assert status == 1
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "SolveReuseError"
    assert "'uk.full.holdout'" in failure["message"]
    assert "'uk.full.gates.calibrated'" not in failure["message"]
    assert "target_fit_reviewed_exclusions.json" in failure["message"]
    assert not (out / cli.FULL_GATE_REPORT_FILENAME).exists()


def test_required_solve_reuse_on_an_empty_store_fails_at_the_preflight(
    tmp_path, monkeypatch, unsigned_registers
):
    def forbidden(*args):
        raise AssertionError("A required solve replay computed a node")

    for kernel in (Fixture, Evidence, Holdout, Targets, Population):
        monkeypatch.setattr(kernel, "run", forbidden)
    status, out = run_dense_main(
        tmp_path,
        monkeypatch,
        "--require-solve-reuse",
        "--graph-store",
        str(_build_1_store(tmp_path)),
    )
    assert status == 1
    failure = json.loads((out / "failure.json").read_text())
    assert failure["error_type"] == "SolveReuseError"
    assert "'uk.full.gates.preflight'" in failure["message"]


def _files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "marker", ["build.json", "failure.json", "logbook-spool/row.json"]
)
def test_solve_reuse_refuses_an_out_that_holds_any_attempt(tmp_path, marker):
    """A published bundle, a failed attempt's sidecar or any Logbook spool row
    each refuse a second build's --out: a retry beside a failed attempt would
    spool a second attempt's row there (microcosm#1301)."""
    store = str(_build_1_store(tmp_path))
    args = arguments(tmp_path, "--require-solve-reuse", "--graph-store", store)
    out = Path(args.out)
    # Other files, and a spool without rows, hold no attempt.
    (out / "logbook-spool").mkdir(parents=True)
    (out / "logbook-spool" / "row.json.tmp").write_text("{}")
    (out / "notes.txt").write_text("operator notes")
    cli.refuse_occupied_solve_reuse_output(args)
    (out / marker).write_text("{}")
    with pytest.raises(FileExistsError) as refused:
        cli.refuse_occupied_solve_reuse_output(args)
    assert f"already holds a build attempt ({marker})" in str(refused.value)
    # A build that reuses no solve keeps the occupied-output rules it had.
    cli.refuse_occupied_solve_reuse_output(arguments(tmp_path))


def test_solve_reuse_refuses_a_retry_into_a_failed_attempts_out(
    tmp_path, monkeypatch, unsigned_registers
):
    """A failed second build leaves failure.json and a Logbook row in its
    --out. A retry into it is refused before its attempt opens, so the
    directory keeps one attempt, as --verify and the assembler need."""

    def forbidden(*args):
        raise AssertionError("A required solve replay computed a node")

    for kernel in (Fixture, Evidence, Holdout, Targets, Population):
        monkeypatch.setattr(kernel, "run", forbidden)
    reuse = ("--require-solve-reuse", "--graph-store", str(_build_1_store(tmp_path)))
    status, out = run_dense_main(tmp_path, monkeypatch, *reuse)
    assert status == 1
    assert (out / "failure.json").is_file()
    (row,) = sorted((out / "logbook-spool").glob("*.json"))
    before = _files(out)
    with pytest.raises(FileExistsError) as refused:
        run_dense_main(tmp_path, monkeypatch, *reuse)
    assert (
        f"already holds a build attempt (failure.json, logbook-spool/{row.name})"
        in str(refused.value)
    )
    assert _files(out) == before


def test_solve_reuse_refuses_an_out_that_holds_a_bundle(
    tmp_path, monkeypatch, unsigned_registers
):
    """Build 2 into build 1's --out would replace the bundle its sign-off was
    drafted from. It is refused before the attempt opens and writes nothing
    there; a dry run is not refused, and a new --out runs (microcosm#1301)."""
    pytest.importorskip("tables")
    store = str(_build_1_store(tmp_path))
    status, out = run_dense_main(tmp_path, monkeypatch, "--graph-store", store)
    assert status == 0
    before = _files(out)
    reuse = ("--require-solve-reuse", "--graph-store", store)
    with pytest.raises(
        FileExistsError, match=r"already holds a build attempt \(build\.json"
    ):
        run_dense_main(tmp_path, monkeypatch, *reuse)
    assert _files(out) == before
    assert run_dense_main(tmp_path, monkeypatch, *reuse, "--dry-run")[0] == 0
    assert _files(out) == before
    second = tmp_path / "build-2"
    status, written = run_dense_main(
        tmp_path, monkeypatch, *reuse, "--out", str(second)
    )
    assert (status, written) == (0, second)
    assert (second / "build.json").is_file()
    assert _files(out) == before


def test_a_release_candidate_refuses_the_brma_and_rent_measurement_modes(
    tmp_path, monkeypatch
):
    """microcosm#1305 review: both none modes are for measurement builds only."""
    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    with pytest.raises(ValueError, match="--brma-localisation none"):
        cli.validate_cli_args(
            arguments(tmp_path, "--release-candidate", "--brma-localisation", "none")
        )
    with pytest.raises(ValueError, match="--private-rent-localisation none"):
        cli.validate_cli_args(
            arguments(
                tmp_path, "--release-candidate", "--private-rent-localisation", "none"
            )
        )


def test_target_geographies_can_name_the_brma_rows(tmp_path):
    assert arguments(
        tmp_path, "--target-geographies", "la,brma"
    ).target_geographies == (
        "la",
        "brma",
    )
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--target-geographies", "la,ward")


def test_the_brma_and_rent_resource_flags_need_their_modes(tmp_path):
    """microcosm#1305 review: an override nothing reads is refused, not ignored."""
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="need --brma-localisation area"):
        cli._prepare_brma_localisation(
            SimpleNamespace(
                brma_localisation="none",
                brma_area_layer=other,
                brma_area_layer_sha256=None,
            ),
            {},
            {},
        )
    with pytest.raises(ValueError, match="need --private-rent-localisation brma_list"):
        cli._prepare_private_rent_localisation(
            SimpleNamespace(
                brma_localisation="none",
                private_rent_localisation="none",
                brma_rent_quantiles=None,
                brma_rent_quantiles_sha256="0" * 64,
            ),
            {},
            {},
        )
    # A release candidate builds on the committed bytes only.
    with pytest.raises(ValueError, match="requires the committed BRMA area layer"):
        cli._prepare_brma_localisation(
            SimpleNamespace(
                brma_localisation="area",
                brma_area_layer=other,
                brma_area_layer_sha256=None,
                release_candidate=True,
            ),
            {},
            {},
        )
    with pytest.raises(
        ValueError, match="requires the committed 'uk_brma_rent_quantiles' resource"
    ):
        cli._prepare_private_rent_localisation(
            SimpleNamespace(
                brma_localisation="area",
                private_rent_localisation="brma_list",
                brma_rent_quantiles=other,
                brma_rent_quantiles_sha256=None,
                release_candidate=True,
            ),
            {},
            {},
        )
