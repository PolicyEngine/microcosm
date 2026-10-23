"""Offline regression tests for the certification orchestration safeguards."""

import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tools import certify_uk_cut_in_policyengine_py as runner

TAG = "microcosm-uk-2024-25-national-20261023T075039Z-a133f37a"


class RunnerTests(unittest.TestCase):
    def test_hugging_face_token_is_explicit_and_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "build": {
                            "cut_tag": TAG,
                            "built_with_model_package": {"version": "2.147.2"},
                            "built_with_core_package": {"version": "3.0"},
                        }
                    }
                )
            )
            download = Mock(return_value=str(manifest))
            with (
                patch.dict(
                    sys.modules,
                    {
                        "huggingface_hub": types.SimpleNamespace(
                            hf_hub_download=download
                        )
                    },
                ),
                patch.dict(os.environ, {"HUGGING_FACE_TOKEN": "private-value"}),
            ):
                self.assertEqual(
                    runner.read_cut(TAG, runner.NATIONAL_RELEASE).model, "2.147.2"
                )
            self.assertEqual(download.call_args.kwargs["token"], "private-value")

    def test_shell_records_failed_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = runner.Shell(root, root / "run.log")
            with (
                patch.object(runner, "resource_guard", return_value={"load1": 1}),
                patch.object(
                    runner,
                    "run_logged",
                    return_value=subprocess.CompletedProcess(["false"], 2, "bad"),
                ),
            ):
                with self.assertRaisesRegex(runner.CertifyError, "exited 2"):
                    shell(["false"], step="failure")
            record = json.loads(shell.timings.read_text())
            self.assertEqual(record["step"], "failure")
            self.assertEqual(record["returncode"], 2)
            self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_guard_prevents_process_and_records_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            shell = runner.Shell(Path(tmp), Path(tmp) / "run.log")
            with (
                patch.object(
                    runner,
                    "resource_guard",
                    side_effect=runner.CertifyError("resource guard"),
                ),
                patch.object(runner, "run_logged") as spawn,
            ):
                with self.assertRaises(runner.CertifyError):
                    shell(["heavy"])
                spawn.assert_not_called()
            self.assertIn(
                "resource guard", json.loads(shell.timings.read_text())["error"]
            )

    def test_output_streams_and_is_retained(self):
        handle = io.StringIO()
        process = Mock()
        process.stdout = iter(["first\n", "second\n"])
        process.wait.return_value = 3
        spawn = Mock()
        spawn.return_value.__enter__ = Mock(return_value=process)
        spawn.return_value.__exit__ = Mock(return_value=False)
        with patch.object(runner.subprocess, "Popen", spawn):
            result = runner.run_logged(["work"], cwd=Path.cwd(), env={}, handle=handle)
        self.assertEqual(handle.getvalue(), "first\nsecond\n")
        self.assertEqual(result.stdout, handle.getvalue())
        self.assertEqual(result.returncode, 3)

    def test_resource_thresholds_are_decimal(self):
        with (
            patch.object(runner.os, "getloadavg", return_value=(110, 1, 1)),
            patch.object(
                runner.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(free=55_000_000_000),
            ),
        ):
            runner.resource_guard(Path.cwd())
        for load, free in ((110.1, 55_000_000_000), (1, 54_999_999_999)):
            with (
                patch.object(runner.os, "getloadavg", return_value=(load, 1, 1)),
                patch.object(
                    runner.shutil,
                    "disk_usage",
                    return_value=types.SimpleNamespace(free=free),
                ),
            ):
                with self.assertRaises(runner.CertifyError):
                    runner.resource_guard(Path.cwd())

    def test_model_surface_is_guarded(self):
        with (
            patch.object(
                runner,
                "resource_guard",
                side_effect=runner.CertifyError("resource guard"),
            ),
            patch.object(runner.subprocess, "run") as spawn,
        ):
            with self.assertRaises(runner.CertifyError):
                runner.model_surface("2.147.2")
            spawn.assert_not_called()

    def test_temp_preflight_rejects_git_directory_or_worktree_file(self):
        for marker_is_directory in (True, False):
            with self.subTest(marker_is_directory=marker_is_directory):
                with tempfile.TemporaryDirectory() as tmp:
                    repository = Path(tmp) / "repository"
                    repository.mkdir()
                    marker = repository / ".git"
                    if marker_is_directory:
                        marker.mkdir()
                    else:
                        marker.write_text("gitdir: /unused/worktree-metadata\n")
                    directory = repository / "private-temp"
                    directory.mkdir()
                    with self.assertRaisesRegex(
                        runner.CertifyError, "Set TMPDIR.*outside every Git"
                    ) as refusal:
                        runner.check_test_temp_directory(directory)
                    self.assertIn(str(repository), str(refusal.exception))

    def test_temp_preflight_resolves_symlinks_before_checking_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp) / "repository"
            repository.mkdir()
            (repository / ".git").mkdir()
            directory = repository / "private-temp"
            directory.mkdir()
            alias = Path(tmp) / "alias"
            alias.symlink_to(directory, target_is_directory=True)
            with self.assertRaisesRegex(runner.CertifyError, "inside Git"):
                runner.check_test_temp_directory(alias)

    def test_temp_preflight_accepts_an_outside_git_location_without_writing(self):
        external = Path("/external-private-temp")
        with (
            patch.object(runner.tempfile, "gettempdir", return_value=str(external)),
            patch.object(Path, "exists", return_value=False),
        ):
            self.assertEqual(runner.check_test_temp_directory(), external)

    def test_temp_preflight_refuses_before_network_or_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse(
                [
                    "--national-tag",
                    TAG,
                    "--drop-regional-release",
                    "--pepy-repo",
                    tmp,
                ]
            )
            with (
                patch.object(
                    runner.subprocess, "check_output", return_value="microcosm"
                ),
                patch.object(
                    runner,
                    "check_test_temp_directory",
                    side_effect=runner.CertifyError("temporary directory inside Git"),
                ),
                patch.object(runner, "read_cut") as manifest,
                patch.object(runner, "model_surface") as surface,
                patch.object(runner.subprocess, "run") as spawn,
            ):
                with self.assertRaisesRegex(runner.CertifyError, "inside Git"):
                    runner.certify(args)
            manifest.assert_not_called()
            surface.assert_not_called()
            spawn.assert_not_called()

    def test_model_surface_pins_core_and_records_intended_pair(self):
        surface = {"all": ["x"], "inputs": ["x"]}
        with tempfile.TemporaryDirectory() as tmp:
            timings = Path(tmp) / "timings.jsonl"
            with (
                patch.object(runner, "resource_guard", return_value={}),
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, json.dumps(surface), ""
                    ),
                ) as spawn,
            ):
                self.assertEqual(
                    runner.model_surface("2.147.2", "3.33.3", timings=timings), surface
                )
            argv = spawn.call_args.args[0]
            self.assertIn("policyengine-uk==2.147.2", argv)
            self.assertIn("policyengine-core==3.33.3", argv)
            record = json.loads(timings.read_text())
            self.assertEqual(record["model_version"], "2.147.2")
            self.assertEqual(record["core_version"], "3.33.3")

    def test_surface_script_reuses_country_registry_without_construction(self):
        constructor = Mock(side_effect=AssertionError("duplicate system construction"))
        country = types.SimpleNamespace(
            variables={
                "input": types.SimpleNamespace(is_input_variable=lambda: True),
                "formula": types.SimpleNamespace(is_input_variable=lambda: False),
            },
            CountryTaxBenefitSystem=constructor,
        )
        output = io.StringIO()
        with (
            patch.dict(sys.modules, {"policyengine_uk": country}),
            redirect_stdout(output),
        ):
            exec(runner._SURFACE_SCRIPT, {})
        constructor.assert_not_called()
        self.assertEqual(
            json.loads(output.getvalue()),
            {"all": ["formula", "input"], "inputs": ["input"]},
        )

    def test_regional_surface_comparison_uses_both_cut_core_versions(self):
        dense_tag = "microcosm-uk-2024-25-dense-20261019T000432Z-ee5c2dbe"
        national = runner.CutVersions(TAG, "2.147.2", "3.33.3", ())
        dense = runner.CutVersions(dense_tag, "2.132.6", "3.29.0", ())
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse(
                [
                    "--national-tag",
                    TAG,
                    "--dense-tag",
                    dense_tag,
                    "--dry-run",
                    "--base-ref",
                    "local",
                    "--pepy-repo",
                    tmp,
                    "--skip-writer-check",
                    "offline regression",
                    "--accept-regional-model-mismatch",
                    "recorded rehearsal mismatch",
                ]
            )
            with (
                patch.object(
                    runner.subprocess,
                    "check_output",
                    side_effect=[
                        "microcosm",
                        "immutable-sha",
                        json.dumps({"data_releases": {"uk": {}}}),
                    ],
                ),
                patch.object(runner, "read_cut", side_effect=[national, dense]),
                patch.object(
                    runner, "model_surface", return_value={"all": [], "inputs": []}
                ) as probe,
            ):
                self.assertEqual(runner.certify(args), 0)
            self.assertEqual(probe.call_args_list[0].args, (dense.model, dense.core))
            self.assertEqual(
                probe.call_args_list[1].args, (national.model, national.core)
            )

    def test_workers_are_limited(self):
        for workers in (0, 5):
            with self.assertRaises(SystemExit):
                runner.parse(
                    [
                        "--national-tag",
                        TAG,
                        "--drop-regional-release",
                        "--test-workers",
                        str(workers),
                    ]
                )
        self.assertEqual(
            runner.parse(
                [
                    "--national-tag",
                    TAG,
                    "--drop-regional-release",
                    "--test-workers",
                    "4",
                ]
            ).test_workers,
            4,
        )

    def test_pr_base_accepts_explicit_stack_target(self):
        common = ["--national-tag", TAG, "--drop-regional-release"]
        self.assertEqual(runner.parse(common).pr_base, "main")
        args = runner.parse(common + ["--pr-base", "budget-local-data-manifest"])
        self.assertEqual(args.pr_base, "budget-local-data-manifest")
        self.assertFalse(args.push)
        self.assertFalse(args.open_pr)

    def test_local_base_is_resolved_once_and_used_for_old_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse(
                [
                    "--national-tag",
                    TAG,
                    "--drop-regional-release",
                    "--dry-run",
                    "--base-ref",
                    "stack-local",
                    "--pepy-repo",
                    tmp,
                    "--skip-writer-check",
                    "offline regression",
                ]
            )
            old = {"data_releases": {"uk": {}}}
            with (
                patch.object(
                    runner.subprocess,
                    "check_output",
                    side_effect=["microcosm\n", "immutable-sha\n", json.dumps(old)],
                ) as read,
                patch.object(runner.subprocess, "run") as spawn,
                patch.object(
                    runner,
                    "read_cut",
                    return_value=runner.CutVersions(TAG, "2.147.2", "3.0", ()),
                ),
            ):
                self.assertEqual(runner.certify(args), 0)
            spawn.assert_not_called()
            self.assertEqual(
                read.call_args_list[1].args[0],
                ["git", "rev-parse", "--verify", "stack-local^{commit}"],
            )
            self.assertEqual(
                read.call_args_list[2].args[0],
                [
                    "git",
                    "show",
                    "immutable-sha:src/policyengine/data/bundle/manifest.json",
                ],
            )

    def test_certification_changes_preserve_auxiliary_untracked_files(self):
        manifest = "src/policyengine/data/bundle/manifest.json"
        changelog = f"changelog.d/certify-uk-{TAG}.changed.md"
        with patch.object(
            runner.subprocess,
            "check_output",
            side_effect=[
                f"pyproject.toml\0{manifest}\0",
                f"{changelog}\0data\0operator-notes.md\0"
                "src/policyengine/data/bundle/operator-private.json\0"
                "tests/fixtures/household_calculator_snapshots/operator-helper.json\0",
            ],
        ):
            paths = runner.certification_changes(Path.cwd(), TAG)
        self.assertEqual(paths, sorted(["pyproject.toml", manifest, changelog]))

    def test_certification_changes_refuse_unrelated_tracked_edits(self):
        with patch.object(
            runner.subprocess,
            "check_output",
            return_value="src/policyengine/core/simulation.py\0",
        ):
            with self.assertRaisesRegex(runner.CertifyError, "unrelated tracked"):
                runner.certification_changes(Path.cwd(), TAG)

    def test_resume_validates_identity_and_preserves_unfinished_generated_files(self):
        for problem in (
            None,
            "branch",
            "registration",
            "owner",
            "head",
            "merge",
            "tracked",
            "cut",
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo, worktree = root / "repo", root / "worktree"
                repo.mkdir()
                worktree.mkdir()
                branch, base = "certify-uk-a133f37a", "b" * 40
                relative = "src/policyengine/data/bundle/manifest.json"
                path = worktree / relative
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "data_releases": {
                                "uk": {
                                    "data_package": {
                                        "release_manifest_revision": "another-cut"
                                        if problem == "cut"
                                        else TAG
                                    }
                                }
                            }
                        }
                    )
                )
                original = path.read_bytes()

                def output(argv, *, cwd, **kwargs):
                    if argv[1:3] == ["worktree", "list"]:
                        location = (
                            root / "elsewhere"
                            if problem == "registration"
                            else worktree
                        )
                        registered_branch = "other" if problem == "branch" else branch
                        return f"worktree {location}\0HEAD {base}\0branch refs/heads/{registered_branch}\0\0"
                    if argv[1:] == ["rev-parse", "--git-common-dir"]:
                        if problem == "owner" and cwd == worktree:
                            return str(root / "other/.git")
                        return str(repo / ".git")
                    if argv[1:] == ["rev-parse", "HEAD"]:
                        return "c" * 40 if problem == "head" else base
                    if argv[1:3] == ["ls-files", "--unmerged"]:
                        return "unmerged" if problem == "merge" else ""
                    if argv[1:3] == ["diff", "--name-only"]:
                        return (
                            "README.md\0" if problem == "tracked" else relative + "\0"
                        )
                    if argv[1:3] == ["ls-files", "--others"]:
                        return "data\0operator-notes.md\0"
                    self.fail(f"unexpected validation command: {argv}")

                with (
                    patch.object(runner.subprocess, "check_output", side_effect=output),
                    patch.object(runner.subprocess, "run") as mutation,
                ):
                    if problem:
                        with self.assertRaises(runner.CertifyError):
                            runner.validate_resume_worktree(
                                repo, worktree, branch, base, TAG
                            )
                    else:
                        self.assertEqual(
                            runner.validate_resume_worktree(
                                repo, worktree, branch, base, TAG
                            ),
                            [relative],
                        )
                    mutation.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def run_mock_pipeline(
        self, *, pytest_code=0, uc_code=0, receipt=None, call_details=None, resume=False
    ):
        if receipt is None and uc_code == 0:
            receipt = {
                "check": "uk_uc_floor",
                "passed": True,
                "measures": [
                    {"name": "benefit_units", "ratio": 0.92524521, "passed": True},
                    {"name": "spend", "ratio": 0.95781282, "passed": True},
                ],
            }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        worktree = root / "certification"
        bundle = {"data_releases": {"uk": {"default_dataset": "candidate"}}}
        if resume:
            path = worktree / "src/policyengine/data/bundle/manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(bundle))
        args = runner.parse(
            [
                "--national-tag",
                TAG,
                "--drop-regional-release",
                "--base-ref",
                "local",
                "--pepy-repo",
                str(root),
                "--worktree",
                str(worktree),
                "--skip-writer-check",
                "offline regression",
                *(["--resume-worktree"] if resume else []),
            ]
        )
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            if argv[:3] == ["git", "worktree", "add"]:
                manifest = worktree / "src/policyengine/data/bundle/manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps(bundle))
            return subprocess.CompletedProcess(argv, 0, "", "")

        def shell(argv, **kwargs):
            calls.append(argv)
            if call_details is not None:
                call_details.append((argv, kwargs))
            if "constant-rewrite" == kwargs.get("step"):
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            if "UC-floor" == kwargs.get("step"):
                if receipt is not None:
                    (root / "certification.uc_floor.json").write_text(
                        json.dumps(receipt)
                    )
                return subprocess.CompletedProcess(argv, uc_code, "traceback", "")
            if "tests" == kwargs.get("step"):
                return subprocess.CompletedProcess(
                    argv, pytest_code, "ERROR: unrecognized arguments: -n", ""
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            patch.object(
                runner.subprocess,
                "check_output",
                side_effect=["microcosm", "immutable-sha", json.dumps(bundle)],
            ),
            patch.object(runner.subprocess, "run", side_effect=command),
            patch.object(
                runner,
                "read_cut",
                return_value=runner.CutVersions(TAG, "2.147.2", "3.0", ()),
            ),
            patch.object(runner.Shell, "__call__", side_effect=shell),
            patch.object(
                runner,
                "check_test_temp_directory",
                return_value=root / "isolated-test-temp",
            ),
            patch.object(runner, "validate_resume_worktree", return_value=[]),
            patch.object(
                runner,
                "certification_changes",
                return_value=["src/policyengine/data/bundle/manifest.json"],
            ),
        ):
            result = runner.certify(args)
        return result, calls, root

    def test_pytest_configuration_error_is_failure_without_failed_lines(self):
        result, calls, root = self.run_mock_pipeline(pytest_code=4)
        self.assertEqual(result, 1)
        self.assertIn(
            "pytest exited 4", (root / "certification.pr-body.md").read_text()
        )
        worktree_call = next(c for c in calls if c[:3] == ["git", "worktree", "add"])
        self.assertEqual(worktree_call[-1], "immutable-sha")

    def test_resume_replays_without_creating_or_resetting_worktree(self):
        result, calls, _ = self.run_mock_pipeline(resume=True)
        self.assertEqual(result, 0)
        self.assertFalse(any(call[:3] == ["git", "worktree", "add"] for call in calls))
        self.assertFalse(any(call[:2] == ["git", "reset"] for call in calls))
        self.assertIn(
            ["git", "add", "--", "src/policyengine/data/bundle/manifest.json"], calls
        )
        self.assertNotIn(["git", "add", "-A"], calls)

    def test_uc_operational_failure_refuses(self):
        with self.assertRaisesRegex(runner.CertifyError, "failed operationally"):
            self.run_mock_pipeline(uc_code=1)

    def test_measured_uc_floor_failure_keeps_human_acceptance(self):
        result, _, root = self.run_mock_pipeline(
            uc_code=1,
            receipt={
                "check": "uk_uc_floor",
                "passed": False,
                "measures": [
                    {"name": "benefit_units", "ratio": 0.8, "passed": False},
                    {"name": "spend", "ratio": 0.95, "passed": True},
                ],
            },
        )
        self.assertEqual(result, 0)
        self.assertIn(
            "UC floor acceptance", (root / "certification.pr-body.md").read_text()
        )
        self.assertIn(
            "FAILED; measured/benchmark ratios: benefit_units=0.800000, spend=0.950000",
            (root / "certification.pr-body.md").read_text(),
        )

    def test_passed_uc_floor_reports_ratios_without_acceptance_request(self):
        result, _, root = self.run_mock_pipeline()
        self.assertEqual(result, 0)
        body = (root / "certification.pr-body.md").read_text()
        self.assertIn(
            "passed; measured/benchmark ratios: benefit_units=0.925245, spend=0.957813",
            body,
        )
        self.assertNotIn("UC floor acceptance", body)

    def test_incomplete_uc_floor_receipt_refuses_even_after_zero_exit(self):
        with self.assertRaisesRegex(runner.CertifyError, "failed operationally"):
            self.run_mock_pipeline(receipt={})

    def test_uc_floor_rejects_contradictory_measure_status(self):
        with self.assertRaisesRegex(runner.CertifyError, "failed operationally"):
            self.run_mock_pipeline(
                receipt={
                    "check": "uk_uc_floor",
                    "passed": True,
                    "measures": [
                        {"name": "benefit_units", "ratio": 0.8, "passed": False},
                        {"name": "spend", "ratio": 0.95, "passed": True},
                    ],
                }
            )

    def test_handoff_marks_wrapper_qualification_pending(self):
        _, _, root = self.run_mock_pipeline()
        body = (root / "certification.pr-body.md").read_text()
        for expected in (
            "Mechanical handoff; wrapper qualification pending",
            "19 producer release-cut gates and passing pytest do not establish",
            "- [ ] **Wrapper qualification.**",
            "full reform contract",
            "metrics review and reform responsiveness",
            "scorecard replay evidence",
            "this script waives no gate",
        ):
            self.assertIn(expected, body)

    def test_country_import_skip_is_scoped_to_constants_and_uc_floor(self):
        details = []
        self.run_mock_pipeline(call_details=details)
        envs = {kwargs.get("step"): kwargs.get("env", {}) for _, kwargs in details}
        skipped = {
            step
            for step, env in envs.items()
            if env.get("POLICYENGINE_SKIP_COUNTRY_IMPORTS") == "1"
        }
        self.assertEqual(skipped, {"constant-rewrite", "UC-floor"})
        self.assertEqual(envs["snapshots"], {"PE_UPDATE_SNAPSHOTS": "1"})
        self.assertEqual(envs["tests"], {})

    def test_stall_diagnostic_is_fullsuite_only_and_does_not_request_exit(self):
        details = []
        self.run_mock_pipeline(call_details=details)
        steps = {kwargs.get("step"): argv for argv, kwargs in details}
        fullsuite = steps["tests"]
        overrides = [
            fullsuite[index + 1]
            for index, option in enumerate(fullsuite)
            if option == "-o"
        ]
        self.assertIn("faulthandler_timeout=300", overrides)
        self.assertFalse(any("exit_on_timeout" in option for option in fullsuite))
        self.assertFalse(any("faulthandler" in option for option in steps["snapshots"]))


if __name__ == "__main__":
    unittest.main()
