"""Offline regression tests for the certification orchestration safeguards."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "certify_runner",
    Path(__file__).resolve().parents[1] / "tools/certify_uk_cut_in_policyengine_py.py",
)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
TAG = "microcosm-uk-2024-25-national-20261023T075039Z-a133f37a"


class RunnerTests(unittest.TestCase):
    def test_hugging_face_token_is_explicit_and_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"build": {
                "cut_tag": TAG,
                "built_with_model_package": {"version": "2.147.2"},
                "built_with_core_package": {"version": "3.0"},
            }}))
            download = Mock(return_value=str(manifest))
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(
                hf_hub_download=download
            )}), patch.dict(os.environ, {"HUGGING_FACE_TOKEN": "private-value"}):
                self.assertEqual(runner.read_cut(TAG, runner.NATIONAL_RELEASE).model,
                                 "2.147.2")
            self.assertEqual(download.call_args.kwargs["token"], "private-value")

    def test_shell_records_failed_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = runner.Shell(root, root / "run.log")
            with patch.object(runner, "resource_guard", return_value={"load1": 1}), \
                 patch.object(runner.subprocess, "run", return_value=
                              subprocess.CompletedProcess(["false"], 2, "bad")):
                with self.assertRaisesRegex(runner.CertifyError, "exited 2"):
                    shell(["false"], step="failure")
            record = json.loads(shell.timings.read_text())
            self.assertEqual(record["step"], "failure")
            self.assertEqual(record["returncode"], 2)
            self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_guard_prevents_process_and_records_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            shell = runner.Shell(Path(tmp), Path(tmp) / "run.log")
            with patch.object(runner, "resource_guard", side_effect=
                              runner.CertifyError("resource guard")), \
                 patch.object(runner.subprocess, "run") as spawn:
                with self.assertRaises(runner.CertifyError):
                    shell(["heavy"])
                spawn.assert_not_called()
            self.assertIn("resource guard", json.loads(shell.timings.read_text())["error"])

    def test_resource_thresholds_are_decimal(self):
        with patch.object(runner.os, "getloadavg", return_value=(110, 1, 1)), \
             patch.object(runner.shutil, "disk_usage", return_value=
                          types.SimpleNamespace(free=55_000_000_000)):
            runner.resource_guard(Path.cwd())
        for load, free in ((110.1, 55_000_000_000), (1, 54_999_999_999)):
            with patch.object(runner.os, "getloadavg", return_value=(load, 1, 1)), \
                 patch.object(runner.shutil, "disk_usage", return_value=
                              types.SimpleNamespace(free=free)):
                with self.assertRaises(runner.CertifyError):
                    runner.resource_guard(Path.cwd())

    def test_model_surface_is_guarded(self):
        with patch.object(runner, "resource_guard", side_effect=
                          runner.CertifyError("resource guard")), \
             patch.object(runner.subprocess, "run") as spawn:
            with self.assertRaises(runner.CertifyError):
                runner.model_surface("2.147.2")
            spawn.assert_not_called()

    def test_workers_are_limited(self):
        for workers in (0, 5):
            with self.assertRaises(SystemExit):
                runner.parse(["--national-tag", TAG, "--drop-regional-release",
                              "--test-workers", str(workers)])
        self.assertEqual(runner.parse([
            "--national-tag", TAG, "--drop-regional-release",
            "--test-workers", "4"
        ]).test_workers, 4)

    def test_local_base_is_resolved_once_and_used_for_old_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse([
                "--national-tag", TAG, "--drop-regional-release", "--dry-run",
                "--base-ref", "stack-local", "--pepy-repo", tmp,
                "--skip-writer-check", "offline regression",
            ])
            old = {"data_releases": {"uk": {}}}
            with patch.object(runner.subprocess, "check_output", side_effect=
                              ["microcosm\n", "immutable-sha\n", json.dumps(old)]) as read, \
                 patch.object(runner.subprocess, "run") as spawn, \
                 patch.object(runner, "read_cut", return_value=
                              runner.CutVersions(TAG, "2.147.2", "3.0", ())):
                self.assertEqual(runner.certify(args), 0)
            spawn.assert_not_called()
            self.assertEqual(read.call_args_list[1].args[0],
                             ["git", "rev-parse", "--verify", "stack-local^{commit}"])
            self.assertEqual(read.call_args_list[2].args[0], ["git", "show",
                "immutable-sha:src/policyengine/data/bundle/manifest.json"])

    def run_mock_pipeline(self, *, pytest_code=0, uc_code=0, receipt=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        worktree = root / "certification"
        bundle = {"data_releases": {"uk": {"default_dataset": "candidate"}}}
        args = runner.parse([
            "--national-tag", TAG, "--drop-regional-release", "--base-ref", "local",
            "--pepy-repo", str(root), "--worktree", str(worktree),
            "--skip-writer-check", "offline regression",
        ])
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
            if "constant-rewrite" == kwargs.get("step"):
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            if "UC-floor" == kwargs.get("step"):
                if receipt is not None:
                    (root / "certification.uc_floor.json").write_text(json.dumps(receipt))
                return subprocess.CompletedProcess(argv, uc_code, "traceback", "")
            if "tests" == kwargs.get("step"):
                return subprocess.CompletedProcess(argv, pytest_code,
                                                    "ERROR: unrecognized arguments: -n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")
        with patch.object(runner.subprocess, "check_output", side_effect=
                          ["microcosm", "immutable-sha", json.dumps(bundle)]), \
             patch.object(runner.subprocess, "run", side_effect=command), \
             patch.object(runner, "read_cut", return_value=
                          runner.CutVersions(TAG, "2.147.2", "3.0", ())), \
             patch.object(runner.Shell, "__call__", side_effect=shell):
            result = runner.certify(args)
        return result, calls, root

    def test_pytest_configuration_error_is_failure_without_failed_lines(self):
        result, calls, root = self.run_mock_pipeline(pytest_code=4)
        self.assertEqual(result, 1)
        self.assertIn("pytest exited 4", (root / "certification.pr-body.md").read_text())
        worktree_call = next(c for c in calls if c[:3] == ["git", "worktree", "add"])
        self.assertEqual(worktree_call[-1], "immutable-sha")

    def test_uc_operational_failure_refuses(self):
        with self.assertRaisesRegex(runner.CertifyError, "failed operationally"):
            self.run_mock_pipeline(uc_code=1)

    def test_measured_uc_floor_failure_keeps_human_acceptance(self):
        result, _, root = self.run_mock_pipeline(uc_code=1, receipt={
            "check": "uk_uc_floor", "passed": False,
            "measures": [{"name": "spend", "passed": False}],
        })
        self.assertEqual(result, 0)
        self.assertIn("UC floor acceptance", (root / "certification.pr-body.md").read_text())


if __name__ == "__main__":
    unittest.main()
