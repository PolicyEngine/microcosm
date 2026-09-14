"""Stdlib-only controls for CI selection; never collect or import data tests.

Also runnable directly with an isolated Python interpreter, without pytest or
the workspace conftest, to verify the inventory and workflow partition contract.
"""

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "ci_test_groups", ROOT / "tools/ci_test_groups.py"
)
groups = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(groups)


class ShardSelectionTests(unittest.TestCase):
    def test_round_robin_keeps_whole_files_and_covers_each_once(self):
        files = [f"packages/demo/tests/test_{i:02d}.py" for i in range(17)]
        pieces = [groups.shard_files(reversed(files), i, 4) for i in range(1, 5)]
        self.assertEqual(pieces, [files[i::4] for i in range(4)])
        self.assertEqual(sorted(p for piece in pieces for p in piece), files)
        self.assertEqual(max(map(len, pieces)) - min(map(len, pieces)), 1)

    def test_single_shard_preserves_sorted_inventory(self):
        self.assertEqual(groups.shard_files(["b", "a"], 1, 1), ["a", "b"])

    def test_invalid_indices_counts_duplicates_and_empty_shards_refuse(self):
        for index, count in ((0, 4), (5, 4), (1, 0), (1, 65), (True, 4), (1, 2.0)):
            with self.subTest(index=index, count=count):
                with self.assertRaises(ValueError):
                    groups.shard_files(["a", "b", "c", "d"], index, count)
        for files, count in (([], 1), (["a"], 2), (["a", "a"], 1)):
            with self.subTest(files=files, count=count):
                with self.assertRaises(ValueError):
                    groups.shard_files(files, 1, count)

    def test_cli_rejects_bad_shards_and_incompatible_actions(self):
        for token in ("0/4", "5/4", "1/0", "1/65", "1", "1/2/3", "1.0/4", "01/4"):
            with self.subTest(token=token), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    groups.main(["--list", "rest", "--shard", token])
                self.assertNotEqual(caught.exception.code, 0)
        for action in (["--verify"], ["--procs", "us-am"]):
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    groups.main([*action, "--shard", "1/4"])
                self.assertNotEqual(caught.exception.code, 0)

    def test_empty_selection_emits_no_paths_and_fails(self):
        output = io.StringIO()
        with (
            patch.object(groups, "tracked_test_files", return_value=()),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit):
                groups.main(["--list", "rest", "--shard", "1/4"])
        self.assertEqual(output.getvalue(), "")

    def test_inventory_command_failure_cannot_become_an_empty_success(self):
        output = io.StringIO()
        failure = subprocess.CalledProcessError(7, ["git", "ls-files"])
        with (
            patch.object(groups, "tracked_test_files", side_effect=failure),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                groups.main(["--list", "wheels", "--shard", "1/4"])
        self.assertEqual(output.getvalue(), "")

    def test_wheels_retain_engine_only_files(self):
        files = (*groups.ENGINE_ONLY, "packages/microcosm-fit/tests/test_fit.py")
        with patch.object(groups, "tracked_test_files", return_value=files):
            self.assertEqual(groups.selected_files("wheels"), sorted(files))
            self.assertEqual(groups.selected_files("rest"), [files[-1]])


class WorkflowCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = groups.tracked_test_files()
        cls.workflow = (ROOT / ".github/workflows/test.yml").read_text()

    def job(self, name):
        match = re.search(
            rf"^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:|\Z)",
            self.workflow,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def test_actual_matrix_shards_cover_each_tier_without_process_overlap(self):
        for job, tier in (
            ("fast", groups.FAST_GROUPS),
            ("engine-us", groups.US_GROUPS),
            ("wheels", ("wheels",)),
        ):
            with self.subTest(job=job):
                # Matrix rows are JSON objects (also valid YAML), keeping this
                # coverage proof stdlib-only and tied to the actual workflow.
                rows = [
                    json.loads(row)
                    for row in re.findall(
                        r"^\s+- (\{\"group\".*\})$", self.job(job), re.MULTILINE
                    )
                ]
                expected = [
                    {
                        "group": group,
                        "shard": f"{i}/{groups.SHARD_COUNTS.get(group, 1)}",
                    }
                    for group in tier
                    for i in range(1, groups.SHARD_COUNTS.get(group, 1) + 1)
                ]
                self.assertEqual(rows, expected)
                seen = []
                for row in rows:
                    index, count = groups.parse_shard(row["shard"])
                    for proc in groups.PROCESSES[row["group"]]:
                        selected = groups.selected_files(
                            f"{row['group']}:{proc}", shard=(index, count)
                        )
                        self.assertTrue(selected)
                        self.assertEqual(
                            {groups.process_for(row["group"], p) for p in selected},
                            {proc},
                        )
                        seen.extend(selected)
                if job == "fast":
                    complete = [p for p in self.files if p not in groups.ENGINE_ONLY]
                elif job == "engine-us":
                    complete = [
                        p
                        for p in self.files
                        if groups.engine_group(p) in groups.US_GROUPS
                    ]
                else:
                    complete = list(self.files)
                self.assertEqual(sorted(seen), sorted(complete))
                self.assertEqual(len(seen), len(set(seen)))

    def test_versions_and_us_process_isolation_are_retained(self):
        for job in ("fast", "engine-us", "engine-uk", "wheels"):
            self.assertIn('python-version: ["3.13", "3.14"]', self.job(job))
        self.assertIn('python-version: ["3.13", "3.14.4"]', self.job("engine-shared"))
        self.assertEqual(groups.PROCESSES["us-am"], ("build", "other-shards"))
        us = self.job("engine-us")
        self.assertIn('for proc in "${procs[@]}"; do', us)
        self.assertIn('pytest "${files[@]}"', us)
        self.assertIn("--extra us --extra uk", us)

    def test_selectors_fail_closed_without_process_substitution(self):
        for job in ("fast", "engine-shared", "engine-us", "engine-uk", "wheels"):
            body = self.job(job)
            self.assertNotIn("< <(", body)
            self.assertIn("set -euo pipefail", body)
            self.assertIn('test -n "$selected"', body)
            self.assertIn('mapfile -t files <<< "$selected"', body)
        for job in ("engine-us", "engine-uk"):
            self.assertIn('test -n "$selected_procs"', self.job(job))

    def test_wheel_smokes_extras_and_engine_absence_checks_remain(self):
        wheel = self.job("wheels")
        for text in (
            'uv build --wheel "$shard"',
            "uv venv /tmp/wheels-venv",
            "dist/*.whl pytest",
            "source import escaped wheel venv",
            "tools/spec_envelope_digests.py be uk",
            "microcosm-frame[us] @ file://",
            "microcosm-build[source-io] @ file://",
            'pd.read_hdf(path, key="person")',
            "policyengine-us must be absent from the base wheel gate",
            "env -u PYTHONPATH /tmp/wheels-venv/bin/python -I -m pytest",
        ):
            self.assertIn(text, wheel)
        self.assertIn('--list wheels --shard "${{ matrix.lane.shard }}"', wheel)
        self.assertNotIn("-m requires", wheel)
        self.assertNotIn("--ignore", wheel)


if __name__ == "__main__":
    unittest.main()
