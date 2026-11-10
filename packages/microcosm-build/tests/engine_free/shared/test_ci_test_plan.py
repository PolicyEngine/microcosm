"""Tests for the single declarative CI test plan."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import shlex
import string
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from tools import ci_test_plan


@pytest.mark.parametrize(
    (
        "relative",
        "expected_group",
        "expected_job",
        "expected_country",
        "expected_engine",
        "expected_integration",
    ),
    [
        (
            "engine_free/shared/test_one.py",
            "engine-free-shared",
            "engine-free",
            "shared",
            None,
            False,
        ),
        (
            "engine_free/us/test_one.py",
            "engine-free-us",
            "engine-free",
            "us",
            None,
            False,
        ),
        (
            "engine_free/uk/test_one.py",
            "engine-free-uk",
            "engine-free",
            "uk",
            None,
            False,
        ),
        (
            "engine_contract/us/test_one.py",
            "engine-contract-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine_scenario/us/test_one.py",
            "engine-scenario-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine_workflow/us/test_one.py",
            "engine-workflow-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine/uk/test_one.py",
            "engine-uk",
            "engine-uk",
            "uk",
            "policyengine_uk",
            False,
        ),
        (
            "integration/uk/test_one.py",
            "integration-uk",
            "integration-uk",
            "uk",
            "policyengine_uk",
            True,
        ),
        (
            "installed_emitter/shared/test_one.py",
            "installed-emitter-shared",
            "installed-emitter",
            "shared",
            None,
            True,
        ),
    ],
)
def test_registry_is_the_only_test_directory_and_job_authority(
    tmp_path: Path,
    relative: str,
    expected_group: str,
    expected_job: str,
    expected_country: str,
    expected_engine: str | None,
    expected_integration: bool,
) -> None:
    path = tmp_path / "package" / "tests" / relative

    assert ci_test_plan.group_name_for_test_path(path) == expected_group
    spec = ci_test_plan.TEST_GROUPS[expected_group]
    assert spec.job == expected_job
    assert spec.country == expected_country
    assert spec.engine_module == expected_engine
    assert spec.integration is expected_integration


def test_collection_directories_resolve_through_the_same_registry(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "package" / "tests" / "engine_contract" / "us"

    assert ci_test_plan.group_name_for_collection_path(directory) == (
        "engine-contract-us"
    )


def test_group_resolution_uses_the_innermost_tests_directory() -> None:
    path = Path(
        "/temporary/tests/checkout/packages/example/tests/engine_scenario/us/test_one.py"
    )

    assert ci_test_plan.group_name_for_test_path(path) == "engine-scenario-us"


def test_registry_directories_are_unique() -> None:
    directories = [spec.directory for spec in ci_test_plan.TEST_GROUPS.values()]

    assert len(directories) == len(set(directories))


def test_workflow_jobs_match_registered_test_and_infrastructure_jobs() -> None:
    source = ci_test_plan.WORKFLOW.read_text(encoding="utf-8")

    assert set(ci_test_plan.load_workflow(source)["jobs"]) == {
        *ci_test_plan.JOBS,
        *ci_test_plan.WORKFLOW_INFRASTRUCTURE_JOBS,
    }


@pytest.mark.parametrize(
    "job", ["invented-compatibility-test", "_invented-compatibility-test"]
)
def test_workflow_rejects_a_separate_behavioral_test_job(job: str) -> None:
    steps = "    steps:\n      - run: node verify.mjs\n"
    source = _edited(_WHEELS, f"{_WHEELS}  {job}:\n{steps}")

    assert _document_errors(source) == (
        f"{job}: {_NOT_DECLARED}",
        f"ci: needs must list {job}",
    )


def _other_jobs(source: str) -> tuple[str, ...]:
    jobs = ci_test_plan.load_workflow(source)["jobs"]
    return tuple(name for name in jobs if name != ci_test_plan.AGGREGATE_JOB)


def _document_errors(source: str) -> tuple[str, ...]:
    return ci_test_plan.workflow_document_errors(ci_test_plan.load_workflow(source))


def test_aggregate_job_speaks_for_every_other_workflow_job() -> None:
    source = ci_test_plan.WORKFLOW.read_text(encoding="utf-8")
    jobs = yaml.safe_load(source)["jobs"]

    assert _document_errors(source) == ()
    assert sorted(jobs["ci"]["needs"]) == sorted(_other_jobs(source))
    assert sorted(jobs["ci"]["needs"]) == sorted(ci_test_plan.plan_jobs())


def test_the_selection_decides_only_the_country_engine_jobs() -> None:
    jobs = yaml.safe_load(ci_test_plan.WORKFLOW.read_text(encoding="utf-8"))["jobs"]

    outputs = {job: ci_test_plan.selection_output(job) for job in jobs if job != "ci"}

    assert outputs == {
        "select-countries": None,
        "lint": None,
        "engine-free": None,
        "engine-us": "run_us",
        "engine-uk": "run_uk",
        "integration-uk": "run_uk",
        "wheels": None,
        "installed-emitter": None,
    }
    assert (
        ci_test_plan.selection_condition("engine-us")
        == "needs['select-countries'].outputs.run_us == 'true'"
    )
    assert ci_test_plan.selection_condition("lint") is None
    assert ci_test_plan.selection_output("no-such-job") is None


def test_the_selection_decides_a_job_only_when_every_group_needs_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = ci_test_plan.TestGroup
    monkeypatch.setattr(
        ci_test_plan,
        "TEST_GROUPS",
        {
            "one-engine": group(("a", "us"), "one", "us", engine_module="us_engine"),
            "mixed-engine": group(
                ("b", "us"), "mixed", "us", engine_module="us_engine"
            ),
            "mixed-free": group(("c", "us"), "mixed", "us"),
            "two-us": group(("d", "us"), "two", "us", engine_module="us_engine"),
            "two-uk": group(("e", "uk"), "two", "uk", engine_module="uk_engine"),
        },
    )

    assert ci_test_plan.selection_output("one") == "run_us"
    assert ci_test_plan.selection_output("mixed") is None
    assert ci_test_plan.selection_output("two") is None


def test_plan_refuses_a_job_that_waits_on_an_output_the_selection_never_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ci_test_plan.selection_errors() == ()

    monkeypatch.setattr(ci_test_plan, "select_countries", lambda paths: {"run_us": 1})

    assert ci_test_plan.selection_errors() == (
        "engine-uk: the selection writes no run_uk output",
        "integration-uk: the selection writes no run_uk output",
    )


def test_verify_reports_the_plans_selection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ci_test_plan, "selection_errors", lambda: ("no such output",))
    monkeypatch.setattr(ci_test_plan, "test_files", lambda: (Path(__file__),))

    with pytest.raises(SystemExit) as refused:
        ci_test_plan.verify()

    assert "\n  no such output\n" in str(refused.value)


def test_verify_reports_the_plan_tests_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ci_test_plan, "plan_tests_errors", lambda: ("tests moved",))
    monkeypatch.setattr(ci_test_plan, "test_files", lambda: (Path(__file__),))

    with pytest.raises(SystemExit) as refused:
        ci_test_plan.verify()

    assert "\n  tests moved\n" in str(refused.value)


_PLAN_TESTS_MOVED = (
    "packages/microcosm-build/tests/engine_free/shared/test_ci_test_plan.py: "
    "engine-free must run this file, the plan's own tests, on every pull request, "
    "because the main ruleset reads that job without ci. If the tests moved, "
    "update PLAN_TESTS",
)


def test_engine_free_runs_these_tests() -> None:
    """The ``main`` ruleset names both ``engine-free`` legs, so it reads that
    job without ``ci``. These tests check the code ``ci`` runs, and they stay
    in that job."""

    assert (
        Path(__file__).resolve()
        == (ci_test_plan.ROOT / ci_test_plan.PLAN_TESTS).resolve()
    )
    assert ci_test_plan.selection_output(ci_test_plan.PLAN_TESTS_JOB) is None
    assert ci_test_plan.plan_tests_errors() == ()


@pytest.mark.parametrize(
    "moved",
    [
        {"job": "engine-us"},
        {"engine_module": "policyengine_us"},
        {"integration": True},
    ],
)
def test_plan_refuses_tests_that_engine_free_would_not_run(
    monkeypatch: pytest.MonkeyPatch, moved: dict[str, object]
) -> None:
    group = ci_test_plan.TEST_GROUPS["engine-free-shared"]
    monkeypatch.setitem(
        ci_test_plan.TEST_GROUPS,
        "engine-free-shared",
        dataclasses.replace(group, **moved),
    )

    assert ci_test_plan.plan_tests_errors() == _PLAN_TESTS_MOVED


def test_plan_refuses_tests_that_left_their_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = ci_test_plan.PLAN_TESTS
    monkeypatch.setattr(ci_test_plan, "ROOT", tmp_path)
    monkeypatch.setattr(ci_test_plan, "PACKAGES", tmp_path / "packages")
    for path, stays in (
        (here, True),
        (here.replace("engine_free/shared", "engine_free/uk"), True),
        (here.replace("engine_free/shared", "engine_contract/us"), False),
        (here.replace("engine_free/shared", "engine_free"), False),
        # No job runs a test outside ``packages/*/tests``.
        (here.replace("packages/microcosm-build/tests", "tools/tests"), False),
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("", encoding="utf-8")
        monkeypatch.setattr(ci_test_plan, "PLAN_TESTS", path)
        assert ci_test_plan.plan_tests_errors() == (
            () if stays else (_PLAN_TESTS_MOVED[0].replace(here, path),)
        ), path

    missing = here.replace("test_ci_test_plan.py", "test_no_such_file.py")
    monkeypatch.setattr(ci_test_plan, "PLAN_TESTS", missing)
    assert ci_test_plan.plan_tests_errors() == (
        _PLAN_TESTS_MOVED[0].replace(here, missing),
    )


def test_plan_refuses_tests_in_a_job_the_selection_may_leave_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ci_test_plan, "selection_output", lambda job: "run_us")

    assert ci_test_plan.plan_tests_errors() == _PLAN_TESTS_MOVED


def test_this_file_holds_the_tests_of_what_ci_runs() -> None:
    """``verify`` keeps this file in ``engine-free``. This keeps the tests of
    ``require-results`` in this file, where that job runs them."""

    for name in (
        "test_require_results_fails_a_failed_run_when_ci_runs_it_as_a_script",
        "test_require_results_reads_the_needs_context_for_the_real_workflow",
        "test_require_results_hears_from_every_job_the_workflow_declares",
        "test_aggregate_passes_exactly_when_every_job_passed_or_was_left_out",
        "test_aggregate_rule_holds_for_every_combination_of_results",
    ):
        assert callable(globals().get(name)), name


def test_the_workflow_reports_the_checks_the_main_ruleset_names() -> None:
    """The ``main`` ruleset names checks that run on every pull request, and a
    matrix leg's name holds its value. When a job this test names changes,
    change the ruleset too."""

    jobs = yaml.safe_load(ci_test_plan.WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    checks: set[str] = set()
    for name, job in jobs.items():
        if ci_test_plan.selection_output(name) is not None:
            continue
        matrix = job.get("strategy", {}).get("matrix")
        if matrix is None:
            checks.add(name)
        else:
            assert len(matrix) == 1, f"{name}: name its checks here and in the ruleset"
            (legs,) = matrix.values()
            checks.update(f"{name} ({leg})" for leg in legs)

    assert checks == {
        "ci",
        "engine-free (3.13)",
        "engine-free (3.14)",
        "installed-emitter",
        "lint",
        "select-countries",
        "wheels",
    }, "the main ruleset names these checks: change it with the job that reports them"


def test_lint_verifies_the_plan_and_the_repository_workflow() -> None:
    """``verify`` alone checks that ``engine-free`` runs these tests."""

    lint = yaml.safe_load(ci_test_plan.WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "lint"
    ]
    runs = [step.get("run") for step in lint["steps"]]

    assert "python3 tools/ci_test_plan.py verify" in runs
    assert "uv run --no-sync python tools/ci_test_plan.py verify-workflow" in runs


_AGGREGATE_WORKFLOW = """\
name: Tests
on:
  pull_request:
    branches: [main]
defaults:
  run:
    shell: bash
jobs:
  select-countries:
    outputs:
      run_us: ${{ steps.plan.outputs.run_us }}
      run_uk: ${{ steps.plan.outputs.run_uk }}
    steps:
      - uses: actions/checkout@v4
      - id: plan
        run: python3 tools/ci_test_plan.py select-countries
  lint:
    steps:
      - run: uv run --no-sync ruff check .
  engine-free:
    steps:
      - run: pytest
  engine-us:
    needs: select-countries
    if: needs['select-countries'].outputs.run_us == 'true'
    steps:
      - run: pytest
  engine-uk:
    needs: [select-countries]
    if: needs['select-countries'].outputs.run_uk == 'true'
    steps:
      - run: pytest
  integration-uk:
    needs:
      - select-countries
    if: needs['select-countries'].outputs.run_uk == 'true'
    steps:
      - run: pytest
  wheels:
    steps:
      - run: build
  installed-emitter:
    steps:
      - run: pytest
  ci:
    if: always()
    needs:
      - select-countries
      - lint
      - engine-free
      - engine-us
      - engine-uk
      - integration-uk
      - wheels
      - installed-emitter
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - env:
          NEEDS: ${{ toJSON(needs) }}
        run: python3 tools/ci_test_plan.py require-results
"""
_WHEELS = "  wheels:\n    steps:\n      - run: build\n"
_CI_IF = "  ci:\n    if: always()\n"
_US_IF = "    if: needs['select-countries'].outputs.run_us == 'true'\n"
_CI_CHECKOUT = (
    "    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
)
_RUN_LINE = "        run: python3 tools/ci_test_plan.py require-results\n"
_RUN_STEP = "      - env:\n          NEEDS: ${{ toJSON(needs) }}\n" + _RUN_LINE
_OUTPUTS_BLOCK = (
    "    outputs:\n      run_us: ${{ steps.plan.outputs.run_us }}\n"
    "      run_uk: ${{ steps.plan.outputs.run_uk }}\n"
)
_PLAN_STEP = (
    "      - id: plan\n        run: python3 tools/ci_test_plan.py select-countries\n"
)
_CI_MUST_ALWAYS_RUN = (
    "ci: `if` must read always(), because GitHub reports a skipped required "
    "check as passing"
)
_US_MUST_READ = (
    "engine-us: `if` must read needs['select-countries'].outputs.run_us == 'true'"
)
_CI_STEPS = (
    "ci: steps must be a checkout that sets only `name`, `uses` and `with`, then "
    "one `run` step"
)
_CI_RUN = (
    "ci: its `run` step must set `run` to `python3 tools/ci_test_plan.py "
    "require-results` on one line, `env` to NEEDS: ${{ toJSON(needs) }} alone, and "
    "no other key but `name`"
)
_CI_KEYS = (
    "ci: job may set only if, needs, permissions, runs-on, timeout-minutes, steps"
)
_PLAN_STEP_MUST_RUN = (
    "select-countries: one step with id `plan` must run `python3 "
    "tools/ci_test_plan.py select-countries` on one line"
)
_STEP_IF = (
    "a step sets an `if` other than `always()`, `success()` or `!cancelled()`. "
    "Write `if: always()`, `if: ${{ !cancelled() }}` or no `if`, so the step runs "
    "whenever its job passes"
)
_SHELL = (
    "job sets `defaults` or a step sets `shell`, which can replace the workflow's "
    "`bash -eo pipefail`"
)
_WORKFLOW_DEFAULTS = "defaults:\n  run:\n    shell: bash\n"
_WORKFLOW_SHELL = (
    "workflow: `defaults` must set `run: shell: bash` and nothing else, so every "
    "`run` step runs under `bash -eo pipefail` and a pipe cannot hide a failed "
    "command",
)
_JOB_CONTINUES = (
    "engine-free: job sets `continue-on-error`, which can let the run pass when "
    "the job fails"
)
_OUTPUTS_MUST_MAP = (
    "select-countries: outputs must map each of run_us, run_uk to "
    "`${{ steps.plan.outputs.<its name> }}`"
)
_NO_IF = (
    "lint: job must have no `if`, because the plan runs it on every run and ci "
    "refuses its skip"
)
_NOT_DECLARED = "the test plan doesn't declare this workflow job"


def _edited(old: str, new: str) -> str:
    assert _AGGREGATE_WORKFLOW.count(old) == 1, old
    return _AGGREGATE_WORKFLOW.replace(old, new)


def test_a_complete_aggregate_job_has_no_workflow_errors() -> None:
    assert _document_errors(_AGGREGATE_WORKFLOW) == ()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # ``ci`` must hear from every job.
        ("      - engine-us\n", "", ("ci: needs must list engine-us",)),
        (
            "      - wheels\n",
            "      - wheels\n      - docs\n",
            ("ci: needs lists docs, which the workflow does not declare",),
        ),
        (
            _WHEELS,
            "",
            (
                "wheels: the workflow lacks this job of the test plan",
                "ci: needs lists wheels, which the workflow does not declare",
            ),
        ),
        # A job outside the plan, however YAML spells it.
        (
            _WHEELS,
            _WHEELS + "  deploy:\n    steps:\n      - run: ship\n",
            (f"deploy: {_NOT_DECLARED}", "ci: needs must list deploy"),
        ),
        (
            _WHEELS,
            _WHEELS + '  "deploy": {steps: [{run: ship}]}\n',
            (f"deploy: {_NOT_DECLARED}", "ci: needs must list deploy"),
        ),
        (
            _WHEELS,
            _WHEELS + "  yes:\n    steps:\n      - run: ship\n",
            (f"True: {_NOT_DECLARED}", "ci: needs must list True"),
        ),
        # ``ci`` must run whatever the other jobs report.
        (_CI_IF, "  ci:\n", (_CI_MUST_ALWAYS_RUN,)),
        (_CI_IF, "  ci:\n    if: success()\n", (_CI_MUST_ALWAYS_RUN,)),
        (_CI_IF, "  ci:\n    if: ${{ !cancelled() }}\n", (_CI_MUST_ALWAYS_RUN,)),
        (_CI_IF, "  ci:\n    if: ${{ always() }}\n", (_CI_MUST_ALWAYS_RUN,)),
        (_CI_IF, "  ci:\n    if:\n", (_CI_MUST_ALWAYS_RUN,)),
        (_CI_IF, "  ci:\n    if: true\n", (_CI_MUST_ALWAYS_RUN,)),
        # An ``if`` that continues on the next line, as a label bypass might.
        (_CI_IF, "  ci:\n    if: always()\n      && false\n", (_CI_MUST_ALWAYS_RUN,)),
        (
            _CI_IF,
            "  ci:\n    if: always()\n"
            "      && !contains(github.event.pull_request.labels.*.name, 'skip-ci')\n",
            (_CI_MUST_ALWAYS_RUN,),
        ),
        (_CI_IF, f"  ci:\n    if: always(){chr(0xA0)}\n", (_CI_MUST_ALWAYS_RUN,)),
        # ``ci`` does nothing but hand the results to ``require-results``.
        (_RUN_LINE, _RUN_LINE.replace("\n", " || true\n"), (_CI_RUN,)),
        # A block scalar ends in a newline, so the command no longer matches.
        (
            _RUN_LINE,
            "        run: |\n          python3 tools/ci_test_plan.py require-results\n",
            (_CI_RUN,),
        ),
        (
            "          NEEDS: ${{ toJSON(needs) }}\n",
            "          RESULTS: ${{ toJSON(needs) }}\n",
            (_CI_RUN,),
        ),
        (
            "          NEEDS: ${{ toJSON(needs) }}\n",
            "          NEEDS: ${{ toJSON(needs) }}\n          PYTHONPATH: elsewhere\n",
            (_CI_RUN,),
        ),
        (
            "      - env:\n          NEEDS: ${{ toJSON(needs) }}\n",
            "      -\n",
            (_CI_RUN,),
        ),
        (
            "      - env:\n          NEEDS: ${{ toJSON(needs) }}\n",
            "      - env: NEEDS\n",
            (_CI_RUN,),
        ),
        (_RUN_STEP, _RUN_STEP + "        shell: true {0}\n", (_CI_RUN,)),
        (_RUN_STEP, _RUN_STEP + "        if: false\n", (_CI_RUN,)),
        (_RUN_STEP, _RUN_STEP + "        continue-on-error: true\n", (_CI_RUN,)),
        (_RUN_STEP, "", (_CI_STEPS,)),
        (_RUN_STEP, _RUN_STEP + "      - run: echo done\n", (_CI_STEPS,)),
        (_RUN_STEP, "      - run: echo first\n" + _RUN_STEP, (_CI_STEPS,)),
        (
            _CI_CHECKOUT,
            _CI_CHECKOUT.replace("actions/checkout@v4", "someone/rewrite@v1"),
            (_CI_STEPS,),
        ),
        (_CI_CHECKOUT, _CI_CHECKOUT + "        if: false\n", (_CI_STEPS,)),
        (
            _CI_CHECKOUT + _RUN_STEP,
            "    runs-on: ubuntu-latest\n    steps: checkout\n",
            ("ci: steps must be a list of mappings", _CI_STEPS),
        ),
        # ``ci`` sets no key that would rename its check or change its steps.
        (
            _CI_IF,
            _CI_IF + "    strategy:\n      matrix:\n        leg: [a, b]\n",
            (_CI_KEYS,),
        ),
        (
            _CI_IF,
            _CI_IF + "    defaults:\n      run:\n        shell: true {0}\n",
            (_CI_KEYS,),
        ),
        (_CI_IF, _CI_IF + "    env:\n      PYTHONPATH: elsewhere\n", (_CI_KEYS,)),
        (_CI_IF, _CI_IF + "    continue-on-error: true\n", (_CI_KEYS,)),
        (_CI_IF, _CI_IF + "    name: Required\n", (_CI_KEYS,)),
        # The selection job publishes the outputs its step writes.
        (
            "      run_us: ${{ steps.plan.outputs.run_us }}\n",
            "      run_us: ${{ steps.plan.outputs.run_uk }}\n",
            (_OUTPUTS_MUST_MAP,),
        ),
        ("      run_uk: ${{ steps.plan.outputs.run_uk }}\n", "", (_OUTPUTS_MUST_MAP,)),
        (_OUTPUTS_BLOCK, "    outputs: none\n", (_OUTPUTS_MUST_MAP,)),
        (_OUTPUTS_BLOCK, "", (_OUTPUTS_MUST_MAP,)),
        (_PLAN_STEP, "", (_PLAN_STEP_MUST_RUN,)),
        (
            _PLAN_STEP,
            _PLAN_STEP + "        if: false\n",
            (f"select-countries: {_STEP_IF}",),
        ),
        (
            _PLAN_STEP,
            "      - id: plan\n        run: echo run_us=false >> $GITHUB_OUTPUT\n",
            (_PLAN_STEP_MUST_RUN,),
        ),
        (
            _PLAN_STEP,
            "      - id: plan\n        run: |\n"
            "          python3 tools/ci_test_plan.py select-countries\n",
            (_PLAN_STEP_MUST_RUN,),
        ),
        (_PLAN_STEP, _PLAN_STEP + _PLAN_STEP, (_PLAN_STEP_MUST_RUN,)),
        # Each other job carries the ``if`` the plan gives it and no other.
        (_US_IF, "", (_US_MUST_READ,)),
        (_US_IF, "    if: always()\n", (_US_MUST_READ,)),
        (_US_IF, "    if: github.event_name == 'push'\n", (_US_MUST_READ,)),
        (
            _US_IF,
            "    if: ${{ needs['select-countries'].outputs.run_us == 'true' }}\n",
            (_US_MUST_READ,),
        ),
        (
            _US_IF,
            "    if: needs['select-countries'].outputs.run_us != 'false'\n",
            (_US_MUST_READ,),
        ),
        (
            _US_IF,
            "    if: needs['select-countries'].outputs.run_uk == 'true'\n",
            (_US_MUST_READ,),
        ),
        (_US_IF, _US_IF.rstrip("\n") + " || always()\n", (_US_MUST_READ,)),
        (_US_IF, _US_IF + "      && false\n", (_US_MUST_READ,)),
        ("  lint:\n", "  lint:\n    if: github.event_name == 'push'\n", (_NO_IF,)),
        ("  lint:\n", "  lint:\n    if:\n", (_NO_IF,)),
        (
            "    needs: select-countries\n",
            "    needs: lint\n",
            ("engine-us: needs must list select-countries",),
        ),
        (
            "    needs: select-countries\n",
            "    needs: {job: select-countries}\n",
            (
                "engine-us: needs must be a job name or a list of job names",
                "engine-us: needs must list select-countries",
            ),
        ),
        # A job that needs a job the selection may leave out skips with it.
        (
            "  wheels:\n",
            "  wheels:\n    needs: engine-uk\n",
            (
                "wheels: needs engine-uk, which the selection may leave out when "
                "it runs wheels, and GitHub would then skip wheels too",
            ),
        ),
        (
            "    needs: select-countries\n",
            "    needs: [select-countries, integration-uk]\n",
            (
                "engine-us: needs integration-uk, which the selection may leave "
                "out when it runs engine-us, and GitHub would then skip "
                "engine-us too",
            ),
        ),
        # Keys and steps that would hide a failure or rename a check.
        (
            "  engine-free:\n",
            "  engine-free:\n    continue-on-error: true\n",
            (_JOB_CONTINUES,),
        ),
        (
            "  engine-free:\n",
            "  engine-free:\n    continue-on-error: ${{ matrix.experimental }}\n",
            (_JOB_CONTINUES,),
        ),
        # PyYAML reads ``no`` as false, and YAML 1.2 reads a string.
        (
            "  engine-free:\n",
            "  engine-free:\n    continue-on-error: no\n",
            (_JOB_CONTINUES,),
        ),
        (
            "  engine-free:\n",
            '  engine-free:\n    "name": ci\n',
            (
                "engine-free: job sets `name`, which would report the job under "
                "another check name",
            ),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        continue-on-error: true\n",
            (
                "wheels: a step sets `continue-on-error`, which can let the job "
                "pass when the step fails",
            ),
        ),
        # A step that can skip leaves its job passing with its tests unrun.
        (
            "  engine-free:\n    steps:\n      - run: pytest\n",
            "  engine-free:\n    steps:\n      - run: pytest\n"
            "        if: github.event_name == 'push'\n",
            (f"engine-free: {_STEP_IF}",),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        if: failure()\n",
            (f"wheels: {_STEP_IF}",),
        ),
        (
            _US_IF + "    steps:\n      - run: pytest\n",
            _US_IF + "    steps:\n      - run: pytest\n"
            "        if: github.event_name == 'push'\n",
            (f"engine-us: {_STEP_IF}",),
        ),
        # A step inside a group of steps counts too.
        (
            "      - run: build\n",
            "      - parallel:\n          - run: build\n"
            "            if: github.event_name == 'push'\n",
            (f"wheels: {_STEP_IF}",),
        ),
        (
            "      - run: build\n",
            "      - parallel:\n          - run: build\n"
            "            continue-on-error: true\n",
            (
                "wheels: a step sets `continue-on-error`, which can let the job "
                "pass when the step fails",
            ),
        ),
        # PyYAML reads ``no`` as false, and YAML 1.2 reads a string.
        (
            "      - run: build\n",
            "      - run: build\n        continue-on-error: no\n",
            (
                "wheels: a step sets `continue-on-error`, which can let the job "
                "pass when the step fails",
            ),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        if: ${{ failure() }}\n",
            (f"wheels: {_STEP_IF}",),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        if: false\n",
            (f"wheels: {_STEP_IF}",),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        if: cancelled()\n",
            (f"wheels: {_STEP_IF}",),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        if: always() && github.event_name == 'push'\n",
            (f"wheels: {_STEP_IF}",),
        ),
        # Every ``run`` step runs under ``bash -eo pipefail``, where a pipe
        # cannot hide the exit status of the command that feeds it.
        (_WORKFLOW_DEFAULTS, "", _WORKFLOW_SHELL),
        (_WORKFLOW_DEFAULTS, "defaults:\n  run:\n    shell: sh\n", _WORKFLOW_SHELL),
        (
            _WORKFLOW_DEFAULTS,
            _WORKFLOW_DEFAULTS + "    working-directory: tools\n",
            _WORKFLOW_SHELL,
        ),
        (
            "  wheels:\n",
            "  wheels:\n    defaults:\n      run:\n        shell: sh\n",
            (f"wheels: {_SHELL}",),
        ),
        (
            "      - run: build\n",
            "      - run: build\n        shell: true {0}\n",
            (f"wheels: {_SHELL}",),
        ),
        (
            "      - run: build\n",
            "      - parallel:\n          - run: build\n            shell: sh\n",
            (f"wheels: {_SHELL}",),
        ),
        (_WHEELS, "  wheels: build\n", ("wheels: job must be a mapping",)),
        (
            _WHEELS,
            "  wheels:\n    steps:\n      - build\n",
            ("wheels: steps must be a list of mappings",),
        ),
    ],
)
def test_workflow_rejects_a_job_whose_failure_could_pass_ci(
    old: str, new: str, expected: tuple[str, ...]
) -> None:
    assert _document_errors(_edited(old, new)) == expected


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # Other spellings of the same YAML.
        ("  wheels:\n", '  "wheels":\n'),
        ("  wheels:\n", "  wheels: # builds every wheel\n"),
        ("jobs:\n", "jobs: # every job of the plan\n"),
        ("jobs:\n", '"jobs":\n'),
        (_CI_IF, '  ci:\n    if: "always()"\n'),
        (_CI_IF, "  ci:\n    if: always() # even when a job fails\n"),
        (_CI_IF, "  ci:\n    if: >-\n      always()\n"),
        (_CI_IF, "  ci:\n    timeout-minutes: 5\n    if: always()\n"),
        (
            "    needs: select-countries\n" + _US_IF,
            _US_IF + "    needs: [select-countries, lint] # selection\n",
        ),
        ("      - wheels\n", "      - 'wheels' # archives\n"),
        ("  lint:\n", "  lint: &lint\n"),
        # A job may need jobs that run whenever it runs.
        ("  wheels:\n", "  wheels:\n    needs: [lint, engine-free]\n"),
        ("  lint:\n", "  lint:\n    needs: wheels\n"),
        (
            "    needs: [select-countries]\n",
            "    needs: [select-countries, integration-uk]\n",
        ),
        # Keys and steps a job may carry.
        (
            "  engine-free:\n    steps:\n",
            "  engine-free:\n    runs-on: ubuntu-latest\n    timeout-minutes: 15\n"
            "    permissions:\n      contents: read\n"
            "    env:\n      NO: a\n      YES: b\n    strategy:\n"
            "      fail-fast: false\n      matrix:\n"
            '        python-version: ["3.13", "3.14"]\n    steps:\n',
        ),
        (
            "      - run: build\n",
            "      - name: Build\n        if: ${{ !cancelled() }}\n"
            "        run: |\n          echo \"it's [not closed\n          jobs:\n",
        ),
        (
            _CI_CHECKOUT,
            "    runs-on: ubuntu-latest\n    permissions:\n      contents: read\n"
            "    steps:\n      - name: Check out\n        uses: actions/checkout@v5\n"
            "        with:\n          persist-credentials: false\n",
        ),
        (
            _RUN_STEP,
            "      - name: Require every job to pass\n        " + _RUN_STEP[8:],
        ),
        # A step may carry a condition that still runs it in a passing job.
        ("      - run: build\n", "      - run: build\n        if: always()\n"),
        ("      - run: build\n", "      - run: build\n        if: ${{ always() }}\n"),
        ("      - run: build\n", "      - run: build\n        if: success()\n"),
        ("      - run: build\n", '      - run: build\n        if: "!cancelled()"\n'),
        # A pipe can't hide a failure under the workflow's ``bash -eo pipefail``.
        ("      - run: build\n", "      - run: build | tee build.log\n"),
        # An action may take an input that shares a step key's name.
        (
            "      - run: build\n",
            "      - uses: actions/setup-node@v7\n        with:\n"
            "          shell: sh\n          if: never\n"
            "          continue-on-error: true\n      - run: build\n",
        ),
        # The selection job may publish more, and its step may carry more.
        (
            _PLAN_STEP,
            "      - name: Select country tests\n        id: plan\n        env:\n"
            "          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: python3 tools/ci_test_plan.py select-countries\n"
            "      - run: echo done\n",
        ),
        (
            "      run_uk: ${{ steps.plan.outputs.run_uk }}\n",
            "      run_uk: ${{ steps.plan.outputs.run_uk }}\n"
            "      docs_only: ${{ steps.plan.outputs.docs_only }}\n",
        ),
    ],
)
def test_workflow_accepts_other_spellings_and_ordinary_edits(
    old: str, new: str
) -> None:
    assert _document_errors(_edited(old, new)) == ()


@pytest.mark.parametrize(
    "document",
    [None, "jobs", [], {}, {"jobs": None}, {"jobs": ["lint"]}, {"on": "push"}],
)
def test_workflow_rejects_a_document_with_no_jobs_mapping(document: object) -> None:
    assert ci_test_plan.workflow_document_errors(document) == (
        "workflow has no jobs mapping",
    )


@pytest.mark.parametrize(
    ("old", "new", "refusal"),
    [
        # PyYAML alone would keep the second ``if`` and say nothing.
        (
            _CI_IF,
            "  ci:\n    if: false\n    if: always()\n",
            "a mapping sets `if` twice",
        ),
        (
            _CI_IF,
            '  ci:\n    "if": false\n    if: always()\n',
            "a mapping sets `if` twice",
        ),
        (
            _CI_IF,
            '  ci:\n    if: false\n    "if": always()\n',
            "a mapping sets `if` twice",
        ),
        (_WHEELS, _WHEELS + _WHEELS, "a mapping sets `wheels` twice"),
        ("jobs:\n", "jobs: {}\njobs:\n", "a mapping sets `jobs` twice"),
        # PyYAML alone would build ``False`` from both keys and keep the last.
        (
            "  wheels:\n",
            "  wheels:\n    env:\n      NO: a\n      OFF: b\n",
            "PyYAML reads `NO` and `OFF` as one key",
        ),
        (
            _WHEELS,
            _WHEELS + "  yes:\n    steps: []\n  on:\n    steps: []\n",
            "PyYAML reads `yes` and `on` as one key",
        ),
        # PyYAML alone would merge the anchored mapping into ``ci``.
        (
            _CI_IF,
            "  ci:\n    <<: *lint\n    if: always()\n",
            "a mapping uses a merge key (`<<`)",
        ),
    ],
)
def test_load_workflow_refuses_what_github_may_read_differently(
    old: str, new: str, refusal: str
) -> None:
    source = _edited(old, new).replace("  lint:\n", "  lint: &lint\n")

    assert isinstance(yaml.safe_load(source), dict)
    with pytest.raises(ValueError) as refused:
        ci_test_plan.load_workflow(source)
    assert str(refused.value).startswith(f"workflow fails to load: {refusal}\n")


@pytest.mark.parametrize("code", [0x85, 0x2028, 0x2029])
def test_load_workflow_names_a_line_break_that_only_yaml_1_1_reads(code: int) -> None:
    """PyYAML alone would end the line here and read the rest as a comment."""

    source = _edited(_CI_IF, f"  ci:\n    if: always(){chr(code)}    # && false\n")
    line = source[: source.index(chr(code))].count("\n") + 1

    assert yaml.safe_load(source)["jobs"]["ci"]["if"] == "always()"
    with pytest.raises(ValueError) as refused:
        ci_test_plan.load_workflow(source)
    assert str(refused.value) == (
        f"workflow line {line} holds U+{code:04X}, which PyYAML reads as a line "
        "break and YAML 1.2 does not"
    )


@pytest.mark.parametrize(
    ("source", "refusal"),
    [
        ("jobs:\n  lint: [\n", "while parsing a flow"),
        (
            "jobs:\n  ? [lint, wheels]\n  : {}\n",
            "a mapping uses a list or a mapping as a key",
        ),
    ],
)
def test_load_workflow_refuses_text_a_yaml_parser_refuses(
    source: str, refusal: str
) -> None:
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(source)
    with pytest.raises(ValueError) as refused:
        ci_test_plan.load_workflow(source)
    assert str(refused.value).startswith(f"workflow fails to load: {refusal}")


_ANCHORED = """\
defaults: &defaults
  runs-on: ubuntu-latest
  env:
    NO: a
    YES: b
jobs:
  lint: *defaults
  wheels: *defaults
"""


def test_load_workflow_reads_what_a_yaml_parser_reads() -> None:
    """Differential: the stricter loader against PyYAML's own."""

    for source in (
        _ANCHORED,
        ci_test_plan.WORKFLOW.read_text(encoding="utf-8"),
        _AGGREGATE_WORKFLOW,
    ):
        assert ci_test_plan.load_workflow(source) == yaml.safe_load(source)


_VALID = yaml.safe_load(_AGGREGATE_WORKFLOW)["jobs"]
_JOB_NAMES = (*_VALID, "deploy")
_COUNTRY = {"engine-us": "us", "engine-uk": "uk", "integration-uk": "uk"}
_IFS = st.sampled_from(
    [
        None,
        "always()",
        "success()",
        "needs['select-countries'].outputs.run_us == 'true'",
        "needs['select-countries'].outputs.run_uk == 'true'",
    ]
)
_NEEDS = st.one_of(
    st.none(),
    st.sampled_from(["select-countries", "lint"]),
    st.lists(st.sampled_from(_JOB_NAMES), unique=True, max_size=9),
)
_STEPS = st.sampled_from(
    [
        *(job["steps"] for job in _VALID.values()),
        [{"run": "pytest", "continue-on-error": False}],
        [{"run": "pytest", "if": "always()"}],
        [{"run": "pytest", "if": "failure()"}],
        [{"run": "pytest", "if": "${{ !cancelled() }}"}],
        [{"run": "pytest", "shell": "sh"}],
        [{"parallel": [{"run": "pytest"}, {"run": "lint", "if": "always()"}]}],
        [{"parallel": [{"run": "pytest", "continue-on-error": True}]}],
        [_VALID["ci"]["steps"][1]],
        [*_VALID["ci"]["steps"], {"run": "echo done"}],
        [*_VALID["select-countries"]["steps"], {"id": "plan", "run": "echo"}],
        [{**_VALID["select-countries"]["steps"][1], "if": "always()"}],
        [{"id": "plan", "run": "echo run_us=false"}],
        [],
    ]
)
_SELECTION_OUTPUTS = st.sampled_from(
    [
        None,
        _VALID["select-countries"]["outputs"],
        {**_VALID["select-countries"]["outputs"], "docs_only": "true"},
        {"run_us": "${{ steps.plan.outputs.run_uk }}", "run_uk": "true"},
    ]
)


@st.composite
def _job(draw: st.DrawFn) -> dict[str, object]:
    """Draw a job from parts, most of which break a rule."""

    job: dict[str, object] = {"steps": draw(_STEPS)}
    for key, value in (("if", draw(_IFS)), ("needs", draw(_NEEDS))):
        if value is not None:
            job[key] = value
    if draw(st.integers(min_value=0, max_value=2)) == 0:
        job["outputs"] = draw(_SELECTION_OUTPUTS)
    if draw(st.integers(min_value=0, max_value=9)) == 0:
        key = draw(
            st.sampled_from(["continue-on-error", "name", "strategy", "defaults"])
        )
        job[key] = "x"
    return job


@st.composite
def _neutral(draw: st.DrawFn, name: str) -> dict[str, object]:
    """Change the test workflow's job in ways no rule reads."""

    job = dict(_VALID[name])
    if draw(st.booleans()):
        job["timeout-minutes"] = 5
    if draw(st.booleans()):
        job["permissions"] = {"contents": "read"}
    steps = [dict(step) for step in job["steps"]]
    if draw(st.booleans()):
        steps = [{"name": "A step", **step} for step in steps]
    if draw(st.booleans()):
        steps = [
            {**step, "uses": "actions/checkout@v5", "with": {"fetch-depth": 1}}
            if "uses" in step
            else step
            for step in steps
        ]
    if name != "ci" and draw(st.booleans()):
        steps.append({"run": "echo done", "if": "always()"})
    job["steps"] = steps
    if "needs" in job and draw(st.booleans()):
        job["needs"] = list(reversed(_needed(job)))
    if name == "select-countries" and draw(st.booleans()):
        job["outputs"] = {**job["outputs"], "docs_only": "true"}
    return job


def _needed(job: dict[str, object]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _may_hide_a_failure(step: dict[str, object]) -> bool:
    runs_while_passing = (
        "always()",
        "success()",
        "!cancelled()",
        "${{ !cancelled() }}",
    )
    return (
        "continue-on-error" in step
        or "shell" in step
        or step.get("if", "success()") not in runs_while_passing
        or any(map(_may_hide_a_failure, step.get("parallel", [])))
    )


def _ci_passes(job: dict[str, object], others: set[str]) -> bool:
    steps = job["steps"]
    if job.get("if") != "always()" or set(_needed(job)) != others:
        return False
    allowed = {"if", "needs", "permissions", "runs-on", "timeout-minutes", "steps"}
    if not set(job) <= allowed or len(steps) != 2:
        return False
    checkout, run = steps
    return (
        str(checkout.get("uses")).startswith("actions/checkout@")
        and set(checkout) <= {"name", "uses", "with"}
        and run.get("run") == "python3 tools/ci_test_plan.py require-results"
        and run.get("env") == {"NEEDS": "${{ toJSON(needs) }}"}
        and set(run) <= {"name", "env", "run"}
    )


def _lets_ci_speak(jobs: dict[str, dict[str, object]]) -> bool:
    """The rule for a workflow, stated independently of the implementation."""

    if set(jobs) != set(_VALID):
        return False
    for name, job in jobs.items():
        steps = job["steps"]
        if name == "ci":
            if not _ci_passes(job, set(jobs) - {"ci"}):
                return False
            continue
        if "continue-on-error" in job or "name" in job or "defaults" in job:
            return False
        if any(map(_may_hide_a_failure, steps)):
            return False
        wanted = (
            f"needs['select-countries'].outputs.run_{_COUNTRY[name]} == 'true'"
            if name in _COUNTRY
            else None
        )
        if job.get("if") != wanted or ("if" in job) != (name in _COUNTRY):
            return False
        if name in _COUNTRY and "select-countries" not in _needed(job):
            return False
        if any(
            _COUNTRY.get(needed, _COUNTRY.get(name)) != _COUNTRY.get(name)
            for needed in _needed(job)
        ):
            return False
        if name == "select-countries":
            outputs = job.get("outputs")
            plans = [step for step in steps if step.get("id") == "plan"]
            if (
                not isinstance(outputs, dict)
                or outputs.get("run_us") != "${{ steps.plan.outputs.run_us }}"
                or outputs.get("run_uk") != "${{ steps.plan.outputs.run_uk }}"
                or len(plans) != 1
                or plans[0].get("run")
                != "python3 tools/ci_test_plan.py select-countries"
            ):
                return False
    return True


@settings(
    max_examples=800,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(data=st.data())
def test_workflow_passes_exactly_when_it_lets_ci_speak_for_every_job(
    data: st.DataObject,
) -> None:
    """Invariant: no errors iff the workflow declares the plan's jobs, none
    hides a failure, ``ci`` always runs, needs every other job and only
    requires their results, the selection job publishes what its step writes,
    and each other job skips only as the plan says."""

    jobs = {
        name: data.draw(_job())
        for name in _JOB_NAMES
        if data.draw(st.integers(min_value=0, max_value=19)) != 0
    }
    # Most draws start from a workflow that passes, so both outcomes occur.
    if data.draw(st.integers(min_value=0, max_value=3)) != 0:
        jobs = {name: dict(job) for name, job in _VALID.items()}
        for name in data.draw(st.lists(st.sampled_from(_JOB_NAMES), max_size=3)):
            edit = data.draw(
                st.sampled_from(["vary", "vary", "replace", "remove", "if", "needs"])
            )
            if edit == "remove":
                jobs.pop(name, None)
            elif edit == "vary" and name in _VALID:
                jobs[name] = data.draw(_neutral(name))
            elif edit in ("vary", "replace") or name not in jobs:
                jobs[name] = data.draw(_job())
            else:
                value = data.draw(_IFS if edit == "if" else _NEEDS)
                jobs[name] = {k: v for k, v in jobs[name].items() if k != edit}
                if value is not None:
                    jobs[name][edit] = value

    errors = ci_test_plan.workflow_document_errors(
        {"defaults": {"run": {"shell": "bash"}}, "jobs": jobs}
    )

    passes = _lets_ci_speak(jobs)
    event(
        "refused"
        if not passes
        else "passes, as the test workflow"
        if jobs == _VALID
        else "passes, changed"
    )
    assert (errors == ()) is passes, errors


@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(data=st.data())
def test_workflow_accepts_changes_no_rule_reads(data: st.DataObject) -> None:
    """Invariant: a job keeps passing when it gains a timeout, permissions,
    step names, another checkout version, a reordered ``needs``, an extra
    step that always runs or an extra selection output."""

    name = data.draw(st.sampled_from(sorted(_VALID)))
    jobs = {**_VALID, name: data.draw(_neutral(name))}

    assert _lets_ci_speak(jobs)
    assert (
        ci_test_plan.workflow_document_errors(
            {"defaults": {"run": {"shell": "bash"}}, "jobs": jobs}
        )
        == ()
    )


def test_verify_workflow_passes_the_repository_workflow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ci_test_plan.main(["verify-workflow"]) == 0
    assert capsys.readouterr().out == "workflow=ok\n"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            _CI_IF,
            "  ci:\n    if: always()\n      && false\n",
            "workflow fails the test plan:\n  " + _CI_MUST_ALWAYS_RUN,
        ),
        (
            _CI_IF,
            "  ci:\n    if: always()\n    if: always()\n",
            "workflow fails to load: a mapping sets `if` twice",
        ),
    ],
)
def test_verify_workflow_refuses_a_workflow_the_plan_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, old: str, new: str, message: str
) -> None:
    workflow = tmp_path / "test.yml"
    workflow.write_text(_edited(old, new), encoding="utf-8")
    monkeypatch.setattr(ci_test_plan, "WORKFLOW", workflow)

    with pytest.raises(SystemExit) as refused:
        ci_test_plan.main(["verify-workflow"])

    assert str(refused.value).startswith(message)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("\n  lint:\n", '\n  "lint":\n'),
        ("\njobs:\n", "\njobs: # every job of the plan\n"),
    ],
)
def test_verify_and_verify_workflow_agree_on_how_yaml_spells_the_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    old: str,
    new: str,
) -> None:
    """``verify-workflow`` alone reads the workflow's jobs, as PyYAML reads
    them, and ``verify`` leaves them to it. So a quoted job key, or a comment
    after ``jobs:``, passes both."""

    source = ci_test_plan.WORKFLOW.read_text(encoding="utf-8")
    assert source.count(old) == 1
    workflow = tmp_path / "test.yml"
    workflow.write_text(source.replace(old, new), encoding="utf-8")
    monkeypatch.setattr(ci_test_plan, "WORKFLOW", workflow)

    assert ci_test_plan.main(["verify-workflow"]) == 0
    assert ci_test_plan.main(["verify"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "verification=ok"


def _run_script(
    *arguments: str, bare: bool = True, **environment: str
) -> subprocess.CompletedProcess[str]:
    """Run the plan as the workflow does: as a script, from the repository root.

    ``bare`` adds ``-S``, which leaves site-packages out, as the runner's
    ``python3`` lacks the workspace's packages.
    """

    return subprocess.run(
        [sys.executable, *(["-S"] if bare else []), ci_test_plan.__file__, *arguments],
        cwd=ci_test_plan.ROOT,
        env=os.environ | environment,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def test_the_workflow_runs_this_module_as_a_script() -> None:
    script = ["python3", "/".join(Path(ci_test_plan.__file__).parts[-2:])]

    assert script == ["python3", "tools/ci_test_plan.py"]
    assert shlex.split(ci_test_plan.REQUIRE_RESULTS_COMMAND) == [
        *script,
        "require-results",
    ]
    assert shlex.split(ci_test_plan.SELECTION_COMMAND) == [*script, "select-countries"]


def test_require_results_fails_a_failed_run_when_ci_runs_it_as_a_script() -> None:
    """``ci`` passes or fails on this script's exit code, which no call of
    ``main`` inside pytest can see."""

    needs = _passing_needs()
    passed = _run_script("require-results", NEEDS=json.dumps(needs))

    assert (passed.returncode, passed.stderr) == (0, "")
    assert passed.stdout.splitlines()[-1] == "results=ok"

    needs["engine-free"]["result"] = "failure"
    failed = _run_script("require-results", NEEDS=json.dumps(needs))

    assert failed.returncode == 1
    assert failed.stderr == "workflow did not pass:\n  engine-free: failure\n"


def test_select_countries_writes_its_outputs_when_run_as_a_script(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"

    run = _run_script(
        "select-countries",
        GITHUB_EVENT_NAME="push",
        GITHUB_EVENT_PATH=str(event_path),
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_TOKEN="",
        GITHUB_OUTPUT=str(output),
    )

    assert (run.returncode, run.stderr) == (0, "")
    assert output.read_text(encoding="utf-8") == "run_us=true\nrun_uk=true\n"


@pytest.mark.parametrize(
    ("command", "bare", "last_line"),
    [("verify", True, "verification=ok"), ("verify-workflow", False, "workflow=ok")],
)
def test_lint_commands_report_when_run_as_scripts(
    command: str, bare: bool, last_line: str
) -> None:
    run = _run_script(command, bare=bare)

    assert (run.returncode, run.stderr) == (0, "")
    assert run.stdout.splitlines()[-1] == last_line


def _docs_listing(files: int) -> tuple[list[int], ci_test_plan.RequestJSON]:
    """A pull request that changes ``files`` documentation files, 100 a page."""

    pages: list[int] = []

    def request_json(_url: str, _token: str) -> list[dict[str, str]]:
        pages.append(len(pages) + 1)
        listed = max(0, min(100, files - 100 * (len(pages) - 1)))
        return [{"filename": "docs/guide.md"}] * listed

    return pages, request_json


def test_a_listing_that_reaches_githubs_limit_selects_both_countries() -> None:
    """GitHub lists at most 3,000 files, so a full listing may leave files out."""

    event = {"pull_request": {"number": 7}}

    pages, request_json = _docs_listing(2_950)
    paths, selected = ci_test_plan.selection_for_event(
        "pull_request", event, "owner/repo", "token", request_json=request_json
    )
    assert len(paths) == 2_950
    assert selected == {"run_us": False, "run_uk": False}

    pages, request_json = _docs_listing(3_000)
    paths, selected = ci_test_plan.selection_for_event(
        "pull_request", event, "owner/repo", "token", request_json=request_json
    )
    assert paths == []
    assert selected == {"run_us": True, "run_uk": True}
    assert len(pages) == 30


_JOBS = ("select-countries", "lint", "engine-free", "engine-us", "wheels")
_SELECTED_BY = {"engine-us": "run_us"}


def _needs(run_us: object = "true", **results: str) -> dict[str, dict[str, object]]:
    base = dict.fromkeys(_JOBS, "success") | {
        name.replace("_", "-"): result for name, result in results.items()
    }
    needs: dict[str, dict[str, object]] = {
        job: {"result": result, "outputs": {}} for job, result in base.items()
    }
    if run_us is not None:
        needs["select-countries"]["outputs"] = {"run_us": run_us}
    return needs


def _errors(needs: dict[str, object]) -> tuple[str, ...]:
    return ci_test_plan.aggregate_result_errors(
        needs, jobs=_JOBS, selected_by=_SELECTED_BY
    )


def test_aggregate_passes_when_every_job_passed_or_was_left_out() -> None:
    assert _errors(_needs()) == ()
    assert _errors(_needs(run_us="false", engine_us="skipped")) == ()
    # A job that ran and passed although the selection left it out still passed.
    assert _errors(_needs(run_us="false")) == ()


_NO_REPORT = (
    "engine-us: skipped, and select-countries did not report run_us as true or false",
)


@pytest.mark.parametrize(
    ("needs", "expected"),
    [
        (_needs(engine_free="failure"), ("engine-free: failure",)),
        (_needs(engine_us="cancelled"), ("engine-us: cancelled",)),
        (_needs(wheels="timed_out"), ("wheels: timed_out",)),
        (
            _needs(lint="skipped"),
            ("lint: skipped, but the plan runs it on every run",),
        ),
        (
            _needs(run_us="true", engine_us="skipped"),
            ("engine-us: skipped, but select-countries selected it (run_us=true)",),
        ),
        (_needs(run_us=None, engine_us="skipped"), _NO_REPORT),
        (_needs(run_us="", engine_us="skipped"), _NO_REPORT),
        (_needs(run_us="False", engine_us="skipped"), _NO_REPORT),
        (_needs(run_us="no", engine_us="skipped"), _NO_REPORT),
        (_needs(run_us=False, engine_us="skipped"), _NO_REPORT),
        (
            _needs(engine_us="skipped")
            | {"select-countries": {"result": "success", "outputs": "run_us=false"}},
            _NO_REPORT,
        ),
        (
            _needs(run_us="false", select_countries="failure", engine_us="skipped"),
            (
                "select-countries: failure",
                "engine-us: skipped, and select-countries did not succeed",
            ),
        ),
        (
            {job: entry for job, entry in _needs().items() if job != "wheels"},
            ("wheels: reported no result",),
        ),
        (_needs() | {"lint": "success"}, ("lint: reported no result",)),
        (
            _needs() | {"docs": {"result": "success"}},
            ("docs: reported a result, but the plan declares no such job",),
        ),
        ({}, tuple(f"{job}: reported no result" for job in _JOBS)),
    ],
)
def test_aggregate_names_each_job_that_did_not_pass(
    needs: dict[str, object], expected: tuple[str, ...]
) -> None:
    assert _errors(needs) == expected


# The same examples on every run, so a required check never fails by chance.
_PROPERTY = settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
_RESULT_VALUES = ("success", "failure", "cancelled", "skipped", "timed_out", None)
_RESULTS = st.sampled_from(_RESULT_VALUES)
_RUN_US = st.one_of(
    st.none(), st.sampled_from(["true", "false"]), st.text(), st.booleans()
)


def _passes(job: str, results: dict[str, str | None], run_us: object) -> bool:
    """The rule, stated independently of the implementation."""

    if results[job] == "success":
        return True
    return (
        results[job] == "skipped"
        and job in _SELECTED_BY
        and results["select-countries"] == "success"
        and isinstance(run_us, str)
        and run_us == "false"
    )


def _context(
    results: dict[str, str | None], run_us: object
) -> dict[str, dict[str, object]]:
    needs: dict[str, dict[str, object]] = {
        job: {"result": result, "outputs": {}}
        for job, result in results.items()
        if result is not None
    }
    if run_us is not None and "select-countries" in needs:
        needs["select-countries"]["outputs"] = {"run_us": run_us}
    return needs


@_PROPERTY
@given(results=st.fixed_dictionaries({job: _RESULTS for job in _JOBS}), run_us=_RUN_US)
def test_aggregate_passes_exactly_when_every_job_passed_or_was_left_out(
    results: dict[str, str | None], run_us: object
) -> None:
    """Invariant: one error for each job that neither passed nor skipped
    because the selection succeeded and reported ``false`` for it."""

    errors = _errors(_context(results, run_us))

    assert [error.split(":")[0] for error in errors] == [
        job for job in _JOBS if not _passes(job, results, run_us)
    ]


@_PROPERTY
@given(
    results=st.fixed_dictionaries({job: _RESULTS for job in _JOBS}),
    run_us=_RUN_US,
    job=st.sampled_from(_JOBS),
    bad=st.sampled_from(["failure", "cancelled", "timed_out"]),
)
def test_no_context_passes_with_a_failed_job(
    results: dict[str, str | None], run_us: object, job: str, bad: str
) -> None:
    """Invariant: a failed, cancelled or timed-out job always fails the check."""

    results = results | {job: bad}

    assert f"{job}: {bad}" in _errors(_context(results, run_us))


def test_aggregate_rule_holds_for_every_combination_of_results() -> None:
    """Every result of every job, crossed with five values of the output."""

    checked = 0
    for combination in itertools.product(_RESULT_VALUES, repeat=len(_JOBS)):
        results = dict(zip(_JOBS, combination, strict=True))
        for run_us in ("true", "false", "", "False", None):
            errors = _errors(_context(results, run_us))
            assert [error.split(":")[0] for error in errors] == [
                job for job in _JOBS if not _passes(job, results, run_us)
            ], (results, run_us)
            checked += 1

    assert checked == 6**5 * 5


_NAMES = st.text(alphabet="abcdefgh-", min_size=1, max_size=5)
_OUTPUT_NAMES = st.sampled_from(["run_us", "run_uk", "run_nz"])


@_PROPERTY
@given(data=st.data())
def test_aggregate_rule_holds_for_any_plan(data: st.DataObject) -> None:
    """Invariant: for any set of jobs and any selection, one error for each
    job that neither passed nor was left out, and one for each job the plan
    does not declare."""

    selection = ci_test_plan.SELECTION_JOB
    jobs = (selection, *data.draw(st.lists(_NAMES, max_size=6, unique=True)))
    selected_by = data.draw(st.dictionaries(st.sampled_from(jobs), _OUTPUT_NAMES))
    results = {job: data.draw(_RESULTS) for job in jobs}
    outputs = data.draw(st.dictionaries(_OUTPUT_NAMES, _RUN_US))
    strangers = data.draw(st.lists(_NAMES.filter(lambda name: name not in jobs)))
    needs: dict[str, object] = {
        job: {"result": result, "outputs": outputs if job == selection else {}}
        for job, result in results.items()
        if result is not None
    }
    needs |= {name: {"result": "success", "outputs": {}} for name in strangers}

    def passes(job: str) -> bool:
        if results[job] == "success":
            return True
        value = outputs.get(selected_by.get(job, ""))
        return (
            results[job] == "skipped"
            and job in selected_by
            and results[selection] == "success"
            and isinstance(value, str)
            and value == "false"
        )

    errors = ci_test_plan.aggregate_result_errors(
        needs, jobs=jobs, selected_by=selected_by
    )

    assert [error.split(":")[0] for error in errors] == [
        *(job for job in jobs if not passes(job)),
        *dict.fromkeys(strangers),
    ]


def _passing_needs() -> dict[str, dict[str, object]]:
    needs: dict[str, dict[str, object]] = {
        job: {"result": "success", "outputs": {}} for job in ci_test_plan.plan_jobs()
    }
    needs["select-countries"]["outputs"] = {"run_us": "true", "run_uk": "false"}
    needs["engine-uk"]["result"] = "skipped"
    needs["integration-uk"]["result"] = "skipped"
    return needs


def test_require_results_reads_the_needs_context_for_the_real_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    needs = _passing_needs()
    monkeypatch.setenv("NEEDS", json.dumps(needs))

    assert ci_test_plan.main(["require-results"]) == 0
    assert "engine-uk=skipped" in capsys.readouterr().out

    needs["lint"]["result"] = "skipped"
    needs["engine-us"]["result"] = "skipped"
    monkeypatch.setenv("NEEDS", json.dumps(needs))
    with pytest.raises(SystemExit) as refused:
        ci_test_plan.main(["require-results"])
    assert str(refused.value) == (
        "workflow did not pass:\n"
        "  lint: skipped, but the plan runs it on every run\n"
        "  engine-us: skipped, but select-countries selected it (run_us=true)"
    )


def test_require_results_hears_from_every_job_the_workflow_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ci_test_plan.WORKFLOW.read_text(encoding="utf-8")

    assert sorted(ci_test_plan.plan_jobs()) == sorted(_other_jobs(source))
    for job in ci_test_plan.plan_jobs():
        needs = _passing_needs()
        del needs[job]
        monkeypatch.setenv("NEEDS", json.dumps(needs))
        with pytest.raises(SystemExit) as refused:
            ci_test_plan.main(["require-results"])
        assert f"  {job}: reported no result" in str(refused.value)


@pytest.mark.parametrize("value", [None, "", "not json", "[]", '"success"'])
def test_require_results_refuses_a_missing_or_malformed_needs_context(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("NEEDS", raising=False)
    else:
        monkeypatch.setenv("NEEDS", value)

    with pytest.raises(SystemExit) as refused:
        ci_test_plan.main(["require-results"])

    assert str(refused.value).startswith("NEEDS must hold")


def test_engine_guard_detection_ignores_strings_but_finds_executable_guards() -> None:
    source = """
TEXT = '@pytest.mark.requires_uk'

def test_example():
    pytest.importorskip("policyengine_uk")
"""
    assert ci_test_plan.engine_guard_lines(source) == (5,)


def test_every_repository_test_has_a_valid_execution_category() -> None:
    invalid = [
        path.relative_to(ci_test_plan.ROOT)
        for path in ci_test_plan.test_files()
        if ci_test_plan.group_name_for_test_path(path) is None
    ]
    assert invalid == []


def test_every_test_is_owned_by_exactly_one_job() -> None:
    selected = [
        path
        for job in ci_test_plan.JOBS
        for path in ci_test_plan.selected_files_for_job(job)
    ]

    assert len(selected) == len(ci_test_plan.test_files())
    assert len(selected) == len(set(selected))


def test_us_reporting_categories_come_from_the_registry() -> None:
    assert ci_test_plan.reporting_groups_for_job("engine-us") == {
        "contract": "engine-contract-us",
        "scenario": "engine-scenario-us",
        "workflow": "engine-workflow-us",
    }


def test_the_us_runner_runs_every_engine_us_group_of_the_plan() -> None:
    assert ci_test_plan.US_RUNNER_JOB == "engine-us"
    assert ci_test_plan.us_runner_errors() == ()


@pytest.mark.parametrize(
    ("category", "never_runs"),
    [
        ("parity", "engine-parity-us"),
        (None, "engine-parity-us"),
        # The runner runs the later of two groups that report one category.
        ("contract", "engine-contract-us"),
    ],
)
def test_verify_refuses_an_engine_us_group_the_us_runner_never_runs(
    monkeypatch: pytest.MonkeyPatch, category: str | None, never_runs: str
) -> None:
    monkeypatch.setitem(
        ci_test_plan.TEST_GROUPS,
        "engine-parity-us",
        ci_test_plan.TestGroup(
            ("engine_parity", "us"),
            "engine-us",
            "us",
            report_category=category,
            engine_module="policyengine_us",
        ),
    )
    monkeypatch.setattr(ci_test_plan, "test_files", lambda: (Path(__file__),))
    refusal = (
        f"{never_runs}: the US runner never runs this group, since it runs one "
        "engine-us group for each report category in contract, scenario, workflow"
    )

    assert ci_test_plan.us_runner_errors() == (refusal,)
    with pytest.raises(SystemExit) as refused:
        ci_test_plan.verify()
    assert f"\n  {refusal}\n" in str(refused.value)


def test_unknown_non_documentation_paths_select_both_countries() -> None:
    assert ci_test_plan.scope_for_path("new-area/unrecognized.config") == "shared"
    assert ci_test_plan.select_countries(["new-area/unrecognized.config"]) == {
        "run_us": True,
        "run_uk": True,
    }


@pytest.mark.parametrize(
    ("path", "scope"),
    [
        ("build/us/spec.yaml", "us"),
        ("build/uk/spec.yaml", "uk"),
        ("packages/example/src/microcosm/build/us_runtime/module.py", "us"),
        ("packages/example/src/microcosm/build/uk_runtime/module.py", "uk"),
        ("tools/build_us_candidate.py", "us"),
        ("tools/build_uk_candidate.py", "uk"),
        ("packages/example/src/microcosm/shared.py", "shared"),
    ],
)
def test_country_specific_source_paths_keep_their_existing_scope(
    path: str, scope: str
) -> None:
    assert ci_test_plan.scope_for_path(path) == scope


def _country_words(name: str) -> set[str]:
    """Return the countries that are words of a file name, found one character
    at a time: a word is a maximal run of ASCII letters, read without case."""

    words = {
        "".join(run).lower()
        for letter, run in itertools.groupby(
            name, lambda character: character in string.ascii_letters
        )
        if letter
    }
    return words & {"us", "uk"}


def _tool_scope(name: str) -> str | None:
    """The scope of a file below ``tools/``: a country when it is a word of the
    file name and the other country's letters appear nowhere in the name."""

    if PurePosixPath(name).suffix.lower() in ci_test_plan.DOC_SUFFIXES:
        return None
    words, lowered = _country_words(name), name.lower()
    if "us" in words and "uk" not in lowered:
        return "us"
    if "uk" in words and "us" not in lowered:
        return "uk"
    return "shared"


# Letters outside ASCII: ``str.lower`` turns the Kelvin sign into "k".
_OTHER_LETTERS = "Kſé"
_NAME_PARTS = st.one_of(
    st.sampled_from(["us", "uk", "US", "Uk", "uS", "UK"]),
    st.text(alphabet=string.ascii_letters + string.digits + _OTHER_LETTERS, max_size=5),
)
# An empty separator joins two parts into one word.
_SEPARATORS = st.sampled_from(["", "", "_", "-", ".", " ", "2", "K"])


@st.composite
def _tool_names(draw: st.DrawFn) -> str:
    parts = draw(st.lists(_NAME_PARTS, min_size=1, max_size=5))
    name = parts[0] + "".join(draw(_SEPARATORS) + part for part in parts[1:])
    assume(name not in ("", ".", ".."))
    return name


@_PROPERTY
@given(name=_tool_names())
def test_a_tool_belongs_to_a_country_exactly_when_its_name_holds_it_alone(
    name: str,
) -> None:
    """Invariant: a file below ``tools/`` that isn't documentation scopes to a
    country exactly when that country is a word of its file name and the
    other country's letters appear nowhere in the name, and to shared
    otherwise."""

    scope = _tool_scope(name)
    event(f"scope {scope}")
    assert ci_test_plan.scope_for_path(f"tools/{name}") == scope


@_PROPERTY
@given(name=_tool_names())
def test_a_tool_change_runs_each_country_whose_letters_its_name_holds(
    name: str,
) -> None:
    """Invariant: a change to a file below ``tools/`` that isn't documentation
    selects each country whose two letters its lower-cased file name holds
    anywhere, as a word or inside one, and both countries when it holds
    neither."""

    assume(PurePosixPath(name).suffix.lower() not in ci_test_plan.DOC_SUFFIXES)
    lowered = name.lower()
    held = {country for country in ("us", "uk") if country in lowered}
    event(f"letters of {sorted(held) or 'neither'}")
    selected = ci_test_plan.select_countries([f"tools/{name}"])
    for country in held or {"us", "uk"}:
        assert selected[f"run_{country}"], (name, selected)


@pytest.mark.parametrize(
    ("path", "scope"),
    [
        ("tools/run_engine_us_tests.sh", "us"),
        ("tools/US-targets.json", "us"),
        # A digit ends a word.
        ("tools/build_us2024_inputs.py", "us"),
        # A country's letters inside another word.
        ("tools/status.py", "shared"),
        ("tools/focus_report.py", "shared"),
        ("tools/census_download.py", "shared"),
        # One country as a word, and the other's letters inside another word.
        ("tools/census_uk_local_targets.py", "shared"),
        ("tools/validate_uk_housing_imputation.py", "shared"),
        ("tools/uk_to_usa_crosswalk.py", "shared"),
        ("tools/us_ukraine_aid.py", "shared"),
        ("tools/run_engine_uk_tests.sh", "uk"),
        # Both countries, or none as a word.
        ("tools/compare_us_uk_targets.py", "shared"),
        ("tools/buildUSCandidate.py", "shared"),
        ("tools/uK_targets.py", "shared"),
        # The file name decides, not its directory.
        ("tools/us_bundle_generation/core.py", "shared"),
        ("tools/route_a/run_us.sh", "us"),
    ],
)
def test_a_tool_takes_the_country_its_file_name_holds_alone(
    path: str, scope: str
) -> None:
    assert ci_test_plan.scope_for_path(path) == scope


def test_a_tool_change_selects_the_country_its_name_holds_alone() -> None:
    both = {"run_us": True, "run_uk": True}
    assert ci_test_plan.select_countries(["tools/status.py"]) == both
    assert ci_test_plan.select_countries(["tools/census_uk_local_targets.py"]) == both
    assert ci_test_plan.select_countries(["tools/run_engine_uk_tests.sh"]) == {
        "run_us": False,
        "run_uk": True,
    }


def _checkout_files(directory: str) -> list[str]:
    """Return the tracked and new files below ``directory``, as git lists them."""

    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            directory,
        ],
        cwd=ci_test_plan.ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(set(listed.stdout.decode().split("\0")) - {""})


def test_every_tool_in_the_checkout_takes_the_scope_its_name_gives_it() -> None:
    """Every file below ``tools/`` in the checkout takes the scope its file
    name gives it, and a change to it selects each country whose letters the
    name holds."""

    files = _checkout_files("tools")
    assert files
    for path in files:
        if ci_test_plan.group_name_for_test_path(PurePosixPath(path)) is not None:
            continue  # A test module takes its group's country.
        name = PurePosixPath(path).name
        scope = _tool_scope(name)
        assert ci_test_plan.scope_for_path(path) == scope, path
        if scope is None:
            continue  # Documentation selects no country.
        selected = ci_test_plan.select_countries([path])
        for country in ("us", "uk"):
            if country in name.lower():
                assert selected[f"run_{country}"], path


def test_documentation_only_changes_select_no_country_engine() -> None:
    documentation = [
        "README.md",
        "DESIGN.md",
        "docs/testing.md",
        "changelog.d/change.changed.md",
    ]

    assert ci_test_plan.select_countries(documentation) == {
        "run_us": False,
        "run_uk": False,
    }


def test_country_specific_and_mixed_changes_select_expected_engines() -> None:
    us_test = "packages/example/tests/engine_scenario/us/test_example.py"
    uk_test = "packages/example/tests/integration/uk/test_example.py"

    assert ci_test_plan.select_countries([us_test]) == {
        "run_us": True,
        "run_uk": False,
    }
    assert ci_test_plan.select_countries([uk_test]) == {
        "run_us": False,
        "run_uk": True,
    }
    assert ci_test_plan.select_countries([us_test, uk_test]) == {
        "run_us": True,
        "run_uk": True,
    }
    assert ci_test_plan.select_countries(["README.md", us_test]) == {
        "run_us": True,
        "run_uk": False,
    }
    assert ci_test_plan.select_countries([]) == {"run_us": True, "run_uk": True}


def test_pull_request_paths_fetches_every_page_and_previous_rename_path() -> None:
    calls: list[str] = []
    first_page = [
        {"filename": f"docs/item-{index}.md", "status": "modified"}
        for index in range(100)
    ]
    second_page = [
        {
            "filename": "packages/example/tests/engine_scenario/us/test_new.py",
            "previous_filename": "packages/example/tests/engine/uk/test_old.py",
            "status": "renamed",
        }
    ]

    def request_json(url: str, _token: str):
        calls.append(url)
        return first_page if url.endswith("page=1") else second_page

    paths = ci_test_plan.pull_request_paths(
        {"pull_request": {"number": 42}},
        "PolicyEngine/microcosm",
        "token",
        request_json=request_json,
    )

    assert len(calls) == 2
    assert paths[-2:] == [
        "packages/example/tests/engine_scenario/us/test_new.py",
        "packages/example/tests/engine/uk/test_old.py",
    ]
    assert ci_test_plan.select_countries(paths) == {
        "run_us": True,
        "run_uk": True,
    }


def test_main_push_selects_both_countries_without_requesting_changed_files() -> None:
    def unexpected_request(_url: str, _token: str):
        raise AssertionError("main pushes must not request changed files")

    paths, selected = ci_test_plan.selection_for_event(
        "push",
        {"after": "a" * 40},
        "PolicyEngine/microcosm",
        "token",
        request_json=unexpected_request,
    )

    assert paths == []
    assert selected == {"run_us": True, "run_uk": True}


def test_main_push_writes_both_country_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.txt"
    event_path.write_text('{"after": "abc"}', encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REPOSITORY", "PolicyEngine/microcosm")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    ci_test_plan.write_country_selection()

    assert output_path.read_text(encoding="utf-8") == "run_us=true\nrun_uk=true\n"
    assert "main push: all country jobs selected" in capsys.readouterr().out


def test_description_is_generated_from_the_registry() -> None:
    description = ci_test_plan.describe()

    for name, spec in ci_test_plan.TEST_GROUPS.items():
        assert name in description
        assert "/".join(spec.directory) in description
