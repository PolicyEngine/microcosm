"""Contracts for executing every test environment in one CI workflow."""

from __future__ import annotations

from test_support.paths import paths_for
from tools import ci_test_plan

_ROOT = paths_for("microcosm-build").repository
_TEST_WORKFLOW = _ROOT / ".github/workflows/test.yml"
_LEGACY_INTEGRATION_WORKFLOW = _ROOT / ".github/workflows/integration-tests.yml"
_INTEGRATION_SCRIPT = _ROOT / "tools/run_integration_tests.sh"
_ENGINE_RUNNER = _ROOT / "tools/run_engine_test_categories.py"
_LEGACY_TEST_GROUPS = _ROOT / "tools/ci_test_groups.py"
_LEGACY_CHANGE_CLASSIFIER = _ROOT / "tools/classify_ci_changes.py"
_ENGINE_FREE_SCRIPT = _ROOT / "tools/run_engine_free_tests.sh"
_ENGINE_US_SCRIPT = _ROOT / "tools/run_engine_us_tests.sh"
_ENGINE_US_SUMMARY_SCRIPT = _ROOT / "tools/add_engine_us_timing_summary.sh"
_ENGINE_UK_SCRIPT = _ROOT / "tools/run_engine_uk_tests.sh"
_LEGACY_CI_RESULTS_SCRIPT = _ROOT / "tools/require_ci_results.sh"
_AGENT_GUIDE = _ROOT / "docs/agent-guide.md"
_AGENT_POINTERS = (_ROOT / "AGENTS.md", _ROOT / "CLAUDE.md")
_AGENT_POINTER_TEXT = (
    "# Microcosm agent instructions\n\n"
    "The canonical instructions for every coding agent are in "
    "[docs/agent-guide.md](docs/agent-guide.md).\n"
)

_WORKFLOW_SCRIPTS = (
    _ENGINE_FREE_SCRIPT,
    _ENGINE_US_SCRIPT,
    _ENGINE_US_SUMMARY_SCRIPT,
    _ENGINE_UK_SCRIPT,
    _INTEGRATION_SCRIPT,
)


def test_tests_workflow_selects_every_test_environment() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")
    scripts = "\n".join(path.read_text(encoding="utf-8") for path in _WORKFLOW_SCRIPTS)
    engine_runner = _ENGINE_RUNNER.read_text(encoding="utf-8")

    assert not _LEGACY_INTEGRATION_WORKFLOW.exists()
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "  integration-uk:\n" in workflow
    for job in {spec.job for spec in ci_test_plan.TEST_GROUPS.values()}:
        assert f"  {job}:\n" in workflow
    assert "uv sync --all-packages --locked --extra uk" in scripts
    assert "list-job integration-uk" in scripts
    assert "--run-integration" in scripts
    assert "reporting_groups_for_job" in engine_runner


def test_integration_job_is_uk_selected_and_read_only() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")
    integration = workflow.split("\n  integration-uk:\n", maxsplit=1)[1].split(
        "\n  wheels:\n", maxsplit=1
    )[0]

    assert "needs: select-countries" in integration
    assert "if: needs['select-countries'].outputs.run_uk == 'true'" in integration
    assert "timeout-minutes: 15" in integration
    assert "permissions:\n      contents: read" in integration
    assert "persist-credentials: false" in integration
    assert "environment:" not in integration
    assert "head.repo.full_name" not in integration
    assert "HF_STAGING_READ_TOKEN: ${{ secrets.HF_STAGING_READ_TOKEN }}" in integration
    assert "run: bash tools/run_integration_tests.sh" in integration


def test_every_workflow_job_caps_its_runtime() -> None:
    """A hung job must not hold an org-shared runner for GitHub's 6-hour default."""
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")
    jobs = "\n" + workflow.split("\njobs:\n", maxsplit=1)[1]
    names = ci_test_plan.workflow_job_names(workflow)

    assert names
    for name, following in zip(names, (*names[1:], None), strict=True):
        block = jobs.split(f"\n  {name}:\n", maxsplit=1)[1]
        if following is not None:
            block = block.split(f"\n  {following}:\n", maxsplit=1)[0]
        assert "\n    timeout-minutes: " in f"\n{block}", name


def test_workflow_uses_one_country_selector_only_for_country_jobs() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")
    selector = workflow.split("\n  select-countries:\n", maxsplit=1)[1].split(
        "\n  lint:\n", maxsplit=1
    )[0]
    engine_free = workflow.split("\n  engine-free:\n", maxsplit=1)[1].split(
        "\n  engine-us:\n", maxsplit=1
    )[0]
    engine_us = workflow.split("\n  engine-us:\n", maxsplit=1)[1].split(
        "\n  engine-uk:\n", maxsplit=1
    )[0]
    engine_uk = workflow.split("\n  engine-uk:\n", maxsplit=1)[1].split(
        "\n  integration-uk:\n", maxsplit=1
    )[0]

    assert "  changes:\n" not in workflow
    assert "  ci-ok:\n" not in workflow
    assert "  select-countries:\n" in workflow
    assert "permissions:\n      contents: read\n      pull-requests: read" in selector
    assert "run_us: ${{ steps.plan.outputs.run_us }}" in workflow
    assert "run_uk: ${{ steps.plan.outputs.run_uk }}" in workflow
    assert "run: python3 tools/ci_test_plan.py select-countries" in workflow
    assert "needs:" not in engine_free
    assert "RUN_US" not in engine_free
    assert "RUN_UK" not in engine_free
    assert "needs: select-countries" in engine_us
    assert "if: needs['select-countries'].outputs.run_us == 'true'" in engine_us
    assert "needs: select-countries" in engine_uk
    assert "if: needs['select-countries'].outputs.run_uk == 'true'" in engine_uk
    assert not _LEGACY_CHANGE_CLASSIFIER.exists()
    assert not _LEGACY_TEST_GROUPS.exists()
    assert not _LEGACY_CI_RESULTS_SCRIPT.exists()


def test_workflow_invokes_versioned_shell_scripts_without_inline_blocks() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "run: |" not in workflow
    for path in _WORKFLOW_SCRIPTS:
        relative = path.relative_to(_ROOT)
        assert f"run: bash {relative}" in workflow
        assert path.read_text(encoding="utf-8").startswith(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
        )


def test_country_engine_jobs_bound_their_process_concurrency() -> None:
    """Country-engine tests should bound their peak memory use."""
    engine_free = _ENGINE_FREE_SCRIPT.read_text(encoding="utf-8")
    engine_us = _ENGINE_US_SCRIPT.read_text(encoding="utf-8")
    engine_uk = _ENGINE_UK_SCRIPT.read_text(encoding="utf-8")

    assert "-n 2 --dist loadfile" in engine_free
    assert "-m tools.run_engine_test_categories" in engine_us
    assert "--json-report engine-us-timings.json" in engine_us
    assert "--markdown-report engine-us-timings.md" in engine_us
    assert "-n 2" not in engine_us
    assert "--dist loadfile" not in engine_us
    assert "-n 2" not in engine_uk
    assert "--dist loadfile" not in engine_uk


def test_ordinary_jobs_report_the_first_failure_with_test_names() -> None:
    """CI must identify each test and finish reporting the first failure."""
    engine_runner = _ENGINE_RUNNER.read_text(encoding="utf-8")
    engine_free = _ENGINE_FREE_SCRIPT.read_text(encoding="utf-8")
    engine_uk = _ENGINE_UK_SCRIPT.read_text(encoding="utf-8")

    for ordinary_job in (engine_free, engine_uk):
        assert "-v --tb=short --maxfail=1" in ordinary_job
    for option in ('"-v"', '"--tb=short"', '"--maxfail=1"'):
        assert option in engine_runner


def test_agent_entrypoints_share_one_provider_neutral_guide() -> None:
    guide = _AGENT_GUIDE.read_text(encoding="utf-8")

    for pointer in _AGENT_POINTERS:
        assert pointer.read_text(encoding="utf-8") == _AGENT_POINTER_TEXT
    assert "tools/ci_test_plan.py" in guide
    assert "only authority" in guide
    assert "tools/classify_ci_changes.py" not in guide
    assert "tools/ci_test_groups.py" not in guide
    assert "ci-ok" not in guide
    assert "Claude" not in guide
    assert "Codex" not in guide
