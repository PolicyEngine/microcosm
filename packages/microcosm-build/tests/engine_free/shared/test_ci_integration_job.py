"""Contracts for executing every test environment in one CI workflow."""

from __future__ import annotations

from test_support.paths import paths_for
from tools import ci_test_groups

_ROOT = paths_for("microcosm-build").repository
_TEST_WORKFLOW = _ROOT / ".github/workflows/test.yml"
_LEGACY_INTEGRATION_WORKFLOW = _ROOT / ".github/workflows/integration-tests.yml"
_INTEGRATION_SCRIPT = _ROOT / "tools/run_integration_tests.sh"


def test_tests_workflow_selects_every_test_environment() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")
    script = _INTEGRATION_SCRIPT.read_text(encoding="utf-8")

    assert not _LEGACY_INTEGRATION_WORKFLOW.exists()
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "  integration-uk:\n" in workflow
    for group in ci_test_groups.GROUP_DIRECTORIES:
        assert group in workflow + script
    assert "uv sync --all-packages --locked --extra uk" in script
    assert "--list integration-uk" in script
    assert "--run-integration" in script


def test_integration_job_is_required_and_read_only() -> None:
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "timeout-minutes: 15" in workflow
    assert "permissions:\n      contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "environment:" not in workflow
    assert "head.repo.full_name" not in workflow
    assert "HF_STAGING_READ_TOKEN: ${{ secrets.HF_STAGING_READ_TOKEN }}" in workflow
    assert "run: bash tools/run_integration_tests.sh" in workflow
    assert (
        "needs: [changes, lint, engine-free, engine-us, engine-uk, "
        "integration-uk, wheels]" in workflow
    )
    assert "INTEGRATION_UK_RESULT: ${{ needs['integration-uk'].result }}" in workflow
    assert 'require_success integration-uk "$INTEGRATION_UK_RESULT"' in workflow


def test_us_engine_uses_two_workers_with_memory_diagnostics() -> None:
    """US engine tests should match engine-free pytest reporting and concurrency."""
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")

    engine_free = workflow.split("\n  engine-free:\n", 1)[1].split(
        "\n  engine-us:\n", 1
    )[0]
    engine_us = workflow.split("\n  engine-us:\n", 1)[1].split("\n  engine-uk:\n", 1)[0]
    engine_uk = workflow.split("\n  engine-uk:\n", 1)[1].split(
        "\n  integration-uk:\n", 1
    )[0]

    assert "-n 2 --dist loadfile" in engine_free
    assert "-n 2 --dist loadfile" in engine_us
    assert "tools/ci_memory_monitor.py" in engine_us
    assert "engine-us-memory-${{ matrix.python-version }}.jsonl" in engine_us
    assert "uses: actions/upload-artifact@v4" in engine_us
    assert "if: always()" in engine_us
    assert "-n 2" not in engine_uk
    assert "--dist loadfile" not in engine_uk


def test_ordinary_jobs_report_the_first_failure_with_test_names() -> None:
    """CI must identify each test and finish reporting the first failure."""
    workflow = _TEST_WORKFLOW.read_text(encoding="utf-8")

    engine_free = workflow.split("\n  engine-free:\n", 1)[1].split(
        "\n  engine-us:\n", 1
    )[0]
    engine_us = workflow.split("\n  engine-us:\n", 1)[1].split("\n  engine-uk:\n", 1)[0]
    engine_uk = workflow.split("\n  engine-uk:\n", 1)[1].split(
        "\n  integration-uk:\n", 1
    )[0]

    for ordinary_job in (engine_free, engine_us, engine_uk):
        assert "-v --tb=short --maxfail=1" in ordinary_job
