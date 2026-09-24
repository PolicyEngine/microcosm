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
