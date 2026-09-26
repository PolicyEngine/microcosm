"""Owned temporary-directory controls; no audit, main, Torch or Microcosm run."""

import importlib.util
import stat
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("owned_temp_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("refuses", [False, True])
def test_owned_fixture_survives_success_and_exact_refusal(
    diagnostic, tmp_path, refuses
):
    context = b'{"code":"READ_SCOPE","event":"open"}\n'
    error = diagnostic.RefusalError("READ_SCOPE", boundary_context=context)
    caught = None
    try:
        with diagnostic.retained_owned_directory(tmp_path) as owned:
            assert owned.parent == tmp_path
            assert owned.name.startswith("spec-seed-owned-")
            assert stat.S_IMODE(owned.stat().st_mode) == 0o700
            fixture = owned / "invented-spec.txt"
            fixture.write_text("invented fixture only")
            if refuses:
                raise error
    except diagnostic.RefusalError as actual:
        caught = actual
    assert caught is (error if refuses else None)
    if caught is not None:
        assert caught.boundary_context is context
    assert fixture.read_text() == "invented fixture only"
    assert owned.is_dir()


def test_each_workspace_is_distinct_and_has_no_implicit_cleanup(
    diagnostic, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("implicit TemporaryDirectory cleanup was registered")

    monkeypatch.setattr(diagnostic.tempfile, "TemporaryDirectory", forbidden)
    with diagnostic.retained_owned_directory(tmp_path) as first:
        (first / "invented.txt").write_text("first")
    with diagnostic.retained_owned_directory(tmp_path) as second:
        (second / "invented.txt").write_text("second")
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert (first / "invented.txt").read_text() == "first"
    assert (second / "invented.txt").read_text() == "second"
