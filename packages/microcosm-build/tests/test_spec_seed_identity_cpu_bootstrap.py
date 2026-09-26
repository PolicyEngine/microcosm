"""Invented bootstrap controls; no Torch, Microcosm, engines or data imported."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    # Normal stdlib-only module loading; main/audit/derive are never called.
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("cpu_bootstrap_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_import_state(monkeypatch):
    # Keep the process import roster isolated without importing any dependency.
    for name in tuple(sys.modules):
        if (
            name == "torch"
            or name == "cuda.bindings"
            or name.startswith("cuda.bindings.")
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


@pytest.mark.parametrize(
    "name", ["json", "torch", "cuda", "cuda.pathfinder", "cuda.bindings_other"]
)
def test_non_cuda_bindings_imports_pass_through(diagnostic, name):
    finder = diagnostic._CpuOnlyCudaBindingsFinder()
    assert finder.find_spec(name, ["invented"], object()) is None
    assert finder.attempts == 0


def test_only_exact_optional_bindings_are_excluded(diagnostic):
    finder = diagnostic._CpuOnlyCudaBindingsFinder()
    with pytest.raises(ModuleNotFoundError) as error:
        finder.find_spec("cuda.bindings")
    assert error.value.name == "cuda.bindings"
    assert finder.attempts == 1


@pytest.mark.parametrize("fails", [False, True])
def test_finder_removed_on_success_and_failure(diagnostic, fresh_import_state, fails):
    before = tuple(sys.meta_path)
    other = object()
    evidence = {}

    class InventedFailureError(Exception):
        pass

    try:
        with diagnostic.cpu_only_torch_import(evidence):
            finder = sys.meta_path[0]
            assert type(finder) is diagnostic._CpuOnlyCudaBindingsFinder
            assert tuple(sys.meta_path[1:]) == before
            sys.meta_path.append(other)
            with pytest.raises(ModuleNotFoundError):
                finder.find_spec("cuda.bindings")
            if fails:
                raise InventedFailureError
    except InventedFailureError:
        assert fails
    assert tuple(sys.meta_path) == (*before, other)
    assert evidence == {
        "cuda_bindings_import_attempts": 1,
        "temporary_finder_removed": True,
    }


@pytest.mark.parametrize("name", ["torch", "cuda.bindings", "cuda.bindings.runtime"])
def test_preloaded_optional_path_refuses_before_finder(
    diagnostic, fresh_import_state, monkeypatch, name
):
    monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    before = tuple(sys.meta_path)
    with pytest.raises(diagnostic.RefusalError, match="^TORCH_BOOTSTRAP_PRELOADED$"):
        with diagnostic.cpu_only_torch_import({}):
            pytest.fail("preloaded dependency was accepted")
    assert tuple(sys.meta_path) == before


def test_actual_fallback_contract_is_required(
    diagnostic, fresh_import_state, monkeypatch
):
    evidence = {
        "temporary_finder_removed": True,
        "cuda_bindings_import_attempts": 1,
    }
    monkeypatch.setitem(
        sys.modules,
        "torch.cuda._utils",
        SimpleNamespace(_HAS_CUDA_BINDINGS=False, _cuda_bindings_runtime=None),
    )
    diagnostic.verify_cpu_only_torch_import(evidence)
    assert evidence["torch_optional_bindings_fallback_verified"] is True


@pytest.mark.parametrize("bindings_present", [False, True])
def test_unconfirmed_or_loaded_bindings_refuse(
    diagnostic, fresh_import_state, monkeypatch, bindings_present
):
    evidence = {
        "temporary_finder_removed": True,
        "cuda_bindings_import_attempts": 1 if bindings_present else 0,
    }
    monkeypatch.setitem(
        sys.modules,
        "torch.cuda._utils",
        SimpleNamespace(_HAS_CUDA_BINDINGS=False, _cuda_bindings_runtime=None),
    )
    if bindings_present:
        monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace())
    with pytest.raises(diagnostic.RefusalError, match="^TORCH_CPU_FALLBACK$"):
        diagnostic.verify_cpu_only_torch_import(evidence)


def _candidate(monkeypatch):
    monkeypatch.setattr(sys, "base_prefix", "/invented-bootstrap-prefix")
    return str(
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    )


def test_only_exact_confirmed_absent_zip_entry_is_removed(diagnostic, monkeypatch):
    candidate = _candidate(monkeypatch)
    others = ["", "/invented/code", "/invented/unrelated.zip", candidate + "-other"]
    monkeypatch.setattr(sys, "path", [candidate, *others, candidate])
    checked = []

    def absent(path):
        checked.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(diagnostic.os, "lstat", absent)
    assert diagnostic.omit_absent_stdlib_zip() is True
    assert checked == [candidate]
    assert sys.path == others


@pytest.mark.parametrize("kind", ["regular", "broken_symlink"])
def test_existing_zip_or_symlink_remains_subject_to_original_guard(
    diagnostic, monkeypatch, kind
):
    candidate = _candidate(monkeypatch)
    monkeypatch.setattr(sys, "path", [candidate, "other"])
    # Any successful lstat, including a broken symlink, prevents omission.
    monkeypatch.setattr(diagnostic.os, "lstat", lambda path: SimpleNamespace(kind=kind))
    assert diagnostic.omit_absent_stdlib_zip() is False
    assert sys.path == [candidate, "other"]


def test_other_stat_errors_do_not_justify_omission(diagnostic, monkeypatch):
    candidate = _candidate(monkeypatch)
    monkeypatch.setattr(sys, "path", [candidate])

    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(diagnostic.os, "lstat", denied)
    with pytest.raises(PermissionError):
        diagnostic.omit_absent_stdlib_zip()
    assert sys.path == [candidate]


def test_unlisted_stdlib_candidate_is_not_probed(diagnostic, monkeypatch):
    _candidate(monkeypatch)
    monkeypatch.setattr(sys, "path", ["other"])
    monkeypatch.setattr(
        diagnostic.os, "lstat", lambda path: pytest.fail("unexpected metadata probe")
    )
    assert diagnostic.omit_absent_stdlib_zip() is False
    assert sys.path == ["other"]


def test_real_non_cuda_import_keeps_original_loader(
    diagnostic, fresh_import_state, monkeypatch
):
    name = "invented_non_cuda_bootstrap_control"
    monkeypatch.delitem(sys.modules, name, raising=False)
    loaded = []

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            loaded.append(module.__name__)
            module.marker = "ordinary_loader"

    loader = Loader()

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return importlib.machinery.ModuleSpec(fullname, loader)
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    before = tuple(sys.meta_path)
    try:
        with diagnostic.cpu_only_torch_import({}):
            module = importlib.import_module(name)
        assert module.marker == "ordinary_loader"
        assert module.__loader__ is loader
        assert loaded == [name]
        assert tuple(sys.meta_path) == before
    finally:
        sys.modules.pop(name, None)
