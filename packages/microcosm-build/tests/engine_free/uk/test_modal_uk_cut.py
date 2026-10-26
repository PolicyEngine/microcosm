"""tools/modal_uk_cut.py, imported against a stub ``modal`` module."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

ROOT = Path(__file__).resolve().parents[5]


def _load_app(monkeypatch):
    """The wrapper as the launcher sees it: ``modal.is_local()`` is true."""

    stub = MagicMock(name="modal")
    stub.is_local.return_value = True
    stub.App.return_value.function.side_effect = lambda **options: (
        lambda function: setattr(function, "modal_options", options) or function
    )
    stub.App.return_value.local_entrypoint.side_effect = lambda **_: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", stub)
    spec = importlib.util.spec_from_file_location(
        "modal_uk_cut_under_test", ROOT / "tools" / "modal_uk_cut.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, stub


def test_a_deliberate_run_passes_the_registry_variable(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, "key-test")
    monkeypatch.setenv("MICROCOSM_UK_CUT_MODAL_RUN", "1")
    module, stub = _load_app(monkeypatch)
    assert module.registry.UK_GATE_SIGNING_KEY_ENV == UK_GATE_SIGNING_KEY_ENV
    stub.Secret.from_dict.assert_called_once_with(
        {"HF_TOKEN": "hf-test", UK_GATE_SIGNING_KEY_ENV: "key-test"}
    )
    assert module.cut.modal_options["secrets"] == [stub.Secret.from_dict.return_value]


def test_without_the_opt_in_no_credential_is_attached(monkeypatch):
    """modal run sends an app's secrets before its entrypoint runs."""

    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, "key-test")
    monkeypatch.delenv("MICROCOSM_UK_CUT_MODAL_RUN", raising=False)
    module, stub = _load_app(monkeypatch)
    stub.Secret.from_dict.assert_not_called()
    assert module.cut.modal_options["secrets"] == []
    with pytest.raises(SystemExit, match="deliberate, paid run"):
        module.main("a" * 40, "t", run=True)


def test_planning_contacts_no_modal_service(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv(UK_GATE_SIGNING_KEY_ENV, raising=False)
    module, stub = _load_app(monkeypatch)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kw: module.subprocess.CompletedProcess(argv, 0),
    )
    stub.reset_mock()
    assert module.plan("a" * 40, "t") == [UK_GATE_SIGNING_KEY_ENV]
    assert stub.mock_calls == []
    assert "missing credentials" in capsys.readouterr().out


def test_the_image_carries_the_registry_for_the_container(monkeypatch):
    module, stub = _load_app(monkeypatch)
    built = stub.Image.debian_slim.return_value.apt_install.return_value
    added = built.pip_install.return_value.env.return_value.add_local_file
    added.assert_called_once_with(str(module.REGISTRY_SOURCE), module.REGISTRY_REMOTE)
    assert module.REGISTRY_SOURCE.is_file()
    # Modal places the app module in /root, beside the registry's copy.
    assert Path(module.REGISTRY_REMOTE).parent == Path("/root")


def test_the_registry_loads_from_a_bare_copy(monkeypatch, tmp_path):
    """In the container only the copied file exists, with no microcosm."""

    module, _ = _load_app(monkeypatch)
    copy = tmp_path / "gate_signing_keys.py"
    shutil.copyfile(module.REGISTRY_SOURCE, copy)
    loaded = module.load_gate_signing_keys(copy)
    assert loaded.UK_GATE_SIGNING_KEY_ENV == UK_GATE_SIGNING_KEY_ENV
    assert set(loaded.UK_GATE_SIGNING_KEYS) == set(module.registry.UK_GATE_SIGNING_KEYS)
