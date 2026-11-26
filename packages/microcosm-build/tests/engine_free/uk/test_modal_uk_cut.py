"""tools/modal_uk_cut.py, imported against a stub ``modal`` module."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

ROOT = Path(__file__).resolve().parents[5]


def _load_app(monkeypatch, tool="modal_uk_cut.py"):
    """A Modal tool as the launcher sees it: ``modal.is_local()`` is true."""

    stub = MagicMock(name="modal")
    stub.is_local.return_value = True
    stub.App.return_value.function.side_effect = lambda **options: (
        lambda function: setattr(function, "modal_options", options) or function
    )
    stub.App.return_value.local_entrypoint.side_effect = lambda **_: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", stub)
    spec = importlib.util.spec_from_file_location(
        f"{tool.removesuffix('.py')}_under_test", ROOT / "tools" / tool
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


def _git_answering(module, commit_minimum):
    """``subprocess.run`` for ``plan``: the commit is on main, and its
    ``minimum_emitter_version.json`` names ``commit_minimum`` (``None``: the
    commit has no such file, as before microcosm#1522)."""

    def run(argv, **kwargs):
        if argv[:2] == ["git", "show"]:
            if commit_minimum is None:
                return module.subprocess.CompletedProcess(
                    argv, 128, stdout="", stderr="fatal: path does not exist"
                )
            return module.subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps({"minimum_emitter_version": commit_minimum}),
                stderr="",
            )
        return module.subprocess.CompletedProcess(argv, 0)

    return run


def _launching_minimum():
    from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION

    return MINIMUM_EMITTER_VERSION


def test_planning_contacts_no_modal_service(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv(UK_GATE_SIGNING_KEY_ENV, raising=False)
    module, stub = _load_app(monkeypatch)
    monkeypatch.setattr(
        module.subprocess, "run", _git_answering(module, _launching_minimum())
    )
    stub.reset_mock()
    assert module.plan("a" * 40, "t") == [UK_GATE_SIGNING_KEY_ENV]
    assert stub.mock_calls == []
    assert "missing credentials" in capsys.readouterr().out


def _image_steps(stub, step):
    """The calls named ``step`` the module made on the stub, in order."""

    return [entry for entry in stub.mock_calls if entry[0].split(".")[-1] == step]


def _image_chain(stub):
    """Each image-definition call as (step name, arguments), in order."""

    return [
        (name.rsplit(".", 1)[-1], args)
        for name, args, _ in stub.mock_calls
        if name.startswith("Image.debian_slim")
    ]


def _emitter_installs(stub):
    return [
        step.args
        for step in _image_steps(stub, "run_commands")
        if any("microcosm-emitter" in command for command in step.args)
    ]


def _load_us_plan(monkeypatch):
    """The US Modal stage runner's plan module, loaded by path like the tools."""

    spec = importlib.util.spec_from_file_location(
        "modal_us_stage_plan_beside_uk_launcher",
        ROOT / "tools" / "modal_us_stage_plan.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve string annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_the_image_carries_the_registry_for_the_container(monkeypatch):
    module, stub = _load_app(monkeypatch)
    # The registry's copy is the last step, called on the image the emitter's
    # install returns: Modal refuses build steps after a local file.
    added = [
        entry
        for entry in stub.mock_calls
        if entry[0]
        == (
            "Image.debian_slim().apt_install().run_commands().pip_install()"
            ".env().run_commands().add_local_file"
        )
    ]
    assert [entry.args for entry in added] == [
        (str(module.REGISTRY_SOURCE), module.REGISTRY_REMOTE)
    ]
    assert _image_chain(stub)[-1][0] == "add_local_file"
    assert module.REGISTRY_SOURCE.is_file()
    # Modal places the app module in /root, beside the registry's copy.
    assert Path(module.REGISTRY_REMOTE).parent == Path("/root")


def test_the_image_trusts_no_compromised_certificate_authority(monkeypatch):
    """As the US Modal image does, before anything is downloaded.

    The compromised eMudhra (emSign) roots: uv before 0.13.4 compiles them in,
    the cached base's pip 25.3 bundles them, and the base's CA store, which
    git and the standard library read, holds them until it is upgraded.
    """

    us_plan = _load_us_plan(monkeypatch)
    _, stub = _load_app(monkeypatch)
    chain = _image_chain(stub)
    assert [name for name, _ in chain] == [
        "debian_slim",
        "apt_install",
        "run_commands",
        "pip_install",
        "env",
        "run_commands",
        "add_local_file",
    ]
    assert "ca-certificates" in chain[1][1]
    assert chain[2][1] == (us_plan.image_pip_bootstrap_command(),)
    assert chain[3][1] == (f"uv=={us_plan.IMAGE_UV_VERSION}",)


def test_the_image_installs_the_emitter_at_the_builds_minimum_release(monkeypatch):
    """The pre-flight starts the separately installed emitter (#1410, #1460)."""

    _, stub = _load_app(monkeypatch)
    assert _emitter_installs(stub) == [
        (
            "UV_TOOL_BIN_DIR=/usr/local/bin uv tool install "
            f"'microcosm-emitter>={_launching_minimum()}'",
            "/usr/local/bin/microcosm-emitter version",
        )
    ]


def test_the_build_is_pointed_at_the_installed_emitter(monkeypatch):
    _, stub = _load_app(monkeypatch)
    (environment,) = _image_steps(stub, "env")
    assert environment.args[0].get("MICROCOSM_EMITTER_COMMAND") == (
        "/usr/local/bin/microcosm-emitter"
    )


def test_uv_never_downloads_a_python_in_the_image(monkeypatch):
    """The image's 3.13 meets the lock and the emitter, as CI tests the lock."""

    _, stub = _load_app(monkeypatch)
    (environment,) = _image_steps(stub, "env")
    assert environment.args[0].get("UV_PYTHON_DOWNLOADS") == "never"


def test_the_launchers_agree_on_the_emitter(monkeypatch):
    us_plan = _load_us_plan(monkeypatch)
    module, _ = _load_app(monkeypatch)
    assert module.emitter_requirement() == us_plan.image_emitter_requirement()
    assert module.EMITTER_COMMAND == us_plan.IMAGE_EMITTER_COMMAND
    assert module.EMITTER_COMMAND_ENV == us_plan.EMITTER_COMMAND_ENV


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        "[]",
        '{"minimum_emitter_version": 17}',
        '{"minimum_emitter_version": "0.17"}',
        '{"minimum_emitter_version": "0.17.0 "}',
    ],
)
def test_an_unreadable_minimum_release_stops_the_image_definition(
    monkeypatch, tmp_path, content
):
    module, _ = _load_app(monkeypatch)
    version_file = tmp_path / "minimum_emitter_version.json"
    if content is not None:
        version_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        module, "MINIMUM_EMITTER_VERSION_FILE", version_file, raising=False
    )
    with pytest.raises(SystemExit, match=re.escape(str(version_file))):
        module._image()


def test_the_container_defines_the_image_without_the_checkout(monkeypatch, tmp_path):
    """The container imports the module at /root, with no checkout around it."""

    module, stub = _load_app(monkeypatch)
    monkeypatch.setattr(
        module,
        "MINIMUM_EMITTER_VERSION_FILE",
        tmp_path / "missing" / "minimum_emitter_version.json",
        raising=False,
    )
    stub.is_local.return_value = False
    stub.reset_mock()
    module._image()
    assert _emitter_installs(stub) == []
    assert _image_steps(stub, "add_local_file") == []


def test_the_plan_names_the_emitter_the_image_installs(monkeypatch, capsys):
    module, _ = _load_app(monkeypatch)
    monkeypatch.setattr(
        module.subprocess, "run", _git_answering(module, _launching_minimum())
    )
    module.plan("a" * 40, "t")
    assert f"emitter microcosm-emitter>={_launching_minimum()}" in (
        capsys.readouterr().out
    )


def test_the_plan_refuses_a_commit_that_needs_a_newer_emitter(monkeypatch):
    """Preflight in the paid container would refuse the image's emitter."""

    module, _ = _load_app(monkeypatch)
    monkeypatch.setattr(module.subprocess, "run", _git_answering(module, "99.0.0"))
    with pytest.raises(SystemExit, match=r"99\.0\.0.*checkout at or after"):
        module.plan("a" * 40, "t")


def test_the_plan_refuses_a_commit_whose_minimum_it_cannot_read(monkeypatch):
    module, _ = _load_app(monkeypatch)
    monkeypatch.setattr(module.subprocess, "run", _git_answering(module, "0.17"))
    with pytest.raises(SystemExit, match="aaaaaaaaa"):
        module.plan("a" * 40, "t")


@pytest.mark.parametrize("commit_minimum", ["launching", "0.13.0", None])
def test_the_plan_accepts_a_commit_the_image_emitter_satisfies(
    monkeypatch, commit_minimum
):
    """The same minimum, an older one, or none: commits before microcosm#1522
    kept an older minimum in code."""

    module, _ = _load_app(monkeypatch)
    if commit_minimum == "launching":
        commit_minimum = _launching_minimum()
    monkeypatch.setattr(
        module.subprocess, "run", _git_answering(module, commit_minimum)
    )
    module.plan("a" * 40, "t")


def test_the_registry_loads_from_a_bare_copy(monkeypatch, tmp_path):
    """In the container only the copied file exists, with no microcosm."""

    module, _ = _load_app(monkeypatch)
    copy = tmp_path / "gate_signing_keys.py"
    shutil.copyfile(module.REGISTRY_SOURCE, copy)
    loaded = module.load_gate_signing_keys(copy)
    assert loaded.UK_GATE_SIGNING_KEY_ENV == UK_GATE_SIGNING_KEY_ENV
    assert set(loaded.UK_GATE_SIGNING_KEYS) == set(module.registry.UK_GATE_SIGNING_KEYS)


def test_an_opted_in_run_without_run_says_the_credentials_were_sent(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("MICROCOSM_UK_CUT_MODAL_RUN", "1")
    module, _ = _load_app(monkeypatch)
    with pytest.raises(SystemExit, match="already sent to Modal"):
        module.main("a" * 40, "t")


def test_missing_credentials_are_asked_for_one_command(monkeypatch):
    monkeypatch.setenv("MICROCOSM_UK_CUT_MODAL_RUN", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv(UK_GATE_SIGNING_KEY_ENV, raising=False)
    module, _ = _load_app(monkeypatch)
    monkeypatch.setattr(
        module.subprocess, "run", _git_answering(module, _launching_minimum())
    )
    with pytest.raises(SystemExit, match="to this command") as refusal:
        module.main("a" * 40, "t", run=True)
    assert "export " not in str(refusal.value).replace("exported", "")


def test_the_container_syncs_without_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, "key-test")
    module, _ = _load_app(monkeypatch)
    synced, ran = [], []
    monkeypatch.setattr(
        module, "_sh", lambda argv, **kw: synced.append((argv, kw.get("env")))
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kw: (
            ran.append(kw["env"]) or module.subprocess.CompletedProcess(argv, 0)
        ),
    )
    assert module.cut("a" * 40, "t", False, []) == 0
    sync_env = next(env for argv, env in synced if argv[:2] == ["uv", "sync"])
    assert "HF_TOKEN" not in sync_env and UK_GATE_SIGNING_KEY_ENV not in sync_env
    assert ran[0]["HF_TOKEN"] == "hf-test"
    assert ran[0][UK_GATE_SIGNING_KEY_ENV] == "key-test"


CHRONICLE = "modal_rebuild_uk_chronicle_artifact.py"


def test_the_chronicle_rebuild_sends_no_token_without_the_opt_in(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD", raising=False)
    module, stub = _load_app(monkeypatch, CHRONICLE)
    stub.Secret.from_dict.assert_not_called()
    assert module.rebuild.modal_options["secrets"] == []
    with pytest.raises(SystemExit, match="paid rebuild"):
        module.main()
    with pytest.raises(SystemExit, match="MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD"):
        module.main(build=True, upload=True)


def test_an_opted_in_chronicle_upload_attaches_the_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("MICROCOSM_UK_CHRONICLE_MODAL_UPLOAD", "1")
    module, stub = _load_app(monkeypatch, CHRONICLE)
    stub.Secret.from_dict.assert_called_once_with({"HF_TOKEN": "hf-test"})
    with pytest.raises(SystemExit, match="already sent to Modal"):
        module.main()


def test_chronicle_planning_contacts_no_modal_service(monkeypatch, capsys):
    module, stub = _load_app(monkeypatch, CHRONICLE)
    stub.reset_mock()
    summary = module.plan()
    assert summary["chronicle_commit"] == module.PIN["source_commit"]
    assert stub.mock_calls == []
