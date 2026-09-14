"""Retired rowwise command delegates exactly to the canonical full build."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from microcosm.build.uk_runtime import full_build_cli


def _load_builder_module():
    path = (
        Path(__file__).resolve().parents[3] / "tools" / "build_uk_rowwise_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _arguments(tmp_path):
    return [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--ladder",
        str(tmp_path / "ladder.parquet"),
        "--ledger-facts",
        str(tmp_path / "chronicle"),
        "--out",
        str(tmp_path / "out"),
    ]


def test_compatibility_command_is_exact_full_build_redirect(tmp_path, monkeypatch):
    seen = []

    def execute(prepared, args):
        seen.append((prepared, args))
        return 17

    monkeypatch.setattr(full_build_cli, "prepare_full_build", lambda args: "prepared")
    monkeypatch.setattr(full_build_cli, "execute_full_build", execute)
    module = _load_builder_module()
    assert module.main is full_build_cli.main
    assert module.main(_arguments(tmp_path)) == 17
    assert seen[0][1].target_geographies is None


def test_country_only_is_an_explicit_filter_in_the_same_command(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(full_build_cli, "prepare_full_build", lambda args: args)
    monkeypatch.setattr(
        full_build_cli,
        "execute_full_build",
        lambda prepared, args: seen.append(args) or 0,
    )
    assert (
        _load_builder_module().main(
            _arguments(tmp_path) + ["--target-geographies", "country"]
        )
        == 0
    )
    assert seen[0].target_geographies == ("country",)


@pytest.mark.parametrize(
    "retired",
    [
        "--constituency-household-targets",
        "--staging-h5",
        "--allow-unpinned-feed",
        "--logbook-dir",
    ],
)
def test_retired_options_refuse_instead_of_selecting_a_legacy_route(
    tmp_path, capsys, retired
):
    with pytest.raises(SystemExit) as error:
        _load_builder_module().main(_arguments(tmp_path) + [retired, "legacy"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
