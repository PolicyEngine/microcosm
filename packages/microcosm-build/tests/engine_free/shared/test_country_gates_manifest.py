"""The narrow gates loader and the per-country resource helper (#1585 item 11).

A build tool that needs only its gate policy reads ``<cc>/gates.json`` through
``load_gates_manifest``, without the full country-spec load; bindings read the
registers their declared parameters name through ``country_resource``.
"""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.country_spec import (
    country_resource,
    load_country_spec,
    load_gates_manifest,
)


def _package_copy(tmp_path: Path, country: str) -> Path:
    root = tmp_path / country
    shutil.copytree(Path(str(files("microcosm.build").joinpath(country))), root)
    return root


@pytest.mark.parametrize("country", ["uk", "be", "am", "nz"])
def test_the_narrow_loader_returns_the_full_loaders_manifest(country) -> None:
    assert load_gates_manifest(country) == load_country_spec(country).gates


def test_the_narrow_loader_reads_no_other_resource(tmp_path) -> None:
    root = _package_copy(tmp_path, "be")
    (root / "undeclared.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not declared"):
        load_country_spec(root)
    assert load_gates_manifest(root) == load_country_spec("be").gates


def test_the_narrow_loader_refuses_what_the_full_loader_refuses(tmp_path) -> None:
    # The US package declares no gates.json yet; the loader never falls back.
    with pytest.raises(ValueError, match="no legacy_json gates.json"):
        load_gates_manifest("us")
    root = _package_copy(tmp_path, "be")
    gates = json.loads((root / "gates.json").read_text(encoding="utf-8"))
    gates["phases"] = ["exported"]
    (root / "gates.json").write_text(json.dumps(gates), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown phase"):
        load_gates_manifest(root)
    (root / "gates.json").unlink()
    with pytest.raises(FileNotFoundError, match="gates.json"):
        load_gates_manifest(root)


def test_country_resource_is_the_packaged_file() -> None:
    assert (
        country_resource("uk", "gates.json").read_text()
        == files("microcosm.build.uk").joinpath("gates.json").read_text()
    )
    for country in ("UK", "u.k", "", "uk/x"):
        with pytest.raises(ValueError, match="country code"):
            country_resource(country, "gates.json")
