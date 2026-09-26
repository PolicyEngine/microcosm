"""The declared UK uprating appliers read vendored engine values, not the engine.

The fast CI tiers carry no country package, so the compile of the fixture
subset must run without policyengine-uk; the pins file carries the values and
the engine version they came from, and the ``requires_uk`` test holds it in
lockstep with the installed engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.uk_runtime import hmrc_uprating as hu

_PINS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "microcosm"
    / "build"
    / "uk"
    / hu.UK_ENGINE_PINS_RESOURCE
)


def test_the_compile_reads_the_vendored_pins_without_the_engine(monkeypatch) -> None:
    def refuse() -> None:
        raise AssertionError("the compile must not construct the engine")

    monkeypatch.setattr(hu, "_engine_system", refuse)
    pins = hu.engine_parameter_pins()
    assert pins["schema_version"] == 1
    assert pins["engine"]["package"] == "policyengine-uk"
    assert tuple(pins["values"]) == hu.UK_ENGINE_INDEX_PARAMETERS
    assert pins["instants"] == [f"{year}-01-01" for year in hu.UK_ENGINE_PIN_YEARS]
    for path in hu.UK_ENGINE_INDEX_PARAMETERS:
        assert list(pins["values"][path]) == pins["instants"]
        for instant in pins["instants"]:
            assert hu.engine_parameter_value(path, instant) > 0
    assert hu._engine_version() == pins["engine"]["version"]
    assert json.loads(_PINS_PATH.read_text(encoding="utf-8")) == pins


def test_an_instant_outside_the_pins_is_refused_by_name() -> None:
    path = hu.UK_ENGINE_INDEX_PARAMETERS[0]
    with pytest.raises(ValueError, match="not in the vendored engine pins"):
        hu.engine_parameter_value(path, "1999-01-01")


def test_a_pins_file_for_the_wrong_roster_is_refused(monkeypatch) -> None:
    hu.engine_parameter_pins.cache_clear()
    payload = json.loads(_PINS_PATH.read_text(encoding="utf-8"))
    payload["values"].pop(hu.UK_ENGINE_INDEX_PARAMETERS[-1])

    class _Resource:
        def read_text(self, encoding: str) -> str:
            return json.dumps(payload)

    class _Package:
        def joinpath(self, name: str) -> _Resource:
            assert name == hu.UK_ENGINE_PINS_RESOURCE
            return _Resource()

    monkeypatch.setattr(hu, "files", lambda package: _Package())
    try:
        with pytest.raises(
            ValueError, match="does not carry schema 1 policyengine-uk pins"
        ):
            hu.engine_parameter_pins()
    finally:
        hu.engine_parameter_pins.cache_clear()


@pytest.mark.requires_uk
def test_the_vendored_pins_match_the_installed_engine() -> None:
    pins = hu.engine_parameter_pins()
    assert pins["engine"]["version"] == hu.installed_engine_version()
    for path, by_instant in pins["values"].items():
        for instant, value in by_instant.items():
            assert hu.live_engine_parameter_value(path, instant) == float(value), (
                path,
                instant,
            )
