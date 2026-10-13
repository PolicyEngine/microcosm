"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating_engine_pins.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating_engine_pins import *


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
