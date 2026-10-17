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


def test_a_build_refuses_an_engine_other_than_the_pinned_one(monkeypatch) -> None:
    from microcosm.build.uk_runtime import hmrc_uprating

    pinned = str(hmrc_uprating.engine_parameter_pins()["engine"]["version"])
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: pinned)
    assert hmrc_uprating.assert_installed_engine_is_pinned() == pinned
    # The freeze: a later release in the environment refuses the build, and
    # so does a post-release of the pinned one, since the match is exact.
    for installed in ("9.9.9", f"{pinned}.post1"):
        monkeypatch.setattr(
            hmrc_uprating, "installed_engine_version", lambda v=installed: v
        )
        with pytest.raises(RuntimeError, match=f"{installed} is installed"):
            hmrc_uprating.assert_installed_engine_is_pinned()
    # Without the engine, the caller's own import reports it.
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: "absent")
    monkeypatch.setattr(hmrc_uprating, "_engine_importable", lambda: False)
    assert hmrc_uprating.assert_installed_engine_is_pinned() == "absent"
    # An engine that imports without distribution metadata has no version to
    # check, so it is refused too.
    monkeypatch.setattr(hmrc_uprating, "_engine_importable", lambda: True)
    with pytest.raises(RuntimeError, match="no distribution metadata"):
        hmrc_uprating.assert_installed_engine_is_pinned()


def test_the_build_and_the_certifier_refuse_an_unpinned_engine(monkeypatch) -> None:
    import sys
    import types
    from datetime import date
    from pathlib import Path

    from microcosm.build.uk_runtime import hmrc_uprating, spine_build
    from microcosm.build.uk_runtime.release_certification import (
        run_uk_release_cut_battery,
    )

    # The import check passes (a stand-in where the tier carries no engine);
    # the installed version is a later release.
    monkeypatch.setitem(
        sys.modules, "policyengine_uk", types.ModuleType("policyengine_uk")
    )
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: "9.9.9")
    # The spine build's engine, which prepare_full_build and
    # prepare_national_build also call before they solve.
    with pytest.raises(RuntimeError, match="9.9.9 is installed"):
        spine_build._rules_engine()
    # The release-cut battery the certifier runs refuses before any gate.
    with pytest.raises(RuntimeError, match="9.9.9 is installed"):
        run_uk_release_cut_battery(
            None,
            report_path=Path("unused.json"),
            release_id="unused",
            diagnostics_sha256="0" * 64,
            coverage_engine=None,
            build_stage_names=(),
            ledger_registries={},
            local_ledger_registries={},
            ledger_unsupported={},
            local_ledger_unsupported={},
            parity_evidence=None,
            fit_weight_records=None,
            input_mass_reference={},
            exclusions_evaluated_on=date(2026, 10, 18),
        )
