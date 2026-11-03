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
    import re

    from microcosm.build.uk_runtime import hmrc_uprating

    pinned = str(hmrc_uprating.engine_parameter_pins()["engine"]["version"])
    # Nothing imports: the pinned metadata alone passes, and the caller's own
    # import reports the missing module.
    monkeypatch.setattr(hmrc_uprating, "_engine_origin", lambda: None)
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: pinned)
    assert hmrc_uprating.assert_installed_engine_is_pinned() == pinned
    # The freeze: a later release, and a post-release or a local build of the
    # pinned one, are refused, since the match is exact.
    for installed in ("9.9.9", f"{pinned}.post1", f"{pinned}+local"):
        monkeypatch.setattr(
            hmrc_uprating, "installed_engine_version", lambda v=installed: v
        )
        with pytest.raises(RuntimeError, match=re.escape(f"{installed} is installed")):
            hmrc_uprating.assert_installed_engine_is_pinned()
    # Without the engine, the caller's own import reports it.
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: "absent")
    assert hmrc_uprating.assert_installed_engine_is_pinned() == "absent"
    # An engine that imports without distribution metadata has no version to
    # check, so it is refused too.
    monkeypatch.setattr(
        hmrc_uprating,
        "_engine_origin",
        lambda: "/elsewhere/policyengine_uk/__init__.py",
    )
    with pytest.raises(RuntimeError, match="no distribution metadata"):
        hmrc_uprating.assert_installed_engine_is_pinned()


def test_the_imported_engine_must_be_the_distributions_own(
    monkeypatch, tmp_path
) -> None:
    from microcosm.build.uk_runtime import hmrc_uprating

    pinned = str(hmrc_uprating.engine_parameter_pins()["engine"]["version"])
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: pinned)
    own = tmp_path / "site" / "policyengine_uk" / "__init__.py"
    own.parent.mkdir(parents=True)
    own.write_text("RELEASE = 'pinned'\n")
    shadow = tmp_path / "shadow" / "policyengine_uk" / "__init__.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("RELEASE = 'other'\n")
    record = hmrc_uprating._record_sha256(str(own))
    monkeypatch.setattr(
        hmrc_uprating, "_engine_distribution_file", lambda: (str(own), record)
    )
    # The distribution's own files, matching its RECORD: passes.
    monkeypatch.setattr(hmrc_uprating, "_engine_origin", lambda: str(own))
    assert hmrc_uprating.assert_installed_engine_is_pinned() == pinned
    # A copy imported from elsewhere: refused.
    monkeypatch.setattr(hmrc_uprating, "_engine_origin", lambda: str(shadow))
    with pytest.raises(RuntimeError, match="shadowing copy"):
        hmrc_uprating.assert_installed_engine_is_pinned()
    # The pinned record beside other files: refused.
    own.write_text("RELEASE = 'other'\n")
    monkeypatch.setattr(hmrc_uprating, "_engine_origin", lambda: str(own))
    with pytest.raises(RuntimeError, match="RECORD"):
        hmrc_uprating.assert_installed_engine_is_pinned()


def test_a_copy_earlier_on_the_path_is_found_and_refused(monkeypatch, tmp_path) -> None:
    import sys
    from pathlib import Path

    from microcosm.build.uk_runtime import hmrc_uprating

    pinned = str(hmrc_uprating.engine_parameter_pins()["engine"]["version"])
    shadow = tmp_path / "policyengine_uk"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "policyengine_uk", raising=False)
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: pinned)
    origin = hmrc_uprating._engine_origin()
    assert Path(origin).resolve() == (shadow / "__init__.py").resolve()
    monkeypatch.setattr(
        hmrc_uprating,
        "_engine_distribution_file",
        lambda: (
            str(tmp_path / "site-packages" / "policyengine_uk" / "__init__.py"),
            None,
        ),
    )
    with pytest.raises(RuntimeError, match="shadowing copy"):
        hmrc_uprating.assert_installed_engine_is_pinned()


def test_a_module_without_a_spec_is_refused_by_the_guard(monkeypatch) -> None:
    import sys
    import types

    from microcosm.build.uk_runtime import hmrc_uprating

    pinned = str(hmrc_uprating.engine_parameter_pins()["engine"]["version"])
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: pinned)
    monkeypatch.setitem(
        sys.modules, "policyengine_uk", types.ModuleType("policyengine_uk")
    )
    with pytest.raises(RuntimeError, match="without a module spec"):
        hmrc_uprating.assert_installed_engine_is_pinned()


def test_the_hmrc_helpers_factory_refuses_an_unpinned_engine(monkeypatch) -> None:
    import sys
    import types

    from microcosm.build.uk_runtime import hmrc_calibration, hmrc_uprating

    stub = types.ModuleType("policyengine_uk")
    stub.Microsimulation = lambda dataset: dataset
    monkeypatch.setitem(sys.modules, "policyengine_uk", stub)
    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: "9.9.9")
    with pytest.raises(RuntimeError, match="9.9.9 is installed"):
        hmrc_calibration._default_simulation_factory(object())


def test_the_build_preparations_refuse_before_reading_any_input(monkeypatch) -> None:
    import argparse

    from microcosm.build.uk_runtime import full_build_cli, hmrc_uprating

    monkeypatch.setattr(hmrc_uprating, "installed_engine_version", lambda: "9.9.9")
    for prepare in (
        full_build_cli.prepare_full_build,
        full_build_cli.prepare_national_build,
    ):
        # An empty request: the guard refuses before any argument is read.
        with pytest.raises(RuntimeError, match="9.9.9 is installed"):
            prepare(argparse.Namespace())


def test_a_spine_built_on_another_engine_is_refused() -> None:
    from microcosm.build.uk_runtime import calibration_run, hmrc_uprating

    check = calibration_run._assert_spine_engine_is_pinned
    pinned = hmrc_uprating.pinned_engine_version()
    check({"rules_engine": {"package": "policyengine-uk", "version": pinned}})
    # A sidecar that records no engine, or a synthetic "unavailable" one, is
    # left to the preflight's stage roster.
    for unrecorded in ({}, {"rules_engine": {"version": "unavailable"}}):
        check(unrecorded)
    with pytest.raises(ValueError, match="built on policyengine-uk 9.9.9"):
        check({"rules_engine": {"package": "policyengine-uk", "version": "9.9.9"}})


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
