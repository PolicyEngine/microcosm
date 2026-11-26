"""The units, results and worker entry points of pooled post-export scoring.

``microcosm.build.us_runtime.post_export_pool`` holds what the release tool's
pooled scorer sends to pool workers and what they send back. These tests
check its own rules: what a pass and a unit accept, how a failure a worker
returns is raised again in the parent, and how a worker finds the release
tool. The scoring itself is tested in ``test_us_post_export_pooled_scoring.py``.
"""

from __future__ import annotations

import pickle
import sys

import pytest

from microcosm.build.us_runtime.post_export_requests import ParameterChangesReform

KEY = ("income_tax", 2024, None)
LATER_KEY = ("income_tax", 2026, None)


@pytest.fixture
def pool_module():
    import microcosm.build.us_runtime.post_export_pool as module

    return module


def _unit(pool_module, **overrides):
    fields = {
        "consumer": "reform_validation",
        "reformed_simulation": 2,
        "reform": ParameterChangesReform({"gov.irs.fixture": {"2024-01-01": 1}}),
        "passes": (pool_module.PostExportPass("reform_validation reform x", (KEY,)),),
        "batch_start": 0,
        "batch_stop": 4,
        "n_batches": 4,
        **overrides,
    }
    return pool_module.PostExportUnit(**fields)


def test_a_pass_holds_unique_keys_in_ascending_period_order(pool_module) -> None:
    scoring_pass = pool_module.PostExportPass("label", [KEY, LATER_KEY])
    assert scoring_pass.keys == (KEY, LATER_KEY)
    with pytest.raises(ValueError, match="ascending period order"):
        pool_module.PostExportPass("label", (LATER_KEY, KEY))
    with pytest.raises(ValueError, match="repeats a key"):
        pool_module.PostExportPass("label", (KEY, KEY))
    with pytest.raises(ValueError, match="has no keys"):
        pool_module.PostExportPass("label", ())
    with pytest.raises(ValueError, match="needs a label"):
        pool_module.PostExportPass("", (KEY,))
    for bad in (
        ("income_tax", "2024", None),
        ("income_tax", True, None),
        ("", 1, None),
    ):
        with pytest.raises(ValueError, match="post-export key"):
            pool_module.PostExportPass("label", (bad,))


def test_a_unit_names_its_batches_and_its_reform(pool_module) -> None:
    unit = _unit(pool_module)
    assert pickle.loads(pickle.dumps(unit)) == unit
    with pytest.raises(TypeError):
        hash(unit)
    assert unit.describe() == (
        "post-export reform_validation reformed simulation 2 (1 pass(es), first "
        "'reform_validation reform x'), all 4 batch(es)"
    )
    baseline = _unit(
        pool_module,
        consumer=None,
        reformed_simulation=None,
        reform=None,
        passes=(pool_module.PostExportPass("merged baseline plan", (KEY,)),),
        batch_start=1,
        batch_stop=3,
    )
    assert baseline.describe() == "post-export merged baseline plan, batches 2-3 of 4"
    for start, stop in ((0, 0), (3, 2), (-1, 2), (0, 5)):
        with pytest.raises(ValueError, match="non-empty range"):
            _unit(pool_module, batch_start=start, batch_stop=stop)
    with pytest.raises(ValueError, match="names no reformed simulation"):
        _unit(pool_module, reform=None)
    with pytest.raises(ValueError, match="names the consumer"):
        _unit(pool_module, consumer=None)
    with pytest.raises(TypeError, match="reformed simulation's number"):
        _unit(pool_module, reformed_simulation=None)
    with pytest.raises(TypeError, match="description"):
        _unit(pool_module, reform=object())
    with pytest.raises(ValueError, match="PostExportPass"):
        _unit(pool_module, passes=())


class _FixtureValueError(ValueError):
    pass


class _UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise TypeError("no text")


@pytest.mark.parametrize(
    ("error", "name", "base", "text"),
    [
        (KeyError("medicare_enrolled"), "KeyError", KeyError, "'medicare_enrolled'"),
        (
            _FixtureValueError("bad value"),
            "_FixtureValueError",
            ValueError,
            "bad value",
        ),
        (RuntimeError("fixture engine refuses x"), "RuntimeError", RuntimeError, None),
        (
            _UnprintableError(),
            "_UnprintableError",
            RuntimeError,
            "<str() of the _UnprintableError raised>",
        ),
    ],
)
def test_a_returned_failure_raises_with_the_workers_class_name_and_text(
    pool_module, error, name, base, text
) -> None:
    """A report that records ``f"{type(error).__name__}: {error}"`` records
    what the serial scorer would have, and ``except`` on the built-in base
    still catches it."""
    try:
        raise error
    except Exception as raised:
        failure = pool_module.PostExportUnitFailure.from_exception("scoring", raised)
    failure = pickle.loads(pickle.dumps(failure))
    rebuilt = failure.exception()
    assert type(rebuilt).__name__ == name
    assert isinstance(rebuilt, base)
    expected = str(error) if text is None else text
    assert str(rebuilt) == expected
    assert f"{type(rebuilt).__name__}: {rebuilt}" == f"{name}: {expected}"
    assert name in rebuilt.worker_traceback
    with pytest.raises(ValueError, match="Unknown post-export failure stage"):
        pool_module.PostExportUnitFailure.from_exception("elsewhere", error)


def test_a_worker_finds_a_loaded_release_tool_or_loads_it_once(
    pool_module, tmp_path, monkeypatch
) -> None:
    tool = tmp_path / "fixture_release_tool.py"
    tool.write_text("LOADS = globals().get('LOADS', 0) + 1\n")
    loaded = pool_module.release_tool("fixture_release_tool", tool)
    try:
        assert loaded.LOADS == 1
        assert sys.modules["fixture_release_tool"] is loaded
        # Found again, under its name or as the main script spawn re-ran.
        assert pool_module.release_tool("fixture_release_tool", tool) is loaded
        del sys.modules["fixture_release_tool"]
        monkeypatch.setitem(sys.modules, "__mp_main__", loaded)
        assert pool_module.release_tool("fixture_release_tool", tool) is loaded
        assert "fixture_release_tool" not in sys.modules
    finally:
        sys.modules.pop("fixture_release_tool", None)
    # A different module already holding the name is left alone.
    other = tmp_path / "other.py"
    other.write_text("LOADS = 0\n")
    monkeypatch.delitem(sys.modules, "__mp_main__")
    occupant = pool_module.release_tool("fixture_release_tool", other)
    try:
        fresh = pool_module.release_tool("fixture_release_tool", tool)
        assert fresh is not occupant and fresh.LOADS == 1
        assert sys.modules["fixture_release_tool"] is occupant
    finally:
        for name in [
            name
            for name, module in sys.modules.items()
            if getattr(module, "__file__", None) in (str(tool), str(other))
        ]:
            sys.modules.pop(name, None)
