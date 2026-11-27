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
    part = _unit(pool_module, batch_start=1, batch_stop=3)
    assert part.describe().endswith(", batches 2-3 of 4")
    for start, stop in ((0, 0), (3, 2), (-1, 2), (0, 5)):
        with pytest.raises(ValueError, match="non-empty range"):
            _unit(pool_module, batch_start=start, batch_stop=stop)
    # Baseline passes run in the certifying process, never in a worker.
    for consumer in (None, "reform_validation"):
        with pytest.raises(ValueError, match="baseline passes run in the certifying"):
            _unit(
                pool_module,
                consumer=consumer,
                reformed_simulation=None,
                reform=None,
                passes=(pool_module.PostExportPass("merged baseline plan", (KEY,)),),
            )
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


# ---------------------------------------------------------------------------
# The default worker settings (tracker decision 1.4 · 6, approved 27 November)
# ---------------------------------------------------------------------------

GIB = 2**30
VARIABLE = "MICROCOSM_POST_EXPORT_WORKERS"


def _plan(pool_module, memory, cores, value=None):
    environment = {} if value is None else {VARIABLE: value}
    return pool_module.plan_post_export_workers(
        environment, total_memory_bytes=int(memory), usable_cores=cores
    )


@pytest.mark.parametrize(
    ("memory", "cores", "workers", "cap"),
    [
        # 1.4 · 6's table: the 32 GiB, 6-core design target, Anthony's Mac,
        # the build Mac, and a 16 GiB machine.
        (32 * GIB, 6, 4, 21 * GIB // 4),
        (32 * GIB, 10, 4, 21 * GIB // 4),
        (128 * GIB, 18, 17, 93 * GIB // 17),
        (16 * GIB, 8, 1, 9 * GIB),
        # The memory term reaches 4 from 30.67 GiB, so a Linux host's 31.3
        # GiB keeps 4; a container given 32 * 10**9 bytes (29.8 GiB) gets 3.
        (int(31.3 * GIB), 10, 4, (3 * int(31.3 * GIB) // 4 - 3 * GIB) // 4),
        (32 * 10**9, 10, 3, (3 * 32 * 10**9 // 4 - 3 * GIB) // 3),
        # The cores term: usable cores less one, at least one.
        (32 * GIB, 3, 2, 21 * GIB // 2),
        (32 * GIB, 1, 1, 21 * GIB),
    ],
)
def test_the_formula_picks_the_workers_and_each_ones_cap(
    pool_module, memory, cores, workers, cap
) -> None:
    for value in (None, "", "auto"):
        plan = _plan(pool_module, memory, cores, value)
        assert not plan.serial
        assert plan.workers == workers
        assert plan.chosen_by == "formula"
        assert plan.memory_cap_bytes == max(5 * GIB, cap)
        assert plan.max_units_per_worker == 2
        assert plan.budget_bytes == int(memory) * 3 // 4
        assert plan.planned_peak_bytes == 3 * GIB + workers * 5 * GIB
        settings = plan.pool_settings()
        assert settings.workers == workers
        assert settings.memory_cap_bytes == plan.memory_cap_bytes
        assert settings.max_units_per_worker == 2
        # Workers inherit the parent's thread variables (1.4 · 6, part 5).
        assert settings.worker_environment is None
        assert (settings.max_attempts, settings.max_units_ahead) == (2, None)


def test_a_machine_too_small_for_one_worker_is_refused_unless_told(
    pool_module,
) -> None:
    """Under about 10.7 GiB, 75% of memory less this process's 3 GiB leaves no
    room for one 5 GiB worker: refused, before anything is loaded, unless the
    operator names a count or serial scoring."""
    for value in (None, "auto"):
        with pytest.raises(pool_module.PostExportWorkersError) as refused:
            _plan(pool_module, 10 * GIB, 8, value)
        message = str(refused.value)
        assert "post-export scoring" in message
        assert "8.00 GiB" in message and "7.50 GiB" in message
        assert VARIABLE in message and "serial" in message
    assert _plan(pool_module, 10.7 * GIB, 8).workers == 1
    explicit = _plan(pool_module, 10 * GIB, 8, "2")
    assert (explicit.workers, explicit.chosen_by) == (2, f"{VARIABLE}=2")
    assert explicit.memory_cap_bytes == 5 * GIB
    assert explicit.planned_peak_bytes > explicit.budget_bytes
    assert "above" in explicit.describe()
    assert _plan(pool_module, 10 * GIB, 8, "serial").serial


def test_an_operator_names_a_count_or_serial_scoring(pool_module) -> None:
    one = _plan(pool_module, 32 * GIB, 10, "1")
    # One pool worker, a separate process with the cap and replacement.
    assert (one.serial, one.workers) == (False, 1)
    assert one.memory_cap_bytes == 21 * GIB
    many = _plan(pool_module, 32 * GIB, 10, "12")
    assert many.workers == 12 and many.memory_cap_bytes == 5 * GIB
    serial = _plan(pool_module, 32 * GIB, 10, "serial")
    assert serial.serial and serial.workers is None
    assert serial.chosen_by == f"{VARIABLE}=serial"
    assert serial.pool_settings() is None
    assert "in this process" in serial.describe()


@pytest.mark.parametrize("value", ["0", "-1", "four", " 4", "4.0", "+4", "Serial"])
def test_any_other_setting_is_refused(pool_module, value) -> None:
    with pytest.raises(pool_module.PostExportWorkersError, match=VARIABLE):
        _plan(pool_module, 32 * GIB, 10, value)


def test_the_plan_says_what_it_chose_and_why(pool_module) -> None:
    plan = _plan(pool_module, 32 * GIB, 10)
    assert plan.record() == {
        "mode": "pooled",
        "chosen_by": "formula",
        "workers": 4,
        "total_memory_bytes": 32 * GIB,
        "usable_cores": 10,
        "budget_bytes": 24 * GIB,
        "planned_peak_bytes": 23 * GIB,
        "memory_cap_bytes": 21 * GIB // 4,
        "max_units_per_worker": 2,
    }
    line = plan.describe()
    for text in ("4 pool workers", "formula", "23.00 GiB", "24.00 GiB", "5.25 GiB"):
        assert text in line, text


def test_a_worker_refuses_a_release_tool_other_than_the_parents(
    pool_module, tmp_path
) -> None:
    """A worker scores only with the code the scoring process runs: a tool
    file that changed after the parent loaded it is refused at start."""
    import hashlib

    tool = tmp_path / "fixture_checked_tool.py"
    tool.write_text("VALUE = 1\n")
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    try:
        with pytest.raises(RuntimeError, match="SHA-256"):
            pool_module.release_tool("fixture_checked_tool", tool, sha256="0" * 64)
        assert "fixture_checked_tool" not in sys.modules
        loaded = pool_module.release_tool("fixture_checked_tool", tool, sha256=digest)
        assert loaded.VALUE == 1
        # Found loaded already, it is the bytes it ran that count: those are
        # the code the worker would score with, whatever the file holds now.
        tool.write_text("VALUE = 2\n")
        assert (
            pool_module.release_tool("fixture_checked_tool", tool, sha256=digest)
            is loaded
        )
        # A tool imported without a record of its bytes (certification's
        # import by name) is checked against its file.
        del loaded.__loaded_sha256__
        with pytest.raises(RuntimeError, match="SHA-256"):
            pool_module.release_tool("fixture_checked_tool", tool, sha256=digest)
    finally:
        sys.modules.pop("fixture_checked_tool", None)
