"""QRF thread sizing: the default worker count, and draws that ignore it.

The draw's thread pool is sized by ``POPULACE_FIT_PREDICT_WORKERS``, else by
:func:`joblib.cpu_count`, which counts the CPUs the process may use (affinity
mask, cgroup CPU quota, ``LOKY_MAX_CPU_COUNT``) where :func:`os.cpu_count`
counts the machine's. Tree building is sized by ``POPULACE_FIT_N_JOBS``, else
by ``n_jobs=-1``, which joblib resolves to the same :func:`joblib.cpu_count`.

Neither count may change an imputed value. The equality tests fit and draw on
fixed seeds once with one fit job and one prediction worker, then again with
several of either, and require the float64 bytes of every draw to be equal.
Each run records the thread pool the draw used, so a draw that silently ran
on one thread fails, and the ``n_jobs`` each forest was configured with.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import pytest

from microcosm.fit import QRF, Regime, RegimeGatedQRF
from microcosm.fit import qrf as qrf_module
from microcosm.fit.qrf import _fit_n_jobs, _predict_workers

PREDICT_WORKERS_ENV = "POPULACE_FIT_PREDICT_WORKERS"
FIT_JOBS_ENV = "POPULACE_FIT_N_JOBS"

#: The "N" of the 1-versus-N comparisons, for both prediction workers and fit
#: jobs. Threads, so it need not be at most the CI runner's CPU count.
SEVERAL = 4

#: Donor and recipient row counts. With ``SEVERAL`` workers the draw splits a
#: recipient pass into ``4 * SEVERAL`` chunks of ceil(rows / 16) rows, the last
#: one shorter; with one worker it runs four larger chunks serially.
DONOR_ROWS = 1_200
RECIPIENT_ROWS = 1_001

N_ESTIMATORS = 8

#: (fit jobs, prediction workers) per run. The first is the serial baseline.
SERIAL = ("1", "1")
COMPARED_SETTINGS = (
    pytest.param((str(SEVERAL), "1"), id="several_fit_jobs"),
    pytest.param(("1", str(SEVERAL)), id="several_prediction_workers"),
    pytest.param((str(SEVERAL), str(SEVERAL)), id="several_of_both"),
)


# ----------------------------------------------------------------------------
# The default prediction worker count
# ----------------------------------------------------------------------------


def _cpu_count_unlike_os(monkeypatch: pytest.MonkeyPatch) -> int:
    """Patch :func:`joblib.cpu_count` to a value :func:`os.cpu_count` is not."""
    joblib_cpu_count = (os.cpu_count() or 1) + 3

    def patched_cpu_count(only_physical_cores: bool = False) -> int:
        # A physical-core count that differs, so a caller asking for it shows.
        return joblib_cpu_count + 100 if only_physical_cores else joblib_cpu_count

    monkeypatch.setattr(joblib, "cpu_count", patched_cpu_count)
    return joblib_cpu_count


@pytest.mark.parametrize("configured", [None, "", "  "])
def test_unset_prediction_workers_default_to_joblib_cpu_count(
    configured: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset or blank override resolves to ``joblib.cpu_count()``."""
    joblib_cpu_count = _cpu_count_unlike_os(monkeypatch)
    if configured is None:
        monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    else:
        monkeypatch.setenv(PREDICT_WORKERS_ENV, configured)

    assert _predict_workers() == joblib_cpu_count
    assert qrf_module.default_predict_workers() == joblib_cpu_count


@pytest.mark.parametrize(("configured", "expected"), [("1", 1), ("2", 2), ("7", 7)])
def test_prediction_worker_override_wins_over_joblib_cpu_count(
    configured: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POPULACE_FIT_PREDICT_WORKERS`` still takes precedence."""
    _cpu_count_unlike_os(monkeypatch)
    monkeypatch.setenv(PREDICT_WORKERS_ENV, configured)

    assert _predict_workers() == expected


@pytest.mark.parametrize("malformed", ["", "4.0", "four"])
def test_malformed_loky_max_cpu_count_is_named_in_the_error(
    malformed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """joblib refuses a non-integer ``LOKY_MAX_CPU_COUNT``; the error says so."""
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", malformed)

    with pytest.raises(ValueError, match="LOKY_MAX_CPU_COUNT"):
        _predict_workers()
    monkeypatch.setenv(PREDICT_WORKERS_ENV, "2")
    assert _predict_workers() == 2


@pytest.mark.parametrize("loky_max_cpu_count", ["1", "2"])
def test_unset_prediction_workers_follow_the_real_joblib_cpu_count(
    loky_max_cpu_count: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpatched joblib honours ``LOKY_MAX_CPU_COUNT``, so the draw does too."""
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", loky_max_cpu_count)

    width = _predict_workers()

    assert width == joblib.cpu_count()
    assert 1 <= width <= int(loky_max_cpu_count)


def test_fit_jobs_default_stays_minus_one_under_joblib_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit jobs keep the literal ``-1`` that chain states record.

    joblib resolves ``-1`` to its own CPU count when it builds trees, so the
    fit already follows :func:`joblib.cpu_count` without recording it.
    """
    _cpu_count_unlike_os(monkeypatch)
    monkeypatch.delenv(FIT_JOBS_ENV, raising=False)
    assert _fit_n_jobs() == -1
    monkeypatch.setenv(FIT_JOBS_ENV, "3")
    assert _fit_n_jobs() == 3


# ----------------------------------------------------------------------------
# Recording which thread pools a run used
# ----------------------------------------------------------------------------


class _PoolRecorder:
    """Records every draw thread pool's width and how many chunks it ran."""

    def __init__(self) -> None:
        self.pool_widths: list[int] = []
        self.chunks_per_pool: list[int] = []

    def executor_class(self) -> type[ThreadPoolExecutor]:
        recorder = self

        class RecordingThreadPoolExecutor(ThreadPoolExecutor):
            def __init__(self, max_workers: int, *args, **kwargs) -> None:
                super().__init__(max_workers, *args, **kwargs)
                recorder.pool_widths.append(max_workers)

            def map(self, function, chunk_bounds, *args, **kwargs) -> Iterator:
                chunk_bounds = list(chunk_bounds)
                recorder.chunks_per_pool.append(len(chunk_bounds))
                return super().map(function, chunk_bounds, *args, **kwargs)

        return RecordingThreadPoolExecutor


def _forest_fit_jobs(target_models: dict) -> list[int]:
    """The ``n_jobs`` of every forest in a fitted model's targets."""
    return [
        forest.model.n_jobs
        for target_model in target_models.values()
        for forest in (target_model.positive, target_model.negative)
        if forest is not None
    ]


def _run_under(
    monkeypatch: pytest.MonkeyPatch,
    setting: tuple[str, str],
    run: Callable[[], tuple[dict[str, np.ndarray], list[int]]],
) -> dict[str, np.ndarray]:
    """Run a fit and draw under one (fit jobs, prediction workers) setting.

    ``run`` returns the draws by name and the ``n_jobs`` its forests were fit
    with. The helper checks those against the setting, and checks that the
    draw used a thread pool of the configured width over several chunks, or
    no pool at all for one worker.
    """
    fit_jobs, predict_workers = setting
    monkeypatch.setenv(FIT_JOBS_ENV, fit_jobs)
    monkeypatch.setenv(PREDICT_WORKERS_ENV, predict_workers)
    recorder = _PoolRecorder()
    monkeypatch.setattr(qrf_module, "ThreadPoolExecutor", recorder.executor_class())

    draws, forest_fit_jobs = run()

    assert forest_fit_jobs, "the run fit no forest"
    assert set(forest_fit_jobs) == {int(fit_jobs)}
    if int(predict_workers) == 1:
        assert recorder.pool_widths == []
    else:
        assert recorder.pool_widths, "the draw never used its thread pool"
        assert set(recorder.pool_widths) == {int(predict_workers)}
        assert min(recorder.chunks_per_pool) > int(predict_workers)
    for name, values in draws.items():
        assert values.dtype == np.float64, name
    return draws


def _assert_byte_identical(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> None:
    assert list(actual) == list(expected)
    for name, values in expected.items():
        assert actual[name].shape == values.shape, name
        assert actual[name].tobytes() == values.tobytes(), name


# ----------------------------------------------------------------------------
# Fixtures: fixed-seed donors and recipients
# ----------------------------------------------------------------------------


def _donor_columns(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, DONOR_ROWS)
    z = rng.uniform(0.0, 10.0, DONOR_ROWS)
    positive = np.exp(1.0 + 0.4 * x + 0.1 * z + rng.normal(0.0, 0.5, DONOR_ROWS))
    sign = rng.choice([-1.0, 0.0, 1.0], size=DONOR_ROWS, p=[0.25, 0.35, 0.4])
    signed = sign * np.abs(50.0 * x + 5.0 * z + rng.normal(0.0, 20.0, DONOR_ROWS))
    follower = 2.0 * np.abs(signed) + 10.0 * z + rng.gamma(2.0, 5.0, DONOR_ROWS)
    return {
        "x": x,
        "z": z,
        "positive": positive,
        "signed": signed,
        "follower": follower,
    }


def _donor_weights(seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 1).uniform(0.5, 4.0, DONOR_ROWS)


def _recipients(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    return pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.1, RECIPIENT_ROWS),
            "z": rng.uniform(-0.5, 10.5, RECIPIENT_ROWS),
        },
        index=pd.RangeIndex(10_000, 10_000 + RECIPIENT_ROWS),
    )


# ----------------------------------------------------------------------------
# Byte-identical draws for 1 and N prediction workers and fit jobs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("setting", COMPARED_SETTINGS)
def test_plain_qrf_draws_are_byte_identical_across_thread_counts(
    setting: tuple[str, str], make_person_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-forest, ungated target draws the same bytes at any thread count."""
    frame = make_person_frame(_donor_columns(seed=11), _donor_weights(seed=11))
    recipients = _recipients(seed=11)

    def run() -> tuple[dict[str, np.ndarray], list[int]]:
        fitted = QRF(n_estimators=N_ESTIMATORS, seed=31).fit(
            frame, ["x", "z"], ["positive"]
        )
        assert fitted.regimes() == {"positive": Regime.POSITIVE_ONLY}
        draws = fitted.predict(recipients)
        return (
            {"positive": draws["positive"].to_numpy()},
            _forest_fit_jobs(fitted._target_models),
        )

    serial = _run_under(monkeypatch, SERIAL, run)
    _assert_byte_identical(serial, _run_under(monkeypatch, setting, run))


@pytest.mark.parametrize("setting", COMPARED_SETTINGS)
def test_three_sign_gated_draws_are_byte_identical_across_thread_counts(
    setting: tuple[str, str], make_person_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero, positive and negative target draws the same bytes at any count.

    The target has a sign gate and a forest per sign, and both forests draw
    through the thread pool.
    """
    frame = make_person_frame(_donor_columns(seed=12), _donor_weights(seed=12))
    recipients = _recipients(seed=12)

    def run() -> tuple[dict[str, np.ndarray], list[int]]:
        fitted = RegimeGatedQRF(n_estimators=N_ESTIMATORS, seed=37).fit(
            frame, ["x", "z"], ["signed"]
        )
        assert fitted.regimes() == {"signed": Regime.THREE_SIGN}
        drawn = fitted.predict(recipients)["signed"].to_numpy()
        assert (drawn == 0.0).any() and (drawn > 0).any() and (drawn < 0).any()
        return {"signed": drawn}, _forest_fit_jobs(fitted._target_models)

    serial = _run_under(monkeypatch, SERIAL, run)
    _assert_byte_identical(serial, _run_under(monkeypatch, setting, run))


@pytest.mark.parametrize("setting", COMPARED_SETTINGS)
def test_chained_fit_draw_next_is_byte_identical_across_thread_counts(
    setting: tuple[str, str], make_person_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target-at-a-time chain steps draw the same bytes at any thread count.

    The second target conditions on the first target's raw draws, so a
    difference in the first would carry on.
    """
    frame = make_person_frame(_donor_columns(seed=13), _donor_weights(seed=13))
    recipients = _recipients(seed=13)
    targets = ["signed", "follower"]

    def run() -> tuple[dict[str, np.ndarray], list[int]]:
        model = RegimeGatedQRF(n_estimators=N_ESTIMATORS, seed=41)
        state = model.start_chain(frame, ["x", "z"], targets)
        raw_priors = pd.DataFrame(index=recipients.index)
        draws: dict[str, np.ndarray] = {}
        forest_fit_jobs: list[int] = []
        while not state.is_complete:
            step = model.fit_draw_next(frame, recipients, raw_priors, state=state)
            draws[step.target] = np.array(step.raw_draw)
            raw_priors[step.target] = step.raw_draw
            forest_fit_jobs.extend(_forest_fit_jobs(step.fitted._target_models))
            state = step.state
        assert list(draws) == targets
        return draws, forest_fit_jobs

    serial = _run_under(monkeypatch, SERIAL, run)
    _assert_byte_identical(serial, _run_under(monkeypatch, setting, run))


def test_chain_changing_fit_jobs_between_steps_draws_the_same_bytes(
    make_person_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step fits with the live count; draws and saved state do not move."""
    frame = make_person_frame(_donor_columns(seed=13), _donor_weights(seed=13))
    recipients = _recipients(seed=13)
    targets = ["signed", "follower"]
    monkeypatch.setenv(PREDICT_WORKERS_ENV, "1")

    def run_chain(fit_jobs_per_target: tuple[str, str]):
        monkeypatch.setenv(FIT_JOBS_ENV, fit_jobs_per_target[0])
        model = RegimeGatedQRF(n_estimators=N_ESTIMATORS, seed=41)
        state = model.start_chain(frame, ["x", "z"], targets)
        raw_priors = pd.DataFrame(index=recipients.index)
        draws: dict[str, np.ndarray] = {}
        for fit_jobs in fit_jobs_per_target:
            monkeypatch.setenv(FIT_JOBS_ENV, fit_jobs)
            step = model.fit_draw_next(frame, recipients, raw_priors, state=state)
            assert set(_forest_fit_jobs(step.fitted._target_models)) == {int(fit_jobs)}
            draws[step.target] = np.array(step.raw_draw)
            raw_priors[step.target] = step.raw_draw
            state = step.state
        return draws, state

    unchanged_draws, unchanged_state = run_chain(("1", "1"))
    changed_draws, changed_state = run_chain(("1", str(SEVERAL)))

    _assert_byte_identical(unchanged_draws, changed_draws)
    assert changed_state.to_dict() == unchanged_state.to_dict()


def test_unset_prediction_workers_size_the_draw_pool_from_joblib(
    make_person_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the override unset, the draw's pool is ``joblib.cpu_count()`` wide."""
    joblib_cpu_count = _cpu_count_unlike_os(monkeypatch)
    frame = make_person_frame(_donor_columns(seed=14), _donor_weights(seed=14))
    monkeypatch.setenv(FIT_JOBS_ENV, "1")
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    recorder = _PoolRecorder()
    monkeypatch.setattr(qrf_module, "ThreadPoolExecutor", recorder.executor_class())

    fitted = QRF(n_estimators=N_ESTIMATORS, seed=43).fit(
        frame, ["x", "z"], ["positive"]
    )
    fitted.predict(_recipients(seed=14))

    assert recorder.pool_widths == [joblib_cpu_count]
