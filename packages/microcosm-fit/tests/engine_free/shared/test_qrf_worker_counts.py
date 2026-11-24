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

Forests with a leaf-size cap (``max_samples_leaf``) keep a seeded subsample of
each larger leaf, taken with Python's global :mod:`random`: for each tree in
turn, the fit seeds it with the tree's seed and shuffles that tree's leaves.
Their tests also record that every leaf was subsampled, on the thread that
called the fit, and show that keeping other samples changes the draw, so equal
bytes are not vacuous.
"""

from __future__ import annotations

import collections
import functools
import os
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import pytest

from microcosm.fit import QRF, Regime, RegimeGatedQRF
from microcosm.fit import qrf as qrf_module
from microcosm.fit.qrf import _fit_n_jobs, _predict_workers
from microcosm.frame import Frame

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


def _fitted_forests(target_models: dict) -> list:
    """Every forest of a fitted model's targets, in target order."""
    return [
        forest
        for target_model in target_models.values()
        for forest in (target_model.positive, target_model.negative)
        if forest is not None
    ]


def _forest_fit_jobs(target_models: dict) -> list[int]:
    """The ``n_jobs`` of every forest in a fitted model's targets."""
    return [forest.model.n_jobs for forest in _fitted_forests(target_models)]


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


# ----------------------------------------------------------------------------
# Capped leaves: the seeded leaf subsample ignores thread counts
# ----------------------------------------------------------------------------

#: The cap of the explicit-cap case. Its two features take six combinations of
#: values, so each tree's leaves hold about 200 bootstrap rows of mixed target
#: values, and every leaf is subsampled to 5.
EXPLICIT_LEAF_CAP = 5

#: Donor rows of the near-discrete case. Its one feature takes two values, so
#: each tree's leaves hold about 6,000 bootstrap rows, more than the automatic
#: cap of ``_DISCRETE_Y_LEAF_BOUND`` (4,096) that a near-discrete target gets.
NEAR_DISCRETE_DONOR_ROWS = 12_000

#: Distinct target values of the near-discrete case.
NEAR_DISCRETE_VALUE_COUNT = 24


@dataclass(frozen=True)
class _CappedLeafCase:
    """Donors, recipients and settings for one capped-leaf fit and draw."""

    donor_columns: dict[str, np.ndarray]
    donor_weights: np.ndarray
    recipients: pd.DataFrame
    features: list[str]
    #: What the case passes to :class:`QRF` (``None`` lets the fit choose).
    configured_leaf_cap: int | None
    #: The cap the forest must end up with.
    expected_leaf_cap: int


def _explicit_cap_case(seed: int) -> _CappedLeafCase:
    """A continuous positive target with ``max_samples_leaf`` set to 5."""
    rng = np.random.default_rng(seed)
    three_value_feature = rng.integers(0, 3, DONOR_ROWS).astype(np.float64)
    two_value_feature = rng.integers(0, 2, DONOR_ROWS).astype(np.float64)
    value = np.exp(
        1.0
        + 0.5 * three_value_feature
        - 0.3 * two_value_feature
        + rng.normal(0.0, 0.6, DONOR_ROWS)
    )
    recipient_rng = np.random.default_rng(seed + 2)
    recipients = pd.DataFrame(
        {
            "three_value_feature": recipient_rng.integers(0, 3, RECIPIENT_ROWS).astype(
                np.float64
            ),
            "two_value_feature": recipient_rng.integers(0, 2, RECIPIENT_ROWS).astype(
                np.float64
            ),
        },
        index=pd.RangeIndex(10_000, 10_000 + RECIPIENT_ROWS),
    )
    return _CappedLeafCase(
        donor_columns={
            "three_value_feature": three_value_feature,
            "two_value_feature": two_value_feature,
            "value": value,
        },
        donor_weights=np.random.default_rng(seed + 1).uniform(0.5, 4.0, DONOR_ROWS),
        recipients=recipients,
        features=["three_value_feature", "two_value_feature"],
        configured_leaf_cap=EXPLICIT_LEAF_CAP,
        expected_leaf_cap=EXPLICIT_LEAF_CAP,
    )


def _near_discrete_case(seed: int) -> _CappedLeafCase:
    """A target with 24 values, which the fit caps at 4,096 per leaf itself.

    The fit applies that cap to a target with 2 to ``_DISCRETE_Y_UNIQUE_MAX``
    (32) distinct values. With only a few values, keeping other leaf samples
    moved few draws (5 of 1,001 rows with three values); with 24 it moves 86.
    """
    rng = np.random.default_rng(seed)
    two_value_feature = rng.integers(0, 2, NEAR_DISCRETE_DONOR_ROWS).astype(np.float64)
    target_values = np.arange(1.0, NEAR_DISCRETE_VALUE_COUNT + 1.0)
    rising_weights = np.linspace(1.0, 3.0, NEAR_DISCRETE_VALUE_COUNT)
    value_where_feature_is_zero = rng.choice(
        target_values, size=NEAR_DISCRETE_DONOR_ROWS
    )
    value_where_feature_is_one = rng.choice(
        target_values,
        size=NEAR_DISCRETE_DONOR_ROWS,
        p=rising_weights / rising_weights.sum(),
    )
    value = np.where(
        two_value_feature == 0.0,
        value_where_feature_is_zero,
        value_where_feature_is_one,
    )
    recipient_feature = np.random.default_rng(seed + 2).integers(0, 2, RECIPIENT_ROWS)
    recipients = pd.DataFrame(
        {"two_value_feature": recipient_feature.astype(np.float64)},
        index=pd.RangeIndex(10_000, 10_000 + RECIPIENT_ROWS),
    )
    return _CappedLeafCase(
        donor_columns={"two_value_feature": two_value_feature, "value": value},
        donor_weights=np.random.default_rng(seed + 1).uniform(
            0.5, 4.0, NEAR_DISCRETE_DONOR_ROWS
        ),
        recipients=recipients,
        features=["two_value_feature"],
        configured_leaf_cap=None,
        expected_leaf_cap=qrf_module._DISCRETE_Y_LEAF_BOUND,
    )


CAPPED_LEAF_CASES = (
    pytest.param(_explicit_cap_case, id="explicit_cap"),
    pytest.param(_near_discrete_case, id="near_discrete_automatic_cap"),
)


def _capped_leaf_fit_and_draw(
    case: _CappedLeafCase,
    frame: Frame,
    *,
    keep_other_samples: bool = False,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Fit and draw one capped-leaf case, recording the leaf-subsample calls.

    For each tree in turn, the fit seeds the global :mod:`random` once with the
    tree's seed, then subsamples each leaf longer than the cap by calling
    :func:`random.shuffle` on the leaf's sample list and keeping the first
    ``cap`` entries, so each leaf's shuffle continues the random stream the
    previous one left. The fit runs with both functions wrapped, and the run
    fails unless:

    - each tree was seeded once with its own seed, and that seed was followed
      by exactly one shuffle per leaf of that tree before the next seed. So
      every leaf was subsampled and no tree's shuffles were mixed into another
      tree's; the order of the trees is not checked, since it changes nothing;
    - every call ran on the thread that called the fit. Worker threads sharing
      the global :mod:`random` could interleave the trees' calls, and in these
      small fixtures a tree's calls usually finish before a thread switch, so
      the recorded seeds and shuffles alone would rarely show it.

    With ``keep_other_samples`` the shuffle wrapper also reverses each shuffled
    list, so the fit keeps a different subsample.
    """
    # [seed, shuffles made after that seed and before the next one], in call
    # order. A shuffle before any seed is recorded under the seed ``None``.
    shuffles_after_each_seed: list[list] = []
    subsample_threads: set[int] = set()
    original_seed = random.seed
    original_shuffle = random.shuffle

    def recording_seed(seed_value, *args, **kwargs) -> None:
        original_seed(seed_value, *args, **kwargs)
        shuffles_after_each_seed.append([seed_value, 0])
        subsample_threads.add(threading.get_ident())

    def recording_shuffle(samples, *args, **kwargs) -> None:
        original_shuffle(samples, *args, **kwargs)
        if keep_other_samples:
            samples[:] = samples[::-1]
        if not shuffles_after_each_seed:
            shuffles_after_each_seed.append([None, 0])
        shuffles_after_each_seed[-1][1] += 1
        subsample_threads.add(threading.get_ident())

    fitting_thread = threading.get_ident()
    with pytest.MonkeyPatch.context() as fit_patch:
        fit_patch.setattr(random, "seed", recording_seed)
        fit_patch.setattr(random, "shuffle", recording_shuffle)
        fitted = QRF(
            n_estimators=N_ESTIMATORS,
            seed=47,
            max_samples_leaf=case.configured_leaf_cap,
        ).fit(frame, case.features, ["value"])

    assert fitted.regimes() == {"value": Regime.POSITIVE_ONLY}
    forests = _fitted_forests(fitted._target_models)
    assert {forest.model.max_samples_leaf for forest in forests} == {
        case.expected_leaf_cap
    }
    seed_and_leaf_count_per_tree = collections.Counter(
        (tree.random_state, tree.tree_.n_leaves)
        for forest in forests
        for tree in forest.model.estimators_
    )
    recorded_seed_and_shuffle_counts = collections.Counter(
        (seed_value, shuffle_count)
        for seed_value, shuffle_count in shuffles_after_each_seed
    )
    assert recorded_seed_and_shuffle_counts == seed_and_leaf_count_per_tree, (
        "the leaf subsample did not seed each tree once and then shuffle each "
        "of that tree's leaves before seeding another tree"
    )
    assert subsample_threads == {fitting_thread}, (
        "the leaf subsample ran on another thread than the one that called the fit"
    )
    draws = fitted.predict(case.recipients)
    return (
        {"value": draws["value"].to_numpy()},
        _forest_fit_jobs(fitted._target_models),
    )


@pytest.mark.parametrize("setting", COMPARED_SETTINGS)
@pytest.mark.parametrize("build_case", CAPPED_LEAF_CASES)
def test_capped_leaf_draws_are_byte_identical_across_thread_counts(
    build_case: Callable[[int], _CappedLeafCase],
    setting: tuple[str, str],
    make_person_frame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forests that subsample their leaves draw the same bytes at any count."""
    case = build_case(seed=15)
    frame = make_person_frame(case.donor_columns, case.donor_weights)

    def run() -> tuple[dict[str, np.ndarray], list[int]]:
        return _capped_leaf_fit_and_draw(case, frame)

    serial = _run_under(monkeypatch, SERIAL, run)
    _assert_byte_identical(serial, _run_under(monkeypatch, setting, run))


@pytest.mark.parametrize("build_case", CAPPED_LEAF_CASES)
def test_capped_leaf_draws_change_when_other_leaf_samples_are_kept(
    build_case: Callable[[int], _CappedLeafCase],
    make_person_frame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capped-leaf draws depend on which samples each leaf keeps.

    So if a thread count changed the subsample, the byte comparison above would
    fail rather than pass on draws that ignore the subsample.
    """
    case = build_case(seed=15)
    frame = make_person_frame(case.donor_columns, case.donor_weights)

    def run_keeping(
        *, other_samples: bool
    ) -> Callable[[], tuple[dict[str, np.ndarray], list[int]]]:
        return functools.partial(
            _capped_leaf_fit_and_draw,
            case,
            frame,
            keep_other_samples=other_samples,
        )

    seeded_subsample_draws = _run_under(
        monkeypatch, SERIAL, run_keeping(other_samples=False)
    )
    other_subsample_draws = _run_under(
        monkeypatch, SERIAL, run_keeping(other_samples=True)
    )

    assert (
        other_subsample_draws["value"].tobytes()
        != seeded_subsample_draws["value"].tobytes()
    )


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
