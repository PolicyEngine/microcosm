"""The primary-QRF worker records the QRF draw's default pool width.

With ``POPULACE_FIT_PREDICT_WORKERS`` unset, the QRF draw sizes its thread pool
by ``joblib.cpu_count()`` (``microcosm.fit.qrf.default_predict_workers``). The
worker identity in ``worker_identity.py`` and the older binding in
``stacked_spine.py`` record that same width, and the environment variable
still overrides it. Each test patches ``joblib.cpu_count`` to a value that
``os.cpu_count()``, the earlier default, does not return.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import os

import joblib

from test_support.microcosm_build.us_stacked_spine import *

PREDICT_WORKERS_ENV = "POPULACE_FIT_PREDICT_WORKERS"


def _patch_joblib_cpu_count(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    def patched_cpu_count(only_physical_cores: bool = False) -> int:
        return value

    monkeypatch.setattr(joblib, "cpu_count", patched_cpu_count)


def _cpu_count_unlike_os(offset: int = 3) -> int:
    return (os.cpu_count() or 1) + offset


def _set_predict_workers(
    monkeypatch: pytest.MonkeyPatch, configured: str | None
) -> None:
    if configured is None:
        monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    else:
        monkeypatch.setenv(PREDICT_WORKERS_ENV, configured)


def _worker_identity_predict_controls() -> dict[str, object]:
    return worker_identity_module._semantic_environment()["semantic_controls"][
        PREDICT_WORKERS_ENV
    ]


def _stacked_spine_predict_controls() -> dict[str, object]:
    binding = stacked_spine_module._legacy_primary_qrf_worker_execution_binding()
    return binding["environment"]["semantic_controls"][PREDICT_WORKERS_ENV]


PREDICT_CONTROL_READERS = (
    pytest.param(_worker_identity_predict_controls, id="worker_identity"),
    pytest.param(_stacked_spine_predict_controls, id="stacked_spine"),
)


@pytest.mark.parametrize("read_controls", PREDICT_CONTROL_READERS)
@pytest.mark.parametrize("configured", [None, "", "  "])
def test_unset_predict_workers_record_joblib_cpu_count(
    read_controls, configured: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    joblib_cpu_count = _cpu_count_unlike_os()
    _patch_joblib_cpu_count(monkeypatch, joblib_cpu_count)
    _set_predict_workers(monkeypatch, configured)

    controls = read_controls()

    assert controls["configured"] == configured
    assert controls["resolved"] == joblib_cpu_count


@pytest.mark.parametrize("read_controls", PREDICT_CONTROL_READERS)
@pytest.mark.parametrize(("configured", "expected"), [("1", 1), ("2", 2), ("7", 7)])
def test_predict_workers_override_wins_over_joblib_cpu_count(
    read_controls,
    configured: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_joblib_cpu_count(monkeypatch, _cpu_count_unlike_os())
    _set_predict_workers(monkeypatch, configured)

    assert read_controls() == {
        "configured": configured,
        "resolved": expected,
        "resolution": "environment_override",
    }


def test_worker_identity_memo_binds_unset_predict_worker_default(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed ``joblib.cpu_count()`` recomputes the cached identity."""
    _source, calls = _worker_identity_memo_inputs
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    smaller, larger = _cpu_count_unlike_os(3), _cpu_count_unlike_os(5)

    _patch_joblib_cpu_count(monkeypatch, smaller)
    first = worker_identity_module.primary_qrf_worker_semantic_identity()
    _patch_joblib_cpu_count(monkeypatch, larger)
    second = worker_identity_module.primary_qrf_worker_semantic_identity()

    predict = PREDICT_WORKERS_ENV
    assert first["environment"]["semantic_controls"][predict]["resolved"] == smaller
    assert second["environment"]["semantic_controls"][predict]["resolved"] == larger
    assert worker_identity_module._canonical_sha256(
        first
    ) != worker_identity_module._canonical_sha256(second)
    assert len(calls) == 2

    _patch_joblib_cpu_count(monkeypatch, smaller)
    assert worker_identity_module.primary_qrf_worker_semantic_identity() is first
    assert len(calls) == 2
