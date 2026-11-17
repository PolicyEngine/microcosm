"""The primary-QRF worker records the QRF draw's default pool width.

With ``POPULACE_FIT_PREDICT_WORKERS`` unset, the QRF draw sizes its thread pool
by ``joblib.cpu_count()`` (``microcosm.fit.qrf.default_predict_workers``). The
worker identity in ``worker_identity.py`` and the older binding in
``stacked_spine.py`` record that same width, and the environment variable
still overrides it. Each test patches ``joblib.cpu_count`` to a value that
``os.cpu_count()``, the earlier default, does not return.

The identity labels that fallback ``"joblib_cpu_count_fallback"``. Identities
written before carry ``"os_cpu_count_fallback"`` and still validate, and the
schema-4 worker shape that legacy pools recorded keeps the old label, so the
scoring-only relocation of those pools still compares equal.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import os
from copy import deepcopy

import joblib

from test_support.microcosm_build.us_stacked_spine import *

PREDICT_WORKERS_ENV = "POPULACE_FIT_PREDICT_WORKERS"
JOBLIB_FALLBACK = "joblib_cpu_count_fallback"
SCHEMA_4_FALLBACK = "os_cpu_count_fallback"


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


@pytest.mark.parametrize(
    ("read_controls", "expected_resolution"),
    (
        pytest.param(
            _worker_identity_predict_controls, JOBLIB_FALLBACK, id="worker_identity"
        ),
        # The older binding builds the frozen schema-4 shape, so it keeps the
        # label that shape has always carried.
        pytest.param(
            _stacked_spine_predict_controls, SCHEMA_4_FALLBACK, id="stacked_spine"
        ),
    ),
)
@pytest.mark.parametrize("configured", [None, "", "  "])
def test_unset_predict_workers_record_joblib_cpu_count(
    read_controls,
    expected_resolution: str,
    configured: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joblib_cpu_count = _cpu_count_unlike_os()
    _patch_joblib_cpu_count(monkeypatch, joblib_cpu_count)
    _set_predict_workers(monkeypatch, configured)

    assert read_controls() == {
        "configured": configured,
        "resolved": joblib_cpu_count,
        "resolution": expected_resolution,
    }


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


def _live_binding_with_unset_predict_workers(
    monkeypatch: pytest.MonkeyPatch, joblib_cpu_count: int
) -> dict[str, object]:
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    _patch_joblib_cpu_count(monkeypatch, joblib_cpu_count)
    return worker_identity_module.primary_qrf_worker_execution_binding()


def _with_resolution(binding: dict[str, object], label: str) -> dict[str, object]:
    """A copy of ``binding`` whose fallback label is ``label``, re-digested."""
    relabelled = deepcopy(binding)
    semantic = relabelled["semantic_identity"]
    semantic["environment"]["semantic_controls"][PREDICT_WORKERS_ENV]["resolution"] = (
        label
    )
    relabelled["semantic_identity_sha256"] = worker_identity_module._canonical_sha256(
        semantic
    )
    return relabelled


@pytest.mark.parametrize("label", [JOBLIB_FALLBACK, SCHEMA_4_FALLBACK])
def test_worker_identity_validation_accepts_both_fallback_labels(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """Identities recorded before the joblib default still validate."""
    binding = _live_binding_with_unset_predict_workers(
        monkeypatch, _cpu_count_unlike_os()
    )

    worker_identity_module.validate_primary_qrf_worker_execution_binding(
        _with_resolution(binding, label), boundary="fallback label"
    )


def test_worker_identity_validation_refuses_an_unknown_fallback_label(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _live_binding_with_unset_predict_workers(
        monkeypatch, _cpu_count_unlike_os()
    )

    with pytest.raises(ValueError, match="resolved worker controls are malformed"):
        worker_identity_module.validate_primary_qrf_worker_execution_binding(
            _with_resolution(binding, "process_cpu_count_fallback"),
            boundary="fallback label",
        )


def test_schema_4_projection_keeps_the_os_cpu_count_fallback_label(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy projection matches what the schema-4 code records.

    The scoring-only relocation of a schema-9 pool allows only two differences
    between the pool's recorded schema-4 worker and this projection of the live
    worker. ``stacked_spine``'s older binding is that schema-4 recorder.
    """
    joblib_cpu_count = _cpu_count_unlike_os()
    binding = _live_binding_with_unset_predict_workers(monkeypatch, joblib_cpu_count)
    semantic = binding["semantic_identity"]
    live_controls = semantic["environment"]["semantic_controls"][PREDICT_WORKERS_ENV]
    assert live_controls["resolution"] == JOBLIB_FALLBACK

    projected = worker_identity_module.legacy_primary_qrf_worker_execution_binding(
        semantic_identity=semantic
    )

    assert projected["environment"]["semantic_controls"][PREDICT_WORKERS_ENV] == {
        "configured": None,
        "resolved": joblib_cpu_count,
        "resolution": SCHEMA_4_FALLBACK,
    }
    assert live_controls["resolution"] == JOBLIB_FALLBACK
    schema_4_recorded = (
        stacked_spine_module._legacy_primary_qrf_worker_execution_binding()
    )
    assert (
        worker_identity_module._legacy_worker_execution_mismatch_paths(
            schema_4_recorded, projected
        )
        == ()
    )


def test_schema_4_projection_keeps_an_override_label(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_joblib_cpu_count(monkeypatch, _cpu_count_unlike_os())
    monkeypatch.setenv(PREDICT_WORKERS_ENV, "2")
    semantic = worker_identity_module.primary_qrf_worker_semantic_identity()

    projected = worker_identity_module.legacy_primary_qrf_worker_execution_binding(
        semantic_identity=semantic
    )

    assert projected["environment"]["semantic_controls"][PREDICT_WORKERS_ENV] == {
        "configured": "2",
        "resolved": 2,
        "resolution": "environment_override",
    }
