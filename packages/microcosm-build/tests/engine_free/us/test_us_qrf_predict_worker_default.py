"""The primary-QRF worker records the QRF draw's default pool width.

With ``POPULACE_FIT_PREDICT_WORKERS`` unset, the QRF draw sizes its thread pool
by ``joblib.cpu_count()`` (``microcosm.fit.qrf.default_predict_workers``). The
worker binding in ``worker_identity.py`` (its ``audit_thread_controls``) and
the older binding in ``stacked_spine.py`` record that same width, and the
environment variable still overrides it. Each test patches
``joblib.cpu_count`` to a value that ``os.cpu_count()``, the earlier default,
does not return.

Both label that fallback ``"joblib_cpu_count_fallback"``. Bindings written
before carry ``"os_cpu_count_fallback"`` and still validate. The schema-4
worker shape that legacy pools recorded is projected as that code recorded it,
the host's ``os.cpu_count()`` under the old label, so the scoring-only
relocation of those pools compares what it compared before.
``test_us_qrf_worker_thread_controls.py`` covers why the counts sit outside
the compared semantic identity.
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
    binding = worker_identity_module.primary_qrf_worker_execution_binding()
    return binding["audit_thread_controls"][PREDICT_WORKERS_ENV]


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
        pytest.param(
            _stacked_spine_predict_controls, JOBLIB_FALLBACK, id="stacked_spine"
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


def test_changed_joblib_cpu_count_reuses_the_cached_identity(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count is read per binding; the cached semantic identity omits it."""
    _source, calls = _worker_identity_memo_inputs
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    smaller, larger = _cpu_count_unlike_os(3), _cpu_count_unlike_os(5)

    _patch_joblib_cpu_count(monkeypatch, smaller)
    first = worker_identity_module.primary_qrf_worker_execution_binding()
    _patch_joblib_cpu_count(monkeypatch, larger)
    second = worker_identity_module.primary_qrf_worker_execution_binding()

    predict = PREDICT_WORKERS_ENV
    assert first["audit_thread_controls"][predict]["resolved"] == smaller
    assert second["audit_thread_controls"][predict]["resolved"] == larger
    assert first["semantic_identity"] == second["semantic_identity"]
    assert len(calls) == 1


def _live_binding_with_unset_predict_workers(
    monkeypatch: pytest.MonkeyPatch, joblib_cpu_count: int
) -> dict[str, object]:
    monkeypatch.delenv(PREDICT_WORKERS_ENV, raising=False)
    _patch_joblib_cpu_count(monkeypatch, joblib_cpu_count)
    return worker_identity_module.primary_qrf_worker_execution_binding()


def _with_resolution(binding: dict[str, object], label: str) -> dict[str, object]:
    """A copy of ``binding`` whose recorded fallback label is ``label``."""
    relabelled = deepcopy(binding)
    relabelled["audit_thread_controls"][PREDICT_WORKERS_ENV]["resolution"] = label
    return relabelled


@pytest.mark.parametrize("label", [JOBLIB_FALLBACK, SCHEMA_4_FALLBACK])
def test_worker_identity_validation_accepts_both_fallback_labels(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """Bindings recorded before the joblib default still validate."""
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

    with pytest.raises(ValueError, match="thread controls are malformed"):
        worker_identity_module.validate_primary_qrf_worker_execution_binding(
            _with_resolution(binding, "process_cpu_count_fallback"),
            boundary="fallback label",
        )


def test_schema_4_projection_records_an_unset_count_as_schema_4_code_did(
    _worker_identity_memo_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy projection reports ``os.cpu_count()`` under the old label.

    Schema-4 code resolved an unset count to ``os.cpu_count()`` and labelled it
    ``"os_cpu_count_fallback"``. The scoring-only relocation of a schema-9 pool
    compares that record with this projection of the live worker and allows
    only the two relocated interpreter paths to differ. The live binding
    itself still records the ``joblib.cpu_count()`` the draw uses.
    """
    joblib_cpu_count = _cpu_count_unlike_os()
    binding = _live_binding_with_unset_predict_workers(monkeypatch, joblib_cpu_count)
    semantic = binding["semantic_identity"]
    live_controls = binding["audit_thread_controls"][PREDICT_WORKERS_ENV]
    live_before = deepcopy(live_controls)

    projected = worker_identity_module.legacy_primary_qrf_worker_execution_binding(
        semantic_identity=semantic
    )

    schema_4_controls = {
        "configured": None,
        "resolved": os.cpu_count() or 1,
        "resolution": SCHEMA_4_FALLBACK,
    }
    assert (
        projected["environment"]["semantic_controls"][PREDICT_WORKERS_ENV]
        == schema_4_controls
    )
    assert (
        live_controls
        == live_before
        == {
            "configured": None,
            "resolved": joblib_cpu_count,
            "resolution": JOBLIB_FALLBACK,
        }
    )
    relocated_python = "/relocated/campaign/.venv/bin/python"
    recorded = deepcopy(projected)
    recorded["argv_template"][0] = relocated_python
    recorded["interpreter"]["executable"] = relocated_python
    recorded["environment"]["semantic_controls"][PREDICT_WORKERS_ENV] = dict(
        schema_4_controls
    )
    assert (
        worker_identity_module._legacy_worker_execution_mismatch_paths(
            recorded, projected
        )
        == worker_identity_module.LEGACY_WORKER_PERMITTED_MISMATCHES
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
