"""The primary-QRF worker records its thread counts but never compares them.

The QRF draw gives byte-identical imputations for any fit-job count and any
prediction-worker count (``microcosm-fit``'s ``test_qrf_worker_counts.py``),
and with ``POPULACE_FIT_PREDICT_WORKERS`` unset the count comes from the
machine (``joblib.cpu_count()``). Schema-2 worker bindings therefore keep
``POPULACE_FIT_N_JOBS`` and ``POPULACE_FIT_PREDICT_WORKERS`` in
``audit_thread_controls``, next to the interpreter paths in ``audit_aliases``,
and out of the semantic identity. Everything that compares a recorded worker
with the live one reads only the semantic identity: loading a pool for
release or scoring, resuming the primary-QRF checkpoint, the pool's
checkpoint identity, and the spec engine's worker resolution. A pool built on
an 18-CPU host therefore loads under the release tool's 16-CPU cap.

Schema-1 bindings, which kept the counts in
``semantic_identity.environment.semantic_controls``, still validate, so a pool
recorded before is refused because its identity changed, not because its
record is unreadable.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import json
from copy import deepcopy

import joblib

from microcosm.build.spec_engine import imputation_semantics
from test_support.microcosm_build.us_stacked_spine import *

FIT_JOBS_ENV = "POPULACE_FIT_N_JOBS"
PREDICT_WORKERS_ENV = "POPULACE_FIT_PREDICT_WORKERS"
JOBLIB_FALLBACK = "joblib_cpu_count_fallback"
SCHEMA_4_FALLBACK = "os_cpu_count_fallback"
BINDING_KEYS = {
    "schema_version",
    "semantic_identity",
    "semantic_identity_sha256",
    "audit_aliases",
    "audit_thread_controls",
}

#: (POPULACE_FIT_N_JOBS, POPULACE_FIT_PREDICT_WORKERS, joblib.cpu_count())
BASE_HOST = (None, None, 18)
RELEASE_UNDER_CPU_CAP = (None, None, 16)
OVERRIDDEN = ("2", "9", 4)
THREAD_SETTINGS = (BASE_HOST, RELEASE_UNDER_CPU_CAP, OVERRIDDEN)


def _apply_thread_setting(
    monkeypatch: pytest.MonkeyPatch,
    setting: tuple[str | None, str | None, int],
) -> None:
    fit_jobs, predict_workers, joblib_cpu_count = setting
    for name, value in (
        (FIT_JOBS_ENV, fit_jobs),
        (PREDICT_WORKERS_ENV, predict_workers),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def patched_cpu_count(only_physical_cores: bool = False) -> int:
        return joblib_cpu_count

    monkeypatch.setattr(joblib, "cpu_count", patched_cpu_count)


def _binding_under(
    monkeypatch: pytest.MonkeyPatch,
    setting: tuple[str | None, str | None, int],
) -> dict[str, object]:
    _apply_thread_setting(monkeypatch, setting)
    return worker_identity_module.primary_qrf_worker_execution_binding()


def _validate_binding(binding: object) -> None:
    worker_identity_module.validate_primary_qrf_worker_execution_binding(
        binding, boundary="thread-control fixture"
    )


def _as_schema_1_record(binding: dict[str, object]) -> dict[str, object]:
    """``binding`` as schema-1 code recorded it: counts in the semantic identity."""
    record = deepcopy(binding)
    controls = record.pop("audit_thread_controls")
    record["schema_version"] = 1
    record["semantic_identity"]["environment"]["semantic_controls"] = controls
    record["semantic_identity_sha256"] = worker_identity_module._canonical_sha256(
        record["semantic_identity"]
    )
    return record


def _validate_primary_config_against_live_worker(
    config: Mapping[str, object], *, boundary: str
) -> None:
    stacked_spine_module._validate_late_resource_binding(
        config,
        producer=stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE,
        entity="tax_unit",
        column=stacked_spine_module.US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT,
        boundary=boundary,
    )


def test_worker_binding_records_thread_controls_outside_the_semantic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding_under(monkeypatch, BASE_HOST)

    assert binding["schema_version"] == 2
    assert binding["schema_version"] == (
        worker_identity_module.PRIMARY_QRF_WORKER_IDENTITY_SCHEMA_VERSION
    )
    assert set(binding) == BINDING_KEYS
    assert set(binding["semantic_identity"]["environment"]) == {
        "policy",
        "overrides",
        "bound_names",
    }
    assert binding["audit_thread_controls"] == {
        FIT_JOBS_ENV: {"configured": None, "resolved": -1},
        PREDICT_WORKERS_ENV: {
            "configured": None,
            "resolved": 18,
            "resolution": JOBLIB_FALLBACK,
        },
    }
    _validate_binding(binding)


def test_overridden_thread_controls_are_recorded_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding_under(monkeypatch, OVERRIDDEN)

    assert binding["audit_thread_controls"] == {
        FIT_JOBS_ENV: {"configured": "2", "resolved": 2},
        PREDICT_WORKERS_ENV: {
            "configured": "9",
            "resolved": 9,
            "resolution": "environment_override",
        },
    }


def test_thread_controls_leave_the_compared_worker_identity_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = [_binding_under(monkeypatch, setting) for setting in THREAD_SETTINGS]

    recorded_thread_controls = {
        json.dumps(binding["audit_thread_controls"], sort_keys=True)
        for binding in bindings
    }
    assert len(recorded_thread_controls) == len(THREAD_SETTINGS)
    first = bindings[0]
    first_projection = worker_identity_module.primary_qrf_worker_semantic_projection(
        first, boundary="first thread setting"
    )
    for other in bindings[1:]:
        assert other["semantic_identity"] == first["semantic_identity"]
        assert other["semantic_identity_sha256"] == first["semantic_identity_sha256"]
        assert worker_identity_module.primary_qrf_worker_bindings_semantically_equal(
            first, other, boundary="thread settings"
        )
        assert (
            worker_identity_module.primary_qrf_worker_semantic_projection(
                other, boundary="other thread setting"
            )
            == first_projection
        )


@pytest.mark.parametrize(
    "release_setting",
    (
        pytest.param(RELEASE_UNDER_CPU_CAP, id="release-capped-at-16-cpus"),
        pytest.param(OVERRIDDEN, id="release-overrides-both-counts"),
    ),
)
def test_pool_recorded_on_another_host_loads_with_the_live_worker(
    monkeypatch: pytest.MonkeyPatch,
    release_setting: tuple[str | None, str | None, int],
) -> None:
    """The release and scoring loaders compare the recorded worker this way."""
    _apply_thread_setting(monkeypatch, BASE_HOST)
    recorded = _fixture_primary_execution_config_binding()
    _apply_thread_setting(monkeypatch, release_setting)
    live_controls = worker_identity_module.primary_qrf_worker_execution_binding()[
        "audit_thread_controls"
    ]
    assert recorded["qrf"]["worker_execution"]["audit_thread_controls"] != (
        live_controls
    )

    _validate_primary_config_against_live_worker(
        recorded, boundary="pool load on another host"
    )


def test_primary_checkpoint_resources_resume_under_other_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = pd.DataFrame(
        {"income": np.array([10.0, 20.0], dtype=np.float64)},
        index=pd.Index([3, 9], name="donor_id"),
    )
    arguments = {
        "primary_qrf_checkpoint_identity_sha256": "a" * 64,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 578,
        "seed": 0,
        "n_estimators": 100,
        "fit_records_enabled": True,
        "tail_bound_diagnostics_enabled": True,
    }
    config_key = "tax_unit.@primary_puf_execution_config"

    _apply_thread_setting(monkeypatch, BASE_HOST)
    recorded = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor, **arguments
    )
    _apply_thread_setting(monkeypatch, OVERRIDDEN)
    live = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor, **arguments
    )

    assert (
        recorded[config_key]["binding"]["qrf"]["worker_execution"][
            "audit_thread_controls"
        ]
        != (
            live[config_key]["binding"]["qrf"]["worker_execution"][
                "audit_thread_controls"
            ]
        )
    )
    assert stacked_spine_module._late_primary_resource_receipts_semantically_equal(
        recorded, live, boundary="primary-QRF checkpoint resume"
    )


def test_primary_qrf_checkpoint_resumes_under_other_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A checkpoint started on one host resumes on another, sidecar untouched.

    The chain itself is stubbed, as in the relocated-resume test in
    ``test_us_stacked_spine.py``; this exercises the sidecar comparison.
    """
    checkpoint_identity = "a" * 64
    checkpoint_dir = tmp_path / checkpoint_identity
    donor = pd.DataFrame({"fixture_donor": [1.0]})

    def initialize(_frame: Frame, _donor: pd.DataFrame, root: Path, **_kwargs) -> None:
        root.mkdir(parents=True)
        (root / stacked_spine_module.PRIMARY_QRF_MANIFEST_FILENAME).write_text(
            "{}",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        stacked_spine_module, "initialize_primary_puf_qrf_chain", initialize
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "primary_puf_qrf_recipient_predictor_universe_receipt",
        lambda _root: {"fixture": "recipient-universe"},
    )
    monkeypatch.setattr(
        stacked_spine_module, "run_primary_puf_qrf_chain", lambda _root, **_kwargs: None
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "finalize_primary_puf_qrf_chain",
        lambda frame, _root, **_kwargs: (frame, frame.resolve_weights("tax_unit").kind),
    )

    def input_binding() -> dict[str, object]:
        resources = stacked_spine_module.stacked_late_primary_resource_receipts(
            donor,
            primary_qrf_checkpoint_identity_sha256=checkpoint_identity,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            seed=0,
            n_estimators=100,
            fit_records_enabled=True,
            tail_bound_diagnostics_enabled=True,
        )
        return stacked_spine_module.stacked_late_primary_checkpoint_input_binding(
            resources
        )

    def run(binding: dict[str, object]):
        return stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            _late_primary_entry(_stacked_gap_fixture()),
            donor,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            fit_records=[],
            tail_bound_diagnostics=[],
            primary_qrf_checkpoint_dir=checkpoint_dir,
            primary_qrf_input_binding=binding,
        )

    _apply_thread_setting(monkeypatch, BASE_HOST)
    initialized = run(input_binding()).receipt["primary_puf_qrf"]
    sidecar_path = checkpoint_dir / "late-producer-input-binding.json"
    sidecar_bytes = sidecar_path.read_bytes()
    assert initialized["resume_status"] == "initialized"

    _apply_thread_setting(monkeypatch, OVERRIDDEN)
    live_binding = input_binding()
    assert live_binding["sha256"] != json.loads(sidecar_bytes)["sha256"], (
        "the recorded thread controls must move the live sidecar digest"
    )
    resumed = run(live_binding).receipt["primary_puf_qrf"]

    assert resumed["resume_status"] == "resumed"
    assert sidecar_path.read_bytes() == sidecar_bytes


def test_pool_checkpoint_identity_ignores_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool's checkpoint identity hashes this resource-semantics receipt."""
    arguments = {
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 578,
        "primary_seed": 0,
        "primary_n_estimators": 100,
        "transfer_seed": 0,
        "transfer_n_estimators": 100,
        "transfer_max_targets_per_fit": (
            acs_transfer_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
        ),
    }
    receipt_digests = set()
    for setting in THREAD_SETTINGS:
        _apply_thread_setting(monkeypatch, setting)
        receipt = stacked_spine_module.stacked_late_producer_resource_semantics_receipt(
            **arguments
        )
        receipt_digests.add(receipt["sha256"])

    assert len(receipt_digests) == 1


def test_spec_engine_worker_resolution_ignores_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = imputation_semantics._worker_execution_template()
    resolutions = []
    for setting in THREAD_SETTINGS:
        _apply_thread_setting(monkeypatch, setting)
        resolutions.append(imputation_semantics._resolve_worker_execution(template))

    assert all(resolution == resolutions[0] for resolution in resolutions)
    assert resolutions[0]["schema_version"] == 2


def test_spec_engine_worker_template_names_every_binding_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = imputation_semantics._worker_execution_template()["template"]
    binding = _binding_under(monkeypatch, BASE_HOST)

    assert set(template) == set(binding)
    assert template["schema_version"] == binding["schema_version"]
    assert template["audit_thread_controls"] == {
        "resolver_op": "primary_qrf_worker_audit_thread_controls"
    }


@pytest.mark.parametrize("label", [JOBLIB_FALLBACK, SCHEMA_4_FALLBACK])
def test_schema_1_worker_records_still_validate(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    record = _as_schema_1_record(_binding_under(monkeypatch, BASE_HOST))
    record["semantic_identity"]["environment"]["semantic_controls"][
        PREDICT_WORKERS_ENV
    ]["resolution"] = label
    record["semantic_identity_sha256"] = worker_identity_module._canonical_sha256(
        record["semantic_identity"]
    )

    _validate_binding(record)
    assert (
        worker_identity_module.current_worker_execution_authentication_receipt(
            record,
            manifest_schema_version=10,
            execution_config_schema_version=6,
            boundary="schema-1 record",
        )["worker_execution_schema_version"]
        == 1
    )


def test_pool_recorded_with_a_schema_1_worker_is_refused_as_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_thread_setting(monkeypatch, BASE_HOST)
    recorded = _fixture_primary_execution_config_binding()
    recorded["qrf"]["worker_execution"] = _as_schema_1_record(
        recorded["qrf"]["worker_execution"]
    )

    assert not worker_identity_module.primary_qrf_worker_bindings_semantically_equal(
        recorded["qrf"]["worker_execution"],
        worker_identity_module.primary_qrf_worker_execution_binding(),
        boundary="schema-1 record",
    )
    with pytest.raises(ValueError, match="semantic worker identity changed"):
        _validate_primary_config_against_live_worker(
            recorded, boundary="schema-1 pool load"
        )


def _schema_2_with_counts_in_the_semantic_identity(
    binding: dict[str, object],
) -> dict[str, object]:
    record = _as_schema_1_record(binding)
    record["schema_version"] = 2
    record["audit_thread_controls"] = deepcopy(binding["audit_thread_controls"])
    return record


def _schema_2_without_audit_thread_controls(
    binding: dict[str, object],
) -> dict[str, object]:
    record = deepcopy(binding)
    del record["audit_thread_controls"]
    return record


def _schema_1_with_audit_thread_controls(
    binding: dict[str, object],
) -> dict[str, object]:
    record = _as_schema_1_record(binding)
    record["audit_thread_controls"] = deepcopy(binding["audit_thread_controls"])
    return record


def _unknown_schema_version(binding: dict[str, object]) -> dict[str, object]:
    record = deepcopy(binding)
    record["schema_version"] = 3
    return record


@pytest.mark.parametrize(
    "malform",
    (
        _schema_2_with_counts_in_the_semantic_identity,
        _schema_2_without_audit_thread_controls,
        _schema_1_with_audit_thread_controls,
        _unknown_schema_version,
    ),
)
def test_worker_records_with_a_mixed_or_unknown_shape_are_refused(
    monkeypatch: pytest.MonkeyPatch, malform
) -> None:
    binding = _binding_under(monkeypatch, BASE_HOST)

    with pytest.raises(ValueError):
        _validate_binding(malform(binding))


@pytest.mark.parametrize(
    ("variable", "field", "value"),
    (
        (FIT_JOBS_ENV, "resolved", 0),
        (FIT_JOBS_ENV, "resolved", -2),
        (FIT_JOBS_ENV, "resolved", True),
        (FIT_JOBS_ENV, "configured", 2),
        (PREDICT_WORKERS_ENV, "resolved", 0),
        (PREDICT_WORKERS_ENV, "resolved", "18"),
        (PREDICT_WORKERS_ENV, "resolution", "process_cpu_count_fallback"),
        (PREDICT_WORKERS_ENV, "unexpected", 1),
    ),
)
def test_malformed_audit_thread_controls_are_refused(
    monkeypatch: pytest.MonkeyPatch, variable: str, field: str, value: object
) -> None:
    binding = _binding_under(monkeypatch, BASE_HOST)
    binding["audit_thread_controls"][variable][field] = value

    with pytest.raises(ValueError):
        _validate_binding(binding)


@pytest.mark.parametrize(
    ("variable", "configured"),
    (
        (FIT_JOBS_ENV, "0"),
        (FIT_JOBS_ENV, "-1"),
        (FIT_JOBS_ENV, "01"),
        (FIT_JOBS_ENV, "four"),
        (PREDICT_WORKERS_ENV, "0"),
        (PREDICT_WORKERS_ENV, "2.0"),
        (PREDICT_WORKERS_ENV, " 3"),
    ),
)
def test_malformed_thread_environment_is_refused_when_binding_the_worker(
    monkeypatch: pytest.MonkeyPatch, variable: str, configured: str
) -> None:
    _apply_thread_setting(monkeypatch, BASE_HOST)
    monkeypatch.setenv(variable, configured)

    with pytest.raises(ValueError, match=variable):
        worker_identity_module.primary_qrf_worker_execution_binding()
