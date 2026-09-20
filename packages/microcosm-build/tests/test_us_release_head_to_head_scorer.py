"""Contract tests for the replacement head-to-head scorer.

The scorer is one common path for incumbent and candidate; these tests pin
the pieces that make the head-to-head honest without running the heavy
microsim materialization: the signature has no target-membership switches,
the scored-column contract cannot go silently missing on either side, the
terminal-battery receipt is observed rather than asserted, and the fixture
end-to-end run is deterministic byte-for-byte.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_head_to_head_module():
    root = Path(__file__).resolve().parents[3]
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    name = "score_us_release_head_to_head"
    if name in sys.modules:
        return sys.modules[name]
    path = root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_frame(
    *,
    measure_values: tuple[float, float],
    channels: tuple[str, str] | None = ("asec", "asec"),
) -> Frame:
    household = {
        "household_id": np.asarray([1, 2], dtype="int64"),
        "state_fips": np.asarray([6, 36], dtype="int64"),
        "m_income": np.asarray(measure_values, dtype="float64"),
        "n_flagged": np.asarray([1.0, 0.0], dtype="float64"),
    }
    person = {
        "person_id": np.asarray([1, 2], dtype="int64"),
        "person_household_id": np.asarray([1, 2], dtype="int64"),
        "person_tax_unit_id": np.asarray([1, 2], dtype="int64"),
        "person_spm_unit_id": np.asarray([1, 2], dtype="int64"),
        "person_family_id": np.asarray([1, 2], dtype="int64"),
        "person_marital_unit_id": np.asarray([1, 2], dtype="int64"),
    }
    tax_unit = {"tax_unit_id": np.asarray([1, 2], dtype="int64")}
    spm_unit = {"spm_unit_id": np.asarray([1, 2], dtype="int64")}
    if channels is not None:
        # The by-origin battery scopes its masks on the battery entities
        # (person, tax_unit, spm_unit), matching the observed live incumbent.
        for prefix, table in (
            ("person", person),
            ("tax_unit", tax_unit),
            ("spm_unit", spm_unit),
        ):
            table[f"{prefix}_support_channel"] = np.asarray(channels, dtype=object)
            table[f"{prefix}_support_clone_index"] = np.asarray([0, 0], dtype="int64")
    tables = {
        "person": pd.DataFrame(person),
        "household": pd.DataFrame(household),
        "tax_unit": pd.DataFrame(tax_unit),
        "spm_unit": pd.DataFrame(spm_unit),
        "family": pd.DataFrame({"family_id": np.asarray([1, 2], dtype="int64")}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([1, 2], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([10.0, 20.0], dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )


def _tiny_registry() -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="tiny_income_total",
                entity="household",
                value=500.0,
                measure="m_income",
                period=2024,
                source="fixture",
                family="fixture_family",
            ),
            TargetSpec(
                name="tiny_flagged_count",
                entity="household",
                value=12.0,
                measure="n_flagged",
                period=2024,
                source="fixture",
                family="fixture_family",
                metadata={"measure_mode": "indicator_sum"},
            ),
        ],
        country="us",
    )


def _fixture_yardstick(module, registry=None) -> object:
    registry = registry if registry is not None else _tiny_registry()
    release = module.release
    loss_weights = release._fiscal_target_loss_weights(registry)
    loss_basis = release._fiscal_target_loss_basis(registry, loss_weights)
    return module.FiscalYardstick(
        registry=registry,
        loss_weights=loss_weights,
        loss_basis=loss_basis,
        identity={
            "country": "us",
            "version": registry.version,
            "target_count": len(registry.specs),
            "ledger_facts": {"filename": "fixture.jsonl", "sha256": "0" * 64},
            "congressional_district_vintage_crosswalk": {
                "filename": "fixture.parquet",
                "sha256": "1" * 64,
            },
            "target_period": 2024,
            "age_targets": False,
            "allow_unaged_dollar_targets": True,
            "target_profile_coverage": {"passed": True, "failures": []},
            "environment": {
                "microcosm_commit": "fixture",
                "policyengine_us_version": "fixture",
            },
        },
    )


def _fixture_artifact(module, *, sha256: str, measure_values: tuple[float, float]):
    return module.LoadedArtifact(
        frame=_tiny_frame(measure_values=measure_values),
        identity={
            "kind": "h5",
            "filename": f"{sha256[:8]}.h5",
            "sha256": sha256,
            "size_bytes": 123,
        },
        loader={"kind": "microcosm_entity_h5", "weight_kind": "calibrated"},
        h5_path=Path(f"/nonexistent/{sha256[:8]}.h5"),
    )


def _tiny_frame_with_marketplace_columns(
    *,
    formula_owned: bool,
    interview_leaf: bool,
) -> Frame:
    frame = _tiny_frame(measure_values=(1.0, 2.0))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    if formula_owned:
        tables["person"]["has_marketplace_health_coverage"] = np.asarray(
            [True, False], dtype=bool
        )
    if interview_leaf:
        tables["person"]["has_marketplace_health_coverage_at_interview"] = np.asarray(
            [True, False], dtype=bool
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )


def _patch_release_seams(module, monkeypatch) -> None:
    """Stub the four heavy release seams; everything downstream runs real."""

    release = module.release

    def _identity_repair(frame):
        return frame, {"mode": "fixture_noop"}

    def _stub_gate(*args, **kwargs):
        return SimpleNamespace(passed=True, failures=(), details={})

    def _stub_materialize(frame, specs, **kwargs):
        registry = TargetRegistry(list(specs), country="us")
        return (
            frame,
            registry,
            {
                "declared_targets": len(specs),
                "compiled_candidate_targets": len(specs),
                "dropped_target_names": [],
            },
        )

    def _stub_cd_probe(h5_path):
        return {
            module.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: None,
            module.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: None,
            "household_congressional_district_geoid": {
                "exists": True,
                "positive_unique_count": 2,
            },
        }

    monkeypatch.setattr(release, "_with_base_population_mass_repair", _identity_repair)
    monkeypatch.setattr(release, "_base_population_scale_gate", _stub_gate)
    monkeypatch.setattr(release, "_health_input_signal_gate", _stub_gate)
    monkeypatch.setattr(release, "_materialize_target_frame", _stub_materialize)
    monkeypatch.setattr(release, "_read_cd_vintage_support_provenance", _stub_cd_probe)


def _score_loaded_as_incumbent(module, monkeypatch, loaded) -> dict[str, object]:
    _patch_release_seams(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "compile_yardstick",
        lambda **kwargs: _fixture_yardstick(module),
    )
    monkeypatch.setattr(module, "load_artifact", lambda path, **kwargs: loaded)
    return module.score_head_to_head(
        incumbent=loaded.h5_path,
        candidate=None,
        ledger_facts=Path("/fixture/facts.jsonl"),
        congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
        maximum_microsim_batch_size=1,
    )


def _complete_battery_comparisons(module) -> dict[str, dict[str, object]]:
    comparisons: dict[str, dict[str, object]] = {}
    for label, row in module._canonical_battery_contract().items():
        metric = row["metric"]
        if metric == "boolean_incidence":
            receipt = {
                "status": "tested",
                "metric": metric,
                "asec_incidence": 0.5,
                "acs_incidence": 0.5,
                "incidence_ratio_acs_over_asec": 1.0,
            }
        elif metric == "categorical_tvd":
            receipt = {
                "status": "tested",
                "metric": metric,
                "total_variation_distance": 0.0,
                "category_shares": {
                    "asec": {"fixture": 1.0},
                    "acs": {"fixture": 1.0},
                },
            }
        else:
            receipt = {
                "status": "tested",
                "metric": metric,
                "legs": {
                    sign: {
                        "asec_incidence": 0.5,
                        "acs_incidence": 0.5,
                        "incidence_ratio_acs_over_asec": 1.0,
                        "quantile_envelope_distance": 0.0,
                    }
                    for sign in ("positive", "negative")
                },
            }
        comparisons[label] = receipt
    return comparisons


def test_head_to_head_signature_has_no_target_membership_switches() -> None:
    module = _load_head_to_head_module()

    assert set(inspect.signature(module.score_head_to_head).parameters) == {
        "incumbent",
        "candidate",
        "ledger_facts",
        "age_targets",
        "allow_unaged_dollar_targets",
        "congressional_district_vintage_crosswalk",
        "maximum_microsim_batch_size",
        "candidate_manifest_sha256",
        "candidate_worker_identity_attestation",
        "population_weight_mode",
        "consumer",
    }


def test_candidate_worker_attestation_propagates_from_cli_to_artifact_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_head_to_head_module()
    incumbent = tmp_path / "incumbent.h5"
    candidate = tmp_path / "candidate.manifest.json"
    attestation = tmp_path / "worker-attestation.json"
    pin = "a" * 64
    args = module._parse_args(
        [
            "--incumbent",
            str(incumbent),
            "--candidate",
            str(candidate),
            "--candidate-manifest-sha256",
            pin,
            "--candidate-worker-identity-attestation",
            str(attestation),
            "--ledger-facts",
            str(tmp_path / "facts.jsonl"),
            "--out-prefix",
            str(tmp_path / "scorecard"),
        ]
    )
    assert args.candidate_worker_identity_attestation == attestation

    calls: list[tuple[Path, str | None, Path | None]] = []

    def fake_load_artifact(
        path: Path,
        *,
        expected_manifest_sha256: str | None = None,
        worker_identity_attestation: Path | None = None,
    ) -> SimpleNamespace:
        calls.append((path, expected_manifest_sha256, worker_identity_attestation))
        return SimpleNamespace()

    monkeypatch.setattr(module, "load_artifact", fake_load_artifact)
    monkeypatch.setattr(
        module,
        "compile_yardstick",
        lambda **_kwargs: SimpleNamespace(identity={}, loss_basis={}),
    )
    monkeypatch.setattr(
        module,
        "score_loaded_artifact",
        lambda **kwargs: ({"name": kwargs["artifact_name"]}, (("fixture",),)),
    )
    monkeypatch.setattr(module, "_assert_identical_scored_contracts", lambda _: None)
    monkeypatch.setattr(module, "_comparison_payload", lambda *_: {})
    monkeypatch.setattr(module, "_canonical_battery_contract", lambda: {})
    monkeypatch.setattr(module, "_assert_rss_below_limit", lambda _: None)

    module.score_head_to_head(
        incumbent=args.incumbent,
        candidate=args.candidate,
        ledger_facts=args.ledger_facts,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        candidate_worker_identity_attestation=(
            args.candidate_worker_identity_attestation
        ),
    )

    assert calls == [
        (incumbent, None, None),
        (candidate, pin, attestation),
    ]


def test_pool_scorecard_preserves_worker_authentication_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_head_to_head_module()
    manifest_path = tmp_path / "candidate.manifest.json"
    pool_path = tmp_path / "candidate.h5"
    attestation_path = tmp_path / "worker-attestation.json"
    authentication = {
        "semantic_identity_sha256": "b" * 64,
        "compatibility_attestation_sha256": "c" * 64,
        "purpose": "scoring_only",
    }
    authenticated = SimpleNamespace(
        path=pool_path,
        sha256="d" * 64,
        size_bytes=123,
        manifest_sha256="e" * 64,
        publication_run_id="fixture-run",
        worker_execution_authentication=authentication,
    )
    captured: dict[str, object] = {}

    def fake_scoring_loader(path: Path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return (
            _tiny_frame(measure_values=(1.0, 2.0)),
            {
                "release_id": "fixture-release",
                "status": "gate_failed",
                "simulation_ready": False,
                "terminal_gates": {"passed": False},
            },
            authenticated,
        )

    monkeypatch.setattr(
        module,
        "load_authenticated_us_multispine_pool_for_scoring",
        fake_scoring_loader,
    )
    monkeypatch.setattr(
        module,
        "_drop_historical_formula_owned_columns",
        lambda frame: (frame, {"count": 0, "columns_by_entity": {}}),
    )

    loaded = module._load_pool_manifest(
        manifest_path,
        expected_manifest_sha256="e" * 64,
        worker_identity_attestation=attestation_path,
    )

    assert captured == {
        "path": manifest_path,
        "expected_manifest_sha256": "e" * 64,
        "worker_identity_attestation": attestation_path,
    }
    assert loaded.identity["worker_execution_authentication"] == authentication
    assert loaded.loader["worker_execution_authentication"] == authentication


def _explicit_consumer(module, *, spm=None, formula_columns=(), system_factory=None):
    engine = SimpleNamespace(
        _engine_computed_columns=lambda tables, **_: set(formula_columns),
        variable_dependency_closure=lambda name: SimpleNamespace(
            input_leaves=("n_flagged",)
        ),
        variable_metadata=lambda name: SimpleNamespace(entity="household"),
        materialize=lambda *args, **kwargs: None,
    )
    return module.HeadToHeadConsumer(
        engine=engine,
        dataset_cls=lambda **kwargs: None,
        microsimulation_cls=lambda **kwargs: None,
        system_factory=system_factory
        or (lambda **kwargs: SimpleNamespace(variables={})),
        zero_variable_reform_factory=lambda system, variable: None,
        spm=spm,
    )


def test_explicit_consumer_reaches_both_artifacts_and_every_slice(monkeypatch):
    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    selection = {"geography_kind": "national"}
    consumer = _explicit_consumer(module, spm=selection)
    selection["geography_kind"] = "county"
    yardstick = _fixture_yardstick(module)
    load_calls = []
    materialize_calls = []
    materialize = module.release._materialize_target_frame

    def load(path, **kwargs):
        load_calls.append(kwargs["consumer"])
        return _fixture_artifact(module, sha256="a" * 64, measure_values=(1.0, 2.0))

    def record_materialize(frame, specs, **kwargs):
        assert kwargs["spm"] == {"geography_kind": "national"}
        materialize_calls.append((frame.n("household"), dict(kwargs)))
        kwargs["spm"]["geography_kind"] = "mutated_by_constructor"
        return materialize(frame, specs, **kwargs)

    monkeypatch.setattr(module, "load_artifact", load)
    monkeypatch.setattr(module, "compile_yardstick", lambda **_: yardstick)
    monkeypatch.setattr(module.release, "_materialize_target_frame", record_materialize)
    module.score_head_to_head(
        incumbent=Path("/fixture/a.h5"),
        candidate=Path("/fixture/b.h5"),
        ledger_facts=Path("/fixture/ledger"),
        congressional_district_vintage_crosswalk=Path("/fixture/crosswalk"),
        maximum_microsim_batch_size=1,
        population_weight_mode="shipped",
        consumer=consumer,
    )
    assert load_calls == [consumer, consumer]
    assert len(materialize_calls) == 4
    for size, kwargs in materialize_calls:
        assert size == 1
        assert kwargs["formula_metadata"] is consumer.engine
        assert kwargs["dataset_cls"] is consumer.dataset_cls
        assert kwargs["microsimulation_cls"] is consumer.microsimulation_cls
        assert kwargs["system_factory"] is consumer.system_factory
        assert (
            kwargs["zero_variable_reform_factory"]
            is consumer.zero_variable_reform_factory
        )
        assert kwargs["target_materialization_cache_dir"] is None
    assert dict(consumer.spm) == {"geography_kind": "national"}
    assert len({id(kwargs["spm"]) for _, kwargs in materialize_calls}) == 4


def test_explicit_consumer_owns_formula_normalization(monkeypatch):
    module = _load_head_to_head_module()
    consumer = _explicit_consumer(module, formula_columns=("m_income",))

    def forbidden():
        raise AssertionError("Default country metadata must not be used")

    monkeypatch.setattr(module.release, "_formula_owned_gate_adapter", forbidden)
    original = _tiny_frame(measure_values=(3.0, 4.0))
    cleaned, receipt = module._drop_historical_formula_owned_columns(
        original, consumer=consumer
    )
    assert receipt == {"count": 1, "columns_by_entity": {"household": ["m_income"]}}
    assert "m_income" not in cleaned.table("household")
    assert "m_income" in original.table("household")
    consumer.engine.variable_dependency_closure = lambda _: SimpleNamespace(
        input_leaves=("absent_input",)
    )
    with pytest.raises(ValueError, match="required input leaves are absent"):
        module._drop_historical_formula_owned_columns(original, consumer=consumer)


@pytest.mark.parametrize("selection", [None, {}, {"geography_kind": "national"}])
def test_explicit_consumer_flat_loader_uses_selected_system(
    monkeypatch, tmp_path, selection
):
    module = _load_head_to_head_module()
    calls = []

    def system(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            variables={"age": SimpleNamespace(entity=SimpleNamespace(key="person"))}
        )

    consumer = _explicit_consumer(module, spm=selection, system_factory=system)
    frame = _tiny_frame(measure_values=(1.0, 2.0))
    path = tmp_path / "fake.h5"
    path.write_bytes(b"invented loader seam")
    monkeypatch.setattr(module, "_h5_layout", lambda _: "legacy_flat")

    def flat(path, *, variable_entity_by_name):
        assert variable_entity_by_name == {"age": "person"}
        return frame, {}

    monkeypatch.setattr(module.fiscal_scorer, "_load_legacy_pe_flat_frame", flat)
    assert module.load_artifact(path, consumer=consumer).frame is frame
    assert calls == ([{}] if selection is None else [{"spm": selection}])


def test_explicit_empty_entity_map_never_resolves_default(monkeypatch, tmp_path):
    module = _load_head_to_head_module()

    def forbidden():
        raise AssertionError(
            "Explicit empty mapping was replaced with default metadata"
        )

    monkeypatch.setattr(
        module.fiscal_scorer, "_policyengine_variable_entity_map", forbidden
    )
    # The missing file proves map selection completes before the ordinary reader
    # fails. No fallback to a country import is permitted for an empty mapping.
    with pytest.raises(FileNotFoundError):
        module.fiscal_scorer._load_legacy_pe_flat_frame(
            tmp_path / "does-not-exist.h5", variable_entity_by_name={}
        )


def test_explicit_consumer_entity_loader_uses_selected_dataset(monkeypatch, tmp_path):
    module = _load_head_to_head_module()
    consumer = _explicit_consumer(module)
    frame = _tiny_frame(measure_values=(1.0, 2.0))
    path = tmp_path / "fake.h5"
    path.write_bytes(b"invented entity loader seam")
    monkeypatch.setattr(module, "_h5_layout", lambda _: "entity_tables")
    monkeypatch.setattr(module, "read_nullable_us_h5_metadata", lambda _: {})

    def load(path, *, dataset_cls):
        assert dataset_cls is consumer.dataset_cls
        return frame

    monkeypatch.setattr(module.release, "_load_frame", load)
    assert module.load_artifact(path, consumer=consumer).frame is frame


def test_explicit_consumer_reaches_observed_origin_battery(monkeypatch):
    module = _load_head_to_head_module()
    consumer = _explicit_consumer(module)
    calls = []

    def materialize(frame, *, engine):
        calls.append(engine)
        return SimpleNamespace(frame=frame, receipt={"persisted_to_artifact": False})

    monkeypatch.setattr(module, "materialize_multispine_agreement_outputs", materialize)
    monkeypatch.setattr(
        module,
        "by_origin_battery_artifact_evidence",
        lambda _: SimpleNamespace(
            passed=True,
            failures=(),
            details={"comparisons": _complete_battery_comparisons(module)},
        ),
    )
    payload = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=("asec", "acs")),
        consumer=consumer,
    )
    assert calls == [consumer.engine]
    assert payload["status"] == "computed_finished_h5"
    assert payload["production_receipt_authenticated"] is False


def test_explicit_consumer_requires_complete_dependencies():
    from dataclasses import replace

    module = _load_head_to_head_module()
    consumer = _explicit_consumer(module)
    with pytest.raises(TypeError, match="dataset_cls"):
        replace(consumer, dataset_cls=None)
    with pytest.raises(TypeError, match="_engine_computed_columns"):
        replace(consumer, engine=object())


def test_explicit_consumer_does_not_reuse_historical_pool_battery(monkeypatch):
    from dataclasses import replace

    module = _load_head_to_head_module()
    consumer = _explicit_consumer(module)
    artifact = replace(
        _fixture_artifact(module, sha256="a" * 64, measure_values=(1.0, 2.0)),
        terminal_gates={"historical": "not evidence for the selected consumer"},
    )

    def forbidden(_):
        raise AssertionError("Historical model battery reused as current evidence")

    monkeypatch.setattr(module, "_battery_payload_from_pool_receipt", forbidden)
    payload = module._terminal_battery_payload(artifact, consumer=consumer)
    assert payload["status"] == "inapplicable"
    assert artifact.terminal_gates == {
        "historical": "not evidence for the selected consumer"
    }


def test_dense_candidate_streaming_plan_is_independent_of_total_pool_size() -> None:
    module = _load_head_to_head_module()
    dense_25pct_households = 918_350

    planned = module._streaming_target_column_payload_upper_bound_bytes(
        total_households=dense_25pct_households,
        household_slice_size=5_000,
        chunk_spec_count=module.MATERIALIZE_SCORE_CHUNK_SPECS,
    )
    one_dense_copy = (
        dense_25pct_households
        * module.MATERIALIZE_SCORE_CHUNK_SPECS
        * np.dtype(np.float64).itemsize
    )

    assert planned < 1024**3
    assert planned < module.MAX_RSS_BYTES
    assert one_dense_copy > module.MAX_RSS_BYTES


def test_canonical_battery_contract_matches_production_registries() -> None:
    module = _load_head_to_head_module()

    contract = module._canonical_battery_contract()

    single = len(module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    joint = len(module.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY)
    assert single == 134
    assert joint == 1
    assert len(contract) == single + joint
    assert sum(len(row["metric_legs"]) for row in contract.values()) == 372
    assert (
        sum(row["metric"] == "monetary_sign_separated" for row in contract.values())
        == 79
    )
    assert sum(row["metric"] == "boolean_incidence" for row in contract.values()) == 51
    assert sum(row["metric"] == "categorical_tvd" for row in contract.values()) == 5
    assert (
        "person/source_operator_immigration/"
        "joint[ssn_card_type,immigration_status_str][clone_0]" in contract
    )
    for row in contract.values():
        assert row["metric_legs"] == list(module._metric_legs(row["metric"]))


def test_observed_origin_battery_is_evidence_not_assertion(monkeypatch) -> None:
    module = _load_head_to_head_module()

    monkeypatch.setattr(
        module,
        "materialize_multispine_agreement_outputs",
        lambda frame: SimpleNamespace(
            frame=frame,
            receipt={"persisted_to_artifact": False},
        ),
    )
    monkeypatch.setattr(
        module,
        "by_origin_battery_artifact_evidence",
        lambda frame: SimpleNamespace(
            passed=True,
            failures=(),
            details={"comparisons": _complete_battery_comparisons(module)},
        ),
    )

    no_columns = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=None)
    )
    asec_only = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=("asec", "asec"))
    )
    both_origins = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=("asec", "acs"))
    )

    assert no_columns["status"] == "inapplicable"
    assert "no support-channel" in no_columns["reason"]
    assert asec_only["status"] == "inapplicable"
    assert "empty ACS side" in asec_only["reason"]
    assert asec_only["observed_origins"]["total_acs_rows"] == 0
    assert asec_only["observed_origins"]["entities"]["person"]["asec_rows"] == 2
    assert both_origins["status"] == "computed_finished_h5"
    assert both_origins["production_receipt_authenticated"] is False
    assert both_origins["metric_leg_count"] == 372
    assert both_origins["scalar_leg_status_counts"] == {"computed": 372}
    for payload in (no_columns, asec_only):
        assert payload["comparison_count"] == 135
        assert all(
            row["status"] == "inapplicable" for row in payload["comparisons"].values()
        )


def test_pool_battery_receipt_refuses_a_silently_missing_scalar_leg() -> None:
    module = _load_head_to_head_module()
    comparisons = _complete_battery_comparisons(module)
    label = next(
        label
        for label, row in comparisons.items()
        if row["metric"] == "boolean_incidence"
    )
    del comparisons[label]["incidence_ratio_acs_over_asec"]
    terminal_gates = {
        "gates": {
            "us_by_origin_battery": {
                "passed": True,
                "failures": [],
                "details": {"comparisons": comparisons},
            }
        }
    }

    with pytest.raises(ValueError, match="omits computed leg"):
        module._battery_payload_from_pool_receipt(terminal_gates)


def test_origin_probe_uses_clone_zero_positive_weight_scope() -> None:
    module = _load_head_to_head_module()
    frame = _tiny_frame(measure_values=(1.0, 2.0), channels=("acs", "asec"))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for entity in ("person", "tax_unit", "spm_unit"):
        tables[entity].loc[tables[entity].index[0], f"{entity}_support_clone_index"] = 1
    clone_scoped = Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    zero_weight_scoped = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([0.0, 20.0], dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )

    observed_by_clone = module._observed_origin_receipt(clone_scoped)
    observed_by_weight = module._observed_origin_receipt(zero_weight_scoped)

    for observed in (observed_by_clone, observed_by_weight):
        assert observed["total_acs_rows"] == 0
        assert observed["total_asec_rows"] == 3
        for entity in ("person", "tax_unit", "spm_unit"):
            receipt = observed["entities"][entity]
            assert receipt["raw_origin_row_counts"] == {"acs": 1, "asec": 1}
            assert receipt["origin_row_counts"] == {"asec": 1}


def test_scored_column_contract_refuses_silently_missing_columns() -> None:
    module = _load_head_to_head_module()
    registry = _tiny_registry()
    complete = _tiny_frame(measure_values=(1.0, 2.0))
    broken_tables = {
        entity: complete.table(entity).copy() for entity in complete.entities
    }
    broken_tables["household"] = broken_tables["household"].drop(columns=["n_flagged"])
    broken = Frame(
        broken_tables,
        US_SCHEMA,
        {"household": complete.weights_for("household")},
    )

    contract = module.scored_column_contract(
        complete, registry.specs, artifact_name="incumbent"
    )

    assert ("household", "m_income", "measure") in contract
    assert ("household", "n_flagged", "measure") in contract
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(broken, registry.specs, artifact_name="candidate")
    with pytest.raises(ValueError, match="differs from incumbent"):
        module._assert_identical_scored_contracts(
            {"incumbent": contract, "candidate": contract[:-1]}
        )


@pytest.mark.requires_us
def test_incumbent_and_candidate_h5_loaders_preserve_scored_contract(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    registry = _tiny_registry()
    incumbent_path = tmp_path / "incumbent.h5"
    candidate_path = tmp_path / "candidate.h5"
    broken_path = tmp_path / "candidate_missing_measure.h5"
    incumbent_frame = _tiny_frame(measure_values=(1.0, 2.0))
    candidate_frame = _tiny_frame(measure_values=(3.0, 4.0))
    broken_tables = {
        entity: candidate_frame.table(entity).copy()
        for entity in candidate_frame.entities
    }
    broken_tables["household"] = broken_tables["household"].drop(columns=["n_flagged"])
    broken_frame = Frame(
        broken_tables,
        US_SCHEMA,
        {"household": candidate_frame.weights_for("household")},
    )
    for path, frame in (
        (incumbent_path, incumbent_frame),
        (candidate_path, candidate_frame),
        (broken_path, broken_frame),
    ):
        write_nullable_us_h5(
            frame,
            path,
            period=2024,
            artifact_kind="replacement_scorecard_fixture",
        )

    incumbent = module.load_artifact(incumbent_path)
    candidate = module.load_artifact(candidate_path)
    broken = module.load_artifact(broken_path)
    incumbent_contract = module.scored_column_contract(
        incumbent.frame,
        registry.specs,
        artifact_name="incumbent",
    )
    candidate_contract = module.scored_column_contract(
        candidate.frame,
        registry.specs,
        artifact_name="candidate",
    )

    module._assert_identical_scored_contracts(
        {"incumbent": incumbent_contract, "candidate": candidate_contract}
    )
    assert incumbent.loader["kind"] == candidate.loader["kind"]
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(
            broken.frame,
            registry.specs,
            artifact_name="candidate",
        )


@pytest.mark.requires_us
def test_historical_formula_owned_h5_scores_with_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_with_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    person_columns = set(loaded.frame.table("person").columns)
    assert "has_marketplace_health_coverage" not in person_columns
    assert "has_marketplace_health_coverage_at_interview" in person_columns

    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {
        "count": 1,
        "columns_by_entity": {
            "person": ["has_marketplace_health_coverage"],
        },
    }
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **1**" in markdown
    assert "`person`: `has_marketplace_health_coverage`" in markdown


@pytest.mark.requires_us
def test_historical_formula_owned_h5_refuses_missing_leaf(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_without_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=False,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_artifact(artifact_path)

    message = str(exc_info.value)
    assert "has_marketplace_health_coverage" in message
    assert "has_marketplace_health_coverage_at_interview" in message
    assert "required input leaves are absent" in message


@pytest.mark.requires_us
def test_clean_historical_h5_scores_with_empty_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "clean.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=False,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {"count": 0, "columns_by_entity": {}}
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **0**" in markdown
    assert "No formula-owned columns were present." in markdown


def test_fixture_end_to_end_is_deterministic_and_shares_one_path(
    monkeypatch, tmp_path
) -> None:
    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    yardstick = _fixture_yardstick(module)
    incumbent = _fixture_artifact(
        module, sha256="a" * 64, measure_values=(100.0, 300.0)
    )
    candidate = _fixture_artifact(
        module, sha256="b" * 64, measure_values=(200.0, 290.0)
    )
    incumbent_path = Path("/fixture/incumbent.h5")
    candidate_path = Path("/fixture/candidate.h5")
    artifacts = {incumbent_path: incumbent, candidate_path: candidate}
    monkeypatch.setattr(module, "compile_yardstick", lambda **kwargs: yardstick)
    monkeypatch.setattr(
        module,
        "load_artifact",
        lambda path, **kwargs: artifacts[path],
    )

    def _score_both() -> dict[str, object]:
        return module.score_head_to_head(
            incumbent=incumbent_path,
            candidate=candidate_path,
            ledger_facts=Path("/fixture/facts.jsonl"),
            congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
            maximum_microsim_batch_size=1,
        )

    payload_one = _score_both()
    payload_two = _score_both()

    assert payload_one == payload_two

    incumbent_fiscal = payload_one["artifacts"]["incumbent"]["fiscal"]
    rows = incumbent_fiscal["targets"]
    assert [row["name"] for row in rows] == [
        "tiny_income_total",
        "tiny_flagged_count",
    ]
    # Hand-computed: estimates are A @ w with w = (10, 20).
    # m_income: 100*10 + 300*20 = 7000 vs target 500 -> relative error 13.0,
    # capped scaled error 1.0. n_flagged: 1*10 + 0*20 = 10 vs target 12
    # -> relative error -1/6.
    assert rows[0]["actual"] == pytest.approx(7000.0)
    assert rows[0]["relative_error"] == pytest.approx(13.0)
    assert rows[0]["capped_scaled_absolute_error"] == pytest.approx(1.0)
    assert rows[1]["actual"] == pytest.approx(10.0)
    assert rows[1]["relative_error"] == pytest.approx(-2.0 / 12.0)
    contribution_sum = sum(row["weighted_loss_contribution"] for row in rows)
    assert contribution_sum == pytest.approx(incumbent_fiscal["weighted_loss"])

    comparison = payload_one["comparison"]
    counts = comparison["per_target_absolute_relative_error"]
    # Candidate: m_income actual 200*10 + 290*20 = 7800 (worse than 7000);
    # n_flagged identical inputs -> equal.
    assert counts["incumbent_lower_count"] == 1
    assert counts["equal_count"] == 1
    assert counts["candidate_lower_count"] == 0
    assert comparison["terminal_battery"]["head_to_head_comparable"] is False
    assert comparison["no_threshold_applied"] is True

    first = module.write_scorecard(payload_one, tmp_path / "one" / "scorecard")
    second = module.write_scorecard(payload_two, tmp_path / "two" / "scorecard")
    for path_one, path_two in zip(first, second, strict=True):
        assert path_one.read_bytes() == path_two.read_bytes()
    markdown = first[1].read_text()
    assert "US release replacement scorecard" in markdown
    assert "empty ACS side" in markdown


@pytest.mark.parametrize("mode", ["rescaled", "shipped"])
def test_population_weight_mode_is_shared_and_reported(monkeypatch, mode):
    from microcosm.frame import MassChange

    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    yardstick = _fixture_yardstick(module)
    artifacts = {
        Path("/fixture/incumbent.h5"): _fixture_artifact(
            module, sha256="a" * 64, measure_values=(100.0, 300.0)
        ),
        Path("/fixture/candidate.h5"): _fixture_artifact(
            module, sha256="b" * 64, measure_values=(200.0, 290.0)
        ),
    }
    repaired = []
    gates = []

    def repair(frame):
        assert mode == "rescaled", "Shipped scoring must not repair weights"
        repaired.append(frame)
        weight = frame.weights_for("household")
        return frame.with_weights(
            "household",
            weight.with_values(weight.values * 2, weight.kind),
            mass=MassChange(factor=2, reason="invented test adjustment"),
        ), {"method": "fixture_double", "applied": True}

    def population_gate(frame, *, mass_repair):
        gates.append((frame.weights_for("household").values.copy(), mass_repair))
        return SimpleNamespace(
            passed=False, failures=("invented population mismatch",), details={}
        )

    monkeypatch.setattr(module.release, "_with_base_population_mass_repair", repair)
    monkeypatch.setattr(module.release, "_base_population_scale_gate", population_gate)
    monkeypatch.setattr(module, "compile_yardstick", lambda **_: yardstick)
    monkeypatch.setattr(module, "load_artifact", lambda path, **_: artifacts[path])
    payload = module.score_head_to_head(
        incumbent=Path("/fixture/incumbent.h5"),
        candidate=Path("/fixture/candidate.h5"),
        ledger_facts=Path("/fixture/facts.jsonl"),
        congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
        maximum_microsim_batch_size=1,
        population_weight_mode=mode,
    )
    assert payload["yardstick"]["population_weight_mode"] == mode
    factor = 2 if mode == "rescaled" else 1
    assert len(repaired) == (2 if mode == "rescaled" else 0)
    assert len(gates) == 2
    for (role, expected), (weights, repair_receipt) in zip(
        (("incumbent", 7000.0), ("candidate", 7800.0)), gates, strict=True
    ):
        result = payload["artifacts"][role]
        assert result["fiscal"]["targets"][0]["actual"] == expected * factor
        receipt = result["normalization_receipts"]
        assert receipt["population_weight_mode"] == mode
        assert receipt["base_population_scale_gate"]["passed"] is False
        assert receipt["base_population_mass_repair"]["applied"] is (mode == "rescaled")
        np.testing.assert_array_equal(weights, np.array([10.0, 20.0]) * factor)
        assert (repair_receipt is None) is (mode == "shipped")
    for artifact in artifacts.values():
        np.testing.assert_array_equal(
            artifact.frame.weights_for("household").values, [10.0, 20.0]
        )
        assert artifact.frame.mass_log == ()
    markdown = module.render_markdown(payload)
    assert (
        "shipped weights unchanged on both sides."
        if mode == "shipped"
        else "rescaled to the Census population on both sides."
    ) in markdown


@pytest.mark.parametrize("entry", ["score_loaded_artifact", "score_head_to_head"])
def test_invalid_population_weight_mode_refuses_before_any_io(monkeypatch, entry):
    module = _load_head_to_head_module()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid mode reached artifact or target I/O")

    monkeypatch.setattr(module, "compile_yardstick", forbidden)
    monkeypatch.setattr(module, "_validate_cd_provenance", forbidden)
    kwargs = (
        dict(
            artifact=None,
            artifact_name="test",
            yardstick=None,
            maximum_microsim_batch_size=1,
        )
        if entry == "score_loaded_artifact"
        else dict(incumbent=None, candidate=None, ledger_facts=None)
    )
    with pytest.raises(ValueError, match="Unknown population weight mode"):
        getattr(module, entry)(**kwargs, population_weight_mode="repair_if_needed")


def test_cli_population_mode_reaches_comparison(monkeypatch, tmp_path):
    module = _load_head_to_head_module()
    args = [
        "--incumbent",
        "invented.h5",
        "--ledger-facts",
        "facts.jsonl",
        "--out-prefix",
        str(tmp_path / "scorecard"),
    ]
    assert module._parse_args(args).population_weight_mode == "rescaled"
    captured = []

    def score(**kwargs):
        captured.append(kwargs["population_weight_mode"])
        return {
            "artifacts": {
                "incumbent": {"identity": {"sha256": "a" * 64}},
                "candidate": None,
            }
        }

    monkeypatch.setattr(module, "score_head_to_head", score)
    monkeypatch.setattr(
        module, "write_scorecard", lambda *_: ("invented.json", "invented.md")
    )
    assert module.main(args + ["--population-weight-mode", "shipped"]) == 0
    assert captured == ["shipped"]


def test_chunked_scoring_recombination_matches_one_shot(monkeypatch) -> None:
    """Chunked materialize-and-score must reproduce a one-shot score_targets
    + production attribution bitwise: same aggregate, same per-target rows."""

    from microcosm.calibrate import score_targets
    from microcosm.calibrate._target_loss_attribution import (
        assemble_target_loss_attribution,
    )

    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    monkeypatch.setattr(module, "MATERIALIZE_SCORE_CHUNK_SPECS", 2)
    registry = TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                value=value,
                measure=measure,
                period=2024,
                source="fixture",
                family="fixture_family",
                signed=value < 0,
                metadata=(
                    {"measure_mode": "indicator_sum"} if measure == "n_flagged" else {}
                ),
            )
            for name, measure, value in (
                ("five_income_a", "m_income", 5_000.0),
                ("five_flagged_b", "n_flagged", 12.0),
                ("five_income_c", "m_income", 7_100.0),
                ("five_zero_d", "n_flagged", 0.0),
                ("five_income_e", "m_income", -250.0),
            )
        ],
        country="us",
    )
    yardstick = _fixture_yardstick(module, registry=registry)
    frame = _tiny_frame(measure_values=(100.0, 300.0))
    artifact = module.LoadedArtifact(
        frame=frame,
        identity={
            "kind": "h5",
            "filename": "five.h5",
            "sha256": "d" * 64,
            "size_bytes": 5,
        },
        loader={"kind": "microcosm_entity_h5", "weight_kind": "calibrated"},
        h5_path=Path("/nonexistent/five.h5"),
    )

    payload, _ = module.score_loaded_artifact(
        artifact=artifact,
        artifact_name="incumbent",
        yardstick=yardstick,
        maximum_microsim_batch_size=1,
    )

    one_shot = score_targets(
        frame,
        registry.to_target_set(),
        target_loss_weights=yardstick.loss_weights,
        target_loss_cap=module.release.US_FISCAL_TARGET_LOSS_CAP,
    )
    attribution = assemble_target_loss_attribution(one_shot)

    chunking = payload["normalization_receipts"]["materialize_score_chunking"]
    assert chunking["chunk_count"] == 3
    assert [chunk["spec_range"] for chunk in chunking["chunks"]] == [
        [0, 2],
        [2, 4],
        [4, 5],
    ]
    assert all(
        chunk["target_compilation"]["household_slices"] == 2
        for chunk in chunking["chunks"]
    )
    assert payload["fiscal"]["weighted_loss"] == float(one_shot.final_loss)
    assert payload["fiscal"]["fraction_within_10pct"] == one_shot.fraction_within_10pct
    rows = payload["fiscal"]["targets"]
    assert len(rows) == 5
    for row, diagnostic, attribution_row in zip(
        rows,
        one_shot.diagnostics,
        attribution.rows,
        strict=True,
    ):
        assert row["actual"] == float(diagnostic.final_estimate)
        assert row["target"] == float(diagnostic.target)
        assert row["relative_error"] == float(diagnostic.relative_error)
        assert row["target_loss_weight"] == attribution_row["target_loss_weight"]
        assert (
            row["target_loss_weight_share"]
            == attribution_row["target_loss_weight_share"]
        )
        assert row["target_loss_scale"] == attribution_row["target_loss_scale"]
        assert (
            row["capped_scaled_absolute_error"]
            == attribution_row["final_capped_scaled_error"]
        )
        assert (
            row["weighted_loss_contribution"]
            == attribution_row["final_loss_contribution"]
        )


def test_dropped_targets_fail_loudly_before_scoring(monkeypatch) -> None:
    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)

    def _dropping_materialize(frame, specs, **kwargs):
        kept = list(specs)[:-1]
        return (
            frame,
            TargetRegistry(kept, country="us"),
            {
                "declared_targets": len(specs),
                "compiled_candidate_targets": len(kept),
                "dropped_target_names": [list(specs)[-1].name],
            },
        )

    monkeypatch.setattr(
        module.release, "_materialize_target_frame", _dropping_materialize
    )

    with pytest.raises(ValueError, match="did not materialize the full fiscal"):
        module.score_loaded_artifact(
            artifact=_fixture_artifact(
                module, sha256="c" * 64, measure_values=(1.0, 2.0)
            ),
            artifact_name="incumbent",
            yardstick=_fixture_yardstick(module),
            maximum_microsim_batch_size=None,
        )


def test_artifact_path_keeps_h5_symlink_name(tmp_path) -> None:
    """A Hugging Face cache snapshot is an .h5-named symlink to an
    extensionless blob; the scorer must keep the snapshot name so the
    dataset loader's suffix validation and the filename identity both see
    the real artifact name."""

    module = _load_head_to_head_module()
    blob = tmp_path / "blobs" / ("a" * 8)
    blob.parent.mkdir()
    blob.write_bytes(b"not-really-h5")
    snapshot = tmp_path / "snapshots" / "populace_us_2024.h5"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)

    kept = module._resolved_artifact_path(snapshot)

    assert kept.name == "populace_us_2024.h5"
    assert kept.suffix == ".h5"
    with pytest.raises(FileNotFoundError):
        module._resolved_artifact_path(tmp_path / "missing.h5")


def test_live_incumbent_identity_annotation() -> None:
    module = _load_head_to_head_module()

    pin = module._POLICYENGINE_LIVE_US_INCUMBENT
    matched = module._live_incumbent_identity_if_matched(pin["sha256"])
    unmatched = module._live_incumbent_identity_if_matched("f" * 64)

    assert unmatched is None
    assert matched is not None
    resolved = matched["policyengine_package_resolved"]
    assert resolved["repo_id"] == "policyengine/populace-us"
    assert resolved["filename"] == "populace_us_2024.h5"
    assert resolved["default_dataset"] == "populace_us_2024"
    assert (
        resolved["revision"]
        == "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
    )


@pytest.mark.parametrize("engine_available", [False, True])
def test_entity_hdf_scorer_engine_markers_follow_actual_collection_hook(
    request, monkeypatch, engine_available
) -> None:
    """PyTables alone must not activate the real country-engine HDF loader."""
    engine_tests = {
        "test_incumbent_and_candidate_h5_loaders_preserve_scored_contract",
        "test_historical_formula_owned_h5_scores_with_drop_receipt",
        "test_historical_formula_owned_h5_refuses_missing_leaf",
        "test_clean_historical_h5_scores_with_empty_drop_receipt",
    }
    assert {
        name
        for name, function in globals().items()
        if name.startswith("test_")
        and inspect.isfunction(function)
        and any(
            mark.name == "requires_us" for mark in getattr(function, "pytestmark", ())
        )
    } == engine_tests
    # Fresh real pytest items read the decorators without reusing skip marks
    # already added to this session's original collection. No test body runs.
    names = sorted(engine_tests) + [
        "test_scored_column_contract_refuses_silently_missing_columns"
    ]
    items = [
        pytest.Function.from_parent(
            request.node.parent, name=name, callobj=globals()[name]
        )
        for name in names
    ]
    conftest_path = Path(__file__).resolve().parents[3] / "conftest.py"
    plugins = [
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if isinstance(getattr(plugin, "__file__", None), str)
        and Path(plugin.__file__).resolve() == conftest_path
    ]
    assert len(plugins) == 1
    root_config = plugins[0]
    requested_specs = []

    def find_spec(name):
        assert name in {"policyengine_us", "policyengine_uk"}
        requested_specs.append(name)
        return object() if name == "policyengine_uk" or engine_available else None

    with monkeypatch.context() as patch:
        patch.setattr(root_config.importlib.util, "find_spec", find_spec)
        root_config.pytest_collection_modifyitems(request.config, items)
    assert requested_specs == ["policyengine_us", "policyengine_uk"]
    assert {item.name for item in items if item.get_closest_marker("skip")} == (
        set() if engine_available else engine_tests
    )
    for item in items:
        skip = item.get_closest_marker("skip")
        if skip is not None:
            assert skip.kwargs["reason"] == "requires policyengine-us extra"


# --------------------------------------------------------------------------
# National / state / CD comparison view (national_and_cd_target_fit)
#
# The head-to-head previously emitted family, entity, value basis and period
# per row and nothing geographic, so the required-release-evidence item
# ``national_and_cd_target_fit``
# (us_runtime/native_survey_handoff.py:42-54) had no producer. These tests
# pin the view axis as a real partition of the canonical loss, pin its
# refusal behaviour, and pin that it decides nothing.
# --------------------------------------------------------------------------


def _hierarchy(name: str, *, level: str, geography_id: str):
    from microcosm.calibrate.hierarchy import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    provider = HierarchyNode(id="fixture_provider", label="Fixture provider")
    return CalibrationHierarchy(
        provider=provider,
        category=HierarchyCategory(
            id="fixture_category",
            label="Fixture category",
            provider_id=provider.id,
        ),
        geography=HierarchyGeography(id=geography_id, label=geography_id, level=level),
        dimensions=(),
        target=HierarchyNode(id=name, label=name),
    )


def _geography_registry() -> TargetRegistry:
    """Four rows, one per resolution route the view classifier declares."""

    return TargetRegistry(
        [
            # Ledger spells national "country"; the compiler renames it.
            TargetSpec(
                name="national_income",
                entity="household",
                value=500.0,
                measure="m_income",
                period=2024,
                source="fixture",
                family="fixture_family",
                metadata={
                    "ledger_geography_level": "country",
                    "ledger_geography_id": "0100000US",
                },
            ),
            # The hierarchy geography tier outranks metadata.
            TargetSpec(
                name="state_income",
                entity="household",
                value=400.0,
                measure="m_income",
                period=2024,
                source="fixture",
                family="fixture_family",
                hierarchy=_hierarchy(
                    "state_income", level="state", geography_id="0400000US06"
                ),
            ),
            # CD membership named only by the shared classifier's layout
            # route, which this module reuses rather than restates.
            TargetSpec(
                name="cd_flagged",
                entity="household",
                value=12.0,
                measure="n_flagged",
                period=2024,
                source="fixture",
                family="fixture_family",
                metadata={
                    "measure_mode": "indicator_sum",
                    "ledger_layout_groupby_dimension": (
                        "irs_soi.congressional_district"
                    ),
                },
            ),
            # No declared geographic evidence at all.
            TargetSpec(
                name="unscoped_income",
                entity="household",
                value=300.0,
                measure="m_income",
                period=2024,
                source="fixture",
                family="fixture_family",
            ),
        ],
        country="us",
    )


def _geography_head_to_head(module, monkeypatch) -> dict[str, object]:
    _patch_release_seams(module, monkeypatch)
    yardstick = _fixture_yardstick(module, _geography_registry())
    incumbent = _fixture_artifact(
        module, sha256="c" * 64, measure_values=(100.0, 300.0)
    )
    candidate = _fixture_artifact(
        module, sha256="d" * 64, measure_values=(200.0, 290.0)
    )
    artifacts = {incumbent.h5_path: incumbent, candidate.h5_path: candidate}
    monkeypatch.setattr(module, "compile_yardstick", lambda **kwargs: yardstick)
    monkeypatch.setattr(module, "load_artifact", lambda path, **kwargs: artifacts[path])
    return module.score_head_to_head(
        incumbent=incumbent.h5_path,
        candidate=candidate.h5_path,
        ledger_facts=Path("/fixture/facts.jsonl"),
        congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
        maximum_microsim_batch_size=1,
    )


def test_view_levels_match_the_native_measurement_kernel_scopes() -> None:
    """One vocabulary for the measurement side and the comparison side.

    ``graph_fiscal_measurement`` is a frozen graph kernel; its ``_LEVELS`` is
    not edited to share a constant, so this test is what keeps the two from
    drifting apart.
    """

    from microcosm.build.us_runtime import graph_fiscal_measurement
    from microcosm.build.us_runtime.target_geography_view import (
        UNRESOLVED_GEOGRAPHY_VIEW_LEVEL,
        US_TARGET_GEOGRAPHY_VIEW_LEVELS,
        US_TARGET_GEOGRAPHY_VIEW_ORDER,
    )

    assert set(US_TARGET_GEOGRAPHY_VIEW_LEVELS) == graph_fiscal_measurement._LEVELS
    assert UNRESOLVED_GEOGRAPHY_VIEW_LEVEL not in US_TARGET_GEOGRAPHY_VIEW_LEVELS
    assert US_TARGET_GEOGRAPHY_VIEW_ORDER == (
        *US_TARGET_GEOGRAPHY_VIEW_LEVELS,
        UNRESOLVED_GEOGRAPHY_VIEW_LEVEL,
    )


def test_view_resolution_routes_and_refusals() -> None:
    from microcosm.build.us_runtime.target_geography_view import (
        us_target_spec_geography_view,
    )

    views = {
        spec.name: us_target_spec_geography_view(spec)
        for spec in _geography_registry().specs
    }

    assert views["national_income"].level == "national"
    assert views["national_income"].level_source == "ledger_geography_level"
    assert views["national_income"].geography_id == "0100000US"

    assert views["state_income"].level == "state"
    assert views["state_income"].level_source == "hierarchy_geography"
    assert views["state_income"].geography_id == "0400000US06"

    assert views["cd_flagged"].level == "congressional_district"
    assert views["cd_flagged"].level_source == "congressional_district_evidence"
    assert views["cd_flagged"].congressional_district_evidence is True

    # A row nothing identifies is refused, never absorbed into national.
    assert views["unscoped_income"].level == "unresolved"
    assert views["unscoped_income"].level_source == "none"
    assert views["unscoped_income"].resolved is False


def test_unknown_declared_level_is_refused_not_guessed() -> None:
    from microcosm.build.us_runtime.target_geography_view import (
        us_target_geography_view,
    )

    # "county" is a real ledger level with no advertised household scope. It
    # must not fall through to national, and it must not be invented into a
    # fourth view.
    view = us_target_geography_view(
        name="county_row",
        metadata={"ledger_geography_level": "county", "ledger_geography_id": "x"},
    )
    assert view.level == "unresolved"
    assert view.geography_id == ""


def test_explicit_level_outranks_cd_evidence_and_reports_the_disagreement() -> None:
    from microcosm.build.us_runtime.target_geography_view import (
        us_target_geography_view,
    )

    view = us_target_geography_view(
        name="irs_soi.ty2022.congressional_district_rollup.state_total",
        metadata={"ledger_geography_level": "state", "ledger_geography_id": "06"},
    )
    assert view.level == "state"
    assert view.congressional_district_evidence is True
    assert view.disagrees_with_congressional_district_evidence is True


def test_geography_view_rollup_partitions_the_canonical_loss(monkeypatch) -> None:
    """A view's contributions are shares of the one aggregate, not a rerun."""

    module = _load_head_to_head_module()
    payload = _geography_head_to_head(module, monkeypatch)

    for role in ("incumbent", "candidate"):
        fiscal = payload["artifacts"][role]["fiscal"]
        rollup = fiscal["by_geography_level"]
        assert [group["geography_level"] for group in rollup] == [
            "national",
            "state",
            "congressional_district",
            "unresolved",
        ]
        assert sum(group["target_count"] for group in rollup) == fiscal["target_count"]
        assert sum(group["loss_contribution"] for group in rollup) == pytest.approx(
            fiscal["weighted_loss"]
        )
        assert sum(group["weight_share"] for group in rollup) == pytest.approx(1.0)
        # The state row is the only one carrying a hierarchy geography id.
        by_level = {group["geography_level"]: group for group in rollup}
        assert by_level["state"]["distinct_geography_id_count"] == 1
        assert by_level["unresolved"]["distinct_geography_id_count"] == 0

    resolution = payload["artifacts"]["incumbent"]["fiscal"][
        "geography_view_resolution"
    ]
    assert resolution["unresolved_target_count"] == 1
    assert resolution["unresolved_target_examples"] == ["unscoped_income"]
    assert resolution["congressional_district_evidence_disagreement_count"] == 0
    assert resolution["level_source_counts"] == {
        "congressional_district_evidence": 1,
        "hierarchy_geography": 1,
        "ledger_geography_level": 1,
        "none": 1,
    }


def test_comparison_view_counts_and_deltas_reconcile(monkeypatch) -> None:
    module = _load_head_to_head_module()
    payload = _geography_head_to_head(module, monkeypatch)
    comparison = payload["comparison"]
    views = comparison["by_geography_level"]["views"]
    counts = comparison["per_target_absolute_relative_error"]
    loss = comparison["fiscal_weighted_loss"]

    assert comparison["by_geography_level"]["required_release_evidence_item"] == (
        "national_and_cd_target_fit"
    )
    # Still evidence, not a verdict: the view axis adds no threshold.
    assert comparison["no_threshold_applied"] is True
    assert comparison["decision"] == "owner_decides_flip"

    assert (
        sum(view["candidate_lower_count"] for view in views)
        == (counts["candidate_lower_count"])
    )
    assert sum(view["equal_count"] for view in views) == counts["equal_count"]
    assert (
        sum(view["incumbent_lower_count"] for view in views)
        == (counts["incumbent_lower_count"])
    )
    assert sum(view["incumbent_loss_contribution"] for view in views) == pytest.approx(
        loss["incumbent"]
    )
    assert sum(view["candidate_loss_contribution"] for view in views) == pytest.approx(
        loss["candidate"]
    )
    assert sum(
        view["candidate_minus_incumbent_loss_contribution"] for view in views
    ) == pytest.approx(loss["candidate_minus_incumbent"])
    for view in views:
        assert view["candidate_minus_incumbent_loss_contribution"] == pytest.approx(
            view["candidate_loss_contribution"] - view["incumbent_loss_contribution"]
        )

    # n_flagged is the only count row and the only CD row; both artifacts
    # share its inputs, so the CD view is exactly tied and never regresses.
    cd = next(
        view for view in views if view["geography_level"] == "congressional_district"
    )
    assert cd["target_count"] == 1
    assert cd["equal_count"] == 1
    assert cd["candidate_minus_incumbent_loss_contribution"] == pytest.approx(0.0)
    assert cd["worst_candidate_regression_target"] is None

    # m_income rows: incumbent 7000, candidate 7800, every target below both,
    # so all three amount rows cap at 1.0 on each side and tie.
    national = next(view for view in views if view["geography_level"] == "national")
    assert national["target_count"] == 1
    assert national["candidate_minus_incumbent_loss_contribution"] == pytest.approx(0.0)

    # Every target appears once, tagged with the view it was scored in.
    assert sorted(
        (row["name"], row["geography_level"]) for row in counts["targets"]
    ) == [
        ("cd_flagged", "congressional_district"),
        ("national_income", "national"),
        ("state_income", "state"),
        ("unscoped_income", "unresolved"),
    ]


def test_comparison_refuses_a_view_that_differs_between_artifacts() -> None:
    module = _load_head_to_head_module()

    def _side(level: str) -> dict[str, object]:
        return {
            "fiscal": {
                "weighted_loss": 0.5,
                "targets": [
                    {
                        "name": "row",
                        "period": 2024,
                        "geography_level": level,
                        "geography_level_source": "ledger_geography_level",
                        "congressional_district_evidence": False,
                        "geography_id": "",
                        "absolute_relative_error": 0.1,
                        "target_loss_weight_share": 1.0,
                        "weighted_loss_contribution": 0.5,
                    }
                ],
            },
            "terminal_battery": {"status": "inapplicable"},
        }

    with pytest.raises(ValueError, match="geography view differs"):
        module._comparison_payload(_side("national"), _side("congressional_district"))


def test_scorecard_markdown_renders_the_view_axis(monkeypatch, tmp_path) -> None:
    module = _load_head_to_head_module()
    payload = _geography_head_to_head(module, monkeypatch)
    assert payload["schema_version"] == 4

    markdown = module.render_markdown(payload)
    assert "## incumbent: loss by national / state / CD view" in markdown
    assert "## candidate: loss by national / state / CD view" in markdown
    assert "### National / state / CD view" in markdown
    assert "`national_and_cd_target_fit`" in markdown
    assert "Unresolved-scope rows: **1**" in markdown
    assert "never counted \nas national" in markdown or (
        "never counted " in markdown and "as national" in markdown
    )

    # Rendering stays deterministic byte-for-byte through the writer.
    first = module.write_scorecard(payload, tmp_path / "one" / "scorecard")
    second = module.write_scorecard(payload, tmp_path / "two" / "scorecard")
    for path_one, path_two in zip(first, second, strict=True):
        assert path_one.read_bytes() == path_two.read_bytes()
