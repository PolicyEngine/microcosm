"""Source pins and unreduced-register ownership in the full UK target path."""

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from microcosm.build.ledger_artifact import CONSUMER_ARTIFACT_SCHEMA_VERSION
from microcosm.build.uk_runtime import full_targets as runtime
from microcosm.calibrate import TargetRegistry, TargetSpec


def _registry(*names):
    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="person",
                value=1.0,
                measure="age",
                period=2024,
                source="test",
                family="population",
                metadata={"contract_target_id": name},
            )
            for name in names
        ],
        country="uk",
    )


@pytest.fixture
def prepared(monkeypatch):
    national = _registry("retained", "excluded")
    approved = _registry("retained")
    local = _registry("local")
    calls = []
    artifact = SimpleNamespace(
        facts=({"test": True},), facts_sha256="a" * 64, manifest_sha256="b" * 64
    )
    pin = SimpleNamespace(
        facts_sha256=artifact.facts_sha256,
        manifest_sha256=artifact.manifest_sha256,
        fact_row_count=1,
        to_dict=lambda: {
            "facts_sha256": artifact.facts_sha256,
            "manifest_sha256": artifact.manifest_sha256,
        },
    )
    monkeypatch.setattr(runtime, "load_uk_national_chronicle_feed", lambda: pin)
    monkeypatch.setattr(
        runtime,
        "load_uk_local_chronicle_pin",
        lambda: {
            "facts_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "fact_row_count": 1,
        },
    )
    monkeypatch.setattr(
        runtime, "load_ledger_consumer_artifact", lambda *a, **kw: artifact
    )
    monkeypatch.setattr(runtime, "load_uk_local_area_crosswalk", lambda: {})

    def compile_national(facts, *, target_period):
        calls.append(("national", target_period))
        return SimpleNamespace(registry=national, unsupported=())

    def compile_local(facts, *, target_period, crosswalk):
        calls.append(("local", target_period))
        return SimpleNamespace(registry=local, unsupported=())

    monkeypatch.setattr(runtime, "compile_uk_target_registry", compile_national)
    monkeypatch.setattr(runtime, "compile_uk_local_target_registry", compile_local)
    monkeypatch.setattr(
        runtime, "load_uk_calibration_measure_exclusions", lambda path: ()
    )
    monkeypatch.setattr(
        runtime,
        "apply_uk_calibration_measure_exclusions",
        lambda registry, exclusions, now: (
            approved,
            {"excluded": {"reason": "reviewed"}},
        ),
    )
    return national, approved, local, artifact, calls


def _load(**kwargs):
    return runtime.load_uk_full_target_inputs(
        "facts.jsonl",
        calibration_year=2024,
        exclusions_evaluated_on=date(2026, 9, 10),
        **kwargs,
    )


def test_full_inputs_preserve_band_edges_and_validation_periods(prepared):
    national, approved, local, artifact, calls = prepared
    result = _load()
    assert result["artifact"] is artifact
    assert result["band_edge_registry"] is national
    assert result["national_registry"] is approved
    assert result["local_registry"] is local
    assert calls == [
        ("national", 2023),
        ("national", 2024),
        ("national", 2025),
        ("local", 2024),
        ("local", 2025),
    ]
    assert result["register_completeness"]["compiled_reference_count"] == 2
    assert result["register_completeness"]["approved_reference_count"] == 1
    assert result["register_completeness"]["compiled_local_reference_count"] == 1
    assert result["local_source_pin"]["fact_row_count"] == 1
    assert result["reviewed_unbound_higher_targets"] == {
        "excluded": {"reason": "reviewed"}
    }


@pytest.mark.parametrize("field", ["facts", "manifest"])
def test_source_pin_mismatch_refuses_before_read(prepared, monkeypatch, field):
    monkeypatch.setattr(
        runtime,
        "load_ledger_consumer_artifact",
        lambda *a, **kw: pytest.fail("source read before pin agreement"),
    )
    with pytest.raises(ValueError, match="committed national feed"):
        _load(**{f"expected_{field}_sha256": "c" * 64})


def test_loaded_source_mismatch_refuses(prepared):
    prepared[3].manifest_sha256 = "c" * 64
    with pytest.raises(ValueError, match="Ledger artifact"):
        _load()
    assert prepared[4] == []


def test_omitted_explicit_pins_still_bind_both_reviewed_source_hashes(
    prepared, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        runtime,
        "load_ledger_consumer_artifact",
        lambda path, **kwargs: calls.append(kwargs) or prepared[3],
    )
    _load()
    assert calls == [
        {"expected_facts_sha256": "a" * 64, "expected_manifest_sha256": "b" * 64}
    ]


def test_local_review_is_required_before_read_or_compilation(prepared, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "load_uk_local_chronicle_pin",
        lambda: {"facts_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "load_ledger_consumer_artifact",
        lambda *a, **kw: pytest.fail("source read before local review"),
    )
    with pytest.raises(ValueError, match="national and local feed pins disagree"):
        _load()
    assert prepared[4] == []


def test_wrong_fact_count_refuses_before_either_compiler(prepared):
    prepared[3].facts = ()
    with pytest.raises(ValueError, match="fact row count"):
        _load()
    assert prepared[4] == []


def test_current_national_and_local_pins_have_separate_reviewed_identities():
    national = runtime.load_uk_national_chronicle_feed()
    local = runtime.load_uk_local_chronicle_pin()
    assert (
        national.facts_sha256
        == local["facts_sha256"]
        == ("4a50ee9568a01bbb57f73d927084ed6b4b9e52249b51a2338455874ae6e382b5")
    )
    assert (
        national.manifest_sha256
        == local["manifest_sha256"]
        == ("a95d0ee9f87f36947eaecdb3de29cf81a91e47ccaa822fed42da677eedca877f")
    )
    assert local["source_commit"] == "ec7169b"
    assert national.fact_row_count == local["fact_row_count"] == 131450


def test_chronicle_source_codec_validates_directory_manifest(tmp_path):
    from microcosm.build.uk_runtime import graph_targets  # noqa: F401
    from microcosm.graph.codecs import SOURCE_CODECS

    payload = b'{"value": 1.0}\n'
    (tmp_path / "consumer_facts.jsonl").write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
                "facts_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    assert SOURCE_CODECS.load_bytes(runtime.CHRONICLE_SOURCE_CODEC, tmp_path) == payload
    (tmp_path / "consumer_facts.jsonl").write_bytes(b'{"value": 2.0}\n')
    with pytest.raises(ValueError, match="manifest hash"):
        SOURCE_CODECS.load_bytes(runtime.CHRONICLE_SOURCE_CODEC, tmp_path)


def test_frozen_register_compares_complete_not_measure_pruned_surface(
    prepared, tmp_path
):
    path = tmp_path / "register.json"
    prepared[0].to_json(path)
    assert (
        _load(register_json=path)["register_completeness"]["frozen_registry_version"]
        == prepared[0].version
    )
    prepared[1].to_json(path)
    with pytest.raises(ValueError, match="full national register differs"):
        _load(register_json=path)


def test_validation_reference_compilation_is_fail_closed(prepared, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "compile_uk_target_registry",
        lambda *a, **kw: SimpleNamespace(
            registry=prepared[0], unsupported=({"target": "missing"},)
        ),
    )
    with pytest.raises(ValueError, match="failed to compile for 2023"):
        _load()
