"""The shared target compiler preserves pins, substitutions, and parity refusal."""

import argparse
from types import SimpleNamespace

import pytest
from test_us_fiscal_refresh_builder import _load_builder_module


def _args(tmp_path):
    return argparse.Namespace(
        ledger_facts=tmp_path / "facts.jsonl",
        ledger_facts_sha256="a" * 64,
        ledger_manifest_sha256="b" * 64,
        age_targets=True,
        allow_unaged_dollar_targets=False,
    )


@pytest.mark.parametrize("parity_passes", [True, False])
def test_shared_target_compilation_preserves_pins_configuration_and_gate_order(
    tmp_path, monkeypatch, parity_passes
):
    builder = _load_builder_module()
    args = _args(tmp_path)
    facts = object()
    artifact = SimpleNamespace(facts=facts)
    initial_registry, substituted_registry, crosswalk = object(), object(), object()
    records = ("invented-substitution",)
    gate = SimpleNamespace(passed=parity_passes, failures=("invented missing family",))
    events = []

    def load(path, **kwargs):
        assert path == args.ledger_facts
        assert kwargs == {
            "expected_facts_sha256": "a" * 64,
            "expected_manifest_sha256": "b" * 64,
        }
        events.append("load")
        return artifact

    def compile_targets(actual, **kwargs):
        assert actual is facts
        assert kwargs == {
            "target_period": builder.PERIOD,
            "congressional_district_vintage_crosswalk": crosswalk,
            "age_targets": True,
            "allow_unaged_dollar_targets": False,
        }
        events.append("compile")
        return initial_registry

    def substitute(actual):
        assert actual is initial_registry
        events.append("substitute")
        return substituted_registry, records

    def check_manifest(*, registry):
        assert registry is substituted_registry
        events.append("manifest")

    def check_parity(actual):
        assert actual is substituted_registry
        events.append("parity")
        return gate

    monkeypatch.setattr(builder, "load_ledger_consumer_artifact", load)
    monkeypatch.setattr(builder, "compile_us_fiscal_target_registry", compile_targets)
    monkeypatch.setattr(
        builder, "apply_us_medicaid_enrollment_substitutions", substitute
    )
    monkeypatch.setattr(
        builder, "assert_target_parity_manifest_current", check_manifest
    )
    monkeypatch.setattr(builder, "us_release_target_parity_gate", check_parity)
    if parity_passes:
        result = builder._compile_fiscal_release_target_registry(
            args, congressional_district_vintage_crosswalk=crosswalk
        )
        assert all(
            a is b
            for a, b in zip(
                result, (artifact, substituted_registry, records, gate), strict=True
            )
        )
    else:
        with pytest.raises(
            RuntimeError, match="Target parity coverage failed: invented missing family"
        ):
            builder._compile_fiscal_release_target_registry(
                args, congressional_district_vintage_crosswalk=crosswalk
            )
    assert events == ["load", "compile", "substitute", "manifest", "parity"]


def test_failed_ledger_identity_stops_before_compilation(tmp_path, monkeypatch):
    builder = _load_builder_module()

    def refuse(*a, **kw):
        raise ValueError("invented source identity mismatch")

    def forbidden(*a, **kw):
        raise AssertionError("must not compile refused source")

    monkeypatch.setattr(builder, "load_ledger_consumer_artifact", refuse)
    monkeypatch.setattr(builder, "compile_us_fiscal_target_registry", forbidden)
    with pytest.raises(ValueError, match="source identity mismatch"):
        builder._compile_fiscal_release_target_registry(
            _args(tmp_path), congressional_district_vintage_crosswalk=None
        )
