"""Source pins and unreduced-register ownership in the full UK target path."""

import hashlib
import json
from dataclasses import replace
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
    monkeypatch.setattr(runtime, "load_uk_chronicle_feed", lambda: pin)
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
    # Nor a support floor (#1095 U3, tested in test_uk_support_floor): the
    # packaged one is signed after these tests' evaluation date.
    monkeypatch.setattr(runtime, "load_uk_support_floor_declaration", lambda path: None)
    # The national reconciliation (#1123) has its own tests; these registers
    # carry no geography, so it passes them through here.
    monkeypatch.setattr(
        runtime,
        "reconcile_uk_national_registry",
        lambda registry: (registry, {"rows_moved_by_exact_signature": 0}),
    )
    # Nor do they carry fact periods: the hold declarations (#1123) are
    # checked against the real compile in test_uk_uprating_holds.
    monkeypatch.setattr(
        runtime,
        "assert_uk_uprating_holds_declared",
        lambda registry, **kwargs: {"scope": kwargs["scope"]},
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
    # The calibration year (2024 here) and the one validation period, 2025.
    # The 2023 production surface left the set with its gate.
    assert calls == [
        ("national", 2024),
        ("national", 2025),
        ("local", 2024),
        ("local", 2025),
    ]
    assert result["register_completeness"]["compiled_reference_count"] == 2
    assert result["register_completeness"]["approved_reference_count"] == 1
    assert result["register_completeness"]["validation_periods"] == {
        "national": [2025],
        "local": [2025],
    }
    assert result["register_completeness"]["validation_periods_unsupported"] == {
        "national": {},
        "local": {},
    }
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


def test_current_national_and_local_pins_share_one_reviewed_identity():
    # main (#904) reads one committed pin, ``uk/chronicle_feed.json``, for the
    # national feed and the local census; the full build checks they agree.
    # The identity is the 272c987 feed re-pinned by microcosm#1359
    # (PolicyEngine/chronicle#391: the Gambling Commission's Gambling Survey
    # for Great Britain 2025 participation tables), on top of the 200bf6d feed
    # of microcosm#1359 (PolicyEngine/chronicle#384: HMRC alcohol and tobacco clearances,
    # betting and gaming receipts and Machine Games Duty liabilities, the
    # Gambling Commission's industry statistics and OBR's excise lines, with
    # the September Consumer Trends edition in place of June's), on top of the
    # abf0921 feed of microcosm#1300 (UC by payment indicator, OBR table 4.9
    # geography, DESNZ gas years and Northern Ireland energy, UV404 by output
    # area). It holds the 735820c HVCTS inputs microcosm#1240 pinned, every
    # 942a1fa row of microcosm#1123 and the 1ee7dfe rows of microcosm#1113.
    national = runtime.load_uk_chronicle_feed()
    local = runtime.load_uk_local_chronicle_pin()
    assert (
        national.facts_sha256
        == local["facts_sha256"]
        == ("39ee4926f650f3d38d8c5220a750af4037339359179d70279d170cfbb3b42775")
    )
    assert (
        national.manifest_sha256
        == local["manifest_sha256"]
        == ("a663586f8b3c6d13aa6fc5f64991951efe528b83bd5ea63c3b7267a20fff1c5f")
    )
    assert (
        national.source_commit
        == local["source_commit"]
        == "e56af9a501e71c9b954c84678a81566feb5b33f9"
    )
    assert national.fact_row_count == local["fact_row_count"] == 1506587


def test_chronicle_source_codec_validates_directory_manifest(tmp_path):
    from microcosm.build.uk_runtime.graph_targets import register_uk_source_codecs
    from microcosm.graph.codecs import SOURCE_CODECS

    # Registration is explicit (never an import side effect), so this test
    # registers before reading the process-global registry.
    register_uk_source_codecs()
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


def test_the_bound_and_frozen_registers_are_the_reconciled_ones(
    prepared, monkeypatch, tmp_path
):
    national = prepared[0]
    reconciled = TargetRegistry(
        [replace(spec, value=2.0) for spec in national.specs], country="uk"
    )
    seen = []
    monkeypatch.setattr(
        runtime,
        "reconcile_uk_national_registry",
        lambda registry: (
            seen.append(registry) or reconciled,
            {"rows_moved_by_exact_signature": 2},
        ),
    )
    excluded_from = []
    monkeypatch.setattr(
        runtime,
        "apply_uk_calibration_measure_exclusions",
        lambda registry, exclusions, now: (
            excluded_from.append(registry) or prepared[1],
            {"excluded": {"reason": "reviewed"}},
        ),
    )
    result = _load()
    # Only the calibration year is reconciled; validation periods stay as
    # compiled, since compile parity measures the facts themselves.
    assert seen == [national]
    assert excluded_from == [reconciled]
    assert result["band_edge_registry"] is reconciled
    assert result["national_reconciliation"] == {"rows_moved_by_exact_signature": 2}
    completeness = result["register_completeness"]
    assert completeness["compiled_registry_version"] == national.version
    assert completeness["reconciled_registry_version"] == reconciled.version
    path = tmp_path / "register.json"
    reconciled.to_json(path)
    _load(register_json=path)
    national.to_json(path)
    with pytest.raises(ValueError, match="full national register differs"):
        _load(register_json=path)


def test_holds_are_checked_on_the_full_compiled_and_local_registers(
    prepared, monkeypatch
):
    national, approved, local, artifact, calls = prepared
    seen = []
    monkeypatch.setattr(
        runtime,
        "assert_uk_uprating_holds_declared",
        lambda registry, **kwargs: (
            seen.append((kwargs["scope"], registry, kwargs["calibration_period"]))
            or {"scope": kwargs["scope"]}
        ),
    )
    result = _load()
    # The full compiled register, not the approved one: a measure exclusion
    # expires, and its target must re-enter already declared.
    assert seen == [("national", national, 2024), ("local", local, 2024)]
    assert result["uprating_holds"] == {
        "national": {"scope": "national"},
        "local": {"scope": "local"},
    }


def test_validation_reference_compilation_is_fail_closed(prepared, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "compile_uk_target_registry",
        lambda *a, **kw: SimpleNamespace(
            registry=prepared[0], unsupported=({"target": "missing"},)
        ),
    )
    # 2025 is a validation period and is skipped; the calibration year is not.
    with pytest.raises(ValueError, match="failed to compile for 2024"):
        _load()


def test_historical_validation_period_without_facts_is_recorded_not_fatal(
    prepared, monkeypatch
):
    national, approved, local, artifact, calls = prepared
    partial = _registry("retained")

    def compile_national(facts, *, target_period):
        calls.append(("national", target_period))
        if target_period == 2025:
            return SimpleNamespace(
                registry=partial,
                unsupported=(
                    {"name": "obr.income_tax", "period": 2025, "reason": "no fact"},
                ),
            )
        return SimpleNamespace(registry=national, unsupported=())

    monkeypatch.setattr(runtime, "compile_uk_target_registry", compile_national)
    result = _load()
    # The partial 2025 registry stays available to the parity gates, as the
    # release-cut producer keeps it; the missing references are recorded.
    assert set(result["uk_ledger_compiled_registries"]) == {2024, 2025}
    assert result["uk_ledger_compiled_registries"][2025] is partial
    # The compile's report travels beside each registry, one entry for every
    # compiled period, so the parity gates can classify the missing row.
    assert result["uk_ledger_compile_unsupported"] == {
        2024: (),
        2025: ({"name": "obr.income_tax", "period": 2025, "reason": "no fact"},),
    }
    assert set(result["uk_ledger_compile_local_unsupported"]) == set(
        result["uk_ledger_compiled_local_registries"]
    )
    completeness = result["register_completeness"]
    assert completeness["validation_periods"] == {
        "national": [2025],
        "local": [2025],
    }
    assert completeness["validation_periods_unsupported"] == {
        "national": {"2025": ["obr.income_tax"]},
        "local": {},
    }


_REGION_ROW = "microcosm.high_value_homes.over_2m_by_region"


def _with_region_rows(*names):
    """A compiled register carrying local-line-only region rows (#1301)."""

    region = _registry(f"{_REGION_ROW}@E12000007", f"{_REGION_ROW}@W92000004")
    region = TargetRegistry(
        [
            replace(spec, metadata={"contract_target_id": _REGION_ROW})
            for spec in region.specs
        ],
        country="uk",
    )
    plain = _registry(*names)
    return TargetRegistry([*plain.specs, *region.specs], country="uk"), plain


def test_the_national_line_drops_local_only_rows_before_reconciliation(monkeypatch):
    compiled, plain = _with_region_rows("retained", "excluded")
    artifact = SimpleNamespace(
        facts=({"feed": True},),
        facts_sha256="a" * 64,
        manifest_sha256="b" * 64,
        provenance=lambda: {"artifact_id": "synthetic"},
    )
    monkeypatch.setattr(
        runtime, "load_ledger_consumer_artifact", lambda *a, **kw: artifact
    )
    monkeypatch.setattr(
        runtime,
        "require_committed_uk_chronicle_feed_pin",
        lambda facts_sha256, **kwargs: SimpleNamespace(to_dict=lambda: {}),
    )
    compiled_facts = []

    def compile_national(facts, *, target_period):
        compiled_facts.append(tuple(facts))
        return SimpleNamespace(registry=compiled, unsupported=())

    monkeypatch.setattr(runtime, "compile_uk_target_registry", compile_national)
    reconciled_from = []
    monkeypatch.setattr(
        runtime,
        "reconcile_uk_national_registry",
        lambda registry: (reconciled_from.append(registry) or registry, {}),
    )
    monkeypatch.setattr(runtime, "uk_cross_grain_coverage_receipt", lambda: {})
    monkeypatch.setattr(runtime, "assert_uk_subset_relations", lambda registry: [])
    monkeypatch.setattr(
        runtime, "load_uk_calibration_measure_exclusions", lambda path: ()
    )
    monkeypatch.setattr(runtime, "load_uk_support_floor_declaration", lambda path: None)
    monkeypatch.setattr(
        runtime,
        "assert_uk_uprating_holds_declared",
        lambda registry, **kwargs: {"scope": kwargs["scope"]},
    )
    monkeypatch.setattr(runtime, "_ledger_provenance", lambda artifact: {})

    result = runtime.load_uk_national_target_inputs(
        "facts.jsonl",
        calibration_year=2025,
        exclusions_evaluated_on=date(2026, 10, 16),
    )

    # Every reference compiles, on the feed and the derived facts together.
    derived = runtime.uk_derived_fact_rows()
    assert compiled_facts == [({"feed": True}, *derived)]
    # The region rows leave before reconciliation: the national surface and
    # its version are the ones it had without them.
    (reconciled,) = reconciled_from
    assert [spec.name for spec in reconciled.specs] == ["retained", "excluded"]
    assert result["national_registry"].version == plain.version
    completeness = result["register_completeness"]
    assert completeness["compiled_registry_version"] == plain.version
    assert completeness["compiled_reference_count"] == 2
    assert completeness["calibration_line"] == {
        "line": "national",
        "all_lines_registry_version": compiled.version,
        "all_lines_reference_count": 4,
        "dropped_reference_count": 2,
        "dropped_target_ids": [_REGION_ROW],
    }
    # The receipt names every derived resource the compile read.
    resources = runtime.load_uk_derived_facts()
    assert sorted(result["derived_fact_provenance"]) == sorted(resources)
    for stem, facts in resources.items():
        assert result["derived_fact_provenance"][stem]["rows_sha256"] == (
            facts.rows_sha256
        )


def test_the_full_line_binds_the_region_rows(prepared, monkeypatch):
    national, approved, local, artifact, calls = prepared
    compiled, _plain = _with_region_rows("retained", "excluded")
    seen_facts = []

    def compile_national(facts, *, target_period):
        seen_facts.append(tuple(facts))
        return SimpleNamespace(registry=compiled, unsupported=())

    monkeypatch.setattr(runtime, "compile_uk_target_registry", compile_national)
    reconciled_from = []
    monkeypatch.setattr(
        runtime,
        "reconcile_uk_national_registry",
        lambda registry: (reconciled_from.append(registry) or registry, {}),
    )
    monkeypatch.setattr(
        runtime,
        "apply_uk_calibration_measure_exclusions",
        lambda registry, exclusions, now: (registry, {}),
    )
    result = _load()
    derived = runtime.uk_derived_fact_rows()
    assert all(facts == (*artifact.facts, *derived) for facts in seen_facts)
    # The local line binds every row: nothing leaves, the registry is the
    # compiled object itself.
    assert reconciled_from == [compiled]
    assert result["national_registry"] is compiled
    line = result["register_completeness"]["calibration_line"]
    assert line["line"] == "local"
    assert line["dropped_reference_count"] == 0
