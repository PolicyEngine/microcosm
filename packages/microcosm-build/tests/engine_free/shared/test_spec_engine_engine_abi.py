"""Tests split from packages/microcosm-build/tests/test_spec_engine_engine_abi.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_engine_abi import *
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def test_engine_version_is_owned_only_by_the_fresh_generated_lock(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    payload = engine_abi_lock_payload_from_domains(domains)

    assert payload["engine"] == {
        "package": "policyengine-us",
        "version": "1.2.3",
    }
    receipt = payload["remaining_stage_input_manifest"]["receipt"]
    assert receipt["ssi_dependency_contract"]["engine_version_ref"] == (
        ENGINE_VERSION_REF
    )
    assert receipt["engine_input_projection_contract"]["engine_version_ref"] == (
        ENGINE_VERSION_REF
    )
    assert "engine_version" not in receipt["ssi_dependency_contract"]
    assert "engine_version" not in receipt["engine_input_projection_contract"]
    assert receipt["ssi_dependency_contract"]["sha256"] == "0" * 64
    assert receipt["engine_input_projection_contract"]["sha256"] == "1" * 64
    assert receipt["engine_input_projection_contract"]["defaults_sha256"] == ("2" * 64)
    assert receipt["sha256"] == "4" * 64
    assert _count_scalar(payload, "1.2.3") == 1
    assert "value" not in domains["vintages"]["records"][0]


def test_nested_receipt_version_must_match_the_single_engine_pin(
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched_manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        manifest = _fake_remaining_stage_input_manifest()
        manifest["receipt"]["ssi_dependency_contract"]["engine_version"] = "9.9.9"
        return manifest

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        mismatched_manifest,
    )
    with pytest.raises(
        SpecValidationError,
        match="engine version differs from the exact generated engine pin",
    ):
        engine_abi_lock_payload_from_domains(_fake_domains())


def test_program_variable_mapping_must_be_total_and_injective(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    programs = domains["take_up"]["programs"]
    programs[1]["variable"] = programs[0]["variable"]
    with pytest.raises(SpecValidationError, match="must be injective"):
        engine_abi_lock_payload_from_domains(domains)

    domains = _fake_domains()
    domains["take_up"]["programs"][0]["variable"] = "unknown_variable"
    with pytest.raises(
        SpecValidationError, match="not total over the fresh engine ABI"
    ):
        engine_abi_lock_payload_from_domains(domains)


def test_stale_or_noncanonical_generated_lock_is_refused(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains = _fake_domains()
    lock_path = tmp_path / "engine_abi.lock.json"
    lock_path.write_bytes(engine_abi_lock_bytes_from_domains(domains))
    registry = load_schema_registry()
    assert_engine_abi_lock_current(
        tmp_path,
        domains,
        schema_registry=registry,
    )

    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "9.9.9",
    )

    def bumped_manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        manifest = _fake_remaining_stage_input_manifest()
        receipt = manifest["receipt"]
        receipt["ssi_dependency_contract"]["engine_version"] = "9.9.9"
        receipt["engine_input_projection_contract"]["engine_version"] = "9.9.9"
        return manifest

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        bumped_manifest,
    )
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )

    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "1.2.3",
    )
    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        _fake_remaining_stage_input_manifest,
    )

    stale = engine_abi_lock_payload_from_domains(domains)
    stale["programs"]["program_00"]["default"] = False
    lock_path.write_text(json.dumps(stale, sort_keys=True), encoding="utf-8")
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )

    stale = engine_abi_lock_payload_from_domains(domains)
    stale["remaining_stage_input_manifest"]["rows"][0]["provision"] = (
        "mutated_provision"
    )
    lock_path.write_bytes(
        json.dumps(stale, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )


def test_remaining_stage_manifest_lock_rows_are_closed_world(
    fake_fresh_engine: None,
) -> None:
    payload = engine_abi_lock_payload_from_domains(_fake_domains())
    payload["remaining_stage_input_manifest"]["rows"][0]["undeclared"] = True

    with pytest.raises(SpecValidationError, match="undeclared"):
        load_schema_registry().validate(
            payload,
            "locks.schema.json#/$defs/engine_abi_lock",
        )


def test_remaining_stage_engine_version_refs_are_closed_and_exact(
    fake_fresh_engine: None,
) -> None:
    payload = engine_abi_lock_payload_from_domains(_fake_domains())
    receipt = payload["remaining_stage_input_manifest"]["receipt"]
    dependency = receipt["ssi_dependency_contract"]
    dependency["engine_version_ref"]["pointer"] = "/engine/package"

    with pytest.raises(SpecValidationError, match="/engine/version"):
        load_schema_registry().validate(
            payload,
            "locks.schema.json#/$defs/engine_abi_lock",
        )


def test_generated_remaining_stage_manifest_is_a_canonical_identity_binding(
    fake_fresh_engine: None,
) -> None:
    before = engine_abi_lock_payload_from_domains(_fake_domains())
    after = copy.deepcopy(before)
    after["remaining_stage_input_manifest"]["rows"][0]["provision"] = (
        "mutated_provision"
    )

    def identity(payload: object) -> str:
        return sha256_json(
            spec_envelope(
                country="us",
                schema_version=1,
                normative_files={},
                resolved_bindings={
                    "generated_authorities": {"engine_abi_lock": payload}
                },
            )
        )

    assert identity(before) != identity(after)


def test_lock_mutation_does_not_change_authored_domain_payloads(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    before = copy.deepcopy(domains)
    engine_abi_lock_payload_from_domains(domains)
    assert domains == before


def test_engine_absent_environment_validates_lock_structurally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the engine distribution, spec loads validate the committed
    lock structurally (present, schema-valid, canonical bytes) instead of
    failing on the currency attestation — the wheels gate's venv installs
    no engine by design. Currency stays fail-closed wherever the engine
    exists."""

    import importlib.metadata as im

    from microcosm.build import country_spec as cs
    from microcosm.build.spec_engine import engine_abi

    def absent(package: str) -> str:
        raise im.PackageNotFoundError(package)

    monkeypatch.setattr(engine_abi, "_installed_engine_version", absent)
    # Bypass the per-process spec cache: a spec an earlier test loaded with the
    # engine present would otherwise come back without this path running.
    cs._load_packaged_country_spec.cache_clear()
    try:
        spec = cs.load_country_spec("us")
    finally:
        cs._load_packaged_country_spec.cache_clear()
    assert spec.country == "us"


def test_engine_absent_environment_still_refuses_a_tampered_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.metadata as im
    import json as json_module

    from microcosm.build.spec_engine import engine_abi
    from microcosm.build.spec_engine.loader import load_schema_registry

    def absent(package: str) -> str:
        raise im.PackageNotFoundError(package)

    monkeypatch.setattr(engine_abi, "_installed_engine_version", absent)
    registry = load_schema_registry()
    us_root = _TEST_PATHS.package / "src" / "microcosm" / "build" / "us"
    spec_dir = us_root / "spec"
    lock_path = tmp_path / engine_abi.ENGINE_ABI_LOCK_FILENAME
    parsed = json_module.loads(
        (us_root / engine_abi.ENGINE_ABI_LOCK_FILENAME).read_bytes()
    )
    lock_path.write_bytes(
        json_module.dumps(parsed, indent=3).encode() + b"\n"
    )  # non-canonical byte form
    import yaml

    domains = {
        "take_up": yaml.safe_load((spec_dir / "take_up.yaml").read_text()),
        "sources": yaml.safe_load((spec_dir / "sources.yaml").read_text()),
    }
    with pytest.raises(Exception, match="canonical"):
        engine_abi.assert_engine_abi_lock_current(
            tmp_path, domains, schema_registry=registry
        )
