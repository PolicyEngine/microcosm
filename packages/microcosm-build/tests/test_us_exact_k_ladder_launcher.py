from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest

import microcosm.build.logbook as logbook_module
import microcosm.build.logbook_family as family_module
from microcosm.build.logbook import load_spool_rows
from microcosm.build.logbook_family import (
    LogbookFamily,
    load_family_spool,
    record_family,
)

FAMILY_ID = "12345678-1234-4234-9234-123456789abc"


@pytest.fixture(autouse=True)
def _isolated_logbook_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)


def _launcher_module():
    tools = Path(__file__).resolve().parents[3] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return importlib.import_module("build_us_exact_k_ladder_release")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_payload(
    *,
    pool_manifest_sha256: str = "a" * 64,
    requested_k: str | int = "N",
    release_id: str = "populace-us-2024-k8-fixture",
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "family": {"id": FAMILY_ID},
        "pool": {
            "release_id": "fixture-publication",
            "manifest_sha256": pool_manifest_sha256,
        },
        "ladder": {"k": requested_k, "seed": 17, "pi_hi": 0.95},
        "targets": {
            "ledger_facts": "ledger",
            "ledger_facts_sha256": "b" * 64,
            "ledger_manifest_sha256": "c" * 64,
            "incumbent_diagnostics": "incumbent.json",
            "incumbent_diagnostics_sha256": "d" * 64,
            "target_surface_sha256": "e" * 64,
        },
        "calibration": {
            "epochs": 3,
            "learning_rate": 0.02,
            "max_weight_ratio": 20.0,
            "l0_refit_lambda_share": 0.8,
            "l2_lambda": 0.0,
            "refit_l2_lambda": 0.0,
        },
        "release": {
            "id": release_id,
            "repo_id": "policyengine/populace-us",
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_packaged_release(
    out: Path,
    *,
    release_id: str,
    household_count: int,
) -> None:
    artifact_root = out / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_path = artifact_root / "populace_us_2024.h5"
    pd.DataFrame(
        {"household_id": range(household_count)},
    ).to_hdf(dataset_path, key="household", format="table")
    release_dir = out / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "build": {"build_id": release_id},
                "default_datasets": {"national": "populace_us_2024"},
                "artifacts": {
                    "populace_us_2024": {
                        "kind": "microdata",
                        "path": dataset_path.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_config_requires_explicit_seed_and_ratified_k(tmp_path: Path) -> None:
    launcher = _launcher_module()
    missing_seed = _config_payload()
    del missing_seed["ladder"]["seed"]
    path = _write_config(tmp_path, missing_seed)

    with pytest.raises(ValueError, match=r"missing=\['seed'\]"):
        launcher._read_config(path)

    bad_k = _config_payload(requested_k=57_241)
    path = _write_config(tmp_path, bad_k)
    with pytest.raises(ValueError, match="exactly 'N', 57240, or 20000"):
        launcher._read_config(path)

    unpinned_retry = _config_payload()
    unpinned_retry["targets"]["ssi_take_up_prior_weight_basis"] = "ssi.json"
    path = _write_config(tmp_path, unpinned_retry)
    with pytest.raises(ValueError, match="and its SHA-256 pin"):
        launcher._read_config(path)

    invalid_family = _config_payload()
    invalid_family["family"]["id"] = "not-a-uuid"
    path = _write_config(tmp_path, invalid_family)
    with pytest.raises(ValueError, match="family.id must be a canonical UUID"):
        launcher._read_config(path)


def test_config_rejects_k_larger_than_manifest_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config = launcher._read_config(
        _write_config(
            tmp_path,
            _config_payload(
                pool_manifest_sha256=_sha256(manifest),
                requested_k=57_240,
                release_id="populace-us-2024-k57240-fixture",
            ),
        )
    )
    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        lambda _, **_kwargs: {
            "publication_run_id": "fixture-publication",
            "agreement_gate": {"passed": True},
            "provenance_counts": {"household": {"rows": 50_000}},
        },
    )

    with pytest.raises(ValueError, match="k=57240 exceeds the pool size 50000"):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )


def test_pool_manifest_sha_pin_is_checked_on_loaded_bytes(tmp_path: Path) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config = launcher._read_config(
        _write_config(tmp_path, _config_payload(pool_manifest_sha256="a" * 64))
    )
    with pytest.raises(ValueError, match="Pool manifest SHA-256 mismatch"):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )


def test_n_resolves_to_realized_pool_size_with_valid_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    (tmp_path / "ledger").mkdir()
    incumbent = tmp_path / "incumbent.json"
    incumbent.write_text(
        json.dumps({"target_surface": {"sha256": "e" * 64}}),
        encoding="utf-8",
    )
    payload = _config_payload(pool_manifest_sha256=_sha256(manifest))
    payload["targets"]["incumbent_diagnostics_sha256"] = _sha256(incumbent)
    config = launcher._read_config(_write_config(tmp_path, payload))
    validated_manifest = {
        "publication_run_id": "fixture-publication",
        "agreement_gate": {"passed": True},
        "provenance_counts": {"household": {"rows": 8}},
    }

    def fake_load_manifest(_, *, expected_manifest_sha256):
        assert expected_manifest_sha256 == config.pool_manifest_sha256
        return validated_manifest

    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        fake_load_manifest,
    )

    k, observed_manifest = launcher._validate_pins_and_resolve_k(
        config=config,
        pool_manifest_path=manifest,
    )

    assert k == 8
    assert observed_manifest is validated_manifest


@pytest.mark.parametrize(
    ("bad_pin", "message"),
    (
        ("incumbent_sha", "Incumbent diagnostics SHA-256 mismatch"),
        ("target_surface", "target-surface SHA-256 mismatch"),
        ("ssi_prior_basis", "prior-weight basis SHA-256 mismatch"),
    ),
)
def test_incumbent_and_target_surface_pins_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_pin: str,
    message: str,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    (tmp_path / "ledger").mkdir()
    incumbent = tmp_path / "incumbent.json"
    incumbent.write_text(
        json.dumps({"target_surface": {"sha256": "e" * 64}}),
        encoding="utf-8",
    )
    payload = _config_payload(pool_manifest_sha256=_sha256(manifest))
    payload["targets"]["incumbent_diagnostics_sha256"] = _sha256(incumbent)
    prior_basis = tmp_path / "ssi.json"
    prior_basis.write_text("{}", encoding="utf-8")
    if bad_pin == "incumbent_sha":
        payload["targets"]["incumbent_diagnostics_sha256"] = "d" * 64
    elif bad_pin == "target_surface":
        payload["targets"]["target_surface_sha256"] = "f" * 64
    else:
        payload["targets"]["ssi_take_up_prior_weight_basis"] = "ssi.json"
        payload["targets"]["ssi_take_up_prior_weight_basis_sha256"] = "1" * 64
    config = launcher._read_config(_write_config(tmp_path, payload))
    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        lambda _, **_kwargs: {
            "publication_run_id": "fixture-publication",
            "agreement_gate": {"passed": True},
            "provenance_counts": {"household": {"rows": 8}},
        },
    )

    with pytest.raises(ValueError, match=message):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )


def test_launcher_arguments_are_accepted_by_the_house_builder_parser(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    payload = _config_payload(
        requested_k=20_000,
        release_id="populace-us-2024-k20000-fixture",
    )
    payload["targets"]["ssi_take_up_prior_weight_basis"] = "ssi.json"
    payload["targets"]["ssi_take_up_prior_weight_basis_sha256"] = "1" * 64
    config = launcher._read_config(_write_config(tmp_path, payload))

    argv = launcher._builder_argv(
        config=config,
        pool_manifest=tmp_path / "pool.manifest.json",
        out=tmp_path / "out",
        k=20_000,
    )
    parsed = launcher.fiscal_release._parse_args(argv)

    assert parsed.exact_k == 20_000
    assert parsed.seed == 17
    assert parsed.pool_release_id == "fixture-publication"
    assert parsed.release_id == "populace-us-2024-k20000-fixture"
    assert parsed.no_staging is True
    assert parsed.ssi_take_up_prior_weight_basis == tmp_path / "ssi.json"
    assert parsed.ssi_take_up_prior_weight_basis_sha256 == "1" * 64


def test_launcher_delegates_to_house_builder_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=_sha256(manifest)),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (8, {}),
    )
    captured: list[str] = []

    def fake_builder(argv):
        captured.extend(argv)
        _write_packaged_release(
            tmp_path / "out",
            release_id="populace-us-2024-k8-fixture",
            household_count=8,
        )

    result = launcher.launch(
        pool_manifest=manifest,
        config_path=config_path,
        out=tmp_path / "out",
        release_builder=fake_builder,
    )

    assert captured[captured.index("--exact-k") + 1] == "N"
    assert captured[captured.index("--seed") + 1] == "17"
    assert "--no-staging" in captured
    assert captured[captured.index("--pool-manifest-sha256") + 1] == _sha256(manifest)
    assert result["automatic_publish"] is False
    assert result["release_dir"] == str(
        tmp_path / "out" / "releases" / "populace-us-2024-k8-fixture"
    )
    assert result["artifact_root"] == str(tmp_path / "out" / "artifacts")
    assert result["pointer_update"] is False
    assert result["pointer_updates"]["production"]["pointer_update"] is False
    assert result["pointer_updates"]["staging"]["pointer_update"] is False
    assert result["publish_argv"][-3:] == [
        "--create-tag",
        "--no-latest",
        "--tag-only",
    ]
    assert "--artifact-root" in result["publish_argv"]
    assert "--repo-id policyengine/populace-us" in result["publish_command"]
    assert result["family_id"] == FAMILY_ID
    assert result["requested_k"] == 8
    assert result["realized_k"] == 8
    assert result["record_unit"] == "household"
    assert result["rung"] is None
    assert json.loads((tmp_path / "out" / "package_result.json").read_text()) == (
        result
    )

    rows = load_spool_rows(tmp_path / "out" / "logbook-spool")
    assert len(rows) == 1
    assert rows[0].rung is None
    assert rows[0].pipeline == "us-exact-k-release"
    assert rows[0].requested_k == 8
    assert rows[0].realized_k == 8
    assert rows[0].record_unit == "household"
    assert rows[0].disposition == "iterating"
    family_records = load_family_spool(tmp_path / "out" / "logbook-spool")
    assert family_records.families == (
        LogbookFamily.create(
            family_id=FAMILY_ID,
            chain_scope="us",
            source_pool_sha256=_sha256(manifest),
        ),
    )
    assert [member.build_id for member in family_records.family_members] == [
        "populace-us-2024-k8-fixture"
    ]


def test_reduced_build_records_numeric_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(
            pool_manifest_sha256=_sha256(manifest),
            requested_k=20_000,
            release_id="populace-us-2024-k20000-fixture",
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (20_000, {}),
    )

    result = launcher.launch(
        pool_manifest=manifest,
        config_path=config_path,
        out=tmp_path / "out",
        release_builder=lambda _argv: _write_packaged_release(
            tmp_path / "out",
            release_id="populace-us-2024-k20000-fixture",
            household_count=20_000,
        ),
    )

    row = load_spool_rows(tmp_path / "out" / "logbook-spool")[0]
    assert (result["requested_k"], result["realized_k"]) == (20_000, 20_000)
    assert (row.requested_k, row.realized_k, row.rung) == (20_000, 20_000, None)


def test_failure_before_n_resolution_records_null_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=_sha256(manifest)),
    )

    def fail_validation(**_kwargs):
        raise ValueError("fixture manifest failure")

    monkeypatch.setattr(launcher, "_validate_pins_and_resolve_k", fail_validation)

    with pytest.raises(ValueError, match="fixture manifest failure"):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=tmp_path / "out",
            release_builder=lambda _argv: pytest.fail("builder must not run"),
        )

    row = load_spool_rows(tmp_path / "out" / "logbook-spool")[0]
    assert (row.rung, row.requested_k, row.realized_k, row.record_unit) == (
        None,
        None,
        None,
        None,
    )
    assert row.disposition == "failed"
    family_records = load_family_spool(tmp_path / "out" / "logbook-spool")
    assert family_records.families == ()
    assert family_records.family_members == ()


def test_failure_after_numeric_resolution_retains_request_without_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(
            pool_manifest_sha256=_sha256(manifest),
            requested_k=20_000,
            release_id="populace-us-2024-k20000-fixture",
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (20_000, {}),
    )

    def fail_builder(_argv):
        raise RuntimeError("fixture build failure")

    with pytest.raises(RuntimeError, match="fixture build failure"):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=tmp_path / "out",
            release_builder=fail_builder,
        )

    row = load_spool_rows(tmp_path / "out" / "logbook-spool")[0]
    assert (row.rung, row.requested_k, row.realized_k, row.record_unit) == (
        None,
        20_000,
        None,
        "household",
    )
    family_records = load_family_spool(tmp_path / "out" / "logbook-spool")
    assert len(family_records.families) == 1
    assert family_records.family_members == ()
    error_receipt = json.loads(
        (
            tmp_path / "out/logbook-receipts/populace-us-2024-k20000-fixture/error.json"
        ).read_text(encoding="utf-8")
    )
    assert error_receipt["gate_verdicts"]["exact_k_build"]["verdict"] == "error"
    assert error_receipt["phases_reached"][-1] == "error"


def test_packaged_household_count_must_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=_sha256(manifest)),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (8, {}),
    )

    with pytest.raises(
        launcher.ExactKRealizedCountMismatchError,
        match="requested=8, realized=7",
    ):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=tmp_path / "out",
            release_builder=lambda _argv: _write_packaged_release(
                tmp_path / "out",
                release_id="populace-us-2024-k8-fixture",
                household_count=7,
            ),
        )

    row = load_spool_rows(tmp_path / "out" / "logbook-spool")[0]
    assert row.disposition == "failed"
    assert (row.requested_k, row.realized_k) == (8, None)
    assert "packaged_cardinality_verified" not in row.phases_reached


def test_release_id_cannot_overwrite_prior_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=_sha256(manifest)),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (8, {}),
    )

    def fail_builder(_argv):
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=tmp_path / "out",
            release_builder=fail_builder,
        )
    receipt_path = (
        tmp_path / "out/logbook-receipts/populace-us-2024-k8-fixture/error.json"
    )
    original_receipt = receipt_path.read_bytes()
    second_builder_called = False

    def second_builder(_argv):
        nonlocal second_builder_called
        second_builder_called = True

    with pytest.raises(FileExistsError, match="release ids are single-use"):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=tmp_path / "out",
            release_builder=second_builder,
        )

    assert second_builder_called is False
    assert receipt_path.read_bytes() == original_receipt
    assert len(load_spool_rows(tmp_path / "out" / "logbook-spool")) == 1


def test_matching_family_retry_is_accepted_and_mismatched_source_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    source_sha256 = _sha256(manifest)
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=source_sha256),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (8, {}),
    )
    matching_out = tmp_path / "matching"
    record_family(
        LogbookFamily.create(
            family_id=FAMILY_ID,
            chain_scope="us",
            source_pool_sha256=source_sha256,
        ),
        spool_dir=matching_out / "logbook-spool",
        post_remote=False,
    )

    launcher.launch(
        pool_manifest=manifest,
        config_path=config_path,
        out=matching_out,
        release_builder=lambda _argv: _write_packaged_release(
            matching_out,
            release_id="populace-us-2024-k8-fixture",
            household_count=8,
        ),
    )
    assert len(load_family_spool(matching_out / "logbook-spool").families) == 1

    mismatched_out = tmp_path / "mismatched"
    record_family(
        LogbookFamily.create(
            family_id=FAMILY_ID,
            chain_scope="us",
            source_pool_sha256="f" * 64,
        ),
        spool_dir=mismatched_out / "logbook-spool",
        post_remote=False,
    )
    builder_called = False

    def unexpected_builder(_argv):
        nonlocal builder_called
        builder_called = True

    with pytest.raises(ValueError, match="divergent retry"):
        launcher.launch(
            pool_manifest=manifest,
            config_path=config_path,
            out=mismatched_out,
            release_builder=unexpected_builder,
        )

    assert builder_called is False
    mismatched_records = load_family_spool(mismatched_out / "logbook-spool")
    assert mismatched_records.families[0].source_pool_sha256 == "f" * 64
    assert mismatched_records.family_members == ()


def test_exact_k_spools_all_records_before_dependency_ordered_remote_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _config_payload(pool_manifest_sha256=_sha256(manifest)),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_pins_and_resolve_k",
        lambda **_: (8, {}),
    )
    monkeypatch.setenv("POPULACE_LEDGER_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("POPULACE_LEDGER_KEY", "writer-jwt")
    spool = tmp_path / "out" / "logbook-spool"
    posted_paths: list[str] = []

    class Response:
        status = 201

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        assert timeout == 10.0
        if not posted_paths:
            assert len(list(spool.glob("*.json"))) == 1
            assert len(list((spool / "families").glob("*.json"))) == 1
            assert len(list((spool / "family_members").glob("*.json"))) == 1
        posted_paths.append(urlparse(request.full_url).path)
        return Response()

    monkeypatch.setattr(logbook_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(family_module, "urlopen", fake_urlopen)

    launcher.launch(
        pool_manifest=manifest,
        config_path=config_path,
        out=tmp_path / "out",
        release_builder=lambda _argv: _write_packaged_release(
            tmp_path / "out",
            release_id="populace-us-2024-k8-fixture",
            household_count=8,
        ),
    )

    assert posted_paths == [
        "/rest/v1/builds",
        "/rest/v1/families",
        "/rest/v1/family_members",
    ]
    assert list(spool.rglob("*.json")) == []


def test_pool_release_id_must_match_authenticated_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    (tmp_path / "ledger").mkdir()
    incumbent = tmp_path / "incumbent.json"
    incumbent.write_text(
        json.dumps({"target_surface": {"sha256": "e" * 64}}),
        encoding="utf-8",
    )
    payload = _config_payload(pool_manifest_sha256=_sha256(manifest))
    payload["pool"]["release_id"] = "invented-pool-release"
    payload["targets"]["incumbent_diagnostics_sha256"] = _sha256(incumbent)
    config = launcher._read_config(_write_config(tmp_path, payload))
    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        lambda _, **_kwargs: {
            "publication_run_id": "fixture-publication",
            "agreement_gate": {"passed": True},
            "provenance_counts": {"household": {"rows": 8}},
        },
    )

    with pytest.raises(
        ValueError,
        match="PoolReleaseIdentityMismatchError: configured pool release id",
    ):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )


def test_staging_credentials_cannot_enable_pointer_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    payload = _config_payload(
        requested_k=20_000,
        release_id="populace-us-2024-k20000-fixture",
    )
    config = launcher._read_config(_write_config(tmp_path, payload))
    monkeypatch.setenv("HF_TOKEN", "credentialed-fixture")
    monkeypatch.setenv("POPULACE_STAGING_REPO_ID", "fixture/staging")
    constructed = False

    class UnexpectedTelemetry:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(
        launcher.fiscal_release,
        "StagingTelemetry",
        UnexpectedTelemetry,
    )
    argv = launcher._builder_argv(
        config=config,
        pool_manifest=tmp_path / "pool.manifest.json",
        out=tmp_path / "out",
        k=20_000,
    )
    parsed = launcher.fiscal_release._parse_args(argv)

    telemetry = launcher.fiscal_release._staging_telemetry(
        parsed,
        release_root=tmp_path / "out",
        release_id=config.release_id,
    )

    assert parsed.no_staging is True
    assert telemetry is None
    assert constructed is False
    assert not (tmp_path / "out" / "staging").exists()
