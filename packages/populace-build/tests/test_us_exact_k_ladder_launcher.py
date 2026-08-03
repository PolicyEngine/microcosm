from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


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
        "schema_version": 1,
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
        lambda _: {
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


def test_pool_manifest_sha_pin_is_checked_before_manifest_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    manifest = tmp_path / "pool.manifest.json"
    manifest.write_text("fixture", encoding="utf-8")
    config = launcher._read_config(
        _write_config(tmp_path, _config_payload(pool_manifest_sha256="a" * 64))
    )
    loaded = False

    def unexpected_load(_):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        unexpected_load,
    )

    with pytest.raises(ValueError, match="Pool manifest SHA-256 mismatch"):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )
    assert not loaded


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
    monkeypatch.setattr(
        launcher,
        "load_simulation_ready_us_multispine_pool_manifest",
        lambda _: validated_manifest,
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
        lambda _: {
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
        return {
            "release_id": "populace-us-2024-k8-fixture",
            "release_dir": str(tmp_path / "out" / "releases" / "release"),
            "artifact_root": str(tmp_path / "out" / "artifacts"),
            "dataset_path": str(tmp_path / "out" / "artifacts" / "fixture.h5"),
            "calibration_path": str(tmp_path / "out" / "artifacts" / "fixture.npz"),
        }

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
    assert result["pointer_update"] is False
    assert result["pointer_updates"]["production"]["pointer_update"] is False
    assert result["pointer_updates"]["staging"]["pointer_update"] is False
    assert result["publish_argv"][-2:] == ["--create-tag", "--no-latest"]
    assert "--artifact-root" in result["publish_argv"]
    assert "--repo-id policyengine/populace-us" in result["publish_command"]
    assert json.loads((tmp_path / "out" / "package_result.json").read_text()) == (
        result
    )


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
        lambda _: {
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
