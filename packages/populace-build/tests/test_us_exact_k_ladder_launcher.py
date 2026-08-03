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
            "release_id": "fixture-pool-release",
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
            "agreement_gate": {"passed": True},
            "provenance_counts": {"household": {"rows": 50_000}},
        },
    )

    with pytest.raises(ValueError, match="k=57240 exceeds the pool size 50000"):
        launcher._validate_pins_and_resolve_k(
            config=config,
            pool_manifest_path=manifest,
        )


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

    assert captured[captured.index("--exact-k") + 1] == "8"
    assert captured[captured.index("--seed") + 1] == "17"
    assert captured[captured.index("--pool-manifest-sha256") + 1] == _sha256(manifest)
    assert result["automatic_publish"] is False
    assert result["pointer_update"] is False
    assert result["publish_argv"][-2:] == ["--create-tag", "--no-latest"]
    assert "--artifact-root" in result["publish_argv"]
    assert "--repo-id policyengine/populace-us" in result["publish_command"]
    assert json.loads((tmp_path / "out" / "package_result.json").read_text()) == (
        result
    )
