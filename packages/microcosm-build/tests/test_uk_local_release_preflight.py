"""The release-candidate pre-flight fails closed on every missing key or flag."""

from __future__ import annotations

import base64
import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "preflight_uk_local_release_candidate",
        REPO_ROOT / "tools" / "preflight_uk_local_release_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _good_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    (candidate / "logbook-spool").mkdir(parents=True)
    (candidate / "logbook-spool" / "row.json").write_text("{}")
    h5 = candidate / "microcosm_uk_2025_local.h5"
    h5.write_bytes(b"h5")
    import hashlib

    manifest = {
        "parameters": {
            "release_candidate": True,
            "epochs": 1500,
            "n_clones": 15,
            "skip_holdout": False,
            "doctrine": {
                "max_weight_ratio": 10.0,
                "target_weight_rule": "grain_equal",
                "solve_epochs": 1500,
                "clone_count": 15,
            },
        },
        "releasable": True,
        "blocked_at_f100": False,
        "solve": {
            "measure_resolution": {"blocks": 1},
            "target_weight_rule_override": {},
        },
        "fit": {"rotated_holdout": {"n_folds": 5, "mean_holdout_loss": 0.2}},
        "ladder_household_uprating": {
            "applied": True,
            "tenure_cells": {"applied": True},
        },
        "measure_exclusions": {"obr.housing_benefit": {"expires_on": "2026-10-03"}},
        "identity": {"spine": {"pin_verified": True}, "ladder": {"pin_verified": True}},
        "outputs": {
            "dataset": {"path": str(h5), "sha256": hashlib.sha256(b"h5").hexdigest()}
        },
    }
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    report = {
        "release_candidate": True,
        "shippable": True,
        "blocked_at_phase": None,
        "attestation": {"signature": "abc"},
        "gates": {
            "uk_local_area_support": {
                "criticality": "release_blocking",
                "status": "passed",
            },
            "uk_local_target_fit": {"criticality": "diagnostic", "status": "failed"},
        },
    }
    (candidate / "microcosm_uk_2025_local.local_gates.json").write_text(
        json.dumps(report)
    )
    return candidate


def test_good_candidate_dir_passes(tmp_path: Path) -> None:
    module = _load()
    assert (
        module.check_candidate_dir(_good_candidate(tmp_path), today=date(2026, 9, 4))
        == []
    )


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda m, r: m["parameters"].__setitem__("release_candidate", False),
            "release_candidate",
        ),
        (lambda m, r: m.__setitem__("releasable", False), "releasable"),
        (
            lambda m, r: m["solve"]["measure_resolution"].__setitem__("blocks", 15),
            "blocks",
        ),
        (
            lambda m, r: m["parameters"]["doctrine"].__setitem__(
                "max_weight_ratio", 100.0
            ),
            "max_weight_ratio",
        ),
        (lambda m, r: m["parameters"].__setitem__("skip_holdout", True), "holdout"),
        (
            lambda m, r: m["ladder_household_uprating"]["tenure_cells"].__setitem__(
                "applied", False
            ),
            "A17",
        ),
        (lambda m, r: m.__setitem__("measure_exclusions", {}), "measure_exclusions"),
        (
            lambda m, r: m["measure_exclusions"]["obr.housing_benefit"].__setitem__(
                "expires_on", "2026-09-01"
            ),
            "expired",
        ),
        (lambda m, r: r.__setitem__("release_candidate", False), "dev posture"),
        (lambda m, r: r.__setitem__("shippable", False), "shippable"),
        (
            lambda m, r: r["gates"]["uk_local_area_support"].__setitem__(
                "status", "failed"
            ),
            "release-blocking gate",
        ),
        (lambda m, r: r.__setitem__("attestation", {}), "signature"),
    ],
)
def test_each_missing_flag_is_named(tmp_path: Path, mutate, needle: str) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest = json.loads((candidate / "rowwise_candidate_manifest.json").read_text())
    report = json.loads(
        (candidate / "microcosm_uk_2025_local.local_gates.json").read_text()
    )
    mutate(manifest, report)
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    (candidate / "microcosm_uk_2025_local.local_gates.json").write_text(
        json.dumps(report)
    )
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any(needle in failure for failure in failures), failures


def test_env_check_names_the_missing_key_and_pins(tmp_path: Path) -> None:
    module = _load()
    pins = tmp_path / "pins.txt"
    spine = tmp_path / "spine.h5"
    spine.write_bytes(b"s")
    ladder = tmp_path / "ladder.npz"
    ladder.write_bytes(b"l")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text("{}\n")
    (ledger / "manifest.json").write_text("{}")
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pins.write_text(
        f"spine={sha(spine)} ladder={sha(ladder)} facts={sha(ledger / 'consumer_facts.jsonl')} manifest={sha(ledger / 'manifest.json')}\n"
    )
    good_env = {module.SIGNING_KEY_ENV: base64.b64encode(b"\x07" * 32).decode()}
    assert (
        module.check_env(
            pins_path=pins,
            spine_h5=spine,
            ladder_npz=ladder,
            ledger_dir=ledger,
            environ=good_env,
        )
        == []
    )
    failures = module.check_env(
        pins_path=pins, spine_h5=spine, ladder_npz=ladder, ledger_dir=ledger, environ={}
    )
    assert any(module.SIGNING_KEY_ENV in f for f in failures)
    pins.write_text("spine=deadbeef ladder=x facts=y manifest=z\n")
    failures = module.check_env(
        pins_path=pins,
        spine_h5=spine,
        ladder_npz=ladder,
        ledger_dir=ledger,
        environ=good_env,
    )
    assert any("digest mismatch" in f for f in failures)
