"""The non-default local-area release contract (microcosm#398)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from microcosm.data.contract import (
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    ReleaseContractError,
    release_dataset_role,
    validate_release_dir,
)
from microcosm.data.release import publish_release

RELEASE_ID = "populace-us-2024-buildo-acs-local-abc1234-20260723T000000Z"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_local_bundle(
    root: Path,
    *,
    release_id: str = RELEASE_ID,
    gate_passed: bool = True,
    default_datasets: dict | None = None,
    pin_revisions: bool = True,
    drop_target_field: str | None = None,
    dataset_role: str = NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
) -> Path:
    release_dir = root / "releases" / release_id
    release_dir.mkdir(parents=True)

    diagnostics = {
        "households": 1_600_000,
        "n_targets": 2,
        "final_loss": 0.058,
        "fraction_within_10pct": 0.879,
        "targets": [
            {
                "name": "pop_state_06",
                "target": 100.0,
                "compiled_target": 100.0,
                "initial_estimate": 90.0,
                "final_estimate": 99.0,
            },
            {
                "name": "usda_snap_state_06_benefits",
                "target": 50.0,
                "compiled_target": 50.0,
                "initial_estimate": 40.0,
                "final_estimate": 49.0,
            },
        ],
    }
    if drop_target_field:
        del diagnostics["targets"][1][drop_target_field]
    gate_summary = {
        "gates": {
            "us_puma_ladder_gate": {"passed": True, "failures": []},
            "calibration": {"passed": gate_passed, "final_loss": 0.058},
        },
        "reviewed_limitations": [],
    }
    coverage = {
        "schema_version": 1,
        "acs_sources": {"manifest": "pinned"},
        "geography_ladder": {"sha256": "f" * 64},
        "transfer_coverage": {"targets": 91},
        "donor_release": {"release_id": "populace-us-2024-buildo-sparse-x"},
    }
    build_manifest = {"build_id": release_id, "build_sha": "abc1234"}

    files = {
        "calibration_diagnostics.json": diagnostics,
        "gate_summary.json": gate_summary,
        "us_source_coverage.json": coverage,
        "build_manifest.json": build_manifest,
    }
    payloads = {
        name: json.dumps(content, indent=1).encode() for name, content in files.items()
    }
    for name, payload in payloads.items():
        (release_dir / name).write_bytes(payload)

    artifact_bytes = b"not-a-real-h5-but-hash-stable"
    (root / "populace_us_2024_acs_local.h5").write_bytes(artifact_bytes)

    revision = release_id if pin_revisions else "some-other-revision"
    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": default_datasets if default_datasets is not None else {},
        "dataset_role": dataset_role,
        "is_default": False,
        "namespace": "buildo_acs_local",
        "build": {"build_id": release_id},
        "artifacts": {
            "populace_us_2024_acs_local": {
                "kind": "microdata",
                "path": "populace_us_2024_acs_local.h5",
                "repo_id": "policyengine/populace-us",
                "revision": revision,
                "sha256": _sha256_bytes(artifact_bytes),
            },
            "gate_summary": {
                "kind": "diagnostics",
                "path": "gate_summary.json",
                "repo_id": "policyengine/populace-us",
                "revision": revision,
                "sha256": _sha256_bytes(payloads["gate_summary.json"]),
            },
        },
        "reviewed_limitations": [{"id": "cd_population_marginal_vintage_2020"}],
    }
    manifest_text = json.dumps(manifest, indent=1)
    (release_dir / "release_manifest.json").write_text(manifest_text)
    ledger = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    ledger["release_manifest.json"] = _sha256_bytes(manifest_text.encode())
    ledger["populace_us_2024_acs_local.h5"] = _sha256_bytes(artifact_bytes)
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )
    return release_dir


def test_valid_local_area_bundle_passes(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)

    assert release_dataset_role(release_dir) == NON_DEFAULT_LOCAL_AREA_DATASET_ROLE
    validate_release_dir(release_dir)


def test_local_area_dispatch_does_not_bypass_uk_tier_identity(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(
        tmp_path,
        release_id="populace-uk-2023-public-k10",
    )

    with pytest.raises(ReleaseContractError, match="unratified tier"):
        validate_release_dir(release_dir)


def test_local_area_bundle_skips_national_critical_targets(tmp_path: Path) -> None:
    # The two-target local surface would fail every national critical-target
    # requirement; the role-keyed contract must not apply them.
    release_dir = _write_local_bundle(tmp_path)

    validate_release_dir(release_dir)


def test_failing_gate_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, gate_passed=False)

    with pytest.raises(ReleaseContractError, match="did not pass"):
        validate_release_dir(release_dir)


def test_default_dataset_claim_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(
        tmp_path,
        default_datasets={"us_2024": "populace_us_2024_acs_local"},
    )

    with pytest.raises(ReleaseContractError, match="empty\\s+object"):
        validate_release_dir(release_dir)


def test_unpinned_artifact_revision_blocks_local_area_publish(
    tmp_path: Path,
) -> None:
    release_dir = _write_local_bundle(tmp_path, pin_revisions=False)

    with pytest.raises(ReleaseContractError, match="not pinned to the release id"):
        validate_release_dir(release_dir)


def test_incomplete_target_row_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, drop_target_field="final_estimate")

    with pytest.raises(ReleaseContractError, match="missing 'final_estimate'"):
        validate_release_dir(release_dir)


def test_unknown_dataset_role_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, dataset_role="research_preview")

    with pytest.raises(ReleaseContractError, match="unknown dataset_role"):
        validate_release_dir(release_dir)


def test_tampered_diagnostics_artifact_hash_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    (release_dir / "gate_summary.json").write_text(
        json.dumps(
            {
                "gates": {"calibration": {"passed": True}},
                "reviewed_limitations": [],
            }
        )
    )

    with pytest.raises(ReleaseContractError, match="observed"):
        validate_release_dir(release_dir)


def test_publish_refuses_pointer_update_for_local_area_role(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)

    with pytest.raises(ValueError, match="never update latest.json"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=object(),
            update_latest=True,
        )


def test_publish_without_pointer_passes_the_role_guard(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)

    # A dummy backend fails the atomic-publication capability check — which
    # sits AFTER the role guard and the artifact verification, proving
    # --no-latest publishes proceed to the Hub surface.
    with pytest.raises(TypeError, match="Hub backend|create_tag"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=object(),
            artifact_root=tmp_path,
            update_latest=False,
        )


def test_empty_checksum_ledger_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    (release_dir / "sha256sums.txt").write_text("")

    with pytest.raises(ReleaseContractError, match="does not cover required file"):
        validate_release_dir(release_dir)


def test_tampered_checksum_ledger_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    ledger = (release_dir / "sha256sums.txt").read_text().splitlines()
    flipped = []
    for line in ledger:
        digest, name = line.split(maxsplit=1)
        if name == "gate_summary.json":
            digest = ("0" if digest[0] != "0" else "1") + digest[1:]
        flipped.append(f"{digest}  {name}")
    (release_dir / "sha256sums.txt").write_text("\n".join(flipped) + "\n")

    with pytest.raises(ReleaseContractError, match="disagrees|does not match"):
        validate_release_dir(release_dir)


def test_unsafe_ledger_path_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    with (release_dir / "sha256sums.txt").open("a") as handle:
        handle.write("a" * 64 + "  ../escape.json\n")

    with pytest.raises(ReleaseContractError, match="unsafe path"):
        validate_release_dir(release_dir)


def test_explicit_null_dataset_role_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["dataset_role"] = None
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError, match="unknown dataset_role"):
        validate_release_dir(release_dir)
