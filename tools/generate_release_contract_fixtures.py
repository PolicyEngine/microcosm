"""Generate or verify the release-contract golden fixtures (version 1).

docs/release-contract.md defines the release contract, C1 to C5. These fixtures
freeze two of its parts in bytes, the way ``generate_staging_contract_fixtures.py``
freezes the staging documents that calibration-diagnostics byte-checks:

* C1, the release manifest: the UK national line's ``build_manifest.json`` and
  ``release_manifest.json`` as the national assembler writes them (the generator
  calls ``uk_national_release_manifests`` and ``_write_json`` from
  ``tools/assemble_uk_release_dir.py``), plus ``inventory.json``, the files and
  certification fields the release contract requires of that line, read from
  ``microcosm.data.contract``.
* C5, publication: the two pointer payloads the publisher writes (the
  repository-global ``latest.json`` and the national line's
  ``latest-national.json``), serialized as ``publish_release`` serializes them,
  and ``registry.json``, a snapshot of ``microcosm.data.registry.REGISTRY``.

Every digest, version and identifier is synthetic. A digest is the SHA-256 of
``release-contract-fixture:<name>``, so none can be mistaken for a real artifact.
The fixtures deliberately carry no gate-battery report or signed certification:
those bind the UK gate spec's live digests, which change with most UK pull
requests, so the certification's shape is frozen as ``inventory.json`` instead and
its signed validity stays covered by the assembler's tests.

Regenerate after a deliberate contract change, in the same pull request that
updates docs/release-contract.md::

    uv run python tools/generate_release_contract_fixtures.py

``--check`` regenerates into a temporary directory and fails unless every byte
matches the committed fixtures.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType

from microcosm.build.staging_v2 import validate_staging_delivery
from microcosm.data import contract
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from microcosm.data.release import (
    LATEST_POINTER_PATH,
    latest_pointer_payload,
    line_pointer_path,
    line_pointer_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "packages"
    / "microcosm-data"
    / "tests"
    / "fixtures"
    / "release_contract"
    / "v1"
)
ASSEMBLER = ROOT / "tools" / "assemble_uk_release_dir.py"

#: One instant for every timestamp the fixtures carry.
FIXTURE_TIME = "2026-01-01T00:00:00+00:00"
#: A calibration attempt id in the grammar the assembler derives cut tags from.
UK_ATTEMPT_ID = "uk-frs-calibration-attempt-20260101T000000Z-00000000"
#: A US national release id in today's ``populace-us-`` grammar, for the
#: repository-global pointer.
US_RELEASE_ID = "populace-us-2024-0000000-20260101"
UK_LINE = "national"
#: Synthetic runtime versions. Each is a resolved PEP 440 version, which is all the
#: contract asks of a runtime pin.
UK_RUNTIME = {
    "python": "3.13.0",
    "policyengine-core": "3.0.0",
    "policyengine-uk": "2.0.0",
    "microcosm-data": "0.1.0",
    "microcosm-build": "0.1.0",
    "microcosm-calibrate": "0.1.0",
}
#: A successful version 2 staging delivery, matching the staging fixtures'
#: ``delivery_success`` case apart from its run id.
UK_STAGING_DELIVERY = {
    "contract_version": 2,
    "enabled": True,
    "mode": "local_and_remote",
    "run_id": "uk-national-release-contract-fixture",
    "configured_repository": "policyengine/populace-uk-staging",
    "upload_attempts": 6,
    "upload_successes": 6,
    "read_back": "passed",
    "last_error_code": None,
    "opt_out_reason": None,
}


def fixture_digest(name: str) -> str:
    """The synthetic SHA-256 the fixtures use for ``name``."""

    return hashlib.sha256(f"release-contract-fixture:{name}".encode()).hexdigest()


def _load_assembler() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assemble_uk_release_dir", ASSEMBLER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _c1_uk_national(root: Path, assembler: ModuleType) -> str:
    """Write the UK national manifests; return the line's cut tag."""

    cut_tag = assembler._cut_tag(UK_ATTEMPT_ID, None)
    measured = {
        name: fixture_digest(name)
        for name in ("candidate", *(name for _, _, name in assembler._EVIDENCE))
    }
    build_manifest, release_manifest = assembler.uk_national_release_manifests(
        measured=measured,
        calibration_filename=f"{assembler._DATASET_KEY}_calibration.npz",
        calibration_sha=fixture_digest("calibration"),
        household_weight_year={"dataset_year": 2024, "basis": "dataset_year"},
        cut_tag=cut_tag,
        attempt_id=UK_ATTEMPT_ID,
        runtime=UK_RUNTIME,
        code_pin=hashlib.sha1(b"release-contract-fixture").hexdigest(),
        target_surface={"sha256": fixture_digest("target surface"), "n_targets": 2},
        target_registry={"version": "release-contract-fixture-v1", "n_specs": 2},
        spine_part={"statuses": {"uk_release_input_coverage": "passed"}},
        seam_part={"statuses": {"uk_input_mass_parity": "passed"}},
        release_cut_part={"statuses": {"uk_aggregate_admin": "passed"}},
        staging_delivery=validate_staging_delivery(UK_STAGING_DELIVERY),
        created_at=FIXTURE_TIME,
        publisher_labels={"ons": "Office for National Statistics"},
    )
    release_dir = root / "c1" / contract._UK_NATIONAL_RELEASE_ID
    assembler._write_json(release_dir / "build_manifest.json", build_manifest)
    assembler._write_json(release_dir / "release_manifest.json", release_manifest)

    inventory = {
        "schema_name": "microcosm.release-contract.c1-inventory",
        "schema_version": 1,
        "release_id": contract._UK_NATIONAL_RELEASE_ID,
        "line": contract.line_for_release_id(contract._UK_NATIONAL_RELEASE_ID),
        "dataset_role": contract.dataset_role_for_line(UK_LINE),
        "release_manifest_schema_version": contract.RELEASE_MANIFEST_SCHEMA_VERSION,
        "required_release_files": list(
            contract.required_release_files(contract._UK_NATIONAL_RELEASE_ID)
        ),
        "release_directory_files": sorted(
            [
                "build_manifest.json",
                "release_manifest.json",
                *(filename for _, filename, _ in assembler._EVIDENCE),
            ]
        ),
        "certification": {
            "file": contract._UK_RELEASE_CERTIFICATION_FILE,
            "schema_version": contract._UK_RELEASE_CERTIFICATION_SCHEMA_VERSION,
            "kind": contract._UK_RELEASE_CERTIFICATION_KIND,
            "required_fields": sorted(contract._UK_CERTIFICATION_REQUIRED_FIELDS),
            "part_evidence_files": dict(contract._UK_CERTIFICATION_PART_EVIDENCE_FILES),
            "score_receipt_file": contract._UK_CERTIFICATION_SCORE_RECEIPT_FILE,
            "gate_battery_phases": list(contract._UK_GATE_BATTERY_PHASES),
            "signing_key_env": contract._UK_GATE_BATTERY_SIGNING_KEY_ENV,
        },
    }
    (root / "c1" / "inventory.json").write_bytes(_json_bytes(inventory))
    return cut_tag


def _c5_publication(root: Path, *, uk_cut_tag: str) -> None:
    c5 = root / "c5"
    c5.mkdir(parents=True, exist_ok=True)
    # publish_release uploads ``json.dumps(payload, indent=1)`` with no newline.
    (c5 / LATEST_POINTER_PATH).write_bytes(
        json.dumps(
            latest_pointer_payload(US_RELEASE_ID, updated_at=FIXTURE_TIME), indent=1
        ).encode()
    )
    (c5 / line_pointer_path(UK_LINE)).write_bytes(
        json.dumps(
            line_pointer_payload(
                contract._UK_NATIONAL_RELEASE_ID,
                line=UK_LINE,
                revision=uk_cut_tag,
                updated_at=FIXTURE_TIME,
            ),
            indent=1,
        ).encode()
    )
    registry = {
        "schema_name": "microcosm.release-contract.registry",
        "schema_version": 1,
        "default_variant": DEFAULT_VARIANT,
        "entries": [
            {"key": list(key), **dataclasses.asdict(REGISTRY[key])}
            for key in sorted(REGISTRY)
        ],
    }
    (c5 / "registry.json").write_bytes(_json_bytes(registry))


def _generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cut_tag = _c1_uk_national(root, _load_assembler())
    _c5_publication(root, uk_cut_tag=cut_tag)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="microcosm-release-contract-") as temporary:
        generated = Path(temporary) / "v1"
        _generate(generated)
        if args.check:
            if _tree(generated) != _tree(args.output):
                raise SystemExit(
                    "release-contract fixtures are not reproducible: regenerate "
                    "with `uv run python tools/generate_release_contract_fixtures.py` "
                    "and update docs/release-contract.md in the same change"
                )
            return 0
        if args.output.exists():
            shutil.rmtree(args.output)
        shutil.copytree(generated, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
