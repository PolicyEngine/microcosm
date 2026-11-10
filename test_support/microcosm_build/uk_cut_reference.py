"""A stand-in for the UK repository's build folders, for cut-reference tests.

``UKReferenceHub`` answers ``hf_hub_download`` calls from memory for
``policyengine/populace-uk-private``: its ``releases.json`` and the build
folders of the national and dense lines, each with a release manifest, a
dataset and the evidence a cut's fetch reads (the national certification or
the dense gate summary). A missing file raises the Hub's not-found error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from test_support.microcosm_data.build_folders import build_folder_release_manifest
from test_support.microcosm_data.hub_errors import missing_on_hub

UK_REPO_ID = "policyengine/populace-uk-private"
UK_NATIONAL_RELEASE = "microcosm-uk-2024-25-national"
UK_DENSE_RELEASE = "microcosm-uk-2024-25-dense"
UK_REGISTERED_KEY_SHA256 = (
    "66bf0e29dbdabc2e3d7a79f21f43ac28d17b592faaf91e67c9bcf9b30c09a6ff"
)

_LINES = {
    UK_NATIONAL_RELEASE: ("national", "microcosm_uk_2024_25"),
    UK_DENSE_RELEASE: ("local_area", "microcosm_uk_2024_25_dense"),
}


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def national_certification(dataset: bytes, **overrides: Any) -> dict[str, Any]:
    """A certification with the fields a cut's fetch reads, shippable."""

    certification: dict[str, Any] = {
        "kind": "uk_release_certification",
        "release_id": UK_NATIONAL_RELEASE,
        "shippable": True,
        "candidate": {"sha256": sha256_of(dataset)},
        "attestation": {"signing_key_sha256": UK_REGISTERED_KEY_SHA256},
    }
    certification.update(overrides)
    return certification


def dense_gate_summary(**gate_overrides: dict[str, Any]) -> dict[str, Any]:
    """A gate summary whose gates all passed, unless overridden."""

    gates: dict[str, Any] = {
        "uk_local_target_fit": {"failures": [], "passed": True, "status": "passed"},
        "uk_local_weight_ratio": {"failures": [], "passed": True, "status": "passed"},
    }
    gates.update(gate_overrides)
    return {
        "schema_version": 1,
        "gates": gates,
        "diagnostic_gates": {},
        "signed_report": "uk_local_gates.json",
        "source": "microcosm_uk_2024_25_dense",
    }


class UKReferenceHub:
    """In-memory ``hf_hub_download`` for the UK repository's build folders."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []
        self._files: dict[str, Path] = {}
        self._releases: dict[str, dict[str, Any]] = {
            release: {"role": role, "current": None, "builds": {}}
            for release, (role, _) in _LINES.items()
        }
        self._write_record()

    def add_file(self, filename: str, content: bytes) -> None:
        local = self.root / "hub" / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        self._files[filename] = local

    def _write_record(self) -> None:
        record = {
            "schema_version": 1,
            "default": UK_NATIONAL_RELEASE,
            "releases": self._releases,
        }
        self.add_file("releases.json", json.dumps(record).encode())

    def publish(
        self,
        release: str,
        build: str,
        *,
        dataset: bytes,
        evidence: dict[str, Any] | None = None,
        current: bool = True,
        model_version: str = "2.174.1",
    ) -> None:
        """Publish ``build`` of ``release`` into its folder; make it current."""

        _, dataset_key = _LINES[release]
        folder = f"releases/{release}/{build}"
        evidence_key = (
            "release_certification"
            if release == UK_NATIONAL_RELEASE
            else "gate_summary"
        )
        if evidence is None:
            evidence = (
                national_certification(dataset)
                if release == UK_NATIONAL_RELEASE
                else dense_gate_summary()
            )
        evidence_bytes = json.dumps(evidence).encode()
        is_national = release == UK_NATIONAL_RELEASE
        tagged = {
            "schema_version": 1,
            "data_package": {"name": "microcosm-data", "version": "0.1.0"},
            "dataset_role": (
                "national_default" if is_national else "non_default_local_area"
            ),
            "default_datasets": {"national": dataset_key} if is_national else {},
            "build": {
                "built_with_core_package": {
                    "name": "policyengine-core",
                    "version": "3.33.3",
                },
                "built_with_model_package": {
                    "name": "policyengine-uk",
                    "version": model_version,
                },
            },
            "compatible_core_packages": [
                {"name": "policyengine-core", "specifier": "==3.33.3"}
            ],
            "compatible_model_packages": [
                {"name": "policyengine-uk", "specifier": f"=={model_version}"}
            ],
            "artifacts": {
                dataset_key: {
                    "kind": "microdata",
                    "path": f"{dataset_key}.h5",
                    "repo_id": UK_REPO_ID,
                    "sha256": sha256_of(dataset),
                },
                evidence_key: {
                    "kind": "diagnostics",
                    "path": f"{evidence_key}.json",
                    "repo_id": UK_REPO_ID,
                    "sha256": sha256_of(evidence_bytes),
                },
            },
        }
        manifest = build_folder_release_manifest(tagged, release=release, build=build)
        self.add_file(f"{folder}/release_manifest.json", json.dumps(manifest).encode())
        self.add_file(f"{folder}/{dataset_key}.h5", dataset)
        self.add_file(f"{folder}/{evidence_key}.json", evidence_bytes)
        entry = self._releases[release]
        entry["builds"][build] = {
            "published_at": "2026-11-07T03:26:09+00:00",
            "withdrawn": False,
        }
        if current:
            entry["current"] = build
        self._write_record()

    def withdraw(self, release: str, build: str, *, current: str | None = None) -> None:
        """Mark ``build`` withdrawn; the release's current moves to ``current``."""

        entry = self._releases[release]
        entry["builds"][build]["withdrawn"] = True
        if entry["current"] == build or current is not None:
            entry["current"] = current
        self._write_record()

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str | None = None,
    ) -> str:
        assert repo_type == "dataset"
        assert repo_id == UK_REPO_ID
        self.calls.append(filename)
        try:
            return str(self._files[filename])
        except KeyError as exc:
            raise missing_on_hub(repo_id, filename, revision) from exc


def stub_reference_evidence(reference, h5_path, output_dir):
    """Write a reference's evidence without reading its dataset as a frame.

    The stand-in datasets here are a few bytes, not H5 files; the input-mass
    file carries the identity the done-checks read and one synthetic column.
    """

    from microcosm.build.uk_runtime.cut_reference import (
        UK_CUT_REFERENCE_INPUT_MASS_NAME,
        UK_CUT_REFERENCE_SURFACE_NAME,
    )

    destination = Path(output_dir) / UK_CUT_REFERENCE_INPUT_MASS_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "uk_input_mass_reference",
                "label": reference.label,
                "identity": {
                    "filename": reference.filename,
                    "revision": reference.build,
                    "sha256": reference.sha256,
                    "vintage": "2024_25",
                },
                "totals": {"employment_income": 1.0},
                "carriers": {"employment_income": 3},
                "since_baseline": {
                    "baseline": reference.label,
                    "ratios": {"employment_income": 1.0},
                },
            }
        )
    )
    surface = Path(output_dir) / UK_CUT_REFERENCE_SURFACE_NAME
    surface.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "uk_reference_surface",
                "label": reference.label,
                "sha256": reference.sha256,
                "minimum_nondefault_mass_share": 1e-6,
                "baseline": {"label": reference.label, "inputs": ["employment_income"]},
                "export_columns": ["person.employment_income"],
                "populated_inputs": ["employment_income"],
                "thin_inputs": [],
            }
        )
    )
    return {"input_mass_reference": destination, "reference_surface": surface}


def reference_surface(
    *carried: str,
    thin: tuple[str, ...] = (),
    export_columns: tuple[str, ...] = ("person.employment_income",),
    label: str = "microcosm-uk-2024-25-national@uk-frs-calibration-attempt-test",
    sha256: str = "0" * 64,
    floor: float = 1e-6,
):
    """A reference surface carrying ``carried`` inputs (and ``thin`` ones)."""

    from microcosm.build.uk_runtime.reference_surface import UKReferenceSurface

    return UKReferenceSurface(
        label=label,
        sha256=sha256,
        export_columns=frozenset(export_columns),
        populated_inputs=frozenset(carried),
        thin_inputs=frozenset(thin),
        minimum_nondefault_mass_share=floor,
    )
