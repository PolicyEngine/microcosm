"""The reported-receipt child of the national default as a source enrichment.

Every fixture is invented: the parent is the synthetic donor the qualification
tests write, its Census sidecars are synthetic, and the parent release evidence
is three small JSON files whose digests the fixture pins in place of the real
ones. The qualification itself is the real tool, so the receipt the release
carries is exactly what the tool records.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
import sys
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import public_assistance_type_source as pats
from microcosm.data import release as publisher
from microcosm.data import source_enrichment as contract
from microcosm.data.contract import ReleaseContractError, validate_release_dir
from microcosm.data.h5_enrichment import file_sha256
from test_support.paths import paths_for
from tools import build_us_acs_donor_receipt_qualification as qualification
from tools import build_us_receipt_enrichment_release as builder

_TEST_PATHS = paths_for("microcosm-build")

pytest.importorskip("tables")

_SYNTHETIC = import_module(
    "test_support.microcosm_build.us_acs_donor_receipt_qualification"
)
_ROOT = _TEST_PATHS.repository
_RELEASE_ID = "populace-us-2024-spm-receipts-test"
_WHEELS = (Path("country.whl"),)


def _stub_identity(inventory) -> dict:
    return {
        "repository": "https://github.com/PolicyEngine/microcosm",
        "git_commit": "c" * 40,
        "git_dirty": False,
        "runtime": {"python": "3.14.0"},
        "source_files_sha256": dict.fromkeys(inventory, "d" * 64),
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


@pytest.fixture
def lineage(tmp_path, monkeypatch):
    """A synthetic parent, its release evidence and its real qualification."""

    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    # The national default's root name, so the qualified child takes the
    # published name the release pins.
    parent = parent_dir / "populace_us_2024.h5"
    _SYNTHETIC._write_parent(parent)
    parent_sha = file_sha256(parent)
    archives, audits, sources = {}, {}, {}
    for year in _SYNTHETIC._INCOME_YEARS:
        path = tmp_path / f"synthetic_person_{year}.csv"
        _SYNTHETIC._write_sidecar(path, year)
        archives[year], audits[year] = _SYNTHETIC._measured_pins(path, year)
        sources[year] = path
    monkeypatch.setattr(pats, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", archives)
    monkeypatch.setattr(pats, "ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS", audits)
    monkeypatch.setattr(qualification, "ASEC_EDUCATION_ASSISTANCE_ARCHIVES", archives)
    monkeypatch.setattr(qualification, "DONOR_SHA256", "f" * 64)
    monkeypatch.setattr(qualification, "SPM_ROLE_PARENT_SHA256", parent_sha)
    monkeypatch.setattr(
        qualification,
        "_producer_identity",
        lambda: _stub_identity(contract.RECEIPT_QUALIFICATION_SOURCE_FILES),
    )
    qualified = tmp_path / "qualified"
    qualification.qualify_donor(
        parent_h5=parent, output_dir=qualified, source_paths=sources
    )

    evidence = tmp_path / "parent-release"
    evidence.mkdir()
    payloads = {
        "release_manifest.json": {
            "build": {"build_id": contract.RECEIPT_PARENT_BUILD_ID},
            "artifacts": {
                "populace_us_2024": {
                    "kind": "microdata",
                    "path": "populace_us_2024.h5",
                    "sha256": parent_sha,
                }
            },
        },
        "calibration_diagnostics.json": {"schema_version": 5, "targets": []},
        "us_source_coverage.json": {"schema_version": 1},
    }
    pins = {}
    for name, payload in payloads.items():
        _write_json(evidence / name, payload)
        pinned = (
            "parent_release_manifest.json" if name == "release_manifest.json" else name
        )
        pins[pinned] = file_sha256(evidence / name)
    for module in (contract, builder):
        monkeypatch.setattr(module, "RECEIPT_PARENT_FILES", pins)
        monkeypatch.setattr(module, "RECEIPT_PARENT_DATASET_SHA256", parent_sha)
    monkeypatch.setattr(
        builder,
        "_producer_identity",
        lambda: _stub_identity(contract.RECEIPT_RELEASE_PRODUCER_FILES),
    )
    return SimpleNamespace(
        parent=parent,
        parent_release=evidence,
        qualified=qualified,
        child=qualified / contract.RECEIPT_DATASET_FILENAME,
        output=tmp_path / "candidate",
        tmp=tmp_path,
    )


def _build(lineage, **overrides) -> dict:
    kwargs = {
        "parent_h5": lineage.parent,
        "parent_release_dir": lineage.parent_release,
        "qualification_dir": lineage.qualified,
        "output_dir": lineage.output,
        "release_id": _RELEASE_ID,
    }
    kwargs.update(overrides)
    return builder.build_candidate(**kwargs)


def _release(lineage) -> Path:
    return lineage.output / "releases" / _RELEASE_ID


def _artifacts(lineage) -> Path:
    return lineage.output / "artifacts"


def _validate(lineage, release=None, **kwargs) -> dict:
    return contract.validate_source_enrichment_candidate(
        release or _release(lineage),
        parent_h5=lineage.parent,
        artifact_root=_artifacts(lineage),
        **kwargs,
    )


def _rewrite(release: Path, name: str, edit) -> None:
    """Edit one JSON file and re-seal every hash that binds it."""

    path = release / name
    path.chmod(0o600)
    payload = json.loads(path.read_text())
    edit(payload)
    _write_json(path, payload)
    if name == contract.RECEIPT_QUALIFICATION_FILE:
        report_path = release / contract.SOURCE_ENRICHMENT_FILE
        report = json.loads(report_path.read_text())
        report["source"]["qualification_sha256"] = file_sha256(path)
        _write_json(report_path, report)
        _reseal(release, contract.SOURCE_ENRICHMENT_FILE)
    _reseal(release, name)


def _reseal(release: Path, name: str) -> None:
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["artifacts"].values():
        if entry["path"] == name:
            entry["sha256"] = file_sha256(release / name)
    _write_json(manifest_path, manifest)


def _legacy_staging_module():
    path = _ROOT / "tools" / "_legacy" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location("_receipt_legacy_staging", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The candidate and what the local chain reads from it
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Certification and publication through the existing gates
# --------------------------------------------------------------------------


def _certify(lineage, monkeypatch, calls: list) -> Path:
    receipt = {
        "status": "passed",
        "dataset_sha256": file_sha256(lineage.child),
        "packages": {
            "policyengine-us": {"version": "2.2.1"},
            "policyengine-core": {"version": "3.32.5"},
            "policyengine": {"version": "5.99.0"},
            "spm-calculator": {"version": "1.0.0"},
        },
    }

    def probe(path, *, require_wheels, compatibility_wheels, native_inputs):
        calls.append(native_inputs)
        assert Path(path).name == contract.RECEIPT_DATASET_FILENAME
        assert compatibility_wheels == _WHEELS
        return receipt

    identities = []

    def authenticate(code, *, inventory=None):
        identities.append((tuple(sorted(code["source_files_sha256"])), inventory))

    monkeypatch.setattr(contract, "run_native_loader_compatibility", probe)
    monkeypatch.setattr(contract, "_check_producer_source_identity", authenticate)
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0")
    certified = contract.certify_source_enrichment(
        _release(lineage),
        lineage.tmp / "certified" / _RELEASE_ID,
        parent_h5=lineage.parent,
        artifact_root=_artifacts(lineage),
        compatibility_wheels=_WHEELS,
    )
    assert {inventory for _, inventory in identities} == {
        contract.RECEIPT_RELEASE_PRODUCER_FILES,
        contract.RECEIPT_QUALIFICATION_SOURCE_FILES,
    }
    return certified


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Native loading of the receipts
# --------------------------------------------------------------------------


def _country(lineage) -> SimpleNamespace:
    with pd.HDFStore(lineage.child, "r") as store:
        return SimpleNamespace(
            **{
                entity: store[entity]
                for entity in (
                    "person",
                    "household",
                    "tax_unit",
                    "spm_unit",
                    "family",
                    "marital_unit",
                )
            }
        )


def _fake_engine(monkeypatch, honours) -> list:
    calls = []

    class Simulation:
        def __init__(self, dataset):
            calls.append(dataset)
            self.dataset = dataset

        def calculate(self, variable, year):
            entity = contract.RECEIPT_COLUMNS[variable]
            table = getattr(self.dataset, entity)
            if variable in honours:
                return table[variable].to_numpy()
            return np.zeros(len(table), dtype=bool)

    country = ModuleType("policyengine_us")
    country.Microsimulation = Simulation
    data = ModuleType("policyengine_us.data")
    data.USSingleYearDataset = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "policyengine_us", country)
    monkeypatch.setitem(sys.modules, "policyengine_us.data", data)
    return calls


# --------------------------------------------------------------------------
# Pins and parity
# --------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
