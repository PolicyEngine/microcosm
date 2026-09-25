# ruff: noqa: F401
from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files as resource_files
from pathlib import Path

import pytest

import microcosm.build.spec_engine.loader as loader_module
from microcosm.build.spec_engine import (
    KernelRegistry,
    ResourceKind,
    SpecResolutionError,
    SpecValidationError,
    bundle_lock_bytes,
    bundle_lock_payload,
    load_bundle,
    load_schema_registry,
)
from microcosm.build.spec_engine.errors import SpecParseError
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL, SeedProtocol

ZERO_SHA = "0" * 64
SELECTION_KERNEL_IDS = [
    "assert_exact_k_support",
    "exact_k_ladder_manifest_payload",
    "exact_k_pcg64_rng",
    "select_exact_k",
]


def _write_bundle(root: Path, files: dict[str, str], rows: list[dict]) -> Path:
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "country": root.name,
        "resources": rows,
    }
    (root / "country_package.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _row(path: str, kind: str) -> dict[str, str]:
    return {"path": path, "kind": kind, "schema_id": f"{kind}.schema.json"}


def _rich_minimal(root: Path, *, note: str = "first", store: str = "local:a") -> Path:
    files = {
        "bundle.yaml": (
            "country: xx\n"
            "dataset_run: {target_period: 2024}\n"
            "identity_generation: 1\n"
            "seed_protocol: legacy-v1\n"
            "status: documentation only\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: ty2024\n"
            "    kind: target_period_ref\n"
            "    authority_ref:\n"
            "      kind: dataset_run\n"
            "      pointer: /dataset_run/target_period\n"
            "    compatible_with: [vintage:release2024]\n"
            "  - id: release2024\n"
            "    kind: release_series_ref\n"
            "    authority_ref:\n"
            "      kind: publication_release\n"
            "      pointer: /release/line/value\n"
            "    compatible_with: [vintage:ty2024]\n"
        ),
        "catalogs.yaml": (
            "columns:\n"
            "  - key: person.age\n"
            "    contract:\n"
            "      entity: person\n"
            "      dtype: int64\n"
            "      unit: count\n"
            "      period: vintage:ty2024\n"
            "      nullable: false\n"
            "      domain: demographics\n"
            "      public_stability: public_stable\n"
            "    docs:\n"
            f"      description: {note}\n"
            "      citations: [official]\n"
        ),
        "publication.yaml": (
            "attempts:\n"
            "  model: append_only_events_then_terminal_seal\n"
            "  terminal_states: [landed, failed, expired]\n"
            "promotion:\n"
            "  latest_flip: human_gate\n"
            "  idempotency: required_key\n"
            "  recovery: [seal_ok_append_fail]\n"
            "release:\n"
            "  line:\n"
            "    value: microcosm-xx-2024\n"
            "    normative: true\n"
            "    note: documentation note\n"
            "  pattern: '{line}-stacked-f001-s0'\n"
            "  rung_fractions:\n"
            "    - {fraction: 0.01, token: f001, percent_basis_points: 100}\n"
            "    - {fraction: 0.04, token: f004, percent_basis_points: 400}\n"
            "    - {fraction: 0.10, token: f010, percent_basis_points: 1000}\n"
            "    - {fraction: 0.25, token: f025, percent_basis_points: 2500}\n"
            "    - {fraction: 1.0, token: f100, percent_basis_points: 10000}\n"
            "audit_chain:\n"
            "  kind: strict_linear\n"
            f"  store: {store}\n"
            "release_graph:\n"
            "  relations: [derived_from]\n"
        ),
        "selection.yaml": resource_files("microcosm.build.us")
        .joinpath("spec/selection.yaml")
        .read_text(encoding="utf-8"),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("vintages.yaml", "vintages"),
        _row("catalogs.yaml", "catalogs"),
        _row("publication.yaml", "publication"),
        _row("selection.yaml", "selection"),
    ]
    return _write_bundle(root, files, rows)


def _append_legacy_json(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")
    manifest_path = root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].append(
        {"path": name, "kind": "legacy_json", "schema_id": "legacy_json"}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _cross_ref_bundle(root: Path, *, source_ref: str = "survey") -> Path:
    files = {
        "bundle.yaml": (
            "country: xx\n"
            "dataset_run: {target_period: 2024}\n"
            "identity_generation: 1\n"
            "seed_protocol: legacy-v1\n"
        ),
        "sources.yaml": (
            "sources:\n"
            "  - id: survey\n"
            "    role: survey\n"
            f"    sha256: '{ZERO_SHA}'\n"
            "    loader: kernel:survey_loader\n"
            "    vintages: [vintage:survey2024]\n"
            "    vintage_authorities:\n"
            "      - id: survey2024\n"
            "        kind: survey_period\n"
            "        value: 2024\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: survey2024\n"
            "    kind: survey_period_ref\n"
            "    authority_ref:\n"
            "      kind: source_record\n"
            "      source: source:survey\n"
            "      authority: survey2024\n"
            "    compatible_with: [vintage:target2024]\n"
            "  - id: target2024\n"
            "    kind: target_period_ref\n"
            "    authority_ref:\n"
            "      kind: dataset_run\n"
            "      pointer: /dataset_run/target_period\n"
            "    compatible_with: [vintage:survey2024]\n"
        ),
        "spine.yaml": (
            "channels:\n"
            "  - id: survey_channel\n"
            f"    source: {source_ref}\n"
            "    observed_geography: state\n"
            "assembly:\n"
            "  mass_anchor_channel: survey_channel\n"
            "  shared_dtype_policy: canonical_string_storage\n"
            "support_roles: []\n"
        ),
        "geography.yaml": (
            "phase: legacy\n"
            "assignment:\n"
            "  anchor: puma\n"
            "  order: legacy_post_transfer\n"
            "  kernels:\n"
            "    assign: kernel:geo_assign\n"
            "    validate: kernel:geo_validate\n"
            "  draw: {}\n"
            "  identified_county_source: source:survey\n"
            "  derive: [state_fips]\n"
            "  assertions: [observed_preserved]\n"
            "  ladder_source: puma_ladder\n"
            "  seed: stream:geography_legacy\n"
        ),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("sources.yaml", "sources"),
        _row("vintages.yaml", "vintages"),
        _row("spine.yaml", "spine"),
        _row("geography.yaml", "geography"),
    ]
    return _write_bundle(root, files, rows)


__all__ = [name for name in globals() if not name.startswith("__")]
