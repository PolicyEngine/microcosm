# ruff: noqa: F401
from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from microcosm.build.spec_engine import (
    F0_KERNEL_REGISTRY,
    SpecValidationError,
    load_schema_registry,
)
from microcosm.build.spec_engine.canonical import (
    documentation_envelope,
    normalize_and_project,
    sha256_json,
    spec_envelope,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.resolver import (
    SpecResolutionError,
    resolve_cross_references,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

Mutation = Callable[[dict[str, Any]], None]
ROOT = _TEST_PATHS.repository
CORE_PATH = ROOT / "tools/us_bundle_generation/core.py"
CORE_SPEC = importlib.util.spec_from_file_location("f0_us_bundle_core", CORE_PATH)
assert CORE_SPEC is not None and CORE_SPEC.loader is not None
CORE_MODULE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE_MODULE)
build_catalogs = CORE_MODULE.build_catalogs
build_bundle = CORE_MODULE.build_bundle
build_geography = CORE_MODULE.build_geography
build_publication = CORE_MODULE.build_publication
build_sources = CORE_MODULE.build_sources
build_vintages = CORE_MODULE.build_vintages
ENGINE_LOCK = json.loads(
    (
        ROOT / "packages/microcosm-build/src/microcosm/build/us/engine_abi.lock.json"
    ).read_text(encoding="utf-8")
)


def _vintage_resources() -> dict[str, Any]:
    return {
        "bundle": build_bundle(),
        "sources": build_sources(),
        "vintages": build_vintages(),
        "publication": build_publication(),
        "imputation": {},
    }


def _resolve_vintages(
    resources: dict[str, Any], *, engine_lock: dict[str, Any] = ENGINE_LOCK
):
    return resolve_cross_references(
        resources,
        kernel_registry=F0_KERNEL_REGISTRY,
        generated_authorities={"engine_abi_lock": engine_lock},
    )


def _source_surface_hashes(
    sources: dict[str, Any],
) -> tuple[str, str, dict[str, object]]:
    registry = load_schema_registry()
    normalized = registry.validate_and_inject_defaults(sources, "sources.schema.json")
    _, frozen_projections = normalize_and_project(
        normalized,
        schema_id="sources.schema.json",
        registry=registry,
    )
    projections = thaw_json(frozen_projections)
    normative = {"spec/sources.yaml": projections["normative"]}
    documentation = {"spec/sources.yaml": projections["documentation"]}
    return (
        sha256_json(
            spec_envelope(country="us", schema_version=1, normative_files=normative)
        ),
        sha256_json(
            documentation_envelope(
                country="us",
                schema_version=1,
                documentation_files=documentation,
            )
        ),
        projections,
    )


def _walk_values(value: object, path: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            rows.append((child_path, child))
            rows.extend(_walk_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_walk_values(child, f"{path}/{index}"))
    return rows


__all__ = [name for name in globals() if not name.startswith("__")]
