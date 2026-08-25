#!/usr/bin/env python3
"""Generate the generation-1 US spec bundle from constants-era authority.

This is an F0 migration tool, not a runtime dependency.  Its retained purpose
is to make the one-way provenance of the initial YAML bundle reviewable and to
provide a deterministic ``--check`` gate while constants remain available.

The dated tombstone is single-authored in ``TOMBSTONE_DATE`` below.  At that
date either delete the extractor with the constants-era authority or explicitly
renew the date in a reviewed change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import yaml

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from microcosm.build.spec_engine import (  # noqa: E402
    ENGINE_ABI_LOCK_FILENAME,
    engine_abi_lock_bytes_from_domains,
    load_bundle,
    scoped_take_up_manifest_program_bindings,
)

TOMBSTONE_DATE = "2026-11-16"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
US_PACKAGE_ROOT = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "us"
)
DEFAULT_OUTPUT_DIR = US_PACKAGE_ROOT / "spec"

DOMAIN_KINDS = (
    "battery",
    "bundle",
    "calibration",
    "catalogs",
    "geography",
    "imputation",
    "publication",
    "selection",
    "sources",
    "spine",
    "take_up",
    "vintages",
)


def _domain_builders() -> dict[str, Callable[[], object]]:
    """Import constants-era extractors only after the ABI bootstrap is current."""

    from tools.us_bundle_generation.contracts import (
        build_battery,
        build_calibration,
        build_selection,
        build_take_up,
    )
    from tools.us_bundle_generation.core import (
        build_bundle,
        build_catalogs,
        build_geography,
        build_publication,
        build_sources,
        build_spine,
        build_vintages,
    )
    from tools.us_bundle_generation.imputation import build_imputation

    builders: dict[str, Callable[[], object]] = {
        "battery": build_battery,
        "bundle": build_bundle,
        "calibration": build_calibration,
        "catalogs": build_catalogs,
        "geography": build_geography,
        "imputation": build_imputation,
        "publication": build_publication,
        "selection": build_selection,
        "sources": build_sources,
        "spine": build_spine,
        "take_up": build_take_up,
        "vintages": build_vintages,
    }
    if tuple(sorted(builders)) != DOMAIN_KINDS:  # pragma: no cover - source invariant
        raise RuntimeError("US domain builder registry differs from DOMAIN_KINDS")
    return builders


LEGACY_COMPATIBILITY_PROJECTIONS = {
    "source_stages.json": ("sources.yaml", "stage_manifest+stages"),
    "support_spine.json": (
        "spine.yaml",
        "support_source_pool_metadata+support_source_pool",
    ),
    "take_up_contract.json": (
        "take_up.yaml",
        "programs+legacy_contract_metadata+fresh_engine_abi_assertion",
    ),
}

# The three compatibility files are generation-0 inputs to existing identity
# machinery.  F0 therefore attests their exact historical bytes instead of
# rewriting their insignificant JSON layout and accidentally changing a raw
# package fingerprint.  Their parsed objects must also equal the projections
# generated from the typed domains below.  The compiler adapter, not these
# frozen files, is the forward YAML -> legacy-payload path.
FROZEN_LEGACY_RESOURCE_SHA256 = {
    "source_stages.json": (
        "619a974abd80c34201a0ccf813968ebc09ed52e144ff1d39722a4d9be7bd7821"
    ),
    "support_spine.json": (
        "68f37dc6ae6e0cde7ebccb53f88dd4a800e63456f838fa214ff98d1db8d815be"
    ),
    "take_up_contract.json": (
        "a9e70fb3e14b0af6cac5cc7935ef554f62dc3dca0377bc1fb57c0e6fa583e813"
    ),
}

# These generation-0 resources remain declared compatibility/package data.
# The three projections above are byte-frozen and equality-attested during the
# F0 bridge; their existing direct consumers retire with the constants adapter.
LEGACY_RESOURCE_PATHS = (
    "ecps_parity_known_gaps.json",
    "ecps_parity_reference.json",
    "federal_eitc_by_state.json",
    "fiscal_target_references.json",
    "obbba_reforms.json",
    "puf_aggregate_record_disaggregation.json",
    "release_input_coverage_manifest.json",
    "soca_capital_gain_distribution_shares.json",
    "soi_baseline_levels.json",
    "soi_table_2_1_interest_components_ty2015.json",
    "source_stages.json",
    "state_program_levels.json",
    "state_program_reforms.json",
    "state_reforms.json",
    "state_spm_poverty_levels.json",
    "support_spine.json",
    "take_up_contract.json",
    "target_parity_feed_families.json",
    "target_parity_manifest.json",
    "tax_expenditure_reforms.json",
)


class _NoAliasSafeDumper(yaml.SafeDumper):
    """Emit plain YAML without aliases that obscure generated ownership."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def _plain(value: object) -> object:
    """Convert extractor values to finite, JSON-compatible builtins."""

    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(child) for child in value]
    if isinstance(value, set | frozenset):
        return [_plain(child) for child in sorted(value, key=str)]
    if isinstance(value, Path):
        return value.as_posix()
    item = getattr(value, "item", None)
    if callable(item):
        return _plain(item())
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"US bundle extractor returned unsupported {type(value).__name__}")


def build_documents() -> dict[str, dict[str, object]]:
    """Return every typed US domain in stable filename order."""

    raw_take_up = json.loads((US_PACKAGE_ROOT / "take_up_contract.json").read_bytes())
    raw_programs = raw_take_up.get("programs")
    if not isinstance(raw_programs, list):
        raise TypeError("generation-0 take-up contract must contain a program array")
    bootstrap_bindings: list[tuple[str, str, str]] = []
    for index, raw_program in enumerate(raw_programs):
        if not isinstance(raw_program, Mapping):
            raise TypeError(f"generation-0 take-up program {index} must be a mapping")
        values = (
            raw_program.get("variable"),
            raw_program.get("entity"),
            raw_program.get("populace_treatment"),
        )
        if not all(isinstance(value, str) and value for value in values):
            raise TypeError(
                f"generation-0 take-up program {index} has incomplete bindings"
            )
        bootstrap_bindings.append(values)

    documents: dict[str, dict[str, object]] = {}
    with scoped_take_up_manifest_program_bindings(tuple(bootstrap_bindings)):
        for kind, builder in sorted(_domain_builders().items()):
            value = _plain(builder())
            if not isinstance(value, dict):
                raise TypeError(f"{kind} builder must return a mapping")
            documents[f"{kind}.yaml"] = value
    return documents


def country_manifest() -> dict[str, object]:
    """Return the one typed resource inventory for the US package."""

    typed_rows = [
        {
            "path": f"spec/{kind}.yaml",
            "kind": kind,
            "schema_id": f"{kind}.schema.json",
        }
        for kind in DOMAIN_KINDS
    ]
    legacy_rows = [
        {"path": path, "kind": "legacy_json", "schema_id": "legacy_json"}
        for path in LEGACY_RESOURCE_PATHS
    ]
    return {"schema_version": 1, "country": "us", "resources": typed_rows + legacy_rows}


def render_yaml(filename: str, value: Mapping[str, object]) -> bytes:
    header = (
        "# GENERATED by tools/generate_us_bundle_from_constants.py; "
        f"migration tombstone {TOMBSTONE_DATE}.\n"
    )
    if filename == "imputation.yaml":
        header += (
            "# F-P: eligibility concepts absent. Required concepts and exact "
            "legacy predictor gaps are declared in waiver_records; no predictors "
            "were invented.\n"
        )
    body = yaml.dump(
        dict(value),
        Dumper=_NoAliasSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )
    return (header + body).encode("utf-8")


def legacy_compatibility_projections(
    documents: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Project legacy package resources exclusively from typed domain fields."""

    sources = documents["sources.yaml"]
    stage_manifest = sources.get("stage_manifest")
    stages = sources.get("stages")
    if not isinstance(stage_manifest, Mapping) or not isinstance(stages, list):
        raise TypeError("sources.yaml must declare stage_manifest and stages")

    spine = documents["spine.yaml"]
    support_metadata = spine.get("support_source_pool_metadata")
    support_source_pool = spine.get("support_source_pool")
    if not isinstance(support_metadata, Mapping) or not isinstance(
        support_source_pool, Mapping
    ):
        raise TypeError(
            "spine.yaml must declare support_source_pool_metadata and "
            "support_source_pool"
        )

    from tools.us_bundle_generation.contracts import (
        project_legacy_take_up_contract,
    )

    take_up = project_legacy_take_up_contract(
        documents["take_up.yaml"],
        sources_document=documents["sources.yaml"],
    )
    return {
        "source_stages.json": {**stage_manifest, "stages": stages},
        "support_spine.json": {
            **support_metadata,
            "support_spine": support_source_pool,
        },
        "take_up_contract.json": take_up,
    }


def assert_frozen_legacy_compatibility(
    documents: Mapping[str, Mapping[str, object]],
) -> None:
    """Attest generation-0 bytes and their generated typed projections."""

    projections = legacy_compatibility_projections(documents)
    for name, projection in projections.items():
        path = US_PACKAGE_ROOT / name
        raw = path.read_bytes()
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        expected_sha256 = FROZEN_LEGACY_RESOURCE_SHA256[name]
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"{name}: generation-0 bytes changed: "
                f"{observed_sha256} != {expected_sha256}"
            )
        parsed = json.loads(raw)
        if parsed != projection:
            raise RuntimeError(
                f"{name}: frozen generation-0 object differs from its typed "
                "bundle projection"
            )


def rendered_files() -> dict[Path, bytes]:
    documents = build_documents()
    assert_frozen_legacy_compatibility(documents)
    files = {
        DEFAULT_OUTPUT_DIR / name: render_yaml(name, value)
        for name, value in documents.items()
    }
    manifest = (
        json.dumps(
            country_manifest(), ensure_ascii=False, indent=2, sort_keys=False
        ).encode("utf-8")
        + b"\n"
    )
    files[US_PACKAGE_ROOT / "country_package.json"] = manifest
    files[US_PACKAGE_ROOT / ENGINE_ABI_LOCK_FILENAME] = (
        engine_abi_lock_bytes_from_domains(
            {
                filename.removesuffix(".yaml"): document
                for filename, document in documents.items()
            }
        )
    )
    return files


def write_generated_files(*, check: bool) -> tuple[Path, ...]:
    """Write, or byte-check, the generated package resources."""

    expected = rendered_files()
    changed = tuple(
        path
        for path, payload in expected.items()
        if not path.exists() or path.read_bytes() != payload
    )
    if check:
        if changed:
            names = ", ".join(
                path.relative_to(REPOSITORY_ROOT).as_posix() for path in changed
            )
            raise SystemExit(f"generated US bundle is stale: {names}")
        return ()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return changed


def write_engine_abi_lock_only(*, check: bool) -> bool:
    """Generate only the fresh lock, without rewriting authored YAML."""

    path = US_PACKAGE_ROOT / ENGINE_ABI_LOCK_FILENAME
    # This bootstrap mode must work while an older generated lock is no longer
    # valid against a newly tightened lock schema.  Derive both authorities
    # from the same in-memory generator pass used by full regeneration: the
    # checked-in YAML may itself be stale while this bootstrap command is
    # needed.  ``build_documents`` scopes the legacy runtime import against the
    # frozen generation-0 program bindings, so no stale typed bundle is read.
    documents = build_documents()
    payload = engine_abi_lock_bytes_from_domains(
        {
            "take_up": documents["take_up.yaml"],
            "sources": documents["sources.yaml"],
        }
    )
    changed = not path.exists() or path.read_bytes() != payload
    if check and changed:
        raise SystemExit(
            "generated US engine ABI lock is stale: "
            f"{path.relative_to(REPOSITORY_ROOT).as_posix()}"
        )
    if changed and not check:
        path.write_bytes(payload)
    return changed


def validate_generated_bundle() -> str:
    """Compile the generated package through the F0 loader."""

    return load_bundle(US_PACKAGE_ROOT).spec_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="refuse if generated bytes differ"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="only render/check bytes (used while changing schemas)",
    )
    parser.add_argument(
        "--engine-lock-only",
        action="store_true",
        help="generate/check only engine_abi.lock.json; never rewrite bundle YAML",
    )
    args = parser.parse_args(argv)
    if args.engine_lock_only:
        changed = write_engine_abi_lock_only(check=args.check)
        if changed and not args.check:
            print("updated generated engine ABI lock")
        return 0
    changed = write_generated_files(check=args.check)
    if not args.skip_validation:
        digest = validate_generated_bundle()
        print(f"US bundle spec_sha256={digest}")
    if changed and not args.check:
        print(f"updated {len(changed)} generated file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
