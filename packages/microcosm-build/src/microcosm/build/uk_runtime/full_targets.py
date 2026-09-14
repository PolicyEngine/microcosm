"""Pinned national and local target inputs for the single UK full build.

Compilation and approved measure exclusions precede geography selection. The
unreduced national register remains available for band edges, and independent
reference-period compilations remain available for release validation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.calibration_run import (
    _ledger_provenance,
    _validate_band_edge_registry,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import (
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    load_uk_local_area_crosswalk,
)
from microcosm.build.uk_runtime.local_target_census import _LEDGER_FACT_FEED_PIN
from microcosm.build.uk_runtime.measure_simulation import (
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_chronicle_feed import (
    load_uk_national_chronicle_feed,
)
from microcosm.build.uk_runtime.weighted_integrity import exclusion_evaluation_date
from microcosm.calibrate import TargetRegistry

CHRONICLE_SOURCE_CODEC = "chronicle-consumer-facts-v1"


def load_chronicle_source_bytes(path: Path, *, store=None) -> bytes:
    """Read facts from a manifest-verified consumer artifact or a bare feed.

    Graph source identity binds every file in an artifact directory, including
    the manifest. The target compiler independently enforces reviewed UK pins.
    """

    del store
    load_ledger_consumer_artifact(path)
    path = Path(path)
    return (path / "consumer_facts.jsonl" if path.is_dir() else path).read_bytes()


def load_uk_local_chronicle_pin() -> dict[str, Any]:
    """Return the independently reviewed local target census feed identity."""

    return dict(_LEDGER_FACT_FEED_PIN)


def load_uk_full_target_inputs(
    facts_path: str | Path,
    *,
    expected_facts_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    measure_exclusions: str | Path | None = None,
    register_json: str | Path | None = None,
    calibration_year: int | None = None,
    exclusions_evaluated_on: date | None = None,
) -> dict[str, Any]:
    """Compile the full target surface with both reviewed source contracts.

    Default hashes are the reviewed national feed pins. Explicit hashes must
    agree with those pins as well: a target-scope filter does not authorize a
    different source. The optional frozen register compares the complete,
    pre-exclusion national register, matching its completeness role.
    """
    pin = load_uk_national_chronicle_feed()
    local_pin = load_uk_local_chronicle_pin()
    for field in ("facts_sha256", "manifest_sha256", "fact_row_count"):
        if local_pin[field] != getattr(pin, field):
            raise ValueError(
                "UK full-build independently reviewed national and local feed "
                f"pins disagree on {field}."
            )
    for label, supplied, committed in (
        ("facts", expected_facts_sha256, pin.facts_sha256),
        ("manifest", expected_manifest_sha256, pin.manifest_sha256),
    ):
        if supplied is not None and supplied != committed:
            raise ValueError(
                f"UK full-build {label} SHA differs from the committed national feed pin."
            )
    artifact = load_ledger_consumer_artifact(
        Path(facts_path),
        expected_facts_sha256=pin.facts_sha256,
        expected_manifest_sha256=pin.manifest_sha256,
    )
    if (
        artifact.facts_sha256 != pin.facts_sha256
        or artifact.manifest_sha256 != pin.manifest_sha256
    ):
        raise ValueError(
            "UK full-build Ledger artifact differs from the national feed pin."
        )
    if len(artifact.facts) != pin.fact_row_count:
        raise ValueError(
            "UK full-build Chronicle fact row count differs from the reviewed "
            "national and local feed pins."
        )
    year = (
        load_uk_frs_release().calibration_year
        if calibration_year is None
        else calibration_year
    )
    if type(year) is not int or year <= 0:
        raise ValueError("calibration_year must be a positive integer.")
    evaluated_on = exclusion_evaluation_date(exclusions_evaluated_on)
    crosswalk = load_uk_local_area_crosswalk()
    national_registries = {}
    local_registries = {}
    for period in sorted({2023, 2025, year}):
        compilation = compile_uk_target_registry(artifact.facts, target_period=period)
        if compilation.unsupported:
            raise ValueError(
                f"UK national target references failed to compile for {period}: "
                f"{compilation.unsupported!r}."
            )
        national_registries[period] = compilation.registry
    for period in sorted({2025, year}):
        compilation = compile_uk_local_target_registry(
            artifact.facts, target_period=period, crosswalk=crosswalk
        )
        if compilation.unsupported:
            raise ValueError(
                f"UK local target references failed to compile for {period}: "
                f"{compilation.unsupported!r}."
            )
        local_registries[period] = compilation.registry
    band_edges = national_registries[year]
    frozen_version = None
    if register_json is not None:
        frozen = TargetRegistry.from_json(Path(register_json))
        frozen_version = frozen.version
        if frozen.version != band_edges.version:
            raise ValueError(
                "Re-derived full national register differs from the frozen scoring register: "
                f"{band_edges.version} vs {frozen.version}."
            )
    exclusions = load_uk_calibration_measure_exclusions(
        None if measure_exclusions is None else Path(measure_exclusions)
    )
    national_registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        band_edges, exclusions, now=evaluated_on
    )
    _validate_band_edge_registry(
        register_registry=national_registry,
        band_edge_registry=band_edges,
        exclusion_receipt=exclusion_receipt,
    )
    by_name = {spec.name: spec for spec in band_edges.specs}
    reviewed_unbound = {
        str(by_name[name].metadata.get("contract_target_id", name)): record
        for name, record in exclusion_receipt.items()
    }
    return {
        "artifact": artifact,
        "calibration_year": year,
        "national_registry": national_registry,
        "band_edge_registry": band_edges,
        "local_registry": local_registries[year],
        "measure_exclusions": exclusion_receipt,
        "reviewed_unbound_higher_targets": reviewed_unbound,
        "national_source_pin": pin.to_dict(),
        "local_source_pin": local_pin,
        "ledger_provenance": _ledger_provenance(artifact),
        "register_completeness": {
            "compiled_registry_version": band_edges.version,
            "approved_registry_version": national_registry.version,
            "frozen_registry_version": frozen_version,
            "compiled_reference_count": len(band_edges.specs),
            "approved_reference_count": len(national_registry.specs),
            "measure_exclusion_count": len(exclusion_receipt),
            "exclusions_evaluated_on": evaluated_on.isoformat(),
            "band_edge_registry_reconciled": True,
            "compiled_local_reference_count": len(local_registries[year].specs),
            "local_registry_version": local_registries[year].version,
        },
        "uk_ledger_compiled_registries": national_registries,
        "uk_ledger_compiled_local_registries": local_registries,
    }
