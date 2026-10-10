"""Pinned national and local target inputs for the single UK full build.

Compilation and approved measure exclusions precede geography selection. The
unreduced national register remains available for band edges, and independent
reference-period compilations remain available for release validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.calibration_run import (
    _ledger_provenance,
    _validate_band_edge_registry,
)
from microcosm.build.uk_runtime.chronicle_feed import (
    load_uk_chronicle_feed,
    require_committed_uk_chronicle_feed_pin,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import (
    assert_uk_local_deferrals_in_force,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
    load_uk_local_area_crosswalk,
    load_uk_local_target_reference_membership,
)
from microcosm.build.uk_runtime.local_target_census import _LEDGER_FACT_FEED_PIN
from microcosm.build.uk_runtime.measure_simulation import (
    apply_uk_calibration_measure_exclusions,
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_reconciliation import (
    reconcile_uk_national_registry,
)
from microcosm.build.uk_runtime.uprating_holds import (
    assert_uk_uprating_holds_declared,
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


#: Historical target periods compiled beside the calibration year so the
#: terminal surface and the certifier can compare against them; skipped, not
#: fatal, when the pinned feed no longer carries their facts.
_VALIDATION_PERIODS: frozenset[int] = frozenset({2023, 2025})
_LOCAL_VALIDATION_PERIODS: frozenset[int] = frozenset({2025})


def _unsupported_names(unsupported: object) -> tuple[str, ...]:
    names = []
    for item in unsupported:
        name = item.get("name") if isinstance(item, Mapping) else None
        names.append(str(name if name is not None else item))
    return tuple(names)


def load_uk_full_target_inputs(
    facts_path: str | Path,
    *,
    expected_facts_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    measure_exclusions: str | Path | None = None,
    register_json: str | Path | None = None,
    calibration_year: int | None = None,
    exclusions_evaluated_on: date | None = None,
    include_validation_periods: bool = True,
) -> dict[str, Any]:
    """Compile the full target surface with both reviewed source contracts.

    ``include_validation_periods=False`` compiles the calibration year alone,
    for receipts that read the surface (``tools/uk_target_surface_receipt.py``);
    the graph node always compiles the validation periods too.

    Default hashes are the reviewed national feed pins. Explicit hashes must
    agree with those pins as well: a target-scope filter does not authorize a
    different source. The optional frozen register compares the complete,
    pre-exclusion national register, matching its completeness role.
    """
    pin = load_uk_chronicle_feed()
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
    # The calibration year is fail-closed. The historical validation periods
    # keep whatever the pinned feed still compiles, as the release-cut producer
    # does (tools/certify_uk_release_cut.py keeps ``compilation.registry`` for
    # every parity period): the references the feed no longer carries are
    # recorded below instead of blocking the build.
    national_registries = {}
    local_registries = {}
    unsupported_validation: dict[str, dict[int, tuple[str, ...]]] = {
        "national": {},
        "local": {},
    }
    national_periods = (
        sorted({*_VALIDATION_PERIODS, year}) if include_validation_periods else [year]
    )
    local_periods = (
        sorted({*_LOCAL_VALIDATION_PERIODS, year})
        if include_validation_periods
        else [year]
    )
    for period in national_periods:
        compilation = compile_uk_target_registry(artifact.facts, target_period=period)
        if compilation.unsupported:
            if period == year:
                raise ValueError(
                    f"UK national target references failed to compile for {period}: "
                    f"{compilation.unsupported!r}."
                )
            unsupported_validation["national"][period] = _unsupported_names(
                compilation.unsupported
            )
        national_registries[period] = compilation.registry
    for period in local_periods:
        compilation = compile_uk_local_target_registry(
            artifact.facts, target_period=period, crosswalk=crosswalk
        )
        if compilation.unsupported:
            if period == year:
                raise ValueError(
                    f"UK local target references failed to compile for {period}: "
                    f"{compilation.unsupported!r}."
                )
            unsupported_validation["local"][period] = _unsupported_names(
                compilation.unsupported
            )
        local_registries[period] = compilation.registry
    # microcosm#1123: reconcile the compiled national register across its own
    # grains before the exclusions, so a control's excluded members still
    # count, and the register the solve binds (and the frozen scoring register
    # it is checked against) carries the reconciled values. The validation
    # periods stay as compiled: compile parity measures the facts themselves.
    compiled_version = national_registries[year].version
    band_edges, national_reconciliation = reconcile_uk_national_registry(
        national_registries[year]
    )
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
    assert_uk_local_deferrals_in_force(
        load_uk_local_target_reference_membership(), evaluated_on
    )
    uprating_holds = {
        # The full compiled register, not the approved one: a measure
        # exclusion expires, and its target re-enters already declared.
        "national": assert_uk_uprating_holds_declared(
            band_edges,
            calibration_period=year,
            evaluated_on=evaluated_on,
            scope="national",
        ),
        "local": assert_uk_uprating_holds_declared(
            local_registries[year],
            calibration_period=year,
            evaluated_on=evaluated_on,
            scope="local",
        ),
    }
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
        "national_reconciliation": national_reconciliation,
        "uprating_holds": uprating_holds,
        "local_registry": local_registries[year],
        "measure_exclusions": exclusion_receipt,
        "reviewed_unbound_higher_targets": reviewed_unbound,
        "national_source_pin": pin.to_dict(),
        "local_source_pin": local_pin,
        "ledger_provenance": _ledger_provenance(artifact),
        "register_completeness": {
            "compiled_registry_version": compiled_version,
            "reconciled_registry_version": band_edges.version,
            "approved_registry_version": national_registry.version,
            "frozen_registry_version": frozen_version,
            "compiled_reference_count": len(band_edges.specs),
            "approved_reference_count": len(national_registry.specs),
            "measure_exclusion_count": len(exclusion_receipt),
            "exclusions_evaluated_on": evaluated_on.isoformat(),
            "band_edge_registry_reconciled": True,
            "compiled_local_reference_count": len(local_registries[year].specs),
            "local_registry_version": local_registries[year].version,
            "validation_periods": {
                "national": sorted(p for p in national_registries if p != year),
                "local": sorted(p for p in local_registries if p != year),
            },
            "validation_periods_unsupported": {
                scope: {str(p): list(names) for p, names in sorted(periods.items())}
                for scope, periods in unsupported_validation.items()
            },
        },
        "uk_ledger_compiled_registries": national_registries,
        "uk_ledger_compiled_local_registries": local_registries,
    }


def load_uk_national_target_inputs(
    facts_path: str | Path,
    *,
    measure_exclusions: str | Path | None = None,
    register_json: str | Path | None = None,
    calibration_year: int | None = None,
    exclusions_evaluated_on: date | None = None,
    allow_unpinned_feed: bool = False,
    expected_facts_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """The national role's target surface: the pinned Ledger artifact, compiled.

    The seam's loader (``national_role._load_national_target_inputs``) as the
    graph's national target node runs it: the artifact must be the committed
    Chronicle feed pin unless ``allow_unpinned_feed`` records a reviewed
    diagnostic run; the compiled register less the measure exclusions is the
    solve surface and the full compiled register keeps the band edges;
    ``register_json`` requires the re-derived scoring surface to be the frozen
    one. No local registry and no ladder: the national line has neither.
    """

    artifact = load_ledger_consumer_artifact(
        Path(facts_path),
        expected_facts_sha256=expected_facts_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    pin = require_committed_uk_chronicle_feed_pin(
        artifact.facts_sha256,
        manifest_sha256=artifact.manifest_sha256,
        allow_unpinned_feed=bool(allow_unpinned_feed),
    )
    year = (
        load_uk_frs_release().calibration_year
        if calibration_year is None
        else calibration_year
    )
    if type(year) is not int or year <= 0:
        raise ValueError("calibration_year must be a positive integer.")
    evaluated_on = exclusion_evaluation_date(exclusions_evaluated_on)
    compilation = compile_uk_target_registry(artifact.facts, target_period=year)
    if compilation.unsupported:
        raise ValueError(
            f"{len(compilation.unsupported)} national target references failed "
            f"to compile for {year}: {compilation.unsupported!r}."
        )
    exclusions = load_uk_calibration_measure_exclusions(
        None if measure_exclusions is None else Path(measure_exclusions)
    )
    reconciled, national_reconciliation = reconcile_uk_national_registry(
        compilation.registry
    )
    registry, exclusion_receipt = apply_uk_calibration_measure_exclusions(
        reconciled, exclusions, now=evaluated_on
    )
    _validate_band_edge_registry(
        register_registry=registry,
        band_edge_registry=reconciled,
        exclusion_receipt=exclusion_receipt,
    )
    uprating_holds = assert_uk_uprating_holds_declared(
        reconciled,
        calibration_period=year,
        evaluated_on=evaluated_on,
        scope="national",
    )
    frozen_version = None
    if register_json is not None:
        try:
            frozen = TargetRegistry.from_json(Path(register_json))
        except ValueError as error:
            raise ValueError(f"frozen scoring register is unusable: {error}") from error
        frozen_version = frozen.version
        if frozen.version != registry.version:
            raise ValueError(
                "re-derived register differs from the frozen scoring register: "
                f"{registry.version} vs {frozen.version}"
            )
    return {
        "artifact": artifact,
        "calibration_year": year,
        "national_registry": registry,
        "band_edge_registry": reconciled,
        "national_reconciliation": national_reconciliation,
        "uprating_holds": uprating_holds,
        "measure_exclusions": exclusion_receipt,
        "chronicle_feed_pin": pin.to_dict(),
        "chronicle_provenance": artifact.provenance(),
        "ledger_provenance": _ledger_provenance(artifact),
        "register_completeness": {
            "compiled_registry_version": compilation.registry.version,
            "reconciled_registry_version": reconciled.version,
            "approved_registry_version": registry.version,
            "frozen_registry_version": frozen_version,
            "compiled_reference_count": len(compilation.registry.specs),
            "approved_reference_count": len(registry.specs),
            "measure_exclusion_count": len(exclusion_receipt),
            "exclusions_evaluated_on": evaluated_on.isoformat(),
            "band_edge_registry_reconciled": True,
        },
    }
