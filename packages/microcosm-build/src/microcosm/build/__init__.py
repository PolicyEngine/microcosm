"""microcosm.build: stage plans and acceptance gates for dataset builds.

A build is a typed :class:`~microcosm.build.plan.StagePlan` over a
:class:`~microcosm.frame.Frame` — each stage declares what it consumes, what it
produces, and (for imputations) which donor survey it draws from; the executor
materializes stages in order and fails loudly on any donor it cannot load (no
silent fallbacks, per the charter). The product of a build is only publishable
once it passes the :mod:`microcosm.build.gates` suite: parity, support,
aggregate-vs-admin, per-family fit — plus the :mod:`microcosm.build.holdout`
rotation for scoring.

Importing this shard asserts compatibility with the installed
:mod:`microcosm.frame` kernel — the constellation mechanism from DESIGN.md.

The public names below load on first use (PEP 562). Importing one submodule,
such as the telemetry emitter service, therefore runs only the compatibility
gate here, not the torch, pandas and calibration stack behind the gates and
country specs.
"""

from importlib import import_module as _import_module
from importlib import metadata as _metadata
from typing import TYPE_CHECKING

#: The microcosm-frame series this shard is built against (pre-1.0: minor-level
#: compatibility; microcosm-calibrate applies the same check to
#: ``microcosm.frame.__version__``).
_REQUIRED_FRAME_SERIES = (0, 1)


def _installed_frame_version() -> str:
    """Return the installed microcosm-frame version without importing the kernel.

    Importing :mod:`microcosm.frame` loads numpy and pandas, which every process
    importing any build submodule would then pay for. The gate guards against a
    resolver assembling an incompatible installed set, and the distribution
    metadata is the version that was installed. A frame importable from the path
    without an installed distribution falls back to its module attribute.
    """

    try:
        return _metadata.version("microcosm-frame")
    except _metadata.PackageNotFoundError:
        from microcosm.frame import __version__

        return __version__


def _assert_frame_compatible(version: str, required: tuple[int, int]) -> None:
    """Raise unless the installed microcosm-frame is the expected series."""
    parts = version.split(".")
    try:
        installed = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):  # pragma: no cover - defensive
        raise ImportError(
            f"microcosm-build cannot parse microcosm-frame version "
            f"{version!r}; expected a {required[0]}.{required[1]}.x kernel."
        ) from None

    if required[0] == 0:
        compatible = installed == required
        expected = f"{required[0]}.{required[1]}.x"
    else:
        compatible = installed[0] == required[0]
        expected = f"{required[0]}.x"

    if not compatible:
        raise ImportError(
            f"microcosm-build requires microcosm-frame {expected}, but "
            f"{version} is installed. Install the matching constellation "
            "(the workspace releases the shards in lockstep): upgrade or pin "
            f"microcosm-frame to {expected}."
        )


_assert_frame_compatible(_installed_frame_version(), _REQUIRED_FRAME_SERIES)

if TYPE_CHECKING:
    from microcosm.build.chronicle_epoch import (
        ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS,
        ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS,
        CHRONICLE_EPOCH,
        DECLARED_IDENTITIES,
        DECLARED_IDENTITY_EPOCHS,
        EPOCHS,
        LEDGER_EPOCH,
        UNDECLARED,
        fact_key_epoch,
        fact_key_epoch_label,
        feed_fact_key_epochs,
        feed_undeclared_fact_key_domains,
        is_chronicle_fact_key,
        parse_fact_key,
    )
    from microcosm.build.country_spec import (
        CountryResourceRow,
        CountrySpec,
        GateSelectionSpec,
        GatesManifest,
        GeographySpineManifest,
        GeographySpineSpec,
        ReleaseContractManifest,
        ResolvedCountrySpec,
        country_stage_plan,
        load_country_spec,
    )
    from microcosm.build.gate_battery import (
        BlockingMode,
        EvidenceContext,
        FunctionBinding,
        GateBatteryBlockedError,
        GateBatteryRun,
        GateBinding,
        GateOutcome,
        GatePhaseReport,
        GateStatus,
        evaluate_phase,
        validate_gate_parameters,
    )
    from microcosm.build.gates import (
        FitWeightRecord,
        GateReport,
        GateResult,
        TargetCoverageRequirement,
        TargetFitRequirement,
        aggregate_admin_gate,
        area_support_gate,
        column_implication_gate,
        default_valued_columns_gate,
        enum_domain_gate,
        export_surface_gate,
        exported_nonzero_gate,
        formula_owned_export_gate,
        input_column_coverage_gate,
        input_mass_parity_gate,
        ledger_compile_parity_gate,
        ledger_compile_parity_signed_differences,
        macro_realism_gate,
        nonconstant_columns_gate,
        nonnegative_columns_gate,
        parity_gate,
        per_family_fit_gate,
        relative_error_loss,
        source_coverage_gate,
        source_stage_input_coverage_gate,
        support_gate,
        tail_concentration_gate,
        target_fit_gate,
        target_profile_coverage_gate,
        target_surface_gate,
        weights_audit_gate,
    )
    from microcosm.build.holdout import (
        hash_holdout_uniform,
        hash_holdout_unit,
        rotated_folds,
        summarize_rotations,
    )
    from microcosm.build.ledger_artifact import (
        LedgerConsumerArtifact,
        add_ledger_artifact_args,
        load_ledger_consumer_artifact,
        resolve_ledger_artifact,
    )
    from microcosm.build.ledger_targets import (
        LedgerTargetMapping,
        LedgerTargetSelection,
        UnsupportedLedgerTarget,
        apply_ledger_target_profile,
        select_ledger_targets,
        select_ledger_targets_from_jsonl,
        target_spec_from_ledger_fact,
    )

    # Only ``logbook_env_names`` is re-exported here. The reader function is
    # spelled ``logbook_env`` — the same name as its module — so binding it on
    # the package would make ``import microcosm.build.logbook_env as env`` hand
    # back the function instead of the module. Import the function from its
    # module: ``from microcosm.build.logbook_env import logbook_env``.
    from microcosm.build.logbook_env import (
        logbook_env_names,
    )
    from microcosm.build.monetary_profile import (
        MonetaryTargetContract,
        MonetaryTargetProfile,
    )
    from microcosm.build.monetary_targets import (
        MonetaryBasis,
        PreparedMonetaryMeasure,
        bind_monetary_target,
        prepare_monetary_measure,
    )
    from microcosm.build.observation import (
        ObservedTransform,
        StageEventObserver,
        StageEventRun,
        StageEventStatus,
        StageObservation,
        StageObservationRun,
        StageObserver,
    )
    from microcosm.build.plan import (
        DonorSpec,
        Stage,
        StagePlan,
        StageRecord,
    )
    from microcosm.build.source_runtime import (
        SourceRuntimeConfig,
        SourceRuntimeContext,
        SourceRuntimeError,
        UnsupportedSourceOperationError,
        run_source_stage,
    )
    from microcosm.build.staging import (
        DEFAULT_STAGING_PREFIX,
        LATEST_STAGING_POINTER,
        RUNS_INDEX,
        STAGING_SCHEMA_VERSION,
        StagingRunBundleWriter,
    )


#: Each public name, grouped by the submodule that defines it. The
#: ``TYPE_CHECKING`` imports above name the same pairs for type checkers and
#: for the source-import closures behind gate-binding and worker identities.
_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "chronicle_epoch": (
        "ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS",
        "ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS",
        "CHRONICLE_EPOCH",
        "DECLARED_IDENTITIES",
        "DECLARED_IDENTITY_EPOCHS",
        "EPOCHS",
        "LEDGER_EPOCH",
        "UNDECLARED",
        "fact_key_epoch",
        "fact_key_epoch_label",
        "feed_fact_key_epochs",
        "feed_undeclared_fact_key_domains",
        "is_chronicle_fact_key",
        "parse_fact_key",
    ),
    "country_spec": (
        "CountryResourceRow",
        "CountrySpec",
        "GateSelectionSpec",
        "GatesManifest",
        "GeographySpineManifest",
        "GeographySpineSpec",
        "ReleaseContractManifest",
        "ResolvedCountrySpec",
        "country_stage_plan",
        "load_country_spec",
    ),
    "gate_battery": (
        "BlockingMode",
        "EvidenceContext",
        "FunctionBinding",
        "GateBatteryBlockedError",
        "GateBatteryRun",
        "GateBinding",
        "GateOutcome",
        "GatePhaseReport",
        "GateStatus",
        "evaluate_phase",
        "validate_gate_parameters",
    ),
    "gates": (
        "FitWeightRecord",
        "GateReport",
        "GateResult",
        "TargetCoverageRequirement",
        "TargetFitRequirement",
        "aggregate_admin_gate",
        "area_support_gate",
        "column_implication_gate",
        "default_valued_columns_gate",
        "enum_domain_gate",
        "export_surface_gate",
        "exported_nonzero_gate",
        "formula_owned_export_gate",
        "input_column_coverage_gate",
        "input_mass_parity_gate",
        "ledger_compile_parity_gate",
        "ledger_compile_parity_signed_differences",
        "macro_realism_gate",
        "nonconstant_columns_gate",
        "nonnegative_columns_gate",
        "parity_gate",
        "per_family_fit_gate",
        "relative_error_loss",
        "source_coverage_gate",
        "source_stage_input_coverage_gate",
        "support_gate",
        "tail_concentration_gate",
        "target_fit_gate",
        "target_profile_coverage_gate",
        "target_surface_gate",
        "weights_audit_gate",
    ),
    "holdout": (
        "hash_holdout_uniform",
        "hash_holdout_unit",
        "rotated_folds",
        "summarize_rotations",
    ),
    "ledger_artifact": (
        "LedgerConsumerArtifact",
        "add_ledger_artifact_args",
        "load_ledger_consumer_artifact",
        "resolve_ledger_artifact",
    ),
    "ledger_targets": (
        "LedgerTargetMapping",
        "LedgerTargetSelection",
        "UnsupportedLedgerTarget",
        "apply_ledger_target_profile",
        "select_ledger_targets",
        "select_ledger_targets_from_jsonl",
        "target_spec_from_ledger_fact",
    ),
    # Never ``logbook_env`` itself: see the note on its import above.
    "logbook_env": ("logbook_env_names",),
    "monetary_profile": (
        "MonetaryTargetContract",
        "MonetaryTargetProfile",
    ),
    "monetary_targets": (
        "MonetaryBasis",
        "PreparedMonetaryMeasure",
        "bind_monetary_target",
        "prepare_monetary_measure",
    ),
    "observation": (
        "ObservedTransform",
        "StageEventObserver",
        "StageEventRun",
        "StageEventStatus",
        "StageObservation",
        "StageObservationRun",
        "StageObserver",
    ),
    "plan": (
        "DonorSpec",
        "Stage",
        "StagePlan",
        "StageRecord",
    ),
    "source_runtime": (
        "SourceRuntimeConfig",
        "SourceRuntimeContext",
        "SourceRuntimeError",
        "UnsupportedSourceOperationError",
        "run_source_stage",
    ),
    "staging": (
        "DEFAULT_STAGING_PREFIX",
        "LATEST_STAGING_POINTER",
        "RUNS_INDEX",
        "STAGING_SCHEMA_VERSION",
        "StagingRunBundleWriter",
    ),
}
_EXPORTS = {
    name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names
}


def __getattr__(name: str) -> object:
    """Load a public name from its defining submodule on first use."""

    try:
        module = _EXPORTS[name]
    except KeyError:
        # AttributeError, not ImportError: ``from microcosm.build import
        # <submodule>`` falls back to importing the submodule only on it.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(_import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__version__ = "0.1.0"

__all__ = [
    "BlockingMode",
    "CountrySpec",
    "CountryResourceRow",
    "DonorSpec",
    "DEFAULT_STAGING_PREFIX",
    "EvidenceContext",
    "FitWeightRecord",
    "FunctionBinding",
    "GateBatteryBlockedError",
    "GateBatteryRun",
    "GateBinding",
    "GateOutcome",
    "GatePhaseReport",
    "GateReport",
    "GateResult",
    "GateSelectionSpec",
    "GateStatus",
    "GatesManifest",
    "evaluate_phase",
    "validate_gate_parameters",
    "GeographySpineManifest",
    "GeographySpineSpec",
    "ReleaseContractManifest",
    "ResolvedCountrySpec",
    "Stage",
    "StageEventObserver",
    "StageEventRun",
    "StageEventStatus",
    "StageObservation",
    "StagePlan",
    "StageRecord",
    "SourceRuntimeConfig",
    "SourceRuntimeContext",
    "SourceRuntimeError",
    "LATEST_STAGING_POINTER",
    "RUNS_INDEX",
    "STAGING_SCHEMA_VERSION",
    "StagingRunBundleWriter",
    "TargetCoverageRequirement",
    "TargetFitRequirement",
    "ACCEPTED_CONSUMER_ARTIFACT_SCHEMA_VERSIONS",
    "ACCEPTED_CONSUMER_FACT_SCHEMA_VERSIONS",
    "CHRONICLE_EPOCH",
    "DECLARED_IDENTITIES",
    "DECLARED_IDENTITY_EPOCHS",
    "EPOCHS",
    "LEDGER_EPOCH",
    "UNDECLARED",
    "LedgerConsumerArtifact",
    "LedgerTargetMapping",
    "LedgerTargetSelection",
    "MonetaryBasis",
    "MonetaryTargetContract",
    "MonetaryTargetProfile",
    "ObservedTransform",
    "PreparedMonetaryMeasure",
    "StageObservationRun",
    "StageObserver",
    "add_ledger_artifact_args",
    "logbook_env_names",
    "fact_key_epoch",
    "fact_key_epoch_label",
    "feed_fact_key_epochs",
    "feed_undeclared_fact_key_domains",
    "is_chronicle_fact_key",
    "parse_fact_key",
    "aggregate_admin_gate",
    "area_support_gate",
    "column_implication_gate",
    "apply_ledger_target_profile",
    "bind_monetary_target",
    "default_valued_columns_gate",
    "enum_domain_gate",
    "export_surface_gate",
    "exported_nonzero_gate",
    "formula_owned_export_gate",
    "hash_holdout_uniform",
    "hash_holdout_unit",
    "input_column_coverage_gate",
    "input_mass_parity_gate",
    "ledger_compile_parity_gate",
    "ledger_compile_parity_signed_differences",
    "load_ledger_consumer_artifact",
    "macro_realism_gate",
    "nonconstant_columns_gate",
    "nonnegative_columns_gate",
    "parity_gate",
    "prepare_monetary_measure",
    "per_family_fit_gate",
    "relative_error_loss",
    "resolve_ledger_artifact",
    "rotated_folds",
    "source_coverage_gate",
    "source_stage_input_coverage_gate",
    "select_ledger_targets",
    "select_ledger_targets_from_jsonl",
    "summarize_rotations",
    "support_gate",
    "tail_concentration_gate",
    "target_fit_gate",
    "target_profile_coverage_gate",
    "target_spec_from_ledger_fact",
    "target_surface_gate",
    "weights_audit_gate",
    "UnsupportedLedgerTarget",
    "UnsupportedSourceOperationError",
    "country_stage_plan",
    "load_country_spec",
    "run_source_stage",
    "__version__",
]
