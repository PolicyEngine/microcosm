"""Generic source-stage manifests for country build plans.

Country packages own source content as data: JSON manifests describing source
artifacts, required transformations, outputs, and validation requirements. The
Python here is the shared interpreter contract only; it is intentionally not a
country-specific donor loader.

Raw microdata roots additionally carry an identity contract (microcosm#848,
Chronicle ADR "Raw microdata in Chronicle is identity, not content"): every
artifact entry whose ``kind`` names microdata declares the SHA-256 of the exact
file a stage reads and a ``chronicle_artifact`` reference to the one witnessed
Chronicle registration of that file. Entries that cannot be pinned yet are
listed, one row each, in the country's ``microdata_pins_pending.json``
allowlist, which is a ratchet: it may shrink, never grow past its committed
baseline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_SOURCE_OPERATION_KINDS",
    "CHRONICLE_ACCESS_CLASSES",
    "FORBIDDEN_EXECUTABLE_LOADER_KEYS",
    "FORBIDDEN_EXECUTABLE_OPERATION_KINDS",
    "FORBIDDEN_SOURCE_DEPENDENCIES",
    "MICRODATA_ARTIFACT_KINDS",
    "MICRODATA_PIN_ALLOWLIST_FILENAME",
    "EMPTY_MICRODATA_PIN_ALLOWLIST",
    "ChronicleArtifactReference",
    "MicrodataArtifactEntry",
    "MicrodataPinAllowlist",
    "MicrodataPinGap",
    "MicrodataPinPendingEntry",
    "SourceManifest",
    "SourceOperationSpec",
    "SourceStageSpec",
    "SupportSpineManifest",
    "SupportSpineSourceSpec",
    "SupportSpineSpec",
    "audit_microdata_pins",
    "load_microdata_pin_allowlist",
    "load_source_manifest",
    "load_support_spine_manifest",
    "microdata_artifact_entries",
    "resolved_chronicle_registrations",
]


FORBIDDEN_SOURCE_DEPENDENCIES = (
    "policyengine_" + "us_data",
    "policyengine-" + "us-data",
    "policyengine_" + "uk_data",
    "policyengine-" + "uk-data",
)

ALLOWED_SOURCE_OPERATION_KINDS = frozenset(
    {
        "aggregate_person_to_household",
        "aggregate_person_to_tax_unit",
        "assign_by_plan_type",
        "assign_binary_from_banded_rates",
        "assign_binary_from_rate",
        "assign_binary_with_anchored_residual",
        "assign_clipped_normal",
        "assign_student_loan_plan_cohorts",
        "assign_uniform_draw",
        "aggregate_person_to_benunit",
        "allocate_per_capita_from_cell_table",
        "allocate_within_group_waterfall",
        "allocate_zero_weight_prior_mass",
        "annualize_periodic_amounts",
        "assemble_group_entities",
        "attribute_self_employed_health_premiums",
        "bridge_donor_column_via_qrf",
        "calibrate_binary_assignment",
        "calibrate_binary_assignment_joint_targets",
        "classify_cgt_band_facts_with_reviewed_fence",
        "classify_hmrc_income_facts_with_reviewed_fences",
        "clone_records",
        "convert_donors_to_target_stock",
        "convert_interest_to_structural_mortgage_inputs",
        "compute_ratio",
        "declare_income_reference_offset",
        "derive",
        "derive_adult_care_inputs",
        "derive_childcare_inputs",
        "derive_child_support_inputs",
        "derive_disability_benefits",
        "disaggregate_top_coded_ages",
        "derive_energy_subsidy",
        "derive_education_inputs",
        "derive_eligibility_inputs",
        "derive_hours_worked",
        "derive_housing_tenure_inputs",
        "derive_immigration_status",
        "derive_medicare_take_up",
        "derive_other_health_insurance_premiums",
        "derive_prior_year_income",
        "derive_snap_abawd_discretionary_exemption",
        "derive_snap_take_up",
        "derive_puf_policyengine_variables",
        "derive_mortgage_balance_hints",
        "derive_pregnancy",
        "derive_relationship_inputs",
        "derive_retirement_distributions",
        "derive_retirement_contributions",
        "derive_workers_compensation",
        "derive_weeks_unemployed",
        "derive_wic_claim",
        "disaggregate_aggregate_records",
        "draw_capital_gains_prior_from_banded_quantiles",
        "fit_labor_market_models",
        "fit_tip_income_model",
        "fit_weighted_acs_rent_qrf",
        "fit_vehicle_model",
        "fit_weighted_imputer",
        "fit_weighted_qrf",
        "fit_weighted_qrf_chain",
        "fit_weighted_qrf_stage1",
        "fit_weighted_qrf_stage2",
        "fold_into",
        "gate_distributional_effective_mass",
        "gate_zero_weight_strata",
        "head_carry",
        "join",
        "impute_retirement_contributions_to_puf_support",
        "impute_childcare_to_puf_support",
        "impute_child_support_to_puf_support",
        "impute_disability_benefits_to_puf_support",
        "impute_energy_subsidy_to_puf_support",
        "impute_cell_means",
        "impute_housing_assistance_to_puf_support",
        "impute_other_health_insurance_premiums_to_puf_support",
        "impute_prior_year_income_to_puf_support",
        "impute_retirement_distributions_to_puf_support",
        "impute_workers_compensation_to_puf_support",
        "impute_weeks_unemployed_to_puf_support",
        "iterative_proportional_fit",
        "map_columns",
        "map_coded_amounts",
        "materialize_hmrc_income_bands_fail_closed",
        "materialize_rules_engine_predictors",
        "rank_preserving_allocation",
        "read_table",
        "read_tables",
        "read_acs_rent_donor",
        "redraw_columns_from_fitted_qrf",
        "redraw_spi_reported_uc",
        "redraw_spi_reporter_capital",
        "record_mass_conservation_receipt",
        "replace_zero_weight_spi_support",
        "retain_adjudicated_frs_hmrc_leaves",
        "sample_categorical_from_count_table",
        "replace_sentinels",
        "split_component_by_share",
        "stack_band_donor_households",
        "stack_zero_weight_donors",
        "strict_read_private_table",
        "support_clip",
        "sub_aea_remainder",
        "taxable_income_proxy",
        "top_up_to_stock",
        "uprate",
        "uprate_to_regional_reference",
        "verify_certified_candidate",
        "verify_pinned_cgt_ods",
        "verify_pinned_hmrc_source_pair",
        "within_band_draws",
        "zero_when_false",
    }
)

FORBIDDEN_EXECUTABLE_OPERATION_KINDS = frozenset(
    {
        "callable",
        "exec",
        "function",
        "import",
        "import_module",
        "module",
        "python",
        "python_callable",
        "python_function",
        "python_module",
    }
)

FORBIDDEN_EXECUTABLE_LOADER_KEYS = frozenset(
    {
        "callable",
        "callback",
        "entry",
        "entrypoint",
        "entry_point",
        "function",
        "handler",
        "import",
        "loader",
        "module",
        "python",
    }
)

ALLOWED_SUPPORT_SPINE_METHODS = frozenset({"pool_raw_asec_years"})

# Artifact kinds whose bytes are raw microdata a build reads. Every entry of one
# of these kinds is a root of the build graph, so it carries a SHA-256 pin and a
# Chronicle registration reference, or an explicit allowlist row saying why not.
MICRODATA_ARTIFACT_KINDS = frozenset(
    {
        "licensed_microdata",
        "private_microdata",
        "public_microdata",
        "restricted_microdata",
        "versioned_derived_microdata",
    }
)

# Chronicle's closed access set. ``public`` is the only class whose bytes are
# archived; ``licensed`` and ``restricted`` registrations are hash-only and the
# bytes stay in the licensed environment the build already operates.
CHRONICLE_ACCESS_CLASSES = frozenset({"public", "licensed", "restricted"})

MICRODATA_PIN_ALLOWLIST_FILENAME = "microdata_pins_pending.json"

_CHRONICLE_ARTIFACT_REQUIRED_KEYS = frozenset(
    {"access", "package_id", "sha256", "source_id", "year"}
)
_CHRONICLE_ARTIFACT_OPTIONAL_KEYS = frozenset({"filename"})
_MICRODATA_PIN_PENDING_KEYS = frozenset({"issue", "locator", "reason", "stage"})
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHRONICLE_SLUG = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")


@dataclass(frozen=True)
class ChronicleArtifactReference:
    """One witnessed Chronicle registration of a raw microdata file.

    ``access`` is Chronicle's closed class. Bytes exist in the raw bucket only
    for ``public`` registrations; ``licensed`` and ``restricted`` ones are
    hash-only by design, so :attr:`raw_object_key` is ``None`` for them.
    """

    source_id: str
    package_id: str
    year: int
    sha256: str
    access: str
    filename: str = ""

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, context: str
    ) -> ChronicleArtifactReference:
        keys = frozenset(raw)
        missing = sorted(_CHRONICLE_ARTIFACT_REQUIRED_KEYS - keys)
        if missing:
            raise ValueError(
                f"{context} chronicle_artifact is missing required key(s): {missing}."
            )
        unknown = sorted(
            keys - _CHRONICLE_ARTIFACT_REQUIRED_KEYS - _CHRONICLE_ARTIFACT_OPTIONAL_KEYS
        )
        if unknown:
            raise ValueError(
                f"{context} chronicle_artifact declares unknown key(s): {unknown}."
            )
        for key in ("source_id", "package_id"):
            value = raw[key]
            if not isinstance(value, str) or not _CHRONICLE_SLUG.fullmatch(value):
                raise ValueError(
                    f"{context} chronicle_artifact {key!r} must be a lowercase "
                    f"slug, got {value!r}."
                )
        year = raw["year"]
        if not isinstance(year, int) or isinstance(year, bool):
            raise ValueError(
                f"{context} chronicle_artifact 'year' must be an integer, got {year!r}."
            )
        sha256 = raw["sha256"]
        if not isinstance(sha256, str) or not _LOWERCASE_SHA256.fullmatch(sha256):
            raise ValueError(
                f"{context} chronicle_artifact 'sha256' must be 64 lowercase hex "
                f"characters, got {sha256!r}."
            )
        access = raw["access"]
        if access not in CHRONICLE_ACCESS_CLASSES:
            raise ValueError(
                f"{context} chronicle_artifact 'access' must be one of "
                f"{sorted(CHRONICLE_ACCESS_CLASSES)}, got {access!r}."
            )
        filename = raw.get("filename", "")
        if not isinstance(filename, str):
            raise ValueError(
                f"{context} chronicle_artifact 'filename' must be a string."
            )
        if access == "public" and not filename:
            raise ValueError(
                f"{context} chronicle_artifact declares public access without a "
                "'filename'; the archived object key needs one."
            )
        return cls(
            source_id=raw["source_id"],
            package_id=raw["package_id"],
            year=year,
            sha256=sha256,
            access=access,
            filename=filename,
        )

    @property
    def raw_object_key(self) -> str | None:
        """Content-addressed raw-bucket key, or ``None`` when no bytes exist."""

        if self.access != "public":
            return None
        return (
            f"raw/{self.source_id}/{self.package_id}/{self.year}/"
            f"{self.sha256}/{self.filename}"
        )

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON-ready registration record for build manifests."""

        payload: dict[str, Any] = {
            "access": self.access,
            "package_id": self.package_id,
            "sha256": self.sha256,
            "source_id": self.source_id,
            "year": self.year,
        }
        if self.filename:
            payload["filename"] = self.filename
        key = self.raw_object_key
        if key is not None:
            payload["raw_object_key"] = key
        return payload


@dataclass(frozen=True)
class MicrodataArtifactEntry:
    """One raw microdata artifact entry, located by stage and locator."""

    stage: str
    locator: str
    kind: str
    artifact: Mapping[str, Any]
    chronicle_artifact: ChronicleArtifactReference | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.stage, self.locator)

    @property
    def sha256(self) -> str | None:
        value = self.artifact.get("sha256")
        return value if isinstance(value, str) else None

    @property
    def member_sha256(self) -> str | None:
        value = self.artifact.get("member_sha256")
        return value if isinstance(value, str) else None

    @property
    def is_pinned(self) -> bool:
        return self.sha256 is not None and self.chronicle_artifact is not None


@dataclass(frozen=True)
class MicrodataPinPendingEntry:
    """One reviewed reason a microdata root is not pinned yet."""

    stage: str
    locator: str
    reason: str
    issue: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.stage, self.locator)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, context: str
    ) -> MicrodataPinPendingEntry:
        keys = frozenset(raw)
        if keys != _MICRODATA_PIN_PENDING_KEYS:
            raise ValueError(
                f"{context} pending row must declare exactly "
                f"{sorted(_MICRODATA_PIN_PENDING_KEYS)}, got {sorted(keys)}."
            )
        for key in sorted(_MICRODATA_PIN_PENDING_KEYS):
            value = raw[key]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{context} pending row key {key!r} must be a non-empty string."
                )
        return cls(
            stage=raw["stage"],
            locator=raw["locator"],
            reason=raw["reason"],
            issue=raw["issue"],
        )


@dataclass(frozen=True)
class MicrodataPinAllowlist:
    """Country allowlist of microdata roots that are not pinned yet.

    ``baseline_count`` is the ratchet: the committed number of rows this country
    is allowed to carry. Loading refuses a file whose row count exceeds it, so a
    new unpinned root cannot land without either pinning something else or a
    reviewed baseline change.
    """

    country: str
    version: int
    policy: str
    baseline_count: int
    pending: tuple[MicrodataPinPendingEntry, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MicrodataPinAllowlist:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        baseline_count = raw.get("baseline_count")
        if not isinstance(country, str) or not country:
            raise ValueError("microdata pin allowlist requires a non-empty 'country'.")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(
                "microdata pin allowlist requires positive integer 'version'."
            )
        if not isinstance(policy, str) or not policy:
            raise ValueError("microdata pin allowlist requires a non-empty 'policy'.")
        if (
            not isinstance(baseline_count, int)
            or isinstance(baseline_count, bool)
            or baseline_count < 0
        ):
            raise ValueError(
                "microdata pin allowlist requires a non-negative integer "
                "'baseline_count'."
            )
        context = f"{country} microdata pin allowlist"
        pending = tuple(
            MicrodataPinPendingEntry.from_mapping(row, context=context)
            for row in _require_mapping_sequence(raw.get("pending", ()))
        )
        keys = [row.key for row in pending]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"{context} repeats pending row(s): {duplicates}.")
        if len(pending) > baseline_count:
            raise ValueError(
                f"{context} carries {len(pending)} pending row(s), above its "
                f"committed baseline of {baseline_count}; the allowlist is a "
                "ratchet and may only shrink."
            )
        return cls(
            country=country,
            version=version,
            policy=policy,
            baseline_count=baseline_count,
            pending=pending,
        )

    def row_map(self) -> Mapping[tuple[str, str], MicrodataPinPendingEntry]:
        return {row.key: row for row in self.pending}


EMPTY_MICRODATA_PIN_ALLOWLIST = MicrodataPinAllowlist(
    country="",
    version=1,
    policy="No allowlist file: every microdata root must be pinned.",
    baseline_count=0,
    pending=(),
)


@dataclass(frozen=True)
class MicrodataPinGap:
    """One contract violation found by :func:`audit_microdata_pins`."""

    stage: str
    locator: str
    problem: str
    detail: str

    def message(self) -> str:
        return f"{self.stage} / {self.locator}: {self.detail}"


@dataclass(frozen=True)
class SourceOperationSpec:
    """One declarative source operation.

    The operation names are generic primitives such as ``read_table``,
    ``replace_sentinels``, ``derive``, or ``fit_weighted_qrf``. Source manifests
    must not point at country-specific Python modules or incumbent data-package
    helpers.
    """

    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceOperationSpec:
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("source operation requires a non-empty string 'kind'.")
        parameters = {k: v for k, v in raw.items() if k != "kind"}
        _reject_executable_loader_shape(kind, parameters)
        _reject_incumbent_dependencies(parameters, context=f"operation {kind!r}")
        return cls(kind=kind, parameters=parameters)


@dataclass(frozen=True)
class SourceStageSpec:
    """Declarative source-stage contract for one build stage."""

    stage: str
    survey: str
    source: str
    grain: str
    artifacts: tuple[Mapping[str, Any], ...]
    operations: tuple[SourceOperationSpec, ...]
    outputs: tuple[str, ...]
    nonnegative_outputs: tuple[str, ...] = ()
    rewrites: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceStageSpec:
        required = ("stage", "survey", "source", "grain", "operations", "outputs")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"source stage is missing required key(s): {missing}.")
        for key in ("stage", "survey", "source", "grain"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError(
                    f"source stage key {key!r} must be a non-empty string."
                )
        artifacts = tuple(_require_mapping_sequence(raw.get("artifacts", ())))
        operations = tuple(
            SourceOperationSpec.from_mapping(operation)
            for operation in _require_mapping_sequence(raw["operations"])
        )
        outputs = tuple(_require_string_sequence(raw["outputs"], key="outputs"))
        nonnegative_outputs = tuple(
            _require_string_sequence(
                raw.get("nonnegative_outputs", ()),
                key="nonnegative_outputs",
            )
        )
        rewrites = tuple(
            _require_string_sequence(raw.get("rewrites", ()), key="rewrites")
        )
        unknown_nonnegative = sorted(set(nonnegative_outputs) - set(outputs))
        if unknown_nonnegative:
            raise ValueError(
                f"stage {raw['stage']!r} marks nonnegative output(s) not in outputs: "
                f"{unknown_nonnegative}."
            )
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("source stage 'notes' must be a string when provided.")
        _reject_executable_parameter_keys(raw, context=f"stage {raw['stage']!r}")
        _reject_incumbent_dependencies(raw, context=f"stage {raw['stage']!r}")
        # Runs last so a stage that smuggles an executable loader is reported as
        # that, not as a malformed microdata root.
        _validate_microdata_artifacts(artifacts, stage=raw["stage"])
        return cls(
            stage=raw["stage"],
            survey=raw["survey"],
            source=raw["source"],
            grain=raw["grain"],
            artifacts=artifacts,
            operations=operations,
            outputs=outputs,
            nonnegative_outputs=nonnegative_outputs,
            rewrites=rewrites,
            notes=notes,
        )


@dataclass(frozen=True)
class SourceManifest:
    """A country source manifest loaded from packaged JSON."""

    country: str
    version: int
    policy: str
    stages: tuple[SourceStageSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SourceManifest:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        if not isinstance(country, str) or not country:
            raise ValueError("source manifest requires a non-empty 'country'.")
        if not isinstance(version, int) or version < 1:
            raise ValueError("source manifest requires positive integer 'version'.")
        if not isinstance(policy, str) or not policy:
            raise ValueError("source manifest requires a non-empty 'policy'.")
        stages = tuple(
            SourceStageSpec.from_mapping(stage)
            for stage in _require_mapping_sequence(raw.get("stages", ()))
        )
        names = [stage.stage for stage in stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate source stage spec(s): {duplicates}.")
        _reject_executable_parameter_keys(raw, context=f"{country} source manifest")
        _reject_incumbent_dependencies(raw, context=f"{country} source manifest")
        return cls(country=country, version=version, policy=policy, stages=stages)

    def stage_map(self) -> Mapping[str, SourceStageSpec]:
        return {stage.stage: stage for stage in self.stages}


@dataclass(frozen=True)
class SupportSpineSourceSpec:
    """One source-year rule in a support-spine manifest.

    ``source_year_offset`` is relative to the build target year so country
    specs do not bake in a dataset period. For example, ``0`` means the
    target-year ASEC file and ``-1`` means the prior ASEC file.
    """

    role: str
    survey: str
    source: str
    source_year_offset: int
    share: float | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineSourceSpec:
        required = ("role", "survey", "source", "source_year_offset")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(
                f"support-spine source is missing required key(s): {missing}."
            )
        for key in ("role", "survey", "source"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError(
                    f"support-spine source key {key!r} must be a non-empty string."
                )
        source_year_offset = raw["source_year_offset"]
        if not isinstance(source_year_offset, int) or isinstance(
            source_year_offset, bool
        ):
            raise ValueError("support-spine source_year_offset must be an integer.")
        share = raw.get("share")
        if share is not None:
            if (
                not isinstance(share, int | float)
                or isinstance(share, bool)
                or float(share) <= 0.0
            ):
                raise ValueError("support-spine source share must be positive.")
            share = float(share)
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("support-spine source notes must be a string.")
        _reject_executable_parameter_keys(
            raw, context=f"support-spine source {raw['role']!r}"
        )
        _reject_incumbent_dependencies(
            raw, context=f"support-spine source {raw['role']!r}"
        )
        return cls(
            role=raw["role"],
            survey=raw["survey"],
            source=raw["source"],
            source_year_offset=source_year_offset,
            share=share,
            notes=notes,
        )

    def resolved_year(self, target_year: int) -> int:
        return int(target_year) + self.source_year_offset


@dataclass(frozen=True)
class SupportSpineSpec:
    """Declarative support-spine construction contract."""

    stage: str
    method: str
    target_year_from_build_config: bool
    sources: tuple[SupportSpineSourceSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineSpec:
        required = ("stage", "method", "target_year_from_build_config", "sources")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"support-spine spec is missing key(s): {missing}.")
        stage = raw["stage"]
        method = raw["method"]
        if not isinstance(stage, str) or not stage:
            raise ValueError("support-spine stage must be a non-empty string.")
        if method not in ALLOWED_SUPPORT_SPINE_METHODS:
            raise ValueError(
                f"support-spine method {method!r} is not supported; allowed "
                f"methods are {sorted(ALLOWED_SUPPORT_SPINE_METHODS)}."
            )
        target_year_from_build_config = raw["target_year_from_build_config"]
        if not isinstance(target_year_from_build_config, bool):
            raise ValueError(
                "support-spine target_year_from_build_config must be boolean."
            )
        if not target_year_from_build_config:
            raise ValueError(
                "support-spine target_year_from_build_config must be true; "
                "period-specific source years belong in runtime build inputs."
            )
        sources = tuple(
            SupportSpineSourceSpec.from_mapping(source)
            for source in _require_mapping_sequence(raw["sources"])
        )
        if not sources:
            raise ValueError("support-spine spec requires at least one source.")
        shares = [source.share for source in sources]
        if any(share is None for share in shares):
            raise ValueError("support-spine sources must declare explicit shares.")
        total = sum(float(share) for share in shares if share is not None)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"support-spine source shares must sum to 1, got {total}.")
        _reject_executable_parameter_keys(raw, context=f"support-spine {stage!r}")
        _reject_incumbent_dependencies(raw, context=f"support-spine {stage!r}")
        return cls(
            stage=stage,
            method=method,
            target_year_from_build_config=target_year_from_build_config,
            sources=sources,
        )


@dataclass(frozen=True)
class SupportSpineManifest:
    """Country support-spine manifest loaded from packaged JSON."""

    country: str
    version: int
    policy: str
    support_spine: SupportSpineSpec

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SupportSpineManifest:
        country = raw.get("country")
        version = raw.get("version")
        policy = raw.get("policy", "")
        if not isinstance(country, str) or not country:
            raise ValueError("support-spine manifest requires a non-empty 'country'.")
        if not isinstance(version, int) or version < 1:
            raise ValueError(
                "support-spine manifest requires positive integer 'version'."
            )
        if not isinstance(policy, str) or not policy:
            raise ValueError("support-spine manifest requires a non-empty 'policy'.")
        support_spine = raw.get("support_spine")
        if not isinstance(support_spine, Mapping):
            raise ValueError("support-spine manifest requires object 'support_spine'.")
        _reject_executable_parameter_keys(
            raw, context=f"{country} support-spine manifest"
        )
        _reject_incumbent_dependencies(raw, context=f"{country} support-spine manifest")
        return cls(
            country=country,
            version=version,
            policy=policy,
            support_spine=SupportSpineSpec.from_mapping(support_spine),
        )


def microdata_artifact_entries(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
) -> tuple[MicrodataArtifactEntry, ...]:
    """Return every raw microdata artifact entry declared by ``source``.

    Entries are the roots of a build graph: the files a stage actually reads.
    Each is located by ``(stage, locator)``, which is unique within a manifest.

    A raw manifest mapping is accepted as well as a loaded
    :class:`SourceManifest`, because the frozen UK HMRC/SPI replay manifest is
    read as JSON by its own contract rather than through the shared loader.
    """

    entries: list[MicrodataArtifactEntry] = []
    for stage, artifacts in _iter_stage_artifacts(source):
        for artifact in artifacts:
            if artifact.get("kind") not in MICRODATA_ARTIFACT_KINDS:
                continue
            entries.append(_microdata_artifact_entry(artifact, stage=stage))
    return tuple(entries)


def resolved_chronicle_registrations(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
) -> tuple[ChronicleArtifactReference, ...]:
    """Return the distinct Chronicle registrations ``source`` resolves to.

    Several stages legitimately read the same file — the FRS ``adult`` tab feeds
    five UK stages — and they all resolve to one registration, so the result is
    deduplicated and ordered by ``(source_id, package_id, year, sha256)``.
    """

    registrations = {
        entry.chronicle_artifact
        for entry in microdata_artifact_entries(source)
        if entry.chronicle_artifact is not None
    }
    return tuple(
        sorted(
            registrations,
            key=lambda ref: (ref.source_id, ref.package_id, ref.year, ref.sha256),
        )
    )


def audit_microdata_pins(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
    *,
    allowlist: MicrodataPinAllowlist | None = None,
) -> tuple[MicrodataPinGap, ...]:
    """Return every microdata root that is neither pinned nor allowlisted.

    A root is pinned when it declares both its own ``sha256`` and a
    ``chronicle_artifact`` reference. Anything else must carry an allowlist row
    naming the stage, locator, reason, and tracking issue. A row that names an
    already-pinned root is itself a gap: stale rows would quietly inflate the
    ratchet baseline.
    """

    rows = (allowlist or EMPTY_MICRODATA_PIN_ALLOWLIST).row_map()
    entries = microdata_artifact_entries(source)
    gaps: list[MicrodataPinGap] = []
    for entry in entries:
        if entry.is_pinned:
            if entry.key in rows:
                gaps.append(
                    MicrodataPinGap(
                        stage=entry.stage,
                        locator=entry.locator,
                        problem="stale_allowlist_row",
                        detail=(
                            "is fully pinned but still carries a pending "
                            "allowlist row; remove the row so the ratchet "
                            "baseline can fall."
                        ),
                    )
                )
            continue
        if entry.key in rows:
            continue
        if entry.sha256 is None:
            detail = (
                f"{entry.kind} artifact declares no 'sha256' pin and has no "
                "pending allowlist row."
            )
        else:
            detail = (
                f"{entry.kind} artifact is hash-pinned but declares no "
                "'chronicle_artifact' registration and has no pending "
                "allowlist row."
            )
        gaps.append(
            MicrodataPinGap(
                stage=entry.stage,
                locator=entry.locator,
                problem="unpinned",
                detail=detail,
            )
        )
    known = {entry.key for entry in entries}
    for key in sorted(rows):
        if key not in known:
            gaps.append(
                MicrodataPinGap(
                    stage=key[0],
                    locator=key[1],
                    problem="orphan_allowlist_row",
                    detail=(
                        "pending allowlist row names no microdata artifact in "
                        "this manifest."
                    ),
                )
            )
    return tuple(gaps)


def load_microdata_pin_allowlist(resource: Any) -> MicrodataPinAllowlist:
    """Load and validate a country ``microdata_pins_pending.json`` allowlist."""

    raw = json.loads(_read_manifest_text(resource))
    if not isinstance(raw, Mapping):
        raise ValueError("microdata pin allowlist root must be a JSON object.")
    return MicrodataPinAllowlist.from_mapping(raw)


def load_source_manifest(resource: Any) -> SourceManifest:
    """Load and validate a source manifest from a path-like resource."""
    raw = json.loads(_read_manifest_text(resource))
    if not isinstance(raw, Mapping):
        raise ValueError("source manifest root must be a JSON object.")
    return SourceManifest.from_mapping(raw)


def load_support_spine_manifest(resource: Any) -> SupportSpineManifest:
    """Load and validate a support-spine manifest from a path-like resource."""
    raw = json.loads(_read_manifest_text(resource))
    if not isinstance(raw, Mapping):
        raise ValueError("support-spine manifest root must be a JSON object.")
    return SupportSpineManifest.from_mapping(raw)


def _iter_stage_artifacts(
    source: SourceManifest | SourceStageSpec | Mapping[str, Any],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    if isinstance(source, SourceStageSpec):
        return ((source.stage, source.artifacts),)
    if isinstance(source, SourceManifest):
        return tuple((stage.stage, stage.artifacts) for stage in source.stages)
    if not isinstance(source, Mapping):
        raise TypeError(
            "expected a SourceManifest, SourceStageSpec, or raw manifest mapping, "
            f"got {type(source).__name__}."
        )
    stages = []
    for raw_stage in _require_mapping_sequence(source.get("stages", ())):
        name = raw_stage.get("stage")
        if not isinstance(name, str) or not name:
            raise ValueError("raw source stage requires a non-empty 'stage'.")
        stages.append(
            (name, tuple(_require_mapping_sequence(raw_stage.get("artifacts", ()))))
        )
    return tuple(stages)


def _validate_microdata_artifacts(
    artifacts: Sequence[Mapping[str, Any]], *, stage: str
) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.get("kind") not in MICRODATA_ARTIFACT_KINDS:
            continue
        entry = _microdata_artifact_entry(artifact, stage=stage)
        if entry.locator in seen:
            raise ValueError(
                f"stage {stage!r} repeats microdata locator {entry.locator!r}; "
                "(stage, locator) identifies a raw input."
            )
        seen.add(entry.locator)


def _microdata_artifact_entry(
    artifact: Mapping[str, Any], *, stage: str
) -> MicrodataArtifactEntry:
    kind = artifact["kind"]
    locator = artifact.get("locator")
    if not isinstance(locator, str) or not locator:
        raise ValueError(
            f"stage {stage!r} {kind} artifact requires a non-empty 'locator'."
        )
    context = f"stage {stage!r} microdata artifact {locator!r}"
    for key in ("sha256", "member_sha256"):
        value = artifact.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not _LOWERCASE_SHA256.fullmatch(value):
            raise ValueError(
                f"{context} key {key!r} must be 64 lowercase hex characters, "
                f"got {value!r}."
            )
    raw_reference = artifact.get("chronicle_artifact")
    reference: ChronicleArtifactReference | None = None
    if raw_reference is not None:
        if not isinstance(raw_reference, Mapping):
            raise ValueError(f"{context} 'chronicle_artifact' must be an object.")
        reference = ChronicleArtifactReference.from_mapping(
            raw_reference, context=context
        )
        declared = artifact.get("sha256")
        if not isinstance(declared, str):
            raise ValueError(
                f"{context} references a Chronicle registration without "
                "declaring its own 'sha256'; the reference must witness the "
                "exact bytes this stage reads."
            )
        if reference.sha256 != declared:
            raise ValueError(
                f"{context} chronicle_artifact sha256 {reference.sha256} does "
                f"not equal the artifact sha256 {declared}; a registration "
                "witnesses one file."
            )
        filename = artifact.get("filename")
        if (
            isinstance(filename, str)
            and filename
            and reference.filename
            and reference.filename != filename
        ):
            raise ValueError(
                f"{context} chronicle_artifact filename "
                f"{reference.filename!r} does not equal the artifact filename "
                f"{filename!r}."
            )
    return MicrodataArtifactEntry(
        stage=stage,
        locator=locator,
        kind=kind,
        artifact=artifact,
        chronicle_artifact=reference,
    )


def _read_manifest_text(resource: Any) -> str:
    if hasattr(resource, "read_text"):
        return resource.read_text(encoding="utf-8")
    return Path(resource).read_text(encoding="utf-8")


def _require_mapping_sequence(raw: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("expected a list of objects.")
    result = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("expected every list item to be an object.")
        result.append(dict(item))
    return tuple(result)


def _require_string_sequence(raw: object, *, key: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list of strings.")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{key} must contain only non-empty strings.")
        values.append(item)
    return tuple(values)


def _reject_incumbent_dependencies(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_incumbent_dependencies(key, context=context)
            _reject_incumbent_dependencies(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_incumbent_dependencies(nested, context=context)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        if dependency in lowered:
            raise ValueError(
                f"{context} references forbidden incumbent dependency {dependency!r}."
            )


def _reject_executable_loader_shape(kind: str, parameters: Mapping[str, Any]) -> None:
    normalized_kind = _normalize_manifest_key(kind)
    if (
        normalized_kind in FORBIDDEN_EXECUTABLE_OPERATION_KINDS
        or _is_executable_loader_key(normalized_kind)
    ):
        raise ValueError(
            f"source operation {kind!r} is executable-loader content, not a "
            "declarative source operation."
        )
    if normalized_kind not in ALLOWED_SOURCE_OPERATION_KINDS:
        raise ValueError(
            f"source operation {kind!r} is not in the allowed manifest operation "
            "vocabulary."
        )
    _reject_executable_parameter_keys(parameters, context=f"operation {kind!r}")


def _reject_executable_parameter_keys(value: object, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                normalized = _normalize_manifest_key(key)
                if _is_executable_loader_key(normalized):
                    raise ValueError(
                        f"{context} uses executable-loader key {key!r}; source "
                        "manifests must be declarative."
                    )
            _reject_executable_parameter_keys(nested, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_executable_parameter_keys(nested, context=context)
        return
    if isinstance(value, str) and _looks_like_python_entrypoint(value):
        raise ValueError(
            f"{context} references executable Python entrypoint {value!r}; source "
            "manifests must be declarative."
        )


def _normalize_manifest_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_executable_loader_key(normalized: str) -> bool:
    if normalized in FORBIDDEN_EXECUTABLE_LOADER_KEYS:
        return True
    tokens = normalized.split("_")
    return any(
        token
        in {
            "callable",
            "callback",
            "entry",
            "entrypoint",
            "function",
            "handler",
            "import",
            "loader",
            "module",
            "python",
        }
        for token in tokens
    )


def _looks_like_python_entrypoint(value: str) -> bool:
    if "://" in value:
        return False
    return bool(
        re.search(
            r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*\b",
            value,
        )
    )
