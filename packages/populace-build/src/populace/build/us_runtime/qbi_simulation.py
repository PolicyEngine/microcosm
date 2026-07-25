"""Versioned Section 199A simulation for raw or legacy PUF arrays.

The pinned 1.8.0 PUF physically carries only an older deterministic W-2
business-wage proxy. A later retired data-package loader made that
artifact appear to contain the current 15-leaf QBI surface by simulating the
missing inputs and writing them into the downloaded file during ``load()``.

This module ports that version-1 model into Populace as pure, seeded NumPy
logic. The assumptions, source order, exposure order, bit generator, and four
independent seeds are packaged in ``qbi_assumptions_v1.json``. Callers must
name ``qbi_simulation_version`` explicitly; no version is inferred from an
unversioned artifact.

The simulation core accepts a normalized assumptions object. A future v2 can
load a different, evidence-anchored assumptions object without changing the
source normalization, orchestration, output contract, or donor-stage wrapper.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np

from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS

__all__ = [
    "QBI_SIMULATION_ARCHIVED_ASSUMPTIONS_URL",
    "QBI_SIMULATION_ARCHIVED_BACKFILL_URL",
    "QBI_SIMULATION_ARCHIVED_DERIVATION_URL",
    "QBI_SIMULATION_ARCHIVED_IMPLEMENTATION_URL",
    "QBI_SIMULATION_ASSUMPTIONS_RESOURCE",
    "QBI_SIMULATION_ASSUMPTIONS_RESOURCES",
    "QBI_SIMULATION_SOURCE_NAMES",
    "QBI_SIMULATION_SUPPORTED_VERSIONS",
    "QBI_SIMULATION_VERSION",
    "QBI_SIMULATION_V2",
    "AggregateEvidenceAnchor",
    "AgiSstbPriorBand",
    "BetaParameters",
    "ExposureBetaParameters",
    "QualificationDerivation",
    "QbiSimulationAssumptions",
    "QbiSimulationAssumptionsV2",
    "QbiSimulationInputs",
    "SstbClassificationAssumptions",
    "SstbCrosswalk",
    "SstbCrosswalkEntry",
    "load_qbi_simulation_assumptions",
    "load_sstb_crosswalk",
    "parse_qbi_simulation_assumptions",
    "parse_sstb_crosswalk",
    "qbi_qrf_excluded_targets",
    "qbi_simulation_summary",
    "resolve_sstb_crosswalk",
    "simulate_qbi_inputs",
    "us_qbi_simulation_stage_spec",
    "with_qbi_simulation_from_puf_arrays",
]

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/policyengine-"
    + "us-data/blob/"
    + _ARCHIVED_COMMIT
    + "/policyengine_"
    + "us_data/"
)
QBI_SIMULATION_ARCHIVED_IMPLEMENTATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L105-L405"
)
QBI_SIMULATION_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/puf/puf.py#L748-L787"
)
QBI_SIMULATION_ARCHIVED_BACKFILL_URL = _ARCHIVED_ROOT + "datasets/puf/puf.py#L993-L1302"
QBI_SIMULATION_ARCHIVED_ASSUMPTIONS_URL = (
    _ARCHIVED_ROOT + "datasets/puf/qbi_assumptions.yaml#L1-L118"
)

QBI_SIMULATION_VERSION = 1
QBI_SIMULATION_V2 = 2
QBI_SIMULATION_SUPPORTED_VERSIONS = (
    QBI_SIMULATION_VERSION,
    QBI_SIMULATION_V2,
)
QBI_SIMULATION_ASSUMPTIONS_RESOURCE = "qbi_assumptions_v1.json"
QBI_SIMULATION_ASSUMPTIONS_RESOURCES = {
    QBI_SIMULATION_VERSION: QBI_SIMULATION_ASSUMPTIONS_RESOURCE,
    QBI_SIMULATION_V2: "qbi_assumptions_v2.json",
}
QBI_SIMULATION_SOURCE_NAMES: tuple[str, ...] = (
    "self_employment_income",
    "farm_operations_income",
    "farm_rent_income",
    "rental_income",
    "estate_income",
    "partnership_s_corp_income",
)
_QUALIFICATION_FLAG_BY_SOURCE = {
    source: f"{source}_would_be_qualified" for source in QBI_SIMULATION_SOURCE_NAMES
}
_SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG = (
    "sstb_self_employment_income_would_be_qualified"
)
_SSTB_SELF_EMPLOYMENT_OUTPUT = "sstb_self_employment_income_before_lsr"
_NON_SSTB_SELF_EMPLOYMENT_SOURCE = "self_employment_income"
_SUPPORTED_MODEL_KINDS = {
    "qualification": "flat_source_bernoulli",
    "sstb": "largest_positive_qualified_source_bernoulli",
    "w2": "source_weighted_margin_receipts_logit_labor_share",
    "ubia": "source_weighted_capital_bernoulli_lognormal",
    "investment": "exposure_bernoulli_beta_carveout",
}
_V2_QUALIFICATION_MODES = frozenset({"derived", "prior"})
_V2_SSTB_CLASSIFICATION_MODE = "crosswalk"
_SSTB_CROSSWALK_SCHEMA_VERSION = 1
_SSTB_CROSSWALK_LIVE_STATUS = "live"
_SSTB_CLASSIFICATIONS = frozenset(
    {
        "clear_sstb",
        "non_sstb",
        "ambiguous",
    }
)


@dataclass(frozen=True)
class BetaParameters:
    """Parameters for a shifted and scaled beta draw."""

    beta_a: float
    beta_b: float
    scale: float = 1.0
    shift: float = 0.0


@dataclass(frozen=True)
class ExposureBetaParameters:
    """Bernoulli receipt probability plus an exposure-share beta draw."""

    source: str
    probability_of_receiving: float
    beta: BetaParameters


@dataclass(frozen=True)
class AggregateEvidenceAnchor:
    """Published aggregate and provisional replay-diagnostic contract."""

    provisional: bool
    published_income_dollars: float | None
    published_component_dollars: float | None
    comparison_component_2022_dollars: float | None
    replay_factor_band: tuple[float, float] | None
    rationale: str

    def validate(self, label: str) -> None:
        """Reject incomplete or internally inconsistent evidence metadata."""

        if self.provisional is not True:
            raise ValueError(f"{label} aggregate anchor must be provisional.")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError(f"{label} aggregate anchor rationale must be nonempty.")
        for name, value in (
            ("published_income_dollars", self.published_income_dollars),
            ("published_component_dollars", self.published_component_dollars),
            (
                "comparison_component_2022_dollars",
                self.comparison_component_2022_dollars,
            ),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"{label} aggregate anchor {name} must be null or nonnegative."
                )
        if self.replay_factor_band is None:
            if self.published_income_dollars is not None:
                raise ValueError(
                    f"{label} published income anchor requires a replay factor band."
                )
            return
        lower, upper = self.replay_factor_band
        if not 0.0 < lower <= upper:
            raise ValueError(
                f"{label} replay factor band must be positive and ordered."
            )
        if self.published_income_dollars is None:
            raise ValueError(
                f"{label} replay factor band requires a published income anchor."
            )


@dataclass(frozen=True)
class QualificationDerivation:
    """One source's deterministic rule or documented residual prior."""

    source: str
    mode: str
    prior_probability: float | None
    rationale: str


@dataclass(frozen=True)
class AgiSstbPriorBand:
    """One lower-inclusive, upper-exclusive passive SSTB prior band."""

    label: str
    lower: float
    upper: float
    probability: float


@dataclass(frozen=True)
class SstbClassificationAssumptions:
    """Host-record SSTB classification inputs and residual priors."""

    mode: str
    crosswalk_resource: str
    occupation_column: str
    industry_column: str | None
    agi_column: str
    ambiguous_prior: float
    ambiguous_prior_status: str
    agi_band_format: str
    passive_passthrough_sstb_prior_by_agi: tuple[AgiSstbPriorBand, ...]
    passive_passthrough_prior_status: str
    rationale: str
    follow_up: str


@dataclass(frozen=True)
class SstbCrosswalkEntry:
    """One validated Census code and its Section 199A SSTB probability."""

    code: int
    classification: str
    probability: float
    provisional: bool
    basis: str | None

    def validate(self, family: str) -> None:
        """Validate classification-to-probability and evidence metadata."""

        if isinstance(self.code, bool) or not isinstance(self.code, int):
            raise ValueError(f"SSTB crosswalk {family} codes must be integers.")
        if not 0 <= self.code <= 9_999:
            raise ValueError(
                f"SSTB crosswalk {family} codes must be four-digit Census codes."
            )
        if self.classification not in _SSTB_CLASSIFICATIONS:
            raise ValueError(
                f"SSTB crosswalk {family} code {self.code:04d} has unknown "
                f"classification {self.classification!r}."
            )
        _validate_probabilities(
            f"SSTB crosswalk {family} probability for {self.code:04d}",
            (self.probability,),
        )
        deterministic_probability = {
            "clear_sstb": 1.0,
            "non_sstb": 0.0,
        }.get(self.classification)
        if (
            deterministic_probability is not None
            and self.probability != deterministic_probability
        ):
            raise ValueError(
                f"SSTB crosswalk {self.classification} code {self.code:04d} "
                f"must have probability {deterministic_probability}."
            )
        if self.classification == "ambiguous":
            if not 0.0 < self.probability < 1.0:
                raise ValueError(
                    f"Ambiguous SSTB code {self.code:04d} must have a prior "
                    "strictly between zero and one."
                )
            if self.provisional is not True:
                raise ValueError(
                    f"Ambiguous SSTB code {self.code:04d} must be provisional."
                )
            if not isinstance(self.basis, str) or not self.basis.strip():
                raise ValueError(
                    f"Ambiguous SSTB code {self.code:04d} must cite its basis."
                )
        elif self.provisional or self.basis is not None:
            raise ValueError(
                f"Deterministic SSTB code {self.code:04d} must not carry "
                "provisional prior metadata."
            )


@dataclass(frozen=True)
class SstbCrosswalk:
    """Validated Census host-code probability crosswalk."""

    schema_version: int
    crosswalk_version: str
    status: str
    industry_vintage: str
    occupation_vintage: str
    legal_basis: str
    wiring_notes: tuple[str, ...]
    sstb_category_values: tuple[str, ...]
    occupation_entries: tuple[SstbCrosswalkEntry, ...]
    industry_entries: tuple[SstbCrosswalkEntry, ...]

    def entries_for(self, family: str) -> tuple[SstbCrosswalkEntry, ...]:
        """Return one signal family's validated entries."""

        if family == "occupation":
            return self.occupation_entries
        if family == "industry":
            return self.industry_entries
        raise ValueError(f"Unknown SSTB crosswalk family {family!r}.")

    def mapping_for(self, family: str) -> dict[int, float]:
        """Return one signal family's code-to-probability mapping."""

        return {entry.code: entry.probability for entry in self.entries_for(family)}

    def validate(self) -> None:
        """Reject malformed instances, including caller-constructed objects."""

        if self.schema_version != _SSTB_CROSSWALK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SSTB crosswalk schema_version {self.schema_version!r}."
            )
        if self.status != _SSTB_CROSSWALK_LIVE_STATUS:
            raise ValueError("QBI v2 requires a live SSTB crosswalk.")
        for name, value in (
            ("crosswalk_version", self.crosswalk_version),
            ("industry_vintage", self.industry_vintage),
            ("occupation_vintage", self.occupation_vintage),
            ("legal_basis", self.legal_basis),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"SSTB crosswalk {name} must be a nonempty string.")
        if not self.wiring_notes or not all(
            isinstance(note, str) and note.strip() for note in self.wiring_notes
        ):
            raise ValueError("SSTB crosswalk wiring_notes must be nonempty strings.")
        if not self.sstb_category_values or not all(
            isinstance(category, str) and category
            for category in self.sstb_category_values
        ):
            raise ValueError(
                "SSTB crosswalk sstb_category_values must be nonempty strings."
            )
        for family, entries in (
            ("occupation", self.occupation_entries),
            ("industry", self.industry_entries),
        ):
            seen: set[int] = set()
            for entry in entries:
                entry.validate(family)
                if entry.code in seen:
                    raise ValueError(
                        f"SSTB crosswalk {family} mapping contains duplicate "
                        f"code {entry.code:04d}."
                    )
                seen.add(entry.code)
        if not self.occupation_entries or not self.industry_entries:
            raise ValueError("Live SSTB crosswalk must carry both Census maps.")


@dataclass(frozen=True)
class QbiSimulationAssumptions:
    """Immutable, order-explicit assumptions consumed by the simulation core."""

    schema_version: int
    qbi_simulation_version: int
    engine: str
    bit_generator: str
    qualification_seed: int
    w2_ubia_seed: int
    investment_seed: int
    sstb_seed: int
    source_order: tuple[str, ...]
    qualification_model: str
    qualification_probabilities: tuple[float, ...]
    sstb_model: str
    sstb_source_order: tuple[str, ...]
    sstb_probabilities: tuple[float, ...]
    w2_model: str
    profit_margin_parameters: tuple[BetaParameters, ...]
    has_employees_slope_per_dollar: float
    has_employees_target_share: float
    intercept_bisection_iterations: int
    labor_ratio_parameters: tuple[BetaParameters, ...]
    ubia_model: str
    ubia_sigma: float
    ubia_multiples: tuple[float, ...]
    capital_intensity_probabilities: tuple[float, ...]
    investment_model: str
    reit_ptp_exposures: tuple[ExposureBetaParameters, ...]
    bdc_exposures: tuple[ExposureBetaParameters, ...]

    def validate(self) -> None:
        """Reject malformed assumptions before consuming any random stream."""

        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported QBI assumptions schema_version {self.schema_version!r}."
            )
        if self.qbi_simulation_version != QBI_SIMULATION_VERSION:
            raise ValueError(
                "QBI assumptions carry unsupported qbi_simulation_version "
                f"{self.qbi_simulation_version!r}."
            )
        if self.source_order != QBI_SIMULATION_SOURCE_NAMES:
            raise ValueError(
                "QBI v1 source_order must preserve the archived random-stream "
                f"order {QBI_SIMULATION_SOURCE_NAMES!r}, got {self.source_order!r}."
            )
        models = {
            "qualification": self.qualification_model,
            "sstb": self.sstb_model,
            "w2": self.w2_model,
            "ubia": self.ubia_model,
            "investment": self.investment_model,
        }
        for family, expected in _SUPPORTED_MODEL_KINDS.items():
            if models[family] != expected:
                raise ValueError(
                    f"QBI v1 {family} model must be {expected!r}, "
                    f"got {models[family]!r}."
                )
        if self.bit_generator != "PCG64":
            raise ValueError(
                "QBI v1 requires NumPy PCG64 to reproduce the archived streams."
            )
        source_count = len(self.source_order)
        ordered_parameter_families = {
            "qualification_probabilities": self.qualification_probabilities,
            "profit_margin_parameters": self.profit_margin_parameters,
            "labor_ratio_parameters": self.labor_ratio_parameters,
            "ubia_multiples": self.ubia_multiples,
            "capital_intensity_probabilities": (self.capital_intensity_probabilities),
        }
        for name, values in ordered_parameter_families.items():
            if len(values) != source_count:
                raise ValueError(
                    f"QBI assumptions {name} has {len(values)} value(s); "
                    f"expected {source_count}."
                )
        if len(self.sstb_source_order) != len(self.sstb_probabilities):
            raise ValueError("QBI SSTB sources and probabilities must align.")
        if not set(self.sstb_source_order) <= set(self.source_order):
            raise ValueError("QBI SSTB sources must be modeled QBI sources.")
        _validate_probabilities(
            "qualification probabilities", self.qualification_probabilities
        )
        _validate_probabilities("SSTB probabilities", self.sstb_probabilities)
        _validate_probabilities(
            "capital-intensity probabilities",
            self.capital_intensity_probabilities,
        )
        if not 0.0 < self.has_employees_target_share < 1.0:
            raise ValueError("QBI employee target share must lie strictly in (0, 1).")
        if self.has_employees_slope_per_dollar < 0.0:
            raise ValueError("QBI employee-logit slope must be nonnegative.")
        if self.intercept_bisection_iterations <= 0:
            raise ValueError("QBI logit calibration requires positive iterations.")
        if self.ubia_sigma < 0.0:
            raise ValueError("QBI UBIA lognormal sigma must be nonnegative.")
        if any(value < 0.0 for value in self.ubia_multiples):
            raise ValueError("QBI UBIA multiples must be nonnegative.")
        for name, parameters in (
            ("profit margin", self.profit_margin_parameters),
            ("labor ratio", self.labor_ratio_parameters),
        ):
            for parameter in parameters:
                _validate_beta_parameters(name, parameter)
        for name, exposures in (
            ("REIT/PTP", self.reit_ptp_exposures),
            ("BDC", self.bdc_exposures),
        ):
            for exposure in exposures:
                if not 0.0 <= exposure.probability_of_receiving <= 1.0:
                    raise ValueError(
                        f"QBI {name} receipt probability for {exposure.source!r} "
                        "must lie in [0, 1]."
                    )
                _validate_beta_parameters(name, exposure.beta)


@dataclass(frozen=True)
class QbiSimulationAssumptionsV2:
    """Strict v2 assumptions with host SSTB and per-family random streams."""

    schema_version: int
    qbi_simulation_version: int
    engine: str
    bit_generator: str
    qualification_seed: int
    sstb_seed: int
    w2_seed: int
    ubia_seed: int
    investment_seed: int
    source_order: tuple[str, ...]
    qualification_derivations: tuple[QualificationDerivation, ...]
    sstb_classification: SstbClassificationAssumptions
    w2_model: str
    profit_margin_parameters: tuple[BetaParameters, ...]
    has_employees_slope_per_dollar: float
    has_employees_target_share: float
    intercept_bisection_iterations: int
    labor_ratio_parameters: tuple[BetaParameters, ...]
    ubia_model: str
    ubia_sigma: float
    ubia_multiples: tuple[float, ...]
    capital_intensity_probabilities: tuple[float, ...]
    investment_model: str
    reit_ptp_anchor: AggregateEvidenceAnchor
    bdc_anchor: AggregateEvidenceAnchor
    reit_ptp_exposures: tuple[ExposureBetaParameters, ...]
    bdc_exposures: tuple[ExposureBetaParameters, ...]

    @property
    def qualification_by_source(self) -> dict[str, QualificationDerivation]:
        """Return the source-indexed qualification contract."""

        return {
            derivation.source: derivation
            for derivation in self.qualification_derivations
        }

    def validate(self) -> None:
        """Reject malformed v2 assumptions before consuming any family stream."""

        if self.schema_version != 2:
            raise ValueError(
                f"Unsupported QBI v2 assumptions schema_version "
                f"{self.schema_version!r}."
            )
        if self.qbi_simulation_version != QBI_SIMULATION_V2:
            raise ValueError(
                "QBI v2 assumptions carry unsupported qbi_simulation_version "
                f"{self.qbi_simulation_version!r}."
            )
        if self.source_order != QBI_SIMULATION_SOURCE_NAMES:
            raise ValueError(
                "QBI v2 source_order must preserve the engine order "
                f"{QBI_SIMULATION_SOURCE_NAMES!r}, got {self.source_order!r}."
            )
        if self.bit_generator != "PCG64":
            raise ValueError("QBI v2 requires NumPy PCG64 family streams.")
        family_seeds = {
            "qualification": self.qualification_seed,
            "sstb": self.sstb_seed,
            "w2": self.w2_seed,
            "ubia": self.ubia_seed,
            "investment": self.investment_seed,
        }
        if any(seed < 0 for seed in family_seeds.values()):
            raise ValueError("QBI v2 family seeds must be nonnegative integers.")
        if len(set(family_seeds.values())) != len(family_seeds):
            raise ValueError("QBI v2 family seeds must be distinct.")
        derivation_sources = tuple(
            derivation.source for derivation in self.qualification_derivations
        )
        if derivation_sources != self.source_order:
            raise ValueError(
                "QBI v2 qualification derivations must follow source_order."
            )
        for derivation in self.qualification_derivations:
            if derivation.mode not in _V2_QUALIFICATION_MODES:
                raise ValueError(
                    "Unknown QBI v2 qualification mode "
                    f"{derivation.mode!r} for {derivation.source!r}."
                )
            if not derivation.rationale.strip():
                raise ValueError(
                    f"QBI v2 qualification rationale for "
                    f"{derivation.source!r} must be nonempty."
                )
            if derivation.mode == "derived":
                if derivation.prior_probability is not None:
                    raise ValueError(
                        f"Derived QBI source {derivation.source!r} must not "
                        "declare a prior."
                    )
            elif derivation.prior_probability is None:
                raise ValueError(
                    f"Prior-mode QBI source {derivation.source!r} must declare "
                    "a probability."
                )
            else:
                _validate_probabilities(
                    f"qualification prior for {derivation.source!r}",
                    (derivation.prior_probability,),
                )

        classification = self.sstb_classification
        if classification.mode != _V2_SSTB_CLASSIFICATION_MODE:
            raise ValueError(
                f"Unknown QBI v2 SSTB classification mode {classification.mode!r}."
            )
        if not classification.crosswalk_resource.endswith(".json"):
            raise ValueError("QBI v2 SSTB crosswalk_resource must name a JSON file.")
        if "/" in classification.crosswalk_resource:
            raise ValueError(
                "QBI v2 SSTB crosswalk_resource must be a package basename."
            )
        if classification.industry_column == classification.occupation_column:
            raise ValueError(
                "QBI v2 SSTB industry and occupation columns must be distinct."
            )
        _validate_probabilities(
            "ambiguous SSTB prior",
            (classification.ambiguous_prior,),
        )
        if classification.agi_band_format != ("lower_inclusive:upper_exclusive"):
            raise ValueError("Unsupported QBI v2 AGI band format.")
        _validate_agi_prior_bands(classification.passive_passthrough_sstb_prior_by_agi)

        models = {
            "w2": self.w2_model,
            "ubia": self.ubia_model,
            "investment": self.investment_model,
        }
        for family, expected in (
            ("w2", _SUPPORTED_MODEL_KINDS["w2"]),
            ("ubia", _SUPPORTED_MODEL_KINDS["ubia"]),
            ("investment", _SUPPORTED_MODEL_KINDS["investment"]),
        ):
            if models[family] != expected:
                raise ValueError(
                    f"QBI v2 {family} model must be {expected!r}, "
                    f"got {models[family]!r}."
                )
        source_count = len(self.source_order)
        ordered_parameter_families = {
            "profit_margin_parameters": self.profit_margin_parameters,
            "labor_ratio_parameters": self.labor_ratio_parameters,
            "ubia_multiples": self.ubia_multiples,
            "capital_intensity_probabilities": (self.capital_intensity_probabilities),
        }
        for name, values in ordered_parameter_families.items():
            if len(values) != source_count:
                raise ValueError(
                    f"QBI v2 assumptions {name} has {len(values)} value(s); "
                    f"expected {source_count}."
                )
        _validate_probabilities(
            "capital-intensity probabilities",
            self.capital_intensity_probabilities,
        )
        if not 0.0 < self.has_employees_target_share < 1.0:
            raise ValueError("QBI employee target share must lie strictly in (0, 1).")
        if self.has_employees_slope_per_dollar < 0.0:
            raise ValueError("QBI employee-logit slope must be nonnegative.")
        if self.intercept_bisection_iterations <= 0:
            raise ValueError("QBI logit calibration requires positive iterations.")
        if self.ubia_sigma < 0.0:
            raise ValueError("QBI UBIA lognormal sigma must be nonnegative.")
        if any(value < 0.0 for value in self.ubia_multiples):
            raise ValueError("QBI UBIA multiples must be nonnegative.")
        for name, parameters in (
            ("profit margin", self.profit_margin_parameters),
            ("labor ratio", self.labor_ratio_parameters),
        ):
            for parameter in parameters:
                _validate_beta_parameters(name, parameter)
        for name, exposures in (
            ("REIT/PTP", self.reit_ptp_exposures),
            ("BDC", self.bdc_exposures),
        ):
            for exposure in exposures:
                if not 0.0 <= exposure.probability_of_receiving <= 1.0:
                    raise ValueError(
                        f"QBI {name} receipt probability for {exposure.source!r} "
                        "must lie in [0, 1]."
                    )
                _validate_beta_parameters(name, exposure.beta)
        self.reit_ptp_anchor.validate("REIT/PTP")
        self.bdc_anchor.validate("BDC")


@dataclass(frozen=True)
class QbiSimulationInputs:
    """Normalized one-dimensional sources required by every model version."""

    self_employment_income: np.ndarray
    farm_operations_income: np.ndarray
    farm_rent_income: np.ndarray
    rental_income: np.ndarray
    estate_income: np.ndarray
    partnership_s_corp_income: np.ndarray
    non_qualified_dividend_income: np.ndarray

    def __post_init__(self) -> None:
        lengths: set[int] = set()
        for field in fields(self):
            values = np.asarray(getattr(self, field.name), dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"QBI input {field.name!r} must be one-dimensional.")
            nonfinite = int(np.count_nonzero(~np.isfinite(values)))
            if nonfinite:
                raise ValueError(
                    f"QBI input {field.name!r} has {nonfinite} nonfinite value(s)."
                )
            object.__setattr__(self, field.name, values)
            lengths.add(len(values))
        if len(lengths) != 1:
            raise ValueError(
                f"QBI input arrays must have one common length, got {sorted(lengths)}."
            )

    def source(self, name: str) -> np.ndarray:
        """Return a named modeled source."""

        if name not in QBI_SIMULATION_SOURCE_NAMES:
            raise KeyError(f"Unknown QBI source {name!r}.")
        return getattr(self, name)

    @property
    def n(self) -> int:
        """Return the common row count."""

        return len(self.self_employment_income)

    @classmethod
    def from_puf_arrays(
        cls,
        arrays: Mapping[str, Sequence[Any]],
    ) -> QbiSimulationInputs:
        """Normalize raw-PUF or legacy processed-PUF arrays.

        Missing modeled sources become zero arrays, matching the archived
        load-time migration. In particular, release 1.8.0 has no
        ``farm_operations_income`` array, but its absent source still consumes
        the full v1 qualification, margin, and labor random draws.
        """

        n = _infer_person_length(arrays)
        self_employment_key = _first_present(
            arrays,
            (
                "self_employment_income",
                "self_employment_income_before_lsr",
                "E00900",
            ),
        )
        self_employment_income = _array_or_zeros(arrays, self_employment_key, n=n)
        if self_employment_key in {
            "self_employment_income",
            "self_employment_income_before_lsr",
        }:
            sstb_key = _first_present(
                arrays,
                (
                    "sstb_self_employment_income",
                    _SSTB_SELF_EMPLOYMENT_OUTPUT,
                ),
            )
            if sstb_key is not None:
                self_employment_income = self_employment_income + _array_or_zeros(
                    arrays, sstb_key, n=n
                )

        farm_operations_income = _first_array(
            arrays, ("farm_operations_income", "E02100"), n=n
        )
        farm_rent_income = _first_array(arrays, ("farm_rent_income", "E27200"), n=n)
        rental_income = _first_array(arrays, ("rental_income",), n=n)
        if rental_income is None:
            rental_income = _difference_or_zeros(arrays, "E25850", "E25860", n=n)
        estate_income = _first_array(arrays, ("estate_income",), n=n)
        if estate_income is None:
            estate_income = _difference_or_zeros(arrays, "E26390", "E26400", n=n)

        partnership_s_corp_income = _first_array(
            arrays, ("partnership_s_corp_income",), n=n
        )
        if partnership_s_corp_income is None and (
            "partnership_income" in arrays or "s_corp_income" in arrays
        ):
            partnership_s_corp_income = _array_or_zeros(
                arrays, "partnership_income", n=n
            ) + _array_or_zeros(arrays, "s_corp_income", n=n)
        if partnership_s_corp_income is None and {
            "E25980",
            "E25960",
            "E26190",
            "E26180",
        } <= set(arrays):
            partnership_s_corp_income = (
                _array_or_zeros(arrays, "E25980", n=n)
                - _array_or_zeros(arrays, "E25960", n=n)
                + _array_or_zeros(arrays, "E26190", n=n)
                - _array_or_zeros(arrays, "E26180", n=n)
            )
        if partnership_s_corp_income is None:
            partnership_s_corp_income = _first_array(arrays, ("E26270",), n=n)

        non_qualified_dividend_income = _first_array(
            arrays, ("non_qualified_dividend_income",), n=n
        )
        if non_qualified_dividend_income is None and {
            "E00600",
            "E00650",
        } <= set(arrays):
            non_qualified_dividend_income = _difference_or_zeros(
                arrays, "E00600", "E00650", n=n
            )
        if non_qualified_dividend_income is None:
            non_qualified_dividend_income = _first_array(
                arrays, ("ordinary_dividend_income",), n=n
            )

        return cls(
            self_employment_income=self_employment_income,
            farm_operations_income=_zeros_if_none(farm_operations_income, n),
            farm_rent_income=_zeros_if_none(farm_rent_income, n),
            rental_income=_zeros_if_none(rental_income, n),
            estate_income=_zeros_if_none(estate_income, n),
            partnership_s_corp_income=_zeros_if_none(partnership_s_corp_income, n),
            non_qualified_dividend_income=_zeros_if_none(
                non_qualified_dividend_income, n
            ),
        )


@lru_cache(maxsize=2)
def load_qbi_simulation_assumptions(
    qbi_simulation_version: int,
) -> QbiSimulationAssumptions | QbiSimulationAssumptionsV2:
    """Load and strictly validate one packaged QBI assumptions version."""

    if qbi_simulation_version not in QBI_SIMULATION_SUPPORTED_VERSIONS:
        raise ValueError(
            "Unsupported qbi_simulation_version "
            f"{qbi_simulation_version!r}; supported versions: "
            f"{QBI_SIMULATION_SUPPORTED_VERSIONS!r}."
        )
    resource = files("populace.build.us").joinpath(
        QBI_SIMULATION_ASSUMPTIONS_RESOURCES[qbi_simulation_version]
    )
    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return parse_qbi_simulation_assumptions(
        payload,
        qbi_simulation_version=qbi_simulation_version,
    )


def parse_qbi_simulation_assumptions(
    payload: Any,
    *,
    qbi_simulation_version: int,
) -> QbiSimulationAssumptions | QbiSimulationAssumptionsV2:
    """Parse one assumptions mapping under its requested versioned schema."""

    if qbi_simulation_version not in QBI_SIMULATION_SUPPORTED_VERSIONS:
        raise ValueError(
            "Unsupported qbi_simulation_version "
            f"{qbi_simulation_version!r}; supported versions: "
            f"{QBI_SIMULATION_SUPPORTED_VERSIONS!r}."
        )
    root = _require_mapping(payload, "QBI assumptions")
    payload_version = _integer(
        root.get("qbi_simulation_version"),
        "qbi_simulation_version",
    )
    if payload_version != qbi_simulation_version:
        raise ValueError(
            "Requested qbi_simulation_version "
            f"{qbi_simulation_version!r} does not match assumptions version "
            f"{payload_version!r}."
        )
    if qbi_simulation_version == QBI_SIMULATION_VERSION:
        return _parse_v1_qbi_simulation_assumptions(root)
    return _parse_v2_qbi_simulation_assumptions(root)


def _parse_v1_qbi_simulation_assumptions(
    root: Mapping[str, Any],
) -> QbiSimulationAssumptions:
    rng = _child_mapping(root, "rng")
    qualification = _child_mapping(root, "qualification")
    sstb = _child_mapping(root, "sstb")
    w2 = _child_mapping(root, "w2")
    employee_logit = _child_mapping(w2, "has_employees_logit")
    ubia = _child_mapping(root, "ubia")
    investment = _child_mapping(root, "investment")
    source_order = _string_tuple(root.get("source_order"), "source_order")
    sstb_source_order = _string_tuple(sstb.get("source_order"), "sstb.source_order")
    reit_order = _string_tuple(
        investment.get("reit_ptp_exposure_order"),
        "investment.reit_ptp_exposure_order",
    )
    bdc_order = _string_tuple(
        investment.get("bdc_exposure_order"),
        "investment.bdc_exposure_order",
    )

    assumptions = QbiSimulationAssumptions(
        schema_version=_integer(root.get("schema_version"), "schema_version"),
        qbi_simulation_version=_integer(
            root.get("qbi_simulation_version"), "qbi_simulation_version"
        ),
        engine=_string(root.get("engine"), "engine"),
        bit_generator=_string(rng.get("bit_generator"), "rng.bit_generator"),
        qualification_seed=_integer(
            rng.get("qualification_seed"), "rng.qualification_seed"
        ),
        w2_ubia_seed=_integer(rng.get("w2_ubia_seed"), "rng.w2_ubia_seed"),
        investment_seed=_integer(rng.get("investment_seed"), "rng.investment_seed"),
        sstb_seed=_integer(rng.get("sstb_seed"), "rng.sstb_seed"),
        source_order=source_order,
        qualification_model=_string(qualification.get("model"), "qualification.model"),
        qualification_probabilities=_ordered_scalars(
            _child_mapping(qualification, "probabilities"),
            source_order,
            "qualification.probabilities",
        ),
        sstb_model=_string(sstb.get("model"), "sstb.model"),
        sstb_source_order=sstb_source_order,
        sstb_probabilities=_ordered_scalars(
            _child_mapping(sstb, "probabilities"),
            sstb_source_order,
            "sstb.probabilities",
        ),
        w2_model=_string(w2.get("model"), "w2.model"),
        profit_margin_parameters=_ordered_betas(
            _child_mapping(w2, "profit_margin_distribution"),
            source_order,
            "w2.profit_margin_distribution",
        ),
        has_employees_slope_per_dollar=_number(
            employee_logit.get("slope_per_dollar"),
            "w2.has_employees_logit.slope_per_dollar",
        ),
        has_employees_target_share=_number(
            employee_logit.get("target_share_among_positive_receipts"),
            "w2.has_employees_logit.target_share_among_positive_receipts",
        ),
        intercept_bisection_iterations=_integer(
            employee_logit.get("intercept_bisection_iterations"),
            "w2.has_employees_logit.intercept_bisection_iterations",
        ),
        labor_ratio_parameters=_ordered_betas(
            _child_mapping(w2, "labor_ratio_distribution"),
            source_order,
            "w2.labor_ratio_distribution",
        ),
        ubia_model=_string(ubia.get("model"), "ubia.model"),
        ubia_sigma=_number(ubia.get("sigma"), "ubia.sigma"),
        ubia_multiples=_ordered_scalars(
            _child_mapping(ubia, "multiple_of_qbi"),
            source_order,
            "ubia.multiple_of_qbi",
        ),
        capital_intensity_probabilities=_ordered_scalars(
            _child_mapping(ubia, "capital_intensity_probabilities"),
            source_order,
            "ubia.capital_intensity_probabilities",
        ),
        investment_model=_string(investment.get("model"), "investment.model"),
        reit_ptp_exposures=_ordered_exposures(
            _child_mapping(investment, "reit_ptp_income_distribution"),
            reit_order,
            "investment.reit_ptp_income_distribution",
        ),
        bdc_exposures=_ordered_exposures(
            _child_mapping(investment, "bdc_income_distribution"),
            bdc_order,
            "investment.bdc_income_distribution",
        ),
    )
    assumptions.validate()
    return assumptions


def _parse_v2_qbi_simulation_assumptions(
    root: Mapping[str, Any],
) -> QbiSimulationAssumptionsV2:
    _validate_v2_payload_keys(root)
    rng = _child_mapping(root, "rng")
    seeds = _child_mapping(rng, "seeds")
    derivations = _child_mapping(root, "qualification_derivations")
    classification = _child_mapping(root, "sstb_classification")
    w2 = _child_mapping(root, "w2")
    employee_logit = _child_mapping(w2, "has_employees_logit")
    ubia = _child_mapping(root, "ubia")
    investment = _child_mapping(root, "investment")
    source_order = _string_tuple(root.get("source_order"), "source_order")
    reit_order = _string_tuple(
        investment.get("reit_ptp_exposure_order"),
        "investment.reit_ptp_exposure_order",
    )
    bdc_order = _string_tuple(
        investment.get("bdc_exposure_order"),
        "investment.bdc_exposure_order",
    )

    qualification_derivations: list[QualificationDerivation] = []
    for source in source_order:
        entry = _require_mapping(
            derivations[source],
            f"qualification_derivations.{source}",
        )
        prior_payload = entry.get("prior")
        prior_probability: float | None = None
        if prior_payload is not None:
            prior = _require_mapping(
                prior_payload,
                f"qualification_derivations.{source}.prior",
            )
            prior_probability = _number(
                prior.get("probability"),
                f"qualification_derivations.{source}.prior.probability",
            )
        qualification_derivations.append(
            QualificationDerivation(
                source=source,
                mode=_string(
                    entry.get("mode"),
                    f"qualification_derivations.{source}.mode",
                ),
                prior_probability=prior_probability,
                rationale=_string(
                    entry.get("rationale"),
                    f"qualification_derivations.{source}.rationale",
                ),
            )
        )

    assumptions = QbiSimulationAssumptionsV2(
        schema_version=_integer(root.get("schema_version"), "schema_version"),
        qbi_simulation_version=_integer(
            root.get("qbi_simulation_version"),
            "qbi_simulation_version",
        ),
        engine=_string(root.get("engine"), "engine"),
        bit_generator=_string(rng.get("bit_generator"), "rng.bit_generator"),
        qualification_seed=_integer(
            seeds.get("qualification"),
            "rng.seeds.qualification",
        ),
        sstb_seed=_integer(seeds.get("sstb"), "rng.seeds.sstb"),
        w2_seed=_integer(seeds.get("w2"), "rng.seeds.w2"),
        ubia_seed=_integer(seeds.get("ubia"), "rng.seeds.ubia"),
        investment_seed=_integer(
            seeds.get("investment"),
            "rng.seeds.investment",
        ),
        source_order=source_order,
        qualification_derivations=tuple(qualification_derivations),
        sstb_classification=SstbClassificationAssumptions(
            mode=_string(
                classification.get("mode"),
                "sstb_classification.mode",
            ),
            crosswalk_resource=_string(
                classification.get("crosswalk_resource"),
                "sstb_classification.crosswalk_resource",
            ),
            occupation_column=_string(
                classification.get("occupation_column"),
                "sstb_classification.occupation_column",
            ),
            industry_column=_optional_string(
                classification.get("industry_column"),
                "sstb_classification.industry_column",
            ),
            agi_column=_string(
                classification.get("agi_column"),
                "sstb_classification.agi_column",
            ),
            ambiguous_prior=_number(
                classification.get("ambiguous_prior"),
                "sstb_classification.ambiguous_prior",
            ),
            ambiguous_prior_status=_string(
                classification.get("ambiguous_prior_status"),
                "sstb_classification.ambiguous_prior_status",
            ),
            agi_band_format=_string(
                classification.get("agi_band_format"),
                "sstb_classification.agi_band_format",
            ),
            passive_passthrough_sstb_prior_by_agi=_parse_agi_prior_bands(
                _child_mapping(
                    classification,
                    "passive_passthrough_sstb_prior_by_agi",
                )
            ),
            passive_passthrough_prior_status=_string(
                classification.get("passive_passthrough_prior_status"),
                "sstb_classification.passive_passthrough_prior_status",
            ),
            rationale=_string(
                classification.get("rationale"),
                "sstb_classification.rationale",
            ),
            follow_up=_string(
                classification.get("follow_up"),
                "sstb_classification.follow_up",
            ),
        ),
        w2_model=_string(w2.get("model"), "w2.model"),
        profit_margin_parameters=_ordered_betas(
            _child_mapping(w2, "profit_margin_distribution"),
            source_order,
            "w2.profit_margin_distribution",
        ),
        has_employees_slope_per_dollar=_number(
            employee_logit.get("slope_per_dollar"),
            "w2.has_employees_logit.slope_per_dollar",
        ),
        has_employees_target_share=_number(
            employee_logit.get("target_share_among_positive_receipts"),
            "w2.has_employees_logit.target_share_among_positive_receipts",
        ),
        intercept_bisection_iterations=_integer(
            employee_logit.get("intercept_bisection_iterations"),
            "w2.has_employees_logit.intercept_bisection_iterations",
        ),
        labor_ratio_parameters=_ordered_betas(
            _child_mapping(w2, "labor_ratio_distribution"),
            source_order,
            "w2.labor_ratio_distribution",
        ),
        ubia_model=_string(ubia.get("model"), "ubia.model"),
        ubia_sigma=_number(ubia.get("sigma"), "ubia.sigma"),
        ubia_multiples=_ordered_scalars(
            _child_mapping(ubia, "multiple_of_qbi"),
            source_order,
            "ubia.multiple_of_qbi",
        ),
        capital_intensity_probabilities=_ordered_scalars(
            _child_mapping(ubia, "capital_intensity_probabilities"),
            source_order,
            "ubia.capital_intensity_probabilities",
        ),
        investment_model=_string(investment.get("model"), "investment.model"),
        reit_ptp_anchor=_parse_aggregate_evidence_anchor(
            _child_mapping(investment, "reit_ptp_anchor"),
            "investment.reit_ptp_anchor",
        ),
        bdc_anchor=_parse_aggregate_evidence_anchor(
            _child_mapping(investment, "bdc_anchor"),
            "investment.bdc_anchor",
        ),
        reit_ptp_exposures=_ordered_exposures(
            _child_mapping(investment, "reit_ptp_income_distribution"),
            reit_order,
            "investment.reit_ptp_income_distribution",
        ),
        bdc_exposures=_ordered_exposures(
            _child_mapping(investment, "bdc_income_distribution"),
            bdc_order,
            "investment.bdc_income_distribution",
        ),
    )
    assumptions.validate()
    return assumptions


def load_sstb_crosswalk(resource_name: str) -> SstbCrosswalk:
    """Load one packaged SSTB crosswalk and reject placeholder content."""

    if not resource_name.endswith(".json") or "/" in resource_name:
        raise ValueError("SSTB crosswalk resource must be a package JSON basename.")
    resource = files("populace.build.us").joinpath(resource_name)
    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return parse_sstb_crosswalk(payload)


def parse_sstb_crosswalk(payload: Any) -> SstbCrosswalk:
    """Validate the live SSTB crosswalk, failing closed on placeholders."""

    root = _require_mapping(payload, "SSTB crosswalk")
    status = _string(root.get("status"), "SSTB crosswalk.status")
    if status == "placeholder":
        raise ValueError(
            "SSTB crosswalk status is 'placeholder'; v2 classification fails "
            "closed until reviewed mapping content is packaged."
        )
    _require_exact_keys(
        root,
        (
            "schema_version",
            "crosswalk_version",
            "status",
            "meta",
            "industry_2017",
            "industry_explicit_nonsstb_neighbors",
            "occupation_2018",
            "occupation_explicit_nonsstb_notes",
        ),
        "SSTB crosswalk",
    )
    schema_version = _integer(
        root.get("schema_version"),
        "SSTB crosswalk.schema_version",
    )
    if schema_version != _SSTB_CROSSWALK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SSTB crosswalk schema_version {schema_version!r}."
        )
    if status != _SSTB_CROSSWALK_LIVE_STATUS:
        raise ValueError(f"Unsupported SSTB crosswalk status {status!r}.")
    meta = _child_mapping(root, "meta")
    _require_exact_keys(
        meta,
        (
            "industry_vintage",
            "occupation_vintage",
            "legal_basis",
            "wiring_notes",
            "sstb_category_values",
        ),
        "SSTB crosswalk.meta",
    )
    categories = _string_tuple(
        meta.get("sstb_category_values"),
        "SSTB crosswalk.meta.sstb_category_values",
    )
    _validate_explicit_nonsstb_entries(
        root.get("industry_explicit_nonsstb_neighbors"),
        "SSTB crosswalk.industry_explicit_nonsstb_neighbors",
    )
    _validate_explicit_nonsstb_entries(
        root.get("occupation_explicit_nonsstb_notes"),
        "SSTB crosswalk.occupation_explicit_nonsstb_notes",
    )
    crosswalk = SstbCrosswalk(
        schema_version=schema_version,
        crosswalk_version=_string(
            root.get("crosswalk_version"),
            "SSTB crosswalk.crosswalk_version",
        ),
        status=status,
        industry_vintage=_string(
            meta.get("industry_vintage"),
            "SSTB crosswalk.meta.industry_vintage",
        ),
        occupation_vintage=_string(
            meta.get("occupation_vintage"),
            "SSTB crosswalk.meta.occupation_vintage",
        ),
        legal_basis=_string(
            meta.get("legal_basis"),
            "SSTB crosswalk.meta.legal_basis",
        ),
        wiring_notes=_string_tuple(
            meta.get("wiring_notes"),
            "SSTB crosswalk.meta.wiring_notes",
        ),
        sstb_category_values=categories,
        occupation_entries=_parse_crosswalk_entries(
            root.get("occupation_2018"),
            family="occupation",
            classification_system_key="soc",
            categories=categories,
        ),
        industry_entries=_parse_crosswalk_entries(
            root.get("industry_2017"),
            family="industry",
            classification_system_key="naics",
            categories=categories,
        ),
    )
    crosswalk.validate()
    return crosswalk


def us_qbi_simulation_stage_spec() -> SourceStageSpec:
    """Load and validate the manifest contract for the QBI source stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()["puf_tax_detail"]
    missing = sorted(set(US_QBI_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            "puf_tax_detail manifest does not declare QBI simulation output(s) "
            f"{missing}."
        )
    version_is_pinned = any(
        artifact.get("populace_qbi_simulation_version") == QBI_SIMULATION_VERSION
        for artifact in spec.artifacts
    )
    if not version_is_pinned:
        raise ValueError(
            "puf_tax_detail artifact declaration must pin structured "
            "populace_qbi_simulation_version="
            f"{QBI_SIMULATION_VERSION}."
        )
    qrf_operations = tuple(
        operation
        for operation in spec.operations
        if operation.kind == "fit_weighted_qrf"
    )
    if len(qrf_operations) != 1:
        raise ValueError(
            "puf_tax_detail must declare exactly one fit_weighted_qrf operation."
        )
    exclusions = _require_mapping(
        qrf_operations[0].parameters.get("qbi_target_exclusions_by_simulation_version"),
        "qbi_target_exclusions_by_simulation_version",
    )
    version_keys = tuple(str(version) for version in QBI_SIMULATION_SUPPORTED_VERSIONS)
    _require_exact_keys(
        exclusions,
        version_keys,
        "qbi_target_exclusions_by_simulation_version",
    )
    for version in QBI_SIMULATION_SUPPORTED_VERSIONS:
        declared = exclusions[str(version)]
        if not isinstance(declared, list) or not all(
            isinstance(column, str) and column for column in declared
        ):
            raise ValueError("QBI target exclusions must be lists of nonempty strings.")
        expected = qbi_qrf_excluded_targets(version)
        if tuple(declared) != expected:
            raise ValueError(
                "puf_tax_detail QBI target exclusions disagree with version "
                f"{version} assumptions: expected {expected!r}, "
                f"got {tuple(declared)!r}."
            )
    return spec


def qbi_qrf_excluded_targets(
    qbi_simulation_version: int,
) -> tuple[str, ...]:
    """Return QBI leaves derived after, rather than imputed by, the QRF."""

    if qbi_simulation_version == QBI_SIMULATION_VERSION:
        return ()
    assumptions = load_qbi_simulation_assumptions(qbi_simulation_version)
    if not isinstance(assumptions, QbiSimulationAssumptionsV2):
        raise TypeError("QBI v2 target selection requires v2 assumptions.")
    excluded = {
        _QUALIFICATION_FLAG_BY_SOURCE[derivation.source]
        for derivation in assumptions.qualification_derivations
        if derivation.mode == "derived"
    }
    excluded.add(_SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG)
    return tuple(column for column in US_QBI_OUTPUT_COLUMNS if column in excluded)


def simulate_qbi_inputs(
    inputs: QbiSimulationInputs,
    *,
    assumptions: QbiSimulationAssumptions | QbiSimulationAssumptionsV2,
    qbi_simulation_version: int,
    sstb_crosswalk: SstbCrosswalk | Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run a version-gated Section 199A simulation.

    Returns exactly the 15 names in :data:`US_QBI_OUTPUT_COLUMNS`. The ordinary
    Schedule C complement is deliberately returned by the stage wrapper,
    because it replaces a source column rather than adding a QBI-contract leaf.
    """

    if qbi_simulation_version != assumptions.qbi_simulation_version:
        raise ValueError(
            "Requested qbi_simulation_version "
            f"{qbi_simulation_version!r} does not match assumptions version "
            f"{assumptions.qbi_simulation_version!r}."
        )
    if qbi_simulation_version not in QBI_SIMULATION_SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported qbi_simulation_version {qbi_simulation_version!r}."
        )
    assumptions.validate()

    if qbi_simulation_version == QBI_SIMULATION_VERSION:
        if not isinstance(assumptions, QbiSimulationAssumptions):
            raise TypeError("QBI v1 simulation requires v1 assumptions.")
        qualification_rng = _rng(
            assumptions.qualification_seed,
            assumptions.bit_generator,
        )
        qualification_flags = tuple(
            qualification_rng.random(inputs.n) < probability
            for probability in assumptions.qualification_probabilities
        )
        qualified_components = _qualified_components(
            inputs,
            source_order=assumptions.source_order,
            qualification_flags=qualification_flags,
        )
        w2_rng = _rng(assumptions.w2_ubia_seed, assumptions.bit_generator)
        w2_wages, ubia = _simulate_w2_and_ubia(
            qualified_components,
            assumptions=assumptions,
            rng=w2_rng,
        )
        flag_by_source = dict(
            zip(assumptions.source_order, qualification_flags, strict=True)
        )
        business_is_sstb = _simulate_business_is_sstb(
            inputs,
            qualification_flags=flag_by_source,
            assumptions=assumptions,
            rng=_rng(assumptions.sstb_seed, assumptions.bit_generator),
        )
    else:
        if not isinstance(assumptions, QbiSimulationAssumptionsV2):
            raise TypeError("QBI v2 simulation requires v2 assumptions.")
        resolve_sstb_crosswalk(assumptions, sstb_crosswalk)
        qualification_flags = _derive_v2_qualification_flags(
            inputs,
            assumptions=assumptions,
        )
        qualified_components = _qualified_components(
            inputs,
            source_order=assumptions.source_order,
            qualification_flags=qualification_flags,
        )
        w2_wages = _simulate_w2(
            qualified_components,
            assumptions=assumptions,
            rng=_rng(assumptions.w2_seed, assumptions.bit_generator),
        )
        ubia = _simulate_ubia(
            qualified_components,
            assumptions=assumptions,
            rng=_rng(assumptions.ubia_seed, assumptions.bit_generator),
        )
        flag_by_source = dict(
            zip(assumptions.source_order, qualification_flags, strict=True)
        )
        # The PUF donor has no host occupation or industry. V2 emits a neutral
        # preliminary route; the authoritative SSTB classifier runs after QRF
        # placement on the cloned host record.
        business_is_sstb = np.zeros(inputs.n, dtype=bool)

    qualified_reit_and_ptp_income, qualified_bdc_income = _simulate_investment_qbi(
        inputs,
        assumptions=assumptions,
        rng=_rng(assumptions.investment_seed, assumptions.bit_generator),
    )

    self_employment_qualified = flag_by_source["self_employment_income"]
    results: dict[str, np.ndarray] = {
        _QUALIFICATION_FLAG_BY_SOURCE[source]: qualified
        for source, qualified in flag_by_source.items()
    }
    results["self_employment_income_would_be_qualified"] = np.where(
        business_is_sstb,
        False,
        self_employment_qualified,
    )
    results[_SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG] = np.where(
        business_is_sstb,
        self_employment_qualified,
        False,
    )
    results["business_is_sstb"] = business_is_sstb
    results["qualified_bdc_income"] = qualified_bdc_income
    results["qualified_reit_and_ptp_income"] = qualified_reit_and_ptp_income
    results[_SSTB_SELF_EMPLOYMENT_OUTPUT] = np.where(
        business_is_sstb,
        inputs.self_employment_income,
        0.0,
    )
    results["sstb_unadjusted_basis_qualified_property"] = np.where(
        business_is_sstb,
        ubia,
        0.0,
    )
    results["sstb_w2_wages_from_qualified_business"] = np.where(
        business_is_sstb,
        w2_wages,
        0.0,
    )
    results["unadjusted_basis_qualified_property"] = ubia
    results["w2_wages_from_qualified_business"] = w2_wages
    return {column: results[column] for column in US_QBI_OUTPUT_COLUMNS}


def with_qbi_simulation_from_puf_arrays(
    arrays: Mapping[str, Sequence[Any]],
    *,
    qbi_simulation_version: int,
    assumptions: QbiSimulationAssumptions | QbiSimulationAssumptionsV2 | None = None,
    sstb_crosswalk: SstbCrosswalk | Mapping[str, Any] | None = None,
) -> dict[str, Sequence[Any]]:
    """Return PUF arrays with a repository-owned QBI simulation applied.

    The input mapping is never mutated. Existing QBI leaves, including release
    1.8.0's stale deterministic W-2 proxy, are replaced. The returned mapping
    also replaces ``self_employment_income`` with its non-SSTB complement so
    the separately emitted SSTB leaf does not double Schedule C income.
    """

    resolved_assumptions = assumptions or load_qbi_simulation_assumptions(
        qbi_simulation_version
    )
    inputs = QbiSimulationInputs.from_puf_arrays(arrays)
    outputs = simulate_qbi_inputs(
        inputs,
        assumptions=resolved_assumptions,
        qbi_simulation_version=qbi_simulation_version,
        sstb_crosswalk=sstb_crosswalk,
    )
    result: dict[str, Sequence[Any]] = dict(arrays)
    result[_NON_SSTB_SELF_EMPLOYMENT_SOURCE] = np.where(
        outputs["business_is_sstb"],
        0.0,
        inputs.self_employment_income,
    )
    result.update(outputs)
    return result


def qbi_simulation_summary(
    outputs: Mapping[str, Sequence[Any]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Return aggregate-only distribution diagnostics for all 15 leaves."""

    missing = [column for column in US_QBI_OUTPUT_COLUMNS if column not in outputs]
    if missing:
        raise ValueError(f"QBI simulation summary missing output(s): {missing}.")
    first = np.asarray(outputs[US_QBI_OUTPUT_COLUMNS[0]])
    resolved_weights = (
        np.ones(len(first), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if resolved_weights.ndim != 1 or len(resolved_weights) != len(first):
        raise ValueError("QBI summary weights must align one-for-one with outputs.")
    if np.any(~np.isfinite(resolved_weights)) or np.any(resolved_weights < 0.0):
        raise ValueError("QBI summary weights must be finite and nonnegative.")
    total_weight = float(resolved_weights.sum())
    summary: dict[str, dict[str, float | int]] = {}
    for column in US_QBI_OUTPUT_COLUMNS:
        values = np.asarray(outputs[column], dtype=np.float64)
        if values.ndim != 1 or len(values) != len(first):
            raise ValueError(f"QBI summary output {column!r} has the wrong shape.")
        finite = np.isfinite(values)
        nonzero = finite & (values != 0.0)
        nonzero_weight = float(resolved_weights[nonzero].sum())
        summary[column] = {
            "nonzero_rows": int(np.count_nonzero(nonzero)),
            "nonzero_share": (
                nonzero_weight / total_weight if total_weight > 0.0 else 0.0
            ),
            "weighted_mean": (
                float(np.sum(np.where(finite, values, 0.0) * resolved_weights))
                / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }
    return summary


def resolve_sstb_crosswalk(
    assumptions: QbiSimulationAssumptionsV2,
    crosswalk: SstbCrosswalk | Mapping[str, Any] | None,
) -> SstbCrosswalk:
    resolved = (
        load_sstb_crosswalk(assumptions.sstb_classification.crosswalk_resource)
        if crosswalk is None
        else (
            crosswalk
            if isinstance(crosswalk, SstbCrosswalk)
            else parse_sstb_crosswalk(crosswalk)
        )
    )
    resolved.validate()
    return resolved


def _derive_v2_qualification_flags(
    inputs: QbiSimulationInputs,
    *,
    assumptions: QbiSimulationAssumptionsV2,
) -> tuple[np.ndarray, ...]:
    rng = _rng(assumptions.qualification_seed, assumptions.bit_generator)
    flags: list[np.ndarray] = []
    for derivation in assumptions.qualification_derivations:
        if derivation.mode == "derived":
            flags.append(inputs.source(derivation.source) != 0.0)
            continue
        if derivation.prior_probability is None:
            raise ValueError(
                f"QBI prior source {derivation.source!r} lacks a probability."
            )
        flags.append(rng.random(inputs.n) < derivation.prior_probability)
    return tuple(flags)


def _qualified_components(
    inputs: QbiSimulationInputs,
    *,
    source_order: tuple[str, ...],
    qualification_flags: tuple[np.ndarray, ...],
) -> np.ndarray:
    return np.column_stack(
        [
            inputs.source(source) * qualified
            for source, qualified in zip(
                source_order,
                qualification_flags,
                strict=True,
            )
        ]
    )


def _simulate_w2_and_ubia(
    qualified_components: np.ndarray,
    *,
    assumptions: QbiSimulationAssumptions,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    qbi = qualified_components.sum(axis=1)
    margins = _draw_source_weighted_beta(
        qualified_components,
        assumptions.profit_margin_parameters,
        rng,
    )
    revenues = np.divide(
        np.maximum(qbi, 0.0),
        margins,
        out=np.zeros_like(qbi, dtype=np.float64),
        where=margins > 0.0,
    )
    intercept = _calibrate_logit_intercept(
        revenues,
        slope=assumptions.has_employees_slope_per_dollar,
        target_share=assumptions.has_employees_target_share,
        iterations=assumptions.intercept_bisection_iterations,
    )
    employee_probability = np.where(
        revenues == 0.0,
        0.0,
        _logistic(intercept + assumptions.has_employees_slope_per_dollar * revenues),
    )
    has_employees = rng.binomial(n=1, p=employee_probability)
    labor_ratios = _draw_source_weighted_beta(
        qualified_components,
        assumptions.labor_ratio_parameters,
        rng,
    )
    w2_wages = revenues * labor_ratios * has_employees

    capital_probability = np.clip(
        _source_weighted_parameter(
            qualified_components,
            assumptions.capital_intensity_probabilities,
        ),
        0.0,
        1.0,
    )
    is_capital_intensive = rng.binomial(n=1, p=capital_probability).astype(bool)
    ubia_multiple = _source_weighted_parameter(
        qualified_components,
        assumptions.ubia_multiples,
    )
    target_mean = ubia_multiple * np.maximum(qbi, 0.0)
    eligible = is_capital_intensive & (target_mean > 0.0)
    safe_target_mean = np.where(target_mean > 0.0, target_mean, 1.0)
    mu = np.log(safe_target_mean) - (assumptions.ubia_sigma**2 / 2.0)
    ubia = np.where(
        eligible,
        rng.lognormal(mean=mu, sigma=assumptions.ubia_sigma),
        0.0,
    )
    return w2_wages, ubia


def _simulate_w2(
    qualified_components: np.ndarray,
    *,
    assumptions: QbiSimulationAssumptionsV2,
    rng: np.random.Generator,
) -> np.ndarray:
    qbi = qualified_components.sum(axis=1)
    margins = _draw_source_weighted_beta(
        qualified_components,
        assumptions.profit_margin_parameters,
        rng,
    )
    revenues = np.divide(
        np.maximum(qbi, 0.0),
        margins,
        out=np.zeros_like(qbi, dtype=np.float64),
        where=margins > 0.0,
    )
    intercept = _calibrate_logit_intercept(
        revenues,
        slope=assumptions.has_employees_slope_per_dollar,
        target_share=assumptions.has_employees_target_share,
        iterations=assumptions.intercept_bisection_iterations,
    )
    employee_probability = np.where(
        revenues == 0.0,
        0.0,
        _logistic(intercept + assumptions.has_employees_slope_per_dollar * revenues),
    )
    has_employees = rng.binomial(n=1, p=employee_probability)
    labor_ratios = _draw_source_weighted_beta(
        qualified_components,
        assumptions.labor_ratio_parameters,
        rng,
    )
    return revenues * labor_ratios * has_employees


def _simulate_ubia(
    qualified_components: np.ndarray,
    *,
    assumptions: QbiSimulationAssumptionsV2,
    rng: np.random.Generator,
) -> np.ndarray:
    qbi = qualified_components.sum(axis=1)
    capital_probability = np.clip(
        _source_weighted_parameter(
            qualified_components,
            assumptions.capital_intensity_probabilities,
        ),
        0.0,
        1.0,
    )
    is_capital_intensive = rng.binomial(n=1, p=capital_probability).astype(bool)
    ubia_multiple = _source_weighted_parameter(
        qualified_components,
        assumptions.ubia_multiples,
    )
    target_mean = ubia_multiple * np.maximum(qbi, 0.0)
    eligible = is_capital_intensive & (target_mean > 0.0)
    safe_target_mean = np.where(target_mean > 0.0, target_mean, 1.0)
    mu = np.log(safe_target_mean) - (assumptions.ubia_sigma**2 / 2.0)
    return np.where(
        eligible,
        rng.lognormal(mean=mu, sigma=assumptions.ubia_sigma),
        0.0,
    )


def _simulate_business_is_sstb(
    inputs: QbiSimulationInputs,
    *,
    qualification_flags: Mapping[str, np.ndarray],
    assumptions: QbiSimulationAssumptions,
    rng: np.random.Generator,
) -> np.ndarray:
    sources = np.column_stack(
        [
            np.maximum(inputs.source(source), 0.0) * qualification_flags[source]
            for source in assumptions.sstb_source_order
        ]
    )
    has_sstb_source = sources.sum(axis=1) > 0.0
    largest_source = np.argmax(sources, axis=1)
    probabilities = np.asarray(assumptions.sstb_probabilities)[largest_source]
    probabilities = np.where(has_sstb_source, probabilities, 0.0)
    return rng.binomial(n=1, p=probabilities).astype(bool)


def _simulate_investment_qbi(
    inputs: QbiSimulationInputs,
    *,
    assumptions: QbiSimulationAssumptions | QbiSimulationAssumptionsV2,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    exposure_bases = {
        "non_qualified_dividend_income": inputs.non_qualified_dividend_income,
        "partnership_s_corp_income": inputs.partnership_s_corp_income,
    }
    qualified_reit_and_ptp_income = np.zeros(inputs.n, dtype=np.float64)
    for exposure in assumptions.reit_ptp_exposures:
        qualified_reit_and_ptp_income += _sample_exposure_scaled_beta(
            exposure_bases[exposure.source],
            exposure,
            rng,
        )
    qualified_bdc_income = np.zeros(inputs.n, dtype=np.float64)
    for exposure in assumptions.bdc_exposures:
        qualified_bdc_income += _sample_exposure_scaled_beta(
            exposure_bases[exposure.source],
            exposure,
            rng,
        )
    return qualified_reit_and_ptp_income, qualified_bdc_income


def _sample_exposure_scaled_beta(
    base: np.ndarray,
    parameters: ExposureBetaParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    positive_base = np.maximum(np.asarray(base, dtype=np.float64), 0.0)
    receives = (positive_base > 0.0) & (
        rng.random(len(positive_base)) < parameters.probability_of_receiving
    )
    share = (
        rng.beta(
            parameters.beta.beta_a,
            parameters.beta.beta_b,
            len(positive_base),
        )
        * parameters.beta.scale
        + parameters.beta.shift
    )
    share = np.clip(share, 0.0, 1.0)
    return np.where(receives, positive_base * share, 0.0)


def _draw_source_weighted_beta(
    qualified_components: np.ndarray,
    parameters: tuple[BetaParameters, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    positive_components = np.maximum(qualified_components, 0.0)
    positive_total = positive_components.sum(axis=1)
    weighted_draw = np.zeros(len(qualified_components), dtype=np.float64)
    for index, parameter in enumerate(parameters):
        draw = (
            rng.beta(
                parameter.beta_a,
                parameter.beta_b,
                len(qualified_components),
            )
            * parameter.scale
            + parameter.shift
        )
        weighted_draw += positive_components[:, index] * draw
    return np.divide(
        weighted_draw,
        positive_total,
        out=np.zeros_like(positive_total, dtype=np.float64),
        where=positive_total > 0.0,
    )


def _source_weighted_parameter(
    qualified_components: np.ndarray,
    parameters: tuple[float, ...],
) -> np.ndarray:
    positive_components = np.maximum(qualified_components, 0.0)
    positive_total = positive_components.sum(axis=1)
    weighted_value = np.zeros(len(qualified_components), dtype=np.float64)
    for index, value in enumerate(parameters):
        weighted_value += positive_components[:, index] * value
    return np.divide(
        weighted_value,
        positive_total,
        out=np.zeros_like(positive_total, dtype=np.float64),
        where=positive_total > 0.0,
    )


def _calibrate_logit_intercept(
    revenues: np.ndarray,
    *,
    slope: float,
    target_share: float,
    iterations: int,
) -> float:
    positive = revenues > 0.0
    if not np.any(positive):
        return 0.0
    clipped_target = np.clip(float(target_share), 1e-9, 1.0 - 1e-9)
    slope_term = float(slope) * revenues[positive]
    target_logit = np.log(clipped_target / (1.0 - clipped_target))
    lower = target_logit - np.max(slope_term) - 80.0
    upper = target_logit - np.min(slope_term) + 80.0
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        mean_probability = _logistic(midpoint + slope_term).mean()
        if mean_probability < clipped_target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def _rng(seed: int, bit_generator: str) -> np.random.Generator:
    if bit_generator != "PCG64":
        raise ValueError(f"Unsupported QBI bit generator {bit_generator!r}.")
    return np.random.default_rng(seed)


def _infer_person_length(arrays: Mapping[str, Sequence[Any]]) -> int:
    candidates = (
        "person_tax_unit_id",
        "person_id",
        "self_employment_income",
        "self_employment_income_before_lsr",
        "E00900",
        "partnership_s_corp_income",
        "partnership_income",
        "s_corp_income",
        "farm_rent_income",
        "rental_income",
        "estate_income",
        "non_qualified_dividend_income",
    )
    for name in candidates:
        if name not in arrays:
            continue
        values = np.asarray(arrays[name])
        if values.ndim == 1:
            return len(values)
    raise ValueError(
        "QBI simulation cannot infer row count from the supplied PUF arrays."
    )


def _first_present(
    arrays: Mapping[str, Sequence[Any]],
    names: tuple[str, ...],
) -> str | None:
    return next((name for name in names if name in arrays), None)


def _first_array(
    arrays: Mapping[str, Sequence[Any]],
    names: tuple[str, ...],
    *,
    n: int,
) -> np.ndarray | None:
    name = _first_present(arrays, names)
    return None if name is None else _array_or_zeros(arrays, name, n=n)


def _array_or_zeros(
    arrays: Mapping[str, Sequence[Any]],
    name: str | None,
    *,
    n: int,
) -> np.ndarray:
    if name is None or name not in arrays:
        return np.zeros(n, dtype=np.float64)
    values = np.asarray(arrays[name], dtype=np.float64)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(
            f"QBI source {name!r} must be a one-dimensional array of length {n}."
        )
    values = np.where(np.isnan(values), 0.0, values)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise ValueError(f"QBI source {name!r} has {nonfinite} infinite value(s).")
    return values


def _difference_or_zeros(
    arrays: Mapping[str, Sequence[Any]],
    minuend: str,
    subtrahend: str,
    *,
    n: int,
) -> np.ndarray:
    if minuend not in arrays and subtrahend not in arrays:
        return np.zeros(n, dtype=np.float64)
    return _array_or_zeros(arrays, minuend, n=n) - _array_or_zeros(
        arrays, subtrahend, n=n
    )


def _zeros_if_none(values: np.ndarray | None, n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64) if values is None else values


def _validate_probabilities(name: str, values: tuple[float, ...]) -> None:
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"QBI {name} must lie in [0, 1].")


def _validate_beta_parameters(name: str, parameters: BetaParameters) -> None:
    if parameters.beta_a <= 0.0 or parameters.beta_b <= 0.0:
        raise ValueError(f"QBI {name} beta shapes must be positive.")
    if parameters.scale < 0.0:
        raise ValueError(f"QBI {name} beta scale must be nonnegative.")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _child_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _require_mapping(parent.get(key), key)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings.")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates.")
    return tuple(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _validate_v2_payload_keys(root: Mapping[str, Any]) -> None:
    """Require every v2 object to match the complete declared schema."""

    _require_exact_keys(
        root,
        (
            "schema_version",
            "qbi_simulation_version",
            "engine",
            "rng",
            "source_order",
            "qualification_derivations",
            "sstb_classification",
            "w2",
            "ubia",
            "investment",
        ),
        "QBI v2 assumptions",
    )
    rng = _child_mapping(root, "rng")
    _require_exact_keys(rng, ("bit_generator", "seeds"), "rng")
    _require_exact_keys(
        _child_mapping(rng, "seeds"),
        ("qualification", "sstb", "w2", "ubia", "investment"),
        "rng.seeds",
    )

    source_order = _string_tuple(root.get("source_order"), "source_order")
    derivations = _child_mapping(root, "qualification_derivations")
    _require_exact_keys(
        derivations,
        source_order,
        "qualification_derivations",
    )
    for source in source_order:
        label = f"qualification_derivations.{source}"
        entry = _require_mapping(derivations[source], label)
        _require_exact_keys(entry, ("mode", "prior", "rationale"), label)
        prior = entry.get("prior")
        if prior is not None:
            _require_exact_keys(
                _require_mapping(prior, f"{label}.prior"),
                ("probability",),
                f"{label}.prior",
            )

    classification = _child_mapping(root, "sstb_classification")
    _require_exact_keys(
        classification,
        (
            "mode",
            "crosswalk_resource",
            "occupation_column",
            "industry_column",
            "agi_column",
            "ambiguous_prior",
            "ambiguous_prior_status",
            "agi_band_format",
            "passive_passthrough_sstb_prior_by_agi",
            "passive_passthrough_prior_status",
            "rationale",
            "follow_up",
        ),
        "sstb_classification",
    )

    w2 = _child_mapping(root, "w2")
    _require_exact_keys(
        w2,
        (
            "model",
            "profit_margin_distribution",
            "has_employees_logit",
            "labor_ratio_distribution",
        ),
        "w2",
    )
    _validate_strict_beta_mapping(
        _child_mapping(w2, "profit_margin_distribution"),
        source_order,
        "w2.profit_margin_distribution",
    )
    employee_logit = _child_mapping(w2, "has_employees_logit")
    _require_exact_keys(
        employee_logit,
        (
            "slope_per_dollar",
            "target_share_among_positive_receipts",
            "intercept_bisection_iterations",
        ),
        "w2.has_employees_logit",
    )
    _validate_strict_beta_mapping(
        _child_mapping(w2, "labor_ratio_distribution"),
        source_order,
        "w2.labor_ratio_distribution",
    )

    ubia = _child_mapping(root, "ubia")
    _require_exact_keys(
        ubia,
        (
            "model",
            "sigma",
            "multiple_of_qbi",
            "capital_intensity_probabilities",
        ),
        "ubia",
    )
    _require_exact_keys(
        _child_mapping(ubia, "multiple_of_qbi"),
        source_order,
        "ubia.multiple_of_qbi",
    )
    _require_exact_keys(
        _child_mapping(ubia, "capital_intensity_probabilities"),
        source_order,
        "ubia.capital_intensity_probabilities",
    )

    investment = _child_mapping(root, "investment")
    _require_exact_keys(
        investment,
        (
            "model",
            "reit_ptp_anchor",
            "reit_ptp_exposure_order",
            "reit_ptp_income_distribution",
            "bdc_anchor",
            "bdc_exposure_order",
            "bdc_income_distribution",
        ),
        "investment",
    )
    reit_order = _string_tuple(
        investment.get("reit_ptp_exposure_order"),
        "investment.reit_ptp_exposure_order",
    )
    bdc_order = _string_tuple(
        investment.get("bdc_exposure_order"),
        "investment.bdc_exposure_order",
    )
    _validate_strict_exposure_mapping(
        _child_mapping(investment, "reit_ptp_income_distribution"),
        reit_order,
        "investment.reit_ptp_income_distribution",
    )
    _validate_strict_exposure_mapping(
        _child_mapping(investment, "bdc_income_distribution"),
        bdc_order,
        "investment.bdc_income_distribution",
    )


def _validate_strict_beta_mapping(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> None:
    _require_exact_keys(values, order, label)
    for source in order:
        _require_exact_keys(
            _require_mapping(values[source], f"{label}.{source}"),
            ("beta_a", "beta_b", "scale", "shift"),
            f"{label}.{source}",
        )


def _validate_strict_exposure_mapping(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> None:
    _require_exact_keys(values, order, label)
    for source in order:
        _require_exact_keys(
            _require_mapping(values[source], f"{label}.{source}"),
            (
                "probability_of_receiving",
                "beta_a",
                "beta_b",
                "scale",
                "shift",
            ),
            f"{label}.{source}",
        )


def _parse_aggregate_evidence_anchor(
    values: Mapping[str, Any],
    label: str,
) -> AggregateEvidenceAnchor:
    _require_exact_keys(
        values,
        (
            "provisional",
            "published_income_dollars",
            "published_component_dollars",
            "comparison_component_2022_dollars",
            "replay_factor_band",
            "rationale",
        ),
        label,
    )
    raw_band = values.get("replay_factor_band")
    factor_band: tuple[float, float] | None = None
    if raw_band is not None:
        if not isinstance(raw_band, list) or len(raw_band) != 2:
            raise ValueError(f"{label}.replay_factor_band must be null or [low, high].")
        factor_band = (
            _number(raw_band[0], f"{label}.replay_factor_band[0]"),
            _number(raw_band[1], f"{label}.replay_factor_band[1]"),
        )

    def optional_number(key: str) -> float | None:
        value = values.get(key)
        return None if value is None else _number(value, f"{label}.{key}")

    anchor = AggregateEvidenceAnchor(
        provisional=values.get("provisional"),
        published_income_dollars=optional_number("published_income_dollars"),
        published_component_dollars=optional_number("published_component_dollars"),
        comparison_component_2022_dollars=optional_number(
            "comparison_component_2022_dollars"
        ),
        replay_factor_band=factor_band,
        rationale=_string(values.get("rationale"), f"{label}.rationale"),
    )
    anchor.validate(label)
    return anchor


def _parse_agi_prior_bands(
    values: Mapping[str, Any],
) -> tuple[AgiSstbPriorBand, ...]:
    if not values:
        raise ValueError("QBI v2 passive SSTB AGI priors must not be empty.")
    bands: list[AgiSstbPriorBand] = []
    for label, probability in values.items():
        if not isinstance(label, str) or label.count(":") != 1:
            raise ValueError(
                "QBI v2 passive SSTB AGI band names must use 'lower:upper' syntax."
            )
        lower_label, upper_label = label.split(":")
        lower = _parse_band_endpoint(lower_label, f"{label} lower endpoint")
        upper = _parse_band_endpoint(upper_label, f"{label} upper endpoint")
        bands.append(
            AgiSstbPriorBand(
                label=label,
                lower=lower,
                upper=upper,
                probability=_number(
                    probability,
                    (
                        "sstb_classification."
                        f"passive_passthrough_sstb_prior_by_agi.{label}"
                    ),
                ),
            )
        )
    result = tuple(sorted(bands, key=lambda band: band.lower))
    _validate_agi_prior_bands(result)
    return result


def _parse_band_endpoint(value: str, label: str) -> float:
    if value == "-inf":
        return float("-inf")
    if value == "inf":
        return float("inf")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"QBI v2 AGI {label} must be numeric or infinite.") from error
    if not np.isfinite(result):
        raise ValueError(f"QBI v2 AGI {label} must use '-inf' or 'inf'.")
    return result


def _validate_agi_prior_bands(bands: tuple[AgiSstbPriorBand, ...]) -> None:
    if not bands:
        raise ValueError("QBI v2 passive SSTB AGI priors must not be empty.")
    if bands[0].lower != float("-inf") or bands[-1].upper != float("inf"):
        raise ValueError("QBI v2 passive SSTB AGI priors must cover all AGI values.")
    for index, band in enumerate(bands):
        if band.lower >= band.upper:
            raise ValueError(f"QBI v2 AGI band {band.label!r} is empty or reversed.")
        _validate_probabilities(
            f"passive SSTB AGI prior {band.label!r}",
            (band.probability,),
        )
        if index and bands[index - 1].upper != band.lower:
            raise ValueError(
                "QBI v2 passive SSTB AGI priors must be contiguous and non-overlapping."
            )


def _parse_crosswalk_entries(
    value: Any,
    *,
    family: str,
    classification_system_key: str,
    categories: tuple[str, ...],
) -> tuple[SstbCrosswalkEntry, ...]:
    label = f"SSTB crosswalk.{family}_2018"
    if family == "industry":
        label = "SSTB crosswalk.industry_2017"
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    result: list[SstbCrosswalkEntry] = []
    for index, raw_entry in enumerate(value):
        entry_label = f"{label}[{index}]"
        entry = _require_mapping(raw_entry, entry_label)
        classification = _string(
            entry.get("classification"),
            f"{entry_label}.classification",
        )
        expected_keys = (
            "census_code",
            "census_title",
            classification_system_key,
            "sstb_category",
            "classification",
            "probability",
            "rationale",
        )
        if classification == "ambiguous":
            expected_keys += ("provisional", "basis")
        _require_exact_keys(entry, expected_keys, entry_label)
        raw_code = entry.get("census_code")
        if (
            not isinstance(raw_code, str)
            or len(raw_code) != 4
            or not raw_code.isdigit()
        ):
            raise ValueError(f"{entry_label}.census_code must use four decimal digits.")
        _string(entry.get("census_title"), f"{entry_label}.census_title")
        _string(
            entry.get(classification_system_key),
            f"{entry_label}.{classification_system_key}",
        )
        entry_categories = _string_tuple(
            entry.get("sstb_category"),
            f"{entry_label}.sstb_category",
        )
        unknown_categories = sorted(set(entry_categories) - set(categories))
        if unknown_categories:
            raise ValueError(
                f"{entry_label}.sstb_category has unknown value(s) "
                f"{unknown_categories}."
            )
        _string(entry.get("rationale"), f"{entry_label}.rationale")
        result.append(
            SstbCrosswalkEntry(
                code=int(raw_code),
                classification=classification,
                probability=_number(
                    entry.get("probability"),
                    f"{entry_label}.probability",
                ),
                provisional=entry.get("provisional", False),
                basis=(
                    _string(entry.get("basis"), f"{entry_label}.basis")
                    if "basis" in entry
                    else None
                ),
            )
        )
    return tuple(sorted(result, key=lambda entry: entry.code))


def _validate_explicit_nonsstb_entries(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    for index, raw_entry in enumerate(value):
        entry_label = f"{label}[{index}]"
        entry = _require_mapping(raw_entry, entry_label)
        _require_exact_keys(
            entry,
            ("census_code", "census_title", "why", "probability"),
            entry_label,
        )
        for key in ("census_code", "census_title", "why"):
            _string(entry.get(key), f"{entry_label}.{key}")
        probability = _number(
            entry.get("probability"),
            f"{entry_label}.probability",
        )
        if probability != 0.0:
            raise ValueError(
                f"{entry_label}.probability must be 0.0 for documented non-SSTB codes."
            )


def _ordered_scalars(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    _require_exact_keys(values, order, label)
    return tuple(_number(values[name], f"{label}.{name}") for name in order)


def _ordered_betas(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> tuple[BetaParameters, ...]:
    _require_exact_keys(values, order, label)
    return tuple(
        _beta_parameters(
            _require_mapping(values[name], f"{label}.{name}"),
            f"{label}.{name}",
        )
        for name in order
    )


def _ordered_exposures(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> tuple[ExposureBetaParameters, ...]:
    _require_exact_keys(values, order, label)
    exposures: list[ExposureBetaParameters] = []
    for source in order:
        parameters = _require_mapping(values[source], f"{label}.{source}")
        exposures.append(
            ExposureBetaParameters(
                source=source,
                probability_of_receiving=_number(
                    parameters.get("probability_of_receiving"),
                    f"{label}.{source}.probability_of_receiving",
                ),
                beta=_beta_parameters(parameters, f"{label}.{source}"),
            )
        )
    return tuple(exposures)


def _beta_parameters(
    values: Mapping[str, Any],
    label: str,
) -> BetaParameters:
    return BetaParameters(
        beta_a=_number(values.get("beta_a"), f"{label}.beta_a"),
        beta_b=_number(values.get("beta_b"), f"{label}.beta_b"),
        scale=_number(values.get("scale", 1.0), f"{label}.scale"),
        shift=_number(values.get("shift", 0.0), f"{label}.shift"),
    )


def _require_exact_keys(
    values: Mapping[str, Any],
    order: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(set(order) - set(values))
    extra = sorted(set(values) - set(order))
    if missing or extra:
        raise ValueError(
            f"{label} keys must match its explicit order; "
            f"missing={missing}, extra={extra}."
        )
