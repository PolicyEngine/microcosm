"""Versioned Section 199A simulation for raw or legacy PUF arrays.

The pinned 1.8.0 PUF physically carries only an older deterministic W-2
business-wage proxy. A later retired ``policyengine-us-data`` loader made that
artifact appear to contain the current 15-leaf QBI surface by simulating the
missing inputs and writing them into the downloaded file during ``load()``.

This module ports that version-1 model into Populace as pure, seeded NumPy
logic. The assumptions, source order, exposure order, bit generator, and four
independent seeds are packaged in ``qbi_assumptions_v1.yaml``. Callers must
name ``qbi_simulation_version`` explicitly; no version is inferred from an
unversioned artifact.

The simulation core accepts a normalized assumptions object. A future v2 can
load a different, evidence-anchored assumptions object without changing the
source normalization, orchestration, output contract, or donor-stage wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
import yaml

from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS

__all__ = [
    "QBI_SIMULATION_ARCHIVED_ASSUMPTIONS_URL",
    "QBI_SIMULATION_ARCHIVED_BACKFILL_URL",
    "QBI_SIMULATION_ARCHIVED_DERIVATION_URL",
    "QBI_SIMULATION_ARCHIVED_IMPLEMENTATION_URL",
    "QBI_SIMULATION_ASSUMPTIONS_RESOURCE",
    "QBI_SIMULATION_SOURCE_NAMES",
    "QBI_SIMULATION_VERSION",
    "BetaParameters",
    "ExposureBetaParameters",
    "QbiSimulationAssumptions",
    "QbiSimulationInputs",
    "load_qbi_simulation_assumptions",
    "qbi_simulation_summary",
    "simulate_qbi_inputs",
    "us_qbi_simulation_stage_spec",
    "with_qbi_simulation_from_puf_arrays",
]

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/policyengine-us-data/blob/"
    f"{_ARCHIVED_COMMIT}/policyengine_us_data/"
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
QBI_SIMULATION_ASSUMPTIONS_RESOURCE = "qbi_assumptions_v1.yaml"
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


@lru_cache(maxsize=1)
def load_qbi_simulation_assumptions(
    qbi_simulation_version: int,
) -> QbiSimulationAssumptions:
    """Load and strictly validate one packaged QBI assumptions version."""

    if qbi_simulation_version != QBI_SIMULATION_VERSION:
        raise ValueError(
            "Unsupported qbi_simulation_version "
            f"{qbi_simulation_version!r}; supported versions: "
            f"({QBI_SIMULATION_VERSION},)."
        )
    resource = files("populace.build.us").joinpath(QBI_SIMULATION_ASSUMPTIONS_RESOURCE)
    with resource.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    root = _require_mapping(payload, "QBI assumptions")
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
    return spec


def simulate_qbi_inputs(
    inputs: QbiSimulationInputs,
    *,
    assumptions: QbiSimulationAssumptions,
    qbi_simulation_version: int,
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
    if qbi_simulation_version != QBI_SIMULATION_VERSION:
        raise ValueError(
            f"Unsupported qbi_simulation_version {qbi_simulation_version!r}."
        )
    assumptions.validate()

    qualification_rng = _rng(assumptions.qualification_seed, assumptions.bit_generator)
    qualification_flags = tuple(
        qualification_rng.random(inputs.n) < probability
        for probability in assumptions.qualification_probabilities
    )
    qualified_components = np.column_stack(
        [
            inputs.source(source) * qualified
            for source, qualified in zip(
                assumptions.source_order,
                qualification_flags,
                strict=True,
            )
        ]
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
    assumptions: QbiSimulationAssumptions | None = None,
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
    assumptions: QbiSimulationAssumptions,
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
