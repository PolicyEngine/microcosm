"""UK weighted-integrity gates: input-mass parity and QRF tail concentration.

Increment 4 of the UK parity plan (#609, under the #578 acceptance rule)
ports the two US integrity gates that were each purchased with a named
incident:

- **Input-mass parity** (#278): a rebuilt or selected artifact can zero an
  untargeted input column while every calibrated target still fits. The gate
  compares weighted per-column totals against a frozen reference; total loss
  fails at any tolerance.
- **QRF tail concentration** (#462): a weighted QRF can broadcast a
  donor-tail point mass so a handful of records carry most of a column's
  shipped weighted mass while the paired count target is hit exactly. The
  tell is weighted-mass concentration, which neither support clipping nor
  count targets can see.

Both gates reuse the shared implementations in :mod:`populace.build.gates`
verbatim — the UK wrappers only add the evidence plumbing for the national
table layout, the reference identity record, and the universal
reviewed-exclusion discipline (mandatory reason, dormant entries reported,
stale entries fail).

Thresholds carry **no committed defaults**: the US numbers are calibrated to
US incidents and the #609 measurement pass has not yet adjudicated UK
boundaries. Arming either gate requires explicit policy values; once the
measurement numbers are posted on #578, the adjudicated constants belong
next to :data:`~populace.build.uk_runtime.terminal_gates.UK_MAX_TO_MEDIAN_WEIGHT_RATIO`
with the same derivation-comment discipline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import (
    GateResult,
    input_mass_parity_gate,
    tail_concentration_gate,
)

__all__ = [
    "UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE",
    "UK_INPUT_MASS_PARITY_GATE_NAME",
    "UK_QRF_TAIL_CONCENTRATION_GATE_NAME",
    "UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE",
    "UKInputMassParityPolicy",
    "UKInputMassReference",
    "UKQRFTailConcentrationPolicy",
    "load_uk_input_mass_reference",
    "load_uk_reviewed_exclusion_register",
    "uk_dataset_input_mass_totals",
    "uk_input_mass_parity_gate",
    "uk_qrf_tail_concentration_columns",
    "uk_qrf_tail_concentration_gate",
]

UK_INPUT_MASS_PARITY_GATE_NAME = "input_mass_parity"
UK_QRF_TAIL_CONCENTRATION_GATE_NAME = "qrf_tail_concentration"
UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE = "input_mass_reviewed_exclusions.json"
UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE = "qrf_tail_reviewed_exclusions.json"

# Mirrors terminal_gates._STRUCTURAL_COLUMNS for the national table layout;
# ids and the weight vector are plumbing, not input mass.
_UK_ENTITY_STRUCTURAL_COLUMNS: Mapping[str, frozenset[str]] = {
    "person": frozenset({"person_id", "person_household_id", "person_benunit_id"}),
    "benunit": frozenset({"benunit_id"}),
    "household": frozenset({"household_id", "household_weight"}),
}


def _reviewed_reason_mapping(values: object, *, label: str) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} reviewed exclusions must be a mapping.")
    normalized = {str(name): str(reason) for name, reason in values.items()}
    missing = sorted(name for name, reason in normalized.items() if not reason.strip())
    if missing:
        raise ValueError(f"{label} reviewed exclusions need reasons: {missing}.")
    return dict(sorted(normalized.items()))


def load_uk_reviewed_exclusion_register(
    source: str | Path | None,
    *,
    resource: str,
) -> dict[str, str]:
    """Load one committed reviewed-exclusion register.

    ``source`` overrides the committed default (``resource``, a JSON file
    under ``populace.build.uk``). The register schema is
    ``{"schema_version": 1, "description": ..., "exclusions": {column:
    reason}}``; every reason must be non-empty, which the gates re-validate
    so a register cannot bypass the discipline by construction order.
    """

    if source is None:
        raw = files("populace.build.uk").joinpath(resource).read_text("utf-8")
        label = resource
    else:
        raw = Path(source).read_text(encoding="utf-8")
        label = str(source)
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: exclusion register must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{label}: exclusion register schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, Mapping):
        raise ValueError(
            f"{label}: exclusion register must carry an 'exclusions' object."
        )
    return _reviewed_reason_mapping(exclusions, label=label)


def load_uk_input_mass_reference(source: str | Path) -> UKInputMassReference:
    """Load frozen reference totals emitted by the #609 measurement tooling.

    Schema: ``{"schema_version": 1, "identity": {filename, revision, sha256,
    vintage}, "totals": {"entity.column": weighted_total}}``. The identity
    names the pinned artifact the totals were measured from and is recorded
    verbatim in the gate details and the attestation evidence digest.
    """

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: input-mass reference must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{path}: input-mass reference schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{path}: input-mass reference needs an 'identity' object.")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError(f"{path}: input-mass reference needs a 'totals' object.")
    return UKInputMassReference(
        totals=dict(totals),
        filename=str(identity.get("filename", "")),
        revision=str(identity.get("revision", "")),
        sha256=str(identity.get("sha256", "")),
        vintage=str(identity.get("vintage", "")),
    )


@dataclass(frozen=True)
class UKInputMassReference:
    """Frozen reference totals and pinned identity for input-mass parity.

    The reference choice is load-bearing (#327): comparing a calibrated
    artifact against a raw base flags correct target-aligned drift as
    failure, so callers must name exactly which frozen artifact the totals
    were measured from. The identity is recorded in the gate details and
    bound into the terminal report's evidence digest.
    """

    totals: Mapping[str, float]
    filename: str
    revision: str
    sha256: str
    vintage: str

    def __post_init__(self) -> None:
        if not isinstance(self.totals, Mapping) or not self.totals:
            raise ValueError(
                "UK input-mass reference totals must be a non-empty mapping."
            )
        normalized: dict[str, float] = {}
        for name, total in self.totals.items():
            column = str(name)
            value = float(total)
            if not column or not math.isfinite(value):
                raise ValueError(
                    "UK input-mass reference totals must map non-empty column "
                    f"names to finite totals; got {name!r} -> {total!r}."
                )
            normalized[column] = value
        object.__setattr__(
            self, "totals", MappingProxyType(dict(sorted(normalized.items())))
        )
        for field_name in ("filename", "revision", "vintage"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"UK input-mass reference {field_name} must be non-empty."
                )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError(
                "UK input-mass reference sha256 must be a lowercase sha256."
            )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "revision": self.revision,
            "sha256": self.sha256,
            "vintage": self.vintage,
        }


@dataclass(frozen=True)
class UKInputMassParityPolicy:
    """Reviewed thresholds and exclusion register for input-mass parity.

    There are deliberately no defaults: the US tolerances (0.5 relative, a
    $1e9 USD floor over a 337k-record pool) are calibrated to US incidents
    and dollar scales. UK values must come from the #609 measurement pass
    over the certified compact, the pinned eFRS incumbent, and the staging
    candidate, and be recorded with their derivation when committed.
    """

    relative_tolerance: float
    minimum_reference_total: float
    reviewed_exclusions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tolerance = float(self.relative_tolerance)
        floor = float(self.minimum_reference_total)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "UK input-mass relative_tolerance must be finite and "
                f"non-negative, got {self.relative_tolerance!r}."
            )
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(
                "UK input-mass minimum_reference_total must be finite and "
                f"non-negative, got {self.minimum_reference_total!r}."
            )
        object.__setattr__(self, "relative_tolerance", tolerance)
        object.__setattr__(self, "minimum_reference_total", floor)
        object.__setattr__(
            self,
            "reviewed_exclusions",
            MappingProxyType(
                _reviewed_reason_mapping(
                    self.reviewed_exclusions, label="UK input-mass"
                )
            ),
        )

    def policy_payload(self) -> dict[str, object]:
        return {
            "relative_tolerance": self.relative_tolerance,
            "minimum_reference_total": self.minimum_reference_total,
            "reviewed_exclusions": dict(self.reviewed_exclusions),
        }


@dataclass(frozen=True)
class UKQRFTailConcentrationPolicy:
    """Reviewed thresholds and exclusion register for QRF tail concentration.

    No committed defaults, for the same reason as
    :class:`UKInputMassParityPolicy`: the US ``top_k=100`` /
    ``max_top_share=0.75`` / ``min_nonzero_records=500`` numbers are
    calibrated to the #462 incident on the CPS spine. The UK boundaries must
    be measured on the staging pool before they are written down.
    """

    top_k: int
    max_top_share: float
    min_nonzero_records: int
    reviewed_exclusions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError(f"UK QRF tail top_k must be an integer, got {self.top_k!r}.")
        if self.top_k < 1:
            raise ValueError(f"UK QRF tail top_k must be at least 1, got {self.top_k!r}.")
        share = float(self.max_top_share)
        if not math.isfinite(share) or not 0.0 < share < 1.0:
            raise ValueError(
                f"UK QRF tail max_top_share must be in (0, 1), got "
                f"{self.max_top_share!r}."
            )
        if isinstance(self.min_nonzero_records, bool) or not isinstance(
            self.min_nonzero_records, int
        ):
            raise ValueError(
                "UK QRF tail min_nonzero_records must be an integer, got "
                f"{self.min_nonzero_records!r}."
            )
        if self.min_nonzero_records <= self.top_k:
            raise ValueError(
                "UK QRF tail min_nonzero_records must exceed top_k so the tail "
                f"is a strict subset of the carriers, got "
                f"min_nonzero_records={self.min_nonzero_records!r} with "
                f"top_k={self.top_k!r}."
            )
        object.__setattr__(self, "max_top_share", share)
        object.__setattr__(
            self,
            "reviewed_exclusions",
            MappingProxyType(
                _reviewed_reason_mapping(self.reviewed_exclusions, label="UK QRF tail")
            ),
        )

    def policy_payload(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "max_top_share": self.max_top_share,
            "min_nonzero_records": self.min_nonzero_records,
            "reviewed_exclusions": dict(self.reviewed_exclusions),
        }


def _entity_table(dataset: Any, entity: str) -> pd.DataFrame:
    table = (
        dataset.get(entity)
        if isinstance(dataset, Mapping)
        else getattr(dataset, entity, None)
    )
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"UK weighted-integrity gates require a {entity} DataFrame."
        )
    return table


def _uk_entity_weights(dataset: Any) -> dict[str, np.ndarray]:
    """Resolve per-entity weights from the national household vector.

    Household rows carry ``household_weight`` directly; person and benunit
    rows inherit the weight of their containing household through the
    membership columns, matching how PolicyEngine-UK broadcasts weights.
    """

    household = _entity_table(dataset, "household")
    person = _entity_table(dataset, "person")
    benunit = _entity_table(dataset, "benunit")
    for entity, table in (("household", household), ("person", person)):
        missing = _UK_ENTITY_STRUCTURAL_COLUMNS[entity] - set(table.columns)
        if missing:
            raise ValueError(
                f"UK {entity} table is missing column(s): {sorted(missing)}."
            )
    if "benunit_id" not in benunit.columns:
        raise ValueError("UK benunit table is missing column(s): ['benunit_id'].")

    household_weights = pd.to_numeric(
        household["household_weight"], errors="coerce"
    ).astype(np.float64)
    if household_weights.isna().any() or not np.isfinite(
        household_weights.to_numpy()
    ).all():
        raise ValueError("UK household weights must be finite numbers.")
    by_household = pd.Series(
        household_weights.to_numpy(), index=household["household_id"]
    )
    if by_household.index.duplicated().any():
        raise ValueError("UK household_id values must be unique.")

    person_weights = person["person_household_id"].map(by_household)
    if person_weights.isna().any():
        raise ValueError(
            "UK person rows reference household_id values with no resolvable "
            "weight."
        )

    # A benunit inherits the weight of the household containing it, so the
    # nesting must hold before the mapping means anything.  The effective-mass
    # coverage gate enforces the same invariant, but that is a separate gate in
    # a batched report: this one has to fail closed on its own evidence rather
    # than total a split benunit against one of its households.
    benunit_membership = person[
        ["person_benunit_id", "person_household_id"]
    ].drop_duplicates()
    split = benunit_membership["person_benunit_id"].duplicated()
    if bool(split.any()):
        offenders = sorted(
            benunit_membership.loc[split, "person_benunit_id"].unique().tolist()
        )[:5]
        raise ValueError(
            "UK weighted-integrity totals require each benunit to belong to "
            f"exactly one household; split benunit id(s): {offenders}."
        )
    benunit_household = benunit_membership.set_index("person_benunit_id")[
        "person_household_id"
    ]
    benunit_weights = benunit["benunit_id"].map(benunit_household).map(by_household)
    if benunit_weights.isna().any():
        raise ValueError(
            "UK benunit rows have no member persons to resolve a household "
            "weight from."
        )
    return {
        "person": person_weights.to_numpy(dtype=np.float64),
        "benunit": benunit_weights.to_numpy(dtype=np.float64),
        "household": household_weights.to_numpy(dtype=np.float64),
    }


def uk_dataset_input_mass_totals(
    dataset: Any,
    *,
    columns: Iterable[str] | None = None,
) -> dict[str, float]:
    """Weighted totals of the national tables' numeric and boolean columns.

    The UK analog of :func:`populace.build.input_mass.input_mass_totals` for
    the person/benunit/household table layout the national builder stages
    (the shared helper operates on :class:`populace.frame.Frame`). Column
    names are namespaced as ``entity.column`` — the convention the UK
    degenerate-surface gate already uses — because the national tables do not
    enforce globally unique column names across entities.
    """

    weights = _uk_entity_weights(dataset)
    requested = None if columns is None else {str(name) for name in columns}
    totals: dict[str, float] = {}
    for entity in ("person", "benunit", "household"):
        table = _entity_table(dataset, entity)
        structural = _UK_ENTITY_STRUCTURAL_COLUMNS[entity]
        entity_weights = weights[entity]
        for column in table.columns:
            if column in structural:
                continue
            name = f"{entity}.{column}"
            if requested is not None and name not in requested:
                continue
            values = table[column]
            if pd.api.types.is_bool_dtype(values):
                numeric = values.fillna(False).to_numpy(dtype=np.float64)
            elif pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                numeric = numeric.fillna(0.0).to_numpy(dtype=np.float64)
            else:
                continue
            totals[name] = float(numeric @ entity_weights)
    return totals


def uk_input_mass_parity_gate(
    candidate_totals: Mapping[str, float],
    reference: UKInputMassReference,
    *,
    policy: UKInputMassParityPolicy,
    candidate_name: str = "uk_release_candidate",
) -> GateResult:
    """Require persisted UK input mass to survive against a frozen reference.

    Wraps :func:`populace.build.gates.input_mass_parity_gate` verbatim — a
    zero candidate total fails at any tolerance (the #278 signature),
    candidate-only columns are reported and never fail, and near-zero
    reference columns are skipped — and adds the universal exclusion
    discipline the shared gate lacks: an exclusion whose column is now
    within tolerance is stale and **fails**; an exclusion outside the audited
    surface (absent from the reference, or below the reference floor) is
    dormant and reported.
    """

    if not isinstance(reference, UKInputMassReference):
        raise TypeError("reference must be UKInputMassReference.")
    if not isinstance(policy, UKInputMassParityPolicy):
        raise TypeError("policy must be UKInputMassParityPolicy.")
    exclusions = dict(policy.reviewed_exclusions)
    base = input_mass_parity_gate(
        candidate_totals,
        reference.totals,
        candidate_name=candidate_name,
        reference_name=reference.filename,
        relative_tolerance=policy.relative_tolerance,
        minimum_reference_total=policy.minimum_reference_total,
        reviewed_exclusions=exclusions,
    )

    stale: list[str] = []
    dormant: list[str] = []
    for column in sorted(exclusions):
        if column not in reference.totals:
            dormant.append(column)
            continue
        if abs(reference.totals[column]) <= policy.minimum_reference_total:
            dormant.append(column)
            continue
        # Re-check the single excluded column without its exclusion, reusing
        # the shared gate's exact semantics (zero-fail, drift, absence).
        probe = input_mass_parity_gate(
            (
                {column: candidate_totals[column]}
                if column in candidate_totals
                else {}
            ),
            {column: reference.totals[column]},
            candidate_name=candidate_name,
            reference_name=reference.filename,
            relative_tolerance=policy.relative_tolerance,
            minimum_reference_total=policy.minimum_reference_total,
        )
        if probe.passed:
            stale.append(column)

    failures = list(base.failures)
    if stale:
        failures.append(
            "Stale reviewed input-mass exclusions — the column is within "
            f"tolerance now, remove the exclusion: {stale}."
        )
    return GateResult(
        name=UK_INPUT_MASS_PARITY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            **dict(base.details),
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "reference_identity": reference.identity,
        },
    )


def uk_qrf_tail_concentration_columns(
    dataset: Any,
    *,
    output_columns: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Assemble the tail-concentration evidence for the declared QRF surface.

    The surface defaults to every output declared by a
    ``fit_weighted_qrf_stage*`` operation in the HMRC source manifest.
    Unlike the US wrapper, there is **no sparsity filter**: on the UK staging
    pool the SPI channel is a large fraction of all persons, so a
    nonzero-share cutoff would silently skip the gate exactly where the risk
    lives (#609). Every declared, present, numeric output is checked;
    ``min_nonzero_records`` in the policy remains the sole thinness guard.
    """

    if output_columns is None:
        # Lazy import: the source-contract module owns manifest knowledge and
        # pulls in the full HMRC runtime, which this module must not require
        # at import time.
        from populace.build.uk_runtime.hmrc_source_contract import (
            uk_hmrc_weighted_qrf_output_columns,
        )

        declared = uk_hmrc_weighted_qrf_output_columns()
    else:
        declared = tuple(str(name) for name in output_columns)
    if not declared:
        raise ValueError("UK QRF tail-concentration surface must be non-empty.")

    person = _entity_table(dataset, "person")
    person_weights = _uk_entity_weights(dataset)["person"]
    values: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    absent: list[str] = []
    non_numeric: list[str] = []
    for column in declared:
        if column not in person.columns:
            absent.append(column)
            continue
        series = person[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(
            series
        ):
            non_numeric.append(column)
            continue
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        values[column] = numeric.to_numpy(dtype=np.float64)
        weights[column] = person_weights
    surface: dict[str, object] = {
        "declared_qrf_outputs": len(declared),
        "checked_columns": sorted(values),
        "absent_columns": absent,
        "non_numeric_columns": non_numeric,
        "density_filter": "none: every declared output is checked (#609)",
    }
    return values, weights, surface


def uk_qrf_tail_concentration_gate(
    column_values: Mapping[str, Iterable[float]],
    column_weights: Mapping[str, Iterable[float]],
    *,
    policy: UKQRFTailConcentrationPolicy,
    surface: Mapping[str, object] | None = None,
) -> GateResult:
    """No declared UK QRF output hides its mass in a handful of records.

    Wraps :func:`populace.build.gates.tail_concentration_gate` — which
    already enforces the universal exclusion discipline (stale entries fail,
    dormant entries are reported) — under the UK gate name, and records the
    manifest-derived surface metadata alongside the shared details.
    """

    if not isinstance(policy, UKQRFTailConcentrationPolicy):
        raise TypeError("policy must be UKQRFTailConcentrationPolicy.")
    base = tail_concentration_gate(
        column_values,
        column_weights,
        top_k=policy.top_k,
        max_top_share=policy.max_top_share,
        min_nonzero_records=policy.min_nonzero_records,
        reviewed_exclusions=dict(policy.reviewed_exclusions),
    )
    details = dict(base.details)
    if surface is not None:
        details["surface"] = dict(surface)
    return GateResult(
        name=UK_QRF_TAIL_CONCENTRATION_GATE_NAME,
        passed=base.passed,
        failures=base.failures,
        details=details,
    )
