"""Assign the passive share of partnership and S-corporation income.

This is a versioned sibling of the archived QBI-input reconciliation, not a
replacement for it.  Survey evidence supplies the Schedule-E-band shape and a
restricted replay build persists the single administrative-anchor shift.  The
runtime only loads those committed resources and makes seeded draws.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.qbi_passive_passthrough_evidence import (
    SCF_PASSIVE_INCOME_BANDS,
    validate_qbi_passive_passthrough_resource,
)
from microcosm.frame import Frame

__all__ = [
    "QBI_PASSIVE_ASSUMPTIONS_RESOURCE",
    "QBI_PASSIVE_EVIDENCE_RESOURCE",
    "US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN",
    "US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS",
    "assign_passive_partnership_s_corp_income",
    "build_qbi_passive_passthrough_assumptions_payload",
    "calibrate_qbi_passive_log_odds_shift",
    "load_qbi_passive_passthrough_assumptions",
    "load_qbi_passive_passthrough_evidence",
    "qbi_passive_expected_aggregate",
    "us_qbi_passive_passthrough_contract_identity",
    "validate_qbi_passive_passthrough_assumptions",
    "with_us_qbi_passive_passthrough_assignment",
]

QBI_PASSIVE_EVIDENCE_RESOURCE = "qbi_passive_passthrough_v1.json"
QBI_PASSIVE_ASSUMPTIONS_RESOURCE = "qbi_passive_passthrough_assumptions_v1.json"
US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN = "passive_partnership_s_corp_income"
US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS: tuple[str, ...] = (
    "partnership_income",
    "s_corp_income",
    "rental_income",
    "estate_income",
)

_ASSIGNMENT_STAGE_VERSION = 1
_ASSUMPTIONS_SCHEMA_VERSION = 1
_DEFAULT_ROOT_SEED = 0
_RNG_FAMILY_ENTROPY = 4722
_PRESENCE_FAMILY = 0
_SHARE_FAMILY = 1
_SOLVER_ITERATIONS = 128
_SOLVER_LOWER_BOUND = -30.0
_SOLVER_UPPER_BOUND = 30.0
_PROBABILITY_CLIP = 1e-12
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _resource_bytes(name: str) -> bytes:
    return files("microcosm.build.us").joinpath(name).read_bytes()


def _resource_payload(name: str) -> dict[str, Any]:
    payload = json.loads(_resource_bytes(name))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object.")
    return payload


def _resource_sha256(name: str) -> str:
    return hashlib.sha256(_resource_bytes(name)).hexdigest()


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _number(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" and at least {minimum}"
        raise ValueError(f"{location} must be finite{suffix}.")
    return result


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer at least {minimum}.")
    return value


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string.")
    return value


def _digest(value: object, location: str) -> str:
    result = _text(value, location)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{location} must be a lowercase SHA-256 digest.")
    return result


def _finite_vector(value: object, location: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{location} must be one-dimensional.")
    if not np.isfinite(result).all():
        raise ValueError(f"{location} must be finite.")
    return result


def _aligned_vectors(**values: object) -> dict[str, np.ndarray]:
    arrays = {name: _finite_vector(value, name) for name, value in values.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Passive pass-through arrays must have equal lengths.")
    return arrays


def _income_band_index(values: np.ndarray) -> np.ndarray:
    return np.select(
        (
            values <= 0.0,
            values <= 25_000.0,
            values <= 100_000.0,
            values <= 250_000.0,
            values <= 1_000_000.0,
        ),
        (0, 1, 2, 3, 4),
        default=5,
    ).astype(np.int8)


def _logistic(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -700.0, 700.0)))


def _logit(values: np.ndarray | float) -> np.ndarray:
    array = np.clip(
        np.asarray(values, dtype=np.float64),
        _PROBABILITY_CLIP,
        1.0 - _PROBABILITY_CLIP,
    )
    return np.log(array / (1.0 - array))


def _evidence_model(
    evidence: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_qbi_passive_passthrough_resource(evidence)
    methodology = _mapping(evidence["methodology"], "methodology")
    probabilities = _finite_vector(
        methodology["quantile_probabilities"],
        "methodology.quantile_probabilities",
    )
    cells = {
        _text(cell["income_band"], "cells[].income_band"): cell
        for cell in evidence["cells"]
    }
    prevalence: list[float] = []
    quantiles: list[np.ndarray] = []
    for band in SCF_PASSIVE_INCOME_BANDS:
        cell = _mapping(cells[band], f"cells.{band}")
        holding = _mapping(
            cell["holding_prevalence"],
            f"cells.{band}.holding_prevalence",
        )
        share = _mapping(
            cell["conditional_share"],
            f"cells.{band}.conditional_share",
        )
        prevalence.append(
            _number(
                holding["estimate"],
                f"cells.{band}.holding_prevalence.estimate",
                minimum=0.0,
            )
        )
        selected_quantiles = _mapping(
            share["selected_quantiles"],
            f"cells.{band}.conditional_share.selected_quantiles",
        )
        quantiles.append(
            _finite_vector(
                [
                    selected_quantiles[name]
                    for name in ("q05", "q25", "q50", "q75", "q95")
                ],
                f"cells.{band}.conditional_share.selected_quantiles",
            )
        )
    prevalence_array = np.asarray(prevalence, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)
    if np.any(prevalence_array > 1.0):
        raise ValueError("SCF passive holding prevalence must lie in [0, 1].")
    if quantile_array.shape != (
        len(SCF_PASSIVE_INCOME_BANDS),
        len(probabilities),
    ):
        raise ValueError("SCF passive conditional-share quantiles are misaligned.")
    if np.any((quantile_array < 0.0) | (quantile_array > 1.0)):
        raise ValueError("SCF passive share quantiles must lie in [0, 1].")
    if np.any(np.diff(probabilities) <= 0.0):
        raise ValueError("SCF passive quantile probabilities must increase.")
    if probabilities[0] <= 0.0 or probabilities[-1] >= 1.0:
        raise ValueError("SCF passive quantile probabilities must lie in (0, 1).")
    if np.any(np.diff(quantile_array, axis=1) < 0.0):
        raise ValueError("SCF passive share quantiles must be nondecreasing.")
    return prevalence_array, probabilities, quantile_array


def _inverse_cdf_means(
    probabilities: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    left = quantiles[:, 0] * probabilities[0]
    middle = np.sum(
        (quantiles[:, :-1] + quantiles[:, 1:]) * np.diff(probabilities) / 2.0,
        axis=1,
    )
    right = quantiles[:, -1] * (1.0 - probabilities[-1])
    return left + middle + right


def _latent_form_eligibility(
    latent_entity_form: object | None,
    *,
    length: int,
) -> np.ndarray:
    if latent_entity_form is None:
        return np.ones(length, dtype=bool)
    forms = np.asarray(latent_entity_form)
    if forms.ndim != 1 or len(forms) != length:
        raise ValueError("latent_entity_form must align one-for-one with inputs.")
    if np.issubdtype(forms.dtype, np.number):
        numeric = np.asarray(forms, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("latent_entity_form numeric codes must be finite.")
        return np.isin(numeric, (1.0, 2.0))
    normalized = np.asarray(
        [str(value).strip().lower() for value in forms],
        dtype=object,
    )
    recognized = {
        "partnership",
        "partnership_or_llc",
        "s_corporation",
    }
    return np.isin(normalized, tuple(recognized))


def validate_qbi_passive_passthrough_assumptions(
    payload: Mapping[str, Any],
) -> None:
    """Fail closed on the persisted passive assignment contract."""

    required = {
        "schema_version",
        "resource",
        "provisional",
        "assignment_stage",
        "evidence",
        "random_streams",
        "calibration",
        "notes",
    }
    if set(payload) != required:
        raise ValueError(
            "Passive pass-through assumptions keys must be exactly "
            f"{sorted(required)}; got {sorted(payload)}."
        )
    if payload["schema_version"] != _ASSUMPTIONS_SCHEMA_VERSION:
        raise ValueError("Unsupported passive pass-through assumptions schema.")
    if payload["resource"] != "qbi_passive_passthrough_assumptions":
        raise ValueError("Unexpected passive pass-through assumptions resource.")
    if payload["provisional"] is not True:
        raise ValueError("Passive pass-through assumptions must remain provisional.")

    stage = _mapping(payload["assignment_stage"], "assignment_stage")
    if stage.get("version") != _ASSIGNMENT_STAGE_VERSION:
        raise ValueError("Unexpected passive assignment stage version.")
    if stage.get("architecture") != "sibling_before_qbi_reconciliation":
        raise ValueError("Unexpected passive assignment stage architecture.")
    if stage.get("output_column") != US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN:
        raise ValueError("Passive assignment output column does not match runtime.")
    if tuple(stage.get("person_input_columns", ())) != (
        US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS
    ):
        raise ValueError("Passive assignment input columns do not match runtime.")

    evidence = _mapping(payload["evidence"], "evidence")
    if evidence.get("resource") != QBI_PASSIVE_EVIDENCE_RESOURCE:
        raise ValueError("Passive assumptions name the wrong evidence resource.")
    if evidence.get("schema_version") != 1:
        raise ValueError("Passive assumptions require evidence schema 1.")
    _digest(evidence.get("sha256"), "evidence.sha256")

    streams = _mapping(payload["random_streams"], "random_streams")
    expected_streams = {
        "bit_generator": "PCG64",
        "root_seed_default": _DEFAULT_ROOT_SEED,
        "family_entropy": _RNG_FAMILY_ENTROPY,
        "presence_family": _PRESENCE_FAMILY,
        "share_family": _SHARE_FAMILY,
        "draw_policy": "full_length_before_support_masks",
    }
    if dict(streams) != expected_streams:
        raise ValueError("Passive assignment random-stream contract drifted.")

    calibration = _mapping(payload["calibration"], "calibration")
    bounds = _mapping(calibration.get("bounds"), "calibration.bounds")
    lower = _number(bounds.get("lower"), "calibration.bounds.lower", minimum=0.0)
    upper = _number(bounds.get("upper"), "calibration.bounds.upper", minimum=0.0)
    target = _mapping(
        calibration.get("provisional_target"),
        "calibration.provisional_target",
    )
    amount = _number(
        target.get("amount"),
        "calibration.provisional_target.amount",
        minimum=0.0,
    )
    if target.get("choice") != "midpoint_of_unresolved_bounds":
        raise ValueError("Passive provisional target choice must remain explicit.")
    if not math.isclose(amount, (lower + upper) / 2.0, abs_tol=1e-6):
        raise ValueError("Passive provisional target is not the bounds midpoint.")
    shift = _number(calibration.get("log_odds_shift"), "calibration.log_odds_shift")
    expected = _number(
        calibration.get("expected_aggregate"),
        "calibration.expected_aggregate",
        minimum=0.0,
    )
    if not math.isclose(expected, amount, rel_tol=1e-10, abs_tol=1.0):
        raise ValueError("Passive expected aggregate does not hit its target.")
    _ = shift
    tolerance = _number(
        calibration.get("replay_relative_tolerance"),
        "calibration.replay_relative_tolerance",
        minimum=0.0,
    )
    seeded = _mapping(calibration.get("seeded_replay"), "calibration.seeded_replay")
    achieved = _number(
        seeded.get("aggregate"),
        "calibration.seeded_replay.aggregate",
        minimum=0.0,
    )
    relative_error = _number(
        seeded.get("relative_error"),
        "calibration.seeded_replay.relative_error",
        minimum=0.0,
    )
    if seeded.get("seed") != _DEFAULT_ROOT_SEED:
        raise ValueError("Passive replay must use the committed default seed.")
    if not math.isclose(relative_error, abs(achieved / amount - 1.0), abs_tol=1e-12):
        raise ValueError("Passive seeded replay relative error is inconsistent.")
    if relative_error > tolerance:
        raise ValueError("Passive seeded replay exceeds its committed tolerance.")
    replay = _mapping(calibration.get("replay_artifact"), "replay_artifact")
    _text(replay.get("filename"), "replay_artifact.filename")
    _digest(replay.get("sha256"), "replay_artifact.sha256")
    _integer(replay.get("bytes"), "replay_artifact.bytes", minimum=1)
    _integer(replay.get("tax_unit_rows"), "replay_artifact.tax_unit_rows", minimum=1)
    _integer(replay.get("person_rows"), "replay_artifact.person_rows", minimum=1)
    _number(
        replay.get("person_weight_total"),
        "replay_artifact.person_weight_total",
        minimum=0.0,
    )
    _integer(
        replay.get("positive_passthrough_rows"),
        "replay_artifact.positive_passthrough_rows",
    )
    _number(
        replay.get("positive_passthrough_weighted_aggregate"),
        "replay_artifact.positive_passthrough_weighted_aggregate",
        minimum=0.0,
    )
    solver = _mapping(calibration.get("solver"), "calibration.solver")
    if solver.get("method") != "bisection_on_expected_weighted_aggregate":
        raise ValueError("Unexpected passive calibration solver.")
    if solver.get("iterations") != _SOLVER_ITERATIONS:
        raise ValueError("Unexpected passive calibration iteration count.")
    if solver.get("lower_bound") != _SOLVER_LOWER_BOUND:
        raise ValueError("Unexpected passive calibration lower bound.")
    if solver.get("upper_bound") != _SOLVER_UPPER_BOUND:
        raise ValueError("Unexpected passive calibration upper bound.")
    notes = payload["notes"]
    if not isinstance(notes, list) or not all(
        isinstance(note, str) and note.strip() for note in notes
    ):
        raise ValueError("Passive assumptions notes must be nonempty strings.")


def load_qbi_passive_passthrough_evidence() -> dict[str, Any]:
    payload = _resource_payload(QBI_PASSIVE_EVIDENCE_RESOURCE)
    validate_qbi_passive_passthrough_resource(payload)
    return payload


def load_qbi_passive_passthrough_assumptions() -> dict[str, Any]:
    payload = _resource_payload(QBI_PASSIVE_ASSUMPTIONS_RESOURCE)
    validate_qbi_passive_passthrough_assumptions(payload)
    expected_evidence_sha = payload["evidence"]["sha256"]
    actual_evidence_sha = _resource_sha256(QBI_PASSIVE_EVIDENCE_RESOURCE)
    if expected_evidence_sha != actual_evidence_sha:
        raise ValueError(
            "Passive assumptions evidence digest does not match the packaged resource."
        )
    return payload


def _assignment_inputs(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    *,
    evidence: Mapping[str, Any],
    log_odds_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = _aligned_vectors(
        partnership_s_corp_income=partnership_s_corp_income,
        schedule_e_income=schedule_e_income,
    )
    passthrough = np.maximum(arrays["partnership_s_corp_income"], 0.0)
    band_index = _income_band_index(arrays["schedule_e_income"])
    prevalence, probabilities, quantiles = _evidence_model(evidence)
    shifted = _logistic(_logit(prevalence) + float(log_odds_shift))
    return passthrough, band_index, shifted, probabilities, quantiles


def assign_passive_partnership_s_corp_income(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    *,
    seed: int = _DEFAULT_ROOT_SEED,
    latent_entity_form: object | None = None,
    evidence: Mapping[str, Any] | None = None,
    assumptions: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return one seeded passive amount per person.

    Presence and conditional-share generators draw over the complete input
    length before eligibility masks are applied.  Adding or removing support
    rows therefore cannot advance another random family.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Passive assignment seed must be a nonnegative integer.")
    evidence_payload = (
        load_qbi_passive_passthrough_evidence() if evidence is None else dict(evidence)
    )
    assumptions_payload = (
        load_qbi_passive_passthrough_assumptions()
        if assumptions is None
        else dict(assumptions)
    )
    validate_qbi_passive_passthrough_assumptions(assumptions_payload)
    shift = float(assumptions_payload["calibration"]["log_odds_shift"])
    passthrough, band_index, shifted, probabilities, quantiles = _assignment_inputs(
        partnership_s_corp_income,
        schedule_e_income,
        evidence=evidence_payload,
        log_odds_shift=shift,
    )
    length = len(passthrough)
    eligible = _latent_form_eligibility(latent_entity_form, length=length)
    presence_rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, _RNG_FAMILY_ENTROPY, _PRESENCE_FAMILY])
        )
    )
    share_rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, _RNG_FAMILY_ENTROPY, _SHARE_FAMILY])
        )
    )
    presence_draw = presence_rng.random(length)
    share_draw = share_rng.random(length)
    sampled_share = np.zeros(length, dtype=np.float64)
    for band in range(len(SCF_PASSIVE_INCOME_BANDS)):
        mask = band_index == band
        sampled_share[mask] = np.interp(
            share_draw[mask],
            probabilities,
            quantiles[band],
        )
    present = presence_draw < shifted[band_index]
    return np.where(
        eligible & (passthrough > 0.0) & present, passthrough * sampled_share, 0.0
    )


def qbi_passive_expected_aggregate(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    person_weights: object,
    *,
    evidence: Mapping[str, Any],
    log_odds_shift: float,
    latent_entity_form: object | None = None,
) -> float:
    arrays = _aligned_vectors(
        partnership_s_corp_income=partnership_s_corp_income,
        schedule_e_income=schedule_e_income,
        person_weights=person_weights,
    )
    if np.any(arrays["person_weights"] < 0.0):
        raise ValueError("Passive replay person weights must be nonnegative.")
    passthrough, band_index, shifted, probabilities, quantiles = _assignment_inputs(
        arrays["partnership_s_corp_income"],
        arrays["schedule_e_income"],
        evidence=evidence,
        log_odds_shift=log_odds_shift,
    )
    eligible = _latent_form_eligibility(
        latent_entity_form,
        length=len(passthrough),
    )
    expected_share = _inverse_cdf_means(probabilities, quantiles)
    values = (
        passthrough
        * shifted[band_index]
        * expected_share[band_index]
        * eligible.astype(np.float64)
    )
    return float(np.dot(arrays["person_weights"], values))


def calibrate_qbi_passive_log_odds_shift(
    partnership_s_corp_income: object,
    schedule_e_income: object,
    person_weights: object,
    *,
    evidence: Mapping[str, Any],
    target: float,
    latent_entity_form: object | None = None,
) -> tuple[float, float]:
    target_value = _number(target, "target", minimum=0.0)

    def aggregate(shift: float) -> float:
        return qbi_passive_expected_aggregate(
            partnership_s_corp_income,
            schedule_e_income,
            person_weights,
            evidence=evidence,
            log_odds_shift=shift,
            latent_entity_form=latent_entity_form,
        )

    lower = _SOLVER_LOWER_BOUND
    upper = _SOLVER_UPPER_BOUND
    lower_value = aggregate(lower)
    upper_value = aggregate(upper)
    if target_value < lower_value or target_value > upper_value:
        raise ValueError(
            "Passive aggregate target is unattainable inside the solver bounds: "
            f"target={target_value}, range=[{lower_value}, {upper_value}]."
        )
    for _ in range(_SOLVER_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        if aggregate(midpoint) < target_value:
            lower = midpoint
        else:
            upper = midpoint
    shift = (lower + upper) / 2.0
    return shift, aggregate(shift)


def build_qbi_passive_passthrough_assumptions_payload(
    *,
    evidence: Mapping[str, Any],
    evidence_sha256: str,
    partnership_s_corp_income: object,
    schedule_e_income: object,
    person_weights: object,
    replay_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve and persist the one-parameter provisional replay calibration."""

    validate_qbi_passive_passthrough_resource(evidence)
    evidence_digest = _digest(evidence_sha256, "evidence_sha256")
    external = _mapping(evidence["external_anchor"], "external_anchor")
    bounds = _mapping(
        external["passive_passthrough_bounds"],
        "external_anchor.passive_passthrough_bounds",
    )
    lower = _number(
        _mapping(bounds["lower"], "bounds.lower")["amount"],
        "bounds.lower.amount",
        minimum=0.0,
    )
    upper = _number(
        _mapping(bounds["upper"], "bounds.upper")["amount"],
        "bounds.upper.amount",
        minimum=0.0,
    )
    target = (lower + upper) / 2.0
    shift, expected = calibrate_qbi_passive_log_odds_shift(
        partnership_s_corp_income,
        schedule_e_income,
        person_weights,
        evidence=evidence,
        target=target,
    )
    arrays = _aligned_vectors(
        partnership_s_corp_income=partnership_s_corp_income,
        schedule_e_income=schedule_e_income,
        person_weights=person_weights,
    )

    provisional_payload: dict[str, Any] = {
        "schema_version": _ASSUMPTIONS_SCHEMA_VERSION,
        "resource": "qbi_passive_passthrough_assumptions",
        "provisional": True,
        "assignment_stage": {
            "version": _ASSIGNMENT_STAGE_VERSION,
            "architecture": "sibling_before_qbi_reconciliation",
            "versioning_decision": (
                "Use a version-1 sibling rather than QBI simulation version 4: "
                "the current renamed tree consumes 15 archived QBI leaves and "
                "does not carry the old opt-in simulation stack."
            ),
            "output_column": US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN,
            "person_input_columns": list(
                US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS
            ),
            "schedule_e_proxy": (
                "partnership_income + s_corp_income + rental_income + estate_income"
            ),
            "latent_form_treatment": (
                "When a v3 latent entity form is available it routes only "
                "partnership and S-corporation records into eligibility; the "
                "SCF probability and share remain band-only because active and "
                "non-active SCF businesses are not entity-linked."
            ),
        },
        "evidence": {
            "resource": QBI_PASSIVE_EVIDENCE_RESOURCE,
            "schema_version": 1,
            "sha256": evidence_digest,
        },
        "random_streams": {
            "bit_generator": "PCG64",
            "root_seed_default": _DEFAULT_ROOT_SEED,
            "family_entropy": _RNG_FAMILY_ENTROPY,
            "presence_family": _PRESENCE_FAMILY,
            "share_family": _SHARE_FAMILY,
            "draw_policy": "full_length_before_support_masks",
        },
        "calibration": {
            "bounds": {"lower": lower, "upper": upper},
            "provisional_target": {
                "amount": target,
                "choice": "midpoint_of_unresolved_bounds",
            },
            "log_odds_shift": shift,
            "expected_aggregate": expected,
            "replay_relative_tolerance": 0.05,
            # A valid placeholder permits the pure seeded assignment to consume
            # the otherwise complete assumptions object.  It is replaced below
            # by the realized full-artifact diagnostic before validation/return.
            "seeded_replay": {
                "seed": _DEFAULT_ROOT_SEED,
                "aggregate": target,
                "relative_error": 0.0,
                "positive_assigned_rows": 0,
            },
            "replay_artifact": {
                **dict(replay_artifact),
                "person_rows": len(arrays["partnership_s_corp_income"]),
                "person_weight_total": float(arrays["person_weights"].sum()),
                "positive_passthrough_rows": int(
                    np.count_nonzero(arrays["partnership_s_corp_income"] > 0.0)
                ),
                "positive_passthrough_weighted_aggregate": float(
                    np.dot(
                        arrays["person_weights"],
                        np.maximum(arrays["partnership_s_corp_income"], 0.0),
                    )
                ),
            },
            "solver": {
                "method": "bisection_on_expected_weighted_aggregate",
                "iterations": _SOLVER_ITERATIONS,
                "lower_bound": _SOLVER_LOWER_BOUND,
                "upper_bound": _SOLVER_UPPER_BOUND,
            },
        },
        "notes": [
            "The target is provisional because Form 8960 line 4c does not "
            "decompose passive pass-through income from rental and royalty income.",
            "Rental income remains unchanged: the current engine includes it "
            "fully in NIIT, and this stage assigns only the pass-through leaf.",
            "The output remains engine-inert under locked PolicyEngine-US "
            "1.764.6 because that registry does not expose the variable; the "
            "hard release gate stays red until the engine pin advances past "
            "1.764.6 to a release containing PolicyEngine-US #9306.",
            "The runtime does not read the restricted replay artifact or solve "
            "the calibration shift.",
            "The sibling stage neither invokes nor advances any archived QBI "
            "random stream and preserves all 15 existing QBI leaves.",
            "The TY2023 administrative target is applied to the pinned "
            "2024-shaped replay without a population-vintage backcast; this "
            "period mismatch is another reason the target remains provisional.",
        ],
    }
    assigned = assign_passive_partnership_s_corp_income(
        arrays["partnership_s_corp_income"],
        arrays["schedule_e_income"],
        seed=_DEFAULT_ROOT_SEED,
        evidence=evidence,
        assumptions=provisional_payload,
    )
    achieved = float(np.dot(arrays["person_weights"], assigned))
    provisional_payload["calibration"]["seeded_replay"] = {
        "seed": _DEFAULT_ROOT_SEED,
        "aggregate": achieved,
        "relative_error": abs(achieved / target - 1.0),
        "positive_assigned_rows": int(np.count_nonzero(assigned > 0.0)),
    }
    validate_qbi_passive_passthrough_assumptions(provisional_payload)
    return provisional_payload


def us_qbi_passive_passthrough_contract_identity() -> dict[str, Any]:
    """Return the content-addressed assignment identity for checkpoint pins."""

    assumptions = load_qbi_passive_passthrough_assumptions()
    identity: dict[str, Any] = {
        "version": 1,
        "assignment_stage_version": _ASSIGNMENT_STAGE_VERSION,
        "output_column": US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN,
        "person_input_columns": list(US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS),
        "evidence_resource": QBI_PASSIVE_EVIDENCE_RESOURCE,
        "evidence_sha256": _resource_sha256(QBI_PASSIVE_EVIDENCE_RESOURCE),
        "assumptions_resource": QBI_PASSIVE_ASSUMPTIONS_RESOURCE,
        "assumptions_sha256": _resource_sha256(QBI_PASSIVE_ASSUMPTIONS_RESOURCE),
        "log_odds_shift": assumptions["calibration"]["log_odds_shift"],
        "random_streams": dict(assumptions["random_streams"]),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    identity["sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def with_us_qbi_passive_passthrough_assignment(
    frame: Frame,
    *,
    seed: int = _DEFAULT_ROOT_SEED,
) -> Frame:
    """Add the passive pass-through person input without touching QBI leaves."""

    person = frame.table("person")
    missing = sorted(
        column
        for column in US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS
        if column not in person
    )
    if missing:
        raise ValueError(
            f"Passive pass-through assignment requires person input(s): {missing}."
        )
    if US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN in person:
        raise ValueError(
            "Passive pass-through assignment refuses to overwrite its owned "
            "output column."
        )
    numeric = person.loc[
        :, list(US_QBI_PASSIVE_PASSTHROUGH_PERSON_INPUT_COLUMNS)
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Passive pass-through assignment inputs must be finite.")
    passthrough = numeric["partnership_income"].to_numpy(dtype=np.float64) + numeric[
        "s_corp_income"
    ].to_numpy(dtype=np.float64)
    schedule_e = (
        passthrough
        + numeric["rental_income"].to_numpy(dtype=np.float64)
        + numeric["estate_income"].to_numpy(dtype=np.float64)
    )
    assigned = assign_passive_partnership_s_corp_income(
        passthrough,
        schedule_e,
        seed=seed,
    )
    result = person.copy()
    result[US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN] = assigned
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({link: frame.link(link).copy() for link in frame.links})
    tables["person"] = result
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
