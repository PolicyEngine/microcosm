"""Reporter-anchored Bernoulli SSI take-up at documented age-band priors.

The retired eCPS exported ``takes_up_ssi_if_eligible`` after preserving every
CPS ASEC ``SSI_VAL > 0`` reporter and filling additional people to a scalar
50 percent rate.  That rate cannot be retained: its only citation estimates
participation among adults age 65 or older, while the retired code applied it
to children and working-age disabled people too.

This stage keeps the source-backed half of that method (the reporter anchor)
and replaces the scope-invalid rate with per-band priors derived from SSA's
December 2024 counts of people receiving a *federal payment*, split into the
three published age bands.  Each prior is the band target over the weighted
PolicyEngine-US ``uncapped_ssi > 0`` candidate capacity — falling back to the
observed reporter share of capacity when that ratio reaches one, so the flag
never degenerates to a constant.  Every source person draws once against the
band prior, seeded and stable per ``person_source_id``, with the decision
fanned to every actual support row; direct ASEC reporters stay true
unconditionally.  There is no count matching here (populace#469): the SSA
band counts bind only as ordinary calibration registry targets
(populace#470), the caller passes those same registry values in as
``targets``, and any post-calibration miss ships in the release scorecard
like every other program's.

The caller supplies December ``uncapped_ssi`` because the fiscal builder owns
PolicyEngine simulations and batching.  Assignment is recomputed from the
supplied frame weights, fixed once before target materialization, and later
diagnosed on release weights without rewriting the persisted decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "SSI_TAKE_UP_ARCHIVED_DERIVATION_URL",
    "SSI_TAKE_UP_ARCHIVED_EXPORT_URL",
    "SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL",
    "SSI_TAKE_UP_ARCHIVED_REPORTER_URL",
    "SSI_TAKE_UP_ARCHIVED_TARGETS_URL",
    "SSI_TAKE_UP_SSA_SOURCE_URL",
    "SSITakeUpAgeTarget",
    "US_SSI_TAKE_UP_AGE_TARGETS",
    "US_SSI_TAKE_UP_ANCHOR",
    "US_SSI_TAKE_UP_NONCONSTANT_PERSON_COLUMNS",
    "US_SSI_TAKE_UP_OUTPUT_COLUMNS",
    "US_SSI_TAKE_UP_REQUIRED_SOURCE_COLUMNS",
    "US_SSI_TAKE_UP_STAGE_NAME",
    "US_SSI_TAKE_UP_TARGET_TABLE_NAME",
    "us_ssi_take_up_diagnostics",
    "us_ssi_take_up_gate",
    "us_ssi_take_up_reporter_source_ids",
    "us_ssi_take_up_stage_spec",
    "with_us_ssi_take_up",
    "write_us_ssi_take_up_diagnostics",
]

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_RETIRED_REPOSITORY = "policyengine-" + "us-data"
_RETIRED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_RETIRED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_RETIRED_PACKAGE}/"
)
SSI_TAKE_UP_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/cps/cps.py#L650-L657"
SSI_TAKE_UP_ARCHIVED_EXPORT_URL = _ARCHIVED_ROOT + "datasets/cps/cps.py#L1497-L1499"
SSI_TAKE_UP_ARCHIVED_REPORTER_URL = _ARCHIVED_ROOT + "datasets/cps/cps.py#L584"
SSI_TAKE_UP_ARCHIVED_RANDOMNESS_URL = _ARCHIVED_ROOT + "datasets/cps/takeup.py#L10-L35"
SSI_TAKE_UP_ARCHIVED_TARGETS_URL = _ARCHIVED_ROOT + "utils/ssi_targets.py#L41-L74"
SSI_TAKE_UP_SSA_SOURCE_URL = (
    "https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-12/table01.html"
)

US_SSI_TAKE_UP_STAGE_NAME = "ssi_take_up"
US_SSI_TAKE_UP_TARGET_TABLE_NAME = "ssa_ssi_federal_payment_recipients_by_age"
US_SSI_TAKE_UP_ANCHOR = "SSI_VAL"
US_SSI_TAKE_UP_OUTPUT_COLUMNS: tuple[str, ...] = ("takes_up_ssi_if_eligible",)
US_SSI_TAKE_UP_NONCONSTANT_PERSON_COLUMNS = US_SSI_TAKE_UP_OUTPUT_COLUMNS
US_SSI_TAKE_UP_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "age",
    US_SSI_TAKE_UP_ANCHOR,
    "person_source_id",
    "person_support_channel",
)

_OUTPUT = US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]
_SOURCE_ID = "person_source_id"
_SUPPORT_CHANNEL = "person_support_channel"
_ASEC_CHANNEL = "asec"
_PUF_CHANNEL = "puf_tax_detail"
_KNOWN_CHANNELS = frozenset((_ASEC_CHANNEL, _PUF_CHANNEL))
_TARGET_PERIOD = "2024-12"
_TARGET_MEASURE = "Total with—Federal payment"
_CANDIDATE_DEFINITION = "uncapped_ssi > 0 at 2024-12"
_WEIGHTS_BASIS = "current_frame_resolved_person_weights"
# Version 2 (populace#469): count-matching-era fields (reachable_goal) are
# gone, band rows split the prior into assignment_prior (the value that
# generated the frozen flags) and prior_recomputed_from_current_weights, and
# the top level carries bernoulli_law_violation_count.
_DIAGNOSTICS_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SSITakeUpAgeTarget:
    """One disjoint SSA federal-payment recipient age stratum.

    Band structure only: the recipient counts themselves live in the ledger
    (``ssa_ssi...by_age`` facts) and reach this stage as caller-supplied
    ``targets`` read from the calibration registry (populace#469/#470), never
    as module constants.
    """

    key: str
    minimum_age: int | None
    maximum_age: int | None
    label: str

    def contains(self, age: np.ndarray) -> np.ndarray:
        result = np.ones(len(age), dtype=bool)
        if self.minimum_age is not None:
            result &= age >= self.minimum_age
        if self.maximum_age is not None:
            result &= age <= self.maximum_age
        return result


US_SSI_TAKE_UP_AGE_TARGETS: tuple[SSITakeUpAgeTarget, ...] = (
    SSITakeUpAgeTarget("under_18", None, 17, "Under 18"),
    SSITakeUpAgeTarget("18_64", 18, 64, "Ages 18–64"),
    SSITakeUpAgeTarget("65_plus", 65, None, "Ages 65+"),
)

_READ_PARAMETERS: dict[str, object] = {
    "table": "person",
    "weight": "person_weight",
}
_ASSIGN_PARAMETERS: dict[str, object] = {
    "output": _OUTPUT,
    "draw": "stable_source_person_draw",
    "rate_key": "ssi_age_band_count_prior",
    "rate_column": "ssi_take_up_assignment_prior",
    "reported_true_anchor": "SSI_VAL > 0",
    "assignment_unit": _SOURCE_ID,
    "fan_to_support_clones": True,
    "age_bands": {
        "under_18": "age < 18",
        "18_64": "18 <= age < 65",
        "65_plus": "age >= 65",
    },
    "rate_derivation": (
        "band_target / weighted_candidate_capacity(uncapped_ssi > 0); "
        "min(reporter_candidate_floor / capacity, 1) once that ratio "
        "reaches one"
    ),
    "rate_target_role": "ssa_ssi_age_band_recipients",
    "target_source": SSI_TAKE_UP_SSA_SOURCE_URL,
    "target_period": _TARGET_PERIOD,
    "target_measure": _TARGET_MEASURE,
}


def us_ssi_take_up_stage_spec() -> SourceStageSpec:
    """Load and strictly validate the packaged source-stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SSI_TAKE_UP_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SSI_TAKE_UP_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SSI_TAKE_UP_STAGE_NAME]
    if spec.grain != "person":
        raise ValueError("US SSI take-up stage must have person grain.")
    if tuple(spec.outputs) != US_SSI_TAKE_UP_OUTPUT_COLUMNS:
        raise ValueError("US SSI take-up manifest outputs drifted from runtime.")
    expected_kinds = [
        "read_table",
        "assign_binary_from_rate",
    ]
    if [operation.kind for operation in spec.operations] != expected_kinds:
        raise ValueError(
            "US SSI take-up stage must declare read then one seeded Bernoulli "
            "assignment; the SSA band counts bind only as calibration registry "
            "targets (populace#469/#470)."
        )
    expected_parameters = (
        _READ_PARAMETERS,
        _ASSIGN_PARAMETERS,
    )
    for operation, expected in zip(spec.operations, expected_parameters, strict=True):
        if dict(operation.parameters) != expected:
            raise ValueError(
                f"US SSI take-up {operation.kind} contract drifted from runtime."
            )
    target_artifacts = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("source") == SSI_TAKE_UP_SSA_SOURCE_URL
    ]
    if len(target_artifacts) != 1:
        raise ValueError("US SSI take-up stage must pin exactly one SSA table.")
    if target_artifacts[0].get("vintage") != _TARGET_PERIOD:
        raise ValueError("US SSI take-up SSA table vintage drifted from runtime.")
    if "target_values" in target_artifacts[0]:
        raise ValueError(
            "US SSI take-up must not hardcode SSA recipient counts; they enter "
            "as ledger-fed calibration registry targets (populace#469/#470)."
        )
    return spec


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _age_band_values(age: np.ndarray) -> np.ndarray:
    bands = np.full(len(age), "", dtype=object)
    for target in US_SSI_TAKE_UP_AGE_TARGETS:
        bands[target.contains(age)] = target.key
    if (bands == "").any():  # pragma: no cover - disjoint bands cover finite ages
        raise ValueError("US SSI take-up could not classify every age.")
    return bands


def _normalize_targets(targets: Mapping[str, float]) -> dict[str, float]:
    expected_keys = tuple(target.key for target in US_SSI_TAKE_UP_AGE_TARGETS)
    actual = {str(key): float(value) for key, value in targets.items()}
    if set(actual) != set(expected_keys):
        raise ValueError(
            "US SSI take-up targets require exactly age bands "
            f"{list(expected_keys)}; got {sorted(actual)}."
        )
    invalid = {
        key: value
        for key, value in actual.items()
        if not np.isfinite(value) or value <= 0
    }
    if invalid:
        raise ValueError(
            f"US SSI take-up targets must be finite and positive: {invalid}."
        )
    return {key: actual[key] for key in expected_keys}


def _stable_source_draw(source_id: str, *, seed: int) -> float:
    value = int.from_bytes(
        hashlib.blake2b(
            f"{seed}:{_OUTPUT}:{source_id}".encode(),
            digest_size=8,
        ).digest(),
        byteorder="big",
        signed=False,
    )
    return value / float(2**64)


def us_ssi_take_up_reporter_source_ids(frame: Frame) -> frozenset[str]:
    """Capture direct ASEC SSI reporter lineage before support pruning."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SSI take-up requires the US schema.")
    person = frame.table("person")
    required = {_SOURCE_ID, _SUPPORT_CHANNEL, US_SSI_TAKE_UP_ANCHOR}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"US SSI take-up reporter lineage missing person column(s): {missing}."
        )
    if person[_SOURCE_ID].isna().any() or person[_SUPPORT_CHANNEL].isna().any():
        raise ValueError("US SSI take-up reporter lineage requires provenance.")
    source_ids = _decoded_strings(person[_SOURCE_ID])
    channels = _decoded_strings(person[_SUPPORT_CHANNEL])
    reported = pd.to_numeric(person[US_SSI_TAKE_UP_ANCHOR], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if source_ids.str.strip().eq("").any() or not np.isfinite(reported).all():
        raise ValueError(
            "US SSI take-up reporter lineage requires nonblank identities and "
            "finite SSI_VAL values."
        )
    reporter_ids = frozenset(
        source_ids[(channels == _ASEC_CHANNEL).to_numpy() & (reported > 0.0)]
    )
    if not reporter_ids:
        raise ValueError("US SSI take-up found no direct ASEC SSI reporters.")
    return reporter_ids


def _source_table(
    frame: Frame,
    *,
    uncapped_ssi: np.ndarray,
    seed: int,
    reporter_source_ids: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row- and source-person-grain tables for assignment."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SSI take-up requires the US schema.")
    person = frame.table("person")
    missing = sorted(set(US_SSI_TAKE_UP_REQUIRED_SOURCE_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(f"US SSI take-up missing person source column(s): {missing}.")
    if len(uncapped_ssi) != len(person):
        raise ValueError(
            "US SSI take-up uncapped_ssi must align with person rows: "
            f"{len(person)} rows, {len(uncapped_ssi)} values."
        )

    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    reported = pd.to_numeric(person[US_SSI_TAKE_UP_ANCHOR], errors="coerce").to_numpy(
        dtype=np.float64
    )
    potential = np.asarray(uncapped_ssi, dtype=np.float64)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("US SSI take-up ages must be finite and nonnegative.")
    if not np.isfinite(reported).all():
        raise ValueError("US SSI take-up SSI_VAL anchors must be finite.")
    if not np.isfinite(potential).all():
        raise ValueError("US SSI take-up uncapped_ssi values must be finite.")
    if not (np.isfinite(weights) & (weights >= 0)).all() or weights.sum() <= 0:
        raise ValueError(
            "US SSI take-up requires finite nonnegative person weights with "
            "positive total."
        )
    if person[_SOURCE_ID].isna().any() or person[_SUPPORT_CHANNEL].isna().any():
        raise ValueError("US SSI take-up requires complete support provenance.")

    source_ids = _decoded_strings(person[_SOURCE_ID])
    channels = _decoded_strings(person[_SUPPORT_CHANNEL])
    if source_ids.str.strip().eq("").any():
        raise ValueError("US SSI take-up source identities must be nonblank.")
    observed_channels = set(channels.unique())
    if observed_channels != set(_KNOWN_CHANNELS):
        raise ValueError(
            "US SSI take-up requires exact ASEC/PUF support channels; "
            f"missing {sorted(_KNOWN_CHANNELS - observed_channels)}, "
            f"unsupported {sorted(observed_channels - _KNOWN_CHANNELS)}."
        )

    direct_anchor = (reported > 0.0) & channels.eq(_ASEC_CHANNEL).to_numpy()
    if reporter_source_ids is None:
        anchored_source_ids = frozenset(source_ids[direct_anchor])
    else:
        anchored_source_ids = frozenset(str(value) for value in reporter_source_ids)
        if not anchored_source_ids:
            raise ValueError("US SSI take-up reporter lineage cannot be empty.")
        if any(not value.strip() for value in anchored_source_ids):
            raise ValueError("US SSI take-up reporter source identities are nonblank.")
        omitted_direct = sorted(set(source_ids[direct_anchor]) - anchored_source_ids)
        if omitted_direct:
            raise ValueError(
                "US SSI take-up reporter lineage omitted direct ASEC anchors; "
                f"examples {omitted_direct[:5]}."
            )

    rows = pd.DataFrame(
        {
            "source_id": source_ids.to_numpy(),
            "channel": channels.to_numpy(),
            "age": age,
            "age_band": _age_band_values(age),
            "weight": weights,
            "candidate": potential > 0.0,
            # Capture lineage on the full support before L0. When pruning keeps
            # only a PUF clone, the explicit source-ID set still preserves the
            # underlying direct ASEC measurement without promoting PUF-only
            # SSI_VAL copies into independent anchors.
            "anchor": source_ids.isin(anchored_source_ids).to_numpy(),
        },
        index=person.index,
    )
    duplicated_channel = rows.duplicated(["source_id", "channel"], keep=False)
    if duplicated_channel.any():
        examples = (
            rows.loc[duplicated_channel, ["source_id", "channel"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "US SSI take-up requires at most one row per source identity and "
            f"support channel; duplicate examples {examples}."
        )
    rows["candidate_weight"] = rows["weight"].where(rows["candidate"], 0.0)

    source = (
        rows.groupby("source_id", sort=True)
        .agg(
            age_band=("age_band", "first"),
            age_band_count=("age_band", "nunique"),
            age_count=("age", "nunique"),
            candidate_weight=("candidate_weight", "sum"),
            total_weight=("weight", "sum"),
            anchor=("anchor", "any"),
            row_count=("source_id", "size"),
        )
        .copy()
    )
    inconsistent = source.index[source["age_band_count"] != 1].tolist()
    if inconsistent:
        raise ValueError(
            "US SSI take-up source identities cross SSA age bands; examples "
            f"{inconsistent[:5]}."
        )
    age_inconsistent = source.index[source["age_count"] != 1].tolist()
    if age_inconsistent:
        raise ValueError(
            "US SSI take-up support rows disagree on source-person age; "
            f"examples {age_inconsistent[:5]}."
        )
    source["draw"] = [
        _stable_source_draw(str(source_id), seed=int(seed))
        for source_id in source.index
    ]
    return rows, source


def _band_prior(target: float, capacity: float, reporter_floor: float) -> float:
    """Return the documented Bernoulli prior for one SSA age band.

    The target/capacity ratio is a meaningful take-up propensity only while
    it subsamples (capacity > target). Once it reaches one it would flag the
    whole band — a constant, signal-free output — so the prior falls back to
    the observed take-up rate among today's candidates: reporter mass over
    candidate capacity. Reform-created eligibles then take up at the rate
    today's modeled eligibles are observed reporting.
    """

    if capacity <= 0:
        return 0.0
    count_ratio = target / capacity
    if count_ratio < 1.0:
        return count_ratio
    return min(reporter_floor / capacity, 1.0)


def _age_band_diagnostics(
    source: pd.DataFrame,
    selected: pd.Series,
    *,
    targets: Mapping[str, float],
    assignment_priors: Mapping[str, float],
) -> list[dict[str, object]]:
    """Summarize one existing source-grain assignment on current weights.

    ``assignment_priors`` are the Bernoulli priors that generated the
    persisted flags. Each band row publishes them verbatim next to
    ``prior_recomputed_from_current_weights`` so release-weight measurements
    never misdocument the one-shot assignment (populace#469; PR #477 review
    finding 4).
    """

    bands: list[dict[str, object]] = []
    for target_definition in US_SSI_TAKE_UP_AGE_TARGETS:
        key = target_definition.key
        target = float(targets[key])
        in_band = source["age_band"].eq(key)
        candidate = in_band & source["candidate_weight"].gt(0.0)
        anchored = in_band & source["anchor"].astype(bool)
        capacity = float(source.loc[candidate, "candidate_weight"].sum())
        reporter_floor = float(
            source.loc[candidate & anchored, "candidate_weight"].sum()
        )
        selected_recipient_weight = float(
            source.loc[candidate & selected, "candidate_weight"].sum()
        )
        max_source_weight = (
            float(source.loc[candidate, "candidate_weight"].max())
            if candidate.any()
            else 0.0
        )
        bands.append(
            {
                "age_band": key,
                "label": target_definition.label,
                "target": target,
                "source_identity_count": int(in_band.sum()),
                "candidate_source_identity_count": int(candidate.sum()),
                "reporter_source_identity_count": int(anchored.sum()),
                "candidate_capacity": capacity,
                "reporter_candidate_floor": reporter_floor,
                "selected_recipient_weight": selected_recipient_weight,
                "signed_target_error": selected_recipient_weight - target,
                "target_shortfall": max(target - selected_recipient_weight, 0.0),
                "anchor_excess": max(reporter_floor - target, 0.0),
                "saturated": bool(capacity < target),
                "assignment_prior": float(assignment_priors[key]),
                "prior_recomputed_from_current_weights": _band_prior(
                    target, capacity, reporter_floor
                ),
                "max_source_candidate_weight": max_source_weight,
            }
        )
    return bands


def _bernoulli_law_violations(
    source: pd.DataFrame,
    selected: pd.Series,
    assignment_priors: Mapping[str, float],
) -> int:
    """Count source identities whose flag breaks the seeded Bernoulli law.

    The law is exact and weight-free: a source person is selected iff
    anchored or its stable draw fell below the band's assignment-time prior.
    Recomputing it against persisted flags catches any post-assignment
    corruption of the frozen decisions (populace#469; PR #477 review
    finding 3).
    """

    priors = source["age_band"].map(dict(assignment_priors)).to_numpy(dtype=np.float64)
    expected = source["anchor"].to_numpy(dtype=bool) | (
        source["draw"].to_numpy(dtype=np.float64) < priors
    )
    return int(np.count_nonzero(expected != selected.to_numpy(dtype=bool)))


def _normalize_assignment_priors(
    assignment_priors: Mapping[str, float],
) -> dict[str, float]:
    expected_keys = tuple(band.key for band in US_SSI_TAKE_UP_AGE_TARGETS)
    actual = {str(key): float(value) for key, value in assignment_priors.items()}
    if set(actual) != set(expected_keys):
        raise ValueError(
            "US SSI take-up assignment priors require exactly age bands "
            f"{list(expected_keys)}; got {sorted(actual)}."
        )
    invalid = {
        key: value
        for key, value in actual.items()
        if not np.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(
            f"US SSI take-up assignment priors must lie in [0, 1]: {invalid}."
        )
    return {key: actual[key] for key in expected_keys}


def _assign_sources(
    source: pd.DataFrame,
    *,
    targets: Mapping[str, float],
) -> tuple[pd.Series, list[dict[str, object]], dict[str, float]]:
    """Assign one flag per source identity; return diagnostics and priors."""

    selected = pd.Series(False, index=source.index, dtype=bool)
    priors: dict[str, float] = {}
    for target_definition in US_SSI_TAKE_UP_AGE_TARGETS:
        key = target_definition.key
        target = float(targets[key])
        in_band = source["age_band"].eq(key)
        candidate = in_band & source["candidate_weight"].gt(0.0)
        anchored = in_band & source["anchor"].astype(bool)
        capacity = float(source.loc[candidate, "candidate_weight"].sum())
        reporter_floor = float(
            source.loc[candidate & anchored, "candidate_weight"].sum()
        )
        prior = _band_prior(target, capacity, reporter_floor)
        priors[key] = prior

        # Seeded Bernoulli at the documented band prior for everyone in the
        # band — candidates and reform-created eligibles alike — with survey
        # reporters anchored unconditionally (populace#469). No count
        # matching: the SSA band counts are ordinary calibration targets
        # (populace#470) and the post-calibration miss ships in the
        # scorecard like every other program's.
        selected.loc[in_band] = source.loc[in_band, "draw"].to_numpy() < prior
        selected.loc[anchored] = True

    bands = _age_band_diagnostics(
        source, selected, targets=targets, assignment_priors=priors
    )
    return selected, bands, priors


def _diagnostics(
    frame: Frame,
    *,
    rows: pd.DataFrame,
    source: pd.DataFrame,
    assigned: np.ndarray,
    bands: list[dict[str, object]],
    targets: Mapping[str, float],
    law_violation_count: int,
) -> dict[str, object]:
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    reporter_lost = int(np.count_nonzero(rows["anchor"].to_numpy() & ~assigned))
    by_source = pd.DataFrame(
        {"source_id": rows["source_id"].to_numpy(), "assigned": assigned}
    )
    mismatches = int(
        (by_source.groupby("source_id", sort=False)["assigned"].nunique() > 1).sum()
    )
    channel_diagnostics: dict[str, dict[str, object]] = {}
    for channel in sorted(_KNOWN_CHANNELS):
        mask = rows["channel"].eq(channel).to_numpy()
        channel_weight = float(weights[mask].sum())
        channel_diagnostics[channel] = {
            "rows": int(mask.sum()),
            "unique_count": int(np.unique(assigned[mask]).size),
            "weighted_true_share": (
                float(weights[mask & assigned].sum()) / channel_weight
                if channel_weight > 0
                else 0.0
            ),
        }
    total_weight = float(weights.sum())
    selected_total = float(
        sum(float(band["selected_recipient_weight"]) for band in bands)
    )
    target_total = float(sum(targets.values()))
    return {
        "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
        "classification": "release_diagnostics",
        "issues": ["PolicyEngine/populace#312"],
        "variable": _OUTPUT,
        "anchor": US_SSI_TAKE_UP_ANCHOR,
        "anchor_channel": _ASEC_CHANNEL,
        "candidate_definition": _CANDIDATE_DEFINITION,
        "target_table": US_SSI_TAKE_UP_TARGET_TABLE_NAME,
        "target_source": SSI_TAKE_UP_SSA_SOURCE_URL,
        "target_period": _TARGET_PERIOD,
        "target_measure": _TARGET_MEASURE,
        "weights_basis": _WEIGHTS_BASIS,
        "target_total": target_total,
        "selected_recipient_weight_total": selected_total,
        "target_shortfall_total": float(
            sum(float(band["target_shortfall"]) for band in bands)
        ),
        "source_identity_count": int(len(source)),
        "weighted_flag_true_count": float(weights[assigned].sum()),
        "weighted_flag_universe": total_weight,
        "weighted_flag_true_share": (
            float(weights[assigned].sum()) / total_weight if total_weight > 0 else 0.0
        ),
        "unique_count": int(np.unique(assigned).size),
        "missing_or_invalid_count": 0,
        "reporter_anchor_lost_count": reporter_lost,
        "source_identity_mismatch_count": mismatches,
        "bernoulli_law_violation_count": int(law_violation_count),
        "channel_diagnostics": channel_diagnostics,
        "age_bands": bands,
    }


def with_us_ssi_take_up(
    frame: Frame,
    *,
    uncapped_ssi: np.ndarray,
    seed: int,
    targets: Mapping[str, float],
    reporter_source_ids: Collection[str] | None = None,
) -> tuple[Frame, dict[str, object]]:
    """Recompute SSI take-up and return the frame plus band diagnostics.

    ``targets`` carries the SSA band recipient counts the caller read from
    the calibration registry (role ``ssa_ssi_age_band_recipients``); they set
    the Bernoulli priors here and bind as ordinary calibration targets
    downstream (populace#469/#470).
    """

    us_ssi_take_up_stage_spec()
    normalized_targets = _normalize_targets(targets)
    rows, source = _source_table(
        frame,
        uncapped_ssi=np.asarray(uncapped_ssi, dtype=np.float64),
        seed=int(seed),
        reporter_source_ids=reporter_source_ids,
    )
    selected, bands, priors = _assign_sources(source, targets=normalized_targets)
    assigned = rows["source_id"].map(selected).to_numpy(dtype=bool)
    diagnostics = _diagnostics(
        frame,
        rows=rows,
        source=source,
        assigned=assigned,
        bands=bands,
        targets=normalized_targets,
        law_violation_count=_bernoulli_law_violations(source, selected, priors),
    )

    person = frame.table("person")
    if _OUTPUT in person:
        current = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            pd.api.types.is_bool_dtype(person[_OUTPUT].dtype)
            and not person[_OUTPUT].isna().any()
            and np.isfinite(current).all()
            and np.isin(current, [0.0, 1.0]).all()
            and np.array_equal(current.astype(bool), assigned)
        ):
            return frame, diagnostics

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_OUTPUT] = assigned
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    return result, diagnostics


def us_ssi_take_up_diagnostics(
    frame: Frame,
    *,
    uncapped_ssi: np.ndarray,
    seed: int,
    targets: Mapping[str, float],
    assignment_priors: Mapping[str, float],
    reporter_source_ids: Collection[str] | None = None,
) -> dict[str, object]:
    """Diagnose a persisted assignment without changing its decisions.

    The fiscal builder assigns take-up once before target materialization and
    publishes these measurements of the frozen flags on the final release
    weights. ``assignment_priors`` are the per-band priors the assignment
    stage documented (its diagnostics' ``assignment_prior`` fields): they are
    republished verbatim and every persisted flag is re-verified against the
    seeded Bernoulli law they define, so silent post-assignment corruption
    fails the gate. Any gap between the measured recipient mass and the SSA
    band targets is calibration's residual and ships in the scorecard — it
    is reported here, never corrected here.
    """

    us_ssi_take_up_stage_spec()
    normalized_targets = _normalize_targets(targets)
    normalized_priors = _normalize_assignment_priors(assignment_priors)
    person = frame.table("person")
    if _OUTPUT not in person:
        raise ValueError(f"US SSI take-up diagnostics require person.{_OUTPUT}.")
    output = person[_OUTPUT]
    if not pd.api.types.is_bool_dtype(output.dtype) or output.isna().any():
        raise ValueError("US SSI take-up diagnostics require a complete boolean flag.")
    assigned = output.to_numpy(dtype=bool)
    rows, source = _source_table(
        frame,
        uncapped_ssi=np.asarray(uncapped_ssi, dtype=np.float64),
        seed=int(seed),
        reporter_source_ids=reporter_source_ids,
    )
    source_assignment = (
        pd.DataFrame({"source_id": rows["source_id"].to_numpy(), "assigned": assigned})
        .groupby("source_id", sort=True)["assigned"]
        .first()
        .reindex(source.index)
        .astype(bool)
    )
    bands = _age_band_diagnostics(
        source,
        source_assignment,
        targets=normalized_targets,
        assignment_priors=normalized_priors,
    )
    return _diagnostics(
        frame,
        rows=rows,
        source=source,
        assigned=assigned,
        bands=bands,
        targets=normalized_targets,
        law_violation_count=_bernoulli_law_violations(
            source, source_assignment, normalized_priors
        ),
    )


def us_ssi_take_up_gate(
    diagnostics: Mapping[str, object],
    *,
    targets: Mapping[str, float],
) -> GateResult:
    """Require source-faithful anchors and documented Bernoulli priors by age."""

    expected_targets = _normalize_targets(targets)
    failures: list[str] = []
    if diagnostics.get("schema_version") != _DIAGNOSTICS_SCHEMA_VERSION:
        failures.append("SSI take-up diagnostics schema version is invalid.")
    if diagnostics.get("classification") != "release_diagnostics":
        failures.append("SSI take-up diagnostics classification is invalid.")
    if diagnostics.get("variable") != _OUTPUT:
        failures.append("SSI take-up diagnostics name the wrong output variable.")
    if diagnostics.get("anchor") != US_SSI_TAKE_UP_ANCHOR:
        failures.append("SSI take-up diagnostics name the wrong reporter anchor.")
    if diagnostics.get("anchor_channel") != _ASEC_CHANNEL:
        failures.append("SSI take-up diagnostics name the wrong anchor channel.")
    if diagnostics.get("candidate_definition") != _CANDIDATE_DEFINITION:
        failures.append("SSI take-up diagnostics carry the wrong candidate definition.")
    if diagnostics.get("target_table") != US_SSI_TAKE_UP_TARGET_TABLE_NAME:
        failures.append("SSI take-up diagnostics carry the wrong SSA target table.")
    if diagnostics.get("target_source") != SSI_TAKE_UP_SSA_SOURCE_URL:
        failures.append("SSI take-up diagnostics carry the wrong SSA target source.")
    if diagnostics.get("target_period") != _TARGET_PERIOD:
        failures.append("SSI take-up diagnostics carry the wrong target period.")
    if diagnostics.get("target_measure") != _TARGET_MEASURE:
        failures.append("SSI take-up diagnostics carry the wrong target measure.")
    if diagnostics.get("weights_basis") != _WEIGHTS_BASIS:
        failures.append("SSI take-up diagnostics carry the wrong weights basis.")
    if int(diagnostics.get("missing_or_invalid_count", -1)) != 0:
        failures.append("SSI take-up output contains missing or invalid values.")
    if int(diagnostics.get("unique_count", 0)) < 2:
        failures.append("SSI take-up output is constant and carries no signal.")
    if int(diagnostics.get("reporter_anchor_lost_count", -1)) != 0:
        failures.append("SSI take-up lost one or more direct ASEC reporter anchors.")
    if int(diagnostics.get("source_identity_mismatch_count", -1)) != 0:
        failures.append("SSI take-up support rows disagree within a source identity.")
    if int(diagnostics.get("bernoulli_law_violation_count", -1)) != 0:
        failures.append(
            "SSI take-up persisted flags violate the seeded Bernoulli law "
            "(anchored, or draw below the documented assignment prior)."
        )

    channels = diagnostics.get("channel_diagnostics")
    if not isinstance(channels, Mapping):
        failures.append("SSI take-up channel diagnostics are missing.")
        channels = {}
    if set(channels) != set(_KNOWN_CHANNELS):
        failures.append(
            "SSI take-up requires exact ASEC/PUF channel diagnostics; got "
            f"{sorted(channels)}."
        )
    for channel in sorted(_KNOWN_CHANNELS):
        values = channels.get(channel)
        if not isinstance(values, Mapping):
            continue
        if int(values.get("rows", 0)) <= 0:
            failures.append(f"SSI take-up channel {channel!r} has no rows.")
        if int(values.get("unique_count", 0)) < 2:
            failures.append(f"SSI take-up channel {channel!r} is constant.")
        share = float(values.get("weighted_true_share", np.nan))
        if not np.isfinite(share) or not 0 < share < 1:
            failures.append(
                f"SSI take-up channel {channel!r} has an invalid weighted share."
            )

    age_bands = diagnostics.get("age_bands")
    if not isinstance(age_bands, list):
        failures.append("SSI take-up age-band diagnostics are missing.")
        age_bands = []
    if len(age_bands) != len(expected_targets) or not all(
        isinstance(row, Mapping) for row in age_bands
    ):
        failures.append(
            "SSI take-up requires exactly one diagnostic row per SSA age band."
        )
    by_key = {
        str(row.get("age_band")): row for row in age_bands if isinstance(row, Mapping)
    }
    if set(by_key) != set(expected_targets):
        failures.append(
            "SSI take-up age bands drifted from the three disjoint SSA strata."
        )
    for key, target in expected_targets.items():
        row = by_key.get(key)
        if row is None:
            continue
        recorded_target = float(row.get("target", np.nan))
        if not np.isclose(recorded_target, target, rtol=0.0, atol=1e-6):
            failures.append(
                f"SSI take-up age band {key!r} target {recorded_target} does "
                f"not match {target}."
            )
            continue
        capacity = float(row.get("candidate_capacity", np.nan))
        floor = float(row.get("reporter_candidate_floor", np.nan))
        selected = float(row.get("selected_recipient_weight", np.nan))
        signed_error = float(row.get("signed_target_error", np.nan))
        shortfall = float(row.get("target_shortfall", np.nan))
        anchor_excess = float(row.get("anchor_excess", np.nan))
        prior = float(row.get("assignment_prior", np.nan))
        recomputed_prior = float(
            row.get("prior_recomputed_from_current_weights", np.nan)
        )
        max_weight = float(row.get("max_source_candidate_weight", np.nan))
        numeric_values = (
            capacity,
            floor,
            selected,
            signed_error,
            shortfall,
            anchor_excess,
            prior,
            recomputed_prior,
            max_weight,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            failures.append(f"SSI take-up age band {key!r} has nonfinite diagnostics.")
            continue
        if capacity <= 0:
            failures.append(
                f"SSI take-up age band {key!r} has zero candidate capacity."
            )
            continue
        if floor < 0 or floor > capacity + 1e-6:
            failures.append(
                f"SSI take-up age band {key!r} has an invalid anchor floor."
            )
            continue
        epsilon = max(1e-6, np.finfo(np.float64).eps * max(capacity, 1.0) * 16.0)
        if selected < floor - epsilon or selected > capacity + epsilon:
            # Anchored reporters are always selected, so the selected weight
            # can never fall below the reporter floor nor exceed capacity.
            failures.append(
                f"SSI take-up age band {key!r} selected weight is outside "
                "the [reporter floor, capacity] envelope."
            )
        if abs(signed_error - (selected - target)) > epsilon:
            failures.append(
                f"SSI take-up age band {key!r} carries the wrong signed error."
            )
        if abs(shortfall - max(target - selected, 0.0)) > epsilon:
            failures.append(
                f"SSI take-up age band {key!r} carries the wrong target shortfall."
            )
        if abs(anchor_excess - max(floor - target, 0.0)) > epsilon:
            failures.append(
                f"SSI take-up age band {key!r} carries the wrong anchor excess."
            )
        # The assignment prior is the value that generated the frozen flags;
        # on release weights it will differ from target/capacity and its
        # integrity is enforced by the Bernoulli-law recheck above, so the
        # gate only requires a valid probability here. The recomputed prior
        # is defined on THIS row's capacity/floor and must match that
        # arithmetic exactly. The band-count MISS is calibration's to close
        # (#470 targets) and ships in the scorecard — never a failure here.
        if not 0.0 <= prior <= 1.0:
            failures.append(
                f"SSI take-up age band {key!r} assignment prior {prior} is "
                "outside [0, 1]."
            )
        if abs(recomputed_prior - _band_prior(target, capacity, floor)) > epsilon:
            failures.append(
                f"SSI take-up age band {key!r} carries the wrong recomputed prior."
            )
        saturated = bool(row.get("saturated"))
        if saturated != (capacity < target):
            failures.append(
                f"SSI take-up age band {key!r} saturation status is inconsistent."
            )

    expected_total = float(sum(expected_targets.values()))
    recorded_total = float(diagnostics.get("target_total", np.nan))
    if not np.isclose(recorded_total, expected_total, rtol=0.0, atol=1e-6):
        failures.append(
            "SSI take-up aggregate target is not the sum of its disjoint age bands."
        )
    selected_total = float(
        sum(
            float(row.get("selected_recipient_weight", np.nan))
            for row in by_key.values()
        )
    )
    recorded_selected_total = float(
        diagnostics.get("selected_recipient_weight_total", np.nan)
    )
    if not np.isclose(
        recorded_selected_total,
        selected_total,
        rtol=0.0,
        atol=1e-6,
    ):
        failures.append("SSI take-up aggregate selected count drifted from age bands.")
    shortfall_total = float(
        sum(float(row.get("target_shortfall", np.nan)) for row in by_key.values())
    )
    recorded_shortfall_total = float(diagnostics.get("target_shortfall_total", np.nan))
    if not np.isclose(
        recorded_shortfall_total,
        shortfall_total,
        rtol=0.0,
        atol=1e-6,
    ):
        failures.append("SSI take-up aggregate shortfall drifted from age bands.")
    return GateResult(
        name="ssi_take_up",
        passed=not failures,
        failures=tuple(failures),
        details=dict(diagnostics),
    )


def write_us_ssi_take_up_diagnostics(
    diagnostics: Mapping[str, object], path: str | Path
) -> Path:
    """Write final-weight SSI take-up diagnostics as strict JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(diagnostics), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
