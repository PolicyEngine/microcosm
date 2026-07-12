"""Source-backed WIC claim propensity from official FNS category rates.

The retired eCPS pipeline calculated PolicyEngine-US ``wic_category_str`` and
then drew ``would_claim_wic`` from the USDA FNS participation rate for that
category.  This stage restores that exported input without inventing a survey
receipt anchor: the hermetic ASEC spine has the age, sex, parent, family, and
newly seeded pregnancy inputs needed to reproduce the demographic categories,
while the official CY2022 FNS estimates supply the rates.

PolicyEngine-US 1.764.6 evaluates categories in this order: pregnant,
breastfeeding mother of an infant, postpartum mother, infant, child, none.
There is no breastfeeding assessment in the hermetic sources.  Rather than
invent one, this stage assigns every female parent whose family includes an
infant to FNS's all-postpartum category.  Pregnancy remains first, so this
stage deliberately runs after the pregnancy stage and corrects the retired
file's accidental ordering (it calculated WIC categories before seeding
pregnancy).

Draws are seeded blake2b hashes keyed by stable source-person identity.  PUF
support clones therefore retain identical claim flags, and reruns with the
same build seed are bit-reproducible.  The frame wrapper passes an existing
surface through only when it exactly equals a fresh deterministic derivation;
this heals the retired pre-pregnancy category ordering while remaining
idempotent for the same build seed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_WIC_CLAIM_NONCONSTANT_PERSON_COLUMNS",
    "US_WIC_CLAIM_OUTPUT_COLUMNS",
    "US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS",
    "US_WIC_CLAIM_STAGE_NAME",
    "WIC_CLAIM_ARCHIVED_DERIVATION_URL",
    "WIC_CLAIM_ARCHIVED_PARAMETERS_URL",
    "WIC_CLAIM_ARCHIVED_RANDOMNESS_URL",
    "WIC_CLAIM_FNS_SOURCE_URL",
    "derive_us_wic_claim_from_manifest",
    "us_wic_claim_signal_gate",
    "us_wic_claim_stage_spec",
    "us_wic_claim_summary",
    "with_us_wic_claim_input",
]

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_"
    "us_data/"
)
WIC_CLAIM_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "datasets/cps/cps.py#L684-L691"
WIC_CLAIM_ARCHIVED_PARAMETERS_URL = (
    _ARCHIVED_ROOT + "parameters/take_up/wic_takeup.yaml#L1-L33"
)
WIC_CLAIM_ARCHIVED_RANDOMNESS_URL = _ARCHIVED_ROOT + "utils/randomness.py#L5-L28"
WIC_CLAIM_FNS_SOURCE_URL = (
    "https://fns-prod.azureedge.us/sites/default/files/resource-files/"
    "wic-eer-2022-summary.pdf"
)

US_WIC_CLAIM_STAGE_NAME = "wic_claim_input"
US_WIC_CLAIM_OUTPUT_COLUMNS: tuple[str, ...] = ("would_claim_wic",)
US_WIC_CLAIM_NONCONSTANT_PERSON_COLUMNS = US_WIC_CLAIM_OUTPUT_COLUMNS

# These are persisted PolicyEngine-facing columns already carried or derived
# on the ASEC spine.  ``own_children_in_household`` supplies ``is_parent``'s
# measured fact; family minimum age reproduces the engine's category formula.
US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "age",
    "is_female",
    "is_pregnant",
    "own_children_in_household",
    "person_family_id",
)

_OUTPUT = US_WIC_CLAIM_OUTPUT_COLUMNS[0]
_PERSON_WEIGHT_COLUMN = "person_weight"
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_PERSON_SUPPORT_SOURCE_ID_COLUMN = "person_support_source_id"
_SOURCE_IDENTITY_COLUMNS = (
    "source_year",
    "source_household_id",
    "source_person_id",
)

_CATEGORY_RATES: dict[str, float] = {
    "pregnant": 0.456,
    "postpartum": 0.689,
    "breastfeeding": 0.663,
    "infant": 0.784,
    "child": 0.460,
    "none": 0.0,
}
_ACTIVE_CATEGORIES = ("pregnant", "postpartum", "infant", "child", "none")
_CATEGORY_ASSIGNMENT_ORDER = _ACTIVE_CATEGORIES
_EXPECTED_OPERATION_PARAMETERS = {
    "seed_from_build_config": True,
    "category_rates": {
        "source": WIC_CLAIM_FNS_SOURCE_URL,
        "vintage": "CY2022",
        "values": _CATEGORY_RATES,
    },
}

# The locked 2022-2024 ASEC pools produce an all-person claim share around
# 3.4%.  Per-category bands bracket the official probability while remaining
# wide enough for weighted stochastic sampling.
_WEIGHTED_CLAIM_SHARE_BAND = (0.015, 0.060)
_CATEGORY_WEIGHTED_CLAIM_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "pregnant": (0.25, 0.67),
    "postpartum": (0.45, 0.90),
    "infant": (0.55, 0.96),
    "child": (0.30, 0.63),
    "none": (0.0, 0.0),
}


def _validate_operation_parameters(operation: SourceOperationSpec) -> dict[str, float]:
    """Validate the exact FNS vintage, source, categories, and rates."""

    parameters = dict(operation.parameters)
    if set(parameters) != {"seed_from_build_config", "category_rates"}:
        raise SourceRuntimeError(
            "US WIC claim derivation requires exactly seed_from_build_config "
            "and category_rates manifest parameters."
        )
    if parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US WIC claim derivation requires seed_from_build_config=true."
        )
    declared = parameters["category_rates"]
    if not isinstance(declared, Mapping):
        raise SourceRuntimeError(
            "US WIC claim category_rates must be a sourced manifest mapping."
        )
    if set(declared) != {"source", "vintage", "values"}:
        raise SourceRuntimeError(
            "US WIC claim category_rates requires exactly source, vintage, and values."
        )
    if declared["source"] != WIC_CLAIM_FNS_SOURCE_URL:
        raise SourceRuntimeError(
            "US WIC claim rates must cite the locked official FNS CY2022 PDF."
        )
    if declared["vintage"] != "CY2022":
        raise SourceRuntimeError(
            "US WIC claim rates must use the locked CY2022 vintage."
        )
    values = declared["values"]
    if not isinstance(values, Mapping) or set(values) != set(_CATEGORY_RATES):
        raise SourceRuntimeError(
            "US WIC claim rates must declare exactly pregnant, postpartum, "
            "breastfeeding, infant, child, and none."
        )
    validated: dict[str, float] = {}
    for category, expected in _CATEGORY_RATES.items():
        value = values[category]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SourceRuntimeError(
                f"US WIC claim rate for {category!r} must be numeric."
            )
        rate = float(value)
        if rate != expected:
            raise SourceRuntimeError(
                f"US WIC claim rate for {category!r} must equal the official "
                f"CY2022 value {expected}, got {rate}."
            )
        validated[category] = rate
    return validated


def us_wic_claim_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged WIC-claim source-stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_WIC_CLAIM_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_WIC_CLAIM_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_WIC_CLAIM_STAGE_NAME]
    if tuple(spec.outputs) != US_WIC_CLAIM_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_WIC_CLAIM_STAGE_NAME!r} manifest outputs do not match the "
            "runtime-owned WIC claim family."
        )
    if [operation.kind for operation in spec.operations] != [
        "read_table",
        "derive_wic_claim",
    ]:
        raise ValueError(
            f"{US_WIC_CLAIM_STAGE_NAME!r} must contain only read_table then "
            "derive_wic_claim."
        )
    if dict(spec.operations[0].parameters) != {
        "table": "person",
        "weight": _PERSON_WEIGHT_COLUMN,
    }:
        raise ValueError(
            f"{US_WIC_CLAIM_STAGE_NAME!r} must read the weighted person table."
        )
    try:
        _validate_operation_parameters(spec.operations[1])
    except SourceRuntimeError as exc:
        raise ValueError(
            f"{US_WIC_CLAIM_STAGE_NAME!r} manifest parameter drift: {exc}"
        ) from exc
    artifact_sources = {
        str(artifact.get("source"))
        for artifact in spec.artifacts
        if artifact.get("source") is not None
    }
    if WIC_CLAIM_FNS_SOURCE_URL not in artifact_sources:
        raise ValueError(
            f"{US_WIC_CLAIM_STAGE_NAME!r} does not pin the official FNS PDF artifact."
        )
    return spec


def _strict_numeric(
    person: pd.DataFrame,
    column: str,
    *,
    minimum: float,
    maximum: float | None = None,
    integer: bool = False,
) -> np.ndarray:
    numeric = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric) & (numeric >= minimum)
    if maximum is not None:
        valid &= numeric <= maximum
    if integer:
        valid &= numeric == np.floor(numeric)
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US WIC claim derivation requires valid {column!r}; invalid row(s): "
            f"{rows}."
        )
    return numeric


def _strict_bool(person: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & np.isin(values, np.array([0.0, 1.0]))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US WIC claim derivation requires boolean {column!r}; invalid "
            f"row(s): {rows}."
        )
    return values.astype(bool)


def _wic_categories(person: pd.DataFrame) -> np.ndarray:
    """Reproduce the PE-US category precedence from persisted inputs."""

    missing = [
        column
        for column in US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS
        if column not in person.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US WIC claim derivation requires person source column(s): {missing}."
        )
    if person["person_family_id"].isna().any():
        rows = np.flatnonzero(person["person_family_id"].isna().to_numpy())[:5].tolist()
        raise SourceRuntimeError(
            "US WIC claim derivation requires nonmissing person_family_id; "
            f"invalid row(s): {rows}."
        )

    # ASEC age is integer years.  Enforcing that source contract makes the
    # collapsed ``< 1`` infant-family test equivalent to PE-US's non-
    # breastfeeding postpartum ``< 0.5`` branch on this hermetic surface.
    age = _strict_numeric(
        person,
        "age",
        minimum=0.0,
        maximum=120.0,
        integer=True,
    )
    female = _strict_bool(person, "is_female")
    pregnant = _strict_bool(person, "is_pregnant")
    own_children = _strict_numeric(
        person,
        "own_children_in_household",
        minimum=0.0,
        integer=True,
    )
    if np.any(pregnant & ~female):
        rows = np.flatnonzero(pregnant & ~female)[:5].tolist()
        raise SourceRuntimeError(
            "US WIC claim derivation found is_pregnant=true for a nonfemale "
            f"person; invalid row(s): {rows}."
        )

    min_family_age = (
        pd.Series(age, index=person.index)
        .groupby(person["person_family_id"], sort=False)
        .transform("min")
        .to_numpy(dtype=np.float64)
    )
    postpartum = female & (own_children > 0.0) & (min_family_age < 1.0)
    return np.select(
        [pregnant, postpartum, age < 1.0, age < 5.0],
        ["pregnant", "postpartum", "infant", "child"],
        default="none",
    )


def _stable_person_keys(person: pd.DataFrame) -> pd.Series:
    present = [column in person.columns for column in _SOURCE_IDENTITY_COLUMNS]
    if any(present) and not all(present):
        missing = [
            column
            for column, is_present in zip(
                _SOURCE_IDENTITY_COLUMNS, present, strict=True
            )
            if not is_present
        ]
        raise SourceRuntimeError(
            "US WIC claim stable source identity is partial; missing column(s): "
            f"{missing}."
        )
    if all(present):
        identity = person.loc[:, list(_SOURCE_IDENTITY_COLUMNS)]
        if identity.isna().any(axis=None):
            rows = np.flatnonzero(identity.isna().any(axis=1).to_numpy())[:5].tolist()
            raise SourceRuntimeError(
                "US WIC claim stable source identity contains missing values at "
                f"row(s): {rows}."
            )
        return (
            identity["source_year"].astype(str)
            + ":"
            + identity["source_household_id"].astype(str)
            + ":"
            + identity["source_person_id"].astype(str)
        )
    if _PERSON_SUPPORT_SOURCE_ID_COLUMN in person.columns:
        source_id = person[_PERSON_SUPPORT_SOURCE_ID_COLUMN]
        if source_id.isna().any():
            rows = np.flatnonzero(source_id.isna().to_numpy())[:5].tolist()
            raise SourceRuntimeError(
                "US WIC claim support source identity contains missing values at "
                f"row(s): {rows}."
            )
        return "support:" + source_id.astype(str)
    if "person_id" not in person.columns or person["person_id"].isna().any():
        raise SourceRuntimeError(
            "US WIC claim derivation requires person_id when stable source "
            "identity is unavailable."
        )
    return "person:" + person["person_id"].astype(str)


def _stable_person_draws(person: pd.DataFrame, *, seed: int) -> np.ndarray:
    keys = _stable_person_keys(person)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{_OUTPUT}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def derive_us_wic_claim_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Assign ``would_claim_wic`` from source categories and FNS rates."""

    if operation.kind != "derive_wic_claim":
        raise SourceRuntimeError(
            f"US WIC claim derivation received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US WIC claim derivation requires the person table to be read first."
        )
    rates = _validate_operation_parameters(operation)
    categories = _wic_categories(frame)
    draws = _stable_person_draws(frame, seed=int(context.config.seed))
    thresholds = np.fromiter(
        (rates[str(category)] for category in categories),
        dtype=np.float64,
        count=len(categories),
    )
    result = frame.copy(deep=True)
    result[_OUTPUT] = draws < thresholds
    return result


def _output_values(person: pd.DataFrame) -> tuple[np.ndarray, int]:
    series = person[_OUTPUT]
    missing = int(series.isna().sum())
    converted = pd.to_numeric(series, errors="coerce")
    coerced_invalid = converted.isna() & ~series.isna()
    if coerced_invalid.any():
        rows = np.flatnonzero(coerced_invalid.to_numpy())[:5].tolist()
        raise SourceRuntimeError(
            f"US WIC claim output {_OUTPUT!r} must be boolean; nonnumeric "
            f"value(s) at row(s): {rows}."
        )
    numeric = converted.to_numpy(dtype=np.float64)
    valid = np.isnan(numeric) | np.isin(numeric, np.array([0.0, 1.0]))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US WIC claim output {_OUTPUT!r} must be boolean; invalid row(s): {rows}."
        )
    return np.nan_to_num(numeric, nan=0.0).astype(bool), missing


def with_us_wic_claim_input(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Materialize the source-backed WIC claim input on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US WIC claim input requires the US schema.")

    person = frame.table("person")
    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_wic_claim_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_wic_claim": derive_us_wic_claim_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    if output["person_id"].duplicated().any():
        raise ValueError("US WIC claim stage produced duplicate person_id rows.")
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[_OUTPUT].isna().any():
        raise ValueError("US WIC claim stage output does not cover every person.")
    expected = aligned[_OUTPUT].to_numpy(dtype=bool)
    if _OUTPUT in person:
        try:
            current, missing = _output_values(person)
        except SourceRuntimeError:
            current, missing = np.zeros(len(person), dtype=bool), len(person)
        if not missing and np.array_equal(current, expected):
            return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_OUTPUT] = expected
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _person_weights(frame: Frame) -> np.ndarray:
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    valid = np.isfinite(weights) & (weights >= 0.0)
    if not valid.all() or float(weights.sum()) <= 0.0:
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            "US WIC claim gate requires finite nonnegative person weights with "
            f"positive total; invalid row(s): {rows}."
        )
    return weights


def _clone_diagnostics(
    person: pd.DataFrame,
    *,
    categories: np.ndarray,
    claims: np.ndarray,
) -> dict[str, int]:
    keys = _stable_person_keys(person)
    work = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "category": categories,
            "claim": claims,
        }
    )
    sizes = work.groupby("key", sort=False).size()
    repeated = sizes.index[sizes > 1]
    claim_unique = work.groupby("key", sort=False)["claim"].nunique()
    category_unique = work.groupby("key", sort=False)["category"].nunique()
    result = {
        "clone_group_count": int(len(repeated)),
        "clone_claim_mismatch_count": int((claim_unique.reindex(repeated) > 1).sum()),
        "clone_category_mismatch_count": int(
            (category_unique.reindex(repeated) > 1).sum()
        ),
    }
    return result


def us_wic_claim_summary(frame: Frame) -> dict[str, object]:
    """Return category, channel, and clone diagnostics for the release gate."""

    person = frame.table("person")
    if _OUTPUT not in person:
        raise SourceRuntimeError(f"US WIC claim summary requires {_OUTPUT!r}.")
    categories = _wic_categories(person)
    claims, missing_count = _output_values(person)
    weights = _person_weights(frame)
    total_weight = float(weights.sum())

    category_counts: dict[str, int] = {}
    category_weights: dict[str, float] = {}
    category_claim_shares: dict[str, float] = {}
    for category in _ACTIVE_CATEGORIES:
        mask = categories == category
        category_weight = float(weights[mask].sum())
        category_counts[category] = int(np.count_nonzero(mask))
        category_weights[category] = category_weight
        category_claim_shares[category] = (
            float(weights[mask & claims].sum()) / category_weight
            if category_weight > 0.0
            else 0.0
        )

    summary: dict[str, object] = {
        "weighted_claim_share": float(weights[claims].sum()) / total_weight,
        "weighted_claim_share_band": list(_WEIGHTED_CLAIM_SHARE_BAND),
        "positive_count": int(np.count_nonzero(claims)),
        "unique_count": int(np.unique(claims).size) if not missing_count else 0,
        "missing_count": missing_count,
        "category_assignment_order": list(_CATEGORY_ASSIGNMENT_ORDER),
        "category_rates": dict(_CATEGORY_RATES),
        "category_counts": category_counts,
        "category_weights": category_weights,
        "category_weighted_claim_shares": category_claim_shares,
        "category_weighted_claim_share_bands": {
            key: list(value)
            for key, value in _CATEGORY_WEIGHTED_CLAIM_SHARE_BANDS.items()
        },
        "breastfeeding_source_available": False,
        "breastfeeding_rate_validated_but_unassigned": _CATEGORY_RATES["breastfeeding"],
        **_clone_diagnostics(person, categories=categories, claims=claims),
    }

    if _PERSON_SUPPORT_CHANNEL_COLUMN in person.columns:
        channels = (
            person[_PERSON_SUPPORT_CHANNEL_COLUMN]
            .fillna("<missing>")
            .astype(str)
            .to_numpy()
        )
        channel_shares: dict[str, float] = {}
        channel_unique_counts: dict[str, int] = {}
        for channel in sorted(set(channels.tolist())):
            mask = channels == channel
            channel_weight = float(weights[mask].sum())
            channel_shares[channel] = (
                float(weights[mask & claims].sum()) / channel_weight
                if channel_weight > 0.0
                else 0.0
            )
            channel_unique_counts[channel] = int(np.unique(claims[mask]).size)
        summary["channel_weighted_claim_shares"] = channel_shares
        summary["channel_unique_counts"] = channel_unique_counts
        summary["channel_weighted_claim_share_band"] = list(_WEIGHTED_CLAIM_SHARE_BAND)
    return summary


def us_wic_claim_signal_gate(frame: Frame) -> GateResult:
    """Require nondefault, category-plausible, clone-consistent WIC claims."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="wic_claim_input_signal",
            passed=False,
            failures=(f"person.{_OUTPUT}: missing",),
            details={"missing": [_OUTPUT]},
        )
    try:
        summary = us_wic_claim_summary(frame)
    except SourceRuntimeError as exc:
        return GateResult(
            name="wic_claim_input_signal",
            passed=False,
            failures=(str(exc),),
            details={},
        )

    failures: list[str] = []
    if int(summary["missing_count"]):
        failures.append(f"{_OUTPUT}: missing values")
    if int(summary["unique_count"]) < 2:
        failures.append(
            f"{_OUTPUT}: constant column — the default-True WIC claim surface "
            "carries no source-backed signal"
        )
    claim_share = float(summary["weighted_claim_share"])
    low, high = _WEIGHTED_CLAIM_SHARE_BAND
    if not low <= claim_share <= high:
        failures.append(
            f"{_OUTPUT}: weighted claim share {claim_share:.6f} outside "
            f"[{low:.3f}, {high:.3f}]"
        )

    category_counts = dict(summary["category_counts"])
    category_weights = dict(summary["category_weights"])
    category_shares = dict(summary["category_weighted_claim_shares"])
    for category in _ACTIVE_CATEGORIES:
        if (
            int(category_counts[category]) == 0
            or float(category_weights[category]) <= 0
        ):
            failures.append(f"{_OUTPUT}: WIC category {category!r} is absent")
            continue
        observed = float(category_shares[category])
        category_low, category_high = _CATEGORY_WEIGHTED_CLAIM_SHARE_BANDS[category]
        if not category_low <= observed <= category_high:
            failures.append(
                f"{_OUTPUT}: {category} weighted claim share {observed:.6f} "
                f"outside [{category_low:.3f}, {category_high:.3f}]"
            )

    for field, label in (
        ("clone_claim_mismatch_count", "claim"),
        ("clone_category_mismatch_count", "category"),
    ):
        count = int(summary[field])
        if count:
            failures.append(f"{_OUTPUT}: {count} support-clone {label} mismatch(es)")

    channel_shares = dict(summary.get("channel_weighted_claim_shares", {}))
    channel_unique = dict(summary.get("channel_unique_counts", {}))
    for channel, share in channel_shares.items():
        if int(channel_unique[channel]) < 2:
            failures.append(f"{_OUTPUT}: {channel} support channel is constant")
        if not low <= float(share) <= high:
            failures.append(
                f"{_OUTPUT}: {channel} weighted claim share {float(share):.6f} "
                f"outside [{low:.3f}, {high:.3f}]"
            )

    return GateResult(
        name="wic_claim_input_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
