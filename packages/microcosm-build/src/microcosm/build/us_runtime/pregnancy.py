"""Pregnancy input seeded from the national pregnancy rate.

Without this stage the published dataset stores no ``is_pregnant``, so the
input defaults to False for everyone and the SNAP ABAWD pregnancy
exemption (7 U.S.C. 2015(o)(3)(E)) can never fire — part of the
unsourced-exemption-input failure mode of microcosm #351.

The CPS ASEC does not ask about pregnancy, so the flag cannot be mapped;
it is seeded, the same way the retired enhanced-CPS pipeline seeded it:
women aged 15-44 receive stochastic draws against a pregnancy rate. The
retired pipeline used CDC VSRR state-level birth counts over ACS female
15-44 populations (point-in-time rate = births x 39/52 for the ~9-month
duration within a year), falling back to a 4.1% national rate when state
rates were unavailable. Microcosm builds are hermetic — no live CDC/ACS
fetches — so this stage seeds at the cited national rate; carrying
state-level rates through a packaged artifact is follow-up work under
microcosm #351.

The rate is data, not code: it lives in the ``pregnancy`` stage of
``microcosm/build/us/source_stages.json`` with its citation and reaches
this module as a manifest operation parameter.

Selection draws are seeded blake2b hashes keyed by the person's stable
source identity (``source_year`` / ``source_household_id`` /
``source_person_id`` when present), so support-channel clones of one
source person always receive the same flag and reruns are
bit-reproducible.

Healing behavior: a frame that already carries ``is_pregnant`` with
signal passes through untouched (idempotent). A constant column —
indistinguishable from the engine's broadcast default — is reseeded.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRNGCapability,
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS",
    "US_PREGNANCY_OUTPUT_COLUMN",
    "US_PREGNANCY_REQUIRED_SOURCE_COLUMNS",
    "US_PREGNANCY_STAGE_NAME",
    "derive_us_pregnancy_from_manifest",
    "us_pregnancy_signal_gate",
    "us_pregnancy_stage_spec",
    "us_pregnancy_summary",
    "with_us_pregnancy_inputs",
]

US_PREGNANCY_STAGE_NAME = "pregnancy"

#: The PolicyEngine-facing person input column this stage owns.
US_PREGNANCY_OUTPUT_COLUMN = "is_pregnant"

#: Release gates require these person columns to carry signal (≥2 values).
US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (US_PREGNANCY_OUTPUT_COLUMN,)

#: Raw CPS ASEC person columns the seeding reads: sex (A_SEX, 2 = female)
#: and age bound the eligible population.
US_PREGNANCY_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "A_SEX",
    "A_AGE",
)

_FEMALE_SEX_CODE = 2
_CHILDBEARING_AGE_RANGE = (15, 44)

#: Weighted share of all persons flagged pregnant must land in this band.
#: Expected ≈ rate (4.1%) x the female-15-44 share of the population
#: (~19%) ≈ 0.8%; the band brackets that with room for weighting.
_PREGNANT_SHARE_BAND = (0.002, 0.02)

_PERSON_WEIGHT_COLUMN = "person_weight"

_DERIVE_PREGNANCY_PARAMETER_KEYS = frozenset(
    {"seed_from_build_config", "pregnancy_rate"}
)


def us_pregnancy_stage_spec() -> SourceStageSpec:
    """Load the packaged ``pregnancy`` source-stage manifest entry."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_PREGNANCY_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_PREGNANCY_STAGE_NAME!r} stage."
        )
    return stage_map[US_PREGNANCY_STAGE_NAME]


def _pregnancy_rate(operation: SourceOperationSpec) -> float:
    """The manifest-declared pregnancy rate, validated."""

    declared = operation.parameters.get("pregnancy_rate")
    if not isinstance(declared, dict) or "value" not in declared:
        raise SourceRuntimeError(
            "Pregnancy seeding requires a manifest pregnancy_rate parameter "
            "with a value and source."
        )
    rate = float(declared["value"])
    if not (0.0 < rate < 1.0):
        raise SourceRuntimeError(f"Pregnancy rate must be in (0, 1), got {rate!r}.")
    return rate


def _stable_person_draws(persons: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Seeded uniform draws keyed by stable source identity per person.

    Support-channel clones share their source identity, so they always
    receive the same draw; frames without source columns key on the
    person id itself.
    """

    if {"source_year", "source_household_id", "source_person_id"} <= set(
        persons.columns
    ):
        keys = (
            persons["source_year"].astype(str)
            + ":"
            + persons["source_household_id"].astype(str)
            + ":"
            + persons["source_person_id"].astype(str)
        )
    else:
        keys = persons["person_id"].astype(str)
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:pregnancy:{key}".encode(),
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


def derive_us_pregnancy_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Seed ``is_pregnant`` among women of childbearing age.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation).
    """

    if operation.kind != "derive_pregnancy":
        raise SourceRuntimeError(
            f"Pregnancy seeding received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "Pregnancy seeding requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DERIVE_PREGNANCY_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"Pregnancy seeding received unsupported parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_PREGNANCY_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"Pregnancy seeding requires raw ASEC column(s): {missing}."
        )

    rate = _pregnancy_rate(operation)
    sex = pd.to_numeric(frame["A_SEX"], errors="coerce")
    age = pd.to_numeric(frame["A_AGE"], errors="coerce")
    low, high = _CHILDBEARING_AGE_RANGE
    eligible = ((sex == _FEMALE_SEX_CODE) & (age >= low) & (age <= high)).to_numpy()
    if context.rng is None:
        draws = _stable_person_draws(frame, seed=context.config.seed)
    else:
        if {"source_year", "source_household_id", "source_person_id"} <= set(
            frame.columns
        ):
            keys = (
                frame["source_year"].astype(str)
                + ":"
                + frame["source_household_id"].astype(str)
                + ":"
                + frame["source_person_id"].astype(str)
            )
        else:
            keys = frame["person_id"].astype(str)
        draws = context.rng.blake2b_uniforms(
            context.rng.token("pregnancy_assignment"),
            stable_keys=keys.tolist(),
        )
    result = frame.copy(deep=True)
    result[US_PREGNANCY_OUTPUT_COLUMN] = eligible & (draws < rate)
    return result


def _pregnancy_carries_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted pregnancy column is trustworthy as-is.

    A constant column (one observed value) is indistinguishable from the
    engine's broadcast default and must be reseeded, not passed through.
    """

    return person[US_PREGNANCY_OUTPUT_COLUMN].dropna().nunique() > 1


def with_us_pregnancy_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    rng: SourceRNGCapability | None = None,
) -> Frame:
    """Run the ``pregnancy`` manifest stage over a US frame.

    A frame already carrying a non-constant ``is_pregnant`` passes through
    untouched (idempotent). Any other surface — column missing, or
    constant at the engine default — is reseeded from the raw ASEC
    columns.

    Args:
        frame: A US-schema frame whose person table still carries the raw
            CPS ASEC source columns (unless the output already carries
            signal).
        seed: Build-wide imputation seed (keys the stable draws).
        time_period: The dataset's time period.

    Returns:
        A new frame whose person table carries ``is_pregnant``.

    Raises:
        ValueError: If the frame is not US-schema or the stage output does
            not cover every person.
        SourceRuntimeError: If required raw ASEC columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US pregnancy inputs require the US schema.")
    person = frame.table("person")
    if US_PREGNANCY_OUTPUT_COLUMN in person.columns and _pregnancy_carries_signal(
        person
    ):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_pregnancy_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_pregnancy": derive_us_pregnancy_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
        rng=rng,
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[US_PREGNANCY_OUTPUT_COLUMN].isna().any():
        raise ValueError(
            "US pregnancy stage output does not cover every person for "
            f"{US_PREGNANCY_OUTPUT_COLUMN!r}."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][US_PREGNANCY_OUTPUT_COLUMN] = aligned[
        US_PREGNANCY_OUTPUT_COLUMN
    ].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_pregnancy_summary(frame: Frame) -> dict[str, object]:
    """Weighted pregnancy-share summary for gates and release manifests."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    pregnant = person[US_PREGNANCY_OUTPUT_COLUMN].astype(bool).to_numpy()
    pregnant_share = (
        float(weights[pregnant].sum()) / total_weight if total_weight > 0 else 0.0
    )
    return {
        "pregnant_share": pregnant_share,
        "pregnant_share_band": list(_PREGNANT_SHARE_BAND),
        "unique_count": int(person[US_PREGNANCY_OUTPUT_COLUMN].dropna().nunique()),
    }


def us_pregnancy_signal_gate(frame: Frame) -> GateResult:
    """Require the pregnancy surface to carry a plausible seeded share.

    Fails when the column is missing or constant, or when the weighted
    pregnant share leaves the plausibility band — reproducing (or
    inverting) the nobody-is-ever-pregnant failure of microcosm #351.
    """

    person = frame.table("person")
    if US_PREGNANCY_OUTPUT_COLUMN not in person.columns:
        return GateResult(
            name="pregnancy_signal",
            passed=False,
            failures=(f"person column missing: {US_PREGNANCY_OUTPUT_COLUMN}.",),
            details={"missing": [US_PREGNANCY_OUTPUT_COLUMN]},
        )

    failures: list[str] = []
    summary = us_pregnancy_summary(frame)
    if int(summary["unique_count"]) < 2:
        failures.append(
            f"{US_PREGNANCY_OUTPUT_COLUMN}: constant column (one observed "
            "value) — the pregnancy surface carries no signal."
        )
    share = float(summary["pregnant_share"])
    low, high = _PREGNANT_SHARE_BAND
    if not (low <= share <= high):
        failures.append(
            f"pregnant share {share:.4f} outside plausibility band [{low}, {high}]."
        )
    return GateResult(
        name="pregnancy_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
