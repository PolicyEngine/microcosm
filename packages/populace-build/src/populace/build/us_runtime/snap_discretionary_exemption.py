"""SNAP ABAWD discretionary exemption seeded at the statutory cap rate.

Without this stage the published dataset stores no
``is_snap_abawd_discretionary_exempt``, so the input defaults to False for
everyone and the state discretionary exemption channel
(7 U.S.C. 2015(o)(6)) never fires — the SNAP half of populace #323.

State agencies may exempt a monthly-average share of ABAWD-covered
individuals from the time limit: 15% for FY1998-2019, 12% for FY2020-23,
and 8% from FY2024 (the rate PolicyEngine-US carries as
``gov.usda.snap.work_requirements.abawd.discretionary_exemption_rate``).
Which individuals are *covered* is engine logic (age band, hours,
exemptions) that the build must not duplicate, so — like SNAP take-up —
the flag is seeded across the potentially covered population (adults
18-64) and the engine intersects it with modeled coverage: because draws
are independent of coverage, the exempt share among covered individuals
lands at the statutory rate in expectation. Seeding at the cap assumes
states fully use their exemptions, which overstates actual practice —
USDA reporting shows usage typically runs well under the allotment — so
the seeded channel is an upper bound; refining toward USDA
exemption-usage reports is follow-up under populace #323.

The rate is data, not code: it lives in the
``snap_abawd_discretionary_exemption`` stage of
``populace/build/us/source_stages.json`` with its citation and reaches
this module as a manifest operation parameter.

Selection draws are seeded blake2b hashes keyed by the person's stable
source identity (``source_year`` / ``source_household_id`` /
``source_person_id`` when present), so support-channel clones of one
source person always receive the same flag and reruns are
bit-reproducible.

Healing behavior: a frame that already carries the column with signal
passes through untouched (idempotent). A constant column —
indistinguishable from the engine's broadcast default — is reseeded.
"""

from __future__ import annotations

import hashlib
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
    "US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS",
    "US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN",
    "US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS",
    "US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME",
    "derive_us_snap_discretionary_exemption_from_manifest",
    "us_snap_discretionary_exemption_signal_gate",
    "us_snap_discretionary_exemption_stage_spec",
    "us_snap_discretionary_exemption_summary",
    "with_us_snap_discretionary_exemption_inputs",
]

US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME = "snap_abawd_discretionary_exemption"

#: The PolicyEngine-facing person input column this stage owns.
US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN = "is_snap_abawd_discretionary_exempt"

#: Release gates require these person columns to carry signal (≥2 values).
US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN,
)

#: Raw CPS ASEC person columns the seeding reads: age bounds the
#: potentially ABAWD-covered population.
US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = ("A_AGE",)

#: The post-OBBBA ABAWD-subject age band (a superset of the pre-OBBBA
#: 18-54 band, so the seeded flag serves both regimes).
_COVERED_AGE_RANGE = (18, 64)

#: Weighted share of all persons flagged exempt must land in this band.
#: Expected ≈ rate (8%) x the adult-18-64 share of the population
#: (~60%) ≈ 4.8%; the band brackets that with room for weighting.
_EXEMPT_SHARE_BAND = (0.015, 0.09)

_PERSON_WEIGHT_COLUMN = "person_weight"

_DERIVE_EXEMPTION_PARAMETER_KEYS = frozenset(
    {"seed_from_build_config", "exemption_rate"}
)


def us_snap_discretionary_exemption_stage_spec() -> SourceStageSpec:
    """Load the packaged ``snap_abawd_discretionary_exemption`` manifest entry."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME!r} stage."
        )
    return stage_map[US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME]


def _exemption_rate(operation: SourceOperationSpec) -> float:
    """The manifest-declared exemption rate, validated."""

    declared = operation.parameters.get("exemption_rate")
    if not isinstance(declared, dict) or "value" not in declared:
        raise SourceRuntimeError(
            "Discretionary-exemption seeding requires a manifest exemption_rate parameter "
            "with a value and source."
        )
    rate = float(declared["value"])
    if not (0.0 < rate < 1.0):
        raise SourceRuntimeError(f"Exemption rate must be in (0, 1), got {rate!r}.")
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
                    f"{seed}:snap_abawd_discretionary_exemption:{key}".encode(),
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


def derive_us_snap_discretionary_exemption_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Seed the discretionary exemption among potentially covered adults.

    The current frame must be the raw-column person table (from the stage's
    ``read_table`` operation).
    """

    if operation.kind != "derive_snap_abawd_discretionary_exemption":
        raise SourceRuntimeError(
            f"Discretionary-exemption seeding received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "Discretionary-exemption seeding requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DERIVE_EXEMPTION_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"Discretionary-exemption seeding received unsupported parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"Discretionary-exemption seeding requires raw ASEC column(s): {missing}."
        )

    rate = _exemption_rate(operation)
    age = pd.to_numeric(frame["A_AGE"], errors="coerce")
    low, high = _COVERED_AGE_RANGE
    eligible = ((age >= low) & (age <= high)).to_numpy()
    draws = _stable_person_draws(frame, seed=context.config.seed)
    result = frame.copy(deep=True)
    result[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN] = eligible & (draws < rate)
    return result


def _exemption_carries_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted exemption column is trustworthy as-is.

    A constant column (one observed value) is indistinguishable from the
    engine's broadcast default and must be reseeded, not passed through.
    """

    return person[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].dropna().nunique() > 1


def with_us_snap_discretionary_exemption_inputs(
    frame: Frame, *, seed: int, time_period: int
) -> Frame:
    """Run the ``snap_abawd_discretionary_exemption`` manifest stage over a US frame.

    A frame already carrying a non-constant exemption column passes through
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
        A new frame whose person table carries the exemption column.

    Raises:
        ValueError: If the frame is not US-schema or the stage output does
            not cover every person.
        SourceRuntimeError: If required raw ASEC columns are missing.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError(
            "US SNAP discretionary-exemption inputs require the US schema."
        )
    person = frame.table("person")
    if (
        US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN in person.columns
        and _exemption_carries_signal(person)
    ):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_snap_discretionary_exemption_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_snap_abawd_discretionary_exemption": derive_us_snap_discretionary_exemption_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].isna().any():
        raise ValueError(
            "US SNAP discretionary-exemption stage output does not cover every person for "
            f"{US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN!r}."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN] = aligned[
        US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN
    ].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_snap_discretionary_exemption_summary(frame: Frame) -> dict[str, object]:
    """Weighted exempt-share summary for gates and release manifests."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    exempt = (
        person[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].astype(bool).to_numpy()
    )
    exempt_share = (
        float(weights[exempt].sum()) / total_weight if total_weight > 0 else 0.0
    )
    return {
        "exempt_share": exempt_share,
        "exempt_share_band": list(_EXEMPT_SHARE_BAND),
        "unique_count": int(
            person[US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN].dropna().nunique()
        ),
    }


def us_snap_discretionary_exemption_signal_gate(frame: Frame) -> GateResult:
    """Require the exemption surface to carry a plausible seeded share.

    Fails when the column is missing or constant, or when the weighted
    exempt share leaves the plausibility band — reproducing (or
    inverting) the nobody-is-ever-exempt failure of populace #323.
    """

    person = frame.table("person")
    if US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN not in person.columns:
        return GateResult(
            name="snap_discretionary_exemption_signal",
            passed=False,
            failures=(
                f"person column missing: {US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN}.",
            ),
            details={"missing": [US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN]},
        )

    failures: list[str] = []
    summary = us_snap_discretionary_exemption_summary(frame)
    if int(summary["unique_count"]) < 2:
        failures.append(
            f"{US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN}: constant column (one observed "
            "value) — the exemption surface carries no signal."
        )
    share = float(summary["exempt_share"])
    low, high = _EXEMPT_SHARE_BAND
    if not (low <= share <= high):
        failures.append(
            f"exempt share {share:.4f} outside plausibility band [{low}, {high}]."
        )
    return GateResult(
        name="snap_discretionary_exemption_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
