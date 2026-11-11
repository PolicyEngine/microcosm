"""The clone-first spine nests, stage by stage (microcosm#931, O10).

Invented households go through the real geographic expansion (G1,
``geographic_expand``), then the SPI support channel (stage 16), the SPI income
band donors (stage 17) and the CGT support split (stage 42), each run by its
stage transform under the packaged stage declaration, G1's included. The
frame runs at K = 1, 2, 3, 7 and 15 geographic clones; it is small, so K = 15
takes well under a second. Each K is held to K = 1:

- **Reported first:** stage 17's mass scale at K = 1 and at each K. Clone 0's
  stage-17 weights nest on K = 1's only while the scale is the one a one-clone
  spine takes. On this frame it is one at every K, and the test asserts it.
- **Exact:**
  - clone 0 of G1's output is the one-clone frame, ids and G1's columns
    included;
  - stage 16 copies, and stage 17 seats, K = 1's households in clone 0;
  - every table of every frame carries its clone index with exactly K values,
    and every copy carries its parent's.
- **K x each clone-0 weight against K = 1's**, at G1, stage 16 and stage 17:
  - Rows an exact-total correction moved, at K or at K = 1, are within
    ``rows x K x 2.2e-16`` of K = 1's, relative. ``rows`` counts the one-clone
    frame's households at that stage.
  - At stage 17, the other incumbents of a funding stratum whose stage-16
    mass holds a moved row are within one ulp per weight-setting step on
    their two paths (:data:`_STEP_ULPS`) at every K: the stratum's funding
    factor reads that mass, so they need not be bitwise at K = 2.
  - Every other row is bitwise at K = 2, a power of two, which scales every
    operation exactly.
  - At K = 3, 7 and 15, every other row is within one ulp where the 1/K split
    is its weight's only rounding. Elsewhere it is within one ulp per
    weight-setting step on its two paths (:data:`_STEP_ULPS`).
- **Stage 42, approximately only:**
  - The walk can stop inside a household's clones.
  - Ties are keyed per row (since b8bd45336), so clone 0 need not win them.
  - The K-ulp snap can count one copy fewer.
  - Rule B's exact-total correction moves at most one row, by the corrected
    rows' tolerance. On the CGT nesting scenario it lands on a copy at 7 and 13
    clones, by less than 1e-12 (the last test, through G1).
  - What does nest is each household's family mass and the divided band's
    copy counts.
- **The stand-in:** G1's output is ``clone_first_frame``'s on the stand-in's
  columns, ids and weights, so the tests written on the stand-in hold for G1.
- **The draws behind the clones:** frs_take_up, frs_person_draws and
  frs_household_draws run on every clone, after G1. On the stand-in at K = 3,
  each take-up flag's population support and the apprentice draw's
  population and eligible supports are K = 1's, since the stages count
  support in true source households. Shares, and counts that follow a draw
  (the apprentices flagged, how many values a flag takes), are left out, and
  the extended hours are certain, so the under-two usage population follows
  no draw.

Invented values only. Assertion messages give K, stages, counts and booleans,
never a weight or an id.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import (
    cgt_support,
    frs_person_draws,
    spi_band_donors,
    spi_support,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_INVESTABLE_WEALTH_COLUMNS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_COPIES_COLUMN,
    CGT_SUPPORT_COPY_INDEX_COLUMN,
    CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
    CGT_SUPPORT_SPLIT_STAGE_NAME,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    UKCGTSupportSplitStageTransform,
    cgt_support_income_band,
)
from microcosm.build.uk_runtime.frs_household_draws import (
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import UKFRSPersonDrawsStageTransform
from microcosm.build.uk_runtime.frs_take_up import UKFRSTakeUpStageTransform
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.geographic_expand import (
    GEOGRAPHIC_EXPAND_STAGE_NAME,
    UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS,
    UKGeographicExpandStageTransform,
    expand_uk_geographic_clones,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_BELOW_TEN_BAND,
    support_band,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    PERSON_IS_SPI_INCOME_BAND_CARRIER,
    SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    UK_SPI_INCOME_BAND_DONORS_STAGE_NAME,
    UKSPIIncomeBandDonorStageTransform,
)
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_RECIPIENT_ROLE_COLUMN,
    SPIDonorAgeModel,
)
from microcosm.build.uk_runtime.spi_spine import UKSPISupportChannelStageTransform
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UK_SPI_SUPPORT_STAGE_NAME,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import Frame, WeightKind
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from test_support.microcosm_build.uk_cgt_support import (
    BAND_INCOMES,
    NESTING_INCOMES,
    NESTING_TOP_CELLS,
    NESTING_WEALTH,
    NESTING_WEIGHTS,
    PARAMETERS,
    support_distribution,
    support_frame,
)
from test_support.microcosm_build.uk_clone_first_frames import (
    base_id_tables,
    clone_first_frame,
)
from test_support.microcosm_build.uk_frs_stochastic import _POLICY, _Contract

G1 = GEOGRAPHIC_EXPAND_STAGE_NAME
SPI = UK_SPI_SUPPORT_STAGE_NAME
DONORS = UK_SPI_INCOME_BAND_DONORS_STAGE_NAME
CGT = CGT_SUPPORT_SPLIT_STAGE_NAME
#: The stages each frame goes through, in spine order.
_STAGES = (G1, SPI, DONORS, CGT)
#: The stages whose clone-0 weights are held to K = 1's row by row.
_WEIGHED_STAGES = (G1, SPI, DONORS)
#: The clone counts held to one clone: the plan's 2, 3 and 7, and a K = 15 on
#: the same small frame.
_CLONE_COUNTS = (2, 3, 7, 15)
#: Machine epsilon, the 2.2e-16 of the corrected rows' tolerance.
_EPSILON = float(np.finfo(np.float64).eps)
#: The ulps an uncorrected row may move at a K that is not a power of two,
#: where something besides the 1/K split rounds its weight: one per
#: weight-setting step on its two paths. At K, a row passes through the split
#: and each stage up to this one; at one clone, through those stages. So a
#: stage-16 row passes 2 + 1 steps and a stage-17 row 3 + 2. One ulp holds
#: only where the split alone rounds (:func:`_split_only`). Elsewhere the
#: rounding repeats at K and at one clone: stage 16's pension-age share (0.8
#: kept, 0.2 moved) and its strata's mass sums, and stage 17's funding
#: factors, each computed from a stratum's summed mass. Stage 17's budget
#: also holds, at every K, the incumbents of a stratum whose mass holds a
#: corrected row (:func:`_in_a_reached_stratum`).
_STEP_ULPS = {G1: 1, SPI: 3, DONORS: 5}

# --- the invented frame -----------------------------------------------------------

_HOUSEHOLDS = 90
_PERIOD = "2024"
#: Invented design weights' seed. The test asserts the premise this seed
#: gives, that stage 16's and stage 17's exact-total corrections both move a
#: row of the one-clone frame, so the corrected rows' class is exercised.
_WEIGHT_SEED = 4
#: Given to stage 16, so no engine is needed.
_STATE_PENSION_AGE = 66
#: Stage 16's declared count times this is the share's expected copies: half
#: of the frame's households, so every clone's region and pension-age strata
#: keep copies at K = 15.
_SAMPLE_FRACTION = 0.0045
#: The invented tape's two regions.
_REGIONS = ("SOUTH_EAST", "LONDON")
#: The Table 3 income band of each household's oldest adult, by its pay.
_CGT_BANDS = (0, 37_700, 50_000)
#: Invented taxpayers per reserved band (GBP 200,000 and up).
_BAND_TAXPAYERS = {
    200_000: 300_000.0,
    500_000: 60_000.0,
    1_000_000: 20_000.0,
    2_000_000: 10_000.0,
}
#: The invented Table 3 joint asks band 0 for far more than its pool, so its
#: whole pool is divided at every K; asks band 37,700 for 2 x 1.25 x 1,000 =
#: 2,500, which the walk crosses partway; and asks band 50,000 for nothing.
_EXHAUSTED_BAND = 0
_CROSSING_BAND = 37_700
_CGT_TOP_CELLS = {_EXHAUSTED_BAND: {250_000: 1e9}, _CROSSING_BAND: {250_000: 1_000.0}}
#: The invented tape's pay at each of its six total-income levels: two below
#: GBP 200,000, then one in each reserved band.
_TAPE_PAY = (30_000.0, 80_000.0, 260_000.0, 620_000.0, 1_300_000.0, 2_600_000.0)


def _base_frame() -> Frame:
    """Invented FRS-shaped households with what stages 16, 17 and 42 read, and
    no lineage.

    - **Ids** are ``base_id_tables``': sernums 1 to 90, with one to four people
      in one or two benefit units.
    - **Members.** The first member is the oldest adult. In every third
      household they are 70, so stage 16 has pension-age strata in both
      regions. Elsewhere they are 45 to 54, the tape's age band, so stage 17
      has candidates in every clone. The second member is a 44-year-old adult,
      the rest are children, and the first two are claimants or partners.
    - **Bands and wealth.** The first member's pay puts the household in one of
      three Table 3 income bands. Each household has wealth of its own, so in
      stage 42's walk only a household's own clones and copies tie.
    """

    tables = base_id_tables(_HOUSEHOLDS)
    household = tables.household[["household_id"]].copy()
    sernum = household["household_id"].to_numpy()
    household["region"] = np.asarray(_REGIONS)[sernum % 2]
    for column in UK_CGT_INVESTABLE_WEALTH_COLUMNS:
        household[column] = 0.0
    household[UK_CGT_INVESTABLE_WEALTH_COLUMNS[0]] = (
        1_000.0 * ((37 * sernum) % 101) + sernum
    )
    person = tables.person.copy()
    member = person["person_id"].to_numpy() % 1_000
    owner = person["person_household_id"].to_numpy()
    person["age"] = np.where(
        member == 1,
        np.where(owner % 3 == 1, 70, 45 + owner % 10),
        np.where(member == 2, 44, 9),
    ).astype("int64")
    person["gender"] = np.where((owner + member) % 2 == 0, "MALE", "FEMALE")
    person[SPI_RECIPIENT_ROLE_COLUMN] = member <= 2
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        person[column] = 0.0
    pay = [BAND_INCOMES[_CGT_BANDS[(sernum // 6) % 3]] for sernum in owner.tolist()]
    person["employment_income"] = np.where(member == 1, pay, 0.0)
    return uk_national_frame(
        person=person,
        benunit=tables.benunit,
        household=household,
        household_weights=np.random.default_rng(_WEIGHT_SEED).uniform(
            200.0, 900.0, _HOUSEHOLDS
        ),
        time_period=_PERIOD,
        weight_kind=WeightKind.DESIGN,
    )


def _tape() -> pd.DataFrame:
    """An invented SPI tape: 480 records over six total-income levels, both
    sexes and both of the frame's regions, all in age band 45 to 54
    (AGERANGE 4), with pay their main income. So every reserved band has tape
    mass in every cell the frame's candidates fill."""

    rng = np.random.default_rng(0)
    records = []
    for index in range(480):
        record = dict.fromkeys(SPI_DONOR_REQUIRED_COLUMNS, 0.0)
        pay = _TAPE_PAY[index % 6]
        record.update(
            {
                "SEX": float(1 + (index // 6) % 2),
                "FACT": float(rng.integers(40, 80)),
                # GORCODE 7 is London, 8 the South East.
                "GORCODE": float(7 if (index // 12) % 2 == 0 else 8),
                "AGERANGE": 4.0,
                "PAY": pay,
                "DIVIDENDS": 0.05 * pay,
                "INCBBS": 0.01 * pay,
                "MAINSRCE": 1.0,
            }
        )
        record["TEI"] = record["PAY"]
        record["TII"] = record["DIVIDENDS"] + record["INCBBS"]
        record["TI"] = record["TEI"] + record["TII"]
        records.append(record)
    return pd.DataFrame(records)


def _age_model() -> SPIDonorAgeModel:
    """A flat single-year population, so the engine-free lane needs no engine."""

    return SPIDonorAgeModel(
        populations={
            (sex, age): 100.0 for sex in ("MALE", "FEMALE") for age in range(91)
        },
        state_pension_age=_STATE_PENSION_AGE,
        source="fixture",
    )


def _packaged_stage(name: str) -> SourceStageSpec:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[name]


def _cgt_transform(top_cells) -> UKCGTSupportSplitStageTransform:
    return UKCGTSupportSplitStageTransform(
        stage=_packaged_stage(CGT),
        distribution=support_distribution(top_cells),
        parameters=PARAMETERS,
    )


# --- the runs ---------------------------------------------------------------------

#: Each stage's exact-total correction. The runs record the weights each is
#: handed, so the rows it moved are where the stage's output differs from them.
_CORRECTIONS = {
    SPI: (spi_support, "_importance_weights_with_exact_total"),
    DONORS: (spi_band_donors, "_exact_incumbent_total"),
    CGT: (cgt_support, "_exact_support_weights"),
}


@dataclass(frozen=True)
class _Run:
    """One clone count's frames, each stage's result, and the rows each
    stage's exact-total correction moved, by household id."""

    clones: int
    frames: Mapping[str, Frame]
    results: Mapping[str, object]
    moved: Mapping[str, frozenset[int]]
    uncorrected: Mapping[str, pd.Series]


def _recording(function: Callable, calls: list[np.ndarray]) -> Callable:
    def recorded(values, *args, **kwargs):
        calls.append(np.array(values, dtype=float, copy=True))
        return function(values, *args, **kwargs)

    return recorded


def _run(base: Frame, clones: int, handed: Mapping[str, list[np.ndarray]]) -> _Run:
    transforms = {
        G1: UKGeographicExpandStageTransform(_packaged_stage(G1), clones=clones),
        SPI: UKSPISupportChannelStageTransform(
            stage=_packaged_stage(SPI),
            sample_fraction=_SAMPLE_FRACTION,
            state_pension_age=_STATE_PENSION_AGE,
        ),
        DONORS: UKSPIIncomeBandDonorStageTransform(
            "invented.tab",
            stage=_packaged_stage(DONORS),
            donor_table=_tape(),
            band_taxpayers=_BAND_TAXPAYERS,
            age_model=_age_model(),
        ),
        CGT: _cgt_transform(_CGT_TOP_CELLS),
    }
    frames: dict[str, Frame] = {}
    results: dict[str, object] = {}
    moved: dict[str, frozenset[int]] = {}
    uncorrected: dict[str, pd.Series] = {}
    frame = base
    for stage, transform in transforms.items():
        for calls in handed.values():
            calls.clear()
        frame = transform(frame)
        frames[stage], results[stage] = frame, transform.last_result
        if stage not in handed:
            continue
        calls = handed[stage]
        assert len(calls) == 1, f"{stage} corrected its total {len(calls)} times."
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float)
        moved[stage] = frozenset(ids[weights != calls[0]].tolist())
        uncorrected[stage] = pd.Series(calls[0], index=ids)
    return _Run(clones, frames, results, moved, uncorrected)


@pytest.fixture(scope="module")
def runs() -> dict[int, _Run]:
    """The invented frame through G1 and stages 16, 17 and 42 at K = 1 and at
    every K of :data:`_CLONE_COUNTS`, each exact-total correction recorded."""

    handed: dict[str, list[np.ndarray]] = {stage: [] for stage in _CORRECTIONS}
    with pytest.MonkeyPatch.context() as patch:
        for stage, (module, name) in _CORRECTIONS.items():
            patch.setattr(
                module, name, _recording(getattr(module, name), handed[stage])
            )
        base = _base_frame()
        return {clones: _run(base, clones, handed) for clones in (1, *_CLONE_COUNTS)}


# --- helpers ------------------------------------------------------------------------


def _weights(frame: Frame) -> pd.Series:
    return pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=frame.table("household")["household_id"].to_numpy(),
    )


def _clone_zero_ids(frame: Frame) -> set[int]:
    household = frame.table("household")
    index = household[ladder_clone_index_column("household")]
    return set(household.loc[index.eq(0), "household_id"].tolist())


def _mass_scale_report(runs: Mapping[int, _Run]) -> str:
    """Stage 17's mass scale at each K, as booleans: equal to K = 1's, and one."""

    one = runs[1].results[DONORS].mass_scale
    return "stage 17's mass scale: " + "; ".join(
        f"K={clones} equal to K=1's {run.results[DONORS].mass_scale == one}, "
        f"one {run.results[DONORS].mass_scale == 1.0}"
        for clones, run in runs.items()
    )


def _funding_strata(frame: Frame) -> dict[int, str]:
    """Each household's stage-17 funding stratum (its region within its
    clone), as the stage labels it."""

    household = frame.table("household")
    labels = spi_band_donors._funding_strata(household)
    return dict(zip(household["household_id"].tolist(), labels.tolist(), strict=True))


def _reached(one: _Run, many: _Run, stage: str) -> frozenset[int]:
    """The one-clone households an exact-total correction moved by
    ``stage``, at this stage or an earlier one, at one clone or in clone 0
    at K."""

    done = _STAGES[1 : _STAGES.index(stage) + 1]
    return frozenset(
        household_id
        for run in (one, many)
        for earlier in done
        for household_id in run.moved[earlier]
    )


def _in_a_reached_stratum(one: _Run, many: _Run, stage: str) -> frozenset[int]:
    """From stage 17 on, the one-clone incumbents of a funding stratum whose
    stage-16 mass holds a row stage 16's correction moved, at one clone or at
    K, that no correction moved themselves (:func:`_reached`).

    The stratum's funding factor reads that mass, so their weights need not
    be bitwise at K = 2; the factor rounds once at K and once at one clone,
    within the stage's step budget. A donor's weight reads no stratum while
    the mass scale is one.
    """

    if stage != DONORS:
        return frozenset()
    hit = set()
    for run in (one, many):
        strata = _funding_strata(run.frames[SPI])
        hit |= {strata[household_id] for household_id in run.moved[SPI]}
    household = one.frames[DONORS].table("household")
    labels = _funding_strata(one.frames[DONORS])
    incumbents = household.loc[
        ~household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR], "household_id"
    ]
    reached = _reached(one, many, stage)
    return frozenset(
        household_id
        for household_id in incumbents.tolist()
        if labels[household_id] in hit and household_id not in reached
    )


def _split_only(one: _Run, stage: str) -> frozenset[int]:
    """The one-clone households whose weight at K is K = 1's over K through
    the 1/K split's one rounding, every later step exact.

    - At G1, every household.
    - At stage 16, the base households without a member at or over State
      Pension age, whose share of 0.5 halves exactly.
    - At stage 17, the donors, whose planned weight over K is scaled by a mass
      scale of one.
    """

    frame = one.frames[stage]
    household = frame.table("household")
    ids = household["household_id"]
    if stage == G1:
        return frozenset(ids.tolist())
    if stage == SPI:
        person = frame.table("person")
        pension_age = person.loc[
            person["age"] >= _STATE_PENSION_AGE, "person_household_id"
        ]
        base = household[support_channel_column("household")].eq(
            BASE_FRS_SUPPORT_CHANNEL
        )
        return frozenset(ids[base & ~ids.isin(set(pension_age))].tolist())
    return frozenset(ids[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]].tolist())


def _clone_index_problems(frame: Frame, clones: int) -> list[str]:
    """Where ``frame`` breaks the clone-index invariant: a table without its
    index, an index of another dtype or other values than exactly 0 to K - 1,
    and members whose index is not their household's and benefit unit's."""

    problems = []
    for entity in UK_NATIONAL_SCHEMA.entities:
        column = ladder_clone_index_column(entity)
        table = frame.table(entity)
        if column not in table.columns:
            problems.append(f"the {entity} table lacks {column!r}")
            continue
        if table[column].dtype != np.dtype("int64"):
            problems.append(f"{column!r} is {table[column].dtype}, not int64")
        if sorted(pd.unique(table[column]).tolist()) != list(range(clones)):
            problems.append(f"{column!r} does not take exactly the values 0 to K - 1")
    person = frame.table("person")
    own = person[ladder_clone_index_column("person")].to_numpy()
    for group in UK_NATIONAL_SCHEMA.group_entities:
        units = frame.table(group).set_index(UK_NATIONAL_SCHEMA.id_column(group))
        theirs = person[UK_NATIONAL_SCHEMA.membership_column(group)].map(
            units[ladder_clone_index_column(group)]
        )
        off = int((theirs.to_numpy() != own).sum())
        if off:
            problems.append(
                f"{off} person row(s) carry another index than their {group}"
            )
    return problems


def _copies_off_their_parents_clone(before: Frame, after: Frame, stage: str) -> int:
    """How many households new at ``stage`` carry another clone index than
    the household they copy. The parent is found by the copy's lineage
    columns, never by id arithmetic: stages 16 and 17 write its id in the
    copy's ``household_source_id``. At stage 42 the parent is the household
    whose support lineage the copy restates."""

    index = ladder_clone_index_column("household")
    parents = before.table("household")
    household = after.table("household")
    new = household.loc[~household["household_id"].isin(set(parents["household_id"]))]
    assert len(new), f"{stage} made no copy to check."
    if stage == CGT:
        keys = [
            support_source_id_column("household"),
            support_clone_index_column("household"),
        ]
        by_key = parents.set_index(keys)[index]
        assert by_key.index.is_unique, "The support lineage does not name one parent."
        found = by_key.reindex(pd.MultiIndex.from_frame(new[keys])).to_numpy()
    else:
        by_id = parents.set_index("household_id")[index]
        found = new[support_source_id_column("household")].map(by_id).to_numpy()
    return int((pd.isna(found) | (found != new[index].to_numpy())).sum())


def _copied(frame: Frame, clone: int) -> set[int]:
    """The households stage 16 copied in ``clone``, by the copies' source ids."""

    household = frame.table("household")
    rows = household[support_channel_column("household")].eq(
        SPI_SYNTHETIC_SUPPORT_CHANNEL
    ) & household[ladder_clone_index_column("household")].eq(clone)
    return set(household.loc[rows, support_source_id_column("household")].tolist())


def _seating(frame: Frame, clone: int) -> dict[int, float]:
    """Stage 17's seats in ``clone``: each donor's source household, and its band."""

    household = frame.table("household")
    donors = household.loc[
        household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]
        & household[ladder_clone_index_column("household")].eq(clone)
    ]
    return dict(
        zip(
            donors[support_source_id_column("household")].tolist(),
            donors[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN].tolist(),
            strict=True,
        )
    )


def _carriers(frame: Frame, clone: int) -> set[int]:
    """Stage 17's carriers in ``clone``, by the person each copy was made from."""

    person = frame.table("person")
    carriers = person.loc[
        person[PERSON_IS_SPI_INCOME_BAND_CARRIER]
        & person[ladder_clone_index_column("person")].eq(clone)
    ]
    return set(carriers[support_source_id_column("person")].tolist())


def _families(frame: Frame) -> pd.DataFrame:
    """Stage 42's households, each with the root of its family: itself for a
    root or an undivided household, else the household it copies, found by
    its support lineage."""

    household = frame.table("household")
    keys = [
        support_source_id_column("household"),
        support_clone_index_column("household"),
    ]
    roots = household.loc[household[CGT_SUPPORT_COPY_INDEX_COLUMN].eq(0)]
    root_of = roots.set_index(keys)["household_id"]
    assert root_of.index.is_unique, "The support lineage does not name one root."
    root = root_of.reindex(pd.MultiIndex.from_frame(household[keys])).to_numpy()
    assert not pd.isna(root).any(), "A copy's support lineage names no root."
    return pd.DataFrame(
        {
            "household_id": household["household_id"].to_numpy(),
            "root": root,
            "source": household[SOURCE_HOUSEHOLD_ID_COLUMN].to_numpy(),
            "support": household[support_clone_index_column("household")].to_numpy(),
            "clone": household[ladder_clone_index_column("household")].to_numpy(),
            "copies": household[CGT_SUPPORT_COPIES_COLUMN].to_numpy(),
            "weight": np.asarray(frame.weights_for("household").values, dtype=float),
        }
    )


def _band_row(result, band: int) -> dict[str, object]:
    (row,) = [
        row for row in result.evidence()["bands"] if row["income_lower_bound"] == band
    ]
    return row


# --- reported first ---------------------------------------------------------------


def test_stage_17_mass_scale_is_reported_first_and_is_one_at_every_k(runs) -> None:
    """Reported first: stage 17's mass scale at K = 1 and at each K.

    Stage 17 funds every clone's donors at one scale, the smallest any clone's
    region allows. So clone 0's weights and funding factors are K = 1's over K
    only while that scale is the one a one-clone spine takes: one at both, or
    bound by a region of clone 0. On this frame every clone's regions fund
    their donors in full, so the scale is exactly one at K = 1 and at every K.
    The weight classes below rest on that. The seats are counted on the
    eligible households in distinct sources, so the seating scale is exactly
    K = 1's.
    """

    report = _mass_scale_report(runs)
    one = runs[1].results[DONORS]
    assert one.mass_scale == 1.0, report
    for clones in _CLONE_COUNTS:
        many = runs[clones].results[DONORS]
        assert many.clone_count == clones
        assert many.mass_scale == one.mass_scale, report
        assert many.seating_scale == one.seating_scale, report
        assert many.evidence()["seating_scale"] == one.evidence()["seating_scale"]


# --- exact ------------------------------------------------------------------------


@pytest.mark.parametrize("clones", (1, *_CLONE_COUNTS))
def test_g1_is_the_clone_first_stand_in(runs, clones: int) -> None:
    """Bit for bit on every column the stand-in carries (the five id columns,
    the household clone index and source id, and the frame's own columns, text
    and booleans among them) and on the weights."""

    base = runs[1].frames[G1]
    expanded = runs[clones].frames[G1]
    stand_in = clone_first_frame(_base_frame(), clones)
    for entity in stand_in.entities:
        table = stand_in.table(entity)
        pd.testing.assert_frame_equal(
            expanded.table(entity)[list(table.columns)], table, check_exact=True
        )
    got, want = expanded.weights_for("household"), stand_in.weights_for("household")
    assert got.kind is want.kind is WeightKind.DESIGN
    assert got.values.tobytes() == want.values.tobytes()
    assert len(expanded.table("household")) == clones * len(base.table("household"))


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_clone_zero_is_the_one_clone_frame_with_its_ids_and_g1_columns(
    runs, clones: int
) -> None:
    one, many = runs[1].frames[G1], runs[clones].frames[G1]
    written = set(UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS)
    for entity in UK_NATIONAL_SCHEMA.entities:
        table = many.table(entity)
        clone_zero = table.loc[table[ladder_clone_index_column(entity)].eq(0)]
        # Clone-major: clone 0 is the frame's first rows, every column and
        # dtype as at one clone, the ids and the source columns among them.
        assert clone_zero.index.tolist() == list(range(len(one.table(entity))))
        pd.testing.assert_frame_equal(clone_zero, one.table(entity), check_exact=True)
        written -= set(table.columns)
    assert not written, "Some of G1's columns were not compared."


@pytest.mark.parametrize("clones", (1, *_CLONE_COUNTS))
def test_every_frame_carries_k_clone_indices_and_every_copy_its_parents(
    runs, clones: int
) -> None:
    run = runs[clones]
    for stage in _STAGES:
        problems = _clone_index_problems(run.frames[stage], clones)
        assert not problems, f"After {stage} at K={clones}: {problems}."
    for before, stage in pairwise(_STAGES):
        off = _copies_off_their_parents_clone(
            run.frames[before], run.frames[stage], stage
        )
        assert off == 0, (
            f"{off} household(s) {stage} made at K={clones} carry another clone "
            "index than the household they copy."
        )


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_stage_16_copies_the_one_clone_households_in_clone_zero(
    runs, clones: int
) -> None:
    """The share divides by the distinct sources, so its probability is K = 1's,
    and each household is drawn on a uniform keyed on its own id, which clone 0
    shares with K = 1. So clone 0 copies exactly K = 1's households, the copies'
    ids included, and the other clones draw on their own ids."""

    one, many = runs[1], runs[clones]
    reference = _copied(one.frames[SPI], 0)
    assert reference
    assert _copied(many.frames[SPI], 0) == reference
    copies_once = {
        household_id
        for household_id in _clone_zero_ids(one.frames[SPI])
        if household_id not in _clone_zero_ids(one.frames[G1])
    }
    assert {
        household_id
        for household_id in _clone_zero_ids(many.frames[SPI])
        if household_id not in _clone_zero_ids(many.frames[G1])
    } == copies_once
    for field in (
        "expected_spi_households",
        "spi_source_households",
        "spi_inclusion_probability",
    ):
        assert getattr(many.results[SPI], field) == getattr(one.results[SPI], field)
    assert any(
        _copied(many.frames[SPI], clone) != reference for clone in range(1, clones)
    )


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_stage_17_seats_the_one_clone_households_in_clone_zero(
    runs, clones: int
) -> None:
    """Clone 0 walks exactly K = 1's candidates on the same keyed uniforms with
    the same seats, so it seats K = 1's households in K = 1's bands, with K =
    1's carriers. Every clone seats the same number in each band."""

    one, many = runs[1].frames[DONORS], runs[clones].frames[DONORS]
    report = _mass_scale_report(runs)
    seating = _seating(one, 0)
    assert seating, report
    assert _seating(many, 0) == seating, report
    assert _carriers(many, 0) == _carriers(one, 0), report
    seats = Counter(seating.values())
    for clone in range(clones):
        assert Counter(_seating(many, clone).values()) == seats, report
    rows_once = {row["lower_bound"]: row for row in runs[1].results[DONORS].band_rows}
    for row in runs[clones].results[DONORS].band_rows:
        once = rows_once[row["lower_bound"]]
        assert row["donor_households"] == clones * once["donor_households"]
        assert row["carriers"] == clones * once["carriers"]


# --- tolerances ---------------------------------------------------------------------


def test_the_corrections_reach_rows_the_weight_classes_compare(runs) -> None:
    """The premise that keeps the corrected rows' class from passing empty: on
    this frame stage 16's and stage 17's exact-total corrections each move a
    row of the one-clone frame, which every K's clone 0 is held to. A change
    to either stage's arithmetic can move the corrections elsewhere; then pick
    another :data:`_WEIGHT_SEED` at which this holds."""

    one = runs[1]
    for stage in (SPI, DONORS):
        assert one.moved[stage] & _clone_zero_ids(one.frames[stage]), stage
    # And stage 17 holds incumbents of a stratum stage 16's correction
    # reached that no correction moved, so their class is exercised too.
    for clones in _CLONE_COUNTS:
        assert _in_a_reached_stratum(one, runs[clones], DONORS), clones


@pytest.mark.parametrize("stage", _WEIGHED_STAGES)
@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_k_times_clone_zero_weights_nest_on_the_one_clone_weights(
    runs, clones: int, stage: str
) -> None:
    """K x each clone-0 weight against K = 1's, held exactly (as fractions):

    - **Rows an exact-total correction moved** (:func:`_reached`): within
      ``rows x K x 2.2e-16`` relative. ``rows`` is the one-clone frame's
      household count at the stage. On a row of average weight the bound is
      about K ulps of the total: a correction moves its row by about an ulp of
      the total, and K x a clone-0 weight scales that by K.
    - **Stage 17's other incumbents of a stratum a correction reached**
      (:func:`_in_a_reached_stratum`): within the stage's :data:`_STEP_ULPS`
      ulps at every K, K = 2 included.
    - **Every other row at K = 2**: bitwise, since halving is exact.
    - **Every other row at K = 3, 7 and 15**: within one ulp where the 1/K split
      alone rounds (:func:`_split_only`), and within :data:`_STEP_ULPS` ulps
      elsewhere.
    """

    one, many = runs[1], runs[clones]
    reference = _weights(one.frames[stage])
    clone_zero = _clone_zero_ids(many.frames[stage])
    assert clone_zero == set(reference.index), (
        f"Clone 0 at K={clones} after {stage} is not the one-clone frame's households."
    )
    scaled = _weights(many.frames[stage]).loc[reference.index]
    rows = len(reference)
    reached = _reached(one, many, stage)
    in_a_reached_stratum = _in_a_reached_stratum(one, many, stage)
    split_only = _split_only(one, stage)
    power_of_two = clones & (clones - 1) == 0
    corrected_bound = Fraction(rows * clones) * Fraction(_EPSILON)
    classes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for household_id, weight, expected in zip(
        reference.index.tolist(), scaled.tolist(), reference.tolist(), strict=True
    ):
        difference = abs(Fraction(weight) * clones - Fraction(expected))
        if household_id in reached:
            classes["corrected"] += 1
            if difference > corrected_bound * Fraction(expected):
                failures["corrected rows beyond rows x K x 2.2e-16 relative"] += 1
        elif household_id in in_a_reached_stratum:
            limit = _STEP_ULPS[stage]
            classes["in a corrected stratum"] += 1
            if difference > limit * Fraction(float(np.spacing(expected))):
                failures[f"rows in a corrected stratum beyond {limit} ulp(s)"] += 1
        elif power_of_two:
            classes["bitwise"] += 1
            if difference:
                failures["uncorrected rows not bitwise at a power-of-two K"] += 1
        else:
            limit = 1 if household_id in split_only else _STEP_ULPS[stage]
            classes[f"within {limit} ulp(s)"] += 1
            if difference > limit * Fraction(float(np.spacing(expected))):
                failures[f"uncorrected rows beyond {limit} ulp(s)"] += 1
    assert not failures, (
        f"After {stage} at K={clones}, of {rows} clone-0 rows "
        f"({dict(classes)}): {dict(failures)}. {_mass_scale_report(runs)}."
    )
    uncorrected = sum(
        count
        for name, count in classes.items()
        if name not in ("corrected", "in a corrected stratum")
    )
    assert uncorrected, f"No uncorrected row was compared after {stage}."


# --- stage 42, approximately ------------------------------------------------------


@pytest.mark.parametrize("clones", _CLONE_COUNTS)
def test_stage_42_nests_only_approximately(runs, clones: int) -> None:
    """Stage 42 nests only approximately; these are the checks that hold.

    - **Roots.** Every row stage 17 wrote stays a root, so clone 0's roots are
      K = 1's.
    - **Family mass.** Each household's family (root and copies) keeps its
      mass. So K x a clone-0 family's mass is K = 1's within the corrected
      rows' class, wherever the walk stopped and however the family was
      divided.
    - **The divided band.** Band 0's whole pool is divided at every K, so the
      selection nests there. Each clone-0 household in it is divided as at
      K = 1, except where the K-ulp snap applies: a one-clone quotient within
      K ulps (and the weights' tolerance) of an integer can count one copy
      fewer.
    - **The crossed band.** Band 37,700's walk stops past its support mass by
      less than the row that crosses it, at K as at one clone. So the selected
      masses differ by less than the heavier crossing row, not by the same
      households. The walk can stop inside a household's clones, and since
      ties are keyed per row, clone 0 need not be among the clones taken. The
      messages say whether either happened.
    - **Rule B.** The exact-total correction moves at most one row, within the
      corrected rows' class of its uncorrected weight; it can be a copy (the
      last test). Every other root and copy of a divided household weighs
      bitwise w / n.
    """

    one, many = runs[1], runs[clones]
    rows = len(one.frames[CGT].table("household"))
    tolerance = rows * clones * _EPSILON
    families = {k: _families(run.frames[CGT]) for k, run in ((1, one), (clones, many))}

    # Every selected household here is divided (K x its weight exceeds 60),
    # so a household is selected exactly where it has more than one copy.
    for k, run in ((1, one), (clones, many)):
        roots = families[k].loc[families[k]["household_id"] == families[k]["root"]]
        assert run.results[CGT].households_selected == int((roots["copies"] > 1).sum())

    # Roots and family mass.
    masses = {
        k: {
            root: math.fsum(group.tolist())
            for root, group in family.groupby("root")["weight"]
        }
        for k, family in families.items()
    }
    many_roots = families[clones].loc[
        (families[clones]["household_id"] == families[clones]["root"])
        & families[clones]["clone"].eq(0),
        "household_id",
    ]
    assert set(many_roots.tolist()) == set(masses[1])
    off = sum(
        1
        for root, mass in masses[1].items()
        if abs(clones * masses[clones][root] - mass) > tolerance * mass
    )
    assert off == 0, f"{off} clone-0 family mass(es) at K={clones} do not nest."

    # The divided band: whole pool at every K, copy counts as at one clone.
    weights_17 = {
        k: _weights(run.frames[DONORS]) for k, run in ((1, one), (clones, many))
    }
    roots_once = families[1].loc[families[1]["household_id"] == families[1]["root"]]
    bands = cgt_support_income_band(
        one.frames[DONORS].table("person"),
        household_ids=roots_once["household_id"].to_numpy(),
        parameters=PARAMETERS,
    )
    for result in (one.results[CGT], many.results[CGT]):
        row = _band_row(result, _EXHAUSTED_BAND)
        assert row["pool_exhausted"]
        assert row["households_selected"] == row["pool_households"] > 0
    copies = {
        k: family.set_index("household_id")["copies"] for k, family in families.items()
    }
    snapped = miscounted = 0
    for household_id in roots_once.loc[bands == _EXHAUSTED_BAND, "household_id"]:
        quotient = weights_17[1][household_id] / CGT_SUPPORT_MAXIMUM_COPY_WEIGHT
        nearest = round(quotient)
        window = clones * float(np.spacing(float(nearest))) + tolerance * quotient
        counted = int(copies[clones][household_id]) - int(copies[1][household_id])
        if abs(quotient - nearest) <= window:
            snapped += 1
            miscounted += abs(counted) > 1
        else:
            miscounted += counted != 0
    assert miscounted == 0, (
        f"{miscounted} clone-0 household(s) of the divided band at K={clones} "
        f"are divided otherwise than at K=1 ({snapped} in the snap window)."
    )

    # The crossed band: the same mass within the heavier crossing row. Its
    # households at K are grouped by what their clones share, the survey
    # household and the support copy, to say where the walk stopped.
    crossed = {
        k: _band_row(run.results[CGT], _CROSSING_BAND)
        for k, run in ((1, one), (clones, many))
    }
    roots = families[clones].loc[
        families[clones]["household_id"] == families[clones]["root"]
    ]
    in_band = cgt_support_income_band(
        many.frames[DONORS].table("person"),
        household_ids=roots["household_id"].to_numpy(),
        parameters=PARAMETERS,
    )
    taken = roots.loc[in_band == _CROSSING_BAND].assign(
        taken=lambda table: table["copies"] > 1
    )
    by_household = taken.groupby(["source", "support"])["taken"]
    partial = by_household.any() & ~by_household.all()
    clone_zero = taken.loc[taken["clone"].eq(0)].set_index(["source", "support"])
    left = partial & clone_zero["taken"].reindex(partial.index).eq(False)
    walk = (
        f"at K={clones} the walk stopped inside {int(partial.sum())} household(s)' "
        f"clones, {int(left.sum())} of them leaving clone 0"
    )
    for row in crossed.values():
        assert not row["pool_exhausted"], walk
        assert row["selected_mass"] >= row["support_mass"], walk
    assert abs(crossed[clones]["selected_mass"] - crossed[1]["selected_mass"]) < max(
        crossed[1]["heaviest_selected_weight"],
        crossed[clones]["heaviest_selected_weight"],
    ), walk

    # Rule B: at most one row moved, within the corrected class; every other
    # root and copy of a divided household at bitwise w / n.
    for k, run in ((1, one), (clones, many)):
        moved = run.moved[CGT]
        assert len(moved) <= 1, f"{len(moved)} rows took the correction at K={k}."
        weights = _weights(run.frames[CGT])
        for household_id in moved:
            before = run.uncorrected[CGT][household_id]
            assert abs(weights[household_id] - before) <= (
                rows * k * _EPSILON * before
            ), f"Rule B's row at K={k} moved beyond the corrected rows' class."
        family = families[k]
        divided = family.loc[family["copies"] > 1]
        expected = (
            weights_17[k].reindex(divided["root"]).to_numpy()
            / divided["copies"].to_numpy()
        )
        off_rule = set(
            divided.loc[divided["weight"].to_numpy() != expected, "household_id"]
        )
        assert off_rule <= moved, (
            f"{len(off_rule - moved)} row(s) of divided households at K={k} are off "
            "w / n without taking the correction."
        )


# --- rule B on a copy, through G1 ---------------------------------------------------
#
# test_uk_cgt_support.py's nesting scenario (test_support's NESTING_* data):
# nine one-person households. Band 0 holds exact multiples of 60, a weight one
# ulp below 300, a light one and a non-multiple, and asks for far more than its
# pool, so the whole pool is divided at every clone count. Band 37,700 asks for
# nothing, so its household (9,000) and the light one are the households the
# split leaves whole. At 7 and 13 clones none of their clones can carry the
# correction, so spi_support's shared fallback puts it on the last row: the
# last copy of the 7,620 household's last clone (127 rows, copy index 126).


@pytest.mark.parametrize("clones", (1, 2, 3, 7, 11, 13, 15))
def test_rule_b_puts_the_correction_on_a_copy_at_seven_and_thirteen_clones(
    monkeypatch, clones: int
) -> None:
    base = support_frame(
        weights=NESTING_WEIGHTS, wealth=NESTING_WEALTH, incomes=NESTING_INCOMES
    )
    handed: list[np.ndarray] = []
    monkeypatch.setattr(
        cgt_support,
        "_exact_support_weights",
        _recording(cgt_support._exact_support_weights, handed),
    )
    expanded = expand_uk_geographic_clones(base, clones=clones).frame
    split = _cgt_transform(NESTING_TOP_CELLS)(expanded)
    stand_in = _cgt_transform(NESTING_TOP_CELLS)(clone_first_frame(base, clones))
    assert len(handed) == 2
    values = handed[0]
    weights = np.asarray(split.weights_for("household").values, dtype=float)

    # G1 is a drop-in for the stand-in here: the split weighs its output's rows
    # bit for bit as it weighs the stand-in's, and keeps the total exactly.
    assert (
        weights.tobytes()
        == np.asarray(stand_in.weights_for("household").values).tobytes()
    )
    assert (
        split.weights_for("household").total == expanded.weights_for("household").total
    )
    moved = np.flatnonzero(weights != values).tolist()
    if clones not in (7, 13):
        assert moved == []
        return
    assert moved == [len(weights) - 1]
    last = split.table("household").iloc[-1]
    assert bool(last[HOUSEHOLD_IS_CGT_SUPPORT_COPY])
    assert int(last[SOURCE_HOUSEHOLD_ID_COLUMN]) == 4
    assert int(last[ladder_clone_index_column("household")]) == clones - 1
    assert int(last[CGT_SUPPORT_COPY_INDEX_COLUMN]) == 126
    # About an ulp of the total off w / n: below 1e-12 of the copy's weight.
    assert abs(weights[-1] - values[-1]) < 1e-12 * values[-1]


# --- the draws behind the clones ----------------------------------------------------
#
# frs_take_up, frs_person_draws and frs_household_draws run on every clone,
# after G1, and count their receipts' support before the copies, in true
# source households (microcosm#931). Invented households: four families, each
# with a claimant in work, a child under two and a child of three in their
# unit, and an employee of an apprentice's age in a unit of their own; and
# four pensioners living alone. Half are in Scotland.

_DRAW_FAMILIES = 4
_DRAW_PENSIONERS = 4
#: Receipt fields that are not counts: the flags' shares, and how many values
#: a flag takes, which grows where clones draw apart on their own ids.
_DRAW_SHARE_FIELDS = frozenset({"target", "weighted_share", "distinct_values"})


class _CertainHoursContract(_Contract):
    """test_support's take-up contract with the extended hours certain.

    The under-two usage flag's population is the children under two in units
    drawn to claim the extended hours. Each clone draws its own claimants, on
    its own ids, so below a rate of one that population, and its support,
    grows with K; at one it is every such child on every clone."""

    rates = {**_Contract.rates, "extended_childcare": 1.0}


def _draws_frame() -> Frame:
    people: list[dict[str, object]] = []
    units: list[dict[str, object]] = []
    households: list[dict[str, object]] = []
    for sernum in range(1, _DRAW_FAMILIES + _DRAW_PENSIONERS + 1):
        households.append(
            {
                "household_id": sernum,
                "region": "SCOTLAND" if sernum % 2 else "LONDON",
                "household_weight": 300.0 + 40.0 * sernum,
            }
        )
        # (unit, age, employment status, UC claimant or partner)
        if sernum <= _DRAW_FAMILIES:
            members = (
                (1, 38, "FT_EMPLOYED", True),
                (1, 1, "CHILD", False),
                (1, 3, "CHILD", False),
                (2, 22, "FT_EMPLOYED", True),
            )
        else:
            members = ((1, 70, "RETIRED", True),)
        units.extend(
            {"benunit_id": sernum * 100 + unit, "is_married": False}
            for unit in sorted({member[0] for member in members})
        )
        for position, (unit, age, status, claimant) in enumerate(members, start=1):
            people.append(
                {
                    "person_id": sernum * 1000 + position,
                    "person_benunit_id": sernum * 100 + unit,
                    "person_household_id": sernum,
                    "age": age,
                    "employment_status": status,
                    "months_since_last_birthday": 6.0,
                    "is_uc_claimant": claimant,
                    "child_benefit_reported": 0,
                    "pension_credit_reported": 0,
                    "universal_credit_reported": 0,
                }
            )
    return uk_national_frame(
        person=pd.DataFrame(people),
        benunit=pd.DataFrame(units),
        household=pd.DataFrame(households),
        time_period=_PERIOD,
    )


def _draw_counts(frame: Frame) -> dict[str, dict[str, object]]:
    """frs_take_up, frs_person_draws and frs_household_draws on ``frame``, in
    spine order, under their packaged declarations, and the count fields of
    their receipts: each take-up flag's population support, with its nesting
    count or why it is unmeasured, and the apprentice draw's population and
    eligible supports. The flags' shares and the apprentices' masses and
    flagged support are left out: the anchored residual and the apprentice
    walk pool over the clones, whose draws are keyed on their own ids."""

    contract = _CertainHoursContract()
    transforms = (
        UKFRSTakeUpStageTransform(
            contract=contract,
            stage=_packaged_stage("frs_take_up"),
            population_policy=_POLICY,
        ),
        UKFRSPersonDrawsStageTransform(
            contract=contract, stage=_packaged_stage("frs_person_draws")
        ),
        UKFRSHouseholdDrawsStageTransform(
            contract=contract, stage=_packaged_stage("frs_household_draws")
        ),
    )
    counts: dict[str, dict[str, object]] = {}
    for transform in transforms:
        frame = transform(frame)
        evidence = transform.checkpoint_metadata()["evidence"]
        counts[evidence["stage"]] = {
            "flags": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in _DRAW_SHARE_FIELDS
                }
                for row in evidence["take_up_signal"]["flags"]
            ]
        }
        if "apprentice_receipt" in evidence:
            receipt = evidence["apprentice_receipt"]
            counts[evidence["stage"]]["apprentices"] = {
                key: receipt[key] for key in ("population_support", "eligible_support")
            }
    return counts


def test_the_draws_count_in_source_households_on_every_clone(monkeypatch) -> None:
    # The apprentices' relief reads an installed engine; their counts do not.
    monkeypatch.setattr(frs_person_draws, "_installed_uk_system", lambda: None)
    base = _draws_frame()
    one = _draw_counts(clone_first_frame(base, 1))
    three = clone_first_frame(base, 3)
    assert _draw_counts(three) == one

    # Not vacuous: every household flag rests on the eight households, which
    # reads "3-9", and so do the apprentices, while three clones' rows would
    # read another band.
    households = len(base.table("household"))
    assert {
        row["population_support"] for row in one["frs_household_draws"]["flags"]
    } == {UK_BELOW_TEN_BAND}
    assert one["frs_person_draws"]["apprentices"] == {
        "population_support": UK_BELOW_TEN_BAND,
        "eligible_support": UK_BELOW_TEN_BAND,
    }
    assert support_band(3 * households) != UK_BELOW_TEN_BAND
    # Without the lineage, as a node not given it would read the frame, the
    # draws count each clone as a household of its own.
    lineage = [SOURCE_HOUSEHOLD_ID_COLUMN, ladder_clone_index_column("household")]
    rows = Frame(
        {
            entity: (
                three.table(entity).drop(columns=lineage)
                if entity == "household"
                else three.table(entity)
            )
            for entity in three.entities
        },
        three.schema,
        {"household": three.weights_for("household")},
        three.strata,
        metadata=three.metadata,
    )
    assert {
        row["population_support"]
        for row in _draw_counts(rows)["frs_household_draws"]["flags"]
    } == {support_band(3 * households)}
