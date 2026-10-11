"""Fixtures for the ACS local lane's ESI employer premium (microcosm #454).

The lane pools a dense ASEC-by-PUF donor release with the ACS spine. These
builders make both sides small: a donor the ``meps_esi_premiums`` stage ran on
before the PUF clone, and an ACS frame as the PUMS loader hands it over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_local_esi_premiums import (
    prepare_acs_local_esi_premium_donor,
)
from microcosm.build.us_runtime.acs_multispine import AcsMultispineResult
from microcosm.build.us_runtime.acs_pums import AcsPumsSource
from microcosm.build.us_runtime.base_pool import with_optional_acs_spine
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    with_us_esi_premium_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
WAGES = US_ESI_PREMIUMS_WAGE_COLUMN
ASEC_PUF = "asec_puf"
ACS = "acs_2024_1yr"
_STATES = (6, 36, 48, 12, 17, 39, 53, 13)
_GROUP_OFFSETS = {
    "tax_unit": 100_000,
    "spm_unit": 200_000,
    "family": 300_000,
    "marital_unit": 400_000,
}


def _one_person_households(
    person: pd.DataFrame,
    household: pd.DataFrame,
    weights: np.ndarray,
    *,
    stratum: str,
    kind: WeightKind,
) -> Frame:
    tables = {"person": person, "household": household}
    for group, offset in _GROUP_OFFSETS.items():
        person[f"person_{group}_id"] = person["person_household_id"] + offset
        tables[group] = pd.DataFrame(
            {f"{group}_id": household["household_id"] + offset}
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray(weights, dtype=np.float64), kind)},
        pd.Series([stratum] * len(person), dtype=object),
    )


def asec_observations(households: int = 320) -> Frame:
    """One-person ASEC households with the raw fields the stage reads.

    Of every 20 people, 7 are employed policyholders, 2 are policyholders
    without a job, 5 are employed without a policy and 6 are neither. Wages
    are the measured wages: positive exactly for the employed. Age and sex do
    not tell the employed from the rest, so only wages can.
    """

    index = np.arange(households)
    kind = index % 20
    holder = kind < 9
    employed = (kind < 7) | ((kind >= 9) & (kind < 14))
    tier = np.where(holder, np.asarray([1, 2, 3])[index % 3], 0)
    person = pd.DataFrame(
        {
            "person_id": index + 1,
            "person_household_id": index + 1,
            # The CPS record id: with the raw fields, it marks a row the
            # stage derived.
            "PERIDNUM": [f"asec-{position:05d}" for position in index],
            "age": 24.0 + index % 41,
            "is_female": index % 2 == 0,
            WAGES: np.where(employed, 28_000.0 + 850.0 * (index % 97), 0.0),
            "NOW_OWNGRP": np.where(holder, 1, 2),
            "NOW_HIPAID": np.where(holder, np.asarray([1, 2, 2, 3])[index % 4], 0),
            "NOW_GRPFTYP2": tier,
            "NOW_GRPFTYP": np.asarray([0, 1, 1, 2])[tier],
            "PEMLR": np.where(employed, 1, 5),
            "NOEMP": np.where(employed, np.asarray([1, 3, 6, 0])[index % 4], 0),
            "PEIO1COW": np.where(
                employed, np.asarray([4, 4, 5, 2, 3, 1])[index % 6], 0
            ),
            "source_year": 2024,
            "source_household_id": index + 1,
            "source_person_id": 1,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": index + 1,
            "state_fips": np.asarray(_STATES, dtype=np.int64)[index % len(_STATES)],
        }
    )
    return _one_person_households(
        person,
        household,
        900.0 + 37.0 * (index % 11),
        stratum=ASEC_PUF,
        kind=WeightKind.DESIGN,
    )


def dense_donor(households: int = 320, *, puf_wage_seed: int = 5) -> Frame:
    """A dense ASEC-by-PUF donor release that ran the ``meps_esi_premiums`` stage.

    The stage assigns one premium per source person; the PUF clone copies it.
    The clone's wages are then replaced, as PUF imputation replaces them: a
    seeded permutation, so a PUF-detail row's wages no longer say whether its
    source person holds a job. Weights are calibrated, as a release's are.
    """

    staged = with_us_esi_premium_inputs(
        asec_observations(households), seed=0, time_period=2024
    )
    cloned = clone_us_frame_for_puf_support(staged)
    person = cloned.table("person").copy()
    puf = person["person_support_clone_index"].to_numpy() == 1
    rng = np.random.default_rng(puf_wage_seed)
    person.loc[puf, WAGES] = rng.permutation(person.loc[puf, WAGES].to_numpy())
    return Frame(
        {
            entity: person if entity == "person" else cloned.table(entity)
            for entity in cloned.entities
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(cloned.weights_for("household").values, dtype=np.float64),
                WeightKind.CALIBRATED,
            )
        },
        cloned.strata,
    )


def raw_acs(households: int = 280, *, first_id: int = 1, children: int = 0) -> Frame:
    """One-person ACS households as the PUMS loader returns them.

    Three in five adults report wages, at any age. ``first_id`` defaults to 1
    so the ACS ids repeat the donor's, as the two real id spaces may.
    ``children`` appends that many people aged 9, whose ``WAGP`` is blank:
    PUMS asks earnings of people aged 15 and older.
    """

    index = np.arange(households + children)
    child = index >= households
    employed = (index % 5 < 3) & ~child
    person = pd.DataFrame(
        {
            "person_id": index + first_id,
            "person_household_id": index + first_id,
            "AGEP": np.where(child, 9, 23 + index % 43),
            "SEX": np.where(index % 2 == 1, 2, 1),
            "ADJINC": 1_000_000,
            "WAGP": np.where(
                child,
                np.nan,
                np.where(employed, 26_000.0 + 1_100.0 * (index % 83), 0.0),
            ),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": index + first_id,
            "state_fips": np.asarray(_STATES, dtype=np.int64)[
                (index + 3) % len(_STATES)
            ],
            "TYPEHUGQ": 1,
        }
    )
    return _one_person_households(
        person,
        household,
        40.0 + 3.0 * (index % 13),
        stratum=ACS,
        kind=WeightKind.DESIGN,
    )


def transferred_acs(
    premiums: np.ndarray,
    wages: np.ndarray,
    weights: np.ndarray,
    *,
    households: np.ndarray | None = None,
    ages: np.ndarray | None = None,
    first_id: int = 1,
) -> Frame:
    """An ACS frame after the lane's transfers.

    ``premiums`` stands for whatever the ESI transfer drew, so a property can
    range over every draw without fitting a forest. ``households`` gives each
    person's household position (nondecreasing, starting at 0; one person per
    household by default) and ``weights`` one weight per household. ``ages``
    defaults to adults; a NaN wage is a blank the source left.
    """

    index = np.arange(len(premiums))
    members = index if households is None else np.asarray(households, dtype=np.int64)
    count = int(members.max()) + 1
    person = pd.DataFrame(
        {
            "person_id": index + first_id,
            "person_household_id": members + first_id,
            "age": (
                30.0 + index % 40 if ages is None else np.asarray(ages, dtype=float)
            ),
            "is_female": index % 2 == 0,
            WAGES: np.asarray(wages, dtype=np.float64),
            EMPLOYER: np.asarray(premiums, dtype=np.float64),
        }
    )
    household_index = np.arange(count)
    household = pd.DataFrame(
        {
            "household_id": household_index + first_id,
            "state_fips": np.asarray(_STATES, dtype=np.int64)[
                household_index % len(_STATES)
            ],
        }
    )
    return _one_person_households(
        person, household, weights, stratum=ACS, kind=WeightKind.DESIGN
    )


def reweighted(frame: Frame, factor: float | np.ndarray) -> Frame:
    """The frame with its household weights multiplied by ``factor``.

    ``factor`` is one number for every household or one per household.
    """

    weights = frame.weights_for("household")
    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": Weights(
                np.asarray(weights.values, dtype=np.float64) * factor, weights.kind
            )
        },
        frame.strata,
        metadata=frame.metadata,
    )


def pooled(donor: Frame, acs: Frame, *, acs_share: float = 0.5) -> Frame:
    """The lane's pooled frame before the anchor."""

    return with_optional_acs_spine(donor, acs, acs_share=acs_share)


def donor_rows(frame: Frame) -> np.ndarray:
    """People whose premium the stage derived: they carry the CPS record id."""

    return frame.table("person")["PERIDNUM"].notna().to_numpy()


def person_weights(frame: Frame) -> np.ndarray:
    return np.asarray(frame.resolve_weights("person").values, dtype=np.float64)


def employer_total(frame: Frame, rows: np.ndarray | None = None) -> float:
    """Weighted employer premium over ``rows`` (every person by default)."""

    weights = person_weights(frame)
    values = frame.table("person")[EMPLOYER].to_numpy(dtype=np.float64)
    rows = np.ones(len(values), dtype=bool) if rows is None else rows
    return float(weights[rows] @ values[rows])


def household_mass_share(frame: Frame, rows: np.ndarray) -> float:
    """Share of household mass in the households of ``rows``."""

    household = frame.table("household")
    members = frame.table("person").loc[rows, "person_household_id"]
    selected = household["household_id"].isin(members).to_numpy()
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    return float(weights[selected].sum() / weights.sum())


def build_lane(
    donor: Frame,
    acs: Frame,
    *,
    acs_share: float = 0.5,
    n_estimators: int = 20,
    seed: int = 0,
    target_families: dict | None = None,
) -> AcsMultispineResult:
    """Run the lane's staging orchestration on fixture frames.

    Only the PUMS loader is replaced; native mapping, the ESI transfer, the
    pooling and the anchor are the production code.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            acs_multispine, "build_acs_pums_unit_frame", lambda *a, **k: (acs, {})
        )
        return acs_multispine.build_optional_acs_multispine(
            donor,
            AcsPumsSource(Path("unused-hus.zip"), Path("unused-pus.zip")),
            acs_share=acs_share,
            target_families=target_families or {},
            esi_premium_donor_factory=prepare_acs_local_esi_premium_donor,
            seed=seed,
            n_estimators=n_estimators,
        )
