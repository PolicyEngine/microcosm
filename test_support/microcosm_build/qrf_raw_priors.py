"""Fixture and QRF spies for the ACS transfer's raw-prior tests.

The ACS transfer chains each family's targets: a later target is fitted on the
donors' observed values of the earlier ones and drawn on the recipients' raw
float64 draws of them (``RegimeGatedQRF.fit_draw_next``). Two exported values
differ from those raw draws:

- a ``discrete_numeric`` target, here ``first_home_mortgage_origination_year``,
  is snapped to the donor support when it is decoded;
- a self-employment component, here ``farm_operations_income`` and
  ``partnership_self_employment_net_earnings``, is set to zero after the
  transfer where the recipient's own total is kept.

:func:`acs_raw_prior_fixture` puts each of those targets ahead of a later
target in its family, and makes the later target's draws move when the chain
feeds it the exported value instead of the raw draw. The spies record or alter
what the chain feeds forward.

The builders are frozen copies: the tests' recorded digests come from these
exact bytes, so edits made to other fixtures cannot move them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.fit import RegimeGatedQRF
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

#: Seed and tree count of the raw-prior fixture.
RAW_PRIOR_FIXTURE_SEED = 29
RAW_PRIOR_FIXTURE_N_ESTIMATORS = 3

ORIGINATION_YEAR = "first_home_mortgage_origination_year"
HSA = "health_savings_account_ald"
FARM_RENT = "farm_rent_income"
ESTATE_FLAG = "estate_income_would_be_qualified"
MORTGAGE_BALANCE = "first_home_mortgage_balance"
#: The fixture's self-employment components, in chain order.
SELF_EMPLOYMENT_COMPONENTS: tuple[str, ...] = (
    "farm_operations_income",
    "partnership_self_employment_net_earnings",
)

#: Families of the raw-prior fixture, in production's relative order: the
#: person family alternates a component and a later target, as the declared
#: ACS plan's person batch ``puf_tax_itemization__batch_5`` does, and the
#: tax-unit family puts the origination year ahead of the HSA deduction, as the
#: tax-unit ``puf_tax_itemization`` family does.
#: ``test_production_draws_the_fixture_targets_in_the_fixture_order`` pins that
#: order against production's batches.
RAW_PRIOR_FIXTURE_FAMILIES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "person": {
        "fixture_self_employment": (
            "farm_operations_income",
            FARM_RENT,
            "partnership_self_employment_net_earnings",
            ESTATE_FLAG,
        ),
    },
    "tax_unit": {
        "fixture_mortgage_vintage": (MORTGAGE_BALANCE, ORIGINATION_YEAR, HSA),
    },
}

_SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))
_COMMON_YEARS = np.asarray([1991.0, 2004.0, 2018.0])
#: Interior support years, each held by three donor units without an HSA.
#: They are in the snap support (every complete donor's year) but absent from
#: the rows the HSA positive forest is fitted on (donors with a positive HSA),
#: so that forest splits the year away from the snap boundaries: some raw year
#: draws then sit on the other side of a split from their snapped value.
_INTERIOR_YEARS = (1997.0, 2011.0)


def _three_sign(
    rng: np.random.Generator,
    size: int,
    *,
    zero: float,
    negative: float,
    mean: float = 7.0,
) -> np.ndarray:
    draw = rng.random(size)
    amount = np.round(rng.lognormal(mean, 1.0, size), 2)
    return np.where(draw < zero, 0.0, np.where(draw < zero + negative, -amount, amount))


def _people(
    rng: np.random.Generator, households: int, *, first_id: int
) -> pd.DataFrame:
    size = 2 * households
    household_ids = first_id + np.repeat(np.arange(households), 2)
    head = np.tile([True, False], households)
    age = np.where(head, rng.integers(20, 86, size), rng.integers(0, 86, size))
    employment = np.where(
        rng.random(size) < 0.35, 0.0, np.round(rng.lognormal(10.3, 0.8, size), 2)
    )
    return pd.DataFrame(
        {
            "person_id": np.arange(first_id, first_id + size, dtype=np.int64),
            "person_household_id": household_ids.astype(np.int64),
            "person_tax_unit_id": (household_ids + 50_000).astype(np.int64),
            "age": age.astype(np.float64),
            "is_female": rng.random(size) < 0.5,
            "is_household_head": head,
            "employment_income_before_lsr": employment,
            "self_employment_income_before_lsr": _three_sign(
                rng, size, zero=0.75, negative=0.05
            ),
            "taxable_interest_income": np.where(
                rng.random(size) < 0.4, 0.0, np.round(rng.lognormal(5.0, 1.5, size), 2)
            ),
        }
    )


def _group_tables(
    person: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    household_ids = np.unique(person["person_household_id"].to_numpy())
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "state_fips": rng.choice([6, 36, 48], len(household_ids)),
        }
    )
    tax_unit = pd.DataFrame(
        {"tax_unit_id": np.unique(person["person_tax_unit_id"].to_numpy())}
    )
    return household, tax_unit


def acs_raw_prior_fixture() -> tuple[Frame, Frame]:
    """A recipient and donor for the raw-prior tests: ``(recipient, donor)``.

    The donor has 160 households of two people, one tax unit each, and every
    family target. Its self-employment components take negative, zero and
    positive values; ``farm_rent_income`` is mostly zero where
    ``farm_operations_income`` is, and the estate flag mostly follows the sign
    of the partnership earnings. 70% of tax units hold a mortgage with an
    origination year, and the HSA deduction grows with that year. The 600
    recipient households lack every target and keep their own
    ``self_employment_income_before_lsr`` (blank on some rows), so the
    transfer draws no total and zeroes every component cell it fills. Some
    recipients lack employment, self-employment or interest income, which
    gives each family four availability patterns.
    """

    rng = np.random.default_rng(43)
    donor_person = _people(rng, 160, first_id=1_000)
    size = len(donor_person)
    operations = _three_sign(rng, size, zero=0.55, negative=0.1)
    donor_person["farm_operations_income"] = operations
    donor_person[FARM_RENT] = np.where(
        operations != 0.0,
        np.round(0.4 * np.abs(operations) * rng.lognormal(0.0, 0.1, size), 2),
        np.where(
            rng.random(size) < 0.05, np.round(rng.lognormal(6.0, 0.5, size), 2), 0.0
        ),
    )
    partnership = _three_sign(rng, size, zero=0.6, negative=0.1, mean=8.0)
    donor_person["partnership_self_employment_net_earnings"] = partnership
    donor_person[ESTATE_FLAG] = np.where(
        partnership > 0, rng.random(size) < 0.85, rng.random(size) < 0.1
    )
    donor_household, donor_tax_unit = _group_tables(donor_person, rng)
    units = len(donor_tax_unit)
    holder = rng.random(units) < 0.7
    year = np.where(holder, rng.choice(_COMMON_YEARS, units), 0.0)
    interior = np.flatnonzero(holder)[: 3 * len(_INTERIOR_YEARS)]
    year[interior] = np.repeat(_INTERIOR_YEARS, 3)
    donor_tax_unit[ORIGINATION_YEAR] = year
    donor_tax_unit[MORTGAGE_BALANCE] = np.where(
        holder, np.round(rng.lognormal(11.5, 0.6, units), 2), 0.0
    )
    hsa_year = np.where(holder, year, rng.choice(_COMMON_YEARS, units))
    hsa = np.where(
        rng.random(units) < 0.85,
        np.round(60.0 * (hsa_year - 1980.0) * rng.lognormal(0.0, 0.03, units), 2),
        0.0,
    )
    hsa[interior] = 0.0
    donor_tax_unit[HSA] = hsa
    donor = Frame(
        {
            "person": donor_person,
            "household": donor_household,
            "tax_unit": donor_tax_unit,
        },
        _SCHEMA,
        {
            "household": Weights(
                np.round(rng.uniform(50.0, 300.0, len(donor_household)), 2),
                WeightKind.CALIBRATED,
            )
        },
        pd.Series("asec_puf", index=donor_person.index, dtype=object),
    )

    recipient_person = _people(rng, 600, first_id=90_000)
    rows = len(recipient_person)
    for column, share in (
        ("employment_income_before_lsr", 0.25),
        ("self_employment_income_before_lsr", 0.2),
        ("taxable_interest_income", 0.3),
    ):
        recipient_person.loc[rng.random(rows) < share, column] = np.nan
    recipient_household, recipient_tax_unit = _group_tables(recipient_person, rng)
    recipient = Frame(
        {
            "person": recipient_person,
            "household": recipient_household,
            "tax_unit": recipient_tax_unit,
        },
        _SCHEMA,
        {
            "household": Weights(
                np.round(rng.uniform(40.0, 90.0, len(recipient_household)), 2),
                WeightKind.DESIGN,
            )
        },
        pd.Series("acs_2024_1yr", index=recipient_person.index, dtype=object),
    )
    return recipient, donor


def origination_year_support(donor: Frame) -> np.ndarray:
    """The donor support the transfer snaps the origination year to.

    Every fixture donor is complete for its family, so this is every donor
    tax unit's year, as the transfer's complete-case encoding takes it.
    """

    return np.unique(donor.table("tax_unit")[ORIGINATION_YEAR].to_numpy(np.float64))


# ---------------------------------------------------------------------------
# QRF spies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainStep:
    """One ``fit_draw_next`` call: what the chain fed in and what it drew.

    ``priors`` is a deep copy of the ``raw_prior_draws`` the caller passed,
    taken before the caller adds the step's own draw to that frame.
    """

    target: str
    priors: pd.DataFrame
    raw_draw: np.ndarray
    index: pd.Index


def prior_recording_qrf(
    steps: list[ChainStep], base: type[RegimeGatedQRF] = RegimeGatedQRF
) -> type[RegimeGatedQRF]:
    """``base`` (the real QRF by default), appending each ``fit_draw_next`` call.

    With one of the spies below as ``base``, ``steps`` records the priors the
    chain passed, before the spy alters them, and the draws the spy produced.
    """

    class PriorRecordingQRF(base):
        def fit_draw_next(
            self,
            frame_or_df: Any,
            recipient_predictors: pd.DataFrame,
            raw_prior_draws: pd.DataFrame,
            **kwargs: Any,
        ) -> Any:
            priors = raw_prior_draws.copy(deep=True)
            result = super().fit_draw_next(
                frame_or_df, recipient_predictors, raw_prior_draws, **kwargs
            )
            steps.append(
                ChainStep(
                    target=result.target,
                    priors=priors,
                    raw_draw=np.array(result.raw_draw, dtype=np.float64, copy=True),
                    index=recipient_predictors.index.copy(),
                )
            )
            return result

    return PriorRecordingQRF


def snapped_prior_qrf(target: str, support: np.ndarray) -> type[RegimeGatedQRF]:
    """The real QRF, conditioning later targets on ``target`` snapped to ``support``.

    This is the change the tests must catch, made as a spy: the chain feeds
    later targets the decoded value's support point instead of the raw draw.
    The snapped values stay float64, so the chain's own prior checks pass.
    """

    class SnappedPriorQRF(RegimeGatedQRF):
        def fit_draw_next(
            self,
            frame_or_df: Any,
            recipient_predictors: pd.DataFrame,
            raw_prior_draws: pd.DataFrame,
            **kwargs: Any,
        ) -> Any:
            if target in raw_prior_draws.columns:
                raw_prior_draws = raw_prior_draws.copy()
                raw_prior_draws[target] = acs_transfer_module._snap_to_support(
                    raw_prior_draws[target].to_numpy(dtype=np.float64), support
                ).astype(np.float64)
            return super().fit_draw_next(
                frame_or_df, recipient_predictors, raw_prior_draws, **kwargs
            )

    return SnappedPriorQRF


def zeroed_prior_qrf(columns: Sequence[str]) -> type[RegimeGatedQRF]:
    """The real QRF, conditioning later targets on ``columns`` set to zero.

    This is the change the tests must catch, made as a spy: the chain feeds
    later targets the self-employment components as the transfer exports them
    (zero) instead of their raw draws.
    """

    class ZeroedPriorQRF(RegimeGatedQRF):
        def fit_draw_next(
            self,
            frame_or_df: Any,
            recipient_predictors: pd.DataFrame,
            raw_prior_draws: pd.DataFrame,
            **kwargs: Any,
        ) -> Any:
            present = [column for column in columns if column in raw_prior_draws]
            if present:
                raw_prior_draws = raw_prior_draws.copy()
                for column in present:
                    raw_prior_draws[column] = np.zeros(
                        len(raw_prior_draws), dtype=np.float64
                    )
            return super().fit_draw_next(
                frame_or_df, recipient_predictors, raw_prior_draws, **kwargs
            )

    return ZeroedPriorQRF


__all__ = [name for name in globals() if not name.startswith("__")]
