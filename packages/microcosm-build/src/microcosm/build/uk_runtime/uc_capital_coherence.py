"""Late UK Universal Credit capital and take-up coherence stage.

Ordering and determinism contract (adversarial-review finding 4): this stage
runs after the last ``universal_credit_reported`` writer and BEFORE
``cgt_incidence_clone``, so the redraw sees only pre-clone benunit ids and
un-split design/prior-mass weights; clone twins then copy the already-drawn
values byte-for-byte, which is why clone re-keying cannot desynchronize them.
The redraw is deterministic in the twin-build sense used across the spine:
identical frame plus the declared seed reproduces identical draws (uniforms
are identity-keyed by ``benunit_id``; the donor CDF sorts by (capital,
benunit_id) with a stable sort). A different vintage or upstream frame
legitimately produces different draws, as with every seeded stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import UC_CAPITAL_UNAVAILABLE
from microcosm.build.uk_runtime.frs_take_up import (
    UKTakeUpPopulationPolicy,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.build.uk_runtime.uc_relationships import (
    UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS,
    benunit_financial_investment_income,
    frs_uc_couple_mask,
)
from microcosm.frame import Frame

UC_CAPITAL_REDRAW_OUTPUT = "frs_benunit_capital"
UC_CAPITAL_REDRAW_SEED = 0
UC_CAPITAL_REDRAW_SALT = UC_CAPITAL_REDRAW_OUTPUT
UC_CAPITAL_COHERENCE_OUTPUT_COLUMNS = (
    "uc_reported_capital",
    "pension_credit_reported_capital",
)
#: Every SPI benefit unit is redrawn: the channel's incomes are SPI draws,
#: so a copied FRS capital answer no longer belongs to the unit (uk-data#495,
#: microcosm#1095). Donors are the base units with an available answer.
UC_CAPITAL_REDRAW_ROWS = "spi_channel"
UC_CAPITAL_REDRAW_DONOR_ROWS = "base_frs_with_available_capital"
UC_CAPITAL_REPORTER_STATUS = "universal_credit_reported_anchor"
UC_CAPITAL_INVESTMENT_INCOME = (
    " + ".join(UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS)
    + ", summed over benefit-unit members"
)
#: Upper edges of the investment-income bands: none, then up to the income
#: GBP 6,000, 16,000, 50,000 and 200,000 of capital yield at 4%, then above.
UC_CAPITAL_INCOME_BAND_EDGES = (0.0, 240.0, 640.0, 2_000.0, 8_000.0)
UC_CAPITAL_IMPLIED_YIELD = 0.04
UC_CAPITAL_MINIMUM_CELL_DONORS = 20
#: The declared order in which a cell short of donors widens; reporter status
#: is never coarsened.
UC_CAPITAL_COARSENING = (
    "merge dependent-children bands",
    "drop couple status",
    "merge adjacent investment-income bands",
    "reporter status only",
)
_UC_CAPITAL_LEVELS = ("exact_cell", *UC_CAPITAL_COARSENING)
#: The UC capital limits the receipt measures the redraw against.
_UC_CAPITAL_RECEIPT_LIMITS = (6_000.0, 16_000.0)
#: policyengine-uk 2.122.2 counts these household property values as Universal
#: Credit and Pension Credit capital (the ``sources`` lists under
#: ``gov.dwp.universal_credit.means_test.capital`` and
#: ``gov.dwp.pension_credit.income.capital``, beside ``savings`` and
#: ``corporate_wealth``). A recorded figure of 0 or more replaces every source,
#: and the engine takes it as already countable, but TOTCAPB4 counts accounts
#: and assets and no property. So the recorded figures add the unit's share of
#: its household's property (uk-data#495, microcosm#1095).
UC_PROPERTY_CAPITAL_SOURCES = (
    "other_residential_property_value",
    "non_residential_property_value",
)
PENSION_CREDIT_PROPERTY_CAPITAL_SOURCES = ("owned_land", *UC_PROPERTY_CAPITAL_SOURCES)
#: Whom each engine proxy shares household capital over. The UC proxy shares
#: by the unit's claimants and partners (``is_uc_claimant``) and the Pension
#: Credit proxy by its members at or over Pension Credit qualifying age, so a
#: household's units add up to its property, less the Universal Credit share
#: a reporting unit does not record (``UC_PROPERTY_CAPITAL_REPORTER_RULE``).
UC_PROPERTY_CAPITAL_OWNERS = "is_uc_claimant"
PENSION_CREDIT_PROPERTY_CAPITAL_OWNERS = "at_or_over_pension_credit_qualifying_age"
UC_PROPERTY_CAPITAL_SHARE = "uc_property_capital_share"
PENSION_CREDIT_PROPERTY_CAPITAL_SHARE = "pension_credit_property_capital_share"


def _property_share_definition(sources: tuple[str, ...], owners: str) -> str:
    return (
        f"({' + '.join(sources)}) of the household x the unit's {owners} "
        f"members / the household's {owners} members"
    )


def _recorded_capital_definition(share: str) -> str:
    return (
        f"{UC_CAPITAL_REDRAW_OUTPUT} + {share} where {UC_CAPITAL_REDRAW_OUTPUT} "
        f">= 0, else {UC_CAPITAL_REDRAW_OUTPUT}"
    )


UC_PROPERTY_CAPITAL_SHARE_DEFINITION = _property_share_definition(
    UC_PROPERTY_CAPITAL_SOURCES, UC_PROPERTY_CAPITAL_OWNERS
)
#: The carrier plus the unit's property share: what a unit records when it
#: reports no Universal Credit, and what the reporter redraw screens on.
UC_CAPITAL_WITH_PROPERTY_DEFINITION = _recorded_capital_definition(
    UC_PROPERTY_CAPITAL_SHARE
)
#: A unit that reports Universal Credit keeps its receipt: DWP assessed its
#: capital to pay it, and the property share is imputed without that receipt
#: (the WAS file has no Universal Credit column) and shared over every
#: claimant or partner in the household. So such a unit records the carrier
#: alone, the take-up stages' rule that a reported receipt is a fact
#: (microcosm#1095).
UC_PROPERTY_CAPITAL_REPORTER_RULE = "no_share"
UC_REPORTED_CAPITAL_DEFINITION = (
    f"{UC_CAPITAL_REDRAW_OUTPUT} + {UC_PROPERTY_CAPITAL_SHARE} where "
    f"{UC_CAPITAL_REDRAW_OUTPUT} >= 0 and NOT {UC_CAPITAL_REPORTER_STATUS}, "
    f"else {UC_CAPITAL_REDRAW_OUTPUT}"
)
#: The stage's declared derivations; ``_assert_stage_parameters`` refuses a
#: manifest that says otherwise.
UC_CAPITAL_COHERENCE_DERIVED = {
    UC_PROPERTY_CAPITAL_SHARE: UC_PROPERTY_CAPITAL_SHARE_DEFINITION,
    PENSION_CREDIT_PROPERTY_CAPITAL_SHARE: _property_share_definition(
        PENSION_CREDIT_PROPERTY_CAPITAL_SOURCES,
        PENSION_CREDIT_PROPERTY_CAPITAL_OWNERS,
    ),
    "uc_reported_capital": UC_REPORTED_CAPITAL_DEFINITION,
    "pension_credit_reported_capital": _recorded_capital_definition(
        PENSION_CREDIT_PROPERTY_CAPITAL_SHARE
    ),
    "would_claim_uc": "would_claim_uc OR universal_credit_reported_anchor",
}


@dataclass(frozen=True)
class UKUCCapitalCoherenceResult:
    """Output frame and receipt counts for the late coherence transform."""

    frame: Frame
    post_fill_reporter_count: int
    redrawn_spi_reporter_count: int
    refreshed_would_claim_count: int
    redrawn_spi_benefit_units: int = 0
    coarsening_levels: Mapping[str, int] = field(default_factory=dict)
    capital_against_investment_income: Mapping[str, object] = field(
        default_factory=dict
    )
    property_shares: Mapping[str, object] = field(default_factory=dict)

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe stage evidence."""

        return {
            "stage": "uc_capital_coherence",
            "post_fill_reporter_count": self.post_fill_reporter_count,
            "redrawn_spi_reporter_count": self.redrawn_spi_reporter_count,
            "redrawn_spi_benefit_units": self.redrawn_spi_benefit_units,
            "refreshed_would_claim_count": self.refreshed_would_claim_count,
            "redraw_seed": UC_CAPITAL_REDRAW_SEED,
            "redraw_salt": UC_CAPITAL_REDRAW_SALT,
            "minimum_cell_donors": UC_CAPITAL_MINIMUM_CELL_DONORS,
            "coarsening_levels": dict(self.coarsening_levels),
            "capital_against_investment_income": dict(
                self.capital_against_investment_income
            ),
            "property_shares": dict(self.property_shares),
        }


@dataclass(frozen=True)
class UKUCCapitalCoherenceStageTransform:
    """Redraw SPI reporter capital and refresh UC take-up after SPI income."""

    stage: SourceStageSpec
    #: Synthetic fixtures too small for the declared donor floor pass a lower
    #: one; every build runs the declared ``UC_CAPITAL_MINIMUM_CELL_DONORS``.
    minimum_cell_donors: int = UC_CAPITAL_MINIMUM_CELL_DONORS
    #: The engine's ages at the build instant; read from policyengine-uk when
    #: not given, as the take-up stages do.
    population_policy: UKTakeUpPopulationPolicy | None = None
    last_result: UKUCCapitalCoherenceResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = cohere_uc_capital(
            frame,
            minimum_cell_donors=self.minimum_cell_donors,
            population_policy=self.population_policy,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UC_CAPITAL_COHERENCE_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return the completed stage's redraw and refresh receipt."""

        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def cohere_uc_capital(
    frame: Frame,
    *,
    minimum_cell_donors: int = UC_CAPITAL_MINIMUM_CELL_DONORS,
    population_policy: UKTakeUpPopulationPolicy | None = None,
) -> UKUCCapitalCoherenceResult:
    """Make late SPI UC receipt, FRS capital, and take-up flags coherent."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    _require_columns(
        person,
        (
            "person_benunit_id",
            "person_household_id",
            "universal_credit_reported",
            "age",
            UC_PROPERTY_CAPITAL_OWNERS,
            *UC_FINANCIAL_INVESTMENT_INCOME_COLUMNS,
        ),
        label="person",
    )
    _require_columns(
        benunit,
        (
            "benunit_id",
            support_channel_column("benunit"),
            "frs_benunit_capital",
            "dependent_children",
            "would_claim_uc",
        ),
        label="benunit",
    )
    _require_columns(
        household,
        ("household_id", *PENSION_CREDIT_PROPERTY_CAPITAL_SOURCES),
        label="household",
    )
    policy = population_policy or uk_take_up_population_policy(uk_time_period(frame))

    reporter = _post_fill_reporter_anchor(person, benunit)
    capital = pd.to_numeric(
        benunit[UC_CAPITAL_REDRAW_OUTPUT], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan, copy=True)
    # Sentinel equality is exact (#833): -1.0 is a fixed literal every
    # producer writes verbatim, and an isclose band would reclassify a
    # corrupted near-sentinel value as a declared absence.
    valid_domain = np.isfinite(capital) & (
        (capital == UC_CAPITAL_UNAVAILABLE) | (capital >= 0.0)
    )
    if not valid_domain.all():
        raise ValueError(
            "frs_benunit_capital values must be exactly the named unavailable "
            "sentinel or nonnegative; the open interval between them has no "
            "meaning under the -1 contract."
        )

    channel = benunit[support_channel_column("benunit")].astype(str)
    base = channel.eq(BASE_FRS_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    spi = channel.eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    if np.any(~(base | spi)):
        raise ValueError("UC capital coherence requires only FRS and SPI channels.")
    investment = benunit_financial_investment_income(person, benunit)
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    receipt_before = _capital_against_investment_income(
        capital, investment, weights, base=base, spi=spi, reporter=reporter
    )
    redraw = spi
    levels: dict[str, int] = dict.fromkeys(_UC_CAPITAL_LEVELS, 0)
    if redraw.any():
        levels = _redraw_spi_capital(
            benunit,
            person=person,
            weights=weights,
            reporter=reporter,
            base=base,
            redraw=redraw,
            capital=capital,
            investment=investment,
            minimum_cell_donors=minimum_cell_donors,
        )
    receipt_after = _capital_against_investment_income(
        capital, investment, weights, base=base, spi=spi, reporter=reporter
    )

    previous_would_claim = _boolean_values(benunit["would_claim_uc"])
    refreshed_would_claim = previous_would_claim | reporter
    benunit[UC_CAPITAL_REDRAW_OUTPUT] = capital
    # Both programmes read a recorded capital (pe-uk#2018 and #2070,
    # uk-data#513): 0 or more replaces every capital source in the assessable
    # capital, so each adds the unit's share of the household's property to
    # the financial carrier. The unavailable sentinel -1 is the engine's own
    # default, so the household proxy applies there and the sentinel passes
    # through unchanged. A unit that reports Universal Credit keeps its
    # receipt and records no UC share.
    uc_proxy_share = uc_property_capital_share(person, benunit, household)
    uc_share = uc_recorded_property_share(person, benunit, household)
    pension_credit_share = pension_credit_property_capital_share(
        person,
        benunit,
        household,
        qualifying_age=policy.state_pension_age,
    )
    uc_capital = recorded_capital_with_property(capital, uc_share)
    pension_credit_capital = recorded_capital_with_property(
        capital, pension_credit_share
    )
    benunit["uc_reported_capital"] = uc_capital
    benunit["pension_credit_reported_capital"] = pension_credit_capital
    benunit["would_claim_uc"] = refreshed_would_claim
    property_receipt = _property_share_receipt(
        weights=weights,
        base=base,
        spi=spi,
        reporter=reporter,
        carrier=capital,
        shares={
            "universal_credit": (uc_share, uc_capital),
            "pension_credit": (pension_credit_share, pension_credit_capital),
        },
        qualifying_age=policy.state_pension_age,
        uc_proxy_share=uc_proxy_share,
    )

    result_frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result_frame)
    return UKUCCapitalCoherenceResult(
        frame=result_frame,
        post_fill_reporter_count=int(reporter.sum()),
        redrawn_spi_reporter_count=int((redraw & reporter).sum()),
        refreshed_would_claim_count=int((~previous_would_claim & reporter).sum()),
        redrawn_spi_benefit_units=int(redraw.sum()),
        coarsening_levels=levels,
        capital_against_investment_income={
            "before": receipt_before,
            "after": receipt_after,
        },
        property_shares=property_receipt,
    )


def uc_property_capital_share(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> np.ndarray:
    """Each unit's share of its household's countable property for UC.

    The household's ``UC_PROPERTY_CAPITAL_SOURCES`` times the unit's claimants
    and partners (``is_uc_claimant``) over the household's, as
    ``uc_assessable_capital`` shares residual household capital. In
    benefit-unit row order.
    """

    role = person[UC_PROPERTY_CAPITAL_OWNERS]
    if not pd.api.types.is_bool_dtype(role.dtype):
        raise ValueError(f"{UC_PROPERTY_CAPITAL_OWNERS} must be a boolean column.")
    return _household_property_share(
        person,
        benunit,
        household,
        sources=UC_PROPERTY_CAPITAL_SOURCES,
        owner=role.to_numpy(dtype=bool),
    )


def uc_recorded_property_share(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> np.ndarray:
    """The UC property share a unit's recorded capital carries.

    :func:`uc_property_capital_share`, and none for a unit that reports
    Universal Credit (``UC_PROPERTY_CAPITAL_REPORTER_RULE``): its receipt is a
    fact, and the imputed property would end it. The other units of its
    household keep their own shares. In benefit-unit row order.
    """

    share = uc_property_capital_share(person, benunit, household)
    return np.where(_post_fill_reporter_anchor(person, benunit), 0.0, share)


def pension_credit_property_capital_share(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    qualifying_age: int,
) -> np.ndarray:
    """Each unit's share of its household's countable property for Pension Credit.

    The household's ``PENSION_CREDIT_PROPERTY_CAPITAL_SOURCES`` times the
    unit's members at or over ``qualifying_age`` over the household's, as
    ``pension_credit_assessable_capital`` shares household capital. A unit with
    no such member gets 0, as the engine assesses it no Pension Credit
    capital. In benefit-unit row order.
    """

    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(age).all():
        raise ValueError("person.age must be finite to share Pension Credit capital.")
    return _household_property_share(
        person,
        benunit,
        household,
        sources=PENSION_CREDIT_PROPERTY_CAPITAL_SOURCES,
        owner=age >= qualifying_age,
    )


def recorded_capital_with_property(
    carrier: np.ndarray, share: np.ndarray
) -> np.ndarray:
    """The carrier plus the property share; the -1 sentinel passes through."""

    carrier = np.asarray(carrier, dtype=float)
    return np.where(carrier >= 0.0, carrier + np.asarray(share, dtype=float), carrier)


def _household_property_share(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    sources: tuple[str, ...],
    owner: np.ndarray,
) -> np.ndarray:
    values = household[list(sources)].apply(pd.to_numeric, errors="coerce")
    property_value = values.sum(axis=1, min_count=len(sources)).to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(property_value).all() or (property_value < 0.0).any():
        raise ValueError(f"household {list(sources)} must be finite and nonnegative.")
    benunit_household = _benunit_households(benunit, person=person)
    owners = pd.Series(np.asarray(owner, dtype=float))
    unit_owners = (
        owners.groupby(person["person_benunit_id"].to_numpy(), sort=False)
        .sum()
        .reindex(benunit["benunit_id"].to_numpy(), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    household_owners = (
        owners.groupby(person["person_household_id"].to_numpy(), sort=False)
        .sum()
        .reindex(benunit_household, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    household_value = (
        pd.Series(property_value, index=household["household_id"].to_numpy())
        .reindex(benunit_household)
        .to_numpy(dtype=float)
    )
    if np.isnan(household_value).any():
        raise ValueError("Household property does not cover every benefit unit.")
    share = np.zeros(len(benunit), dtype=float)
    owned = household_owners > 0.0
    # A ratio of 1 leaves a one-unit household's property exact.
    share[owned] = household_value[owned] * (
        unit_owners[owned] / household_owners[owned]
    )
    return share


def _property_share_receipt(
    *,
    weights: np.ndarray,
    base: np.ndarray,
    spi: np.ndarray,
    reporter: np.ndarray,
    carrier: np.ndarray,
    shares: Mapping[str, tuple[np.ndarray, np.ndarray]],
    qualifying_age: int,
    uc_proxy_share: np.ndarray,
) -> dict[str, object]:
    """Aggregate receipt of the property the recorded capitals now carry."""

    available = carrier >= 0.0
    limit = _UC_CAPITAL_RECEIPT_LIMITS[-1]
    receipt: dict[str, object] = {"pension_credit_qualifying_age": qualifying_age}
    for programme, (share, recorded) in shares.items():
        with_share = available & (share > 0.0)
        receipt[programme] = {
            name: {
                "benefit_units_with_share": int((rows & with_share).sum()),
                "weighted_share_total": float(
                    (weights * share)[rows & available].sum()
                ),
                "units_moved_above_16000": int(
                    (rows & available & (carrier <= limit) & (recorded > limit)).sum()
                ),
            }
            for name, rows in (("frs", base), ("spi", spi))
        }
    uc_recorded = shares["universal_credit"][1]
    receipt["uc_reporters_recorded_capital_above_16000"] = {
        name: int((rows & reporter & (uc_recorded > limit)).sum())
        for name, rows in (("frs", base), ("spi", spi))
    }
    # The units the reporter rule leaves without the share the proxy gives
    # them, and that share.
    withheld = available & reporter & (uc_proxy_share > 0.0)
    receipt["uc_reporters_keeping_receipt"] = {
        name: {
            "benefit_units": int((rows & withheld).sum()),
            "weighted_share_withheld": float(
                (weights * uc_proxy_share)[rows & withheld].sum()
            ),
            "units_kept_at_or_below_16000": int(
                (
                    rows
                    & withheld
                    & (carrier <= limit)
                    & (carrier + uc_proxy_share > limit)
                ).sum()
            ),
        }
        for name, rows in (("frs", base), ("spi", spi))
    }
    return receipt


def _post_fill_reporter_anchor(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    amounts = pd.to_numeric(
        person["universal_credit_reported"], errors="coerce"
    ).fillna(0.0)
    reporter_ids = person.loc[amounts > 0.0, "person_benunit_id"]
    return benunit["benunit_id"].isin(reporter_ids).to_numpy(dtype=bool)


def _redraw_spi_capital(
    benunit: pd.DataFrame,
    *,
    person: pd.DataFrame,
    weights: np.ndarray,
    reporter: np.ndarray,
    base: np.ndarray,
    redraw: np.ndarray,
    capital: np.ndarray,
    investment: np.ndarray,
    minimum_cell_donors: int = UC_CAPITAL_MINIMUM_CELL_DONORS,
) -> dict[str, int]:
    """Redraw ``redraw`` units' capital from base donors in their cell.

    A cell is reporter status x investment-income band x couple x
    dependent-children band. A cell with fewer than ``minimum_cell_donors``
    donors widens in the declared ``UC_CAPITAL_COARSENING`` order; reporter
    status is never coarsened, and an empty reporter-status pool refuses.
    Returns the number of redrawn units drawn at each level.
    """

    if not isinstance(minimum_cell_donors, int) or minimum_cell_donors <= 0:
        raise ValueError("minimum_cell_donors must be a positive integer.")
    child_band = _dependent_children_band(benunit["dependent_children"])
    couple = frs_uc_couple_mask(person, benunit)
    income_band = _investment_income_band(investment)
    last_band = len(UC_CAPITAL_INCOME_BAND_EDGES)
    # Domain-validated upstream: every non-sentinel value is >= 0.
    donor = base & (capital >= 0.0) & (weights > 0.0)
    target_ids = benunit["benunit_id"].to_numpy()
    draws = stable_identity_uniforms(
        target_ids,
        seed=UC_CAPITAL_REDRAW_SEED,
        salt=UC_CAPITAL_REDRAW_SALT,
    )
    source_capital = capital.copy()
    levels = dict.fromkeys(_UC_CAPITAL_LEVELS, 0)
    cells = sorted(
        set(
            zip(
                reporter[redraw],
                income_band[redraw],
                couple[redraw],
                child_band[redraw],
                strict=True,
            )
        )
    )
    for is_reporter, band, is_couple, children in cells:
        target_cell = (
            redraw
            & (reporter == is_reporter)
            & (income_band == band)
            & (couple == is_couple)
            & (child_band == children)
        )
        same_status = donor & (reporter == is_reporter)
        candidates = [
            (
                "exact_cell",
                same_status
                & (income_band == band)
                & (couple == is_couple)
                & (child_band == children),
            ),
            (
                UC_CAPITAL_COARSENING[0],
                same_status & (income_band == band) & (couple == is_couple),
            ),
            (UC_CAPITAL_COARSENING[1], same_status & (income_band == band)),
            *(
                (
                    UC_CAPITAL_COARSENING[2],
                    same_status & (np.abs(income_band - band) <= width),
                )
                for width in range(1, last_band + 1)
            ),
            (UC_CAPITAL_COARSENING[3], same_status),
        ]
        level, donor_cell = next(
            (
                (name, pool)
                for name, pool in candidates
                if int(pool.sum()) >= minimum_cell_donors
            ),
            candidates[-1],
        )
        if not donor_cell.any():
            raise ValueError(
                "UC capital redraw has no positive-weight base-FRS donors with "
                f"available capital for UC reporter status {bool(is_reporter)}."
            )
        donor_rows = pd.DataFrame(
            {
                "benunit_id": target_ids[donor_cell],
                "capital": source_capital[donor_cell],
                "weight": weights[donor_cell],
            }
        ).sort_values(["capital", "benunit_id"], kind="mergesort")
        donor_values = donor_rows["capital"].to_numpy(dtype=float)
        donor_weights = donor_rows["weight"].to_numpy(dtype=float)
        cdf = np.cumsum(donor_weights) / float(donor_weights.sum())
        selected = np.searchsorted(cdf, draws[target_cell], side="right")
        capital[target_cell] = donor_values[np.minimum(selected, len(cdf) - 1)]
        levels[level] += int(target_cell.sum())
    return levels


def _investment_income_band(investment: np.ndarray) -> np.ndarray:
    """Band 0 for no investment income, then one band per declared edge."""

    values = np.asarray(investment, dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("investment income must be finite and nonnegative.")
    return np.searchsorted(
        np.asarray(UC_CAPITAL_INCOME_BAND_EDGES), values, side="left"
    ).astype(np.int8)


def _capital_against_investment_income(
    capital: np.ndarray,
    investment: np.ndarray,
    weights: np.ndarray,
    *,
    base: np.ndarray,
    spi: np.ndarray,
    reporter: np.ndarray,
) -> dict[str, object]:
    """Units whose investment income implies capital above a UC limit they lack."""

    implied = np.asarray(investment, dtype=float) / UC_CAPITAL_IMPLIED_YIELD
    receipt: dict[str, object] = {}
    for limit in _UC_CAPITAL_RECEIPT_LIMITS:
        incoherent = (implied > limit) & (capital <= limit)
        receipt[f"implied_above_{int(limit)}_capital_at_or_below"] = {
            name: {
                "benefit_units": int((rows & incoherent).sum()),
                "weighted_benefit_units": float(weights[rows & incoherent].sum()),
            }
            for name, rows in (("frs", base), ("spi", spi))
        }
    upper = _UC_CAPITAL_RECEIPT_LIMITS[-1]
    receipt[f"spi_reporters_capital_above_{int(upper)}"] = int(
        (spi & reporter & (capital > upper)).sum()
    )
    return receipt


def _household_to_benunit_weights(
    benunit: pd.DataFrame,
    *,
    person: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
) -> np.ndarray:
    weight_by_household = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"],
    )
    weights = pd.Series(_benunit_households(benunit, person=person)).map(
        weight_by_household
    )
    if weights.isna().any():
        raise ValueError("Household weights do not cover every benefit unit.")
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Mapped benefit-unit weights must be finite and nonnegative.")
    return values


def _benunit_households(benunit: pd.DataFrame, *, person: pd.DataFrame) -> np.ndarray:
    """Each benefit unit's household id, in benefit-unit row order."""

    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    if placements["person_benunit_id"].duplicated().any():
        raise ValueError("Every benefit unit must map to exactly one household.")
    return (
        benunit["benunit_id"]
        .map(placements.set_index("person_benunit_id")["person_household_id"])
        .to_numpy()
    )


def _dependent_children_band(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0.0).any()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError("dependent_children must contain nonnegative integers.")
    return np.minimum(numeric, 3.0).astype(np.int8)


def _boolean_values(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{values.name} must contain only boolean/0/1 values.")
    return numeric.to_numpy(dtype=bool)


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    derived = [
        operation.parameters.get("derived")
        for operation in stage.operations
        if operation.kind == "derive"
    ]
    if derived != [UC_CAPITAL_COHERENCE_DERIVED]:
        raise ValueError(
            "uc_capital_coherence must declare one derive operation with the "
            f"code's derivations {UC_CAPITAL_COHERENCE_DERIVED}, got {derived}."
        )
    redraw = [
        operation
        for operation in stage.operations
        if operation.kind == "redraw_spi_reporter_capital"
    ]
    if len(redraw) != 1:
        raise ValueError(
            "uc_capital_coherence must declare one redraw_spi_reporter_capital "
            "operation."
        )
    parameters = redraw[0].parameters
    expected = {
        "output": UC_CAPITAL_REDRAW_OUTPUT,
        "rows": UC_CAPITAL_REDRAW_ROWS,
        "donor_rows": UC_CAPITAL_REDRAW_DONOR_ROWS,
        "reporter_status": UC_CAPITAL_REPORTER_STATUS,
        "investment_income": UC_CAPITAL_INVESTMENT_INCOME,
        "investment_income_band_edges": list(UC_CAPITAL_INCOME_BAND_EDGES),
        "minimum_cell_donors": UC_CAPITAL_MINIMUM_CELL_DONORS,
        "coarsening": list(UC_CAPITAL_COARSENING),
        "seed": UC_CAPITAL_REDRAW_SEED,
        "salt": UC_CAPITAL_REDRAW_SALT,
        "couple_status": "is_uc_couple",
    }
    actual = {key: parameters.get(key) for key in expected}
    if actual != expected:
        drifted = sorted(key for key in expected if actual[key] != expected[key])
        raise ValueError(
            "uc_capital_coherence redraw parameters drifted on "
            f"{drifted}: expected {expected}, got {actual}."
        )


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, label: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"UC capital coherence {label} columns missing: {missing}.")
