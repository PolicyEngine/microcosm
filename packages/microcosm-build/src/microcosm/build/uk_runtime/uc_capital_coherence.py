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
        }


@dataclass(frozen=True)
class UKUCCapitalCoherenceStageTransform:
    """Redraw SPI reporter capital and refresh UC take-up after SPI income."""

    stage: SourceStageSpec
    #: Synthetic fixtures too small for the declared donor floor pass a lower
    #: one; every build runs the declared ``UC_CAPITAL_MINIMUM_CELL_DONORS``.
    minimum_cell_donors: int = UC_CAPITAL_MINIMUM_CELL_DONORS
    last_result: UKUCCapitalCoherenceResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = cohere_uc_capital(frame, minimum_cell_donors=self.minimum_cell_donors)
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
    frame: Frame, *, minimum_cell_donors: int = UC_CAPITAL_MINIMUM_CELL_DONORS
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
    _require_columns(household, ("household_id",), label="household")

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
    benunit["uc_reported_capital"] = capital.copy()
    # Pension Credit reads the same recorded capital (pe-uk#2018 and #2070,
    # uk-data#513): 0 or more replaces every capital source in its assessable
    # capital, and the unavailable sentinel -1 is the engine's own default, so
    # the household proxy applies there.
    benunit["pension_credit_reported_capital"] = capital.copy()
    benunit["would_claim_uc"] = refreshed_would_claim

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
    )


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
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    counts = placements.groupby("person_benunit_id", sort=False)[
        "person_household_id"
    ].nunique()
    if (counts != 1).any():
        raise ValueError("Every benefit unit must map to exactly one household.")
    household_by_benunit = placements.set_index("person_benunit_id")[
        "person_household_id"
    ]
    weight_by_household = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"],
    )
    mapped_households = benunit["benunit_id"].map(household_by_benunit)
    weights = mapped_households.map(weight_by_household)
    if weights.isna().any():
        raise ValueError("Household weights do not cover every benefit unit.")
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Mapped benefit-unit weights must be finite and nonnegative.")
    return values


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
