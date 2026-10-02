"""UK WAS wealth imputation stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_brma import _benunit_household_map
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.build.uk_runtime.tenure_constants import (
    UK_OWNER_TENURE_CATEGORIES,
    UK_TENURE_CATEGORIES,
    UK_TENURE_OWNED_OUTRIGHT,
    UK_TENURE_OWNED_WITH_MORTGAGE,
    UK_TENURE_PRIVATE_RENT,
    UK_TENURE_SOCIAL_RENT,
    UK_TENURE_TYPE_TO_CATEGORY,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

WAS_DONOR_FILENAME = "was_round_8_hhold_eul_may_2025_230525.tab"
WAS_DONOR_SHA256 = "18b3eb980c02c99f3d8a3254af859bee31682b2bdc11703877677292b3ce9374"
WAS_DONOR_SIZE_BYTES = 39_073_613

#: The tenure category column both sides carry (never a predictor itself: the
#: strata read it and the one-hot flags below enter the forests).
UK_WAS_TENURE_CATEGORY_COLUMN = "tenure_category"
#: One flag per tenure category but the first (owned outright is the base
#: level), built the same way on the donor (``Ten1R8``, ``DVPriRntR8``) and on
#: the recipient (the frame's ``tenure_type``), so housing-association and
#: council renters are matched to social-renter donors (microcosm#1063).
UK_WAS_TENURE_PREDICTORS = tuple(
    f"tenure_{category}" for category in UK_TENURE_CATEGORIES[1:]
)
UK_WAS_WEALTH_PREDICTORS = (
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "num_bedrooms",
    "council_tax",
    *UK_WAS_TENURE_PREDICTORS,
    "region",
)
UK_WAS_ENGINE_PREDICTORS = (
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
)
UK_WAS_DEBT_OUTPUT_COLUMNS = ("mortgage_debt", "consumer_debt")
#: Columns drawn only inside a tenure stratum and zero by rule outside it
#: (microcosm#1063): the main residence exists only for owner-occupiers, and
#: the mortgage on it only on a mortgaged tenure. Each maps to the tenure
#: categories of its stratum; the manifest declares the same mapping as
#: ``stratified_targets`` and the run asserts it. ``mortgage_debt`` itself is
#: not stratified: it is the main-residence mortgage plus the mortgages on
#: other property, which any tenure may hold (review of #1089, item 3).
UK_WAS_STRATIFIED_TARGETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "main_residence_value": UK_OWNER_TENURE_CATEGORIES,
        "main_residence_mortgage": (UK_TENURE_OWNED_WITH_MORTGAGE,),
    }
)
#: Components the chain draws but the stage does not emit: the remainder of a
#: published total over the components the engine reads, so every total is the
#: sum of its drawn parts rather than a draw of its own.
UK_WAS_OTHER_PROPERTY_COLUMN = "other_property_value"
UK_WAS_OTHER_FINANCIAL_COLUMN = "other_financial_assets"
#: The household's mortgages other than the one on its main residence
#: (``HMortGR8`` less ``TotMortR8`` on the donor): buy-to-let, second homes
#: and the rest, held on every tenure, drawn without a stratum.
UK_WAS_OTHER_MORTGAGE_COLUMN = "other_mortgage"
UK_WAS_INTERNAL_COMPONENT_COLUMNS = (
    UK_WAS_OTHER_PROPERTY_COLUMN,
    UK_WAS_OTHER_FINANCIAL_COLUMN,
    UK_WAS_OTHER_MORTGAGE_COLUMN,
)
#: Totals derived from drawn components, in derivation order. WAS round 8
#: satisfies each identity on every donor row (checked when the donor is
#: cleaned): DVPropertyR8 is the sum of the property values, HFINWR8_SUM is
#: at least the listed financial assets, and net financial wealth is gross
#: less the financial liabilities.
UK_WAS_DERIVED_TOTALS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "property_wealth": (
            "owned_land",
            "main_residence_value",
            "other_residential_property_value",
            "non_residential_property_value",
            UK_WAS_OTHER_PROPERTY_COLUMN,
        ),
        "corporate_wealth": ("corporate_wealth_excl_isa", "stocks_and_shares_isa"),
        "gross_financial_wealth": (
            "savings",
            "cash_isa",
            "corporate_wealth",
            UK_WAS_OTHER_FINANCIAL_COLUMN,
        ),
        # HMortGR8 is the main-residence mortgage (TotMortR8, zero off the
        # mortgaged tenure) plus the mortgages on other property, held on any
        # tenure; the engine's mortgage_debt is their sum.
        "mortgage_debt": ("main_residence_mortgage", UK_WAS_OTHER_MORTGAGE_COLUMN),
    }
)
#: ``net_financial_wealth = gross_financial_wealth - consumer_debt -
#: student_loan_balance`` (the liabilities are subtracted, so it is declared
#: apart from the sums above).
UK_WAS_NET_FINANCIAL_LIABILITIES = ("consumer_debt", "student_loan_balance")
#: Absolute tolerance, in pounds, of the donor identity checks.
UK_WAS_IDENTITY_TOLERANCE_GBP = 1.0
UK_WAS_WEALTH_OUTPUT_COLUMNS = (
    "owned_land",
    "property_wealth",
    "corporate_wealth",
    "private_pension_wealth",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "savings",
    "num_vehicles",
    "cash_isa",
    "stocks_and_shares_isa",
    "student_loan_balance",
    "mortgage_debt",
    "consumer_debt",
)
UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
    if column != "student_loan_balance"
)
UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS = tuple(
    column
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
    if column != "net_financial_wealth"
)
UK_WAS_WEALTH_DECLARED_SEEDS = {"was_wealth": 0}
UK_WAS_WEALTH_FIT_NAME = "uk_was_2018_20_wealth"
UK_WAS_WEALTH_STAGE_NAME = "was_wealth"

REGIONS: Mapping[int, str] = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
}
REGION_REMAP = {"NORTHERN_IRELAND": "WALES"}
UK_WAS_ENGINE_PREDICTOR_ENTITIES: Mapping[str, str] = {
    "household_net_income": "household",
    "num_adults": "benunit",
    "num_children": "benunit",
    "private_pension_income": "person",
    "employment_income": "person",
    "self_employment_income": "person",
    "capital_income": "person",
}

#: WAS round-8 ``Ten1R8`` owner codes onto the shared tenure categories: 1 owns
#: outright, 2 is buying with a mortgage and 3 is part rent, part mortgage
#: (UKDS SN 7215 data dictionary). Codes 4 (rented) and 5 (rent-free) split on
#: ``DVPriRntR8``: 1 is a private renting household, anything else rents from a
#: council or housing association (the 1,726 rented rows with ``DVPriRntR8``
#: 2 are exactly the ``LLordR8`` 1 and 2 rows of the pinned tab). Any other
#: code is unclassified, counted in the receipt and read as private rent, the
#: FRS spine's own fallback for an unknown tenure.
WAS_OWNER_TENURE_CODES: Mapping[int, str] = MappingProxyType(
    {
        1: UK_TENURE_OWNED_OUTRIGHT,
        2: UK_TENURE_OWNED_WITH_MORTGAGE,
        3: UK_TENURE_OWNED_WITH_MORTGAGE,
    }
)
WAS_RENTED_TENURE_CODES: tuple[int, ...] = (4, 5)
WAS_PRIVATE_RENT_CODE = 1

#: UKDS negative sentinel codes observed in the round-8 household tab.
#: Recoded to zero ONLY for columns whose domain cannot be negative and
#: which the 2026-08-19 licensed audit found carrying them: vcarnr8 (2 rows
#: of -8) and HBedRmR8 (95.8% -8 - the bedrooms question is effectively
#: unasked in the WAS household file; the predictor-quality question is
#: registered for the end-of-workstream revisit on microcosm#145).
#: DVPriRntR8's -9 is structural not-applicable (not a private renter), so
#: the private-renter reading (code 1) is already correct; genuinely negative
#: domains (net financial wealth, self-employment losses, BHC income) are
#: never recoded. Signed difference vs the incumbent, which trains on the
#: raw sentinel values.
_SENTINEL_CODES = (-9.0, -8.0, -7.0, -6.0)
_SENTINEL_RECODE_COLUMNS = ("num_vehicles", "num_bedrooms")

_RAW_TO_CLEAN = {
    "R8xshhwgt": "weight",
    "DVLUKValR8_sum": "owned_land",
    "DVPropertyR8": "property_wealth",
    "DVFESHARESR8_aggr": "emp_shares_options",
    "DVFShUKVR8_aggr": "uk_shares",
    "DVIISAVR8_aggr": "stocks_and_shares_isa",
    "DVCISAVR8_aggr": "cash_isa",
    "DVFCollVR8_aggr": "unit_investment_trusts",
    "totalpenr8_aggr": "pensions",
    "dvvaldbt_scaper8_aggr": "db_pensions",
    "NumAdultR8": "num_adults",
    "NumCh18R8": "num_children",
    "DVGIPPENR8_AGGR": "private_pension_income",
    "DVGISER8_AGGR": "self_employment_income",
    "DVGIINVR8_aggr": "capital_income",
    "DVGIEMPR8_AGGR": "employment_income",
    "HBedRmR8": "num_bedrooms",
    "GORR8": "region_code",
    "DVPriRntR8": "private_rent_code",
    "CTAmtR8": "council_tax",
    "HFINWNTR8_Sum": "net_financial_wealth",
    # WAS round-8 household derived variables (UKDS SN 7215, DOI
    # 10.5255/UKDA-SN-7215-20; round-8 user guide and derived-variable
    # specification). HMortGR8 is total outstanding mortgage debt on the
    # household's property (uk-data#426's mortgage measure); HFINWNTR8_exSLC_Sum
    # is net financial wealth excluding student loans, so gross less it is the
    # non-student-loan financial liabilities used for consumer_debt; Ten1R8 is
    # the tenure code (1 own outright, 2 buying with mortgage, 3 part own/part
    # rent, 4 rent, 5 rent free, 6 other). The I1 audit on the pinned tab
    # confirmed the tenure reading empirically: mortgage debt is positive in
    # 99.9% of code-2 and 100% of code-3 rows against 6.9% / 2.4% / 10.8% for
    # codes 1 / 4 / 5 (experiments/685-net-new-stages-receipts.md, Part A).
    "HFINWNTR8_exSLC_Sum": "net_financial_wealth_exsl",
    "HFINWR8_SUM": "gross_financial_wealth",
    "HMortGR8": "mortgage_debt",
    # Diagnostics only (microcosm#1063): the main-residence mortgage and the
    # debt on other property, so the receipt can say what the tenure rule on
    # ``mortgage_debt`` leaves out. Neither is a predictor or an output.
    "TotMortR8": "main_residence_mortgage",
    "OthMortR8_sum": "other_property_mortgage",
    "Ten1R8": "tenure_code",
    "DVhvalueR8": "main_residence_value",
    "DVHseValR8_sum": "other_residential_property_value",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVTotinc_bhcR8": "household_net_income",
    "DVSaValR8_aggr": "savings",
    "vcarnr8": "num_vehicles",
    "Tot_LosR8_aggr": "total_loans",
    "Tot_los_exc_SLCR8_aggr": "total_loans_exc_slc",
}


@dataclass
class UKWASWealthResult:
    """Transformed frame, donor-support clip receipt and coherence receipts."""

    frame: Frame
    support_clip: UKSupportClipReceipt
    #: Tenure coherence of the housing columns: the two structural-zero counts
    #: the stage-health gate requires to be zero, and the donor comparisons.
    tenure_coherence: Mapping[str, object] = field(default_factory=dict)
    #: Rows on which a derived total departs from its components (zero by
    #: construction unless the support clip moved a total) and the rows the
    #: donor-range cap adjusted.
    identities: Mapping[str, object] = field(default_factory=dict)

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_WAS_WEALTH_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
            "tenure_coherence": dict(self.tenure_coherence),
            "identities": dict(self.identities),
        }


#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_WAS_WEALTH_MASS_CONSERVATION_REASON = "WAS wealth imputation on the source spine: household weights pass through unchanged and total household mass is conserved."


@dataclass
class UKWASWealthStageTransform:
    """Whole-stage callable for WAS-trained UK wealth imputation.

    Not frozen: like the HMRC restoration stage, the transform carries
    mutable post-run fit-weight evidence for the terminal weights audit.
    """

    stage: SourceStageSpec
    engine: object
    was_tab_path: str | Path | None = None
    donor: pd.DataFrame | None = None
    #: Fit-weight evidence from the most recent run, read by the national
    #: build's weights-audit collector (the HMRC-stage precedent).
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_result: UKWASWealthResult | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        """Return immutable fit-weight evidence from the most recent run."""

        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        donor = (
            clean_was_household_table(self.donor)
            if self.donor is not None
            else clean_was_household_table(
                read_pinned_tab(
                    _require_path(self.was_tab_path), _donor_artifact(self.stage)
                )
            )
        )
        _assert_chain_declaration(self.stage)
        household_predictors = recipient_predictors(frame, self.engine)
        imputation = impute_was_wealth(
            donor,
            household_predictors,
            seed=UK_WAS_WEALTH_DECLARED_SEEDS["was_wealth"],
            n_estimators=_qrf_n_estimators(self.stage),
        )
        self.last_fit_weight_records = imputation.fit_weight_records
        capped, capped_rows = cap_derived_totals_to_donor_range(imputation.draws, donor)
        clip_result = support_clip_to_donor(capped, donor)
        household_draws = clip_result.clipped
        identities = {
            **wealth_identity_violations(household_draws),
            **capped_rows,
        }
        household_draws["num_vehicles"] = (
            np.rint(household_draws["num_vehicles"]).clip(lower=0).astype("int64")
        )
        tenure_coherence = tenure_coherence_receipt(
            donor=donor,
            recipient_category=household_predictors[UK_WAS_TENURE_CATEGORY_COLUMN],
            recipient_draws=household_draws,
            recipient_weights=frame.weights_for("household").values,
        )
        person = frame.table("person").copy()
        household = frame.table("household").copy()
        for column in UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS:
            household[column] = household_draws[column].to_numpy()
        person["student_loan_balance"] = allocate_student_loan_balance_to_people(
            household_balances=household_draws["student_loan_balance"].clip(lower=0),
            household_ids=household["household_id"],
            person=person,
        )
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_WAS_WEALTH_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_result = UKWASWealthResult(
            frame=result,
            support_clip=clip_result.receipt,
            tenure_coherence=tenure_coherence,
            identities=identities,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_WAS_WEALTH_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def clean_was_household_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the WAS donor table with exact lower-case column matching."""

    lowered = {str(column).lower(): column for column in raw.columns}
    if len(lowered) != len(raw.columns):
        raise ValueError("WAS donor has duplicate columns after lower-case matching.")
    renamed: dict[str, str] = {}
    missing: list[str] = []
    for source, target in _RAW_TO_CLEAN.items():
        actual = lowered.get(source.lower())
        if actual is None:
            missing.append(source)
        else:
            renamed[actual] = target
    if missing:
        raise ValueError(f"WAS donor is missing required column(s): {missing}.")
    cleaned = raw.rename(columns=renamed)[list(renamed.values())].copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.fillna(0)
    for column in _SENTINEL_RECODE_COLUMNS:
        values = cleaned[column]
        cleaned[column] = values.where(~values.isin(_SENTINEL_CODES), 0)
    # Private renting households, the flag the Lifetime ISA stage reads on
    # both sides (``was_lisa``); this stage's own tenure predictors are the
    # four-way flags below.
    cleaned["is_renting"] = cleaned["private_rent_code"] == WAS_PRIVATE_RENT_CODE
    # Private pension wealth other than current-employment defined-benefit
    # entitlements (WAS total private pension wealth less DVValDBT_SCAPE;
    # defined-benefit-type components are valued at the SCAPE discount rate,
    # money-purchase components are reported fund values): DC pots, AVCs,
    # current personal pensions, retained DB/DC rights, pensions in payment
    # and pensions expected from a former spouse or partner.
    # The incumbent folds this into corporate_wealth, where the means-tested
    # capital tests count it; pension rights are disregarded capital (UC Regs
    # 2013 Sch 10 para 10 and the parallel HB/JSA/ESA/IS/SPC paragraphs), so
    # the stage emits it as its own column and keeps corporate_wealth to the
    # share-like holdings (uk-data#452).
    cleaned["private_pension_wealth"] = cleaned["pensions"] - cleaned["db_pensions"]
    cleaned["corporate_wealth_excl_isa"] = (
        cleaned["emp_shares_options"]
        + cleaned["uk_shares"]
        + cleaned["unit_investment_trusts"]
    )
    cleaned["corporate_wealth"] = (
        cleaned["corporate_wealth_excl_isa"] + cleaned["stocks_and_shares_isa"]
    )
    cleaned["student_loan_balance"] = (
        cleaned["total_loans"] - cleaned["total_loans_exc_slc"]
    )
    cleaned["consumer_debt"] = (
        cleaned["gross_financial_wealth"] - cleaned["net_financial_wealth_exsl"]
    ).clip(lower=0)
    cleaned[UK_WAS_TENURE_CATEGORY_COLUMN] = was_tenure_category(
        cleaned["tenure_code"], cleaned["private_rent_code"]
    )
    cleaned["tenure_code_unclassified"] = ~cleaned["tenure_code"].isin(
        {*WAS_OWNER_TENURE_CODES, *WAS_RENTED_TENURE_CODES}
    )
    for predictor in UK_WAS_TENURE_PREDICTORS:
        cleaned[predictor] = cleaned[UK_WAS_TENURE_CATEGORY_COLUMN] == (
            predictor.removeprefix("tenure_")
        )
    cleaned[UK_WAS_OTHER_PROPERTY_COLUMN] = _donor_remainder(
        cleaned,
        total="property_wealth",
        components=UK_WAS_DERIVED_TOTALS["property_wealth"][:-1],
    )
    cleaned[UK_WAS_OTHER_FINANCIAL_COLUMN] = _donor_remainder(
        cleaned,
        total="gross_financial_wealth",
        components=UK_WAS_DERIVED_TOTALS["gross_financial_wealth"][:-1],
    )
    cleaned[UK_WAS_OTHER_MORTGAGE_COLUMN] = _donor_remainder(
        cleaned,
        total="mortgage_debt",
        components=UK_WAS_DERIVED_TOTALS["mortgage_debt"][:-1],
    )
    _assert_donor_net_financial_identity(cleaned)
    cleaned["region"] = cleaned["region_code"].map(REGIONS)
    return cleaned[
        [
            *UK_WAS_WEALTH_PREDICTORS,
            "weight",
            UK_WAS_TENURE_CATEGORY_COLUMN,
            "tenure_code_unclassified",
            "is_renting",
            "corporate_wealth_excl_isa",
            *UK_WAS_INTERNAL_COMPONENT_COLUMNS,
            "main_residence_mortgage",
            "other_property_mortgage",
            *UK_WAS_WEALTH_HOUSEHOLD_OUTPUT_COLUMNS,
            "student_loan_balance",
        ]
    ]


def was_tenure_category(
    tenure_code: pd.Series, private_rent_code: pd.Series
) -> pd.Series:
    """The donor's tenure category from ``Ten1R8`` and ``DVPriRntR8``."""

    code = pd.to_numeric(tenure_code, errors="coerce")
    private = pd.to_numeric(private_rent_code, errors="coerce") == WAS_PRIVATE_RENT_CODE
    category = pd.Series(UK_TENURE_PRIVATE_RENT, index=tenure_code.index, dtype=object)
    for owner_code, owner_category in WAS_OWNER_TENURE_CODES.items():
        category[code == owner_code] = owner_category
    rented = code.isin(WAS_RENTED_TENURE_CODES)
    category[rented & ~private] = UK_TENURE_SOCIAL_RENT
    return category


def recipient_tenure_category(tenure_type: pd.Series) -> pd.Series:
    """The recipient's tenure category from the frame's ``tenure_type``."""

    names = tenure_type.map(_enum_name)
    unknown = sorted(set(names) - set(UK_TENURE_TYPE_TO_CATEGORY))
    if unknown:
        raise ValueError(
            f"recipient tenure_type carries values outside the engine's enum: {unknown}."
        )
    return names.map(UK_TENURE_TYPE_TO_CATEGORY)


def _donor_remainder(
    cleaned: pd.DataFrame, *, total: str, components: Sequence[str]
) -> pd.Series:
    """A published total less its listed components, refused when negative.

    The remainder becomes a drawn component, so a donor row whose components
    exceed its total would make the total underivable from its parts.
    """

    remainder = cleaned[total] - cleaned[list(components)].sum(axis=1)
    short = remainder < -UK_WAS_IDENTITY_TOLERANCE_GBP
    if short.any():
        raise ValueError(
            f"WAS donor {total} is below the sum of {list(components)} on "
            f"{int(short.sum())} row(s); the total cannot be derived from its "
            "components."
        )
    return remainder.clip(lower=0.0)


def _assert_donor_net_financial_identity(cleaned: pd.DataFrame) -> None:
    """``net = gross - consumer_debt - student_loan_balance`` on every donor row."""

    derived = (
        cleaned["gross_financial_wealth"]
        - cleaned["consumer_debt"]
        - cleaned["student_loan_balance"]
    )
    off = (derived - cleaned["net_financial_wealth"]).abs() > (
        UK_WAS_IDENTITY_TOLERANCE_GBP
    )
    if off.any():
        raise ValueError(
            "WAS donor net financial wealth is not gross financial wealth less "
            f"consumer debt and the student loan balance on {int(off.sum())} "
            "row(s)."
        )


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize the WAS predictor surface on recipient households.

    Engine variables live at their native entity; person- and benunit-level
    predictors are summed to household grain, reproducing the incumbent's
    ``map_to="household"`` semantics.
    """

    materialized = engine.materialize(
        frame, UK_WAS_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    benunit = frame.table("benunit")
    household_ids = pd.Index(household["household_id"])
    result = pd.DataFrame(index=household.index)
    for predictor in UK_WAS_ENGINE_PREDICTORS:
        entity = UK_WAS_ENGINE_PREDICTOR_ENTITIES[predictor]
        declared = str(engine.variable_metadata(predictor).entity)
        if declared != entity:
            raise ValueError(
                f"engine declares {predictor!r} at entity {declared!r}; "
                f"the WAS wealth stage expects {entity!r}."
            )
        values = np.asarray(materialized[predictor])
        if entity == "household":
            if values.shape != (len(household),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(household)},)."
                )
            result[predictor] = values
        elif entity == "benunit":
            if values.shape != (len(benunit),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(benunit)},)."
                )
            groups = benunit["benunit_id"].map(_benunit_household_map(person))
            summed = pd.Series(values.astype(float)).groupby(groups.to_numpy()).sum()
            result[predictor] = summed.reindex(household_ids).fillna(0.0).to_numpy()
        else:
            if values.shape != (len(person),):
                raise ValueError(
                    f"materialized {predictor!r} has shape {values.shape}; "
                    f"expected ({len(person)},)."
                )
            summed = (
                pd.Series(values.astype(float))
                .groupby(person["person_household_id"].to_numpy())
                .sum()
            )
            result[predictor] = summed.reindex(household_ids).fillna(0.0).to_numpy()
    for predictor in ("num_bedrooms", "council_tax", "region"):
        if predictor not in household.columns:
            raise KeyError(f"recipient household table is missing {predictor!r}.")
        result[predictor] = household[predictor].to_numpy()
    # The donor records the bill the household pays and the spine's
    # council_tax is the liability before council tax reduction, so the
    # predictor is the paid bill (uk-data#496/#499, microcosm#1095).
    if "council_tax_rebate" not in household.columns:
        raise KeyError("recipient household table is missing 'council_tax_rebate'.")
    result["council_tax"] = np.maximum(
        household["council_tax"].to_numpy(dtype=float)
        - household["council_tax_rebate"].to_numpy(dtype=float),
        0.0,
    )
    result["region"] = result["region"].map(_enum_name).replace(REGION_REMAP)
    if "tenure_type" not in household.columns:
        raise KeyError("recipient household table is missing 'tenure_type'.")
    category = recipient_tenure_category(household["tenure_type"])
    for predictor in UK_WAS_TENURE_PREDICTORS:
        result[predictor] = (category == predictor.removeprefix("tenure_")).to_numpy()
    result[UK_WAS_TENURE_CATEGORY_COLUMN] = category.to_numpy()
    return result.loc[:, (*UK_WAS_WEALTH_PREDICTORS, UK_WAS_TENURE_CATEGORY_COLUMN)]


@dataclass(frozen=True)
class UKWASWealthImputationResult:
    """WAS wealth draws plus the auditable fit-weight evidence."""

    draws: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]
    #: The per-segment RNG roots derived from the declared stage seed.
    segment_seeds: tuple[int, ...] = ()


#: The chain's segments, each with its own model and child seed:
#: 1 land; 2 main residence (owner stratum); 3 the other property components;
#: 4 pension and share-like wealth; 5 the financial components, vehicles and
#: the student loan balance; 6 mortgage debt (mortgaged stratum); 7 consumer
#: debt. Totals are derived between segments from the components drawn so far.
UK_WAS_CHAIN_SEGMENTS = 8


def was_wealth_segment_seeds(
    seed: int, segments: int = UK_WAS_CHAIN_SEGMENTS
) -> tuple[int, ...]:
    """Derive one independent RNG root per chain segment from the stage seed.

    :meth:`RegimeGatedQRF.start_chain` spawns its fit and draw streams from
    the model seed on every call, so one model reused across segments would
    restart the same streams each time and couple the k-th target of every
    segment (the same quantile and sign-gate uniforms per recipient). On the
    licensed donor that coupling collapsed P(shares > 0 | property_wealth = 0)
    to 0.011 against 0.055 observed; the production child seeds recover 0.039
    (hold-out receipt re-run with exactly this derivation). The declared stage
    seed stays the root and the children are deterministic; ``spawn`` is
    prefix-stable, so the first ``k`` seeds do not depend on ``segments``.
    """

    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(seed)).spawn(int(segments))
    )


def impute_was_wealth(
    donor: pd.DataFrame,
    recipient_predictor_frame: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
    segments: int = UK_WAS_CHAIN_SEGMENTS,
) -> UKWASWealthImputationResult:
    """Fit segmented checkpointed QRF chains and draw WAS wealth outputs.

    The targets of ``UK_WAS_STRATIFIED_TARGETS`` are fitted on the donors of
    their tenure stratum and drawn for the recipients of the same stratum
    only; every other recipient holds zero by rule. The totals of
    ``UK_WAS_DERIVED_TOTALS`` and ``net_financial_wealth`` are never drawn:
    each is computed from its drawn components, so the accounting identities
    the donor satisfies hold on every recipient.

    ``segments`` runs only the first *k* chain segments (a test seam: each
    segment draws from its own child seed, so the earlier segments' draws are
    identical whether or not the later ones run).
    """

    from microcosm.fit import RegimeGatedQRF

    if not 1 <= int(segments) <= UK_WAS_CHAIN_SEGMENTS:
        raise ValueError(
            f"segments must be between 1 and {UK_WAS_CHAIN_SEGMENTS}, got {segments!r}."
        )
    for table, side in ((donor, "donor"), (recipient_predictor_frame, "recipient")):
        if UK_WAS_TENURE_CATEGORY_COLUMN not in table.columns:
            raise KeyError(
                f"WAS {side} table is missing {UK_WAS_TENURE_CATEGORY_COLUMN!r}."
            )
    donor_category = donor[UK_WAS_TENURE_CATEGORY_COLUMN].astype(str).to_numpy()
    recipient_category = (
        recipient_predictor_frame[UK_WAS_TENURE_CATEGORY_COLUMN].astype(str).to_numpy()
    )

    donor_encoded, recipient_encoded, encoded_predictors = encode_qrf_predictor_pair(
        donor, recipient_predictor_frame
    )
    # The tenure rule on the donor's conditioning copies: a later target is
    # fitted on values the recipients can hold. The main-residence mortgage is
    # zero off the mortgaged tenure on the pinned tab already; the mortgages
    # on other property are carried on every tenure.
    for target, categories in UK_WAS_STRATIFIED_TARGETS.items():
        donor_encoded.loc[~np.isin(donor_category, categories), target] = 0.0
    segment_seeds = was_wealth_segment_seeds(seed)
    segment_models = iter(
        RegimeGatedQRF(n_estimators=n_estimators, seed=segment_seed)
        for segment_seed in segment_seeds
    )
    raw = pd.DataFrame(index=recipient_encoded.index)
    fit_records: list[FitWeightRecord] = []

    def run_segment(
        base_predictors: Sequence[str],
        targets: Sequence[str],
        *,
        stratum: Sequence[str] | None = None,
    ) -> None:
        model = next(segment_models)
        if stratum is None:
            donor_rows, recipient_rows = donor_encoded, recipient_encoded
        else:
            donor_rows = donor_encoded.loc[np.isin(donor_category, stratum)]
            recipient_rows = recipient_encoded.loc[np.isin(recipient_category, stratum)]
            if donor_rows.empty:
                raise ValueError(
                    f"WAS donor has no household in the tenure stratum "
                    f"{list(stratum)} that {list(targets)} is fitted on."
                )
        state = model.start_chain(
            donor_rows,
            list(base_predictors),
            list(targets),
            weights="weight",
        )
        segment_raw = pd.DataFrame(index=recipient_rows.index)
        recipient_base = recipient_rows.loc[:, list(base_predictors)]
        for target in targets:
            raw[target] = 0.0
            if recipient_rows.empty:
                # No recipient holds the target: nothing is fitted, and the
                # audit still records the weight kind the chain resolved.
                fit_records.append(
                    FitWeightRecord(
                        f"{UK_WAS_WEALTH_FIT_NAME}:{target}", state.weight_kind
                    )
                )
                continue
            result = model.fit_draw_next(
                donor_rows,
                recipient_base,
                segment_raw,
                state=state,
                weights="weight",
            )
            fit_records.append(
                FitWeightRecord(
                    f"{UK_WAS_WEALTH_FIT_NAME}:{target}", result.weight_kind
                )
            )
            segment_raw[target] = result.raw_draw
            raw.loc[recipient_rows.index, target] = np.asarray(
                result.raw_draw, dtype=float
            )
            state = result.state

    def derive_total(total: str) -> None:
        raw[total] = raw.loc[:, list(UK_WAS_DERIVED_TOTALS[total])].sum(axis=1)

    base = encoded_predictors
    run_segment(base, ("owned_land",))
    if segments == 1:
        return _partial_result(raw, fit_records, segment_seeds)
    recipient_encoded["owned_land"] = raw["owned_land"]
    run_segment(
        (*base, "owned_land"),
        ("main_residence_value",),
        stratum=UK_WAS_STRATIFIED_TARGETS["main_residence_value"],
    )
    if segments == 2:
        return _partial_result(raw, fit_records, segment_seeds)
    recipient_encoded["main_residence_value"] = raw["main_residence_value"]
    run_segment(
        (*base, "owned_land", "main_residence_value"),
        (
            "other_residential_property_value",
            "non_residential_property_value",
            UK_WAS_OTHER_PROPERTY_COLUMN,
        ),
    )
    derive_total("property_wealth")
    if segments == 3:
        return _partial_result(raw, fit_records, segment_seeds)
    donor_encoded["corporate_wealth"] = donor_encoded["corporate_wealth"].astype(float)
    donor_encoded["private_pension_wealth"] = donor_encoded[
        "private_pension_wealth"
    ].astype(float)
    recipient_encoded["property_wealth"] = raw["property_wealth"]
    # Private pension wealth is drawn first in the position the old folded
    # corporate_wealth (84.7% pension by donor mass) occupied; the share-like
    # components condition on it, and the fold into corporate_wealth follows.
    run_segment(
        (*base, "owned_land", "property_wealth"),
        (
            "private_pension_wealth",
            "corporate_wealth_excl_isa",
            "stocks_and_shares_isa",
        ),
    )
    derive_total("corporate_wealth")
    if segments == 4:
        return _partial_result(raw, fit_records, segment_seeds)
    recipient_encoded["private_pension_wealth"] = raw["private_pension_wealth"]
    recipient_encoded["corporate_wealth"] = raw["corporate_wealth"]
    # Downstream targets condition on both components, carrying the
    # information the old folded corporate_wealth supplied as one column.
    run_segment(
        (
            *base,
            "owned_land",
            "property_wealth",
            "private_pension_wealth",
            "corporate_wealth",
        ),
        (
            "savings",
            "cash_isa",
            UK_WAS_OTHER_FINANCIAL_COLUMN,
            "num_vehicles",
            "student_loan_balance",
        ),
    )
    derive_total("gross_financial_wealth")
    if segments == 5:
        return _partial_result(raw, fit_records, segment_seeds)
    prior_outputs = tuple(
        column
        for column in UK_WAS_WEALTH_OUTPUT_COLUMNS
        if column not in (*UK_WAS_DEBT_OUTPUT_COLUMNS, "net_financial_wealth")
    )
    for output in prior_outputs:
        recipient_encoded[output] = raw[output]
    run_segment(
        (*base, *prior_outputs),
        ("main_residence_mortgage",),
        stratum=UK_WAS_STRATIFIED_TARGETS["main_residence_mortgage"],
    )
    if segments == 6:
        return _partial_result(raw, fit_records, segment_seeds)
    recipient_encoded["main_residence_mortgage"] = raw["main_residence_mortgage"]
    # The mortgages on other property are held on every tenure (an outright
    # owner's or a renter's buy-to-let), so they are drawn without a stratum
    # and conditioned on the property the chain already placed.
    run_segment(
        (*base, *prior_outputs, "main_residence_mortgage"),
        (UK_WAS_OTHER_MORTGAGE_COLUMN,),
    )
    derive_total("mortgage_debt")
    if segments == 7:
        return _partial_result(raw, fit_records, segment_seeds)
    recipient_encoded["mortgage_debt"] = raw["mortgage_debt"]
    run_segment((*base, *prior_outputs, "mortgage_debt"), ("consumer_debt",))
    raw["net_financial_wealth"] = derived_net_financial_wealth(raw)
    return UKWASWealthImputationResult(
        draws=raw.loc[:, (*UK_WAS_WEALTH_OUTPUT_COLUMNS, *UK_WAS_DRAWN_ONLY_COLUMNS)],
        fit_weight_records=tuple(fit_records),
        segment_seeds=segment_seeds,
    )


#: Drawn components the stage does not emit, kept beside the outputs so the
#: totals can be re-derived after the donor-range cap.
UK_WAS_DRAWN_ONLY_COLUMNS = (
    "corporate_wealth_excl_isa",
    "main_residence_mortgage",
    *UK_WAS_INTERNAL_COMPONENT_COLUMNS,
)


def derived_net_financial_wealth(draws: pd.DataFrame) -> pd.Series:
    """Gross financial wealth less the financial liabilities."""

    liabilities = sum(
        draws[column].clip(lower=0.0) for column in UK_WAS_NET_FINANCIAL_LIABILITIES
    )
    return draws["gross_financial_wealth"] - liabilities


def _partial_result(
    raw: pd.DataFrame,
    fit_records: list[FitWeightRecord],
    segment_seeds: tuple[int, ...],
) -> UKWASWealthImputationResult:
    produced = [
        column
        for column in (*UK_WAS_WEALTH_OUTPUT_COLUMNS, *UK_WAS_DRAWN_ONLY_COLUMNS)
        if column in raw
    ]
    return UKWASWealthImputationResult(
        draws=raw.loc[:, produced],
        fit_weight_records=tuple(fit_records),
        segment_seeds=segment_seeds,
    )


#: Components a derived total gives up, in order, when the sum of donor-valued
#: draws leaves the donor's range for the total. The remainder goes first; the
#: main residence and the share-like total are never reduced (each is within
#: its own donor range, which the total's range contains).
_UK_WAS_TOTAL_CAP_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "property_wealth": (
            UK_WAS_OTHER_PROPERTY_COLUMN,
            "non_residential_property_value",
            "other_residential_property_value",
            "owned_land",
        ),
        "corporate_wealth": ("corporate_wealth_excl_isa", "stocks_and_shares_isa"),
        "gross_financial_wealth": (
            UK_WAS_OTHER_FINANCIAL_COLUMN,
            "cash_isa",
            "savings",
        ),
        "mortgage_debt": (UK_WAS_OTHER_MORTGAGE_COLUMN, "main_residence_mortgage"),
    }
)


def cap_derived_totals_to_donor_range(
    draws: pd.DataFrame, donor: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep every derived total inside the donor's range without breaking it.

    Each component is a donor value, but their sum can pass the donor's
    largest total. The excess is taken out of the components in the declared
    order and the total recomputed, so the identity survives where a clip of
    the total alone would break it. Returns the adjusted draws and the rows
    adjusted per total (zero on a build whose joint draws stay in range).
    """

    capped = draws.copy()
    fired: dict[str, int] = {}

    def reduce(total: str, excess: pd.Series) -> None:
        remaining = excess.clip(lower=0.0)
        for component in _UK_WAS_TOTAL_CAP_ORDER[total]:
            take = np.minimum(remaining, capped[component].clip(lower=0.0))
            capped[component] = capped[component] - take
            remaining = remaining - take
        recomputed = capped.loc[:, list(UK_WAS_DERIVED_TOTALS[total])].sum(axis=1)
        # The reduced components sum back to the maximum up to float
        # rounding; a sum a few ulp above it would be clipped as a real
        # excess by the support clip, so it lands on the maximum exactly
        # (inside the identity tolerance).
        maximum = float(pd.to_numeric(donor[total], errors="coerce").max())
        capped[total] = np.where(
            (recomputed > maximum) & (recomputed - maximum <= 1e-6),
            maximum,
            recomputed,
        )

    for total, components in UK_WAS_DERIVED_TOTALS.items():
        # Recomputed first: a reduced share-like total moves gross financial
        # wealth with it.
        capped[total] = capped.loc[:, list(components)].sum(axis=1)
        maximum = float(pd.to_numeric(donor[total], errors="coerce").max())
        excess = capped[total] - maximum
        fired[f"{total}_capped_rows"] = int((excess > 0).sum())
        if fired[f"{total}_capped_rows"]:
            reduce(total, excess)
    net = pd.to_numeric(donor["net_financial_wealth"], errors="coerce")
    capped["net_financial_wealth"] = derived_net_financial_wealth(capped)
    above = capped["net_financial_wealth"] - float(net.max())
    fired["net_financial_wealth_capped_high_rows"] = int((above > 0).sum())
    if fired["net_financial_wealth_capped_high_rows"]:
        reduce("gross_financial_wealth", above)
        capped["net_financial_wealth"] = derived_net_financial_wealth(capped)
    below = float(net.min()) - capped["net_financial_wealth"]
    fired["net_financial_wealth_capped_low_rows"] = int((below > 0).sum())
    if fired["net_financial_wealth_capped_low_rows"]:
        remaining = below.clip(lower=0.0)
        for liability in UK_WAS_NET_FINANCIAL_LIABILITIES:
            take = np.minimum(remaining, capped[liability].clip(lower=0.0))
            capped[liability] = capped[liability] - take
            remaining = remaining - take
        capped["net_financial_wealth"] = derived_net_financial_wealth(capped)
    return capped, fired


def wealth_identity_violations(table: pd.DataFrame) -> dict[str, int]:
    """Rows on which a total departs from its components, by identity."""

    tolerance = UK_WAS_IDENTITY_TOLERANCE_GBP
    violations: dict[str, int] = {}
    for total, components in UK_WAS_DERIVED_TOTALS.items():
        available = [column for column in components if column in table.columns]
        if len(available) == len(components):
            gap = table[total] - table.loc[:, available].sum(axis=1)
            violations[f"{total}_violation_rows"] = int((gap.abs() > tolerance).sum())
        else:
            # The emitted frame carries no remainder column: the total must
            # still cover the components it does carry.
            gap = table[total] - table.loc[:, available].sum(axis=1)
            violations[f"{total}_below_components_rows"] = int((gap < -tolerance).sum())
    gap = table["net_financial_wealth"] - derived_net_financial_wealth(table)
    violations["net_financial_wealth_violation_rows"] = int(
        (gap.abs() > tolerance).sum()
    )
    return violations


def encode_qrf_predictor_pair(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    predictors: Sequence[str] = UK_WAS_WEALTH_PREDICTORS,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """One-hot the region predictor jointly across donor and recipient.

    Mirrors the SPI stage's paired dummy encoding and the incumbent's
    dummy-encoded region. Donor rows with an unmapped region code (the
    incumbent's absent GOR code 3) become all-zero dummy rows. The tenure
    category is a label for the strata, not a number, and is left out of the
    encoded tables; its one-hot flags enter as 0/1 floats.
    """

    numeric_predictors = tuple(
        predictor for predictor in predictors if predictor != "region"
    )
    combined_region = pd.concat(
        [
            donor["region"].reset_index(drop=True),
            recipient["region"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(combined_region, prefix="region", dtype=float)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)

    def _encode(table: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
        encoded = table.drop(
            columns=["region", UK_WAS_TENURE_CATEGORY_COLUMN], errors="ignore"
        ).copy()
        for column in ("is_renting", *UK_WAS_TENURE_PREDICTORS):
            if column in encoded.columns:
                encoded[column] = encoded[column].astype(bool).astype(float)
        for column in encoded.columns:
            if column == "weight":
                continue
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(
                0.0
            )
        block = block.copy()
        block.index = encoded.index
        return pd.concat([encoded, block], axis=1)

    donor_encoded = _encode(donor, dummies.iloc[: len(donor)])
    recipient_encoded = _encode(recipient, dummies.iloc[len(donor) :])
    return (
        donor_encoded,
        recipient_encoded,
        (*numeric_predictors, *tuple(dummies.columns)),
    )


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[mask].sum() / total) if total > 0.0 else 0.0


#: Housing columns whose share of positive values the receipt reports by
#: tenure on both sides.
UK_WAS_TENURE_RECEIPT_COLUMNS = (
    "main_residence_value",
    "main_residence_mortgage",
    UK_WAS_OTHER_MORTGAGE_COLUMN,
    "mortgage_debt",
    "property_wealth",
    "other_residential_property_value",
)


def tenure_coherence_receipt(
    *,
    donor: pd.DataFrame,
    recipient_category: pd.Series,
    recipient_draws: pd.DataFrame,
    recipient_weights: Sequence[float],
) -> dict[str, object]:
    """Tenure coherence of the housing columns, donor against recipient.

    The first two counts are the structural zeros the stage-health gate
    requires: no main-residence mortgage off a mortgaged tenure and no
    main-residence value off an owner tenure. The rest compares the recipient
    frame with the donor at each side's own weights, and records the mortgage
    mass held off the mortgaged tenure on each side (the mortgages on other
    property, which the chain carries on every tenure since the review of
    #1089; on the donor, TotMortR8 and OthMortR8_sum are read for that record
    only).
    """

    donor_category = donor[UK_WAS_TENURE_CATEGORY_COLUMN].astype(str).to_numpy()
    donor_weights = pd.to_numeric(donor["weight"], errors="coerce").to_numpy(float)
    category = np.asarray(recipient_category.astype(str))
    weights = np.asarray(recipient_weights, dtype=float)
    owner = np.isin(category, UK_OWNER_TENURE_CATEGORIES)
    mortgaged = category == UK_TENURE_OWNED_WITH_MORTGAGE
    donor_owner = np.isin(donor_category, UK_OWNER_TENURE_CATEGORIES)
    donor_mortgaged = donor_category == UK_TENURE_OWNED_WITH_MORTGAGE

    def values(table: pd.DataFrame, column: str) -> np.ndarray:
        return pd.to_numeric(table[column], errors="coerce").fillna(0.0).to_numpy(float)

    main = values(recipient_draws, "main_residence_value")
    debt = values(recipient_draws, "main_residence_mortgage")
    total_debt = values(recipient_draws, "mortgage_debt")
    donor_main = values(donor, "main_residence_value")
    donor_debt = values(donor, "main_residence_mortgage")
    donor_total_debt = values(donor, "mortgage_debt")
    donor_debt_mass = float((donor_total_debt * donor_weights).sum())
    donor_off_tenure = float((donor_total_debt * donor_weights)[~donor_mortgaged].sum())
    off_tenure = float((total_debt * weights)[~mortgaged].sum())
    positive_share: dict[str, dict[str, dict[str, float]]] = {}
    for column in UK_WAS_TENURE_RECEIPT_COLUMNS:
        recipient_positive = values(recipient_draws, column) > 0.0
        donor_positive = values(donor, column) > 0.0
        positive_share[column] = {
            tenure: {
                "donor": _weighted_share(
                    donor_positive[donor_category == tenure],
                    donor_weights[donor_category == tenure],
                ),
                "recipient": _weighted_share(
                    recipient_positive[category == tenure],
                    weights[category == tenure],
                ),
            }
            for tenure in UK_TENURE_CATEGORIES
        }
    return {
        "main_residence_mortgage_off_mortgaged_tenure_rows": int(
            ((debt > 0.0) & ~mortgaged).sum()
        ),
        # Carried, not a defect: the mortgages on other property that
        # households off the mortgaged tenure hold.
        "mortgage_debt_rows_off_mortgaged_tenure": int(
            ((total_debt > 0.0) & ~mortgaged).sum()
        ),
        "mortgage_debt_mass_off_mortgaged_tenure": off_tenure,
        "main_residence_value_off_owner_tenure_rows": int(
            ((main > 0.0) & ~owner).sum()
        ),
        "owner_rows": int(owner.sum()),
        "owner_rows_without_main_residence_value": int(((main <= 0.0) & owner).sum()),
        "owner_share_without_main_residence_value": _weighted_share(
            main[owner] <= 0.0, weights[owner]
        ),
        "donor_owner_share_without_main_residence_value": _weighted_share(
            donor_main[donor_owner] <= 0.0, donor_weights[donor_owner]
        ),
        "mortgaged_share_without_main_residence_mortgage": _weighted_share(
            debt[mortgaged] <= 0.0, weights[mortgaged]
        ),
        "donor_mortgaged_share_without_main_residence_mortgage": _weighted_share(
            donor_debt[donor_mortgaged] <= 0.0, donor_weights[donor_mortgaged]
        ),
        # The main-residence mortgage against the main-residence value, read
        # before the regional property uprating, which moves the value and
        # leaves the debt alone.
        "mortgaged_share_debt_above_main_residence_value": _weighted_share(
            debt[mortgaged] > main[mortgaged], weights[mortgaged]
        ),
        "donor_mortgaged_share_debt_above_main_residence_value": _weighted_share(
            donor_debt[donor_mortgaged] > donor_main[donor_mortgaged],
            donor_weights[donor_mortgaged],
        ),
        "donor_mortgage_debt_mass": donor_debt_mass,
        "donor_mortgage_debt_mass_off_mortgaged_tenure": donor_off_tenure,
        "donor_mortgage_debt_share_off_mortgaged_tenure": (
            donor_off_tenure / donor_debt_mass if donor_debt_mass > 0.0 else 0.0
        ),
        "donor_other_property_mortgage_mass_off_mortgaged_tenure": float(
            (values(donor, "other_property_mortgage").clip(min=0.0) * donor_weights)[
                ~donor_mortgaged
            ].sum()
        ),
        "donor_main_residence_mortgage_mass_off_mortgaged_tenure": float(
            (values(donor, "main_residence_mortgage").clip(min=0.0) * donor_weights)[
                ~donor_mortgaged
            ].sum()
        ),
        "donor_unclassified_tenure_rows": int(
            donor["tenure_code_unclassified"].astype(bool).sum()
        ),
        "donor_rows_by_tenure": {
            tenure: int((donor_category == tenure).sum())
            for tenure in UK_TENURE_CATEGORIES
        },
        "recipient_rows_by_tenure": {
            tenure: int((category == tenure).sum()) for tenure in UK_TENURE_CATEGORIES
        },
        "positive_share_by_tenure": positive_share,
    }


def support_clip_to_donor(
    draws: pd.DataFrame, donor: pd.DataFrame
) -> UKSupportClipResult:
    """Clip output draws to donor-realized support."""

    return support_clip_to_donor_with_receipt(
        draws,
        donor,
        columns=UK_WAS_WEALTH_OUTPUT_COLUMNS,
        stage=UK_WAS_WEALTH_STAGE_NAME,
    )


def allocate_student_loan_balance_to_people(
    *,
    household_balances: pd.Series,
    household_ids: Sequence[object],
    person: pd.DataFrame,
) -> np.ndarray:
    """Allocate household student-loan balances to plausible holders by id."""

    balances_by_household = pd.Series(
        np.asarray(household_balances, dtype=float),
        index=pd.Index(household_ids),
    )
    allocated = np.zeros(len(person), dtype=float)
    if len(person) == 0:
        return allocated
    group_indices = person.groupby("person_household_id", sort=False).indices
    age = _numeric_person(person, "age", 0.0)
    repayments = _numeric_person(person, "student_loan_repayments", 0.0)
    student_loans = _numeric_person(person, "student_loans", 0.0)
    highest_education = _string_person(person, "highest_education", "UPPER_SECONDARY")
    current_education = _string_person(person, "current_education", "NOT_IN_EDUCATION")
    for household_id, household_balance in balances_by_household.items():
        if household_balance <= 0 or household_id not in group_indices:
            continue
        idx = np.asarray(group_indices[household_id], dtype=int)
        tier_masks = (
            repayments[idx] > 0,
            student_loans[idx] > 0,
            highest_education[idx] == "TERTIARY",
            current_education[idx] == "TERTIARY",
            (age[idx] >= 18) & (age[idx] <= 55),
            np.ones(len(idx), dtype=bool),
        )
        selected_mask = next(mask for mask in tier_masks if mask.any())
        selected = idx[selected_mask]
        if tier_masks[0].any() and repayments[idx][tier_masks[0]].sum() > 0:
            repayers = idx[tier_masks[0]]
            weights = repayments[repayers]
            allocated[repayers] += household_balance * weights / weights.sum()
        else:
            allocated[selected] += household_balance / len(selected)
    return allocated


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return exact donor min/max ranges for synthetic receipts and tests."""

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_WAS_WEALTH_OUTPUT_COLUMNS:
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def _numeric_person(person: pd.DataFrame, column: str, default: float) -> np.ndarray:
    values = (
        person[column] if column in person else pd.Series(default, index=person.index)
    )
    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=float)


def _string_person(person: pd.DataFrame, column: str, default: str) -> np.ndarray:
    values = (
        person[column] if column in person else pd.Series(default, index=person.index)
    )
    return values.fillna(default).map(_enum_name).astype(str).to_numpy()


def _donor_artifact(stage: SourceStageSpec) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == "was_qrf_donor":
            return artifact
    raise ValueError(
        "was_wealth stage declares no was_qrf_donor artifact; refusing to read "
        "an unpinned WAS tab."
    )


def _assert_chain_declaration(stage: SourceStageSpec) -> None:
    """The chain op must declare the strata and the derived totals the run uses."""

    operation = next(
        op for op in stage.operations if op.kind == "fit_weighted_qrf_chain"
    )
    declared_strata = {
        str(target): tuple(categories)
        for target, categories in dict(
            operation.parameters.get("stratified_targets", {})
        ).items()
    }
    if declared_strata != dict(UK_WAS_STRATIFIED_TARGETS):
        raise ValueError(
            "was_wealth stratified_targets drifted: manifest declares "
            f"{declared_strata!r}, runtime uses {dict(UK_WAS_STRATIFIED_TARGETS)!r}."
        )
    declared_totals = {
        str(total): tuple(components)
        for total, components in dict(
            operation.parameters.get("derived_totals", {})
        ).items()
    }
    runtime_totals = {
        **dict(UK_WAS_DERIVED_TOTALS),
        "net_financial_wealth": (
            "gross_financial_wealth",
            *(f"-{column}" for column in UK_WAS_NET_FINANCIAL_LIABILITIES),
        ),
    }
    if declared_totals != runtime_totals:
        raise ValueError(
            "was_wealth derived_totals drifted: manifest declares "
            f"{declared_totals!r}, runtime uses {runtime_totals!r}."
        )


def _qrf_n_estimators(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf_chain":
            value = operation.parameters.get("n_estimators", 100)
            if isinstance(value, int) and value > 0:
                return value
    return 100


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("WAS wealth stage requires a caller-supplied WAS tab path.")
    return Path(path).expanduser().resolve()


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
