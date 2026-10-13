"""Raw FRS tab ingest for the UK spine Frame."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.trace import sha256_file
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.uc_relationships import frs_uc_claimant_mask
from microcosm.frame import Frame, WeightKind

__all__ = [
    "FRS_SPINE_TABLES",
    "OUTPUT_COLUMNS",
    "REGION_MAP",
    "TIME_PERIOD",
    "UC_CAPITAL_UNAVAILABLE",
    "WEEKS_IN_YEAR",
    "UKFRSSpineStageTransform",
    "artifact_by_table",
    "build_uk_frs_spine_frame",
    "map_codes",
    "normalize_ids",
    "number",
    "positive",
    "raw_number",
    "read_pinned_tab",
    "reject_nan",
    "sum_to_entity",
    "uk_frs_spine_seed_frame",
]

FRS_SPINE_TABLES = (
    "accounts",
    "adult",
    "benefits",
    "benunit",
    "child",
    "chldcare",
    "extchild",
    "househol",
    "job",
    "maint",
    "mortgage",
    "oddjob",
    "penprov",
    "pension",
)

WEEKS_IN_YEAR = 365.25 / 7
TIME_PERIOD = "2024"

# Raw TOTCAPB4 blanks, nonnumeric values, and negative codes mean that the
# benefit-unit capital observation is unavailable. They map to this engine
# sentinel; observed zero remains a valid capital value.
UC_CAPITAL_UNAVAILABLE = -1.0

# FRS GVTREGNO uses skip-3 coding: code 3 (the retired Merseyside code) is
# absent from the domain, so the real codes are [1, 2, 4..13] with
# 12 = Scotland and 13 = Northern Ireland — matching the incumbent's
# ``[1, 2] + range(4, 15)`` categorical and its ``SCOTLAND_GVTREGNO = 12``
# water-charge branch below. Verified on the 2023-24 tabs: zero code-3
# households, 1,844 code-13 (Northern Ireland) households. A contiguous
# 1-12 map shifts every label from Yorkshire up by one region and drops
# Northern Ireland to UNKNOWN (#692 review). The SPI tape's GORCODE is a
# different coding with its own history — do not import lessons across.
REGION_MAP = {
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
    13: "NORTHERN_IRELAND",
}

TENURE_MAP = {
    1: "RENT_FROM_COUNCIL",
    2: "RENT_FROM_HA",
    3: "RENT_PRIVATELY",
    4: "RENT_PRIVATELY",
    5: "OWNED_OUTRIGHT",
    6: "OWNED_WITH_MORTGAGE",
}

ACCOMMODATION_MAP = {
    1: "HOUSE_DETACHED",
    2: "HOUSE_SEMI_DETACHED",
    3: "HOUSE_TERRACED",
    4: "FLAT",
    5: "CONVERTED_HOUSE",
    6: "MOBILE",
    7: "OTHER",
}

COUNCIL_TAX_BAND_MAP = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
}

MARITAL_MAP = {
    1: "MARRIED",
    2: "SINGLE",
    3: "SINGLE",
    4: "WIDOWED",
    5: "SEPARATED",
    6: "DIVORCED",
}

# FRS BENEFITS codes summed into each reported amount. Scotland's Adult
# Disability Payment (117 daily living, 118 mobility) mirrors PIP's components
# and rates, and Child Disability Payment (121 care, 122 mobility) mirrors DLA's,
# so each lands in the column of the benefit it replaces (FRS 2024-25 variable
# listing, UKDS SN 9563; uk-data#500, microcosm#1095). Codes 69 and 70 are
# benefit recoveries, not income, and stay unmapped.
BENEFIT_CODES = {
    "child_benefit": (3,),
    "income_support": (19,),
    "housing_benefit": (94,),
    "attendance_allowance": (12,),
    "dla_sc": (1, 121),
    "dla_m": (2, 122),
    "iidb": (15,),
    "carers_allowance": (13,),
    "sda": (10,),
    "afcs": (8,),
    "ssmg": (22,),
    "pension_credit": (4,),
    "child_tax_credit": (91,),
    "working_tax_credit": (90,),
    "state_pension": (5,),
    "winter_fuel_allowance": (62,),
    "incapacity_benefit": (17,),
    "universal_credit": (95,),
    "pip_m": (97, 118),
    "pip_dl": (96, 117),
}

# FRS adult HOURTOT: total hours of care provided per week, a banded derived
# variable (FRS derived-variable specification). Each code maps to the lower
# edge of its band in weekly hours; the engine's care_hours input is weekly
# hours, and its only consumer today tests the 35-hour Carer's Allowance line,
# which the codes 5, 6, 7 and 10 (35-49, 50-99, 100 or more, varies at 35 or
# more) reach and the codes 1-4, 8 and 9 do not (#882).
FRS_CARE_HOURS_BY_BAND = {
    0: 0.0,  # no care provided
    1: 0.0,  # 0-4 hours
    2: 5.0,  # 5-9 hours
    3: 10.0,  # 10-19 hours
    4: 20.0,  # 20-34 hours
    5: 35.0,  # 35-49 hours
    6: 50.0,  # 50-99 hours
    7: 100.0,  # 100 hours or more
    8: 0.0,  # varies, under 20 hours
    9: 20.0,  # varies, 20-34 hours
    10: 35.0,  # varies, 35 hours or more
}

# FRS adult RENTPROF (question RentProf): whether the rent from other property
# in ROYYR1 is a profit (1) or a loss (2). ROYYR1 itself is always positive.
FRS_RENTPROF_LOSS = 2

OUTPUT_COLUMNS = (
    "person_id",
    "person_benunit_id",
    "person_household_id",
    "age",
    "gender",
    "marital_status",
    "hours_worked",
    "care_hours",
    "is_household_head",
    "is_benunit_head",
    "is_parent",
    "is_blind",
    "is_uc_claimant",
    "is_claimant_or_partner",
    "is_hbai_dependent_child",
    "uc_is_in_startup_period",
    "uc_is_in_gainful_self_employment",
    "rent_paid_as_boarder",
    "rent_paid_as_lodger",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "tax_free_savings_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "reports_rent_from_other_property",
    "maintenance_income",
    "miscellaneous_income",
    "private_transfer_income",
    "lump_sum_income",
    "student_loan_repayments",
    "statutory_sick_pay",
    "statutory_maternity_pay",
    "student_loans",
    "access_fund",
    "education_grants",
    "healthy_start_vouchers",
    "free_school_breakfasts",
    "free_school_fruit_veg",
    "free_school_meals",
    "council_tax_benefit_reported",
    "maintenance_expenses",
    "childcare_expenses",
    "personal_pension_contributions",
    "employee_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "salary_sacrifice_reported",
    "salary_sacrifice_asked",
    "child_benefit_reported",
    "income_support_reported",
    "housing_benefit_reported",
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "iidb_reported",
    "carers_allowance_reported",
    "would_claim_carers_allowance",
    "sda_reported",
    "afcs_reported",
    "ssmg_reported",
    "pension_credit_reported",
    "child_tax_credit_reported",
    "working_tax_credit_reported",
    "state_pension_reported",
    "winter_fuel_allowance_reported",
    "incapacity_benefit_reported",
    "universal_credit_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
    "bsp_reported",
    "benunit_id",
    "frs_benunit_capital",
    "is_married",
    "dependent_children",
    "liable_for_share_of_household_rent",
    "household_id",
    "region",
    "tenure_type",
    "accommodation_type",
    "num_bedrooms",
    "council_tax_reported",
    "council_tax_band",
    "council_tax_rebate",
    "council_tax_single_adult_raw",
    "water_and_sewerage_charges",
    "domestic_rates",
    "rent",
    "subrent",
    "mortgage_interest_repayment",
    "mortgage_capital_repayment",
    "structural_insurance_payments",
    "housing_service_charges",
    "external_child_payments",
)


class UKFRSSpineStageTransform:
    """Whole-stage callable for the root FRS spine ingest."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        self._sentinel_mapped_rows: int | None = None
        self._access_fund: dict[str, object] | None = None

    def __call__(self, frame: Frame) -> Frame:
        evidence: dict[str, object] = {}
        result = build_uk_frs_spine_frame(
            self.raw_dir, stage=self.stage, evidence=evidence
        )
        capital = result.table("benunit")["frs_benunit_capital"]
        self._sentinel_mapped_rows = int((capital == UC_CAPITAL_UNAVAILABLE).sum())
        self._access_fund = dict(evidence["access_fund"])
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        """Report how loudly the capital availability rule and the
        access-fund repair fired."""

        if self._sentinel_mapped_rows is None or self._access_fund is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": "frs_spine",
                "frs_benunit_capital": {
                    "unavailable_sentinel": UC_CAPITAL_UNAVAILABLE,
                    "mapped_rows": self._sentinel_mapped_rows,
                },
                "access_fund": dict(self._access_fund),
            }
        }


def uk_frs_spine_seed_frame() -> Frame:
    """A minimal valid Frame for the root stage; the stage ignores its rows."""

    person = pd.DataFrame(
        {
            "person_id": [1],
            "person_benunit_id": [1],
            "person_household_id": [1],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1]})
    household = pd.DataFrame({"household_id": [1], "household_weight": [1.0]})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=TIME_PERIOD,
        weight_kind=WeightKind.DESIGN,
    )


def build_uk_frs_spine_frame(
    raw_dir: str | Path,
    *,
    stage: SourceStageSpec,
    evidence: dict[str, object] | None = None,
) -> Frame:
    """Build the direct raw FRS spine Frame from pinned local tab files.

    ``evidence``, when given, receives the access-fund repair's record under
    ``"access_fund"`` (see :func:`access_fund_annual`).
    """

    raw_root = Path(raw_dir)
    artifacts = _artifact_by_table(stage)
    tables = {
        table: _read_pinned_tab(
            raw_root / str(artifacts[table]["locator"]), artifacts[table]
        )
        for table in FRS_SPINE_TABLES
    }
    normalized = {name: _normalize_ids(table) for name, table in tables.items()}
    frame = _assemble_frame(normalized, evidence=evidence)
    validate_uk_national_frame(frame)
    return frame


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table: dict[str, Mapping[str, Any]] = {}
    for artifact in stage.artifacts:
        table = artifact.get("table")
        if table in by_table:
            raise ValueError(f"Duplicate FRS spine artifact for table {table!r}.")
        if isinstance(table, str):
            by_table[table] = artifact
    missing = sorted(set(FRS_SPINE_TABLES) - set(by_table))
    if missing:
        raise ValueError(f"FRS spine manifest is missing tab artifact(s): {missing}.")
    unknown = sorted(set(by_table) - set(FRS_SPINE_TABLES))
    if unknown:
        raise ValueError(f"FRS spine manifest declares unknown tab(s): {unknown}.")
    return by_table


def _read_pinned_tab(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    columns: Collection[str] | None = None,
) -> pd.DataFrame:
    """Read a size- and sha256-pinned tab with lower-cased column names.

    ``columns`` (matched case-insensitively) reads only those columns, for wide
    tabs where a full read costs gigabytes (the WAS round-8 person tab has
    4,079 columns); every one must be present, and the pins are checked on the
    whole file first either way.
    """

    if not path.exists():
        raise FileNotFoundError(f"FRS spine tab is missing: {path}.")
    expected_size = int(artifact["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"FRS spine tab {path.name} is {actual_size} bytes, "
            f"not the pinned {expected_size}."
        )
    expected_sha = str(artifact["sha256"])
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"FRS spine tab {path.name} hashes to {actual_sha}, "
            f"not the pinned {expected_sha}."
        )
    if columns is None:
        raw = pd.read_csv(path, sep="\t")
    else:
        wanted = {str(column).lower() for column in columns}
        raw = pd.read_csv(
            path, sep="\t", usecols=lambda column: str(column).lower() in wanted
        )
    raw.columns = raw.columns.str.lower()
    if columns is not None:
        if raw.columns.duplicated().any():
            duplicates = sorted(set(raw.columns[raw.columns.duplicated()]))
            raise ValueError(
                f"{path.name} has duplicate column(s) after lower-casing: {duplicates}."
            )
        missing = sorted(wanted - set(raw.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required column(s): {missing}.")
    converted = raw.apply(pd.to_numeric, errors="coerce")
    if path.stem == "job" and "salsac" in raw.columns:
        converted["salsac_raw"] = raw["salsac"].astype(str)
    # UCSTART is a month/day/year date, which numeric conversion would blank.
    if path.stem == "benefits" and "ucstart" in raw.columns:
        converted["ucstart_raw"] = raw["ucstart"].astype("string")
    return converted


def _normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    if "sernum" in frame.columns:
        frame = frame.rename(columns={"sernum": "household_id"})
    if "benunit" in frame.columns:
        frame = frame.rename(columns={"benunit": "benunit_id"})
        frame["benunit_id"] = (
            _number(frame, "household_id") * 100 + _number(frame, "benunit_id")
        ).astype("int64")
    if "person" in frame.columns:
        frame = frame.rename(columns={"person": "person_id"})
        frame["person_id"] = (
            _number(frame, "household_id") * 1000 + _number(frame, "person_id")
        ).astype("int64")
    return frame


def _assemble_frame(
    frs: Mapping[str, pd.DataFrame], *, evidence: dict[str, object] | None = None
) -> Frame:
    person = (
        pd.concat([frs["adult"], frs["child"]], ignore_index=True, sort=False)
        .fillna(0)
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
    adult_ids = set(frs["adult"]["person_id"].astype("int64"))
    benunit_raw = frs["benunit"].sort_values("benunit_id").reset_index(drop=True)
    household_raw = frs["househol"].set_index("household_id").sort_index()
    household_ids = np.sort(person["household_id"].unique())
    household = household_raw.loc[household_ids]

    pe_person = pd.DataFrame(
        {
            "person_id": person["person_id"].astype("int64"),
            "person_benunit_id": person["benunit_id"].astype("int64"),
            "person_household_id": person["household_id"].astype("int64"),
        }
    )
    pe_benunit = pd.DataFrame({"benunit_id": benunit_raw["benunit_id"].astype("int64")})
    pe_benunit["frs_benunit_capital"] = _frs_benunit_capital(benunit_raw)
    pe_household = pd.DataFrame(
        {"household_id": household_ids.astype("int64")},
        index=household.index,
    )
    pe_household["household_weight"] = _raw_number(household, "gross4").to_numpy()

    age = _number(person, "age80") + _number(person, "age")
    # The graph declares person.age int64 from the root (#845); the raw
    # AGE80/AGE columns are integer codes, and blanks coerce to 0 above, so
    # a non-integral value here is a vintage defect rather than data.
    if not np.array_equal(age.to_numpy(), np.floor(age.to_numpy())):
        raise ValueError("FRS age80/age columns must be integral to build person.age.")
    pe_person["age"] = age.astype("int64")
    pe_person["gender"] = np.where(_number(person, "sex") == 1, "MALE", "FEMALE")
    pe_person["marital_status"] = _map_codes(person, "marital", MARITAL_MAP, "SINGLE")
    pe_person["hours_worked"] = _positive(person, "tothours") * WEEKS_IN_YEAR
    pe_person["care_hours"] = frs_care_hours(person)
    pe_person["is_household_head"] = _number(person, "hrpid") == 1
    pe_person["is_benunit_head"] = _number(person, "uperson") == 1
    dependent_children = _dependent_children(benunit_raw, frs["child"])
    dependent_by_benunit = pd.Series(
        dependent_children.to_numpy(), index=benunit_raw["benunit_id"]
    )
    pe_person["is_parent"] = pe_person["person_id"].isin(adult_ids) & pe_person[
        "person_benunit_id"
    ].map(dependent_by_benunit).fillna(0).gt(0)
    # Registered blind or severely sight impaired with the local authority
    # (SPCREG1, asked on the adult and child tabs alike). Registration follows
    # the consultant ophthalmologist's certificate the engine's is_blind
    # names; partial-sight registration (SPCREG2) does not meet that test
    # (uk-data#523).
    pe_person["is_blind"] = _number(person, "spcreg1") == FRS_REGISTERED_YES

    pe_person["employment_income"] = _positive(person, "inearns") * WEEKS_IN_YEAR
    pe_person["self_employment_income"] = _positive(person, "seincam2") * WEEKS_IN_YEAR
    _add_private_pension(pe_person, person, frs["pension"])
    _add_accounts(pe_person, person, frs["accounts"])
    _add_person_income(pe_person, person, household, frs["oddjob"], evidence=evidence)
    _add_benefits(pe_person, person, frs["benefits"])
    # The engine pays Carer's Allowance on hours or receipt; a dataset keeps it
    # on reported receipt so that care_hours qualifies carers for the UC carer
    # element without paying the allowance to every carer (#882).
    pe_person["would_claim_carers_allowance"] = (
        pe_person["carers_allowance_reported"] > 0
    )
    _add_person_expenses(pe_person, person, frs)

    pe_benunit["is_married"] = _number(benunit_raw, "famtypb2").isin([5, 7])
    pe_benunit["dependent_children"] = dependent_children
    # Preserve the FRS claimant/partner roles as a country-model input. Age
    # does not promote a dependent child to a partner, and legal marriage
    # alone does not establish that a partner lives in this benefit unit.
    pe_person["is_uc_claimant"] = frs_uc_claimant_mask(pe_person, pe_benunit)
    pe_benunit["liable_for_share_of_household_rent"] = (
        frs_liable_for_share_of_household_rent(
            benunit_raw, person, household, frs["benefits"]
        )
    )
    boarder_rent, lodger_rent = frs_rent_paid_to_householder(person)
    pe_person["rent_paid_as_boarder"] = boarder_rent
    pe_person["rent_paid_as_lodger"] = lodger_rent
    # policyengine-uk's person types (pe-uk#1896). Every FRS person is on the
    # adult table or the child table: the adult table holds each benefit
    # unit's head and any partner (uk-data#524), the child table its HBAI
    # dependent children (uk-data#486). Supplying both stops the engine
    # inferring them from ages.
    # The claimant mask above already refuses a benefit unit without one or
    # two adult records, so membership is a well-formed role.
    adult_record = pe_person["person_id"].isin(adult_ids).to_numpy()
    pe_person["is_claimant_or_partner"] = adult_record
    pe_person["is_hbai_dependent_child"] = ~adult_record
    pe_person["uc_is_in_startup_period"] = frs_uc_start_up_period(
        person,
        pe_person,
        pe_benunit,
        household,
        job=frs["job"],
        benefits=frs["benefits"],
    )
    pe_person["uc_is_in_gainful_self_employment"] = uc_gainful_self_employment(
        _number(person, "empstati").isin(FRS_SELF_EMPLOYED_EMPSTATI).to_numpy(),
        pe_person["self_employment_income"],
        pe_person["employment_income"],
    )

    _add_household_columns(pe_household, household, frs)

    _reject_nan(pe_person, "person")
    _reject_nan(pe_benunit, "benunit")
    _reject_nan(pe_household, "household")
    return uk_national_frame(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        time_period=TIME_PERIOD,
        weight_kind=WeightKind.DESIGN,
    )


def _add_private_pension(
    pe_person: pd.DataFrame, person: pd.DataFrame, pension: pd.DataFrame
) -> None:
    pension_payment = _sum_to_entity(
        _number(pension, "penpay") * (_number(pension, "penpay") > 0),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pension_tax_paid = _sum_to_entity(
        _number(pension, "ptamt")
        * ((_number(pension, "ptinc") == 2) & (_number(pension, "ptamt") > 0)),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pension_deductions_removed = _sum_to_entity(
        _number(pension, "poamt")
        * (
            ((_number(pension, "poinc") == 2) | (_number(pension, "penoth") == 1))
            & (_number(pension, "poamt") > 0)
        ),
        pension.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )
    pe_person["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * WEEKS_IN_YEAR


def _add_accounts(
    pe_person: pd.DataFrame, person: pd.DataFrame, accounts: pd.DataFrame
) -> None:
    account = _number(accounts, "account")
    accint = _number(accounts, "accint")
    acctax = _number(accounts, "acctax")
    invtax = _number(accounts, "invtax")
    person_id = accounts.get("person_id", pd.Series(dtype="float64"))
    inverted_basic_rate = 1.25
    tax_free = (
        _sum_to_entity(accint * (account == 21), person_id, person["person_id"])
        * WEEKS_IN_YEAR
    )
    taxable = (
        _sum_to_entity(
            (accint * np.where(acctax == 1, inverted_basic_rate, 1))
            * account.isin((1, 3, 5, 27, 28)),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    dividends = (
        _sum_to_entity(
            (accint * np.where(invtax == 1, inverted_basic_rate, 1))
            * (((account == 6) & (invtax == 1)) | account.isin((7, 8))),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["tax_free_savings_income"] = np.maximum(0, tax_free)
    pe_person["savings_interest_income"] = np.maximum(0, taxable + tax_free)
    pe_person["dividend_income"] = np.maximum(0, dividends)


def _add_person_income(
    pe_person: pd.DataFrame,
    person: pd.DataFrame,
    household: pd.DataFrame,
    oddjob: pd.DataFrame,
    *,
    evidence: dict[str, object] | None = None,
) -> None:
    pe_person["property_income"] = frs_property_income(person, household)
    pe_person[FRS_LANDLORD_CARRIER_COLUMN] = frs_reports_rent_from_other_property(
        person
    )
    maintenance_to_self = np.maximum(
        np.where(
            _number(person, "mntus1") == 2,
            _number(person, "mntusam1"),
            _number(person, "mntamt1"),
        ),
        0,
    )
    pe_person["maintenance_income"] = (
        maintenance_to_self + _positive(person, "mntamt2").to_numpy()
    ) * WEEKS_IN_YEAR
    pe_person["miscellaneous_income"] = (
        _odd_job_income(person, oddjob)
        + _sum_positive_fields(
            person, ("allpay2", "royyr2", "royyr3", "royyr4", "chamtern", "chamttst")
        )
    ) * WEEKS_IN_YEAR
    pe_person["private_transfer_income"] = (
        _sum_positive_fields(
            person, ("apamt", "apdamt", "pareamt", "allpay2", "allpay3", "allpay4")
        )
        * WEEKS_IN_YEAR
    )
    pe_person["lump_sum_income"] = _number(person, "redamt")
    pe_person["student_loan_repayments"] = _number(person, "slrepamt") * WEEKS_IN_YEAR
    pe_person["statutory_sick_pay"] = _number(person, "sspadj") * WEEKS_IN_YEAR
    pe_person["statutory_maternity_pay"] = _number(person, "smpadj") * WEEKS_IN_YEAR
    pe_person["student_loans"] = _positive(person, "tuborr")
    access_fund_evidence: dict[str, object] = {}
    pe_person["access_fund"] = access_fund_annual(person, evidence=access_fund_evidence)
    if evidence is not None:
        evidence["access_fund"] = access_fund_evidence
    pe_person["education_grants"] = np.maximum(
        _number(person, "grtdir1") + _number(person, "grtdir2"), 0
    )
    # In-kind benefits recorded per person on the FRS tapes. Each is a direct
    # weeklyised amount with no derivation: healthy-start vouchers appear on
    # both the adult and child tapes, the three school ones only on the child
    # tape, so absent columns read as zero for adults through `_number`.
    pe_person["healthy_start_vouchers"] = _positive(person, "heartval") * WEEKS_IN_YEAR
    pe_person["free_school_breakfasts"] = _positive(person, "fsbval") * WEEKS_IN_YEAR
    pe_person["free_school_fruit_veg"] = _positive(person, "fsfvval") * WEEKS_IN_YEAR
    pe_person["free_school_meals"] = _positive(person, "fsmval") * WEEKS_IN_YEAR


#: The internal landlord carrier ``frs_reports_rent_from_other_property`` writes.
FRS_LANDLORD_CARRIER_COLUMN = "reports_rent_from_other_property"


def frs_reports_rent_from_other_property(person: pd.DataFrame) -> np.ndarray:
    """Whether the person reports rent from other property, at a profit or a loss.

    ROYYR1 is entered as a positive amount whether the letting made a profit or
    a loss (RENTPROF 2 marks the loss), so a positive ROYYR1 marks a landlord
    either way. ``property_income`` counts a loss as zero
    (``frs_property_income``), so this internal carrier keeps the landlord
    signal the CGT stages read for loss-making landlords too. It is not an
    engine input and does not leave the release.
    """

    return (_positive(person, "royyr1") > 0).to_numpy(dtype=bool)


def frs_property_income(person: pd.DataFrame, household: pd.DataFrame) -> np.ndarray:
    """Annual property income each person reports in the FRS.

    Two FRS amounts, both weekly in the released data:

    - SUBRENT, the rent the household received from sub-letting. The FRS asks
      every household (SubLet), whatever its tenure, and DWP's derived
      SUBLTAMT and INRINC count it for every tenure, so renting and rent-free
      households count too. It goes to the household reference person.
      ``household`` must be indexed by ``household_id``.
    - ROYYR1, the person's rent before tax from other property, in the UK or
      abroad, after paying for the things on show card K6 (question
      PropRent). The card lists mortgage payments and interest on a loan to
      buy the property alongside repairs, rent, rates, insurance and
      services, so ROYYR1 is also net of finance costs and mortgage capital,
      which a landlord cannot deduct for tax. The questionnaire cannot take a
      negative amount, so a loss is entered as a positive amount and
      RENTPROF = 2 (question RentProf) marks it. A loss counts as zero: the
      engine has no property loss input, and a loss is not set against the
      household's SUBRENT. That matches the general rule for the year: an
      individual's UK property loss is carried forward against future profits
      of the same property business (ITA 2007 ss. 118-119). Sideways relief
      against general income exists only for the part of a loss from capital
      allowances or agricultural expenses (ITA 2007 s. 120), which the FRS
      does not identify.

    Because ROYYR1 nets mortgage capital and interest, it sits below the SPI's
    net income from property (after allowable expenses, before residential
    finance costs) for landlords with a mortgage. Binding the SPI amounts on
    this variable (microcosm#1106) therefore leans on reweighting unless the
    other-property mortgage is added back.

    SUBRENT is used as reported. SUBALLOW records whether it is before (1) or
    after (2) allowable expenses, but the FRS records no sub-letting expense
    amount to take off the before-expenses answers.

    Negative values are FRS missing-value codes (-1 to -9), not amounts, so
    each amount is floored at zero before the two are added.

    CVPAY is not included. It is the rent a boarder or lodger pays the
    householder (question CvPay, "How much rent did [name] pay"), recorded on
    the boarder's or lodger's own adult record, so it is not their income.
    Within the household the payment is a transfer, so household totals are
    unaffected, but the householder's receipt is not credited to anyone: the
    householder's benefit unit understates that income.

    SUBRENT and lodger receipts qualify for Rent a Room relief (ITTOIA 2005
    Part 7 Chapter 1: £7,500 a year, £3,750 if shared) when the letting is of
    furnished accommodation in the householder's only or main residence
    (s. 786). Here SUBRENT is taxable
    ``property_income``, so the engine applies only the £1,000 property
    allowance and overstates tax on it, and lodger receipts are not counted at
    all (above). Routing SUBRENT to policyengine-uk's ``sublet_income`` and the
    lodgers' payments to the householder needs the engine to apply the relief
    first.
    """

    is_head = (_number(person, "hrpid") == 1).to_numpy(dtype=float)
    subrent = pd.Series(
        _positive(household, "subrent").to_numpy(), index=household.index
    )
    persons_household_subrent = (
        person["household_id"].map(subrent).fillna(0).to_numpy(dtype="float64")
    )
    rent_from_other_property = (
        _positive(person, "royyr1")
        .where(_number(person, "rentprof") != FRS_RENTPROF_LOSS, 0)
        .to_numpy(dtype="float64")
    )
    return (
        is_head * persons_household_subrent + rent_from_other_property
    ) * WEEKS_IN_YEAR


def _odd_job_income(person: pd.DataFrame, oddjob: pd.DataFrame) -> np.ndarray:
    return _sum_to_entity(
        _number(oddjob, "ojamt") * (_number(oddjob, "ojnow") == 1),
        oddjob.get("person_id", pd.Series(dtype="float64")),
        person["person_id"],
    )


def _add_benefits(
    pe_person: pd.DataFrame, person: pd.DataFrame, benefits: pd.DataFrame
) -> None:
    benefit = _number(benefits, "benefit")
    var2 = _number(benefits, "var2")
    amount = _number(benefits, "benamt")
    person_id = benefits.get("person_id", pd.Series(dtype="float64"))
    for name, codes in BENEFIT_CODES.items():
        pe_person[f"{name}_reported"] = (
            _sum_to_entity(amount * benefit.isin(codes), person_id, person["person_id"])
            * WEEKS_IN_YEAR
        )
    pe_person["jsa_contrib_reported"] = (
        _sum_to_entity(
            amount * var2.isin((1, 3)) * (benefit == 14),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["jsa_income_reported"] = (
        _sum_to_entity(
            amount * var2.isin((2, 4)) * (benefit == 14),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_contrib_reported"] = (
        _sum_to_entity(
            amount * var2.isin((1, 3)) * (benefit == 16),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_income_reported"] = (
        _sum_to_entity(
            amount * var2.isin((2, 4)) * (benefit == 16),
            person_id,
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    pe_person["bsp_reported"] = (
        _sum_to_entity(amount * benefit.isin((6, 9)), person_id, person["person_id"])
        * WEEKS_IN_YEAR
    )
    pe_person["winter_fuel_allowance_reported"] = (
        pe_person["winter_fuel_allowance_reported"] / WEEKS_IN_YEAR
    )


def _add_person_expenses(
    pe_person: pd.DataFrame, person: pd.DataFrame, frs: Mapping[str, pd.DataFrame]
) -> None:
    household = frs["househol"].set_index("household_id").sort_index()
    ctrebamt = pd.Series(_number(household, "ctrebamt").values, index=household.index)
    pe_person["council_tax_benefit_reported"] = np.maximum(
        (_number(person, "hrpid") == 1).to_numpy(dtype=float)
        * person["household_id"].map(ctrebamt).fillna(0).to_numpy()
        * WEEKS_IN_YEAR,
        0,
    )
    maintenance = frs["maint"]
    pe_person["maintenance_expenses"] = (
        _sum_to_entity(
            np.where(
                _number(maintenance, "mrus") == 2,
                _number(maintenance, "mruamt"),
                _number(maintenance, "mramt"),
            ),
            maintenance.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    childcare = frs["chldcare"]
    pe_person["childcare_expenses"] = (
        _sum_to_entity(
            _number(childcare, "chamt")
            * (_number(childcare, "cost") == 1)
            * (_number(childcare, "registrd") == 1),
            childcare.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR
    )
    penprov = frs["penprov"]
    pension_amount = _number(penprov, "penamt")
    # Personal and stakeholder pension contributions as reported, no longer
    # clipped at the 95th percentile of every PENPROV amount: the clip removed
    # 28% of the reported amount (1.8% of rows) against HMRC's relief-at-source
    # total, which SPI Table 3.8 now binds by income band; the engine caps the
    # relief itself at the annual allowance (microcosm#1069 c8).
    pe_person["personal_pension_contributions"] = np.maximum(
        0,
        _sum_to_entity(
            pension_amount[_number(penprov, "stemppen").isin((5, 6))],
            penprov.loc[_number(penprov, "stemppen").isin((5, 6)), "person_id"],
            person["person_id"],
        )
        * WEEKS_IN_YEAR,
    )
    job = frs["job"]
    pe_person["employee_pension_contributions"] = np.maximum(
        0,
        _sum_to_entity(
            _number(job, "deduc1").fillna(0),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR,
    )
    pe_person["pension_contributions_via_salary_sacrifice"] = np.maximum(
        0,
        _sum_to_entity(
            _number(job, "spnamt").fillna(0),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        )
        * WEEKS_IN_YEAR,
    )
    salsac = job.get("salsac_raw", pd.Series(dtype=object)).map(
        {"1": 1, "2": 0, " ": -1, "": -1}
    )
    salsac = salsac.fillna(-1).astype(int)
    pe_person["salary_sacrifice_reported"] = np.clip(
        _sum_to_entity(
            (salsac == 1).astype(int),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        ),
        0,
        1,
    )
    pe_person["salary_sacrifice_asked"] = np.clip(
        _sum_to_entity(
            (salsac >= 0).astype(int),
            job.get("person_id", pd.Series(dtype="float64")),
            person["person_id"],
        ),
        0,
        1,
    )


def _add_household_columns(
    pe_household: pd.DataFrame,
    household: pd.DataFrame,
    frs: Mapping[str, pd.DataFrame],
) -> None:
    pe_household["region"] = _map_codes(household, "gvtregno", REGION_MAP, "UNKNOWN")
    pe_household["tenure_type"] = _map_codes(
        household, "ptentyp2", TENURE_MAP, "RENT_PRIVATELY"
    )
    pe_household["accommodation_type"] = _map_codes(
        household, "typeacc", ACCOMMODATION_MAP, "HOUSE_DETACHED"
    )
    pe_household["num_bedrooms"] = _number(household, "bedroom6")
    pe_household["council_tax_reported"] = _positive(household, "ctannual")
    pe_household["council_tax_band"] = _map_codes(
        household, "ctband", COUNCIL_TAX_BAND_MAP, "D"
    )
    pe_household["council_tax_rebate"] = (
        _positive(household, "ctrebamt") * WEEKS_IN_YEAR
    )
    pe_household["council_tax_single_adult_raw"] = _number(household, "adulth")
    scotland = _number(household, "gvtregno") == 12
    pe_household["water_and_sewerage_charges"] = (
        np.where(
            scotland,
            scottish_water_and_sewerage_weekly(household),
            _number(household, "watsewrt"),
        )
        * WEEKS_IN_YEAR
    )
    pe_household["domestic_rates"] = (
        np.select(
            [
                _number(household, "niratlia") >= 0,
                _number(household, "rt2rebam") >= 0,
                True,
            ],
            [_number(household, "niratlia"), _number(household, "rt2rebam"), 0],
        )
        * WEEKS_IN_YEAR
    ).astype(float)
    pe_household["rent"] = _number(household, "hhrent") * WEEKS_IN_YEAR
    pe_household["subrent"] = _positive(household, "subrent") * WEEKS_IN_YEAR
    pe_household["mortgage_interest_repayment"] = (
        _number(household, "mortint") * WEEKS_IN_YEAR
    )
    mortgage = frs["mortgage"]
    mortgage_capital = np.where(
        _number(mortgage, "rmort") == 1,
        _number(mortgage, "rmamt"),
        _number(mortgage, "borramt"),
    )
    mortgage_end = _number(mortgage, "mortend").replace(0, np.nan)
    pe_household["mortgage_capital_repayment"] = _sum_to_entity(
        pd.Series(mortgage_capital).divide(mortgage_end).fillna(0),
        mortgage.get("household_id", pd.Series(dtype="float64")),
        household.index,
    )
    pe_household["structural_insurance_payments"] = (
        _number(household, "struins") * WEEKS_IN_YEAR
    )
    pe_household["housing_service_charges"] = (
        sum(_positive(household, f"chrgamt{i}") for i in range(1, 10)) * WEEKS_IN_YEAR
    )
    extchild = frs["extchild"]
    pe_household["external_child_payments"] = _sum_to_entity(
        _number(extchild, "nhhamt") * WEEKS_IN_YEAR,
        extchild.get("household_id", pd.Series(dtype="float64")),
        household.index,
    )


def _dependent_children(benunit: pd.DataFrame, child: pd.DataFrame) -> pd.Series:
    if "depchldb" in benunit.columns:
        return _number(benunit, "depchldb")
    return (
        child.groupby("benunit_id")
        .size()
        .reindex(benunit["benunit_id"])
        .fillna(0)
        .astype(float)
        .reset_index(drop=True)
    )


def _sum_to_entity(values: Any, source_ids: Any, target_ids: pd.Series) -> np.ndarray:
    source = pd.Series(np.asarray(source_ids))
    value = pd.Series(np.asarray(values, dtype="float64")).fillna(0)
    if source.empty or value.empty:
        return np.zeros(len(target_ids), dtype="float64")
    totals = value.groupby(source).sum()
    return totals.reindex(target_ids).fillna(0).to_numpy(dtype="float64")


def _sum_positive_fields(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    total = np.zeros(len(frame), dtype="float64")
    for column in columns:
        total += _positive(frame, column).to_numpy()
    return total


def _map_codes(
    frame: pd.DataFrame,
    column: str,
    mapping: Mapping[int, str],
    default: str,
) -> pd.Series:
    values = _number(frame, column).astype("int64", errors="ignore")
    return values.map(mapping).fillna(default).astype(object)


def frs_care_hours(person: pd.DataFrame) -> pd.Series:
    """Weekly care hours from the banded FRS HOURTOT code (lower band edge).

    Blank or non-numeric codes (children, and adults not asked) map to zero;
    an unknown code is a vintage defect and refuses the build.
    """

    codes = _raw_number(person, "hourtot").fillna(0)
    if not np.array_equal(codes.to_numpy(), np.floor(codes.to_numpy())):
        raise ValueError(
            "FRS hourtot codes must be integral to build person.care_hours."
        )
    unknown = sorted(set(codes.astype(int)) - set(FRS_CARE_HOURS_BY_BAND))
    if unknown:
        raise ValueError(f"FRS hourtot carries unknown band code(s) {unknown}.")
    return codes.astype(int).map(FRS_CARE_HOURS_BY_BAND).astype("float64")


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype="float64"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _raw_number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), np.nan), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _frs_benunit_capital(benunit: pd.DataFrame) -> pd.Series:
    """Map TOTCAPB4 to capital, using the sentinel only for unavailable values.

    The I1 FRS 2024-25 audit found all 18,850 rows populated and nonnegative.
    Future raw blanks, nonnumeric values, or negative codes map to
    ``UC_CAPITAL_UNAVAILABLE``; observed zero is preserved.
    """

    raw = _raw_number(benunit, "totcapb4")
    return raw.where(raw.notna() & raw.ge(0), UC_CAPITAL_UNAVAILABLE).astype(float)


def _positive(frame: pd.DataFrame, column: str) -> pd.Series:
    return np.maximum(_number(frame, column), 0)


#: HOUSEHOL.HHSTAT 2: a shared household (shared on an equal basis, the head
#: of household unclear or arbitrary).
FRS_HHSTAT_SHARED = 2
#: ADULT.CONVBL 1: the payer's rent to the householder includes meals (a
#: boarder); anything else is lodging alone (a lodger).
FRS_CONVBL_BOARD_AND_LODGING = 1


def frs_liable_for_share_of_household_rent(
    benunit: pd.DataFrame,
    person: pd.DataFrame,
    household: pd.DataFrame,
    benefits: pd.DataFrame,
) -> np.ndarray:
    """Whether each benefit unit shares liability for its household's rent.

    In a shared household (HHSTAT 2) each benefit unit after the first is
    asked the rent it pays (SRENTAMT, on its adults' records) and the Housing
    Benefit it gets (HBOTHAMT). SRENTAMT is asked after state help with the
    rent, so a unit whose share Universal Credit meets in full can report
    zero in both; a linked UC record with a housing element (UCHOUSEL on a
    BENEFIT 95 record) still makes it liable. A later unit with any of the
    three is one of the people liable for HHRENT, the whole dwelling's rent,
    which policyengine-uk splits among them (pe-uk#2006, uk-data#512).
    Conventional households (HHSTAT 1) are left out: their later units may
    owe the landlord a share or pay the householder as boarders or lodgers,
    and the survey does not say which.
    """

    missing = [
        f"{table}.{column}"
        for table, frame, column in (
            ("househol", household, "hhstat"),
            ("adult", person, "srentamt"),
            ("benunit", benunit, "hbothamt"),
            ("benefits", benefits, "uchousel"),
        )
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Shared rent liability needs the FRS columns {missing}.")
    benunit_id = benunit["benunit_id"].to_numpy()
    shared = (
        _number(household, "hhstat")
        .reindex(benunit["household_id"].to_numpy())
        .eq(FRS_HHSTAT_SHARED)
        .to_numpy()
    )
    share_paid = (
        _positive(person, "srentamt")
        .groupby(person["benunit_id"].to_numpy())
        .sum()
        .reindex(benunit_id, fill_value=0.0)
        .to_numpy()
    )
    housing_benefit = _positive(benunit, "hbothamt").to_numpy()
    with_housing_element = benefits.loc[
        _number(benefits, "benefit").isin(BENEFIT_CODES["universal_credit"])
        & _number(benefits, "uchousel").gt(0),
        "benunit_id",
    ]
    uc_housing = np.isin(benunit_id, with_housing_element.to_numpy())
    later_unit = benunit_id % 100 > 1
    return later_unit & shared & ((share_paid > 0) | (housing_benefit > 0) | uc_housing)


def frs_rent_paid_to_householder(person: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annual rent a boarder and a lodger pays the householder (CVPAY on the payer).

    CONVBL 1 marks board and lodging, a room and at least some meals; anything
    else is lodging alone (pe-uk#2006's rent_paid_as_boarder and
    rent_paid_as_lodger, uk-data#506/#511).
    """

    paid = _positive(person, "cvpay").to_numpy() * WEEKS_IN_YEAR
    boarder = _number(person, "convbl").eq(FRS_CONVBL_BOARD_AND_LODGING).to_numpy()
    return np.where(boarder, paid, 0.0), np.where(boarder, 0.0, paid)


#: Universal Credit start-up period (UC Regs 2013 reg 63; uk-data#527). The
#: codes are from the FRS 2024-25 variable listing (SN 9563).
UC_START_UP_PERIOD_MONTHS = 12
#: ADULT.EMPSTATI full- and part-time self-employed.
FRS_SELF_EMPLOYED_EMPSTATI = (3, 4)
#: ADULT.SDEMP01-12 self employed working full- and part-time.
FRS_SELF_EMPLOYED_ACTIVITIES = (3, 4)
#: JOB.ETYPE: every self-employed description (1 is an employee).
FRS_SELF_EMPLOYED_JOB_ETYPES = (2, 3, 4, 5, 6, 7)
#: JOB.JOBBUS "A business", against "Job".
FRS_JOBBUS_BUSINESS = 2
#: ADULT.SAMESIT "No": the respondent's situation did not change in 12 months.
FRS_SITUATION_UNCHANGED = 2
#: The identity-keyed stream that draws the claim recency of a UC record
#: without a linked claim start date, one draw per benefit unit.
UC_START_UP_CLAIM_RECENCY_SEED = 0
UC_START_UP_CLAIM_RECENCY_SALT = "uc_is_in_startup_period"
_UC_START_UP_PERSON_COLUMNS = (
    "empstati",
    "samesit",
    "seincam2",
    *(f"sdemp{month:02d}" for month in range(1, 13)),
)
_UC_START_UP_JOB_COLUMNS = ("etype", "jobtype", "seend", "sejblong", "jobbus")


def frs_interview_dates(household: pd.DataFrame) -> pd.Series:
    """Interview dates from HOUSEHOL.INTDATE, a SAS date (days since 1960-01-01)."""

    raw = _raw_number(household, "intdate")
    if raw.isna().any():
        raise ValueError("FRS INTDATE (interview date) is missing for some households.")
    return pd.to_datetime(raw, unit="D", origin="1960-01-01")


def parse_frs_uc_claim_start(raw: pd.Series) -> pd.Series:
    """UC claim start dates from BENEFITS.UCSTART, written month/day/year.

    UCSTART comes from DWP administrative data; it is blank where the survey's
    UC record has no linked administrative record. Any other value refuses,
    so a changed format cannot pass silently.
    """

    text = pd.Series(raw, dtype="string").str.strip()
    text = text.mask(text == "")
    parsed = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    unparsed = text.notna() & parsed.isna()
    if unparsed.any():
        raise ValueError(
            f"{int(unparsed.sum())} FRS UCSTART values are not month/day/year dates."
        )
    return parsed


def completed_months(later: pd.Series, earlier: pd.Series) -> np.ndarray:
    """Whole calendar months from ``earlier`` to ``later``; NaN where either is missing."""

    later = pd.DatetimeIndex(later)
    earlier = pd.DatetimeIndex(earlier)
    months = (
        (later.year - earlier.year) * 12
        + (later.month - earlier.month)
        - (later.day < earlier.day)
    )
    return np.where(later.isna() | earlier.isna(), np.nan, months)


def uc_trade_years(
    years_in_job: np.ndarray,
    describes_business: np.ndarray,
    self_employed_all_year: np.ndarray,
) -> np.ndarray:
    """Completed years in the trade behind a self-employed job; NaN when unknown.

    SEJBLONG asks someone running a business how long they have run it and
    anyone else how long they have been in their current self-employed job.
    For a person self-employed throughout the last 12 months, a job under a
    year old is a new engagement in the same trade (ADM H4102 example 4 treats
    a hairdresser turned hairstylist as one trade), so it does not date the
    trade.
    """

    years = np.asarray(years_in_job, dtype=float)
    new_engagement = (
        (years < 1)
        & ~np.asarray(describes_business, dtype=bool)
        & np.asarray(self_employed_all_year, dtype=bool)
    )
    return np.where(new_engagement, np.nan, years)


def frs_uc_start_up_period(
    person: pd.DataFrame,
    pe_person: pd.DataFrame,
    pe_benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    job: pd.DataFrame,
    benefits: pd.DataFrame,
) -> np.ndarray:
    """Whether each person is in a Universal Credit start-up period at interview.

    Reg 63(1), as substituted from 23 September 2020 (SI 2019/1152), starts a
    12-month start-up period when DWP finds a claimant in gainful
    self-employment, unless the minimum income floor already applied for the
    trade; the period is no longer limited to new trades. DWP decides that at
    the start of a claim or when a new trade is reported, so the period is
    running when the person is self-employed and either the benefit unit's UC
    claim (UCSTART) or the trade (SEJBLONG) began less than 12 calendar months
    before interview (INTDATE). A UC record without a linked claim date is
    drawn, keyed on its benefit unit, at the survey-weighted share of linked
    self-employed claimants whose claim began inside the window (uk-data#527).
    The FRS cannot see earlier awards, so a re-claim after the floor applied,
    or a second period within five years, reads as a start-up period; nor a
    move into the all-work-related-requirements group on an old claim, which
    starts a period the flag misses.
    """

    missing = [
        f"{table}.{column}"
        for table, frame, columns in (
            ("adult", person, _UC_START_UP_PERSON_COLUMNS),
            ("job", job, _UC_START_UP_JOB_COLUMNS),
            ("benefits", benefits, ("ucstart_raw",)),
        )
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"The UC start-up period needs the FRS columns {missing} (uk-data#527)."
        )
    interview = frs_interview_dates(household)
    uc_rows = benefits.loc[
        _number(benefits, "benefit").isin(BENEFIT_CODES["universal_credit"])
    ]
    claim_months = pd.Series(
        completed_months(
            interview.reindex(uc_rows["household_id"].to_numpy()),
            parse_frs_uc_claim_start(uc_rows["ucstart_raw"]),
        ),
        index=uc_rows["benunit_id"].to_numpy(),
    )
    benunit_ids = pe_benunit["benunit_id"].to_numpy()
    # One claim per benefit unit; the latest start where its rows disagree.
    months = claim_months.groupby(level=0).min().reindex(benunit_ids).to_numpy()
    unlinked = np.isin(benunit_ids, uc_rows["benunit_id"].to_numpy()) & np.isnan(months)

    held = job.loc[
        _number(job, "etype").isin(FRS_SELF_EMPLOYED_JOB_ETYPES)
        & ~(_number(job, "seend") > 0)
    ]
    # The person's first self-employed job (main job first) is the trade the
    # start-up period follows.
    first_job = (
        held.sort_values(["person_id", "jobtype"])
        .drop_duplicates("person_id")
        .set_index("person_id")
    )
    person_ids = person["person_id"].to_numpy()
    main_job_self_employed = (
        _number(person, "empstati").isin(FRS_SELF_EMPLOYED_EMPSTATI).to_numpy()
    )
    calendar = pd.concat(
        [_number(person, f"sdemp{month:02d}") for month in range(1, 13)], axis=1
    )
    self_employed_all_year = main_job_self_employed & (
        _number(person, "samesit").eq(FRS_SITUATION_UNCHANGED).to_numpy()
        | calendar.isin(FRS_SELF_EMPLOYED_ACTIVITIES).all(axis=1).to_numpy()
    )
    sejblong = _raw_number(first_job, "sejblong")
    trade_years = uc_trade_years(
        sejblong.where(sejblong >= 0).reindex(person_ids).to_numpy(),
        _number(first_job, "jobbus")
        .eq(FRS_JOBBUS_BUSINESS)
        .reindex(person_ids, fill_value=False)
        .to_numpy(),
        self_employed_all_year,
    )
    self_employed = (
        main_job_self_employed
        | np.isin(person_ids, held["person_id"].to_numpy())
        | (_number(person, "seincam2").to_numpy() != 0)
    )

    person_benunit = pd.Index(benunit_ids).get_indexer(pe_person["person_benunit_id"])
    if (person_benunit < 0).any():
        raise ValueError("A person's benefit unit is missing from the FRS spine.")
    person_months = months[person_benunit]
    linked = self_employed & ~np.isnan(person_months)
    weight = (
        _raw_number(household, "gross4")
        .reindex(pe_person["person_household_id"].to_numpy())
        .to_numpy()
    )
    # The household-weighted share of linked self-employed claimants whose
    # claim began inside the window; it treats a missing link as unrelated to
    # the claim's age.
    linked_share = (
        float(
            np.average(
                (person_months[linked] < UC_START_UP_PERIOD_MONTHS).astype(float),
                weights=weight[linked],
            )
        )
        if weight[linked].sum() > 0
        else 0.0
    )
    draws = stable_identity_uniforms(
        benunit_ids,
        seed=UC_START_UP_CLAIM_RECENCY_SEED,
        salt=UC_START_UP_CLAIM_RECENCY_SALT,
    )
    claim_in_window = (months < UC_START_UP_PERIOD_MONTHS) | (
        unlinked & (draws < linked_share)
    )
    return self_employed & (claim_in_window[person_benunit] | (trade_years < 1))


def uc_gainful_self_employment(
    self_employed_main_job: Any, self_employment_income: Any, employment_income: Any
) -> np.ndarray:
    """Whether each person is in gainful self-employment for Universal Credit.

    UC Regs 2013 reg 64(a) asks whether the person carries on a trade as their
    main employment. DWP's Advice for Decision Making starts from hours (H4031)
    but lets earnings outweigh them (H4034). So the flag holds for a
    self-employed main job (FRS EMPSTATI 3 or 4) whatever its profit, since a
    trade at a loss or breaking even is still carried on (H4013, H4054,
    H4503), and for a side trade whose profit is above the person's employment
    income (uk-data#525). It overrides the engine's default, which reads any
    nonzero self-employment income as gainful. The flag is fixed at the
    survey-year (or SPI-drawn) incomes: uprating reprices the two incomes by
    different indices, and the flag does not follow.
    """

    profit = np.asarray(self_employment_income, dtype=float)
    pay = np.asarray(employment_income, dtype=float)
    main_job = np.asarray(self_employed_main_job, dtype=bool)
    return main_job | ((profit > 0) & (profit > pay))


#: FRS yes code for the local-authority registration questions (SPCREG1-3).
FRS_REGISTERED_YES = 1
#: FRS period codes (SN 9563 code frame) for an amount reported per calendar
#: month and per year.
FRS_PERIOD_CALENDAR_MONTH = 5
FRS_PERIOD_YEAR = 52
#: A calendar-month access-fund award is read as annual only when it
#: annualises to more than this multiple of the largest annual-coded award, so
#: a large but plausible monthly payment is left as reported.
ACCESS_FUND_MONTHLY_REPAIR_MULTIPLE = 2.0


def access_fund_annual(
    person: pd.DataFrame, evidence: dict[str, object] | None = None
) -> pd.Series:
    """Annual access-fund award, with one period-code repair.

    The FRS weeklyises ``ACCSSAMT`` from the reported amount and its period
    code ``ACCSSPD``, and the spine annualises it. An access-fund award is paid
    per academic year or term, and on the 2024-25 tab the calendar-month
    amounts run up to the size of a whole annual award: read as monthly, such
    an award annualises to more than ten times every award the survey records
    as annual (microcosm#1095, from the review of #1100). So a calendar-month
    award (code 5) that annualises to more than
    ``ACCESS_FUND_MONTHLY_REPAIR_MULTIPLE`` times the largest award the same
    tab records as annual (code 52) is read as the annual award: its
    per-period amount, ``ACCSSAMT`` x 52/12. On the 2024-25 tab that repairs
    fewer than 10 awards and leaves large but plausible monthly payments as
    reported. Every other award stays as the FRS weeklyised it, including the
    few with no period code on the tab. The threshold moves
    with the tab's largest annual-coded award, so a tab with calendar-month
    awards but no annual-coded award refuses rather than repair nothing
    silently, and so does a tab without the ``ACCSSPD`` column, since the rule
    cannot be applied to either.

    ``evidence``, when given, records the paid awards by period code, the
    annual threshold (``None`` when no award is paid) and how many awards
    were repaired, for the stage's checkpoint evidence.
    """

    amount = _positive(person, "accssamt")
    annual = amount * WEEKS_IN_YEAR
    paid = amount > 0
    record: dict[str, object] = {
        "paid_awards": int(paid.sum()),
        "annual_coded_awards": 0,
        "calendar_month_awards": 0,
        "repair_multiple": ACCESS_FUND_MONTHLY_REPAIR_MULTIPLE,
        "repair_threshold_annual": None,
        "repaired_awards": 0,
    }
    if evidence is not None:
        evidence.update(record)
    if not bool(paid.any()):
        return annual
    if "accsspd" not in person.columns:
        raise KeyError("FRS access fund needs the period code ACCSSPD beside ACCSSAMT.")
    period = _raw_number(person, "accsspd")
    annual_coded = paid & period.eq(FRS_PERIOD_YEAR)
    calendar_month = paid & period.eq(FRS_PERIOD_CALENDAR_MONTH)
    if not bool(annual_coded.any()):
        if bool(calendar_month.any()):
            raise ValueError(
                "FRS access fund has calendar-month awards (ACCSSPD 5) but no "
                "annual-coded award (ACCSSPD 52), so the calendar-month repair "
                "has no threshold on this tab."
            )
        return annual
    threshold = ACCESS_FUND_MONTHLY_REPAIR_MULTIPLE * float(annual[annual_coded].max())
    implausible = calendar_month & annual.gt(threshold)
    if evidence is not None:
        evidence.update(
            annual_coded_awards=int(annual_coded.sum()),
            calendar_month_awards=int(calendar_month.sum()),
            repair_threshold_annual=threshold,
            repaired_awards=int(implausible.sum()),
        )
    return annual.where(~implausible, amount * 52 / 12)


#: The Water Charges Reduction Scheme's maximum reduction, 2021-27. A council
#: tax reduction recipient's charges fall by R = 35 x (A/B) - D percentage
#: points of the gross charge, unless R < 0, on top of the council tax status
#: discount D, where A is the reduction and B the council tax before it, after
#: discounts (Scottish Government, "Water services - charging principles: 2021
#: to 2027", Annex A): the combined reduction is the larger of D and 35% x A/B.
SCOTTISH_WATER_CHARGES_REDUCTION_MAXIMUM = 0.35
#: Share of the gross charges a full council tax reduction recipient without a
#: status discount pays, and the share DWP's CTANNUAL carries for every
#: recipient, which ``frs_council_tax`` nets (uk-data#499, microcosm#1095).
SCOTTISH_WATER_CHARGES_REDUCTION_RECIPIENT_SHARE = (
    1.0 - SCOTTISH_WATER_CHARGES_REDUCTION_MAXIMUM
)
#: FRS 2024-25 CT25D50D codes to the status discount they record, for
#: households with CTDISC 1 (SN 9563 variable listing).
FRS_STATUS_DISCOUNT_BY_CODE = {1: 0.25, 2: 0.5}
SCOTTISH_WATER_AND_SEWERAGE_RAW_COLUMNS = (
    "cwatamt1",
    "csewamt1",
    "cwatamtd",
    "ctdisc",
    "ct25d50d",
    "ctreb",
)


def scottish_water_and_sewerage_weekly(household: pd.DataFrame) -> pd.Series:
    """Weekly Scottish water + sewerage charge the household pays.

    Scotland is not asked ``WATSEWRT`` — its water and sewerage charges ride on
    the council tax bill — so the amount is assembled from the council-tax
    water/sewerage cells.

    FRS 2024-25 retired the two cells the incumbent used. ``CWATAMT``/
    ``CSEWAMT`` ("Wat./Sew. Charge: Final value after discount") are still
    present as headers but carry no data at all in this vintage, and the FRS
    replaced them with the derived ``CWATAMT1``/``CSEWAMT1`` ("Weeklyised gross
    annual dom. water/sew. charge on bill", Scotland only, "DV created in
    2024-25 as variable was removed from the dataset for 2024-25" — SN 9563
    ``9563_dv_summary_2425.xlsx``).

    The household pays the gross charges less the status discount its council
    tax records, which applies to the water and sewerage charges too (25% or
    50%: ``CTDISC`` 1 with ``CT25D50D`` 1 or 2). For a council tax reduction
    recipient (``CTREB`` 1) the Water Charges Reduction Scheme tops that
    discount up to ``SCOTTISH_WATER_CHARGES_REDUCTION_MAXIMUM`` of the gross
    charges, so a recipient pays the gross charges less the larger of its
    discount and 35% (uk-data#499, microcosm#1095). ``CWATAMTD`` ("Deriv
    Council Tax water charge -Scot") is not that amount: on the 2024-25 tab it
    sits at a median 0.83 of the gross water charge for undiscounted
    non-recipients and 0.87 for undiscounted recipients, so it is read only
    where no gross water cell exists.

    The scheme's 35% tapers with the share A/B of the council tax the
    reduction covers, but every recipient here takes the full-reduction case,
    A/B = 1. 86% of Scotland's recipients receive full council tax reduction
    (Council Tax Reduction in Scotland 2024-25, March 2025, average award
    GBP 16.55 a week), while none of the 234 Scottish recipients with a
    recorded ``CTREBAMT`` on the 2024-25 tab pays nothing after it (19% of
    English recipients do) and their recorded amounts have a median of GBP 6 a
    week, so the recorded amount cannot carry the share.

    Every value is weeklyised (all the amount cells sit in the FRS "weekly
    variables" listing), so callers apply ``WEEKS_IN_YEAR`` themselves.

    On the 2024-25 tab the domain splits cleanly: 1,641 Scottish households
    carry a positive gross water charge; 22 carry a recorded ``CWATAMTD``
    with no gross bill cell, and their ``CSEWAMT1`` is zero, so they pay the
    recorded water charge; 21 carry no council-tax cells at all and fall to
    zero.
    """

    missing = sorted(set(SCOTTISH_WATER_AND_SEWERAGE_RAW_COLUMNS) - set(household))
    if missing:
        raise KeyError(f"Scottish water assembly needs raw household cells {missing}.")
    water_gross = _positive(household, "cwatamt1")
    sewerage_gross = _positive(household, "csewamt1")
    billed = water_gross > 0
    # A household with no gross water bill carries no gross sewerage either,
    # so its recorded CWATAMTD is the whole charge. That claim is a property
    # of the vintage, not of the code — so it is asserted, and a vintage that
    # breaks it refuses at build time instead of mixing the two bases.
    unbilled_sewerage = ~billed & (sewerage_gross > 0)
    if bool(unbilled_sewerage.any()):
        raise ValueError(
            "Scottish water assembly: "
            f"{int(unbilled_sewerage.sum())} household(s) carry a gross sewerage "
            "charge (CSEWAMT1 > 0) with no gross water bill (CWATAMT1 == 0), so "
            "their charge cannot be assembled on one basis; this vintage needs "
            "an adjudicated rule for these households."
        )
    discount = (
        _number(household, "ct25d50d")
        .map(FRS_STATUS_DISCOUNT_BY_CODE)
        .where(_number(household, "ctdisc") == 1, 0.0)
        .fillna(0.0)
    )
    reduction = np.where(
        _number(household, "ctreb") == 1,
        np.maximum(discount, SCOTTISH_WATER_CHARGES_REDUCTION_MAXIMUM),
        discount,
    )
    paid = (water_gross + sewerage_gross) * (1.0 - reduction)
    return pd.Series(
        np.where(billed, paid, _positive(household, "cwatamtd")), index=household.index
    )


def _reject_nan(frame: pd.DataFrame, entity: str) -> None:
    if frame.isna().any().any():
        bad = sorted(frame.columns[frame.isna().any()].tolist())
        raise ValueError(f"FRS spine {entity} produced NaN column(s): {bad}.")


artifact_by_table = _artifact_by_table
read_pinned_tab = _read_pinned_tab
normalize_ids = _normalize_ids
sum_to_entity = _sum_to_entity
map_codes = _map_codes
number = _number
raw_number = _raw_number
positive = _positive
reject_nan = _reject_nan
