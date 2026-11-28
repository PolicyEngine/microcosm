"""Candidate UK firm calibration targets beyond the committed contract (M2).

``UK_FIRM_TARGET_CANDIDATES`` declares the published firm statistics that
could calibrate the policyengine-uk firm inputs the generator does not draw
yet (paybill, profit, capital expenditure, R&D, rateable value) and the ONS
attribute tables the firm attribute draws read. They are declarations only:

- no observed values, band edges or Ledger record-set ids
  (``ledger_record_set_id`` stays ``None`` until a Ledger source records the
  statistic);
- nothing reads them in calibration; ``uk_firms_targets.json`` keeps the
  committed contract's targets;
- ``feeds_inputs`` names the formula-less policyengine-uk firm inputs a
  target would inform, so most inputs registered as not set can point at a
  candidate that would inform them. Some not-set inputs have no candidate
  yet; the tests pin which.

``status`` is ``pending_ledger_source`` for a statistic no Ledger source or
Microcosm fixture holds in the form the candidate needs, and
``fixture_available`` for an ONS UK Business table Microcosm holds as a test
fixture at the candidate's dimensions. ``publication`` is the title as the
publisher gives it: an edition title (with its year) where the candidate
names one edition, such as the ONS UK Business 2025 tables Microcosm holds,
and a series title otherwise, where the edition a target reads is the Ledger
source's choice. Concepts follow
``uk_firms_targets.json`` (``uk.firm.count``, ``uk.firm.annual_turnover``,
``uk.firm.sic_code``, ``uk.firm.turnover_band``, ``uk.firm.employment_band``),
with new ``uk.firm.*`` and ``uk.tax.*`` ids in the same style where the
contract has none.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A statistic no Ledger source or Microcosm fixture holds yet, at the
#: candidate's dimensions.
PENDING_LEDGER_SOURCE = "pending_ledger_source"
#: An ONS UK Business table Microcosm holds as a test fixture, at the
#: candidate's dimensions.
FIXTURE_AVAILABLE = "fixture_available"
UK_FIRM_TARGET_CANDIDATE_STATUSES = (PENDING_LEDGER_SOURCE, FIXTURE_AVAILABLE)

_ONS_UK_BUSINESS_2025 = "UK business; activity, size and location: 2025"
_DBT_BPE_2025 = "Business population estimates for the UK and regions 2025"
#: The UK-level ABS release, which carries the SIC-level estimates. The
#: combined "UK and regional" title was retired after the 2021 results.
_ONS_ABS = "Non-financial business economy, UK (Annual Business Survey)"
_HMRC_CT = "Corporation Tax statistics"
_HMRC_RD = "Research and Development Tax Credits Statistics"
_HMRC_PAYE_RTI = (
    "Earnings and employment from Pay As You Earn Real Time Information, UK"
)
_MHCLG_NNDR = (
    "National non-domestic rates collected by councils in England: "
    "forecast 2025 to 2026"
)
_VOA_STOCK = "Non-domestic rating: stock of properties"


@dataclass(frozen=True)
class UKFirmTargetFilter:
    """A predicate on a firm concept: ``concept operator value``."""

    concept: str
    operator: str
    value: object


@dataclass(frozen=True)
class UKFirmTargetCandidate:
    """One candidate firm target: what it would measure and what it would feed."""

    target_id: str
    family: str
    source_name: str
    publication: str
    concept: str
    status: str
    feeds_inputs: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: tuple[UKFirmTargetFilter, ...] = ()
    #: The table of ``publication`` the candidate reads, where it names one.
    source_table: str | None = None
    entity: str = "firm"
    ledger_record_set_id: str | None = None


_COMPANY = UKFirmTargetFilter("uk.firm.legal_form", "==", "COMPANY")

UK_FIRM_TARGET_CANDIDATES: tuple[UKFirmTargetCandidate, ...] = (
    # ONS UK Business 2025 tables Microcosm holds as fixtures.
    UKFirmTargetCandidate(
        target_id="ons.uk_business.enterprise_count.sic_region_turnover_bands",
        family="ons_uk_business",
        source_name="ons",
        publication=_ONS_UK_BUSINESS_2025,
        source_table="Table 8",
        concept="uk.firm.count",
        dimensions=("uk.firm.sic_code", "uk.firm.region", "uk.firm.turnover_band"),
        status=FIXTURE_AVAILABLE,
        feeds_inputs=("firm_region", "firm_in_england"),
    ),
    UKFirmTargetCandidate(
        target_id=(
            "ons.uk_business.enterprise_count.industry_legal_status_turnover_bands"
        ),
        family="ons_uk_business",
        source_name="ons",
        publication=_ONS_UK_BUSINESS_2025,
        source_table="Table 13",
        concept="uk.firm.count",
        dimensions=(
            "uk.firm.industry_group",
            "uk.firm.legal_status",
            "uk.firm.turnover_band",
        ),
        status=FIXTURE_AVAILABLE,
        feeds_inputs=("firm_legal_form", "firm_is_public_body"),
    ),
    UKFirmTargetCandidate(
        target_id=(
            "ons.uk_business.enterprise_count.region_legal_status_turnover_bands"
        ),
        family="ons_uk_business",
        source_name="ons",
        publication=_ONS_UK_BUSINESS_2025,
        source_table="Table 15",
        concept="uk.firm.count",
        dimensions=(
            "uk.firm.region",
            "uk.firm.legal_status",
            "uk.firm.turnover_band",
        ),
        status=FIXTURE_AVAILABLE,
        feeds_inputs=(
            "firm_region",
            "firm_in_england",
            "firm_legal_form",
            "firm_is_public_body",
        ),
    ),
    UKFirmTargetCandidate(
        # Table 3 is SIC division x employment band x region; Microcosm holds
        # no regional extract of it, so it waits for a source like the rest.
        target_id="ons.uk_business.enterprise_count.region_employment_bands",
        family="ons_uk_business",
        source_name="ons",
        publication=_ONS_UK_BUSINESS_2025,
        source_table="Table 3",
        concept="uk.firm.count",
        dimensions=(
            "uk.firm.sic_code",
            "uk.firm.region",
            "uk.firm.employment_band",
        ),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_region", "firm_in_england", "firm_employees"),
    ),
    # DBT business population estimates: all private-sector businesses,
    # unregistered ones included. BPE's legal forms (companies, sole
    # proprietorships, partnerships) are not ONS legal status codes 1-7.
    # The opt-in headcount redraw (firm_headcount.py, bpe_dir) reads the
    # registered 0-4 employee rows of BPE 2025 Table 3, so those rows are
    # in-sample for a build that uses it.
    UKFirmTargetCandidate(
        target_id="dbt.bpe.business_count.legal_form_employment_bands",
        family="dbt_bpe",
        source_name="dbt",
        publication=_DBT_BPE_2025,
        concept="uk.firm.count",
        dimensions=("uk.firm.legal_form", "uk.firm.employment_band"),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_legal_form", "firm_employees"),
    ),
    UKFirmTargetCandidate(
        target_id="dbt.bpe.business_count.region_employment_bands",
        family="dbt_bpe",
        source_name="dbt",
        publication=_DBT_BPE_2025,
        concept="uk.firm.count",
        dimensions=("uk.firm.region", "uk.firm.employment_band"),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_region", "firm_in_england", "firm_employees"),
    ),
    UKFirmTargetCandidate(
        target_id="dbt.bpe.turnover.sic_employment_bands",
        family="dbt_bpe",
        source_name="dbt",
        publication=_DBT_BPE_2025,
        concept="uk.firm.annual_turnover",
        dimensions=("uk.firm.sic_code", "uk.firm.employment_band"),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_turnover", "firm_employees"),
    ),
    # ONS Annual Business Survey: the non-financial business economy.
    UKFirmTargetCandidate(
        target_id="ons.abs.turnover.sic",
        family="ons_abs",
        source_name="ons",
        publication=_ONS_ABS,
        concept="uk.firm.annual_turnover",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_turnover",),
    ),
    UKFirmTargetCandidate(
        target_id="ons.abs.purchases.sic",
        family="ons_abs",
        source_name="ons",
        publication=_ONS_ABS,
        concept="uk.firm.input_costs",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_input_costs", "firm_input_vat"),
    ),
    UKFirmTargetCandidate(
        target_id="ons.abs.employment_costs.sic",
        family="ons_abs",
        source_name="ons",
        publication=_ONS_ABS,
        concept="uk.firm.paybill",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_paybill",),
    ),
    UKFirmTargetCandidate(
        target_id="ons.abs.capital_expenditure.sic",
        family="ons_abs",
        source_name="ons",
        publication=_ONS_ABS,
        concept="uk.firm.capital_expenditure",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_capex_main_pool_new",
            "firm_capex_main_pool_used",
            "firm_capex_special_rate_new",
            "firm_capex_special_rate_used",
            "firm_capex_structures",
        ),
    ),
    # HMRC Corporation Tax statistics: companies only. Companies and
    # liability are banded by corporation tax liability (Tables 10A/10B),
    # not by taxable profit.
    UKFirmTargetCandidate(
        target_id="hmrc.corporation_tax.company_count.liability_bands",
        family="hmrc_corporation_tax",
        source_name="hmrc",
        publication=_HMRC_CT,
        concept="uk.firm.count",
        dimensions=("uk.tax.corporation_tax.liability_band",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_legal_form",
            "firm_taxable_trading_profit",
            "firm_non_trading_profits",
        ),
    ),
    UKFirmTargetCandidate(
        target_id="hmrc.corporation_tax.liability.liability_bands",
        family="hmrc_corporation_tax",
        source_name="hmrc",
        publication=_HMRC_CT,
        concept="uk.tax.corporation_tax.liability",
        dimensions=("uk.tax.corporation_tax.liability_band",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_taxable_trading_profit",
            "firm_non_trading_profits",
            "firm_losses_brought_forward",
            "firm_n_associated_companies",
        ),
    ),
    UKFirmTargetCandidate(
        target_id="hmrc.corporation_tax.liability.sic",
        family="hmrc_corporation_tax",
        source_name="hmrc",
        publication=_HMRC_CT,
        concept="uk.tax.corporation_tax.liability",
        dimensions=("uk.firm.sic_code",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_taxable_trading_profit", "firm_is_bank"),
    ),
    # HMRC capital allowances: Tables 12-14 of the Corporation Tax
    # statistics (12A claims by allowance type; 14A qualifying expenditure
    # by type). There is no separate capital allowances series.
    UKFirmTargetCandidate(
        target_id="hmrc.capital_allowances.amount_claimed.allowance_type",
        family="hmrc_capital_allowances",
        source_name="hmrc",
        publication=_HMRC_CT,
        source_table="Table 12A",
        concept="uk.tax.corporation_tax.capital_allowances",
        dimensions=("uk.tax.capital_allowance_type",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_capex_main_pool_new",
            "firm_capex_main_pool_used",
            "firm_capex_special_rate_new",
            "firm_capex_special_rate_used",
            "firm_capex_structures",
            "firm_aia_share",
            "firm_main_pool_brought_forward",
            "firm_special_rate_pool_brought_forward",
            "firm_sba_qualifying_cost_brought_forward",
            "firm_disposal_proceeds_main_pool",
        ),
    ),
    # HMRC R&D tax relief statistics: claims and expenditure by scheme.
    UKFirmTargetCandidate(
        target_id="hmrc.rd_tax_relief.claim_count.scheme",
        family="hmrc_rd_tax_relief",
        source_name="hmrc",
        publication=_HMRC_RD,
        concept="uk.firm.count",
        dimensions=("uk.tax.rd_relief_scheme",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_is_sme", "firm_qualifying_rd_expenditure"),
    ),
    UKFirmTargetCandidate(
        target_id="hmrc.rd_tax_relief.qualifying_expenditure.scheme",
        family="hmrc_rd_tax_relief",
        source_name="hmrc",
        publication=_HMRC_RD,
        concept="uk.firm.qualifying_rd_expenditure",
        dimensions=("uk.tax.rd_relief_scheme",),
        filters=(_COMPANY,),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_is_sme",
            "firm_qualifying_rd_expenditure",
            "firm_total_relevant_expenditure",
        ),
    ),
    # HMRC PAYE Real Time Information: payrolled employees and their pay.
    UKFirmTargetCandidate(
        target_id="hmrc.paye_rti.payrolled_employees.sic",
        family="hmrc_paye_rti",
        source_name="hmrc",
        publication=_HMRC_PAYE_RTI,
        concept="uk.firm.employees",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_employees",),
    ),
    UKFirmTargetCandidate(
        # PAYE RTI regions are the employee's residence (NUTS1), not the
        # employer's region, so this does not inform firm_region or
        # firm_in_england.
        target_id="hmrc.paye_rti.payrolled_employees.region",
        family="hmrc_paye_rti",
        source_name="hmrc",
        publication=_HMRC_PAYE_RTI,
        concept="uk.firm.employees",
        dimensions=("uk.firm.region",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_employees",),
    ),
    UKFirmTargetCandidate(
        # Derived, not published: PAYE RTI gives mean pay and payrolled
        # employees by SIC section, and the paybill would be their product.
        target_id="hmrc.paye_rti.paybill.sic",
        family="hmrc_paye_rti",
        source_name="hmrc",
        publication=_HMRC_PAYE_RTI,
        concept="uk.firm.paybill",
        dimensions=("uk.firm.sic_code",),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_paybill",),
    ),
    # Business rates in England: MHCLG collection and VOA rating lists.
    UKFirmTargetCandidate(
        target_id="mhclg.nndr.gross_rates_payable.england",
        family="mhclg_nndr",
        source_name="mhclg",
        publication=_MHCLG_NNDR,
        concept="uk.tax.business_rates.gross_liability",
        filters=(UKFirmTargetFilter("uk.firm.in_england", "==", True),),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=("firm_rateable_value", "firm_in_england", "firm_n_properties"),
    ),
    UKFirmTargetCandidate(
        target_id="mhclg.nndr.reliefs.relief_type",
        family="mhclg_nndr",
        source_name="mhclg",
        publication=_MHCLG_NNDR,
        concept="uk.tax.business_rates.relief",
        dimensions=("uk.tax.business_rates_relief_type",),
        filters=(UKFirmTargetFilter("uk.firm.in_england", "==", True),),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_rateable_value",
            "firm_is_charity",
            "firm_is_rhl_property",
            "firm_is_pub_or_live_music_venue",
            "firm_property_occupied",
            "firm_months_empty",
        ),
    ),
    # The VOA stock of properties covers England and Wales only.
    UKFirmTargetCandidate(
        target_id="voa.non_domestic_rating.rateable_value.sector_region",
        family="voa_non_domestic_rating",
        source_name="voa",
        publication=_VOA_STOCK,
        concept="uk.firm.rateable_value",
        dimensions=("uk.property.non_domestic_sector", "uk.firm.region"),
        status=PENDING_LEDGER_SOURCE,
        feeds_inputs=(
            "firm_rateable_value",
            "firm_is_industrial_property",
            "firm_n_properties",
            "firm_region",
        ),
    ),
)


def uk_firm_target_candidates_feeding(input_name: str) -> tuple[str, ...]:
    """Ids of the candidates whose ``feeds_inputs`` name ``input_name``."""

    return tuple(
        candidate.target_id
        for candidate in UK_FIRM_TARGET_CANDIDATES
        if input_name in candidate.feeds_inputs
    )


__all__ = [
    "FIXTURE_AVAILABLE",
    "PENDING_LEDGER_SOURCE",
    "UK_FIRM_TARGET_CANDIDATES",
    "UK_FIRM_TARGET_CANDIDATE_STATUSES",
    "UKFirmTargetCandidate",
    "UKFirmTargetFilter",
    "uk_firm_target_candidates_feeding",
]
