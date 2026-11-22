"""UK firm tables in policyengine-uk firm variable names, in pounds.

Two views of the experimental UK firm generator's rows:

- ``build_candidate_rules_table`` is the pre-calibration table the VAT rules
  (``firm_rules.UKFirmRules``) evaluate: no weight, no VAT scope and no
  registration, with all turnover standard-rated so the rules return each
  firm's liability if registered.
- ``to_policyengine_uk_firm_table`` is the calibrated export, checked by
  ``validate_uk_firm_table``.

The export's layout is a base plus optional input groups. The base is
policyengine-uk's firm inputs the generator always gives
(``UK_FIRM_POLICYENGINE_UK_COLUMNS``) followed by the Microcosm-only columns
(``UK_FIRM_MICROCOSM_COLUMNS``): ``UK_FIRM_TABLE_COLUMNS``.
``UK_FIRM_EXPORT_GROUPS`` registers each optional group
(``UKFirmExportGroup``): the source columns that switch it on, the
policyengine-uk and Microcosm-only columns it adds, the groups it extends,
and how it fills and checks its columns. The export emits the base and every
group whose source columns the rows carry; ``uk_firm_table_columns`` gives a
set of groups' layout, which places every group's policyengine-uk columns,
in registry order, after the base ones, and its Microcosm-only columns after
the base Microcosm-only ones. Adding an input is one registry entry.

The groups today:

- ``attributes``: the ONS attribute draws (``ons_legal_status`` and
  ``ons_region_code``) give ``firm_legal_form``, ``firm_region``,
  ``firm_in_england`` and ``firm_is_public_body``; the draws stay as
  Microcosm-only columns. Its layout is ``UK_FIRM_EXTENDED_TABLE_COLUMNS``.
- ``paybill``: ``firm_paybill`` (pounds, from
  ``firm_paybill.assign_uk_firm_paybill``), carried unchanged. It extends
  ``attributes``; with it the layout is ``UK_FIRM_PAYBILL_TABLE_COLUMNS``.
- ``employment_allowance``: the assumed ``firm_claims_employment_allowance``
  and ``firm_is_single_director_company`` flags (from
  ``firm_employment_allowance.assign_uk_firm_employment_allowance``), checked
  against ``ons_legal_status`` and ``firm_employees``. It extends
  ``attributes``.
- ``profit``: the assumed ``firm_taxable_trading_profit`` (pounds, from
  ``firm_profit.assign_uk_firm_taxable_profit``), zero for non-companies.
  It extends ``attributes``.
- ``rateable_value``: the assumed ``firm_rateable_value`` (pounds, from
  ``firm_rateable_value.assign_uk_firm_rateable_value``), zero outside
  England. It extends ``attributes``.

``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS`` records, for every formula-less
policyengine-uk firm input, whether the fullest table (every group) sets it.
The base table leaves ``firm_region`` and ``firm_legal_form`` unset, so
policyengine-uk's defaults (London, company) apply; without a paybill source
``firm_paybill`` is unset too, and the other money inputs are never set.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_employment_allowance import (
    UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
    uk_firm_employment_allowance_flags,
)
from microcosm.build.uk_runtime.occupation_constants import (
    UK_SIC2007_SECTION_DIVISIONS,
    uk_sic2007_sections,
)
from microcosm.calibrate.geography_constants import (
    UK_REGION_TIER,
    UK_REGION_TIER_ENUM,
)

#: Share of a firm's input costs charged as VAT. A data assumption, not VAT
#: law: every purchase is taken to be standard-rated at 20%. Replace it with
#: sector purchase mixes (zero-rated, reduced-rated and exempt purchases) in
#: M3.
ASSUMED_INPUT_VAT_RATE = 0.20

#: policyengine-uk's ``SICSection`` member for a firm whose SIC 2007 division
#: lies in no section.
UK_FIRM_SIC_SECTION_UNKNOWN = "UNKNOWN"

#: policyengine-uk's ``SICSection`` member names: the SIC 2007 sections A-U of
#: ``UK_SIC2007_SECTION_DIVISIONS`` and the unknown member.
UK_FIRM_SIC_SECTIONS: tuple[str, ...] = (
    UK_FIRM_SIC_SECTION_UNKNOWN,
    *UK_SIC2007_SECTION_DIVISIONS,
)

#: Generator columns ``build_candidate_rules_table`` reads.
UK_FIRM_CANDIDATE_SOURCE_COLUMNS = (
    "firm_id",
    "sic_code",
    "annual_turnover_k",
    "annual_input_k",
    "employment",
)

#: The candidate table the VAT rules evaluate, in order.
UK_FIRM_CANDIDATE_RULES_COLUMNS = (
    "firm_id",
    "firm_turnover",
    "firm_input_costs",
    "firm_employees",
    "firm_sic_section",
    "firm_standard_rated_supplies",
    "firm_exempt_supplies",
    "firm_input_vat",
)

#: Generator columns ``to_policyengine_uk_firm_table`` reads.
UK_FIRM_EXPORT_SOURCE_COLUMNS = (
    *UK_FIRM_CANDIDATE_SOURCE_COLUMNS,
    "firm_weight",
    "vat_scope",
    "vat_registered",
    "in_frame",
    "base_weight",
    "data_vintage",
    "source_firm_id",
    "source_firm_key",
)

#: policyengine-uk firm input variables in the export, in order.
UK_FIRM_POLICYENGINE_UK_COLUMNS = (
    "firm_id",
    "firm_weight",
    "firm_turnover",
    "firm_input_costs",
    "firm_employees",
    "firm_sic_section",
    "firm_standard_rated_supplies",
    "firm_exempt_supplies",
    "firm_vat_voluntarily_registered",
    "firm_input_vat",
)

#: Microcosm-only columns the export carries after the policyengine-uk ones.
UK_FIRM_MICROCOSM_COLUMNS = (
    "in_frame",
    "vat_scope",
    "vat_registered",
    "base_weight",
    "data_vintage",
    "source_firm_id",
    "source_firm_key",
)

#: The export's columns, in order.
UK_FIRM_TABLE_COLUMNS = (*UK_FIRM_POLICYENGINE_UK_COLUMNS, *UK_FIRM_MICROCOSM_COLUMNS)

#: The ONS attribute draws the extended export reads and carries: the ONS
#: legal status code (1-7, UK Business: Activity, Size and Location tables)
#: and the GSS code of the firm's English region or nation. Microcosm-only.
UK_FIRM_ONS_ATTRIBUTE_COLUMNS = ("ons_legal_status", "ons_region_code")

#: policyengine-uk ``FirmLegalForm`` member names, in declaration order.
UK_FIRM_LEGAL_FORMS: tuple[str, ...] = (
    "COMPANY",
    "SOLE_TRADER",
    "PARTNERSHIP",
    "OTHER",
)

#: ONS legal status code -> policyengine-uk ``FirmLegalForm`` member name.
#:
#: 1 company (including building societies), 2 sole proprietor and
#: 3 partnership map one to one. 4 public corporation or nationalised body,
#: 5 central government, 6 local authority and 7 non-profit body or mutual
#: association all map to ``OTHER``. That is an assumption: policyengine-uk
#: charges corporation tax on companies only, so ``OTHER`` takes those bodies
#: out of corporation tax (public corporations and some non-profit bodies do
#: pay it). The raw code stays in ``ons_legal_status``.
UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM: Mapping[int, str] = MappingProxyType(
    {
        1: "COMPANY",
        2: "SOLE_TRADER",
        3: "PARTNERSHIP",
        4: "OTHER",
        5: "OTHER",
        6: "OTHER",
        7: "OTHER",
    }
)

#: ONS legal statuses read as public authorities for ``firm_is_public_body``:
#: central government (5) and local authorities (6). An assumption: public
#: corporations (4) are left out.
UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES: frozenset[int] = frozenset({5, 6})

#: GSS region codes -> policyengine-uk ``Region`` member names: the shared
#: region tier (nine English regions, then Wales, Scotland and Northern
#: Ireland). ``Region.UNKNOWN`` is never emitted.
UK_FIRM_REGION_CODE_TO_REGION: Mapping[str, str] = UK_REGION_TIER_ENUM

#: GSS codes of the nine English regions, for ``firm_in_england``.
UK_FIRM_ENGLISH_REGION_CODES: frozenset[str] = frozenset(
    code for tier, code in UK_REGION_TIER if tier == "region"
)

#: policyengine-uk firm inputs the extended export adds, in order.
UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS = (
    "firm_legal_form",
    "firm_region",
    "firm_in_england",
    "firm_is_public_body",
)

#: policyengine-uk firm inputs the paybill group adds.
UK_FIRM_PAYBILL_POLICYENGINE_UK_COLUMNS = ("firm_paybill",)

#: Values of ``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS``:
#:
#: - ``set``: mapped directly from a generator column or ONS draw;
#: - ``derived``: computed from other columns by a rule;
#: - ``assumed``: carries a data assumption beyond the source;
#: - ``not_set``: not exported; policyengine-uk's default applies.
UK_FIRM_INPUT_STATUSES = ("set", "derived", "assumed", "not_set")

#: Every formula-less policyengine-uk firm input (branch firm-entity-engine
#: at ``57a99a67a``, the committed snapshot; policyengine_uk/variables/firm/) -> how the paybill export, the fullest
#: one, fills it. The keys with status other than ``not_set`` are exactly the
#: base policyengine-uk columns and those of every ``UK_FIRM_EXPORT_GROUPS``
#: entry (``UK_FIRM_PAYBILL_EXTENDED_POLICYENGINE_UK_COLUMNS``); a table
#: without some groups sets a subset of them.
UK_FIRM_POLICYENGINE_UK_INPUT_STATUS: Mapping[str, str] = MappingProxyType(
    {
        # Generator columns, mapped directly (thousands of pounds to pounds,
        # SIC division to section).
        "firm_id": "set",
        "firm_weight": "set",
        "firm_turnover": "set",
        "firm_input_costs": "set",
        "firm_employees": "set",
        "firm_sic_section": "set",
        # GSS code to Region member, one to one.
        "firm_region": "set",
        # Turnover split by VAT scope; voluntary registration from the
        # registration threshold.
        "firm_standard_rated_supplies": "derived",
        "firm_exempt_supplies": "derived",
        "firm_vat_voluntarily_registered": "derived",
        # policyengine-uk's firm_in_england means the firm's property is in
        # England; the export reads it from the enterprise's ONS region, so a
        # multi-site enterprise's property is assumed to sit in its region.
        "firm_in_england": "assumed",
        # ASSUMED_INPUT_VAT_RATE on all input costs; ONS statuses 4-7 to
        # OTHER; ONS statuses 5 and 6 as public authorities.
        "firm_input_vat": "assumed",
        "firm_legal_form": "assumed",
        "firm_is_public_body": "assumed",
        # Headcount x ONS ASHE 2025 (tax year 2024-25) all-employees mean
        # annual gross pay of the firm's SIC section (firm_paybill.py): every
        # firm pays its section's mean, and ONS employment, which counts
        # working proprietors, stands in for employees. Only when the build
        # is given a paybill source.
        "firm_paybill": "assumed",
        # Not drawn: no source yet. Placeholders would flow into NIC, CT,
        # VAT and business rates revenue.
        # Assumed (firm_profit.py): HMRC Corporation Tax statistics
        # liability bands (2024-25) inverted through the rate schedule and
        # allotted to companies by turnover rank. In-sample for CT. Only when
        # the build is given a profit source.
        "firm_taxable_trading_profit": "assumed",
        # Assumed (firm_rateable_value.py): VOA 2023-list hereditaments in
        # England allotted to firms by headcount rank. Only when the build is
        # given a rateable value source.
        "firm_rateable_value": "assumed",
        "firm_n_associated_companies": "not_set",
        "firm_is_bank": "not_set",
        "firm_is_charity": "not_set",
        # Assumed from ONS legal status and employment
        # (firm_employment_allowance.py): a company with employment 1 is a
        # single-director company. Only when the build sets the Employment
        # Allowance flags.
        "firm_is_single_director_company": "assumed",
        "firm_in_freeport": "not_set",
        # VAT.
        "firm_zero_rated_supplies": "not_set",
        "firm_reduced_rated_supplies": "not_set",
        "firm_is_limited_cost_trader": "not_set",
        "firm_uses_flat_rate_scheme": "not_set",
        # Business rates.
        "firm_is_industrial_property": "not_set",
        "firm_is_pub_or_live_music_venue": "not_set",
        "firm_is_rhl_property": "not_set",
        "firm_months_empty": "not_set",
        "firm_n_properties": "not_set",
        "firm_property_occupied": "not_set",
        # Capital allowances.
        "firm_aia_share": "not_set",
        "firm_capex_main_pool_new": "not_set",
        "firm_capex_main_pool_used": "not_set",
        "firm_capex_special_rate_new": "not_set",
        "firm_capex_special_rate_used": "not_set",
        "firm_capex_structures": "not_set",
        "firm_disposal_proceeds_main_pool": "not_set",
        "firm_main_pool_brought_forward": "not_set",
        "firm_sba_qualifying_cost_brought_forward": "not_set",
        "firm_special_rate_pool_brought_forward": "not_set",
        # Corporation tax.
        "firm_bank_surcharge_allowance_share": "not_set",
        "firm_deductions_allowance_share": "not_set",
        "firm_exempt_distributions": "not_set",
        "firm_losses_brought_forward": "not_set",
        "firm_non_trading_profits": "not_set",
        # Employer costs.
        "firm_apprenticeship_levy_allowance_share": "not_set",
        # Assumed (firm_employment_allowance.py): sole proprietors with
        # employment of at most 1 and partnerships with employment of at
        # most 2 are their proprietors alone and do not claim. Only when the build sets the
        # Employment Allowance flags.
        "firm_claims_employment_allowance": "assumed",
        "firm_zero_rated_class_1_earnings": "not_set",
        # Pay distribution within the paybill (policyengine-uk ca037fad2).
        "firm_employees_above_auto_enrolment_upper_limit": "not_set",
        "firm_employees_below_auto_enrolment_earnings_trigger": "not_set",
        "firm_employees_below_secondary_threshold": "not_set",
        "firm_paybill_above_auto_enrolment_upper_limit": "not_set",
        "firm_paybill_below_auto_enrolment_earnings_trigger": "not_set",
        "firm_paybill_below_secondary_threshold": "not_set",
        # National Minimum Wage floor (policyengine-uk ca037fad2).
        "firm_low_paid_hours": "not_set",
        "firm_low_paid_paybill": "not_set",
        # R&D.
        "firm_is_sme": "not_set",
        "firm_qualifying_rd_expenditure": "not_set",
        "firm_total_relevant_expenditure": "not_set",
    }
)

_BOOL_COLUMNS = (
    "firm_vat_voluntarily_registered",
    "in_frame",
    "vat_scope",
    "vat_registered",
)
_NON_NEGATIVE_COLUMNS = (
    "firm_weight",
    "base_weight",
    "firm_turnover",
    "firm_input_costs",
    "firm_standard_rated_supplies",
    "firm_exempt_supplies",
    "firm_input_vat",
)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} misses columns {missing}.")


def _firm_ids(frame: pd.DataFrame) -> np.ndarray:
    ids = frame["firm_id"]
    if not pd.api.types.is_integer_dtype(ids):
        raise ValueError("UK firm firm_id must be integers.")
    if ids.duplicated().any():
        raise ValueError("UK firm firm_id repeats.")
    return ids.to_numpy(np.int64)


def _gbp(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(np.float64) * 1000.0
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"UK firm {column} must be finite and non-negative.")
    return values


def _employees(frame: pd.DataFrame) -> np.ndarray:
    values = frame["employment"].to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        as_float = values.astype(np.float64)
        if not np.all(np.isfinite(as_float)) or np.any(as_float != np.round(as_float)):
            raise ValueError("UK firm employment must be whole numbers.")
    values = values.astype(np.int64)
    if np.any(values < 0):
        raise ValueError("UK firm employment must be non-negative.")
    return values


def _ons_attributes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``ons_legal_status`` as int64 and ``ons_region_code`` as str, checked."""

    status = frame["ons_legal_status"]
    if status.isna().any() or not pd.api.types.is_integer_dtype(status):
        raise ValueError("UK firm ons_legal_status must be integers.")
    status_values = status.to_numpy(np.int64)
    if not np.isin(status_values, list(UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM)).all():
        raise ValueError(
            "UK firm ons_legal_status must be ONS legal status codes "
            f"{sorted(UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM)}."
        )
    codes = frame["ons_region_code"]
    if not codes.map(lambda code: isinstance(code, str)).all():
        raise ValueError("UK firm ons_region_code must be strings.")
    code_values = codes.to_numpy(object)
    if not pd.Series(code_values).isin(list(UK_FIRM_REGION_CODE_TO_REGION)).all():
        raise ValueError(
            "UK firm ons_region_code must be GSS codes of the English regions "
            "or the nations."
        )
    return status_values, code_values


def _attribute_columns(
    status: np.ndarray, region_code: np.ndarray
) -> dict[str, np.ndarray]:
    """The policyengine-uk columns the ONS attribute draws give."""

    return {
        "firm_legal_form": np.array(
            [UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM[int(s)] for s in status],
            dtype=object,
        ),
        "firm_region": np.array(
            [UK_FIRM_REGION_CODE_TO_REGION[code] for code in region_code],
            dtype=object,
        ),
        "firm_in_england": np.isin(region_code, list(UK_FIRM_ENGLISH_REGION_CODES)),
        "firm_is_public_body": np.isin(
            status, list(UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES)
        ),
    }


def _validate_attribute_columns(df: pd.DataFrame) -> None:
    status, region_code = _ons_attributes(df)
    for column in ("firm_in_england", "firm_is_public_body"):
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"UK firm {column} must be boolean.")
    if not df["firm_legal_form"].isin(UK_FIRM_LEGAL_FORMS).all():
        raise ValueError("UK firm firm_legal_form must be FirmLegalForm member names.")
    if not df["firm_region"].isin(list(UK_FIRM_REGION_CODE_TO_REGION.values())).all():
        raise ValueError("UK firm firm_region must be Region member names.")
    expected = _attribute_columns(status, region_code)
    for column, values in expected.items():
        if not np.array_equal(df[column].to_numpy(values.dtype), values):
            source = (
                "ons_region_code"
                if column in ("firm_region", "firm_in_england")
                else "ons_legal_status"
            )
            raise ValueError(f"UK firm {column} does not follow {source}.")


def _attribute_group_columns(ordered: pd.DataFrame) -> dict[str, np.ndarray]:
    status, region_code = _ons_attributes(ordered)
    return {
        **_attribute_columns(status, region_code),
        "ons_legal_status": status,
        "ons_region_code": region_code,
    }


def _paybill_group_columns(ordered: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"firm_paybill": ordered["firm_paybill"].to_numpy(np.float64)}


def _employment_allowance_group_columns(ordered: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: ordered[column].to_numpy(bool)
        for column in UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS
    }


def _validate_employment_allowance_columns(df: pd.DataFrame) -> None:
    for column in UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS:
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"UK firm {column} must be boolean.")
    expected = uk_firm_employment_allowance_flags(
        df["ons_legal_status"].to_numpy(np.int64),
        df["firm_employees"].to_numpy(np.int64),
    )
    for column, values in expected.items():
        if not np.array_equal(df[column].to_numpy(bool), values):
            raise ValueError(
                f"UK firm {column} does not follow ons_legal_status and firm_employees."
            )


def _single_column(column: str) -> Callable[[pd.DataFrame], dict[str, np.ndarray]]:
    def columns(ordered: pd.DataFrame) -> dict[str, np.ndarray]:
        return {column: ordered[column].to_numpy(np.float64)}

    return columns


def _non_negative(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_numpy(np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"UK firm {column} must be finite and non-negative.")
    return values


def _validate_profit_columns(df: pd.DataFrame) -> None:
    profit = _non_negative(df, "firm_taxable_trading_profit")
    not_company = df["ons_legal_status"].to_numpy(np.int64) != 1
    if np.any(profit[not_company]):
        raise ValueError(
            "UK firm firm_taxable_trading_profit must be zero for firms that "
            "are not companies."
        )


def _validate_rateable_value_columns(df: pd.DataFrame) -> None:
    value = _non_negative(df, "firm_rateable_value")
    if np.any(value[~df["firm_in_england"].to_numpy(bool)]):
        raise ValueError("UK firm firm_rateable_value must be zero outside England.")


def _validate_paybill_columns(df: pd.DataFrame) -> None:
    for column in UK_FIRM_PAYBILL_POLICYENGINE_UK_COLUMNS:
        values = df[column].to_numpy(np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"UK firm {column} must be finite and non-negative.")
    no_employees = df["firm_employees"].to_numpy() == 0
    if np.any(df["firm_paybill"].to_numpy(np.float64)[no_employees]):
        raise ValueError("UK firm firm_paybill must be zero without employees.")


@dataclass(frozen=True)
class UKFirmExportGroup:
    """An optional input group of the policyengine-uk firm export.

    ``source_columns`` are the generator rows' columns that switch the group
    on: the export emits it when the rows carry all of them and raises when
    they carry only some. ``requires`` names the groups it extends, which
    must be emitted too. ``policyengine_uk_columns`` and
    ``microcosm_columns`` are the columns it adds, in order.
    ``columns(ordered)`` fills both from the rows ordered by ``firm_id``;
    ``validate(table)`` raises ``ValueError`` unless the table's columns of
    the group meet its contract.
    """

    name: str
    source_columns: tuple[str, ...]
    policyengine_uk_columns: tuple[str, ...]
    microcosm_columns: tuple[str, ...]
    requires: tuple[str, ...]
    columns: Callable[[pd.DataFrame], Mapping[str, np.ndarray]]
    validate: Callable[[pd.DataFrame], None]


#: The export's optional input groups, in column order. A group appears
#: after every group it requires.
UK_FIRM_EXPORT_GROUPS: tuple[UKFirmExportGroup, ...] = (
    UKFirmExportGroup(
        name="attributes",
        source_columns=UK_FIRM_ONS_ATTRIBUTE_COLUMNS,
        policyengine_uk_columns=UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS,
        microcosm_columns=UK_FIRM_ONS_ATTRIBUTE_COLUMNS,
        requires=(),
        columns=_attribute_group_columns,
        validate=_validate_attribute_columns,
    ),
    UKFirmExportGroup(
        name="paybill",
        source_columns=("firm_paybill",),
        policyengine_uk_columns=UK_FIRM_PAYBILL_POLICYENGINE_UK_COLUMNS,
        microcosm_columns=(),
        requires=("attributes",),
        columns=_paybill_group_columns,
        validate=_validate_paybill_columns,
    ),
    UKFirmExportGroup(
        name="employment_allowance",
        source_columns=UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
        policyengine_uk_columns=UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
        microcosm_columns=(),
        requires=("attributes",),
        columns=_employment_allowance_group_columns,
        validate=_validate_employment_allowance_columns,
    ),
    UKFirmExportGroup(
        name="profit",
        source_columns=("firm_taxable_trading_profit",),
        policyengine_uk_columns=("firm_taxable_trading_profit",),
        microcosm_columns=(),
        requires=("attributes",),
        columns=_single_column("firm_taxable_trading_profit"),
        validate=_validate_profit_columns,
    ),
    UKFirmExportGroup(
        name="rateable_value",
        source_columns=("firm_rateable_value",),
        policyengine_uk_columns=("firm_rateable_value",),
        microcosm_columns=(),
        requires=("attributes",),
        columns=_single_column("firm_rateable_value"),
        validate=_validate_rateable_value_columns,
    ),
)

_GROUPS_BY_NAME: Mapping[str, UKFirmExportGroup] = MappingProxyType(
    {group.name: group for group in UK_FIRM_EXPORT_GROUPS}
)


def _check_export_groups() -> None:
    seen: set[str] = set()
    columns = list(UK_FIRM_TABLE_COLUMNS)
    for group in UK_FIRM_EXPORT_GROUPS:
        if group.name in seen:
            raise ValueError(f"UK firm export group {group.name} repeats.")
        unknown = [name for name in group.requires if name not in seen]
        if unknown:
            raise ValueError(
                f"UK firm export group {group.name} requires {unknown}, which "
                "must come before it."
            )
        seen.add(group.name)
        columns += [*group.policyengine_uk_columns, *group.microcosm_columns]
    if len(set(columns)) != len(columns):
        raise ValueError("UK firm export groups repeat a column.")


_check_export_groups()


def _ordered_groups(groups: Iterable[str]) -> tuple[UKFirmExportGroup, ...]:
    names = set(groups)
    unknown = sorted(names - set(_GROUPS_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown UK firm export groups {unknown}.")
    ordered = tuple(g for g in UK_FIRM_EXPORT_GROUPS if g.name in names)
    for group in ordered:
        missing = [name for name in group.requires if name not in names]
        if missing:
            raise ValueError(
                f"UK firm export group {group.name} extends {missing}, which "
                "are not included."
            )
    return ordered


def uk_firm_policyengine_uk_columns(groups: Iterable[str] = ()) -> tuple[str, ...]:
    """policyengine-uk columns of the export with ``groups``, in order."""

    return (
        *UK_FIRM_POLICYENGINE_UK_COLUMNS,
        *(
            column
            for group in _ordered_groups(groups)
            for column in group.policyengine_uk_columns
        ),
    )


def uk_firm_table_columns(groups: Iterable[str] = ()) -> tuple[str, ...]:
    """The export's columns with ``groups``, in order.

    The base and each group's policyengine-uk columns, then the base and
    each group's Microcosm-only columns, groups in registry order. A group
    without a group it requires raises ``ValueError``.
    """

    names = tuple(groups)
    ordered = _ordered_groups(names)
    return (
        *uk_firm_policyengine_uk_columns(names),
        *UK_FIRM_MICROCOSM_COLUMNS,
        *(column for group in ordered for column in group.microcosm_columns),
    )


def _layouts() -> Mapping[tuple[str, ...], tuple[str, ...]]:
    layouts: dict[tuple[str, ...], tuple[str, ...]] = {}
    names = [group.name for group in UK_FIRM_EXPORT_GROUPS]
    for size in range(len(names) + 1):
        for combination in itertools.combinations(names, size):
            chosen = set(combination)
            if all(
                set(_GROUPS_BY_NAME[name].requires) <= chosen for name in combination
            ):
                layouts[combination] = uk_firm_table_columns(combination)
    return MappingProxyType(layouts)


#: Every valid set of export groups (names in registry order) -> its layout.
UK_FIRM_TABLE_LAYOUTS: Mapping[tuple[str, ...], tuple[str, ...]] = _layouts()


def uk_firm_export_groups(firms: pd.DataFrame) -> tuple[str, ...]:
    """Names of the export groups whose source columns ``firms`` carries.

    Raises ``ValueError`` when ``firms`` carries some of a group's source
    columns but not all, or a group without one it extends.
    """

    present: list[str] = []
    for group in UK_FIRM_EXPORT_GROUPS:
        carried = [c for c in group.source_columns if c in firms.columns]
        if not carried:
            continue
        if len(carried) != len(group.source_columns):
            needs = "both" if len(group.source_columns) == 2 else "all of them"
            raise ValueError(
                f"UK firm population carries {carried} without the rest of "
                f"{list(group.source_columns)}; the {group.name} export needs "
                f"{needs}."
            )
        for name in group.requires:
            if name not in present:
                raise ValueError(
                    f"UK firm population carries {', '.join(carried)} without "
                    f"{list(_GROUPS_BY_NAME[name].source_columns)}; the "
                    f"{group.name} export extends the {name} one."
                )
        present.append(group.name)
    return tuple(present)


#: policyengine-uk firm input variables in the extended export (base and
#: attributes), in order.
UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS = uk_firm_policyengine_uk_columns(
    ("attributes",)
)

#: Microcosm-only columns the extended export carries after the
#: policyengine-uk ones: the base ones and the ONS attribute draws.
UK_FIRM_MICROCOSM_ONLY_COLUMNS = (
    *UK_FIRM_MICROCOSM_COLUMNS,
    *_GROUPS_BY_NAME["attributes"].microcosm_columns,
)

#: The extended export's columns (base and attributes), in order.
UK_FIRM_EXTENDED_TABLE_COLUMNS = uk_firm_table_columns(("attributes",))

#: policyengine-uk firm input variables in the paybill export (base,
#: attributes and paybill), in order.
UK_FIRM_PAYBILL_EXTENDED_POLICYENGINE_UK_COLUMNS = uk_firm_policyengine_uk_columns(
    ("attributes", "paybill")
)

#: The paybill export's columns (base, attributes and paybill), in order.
UK_FIRM_PAYBILL_TABLE_COLUMNS = uk_firm_table_columns(("attributes", "paybill"))


def uk_firm_sic_sections(sic_code: pd.Series | np.ndarray) -> np.ndarray:
    """policyengine-uk ``SICSection`` member names for two-digit SIC divisions.

    ``sic_code`` holds a SIC 2007 division (01-99) as an integer or as the
    generator's zero-padded string (``"00011"`` is division 11; it is not a
    five-digit SIC class). A division in no section gives ``"UNKNOWN"``; a
    value outside 0-99 or not a whole number raises ``ValueError``.
    """

    raw = pd.Series(np.asarray(sic_code, dtype=object))
    try:
        numeric = pd.to_numeric(raw, errors="raise").to_numpy(np.float64)
    except (TypeError, ValueError):
        raise ValueError("UK firm sic_code must hold SIC 2007 divisions.") from None
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric != np.round(numeric))
        or np.any(numeric < 0)
        or np.any(numeric > 99)
    ):
        raise ValueError(
            "UK firm sic_code must hold two-digit SIC 2007 divisions (0-99)."
        )
    sections = uk_sic2007_sections(numeric.astype(np.int64))
    sections[sections == ""] = UK_FIRM_SIC_SECTION_UNKNOWN
    return sections


def build_candidate_rules_table(firms: pd.DataFrame) -> pd.DataFrame:
    """The pre-calibration candidate table the VAT rules evaluate.

    Reads ``UK_FIRM_CANDIDATE_SOURCE_COLUMNS`` (thousands of pounds for
    turnover and input costs) and returns ``UK_FIRM_CANDIDATE_RULES_COLUMNS``
    in pounds, ordered by ``firm_id``. All turnover is standard-rated and
    nothing is exempt, so the rules' liability is the liability if
    registered; VAT scope and registration come later.
    """

    _require(firms, UK_FIRM_CANDIDATE_SOURCE_COLUMNS, "UK firm candidates")
    firm_id = _firm_ids(firms)
    turnover = _gbp(firms, "annual_turnover_k")
    input_costs = _gbp(firms, "annual_input_k")
    table = pd.DataFrame(
        {
            "firm_id": firm_id,
            "firm_turnover": turnover,
            "firm_input_costs": input_costs,
            "firm_employees": _employees(firms),
            "firm_sic_section": uk_firm_sic_sections(firms["sic_code"]),
            "firm_standard_rated_supplies": turnover,
            "firm_exempt_supplies": np.zeros_like(turnover),
            "firm_input_vat": ASSUMED_INPUT_VAT_RATE * input_costs,
        },
        columns=list(UK_FIRM_CANDIDATE_RULES_COLUMNS),
    )
    return table.sort_values("firm_id", kind="stable").reset_index(drop=True)


def to_policyengine_uk_firm_table(
    firms: pd.DataFrame,
    *,
    threshold_k: float,
) -> pd.DataFrame:
    """The calibrated firm table in policyengine-uk names and pounds.

    Reads ``UK_FIRM_EXPORT_SOURCE_COLUMNS`` and returns
    ``uk_firm_table_columns`` of the ``UK_FIRM_EXPORT_GROUPS`` whose source
    columns ``firms`` carries (``uk_firm_export_groups``), ordered by
    ``firm_id``: ``UK_FIRM_TABLE_COLUMNS`` with none,
    ``UK_FIRM_EXTENDED_TABLE_COLUMNS`` with both
    ``UK_FIRM_ONS_ATTRIBUTE_COLUMNS`` and ``UK_FIRM_PAYBILL_TABLE_COLUMNS``
    when attributed ``firms`` also carry ``firm_paybill`` (pounds a year,
    from ``firm_paybill.assign_uk_firm_paybill``). Some of a group's source
    columns without the rest (one ONS attribute column without the other),
    or a group without one it extends (``firm_paybill`` without the ONS
    attribute columns), raises ``ValueError``. Columns:

    - turnover and input costs are the generator's thousands of pounds times
      1000;
    - a firm in VAT scope has all turnover standard-rated, one out of scope
      has it all exempt;
    - ``firm_vat_voluntarily_registered`` is a registered firm whose turnover
      does not exceed ``threshold_k`` (thousands of pounds);
    - ``firm_input_vat`` is ``ASSUMED_INPUT_VAT_RATE`` times input costs for
      every firm; policyengine-uk's partial exemption share restricts its
      recovery;
    - extended only: ``firm_legal_form`` maps ``ons_legal_status`` through
      ``UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM``, ``firm_region`` maps
      ``ons_region_code`` through ``UK_FIRM_REGION_CODE_TO_REGION``,
      ``firm_in_england`` is an English region and ``firm_is_public_body`` an
      ONS status in ``UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES``. A code outside
      those maps raises: nothing falls back to policyengine-uk's London or
      company defaults;
    - paybill only: ``firm_paybill`` is carried from the row unchanged.

    The result passes ``validate_uk_firm_table``.
    """

    _require(firms, UK_FIRM_EXPORT_SOURCE_COLUMNS, "UK firm population")
    groups = _ordered_groups(uk_firm_export_groups(firms))
    if not np.isfinite(threshold_k) or threshold_k <= 0:
        raise ValueError("UK firm threshold_k must be positive.")
    for column in ("vat_scope", "vat_registered", "in_frame"):
        # A missing value would otherwise read as True.
        if not pd.api.types.is_bool_dtype(firms[column]):
            raise ValueError(f"UK firm {column} must be boolean.")
    candidates = build_candidate_rules_table(firms)
    ordered = firms.assign(_firm_id=_firm_ids(firms)).sort_values(
        "_firm_id", kind="stable"
    )
    scope = ordered["vat_scope"].to_numpy(bool)
    registered = ordered["vat_registered"].to_numpy(bool)
    turnover_k = ordered["annual_turnover_k"].to_numpy(np.float64)
    turnover = candidates["firm_turnover"].to_numpy()
    table = pd.DataFrame(
        {
            "firm_id": candidates["firm_id"].to_numpy(),
            "firm_weight": ordered["firm_weight"].to_numpy(np.float64),
            "firm_turnover": turnover,
            "firm_input_costs": candidates["firm_input_costs"].to_numpy(),
            "firm_employees": candidates["firm_employees"].to_numpy(),
            "firm_sic_section": candidates["firm_sic_section"].to_numpy(),
            "firm_standard_rated_supplies": np.where(scope, turnover, 0.0),
            "firm_exempt_supplies": np.where(scope, 0.0, turnover),
            "firm_vat_voluntarily_registered": registered & (turnover_k <= threshold_k),
            "firm_input_vat": candidates["firm_input_vat"].to_numpy(),
            "in_frame": ordered["in_frame"].to_numpy(bool),
            "vat_scope": scope,
            "vat_registered": registered,
            "base_weight": ordered["base_weight"].to_numpy(np.float64),
            "data_vintage": ordered["data_vintage"].astype(str).to_numpy(),
            "source_firm_id": ordered["source_firm_id"].to_numpy(np.int64),
            "source_firm_key": ordered["source_firm_key"].astype(str).to_numpy(),
        },
        columns=list(UK_FIRM_TABLE_COLUMNS),
    )
    if groups:
        added: dict[str, np.ndarray] = {}
        for group in groups:
            added.update(group.columns(ordered))
        layout = uk_firm_table_columns([group.name for group in groups])
        table = table.assign(**added)[list(layout)]
    validate_uk_firm_table(table)
    return table


def validate_uk_firm_table(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``df`` meets the UK firm table contract.

    The contract: exactly one of ``UK_FIRM_TABLE_LAYOUTS`` (the base table
    and each valid set of ``UK_FIRM_EXPORT_GROUPS``), in order; no missing
    value in any column (policyengine-uk rejects NaN); integer ``firm_id``,
    unique and ascending; finite non-negative weights and money;
    non-negative integer employees; ``SICSection`` member names; boolean
    flags; supplies that split turnover by VAT scope; voluntary registration
    only for registered firms; and each present group's own checks. The
    attribute group needs ONS legal status codes, GSS region codes, and
    ``firm_legal_form``, ``firm_region``, ``firm_in_england`` and
    ``firm_is_public_body`` as those codes give them; the paybill group a
    finite non-negative ``firm_paybill`` that is zero for a firm without
    employees.
    """

    columns = tuple(df.columns)
    groups = next(
        (names for names, layout in UK_FIRM_TABLE_LAYOUTS.items() if layout == columns),
        None,
    )
    if groups is None:
        raise ValueError(
            f"UK firm table columns {list(df.columns)} are not one of "
            f"{[list(layout) for layout in UK_FIRM_TABLE_LAYOUTS.values()]}."
        )
    missing = [column for column in columns if df[column].isna().any()]
    if missing:
        raise ValueError(f"UK firm table has missing values in {missing}.")
    firm_id = _firm_ids(df)
    if np.any(np.diff(firm_id) < 0):
        raise ValueError("UK firm table must be ordered by firm_id.")
    for column in _NON_NEGATIVE_COLUMNS:
        values = df[column].to_numpy(np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"UK firm {column} must be finite and non-negative.")
    employees = df["firm_employees"]
    if not pd.api.types.is_integer_dtype(employees) or (employees < 0).any():
        raise ValueError("UK firm firm_employees must be non-negative integers.")
    if not df["firm_sic_section"].isin(UK_FIRM_SIC_SECTIONS).all():
        raise ValueError("UK firm firm_sic_section must be SICSection member names.")
    for column in _BOOL_COLUMNS:
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"UK firm {column} must be boolean.")
    turnover = df["firm_turnover"].to_numpy(np.float64)
    standard = df["firm_standard_rated_supplies"].to_numpy(np.float64)
    exempt = df["firm_exempt_supplies"].to_numpy(np.float64)
    scope = df["vat_scope"].to_numpy(bool)
    if not np.allclose(standard + exempt, turnover, rtol=1e-9, atol=1e-6):
        raise ValueError("UK firm supplies must add up to turnover.")
    if np.any(exempt[scope] != 0) or np.any(standard[~scope] != 0):
        raise ValueError("UK firm supplies must follow vat_scope.")
    voluntary = df["firm_vat_voluntarily_registered"].to_numpy(bool)
    if np.any(voluntary & ~df["vat_registered"].to_numpy(bool)):
        raise ValueError("UK firm voluntary registration needs vat_registered.")
    for group in _ordered_groups(groups):
        group.validate(df)


__all__ = [
    "ASSUMED_INPUT_VAT_RATE",
    "UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_CANDIDATE_RULES_COLUMNS",
    "UK_FIRM_CANDIDATE_SOURCE_COLUMNS",
    "UK_FIRM_ENGLISH_REGION_CODES",
    "UK_FIRM_EXPORT_GROUPS",
    "UK_FIRM_EXPORT_SOURCE_COLUMNS",
    "UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_EXTENDED_TABLE_COLUMNS",
    "UK_FIRM_INPUT_STATUSES",
    "UK_FIRM_LEGAL_FORMS",
    "UK_FIRM_MICROCOSM_COLUMNS",
    "UK_FIRM_MICROCOSM_ONLY_COLUMNS",
    "UK_FIRM_ONS_ATTRIBUTE_COLUMNS",
    "UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM",
    "UK_FIRM_PAYBILL_EXTENDED_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_PAYBILL_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_PAYBILL_TABLE_COLUMNS",
    "UK_FIRM_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_POLICYENGINE_UK_INPUT_STATUS",
    "UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES",
    "UK_FIRM_REGION_CODE_TO_REGION",
    "UK_FIRM_SIC_SECTIONS",
    "UK_FIRM_SIC_SECTION_UNKNOWN",
    "UK_FIRM_TABLE_COLUMNS",
    "UK_FIRM_TABLE_LAYOUTS",
    "UKFirmExportGroup",
    "build_candidate_rules_table",
    "to_policyengine_uk_firm_table",
    "uk_firm_export_groups",
    "uk_firm_policyengine_uk_columns",
    "uk_firm_sic_sections",
    "uk_firm_table_columns",
    "validate_uk_firm_table",
]
