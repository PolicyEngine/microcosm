"""UK firm tables in policyengine-uk firm variable names, in pounds.

Two views of the experimental UK firm generator's rows:

- ``build_candidate_rules_table`` is the pre-calibration table the VAT rules
  (``firm_rules.UKFirmRules``) evaluate: no weight, no VAT scope and no
  registration, with all turnover standard-rated so the rules return each
  firm's liability if registered.
- ``to_policyengine_uk_firm_table`` is the calibrated export, with
  policyengine-uk's firm inputs followed by the Microcosm-only columns
  (``UK_FIRM_TABLE_COLUMNS``), checked by ``validate_uk_firm_table``. When
  the rows carry the ONS attribute draws (``ons_legal_status`` and
  ``ons_region_code``), it is the extended table
  (``UK_FIRM_EXTENDED_TABLE_COLUMNS``), which adds ``firm_legal_form``,
  ``firm_region``, ``firm_in_england`` and ``firm_is_public_body``.

``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS`` records, for every formula-less
policyengine-uk firm input, whether the extended table sets it. The base table
leaves ``firm_region`` and ``firm_legal_form`` unset, so policyengine-uk's
defaults (London, company) apply; ``firm_paybill`` and the other money inputs
are never set.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd

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

#: policyengine-uk firm input variables in the extended export, in order.
UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS = (
    *UK_FIRM_POLICYENGINE_UK_COLUMNS,
    *UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS,
)

#: Microcosm-only columns the extended export carries after the
#: policyengine-uk ones. None is a policyengine-uk variable name (``region``
#: is a household variable there, so the region code is ``ons_region_code``).
UK_FIRM_MICROCOSM_ONLY_COLUMNS = (
    *UK_FIRM_MICROCOSM_COLUMNS,
    *UK_FIRM_ONS_ATTRIBUTE_COLUMNS,
)

#: The extended export's columns, in order.
UK_FIRM_EXTENDED_TABLE_COLUMNS = (
    *UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS,
    *UK_FIRM_MICROCOSM_ONLY_COLUMNS,
)

#: Values of ``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS``:
#:
#: - ``set``: mapped directly from a generator column or ONS draw;
#: - ``derived``: computed from other columns by a rule;
#: - ``assumed``: carries a data assumption beyond the source;
#: - ``not_set``: not exported; policyengine-uk's default applies.
UK_FIRM_INPUT_STATUSES = ("set", "derived", "assumed", "not_set")

#: Every formula-less policyengine-uk firm input (branch firm-entity-engine,
#: policyengine_uk/variables/firm/) -> how the extended export fills it.
#: The keys with status other than ``not_set`` are exactly
#: ``UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS``.
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
        # registration threshold; England from the region code.
        "firm_standard_rated_supplies": "derived",
        "firm_exempt_supplies": "derived",
        "firm_vat_voluntarily_registered": "derived",
        "firm_in_england": "derived",
        # ASSUMED_INPUT_VAT_RATE on all input costs; ONS statuses 4-7 to
        # OTHER; ONS statuses 5 and 6 as public authorities.
        "firm_input_vat": "assumed",
        "firm_legal_form": "assumed",
        "firm_is_public_body": "assumed",
        # Not drawn: no source yet. Placeholders would flow into NIC, CT,
        # VAT and business rates revenue.
        "firm_paybill": "not_set",
        "firm_taxable_trading_profit": "not_set",
        "firm_rateable_value": "not_set",
        "firm_n_associated_companies": "not_set",
        "firm_is_bank": "not_set",
        "firm_is_charity": "not_set",
        "firm_is_single_director_company": "not_set",
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
        "firm_deductions_allowance_share": "not_set",
        "firm_exempt_distributions": "not_set",
        "firm_losses_brought_forward": "not_set",
        "firm_non_trading_profits": "not_set",
        # Employer costs.
        "firm_apprenticeship_levy_allowance_share": "not_set",
        "firm_claims_employment_allowance": "not_set",
        "firm_zero_rated_class_1_earnings": "not_set",
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
    ``UK_FIRM_TABLE_COLUMNS``, ordered by ``firm_id``. When ``firms`` also
    carries both ``UK_FIRM_ONS_ATTRIBUTE_COLUMNS`` it returns
    ``UK_FIRM_EXTENDED_TABLE_COLUMNS``; one without the other raises
    ``ValueError``. Columns:

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
      company defaults.

    The result passes ``validate_uk_firm_table``.
    """

    _require(firms, UK_FIRM_EXPORT_SOURCE_COLUMNS, "UK firm population")
    present = [c for c in UK_FIRM_ONS_ATTRIBUTE_COLUMNS if c in firms.columns]
    if present and len(present) != len(UK_FIRM_ONS_ATTRIBUTE_COLUMNS):
        raise ValueError(
            f"UK firm population carries {present} without the rest of "
            f"{list(UK_FIRM_ONS_ATTRIBUTE_COLUMNS)}; the extended export needs "
            "both."
        )
    extended = bool(present)
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
    if extended:
        status, region_code = _ons_attributes(ordered)
        attributes = {
            **_attribute_columns(status, region_code),
            "ons_legal_status": status,
            "ons_region_code": region_code,
        }
        table = table.assign(**attributes)[list(UK_FIRM_EXTENDED_TABLE_COLUMNS)]
    validate_uk_firm_table(table)
    return table


def validate_uk_firm_table(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``df`` meets the UK firm table contract.

    The contract: exactly ``UK_FIRM_TABLE_COLUMNS`` or exactly
    ``UK_FIRM_EXTENDED_TABLE_COLUMNS``, in order; no missing value in any
    column (policyengine-uk rejects NaN); integer ``firm_id``, unique and
    ascending; finite non-negative weights and money; non-negative integer
    employees; ``SICSection`` member names; boolean flags; supplies that
    split turnover by VAT scope; and voluntary registration only for
    registered firms. The extended table also needs ONS legal status codes,
    GSS region codes, and ``firm_legal_form``, ``firm_region``,
    ``firm_in_england`` and ``firm_is_public_body`` as those codes give them.
    """

    columns = tuple(df.columns)
    if columns not in (UK_FIRM_TABLE_COLUMNS, UK_FIRM_EXTENDED_TABLE_COLUMNS):
        raise ValueError(
            f"UK firm table columns {list(df.columns)} are not "
            f"{list(UK_FIRM_TABLE_COLUMNS)} or "
            f"{list(UK_FIRM_EXTENDED_TABLE_COLUMNS)}."
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
    if columns == UK_FIRM_EXTENDED_TABLE_COLUMNS:
        _validate_attribute_columns(df)


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


__all__ = [
    "ASSUMED_INPUT_VAT_RATE",
    "UK_FIRM_ATTRIBUTE_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_CANDIDATE_RULES_COLUMNS",
    "UK_FIRM_CANDIDATE_SOURCE_COLUMNS",
    "UK_FIRM_ENGLISH_REGION_CODES",
    "UK_FIRM_EXPORT_SOURCE_COLUMNS",
    "UK_FIRM_EXTENDED_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_EXTENDED_TABLE_COLUMNS",
    "UK_FIRM_INPUT_STATUSES",
    "UK_FIRM_LEGAL_FORMS",
    "UK_FIRM_MICROCOSM_COLUMNS",
    "UK_FIRM_MICROCOSM_ONLY_COLUMNS",
    "UK_FIRM_ONS_ATTRIBUTE_COLUMNS",
    "UK_FIRM_ONS_LEGAL_STATUS_TO_LEGAL_FORM",
    "UK_FIRM_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_POLICYENGINE_UK_INPUT_STATUS",
    "UK_FIRM_PUBLIC_BODY_ONS_LEGAL_STATUSES",
    "UK_FIRM_REGION_CODE_TO_REGION",
    "UK_FIRM_SIC_SECTIONS",
    "UK_FIRM_SIC_SECTION_UNKNOWN",
    "UK_FIRM_TABLE_COLUMNS",
    "build_candidate_rules_table",
    "to_policyengine_uk_firm_table",
    "uk_firm_sic_sections",
    "validate_uk_firm_table",
]
