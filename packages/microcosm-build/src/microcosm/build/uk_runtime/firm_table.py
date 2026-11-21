"""UK firm tables in policyengine-uk firm variable names, in pounds.

Two views of the experimental UK firm generator's rows:

- ``build_candidate_rules_table`` is the pre-calibration table the VAT rules
  (``firm_rules.UKFirmRules``) evaluate: no weight, no VAT scope and no
  registration, with all turnover standard-rated so the rules return each
  firm's liability if registered.
- ``to_policyengine_uk_firm_table`` is the calibrated export, with
  policyengine-uk's firm inputs followed by the Microcosm-only columns
  (``UK_FIRM_TABLE_COLUMNS``), checked by ``validate_uk_firm_table``.

``firm_region``, ``firm_legal_form`` and ``firm_paybill`` are not set: the
generator draws none of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.occupation_constants import (
    UK_SIC2007_SECTION_DIVISIONS,
    uk_sic2007_sections,
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
    ``UK_FIRM_TABLE_COLUMNS``, ordered by ``firm_id``:

    - turnover and input costs are the generator's thousands of pounds times
      1000;
    - a firm in VAT scope has all turnover standard-rated, one out of scope
      has it all exempt;
    - ``firm_vat_voluntarily_registered`` is a registered firm whose turnover
      does not exceed ``threshold_k`` (thousands of pounds);
    - ``firm_input_vat`` is ``ASSUMED_INPUT_VAT_RATE`` times input costs for
      every firm; policyengine-uk's partial exemption share restricts its
      recovery.

    The result passes ``validate_uk_firm_table``.
    """

    _require(firms, UK_FIRM_EXPORT_SOURCE_COLUMNS, "UK firm population")
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
    validate_uk_firm_table(table)
    return table


def validate_uk_firm_table(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``df`` meets the UK firm table contract.

    The contract: exactly ``UK_FIRM_TABLE_COLUMNS`` in order; integer
    ``firm_id``, unique and ascending; finite non-negative weights and money;
    non-negative integer employees; ``SICSection`` member names; boolean flags;
    supplies that split turnover by VAT scope; and voluntary registration only
    for registered firms.
    """

    if tuple(df.columns) != UK_FIRM_TABLE_COLUMNS:
        raise ValueError(
            f"UK firm table columns {list(df.columns)} are not "
            f"{list(UK_FIRM_TABLE_COLUMNS)}."
        )
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


__all__ = [
    "ASSUMED_INPUT_VAT_RATE",
    "UK_FIRM_CANDIDATE_RULES_COLUMNS",
    "UK_FIRM_CANDIDATE_SOURCE_COLUMNS",
    "UK_FIRM_EXPORT_SOURCE_COLUMNS",
    "UK_FIRM_MICROCOSM_COLUMNS",
    "UK_FIRM_POLICYENGINE_UK_COLUMNS",
    "UK_FIRM_SIC_SECTIONS",
    "UK_FIRM_SIC_SECTION_UNKNOWN",
    "UK_FIRM_TABLE_COLUMNS",
    "build_candidate_rules_table",
    "to_policyengine_uk_firm_table",
    "uk_firm_sic_sections",
    "validate_uk_firm_table",
]
