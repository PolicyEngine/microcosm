"""AI-exposure scores for UK SOC 2020 occupation codes.

This module ships a pre-built crosswalk
(``ai_exposure_data/uk_soc2020_ai_exposure.csv``) that attaches occupation-level
AI-exposure measures to every SOC 2020 unit group (4-digit), plus an aggregated
major-group (1-digit) table weighted by ASHE employment. Lookups fall back from
4-digit to 3-digit to 2-digit parent means so that partially coded microdata
still receives a score.

Measures
--------
``c_aioe`` (primary)
    Complementarity-adjusted AI occupational exposure,
    ``AIOE * (1 - (theta - theta_min))``, following Pizzinelli et al. (2023,
    IMF WP/23/216) and its UK/Ireland application in Williamson et al. (2025,
    J Labour Market Res 59:30). AIOE is Felten, Raj and Seamans (2021).
``complementarity`` / ``complementarity_theta``
    The AI complementarity score theta in [0, 1], reconstructed from the recipe
    published in IMF WP/23/216 (11 O*NET work contexts plus job zones grouped
    into six components) using O*NET 27.3 data. The reconstruction reproduces
    the paper's published extremes exactly (minimum at US SOC 51-9031, maximum
    at US SOC 29-1022) with a level offset of about +0.03-0.04 attributable to
    the O*NET vintage; ``theta_min`` is taken from the reconstruction for
    internal consistency. Occupation-level theta is not otherwise openly
    published (the IMF and IGEES datasets are available on request only).
``dsit_aioe`` / ``dsit_llm``
    Standardised AIOE and LLM-only exposure on UK SOC 2010 4-digit codes from
    the DSIT/DfE report "The impact of AI on UK jobs and training" (Nov 2023),
    Annex 1, mapped SOC 2010 -> SOC 2020 with the ONS coding index. These are
    the official UK mappings of the Felten scores.
``eloundou_beta``
    Eloundou et al. (2023) "GPTs are GPTs" human-annotated beta exposure
    (share of tasks where LLMs can halve completion time, direct or with
    tooling), in [0, 1].
``felten_aioe``
    Felten, Raj and Seamans (2021) standardised AIOE.

Crosswalk construction
----------------------
US-based measures are chained US SOC 2018 -> US SOC 2010 (BLS crosswalk,
Nov 2017) -> ISCO-08 (BLS ISCO-08/SOC 2010 crosswalk, 2012) -> SOC 2020 (ONS
SOC 2020 Volume 2 coding index, which assigns an ISCO-08 code to every SOC 2020
index entry). Many-to-many links are resolved with employment-unweighted means
and flagged ``chained`` in ``mapping_quality``; ``direct`` marks single-path
links and ``imputed-from-parent`` marks unit groups filled from 3-digit or
2-digit sibling means. The major-group table weights unit groups by ASHE
Table 14 (2021, provisional) employee-job counts on SOC 2020 4-digit codes.

Sources and licences
--------------------
- Eloundou et al. (2023), github.com/openai/GPTs-are-GPTs (MIT licence).
- Felten, Raj and Seamans (2021), github.com/AIOE-Data/AIOE (public GitHub
  release accompanying the paper).
- Pizzinelli et al. (2023), IMF Working Paper WP/23/216 (methodology);
  O*NET 27.3 database, US Department of Labor (CC BY 4.0).
- DSIT/DfE (2023), "The impact of AI on UK jobs and training", Annex 1,
  gov.uk (Open Government Licence v3.0).
- ONS SOC 2020 Volumes 1-2 and ASHE Table 14 (Open Government Licence v3.0).
- BLS SOC crosswalks (US public domain).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd

MEASURE_TO_COLUMN = {
    "c_aioe": "c_aioe",
    "complementarity": "complementarity_theta",
    "complementarity_theta": "complementarity_theta",
    "dsit_aioe": "dsit_aioe",
    "dsit_llm": "dsit_llm",
    "eloundou_beta": "eloundou_beta",
    "felten_aioe": "felten_aioe",
}
MAPPING_QUALITIES = ("direct", "chained", "imputed-from-parent")

_DATA_DIR = "ai_exposure_data"
_UNIT_GROUP_FILE = "uk_soc2020_ai_exposure.csv"
_MAJOR_GROUP_FILE = "uk_soc2020_major_group_ai_exposure.csv"


def _data_path(filename: str):
    return importlib_resources.files("populace.build.uk_runtime").joinpath(
        _DATA_DIR, filename
    )


@lru_cache(maxsize=1)
def load_ai_exposure_table() -> pd.DataFrame:
    """SOC 2020 unit-group (4-digit) AI-exposure crosswalk."""

    with importlib_resources.as_file(_data_path(_UNIT_GROUP_FILE)) as path:
        table = pd.read_csv(path, dtype={"soc2020_code": str})
    return table.set_index("soc2020_code")


@lru_cache(maxsize=1)
def load_major_group_ai_exposure_table() -> pd.DataFrame:
    """SOC 2020 major-group (1-digit) employment-weighted AI-exposure table."""

    with importlib_resources.as_file(_data_path(_MAJOR_GROUP_FILE)) as path:
        table = pd.read_csv(path, dtype={"soc2020_major_group": str})
    return table.set_index("soc2020_major_group")


def _measure_column(measure: str) -> str:
    if measure not in MEASURE_TO_COLUMN:
        raise ValueError(
            f"Unknown measure {measure!r}; expected one of {sorted(MEASURE_TO_COLUMN)}"
        )
    return MEASURE_TO_COLUMN[measure]


def _normalise_codes(codes) -> np.ndarray:
    values = pd.Series(np.asarray(codes, dtype=object)).astype("string")
    values = values.str.strip().str.replace(r"\.0$", "", regex=True)
    values = values.where(values.str.fullmatch(r"\d{1,4}"))
    return values.to_numpy(dtype=object)


@lru_cache(maxsize=len(MEASURE_TO_COLUMN) * 4)
def _prefix_means(column: str, length: int) -> dict[str, float]:
    table = load_ai_exposure_table()
    grouped = table[column].groupby(table.index.str[:length]).mean()
    return grouped.to_dict()


def exposure_for_soc(codes, measure: str = "c_aioe") -> np.ndarray:
    """AI-exposure scores for an array of SOC 2020 codes.

    Parameters
    ----------
    codes:
        Array-like of SOC 2020 codes (strings or integers). 4-digit unit
        groups are matched exactly; 3-digit and 2-digit codes (and 4-digit
        codes missing from the table) resolve to unweighted means over the
        matching unit groups, falling back 4-digit -> 3-digit -> 2-digit.
    measure:
        One of ``MEASURE_TO_COLUMN`` keys, e.g. ``"c_aioe"``,
        ``"complementarity"``, ``"eloundou_beta"``, ``"felten_aioe"``,
        ``"dsit_aioe"``, ``"dsit_llm"``.

    Returns
    -------
    numpy.ndarray of float, NaN where no code (or parent code) matches; a
    warning lists unmatched codes.
    """

    column = _measure_column(measure)
    table = load_ai_exposure_table()
    exact = table[column].to_dict()
    normalised = _normalise_codes(codes)

    result = np.full(len(normalised), np.nan, dtype=float)
    unmatched: list[str] = []
    for position, code in enumerate(normalised):
        if code is None or code is pd.NA:
            unmatched.append(str(np.asarray(codes, dtype=object)[position]))
            continue
        value = exact.get(code) if len(code) == 4 else None
        if value is None:
            for length in (3, 2):
                if len(code) < length:
                    continue
                value = _prefix_means(column, length).get(code[:length])
                if value is not None:
                    break
        if value is None or np.isnan(value):
            unmatched.append(code)
        else:
            result[position] = value
    if unmatched:
        warnings.warn(
            "No SOC 2020 AI-exposure match (returning NaN) for codes: "
            + ", ".join(sorted(set(unmatched))),
            stacklevel=2,
        )
    return result


def exposure_for_major_group(groups, measure: str = "c_aioe") -> np.ndarray:
    """Employment-weighted AI-exposure scores for SOC 2020 major groups (1-9).

    Uses the shipped major-group table (ASHE Table 14 2021 employee-job
    weights). Unknown groups return NaN with a warning.
    """

    column = _measure_column(measure)
    table = load_major_group_ai_exposure_table()
    lookup = table[column].to_dict()
    normalised = _normalise_codes(groups)

    result = np.full(len(normalised), np.nan, dtype=float)
    unmatched: list[str] = []
    for position, group in enumerate(normalised):
        value = None if group is None or group is pd.NA else lookup.get(group)
        if value is None or np.isnan(value):
            unmatched.append(str(np.asarray(groups, dtype=object)[position]))
        else:
            result[position] = value
    if unmatched:
        warnings.warn(
            "No SOC 2020 major-group AI-exposure match (returning NaN) for: "
            + ", ".join(sorted(set(unmatched))),
            stacklevel=2,
        )
    return result
