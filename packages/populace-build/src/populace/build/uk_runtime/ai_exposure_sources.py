"""Official-source builders for the packaged UK AI-exposure tables.

The lookup module (:mod:`populace.build.uk_runtime.ai_exposure`) ships
pre-built CSVs; this module records where those CSVs come from and provides
``build_*`` functions that reconstruct them from pinned public sources,
following the pattern of :mod:`populace.build.uk_runtime.geography_sources`
(pinned source-URL constants plus composable builders).

Derivation chain (see the ai_exposure module docstring for citations):

1. ``build_complementarity_theta`` reconstructs the Pizzinelli et al. (2023,
   IMF WP/23/216) AI complementarity score theta from the O*NET 27.3
   database (11 work contexts plus job zones grouped into six components).
2. ``build_us_measure_chain`` chains US-SOC-based measures (Felten AIOE,
   Eloundou et al. beta, theta) US SOC 2018 -> US SOC 2010 -> ISCO-08 ->
   UK SOC 2020 via the BLS crosswalks and the ONS SOC 2020 Volume 2 coding
   index.
3. ``build_dsit_soc2010_to_soc2020`` maps the DSIT/DfE (Nov 2023) Annex 1
   scores from SOC 2010 to SOC 2020.
4. ``build_ai_exposure_table`` composes 1-3 into the packaged unit-group
   (4-digit) table, imputing missing unit groups from parent means.
5. ``build_major_group_ai_exposure`` aggregates the unit-group table to
   1-digit major groups, weighted by packaged ASHE Table 14 employee jobs.

Functions 1-4 download pinned sources at call time and are faithful
reconstructions of the documented derivation; steps whose exact
specification is not recoverable from the shipped documentation carry TODO
comments instead of silent guesses. Function 5 runs entirely from packaged
data and is used to regenerate ``uk_soc2020_major_group_ai_exposure.csv``.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.ai_exposure import (
    MEASURE_TO_COLUMN,
    load_ai_exposure_table,
)

# --------------------------------------------------------------------------
# Pinned source URLs
# --------------------------------------------------------------------------

#: O*NET 27.3 database (US Department of Labor, CC BY 4.0) — the vintage used
#: to reconstruct the IMF WP/23/216 complementarity score theta.
ONET_27_3_WORK_CONTEXT_URL = (
    "https://www.onetcenter.org/dl_files/database/db_27_3_text/Work%20Context.txt"
)
ONET_27_3_JOB_ZONES_URL = (
    "https://www.onetcenter.org/dl_files/database/db_27_3_text/Job%20Zones.txt"
)

#: BLS US SOC 2010 <-> 2018 crosswalk (Nov 2017 release, US public domain).
BLS_SOC2010_SOC2018_CROSSWALK_URL = (
    "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx"
)

#: BLS ISCO-08 <-> US SOC 2010 crosswalk (2012, US public domain).
BLS_ISCO08_SOC2010_CROSSWALK_URL = (
    "https://www.bls.gov/soc/soccrosswalks/isco_soc_crosswalk.xls"
)

#: ONS SOC 2020 Volume 2 coding index (OGL v3.0). Every coding-index entry
#: carries both a SOC 2020 unit group and an ISCO-08 code, and SOC 2010
#: codes, providing both the ISCO-08 -> SOC 2020 and SOC 2010 -> SOC 2020
#: links used below.
ONS_SOC2020_VOLUME2_CODING_INDEX_URL = (
    "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/"
    "standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrules"
    "andconventions/soc2020volume2thecodingindexandcodingrulesandconventions"
    "excel180523.xlsx"
)

#: DSIT/DfE (Nov 2023) "The impact of AI on UK jobs and training", Annex 1
#: (OGL v3.0): standardised AIOE and LLM exposure on SOC 2010 4-digit codes.
DSIT_2023_ANNEX1_URL = (
    "https://assets.publishing.service.gov.uk/media/6552862d046ed400148b99fe/"
    "Annex_A_The_Impact_of_AI_on_UK_Jobs_and_Training.xlsx"
)

#: Felten, Raj and Seamans (2021) AI Occupational Exposure, public GitHub
#: release accompanying the paper.
FELTEN_AIOE_URL = (
    "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Appendix%20file/"
    "Appendix%20A%20AIOE.csv"
)

#: Eloundou et al. (2023) "GPTs are GPTs" occupation-level exposure (MIT
#: licence); ``beta`` is the human-annotated share of tasks where LLMs can
#: halve completion time (direct or with tooling).
ELOUNDOU_GPTS_ARE_GPTS_URL = (
    "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/"
    "occ_lvl_exposure_data.csv"
)

#: IMF WP/23/216 published extremes of the complementarity score theta,
#: used to validate the O*NET reconstruction.
THETA_MIN_SOC2018 = "51-9031"  # Cutters and trimmers, hand.
THETA_MAX_SOC2018 = "29-1022"  # Oral and maxillofacial surgeons.

#: The six theta components of IMF WP/23/216: 11 O*NET Work Context elements
#: (CX scale, 1-5) plus the O*NET Job Zone (1-5), each normalised to [0, 1].
#: Elements listed with ``reverse=True`` enter as ``1 - value`` (more routine
#: or automated work is less AI-complementary).
#: TODO: The exact element list is reconstructed from the WP/23/216 annex
#: description ("11 work contexts plus job zones grouped into six
#: components"); verify each element name against the paper's annex table
#: before treating individual component scores as canonical. The composite
#: reproduces the published extremes (asserted below).
THETA_WORK_CONTEXT_COMPONENTS: dict[str, tuple[tuple[str, bool], ...]] = {
    "communication": (
        ("Public Speaking", False),
        ("Face-to-Face Discussions", False),
        ("Contact With Others", False),
    ),
    "responsibility": (
        ("Responsibility for Outcomes and Results", False),
        ("Responsible for Others' Health and Safety", False),
    ),
    "physical_conditions": (("Physical Proximity", False),),
    "criticality": (
        ("Consequence of Error", False),
        ("Impact of Decisions on Co-workers or Company Results", False),
        ("Frequency of Decision Making", False),
    ),
    "routine": (
        ("Degree of Automation", True),
        ("Structured versus Unstructured Work", False),
    ),
}

#: Packaged ASHE Table 14 tidy CSV used to weight the major-group table
#: (see populace.build.uk_runtime.occupation_targets).
PACKAGED_ASHE_TABLE14_CSV = "ashe_table14_2025_soc4.csv"
MAJOR_GROUP_WEIGHTING_LABEL = "ashe_2025_table14_jobs"

SOC2020_MAJOR_GROUP_TITLES = {
    "1": "Managers, Directors And Senior Officials",
    "2": "Professional Occupations",
    "3": "Associate Professional Occupations",
    "4": "Administrative And Secretarial Occupations",
    "5": "Skilled Trades Occupations",
    "6": "Caring, Leisure And Other Service Occupations",
    "7": "Sales And Customer Service Occupations",
    "8": "Process, Plant And Machine Operatives",
    "9": "Elementary Occupations",
}

MAJOR_GROUP_TABLE_COLUMNS = (
    "soc2020_major_group",
    "major_group_title",
    "c_aioe",
    "complementarity_theta",
    "felten_aioe",
    "eloundou_beta",
    "dsit_aioe",
    "dsit_llm",
    "weighting",
    "employment_jobs_thousands",
    "n_unit_groups",
    "n_unit_groups_weighted",
)


# --------------------------------------------------------------------------
# 1. Complementarity theta from O*NET 27.3 (IMF WP/23/216 recipe)
# --------------------------------------------------------------------------


def build_complementarity_theta(
    work_context: pd.DataFrame | None = None,
    job_zones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconstruct the IMF WP/23/216 complementarity score theta.

    Six components — five from 11 O*NET Work Context elements (CX scale,
    normalised from 1-5 to [0, 1]) and one from the O*NET Job Zone — are
    averaged into theta in [0, 1] per US SOC 2018 occupation (O*NET-SOC
    8-digit codes are collapsed to 6-digit SOC by unweighted mean).

    The reconstruction is validated against the paper's published extremes:
    the minimum must fall at US SOC 51-9031 (cutters and trimmers, hand) and
    the maximum at 29-1022 (oral and maxillofacial surgeons).

    Args:
        work_context: O*NET 27.3 Work Context table (downloaded from
            :data:`ONET_27_3_WORK_CONTEXT_URL` when omitted).
        job_zones: O*NET 27.3 Job Zones table (downloaded from
            :data:`ONET_27_3_JOB_ZONES_URL` when omitted).

    Returns:
        DataFrame with columns ``soc2018_code``, ``complementarity_theta``
        and one column per component.
    """

    if work_context is None:
        work_context = _read_table_url(ONET_27_3_WORK_CONTEXT_URL)
    if job_zones is None:
        job_zones = _read_table_url(ONET_27_3_JOB_ZONES_URL)

    context = work_context[work_context["Scale ID"] == "CX"].copy()
    context["soc2018_code"] = context["O*NET-SOC Code"].str[:7]
    # CX scale runs 1-5; normalise to [0, 1].
    context["value"] = (pd.to_numeric(context["Data Value"]) - 1.0) / 4.0
    element_means = (
        context.groupby(["soc2018_code", "Element Name"])["value"]
        .mean()
        .unstack("Element Name")
    )

    components = pd.DataFrame(index=element_means.index)
    for component, elements in THETA_WORK_CONTEXT_COMPONENTS.items():
        parts = []
        for element_name, reverse in elements:
            if element_name not in element_means.columns:
                raise ValueError(
                    f"O*NET Work Context is missing element {element_name!r} "
                    f"needed for theta component {component!r}."
                )
            values = element_means[element_name]
            parts.append(1.0 - values if reverse else values)
        components[component] = pd.concat(parts, axis=1).mean(axis=1)

    zones = job_zones.copy()
    zones["soc2018_code"] = zones["O*NET-SOC Code"].str[:7]
    # Job zones run 1-5; normalise to [0, 1].
    components["skills"] = (
        zones.groupby("soc2018_code")["Job Zone"]
        .mean()
        .sub(1.0)
        .div(4.0)
        .reindex(components.index)
    )

    # TODO: WP/23/216 does not publish whether the six components are
    # weighted; an unweighted mean reproduces the published extremes.
    theta = components.mean(axis=1).dropna()

    observed_min = theta.idxmin()
    observed_max = theta.idxmax()
    if observed_min != THETA_MIN_SOC2018 or observed_max != THETA_MAX_SOC2018:
        raise AssertionError(
            "Reconstructed theta extremes do not match IMF WP/23/216: "
            f"expected min at {THETA_MIN_SOC2018} and max at "
            f"{THETA_MAX_SOC2018}, found min at {observed_min} and max at "
            f"{observed_max}."
        )

    result = components.copy()
    result["complementarity_theta"] = theta
    return result.reset_index().rename(columns={"index": "soc2018_code"})


# --------------------------------------------------------------------------
# 2. US measure chain: US SOC 2018 -> US SOC 2010 -> ISCO-08 -> UK SOC 2020
# --------------------------------------------------------------------------


def build_dsit_soc2010_to_soc2020(
    dsit_annex1: pd.DataFrame | None = None,
    coding_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map DSIT Annex 1 exposure scores from SOC 2010 to SOC 2020.

    The ONS SOC 2020 Volume 2 coding index lists each index entry with both
    its SOC 2010 and SOC 2020 unit group; the SOC 2010 -> SOC 2020 mapping
    is the set of (2010, 2020) pairs, with many-to-many links resolved as
    unweighted means over the linked SOC 2010 scores.

    Args:
        dsit_annex1: DSIT Annex 1 table with SOC 2010 codes and the
            standardised AIOE and LLM exposure columns (downloaded from
            :data:`DSIT_2023_ANNEX1_URL` when omitted).
        coding_index: ONS SOC 2020 Volume 2 coding index (downloaded from
            :data:`ONS_SOC2020_VOLUME2_CODING_INDEX_URL` when omitted).

    Returns:
        DataFrame with columns ``soc2020_code``, ``dsit_aioe``, ``dsit_llm``.
    """

    if dsit_annex1 is None:
        # TODO: Verify the Annex 1 sheet name and header row against the
        # published workbook; gov.uk annexes commonly carry cover sheets.
        dsit_annex1 = _read_excel_url(DSIT_2023_ANNEX1_URL, dtype=str)
    if coding_index is None:
        # TODO: Verify the coding-index sheet name and the SOC 2010/SOC 2020
        # column headers against the published ONS workbook.
        coding_index = _read_excel_url(ONS_SOC2020_VOLUME2_CODING_INDEX_URL, dtype=str)

    annex = _normalise_columns(
        dsit_annex1,
        {
            "soc2010_code": ("soc2010", "soc 2010", "soc2010 code", "soc code"),
            "dsit_aioe": ("aioe", "ai occupational exposure"),
            "dsit_llm": ("llm", "llm exposure", "large language model"),
        },
        label="DSIT Annex 1",
    )
    annex["soc2010_code"] = annex["soc2010_code"].str.strip().str[:4]
    annex["dsit_aioe"] = pd.to_numeric(annex["dsit_aioe"])
    annex["dsit_llm"] = pd.to_numeric(annex["dsit_llm"])

    index = _normalise_columns(
        coding_index,
        {
            "soc2010_code": ("soc2010", "soc 2010"),
            "soc2020_code": ("soc2020", "soc 2020"),
        },
        label="ONS SOC 2020 coding index",
    )
    pairs = (
        index.assign(
            soc2010_code=index["soc2010_code"].str.strip().str[:4],
            soc2020_code=index["soc2020_code"].str.strip().str[:4],
        )
        .query("soc2010_code.str.fullmatch('\\\\d{4}')")
        .query("soc2020_code.str.fullmatch('\\\\d{4}')")
        .drop_duplicates(["soc2010_code", "soc2020_code"])
    )
    mapped = pairs.merge(annex, on="soc2010_code", how="inner")
    return (
        mapped.groupby("soc2020_code")[["dsit_aioe", "dsit_llm"]].mean().reset_index()
    )


def build_us_measure_chain(
    theta: pd.DataFrame | None = None,
    felten_aioe: pd.DataFrame | None = None,
    eloundou: pd.DataFrame | None = None,
    soc2010_soc2018: pd.DataFrame | None = None,
    isco08_soc2010: pd.DataFrame | None = None,
    coding_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Chain US-SOC-based measures onto UK SOC 2020 unit groups.

    US SOC 2018 scores (theta, Felten AIOE, Eloundou beta) are chained
    US SOC 2018 -> US SOC 2010 (BLS crosswalk, Nov 2017) -> ISCO-08 (BLS
    ISCO-08/SOC 2010 crosswalk, 2012) -> SOC 2020 (ONS SOC 2020 Volume 2
    coding index, which assigns an ISCO-08 code to every SOC 2020 index
    entry). Many-to-many links are resolved with employment-unweighted
    means; ``mapping_quality`` is ``direct`` for single-path links and
    ``chained`` otherwise.

    Returns:
        DataFrame with columns ``soc2020_code``, ``complementarity_theta``,
        ``felten_aioe``, ``eloundou_beta``, ``mapping_quality``,
        ``n_isco_links``.
    """

    if theta is None:
        theta = build_complementarity_theta()
    if felten_aioe is None:
        # TODO: Verify the AIOE column header in the AIOE-Data/AIOE release.
        felten_aioe = _read_csv_url(FELTEN_AIOE_URL)
    if eloundou is None:
        # TODO: Verify the occupation-level beta column header in the
        # openai/GPTs-are-GPTs release.
        eloundou = _read_csv_url(ELOUNDOU_GPTS_ARE_GPTS_URL)
    if soc2010_soc2018 is None:
        soc2010_soc2018 = _read_excel_url(BLS_SOC2010_SOC2018_CROSSWALK_URL, dtype=str)
    if isco08_soc2010 is None:
        isco08_soc2010 = _read_excel_url(BLS_ISCO08_SOC2010_CROSSWALK_URL, dtype=str)
    if coding_index is None:
        coding_index = _read_excel_url(ONS_SOC2020_VOLUME2_CODING_INDEX_URL, dtype=str)

    crosswalk_2018_2010 = _normalise_columns(
        soc2010_soc2018,
        {
            "soc2018_code": ("2018 soc code",),
            "soc2010_code": ("2010 soc code",),
        },
        label="BLS SOC 2010/2018 crosswalk",
    ).drop_duplicates()
    crosswalk_isco = _normalise_columns(
        isco08_soc2010,
        {
            "isco08_code": ("isco-08 code", "isco08", "isco 08 code"),
            "soc2010_code": ("2010 soc code", "soc2010", "soc code"),
        },
        label="BLS ISCO-08/SOC 2010 crosswalk",
    ).drop_duplicates()
    index = _normalise_columns(
        coding_index,
        {
            "soc2020_code": ("soc2020", "soc 2020"),
            "isco08_code": ("isco08", "isco-08", "isco 08"),
        },
        label="ONS SOC 2020 coding index",
    ).drop_duplicates(["soc2020_code", "isco08_code"])

    # Merge US SOC 2018 scores.
    # TODO: Column names for the Felten and Eloundou releases below follow
    # the papers' descriptions; verify against the downloaded files.
    scores = theta[["soc2018_code", "complementarity_theta"]].copy()
    felten = _normalise_columns(
        felten_aioe,
        {"soc2018_code": ("soc", "soc code"), "felten_aioe": ("aioe",)},
        label="Felten AIOE",
    )
    gpts = _normalise_columns(
        eloundou,
        {"soc2018_code": ("soc", "soc code"), "eloundou_beta": ("beta",)},
        label="Eloundou GPTs-are-GPTs",
    )
    scores = scores.merge(felten, on="soc2018_code", how="outer").merge(
        gpts, on="soc2018_code", how="outer"
    )
    measures = ["complementarity_theta", "felten_aioe", "eloundou_beta"]
    for column in measures:
        scores[column] = pd.to_numeric(scores[column])

    # US SOC 2018 -> 2010 -> ISCO-08 -> SOC 2020, unweighted means per step.
    on_2010 = (
        scores.merge(crosswalk_2018_2010, on="soc2018_code", how="inner")
        .groupby("soc2010_code")[measures]
        .mean()
        .reset_index()
    )
    on_isco = (
        on_2010.merge(crosswalk_isco, on="soc2010_code", how="inner")
        .groupby("isco08_code")
        .agg({**{m: "mean" for m in measures}, "soc2010_code": "nunique"})
        .rename(columns={"soc2010_code": "n_soc2010_links"})
        .reset_index()
    )
    linked = index.merge(on_isco, on="isco08_code", how="inner")
    result = (
        linked.groupby("soc2020_code")
        .agg(
            {
                **{m: "mean" for m in measures},
                "isco08_code": "nunique",
                "n_soc2010_links": "max",
            }
        )
        .rename(columns={"isco08_code": "n_isco_links"})
        .reset_index()
    )
    result["mapping_quality"] = np.where(
        (result["n_isco_links"] == 1) & (result["n_soc2010_links"] == 1),
        "direct",
        "chained",
    )
    return result.drop(columns=["n_soc2010_links"])


# --------------------------------------------------------------------------
# 3. The packaged unit-group table
# --------------------------------------------------------------------------


def build_ai_exposure_table(
    us_chain: pd.DataFrame | None = None,
    dsit: pd.DataFrame | None = None,
    all_soc2020_codes: pd.Series | None = None,
) -> pd.DataFrame:
    """Compose the packaged SOC 2020 unit-group AI-exposure table.

    Combines the chained US measures with the DSIT SOC 2010 -> SOC 2020
    scores, computes the composite ``c_aioe = felten_aioe *
    (1 - (theta - theta_min))`` with ``theta_min`` taken from the theta
    reconstruction, and fills unit groups missing any measure from 3-digit
    (then 2-digit) sibling means, flagged ``imputed-from-parent``.

    Args:
        us_chain: Output of :func:`build_us_measure_chain`.
        dsit: Output of :func:`build_dsit_soc2010_to_soc2020`.
        all_soc2020_codes: Complete list of SOC 2020 unit groups (defaults
            to the packaged table's index so coverage matches the shipped
            412 unit groups).

    Returns:
        DataFrame indexed like the packaged
        ``uk_soc2020_ai_exposure.csv`` (one row per SOC 2020 unit group).
    """

    if us_chain is None:
        us_chain = build_us_measure_chain()
    if dsit is None:
        dsit = build_dsit_soc2010_to_soc2020()
    if all_soc2020_codes is None:
        # TODO: Derive the canonical 412 unit groups from the ONS SOC 2020
        # Volume 1 structure instead of the packaged table when rebuilding
        # from scratch.
        all_soc2020_codes = load_ai_exposure_table().index.to_series()

    table = (
        pd.DataFrame({"soc2020_code": all_soc2020_codes.astype(str).values})
        .merge(us_chain, on="soc2020_code", how="left")
        .merge(dsit, on="soc2020_code", how="left")
        .set_index("soc2020_code")
        .sort_index()
    )
    theta_min = table["complementarity_theta"].min()
    table["c_aioe"] = table["felten_aioe"] * (
        1.0 - (table["complementarity_theta"] - theta_min)
    )

    measure_columns = sorted(set(MEASURE_TO_COLUMN.values()))
    missing_any = table[measure_columns].isna().any(axis=1)
    for length in (3, 2):
        still_missing = table[measure_columns].isna().any(axis=1)
        if not still_missing.any():
            break
        prefix = table.index.str[:length]
        parent_means = table[measure_columns].groupby(prefix).transform("mean")
        table[measure_columns] = table[measure_columns].fillna(parent_means)
    table.loc[missing_any, "mapping_quality"] = "imputed-from-parent"
    return table


# --------------------------------------------------------------------------
# 4. Major-group aggregation (runs from packaged data)
# --------------------------------------------------------------------------


def build_major_group_ai_exposure(
    ashe_csv: str | Path | None = None,
    *,
    weighting_label: str = MAJOR_GROUP_WEIGHTING_LABEL,
) -> pd.DataFrame:
    """ASHE-employment-weighted major-group (1-digit) AI-exposure table.

    Aggregates the packaged 4-digit unit-group exposure table to SOC 2020
    major groups 1-9, weighting each unit group by its ASHE Table 14
    employee-job count. Unit groups whose ASHE cell is suppressed (blank in
    the tidy CSV) carry no weight and are excluded from the weighted means.

    Args:
        ashe_csv: Tidy ASHE Table 14 CSV with ``soc_code`` and
            ``employment_jobs`` columns. Defaults to the packaged
            ``occupation_targets_data/ashe_table14_2025_soc4.csv``.
        weighting_label: Value recorded in the output ``weighting`` column.

    Returns:
        DataFrame with one row per major group, matching the packaged
        ``uk_soc2020_major_group_ai_exposure.csv`` schema.
    """

    if ashe_csv is None:
        resource = importlib_resources.files("populace.build.uk_runtime").joinpath(
            "occupation_targets_data", PACKAGED_ASHE_TABLE14_CSV
        )
        with importlib_resources.as_file(resource) as path:
            ashe = pd.read_csv(path, dtype={"soc_code": str})
    else:
        ashe = pd.read_csv(ashe_csv, dtype={"soc_code": str})
    if not {"soc_code", "employment_jobs"}.issubset(ashe.columns):
        raise ValueError(
            "ASHE CSV must include 'soc_code' and 'employment_jobs' columns."
        )
    ashe = ashe.copy()
    ashe["soc_code"] = ashe["soc_code"].astype(str).str.strip()
    ashe = ashe[ashe["soc_code"].str.fullmatch(r"\d{4}")]
    ashe["employment_jobs"] = pd.to_numeric(ashe["employment_jobs"], errors="coerce")
    weights = ashe.set_index("soc_code")["employment_jobs"]

    table = load_ai_exposure_table().copy()
    table["weight"] = weights.reindex(table.index)
    table["major_group"] = table.index.str[0]

    measure_columns = [
        "c_aioe",
        "complementarity_theta",
        "felten_aioe",
        "eloundou_beta",
        "dsit_aioe",
        "dsit_llm",
    ]
    rows = []
    for group in sorted(table["major_group"].unique()):
        subset = table[table["major_group"] == group]
        weighted = subset[subset["weight"].notna() & (subset["weight"] > 0)]
        if weighted.empty:
            raise ValueError(f"Major group {group} has no ASHE-weighted unit groups.")
        row: dict[str, Any] = {
            "soc2020_major_group": group,
            "major_group_title": SOC2020_MAJOR_GROUP_TITLES.get(group, ""),
        }
        for column in measure_columns:
            row[column] = float(
                np.average(weighted[column], weights=weighted["weight"])
            )
        row["weighting"] = weighting_label
        row["employment_jobs_thousands"] = float(weighted["weight"].sum()) / 1000.0
        row["n_unit_groups"] = int(len(subset))
        row["n_unit_groups_weighted"] = int(len(weighted))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(MAJOR_GROUP_TABLE_COLUMNS))


def write_major_group_ai_exposure(table: pd.DataFrame, path: str | Path) -> None:
    """Write the major-group table to CSV in the packaged format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.6f")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _normalise_columns(
    frame: pd.DataFrame,
    targets: dict[str, tuple[str, ...]],
    *,
    label: str,
) -> pd.DataFrame:
    """Rename fuzzy source headers to canonical names; keep only those."""

    lower_to_column = {str(column).strip().lower(): column for column in frame.columns}
    rename: dict[str, str] = {}
    for canonical, candidates in targets.items():
        found = None
        if canonical in frame.columns:
            found = canonical
        else:
            for candidate in candidates:
                for lowered, column in lower_to_column.items():
                    if candidate in lowered:
                        found = column
                        break
                if found is not None:
                    break
        if found is None:
            raise ValueError(f"{label} is missing a column matching {canonical!r}.")
        rename[found] = canonical
    return frame.rename(columns=rename)[list(targets)].copy()


def _read_url_bytes(
    url: str,
    *,
    timeout: int = 300,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PolicyEngine-Populace/0.1"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            should_retry = error.code in {429, 500, 502, 503, 504}
            if not should_retry or attempt == retries - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(retry_delay * (2**attempt))
    raise RuntimeError(f"Could not download {url}.")


def _read_csv_url(url: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_read_url_bytes(url)), **kwargs)


def _read_table_url(url: str, **kwargs: Any) -> pd.DataFrame:
    """Read a tab-delimited O*NET database text file."""

    return pd.read_csv(io.BytesIO(_read_url_bytes(url)), sep="\t", **kwargs)


def _read_excel_url(url: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(_read_url_bytes(url)), **kwargs)
