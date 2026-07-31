"""SIPP-informed child disability inputs for ages 0--14.

CPS ASEC's six-question disability battery starts at age 15.  The certified
Populace base therefore carries ``is_disabled = False`` for every younger
child, making child SSI structurally absent (populace #453).  This stage
owns the generic under-15 ``is_disabled`` repair without replacing the adult
ASEC surface:

* Ages 1--14 receive seeded Bernoulli draws from a survey-weighted logistic
  classifier fit to December 2023 SIPP.  The outcome follows Census's 2023
  SIPP User's Guide construction of ``RDIS_ALT``.  For children, the effective
  item list is ``EHEARING``, ``ESEEING``, ``ECOGNIT``, ``EAMBULAT``,
  ``ESELFCARE``, ``EDDELAY``, ``EPLAYDIF``, and ``ESCHOOLWK``: the first two
  apply from age 1; the next three from age 5; ``EDDELAY`` applies at ages
  1--4; and ``EPLAYDIF`` / ``ESCHOOLWK`` apply at ages 5--14.  The full
  all-age list also contains ``EERRANDS``, ``EFINDJOB``, ``EJOBCANT``, and
  ``EDISABL``; those four are out-of-universe below 15.  A computed any-item
  indicator is checked row for row against Census's ``RDIS_ALT`` before
  fitting.
* The deliberately small shared predictor set is age, sex, and a
  household-income proxy.  SIPP's personal monthly ``TPTOTINC`` is summed over
  December residence members aged 15+ and annualized; the recipient analogue
  is household-summed ``employment_income_before_lsr`` for members aged 15+.
  Excluding younger members avoids using a child's own mostly-blank
  ``TPTOTINC`` as a near-direct SSI-receipt indicator.  Receiver probabilities
  are calibrated separately to the observed age-1--4 and age-5--14 rates.
* Age 0 alone receives a seeded unconditional draw at the nearest observed
  exact-age rate, age 1.  The User's Guide states that SIPP collects no
  disability data below age 1, and every disability item plus ``RDIS_ALT`` is
  blank for all 203 pinned December age-0 records.  The former pooled ages
  1--4 transport was 34.0 percent above the age-1 rate because age 4 carried
  much higher incidence; it is not reused.

The Census construct is not the SSI legal standard.  Section 4.3.4 of the
2023 SIPP User's Guide defines ``RDIS_ALT`` as a general difficulty summary and
states that it excludes the separately sponsored SSA disability questions:
https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2023_SIPP_Users_Guide_OCT24.pdf.
By contrast, 20 CFR 416.906 requires a medically determinable impairment with
marked and severe functional limitations and the statutory duration (or death)
condition; a child filing a new application also cannot be doing substantial
gainful activity:
https://www.ecfr.gov/current/title-20/chapter-III/part-416/subpart-I/section-416.906.
The stage therefore never copies ``RDIS_ALT`` into
``meets_ssi_disability_criteria``.  It models the marked-and-severe, pre-SGA
criteria layer using the eight effective child battery items as predictors in
a weighted 100-tree receipt classifier.  For each
general-disabled receiver it hot-decks one exact-age/sex positive battery
profile (age 0 uses age 1), converts that profile to a receipt-probability
severity score, logit-calibrates scores to the weighted *reported* monthly
``RSSI_MNYN`` receipt share within ``RDIS_ALT``-positive children, and makes
one seeded source-person draw.  This produces a proper severe subset while
retaining the RDIS_ALT-faithful general signal.

Draws are keyed by stable person source identity, so support clones and row
reordering do not change assignment.  Existing child ``True`` values are never
cleared, and writes are restricted to under-15 rows so adult stored values are
untouched even when a source uses noncanonical boolean bytes.

This stage owns both under-15 assignments.  The later
``ssi_disability_criteria`` stage recomputes the archived adult SIPP/QRF
surface but preserves this exact child severe assignment, so SSI sees no
second random decision.  The generic signal also intentionally reaches the
other PolicyEngine-US program families for which child disability is a real
input: HUD housing, SNAP, Medicaid, TANF, child-care assistance, and tax
provisions.  The stage gate reports weighted before/after shares and the
criteria-within-general share so that deliberate blast radius is measured
rather than silent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.full_sipp_donor import full_sipp_sha256
from populace.build.us_runtime.sipp_financial_assets import (
    fetch_sipp_2023_financial_asset_donor,
)
from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_role_series,
)
from populace.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "CHILD_DISABILITY_SIPP_DICTIONARY_URL",
    "CHILD_DISABILITY_SIPP_USERS_GUIDE_URL",
    "SSI_CHILD_DISABILITY_STANDARD_URL",
    "SIPP_2023_CHILD_DISABILITY_DONOR_REVISION",
    "SIPP_2023_CHILD_DISABILITY_DONOR_SHA256",
    "SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES",
    "SIPP_2023_CHILD_DISABILITY_DONOR_URL",
    "SIPP_2023_CHILD_DISABILITY_LOCAL_PATH",
    "SIPP_CHILD_DISABILITY_AGE_0_PARAMETERS",
    "SIPP_CHILD_DISABILITY_FIT_PARAMETERS",
    "SIPP_CHILD_DISABILITY_MODEL_PREDICTORS",
    "SIPP_CHILD_DISABILITY_READ_PARAMETERS",
    "SIPP_CHILD_DISABILITY_SOURCE_COLUMNS",
    "SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS",
    "SIPP_CHILD_SSI_SEVERITY_PARAMETERS",
    "US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE",
    "US_CHILD_DISABILITY_AGE_0_SHARE_BAND",
    "US_CHILD_DISABILITY_AGE_1_TARGET_RATE",
    "US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND",
    "US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE",
    "US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND",
    "US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE",
    "US_CHILD_DISABILITY_OUTPUT_COLUMNS",
    "US_CHILD_DISABILITY_STAGE_NAME",
    "US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR",
    "US_CHILD_SSI_SEVERITY_SHARE_BAND",
    "load_sipp_2023_child_disability_donor",
    "resolve_sipp_2023_child_disability_donor",
    "us_child_disability_signal_gate",
    "us_child_disability_stage_spec",
    "us_child_disability_summary",
    "with_us_child_disability_inputs",
]

CHILD_DISABILITY_SIPP_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
)
CHILD_DISABILITY_SIPP_USERS_GUIDE_URL = (
    "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
    "methodology/2023_SIPP_Users_Guide_OCT24.pdf"
)
SSI_CHILD_DISABILITY_STANDARD_URL = (
    "https://www.ecfr.gov/current/title-20/chapter-III/part-416/"
    "subpart-I/section-416.906"
)

# Reuse the exact full-file coordinate already pinned by the SIPP vehicle,
# voluntary-filing, Head Start, and SSI-disability-criteria stages.
SIPP_2023_CHILD_DISABILITY_DONOR_REVISION = SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
SIPP_2023_CHILD_DISABILITY_DONOR_SHA256 = SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256
SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES = (
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES
)
SIPP_2023_CHILD_DISABILITY_DONOR_URL = SIPP_2023_VOLUNTARY_FILING_DONOR_URL

# Developer-machine fast path requested by populace #453.  Release builds use
# it only when no explicit --sipp-vehicle-donor path was supplied; the loader
# still verifies the immutable artifact's byte length and SHA-256.
SIPP_2023_CHILD_DISABILITY_LOCAL_PATH = (
    Path("/Users/maxghenis/PolicyEngine")
    / ("policyengine-" + "us-data")
    / ("policyengine_" + "us_data")
    / "storage"
    / "pu2023.csv"
)

US_CHILD_DISABILITY_STAGE_NAME = "child_disability"
US_CHILD_DISABILITY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_disabled",
    "meets_ssi_disability_criteria",
)

_CORE_CHILD_SOURCE_COLUMNS: tuple[str, ...] = (
    "EHEARING",
    "ESEEING",
    "ECOGNIT",
    "EAMBULAT",
    "ESELFCARE",
)
_CHILD_SPECIFIC_SOURCE_COLUMNS: tuple[str, ...] = (
    "EDDELAY",
    "EPLAYDIF",
    "ESCHOOLWK",
)
_OUTCOME_SOURCE_COLUMNS = (
    *_CORE_CHILD_SOURCE_COLUMNS,
    *_CHILD_SPECIFIC_SOURCE_COLUMNS,
)
SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS: tuple[str, ...] = _OUTCOME_SOURCE_COLUMNS
SIPP_CHILD_DISABILITY_SOURCE_COLUMNS: tuple[str, ...] = (
    "SSUID",
    "ERESIDENCEID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "TPTOTINC",
    "RSSI_MNYN",
    "ASSI_MNYN",
    "THHLDSTATUS",
    "RDIS_ALT",
    *_OUTCOME_SOURCE_COLUMNS,
)
SIPP_CHILD_DISABILITY_MODEL_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    "household_income_proxy",
)

_GENERAL_OUTPUT = US_CHILD_DISABILITY_OUTPUT_COLUMNS[0]
_CRITERIA_OUTPUT = US_CHILD_DISABILITY_OUTPUT_COLUMNS[1]
_DONOR_WEIGHT_COLUMN = "sipp_weight"
_HOUSEHOLD_INCOME_COLUMN = "household_income_proxy"
_MONTHLY_SSI_RECEIVED_COLUMN = "received_monthly_ssi"
_MONTHLY_SSI_REPORTED_COLUMN = "monthly_ssi_label_is_reported"
_MIN_MODEL_AGE = 1
_MAX_MODEL_AGE = 14
_MAX_EARLY_CHILD_AGE = 4
_PERSON_SOURCE_ID_COLUMN = "person_source_id"
_ASEC_SUPPORT_CHANNEL = BASE_ASEC_SUPPORT_CHANNEL
_PUF_SUPPORT_CHANNEL = PUF_TAX_DETAIL_SUPPORT_CHANNEL
_SEVERITY_N_ESTIMATORS = 100
_SEVERITY_MODEL_SEED = 42

# Immutable-source audit.  The exact reproducible filters are:
#
#   december = pu2023.query("MONTHCODE == 12")
#   age_1_4 = december.query("1 <= TAGE <= 4")
#   age_5_14 = december.query("5 <= TAGE <= 14")
#   valid = RDIS_ALT in [1, 2]
#   positive = any(EHEARING, ESEEING, ECOGNIT, EAMBULAT, ESELFCARE,
#                  EDDELAY, EPLAYDIF, ESCHOOLWK) == 1
#   core_positive = any(EHEARING, ESEEING, ECOGNIT, EAMBULAT,
#                       ESELFCARE) == 1
#   child_positive = any(EDDELAY, EPLAYDIF, ESCHOOLWK) == 1
#   child_only = child_positive and not core_positive
#   monthly_ssi = RSSI_MNYN == 1
#   monthly_ssi_reported = ASSI_MNYN == 1 and RSSI_MNYN in [1, 2]
#
# For each band, ``*_rows`` and ``*_weight`` are the count and WPFINWGT sum
# before the RDIS_ALT universe filter; ``*_valid_*`` applies ``valid``;
# ``*_missing_*`` applies its complement; ``*_positive_*`` applies
# ``valid & positive``; and ``*_positive_rate`` divides positive weight by
# valid weight.  The monthly-SSI ``all_rate`` divides by all band weight while
# ``valid_rate`` divides by valid weight.  The child-only and SSI/positive
# intersections use the named booleans above.  Age-0 support is queried with
# ``TAGE == 0`` and counts any nonmissing outcome item or RDIS_ALT value.
#
# Census's dictionary universe is TAGE >= 1 and THHLDSTATUS in [1, 2].
# Seven child movers (THHLDSTATUS == 4) are outside it and remain missing, not
# negative.  On every valid age-1--14 row, ``positive`` exactly equals
# ``RDIS_ALT == 1``.
_PINNED_RAW_ROWS = 476_744
_PINNED_DECEMBER_ROWS = 39_513
_PINNED_AGE_0_ROWS = 203
_PINNED_AGE_0_WEIGHT = 2_246_712.4918057
_PINNED_AGE_0_OBSERVED_ITEM_ROWS = 0
_PINNED_AGE_0_MONTHLY_SSI_ROWS = 0
_PINNED_AGE_0_MONTHLY_SSI_WEIGHT = 0.0
_PINNED_AGE_1_ROWS = 328
_PINNED_AGE_1_WEIGHT = 3_540_723.0516442
_PINNED_AGE_1_VALID_ROWS = 328
_PINNED_AGE_1_VALID_WEIGHT = 3_540_723.0516442
_PINNED_AGE_1_POSITIVE_ROWS = 13
_PINNED_AGE_1_POSITIVE_WEIGHT = 133_805.00840989998
_PINNED_AGE_1_RATE = 0.03779030623357146
_PINNED_AGE_1_4_ROWS = 1_451
_PINNED_AGE_1_4_WEIGHT = 14_919_709.707403801
_PINNED_AGE_1_4_VALID_ROWS = 1_447
_PINNED_AGE_1_4_VALID_WEIGHT = 14_861_539.117022298
_PINNED_AGE_1_4_MISSING_ROWS = 4
_PINNED_AGE_1_4_MISSING_WEIGHT = 58_170.590381500006
_PINNED_AGE_1_4_POSITIVE_ROWS = 85
_PINNED_AGE_1_4_POSITIVE_WEIGHT = 752_667.5853464998
_PINNED_AGE_1_4_RATE = 0.05064533218395932
_PINNED_AGE_1_4_MONTHLY_SSI_ROWS = 7
_PINNED_AGE_1_4_MONTHLY_SSI_WEIGHT = 53_096.3144829
_PINNED_AGE_1_4_MONTHLY_SSI_ALL_RATE = 0.0035588034569165464
_PINNED_AGE_1_4_MONTHLY_SSI_VALID_RATE = 0.0035727332185994027
_PINNED_AGE_1_4_CHILD_ONLY_ROWS = 63
_PINNED_AGE_1_4_CHILD_ONLY_WEIGHT = 579_339.2190467998
_PINNED_AGE_1_4_MONTHLY_SSI_POSITIVE_ROWS = 5
_PINNED_AGE_1_4_MONTHLY_SSI_CORE_POSITIVE_ROWS = 0
_PINNED_AGE_5_14_ROWS = 4_242
_PINNED_AGE_5_14_WEIGHT = 40_772_887.19759989
_PINNED_AGE_5_14_VALID_ROWS = 4_239
_PINNED_AGE_5_14_VALID_WEIGHT = 40_754_911.58143689
_PINNED_AGE_5_14_MISSING_ROWS = 3
_PINNED_AGE_5_14_MISSING_WEIGHT = 17_975.616163
_PINNED_AGE_5_14_POSITIVE_ROWS = 567
_PINNED_AGE_5_14_POSITIVE_WEIGHT = 5_074_723.2084883
_PINNED_AGE_5_14_RATE = 0.1245180767561705
_PINNED_AGE_5_14_MONTHLY_SSI_ROWS = 70
_PINNED_AGE_5_14_MONTHLY_SSI_WEIGHT = 698_957.5609709
_PINNED_AGE_5_14_MONTHLY_SSI_ALL_RATE = 0.017142704601307814
_PINNED_AGE_5_14_MONTHLY_SSI_VALID_RATE = 0.01715026566980119
_PINNED_AGE_5_14_CHILD_ONLY_ROWS = 72
_PINNED_AGE_5_14_CHILD_ONLY_WEIGHT = 579_374.2673576999
_PINNED_REPORTED_MONTHLY_SSI_ROWS = 4_517
_PINNED_REPORTED_MONTHLY_SSI_WEIGHT = 44_292_243.44820082
_PINNED_REPORTED_MONTHLY_SSI_RECEIPT_ROWS = 65
_PINNED_REPORTED_MONTHLY_SSI_RECEIPT_WEIGHT = 634_869.4032926
_PINNED_REPORTED_RDIS_POSITIVE_ROWS = 520
_PINNED_REPORTED_RDIS_POSITIVE_WEIGHT = 4_658_694.167000299
_PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_ROWS = 43
_PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_WEIGHT = 432_260.9795292
_PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_RATE = 0.09278586746284095
_PINNED_RDIS_ALT_UNIVERSE_MISMATCH_ROWS = 0
_PINNED_RDIS_ALT_MISMATCH_ROWS = 0
_PINNED_FLOAT_RTOL = 1e-10

US_CHILD_DISABILITY_AGE_1_TARGET_RATE = round(_PINNED_AGE_1_RATE, 12)
US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE = round(_PINNED_AGE_1_4_RATE, 12)
US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE = round(_PINNED_AGE_5_14_RATE, 12)
# SIPP has no disability observation below age 1.  Transport only the adjacent
# observed age, and label it as a fallback rather than a survey observation.
# Pooling ages 1--4 would inflate the nearest-age estimate by 34.0167 percent.
US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE = US_CHILD_DISABILITY_AGE_1_TARGET_RATE
_AGE_0_FINITE_DRAW_ABSOLUTE_TOLERANCE = 0.012
_FINITE_DRAW_ABSOLUTE_TOLERANCE = 0.015
US_CHILD_DISABILITY_AGE_0_SHARE_BAND = (
    US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE - _AGE_0_FINITE_DRAW_ABSOLUTE_TOLERANCE,
    US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE + _AGE_0_FINITE_DRAW_ABSOLUTE_TOLERANCE,
)
US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND = (
    US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE - _FINITE_DRAW_ABSOLUTE_TOLERANCE,
    US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE + _FINITE_DRAW_ABSOLUTE_TOLERANCE,
)
US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND = (
    US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE - _FINITE_DRAW_ABSOLUTE_TOLERANCE,
    US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE + _FINITE_DRAW_ABSOLUTE_TOLERANCE,
)
US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR = round(
    _PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_RATE,
    12,
)
_SEVERITY_FINITE_DRAW_ABSOLUTE_TOLERANCE = 0.03
US_CHILD_SSI_SEVERITY_SHARE_BAND = (
    US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR - _SEVERITY_FINITE_DRAW_ABSOLUTE_TOLERANCE,
    US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR + _SEVERITY_FINITE_DRAW_ABSOLUTE_TOLERANCE,
)

SIPP_CHILD_DISABILITY_READ_PARAMETERS: dict[str, object] = {
    "table": "sipp_person",
    "delimiter": "|",
    "month_column": "MONTHCODE",
    "month": 12,
    "source_columns": list(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS),
}
SIPP_CHILD_DISABILITY_FIT_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_CHILD_DISABILITY_MODEL_PREDICTORS),
    "target": _GENERAL_OUTPUT,
    "target_rule": (
        "Census RDIS_ALT child construction: any(EHEARING, ESEEING, "
        "ECOGNIT, EAMBULAT, ESELFCARE, EDDELAY, EPLAYDIF, ESCHOOLWK) == 1"
    ),
    "target_source_items": list(_OUTCOME_SOURCE_COLUMNS),
    "target_validation": "computed outcome == (RDIS_ALT == 1) on every valid row",
    "valid_universe": "RDIS_ALT in [1, 2]",
    "weight": _DONOR_WEIGHT_COLUMN,
    "age_domain": [_MIN_MODEL_AGE, _MAX_MODEL_AGE],
    "calibration_age_bands": {
        "age_1_4": [1, 4],
        "age_5_14": [5, 14],
    },
    "pinned_weighted_rates": {
        "age_1_4": US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE,
        "age_5_14": US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE,
    },
    "classifier": "weighted_logistic_irls",
    "income_proxy": (
        "sum December TPTOTINC for age-15+ members within SSUID + "
        "ERESIDENCEID, multiply by 12; recipient analogue is age-15+ "
        "household-summed employment_income_before_lsr"
    ),
    "income_transform": "log1p_nonnegative_then_standardize",
    "calibrate_receiver_intercept_to_sipp_weighted_rate_by_age_band": True,
    "assignment": "seeded_bernoulli_by_person_source_id",
    "seed_from_build_config": True,
}
SIPP_CHILD_DISABILITY_AGE_0_PARAMETERS: dict[str, object] = {
    "age_domain": [0, 0],
    "rate": US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE,
    "rate_role": "nearest_observed_age_1_rate_transport_for_unobserved_age_0",
    "source_age_domain": [1, 1],
    "source_rate_query": (
        "MONTHCODE == 12 and TAGE == 1 and RDIS_ALT in [1, 2], "
        "weighted mean of computed RDIS_ALT child item union"
    ),
    "age_0_support_query": (
        "MONTHCODE == 12 and TAGE == 0: 203 rows; every child disability "
        "item and RDIS_ALT is missing"
    ),
    "monthly_ssi_audit_queries": {
        "age_1_4": (
            "MONTHCODE == 12 and 1 <= TAGE <= 4 and RSSI_MNYN == 1, "
            "weighted by WPFINWGT"
        ),
        "age_5_14": (
            "MONTHCODE == 12 and 5 <= TAGE <= 14 and RSSI_MNYN == 1, "
            "weighted by WPFINWGT"
        ),
    },
    "assignment": "seeded_bernoulli_by_person_source_id",
    "seed_from_build_config": True,
}
SIPP_CHILD_SSI_SEVERITY_PARAMETERS: dict[str, object] = {
    "predictors": list(SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS),
    "target": _CRITERIA_OUTPUT,
    "target_proxy": "RSSI_MNYN == 1",
    "observed_label_rule": "ASSI_MNYN == 1 and RSSI_MNYN in [1, 2]",
    "training_universe": (
        "MONTHCODE == 12 and 1 <= TAGE <= 14 and RDIS_ALT == 1 "
        "and WPFINWGT > 0 and observed_label_rule"
    ),
    "weight": _DONOR_WEIGHT_COLUMN,
    "classifier": "weighted_random_forest_classifier",
    "n_estimators": _SEVERITY_N_ESTIMATORS,
    "model_seed": _SEVERITY_MODEL_SEED,
    "receiver_item_profile": (
        "stable WPFINWGT hot deck from exact-age and sex RDIS_ALT-positive "
        "donors; receiver age 0 maps to donor age 1"
    ),
    "probability_calibration": (
        "logit shift over general-disabled receiver children to the weighted "
        "reported monthly-SSI share within RDIS_ALT-positive donor children"
    ),
    "pinned_conditional_receipt_rate": US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR,
    "assignment": "seeded_bernoulli_by_person_source_id",
    "seed_from_build_config": True,
    "legal_standard": "20 CFR 416.906 marked and severe functional limitations",
    "legal_source": SSI_CHILD_DISABILITY_STANDARD_URL,
    "general_difficulty_source": CHILD_DISABILITY_SIPP_USERS_GUIDE_URL,
}


def us_child_disability_stage_spec() -> SourceStageSpec:
    """Load and strictly validate the packaged child-disability stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_CHILD_DISABILITY_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_CHILD_DISABILITY_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_CHILD_DISABILITY_STAGE_NAME]
    if spec.grain != "person":
        raise ValueError("US child-disability stage must have person grain.")
    if tuple(spec.outputs) != US_CHILD_DISABILITY_OUTPUT_COLUMNS:
        raise ValueError(
            "US child-disability manifest outputs drifted from the runtime-owned "
            "family."
        )
    expected_operations = [
        ("read_table", SIPP_CHILD_DISABILITY_READ_PARAMETERS),
        ("fit_weighted_logistic", SIPP_CHILD_DISABILITY_FIT_PARAMETERS),
        ("assign_binary_from_rate", SIPP_CHILD_DISABILITY_AGE_0_PARAMETERS),
        ("fit_weighted_imputer", SIPP_CHILD_SSI_SEVERITY_PARAMETERS),
    ]
    if [operation.kind for operation in spec.operations] != [
        kind for kind, _ in expected_operations
    ]:
        raise ValueError(
            "US child-disability stage must contain read_table, "
            "fit_weighted_logistic, assign_binary_from_rate, then the "
            "receipt-anchored fit_weighted_imputer."
        )
    for operation, (kind, parameters) in zip(
        spec.operations, expected_operations, strict=True
    ):
        if dict(operation.parameters) != parameters:
            raise ValueError(
                f"US child-disability {kind} contract drifted from the reviewed "
                "source transform."
            )
    pinned = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("revision") == SIPP_2023_CHILD_DISABILITY_DONOR_REVISION
        and artifact.get("sha256") == SIPP_2023_CHILD_DISABILITY_DONOR_SHA256
        and artifact.get("size_bytes") == SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES
    ]
    if not pinned:
        raise ValueError(
            "US child-disability stage does not pin the full SIPP revision, "
            "SHA-256, and byte length."
        )
    return spec


def resolve_sipp_2023_child_disability_donor(
    path: str | Path | None = None,
) -> Path:
    """Resolve and verify the one full-SIPP path shared by all six stages.

    An explicit user path is fail-fast: missing, stale, or wrong bytes raise
    here without falling back.  The implicit developer path instead enters the
    pinned financial-asset fetch chain, where an invalid candidate falls
    through to the immutable Hugging Face revision.
    """

    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        actual_size = explicit.stat().st_size
        if actual_size != SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES:
            raise ValueError(
                "Explicit full SIPP donor failed byte-length verification: "
                f"expected {SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES}, "
                f"got {actual_size}."
            )
        actual_sha256 = _sha256_file(explicit)
        if actual_sha256 != SIPP_2023_CHILD_DISABILITY_DONOR_SHA256:
            raise ValueError(
                "Explicit full SIPP donor failed SHA-256 verification: "
                f"expected {SIPP_2023_CHILD_DISABILITY_DONOR_SHA256}, "
                f"got {actual_sha256}."
            )
        return explicit
    return fetch_sipp_2023_financial_asset_donor(
        local_path=SIPP_2023_CHILD_DISABILITY_LOCAL_PATH,
        expected_sha256=SIPP_2023_CHILD_DISABILITY_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES,
    )


def _sha256_file(path: Path) -> str:
    return full_sipp_sha256(path)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float64)


def _assert_pinned_audit(audit: dict[str, int | float]) -> None:
    expected_counts = {
        "raw_rows": _PINNED_RAW_ROWS,
        "december_rows": _PINNED_DECEMBER_ROWS,
        "age_0_rows": _PINNED_AGE_0_ROWS,
        "age_0_observed_item_rows": _PINNED_AGE_0_OBSERVED_ITEM_ROWS,
        "age_0_monthly_ssi_rows": _PINNED_AGE_0_MONTHLY_SSI_ROWS,
        "age_1_rows": _PINNED_AGE_1_ROWS,
        "age_1_valid_rows": _PINNED_AGE_1_VALID_ROWS,
        "age_1_positive_rows": _PINNED_AGE_1_POSITIVE_ROWS,
        "age_1_4_rows": _PINNED_AGE_1_4_ROWS,
        "age_1_4_valid_rows": _PINNED_AGE_1_4_VALID_ROWS,
        "age_1_4_missing_rows": _PINNED_AGE_1_4_MISSING_ROWS,
        "age_1_4_positive_rows": _PINNED_AGE_1_4_POSITIVE_ROWS,
        "age_1_4_child_only_rows": _PINNED_AGE_1_4_CHILD_ONLY_ROWS,
        "age_1_4_monthly_ssi_rows": _PINNED_AGE_1_4_MONTHLY_SSI_ROWS,
        "age_1_4_monthly_ssi_positive_rows": (
            _PINNED_AGE_1_4_MONTHLY_SSI_POSITIVE_ROWS
        ),
        "age_1_4_monthly_ssi_core_positive_rows": (
            _PINNED_AGE_1_4_MONTHLY_SSI_CORE_POSITIVE_ROWS
        ),
        "age_5_14_rows": _PINNED_AGE_5_14_ROWS,
        "age_5_14_valid_rows": _PINNED_AGE_5_14_VALID_ROWS,
        "age_5_14_missing_rows": _PINNED_AGE_5_14_MISSING_ROWS,
        "age_5_14_positive_rows": _PINNED_AGE_5_14_POSITIVE_ROWS,
        "age_5_14_child_only_rows": _PINNED_AGE_5_14_CHILD_ONLY_ROWS,
        "age_5_14_monthly_ssi_rows": _PINNED_AGE_5_14_MONTHLY_SSI_ROWS,
        "reported_monthly_ssi_rows": _PINNED_REPORTED_MONTHLY_SSI_ROWS,
        "reported_monthly_ssi_receipt_rows": (
            _PINNED_REPORTED_MONTHLY_SSI_RECEIPT_ROWS
        ),
        "reported_rdis_positive_rows": _PINNED_REPORTED_RDIS_POSITIVE_ROWS,
        "reported_rdis_positive_receipt_rows": (
            _PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_ROWS
        ),
        "rdis_alt_universe_mismatch_rows": (_PINNED_RDIS_ALT_UNIVERSE_MISMATCH_ROWS),
        "rdis_alt_mismatch_rows": _PINNED_RDIS_ALT_MISMATCH_ROWS,
    }
    mismatches: dict[str, tuple[int | float, int | float]] = {
        key: (expected, audit[key])
        for key, expected in expected_counts.items()
        if audit[key] != expected
    }
    expected_floats = {
        "age_0_weight": _PINNED_AGE_0_WEIGHT,
        "age_0_monthly_ssi_weight": _PINNED_AGE_0_MONTHLY_SSI_WEIGHT,
        "age_1_weight": _PINNED_AGE_1_WEIGHT,
        "age_1_valid_weight": _PINNED_AGE_1_VALID_WEIGHT,
        "age_1_positive_weight": _PINNED_AGE_1_POSITIVE_WEIGHT,
        "age_1_positive_rate": _PINNED_AGE_1_RATE,
        "age_1_4_weight": _PINNED_AGE_1_4_WEIGHT,
        "age_1_4_valid_weight": _PINNED_AGE_1_4_VALID_WEIGHT,
        "age_1_4_missing_weight": _PINNED_AGE_1_4_MISSING_WEIGHT,
        "age_1_4_positive_weight": _PINNED_AGE_1_4_POSITIVE_WEIGHT,
        "age_1_4_positive_rate": _PINNED_AGE_1_4_RATE,
        "age_1_4_child_only_weight": _PINNED_AGE_1_4_CHILD_ONLY_WEIGHT,
        "age_1_4_monthly_ssi_weight": _PINNED_AGE_1_4_MONTHLY_SSI_WEIGHT,
        "age_1_4_monthly_ssi_all_rate": (_PINNED_AGE_1_4_MONTHLY_SSI_ALL_RATE),
        "age_1_4_monthly_ssi_valid_rate": (_PINNED_AGE_1_4_MONTHLY_SSI_VALID_RATE),
        "age_5_14_weight": _PINNED_AGE_5_14_WEIGHT,
        "age_5_14_valid_weight": _PINNED_AGE_5_14_VALID_WEIGHT,
        "age_5_14_missing_weight": _PINNED_AGE_5_14_MISSING_WEIGHT,
        "age_5_14_positive_weight": _PINNED_AGE_5_14_POSITIVE_WEIGHT,
        "age_5_14_positive_rate": _PINNED_AGE_5_14_RATE,
        "age_5_14_child_only_weight": _PINNED_AGE_5_14_CHILD_ONLY_WEIGHT,
        "age_5_14_monthly_ssi_weight": _PINNED_AGE_5_14_MONTHLY_SSI_WEIGHT,
        "age_5_14_monthly_ssi_all_rate": (_PINNED_AGE_5_14_MONTHLY_SSI_ALL_RATE),
        "age_5_14_monthly_ssi_valid_rate": (_PINNED_AGE_5_14_MONTHLY_SSI_VALID_RATE),
        "reported_monthly_ssi_weight": _PINNED_REPORTED_MONTHLY_SSI_WEIGHT,
        "reported_monthly_ssi_receipt_weight": (
            _PINNED_REPORTED_MONTHLY_SSI_RECEIPT_WEIGHT
        ),
        "reported_rdis_positive_weight": _PINNED_REPORTED_RDIS_POSITIVE_WEIGHT,
        "reported_rdis_positive_receipt_weight": (
            _PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_WEIGHT
        ),
        "reported_rdis_positive_receipt_rate": (
            _PINNED_REPORTED_RDIS_POSITIVE_RECEIPT_RATE
        ),
    }
    for key, expected in expected_floats.items():
        actual = float(audit[key])
        if not np.isclose(actual, expected, rtol=_PINNED_FLOAT_RTOL, atol=1e-9):
            mismatches[key] = (expected, actual)
    if mismatches:
        raise ValueError(
            "Pinned SIPP child-disability source audit drifted from the "
            f"reviewed 2023 transform: {mismatches}."
        )


def load_sipp_2023_child_disability_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load December SIPP children and construct the three-feature donor.

    All December persons are retained until residence income is aggregated;
    only then is the valid age-1--14 training universe selected.  Tests pass a tiny
    pipe-delimited fixture and disable artifact pins.
    """

    source_path = Path(path)
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    actual_size = source_path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ValueError(
            "SIPP 2023 child-disability donor failed byte-length verification: "
            f"expected {expected_size_bytes}, got {actual_size}."
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SIPP 2023 child-disability donor failed SHA-256 verification: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

    header = pd.read_csv(source_path, sep="|", nrows=0)
    missing = sorted(set(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"SIPP child-disability donor missing column(s): {missing}.")

    raw_rows = 0
    december_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_path,
        sep="|",
        usecols=list(SIPP_CHILD_DISABILITY_SOURCE_COLUMNS),
        chunksize=int(chunksize),
        low_memory=False,
    ):
        raw_rows += len(chunk)
        month = _numeric(chunk, "MONTHCODE")
        december = chunk.loc[month.eq(12.0)].copy()
        if not december.empty:
            december_parts.append(december)
    if not december_parts:
        raise ValueError("SIPP child-disability donor contains no December rows.")
    december = pd.concat(december_parts, ignore_index=True)

    identity_columns = ["SSUID", "ERESIDENCEID", "PNUM"]
    if december.loc[:, identity_columns].isna().any(axis=None):
        raise ValueError("SIPP child-disability December identity is incomplete.")
    if december.duplicated(identity_columns).any():
        raise ValueError("SIPP child-disability December identity is not unique.")

    age = _numeric(december, "TAGE")
    sex = _numeric(december, "ESEX")
    weight = _numeric(december, "WPFINWGT")
    valid_base = (
        age.between(0.0, 120.0, inclusive="both")
        & sex.isin([1.0, 2.0])
        & np.isfinite(weight)
        & weight.ge(0.0)
    )
    if not valid_base.all():
        rows = np.flatnonzero(~valid_base.to_numpy())[:5].tolist()
        raise ValueError(
            "SIPP child-disability age, sex, and weights must be valid and "
            "weights nonnegative; "
            f"invalid December row(s): {rows}."
        )

    residence_key = pd.MultiIndex.from_frame(december.loc[:, ["SSUID", "ERESIDENCEID"]])
    monthly_income = (
        _numeric(december, "TPTOTINC")
        .fillna(0.0)
        .clip(lower=0.0)
        .where(age.ge(15.0), 0.0)
    )
    annual_household_income = (
        pd.Series(monthly_income.to_numpy() * 12.0, index=december.index)
        .groupby(residence_key, sort=False)
        .transform("sum")
    )

    outcome = pd.Series(
        np.column_stack(
            [_numeric(december, column).eq(1.0) for column in _OUTCOME_SOURCE_COLUMNS]
        ).any(axis=1),
        index=december.index,
    )
    core_outcome = pd.Series(
        np.column_stack(
            [
                _numeric(december, column).eq(1.0)
                for column in _CORE_CHILD_SOURCE_COLUMNS
            ]
        ).any(axis=1),
        index=december.index,
    )
    child_specific_outcome = pd.Series(
        np.column_stack(
            [
                _numeric(december, column).eq(1.0)
                for column in _CHILD_SPECIFIC_SOURCE_COLUMNS
            ]
        ).any(axis=1),
        index=december.index,
    )
    child_only_outcome = child_specific_outcome & ~core_outcome
    monthly_ssi = _numeric(december, "RSSI_MNYN")
    monthly_ssi_allocation = _numeric(december, "ASSI_MNYN")
    received_monthly_ssi = monthly_ssi.eq(1.0)
    reported_monthly_ssi = monthly_ssi_allocation.eq(1.0) & monthly_ssi.isin([1.0, 2.0])
    household_status = _numeric(december, "THHLDSTATUS")
    rdis_alt = _numeric(december, "RDIS_ALT")
    expected_rdis_alt_universe = age.ge(1.0) & household_status.isin([1.0, 2.0])
    valid_rdis_alt = rdis_alt.isin([1.0, 2.0])
    rdis_alt_universe_mismatch = expected_rdis_alt_universe.ne(valid_rdis_alt)
    if rdis_alt_universe_mismatch.any():
        rows = np.flatnonzero(rdis_alt_universe_mismatch.to_numpy())[:5].tolist()
        raise ValueError(
            "SIPP child-disability RDIS_ALT observed universe drifted from "
            "TAGE >= 1 and THHLDSTATUS in [1, 2]; "
            f"mismatched December row(s): {rows}."
        )

    age_0 = age.eq(0.0)
    age_1 = age.eq(1.0)
    age_1_4 = age.between(1.0, 4.0, inclusive="both")
    age_5_14 = age.between(5.0, 14.0, inclusive="both")
    model_band = age.between(_MIN_MODEL_AGE, _MAX_MODEL_AGE, inclusive="both")
    rdis_alt_mismatch = model_band & valid_rdis_alt & outcome.ne(rdis_alt.eq(1.0))
    if rdis_alt_mismatch.any():
        rows = np.flatnonzero(rdis_alt_mismatch.to_numpy())[:5].tolist()
        raise ValueError(
            "Computed SIPP child-disability item union disagrees with RDIS_ALT; "
            f"mismatched valid age-1--14 row(s): {rows}."
        )
    model_eligible = model_band & valid_rdis_alt & weight.gt(0.0)
    severity_label_observed = model_eligible & reported_monthly_ssi
    severity_training = severity_label_observed & outcome
    severity_training_weight = float(weight.loc[severity_training].sum())
    severity_receipt_weight = float(
        weight.loc[severity_training & received_monthly_ssi].sum()
    )

    donor_columns: dict[str, np.ndarray] = {
        "age": age.loc[model_eligible].to_numpy(dtype=np.float64),
        "is_female": sex.loc[model_eligible].eq(2.0).to_numpy(),
        _HOUSEHOLD_INCOME_COLUMN: annual_household_income.loc[model_eligible].to_numpy(
            dtype=np.float64
        ),
        _GENERAL_OUTPUT: outcome.loc[model_eligible].to_numpy(),
        _DONOR_WEIGHT_COLUMN: weight.loc[model_eligible].to_numpy(dtype=np.float64),
        _MONTHLY_SSI_RECEIVED_COLUMN: received_monthly_ssi.loc[
            model_eligible
        ].to_numpy(),
        _MONTHLY_SSI_REPORTED_COLUMN: reported_monthly_ssi.loc[
            model_eligible
        ].to_numpy(),
    }
    for column in SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS:
        donor_columns[column] = (
            _numeric(december, column).loc[model_eligible].eq(1.0).to_numpy()
        )
    donor = pd.DataFrame(donor_columns).reset_index(drop=True)
    if donor.empty or donor[_GENERAL_OUTPUT].nunique() != 2:
        raise ValueError(
            "SIPP child-disability donor must contain both outcome classes."
        )

    def band_audit(prefix: str, band: pd.Series) -> dict[str, int | float]:
        valid = band & valid_rdis_alt
        missing = band & ~valid_rdis_alt
        positive = valid & outcome
        monthly_ssi = band & received_monthly_ssi
        total_weight = float(weight.loc[band].sum())
        valid_weight = float(weight.loc[valid].sum())
        positive_weight = float(weight.loc[positive].sum())
        monthly_ssi_weight = float(weight.loc[monthly_ssi].sum())

        def rate(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > 0.0 else 0.0

        return {
            f"{prefix}_rows": int(band.sum()),
            f"{prefix}_weight": total_weight,
            f"{prefix}_valid_rows": int(valid.sum()),
            f"{prefix}_valid_weight": valid_weight,
            f"{prefix}_missing_rows": int(missing.sum()),
            f"{prefix}_missing_weight": float(weight.loc[missing].sum()),
            f"{prefix}_positive_rows": int(positive.sum()),
            f"{prefix}_positive_weight": positive_weight,
            f"{prefix}_positive_rate": rate(positive_weight, valid_weight),
            f"{prefix}_child_only_rows": int((valid & child_only_outcome).sum()),
            f"{prefix}_child_only_weight": float(
                weight.loc[valid & child_only_outcome].sum()
            ),
            f"{prefix}_monthly_ssi_rows": int(monthly_ssi.sum()),
            f"{prefix}_monthly_ssi_weight": monthly_ssi_weight,
            f"{prefix}_monthly_ssi_all_rate": rate(monthly_ssi_weight, total_weight),
            f"{prefix}_monthly_ssi_valid_rate": rate(
                monthly_ssi_weight,
                valid_weight,
            ),
        }

    audit: dict[str, int | float] = {
        "raw_rows": raw_rows,
        "december_rows": len(december),
        "age_0_rows": int(age_0.sum()),
        "age_0_weight": float(weight.loc[age_0].sum()),
        "age_0_observed_item_rows": int(
            december.loc[age_0, [*_OUTCOME_SOURCE_COLUMNS, "RDIS_ALT"]]
            .notna()
            .any(axis=1)
            .sum()
        ),
        "age_0_monthly_ssi_rows": int((age_0 & received_monthly_ssi).sum()),
        "age_0_monthly_ssi_weight": float(
            weight.loc[age_0 & received_monthly_ssi].sum()
        ),
        **band_audit("age_1", age_1),
        **band_audit("age_1_4", age_1_4),
        "age_1_4_monthly_ssi_positive_rows": int(
            (age_1_4 & valid_rdis_alt & received_monthly_ssi & outcome).sum()
        ),
        "age_1_4_monthly_ssi_core_positive_rows": int(
            (age_1_4 & valid_rdis_alt & received_monthly_ssi & core_outcome).sum()
        ),
        **band_audit("age_5_14", age_5_14),
        "reported_monthly_ssi_rows": int(severity_label_observed.sum()),
        "reported_monthly_ssi_weight": float(weight.loc[severity_label_observed].sum()),
        "reported_monthly_ssi_receipt_rows": int(
            (severity_label_observed & received_monthly_ssi).sum()
        ),
        "reported_monthly_ssi_receipt_weight": float(
            weight.loc[severity_label_observed & received_monthly_ssi].sum()
        ),
        "reported_rdis_positive_rows": int(severity_training.sum()),
        "reported_rdis_positive_weight": severity_training_weight,
        "reported_rdis_positive_receipt_rows": int(
            (severity_training & received_monthly_ssi).sum()
        ),
        "reported_rdis_positive_receipt_weight": severity_receipt_weight,
        "reported_rdis_positive_receipt_rate": float(
            severity_receipt_weight / severity_training_weight
            if severity_training_weight > 0.0
            else 0.0
        ),
        "rdis_alt_universe_mismatch_rows": int(rdis_alt_universe_mismatch.sum()),
        "rdis_alt_mismatch_rows": int(rdis_alt_mismatch.sum()),
    }
    pinned_transform = bool(
        actual_size == SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES
        and expected_sha256 == SIPP_2023_CHILD_DISABILITY_DONOR_SHA256
    )
    if pinned_transform:
        _assert_pinned_audit(audit)
    audit["pinned_transform"] = int(pinned_transform)
    donor.attrs["source_audit"] = audit
    return donor


@dataclass(frozen=True)
class _WeightedLogisticModel:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        raw = _logistic_feature_array(features)
        standardized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(raw)), standardized])
        return _sigmoid(design @ self.coefficients)


def _logistic_feature_array(features: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(SIPP_CHILD_DISABILITY_MODEL_PREDICTORS) - set(features))
    if missing:
        raise ValueError(
            f"Child-disability classifier missing predictor column(s): {missing}."
        )
    age = pd.to_numeric(features["age"], errors="coerce").to_numpy(np.float64)
    female = pd.to_numeric(features["is_female"], errors="coerce").to_numpy(np.float64)
    income = pd.to_numeric(
        features[_HOUSEHOLD_INCOME_COLUMN], errors="coerce"
    ).to_numpy(np.float64)
    valid = (
        np.isfinite(age)
        & np.isfinite(female)
        & np.isin(female, [0.0, 1.0])
        & np.isfinite(income)
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            "Child-disability classifier predictors must be finite and sex "
            f"must be boolean; invalid row(s): {rows}."
        )
    return np.column_stack([age, female, np.log1p(np.maximum(income, 0.0))])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_weighted_logistic(donor: pd.DataFrame) -> _WeightedLogisticModel:
    required = {
        *SIPP_CHILD_DISABILITY_MODEL_PREDICTORS,
        _GENERAL_OUTPUT,
        _DONOR_WEIGHT_COLUMN,
    }
    missing = sorted(required - set(donor))
    if missing:
        raise ValueError(f"SIPP child-disability donor missing column(s): {missing}.")

    raw = _logistic_feature_array(donor)
    outcome = pd.to_numeric(donor[_GENERAL_OUTPUT], errors="coerce").to_numpy(
        np.float64
    )
    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        np.float64
    )
    if not (np.isfinite(outcome) & np.isin(outcome, [0.0, 1.0])).all():
        raise ValueError("SIPP child-disability outcome must be boolean.")
    if np.unique(outcome).size != 2:
        raise ValueError("SIPP child-disability outcome must contain both classes.")
    if not (np.isfinite(weights) & (weights > 0.0)).all():
        raise ValueError(
            "SIPP child-disability model weights must be finite and positive."
        )

    normalized_weights = weights * (len(weights) / weights.sum())
    feature_mean = np.average(raw, axis=0, weights=normalized_weights)
    centered = raw - feature_mean
    variance = np.average(centered**2, axis=0, weights=normalized_weights)
    feature_scale = np.sqrt(np.maximum(variance, 1e-12))
    design = np.column_stack([np.ones(len(raw)), centered / feature_scale])

    weighted_rate = float(np.average(outcome, weights=normalized_weights))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = np.log(weighted_rate / (1.0 - weighted_rate))
    ridge = np.diag([0.0, 1e-6, 1e-6, 1e-6])
    for _ in range(100):
        probability = _sigmoid(design @ coefficients)
        bernoulli_variance = np.maximum(probability * (1.0 - probability), 1e-8)
        working_weight = normalized_weights * bernoulli_variance
        working_response = (
            design @ coefficients + (outcome - probability) / bernoulli_variance
        )
        lhs = design.T @ (working_weight[:, None] * design) + ridge
        rhs = design.T @ (working_weight * working_response)
        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return _WeightedLogisticModel(coefficients, feature_mean, feature_scale)


def _decoded_strings(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _stable_person_keys(person: pd.DataFrame) -> pd.Series:
    if _PERSON_SOURCE_ID_COLUMN in person:
        if person[_PERSON_SOURCE_ID_COLUMN].isna().any():
            raise ValueError("Child-disability person_source_id must be complete.")
        return _decoded_strings(person[_PERSON_SOURCE_ID_COLUMN])
    source_columns = ("source_year", "source_household_id", "source_person_id")
    if set(source_columns) <= set(person):
        if person.loc[:, source_columns].isna().any(axis=None):
            raise ValueError("Child-disability source identity must be complete.")
        return (
            person["source_year"].astype(str)
            + ":"
            + person["source_household_id"].astype(str)
            + ":"
            + person["source_person_id"].astype(str)
        )
    if "person_id" not in person or person["person_id"].isna().any():
        raise ValueError("Child-disability assignment requires stable person ids.")
    return person["person_id"].astype(str)


def _stable_uniform_draws(keys: pd.Series, *, seed: int, stream: str) -> np.ndarray:
    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{int(seed)}:{US_CHILD_DISABILITY_STAGE_NAME}:{stream}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _recipient_features(
    frame: Frame,
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    person = frame.table("person")
    required = {
        "person_id",
        "person_household_id",
        "age",
        "is_female",
        "employment_income_before_lsr",
        _GENERAL_OUTPUT,
    }
    missing = sorted(required - set(person))
    if missing:
        raise ValueError(
            f"US child-disability receiver missing person column(s): {missing}."
        )
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    female = pd.to_numeric(person["is_female"], errors="coerce").to_numpy(np.float64)
    earnings = pd.to_numeric(
        person["employment_income_before_lsr"], errors="coerce"
    ).to_numpy(np.float64)
    valid = (
        np.isfinite(age)
        & (age >= 0.0)
        & (age <= 120.0)
        & np.isfinite(female)
        & np.isin(female, [0.0, 1.0])
        & np.isfinite(earnings)
        & (earnings >= 0.0)
    )
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            "US child-disability receiver age, sex, and employment income "
            f"must be valid; invalid row(s): {rows}."
        )
    if person["person_household_id"].isna().any():
        raise ValueError("US child-disability receiver household links are missing.")

    household = person["person_household_id"]
    features = pd.DataFrame(
        {
            "age": age,
            "is_female": female,
            _HOUSEHOLD_INCOME_COLUMN: (
                pd.Series(np.where(age >= 15.0, earnings, 0.0), index=person.index)
                .groupby(household, sort=False)
                .transform("sum")
            ),
        },
        index=person.index,
    )
    keys = _stable_person_keys(person)
    consistency = (
        pd.DataFrame({"key": keys, "age": age, "is_female": female}, index=person.index)
        .groupby("key", sort=False)[["age", "is_female"]]
        .nunique()
    )
    inconsistent = consistency.index[(consistency > 1).any(axis=1)].tolist()
    if inconsistent:
        raise ValueError(
            "US child-disability support clones disagree on age or sex for "
            f"source id(s): {inconsistent[:5]}."
        )

    if has_support_role_metadata(person, entity="person"):
        try:
            roles = support_role_series(person, entity="person")
        except ValueError as exc:
            raise ValueError(
                f"Child-disability receiver has invalid support-role metadata: {exc}"
            ) from exc
    else:
        roles = pd.Series(_ASEC_SUPPORT_CHANNEL, index=person.index)
    role_rows = pd.DataFrame({"source_id": keys, "role": roles})
    duplicate_roles = role_rows.duplicated(["source_id", "role"], keep=False)
    if duplicate_roles.any():
        bad = (
            role_rows.loc[duplicate_roles, ["source_id", "role"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            "Child-disability source units carry duplicated same-role rows; "
            f"invalid source role(s): {list(bad)[:5]}."
        )

    order = pd.DataFrame({"key": keys}, index=person.index)
    order["role_priority"] = roles.map(
        {_ASEC_SUPPORT_CHANNEL: 0, _PUF_SUPPORT_CHANNEL: 1}
    )
    if order["role_priority"].isna().any():
        unexpected = sorted(set(roles[order["role_priority"].isna()].astype(str)))
        raise ValueError(
            f"Child-disability receiver has unsupported support role(s): {unexpected}."
        )
    order["person_key"] = person["person_id"].astype(str)
    canonical_index = (
        order.sort_values(["key", "role_priority", "person_key"], kind="mergesort")
        .drop_duplicates("key", keep="first")
        .index
    )
    return features, keys, canonical_index


def _calibrate_probabilities(
    base_probability: np.ndarray,
    *,
    current_true: np.ndarray,
    weights: np.ndarray,
    target_share: float,
) -> np.ndarray:
    """Shift probability logits so expected final OR-share hits a target."""

    probability = np.clip(
        np.asarray(base_probability, dtype=np.float64), 1e-9, 1 - 1e-9
    )
    current_true = np.asarray(current_true, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Child-disability calibration requires positive weight.")
    current_share = float(weights[current_true].sum()) / total
    if current_share >= target_share:
        return np.where(current_true, 1.0, 0.0)

    logit = np.log(probability / (1.0 - probability))

    def expected(shift: float) -> float:
        added = _sigmoid(logit + shift)
        final_probability = np.where(current_true, 1.0, added)
        return float(np.dot(weights, final_probability) / total)

    low, high = -30.0, 30.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if expected(midpoint) < target_share:
            low = midpoint
        else:
            high = midpoint
    shifted = _sigmoid(logit + (low + high) / 2.0)
    return np.where(current_true, 1.0, shifted)


@dataclass(frozen=True)
class _ChildSSISeverityModel:
    classifier: RandomForestClassifier
    receipt_anchor: float

    def predict_score(self, profiles: pd.DataFrame) -> np.ndarray:
        values = _severity_predictor_array(profiles)
        probabilities = self.classifier.predict_proba(values)
        positive = np.flatnonzero(self.classifier.classes_ == 1)
        if positive.size != 1:
            raise ValueError(
                "Child SSI severity classifier has no unique receipt-positive class."
            )
        result = probabilities[:, int(positive[0])]
        if not np.isfinite(result).all():
            raise ValueError("Child SSI severity classifier produced nonfinite scores.")
        return result


def _severity_predictor_array(features: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS) - set(features))
    if missing:
        raise ValueError(
            f"Child SSI severity model missing predictor column(s): {missing}."
        )
    values = (
        features.loc[:, list(SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not (np.isfinite(values) & np.isin(values, [0.0, 1.0])).all():
        raise ValueError("Child SSI severity battery predictors must be boolean.")
    return values


def _fit_child_ssi_severity_model(donor: pd.DataFrame) -> _ChildSSISeverityModel:
    required = {
        *SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS,
        _GENERAL_OUTPUT,
        _MONTHLY_SSI_RECEIVED_COLUMN,
        _MONTHLY_SSI_REPORTED_COLUMN,
        _DONOR_WEIGHT_COLUMN,
    }
    missing = sorted(required - set(donor))
    if missing:
        raise ValueError(f"SIPP child SSI severity donor missing column(s): {missing}.")
    general = pd.to_numeric(donor[_GENERAL_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    received = pd.to_numeric(
        donor[_MONTHLY_SSI_RECEIVED_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    reported = pd.to_numeric(
        donor[_MONTHLY_SSI_REPORTED_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64)
    weights = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    for name, values in (
        (_GENERAL_OUTPUT, general),
        (_MONTHLY_SSI_RECEIVED_COLUMN, received),
        (_MONTHLY_SSI_REPORTED_COLUMN, reported),
    ):
        if not (np.isfinite(values) & np.isin(values, [0.0, 1.0])).all():
            raise ValueError(f"SIPP child SSI severity {name} must be boolean.")
    if not (np.isfinite(weights) & (weights > 0.0)).all():
        raise ValueError("SIPP child SSI severity weights must be finite and positive.")

    training = general.astype(bool) & reported.astype(bool)
    labels = received[training].astype(np.int8)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError(
            "SIPP child SSI severity training data must contain reported "
            "receipt and nonreceipt among RDIS_ALT-positive children."
        )
    training_weights = weights[training]
    receipt_anchor = float(np.average(labels, weights=training_weights))
    if not 0.0 < receipt_anchor < 1.0:
        raise ValueError(
            "SIPP child SSI receipt anchor must be strictly between 0 and 1."
        )
    audit = donor.attrs.get("source_audit", {})
    if bool(audit.get("pinned_transform")) and not np.isclose(
        receipt_anchor,
        US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR,
        rtol=_PINNED_FLOAT_RTOL,
        atol=1e-12,
    ):
        raise ValueError(
            "Pinned SIPP child SSI receipt anchor drifted: "
            f"expected {US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR}, "
            f"got {receipt_anchor}."
        )

    classifier = RandomForestClassifier(
        n_estimators=_SEVERITY_N_ESTIMATORS,
        random_state=_SEVERITY_MODEL_SEED,
        n_jobs=1,
    )
    classifier.fit(
        _severity_predictor_array(donor.loc[training]),
        labels,
        sample_weight=training_weights,
    )
    return _ChildSSISeverityModel(classifier, receipt_anchor)


def _hot_deck_child_battery_profiles(
    donor: pd.DataFrame,
    *,
    receiver_features: pd.DataFrame,
    receiver_keys: pd.Series,
    receiver_general: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Draw exact-age/sex RDIS_ALT-positive battery profiles by source id."""

    required = {
        "age",
        "is_female",
        _GENERAL_OUTPUT,
        _DONOR_WEIGHT_COLUMN,
        *SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS,
    }
    missing = sorted(required - set(donor))
    if missing:
        raise ValueError(f"SIPP child SSI profile donor missing column(s): {missing}.")
    donor_age = pd.to_numeric(donor["age"], errors="coerce").to_numpy(dtype=np.float64)
    donor_female = pd.to_numeric(donor["is_female"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    donor_general = pd.to_numeric(donor[_GENERAL_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    donor_weight = pd.to_numeric(donor[_DONOR_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_donor = (
        np.isfinite(donor_age)
        & np.equal(donor_age, np.floor(donor_age))
        & (donor_age >= _MIN_MODEL_AGE)
        & (donor_age <= _MAX_MODEL_AGE)
        & np.isfinite(donor_female)
        & np.isin(donor_female, [0.0, 1.0])
        & np.isfinite(donor_general)
        & np.isin(donor_general, [0.0, 1.0])
        & np.isfinite(donor_weight)
        & (donor_weight > 0.0)
    )
    if not valid_donor.all():
        rows = np.flatnonzero(~valid_donor)[:5].tolist()
        raise ValueError(
            "SIPP child SSI profile donor has invalid age, sex, outcome, or "
            f"weight at row(s) {rows}."
        )
    donor_profiles = _severity_predictor_array(donor)
    positive_donor = donor_general.astype(bool)

    receiver_age = pd.to_numeric(receiver_features["age"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    receiver_female = pd.to_numeric(
        receiver_features["is_female"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    receiver_general = np.asarray(receiver_general, dtype=bool)
    relevant = receiver_general
    valid_receiver_age = (
        np.isfinite(receiver_age)
        & np.equal(receiver_age, np.floor(receiver_age))
        & (receiver_age >= 0.0)
        & (receiver_age <= _MAX_MODEL_AGE)
    )
    if not valid_receiver_age[relevant].all():
        rows = np.flatnonzero(relevant & ~valid_receiver_age)[:5].tolist()
        raise ValueError(
            "Child SSI profile receivers must have whole ages 0--14; "
            f"invalid row(s): {rows}."
        )

    profiles = np.zeros(
        (len(receiver_features), len(SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS)),
        dtype=np.float64,
    )
    transported_age = np.where(receiver_age == 0.0, 1.0, receiver_age)
    for age_value in range(_MIN_MODEL_AGE, _MAX_MODEL_AGE + 1):
        for female_value in (0.0, 1.0):
            receiver_cell = (
                relevant
                & (transported_age == float(age_value))
                & (receiver_female == female_value)
            )
            if not receiver_cell.any():
                continue
            donor_cell = (
                positive_donor
                & (donor_age == float(age_value))
                & (donor_female == female_value)
            )
            if not donor_cell.any():
                raise ValueError(
                    "SIPP child SSI profile donor has no RDIS_ALT-positive "
                    f"support for age {age_value}, female={bool(female_value)}."
                )
            pool_positions = np.flatnonzero(donor_cell)
            cumulative = np.cumsum(donor_weight[pool_positions], dtype=np.float64)
            draws = _stable_uniform_draws(
                receiver_keys.loc[receiver_cell],
                seed=seed,
                stream="ssi_battery_profile",
            )
            selected = np.searchsorted(
                cumulative,
                draws * cumulative[-1],
                side="right",
            )
            selected = np.minimum(selected, len(pool_positions) - 1)
            profiles[receiver_cell] = donor_profiles[pool_positions[selected]]
    return pd.DataFrame(
        profiles,
        index=receiver_features.index,
        columns=list(SIPP_CHILD_SSI_SEVERITY_MODEL_PREDICTORS),
    )


def _child_ssi_criteria_assignment(
    donor: pd.DataFrame,
    *,
    canonical_features: pd.DataFrame,
    canonical_keys: pd.Series,
    canonical_general: np.ndarray,
    all_keys: pd.Series,
    all_general: np.ndarray,
    all_weights: np.ndarray,
    under_15: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return the seeded receipt-anchored severe subset of general disability."""

    model = _fit_child_ssi_severity_model(donor)
    profiles = _hot_deck_child_battery_profiles(
        donor,
        receiver_features=canonical_features,
        receiver_keys=canonical_keys,
        receiver_general=canonical_general,
        seed=seed,
    )
    scores = np.zeros(len(canonical_features), dtype=np.float64)
    if canonical_general.any():
        scores[canonical_general] = model.predict_score(profiles.loc[canonical_general])
    score_by_key = pd.Series(scores, index=canonical_keys.to_numpy())
    all_scores = all_keys.map(score_by_key).to_numpy(dtype=np.float64)
    eligible = under_15 & np.asarray(all_general, dtype=bool)
    if not eligible.any():
        # A sliced or very small receiver can legitimately contain no general
        # disability. The severe subset is then empty by construction; release
        # frames still have to pass the separate nonconstant/share gate.
        return np.zeros(len(all_keys), dtype=bool)
    probability = _calibrate_probabilities(
        all_scores[eligible],
        current_true=np.zeros(int(eligible.sum()), dtype=bool),
        weights=np.asarray(all_weights, dtype=np.float64)[eligible],
        target_share=model.receipt_anchor,
    )
    draws = _stable_uniform_draws(
        all_keys.loc[eligible],
        seed=seed,
        stream="ssi_severity",
    )
    result = np.zeros(len(all_keys), dtype=bool)
    result[eligible] = draws < probability
    return result


def with_us_child_disability_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
) -> Frame:
    """Assign under-15 general disability and its receipt-anchored SSI subset."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US child-disability inputs require the US schema.")
    del time_period  # The source and target vintages are fixed in the manifest.
    us_child_disability_stage_spec()
    person = frame.table("person")
    features, keys, canonical_index = _recipient_features(frame)
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    current_numeric = pd.to_numeric(person[_GENERAL_OUTPUT], errors="coerce").to_numpy(
        np.float64
    )
    under_15 = (age >= 0.0) & (age <= _MAX_MODEL_AGE)
    valid_child = np.isfinite(current_numeric) & np.isin(current_numeric, [0.0, 1.0])
    if not valid_child[under_15].all():
        raise ValueError(
            "US child-disability input is_disabled must be boolean for under-15 rows."
        )
    criteria_column_existed = _CRITERIA_OUTPUT in person
    criteria_numeric = (
        pd.to_numeric(person[_CRITERIA_OUTPUT], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if criteria_column_existed
        else np.zeros(len(person), dtype=np.float64)
    )
    valid_criteria = np.isfinite(criteria_numeric) & np.isin(
        criteria_numeric,
        [0.0, 1.0],
    )
    if not valid_criteria[under_15].all():
        raise ValueError(
            "US child-disability input meets_ssi_disability_criteria must be "
            "boolean for under-15 rows."
        )
    # Adult storage is outside this stage's ownership.  Keep an internal
    # child-only boolean vector and never normalize or assign the adult slice.
    current = np.zeros(len(person), dtype=bool)
    current[under_15] = current_numeric[under_15].astype(bool)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0.0)).all():
        raise ValueError("US child-disability person weights must be nonnegative.")

    model = _fit_weighted_logistic(sipp_donor)
    canonical = features.loc[canonical_index]
    canonical_probability = model.predict_probability(canonical)
    probability_by_key = pd.Series(
        canonical_probability,
        index=keys.loc[canonical_index].to_numpy(),
    )
    model_probability = keys.map(probability_by_key).to_numpy(np.float64)
    if not np.isfinite(model_probability).all():
        raise ValueError("Child-disability model did not cover every source person.")

    donor_outcome = pd.to_numeric(
        sipp_donor[_GENERAL_OUTPUT], errors="coerce"
    ).to_numpy(np.float64)
    donor_weight = pd.to_numeric(
        sipp_donor[_DONOR_WEIGHT_COLUMN], errors="coerce"
    ).to_numpy(np.float64)
    donor_age = pd.to_numeric(sipp_donor["age"], errors="coerce").to_numpy(np.float64)

    def donor_rate(low: int, high: int) -> float:
        donor_band = (donor_age >= low) & (donor_age <= high)
        if not donor_band.any() or float(donor_weight[donor_band].sum()) <= 0.0:
            raise ValueError(
                "SIPP child-disability donor has no positive-weight support for "
                f"ages {low}--{high}."
            )
        return float(
            np.average(donor_outcome[donor_band], weights=donor_weight[donor_band])
        )

    result = current.copy()

    def assign_modeled_band(
        *,
        low: int,
        high: int,
        target_rate: float,
        stream: str,
    ) -> None:
        receiver_band = (age >= low) & (age <= high)
        if not receiver_band.any():
            return
        probability = _calibrate_probabilities(
            model_probability[receiver_band],
            current_true=current[receiver_band],
            weights=weights[receiver_band],
            target_share=target_rate,
        )
        draws = _stable_uniform_draws(
            keys.loc[receiver_band],
            seed=seed,
            stream=stream,
        )
        result[receiver_band] |= draws < probability

    assign_modeled_band(
        low=1,
        high=4,
        target_rate=donor_rate(1, 4),
        stream="age_1_4",
    )
    assign_modeled_band(
        low=5,
        high=14,
        target_rate=donor_rate(5, 14),
        stream="age_5_14",
    )

    age_0 = age == 0.0
    if age_0.any():
        probability_0 = _calibrate_probabilities(
            np.full(int(age_0.sum()), US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE),
            current_true=current[age_0],
            weights=weights[age_0],
            target_share=US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE,
        )
        draws_0 = _stable_uniform_draws(keys.loc[age_0], seed=seed, stream="age_0")
        result[age_0] |= draws_0 < probability_0

    # The operation is additive: no pre-existing True is ever cleared.
    if np.any(current[under_15] & ~result[under_15]):
        raise AssertionError("Child-disability stage cleared an existing True value.")
    criteria_result = _child_ssi_criteria_assignment(
        sipp_donor,
        canonical_features=features.loc[canonical_index],
        canonical_keys=keys.loc[canonical_index],
        canonical_general=result[canonical_index],
        all_keys=keys,
        all_general=result,
        all_weights=weights,
        under_15=under_15,
        seed=int(seed),
    )
    if np.any(criteria_result[under_15] & ~result[under_15]):
        raise AssertionError(
            "Child SSI disability criteria escaped the general-disability subset."
        )
    clone_assignments = pd.DataFrame(
        {
            "source_id": keys,
            _GENERAL_OUTPUT: result,
            _CRITERIA_OUTPUT: criteria_result,
        }
    )
    divergence = clone_assignments.groupby("source_id", sort=False)[
        [_GENERAL_OUTPUT, _CRITERIA_OUTPUT]
    ].nunique()
    if (divergence > 1).any(axis=None):
        bad = divergence.index[(divergence > 1).any(axis=1)].tolist()
        raise AssertionError(
            "Child-disability source clones received different assignments; "
            f"source id(s): {bad[:5]}."
        )
    if (
        criteria_column_existed
        and np.array_equal(
            current[under_15],
            result[under_15],
        )
        and np.array_equal(
            criteria_numeric[under_15].astype(bool),
            criteria_result[under_15],
        )
    ):
        return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    if not criteria_column_existed:
        # The later SSI-criteria stage owns adults and will replace this
        # neutral surface. Initializing the newly introduced whole-person
        # column avoids NaNs between the two stages.
        tables["person"][_CRITERIA_OUTPUT] = False
    tables["person"].loc[under_15, _GENERAL_OUTPUT] = result[under_15]
    tables["person"].loc[under_15, _CRITERIA_OUTPUT] = criteria_result[under_15]
    adult = age >= 15.0
    if not tables["person"].loc[adult, person.columns].equals(person.loc[adult]):
        raise AssertionError("Child-disability stage changed a row aged 15 or over.")
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_child_disability_summary(frame: Frame) -> dict[str, object]:
    """Return general and receipt-anchored child-disability diagnostics."""

    person = frame.table("person")
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(np.float64)
    values = pd.to_numeric(person[_GENERAL_OUTPUT], errors="coerce").to_numpy(
        np.float64
    )
    criteria_values = (
        pd.to_numeric(person[_CRITERIA_OUTPUT], errors="coerce").to_numpy(np.float64)
        if _CRITERIA_OUTPUT in person
        else np.zeros(len(person), dtype=np.float64)
    )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    if not (np.isfinite(weights) & (weights >= 0.0)).all():
        raise ValueError("Child-disability gate weights must be nonnegative.")
    owned = (age >= 0.0) & (age <= _MAX_MODEL_AGE)
    finite_boolean = np.isfinite(values) & np.isin(values, [0.0, 1.0])
    disabled = finite_boolean & (values == 1.0)
    criteria_finite_boolean = np.isfinite(criteria_values) & np.isin(
        criteria_values,
        [0.0, 1.0],
    )
    severe = criteria_finite_boolean & (criteria_values == 1.0)

    def band_summary(mask: np.ndarray) -> tuple[float, float, int]:
        band_weight = float(weights[mask].sum())
        disabled_weight = float(weights[mask & disabled].sum())
        share = disabled_weight / band_weight if band_weight > 0.0 else 0.0
        unique_count = int(pd.Series(values[mask & np.isfinite(values)]).nunique())
        return share, band_weight, unique_count

    age_0 = age == 0.0
    age_1_4 = (age >= 1.0) & (age <= _MAX_EARLY_CHILD_AGE)
    age_5_14 = (age >= 5.0) & (age <= _MAX_MODEL_AGE)
    share_0, weight_0, unique_0 = band_summary(age_0)
    share_1_4, weight_1_4, unique_1_4 = band_summary(age_1_4)
    share_5_14, weight_5_14, unique_5_14 = band_summary(age_5_14)
    general_disabled_weight = float(weights[owned & disabled].sum())
    severe_weight = float(weights[owned & severe].sum())
    criteria_within_general_share = (
        severe_weight / general_disabled_weight
        if general_disabled_weight > 0.0
        else 0.0
    )
    return {
        "age_0_disabled_share": share_0,
        "age_0_weight": weight_0,
        "age_0_unique_count": unique_0,
        "age_0_fallback_rate": US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE,
        "age_0_share_band": list(US_CHILD_DISABILITY_AGE_0_SHARE_BAND),
        "age_1_4_disabled_share": share_1_4,
        "age_1_4_weight": weight_1_4,
        "age_1_4_unique_count": unique_1_4,
        "age_1_4_sipp_observed_rate": US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE,
        "age_1_4_share_band": list(US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND),
        "age_5_14_disabled_share": share_5_14,
        "age_5_14_weight": weight_5_14,
        "age_5_14_unique_count": unique_5_14,
        "age_5_14_sipp_observed_rate": US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE,
        "age_5_14_share_band": list(US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND),
        "child_general_disabled_weight": general_disabled_weight,
        "child_ssi_criteria_weight": severe_weight,
        "child_ssi_criteria_within_general_share": criteria_within_general_share,
        "child_ssi_criteria_receipt_anchor": US_CHILD_SSI_SEVERITY_RECEIPT_ANCHOR,
        "child_ssi_criteria_share_band": list(US_CHILD_SSI_SEVERITY_SHARE_BAND),
        "child_ssi_criteria_unique_count": int(
            pd.Series(criteria_values[owned & np.isfinite(criteria_values)]).nunique()
        ),
        "criteria_outside_general_count": int(
            np.count_nonzero(owned & severe & ~disabled)
        ),
        "missing_or_nonfinite_count": int(
            np.count_nonzero(owned & ~np.isfinite(values))
        ),
        "invalid_boolean_count": int(
            np.count_nonzero(owned & np.isfinite(values) & ~finite_boolean)
        ),
        "criteria_missing_or_nonfinite_count": int(
            np.count_nonzero(owned & ~np.isfinite(criteria_values))
        ),
        "criteria_invalid_boolean_count": int(
            np.count_nonzero(
                owned & np.isfinite(criteria_values) & ~criteria_finite_boolean
            )
        ),
    }


def us_child_disability_signal_gate(
    frame: Frame,
    *,
    input_frame: Frame | None = None,
) -> GateResult:
    """Require plausible child shares and, when supplied, exact adult parity."""

    person = frame.table("person")
    if (
        _GENERAL_OUTPUT not in person
        or _CRITERIA_OUTPUT not in person
        or "age" not in person
    ):
        missing = [
            column
            for column in ("age", _GENERAL_OUTPUT, _CRITERIA_OUTPUT)
            if column not in person
        ]
        return GateResult(
            name="child_disability_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )
    summary = us_child_disability_summary(frame)
    failures: list[str] = []
    if summary["missing_or_nonfinite_count"]:
        failures.append(
            f"{_GENERAL_OUTPUT}: {summary['missing_or_nonfinite_count']} missing or "
            "nonfinite value(s)."
        )
    if summary["invalid_boolean_count"]:
        failures.append(
            f"{_GENERAL_OUTPUT}: {summary['invalid_boolean_count']} non-boolean "
            "value(s)."
        )
    if summary["criteria_missing_or_nonfinite_count"]:
        failures.append(
            f"{_CRITERIA_OUTPUT}: "
            f"{summary['criteria_missing_or_nonfinite_count']} missing or "
            "nonfinite value(s)."
        )
    if summary["criteria_invalid_boolean_count"]:
        failures.append(
            f"{_CRITERIA_OUTPUT}: {summary['criteria_invalid_boolean_count']} "
            "non-boolean value(s)."
        )
    if summary["criteria_outside_general_count"]:
        failures.append(
            f"{_CRITERIA_OUTPUT}: {summary['criteria_outside_general_count']} "
            "under-15 positive value(s) fall outside is_disabled."
        )
    for label, share_key, weight_key, unique_key, band in (
        (
            "0",
            "age_0_disabled_share",
            "age_0_weight",
            "age_0_unique_count",
            US_CHILD_DISABILITY_AGE_0_SHARE_BAND,
        ),
        (
            "1-4",
            "age_1_4_disabled_share",
            "age_1_4_weight",
            "age_1_4_unique_count",
            US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND,
        ),
        (
            "5-14",
            "age_5_14_disabled_share",
            "age_5_14_weight",
            "age_5_14_unique_count",
            US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND,
        ),
    ):
        if float(summary[weight_key]) <= 0.0:
            failures.append(f"age {label}: weighted domain is empty.")
            continue
        share = float(summary[share_key])
        low, high = band
        if not low <= share <= high:
            failures.append(
                f"age {label}: disabled share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
        if int(summary[unique_key]) < 2:
            failures.append(f"age {label}: disability surface is constant.")

    severity_share = float(summary["child_ssi_criteria_within_general_share"])
    severity_low, severity_high = US_CHILD_SSI_SEVERITY_SHARE_BAND
    if not severity_low <= severity_share <= severity_high:
        failures.append(
            "under-15 SSI-criteria share within general disability "
            f"{severity_share:.4f} outside receipt-anchored band "
            f"[{severity_low}, {severity_high}]."
        )
    if int(summary["child_ssi_criteria_unique_count"]) < 2:
        failures.append("under-15 SSI-criteria surface is constant.")

    adult_unchanged: bool | None = None
    if input_frame is not None:
        input_person = input_frame.table("person")
        same_identity = (
            "person_id" in input_person
            and "person_id" in person
            and person["person_id"].equals(input_person["person_id"])
        )
        if same_identity and "age" in input_person:
            input_age = pd.to_numeric(input_person["age"], errors="coerce")
            adult = input_age >= 15
            adult_unchanged = set(input_person.columns) <= set(person.columns) and (
                person.loc[adult, input_person.columns].equals(input_person.loc[adult])
            )
        else:
            adult_unchanged = False
        if not adult_unchanged:
            failures.append("age 15+: output rows differ from the stage input.")
    details = dict(summary)
    if input_frame is not None:
        input_summary = us_child_disability_summary(input_frame)
        details["weighted_child_is_disabled_share_change"] = {
            "age_0": {
                "before": input_summary["age_0_disabled_share"],
                "after": summary["age_0_disabled_share"],
                "absolute_change": (
                    float(summary["age_0_disabled_share"])
                    - float(input_summary["age_0_disabled_share"])
                ),
            },
            "age_1_4": {
                "before": input_summary["age_1_4_disabled_share"],
                "after": summary["age_1_4_disabled_share"],
                "absolute_change": (
                    float(summary["age_1_4_disabled_share"])
                    - float(input_summary["age_1_4_disabled_share"])
                ),
            },
            "age_5_14": {
                "before": input_summary["age_5_14_disabled_share"],
                "after": summary["age_5_14_disabled_share"],
                "absolute_change": (
                    float(summary["age_5_14_disabled_share"])
                    - float(input_summary["age_5_14_disabled_share"])
                ),
            },
        }
    details["age_15_plus_unchanged"] = adult_unchanged
    return GateResult(
        name="child_disability_signal",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
