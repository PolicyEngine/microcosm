"""Pure, unbound ACS housing-unit semantics; no source authentication.

ACS TYPEHUGQ/NP observations do not supply the CPS interview-scope or
household-kind observations. Those output axes therefore remain source-specific
not-observed codes, even for a known occupied ACS housing unit. Invalid TEN
payloads are uninterpreted placeholders, never observed zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ACSHousingUniverseRefusalError", "classify_acs_housing_universe"]


class ACSHousingUniverseRefusalError(ValueError):
    """Refuse an invalid input contract without reporting source values or keys."""


def _owned_index(index: pd.Index) -> pd.Index:
    """Copy axis storage, reconstructing indexes with shared deep-copy caches."""
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(index.start, index.stop, index.step, name=index.name)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex(
            levels=[_owned_index(level) for level in index.levels],
            codes=[code.copy() for code in index.codes],
            names=list(index.names),
            sortorder=index.sortorder,
            verify_integrity=False,
        )
    if isinstance(index, pd.CategoricalIndex):
        return pd.CategoricalIndex(
            pd.Categorical.from_codes(
                index.codes.copy(),
                categories=_owned_index(index.categories),
                ordered=index.ordered,
            ),
            name=index.name,
        )
    return index.copy(deep=True)


def _header() -> dict:
    """Fresh semantic metadata only: no source rows, keys, counts, or pins."""
    return {
        "schema_version": 1,
        "artifact_kind": "microcosm.acs_housing_universe_unbound.v1",
        "source_authentication": "unbound_semantic_classifier",
        "release_eligible": False,
        "source_mapping": {
            "TYPEHUGQ": {
                "1": "housing_unit",
                "2": "institutional_group_quarters",
                "3": "noninstitutional_group_quarters",
            },
            "NP": "nonnegative int64; no additional upper bound",
            "TEN": "native ACS codes; see tenure_subtype codebook",
            "occupied_hu": (
                "TYPEHUGQ=1 and NP>0 establishes occupancy; TYPEHUGQ=2/3 "
                "is outside the HU universe; TYPEHUGQ=1 and NP=0 leaves "
                "occupancy unresolved and does not assert vacancy"
            ),
            "hu_tenure_class": (
                "only known occupied HUs: valid TEN=1/2 owner, valid TEN=3/4 "
                "renter, invalid TEN unclassified but retained in the HU total"
            ),
            "interview_scope": "CPS interview observation not supplied by ACS inputs",
            "household_kind": "CPS household-kind observation not supplied by ACS inputs",
            "ignored_columns": "all other columns, including model tenure_type",
            "row_policy": "retain every row in original order, including all GQ rows",
        },
        "codebooks": {
            "interview_scope": {"0": "not_observed_in_these_ACS_inputs"},
            "physical_unit": {"1": "housing_unit", "2": "group_quarters"},
            "household_kind": {"0": "not_observed_in_these_ACS_inputs"},
            "tenure_subtype": {
                "0": "unavailable",
                "1": "owned_with_mortgage_or_loan",
                "2": "owned_outright",
                "3": "paying_rent",
                "4": "no_rent_paid",
            },
            "occupied_hu": {
                "0": "occupancy_not_established",
                "1": "occupied_housing_unit",
                "2": "outside_housing_unit_universe",
            },
            "hu_tenure_class": {
                "0": "outside_or_unresolved_occupied_HU_universe",
                "1": "owner",
                "2": "renter_including_no_rent",
                "3": "occupied_HU_tenure_unclassified",
            },
            "unresolved_reasons": {
                "0": "no_unresolved_reason",
                "1": "housing_unit_occupancy_not_established",
                "2": "occupied_housing_unit_tenure_unavailable",
            },
            "TEN_valid": {"0": "unavailable", "1": "observed_native_TEN"},
        },
        "validity_placeholder_semantics": {
            "input": "exact numpy int64 TYPEHUGQ/NP/TEN and positional bool ndarray mask",
            "invalid_TEN": (
                "any int64 payload is an uninterpreted placeholder; never an "
                "observed zero and never checked against the observed TEN domain"
            ),
            "output": "uint8 codes; tenure_subtype=0 iff TEN_valid=0 on every row",
            "GQ": "unavailable TEN, including a blank, adds no conflict or reason",
            "unresolved_reasons": "bit mask: 1 occupancy not established; 2 occupied HU TEN unavailable",
        },
        "limits": [
            "no_authenticated_source_or_source_pins",
            "no_CPS_interview_or_household_kind_observations",
            "no_B19001_income_concept_bridge",
            "no_scientific_certification",
            "no_stacked_population_or_denominator_certification",
        ],
    }


def classify_acs_housing_universe(
    households: pd.DataFrame, *, tenure_valid: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Classify supplied ACS observations without authenticating their source.

    Require an exact DataFrame with unique columns and native numpy int64
    TYPEHUGQ, NP, and TEN. TYPEHUGQ must be 1/2/3 and NP nonnegative. The
    positional validity argument must be an exact one-dimensional bool ndarray
    of matching length. Only valid TEN cells are observed and require codes
    1/2/3/4; invalid payloads can be any int64. No coercion or nullable path is
    provided. Other columns, including tenure_type, are ignored.

    Return an owned uint8 DataFrame preserving every row and its index, plus a
    fresh, data-independent semantic header. Both CPS-specific axes stay zero
    (not observed). NP=0 does not establish vacancy. Unknown occupied-HU tenure
    stays explicitly unclassified within the total; GQ tenure never changes
    the HU universe. The classifier performs no I/O or source authentication.
    """
    if type(households) is not pd.DataFrame:
        raise ACSHousingUniverseRefusalError("households_must_be_exact_dataframe")
    if not households.columns.is_unique:
        raise ACSHousingUniverseRefusalError("household_columns_must_be_unique")
    required = ("TYPEHUGQ", "NP", "TEN")
    if not set(required).issubset(set(households.columns)):
        raise ACSHousingUniverseRefusalError("required_ACS_columns_missing")
    for column in required:
        dtype = households[column].dtype
        if not isinstance(dtype, np.dtype) or dtype != np.dtype(np.int64):
            raise ACSHousingUniverseRefusalError("ACS_columns_must_be_numpy_int64")
    if (
        type(tenure_valid) is not np.ndarray
        or tenure_valid.dtype != np.dtype(np.bool_)
        or tenure_valid.ndim != 1
        or len(tenure_valid) != len(households)
    ):
        raise ACSHousingUniverseRefusalError(
            "TEN_validity_must_be_matching_bool_vector"
        )

    unit_type = households["TYPEHUGQ"].to_numpy(copy=False)
    persons = households["NP"].to_numpy(copy=False)
    tenure = households["TEN"].to_numpy(copy=False)
    if np.any((unit_type < 1) | (unit_type > 3)):
        raise ACSHousingUniverseRefusalError("TYPEHUGQ_outside_domain")
    if np.any(persons < 0):
        raise ACSHousingUniverseRefusalError("NP_must_be_nonnegative")
    observed_tenure = tenure[tenure_valid]
    if np.any((observed_tenure < 1) | (observed_tenure > 4)):
        raise ACSHousingUniverseRefusalError("valid_TEN_outside_domain")

    housing_unit = unit_type == 1
    occupied = housing_unit & (persons > 0)
    occupancy_unknown = housing_unit & (persons == 0)
    tenure_subtype = np.zeros(len(households), dtype=np.uint8)
    tenure_subtype[tenure_valid] = observed_tenure
    occupied_hu = np.zeros(len(households), dtype=np.uint8)
    occupied_hu[occupied] = 1
    occupied_hu[~housing_unit] = 2
    hu_tenure_class = np.zeros(len(households), dtype=np.uint8)
    hu_tenure_class[occupied & tenure_valid & (tenure_subtype <= 2)] = 1
    hu_tenure_class[occupied & tenure_valid & (tenure_subtype >= 3)] = 2
    hu_tenure_class[occupied & ~tenure_valid] = 3
    reasons = np.zeros(len(households), dtype=np.uint8)
    reasons[occupancy_unknown] |= 1
    reasons[occupied & ~tenure_valid] |= 2

    output = pd.DataFrame(
        {
            "interview_scope": np.zeros(len(households), dtype=np.uint8),
            "physical_unit": np.where(housing_unit, 1, 2).astype(np.uint8),
            "household_kind": np.zeros(len(households), dtype=np.uint8),
            "tenure_subtype": tenure_subtype,
            "occupied_hu": occupied_hu,
            "hu_tenure_class": hu_tenure_class,
            "unresolved_reasons": reasons,
            "TEN_valid": tenure_valid.astype(np.uint8),
        },
        index=_owned_index(households.index),
        copy=True,
    )
    return output, _header()
