"""Interview-date coverage recodes; detached values grant no source authority.

Only literal source yes/no codes determine coverage. Allocation is separate
provenance, and a broader ACS item never becomes a narrower canonical flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import support_provenance as provenance
from .cps_carried import US_REPORTED_COVERAGE_PERSON_INPUTS

PROTOCOL = "microcosm.us.current-survey-health-coverage.v1"
STRING_DTYPE = pd.StringDtype(storage="python", na_value=pd.NA)


@dataclass(frozen=True)
class CoverageField:
    output: str
    asec: str
    acs: str | None
    acs_gap: str | None = None

    @property
    def columns(self):
        return (self.output, self.output + "__known", self.output + "__source_status")


# Separate NOW_CAID from the broader NOW_MCAID legacy mapping. These are
# source recodes, not benefit eligibility or prior-calendar-year coverage.
FIELDS = (
    CoverageField("has_esi", "NOW_GRP", "HINS1"),
    CoverageField(
        "has_marketplace_health_coverage_at_interview",
        "NOW_MRK",
        None,
        "direct_purchase_marketplace_split_unobserved",
    ),
    CoverageField(
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "NOW_NONM",
        None,
        "direct_purchase_marketplace_split_unobserved",
    ),
    CoverageField(
        "has_medicaid_health_coverage_at_interview",
        "NOW_CAID",
        None,
        "means_tested_program_split_unobserved",
    ),
    CoverageField(
        "has_other_means_tested_health_coverage_at_interview",
        "NOW_OTHMT",
        None,
        "means_tested_program_split_unobserved",
    ),
    CoverageField(
        "has_tricare_health_coverage_at_interview",
        "NOW_MIL",
        None,
        "military_coverage_type_unresolved",
    ),
    CoverageField(
        "has_champva_health_coverage_at_interview",
        "NOW_CHAMPVA",
        None,
        "champva_not_separately_observed",
    ),
    CoverageField(
        "has_va_health_coverage_at_interview",
        "NOW_VACARE",
        None,
        "va_champva_recode_scope_unresolved",
    ),
    CoverageField(
        "has_indian_health_service_coverage_at_interview", "NOW_IHSFLG", "HINS7"
    ),
)
ASEC_VALUE_COLUMNS = tuple(f.asec for f in FIELDS) + ("NOW_MCAID",)
ASEC_ALLOCATION_COLUMNS = tuple("I_" + c for c in ASEC_VALUE_COLUMNS)
ACS_VALUE_COLUMNS = tuple("HINS" + str(i) for i in range(1, 8))
ACS_ALLOCATION_COLUMNS = tuple("FHINS" + str(i) + "P" for i in range(1, 8))
ACS_EDIT_COLUMNS = ("FHINS3C", "FHINS4C", "FHINS5C")
RAW_COLUMNS = (
    *ASEC_VALUE_COLUMNS,
    *ASEC_ALLOCATION_COLUMNS,
    *ACS_VALUE_COLUMNS,
    *ACS_ALLOCATION_COLUMNS,
    *ACS_EDIT_COLUMNS,
)
SOURCE_PREFIX = "health_source_"


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_HEALTH_" + reason)


def _field(name):
    require(
        set(f.output for f in FIELDS) == set(US_REPORTED_COVERAGE_PERSON_INPUTS),
        "PROFILE_ROSTER",
    )
    matches = [f for f in FIELDS if f.output == name]
    require(len(matches) == 1, "FIELD")
    return matches[0]


def recode(tokens, allocations, *, survey):
    """Keep missing, unrecognized, allocated and reported source states apart.

    This classifier accepts literal strings only. A valid source yes/no value
    survives unknown allocation provenance, without claiming a raw response.
    Neither blank nor an out-of-dictionary code means no coverage.
    """
    require(survey in ("asec", "acs"), "SURVEY")
    values, flags = tuple(tokens), tuple(allocations)
    require(
        len(values) == len(flags)
        and all(type(x) is str and len(x) <= 64 for x in (*values, *flags)),
        "LITERAL_CONTRACT",
    )
    allocation_labels = (
        {"0": "reported", "1": "hotdeck", "2": "logical", "3": "whole_unit"}
        if survey == "asec"
        else {"0": "not_allocated", "1": "allocated"}
    )
    canonical, known, status = [], [], []
    for token, allocation in zip(values, flags, strict=True):
        valid = token in ("1", "2")
        canonical.append(token == "1" if valid else pd.NA)
        known.append(valid)
        if not valid:
            status.append(
                "missing_source_value" if token == "" else "unrecognized_source_value"
            )
        else:
            method = allocation_labels.get(allocation, "allocation_unknown")
            status.append(("source_yes_" if token == "1" else "source_no_") + method)
    return pd.DataFrame(
        {
            "value": pd.array(canonical, dtype="boolean"),
            "known": np.asarray(known, dtype=bool),
            "status": pd.array(status, dtype=STRING_DTYPE),
        }
    )


def recode_field(raw, name):
    """Transform one field on the full original-person axis, retaining gaps."""
    spec = _field(name)
    require(
        type(raw) is pd.DataFrame
        and raw.index.is_unique
        and raw.index.name == "person_id"
        and {"source", *RAW_COLUMNS} <= set(raw)
        and raw.source.isin(("acs", "asec")).all(),
        "RAW_AXIS",
    )
    result = pd.DataFrame(index=raw.index.copy())
    result[spec.output] = pd.array([pd.NA] * len(raw), dtype="boolean")
    result[spec.output + "__known"] = False
    result[spec.output + "__source_status"] = pd.array(
        [pd.NA] * len(raw), dtype=STRING_DTYPE
    )
    for survey, raw_name in (("asec", spec.asec), ("acs", spec.acs)):
        mask = raw.source.eq(survey)
        if raw_name is None:
            result.loc[mask, spec.output + "__source_status"] = (
                "semantic_gap:" + spec.acs_gap
            )
            continue
        flag_name = "I_" + raw_name if survey == "asec" else "F" + raw_name + "P"
        part = recode(raw.loc[mask, raw_name], raw.loc[mask, flag_name], survey=survey)
        for output, value in zip(spec.columns, part, strict=True):
            result.loc[mask, output] = part[value].array
    return result


def source_columns(raw):
    """Namespace literal observations so they cannot impersonate output leaves."""
    require(raw.index.name == "person_id" and raw.index.is_unique, "RAW_AXIS")
    result = pd.DataFrame(index=raw.index.copy())
    for name in RAW_COLUMNS:
        require(all(x is None or type(x) is str for x in raw[name]), "RAW_LITERAL_TYPE")
        result[SOURCE_PREFIX + name] = pd.array(raw[name], dtype=STRING_DTYPE)
    for name in ("source", "source_year", "survey_year"):
        result[SOURCE_PREFIX + name] = (
            pd.array(raw[name], dtype=STRING_DTYPE)
            if name == "source"
            else raw[name].array.copy()
        )
    return result


def attach_columns(origins, receiving, columns):
    """Fan original observations out to exactly their unchanged two clones.

    The owning host authenticates origins and source columns before calling.
    Existing fields are never overwritten, including another family's leaves.
    """
    require(
        type(origins) is pd.DataFrame
        and type(columns) is pd.DataFrame
        and origins.index.is_unique
        and origins.index.name == "person_id"
        and columns.index.equals(origins.index)
        and columns.columns.is_unique,
        "ATTACH_SOURCE_AXIS",
    )
    people = receiving.person
    ids = people.person_id
    original = people[provenance.support_source_id_column("person")].to_numpy()
    clone = people[provenance.support_clone_index_column("person")].to_numpy()
    native = people[provenance.spine_source_id_column("person")].to_numpy()
    channel = people[provenance.support_channel_column("person")].to_numpy()
    require(
        ids.dtype == np.dtype("int64")
        and ids.is_unique
        and original.dtype == clone.dtype == native.dtype == np.dtype("int64")
        and len(original) == 2 * len(origins)
        and set(original) == set(origins.index),
        "CLONE_SOURCE_ROSTER",
    )
    expected = origins.reindex(original)
    require(
        np.array_equal(native, expected.native_person_id.to_numpy())
        and np.array_equal(channel, expected.source.to_numpy()),
        "CLONE_SOURCE_IDENTITY",
    )
    pairs = pd.MultiIndex.from_arrays((original, clone))
    require(pairs.is_unique and np.isin(clone, (0, 1)).all(), "CLONE_PAIR")
    require(not set(columns) & set(people), "ATTACH_OWNERSHIP_COLLISION")
    return {
        ("person", name): pd.Series(
            columns[name].reindex(original).array.copy(),
            index=pd.Index(ids.to_numpy(), name="person_id"),
            name=name,
        )
        for name in columns
    }
