"""Full-source ASEC teenager O/D donors and explicitly unmeasured children.

All projections are descriptive. Consumers must retain and requalify the real
preparation around model/store I/O; no DataFrame, receipt or copy issues source
authority. This module neither fits nor draws, and writes no canonical values.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from . import current_acs_income_anchor_source as acs
from . import current_asec_dividend_source as dividend
from . import current_asec_income_routing_source as routing
from . import current_asec_interest_source as interest
from . import graph_full_puf_enrichment as physical
from . import survey_population_domains as domains
from . import survey_population_preparation as source
from .property_income_constants import PROPERTY_COMPONENTS

PROTOCOL = "microcosm.us.child-property-source.v1"
TARGETS = (PROPERTY_COMPONENTS[0], PROPERTY_COMPONENTS[2])
DONOR_AGES = (15, 16, 17)
MAX_ROWS = 600_000
MAX_PROJECTION_BYTES = 64 * 1024**2
_ORDINARY_STATUSES = ("known_receipt", "known_nonreceipt", "observed_zero_component")
_DIVIDEND_STATUSES = ("known_receipt", "known_nonreceipt")
_SOURCE_ERRORS = ("invalid_amount_literal", "unrecognized_receipt_literal")


def require(condition, code):
    if not condition:
        raise ValueError("CHILD_PROPERTY_" + code)


def _json(value):
    result = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    require(len(result) <= MAX_PROJECTION_BYTES, "PROJECTION_SIZE")
    return result


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _modules():
    return (
        sys.modules[__name__],
        interest,
        dividend,
        routing,
        acs,
        interest.physical,
        physical,
        physical.store_ops,
        physical.population_ops,
    )


def _source_bytes():
    return tuple(
        (module.__name__, _sha(Path(module.__file__).read_bytes()))
        for module in _modules()
    )


def _live():
    result = dict(source._live())
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = source._function_seal(value)
            elif callable(value) and isinstance(
                getattr(value, "__wrapped__", None), FunctionType
            ):
                result[module.__name__, name] = (
                    value,
                    source._function_seal(value.__wrapped__),
                )
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = source._function_seal(
                            function
                        )
    result["child_property_contract"] = source._runtime_marker(
        (
            PROTOCOL,
            TARGETS,
            DONOR_AGES,
            MAX_ROWS,
            MAX_PROJECTION_BYTES,
            _ORDINARY_STATUSES,
            _DIVIDEND_STATUSES,
            _SOURCE_ERRORS,
            interest.PROTOCOL,
            interest.READ_COLUMNS,
            interest.AMOUNT_FIELDS,
            interest.ADDITIONAL_AMOUNT_ENTRIES,
            interest.RECEIPT_ENTRIES,
            interest.ACCOUNT_ENTRIES,
            interest.ALLOCATION_ENTRIES,
            interest.TOPCODE_ENTRIES,
            dividend.PROTOCOL,
            dividend.READ_COLUMNS,
            dividend.AMOUNT_FIELDS,
            dividend.RECEIPT_ENTRIES,
            dividend.SURVIVOR_ENTRIES,
            dividend.SURVIVOR_CODES,
            dividend.ALLOCATION_ENTRIES,
            dividend.ALLOCATION_CONFLICT_NOTE,
            dividend.TOPCODE_ENTRIES,
            routing.COORDINATE_COLUMNS,
            routing.COORDINATE_WIDTHS,
            routing.TOKEN_MAX_CHARS,
            routing.DOMAINS_RESOURCE,
            routing.DOMAINS_SHA256,
            routing.CURRENT_INCOME_YEAR,
            routing.RECEIPT_CODE_DOMAIN,
            routing.ACCOUNT_CODES,
            routing.ALLOCATION_ANNVAL_CODES,
            routing.ALLOCATION_COMPOSITE_CODES,
            type(routing.ALLOCATION_CODE_MEANINGS_UNPUBLISHED),
            tuple(sorted(routing.ALLOCATION_CODE_MEANINGS_UNPUBLISHED)),
            routing.DOLLAR_ZERO_SEMANTICS,
            routing.KNOWN_AMOUNT_STATUSES,
            routing.DICTIONARY_URL,
            routing.DICTIONARY_SHA256,
            acs.PROTOCOL,
            acs.COLUMNS,
            acs.ANCHORS,
            acs.STRING_DTYPE,
            acs.STRING_DTYPE.storage,
            acs.STRING_DTYPE.na_value,
        )
    )
    return result


def _age(value):
    require(
        type(value) is str and re.fullmatch(r"[0-9]{1,2}", value, re.ASCII) is not None,
        "AGE_LITERAL",
    )
    age = int(value)
    require(0 <= age <= 99, "AGE_RANGE")
    return age


def _integer_literal(value, name, maximum):
    require(
        type(value) is str
        and re.fullmatch(r"[0-9]+", value, re.ASCII) is not None
        and len(value) <= 64,
        name + "_LITERAL",
    )
    result = int(value)
    require(0 <= result <= maximum, name + "_RANGE")
    return result


def _raw_axis(frame):
    require(
        type(frame) is pd.DataFrame
        and frame.columns.is_unique
        and frame.index.dtype == np.dtype("int64")
        and frame.index.is_unique
        and frame.index.name == "native_person_id"
        and 0 < len(frame) <= MAX_ROWS
        and set(routing.COORDINATE_COLUMNS) <= set(frame),
        "FULL_SOURCE_AXIS",
    )
    require(
        all(type(v) is str and 0 < len(v) <= 64 for v in frame.PERIDNUM)
        and frame.PERIDNUM.is_unique,
        "PERIDNUM",
    )
    ages = np.asarray([_age(v) for v in frame.A_AGE], dtype=np.int64)
    homes = [_integer_literal(v, "PH_SEQ", 99999) for v in frame.PH_SEQ]
    lines = [_integer_literal(v, "A_LINENO", 16) for v in frame.A_LINENO]
    require(all(v >= 1 for v in lines), "A_LINENO_RANGE")
    return ages, homes, lines


def _catalogue_members(households):
    require(type(households) is tuple and len(households) <= MAX_ROWS, "CATALOGUE_TYPE")
    people, seen_households = {}, set()
    for household in households:
        require(
            type(household) is domains.AsecHousehold
            and type(household.key) is domains.HouseholdKey
            and household.key.source is domains.Source.ASEC
            and household.key.source_year == 2024
            and household.key.survey_year == 2025
            and type(household.persons) is tuple
            and household.h_hhtype == "1",
            "CATALOGUE_PERIOD_OR_TYPE",
        )
        require(
            _integer_literal(household.h_numper, "H_NUMPER", MAX_ROWS)
            == len(household.persons),
            "CATALOGUE_MEMBERSHIP_COUNT",
        )
        key = household.key
        home = _integer_literal(key.native_id, "H_SEQ", 99999)
        require(home not in seen_households, "DUPLICATE_HOUSEHOLD")
        seen_households.add(home)
        # Match the existing native/selection conversion; never apply sample or clone multipliers.
        units = _integer_literal(household.hsup_wgt, "HSUP_WGT", 999999999)
        weight = float(Fraction(units, 100))
        for person in household.persons:
            require(
                type(person) is domains.AsecPerson and person.household_key == key,
                "CATALOGUE_PERSON",
            )
            line = _integer_literal(person.a_lineno, "A_LINENO", 16)
            require(
                line >= 1 and person.peridnum not in people,
                "CATALOGUE_DUPLICATE_PERSON",
            )
            people[person.peridnum] = (
                home,
                line,
                _age(person.age),
                weight,
                key.native_id,
            )
    return people


def _bad_status(status):
    return status in _SOURCE_ERRORS or status.startswith("contradictory_")


@dataclass(frozen=True)
class ChildPropertyDonorProjection:
    """Full original cohort diagnostics plus the qualified observed donor table."""

    donors: pd.DataFrame
    diagnostics: pd.DataFrame
    interest: pd.DataFrame
    dividend: pd.DataFrame


def project_child_property_donors(
    interest_literals: pd.DataFrame,
    dividend_literals: pd.DataFrame,
    households: tuple,
) -> ChildPropertyDonorProjection:
    """Pure complete-roster projection; arguments grant no source authority."""
    ages, homes, lines = _raw_axis(interest_literals)
    other_ages, other_homes, other_lines = _raw_axis(dividend_literals)
    require(
        interest_literals.index.identical(dividend_literals.index)
        and np.array_equal(ages, other_ages)
        and homes == other_homes
        and lines == other_lines
        and interest_literals[list(routing.COORDINATE_COLUMNS)].equals(
            dividend_literals[list(routing.COORDINATE_COLUMNS)]
        ),
        "FULL_SOURCE_ALIGNMENT",
    )
    roster = _catalogue_members(households)
    require(set(interest_literals.PERIDNUM) == set(roster), "COMPLETE_CATALOGUE_JOIN")
    weights, donor_keys, household_keys = [], [], []
    for raw, age, home, line in zip(
        interest_literals.itertuples(), ages, homes, lines, strict=True
    ):
        expected_home, expected_line, expected_age, weight, original_home = roster[
            raw.PERIDNUM
        ]
        require(
            (home, line, int(age)) == (expected_home, expected_line, expected_age),
            "CATALOGUE_COORDINATE",
        )
        weights.append(weight)
        household_keys.append(("asec", 2024, 2025, original_home))
        donor_keys.append(("asec", 2024, 2025, original_home, raw.PERIDNUM, str(line)))
    observed_interest = interest.project_interest_literals(interest_literals)
    observed_dividend = dividend.project_dividend_literals(dividend_literals)
    observed_interest.index = interest_literals.index.copy()
    observed_dividend.index = dividend_literals.index.copy()
    ordinary = observed_interest.TRDINT_VAL_amount.to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    dividends = observed_dividend.DIV_VAL_amount.to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    oi, di = (
        observed_interest.TRDINT_VAL_reporting_status,
        observed_dividend.DIV_VAL_reporting_status,
    )
    known_o = observed_interest.TRDINT_VAL_amount_known.to_numpy()
    known_d = observed_dividend.DIV_VAL_amount_known.to_numpy()
    require(
        np.array_equal(known_o, np.isfinite(ordinary))
        and np.array_equal(known_d, np.isfinite(dividends))
        and np.array_equal(known_o, oi.isin(_ORDINARY_STATUSES))
        and np.array_equal(known_d, di.isin(_DIVIDEND_STATUSES))
        and (ordinary[known_o] >= 0).all()
        and (dividends[known_d] >= 0).all(),
        "AMOUNT_KNOWNNESS",
    )
    cohort = np.isin(ages, DONOR_AGES)
    errors = np.asarray(
        [_bad_status(a) or _bad_status(b) for a, b in zip(oi, di, strict=True)]
    )
    diagnostics = pd.DataFrame(index=interest_literals.index.copy())
    diagnostics["source_age"] = ages
    diagnostics["cohort"] = cohort
    diagnostics["source_error"] = errors
    diagnostics["ordinary_known"] = known_o
    diagnostics["dividend_known"] = known_d
    diagnostics["original_household_design_weight"] = weights
    diagnostics["weight_zero"] = np.asarray(weights) == 0
    diagnostics["observed_zero_pair"] = (
        known_o & known_d & (ordinary == 0) & (dividends == 0)
    )
    diagnostics["observed_positive_pair"] = (
        known_o & known_d & ((ordinary > 0) | (dividends > 0))
    )
    eligible = cohort & known_o & known_d & ~errors
    diagnostics["observed_donor"] = eligible
    diagnostics["positive_mass_donor"] = eligible & (np.asarray(weights) > 0)
    diagnostics["reason"] = np.select(
        [~cohort, errors, ~known_o | ~known_d, np.asarray(weights) == 0],
        [
            "outside_donor_age_band",
            "source_review_required",
            "unresolved_observed_pair",
            "zero_design_mass",
        ],
        default="observed_pair",
    )
    donors = (
        pd.DataFrame(
            {
                TARGETS[0]: ordinary,
                TARGETS[1]: dividends,
                "original_household_design_weight": weights,
                "donor_key": donor_keys,
                "household_key": household_keys,
            },
            index=interest_literals.index.copy(),
        )
        .loc[eligible]
        .copy()
    )
    # Source allocation/edit/topcode evidence remains independently available, with original rows.
    return ChildPropertyDonorProjection(
        donors, diagnostics, observed_interest, observed_dividend
    )


def project_child_property_recipients(
    origins, asec_interest, asec_dividend, acs_anchors
):
    """Classify all selected originals; under15 source errors precede eligibility."""
    require(type(origins) is dict and set(origins) >= {"persons"}, "ORIGINS")
    payload = origins["persons"]
    table = pd.DataFrame(payload["rows"], columns=payload["columns"])
    required = {
        "person_id",
        "source",
        "source_year",
        "survey_year",
        "raw_native_household_id",
        "raw_native_person_id",
        "native_line_numeric_original",
        "selected_receiving_person_id",
    }
    require(
        required <= set(table)
        and table.person_id.dtype == np.dtype("int64")
        and table.person_id.is_unique
        and len(table) <= MAX_ROWS,
        "RECIPIENT_AXIS",
    )
    expected = pd.Index(table.person_id, name="person_id")
    for frame, channel in (
        (asec_interest, "asec"),
        (asec_dividend, "asec"),
        (acs_anchors, "acs"),
    ):
        require(
            type(frame) is pd.DataFrame
            and frame.index.dtype == np.dtype("int64")
            and frame.index.is_unique
            and set(frame.index)
            == set(table.loc[table.source.eq(channel), "person_id"]),
            "RECIPIENT_SOURCE_JOIN",
        )
    rows = []
    for row in table.itertuples(index=False):
        require(
            row.source in ("asec", "acs")
            and row.source_year == 2024
            and row.survey_year == (2025 if row.source == "asec" else 2024),
            "RECIPIENT_PERIOD",
        )
        require(
            all(
                type(v) is str and bool(v)
                for v in (
                    row.raw_native_household_id,
                    row.raw_native_person_id,
                    row.native_line_numeric_original,
                )
            ),
            "RECIPIENT_COORDINATE",
        )
        key = (
            row.source,
            int(row.source_year),
            int(row.survey_year),
            row.raw_native_household_id,
            row.raw_native_person_id,
            row.native_line_numeric_original,
        )
        if row.source == "asec":
            a, b = asec_interest.loc[row.person_id], asec_dividend.loc[row.person_id]
            require(
                a.native_person_id
                == b.native_person_id
                == row.selected_receiving_person_id
                and a.source_age == b.source_age,
                "RECIPIENT_SOURCE_IDENTITY",
            )
            age = int(a.source_age)
            require(
                a.source_age == age
                and not isinstance(a.source_age, (bool, np.bool_))
                and 0 <= age <= 99,
                "RECIPIENT_AGE",
            )
            o_status, d_status = (
                a.TRDINT_VAL_reporting_status,
                b.DIV_VAL_reporting_status,
            )
            known_o, known_d = (
                bool(a.TRDINT_VAL_amount_known),
                bool(b.DIV_VAL_amount_known),
            )
            regular = all(
                status == "outside_reporting_universe"
                for status in (o_status, d_status)
            )
            regular &= all(
                status in ("in_printed_range", "missing")
                for status in (a.TRDINT_VAL_literal_status, b.DIV_VAL_literal_status)
            )
            regular &= all(
                pd.isna(v) or v == 0
                for v in (a.TRDINT_VAL_published_amount, b.DIV_VAL_published_amount)
            )
            invalid = (
                _bad_status(o_status)
                or _bad_status(d_status)
                or (age < 15 and not regular)
            )
        else:
            a = acs_anchors.loc[row.person_id]
            require(
                a.native_person_id == row.selected_receiving_person_id,
                "RECIPIENT_SOURCE_IDENTITY",
            )
            age = _age(a.AGEP)
            known_o = known_d = False
            regular = (
                a.property_income_status == "outside_universe_blank" and a.INTP == ""
            )
            regular &= bool(a.property_income_adjustment_known)
            invalid = age < 15 and not regular
        eligible = age < 15 and regular and not known_o and not known_d and not invalid
        reason = (
            "outside_recipient_age_band"
            if age >= 15
            else "source_review_required"
            if invalid
            else "unsupported_known_incumbent"
            if known_o or known_d
            else "explicit_unmeasured_child"
            if eligible
            else "unsupported_child_evidence"
        )
        rows.append(
            (row.person_id, key, age, row.source, known_o, known_d, eligible, reason)
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "person_id",
            "recipient_key",
            "source_age",
            "source",
            "ordinary_source_known",
            "dividend_source_known",
            "eligible",
            "reason",
        ],
    ).set_index("person_id")
    require(
        result.index.equals(expected) and result.recipient_key.is_unique,
        "RECIPIENT_IDENTITY",
    )
    return result


@dataclass(frozen=True)
class QualifiedChildPropertySources:
    """Descriptive full donor/selected-recipient data; retain the real owners."""

    donor_projection: ChildPropertyDonorProjection
    recipients: pd.DataFrame
    evidence: dict


def _table_stamp(table):
    # Graph's maintained physical scalar seal does not admit tuple-valued cells.
    # Encode only the explicitly declared coordinate tuples, retaining scalar tags.
    projected = table.copy(deep=True)
    for name in ("donor_key", "household_key", "recipient_key"):
        if name in projected:
            encoded = []
            for key in projected[name]:
                require(
                    type(key) is tuple
                    and key
                    and all(type(v) in (str, int) for v in key),
                    "KEY_STORAGE",
                )
                encoded.append(
                    _json([["int" if type(v) is int else "str", str(v)] for v in key])
                )
            projected[name] = encoded
    return physical._table_stamp(projected)


def child_property_sources_seal(value):
    """Complete private value seal; not an admission or a safe public summary."""
    require(
        type(value) is QualifiedChildPropertySources
        and type(value.donor_projection) is ChildPropertyDonorProjection,
        "QUALIFIED_TYPE",
    )
    projected = value.donor_projection
    return tuple(
        _table_stamp(frame)
        for frame in (
            projected.donors,
            projected.diagnostics,
            projected.interest,
            projected.dividend,
            value.recipients,
        )
    ) + (_json(value.evidence),)


def qualify_child_property_sources(preparation):
    """Borrow real complete-source owners, project, then requalify before return."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    live = _live()
    require(
        live == _LIVE and _source_bytes() == _SOURCE_BYTES and _live() == live,
        "IMPLEMENTATION_CHANGED",
    )
    entry = preparation._checked()
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    state = entry[2]
    catalogue = state.catalogues[1]
    catalogue_entry = catalogue._checked()
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    households = catalogue_entry[2].households
    observed_interest = interest.qualify_current_asec_interest(preparation)
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    observed_dividend = dividend.qualify_current_asec_dividend(preparation)
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    observed_acs = acs.qualify_current_acs_income_anchors(preparation)
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    observations = (
        interest.interest_values_seal(observed_interest),
        dividend.dividend_values_seal(observed_dividend),
        acs.income_anchor_seal(observed_acs),
    )
    donor = project_child_property_donors(
        observed_interest.asec_literals, observed_dividend.asec_literals, households
    )
    require(
        not (donor.diagnostics.cohort & donor.diagnostics.source_error).any(),
        "DONOR_SOURCE_REVIEW_REQUIRED",
    )
    origins = json.loads(entry[1])["origins"]
    recipients = project_child_property_recipients(
        origins,
        observed_interest.person,
        observed_dividend.person,
        observed_acs.anchors,
    )
    require(
        not recipients.loc[recipients.source_age < 15, "reason"]
        .ne("explicit_unmeasured_child")
        .any(),
        "RECIPIENT_SOURCE_REVIEW_REQUIRED",
    )
    evidence = dict(
        protocol=PROTOCOL,
        donor_source="asec",
        donor_source_year=2024,
        donor_survey_year=2025,
        donor_age_band=list(DONOR_AGES),
        ordinary_field="TRDINT_VAL",
        dividend_field="DIV_VAL",
        donor_scope="complete_original_current_year_source",
        donor_sampling_fraction_applied=False,
        weight_source="original_household_design",
        weight_conversion="float64(Fraction(HSUP_WGT,100))",
        preparation_sha256=_sha(entry[1]),
        catalogue_sha256=_sha(catalogue_entry[1]),
        interest_evidence=observed_interest.evidence,
        dividend_evidence=observed_dividend.evidence,
        acs_evidence=observed_acs.evidence,
        full_source_rows=len(donor.diagnostics),
        observed_donor_rows=len(donor.donors),
        eligible_child_rows=int(recipients.eligible.sum()),
        source_admission_issued=False,
        model_fitted=False,
        release_eligible=False,
    )
    result = QualifiedChildPropertySources(
        donor, recipients, json.loads(_json(evidence))
    )
    seal = child_property_sources_seal(result)
    # Every external/source callback is followed by live identity checks.
    final_catalogue = catalogue._checked()
    require(_live() == live, "IMPLEMENTATION_CHANGED")
    final_entry = preparation._checked()
    require(
        _live() == live and _source_bytes() == _SOURCE_BYTES and _live() == live,
        "IMPLEMENTATION_CHANGED",
    )
    require(
        final_entry is entry
        and final_catalogue is catalogue_entry
        and source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and state.catalogues[1] is catalogue
        and source.asec_catalogue._ISSUED.get(id(catalogue)) is catalogue_entry
        and catalogue.payload == catalogue_entry[1]
        and catalogue_entry[2].households is households,
        "FINAL_OWNERS",
    )
    source._pure_final(state)
    require(
        observations
        == (
            interest.interest_values_seal(observed_interest),
            dividend.dividend_values_seal(observed_dividend),
            acs.income_anchor_seal(observed_acs),
        ),
        "FINAL_OBSERVATIONS_CHANGED",
    )
    require(
        child_property_sources_seal(result) == seal and _live() == live,
        "FINAL_VALUES_CHANGED",
    )
    return result


_SOURCE_BYTES = _source_bytes()
_LIVE = _live()
