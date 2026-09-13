"""Filer/joint-spouse report sums for conditioning, never beneficiary amounts.

The public entry point requires a live authenticated survey preparation. Its
retained modeled return roles are inputs, not roles inferred from the coarsened
PUF filing-class predictor. The private arithmetic is not a source issuer.
Graph hosts must retain/requalify the preparation and authenticate clone
inheritance, routing and typed model edges independently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import current_social_security_source as reports
from . import full_puf_enrichment as full
from . import survey_population_preparation as source
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.puf55-survey-ss-measurement.v1"
MEASUREMENT = "social_security_filer_joint_spouse_report_sum_proxy"
TOTAL = full.SURVEY_SS_TOTAL_PREDICTOR
KNOWN = "puf_conditioning_social_security_total_known"
ROUTE = "puf55_conditioning_profile"
COLUMNS = (TOTAL, KNOWN, ROUTE)
_PERSON_COLUMNS = ("person_id", "person_tax_unit_id", "tax_unit_role_input")
_UNIT_COLUMNS = ("tax_unit_id", "filing_status_input")
_REPORT_COLUMNS = (
    "social_security_source_total",
    "source_reporting_universe",
)
_STATUSES = {"SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"}


def _require(condition, reason):
    if not condition:
        raise ValueError("PUF55_SURVEY_SS_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _axis(values):
    _require(
        values.dtype == np.dtype("int64")
        and not values.isna().any()
        and values.ge(0).all(),
        "AXIS",
    )


def _measure(person, tax_unit, report):
    """Pure numerical checks on already-qualified values; no admission issued."""
    for table, columns in (
        (person, _PERSON_COLUMNS),
        (tax_unit, _UNIT_COLUMNS),
        (report, _REPORT_COLUMNS),
    ):
        _require(
            type(table) is pd.DataFrame
            and table.columns.is_unique
            and set(columns) <= set(table),
            "COLUMNS",
        )
    for column in (person.person_id, person.person_tax_unit_id, tax_unit.tax_unit_id):
        _axis(column)
    _require(
        len(tax_unit) > 0
        and person.person_id.is_unique
        and tax_unit.tax_unit_id.is_unique
        and set(person.person_tax_unit_id) == set(tax_unit.tax_unit_id)
        and report.index.name == "person_id"
        and report.index.dtype == np.dtype("int64")
        and report.index.is_unique
        and set(report.index) == set(person.person_id),
        "MEMBERSHIP",
    )
    roles, statuses = person.tax_unit_role_input, tax_unit.filing_status_input
    _require(
        not roles.isna().any()
        and roles.map(lambda v: type(v) is str).all()
        and roles.isin(("HEAD", "SPOUSE", "DEPENDENT")).all()
        and not statuses.isna().any()
        and statuses.map(lambda v: type(v) is str).all()
        and statuses.isin(_STATUSES).all(),
        "ROLE_VALUES",
    )
    # Validate every return's roles before deciding any missing-report route.
    # SURVIVING_SPOUSE maps to PUF class 2 elsewhere but has no actual spouse.
    groups = person.groupby("person_tax_unit_id", sort=False)
    included = []
    for tid, status in tax_unit[list(_UNIT_COLUMNS)].itertuples(index=False, name=None):
        members = groups.get_group(tid)
        head = members.tax_unit_role_input.eq("HEAD")
        spouse = members.tax_unit_role_input.eq("SPOUSE")
        _require(
            head.sum() == 1 and spouse.sum() == int(status == "JOINT"),
            "ROLE_STRUCTURE",
        )
        included.append(members.loc[head | spouse, "person_id"].to_numpy(copy=True))
    amount, universe = report[_REPORT_COLUMNS[0]], report[_REPORT_COLUMNS[1]]
    _require(
        pd.api.types.is_numeric_dtype(amount.dtype)
        and not pd.api.types.is_bool_dtype(amount.dtype)
        and not pd.api.types.is_complex_dtype(amount.dtype),
        "REPORT_PHYSICAL_TYPE",
    )
    _require(universe.dtype == np.dtype("bool"), "UNIVERSE_PHYSICAL_TYPE")
    if pd.api.types.is_integer_dtype(amount.dtype):
        _require(amount.dropna().between(-(2**53), 2**53).all(), "REPORT_PRECISION")
    numeric = amount.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    available = np.isfinite(numeric)
    _require(
        (np.isnan(numeric) | available).all() and (numeric[available] >= 0).all(),
        "REPORT_DOMAIN",
    )
    _require(
        np.isnan(numeric[~universe.to_numpy()]).all(), "OUTSIDE_UNIVERSE_OBSERVATION"
    )
    values, known, routes = [], [], []
    missing_people, included_people = 0, 0
    for ids in included:
        selected = report.loc[ids]
        observed = selected[_REPORT_COLUMNS[0]].to_numpy(
            dtype=np.float64, na_value=np.nan, copy=True
        )
        present = selected[_REPORT_COLUMNS[1]].to_numpy() & np.isfinite(observed)
        complete = bool(present.all())
        included_people += len(ids)
        missing_people += int((~present).sum())
        # No pandas skipna, empty sum, dependent zero, component sum or partial
        # report sum can manufacture a known measurement.
        total = float(np.sum(observed)) if complete else np.nan
        _require(not complete or np.isfinite(total), "SUM_OVERFLOW")
        values.append(total)
        known.append(complete)
        routes.append(
            (full.PUF55_SURVEY_SS if complete else full.PUF55_SURVEY_SS_NO_TOTAL).value
        )
    result = pd.DataFrame(
        {
            TOTAL: np.asarray(values, dtype=np.float64),
            KNOWN: np.asarray(known, dtype=bool),
            ROUTE: pd.array(routes, dtype="string"),
        },
        index=pd.Index(tax_unit.tax_unit_id.to_numpy(copy=True), name="tax_unit_id"),
    )
    counts = {
        "returns": len(result),
        "included_reporters": included_people,
        "excluded_dependents": len(person) - included_people,
        "unavailable_included_reports": missing_people,
        "nine_predictor_returns": int(result[KNOWN].sum()),
        "eight_predictor_returns": int((~result[KNOWN]).sum()),
    }
    return result, counts


def _projection_digest(table):
    _require(
        type(table) is pd.DataFrame
        and tuple(table) == COLUMNS
        and table.index.name == "tax_unit_id"
        and table.index.dtype == np.dtype("int64")
        and table.index.is_unique
        and table[TOTAL].dtype == np.dtype("float64")
        and table[KNOWN].dtype == np.dtype("bool"),
        "PROJECTION_STORAGE",
    )
    digest = hashlib.sha256()
    for tid, total, known, route in table.itertuples(index=True, name=None):
        payload = source._encode(
            [
                int(tid),
                None if np.isnan(total) else float(total).hex(),
                bool(known),
                route,
            ]
        )
        digest.update(len(payload).to_bytes(4, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_identity(people, projection):
    """Check detached source output against the exact retained source cells.

    SS qualification already authenticates original current ASEC literals and
    ACS SSP adjustment. This additional comparison prevents a return callback
    from changing its detached amount/universe without changing the live owner.
    """
    _require(type(projection) is reports.CurrentSocialSecurityProjection, "REPORT_TYPE")
    report = projection.person
    _require(
        report.index.is_unique and set(report.index) == set(people.person_id),
        "SOURCE_AXIS",
    )
    report = report.reindex(people.person_id)
    channels = people[support_channel_column("person")]
    _require(
        np.array_equal(report.source.to_numpy(), channels.to_numpy())
        and np.array_equal(
            report.native_person_id.to_numpy(),
            people[spine_source_id_column("person")].to_numpy(),
        ),
        "SOURCE_ORIGIN",
    )
    for channel, age_column, amount_column in (
        ("asec", "A_AGE", "SS_VAL"),
        ("acs", "AGEP", "acs_social_security_income"),
    ):
        mask = channels.eq(channel).to_numpy()
        selected = people.loc[mask]
        age = reports._numeric(selected[age_column])
        expected = reports._numeric(selected[amount_column])
        eligible = age >= 15
        expected[~eligible] = np.nan
        observed = report.iloc[np.flatnonzero(mask)]
        _require(
            np.array_equal(observed.source_reporting_universe.to_numpy(), eligible)
            and np.array_equal(
                reports._numeric(observed.social_security_source_total),
                expected,
                equal_nan=True,
            ),
            "SOURCE_REPORT_IDENTITY",
        )


@dataclass(frozen=True)
class Puf55SurveySSMeasurement:
    """Detached values and live upstream reference, not a reusable certificate."""

    preparation: source.AuthenticatedSurveyPopulationPreparation
    tax_unit: pd.DataFrame
    evidence: bytes


def qualify_puf55_survey_ss_measurement(preparation):
    """Project native stacked returns; clone inheritance remains the host's duty.

    Roles are the modeled roles retained by the exact preparation, not a claim
    that this helper reran current-money tax-unit construction. Receipt bytes,
    caller-made tables, and this result cannot replace the live upstream owner.
    """
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    projection = reports.qualify_current_social_security(preparation)
    _source_identity(state.frame.person, projection)
    report_evidence = source._encode(projection.evidence)
    _require(
        projection.evidence["preparation_sha256"] == _sha(entry[1]),
        "REPORT_PREPARATION",
    )
    values, counts = _measure(
        state.frame.person, state.frame.table("tax_unit"), projection.person
    )
    stamp = _projection_digest(values)
    evidence = source._encode(
        {
            "protocol": PROTOCOL,
            "measurement": MEASUREMENT,
            "preparation_sha256": _sha(entry[1]),
            "source_report_evidence_sha256": _sha(report_evidence),
            "projection_sha256": stamp,
            "counts": counts,
            "columns": list(COLUMNS),
            "membership": "HEAD plus SPOUSE only for literal JOINT; no dependents",
            "role_boundary": "exact modeled return roles retained by live preparation",
            "current_money_tax_units_reconstructed_here": False,
            "knownness": "all included source reports available; not benefit ownership",
            "fallback": full.PUF55_SURVEY_SS_NO_TOTAL.value,
            "family_and_child_reports": "retained without allocation or deduplication",
            "source_periods": {
                "asec": "2024 calendar income; 2025 interview",
                "acs": "rolling prior 12 months at 2024 interview; ADJINC price basis",
            },
            "repayment_and_railroad_alignment": "unresolved; no adjustment applied",
            "donor_E02400_editing": "unresolved; separate donor transport convention",
            "person_social_security_changed": False,
            "individual_beneficiary_assignment_claim": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        },
        maximum=64 * 1024,
    )
    result = Puf55SurveySSMeasurement(preparation, values, evidence)
    # Finish live owner I/O, then only compare retained owners and computed
    # values. The returned projection is never itself treated as authority.
    final_entry = preparation._checked()
    _require(
        final_entry is entry
        and source._ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1],
        "FINAL_ISSUANCE",
    )
    source._pure_final(state)
    _source_identity(state.frame.person, projection)
    fresh, fresh_counts = _measure(
        state.frame.person, state.frame.table("tax_unit"), projection.person
    )
    _require(
        source._encode(projection.evidence) == report_evidence
        and fresh_counts == counts
        and _projection_digest(fresh) == stamp
        and result.preparation is preparation
        and result.evidence == evidence
        and _projection_digest(result.tax_unit) == stamp,
        "FINAL_PROJECTION",
    )
    return result
