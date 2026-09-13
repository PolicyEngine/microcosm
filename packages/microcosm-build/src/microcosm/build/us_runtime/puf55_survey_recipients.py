"""Disjoint PUF55 recipient values from an actual, authenticated financial run.

Preparation only: no donor, fit, attachment or Population admission. Retained
modeled return roles are not observed filers or a current-money reconstruction.
The six existing money features retain their all-member aggregation contract.
Source SS reports/components/masks are detached projections on both clone arms;
the receiving Frame and its canonical benefit cells are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.fit import model_input
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

from . import graph_atomic_survey_financial as financial
from . import puf55_survey_ss_measurement as ss
from . import puf_full_source as puf_source
from . import support_provenance as provenance

source, full, support = ss.source, ss.full, ss.full.support
PROTOCOL = "microcosm.us.puf55-survey-recipients.v1"
PROFILES = (full.PUF55_SURVEY_SS, full.PUF55_SURVEY_SS_NO_TOTAL)
MONEY_PREDICTORS = full.PUF59.predictors[2:]
MONEY_COLUMNS = (
    ("employment_income_before_lsr",),
    ("self_employment_income_before_lsr",),
    ("taxable_interest_income",),
    ("non_qualified_dividend_income", "qualified_dividend_income"),
    ("short_term_capital_gains",),
    ("long_term_capital_gains_before_response",),
)
_STATUS = {
    "SINGLE": 1,
    "JOINT": 2,
    "SEPARATE": 3,
    "HEAD_OF_HOUSEHOLD": 4,
    "SURVIVING_SPOUSE": 5,
}
MAX_RECEIPT_BYTES = 128 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError("PUF55_SURVEY_RECIPIENTS_" + reason)


def _table_digest(table):
    """Exact axes, storage domains, values and unknowns for our detached tables."""
    _require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE")
    digest = hashlib.sha256()
    digest.update(
        source._encode(
            [
                list(table.columns),
                type(table.columns).__name__,
                str(table.columns.dtype),
                list(table.columns.names),
                type(table.index).__name__,
                str(table.index.dtype),
                list(table.index.names),
                [
                    [
                        str(d),
                        getattr(d, "storage", None),
                        str(getattr(d, "na_value", "")),
                    ]
                    for d in table.dtypes
                ],
            ]
        )
    )
    for row in table.itertuples(index=True, name=None):
        for value in row:
            payload = source._frame_cell_encode(value)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    for series in (pd.Series(table.index), *(table[c] for c in table)):
        if series.dtype == np.dtype("object"):
            # These projections use object storage only for non-null strings.
            # Their bytes were encoded above; ndarray.tobytes() here would hash
            # process-local Python pointers and make equivalent replay unstable.
            _require(
                series.map(lambda value: type(value) is str).all(),
                "OBJECT_STRING_STORAGE",
            )
            continue
        for part in financial.reconstruction._storage_parts(
            series, np.ones(len(series), dtype=np.bool_)
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _literal_snapshot(state):
    """Capture only the selected literal rows against the actual closed pin.

    Immutable strings come from the independently hashed original CSV stream,
    not from a detached qualifier result or carried reason/allocation columns.
    The existing coverage capture owns file handling; this adds no source issuer.
    """
    owner = ss.reports.coverage
    native = state.native[1]
    entry = source.asec_native._ISSUED.get(id(native))
    _require(
        entry is not None and entry[0]() is native and entry[1] == native.payload,
        "LITERAL_NATIVE_OWNER",
    )
    pins = tuple(pin for pin in owner._MEMBER_PINS if pin[0] == 2024)
    _require(len(pins) == 1, "LITERAL_PIN")
    year, member, archive, digest, rows, size = pins[0]
    retained = [
        row
        for row in entry[2].coverage.receipt["sources"]
        if row["source_year"] == year
    ]
    _require(
        len(retained) == 1
        and all(
            retained[0][key] == value
            for key, value in (
                ("member", member),
                ("archive_sha256", archive),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "LITERAL_NATIVE_PIN",
    )
    people = state.frame.person
    keys = set(
        people.loc[
            people[provenance.support_channel_column("person")].eq("asec"), "PERIDNUM"
        ]
    )
    reader = ss.reports.source_csv_builtin.capture_csv_reader(csv)
    _require(reader is not None, "LITERAL_CSV_READER")
    captured_rows, count, total = {}, 0, 0
    hashed = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="microcosm-puf55-ss-literals-") as tmp:
        path = Path(tmp) / member
        owner._capture(
            state.root / "asec" / member,
            path,
            size=size,
            digest=digest,
            budget=[owner._BODY_MAX],
        )
        with path.open("rb") as stream:

            def lines():
                nonlocal total
                while chunk := stream.readline(owner._ROW_MAX + 1):
                    _require(
                        len(chunk) <= owner._ROW_MAX and total + len(chunk) <= size,
                        "LITERAL_BYTES",
                    )
                    first = total == 0
                    total += len(chunk)
                    hashed.update(chunk)
                    yield chunk.decode("utf-8-sig" if first else "utf-8")

            records = reader(lines(), strict=True)
            header = next(records, [])
            _require(
                len(header) == len(set(header))
                and set(ss.reports.READ_COLUMNS) <= set(header),
                "LITERAL_HEADER",
            )
            positions = [header.index(c) for c in ss.reports.READ_COLUMNS]
            for record in records:
                count += 1
                _require(count <= rows and len(record) == len(header), "LITERAL_ROWS")
                selected = tuple(record[i] for i in positions)
                if selected[0] in keys:
                    _require(selected[0] not in captured_rows, "LITERAL_DUPLICATE")
                    captured_rows[selected[0]] = selected
    _require(
        count == rows
        and total == size
        and hashed.hexdigest() == digest
        and set(captured_rows) == keys,
        "LITERAL_SOURCE_BYTES",
    )
    return pins[0], tuple(captured_rows[key] for key in sorted(captured_rows))


def _source_report(state, snapshot):
    """Pure full projection from pinned literals and retained native source cells."""
    pin, records = snapshot
    _require(
        tuple(p for p in ss.reports.coverage._MEMBER_PINS if p[0] == 2024) == (pin,),
        "LITERAL_PIN_CHANGED",
    )
    people = state.frame.person
    channel = people[provenance.support_channel_column("person")]
    result = pd.DataFrame(index=pd.Index(people.person_id, name="person_id"))
    result["native_person_id"] = people[
        provenance.spine_source_id_column("person")
    ].to_numpy()
    result["source"] = channel.to_numpy()
    result["social_security_source_total"] = np.nan
    result["source_reporting_universe"] = False
    result["source_reporting_unit"] = "person_report_record"
    components = ss.reports.basis_owner.COMPONENTS
    for column in components:
        result[column], result["allowed_" + column] = np.nan, True
    result["basis_origin"] = "unresolved"
    result["allocation_origin"] = "unresolved_allocation_provenance"
    asec = people.loc[channel.eq("asec")]
    literals = (
        pd.DataFrame(records, columns=ss.reports.READ_COLUMNS)
        .set_index("PERIDNUM", drop=False)
        .loc[asec.PERIDNUM]
    )
    _require(
        np.array_equal(literals.A_AGE.astype("int64"), asec.A_AGE)
        and np.array_equal(
            ss.reports._codes(literals.SS_VAL, range(100000)),
            ss.reports._numeric(asec.SS_VAL),
        ),
        "LITERAL_SOURCE_IDENTITY",
    )
    amount, basis, allowed, labels = ss.reports.basis_owner.asec_reporting_basis(
        ss.reports._numeric(asec.SS_VAL),
        ss.reports._numeric(asec.A_AGE),
        ss.reports._codes(literals.SS_YN, {0, 1, 2}),
        ss.reports._codes(literals.RESNSS1, range(9)),
        ss.reports._codes(literals.RESNSS2, range(9)),
    )
    ids = asec.person_id.to_numpy()
    result.loc[ids, "social_security_source_total"] = amount
    result.loc[ids, "source_reporting_universe"] = ss.reports._numeric(asec.A_AGE) >= 15
    result.loc[ids, "source_reporting_unit"] = (
        "person_report_may_combine_family_payments"
    )
    for j, column in enumerate(components):
        result.loc[ids, column], result.loc[ids, "allowed_" + column] = (
            basis[:, j],
            allowed[:, j],
        )
    result.loc[ids, "basis_origin"] = labels
    result.loc[ids, "allocation_origin"] = ss.reports._allocation_labels(
        literals
    ).allocation_origin.to_numpy()
    acs = people.loc[channel.eq("acs")]
    eligible = ss.reports._numeric(acs.AGEP) >= 15
    total = ss.reports._numeric(acs.acs_social_security_income)
    result.loc[acs.person_id, "social_security_source_total"] = total
    result.loc[acs.person_id, "source_reporting_universe"] = eligible
    zero = acs.person_id.to_numpy()[eligible & (total == 0)]
    result.loc[zero, list(components)] = 0.0
    result.loc[zero, ["allowed_" + c for c in components]] = False
    result.loc[zero, "basis_origin"] = "known_total_zero"
    result.loc[acs.person_id.to_numpy()[eligible & (total > 0)], "basis_origin"] = (
        "acs_combined_positive_requires_model"
    )
    result.loc[acs.person_id.to_numpy()[~eligible], "basis_origin"] = (
        "acs_below15_outside_reporting_universe"
    )
    return result


def _aligned(source_frame, frame, entity):
    original, receiving = source_frame.table(entity), frame.table(entity)
    identity = entity + "_id"
    key, arm = (
        provenance.support_source_id_column(entity),
        provenance.support_clone_index_column(entity),
    )
    for column in (
        original[identity],
        receiving[identity],
        receiving[key],
        receiving[arm],
    ):
        ss._axis(column)
    _require(original[identity].is_unique and receiving[identity].is_unique, "AXIS")
    pairs = receiving[[key, arm]]
    _require(
        receiving[arm].isin((0, 1)).all()
        and not pairs.duplicated().any()
        and set(receiving[key]) == set(original[identity])
        and pairs.groupby(key, sort=False).size().eq(2).all(),
        "CLONE_PAIRS",
    )
    aligned = original.set_index(identity).loc[receiving[key]].copy(deep=True)
    for column in (
        provenance.spine_source_id_column(entity),
        provenance.support_channel_column(entity),
    ):
        _require(
            np.array_equal(aligned[column].to_numpy(), receiving[column].to_numpy()),
            "CLONE_SOURCE_IDENTITY",
        )
    aligned.index = pd.Index(receiving[identity].to_numpy(copy=True), name=identity)
    return aligned


def _current_money_surface(frame, recipient_mask):
    """Aggregate already-qualified financial leaves; issue no source authority.

    The public caller rechecks the actual financial run before this arithmetic.
    Its source owner already verifies ACS adjustment and age-universe zeros.
    A raw stacked-spine universe gate would incorrectly require ACS columns on
    an ASEC-only surface and is not the input boundary for these produced leaves.
    """
    person, units = frame.person, frame.table("tax_unit")
    plans = {}
    for predictor, columns in zip(MONEY_PREDICTORS, MONEY_COLUMNS, strict=True):
        plan = support._strict_predictor_source_plan(
            predictor, tax_unit=units, person=person
        )
        _require(
            plan.entity == "person" and plan.columns == columns,
            "CURRENT_MONEY_SOURCE_PLAN",
        )
        plans[predictor] = plan
    support._require_complete_recipient_predictor_sources(
        frame, recipient_mask, MONEY_PREDICTORS, source_plans=plans
    )
    features = support._tax_unit_feature_frame(
        frame, MONEY_PREDICTORS, preserve_nulls=True, source_plans=plans
    )
    support._require_complete_recipient_predictors(
        features, recipient_mask, MONEY_PREDICTORS
    )
    evidence = {
        "input_boundary": "current_financial_person_leaves_after_upstream_qualification",
        "aggregation": "sum_all_modeled_tax_unit_members",
        "predictor_source_mapping": {
            predictor: {"entity": plan.entity, "columns": list(plan.columns)}
            for predictor, plan in plans.items()
        },
        "recipient_tax_unit_rows": int(recipient_mask.sum()),
        "recipient_feature_values_sha256": support._predictor_feature_values_sha256(
            features.loc[recipient_mask, list(MONEY_PREDICTORS)],
            units.loc[recipient_mask, "tax_unit_id"],
        ),
        "raw_acs_universe_requalified_here": False,
        "source_admission_issued": False,
    }
    evidence["sha256"] = support._receipt_sha256(evidence)
    return features, evidence


def _project(source_frame, frame, report, measured):
    """Pure projection after live ownership checks; never an authority issuer."""
    people, units = frame.person, frame.table("tax_unit")
    original_people = _aligned(source_frame, frame, "person")
    original_units = _aligned(source_frame, frame, "tax_unit")
    unit_source = units.set_index("tax_unit_id")[
        provenance.support_source_id_column("tax_unit")
    ]
    _require(
        np.array_equal(
            people.tax_unit_role_input.to_numpy(),
            original_people.tax_unit_role_input.to_numpy(),
        )
        and np.array_equal(
            units.filing_status_input.to_numpy(),
            original_units.filing_status_input.to_numpy(),
        )
        and np.array_equal(
            people.person_tax_unit_id.map(unit_source).to_numpy(),
            original_people.person_tax_unit_id.to_numpy(),
        ),
        "MODELED_ROLE_INHERITANCE",
    )
    person_ids = people[provenance.support_source_id_column("person")].to_numpy()
    person_report = report.loc[person_ids].copy(deep=True)
    person_report.index = pd.Index(
        people.person_id.to_numpy(copy=True), name="person_id"
    )
    # Recheck every receiving role structure, and independently reproduce the
    # source measurement on both arms. Invalid roles never select route eight.
    received_measurement, _ = ss._measure(people, units, person_report)
    expected = measured.loc[
        units[provenance.support_source_id_column("tax_unit")]
    ].copy(deep=True)
    expected.index = received_measurement.index
    _require(
        ss._projection_digest(received_measurement) == ss._projection_digest(expected),
        "CLONE_MEASUREMENT",
    )
    dependent_count = (
        people.tax_unit_role_input.eq("DEPENDENT")
        .groupby(people.person_tax_unit_id)
        .sum()
    )
    first = puf_source.puf_2015_receiver_size_measurement(
        units.filing_status_input.map(_STATUS).to_numpy(dtype=np.int64),
        dependent_count.reindex(units.tax_unit_id).to_numpy(dtype=np.int64),
    )
    for name, values in first.items():
        received_measurement[name] = values
    recipients = (
        units[provenance.support_clone_index_column("tax_unit")].eq(1).to_numpy()
    )
    features, money_evidence = _current_money_surface(frame, recipients)
    for name in MONEY_PREDICTORS:
        received_measurement[name] = features[name].to_numpy(
            dtype=np.float64, copy=True
        )
    matrices, selected = [], []
    for profile in PROFILES:
        mask = recipients & received_measurement[ss.ROUTE].eq(profile.value).to_numpy()
        ids = units.loc[mask, "tax_unit_id"].to_numpy(copy=True)
        selected.extend(ids.tolist())
        if not len(ids):
            continue
        _require(
            len(ids) * (1 + len(profile.predictors)) * 8 <= RAW_BYTES_MAX_BYTES,
            "MATRIX_SIZE",
        )
        table = received_measurement.loc[ids, list(profile.predictors)].astype(
            "float64"
        )
        payload = model_input.encode_recipient_matrix(
            table, entity="tax_unit", entity_ids=ids
        )
        _require(len(payload) <= RAW_BYTES_MAX_BYTES, "MATRIX_SIZE")
        matrices.append((profile.value, payload))
    _require(
        len(selected) == len(set(selected))
        and set(selected) == set(units.loc[recipients, "tax_unit_id"]),
        "RECIPIENT_PARTITION",
    )
    return person_report, received_measurement, tuple(matrices), money_evidence


@dataclass(frozen=True)
class Puf55SurveyRecipients:
    """Detached outputs with their actual upstream run; public copies are values."""

    financial_run: financial.AtomicSurveyFinancialRunValues
    person: pd.DataFrame
    tax_unit: pd.DataFrame
    matrices: tuple[tuple[str, bytes], ...]
    receipt: bytes


def _result_stamp(result):
    _require(
        type(result) is Puf55SurveyRecipients
        and type(result.receipt) is bytes
        and type(result.matrices) is tuple
        and all(
            type(row) is tuple
            and len(row) == 2
            and type(row[0]) is str
            and type(row[1]) is bytes
            for row in result.matrices
        ),
        "RESULT_TYPE",
    )
    return (
        _table_digest(result.person),
        _table_digest(result.tax_unit),
        result.matrices,
        result.receipt,
    )


def qualify_puf55_survey_recipients(financial_run):
    """Require the actual financial run and requalify source observations.

    This admits numerical preparation under retained modeled roles only. It
    cannot establish current-money tax-unit reconstruction or release validity.
    Source receipt objects and caller-supplied role tables are not accepted.
    """
    financial.check_atomic_survey_financial_run(financial_run)
    entry = financial._run_entry(financial_run)
    state = entry[2]
    preparation = state.prefix.preparation
    source_frame = state.preparation_entry[2].frame
    frame = state.financial_population.frame
    snapshot = _literal_snapshot(state.preparation_entry[2])
    report = ss.reports.qualify_current_social_security(preparation)
    ss._source_identity(source_frame.person, report)
    authoritative_report = _source_report(state.preparation_entry[2], snapshot)
    _require(
        _table_digest(report.person) == _table_digest(authoritative_report),
        "COMPLETE_SOURCE_REPORT",
    )
    expected_measurement, measurement_counts = ss._measure(
        source_frame.person, source_frame.table("tax_unit"), authoritative_report
    )
    person, units, matrices, money_evidence = _project(
        source_frame, frame, authoritative_report, expected_measurement
    )
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "financial_run_sha256": financial.codec.sha(entry[1]),
            "preparation_sha256": financial.codec.sha(state.preparation_entry[1]),
            "person_projection_sha256": _table_digest(person),
            "tax_unit_projection_sha256": _table_digest(units),
            "source_report_projection_sha256": _table_digest(authoritative_report),
            "asec_current_literal_member_sha256": snapshot[0][3],
            "asec_selected_literal_rows": len(snapshot[1]),
            "asec_income_year": 2024,
            "asec_interview_year": 2025,
            "acs_income_window": "rolling prior12 months at2024 interview; ADJINC price basis",
            "reporting_grain": "person report; family/child aggregation and repayment/Railroad alignment unresolved",
            "ss_measurement": ss.MEASUREMENT,
            "native_measurement_counts": measurement_counts,
            "measurement_knownness": "all included HEAD/actual JOINT SPOUSE reports available; no dependent, component or partial-sum fallback",
            "money_predictor_evidence": money_evidence,
            "routes": [
                {
                    "profile": name,
                    "matrix_sha256": financial.codec.sha(payload),
                    "rows": len(model_input.decode_recipient_matrix(payload).features),
                    "predictors": list(full.PufOutputProfile(name).predictors),
                    "outputs": list(full.PufOutputProfile(name).targets),
                }
                for name, payload in matrices
            ],
            "recipient_tax_units": sum(
                len(model_input.decode_recipient_matrix(p).features)
                for _, p in matrices
            ),
            "receiving_tax_units": len(units),
            "receiving_persons": len(person),
            "partition": "all clone-one tax units exactly once; empty routes omitted in nine/eight order",
            "role_boundary": "retained modeled HEAD/actual JOINT SPOUSE roles; not observed filer status",
            "money_contract": "existing six features sum all modeled tax-unit members; current financial leaves only",
            "current_money_tax_units_reconstructed_here": False,
            "person_social_security_changed": False,
            "population_changed": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        },
        maximum=MAX_RECEIPT_BYTES,
    )
    result = Puf55SurveyRecipients(financial_run, person, units, matrices, receipt)
    stamp = _result_stamp(result)
    financial.check_atomic_survey_financial_run(financial_run)
    financial._pure_run(financial_run, entry)
    fresh = _source_report(state.preparation_entry[2], snapshot)
    fresh_measurement, _ = ss._measure(
        source_frame.person, source_frame.table("tax_unit"), fresh
    )
    final_person, final_units, final_matrices, final_money = _project(
        source_frame, frame, fresh, fresh_measurement
    )
    _require(
        financial._run_entry(financial_run) is entry
        and result.financial_run is financial_run
        and _result_stamp(result) == stamp
        and _table_digest(fresh) == _table_digest(authoritative_report)
        and (
            _table_digest(final_person),
            _table_digest(final_units),
            final_matrices,
            receipt,
        )
        == stamp
        and final_money == money_evidence,
        "FINAL_PROJECTION",
    )
    return result
