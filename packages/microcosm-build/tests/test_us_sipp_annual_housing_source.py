"""Invented-only integration of the real annual and frozen housing recipes.

The rows and stream audit facts are invented caller-supplied declarations.
Frozen source pins assert identity without authenticating any source bytes.
No source stream, fit, target, or genuine artifact is opened by these tests.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from types import SimpleNamespace

import pytest

from microcosm.build import sipp_annual_housing_source as evidence
from microcosm.build import sipp_annual_panel as annual
from microcosm.build import sipp_december_source as december
from microcosm.build import sipp_housing_projection as housing
from microcosm.build import sipp_measurements as measurements
from microcosm.graph.canonical import canonical_json
from tools.prepare_sipp_annual_panel_fixture import person_rows

SSUID1 = "00000000000001"
SSUID2 = "00000000000002"
DECLARED_SOURCE = measurements.SippSource(
    december._PINS.sha256, december._PINS.metadata_sha256, 2023, 2022
)
RENTER = {
    "ETENURE": "2",
    "ATENURE": "1",
    "TRENTMORT": "900",
    "ARENTMORT": "1",
    "TUTILS": "150",
    "AUTILS": "1",
    "ERENTSUB": "2",
    "ARENTSUB": "1",
    "EVOUCHER": "",
    "AVOUCHER": "0",
    "EPRDEBT": "",
    "APRDEBT": "0",
    "EMHDEBT": "",
    "AMHDEBT": "0",
    "EEHC_TEN": "2",
    "AEHC_TEN": "1",
    "EEHC_RENTSUB": "2",
    "AEHC_RENTSUB": "1",
    "EEHC_VOUCHER": "",
    "AEHC_VOUCHER": "0",
    "TEHC_ST": "24",
    "AEHC_ST": "1",
    "RHNUMU18": "0",
    "AHNUMU18": "1",
}


def regroup(rows):
    """Write complete invented roster copies before a test mutates evidence."""
    groups = defaultdict(list)
    for row in rows:
        if annual.live(row) and row["ERESIDENCEID"]:
            groups[row["SSUID"], row["ERESIDENCEID"], row["MONTHCODE"]].append(row)
    for members in groups.values():
        members.sort(key=lambda row: int(row["PNUM"]))
        total = sum(
            (Decimal(row["TPTOTINC"]) for row in members if row["TPTOTINC"]),
            Decimal(0),
        )
        for row in members:
            row.update(
                RHNUMPER=str(len(members)),
                RHNUMPERWT2=str(len(members)),
                THTOTINC=str(total),
            )
            for slot in range(1, 17):
                other = members[slot - 1] if slot <= len(members) else None
                row[f"RREL_PNUM{slot}"] = other["PNUM"] if other else ""
                row[f"RREL{slot}"] = (
                    ("99" if other["PNUM"] == row["PNUM"] else "19") if other else ""
                )
    return rows


def invented_rows():
    mover = person_rows(pnum=102, amount="20")
    for row in mover:
        row["THHLDSTATUS"] = "4"
        if int(row["MONTHCODE"]) >= 7:
            row["ERESIDENCEID"] = "100002"
    child = person_rows(pnum=103, amount="", weight="0")
    for row in child:
        row.update(TAGE_EHC="10", TPTOTINC="", APTOTINC="0")
    incomplete = person_rows(ssuid=SSUID2, amount="30")
    incomplete[0].update(TPTOTINC="", APTOTINC="1")
    return regroup(person_rows() + mover + child + incomplete)


def supplied_fixture(rows=None, housing_rows=None):
    """Create fake audit declarations from only the invented row list."""
    rows = invented_rows() if rows is None else rows
    selected = tuple(sorted({row["SSUID"] for row in rows}))
    invented_bytes = canonical_json(rows)
    digest = hashlib.sha256(invented_bytes).hexdigest()
    december_rows = [row for row in rows if row["MONTHCODE"] == "12"]
    anchored = [
        row
        for row in december_rows
        if annual.live(row) and annual.usable(row, "ERESIDENCEID")[0] is not None
    ]
    households = {(row["SSUID"], row["ERESIDENCEID"]) for row in anchored}
    audit = {"bytes": len(invented_bytes), "rows": len(rows), "sha256": digest}
    index = {
        **audit,
        "december_person_rows": len(december_rows),
        "eligible_households": len(households),
        "eligible_ssuids": len(selected),
    }
    panel = annual.Panel(
        {annual.key(row): dict(row) for row in rows}, selected, index, dict(audit)
    )
    source = evidence.SourceBinding(
        declared=DECLARED_SOURCE,
        panel_stream_sha256=digest,
        panel_stream_bytes=len(invented_bytes),
        panel_stream_rows=len(rows),
        index_december_person_rows=len(december_rows),
        index_eligible_households=len(households),
        index_eligible_ssuids=len(selected),
        evidence_origin="invented_only",
    )
    selection = evidence.SelectionBinding(
        annual.SELECTION, annual.SELECTION_LIMIT, selected, "first_32_ssuids"
    )
    if housing_rows is None:
        housing_rows = []
        for row in anchored:
            values = dict.fromkeys(housing.PROJECTION_FIELDS, "")
            values.update(RENTER)
            values.update({field: row[field] for field in evidence.SHARED_COLUMNS})
            values["RHNUMU18"] = str(
                sum(
                    member["SSUID"] == row["SSUID"]
                    and member["ERESIDENCEID"] == row["ERESIDENCEID"]
                    and bool(member["TAGE_EHC"])
                    and Decimal(member["TAGE_EHC"]) < 18
                    for member in anchored
                )
            )
            housing_rows.append(measurements.SippRow(values))
    parent = housing.project(
        housing_rows, source=DECLARED_SOURCE, selection=selection.housing_mode
    )
    return SimpleNamespace(
        panel=panel,
        source=source,
        selection=selection,
        housing_rows=housing_rows,
        parent=parent,
    )


def project(fixture, **overrides):
    arguments = {
        "housing": fixture.parent,
        "expected_housing_receipt_sha256": fixture.parent.receipt_sha256,
        "source": fixture.source,
        "selection_binding": fixture.selection,
        "housing_rows": fixture.housing_rows,
    }
    arguments.update(overrides)
    return evidence.project(fixture.panel, **arguments)


def decode(fixture, payload, **overrides):
    arguments = {
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_source_binding": fixture.source,
        "expected_recipe_sha256": evidence.RECIPE_SHA256,
        "expected_selection_binding": fixture.selection,
        "expected_housing_receipt_sha256": fixture.parent.receipt_sha256,
    }
    arguments.update(overrides)
    return evidence.decode_artifact(payload, **arguments)


def document(fixture):
    return json.loads(project(fixture).artifact_json)


def first_record(fixture):
    return document(fixture)["annual"]["records"][0]


def test_real_recipes_round_trip_mover_child_and_unavailable_month():
    fixture = supplied_fixture()
    parent_bytes = fixture.parent.receipt_json
    result = project(fixture)
    assert decode(fixture, result.artifact_json) == result
    assert fixture.parent.receipt_json == parent_bytes
    assert (
        housing.project(
            fixture.housing_rows, source=DECLARED_SOURCE, selection="first_32_ssuids"
        )
        == fixture.parent
    )
    encoded = json.loads(result.artifact_json)
    records = encoded["annual"]["records"]
    assert [record["key"] for record in records] == [
        [SSUID1, 100001],
        [SSUID1, 100002],
        [SSUID2, 100001],
    ]
    assert [record["amount"] for record in records] == ["120", "240", None]
    assert records[2]["reasons"] == ["missing_value"]
    assert records[0]["members"] == [[SSUID1, 101], [SSUID1, 103]]
    assert records[0]["children_under15"] == [[SSUID1, 103]]
    assert records[0]["weight"]["value"] == "1"
    assert records[0]["december_person_months"] == [
        [SSUID1, 101, 12],
        [SSUID1, 103, 12],
    ]
    assert [month["residence"] for month in records[1]["contributor_months"]] == (
        [100001] * 6 + [100002] * 6
    )
    assert [month["month"] for month in records[1]["contributor_months"]] == list(
        range(1, 13)
    )
    for row, original, predictor in zip(
        encoded["december_housing"]["records"],
        fixture.parent.households,
        fixture.parent.predictors,
        strict=True,
    ):
        for field in ("rent", "utilities"):
            expected = december._measurement(getattr(original, field))
            assert row[field] == json.loads(canonical_json(expected))
        assert row["tenure"] == original.tenure
        assert row["assistance"] == original.assistance
        assert row["exclusion"] == original.exclusion
        assert row["predictors"]["status"] == predictor.status
    diagnostics = annual.diagnostics(fixture.panel, DECLARED_SOURCE.data_sha256)
    aggregate = diagnostics["annual_december_adult_contributors"]
    assert aggregate["complete_household_sum"] == "360"
    assert aggregate["complete_household_weighted_sum"] == "600"
    assert aggregate["full_selected_household_sum"] is None
    assert aggregate["counts"]["complete_households"] == 2
    assert result.quality["annual"]["complete_households"] == 2
    assert result.quality["join"]["multi_residence_ssuids"] == 1
    assert encoded["december_housing"]["parent"]["receipt_sha256"] == (
        fixture.parent.receipt_sha256
    )


def test_private_values_never_enter_quality_or_admission():
    fixture = supplied_fixture()
    result = project(fixture)
    private = json.loads(result.artifact_json)
    receipt = json.loads(result.receipt_json)
    assert receipt["count_only"] is True
    assert receipt["amounts_present"] is False
    for identifier in (SSUID1, SSUID2):
        assert identifier not in result.receipt_json.decode()
    forbidden = {"value", "amount", "weighted_sum", "rent", "utilities", "rows"}

    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(result.quality)
    assert private["admission"]["fit_admission"] is None
    assert private["admission"]["release_admission"] is None
    assert not any(private["admission"].values())
    assert (
        private["source_binding"]["authenticated_source_bytes_established_here"]
        is False
    )
    assert (
        private["source_binding"]["declared_source_matches_panel_stream_digest"]
        is False
    )
    assert private["selection_binding"]["declarations_are_not_interchangeable"] is True


def test_recipe_has_exact_future_union_and_no_weight_extension():
    assert len(annual.FIELDS) == 67
    assert len(housing.PROJECTION_FIELDS) == 38
    assert len(evidence.SHARED_COLUMNS) == 10
    assert len(evidence.HOUSING_ADDITIONS) == 28
    assert len(evidence.FIELD_UNION) == 95
    assert set(evidence.FIELD_UNION) == set(annual.FIELDS) | set(
        housing.PROJECTION_FIELDS
    )
    assert annual.SELECTION_LIMIT == 32
    recipe = evidence.recipe_document()
    assert recipe["non_claims"]["weight_reference_fields_included"] is False
    assert hashlib.sha256(evidence.recipe_bytes()).hexdigest() == evidence.RECIPE_SHA256
    evidence.verify_recipe()


@pytest.mark.parametrize("component", ["annual", "housing", "contract", "recipe"])
def test_contract_or_module_pin_mutation_refuses(monkeypatch, component):
    fixture = supplied_fixture()
    if component == "annual":
        monkeypatch.setattr(evidence, "ANNUAL_MODULE_SHA256", "0" * 64)
    elif component == "housing":
        monkeypatch.setitem(
            evidence.FROZEN_HOUSING_MODULE_SHA256, housing.__name__, "0" * 64
        )
    elif component == "contract":
        monkeypatch.setattr(annual, "CONTRACT_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(evidence, "RECIPE_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        project(fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_sha256", "0" * 64),
        ("metadata_sha256", "0" * 64),
        ("survey_year", 2022),
        ("reference_year", 2023),
    ],
)
def test_wrong_source_or_vintage_refuses(field, value):
    fixture = supplied_fixture()
    source = dataclasses.replace(
        fixture.source, declared=dataclasses.replace(DECLARED_SOURCE, **{field: value})
    )
    with pytest.raises(ValueError):
        project(fixture, source=source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("annual_rule", "first_32_ssuids"),
        ("annual_limit", 256),
        ("annual_selected_ssuids", (SSUID2, SSUID1)),
        ("annual_selected_ssuids", (SSUID1,)),
        ("housing_mode", "all_ssuids"),
    ],
)
def test_wrong_selection_declaration_refuses(field, value):
    fixture = supplied_fixture()
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(
            fixture,
            selection_binding=dataclasses.replace(fixture.selection, **{field: value}),
        )


def test_wrong_parent_and_external_decode_expectations_refuse():
    fixture = supplied_fixture()
    result = project(fixture)
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, expected_housing_receipt_sha256="0" * 64)
    for override in (
        {"expected_sha256": "0" * 64},
        {"expected_recipe_sha256": "0" * 64},
        {"expected_housing_receipt_sha256": "0" * 64},
        {
            "expected_selection_binding": dataclasses.replace(
                fixture.selection, housing_mode="all_ssuids"
            )
        },
        {
            "expected_source_binding": dataclasses.replace(
                fixture.source, panel_stream_rows=999
            )
        },
        {
            "expected_source_binding": dataclasses.replace(
                fixture.source, index_eligible_ssuids=3
            )
        },
    ):
        with pytest.raises(evidence.AnnualHousingSourceError):
            decode(fixture, result.artifact_json, **override)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("annual", "records", 0, "amount"), "999"),
        (("annual", "records", 0, "reasons"), ["forged"]),
        (("annual", "records", 0, "contributor_months", 0, "month"), 2),
        (("december_housing", "records", 0, "rent", "value"), "999"),
        (("december_housing", "records", 0, "predictors", "status"), "forged"),
        (("december_housing", "parent", "receipt_sha256"), "0" * 64),
        (("annual", "passes", 0, "eligible_ssuids"), 3),
        (("source_binding", "authenticated_source_bytes_established_here"), True),
        (("selection_binding", "annual", "rule"), "first_32_ssuids"),
        (("quality", "annual", "complete_households"), 3),
        (("join", "households", 0, "members"), [[SSUID1, 101]]),
        (("admission", "fit_admission"), "approved"),
        (("implementation", "annual_module_sha256"), "0" * 64),
        (("recipe", "non_claims", "weight_reference_fields_included"), True),
    ],
)
def test_derived_and_binding_tampering_refuses_even_after_redigest(path, replacement):
    fixture = supplied_fixture()
    encoded = document(fixture)
    node = encoded
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, canonical_json(encoded))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("annual",),
        ("annual", "rows", 0),
        ("annual", "rows", 0, "values"),
        ("december_housing",),
        ("december_housing", "rows", 0),
        ("december_housing", "rows", 0, "values"),
        ("selection_binding", "annual"),
        ("source_binding",),
        ("quality",),
    ],
)
def test_extra_fields_at_nested_boundaries_refuse(path):
    fixture = supplied_fixture()
    encoded = document(fixture)
    node = encoded
    for key in path:
        node = node[key]
    node["unexpected"] = "not part of the contract"
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, canonical_json(encoded))


@pytest.mark.parametrize(
    "payload",
    [
        b'{"format":"x","format":"x"}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e9999}',
        b'{"x":"\\ud800"}',
        b"[" * 1500 + b"0" + b"]" * 1500,
        b"\xff",
        b"{}\n",
    ],
)
def test_malformed_nonfinite_deep_or_noncanonical_transport_refuses(payload):
    fixture = supplied_fixture()
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, payload)


def test_canonical_whitespace_and_bounded_bytes_refuse(monkeypatch):
    fixture = supplied_fixture()
    payload = project(fixture).artifact_json
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, payload + b"\n")
    monkeypatch.setattr(evidence, "MAX_ARTIFACT_BYTES", len(payload) - 1)
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, payload)


@pytest.mark.parametrize(
    "field", ["MAX_ANNUAL_ROWS", "MAX_HOUSING_ROWS", "MAX_HOUSEHOLDS"]
)
def test_small_resource_caps_refuse_without_large_allocations(monkeypatch, field):
    fixture = supplied_fixture()
    monkeypatch.setattr(evidence, field, 1)
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture)


@pytest.mark.parametrize("lane", ["annual", "december_housing"])
def test_duplicate_person_month_refuses_decode(lane):
    fixture = supplied_fixture()
    encoded = document(fixture)
    encoded[lane]["rows"].append(copy.deepcopy(encoded[lane]["rows"][0]))
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, canonical_json(encoded))


@pytest.mark.parametrize(
    "change",
    [
        "short_ssuid",
        "wrong_key",
        "wrong_residence",
        "wrong_person",
        "wrong_month",
        "omitted_child",
    ],
)
def test_person_residence_month_axis_mismatches_refuse(change):
    fixture = supplied_fixture()
    rows = list(fixture.housing_rows)
    values = dict(rows[0].values)
    if change == "short_ssuid":
        values["SSUID"] = "1"
    elif change == "wrong_residence":
        values["ERESIDENCEID"] = "100003"
    elif change == "wrong_person":
        values["PNUM"] = "110"
    elif change == "wrong_month":
        values["MONTHCODE"] = "11"
    elif change == "omitted_child":
        rows = [row for row in rows if row.values["PNUM"] != "103"]
    else:
        identity = next(iter(fixture.panel.rows))
        fixture.panel.rows[(identity[0], 110, identity[2])] = fixture.panel.rows.pop(
            identity
        )
    if change not in ("omitted_child", "wrong_key"):
        rows[0] = measurements.SippRow(values)
    with pytest.raises(ValueError):
        project(fixture, housing_rows=rows)


@pytest.mark.parametrize("lane", ["annual", "housing"])
@pytest.mark.parametrize("mutation", ["missing_field", "extra_field", "nonlexical"])
def test_creator_rejects_transport_that_decoder_would_refuse(lane, mutation):
    fixture = supplied_fixture()
    if lane == "annual":
        values = next(iter(fixture.panel.rows.values()))
        field = "APTOTINC"
    else:
        values = dict(fixture.housing_rows[0].values)
        fixture.housing_rows[0] = measurements.SippRow(values)
        field = "TRENTMORT"
    if mutation == "missing_field":
        del values[field]
    elif mutation == "extra_field":
        values["unexpected"] = "1"
    else:
        values[field] = 1
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture)


@pytest.mark.parametrize("missing", ["self", "child", "saturated_tail"])
def test_incomplete_december_membership_refuses_complete_housing_claim(missing):
    rows = invented_rows()
    for row in rows:
        if (
            row["SSUID"] != SSUID1
            or row["ERESIDENCEID"] != "100001"
            or row["MONTHCODE"] != "12"
        ):
            continue
        if missing == "saturated_tail":
            row.update(RREL_PNUM16="199", RREL16="19")
        else:
            target = row["PNUM"] if missing == "self" else "103"
            for slot in range(1, 17):
                if row[f"RREL_PNUM{slot}"] == target:
                    row[f"RREL_PNUM{slot}"] = row[f"RREL{slot}"] = ""
    fixture = supplied_fixture(rows)
    with pytest.raises(
        evidence.AnnualHousingSourceError, match="INCOMPLETE_DECEMBER_MEMBERSHIP"
    ):
        project(fixture)


@pytest.mark.parametrize("state", ["unanchored", "inactive"])
def test_unresolved_december_rows_preserved_and_housing_mismatch_refused(state):
    rows = invented_rows()
    extra = person_rows(ssuid=SSUID2, pnum=104)
    for row in extra:
        row.update(ERESIDENCEID="", ARESIDENCEID="0")
        if state == "inactive":
            row.update(THHLDSTATUS="5", THTOTINC="", AHTOTINC="0")
    fixture = supplied_fixture(rows + extra)
    result = project(fixture)
    encoded = json.loads(result.artifact_json)
    unresolved = encoded["join"]["unresolved_coverage"]
    assert len(unresolved) == 1
    assert unresolved[0]["counted_as_complete"] is False
    assert unresolved[0]["person_month"] == [SSUID2, 104, 12]
    assert len(encoded["annual"]["rows"]) == 60
    assert result.quality["join"]["unresolved_december_row_count"] == 1
    assert decode(fixture, result.artifact_json) == result
    unexpected = dict(fixture.housing_rows[-1].values)
    unexpected["PNUM"] = "104"
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(
            fixture,
            housing_rows=[*fixture.housing_rows, measurements.SippRow(unexpected)],
        )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("TPTOTINC", "", "missing_value"),
        ("APTOTINC", "0", "structural_out_of_universe"),
        ("APTOTINC", "", "missing_status"),
        ("RIN_UNIV", "2", "outside_or_unknown_frame"),
    ],
)
def test_unavailable_income_keeps_lexical_evidence_and_never_zero(field, value, reason):
    rows = regroup(person_rows())
    rows[0][field] = value
    fixture = supplied_fixture(rows)
    result = project(fixture)
    encoded = json.loads(result.artifact_json)
    record = encoded["annual"]["records"][0]
    assert record["amount"] is None
    assert reason in record["reasons"]
    assert encoded["annual"]["rows"][0]["values"][field] == value
    assert decode(fixture, result.artifact_json) == result


def test_missing_month_and_unknown_child_receipt_remain_unavailable():
    missing = supplied_fixture(person_rows()[1:])
    assert first_record(missing)["amount"] is None
    assert "missing_contributor_months" in first_record(missing)["reasons"]
    rows = person_rows()
    rows[0].update(TAGE_EHC="14", RSSI_MNYN="", ASSI_MNYN="0")
    fixture = supplied_fixture(rows)
    record = first_record(fixture)
    assert record["amount"] is None
    assert "unknown_child_receipt" in record["reasons"]
    assert record["contributor_months"][0]["scope"] == "unknown_child_receipt"


@pytest.mark.parametrize(
    "amount,flag,expected",
    [
        ("-5", "1", "-60"),
        ("0", "1", "0"),
        ("10", "2", "120"),
        ("10", "9.0", "120"),
    ],
)
def test_signed_zero_allocated_and_code9_values_preserved(amount, flag, expected):
    rows = person_rows(amount=amount)
    for row in rows:
        row["APTOTINC"] = flag
    fixture = supplied_fixture(rows)
    result = project(fixture)
    encoded = json.loads(result.artifact_json)
    record = encoded["annual"]["records"][0]
    assert record["amount"] == expected
    assert record["contributor_months"][0]["income_status_code"] == int(Decimal(flag))
    assert encoded["annual"]["rows"][0]["values"]["APTOTINC"] == flag
    assert decode(fixture, result.artifact_json) == result


def test_unknown_age_preserves_lower_bound_separately():
    rows = regroup(person_rows() + person_rows(pnum=102))
    for row in rows:
        if row["PNUM"] == "102":
            row["TAGE_EHC"] = ""
    fixture = supplied_fixture(rows)
    encoded = document(fixture)
    predictor = encoded["december_housing"]["records"][0]["predictors"]
    assert predictor["maximum_member_age"]["value"] is None
    assert predictor["maximum_member_age_lower_bound"]["value"] == "30"
    assert encoded["annual"]["records"][0]["amount"] is None
    assert "unknown_december_age" in encoded["annual"]["records"][0]["reasons"]


@pytest.mark.parametrize("state", ["copy_conflict", "owner", "assistance_unknown"])
def test_housing_unavailable_states_and_mortgage_not_injected_as_rent(state):
    fixture = supplied_fixture()
    rows = []
    for position, row in enumerate(fixture.housing_rows):
        values = dict(row.values)
        if values["SSUID"] == SSUID1 and values["ERESIDENCEID"] == "100001":
            if state == "copy_conflict" and position == 0:
                values["TRENTMORT"] = "901"
            elif state == "owner":
                values.update(
                    ETENURE="1",
                    EEHC_TEN="1",
                    EPRDEBT="1",
                    APRDEBT="1",
                    ERENTSUB="",
                    ARENTSUB="0",
                    EEHC_RENTSUB="",
                    AEHC_RENTSUB="0",
                )
            elif state == "assistance_unknown":
                values.update(
                    ERENTSUB="", ARENTSUB="", EEHC_RENTSUB="", AEHC_RENTSUB=""
                )
        rows.append(measurements.SippRow(values))
    fixture.housing_rows = rows
    fixture.parent = housing.project(
        rows, source=DECLARED_SOURCE, selection="first_32_ssuids"
    )
    result = project(fixture)
    record = json.loads(result.artifact_json)["december_housing"]["records"][0]
    if state == "copy_conflict":
        assert record["rent"]["value"] is None
        assert record["exclusion"] == "rent_copy_conflict"
    elif state == "owner":
        assert record["rent"]["value"] is None
        assert "mortgage_excluded" in record["rent"]["reasons"]
    else:
        assert record["assistance"] == "assistance_unknown"
        assert record["exclusion"] == "assistance_unknown"
    assert decode(fixture, result.artifact_json) == result


@pytest.mark.parametrize(
    "weights,expected,state",
    [
        (("2", "0"), "1", "positive"),
        (("0", "0"), "0", "zero"),
        (("", ""), None, "unknown"),
        (("2", ""), None, "unknown"),
    ],
)
def test_weight_denominator_keeps_every_member_and_unknowns(weights, expected, state):
    rows = regroup(
        person_rows(weight=weights[0]) + person_rows(pnum=102, weight=weights[1])
    )
    fixture = supplied_fixture(rows)
    result = project(fixture)
    encoded = json.loads(result.artifact_json)
    record = encoded["annual"]["records"][0]
    assert record["weight"]["value"] == expected
    assert record["weight_state"] == state
    assert len(record["members"]) == 2
    assert len(encoded["december_housing"]["records"][0]["person_weights"]) == 2
    assert encoded["december_housing"]["records"][0]["weight"]["value"] == expected
    assert record["amount"] is not None
    assert decode(fixture, result.artifact_json) == result


def write_invented_artifacts(output_dir):
    """Emit private invented evidence for the controlled final-head runner."""
    import os
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, exist_ok=True)
    fixture = supplied_fixture()
    result = project(fixture)
    assert decode(fixture, result.artifact_json) == result
    aggregate = annual.diagnostics(fixture.panel, DECLARED_SOURCE.data_sha256)[
        "annual_december_adult_contributors"
    ]
    assert aggregate["complete_household_sum"] == "360"
    assert aggregate["complete_household_weighted_sum"] == "600"
    assert aggregate["full_selected_household_sum"] is None
    outputs = {}
    for name, payload in (
        ("invented-annual-housing-evidence.json", result.artifact_json),
        ("invented-annual-housing-quality.json", result.receipt_json),
    ):
        path = output_dir / name
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        outputs[name] = {
            "path": str(path.resolve()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mode": "0600",
        }
    return {
        "invented_only": True,
        "source_authenticated": False,
        "selected_ssuids": 2,
        "december_residences": 3,
        "persons": 4,
        "person_month_rows": 48,
        "annual_complete_households": 2,
        "annual_incomplete_households": 1,
        "mover_persons": 1,
        "under15_december_members": 1,
        "zero_weight_december_members": 1,
        "parent_housing_receipt_sha256": fixture.parent.receipt_sha256,
        "annual_diagnostic_parity": True,
        "codec_round_trip": True,
        "amounts_reported_here": False,
        "outputs": outputs,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("APTOTINC", "-99"),
        ("TPTOTINC", ".N"),
        ("TPTOTINC", "NaN"),
        ("TPTOTINC", "Infinity"),
    ],
)
def test_unrecognized_annual_tokens_refuse_without_reclassification(field, value):
    rows = person_rows()
    rows[0][field] = value
    fixture = supplied_fixture(rows)
    with pytest.raises(ValueError):
        project(fixture)


@pytest.mark.parametrize(
    "field",
    [
        "index_december_person_rows",
        "index_eligible_households",
        "index_eligible_ssuids",
    ],
)
def test_external_source_index_counts_are_bound_on_project_and_decode(field):
    fixture = supplied_fixture()
    result = project(fixture)
    altered = dataclasses.replace(
        fixture.source, **{field: getattr(fixture.source, field) + 1}
    )
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, source=altered)
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, result.artifact_json, expected_source_binding=altered)


def test_supplied_rows_cannot_claim_mismatched_stream_is_same_source():
    fixture = supplied_fixture()
    ordinary = dataclasses.replace(
        fixture.source, evidence_origin="supplied_rows_unverified"
    )
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, source=ordinary)
    # Even matching declarations can never elevate supplied evidence to source authentication.
    fixture.panel.index_audit["sha256"] = DECLARED_SOURCE.data_sha256
    fixture.panel.panel_audit["sha256"] = DECLARED_SOURCE.data_sha256
    matching = dataclasses.replace(
        ordinary, panel_stream_sha256=DECLARED_SOURCE.data_sha256
    )
    result = project(fixture, source=matching)
    encoded = json.loads(result.artifact_json)
    assert (
        encoded["source_binding"]["declared_source_matches_panel_stream_digest"] is True
    )
    assert (
        encoded["source_binding"]["authenticated_source_bytes_established_here"]
        is False
    )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_nonfinite_inside_otherwise_valid_derived_record_refuses(constant):
    fixture = supplied_fixture()
    payload = project(fixture).artifact_json
    altered = payload.replace(b'"amount":"120"', b'"amount":' + constant.encode(), 1)
    assert altered != payload
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, altered)


def test_surrogate_inside_otherwise_valid_lexical_evidence_refuses():
    fixture = supplied_fixture()
    encoded = document(fixture)
    encoded["annual"]["rows"][0]["values"]["TPTOTINC"] = "\ud800"
    payload = json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(evidence.AnnualHousingSourceError):
        decode(fixture, payload)


def test_housing_parent_measurement_tampering_refuses_even_with_original_receipt():
    fixture = supplied_fixture()
    household = fixture.parent.households[0]
    altered_household = dataclasses.replace(
        household, rent=dataclasses.replace(household.rent, value=Decimal("901"))
    )
    altered_parent = dataclasses.replace(
        fixture.parent, households=(altered_household, *fixture.parent.households[1:])
    )
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, housing=altered_parent)


@pytest.mark.parametrize("eligible", [3, 40])
def test_selection_cannot_truncate_externally_bound_eligible_index(eligible):
    fixture = supplied_fixture()
    fixture.panel.index_audit.update(
        eligible_ssuids=eligible,
        eligible_households=eligible,
        december_person_rows=eligible,
    )
    binding = dataclasses.replace(
        fixture.source,
        index_eligible_ssuids=eligible,
        index_eligible_households=eligible,
        index_december_person_rows=eligible,
    )
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, source=binding)


def test_every_selected_ssuid_requires_eligible_december_evidence():
    rows = invented_rows()
    for row in rows:
        if row["SSUID"] == SSUID2 and row["MONTHCODE"] == "12":
            row["RIN_UNIV"] = "2"
    fixture = supplied_fixture(rows)
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not bytes",
        bytearray(b"{}"),
        b"null",
        b"[]",
        b"0",
        b'"string"',
        b'{"format":null,"format":null}',
        b"[" * 1500 + b"0" + b"]" * 1500,
    ],
)
def test_malformed_parent_transport_refuses_with_sanitized_error(payload):
    fixture = supplied_fixture()
    digest = (
        hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else "0" * 64
    )
    parent = dataclasses.replace(
        fixture.parent,
        receipt_json=payload,
        receipt_sha256=digest,
    )
    with pytest.raises(evidence.AnnualHousingSourceError):
        project(fixture, housing=parent, expected_housing_receipt_sha256=digest)


@pytest.mark.parametrize(
    "field,count", [("index_december_person_rows", 3), ("index_eligible_households", 2)]
)
def test_index_counts_cannot_understate_retained_coverage(field, count):
    fixture = supplied_fixture()
    source = dataclasses.replace(fixture.source, **{field: count})
    audit_field = field.removeprefix("index_")
    fixture.panel.index_audit[audit_field] = count
    with pytest.raises(evidence.AnnualHousingSourceError, match="INDEX_COVERAGE"):
        project(fixture, source=source)


@pytest.mark.parametrize("field", ["survey_year", "reference_year"])
def test_declared_vintage_is_a_strict_integer(field):
    fixture = supplied_fixture()
    declared = dataclasses.replace(
        fixture.source.declared,
        **{field: float(getattr(fixture.source.declared, field))},
    )
    with pytest.raises(evidence.AnnualHousingSourceError, match="VINTAGE_TYPE"):
        project(fixture, source=dataclasses.replace(fixture.source, declared=declared))


@pytest.mark.parametrize("invalid_key", [1, "invalid", ("00000000000001", True, 12)])
def test_writer_refuses_malformed_panel_keys_without_typeerror(invalid_key):
    fixture = supplied_fixture()
    row = next(iter(fixture.panel.rows.values()))
    fixture.panel.rows = {invalid_key: row}
    with pytest.raises(evidence.AnnualHousingSourceError, match="ANNUAL_ROW_KEY"):
        project(fixture)
