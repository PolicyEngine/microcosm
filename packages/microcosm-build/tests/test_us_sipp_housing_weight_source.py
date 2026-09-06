"""Invented CSV only: December weight/reference alternatives preserve parents."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from decimal import localcontext

import pytest

from microcosm.build import sipp_december_source as s
from microcosm.build import sipp_housing_projection as hp
from microcosm.build import sipp_housing_weight_source as ws
from microcosm.build import sipp_measurements as m
from microcosm.graph.canonical import canonical_json

SOURCE = m.SippSource(s._PINS.sha256, s._PINS.metadata_sha256, 2023, 2022)


def row(pnum=101, weight="100", **overrides):
    values = dict.fromkeys((*hp.PROJECTION_FIELDS, *ws.FIELDS), "")
    values.update(
        SSUID="0001",
        PNUM=str(pnum),
        MONTHCODE="12",
        ERESIDENCEID="100001",
        WPFINWGT=weight,
        ETENURE="2",
        ATENURE="1",
        TRENTMORT="900",
        ARENTMORT="1",
        TUTILS="100",
        AUTILS="1",
        ERENTSUB="2",
        ARENTSUB="1",
        AVOUCHER="0",
        EPRDEBT="",
        APRDEBT="0",
        EMHDEBT="",
        AMHDEBT="0",
        EEHC_TEN="2",
        AEHC_TEN="1",
        EEHC_RENTSUB="2",
        AEHC_RENTSUB="1",
        AEHC_VOUCHER="0",
        TEHC_ST="24",
        AEHC_ST="1",
        RHNUMPER="1",
        AHNUMPER="9",
        RHNUMU18="0",
        AHNUMU18="9",
        THTOTINC="2000",
        AHTOTINC="9",
        TAGE_EHC="30",
        ERELRPE="1" if pnum == 101 else "7",
        ARELRP="1",
        EHRESIDENCID="100001",
        AHRESIDENCID="1",
        SHHADID="011",
        THHLDSTATUS="1",
        RIN_UNIV="1",
        GVARSTR="001",
        GHLFSAM="01",
    )
    values.update(overrides)
    return m.SippRow(values)


def parent(rows):
    return hp.project(rows, source=SOURCE, selection="first_32_ssuids")


def project(rows, *, original=None, complete_membership=True):
    p = original or parent(rows)
    return ws.project(
        rows,
        parent=p,
        expected_parent_receipt_sha256=p.receipt_sha256,
        source=SOURCE,
        complete_membership=complete_membership,
    )


def candidates(result):
    h = result.households[0]
    return h["december_member_mean_weight"], h[
        "interview_reference_person_member_december_weight"
    ]


def test_two_source_alternatives_preserve_weights_zero_members_and_frozen_bytes():
    rows = [row(weight="0.000000"), row(102, "300")]
    p = parent(rows)
    original = (
        p.receipt_json,
        hp._measurement_digest(p.households),
        hp._predictor_digest(p.predictors),
        s._membership(p.households),
    )
    result = project(rows, original=p)
    mean, reference = candidates(result)
    assert mean["value"] == "150" and mean["sum_member_weights"] == "300"
    assert mean["member_count"] == 2 and mean["zero_member_weights"] == 1
    assert reference["value"] == "0" and reference["status"] == "zero"
    assert reference["proven_december_reference_person"] is False
    assert (
        p.receipt_json,
        hp._measurement_digest(p.households),
        hp._predictor_digest(p.predictors),
        s._membership(p.households),
    ) == original
    assert result.diagnostics["admitted_unassisted_renters"]["households"] == 1
    assert result.diagnostics["reported_rent_donors"]["households"] == 1


def test_csv_projection_and_codec_roundtrip_private_rows_separate_from_aggregate_receipt():
    rows = [row(), row(102, "300")]
    header = ["IGNORED", *reversed(ws.FIELDS)]
    text = (
        "|".join(header)
        + "\n"
        + "\n".join(
            "|".join("not-read" if f == "IGNORED" else r.values[f] for f in header)
            for r in rows
        )
    )
    decoded_rows = list(ws.iter_csv_rows(io.StringIO(text)))
    p = parent(rows)
    result = project(decoded_rows, original=p)
    decoded = ws.decode_artifact(
        result.artifact_json,
        expected_sha256=result.artifact_sha256,
        parent=p,
        expected_parent_receipt_sha256=p.receipt_sha256,
        source=SOURCE,
    )
    assert decoded == result
    assert "households" not in json.loads(result.receipt_json)
    assert "rows" not in json.loads(result.receipt_json)
    assert "0001" not in result.receipt_json.decode()
    assert candidates(result)[0]["value"] == "200"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "width"])
def test_csv_header_and_row_integrity(kind):
    fields = list(ws.FIELDS)
    if kind == "missing":
        fields.remove("ARELRP")
    if kind == "duplicate":
        fields.append("ARELRP")
    body = "|".join(fields) + "\n" + "|".join("1" for _ in fields)
    if kind == "width":
        body += "|extra"
    with pytest.raises(ws.WeightSourceError, match="HEADER|WIDTH"):
        list(ws.iter_csv_rows(io.StringIO(body)))


@pytest.mark.parametrize(
    "refs, expected",
    [
        (["7", "7"], "no_interview_reference_member"),
        (["1", "2"], "multiple_interview_reference_members"),
        (["1", ""], "reference_classification_incomplete"),
    ],
)
def test_no_implicit_reference_fallback(refs, expected):
    result = project([row(101, "10", ERELRPE=refs[0]), row(102, "90", ERELRPE=refs[1])])
    mean, reference = candidates(result)
    assert mean["value"] == "50"
    assert reference["value"] is None and reference["status"] == expected


@pytest.mark.parametrize(
    "overrides, status",
    [
        ({"RIN_UNIV": "2"}, "reference_member_out_of_monthly_frame"),
        ({"RIN_UNIV": ""}, "reference_monthly_universe_unknown"),
        ({"THHLDSTATUS": "9"}, "reference_classification_incomplete"),
        ({"ERELRPE": "19"}, "reference_classification_incomplete"),
        ({"ARELRP": "-99"}, "reference_classification_incomplete"),
        ({"ARELRP": "10"}, "reference_classification_incomplete"),
        ({"ARELRP": "0"}, "reference_classification_incomplete"),
    ],
)
def test_reference_requires_known_documented_universe_and_flag(overrides, status):
    _, reference = candidates(project([row(**overrides)]))
    assert reference["value"] is None and reference["status"] == status


def test_mover_interview_niu_stays_out_and_does_not_rekey_december_roster():
    result = project(
        [
            row(),
            row(
                102,
                "200",
                THHLDSTATUS="4",
                ERELRPE="",
                ARELRP="0",
                EHRESIDENCID="",
                AHRESIDENCID="0",
                SHHADID="012",
            ),
        ]
    )
    mean, reference = candidates(result)
    assert mean["value"] == "150" and reference["value"] == "100"
    h = result.households[0]
    assert h["key"] == ["0001", 100001]
    assert h["members"][1]["fields"]["ERELRPE"]["state"] == "not_in_universe"
    assert h["quality"]["EHRESIDENCID"]["unknown_or_inapplicable_members"] == 1
    assert h["quality"]["SHHADID"]["state"] == "conflict"


def test_address_mismatch_and_imputed_reference_are_visible_without_changing_candidate():
    result = project([row(ARELRP="2", EHRESIDENCID="200001", AHRESIDENCID="9")])
    _, reference = candidates(result)
    assert (
        reference["value"] == "100"
        and reference["relationship_quality"] == "hot_deck_imputed"
    )
    assert reference["interview_december_address_comparison"] == "different_identifier"
    h = result.households[0]
    assert (
        h["members"][0]["fields"]["EHRESIDENCID"]["quality"]
        == "component_derived_allocation_unresolved"
    )


@pytest.mark.parametrize(
    "raw, state",
    [
        ("", "blank"),
        (None, "null"),
        (".N", "invalid_lexical"),
        ("-999", "invalid_domain"),
        ("500000", "invalid_domain"),
    ],
)
def test_no_guessed_address_missing_decoder(raw, state):
    h = project([row(EHRESIDENCID=raw)]).households[0]
    assert h["members"][0]["fields"]["EHRESIDENCID"]["lexical_state"] == state
    assert (
        h["interview_reference_person_member_december_weight"][
            "interview_december_address_comparison"
        ]
        == "value_unknown"
    )


@pytest.mark.parametrize("bad_weight", ["", None, "-1", "10000000", "NaN"])
def test_missing_invalid_member_weight_propagates_without_shrinking_mean(bad_weight):
    result = project([row(), row(102, bad_weight)])
    mean, reference = candidates(result)
    assert (
        mean["value"] is None
        and mean["member_count"] == 2
        and mean["unknown_member_weights"] == 1
    )
    assert reference["value"] == "100"


def test_missing_reference_weight_cannot_fall_back_to_other_member():
    mean, reference = candidates(project([row(weight=""), row(102, "300")]))
    assert mean["value"] is None
    assert (
        reference["value"] is None and reference["status"] == "reference_weight_unknown"
    )


def test_absent_flag_and_valid_zero_remain_distinct():
    r = row(weight="0")
    del r.values["ARELRP"]
    result = project([r])
    mean, reference = candidates(result)
    assert mean["value"] == "0" and reference["value"] is None
    assert (
        result.households[0]["members"][0]["fields"]["ARELRP"]["lexical_state"]
        == "absent"
    )


def test_inconsistent_design_codes_are_measured_without_splitting_household():
    result = project([row(), row(102, "200", GVARSTR="240", GHLFSAM="02")])
    assert len(result.households) == 1
    assert result.households[0]["quality"]["GVARSTR"]["state"] == "conflict"
    assert result.households[0]["quality"]["GHLFSAM"]["state"] == "conflict"


def test_complete_parent_membership_is_required_and_changed_member_cannot_pass():
    rows = [row(), row(102, "300")]
    p = parent(rows)
    with pytest.raises(ws.WeightSourceError, match="COMPLETE_MEMBERSHIP"):
        project(rows, original=p, complete_membership=False)
    with pytest.raises(ws.WeightSourceError, match="ROSTER"):
        project(rows[:1], original=p)
    with pytest.raises(ws.WeightSourceError, match="ROSTER"):
        project([row(), row(103, "300")], original=p)
    with pytest.raises(ws.WeightSourceError, match="WEIGHT_DISAGREEMENT"):
        project([row(weight="101"), rows[1]], original=p)
    with pytest.raises(ws.WeightSourceError, match="WEIGHT_DISAGREEMENT"):
        project([row(weight="100.0"), rows[1]], original=p)


@pytest.mark.parametrize("month", ["12", "11"])
def test_duplicate_person_month_refuses_even_if_not_december(month):
    r = row(MONTHCODE=month)
    with pytest.raises(m.MeasurementError, match="DUPLICATE_PERSON_MONTH"):
        project([row(), r, r], original=parent([row()]))


def test_fixed_decimal_arithmetic_and_sorted_input_are_replay_stable():
    rows = [row(weight="1.000001"), row(102, "1.000002"), row(103, "0")]
    p = parent(rows)
    baseline = project(rows, original=p)
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.rounding = "ROUND_DOWN"
        assert project(list(reversed(rows)), original=p) == baseline
    mean, _ = candidates(baseline)
    assert mean["sum_member_weights"] == "2.000003"
    assert (
        mean["value"]
        == "0.66666766666666666666666666666666666666666666666666666666666666666666666666666667"
    )


def test_parent_identity_and_live_object_tampering_refuse():
    rows = [row()]
    p = parent(rows)
    with pytest.raises(ws.WeightSourceError, match="PARENT_RECEIPT_SHA"):
        ws.project(
            rows,
            parent=p,
            expected_parent_receipt_sha256="0" * 64,
            source=SOURCE,
            complete_membership=True,
        )
    changed = dataclasses.replace(
        p, households=(dataclasses.replace(p.households[0], exclusion="fake"),)
    )
    with pytest.raises(ws.WeightSourceError, match="PARENT_CONTENT"):
        project(rows, original=changed)
    with pytest.raises(hp.HousingProjectionError, match="SOURCE_PINS"):
        ws.project(
            rows,
            parent=p,
            expected_parent_receipt_sha256=p.receipt_sha256,
            source=dataclasses.replace(SOURCE, data_sha256="0" * 64),
            complete_membership=True,
        )


@pytest.mark.parametrize("tamper", ["mean", "parent", "extra", "rows"])
def test_codec_rejects_resigned_content_tampering(tamper):
    rows = [row()]
    p = parent(rows)
    result = project(rows, original=p)
    document = json.loads(result.artifact_json)
    if tamper == "mean":
        document["households"][0]["december_member_mean_weight"]["value"] = "999"
    if tamper == "parent":
        document["parent"]["measurement_digest"] = "0" * 64
    if tamper == "extra":
        document["hidden_fallback"] = True
    if tamper == "rows":
        document["rows"][0]["values"]["ERELRPE"] = "2"
    payload = canonical_json(document)
    with pytest.raises(ws.WeightSourceError, match="RECOMPUTATION"):
        ws.decode_artifact(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            parent=p,
            expected_parent_receipt_sha256=p.receipt_sha256,
            source=SOURCE,
        )


def test_recipe_and_parent_code_pins_fail_closed(monkeypatch):
    ws.verify_recipe()
    monkeypatch.setattr(ws, "RECIPE_SHA256", "0" * 64)
    with pytest.raises(ws.WeightSourceError, match="RECIPE_TAMPERED"):
        project([row()])


def test_changed_new_field_does_not_change_any_parent_receipt_byte():
    a = parent([row()])
    b = parent([row(ARELRP="10", EHRESIDENCID="200001", SHHADID="049", GVARSTR="240")])
    assert a.receipt_json == b.receipt_json
    assert (
        project([row()], original=a).artifact_json
        != project(
            [row(ARELRP="10", EHRESIDENCID="200001", SHHADID="049", GVARSTR="240")],
            original=b,
        ).artifact_json
    )


def test_new_rows_cannot_clear_a_frozen_parent_weight_unknown():
    clean = row()
    p = parent([m.SippRow(clean.values, ("caller_source_contradiction",))])
    mean, reference = candidates(project([clean], original=p))
    assert mean["value"] is None and reference["value"] is None


def test_parent_implementation_pin_is_enforced_without_modifying_frozen_file(
    monkeypatch,
):
    p = parent([row()])
    changed = dict(ws.FROZEN_PARENT_MODULE_SHA256)
    changed["microcosm.build.sipp_measurements"] = "0" * 64
    monkeypatch.setattr(hp, "module_digests", lambda: changed)
    with pytest.raises(ws.WeightSourceError, match="PARENT_CODE_TAMPERED"):
        project([row()], original=p)


def test_unknown_frame_on_nonreference_member_is_visible_not_mean_denominator_filter():
    result = project([row(), row(102, "300", RIN_UNIV="9")])
    mean, reference = candidates(result)
    assert mean["value"] == "200" and reference["value"] == "100"
    assert (
        result.households[0]["quality"]["all_members_confirmed_in_monthly_frame"]
        is False
    )


def test_decoder_refuses_noncanonical_bytes_and_wrong_expected_sha():
    rows = [row()]
    p = parent(rows)
    result = project(rows, original=p)
    for payload, digest, message in [
        (result.artifact_json, "0" * 64, "ARTIFACT_SHA"),
        (
            result.artifact_json + b"\n",
            hashlib.sha256(result.artifact_json + b"\n").hexdigest(),
            "ARTIFACT_FORMAT",
        ),
    ]:
        with pytest.raises(ws.WeightSourceError, match=message):
            ws.decode_artifact(
                payload,
                expected_sha256=digest,
                parent=p,
                expected_parent_receipt_sha256=p.receipt_sha256,
                source=SOURCE,
            )


def test_reported_rent_scope_matches_parent_with_allocated_tenure():
    rows = [row(ATENURE="2")]
    p = parent(rows)
    result = project(rows, original=p)
    assert p.households[0].exclusion is None
    assert p.diagnostics["rent_quality_admitted"] == {"reported": 1}
    assert result.diagnostics["reported_rent_donors"]["households"] == 1


def test_populated_relationship_outside_universe_blocks_unique_reference():
    result = project([row(), row(102, "300", THHLDSTATUS="4", ERELRPE="1")])
    mean, reference = candidates(result)
    assert mean["value"] == "200"
    assert reference["known_interview_reference_members"] == 1
    assert reference["unknown_reference_classifications"] == 1
    assert reference["value"] is None
    assert reference["status"] == "reference_classification_incomplete"
    assert (
        result.households[0]["members"][1]["fields"]["ERELRPE"]["state"]
        == "value_outside_universe"
    )


def test_two_residences_in_one_ssuid_do_not_share_reference_members():
    result = project(
        [
            row(),
            row(102, "300", ERESIDENCEID="100002", ERELRPE="2", EHRESIDENCID="100002"),
        ]
    )
    assert len(result.households) == 2
    assert [h["key"] for h in result.households] == [["0001", 100001], ["0001", 100002]]
    refs = [
        h["interview_reference_person_member_december_weight"]
        for h in result.households
    ]
    assert [r["value"] for r in refs] == ["100", "300"]
    assert [r["known_interview_reference_members"] for r in refs] == [1, 1]


def test_status_two_new_member_is_reference_eligible():
    _, reference = candidates(project([row(THHLDSTATUS="2", ERELRPE="2")]))
    assert reference["value"] == "100" and reference["status"] == "positive"


def test_parent_upper_endpoint_disagreement_is_visible():
    result = project([row(weight="10000000")])
    mean, _ = candidates(result)
    assert mean["value"] is None and mean["parent_mean_equal"] is False
    assert (
        result.diagnostics["all_households"]["mean_parent_disagreement_households"] == 1
    )


def test_csv_quotes_are_literal_as_in_frozen_parent_parser():
    fields = list(ws.FIELDS)
    values = row(ERELRPE='"1"').values
    raw = "|".join(fields) + "\n" + "|".join(values[f] for f in fields) + "\n"
    decoded = list(ws.iter_csv_rows(io.StringIO(raw)))
    assert decoded[0].values["ERELRPE"] == '"1"'
    complete = m.SippRow({**row().values, **decoded[0].values})
    _, reference = candidates(project([complete]))
    assert reference["status"] == "reference_classification_incomplete"
