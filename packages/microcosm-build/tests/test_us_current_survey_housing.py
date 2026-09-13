"""Invented housing sources and household donors; no native data or engine."""

import hashlib
import io
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_housing as housing
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def add_housing_source_fields(arguments, patch):
    """Add original interview observations before real fixture issuers run."""
    from microcosm.build.us_runtime import native_household_origin as origin

    path = arguments["source_dir"] / "asec" / "hhpub25.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows["H_TENURE"] = "2"
    rows["HPUBLIC"] = ["1" if i == 0 else "2" for i in range(len(rows))]
    rows["HLORENT"] = ["0" if i == 0 else "2" for i in range(len(rows))]
    rows["I_HPUBLI"] = "0"
    rows["I_HLOREN"] = "0"
    rows.to_csv(path, index=False)
    payload = path.read_bytes()
    patch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        tuple(
            replace(
                pin,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=len(rows),
            )
            for pin in origin._ASEC_MEMBER_PINS
        ),
    )
    return arguments


def _invented():
    hids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    ids = np.array([11, 12, 13, 21, 31, 41, 51, 61], dtype=np.int64)
    hh = np.array([10, 10, 10, 20, 30, 40, 50, 60], dtype=np.int64)
    spm = np.array([100, 100, 101, 200, 300, 400, 500, 600], dtype=np.int64)
    sources = ["asec"] * 6 + ["acs"] * 2
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": hh,
            "person_spm_unit_id": spm,
            "person_tax_unit_id": spm,
            "person_family_id": spm,
            "person_marital_unit_id": spm,
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": hids}),
        **{
            e: pd.DataFrame({e + "_id": np.unique(spm)})
            for e in US_SCHEMA.group_entities
            if e != "household"
        },
    }
    for entity, table in tables.items():
        entity_ids = table[entity + "_id"].to_numpy()
        table[housing.provenance.support_source_id_column(entity)] = entity_ids
        table[housing.provenance.spine_source_id_column(entity)] = entity_ids + 10000
        table[housing.provenance.support_clone_index_column(entity)] = np.zeros(
            len(table), dtype="int64"
        )
        if entity == "person":
            channel = sources
        elif entity == "household":
            channel = ["asec"] * 4 + ["acs"] * 2
        else:
            channel = ["asec"] * 5 + ["acs"] * 2
        table[housing.provenance.support_channel_column(entity)] = pd.array(
            channel, dtype="string"
        )
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.array([100.0, 1.0, 3.0, 2.0, 5.0, 4.0]), WeightKind.DESIGN
            )
        },
        pd.Series(["invented"] * len(person), dtype="string"),
    )
    origins = pd.DataFrame(
        {
            "source": ["asec"] * 4 + ["acs"] * 2,
            "source_year": [2024] * 6,
            "survey_year": [2025] * 4 + [2024] * 2,
            "raw_native_id": ["1", "2", "3", "4", "2024HU0000005", "2024GQ0000006"],
            "selected_receiving_household_id": hids + 10000,
        },
        index=pd.Index(hids, name="household_id"),
    )
    porigins = pd.DataFrame(index=pd.Index(ids, name="person_id"))
    selected = {
        k: {} for k in ("asec_person", "acs_person", "asec_household", "acs_household")
    }
    keys = {}
    for index, pid in enumerate(ids):
        source = sources[index]
        hid = int(hh[index])
        line = index + 1 if hid == 10 else 1
        if source == "asec":
            key = (hid // 10, str(pid).zfill(22), line)
            selected["asec_person"][key] = {
                "PERIDNUM": key[1],
                "PH_SEQ": str(key[0]),
                "A_LINENO": str(line),
                "A_EXPRRP": "1" if line == 1 else "5",
                "P_SEQ": str(line),
            }
        else:
            serial = origins.loc[hid, "raw_native_id"]
            key = (serial, "1", 1)
            selected["acs_person"][key] = {
                "SERIALNO": serial,
                "SPORDER": "1",
                "RELSHIPP": "20" if hid == 50 else "37",
            }
        keys[source, key] = pid
    for hid in hids[:4]:
        number = int(hid // 10)
        selected["asec_household"][number] = {
            "H_SEQ": str(number),
            "H_TENURE": "1" if hid == 40 else "2",
            "H_HHTYPE": "1",
            "HRHTYPE": "1",
            "H_LIVQRT": "1",
            "HPUBLIC": {10: "1", 20: "2", 30: "0", 40: "0"}[int(hid)],
            "HLORENT": "2" if hid == 20 else "0",
            "I_HPUBLI": "0",
            "I_HLOREN": "0",
        }
    for hid, kind, tenure in ((50, "1", "3"), (60, "2", "")):
        serial = origins.loc[hid, "raw_native_id"]
        selected["acs_household"][serial] = {
            "SERIALNO": serial,
            "TYPEHUGQ": kind,
            "NP": "1",
            "TEN": tenure,
        }
    pfeatures = pd.DataFrame(
        {
            housing.predictors.FEATURES[0]: [50, 10, 30, 40, 60, 35, 55, 70],
            housing.predictors.FEATURES[1]: [100, 0, 50, 200, 300, 1000, 0, 0],
            housing.predictors.FEATURES[2]: [0] * 8,
        },
        index=porigins.index,
        dtype="float64",
    )
    return frame, origins, porigins, keys, selected, pfeatures


def _qualified():
    frame, origins, porigins, keys, selected, features = _invented()
    native = housing._source_values(frame, origins, porigins, keys, selected)
    return housing._qualified_values(frame, origins, native, features, {})


def _cloned(frame):
    tables = {}
    for entity in frame.entities:
        parts = []
        for clone in (1, 0):
            part = frame.table(entity).copy(deep=True)
            part[entity + "_id"] += clone * 1000
            part[housing.provenance.support_clone_index_column(entity)] = clone
            if entity == "person":
                for group in US_SCHEMA.group_entities:
                    part["person_" + group + "_id"] += clone * 1000
            parts.append(part)
        tables[entity] = (
            pd.concat(parts, ignore_index=True).iloc[::-1].reset_index(drop=True)
        )
        if entity != "person":
            tables[entity] = (
                tables[entity].sort_values(entity + "_id").reset_index(drop=True)
            )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(tables["household"])), WeightKind.IMPORTANCE
            )
        },
        pd.Series(["invented"] * len(tables["person"]), dtype="string"),
    )


def test_known_unknown_and_explicit_model_exclusions_are_distinct():
    q = _qualified()
    assert q.native.housing_observed_receipt.tolist()[:2] == [True, False]
    assert q.native.housing_observed_receipt.iloc[2:].isna().all()
    assert q.native.housing_observed_receipt__known.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert q.native.loc[40, "housing_receipt__origin"] == "modeled_owner_exclusion"
    assert (
        q.native.loc[60, "housing_receipt__origin"]
        == "modeled_group_quarters_exclusion"
    )
    assert housing.model_input.decode_recipient_matrix(
        q.matrix
    ).features.index.tolist() == [30, 50]
    assert q.native.loc[60, "housing_source_head_person_id"] == -1


def test_household_donors_preserve_design_mass_and_aggregate_members_once():
    q = _qualified()
    assert q.donor_frame.table("household").household_id.tolist() == [10, 20]
    assert q.keep.tolist() == [True, True, True, True, False, False, False, False]
    assert q.donor_frame.weights_for("household").values.tolist() == [100, 1]
    assert q.donor_frame.weights_for("household").kind is WeightKind.DESIGN
    assert q.donor_columns[housing.TARGET].tolist() == [1.0, 0.0]
    assert q.donor_columns.loc[10, housing.FEATURES[1]] == 3
    assert q.donor_columns.loc[10, housing.FEATURES[2]] == 150
    assert q.evidence["donor_households"] == 2 and q.evidence["donor_persons"] == 4


def test_source_householder_role_never_uses_sequence_as_authority():
    frame, origins, porigins, keys, selected, _ = _invented()
    selected["asec_person"][(1, str(11).zfill(22), 1)]["P_SEQ"] = "2"
    selected["asec_person"][(1, str(12).zfill(22), 2)]["P_SEQ"] = "1"
    native = housing._source_values(frame, origins, porigins, keys, selected)
    assert native.loc[10, "housing_source_head_person_id"] == 11


@pytest.mark.parametrize(
    "defect",
    ("owner_positive", "unknown_tenure", "noninterview", "missing_head", "gq_head"),
)
def test_unresolved_universe_and_invalid_reference_roles_refuse(defect):
    frame, origins, porigins, keys, selected, _ = _invented()
    if defect == "owner_positive":
        selected["asec_household"][4]["HPUBLIC"] = "1"
    elif defect == "unknown_tenure":
        selected["asec_household"][3]["H_TENURE"] = "0"
    elif defect == "noninterview":
        selected["asec_household"][3]["H_HHTYPE"] = "2"
    elif defect == "missing_head":
        selected["asec_person"][(1, str(11).zfill(22), 1)]["A_EXPRRP"] = "5"
    else:
        selected["acs_person"][("2024GQ0000006", "1", 1)]["RELSHIPP"] = "20"
    with pytest.raises(ValueError, match="CURRENT_SURVEY_HOUSING_"):
        housing._source_values(frame, origins, porigins, keys, selected)


def test_clone_attachment_shares_draws_and_routes_only_householder_spm():
    q = _qualified()
    receiving = _cloned(q.source_frame)
    parent_stamp = housing.source._frame_identity(receiving)
    index = housing.model_input.decode_recipient_matrix(q.matrix).features.index
    draw = pd.DataFrame({housing.TARGET: [1.0, 0.0]}, index=index)
    result = housing.attach_columns(q, receiving, draw)
    for hid in (10, 1010, 30, 1030):
        assert result["household", "housing_receipt"].loc[hid]
    for sid in (100, 1100, 300, 1300):
        assert result["spm_unit", housing.SPM_OUTPUTS[0]].loc[sid]
    for sid in (101, 1101, 600, 1600):
        assert not result["spm_unit", housing.SPM_OUTPUTS[0]].loc[sid]
    assert pd.isna(result["household", "housing_observed_receipt"].loc[30])
    assert not result["household", "housing_observed_receipt__known"].loc[1030]
    assert housing.source._frame_identity(receiving) == parent_stamp
    assert not {"housing_assistance", "spm_unit_housing_subsidy"} & {
        c for _, c in result
    }
    declared = housing.output_columns(q, receiving)
    assert {(o.entity, o.column) for o in declared} == set(result)


@pytest.mark.parametrize(
    "defect", ("fractional", "axis", "clone", "native", "collision")
)
def test_attachment_refuses_invalid_draw_or_clone_identity(defect):
    q = _qualified()
    receiving = _cloned(q.source_frame)
    index = housing.model_input.decode_recipient_matrix(q.matrix).features.index
    draw = pd.DataFrame({housing.TARGET: [1.0, 0.0]}, index=index)
    if defect == "fractional":
        draw.iloc[0, 0] = 0.5
    elif defect == "axis":
        draw = draw.iloc[::-1]
    elif defect in ("clone", "native"):
        col = (
            housing.provenance.support_clone_index_column("household")
            if defect == "clone"
            else housing.provenance.spine_source_id_column("household")
        )
        receiving.table("household").loc[0, col] = 9
    else:
        receiving.table("spm_unit")[housing.SPM_OUTPUTS[0]] = False
    with pytest.raises(ValueError, match="CURRENT_SURVEY_HOUSING_"):
        housing.attach_columns(q, receiving, draw)


def test_no_recipient_requires_no_donor_model():
    frame, origins, porigins, keys, selected, features = _invented()
    selected["asec_household"][3].update(HPUBLIC="2", HLORENT="2")
    selected["acs_household"]["2024HU0000005"]["TEN"] = "1"
    native = housing._source_values(frame, origins, porigins, keys, selected)
    q = housing._qualified_values(frame, origins, native, features, {})
    assert q.matrix is None and q.donor_frame is None
    assert housing.attach_columns(q, _cloned(frame), None)


def test_qualifier_live_seal_catches_mapper_and_constant_changes(monkeypatch):
    before = housing.live()
    monkeypatch.setattr(housing.participation, "route_participation", lambda *_: None)
    assert housing.live() != before
    monkeypatch.undo()
    before = housing.live()
    monkeypatch.setattr(housing, "OWNER_ASSUMPTION", ("A5", "changed"))
    assert housing.live() != before


@pytest.mark.parametrize("mutation", ("mapper", "constant"))
def test_owner_callback_cannot_change_live_mapper_or_constants(monkeypatch, mutation):
    class ChangedBorrower:
        def _checked(self):
            if mutation == "mapper":
                monkeypatch.setattr(
                    housing.participation, "observed_participation", lambda *_: None
                )
            else:
                monkeypatch.setattr(housing, "OWNER_ASSUMPTION", ("A5", "changed"))
            # No source state is supplied: the entry fence must reject the
            # changed callback before consuming any alleged owner result.
            return None

    with pytest.raises(ValueError, match="CODE_CHANGED_DURING_OWNER"):
        housing._original_columns(ChangedBorrower())


def test_attachment_checks_live_configuration_after_routing_callback(monkeypatch):
    q = _qualified()
    receiving = _cloned(q.source_frame)
    index = housing.model_input.decode_recipient_matrix(q.matrix).features.index
    draw = pd.DataFrame({housing.TARGET: [1.0, 0.0]}, index=index)
    route = housing.participation.route_participation

    def changed_route(*args):
        result = route(*args)
        monkeypatch.setattr(housing, "OWNER_ASSUMPTION", ("A5", "changed"))
        return result

    monkeypatch.setattr(housing.participation, "route_participation", changed_route)
    with pytest.raises(ValueError, match="ATTACH_CODE_OR_VALUES_CHANGED"):
        housing.attach_columns(q, receiving, draw)


def test_partial_observed_head_column_remains_unchanged():
    q = _qualified()
    receiving = _cloned(q.source_frame)
    receiving.person["is_household_head"] = pd.array(
        [pd.NA] * len(receiving.person), dtype="boolean"
    )
    gq = receiving.person.person_household_id.isin((60, 1060))
    receiving.person.loc[gq, "is_household_head"] = False
    before = receiving.person.is_household_head.copy()
    index = housing.model_input.decode_recipient_matrix(q.matrix).features.index
    housing.attach_columns(
        q, receiving, pd.DataFrame({housing.TARGET: [1.0, 0.0]}, index=index)
    )
    pd.testing.assert_series_equal(receiving.person.is_household_head, before)


def test_bounded_literal_scan_refuses_duplicate_and_exhausts_unselected_rows():
    payload = b"H_SEQ,HPUBLIC\n1,1\n2,0\n"
    rows, count = housing._scan(
        io.BytesIO(payload),
        ("H_SEQ", "HPUBLIC"),
        key=housing._hh_key,
        wanted={1},
        maximum=2,
    )
    assert count == 2 and rows[1]["HPUBLIC"] == "1"
    with pytest.raises(ValueError, match="LITERAL_DUPLICATE"):
        housing._scan(
            io.BytesIO(payload + b"01,2\n"),
            ("H_SEQ", "HPUBLIC"),
            key=housing._hh_key,
            wanted={1},
            maximum=3,
        )


def test_actual_original_member_qualifier_has_spm_membership_and_named_heads(
    tmp_path, monkeypatch
):
    from test_us_current_asec_demographics import _demographic_arguments

    arguments = add_housing_source_fields(
        _demographic_arguments(tmp_path, monkeypatch, zero=False), monkeypatch
    )
    prepared = housing.source.prepare_authenticated_survey_population(**arguments)
    frame, origins, porigins, keys, selected, evidence = housing._original_columns(
        prepared
    )
    assert "spm_unit" in frame.entities and "person_spm_unit_id" in frame.person
    assert "is_household_head" in frame.person
    assert (
        frame.person.loc[
            frame.person.person_support_channel.eq("asec"), "is_household_head"
        ]
        .isna()
        .all()
    )
    native = housing._source_values(frame, origins, porigins, keys, selected)
    asec = origins.source.eq("asec")
    assert (
        native.loc[asec, "housing_observed_receipt"].tolist()
        == native.loc[asec, "housing_source_HPUBLIC"].eq("1").tolist()
    )
    assert sorted(native.loc[asec, "housing_observed_receipt"].tolist()) == [
        False,
        True,
    ]
    assert native.loc[asec, "housing_observed_receipt__known"].all()
    assert (
        native.loc[
            native.housing_participation_universe.eq("occupied_housing_unit"),
            "housing_source_head_person_id",
        ]
        .ge(0)
        .all()
    )
    assert evidence["asec_household_member_sha256"]
    prepared.checked_view()
