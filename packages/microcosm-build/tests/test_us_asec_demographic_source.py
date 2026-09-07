"""Invented CSV cohorts only; no genuine source or population execution."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_demographic_source as demographic


def _classify(
    *,
    sex=(1, 2),
    flag=(0, 4),
    relationship=(1, 5),
    sequence=(1, 2),
    households=None,
    years=None,
):
    rows = len(sex)
    return demographic.classify_asec_demographic_observations(
        income_year=np.array(
            years if years is not None else [2022] * rows, dtype=np.int64
        ),
        household_id=np.array(
            households if households is not None else [1] * rows, dtype=np.int64
        ),
        a_sex=np.array(sex, dtype=np.int64),
        axsex=np.array(flag, dtype=np.int64),
        a_exprrp=np.array(relationship, dtype=np.int64),
        p_seq=np.array(sequence, dtype=np.int64),
    )


def _member(tmp_path, **changes):
    values = {
        "PERIDNUM": ["0" * 21 + "1", "0" * 21 + "2"],
        "PH_SEQ": [7, 7],
        "A_LINENO": [1, 2],
        "A_AGE": [55, 14],
        "A_SEX": [1, 2],
        "AXSEX": [0, 4],
        "A_EXPRRP": [1, 5],
        "P_SEQ": [1, 2],
        # Neither family membership nor line order may supply headship.
        "A_FAMREL": [1, 1],
    }
    values.update(changes)
    path = tmp_path / "invented.csv"
    pd.DataFrame(values).to_csv(path, index=False)
    return path


def _read(path):
    return demographic.read_demographic_member(path, rows=2, size=path.stat().st_size)


def _repoint_published_households(parent, attachment, monkeypatch, native_ids):
    """Rewrite the parent roster's published household id, person by person.

    The invented parent publishes native id 7 in every cohort, so this is how
    a fixture gives one cohort two published households without touching the
    grouping column, whose distinct values the Frame requires to be exactly
    the household table's ids. ``source_household_id`` is pooled source
    provenance and so is inside the checkpoint's own structural identity: the
    identity is rebound to the rewritten Frame rather than defeated.
    """
    parent_identity = None
    for path in (parent, attachment):
        loaded = load_frame_checkpoint(path)
        loaded.frame.person["source_household_id"] = np.array(
            native_ids, dtype=np.int64
        )
        identity = frame_identity(loaded.frame)
        for key in ("identity", "source_construction_identity"):
            if key in loaded.metadata:
                loaded.metadata[key] = identity.to_payload()
        if path == attachment:
            digest = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = digest
            receipt = loaded.metadata["household_observations"]
            receipt["input_checkpoint_sha256"] = digest
            receipt["input_structural_identity_sha256"] = parent_identity.sha256
        else:
            parent_identity = identity
        write_frame_checkpoint(path, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        legacy,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(attachment.read_bytes()),
            legacy._SOURCE_PINS[2],
        ),
    )


def _sources(
    tmp_path, monkeypatch, *, reverse=True, relationship=(1, 5), native_ids=None
):
    path = Path(__file__).with_name("test_us_asec_person_income_source.py")
    spec = importlib.util.spec_from_file_location("demographic_invented_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, attachment, members = module.invented_sources(tmp_path, monkeypatch)
    if native_ids is not None:
        _repoint_published_households(parent, attachment, monkeypatch, native_ids)
    source = legacy.load_authenticated_current_money_source(parent, attachment)
    pins = []
    for year, member, archive, _, _rows, _ in demographic._MEMBER_PINS:
        table = pd.read_csv(members[year], dtype=str, keep_default_na=False)
        for i, row in table.iterrows():
            position = (int(row.PERIDNUM) - 1) % 2
            for name, values in {
                "A_SEX": [1, 2],
                "AXSEX": [0, 4],
                "A_EXPRRP": relationship,
                "P_SEQ": [1, 2],
            }.items():
                table.loc[i, name] = str(values[position])
            if native_ids is not None:
                table.loc[i, "PH_SEQ"] = str(native_ids[int(row.PERIDNUM) - 1])
        table = table.sort_values("PERIDNUM", ascending=not reverse)
        table.to_csv(members[year], index=False)
        pins.append(
            (
                year,
                member,
                archive,
                demographic._sha(members[year].read_bytes()),
                2,
                members[year].stat().st_size,
            )
        )
    monkeypatch.setattr(demographic, "_MEMBER_PINS", tuple(pins))
    return source, members


def test_exact_readset_and_declared_production_fields(tmp_path):
    table = _read(_member(tmp_path))
    assert set(table) == set(demographic.ASEC_DEMOGRAPHIC_SOURCE_COLUMNS)
    assert "A_FAMREL" not in table
    contract = demographic.contract_document()
    declared = contract["production_fields"]
    assert {
        v["column"] for v in declared["restored_observations"] + declared["derived"]
    } == set(demographic.ALIASES + demographic.DERIVED)
    assert contract["cohorts"]["2022"]["survey_year"] == 2023
    assert contract["cohorts"]["2024"]["survey_year"] == 2025
    assert not contract["wired_into_prepared_source"]
    assert not contract["other_than_two_is_male"]


@pytest.mark.parametrize("field", demographic.OBSERVATIONS)
@pytest.mark.parametrize("token", ["", "NA", "1.0", "-1", "+1", " 1", "1 ", "１"])
def test_missing_or_noncanonical_source_tokens_refuse_without_coercion(
    tmp_path, field, token
):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBER_TOKEN_WIDTH"
    ):
        _read(_member(tmp_path, **{field: [token, "1"]}))


@pytest.mark.parametrize("sex", [0, 3, 9])
def test_unknown_sex_never_becomes_male(sex):
    result = _classify(sex=(sex, 2))
    assert result.states["asec_sex_binding_state"].tolist() == [0, 2]
    assert result.states["asec_sex_unbound_reason"].tolist() == [1, 0]


@pytest.mark.parametrize("flag", [1, 2, 3, 5, 9])
def test_unnamed_or_invalid_allocation_flag_leaves_sex_unbound(flag):
    result = _classify(flag=(flag, 4))
    assert result.states["asec_sex_binding_state"].tolist() == [0, 2]
    assert result.states["asec_sex_allocation_state"].tolist() == [0, 2]


def test_valid_sex_retains_allocation_provenance():
    result = _classify()
    assert result.states["asec_sex_binding_state"].tolist() == [1, 2]
    assert result.states["asec_sex_allocation_state"].tolist() == [1, 2]


def test_domain_diagnostics_distinguish_unnamed_and_out_of_range_codes():
    result = _classify(sex=(0, 2), flag=(1, 5), relationship=(1, 6), sequence=(1, 99))
    domains = result.diagnostics["by_cohort"]["2022"]["field_domains"]
    assert domains["A_SEX"]["rows_outside_declared_range"] == 1
    assert domains["AXSEX"]["rows_in_unnamed_range_codes"] == 1
    assert domains["AXSEX"]["rows_outside_declared_range"] == 1
    assert domains["A_EXPRRP"]["rows_in_unnamed_range_codes"] == 1
    assert domains["P_SEQ"]["rows_outside_declared_range"] == 1


@pytest.mark.parametrize(
    "field,token",
    [("A_SEX", "01"), ("AXSEX", "04"), ("A_EXPRRP", "001"), ("P_SEQ", "001")],
)
def test_source_tokens_cannot_exceed_printed_width(tmp_path, field, token):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBER_TOKEN_WIDTH"
    ):
        _read(_member(tmp_path, **{field: [token, "1"]}))


@pytest.mark.parametrize(
    "years,households", [([2021, 2021], [1, 1]), ([2022, 2022], [0, 0])]
)
def test_unknown_cohort_or_nonpositive_household_refuses(years, households):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="CLASSIFY_COORDINATES"
    ):
        _classify(years=years, households=households)


@pytest.mark.parametrize(
    "relationship,reason",
    [((5, 13), [2, 2]), ((1, 2), [3, 3]), ((1, 6), [4, 1]), ((1, 0), [4, 1])],
)
def test_missing_duplicate_and_unknown_reference_do_not_use_sequence_fallback(
    relationship, reason
):
    result = _classify(relationship=relationship)
    assert result.states["asec_household_reference_state"].tolist() == [0, 0]
    assert result.states["asec_household_reference_unbound_reason"].tolist() == reason
    assert (
        result.diagnostics["p_seq_crosscheck"]["unbound_households_with_one_p_seq_one"]
        == 1
    )


def test_sequence_disagreement_does_not_override_explicit_reference():
    result = _classify(relationship=(13, 2))
    assert result.states["asec_household_reference_state"].tolist() == [2, 1]
    crosscheck = result.diagnostics["p_seq_crosscheck"]
    assert crosscheck["p_seq_one_without_reference_code"] == 1
    assert crosscheck["reference_code_without_p_seq_one"] == 1
    assert not crosscheck["used_as_fallback"]


def test_sequence_crosscheck_compares_source_codes_even_when_headship_is_unbound():
    crosscheck = _classify(relationship=(1, 2)).diagnostics["p_seq_crosscheck"]
    assert crosscheck["agreement_rows"] == 1
    assert crosscheck["p_seq_one_without_reference_code"] == 0
    assert crosscheck["reference_code_without_p_seq_one"] == 1


def test_subfamily_role_and_row_order_do_not_supply_household_headship():
    inputs = dict(
        sex=(1, 2, 1, 2),
        flag=(0, 4, 0, 4),
        relationship=(1, 5, 10, 7),
        sequence=(3, 1, 2, 4),
    )
    result = _classify(**inputs)
    assert result.states["asec_household_reference_state"].tolist() == [1, 2, 2, 2]
    reordered = _classify(**{k: tuple(reversed(v)) for k, v in inputs.items()})
    for name, values in result.states.items():
        np.testing.assert_array_equal(values[::-1], reordered.states[name])
    assert result.diagnostics == reordered.diagnostics
    assert not result.diagnostics["family_relationship_field_read"]


@pytest.mark.parametrize(
    "codes,expected",
    [
        ([20, 21, 37, 38], [1, 2, 3, 3]),
        ([20, 37, 37, 38], [0, 0, 3, 3]),
        ([20, 99, 37, 38], [0, 0, 3, 3]),
        ([21, 22, 37, 38], [0, 0, 3, 3]),
        ([20, 20, 37, 38], [0, 0, 3, 3]),
    ],
)
def test_acs_housing_units_and_both_group_quarters_codes_stay_distinct(codes, expected):
    relshipp = np.array(codes, dtype=np.int64)
    before = relshipp.copy()
    states, diagnostics = demographic.acs_household_reference_states(
        household_id=np.array([1, 1, 2, 3], dtype=np.int64), relshipp=relshipp
    )
    assert states.tolist() == expected
    np.testing.assert_array_equal(relshipp, before)
    assert diagnostics["group_quarters_only_households"] == 2
    assert not diagnostics["manufactured_group_quarters_reference_person"]


@pytest.mark.parametrize("reverse", [False, True])
def test_three_cohort_restoration_joins_native_keys_and_preserves_parent(
    tmp_path, monkeypatch, reverse
):
    source, members = _sources(tmp_path, monkeypatch, reverse=reverse)
    before = legacy._frame_signature(source.frame)
    observations = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members
    )
    assert observations.array("asec_A_SEX").tolist() == [1, 2] * 3
    assert observations.array("asec_AXSEX").tolist() == [0, 4] * 3
    assert observations.array("asec_A_EXPRRP").tolist() == [1, 5] * 3
    assert [r["joined_rows"] for r in observations.receipt["sources"]] == [2] * 3
    assert set(observations.receipt["diagnostics"]["by_cohort"]) == {
        "2022",
        "2023",
        "2024",
    }
    attached = demographic.attach_asec_demographic_source(source, observations)
    attached.validate()
    for entity in source.frame.entities:
        actual = attached.frame.table(entity)
        if entity == "person":
            actual = actual.drop(
                columns=list(demographic.ALIASES + demographic.DERIVED)
            )
        pd.testing.assert_frame_equal(
            actual, source.frame.table(entity), check_exact=True
        )
    assert legacy._frame_signature(source.frame) == before
    assert not observations.array("asec_A_SEX").flags.writeable
    assert "is_female" not in attached.frame.person
    assert "is_household_head" not in attached.frame.person


def test_restored_reference_can_disagree_with_unverified_incumbent_recode(
    tmp_path, monkeypatch
):
    source, members = _sources(tmp_path, monkeypatch, relationship=(13, 2))
    observations = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members
    )
    assert observations.array("asec_household_reference_state").tolist() == [2, 1] * 3
    assert [
        r["incumbent_relationship_crosscheck"]["mismatch_rows"]
        for r in observations.receipt["sources"]
    ] == [2] * 3


def test_candidate_bundle_is_reconstructed_and_mutation_is_refused(
    tmp_path, monkeypatch
):
    source, members = _sources(tmp_path, monkeypatch)
    destination = tmp_path / "demographic-bundle"
    receipt = demographic.write_asec_demographic_source(
        source, member_paths=members, output_dir=destination
    )
    candidate = destination / demographic.FILENAME
    observed = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members, candidate_path=candidate
    )
    assert observed.content_sha256 == receipt["content_sha256"]
    payload = bytearray(candidate.read_bytes())
    payload[-1] ^= 1
    candidate.write_bytes(payload)
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="DEMOGRAPHIC_CANONICAL_BYTES"
    ):
        demographic.load_authenticated_asec_demographic_source(
            source, member_paths=members, candidate_path=candidate
        )


def _verify(years, natives, groups):
    return demographic.verify_household_membership(
        income_year=np.array(years, dtype=np.int64),
        source_household_id=np.array(natives, dtype=np.int64),
        household_id=np.array(groups, dtype=np.int64),
    )


#: Three cohorts, each publishing one household, all under the same native id.
_REUSED_YEARS = [2022, 2022, 2023, 2023, 2024, 2024]
_REUSED_NATIVE_IDS = [7, 7, 7, 7, 7, 7]
_REUSED_GROUPS = [1, 1, 2, 2, 3, 3]


def test_native_household_id_reuse_across_cohorts_is_admitted_and_counted():
    result = _verify(_REUSED_YEARS, _REUSED_NATIVE_IDS, _REUSED_GROUPS)
    assert result["bijective"]
    assert result["grouping_households"] == result["published_households"] == 3
    assert result["membership_edges"] == 3
    assert result["native_household_ids_reused_across_cohorts"] == 1
    assert result["native_household_id_reuse_across_cohorts_is_admitted"]
    assert result["published_household_key"] == ["income_year", "source_household_id"]
    assert result["by_cohort"]["2023"] == {
        "income_year": 2023,
        "survey_year": 2024,
        "rows": 2,
        "grouping_households": 1,
        "published_households": 1,
        "native_household_ids": 1,
    }
    assert result["counters_zero_in_every_successful_receipt"] == [
        "grouping_households_spanning_multiple_cohorts",
        "grouping_households_spanning_multiple_published_households",
        "published_households_spanning_multiple_grouping_households",
    ]
    assert all(result[name] == 0 for name in demographic.MEMBERSHIP_REFUSAL_COUNTERS)


def test_membership_verification_does_not_depend_on_row_order():
    forward = _verify(_REUSED_YEARS, _REUSED_NATIVE_IDS, _REUSED_GROUPS)
    order = [5, 0, 3, 2, 4, 1]
    shuffled = _verify(
        [_REUSED_YEARS[i] for i in order],
        [_REUSED_NATIVE_IDS[i] for i in order],
        [_REUSED_GROUPS[i] for i in order],
    )
    assert forward == shuffled


@pytest.mark.parametrize(
    "years,natives,groups,reason",
    [
        # Two published households of one cohort collapsed into one grouping
        # household: its single reference person would otherwise be read as
        # the head of both, and the other published household would silently
        # lose its own.
        ([2022, 2022], [7, 8], [1, 1], "MEMBERSHIP_MERGED_NATIVE_HOUSEHOLDS"),
        (
            [2022, 2022, 2022],
            [7, 7, 8],
            [1, 1, 1],
            "MEMBERSHIP_MERGED_NATIVE_HOUSEHOLDS",
        ),
        # One published household split across two grouping households: each
        # part is classified on a subset of the roster the reference-person
        # cardinality is defined over.
        ([2022, 2022], [7, 7], [1, 2], "MEMBERSHIP_SPLIT_NATIVE_HOUSEHOLDS"),
        ([2022] * 3, [7, 7, 7], [1, 1, 2], "MEMBERSHIP_SPLIT_NATIVE_HOUSEHOLDS"),
        # A grouping household spanning cohorts, with the native id reused and
        # with two different native ids.
        ([2022, 2023], [7, 7], [1, 1], "MEMBERSHIP_COHORT_COLLISION"),
        ([2022, 2023], [7, 8], [1, 1], "MEMBERSHIP_COHORT_COLLISION"),
    ],
)
def test_unequal_household_partitions_refuse_without_naming_a_household(
    years, natives, groups, reason
):
    with pytest.raises(demographic.DemographicSourceRefusalError) as error:
        _verify(years, natives, groups)
    assert str(error.value) == f"person_household_id: {reason}"


@pytest.mark.parametrize(
    "years,natives,groups,message",
    [
        ([2021, 2021], [7, 7], [1, 1], "income_year: MEMBERSHIP_COHORT"),
        ([2022, 2022], [0, 7], [1, 1], "source_household_id: MEMBERSHIP_COORDINATE"),
        (
            [2022, 2022],
            [7, 7],
            [0, 1],
            "person_household_id: MEMBERSHIP_COORDINATE",
        ),
    ],
)
def test_membership_coordinates_must_be_declared_and_positive(
    years, natives, groups, message
):
    with pytest.raises(demographic.DemographicSourceRefusalError) as error:
        _verify(years, natives, groups)
    assert str(error.value) == message


@pytest.mark.parametrize("name", ["income_year", "source_household_id", "household_id"])
def test_membership_refuses_non_integer_coordinate_arrays(name):
    arrays = {
        "income_year": np.array([2022, 2022], dtype=np.int64),
        "source_household_id": np.array([7, 7], dtype=np.int64),
        "household_id": np.array([1, 1], dtype=np.int64),
    }
    arrays[name] = arrays[name].astype(np.float64)
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBERSHIP_INPUT"
    ):
        demographic.verify_household_membership(**arrays)


def test_classifier_refuses_a_grouping_household_spanning_two_cohorts():
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="CLASSIFY_HOUSEHOLD_COHORT"
    ):
        _classify(years=[2022, 2023], households=[1, 1])


@pytest.mark.parametrize("field", ["PH_SEQ", "A_LINENO", "A_AGE"])
@pytest.mark.parametrize("token", ["", " ", "  1", "1.0", "-1", "+1", "1 ", "NA"])
def test_blank_or_noncanonical_coordinate_tokens_refuse(tmp_path, field, token):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBER_INTEGER"
    ):
        _read(_member(tmp_path, **{field: [token, "1"]}))


def test_blank_person_key_token_refuses(tmp_path):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBER_PERSON_KEY"
    ):
        _read(_member(tmp_path, PERIDNUM=["", "0" * 21 + "2"]))


def test_nonpositive_published_household_coordinate_refuses(tmp_path):
    with pytest.raises(
        demographic.DemographicSourceRefusalError, match="MEMBER_COORDINATES"
    ):
        _read(_member(tmp_path, PH_SEQ=["0", "7"]))


def test_restoration_records_the_verified_household_membership(tmp_path, monkeypatch):
    source, members = _sources(tmp_path, monkeypatch)
    observations = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members
    )
    membership = observations.receipt["household_membership"]
    assert observations.receipt["household_grouping_column"] == "person_household_id"
    assert membership["grouping_column"] == "person_household_id"
    assert membership["equivalence"] == "bidirectional_partition_bijection"
    assert membership["verified_before_classification"]
    assert membership["bijective"]
    assert membership["rows"] == 6
    assert membership["grouping_households"] == membership["published_households"] == 3
    assert membership["membership_edges"] == 3
    # The parent publishes the same native household id in all three cohorts.
    assert membership["native_household_ids_reused_across_cohorts"] == 1
    assert set(membership["by_cohort"]) == {"2022", "2023", "2024"}
    assert all(
        membership[name] == 0 for name in demographic.MEMBERSHIP_REFUSAL_COUNTERS
    )
    assert (
        demographic.contract_document()["household_membership_verification"][
            "native_household_id_reuse_across_cohorts"
        ]
        == "admitted_and_counted"
    )
    assert observations.array("asec_household_reference_state").tolist() == [1, 2] * 3


def test_merged_published_households_refuse_before_the_demographic_reader(
    tmp_path, monkeypatch
):
    """Pin the layering the demographic fence restates.

    The 2022 cohort now publishes households 7 and 8 while the parent roster
    still groups both persons together. The parent's own authentication
    refuses that roster first: ``asec_household_observations._household_keys``
    requires the same equivalence, and it is reached from ``_scope``, so from
    every ``AuthenticatedCurrentMoneySource.validate()`` the demographic
    reader performs. The demographic module refuses the same shape on its own
    evidence, so relaxing the parent fence would not silently admit a merged
    household into a headship verdict.
    """
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_CONTRACT_REFUSAL"):
        _sources(tmp_path, monkeypatch, native_ids=[7, 8, 7, 7, 7, 7])
    with pytest.raises(demographic.DemographicSourceRefusalError) as error:
        _verify(_REUSED_YEARS, [7, 8, 7, 7, 7, 7], _REUSED_GROUPS)
    assert str(error.value) == (
        "person_household_id: MEMBERSHIP_MERGED_NATIVE_HOUSEHOLDS"
    )


def test_restoration_receipt_names_its_structurally_zero_counters(
    tmp_path, monkeypatch
):
    source, members = _sources(tmp_path, monkeypatch)
    observations = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members
    )
    for entry in observations.receipt["sources"]:
        assert entry["counters_zero_in_every_successful_receipt"] == [
            "accepted_source_missing_tokens",
            "incumbent_conflicts",
            "native_key_or_age_conflicts",
            "unreferenced_source_rows",
        ]
        assert entry["zero_counter_semantics"] == demographic.ZERO_COUNTER_SEMANTICS
        assert entry["unreferenced_source_rows"] == 0
        assert entry["incumbent_conflicts"] == 0
        assert entry["native_key_or_age_conflicts"] == 0
