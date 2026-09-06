"""Invented CSV cohorts only; no genuine source or population execution."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _sources(tmp_path, monkeypatch, *, reverse=True, relationship=(1, 5)):
    path = Path(__file__).with_name("test_us_asec_person_income_source.py")
    spec = importlib.util.spec_from_file_location("demographic_invented_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, attachment, members = module.invented_sources(tmp_path, monkeypatch)
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
