"""Invented archives and real issuers; no population files or country imports."""

import builtins
import copy
import json
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_acs_person_coverage_authentication import (
    Member,
    _csv,
    _household,
    _person,
    build_fixture,
)

from microcosm.build import acs_spm_partition as partition
from microcosm.build.acs_spm_source_assembly import (
    AcsSpmSourceAssemblyOptions,
    require_acs_spm_source_capability,
)
from microcosm.build.us_runtime import acs_native_coverage_binding as native
from microcosm.build.us_runtime import survey_population_preparation as preparation

OPTIONS = AcsSpmSourceAssemblyOptions(partition.ACS_SPM_DEVELOPMENT_POLICY, True)
requires_assembler = pytest.mark.xfail(
    not partition.probe_acs_spm_assembler().supported,
    raises=partition.UnsupportedAssembler,
    strict=True,
    reason="Explicit native construction requires the reviewed canonical assembler",
)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        native.housing.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=64 * 1024**3),
    )
    households = [
        _household("2024HU0000001", NP=3, TEN=3),
        _household("2024GQ0000001", NP=1, TYPEHUGQ=2, WGTP=0, TEN=""),
    ]
    people = [
        _person("2024HU0000001", 1, 20, AGEP=45, MIL="4", ESR="6"),
        _person("2024HU0000001", 2, 36, AGEP=16, MIL="", ESR=""),
        _person("2024HU0000001", 3, 35, AGEP=14, MIL="", ESR=""),
        _person("2024GQ0000001", 1, 37, AGEP=40, MIL="4", ESR="6", PWGTP=77),
    ]
    value = build_fixture(
        tmp_path,
        monkeypatch,
        household_members=(Member("psam_husa.csv", _csv(households)),),
        person_members=(Member("psam_pusa.csv", _csv(people)),),
    )
    return dict(
        source_dir=value.source_dir,
        snapshot_root=value.snapshot_root,
        serialnos=tuple(sorted(row["SERIALNO"] for row in households)),
    )


def request(path, options=OPTIONS):
    document = json.loads(path.read_bytes())
    document["protocol"] = preparation.SPM_REQUEST_PROTOCOL
    document["acs_spm_construction"] = asdict(options)
    path.write_bytes(preparation._encode(document, preparation.MAX_REQUEST_BYTES))
    return document


def rows(table):
    return pd.DataFrame(table["rows"], columns=table["columns"])


def test_default_native_route_never_imports_optional_assembler(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("spm_calculator")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = native.issue_acs_native_coverage(**arguments)
    assert result.construction_evidence is None
    assert "spm_construction" not in result.receipt


@requires_assembler
def test_real_native_owner_retains_actual_rows_and_conservation(tmp_path, monkeypatch):
    require_acs_spm_source_capability(OPTIONS)
    arguments = fixture(tmp_path, monkeypatch)
    original = native.issue_acs_native_coverage(**arguments)
    result = native.issue_acs_native_coverage(**arguments, spm_construction=OPTIONS)
    evidence = result.construction_evidence
    frame, before = result.frame, original.frame
    assert "acs_spm_source_assembly" not in frame.metadata
    assert frame.metadata == before.metadata
    assert result.receipt["spm_construction"]["metadata_transport"] == (
        "aggregate_and_rowwise_evidence_retained_by_owner_not_normative_frame_metadata"
    )
    assert evidence["implementation"]["options"] == asdict(OPTIONS)
    assert evidence["archives"] == result.receipt["archives"]
    assert (
        evidence["source_binding"] == "actual_construction_from_owner_captured_archives"
    )
    assert not evidence["engine_role_delivered"]
    assert not evidence["annual_universe_declared"]
    assert not evidence["release_eligible"]
    crosswalk = rows(evidence["native_crosswalk"])
    membership = rows(evidence["partition"]["membership"])
    assert len(crosswalk) == len(membership) == frame.n("person") == 4
    assert crosswalk.set_index("person_id").new_spm_unit_id.to_dict() == (
        frame.person.set_index("person_id").person_spm_unit_id.to_dict()
    )
    assert frame.n("spm_unit") == 3
    assert set(membership.role_source) >= {
        "observed_relationship_rule",
        "outside_acs_household_universe",
    }
    assert evidence["registry"]["entries"] == crosswalk.to_dict("records")
    for entity in before.entities:
        if entity == "spm_unit":
            continue
        columns = [c for c in before.table(entity) if c != "person_spm_unit_id"]
        pd.testing.assert_frame_equal(
            before.table(entity)[columns], frame.table(entity)[columns]
        )
    assert (
        before.weights_for("household").values.tobytes()
        == frame.weights_for("household").values.tobytes()
    )
    assert "is_spm_independent_minor_role" not in frame.person
    assert "spm_unit_spm_universe_status" not in frame.table("spm_unit")
    evidence["partition"]["membership"]["rows"].clear()
    assert len(result.construction_evidence["partition"]["membership"]["rows"]) == 4
    with pytest.raises(
        native.ACSNativeCoverageBindingError, match="ISSUANCE_NOT_OWNED"
    ):
        native.verify_acs_native_coverage(copy.copy(result))


@requires_assembler
@pytest.mark.parametrize("change", ["bytes", "missing", "frame"])
def test_native_mutated_evidence_or_frame_refuses(tmp_path, monkeypatch, change):
    require_acs_spm_source_capability(OPTIONS)
    result = native.issue_acs_native_coverage(
        **fixture(tmp_path, monkeypatch), spm_construction=OPTIONS
    )
    owned = native._owned(result)
    if change == "frame":
        owned.frame.person.loc[0, "person_spm_unit_id"] += 100
    else:
        value = (
            None
            if change == "missing"
            else owned.prepared.construction_evidence_json + b" "
        )
        object.__setattr__(owned.prepared, "construction_evidence_json", value)
    with pytest.raises(native.ACSNativeCoverageBindingError):
        native.verify_acs_native_coverage(result)


@pytest.mark.parametrize(
    "options", [{}, True, {"policy": "invented", "minor_partner_role": True}]
)
def test_bad_native_options_refuse_before_any_source_path(tmp_path, options):
    with pytest.raises(ValueError, match="OPTIONS_TYPE"):
        native.issue_acs_native_coverage(
            tmp_path / "absent",
            snapshot_root=tmp_path / "capture",
            spm_construction=options,
        )


def test_unsupported_native_capability_refuses_before_any_source_path(
    tmp_path, monkeypatch
):
    probe = partition.AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None)
    monkeypatch.setattr(partition, "probe_acs_spm_assembler", lambda: probe)
    with pytest.raises(partition.UnsupportedAssembler):
        native.issue_acs_native_coverage(
            tmp_path / "absent",
            snapshot_root=tmp_path / "capture",
            spm_construction=OPTIONS,
        )


@pytest.mark.parametrize(
    "change", ["missing", "extra", "integer_bool", "policy", "v1_extra", "null"]
)
def test_request_v2_is_closed_and_explicit(tmp_path, change):
    path = tmp_path / "selection-request.json"
    document = {
        "protocol": preparation.SPM_REQUEST_PROTOCOL,
        "declaration": preparation.domains.DECLARATION,
        "fraction": [1, 1],
        "seed": 41,
        "acs_spm_construction": asdict(OPTIONS),
    }
    if change == "missing":
        document.pop("acs_spm_construction")
    elif change == "extra":
        document["acs_spm_construction"]["links"] = []
    elif change == "integer_bool":
        document["acs_spm_construction"]["minor_partner_role"] = 1
    elif change == "policy":
        document["acs_spm_construction"]["policy"] = "guessed"
    elif change == "v1_extra":
        document["protocol"] = preparation.REQUEST_PROTOCOL
    else:
        document["acs_spm_construction"] = None
    path.write_bytes(preparation._encode(document))
    with pytest.raises(preparation.SurveyPopulationPreparationError):
        preparation.read_survey_population_request(tmp_path)


@requires_assembler
def test_v2_preparation_preserves_owned_native_evidence_and_replays(
    tmp_path, monkeypatch
):
    require_acs_spm_source_capability(OPTIONS)
    from test_us_survey_population_preparation import fixture as population_fixture

    arguments = population_fixture(tmp_path, monkeypatch)
    path = arguments["source_dir"] / "selection-request.json"
    default = preparation.prepare_authenticated_survey_population(**arguments)
    default_payload = default.to_bytes()
    assert default.acs_spm_construction_evidence is None
    document = request(path)
    assert preparation.read_survey_population_request(arguments["source_dir"]) == (
        Fraction(1),
        41,
    )
    result = preparation.prepare_authenticated_survey_population(**arguments)
    payload = result.to_bytes()
    assert payload != default_payload
    assert result.receipt["request"] == document
    evidence = result.acs_spm_construction_evidence
    native_ids = {r["new_spm_unit_id"] for r in evidence["registry"]["entries"]}
    origin = result.receipt["origins"]["entities"]["spm_unit"]
    assert {row[2] for row in origin if row[1] == "acs"} == native_ids
    replay = preparation.prepare_authenticated_survey_population(
        **arguments, candidate=payload
    )
    assert replay.to_bytes() == payload
    assert replay.acs_spm_construction_evidence == evidence
    with pytest.raises(
        preparation.SurveyPopulationPreparationError, match="CANDIDATE_MISMATCH"
    ):
        preparation.prepare_authenticated_survey_population(
            **arguments, candidate=default_payload
        )
    owned = native._owned(preparation._ISSUED[id(result)][2].native[0])
    object.__setattr__(owned.prepared, "construction_evidence_json", b"{}")
    with pytest.raises(preparation.SurveyPopulationPreparationError):
        result.checked_view()


@requires_assembler
def test_optional_native_source_mutation_still_refuses(tmp_path, monkeypatch):
    require_acs_spm_source_capability(OPTIONS)
    arguments = fixture(tmp_path, monkeypatch)
    result = native.issue_acs_native_coverage(**arguments, spm_construction=OPTIONS)
    path = arguments["source_dir"] / "csv_pus.zip"
    with path.open("ab") as stream:
        stream.write(b"changed invented archive")
    with pytest.raises(native.ACSNativeCoverageBindingError):
        native.verify_acs_native_coverage(result)


@requires_assembler
def test_v2_graph_cold_required_replay_and_option_key_separation(tmp_path, monkeypatch):
    require_acs_spm_source_capability(OPTIONS)
    from test_us_graph_survey_population import authenticated_arguments

    from microcosm.build.us_runtime import graph_survey_population as graph

    arguments = authenticated_arguments(tmp_path, monkeypatch)
    path = arguments["source_dir"] / "selection-request.json"
    request(path)
    cold_values = graph.run_authenticated_survey_population(
        **arguments, clones=False, return_values=True
    )
    cold = cold_values.manifest
    assert cold_values.preparation.acs_spm_construction_evidence["implementation"][
        "options"
    ] == asdict(OPTIONS)
    warm = graph.run_authenticated_survey_population(
        **arguments, clones=False, resume="require"
    )
    assert cold.key == warm.key
    assert all(node.store_hit for node in warm.nodes.values())
    graph._same_frame(
        cold.population(graph.CREATE_NODE), warm.population(graph.CREATE_NODE)
    )
    changed_options = AcsSpmSourceAssemblyOptions(OPTIONS.policy, False)
    request(path, changed_options)
    from microcosm.graph.errors import StoreMissError

    with pytest.raises(StoreMissError, match="resume='require' found cache misses"):
        graph.run_authenticated_survey_population(
            **arguments, clones=False, resume="require"
        )
    # A changed request cannot silently load the old required graph result.
    changed = graph.run_authenticated_survey_population(**arguments, clones=False)
    assert changed.key != cold.key


def test_unreviewed_compatible_assembler_refuses_before_source_path(
    tmp_path, monkeypatch
):
    probe = partition.AcsSpmAssemblerProbe(
        True, "supported", "/invented/units.py", "a" * 64
    )
    monkeypatch.setattr(partition, "probe_acs_spm_assembler", lambda: probe)
    with pytest.raises(
        native.housing.ACSHousingSourceError, match="SPM_ASSEMBLER_UNREVIEWED"
    ):
        native.issue_acs_native_coverage(
            tmp_path / "absent",
            snapshot_root=tmp_path / "capture",
            spm_construction=OPTIONS,
        )
