"""Invented original member bytes and a combined pool; no genuine data reads."""

import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.us_runtime import native_household_origin as origin
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import Frame, Weights


def _peer(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    acs_tests = _peer("test_us_acs_housing_source")
    fixture = acs_tests.build_fixture(tmp_path / "acs", monkeypatch)
    acs = origin.produce_acs_native_origins(
        fixture.source_dir,
        snapshot_root=fixture.snapshot_root,
        output_dir=fixture.output_dir(),
    )
    members = []
    paths = {}
    for year, key in [(2022, 47), (2023, 11), (2024, 900)]:
        payload = f"H_SEQ,UNUSED\n{key},ignored\n".encode()
        path = tmp_path / f"invented-hh-{year}.csv"
        path.write_bytes(payload)
        paths[year] = path
        members.append(
            origin.AsecNativeMemberPin(
                income_year=year,
                survey_year=year + 1,
                canonical_member_id=f"invented/asec/{year + 1}/household",
                member_name=f"hhpub{str(year + 1)[-2:]}.csv",
                archive_sha256=origin.ASEC_EDUCATION_ASSISTANCE_ARCHIVES[
                    year
                ].zip_sha256,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=1,
            )
        )
    monkeypatch.setattr(origin, "_ASEC_MEMBER_PINS", tuple(members))
    asec = origin.produce_asec_native_origins(paths)
    clone_tests = _peer("test_us_graph_combined_clone")
    a = clone_tests._asec_arm()
    c = clone_tests._acs_arm()
    a.table("household")["asec_H_SEQ"] = pd.Series([11, 47, 900], dtype="Int64")
    c.table("household")["SERIALNO"] = pd.Series(
        ["2024HU0000001", "2024HU0000002"], dtype="string"
    )
    pool = clone_tests._pool(a, c)
    return pool, (acs, asec), paths, fixture


def test_empty_registry_refuses_before_source_open(tmp_path, monkeypatch):
    monkeypatch.setattr(origin, "_ASEC_MEMBER_PINS", ())
    with pytest.raises(
        origin.NativeOriginError, match="ASEC_MEMBER_REGISTRY_UNAVAILABLE"
    ):
        origin.produce_asec_native_origins({2022: tmp_path / "absent"})


def test_a_well_shaped_source_object_is_not_authority():
    with pytest.raises(origin.NativeOriginError, match="SOURCE_CONSTRUCTOR"):
        origin.AuthenticatedNativeOriginSource(b"{}")


def test_native_keys_are_lexical_and_native_origins_are_not_generated_ids(inputs):
    frame, sources, _, _ = inputs
    bound = origin.bind_population_origins(frame, sources=sources)
    assert bound.summary["households"] == 5
    assert bound.summary["original_household_origins"] == 5
    assert bound.summary["zero_weight_households_with_lineage"] == 1
    assert len(set(row["origin_key"] for row in bound.document["households"])) == 5
    assert "2024HU0000001" in sources[0].payload.decode()
    assert "2024HU0000001" not in json.dumps(bound.summary)
    assert "asec_H_SEQ" not in json.dumps(bound.summary)


def test_source_byte_mutation_refuses(inputs):
    _, _, paths, _ = inputs
    path = paths[2022]
    path.write_bytes(path.read_bytes().replace(b"47", b"48"))
    with pytest.raises(origin.NativeOriginError, match="ASEC_MEMBER_SHA256"):
        origin.produce_asec_native_origins(paths)


def test_clone_descendants_share_origin_and_zero_weight_lineage(inputs):
    frame, sources, _, _ = inputs
    parent = origin.bind_population_origins(frame, sources=sources)
    expanded = clone_us_frame_for_puf_support(frame)
    child = origin.bind_population_origins(expanded, sources=sources, parent=parent)
    assert child.summary["households"] == 10
    assert child.summary["original_household_origins"] == 5
    assert child.summary["zero_weight_households_with_lineage"] == 2
    assert child.summary["parent_pair_mass_conserved"] is True
    rows = child.document["households"]
    for source_id in {r["entity_source_id"] for r in rows}:
        pair = [r for r in rows if r["entity_source_id"] == source_id]
        assert {r["clone_index"] for r in pair} == {0, 1}
        assert len({r["origin_key"] for r in pair}) == 1


def test_unknown_origin_even_on_zero_weight_row_refuses(inputs):
    frame, sources, _, _ = inputs
    table = frame.table("household")
    table.loc[table.asec_H_SEQ == "47", "asec_H_SEQ"] = "999"
    with pytest.raises(origin.NativeOriginError, match="NATIVE_KEY_UNRESOLVED"):
        origin.bind_population_origins(frame, sources=sources)


def test_source_grain_payload_cannot_claim_an_expanded_population(inputs):
    frame, sources, _, _ = inputs
    bound = origin.bind_population_origins(frame, sources=sources)
    expanded = clone_us_frame_for_puf_support(frame)
    with pytest.raises(origin.NativeOriginError, match="POPULATION_CONTENT"):
        origin.verify_population_origin_binding(expanded, bound)


def test_membership_change_invalidates_full_population_binding(inputs):
    frame, sources, _, _ = inputs
    bound = origin.bind_population_origins(frame, sources=sources)
    frame.person.loc[0, "person_household_id"] = frame.person.person_household_id.iloc[
        2
    ]
    with pytest.raises(origin.NativeOriginError, match="POPULATION_CONTENT"):
        origin.verify_population_origin_binding(frame, bound)


def test_source_capsule_cannot_be_replaced(inputs):
    _, sources, _, _ = inputs
    with pytest.raises(origin.NativeOriginError, match="SOURCE_CONSTRUCTOR"):
        replace(sources[0], payload=b"{}")


def test_reordered_population_rebinds_by_native_key(inputs):
    frame, sources, _, _ = inputs
    parent = origin.bind_population_origins(frame, sources=sources)
    reordered = Frame(
        {
            e: frame.table(e).iloc[::-1].reset_index(drop=True)
            if e == "person"
            else frame.table(e)
            for e in frame.entities
        },
        frame.schema,
        {
            e: Weights(
                frame.weights_for(e).values[::-1]
                if e == "person"
                else frame.weights_for(e).values,
                frame.weights_for(e).kind,
            )
            for e in frame.weighted_entities
        },
        frame.strata.iloc[::-1].reset_index(drop=True),
    )
    with pytest.raises(origin.NativeOriginError, match="POPULATION_CONTENT"):
        origin.verify_population_origin_binding(reordered, parent)
    child = origin.bind_population_origins(reordered, sources=sources, parent=parent)
    assert {
        r["household_id"]: r["origin_key"] for r in child.document["households"]
    } == {r["household_id"]: r["origin_key"] for r in parent.document["households"]}
    assert (
        child.document["ordered_entity_ids_sha256"]
        != parent.document["ordered_entity_ids_sha256"]
    )


def test_whole_household_selection_retains_ancestry(inputs):
    frame, sources, _, _ = inputs
    parent = origin.bind_population_origins(frame, sources=sources)
    selected = frame.select(
        frame.person.person_household_id != frame.person.person_household_id.iloc[0]
    )
    child = origin.bind_population_origins(selected, sources=sources, parent=parent)
    assert child.summary["households"] == 4
    assert (
        child.document["parent_binding_sha256"]
        == hashlib.sha256(parent.payload).hexdigest()
    )
    # Removing only a person changes a household; it is not household pruning.
    partial = frame.select(frame.person.person_id != frame.person.person_id.iloc[0])
    with pytest.raises(origin.NativeOriginError, match="PARENT_HOUSEHOLD_SELECTION"):
        origin.bind_population_origins(partial, sources=sources, parent=parent)


def test_changed_membership_inside_same_household_refuses(inputs):
    frame, sources, _, _ = inputs
    parent = origin.bind_population_origins(frame, sources=sources)
    expanded = clone_us_frame_for_puf_support(frame)
    # First ASEC household has two marital units. Switch one person's unit
    # while keeping their household, source identity and origin unchanged.
    person = expanded.person
    row = person.index[person.person_id == frame.person.person_id.iloc[0]][0]
    other = person[
        (person.person_household_id == person.loc[row, "person_household_id"])
        & (person.person_marital_unit_id != person.loc[row, "person_marital_unit_id"])
    ]
    person.loc[row, "person_marital_unit_id"] = other.person_marital_unit_id.iloc[0]
    with pytest.raises(
        origin.NativeOriginError,
        match="ENTITY_HOUSEHOLD_COVERAGE|PARENT_MEMBERSHIP_LINEAGE",
    ):
        origin.bind_population_origins(expanded, sources=sources, parent=parent)


def test_member_staging_aliases_preserve_native_identity(inputs, tmp_path):
    _, sources, paths, fixture = inputs
    aliases = {}
    for year, path in paths.items():
        alias = tmp_path / f"alias-{year}.bytes"
        shutil.copyfile(path, alias)
        aliases[year] = alias
    assert origin.produce_asec_native_origins(aliases).payload == sources[1].payload
    copied = tmp_path / "other-acs-staging"
    shutil.copytree(fixture.source_dir, copied)
    assert (
        origin.produce_acs_native_origins(
            copied, snapshot_root=fixture.snapshot_root, output_dir=fixture.output_dir()
        ).payload
        == sources[0].payload
    )


def test_same_native_integer_in_different_cohorts_is_distinct(inputs, monkeypatch):
    _, _, paths, _ = inputs
    changed = []
    for pin in origin._ASEC_MEMBER_PINS:
        body = b"H_SEQ,UNUSED\n47,ignored\n"
        paths[pin.income_year].write_bytes(body)
        changed.append(
            replace(
                pin,
                member_sha256=hashlib.sha256(body).hexdigest(),
                size_bytes=len(body),
            )
        )
    monkeypatch.setattr(origin, "_ASEC_MEMBER_PINS", tuple(changed))
    source = origin.produce_asec_native_origins(paths)
    assert len({row[-1] for row in source.document["records"]}) == 3


@pytest.mark.parametrize("bad", ["47.0", " 47", "0047", "", None])
def test_carried_native_integer_storage_is_lossless_only(inputs, bad):
    frame, sources, _, _ = inputs
    table = frame.table("household")
    table.loc[table.asec_H_SEQ == "47", "asec_H_SEQ"] = bad
    with pytest.raises(
        origin.NativeOriginError, match="CARRIED_NATIVE_INTEGER|NATIVE_INTEGER"
    ):
        origin.bind_population_origins(frame, sources=sources)


def test_non_household_weight_must_preserve_parent_pair(inputs):
    frame, sources, _, _ = inputs
    parent = origin.bind_population_origins(frame, sources=sources)
    expanded = clone_us_frame_for_puf_support(frame)
    altered = Frame(
        {e: expanded.table(e) for e in expanded.entities},
        expanded.schema,
        {
            e: Weights(
                expanded.resolve_weights(e).values * (2 if e == "tax_unit" else 1),
                expanded.resolve_weights(e).kind,
            )
            for e in expanded.entities
        },
        expanded.strata,
    )
    with pytest.raises(origin.NativeOriginError, match="CLONE_ENTITY_PAIR_WEIGHT"):
        origin.bind_population_origins(altered, sources=sources, parent=parent)


@pytest.mark.parametrize("position", [0, 106, 123])
def test_member_header_locates_native_key_without_fixed_offsets(
    inputs, monkeypatch, position
):
    _, _, paths, _ = inputs
    pin = origin._ASEC_MEMBER_PINS[0]
    header = [f"UNUSED_{i}" for i in range(position)] + ["H_SEQ"]
    row = ["ignored"] * position + ["47"]
    body = (",".join(header) + "\n" + ",".join(row) + "\n").encode()
    paths[pin.income_year].write_bytes(body)
    replacement = replace(
        pin, member_sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body)
    )
    monkeypatch.setattr(
        origin, "_ASEC_MEMBER_PINS", (replacement, *origin._ASEC_MEMBER_PINS[1:])
    )
    assert origin.produce_asec_native_origins(paths).document["records"][0][1] == 47


def test_duplicate_original_household_keys_refuse(inputs, monkeypatch):
    _, _, paths, _ = inputs
    pin = origin._ASEC_MEMBER_PINS[0]
    body = b"H_SEQ,UNUSED\n47,ignored\n47,ignored\n"
    paths[pin.income_year].write_bytes(body)
    replacement = replace(
        pin,
        member_sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        rows=2,
    )
    monkeypatch.setattr(
        origin, "_ASEC_MEMBER_PINS", (replacement, *origin._ASEC_MEMBER_PINS[1:])
    )
    with pytest.raises(origin.NativeOriginError, match="SOURCE_NATIVE_KEY_AMBIGUOUS"):
        origin.produce_asec_native_origins(paths)


@pytest.mark.parametrize("changed", [" 2024HU0000001", "2024hu0000001", "2024HU000001"])
def test_acs_native_identity_is_not_trimmed_normalized_or_numeric(inputs, changed):
    frame, sources, _, _ = inputs
    table = frame.table("household")
    table.loc[table.SERIALNO == "2024HU0000001", "SERIALNO"] = changed
    with pytest.raises(origin.NativeOriginError, match="NATIVE_KEY_UNRESOLVED"):
        origin.bind_population_origins(frame, sources=sources)
