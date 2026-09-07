"""Invented current-population and authentic source-codec boundary witnesses."""

import copy
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import acs_hu_source_transport as transport
from microcosm.build.us_runtime import cd_current_source_evidence as current
from microcosm.build.us_runtime import enriched_origin_calibration as full
from microcosm.build.us_runtime import graph_cd_source_evidence as graph_cd
from microcosm.build.us_runtime import graph_composed_population as composed
from microcosm.build.us_runtime import graph_native_household_origin as origins
from microcosm.build.us_runtime.graph_housing_universe import bind_housing_universe
from microcosm.frame import Frame, Weights
from microcosm.graph import ArtifactType
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import opaque_artifact_key


def frame_with(frame, *, tables=None, weights=None, metadata=None):
    return Frame(
        {e: frame.table(e) for e in frame.entities} if tables is None else tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities}
        if weights is None
        else weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata if metadata is None else metadata,
    )


def test_actual_small_graph_and_source_boundaries(tmp_path, monkeypatch):
    """CI entry point; standalone proof additionally supplies execution guards."""
    checkout = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(checkout / "tools"))
    from cd_source_evidence_checks import run

    root = tmp_path.resolve() / "invented-cd-ci"
    root.mkdir()
    cold = run("cold", root, checkout)
    assert cold["nodes"] == 20 and cold["hits"] == 17 and len(cold["calls"]) == 3
    checked = run("checks", root, checkout)
    assert checked["checks_count"] >= 70 and not checked["calls"]


def exercise(root, checkout, manifest, store, graph, observed, result, keys):
    from cd_source_evidence_checks import verified
    from survey_lineage_prep.invented import peer
    from us_survey_lineage_graph import (
        CLONE_BINDING,
        CLONE_POPULATION,
        NATIVE_BINDING,
        NATIVE_POPULATION,
    )

    records = []

    def save_progress():
        (root / "completed-witnesses.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )

    def passes(name, action):
        value = action()
        records.append({"name": name, "status": "PASS"})
        save_progress()
        return value

    def refuses(name, action, code=None):
        try:
            action()
        except (ValueError, TypeError, KeyError) as error:
            if code is not None:
                assert code in str(error), (name, type(error).__name__, str(error))
            records.append(
                {
                    "name": name,
                    "status": "REFUSED",
                    "code": code or type(error).__name__,
                }
            )
            save_progress()
        else:
            raise AssertionError(name + " unexpectedly accepted")

    population = observed[CLONE_POPULATION]
    clone = origins.verify_materialized_population_origins(
        manifest,
        store,
        population_node=CLONE_POPULATION,
        binding_node=CLONE_BINDING,
        parent_binding_node=NATIVE_BINDING,
    )
    native = origins.verify_materialized_population_origins(
        manifest, store, population_node=NATIVE_POPULATION, binding_node=NATIVE_BINDING
    )
    passes(
        "actual_full_materialized_verifier",
        lambda: verified(root, checkout, manifest, store, graph, observed, keys),
    )
    refuses(
        "wrong_expected_producer_key",
        lambda: verified(
            root,
            checkout,
            manifest,
            store,
            graph,
            observed,
            {**keys, graph_cd.ACS_NODE: "0" * 64},
        ),
        "CD_EXPECTED_PRODUCER_KEY",
    )
    refuses(
        "missing_expected_node",
        lambda: verified(
            root,
            checkout,
            manifest,
            store,
            graph,
            observed,
            {k: v for k, v in keys.items() if k != graph_cd.ACS_NODE},
        ),
        "CD_MANIFEST_NODE_ROSTER",
    )
    passes(
        "original_current_binding",
        lambda: current.authenticate_current_lineage(population, population, clone),
    )
    refuses(
        "source_grain_as_clone_binding",
        lambda: current.authenticate_current_lineage(population, population, native),
    )
    refuses(
        "forged_successor_dictionary",
        lambda: current.authenticate_current_lineage(
            population,
            replace(population, version="other"),
            clone,
            parent={},
            successor={},
        ),
        "CD_ACTUAL_SUCCESSOR_REQUIRED",
    )

    housing_value = full.edge_value(
        manifest,
        store,
        composed.CREATE_NODE,
        "housing_universe",
        graph_cd.US_ASEC_HOUSING_UNIVERSE_TYPE,
    )
    prepared_value = full.edge_value(
        manifest,
        store,
        composed.CREATE_NODE,
        "prepared_receipt",
        composed.US_ASEC_PREPARED_RECEIPT_TYPE,
    )
    body = bind_housing_universe(
        housing_value.payload, prepared_receipt=json.loads(prepared_value.payload)
    )
    rows, _ = current._rows(population.frame, clone.document)
    source_ids = body.array("household_id")
    mapping = passes(
        "asec_each_sorted_role",
        lambda: current.verify_asec_roles(
            population.frame, rows, body, source_household_ids=source_ids
        ),
    )
    positions = np.asarray([r["source_position"] for r in mapping], dtype="int64")
    selected = population.frame.table("household").iloc[
        [r["current_position"] for r in mapping]
    ]
    direct = current.composed._arm_household_evidence(
        selected,
        np.ones(len(mapping), dtype=bool),
        np.asarray([r["source_household_id"] for r in mapping], dtype="int64"),
    )
    refuses(
        "whole_clone_view_duplicates_are_not_native_lineage_failure",
        lambda: current.verify_graph_housing_rows(
            direct,
            body,
            positions=positions,
            income_years=np.asarray([r["income_year"] for r in mapping], dtype="int64"),
        ),
    )
    order = list(reversed(range(len(rows))))
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["household"] = tables["household"].iloc[order].reset_index(drop=True)

    # Frame correctly requires sorted group IDs. This is an explicit read-only
    # source-coordinate view for the wrapper, never a fabricated Frame.
    class CoordinateView:
        person = population.frame.person

        def table(self, entity):
            assert entity == "household"
            return tables[entity]

    reordered = CoordinateView()
    mapped = passes(
        "asec_reordered_role_views_preserve_current_mapping",
        lambda: current.verify_asec_roles(
            reordered, [rows[i] for i in order], body, source_household_ids=source_ids
        ),
    )
    assert sorted(
        (r["household_id"], r["source_position"], r["clone_role"]) for r in mapped
    ) == sorted(
        (r["household_id"], r["source_position"], r["clone_role"]) for r in mapping
    )
    bad_rows = copy.deepcopy(rows)
    asec_positions = [
        i
        for i, r in enumerate(rows)
        if r["channel"] == "asec" and r["clone_index"] == 1
    ]
    bad_rows[asec_positions[0]]["clone_index"] = 0
    refuses(
        "asec_duplicate_within_role",
        lambda: current.verify_asec_roles(
            population.frame, bad_rows, body, source_household_ids=source_ids
        ),
        "CD_ASEC_ROLE_DUPLICATE",
    )
    bad_rows = copy.deepcopy(rows)
    bad_rows[asec_positions[0]]["channel"] = "acs"
    refuses(
        "asec_missing_role_row",
        lambda: current.verify_asec_roles(
            population.frame, bad_rows, body, source_household_ids=source_ids
        ),
        "CD_ASEC_ROLE_COVERAGE",
    )
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["person"] = tables["person"].copy()
    hh_id = rows[asec_positions[0]]["household_id"]
    tables["person"].loc[
        tables["person"].person_household_id == hh_id, "source_year"
    ] = 1900
    refuses(
        "asec_cohort_mismatch",
        lambda: current.verify_asec_roles(
            frame_with(population.frame, tables=tables),
            rows,
            body,
            source_household_ids=source_ids,
        ),
    )
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["household"] = tables["household"].copy()
    tables["household"].loc[asec_positions[0], "asec_H_SEQ"] = 123456
    refuses(
        "asec_native_hseq_mismatch",
        lambda: current.verify_asec_roles(
            frame_with(population.frame, tables=tables),
            rows,
            body,
            source_household_ids=source_ids,
        ),
    )

    value = full.edge_value(
        manifest, store, graph_cd.ACS_NODE, "projection", transport.TYPE
    )
    serialnos = tuple(
        sorted(
            set(
                population.frame.table("household").loc[
                    population.frame.table("household").household_support_channel
                    == "acs",
                    "SERIALNO",
                ]
            )
        )
    )
    bound = passes(
        "actual_frame_less_transport",
        lambda: transport.bind_source_projection(
            value, expected_edge=graph_cd.edge_pin(value), selected_serialnos=serialnos
        ),
    )
    assert type(bound) is transport.BoundACSSourceProjection and not isinstance(
        bound, transport.source.AuthenticatedACSHousingSource
    )
    refuses(
        "no_authentic_source_constructor",
        lambda: transport.encode_source_projection(bound, selected_serialnos=serialnos),
        "ACS_AUTHENTICATED_SOURCE_REQUIRED",
    )
    refuses(
        "no_bound_token_constructor",
        lambda: transport.BoundACSSourceProjection(b"", b"", b"", b"", b""),
        "ACS_BOUND_CONSTRUCTOR",
    )
    refuses(
        "wrong_external_edge",
        lambda: transport.bind_source_projection(
            value,
            expected_edge={**graph_cd.edge_pin(value), "producer_key": "0" * 64},
            selected_serialnos=serialnos,
        ),
        "ACS_TRANSPORT_ACTUAL_EDGE",
    )
    refuses(
        "wrong_selection",
        lambda: transport.bind_source_projection(
            value,
            expected_edge=graph_cd.edge_pin(value),
            selected_serialnos=serialnos[:1],
        ),
    )
    corrupt = replace(
        value, payload=value.payload[:-1] + bytes([value.payload[-1] ^ 1])
    )
    refuses(
        "corrupt_transport_payload",
        lambda: transport.bind_source_projection(
            corrupt,
            expected_edge=graph_cd.edge_pin(corrupt),
            selected_serialnos=serialnos,
        ),
        "ACS_TRANSPORT_PAYLOAD",
    )
    bad = replace(value, key="0" * 64)
    refuses(
        "wrong_opaque_key",
        lambda: transport.bind_source_projection(
            bad, expected_edge=graph_cd.edge_pin(bad), selected_serialnos=serialnos
        ),
        "ACS_TRANSPORT_ACTUAL_EDGE",
    )
    bad = replace(value, type=ArtifactType("wrong", 1))
    refuses(
        "wrong_transport_type",
        lambda: transport.bind_source_projection(
            bad, expected_edge=graph_cd.edge_pin(bad), selected_serialnos=serialnos
        ),
        "ACS_TRANSPORT_ACTUAL_EDGE",
    )

    def repack(header=None, projection=None, receipt=None):
        h = dict(bound.header if header is None else header)
        r = bound.receipt_json if receipt is None else transport.source._json(receipt)
        p = (
            bound.projection_json
            if projection is None
            else transport.source._json(projection)
        )
        h.update(
            source_receipt_bytes=len(r),
            source_receipt_sha256=transport.sha(r),
            selected_projection_bytes=len(p),
            selected_projection_sha256=transport.sha(p),
        )
        hb = canonical_json(h)
        payload = transport.MAGIC + struct.pack("<I", len(hb)) + hb + r + p
        payload += bytes.fromhex(transport.sha(payload))
        changed = replace(value, payload=payload)
        return transport.bind_source_projection(
            changed,
            expected_edge=graph_cd.edge_pin(changed),
            selected_serialnos=serialnos,
        )

    refuses(
        "wrong_original_custody",
        lambda: repack(
            header={**bound.header, "original_full_projection_sha256": "0" * 64}
        ),
        "ACS_ORIGINAL_CUSTODY",
    )
    refuses(
        "receipt_selected_not_all",
        lambda: repack(receipt={**bound.receipt, "selection": "selected"}),
        "ACS_TRANSPORT_SOURCE_SCOPE",
    )
    refuses(
        "receipt_archive_forgery",
        lambda: repack(receipt={**bound.receipt, "archives": []}),
        "ACS_TRANSPORT_ARCHIVE_PINS",
    )
    projection = copy.deepcopy(bound.projection)
    # Exact projection row shape is source-defined, not a made-up record codec.
    assert type(projection["households"][0]) is list
    projection["households"][0][
        projection["household_columns"].index("source_member")
    ] = "wrong.csv"
    refuses(
        "wrong_source_member_lineage",
        lambda: repack(projection=projection),
        "ACS_MEMBER_ROW_LINEAGE",
    )

    def changed_population(frame=None, **kwargs):
        return replace(
            population, frame=population.frame if frame is None else frame, **kwargs
        )

    passes(
        "full_boundary_unchanged",
        lambda: current.verify_population_unchanged(population, population),
    )
    changed = changed_population(
        frame_with(
            population.frame,
            metadata={**dict(population.frame.metadata), "invented_probe": True},
        )
    )
    refuses(
        "nonowned_metadata_mutation",
        lambda: current.verify_population_unchanged(population, changed),
    )
    typed_before = changed_population(
        frame_with(
            population.frame,
            metadata={**dict(population.frame.metadata), "invented_probe": 1},
        )
    )
    refuses(
        "typed_metadata_bool_vs_int",
        lambda: current.verify_population_unchanged(typed_before, changed),
    )
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["household"] = tables["household"][
        list(reversed(tables["household"].columns))
    ]
    changed_order = changed_population(frame_with(population.frame, tables=tables))
    refuses(
        "ordered_nonowned_columns",
        lambda: current.verify_population_unchanged(population, changed_order),
        "ORDERED_COLUMNS_OR_AXIS",
    )
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["person"] = tables["person"].copy()
    nullable = next(
        c
        for c in tables["person"]
        if str(tables["person"][c].dtype) == "Int64"
        and tables["person"][c].isna().any()
    )
    hidden = tables["person"][nullable].array.copy()
    hidden._data[np.flatnonzero(hidden._mask)[0]] += 1
    tables["person"][nullable] = hidden
    changed_hidden = changed_population(frame_with(population.frame, tables=tables))
    refuses(
        "supported_nullable_hidden_backing",
        lambda: current.verify_population_unchanged(population, changed_hidden),
    )
    from microcosm.frame.weights import MassChangeRecord

    f = population.frame
    changed_log = Frame(
        {e: f.table(e) for e in f.entities},
        f.schema,
        {e: f.weights_for(e) for e in f.weighted_entities},
        f.strata,
        metadata=f.metadata,
        mass_log=(
            *f.mass_log,
            MassChangeRecord("household", 0.0, 0.0, 1.0, "invented mutation"),
        ),
    )
    refuses(
        "frame_mass_log_mutation",
        lambda: current.verify_population_unchanged(
            population, changed_population(changed_log)
        ),
        "FRAME_MASS_LOG",
    )
    changed_anchors = {e: a.copy() for e, a in population.design_weights.items()}
    assert changed_anchors
    entity = next(iter(changed_anchors))
    changed_anchors[entity][0] += 1
    refuses(
        "nonempty_original_anchor_mutation",
        lambda: current.verify_population_unchanged(
            population, changed_population(design_weights=changed_anchors)
        ),
    )
    zero_anchors = {e: a.copy() for e, a in population.design_weights.items()}
    zero_anchors[entity][0] = 0
    zero_before = changed_population(design_weights=zero_anchors)
    passes(
        "materialized_zero_anchor_unchanged",
        lambda: current.verify_population_unchanged(zero_before, zero_before),
    )
    refuses(
        "materialized_zero_anchor_mutation",
        lambda: current.verify_population_unchanged(zero_before, population),
    )
    assert population.mass_ledger
    refuses(
        "population_ledger_omission",
        lambda: current.verify_population_unchanged(
            population, changed_population(mass_ledger=population.mass_ledger[:-1])
        ),
    )
    w = population.frame.weights_for("household")
    values = w.values.copy()
    values[asec_positions[:2]] = 0
    zero = changed_population(
        frame_with(population.frame, weights={"household": Weights(values, w.kind)})
    )
    passes(
        "materialized_zero_weight_unchanged",
        lambda: current.verify_population_unchanged(zero, zero),
    )
    passes(
        "zero_weight_identifier_coverage",
        lambda: current._rows(zero.frame, clone.document),
    )
    omitted = zero.frame.select(zero.frame.person.person_household_id != hh_id)
    assert omitted.n("household") == zero.frame.n("household") - 1
    assert math.fsum(omitted.weights_for("household").values) == math.fsum(
        zero.frame.weights_for("household").values
    )
    refuses(
        "zero_weight_whole_household_and_members_omitted",
        lambda: current._rows(omitted, clone.document),
        "CD_CURRENT_HOUSEHOLDS",
    )
    refuses(
        "unadmitted_weight_change",
        lambda: current.authenticate_current_lineage(population, zero, clone),
        "CD_ACTUAL_SUCCESSOR_REQUIRED",
    )
    # Mutating a linked person at zero household mass must still fail identity.
    tables = {e: zero.frame.table(e) for e in zero.frame.entities}
    tables["person"] = tables["person"].copy()
    selected_person = tables["person"].index[
        tables["person"].person_household_id == hh_id
    ][0]
    tables["person"].loc[selected_person, "person_household_id"] = rows[
        asec_positions[1]
    ]["household_id"]
    wrong_membership = frame_with(zero.frame, tables=tables)
    refuses(
        "zero_weight_membership_mutation",
        lambda: current._rows(wrong_membership, clone.document),
    )

    geo_inputs = {
        item.name: full.edge_value(
            manifest, store, item.producer, item.artifact, item.type
        )
        for item in graph.node(graph_cd.GEOGRAPHY_NODE).artifact_inputs
    }
    geo = graph_cd.native_geography_body(
        observed[NATIVE_POPULATION].frame,
        native.document,
        **{k: geo_inputs[k] for k in ("assignment", "ladder", "crosswalk")},
    )
    passes(
        "native119th_geography_carry",
        lambda: current.attest_current_geography(
            population.frame,
            clone.document,
            native_rows=geo["records"],
            assignment_evidence=geo["assignment_evidence"],
        ),
    )
    for label, change in (
        ("wrong_vintage", {"district_vintage": "118th_congress"}),
        ("missing_dc_roster", {"target_district_count": 435}),
        ("redraw_seed", {"seed": 1}),
    ):
        refuses(
            label,
            lambda change=change: current.attest_current_geography(
                population.frame,
                clone.document,
                native_rows=geo["records"],
                assignment_evidence={**geo["assignment_evidence"], **change},
            ),
            "CD_GEOGRAPHY_METHOD",
        )
    tables = {e: population.frame.table(e) for e in population.frame.entities}
    tables["household"] = tables["household"].copy()
    origin_key = rows[0]["origin_key"]
    affected = [i for i, r in enumerate(rows) if r["origin_key"] == origin_key]
    tables["household"].loc[affected, "congressional_district_geoid"] = 1100
    bad_frame = frame_with(population.frame, tables=tables)
    refuses(
        "coherent_two_role_geography_change_vs_actual_native",
        lambda: current.attest_current_geography(
            bad_frame,
            clone.document,
            native_rows=geo["records"],
            assignment_evidence=geo["assignment_evidence"],
        ),
        "CD_GEOGRAPHY_CARRIED",
    )

    # Separate real source codec with all eight source classes. This is not a
    # claim that the five-origin graph fixture contains GQ or vacant records.
    source_tests = peer(checkout, "test_us_acs_housing_source")
    with pytest.MonkeyPatch.context() as patch:
        fixture = source_tests.build_fixture(root / "source-classes", patch)
        observation = fixture.produce()
        serials = tuple(sorted(observation.households.SERIALNO))
        payload = transport.encode_source_projection(
            observation, selected_serialnos=serials
        )
        v = replace(
            value,
            payload=payload,
            key=opaque_artifact_key("f" * 64, "projection"),
            producer_key="f" * 64,
        )
        class_bound = transport.bind_source_projection(
            v, expected_edge=graph_cd.edge_pin(v), selected_serialnos=serials
        )
        codes = transport.source._tables(class_bound.projection)[-1]
        assert set(codes.occupied_hu) == {0, 1, 2}
        assert set(codes.hu_tenure_class) == {0, 1, 2, 3}
        assert np.any(codes.unresolved_reasons.to_numpy() != 0)
        records.append(
            {
                "name": "real_source_transport_occupied_unknown_vacancy_gq",
                "status": "PASS",
            }
        )
    io_tests = peer(checkout, "test_cd_source_adapters_io")
    io_root = root / "io-checks"
    io_root.mkdir()
    records.extend(io_tests.exercise(io_root, result=result))
    from microcosm.build.us_runtime import graph_cd_source_implementation as identity

    roots = identity._package_roots()
    unchanged = identity.implementation_hash()
    upstream = origins.NativeACSOriginKernel().implementation_hash()

    class AlteredBytes:
        def __init__(self, path, suffix):
            self.path, self.suffix = path, suffix

        def read_bytes(self):
            return self.path.read_bytes() + self.suffix

    class ScopedRoot:
        def __init__(self, selected, suffix):
            self.selected, self.suffix = selected, suffix

        def __truediv__(self, name):
            path = roots["microcosm.build"] / name
            return AlteredBytes(path, self.suffix) if name == self.selected else path

    for selected, suffix, label in (
        (
            "us_runtime/cd_current_source_evidence.py",
            b"\n# identity witness\n",
            "imported_verifier_bytes_rekey_extension",
        ),
        (
            "cd_benchmark/us-cd-published-benchmark-v4.json",
            b"\n",
            "protocol_resource_bytes_rekey_extension",
        ),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                identity,
                "_package_roots",
                lambda selected=selected, suffix=suffix: {
                    **roots,
                    "microcosm.build": ScopedRoot(selected, suffix),
                },
            )
            assert identity.implementation_hash() != unchanged
            assert origins.NativeACSOriginKernel().implementation_hash() == upstream
            records.append({"name": label, "status": "PASS"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            identity,
            "_package_roots",
            lambda: {
                **roots,
                "microcosm.build": ScopedRoot(
                    "us_runtime/cd_current_source_evidence.py",
                    b"\nimport deliberately_unclassified_cd_dependency\n",
                ),
            },
        )
        refuses(
            "new_unclassified_import",
            identity.implementation_hash,
            "CD_UNCLASSIFIED_DEPENDENCY",
        )
    inventory_tests = peer(checkout, "test_us_spine_blindness")
    for name in (
        "test_us_runtime_module_classification_is_complete",
        "test_runtime_classification_rejects_a_new_unreviewed_module",
        "test_us_runtime_frame_rebuilds_preserve_immutable_metadata",
        "test_runtime_population_operators_are_source_spine_blind",
    ):
        passes(name, getattr(inventory_tests, name))
    return {
        "checks": records,
        "checks_count": len(records),
        "nodes": len(keys),
        "calls": [],
        "hits": len(keys),
        "anchor_roster": sorted(population.design_weights),
        "limits": [
            "zero weights/anchors are separate materialized comparator witnesses",
            "eight-class source transport is distinct from the five-origin composed fixture",
            "no genuine or scientific definition admission",
        ],
    }
