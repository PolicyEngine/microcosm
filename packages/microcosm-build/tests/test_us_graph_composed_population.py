"""Composed prepared-ASEC/native-ACS population: real sources, real kernels.

Every source byte is invented. The ASEC arm is the real ten-file prepared
directory read by the real preparation; the ACS arm is a real two-archive PUMS
pair read by the real ACS codec. Private source pins are substituted in test
scope only, including inside the fresh interpreters. Nothing here certifies a
population, a calibration or a release.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_income_observations as income_source
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import graph_composed_population as composed
from microcosm.build.us_runtime import graph_sources
from microcosm.build.us_runtime.asec_current_money import _sha
from microcosm.build.us_runtime.asec_current_money_selection import (
    parse_current_money_body,
)
from microcosm.build.us_runtime.graph_asec_income import bind_income_observations
from microcosm.build.us_runtime.graph_asec_prepared import PreparedGraphError
from microcosm.build.us_runtime.graph_context import (
    _mass_records,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_geography import (
    GEOGRAPHY_PHASE,
    USGeographyAssignKernel,
    USGeographyBoundaryKernel,
    USGeographyGateKernel,
    USGeographyLookupKernel,
)
from microcosm.build.us_runtime.graph_housing_universe import (
    HOUSEHOLD_EVIDENCE_COLUMNS,
    bind_housing_universe,
)
from microcosm.build.us_runtime.graph_sources import (
    USSpineHarmonizeKernel,
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.build.us_runtime.stacked_spine import (
    ACS_STACKED_SUPPORT_CHANNEL,
    STACKED_SPINE_MANIFEST_KEY,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    support_channel_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind
from microcosm.graph import (
    ContentStore,
    NodeRejectedError,
    Owned,
    StoreMiss,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token

SEED = 578
#: Whole-source composition. Source preparation is not sampling: both arms are
#: read complete and the declared seeded draw keeps every household.
FRACTION = 1.0
KERNELS = (
    composed.USComposedPopulationCreateKernel,
    USSpineHarmonizeKernel,
    USGeographyLookupKernel,
    USGeographyBoundaryKernel,
    USGeographyAssignKernel,
    USGeographyGateKernel,
)
ASEC_SOURCE = composed.ASEC_PREPARED_SOURCE_NAME
ACS_SOURCE = composed.ACS_NATIVE_SOURCE_NAME
#: The ACS arm's native PUMA. The invented ladder holds one PUMA per state.
NATIVE_PUMA = "00100"


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"composed_{name[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restore_test_pins(root):
    from microcosm.build.us_runtime import asec_student_controls as student

    pins = json.loads((Path(root) / "invented-pins.json").read_text())
    legacy._SOURCE_PINS = (
        pins["source"][0],
        pins["source"][1],
        tuple(tuple(item) for item in pins["source"][2]),
    )
    restoration._MEMBER_PINS = tuple(tuple(item) for item in pins["members"])
    student._MEMBER_PINS = tuple(tuple(item) for item in pins["students"])
    income_source._MEMBER_PINS = tuple(tuple(item) for item in pins["income"])


def _household_row(serialno, **overrides):
    row = {
        "SERIALNO": serialno,
        "ST": "06",
        "PUMA": NATIVE_PUMA,
        "WGTP": 10,
        "NP": 1,
        "ADJHSG": 1_000_000,
        "TEN": 1,
        "RNTP": None,
        "GRNTP": None,
        "TAXAMT": 2_400,
        "TYPEHUGQ": 1,
    }
    row.update(overrides)
    return row


def _person_row(serialno, sporder, relationship, **overrides):
    row = {
        "SERIALNO": serialno,
        "SPORDER": sporder,
        "RELSHIPP": relationship,
        "AGEP": 40,
        "SEX": 1,
        "MAR": 1,
        "ADJINC": 1_000_000,
        "WAGP": 50_000,
        "SEMP": 0,
        "SSP": 0,
        "SSIP": 0,
        "RETP": 0,
        "INTP": 0,
        "PWGTP": 10,
    }
    row.update(overrides)
    return row


def _write_csv_zip(path, members):
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, rows in members.items():
            archive.writestr(name, pd.DataFrame(rows).to_csv(index=False))


def invented_acs_archives(root):
    """Two invented PUMS archives: reordered, multi-person, leading-zero IDs.

    ``0000000000123`` and ``0000000000045`` are deliberately leading-zero
    SERIALNO strings split across archive members out of order, so a numeric
    coercion anywhere on the route would be visible as a changed identifier.
    The three-person household carries a nonrelative, which splits it across
    two tax units and three marital units; one group-quarters record with the
    zero design weight the reader requires is included.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv_zip(
        root / "csv_hus.zip",
        {
            "psam_husa.csv": [
                _household_row("0000000000123", NP=3, ST="36"),
                _household_row(
                    "2024GQ0000004",
                    NP=1,
                    TYPEHUGQ=2,
                    WGTP=0,
                    TEN=None,
                    TAXAMT=None,
                ),
            ],
            "psam_husb.csv": [
                _household_row(
                    "0000000000045",
                    NP=2,
                    TEN=3,
                    RNTP=1_200,
                    GRNTP=1_400,
                    TAXAMT=None,
                    WGTP=25,
                ),
            ],
        },
    )
    _write_csv_zip(
        root / "csv_pus.zip",
        {
            "psam_pusa.csv": [
                _person_row("0000000000123", 1, 20),
                _person_row("0000000000045", 2, 25, AGEP=9, MAR=5, WAGP=None),
                _person_row("2024GQ0000004", 1, 38, AGEP=22, PWGTP=17),
            ],
            "psam_pusb.csv": [
                _person_row("0000000000123", 3, 36, AGEP=31, MAR=5, WAGP=15_000),
                _person_row("0000000000123", 2, 21, SEX=2, AGEP=38),
                _person_row("0000000000045", 1, 20, AGEP=33, SEX=2, PWGTP=25),
            ],
        },
    )
    return root


def _lookup_files(root):
    return _module("test_us_graph_geography.py")._lookup_files(Path(root))


def _sources(root):
    root = Path(root)
    sources = {name: Path(path) for name, path in _lookup_files(root).items()}
    sources[ASEC_SOURCE] = root / "prepared-source"
    sources[ACS_SOURCE] = root / "acs"
    return sources


def _columns(root):
    return tuple(
        Owned(*entry) for entry in json.loads((Path(root) / "columns.json").read_text())
    )


def _graph(root):
    return composed.composed_population_graph(
        _columns(root), sample_fraction=FRACTION, sample_seed=SEED
    )


def _run(root, *, resume="forbid", sources=None):
    root = Path(root)
    store = ContentStore(root / "store", codecs=us_source_codecs())
    manifest = run_graph(
        compile_graph(_graph(root)),
        sources=_sources(root) if sources is None else sources,
        store=store,
        kernels=composed.composed_population_registry(),
        resume=resume,
    )
    return manifest, store


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _direct(root):
    """Call the real preparation, ACS codec, bridge, stack and geography."""
    root = Path(root)
    result = composed.compose_from_sources(
        root / "prepared-source",
        root / "acs",
        sample_fraction=FRACTION,
        sample_seed=SEED,
    )
    (root / "columns.json").write_bytes(
        canonical_json(
            [
                [item.entity, item.column, item.dtype]
                for item in frame_column_declarations(result.frame)
            ]
        )
    )
    write_frame_checkpoint(root / "direct-prepared.h5", result.frame)
    (root / "direct-preparation.json").write_bytes(
        canonical_json(composed._json_data(result.preparation))
    )
    (root / "direct-origin.json").write_bytes(canonical_json(result.source_origin))
    (root / "direct-money.bin").write_bytes(result.money_payload)
    (root / "direct-receipt.json").write_bytes(result.receipt_payload)
    (root / "direct-housing.bin").write_bytes(result.housing_universe_payload)
    (root / "direct-income.bin").write_bytes(result.income_observations_payload)
    (root / "direct-context.bin").write_bytes(encode_us_frame_context(result.frame))
    return {
        "pid": os.getpid(),
        "storage_bridge": list(result.storage_bridge),
        "rows": {entity: int(result.frame.n(entity)) for entity in result.frame.entities},
    }


def _cache_only(root):
    """Replay from the store alone; any kernel or source read is a failure."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("cache-only replay executed a kernel or source loader")

    for cls in KERNELS:
        cls.run = forbidden
    # Only the decode paths this graph's kernel calls are replaced. The
    # registered codec loaders keep their declared references: substituting a
    # public loader is itself refused before cache lookup.
    composed.prepare_asec_current_money_population = forbidden
    composed.load_graph_acs = forbidden
    composed.compose_from_sources = forbidden
    manifest, store = _run(root, resume="require")
    return {
        "pid": os.getpid(),
        "key": manifest.key,
        "nodes": {name: record.key for name, record in manifest.nodes.items()},
        "artifacts": {
            name: dict(record.opaque_artifacts)
            for name, record in manifest.nodes.items()
        },
        "all_hits": all(record.hit for record in manifest.nodes.values()),
        "kernel_calls": len(calls),
        "context_sha256": _sha(
            _payload(manifest, store, composed.HARMONIZE_NODE, "frame_context")
        ),
        "geography_context_sha256": _sha(
            _payload(manifest, store, GEOGRAPHY_PHASE, "frame_context")
        ),
        "origin_sha256": _sha(
            _payload(manifest, store, composed.CREATE_NODE, "source_origin")
        ),
    }


def _process(root, mode):
    code = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("composed_graph_process", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._restore_test_pins(sys.argv[2])
print(json.dumps(getattr(module, sys.argv[3])(sys.argv[2]), sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, str(root), mode],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    (Path(root) / f"{mode}.stdout.log").write_text(result.stdout)
    (Path(root) / f"{mode}.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-composed-population")
    from microcosm.build.us_runtime import asec_student_controls as student

    with pytest.MonkeyPatch.context() as patch:
        _module("test_us_asec_prepared_source.py").invented_prepared_directory(
            root,
            patch,
            households_per_cohort=2,
            housing_contexts={
                2023: {"H_TENURE": 0},
                2024: {"H_LIVQRT": 10, "HRHTYPE": 10},
            },
        )
        invented_acs_archives(root / "acs")
        (root / "invented-pins.json").write_bytes(
            canonical_json(
                {
                    "source": legacy._SOURCE_PINS,
                    "members": restoration._MEMBER_PINS,
                    "students": student._MEMBER_PINS,
                    "income": income_source._MEMBER_PINS,
                }
            )
        )
        direct = _process(root, "_direct")
        calls, contexts = [], {}
        for cls in KERNELS:
            original = cls.run

            def capture(self, context, *, _run=original):
                contexts.setdefault(context.node.id, context)
                calls.append(context.node.id)
                return _run(self, context)

            patch.setattr(cls, "run", capture)
        manifest, store = _run(root)
        yield root, manifest, store, calls, direct, contexts


def test_cold_graph_matches_the_direct_composition(executed):
    root, manifest, store, calls, direct, _contexts = executed
    assert calls == [
        composed.CREATE_NODE,
        composed.HARMONIZE_NODE,
        f"{GEOGRAPHY_PHASE}.lookup",
        f"{GEOGRAPHY_PHASE}.boundary",
        GEOGRAPHY_PHASE,
        f"{GEOGRAPHY_PHASE}.gate",
    ]
    assert not any(record.hit for record in manifest.nodes.values())
    prepared = manifest.population(composed.CREATE_NODE)
    expected = load_frame_checkpoint(root / "direct-prepared.h5").frame
    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            prepared.table(entity), expected.table(entity), check_exact=True
        )
    assert (
        prepared.weights_for("household").values.tobytes()
        == expected.weights_for("household").values.tobytes()
    )
    pd.testing.assert_series_equal(prepared.strata, expected.strata)
    assert prepared.weights_for("household").kind is WeightKind.DESIGN
    assert (
        _payload(manifest, store, composed.CREATE_NODE, "frame_context")
        == (root / "direct-context.bin").read_bytes()
    )
    assert (
        _payload(manifest, store, composed.CREATE_NODE, "preparation")
        == (root / "direct-preparation.json").read_bytes()
    )
    # Both arms are whole; the declared draw keeps every source household.
    assert direct["rows"] == {entity: int(prepared.n(entity)) for entity in prepared.entities}


def test_only_string_storage_is_bridged_between_the_two_arms(executed):
    _root, _manifest, _store, _calls, direct, _contexts = executed
    bridged = {(item["entity"], item["column"]) for item in direct["storage_bridge"]}
    assert bridged == {
        ("person", "PERIDNUM"),
        ("person", "tax_unit_role_input"),
        ("tax_unit", "filing_status_input"),
    }
    for item in direct["storage_bridge"]:
        assert item["to"] == str(CANONICAL_STRING_DTYPE)
        assert item["from"] != item["to"]


def test_both_arms_survive_with_their_native_identity_and_absences(executed):
    _root, manifest, _store, _calls, _direct, _contexts = executed
    frame = manifest.population(composed.HARMONIZE_NODE)
    household = frame.table("household")
    person = frame.person
    channel = household[support_channel_column("household")]
    assert set(channel) == {BASE_ASEC_SUPPORT_CHANNEL, ACS_STACKED_SUPPORT_CHANNEL}
    acs_rows = channel == ACS_STACKED_SUPPORT_CHANNEL
    asec_rows = channel == BASE_ASEC_SUPPORT_CHANNEL
    # Raw ACS SERIALNO strings keep their leading zeros; the ASEC arm has none.
    assert household["SERIALNO"].dtype == dtype_for_token("string")
    assert sorted(household.loc[acs_rows, "SERIALNO"]) == [
        "0000000000045",
        "0000000000123",
        "2024GQ0000004",
    ]
    assert household.loc[asec_rows, "SERIALNO"].isna().all()
    # Native ASEC household sequence numbers survive; the ACS arm has none.
    assert household["asec_H_SEQ"].notna()[asec_rows.to_numpy()].all()
    assert household.loc[acs_rows, "asec_H_SEQ"].isna().all()
    # A source-specific column absent on the opposite arm stays missing, never
    # laundered to zero: TYPEHUGQ is ACS-only, H_TENURE is ASEC-only.
    assert household.loc[asec_rows, "TYPEHUGQ"].isna().all()
    assert household.loc[acs_rows, "H_TENURE"].isna().all()
    assert not (household["TYPEHUGQ"].fillna(-1) == 0).any()
    person_channel = person[support_channel_column("person")]
    assert person.loc[person_channel == ACS_STACKED_SUPPORT_CHANNEL, "PERIDNUM"].isna().all()
    assert person.loc[person_channel == BASE_ASEC_SUPPORT_CHANNEL, "PERIDNUM"].notna().all()
    # Cohort/vintage axis: the ASEC arm restores three income years.
    asec_years = person.loc[person_channel == BASE_ASEC_SUPPORT_CHANNEL, "source_year"]
    acs_years = person.loc[person_channel == ACS_STACKED_SUPPORT_CHANNEL, "source_year"]
    assert set(asec_years) == {2022, 2023, 2024}
    assert set(acs_years) == {2024}


def test_membership_and_lineage_are_ordered_and_complete(executed):
    _root, manifest, _store, _calls, _direct, _contexts = executed
    frame = manifest.population(composed.HARMONIZE_NODE)
    person = frame.person
    for group in US_SCHEMA.group_entities:
        table = frame.table(group)
        ids = table[US_SCHEMA.entity_id_column(group)].to_numpy()
        membership = person[US_SCHEMA.membership_column(group)].to_numpy()
        # Group rows are the sorted distinct membership values, and every
        # person names a group row that exists.
        np.testing.assert_array_equal(ids, np.unique(membership))
        assert np.isin(membership, ids).all()
        # Assembly's pre-clone source ID is the group's own post-remap ID.
        np.testing.assert_array_equal(
            table[support_source_id_column(group)].to_numpy(), ids
        )
    # The three-person ACS household really is one household with two tax units.
    channel = person[support_channel_column("person")]
    acs = person[channel == ACS_STACKED_SUPPORT_CHANNEL]
    sizes = acs.groupby("person_household_id").size().sort_values().tolist()
    assert sizes == [1, 2, 3]
    biggest = acs.person_household_id.value_counts().idxmax()
    assert acs.loc[acs.person_household_id == biggest, "person_tax_unit_id"].nunique() == 2


def test_source_origin_artifact_binds_each_arm_to_its_output_rows(executed):
    root, manifest, store, _calls, _direct, _contexts = executed
    payload = _payload(manifest, store, composed.CREATE_NODE, "source_origin")
    assert payload == (root / "direct-origin.json").read_bytes()
    document = composed.bind_composed_source_origin(payload)
    frame = manifest.population(composed.CREATE_NODE)
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        channels = table[support_channel_column(entity)].astype(str).tolist()
        assert document["assembled"][entity]["rows_by_channel"] == {
            name: channels.count(name) for name in sorted(set(channels))
        }
        assert document["assembled"][entity]["ordered_source_to_output_sha256"] == _sha(
            canonical_json(
                [
                    [channel, source_id, output_id]
                    for channel, source_id, output_id in zip(
                        channels,
                        table[support_source_id_column(entity)].to_numpy("int64").tolist(),
                        table[US_SCHEMA.entity_id_column(entity)].to_numpy("int64").tolist(),
                        strict=True,
                    )
                ]
            )
        )
    acs_arm = document["arms"][ACS_STACKED_SUPPORT_CHANNEL]
    assert acs_arm["native_identity"]["household.SERIALNO"]["missing"] == 0
    assert acs_arm["person_cohorts"] == {"2024": 6}
    asec_arm = document["arms"][BASE_ASEC_SUPPORT_CHANNEL]
    assert set(asec_arm["person_cohorts"]) == {"2022", "2023", "2024"}
    assert "household.asec_H_SEQ" in asec_arm["native_identity"]
    assert document["release_eligible"] is False


def test_harmonization_declares_the_typed_importance_anchor_and_mass(executed):
    _root, manifest, store, _calls, _direct, _contexts = executed
    prepared = manifest.population(composed.CREATE_NODE)
    harmonized = manifest.population(composed.HARMONIZE_NODE)
    assert prepared.weights_for("household").kind is WeightKind.DESIGN
    assert harmonized.weights_for("household").kind is WeightKind.IMPORTANCE
    assert not np.array_equal(
        prepared.weights_for("household").values,
        harmonized.weights_for("household").values,
    )
    document = json.loads(
        _payload(manifest, store, composed.HARMONIZE_NODE, "frame_context")
    )
    assert document["weight_sources"] == {"household": "importance"}
    _mass_records(document["mass_log"])
    assert document["mass_log"]
    receipt = manifest.nodes[composed.HARMONIZE_NODE].receipt
    assert receipt["mass"]["policy"] == "declared"
    assert receipt["mass"]["before"] > 0.0 and receipt["mass"]["after"] > 0.0
    assert set(receipt["mass"]["stratum_before"]) == set(receipt["mass"]["stratum_after"])
    assert set(document["metadata"][STACKED_SPINE_MANIFEST_KEY]["household_mass_shares"]) == {
        BASE_ASEC_SUPPORT_CHANNEL,
        ACS_STACKED_SUPPORT_CHANNEL,
    }


def test_prepared_evidence_travels_and_still_reparses(executed):
    root, manifest, store, _calls, _direct, _contexts = executed
    receipt_payload = _payload(manifest, store, composed.CREATE_NODE, "prepared_receipt")
    money_payload = _payload(manifest, store, composed.CREATE_NODE, "current_money")
    housing_payload = _payload(manifest, store, composed.CREATE_NODE, "housing_universe")
    income_payload = _payload(
        manifest, store, composed.CREATE_NODE, "income_observations"
    )
    assert receipt_payload == (root / "direct-receipt.json").read_bytes()
    assert money_payload == (root / "direct-money.bin").read_bytes()
    assert housing_payload == (root / "direct-housing.bin").read_bytes()
    assert income_payload == (root / "direct-income.bin").read_bytes()
    prepared_receipt = json.loads(receipt_payload)
    body = parse_current_money_body(
        money_payload,
        expected_header_sha256=prepared_receipt["money_header_sha256"],
        expected_content_sha256=prepared_receipt["money_content_sha256"],
        field_entities=tuple(
            tuple(item) for item in prepared_receipt["field_entities"]
        ),
    )
    # The evidence describes the ASEC arm alone, not the composed population.
    frame = manifest.population(composed.CREATE_NODE)
    channel = frame.person[support_channel_column("person")]
    assert body.person_rows == int((channel == BASE_ASEC_SUPPORT_CHANNEL).sum())
    bind_housing_universe(housing_payload, prepared_receipt=prepared_receipt)
    bind_income_observations(income_payload, prepared_receipt=prepared_receipt)
    household = frame.table("household")
    asec = household[support_channel_column("household")] == BASE_ASEC_SUPPORT_CHANNEL
    for column in HOUSEHOLD_EVIDENCE_COLUMNS:
        assert household.loc[asec, column].notna().all()
        assert household.loc[~asec, column].isna().all()


def test_geography_assigns_over_both_arms_and_keeps_native_acs_pumas(executed):
    _root, manifest, store, _calls, _direct, _contexts = executed
    assigned = manifest.population(f"{GEOGRAPHY_PHASE}.boundary")
    household = assigned.table("household")
    channel = household[support_channel_column("household")]
    acs = channel == ACS_STACKED_SUPPORT_CHANNEL
    assert (household.loc[acs, "puma"] == f"06{NATIVE_PUMA}").sum() + (
        household.loc[acs, "puma"] == f"36{NATIVE_PUMA}"
    ).sum() == int(acs.sum())
    assert household["puma"].notna().all()
    assert household["county_fips"].notna().all()
    assert (household["congressional_district_geoid"] > 0).all()
    assert assigned.weights_for("household").kind is WeightKind.IMPORTANCE
    document = json.loads(_payload(manifest, store, GEOGRAPHY_PHASE, "frame_context"))
    assert document["weight_sources"] == {"household": "importance"}


def test_checkpoint_round_trip_reproduces_the_typed_context(executed):
    root, manifest, store, _calls, _direct, _contexts = executed
    payload = _payload(manifest, store, GEOGRAPHY_PHASE, "frame_context")
    document = json.loads(payload)
    actual = manifest.population(f"{GEOGRAPHY_PHASE}.boundary")
    restored = Frame(
        {entity: actual.table(entity) for entity in actual.entities},
        actual.schema,
        {"household": actual.weights_for("household")},
        actual.strata,
        mass_log=_mass_records(document["mass_log"]),
        metadata=document["metadata"],
    )
    assert encode_us_frame_context(restored) == payload
    write_frame_checkpoint(root / "composed.h5", restored, metadata={"scope": "fixture"})
    loaded = load_frame_checkpoint(
        root / "composed.h5", frame_metadata=document["metadata"]
    ).frame
    assert encode_us_frame_context(loaded) == payload
    for entity in actual.entities:
        pd.testing.assert_frame_equal(
            loaded.table(entity), actual.table(entity), check_exact=True
        )


def test_fresh_process_warm_replay_is_all_hits_and_runs_no_kernel(executed):
    root, manifest, store, _calls, direct, _contexts = executed
    warm = _process(root, "_cache_only")
    assert warm["pid"] != direct["pid"] != os.getpid()
    assert warm["all_hits"] is True
    assert warm["kernel_calls"] == 0
    assert warm["key"] == manifest.key
    assert warm["nodes"] == {name: record.key for name, record in manifest.nodes.items()}
    assert warm["artifacts"] == {
        name: dict(record.opaque_artifacts) for name, record in manifest.nodes.items()
    }
    assert warm["context_sha256"] == _sha(
        _payload(manifest, store, composed.HARMONIZE_NODE, "frame_context")
    )
    assert warm["geography_context_sha256"] == _sha(
        _payload(manifest, store, GEOGRAPHY_PHASE, "frame_context")
    )
    assert warm["origin_sha256"] == _sha(
        _payload(manifest, store, composed.CREATE_NODE, "source_origin")
    )


def test_registry_declares_exactly_the_kernels_this_graph_runs(executed):
    root, *_rest = executed
    assert composed.composed_population_registry().refs() == tuple(
        sorted(cls.ref for cls in KERNELS)
    )
    assert composed.composed_population_registry(geography=False).refs() == tuple(
        sorted((USSpineHarmonizeKernel.ref, composed.USComposedPopulationCreateKernel.ref))
    )
    graph = _graph(root)
    assert tuple(source.name for source in graph.sources) == (
        ASEC_SOURCE,
        ACS_SOURCE,
        "us_puma_ladder_2020",
        "us_cd_crosswalk_117_119",
    )
    # The reviewed raw-stage assembly seam is untouched by this lane.
    assert graph_sources.USAssemblyCreateKernel.ref == "us.production.spine_prepare@2"
    assert tuple(
        source.name for source in graph_sources.us_assembly_graph(
            _columns(root), sample_fraction=FRACTION, sample_seed=SEED
        ).sources
    ) == ("asec_raw_stage", "acs_native")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("phase", "NODE_PHASE"),
        ("extra_param", "NODE_PARAMS"),
        ("source_order", "NODE_SOURCES"),
        ("artifact_outputs", "NODE_ARTIFACT_OUTPUTS"),
        ("columns", "COMPOSED_COLUMN_INVENTORY"),
    ],
)
def test_create_kernel_refuses_a_misdeclared_node(executed, mutation, reason):
    _root, _manifest, _store, _calls, _direct, contexts = executed
    context = contexts[composed.CREATE_NODE]
    node = context.node
    if mutation == "phase":
        node = replace(node, params={**node.params, "phase": "assemble_stacked_spine"})
    elif mutation == "extra_param":
        node = replace(node, params={**node.params, "unexpected": 1})
    elif mutation == "source_order":
        node = replace(node, sources=(ACS_SOURCE, ASEC_SOURCE))
    elif mutation == "artifact_outputs":
        node = replace(node, artifact_outputs=node.artifact_outputs[:-1])
    else:
        node = replace(node, outputs=node.outputs[:-1])
    with pytest.raises(PreparedGraphError, match=reason):
        composed.USComposedPopulationCreateKernel().run(
            replace(context, node=node, params=dict(node.params))
        )


def test_harmonize_refuses_a_retyped_context_or_a_split_producer(executed):
    _root, _manifest, _store, _calls, _direct, contexts = executed
    context = contexts[composed.HARMONIZE_NODE]
    artifacts = dict(context.artifacts)
    bound = artifacts["frame_context"]
    document = json.loads(bound.payload)
    document["entities"]["household"]["ordered_ids_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="context identity differs"):
        USSpineHarmonizeKernel().run(
            replace(
                context,
                artifacts={
                    **artifacts,
                    "frame_context": replace(bound, payload=canonical_json(document)),
                },
            )
        )
    with pytest.raises(ValueError, match="typed producer"):
        USSpineHarmonizeKernel().run(
            replace(
                context,
                artifacts={
                    **artifacts,
                    "preparation": replace(
                        artifacts["preparation"], producer_key="f" * 64
                    ),
                },
            )
        )


def test_origin_binding_refuses_a_tampered_or_retyped_payload(executed):
    root, manifest, store, _calls, _direct, _contexts = executed
    payload = _payload(manifest, store, composed.CREATE_NODE, "source_origin")
    document = json.loads(payload)
    assert composed.bind_composed_source_origin(payload)["schema"]
    for key, value, reason in (
        ("schema", "microcosm.us.other.v1", "ORIGIN_SCHEMA"),
        ("phase", "assemble_stacked_spine", "ORIGIN_PHASE"),
        ("release_eligible", True, "ORIGIN_RELEASE_CLAIM"),
    ):
        with pytest.raises(PreparedGraphError, match=reason):
            composed.bind_composed_source_origin(
                canonical_json({**document, key: value})
            )
    with pytest.raises(PreparedGraphError, match="ORIGIN_ARMS"):
        composed.bind_composed_source_origin(
            canonical_json(
                {**document, "arms": {BASE_ASEC_SUPPORT_CHANNEL: document["arms"][BASE_ASEC_SUPPORT_CHANNEL]}}
            )
        )
    with pytest.raises(PreparedGraphError, match="ORIGIN_CANONICAL"):
        composed.bind_composed_source_origin(b"  " + payload)
    del root


def test_changed_acs_source_bytes_cannot_replay_the_cache(executed):
    """A different ACS archive is a different node, never a silent cache hit."""
    root, cold, _store, _calls, _direct, _contexts = executed
    path = root / "acs" / "csv_pus.zip"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(StoreMiss):
            _run(root, resume="require")
        recomputed, _store = _run(root, resume="forbid")
        assert (
            recomputed.nodes[composed.CREATE_NODE].key
            != cold.nodes[composed.CREATE_NODE].key
        )
    finally:
        path.write_bytes(original)


def test_changed_prepared_source_bytes_cannot_replay_or_receive_authority(executed):
    root, *_rest = executed
    fixture = _module("test_us_asec_prepared_source.py")
    path = root / "prepared-source" / fixture.prepared.MEMBER_FILENAMES[0][1]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(StoreMiss):
            _run(root, resume="require")
        with pytest.raises(NodeRejectedError, match="SOURCE_BYTES_MISMATCH"):
            _run(root, resume="forbid")
    finally:
        path.write_bytes(original)


def test_an_acs_source_directory_that_is_not_the_two_archives_refuses(executed, tmp_path):
    root, *_rest = executed
    broken = tmp_path / "acs-broken"
    shutil.copytree(root / "acs", broken)
    (broken / "extra.txt").write_bytes(b"not an archive")
    sources = _sources(root)
    sources[ACS_SOURCE] = broken
    with pytest.raises(NodeRejectedError, match="exactly the two archives"):
        _run(root, sources=sources)


def test_a_prepared_source_missing_an_evidence_member_refuses(executed, tmp_path):
    root, *_rest = executed
    fixture = _module("test_us_asec_prepared_source.py")
    broken = tmp_path / "prepared-broken"
    shutil.copytree(root / "prepared-source", broken)
    (broken / fixture.prepared.MEMBER_FILENAMES[0][1]).unlink()
    sources = _sources(root)
    sources[ASEC_SOURCE] = broken
    with pytest.raises(NodeRejectedError, match="SOURCE_FILE_ROSTER"):
        _run(root, sources=sources)


def test_relocated_identical_sources_reuse_the_required_cache(executed, tmp_path, monkeypatch):
    root, cold, _store, _calls, _direct, _contexts = executed
    sources = _sources(root)
    for name in (ASEC_SOURCE, ACS_SOURCE):
        moved = tmp_path / f"relocated-{name}"
        shutil.copytree(sources[name], moved)
        sources[name] = moved

    def forbidden(*args, **kwargs):
        raise AssertionError("identical relocated sources executed a kernel")

    for cls in KERNELS:
        monkeypatch.setattr(cls, "run", forbidden)
    warm, _store = _run(root, resume="require", sources=sources)
    assert warm.key == cold.key
    assert all(record.hit for record in warm.nodes.values())


def _structural_acs_frame():
    """An invented native-ACS-shaped arm with structure neither fixture produces.

    Household 3 holds four persons across **two** SPM units and two tax units,
    and its person rows are interleaved with household 7's rather than being
    contiguous or sorted. Both real fixtures put exactly one SPM unit in every
    household and order persons by household, so this is the only way to
    exercise reordered multi-person, multi-SPM assembly on real operators.
    Only the external ACS reader is replaced; every kernel, operator, artifact
    and store below it is the production one.
    """
    from microcosm.frame import Weights

    household_of = np.array([3, 7, 3, 3, 3], dtype=np.int64)
    spm_of = np.array([3, 7, 3, 30, 30], dtype=np.int64)
    tax_of = np.array([3, 7, 3, 30, 30], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.array([41, 12, 44, 43, 42], dtype=np.int64),
            "person_household_id": household_of,
            "person_spm_unit_id": spm_of,
            "person_tax_unit_id": tax_of,
            "person_family_id": household_of,
            "person_marital_unit_id": np.array([3, 7, 33, 30, 31], dtype=np.int64),
            "age": np.array([44.0, 61.0, 12.0, 29.0, 27.0]),
            "source_year": np.array([2024] * 5, dtype=np.int64),
            "PWGTP": np.array([9, 4, 9, 9, 9], dtype=np.int64),
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        column = US_SCHEMA.membership_column(group)
        tables[group] = pd.DataFrame(
            {US_SCHEMA.entity_id_column(group): np.unique(person[column].to_numpy())}
        )
    household = tables["household"]
    household["state_fips"] = np.array([6, 36], dtype=np.int64)
    household["TYPEHUGQ"] = np.array([1, 1], dtype=np.int64)
    household["SERIALNO"] = pd.Series(
        ["0000000000123", "0000000000045"], dtype=CANONICAL_STRING_DTYPE
    )
    household["puma"] = pd.Series(
        [f"06{NATIVE_PUMA}", f"36{NATIVE_PUMA}"], dtype=CANONICAL_STRING_DTYPE
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([13.0, 4.0]), WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr"] * len(person), dtype=object),
    )


@pytest.fixture(scope="module")
def structural(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-composed-structural")
    from microcosm.build.us_runtime import asec_student_controls as student

    with pytest.MonkeyPatch.context() as patch:
        _module("test_us_asec_prepared_source.py").invented_prepared_directory(root, patch)
        invented_acs_archives(root / "acs")
        (root / "invented-pins.json").write_bytes(
            canonical_json(
                {
                    "source": legacy._SOURCE_PINS,
                    "members": restoration._MEMBER_PINS,
                    "students": student._MEMBER_PINS,
                    "income": income_source._MEMBER_PINS,
                }
            )
        )
        arm = _structural_acs_frame()
        patch.setattr(
            composed, "load_graph_acs", lambda path, **kw: _structural_acs_frame()
        )
        direct = _direct(root)
        manifest, store = _run(root)
        yield root, manifest, store, arm, direct


def test_reordered_multi_spm_household_survives_assembly_intact(structural):
    _root, manifest, _store, arm, _direct_result = structural
    frame = manifest.population(composed.HARMONIZE_NODE)
    person = frame.person
    acs = person[person[support_channel_column("person")] == ACS_STACKED_SUPPORT_CHANNEL]
    # One assembled household really holds two SPM units, and two of the three
    # SPM units really hold more than one person.
    per_household = acs.groupby("person_household_id").person_spm_unit_id.nunique()
    assert sorted(per_household.tolist()) == [1, 2]
    assert sorted(acs.groupby("person_spm_unit_id").size().tolist()) == [1, 2, 2]
    assert sorted(acs.groupby("person_household_id").size().tolist()) == [1, 4]
    # Row order and the membership pattern are carried through unchanged; only
    # the collision-safe offset moves the identifiers.
    source = arm.person
    for column in (
        "person_id",
        *(US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities),
    ):
        actual = acs[column].to_numpy("int64")
        original = source[column].to_numpy("int64")
        offsets = set((actual - original).tolist())
        assert len(offsets) == 1, column
        np.testing.assert_array_equal(
            np.argsort(actual, kind="stable"), np.argsort(original, kind="stable")
        )
    np.testing.assert_array_equal(
        acs["age"].to_numpy(), source["age"].to_numpy()
    )
    # The ASEC arm has no age at all; absence stays absence.
    asec = person[person[support_channel_column("person")] == BASE_ASEC_SUPPORT_CHANNEL]
    assert asec["age"].isna().all()
    assert sorted(frame.table("household")["SERIALNO"].dropna().tolist()) == [
        "0000000000045",
        "0000000000123",
    ]


def test_structural_arm_geography_and_origin_mapping_stay_exact(structural):
    root, manifest, store, arm, _direct_result = structural
    document = composed.bind_composed_source_origin(
        _payload(manifest, store, composed.CREATE_NODE, "source_origin")
    )
    acs_arm = document["arms"][ACS_STACKED_SUPPORT_CHANNEL]
    assert acs_arm["rows"] == {entity: int(arm.n(entity)) for entity in arm.entities}
    assert acs_arm["household_mass"] == float(arm.weights_for("household").total)
    assert acs_arm["native_identity"]["household.SERIALNO"]["ordered_sha256"] == _sha(
        canonical_json(arm.table("household")["SERIALNO"].tolist())
    )
    assigned = manifest.population(f"{GEOGRAPHY_PHASE}.boundary")
    household = assigned.table("household")
    acs_rows = household[support_channel_column("household")] == ACS_STACKED_SUPPORT_CHANNEL
    assert sorted(household.loc[acs_rows, "puma"].tolist()) == [
        f"06{NATIVE_PUMA}",
        f"36{NATIVE_PUMA}",
    ]
    assert household["congressional_district_geoid"].gt(0).all()
    del root
