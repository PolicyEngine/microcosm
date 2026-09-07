"""Binding the carried prepared-ASEC evidence to the composed ASEC rows.

Every source byte is invented. The ASEC arm is the real ten-file prepared
directory read by the real preparation; the ACS arm is a real two-archive PUMS
pair read by the real ACS codec, or an invented native-shaped arm where a
structure neither archive can express is needed. Private source pins are
substituted in test scope only, including inside the fresh interpreters.

Nothing here certifies a population, a calibration or a release, and no engine
runs: the two measurement nodes are pure.
"""

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

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
from microcosm.build.us_runtime import graph_composed_asec_binding as binding
from microcosm.build.us_runtime import graph_composed_asec_measures as measures
from microcosm.build.us_runtime import graph_composed_population as composed
from microcosm.build.us_runtime.asec_current_money import _json, _sha
from microcosm.build.us_runtime.asec_current_money_selection import (
    US_ASEC_SELECTED_MONEY_TYPE,
    US_ASEC_SELECTION_TYPE,
    decode_selected_current_money,
    encode_selected_current_money,
    parse_current_money_body,
    select_current_money,
)
from microcosm.build.us_runtime.asec_prepared_source import (
    prepare_asec_current_money_population,
)
from microcosm.build.us_runtime.cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
    CPS_CARRIED_CURRENT_ROUTING_COLUMNS,
    CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    derive_cps_carried_current_leaves,
)
from microcosm.build.us_runtime.graph_asec_income import (
    ACCOUNTING_COLUMNS,
    EVIDENCE_COLUMNS,
    RESULT_COLUMNS,
    bind_income_observations,
    derive_reported_income,
    encode_reported_income,
)
from microcosm.build.us_runtime.graph_asec_prepared import (
    HOUSEHOLD_WEIGHT_CARRIER,
    SELECTION_STRATUM_COLUMN,
    PreparedGraphError,
    USAsecPreparedSelectionKernel,
)
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _row_identity,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_geography import GEOGRAPHY_PHASE
from microcosm.build.us_runtime.graph_housing_universe import (
    HOUSEHOLD_EVIDENCE_COLUMNS,
    bind_housing_universe,
    verify_graph_housing_rows,
)
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.build.us_runtime.stacked_spine import ACS_STACKED_SUPPORT_CHANNEL
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    ContentStore,
    Graph,
    KernelContext,
    Node,
    NodeRejectedError,
    NumericScope,
    Owned,
    Slice,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import storage_equal

SEED = 578
FRACTION = 1.0
BASE_VERSION = f"{GEOGRAPHY_PHASE}.boundary"
KERNELS = (
    composed.USComposedPopulationCreateKernel,
    binding.USComposedAsecBindKernel,
    measures.USComposedAsecLeavesKernel,
    measures.USComposedAsecReportedIncomeKernel,
)
ASEC_SOURCE = composed.ASEC_PREPARED_SOURCE_NAME
ACS_SOURCE = composed.ACS_NATIVE_SOURCE_NAME
#: Placeholders only for the direct oracle, which has no graph producer keys.
_ORACLE_DIGEST = _sha(b"microcosm.test.composed-asec-binding.oracle")


def _import_roots():
    """Where a process actually resolved the packages under test.

    A fresh interpreter that picked up a different checkout would replay a
    different implementation while still reporting cache hits, so every
    subprocess reports its roots and the tests compare them with this one's.
    """
    import microcosm.build
    import microcosm.frame
    import microcosm.graph

    return {
        name: str(Path(module.__file__).resolve().parent)
        for name, module in (
            ("build", microcosm.build),
            ("frame", microcosm.frame),
            ("graph", microcosm.graph),
        )
    }


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"bound_{name[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _composed_tests():
    return _module("test_us_graph_composed_population.py")


def _restore_test_pins(root):
    _composed_tests()._restore_test_pins(root)


def _columns(root):
    return tuple(
        Owned(*entry) for entry in json.loads((Path(root) / "columns.json").read_text())
    )


def _sources(root):
    return _composed_tests()._sources(root)


def _bound_graph(root):
    return measures.composed_asec_population_graph(
        _columns(root), sample_fraction=FRACTION, sample_seed=SEED
    )


def _run(root, *, graph=None, store_name="store", resume="forbid"):
    root = Path(root)
    store = ContentStore(root / store_name, codecs=us_source_codecs())
    manifest = run_graph(
        compile_graph(_bound_graph(root) if graph is None else graph),
        sources=_sources(root),
        store=store,
        kernels=measures.composed_asec_population_registry(),
        resume=resume,
    )
    return manifest, store


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _direct_composition(root):
    """Compose once with the real helpers and publish the declared inventory."""
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
    return {
        "pid": os.getpid(),
        "roots": _import_roots(),
        "person_rows": int(result.frame.n("person")),
    }


def _direct_binding(root):
    """Resolve and derive with the pure helpers alone, outside any executor.

    The graph's own declared inputs are handed to the pure functions: the
    composed population as a checkpoint and the seven typed payloads exactly as
    the store holds them. What the comparison then proves is that the three
    kernels add nothing to those functions — not one byte.
    """
    root = Path(root)
    producers = json.loads((root / "producers.json").read_text())
    keys = json.loads((root / "graph-keys.json").read_text())
    frame = load_frame_checkpoint(root / "bound-population.h5").frame
    payloads = {
        name: (root / f"graph-{name}.bin").read_bytes()
        for name in (
            "frame_context",
            "source_origin",
            "asec_frame_context",
            "current_money",
            "prepared_receipt",
            "housing_universe",
            "income_observations",
        )
    }
    bound = binding.resolve_composed_asec_binding(
        {entity: frame.table(entity) for entity in US_SCHEMA.entities},
        context_payload=payloads["frame_context"],
        origin_payload=payloads["source_origin"],
        prepared_context_payload=payloads["asec_frame_context"],
        money_payload=payloads["current_money"],
        receipt_payload=payloads["prepared_receipt"],
        housing_payload=payloads["housing_universe"],
        income_payload=payloads["income_observations"],
        producers=producers,
    )
    (root / "direct-binding.json").write_bytes(bound.payload)
    (root / "direct-selected.bin").write_bytes(bound.selected_money)
    (root / "direct-arm-rows.bin").write_bytes(
        binding.encode_composed_asec_arm_rows(
            bound.positions, bound.original_ids, binding_sha256=_sha(bound.payload)
        )
    )
    receipt = json.loads(payloads["prepared_receipt"])
    person = frame.table("person").loc[bound.masks["person"]]
    spm = frame.table("spm_unit").loc[bound.masks["spm_unit"]]
    selected = decode_selected_current_money(
        bound.selected_money,
        expected_parent_header_sha256=receipt["money_header_sha256"],
        expected_parent_content_sha256=receipt["money_content_sha256"],
        expected_prepared_receipt_sha256=_sha(payloads["prepared_receipt"]),
        expected_selection_sha256=_sha(bound.payload),
    )
    leaves = derive_cps_carried_current_leaves(
        selected,
        routing=person.loc[:, list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS)],
        spm_membership=person[US_SCHEMA.membership_column("spm_unit")].to_numpy(
            dtype="int64"
        ),
        spm_ids=spm[US_SCHEMA.entity_id_column("spm_unit")].to_numpy(dtype="int64"),
    )
    income = bind_income_observations(
        payloads["income_observations"], prepared_receipt=receipt
    )
    accounting = derive_reported_income(
        selected,
        income,
        person_ids=bound.original_ids["person"],
        income_years=bound.income_years,
        person_positions_sha256=_sha(bound.positions["person"].tobytes()),
        prepared_receipt_sha256=_sha(payloads["prepared_receipt"]),
        selection_sha256=_sha(bound.payload),
        selected_money_sha256=_sha(bound.selected_money),
        income_payload_sha256=_sha(payloads["income_observations"]),
        # The accounting node reads the corrected-leaf node's context and the
        # binding node's own producer key, so the oracle uses those, not the
        # population context the resolution itself was given.
        source_producer_key=keys["create"],
        selection_producer_key=keys["bind"],
        frame_context_sha256=_sha((root / "graph-leaves-context.bin").read_bytes()),
    )
    (root / "direct-reported-income.bin").write_bytes(
        encode_reported_income(accounting)
    )
    (root / "direct-leaves.json").write_bytes(
        canonical_json(
            {
                "person": {
                    name: values.tolist() for name, values in leaves.person.items()
                },
                "spm_unit": {
                    name: values.tolist() for name, values in leaves.spm_unit.items()
                },
                "person_ids": person[US_SCHEMA.entity_id_column("person")]
                .to_numpy(dtype="int64")
                .tolist(),
                "spm_ids": spm[US_SCHEMA.entity_id_column("spm_unit")]
                .to_numpy(dtype="int64")
                .tolist(),
            }
        )
    )
    return {
        "pid": os.getpid(),
        "roots": _import_roots(),
        "arm_person_rows": int(bound.masks["person"].sum()),
    }


def _cache_only(root):
    """Replay from the store alone; any kernel or source read is a failure."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("cache-only replay executed a kernel or source loader")

    for cls in (
        *KERNELS,
        *_composed_tests().KERNELS,
    ):
        cls.run = forbidden
    composed.prepare_asec_current_money_population = forbidden
    composed.load_graph_acs = forbidden
    composed.compose_from_sources = forbidden
    binding.resolve_composed_asec_binding = forbidden
    measures.derive_cps_carried_current_leaves = forbidden
    measures.derive_reported_income = forbidden
    manifest, store = _run(root, resume="require")
    return {
        "pid": os.getpid(),
        "roots": _import_roots(),
        "key": manifest.key,
        "nodes": {name: record.key for name, record in manifest.nodes.items()},
        "all_hits": all(record.hit for record in manifest.nodes.values()),
        "kernel_calls": len(calls),
        "binding_sha256": _sha(
            _payload(manifest, store, binding.BIND_NODE, "asec_binding")
        ),
        "reported_income_sha256": _sha(
            _payload(manifest, store, measures.REPORTED_INCOME_NODE, "reported_income")
        ),
        "leaves_context_sha256": _sha(
            _payload(manifest, store, measures.LEAVES_NODE, "frame_context")
        ),
    }


def _process(root, mode):
    code = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("bound_graph_process", sys.argv[1])
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


def _write_pins(root):
    from microcosm.build.us_runtime import asec_student_controls as student

    (Path(root) / "invented-pins.json").write_bytes(
        canonical_json(
            {
                "source": legacy._SOURCE_PINS,
                "members": restoration._MEMBER_PINS,
                "students": student._MEMBER_PINS,
                "income": income_source._MEMBER_PINS,
            }
        )
    )


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-composed-asec-binding")
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
        _composed_tests().invented_acs_archives(root / "acs")
        _write_pins(root)
        _process(root, "_direct_composition")
        calls = []
        for cls in KERNELS:
            original = cls.run

            def capture(self, context, *, _run=original):
                calls.append(context.node.id)
                return _run(self, context)

            patch.setattr(cls, "run", capture)
        manifest, store = _run(root)
        bound_calls = list(calls)
        # The unbound composed population, run from the same sources into its
        # own store: the reference every ACS-preservation claim is made against.
        baseline_manifest, baseline_store = _run(
            root,
            graph=composed.composed_population_graph(
                _columns(root), sample_fraction=FRACTION, sample_seed=SEED
            ),
            store_name="store-baseline",
        )
        (root / "producers.json").write_bytes(
            canonical_json(
                {
                    "population": manifest.nodes[GEOGRAPHY_PHASE].key,
                    "prepared_source": manifest.nodes[composed.CREATE_NODE].key,
                    "source_origin": manifest.nodes[composed.CREATE_NODE].key,
                }
            )
        )
        (root / "graph-keys.json").write_bytes(
            canonical_json(
                {
                    "bind": manifest.nodes[binding.BIND_NODE].key,
                    "create": manifest.nodes[composed.CREATE_NODE].key,
                }
            )
        )
        (root / "graph-leaves-context.bin").write_bytes(
            _payload(manifest, store, measures.LEAVES_NODE, "frame_context")
        )
        write_frame_checkpoint(root / "bound-population.h5", _bound_frame(manifest))
        (root / "graph-frame_context.bin").write_bytes(
            _payload(manifest, store, GEOGRAPHY_PHASE, "frame_context")
        )
        for name in (
            "source_origin",
            "asec_frame_context",
            "current_money",
            "prepared_receipt",
            "housing_universe",
            "income_observations",
        ):
            (root / f"graph-{name}.bin").write_bytes(
                _payload(manifest, store, composed.CREATE_NODE, name)
            )
        direct = _process(root, "_direct_binding")
        yield root, manifest, store, bound_calls, direct, baseline_manifest
        del baseline_store


def _bound_frame(manifest):
    return manifest.population(BASE_VERSION)


def _arm(frame, entity):
    table = frame.table(entity)
    return table[table[support_channel_column(entity)] == BASE_ASEC_SUPPORT_CHANNEL]


def _other_arm(frame, entity):
    table = frame.table(entity)
    return table[table[support_channel_column(entity)] == ACS_STACKED_SUPPORT_CHANNEL]


def test_the_accepted_prepared_kernels_refuse_this_population(executed):
    """The real refusal, on the real composed population, before anything new.

    The prepared slice's selection kernel is declared verbatim over the composed
    population with the composed CREATE's carried artifacts. It refuses at its
    first contract: a composed context is not an empty one. The layers behind
    that first refusal are exercised directly below, so fixing one would not
    have admitted the kernel.
    """
    root, *_rest = executed
    nodes = composed.composed_population_nodes(
        _columns(root), sample_fraction=FRACTION, sample_seed=SEED
    )
    carried = binding.bind_node_artifact_inputs(population_context=composed.CREATE_NODE)
    select = Node(
        id="composed_probe.select_households",
        kernel=USAsecPreparedSelectionKernel.ref,
        base=composed.HARMONIZE_NODE,
        structural=StructuralDelta.FILTER,
        mass="declared",
        inputs=(
            Slice("person", (SELECTION_STRATUM_COLUMN,)),
            Slice("household", (HOUSEHOLD_WEIGHT_CARRIER, *HOUSEHOLD_EVIDENCE_COLUMNS)),
        ),
        params={
            "phase": "prepare_asec_current_money_slice",
            "fraction": FRACTION,
            "seed": SEED,
        },
        artifact_inputs=tuple(
            edge
            for edge in carried
            if edge.name
            in {
                "frame_context",
                "current_money",
                "prepared_receipt",
                "housing_universe",
            }
        ),
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("selection", US_ASEC_SELECTION_TYPE),
            ArtifactOutput("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
        ),
    )
    registry = composed.composed_population_registry(geography=False)
    registry.register(USAsecPreparedSelectionKernel())
    store = ContentStore(Path(root) / "store-refusal", codecs=us_source_codecs())
    with pytest.raises(NodeRejectedError, match="PREPARED_CONTEXT_METADATA"):
        run_graph(
            compile_graph(
                Graph(
                    country="us",
                    sources=composed.COMPOSED_SOURCES,
                    nodes=(*nodes, select),
                )
            ),
            sources={
                ASEC_SOURCE: Path(root) / "prepared-source",
                ACS_SOURCE: Path(root) / "acs",
            },
            store=store,
            kernels=registry,
            resume="forbid",
        )


def test_the_layers_behind_that_refusal_are_real_too(executed):
    """Each further prepared-slice contract the composed population also fails."""
    root, manifest, store, *_rest = executed
    frame = _bound_frame(manifest)
    receipt = json.loads(
        _payload(manifest, store, composed.CREATE_NODE, "prepared_receipt")
    )
    context = json.loads(_payload(manifest, store, GEOGRAPHY_PHASE, "frame_context"))
    # The prepared receipt describes the ASEC arm, not the composed population.
    assert receipt["entity_rows"]["person"] < context["entities"]["person"]["rows"]
    body = parse_current_money_body(
        _payload(manifest, store, composed.CREATE_NODE, "current_money"),
        expected_header_sha256=receipt["money_header_sha256"],
        expected_content_sha256=receipt["money_content_sha256"],
        field_entities=tuple(tuple(item) for item in receipt["field_entities"]),
    )
    assert body.person_rows != context["entities"]["person"]["rows"]
    # The housing verifier cannot be pointed at the composed household table:
    # its positions must cover exactly the rows it is given.
    universe = bind_housing_universe(
        _payload(manifest, store, composed.CREATE_NODE, "housing_universe"),
        prepared_receipt=receipt,
    )
    household = frame.table("household")
    with pytest.raises(Exception, match="BOUND_HU_POSITIONS"):
        verify_graph_housing_rows(
            household,
            universe,
            positions=np.arange(len(household), dtype="int64"),
        )
    # And the composed row identifiers resolve to nothing in the source buffers.
    income = bind_income_observations(
        _payload(manifest, store, composed.CREATE_NODE, "income_observations"),
        prepared_receipt=receipt,
    )
    composed_ids = frame.table("person")["person_id"].to_numpy(dtype="int64")
    assert (pd.Index(income.array("person_id")).get_indexer(composed_ids) < 0).any()
    del root


def test_cold_graph_matches_the_direct_binding(executed):
    """The graph and the pure helpers produce the same bytes, not merely values."""
    root, manifest, store, calls, direct, _baseline = executed
    assert calls == [
        composed.CREATE_NODE,
        binding.BIND_NODE,
        measures.LEAVES_NODE,
        measures.REPORTED_INCOME_NODE,
    ]
    assert direct["pid"] != os.getpid()
    # The direct oracle ran against this checkout, not a shared install.
    assert direct["roots"] == _import_roots()
    for name, node, artifact in (
        ("direct-binding.json", binding.BIND_NODE, "asec_binding"),
        ("direct-arm-rows.bin", binding.BIND_NODE, "arm_rows"),
        ("direct-selected.bin", binding.BIND_NODE, "selected_current_money"),
        (
            "direct-reported-income.bin",
            measures.REPORTED_INCOME_NODE,
            "reported_income",
        ),
    ):
        assert (
            _payload(manifest, store, node, artifact)
            == (Path(root) / name).read_bytes()
        ), name
    frame = _bound_frame(manifest)
    leaves = json.loads((Path(root) / "direct-leaves.json").read_text())
    person = _arm(frame, "person")
    spm = _arm(frame, "spm_unit")
    assert person["person_id"].tolist() == leaves["person_ids"]
    assert spm["spm_unit_id"].tolist() == leaves["spm_ids"]
    for name in CPS_CARRIED_CURRENT_PERSON_LEAVES:
        np.testing.assert_array_equal(
            person[name].to_numpy(), np.asarray(leaves["person"][name])
        )
    for name in CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES:
        np.testing.assert_array_equal(
            spm[name].to_numpy(), np.asarray(leaves["spm_unit"][name])
        )


def test_every_native_acs_cell_and_absence_survives_byte_for_byte(executed):
    """The unbound composed population is the reference; the ACS arm must equal it.

    Every column the baseline carries is compared with ``storage_equal`` on the
    ACS rows, so a changed value, a changed dtype and a changed null mask all
    fail. Group quarters are included: the GQ household and its person are ACS
    rows like any other.
    """
    _root, manifest, _store, _calls, _direct, baseline = executed
    bound = _bound_frame(manifest)
    reference = baseline.population(BASE_VERSION)
    for entity in US_SCHEMA.entities:
        before = reference.table(entity)
        after = bound.table(entity)
        assert len(before) == len(after)
        mask = (
            after[support_channel_column(entity)] == ACS_STACKED_SUPPORT_CHANNEL
        ).to_numpy(dtype=bool)
        assert mask.any()
        for column in before.columns:
            assert column in after, f"{entity}.{column} was dropped"
            assert storage_equal(before[column], after[column], mask), (
                f"{entity}.{column} changed on the native ACS arm"
            )
        added = [name for name in after.columns if name not in set(before.columns)]
        for column in added:
            if entity in binding.BOUND_ENTITIES and column == binding.arm_row_column(
                entity
            ):
                # The arm-membership cell is the one declared whole-population
                # write, and it says exactly "not this arm" on every ACS row.
                assert not after.loc[mask, column].any()
                continue
            assert after.loc[mask, column].isna().all(), (
                f"{entity}.{column} was filled on the native ACS arm"
            )
    # The invented ACS arm really does carry a group-quarters household, and it
    # is still there, still ACS, still unmeasured.
    household = bound.table("household")
    gq = household["TYPEHUGQ"] == 2
    assert int(gq.sum()) == 1
    assert (
        household.loc[gq, support_channel_column("household")]
        == ACS_STACKED_SUPPORT_CHANNEL
    ).all()
    assert not household.loc[gq, binding.arm_row_column("household")].any()


def test_the_measurements_are_absent_not_zero_on_the_other_arm(executed):
    """A new measured column is a declared null off the arm, never a zero."""
    _root, manifest, *_rest = executed
    frame = _bound_frame(manifest)
    person = _other_arm(frame, "person")
    for name in RESULT_COLUMNS:
        assert person[name].isna().all()
    spm = _other_arm(frame, "spm_unit")
    for name in CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES:
        assert spm[name].isna().all()
    # A leaf the native ACS arm already carries keeps the ACS value instead.
    for name in measures.ACS_SHARED_LEAVES:
        assert not _other_arm(frame, "person")[name].isna().all()


def test_leaves_and_accounting_match_the_standalone_prepared_derivation(executed):
    """The arithmetic is the accepted one: composition changed no number.

    The oracle never composes. It prepares the ASEC population on its own,
    selects every row with the reviewed selector and runs the same two reviewed
    derivations, then the results are matched by the arm's original person id.
    """
    root, manifest, store, *_rest = executed
    _restore_test_pins(root)
    prepared = prepare_asec_current_money_population(Path(root) / "prepared-source")
    receipt = prepared.receipt
    body = parse_current_money_body(
        prepared.money_payload,
        expected_header_sha256=receipt["money_header_sha256"],
        expected_content_sha256=receipt["money_content_sha256"],
        field_entities=tuple(tuple(item) for item in receipt["field_entities"]),
    )
    person = prepared.frame.table("person")
    spm = prepared.frame.table("spm_unit")
    person_positions = np.arange(len(person), dtype="int64")
    household_positions = np.arange(
        len(prepared.frame.table("household")), dtype="int64"
    )
    selected = select_current_money(
        body,
        person_positions=person_positions,
        household_positions=household_positions,
        prepared_receipt_sha256=_sha(prepared.receipt_payload),
        selection_sha256=_ORACLE_DIGEST,
        person_identity_sha256=_row_identity(person, "person")["ordered_ids_sha256"],
        household_identity_sha256=_row_identity(
            prepared.frame.table("household"), "household"
        )["ordered_ids_sha256"],
    )
    leaves = derive_cps_carried_current_leaves(
        selected,
        routing=person.loc[:, list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS)],
        spm_membership=person[US_SCHEMA.membership_column("spm_unit")].to_numpy(
            dtype="int64"
        ),
        spm_ids=spm[US_SCHEMA.entity_id_column("spm_unit")].to_numpy(dtype="int64"),
    )
    frame = _bound_frame(manifest)
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    rows = binding.bind_composed_asec_arm_rows(
        _payload(manifest, store, binding.BIND_NODE, "arm_rows"),
        binding_document=document,
    )
    order = pd.Index(person["person_id"].to_numpy(dtype="int64")).get_indexer(
        rows.array("person", "original_ids")
    )
    assert (order >= 0).all()
    arm_person = _arm(frame, "person")
    for name in CPS_CARRIED_CURRENT_PERSON_LEAVES:
        np.testing.assert_array_equal(
            arm_person[name].to_numpy(), leaves.person[name][order]
        )
    income = bind_income_observations(
        prepared.income_observations_payload, prepared_receipt=receipt
    )
    accounting = derive_reported_income(
        selected,
        income,
        person_ids=person["person_id"].to_numpy(dtype="int64"),
        income_years=person["source_year"].to_numpy(dtype="int64"),
        person_positions_sha256=_sha(person_positions.tobytes()),
        prepared_receipt_sha256=_sha(prepared.receipt_payload),
        selection_sha256=_ORACLE_DIGEST,
        selected_money_sha256=_sha(encode_selected_current_money(selected)),
        income_payload_sha256=_sha(prepared.income_observations_payload),
        source_producer_key=_ORACLE_DIGEST,
        selection_producer_key=_ORACLE_DIGEST,
        frame_context_sha256=_ORACLE_DIGEST,
    )
    for name in RESULT_COLUMNS:
        np.testing.assert_array_equal(
            arm_person[name].to_numpy(), accounting.array(name)[order]
        )
    # The raw observation axes travel unchanged: quality, zero-origin and
    # allocation stay separate from the computed quantities.
    np.testing.assert_array_equal(
        accounting.array("A_AGE")[order],
        income.array("A_AGE")[rows.array("person", "source_positions")],
    )
    # The observation axes ride in the artifact, byte for byte, and none of
    # them is written back as a population cell.
    graph_accounting = _payload(
        manifest, store, measures.REPORTED_INCOME_NODE, "reported_income"
    )
    header = json.loads(
        graph_accounting[
            len(b"MCAINACC\x03") + 4 : len(b"MCAINACC\x03")
            + 4
            + int.from_bytes(
                graph_accounting[len(b"MCAINACC\x03") : len(b"MCAINACC\x03") + 4],
                "little",
            )
        ]
    )
    assert [item["name"] for item in header["columns"]] == list(ACCOUNTING_COLUMNS)
    assert set(EVIDENCE_COLUMNS) <= set(ACCOUNTING_COLUMNS)
    person_columns = set(_bound_frame(manifest).table("person").columns)
    assert not person_columns & set(EVIDENCE_COLUMNS)
    for axis in EVIDENCE_COLUMNS:
        assert len(accounting.array(axis)) == len(order)


def test_three_income_cohorts_bind_with_their_observed_periods(executed):
    """Three restored cohorts, each with the period its source actually declares."""
    _root, manifest, store, *_rest = executed
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    assert sorted(document["cohorts"]) == ["2022", "2023", "2024"]
    assert all(count > 0 for count in document["cohorts"].values())
    age = next(
        item for item in document["demographics"]["bound"] if item["column"] == "age"
    )
    assert age["asec_source_column"] == "A_AGE"
    assert age["aged_from_a_date"] is False
    assert age["cross_arm_period_harmonized"] is False
    assert age["observed_period_by_cohort"] == {
        year: {"income_year": int(year), "asec_survey_year": int(year) + 1}
        for year in ("2022", "2023", "2024")
    }
    unbound = {item["column"]: item for item in document["demographics"]["unbound"]}
    assert set(unbound) == {"is_female", "is_household_head"}
    assert unbound["is_female"]["required_asec_source_columns"] == ["A_SEX", "AXSEX"]
    assert unbound["is_female"]["present_in_prepared_arm"] == []
    frame = _bound_frame(manifest)
    # The gap is a fact about this population, not a claim: the arm really has
    # no sex or head cell, and the binding filled neither.
    assert _arm(frame, "person")["is_female"].isna().all()
    assert _arm(frame, "person")["is_household_head"].isna().all()


def test_bound_age_equals_the_source_observation_and_leaves_acs_age_alone(executed):
    """``age`` is the arm's own ``A_AGE``; the ACS arm's ``age`` is untouched."""
    _root, manifest, _store, _calls, _direct, baseline = executed
    frame = _bound_frame(manifest)
    arm = _arm(frame, "person")
    np.testing.assert_array_equal(
        arm["age"].to_numpy(), arm["A_AGE"].to_numpy(dtype="int64").astype("float64")
    )
    assert not arm["age"].isna().any()
    reference = baseline.population(BASE_VERSION).table("person")
    assert reference["age"].isna().sum() == len(arm)
    acs = _other_arm(frame, "person")
    mask = (
        frame.table("person")[support_channel_column("person")]
        == ACS_STACKED_SUPPORT_CHANNEL
    ).to_numpy(dtype=bool)
    assert storage_equal(reference["age"], frame.table("person")["age"], mask)
    del acs


def test_negative_zero_and_absent_source_amounts_survive_unchanged(executed):
    """Signed losses stay signed, zeros stay zero, and nothing absent is invented."""
    _root, manifest, *_rest = executed
    arm = _arm(_bound_frame(manifest), "person")
    losses = arm["self_employment_income_before_lsr"].to_numpy()
    assert (losses < 0).any(), "the invented arm should carry a signed loss"
    assert (arm["farm_operations_income"].to_numpy() <= 0).any()
    assert (arm["asec_reported_total_income_2024_price"].to_numpy() < 0).any()
    zeros = arm["social_security_survivors"].to_numpy()
    assert (zeros == 0).any()
    for name in (*CPS_CARRIED_CURRENT_PERSON_LEAVES, *RESULT_COLUMNS):
        assert np.isfinite(arm[name].to_numpy()).all(), name


def test_a_fraction_below_one_binds_only_the_drawn_rows(executed):
    """A partial draw binds fewer rows and says so; the evidence still is whole."""
    root, *_rest = executed
    _restore_test_pins(root)
    half = composed.compose_from_sources(
        Path(root) / "prepared-source",
        Path(root) / "acs",
        sample_fraction=0.5,
        sample_seed=SEED,
    )
    frame = half.frame
    bound = binding.resolve_composed_asec_binding(
        {entity: frame.table(entity) for entity in US_SCHEMA.entities},
        context_payload=encode_us_frame_context(frame),
        origin_payload=canonical_json(half.source_origin),
        prepared_context_payload=half.asec_frame_context,
        money_payload=half.money_payload,
        receipt_payload=half.receipt_payload,
        housing_payload=half.housing_universe_payload,
        income_payload=half.income_observations_payload,
        producers={
            "population": _ORACLE_DIGEST,
            "prepared_source": _ORACLE_DIGEST,
            "source_origin": _ORACLE_DIGEST,
        },
    )
    document = binding.bind_composed_asec_document(bound.payload)
    assert document["composed_is_whole_arm"] is False
    assert document["arm"]["person"]["rows"] < document["prepared_arm_rows"]["person"]
    assert document["origin"]["sample_fraction"] == 0.5
    # The source positions are a real subset of the whole arm, resolved by id.
    positions = bound.positions["person"]
    assert len(positions) == document["arm"]["person"]["rows"]
    assert positions.max() < document["prepared_arm_rows"]["person"]
    assert not np.array_equal(positions, np.arange(len(positions)))


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("receipt", "PREPARED_EVIDENCE_ROWS:person"),
        ("origin", "ORIGIN_CANONICAL"),
        ("context", "COMPOSED_CONTEXT_CANONICAL"),
        ("prepared_context", "PREPARED_CONTEXT_METADATA"),
        ("money", "PAYLOAD_CHECKSUM"),
        ("housing", "BOUND_HU_PAYLOAD"),
        ("income", "BOUND_INCOME_PAYLOAD"),
    ],
)
def test_a_mismatched_input_refuses(executed, mutation, reason):
    """Every consumed artifact is bound to its producer's own receipt."""
    root, manifest, store, *_rest = executed
    _restore_test_pins(root)
    frame = _bound_frame(manifest)
    payloads = {
        "context": _payload(manifest, store, GEOGRAPHY_PHASE, "frame_context"),
        "origin": _payload(manifest, store, composed.CREATE_NODE, "source_origin"),
        "prepared_context": _payload(
            manifest, store, composed.CREATE_NODE, "asec_frame_context"
        ),
        "money": _payload(manifest, store, composed.CREATE_NODE, "current_money"),
        "receipt": _payload(manifest, store, composed.CREATE_NODE, "prepared_receipt"),
        "housing": _payload(manifest, store, composed.CREATE_NODE, "housing_universe"),
        "income": _payload(
            manifest, store, composed.CREATE_NODE, "income_observations"
        ),
    }
    if mutation in ("context", "origin", "prepared_context", "receipt"):
        document = json.loads(payloads[mutation])
        if mutation == "receipt":
            document["entity_rows"]["person"] += 1
            payloads[mutation] = canonical_json(document)
        elif mutation == "prepared_context":
            # The composed population's own context is not the prepared arm's.
            payloads[mutation] = payloads["context"]
        else:
            payloads[mutation] = json.dumps(document).encode()
    else:
        flipped = bytearray(payloads[mutation])
        flipped[-33] ^= 0x01
        payloads[mutation] = bytes(flipped)
    with pytest.raises(Exception, match=reason):
        binding.resolve_composed_asec_binding(
            {entity: frame.table(entity) for entity in US_SCHEMA.entities},
            context_payload=payloads["context"],
            origin_payload=payloads["origin"],
            prepared_context_payload=payloads["prepared_context"],
            money_payload=payloads["money"],
            receipt_payload=payloads["receipt"],
            housing_payload=payloads["housing"],
            income_payload=payloads["income"],
            producers={
                "population": _ORACLE_DIGEST,
                "prepared_source": _ORACLE_DIGEST,
                "source_origin": _ORACLE_DIGEST,
            },
        )


def test_an_origin_document_from_another_population_refuses(executed):
    """The origin's provenance digests are recomputed against the actual rows."""
    root, manifest, store, *_rest = executed
    _restore_test_pins(root)
    other = composed.compose_from_sources(
        Path(root) / "prepared-source",
        Path(root) / "acs",
        sample_fraction=0.5,
        sample_seed=SEED,
    )
    frame = _bound_frame(manifest)
    with pytest.raises(PreparedGraphError, match="ORIGIN_DOES_NOT_DESCRIBE_THIS_FRAME"):
        binding.resolve_composed_asec_binding(
            {entity: frame.table(entity) for entity in US_SCHEMA.entities},
            context_payload=_payload(manifest, store, GEOGRAPHY_PHASE, "frame_context"),
            origin_payload=canonical_json(other.source_origin),
            prepared_context_payload=_payload(
                manifest, store, composed.CREATE_NODE, "asec_frame_context"
            ),
            money_payload=_payload(
                manifest, store, composed.CREATE_NODE, "current_money"
            ),
            receipt_payload=_payload(
                manifest, store, composed.CREATE_NODE, "prepared_receipt"
            ),
            housing_payload=_payload(
                manifest, store, composed.CREATE_NODE, "housing_universe"
            ),
            income_payload=_payload(
                manifest, store, composed.CREATE_NODE, "income_observations"
            ),
            producers={
                "population": _ORACLE_DIGEST,
                "prepared_source": _ORACLE_DIGEST,
                "source_origin": _ORACLE_DIGEST,
            },
        )


def test_the_bind_node_refuses_evidence_from_another_producer(executed):
    """The five positional evidence edges must all come from the one CREATE."""
    root, *_rest = executed
    graph = _bound_graph(root)
    bind = next(node for node in graph.nodes if node.id == binding.BIND_NODE)
    rerouted = tuple(
        edge
        if edge.name != "asec_frame_context"
        else ArtifactInput(
            edge.name, GEOGRAPHY_PHASE, "frame_context", US_FRAME_CONTEXT_TYPE
        )
        for edge in bind.artifact_inputs
    )
    replaced = Node(
        id=bind.id,
        kernel=bind.kernel,
        population=bind.population,
        inputs=bind.inputs,
        outputs=bind.outputs,
        params=dict(bind.params),
        artifact_inputs=rerouted,
        artifact_outputs=bind.artifact_outputs,
    )
    nodes = tuple(node for node in graph.nodes if node.id != bind.id) + (replaced,)
    store = ContentStore(Path(root) / "store-producer", codecs=us_source_codecs())
    with pytest.raises(NodeRejectedError, match="NODE_EVIDENCE_PRODUCER"):
        run_graph(
            compile_graph(Graph(country="us", sources=graph.sources, nodes=nodes)),
            sources=_sources(root),
            store=store,
            kernels=measures.composed_asec_population_registry(),
            resume="forbid",
        )


def _create_artifacts(manifest, store, *, origin_producer=None):
    """The bind node's seven typed edges as the executor would hand them over.

    Every payload is the one the graph actually stored, and every producer key
    is the producing node's own key, so nothing about the identities under test
    is invented. ``origin_producer`` substitutes the origin's producing
    execution while leaving its bytes alone — the case the origin's own content
    recomputation cannot see.
    """
    keys = {
        composed.CREATE_NODE: manifest.nodes[composed.CREATE_NODE].key,
        GEOGRAPHY_PHASE: manifest.nodes[GEOGRAPHY_PHASE].key,
    }
    values = {}
    for name, kind in binding._ARTIFACT_TYPES:
        node = GEOGRAPHY_PHASE if name == "frame_context" else composed.CREATE_NODE
        producer = keys[node]
        if name == "source_origin" and origin_producer is not None:
            producer = origin_producer
        payload = _payload(manifest, store, node, name)
        values[name] = ArtifactValue(
            payload=payload,
            type=kind,
            key=_sha(payload),
            producer_key=producer,
            numerics=NumericScope(),
        )
    return values


def _bind_context(root, manifest, store, *, node=None, origin_producer=None):
    """A kernel context over the real bound population and the real payloads."""
    bind = node
    if bind is None:
        bind = next(
            item for item in _bound_graph(root).nodes if item.id == binding.BIND_NODE
        )
    frame = _bound_frame(manifest)
    return KernelContext(
        node=bind,
        tables={entity: frame.table(entity) for entity in US_SCHEMA.entities},
        weights={},
        strata=pd.Series(dtype="int64"),
        params=dict(bind.params),
        rng=np.random.default_rng(SEED),
        artifacts=_create_artifacts(manifest, store, origin_producer=origin_producer),
    )


def test_the_origin_must_come_from_the_same_create_execution(executed):
    """``source_origin`` is positional over one CREATE execution, like the rest.

    Recomputing the origin's per-entity digests against the actual tables
    catches a document describing a different frame, but two CREATE executions
    that describe identical tables recompute identically. The producer identity
    the binding document records is therefore pinned, not carried unverified.
    """
    root, manifest, store, *_rest = executed
    kernel = binding.USComposedAsecBindKernel()

    # Positive control: with all six CREATE edges on one execution the kernel
    # reproduces the graph's own binding byte for byte.
    result = kernel.run(_bind_context(root, manifest, store))
    assert result.artifacts["asec_binding"] == _payload(
        manifest, store, binding.BIND_NODE, "asec_binding"
    )

    # Same origin bytes, a different producing execution.
    with pytest.raises(PreparedGraphError, match="ARTIFACT_PRODUCER_SPLIT"):
        kernel.run(_bind_context(root, manifest, store, origin_producer=_ORACLE_DIGEST))

    # And an origin edge declared from another producer entirely.
    bind = next(
        item for item in _bound_graph(root).nodes if item.id == binding.BIND_NODE
    )
    rerouted = Node(
        id=bind.id,
        kernel=bind.kernel,
        population=bind.population,
        inputs=bind.inputs,
        outputs=bind.outputs,
        params=dict(bind.params),
        artifact_inputs=tuple(
            edge
            if edge.name != "source_origin"
            else ArtifactInput(edge.name, GEOGRAPHY_PHASE, edge.artifact, edge.type)
            for edge in bind.artifact_inputs
        ),
        artifact_outputs=bind.artifact_outputs,
    )
    with pytest.raises(PreparedGraphError, match="NODE_EVIDENCE_PRODUCER"):
        kernel.run(_bind_context(root, manifest, store, node=rerouted))


def test_a_prepared_arm_carrying_the_demographic_sources_binds_and_records_them(
    executed,
):
    """A prepared roster carrying the declared sources is recorded, not refused.

    Presence is an observation about the population in hand and binds nothing,
    so the stage must record it and keep going — refusing on it would stop
    every genuine run while proving nothing about what this stage writes. The
    diagnostic-only ``P_SEQ`` is recorded on its own axis and never counts
    towards the requirement.
    """
    root, manifest, store, *_rest = executed
    frame = _bound_frame(manifest)
    prepared = json.loads(
        _payload(manifest, store, composed.CREATE_NODE, "asec_frame_context")
    )
    columns = prepared["entities"]["person"]["columns"]
    declared = ("A_SEX", "AXSEX", "P_SEQ")
    # A_EXPRRP is already on this roster: microunit requires it, so asec_pool
    # supplies it — derived from A_LINENO == 1 for a cohort whose locked input
    # omits it. That is exactly why presence carries a caveat here.
    assert not set(declared) & set(columns)
    assert "A_EXPRRP" in columns
    prepared["entities"]["person"]["columns"] = [*columns, *declared]
    bound = binding.resolve_composed_asec_binding(
        {entity: frame.table(entity) for entity in US_SCHEMA.entities},
        context_payload=_payload(manifest, store, GEOGRAPHY_PHASE, "frame_context"),
        origin_payload=_payload(manifest, store, composed.CREATE_NODE, "source_origin"),
        prepared_context_payload=canonical_json(prepared),
        money_payload=_payload(manifest, store, composed.CREATE_NODE, "current_money"),
        receipt_payload=_payload(
            manifest, store, composed.CREATE_NODE, "prepared_receipt"
        ),
        housing_payload=_payload(
            manifest, store, composed.CREATE_NODE, "housing_universe"
        ),
        income_payload=_payload(
            manifest, store, composed.CREATE_NODE, "income_observations"
        ),
        producers={
            "population": _ORACLE_DIGEST,
            "prepared_source": _ORACLE_DIGEST,
            "source_origin": _ORACLE_DIGEST,
        },
    )
    # The typed document admits it: the unbound entries are held to being
    # unbound and complete, not to the source lacking the column.
    document = binding.bind_composed_asec_document(bound.payload)
    unbound = {item["column"]: item for item in document["demographics"]["unbound"]}
    assert set(unbound) == set(binding.UNBOUND_DEMOGRAPHIC_COLUMNS)
    assert unbound["is_female"]["present_in_prepared_arm"] == ["AXSEX", "A_SEX"]
    assert unbound["is_female"]["absent_from_prepared_arm"] == []
    assert unbound["is_female"]["diagnostic_only_source_columns"] == []
    assert unbound["is_household_head"]["present_in_prepared_arm"] == ["A_EXPRRP"]
    assert unbound["is_household_head"]["absent_from_prepared_arm"] == []
    assert all(
        item["presence_certifies_observation"] is False and item["presence_caveats"]
        for item in unbound.values()
    )
    # P_SEQ is present, and it is recorded on the diagnostic axis only: it is
    # not one of the required columns and claims no proof.
    assert unbound["is_household_head"]["diagnostic_only_source_columns"] == ["P_SEQ"]
    assert unbound["is_household_head"]["diagnostic_only_present_in_prepared_arm"] == [
        "P_SEQ"
    ]
    assert all(
        item["diagnostic_only_is_semantic_proof"] is False for item in unbound.values()
    )
    assert all(item["bound_by_this_stage"] is False for item in unbound.values())
    # Nothing was bound from them: the node still owns exactly three booleans
    # and the only bound demographic is still age.
    assert [item["column"] for item in document["demographics"]["bound"]] == ["age"]
    assert {owned.column for owned in binding.arm_row_declarations()} == {
        binding.arm_row_column(entity) for entity in binding.BOUND_ENTITIES
    }
    del root


def test_the_unbound_demographic_presence_is_recorded_without_claiming_observation(
    executed,
):
    """On this invented parent the same fields are recorded absent, not present."""
    _root, manifest, store, *_rest = executed
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    unbound = {item["column"]: item for item in document["demographics"]["unbound"]}
    assert unbound["is_female"]["present_in_prepared_arm"] == []
    assert unbound["is_female"]["absent_from_prepared_arm"] == ["AXSEX", "A_SEX"]
    # A_EXPRRP is present because asec_pool derives it, not because the invented
    # source observed it; the entry records presence and its caveat, not proof.
    assert unbound["is_household_head"]["present_in_prepared_arm"] == ["A_EXPRRP"]
    assert unbound["is_household_head"]["absent_from_prepared_arm"] == []
    assert unbound["is_household_head"]["presence_certifies_observation"] is False
    assert unbound["is_household_head"]["diagnostic_only_present_in_prepared_arm"] == []
    assert all(item["bound_by_this_stage"] is False for item in unbound.values())


@pytest.mark.parametrize(
    "mutation",
    ["sex_readset", "head_readset", "contract", "acs_gq", "crosscheck_claim"],
)
def test_unbound_demographic_contract_cannot_be_rewritten_by_a_binding_payload(
    executed, mutation
):
    _root, manifest, store, *_rest = executed
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    sex, head = document["demographics"]["unbound"]
    if mutation == "sex_readset":
        sex["required_asec_source_columns"] = ["A_SEX"]
        sex["absent_from_prepared_arm"] = ["A_SEX"]
    elif mutation == "head_readset":
        head["required_asec_source_columns"] = ["P_SEQ"]
        head["present_in_prepared_arm"] = []
        head["absent_from_prepared_arm"] = ["P_SEQ"]
        head["diagnostic_only_source_columns"] = []
    elif mutation == "contract":
        head["explicit_asec_source_contract"] = "P_SEQ identifies the head"
    elif mutation == "acs_gq":
        head["acs_arm_mapping"] = "Every ACS household has an observed head"
    else:
        head["diagnostic_only_is_semantic_proof"] = True
    with pytest.raises(PreparedGraphError, match="BINDING_UNBOUND_DEMOGRAPHICS"):
        binding.bind_composed_asec_document(canonical_json(document))


@pytest.mark.parametrize(
    ("attribute", "extra", "declarations"),
    [
        (
            "CPS_CARRIED_CURRENT_PERSON_LEAVES",
            "is_female",
            "composed_leaf_declarations",
        ),
        (
            "RESULT_COLUMNS",
            "is_household_head",
            "composed_reported_income_declarations",
        ),
    ],
)
def test_a_mapping_that_bound_an_unbound_demographic_refuses(
    monkeypatch, attribute, extra, declarations
):
    """The recorded gap fails closed on a write, which is what could stale it."""
    monkeypatch.setattr(measures, attribute, (*getattr(measures, attribute), extra))
    with pytest.raises(
        PreparedGraphError, match=f"ASEC_DEMOGRAPHIC_COLUMN_BOUND:{extra}"
    ):
        getattr(measures, declarations)()


@pytest.mark.parametrize(
    "roster",
    ["short", "not_a_list", "mapping", "not_objects"],
    ids=["short", "string", "mapping", "non-objects"],
)
def test_the_arm_rows_header_refuses_a_malformed_buffer_roster(executed, roster):
    """A header whose ``buffers`` is not the four declared entries is typed.

    Every other malformed shape in this artifact refuses with a reason; a
    strict zip against a wrong-length roster would raise a bare ValueError, and
    a roster of the right length whose entries are not objects would refuse
    with a per-buffer reason that misdescribes what is actually wrong.
    """
    _root, manifest, store, *_rest = executed
    payload = _payload(manifest, store, binding.BIND_NODE, "arm_rows")
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    assert binding.bind_composed_asec_arm_rows(payload, binding_document=document)
    start = len(binding.ARM_ROWS_MAGIC) + 4
    size = struct.unpack_from("<I", payload, len(binding.ARM_ROWS_MAGIC))[0]
    header = json.loads(payload[start : start + size])
    body = payload[start + size : -32]
    broken = dict(
        header,
        buffers={
            "short": header["buffers"][:3],
            "not_a_list": "four buffers",
            "mapping": {item["name"]: item for item in header["buffers"]},
            "not_objects": [item["sha256"] for item in header["buffers"]],
        }[roster],
    )
    reframed = _json(broken)
    candidate = (
        binding.ARM_ROWS_MAGIC + struct.pack("<I", len(reframed)) + reframed + body
    )
    with pytest.raises(PreparedGraphError, match="ARM_ROWS_BUFFER_ROSTER"):
        binding.bind_composed_asec_arm_rows(
            candidate + bytes.fromhex(_sha(candidate)),
            binding_document=document,
        )


def test_a_tampered_binding_document_cannot_admit_the_arm_rows(executed):
    """The row buffers are admitted only against the exact binding that cut them."""
    root, manifest, store, *_rest = executed
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    tampered = dict(document)
    tampered["claim"] = "something_else"
    with pytest.raises(Exception, match="ARM_ROWS_BINDING"):
        binding.bind_composed_asec_arm_rows(
            _payload(manifest, store, binding.BIND_NODE, "arm_rows"),
            binding_document=tampered,
        )
    del root


def test_the_arm_rows_artifact_refuses_a_tampered_payload(executed):
    root, manifest, store, *_rest = executed
    payload = _payload(manifest, store, binding.BIND_NODE, "arm_rows")
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    assert binding.bind_composed_asec_arm_rows(payload, binding_document=document)
    flipped = bytearray(payload)
    flipped[-40] ^= 0x01
    with pytest.raises(Exception, match="ARM_ROWS_CHECKSUM"):
        binding.bind_composed_asec_arm_rows(bytes(flipped), binding_document=document)
    del root


@pytest.mark.parametrize(
    ("original", "arm", "reason"),
    [
        ([], [1, 2, 3], "BINDING_EMPTY:person"),
        ([1, 1], [1, 2], "BINDING_AMBIGUOUS_ROWS:person"),
        ([1, 2], [1, 1], "BINDING_AMBIGUOUS_SOURCE:person"),
        ([1, 9], [1, 2], "BINDING_UNRESOLVED:person"),
        ([2, 1], [1, 2], "BINDING_NON_MONOTONE:person"),
    ],
)
def test_the_resolution_refuses_an_unusable_mapping(original, arm, reason):
    """Empty, ambiguous, unresolved and reordered mappings all refuse."""
    with pytest.raises(PreparedGraphError, match=reason):
        binding._resolved_positions(
            np.asarray(original, dtype="int64"),
            np.asarray(arm, dtype="int64"),
            entity="person",
        )


def test_the_declarations_write_only_under_the_arm_mask(executed):
    """Every measured cell is masked; nothing claims a whole-population write."""
    root, *_rest = executed
    graph = _bound_graph(root)
    measured = [
        node
        for node in graph.nodes
        if node.id in (measures.LEAVES_NODE, measures.REPORTED_INCOME_NODE)
    ]
    assert len(measured) == 2
    for node in measured:
        assert node.outputs
        for owned in node.outputs:
            assert owned.rows == binding.arm_row_column(owned.entity)
        for slice_ in node.inputs:
            assert slice_.rows == binding.arm_row_column(slice_.entity)
    bind_node = next(node for node in graph.nodes if node.id == binding.BIND_NODE)
    assert {owned.column for owned in bind_node.outputs} == {
        binding.arm_row_column(entity) for entity in binding.BOUND_ENTITIES
    }
    assert all(owned.dtype == "bool" for owned in bind_node.outputs)


def test_the_graph_declares_the_same_binding_without_geography(executed):
    """The geography-free composition takes the same three nodes, one version up."""
    root, *_rest = executed
    graph = measures.composed_asec_population_graph(
        _columns(root),
        sample_fraction=FRACTION,
        sample_seed=SEED,
        geography=False,
    )
    compiled = compile_graph(graph)
    ids = {node.id for node in graph.nodes}
    assert {
        binding.BIND_NODE,
        measures.LEAVES_NODE,
        measures.REPORTED_INCOME_NODE,
    } <= ids
    assert GEOGRAPHY_PHASE not in ids
    assert compiled.versions[binding.BIND_NODE] == composed.HARMONIZE_NODE


def test_registry_declares_exactly_the_kernels_this_graph_runs(executed):
    root, *_rest = executed
    graph = _bound_graph(root)
    registry = measures.composed_asec_population_registry()
    declared = {node.kernel for node in graph.nodes}
    assert declared == set(registry.refs())
    assert {
        binding.USComposedAsecBindKernel.ref,
        measures.USComposedAsecLeavesKernel.ref,
        measures.USComposedAsecReportedIncomeKernel.ref,
    } <= declared


def test_fresh_process_warm_replay_is_all_hits_and_runs_no_kernel(executed):
    """A new interpreter replays the whole graph from the store alone."""
    root, manifest, store, *_rest = executed
    replay = _process(root, "_cache_only")
    assert replay["pid"] != os.getpid()
    # The fresh interpreter resolved this checkout, not a shared install.
    assert replay["roots"] == _import_roots()
    assert all(
        root_path.startswith(str(Path(__file__).resolve().parents[3]))
        for root_path in replay["roots"].values()
    )
    assert replay["all_hits"] is True
    assert replay["kernel_calls"] == 0
    assert replay["key"] == manifest.key
    assert replay["nodes"] == {
        name: record.key for name, record in manifest.nodes.items()
    }
    assert replay["binding_sha256"] == _sha(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    assert replay["reported_income_sha256"] == _sha(
        _payload(manifest, store, measures.REPORTED_INCOME_NODE, "reported_income")
    )


def _structural_acs_frame():
    """A native-ACS-shaped arm whose group ids interleave with the ASEC arm's.

    The ASEC fixture's household/SPM ids are ``1..3`` and ``1001..1003``. These
    ids are ``500``/``501``, which collide with neither, so assembly applies no
    offset and the sorted group tables really do interleave: the ASEC rows are
    neither a prefix nor contiguous. Household 500 also holds four persons
    across two SPM units and two tax units, with its person rows out of order.
    """
    from microcosm.frame import Weights

    household_of = np.array([500, 501, 500, 500, 500], dtype=np.int64)
    spm_of = np.array([500, 501, 500, 5000, 5000], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.array([41, 12, 44, 43, 42], dtype=np.int64),
            "person_household_id": household_of,
            "person_spm_unit_id": spm_of,
            "person_tax_unit_id": spm_of,
            "person_family_id": household_of,
            "person_marital_unit_id": np.array(
                [500, 501, 5001, 5000, 5002], dtype=np.int64
            ),
            "age": np.array([44.0, 61.0, 12.0, 29.0, 27.0]),
            "source_year": np.array([2024] * 5, dtype=np.int64),
            "source_household_id": household_of,
            "source_person_id": pd.Series(
                ["1", "1", "3", "2", "4"], dtype=CANONICAL_STRING_DTYPE
            ),
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
    household["TYPEHUGQ"] = np.array([1, 2], dtype=np.int64)
    household["SERIALNO"] = pd.Series(
        ["0000000000123", "2024GQ0000004"], dtype=CANONICAL_STRING_DTYPE
    )
    household["puma"] = pd.Series(
        [f"06{_NATIVE_PUMA}", f"36{_NATIVE_PUMA}"], dtype=CANONICAL_STRING_DTYPE
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([13.0, 0.0]), WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr"] * len(person), dtype=object),
    )


_NATIVE_PUMA = "00100"


@pytest.fixture(scope="module")
def structural(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-composed-asec-structural")
    with pytest.MonkeyPatch.context() as patch:
        _module("test_us_asec_prepared_source.py").invented_prepared_directory(
            root, patch, households_per_cohort=2
        )
        _composed_tests().invented_acs_archives(root / "acs")
        _write_pins(root)
        patch.setattr(
            composed, "load_graph_acs", lambda path, **kw: _structural_acs_frame()
        )
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
        manifest, store = _run(root, store_name="store-structural")
        yield root, manifest, store


def test_interleaved_group_ids_and_multiple_spm_units_bind(structural):
    """The arm is neither a prefix nor contiguous, and still resolves exactly."""
    _root, manifest, store = structural
    frame = _bound_frame(manifest)
    household = frame.table("household")
    channels = household[support_channel_column("household")].tolist()
    assert channels.count(BASE_ASEC_SUPPORT_CHANNEL) == 6
    # Genuinely interleaved: an ACS household sits between two ASEC ones.
    first_acs = channels.index(ACS_STACKED_SUPPORT_CHANNEL)
    assert first_acs > 0
    assert BASE_ASEC_SUPPORT_CHANNEL in channels[first_acs:]
    # And the resolved source positions are still the arm's own, in order.
    document = binding.bind_composed_asec_document(
        _payload(manifest, store, binding.BIND_NODE, "asec_binding")
    )
    rows = binding.bind_composed_asec_arm_rows(
        _payload(manifest, store, binding.BIND_NODE, "arm_rows"),
        binding_document=document,
    )
    positions = rows.array("household", "source_positions")
    np.testing.assert_array_equal(positions, np.arange(6, dtype="int64"))
    arm_household = _arm(frame, "household")
    np.testing.assert_array_equal(
        arm_household[spine_source_id_column("household")].to_numpy(dtype="int64"),
        rows.array("household", "original_ids"),
    )
    # The multi-SPM ACS household really exists and is left unmeasured.
    acs_person = _other_arm(frame, "person")
    per_household = acs_person.groupby(
        "person_household_id"
    ).person_spm_unit_id.nunique()
    assert sorted(per_household.tolist()) == [1, 2]
    assert _other_arm(frame, "spm_unit")[binding.arm_row_column("spm_unit")].sum() == 0


def test_native_acs_group_quarters_survive_the_binding(structural):
    """A zero-weight GQ household stays ACS, unmeasured and unmodified."""
    _root, manifest, _store = structural
    frame = _bound_frame(manifest)
    household = frame.table("household")
    gq = household["TYPEHUGQ"] == 2
    assert int(gq.sum()) == 1
    assert not household.loc[gq, binding.arm_row_column("household")].any()
    assert household.loc[gq, "SERIALNO"].tolist() == ["2024GQ0000004"]
    gq_person = frame.table("person")
    gq_rows = gq_person["person_household_id"].isin(
        household.loc[gq, "household_id"].tolist()
    )
    assert int(gq_rows.sum()) == 1
    for name in RESULT_COLUMNS:
        assert gq_person.loc[gq_rows, name].isna().all()
