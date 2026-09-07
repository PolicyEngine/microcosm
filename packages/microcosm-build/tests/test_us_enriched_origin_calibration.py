"""Actual source/PUF/cap composition, using only the accepted invented fixtures."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_us

# The accepted target helper has an optional genuine aggregate test fixture.
# This file uses only its explicit invented constructors, with no ambient path.
os.environ.pop("MICROCOSM_US_NATIONAL_ACS_BUNDLE", None)


@pytest.fixture(scope="module", autouse=True)
def invented_only_guard():
    """Deny known genuine trees/network and all policy-engine calculations."""
    import resource
    import threading

    denials = []
    stop = threading.Event()

    def memory_watch():
        while not stop.wait(0.1):
            # On the pinned Darwin runtime ru_maxrss is bytes. This worker
            # cap supplements the launcher's enforced CPU/wall limits.
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024**3:
                os._exit(86)

    monitor = threading.Thread(target=memory_watch, daemon=True)
    monitor.start()

    def guard(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            text = os.fsdecode(path)
            if any(
                s in text
                for s in (
                    "/_buildo-runtime/",
                    "/.cache/microcosm/",
                    "/storage/",
                    "/native-asec-original-",
                )
            ):
                denials.append("genuine_path")
                raise RuntimeError("GENUINE_PATH_FORBIDDEN")

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            denials.append("network")
            raise RuntimeError("NETWORK_FORBIDDEN")
        if event in ("open", "os.listdir", "os.scandir"):
            guard(args[0])

    sys.addaudithook(audit)
    from unittest.mock import patch

    original_stat, original_lstat = os.stat, os.lstat

    def stat(path, *args, **kwargs):
        guard(path)
        return original_stat(path, *args, **kwargs)

    def lstat(path, *args, **kwargs):
        guard(path)
        return original_lstat(path, *args, **kwargs)

    patches = [patch("os.stat", stat), patch("os.lstat", lstat)]
    import policyengine_core.simulations as core
    import policyengine_us as us

    def no_calculation(*args, **kwargs):
        denials.append("calculation")
        raise RuntimeError("POLICY_CALCULATION_FORBIDDEN")

    for package in (core, us):
        for name in ("Simulation", "Microsimulation", "IndividualSim"):
            cls = getattr(package, name, None)
            if cls is not None:
                for method in (
                    "calculate",
                    "calculate_add",
                    "calculate_divide",
                    "calculate_output",
                    "calculate_dataframe",
                    "_calculate",
                ):
                    if hasattr(cls, method):
                        patches.append(patch.object(cls, method, no_calculation))
    for item in patches:
        item.start()
    yield denials
    stop.set()
    monitor.join(timeout=1)
    for item in reversed(patches):
        item.stop()
    assert denials == []


def peer(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_enriched_integration_module_has_separate_contract():
    from microcosm.build.us_runtime import enriched_origin_calibration as stage

    assert stage.ENRICHED_SCHEMA != stage.SUCCESSOR_SCHEMA
    assert stage.RECIPE == {
        "epochs": 500,
        "learning_rate": 0.05,
        "seed": 0,
        "incoming_multiplier": 4.0,
        "design_multiplier": 8.0,
        "mass": "free",
    }


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_masked_integer_boolean_change_is_replay_only(implementation, dtype):
    import pandas as pd

    from microcosm.graph.population import storage_equal

    original = pd.Series([1, None], dtype=dtype)
    restored = original.copy(deep=True)
    original.array._data[1] = 1
    restored.array._data[1] = 0
    assert implementation.cache_canonical_storage_equal(original, restored)
    assert not storage_equal(original, restored)


@pytest.mark.parametrize("dtype", ["Int64", "boolean", "Float64"])
@pytest.mark.parametrize("mutation", ["observed", "mask", "dtype"])
def test_replay_still_rejects_changed_cells(implementation, dtype, mutation):
    import pandas as pd

    before = pd.Series([1, None], dtype=dtype)
    after = before.copy(deep=True)
    if mutation == "observed":
        after.array._data[0] = 0
    elif mutation == "mask":
        after.array._mask[1] = False
    else:
        after = after.astype("Float64" if dtype != "Float64" else "Int64")
    assert not implementation.cache_canonical_storage_equal(before, after)


def test_hidden_float_backing_is_never_normalized(implementation):
    import pandas as pd

    before = pd.Series([1, None], dtype="Float64")
    after = before.copy(deep=True)
    after.array._data[1] = 17.0
    assert not implementation.cache_canonical_storage_equal(before, after)


def tiny_population(*, dtype="Int64", hidden=0, metadata=None):
    import numpy as np
    import pandas as pd

    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
    from microcosm.graph.population import Population

    tables = {e: pd.DataFrame({e + "_id": [1, 2]}) for e in US_SCHEMA.entities}
    for entity in US_SCHEMA.group_entities:
        tables["person"]["person_" + entity + "_id"] = [1, 2]
    tables["person"]["invented_unknown"] = pd.Series([1, None], dtype=dtype)
    tables["person"]["invented_unknown"].array._data[1] = hidden
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 3.0]), WeightKind.DESIGN)},
        metadata=metadata or {"typed": 1},
    )
    return Population.from_frame(frame, version="invented")


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_actual_population_replay_is_distinct_from_patch(implementation, dtype):
    before = tiny_population(dtype=dtype, hidden=1)
    restored = tiny_population(dtype=dtype, hidden=0)
    implementation.verify_cache_replay(before, restored)
    with pytest.raises(implementation.budgets.OriginBudgetError, match="CARRIED_CELL"):
        implementation.same_carried(before, restored)


def test_typed_metadata_refuses_int_to_boolean(implementation):
    before = tiny_population(metadata={"typed": 1})
    after = tiny_population(metadata={"typed": True})
    for check in (implementation.verify_cache_replay, implementation.same_carried):
        with pytest.raises(
            implementation.budgets.OriginBudgetError, match="FRAME_METADATA"
        ):
            check(before, after)


def test_typed_index_name_refuses_int_to_boolean(implementation):
    before, after = tiny_population(), tiny_population()
    before.frame.person.index.name = 1
    after.frame.person.index.name = True
    for check in (implementation.verify_cache_replay, implementation.same_carried):
        with pytest.raises(
            implementation.budgets.OriginBudgetError, match="ORDERED_COLUMNS_OR_AXIS"
        ):
            check(before, after)


def test_source_owned_zero_backing_is_not_a_replay_exception(implementation):
    import numpy as np
    import pandas as pd

    from microcosm.graph.population import Population

    stage = implementation
    frame = tiny_population().frame
    for column, _ in stage.source_demographics._RETAINED_STATES:
        frame.person[column] = pd.Series(
            pd.arrays.IntegerArray(
                np.zeros(2, dtype="int64"),
                np.ones(2, dtype=bool),
            )
        )
    before = Population.from_frame(frame, version="invented")
    after = changed_population(before)
    first = stage.source_demographics._RETAINED_STATES[0][0]
    after.frame.person[first].array._data[1] = 1
    for check in (stage.verify_cache_replay, stage.same_carried):
        with pytest.raises(
            stage.budgets.OriginBudgetError, match="EXPLICIT_NULL_BACKING"
        ):
            check(before, after)


def test_object_strata_replay_uses_real_scalar_codec(implementation):
    import pandas as pd

    before = pd.Series(["".join(["invented", " stratum ", "one"])], dtype=object)
    after = pd.Series([bytes(before.iloc[0], "utf8").decode("utf8")], dtype=object)
    assert before.iloc[0] is not after.iloc[0]
    assert not implementation.storage_equal(before, after)
    assert implementation.cache_strata_equal(before, after)
    assert not implementation.cache_canonical_storage_equal(before, after)


@pytest.mark.parametrize(
    "left,right",
    [(1, True), (1, 1.0), (0.0, -0.0), (None, ""), ("a", b"a"), ("a", "b")],
)
def test_object_strata_tags_and_bits_remain_exact(implementation, left, right):
    import pandas as pd

    assert not implementation.cache_strata_equal(
        pd.Series([left], dtype=object), pd.Series([right], dtype=object)
    )


@pytest.fixture(scope="module")
def implementation():
    from microcosm.build.us_runtime import enriched_origin_calibration

    return enriched_origin_calibration


def count_context(stage):
    import numpy as np
    import pandas as pd

    from microcosm.graph import KernelContext

    node = stage.enriched_age_count_node()
    return KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame({"person_household_id": [1, 2], "age": [7, 63]}),
            "household": pd.DataFrame({"household_id": [1, 2]}),
            "tax_unit": pd.DataFrame({stage.detail.OUTPUT: [float("nan"), 11.0]}),
        },
        weights={},
        strata=pd.Series(["invented", "invented"]),
        params=node.params,
        rng=np.random.default_rng(0),
    )


def test_scoped_count_uses_unchanged_calculation(implementation):
    from dataclasses import replace

    stage = implementation
    context = count_context(stage)
    actual = stage.EnrichedAgeCountKernel().run(context)
    expected = stage.ages.NationalAgeCountKernel().run(
        replace(
            context,
            node=stage.ages.national_age_count_node(population=stage.BOUNDARY),
            tables={e: context.tables[e] for e in ("person", "household")},
        )
    )
    assert actual.receipt == expected.receipt
    assert actual.columns.keys() == expected.columns.keys()
    assert all(
        stage.storage_equal(actual.columns[c], expected.columns[c])
        for c in expected.columns
    )


@pytest.mark.parametrize(
    "mutation",
    ["dependency", "dtype", "table", "params", "schema_bool", "schema_float"],
)
def test_scoped_count_dependency_refusals(implementation, mutation):
    from dataclasses import replace

    stage = implementation
    context = count_context(stage)
    if mutation == "dependency":
        context = replace(
            context, node=replace(context.node, inputs=context.node.inputs[:-1])
        )
    elif mutation == "dtype":
        tables = dict(context.tables)
        tables["tax_unit"] = tables["tax_unit"].astype("float32")
        context = replace(context, tables=tables)
    elif mutation == "table":
        context = replace(
            context, tables={e: t for e, t in context.tables.items() if e != "tax_unit"}
        )
    elif mutation in ("schema_bool", "schema_float"):
        params = {
            **context.params,
            "schema_version": True if mutation == "schema_bool" else 1.0,
        }
        context = replace(
            context, node=replace(context.node, params=params), params=params
        )
    else:
        context = replace(context, params={**context.params, "maximum_age": 121})
    with pytest.raises(ValueError):
        stage.EnrichedAgeCountKernel().run(context)


def prepare(root):
    """Prepare accepted invented source/price shapes; bootstrap is not the cold run."""
    root.mkdir()
    old = peer("test_us_puf_diagnostic_consumer")
    (root / "price").mkdir()
    payloads, price_pins, _ = old.fixture_transport(root / "price")
    host = old.host_fixture(root / "host")
    columns = old.frame_column_declarations(
        host[4].population(old.survey.CLONE_POPULATION)
    )
    (root / "transport").mkdir()
    for name, payload in payloads.items():
        (root / "transport" / (name + ".bin")).write_bytes(payload)
    targets = peer("test_us_national_age_activation")
    target_sha = targets.invented_bundle(root / "targets")
    (root / "inputs.json").write_text(
        json.dumps(
            {
                "price_pins": price_pins,
                "host_pins": host[6],
                "target_sha": target_sha,
                "columns": [[c.entity, c.column, c.dtype] for c in columns],
            }
        )
    )


def run_fixture(root, *, warm=False):
    from microcosm.build.us_runtime import enriched_origin_calibration as stage
    from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
    from microcosm.fit.graph_legacy_qrf import LegacyQRFTrainKernel
    from microcosm.graph import ContentStore, Owned

    old = peer("test_us_puf_diagnostic_consumer")
    old.invented.restore(root / "host", old.CHECKOUT)
    data = json.loads((root / "inputs.json").read_text())
    identity = json.loads((root / "host/projection-identity.json").read_text())
    identity["cohort_rows"] = tuple(tuple(v) for v in identity["cohort_rows"])
    graph = old.survey.integrated_graph(
        tuple(Owned(*c) for c in json.loads((root / "host/columns.json").read_text())),
        sample_fraction=1.0,
        sample_seed=17,
        **identity,
    )
    kernels = old.survey.integrated_registry()
    old.stage.register_diagnostic_kernels(kernels)
    for kernel in (
        old.shared.DetailBoundaryKernel(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        stage.EnrichedAgeCountKernel(),
        stage.caps.FrozenOriginBudgetKernel(),
        stage.caps.GroupedDemographicCalibrationKernel(),
    ):
        kernels.register(kernel)
    calls, source_calls = [], []
    for kernel in kernels.as_mapping().values():
        original = kernel.run

        def counted(context, original=original):
            calls.append(context.node.id)
            if warm:
                raise AssertionError("WARM_KERNEL_CALLED")
            return original(context)

        kernel.run = counted
    codecs = old.us_source_codecs()
    codecs.register_bytes("raw-bytes-v1", old.load_raw_bytes)
    store = ContentStore(root / "integration-store", codecs=codecs)
    if warm:

        def forbidden(*args, **kwargs):
            source_calls.append(1)
            raise AssertionError("WARM_GRAPH_SOURCE_CODEC_CALLED")

        codecs.load = codecs.load_bytes = forbidden
    targets = peer("test_us_national_age_activation")
    registry = targets.activate_national_age_targets(
        root / "targets", declaration=targets.invented_declaration(data["target_sha"])
    )
    result = stage.run_enriched_calibration(
        graph,
        sources={
            **old.invented.sources(root / "host"),
            **{
                name: root / "transport" / (role + ".bin")
                for role, name in old.stage.SOURCE_NAMES.items()
            },
        },
        store=store,
        kernels=kernels,
        native_binding=old.survey.NATIVE_BINDING,
        clone_binding=old.survey.CLONE_BINDING,
        clone_population=old.survey.CLONE_POPULATION,
        columns=tuple(Owned(*c) for c in data["columns"]),
        price_pins=data["price_pins"],
        host_pins=data["host_pins"],
        route="test_fixture",
        target_registry=registry,
        evidence_directory=root / "admission",
        resume="require" if warm else "auto",
    )
    result.update(
        store=store,
        kernel_calls=calls,
        source_calls=source_calls,
        host_pins=data["host_pins"],
    )
    manifest = result["manifest"]
    evidence = {
        "phases": result["phases"],
        "kernel_calls": calls,
        "graph_source_calls": source_calls,
        "keys": {n: r.key for n, r in manifest.nodes.items()},
        "opaque": {
            n + "/" + a: stage.sha(store.load_bytes(k))
            for n, r in manifest.nodes.items()
            for a, k in r.opaque_artifacts.items()
        },
        "budget": result["budget"].digest,
        "parent": result["parent"].digest,
        "successor": result["successor"].digest,
        "successor_key": result["successor_key"],
        "content": stage.budgets._frame_identity(
            result["populations"][stage.caps.CALIBRATION_NODE].frame
        ),
    }
    (root / ("warm.json" if warm else "cold.json")).write_text(
        json.dumps(evidence, sort_keys=True)
    )
    return result


@pytest.fixture(scope="module")
def cold_run(tmp_path_factory):
    prepared = os.environ.get("MICROCOSM_ENRICHED_INVENTED_PREPARED")
    if prepared:
        root = Path(prepared)
        assert root.is_absolute() and root.name.startswith("invented-enriched-")
        assert not (root / "integration-store").exists()
        assert not (root / "admission").exists()
        return root, run_fixture(root)
    root = tmp_path_factory.mktemp("enriched-literals") / "invented"
    prepare(root)
    return root, run_fixture(root)


def test_actual_26_node_staged_pipeline(cold_run):
    root, result = cold_run
    assert len(result["kernel_calls"]) == len(set(result["kernel_calls"])) == 26
    assert [p["nodes"] for p in result["phases"]] == [17, 20, 23, 24, 26]
    assert [p["hits"] for p in result["phases"]] == [0, 17, 20, 23, 24]
    assert sum(p["nodes"] for p in result["phases"]) == 110
    from microcosm.build.us_runtime import enriched_origin_calibration as stage

    for phase in result["phases"]:
        if stage.COUNT in phase["order"]:
            assert phase["order"].index(stage.ATTACH) < phase["order"].index(
                stage.COUNT
            )
        if stage.caps.CALIBRATION_NODE in phase["order"]:
            assert phase["order"].index(stage.COUNT) < phase["order"].index(
                stage.caps.CALIBRATION_NODE
            )
        assert all(
            result["phases"][-1]["keys"][n] == k for n, k in phase["keys"].items()
        )
    assert result["budget"].document["expanded_original_design_anchors"]["household"][
        "weight_hex"
    ]
    assert (root / "cold.json").is_file()


def test_cd_adapter_authenticates_issued_successor_frame_log(cold_run):
    """Real669-issued bindings must authenticate the later current Frame log."""
    from dataclasses import replace

    from microcosm.build.us_runtime import cd_current_source_evidence as cd
    from microcosm.build.us_runtime import enriched_origin_calibration as stage
    from microcosm.build.us_runtime import graph_native_household_origin as origins
    from microcosm.frame import Frame, MassChangeRecord
    from us_survey_lineage_graph import CLONE_BINDING, CLONE_POPULATION, NATIVE_BINDING

    _, result = cold_run
    clone = result["populations"][CLONE_POPULATION]
    current = result["populations"][stage.caps.CALIBRATION_NODE]
    binding = origins.verify_materialized_population_origins(
        result["manifest"],
        result["store"],
        population_node=CLONE_POPULATION,
        binding_node=CLONE_BINDING,
        parent_binding_node=NATIVE_BINDING,
    )
    kwargs = {"parent": result["parent"], "successor": result["successor"]}
    valid = cd.authenticate_current_lineage(clone, current, binding, **kwargs)
    assert valid["kind"] == "admitted669_weight_successor"
    assert current.design_weights
    f = current.frame
    modified = Frame(
        {e: f.table(e) for e in f.entities},
        f.schema,
        {e: f.weights_for(e) for e in f.weighted_entities},
        f.strata,
        metadata=f.metadata,
        mass_log=(
            *f.mass_log,
            MassChangeRecord("household", 0.0, 0.0, 1.0, "invented log-only mutation"),
        ),
    )
    changed = replace(current, frame=modified)
    # This is precisely the omitted boundary: all other authenticated identities
    # agree, including the independently recorded Population ledger and weights.
    before_id, after_id = (
        cd.current_population_identity(current),
        cd.current_population_identity(changed),
    )
    assert {k: v for k, v in before_id.items() if k != "frame_mass_log_sha256"} == {
        k: v for k, v in after_id.items() if k != "frame_mass_log_sha256"
    }
    assert before_id["frame_mass_log_sha256"] != after_id["frame_mass_log_sha256"]
    with pytest.raises(cd.CurrentEvidenceError, match="CD_CURRENT_SUCCESSOR_BINDING"):
        cd.authenticate_current_lineage(clone, changed, binding, **kwargs)


def test_successor_binds_actual_calibration_and_diagnostic_edge(cold_run):
    from microcosm.build.us_runtime import enriched_origin_calibration as stage

    _, result = cold_run
    receipt = result["manifest"].node(stage.caps.CALIBRATION_NODE)
    diagnostic = stage.edge_value(
        result["manifest"],
        result["store"],
        stage.caps.CALIBRATION_NODE,
        "diagnostics",
        stage.caps.demographic.DIAGNOSTICS_TYPE,
    )
    successor = result["successor"]
    assert successor.document["calibration_node_key"] == receipt.key
    assert successor.document["diagnostics_artifact"] == {
        "type": stage.normative(diagnostic.type),
        **stage.edge_pin(diagnostic),
    }
    expected_key = stage.sha256_domain(
        "puf-age-enriched-weight-successor",
        stage.canonical_json(
            {
                "calibration_node": receipt.key,
                "parent": result["parent"].digest,
                "payload": successor.digest,
                "type": stage.SUCCESSOR_SCHEMA,
            }
        ),
    )
    assert result["successor_key"] == expected_key
    assert result["store"].load_bytes(expected_key) == successor.payload


def changed_population(population, mutate=None, **frame_changes):
    from dataclasses import replace

    from microcosm.frame import Frame

    frame = population.frame
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    if mutate:
        mutate(tables)
    parameters = dict(metadata=frame.metadata, mass_log=frame.mass_log)
    parameters.update(frame_changes)
    changed = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        **parameters,
    )
    return replace(population, frame=changed)


@pytest.mark.parametrize(
    "mutation", ["columns", "metadata", "frame_log", "owner", "anchor"]
)
def test_materialized_nonowned_properties_refuse(cold_run, implementation, mutation):
    from dataclasses import replace

    from microcosm.frame import MassChangeRecord

    _, result = cold_run
    before = result["populations"][implementation.ATTACH]
    after = changed_population(before)
    if mutation == "columns":
        after = changed_population(
            before,
            lambda t: t.update(person=t["person"][list(t["person"].columns)[::-1]]),
        )
    elif mutation == "metadata":
        after = changed_population(
            before, metadata={**before.frame.metadata, "invented_new": True}
        )
    elif mutation == "frame_log":
        after = changed_population(
            before,
            mass_log=(
                *before.frame.mass_log,
                MassChangeRecord("household", 1.0, 1.0, 1.0, "invented"),
            ),
        )
    elif mutation == "owner":
        owners = dict(after.owners)
        owners[next(iter(owners))] = "invented_wrong_writer"
        after = replace(after, owners=owners)
    else:
        anchors = {e: v.copy() for e, v in after.design_weights.items()}
        anchors["household"][0] += 1
        after = replace(after, design_weights=anchors)
    with pytest.raises(implementation.budgets.OriginBudgetError):
        implementation.same_carried(before, after)


@pytest.mark.parametrize("mutation", ["output", "native_null", "mask_same_count"])
def test_materialized_puf_values_bound_to_real_draw(cold_run, implementation, mutation):
    import numpy as np

    _, result = cold_run
    stage = implementation
    manifest, store = result["manifest"], result["store"]
    matrix = stage.edge_value(
        manifest, store, stage.MATRIX, "matrix", stage.model_input.RECIPIENT_MATRIX_TYPE
    )
    placement = stage.edge_value(
        manifest, store, stage.MATRIX, "placement", stage.puf.PLACEMENT_TYPE
    )
    before = result["populations"][stage.ATTACH]
    mask = before.frame.table("tax_unit")[stage.detail.MASK].to_numpy().copy()
    selected, native = np.flatnonzero(mask)[0], np.flatnonzero(~mask)[0]

    def mutate(tables):
        table = tables["tax_unit"]
        if mutation == "output":
            table.loc[table.index[selected], stage.detail.OUTPUT] += 1.0
        elif mutation == "native_null":
            table.loc[table.index[native], stage.detail.OUTPUT] = 0.0
        else:
            table.loc[table.index[[selected, native]], stage.detail.MASK] = [
                False,
                True,
            ]
            assert int(table[stage.detail.MASK].sum()) == int(mask.sum())

    after = changed_population(before, mutate)
    with pytest.raises(stage.budgets.OriginBudgetError):
        stage.verify_draw(manifest, store, matrix, placement, mask, after)


@pytest.mark.parametrize("mutation", ["count", "ledger_prefix", "owner"])
def test_materialized_counts_and_full_ledger_refuse(cold_run, implementation, mutation):
    from dataclasses import replace

    _, result = cold_run
    stage = implementation
    before, after = (result["populations"][n] for n in (stage.ATTACH, stage.COUNT))
    if mutation == "count":
        column = stage.ages.NATIONAL_AGE_ACTIVATION.bands[0].column

        def mutate(t):
            t["household"].loc[t["household"].index[0], column] += 1

        after = changed_population(after, mutate)
    elif mutation == "ledger_prefix":
        assert len(after.mass_ledger) > 1
        after = replace(after, mass_ledger=after.mass_ledger[1:])
    else:
        owners = dict(after.owners)
        owners[("household", stage.ages.NATIONAL_AGE_ACTIVATION.bands[0].column)] = (
            stage.ATTACH
        )
        after = replace(after, owners=owners)
    with pytest.raises(stage.budgets.OriginBudgetError):
        stage.verify_counts(before, after)


class PayloadView:
    """Invented in-memory corruption; never rewrite the accepted cache."""

    def __init__(self, store, changes):
        self.store, self.changes = store, changes

    def load_bytes(self, key):
        return self.changes[key] if key in self.changes else self.store.load_bytes(key)


def test_successor_preserves_validated_nonzero_correction_counter(cold_run):
    """Self-consistent invented diagnostic metadata, not a claimed solver trace."""
    from dataclasses import replace

    from microcosm.build.us_runtime import enriched_origin_calibration as stage

    _, result = cold_run
    manifest, store, budget = result["manifest"], result["store"], result["budget"]
    after = result["populations"][stage.caps.CALIBRATION_NODE]
    values = after.frame.weights_for("household").values
    receipt = manifest.node(stage.caps.CALIBRATION_NODE)
    key = receipt.opaque_artifacts["diagnostics"]
    document = json.loads(store.load_bytes(key))
    expected = budget.diagnostics(values, 1)
    document["build"].update(expected)
    document["options"]["grouped_upper_bounds"] = budget.grouped_bounds.diagnostics(
        values, 1
    )
    payload = stage.canonical_json(document)
    updated = {**receipt.receipt, **expected, "diagnostics_sha256": stage.sha(payload)}
    receipts = dict(manifest.nodes)
    receipts[stage.caps.CALIBRATION_NODE] = replace(receipt, receipt=updated)
    manifest = replace(manifest, nodes=receipts)
    successor = stage.weight_successor(
        result["parent"],
        budget,
        result["populations"][stage.COUNT],
        after,
        node=result["graph"].node(stage.caps.CALIBRATION_NODE),
        partition=result["graph"].mass_partition,
        manifest=manifest,
        store=PayloadView(store, {key: payload}),
    )
    assert successor.document["diagnostics"] == expected
    assert successor.document["diagnostics"]["last_corrected_group_count"] == 1
    assert successor.document["diagnostics_artifact"]["payload_sha256"] == stage.sha(
        payload
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "budget",
        "receipt_sha",
        "budget_sha256",
        "budget_artifact_key",
        "accepted_weight_sha256",
    ],
)
def test_materialized_diagnostic_identities_refuse(cold_run, implementation, mutation):
    from dataclasses import replace

    _, result = cold_run
    stage = implementation
    manifest, store = result["manifest"], result["store"]
    receipt = manifest.node(stage.caps.CALIBRATION_NODE)
    updated = dict(receipt.receipt)
    changes = {}
    if mutation == "budget":
        key = manifest.node(stage.caps.BUDGET_NODE).opaque_artifacts["budget"]
        changes[key] = b"{}"
    elif mutation == "receipt_sha":
        updated["diagnostics_sha256"] = "0" * 64
    else:
        key = receipt.opaque_artifacts["diagnostics"]
        document = json.loads(store.load_bytes(key))
        document["build"][mutation] = "0" * 64
        changes[key] = stage.canonical_json(document)
        updated["diagnostics_sha256"] = stage.sha(changes[key])
    receipts = dict(manifest.nodes)
    receipts[stage.caps.CALIBRATION_NODE] = replace(receipt, receipt=updated)
    manifest = replace(manifest, nodes=receipts)
    with pytest.raises(stage.budgets.OriginBudgetError):
        stage.verify_final_diagnostics(
            manifest,
            PayloadView(store, changes),
            result["budget"],
            result["populations"][stage.caps.CALIBRATION_NODE],
        )


@pytest.mark.parametrize("mutation", ["frame", "transport", "receipt"])
def test_materialized_donor_refuses_before_fit_boundary(
    cold_run, implementation, mutation
):
    from dataclasses import replace

    root, result = cold_run
    stage = implementation
    data = json.loads((root / "inputs.json").read_text())
    payloads = {
        role: (root / "transport" / (role + ".bin")).read_bytes()
        for role in stage.consumer.ROLES
    }
    expected, payload = stage.consumer.qualify_price_transport(
        payloads,
        data["price_pins"],
        route="test_fixture",
        recipe=stage.consumer.recipe_document(),
    )
    manifest, store = result["manifest"], result["store"]
    donor = result["populations"][stage.puf.PREFIX + ".donor"]
    receipt = manifest.node(stage.puf.PREFIX + ".donor")
    if mutation == "frame":

        def mutate(t):
            t["tax_unit"].loc[t["tax_unit"].index[0], stage.detail.FEATURES[0]] += 1

        donor = changed_population(donor, mutate)
    elif mutation == "transport":
        store = PayloadView(store, {receipt.opaque_artifacts["transport"]: b"{}"})
    else:
        receipts = dict(manifest.nodes)
        receipts[stage.puf.PREFIX + ".donor"] = replace(
            receipt, receipt={**receipt.receipt, "transport_sha256": "0" * 64}
        )
        manifest = replace(manifest, nodes=receipts)
    with pytest.raises(stage.budgets.OriginBudgetError):
        stage.verify_donor(manifest, store, donor, expected, payload)
