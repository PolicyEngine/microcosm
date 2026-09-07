"""Actual source/PUF/cap composition, using only the accepted invented fixtures."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

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
        stage.ages.NationalAgeCountKernel(),
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
    assert result["budget"].document["expanded_original_design_anchors"]["household"][
        "weight_hex"
    ]
    assert (root / "cold.json").is_file()
