"""Invented full factory and grouped adapter: actual cache keys and zero calls."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import origin_group_calibration_graph as stage
from microcosm.build.us_runtime.graph_sources import us_source_codecs
from microcosm.build.us_runtime.support_provenance import support_clone_index_column
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.graph import ContentStore, Owned

CHECKOUT = Path(__file__).resolve().parents[3]


def _load(name, relative):
    path = CHECKOUT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(root):
    invented = _load("origin_cap_invented", "tools/survey_lineage_prep/invented.py")
    root.mkdir()
    invented.make(root, CHECKOUT)


def _registry():
    # An invented household indicator intentionally differs between clone roles.
    # This tests redistribution; it is not a measured demographic reference.
    return TargetRegistry(
        [
            TargetSpec(
                name="S0101_C01_001",
                entity="household",
                measure=support_clone_index_column("household"),
                value=1e8,
                source="invented role-indicator redistribution fixture",
                period="2024",
                family="acs.S0101",
                metadata={
                    "table": "S0101",
                    "reference_sha256": "a" * 64,
                    "geography": "0100000US",
                    "universe": "population",
                    "role": "calibration",
                    "evidence_scope": "invented",
                },
            )
        ],
        country="us",
    )


def _run(
    root, *, warm=False, observe=False, design_multiplier=4.0, incoming_multiplier=2.0
):
    invented = _load("origin_cap_invented", "tools/survey_lineage_prep/invented.py")
    factory = _load("origin_cap_factory", "tools/us_survey_lineage_graph.py")
    invented.restore(root, CHECKOUT)
    identity = json.loads((root / "projection-identity.json").read_text())
    graph = factory.integrated_graph(
        tuple(Owned(*v) for v in json.loads((root / "columns.json").read_text())),
        sample_fraction=1.0,
        sample_seed=17,
        projection_sha256=identity["projection_sha256"],
        rows=identity["rows"],
        cohort_rows=tuple(tuple(v) for v in identity["cohort_rows"]),
    )
    kernels = factory.integrated_registry()
    observations = []
    kernels.register(stage.FrozenOriginBudgetKernel())
    kernels.register(
        stage.GroupedDemographicCalibrationKernel(
            _post_projection_observer=observations.append if observe else None
        )
    )
    calls = []
    # Preserve all source capabilities and implementation hashes. Only run is
    # replaced with counting/raising wrappers, never a claimed numeric contract.
    for ref in {node.kernel for node in graph.nodes} | {
        stage.FrozenOriginBudgetKernel.ref,
        stage.GroupedDemographicCalibrationKernel.ref,
    }:
        kernel = kernels.get(ref)
        original = kernel.run

        def run(context, original=original, ref=ref):
            calls.append(ref)
            if warm:
                raise AssertionError("REQUIRED replay executed " + ref)
            return original(context)

        kernel.run = run
    store = ContentStore(root / "calibration-store", codecs=us_source_codecs())
    result = stage.run_origin_group_calibration(
        graph,
        sources=invented.sources(root),
        store=store,
        kernels=kernels,
        native_binding_node=factory.NATIVE_BINDING,
        expanded_binding_node=factory.CLONE_BINDING,
        target_registry=_registry(),
        budget_path=root / "frozen-budget.json",
        epochs=25,
        learning_rate=0.2,
        design_multiplier=design_multiplier,
        incoming_multiplier=incoming_multiplier,
        resume="require" if warm else "auto",
    )
    frame = result.manifest.population(stage.CALIBRATION_NODE)
    weights = frame.weights_for("household").values
    if observe:
        assert len(observations) == 26
        assert any(o["corrected_group_count"] > 0 for o in observations[1:])
        assert observations[-1]["weights"].tobytes() == weights.tobytes()
        for observation in observations:
            assert observation["weights"].dtype == np.float64
            result.budget.grouped_bounds.check(observation["weights"], positive=True)
    rows = frame.table("household")
    by_id = dict(zip(rows.household_id, weights, strict=True))
    redistributed = []
    for origin in result.budget.document["origins"]:
        incoming = float.fromhex(origin["frozen_incoming_hex"])
        scalar_old_cap = incoming  # rI=2 times the initial I/2 role allocation.
        redistributed.append(
            any(by_id[m["household_id"]] > scalar_old_cap for m in origin["members"])
        )
    return {
        "pid": os.getpid(),
        "calls": calls,
        "keys": {n: v.key for n, v in result.manifest.nodes.items()},
        "hits": {n: v.hit for n, v in result.manifest.nodes.items()},
        "manifest_key": result.manifest.key,
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "budget_sha256": result.budget.digest,
        "successor_sha256": result.successor.digest,
        "successor_key": result.successor_key,
        "diagnostics": result.diagnostics,
        "corrected_updates": sum(o["corrected_group_count"] > 0 for o in observations),
        "redistributed": redistributed,
        "households": frame.n("household"),
        "persons": frame.n("person"),
    }


def test_integrated_factory_real_unequal_adam_and_fresh_required_replay(tmp_path):
    root = tmp_path / "invented"
    _fixture(root)
    cold = _run(root, observe=True)
    assert any(cold["redistributed"])
    assert cold["calls"].count(stage.GroupedDemographicCalibrationKernel.ref) == 1
    output = root / "fresh-warm.json"
    prefix = Path("/private/tmp") / ("origin-cap-warm-" + uuid.uuid4().hex)
    assert prefix.is_absolute() and not prefix.exists()
    script = """
import hashlib, importlib.util, json, pathlib, subprocess, sys
checkout=pathlib.Path(sys.argv[1]); source=pathlib.Path(sys.argv[2])
sys.path[:0]=[str(p) for p in sorted((checkout/'packages').glob('*/src'))]
before={}
for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=checkout,text=True).splitlines():
 p=checkout/rel
 if p.is_file(): before[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('origin_cap_fresh',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
result=module._run(pathlib.Path(sys.argv[3]),warm=True)
loaded=[]
for m in list(sys.modules.values()):
 p=getattr(m,'__file__',None)
 if not p:continue
 p=pathlib.Path(p).resolve()
 if not p.is_relative_to(checkout) or p.suffix!='.py':continue
 digest=hashlib.sha256(p.read_bytes()).hexdigest(); rel=str(p.relative_to(checkout))
 committed=subprocess.check_output(['git','show','HEAD:'+rel],cwd=checkout)
 assert digest==before[str(p)]==hashlib.sha256(committed).hexdigest(),rel
 loaded.append({'path':rel,'sha256':digest})
result['loaded_sources']=loaded
pathlib.Path(sys.argv[4]).write_text(json.dumps(result,sort_keys=True))
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={prefix}",
            "-c",
            script,
            str(CHECKOUT),
            __file__,
            str(root),
            str(output),
        ],
        check=True,
        timeout=150,
    )
    warm = json.loads(output.read_text())
    assert warm["pid"] != cold["pid"]
    assert warm["calls"] == []
    assert all(warm["hits"].values())
    assert warm["loaded_sources"]
    for key in (
        "keys",
        "manifest_key",
        "weights_sha256",
        "budget_sha256",
        "successor_sha256",
        "successor_key",
        "diagnostics",
    ):
        assert warm[key] == cold[key], key
    evidence = root / "observed-evidence.json"
    evidence.write_text(json.dumps({"cold": cold, "warm": warm}, sort_keys=True))
    print("ORIGIN_CAP_OBSERVED_EVIDENCE=" + str(evidence))


def test_scalar_row_cap_is_not_forwarded_in_grouped_adapter():
    node = stage.grouped_demographic_calibration_node(
        _registry(),
        base="clone",
        native_binding_node="native",
        expanded_binding_node="expanded",
        epochs=2,
        learning_rate=0.1,
        design_multiplier=4.0,
        incoming_multiplier=2.0,
    )
    assert node.params["max_weight_ratio"] == 4.0  # executor DESIGN guard only
    assert [a.name for a in node.artifact_inputs] == [
        "budget",
        "native_origin",
        "expanded_origin",
    ]


@pytest.mark.parametrize(
    "parameter", ["epochs", "learning_rate", "design_multiplier", "incoming_multiplier"]
)
def test_invalid_adapter_options_refuse(parameter):
    options = dict(
        epochs=2, learning_rate=0.1, design_multiplier=4.0, incoming_multiplier=2.0
    )
    options[parameter] = 0
    with pytest.raises(ValueError, match="GROUPED_"):
        stage.grouped_demographic_calibration_node(
            _registry(),
            base="clone",
            native_binding_node="native",
            expanded_binding_node="expanded",
            **options,
        )
