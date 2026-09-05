"""Invented aggregate receipts only; no engine or source-data fixture access."""

from __future__ import annotations

import copy
import hashlib

# Load both pure files without executing microcosm.build's package initializer.
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/microcosm/build"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evidence = load_module("_test_graph_evidence", "graph_evidence.py")
writer = load_module("_test_graph_candidate_quality", "candidate_quality.py")
EvidenceError = evidence.EvidenceError
GENUINE_PINS = evidence.GENUINE_PINS
export_graph_evidence = evidence.export_graph_evidence
implementation_set_digest = evidence.implementation_set_digest
read_candidate_quality = writer.read_candidate_quality

IDS = [
    f"asec_prepared.{name}"
    for name in (
        "create",
        "select_households",
        "cps_carried_current",
        "engine_current_money",
    )
]
SOURCES = [f"asec_household_cohort_{year}.h5" for year in (2022, 2023, 2024)] + [
    "asec_household_observations.checkpoint.h5",
    "asec_person_income_source.checkpoint.h5",
    "asec_raw_stage.checkpoint.h5",
    "pppub23.csv",
    "pppub24.csv",
    "pppub25.csv",
    "restoration.receipt.json",
]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def invented(tmp_path):
    root = tmp_path / "invented-input"
    root.mkdir()
    impl = ("1" * 64, "2" * 64)
    outputs = {}

    def write(name, data):
        raw = canonical(data) + b"\n" if isinstance(data, dict) else data
        (root / name).write_bytes(raw)
        outputs[name] = {"bytes": len(raw), "sha256": digest(raw)}
        return outputs[name]["sha256"]

    source = {
        name: {
            "sha256": digest(name.encode()),
            "bytes": 1,
            "device": 1,
            "inode": i,
            "mtime_ns": 1,
            "ctime_ns": 1,
        }
        for i, name in enumerate(SOURCES)
    }
    runtime = {
        "packages": {
            name: {
                "files": 1,
                "version": "0.0.1",
                "source_and_parameters_sha256": digest(name.encode()),
            }
            for name in ("policyengine-us", "policyengine-core")
        },
        "policy": "installed_baseline_no_reforms",
    }
    loaded = {
        "microcosm.invented": {
            "matches_head_blob": True,
            "git_blob_oid": "a" * 40,
            "sha256": "b" * 64,
            "path": "invented.py",
        }
    }
    caps = dict(
        consumes_se=False,
        dependencies=["numpy"],
        determinism="deterministic",
        numeric="platform_bitwise",
        role="compute",
        seed_source="none",
        structural="none",
        tolerance=None,
    )
    names = [
        ["current_money", "frame_context", "prepared_receipt"],
        ["frame_context", "selected_current_money", "selection"],
        ["frame_context"],
        ["engine_evaluation"],
    ]
    nodes = {}
    for i, node_id in enumerate(IDS):
        key = digest(node_id.encode())
        typed = {
            name: {
                "artifact": name,
                "key": digest((node_id + name).encode()),
                "numerics": {
                    "numeric": "platform_bitwise",
                    "platform": "test",
                    "tolerance": None,
                },
                "producer": node_id,
                "producer_key": key,
                "type": {"name": "microcosm.invented." + name, "schema_version": 1},
            }
            for name in names[i]
        }
        inputs = {}
        if i:
            inputs = copy.deepcopy(nodes[IDS[i - 1]]["typed_artifacts"]["outputs"])
        nodes[node_id] = dict(
            artifacts=[{"entity": "person", "column": "invented", "key": "c" * 64}],
            capabilities={
                **caps,
                "structural": ("create", "filter", "none", "none")[i],
                "determinism": "seeded" if i == 1 else "deterministic",
                "seed_source": "param" if i == 1 else "none",
            },
            frame_key=None,
            hit=False,
            kernel_impl_hash=impl[int(i == 3)],
            kernel_ref="us." + node_id + "@1",
            key=key,
            legacy_capabilities=False,
            opaque_artifacts={k: v["key"] for k, v in typed.items()},
            receipt={
                "phase": "prepare_asec_current_money_slice",
                "release_eligible": False,
            },
            seed=0,
            typed_artifacts={"inputs": inputs, "outputs": typed},
            wall_time=0,
            weight_key=None,
        )
    body = {
        "nodes": {
            k: {f: v for f, v in n.items() if f not in ("hit", "wall_time")}
            for k, n in nodes.items()
        },
        "tier": None,
    }
    key = digest(b"manifest\0" + canonical(body))
    checkpoint = write("direct-frame.h5", b"opaque invented checkpoint")
    for mode in ("cold", "warm"):
        write(f"{mode}-frame.h5", b"opaque invented checkpoint")
        for node_id in IDS:
            for name in names[IDS.index(node_id)]:
                write(f"{mode}-{node_id}-{name}.bin", (node_id + name).encode())
        phase_nodes = copy.deepcopy(nodes)
        for n in phase_nodes.values():
            n["hit"] = mode == "warm"
        write(
            f"{mode}-manifest.json",
            dict(
                content_addressed=body,
                country="us",
                decisions=[],
                finished_at="2026-09-05T21:35:25+00:00",
                host="private-host.local",
                key=key,
                known_failures=[],
                nodes=phase_nodes,
                schema_version=3,
                started_at="2026-09-05T21:33:21+00:00",
                tier=None,
            ),
        )
    for i, mode in enumerate(("direct", "cold", "warm")):
        if mode == "direct":
            result = dict(
                entity_rows={"household": 15, "person": 52},
                whole_households=True,
                original_design_weight_bytes_preserved=True,
                source_preparations=1,
                roots=[
                    "capital_gains",
                    "dividend_income",
                    "employment_income",
                    "social_security",
                ],
                selected_field_axes_sha256={
                    f"field{j}": {
                        axis: "e" * 64
                        for axis in ("amounts", "statuses", "validity", "zero_origin")
                    }
                    for j in range(33)
                },
            )
        else:
            result = dict(
                all_hits=mode == "warm",
                artifacts={k: v["opaque_artifacts"] for k, v in nodes.items()},
                calls=[IDS[0], "source_preparation", *IDS[1:]]
                if mode == "cold"
                else [],
                checkpoint_sha256=checkpoint,
                direct_checkpoint_table_weight_parity=True,
                direct_real_engine_array_parity=True,
                money_33_fields_all_axes_parity=True,
                key=key,
                nodes={k: v["key"] for k, v in nodes.items()},
                source_preparations=int(mode == "cold"),
            )
        write(
            f"{mode}-receipt.json",
            dict(
                schema="microcosm.first_genuine_graph_worker.v1",
                head=GENUINE_PINS.run_head,
                phase=mode,
                pid=i + 1,
                status="PASS",
                release_eligible=False,
                result=result,
                engine_runtime_identity=runtime,
                loaded_modules=loaded,
                seconds=0,
            ),
        )
    run = dict(
        schema="microcosm.first_genuine_graph_run.v1",
        status="PASS",
        phase="source_postflight",
        head=GENUINE_PINS.run_head,
        runner_sha256=GENUINE_PINS.runner_sha256,
        roster_sha256=GENUINE_PINS.roster_sha256,
        approval_sha256=GENUINE_PINS.approval_sha256,
        calibration=False,
        fit=False,
        holdouts=False,
        population_tax_benefit_claim=False,
        prior_wages_added=False,
        reforms=False,
        release_eligible=False,
        source_preparations_planned=2,
        source_preparations_verified=2,
        original_after=source,
        staged_after=source,
        staging={"original_before": source, "original_after": source, "staged": source},
        stages={m: {"exit_code": 0} for m in ("direct", "cold", "warm")},
        outputs=outputs,
    )
    declaration = dict(
        schema="microcosm.engineering_candidate_declaration.v1",
        country="us",
        candidate_release_id="invented-engineering",
        run_id="invented-engineering-run",
        population={"period": "2024", "entity": "household", "sha256": checkpoint},
        basis="operator_declared_target_period_and_population_entity",
        reason="Invented target 2024 engineering slice. Uncalibrated and excluded from release claims.",
        created_at="2026-09-05T22:11:28+00:00",
    )

    def seal():
        # Refresh only invented file identities, never consult a real input run.
        for name in list(outputs):
            raw = (root / name).read_bytes()
            outputs[name] = {"bytes": len(raw), "sha256": digest(raw)}
        raw = canonical(run) + b"\n"
        (root / "run-receipt.json").write_bytes(raw)
        declaration["run_receipt_sha256"] = digest(raw)
        dec = canonical(declaration) + b"\n"
        (root / "declaration.json").write_bytes(dec)
        return replace(
            GENUINE_PINS,
            run_receipt_sha256=digest(raw),
            operator_declaration_sha256=digest(dec),
            implementation_hashes=impl,
        )

    return root, run, declaration, seal


def export_fixture(tmp_path, mutate=None):
    root, run, declaration, seal = invented(tmp_path)
    if mutate:
        mutate(root, run, declaration)
    return export_graph_evidence(
        run_root=root,
        declaration_path=root / "declaration.json",
        output_root=tmp_path / "quality",
        pins=seal(),
    )


def mutate_json(root, filename, fn):
    value = json.loads((root / filename).read_bytes())
    fn(value)
    (root / filename).write_bytes(canonical(value) + b"\n")


def test_known_ascii_implementation_set_and_permutations():
    entries = {
        "b": {"kernel_ref": "test.b@1", "kernel_impl_hash": "2" * 64},
        "a": {"kernel_ref": "test.a@1", "kernel_impl_hash": "1" * 64},
    }
    expected = "9973c87a16e71a92d98c70278c07a46a3d224e103e643b2bc359e476dfc31fb9"
    assert implementation_set_digest(entries) == expected
    assert implementation_set_digest(dict(reversed(list(entries.items())))) == expected


@pytest.mark.parametrize(
    "entries",
    [
        {},
        {"bad id": {"kernel_ref": "x@1", "kernel_impl_hash": "1" * 64}},
        {"é": {"kernel_ref": "x@1", "kernel_impl_hash": "1" * 64}},
        {"a": {"kernel_ref": "x@1", "kernel_impl_hash": "Q" * 64}},
    ],
)
def test_invalid_implementation_set(entries):
    with pytest.raises(EvidenceError):
        implementation_set_digest(entries)


def test_invented_export_reads_with_unchanged_writer_and_no_index(tmp_path):
    reference = export_fixture(tmp_path)
    candidate = read_candidate_quality(tmp_path / "quality", reference["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    summary = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["build_manifest"]["path"]
        ).read_bytes()
    )
    assert candidate["execution_status"] == "completed"
    assert candidate["release_eligibility"]["status"] == "excluded"
    assert candidate["population"]["period"] == "2024"
    assert not (tmp_path / "quality/index.json").exists()
    assert "population_sha256" not in summary
    assert (
        summary["graph_receipts_sha256"]
        == candidate["attachments"]["graph_receipts"]["sha256"]
    )
    assert len(graph["implementation_set"]["by_node"]) == 4
    assert (
        len(
            {
                n["kernel_impl_hash"]
                for n in graph["implementation_set"]["by_node"].values()
            }
        )
        == 2
    )
    assert [e["kernel_calls"] for e in graph["executions"]] == [4, 0]
    assert [e["source_preparation_calls"] for e in graph["executions"]] == [1, 0]
    assert len({e["execution_manifest_sha256"] for e in graph["executions"]}) == 2
    assert len({e["computation_content_key"] for e in graph["executions"]}) == 1
    assert all(n["gate"] is None for e in graph["executions"] for n in e["nodes"])
    assert all(
        n["wall_time_seconds"] == 0 and n["seed"] == "0"
        for e in graph["executions"]
        for n in e["nodes"]
    )
    raw = canonical(graph) + canonical(summary) + canonical(candidate)
    for forbidden in (
        b"candidate_manifest_sha256",
        b"source_implementation_sha256",
        b"private-host",
        b".h5",
        b"pid",
        b"inode",
        b"vm_stat",
    ):
        assert forbidden not in raw


@pytest.mark.parametrize(
    "field",
    [
        "direct_checkpoint_table_weight_parity",
        "direct_real_engine_array_parity",
        "money_33_fields_all_axes_parity",
    ],
)
def test_false_recorded_parity_refuses_without_output(tmp_path, field):
    def mutate(root, _run, _declaration):
        mutate_json(
            root, "cold-receipt.json", lambda x: x["result"].update({field: False})
        )

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "case",
    [
        "unknown_schema",
        "missing_phase",
        "source_changed",
        "run_head",
        "runner",
        "roster",
        "approval",
        "period_integer",
        "entity",
        "checkpoint",
        "unsafe_reason",
        "candidate_digest",
        "status_failed",
        "release_claim",
    ],
)
def test_control_identity_refusals_leave_no_candidate(tmp_path, case):
    def mutate(root, run, declaration):
        if case == "unknown_schema":
            run["schema"] = "unreviewed.v2"
        elif case == "missing_phase":
            (root / "cold-receipt.json").unlink()
            run["outputs"].pop("cold-receipt.json")
        elif case == "source_changed":
            run["original_after"] = copy.deepcopy(run["original_after"])
            next(iter(run["original_after"].values()))["sha256"] = "f" * 64
        elif case == "run_head":
            run["head"] = "e" * 40
        elif case in ("runner", "roster", "approval"):
            run[case + "_sha256"] = "f" * 64
        elif case == "period_integer":
            declaration["population"]["period"] = 2024
        elif case == "entity":
            declaration["population"]["entity"] = "person"
        elif case == "checkpoint":
            declaration["population"]["sha256"] = "f" * 64
        elif case == "unsafe_reason":
            declaration["reason"] = "/Users/private/secret"
        elif case == "candidate_digest":
            declaration["candidate_manifest_sha256"] = "f" * 64
        elif case == "status_failed":
            run["status"] = "FAIL"
        elif case == "release_claim":
            run["release_eligible"] = True

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing_node",
        "extra_node",
        "hash_swap",
        "ref_swap",
        "producer",
        "producer_key",
        "typed_key",
        "type_version",
        "unknown_node_field",
        "count",
        "pid",
        "runtime",
        "manifest_key",
        "checkpoint",
        "missing_duration",
    ],
)
def test_graph_and_execution_refusals(tmp_path, case):
    def mutate(root, run, declaration):
        filename = "cold-manifest.json"

        def change(x):
            node = x["nodes"][IDS[1]]
            if case == "missing_node":
                x["nodes"].pop(IDS[0])
            elif case == "extra_node":
                x["nodes"]["source_preparation"] = copy.deepcopy(node)
            elif case == "hash_swap":
                node["kernel_impl_hash"] = "2" * 64
            elif case == "ref_swap":
                node["kernel_ref"] = "us." + IDS[0] + "@1"
            elif case in ("producer", "producer_key", "typed_key", "type_version"):
                item = next(iter(node["typed_artifacts"]["inputs"].values()))
                if case == "type_version":
                    item["type"]["schema_version"] = 2
                else:
                    item[
                        {
                            "producer": "producer",
                            "producer_key": "producer_key",
                            "typed_key": "key",
                        }[case]
                    ] = "f" * 64
            elif case == "unknown_node_field":
                node["raw_person_values"] = [1, 2]
            elif case == "manifest_key":
                x["key"] = run["outputs"][filename]["sha256"]
            elif case == "missing_duration":
                node.pop("wall_time")

        if case in ("count", "pid", "runtime", "checkpoint"):
            filename = "cold-receipt.json"

            def change(x):
                if case == "count":
                    x["result"]["calls"].pop()
                elif case == "pid":
                    x["pid"] = 1
                elif case == "runtime":
                    x["engine_runtime_identity"]["packages"]["policyengine-us"][
                        "version"
                    ] = "changed"
                else:
                    x["result"]["checkpoint_sha256"] = "f" * 64

        mutate_json(root, filename, change)

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()


def test_default_pins_refuse_invented_receipts_and_missing_declaration(tmp_path):
    root, _, _, seal = invented(tmp_path)
    pins = seal()
    for declaration, active_pins in (
        (root / "declaration.json", GENUINE_PINS),
        (root / "absent.json", pins),
    ):
        with pytest.raises(EvidenceError):
            export_graph_evidence(
                run_root=root,
                declaration_path=declaration,
                output_root=tmp_path / "quality",
                pins=active_pins,
            )
    assert not (tmp_path / "quality").exists()


def test_existing_output_is_never_modified(tmp_path):
    root, _, _, seal = invented(tmp_path)
    pins = seal()
    out = tmp_path / "quality"
    out.mkdir()
    (out / "index.json").write_text("preserve")
    with pytest.raises(EvidenceError):
        export_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=out,
            pins=pins,
        )
    assert list(out.iterdir()) == [out / "index.json"]
    assert (out / "index.json").read_text() == "preserve"


def test_stdlib_only_export_in_fresh_process(tmp_path):
    import subprocess
    from dataclasses import asdict

    root, _, _, seal = invented(tmp_path)
    pins = asdict(seal())
    code = r"""
import importlib.abc, importlib.util, json, sys
class BlockExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("microcosm", "policyengine", "numpy", "h5py", "tables", "pandas")):
            raise AssertionError("Output-only import boundary violated: " + fullname)
sys.meta_path.insert(0, BlockExecutionImports())
spec = importlib.util.spec_from_file_location("exporter", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["exporter"] = m; spec.loader.exec_module(m)
pins = json.loads(sys.argv[5]); pins["implementation_hashes"] = tuple(pins["implementation_hashes"])
result = m.export_graph_evidence(run_root=sys.argv[2], declaration_path=sys.argv[3], output_root=sys.argv[4], pins=m.EvidencePins(**pins))
assert not any(name.startswith(("microcosm", "policyengine", "numpy", "h5py", "tables", "pandas")) for name in sys.modules)
print(json.dumps(result))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(SOURCE / "graph_evidence.py"),
            str(root),
            str(root / "declaration.json"),
            str(tmp_path / "quality"),
            json.dumps(pins),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert read_candidate_quality(
        tmp_path / "quality", json.loads(completed.stdout)["sha256"]
    )


def test_receipt_decode_allowlist_and_nullable_duration(tmp_path, monkeypatch):
    decoded = []
    original = evidence._pinned_json

    def track(path, expected):
        decoded.append(path.name)
        return original(path, expected)

    monkeypatch.setattr(evidence, "_pinned_json", track)

    def mutate(root, _run, _declaration):
        for mode in ("cold", "warm"):
            mutate_json(
                root,
                f"{mode}-manifest.json",
                lambda x: x["nodes"][IDS[0]].update(wall_time=None),
            )

    ref = export_fixture(tmp_path, mutate)
    assert set(decoded) == {
        "run-receipt.json",
        "declaration.json",
        "direct-receipt.json",
        "cold-receipt.json",
        "warm-receipt.json",
        "cold-manifest.json",
        "warm-manifest.json",
    }
    candidate = read_candidate_quality(tmp_path / "quality", ref["sha256"])
    graph = json.loads(
        (
            tmp_path / "quality" / candidate["attachments"]["graph_receipts"]["path"]
        ).read_bytes()
    )
    assert graph["executions"][0]["nodes"][0]["wall_time_seconds"] is None


@pytest.mark.parametrize(
    "case",
    [
        "typed_key",
        "producer",
        "producer_key",
        "type_version",
        "ref_swap",
        "hash_swap",
        "empty_input",
    ],
)
def test_resealed_graph_contradictions_reach_structural_verifier(tmp_path, case):
    root, run, _declaration, seal = invented(tmp_path)
    manifest = json.loads((root / "cold-manifest.json").read_bytes())
    node = manifest["nodes"][IDS[1]]
    item = next(iter(node["typed_artifacts"]["inputs"].values()))
    if case == "typed_key":
        item["key"] = "f" * 64
    elif case == "producer":
        item["producer"] = IDS[3]
    elif case == "producer_key":
        item["producer_key"] = "f" * 64
    elif case == "type_version":
        item["type"]["schema_version"] = 3
    elif case == "ref_swap":
        node["kernel_ref"] = "us." + IDS[0] + "@1"
    elif case == "hash_swap":
        node["kernel_impl_hash"] = "2" * 64
    else:
        item.clear()
    manifest["content_addressed"]["nodes"] = {
        key: {
            field: value
            for field, value in n.items()
            if field not in ("hit", "wall_time")
        }
        for key, n in manifest["nodes"].items()
    }
    key = digest(b"manifest\0" + canonical(manifest["content_addressed"]))
    manifest["key"] = key
    (root / "cold-manifest.json").write_bytes(canonical(manifest) + b"\n")
    mutate_json(root, "cold-receipt.json", lambda x: x["result"].update(key=key))
    with pytest.raises(EvidenceError):
        export_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "quality",
            pins=seal(),
        )
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(
    "case",
    [
        "direct_count",
        "cold_count",
        "run_count",
        "exit_code",
        "node_rejected",
        "node_failed",
        "node_unreached",
        "opaque_changed",
    ],
)
def test_count_types_and_contradictory_receipt_states(tmp_path, case):
    root, run, _declaration, seal = invented(tmp_path)
    if case in ("direct_count", "cold_count"):
        mode = case.split("_")[0]
        mutate_json(
            root,
            f"{mode}-receipt.json",
            lambda x: x["result"].update(source_preparations=True),
        )
    elif case == "run_count":
        run["source_preparations_verified"] = True
    elif case == "exit_code":
        run["stages"]["cold"]["exit_code"] = False
    elif case == "opaque_changed":
        (root / f"warm-{IDS[0]}-current_money.bin").write_bytes(
            b"altered opaque output"
        )
    else:
        manifest = json.loads((root / "cold-manifest.json").read_bytes())
        receipt = manifest["nodes"][IDS[0]]["receipt"]
        if case == "node_rejected":
            receipt["rejected"] = True
        elif case == "node_failed":
            receipt["outcome"] = "fail"
        else:
            receipt["execution"] = {
                "schema": "microcosm.graph.execution.v1",
                "state": "unreached",
            }
        manifest["content_addressed"]["nodes"] = {
            key: {
                field: value
                for field, value in n.items()
                if field not in ("hit", "wall_time")
            }
            for key, n in manifest["nodes"].items()
        }
        key = digest(b"manifest\0" + canonical(manifest["content_addressed"]))
        manifest["key"] = key
        (root / "cold-manifest.json").write_bytes(canonical(manifest) + b"\n")
        mutate_json(root, "cold-receipt.json", lambda x: x["result"].update(key=key))
    with pytest.raises(EvidenceError):
        export_graph_evidence(
            run_root=root,
            declaration_path=root / "declaration.json",
            output_root=tmp_path / "quality",
            pins=seal(),
        )
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize("mode", ["direct", "cold"])
def test_evaluation_digest_contradictions_refuse(tmp_path, mode):
    def mutate(root, _run, _declaration):
        mutate_json(
            root,
            f"{mode}-receipt.json",
            lambda x: x["result"].update(evaluation_sha256="f" * 64),
        )

    with pytest.raises(EvidenceError):
        export_fixture(tmp_path, mutate)
    assert not (tmp_path / "quality").exists()
