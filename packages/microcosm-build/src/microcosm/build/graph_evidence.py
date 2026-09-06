"""Pure output-only export of the explicitly reviewed first genuine graph run.

Run directly by file path to avoid the build package's eager imports. Only pinned
aggregate receipts and execution manifests are decoded. Named output bytes are
hashed opaquely; no source storage, numerical arrays or execution kernels are read.
Recorded parity remains a producer attestation, never a scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 4 * 1024 * 1024
IDS = tuple(
    f"asec_prepared.{name}"
    for name in (
        "create",
        "select_households",
        "cps_carried_current",
        "engine_current_money",
    )
)
LABELS = (
    "Prepare current money sources",
    "Select whole households",
    "Carry current money leaves",
    "Evaluate four engine roots",
)
SOURCES = {
    **{
        f"asec_household_cohort_{year}.h5": f"asec_cohort_{year}"
        for year in (2022, 2023, 2024)
    },
    "asec_household_observations.checkpoint.h5": "asec_household_observations",
    "asec_person_income_source.checkpoint.h5": "asec_person_income_source",
    "asec_raw_stage.checkpoint.h5": "asec_raw_stage",
    "pppub23.csv": "asec_raw_2023",
    "pppub24.csv": "asec_raw_2024",
    "pppub25.csv": "asec_raw_2025",
    "restoration.receipt.json": "asec_restoration_receipt",
}
ARTIFACT_NAMES = (
    ("current_money", "frame_context", "prepared_receipt"),
    ("frame_context", "selected_current_money", "selection"),
    ("frame_context",),
    ("engine_evaluation",),
)
RECEIPT_METADATA = (
    {"capabilities", "current_money_consumers", "implementation", "prepared"},
    {
        "capabilities",
        "implementation",
        "mass",
        "selected_money_sha256",
        "selection_sha256",
    },
    {
        "capabilities",
        "contract",
        "implementation",
        "person_rows",
        "selected_money_sha256",
        "selection_sha256",
        "spm_unit_rows",
    },
    {
        "capabilities",
        "cells_written",
        "claim",
        "engine",
        "evaluation_sha256",
        "implementation",
    },
)

ROOTS = ("capital_gains", "dividend_income", "employment_income", "social_security")
PARITY = (
    "direct_checkpoint_table_weight_parity",
    "direct_real_engine_array_parity",
    "money_33_fields_all_axes_parity",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,159}\Z")
_KERNEL = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:@-]{0,159}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class EvidenceError(ValueError):
    """The bounded recorded evidence cannot support this export."""


@dataclass(frozen=True)
class EvidencePins:
    """Expected identities; alternative pins are for explicitly invented tests."""

    run_receipt_sha256: str
    operator_declaration_sha256: str
    run_head: str
    runner_sha256: str
    roster_sha256: str
    approval_sha256: str
    implementation_hashes: tuple[str, str]


GENUINE_PINS = EvidencePins(
    "68fedbc5e4de11d672e75e59d0397605cee0a0d21595226a51b2b1440a493812",
    "d4d1e05f5d198ef676b8a7fd1ca7aba3f515a95def710bde38d1f0fc2c1df5b5",
    "d4bddcb6ac2aba52903cd8dace90330cee4a21e1",
    "278abdc72a8c167d07195c2893b6682ac2f78aaf6b9867ef318e555d0ff659b0",
    "760fb7dfd391ecaa1e02e971a88c724305b565db344f4bbe01a2eb8ff5ad7bed",
    "f1db72de94aaac043bccf72936e8f7866407dd53532479c4ca21b99029a8106e",
    (
        "afd8031361cef780fb90880440320d4a1bb5c5a5d7caddd66a51f15126408679",
        "c36e9704508e093ff536862ca6b28f4fb1a941dbf05fc95f8ef4e5159cd78c6b",
    ),
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise EvidenceError(reason)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def obj(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    require(isinstance(value, dict), "Expected a supported receipt object.")
    require(
        required <= value.keys() <= required | (optional or set()),
        "Missing or unknown receipt fields.",
    )
    return value


def digest(value: Any) -> str:
    require(
        isinstance(value, str) and _SHA.fullmatch(value) is not None, "Invalid digest."
    )
    return value


def identifier(value: Any, *, kernel: bool = False) -> str:
    require(
        isinstance(value, str)
        and (_KERNEL if kernel else _IDENTIFIER).fullmatch(value) is not None,
        "Invalid bounded identifier.",
    )
    return value


def safe_text(value: Any, maximum: int = 2048) -> str:
    require(
        isinstance(value, str) and 1 <= len(value) <= maximum,
        "Invalid bounded presentation text.",
    )
    require(
        not any(ord(c) < 32 or ord(c) == 127 for c in value)
        and not any(
            s in value.lower()
            for s in ("/", "\\", ".local", ".h5", "macbook", "/users/")
        ),
        "Unsafe presentation text.",
    )
    return value


def count(value: Any) -> int:
    require(type(value) is int and 0 <= value <= 2**53 - 1, "Invalid aggregate count.")
    return value


def duration(value: Any) -> int | float | None:
    if value is None:
        return None
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "Invalid aggregate duration or tolerance.",
    )
    return value


def bounded_tree(value: Any, depth: int = 0, *, output: bool = False) -> None:
    require(depth <= 16, "Receipt nesting exceeds the bound.")
    if isinstance(value, dict):
        require(
            len(value) <= (1024 if output else 4096), "Receipt map exceeds the bound."
        )
        require(
            "candidate_manifest_sha256" not in value,
            "Circular candidate digest is forbidden.",
        )
        for key, child in value.items():
            require(isinstance(key, str) and len(key) <= 512, "Invalid receipt key.")
            bounded_tree(child, depth + 1, output=output)
    elif isinstance(value, list):
        require(
            len(value) <= (128 if output else 4096), "Receipt list exceeds the bound."
        )
        for child in value:
            bounded_tree(child, depth + 1, output=output)
    elif isinstance(value, str):
        require(
            len(value) <= (2048 if output else 65536), "Receipt text exceeds the bound."
        )
        if output:
            safe_text(value)
    elif type(value) in (float, int):
        require(math.isfinite(value), "Nonfinite receipt number.")
    else:
        require(value is None or type(value) is bool, "Unsupported JSON value.")


def _stable_stat(before, after) -> bool:
    return all(
        getattr(before, key) == getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _file_bytes(path: Path, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
                "Evidence input must be a bounded regular file.",
            )
            raw = stream.read(maximum + 1)
            require(
                len(raw) == before.st_size
                and _stable_stat(before, os.fstat(stream.fileno())),
                "Evidence changed during reading.",
            )
            return raw
    except OSError as exc:
        raise EvidenceError("Required evidence input is unreadable.") from exc


def _json(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        bounded_tree(value)
        require(isinstance(value, dict), "Receipt must be an object.")
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise EvidenceError("Invalid bounded receipt JSON.") from exc


def _pinned_json(path: Path, expected: str) -> dict:
    raw = _file_bytes(path)
    require(sha256(raw) == digest(expected), "Pinned receipt digest mismatch.")
    return _json(raw)


def _opaque_hash(path: Path, expected: dict) -> None:
    """Stream output bytes for integrity only, without importing any codec."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size == expected["bytes"],
                "Opaque output size mismatch.",
            )
            h = hashlib.sha256()
            while data := stream.read(1024 * 1024):
                h.update(data)
            require(
                _stable_stat(before, os.fstat(stream.fileno()))
                and h.hexdigest() == expected["sha256"],
                "Opaque output digest mismatch.",
            )
    except OSError as exc:
        raise EvidenceError("Required opaque output is unreadable.") from exc


def implementation_set_digest(by_node: dict) -> str:
    require(
        isinstance(by_node, dict) and 1 <= len(by_node) <= 64,
        "Implementation map must contain a bounded node roster.",
    )
    for node_id, entry in by_node.items():
        identifier(node_id)
        obj(entry, {"kernel_ref", "kernel_impl_hash"})
        identifier(entry["kernel_ref"], kernel=True)
        digest(entry["kernel_impl_hash"])
    return sha256(
        "".join(
            f"{node_id}\t{by_node[node_id]['kernel_impl_hash']}\n"
            for node_id in sorted(by_node)
        ).encode("ascii")
    )


def _runtime_identity(value: dict) -> str:
    obj(value, {"packages", "policy"})
    require(
        value["policy"] == "installed_baseline_no_reforms",
        "Unsupported runtime policy.",
    )
    obj(value["packages"], {"policyengine-us", "policyengine-core"})
    for package in value["packages"].values():
        obj(package, {"files", "version", "source_and_parameters_sha256"})
        count(package["files"])
        identifier(package["version"])
        digest(package["source_and_parameters_sha256"])
    return sha256(canonical_json(value))


def _source_identity(run: dict) -> dict:
    staging = obj(
        run["staging"],
        {"original_before", "original_after", "staged"},
        {"logical_source_bytes", "method"},
    )
    before = staging["original_before"]
    require(
        set(staging["staged"]) == set(SOURCES), "Staged source receipt roster mismatch."
    )
    require(set(before) == set(SOURCES), "Source receipt roster mismatch.")
    require(
        before == staging["original_after"] == run["original_after"]
        and staging["staged"] == run["staged_after"],
        "Recorded source mutation.",
    )
    for name in SOURCES:
        original = obj(
            before[name], {"sha256", "bytes", "device", "inode", "mtime_ns", "ctime_ns"}
        )
        staged = obj(staging["staged"][name], set(original))
        digest(original["sha256"])
        for field in ("bytes", "device", "inode", "mtime_ns", "ctime_ns"):
            require(
                type(original[field]) is int and original[field] >= 0,
                "Invalid source receipt counter.",
            )
        require(
            original["sha256"] == staged["sha256"]
            and original["bytes"] == staged["bytes"],
            "Staged source content mismatch.",
        )
    return {SOURCES[name]: before[name]["sha256"] for name in sorted(SOURCES)}


def _typed(value: dict, name: str, *, is_input: bool) -> dict:
    obj(value, {"artifact", "key", "numerics", "producer", "producer_key", "type"})
    obj(value["type"], {"name", "schema_version"})
    obj(value["numerics"], {"numeric", "platform", "tolerance"})
    version = count(value["type"]["schema_version"])
    require(version > 0, "Invalid typed schema version.")
    result = {
        "name": identifier(name),
        "artifact": identifier(value["artifact"]),
        "type": identifier(value["type"]["name"]),
        "type_version": version,
        "artifact_key": digest(value["key"]),
        "numeric": identifier(value["numerics"]["numeric"]),
        "tolerance": duration(value["numerics"]["tolerance"]),
    }
    identifier(value["producer"])
    digest(value["producer_key"])
    if is_input:
        result.update(producer=value["producer"], producer_key=value["producer_key"])
    return result


def _nodes(manifest: dict, pins: EvidencePins, *, warm: bool) -> list[dict]:
    raw_nodes = manifest["nodes"]
    require(
        isinstance(raw_nodes, dict) and set(raw_nodes) == set(IDS),
        "Exact graph node roster mismatch.",
    )
    result = []
    output_keys = set()
    for i, node_id in enumerate(IDS):
        n = obj(
            raw_nodes[node_id],
            {
                "artifacts",
                "capabilities",
                "frame_key",
                "hit",
                "kernel_impl_hash",
                "kernel_ref",
                "key",
                "legacy_capabilities",
                "opaque_artifacts",
                "receipt",
                "seed",
                "typed_artifacts",
                "wall_time",
                "weight_key",
            },
        )
        require(
            n["hit"] is warm and n["legacy_capabilities"] is False,
            "Cache evidence contradicts phase.",
        )
        require(
            n["kernel_ref"] == f"us.{node_id}@1"
            and n["kernel_impl_hash"] == pins.implementation_hashes[int(i == 3)],
            "Node implementation identity mismatch.",
        )
        cap = obj(
            n["capabilities"],
            {
                "consumes_se",
                "dependencies",
                "determinism",
                "numeric",
                "role",
                "seed_source",
                "structural",
                "tolerance",
            },
        )
        require(
            cap["role"] == "compute"
            and cap["determinism"] == ("seeded" if i == 1 else "deterministic")
            and cap["numeric"] == "platform_bitwise"
            and cap["consumes_se"] is False
            and cap["structural"] == ("create", "filter", "none", "none")[i]
            and cap["seed_source"] == ("param" if i == 1 else "none")
            and cap["tolerance"] is None,
            "Unsupported node capabilities.",
        )
        require(
            isinstance(cap["dependencies"], list)
            and len(cap["dependencies"]) <= 64
            and len(set(cap["dependencies"])) == len(cap["dependencies"]),
            "Invalid dependency list.",
        )
        dependencies = [identifier(v) for v in cap["dependencies"]]
        require(
            isinstance(n["receipt"], dict)
            and n["receipt"].get("release_eligible") is False,
            "Missing or contradictory node receipt.",
        )
        obj(n["receipt"], {"phase", "release_eligible"}, RECEIPT_METADATA[i])
        phase = identifier(n["receipt"].get("phase"))
        require(
            phase == "prepare_asec_current_money_slice",
            "Unsupported node receipt phase.",
        )
        typed = obj(n["typed_artifacts"], {"inputs", "outputs"})
        require(
            all(
                isinstance(typed[k], dict) and len(typed[k]) <= 128
                for k in ("inputs", "outputs")
            ),
            "Typed artifact map exceeds bounds.",
        )
        require(
            set(typed["outputs"]) == set(ARTIFACT_NAMES[i])
            and set(n["opaque_artifacts"]) == set(typed["outputs"]),
            "Typed output roster mismatch.",
        )
        for name, artifact in typed["outputs"].items():
            require(
                artifact["producer"] == node_id
                and artifact["producer_key"] == n["key"]
                and artifact["artifact"] == name
                and artifact["key"] == n["opaque_artifacts"][name],
                "Typed output identity mismatch.",
            )
            require(artifact["key"] not in output_keys, "Duplicate typed output key.")
            output_keys.add(artifact["key"])
        for name, artifact in typed["inputs"].items():
            producer = raw_nodes.get(artifact.get("producer"))
            require(
                producer is not None and IDS.index(artifact["producer"]) < i,
                "Missing, cyclic or unreached typed producer.",
            )
            source = producer["typed_artifacts"]["outputs"].get(
                artifact.get("artifact")
            )
            require(
                source == artifact and artifact["producer_key"] == producer["key"],
                "Typed input differs from its actual producer.",
            )
            identifier(name)
        require(
            type(n["seed"]) is int and len(str(n["seed"])) <= 32, "Invalid exact seed."
        )
        require(
            isinstance(n["artifacts"], list) and len(n["artifacts"]) <= 4096,
            "Invalid column artifacts.",
        )
        columns = Counter()
        column_ids = set()
        for column in n["artifacts"]:
            obj(column, {"entity", "column", "key"})
            entity, name = identifier(column["entity"]), identifier(column["column"])
            digest(column["key"])
            require((entity, name) not in column_ids, "Duplicate column artifact.")
            column_ids.add((entity, name))
            columns[entity] += 1
        require(len(columns) <= 64, "Column entity count exceeds bound.")
        result.append(
            dict(
                id=node_id,
                label=LABELS[i],
                group="engineering",
                group_label="Engineering graph",
                role="compute",
                receipt_present=True,
                failed=False,
                blocked=False,
                gate_exception=False,
                execution_record=None,
                receipt_outcome=None,
                cache="hit" if warm else "computed",
                node_key=digest(n["key"]),
                kernel_ref=identifier(n["kernel_ref"], kernel=True),
                kernel_impl_hash=digest(n["kernel_impl_hash"]),
                determinism=cap["determinism"],
                numeric=cap["numeric"],
                tolerance=None,
                runtime_dependencies=dependencies,
                seed=str(n["seed"]),
                wall_time_seconds=duration(n["wall_time"]),
                frame_key=None if n["frame_key"] is None else digest(n["frame_key"]),
                weight_key=None if n["weight_key"] is None else digest(n["weight_key"]),
                phase=phase,
                typed_inputs=[
                    _typed(v, k, is_input=True)
                    for k, v in sorted(typed["inputs"].items())
                ],
                typed_outputs=[
                    _typed(v, k, is_input=False)
                    for k, v in sorted(typed["outputs"].items())
                ],
                opaque_artifact_keys=[
                    {"name": identifier(k), "key": digest(v)}
                    for k, v in sorted(n["opaque_artifacts"].items())
                ],
                column_artifact_counts=[
                    {"entity": e, "count": count(c)} for e, c in sorted(columns.items())
                ],
                column_artifact_total=len(n["artifacts"]),
                gate=None,
                source_pins=[],
                measures=[],
            )
        )
    return result


def _load_writer():
    # File loading deliberately avoids microcosm.build.__init__ and us_runtime.
    path = Path(__file__).with_name("candidate_quality.py")
    spec = importlib.util.spec_from_file_location(
        "_graph_evidence_candidate_quality", path
    )
    require(spec is not None and spec.loader is not None, "Writer module unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_graph_evidence(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: EvidencePins = GENUINE_PINS,
) -> dict[str, str]:
    """Authenticate recorded output evidence, then create one exclusive candidate.

    The default pins authorize only G and its exact operator declaration. Tests
    may supply explicit invented pins. Neither path discovers or updates an index.
    All evidence validation completes before creating the output directory.
    """
    run_root, output_root = Path(run_root), Path(output_root)
    require(
        not output_root.exists() and not output_root.is_symlink(),
        "Output root must be newly created.",
    )
    run = _pinned_json(run_root / "run-receipt.json", pins.run_receipt_sha256)
    declaration = _pinned_json(Path(declaration_path), pins.operator_declaration_sha256)
    obj(
        declaration,
        {
            "schema",
            "country",
            "candidate_release_id",
            "run_id",
            "population",
            "basis",
            "reason",
            "run_receipt_sha256",
            "created_at",
        },
    )
    require(
        declaration["schema"] == "microcosm.engineering_candidate_declaration.v1"
        and declaration["country"] == "us"
        and declaration["run_receipt_sha256"] == pins.run_receipt_sha256,
        "Operator declaration identity mismatch.",
    )
    population = obj(declaration["population"], {"period", "entity", "sha256"})
    require(
        population["period"] == "2024" and population["entity"] == "household",
        "Exact declared target period and entity are required.",
    )
    checkpoint = digest(population["sha256"])
    identifier(declaration["run_id"])
    candidate_id = declaration["candidate_release_id"]
    require(
        isinstance(candidate_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,299}", candidate_id) is not None,
        "Invalid declared candidate id.",
    )
    require(
        declaration["basis"] == "operator_declared_target_period_and_population_entity",
        "Unsupported period basis.",
    )
    safe_text(declaration["reason"])
    required = {
        "schema",
        "status",
        "phase",
        "head",
        "runner_sha256",
        "roster_sha256",
        "approval_sha256",
        "calibration",
        "fit",
        "holdouts",
        "population_tax_benefit_claim",
        "prior_wages_added",
        "reforms",
        "release_eligible",
        "source_preparations_planned",
        "source_preparations_verified",
        "original_after",
        "staged_after",
        "staging",
        "stages",
        "outputs",
    }
    obj(run, required, {"final_resources", "observed_capacity", "resources", "seconds"})
    require(
        run["schema"] == "microcosm.first_genuine_graph_run.v1"
        and run["status"] == "PASS"
        and run["phase"] == "source_postflight",
        "Unsupported or incomplete genuine run.",
    )
    for field in (
        "calibration",
        "fit",
        "holdouts",
        "population_tax_benefit_claim",
        "prior_wages_added",
        "reforms",
        "release_eligible",
    ):
        require(run[field] is False, "Run contradicts the excluded engineering scope.")
    for field, expected in (
        ("head", pins.run_head),
        ("runner_sha256", pins.runner_sha256),
        ("roster_sha256", pins.roster_sha256),
        ("approval_sha256", pins.approval_sha256),
    ):
        require(run[field] == expected, "Frozen run identity mismatch.")
    require(
        count(run["source_preparations_planned"])
        == count(run["source_preparations_verified"])
        == 2,
        "Recorded source preparation total mismatch.",
    )
    obj(run["stages"], {"direct", "cold", "warm"})
    require(
        all(
            type(v.get("exit_code")) is int and v["exit_code"] == 0
            for v in run["stages"].values()
        ),
        "A producer process failed.",
    )
    sources = _source_identity(run)
    output_records = run["outputs"]
    required_outputs = {f"{mode}-receipt.json" for mode in ("direct", "cold", "warm")}
    required_outputs |= {f"{mode}-manifest.json" for mode in ("cold", "warm")}
    required_outputs |= {f"{mode}-frame.h5" for mode in ("direct", "cold", "warm")}
    required_outputs |= {
        f"{mode}-{node_id}-{name}.bin"
        for mode in ("cold", "warm")
        for node_id, names in zip(IDS, ARTIFACT_NAMES, strict=True)
        for name in names
    }
    optional_outputs = {
        "columns.json",
        "preflight.json",
        "source-staging.json",
        "direct-context.bin",
        "direct-evaluation.bin",
        "direct-money.bin",
        "direct-positions.npz",
        "direct-prepared.json",
        "direct-sampler.json",
        "direct-selected-money.bin",
        "cold-aggregate-quality.json",
        "warm-aggregate-quality.json",
    }
    obj(output_records, required_outputs, optional_outputs)
    for item in output_records.values():
        obj(item, {"bytes", "sha256"})
        count(item["bytes"])
        digest(item["sha256"])
    for name in sorted(required_outputs):
        _opaque_hash(run_root / name, output_records[name])
    for node_id, names in zip(IDS, ARTIFACT_NAMES, strict=True):
        for name in names:
            require(
                output_records[f"cold-{node_id}-{name}.bin"]
                == output_records[f"warm-{node_id}-{name}.bin"],
                "Opaque graph output bytes differ between cold and warm.",
            )
    for mode in ("direct", "cold", "warm"):
        require(
            output_records[f"{mode}-frame.h5"]["sha256"] == checkpoint,
            "Final output checkpoints differ.",
        )
    receipts = {
        m: _pinned_json(
            run_root / f"{m}-receipt.json",
            output_records[f"{m}-receipt.json"]["sha256"],
        )
        for m in ("direct", "cold", "warm")
    }
    runtime_hashes, pids, loaded_sets = set(), set(), []
    for mode, receipt in receipts.items():
        obj(
            receipt,
            {
                "schema",
                "head",
                "phase",
                "pid",
                "status",
                "release_eligible",
                "result",
                "engine_runtime_identity",
                "loaded_modules",
                "seconds",
            },
            {"final_resources", "initial_resources", "hdf5_version", "library_output"},
        )
        require(
            receipt["schema"] == "microcosm.first_genuine_graph_worker.v1"
            and receipt["head"] == pins.run_head
            and receipt["phase"] == mode
            and receipt["status"] == "PASS"
            and receipt["release_eligible"] is False,
            "Phase receipt identity or success mismatch.",
        )
        pids.add(count(receipt["pid"]))
        runtime_hashes.add(_runtime_identity(receipt["engine_runtime_identity"]))
        loaded = receipt["loaded_modules"]
        require(
            isinstance(loaded, dict) and 1 <= len(loaded) <= 1024,
            "Loaded module roster missing.",
        )
        for name, module in loaded.items():
            identifier(name)
            obj(module, {"git_blob_oid", "matches_head_blob", "path", "sha256"})
            require(
                module["matches_head_blob"] is True
                and re.fullmatch(r"[a-f0-9]{40}", module["git_blob_oid"]) is not None,
                "Loaded source differs from recorded head.",
            )
            digest(module["sha256"])
        loaded_sets.append(loaded)
    require(
        len(pids) == 3
        and 0 not in pids
        and len(runtime_hashes) == 1
        and loaded_sets[0] == loaded_sets[1] == loaded_sets[2],
        "Distinct phase executions and identical loaded runtime are required.",
    )
    direct = receipts["direct"]["result"]
    obj(
        direct,
        {
            "entity_rows",
            "whole_households",
            "original_design_weight_bytes_preserved",
            "source_preparations",
            "roots",
            "selected_field_axes_sha256",
        },
        {
            "eligible_households_by_cohort",
            "evaluation_sha256",
            "prepared_receipt_sha256",
            "realized_households_by_cohort",
        },
    )
    require(
        direct["whole_households"] is True
        and direct["original_design_weight_bytes_preserved"] is True
        and count(direct["source_preparations"]) == 1
        and direct["roots"] == list(ROOTS),
        "Direct engineering attestation mismatch.",
    )
    require(
        set(direct["entity_rows"])
        <= {"household", "person", "family", "spm_unit", "tax_unit", "marital_unit"}
        and {"household", "person"} <= set(direct["entity_rows"]),
        "Unsupported entity counts.",
    )
    for value in direct["entity_rows"].values():
        count(value)
    require(
        direct["entity_rows"]["household"] == 15
        and direct["entity_rows"]["person"] == 52,
        "Unsupported engineering slice size.",
    )
    require(
        len(direct["selected_field_axes_sha256"]) == 33,
        "Required measurement axes missing.",
    )
    for name, axes in direct["selected_field_axes_sha256"].items():
        identifier(name)
        obj(axes, {"amounts", "statuses", "validity", "zero_origin"})
        for value in axes.values():
            digest(value)
    if "evaluation_sha256" in direct:
        require(
            "direct-evaluation.bin" in output_records
            and digest(direct["evaluation_sha256"])
            == output_records["direct-evaluation.bin"]["sha256"],
            "Direct evaluation output identity mismatch.",
        )
        _opaque_hash(
            run_root / "direct-evaluation.bin", output_records["direct-evaluation.bin"]
        )
    if "prepared_receipt_sha256" in direct:
        require(
            "direct-prepared.json" in output_records
            and digest(direct["prepared_receipt_sha256"])
            == output_records["direct-prepared.json"]["sha256"],
            "Direct prepared receipt identity mismatch.",
        )
        _opaque_hash(
            run_root / "direct-prepared.json", output_records["direct-prepared.json"]
        )
    executions, checks, content_keys, manifest_digests = [], {}, set(), set()
    checks["direct"] = dict(
        population_sha256=checkpoint,
        receipt_sha256=output_records["direct-receipt.json"]["sha256"],
        household_count=15,
        person_count=52,
        source_preparation_calls=1,
    )
    by_node = None
    for mode in ("cold", "warm"):
        warm = mode == "warm"
        recorded = receipts[mode]["result"]
        obj(
            recorded,
            {
                "all_hits",
                "artifacts",
                "calls",
                "checkpoint_sha256",
                *PARITY,
                "key",
                "nodes",
                "source_preparations",
            },
            {"evaluation_sha256"},
        )
        require(
            all(recorded[field] is True for field in PARITY),
            "Recorded engineering parity is false or absent.",
        )
        require(
            recorded["checkpoint_sha256"] == checkpoint
            and recorded["all_hits"] is warm
            and count(recorded["source_preparations"]) == int(not warm)
            and recorded["calls"]
            == ([] if warm else [IDS[0], "source_preparation", *IDS[1:]]),
            "Graph call or checkpoint evidence contradicts the phase.",
        )
        manifest_digest = output_records[f"{mode}-manifest.json"]["sha256"]
        manifest = _pinned_json(run_root / f"{mode}-manifest.json", manifest_digest)
        obj(
            manifest,
            {
                "content_addressed",
                "country",
                "decisions",
                "finished_at",
                "host",
                "key",
                "known_failures",
                "nodes",
                "schema_version",
                "started_at",
                "tier",
            },
        )
        require(
            type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 3
            and manifest["country"] == "us"
            and manifest["tier"] is None
            and manifest["known_failures"] == []
            and manifest["decisions"] == [],
            "Unsupported execution manifest or failures.",
        )
        obj(manifest["content_addressed"], {"nodes", "tier"})
        require(
            manifest["content_addressed"]
            == {
                "tier": None,
                "nodes": {
                    k: {
                        field: value
                        for field, value in node.items()
                        if field not in {"hit", "wall_time"}
                    }
                    for k, node in manifest["nodes"].items()
                },
            },
            "Content body differs from execution nodes.",
        )
        content_key = sha256(
            b"manifest\0" + canonical_json(manifest["content_addressed"])
        )
        require(
            content_key == manifest["key"] == recorded["key"]
            and content_key != manifest_digest,
            "Computation content key and execution byte digest are distinct identities.",
        )
        nodes = _nodes(manifest, pins, warm=warm)
        engine_receipt = manifest["nodes"][IDS[3]]["receipt"]
        if "evaluation_sha256" in recorded or "evaluation_sha256" in engine_receipt:
            expected_evaluation = output_records[
                f"{mode}-{IDS[3]}-engine_evaluation.bin"
            ]["sha256"]
            require(
                recorded.get("evaluation_sha256")
                == engine_receipt.get("evaluation_sha256")
                == expected_evaluation,
                "Graph engine evaluation output identity mismatch.",
            )

        require(
            recorded["nodes"] == {n["id"]: n["node_key"] for n in nodes}
            and recorded["artifacts"]
            == {
                n["id"]: {a["name"]: a["key"] for a in n["opaque_artifact_keys"]}
                for n in nodes
            },
            "Recorded node or artifact keys differ from manifest.",
        )
        mapping = {
            n["id"]: {
                "kernel_ref": n["kernel_ref"],
                "kernel_impl_hash": n["kernel_impl_hash"],
            }
            for n in nodes
        }
        require(
            by_node is None or by_node == mapping,
            "Execution implementation maps differ.",
        )
        by_node = mapping
        kernel_calls = sum(n["cache"] == "computed" for n in nodes)
        require(
            kernel_calls == (0 if warm else 4),
            "Computed count differs from kernel calls.",
        )
        executions.append(
            dict(
                mode=mode,
                label="Fresh warm reuse" if warm else "Cold graph execution",
                description="Recorded complete cache reuse in a distinct process."
                if warm
                else "Recorded execution of four graph kernels and one separate source preparation.",
                execution_manifest_sha256=manifest_digest,
                computation_content_key=content_key,
                kernel_calls=kernel_calls,
                source_preparation_calls=int(not warm),
                computed_count=kernel_calls,
                cache_hit_count=4 - kernel_calls,
                failed_node_ids=[],
                blocked_node_ids=[],
                gate_exception_node_ids=[],
                nodes=nodes,
            )
        )
        checks[mode] = dict(
            population_sha256=checkpoint,
            receipt_sha256=output_records[f"{mode}-receipt.json"]["sha256"],
            execution_manifest_sha256=manifest_digest,
            computation_content_key=content_key,
            kernel_calls=kernel_calls,
            source_preparation_calls=int(not warm),
            **{key: recorded[key] for key in PARITY},
        )
        content_keys.add(content_key)
        manifest_digests.add(manifest_digest)
    require(
        len(content_keys) == 1 and len(manifest_digests) == 2,
        "Warm computation identity or execution distinction failed.",
    )
    implementation_set = {
        "sha256": implementation_set_digest(by_node),
        "by_node": by_node,
    }
    identity = dict(
        run_head=pins.run_head,
        run_receipt_sha256=pins.run_receipt_sha256,
        runner_sha256=pins.runner_sha256,
        roster_sha256=pins.roster_sha256,
        approval_sha256=pins.approval_sha256,
        exporter_sha256=sha256(_file_bytes(Path(__file__))),
        operator_declaration_sha256=pins.operator_declaration_sha256,
        runtime_identity_sha256=next(iter(runtime_hashes)),
    )
    common = dict(
        country="us",
        candidate_release_id=candidate_id,
        run_id=declaration["run_id"],
        **identity,
        implementation_set=implementation_set,
        source_sha256=sources,
    )
    graph = dict(
        schema_version="microcosm.graph_receipts.v1",
        **common,
        population_sha256=checkpoint,
        period=population["period"],
        entity=population["entity"],
        period_basis=declaration["basis"],
        period_reason=declaration["reason"],
        computation_content_key=next(iter(content_keys)),
        provenance_note="Authenticated publisher aggregate attachment. Exporter verifies recorded producer parity attestations without recomputing arrays. Original execution manifest digests are pointers; the browser does not independently interpret full raw execution receipts.",
        label_basis="Presentation labels are declared metadata. Edges follow actual typed producer and input references only.",
        coverage={
            "basis": "recorded_execution_roster",
            "complete": True,
            "reason": "All four recorded graph nodes are included. Source preparation is a separate counter. No scientific gate or dependency closure claim is inferred from parity or cache equality.",
        },
        content_key_note="Shared computation content key is distinct from each exact execution manifest byte digest.",
        executions=executions,
    )
    bounded_tree(graph, output=True)
    graph_raw = canonical_json(graph) + b"\n"
    summary = dict(
        schema_version="microcosm.graph_build_summary.v1",
        **common,
        graph_receipts_sha256=sha256(graph_raw),
        checks=checks,
    )
    bounded_tree(summary, output=True)
    summary_raw = canonical_json(summary) + b"\n"
    require(
        max(len(graph_raw), len(summary_raw)) <= MAX_JSON_BYTES,
        "Export attachment exceeds size bound.",
    )
    metadata = dict(
        country="us",
        candidate_release_id=candidate_id,
        run_id=declaration["run_id"],
        created_at=declaration["created_at"],
        execution_status="completed",
        release_eligibility={"status": "excluded", "reason": declaration["reason"]},
        population=dict(population),
        provenance={
            **identity,
            "implementation_set_sha256": implementation_set["sha256"],
        },
    )
    writer = _load_writer()
    # Preflight writer's unmodified metadata validator before reserving output.
    writer._metadata(metadata)
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise EvidenceError("Exclusive output root cannot be created.") from exc
    with tempfile.TemporaryDirectory(prefix="graph-evidence-attachments-") as temporary:
        temporary = Path(temporary)
        (temporary / "graph.json").write_bytes(graph_raw)
        (temporary / "summary.json").write_bytes(summary_raw)
        base_identity = {
            k: metadata[k] for k in ("country", "candidate_release_id", "run_id")
        }
        attachments = {
            "graph_receipts": {
                "status": "available",
                "scope": "population",
                "source": str(temporary / "graph.json"),
                "identity": {**base_identity, "population_sha256": checkpoint},
            },
            "build_manifest": {
                "status": "available",
                "scope": "run",
                "source": str(temporary / "summary.json"),
                "identity": base_identity,
            },
        }
        for name in (
            *writer.EXPECTED_ATTACHMENTS,
            "holdout_validation",
            "model_validation",
        ):
            if name not in attachments:
                attachments[name] = {
                    "status": "missing",
                    "reason": "Not evaluated for this uncalibrated fifteen household engineering slice.",
                }
        reference = writer.write_candidate_quality(
            metadata, attachments, output_root, update_index=False
        )
    writer.read_candidate_quality(output_root, reference["sha256"])
    return {
        **reference,
        "graph_receipts_sha256": sha256(graph_raw),
        "build_summary_sha256": sha256(summary_raw),
        "exporter_sha256": identity["exporter_sha256"],
        "writer_sha256": sha256(_file_bytes(Path(writer.__file__))),
    }


def export_graph_evidence(
    *,
    run_root: Path,
    declaration_path: Path,
    output_root: Path,
    pins: EvidencePins = GENUINE_PINS,
) -> dict[str, str]:
    """Export the exact pinned run, refusing malformed inputs before publication."""
    try:
        return _export_graph_evidence(
            run_root=run_root,
            declaration_path=declaration_path,
            output_root=output_root,
            pins=pins,
        )
    except EvidenceError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise EvidenceError("Malformed or contradictory bounded evidence.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--operator-declaration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_graph_evidence(
                run_root=args.run_root,
                declaration_path=args.operator_declaration,
                output_root=args.output_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
