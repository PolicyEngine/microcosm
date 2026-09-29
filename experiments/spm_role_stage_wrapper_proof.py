#!/usr/bin/env python3
"""Run the real SPM role wrapper and gate against existing pinned populations.

Preparation does not execute this pilot. Once the source and pins are ready:

    .venv/bin/python experiments/spm_role_stage_wrapper_proof.py base \
        --out experiments/893-spm-role-stage-wrapper-base-q3-receipt.json
    .venv/bin/python experiments/spm_role_stage_wrapper_proof.py buildp \
        --out experiments/893-spm-role-stage-wrapper-buildp-receipt.json

Input paths and hashes come from the historical receipts committed at HEAD.
All three Census CSVs must already exist locally and match the runtime pins.
No input is written, no model simulation runs, and no download is attempted.
Only a new aggregate JSON receipt is written, with exclusive creation. Existing
receipts are never replaced. Diagnostics suppress exception text and raw rows.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RECEIPTS = {
    "base": "experiments/893-spm-role-stage-base-q3-receipt.json",
    "buildp": "experiments/893-spm-role-stage-buildp-agreement-receipt.json",
}
SOURCE_PATHS = ("packages", "tools", "pyproject.toml", "uv.lock")
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".lock"}


class ProofRefusalError(Exception):
    """A named aggregate check failed; no row-bearing detail is emitted."""


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise ProofRefusalError(label)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _source_fingerprint() -> dict[str, Any]:
    """Bind HEAD plus dirty runtime code without serializing the patch."""
    patch = _git("diff", "--binary", "HEAD", "--", *SOURCE_PATHS)
    untracked = {}
    for raw in _git(
        "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS
    ).split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        path = ROOT / relative
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            untracked[relative] = _sha256(path)
    payload = {
        "head": _git("rev-parse", "HEAD").decode().strip(),
        "tracked_source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_source_sha256": untracked,
        "proof_script_sha256": _sha256(Path(__file__).resolve()),
        "original_proof_script_sha256": _sha256(
            ROOT / "experiments/spm_role_stage_proof.py"
        ),
    }
    return {**payload, "dirty_source_digest": _json_digest(payload)}


def _historical_receipt(mode: str) -> dict[str, Any]:
    relative = HISTORICAL_RECEIPTS[mode]
    committed = _git("show", f"HEAD:{relative}")
    _require((ROOT / relative).read_bytes() == committed, "historical_receipt_changed")
    return json.loads(committed)


@contextlib.contextmanager
def _offline() -> Iterator[None]:
    """Reject accidental Python socket connections, including source fetches."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise ProofRefusalError("network_connection_attempted")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex


@contextlib.contextmanager
def _quiet_diagnostics() -> Iterator[None]:
    """Suppress Python and native-library diagnostics while touching populations."""
    sys.stdout.flush()
    sys.stderr.flush()
    originals = [os.dup(descriptor) for descriptor in (1, 2)]
    try:
        with open(os.devnull, "w") as quiet:
            for descriptor in (1, 2):
                os.dup2(quiet.fileno(), descriptor)
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                yield
    finally:
        for descriptor, original in zip((1, 2), originals, strict=True):
            os.dup2(original, descriptor)
            os.close(original)


def _table_digest(table: Any) -> str:
    import pandas as pd

    schema = [(str(name), repr(dtype)) for name, dtype in table.dtypes.items()]
    digest = hashlib.sha256(_json_digest(schema).encode())
    digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _frame_snapshot(frame: Any) -> dict[str, Any]:
    import pandas as pd

    return {
        "tables": {
            entity: _table_digest(frame.table(entity)) for entity in frame.entities
        },
        "links": {name: _table_digest(frame.link(name)) for name in frame.links},
        "weights": {
            entity: {
                "kind": frame.weights_for(entity).kind.value,
                "sha256": hashlib.sha256(
                    frame.weights_for(entity).values.tobytes()
                ).hexdigest(),
            }
            for entity in frame.weighted_entities
        },
        "strata_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame.strata, index=True).to_numpy().tobytes()
        ).hexdigest(),
    }


def _preservation(before: Any, after: Any, role_name: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from microcosm.build.us_runtime.spm_independence_role import (
        US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
    )

    _require(before.schema == after.schema, "schema_changed")
    _require(before.entities == after.entities, "entities_changed")
    _require(before.links == after.links, "links_changed")
    tables = {}
    for entity in before.entities:
        original = before.table(entity)
        output = after.table(entity)
        additions = [column for column in output if column not in original]
        expected_additions = (
            [role_name] if entity == "person" and role_name not in original else []
        )
        _require(additions == expected_additions, "unexpected_output_columns")
        pd.testing.assert_frame_equal(
            original, output.loc[:, original.columns], check_exact=True
        )
        tables[entity] = {
            "rows": len(original),
            "existing_columns": len(original.columns),
        }
    for name in before.links:
        pd.testing.assert_frame_equal(
            before.link(name), after.link(name), check_exact=True
        )
    _require(
        before.weighted_entities == after.weighted_entities, "weight_entities_changed"
    )
    for entity in before.weighted_entities:
        old, new = before.weights_for(entity), after.weights_for(entity)
        _require(old.kind == new.kind, "weight_kind_changed")
        _require(np.array_equal(old.values, new.values), "weight_values_changed")
    pd.testing.assert_series_equal(before.strata, after.strata, check_exact=True)
    _require(before.mass_log == after.mass_log, "mass_log_changed")
    for key, value in before.metadata.items():
        if key != US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY:
            _require(after.metadata.get(key) == value, "existing_metadata_changed")
    return {
        "all_existing_tables_and_columns_equal": True,
        "person_order_ids_ages_and_memberships_equal": True,
        "weights_and_kinds_equal": True,
        "strata_and_mass_log_equal": True,
        "existing_metadata_preserved": True,
        "tables": tables,
    }


def _composition(frame: Any) -> dict[str, Any]:
    from microcosm.build.us_runtime.spm_composition import check_spm_composition

    check = check_spm_composition(frame)
    return {
        "status": check.status,
        **{
            key: check.details[key]
            for key in (
                "n_units",
                "n_units_without_classified_adult",
                "n_units_without_member_aged_18_or_over",
                "role_source",
            )
        },
    }


def prove(mode: str, state: dict[str, str], source: dict[str, Any]) -> dict[str, Any]:
    from importlib.metadata import version

    import numpy as np
    from spm_role_stage_proof import _environment, _load_frame

    from microcosm.build.us_runtime.spm_independence_role import (
        US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
        us_spm_independence_role_signal_gate,
        with_us_spm_independence_role,
    )
    from microcosm.build.us_runtime.spm_role_source import (
        ASEC_SPM_ROLE_SOURCES,
        EVIDENCE_SPM_ROLE,
        NATIVE_SPM_ROLE,
        derive_spm_role_source,
    )
    from microcosm.data.source_enrichment import (
        PARENT_DATASET_SHA256,
        SOURCE_EVIDENCE_SHA256,
    )

    started = time.perf_counter()
    historical = _historical_receipt(mode)
    base_receipt = _historical_receipt("base")
    input_path = Path(historical["base_h5" if mode == "base" else "parent_h5"])
    input_sha = historical["base_sha256" if mode == "base" else "parent_sha256"]
    if mode == "buildp":
        _require(input_sha == PARENT_DATASET_SHA256, "buildp_parent_pin_drift")
    source_paths = {}
    inputs = {"population": (input_path, input_sha)}
    for year, pin in ASEC_SPM_ROLE_SOURCES.items():
        recorded = base_receipt["source_csvs"][str(year)]
        _require(recorded["pinned_sha256"] == pin.csv_sha256, "source_pin_drift")
        _require(recorded["size_bytes"] == pin.csv_size_bytes, "source_size_pin_drift")
        path = Path(recorded["path"])
        source_paths[year] = path
        inputs[f"asec_{year}"] = (path, pin.csv_sha256)
    _require(set(source_paths) == {2022, 2023, 2024}, "unexpected_source_years")
    if mode == "buildp":
        reference_sha = historical["reference_evidence_sha256"]
        _require(reference_sha == SOURCE_EVIDENCE_SHA256, "buildp_evidence_pin_drift")
        inputs["buildp_evidence"] = (
            Path(historical["reference_evidence_csv"]),
            reference_sha,
        )
    state["phase"] = "verify_input_pins"
    for path, expected in inputs.values():
        _require(path.is_file(), "required_local_input_missing")
        _require(_sha256(path) == expected, "input_digest_mismatch")

    state["phase"] = "load_population"
    checkpoint = time.perf_counter()
    frame = _load_frame(input_path)
    load_seconds = time.perf_counter() - checkpoint
    snapshot = _frame_snapshot(frame)
    before = _composition(frame)
    state["phase"] = "actual_stage_wrapper"
    checkpoint = time.perf_counter()
    enriched = with_us_spm_independence_role(
        frame, seed=0, time_period=2024, asec_spm_role_source_paths=source_paths
    )
    wrapper_seconds = time.perf_counter() - checkpoint
    state["phase"] = "actual_stage_signal_gate"
    gate = us_spm_independence_role_signal_gate(enriched)
    gate_details_sha256 = _json_digest(gate.details)
    _require(gate.passed, "actual_stage_signal_gate_failed")
    _require(
        US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY in enriched.metadata,
        "stage_provenance_missing",
    )
    state["phase"] = "original_derivation_agreement"
    checkpoint = time.perf_counter()
    reference = derive_spm_role_source(
        input_path, source_paths, expected_parent_sha256=input_sha
    )
    derivation_seconds = time.perf_counter() - checkpoint
    person = enriched.table("person")
    for column in ("person_id", "person_spm_unit_id"):
        _require(
            np.array_equal(
                person[column].to_numpy(), reference.evidence[column].to_numpy()
            ),
            "derivation_identity_order_disagreement",
        )
    role = person[NATIVE_SPM_ROLE].to_numpy(dtype=bool)
    _require(np.array_equal(role, reference.role), "wrapper_derivation_disagreement")
    for key in (
        "persons_joined",
        "native_spm_units",
        "unmatched_persons",
        "total_source_people",
        "total_source_units",
        "adult_child_person_count_mismatch_units",
        "independent_minor_persons",
        "true_role_persons",
        "minor_only_units_resolved",
        "classification_changed_units_vs_age_only",
    ):
        _require(
            reference.provenance[key] == historical["derivation"][key],
            "historical_derivation_count_drift",
        )
    evidence = person[["person_id", "person_spm_unit_id"]].copy()
    evidence[EVIDENCE_SPM_ROLE] = role
    evidence_bytes = evidence.to_csv(index=False, lineterminator="\n").encode()
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    if mode == "buildp":
        _require(
            evidence_bytes == inputs["buildp_evidence"][0].read_bytes(),
            "buildp_evidence_disagreement",
        )
    state["phase"] = "preservation_audit"
    preservation = _preservation(frame, enriched, NATIVE_SPM_ROLE)
    _require(_frame_snapshot(frame) == snapshot, "wrapper_mutated_input_frame")
    after = _composition(enriched)
    for label, measured in (("before", before), ("after", after)):
        for key, value in measured.items():
            _require(value == historical[label][key], "historical_composition_drift")
    state["phase"] = "verify_unchanged_input_bytes"
    for path, expected in inputs.values():
        _require(_sha256(path) == expected, "input_bytes_changed")
    _require(_source_fingerprint() == source, "source_changed_during_proof")
    _historical_receipt(mode)
    _historical_receipt("base")
    details = gate.details
    provenance = enriched.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY]
    return {
        "receipt": f"spm_role_stage_wrapper_proof.{mode}",
        "status": "PASS",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "environment": {
            **_environment(),
            **{
                name: version(name)
                for name in ("microcosm-build", "microcosm-frame", "microcosm-data")
            },
        },
        "stage_arguments": {"seed": 0, "time_period": 2024},
        "historical_receipt_sha256": _sha256(ROOT / HISTORICAL_RECEIPTS[mode]),
        "inputs": {
            label: {"sha256": expected, "size_bytes": path.stat().st_size}
            for label, (path, expected) in inputs.items()
        },
        "input_bytes_unchanged": True,
        "input_frame_unchanged": True,
        "before": before,
        "after": after,
        "agreement": {
            "persons_compared": len(role),
            "roles_differ": 0,
            "evidence_sha256": evidence_sha,
            "buildp_reference_bytes_equal": True if mode == "buildp" else None,
        },
        "preservation": preservation,
        "signal_gate": {
            "passed": gate.passed,
            "failure_count": len(gate.failures),
            "json_details_sha256": gate_details_sha256,
            **{
                key: details[key]
                for key in (
                    "role_share",
                    "role_share_band",
                    "minor_role_share",
                    "minor_role_share_band",
                    "persons_aged_15_to_17",
                    "independent_minor_persons",
                    "role_missing_values",
                )
            },
            "frame_projection_sha256": provenance["frame_projection_sha256"],
            "unmatched_persons": provenance["unmatched_persons"],
            "adult_child_person_count_mismatch_units": provenance[
                "adult_child_person_count_mismatch_units"
            ],
        },
        "timing_seconds": {
            "load": round(load_seconds, 2),
            "wrapper": round(wrapper_seconds, 2),
            "original_derivation": round(derivation_seconds, 2),
            "total": round(time.perf_counter() - started, 2),
        },
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(HISTORICAL_RECEIPTS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    state = {"phase": "prepare"}
    try:
        _require(not args.out.exists(), "output_already_exists")
        _require(args.out.parent.is_dir(), "output_directory_missing")
        with _quiet_diagnostics(), _offline():
            source = _source_fingerprint()
            receipt = prove(args.mode, state, source)
        state["phase"] = "write_new_aggregate_receipt"
        payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.out.open("x") as stream:
            stream.write(payload)
    except BaseException as error:
        # Exception messages and tracebacks can contain microdata values.
        failure = {
            "status": "FAIL",
            "phase": state["phase"],
            "error_type": type(error).__name__,
        }
        if isinstance(error, ProofRefusalError):
            failure["check"] = str(error)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "mode": args.mode, "receipt": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
