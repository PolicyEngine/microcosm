#!/usr/bin/env python3
"""Verify a saved attendance candidate through both native population loaders.

This checks artifact integrity and preservation, not statistical validity. The
report contains hashes and aggregate counts only; source records stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    assert_bound_childcare_attendance,
)
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.build.us_runtime.l0_refit_export import load_us_frame
from microcosm.frame import Frame


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _assert_table_equal(expected, actual):
    # HDF may restore strings in Python rather than Arrow storage. Normalize
    # that backend only; retain null semantics, values and all other dtypes.
    tables = []
    for original in (expected, actual):
        table = original.copy()
        for column in table:
            dtype = table[column].dtype
            if isinstance(dtype, pd.StringDtype):
                table[column] = table[column].astype(
                    pd.StringDtype(storage="python", na_value=dtype.na_value)
                )
        tables.append(table)
    pd.testing.assert_frame_equal(*tables, check_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-native-h5", type=Path, required=True)
    parser.add_argument("--source-stage-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report path must be new")
    if _sha256(args.parent_h5) != args.parent_sha256:
        parser.error("Parent population hash mismatch")
    source = json.loads(args.source_stage_report.read_text())
    checkpoint_hash = _sha256(args.candidate_checkpoint)
    native_hash = _sha256(args.candidate_native_h5)
    for key, actual in (
        ("parent_population_sha256", args.parent_sha256),
        ("candidate_checkpoint_sha256", checkpoint_hash),
        ("native_candidate_sha256", native_hash),
    ):
        if source[key] != actual:
            raise ValueError(f"Source-stage report does not bind {key}.")
    if (
        not source["production_stage_executed"]
        or not source["native_candidate_written"]
    ):
        raise ValueError("Source-stage report does not attest a native stage export.")
    parent = load_legacy_calibrated_us_h5(args.parent_h5)
    stored = load_frame_checkpoint(args.candidate_checkpoint)
    frame = stored.frame
    candidate = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=stored.metadata["frame_metadata"],
    )
    binding = assert_bound_childcare_attendance(candidate)
    columns = list(US_CHILDCARE_ATTENDANCE_COLUMNS)
    verified_loaders = []
    for loader in (load_legacy_calibrated_us_h5, load_us_frame):
        loaded = loader(args.candidate_native_h5)
        restored = assert_bound_childcare_attendance(loaded)
        if restored["binding_sha256"] != binding["binding_sha256"]:
            raise ValueError("Native reload changed the attendance binding.")
        _assert_table_equal(
            loaded.table("person")[columns], candidate.table("person")[columns]
        )
        for entity in parent.entities:
            original = parent.table(entity)
            _assert_table_equal(original, candidate.table(entity)[original.columns])
            _assert_table_equal(original, loaded.table(entity)[original.columns])
        for entity in parent.weighted_entities:
            original_weights = parent.weights_for(entity)
            for restored_frame in (candidate, loaded):
                actual_weights = restored_frame.weights_for(entity)
                if actual_weights.kind != original_weights.kind:
                    raise ValueError("Candidate changed the original weight kind.")
                np.testing.assert_array_equal(
                    original_weights.values, actual_weights.values
                )
        verified_loaders.append(loader.__module__ + "." + loader.__name__)
    people = candidate.table("person")
    result = {
        "parent_sha256": args.parent_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "native_sha256": native_hash,
        "source_stage_report_sha256": _sha256(args.source_stage_report),
        "verification_code_sha256": _sha256(Path(__file__)),
        "attendance_recipe": binding["execution"]["recipe"],
        "content_binding_sha256": binding["binding_sha256"],
        "engine_version": version("policyengine-us"),
        "core_version": version("policyengine-core"),
        "people": len(people),
        "households": len(candidate.table("household")),
        "under13_children": int(people.age.between(0, 12).sum()),
        "verified_native_loaders": verified_loaders,
        "all_original_columns_and_weights_preserved": True,
        "all_attendance_values_preserved_on_native_reload": True,
        "comparison": "exact values and dtypes, normalizing only pandas string storage backend",
        "interpretation": "artifact integrity and preservation only; no statistical certification or publication",
        "production_ready": False,
    }
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
