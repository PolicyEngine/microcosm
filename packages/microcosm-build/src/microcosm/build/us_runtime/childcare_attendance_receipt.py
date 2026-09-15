"""Content-bound attendance lineage across checkpoints, selection and native H5.

Hashes detect changed values or execution metadata, not maliciously forged
receipts. Publication authorization and independent source review are separate.
Weights may change and whole-household exports may select/reorder people; each
retained person's identity, age, membership and attendance must remain exact.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    childcare_attendance_contract,
)
from microcosm.frame import Frame

ATTENDANCE_RECEIPT_KEY = "childcare_attendance_binding"
ATTENDANCE_H5_KEY = "_childcare_attendance_receipt"
ATTENDANCE_CONTEXT_KEYS = (
    "nsece_childcare_attendance",
    "childcare_predictor_harmonization",
    "childcare_outside_domain_baseline",
    "childcare_attendance_stage",
)


def _plain(value):
    return json.loads(json.dumps(value, default=dict, allow_nan=False))


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def attendance_recipe_identity():
    """Bind the packaged source contract and the code implementing its recipe."""
    directory = Path(__file__).parent
    names = (
        "childcare_attendance.py",
        "childcare_population.py",
        "nsece_childcare.py",
        "nsece_childcare_bridge.py",
        "nsece_childcare_dependence.py",
        "childcare_attendance_stage.py",
        "childcare_attendance_receipt.py",
    )
    versions = {}
    for name in ("numpy", "pandas", "policyengine-us"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "contract_sha256": _digest(childcare_attendance_contract()),
        "runtime_versions": versions,
        "code_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names
        },
    }


def _context(frame):
    return _plain(
        {
            key: frame.metadata[key]
            for key in ATTENDANCE_CONTEXT_KEYS
            if key in frame.metadata
        }
    )


def _row_digests(people, execution_sha256):
    required = [
        "person_id",
        "person_household_id",
        "age",
        *US_CHILDCARE_ATTENDANCE_COLUMNS,
    ]
    if not set(required).issubset(people):
        raise ValueError(
            "Attendance binding requires person IDs, household links, age and all outputs."
        )
    if (
        people.reindex(columns=required[:3]).isna().any().any()
        or people.person_id.duplicated().any()
    ):
        raise ValueError(
            "Attendance binding requires unique person IDs and complete identity/age."
        )
    result = {}
    for row in people.reindex(columns=required).itertuples(index=False, name=None):
        key = str(row[0])
        if key in result:
            raise ValueError(
                "Attendance person IDs have ambiguous canonical representations."
            )
        result[key] = _digest(
            [
                execution_sha256,
                key,
                str(row[1]),
                *[None if pd.isna(x) else float(x) for x in row[2:]],
            ]
        )
    return result


def bind_childcare_attendance(frame):
    """Seal a freshly executed source stage; never use this to bless loaded values."""
    context = _context(frame)
    if "nsece_childcare_attendance" not in context:
        raise ValueError("Cannot bind attendance without a source execution receipt.")
    execution = {"recipe": attendance_recipe_identity(), "context": context}
    execution_sha256 = _digest(execution)
    binding = {
        "schema_version": 1,
        "execution": execution,
        "execution_sha256": execution_sha256,
        # Frame metadata uses small immutable mappings with linear key lookup.
        # A sequence keeps freezing/serializing a population-sized inventory
        # linear; construct a temporary dict only while verifying it.
        "rows": list(_row_digests(frame.table("person"), execution_sha256).items()),
    }
    binding["binding_sha256"] = _digest(binding)
    return Frame(
        {e: frame.table(e) for e in frame.entities},
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={**frame.metadata, ATTENDANCE_RECEIPT_KEY: binding},
    )


def assert_bound_childcare_attendance(frame, *, require_stage=True):
    """Validate the exact retained values and recipe; reject stale/unbound inputs."""
    binding = _plain(frame.metadata.get(ATTENDANCE_RECEIPT_KEY, {}))
    if binding.get("schema_version") != 1:
        raise ValueError(
            "Attendance needs a content-bound source receipt; rebuild from the original parent."
        )
    claimed = binding.pop("binding_sha256", None)
    if claimed != _digest(binding):
        raise ValueError("Attendance receipt content hash mismatch.")
    execution = binding["execution"]
    if binding["execution_sha256"] != _digest(execution):
        raise ValueError("Attendance execution hash mismatch.")
    if execution["recipe"] != attendance_recipe_identity():
        raise ValueError("Attendance recipe changed; rebuild from the original parent.")
    if execution["context"] != _context(frame):
        raise ValueError("Attendance metadata disagrees with its bound execution.")
    context = execution["context"]
    if require_stage:
        stage = context.get("childcare_attendance_stage", {})
        source = context.get("nsece_childcare_attendance", {})
        expected = [x["sha256"] for x in childcare_attendance_contract()["artifacts"]]
        if (
            [x.get("sha256") for x in source.get("artifacts", [])] != expected
            or stage.get("stage") != "nsece_childcare_attendance"
            or stage.get("seed") != source.get("seed")
            or stage.get("modeled_age_domain") != [0, 12]
            or stage.get("outside_domain_policy")
            not in ("require_observed", "inherit_engine_baseline")
        ):
            raise ValueError(
                "Attendance release requires the pinned production source stage receipt."
            )
    rows = dict(binding["rows"])
    if len(rows) != len(binding["rows"]):
        raise ValueError("Attendance receipt contains duplicate person identities.")
    for person, digest in _row_digests(
        frame.table("person"), binding["execution_sha256"]
    ).items():
        if rows.get(person) != digest:
            raise ValueError(
                "Attendance values, identities or membership differ from the source receipt."
            )
    return {
        "schema_version": 1,
        "binding_sha256": claimed,
        "execution_sha256": binding["execution_sha256"],
        "source_people": len(binding["rows"]),
        "retained_people": len(frame.table("person")),
        "execution": execution,
    }


def restore_native_childcare_receipt(path, frame):
    """Load the receipt through either native ingress, preserving its validation."""
    with pd.HDFStore(path, mode="r") as store:
        metadata = (
            json.loads(store[ATTENDANCE_H5_KEY].iloc[0])
            if ATTENDANCE_H5_KEY in store
            else {}
        )
    if metadata:
        frame = Frame(
            {e: frame.table(e) for e in frame.entities},
            frame.schema,
            {e: frame.weights_for(e) for e in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata={**frame.metadata, **metadata},
        )
        assert_bound_childcare_attendance(frame, require_stage=False)
    else:
        people = frame.table("person")
        columns = [c for c in US_CHILDCARE_ATTENDANCE_COLUMNS if c in people]
        if columns and np.any(people[columns].fillna(0).to_numpy(dtype=float) != 0):
            raise ValueError(
                "Native attendance values lack a bound receipt; rebuild from the original parent."
            )
    return frame


def childcare_attendance_public_metadata(frame):
    """Aggregate-only evidence: never publish the private person hash inventory."""
    summary = assert_bound_childcare_attendance(frame, require_stage=False)
    return {**_context(frame), ATTENDANCE_RECEIPT_KEY: summary}
