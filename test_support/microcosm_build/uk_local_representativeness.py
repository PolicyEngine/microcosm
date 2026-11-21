"""A finished synthetic sized UK local run for the representativeness tests.

It writes only what :func:`~microcosm.build.uk_runtime.representativeness.read_uk_local_run`
reads: the problem, the dense solution, the size refit's solution and receipt
(flat artifacts digested in ``evidence-index.json``), and the pool frame in the
run's content store, named by ``numerical.graph.json``. D and S are whatever
the test passes, so each test sets the distributions it compares.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import uk_national_frame
from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
from microcosm.calibrate.artifacts import (
    decode_problem,
    encode_problem,
    encode_solution,
)
from microcosm.frame import WeightKind
from microcosm.graph.canonical import canonical_json
from microcosm.graph.store import ContentStore

#: Constituency and local authority pairs, one per synthetic area.
SYNTHETIC_AREAS = (
    ("E14000001", "E06000001"),
    ("E14000002", "E06000002"),
    ("W07000041", "W06000001"),
    ("S14000001", "S12000005"),
)
_TENURES = ("OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL")
_HOUSEHOLD_TYPES = ("ONE_PERSON", "COUPLE_NO_CHILDREN", "LONE_PARENT", "OTHER")


def synthetic_uk_local_pool(per_area: int = 40, *, seed: int = 5):
    """``per_area`` households in each synthetic area, each its own source."""

    rng = np.random.default_rng(seed)
    households = per_area * len(SYNTHETIC_AREAS)
    ids = np.arange(101, 101 + households)
    area = np.repeat(np.arange(len(SYNTHETIC_AREAS)), per_area)
    sizes = 1 + (np.arange(households) % 3)
    person_household = np.repeat(ids, sizes)
    first = np.r_[True, person_household[1:] != person_household[:-1]]
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": rng.uniform(0.5, 3.0, households),
            "source_household_id": ids + 10_000,
            "constituency_code": [SYNTHETIC_AREAS[a][0] for a in area],
            "local_authority_code": [SYNTHETIC_AREAS[a][1] for a in area],
            "tenure_type": [_TENURES[i % len(_TENURES)] for i in range(households)],
            "ons_household_type": [
                _HOUSEHOLD_TYPES[i % len(_HOUSEHOLD_TYPES)] for i in range(households)
            ],
        }
    )
    persons = len(person_household)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, persons + 1),
            "person_household_id": person_household,
            "person_benunit_id": person_household + 1_000,
            "age": rng.integers(20, 85, persons),
            "is_household_head": first,
            "employment_income": rng.uniform(0.0, 50_000.0, persons),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids + 1_000}),
        household=household,
        time_period="2025",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _write(run_dir: Path, evidence: dict, key: str, filename: str, payload: bytes):
    (run_dir / filename).write_bytes(payload)
    evidence[key] = {
        "filename": filename,
        "key": hashlib.sha256(key.encode()).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def write_synthetic_sized_run(
    run_dir: Path,
    frame,
    *,
    dense_weights: np.ndarray,
    support: np.ndarray | None,
    weights: np.ndarray | None,
) -> dict:
    """Write a finished run: D on every pool row, S on ``support`` (or no S)."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ids = frame.table("household")["household_id"].tolist()
    count = len(ids)
    compiled = build_constraint_matrix(
        frame,
        TargetSet(
            [
                Target(
                    "ons.households_total",
                    "household",
                    lambda _frame: np.ones(count),
                    float(count),
                    metadata={"materialization": "uk_national_measure"},
                )
            ]
        ),
        "household",
    )
    problem = encode_problem(
        compiled,
        entity_ids=ids,
        target_metadata=[dict(target.metadata) for target in compiled.targets],
        bindings={"mass_reason": "synthetic sized run"},
    )
    sha = decode_problem(problem).sha256
    evidence: dict[str, dict] = {}
    _write(
        run_dir,
        evidence,
        "uk.full.problem/problem",
        "uk.full.problem.problem.artifact",
        problem,
    )
    _write(
        run_dir,
        evidence,
        "uk.full.dense/solution",
        "uk.full.dense.solution.artifact",
        encode_solution(dense_weights, entity_ids=ids, problem_sha256=sha),
    )
    if support is not None:
        support = np.asarray(support, dtype=np.int64)
        kept = [ids[i] for i in support]
        _write(
            run_dir,
            evidence,
            "uk.full.size_refit/solution",
            "uk.full.size_refit.solution.artifact",
            encode_solution(weights, entity_ids=kept, problem_sha256=sha),
        )
        _write(
            run_dir,
            evidence,
            "uk.full.size_refit/size",
            "uk.full.size_refit.size.json",
            canonical_json(
                {
                    "problem_sha256": sha,
                    "pool_row_indices": support.tolist(),
                    "household_ids": kept,
                }
            ),
        )
    (run_dir / "evidence-index.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True)
    )
    store = ContentStore(run_dir / ".graph-store")
    frame_key = hashlib.sha256(b"synthetic-uk-local-pool").hexdigest()
    store.put_frame(frame_key, frame)
    (run_dir / "numerical.graph.json").write_text(
        json.dumps(
            {"content_addressed": {"nodes": {"uk.full.pool": {"frame_key": frame_key}}}}
        )
    )
    return {"problem_sha256": sha, "evidence": evidence}


__all__ = [
    "SYNTHETIC_AREAS",
    "synthetic_uk_local_pool",
    "write_synthetic_sized_run",
]
