"""A finished synthetic UK size build on disk for the size-experiment tests.

It writes what a ``--dataset-households`` graph build leaves behind and the
harness reads: the flat evidence artifacts named and digested in
``evidence-index.json``, the pool frame in the run's content store and the
``numerical.graph.json`` record that names it. The payloads are encoded the
way the graph's size nodes encode them, so the harness meets real formats.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import uk_national_frame
from microcosm.build.uk_runtime.dataset_size import (
    draw_uk_dataset_size,
    refit_uk_dataset_size,
    select_uk_dataset_size,
)
from microcosm.build.uk_runtime.target_weights import (
    uk_rule_loss_weights,
    uk_target_rows,
)
from microcosm.calibrate import Target, TargetSet, build_constraint_matrix, calibrate
from microcosm.calibrate.artifacts import (
    decode_problem,
    encode_calibration_result,
    encode_problem,
    encode_solution,
)
from microcosm.frame import WeightKind
from microcosm.graph.canonical import canonical_json
from microcosm.graph.store import ContentStore

SYNTHETIC_SIZE_SETTINGS = dict(epochs=3, learning_rate=0.05, seed=7, pi_hi=0.5)
_CONSTITUENCIES = ("E14000001", "E14000002", "W07000041", "N05000001")
_AUTHORITIES = ("E06000001", "E06000002", "W06000001", "N09000001")
_TENURES = ("OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL")


def synthetic_uk_pool(households: int = 16, *, seed: int = 3):
    """A small cloned pool with the household metadata the profile reads."""

    rng = np.random.default_rng(seed)
    ids = np.arange(101, 101 + households)
    area = np.arange(households) % len(_CONSTITUENCIES)
    sizes = 1 + (np.arange(households) % 3)
    person_household = np.repeat(ids, sizes)
    persons = len(person_household)
    first = np.r_[True, person_household[1:] != person_household[:-1]]
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": rng.uniform(0.5, 3.0, households),
            "source_household_id": ids // 2,
            "constituency_code": [_CONSTITUENCIES[a] for a in area],
            "local_authority_code": [_AUTHORITIES[a] for a in area],
            "tenure_type": [_TENURES[i % len(_TENURES)] for i in range(households)],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, persons + 1),
            "person_household_id": person_household,
            "person_benunit_id": person_household + 1000,
            "age": rng.integers(20, 85, persons),
            "is_household_head": first,
            "employment_income": rng.uniform(0.0, 50_000.0, persons),
        }
    )
    benunit = pd.DataFrame({"benunit_id": ids + 1000})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2025",
        weight_kind=WeightKind.IMPORTANCE,
    )


def _targets(frame) -> TargetSet:
    household = frame.table("household")
    person = frame.table("person")
    ids = household["household_id"].to_numpy()
    size = person.groupby("person_household_id").size().reindex(ids).to_numpy()
    income = (
        person.groupby("person_household_id")["employment_income"]
        .sum()
        .reindex(ids)
        .to_numpy()
    )
    design = frame.weights_for("household").values

    def indicator(mask):
        values = np.asarray(mask, dtype=np.float64)

        def measure(frame):
            return values

        return measure

    def local(name, area_type, code, metric, family, mask):
        mask = np.asarray(mask, dtype=np.float64)
        return Target(
            name,
            "household",
            indicator(mask),
            float(1.1 * (mask * design).sum()),
            metadata={
                "area_type": area_type,
                "area_code": code,
                "metric": metric,
                "family": family,
                "materialization": "uk_local_surface",
            },
        )

    def national(name, family, geography, unit, values, scale=0.95):
        values = np.asarray(values, dtype=np.float64)
        return Target(
            name,
            "household",
            indicator(values),
            float(scale * (values * design).sum()),
            metadata={
                "family": family,
                "ledger_geography_id": geography,
                "ledger_measure_unit": unit,
                "materialization": "uk_national_measure",
            },
        )

    rows = []
    for code in _CONSTITUENCIES:
        rows.append(
            local(
                f"households@{code}",
                "constituency",
                code,
                "households",
                "census_households",
                household["constituency_code"].to_numpy() == code,
            )
        )
    for code in _AUTHORITIES:
        rows.append(
            local(
                f"private_rent@{code}",
                "la",
                code,
                "tenure/private_rent",
                "tenure",
                (household["local_authority_code"].to_numpy() == code)
                & (household["tenure_type"].to_numpy() == "RENT_PRIVATELY"),
            )
        )
    rows += [
        national("ons.households_total", "ons_households", "K02000001", "count", np.ones(len(ids))),
        national(
            "ons.household_composition.lone_households_under_65",
            "ons_household_composition",
            "K02000001",
            "count",
            size == 1,
        ),
        national("ons.population.wales", "ons_population", "W92000004", "count",
                 household["constituency_code"].str.startswith("W").to_numpy() * size),
        national("hmrc.employment_income", "hmrc_spi", "K02000001", "gbp", income, scale=1.05),
    ]  # fmt: skip
    return TargetSet(rows)


def _write(
    run_dir: Path, evidence: dict, key: str, filename: str, payload: bytes
) -> None:
    (run_dir / filename).write_bytes(payload)
    evidence[key] = {
        "filename": filename,
        "key": hashlib.sha256(key.encode()).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def synthetic_uk_size_run(
    run_dir: Path,
    *,
    households: int = 16,
    k: int = 10,
    selection_l2_lambda: float = 0.0,
    refit_l2_lambda: float = 0.0,
) -> dict:
    """Write a finished size build of ``k`` of ``households`` rows to ``run_dir``.

    ``selection_l2_lambda`` / ``refit_l2_lambda`` build it with the size
    stages' L2 penalties on, as a ``--selection-l2-lambda`` /
    ``--refit-l2-lambda`` build would.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = synthetic_uk_pool(households)
    ids = frame.table("household")["household_id"].tolist()
    targets = _targets(frame)
    compiled = build_constraint_matrix(frame, targets, "household")
    metadata = [dict(target.metadata) for target in compiled.targets]
    rows = uk_target_rows(
        compiled.names,
        metadata,
        compiled.target_vector,
        local=[row["materialization"] == "uk_local_surface" for row in metadata],
    )
    weights = uk_rule_loss_weights(rows, rule="grain_equal")
    problem_bytes = encode_problem(
        compiled,
        entity_ids=ids,
        target_metadata=metadata,
        bindings={
            "mass_reason": "synthetic size build",
            "max_weight_ratio": 10.0,
            "target_loss_cap": 10.0,
            "target_loss_weights": weights.tolist(),
        },
    )
    problem = decode_problem(problem_bytes)
    settings = dict(SYNTHETIC_SIZE_SETTINGS)
    dense = calibrate(
        frame,
        problem.to_target_set(),
        weight_entity="household",
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        seed=settings["seed"],
        mass="free",
        mass_reason="synthetic size build",
        max_weight_ratio=10.0,
        target_loss_weights=weights,
        target_loss_cap=10.0,
    )
    common = dict(
        households=k,
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        seed=settings["seed"],
    )
    selection_l2 = (
        {} if not selection_l2_lambda else {"selection_l2_lambda": selection_l2_lambda}
    )
    refit_l2 = {} if not refit_l2_lambda else {"refit_l2_lambda": refit_l2_lambda}
    selection = select_uk_dataset_size(
        frame, dense, pi_hi=settings["pi_hi"], **selection_l2, **common
    )
    draw = draw_uk_dataset_size(
        frame, dense, selection=selection, households=k, seed=settings["seed"],
        pi_hi=settings["pi_hi"],
    )  # fmt: skip
    sized = refit_uk_dataset_size(
        frame,
        dense,
        selection=selection,
        draw=draw,
        pi_hi=settings["pi_hi"],
        **selection_l2,
        **refit_l2,
        **common,
    )
    compact_ids = sized.result.frame.table("household")["household_id"].tolist()
    evidence: dict[str, dict] = {}
    sha = problem.sha256
    _write(
        run_dir,
        evidence,
        "uk.full.problem/problem",
        "uk.full.problem.problem.artifact",
        problem_bytes,
    )
    _write(
        run_dir,
        evidence,
        "uk.full.dense/result",
        "uk.full.dense.result.artifact",
        encode_calibration_result(dense, entity_ids=ids, problem_sha256=sha),
    )
    _write(
        run_dir,
        evidence,
        "uk.full.dense/solution",
        "uk.full.dense.solution.artifact",
        encode_solution(dense.weights, entity_ids=ids, problem_sha256=sha),
    )
    _write(
        run_dir,
        evidence,
        "uk.full.size_search/result",
        "uk.full.size_search.result.artifact",
        encode_calibration_result(
            selection.selection, entity_ids=ids, problem_sha256=sha
        ),
    )
    _write(
        run_dir,
        evidence,
        "uk.full.size_search/selection",
        "uk.full.size_search.selection.json",
        canonical_json(
            {
                "method": "contribution_informed_l0",
                "problem_sha256": sha,
                "protected": selection.protected.tolist(),
                "households": k,
                "epochs": settings["epochs"],
                "learning_rate": settings["learning_rate"],
                "seed": settings["seed"],
                "pi_hi": settings["pi_hi"],
            }
        ),
    )
    draw_payload = {"method": "exact_count", **asdict(draw)}
    for key in ("support", "inclusion_probabilities"):
        draw_payload[key] = draw_payload[key].tolist()
    _write(
        run_dir,
        evidence,
        "uk.full.size_draw/draw",
        "uk.full.size_draw.draw.json",
        canonical_json({"problem_sha256": sha, **draw_payload}),
    )
    _write(
        run_dir,
        evidence,
        "uk.full.size_refit/solution",
        "uk.full.size_refit.solution.artifact",
        encode_solution(
            sized.result.weights, entity_ids=compact_ids, problem_sha256=sha
        ),
    )
    _write(
        run_dir,
        evidence,
        "uk.full.size_refit/size",
        "uk.full.size_refit.size.json",
        canonical_json(
            {**sized.receipt, "problem_sha256": sha, "household_ids": compact_ids}
        ),
    )
    (run_dir / "evidence-index.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True)
    )
    store = ContentStore(run_dir / ".graph-store")
    frame_key = hashlib.sha256(b"synthetic-uk-pool").hexdigest()
    store.put_frame(frame_key, frame)
    (run_dir / "numerical.graph.json").write_text(
        json.dumps({"content_addressed": {"nodes": {"uk.full.pool": {"frame_key": frame_key}}}})
    )  # fmt: skip
    return {
        "frame": frame,
        "problem": problem,
        "dense": dense,
        "selection": selection,
        "draw": draw,
        "sized": sized,
        "settings": settings,
        "k": k,
    }


__all__ = ["SYNTHETIC_SIZE_SETTINGS", "synthetic_uk_pool", "synthetic_uk_size_run"]
