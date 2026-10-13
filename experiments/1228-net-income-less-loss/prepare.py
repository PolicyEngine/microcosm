"""Build the #1228 re-solve inputs from the Route A release's own checkpoint.

Inputs (read only):

* ``release-checkpoints/target_frame_checkpoint.h5``: the release's
  materialized target frame, one household column per target (5,694) plus the
  person and tax-unit input tables.
* ``calibration_diagnostics.json``: each target's name, measure column,
  compiled value, loss weight, initial and final estimates, and metadata.
* ``populace_us_2024_calibration.npz``: the initial and published weights.

Outputs (under ``--out``):

* ``target_matrix.npz``: the release's 5,694 target rows as CSR (float64),
  in the diagnostics' order. Checked: ``A @ initial`` and ``A @ final``
  reproduce every target's recorded initial and final estimate.
* ``concept_rows.npz``: the 312 business, partnership and rental rows rebuilt
  on the signed net-income-(less-loss) concepts of #1228, as CSR, with their
  diagnostics row numbers. The script first rebuilds the same 312 rows on the
  release's own concepts and requires them to equal the checkpoint's columns,
  so both sets come from one verified materialization.
* ``targets_meta.parquet``: one row per target: name, value, family, SOI
  measure id, geography, release loss weight, and the loss weights of every
  arm (recomputed with ``target_loss_weights`` over the arm's specs; the
  release arm must reproduce the recorded weights).
* ``concept_totals.npz``: household-level tax-unit sums (positive part,
  negative part, positive count, negative count) of each income concept, so
  any weight vector's model totals can be read off without a simulation.
* ``weights.npz``: ``initial`` and ``published``.
* ``MANIFEST.json``: sha256 of each output, the input paths and digests, the
  structural checks, and this repository's HEAD.

Run: ``uv run --no-sync python experiments/1228-net-income-less-loss/prepare.py --out DIR``
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RELEASE_ID = "populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z"
RUN = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/"
    "overnight-20260923/route-a/run-4b57d15a287c"
)
CHECKPOINT = RUN / "release-checkpoints" / "target_frame_checkpoint.h5"
RELEASE = RUN / "release-out" / RELEASE_ID
DIAGNOSTICS = RELEASE / "releases" / RELEASE_ID / "calibration_diagnostics.json"
CALIBRATION_NPZ = RELEASE / "artifacts" / "populace_us_2024_calibration.npz"

#: The six HT2 measure ids (#1228), their release concept and their
#: net-income-(less-loss) concept.
MEASURES = {
    "schedule_c_income": ("schedule_c_nonsstb", "schedule_c_total"),
    "partnership_scorp_income": ("partnership_s_corp", "partnership_s_corp"),
    "rental_royalty_income": ("rental_plus_farm_rent", "rental_only"),
}
#: Person inputs summed to the tax unit for each concept.
CONCEPT_INPUTS = {
    "schedule_c_nonsstb": ("self_employment_income_before_lsr",),
    "schedule_c_total": (
        "self_employment_income_before_lsr",
        "sstb_self_employment_income_before_lsr",
    ),
    "sstb_only": ("sstb_self_employment_income_before_lsr",),
    "partnership_s_corp": ("partnership_income", "s_corp_income"),
    "rental_only": ("rental_income",),
    "rental_plus_farm_rent": ("rental_income", "farm_rent_income"),
    "farm_rent_only": ("farm_rent_income",),
}
#: Arms: which of the 312 rows each keeps, and on which concept.
#: ``release``: all on the release concepts. ``concept_exact``: all on the
#: #1228 concepts. ``concept_exact_no_business_count``: the same without the
#: 52 Schedule C return counts (the frame has almost no Schedule C loss
#: carriers). ``drop``: the 260 rows whose release concept differs from the
#: IRS column are dropped; the rental amount keeps its release concept.
ARMS = (
    "release",
    "concept_exact",
    "concept_exact_no_business_count",
    "drop",
)


def log(message: str, started: float) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"[{time.time() - started:7.1f}s rss={rss:.2f}GB] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointTable:
    """One table of the checkpoint: columns stored one group per column."""

    def __init__(self, handle: h5py.File, table: str):
        self.group = handle[f"tables/{table}"]
        self.names = json.loads(self.group.attrs["columns_json"])
        self.index = {name: position for position, name in enumerate(self.names)}

    def column(self, name: str) -> np.ndarray:
        key = f"{self.index[name]:05d}"
        node = self.group["columns"][key]
        assert node.attrs["name"] == name, (node.attrs["name"], name)
        return node["values"][()]


def npz_array(path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        return np.lib.format.read_array(
            io.BytesIO(archive.read(f"{name}.npy")), allow_pickle=False
        )


def measure_id(entry: dict) -> str:
    return str(entry["metadata"].get("source_measure_id", ""))


def concept_stem(entry: dict) -> str | None:
    measure = measure_id(entry)
    for stem in MEASURES:
        if measure in {f"{stem}_amount", f"{stem}_returns"}:
            return stem
    return None


def tax_unit_concepts(person: CheckpointTable, household_ids: np.ndarray):
    """Per-tax-unit totals of each concept, and each tax unit's household row."""

    tax_unit_of_person = person.column("person_tax_unit_id").astype(np.int64)
    household_of_person = person.column("person_household_id").astype(np.int64)
    units, unit_position = np.unique(tax_unit_of_person, return_inverse=True)
    household_row = pd.Series(np.arange(len(household_ids)), index=household_ids)
    unit_household = np.full(len(units), -1, dtype=np.int64)
    unit_household[unit_position] = household_row.reindex(
        household_of_person
    ).to_numpy()
    assert (unit_household >= 0).all(), "a person's household is not in the frame"
    # Every person of a tax unit shares its household.
    check = np.full(len(units), -1, dtype=np.int64)
    np.maximum.at(
        check, unit_position, household_row.reindex(household_of_person).to_numpy()
    )
    assert np.array_equal(check, unit_household), "a tax unit spans households"
    inputs = {
        column: person.column(column).astype(np.float64)
        for columns in CONCEPT_INPUTS.values()
        for column in columns
    }
    totals = {}
    for concept, columns in CONCEPT_INPUTS.items():
        person_total = np.zeros(len(tax_unit_of_person), dtype=np.float64)
        for column in columns:
            person_total += inputs[column]
        unit_total = np.zeros(len(units), dtype=np.float64)
        np.add.at(unit_total, unit_position, person_total)
        totals[concept] = unit_total
    return totals, unit_household


def household_row(
    unit_values: np.ndarray, unit_household: np.ndarray, n_households: int
) -> np.ndarray:
    out = np.zeros(n_households, dtype=np.float64)
    np.add.at(out, unit_household, unit_values)
    return out


def materialize(
    entry: dict,
    *,
    concept_totals: dict[str, np.ndarray],
    unit_household: np.ndarray,
    household_state: np.ndarray,
    exact: bool,
) -> np.ndarray:
    """One target's household column on the release or the #1228 concept.

    Mirrors tools/build_us_fiscal_refresh_release.py ``_soi_component_row``:
    the release takes the positive part of business and partnership income
    and counts a positive total; rental keeps the sign but counts a positive
    total. The #1228 concepts keep the sign and count a nonzero total.
    """

    stem = concept_stem(entry)
    release_concept, exact_concept = MEASURES[stem]
    values = concept_totals[exact_concept if exact else release_concept]
    returns = measure_id(entry).endswith("_returns")
    if exact:
        unit = (values != 0).astype(np.float64) if returns else values
    elif returns:
        unit = (values > 0).astype(np.float64)
    elif stem == "rental_royalty_income":
        unit = values
    else:
        unit = np.maximum(values, 0.0)
    metadata = entry["metadata"]
    assert metadata.get("agi_lower_bound") == "-inf", metadata.get("agi_lower_bound")
    assert metadata.get("agi_upper_bound") == "inf", metadata.get("agi_upper_bound")
    assert metadata.get("filing_status") == "All", metadata.get("filing_status")
    assert "itemized_only" not in metadata and "taxable_only" not in metadata
    column = household_row(unit, unit_household, len(household_state))
    state_fips = metadata.get("state_fips")
    if state_fips:
        column = np.where(household_state == int(state_fips), column, 0.0)
    return column


def arm_rows(entries: list[dict], arm: str) -> np.ndarray:
    keep = []
    for row, entry in enumerate(entries):
        measure = measure_id(entry)
        if arm == "concept_exact_no_business_count" and (
            measure == "schedule_c_income_returns"
        ):
            continue
        if (
            arm == "drop"
            and concept_stem(entry) is not None
            and (measure != "rental_royalty_income_amount")
        ):
            continue
        keep.append(row)
    return np.asarray(keep, dtype=np.int64)


def arm_specs(entries: list[dict], rows: np.ndarray, *, exact: bool) -> list:
    """Spec stand-ins for ``target_loss_weights``: what the formula reads."""

    concepts = {
        "schedule_c_income": "business_net_income_less_loss",
        "partnership_scorp_income": "partnership_and_s_corp_net_income_less_loss",
        "rental_royalty_income": "rent_and_royalty_net_income_less_loss",
    }
    specs = []
    for row in rows:
        entry = entries[int(row)]
        metadata = dict(entry["metadata"])
        stem = concept_stem(entry)
        if exact and stem is not None:
            metadata["source_variable"] = concepts[stem]
            metadata["variable"] = concepts[stem]
        specs.append(
            SimpleNamespace(
                name=entry["target_name"],
                value=float(entry["compiled_target"]),
                family=entry["registry"]["family"],
                entity=entry["entity"],
                period=entry["period"],
                filter=entry["filter"],
                metadata=metadata,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    diagnostics = json.loads(DIAGNOSTICS.read_text())
    entries: list[dict] = diagnostics["targets"]
    log(f"{len(entries)} targets from diagnostics", started)
    initial = npz_array(CALIBRATION_NPZ, "initial_household_weight")
    published = npz_array(CALIBRATION_NPZ, "household_weight")

    with h5py.File(CHECKPOINT, "r") as handle:
        household = CheckpointTable(handle, "household")
        person = CheckpointTable(handle, "person")
        household_ids = household.column("household_id").astype(np.int64)
        household_state = household.column("state_fips").astype(np.int64)
        n_households = len(household_ids)
        assert n_households == len(initial) == len(published)
        checkpoint_weights = handle["weights/household/values"][()]
        assert np.allclose(checkpoint_weights, initial, rtol=0, atol=1e-9), (
            "checkpoint weights are not the release's initial weights"
        )

        indptr = [0]
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        release_columns: dict[int, np.ndarray] = {}
        for row, entry in enumerate(entries):
            column = household.column(entry["measure"]["name"]).astype(np.float64)
            nonzero = np.flatnonzero(column)
            indices.append(nonzero.astype(np.int32))
            data.append(column[nonzero])
            indptr.append(indptr[-1] + len(nonzero))
            if concept_stem(entry) is not None:
                release_columns[row] = column
            if row % 500 == 0:
                log(f"read row {row}", started)
        matrix = sparse.csr_array(
            (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
            shape=(len(entries), n_households),
        )
        log(f"matrix {matrix.shape} nnz={matrix.nnz}", started)

        concept_totals, unit_household = tax_unit_concepts(person, household_ids)
    log(f"{len(unit_household)} tax units", started)

    checks: dict[str, object] = {}
    for label, weights, key in (
        ("initial", initial, "initial_estimate"),
        ("published", published, "final_estimate"),
    ):
        estimates = matrix @ weights
        recorded = np.asarray([entry[key] for entry in entries], dtype=np.float64)
        gap = np.abs(estimates - recorded) / np.maximum(np.abs(recorded), 1.0)
        checks[f"{label}_estimates_max_relative_gap"] = float(gap.max())
        assert gap.max() < 1e-6, (label, float(gap.max()))

    exact_rows = sorted(release_columns)
    assert len(exact_rows) == 312, len(exact_rows)
    rebuild_gap = 0.0
    exact_columns = []
    for row in exact_rows:
        entry = entries[row]
        kwargs = dict(
            concept_totals=concept_totals,
            unit_household=unit_household,
            household_state=household_state,
        )
        rebuilt = materialize(entry, exact=False, **kwargs)
        reference = release_columns[row]
        scale = max(float(np.abs(reference).max()), 1.0)
        rebuild_gap = max(rebuild_gap, float(np.abs(rebuilt - reference).max()) / scale)
        exact_columns.append(materialize(entry, exact=True, **kwargs))
    checks["release_concept_rebuild_max_relative_gap"] = rebuild_gap
    assert rebuild_gap < 1e-6, rebuild_gap
    exact_matrix = sparse.csr_array(np.vstack(exact_columns))
    log(
        f"rebuilt 312 release rows (max gap {rebuild_gap:.2e}); exact rows nnz={exact_matrix.nnz}",
        started,
    )

    from microcosm.build.us_runtime.target_loss_weights import target_loss_weights

    recorded_weights = np.asarray(
        [entry["target_loss_weight"] for entry in entries], dtype=np.float64
    )
    arm_weights: dict[str, np.ndarray] = {}
    arm_row_sets: dict[str, np.ndarray] = {}
    for arm in ARMS:
        rows = arm_rows(entries, arm)
        weights = target_loss_weights(
            arm_specs(entries, rows, exact=arm.startswith("concept_exact"))
        )
        full = np.full(len(entries), np.nan)
        full[rows] = weights
        arm_weights[arm] = full
        arm_row_sets[arm] = rows
        log(f"arm {arm}: {len(rows)} targets", started)
    release_gap = float(
        np.max(np.abs(arm_weights["release"] - recorded_weights) / recorded_weights)
    )
    checks["release_loss_weights_max_relative_gap"] = release_gap
    assert release_gap < 1e-9, release_gap
    exact_gap = float(
        np.max(
            np.abs(arm_weights["concept_exact"] - recorded_weights) / recorded_weights
        )
    )
    checks["concept_exact_loss_weights_max_relative_gap"] = exact_gap

    meta = pd.DataFrame(
        {
            "row": np.arange(len(entries)),
            "name": [entry["target_name"] for entry in entries],
            "value": [float(entry["compiled_target"]) for entry in entries],
            "family": [entry["registry"]["family"] for entry in entries],
            "measure_id": [measure_id(entry) for entry in entries],
            "variable": [entry["metadata"].get("variable", "") for entry in entries],
            "geography_level": [
                entry["hierarchy"]["geography"]["level"] for entry in entries
            ],
            "state_fips": [
                entry["metadata"].get("state_fips", "") for entry in entries
            ],
            "net_income_concept": [concept_stem(entry) or "" for entry in entries],
            "initial_estimate": [entry["initial_estimate"] for entry in entries],
            "published_estimate": [entry["final_estimate"] for entry in entries],
            **{f"loss_weight_{arm}": arm_weights[arm] for arm in ARMS},
        }
    )
    meta.to_parquet(out / "targets_meta.parquet", index=False)
    sparse.save_npz(out / "target_matrix.npz", matrix, compressed=True)
    np.savez_compressed(
        out / "concept_rows.npz",
        rows=np.asarray(exact_rows, dtype=np.int64),
        data=exact_matrix.data,
        indices=exact_matrix.indices,
        indptr=exact_matrix.indptr,
        shape=np.asarray(exact_matrix.shape),
    )
    household_totals = {}
    for concept, values in concept_totals.items():
        for part, unit in (
            ("pos_sum", np.maximum(values, 0.0)),
            ("neg_sum", np.minimum(values, 0.0)),
            ("pos_count", (values > 0).astype(np.float64)),
            ("neg_count", (values < 0).astype(np.float64)),
        ):
            household_totals[f"{concept}__{part}"] = household_row(
                unit, unit_household, n_households
            )
    household_totals["tax_units"] = household_row(
        np.ones(len(unit_household)), unit_household, n_households
    )
    np.savez_compressed(
        out / "concept_totals.npz", state_fips=household_state, **household_totals
    )
    np.savez_compressed(
        out / "weights.npz",
        initial=initial,
        published=published,
        **{f"rows_{arm}": arm_row_sets[arm] for arm in ARMS},
    )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "release_id": RELEASE_ID,
        "inputs": {
            str(path): sha256(path)
            for path in (CHECKPOINT, DIAGNOSTICS, CALIBRATION_NPZ)
        },
        "outputs": {
            name: sha256(out / name)
            for name in (
                "target_matrix.npz",
                "concept_rows.npz",
                "targets_meta.parquet",
                "concept_totals.npz",
                "weights.npz",
            )
        },
        "checks": checks,
        "arms": {arm: int(len(arm_row_sets[arm])) for arm in ARMS},
        "release_options": diagnostics["options"],
        "release_final_loss": diagnostics["final_loss"],
        "repository_head": head,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    log(f"wrote {out}", started)


if __name__ == "__main__":
    main()
