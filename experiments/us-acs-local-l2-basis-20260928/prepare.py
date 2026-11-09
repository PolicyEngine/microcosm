"""Build the L2-basis sweep inputs next to the copied sparse checkpoint.

Inputs (read only):

* ``checkpoint/target_matrix.npz``, ``targets.json``, ``target_frame_lean.h5``
  and ``identity.json``: the sparse copy of the 09-23 ACS local release's dense
  calibration checkpoint (verified by ``verify_csr.py``, which must have passed).
* The 09-23 staging H5, for each household's spine (``acs_2024_1yr`` or the
  ``asec_puf`` donor spine).
* The 09-23 ``calibration_diagnostics.json`` and ``weights_latest.npz``, for
  the released per-target estimates.

Outputs (under the checkpoint directory):

* ``households.parquet``: one row per CSR column, in CSR column order —
  ``household_id``, ``state_fips`` (2-digit string), ``congressional_district_geoid``
  (4-digit string), ``spine`` and ``design_weight``.
* ``targets_meta.parquet``: one row per CSR row — ``row``, ``name``, ``value``,
  the target taxonomy (``family``, ``subfamily`` and the SOI fields parsed from
  the target name), geography (``geography_level``, ``geography_code``,
  ``state_fips``, ``state_postal``, ``congressional_district_geoid``), ``nnz``,
  ``fold`` (rotated holdout fold), and the design and released 09-23 estimates.
* ``holdout_folds.npz``: ``microcosm.build.holdout.rotated_folds(4459,
  n_folds=5, seed=20260529)`` as ``fold_0`` .. ``fold_4`` plus ``fold_of_target``.
* ``MANIFEST.json``: sha256 of every file above plus provenance and the
  structural checks below.

Checks (the script exits non-zero if any fails): the staging household ids
equal the checkpoint's; the staging spine equals the structure H5's
``household_spine``; every target parses into the taxonomy; every SOI name's
state postal code matches its ``state_fips``; every target's nonzero households
lie inside the target's geography (state or congressional district).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
RUN_0923 = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/run"
)
DENSE_DIR = RUN_0923 / "release" / "checkpoints"
STAGING_H5 = RUN_0923 / "staging" / "acs_multispine_staging.h5"
COPY_SOURCE = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-state-cd-20260927/state"
)
RELEASE_TAG = "populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z"
HOLDOUT_N_FOLDS = 5
HOLDOUT_SEED = 20260529
SPINES = ("acs_2024_1yr", "asec_puf")

# SOI Historic Table 2 variables on the state_broad table, grouped by concept.
SOI_CONCEPTS = {
    "income": (
        "adjusted_gross_income",
        "wages_salaries",
        "taxable_interest",
        "tax_exempt_interest",
        "ordinary_dividends",
        "qualified_dividends",
        "net_capital_gains",
        "schedule_c_income",
        "partnership_scorp_income",
        "rental_royalty_income",
        "taxable_ira_distributions",
        "taxable_pension_income",
        "taxable_social_security",
        "unemployment_compensation",
    ),
    "deduction": (
        "itemized_deductions",
        "limited_state_local_taxes",
        "real_estate_taxes",
        "medical_dental_expense",
    ),
    "tax": (
        "taxable_income",
        "income_tax_before_credits",
        "income_tax_liability",
    ),
    "credit": ("ctc", "actc", "premium_tax_credit"),
    "eitc": (
        "eitc",
        "eitc_no_children",
        "eitc_one_child",
        "eitc_two_children",
        "eitc_three_or_more_children",
    ),
}
CONCEPT_OF = {
    base: concept for concept, bases in SOI_CONCEPTS.items() for base in bases
}
AGI_BANDS = (
    "all",
    "under_1",
    "1_to_10k",
    "10k_to_25k",
    "25k_to_50k",
    "50k_to_75k",
    "75k_to_100k",
    "100k_to_200k",
    "200k_to_500k",
    "500k_plus",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "kernel_dirty_paths": run(
            "status",
            "--porcelain",
            "--",
            "packages/microcosm-calibrate",
            "packages/microcosm-frame",
            "packages/microcosm-build",
        ).splitlines(),
    }


def soi_fields(name: str, source_measure_id: str) -> dict:
    """Parse ``irs_soi.ty2022.historic_table_2.<table>.<st>.<band|st>.<variable>``."""

    parts = name.split(".")
    if len(parts) != 7 or parts[:3] != ["irs_soi", "ty2022", "historic_table_2"]:
        raise ValueError(f"unexpected SOI target name {name!r}")
    table, postal, qualifier, variable = parts[3], parts[4], parts[5], parts[6]
    if variable != source_measure_id:
        raise ValueError(f"{name}: variable {variable!r} != {source_measure_id!r}")
    if table == "state_eitc":
        if qualifier != postal:
            raise ValueError(
                f"{name}: state_eitc qualifier {qualifier!r} != {postal!r}"
            )
        band = "all"
    elif table in ("state_broad", "state_agi"):
        band = qualifier
    else:
        raise ValueError(f"{name}: unknown SOI table {table!r}")
    if band not in AGI_BANDS or (table == "state_agi") == (band == "all"):
        raise ValueError(f"{name}: unexpected AGI band {band!r} on {table}")
    if variable == "return_count":
        base, kind = "return_count", "count"
    elif variable == "adjusted_gross_income":
        base, kind = "adjusted_gross_income", "amount"
    elif variable.endswith("_amount"):
        base, kind = variable.removesuffix("_amount"), "amount"
    elif variable.endswith("_returns"):
        base, kind = variable.removesuffix("_returns"), "count"
    elif variable.endswith("_claims"):
        base, kind = variable.removesuffix("_claims"), "count"
    else:
        raise ValueError(f"{name}: cannot classify SOI variable {variable!r}")
    if base == "return_count":
        concept = "filing"
    else:
        concept = CONCEPT_OF.get(base)
        if concept is None:
            raise ValueError(f"{name}: SOI base {base!r} has no concept")
    if table == "state_agi":
        subfamily = f"soi_agi_band_{kind}"
    elif table == "state_eitc":
        if concept != "eitc":
            raise ValueError(f"{name}: state_eitc row outside the EITC concept")
        subfamily = f"soi_eitc_{kind}"
    elif concept == "filing":
        subfamily = "soi_return_count"
    else:
        if concept == "eitc":
            raise ValueError(f"{name}: EITC variable on the state_broad table")
        subfamily = f"soi_{concept}_{kind}"
    return {
        "subfamily": subfamily,
        "soi_table": table,
        "soi_variable": variable,
        "soi_base": base,
        "soi_kind": kind,
        "soi_concept": concept,
        "agi_band": band,
        "name_postal": postal.upper(),
    }


def classify(record: dict) -> dict:
    source_family = record["family"]
    level = record["geography_level"]
    measure_id = record.get("source_measure_id")
    fields: dict = {
        "source_family": source_family,
        "source_measure_id": measure_id,
        "soi_table": None,
        "soi_variable": None,
        "soi_base": None,
        "soi_kind": None,
        "soi_concept": None,
        "agi_band": None,
    }
    if source_family == "usda_snap":
        subfamily = {
            "average_monthly_households": "snap_households",
            "total_benefits": "snap_benefits",
        }[measure_id]
        fields.update(family="snap", subfamily=subfamily)
    elif source_family == "cms_medicaid":
        if measure_id != "total_medicaid_enrollment":
            raise ValueError(f"{record['name']}: unexpected Medicaid measure")
        fields.update(family="medicaid", subfamily="medicaid_enrollment")
    elif source_family == "census_population_ladder":
        family = {"state": "pop_state", "congressional_district": "pop_cd"}[level]
        expected = (
            f"pop_state_{record['state_fips']}"
            if family == "pop_state"
            else f"pop_cd_{record['congressional_district_geoid']}"
        )
        if record["name"] != expected:
            raise ValueError(f"{record['name']}: expected {expected}")
        fields.update(family=family, subfamily=family)
    elif source_family == "irs_soi":
        parsed = soi_fields(record["name"], measure_id)
        name_postal = parsed.pop("name_postal")
        fields.update(family="soi", **parsed)
        fields["_name_postal"] = name_postal
    else:
        raise ValueError(f"{record['name']}: unknown family {source_family!r}")
    return fields


def main() -> int:
    started = time.time()
    failures: list[str] = []
    verification = json.loads((HERE / "results" / "csr_verification.json").read_text())
    if not verification.get("ok"):
        raise SystemExit("csr_verification.json is not ok; run verify_csr.py first")

    from microcosm.build.holdout import rotated_folds
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL

    matrix = sparse.load_npz(CHECKPOINT / "target_matrix.npz").tocsr()
    records = json.loads((CHECKPOINT / "targets.json").read_text())
    n_targets, n_households = matrix.shape
    if len(records) != n_targets:
        raise SystemExit("targets.json length differs from the CSR row count")

    # ---- households ------------------------------------------------------
    struct = pd.read_hdf(CHECKPOINT / "target_frame_lean.h5", "household")
    if len(struct) != n_households:
        raise SystemExit("structure H5 household count differs from the CSR")
    household_id = struct["household_id"].to_numpy(np.int64)
    if not np.all(np.diff(household_id) > 0):
        raise SystemExit("household ids are not strictly increasing")

    log("reading staging spine")
    with pd.HDFStore(STAGING_H5, mode="r") as store:
        storer = store.get_storer("household")
        group = storer.group

        def items(block: str) -> list[str]:
            return [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in getattr(group, f"{block}_items")[:]
            ]

        spine_block = next(
            f"block{i}"
            for i in range(storer.nblocks)
            if items(f"block{i}") == ["household_spine"]
        )
        id_block = next(
            f"block{i}"
            for i in range(storer.nblocks)
            if "household_id" in items(f"block{i}")
        )
        staging_spine = np.asarray(
            storer.read_array(f"{spine_block}_values"), dtype=object
        )
        staging_ids = storer.read_array(f"{id_block}_values")[
            items(id_block).index("household_id")
        ]
    staging_spine = staging_spine.astype(str)
    ok = np.array_equal(np.asarray(staging_ids, dtype=np.int64), household_id)
    if not ok:
        failures.append("staging_household_ids_equal_checkpoint")
    struct_spine = struct["household_spine"].astype(str).to_numpy()
    if not np.array_equal(staging_spine, struct_spine):
        failures.append("staging_spine_equals_structure_spine")
    spine_counts = Counter(staging_spine.tolist())
    if set(spine_counts) != set(SPINES):
        failures.append("spine_labels")

    state_fips = np.char.zfill(struct["state_fips"].to_numpy(np.int64).astype(str), 2)
    cd_geoid = np.char.zfill(
        struct["congressional_district_geoid"].to_numpy(np.int64).astype(str), 4
    )
    if not np.all(np.char.startswith(cd_geoid, state_fips)):
        failures.append("cd_geoid_prefix_equals_state_fips")
    design_weight = struct["household_weight"].to_numpy(np.float64)
    households = pd.DataFrame(
        {
            "household_id": household_id,
            "state_fips": pd.Categorical(state_fips),
            "congressional_district_geoid": pd.Categorical(cd_geoid),
            "spine": pd.Categorical(staging_spine, categories=list(SPINES)),
            "design_weight": design_weight,
        }
    )

    # ---- targets ---------------------------------------------------------
    folds = rotated_folds(n_targets, n_folds=HOLDOUT_N_FOLDS, seed=HOLDOUT_SEED)
    fold_of_target = np.full(n_targets, -1, dtype=np.int8)
    for index, fold in enumerate(folds):
        fold_of_target[fold] = index
    if (fold_of_target < 0).any():
        failures.append("folds_cover_targets")

    diagnostics = json.loads((DENSE_DIR / "calibration_diagnostics.json").read_text())
    diag = diagnostics["targets"]
    if [row["name"] for row in diag] != [record["name"] for record in records]:
        raise SystemExit("09-23 diagnostics are not row-aligned with targets.json")
    w_0923 = np.load(DENSE_DIR / "weights_latest.npz")["weights"].astype(np.float64)
    matrix64 = matrix.astype(np.float64)
    design_estimate = matrix64 @ design_weight
    release_estimate = matrix64 @ w_0923
    nnz = np.diff(matrix.indptr)

    rows = []
    for index, record in enumerate(records):
        fields = classify(record)
        name_postal = fields.pop("_name_postal", None)
        level = record["geography_level"]
        state = record["state_fips"]
        postal = US_STATE_FIPS_TO_POSTAL.get(state)
        if postal is None:
            failures.append(f"unknown_state_fips:{record['name']}")
        if name_postal is not None and name_postal != postal:
            failures.append(f"soi_name_postal_mismatch:{record['name']}")
        cd = record["congressional_district_geoid"]
        if level == "state":
            code = state
            if cd is not None:
                failures.append(f"state_target_with_cd:{record['name']}")
        elif level == "congressional_district":
            code = cd
            if cd is None or not cd.startswith(state):
                failures.append(f"cd_target_geoid:{record['name']}")
        else:
            failures.append(f"unknown_level:{record['name']}")
            code = None
        value = float(record["value"])
        rows.append(
            {
                "row": index,
                "name": record["name"],
                "value": value,
                "family": fields.pop("family"),
                "subfamily": fields.pop("subfamily"),
                **fields,
                "geography_level": level,
                "geography_code": code,
                "state_fips": state,
                "state_postal": postal,
                "congressional_district_geoid": cd,
                "nnz": int(nnz[index]),
                "fold": int(fold_of_target[index]),
                "design_estimate": float(design_estimate[index]),
                "release_0923_estimate": float(release_estimate[index]),
                "release_0923_final_estimate_diagnostics": float(
                    diag[index]["final_estimate"]
                ),
            }
        )
    meta = pd.DataFrame(rows)
    if not np.array_equal(
        meta["release_0923_estimate"].to_numpy(),
        meta["release_0923_final_estimate_diagnostics"].to_numpy(),
    ):
        failures.append("release_estimates_reproduce_diagnostics")
    meta = meta.drop(columns=["release_0923_final_estimate_diagnostics"])

    # ---- geography containment: every nonzero lies inside its geography ----
    state_codes = {code: i for i, code in enumerate(sorted(set(state_fips.tolist())))}
    cd_codes = {code: i for i, code in enumerate(sorted(set(cd_geoid.tolist())))}
    hh_state = np.asarray([state_codes[s] for s in state_fips], dtype=np.int32)
    hh_cd_series = pd.Series(cd_geoid).map(cd_codes)
    hh_cd = hh_cd_series.to_numpy(np.int32)
    target_state = np.asarray(
        [state_codes.get(s, -1) for s in meta["state_fips"]], dtype=np.int32
    )
    target_cd = np.asarray(
        [
            cd_codes.get(c, -1) if c is not None else -1
            for c in meta["congressional_district_geoid"]
        ],
        dtype=np.int32,
    )
    is_cd_target = (meta["geography_level"] == "congressional_district").to_numpy()
    row_of_nnz = np.repeat(np.arange(n_targets), nnz)
    outside = np.where(
        is_cd_target[row_of_nnz],
        hh_cd[matrix.indices] != target_cd[row_of_nnz],
        hh_state[matrix.indices] != target_state[row_of_nnz],
    )
    outside_per_row = np.bincount(row_of_nnz, weights=outside, minlength=n_targets)
    containment = {
        "rows_with_nonzeros_outside_geography": int((outside_per_row > 0).sum()),
        "nonzeros_outside_geography": int(outside.sum()),
        "empty_rows": int((nnz == 0).sum()),
        "cd_targets_with_unknown_cd": int(((target_cd < 0) & is_cd_target).sum()),
        "state_targets_with_unknown_state": int((target_state < 0).sum()),
    }
    if containment["rows_with_nonzeros_outside_geography"] or containment["empty_rows"]:
        failures.append("geography_containment")
    if (
        containment["cd_targets_with_unknown_cd"]
        or containment["state_targets_with_unknown_state"]
    ):
        failures.append("target_geography_codes_known")
    pop_cd_codes = set(
        meta.loc[meta["family"] == "pop_cd", "congressional_district_geoid"]
    )
    containment["households_cds"] = len(cd_codes)
    containment["pop_cd_targets"] = len(pop_cd_codes)
    containment["household_cds_without_pop_cd_target"] = sorted(
        set(cd_codes) - pop_cd_codes
    )
    containment["households_states"] = len(state_codes)

    # ---- write ----------------------------------------------------------
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    households.to_parquet(CHECKPOINT / "households.parquet", index=False)
    meta.to_parquet(CHECKPOINT / "targets_meta.parquet", index=False)
    np.savez(
        CHECKPOINT / "holdout_folds.npz",
        **{
            f"fold_{i}": np.asarray(fold, dtype=np.int64)
            for i, fold in enumerate(folds)
        },
        fold_of_target=fold_of_target,
        n_targets=np.int64(n_targets),
        n_folds=np.int64(HOLDOUT_N_FOLDS),
        seed=np.int64(HOLDOUT_SEED),
    )

    taxonomy = {
        "family": meta["family"].value_counts().sort_index().to_dict(),
        "subfamily": meta["subfamily"].value_counts().sort_index().to_dict(),
        "family_by_fold": {
            str(k): v
            for k, v in meta.groupby("fold")["family"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict("index")
            .items()
        },
        "soi_concept_of_base": CONCEPT_OF,
        "rules": {
            "snap": "usda_snap; subfamily by source_measure_id (households / benefits)",
            "medicaid": "cms_medicaid total_medicaid_enrollment",
            "pop_state": "census_population_ladder at state level",
            "pop_cd": "census_population_ladder at congressional_district level",
            "soi": (
                "irs_soi Historic Table 2 (TY2022). state_agi -> soi_agi_band_{count,amount} "
                "(taxable interest by AGI band); state_eitc -> soi_eitc_{count,amount}; "
                "state_broad -> soi_{income,deduction,tax,credit}_{count,amount} by "
                "variable concept, plus soi_return_count. count = *_returns, *_claims, "
                "return_count; amount = *_amount, adjusted_gross_income."
            ),
        },
    }
    spine_summary = {
        spine: {
            "n": int(spine_counts[spine]),
            "design_mass_share": float(
                design_weight[staging_spine == spine].sum() / design_weight.sum()
            ),
        }
        for spine in SPINES
    }

    output_files = [
        "households.parquet",
        "targets_meta.parquet",
        "holdout_folds.npz",
    ]
    copied = [
        "target_matrix.npz",
        "targets.json",
        "target_frame_lean.h5",
        "identity.json",
    ]
    log("hashing outputs and sources")
    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "purpose": (
            "Inputs for the chi-square L2 basis / softmax mass sweep on the "
            f"published ACS local release {RELEASE_TAG}."
        ),
        "release_tag": RELEASE_TAG,
        "files": {name: sha256(CHECKPOINT / name) for name in copied + output_files},
        "copied_from": {
            "directory": str(COPY_SOURCE),
            "files": copied,
            "note": "cp -p of the other session's sparse conversion; read-only source",
        },
        "sources": {
            "dense_checkpoint": str(DENSE_DIR),
            "dense_calibration_diagnostics_sha256": sha256(
                DENSE_DIR / "calibration_diagnostics.json"
            ),
            "dense_weights_latest_sha256": sha256(DENSE_DIR / "weights_latest.npz"),
            "dense_targets_json_sha256": sha256(DENSE_DIR / "targets.json"),
            "staging_h5": str(STAGING_H5),
            "staging_h5_sha256": sha256(STAGING_H5),
            "staging_h5_sha256_pinned_by_run_identity": json.loads(
                (DENSE_DIR / "run_identity.json").read_text()
            )["staging_sha256"],
        },
        "csr_verification": {
            "path": str(HERE / "results" / "csr_verification.json"),
            "sha256": sha256(HERE / "results" / "csr_verification.json"),
            "ok": verification["ok"],
        },
        "shape": {
            "targets": n_targets,
            "households": n_households,
            "nnz": int(matrix.nnz),
        },
        "holdout": {
            "function": "microcosm.build.holdout.rotated_folds",
            "n_targets": n_targets,
            "n_folds": HOLDOUT_N_FOLDS,
            "seed": HOLDOUT_SEED,
            "fold_sizes": [int(len(fold)) for fold in folds],
        },
        "spines": spine_summary,
        "taxonomy": taxonomy,
        "geography_containment": containment,
        "design": {
            "total": float(design_weight.sum()),
            "kish_ess": float(
                design_weight.sum() ** 2 / np.square(design_weight).sum()
            ),
        },
        "git": git_state(),
        "failures": failures,
        "wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(peak_rss_gb(), 3),
    }
    if (
        manifest["sources"]["staging_h5_sha256"]
        != manifest["sources"]["staging_h5_sha256_pinned_by_run_identity"]
    ):
        failures.append("staging_sha256_matches_run_identity")
    manifest["ok"] = not failures
    (CHECKPOINT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(
        f"wrote {CHECKPOINT / 'MANIFEST.json'} ok={manifest['ok']} "
        f"failures={failures[:10]} peak_rss_gb={manifest['peak_rss_gb']}"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("taxonomy", "geography_containment", "spines")},
            indent=1,
        )[:4000]
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
