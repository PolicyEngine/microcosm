#!/usr/bin/env python3
"""Prove the SPM independence role on real populations, read-only.

Two receipts behind ``docs/us-spm-role-stage.md``:

``base``
    Derive the role for a base built from raw sources (the phase-2 base at
    ``~/PolicyEngine/_buildq-runtime/out/base-q3/``) from the pinned complete
    Census ASEC person CSVs, through the certified derivation
    (``microcosm.build.us_runtime.spm_role_source.derive_spm_role_source``),
    attach it to the base frame, and show ``check_spm_composition`` go from
    its FAIL count to PASS with no unit re-grouped and no person invented.

``buildp``
    Run the same derivation against the reviewed Build P parent and compare
    the result person for person with the enrichment lane's pinned reference
    evidence (``source_spm_person_independence.csv``, SHA-256 pinned in
    ``microcosm.data.source_enrichment.SOURCE_EVIDENCE_SHA256``).

Nothing is written except the JSON receipt. The H5 inputs are opened
read-only and their digests are checked before and after every read (the
derivation does this itself). Run:

    uv run python experiments/spm_role_stage_proof.py base \
        --base-h5 ~/PolicyEngine/_buildq-runtime/out/base-q3/base_populace_us_2024_puf_support.h5 \
        --out experiments/893-spm-role-stage-base-q3-receipt.json

    uv run python experiments/spm_role_stage_proof.py buildp \
        --parent-h5 ~/.cache/huggingface/hub/datasets--policyengine--populace-us/blobs/<parent sha> \
        --reference-evidence-csv <certified source_spm_person_independence.csv> \
        --out experiments/893-spm-role-stage-buildp-agreement-receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.release_gate_preflight import check_spm_composition
from microcosm.build.us_runtime.spm_role_source import (
    ASEC_SPM_ROLE_SOURCES,
    NATIVE_SPM_ROLE,
    derive_spm_role_source,
)
from microcosm.frame import Frame

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CACHE = Path.home() / ".cache" / "microcosm" / "cps" / "asec_education"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(path: Path):
    """The release tool's own loader, so the check reads what a release reads."""

    tools = REPOSITORY_ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import build_us_fiscal_refresh_release as release  # noqa: PLC0415

    return release._load_frame(path)


def _with_role(frame: Frame, role: np.ndarray) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][NATIVE_SPM_ROLE] = np.asarray(role, dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _check(frame: Frame) -> dict:
    result = check_spm_composition(frame)
    return {
        "status": result.status,
        "summary": result.summary,
        "n_units": result.details["n_units"],
        "n_units_without_classified_adult": result.details[
            "n_units_without_classified_adult"
        ],
        "n_units_without_member_aged_18_or_over": result.details[
            "n_units_without_member_aged_18_or_over"
        ],
        "role_source": result.details["role_source"],
        "fallback_columns_present": result.details["fallback_columns_present"],
        "offending_unit_ids_reported": [
            row["spm_unit_id"] for row in result.rows
        ],
    }


def _source_paths(cache: Path) -> dict[int, Path]:
    return {year: cache / pin.member for year, pin in ASEC_SPM_ROLE_SOURCES.items()}


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        **{
            name: version(name)
            for name in (
                "policyengine-us",
                "spm-calculator",
                "policyengine-core",
                "pandas",
                "numpy",
                "tables",
            )
        },
    }


def _role_shares(person: pd.DataFrame, role: np.ndarray) -> dict:
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    minor = (age >= 15.0) & (age < 18.0)
    years = pd.to_numeric(person["source_year"], errors="coerce").astype(int)
    by_year = {}
    for year in sorted(years.unique()):
        mask = (years == year).to_numpy()
        by_year[str(int(year))] = {
            "persons": int(mask.sum()),
            "role_share": float(role[mask].mean()),
            "persons_aged_15_to_17": int((mask & minor).sum()),
            "role_share_aged_15_to_17": float(role[mask & minor].mean()),
        }
    return {
        "persons": int(len(person)),
        "role_true": int(role.sum()),
        "role_share": float(role.mean()),
        "persons_aged_15_to_17": int(minor.sum()),
        "role_true_aged_15_to_17": int((role & minor).sum()),
        "role_share_aged_15_to_17": float(role[minor].mean()),
        "by_source_year": by_year,
    }


def _clone_agreement(person: pd.DataFrame, role: np.ndarray) -> dict:
    """Support clones repeat a source person; their roles must agree."""

    key = pd.DataFrame(
        {
            "source_year": person["source_year"].to_numpy(),
            "source_person_id": person["source_person_id"].astype(str).to_numpy(),
            "role": role,
        }
    )
    per_source = key.groupby(["source_year", "source_person_id"], sort=False)[
        "role"
    ].nunique()
    channels = (
        person["person_support_channel"].astype(str).value_counts().to_dict()
        if "person_support_channel" in person
        else {}
    )
    return {
        "distinct_source_persons": int(len(per_source)),
        "source_persons_with_disagreeing_clone_roles": int((per_source > 1).sum()),
        "person_support_channels": {str(k): int(v) for k, v in channels.items()},
    }


def _resolution_audit(before: Frame, after: Frame, role: np.ndarray) -> dict:
    """Every offending unit is resolved by its own 15-to-17-year-old, not a new person."""

    person_before = before.table("person")
    person_after = after.table("person")
    age = pd.to_numeric(person_after["age"], errors="coerce").to_numpy(np.float64)
    membership = person_after["person_spm_unit_id"].to_numpy()
    adult_18 = age >= 18.0
    classified = adult_18 | ((age >= 15.0) & role)
    units = pd.DataFrame(
        {"unit": membership, "adult_18": adult_18, "classified": classified}
    ).groupby("unit", sort=False)[["adult_18", "classified"]].sum()
    no_adult_18 = units.index[units["adult_18"] == 0]
    member_of = np.isin(membership, no_adult_18.to_numpy())
    resolvers = member_of & (age >= 15.0) & (age < 18.0) & role
    return {
        "persons_unchanged": int(len(person_before)) == int(len(person_after)),
        "person_ids_unchanged": bool(
            np.array_equal(
                person_before["person_id"].to_numpy(),
                person_after["person_id"].to_numpy(),
            )
        ),
        "spm_membership_unchanged": bool(
            np.array_equal(
                person_before["person_spm_unit_id"].to_numpy(),
                person_after["person_spm_unit_id"].to_numpy(),
            )
        ),
        "spm_unit_table_unchanged": before.table("spm_unit").equals(
            after.table("spm_unit")
        ),
        "ages_unchanged": bool(
            np.array_equal(
                pd.to_numeric(person_before["age"], errors="coerce").to_numpy(),
                pd.to_numeric(person_after["age"], errors="coerce").to_numpy(),
            )
        ),
        "persons_aged_18_or_over_unchanged": int(adult_18.sum())
        == int(
            (
                pd.to_numeric(person_before["age"], errors="coerce").to_numpy(
                    np.float64
                )
                >= 18.0
            ).sum()
        ),
        "units_without_member_aged_18_or_over": int(len(no_adult_18)),
        "of_which_resolved_by_own_15_to_17_year_old_with_role": int(
            (units.loc[no_adult_18, "classified"] > 0).sum()
        ),
        "of_which_still_unclassified": int(
            (units.loc[no_adult_18, "classified"] == 0).sum()
        ),
        "resolving_persons_aged_15_to_17": int(resolvers.sum()),
    }


def prove_base(args: argparse.Namespace) -> dict:
    base = args.base_h5.expanduser().resolve()
    started = time.perf_counter()
    sha = _sha256(base)
    sidecar = base.with_name(base.name + ".sha256")
    sidecar_sha = (
        sidecar.read_text().split()[0] if sidecar.exists() else None
    )
    source_paths = _source_paths(args.source_cache.expanduser())

    t0 = time.perf_counter()
    frame = _load_frame(base)
    load_seconds = time.perf_counter() - t0
    before = _check(frame)

    t0 = time.perf_counter()
    result = derive_spm_role_source(base, source_paths, expected_parent_sha256=sha)
    derive_seconds = time.perf_counter() - t0

    person = frame.table("person")
    order_preserved = bool(
        np.array_equal(
            result.evidence["person_id"].to_numpy(), person["person_id"].to_numpy()
        )
    )
    if not order_preserved:
        raise SystemExit("Derived role order does not match the frame's person order.")
    enriched = _with_role(frame, result.role)
    after = _check(enriched)

    receipt = {
        "receipt": "spm_role_stage_proof.base",
        "base_h5": str(base),
        "base_sha256": sha,
        "base_sha256_sidecar": sidecar_sha,
        "base_sha256_matches_sidecar": sidecar_sha == sha,
        "source_csvs": {
            str(year): {
                "path": str(path),
                "pinned_sha256": ASEC_SPM_ROLE_SOURCES[year].csv_sha256,
                "size_bytes": path.stat().st_size,
            }
            for year, path in sorted(source_paths.items())
        },
        "environment": _environment(),
        "frame": {
            "households": int(frame.n("household")),
            "persons": int(frame.n("person")),
            "spm_units": int(frame.n("spm_unit")),
        },
        "before": before,
        "derivation": result.provenance,
        "after": after,
        "role": _role_shares(person, result.role),
        "clones": _clone_agreement(person, result.role),
        "resolution": _resolution_audit(frame, enriched, result.role),
        "timing_seconds": {
            "frame_load": round(load_seconds, 2),
            "derive_role": round(derive_seconds, 2),
            "total": round(time.perf_counter() - started, 2),
        },
    }
    receipt["base_sha256_after"] = _sha256(base)
    receipt["base_unchanged"] = receipt["base_sha256_after"] == sha
    return receipt


def prove_buildp(args: argparse.Namespace) -> dict:
    from microcosm.data.source_enrichment import (
        EXPECTED_COUNTS,
        PARENT_BUILD_ID,
        PARENT_DATASET_SHA256,
        SOURCE_EVIDENCE_SHA256,
    )

    parent = args.parent_h5.expanduser().resolve()
    reference = args.reference_evidence_csv.expanduser().resolve()
    started = time.perf_counter()
    parent_sha = _sha256(parent)
    if parent_sha != PARENT_DATASET_SHA256:
        raise SystemExit(
            f"{parent} is not the reviewed Build P parent "
            f"({parent_sha} != {PARENT_DATASET_SHA256})."
        )
    reference_bytes = reference.read_bytes()
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    source_paths = _source_paths(args.source_cache.expanduser())

    t0 = time.perf_counter()
    result = derive_spm_role_source(
        parent, source_paths, expected_parent_sha256=PARENT_DATASET_SHA256
    )
    derive_seconds = time.perf_counter() - t0
    evidence_bytes = result.evidence.to_csv(index=False, lineterminator="\n").encode()
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()

    certified = pd.read_csv(reference)
    merged = result.evidence.merge(
        certified,
        on=["person_id", "person_spm_unit_id"],
        how="outer",
        suffixes=("_stage", "_certified"),
        indicator=True,
    )
    role_column = [c for c in certified.columns if c.startswith("is_spm")][0]
    both = merged["_merge"].eq("both")
    equal = (
        merged.loc[both, f"{role_column}_stage"].astype(bool)
        == merged.loc[both, f"{role_column}_certified"].astype(bool)
    )

    t0 = time.perf_counter()
    frame = _load_frame(parent)
    load_seconds = time.perf_counter() - t0
    before = _check(frame)
    after = _check(_with_role(frame, result.role))

    receipt = {
        "receipt": "spm_role_stage_proof.buildp",
        "parent_build_id": PARENT_BUILD_ID,
        "parent_h5": str(parent),
        "parent_sha256": parent_sha,
        "reference_evidence_csv": str(reference),
        "reference_evidence_sha256": reference_sha,
        "reference_evidence_sha256_pinned": SOURCE_EVIDENCE_SHA256,
        "reference_matches_pin": reference_sha == SOURCE_EVIDENCE_SHA256,
        "environment": _environment(),
        "derivation": result.provenance,
        "expected_counts": dict(EXPECTED_COUNTS),
        "expected_counts_match": {
            key: result.provenance.get(key) == value
            for key, value in EXPECTED_COUNTS.items()
        },
        "agreement": {
            "stage_evidence_sha256": evidence_sha,
            "stage_evidence_bytes_equal_reference": evidence_bytes == reference_bytes,
            "rows_stage": int(len(result.evidence)),
            "rows_certified": int(len(certified)),
            "rows_in_both": int(both.sum()),
            "rows_only_in_stage": int(merged["_merge"].eq("left_only").sum()),
            "rows_only_in_certified": int(merged["_merge"].eq("right_only").sum()),
            "roles_equal": int(equal.sum()),
            "roles_differ": int((~equal).sum()),
        },
        "frame": {
            "households": int(frame.n("household")),
            "persons": int(frame.n("person")),
            "spm_units": int(frame.n("spm_unit")),
        },
        "before": before,
        "after": after,
        "timing_seconds": {
            "frame_load": round(load_seconds, 2),
            "derive_role": round(derive_seconds, 2),
            "total": round(time.perf_counter() - started, 2),
        },
    }
    receipt["parent_sha256_after"] = _sha256(parent)
    receipt["parent_unchanged"] = receipt["parent_sha256_after"] == parent_sha
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("base")
    base.add_argument("--base-h5", type=Path, required=True)
    base.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    base.add_argument("--out", type=Path, required=True)
    buildp = sub.add_parser("buildp")
    buildp.add_argument("--parent-h5", type=Path, required=True)
    buildp.add_argument("--reference-evidence-csv", type=Path, required=True)
    buildp.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    buildp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prove_base(args) if args.command == "base" else prove_buildp(args)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("before", "after") if k in receipt}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
