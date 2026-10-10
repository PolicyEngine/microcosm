"""Run the meps_esi_premiums stage on a real pinned ASEC pool.

Pools sha-pinned processed ASEC H5 inputs exactly as the base build does,
restoring the reviewed Census person columns (#720, #454) from the pinned
official person members, then runs the stage, its signal gate, the release
anchor gate (pre-calibration) and the PUF-support clone, and writes the
measured lineage to ``receipts/stage_on_pool_<first>_<last>.json``.

    uv run python experiments/us-esi-454/run_stage_on_pinned_pool.py \
        --asec-h5 2023=<census_cps_2023.h5> --asec-h5 2024=<...> \
        --asec-h5 2025=<...> \
        --census-person-dir ~/.cache/microcosm/cps/asec_education

The default pool is income years 2023-2025 (``docs/us-asec-source-pins.md``);
2022-2024 is the historical Build J/N/P pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.asec_pool import (
    AsecSource,
    build_pooled_asec_unit_frame,
)
from microcosm.build.us_runtime.asec_sources import ASEC_SOURCE_ARTIFACTS
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    us_esi_premiums_summary,
    with_us_esi_premium_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spm_role_source import ASEC_SPM_ROLE_SOURCES

HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _by_vintage(frame, years) -> dict[str, dict[str, float]]:
    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    out = {}
    for year in years:
        mask = person["source_year"].to_numpy() == year
        employer = person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy()
        holders = mask & (employer > 0)
        out[str(year)] = {
            "weighted_persons": float(weights[mask].sum()),
            "employer_premium_total": float(weights[mask] @ employer[mask]),
            "employer_premium_positive_persons": float(weights[holders].sum()),
            "mean_per_positive_person": float(
                weights[holders] @ employer[holders] / weights[holders].sum()
            ),
        }
    return out


def _other_policyholder_sensitivity(frame, target_year: int) -> dict[str, object]:
    """How the employed column moves with the price of non-employed coverage.

    No MEPS-IC table prices retiree coverage, so the stage prices every
    policyholder outside the column as an active employee. This re-scales
    the anchor universe under two alternatives: the 65-and-over policyholders
    outside the column priced at half the active cell, and nobody outside the
    column priced at all (the whole anchor loaded onto employed workers). It
    also restates the as-built column and the all-on-employed reading against
    BEA NIPA 7.8 line 17, the second estimate of the anchor's concept.
    """

    person = esi._person_with_state(frame)
    codes, raw, _premium, _contribution = esi._person_raw_shares(person)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    universe = codes.universe
    other = codes.policyholder & ~universe
    age = person["A_AGE"].to_numpy(dtype=np.float64)
    older = other & (age >= 65)
    anchor = float(esi.EMPLOYER_PREMIUM_ANCHOR["values"][str(target_year)])
    bea = float(esi.EMPLOYER_PREMIUM_CROSS_CHECK["values"][str(target_year)])
    employed = float(weights[universe] @ raw[universe])
    everyone = float(weights @ raw)
    older_raw = float(weights[older] @ raw[older])
    positive = float(weights[universe & (raw > 0)].sum())
    pemlr = person["PEMLR"].to_numpy(dtype=np.int64)

    def column(total_raw: float, total_anchor: float = anchor) -> dict[str, float]:
        total = total_anchor * employed / total_raw
        return {
            "employer_premium_total": total,
            "mean_per_positive_person": total / positive,
        }

    return {
        "as_built_other_priced_as_active": column(everyone),
        "other_65_plus_priced_at_half": column(everyone - 0.5 * older_raw),
        "other_not_priced_all_anchor_on_employed": column(employed),
        "bea_anchor_other_priced_as_active": column(everyone, bea),
        "bea_anchor_all_on_employed": column(employed, bea),
        "weighted_other_policyholders_by_pemlr": {
            str(code): float(weights[other & (pemlr == code)].sum())
            for code in sorted(set(pemlr[other].tolist()))
        },
        "weighted_other_policyholders_65_plus": float(weights[older].sum()),
        "raw_other_65_plus_share_of_other": older_raw / (everyone - employed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asec-h5",
        action="append",
        required=True,
        help="Pinned processed ASEC input as YEAR=PATH; once per pooled year.",
    )
    parser.add_argument("--census-person-dir", type=Path, required=True)
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = {
        int(year): Path(path).expanduser()
        for year, path in (value.split("=", 1) for value in args.asec_h5)
    }
    years = tuple(sorted(paths))
    started = time.perf_counter()
    sources = []
    inputs = []
    for year in years:
        artifact = ASEC_SOURCE_ARTIFACTS[year]
        h5 = paths[year]
        digest = _sha256(h5)
        if digest != artifact.sha256:
            raise SystemExit(
                f"{h5}: sha256 {digest} is not the pinned {artifact.sha256}"
            )
        pin = ASEC_SPM_ROLE_SOURCES[year]
        member = args.census_person_dir / pin.member
        member_digest = _sha256(member)
        if member_digest != pin.csv_sha256:
            raise SystemExit(
                f"{member}: sha256 {member_digest} is not {pin.csv_sha256}"
            )
        sources.append(AsecSource(year=year, path=h5, census_person_source=member))
        inputs.append(
            {
                "income_year": year,
                "h5": artifact.filename,
                "h5_sha256": digest,
                "census_person_member": pin.member,
                "census_person_member_sha256": member_digest,
                "census_archive_sha256": pin.archive_sha256,
            }
        )
    frame, metadata = build_pooled_asec_unit_frame(
        sources, target_year=args.target_year
    )
    staged = with_us_esi_premium_inputs(
        frame, seed=args.seed, time_period=args.target_year
    )
    signal = us_esi_premiums_signal_gate(staged)
    anchor = us_esi_premiums_anchor_gate(staged, time_period=args.target_year)
    cloned = clone_us_frame_for_puf_support(staged)
    cloned_signal = us_esi_premiums_signal_gate(cloned)
    cloned_anchor = us_esi_premiums_anchor_gate(cloned, time_period=args.target_year)
    receipt = {
        "issue": "PolicyEngine/microcosm#454",
        "inputs": inputs,
        "seed": args.seed,
        "target_year": args.target_year,
        "pooled_income_years": list(years),
        "pool_census_person_columns": [
            {
                key: _jsonable(source["census_person_columns"][key])
                for key in (
                    "income_year",
                    "member",
                    "member_sha256",
                    "archive_sha256",
                    "joined_person_rows",
                    "columns_added",
                    "columns_verified_equal",
                    "member_values",
                )
            }
            for source in metadata["sources"]
        ],
        "summary": _jsonable(us_esi_premiums_summary(staged)),
        "by_vintage": _by_vintage(staged, years),
        "other_policyholder_sensitivity": _other_policyholder_sensitivity(
            staged, args.target_year
        ),
        "signal_gate": {"passed": signal.passed, "failures": list(signal.failures)},
        "anchor_gate": {
            "passed": anchor.passed,
            "failures": list(anchor.failures),
            "details": _jsonable(dict(anchor.details)),
        },
        "puf_support_clone": {
            "signal_gate": {
                "passed": cloned_signal.passed,
                "failures": list(cloned_signal.failures),
            },
            "anchor_gate": {
                "passed": cloned_anchor.passed,
                "failures": list(cloned_anchor.failures),
                **{
                    key: cloned_anchor.details.get(key)
                    for key in (
                        "employer_premium_total",
                        "anchor_universe_employer_total",
                    )
                },
            },
            "person_rows": len(cloned.table("person")),
        },
        "person_rows": len(staged.table("person")),
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = HERE / "receipts" / f"stage_on_pool_{years[0]}_{years[-1]}.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("signal_gate", "anchor_gate")}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
