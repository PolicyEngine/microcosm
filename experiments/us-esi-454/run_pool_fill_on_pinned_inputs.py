"""Measure the ESI premium stage on a real stacked ASEC + ACS pool.

Runs the part of the stacked pool that decides the ESI premium inputs, on the
real pinned inputs:

1. pool the sha-pinned processed ASEC inputs as the base build does, with the
   reviewed Census person columns restored;
2. build the ACS 2024 1-year frame from the sha-pinned PUMS archives and map
   its native inputs;
3. sample both arms and assemble one stacked spine, as the pool tool does;
4. run the pool's pre-clone operators that the fill needs on the CPS-source
   rows: CPS-carried inputs, relationships and the ESI premium stage;
5. fill the ESI premium family on ACS rows with the pool's early gap-fill
   (the real weighted QRF, under a test authority limited to this family);
6. run ``with_us_esi_premium_pool_anchor``; and
7. grade each output with the pool's by-origin battery, before and after the
   anchor, and with the stage's gates.

It skips everything else the pool does (geography, the other pre-clone
operators, the PUF pass, the late producers), none of which writes or reads
the ESI premium family before the anchor. Without the housing operator the
ASEC donors carry no tenure, so the fill runs without the tenure predictor
that a full pool build has.

    uv run python experiments/us-esi-454/run_pool_fill_on_pinned_inputs.py \
        --asec-h5 2023=<census_cps_2023.h5> --asec-h5 2024=<...> \
        --asec-h5 2025=<...> \
        --census-person-dir ~/.cache/microcosm/cps/asec_education \
        --acs-household-zip <csv_hus.zip> --acs-person-zip <csv_pus.zip> \
        --sample-fraction 0.10 --sample-seed 578

Writes ``receipts/pool_fill_<first>_<last>_f<fraction>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime import stacked_spine
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_pums import AcsPumsSource, build_acs_pums_unit_frame
from microcosm.build.us_runtime.acs_sources import load_acs_source_manifest
from microcosm.build.us_runtime.asec_pool import (
    AsecSource,
    build_pooled_asec_unit_frame,
)
from microcosm.build.us_runtime.asec_sources import ASEC_SOURCE_ARTIFACTS
from microcosm.build.us_runtime.cps_carried import derive_us_cps_carried_inputs
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    with_us_esi_premium_inputs,
    with_us_esi_premium_pool_anchor,
)
from microcosm.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from microcosm.build.us_runtime.relationship_inputs import with_us_relationship_inputs
from microcosm.build.us_runtime.spm_role_source import ASEC_SPM_ROLE_SOURCES
from microcosm.build.us_runtime.stacked_spine import (
    GapFillDirection,
    assemble_stacked_spine,
)

HERE = Path(__file__).resolve().parent
FAMILY = "source_operator_esi_premiums"
EVIDENCE = "PERIDNUM"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _asec_arm(args) -> tuple[object, list[dict]]:
    paths = {
        int(year): Path(path).expanduser()
        for year, path in (value.split("=", 1) for value in args.asec_h5)
    }
    sources, inputs = [], []
    for year in sorted(paths):
        artifact = ASEC_SOURCE_ARTIFACTS[year]
        digest = _sha256(paths[year])
        if digest != artifact.sha256:
            raise SystemExit(f"{paths[year]}: sha256 {digest} is not the pinned one")
        pin = ASEC_SPM_ROLE_SOURCES[year]
        member = args.census_person_dir / pin.member
        member_digest = _sha256(member)
        if member_digest != pin.csv_sha256:
            raise SystemExit(f"{member}: sha256 {member_digest} is not the pinned one")
        sources.append(
            AsecSource(year=year, path=paths[year], census_person_source=member)
        )
        inputs.append(
            {
                "income_year": year,
                "h5": artifact.filename,
                "h5_sha256": digest,
                "census_person_member": pin.member,
                "census_person_member_sha256": member_digest,
            }
        )
    frame, _metadata = build_pooled_asec_unit_frame(
        sources, target_year=args.target_year
    )
    assert_operator_free_source_frame(frame, label="pooled ASEC")
    return frame, inputs


def _acs_arm(args) -> tuple[object, dict]:
    manifest = load_acs_source_manifest()
    digests = {
        "household": _sha256(args.acs_household_zip),
        "person": _sha256(args.acs_person_zip),
    }
    pinned = {artifact.role: artifact.sha256 for artifact in manifest.artifacts}
    if digests != pinned:
        raise SystemExit(f"ACS archives {digests} are not the pinned {pinned}")
    inputs = {"vintage": manifest.vintage, "sha256": digests}
    cache = args.acs_frame_checkpoint
    if cache is not None and cache.is_file():
        # Building the ACS unit structure takes most of a run; reuse a frame
        # this script wrote from the same pinned archives.
        loaded = load_frame_checkpoint(cache)
        if loaded.metadata != inputs:
            raise SystemExit(f"{cache} was built from {loaded.metadata}, not {inputs}")
        return loaded.frame, inputs
    built, _build = build_acs_pums_unit_frame(
        AcsPumsSource(
            household_zip=args.acs_household_zip,
            person_zip=args.acs_person_zip,
            vintage=manifest.vintage,
        )
    )
    mapped = map_acs_native_inputs(built)
    assert_operator_free_source_frame(
        mapped.frame, label="ACS", native_inputs=mapped.native_inputs
    )
    if cache is not None:
        write_frame_checkpoint(cache, mapped.frame, metadata=inputs)
    return mapped.frame, inputs


def _prepare_cps_rows(stack):
    """The pool's pre-clone operators the ESI fill depends on, in pool order."""

    operators = {
        "derive_us_cps_carried_inputs": derive_us_cps_carried_inputs,
        "with_us_relationship_inputs": lambda current: with_us_relationship_inputs(
            current, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
        ),
        "with_us_esi_premium_inputs": lambda available: (
            pool._with_pool_us_esi_premium_inputs(available, pool=stack)
        ),
    }
    staged = pool._run_source_operator_chain(
        stack,
        phase="pre_clone",
        operator_names=tuple(operators),
        operators=operators,
    )
    share = staged.receipt["suboperators"][-1]["kernel_receipt"]["anchor_share"]
    return staged.frame, share


def _authority():
    targets = tuple(sorted(US_ESI_PREMIUMS_OUTPUT_COLUMNS))
    surface = {"person": {FAMILY: targets}}
    return stacked_spine._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(
            GapFillDirection(
                name="asec_survey_to_acs",
                recipient_channel="acs",
                donor_channel="asec",
                target_families=surface,
            ),
        ),
        metric_registry={
            ("person", FAMILY, target, 0): "monetary_sign_separated"
            for target in targets
        },
    )


def _side_summary(frame) -> dict[str, dict]:
    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    cps = person[EVIDENCE].notna().to_numpy()
    wages = person[US_ESI_PREMIUMS_WAGE_COLUMN].to_numpy(dtype=np.float64)
    out: dict[str, dict] = {}
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        values = person[column].to_numpy(dtype=np.float64)
        sides = {}
        for name, rows in (("cps", cps), ("acs", ~cps)):
            positive = rows & (values > 0)
            mass = float(weights[rows].sum())
            sides[name] = {
                "rows": int(rows.sum()),
                "weighted_persons": mass,
                "weighted_total": float(weights[rows] @ values[rows]),
                "positive_share": float(weights[positive].sum()) / mass,
                "mean_per_positive_person": float(
                    weights[positive] @ values[positive] / weights[positive].sum()
                ),
                "positive_rows_without_wages": int((positive & ~(wages > 0)).sum()),
                "positive_weight_without_wages_share": float(
                    weights[positive & ~(wages > 0)].sum() / weights[positive].sum()
                ),
            }
        out[column] = sides
    return out


def _battery(frame, authority) -> dict:
    gate = stacked_spine._by_origin_battery_with_test_authority(
        frame, authority=authority
    )
    return {
        "passed": gate.passed,
        "failures": list(gate.failures),
        "comparisons": _jsonable(gate.details.get("comparisons", gate.details)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asec-h5", action="append", required=True)
    parser.add_argument("--census-person-dir", type=Path, required=True)
    parser.add_argument("--acs-household-zip", type=Path, required=True)
    parser.add_argument("--acs-person-zip", type=Path, required=True)
    parser.add_argument(
        "--acs-frame-checkpoint",
        type=Path,
        help="Reuse (or write) the built, native-mapped ACS frame here.",
    )
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--sample-seed", type=int, default=578)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--target-year", type=int, default=pool.POOL_TIME_PERIOD)
    args = parser.parse_args()
    started = time.perf_counter()

    asec, asec_inputs = _asec_arm(args)
    acs, acs_inputs = _acs_arm(args)
    stack = assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
    ).frame
    del acs
    staged, share = _prepare_cps_rows(stack)
    authority = _authority()
    filled = stacked_spine._gap_fill_stacked_spine_with_test_authority(
        staged,
        authority=authority,
        seed=pool.POOL_RANDOM_SEED,
        n_estimators=args.n_estimators,
    )
    anchored, anchor_receipt = with_us_esi_premium_pool_anchor(filled.frame)

    alone = with_us_esi_premium_inputs(
        asec, seed=pool.POOL_RANDOM_SEED, time_period=args.target_year
    )
    alone_weights = np.asarray(alone.resolve_weights("person").values, dtype=np.float64)
    alone_total = float(
        alone_weights
        @ alone.table("person")[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=float)
    )
    pool_weights = np.asarray(
        anchored.resolve_weights("person").values, dtype=np.float64
    )
    pool_total = float(
        pool_weights
        @ anchored.table("person")[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=float)
    )
    signal = us_esi_premiums_signal_gate(anchored)
    anchor_gate = us_esi_premiums_anchor_gate(anchored, time_period=args.target_year)
    receipt = {
        "issue": "PolicyEngine/microcosm#454",
        "asec_inputs": asec_inputs,
        "acs_inputs": acs_inputs,
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "n_estimators": args.n_estimators,
        "target_year": args.target_year,
        "cps_household_mass_share": share,
        "anchor": esi._anchor(args.target_year),
        "single_source_employer_premium_total": alone_total,
        "pool_employer_premium_total": pool_total,
        "pool_over_single_source_minus_one": pool_total / alone_total - 1.0,
        "gap_fill_targets": _jsonable(
            filled.receipt["directions"]["asec_survey_to_acs"]["targets"]
        ),
        "pool_anchor": _jsonable(anchor_receipt),
        "before_anchor": {
            "sides": _side_summary(filled.frame),
            "by_origin_battery": _battery(filled.frame, authority),
        },
        "after_anchor": {
            "sides": _side_summary(anchored),
            "by_origin_battery": _battery(anchored, authority),
        },
        "signal_gate": {"passed": signal.passed, "failures": list(signal.failures)},
        "anchor_gate": {
            "passed": anchor_gate.passed,
            "failures": list(anchor_gate.failures),
        },
        "seconds": time.perf_counter() - started,
    }
    years = sorted(item["income_year"] for item in asec_inputs)
    token = f"{args.sample_fraction:.2f}".replace(".", "")
    out = HERE / "receipts" / f"pool_fill_{years[0]}_{years[-1]}_f{token}.json"
    out.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(receipt), indent=2, sort_keys=True)[:6000])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
