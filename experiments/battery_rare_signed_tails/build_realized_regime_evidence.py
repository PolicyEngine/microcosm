"""Rebuild the frozen rare signed-tail regime evidence.

This is an adjudication-side extractor, not a production build step.  It reads
the immutable f025 target-bank checkpoints and the arm-split adjudication,
verifies their identities, and emits a compact, deterministic JSON projection.

The schema-1 bank predates regime persistence.  Its metadata authenticates the
fit boundary (target, availability pattern, predictors, donor index, weights,
and zero tolerance), while the sign counts below are the separately audited
fit-boundary reconstruction.  Future schema-2 banks carry the realized labels
directly and do not need this retrospective join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import h5py

AUTHORITY_SHA256 = {
    "ADJUDICATION.md": (
        "2cdf79233d56f63ca8e65305cb1c746706d88a7a57ac26ae25db575676daf3b8"
    ),
    "adjudication.json": (
        "1c1597c8080cf4e0db0079c2a3e1bc345001525bc3ec6c296aa7575b47136c32"
    ),
    "f025_gates": (
        "685cad63d4dc62234da72501c5a3ce9ec5a81fcd3f7b412b61474a9c1d8b423b"
    ),
    "transferred_receipts": (
        "c91a61bb20f0f2c2612bac8fdd21d2227cf746b85e2c5c2648857fd667e68aed"
    ),
}

EXPECTED_TRANSFERRED_CHECKPOINT = {
    "sha256": "bdc9355d92659bb28d58b1ddcd647ec303f2ad217661e17d5b4b0984e04532e8",
    "size_bytes": 5_578_512_480,
    "identity_sha256": (
        "7847224d1daa6b21cb4d063b1b54b62516b24936a302513143a685b61b7cf8e4"
    ),
}

EARLY_ORDINALS = frozenset(
    {8, 9, 10, 13, 14, 21, 23, 25, 26, 28, 36, 38, 44, 46, 50, 73, 75}
)
SPARSE_EVIDENCE_ORDINALS = frozenset({16, 28, 33, 46, 75})
RETIREMENT_CAP_ORDINALS = frozenset({78, 80, 82})

PATTERN_PROFILES = {
    "early_person": (
        "pattern_00_677f6490",
        "pattern_01_5874881e",
        "pattern_02_7c3bceda",
        "pattern_03_04f75638",
    ),
    "late_person": (
        "pattern_00_36785ebf",
        "pattern_01_c6777728",
        "pattern_02_5e7dd311",
        "pattern_03_76a0101a",
    ),
    "late_group": (
        "pattern_00_741356db",
        "pattern_01_7c62ad55",
        "pattern_02_d9f964c9",
        "pattern_03_2e5c739d",
    ),
}

# (negative, zero, positive), reconstructed at the target-complete fit boundary.
_FIT_SUPPORT_TEXT = """
2 0 107999 74
8 0 107061 1012
9 0 96432 11641
10 0 96432 11641
12 0 106243 1830
13 0 103874 4199
14 0 103874 4199
16 0 108055 18
18 0 106220 1853
21 0 92633 15440
23 0 106138 1935
25 0 107810 263
26 0 107675 398
28 0 108012 61
30 0 107873 200
32 0 108007 66
33 0 108046 27
35 0 107476 597
36 216 104057 3800
38 216 104057 3800
40 99 107534 440
42 99 107534 440
43 0 107932 141
44 89 107422 562
46 89 107422 562
48 111 107694 268
49 111 107694 268
50 0 107050 1023
52 1182 103789 3102
55 385 106982 706
56 385 106982 706
58 0 107211 862
60 0 102255 5818
62 0 101937 6136
64 0 107730 343
68 0 107676 397
70 0 107430 643
71 0 107448 625
73 48 105417 2608
75 48 105417 2608
78 0 106307 1766
80 0 108052 21
82 0 107157 916
84 0 107500 573
86 0 108028 45
88 0 42931 1030
90 0 57352 278
92 0 57295 335
"""

FIT_SUPPORT = {
    int(ordinal): (int(negative), int(zero), int(positive))
    for ordinal, negative, zero, positive in (
        line.split() for line in _FIT_SUPPORT_TEXT.splitlines() if line.strip()
    )
}

# The old 5,000-row retirement training cap retained these carriers from the
# full native ASEC target support.  Keogh is outside the 48 QED checks but uses
# the same faulty cap and is included in the output's one-sided-leg finding.
RETIREMENT_CAP_SUPPORT = {
    "keogh_distributions": {"retained": 0, "native": 2},
    "taxable_401k_distributions": {"retained": 102, "native": 2_057},
    "taxable_403b_distributions": {"retained": 4, "native": 161},
    "taxable_sep_distributions": {"retained": 2, "native": 61},
}

_LEG_RE = re.compile(
    r"/(?P<target>[^/\[]+)\[clone_0\]/(?P<sign>positive|negative)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path}: expected sha256 {expected}, got {actual}")


def _profile_name(*, ordinal: int, entity: str) -> str:
    if ordinal in EARLY_ORDINALS:
        if entity != "person":
            raise RuntimeError(f"early ordinal {ordinal} is unexpectedly {entity}")
        return "early_person"
    return "late_person" if entity == "person" else "late_group"


def _regime(negative: int, zero: int, positive: int) -> str:
    if negative and positive:
        return "three_sign"
    if zero and positive:
        return "zero_inflated_positive"
    raise RuntimeError(
        "frozen QED target unexpectedly lacks its audited gated support: "
        f"negative={negative}, zero={zero}, positive={positive}"
    )


def _mechanism(ordinal: int) -> dict[str, str]:
    if ordinal in SPARSE_EVIDENCE_ORDINALS:
        return {
            "primary": "donor_evidence_starvation",
            "smallest_honest_change": (
                "after a source-label audit supplies dense evidence, fit a "
                "target-specific rare-event/sign-and-magnitude model"
            ),
            "status": "blocked_on_additional_evidence",
        }
    if ordinal in RETIREMENT_CAP_ORDINALS:
        return {
            "primary": "upstream_retirement_support_deletion",
            "smallest_honest_change": (
                "retain the union of retirement-target carriers before the "
                "fixed-size training cap and reweight only sampled common zeros"
            ),
            "status": "implemented",
        }
    if ordinal == 2:
        return {
            "primary": "adult_care_post_transfer_amount_shape",
            "smallest_honest_change": (
                "with held-out evidence, calibrate positive amounts on newly "
                "imputed mutable cells before adult-care reconciliation"
            ),
            "status": "blocked_on_held_out_calibration_evidence",
        }
    if ordinal == 8:
        return {
            "primary": "early_transfer_unemployment_amount_shape",
            "smallest_honest_change": (
                "with predictors and carrier membership frozen, apply a "
                "target-only weighted positive-magnitude map"
            ),
            "status": "blocked_on_held_out_calibration_evidence",
        }
    if ordinal in {21, 23, 25, 26}:
        return {
            "primary": "retirement_mapping_label_or_model_shape",
            "smallest_honest_change": (
                "audit source mappings and labels on a dense rung, refit the "
                "applicable model, and run the frozen predictor factorial"
            ),
            "status": "blocked_on_mapping_labels_and_dense_evidence",
        }
    if ordinal in {58, 60}:
        return {
            "primary": "late_transfer_then_qbi_reconciliation_shape",
            "smallest_honest_change": (
                "calibrate the joint pre-cap QBI surface while preserving the "
                "existing coupled caps and identities"
            ),
            "status": "blocked_on_qbi_owner_and_held_out_evidence",
        }
    if ordinal in {62, 64}:
        return {
            "primary": "late_transfer_conditional_magnitude_shape",
            "smallest_honest_change": (
                "apply a target-scoped weighted conditional-magnitude map to "
                "mutable late-transfer draws"
            ),
            "status": "blocked_on_held_out_calibration_evidence",
        }
    if ordinal in {68, 70, 71, 73, 84, 86, 88}:
        return {
            "primary": "source_operator_conditional_magnitude_shape",
            "smallest_honest_change": (
                "apply a target-and-sign amount map within the owning source "
                "family while preserving reconciliation and producer cells"
            ),
            "status": "blocked_on_owner_specific_held_out_evidence",
        }
    return {
        "primary": "gated_conditional_magnitude_shape",
        "smallest_honest_change": (
            "apply a target-and-sign-scoped weighted conditional-magnitude map "
            "at the owning transfer boundary, mutating imputed cells only"
        ),
        "status": "blocked_on_owner_specific_held_out_evidence",
    }


def _read_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return json.loads(bytes(h5["metadata_json"][:]))


def _find_checkpoint(bank_root: Path, *, ordinal: int, target: str) -> Path:
    partition = "asec_survey_to_acs" if ordinal in EARLY_ORDINALS else "late_producer_dag"
    matches = list((bank_root / partition).rglob(f"*__{target}.h5"))
    if len(matches) != 1:
        raise RuntimeError(
            f"ordinal {ordinal} target {target}: expected one {partition} "
            f"checkpoint, found {matches}"
        )
    return matches[0]


def _checkpoint_projection(
    path: Path,
    *,
    bank_root: Path,
    expected_patterns: tuple[str, ...],
    support: tuple[int, int, int],
) -> dict[str, Any]:
    metadata = _read_metadata(path)
    if metadata.get("schema_version") != 1:
        raise RuntimeError(f"{path}: retrospective evidence expects schema 1")
    steps = metadata.get("pattern_steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise RuntimeError(f"{path}: expected four pattern steps")
    pattern_ids = tuple(step["pattern"] for step in steps)
    if pattern_ids != expected_patterns:
        raise RuntimeError(
            f"{path}: pattern inventory {pattern_ids} != {expected_patterns}"
        )
    negative, zero, positive = support
    label = _regime(negative, zero, positive)
    patterns = []
    donor_identities = set()
    for step in steps:
        state = step["state_after"]
        donor_index = state["donor_index"]
        donor_identity = (donor_index["length"], donor_index["sha256"])
        donor_identities.add(donor_identity)
        if sum(support) != donor_index["length"]:
            raise RuntimeError(
                f"{path}: support {support} does not close donor length "
                f"{donor_index['length']}"
            )
        patterns.append(
            {
                "pattern_id": step["pattern"],
                "predictors": state["predictors"],
                "recipient_rows": state["recipient_index"]["length"],
                "recipient_index_sha256": state["recipient_index"]["sha256"],
                "seed": state["model_config"]["seed"],
                "weight_kind": state["weight_kind"],
                "weight_sha256": state["weight_sha256"],
                "zero_atol": state["model_config"]["zero_atol"],
                "realized_regime": label,
            }
        )
    if len(donor_identities) != 1:
        raise RuntimeError(f"{path}: patterns do not share one donor identity")
    donor_rows, donor_index_sha256 = donor_identities.pop()
    target = metadata["target"]
    return {
        "checkpoint_relpath": path.relative_to(bank_root).as_posix(),
        "checkpoint_schema_version": metadata["schema_version"],
        "content_metadata_sha256": metadata["content_metadata_sha256"],
        "entity": target["entity"],
        "transfer_family": target["family"],
        "model_target": target["model_target"],
        "model_targets": target["model_targets"],
        "exported_targets": target["exported_targets"],
        "donor_identity": {
            "rows": donor_rows,
            "index_sha256": donor_index_sha256,
        },
        "donor_support": {
            "negative": negative,
            "zero": zero,
            "positive": positive,
            "total": sum(support),
            "evidence_kind": "fit_boundary_reconstruction",
            "common_across_patterns": True,
        },
        "patterns": patterns,
    }


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    adjudication_path = args.adjudication_dir / "adjudication.json"
    adjudication_md_path = args.adjudication_dir / "ADJUDICATION.md"
    _assert_sha256(adjudication_path, AUTHORITY_SHA256["adjudication.json"])
    _assert_sha256(adjudication_md_path, AUTHORITY_SHA256["ADJUDICATION.md"])
    _assert_sha256(args.f025_gates, AUTHORITY_SHA256["f025_gates"])
    _assert_sha256(
        args.transferred_receipts,
        AUTHORITY_SHA256["transferred_receipts"],
    )

    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    receipt = json.loads(args.transferred_receipts.read_text(encoding="utf-8"))
    checkpoint = receipt["checkpoint"]
    observed_checkpoint = {
        "sha256": checkpoint["sha256"],
        "size_bytes": checkpoint["size_bytes"],
        "identity_sha256": receipt["identity_sha256"],
    }
    if observed_checkpoint != EXPECTED_TRANSFERRED_CHECKPOINT:
        raise RuntimeError(
            "transferred checkpoint receipt differs: "
            f"{observed_checkpoint} != {EXPECTED_TRANSFERRED_CHECKPOINT}"
        )

    rows = [
        row
        for row in adjudication["adjudications"]
        if row["criterion"] == "quantile"
    ]
    if len(rows) != 48 or {row["ordinal"] for row in rows} != set(FIT_SUPPORT):
        raise RuntimeError("adjudication QED inventory differs from audited support")

    targets_by_checkpoint: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for row in rows:
        ordinal = row["ordinal"]
        match = _LEG_RE.search(row["leg_id"])
        if match is None:
            raise RuntimeError(f"cannot parse QED leg {row['leg_id']}")
        target_name = match.group("target")
        support = FIT_SUPPORT[ordinal]
        profile = _profile_name(ordinal=ordinal, entity=row["entity"])
        checkpoint_path = _find_checkpoint(
            args.bank_root,
            ordinal=ordinal,
            target=target_name,
        )
        checkpoint_relpath = checkpoint_path.relative_to(args.bank_root).as_posix()
        projection = _checkpoint_projection(
            checkpoint_path,
            bank_root=args.bank_root,
            expected_patterns=PATTERN_PROFILES[profile],
            support=support,
        )
        existing = targets_by_checkpoint.setdefault(checkpoint_relpath, projection)
        if existing != projection:
            raise RuntimeError(f"duplicate target evidence differs for {checkpoint_relpath}")

        baseline = row["baseline"]
        checks.append(
            {
                "ordinal": ordinal,
                "check_id": row["check_id"],
                "leg_id": row["leg_id"],
                "entity": row["entity"],
                "gate_family": row["family"],
                "target_ref": checkpoint_relpath,
                "tested_sign": match.group("sign"),
                "transfer_partition": (
                    "early_gap_fill"
                    if ordinal in EARLY_ORDINALS
                    else "late_producer_complement"
                ),
                "pattern_profile": profile,
                "fitter_regime": _regime(*support),
                "donor_support_starved": ordinal
                in (SPARSE_EVIDENCE_ORDINALS | RETIREMENT_CAP_ORDINALS),
                "mechanism": _mechanism(ordinal),
                "terminal": {
                    "asec_carrier_rows": baseline["nonzero_rows"]["asec"],
                    "acs_carrier_rows": baseline["nonzero_rows"]["acs"],
                    "quantile_envelope_distance": baseline[
                        "quantile_envelope_distance"
                    ],
                },
            }
        )

    targets = [targets_by_checkpoint[key] for key in sorted(targets_by_checkpoint)]
    check_regimes = Counter(check["fitter_regime"] for check in checks)
    target_regimes = Counter(
        target["patterns"][0]["realized_regime"] for target in targets
    )
    route_counts = Counter(check["transfer_partition"] for check in checks)
    validation = {
        "check_count": len(checks),
        "unique_target_count": len(targets),
        "target_pattern_record_count": sum(len(target["patterns"]) for target in targets),
        "check_pattern_link_count": len(checks) * 4,
        "check_regime_counts": dict(sorted(check_regimes.items())),
        "unique_target_regime_counts": dict(sorted(target_regimes.items())),
        "transfer_partition_counts": dict(sorted(route_counts.items())),
        "degenerate_or_single_sign_check_count": sum(
            label not in {"zero_inflated_positive", "three_sign"}
            for label in check_regimes.elements()
        ),
    }
    expected_validation = {
        "check_count": 48,
        "unique_target_count": 42,
        "target_pattern_record_count": 168,
        "check_pattern_link_count": 192,
        "check_regime_counts": {
            "three_sign": 13,
            "zero_inflated_positive": 35,
        },
        "unique_target_regime_counts": {
            "three_sign": 7,
            "zero_inflated_positive": 35,
        },
        "transfer_partition_counts": {
            "early_gap_fill": 17,
            "late_producer_complement": 31,
        },
        "degenerate_or_single_sign_check_count": 0,
    }
    if validation != expected_validation:
        raise RuntimeError(f"validation differs: {validation} != {expected_validation}")

    return {
        "schema_version": 1,
        "title": "Rare signed-tail frozen realized-regime evidence",
        "evidence_boundary": {
            "bank_schema": 1,
            "pattern_metadata": "persisted_frozen_checkpoint",
            "support_and_regime": "fit_boundary_reconstruction",
            "warning": (
                "do not recompute early-target support from the final transferred "
                "pool because later producers may mutate those fields"
            ),
        },
        "authority": {
            "adjudication_md_sha256": AUTHORITY_SHA256["ADJUDICATION.md"],
            "adjudication_json_sha256": AUTHORITY_SHA256["adjudication.json"],
            "f025_gates_sha256": AUTHORITY_SHA256["f025_gates"],
            "transferred_receipts_sha256": AUTHORITY_SHA256[
                "transferred_receipts"
            ],
            "transferred_checkpoint": observed_checkpoint,
        },
        "pattern_profiles": {
            key: list(value) for key, value in PATTERN_PROFILES.items()
        },
        "retirement_cap_support": RETIREMENT_CAP_SUPPORT,
        "keogh_one_sided_leg": {
            "structurally_absent": False,
            "native_asec_positive_rows": 2,
            "native_asec_positive_values": [2_040.0, 30_000.0],
            "old_training_cap_positive_rows": 0,
            "frozen_acs_positive_rows": 0,
            "loss_stage": "retirement_training_cap_before_transfer_fit",
        },
        "targets": targets,
        "checks": checks,
        "validation": validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication-dir", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--f025-gates", type=Path, required=True)
    parser.add_argument("--transferred-receipts", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("realized_regimes.json"),
    )
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    rendered = json.dumps(build_evidence(args), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale realized-regime evidence: {args.out}")
        print(f"realized-regime evidence current: {args.out}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote realized-regime evidence: {args.out}")


if __name__ == "__main__":
    main()
