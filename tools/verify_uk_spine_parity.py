"""Whole-spine parity: the microcosm UK spine against the frozen incumbent (#686).

This is the instrument the swap decision rests on. It diffs a candidate spine's
extracted input surface against the committed enhanced-FRS parity reference and
holds every difference to the committed signed-differences register. The rule it
enforces is the one #686 states: **anything differing that is not signed is a
defect.**

Three surfaces are compared:

* ``entity_counts`` — the record-count identity, exactly. The spine and the
  pinned incumbent are both pre-clone, so these must match to the row.
* ``nonzero_shares`` — per-column unweighted owning-entity nonzero share, at the
  reference's own 6-decimal grain, plus the column-set difference in both
  directions.
* ``weighted_totals`` — optional, and licensed. Supplied as the two register
  sidecars, compared as relative deltas only.

Two fences protect the verdict from being manufactured. The reference side is
always the committed instrument, never anything derived from the candidate; and
the tool refuses inputs that alias each other, so a candidate cannot be compared
against itself. Under ``--strict`` an unused register entry is also a failure,
so the register cannot rot into a blanket amnesty as the spine changes.

Disclosure control: output is column names, counts, shares already carried by
the committed reference, and relative deltas — never unit-record values. The
licensed weighted registers stay outside the repository.

Exit code: 0 when parity holds (with or without signed differences), 1 when an
unsigned difference is found or a strict check fails, and 2 when no verdict is
possible — an unsafe invocation or an unreadable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "microcosm-build" / "src"))  # noqa: E402

from microcosm.build.uk_runtime.parity_reference import (  # noqa: E402
    load_efrs_parity_reference,
)
from microcosm.build.uk_runtime.signed_differences import (  # noqa: E402
    UKSignedDifferenceRegister,
    load_uk_spine_swap_signed_differences,
)

#: The reference records shares rounded to six decimals, so a candidate
#: extracted by the same producer agrees to that grain or it genuinely differs.
SHARE_EPSILON = 1e-6

#: Weighted totals ride calibrated weights; a relative delta below this is
#: numerical noise rather than a divergence to sign.
TOTALS_EPSILON = 1e-9

VERDICT_PARITY = "parity"
VERDICT_SIGNED_PARITY = "signed_parity"
VERDICT_DEFECT = "defect"


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object.")
    return payload


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _candidate_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return {}
    return {
        key: source.get(key)
        for key in ("filename", "sha256", "size_bytes", "vintage", "period")
        if source.get(key) is not None
    }


def _compare_entity_counts(
    reference_stats: Mapping[str, Any],
    candidate_stats: Mapping[str, Any],
    register: UKSignedDifferenceRegister,
) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {}
    unsigned: list[str] = []
    for entity in sorted(set(reference_stats) | set(candidate_stats)):
        expected = (reference_stats.get(entity) or {}).get("records")
        observed = (candidate_stats.get(entity) or {}).get("records")
        equal = expected == observed
        entry = {"reference": expected, "candidate": observed, "equal": equal}
        if not equal:
            signed = register.matching(surface="entity_counts", column=entity)
            entry["signed_id"] = signed.id if signed else None
            if signed is None:
                unsigned.append(entity)
        report[entity] = entry
    return report, unsigned


def _compare_shares(
    reference_shares: Mapping[str, float],
    candidate_shares: Mapping[str, float],
    entities: Mapping[str, str],
    register: UKSignedDifferenceRegister,
) -> tuple[dict[str, Any], list[str]]:
    unsigned: list[str] = []
    differing: dict[str, Any] = {}
    compared = sorted(set(reference_shares) & set(candidate_shares))
    for column in compared:
        expected = float(reference_shares[column])
        observed = float(candidate_shares[column])
        if abs(observed - expected) <= SHARE_EPSILON:
            continue
        signed = register.matching(surface="nonzero_shares", column=column)
        differing[column] = {
            "entity": entities.get(column),
            "reference": expected,
            "candidate": observed,
            "delta": observed - expected,
            "signed_id": signed.id if signed else None,
        }
        if signed is None:
            unsigned.append(column)

    def _missing(names: list[str], expectation: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for column in sorted(names):
            signed = register.matching(surface="nonzero_shares", column=column)
            out[column] = {
                "entity": entities.get(column),
                "signed_id": signed.id if signed else None,
                "expectation": expectation,
            }
            if signed is None:
                unsigned.append(column)
        return out

    report = {
        "compared": len(compared),
        "differing": differing,
        "missing_in_candidate": _missing(
            list(set(reference_shares) - set(candidate_shares)),
            "column_missing_in_candidate",
        ),
        "extra_in_candidate": _missing(
            list(set(candidate_shares) - set(reference_shares)),
            "column_missing_in_reference",
        ),
    }
    return report, unsigned


def _compare_weighted_totals(
    reference_totals: Mapping[str, float],
    candidate_totals: Mapping[str, float],
    register: UKSignedDifferenceRegister,
) -> tuple[dict[str, Any], list[str]]:
    unsigned: list[str] = []
    differing: dict[str, Any] = {}
    compared = sorted(set(reference_totals) & set(candidate_totals))
    for column in compared:
        expected = float(reference_totals[column])
        observed = float(candidate_totals[column])
        if expected == 0.0 and observed == 0.0:
            continue
        if expected == 0.0:
            relative = float("inf")
        else:
            relative = (observed - expected) / expected
        if abs(relative) <= TOTALS_EPSILON:
            continue
        signed = register.matching(surface="weighted_totals", column=column)
        # Deltas only: the absolute totals are licensed and stay outside.
        differing[column] = {
            "relative_delta": relative,
            "signed_id": signed.id if signed else None,
        }
        if signed is None:
            unsigned.append(column)
    return (
        {
            "compared": len(compared),
            "differing": differing,
            "only_in_reference": sorted(set(reference_totals) - set(candidate_totals)),
            "only_in_candidate": sorted(set(candidate_totals) - set(reference_totals)),
        },
        unsigned,
    )


def verify_uk_spine_parity(
    *,
    candidate_json: Path,
    register: UKSignedDifferenceRegister,
    reference_weighted_totals: Path | None = None,
    candidate_weighted_totals: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Compare a candidate extraction against the committed reference."""

    reference = load_efrs_parity_reference()
    candidate = _load_json(candidate_json)

    candidate_identity = _candidate_identity(candidate)
    # The reference side must never be derived from the candidate: a copied
    # reference would make this pass by construction.
    if candidate_identity.get("sha256") == reference.source.sha256:
        raise ValueError(
            "the candidate extraction names the pinned incumbent's own sha256; "
            "the reference side must be independent of the candidate."
        )

    candidate_shares = candidate.get("nonzero_shares")
    if not isinstance(candidate_shares, Mapping):
        raise ValueError(f"{candidate_json}: 'nonzero_shares' must be an object.")
    candidate_stats = candidate.get("entity_stats")
    if not isinstance(candidate_stats, Mapping):
        raise ValueError(f"{candidate_json}: 'entity_stats' must be an object.")

    # EfrsParityReference exposes the share surface, not the record counts, so
    # the counts come from the same packaged resource it was loaded from —
    # read through importlib.resources so this works from an installed wheel
    # as well as the source tree.
    reference_payload = json.loads(
        files("microcosm.build.uk")
        .joinpath("efrs_parity_reference.json")
        .read_text(encoding="utf-8")
    )
    reference_stats = dict(reference_payload.get("entity_stats") or {})

    counts_report, counts_unsigned = _compare_entity_counts(
        reference_stats, candidate_stats, register
    )
    shares_report, shares_unsigned = _compare_shares(
        reference.nonzero_shares,
        {name: float(value) for name, value in candidate_shares.items()},
        reference.input_entities,
        register,
    )

    matched_ids = {
        entry["signed_id"] for entry in counts_report.values() if entry.get("signed_id")
    }
    for section in (
        shares_report["differing"],
        shares_report["missing_in_candidate"],
        shares_report["extra_in_candidate"],
    ):
        matched_ids.update(
            entry["signed_id"] for entry in section.values() if entry.get("signed_id")
        )

    report: dict[str, Any] = {
        "check": "uk_whole_spine_parity",
        "schema_version": 1,
        "reference": {
            "resource": "efrs_parity_reference.json",
            "source": {
                "filename": reference.source.filename,
                "revision": reference.source.revision,
                "sha256": reference.source.sha256,
                "vintage": reference.source.vintage,
            },
        },
        "candidate": {"extraction": str(candidate_json), **candidate_identity},
        "entity_counts": counts_report,
        "nonzero_shares": shares_report,
    }

    unsigned = list(counts_unsigned) + list(shares_unsigned)

    if reference_weighted_totals is not None and candidate_weighted_totals is not None:
        left = _load_json(reference_weighted_totals)
        right = _load_json(candidate_weighted_totals)
        left_totals = left.get("totals")
        right_totals = right.get("totals")
        if not isinstance(left_totals, Mapping) or not isinstance(
            right_totals, Mapping
        ):
            raise ValueError("weighted-totals sidecars must carry a 'totals' object.")
        totals_report, totals_unsigned = _compare_weighted_totals(
            {k: float(v) for k, v in left_totals.items()},
            {k: float(v) for k, v in right_totals.items()},
            register,
        )
        totals_report["reference_identity"] = left.get("identity")
        totals_report["candidate_identity"] = right.get("identity")
        report["weighted_totals"] = totals_report
        unsigned.extend(totals_unsigned)
        matched_ids.update(
            entry["signed_id"]
            for entry in totals_report["differing"].values()
            if entry.get("signed_id")
        )

    unused = sorted(
        difference.id
        for difference in register.differences
        if difference.id not in matched_ids
    )
    report["register"] = {
        "resource": "spine_swap_signed_differences.json",
        "entries": len(register.differences),
        "matched_ids": sorted(matched_ids),
        "unused_ids": unused,
    }
    report["unsigned_differences"] = sorted(set(unsigned))

    if report["unsigned_differences"]:
        report["verdict"] = VERDICT_DEFECT
    elif matched_ids:
        report["verdict"] = VERDICT_SIGNED_PARITY
    else:
        report["verdict"] = VERDICT_PARITY
    report["strict"] = strict
    report["strict_failure"] = bool(strict and unused)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diff a candidate UK spine's extracted input surface against the "
            "committed enhanced-FRS parity reference, holding every difference "
            "to the committed signed-differences register."
        )
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        required=True,
        help=(
            "Candidate extraction written by build_uk_efrs_parity_reference.py "
            "--candidate-h5 --emit-candidate-json."
        ),
    )
    parser.add_argument(
        "--register",
        type=Path,
        default=None,
        help=(
            "Override the committed signed-differences register (tests only; "
            "the packaged resource is the contract)."
        ),
    )
    parser.add_argument(
        "--reference-weighted-totals",
        type=Path,
        default=None,
        help="Licensed weighted-totals sidecar for the pinned incumbent.",
    )
    parser.add_argument(
        "--candidate-weighted-totals",
        type=Path,
        default=None,
        help="Licensed weighted-totals sidecar for the candidate spine.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Also fail when a register entry matched nothing. This is the "
            "swap-acceptance posture: it stops the register drifting into a "
            "blanket amnesty as the spine changes."
        ),
    )
    parser.add_argument(
        "--receipt-json",
        type=Path,
        default=None,
        help="Also write the receipt JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    totals = (args.reference_weighted_totals, args.candidate_weighted_totals)
    if any(totals) and not all(totals):
        print(
            "error: --reference-weighted-totals and --candidate-weighted-totals "
            "must be supplied together.",
            file=sys.stderr,
        )
        return 2
    if all(totals) and _paths_alias(*totals):
        print(
            "error: the two weighted-totals sidecars must be distinct.",
            file=sys.stderr,
        )
        return 2
    if args.receipt_json is not None and _paths_alias(
        args.receipt_json, args.candidate_json
    ):
        print("error: --receipt-json must not alias an input.", file=sys.stderr)
        return 2

    try:
        register = (
            load_uk_spine_swap_signed_differences(str(args.register))
            if args.register is not None
            else load_uk_spine_swap_signed_differences()
        )
        report = verify_uk_spine_parity(
            candidate_json=args.candidate_json,
            register=register,
            reference_weighted_totals=args.reference_weighted_totals,
            candidate_weighted_totals=args.candidate_weighted_totals,
            strict=args.strict,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except Exception as error:  # noqa: BLE001 - message is ours, not the data's
        print(
            f"error: parity verification could not be completed: {error}",
            file=sys.stderr,
        )
        return 2

    print(rendered)
    if args.receipt_json is not None:
        args.receipt_json.write_text(rendered + "\n", encoding="utf-8")

    if report["verdict"] == VERDICT_DEFECT:
        print(
            "DEFECT: "
            f"{len(report['unsigned_differences'])} difference(s) are not in the "
            "signed register: " + ", ".join(report["unsigned_differences"][:20]),
            file=sys.stderr,
        )
        return 1
    if report["strict_failure"]:
        print(
            "STRICT: register entries matched nothing: "
            + ", ".join(report["register"]["unused_ids"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
