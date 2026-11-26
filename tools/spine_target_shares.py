"""Report which rows carry each calibration target, from saved inputs.

For each target: the share of its weighted total that each set of rows
("spine") carries, against that spine's share of the weight among the records
the target is about. A target one spine carries almost alone pulls weight
onto that spine and lowers the effective sample size
(``microcosm.calibrate.spine_target_shares``; microcosm#1729).

The tool reads saved calibration inputs. It runs no engine, solves nothing
and writes only its report, so it needs no build.

Two kinds of input:

``--us-acs-local-checkpoint DIR``
    The checkpoint directory of ``tools/build_us_acs_local_release.py``
    (``--stage materialize`` wrote it). Reads the target matrix, the target
    roles and the lean frame. Reports at the design weights, and at the
    calibrated weights when ``weights_latest.npz`` is there. A target's scope
    is its own state, district or county.

``--matrix M.npz --weights W --spines S``
    Any pooled build: a SciPy sparse matrix saved with ``scipy.sparse.
    save_npz`` (targets x records), a weight vector and each record's spine
    label (``.npy``, or ``.npz`` with the key ``--weights-key`` /
    ``--spines-key``). ``--names N.json`` names the targets.

Examples:

    uv run python tools/spine_target_shares.py \\
        --us-acs-local-checkpoint <run>/checkpoints --out shares.json

    uv run python tools/spine_target_shares.py \\
        --matrix problem.npz --weights weights.npy --spines spines.npy

``--spine NAME`` prints the targets that spine over-carries most. ``--top``
sets how many. ``--minimum-carriers N`` withholds a target that a spine
carries with fewer than ``N`` distinct carriers: pass the disclosure floor
for a build on licensed microdata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: The household columns of the ACS local lean frame.
US_SPINE_COLUMN = "household_spine"
US_SOURCE_COLUMN = "household_source_id"
#: Scope kind -> (household column, role key, digits of the code).
US_SCOPE_COLUMNS = {
    "state": ("state_fips", "state_fips", 2),
    "congressional_district": (
        "congressional_district_geoid",
        "congressional_district_geoid",
        4,
    ),
    "county": ("county_fips", "county_fips", 5),
}


def _padded(values, digits: int) -> np.ndarray:
    """Geography codes as zero-padded strings (``6`` and ``"06"`` are one)."""

    import pandas as pd

    numbers = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce")
    return np.asarray(
        [
            "" if np.isnan(number) else f"{int(number):0{digits}d}"
            for number in numbers.to_numpy(dtype=np.float64)
        ],
        dtype=object,
    )


def us_acs_local_inputs(households, roles) -> dict:
    """The spine, source, scope and group inputs of an ACS local checkpoint.

    ``households`` is the lean frame's household table and ``roles`` the
    ``target_roles.json`` records, row-aligned with the target matrix. A
    target's scope is the state, district or county its role names; a role
    that names none of them is scoped to every record.
    """

    if US_SPINE_COLUMN not in households.columns:
        raise SystemExit(
            f"The lean frame has no {US_SPINE_COLUMN} column, so its rows "
            "cannot be told apart by spine."
        )
    # A county run records the household column its county rows bind.
    county_column = next(
        (role["county_column"] for role in roles if role.get("county_column")),
        US_SCOPE_COLUMNS["county"][0],
    )
    scope_codes = {}
    for kind, (column, _key, digits) in US_SCOPE_COLUMNS.items():
        column = county_column if kind == "county" else column
        if column in households.columns:
            scope_codes[kind] = _padded(households[column].to_numpy(), digits)
    scopes = []
    for role in roles:
        kind = role.get("geography_level")
        scope = None
        if kind in scope_codes:
            _column, key, digits = US_SCOPE_COLUMNS[kind]
            code = role.get(key)
            if code not in (None, ""):
                scope = (kind, str(_padded([code], digits)[0]))
        scopes.append(scope)
    spine = households[US_SPINE_COLUMN].astype(str).to_numpy()
    return {
        "spines": spine,
        "source_ids": (
            households[US_SOURCE_COLUMN].to_numpy()
            if US_SOURCE_COLUMN in households.columns
            else None
        ),
        "scope_codes": scope_codes,
        "target_scopes": scopes,
        "names": [role["name"] for role in roles],
        "groups": [
            f"{role.get('family')}/{role.get('geography_level')}" for role in roles
        ],
        "roles": [role.get("role") for role in roles],
    }


def _load_us_acs_local(checkpoint_dir: Path) -> tuple:
    """``(matrix, inputs, weights by label)`` of an ACS local checkpoint."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_us_acs_local_release as release

    frame, design_weights, _registry, roles, matrix = release.load_checkpoint_surface(
        checkpoint_dir
    )
    inputs = us_acs_local_inputs(frame.table("household"), roles)
    weights = {"design": np.asarray(design_weights, dtype=np.float64)}
    saved = checkpoint_dir / "weights_latest.npz"
    if saved.exists():
        calibrated = np.load(saved)["weights"]
        if len(calibrated) == matrix.shape[1]:
            weights["calibrated"] = np.asarray(calibrated, dtype=np.float64)
    return matrix, inputs, weights


def _load_vector(path: Path, key: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    if key not in loaded:
        raise SystemExit(f"{path} has no array {key!r} (it has {sorted(loaded)}).")
    return loaded[key]


def _load_generic(args) -> tuple:
    from scipy import sparse

    matrix = sparse.load_npz(args.matrix)
    inputs = {
        "spines": _load_vector(args.spines, args.spines_key).astype(str),
        "source_ids": None,
        "scope_codes": None,
        "target_scopes": None,
        "names": (
            None if args.names is None else json.loads(Path(args.names).read_text())
        ),
        "groups": None,
        "roles": None,
    }
    weights = {"weights": _load_vector(args.weights, args.weights_key)}
    return matrix, inputs, weights


def build_report(matrix, inputs, weights, *, minimum_carriers=None) -> dict:
    """The report for each weight vector, with group summaries when known."""

    from microcosm.calibrate.spine_target_shares import spine_target_shares

    report: dict[str, object] = {}
    for label, vector in weights.items():
        shares = spine_target_shares(
            matrix,
            vector,
            inputs["spines"],
            names=inputs["names"],
            target_scopes=inputs["target_scopes"],
            scope_codes=inputs["scope_codes"],
            source_ids=inputs["source_ids"],
            minimum_carriers=minimum_carriers,
        )
        entry = shares.report()
        if inputs["groups"] is not None:
            entry["by_family_and_level"] = shares.summary(inputs["groups"])
        if inputs["roles"] is not None:
            for row, role in zip(entry["targets"], inputs["roles"], strict=True):
                row["role"] = role
        report[label] = entry
    return report


def _print_ranking(report: dict, spine: str, top: int) -> None:
    for label, entry in report.items():
        if spine not in entry["spines"]:
            raise SystemExit(
                f"--spine must be one of {entry['spines']}, got {spine!r}."
            )
        rows = [
            row
            for row in entry["targets"]
            if "share" in row and row["excess"][spine] is not None
        ]
        rows.sort(key=lambda row: -row["excess"][spine])
        mass = entry["overall_mass_share"][spine]
        print(f"[{label}] {spine} holds {mass:.1%} of the weight; it over-carries:")
        for row in rows[:top]:
            print(
                f"  {row['share'][spine]:7.1%} of the target against "
                f"{row['mass_share'][spine]:6.1%} of the mass  {row['name']}"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--us-acs-local-checkpoint", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-key", default="weights")
    parser.add_argument("--spines", type=Path)
    parser.add_argument("--spines-key", default="spines")
    parser.add_argument("--names", type=Path)
    parser.add_argument("--minimum-carriers", type=int, default=None)
    parser.add_argument("--spine", help="Print the targets this spine over-carries.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args(argv)
    generic = (args.matrix, args.weights, args.spines)
    if args.us_acs_local_checkpoint is not None:
        if any(value is not None for value in generic):
            parser.error(
                "--us-acs-local-checkpoint replaces --matrix/--weights/--spines."
            )
    elif any(value is None for value in generic):
        parser.error(
            "pass --us-acs-local-checkpoint, or all of --matrix, --weights and --spines."
        )
    if args.minimum_carriers is not None and args.minimum_carriers < 1:
        parser.error("--minimum-carriers must be at least 1.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.us_acs_local_checkpoint is not None:
        matrix, inputs, weights = _load_us_acs_local(args.us_acs_local_checkpoint)
    else:
        matrix, inputs, weights = _load_generic(args)
    report = build_report(
        matrix, inputs, weights, minimum_carriers=args.minimum_carriers
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"wrote {args.out}")
    for label, entry in report.items():
        shares = ", ".join(
            f"{spine} {share:.1%}"
            for spine, share in entry["overall_mass_share"].items()
        )
        print(
            f"[{label}] {entry['n_targets']} targets; weight by spine: {shares}"
            + (f"; {entry['n_withheld']} withheld" if entry["n_withheld"] else "")
        )
    if args.spine is not None:
        _print_ranking(report, args.spine, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
