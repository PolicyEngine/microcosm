"""Project each humanitarian draw's stacked ACS supply against its residual.

Usage:
  python -I stacked_supply.py RECEIPT.json --acs-dir <dir> --archive-dir <dir>
      [--out OUT.json]

A stacked (multispine) pool joins the ASEC pool and the ACS 2024 1-year, each
arm carrying half the ASEC pool's household mass
(``stacked_spine.DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES``, anchored on the ASEC
channel by ``spine_assembly.assemble_spines``). The immigration stage labels
the ASEC rows at the full pool's person mass (``multispine_pool``'s
``person_weight_scale``); ``reconcile_us_immigration_humanitarian_transfer``
then asks the ACS rows for each draw's residual: its target less the ASEC
arm's labelled mass.

This script does not build a stacked pool. It projects one from aggregates:

- the ASEC arm's labelled mass from RECEIPT.json's
  ``humanitarian_draw_achieved`` (a ``stage_receipt_pooled.py`` run on the
  default pool), scaled to the stacked weights;
- the ACS arm's supply from the ACS 2024 1-year PUMS (``csv_hus.zip`` and
  ``csv_pus.zip`` in ``--acs-dir``, checked against the SHA-256 values in
  ``us_runtime/acs_2024_1yr_sources.json``), weighted as ``acs_pums`` weights the
  frame (WGTP of each occupied unit, PWGTP for a group-quarters person), with
  each draw's candidates chosen by the kernel's own
  ``_humanitarian_draw_candidates``.

Every non-citizen counts as SSN-eligible, and no draw removes another's
candidates, so each supply is an upper bound. Refugee, asylee and
withholding draws take only the indicator-documented code the transfer
assigns, so their bounds are loose; parole and TPS draws take every
non-citizen code, so theirs are tight up to the order of the draws.

Two lower bounds bracket each draw's share of its target:

- ``asec_arm_share_of_target``: the ASEC arm's labelled mass alone. The
  reconciliation never relabels those rows, so every draw reaches at least
  this share.
- ``worst_case_share_of_target``: the ASEC arm plus the ACS supply left after
  every earlier draw in manifest order takes as much of the draw's candidates
  as it can, up to its own residual (a draw can pass its residual by the
  weight of the one tie group that crosses it, which this bound leaves out;
  ACS draws key on single persons). The bound holds for parole and TPS
  draws, whose SSN rule this script reads exactly. A refugee, asylee or
  withholding draw takes only rows the transfer imputes as
  ``OTHER_NON_CITIZEN``, which PUMS cannot show, so its worst case is its
  ASEC arm share.

The output holds aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import immigration

# The archives and SHA-256 values of stage_receipt_pooled.py.
ARCHIVES = (
    (
        2023,
        "asecpub24csv.zip",
        "pppub24.csv",
        "hhpub24.csv",
        "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
    ),
    (
        2024,
        "asecpub25csv.zip",
        "pppub25.csv",
        "hhpub25.csv",
        "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
    ),
    (
        2025,
        "asecpub26csv.zip",
        "pppub26.csv",
        "hhpub26.csv",
        "fe819d0d2fc4470c282e76c247b8aa8b6054811619730307e9f2fe6186707d93",
    ),
)
ANCHOR_INCOME_YEAR = 2024
TIME_PERIOD = 2024
ARM_SHARE = 0.5

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("receipt", type=Path)
parser.add_argument("--acs-dir", type=Path, required=True)
parser.add_argument(
    "--archive-dir",
    type=Path,
    action="append",
    required=True,
    help="directory holding ASEC zips (repeatable; each zip is found by name)",
)
parser.add_argument("--out", type=Path)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_path(name: str) -> Path:
    found = [d / name for d in args.archive_dir if (d / name).is_file()]
    if not found:
        raise SystemExit(f"{name} is in none of {args.archive_dir}")
    return found[0]


def _read_pums(path: Path, token: str, usecols: list[str]) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            low = name.lower()
            if low.endswith(".csv") and token in low:
                with archive.open(name) as handle:
                    parts.append(
                        pd.read_csv(handle, usecols=usecols, dtype={"SERIALNO": str})
                    )
    return pd.concat(parts, ignore_index=True)


# The ASEC pool's household and person masses, rescaled as the pool rescales.
asec_files = []
for income_year, archive_name, member, household_member, pinned in ARCHIVES:
    path = _archive_path(archive_name)
    digest = _sha256(path)
    if digest != pinned:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the pinned {pinned}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        with archive.open(next(n for n in names if n.lower().endswith(member))) as f:
            persons = pd.read_csv(f, usecols=["PH_SEQ"])
        with archive.open(
            next(n for n in names if n.lower().endswith(household_member))
        ) as f:
            households = pd.read_csv(f, usecols=["H_SEQ", "HSUP_WGT"])
    weight = households.drop_duplicates("H_SEQ").set_index("H_SEQ")["HSUP_WGT"] * 0.01
    person_mass = float(persons["PH_SEQ"].map(weight).sum())
    household_mass = float(weight.loc[persons["PH_SEQ"].unique()].sum())
    asec_files.append(
        {
            "income_year": income_year,
            "archive": archive_name,
            "sha256": digest,
            "person_mass": person_mass,
            "household_mass": household_mass,
        }
    )
anchor_person_mass = next(
    f["person_mass"] for f in asec_files if f["income_year"] == ANCHOR_INCOME_YEAR
)
pool_person_mass = 0.0
pool_household_mass = 0.0
for entry in asec_files:
    scale = anchor_person_mass / len(ARCHIVES) / entry["person_mass"]
    pool_person_mass += entry["person_mass"] * scale
    pool_household_mass += entry["household_mass"] * scale

# The ACS frame's weights: WGTP per occupied unit, PWGTP for a GQ person.
hus_path = args.acs_dir / "csv_hus.zip"
pus_path = args.acs_dir / "csv_pus.zip"
# The archives must be the ones the stacked pool pins.
acs_pins = json.loads(
    (Path(immigration.__file__).parent / "acs_2024_1yr_sources.json").read_text()
)
acs_digests = {}
for artifact in acs_pins["artifacts"]:
    path = args.acs_dir / artifact["filename"]
    digest = _sha256(path)
    if digest != artifact["sha256"]:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the pinned one")
    acs_digests[artifact["filename"]] = digest
hus = _read_pums(hus_path, "hus", ["SERIALNO", "WGTP"])
pus = _read_pums(pus_path, "pus", ["SERIALNO", "CIT", "POBP", "YOEP", "AGEP", "PWGTP"])
unit_weight = pd.Series(hus["WGTP"].to_numpy(dtype=np.float64), index=hus["SERIALNO"])
person_unit_weight = pus["SERIALNO"].map(unit_weight).to_numpy(dtype=np.float64)
gq = person_unit_weight == 0
if pus.loc[gq, "SERIALNO"].duplicated().any():
    raise SystemExit("an ACS WGTP=0 unit holds more than one person")
acs_weight = np.where(gq, pus["PWGTP"].to_numpy(dtype=np.float64), person_unit_weight)
occupied = pus["SERIALNO"].drop_duplicates()
acs_household_mass = float(unit_weight.loc[occupied].clip(lower=0).sum()) + float(
    acs_weight[gq].sum()
)
acs_person_mass = float(acs_weight.sum())

kappa = ARM_SHARE * pool_household_mass / acs_household_mass
asec_arm_person_mass = ARM_SHARE * pool_person_mass
acs_arm_person_mass = kappa * acs_person_mass
full_person_mass = asec_arm_person_mass + acs_arm_person_mass
# The stage labels ASEC rows at the full person mass; stacked ASEC rows carry
# asec_arm_person_mass of it. Candidate masses in the stage scale by rho.
rho = full_person_mass / pool_person_mass
to_stacked = asec_arm_person_mass / full_person_mass

person = pd.DataFrame(
    {
        "CIT": pus["CIT"].to_numpy(dtype=np.float64),
        "POBP": pus["POBP"].to_numpy(dtype=np.float64),
        "YOEP": pus["YOEP"].to_numpy(dtype=np.float64),
        "age": pus["AGEP"].to_numpy(dtype=np.float64),
    }
)
profile = immigration._source_aware_immigration_profile(person, time_period=TIME_PERIOD)
ssn_codes = np.where(profile.is_citizen, 1, 3)

receipt = json.loads(args.receipt.read_text())
achieved = receipt["humanitarian_draw_achieved"]
draws = {}
# Each earlier draw's candidates and residual, in manifest order.
earlier: list[tuple[np.ndarray, float]] = []
for draw in immigration.us_immigration_controls().humanitarian:
    target = float(draw.target)
    stage = float(achieved[draw.label]["population"])
    stage_exhausted = stage < target * (1 - 1e-9)
    stage_mass = min(target, rho * stage) if stage_exhausted else min(target, stage)
    immutable = stage_mass * to_stacked
    residual = max(0.0, target - immutable)
    candidates = immigration._humanitarian_draw_candidates(
        draw, ssn_codes=ssn_codes, profile=profile
    )
    acs_candidates = float(acs_weight[candidates].sum())
    supply = kappa * acs_candidates
    # An earlier draw takes at most its residual, and at most its overlap with
    # this draw's candidates.
    removed = sum(
        min(prior_residual, kappa * float(acs_weight[candidates & prior].sum()))
        for prior, prior_residual in earlier
    )
    earlier.append((candidates, residual))
    worst_supply = max(0.0, supply - removed)
    exact_ssn_rule = draw.category in {"paroled_one_year", "tps"}
    worst_reach = immutable + (min(worst_supply, residual) if exact_ssn_rule else 0.0)
    draws[draw.label] = {
        "target": target,
        "asec_stage_population": stage,
        "asec_arm_population": immutable,
        "acs_residual": residual,
        "acs_candidates_at_acs_weights": acs_candidates,
        "acs_arm_supply_bound": supply,
        "earlier_draws_worst_case_removal": removed,
        "shortfall_at_bound": max(0.0, residual - supply),
        "projected_share_of_target": (
            (immutable + min(supply, residual)) / target if target > 0 else None
        ),
        "asec_arm_share_of_target": immutable / target if target > 0 else None,
        "worst_case_share_of_target": worst_reach / target if target > 0 else None,
    }
    print(
        f"{draw.label:32s} target {target:9,.0f}  ASEC arm {immutable:9,.0f}  "
        f"residual {residual:9,.0f}  ACS bound {supply:9,.0f}  "
        f"short {max(0.0, residual - supply):9,.0f}  "
        f"worst-case share {worst_reach / target if target > 0 else float('nan'):.3f}"
    )
positive = [entry for entry in draws.values() if entry["target"] > 0]
lowest_worst_case = min(entry["worst_case_share_of_target"] for entry in positive)
print(f"lowest worst-case share of target: {lowest_worst_case:.4f}")

# OAW before this change took exact-year Afghan arrivals from 2021 on.
afghan = (
    (~profile.is_citizen)
    & (profile.birth_country == 200)
    & profile.exact_arrival
    & (profile.arrival_year >= 2021)
)
oaw_years = {
    "2021-2022": float(acs_weight[afghan & (profile.arrival_year <= 2022)].sum()),
    "2021-2024": float(acs_weight[afghan].sum()),
}
print(
    f"kappa {kappa:.4f}; ACS Afghan non-citizen arrivals at ACS weights: "
    + ", ".join(f"{k} {v:,.0f}" for k, v in oaw_years.items())
)

if args.out is not None:
    out = {
        "receipt_file": args.receipt.name,
        "receipt_sha256": _sha256(args.receipt),
        "script_sha256": _sha256(Path(__file__)),
        "acs": {
            **acs_digests,
            "household_mass": acs_household_mass,
            "person_mass": acs_person_mass,
        },
        "asec_archives": asec_files,
        "asec_pool_household_mass": pool_household_mass,
        "asec_pool_person_mass": pool_person_mass,
        "arm_household_share": ARM_SHARE,
        "acs_arm_weight_scale": kappa,
        "asec_arm_person_mass": asec_arm_person_mass,
        "acs_arm_person_mass": acs_arm_person_mass,
        "afghan_noncitizen_arrivals_at_acs_weights": oaw_years,
        "composition_band_floor": immigration._HUMANITARIAN_TARGET_RELATIVE_BAND[0],
        "lowest_worst_case_share_of_target": lowest_worst_case,
        "draws": draws,
    }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
