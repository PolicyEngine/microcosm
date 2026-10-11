"""Person-level joint retirement-contribution donors from the full 2022 SCF.

The SCF observes new IRA/Keogh contributions in 2021 (X6791--X6794,
excluding rollovers), jointly with employee contributions to current main-job
plans in 2022. This period mismatch is retained explicitly; account ownership
alone never establishes contribution participation. SIPP 2023's IRA
contribution amount covers only accounts through the main employer/business,
so it cannot supply the same personal-IRA contribution observation.

The SCF pools the annual contribution amount across IRA/Keogh types. Mixed
IRA/Keogh owners are excluded because that amount does not establish which
family received contributions. Within an IRA-only pool, account-balance shares
are a declared traditional/Roth proxy; single-type owners retain the whole
observed amount even with a zero balance. Pure Keogh owners supply SE-pension
contributions. Workplace
traditional/Roth amounts retain the existing 85/15 proxy because the public
SCF does not distinguish the tax treatment of employee contributions.

The public workplace type 2 combines 401(k), 403(b), thrift and 457 plans.
Employee-funded account-type plans are carried through the existing 401(k)
leaves; self-employed main-job contributions and Keogh contributions use the
SE-pension leaf. Defined-benefit plans and employer allocations are excluded
(the separately deferred option A2). Only two workplace plans are detailed;
persons with additional plans or unusable source tuples are excluded rather
than silently classifying an unobserved amount as zero.

Primary source and dictionary:
https://www.federalreserve.gov/econres/scf_2022.htm
https://www.federalreserve.gov/econres/files/codebk2022.txt
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.joint_retirement import RETIREMENT_COLUMNS

SCF_RETIREMENT_URL = "https://www.federalreserve.gov/econres/files/scf2022s.zip"
SCF_RETIREMENT_MEMBER = "p22i6.dta"
SCF_RETIREMENT_ZIP_SHA256 = (
    "409e6811df895766d50b2f597c10b1b3c5813e7d3e0e45d910ad26c0cb07f4eb"
)
SCF_RETIREMENT_MEMBER_SHA256 = (
    "61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a"
)
SCF_RETIREMENT_OUTPUT_COLUMNS = RETIREMENT_COLUMNS
_PERSON_FIELDS = (
    # Age, sex, main-job status, hours, weeks, detailed plan count, IRA flag,
    # IRA contribution, ownership flags, and corresponding account balances.
    (
        "x8022",
        "x8021",
        "x4106",
        "x4110",
        "x4111",
        "x4201",
        "x6791",
        "x6792",
        "x6444",
        "x6446",
        "x6447",
        "x3605",
        "x6551",
        "x6552",
        "x6553",
        "x6554",
    ),
    (
        "x104",
        "x103",
        "x4706",
        "x4710",
        "x4711",
        "x4801",
        "x6793",
        "x6794",
        "x6448",
        "x6450",
        "x6451",
        "x3615",
        "x6559",
        "x6560",
        "x6561",
        "x6562",
    ),
)
_PLAN_PREFIXES = (("x110", "x111"), ("x113", "x114"))
_PLAN_SUFFIXES = ("01", "40", "42", "43", "45", "46")
SCF_RETIREMENT_SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        ("y1", "yy1", "x42001")
        + tuple(field for person in _PERSON_FIELDS for field in person)
        + tuple(
            prefix + suffix
            for prefixes in _PLAN_PREFIXES
            for prefix in prefixes
            for suffix in _PLAN_SUFFIXES
        )
    )
)
_DC_PLAN_TYPES = {2, 5, 6, 21, 22, 25}
_DB_PLAN_TYPES = {1, 7, 8, 30}


def fetch_scf_retirement_extract(cache_dir: str | Path) -> Path:
    """Fetch fixed public bytes, verifying both archive and extracted member."""

    root = Path(cache_dir).expanduser()
    target = root / SCF_RETIREMENT_MEMBER
    if target.exists():
        if (
            hashlib.sha256(target.read_bytes()).hexdigest()
            != SCF_RETIREMENT_MEMBER_SHA256
        ):
            raise ValueError("Cached SCF retirement member SHA-256 mismatch.")
        return target
    root.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SCF_RETIREMENT_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SCF_RETIREMENT_ZIP_SHA256:
        raise ValueError("SCF retirement archive SHA-256 mismatch.")
    member = zipfile.ZipFile(io.BytesIO(archive)).read(SCF_RETIREMENT_MEMBER)
    if hashlib.sha256(member).hexdigest() != SCF_RETIREMENT_MEMBER_SHA256:
        raise ValueError("SCF retirement member SHA-256 mismatch.")
    target.write_bytes(member)
    return target


def _annual_amount(
    amount: float, frequency: float, hours: float, weeks: float
) -> float:
    """Annualize observed payments; unidentified frequency stays unidentified."""

    if not np.isfinite(amount) or amount < 0:
        return np.nan
    fixed_multipliers = {4: 12.0, 5: 4.0, 6: 1.0, 8: 1.0, 11: 2.0, 12: 6.0, 31: 24.0}
    if frequency in fixed_multipliers:
        return amount * fixed_multipliers[frequency]
    if frequency in {2, 3, 18} and np.isfinite(weeks) and 0 < weeks <= 52:
        if frequency == 2:
            return amount * weeks
        if frequency == 3:
            return amount * weeks / 2
        if np.isfinite(hours) and 0 < hours <= 168:
            return amount * hours * weeks
    return np.nan


def _plan_amount(row: pd.Series, prefix: str, hours: float, weeks: float) -> float:
    kind, status = row[prefix + "01"], row[prefix + "40"]
    if kind in _DB_PLAN_TYPES or kind == 0:
        return 0.0
    if kind not in _DC_PLAN_TYPES or status not in {0, 1, 3, 5}:
        return np.nan
    if status in {0, 5}:
        return 0.0
    current = row[prefix + "42"]
    if status == 1 and current > 0:
        return _annual_amount(current, row[prefix + "43"], hours, weeks)
    previous = row[prefix + "45"]
    if previous > 0:
        return _annual_amount(previous, row[prefix + "46"], hours, weeks)
    return np.nan


def _ira_components(row: pd.Series, fields: tuple[str, ...]) -> np.ndarray:
    flag, amount = row[fields[6]], row[fields[7]]
    if flag in {0, 5}:
        return np.zeros(3)
    if flag != 1 or not np.isfinite(amount) or amount <= 0:
        return np.full(3, np.nan)
    flags = row[list(fields[8:12])].to_numpy(dtype=np.float64)
    if not np.isin(flags, [0, 1, 5]).all():
        return np.full(3, np.nan)
    owned = np.array([flags[0] == 1, flags[1] == 1 or flags[2] == 1, flags[3] == 1])
    if owned[2] and owned[:2].any():
        return np.full(3, np.nan)
    balances = row[list(fields[12:16])].to_numpy(dtype=np.float64)
    if not np.isfinite(balances).all() or (balances < -1).any():
        return np.full(3, np.nan)
    balances = np.maximum(balances, 0)
    masses = np.array([balances[0], balances[1] + balances[2], balances[3]]) * owned
    if owned.sum() == 1:
        shares = owned.astype(np.float64)
    elif masses.sum() > 0:
        shares = masses / masses.sum()
    else:
        return np.full(3, np.nan)
    # Return traditional IRA, Roth IRA, Keogh in leaf order.
    return amount * shares[[1, 0, 2]]


def extract_scf_retirement_donor(
    raw: pd.DataFrame, *, roth_dc_share: float = 0.15
) -> pd.DataFrame:
    """Extract whole person contribution vectors, never separately draw leaves.

    All five public implicates are retained, each with X42001 / 5 weight.
    Positive donor vectors only are returned, for conditioning on CPS savers.
    Exclusion counts and the period/split limitations travel in ``attrs``.
    """

    if not np.isfinite(roth_dc_share) or not 0 <= roth_dc_share <= 1:
        raise ValueError("roth_dc_share must be between zero and one.")
    raw = raw.rename(columns=str.lower)
    missing = sorted(set(SCF_RETIREMENT_SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"SCF retirement source is missing columns: {missing}.")
    if raw[["y1", "yy1"]].duplicated().any():
        raise ValueError("SCF family/implicate keys must be unique.")
    records = []
    excluded = 0
    for _, row in raw.iterrows():
        for person, fields in enumerate(_PERSON_FIELDS):
            age, sex, job, hours, weeks, plans = (row[field] for field in fields[:6])
            if person == 1 and age == 0:
                continue
            if (
                not np.isfinite(age)
                or age <= 0
                or sex not in {1, 2}
                or job not in {0, 1, 2}
                or not np.isfinite(plans)
                or plans > 2
                or plans < 0
                or not np.isfinite(row["x42001"])
                or row["x42001"] <= 0
            ):
                excluded += 1
                continue
            ira = _ira_components(row, fields)
            dc = sum(
                _plan_amount(row, prefix, hours, weeks)
                for prefix in _PLAN_PREFIXES[person]
            )
            if not np.isfinite(ira).all() or not np.isfinite(dc):
                excluded += 1
                continue
            se = ira[2] + (dc if job == 2 else 0.0)
            employer = dc if job != 2 else 0.0
            vector = np.array(
                [
                    employer * (1 - roth_dc_share),
                    employer * roth_dc_share,
                    ira[0],
                    ira[1],
                    se,
                ]
            )
            if vector.sum() <= 0:
                continue
            records.append(
                dict(
                    donor_id=f"{int(row['y1'])}:{int(row['yy1'])}:{person}",
                    age=float(age),
                    is_female=sex == 2,
                    has_wages=job == 1,
                    has_self_employment=job == 2,
                    scf_weight=float(row["x42001"]) / 5,
                    total_contributions=float(vector.sum()),
                    rank_value=float(vector.sum()),
                    **dict(zip(SCF_RETIREMENT_OUTPUT_COLUMNS, vector, strict=True)),
                )
            )
    donor = pd.DataFrame.from_records(records)
    if donor.empty:
        raise ValueError(
            "SCF retirement donor has no usable positive contribution tuples."
        )
    donor.attrs["scf_retirement"] = {
        "survey_year": 2022,
        "ira_contribution_year": 2021,
        "workplace_contribution_period": "current 2022; previous-year amount when current varies or stopped",
        "ira_tax_type_method": "observed owner account-balance shares; single-type ownership keeps full contribution",
        "ira_keogh_mixed_owners": "excluded: pooled annual amount does not identify contributing family",
        "roth_dc_share": roth_dc_share,
        "excluded_person_implicates": excluded,
        "source_person_implicates": len(raw) * 2,
    }
    return donor


def load_scf_retirement_donor(
    source_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    roth_dc_share: float = 0.15,
) -> pd.DataFrame:
    """Read the pinned public file or an explicitly supplied local SCF artifact."""

    if source_path is None:
        if cache_dir is None:
            raise ValueError(
                "An explicit writable SCF retirement cache_dir is required."
            )
        source_path = fetch_scf_retirement_extract(cache_dir)
    raw = pd.read_stata(
        source_path,
        columns=list(SCF_RETIREMENT_SOURCE_COLUMNS),
        convert_categoricals=False,
    )
    return extract_scf_retirement_donor(raw, roth_dc_share=roth_dc_share)
