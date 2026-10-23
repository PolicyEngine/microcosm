"""SCF 2025 federal student loan stage tests (microcosm#1073).

Federal education debt reads the 2025 SCF: per-loan federal flags, whose
education each loan paid for, the year borrowed and the program financed. The
stage writes four person inputs that policyengine-us#9721 and #9724 add.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.scf_education_debt as education_debt
from microcosm.build.us_runtime.scf_education_debt import (
    SCF_2025_FULL_EXTRACT,
    SCF_2025_SUMMARY_EXTRACT,
    SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
    SCF_EDUCATION_DEBT_COMPOSITION_TARGETS,
    SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS,
    SCF_EDUCATION_DEBT_PREDICTORS,
    SCF_EDUCATION_DEBT_TOTAL_TARGET,
    SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
    SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
    SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
    SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
    SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN,
    SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
    SCF_EDUCATION_LOAN_YEAR_COLUMNS,
    SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
    SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
    US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS,
    ScfPublicFile,
    derive_scf_education_debt_targets,
    draw_scf_education_debt_households,
    education_debt_fit_table,
    fetch_scf_public_file,
    impute_us_scf_education_debt,
    load_scf_2025_education_debt_donor,
    scf_education_debt_composition_predictors,
    scf_education_debt_donor_fit_table,
    us_scf_education_debt_fit_table,
    us_scf_education_debt_signal_gate,
    us_scf_education_debt_stage_spec,
    us_scf_education_debt_summary,
    with_us_scf_education_debt_inputs,
)
from microcosm.build.us_runtime.scf_wealth import SCF_WEALTH_PREDICTORS
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_DONOR_WEIGHT_COLUMN = "scf_weight"
_LOAN_ITEMS = {
    "balance": SCF_EDUCATION_LOAN_BALANCE_COLUMNS,
    "federal": SCF_EDUCATION_LOAN_FEDERAL_COLUMNS,
    "whose": SCF_EDUCATION_LOAN_WHOSE_COLUMNS,
    "year": SCF_EDUCATION_LOAN_YEAR_COLUMNS,
    "program": SCF_EDUCATION_LOAN_PROGRAM_COLUMNS,
    "borrowed": SCF_EDUCATION_LOAN_AMOUNT_BORROWED_COLUMNS,
}


def _family(
    *loans: dict[str, float],
    remaining: float = 0.0,
    reversed_codes: bool = False,
    listed: tuple[tuple[int, int], ...] = (),
) -> dict[str, float]:
    """One full-file row: up to six loans given as item dictionaries.

    ``listed`` gives household-listing persons 3-12 as (relationship, public
    age code) pairs; ``reversed_codes`` sets X8000 = 1.
    """

    row = {column: 0.0 for columns in _LOAN_ITEMS.values() for column in columns}
    row[SCF_EDUCATION_LOAN_REMAINING_BALANCE_COLUMN] = remaining
    row["x8000"] = 1.0 if reversed_codes else 5.0
    for column in (
        *SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS,
        *SCF_HOUSEHOLD_LISTING_AGE_COLUMNS,
    ):
        row[column] = 0.0
    for index, (relationship, age_code) in enumerate(listed):
        row[SCF_HOUSEHOLD_LISTING_RELATIONSHIP_COLUMNS[index]] = float(relationship)
        row[SCF_HOUSEHOLD_LISTING_AGE_COLUMNS[index]] = float(age_code)
    for index, loan in enumerate(loans):
        for item, value in loan.items():
            row[_LOAN_ITEMS[item][index]] = value
    return row


def _loan(
    balance: float,
    *,
    federal: int = 1,
    whose: int = 1,
    year: int = 2018,
    program: int = 12,
    borrowed: float | None = None,
) -> dict[str, float]:
    return {
        "balance": balance,
        "federal": federal,
        "whose": whose,
        "year": year,
        "program": program,
        "borrowed": balance if borrowed is None else borrowed,
    }


def _derive(*families: dict[str, float]) -> pd.DataFrame:
    return derive_scf_education_debt_targets(pd.DataFrame.from_records(families))


# --------------------------------------------------------------------------- #
# Donor targets                                                                 #
# --------------------------------------------------------------------------- #
def test_only_federal_loans_count_and_holders_follow_whose_education() -> None:
    row = _derive(
        _family(
            _loan(10_000, whose=1, year=2016, program=12),
            _loan(20_000, whose=2, year=2010, program=13, borrowed=25_000),
            _loan(5_000, whose=3, year=2019, program=12),
            _loan(7_000, federal=5, whose=1),
            _loan(3_000, federal=0, whose=1),
        )
    ).iloc[0]

    assert row["head_federal_student_loan_balance"] == 10_000
    assert row["spouse_federal_student_loan_balance"] == 20_000
    assert row["federal_parent_plus_loan_balance"] == 5_000
    assert row[SCF_EDUCATION_DEBT_TOTAL_TARGET] == 35_000
    assert row["head_own_share"] == pytest.approx(10_000 / 35_000)
    assert row["spouse_own_share"] == pytest.approx(20_000 / 35_000)
    # The head holds the 2016 own loan and the 2019 Parent PLUS loan.
    assert row["head_is_ibr_new_borrower"] == 1.0
    assert row["spouse_is_ibr_new_borrower"] == 0.0
    assert row["head_graduate_share"] == 0.0
    assert row["spouse_graduate_share"] == 1.0
    assert row["education_debt"] == 45_000
    assert row["federal_status_unrecorded_balance"] == 3_000


def test_remaining_loans_take_the_family_federal_share_and_holder_split() -> None:
    row = _derive(
        _family(
            _loan(6_000, whose=1),
            _loan(4_000, federal=5, whose=1),
            _loan(2_000, whose=3),
            remaining=6_000,
        )
    ).iloc[0]

    # Loans 1-6 total 12,000, so each is scaled by 1 + 6,000 / 12,000.
    assert row["head_federal_student_loan_balance"] == pytest.approx(9_000)
    assert row["federal_parent_plus_loan_balance"] == pytest.approx(3_000)
    assert row[SCF_EDUCATION_DEBT_TOTAL_TARGET] == pytest.approx(12_000)
    assert row["education_debt"] == pytest.approx(18_000)
    assert row["remaining_loans_without_detail"] == 0.0


def test_remaining_loans_without_detail_are_dropped_and_reported() -> None:
    row = _derive(_family(remaining=4_000)).iloc[0]

    assert row[SCF_EDUCATION_DEBT_TOTAL_TARGET] == 0.0
    assert row["remaining_loans_without_detail"] == 4_000
    assert row["head_own_share"] == 0.0
    assert row["head_is_ibr_new_borrower"] == 0.0


@pytest.mark.parametrize("code", [6, 7, -7, 0, 4])
def test_multiple_unrelated_other_and_unknown_codes_go_to_the_head(code) -> None:
    row = _derive(_family(_loan(8_000, whose=code))).iloc[0]

    assert row["head_federal_student_loan_balance"] == 8_000
    assert row["spouse_federal_student_loan_balance"] == 0.0
    assert row["federal_parent_plus_loan_balance"] == 0.0
    assert row["head_own_share"] == 1.0


def test_graduate_share_weights_by_amount_borrowed_with_balance_fallback() -> None:
    weighted = _derive(
        _family(
            _loan(5_000, program=14, borrowed=30_000),
            _loan(9_000, program=11, borrowed=10_000),
        )
    ).iloc[0]
    fallback = _derive(
        _family(
            _loan(5_000, program=15, borrowed=0.0),
            _loan(15_000, program=16, borrowed=0.0),
        )
    ).iloc[0]
    parent_plus_only = _derive(_family(_loan(5_000, whose=3, program=13))).iloc[0]

    assert weighted["head_graduate_share"] == pytest.approx(0.75)
    assert fallback["head_graduate_share"] == pytest.approx(0.25)
    # A Parent PLUS loan is not the parent's own loan.
    assert parent_plus_only["head_graduate_share"] == 0.0


def test_new_borrower_status_counts_2014_as_new_and_needs_every_loan() -> None:
    rows = _derive(
        _family(_loan(1_000, year=2014)),
        _family(_loan(1_000, year=2013)),
        _family(_loan(1_000, year=2020), _loan(1_000, year=2005, whose=3)),
        _family(_loan(1_000, year=2020), _loan(1_000, year=2005, federal=5)),
    )

    assert rows["head_is_ibr_new_borrower"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_derivation_refuses_a_missing_loan_item() -> None:
    frame = pd.DataFrame.from_records([_family(_loan(1_000))]).drop(columns="x9338")
    with pytest.raises(ValueError, match="x9338"):
        derive_scf_education_debt_targets(frame)
    listing = pd.DataFrame.from_records([_family(_loan(1_000))]).drop(columns="x8000")
    with pytest.raises(ValueError, match="x8000"):
        derive_scf_education_debt_targets(listing)


def test_reversed_families_swap_the_codes_of_loans_4_and_6_back() -> None:
    loans = [_loan(1_000.0 * (index + 1), whose=1) for index in range(6)]
    reversed_row, unreversed_row = _derive(
        _family(*loans, reversed_codes=True), _family(*loans)
    ).to_dict("records")

    # Loans 4 and 6 (4,000 and 6,000) still name the original respondent, now
    # the spouse or partner, in a reversed family; loans 1, 2, 3 and 5 do not.
    assert reversed_row["head_federal_student_loan_balance"] == 11_000
    assert reversed_row["spouse_federal_student_loan_balance"] == 10_000
    assert unreversed_row["head_federal_student_loan_balance"] == 21_000
    assert unreversed_row["spouse_federal_student_loan_balance"] == 0.0
    spouse_coded = _derive(
        _family(
            _loan(1_000, whose=2),
            _loan(1_000, whose=2),
            _loan(1_000, whose=2),
            _loan(4_000, whose=2),
            _loan(1_000, whose=3),
            _loan(6_000, whose=2),
            reversed_codes=True,
        )
    ).iloc[0]
    assert spouse_coded["head_federal_student_loan_balance"] == 10_000
    assert spouse_coded["spouse_federal_student_loan_balance"] == 3_000
    # Code 3 names neither partner, so it is never swapped.
    assert spouse_coded["federal_parent_plus_loan_balance"] == 1_000


_CHILD = 4
_GRANDCHILD = 5


@pytest.mark.parametrize(
    ("listed", "loan", "holder"),
    [
        # A child aged 18-27 (code 25) who could have borrowed it in 2022.
        (((_CHILD, 25),), _loan(9_000, whose=3, year=2022), "child"),
        # No listed child: the student lives elsewhere.
        ((), _loan(9_000, whose=3, year=2022), "parent_plus"),
        # Only a 13-17-year-old (code 17) and a grandchild.
        (((_CHILD, 17), (_GRANDCHILD, 25)), _loan(9_000, whose=3, year=2022),
         "parent_plus"),
        # At most 27 now, so at most 2 when a 2000 loan was borrowed.
        (((_CHILD, 25),), _loan(9_000, whose=3, year=2000), "parent_plus"),
        # Above the $31,000 dependent limit while possibly dependent.
        (((_CHILD, 25),), _loan(9_000, whose=3, year=2022, borrowed=40_000),
         "parent_plus"),
        # At the limit itself, which the child's own loans can reach.
        (((_CHILD, 25),), _loan(9_000, whose=3, year=2022, borrowed=31_000),
         "child"),
        # A graduate loan is never Parent PLUS, whatever its size.
        (((_CHILD, 25),),
         _loan(9_000, whose=3, year=2022, program=13, borrowed=80_000), "child"),
        # A child aged 38-42 (code 40) was at least 28 in 2015: independent.
        (((_CHILD, 40),), _loan(9_000, whose=3, year=2015, borrowed=50_000),
         "child"),
        # Aged 28-32 (code 30), so possibly 23 in 2018: possibly dependent.
        (((_CHILD, 30),), _loan(9_000, whose=3, year=2018, borrowed=50_000),
         "parent_plus"),
    ],
)  # fmt: skip
def test_loans_for_a_child_go_to_a_co_resident_adult_child_who_fits(
    listed, loan, holder
) -> None:
    row = _derive(_family(loan, listed=listed)).iloc[0]

    child = row["child_federal_student_loan_balance"]
    parent_plus = row["federal_parent_plus_loan_balance"]
    expected = (9_000, 0.0) if holder == "child" else (0.0, 9_000)
    assert (child, parent_plus) == expected
    assert row["child_own_share"] + row["parent_plus_share"] == 1.0
    assert row["head_federal_student_loan_balance"] == 0.0


def test_child_targets_carry_the_childs_flags_shares_and_ages() -> None:
    row = _derive(
        _family(
            _loan(10_000, whose=1, year=2010),
            _loan(6_000, whose=3, year=2020, program=12),
            _loan(4_000, whose=3, year=2023, program=13, borrowed=12_000),
            _loan(2_000, whose=3, year=2001),
            listed=((_CHILD, 30), (_CHILD, 25), (_CHILD, 12)),
        )
    ).iloc[0]

    assert row["child_federal_student_loan_balance"] == 10_000
    # The 2001 loan fits no listed adult child (at most 32 now, 8 then).
    assert row["federal_parent_plus_loan_balance"] == 2_000
    assert row[SCF_EDUCATION_DEBT_TOTAL_TARGET] == 22_000
    assert row["child_own_share"] == pytest.approx(10_000 / 22_000)
    assert row["child_graduate_share"] == pytest.approx(12_000 / 18_000)
    assert row["child_is_ibr_new_borrower"] == 1.0
    # The head holds a 2010 loan and the 2001 Parent PLUS loan.
    assert row["head_is_ibr_new_borrower"] == 0.0
    assert row[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] == 2.0
    assert row["youngest_adult_child_age_code"] == 25.0
    assert row["child_borrower_age_code"] == 25.0
    shares = [f"{holder}_own_share" for holder in ("head", "spouse", "child")]
    assert row[[*shares, "parent_plus_share"]].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Loader                                                                        #
# --------------------------------------------------------------------------- #
def _summary_and_full(n_families: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float]] = []
    full_rows: list[dict[str, float]] = []
    for family in range(1, n_families + 1):
        loans = [_loan(1_000.0 * family, whose=1 + family % 3)] if family % 2 else []
        for implicate in range(1, 6):
            y1 = family * 10 + implicate
            full = {"y1": float(y1), "yy1": float(family), "x19": 40.0 + family}
            full.update(_family(*loans))
            full_rows.append(full)
            debt = sum(loan["balance"] for loan in loans)
            summary_rows.append(
                {
                    "y1": float(y1),
                    "yy1": float(family),
                    "wgt": 1_000.0 + family,
                    "age": 25.0 + family,
                    "hhsex": 1.0 + family % 2,
                    "racecl5": float(family % 5 + 1),
                    "racecl4": float(family % 4 + 1),
                    "married": float(family % 2),
                    "kids": float(family % 3),
                    "wageinc": 30_000.0 + family,
                    "intdivinc": 100.0 * family,
                    "ssretinc": 0.0,
                    "income": 40_000.0 + 1_000.0 * family,
                    "kginc": 500.0 * family,
                    "edn_inst": debt,
                    "educ": 12.0,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(full_rows).iloc[::-1]


def _write(tmp_path, summary: pd.DataFrame, full: pd.DataFrame):
    summary_path = tmp_path / "rscfp2025.dta"
    full_path = tmp_path / "p25i6.dta"
    summary.to_stata(summary_path, write_index=False)
    full.to_stata(full_path, write_index=False)
    return summary_path, full_path


def test_loader_joins_both_files_and_derives_targets_and_predictors(tmp_path) -> None:
    summary, full = _summary_and_full()
    donor = load_scf_2025_education_debt_donor(
        *_write(tmp_path, summary, full), expected_rows=None
    )

    assert len(donor) == len(summary)
    assert set(SCF_EDUCATION_DEBT_PREDICTORS) <= set(donor.columns)
    assert set(SCF_EDUCATION_DEBT_COMPOSITION_TARGETS) <= set(donor.columns)
    first = donor.iloc[0]
    assert first["household_income"] == pytest.approx(41_000.0 - 500.0)
    assert first["spouse_age"] == 41.0
    assert sorted(donor["implicate"].unique()) == [1, 2, 3, 4, 5]
    family_three = donor.loc[donor["yy1"] == 3].iloc[0]
    assert family_three[SCF_EDUCATION_DEBT_TOTAL_TARGET] == 3_000.0
    assert family_three["head_federal_student_loan_balance"] == 3_000.0


def test_loader_refuses_drifted_education_debt_and_row_counts(tmp_path) -> None:
    summary, full = _summary_and_full()
    with pytest.raises(ValueError, match="expected 30"):
        load_scf_2025_education_debt_donor(
            *_write(tmp_path, summary.iloc[:-5], full.iloc[:-5]), expected_rows=30
        )
    drifted = summary.copy()
    drifted.loc[0, "edn_inst"] = 99_999.0
    with pytest.raises(ValueError, match="edn_inst"):
        load_scf_2025_education_debt_donor(
            *_write(tmp_path, drifted, full), expected_rows=None
        )


def test_loader_refuses_an_unmatched_implicate(tmp_path) -> None:
    summary, full = _summary_and_full()
    full = full.copy()
    full.iloc[0, full.columns.get_loc("y1")] = 999.0
    with pytest.raises(ValueError, match="unmatched"):
        load_scf_2025_education_debt_donor(
            *_write(tmp_path, summary, full), expected_rows=None
        )


# --------------------------------------------------------------------------- #
# Declaration and source files                                                  #
# --------------------------------------------------------------------------- #
def test_stage_declares_the_person_outputs_and_pinned_2025_files() -> None:
    spec = us_scf_education_debt_stage_spec()

    assert set(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS) <= set(spec.outputs)
    locators = {artifact.get("locator"): artifact for artifact in spec.artifacts}
    for public_file in (SCF_2025_SUMMARY_EXTRACT, SCF_2025_FULL_EXTRACT):
        artifact = locators[public_file.url]
        assert artifact["vintage"] == "2025"
        assert artifact["sha256"] == public_file.zip_sha256
        assert artifact["member_sha256"] == public_file.member_sha256
        assert artifact["expected_rows"] == public_file.rows
    # Net worth and the other wealth outputs stay on the 2022 files.
    assert spec.artifacts[0]["vintage"] == "2022"
    assert spec.artifacts[1]["vintage"] == "2022"
    assert spec.artifacts[2]["member"] == "pu2023.csv"
    # Both QRFs condition on the production predictors; the borrower fit adds
    # the adult children and the drawn total.
    fits = {
        operation.parameters["donor"]: operation.parameters
        for operation in spec.operations
        if operation.kind == "fit_weighted_qrf"
        and str(operation.parameters.get("donor", "")).startswith("scf_2025")
    }
    assert tuple(fits["scf_2025_education_debt"]["predictors"]) == (
        SCF_EDUCATION_DEBT_PREDICTORS
    )
    assert tuple(fits["scf_2025_education_debt_borrowers"]["predictors"]) == (
        *SCF_EDUCATION_DEBT_PREDICTORS,
        SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
        SCF_EDUCATION_DEBT_TOTAL_TARGET,
    )
    assert scf_education_debt_composition_predictors() == tuple(
        fits["scf_2025_education_debt_borrowers"]["predictors"]
    )
    assert tuple(fits["scf_2025_education_debt_borrowers"]["targets"]) == (
        SCF_EDUCATION_DEBT_COMPOSITION_TARGETS
    )
    # Parent PLUS is drawn first, for itself, not left as a residual.
    assert SCF_EDUCATION_DEBT_COMPOSITION_TARGETS[0] == "parent_plus_share"


def test_production_predictors_drop_only_the_reviewed_exclusions() -> None:
    candidates = education_debt.SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS
    excluded = education_debt.SCF_EDUCATION_DEBT_EXCLUDED_PREDICTORS
    assert set(excluded) == {
        "interest_dividend_income",
        "social_security_pension_income",
    }
    assert all(reason.strip() for reason in excluded.values())
    assert SCF_EDUCATION_DEBT_PREDICTORS == tuple(
        predictor for predictor in candidates if predictor not in excluded
    )
    assert education_debt.SCF_EDUCATION_DEBT_RECIPIENT_DEFINITION == (
        education_debt.RecipientDefinition(
            reference_person="householder_couple",
            partnered="spouse_or_partner",
            children="of_reference_or_partner",
            income_unit="reference_couple_and_children",
        )
    )


def _zip_with(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_verifies_both_pins_and_reuses_a_verified_cache(
    tmp_path, monkeypatch
) -> None:
    member = b"stata bytes"
    archive = _zip_with("p25i6.dta", member)
    pinned = ScfPublicFile(
        url="https://example.invalid/scf2025s.zip",
        member="p25i6.dta",
        zip_sha256=hashlib.sha256(archive).hexdigest(),
        member_sha256=hashlib.sha256(member).hexdigest(),
        rows=1,
    )
    fetched: list[str] = []

    def fake_urlopen(request, timeout):
        fetched.append(request.full_url)
        return _Response(archive)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    path = fetch_scf_public_file(pinned, tmp_path)
    assert path.read_bytes() == member
    assert fetch_scf_public_file(pinned, tmp_path) == path
    assert fetched == [pinned.url]

    wrong_zip = ScfPublicFile(
        url=pinned.url,
        member=pinned.member,
        zip_sha256="0" * 64,
        member_sha256=pinned.member_sha256,
        rows=1,
    )
    with pytest.raises(ValueError, match="sha-256"):
        fetch_scf_public_file(wrong_zip, tmp_path / "other")


# --------------------------------------------------------------------------- #
# Imputation                                                                    #
# --------------------------------------------------------------------------- #
def _ready_donor(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(1073)
    donor = pd.DataFrame({name: rng.normal(size=n) for name in SCF_WEALTH_PREDICTORS})
    donor["age"] = rng.integers(20, 85, n).astype(np.float64)
    donor["is_female"] = rng.integers(0, 2, n).astype(np.float64)
    donor["cps_race"] = rng.integers(1, 8, n).astype(np.float64)
    donor["is_married"] = rng.integers(0, 2, n).astype(np.float64)
    donor["own_children_in_household"] = rng.integers(0, 3, n).astype(np.float64)
    donor["employment_income"] = rng.gamma(2.0, 25_000.0, n)
    donor["interest_dividend_income"] = rng.gamma(1.0, 800.0, n)
    donor["social_security_pension_income"] = rng.gamma(1.0, 5_000.0, n)
    donor["household_income"] = donor["employment_income"] + rng.gamma(1.0, 5_000.0, n)
    donor[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] = np.where(
        donor["age"] >= 40, rng.integers(0, 3, n), 0
    ).astype(np.float64)
    borrower = (rng.random(n) < 0.2) & (donor["age"] < 65)
    total = np.where(borrower, rng.gamma(1.5, 25_000.0, n).round(), 0.0)
    # Holders: 0 head, 1 spouse, 2 Parent PLUS, 3 an adult child.
    holder = rng.choice(4, n, p=[0.55, 0.25, 0.1, 0.1])
    holder = np.where(
        (holder == 3) & (donor[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] == 0),
        0,
        holder,
    )
    for index, column in enumerate(
        ("head_own_share", "spouse_own_share", "parent_plus_share", "child_own_share")
    ):
        donor[column] = np.where(borrower, (holder == index).astype(float), 0.0)
    donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] = total
    for prefix, choices in (
        ("head", [0.0, 0.5, 1.0]),
        ("spouse", [0.0, 1.0]),
        ("child", [0.0, 1.0]),
    ):
        donor[f"{prefix}_graduate_share"] = np.where(
            donor[f"{prefix}_own_share"] > 0, rng.choice(choices, n), 0.0
        )
    head_holds = (donor["head_own_share"] + donor["parent_plus_share"]) > 0
    for prefix, holds in (
        ("head", head_holds),
        ("spouse", donor["spouse_own_share"] > 0),
        ("child", donor["child_own_share"] > 0),
    ):
        donor[f"{prefix}_is_ibr_new_borrower"] = np.where(
            holds, (rng.random(n) < 0.5).astype(float), 0.0
        )
    donor[_DONOR_WEIGHT_COLUMN] = rng.uniform(500.0, 2_000.0, n)
    return donor


def _features(donor: pd.DataFrame) -> pd.DataFrame:
    return donor.loc[
        :, [*SCF_EDUCATION_DEBT_PREDICTORS, SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR]
    ]


def _person_rows(n_households: int = 150, *, adult_children: bool = False):
    """Couples (A_SPOUSE pointers) and single adults with a child.

    With ``adult_children``, every couple's child is 19 to 24 and every fourth
    couple also has a 30-year-old child (line 4).
    """

    rng = np.random.default_rng(97)
    records: list[dict[str, object]] = []
    person_id = 1
    for household_id in range(1, n_households + 1):
        couple = household_id % 3 != 0
        members = (
            [(1, 2, 1, 0), (2, 1, 1, 1), (3, 0, 7, 0)]
            if couple
            else [(1, 0, 7, 0), (2, 0, 7, 1)]
        )
        if adult_children and couple and household_id % 4 == 0:
            members.append((4, 0, 7, 1))
        for line, spouse, marital, female in members:
            child = line >= 3 or (not couple and line == 2)
            if not child:
                age = float(rng.integers(24, 60))
            elif adult_children and couple:
                age = 30.0 if line == 4 else float(rng.integers(19, 25))
            else:
                age = float(rng.integers(5, 17))
            records.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "PH_SEQ": household_id,
                    "A_LINENO": line,
                    "A_SPOUSE": spouse,
                    "age": age,
                    "is_female": bool(female),
                    "PRDTRACE": int(rng.integers(1, 5)),
                    "PRDTHSP": int(rng.integers(0, 2)),
                    "A_MARITL": marital,
                    "PEPAR1": 1 if child else -1,
                    "PEPAR2": -1,
                    "employment_income_before_lsr": 0.0
                    if child
                    else float(rng.gamma(2.0, 20_000.0)),
                    "taxable_interest_income": float(rng.gamma(1.0, 300.0)),
                }
            )
            person_id += 1
    return pd.DataFrame.from_records(records)


def _us_frame(person: pd.DataFrame) -> Frame:
    person = person.copy()
    household_ids = np.sort(person["person_household_id"].unique())
    n_person = len(person)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n_person) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
        "family": pd.DataFrame({"family_id": household_ids + 3_000}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.arange(n_person) + 4_000}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(len(household_ids), 1_000.0),
                kind=WeightKind.DESIGN,
            )
        },
    )


def test_household_draws_are_seeded_coherent_and_nonnegative() -> None:
    donor = _ready_donor()
    features = _features(donor).iloc[:200]
    first = draw_scf_education_debt_households(features, donor, seed=3, n_estimators=15)
    second = draw_scf_education_debt_households(
        features, donor, seed=3, n_estimators=15
    )
    pd.testing.assert_frame_equal(first, second)

    total = first[SCF_EDUCATION_DEBT_TOTAL_TARGET].to_numpy()
    parts = (
        first["head_federal_student_loan_balance"]
        + first["spouse_federal_student_loan_balance"]
        + first["child_federal_student_loan_balance"]
        + first["federal_parent_plus_loan_balance"]
    ).to_numpy()
    assert (total >= 0).all() and (total > 0).any()
    np.testing.assert_array_equal(parts, total)
    borrowers = total > 0
    shares = first[list(SCF_EDUCATION_DEBT_HOLDER_SHARE_TARGETS)].sum(axis=1)
    np.testing.assert_allclose(shares[borrowers], 1.0)
    assert not first.loc[~borrowers, list(SCF_EDUCATION_DEBT_COMPOSITION_TARGETS)].any(
        axis=None
    )
    # A household without an adult child holds no child balance.
    no_adult_child = features[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] == 0
    assert (first.loc[no_adult_child, "child_federal_student_loan_balance"] == 0).all()


def test_a_donor_family_without_parent_plus_rarely_draws_one() -> None:
    """Parent PLUS is drawn for itself, not left over by the own shares.

    Own debt splits between the two partners in fractions, and only families
    with a reference person 55 or older hold Parent PLUS. The draw for every
    donor family's own predictors gives Parent PLUS to few of the others.
    """

    rng = np.random.default_rng(1074)
    donor = _ready_donor(1_500)
    borrower = donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0
    older = donor["age"] >= 55
    head_share = rng.uniform(0.2, 0.8, len(donor))
    donor["head_own_share"] = np.where(borrower & ~older, head_share, 0.0)
    donor["spouse_own_share"] = np.where(borrower & ~older, 1.0 - head_share, 0.0)
    donor["parent_plus_share"] = np.where(borrower & older, 1.0, 0.0)
    donor["child_own_share"] = 0.0
    donor["child_graduate_share"] = 0.0
    donor["child_is_ibr_new_borrower"] = 0.0
    drawn = draw_scf_education_debt_households(
        _features(donor), donor, seed=8, n_estimators=20
    )
    with_debt = drawn[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0
    draws_plus = drawn["federal_parent_plus_loan_balance"] > 0
    assert draws_plus[with_debt & ~older].mean() < 0.02
    assert draws_plus[with_debt & older].mean() > 0.9
    # A donor with no Parent PLUS at all never draws it.
    donor["parent_plus_share"] = 0.0
    donor["head_own_share"] = np.where(borrower, head_share, 0.0)
    donor["spouse_own_share"] = np.where(borrower, 1.0 - head_share, 0.0)
    none = draw_scf_education_debt_households(
        _features(donor), donor, seed=9, n_estimators=20
    )
    assert (none["federal_parent_plus_loan_balance"] == 0).all()


def test_imputation_reaches_only_reference_persons_spouses_and_adult_children() -> None:
    frame = _us_frame(_person_rows())
    imputed = impute_us_scf_education_debt(
        frame, _ready_donor(), seed=5, n_estimators=15
    )
    person = frame.table("person")

    assert list(imputed.columns) == list(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS)
    adult = person["age"] >= 18
    balance = (
        imputed["federal_student_loan_balance"]
        + imputed["federal_parent_plus_loan_balance"]
    )
    assert (balance >= 0).all() and (balance > 0).any()
    assert (balance[~adult] == 0).all()
    # Parent PLUS sits on the reference person (line 1 in these fixtures).
    assert (
        imputed.loc[person["A_LINENO"] != 1, "federal_parent_plus_loan_balance"] == 0
    ).all()
    own = imputed["federal_student_loan_balance"]
    assert (imputed.loc[own <= 0, "federal_student_loan_graduate_share"] == 0).all()
    assert imputed["federal_student_loan_graduate_share"].between(0, 1).all()
    assert imputed["is_income_based_repayment_new_borrower"].dtype == bool
    assert not (
        imputed["is_income_based_repayment_new_borrower"] & (balance <= 0)
    ).any()
    single_adults = person["A_MARITL"].eq(7) & adult
    assert (person.loc[single_adults, "A_LINENO"] == 1).all()


def test_a_spouse_draw_without_a_spouse_moves_to_the_reference_person() -> None:
    frame = _us_frame(_person_rows(30))
    donor = _ready_donor()
    # Every donor family borrows, so every household draws a positive total and
    # the test checks only where the spouse's draw lands. With the synthetic
    # donor's random incidence, a 30-household frame otherwise draws no debt for
    # any couple on about 2 seeds in 20.
    donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] = np.maximum(
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET], 1_000.0
    )
    donor["head_own_share"] = 0.0
    donor["parent_plus_share"] = 0.0
    donor["child_own_share"] = 0.0
    donor["spouse_own_share"] = np.where(
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0, 1.0, 0.0
    )
    donor["head_graduate_share"] = 0.0
    donor["head_is_ibr_new_borrower"] = 0.0
    imputed = impute_us_scf_education_debt(frame, donor, seed=11, n_estimators=10)
    person = frame.table("person")

    single = person["person_household_id"] % 3 == 0
    reference = person["A_LINENO"] == 1
    couple_spouse = ~single & (person["A_LINENO"] == 2)
    own = imputed["federal_student_loan_balance"]
    assert (own[single & ~reference] == 0).all()
    assert (own[single & reference] > 0).any()
    assert (own[~single & reference] == 0).all()
    assert (own[couple_spouse] > 0).any()
    assert (imputed["federal_parent_plus_loan_balance"] == 0).all()


def test_a_childs_draw_goes_to_the_youngest_adult_child() -> None:
    frame = _us_frame(_person_rows(40, adult_children=True))
    donor = _ready_donor()
    donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] = np.maximum(
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET], 1_000.0
    )
    for column in ("head_own_share", "spouse_own_share", "parent_plus_share"):
        donor[column] = 0.0
    donor["child_own_share"] = 1.0
    donor["child_graduate_share"] = 1.0
    donor["child_is_ibr_new_borrower"] = 1.0
    donor["head_is_ibr_new_borrower"] = 0.0
    donor[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR] = 1.0
    imputed = impute_us_scf_education_debt(frame, donor, seed=4, n_estimators=10)
    person = frame.table("person")

    single = person["person_household_id"] % 3 == 0
    youngest_child = ~single & (person["A_LINENO"] == 3)
    older_child = person["A_LINENO"] == 4
    own = imputed["federal_student_loan_balance"]
    plus = imputed["federal_parent_plus_loan_balance"]
    # Couples have an adult child, who holds the whole draw as an own loan.
    assert (own[youngest_child] > 0).all()
    assert (own[~youngest_child & ~single] == 0).all()
    assert (own[older_child] == 0).all()
    assert (
        imputed.loc[youngest_child, "federal_student_loan_graduate_share"] == 1
    ).all()
    assert imputed.loc[youngest_child, "is_income_based_repayment_new_borrower"].all()
    # Single parents' children are minors: the child's draw is the head's
    # Parent PLUS loan there, and nobody else in the household holds any.
    single_head = single & (person["A_LINENO"] == 1)
    assert (plus[single_head] > 0).all()
    assert (plus[~single_head] == 0).all()
    assert (own[single] == 0).all()


def test_adult_children_count_children_of_either_partner_aged_18_or_over() -> None:
    frame = _us_frame(
        pd.DataFrame.from_records(
            _household(
                [
                    {"age": 50.0, "A_SPOUSE": 2, "A_MARITL": 1},
                    {"age": 48.0, "A_SPOUSE": 1, "A_MARITL": 1, "is_female": True},
                    {"age": 25.0, "PEPAR1": 2},
                    {"age": 19.0, "PEPAR1": 1, "PEPAR2": 2},
                    {"age": 19.0, "PEPAR1": 1},
                    {"age": 12.0, "PEPAR1": 1},
                    {"age": 22.0},
                ]
            )
        )
    )
    recipients = education_debt._recipients(frame)

    assert recipients.features[
        SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR
    ].tolist() == [3.0]
    # The youngest adult child, the lower line number on a tie.
    assert recipients.child_position.tolist() == [3]


def test_with_inputs_is_idempotent_and_passes_its_gate() -> None:
    frame = _us_frame(_person_rows())
    out = with_us_scf_education_debt_inputs(
        frame,
        seed=1,
        time_period=TIME_PERIOD,
        scf_education_debt_donor=_ready_donor(),
        n_estimators=15,
    )
    again = with_us_scf_education_debt_inputs(
        out,
        seed=2,
        time_period=TIME_PERIOD,
        scf_education_debt_donor=_ready_donor(),
        n_estimators=15,
    )
    assert again is out
    gate = us_scf_education_debt_signal_gate(out)
    assert gate.passed, gate.failures
    summary = us_scf_education_debt_summary(out)
    assert summary["household_income_columns_present"] == [
        "employment_income_before_lsr",
        "taxable_interest_income",
    ]
    fit = us_scf_education_debt_fit_table(out)
    assert 0 < fit["households"]["share_with_federal_education_debt"] < 1
    assert set(fit["borrowers"]["balance_bands"]) == {
        label for label, _, _ in education_debt.SCF_EDUCATION_DEBT_BALANCE_BANDS
    }


def test_gate_refuses_missing_negative_and_incoherent_columns() -> None:
    frame = _us_frame(_person_rows())
    missing = us_scf_education_debt_signal_gate(frame)
    assert not missing.passed
    assert "person columns missing" in missing.failures[0]

    out = with_us_scf_education_debt_inputs(
        frame,
        seed=1,
        time_period=TIME_PERIOD,
        scf_education_debt_donor=_ready_donor(),
        n_estimators=15,
    )
    tables = {entity: out.table(entity).copy() for entity in out.entities}
    person = tables["person"]
    person.loc[person.index[0], "federal_student_loan_balance"] = -1.0
    no_balance = person["federal_student_loan_balance"] <= 0
    person.loc[no_balance, "federal_student_loan_graduate_share"] = 0.5
    person["is_income_based_repayment_new_borrower"] = True
    broken = Frame(
        tables,
        out.schema,
        {entity: out.weights_for(entity) for entity in out.weighted_entities},
    )
    failures = " ".join(us_scf_education_debt_signal_gate(broken).failures)
    assert "negative" in failures
    assert "without a federal_student_loan_balance" in failures
    assert "without a federal balance" in failures


# --------------------------------------------------------------------------- #
# Fit table                                                                     #
# --------------------------------------------------------------------------- #
def test_fit_table_matches_hand_computed_shares_medians_and_bands() -> None:
    table = education_debt_fit_table(
        household_debt=np.array([0.0, 4_000.0, 30_000.0, 250_000.0]),
        household_weight=np.array([2.0, 1.0, 1.0, 1.0]),
        head_age=np.array([70.0, 24.0, 30.0, 40.0]),
        race_group=np.array([1, 1, 2, 3]),
        income_quintile=np.array([1, 2, 3, 5]),
        borrower_balance=np.array([4_000.0, 30_000.0, 250_000.0]),
        borrower_age=np.array([24.0, 30.0, 40.0]),
        borrower_weight=np.array([1.0, 1.0, 2.0]),
        parent_plus_total=25_000.0,
        graduate_numerator=10.0,
        graduate_denominator=40.0,
        new_borrower_weight=1.0,
    )

    households = table["households"]
    assert households["share_with_federal_education_debt"] == pytest.approx(3 / 5)
    assert households["mean_among_borrowers"] == pytest.approx(284_000 / 3)
    assert households["median_among_borrowers"] == 30_000.0
    assert households["parent_plus_share_of_total"] == pytest.approx(25_000 / 284_000)
    assert table["household_balance_bands"]["under_5k"] == pytest.approx(1 / 3)
    assert table["household_balance_bands"]["200k_plus"] == pytest.approx(1 / 3)
    assert (
        table["by_head_age_band"]["62_and_older"]["share_with_federal_education_debt"]
        == 0.0
    )
    assert table["by_race"]["white_non_hispanic"][
        "share_with_federal_education_debt"
    ] == pytest.approx(1 / 3)
    borrowers = table["borrowers"]
    assert borrowers["median_balance"] == 30_000.0
    assert borrowers["age_bands"]["24_and_younger"] == pytest.approx(0.25)
    assert borrowers["age_bands"]["35_to_49"] == pytest.approx(0.5)
    assert borrowers["graduate_share_of_own_original_principal"] == 0.25
    assert borrowers["ibr_new_borrower_share"] == pytest.approx(0.25)


def test_donor_fit_table_tabulates_observed_and_imputed_rows_alike() -> None:
    donor = _ready_donor(200)
    donor["head_federal_student_loan_balance"] = (
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] * donor["head_own_share"]
    )
    donor["spouse_federal_student_loan_balance"] = (
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] * donor["spouse_own_share"]
    )
    donor["child_federal_student_loan_balance"] = (
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] * donor["child_own_share"]
    )
    donor["federal_parent_plus_loan_balance"] = (
        donor[SCF_EDUCATION_DEBT_TOTAL_TARGET]
        - donor["head_federal_student_loan_balance"]
        - donor["spouse_federal_student_loan_balance"]
        - donor["child_federal_student_loan_balance"]
    )
    donor["racecl4"] = (np.arange(len(donor)) % 4 + 1).astype(float)
    donor["spouse_age"] = donor["age"]
    donor["child_borrower_age_code"] = np.where(donor["child_own_share"] > 0, 25.0, 0.0)
    donor["youngest_adult_child_age_code"] = donor["child_borrower_age_code"]
    observed = scf_education_debt_donor_fit_table(donor)
    same = scf_education_debt_donor_fit_table(donor, imputed=donor)
    assert json.dumps(observed, sort_keys=True) == json.dumps(same, sort_keys=True)
    expected_share = float(
        donor.loc[
            donor[SCF_EDUCATION_DEBT_TOTAL_TARGET] > 0, _DONOR_WEIGHT_COLUMN
        ].sum()
        / donor[_DONOR_WEIGHT_COLUMN].sum()
    )
    assert observed["households"]["share_with_federal_education_debt"] == (
        pytest.approx(expected_share)
    )
    # Each child borrower counts once, its weight spread over ages 18-27.
    children = donor["child_own_share"] > 0
    assert children.any()
    head_or_spouse = (
        (donor["head_own_share"] + donor["parent_plus_share"] > 0).astype(float)
        + (donor["spouse_own_share"] > 0)
        + children
    )
    assert observed["borrowers"]["borrower_weight"] == pytest.approx(
        float(np.dot(head_or_spouse, donor[_DONOR_WEIGHT_COLUMN]))
    )


# --------------------------------------------------------------------------- #
# Recipient definitions                                                         #
# --------------------------------------------------------------------------- #
def _household(members: list[dict[str, object]], household_id: int = 1) -> list[dict]:
    rows = []
    for index, member in enumerate(members, start=1):
        rows.append(
            {
                "person_id": household_id * 100 + index,
                "person_household_id": household_id,
                "PH_SEQ": household_id,
                "A_LINENO": index,
                "A_SPOUSE": 0,
                "PECOHAB": 0,
                "age": 40.0,
                "is_female": False,
                "PRDTRACE": 1,
                "PRDTHSP": 0,
                "A_MARITL": 7,
                "PEPAR1": -1,
                "PEPAR2": -1,
                "employment_income_before_lsr": 0.0,
                **member,
            }
        )
    return rows


def _definition_frame() -> Frame:
    rows = [
        # 1: a 30-year-old woman householder with her cohabiting partner (31),
        # his child, and her 62-year-old mother.
        *_household(
            [
                {"age": 30.0, "is_female": True, "PECOHAB": 2,
                 "employment_income_before_lsr": 40_000.0},
                {"age": 31.0, "PECOHAB": 1, "employment_income_before_lsr": 50_000.0},
                {"age": 8.0, "PEPAR1": 2},
                {"age": 62.0, "is_female": True,
                 "employment_income_before_lsr": 30_000.0},
            ],
            household_id=1,
        ),
        # 2: a married same-sex couple, the householder younger.
        *_household(
            [
                {"age": 35.0, "A_SPOUSE": 2, "A_MARITL": 1},
                {"age": 45.0, "A_SPOUSE": 1, "A_MARITL": 1},
            ],
            household_id=2,
        ),
    ]  # fmt: skip
    person = pd.DataFrame.from_records(rows)
    return _us_frame(person)


def test_recipient_definition_refuses_unknown_rules() -> None:
    with pytest.raises(ValueError, match="reference_person"):
        education_debt.RecipientDefinition(reference_person="eldest")
    with pytest.raises(ValueError, match="income_unit"):
        education_debt.RecipientDefinition(income_unit="tax_unit")


def test_default_definition_follows_the_scf_definitions() -> None:
    frame = _definition_frame()
    recipients = education_debt._recipients(frame)
    person = frame.table("person")
    features = recipients.features

    # The householder's cohabiting partner is the man of the couple; the
    # same-sex couple's older spouse leads.
    assert person["age"].to_numpy()[recipients.head_position].tolist() == [31.0, 45.0]
    assert features["is_married"].tolist() == [1.0, 1.0]
    # Household 1 counts the partner's child; the mother is no one's child.
    assert features["own_children_in_household"].tolist() == [1.0, 0.0]
    # Income sums over the couple and their children; the mother's wages are
    # outside the SCF primary economic unit.
    assert features["employment_income"].tolist() == [90_000.0, 0.0]
    assert list(features.columns) == [
        *education_debt.SCF_EDUCATION_DEBT_CANDIDATE_PREDICTORS,
        SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR,
    ]
    assert features[SCF_EDUCATION_DEBT_ADULT_CHILDREN_PREDICTOR].tolist() == [0.0, 0.0]


def test_first_receipt_definition_reproduces_the_oldest_adult_rule() -> None:
    frame = _definition_frame()
    recipients = education_debt._recipients(
        frame, education_debt.SCF_EDUCATION_DEBT_FIRST_RECEIPT_DEFINITION
    )
    person = frame.table("person")

    assert person["age"].to_numpy()[recipients.head_position].tolist() == [62.0, 45.0]
    assert recipients.features["is_married"].tolist() == [0.0, 1.0]
    assert recipients.features["own_children_in_household"].tolist() == [0.0, 0.0]


def test_income_units_sum_the_declared_members() -> None:
    frame = _definition_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    # The mother forms her own SPM unit.
    spm = tables["person"]["person_spm_unit_id"].to_numpy().copy()
    spm[3] = 9_999
    tables["person"]["person_spm_unit_id"] = spm
    tables["spm_unit"] = pd.DataFrame({"spm_unit_id": np.unique(spm)})
    frame = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )
    for unit, expected in (
        ("household", 120_000.0),
        ("spm_unit", 90_000.0),
        ("reference_couple_and_children", 90_000.0),
    ):
        definition = education_debt.RecipientDefinition(income_unit=unit)
        features = education_debt._recipients(frame, definition).features
        assert features["employment_income"].iloc[0] == expected, unit
