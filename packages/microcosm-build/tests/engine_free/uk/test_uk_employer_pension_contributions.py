"""Employer pension contributions from ASHE rate bands (microcosm#1069 c9, R6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.employer_pension_contributions import (
    ASHE_EMPLOYER_RATE_BANDS,
    ashe_employer_rates_from_rows,
    draw_employer_pension_contributions,
    employer_scheme_and_sector,
    load_ashe_employer_rates,
)

BANDS = [band for band, _, _ in ASHE_EMPLOYER_RATE_BANDS]


def _share_rows(pension_type: str, sector: str, shares: list[float]) -> list[dict]:
    dimensions = {"contribution_basis": "full_pay", "pension_type": pension_type}
    if sector != "all":
        dimensions.update(
            {
                "sector": sector,
                "sic2007_section": "all",
                "sic2007_summary_category": "all",
            }
        )
    return [
        {
            "concept": "ons.workplace_pension_employer_contribution_band_share",
            "dimensions": {**dimensions, "employer_contribution_rate_band": band},
            "value": share,
        }
        for band, share in zip(BANDS, shares, strict=True)
    ]


def _job_rows(sector: str, db: float, dc: float) -> list[dict]:
    return [
        {
            "concept": "ons.workplace_pension_jobs_by_type",
            "dimensions": {
                "pension_type": pension_type,
                "sector": sector,
                "sic2007_section": "all",
                "sic2007_summary_category": "all",
                "weekly_earnings_band": "all",
            },
            "value": jobs,
        }
        for pension_type, jobs in (
            ("defined_benefit", db),
            ("defined_contribution", dc),
        )
    ]


def _rates():
    rows = [
        # Defined benefit: everything at 20% and over; defined contribution and
        # the all-types row: everything under 4%.
        *_share_rows("defined_benefit", "all", [0, 0, 0, 0, 0, 0, 100]),
        *_share_rows("defined_contribution", "all", [100, 0, 0, 0, 0, 0, 0]),
        *_share_rows("group_personal_pension", "all", [0, 100, 0, 0, 0, 0, 0]),
        *_share_rows("group_stakeholder_pension", "all", [0, 0, 100, 0, 0, 0, 0]),
        *_share_rows("all", "all", [100, 0, 0, 0, 0, 0, 0]),
        *_job_rows("Public", 90.0, 10.0),
        *_job_rows("Private", 10.0, 90.0),
        *_job_rows("all", 50.0, 50.0),
    ]
    return ashe_employer_rates_from_rows(rows)


def test_rates_normalise_the_band_shares_and_split_occupational_by_sector() -> None:
    rates = _rates()

    np.testing.assert_allclose(rates.distribution("defined_benefit", "Public")[-1], 1.0)
    assert rates.defined_benefit_share == {"Public": 0.9, "Private": 0.1, "all": 0.5}


def test_each_member_draws_a_rate_inside_their_band() -> None:
    rates = _rates()
    count = 4_000
    ids = np.arange(1, count + 1)
    scheme = np.tile([1, 2, 3, 4, 0], count // 5)
    sector = np.tile(["Public", "Private"], count // 2).astype(object)
    income = np.full(count, 30_000.0)

    amounts, receipt = draw_employer_pension_contributions(
        ids, employment_income=income, scheme=scheme, sector=sector, rates=rates
    )

    rate = amounts / income
    assert (amounts[scheme == 0] == 0.0).all()
    # Group personal pensions sit in 4% to < 8%.
    gpp = rate[scheme == 1]
    assert ((gpp >= 0.04) & (gpp < 0.08)).all()
    # Public-sector occupational members are mostly defined benefit (20% to 30%).
    public_occupational = rate[(scheme == 2) & (sector == "Public")]
    assert (
        (public_occupational >= 0.2) & (public_occupational <= 0.3)
    ).mean() == pytest.approx(0.9, abs=0.05)
    private_occupational = rate[(scheme == 2) & (sector == "Private")]
    assert (private_occupational < 0.04).mean() == pytest.approx(0.9, abs=0.05)
    stakeholder = rate[scheme == 3]
    assert ((stakeholder >= 0.08) & (stakeholder < 0.10)).all()
    assert receipt["members_with_earnings"] == int((scheme > 0).sum())


def test_a_pension_type_without_any_ashe_row_is_refused() -> None:
    with pytest.raises(ValueError, match="no employer rate distribution"):
        _rates().distribution("group_self_invested_personal_pension", "Private")


def test_members_without_earnings_get_no_employer_contribution() -> None:
    amounts, receipt = draw_employer_pension_contributions(
        np.asarray([1, 2]),
        employment_income=np.asarray([0.0, 20_000.0]),
        scheme=np.asarray([2, 1]),
        sector=np.asarray(["Public", "Private"], dtype=object),
        rates=_rates(),
    )
    assert amounts[0] == 0.0
    assert receipt["members_with_earnings"] == 1


def test_scheme_priority_and_first_stated_sector() -> None:
    penprov = pd.DataFrame({"source_person_id": [1, 1, 2, 3], "stemppen": [1, 2, 5, 4]})
    job = pd.DataFrame({"source_person_id": [1, 1, 3], "jobsect": [-1, 2, 1]})

    scheme, sector = employer_scheme_and_sector(np.asarray([1, 2, 3, 4]), penprov, job)

    # Person 1 is in a group personal and an occupational scheme: occupational
    # wins. Person 2 has only a personal pension (no employer scheme).
    assert scheme.tolist() == [2, 0, 4, 0]
    assert sector.tolist() == ["Public", "all", "Private", "all"]


def test_vendored_rates_load_from_the_pinned_feed() -> None:
    rates = load_ashe_employer_rates()

    for pension_type in (
        "defined_benefit",
        "defined_contribution",
        "group_personal_pension",
    ):
        assert rates.distribution(pension_type, "all").sum() == pytest.approx(1.0)
    assert (
        rates.defined_benefit_share["Public"] > rates.defined_benefit_share["Private"]
    )
