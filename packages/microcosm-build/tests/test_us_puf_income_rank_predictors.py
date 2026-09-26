"""PUF imputation predictors never copy the recipient's own income (microcosm#982).

The PUF QRF used to condition on the recipient's six survey income items, and
the donor side of each of those predictors was the donor's own imputed output.
The forest then learned output = input: every PUF-clone record came back with
its survey wages, and nothing was imputed above ASEC topcodes. These tests pin
the replacement: demographics plus the unit's weighted income rank in its own
population, matched to the donor's rank in the PUF population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_DEMOGRAPHIC_SOURCES,
    PUF_TAX_DETAIL_EARNINGS_COMPONENTS,
    PUF_TAX_DETAIL_EARNINGS_INDICATOR_SOURCE,
    PUF_TAX_DETAIL_INCOME_RANK_COMPONENTS,
    PUF_TAX_DETAIL_INCOME_RANK_SOURCE,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_RANK = "puf_predictor_income_rank_share"
_EARNS = "puf_predictor_has_earnings"


def _output_leaves(source: str) -> set[str]:
    leaves = {source, *puf_support._PREDICTOR_LEAF_ALIASES.get(source, ())}
    if source == "dividend_income":
        leaves |= {"qualified_dividend_income", "non_qualified_dividend_income"}
    return leaves


def test_default_predictors_never_resolve_to_an_imputed_output() -> None:
    outputs = set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS) | set(
        PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    sources = [
        puf_support._predictor_source_column(predictor)
        for predictor in PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    ]
    for predictor, source in zip(
        PUF_TAX_DETAIL_DEFAULT_PREDICTORS, sources, strict=True
    ):
        assert not _output_leaves(source) & outputs, predictor
    # Survey income enters only through the bounded rank and the earnings
    # participation flag, never as a level.
    components = set(PUF_TAX_DETAIL_INCOME_RANK_COMPONENTS) | set(
        PUF_TAX_DETAIL_EARNINGS_COMPONENTS
    )
    assert not set(sources) & components
    assert PUF_TAX_DETAIL_INCOME_RANK_SOURCE in sources
    assert PUF_TAX_DETAIL_EARNINGS_INDICATOR_SOURCE in sources
    assert set(PUF_TAX_DETAIL_DEMOGRAPHIC_SOURCES) <= set(sources)
    # Every component is a donor output, so the donor's rank and flag are
    # taken over exactly the income the imputation hands the recipient.
    for component in components:
        assert _output_leaves(component) & outputs, component


def test_earnings_indicator_flags_any_nonzero_earnings_and_keeps_missing() -> None:
    wages = [0.0, 5.0, 0.0, np.nan, 0.0]
    self_employment = [0.0, 0.0, -3.0, 0.0, 0.0]
    flag = puf_support._earnings_indicator([wages, self_employment])
    # A self-employment loss is earnings participation; a missing amount is
    # not "does not work".
    np.testing.assert_array_equal(flag, [0.0, 1.0, 1.0, np.nan, 0.0])


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        # Each unit sits at the midpoint of the weight slice it stands for.
        ([30.0, 20.0, 10.0], [1.0, 1.0, 2.0], [0.125, 0.375, 0.75]),
        # A tie group is one slice and shares its midpoint.
        ([5.0, 5.0, 1.0], [1.0, 3.0, 4.0], [0.25, 0.25, 0.75]),
        # Input order is irrelevant.
        ([10.0, 30.0, 20.0], [2.0, 1.0, 1.0], [0.75, 0.125, 0.375]),
    ],
)
def test_income_rank_is_the_weighted_midpoint_from_the_top(
    values: list[float],
    weights: list[float],
    expected: list[float],
) -> None:
    np.testing.assert_allclose(
        puf_support._weighted_top_rank_share(values, weights), expected
    )


def test_income_rank_excludes_missing_values_and_ignores_weight_scale() -> None:
    ranked = puf_support._weighted_top_rank_share([1.0, np.nan, 0.0], [1.0, 100.0, 1.0])
    np.testing.assert_allclose(ranked, [0.25, np.nan, 0.75])
    # Clone copies split one household's weight; halving every weight must
    # leave every rank unchanged.
    values = np.asarray([7.0, 3.0, 3.0, 9.0, 0.0])
    weights = np.asarray([2.0, 5.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        puf_support._weighted_top_rank_share(values, weights),
        puf_support._weighted_top_rank_share(values, weights / 2.0),
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, -1.0], "finite and nonnegative"),
        ([1.0, np.inf], "finite and nonnegative"),
        ([0.0, 0.0], "positive total mass"),
    ],
)
def test_income_rank_rejects_invalid_weights(
    weights: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        puf_support._weighted_top_rank_share([1.0, 2.0], weights)


def test_tax_unit_demographics_read_head_spouse_and_dependents() -> None:
    demographics = puf_support._tax_unit_demographics(
        [10, 20, 30],
        [10, 10, 10, 10, 20, 30],
        age=[44, 41, 12, 9, 70, 16],
        is_female=[True, False, False, True, False, True],
        role=[b"HEAD", "spouse", "DEPENDENT", "DEPENDENT", "HEAD", "DEPENDENT"],
        preserve_nulls=True,
    )
    assert demographics["head_age"].tolist()[:2] == [44.0, 70.0]
    assert demographics["spouse_age"].tolist() == [41.0, 0.0, 0.0]
    assert demographics["head_is_female"].tolist()[:2] == [1.0, 0.0]
    assert demographics["dependent_count"].tolist() == [2.0, 0.0, 1.0]
    # A unit without a head is malformed: null under the strict policy so the
    # completeness check names it, zero under the legacy policy.
    assert np.isnan(demographics["head_age"].iloc[2])
    assert np.isnan(demographics["head_is_female"].iloc[2])
    legacy = puf_support._tax_unit_demographics(
        [30],
        [30],
        age=[16],
        is_female=[True],
        role=["DEPENDENT"],
        preserve_nulls=False,
    )
    assert legacy["head_age"].tolist() == [0.0]


def test_tax_unit_demographics_refuse_two_heads() -> None:
    with pytest.raises(ValueError, match="more than one head"):
        puf_support._tax_unit_demographics(
            [10],
            [10, 10],
            age=[40, 41],
            is_female=[True, False],
            role=["HEAD", "HEAD"],
            preserve_nulls=False,
        )


def _donor_arrays() -> dict[str, list]:
    return {
        "tax_unit_id": [1, 2, 3],
        "household_weight": [1.0, 2.0, 1.0],
        "filing_status": [b"JOINT", b"SINGLE", b"HEAD_OF_HOUSEHOLD"],
        "person_tax_unit_id": [1, 1, 1, 2, 3, 3],
        "age": [52, 50, 17, 33, 38, 6],
        "is_male": [1, 0, 1, 0, 0, 1],
        "is_tax_unit_head": [1, 0, 0, 1, 1, 0],
        "is_tax_unit_spouse": [0, 1, 0, 0, 0, 0],
        "is_tax_unit_dependent": [0, 0, 1, 0, 0, 1],
        "employment_income": [900.0, 100.0, 0.0, 400.0, 0.0, 0.0],
        "self_employment_income": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
        "taxable_interest_income": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "qualified_dividend_income": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "non_qualified_dividend_income": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "short_term_capital_gains": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "long_term_capital_gains": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


_DONOR_OUTPUTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
)


def test_donor_carries_demographics_and_its_rank_in_the_puf_population() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        _donor_arrays(),
        person_outputs=_DONOR_OUTPUTS,
        tax_unit_outputs=(),
    )
    assert donor["puf_predictor_head_age"].tolist() == [52.0, 33.0, 38.0]
    assert donor["puf_predictor_spouse_age"].tolist() == [50.0, 0.0, 0.0]
    assert donor["puf_predictor_head_is_female"].tolist() == [0.0, 1.0, 1.0]
    assert donor["puf_predictor_dependent_count"].tolist() == [1.0, 0.0, 1.0]
    # Totals 1000, 400, 50 with weights 1, 2, 1 out of 4.
    np.testing.assert_allclose(donor[_RANK], [0.125, 0.5, 0.875])
    # Units 1 and 2 have wages, unit 3 self-employment income only.
    assert donor[_EARNS].tolist() == [1.0, 1.0, 1.0]
    for column in ("puf_predictor_employment_income", "employment_income"):
        assert column not in donor


def test_donor_without_person_roles_leaves_demographic_predictors_absent() -> None:
    arrays = _donor_arrays()
    del arrays["is_tax_unit_head"]
    donor = puf_tax_unit_donor_from_arrays(
        arrays,
        person_outputs=_DONOR_OUTPUTS,
        tax_unit_outputs=(),
    )
    assert "puf_predictor_head_age" not in donor
    # Absent predictors fail the imputation's donor-column check by name.
    with pytest.raises(ValueError, match="puf_predictor_head_age"):
        impute_us_puf_tax_detail_support(
            _recipient_frame([100.0], topcode=None),
            donor,
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=2,
            seed=0,
        )


def test_donor_person_with_two_roles_is_refused() -> None:
    arrays = _donor_arrays()
    arrays["is_tax_unit_dependent"] = [1, 0, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="more than one tax-unit role"):
        puf_tax_unit_donor_from_arrays(
            arrays,
            person_outputs=_DONOR_OUTPUTS,
            tax_unit_outputs=(),
        )


def _recipient_frame(
    wages: list[float],
    *,
    topcode: float | None,
    roles: list[str] | None = None,
) -> Frame:
    """Single-person survey tax units, cloned into the PUF-detail channel."""

    n = len(wages)
    ids = np.arange(1, n + 1, dtype="int64")
    survey_wages = np.asarray(wages, dtype=np.float64)
    if topcode is not None:
        survey_wages = np.minimum(survey_wages, topcode)
    zeros = np.zeros(n, dtype=np.float64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "age": np.full(n, 45, dtype="int64"),
            "is_female": np.zeros(n, dtype=bool),
            "tax_unit_role_input": roles if roles is not None else ["HEAD"] * n,
            "employment_income_before_lsr": survey_wages,
            "self_employment_income_before_lsr": zeros,
            "taxable_interest_income": zeros,
            "qualified_dividend_income": zeros,
            "non_qualified_dividend_income": zeros,
            "short_term_capital_gains": zeros,
            "long_term_capital_gains_before_response": zeros,
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.full(n, 8, dtype="int64")}
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": ids, "filing_status_input": ["SINGLE"] * n}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return clone_us_frame_for_puf_support(
        Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(np.ones(n), WeightKind.DESIGN)},
        )
    )


def test_recipient_rank_covers_only_the_puf_detail_rows() -> None:
    frame = _recipient_frame([300.0, 100.0, 200.0, 100.0], topcode=None)
    features = puf_support._tax_unit_feature_frame(
        frame, PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    )
    tax_unit = frame.table("tax_unit")
    detail = tax_unit[support_channel_column("tax_unit")].eq(
        PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    rank = features[_RANK]
    assert rank[~detail.to_numpy()].isna().all()
    # Survey totals 300, 100, 200, 100 with equal weight: the two 100s tie.
    detail_rank = pd.Series(
        rank[detail.to_numpy()].to_numpy(),
        index=tax_unit.loc[detail, support_source_id_column("tax_unit")].to_numpy(),
    ).sort_index()
    np.testing.assert_allclose(detail_rank, [0.125, 0.75, 0.375, 0.75])
    assert features.loc[detail, "puf_predictor_head_age"].eq(45.0).all()
    assert features.loc[detail, _EARNS].eq(1.0).all()
    assert features.loc[detail, "puf_predictor_spouse_age"].eq(0.0).all()


def test_recipient_demographics_require_constructed_tax_unit_roles() -> None:
    frame = _recipient_frame([100.0, 200.0], topcode=None)
    frame.table("person").drop(columns="tax_unit_role_input", inplace=True)
    with pytest.raises(ValueError, match="tax_unit_role_input"):
        puf_support._tax_unit_feature_frame(frame, PUF_TAX_DETAIL_DEFAULT_PREDICTORS)


def test_strict_recipient_surface_names_a_tax_unit_without_a_head() -> None:
    frame = _recipient_frame(
        [100.0, 200.0, 300.0],
        topcode=None,
        roles=["HEAD", "DEPENDENT", "HEAD"],
    )
    donor = puf_tax_unit_donor_from_arrays(
        _donor_arrays(),
        person_outputs=_DONOR_OUTPUTS,
        tax_unit_outputs=(),
    )
    # The rank's earnings sources fall under the stacked-spine ACS universe
    # rules, which need raw PUMS columns this frame does not carry; the
    # demographic predictors alone isolate the role check.
    demographic_predictors = tuple(
        predictor
        for predictor in PUF_TAX_DETAIL_DEFAULT_PREDICTORS
        if predictor not in (_RANK, _EARNS)
    )
    with pytest.raises(ValueError, match="puf_predictor_head_age"):
        impute_us_puf_tax_detail_support(
            frame,
            donor,
            predictors=demographic_predictors,
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=2,
            seed=0,
            require_complete_recipient_predictors=True,
            absent_cells=puf_support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
        )


def _heavy_tail_donor() -> pd.DataFrame:
    """300 PUF units with wages spread geometrically from $5k to $20M."""

    n = 300
    wages = 5_000.0 * np.power(4_000.0, np.arange(n) / (n - 1))
    zeros = np.zeros(n)
    return pd.DataFrame(
        {
            "filing_status_code": np.ones(n),
            "tax_unit_person_count": np.ones(n),
            "head_age": np.full(n, 45.0),
            "spouse_age": zeros,
            "head_is_female": zeros,
            "dependent_count": zeros,
            "employment_income_before_lsr": wages,
            "self_employment_income_before_lsr": zeros,
            "taxable_interest_income": zeros,
            "qualified_dividend_income": zeros,
            "non_qualified_dividend_income": zeros,
            "short_term_capital_gains": zeros,
            "long_term_capital_gains_before_response": zeros,
            "weight": np.ones(n),
        }
    )


def _imputed_puf_wages(
    frame: Frame,
    predictors: tuple[str, ...],
    *,
    donor: pd.DataFrame | None = None,
) -> pd.Series:
    imputed = impute_us_puf_tax_detail_support(
        frame,
        _heavy_tail_donor() if donor is None else donor,
        predictors=predictors,
        person_outputs=("employment_income_before_lsr",),
        tax_unit_outputs=(),
        n_estimators=8,
        seed=0,
    )
    person = imputed.table("person")
    detail = person[support_channel_column("person")].eq(PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    # Each survey person is its own tax unit, so the source person id names
    # the survey unit behind every PUF-detail clone.
    return (
        person.loc[detail]
        .set_index(support_source_id_column("person"))["employment_income_before_lsr"]
        .sort_index()
    )


def test_topcoded_survey_units_receive_puf_income_above_the_topcode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    # 30 survey units; the top five report above a $300k topcode, which the
    # survey records as a five-way tie at $300k.
    true_wages = [10_000.0 * (i + 1) for i in range(25)] + [2e6, 3e6, 5e6, 8e6, 1.2e7]
    frame = _recipient_frame(true_wages, topcode=300_000.0)
    topcoded = pd.Index(range(26, 31))

    # The fixture discriminates: the old self-predicting design hands the
    # topcoded units wages near their $300k survey value.
    self_predicting = (
        "puf_predictor_filing_status_code",
        "puf_predictor_tax_unit_person_count",
        "puf_predictor_employment_income",
    )
    old = _imputed_puf_wages(frame, self_predicting)
    assert old.loc[topcoded].max() < 1_000_000.0

    new = _imputed_puf_wages(frame, PUF_TAX_DETAIL_DEFAULT_PREDICTORS)
    assert new.loc[topcoded].min() > 1_000_000.0
    # Ranking keeps the imputation coherent with the survey ordering.
    survey = frame.table("person").drop_duplicates(support_source_id_column("person"))
    survey_wages = survey.set_index(support_source_id_column("person"))[
        "employment_income_before_lsr"
    ].sort_index()
    assert new.corr(survey_wages, method="spearman") > 0.8


def _mixed_participation_donor() -> pd.DataFrame:
    """300 PUF units: workers with wages, non-workers with equal interest.

    Workers and non-workers interleave across the whole income range, as they
    do in the PUF, so income rank alone cannot tell them apart.
    """

    donor = _heavy_tail_donor()
    works = np.arange(len(donor)) % 2 == 0
    income = donor["employment_income_before_lsr"].to_numpy()
    donor["employment_income_before_lsr"] = np.where(works, income, 0.0)
    donor["taxable_interest_income"] = np.where(works, 0.0, income)
    return donor


def test_survey_non_workers_receive_no_puf_wages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    # Ten survey retirees live on interest; twenty workers on wages. Their
    # income totals interleave, as the donors' do.
    levels = [8_000.0 * (i + 1) for i in range(30)]
    frame = _recipient_frame(
        [0.0 if i % 3 == 0 else level for i, level in enumerate(levels)],
        topcode=None,
    )
    person = frame.table("person")
    retiree_rows = person["employment_income_before_lsr"].eq(0.0)
    source_ids = person[support_source_id_column("person")]
    person.loc[retiree_rows, "taxable_interest_income"] = source_ids[retiree_rows].map(
        lambda source_id: levels[source_id - 1]
    )
    retirees = pd.Index(
        person.loc[retiree_rows, support_source_id_column("person")].unique()
    )
    donor = _mixed_participation_donor()

    # The fixture discriminates: without the participation flag, retirees
    # draw from the rank-matched donors, half of whom work.
    no_flag = tuple(
        predictor
        for predictor in PUF_TAX_DETAIL_DEFAULT_PREDICTORS
        if predictor != _EARNS
    )
    without_flag = _imputed_puf_wages(frame, no_flag, donor=donor)
    assert without_flag.loc[retirees].gt(0.0).any()

    wages = _imputed_puf_wages(frame, PUF_TAX_DETAIL_DEFAULT_PREDICTORS, donor=donor)
    assert wages.loc[retirees].eq(0.0).all()
    assert wages.drop(retirees).gt(0.0).all()
