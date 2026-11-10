"""Property and differential tests for the prior-year wage the stage keeps.

Invariants, for every pooled ASEC person table the stage accepts:

* **Kept, not recomputed.** After the support-clone pass the frame holds
  ``cps_employment_income_last_year`` and no ``employment_income_last_year``.
  On every ASEC row the auxiliary wage equals the pre-clone derivation's wage;
  on every PUF-support row it equals the joint QRF's wage prediction.
* **Bounds.** The auxiliary wage is finite and nonnegative on every row.
* **Oracle agreement (differential).** An independent dictionary join of each
  person to the previous ASEC year's record agrees with the stage on every
  ASEC row. Where that record exists, its two allocation flags read zero and
  neither amount is a -1 or -9999 sentinel, the wage is the record's
  ``WSAL_VAL`` and ``previous_year_income_available`` is true. Otherwise
  availability is false and the wage is the current ``WSAL_VAL`` (0 for a
  sentinel) when the pool holds the previous year, or 0 when it does not.
* **Reconciliation.** The source reconciliation gate passes on the stage's
  own output and counts exactly one mismatch when one ASEC wage moves.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

from hypothesis import example, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_prior_year_income import *

_SENTINELS = (-1, -9999)
_SETTINGS = settings(max_examples=100, deadline=None, database=None)


def _amount(draw, low: int) -> int:
    """A dollar amount, or a Census sentinel one draw in four."""

    if draw(st.integers(0, 3)) == 0:
        return draw(st.sampled_from(_SENTINELS))
    return draw(st.integers(low, 250_000))


@st.composite
def _pooled_asec(draw) -> pd.DataFrame:
    """Distinct (year, PERIDNUM) rows over one to three pooled ASEC years.

    Half the pools hold consecutive years and half any years from 2021 to
    2025, so most draws join people across adjacent years and many leave a
    year without its predecessor. Each person appears in each year by a coin
    flip, so the same PERIDNUM recurs across years.
    """

    if draw(st.booleans()):
        first = draw(st.integers(2021, 2023))
        years = list(range(first, first + draw(st.integers(1, 3))))
    else:
        years = sorted(draw(st.sets(st.integers(2021, 2025), min_size=1, max_size=3)))
    people = [f"P{index}" for index in range(5)]
    rows = []
    for year in years:
        present = [person for person in people if draw(st.booleans())] or people[:1]
        for person in present:
            rows.append(
                {
                    "source_year": year,
                    "PERIDNUM": person,
                    "WSAL_VAL": _amount(draw, 0),
                    "SEMP_VAL": _amount(draw, -50_000),
                    "I_ERNVAL": draw(st.sampled_from([0, 0, 1, 2])),
                    "I_SEVAL": draw(st.sampled_from([0, 1])),
                }
            )
    return pd.DataFrame(rows)


# One pool that reaches every branch of the join on every run. Each 2022 row
# has a 2021 record: P0's is usable; P1's carries a self-employment allocation
# flag, P2's a wage sentinel, P3's a self-employment sentinel and P4's a wage
# allocation flag, so those four fall back to the current wage (0 for P3,
# whose current wage is a sentinel). The 2021 rows and P0's 2024 row have no
# previous year in the pool and default to 0.
_EVERY_BRANCH = pd.DataFrame(
    {
        "source_year": [
            2021,
            2022,
            2021,
            2022,
            2021,
            2022,
            2021,
            2022,
            2021,
            2022,
            2024,
        ],
        "PERIDNUM": ["P0", "P0", "P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P0"],
        "WSAL_VAL": [100, 200, 300, 400, -1, 500, 50, -1, 70, 80, 900],
        "SEMP_VAL": [10, 20, 0, 0, 0, 0, -9999, 0, 0, 0, 0],
        "I_ERNVAL": [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        "I_SEVAL": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)
_EVERY_BRANCH_EXPECTED = (
    [0.0, 100.0, 0.0, 400.0, 0.0, 500.0, 0.0, 0.0, 0.0, 80.0, 0.0],
    [False, True, False, False, False, False, False, False, False, False, False],
)


def _oracle(rows: pd.DataFrame) -> tuple[list[float], list[bool]]:
    """Join each row to its previous-year record without the stage's code."""

    records = {
        (int(row.source_year), str(row.PERIDNUM)): row for row in rows.itertuples()
    }
    years = {int(year) for year in rows["source_year"]}
    wages: list[float] = []
    available: list[bool] = []
    for row in rows.itertuples():
        prior = records.get((int(row.source_year) - 1, str(row.PERIDNUM)))
        usable = (
            prior is not None
            and prior.I_ERNVAL == 0
            and prior.I_SEVAL == 0
            and prior.WSAL_VAL not in _SENTINELS
            and prior.SEMP_VAL not in _SENTINELS
        )
        if usable:
            wages.append(float(prior.WSAL_VAL))
        elif int(row.source_year) - 1 in years and row.WSAL_VAL not in _SENTINELS:
            wages.append(float(row.WSAL_VAL))
        else:
            wages.append(0.0)
        available.append(bool(usable))
    return wages, available


class _ArangeFitted:
    """Predict 1,000 plus the row's position, so each PUF row's value shows
    which prediction it received."""

    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        positions = np.arange(len(test), dtype=np.float64)
        return pd.DataFrame(
            {
                "employment_income_last_year": positions + 1_000.0,
                "self_employment_income_last_year": 50.0 - positions,
            },
            index=test.index,
        )


class _ArangeQRF:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self, training, predictors, targets, *, weights) -> _ArangeFitted:
        return _ArangeFitted()


def _run_stage(rows: pd.DataFrame) -> tuple[Frame, Frame]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "QRF", _ArangeQRF)
        direct = with_us_prior_year_income_inputs(
            _frame(rows), seed=3, time_period=2024
        )
        result = with_us_prior_year_income_inputs(
            clone_us_frame_for_puf_support(direct), seed=3, time_period=2024
        )
    return direct, result


def test_the_oracle_reads_every_branch_of_the_join() -> None:
    assert _oracle(_EVERY_BRANCH) == _EVERY_BRANCH_EXPECTED


@_SETTINGS
@given(_pooled_asec())
@example(_EVERY_BRANCH)
def test_the_support_clone_pass_keeps_the_wage_it_derived(rows) -> None:
    direct, result = _run_stage(rows)
    person = result.table("person")
    channel = person["person_support_channel"].astype(str)
    asec = channel == BASE_ASEC_SUPPORT_CHANNEL
    puf = channel == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    wages = person[US_PRIOR_YEAR_EMPLOYMENT_INCOME_COLUMN].to_numpy(dtype=np.float64)

    assert "employment_income_last_year" not in person
    assert int(asec.sum()) == len(rows)
    assert int(puf.sum()) == len(rows)
    assert np.isfinite(wages).all()
    assert (wages >= 0).all()
    np.testing.assert_array_equal(
        wages[asec.to_numpy()],
        direct.table("person")["employment_income_last_year"].to_numpy(
            dtype=np.float64
        ),
    )
    np.testing.assert_array_equal(
        wages[puf.to_numpy()], 1_000.0 + np.arange(len(rows), dtype=np.float64)
    )


@_SETTINGS
@given(_pooled_asec())
@example(_EVERY_BRANCH)
def test_the_asec_wage_agrees_with_an_independent_join(rows) -> None:
    _direct, result = _run_stage(rows)
    person = result.table("person")
    asec = person["person_support_channel"].astype(str) == BASE_ASEC_SUPPORT_CHANNEL
    expected_wages, expected_available = _oracle(rows)

    assert (
        person.loc[asec, US_PRIOR_YEAR_EMPLOYMENT_INCOME_COLUMN].tolist()
        == expected_wages
    )
    assert (
        person.loc[asec, "previous_year_income_available"].astype(bool).tolist()
        == expected_available
    )


@_SETTINGS
@given(_pooled_asec(), st.data())
def test_reconciliation_counts_exactly_the_moved_asec_wage(rows, data) -> None:
    _direct, result = _run_stage(rows)
    assert us_prior_year_income_source_reconciliation_gate(result).passed

    person = result.table("person").copy()
    asec_rows = person.index[
        person["person_support_channel"].astype(str) == BASE_ASEC_SUPPORT_CHANNEL
    ]
    moved = data.draw(st.sampled_from(list(asec_rows)))
    person.loc[moved, US_PRIOR_YEAR_EMPLOYMENT_INCOME_COLUMN] += data.draw(
        st.sampled_from([0.5, 1.0, 1_000.0])
    )
    gate = us_prior_year_income_source_reconciliation_gate(
        module._replace_person_table(result, person)
    )
    assert not gate.passed
    assert gate.details["mismatch_counts"][US_PRIOR_YEAR_EMPLOYMENT_INCOME_COLUMN] == 1
