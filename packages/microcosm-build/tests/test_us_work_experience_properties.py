"""Property and differential tests for the #719 work-experience columns.

Invariants, for every input the three producers accept:

* **Nesting.** ``WEMIND == WEIND_TO_WEMIND[WEIND]``: the major industry group
  is a function of the detailed one.
* **Forward worker identity.** ``WKSWORK > 0`` implies ``WEIND`` in 1--22
  (someone who worked last year has a worker industry code). Holds on every
  spine.
* **Converse worker identity (ASEC only).** ``WEIND`` in 1--22 implies
  ``WKSWORK > 0``. ACS ``INDP`` covers a job held in the past five years, so
  the release carry deliberately does not check it (intended asymmetry).
* **Round trip.** The release carry emits the inputs unchanged as
  ``detailed_industry_recode``/``major_industry_recode`` and
  ``worked_last_year == (WKSWORK > 0)``.
* **ACS universe.** Below age 16 every ACS output is 0; a blank ``INDP`` from
  age 16 is the did-not-work group 23; an observed ``INDP`` maps through
  ``ACS_INDP_TO_WEIND``; ``WKSWORK`` is ``WKWN`` or 0.

The differential tests run the same generated rows through every site that
enforces an identity and require the sites to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.acs_release_predictors as acs
from microcosm.build.us_runtime.asec_census_person_columns import (
    WEIND_TO_WEMIND,
    AsecCensusPersonColumnsError,
    _check_work_experience_universe,
)
from microcosm.build.us_runtime.org_wages import derive_us_org_occupation_inputs

_WORKER = range(1, 23)
_SETTINGS = settings(max_examples=200, deadline=None)


def _carry_frame(weind, wemind, wkswork) -> pd.DataFrame:
    rows = len(weind)
    return pd.DataFrame(
        {
            "PRDTRACE": np.ones(rows, dtype=np.int64),
            "PRDTHSP": np.zeros(rows, dtype=np.int64),
            "POCCU2": np.full(rows, 8, dtype=np.int64),
            "WEIND": np.asarray(weind, dtype=np.int64),
            "WEMIND": np.asarray(wemind, dtype=np.int64),
            "WKSWORK": np.asarray(wkswork, dtype=np.int64),
        }
    )


@st.composite
def _asec_rows(draw) -> tuple[list[int], list[int], list[int]]:
    """Rows that satisfy both worker identities and the nesting."""

    weind = draw(st.lists(st.integers(0, 23), min_size=1, max_size=40))
    weeks = [draw(st.integers(1, 52)) if code in _WORKER else 0 for code in weind]
    return weind, [WEIND_TO_WEMIND[code] for code in weind], weeks


@st.composite
def _arbitrary_rows(draw) -> tuple[list[int], list[int], list[int]]:
    """In-range codes with no identity promised."""

    rows = draw(st.integers(1, 40))
    return (
        draw(st.lists(st.integers(0, 23), min_size=rows, max_size=rows)),
        draw(st.lists(st.integers(0, 15), min_size=rows, max_size=rows)),
        draw(st.lists(st.integers(0, 52), min_size=rows, max_size=rows)),
    )


def _nested(weind, wemind) -> bool:
    return all(WEIND_TO_WEMIND[d] == m for d, m in zip(weind, wemind, strict=True))


def _forward(weind, wkswork) -> bool:
    return all(d in _WORKER for d, w in zip(weind, wkswork, strict=True) if w > 0)


def _converse(weind, wkswork) -> bool:
    return all(w > 0 for d, w in zip(weind, wkswork, strict=True) if d in _WORKER)


def _carry_accepts(weind, wemind, wkswork) -> bool:
    try:
        derive_us_org_occupation_inputs(_carry_frame(weind, wemind, wkswork))
    except ValueError:
        return False
    return True


def _restore_accepts(weind, wemind, wkswork) -> bool:
    try:
        _check_work_experience_universe(
            _carry_frame(weind, wemind, wkswork), "generated vintage"
        )
    except AsecCensusPersonColumnsError:
        return False
    return True


@_SETTINGS
@given(_asec_rows())
def test_release_carry_round_trips_coherent_rows(rows) -> None:
    weind, wemind, wkswork = rows
    out = derive_us_org_occupation_inputs(_carry_frame(weind, wemind, wkswork))
    assert out["detailed_industry_recode"].tolist() == weind
    assert out["major_industry_recode"].tolist() == wemind
    assert out["worked_last_year"].tolist() == [w > 0 for w in wkswork]
    assert out["detailed_industry_recode"].dtype == np.int16
    assert out["worked_last_year"].dtype == bool


@_SETTINGS
@given(_arbitrary_rows())
def test_restore_and_carry_accept_exactly_their_stated_identities(rows) -> None:
    """Differential: each site accepts iff its stated identities hold."""

    weind, wemind, wkswork = rows
    nested = _nested(weind, wemind)
    forward = _forward(weind, wkswork)
    converse = _converse(weind, wkswork)
    assert _restore_accepts(weind, wemind, wkswork) == (nested and forward and converse)
    assert _carry_accepts(weind, wemind, wkswork) == (nested and forward)
    # Intended asymmetry: the ASEC-only restoration is the stricter site.
    if _restore_accepts(weind, wemind, wkswork):
        assert _carry_accepts(weind, wemind, wkswork)


def _acs_template() -> dict[str, object]:
    return {
        "SERIALNO": "2024HU0000001",
        "SPORDER": 1,
        "DEAR": 2,
        "DEYE": 2,
        "DREM": 2,
        "DPHY": 2,
        "DDRS": 2,
        "DOUT": 2,
        "RAC1P": 1,
        "HISP": 1,
        "SSIP": 0.0,
        "ADJINC": 1_000_000,
    }


_OBSERVED_INDP = sorted(code for code in acs.ACS_INDP_TO_WEIND if code != 9920)


@st.composite
def _acs_person(draw) -> dict[str, object]:
    """One ACS person inside the source universes the join admits."""

    row = _acs_template()
    age = draw(st.integers(0, 95))
    row["AGEP"] = age
    if age < 16:
        for column in ("DREM", "DPHY", "DDRS") if age < 5 else ():
            row[column] = np.nan
        if age < 15:
            row["DOUT"] = np.nan
        row.update(OCCP=np.nan, INDP=np.nan, WKWN=np.nan, ESR=np.nan)
        return row
    kind = draw(st.sampled_from(["worked", "past_five_years", "never", "unemployed"]))
    if kind == "worked":
        row.update(
            INDP=draw(st.sampled_from(_OBSERVED_INDP)),
            OCCP=1005,
            ESR=draw(st.sampled_from([1, 2, 4, 5])),
            WKWN=draw(st.integers(1, 52)),
        )
    elif kind == "past_five_years":
        row.update(
            INDP=draw(st.sampled_from(_OBSERVED_INDP)), OCCP=1005, ESR=6, WKWN=np.nan
        )
    elif kind == "never":
        row.update(INDP=np.nan, OCCP=np.nan, ESR=6, WKWN=np.nan)
    else:
        row.update(INDP=9920, OCCP=9920, ESR=3, WKWN=np.nan)
    return row


@_SETTINGS
@given(st.lists(_acs_person(), min_size=1, max_size=30))
def test_acs_crosswalk_satisfies_every_stated_invariant(people) -> None:
    joined = pd.DataFrame(people)
    joined["person_source_id"] = np.arange(len(joined))
    mapped = acs._crosswalk_people(joined)

    weind = mapped["WEIND"].astype(int).tolist()
    wemind = mapped["WEMIND"].astype(int).tolist()
    wkswork = mapped["WKSWORK"].astype(int).tolist()
    assert _nested(weind, wemind)
    assert _forward(weind, wkswork)
    for person, detailed, weeks in zip(people, weind, wkswork, strict=True):
        if person["AGEP"] < 16:
            assert (detailed, weeks) == (0, 0)
        elif pd.isna(person["INDP"]):
            assert detailed == 23
        else:
            assert detailed == acs.ACS_INDP_TO_WEIND[int(person["INDP"])]
        expected_weeks = 0 if pd.isna(person["WKWN"]) else int(person["WKWN"])
        assert weeks == expected_weeks
    # The join's rows are admissible to the release carry that consumes them.
    assert _carry_accepts(weind, wemind, wkswork)


def test_every_crosswalk_code_nests_into_its_major_group() -> None:
    """Differential against the pinned nesting for all 267 table codes."""

    for code, detailed in acs.ACS_INDP_TO_WEIND.items():
        assert detailed in _WORKER or detailed == 23, code
        assert WEIND_TO_WEMIND[detailed] in range(1, 16), code


@pytest.mark.parametrize("weeks", [1, 52])
def test_armed_forces_and_did_not_work_codes_bracket_the_worker_range(weeks) -> None:
    assert _carry_accepts([22], [14], [weeks])
    assert not _carry_accepts([23], [15], [weeks])
    assert _carry_accepts([23], [15], [0])
