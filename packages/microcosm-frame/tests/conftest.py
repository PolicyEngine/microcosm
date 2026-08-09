"""Shared fixtures: a simple two-household bundle and synthetic CPS households.

The CPS fixtures construct pointer semantics realistically: ``A_SPOUSE``,
``PEPAR1``, and ``PEPAR2`` are line numbers (``A_LINENO`` values) within the
household, ``0`` meaning "none"; ``A_EXPRRP`` uses the Census relationship
recode.
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

# Census A_EXPRRP relationship-to-reference-person recode values, spelled out
# so the fixtures document the exact codes they exercise.
REF_PERSON_WITH_RELATIVES = 1
REF_PERSON_WITHOUT_RELATIVES = 2
WIFE = 4
OWN_CHILD = 5
GRANDCHILD = 7

# CPS marital-status (A_MARITL) codes.
MARRIED_SPOUSE_PRESENT = 1
WIDOWED = 4
NEVER_MARRIED = 7


@pytest.fixture
def simple_schema() -> EntitySchema:
    """A person + household schema."""
    return EntitySchema(group_entities=("household",))


@pytest.fixture
def make_bundle(simple_schema):
    """Factory for a five-person, two-household bundle.

    Persons 0-1 live in household 1 (weight 100), persons 2-4 in household 2
    (weight 200). Incomes are hand-computable.
    """

    def _make(
        *,
        weight_values: tuple[float, ...] = (100.0, 200.0),
        kind: WeightKind = WeightKind.DESIGN,
        strata: pd.Series | None = None,
        person_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
        household_ids: tuple[int, ...] = (1, 2),
    ) -> Frame:
        person = pd.DataFrame(
            {
                "person_id": np.asarray(person_ids, dtype="int64"),
                "person_household_id": np.asarray(
                    [
                        household_ids[0],
                        household_ids[0],
                        household_ids[1],
                        household_ids[1],
                        household_ids[1],
                    ],
                    dtype="int64",
                ),
                "age": [40, 8, 30, 28, 1],
                "income": [50000.0, 0.0, 30000.0, 20000.0, 0.0],
            }
        )
        household = pd.DataFrame(
            {
                "household_id": np.asarray(household_ids, dtype="int64"),
                "state": ["CA", "NY"],
            }
        )
        weights = Weights(values=np.asarray(weight_values), kind=kind)
        return Frame(
            {"person": person, "household": household},
            simple_schema,
            {"household": weights},
            strata=strata,
        )

    return _make


def _person_row(
    *,
    ph_seq: int,
    line: int,
    age: int,
    marital: int,
    spouse: int = 0,
    parent1: int = 0,
    parent2: int = 0,
    relationship: int,
    spm_id: int | None = None,
    family: int = 1,
    wage: float = 0.0,
) -> dict:
    """Build one CPS-like person record with both raw and harmonized columns."""
    row = {
        "PH_SEQ": ph_seq,
        "A_LINENO": line,
        "A_AGE": age,
        "A_MARITL": marital,
        "A_SPOUSE": spouse,
        "PEPAR1": parent1,
        "PEPAR2": parent2,
        "A_EXPRRP": relationship,
        "PF_SEQ": family,
        "wage_income": wage,
        "household_id": ph_seq,
        "age": age,
    }
    if spm_id is not None:
        row["SPM_ID"] = spm_id
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Stack person records and attach a dense person_id and a shifted index."""
    frame = pd.DataFrame(rows)
    frame["person_id"] = np.arange(len(frame), dtype="int64")
    # A non-trivial (non-range) index catches index-alignment bugs.
    frame.index = pd.RangeIndex(start=100, stop=100 + len(frame))
    return frame


def _married_with_kids() -> pd.DataFrame:
    """A married couple (lines 1,2) with two own children (lines 3,4)."""
    return _frame(
        [
            _person_row(
                ph_seq=10,
                line=1,
                age=42,
                marital=MARRIED_SPOUSE_PRESENT,
                spouse=2,
                relationship=REF_PERSON_WITH_RELATIVES,
                spm_id=500,
                wage=70000,
            ),
            _person_row(
                ph_seq=10,
                line=2,
                age=40,
                marital=MARRIED_SPOUSE_PRESENT,
                spouse=1,
                relationship=WIFE,
                spm_id=500,
                wage=30000,
            ),
            _person_row(
                ph_seq=10,
                line=3,
                age=12,
                marital=NEVER_MARRIED,
                parent1=1,
                parent2=2,
                relationship=OWN_CHILD,
                spm_id=500,
            ),
            _person_row(
                ph_seq=10,
                line=4,
                age=9,
                marital=NEVER_MARRIED,
                parent1=1,
                parent2=2,
                relationship=OWN_CHILD,
                spm_id=500,
            ),
        ]
    )


def _single_filer() -> pd.DataFrame:
    """A lone adult household (line 1)."""
    return _frame(
        [
            _person_row(
                ph_seq=20,
                line=1,
                age=33,
                marital=NEVER_MARRIED,
                relationship=REF_PERSON_WITHOUT_RELATIVES,
                spm_id=700,
                wage=48000,
            ),
        ]
    )


def _multigenerational() -> pd.DataFrame:
    """Grandparent (1), independent adult child (2), grandchild (3).

    The adult child earns well above the dependent gross-income limit, so the
    tax rules must put them in their own tax unit (as HEAD), claiming the
    grandchild — not as a dependent of the grandparent. The two families are
    distinguished by PF_SEQ within the single household.
    """
    return _frame(
        [
            _person_row(
                ph_seq=30,
                line=1,
                age=66,
                marital=WIDOWED,
                relationship=REF_PERSON_WITH_RELATIVES,
                spm_id=900,
                family=1,
                wage=20000,
            ),
            _person_row(
                ph_seq=30,
                line=2,
                age=27,
                marital=NEVER_MARRIED,
                parent1=1,
                relationship=OWN_CHILD,
                spm_id=900,
                family=2,
                wage=55000,
            ),
            _person_row(
                ph_seq=30,
                line=3,
                age=3,
                marital=NEVER_MARRIED,
                parent1=2,
                relationship=GRANDCHILD,
                spm_id=900,
                family=2,
            ),
        ]
    )


@pytest.fixture
def married_with_kids() -> pd.DataFrame:
    return _married_with_kids()


@pytest.fixture
def single_filer() -> pd.DataFrame:
    return _single_filer()


@pytest.fixture
def multigenerational() -> pd.DataFrame:
    return _multigenerational()


@pytest.fixture
def three_household_frame() -> pd.DataFrame:
    """All three synthetic households stacked with dense person ids."""
    frame = pd.concat(
        [_married_with_kids(), _single_filer(), _multigenerational()],
        ignore_index=True,
    )
    frame["person_id"] = np.arange(len(frame), dtype="int64")
    return frame


def household_weights_for(frame: pd.DataFrame, base: float = 1000.0) -> Weights:
    """Design weights, one per sorted distinct household id in ``frame``."""
    n = frame["household_id"].nunique()
    return Weights(
        values=base + 100.0 * np.arange(n, dtype="float64"),
        kind=WeightKind.DESIGN,
    )


@pytest.fixture
def make_household_weights():
    """Factory fixture for per-household design weights."""
    return household_weights_for
