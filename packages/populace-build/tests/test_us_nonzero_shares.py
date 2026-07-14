"""Per-record nonzero shares: the evidence the parity gate runs on."""

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.nonzero_shares import nonzero_share, us_nonzero_shares
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _frame(**person_extra: object) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 20], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 200], dtype="int64"),
            "person_spm_unit_id": np.asarray([1000, 2000, 2000], dtype="int64"),
            "person_family_id": np.asarray([10000, 20000, 20000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [100000, 200000, 200001], dtype="int64"
            ),
            "student_loan_interest": [100.0, 50.0, 0.0],
            "has_esi": [True, False, True],
            "ssn_card_type": ["CITIZEN", "CITIZEN", "NONE"],
            **person_extra,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([10, 20], dtype="int64")}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([100, 200], dtype="int64"),
                    "health_savings_account_ald": [500.0, 0.0],
                }
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([1000, 2000], dtype="int64")}
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([10000, 20000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100000, 200000, 200001], dtype="int64")}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([10.0, 30.0]), WeightKind.CALIBRATED)},
    )


class TestNonzeroShare:
    def test_numeric_share_counts_nonzero_records(self) -> None:
        # 2 of 3 records are non-zero.
        assert nonzero_share(pd.Series([100.0, 50.0, 0.0])) == pytest.approx(2 / 3)

    def test_boolean_share_is_true_fraction(self) -> None:
        assert nonzero_share(pd.Series([True, False, True])) == pytest.approx(2 / 3)

    def test_string_share_is_nonempty_fraction(self) -> None:
        assert nonzero_share(pd.Series(["A", "", "B"])) == pytest.approx(2 / 3)

    def test_nan_counts_as_zero(self) -> None:
        # NaN is not signal: only the one finite non-zero value counts.
        assert nonzero_share(pd.Series([np.nan, 0.0, 7.0])) == pytest.approx(1 / 3)


class TestUsNonzeroShares:
    def test_share_is_unweighted_record_share_per_entity(self) -> None:
        shares = us_nonzero_shares(_frame())

        # Person column: 2 of 3 person records non-zero (weights ignored).
        assert shares["student_loan_interest"] == pytest.approx(2 / 3)
        # Boolean: 2 of 3 True.
        assert shares["has_esi"] == pytest.approx(2 / 3)
        # Tax-unit column: 1 of 2 tax-unit records non-zero.
        assert shares["health_savings_account_ald"] == pytest.approx(0.5)

    def test_structural_columns_are_skipped(self) -> None:
        shares = us_nonzero_shares(_frame())

        assert "person_id" not in shares
        assert "person_household_id" not in shares
        assert "household_id" not in shares
        assert "tax_unit_id" not in shares

    def test_string_columns_are_measured(self) -> None:
        # Non-structural string columns are policy layers (e.g. ssn_card_type):
        # all three records carry a non-empty value.
        shares = us_nonzero_shares(_frame())
        assert shares["ssn_card_type"] == pytest.approx(1.0)

    def test_column_restriction_filters_raw_source_columns(self) -> None:
        shares = us_nonzero_shares(
            _frame(WSAL_VAL=[1.0, 2.0, 3.0]),
            columns=["student_loan_interest"],
        )

        assert set(shares) == {"student_loan_interest"}


class TestTypedWeightMirror:
    def test_household_weight_measures_typed_frame_weights(self) -> None:
        """The parity universe measures weights as the export persists them.

        Frame weights are typed pipeline state, not a table column; the H5
        adapter materializes them as ``household_weight`` and the coverage
        gate mirrors that. Build M's sparse run failed eCPS parity because
        this module read the deliberately absent table column as an all-zero
        layer while the reference's finished export carries it — the mirror
        keeps the two gates and the export telling one story.
        """

        shares = us_nonzero_shares(_frame(), columns=["household_weight"])
        assert shares["household_weight"] == 1.0

    def test_table_column_still_wins_when_present(self) -> None:
        frame = _frame()
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["household_weight"] = [0.0, 5.0]
        rebuilt = Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        )
        shares = us_nonzero_shares(rebuilt, columns=["household_weight"])
        assert shares["household_weight"] == 0.5
