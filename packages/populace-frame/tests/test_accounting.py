"""Weighted accounting: sums, means, quantiles, Gini, group sums."""

import numpy as np
import pandas as pd
import pytest

from populace.frame import (
    Frame,
    WeightKind,
    Weights,
    gini,
    groupby_wsum,
    wmean,
    wmedian,
    wquantile,
    wsum,
)


def _bundle(values, weights, schema_factory=None):
    """One household per person so person weights equal household weights."""
    from populace.frame import EntitySchema

    n = len(values)
    person = pd.DataFrame(
        {
            "person_id": range(n),
            "person_household_id": range(n),
            "x": np.asarray(values, dtype=np.float64),
        }
    )
    household = pd.DataFrame({"household_id": range(n)})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                values=np.asarray(weights, dtype=np.float64), kind=WeightKind.DESIGN
            )
        },
    )


class TestSumsAndMeans:
    def test_wmean_is_mass_weighted(self, make_bundle) -> None:
        bundle = make_bundle()
        # (100*(50000+0) + 200*(30000+20000+0)) / (2*100 + 3*200)
        assert wmean(bundle, "income") == pytest.approx(15_000_000.0 / 800.0)

    def test_boolean_columns_count_weighted_records(self, make_bundle) -> None:
        bundle = make_bundle()
        person = bundle.person.copy()
        person["is_adult"] = person["age"] >= 18
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            bundle.schema,
            {"household": bundle.weights_for("household")},
        )
        # Adults: persons 0 (w 100), 2 and 3 (w 200 each) = 500.
        assert wsum(rebuilt, "is_adult") == 500.0

    def test_group_level_column_uses_group_weights(self, make_bundle) -> None:
        bundle = make_bundle()
        household = bundle.table("household").copy()
        household["rent"] = [1000.0, 2000.0]
        rebuilt = Frame(
            {"person": bundle.person, "household": household},
            bundle.schema,
            {"household": bundle.weights_for("household")},
        )
        assert wsum(rebuilt, "rent") == 100.0 * 1000.0 + 200.0 * 2000.0

    def test_nan_propagates_instead_of_silently_dropping(self) -> None:
        bundle = _bundle([1.0, np.nan, 3.0], [1.0, 1.0, 1.0])
        assert np.isnan(wsum(bundle, "x"))
        assert np.isnan(wmean(bundle, "x"))

    def test_non_numeric_column_is_refused(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="non-numeric"):
            wsum(bundle, "state")

    def test_entity_mismatch_is_named(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="lives on entity 'person'"):
            wsum(bundle, "income", entity="household")

    def test_unknown_column_is_named(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="'ghost'"):
            wmean(bundle, "ghost")


class TestQuantiles:
    def test_scalar_quantile_is_inverse_cdf(self) -> None:
        bundle = _bundle([10.0, 20.0, 30.0], [1.0, 1.0, 2.0])
        # cumulative proportions: .25, .5, 1.0
        assert wquantile(bundle, "x", 0.5) == 20.0
        assert wquantile(bundle, "x", 0.51) == 30.0
        assert wquantile(bundle, "x", 1.0) == 30.0

    def test_zero_weight_rows_are_never_selected(self) -> None:
        bundle = _bundle([10.0, 20.0, 30.0], [0.0, 1.0, 1.0])
        assert wquantile(bundle, "x", 0.0) == 20.0

    def test_array_q_returns_series_indexed_by_q(self) -> None:
        bundle = _bundle([10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0])
        result = wquantile(bundle, "x", np.array([0.25, 0.75]))
        assert isinstance(result, pd.Series)
        assert result.loc[0.25] == 10.0
        assert result.loc[0.75] == 30.0

    def test_out_of_range_q_is_rejected(self) -> None:
        bundle = _bundle([1.0], [1.0])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            wquantile(bundle, "x", 1.5)

    def test_wmedian_weights_decide_the_middle(self) -> None:
        bundle = _bundle([10.0, 50.0], [9.0, 1.0])
        assert wmedian(bundle, "x") == 10.0


class TestGini:
    def test_known_two_point_distribution(self) -> None:
        # Two equally weighted people, one has everything: G = 0.5.
        bundle = _bundle([0.0, 100.0], [1.0, 1.0])
        assert gini(bundle, "x") == pytest.approx(0.5)

    def test_weights_change_the_index(self) -> None:
        # The same values, but the holder carries 1/10 the weight:
        # population is 10 zeros + 1 holder -> G = 10/11.
        bundle = _bundle([0.0, 100.0], [10.0, 1.0])
        assert gini(bundle, "x") == pytest.approx(10.0 / 11.0)

    def test_zero_total_short_circuits_to_zero(self) -> None:
        bundle = _bundle([0.0, 0.0], [1.0, 2.0])
        assert gini(bundle, "x") == 0.0

    def test_negatives_warn_when_unhandled(self) -> None:
        bundle = _bundle([-10.0, 110.0], [1.0, 1.0])
        with pytest.warns(UserWarning, match="negative"):
            gini(bundle, "x")

    def test_negatives_zero_option_clips(self) -> None:
        clipped = _bundle([-10.0, 100.0], [1.0, 1.0])
        reference = _bundle([0.0, 100.0], [1.0, 1.0])
        assert gini(clipped, "x", negatives="zero") == gini(reference, "x")

    def test_negatives_shift_option_translates(self) -> None:
        shifted = _bundle([-10.0, 90.0], [1.0, 1.0])
        reference = _bundle([0.0, 100.0], [1.0, 1.0])
        assert gini(shifted, "x", negatives="shift") == gini(reference, "x")

    def test_unknown_negatives_option_is_rejected(self) -> None:
        bundle = _bundle([1.0, 2.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="negatives"):
            gini(bundle, "x", negatives="absolute")


class TestGroupbyWsum:
    def test_same_entity_grouping(self, make_bundle) -> None:
        bundle = make_bundle()
        person = bundle.person.copy()
        person["sex"] = ["F", "M", "F", "M", "F"]
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            bundle.schema,
            {"household": bundle.weights_for("household")},
        )
        result = groupby_wsum(rebuilt, "income", by="sex")
        # F: 100*50000 + 200*30000 + 200*0 = 11,000,000
        # M: 100*0 + 200*20000 = 4,000,000
        assert result.to_dict() == {"F": 11_000_000.0, "M": 4_000_000.0}
        assert result.index.name == "sex"

    def test_person_column_grouped_by_group_entity_column(self, make_bundle) -> None:
        bundle = make_bundle()
        result = groupby_wsum(bundle, "income", by="state")
        assert result.to_dict() == {"CA": 5_000_000.0, "NY": 10_000_000.0}

    def test_group_column_by_person_column_is_refused(self, make_bundle) -> None:
        bundle = make_bundle()
        household = bundle.table("household").copy()
        household["rent"] = [1000.0, 2000.0]
        rebuilt = Frame(
            {"person": bundle.person, "household": household},
            bundle.schema,
            {"household": bundle.weights_for("household")},
        )
        with pytest.raises(ValueError, match="Cannot group"):
            groupby_wsum(rebuilt, "rent", by="age")

    def test_group_entity_column_grouped_on_itself(self, make_bundle) -> None:
        bundle = make_bundle()
        household = bundle.table("household").copy()
        household["rent"] = [1000.0, 2000.0]
        rebuilt = Frame(
            {"person": bundle.person, "household": household},
            bundle.schema,
            {"household": bundle.weights_for("household")},
        )
        result = groupby_wsum(rebuilt, "rent", by="state")
        assert result.to_dict() == {"CA": 100_000.0, "NY": 400_000.0}
