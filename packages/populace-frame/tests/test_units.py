"""US unit structure on synthetic CPS households, returning a bundle.

A representative subset of the proven unit-assignment behaviors: tax units
delegated to microunit (married couple with kids, single filer,
multigenerational household where the adult child files separately), the SPM
passthrough and household fallback, family splitting by PF_SEQ, marital-unit
spouse pairing, determinism, validation errors, and the bundle-shaped result
(membership columns, weight alignment, global column uniqueness).
"""

import numpy as np
import pandas as pd
import pytest

from populace.frame import (
    MICROUNIT_REQUIRED_COLUMNS,
    TAX_UNIT_FILING_STATUS_COLUMN,
    US_SCHEMA,
    EntitySchema,
    Frame,
    WeightKind,
    Weights,
)

pytest.importorskip("microunit")

from populace.frame import assign_us_unit_structure  # noqa: E402
from populace.frame.units import _microunit_input_frame  # noqa: E402


def _weights(frame: pd.DataFrame) -> Weights:
    n = frame["household_id"].nunique()
    return Weights(values=np.full(n, 1000.0), kind=WeightKind.DESIGN)


def _assign(frame: pd.DataFrame, **kwargs) -> Frame:
    kwargs.setdefault("household_weights", _weights(frame))
    return assign_us_unit_structure(frame, year=2024, **kwargs)


def _by_line(bundle: Frame, ph_seq: int, line: int) -> pd.Series:
    """Return the augmented person row for a given household line number."""
    person = bundle.person
    mask = (person["PH_SEQ"] == ph_seq) & (person["A_LINENO"] == line)
    matched = person[mask]
    assert len(matched) == 1, f"expected one row for PH_SEQ={ph_seq} line={line}"
    return matched.iloc[0]


class TestTaxUnits:
    def test_married_couple_with_kids_is_one_joint_unit(
        self, married_with_kids
    ) -> None:
        bundle = _assign(married_with_kids)
        head = _by_line(bundle, 10, 1)
        spouse = _by_line(bundle, 10, 2)
        child_a = _by_line(bundle, 10, 3)
        child_b = _by_line(bundle, 10, 4)

        tax_id = head["person_tax_unit_id"]
        assert spouse["person_tax_unit_id"] == tax_id
        assert child_a["person_tax_unit_id"] == tax_id
        assert child_b["person_tax_unit_id"] == tax_id

        assert head["tax_unit_role_input"] == "HEAD"
        assert spouse["tax_unit_role_input"] == "SPOUSE"
        assert child_a["tax_unit_role_input"] == "DEPENDENT"
        assert child_b["tax_unit_role_input"] == "DEPENDENT"

        filing = bundle.table("tax_unit").set_index("tax_unit_id")[
            TAX_UNIT_FILING_STATUS_COLUMN
        ]
        assert filing.loc[tax_id] == "JOINT"

    def test_single_filer_is_one_single_unit(self, single_filer) -> None:
        bundle = _assign(single_filer)
        person = _by_line(bundle, 20, 1)
        assert person["tax_unit_role_input"] == "HEAD"
        filing = bundle.table("tax_unit").set_index("tax_unit_id")[
            TAX_UNIT_FILING_STATUS_COLUMN
        ]
        assert filing.loc[person["person_tax_unit_id"]] == "SINGLE"
        assert bundle.n("tax_unit") == 1

    def test_independent_adult_child_is_a_separate_tax_unit(
        self, multigenerational
    ) -> None:
        bundle = _assign(multigenerational)
        grandparent = _by_line(bundle, 30, 1)
        adult_child = _by_line(bundle, 30, 2)
        grandchild = _by_line(bundle, 30, 3)

        # The adult child files separately from the grandparent...
        assert adult_child["person_tax_unit_id"] != grandparent["person_tax_unit_id"]
        assert adult_child["tax_unit_role_input"] == "HEAD"
        assert grandparent["tax_unit_role_input"] == "HEAD"
        # ...and claims the grandchild as a dependent in their own unit.
        assert grandchild["person_tax_unit_id"] == adult_child["person_tax_unit_id"]
        assert grandchild["tax_unit_role_input"] == "DEPENDENT"

    def test_tax_unit_ids_globally_dense_across_households(
        self, three_household_frame
    ) -> None:
        bundle = _assign(three_household_frame)
        ids = sorted(bundle.person["person_tax_unit_id"].unique().tolist())
        # 4 tax units: couple, single, grandparent, adult-child-with-grandchild.
        assert ids == [1, 2, 3, 4]
        assert bundle.table("tax_unit")["tax_unit_id"].tolist() == [1, 2, 3, 4]


class TestSpmUnitsAndFamilies:
    def test_distinct_spm_ids_within_household_split(self, married_with_kids) -> None:
        frame = married_with_kids
        frame = frame.copy()
        frame.loc[frame.index[:2], "SPM_ID"] = 100  # split the household
        bundle = _assign(frame)
        ids = bundle.person["person_spm_unit_id"]
        assert ids.nunique() == 2
        assert bundle.table("spm_unit")["spm_unit_id"].tolist() == [1, 2]

    def test_missing_spm_id_falls_back_to_household(self, married_with_kids) -> None:
        frame = married_with_kids.drop(columns=["SPM_ID"])
        bundle = _assign(frame)
        assert bundle.person["person_spm_unit_id"].nunique() == 1
        assert bundle.table("spm_unit")["spm_unit_id"].tolist() == [1]

    def test_family_splits_within_household_by_pf_seq(
        self, multigenerational
    ) -> None:
        bundle = _assign(multigenerational)
        grandparent = _by_line(bundle, 30, 1)
        adult_child = _by_line(bundle, 30, 2)
        grandchild = _by_line(bundle, 30, 3)
        # PF_SEQ 1 vs 2 -> two families even though it is one household.
        assert grandparent["person_family_id"] != adult_child["person_family_id"]
        assert grandchild["person_family_id"] == adult_child["person_family_id"]


class TestMaritalUnits:
    def test_spouses_share_and_children_are_singletons(
        self, married_with_kids
    ) -> None:
        bundle = _assign(married_with_kids)
        couple_id = _by_line(bundle, 10, 1)["person_marital_unit_id"]
        assert _by_line(bundle, 10, 2)["person_marital_unit_id"] == couple_id
        child_a = _by_line(bundle, 10, 3)["person_marital_unit_id"]
        child_b = _by_line(bundle, 10, 4)["person_marital_unit_id"]
        assert couple_id not in (child_a, child_b)
        assert child_a != child_b

    def test_dangling_spouse_pointer_does_not_pair_and_does_not_crash(
        self, married_with_kids
    ) -> None:
        # A one-directional A_SPOUSE is not a valid CPS marriage record; it
        # must degrade to singletons, not error.
        frame = married_with_kids.copy()
        frame.loc[frame.index[1], "A_SPOUSE"] = 0
        bundle = _assign(frame)
        unit_ids = bundle.person.set_index("A_LINENO")["person_marital_unit_id"]
        assert unit_ids.loc[1] != unit_ids.loc[2]


class TestBundleResult:
    def test_returns_validated_us_bundle(self, three_household_frame) -> None:
        bundle = _assign(three_household_frame)
        assert isinstance(bundle, Frame)
        assert bundle.schema == US_SCHEMA
        for group in US_SCHEMA.group_entities:
            assert US_SCHEMA.membership_column(group) in bundle.person.columns
            assert US_SCHEMA.id_column(group) in bundle.table(group).columns

    def test_household_id_moves_to_membership_column(
        self, three_household_frame
    ) -> None:
        bundle = _assign(three_household_frame)
        # The bare name belongs to the household table (global uniqueness).
        assert "household_id" not in bundle.person.columns
        assert bundle.person["person_household_id"].tolist() == (
            three_household_frame["household_id"].tolist()
        )
        assert bundle.table("household")["household_id"].tolist() == [10, 20, 30]

    def test_caller_frame_is_not_mutated_and_index_preserved(
        self, married_with_kids
    ) -> None:
        before = married_with_kids.copy()
        bundle = _assign(married_with_kids)
        pd.testing.assert_frame_equal(married_with_kids, before)
        assert bundle.person.index.tolist() == married_with_kids.index.tolist()

    def test_membership_columns_are_int64_and_complete(
        self, three_household_frame
    ) -> None:
        bundle = _assign(three_household_frame)
        for column in (
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        ):
            assert bundle.person[column].dtype == np.int64
            assert bundle.person[column].notna().all()

    def test_identical_input_yields_identical_bundles(
        self, three_household_frame
    ) -> None:
        first = _assign(three_household_frame.copy())
        second = _assign(three_household_frame.copy())
        pd.testing.assert_frame_equal(first.person, second.person)
        for group in US_SCHEMA.group_entities:
            pd.testing.assert_frame_equal(first.table(group), second.table(group))

    def test_strata_are_carried_onto_the_bundle(self, single_filer) -> None:
        strata = pd.Series(["tail_verbatim"], index=single_filer.index)
        bundle = _assign(single_filer, strata=strata)
        assert bundle.strata.tolist() == ["tail_verbatim"]


class TestWeights:
    def test_per_person_weights_collapse_to_household(self, married_with_kids) -> None:
        # Four persons, one household.
        per_person = Weights(values=np.full(4, 1234.0), kind=WeightKind.DESIGN)
        bundle = assign_us_unit_structure(
            married_with_kids, year=2024, household_weights=per_person
        )
        assert bundle.weights_for("household").values.tolist() == [1234.0]

    def test_unequal_per_person_weights_within_household_rejected(
        self, married_with_kids
    ) -> None:
        per_person = Weights(
            values=np.array([1.0, 1.0, 2.0, 1.0]), kind=WeightKind.DESIGN
        )
        with pytest.raises(ValueError, match="constant within each household"):
            assign_us_unit_structure(
                married_with_kids, year=2024, household_weights=per_person
            )

    def test_wrong_length_weights_rejected(self, married_with_kids) -> None:
        wrong = Weights(values=np.array([1.0, 2.0]), kind=WeightKind.DESIGN)
        with pytest.raises(ValueError, match="per household"):
            assign_us_unit_structure(
                married_with_kids, year=2024, household_weights=wrong
            )

    def test_untyped_weights_rejected(self, married_with_kids) -> None:
        with pytest.raises(TypeError, match="Weights"):
            assign_us_unit_structure(
                married_with_kids,
                year=2024,
                household_weights=np.array([1000.0]),
            )


class TestValidation:
    @pytest.mark.parametrize("missing", ["A_SPOUSE", "A_EXPRRP", "PH_SEQ"])
    def test_missing_required_column_names_it(
        self, married_with_kids, missing: str
    ) -> None:
        frame = married_with_kids.drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            _assign(frame)

    def test_required_columns_are_stable(self) -> None:
        assert "PH_SEQ" in MICROUNIT_REQUIRED_COLUMNS
        assert "A_EXPRRP" in MICROUNIT_REQUIRED_COLUMNS

    def test_non_us_schema_is_rejected(self, married_with_kids) -> None:
        with pytest.raises(ValueError, match="US entity schema"):
            _assign(
                married_with_kids,
                schema=EntitySchema(group_entities=("household",)),
            )


class TestHarmonizedIncomeMapping:
    """The microunit input view sources income from harmonized names."""

    def test_harmonized_income_is_mapped_onto_asec_names(
        self, single_filer
    ) -> None:
        frame = single_filer.copy()
        frame["self_employment_income"] = [1200.0]
        assert "WSAL_VAL" not in frame.columns  # only the harmonized names exist

        view = _microunit_input_frame(frame)
        assert view["WSAL_VAL"].tolist() == [48000.0]
        assert view["SEMP_VAL"].tolist() == [1200.0]
        # Harmonized / non-ASEC columns are not carried into the engine view.
        assert "wage_income" not in view.columns
        assert "household_id" not in view.columns

    def test_raw_asec_income_takes_precedence_over_harmonized(
        self, single_filer
    ) -> None:
        frame = single_filer.copy()
        frame["WSAL_VAL"] = [12345.0]  # raw ASEC value present too
        view = _microunit_input_frame(frame)
        assert view["WSAL_VAL"].tolist() == [12345.0]
