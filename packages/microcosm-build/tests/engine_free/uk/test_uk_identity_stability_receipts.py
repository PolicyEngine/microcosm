"""The E7 identity receipt must never pass vacuously (#747 review).

A receipt's whole value is that it certifies something. The E7 branch
recomputed the support-channel layer only when the synthetic flag was
present and otherwise returned an empty recomputation — over which the
mismatch loops never ran, so the receipt reported
``identical_under_permutation: true`` and ``matches_stored_columns: true``
with exit 0 on an artifact where nothing had been checked. These tests pin
the refusals that replaced that silence.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.age_tail import (
    UK_AGE_TOP_CODE,
    disaggregate_uk_age_top_code,
)
from microcosm.build.uk_runtime.cgt_imputation import UK_CGT_INVESTABLE_WEALTH_COLUMNS
from microcosm.build.uk_runtime.cgt_structure import (
    HOUSEHOLD_IS_CGT_CLONE,
    clone_cgt_incidence,
    load_advani_summers_distribution,
)
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_COPIES_COLUMN,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    split_cgt_support_households,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_OUTPUT_COLUMNS,
    UKETBServicesStageTransform,
    allocate_nhs_by_age_gender,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_cgt_support import (
    BAND_INCOMES,
    PARAMETERS,
    support_distribution,
    support_frame,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_TOOL_PATH = _TEST_PATHS.repository / "tools" / "verify_uk_identity_stability.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_uk_identity_stability", _TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    synthetic: bool = True,
    source_keys: bool = True,
    stored_channel: bool = True,
):
    """A two-household frame carrying the E7 support-channel layer."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [10, 10, 20],
            "person_household_id": [100, 100, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [10.0, 20.0],
        }
    )
    if synthetic:
        household["household_is_spi_synthetic"] = [False, True]
    if source_keys:
        household["source_year"] = [2024, 2024]
        household["source_household_id"] = [100, 100]
    if stored_channel:
        household["household_support_channel"] = ["frs", "spi"]
        household["household_support_clone_index"] = [0, 1]
        household["source_household_key"] = ["2024:100", "2024:100"]
        person["person_support_channel"] = ["frs", "frs", "spi"]
        benunit["benunit_support_channel"] = ["frs", "spi"]
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestE7Receipt:
    def test_a_complete_artifact_receipts_green(self) -> None:
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(_frame(), permutation_seed=7)
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        # The receipt names what it compared, so a green result is auditable.
        assert receipt["columns_compared"]["household"] == [
            "household_support_channel",
            "household_support_clone_index",
            "source_household_key",
        ]

    def test_an_artifact_without_the_e7_layer_is_refused(self) -> None:
        # Previously this returned a green receipt over an empty comparison.
        tool = _load_tool()
        with pytest.raises(ValueError, match="no\\s+household_is_spi_synthetic"):
            tool.e7_identity_receipt(_frame(synthetic=False), permutation_seed=7)

    def test_missing_source_keys_are_refused_not_skipped(self) -> None:
        # Skipping the source key would silently shrink the receipt's
        # coverage while still reporting a pass.
        tool = _load_tool()
        with pytest.raises(ValueError, match="source key cannot be recomputed"):
            tool.e7_identity_receipt(_frame(source_keys=False), permutation_seed=7)

    def test_a_column_absent_from_the_store_is_a_mismatch(self) -> None:
        # The store not carrying a column this receipt certifies is a failed
        # comparison, not a narrower one.
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(
            _frame(stored_channel=False), permutation_seed=7
        )
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert (
            "household_support_channel"
            in receipt["stored_column_mismatches"]["household"]
        )

    def test_a_corrupted_stored_channel_is_caught(self) -> None:
        tool = _load_tool()
        frame = _frame()
        household = frame.table("household")
        household.loc[household.index[-1], "household_support_channel"] = "frs"
        receipt = tool.e7_identity_receipt(frame, permutation_seed=7)
        assert receipt["matches_stored_columns"] is False


class TestE6Receipt:
    def test_nhs_receipt_uses_stage_time_disaggregated_age(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeModel:
            def __init__(self, *, n_estimators, seed):
                del n_estimators, seed

            def start_chain(self, donor, predictors, targets, *, weights):
                del donor, predictors, weights
                return {"targets": list(targets)}

            def fit_draw_next(self, donor, recipient_base, raw, *, state, weights):
                del recipient_base, raw, weights
                target = state["targets"][0]
                state = {"targets": state["targets"][1:]}
                return SimpleNamespace(
                    raw_draw=pd.Series(
                        [float(donor[target].iloc[0])], index=donor.index[:1]
                    ),
                    weight_kind="explicit",
                    state=state,
                )

        import microcosm.fit as fit_module

        monkeypatch.setattr(fit_module, "RegimeGatedQRF", _FakeModel)

        class _FakeEngine:
            country = "uk"

            _entities = {
                "is_adult": "person",
                "is_child": "person",
                "is_SP_age": "person",
                "dla": "person",
                "pip": "person",
                "hbai_household_net_income": "household",
                "current_education": "person",
            }

            def variable_metadata(self, name):
                return SimpleNamespace(entity=self._entities[name])

            def materialize(self, frame, variables, period):
                del period
                person_rows = len(frame.table("person"))
                household_rows = len(frame.table("household"))
                values = {
                    "is_adult": np.ones(person_rows),
                    "is_child": np.zeros(person_rows),
                    "is_SP_age": np.ones(person_rows),
                    "dla": np.zeros(person_rows),
                    "pip": np.zeros(person_rows),
                    "hbai_household_net_income": np.full(household_rows, 100.0),
                    "current_education": np.full(
                        person_rows, "NOT_IN_EDUCATION", dtype=object
                    ),
                }
                return {variable: values[variable] for variable in variables}

        frame = uk_national_frame(
            person=pd.DataFrame(
                {
                    "person_id": [1],
                    "person_benunit_id": [10],
                    "person_household_id": [100],
                    "age": [float(UK_AGE_TOP_CODE)],
                    "gender": ["FEMALE"],
                }
            ),
            benunit=pd.DataFrame({"benunit_id": [10], "benunit_household_id": [100]}),
            household=pd.DataFrame({"household_id": [100], "household_weight": [1.0]}),
            time_period="2024",
            weight_kind=WeightKind.DESIGN,
        )
        disaggregate_uk_age_top_code(
            frame,
            band_populations={
                ("MALE", "80_84"): 1.0,
                ("MALE", "85_89"): 1.0,
                ("MALE", "90_plus"): 1.0,
                ("FEMALE", "80_84"): 1.0,
                ("FEMALE", "85_89"): 1e9,
                ("FEMALE", "90_plus"): 1.0,
            },
        )
        stage = SourceStageSpec(
            stage="etb_services",
            survey="etb",
            source="fixture",
            grain="household",
            artifacts=(),
            operations=(
                SourceOperationSpec(kind="derive", parameters={}),
                SourceOperationSpec(
                    kind="fit_weighted_qrf_chain", parameters={"seed": 0}
                ),
                SourceOperationSpec(
                    kind="compute_ratio",
                    parameters={
                        "output": "rail_usage",
                        "denominator_key": "rail_fare_index_2024",
                    },
                ),
            ),
            outputs=(),
        )
        donor = pd.DataFrame(
            {
                "year": [2024],
                "adults": [1],
                "childs": [0],
                "disinc": [100.0],
                "educ": [1.0],
                "rail": [1.0],
                "bussub": [1.0],
                "hhold_adj_weight": [1.0],
                "noretd": [1],
                "primed": [0],
                "secoed": [0],
                "furted": [0],
                "disliv": [0.0],
                "pips": [0.0],
            }
        )
        frame = UKETBServicesStageTransform(
            stage=stage, engine=_FakeEngine(), donor=donor
        )(frame)

        person = frame.table("person")
        household = frame.table("household")
        final_age_nhs = allocate_nhs_by_age_gender(
            person,
            household_weights=frame.weights_for("household").values,
            household=household,
            nhs_table=None,
        )
        stored = person.set_index("person_id")[list(UK_NHS_OUTPUT_COLUMNS)]
        final_age_nhs.index = stored.index
        for column in UK_NHS_OUTPUT_COLUMNS:
            assert np.allclose(
                stored[column].to_numpy(dtype=float),
                final_age_nhs[column].to_numpy(dtype=float),
            )

        clamped_person = person.copy()
        clamped_person["age"] = np.minimum(
            pd.to_numeric(clamped_person["age"], errors="raise").to_numpy(dtype=float),
            float(UK_AGE_TOP_CODE),
        )
        clamped_nhs = allocate_nhs_by_age_gender(
            clamped_person,
            household_weights=frame.weights_for("household").values,
            household=household,
            nhs_table=None,
        )
        clamped_nhs.index = stored.index
        assert any(
            not np.allclose(
                stored[column].to_numpy(dtype=float),
                clamped_nhs[column].to_numpy(dtype=float),
            )
            for column in UK_NHS_OUTPUT_COLUMNS
        )

        receipt = _load_tool().e6_identity_receipt(frame, permutation_seed=7)
        assert receipt["nhs_age_basis"] == "stage_time_disaggregated"
        assert receipt["matches_stored_columns"] is True
        assert receipt["stored_column_mismatches"] == {}


def test_e8_carrier_recompute_uses_disaggregated_age():
    """The support split's carrier and its receipt use final disaggregated age.

    The split classifies each household by its oldest adult's income band
    (``cgt_support_income_band`` over ``_oldest_adult_indices``), and the E8
    recompute reruns that on the stored age. Two adults tie on the former
    top-coded surface, while disaggregation lifts one to 90. The selected
    carrier must be the lifted person; clamping back to 80 demonstrates that
    the basis choice is load-bearing.
    """

    from microcosm.build.uk_runtime.age_tail import UK_AGE_TOP_CODE as TOP
    from microcosm.build.uk_runtime.cgt_structure import _oldest_adult_indices

    top_coded = pd.DataFrame(
        {
            "person_id": [0, 1],
            "person_household_id": [7, 7],
            "age": [float(TOP), float(TOP)],
        }
    )
    disaggregated = top_coded.assign(age=[float(TOP), 90.0])

    stage_choice = _oldest_adult_indices(disaggregated, household_ids={7})
    # The lifted person (row 1) wins outright; the top-coded tie would have
    # gone to row 0 on the stable person_id order.
    assert stage_choice.tolist() == [1]

    clamped = disaggregated.assign(
        age=np.minimum(
            pd.to_numeric(disaggregated["age"], errors="coerce").to_numpy(dtype=float),
            float(TOP),
        )
    )
    assert _oldest_adult_indices(clamped, household_ids={7}).tolist() != (
        stage_choice.tolist()
    )


def _e9_frame(*, clone_flag: bool = False):
    """Three benunits over two households with a region, ready for the E9 stage."""

    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [100, 100, 200, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20, 30]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "NORTH_EAST"],
        }
    )
    if clone_flag:
        household[HOUSEHOLD_IS_CGT_CLONE] = [False, True]
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestE9Receipt:
    def test_stage_output_recomputes_identically_and_matches_stored(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(), resource=load_uc_deduction_distributions()
        ).frame
        receipt = tool.e9_identity_receipt(staged, permutation_seed=7)

        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        assert receipt["benunits_recomputed"] == 3
        assert receipt["benunits_excluded_as_copies"] == 0

    def test_a_tampered_stored_rate_is_reported(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(), resource=load_uc_deduction_distributions()
        ).frame
        benunit = staged.table("benunit").copy()
        benunit.loc[0, "uc_latent_deduction_rate"] = 0.123
        tampered = uk_national_frame(
            person=staged.table("person").copy(),
            benunit=benunit,
            household=staged.table("household").copy(),
            time_period="2024",
            weight_kind=WeightKind.DESIGN,
            household_weights=staged.weights_for("household").values,
        )
        receipt = tool.e9_identity_receipt(tampered, permutation_seed=7)

        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert receipt["stored_mismatches"] == {"benunit": ["uc_latent_deduction_rate"]}

    def test_cloned_households_are_excluded_from_the_recompute(self) -> None:
        from microcosm.build.uk_runtime.uc_deduction_attributes import (
            assign_uc_deduction_attributes,
            load_uc_deduction_distributions,
        )

        tool = _load_tool()
        staged = assign_uc_deduction_attributes(
            _e9_frame(clone_flag=True), resource=load_uc_deduction_distributions()
        ).frame
        receipt = tool.e9_identity_receipt(staged, permutation_seed=7)

        assert receipt["benunits_recomputed"] == 1
        assert receipt["benunits_excluded_as_copies"] == 2
        assert receipt["matches_stored_columns"] is True


class TestRosterOrderedScoping:
    """E5/E6 run after the SPI stack and before the CGT layers."""

    def test_only_layers_stacked_after_a_stage_are_stripped(self) -> None:
        tool = _load_tool()
        household = pd.DataFrame(
            {
                "household_is_spi_synthetic": [False],
                "household_is_capital_gains_clone": [False],
                "household_is_cgt_support_copy": [False],
            }
        )
        expected_later = [
            "household_is_capital_gains_clone",
            "household_is_cgt_support_copy",
        ]
        assert tool._flags_stacked_after("was_wealth", household) == expected_later
        assert tool._flags_stacked_after("nts_bus_travel", household) == (
            expected_later
        )
        # The E4 draws still precede every stacking layer.
        assert tool._flags_stacked_after("frs_take_up", household) == [
            "household_is_spi_synthetic",
            *expected_later,
        ]

    def test_e5_receipt_covers_the_spi_rows_the_stage_imputed(self) -> None:
        from microcosm.build.uk_runtime.regional_uprating import (
            uprate_household_property_by_region,
        )

        tool = _load_tool()
        resource = {
            "values": [{"region": "LONDON", "avg_house_price": 400.0, "dwellings": 1}]
        }
        household = pd.DataFrame(
            {
                "household_id": [100, 200, 300],
                "region": ["LONDON", "LONDON", "LONDON"],
                "household_support_channel": ["frs", "frs", "spi"],
                "household_is_spi_synthetic": [False, False, True],
                "main_residence_value": [100.0, 300.0, 1_000.0],
                "property_wealth": [100.0, 300.0, 1_000.0],
            }
        )
        stored = uprate_household_property_by_region(household, resource)
        frame = uk_national_frame(
            person=pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [10, 20, 30],
                    "person_household_id": [100, 200, 300],
                }
            ),
            benunit=pd.DataFrame({"benunit_id": [10, 20, 30]}),
            household=stored,
            time_period="2024",
            weight_kind=WeightKind.IMPORTANCE,
            household_weights=np.array([10.0, 10.0, 20.0]),
        )

        scoped = tool._frame_as_stage_saw(frame, "was_wealth")
        assert scoped.table("household")["household_id"].tolist() == [100, 200, 300]
        receipt = tool.e5_identity_receipt(
            scoped, regional_resource=resource, permutation_seed=7
        )
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True


# --- the CGT support split (microcosm#1045) -----------------------------------
#
# Six one-person households: band 0 (income 20,000) ids 1-4 and band 37,700
# (income 60,000) ids 5-6. The synthetic joint publishes 60 top-band taxpayers
# in band 0 (support mass 2 x 2 x 60 = 240) and 10 in band 37,700 (40). Band 0
# walks by wealth descending then id ascending: id 2 (130) then id 3 (121)
# cross 240, so both split into three copies; band 37,700 takes id 6 (61) at
# once, two copies. Ids 1, 4 and 5 stay whole; id 5 is the heaviest unselected
# incumbent and takes the exact-total correction. Pre-split multiplier 10.

_SPLIT_WEIGHTS = [70.0, 130.0, 121.0, 30.0, 90.0, 61.0]
_SPLIT_WEALTH = [100.0, 500.0, 500.0, 200.0, 10.0, 20.0]
_SPLIT_INCOMES = [BAND_INCOMES[0]] * 4 + [BAND_INCOMES[37_700]] * 2
_SPLIT_TOP_CELLS = {0: {250_000: 60.0, 500_000: None}, 37_700: {250_000: 10.0}}


def _split_then_cloned():
    """The split stage then the clone stage on the scenario; the split result too."""

    split = split_cgt_support_households(
        support_frame(
            weights=_SPLIT_WEIGHTS, wealth=_SPLIT_WEALTH, incomes=_SPLIT_INCOMES
        ),
        distribution=support_distribution(_SPLIT_TOP_CELLS),
        parameters=PARAMETERS,
    )
    cloned = clone_cgt_incidence(
        split.frame, distribution=load_advani_summers_distribution()
    )
    return split, cloned.frame


def _rebuild(frame, *, household=None, household_weights=None, mass_log=None):
    """``frame`` with one of its household table, weights or mass log replaced."""

    return uk_national_frame(
        person=frame.table("person").copy(),
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy() if household is None else household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=(
            frame.weights_for("household").values
            if household_weights is None
            else household_weights
        ),
        mass_log=frame.mass_log if mass_log is None else mass_log,
    )


def _hand_split_frame(*, cloned: bool = True, anchored: bool = False):
    """Three source households, the first split into three at a third each.

    Pre-split ids 1..3 (persons and benunits alike, multiplier 10) at source
    weights 90 / 20 / 40: root 1 and its copies 11 and 21 carry 30 each,
    flagged and counted as the stage leaves them. With ``cloned`` the clone
    layer follows (multiplier 100) at equal halves, or with ``anchored`` at
    an uneven pair split whose pair sums are still the pre-clone weights, as
    the #970 anchor leaves them.
    """

    ids = np.array([1, 2, 3, 11, 21], dtype="int64")
    flags = [False, False, False, True, True]
    copies = [3, 1, 1, 3, 3]
    pre_clone = np.array([30.0, 20.0, 40.0, 30.0, 30.0])
    if cloned:
        ids = np.r_[ids, ids + 100]
        flags = flags + flags
        copies = copies + copies
        if anchored:
            weights = np.r_[
                [20.0, 12.0, 25.0, 18.0, 16.0], [10.0, 8.0, 15.0, 12.0, 14.0]
            ]
        else:
            weights = np.r_[pre_clone / 2, pre_clone / 2]
    else:
        weights = pre_clone
    household = pd.DataFrame(
        {
            "household_id": ids,
            HOUSEHOLD_IS_CGT_SUPPORT_COPY: flags,
            CGT_SUPPORT_COPIES_COLUMN: np.asarray(copies, dtype="int64"),
        }
    )
    if cloned:
        household[HOUSEHOLD_IS_CGT_CLONE] = [False] * 5 + [True] * 5
    return uk_national_frame(
        person=pd.DataFrame(
            {"person_id": ids, "person_benunit_id": ids, "person_household_id": ids}
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=weights,
    )


class TestPreSplitFold:
    """Folding copies onto roots, after clones onto originals, inverts the split."""

    @pytest.mark.parametrize("anchored", [False, True])
    def test_fold_restores_the_source_weights_through_both_layers(
        self, anchored: bool
    ) -> None:
        tool = _load_tool()
        frame = _hand_split_frame(anchored=anchored)
        folded = tool._pre_split_household_weights(frame)
        # Roots and unsplit households are back at their source weights; the
        # copies carry their pair sums and the clones their own weights (the
        # scoping drops both).
        assert folded[:3].tolist() == [90.0, 20.0, 40.0]
        assert folded[3:5].tolist() == [30.0, 30.0]
        np.testing.assert_array_equal(
            folded[5:], frame.weights_for("household").values[5:]
        )

    def test_fold_without_a_clone_layer_folds_the_copies_only(self) -> None:
        tool = _load_tool()
        folded = tool._pre_split_household_weights(_hand_split_frame(cloned=False))
        assert folded.tolist() == [90.0, 20.0, 40.0, 30.0, 30.0]

    def test_stage_scoping_returns_the_pre_split_table_at_source_weights(
        self,
    ) -> None:
        tool = _load_tool()
        scoped = tool._frame_as_stage_saw(
            _hand_split_frame(anchored=True), "was_wealth"
        )
        assert scoped.table("household")["household_id"].tolist() == [1, 2, 3]
        assert scoped.weights_for("household").values.tolist() == [90.0, 20.0, 40.0]
        assert scoped.table("person")["person_id"].tolist() == [1, 2, 3]
        assert scoped.table("benunit")["benunit_id"].tolist() == [1, 2, 3]

    def test_fold_inverts_the_stage_on_a_frame_it_built(self) -> None:
        tool = _load_tool()
        _, frame = _split_then_cloned()
        household = frame.table("household")
        pre_split = ~household[HOUSEHOLD_IS_CGT_CLONE].to_numpy(
            dtype=bool
        ) & ~household[HOUSEHOLD_IS_CGT_SUPPORT_COPY].to_numpy(dtype=bool)
        folded = tool._pre_split_household_weights(frame)
        assert household.loc[pre_split, "household_id"].tolist() == [1, 2, 3, 4, 5, 6]
        np.testing.assert_allclose(folded[pre_split], _SPLIT_WEIGHTS, rtol=1e-12)

    def test_dropping_the_split_layer_under_a_kept_clone_layer_is_refused(
        self,
    ) -> None:
        tool = _load_tool()
        with pytest.raises(ValueError, match="requires dropping the clone layer"):
            tool._drop_stacked_layers(
                _hand_split_frame(), [HOUSEHOLD_IS_CGT_SUPPORT_COPY]
            )

    def test_a_copy_without_a_root_fails_closed(self) -> None:
        tool = _load_tool()
        frame = _hand_split_frame(cloned=False)
        household = frame.table("household").copy()
        # Flagging id 2 as a copy gives it index 2 // 10 = 0: no root.
        household.loc[household["household_id"] == 2, HOUSEHOLD_IS_CGT_SUPPORT_COPY] = (
            True
        )
        with pytest.raises(ValueError, match="without a root household"):
            tool._pre_split_household_weights(_rebuild(frame, household=household))

    def test_the_stored_copy_index_is_read_and_held_to_the_id_scheme(self) -> None:
        """The explicit index (microcosm#1045 review) must agree with the ids."""
        from microcosm.build.uk_runtime.cgt_support import (
            CGT_SUPPORT_COPY_INDEX_COLUMN,
        )

        tool = _load_tool()
        _, frame = _split_then_cloned()
        household = frame.table("household")
        lineage = tool._support_copy_lineage(
            frame.table("person"), frame.table("benunit"), household
        )
        stored = household[CGT_SUPPORT_COPY_INDEX_COLUMN].to_numpy()
        np.testing.assert_array_equal(
            stored[lineage.copy_positions], lineage.copy_index
        )
        assert (stored[lineage.pre_split] == 0).all()

        tampered = household.copy()
        copy_id = int(household["household_id"].to_numpy()[lineage.copy_positions[0]])
        tampered.loc[
            tampered["household_id"] == copy_id, CGT_SUPPORT_COPY_INDEX_COLUMN
        ] = 7
        with pytest.raises(ValueError, match="disagrees with the id scheme"):
            tool._pre_split_household_weights(_rebuild(frame, household=tampered))

        unflagged = household.copy()
        unflagged.loc[
            unflagged["household_id"] == copy_id, HOUSEHOLD_IS_CGT_SUPPORT_COPY
        ] = False
        with pytest.raises(ValueError, match="flag and stored copy index disagree"):
            tool._pre_split_household_weights(_rebuild(frame, household=unflagged))

        receipt, problems = TestE8SupportSplitRecompute._receipt(frame)
        assert problems == {}
        assert receipt["copy_index_column_stored"] is True


class TestE8SupportSplitRecompute:
    """The split rerun on the folded pre-split frame reproduces the stored layer."""

    @staticmethod
    def _receipt(frame) -> tuple[dict[str, object], dict[str, object]]:
        problems: dict[str, object] = {}
        receipt = _load_tool()._e8_support_split(
            frame,
            problems,
            distribution=support_distribution(_SPLIT_TOP_CELLS),
            parameters=PARAMETERS,
            permutation_seed=7,
        )
        return receipt, problems

    def test_a_frame_the_stage_built_receipts_green(self) -> None:
        split, frame = _split_then_cloned()
        receipt, problems = self._receipt(frame)
        assert problems == {}
        assert receipt["copies"] == split.copies_created == 5
        assert receipt["families"] == split.households_selected == 3
        assert receipt["copies_recomputed"] == 5
        assert receipt["households_selected"] == 3
        assert receipt["pre_split_households"] == 6
        assert receipt["id_multiplier"] == 10
        # 2 x 1.25 x (60 + 10) published in the two bands.
        assert receipt["support_mass"] == 175.0
        assert receipt["mass"]["old_total"] == receipt["mass"]["new_total"]
        assert receipt["max_abs_family_weight_diff"] < 1e-9
        assert [row["households_selected"] for row in receipt["bands"]] == [
            2,
            1,
            0,
            0,
            0,
            0,
        ]

    def test_a_moved_copy_weight_is_reported(self) -> None:
        _, frame = _split_then_cloned()
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        # A unit of mass moves between root 2's two copies: the family total
        # holds, so the selection stands, but two members leave w / n.
        weights[ids == 12] += 1.0
        weights[ids == 22] -= 1.0
        _, problems = self._receipt(_rebuild(frame, household_weights=weights))
        assert problems == {"support_split_family_weights": 2}

    def test_a_tampered_copy_count_is_reported(self) -> None:
        _, frame = _split_then_cloned()
        household = frame.table("household").copy()
        household.loc[household["household_id"] == 2, CGT_SUPPORT_COPIES_COLUMN] = 2
        _, problems = self._receipt(_rebuild(frame, household=household))
        # Both copies now disagree with their root, the family has one copy
        # too many for its count, the rule says three, and w / 2 fits no
        # member; the selection itself is untouched.
        assert problems["support_split_flags"] == {
            "copies_disagreeing_with_root": 2,
            "families_with_missing_or_extra_copies": 1,
        }
        assert problems["support_split_copies_stored"] == 1
        assert problems["support_split_family_weights"] == 3
        assert "support_split_selection_stored" not in problems
        assert "support_split_selection_permutation" not in problems

    def test_a_household_the_rule_divides_but_the_store_kept_whole_is_reported(
        self,
    ) -> None:
        _, frame = _split_then_cloned()
        household = frame.table("household").copy()
        # Household 1 (weight 70, whole in the store) becomes the wealthiest
        # in band 0, so the rule now divides it into two copies; band 0's
        # support mass of 150 is then covered by ids 1 and 2, so the stored
        # family 3 is missing from the recomputed rule as well.
        household.loc[
            household["household_id"] == 1, UK_CGT_INVESTABLE_WEALTH_COLUMNS[0]
        ] = 1e6
        _, problems = self._receipt(_rebuild(frame, household=household))
        assert problems == {
            "support_split_selection_stored": {"missing": 1, "extra": 1}
        }

    def test_a_missing_mass_record_is_reported(self) -> None:
        _, frame = _split_then_cloned()
        receipt, problems = self._receipt(_rebuild(frame, mass_log=()))
        assert problems == {"support_split_mass_record": "missing"}
        assert "mass" not in receipt

    def test_an_artifact_without_the_split_layer_is_refused(self) -> None:
        frame = support_frame(
            weights=_SPLIT_WEIGHTS, wealth=_SPLIT_WEALTH, incomes=_SPLIT_INCOMES
        )
        with pytest.raises(ValueError, match="support-split layer is absent"):
            self._receipt(frame)
