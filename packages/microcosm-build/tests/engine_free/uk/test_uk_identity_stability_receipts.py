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
from microcosm.build.uk_runtime import copy_id_table
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
    geographic_clone_count,
    split_cgt_support_households,
)
from microcosm.build.uk_runtime.etb_services import (
    UK_NHS_OUTPUT_COLUMNS,
    UKETBServicesStageTransform,
    allocate_nhs_by_age_gender,
)
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_cgt_support import (
    BAND_INCOMES,
    PARAMETERS,
    support_distribution,
    support_frame,
)
from test_support.microcosm_build.uk_clone_first_frames import clone_first_frame
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

    def test_the_income_band_donors_sit_at_their_own_clone_index(self) -> None:
        # The donors carry the synthetic flag of the SPI support copy but are
        # stacked at clone index 2; reading every synthetic household as index
        # 1 failed the receipt on every spine built since the donor stage.
        tool = _load_tool()
        frame = _frame()
        household = frame.table("household")
        household["household_is_spi_income_band_donor"] = [False, True]
        household["household_support_clone_index"] = [0, 2]
        receipt = tool.e7_identity_receipt(frame, permutation_seed=7)
        assert receipt["matches_stored_columns"] is True

        household["household_support_clone_index"] = [0, 1]
        receipt = tool.e7_identity_receipt(frame, permutation_seed=7)
        assert receipt["matches_stored_columns"] is False

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
            def __init__(self, *, n_estimators, seed, gate_calibration):
                del n_estimators, seed, gate_calibration

            def start_chain(self, donor, predictors, targets, *, weights):
                del donor, predictors, weights
                return {"targets": list(targets)}

            def fit_draw_next(
                self,
                donor,
                recipient_base,
                raw,
                *,
                state,
                weights,
                quantiles,
                sign_uniforms,
            ):
                del recipient_base, raw, weights, quantiles, sign_uniforms
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
                "hbai_household_net_income_before_reset_to_zero": "household",
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
                    "hbai_household_net_income_before_reset_to_zero": np.full(
                        household_rows, 100.0
                    ),
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
                    "is_uc_claimant": [True],
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


class TestPreSalarySacrificeRestoration:
    """A receipt of a stage before salary_sacrifice bands on pre-conversion pay.

    The conversion rewrites a converted record's pay and employee contribution
    in place after the CGT stages banded on them (microcosm#1069 c9); the
    stage keeps the pre-conversion pay on its carrier and the tool reverses
    the rewrite before recomputing (microcosm#1063).
    """

    @staticmethod
    def _converted(frame):
        """The split-then-cloned scenario after a conversion on household 6.

        Household 6 is the band-37,700 family the split divides (wealth 20 over
        household 5's 10). Its carrier sacrifices enough pay to fall into band
        0, the way the stage rewrites a converted record: pay lowered by the
        amount moved into the salary-sacrifice column, and the pre-conversion
        pay kept on the carrier. Here the whole contribution moves, so no
        employee contribution remains.
        """

        from microcosm.build.uk_runtime.salary_sacrifice import (
            SALSAC_OUTPUT,
            SALSAC_PRE_CONVERSION_PAY_COLUMN,
        )

        person = frame.table("person").copy()
        person["employee_pension_contributions"] = 0.0
        person[SALSAC_OUTPUT] = 0.0
        person[SALSAC_PRE_CONVERSION_PAY_COLUMN] = 0.0
        members = person.loc[person["person_household_id"] == 6]
        carrier = members.sort_values(
            ["age", "person_id"], ascending=[False, True]
        ).index[0]
        pay = float(person.loc[carrier, "employment_income"])
        sacrificed = pay - float(BAND_INCOMES[0])
        assert sacrificed > 0.0
        person.loc[carrier, "employment_income"] = pay - sacrificed
        person.loc[carrier, SALSAC_OUTPUT] = sacrificed
        person.loc[carrier, SALSAC_PRE_CONVERSION_PAY_COLUMN] = pay
        rewritten = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=frame.table("household").copy(),
            time_period="2024",
            weight_kind=WeightKind.IMPORTANCE,
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        return rewritten, int(person.loc[carrier, "person_id"]), pay, sacrificed

    def test_the_support_split_receipt_bands_on_the_restored_pay(self) -> None:
        tool = _load_tool()
        _, frame = _split_then_cloned()
        rewritten, _, _, _ = self._converted(frame)

        def receipt(target):
            problems: dict[str, object] = {}
            tool._e8_support_split(
                target,
                problems,
                distribution=support_distribution(_SPLIT_TOP_CELLS),
                parameters=PARAMETERS,
                permutation_seed=7,
            )
            return problems

        # On the stored pay the rule moves household 6 to band 0, where the
        # 500-wealth households outrank it, and divides household 5 instead.
        assert receipt(rewritten) == {
            "support_split_selection_stored": {"missing": 1, "extra": 1}
        }
        assert receipt(tool._pre_salary_sacrifice_frame(rewritten)) == {}

    def test_the_restoration_reverses_the_conversion_exactly(self) -> None:
        from microcosm.build.uk_runtime.salary_sacrifice import (
            SALSAC_OUTPUT,
            SALSAC_PRE_CONVERSION_PAY_COLUMN,
        )

        tool = _load_tool()
        _, frame = _split_then_cloned()
        rewritten, carrier_id, pay, sacrificed = self._converted(frame)
        restored = tool._pre_salary_sacrifice_frame(rewritten)
        person = restored.table("person").set_index("person_id")
        assert person.loc[carrier_id, "employment_income"] == pay
        assert person.loc[carrier_id, "employee_pension_contributions"] == sacrificed
        assert person.loc[carrier_id, SALSAC_OUTPUT] == 0.0
        others = person.drop(index=carrier_id)
        before = rewritten.table("person").set_index("person_id").drop(index=carrier_id)
        pd.testing.assert_frame_equal(others, before, check_exact=True)
        # The carrier itself is untouched, the weights and mass log pass through.
        assert (person[SALSAC_PRE_CONVERSION_PAY_COLUMN] > 0).sum() == 1
        assert restored.weights_for("household").values.tolist() == (
            rewritten.weights_for("household").values.tolist()
        )
        assert restored.mass_log == rewritten.mass_log
        # Scoping to a stage before salary_sacrifice restores; a later stage
        # keeps the stored pay; an artifact without the carrier passes through.
        split_view = tool._frame_as_stage_saw(rewritten, "cgt_support_split")
        assert (
            split_view.table("person")
            .set_index("person_id")
            .loc[carrier_id, "employment_income"]
            == pay
        )
        later_view = tool._frame_as_stage_saw(rewritten, "student_loans")
        assert (
            later_view.table("person")
            .set_index("person_id")
            .loc[carrier_id, "employment_income"]
            == pay - sacrificed
        )
        assert tool._pre_salary_sacrifice_frame(frame) is frame

    def test_a_contribution_the_record_kept_is_restored_with_the_amount_moved(
        self,
    ) -> None:
        # The conversion moves at most half the pay and leaves the rest of the
        # contribution in place (uk-data#518), so the contribution before it is
        # what the record kept plus what moved.
        tool = _load_tool()
        _, frame = _split_then_cloned()
        rewritten, carrier_id, pay, sacrificed = self._converted(frame)
        person = rewritten.table("person").copy()
        person.loc[
            person["person_id"] == carrier_id, "employee_pension_contributions"
        ] = 125.0
        kept = uk_national_frame(
            person=person,
            benunit=rewritten.table("benunit"),
            household=rewritten.table("household"),
            time_period="2024",
            weight_kind=WeightKind.IMPORTANCE,
            household_weights=rewritten.weights_for("household").values,
            mass_log=rewritten.mass_log,
        )
        restored = tool._pre_salary_sacrifice_frame(kept).table("person")
        restored = restored.set_index("person_id")
        assert restored.loc[carrier_id, "employment_income"] == pay
        assert restored.loc[carrier_id, "employee_pension_contributions"] == (
            125.0 + sacrificed
        )

    def test_a_carrier_disagreeing_with_the_columns_is_refused(self) -> None:
        from microcosm.build.uk_runtime.salary_sacrifice import (
            SALSAC_OUTPUT,
            SALSAC_PRE_CONVERSION_PAY_COLUMN,
        )

        tool = _load_tool()
        _, frame = _split_then_cloned()
        rewritten, carrier_id, _, sacrificed = self._converted(frame)
        person = rewritten.table("person").copy()
        # Pay plus the sacrifice must give the pay before the conversion; a
        # sacrifice that does not is not the state the conversion leaves.
        person.loc[person["person_id"] == carrier_id, SALSAC_OUTPUT] = sacrificed + 1.0
        tampered = uk_national_frame(
            person=person,
            benunit=rewritten.table("benunit"),
            household=rewritten.table("household"),
            time_period="2024",
            weight_kind=WeightKind.IMPORTANCE,
            household_weights=rewritten.weights_for("household").values,
            mass_log=rewritten.mass_log,
        )
        with pytest.raises(ValueError, match=SALSAC_PRE_CONVERSION_PAY_COLUMN):
            tool._pre_salary_sacrifice_frame(tampered)


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

    def test_the_high_value_copies_are_stripped_from_the_stages_before_them(
        self,
    ) -> None:
        # Stage 26's copies (microcosm#1240) are stacked after the wealth
        # stages and before the travel block.
        tool = _load_tool()
        household = pd.DataFrame(
            {
                "household_is_spi_synthetic": [False],
                HIGH_VALUE_SUPPORT_COPY_COLUMN: [False],
                "household_is_capital_gains_clone": [False],
                "household_is_cgt_support_copy": [False],
            }
        )
        cgt_layers = [
            "household_is_capital_gains_clone",
            "household_is_cgt_support_copy",
        ]
        assert tool._flags_stacked_after("was_wealth", household) == [
            HIGH_VALUE_SUPPORT_COPY_COLUMN,
            *cgt_layers,
        ]
        assert tool._flags_stacked_after("nts_bus_travel", household) == cgt_layers
        assert tool._flags_stacked_after("frs_take_up", household) == [
            "household_is_spi_synthetic",
            HIGH_VALUE_SUPPORT_COPY_COLUMN,
            *cgt_layers,
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
# incumbent and takes the exact-total correction. Copy k of root r carries
# r + k x 10**13, the copy-id table's cgt_support_split field (microcosm#931).

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
        clone_count=1,
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

    Pre-split ids 1..3 (persons and benunits alike) at source weights 90 / 20
    / 40: root 1 and its copies 1 + 10**13 and 1 + 2 x 10**13, minted in the
    copy-id table's ``cgt_support_split`` field as the stage mints them,
    carry 30 each, flagged and counted as the stage leaves them. With
    ``cloned`` the clone layer follows in the ``cgt_incidence_clone`` field
    (10**17) at equal halves, or with ``anchored`` at an uneven pair split
    whose pair sums are still the pre-clone weights, as the #970 anchor
    leaves them.
    """

    ids = np.r_[
        np.array([1, 2, 3], dtype="int64"),
        copy_id_table.mint([1, 1], "cgt_support_split", [1, 2]),
    ]
    flags = [False, False, False, True, True]
    copies = [3, 1, 1, 3, 3]
    pre_clone = np.array([30.0, 20.0, 40.0, 30.0, 30.0])
    if cloned:
        ids = np.r_[ids, copy_id_table.mint(ids, "cgt_incidence_clone", 1)]
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
        refusal = (
            "CGT support copy(ies) without a root household at the declared radix "
            "of the copy-id table's 'cgt_support_split' field."
        )
        household = frame.table("household").copy()
        # Flagging id 2 as a copy gives it ordinal 0 in the cgt_support_split
        # field: no root.
        household.loc[household["household_id"] == 2, HOUSEHOLD_IS_CGT_SUPPORT_COPY] = (
            True
        )
        with pytest.raises(ValueError) as caught:
            tool._pre_split_household_weights(_rebuild(frame, household=household))
        assert str(caught.value) == f"1 {refusal}"
        # Flagging root 1 instead leaves its two copies' parent in the frame
        # but outside the rows the split copied: all three are orphans.
        household = frame.table("household").copy()
        household.loc[household["household_id"] == 1, HOUSEHOLD_IS_CGT_SUPPORT_COPY] = (
            True
        )
        with pytest.raises(ValueError) as caught:
            tool._pre_split_household_weights(_rebuild(frame, household=household))
        assert str(caught.value) == f"3 {refusal}"

    def test_the_stored_copy_index_is_read_and_held_to_the_id_scheme(self) -> None:
        """The explicit index (microcosm#1045 review) must agree with the ids."""
        from microcosm.build.uk_runtime.cgt_support import (
            CGT_SUPPORT_COPY_INDEX_COLUMN,
        )

        tool = _load_tool()
        _, frame = _split_then_cloned()
        household = frame.table("household")
        lineage = tool._support_copy_lineage(household)
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
        with pytest.raises(ValueError) as caught:
            tool._pre_split_household_weights(_rebuild(frame, household=tampered))
        assert str(caught.value) == (
            "CGT support copy index stored on 1 copy(ies) disagrees with the id "
            "scheme (the id's ordinal in the copy-id table's 'cgt_support_split' "
            "field)."
        )
        assert str(copy_id) not in str(caught.value)

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
        # The receipt records the split field's declared radix.
        assert receipt["id_multiplier"] == copy_id_table.radix("cgt_support_split")
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
        first, second = copy_id_table.mint([2, 2], "cgt_support_split", [1, 2])
        weights[ids == first] += 1.0
        weights[ids == second] -= 1.0
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


#: Both income bands of the split scenario ask for far more than their pools,
#: so every positive-weight household is selected at every clone count.
_EXHAUSTING_TOP_CELLS = {0: {250_000: 1e6}, 37_700: {250_000: 1e6}}
_SPLIT_SCENARIO = {
    "weights": _SPLIT_WEIGHTS,
    "wealth": _SPLIT_WEALTH,
    "incomes": _SPLIT_INCOMES,
    "top_cells": _EXHAUSTING_TOP_CELLS,
}
#: The nesting scenario of the CGT support tests (test_uk_cgt_support.py),
#: nine one-person households. Band 0 holds exact multiples of 60 (900,
#: 1,800, 4,020 and 7,620: 15, 30, 67 and 127 copies), a weight one ulp below
#: 300 (5 copies), a light one (40, uncopied) and a non-multiple (1,234.5, id
#: 7, 21 copies); band 0's pool is exhausted at every clone count, band
#: 37,700 asks for nothing, so its household (9,000) is the first candidate
#: for the exact-total correction, and the zero-weight household is outside
#: every pool. At 7 and 13 clones no household the split leaves whole (the
#: 9,000 household's clones and the light one's) can carry the correction, so
#: the shared fallback puts it on the last copy of the 7,620 family's last
#: clone, off w / n by about an ulp of the frame's total weight (a relative
#: error below 1e-12, within the 1e-12 tolerances); at 1, 3, 11 and 15 clones
#: no correction is needed.
_NESTING_SCENARIO = {
    "weights": [
        900.0,
        1_800.0,
        4_020.0,
        7_620.0,
        float(np.nextafter(300.0, 0.0)),
        40.0,
        1_234.5,
        9_000.0,
        0.0,
    ],
    "wealth": [7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1e3, 1.0, 1e6],
    "incomes": [BAND_INCOMES[0]] * 7 + [BAND_INCOMES[37_700], BAND_INCOMES[0]],
    "top_cells": {0: {250_000: 1_000_000.0}},
}


def _clone_first_split_then_cloned(clone_count: int, scenario=_SPLIT_SCENARIO):
    """A scenario as ``clone_count`` geographic clones, split then cloned."""

    frame = clone_first_frame(
        support_frame(
            weights=scenario["weights"],
            wealth=scenario["wealth"],
            incomes=scenario["incomes"],
        ),
        clone_count,
    )
    split = split_cgt_support_households(
        frame,
        distribution=support_distribution(scenario["top_cells"]),
        parameters=PARAMETERS,
        clone_count=geographic_clone_count(frame.table("household")),
    )
    cloned = clone_cgt_incidence(
        split.frame, distribution=load_advani_summers_distribution()
    )
    return split, cloned.frame


def _exhausting_receipt(
    frame, scenario=_SPLIT_SCENARIO
) -> tuple[dict[str, object], dict[str, object]]:
    problems: dict[str, object] = {}
    receipt = _load_tool()._e8_support_split(
        frame,
        problems,
        distribution=support_distribution(scenario["top_cells"]),
        parameters=PARAMETERS,
        permutation_seed=7,
    )
    return receipt, problems


def _ulps_above(value: float, ulps: int) -> float:
    for _ in range(ulps):
        value = float(np.nextafter(value, np.inf))
    return value


def _floor_plus_one(weights, maximum_copy_weight, *, clone_count):
    """A stage bug: ``floor(q) + 1`` copies for ``q = K x w / 60``, not ``ceil(q)``.

    It differs from the stage only where ``q`` is an integer, where it splits
    the family consistently into one copy more, root and copies together,
    every member at ``w / (n + 1)`` and indexed in full.
    """

    quotient = (
        clone_count * np.asarray(weights, dtype=float) / float(maximum_copy_weight)
    )
    return np.maximum(np.floor(quotient).astype("int64") + 1, 1)


def _split_with_floor_plus_one(monkeypatch, clone_count: int, scenario):
    """``_clone_first_split_then_cloned`` with the split counting ``floor(q) + 1``."""

    from microcosm.build.uk_runtime import cgt_support

    with monkeypatch.context() as patched:
        patched.setattr(cgt_support, "cgt_support_copy_counts", _floor_plus_one)
        return _clone_first_split_then_cloned(clone_count, scenario)


def _hold_reaches(frame, scenario, tool=None) -> tuple[dict, dict, dict[int, int]]:
    """The support-split receipt and problems, with the hold's reach by family.

    The reach counts map how many floats the hold reached around a family's
    shared pair sum (0 where it took the family exactly) to how many
    families it held so; a family it did not hold is not counted.
    """

    tool = _load_tool() if tool is None else tool
    dividing_to = tool._floats_dividing_to
    reaches: list[np.ndarray] = []

    def recorded(quotients, divisors, floats):
        reaches.append(np.asarray(floats))
        return dividing_to(quotients, divisors, floats)

    tool._floats_dividing_to = recorded
    problems: dict[str, object] = {}
    try:
        receipt = tool._e8_support_split(
            frame,
            problems,
            distribution=support_distribution(scenario["top_cells"]),
            parameters=PARAMETERS,
            permutation_seed=7,
        )
    finally:
        tool._floats_dividing_to = dividing_to
    reach = np.concatenate(reaches) if reaches else np.zeros(0, dtype=np.int64)
    values, counts = np.unique(reach, return_counts=True)
    return receipt, problems, dict(zip(values.tolist(), counts.tolist(), strict=True))


def _held_families(frame, scenario) -> tuple[dict, dict, int, int]:
    """The support-split receipt and problems, with the hold's reach.

    Also returns how many families the hold took exactly and how many to
    within one float or two; where no family is held, both are 0.
    """

    receipt, problems, reaches = _hold_reaches(frame, scenario)
    within = sum(count for reach, count in reaches.items() if reach > 0)
    return receipt, problems, reaches.get(0, 0), within


def _through_the_residential_split(twinned):
    """A split-then-cloned frame through stages 44 to 46, arms folded away.

    A hand redraw stands in for stage 44: each clone carrier the clone stage
    gave a prior takes, by its person id, a liable gain, a gain below the
    annual exempt amount, a loss or nothing, so the #970 anchor (45) finds
    all four of its groups. The residential split (46) runs on
    Table 8 totals sized to the frame's liable mass, as in
    ``_residentially_split``, and its arms are then folded onto their
    sources, the frame the E8 entry receipts the support split on.
    """

    import dataclasses

    from microcosm.build.uk_runtime.cgt_asset_type import (
        CGT_STOCK_HOUSEHOLD_COLUMNS,
        CGT_STOCK_PERSON_COLUMNS,
        load_hmrc_cgt_asset_type_facts,
    )
    from microcosm.build.uk_runtime.cgt_residential_split import (
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
        split_cgt_residential_households,
    )
    from microcosm.build.uk_runtime.cgt_structure import anchor_cgt_incidence
    from microcosm.build.uk_runtime.national_frame import uk_household_weight_kind

    person = twinned.table("person").copy()
    household = twinned.table("household").copy()
    for column in CGT_STOCK_PERSON_COLUMNS:
        person[column] = person.get(column, 0.0)
    for column in CGT_STOCK_HOUSEHOLD_COLUMNS:
        household[column] = household.get(column, 0.0)
    ids = person["person_id"].to_numpy()
    group = ids % 4
    gains = np.select(
        [group == 0, group == 1, group == 2],
        [20_000.0 + 10.0 * (ids % 7), 500.0 + (ids % 11), -300.0 - (ids % 5)],
        0.0,
    )
    carrier = person["capital_gains"].to_numpy() != 0.0
    person["capital_gains"] = np.where(carrier, gains, 0.0)
    anchored = anchor_cgt_incidence(
        uk_national_frame(
            person=person,
            benunit=twinned.table("benunit"),
            household=household,
            time_period="2024",
            weight_kind=uk_household_weight_kind(twinned),
            household_weights=twinned.weights_for("household").values,
            mass_log=twinned.mass_log,
        ),
        distribution=load_advani_summers_distribution(),
        parameters=PARAMETERS,
    ).frame
    anchored_person = anchored.table("person")
    by_household = pd.Series(
        anchored.weights_for("household").values,
        index=anchored.table("household")["household_id"],
    )
    person_weight = anchored_person["person_household_id"].map(by_household)
    liable_gains = anchored_person["capital_gains"]
    liable = liable_gains > PARAMETERS.annual_exempt_amount
    count = 0.3 * float(person_weight[liable].sum())
    total = 0.3 * float((person_weight[liable] * liable_gains[liable]).sum())
    facts = dataclasses.replace(
        load_hmrc_cgt_asset_type_facts(),
        table8a_taxpayers_total=count,
        table8a_gains_total=total,
        table8b_individuals_taxpayers=count,
        table8b_individuals_gains=total,
        table8b_all_taxpayers=count,
        table8b_all_gains=total,
    )
    with_arms = split_cgt_residential_households(
        anchored, facts=facts, parameters=PARAMETERS
    ).frame
    return _load_tool()._drop_stacked_layers(
        with_arms, [HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE]
    )


class TestSupportSplitRecomputeAcrossGeographicClones:
    """The recompute counts copies on the frame's geographic clone count.

    The stage reads ``K`` from its input and counts ``ceil(K x w / 60)``
    copies (microcosm#931); the tool reads ``K`` from the pre-split table the
    same way. At one clone its receipt is the one the rule before ``K`` gave.
    The fold re-sums the copies, so neither the count check nor the family
    test recounts it. At any clone count a family of more than three members
    is held exactly where at least four of its members show their two clone
    halves at equal weights and every member that does shares one pair sum
    ``c``, whatever the others carry: the stored ``n`` must be the count of a
    weight that divides by ``n`` to ``c``. Where every member shares ``c`` but
    fewer show equal halves, the weight may divide to within one float of
    ``c``, or two where ``c`` sits that close to a power of two, a reach that
    at one clone can take in the multiple of 60 itself, where a family split
    consistently into one copy more goes unreported. Every other family, and
    whether each household is a family at all, is held to the family weights'
    relative tolerance of 1e-9, the counts at the folded weight times
    ``1 - 1e-9`` and ``1 + 1e-9``, unless the hold decides. That bracket opens
    for a household whose source-scaled weight ``K x w`` lies within a
    relative 1e-9 of a multiple of 60 (at ``K = 1``, its own weight), where it
    cannot tell ``n`` copies from ``n + 1``, or a family of two from a whole
    household: a stored count moved on its own is left to the flags and the
    family weights, and a household split consistently into one copy more
    goes unreported.
    """

    def test_the_recompute_walks_a_tie_at_the_crossing_as_the_stage_did(
        self, monkeypatch
    ) -> None:
        # At three clones the second household's clones tie on wealth across
        # band 0's support mass of 1,500, and the stage takes the one its keyed
        # uniform puts first, not clone 0 (microcosm#931). The recompute walks
        # under the declared tie-break too, and agrees; walked by id, as the
        # rule before did, it takes clone 0 and reports the family the stage
        # made as missing and clone 0 as one it did not make.
        from microcosm.build.uk_runtime import cgt_support

        scenario = {
            "weights": [900.0, 1_800.0, 600.0],
            "wealth": [3e3, 2e3, 1e3],
            "incomes": [BAND_INCOMES[0]] * 3,
            "top_cells": {0: {250_000: 600.0}},
        }
        split, frame = _clone_first_split_then_cloned(3, scenario)
        assert split.households_selected == 4
        household = frame.table("household")
        families = household.loc[
            household[CGT_SUPPORT_COPIES_COLUMN] == 30, "household_clone_index"
        ]
        assert set(families) == {1}
        calls: list[tuple[int, str]] = []
        select = cgt_support.select_cgt_support_households

        def recording(*args, **kwargs):
            calls.append((kwargs["seed"], kwargs["salt"]))
            return select(*args, **kwargs)

        monkeypatch.setattr(cgt_support, "select_cgt_support_households", recording)
        _, problems = _exhausting_receipt(frame, scenario)
        assert problems == {}
        assert calls and set(calls) == {cgt_support.cgt_support_tie_break()}
        monkeypatch.setattr(
            cgt_support,
            "keyed_tie_order",
            lambda primary, ids, **_: np.lexsort((np.asarray(ids), primary)),
        )
        _, by_id = _exhausting_receipt(frame, scenario)
        assert by_id["support_split_selection_stored"] == {"missing": 1, "extra": 1}

    def test_at_one_clone_the_receipt_is_the_rule_before_the_clone_count(
        self, monkeypatch
    ) -> None:
        from microcosm.build.uk_runtime import cgt_support

        _, frame = _split_then_cloned()
        receipt, problems = TestE8SupportSplitRecompute._receipt(frame)

        def ceiling_before_the_clone_count(
            weights, maximum_copy_weight, *, clone_count
        ):
            assert clone_count == 1
            values = np.asarray(weights, dtype=float)
            return np.maximum(
                np.ceil(values / float(maximum_copy_weight)).astype("int64"), 1
            )

        monkeypatch.setattr(
            cgt_support, "cgt_support_copy_counts", ceiling_before_the_clone_count
        )
        before, before_problems = TestE8SupportSplitRecompute._receipt(frame)

        assert problems == before_problems == {}
        assert receipt == before
        # No key is new: the clone count is a stage receipt key only.
        assert set(receipt) == {
            "mass",
            "id_multiplier",
            "pre_split_households",
            "copies",
            "copy_index_column_stored",
            "families",
            "max_abs_family_weight_diff",
            "bands",
            "support_mass",
            "households_selected",
            "copies_recomputed",
        }

    def test_at_three_clones_the_recompute_matches_the_stored_layer(self) -> None:
        one_split, one = _clone_first_split_then_cloned(1)
        three_split, three = _clone_first_split_then_cloned(3)
        assert one_split.evidence()["parameters"]["clone_count"] == 1
        assert three_split.evidence()["parameters"]["clone_count"] == 3

        one_receipt, one_problems = _exhausting_receipt(one)
        three_receipt, three_problems = _exhausting_receipt(three)

        assert one_problems == three_problems == {}
        # Five of the six households are divided at one clone (30 is not);
        # each of their three clones is divided the same way.
        assert one_receipt["families"] == 5
        for key in ("families", "copies", "copies_recomputed", "pre_split_households"):
            assert three_receipt[key] == 3 * one_receipt[key], key

    def test_a_recompute_that_ignored_the_clone_count_would_report_it(
        self, monkeypatch
    ) -> None:
        from microcosm.build.uk_runtime import cgt_support

        _, three = _clone_first_split_then_cloned(3)
        monkeypatch.setattr(cgt_support, "geographic_clone_count", lambda household: 1)

        _, problems = _exhausting_receipt(three)

        # At one third of their weight no clone needs a copy, so none of the
        # 15 stored families is recomputed.
        assert problems == {
            "support_split_selection_stored": {"missing": 15, "extra": 0}
        }

    @pytest.mark.parametrize("clone_count", [7, 11, 13, 15])
    def test_exact_multiples_of_60_receipt_green_at_any_clone_count(
        self, clone_count: int
    ) -> None:
        # At 11 and 13 clones the fold re-sums the 7,620 family's 127 copies
        # to a weight whose K x w / 60 sits more ulps above 127 than the
        # count's snap absorbs, so an exact recount would count 128 in all 11
        # clones at 11 and in 12 of the 13 at 13 (the clone whose last copy
        # took the exact-total correction, which no household left whole
        # could carry there, re-sums to 127). The hold takes every family
        # exactly on its members' shared weight and accepts the stage's 127,
        # except, at 7 and 13 clones, that last clone's 7,620 family: its
        # corrected copy shows equal halves off the others' weight, so the
        # family falls to the bracket, which accepts 127 too.
        one_split, one = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
        split, many = _clone_first_split_then_cloned(clone_count, _NESTING_SCENARIO)

        one_receipt, one_problems = _exhausting_receipt(one, _NESTING_SCENARIO)
        receipt, problems, exact, within = _held_families(many, _NESTING_SCENARIO)

        assert one_problems == problems == {}
        assert one_receipt["families"] == 6
        assert one_receipt["copies"] == one_split.copies_created == 259
        assert receipt["families"] == clone_count * 6
        assert receipt["copies"] == split.copies_created == clone_count * 259
        assert exact == clone_count * 6 - (clone_count in (7, 13))
        assert within == 0

    @pytest.mark.parametrize("clone_count", [1, 3, 11])
    def test_a_count_moved_off_a_non_multiple_is_reported(
        self, clone_count: int
    ) -> None:
        # Household 7 (1,234.5: 21 copies, K x w / 60 = 20.575) lies far
        # from an integer, so a count of 20 or 22 is outside the tolerance.
        _, frame = _clone_first_split_then_cloned(clone_count, _NESTING_SCENARIO)
        for moved in (20, 22):
            household = frame.table("household").copy()
            household.loc[household["household_id"] == 7, CGT_SUPPORT_COPIES_COLUMN] = (
                moved
            )

            _, problems = _exhausting_receipt(
                _rebuild(frame, household=household), _NESTING_SCENARIO
            )

            assert problems["support_split_copies_stored"] == 1, moved

    @pytest.mark.parametrize("clone_count", [1, 3])
    def test_a_count_moved_on_an_exact_multiple_is_left_to_the_flags(
        self, clone_count: int
    ) -> None:
        # Household 1 (900: 15 copies) moved to 16 on its root alone. Its 15
        # members no longer match that count, so the family is not held at any
        # clone count, and K x w / 60 is exactly 15, where the bracket cannot
        # tell 15 from 16 (microcosm#931 review). The count check passes; the
        # flags and the family weights report the family instead.
        _, frame = _clone_first_split_then_cloned(clone_count, _NESTING_SCENARIO)
        household = frame.table("household").copy()
        household.loc[household["household_id"] == 1, CGT_SUPPORT_COPIES_COLUMN] = 16

        _, problems = _exhausting_receipt(
            _rebuild(frame, household=household), _NESTING_SCENARIO
        )

        assert "support_split_copies_stored" not in problems
        assert problems["support_split_flags"] == {
            "copies_disagreeing_with_root": 14,
            "families_with_missing_or_extra_copies": 1,
        }
        assert problems["support_split_family_weights"] == 15

    def test_at_one_clone_a_family_split_into_one_more_copy_on_a_multiple_is_reported(
        self, monkeypatch
    ) -> None:
        # The floor(q) + 1 stage splits the four exact multiples (900, 1,800,
        # 4,020 and 7,620) consistently into one copy more, root and copies
        # together, every member at w / (n + 1). Each family's members still
        # carry one weight, and every weight that divides by n + 1 to it
        # counts n copies, so the K = 1 hold reports all four. The bracket,
        # which the tool before applied here, reported none.
        split, frame = _split_with_floor_plus_one(monkeypatch, 1, _NESTING_SCENARIO)

        receipt, problems, exact, within = _held_families(frame, _NESTING_SCENARIO)

        assert problems == {"support_split_copies_stored": 4}
        assert receipt["copies"] == split.copies_created == 259 + 4
        assert exact == receipt["families"] == 6
        assert within == 0

    @pytest.mark.parametrize("clone_count", [3, 7, 13, 15])
    def test_beyond_one_clone_a_family_split_into_one_more_copy_on_a_multiple_is_reported(
        self, monkeypatch, clone_count: int
    ) -> None:
        # The same four families, split into one copy more in every clone. The
        # stage counts n on each clone's own weight w and gives every member
        # fl(w / n), so the hold applies at any clone count: each family's
        # members still share one weight, every weight that divides by n + 1
        # to it counts n copies under the snapped count, and the hold reports
        # all four in every clone. The bracket, which the tool before applied
        # at K > 1, reported none.
        split, frame = _split_with_floor_plus_one(
            monkeypatch, clone_count, _NESTING_SCENARIO
        )

        receipt, problems, exact, within = _held_families(frame, _NESTING_SCENARIO)

        assert problems == {"support_split_copies_stored": 4 * clone_count}
        assert receipt["copies"] == split.copies_created == clone_count * (259 + 4)
        assert exact == receipt["families"] == 6 * clone_count
        assert within == 0

    def test_at_one_clone_families_one_to_three_ulps_above_a_multiple_stay_green(
        self,
    ) -> None:
        # Weights one to three ulps above a multiple of 60 count one copy more
        # than the multiple, and the fold's re-sum of their copies can land
        # back on the multiple, so an exact recount of the fold misses two
        # copies here. Every member carries fl(w / n), so the K = 1 hold
        # accepts each family's count exactly.
        scenario = {
            "weights": [
                _ulps_above(900.0, 1),
                _ulps_above(1_800.0, 2),
                _ulps_above(4_020.0, 3),
                _ulps_above(7_620.0, 1),
                _ulps_above(300.0, 2),
                40.0,
                9_000.0,
            ],
            "wealth": [7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1.0],
            "incomes": [BAND_INCOMES[0]] * 6 + [BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        split, frame = _clone_first_split_then_cloned(1, scenario)

        receipt, problems, exact, within = _held_families(frame, scenario)

        assert problems == {}
        assert exact == receipt["families"] == 5
        assert within == 0
        assert receipt["copies"] == split.copies_created == 244
        assert receipt["copies_recomputed"] == 242

    def test_at_one_clone_a_family_with_a_corrected_member_is_held_on_its_equal_halves(
        self, monkeypatch
    ) -> None:
        # One copy of the 900 family, which the floor(q) + 1 stage split into
        # 16, carries a rounding amount more on its original half, as where
        # the incidence clone's exact-total correction falls on a member. Its
        # members no longer carry one weight, but the other 15 still show
        # their two clone halves at equal weights and share w / 16, so the
        # hold takes the family exactly on them and reports it with the
        # other three. The tool before held only families whose members all
        # carried one weight, and left this one to the bracket.
        split, frame = _split_with_floor_plus_one(monkeypatch, 1, _NESTING_SCENARIO)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        (copy,) = copy_id_table.mint([1], "cgt_support_split", [1])
        weights[ids == copy] *= 1.0 + 1e-12

        _, problems, exact, within = _held_families(
            _rebuild(frame, household_weights=weights), _NESTING_SCENARIO
        )

        assert problems == {"support_split_copies_stored": 4}
        assert exact == 6
        assert within == 0

    @pytest.mark.parametrize(
        ("stage", "equal", "reported", "exact"),
        [
            ("floor_plus_one", 4, 4, 6),
            ("floor_plus_one", 3, 3, 5),
            ("correct", 4, 0, 6),
            ("correct", 3, 0, 5),
        ],
    )
    def test_at_one_clone_four_equal_halves_vouch_for_a_family_whose_others_differ(
        self, monkeypatch, stage: str, equal: int, reported: int, exact: int
    ) -> None:
        # Every member of the 900 family but ``equal`` of them has its halves
        # re-split unevenly with its pair sum one float above its stage-time
        # weight, as the #970 anchor's roundings can leave a trimmed pair, so
        # its members carry two weights. Four members left with equal halves
        # sharing one weight vouch for w / n: at most three members carry a
        # corrected weight, so one of the four does not, and the hold takes
        # the family exactly on them. It then reports the floor(q) + 1
        # stage's 16 and accepts the correct stage's 15. With three left, any
        # of them may be corrected, the family falls to the bracket, and the
        # floor(q) + 1 family goes unreported with the bracket's blind spot.
        if stage == "floor_plus_one":
            _, frame = _split_with_floor_plus_one(monkeypatch, 1, _NESTING_SCENARIO)
            count = 16
        else:
            _, frame = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
            count = 15
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        members = np.r_[
            np.int64(1),
            copy_id_table.mint([1] * (count - 1), "cgt_support_split", range(1, count)),
        ]
        clones = copy_id_table.mint(members, "cgt_incidence_clone", 1)
        for member, clone in list(zip(members, clones, strict=True))[equal:]:
            moved = float(
                np.nextafter(
                    weights[ids == member][0] + weights[ids == clone][0], np.inf
                )
            )
            weights[ids == clone] = 16.0
            weights[ids == member] = moved - 16.0

        _, problems, held_exactly, within = _held_families(
            _rebuild(frame, household_weights=weights), _NESTING_SCENARIO
        )

        assert problems == (
            {"support_split_copies_stored": reported} if reported else {}
        )
        assert held_exactly == exact
        assert within == 0

    def test_at_one_clone_the_exact_tier_takes_its_weight_from_the_equal_halves(
        self,
    ) -> None:
        # The correct stage splits a household a float above 900 into 16,
        # every member at w / 16. One member's halves are re-split unevenly
        # with its pair sum one float below that weight, so the least pair sum
        # over the family's members lies a float below w / 16. The other 15
        # show equal halves sharing w / 16, so the hold takes the family
        # exactly on their weight and accepts the stage's 16. Taking the least
        # pair sum over every member instead would divide to the float below,
        # whose weight 900 counts 15 copies, and report the correct stage.
        scenario = {
            "weights": [_ulps_above(900.0, 1), 9_000.0],
            "wealth": [7e3, 1.0],
            "incomes": [BAND_INCOMES[0], BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        split, frame = _clone_first_split_then_cloned(1, scenario)
        count = 16
        assert split.copies_created == count - 1
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        members = np.r_[
            np.int64(1),
            copy_id_table.mint([1] * (count - 1), "cgt_support_split", range(1, count)),
        ]
        clones = copy_id_table.mint(members, "cgt_incidence_clone", 1)
        member, clone = members[-1], clones[-1]
        shared = weights[ids == member][0] + weights[ids == clone][0]
        below = float(np.nextafter(shared, 0.0))
        weights[ids == clone] = 16.0
        weights[ids == member] = below - 16.0
        assert weights[ids == member][0] + weights[ids == clone][0] == below

        _, problems, reaches = _hold_reaches(
            _rebuild(frame, household_weights=weights), scenario
        )

        assert problems == {}
        assert reaches == {0: 1}

    @pytest.mark.parametrize("clone_count", [1, 3])
    def test_a_family_of_three_or_fewer_is_left_to_the_bracket(
        self, monkeypatch, clone_count: int
    ) -> None:
        # Each of the three exact-total corrections after the split moves one
        # row, so in a family of three or fewer every member may carry a
        # corrected weight, and none vouches for w / n: the family is held to
        # the bracket at any clone count. Under the floor(q) + 1 stage the
        # 120 family becomes three members in every clone and goes
        # unreported; the 240 family becomes five and is reported. The
        # correct stage receipts green.
        assert _load_tool()._SUPPORT_SPLIT_CORRECTED_ROWS == 3
        scenario = {
            "weights": [120.0, 240.0, 9_000.0],
            "wealth": [7e3, 6e3, 1.0],
            "incomes": [BAND_INCOMES[0]] * 2 + [BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        _, correct = _clone_first_split_then_cloned(clone_count, scenario)
        _, mutated = _split_with_floor_plus_one(monkeypatch, clone_count, scenario)

        _, correct_problems = _exhausting_receipt(correct, scenario)
        receipt, problems, exact, within = _held_families(mutated, scenario)

        assert correct_problems == {}
        assert problems == {"support_split_copies_stored": clone_count}
        assert receipt["families"] == 2 * clone_count
        assert exact == clone_count
        assert within == 0

    def test_at_one_clone_members_moved_one_float_are_held_to_within_one_float(
        self,
    ) -> None:
        # Every member of the 900 family (15 at 60) has its halves re-split
        # unevenly, as the #970 anchor re-splits a trimmed pair, with its pair
        # sum one float above 60, as the anchor's roundings can leave it. No
        # member shows its clone pair untouched, so the hold reaches one float
        # either side of the common weight; 900 divides by 15 to 60, within
        # reach, and the stage's 15 stands. Held exactly, the count would be
        # reported, since every weight dividing by 15 to the moved weight
        # counts 16.
        _, frame = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        members = np.r_[
            np.int64(1), copy_id_table.mint([1] * 14, "cgt_support_split", range(1, 15))
        ]
        moved = float(np.nextafter(60.0, np.inf))
        weights[np.isin(ids, members)] = 31.0
        weights[np.isin(ids, copy_id_table.mint(members, "cgt_incidence_clone", 1))] = (
            moved - 31.0
        )
        moved_frame = _rebuild(frame, household_weights=weights)

        _, problems, exact, within = _held_families(moved_frame, _NESTING_SCENARIO)
        exactly = _load_tool()
        exactly._SUPPORT_SPLIT_PAIR_ROUNDING_FLOATS = 0
        exact_problems: dict[str, object] = {}
        exactly._e8_support_split(
            moved_frame,
            exact_problems,
            distribution=support_distribution(_NESTING_SCENARIO["top_cells"]),
            parameters=PARAMETERS,
            permutation_seed=7,
        )

        assert problems == {}
        assert exact == 5
        assert within == 1
        assert exact_problems == {"support_split_copies_stored": 1}

    @pytest.mark.parametrize(
        ("clone_count", "reported"),
        [(1, 3), (3, 12)],
    )
    def test_through_the_anchor_and_the_residential_split_the_hold_stays_sound(
        self, monkeypatch, clone_count: int, reported: int
    ) -> None:
        # The nesting scenario through the support split, the incidence
        # clone, a hand redraw giving the clone carriers gains in all four of
        # the anchor's groups, the #970 anchor and the residential split, with
        # the arms folded away as the E8 entry folds them. The anchor trims
        # some pairs and the arms divide the liable clones, so in every clone
        # two families show fewer than four equal halves and differ (every
        # carrier of the 900 family falls in one trimmed group) and are held
        # to within one float; the other four are held exactly on their equal
        # halves. The correct stage receipts green. Under the floor(q) + 1
        # stage, at one clone three of the four families on a multiple are
        # reported; the fourth, the 900 family, is held to within one float,
        # a reach that takes in the multiple. At three clones all four are
        # reported in every clone: the multiple sits a few floats below the
        # snapped count's step, beyond that reach.
        _, correct = _clone_first_split_then_cloned(clone_count, _NESTING_SCENARIO)
        _, mutated = _split_with_floor_plus_one(
            monkeypatch, clone_count, _NESTING_SCENARIO
        )

        correct_receipt, correct_problems, correct_exact, correct_within = (
            _held_families(_through_the_residential_split(correct), _NESTING_SCENARIO)
        )
        _, problems, exact, within = _held_families(
            _through_the_residential_split(mutated), _NESTING_SCENARIO
        )

        assert correct_problems == {}
        assert correct_receipt["families"] == 6 * clone_count
        assert (correct_exact, correct_within) == (4 * clone_count, 2 * clone_count)
        assert problems == {"support_split_copies_stored": reported}
        assert (exact, within) == (4 * clone_count, 2 * clone_count)

    @pytest.mark.parametrize("member", ["root", "copy"])
    def test_at_one_clone_a_family_folded_just_above_a_multiple_is_green(
        self, member: str
    ) -> None:
        # The root or one copy of household 1 (900: 15 copies of 60) carries a
        # rounding amount more on its original half, as where the incidence
        # clone's exact-total correction falls on a member, so the family
        # folds to just above 900, where an exact recount gives 16. The moved
        # member's halves differ, and the other 14 still show equal halves
        # at 60, so the hold takes the family exactly on them and accepts the
        # stage's 15, and the receipt is green; its exact recount still
        # records the extra copy. Held exactly on the root's weight, the
        # moved root would report the family.
        split, frame = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        (copy,) = copy_id_table.mint([1], "cgt_support_split", [1])
        weights[ids == (1 if member == "root" else copy)] *= 1.0 + 1e-12

        receipt, problems, exact, within = _held_families(
            _rebuild(frame, household_weights=weights), _NESTING_SCENARIO
        )

        assert problems == {}
        assert receipt["copies"] == split.copies_created == 259
        assert receipt["copies_recomputed"] == 260
        assert (exact, within) == (6, 0)

    @pytest.mark.parametrize(
        ("clone_count", "source_weight", "copies", "moved_floats"),
        [
            # 900 at 15 clones: every member at exactly 4, a power of two,
            # moved two floats below it, as far as the bound allows there.
            (15, 900.0, 15, -2),
            # Two floats below 900 at 15 clones: c two floats below 4.
            (15, float(np.nextafter(np.nextafter(900.0, 0.0), 0.0)), 15, 1),
            # Two floats below 448 at 7 clones: c two floats below 8.
            (7, float(np.nextafter(np.nextafter(448.0, 0.0), 0.0)), 8, 1),
            # Two floats below 192 at 3 clones: c three floats below 16.
            (3, float(np.nextafter(np.nextafter(192.0, 0.0), 0.0)), 4, 1),
        ],
    )
    def test_beyond_one_clone_members_near_a_power_of_two_are_held_to_two_floats(
        self, clone_count: int, source_weight: float, copies: int, moved_floats: int
    ) -> None:
        # (45 / K, 60 / K] holds a power of two at these clone counts. Every
        # member of the family has its halves re-split unevenly, as the #970
        # anchor re-splits a trimmed pair, with its pair sum moved off the
        # stage's fl(w / n) and onto a weight within two floats of the power,
        # so the hold reaches two floats either side. The stage's weight lies
        # within that reach: one float either way is the most the anchor's
        # roundings move a pair sum where the floats either side are as wide,
        # and below a power of two they are half as wide. The receipt is
        # green. One float of reach would leave the stage's weight out of the
        # run where c is the power itself.
        tool = _load_tool()
        scenario = {
            "weights": [source_weight, 9_000.0],
            "wealth": [7e3, 1.0],
            "incomes": [BAND_INCOMES[0], BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        split, frame = _clone_first_split_then_cloned(clone_count, scenario)
        assert split.copies_created == clone_count * (copies - 1)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        roots = np.r_[
            np.int64(1),
            copy_id_table.mint(
                [1] * (clone_count - 1), "geographic_support", range(1, clone_count)
            ),
        ]
        for root in roots:
            members = np.r_[
                root,
                copy_id_table.mint(
                    [root] * (copies - 1), "cgt_support_split", range(1, copies)
                ),
            ]
            clones = copy_id_table.mint(members, "cgt_incidence_clone", 1)
            stage_weight = 2.0 * weights[ids == root][0]
            moved = stage_weight
            for _ in range(abs(moved_floats)):
                moved = float(np.nextafter(moved, np.inf if moved_floats > 0 else 0.0))
            assert tool._pair_rounding_floats(np.array([moved])).tolist() == [2]
            clone_half = float(2.0 ** (np.floor(np.log2(moved)) - 2))
            weights[np.isin(ids, members)] = moved - clone_half
            weights[np.isin(ids, clones)] = clone_half
        if moved_floats == -2:
            least, greatest = tool._floats_dividing_to(
                np.array([moved]), np.array([copies]), np.array([1])
            )
            assert not least[0] <= stage_weight * copies <= greatest[0]

        _, problems, reaches = _hold_reaches(
            _rebuild(frame, household_weights=weights), scenario, tool
        )

        assert problems == {}
        assert reaches == {2: clone_count}

    def test_a_household_the_correction_would_divide_is_passed_over(
        self, monkeypatch
    ) -> None:
        # Every household is selected, and the heaviest left at one copy
        # weighs 60: the correction it could carry would lift it to a weight
        # that counts two copies (test_uk_cgt_support.py). The split passes
        # it over (microcosm#931), so its weight stays 60 and the recompute
        # divides the households the stage divided. Had the 60 household
        # carried the correction, it would be a whole household at a weight
        # that counts two copies. The family test holds a household to the
        # bracket, and that weight lies within a relative 1e-9 of 60, where
        # the bracket cannot tell a family of two from a whole household, so
        # the receipt would be green as well (microcosm#931 review): only the
        # stage's own test sees the check.
        from microcosm.build.uk_runtime import cgt_support

        scenario = {
            "weights": [400.0, 268.7, 60.0, 31.5],
            "wealth": [5.0, 6.0, 7.0, 8.0],
            "incomes": [BAND_INCOMES[0]] * 4,
            "top_cells": {0: {250_000: 1_000.0}},
        }

        def stage_weight(split) -> float:
            household = split.frame.table("household")
            weights = split.frame.weights_for("household").values
            return float(weights[household["household_id"].to_numpy() == 3][0])

        split, frame = _clone_first_split_then_cloned(1, scenario)
        _, problems = _exhausting_receipt(frame, scenario)
        assert problems == {}
        assert stage_weight(split) == 60.0

        correct = cgt_support._exact_support_weights

        def without_the_check(values, *, target, whole, keeps_the_split=None):
            return correct(values, target=target, whole=whole)

        monkeypatch.setattr(cgt_support, "_exact_support_weights", without_the_check)
        unchecked_split, unchecked = _clone_first_split_then_cloned(1, scenario)
        monkeypatch.undo()
        _, problems = _exhausting_receipt(unchecked, scenario)
        assert problems == {}
        assert stage_weight(unchecked_split) > 60.0
        assert cgt_support.cgt_support_copy_counts(
            [stage_weight(unchecked_split)], 60.0, clone_count=1
        ).tolist() == [2]

    def test_a_two_copy_family_that_folds_onto_a_multiple_is_not_reported_missing(
        self,
    ) -> None:
        # A household one float above 60 is divided into two copies, each at
        # 30 plus a float. One member's pair sum then moves one float down,
        # as the #970 anchor's roundings can move a trimmed pair, and the
        # family's fold re-sums to 60, where an exact recount gives one copy:
        # the tool before reported the family missing, though the stage made
        # it right (microcosm#931 review). The family test now holds it to
        # the bracket, which counts two copies at the fold times 1 + 1e-9, so
        # it accepts the stored family, and its count of two, and the receipt
        # is green; the exact recount still records the lost copy.
        scenario = {
            "weights": [_ulps_above(60.0, 1), 9_000.0],
            "wealth": [7e3, 1.0],
            "incomes": [BAND_INCOMES[0], BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        split, frame = _clone_first_split_then_cloned(1, scenario)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        (clone,) = copy_id_table.mint([1], "cgt_incidence_clone", 1)
        assert weights[ids == 1][0] + weights[ids == clone][0] == _ulps_above(30.0, 1)
        weights[ids == 1] = 20.0
        weights[ids == clone] = 10.0
        moved = _rebuild(frame, household_weights=weights)
        assert _load_tool()._pre_split_household_weights(moved)[ids == 1][0] == 60.0

        receipt, problems = _exhausting_receipt(moved, scenario)

        assert problems == {}
        assert receipt["families"] == 1
        assert receipt["copies"] == split.copies_created == 1
        assert receipt["copies_recomputed"] == 0

    @pytest.mark.parametrize("clone_count", [1, 3])
    def test_a_family_dropped_or_added_away_from_a_multiple_is_reported(
        self, monkeypatch, clone_count: int
    ) -> None:
        # A stage that keeps the 100 household whole and divides the 50 one
        # into two copies, in every clone, against the rule's two and one:
        # both weights lie far from a multiple of 60, so the bracket holds
        # each to the rule's count, and the family test reports one family
        # missing and one extra per clone.
        from microcosm.build.uk_runtime import cgt_support

        scenario = {
            "weights": [100.0, 50.0, 9_000.0],
            "wealth": [7e3, 6e3, 1.0],
            "incomes": [BAND_INCOMES[0]] * 2 + [BAND_INCOMES[37_700]],
            "top_cells": {0: {250_000: 1_000_000.0}},
        }
        counted = cgt_support.cgt_support_copy_counts

        def dropped_and_added(weights, maximum_copy_weight, *, clone_count):
            counts = counted(weights, maximum_copy_weight, clone_count=clone_count)
            source = clone_count * np.asarray(weights, dtype=float)
            counts = np.where(np.isclose(source, 100.0), 1, counts)
            return np.where(np.isclose(source, 50.0), 2, counts)

        with monkeypatch.context() as patched:
            patched.setattr(cgt_support, "cgt_support_copy_counts", dropped_and_added)
            _, frame = _clone_first_split_then_cloned(clone_count, scenario)

        receipt, problems = _exhausting_receipt(frame, scenario)

        assert problems == {
            "support_split_selection_stored": {
                "missing": clone_count,
                "extra": clone_count,
            }
        }
        assert receipt["families"] == clone_count

    def test_a_family_whose_members_weigh_nothing_is_reported_not_refused(
        self,
    ) -> None:
        # Every member of the 900 family carries no weight, which no family
        # the split divides does. The hold is asked about every stored
        # family, and its members share one pair sum, but it takes only a
        # positive normal one, so the family falls to the bracket and the
        # family test reports it missing (the rule no longer selects it),
        # where a run of weights stepped from zero would reach below zero.
        _, frame = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        members = np.r_[
            np.int64(1), copy_id_table.mint([1] * 14, "cgt_support_split", range(1, 15))
        ]
        weights[np.isin(ids, members)] = 0.0
        weights[np.isin(ids, copy_id_table.mint(members, "cgt_incidence_clone", 1))] = (
            0.0
        )

        _, problems, exact, within = _held_families(
            _rebuild(frame, household_weights=weights), _NESTING_SCENARIO
        )

        assert problems == {
            "support_split_selection_stored": {"missing": 1, "extra": 0}
        }
        assert (exact, within) == (5, 0)

    def test_a_household_the_correction_would_move_the_walk_past_keeps_it_green(
        self, monkeypatch
    ) -> None:
        # The one-copy household sits an ulp below its band's support mass of
        # 40, and the correction it could carry, which still counts one copy,
        # lifts it onto 40, where the walk rerun stops at it
        # (test_uk_cgt_support.py). The split's rerun of the walk passes it
        # over, so the recompute divides the household the stage divided.
        # With the rerun answering as the stage's walk did, it carries the
        # correction, and the recompute misses that family.
        from microcosm.build.uk_runtime import cgt_support

        scenario = {
            "weights": [float(np.nextafter(40.0, -np.inf)), 82.4],
            "wealth": [9.0, 8.0],
            "incomes": [BAND_INCOMES[0]] * 2,
            "top_cells": {0: {250_000: 16.0}},
        }
        _, frame = _clone_first_split_then_cloned(1, scenario)
        _, problems = _exhausting_receipt(frame, scenario)
        assert problems == {}

        select = cgt_support.select_cgt_support_households
        walks: list[np.ndarray] = []

        def as_the_stage_walked(*args, **kwargs):
            selected, rows = select(*args, **kwargs)
            walks.append(selected)
            return walks[0], rows

        monkeypatch.setattr(
            cgt_support, "select_cgt_support_households", as_the_stage_walked
        )
        _, unchecked = _clone_first_split_then_cloned(1, scenario)
        monkeypatch.undo()
        _, problems = _exhausting_receipt(unchecked, scenario)
        assert len(walks) == 2
        assert problems == {
            "support_split_selection_stored": {"missing": 1, "extra": 0}
        }

    def test_at_one_clone_three_equal_halves_do_not_vouch_for_a_family(
        self,
    ) -> None:
        # Every member of the 900 family (15 at 60) folds one float above 60,
        # as the #970 anchor's roundings can leave them, and three of them
        # show equal halves, as three members the corrections after the split
        # moved could. Three corrections can account for three such members,
        # so none of them vouches for w / n, and the hold reaches one float
        # either side of the common weight: 900 divides by 15 to 60, within
        # reach, and the stage's 15 stands. Were three equal-halves members
        # enough to hold the family exactly, every weight dividing by 15 to
        # the moved weight would count 16, and the family would be reported.
        _, frame = _clone_first_split_then_cloned(1, _NESTING_SCENARIO)
        ids = frame.table("household")["household_id"].to_numpy()
        weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
        members = np.r_[
            np.int64(1), copy_id_table.mint([1] * 14, "cgt_support_split", range(1, 15))
        ]
        clones = copy_id_table.mint(members, "cgt_incidence_clone", 1)
        moved = float(np.nextafter(60.0, np.inf))
        for position, (member, clone) in enumerate(zip(members, clones, strict=True)):
            if position < _load_tool()._SUPPORT_SPLIT_CORRECTED_ROWS:
                weights[ids == member] = moved / 2.0
                weights[ids == clone] = moved / 2.0
            else:
                weights[ids == member] = 31.0
                weights[ids == clone] = moved - 31.0
        moved_frame = _rebuild(frame, household_weights=weights)

        _, problems, exact, within = _held_families(moved_frame, _NESTING_SCENARIO)

        assert problems == {}
        assert (exact, within) == (5, 1)

    def test_the_hold_counts_every_weight_change_after_the_split(self) -> None:
        """The count hold's premises, read from the spine graph and the
        stages' code (microcosm#931 review).

        ``_SUPPORT_SPLIT_CORRECTED_ROWS`` counts the one-row exact-total
        corrections that can land on a support family member between the split
        and the tool's read, and ``_SUPPORT_SPLIT_PAIR_ROUNDING_FLOATS`` (two
        floats near a power of two) how far the rest can move an uncorrected
        one, at any clone count. A spine node changes the
        household weights only where it creates or expands rows or declares
        weights: the executor refuses any other node's weights and frames.
        After the split those are
        the incidence clone, the #970 anchor and the residential split, and
        the boundaries between them keep every row. The split, the clone and
        the residential split each correct their total once, through
        spi_support's helpers, which move one row; the anchor corrects none
        and caps a clone half's factor at one, and refuses gains on a clone
        member other than its carrier (test_uk_cgt_structure.py), so a liable
        clone half takes one arm. Every later stage that changes no weight
        builds its frames on its input's weights, so the legacy stage plan
        moves none either.
        """
        import ast
        import inspect
        from collections import Counter

        from microcosm.build.uk_runtime import (
            cgt_residential_split,
            cgt_structure,
            cgt_support,
            graph_kernels,
            spi_support,
        )
        from microcosm.build.uk_runtime.cgt_structure import CGT_ANCHOR_MAXIMUM_FACTOR
        from microcosm.build.uk_runtime.graph import uk_spine_graph
        from microcosm.graph import StructuralDelta
        from microcosm.graph.decl import compile_graph

        tool = _load_tool()
        graph = uk_spine_graph()
        order = list(compile_graph(graph).order)
        later = order[order.index("cgt_support_split") + 1 :]
        weighing = [
            node_id
            for node_id in later
            if graph.node(node_id).structural
            in {StructuralDelta.CREATE, StructuralDelta.EXPAND}
            or graph.node(node_id).weights is not None
        ]
        assert weighing == [
            "cgt_incidence_clone",
            "cgt_incidence_anchor",
            "cgt_residential_split",
        ]
        boundaries = {
            graph.node(node_id).kernel
            for node_id in later
            if graph.node(node_id).structural is StructuralDelta.FILTER
        }
        assert boundaries == {"uk.identity@1"}

        # Each reference that can correct a frame's total, by the function or
        # method making it. A reference is a bare name or an attribute
        # (spi_support._importance_weights_with_exact_total(...),
        # cgt_support._exact_support_weights(...)), called or not, to one of
        # spi_support's two helpers, a wrapper or an alias:
        # - a wrapper is a function of any of the three stages' modules named
        #   _exact... that refers to one;
        # - an alias is a name one of those modules binds to one outside every
        #   function, at module level or in a class body: imported under
        #   another name (from ... import _importance_weights_with_exact_total
        #   as _total), or assigned from an expression that refers to one
        #   (_total = _importance_weights_with_exact_total, or a
        #   functools.partial of it).
        # Wrappers and aliases are found to a fixed point over the three
        # modules, so one module's wrapper or alias counts where another module
        # refers to it, and a reference to an alias counts as a reference to a
        # wrapper does. Every function, method and nested function counts its
        # own references, not those of the functions it nests. A lambda's
        # reference counts in the function that writes it, and a local alias, a
        # local partial or a local import under another name counts once, where
        # it names the helper or wrapper. What is counted is references, not
        # calls: a loop that corrects twice, or a local alias called twice,
        # counts once.
        helpers = {"_exact_total_correction", "_importance_weights_with_exact_total"}
        stages = ("cgt_support_split", *weighing)
        modules = {graph_kernels._stage_module(stage) for stage in stages}
        assert modules == {cgt_support, cgt_structure, cgt_residential_split}
        defined = ast.FunctionDef | ast.AsyncFunctionDef
        nests = defined | ast.ClassDef

        def collect(node, prefix: str, into: dict[str, ast.AST]) -> None:
            """Every function and method under ``node``, by qualified name."""

            for child in ast.iter_child_nodes(node):
                if isinstance(child, defined):
                    into[prefix + child.name] = child
                    collect(child, f"{prefix}{child.name}.", into)
                elif isinstance(child, ast.ClassDef):
                    collect(child, f"{prefix}{child.name}.", into)
                else:
                    collect(child, prefix, into)

        def referred(function) -> list[str]:
            """The names ``function`` refers to outside the functions it nests:
            each bare name and each attribute read, a call's included, and each
            name it imports under another name."""

            names: list[str] = []
            pending = list(ast.iter_child_nodes(function))
            while pending:
                node = pending.pop()
                if isinstance(node, nests):
                    continue
                if isinstance(node, ast.Name):
                    names.append(node.id)
                elif isinstance(node, ast.Attribute):
                    names.append(node.attr)
                elif isinstance(node, ast.alias) and node.asname is not None:
                    names.append(node.name)
                pending.extend(ast.iter_child_nodes(node))
            return names

        def named(node) -> set[str]:
            """Each bare name and attribute name under ``node``, itself included."""

            return {
                item.id if isinstance(item, ast.Name) else item.attr
                for item in ast.walk(node)
                if isinstance(item, ast.Name | ast.Attribute)
            }

        def outside_functions(node):
            """The statements under ``node`` outside every function: at module
            level, in a block there, or in a class body."""

            for child in ast.iter_child_nodes(node):
                if isinstance(child, defined):
                    continue
                if isinstance(child, ast.stmt):
                    yield child
                yield from outside_functions(child)

        def aliases(tree, counted: set[str]) -> set[str]:
            """The names ``tree`` binds outside every function to a name in
            ``counted``: imported under another name, or assigned from an
            expression that refers to it (every name in the targets)."""

            bound: set[str] = set()
            for node in outside_functions(tree):
                if isinstance(node, ast.ImportFrom):
                    bound |= {
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in counted
                    }
                elif (
                    isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)
                    and node.value is not None
                    and named(node.value) & counted
                ):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    bound |= set().union(*(named(target) for target in targets))
            return bound

        every: dict[str, dict[str, ast.AST]] = {}
        trees: dict[str, ast.Module] = {}
        for module in modules:
            tree = ast.parse(inspect.getsource(module))
            functions: dict[str, ast.AST] = {}
            collect(tree, "", functions)
            assert {
                node.name for node in ast.walk(tree) if isinstance(node, defined)
            } == {function.name for function in functions.values()}
            every[module.__name__] = functions
            trees[module.__name__] = tree
        # The names a reference to which counts: the helpers, the wrappers'
        # names and the aliases, to a fixed point over all three modules.
        counted = set(helpers)
        while True:
            wrappers = {
                (module_name, name)
                for module_name, functions in every.items()
                for name, function in functions.items()
                if function.name.startswith("_exact")
                and set(referred(function)) & counted
            }
            more = (
                helpers
                | {every[module_name][name].name for module_name, name in wrappers}
                | set().union(*(aliases(tree, counted) for tree in trees.values()))
            )
            if more == counted:
                break
            counted = more
        # The wrappers are the support split's and the residential split's own
        # exact-total corrections. Each tries _exact_total_correction on a
        # household its split left whole (the support split on up to 64 in
        # turn, taking the first that carries it) and falls back to
        # _importance_weights_with_exact_total where none does. So each
        # corrects once and refers once to each helper, and a second
        # correction written into one fails here.
        assert {
            (module_name, name): Counter(
                item for item in referred(every[module_name][name]) if item in counted
            )
            for module_name, name in wrappers
        } == {
            (cgt_support.__name__, "_exact_support_weights"): Counter(helpers),
            (cgt_residential_split.__name__, "_exact_residential_weights"): Counter(
                helpers
            ),
        }
        corrections: dict[tuple[str, str], int] = {}
        for module_name, functions in every.items():
            for name, function in functions.items():
                if (module_name, name) in wrappers:
                    continue
                count = sum(item in counted for item in referred(function))
                if count:
                    corrections[(module_name, name)] = count
        assert corrections == {
            (cgt_support.__name__, "split_cgt_support_households"): 1,
            (cgt_structure.__name__, "clone_cgt_incidence"): 1,
            (cgt_residential_split.__name__, "split_cgt_residential_households"): 1,
        }
        assert len(corrections) == tool._SUPPORT_SPLIT_CORRECTED_ROWS
        # spi_support's helpers move one row each.
        rng = np.random.default_rng(3)
        for _ in range(20):
            values = rng.lognormal(3.0, 1.0, 50)
            target = float(values.sum()) * (1.0 + 1e-15)
            for corrected in (
                spi_support._exact_total_correction(
                    values, target=target, correction_index=int(np.argmax(values))
                ),
                np.asarray(
                    spi_support._importance_weights_with_exact_total(
                        values, target
                    ).values
                ),
            ):
                if corrected is not None:
                    assert int((corrected != values).sum()) <= 1
        # The anchor scales a clone half by at most one.
        assert CGT_ANCHOR_MAXIMUM_FACTOR == 1.0
        assert tool._SUPPORT_SPLIT_PAIR_ROUNDING_FLOATS == 1
        assert tool._SUPPORT_SPLIT_PAIR_ROUNDING_FLOATS_NEAR_A_POWER_OF_TWO == 2

        # The stages after the split that change no weight pass their input's
        # household weights to every frame they build.
        passed = {
            "frame.weights_for('household').values",
            "np.asarray(frame.weights_for('household').values, dtype=float)",
        }

        def binding(assignments, name: str, line: int):
            """The value last bound to ``name`` before ``line``."""

            found = [
                value
                for lineno, target, value in assignments
                if target == name and lineno < line
            ]
            return found[-1] if found else None

        built: dict[str, int] = {}
        for stage in later:
            if stage in weighing or stage not in graph_kernels._STAGE_MODULES:
                continue
            built[stage] = 0
            module = graph_kernels._stage_module(stage)
            tree = ast.parse(inspect.getsource(module))
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                assignments = sorted(
                    (
                        (sub.lineno, sub.targets[0].id, sub.value)
                        for sub in ast.walk(function)
                        if isinstance(sub, ast.Assign)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                    ),
                    key=lambda item: item[0],
                )
                for sub in ast.walk(function):
                    if not isinstance(sub, ast.Call):
                        continue
                    name = getattr(sub.func, "id", getattr(sub.func, "attr", None))
                    assert name not in helpers, (stage, function.name)
                    for keyword in sub.keywords:
                        if keyword.arg != "household_weights":
                            continue
                        value = keyword.value
                        if isinstance(value, ast.Name):
                            value = binding(assignments, value.id, sub.lineno) or value
                        elif isinstance(value, ast.Attribute) and isinstance(
                            value.value, ast.Name
                        ):
                            base = binding(assignments, value.value.id, sub.lineno)
                            if base is not None:
                                value = ast.Attribute(base, value.attr, ast.Load())
                        assert ast.unparse(value) in passed, (stage, function.name)
                        built[stage] += 1
        assert len(built) == 9 and all(built.values()), built


# --- copy ids at the copy-id table's declared radices (microcosm#931) ----------
#
# The decoders read every copy id at its field's declared radix, never at a
# power of ten above the frame's largest id, and the tool refuses a spine whose
# copy ids follow the rule the table replaced. Every frame here is synthetic.


def _lineage_household(household: pd.DataFrame) -> pd.DataFrame:
    """A copy chain's path columns as the spine's own lineage columns."""

    from microcosm.build.uk_runtime.cgt_residential_split import (
        CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
        HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    )
    from microcosm.build.uk_runtime.cgt_support import CGT_SUPPORT_COPY_INDEX_COLUMN
    from test_support.microcosm_build.uk_clone_first_frames import COPY_PATH_COLUMNS

    lineage = household.drop(columns=list(COPY_PATH_COLUMNS))
    lineage["household_support_clone_index"] = household["spi"]
    lineage["household_support_channel"] = np.where(household["spi"] > 0, "spi", "frs")
    lineage[HIGH_VALUE_SUPPORT_INDEX_COLUMN] = household["hv"]
    lineage[HIGH_VALUE_SUPPORT_COPY_COLUMN] = household["hv"] > 0
    lineage[CGT_SUPPORT_COPY_INDEX_COLUMN] = household["cgt"]
    lineage[HOUSEHOLD_IS_CGT_SUPPORT_COPY] = household["cgt"] > 0
    lineage[HOUSEHOLD_IS_CGT_CLONE] = household["twin"] > 0
    lineage[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] = household["arm"]
    lineage[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] = household["arm"] > 0
    return lineage


def _chain(rule: str):
    """The six copy layers on synthetic sernums, minted by ``rule``."""

    from test_support.microcosm_build.uk_clone_first_frames import (
        base_id_tables,
        copy_chain,
    )

    return copy_chain(base_id_tables(), rule=rule)


class TestCopyIdsAtTheDeclaredTable:
    def test_a_spine_the_table_minted_follows_it(self) -> None:
        tool = _load_tool()
        household = _lineage_household(_chain("table").household)
        assert tool._copy_id_rows_off_the_declared_table(household) == 0
        tool._require_copy_ids_on_the_declared_table(household)

    def test_a_spine_minted_by_the_retired_rule_is_refused_by_count(self) -> None:
        tool = _load_tool()
        household = _lineage_household(_chain("retired").household)
        off = tool._copy_id_rows_off_the_declared_table(household)
        # Every copy row: the retired rule put each layer's copies a power of
        # ten above the largest id the layer saw, never in the layer's field.
        copies = int((household["household_id"] >= 10**5).sum())
        assert off == copies > 0
        with pytest.raises(ValueError) as caught:
            tool._require_copy_ids_on_the_declared_table(household)
        assert str(caught.value) == f"{off} {tool._OFF_TABLE_REFUSAL}"
        assert "commit that built it" in str(caught.value)

    def test_a_copy_whose_lineage_disagrees_with_its_id_is_refused(self) -> None:
        tool = _load_tool()
        household = _lineage_household(_chain("table").household)
        # One twin loses its flag; its id still holds the twin field's ordinal.
        twin = int(np.flatnonzero(household[HOUSEHOLD_IS_CGT_CLONE].to_numpy())[0])
        household.loc[twin, HOUSEHOLD_IS_CGT_CLONE] = False
        assert tool._copy_id_rows_off_the_declared_table(household) == 1

    def test_the_gate_holds_each_stored_ordinal_to_the_ids_exactly(self) -> None:
        # The SPI channel's ordinal (copy 1, band donor 2) and the high-value
        # copy's are checked against the id only here; a nonzero-marker reading
        # would pass a band donor recorded as the SPI copy.
        tool = _load_tool()
        household = _lineage_household(_chain("table").household)
        for column, stored, tampered_to in (
            ("household_support_clone_index", 2, 1),
            (HIGH_VALUE_SUPPORT_INDEX_COLUMN, 2, 3),
        ):
            tampered = household.copy()
            row = int(np.flatnonzero(tampered[column].to_numpy() == stored)[0])
            tampered.loc[row, column] = tampered_to
            assert tool._copy_id_rows_off_the_declared_table(tampered) == 1, column

    def test_the_gate_expects_nothing_in_the_vintage_field(self) -> None:
        # Nothing mints a vintage ordinal yet, so a base id moved into that
        # field is off the table although no lineage column names the field.
        tool = _load_tool()
        moved = _lineage_household(_chain("table").household)
        moved.loc[0, "household_id"] = int(
            moved.loc[0, "household_id"]
        ) + copy_id_table.radix("vintage")
        assert tool._copy_id_rows_off_the_declared_table(moved) == 1

    def test_the_tool_refuses_a_spine_built_before_the_table(
        self, monkeypatch, tmp_path
    ) -> None:
        import sys

        from test_support.microcosm_build.uk_clone_first_frames import (
            IdTables,
            id_frame,
        )

        tool = _load_tool()
        tables = _chain("retired")
        frame = id_frame(
            IdTables(
                person=tables.person,
                benunit=tables.benunit,
                household=_lineage_household(tables.household),
            )
        )
        off = tool._copy_id_rows_off_the_declared_table(frame.table("household"))
        monkeypatch.setattr(tool, "load_uk_national_frame", lambda path: (frame, {}))
        output = tmp_path / "receipt.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_uk_identity_stability.py",
                "--input-h5",
                str(tmp_path / "spine.h5"),
                "--output",
                str(output),
                "--check",
                "e7",
            ],
        )
        with pytest.raises(ValueError) as caught:
            tool.main()
        assert str(caught.value) == f"{off} {tool._OFF_TABLE_REFUSAL}"
        assert not output.exists()

    def test_the_arms_decode_at_the_declared_radix_above_2_53(self) -> None:
        tool = _load_tool()
        tables = _chain("table")
        path = ["source_household_id", "spi", "hv", "cgt", "twin"]
        lineage = tool._residential_arm_lineage(_lineage_household(tables.household))
        household = tables.household
        arms = household.iloc[lineage.arm_positions]
        assert len(arms) > 0
        assert int(arms["household_id"].min()) > 2**53
        assert lineage.radix == copy_id_table.radix("cgt_residential_split")
        np.testing.assert_array_equal(lineage.arm_index, arms["arm"].to_numpy())
        # Each arm's source is the row with its copy path and no arm, found by
        # lineage, never by id arithmetic.
        sources = household.loc[household["arm"] == 0, path]
        position_by_path = {
            tuple(row): position
            for position, row in zip(
                sources.index, sources.itertuples(index=False), strict=True
            )
        }
        expected = [
            position_by_path[tuple(row)] for row in arms[path].itertuples(index=False)
        ]
        np.testing.assert_array_equal(lineage.source_positions, expected)

    def test_a_stored_arm_index_off_the_id_and_an_orphan_arm_are_refused(
        self,
    ) -> None:
        from microcosm.build.uk_runtime.cgt_residential_split import (
            CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
        )

        tool = _load_tool()
        household = _lineage_household(_chain("table").household)
        arm = int(np.flatnonzero(household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] > 0)[0])
        tampered = household.copy()
        tampered.loc[arm, CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] = 7
        with pytest.raises(ValueError, match="disagrees with the id scheme") as caught:
            tool._residential_arm_lineage(tampered)
        assert str(household["household_id"].iloc[arm]) not in str(caught.value)
        orphan = household.copy()
        source = int(orphan["household_id"].iloc[arm]) - int(
            orphan[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN].iloc[arm]
        ) * copy_id_table.radix("cgt_residential_split")
        orphan = orphan.loc[orphan["household_id"] != source].reset_index(drop=True)
        with pytest.raises(ValueError, match="without a source household") as caught:
            tool._residential_arm_lineage(orphan)
        assert str(source) not in str(caught.value)

    def test_a_support_copy_decodes_exactly_above_2_53(self) -> None:
        from microcosm.build.uk_runtime.cgt_support import (
            CGT_SUPPORT_COPY_INDEX_COLUMN,
        )

        tool = _load_tool()
        # The field's largest ordinal: 1 + 9,999 x 10**13 has no float64 twin,
        # so a float read would name a root the frame does not hold.
        copy = int(copy_id_table.mint(1, "cgt_support_split", 9_999))
        assert copy > 2**53 and float(copy) != copy
        household = pd.DataFrame(
            {
                "household_id": np.array([1, 2, copy], dtype=np.int64),
                HOUSEHOLD_IS_CGT_SUPPORT_COPY: [False, False, True],
                CGT_SUPPORT_COPY_INDEX_COLUMN: np.array([0, 0, 9_999], dtype=np.int64),
            }
        )
        lineage = tool._support_copy_lineage(household)
        assert lineage.copy_index.tolist() == [9_999]
        assert lineage.root_positions.tolist() == [0]
        floated = household.assign(household_id=household["household_id"].astype(float))
        with pytest.raises(TypeError, match="integer household ids"):
            tool._support_copy_lineage(floated)


# --- the receipts on frames the CGT stages built (microcosm#931) -----------------
#
# The split scenario above run through the support split (42), the incidence
# twin (43), a redraw by hand (44: each twin's carrier takes a liable gain),
# the anchor (45) and the residential split (46), with Table 8 totals sized to
# the frame's liable mass. Every value is synthetic.


def _residentially_split():
    """The split scenario after stages 42 to 46, and the Table 8 facts it used."""

    import dataclasses

    from microcosm.build.uk_runtime.cgt_asset_type import (
        CGT_STOCK_HOUSEHOLD_COLUMNS,
        CGT_STOCK_PERSON_COLUMNS,
        load_hmrc_cgt_asset_type_facts,
    )
    from microcosm.build.uk_runtime.cgt_residential_split import (
        split_cgt_residential_households,
    )
    from microcosm.build.uk_runtime.cgt_structure import anchor_cgt_incidence
    from microcosm.build.uk_runtime.national_frame import uk_household_weight_kind
    from test_support.microcosm_build.uk_cgt_support import support_tables

    person, benunit, household, weights = support_tables(
        weights=_SPLIT_WEIGHTS, wealth=_SPLIT_WEALTH, incomes=_SPLIT_INCOMES
    )
    # The residential split reads the asset stocks; those the scenario does
    # not carry are zero.
    for column in CGT_STOCK_PERSON_COLUMNS:
        person[column] = person.get(column, 0.0)
    for column in CGT_STOCK_HOUSEHOLD_COLUMNS:
        household[column] = household.get(column, 0.0)
    split = split_cgt_support_households(
        uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            household_weights=weights,
            time_period="2024",
        ),
        distribution=support_distribution(_SPLIT_TOP_CELLS),
        parameters=PARAMETERS,
        clone_count=1,
    )
    twinned = clone_cgt_incidence(
        split.frame, distribution=load_advani_summers_distribution()
    ).frame
    redrawn_person = twinned.table("person").copy()
    carrier = redrawn_person["capital_gains"].to_numpy() != 0.0
    redrawn_person["capital_gains"] = np.where(
        carrier, 20_000.0 + 1_000.0 * (redrawn_person["person_id"] % 7), 0.0
    )
    anchored = anchor_cgt_incidence(
        uk_national_frame(
            person=redrawn_person,
            benunit=twinned.table("benunit"),
            household=twinned.table("household"),
            time_period="2024",
            weight_kind=uk_household_weight_kind(twinned),
            household_weights=twinned.weights_for("household").values,
            mass_log=twinned.mass_log,
        ),
        distribution=load_advani_summers_distribution(),
        parameters=PARAMETERS,
    ).frame
    anchored_person = anchored.table("person")
    by_household = pd.Series(
        anchored.weights_for("household").values,
        index=anchored.table("household")["household_id"],
    )
    person_weight = anchored_person["person_household_id"].map(by_household)
    gains = anchored_person["capital_gains"]
    liable = gains > PARAMETERS.annual_exempt_amount
    count = 0.3 * float(person_weight[liable].sum())
    total = 0.3 * float((person_weight[liable] * gains[liable]).sum())
    facts = dataclasses.replace(
        load_hmrc_cgt_asset_type_facts(),
        table8a_taxpayers_total=count,
        table8a_gains_total=total,
        table8b_individuals_taxpayers=count,
        table8b_individuals_gains=total,
        table8b_all_taxpayers=count,
        table8b_all_gains=total,
    )
    result = split_cgt_residential_households(
        anchored, facts=facts, parameters=PARAMETERS
    )
    return result.frame, facts


class TestReceiptsOnFramesTheStagesBuilt:
    def test_the_old_spine_gate_passes_every_layer_the_stages_minted(self) -> None:
        tool = _load_tool()
        _, twinned = _split_then_cloned()
        with_arms, _ = _residentially_split()
        for frame in (twinned, with_arms):
            assert (
                tool._copy_id_rows_off_the_declared_table(frame.table("household")) == 0
            )

    def test_the_arms_the_stage_built_decode_and_receipt_green(self) -> None:
        from microcosm.build.uk_runtime.cgt_residential_split import (
            HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
        )

        tool = _load_tool()
        frame, facts = _residentially_split()
        problems: dict[str, object] = {}
        residential = tool._e8_residential_split(
            frame, problems, parameters=PARAMETERS, facts=facts, permutation_seed=7
        )
        assert problems == {}
        radix = copy_id_table.radix("cgt_residential_split")
        assert residential["id_multiplier"] == radix == 10**18
        # One arm for each of the 11 twins (six households and five copies),
        # whose carrier alone gains.
        assert residential["arms"] == residential["households_split"] == 11
        # Every arm's people are its source's people shifted by the arm's
        # index times the radix: the decode found the row the stage copied.
        household = frame.table("household")
        lineage = tool._residential_arm_lineage(household)
        ids = household["household_id"].to_numpy()
        assert int(ids[lineage.arm_positions].min()) > 2**53
        members = (
            frame.table("person")
            .groupby("person_household_id")["person_id"]
            .apply(lambda people: sorted(int(value) for value in people))
        )
        for arm, source, index in zip(
            lineage.arm_positions,
            lineage.source_positions,
            lineage.arm_index,
            strict=True,
        ):
            shifted = [value - int(index) * radix for value in members[ids[arm]]]
            assert shifted == members[ids[source]]
        # The layers below, with the arms folded away, receipt green too.
        below = tool._drop_stacked_layers(frame, [HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE])
        pairs = tool._e8_clone_pairs(below, problems, parameters=PARAMETERS)
        support = tool._e8_support_split(
            below,
            problems,
            distribution=support_distribution(_SPLIT_TOP_CELLS),
            parameters=PARAMETERS,
            permutation_seed=7,
        )
        assert problems == {}
        assert pairs["anchored"] is True and pairs["pairs"] == 11
        assert support["copies"] == 5 and support["families"] == 3


# --- the high-value home copies and the regional home values -----------------
#
# The split scenario's six one-person households, all FRS-base owners, through
# the real stages from the regional uprating on. The uprating (stage 25) sets
# each region's unweighted mean of base owners' values to its public price;
# the high-value home support (stage 26, microcosm#1240) divides the top of
# each region, at stratum 0.6, into three rows at a third of the weight (id 5
# in Wales, ids 4 and 6 in London); the regional home values (stage 27,
# microcosm#1112) read every value off the region's knots and keep the value
# they replaced. With ``cgt`` the CGT support split and the clone follow.

_HOME_VALUES = [100_000.0, 200_000.0, 300_000.0, 400_000.0, 500_000.0, 600_000.0]
_OTHER_PROPERTY = [0.0, 50_000.0, 10_000.0, 0.0, 120_000.0, 30_000.0]


def _home_values_chain(*, cgt: bool, tamper: bool = False) -> SimpleNamespace:
    """The scenario before stage 26, after it, after stage 27, and at the end.

    With ``tamper`` the first household's uprated value and wealth move off
    the uprating's factor before stage 26, as an uprating that stored a value
    off its fixed point would leave them.
    """

    from microcosm.build.uk_runtime.high_value_home_support import (
        split_high_value_support_households,
    )
    from microcosm.build.uk_runtime.regional_home_values import (
        assign_regional_home_values,
    )
    from microcosm.build.uk_runtime.regional_uprating import (
        load_regional_land_values_resource,
        uprate_household_property_by_region,
    )
    from test_support.microcosm_build.uk_cgt_support import support_tables

    person, benunit, household, weights = support_tables(
        weights=_SPLIT_WEIGHTS, wealth=_SPLIT_WEALTH, incomes=_SPLIT_INCOMES
    )
    household["tenure_type"] = "OWNED_OUTRIGHT"
    household["main_residence_value"] = _HOME_VALUES
    household["property_wealth"] = np.add(_HOME_VALUES, _OTHER_PROPERTY)
    household["source_household_id"] = household["household_id"]
    uprated = uprate_household_property_by_region(
        household, load_regional_land_values_resource()
    )
    if tamper:
        uprated.loc[0, ["main_residence_value", "property_wealth"]] *= 1.25
    pre_split = uk_national_frame(
        person=person,
        benunit=benunit,
        household=uprated,
        household_weights=weights,
        time_period="2024",
    )
    split = split_high_value_support_households(
        pre_split, stratum_lower=0.6, copies=3
    ).frame
    valued = assign_regional_home_values(split).frame
    final = valued
    if cgt:
        final = clone_cgt_incidence(
            split_cgt_support_households(
                valued,
                distribution=support_distribution(_SPLIT_TOP_CELLS),
                parameters=PARAMETERS,
                clone_count=1,
            ).frame,
            distribution=load_advani_summers_distribution(),
        ).frame
    return SimpleNamespace(pre_split=pre_split, split=split, valued=valued, final=final)


def _cgt_layers(frame) -> list[str]:
    household = frame.table("household")
    return [
        flag
        for flag in (HOUSEHOLD_IS_CGT_CLONE, HOUSEHOLD_IS_CGT_SUPPORT_COPY)
        if flag in household.columns
    ]


class TestHighValueCopiesAndRegionalHomeValues:
    """E5 reads the uprating where stage 25 left it, on the rows it saw."""

    @pytest.mark.parametrize("cgt", [False, True])
    def test_the_wealth_scope_folds_every_copy_onto_its_root(self, cgt: bool) -> None:
        tool = _load_tool()
        chain = _home_values_chain(cgt=cgt)
        pre_split = chain.pre_split.weights_for("household").values
        # The split leaves each root a third of its weight, its two copies
        # the other thirds.
        split_household = chain.split.table("household")
        families = split_household["high_value_support_copies"].to_numpy()
        assert families.tolist() == [1, 1, 1, 3, 3, 3] + [3] * 6
        np.testing.assert_allclose(
            chain.split.weights_for("household").values[:6],
            pre_split / families[:6],
            rtol=1e-12,
        )
        for scoped in (
            tool._frame_as_stage_saw(chain.final, "was_wealth"),
            tool._frs_only_frame(chain.final),
        ):
            household = scoped.table("household")
            assert household["household_id"].tolist() == [1, 2, 3, 4, 5, 6]
            assert not household[HIGH_VALUE_SUPPORT_COPY_COLUMN].any()
            assert scoped.table("person")["person_id"].tolist() == [1, 2, 3, 4, 5, 6]
            # Each root's copies add their thirds back: its pre-split mass.
            np.testing.assert_allclose(
                scoped.weights_for("household").values, pre_split, rtol=1e-12
            )
        # The travel block runs after the split: its scope keeps the copies.
        travel = tool._frame_as_stage_saw(chain.final, "nts_bus_travel")
        assert int(travel.table("household")[HIGH_VALUE_SUPPORT_COPY_COLUMN].sum()) == 6

    @pytest.mark.parametrize("cgt", [False, True])
    def test_e5_holds_the_uprating_where_it_left_the_values(self, cgt: bool) -> None:
        tool = _load_tool()
        final = _home_values_chain(cgt=cgt).final
        receipt = tool.e5_identity_receipt(
            tool._frame_as_stage_saw(final, "was_wealth"), permutation_seed=7
        )
        assert receipt["columns_by_entity"]["household"] == [
            "main_residence_value",
            "property_wealth",
        ]
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        # Either half of the scope alone stays red: the values stage 27
        # stored, or the copies, FRS-base owners that enter the uprating's
        # mean when they stay in scope.
        stored_values = tool._drop_stacked_layers(
            final, [HIGH_VALUE_SUPPORT_COPY_COLUMN, *_cgt_layers(final)]
        )
        copies_kept = tool._drop_stacked_layers(
            tool._pre_regional_home_values_frame(final), _cgt_layers(final)
        )
        for partial in (stored_values, copies_kept):
            red = tool.e5_identity_receipt(partial, permutation_seed=7)
            assert red["matches_stored_columns"] is False
            assert red["stored_column_mismatches"] == {
                "household": ["main_residence_value", "property_wealth"]
            }

    @pytest.mark.parametrize("cgt", [False, True])
    def test_e5_reports_an_uprated_value_off_its_fixed_point(self, cgt: bool) -> None:
        tool = _load_tool()
        final = _home_values_chain(cgt=cgt, tamper=True).final
        receipt = tool.e5_identity_receipt(
            tool._frame_as_stage_saw(final, "was_wealth"), permutation_seed=7
        )
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert receipt["stored_column_mismatches"] == {
            "household": ["main_residence_value", "property_wealth"]
        }

    @pytest.mark.parametrize("tamper", [False, True])
    def test_the_tool_entry_point_receipts_e5(
        self, tamper: bool, monkeypatch, tmp_path
    ) -> None:
        import json
        import sys

        tool = _load_tool()
        final = _home_values_chain(cgt=True, tamper=tamper).final
        monkeypatch.setattr(tool, "load_uk_national_frame", lambda path: (final, {}))
        output = tmp_path / "receipt.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_uk_identity_stability.py",
                "--input-h5",
                str(tmp_path / "spine.h5"),
                "--output",
                str(output),
                "--check",
                "e5",
            ],
        )
        assert tool.main() == (1 if tamper else 0)
        assert json.loads(output.read_text())["matches_stored_columns"] is not tamper

    def test_the_reversal_restores_what_stage_27_replaced(self) -> None:
        from microcosm.build.uk_runtime.regional_home_values import RANK_BASIS_COLUMN

        tool = _load_tool()
        chain = _home_values_chain(cgt=False)
        before = chain.split.table("household")
        after = chain.valued.table("household")
        assert (
            after["main_residence_value"].to_numpy()
            != before["main_residence_value"].to_numpy()
        ).all()
        reversed_frame = tool._pre_regional_home_values_frame(chain.valued)
        restored = reversed_frame.table("household")
        assert RANK_BASIS_COLUMN not in restored.columns
        np.testing.assert_array_equal(
            restored["main_residence_value"].to_numpy(),
            before["main_residence_value"].to_numpy(),
        )
        np.testing.assert_allclose(
            restored["property_wealth"].to_numpy(),
            before["property_wealth"].to_numpy(),
            rtol=1e-12,
        )
        # Nothing to take back on a frame without the rank basis.
        assert tool._pre_regional_home_values_frame(reversed_frame) is reversed_frame
        assert tool._pre_regional_home_values_frame(chain.split) is chain.split

    @pytest.mark.parametrize("key", ["outputs", "rewrites"])
    def test_the_reversal_refuses_a_later_writer_of_the_values(
        self, key: str, monkeypatch
    ) -> None:
        # The reversal assumes stage 27 writes the two columns last. A roster
        # where a later stage declares one of them fails closed, instead of
        # hiding that stage's rewrite from E5.
        tool = _load_tool()
        stages = tool._source_stage_roster()
        names = [stage["stage"] for stage in stages]
        later = names.index("regional_home_values") + 1
        edited = [dict(stage) for stage in stages]
        edited[later][key] = [*edited[later].get(key, ()), "main_residence_value"]
        valued = _home_values_chain(cgt=False).valued
        assert tool._pre_regional_home_values_frame(valued) is not valued
        monkeypatch.setattr(tool, "_source_stage_roster", lambda: edited)
        with pytest.raises(ValueError, match=r"^1 stage\(s\) after regional_home"):
            tool._pre_regional_home_values_frame(valued)

    def test_dropping_the_copies_under_a_kept_cgt_layer_is_refused(self) -> None:
        tool = _load_tool()
        final = _home_values_chain(cgt=True).final
        with pytest.raises(ValueError, match="requires dropping the CGT layers"):
            tool._drop_stacked_layers(final, [HIGH_VALUE_SUPPORT_COPY_COLUMN])

    def test_a_copy_off_its_lineage_fails_closed_by_count(self) -> None:
        tool = _load_tool()
        household = _home_values_chain(cgt=True).final.table("household")
        lineage = tool._high_value_copy_lineage(household)
        assert len(lineage.copy_positions) == 6
        ids = household["household_id"].to_numpy()
        # The roots' ids are single digits, like the counts; every copy's is
        # long enough to show if a message printed it.
        copy_ids = ids[ids >= copy_id_table.UK_COPY_ID_BASE_RADIX]
        first_copy = int(lineage.copy_positions[0])
        root = int(lineage.root_positions[0])

        index_moved = household.copy()
        index_moved.loc[first_copy, HIGH_VALUE_SUPPORT_INDEX_COLUMN] = 3 - int(
            lineage.copy_index[0]
        )
        unflagged = household.copy()
        unflagged.loc[first_copy, HIGH_VALUE_SUPPORT_COPY_COLUMN] = False
        copy_missing = household.drop(index=first_copy).reset_index(drop=True)
        root_missing = household.drop(index=root).reset_index(drop=True)
        for tampered, message in (
            (
                index_moved,
                "High-value home copy index stored on 1 copy(ies) disagrees with "
                "the id scheme (the id's ordinal in the copy-id table's "
                "'high_value_support' field).",
            ),
            (
                unflagged,
                "High-value home copy flag and stored copy index disagree on 1 "
                "household(s).",
            ),
            (
                copy_missing,
                "'high_value_support_copies' stored on 1 household(s) disagrees "
                "with the copies their ids decode to.",
            ),
            (
                root_missing,
                "2 high-value home copy(ies) without a root household at the "
                "declared radix of the copy-id table's 'high_value_support' field.",
            ),
        ):
            with pytest.raises(ValueError) as caught:
                tool._high_value_copy_lineage(tampered)
            assert str(caught.value) == message
            assert not any(str(int(value)) in str(caught.value) for value in copy_ids)


#: Row-creating stages whose declared flag the tool's stacking map has no
#: entry for, with why it needs none (microcosm#931 review).
_STACKING_MAP_EXCEPTIONS = {
    "spi_income_band_donors": (
        "its donors are rewritten onto household_is_spi_synthetic too, so the "
        "support channel's entry drops them"
    ),
    "cgt_incidence_anchor": (
        "declares only the incidence clone's flag, which it pairs on, and "
        "creates no rows"
    ),
}


def test_the_stacking_map_names_every_structural_stage_flag() -> None:
    """The map's own comment says a new stacking stage must add its flag; this
    holds it to the roster. Each structural stage's declared flag maps to
    that stage, or the stage is an exception listed with its reason."""
    from microcosm.build.uk_runtime.graph import UK_SPINE_STRUCTURAL_STAGES

    tool = _load_tool()
    stages = {
        stage["stage"]: stage
        for stage in tool._source_stage_roster()
        if stage["stage"] in UK_SPINE_STRUCTURAL_STAGES
    }
    assert set(stages) == set(UK_SPINE_STRUCTURAL_STAGES)
    declared = {
        name: [
            operation["flag_column"]
            for operation in stage.get("operations", ())
            if "flag_column" in operation
        ]
        for name, stage in stages.items()
    }
    mapped = {}
    for name, flags in declared.items():
        if name in _STACKING_MAP_EXCEPTIONS:
            continue
        (flag,) = flags
        mapped[flag] = name
    assert mapped == tool._STACKING_STAGE_BY_FLAG
    # The exceptions' reasons, as the roster declares them.
    assert "household_is_spi_synthetic" in stages["spi_income_band_donors"].get(
        "rewrites", ()
    )
    assert declared["cgt_incidence_anchor"] == []
    assert {
        operation.get("clone_flag_column")
        for operation in stages["cgt_incidence_anchor"]["operations"]
    } - {None} == {"household_is_capital_gains_clone"}
