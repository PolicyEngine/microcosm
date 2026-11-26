"""The release tool's repeated December SSI and Medicaid passes on a small frame.

``tools/build_us_fiscal_refresh_release.py`` computes December
``uncapped_ssi`` (``_ssi_person_uncapped_amount``) and ``is_medicaid_eligible``
(``_medicaid_person_eligibility``) for every person before calibration and
again on the calibrated export frame. These tests run those two functions, in
household batches, on seven households, and compare the arrays byte for byte:

* after writing every column the stages between the passes write, in two
  different value sets;
* after calibrated weights, and after an L0 selection that drops households
  (compared by person ID).

``engine_contract/us/test_us_post_calibration_take_up_inputs.py`` checks the
same columns against the engine's static dependency graphs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
)
from test_support.microcosm_build.us_post_calibration_take_up_inputs import (
    DISABLED_WORKER_PERSON_ID,
    SELF_EMPLOYED_PERSON_ID,
    by_person_id,
    stages_between_medicaid_passes,
    stages_between_ssi_passes,
    take_up_pass_frame,
    with_person_columns,
    with_stage_outputs,
)

#: Three household batches over the seven households.
_BATCH_SIZE = 3


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def _december_uncapped_ssi(builder, frame, batch_size=_BATCH_SIZE) -> np.ndarray:
    return builder._ssi_person_uncapped_amount(
        frame, maximum_microsim_batch_size=batch_size
    )


def _medicaid_eligibility(builder, frame, batch_size=_BATCH_SIZE) -> np.ndarray:
    return builder._medicaid_person_eligibility(
        frame, maximum_microsim_batch_size=batch_size
    )


def _changed_person_ids(frame, before, after) -> list[int]:
    changed = by_person_id(frame, after) != by_person_id(frame, before)
    return changed.index[changed.to_numpy()].tolist()


def test_columns_written_between_the_ssi_passes_leave_december_ssi_unchanged(
    builder, system
) -> None:
    frame = take_up_pass_frame()
    stages = stages_between_ssi_passes(builder)
    before = _december_uncapped_ssi(builder, frame)
    # The aged single person, the aged couple and the disabled worker.
    assert np.count_nonzero(before > 0) >= 3

    for variant in (0, 1):
        staged = with_stage_outputs(frame, stages, variant=variant, system=system)
        after = _december_uncapped_ssi(builder, staged)
        assert after.dtype == before.dtype
        assert after.tobytes() == before.tobytes(), variant

    # An input of the graph does change the array, so the comparison can fail.
    more_earnings = with_person_columns(
        frame, DISABLED_WORKER_PERSON_ID, employment_income_before_lsr=9_000.0
    )
    assert _changed_person_ids(
        frame, before, _december_uncapped_ssi(builder, more_earnings)
    ) == [DISABLED_WORKER_PERSON_ID]


def test_snap_take_up_leaves_medicaid_eligibility_unchanged_in_2024(
    builder, system
) -> None:
    frame = take_up_pass_frame()
    medicaid_take_up, snap_state_take_up, _ = stages_between_medicaid_passes()
    before = _medicaid_eligibility(builder, frame)
    assert before.any() and not before.all()

    for variant in (0, 1):
        staged = with_stage_outputs(
            frame,
            (medicaid_take_up, snap_state_take_up),
            variant=variant,
            system=system,
        )
        after = _medicaid_eligibility(builder, staged)
        assert after.dtype == before.dtype
        assert after.tobytes() == before.tobytes(), variant


def test_premium_attribution_changes_medicaid_eligibility(builder) -> None:
    """The other-health stage sets ``is_self_employed`` from self-employment
    income and attributes reported premiums to ``health_insurance_premiums``.
    The self-employed health insurance deduction then lowers this adult's
    income below California's Medicaid limit, so the eligibility computed
    after the stage differs from the one computed before it."""

    frame = take_up_pass_frame()
    before = _medicaid_eligibility(builder, frame)
    attributed = with_person_columns(
        frame,
        SELF_EMPLOYED_PERSON_ID,
        health_insurance_premiums=5_000.0,
        is_self_employed=True,
    )
    after = _medicaid_eligibility(builder, attributed)

    assert _changed_person_ids(frame, before, after) == [SELF_EMPLOYED_PERSON_ID]
    assert not by_person_id(frame, before)[SELF_EMPLOYED_PERSON_ID]
    assert by_person_id(frame, after)[SELF_EMPLOYED_PERSON_ID]


def test_calibrated_weights_and_household_selection_keep_both_arrays(
    builder,
) -> None:
    """Both export-frame constructions give each remaining person the value
    the pre-calibration pass gave them, at any batch composition."""

    frame = take_up_pass_frame()
    ssi_before = by_person_id(frame, _december_uncapped_ssi(builder, frame))
    medicaid_before = by_person_id(frame, _medicaid_eligibility(builder, frame))
    reweighted = builder._with_calibrated_weights(
        frame, np.asarray([5.5, 0.25, 9.0, 1.75, 3.0, 12.0, 0.5])
    )
    # L0 selection keeps households 1, 3, 4, 6 and 7.
    selected = builder._with_l0_refit_weights(
        frame,
        SimpleNamespace(
            weight_entity="household",
            selected_entity_ids=np.asarray([1, 3, 4, 6, 7]),
            weights=np.asarray([2.5, 8.0, 0.75, 4.0, 1.5]),
        ),
    )
    assert selected.n("household") == 5

    for export_frame, batch_size in ((reweighted, 2), (selected, 1)):
        person_ids = export_frame.table("person")["person_id"].to_numpy()
        for compute, before in (
            (_december_uncapped_ssi, ssi_before),
            (_medicaid_eligibility, medicaid_before),
        ):
            after = compute(builder, export_frame, batch_size)
            expected = before.reindex(person_ids).to_numpy()
            assert after.dtype == expected.dtype
            assert after.tobytes() == expected.tobytes(), (compute, batch_size)


def test_reused_december_ssi_equals_a_fresh_pass_on_the_export_frame(
    builder, system
) -> None:
    """Every household keeps its inputs, so the post-calibration December SSI
    pass takes every value from the first pass and runs one batch as a
    check."""

    frame = take_up_pass_frame()
    record = builder._household_pass_record(
        frame, _december_uncapped_ssi(builder, frame), variable="uncapped_ssi"
    )
    staged = with_stage_outputs(
        frame, stages_between_ssi_passes(builder), variant=0, system=system
    )
    export_frame = builder._with_l0_refit_weights(
        staged,
        SimpleNamespace(
            weight_entity="household",
            selected_entity_ids=np.asarray([1, 3, 4, 6, 7]),
            weights=np.asarray([2.5, 8.0, 0.75, 4.0, 1.5]),
        ),
    )

    reuse = builder._reuse_household_pass(
        record,
        export_frame,
        recompute=lambda run_frame: _december_uncapped_ssi(builder, run_frame),
        maximum_microsim_batch_size=_BATCH_SIZE,
        description="December SSI candidate amounts",
    )

    assert reuse.values.tobytes() == (
        _december_uncapped_ssi(builder, export_frame).tobytes()
    )
    assert (reuse.reused_households, reuse.recomputed_households) == (5, 0)
    assert (reuse.checked_households, reuse.check_failed) == (_BATCH_SIZE, False)


def test_reused_medicaid_eligibility_reruns_the_households_whose_inputs_changed(
    builder, system
) -> None:
    """SNAP take-up flags that differ from the engine default and the
    other-health stage's premium attribution change the inputs of four
    households; the other three keep their first-pass values."""

    frame = take_up_pass_frame()
    record = builder._household_pass_record(
        frame, _medicaid_eligibility(builder, frame), variable="is_medicaid_eligible"
    )
    medicaid_take_up, snap_state_take_up, _ = stages_between_medicaid_passes()
    # Variant 0 sets every other SPM unit's flag to False (the engine default
    # is True): households 2, 4 and 6. Attribution changes household 4.
    staged = with_stage_outputs(
        frame, (medicaid_take_up, snap_state_take_up), variant=0, system=system
    )
    attributed = with_person_columns(
        staged,
        SELF_EMPLOYED_PERSON_ID,
        health_insurance_premiums=5_000.0,
        is_self_employed=True,
    )
    export_frame = builder._with_calibrated_weights(
        attributed, np.asarray([5.5, 0.25, 9.0, 1.75, 3.0, 12.0, 0.5])
    )

    reuse = builder._reuse_household_pass(
        record,
        export_frame,
        recompute=lambda run_frame: _medicaid_eligibility(builder, run_frame),
        maximum_microsim_batch_size=2,
        description="Medicaid eligibility",
    )

    fresh = _medicaid_eligibility(builder, export_frame)
    assert reuse.values.tobytes() == fresh.tobytes()
    assert (reuse.reused_households, reuse.recomputed_households) == (4, 3)
    assert (reuse.checked_households, reuse.check_failed) == (2, False)
    # The self-employed adult's eligibility comes from the rerun, not the
    # first pass, which had it ineligible.
    assert by_person_id(export_frame, reuse.values)[SELF_EMPLOYED_PERSON_ID]


@pytest.mark.parametrize(
    ("variable", "period"),
    [("uncapped_ssi", "2024-12"), ("is_medicaid_eligible", 2024)],
)
def test_the_engine_reads_only_listed_columns_while_it_computes_the_pass(
    variable, period, builder, system, monkeypatch
) -> None:
    """Every variable policyengine-us calculates or looks up while it computes
    the pass variable is in the reuse's input list, except the household
    weight, which only weights the returned array."""

    from policyengine_core.simulations.simulation import Simulation
    from policyengine_us import Microsimulation

    from microcosm.build.us_runtime.household_pass_inputs import (
        us_household_pass_inputs,
    )

    reads: set[str] = set()
    recording = False
    calculate = Simulation.calculate
    get_holder = Simulation.get_holder

    def recording_calculate(self, variable_name, *args, **kwargs):
        if recording:
            reads.add(variable_name)
        return calculate(self, variable_name, *args, **kwargs)

    def recording_get_holder(self, variable_name):
        if recording:
            reads.add(variable_name)
        return get_holder(self, variable_name)

    monkeypatch.setattr(Simulation, "calculate", recording_calculate)
    monkeypatch.setattr(Simulation, "get_holder", recording_get_holder)
    frame = take_up_pass_frame()
    simulation = Microsimulation(
        dataset=builder._dataset_from_frame(
            frame, assert_no_formula_owned_columns=False, period_only=True
        )
    )
    recording = True
    simulation.calculate(variable, period=period, map_to="person")
    recording = False

    listed = {
        column.name
        for column in us_household_pass_inputs(
            variable,
            metadata_index=builder._formula_owned_gate_adapter(),
            system=system,
        ).columns
    }
    assert len(reads) > 100
    assert reads - listed == {"household_weight"}
