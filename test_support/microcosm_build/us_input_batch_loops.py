"""Real-engine checks for the release's input-stage household-batch loops.

Before target compilation the US release runs policyengine-us over household
batches in five loops: the ACA source inputs, Medicaid eligibility, December
SSI candidate amounts and SNAP eligibility (in the release tool), and the
other-health-insurance modeled premiums (``us_runtime.other_health_insurance``).
The tests compare each loop with its frozen copy from main
(:mod:`test_support.microcosm_build.us_input_batch_loops_main_baseline`) byte
for byte, and check that each batch engine is checked for population-aggregate
variables and released before the next one is built.

:class:`BatchEngineRecorder` replaces ``policyengine_us.Microsimulation`` with
a subclass that records every engine the loops build. It can also make an
engine, or a branch the engine then drops, compute a population-aggregate
variable, or fail, the first time a loop asks it for its first variable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.other_health_insurance as other_health_insurance
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build import (
    us_input_batch_loops_main_baseline as main_baseline,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
    _real_engine_frame,
)

#: The population aggregate the probes compute: a sum of person weights by
#: state, so a household batch gets its own batch's sum.
POPULATION_AGGREGATE_PROBE_VARIABLE = "medicaid_slcsp_state_denominator"
PROBE_BRANCH_NAME = "population_aggregate_probe"
PROBE_FAILURE_MESSAGE = "probe: batch calculation failed"


def load_builder_module():
    return _load_builder_module()


def real_engine_frame() -> Frame:
    """Four households over three states; household 1 holds two tax units."""

    return _real_engine_frame()


def input_loop_identity_frame() -> Frame:
    """Five households over three states whose rows are out of household order.

    Group ids run against household order (household 1 holds the highest tax
    unit, SPM unit, family and marital unit ids, so the sorted group tables
    list it last), and the person rows are reversed. A loop that placed a
    batch's results at a running row offset, instead of by id, writes other
    bytes than main's. Household 5 is a New York parent with two children, so
    the CHIP premium is nonzero; three people report non-Medicare premiums;
    household weights differ.
    """

    household = np.asarray([1, 1, 2, 3, 3, 3, 4, 5, 5, 5], dtype="int64")
    reverse_household = 6 - household
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 11, dtype="int64"),
            "person_household_id": household,
            "person_tax_unit_id": reverse_household * 10
            + np.asarray([0, 1, 0, 0, 1, 2, 0, 0, 0, 0]),
            "person_spm_unit_id": reverse_household * 100
            + np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            "person_family_id": reverse_household * 1_000
            + np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            "person_marital_unit_id": reverse_household * 10_000
            + np.asarray([0, 1, 0, 0, 1, 2, 0, 0, 1, 2]),
            "age": np.asarray(
                [34.0, 71.0, 45.0, 29.0, 67.0, 19.0, 52.0, 35.0, 6.0, 10.0]
            ),
            "employment_income_before_lsr": np.asarray(
                [48_000.0, 0.0, 180_000.0, 9_000.0, 0.0, 14_000.0, 72_000.0, 62_000.0]
                + [0.0, 0.0]
            ),
            "health_insurance_premiums_without_medicare_part_b": np.asarray(
                [1_200.0, 0.0, 0.0, 2_400.0, 0.0, 0.0, 0.0, 3_600.0, 0.0, 0.0]
            ),
        }
    )
    tables = {"person": person.iloc[::-1].reset_index(drop=True)}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.arange(1, 6, dtype="int64"),
            "state_fips": np.asarray([6, 36, 24, 6, 36], dtype="int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([3.0, 1.0, 5.0, 2.0, 4.0]), WeightKind.DESIGN
            )
        },
    )


def aca_target_tables(builder) -> dict[str, pd.DataFrame]:
    """CMS APTC recipients in two of the frame's states, plus the bronze
    consumer table, whose presence switches the source table to person-count
    weights, so the take-up rates and weights carry values."""

    return {
        builder.US_ACA_APTC_TARGET_TABLE: pd.DataFrame(
            {"state_fips": ["06", "36"], "target": [2.0, 1.5]}
        ),
        "cms_aca_bronze_aptc_consumers_by_state": pd.DataFrame(
            {"state_fips": ["06"], "target": [1.0]}
        ),
    }


def _rebuilt_frame(frame: Frame, tables: dict[str, pd.DataFrame]) -> Frame:
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(frame.weights_for("household").values, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def spm_unit_spanning_households_frame() -> Frame:
    """The real-engine frame with person 3 (household 2) moved into SPM unit
    100, which belongs to household 1, so that SPM unit spans two households."""

    frame = real_engine_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person.loc[person["person_id"] == 3, "person_spm_unit_id"] = 100
    spm_unit = tables["spm_unit"]
    tables["spm_unit"] = spm_unit.loc[spm_unit["spm_unit_id"] != 200].reset_index(
        drop=True
    )
    return _rebuilt_frame(frame, tables)


def medicaid_claim_outside_household_frame() -> Frame:
    """The real-engine frame with person 6 (household 3) claimed for Medicaid
    by tax unit 10, which belongs to household 1."""

    frame = real_engine_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person["medicaid_claiming_tax_unit_id"] = np.where(
        person["person_id"].to_numpy() == 6, 10, 0
    ).astype("int64")
    return _rebuilt_frame(frame, tables)


def _microsimulation_class():
    import policyengine_us

    return policyengine_us.Microsimulation


def _run_aca(builder, frame: Frame, **options):
    return builder._aca_source_tax_unit_table_batched(
        frame,
        aca_target_tables(builder),
        microsimulation_cls=_microsimulation_class(),
        **options,
    )


def _run_main_aca(builder, frame: Frame, *, maximum_microsim_batch_size):
    return main_baseline.main_aca_source_tax_unit_table_batched(
        builder,
        frame,
        aca_target_tables(builder),
        microsimulation_cls=_microsimulation_class(),
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )


def _run_other_health(builder, frame: Frame, **options):
    return other_health_insurance._materialize_modeled_premiums(
        frame, time_period=builder.PERIOD, **options
    )


def _run_main_other_health(builder, frame: Frame, *, maximum_microsim_batch_size):
    return main_baseline.main_materialize_modeled_premiums(
        frame,
        time_period=builder.PERIOD,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )


@dataclass(frozen=True)
class InputLoop:
    """One input-stage loop, its frozen main copy, and what it asks first."""

    name: str
    #: The first variable the loop asks each batch engine for; the recorder's
    #: probes run when an engine is first asked for it.
    first_requested_variable: str
    #: The start of the loop's refusal messages.
    pass_label: str
    run: Callable
    run_main: Callable
    #: The output's arrays that :func:`input_loop_identity_frame` must make
    #: carry engine values (not only zero fills), so a byte comparison compares
    #: engine results; each must hold two different values.
    engine_value_arrays: Callable[[object], dict[str, np.ndarray]]


INPUT_LOOPS = (
    InputLoop(
        name="aca_source_inputs",
        first_requested_variable="is_aca_ptc_eligible",
        pass_label="The ACA source-input pass",
        run=_run_aca,
        run_main=_run_main_aca,
        engine_value_arrays=lambda output: {
            column: output[column].to_numpy()
            for column in (
                "tax_unit_weight",
                "is_aca_ptc_eligible",
                "health_insurance_premiums_without_medicare_part_b",
                "assigned_aca_ptc",
                "weighted_assigned_aca_ptc",
                "slcsp",
                "aca_take_up_rate",
            )
        },
    ),
    InputLoop(
        name="medicaid_eligibility",
        first_requested_variable="is_medicaid_eligible",
        pass_label="The Medicaid eligibility pass",
        run=lambda builder, frame, **options: builder._medicaid_person_eligibility(
            frame, **options
        ),
        run_main=lambda builder, frame, **options: (
            main_baseline.main_medicaid_person_eligibility(builder, frame, **options)
        ),
        engine_value_arrays=lambda output: {"is_medicaid_eligible": output},
    ),
    InputLoop(
        name="december_ssi_amounts",
        first_requested_variable="uncapped_ssi",
        pass_label="The December SSI pass",
        run=lambda builder, frame, **options: builder._ssi_person_uncapped_amount(
            frame, **options
        ),
        run_main=lambda builder, frame, **options: (
            main_baseline.main_ssi_person_uncapped_amount(builder, frame, **options)
        ),
        engine_value_arrays=lambda output: {"uncapped_ssi": output},
    ),
    InputLoop(
        name="snap_eligibility",
        first_requested_variable="is_snap_eligible",
        pass_label="The SNAP eligibility pass",
        run=lambda builder, frame, **options: builder._snap_spm_unit_eligibility(
            frame, **options
        ),
        run_main=lambda builder, frame, **options: (
            main_baseline.main_snap_spm_unit_eligibility(builder, frame, **options)
        ),
        engine_value_arrays=lambda output: {"is_snap_eligible": output},
    ),
    InputLoop(
        name="other_health_insurance_premiums",
        first_requested_variable="chip_premium",
        pass_label="The other-health-insurance premium pass",
        run=_run_other_health,
        run_main=_run_main_other_health,
        # medicaid_premium is zero for everyone in policyengine-us 2.2.1's
        # current-year schedules; the CHIP and marketplace premiums differ.
        engine_value_arrays=lambda output: {
            variable: output[variable]
            for variable in ("chip_premium", "marketplace_net_premium")
        },
    ),
)


def engine_released(engine) -> bool:
    """``release_engine_simulation`` empties an engine's population map."""

    return getattr(engine, "populations", None) == {}


class BatchEngineRecorder:
    """Record every batch engine the loops build, optionally with a probe.

    ``probe`` is ``None`` (record only), ``"engine"`` (the engine computes
    :data:`POPULATION_AGGREGATE_PROBE_VARIABLE`), ``"detached_branch"`` (a
    branch computes it and the engine then drops the branch) or ``"failure"``
    (the engine raises). A probe runs once per engine, the first time the
    engine is asked for ``first_requested_variable``.
    """

    def __init__(
        self,
        monkeypatch,
        *,
        period: int,
        first_requested_variable: str | None = None,
        probe: str | None = None,
    ) -> None:
        import policyengine_us

        if probe not in (None, "engine", "detached_branch", "failure"):
            raise ValueError(f"Unknown probe {probe!r}.")
        self.engines: list[object] = []
        #: One entry per engine: whether every engine built before it had
        #: been released when it was built.
        self.earlier_engines_released_at_build: list[bool] = []
        recorder = self
        base_class = policyengine_us.Microsimulation

        class RecordingMicrosimulation(base_class):
            def __init__(self, *args, **kwargs):
                recorder.earlier_engines_released_at_build.append(
                    all(engine_released(engine) for engine in recorder.engines)
                )
                super().__init__(*args, **kwargs)
                recorder.engines.append(self)

            def calculate(self, variable_name, *args, **kwargs):
                if (
                    probe is not None
                    and variable_name == first_requested_variable
                    and not getattr(self, "_population_aggregate_probe_ran", False)
                ):
                    self._population_aggregate_probe_ran = True
                    if probe == "failure":
                        raise RuntimeError(PROBE_FAILURE_MESSAGE)
                    if probe == "engine":
                        self.calculate(POPULATION_AGGREGATE_PROBE_VARIABLE, period)
                    else:
                        branch = self.get_branch(PROBE_BRANCH_NAME)
                        branch.calculate(POPULATION_AGGREGATE_PROBE_VARIABLE, period)
                        del self.branches[PROBE_BRANCH_NAME]
                return super().calculate(variable_name, *args, **kwargs)

        monkeypatch.setattr(
            policyengine_us, "Microsimulation", RecordingMicrosimulation
        )


def _assert_same_array_bytes(actual, expected, *, label: str) -> None:
    assert type(actual) is type(expected), (label, type(actual), type(expected))
    if isinstance(expected, pd.Series):
        # to_numpy() drops a pandas extension dtype (str, category, Int64).
        assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
        pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    if expected.dtype == object:
        # Object arrays hold pointers; compare the values and their types.
        assert actual.tolist() == expected.tolist(), label
        assert [type(value) for value in actual.tolist()] == [
            type(value) for value in expected.tolist()
        ], label
        return
    assert actual.tobytes() == expected.tobytes(), label


def assert_same_bytes(actual, expected, *, label: str) -> None:
    """The same container types, and every array, column, pandas and NumPy
    dtype, index and column order identical."""

    assert type(actual) is type(expected), (label, type(actual), type(expected))
    if isinstance(expected, pd.DataFrame):
        assert list(actual.columns) == list(expected.columns), label
        pd.testing.assert_series_equal(actual.dtypes, expected.dtypes, obj=label)
        pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
        for column in expected.columns:
            _assert_same_array_bytes(
                actual[column], expected[column], label=f"{label}: {column}"
            )
        return
    if isinstance(expected, dict):
        assert list(actual) == list(expected), label
        for key, values in expected.items():
            _assert_same_array_bytes(actual[key], values, label=f"{label}: {key}")
        return
    _assert_same_array_bytes(actual, expected, label=label)


__all__ = [
    "INPUT_LOOPS",
    "POPULATION_AGGREGATE_PROBE_VARIABLE",
    "PROBE_FAILURE_MESSAGE",
    "BatchEngineRecorder",
    "InputLoop",
    "assert_same_bytes",
    "aca_target_tables",
    "engine_released",
    "input_loop_identity_frame",
    "load_builder_module",
    "medicaid_claim_outside_household_frame",
    "real_engine_frame",
    "spm_unit_spanning_households_frame",
]
