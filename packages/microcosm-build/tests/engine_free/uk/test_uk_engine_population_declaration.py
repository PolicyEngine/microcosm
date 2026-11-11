"""The build's constants name what the UK engine-population declaration does.

microcosm-frame cannot import microcosm-build, so the two things both name
are held together here (microcosm#931): the household column spine engine
calls batch by, which must be the ladder frame's clone index, and the frame
columns each share formula's allocation key adds, which the measure step
sums for the pool-share correction.

The spine side of that batching is held here too. The adapter runs a
call over a frame that carries the clone index clone by clone, and one over a
frame without it as one simulation over the rows it holds, so no engine
stage after geographic_expand may hand the engine a frame that lost the
index: each is an open surface, given the whole frame, and the three stages
that build a temporary frame for the engine keep the index on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import graph, graph_kernels
from microcosm.build.uk_runtime.frs_take_up import UKTakeUpPopulationPolicy
from microcosm.build.uk_runtime.full_measure import UK_WEIGHT_SHARE_FORMULA_INPUTS
from microcosm.build.uk_runtime.geographic_expand import GEOGRAPHIC_EXPAND_STAGE_NAME
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.pension_credit_take_up import (
    PENSION_CREDIT_MIXED_AGE_SAVING,
    redraw_pension_credit_take_up,
)
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.build.uk_runtime.spi_support import support_channel_column
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.build.uk_runtime.uc_reporter_redraw import redraw_spi_reported_uc
from microcosm.build.uk_runtime.uc_take_up import (
    UC_AWARD_BAND_LABELS,
    UC_FAMILY_TYPES,
    redraw_uc_take_up,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
)
from test_support.microcosm_build.uk_clone_first_frames import clone_first_frame
from test_support.microcosm_build.uk_uc_reporter_redraw import (
    _frame as _reporter_frame,
)
from test_support.microcosm_build.uk_uc_reporter_redraw import (
    _StubEngine as _ReporterEngine,
)
from test_support.microcosm_build.uk_uc_reporter_redraw import _StubQRFFactory

#: The clone count the temporary frames are checked at.
_CLONES = 2


def test_spine_engine_calls_batch_by_the_ladder_households_clone_index():
    assert UK_SPINE_ENGINE_BATCH_COLUMN == ladder_clone_index_column("household")


def test_the_share_inputs_the_build_sums_are_the_declared_key_adds():
    declared = {
        formula: UK_ENGINE_POPULATION.key_adds[key]
        for formula, key in UK_ENGINE_POPULATION.corrected.items()
    }
    # Tuples: the same columns in the same order.
    assert dict(UK_WEIGHT_SHARE_FORMULA_INPUTS) == declared


def test_every_engine_stage_after_the_clones_is_given_the_whole_frame():
    """Every stage after geographic_expand whose node binds the engine
    modules (``graph_kernels._STAGE_HELPER_MODULES``; test_uk_graph holds
    these to exactly the stages that call the engine or take one) declares no
    consumes, an open surface, so its node's slices hold the household clone
    index the adapter batches by."""

    order = [stage.stage for stage in load_country_spec("uk").sources.stages]
    after = order[order.index(GEOGRAPHIC_EXPAND_STAGE_NAME) + 1 :]
    engine_modules = set(graph_kernels._ENGINE_MODULES)
    engine_stages = [
        stage
        for stage in after
        if engine_modules <= set(graph_kernels._STAGE_HELPER_MODULES.get(stage, ()))
    ]
    # The premise: the engine stages after the clones include the three that
    # build a temporary frame, checked below.
    assert {
        "uc_reporter_redraw",
        "uc_take_up",
        "pension_credit_take_up",
    } <= set(engine_stages)
    spine = graph.uk_spine_graph()
    for stage in engine_stages:
        assert graph._STAGE_CONSUMES[stage] is None, stage
        household = {
            column
            for item in spine.node(stage).inputs
            if item.entity == "household"
            for column in item.columns
        }
        assert UK_SPINE_ENGINE_BATCH_COLUMN in household, stage


class _ClonedCallRecorder(PolicyEngineUKEngine):
    """A UK engine built for a spine of :data:`_CLONES` clones whose
    simulations are a stand-in's. Each call records how many simulations the
    adapter runs over the frame it is handed, by the adapter's own decision
    and refusals: one per clone where the frame carries the clone index, one
    where it lost it, and a refusal where it carries another count. It then
    answers with the stand-in's values."""

    def __init__(self, stand_in) -> None:
        super().__init__(clones=_CLONES)
        self.stand_in = stand_in
        self.batches: list[int] = []

    def materialize(self, bundle, variables, period, **kwargs):
        self.batches.append(self.simulation_batches(bundle))
        return self.stand_in.materialize(bundle, variables, period, **kwargs)


class _UCEngine:
    """Every unit is paid an award; nobody is a child."""

    country = "uk"

    def materialize(self, frame, variables, period, *, dataset_period=None):
        units = len(frame.table("benunit"))
        values = {
            "universal_credit": np.full(units, 900.0),
            "legacy_benefits_closed": np.zeros(units, dtype=bool),
            "is_child_or_qualifying_young_person_for_universal_credit": np.zeros(
                len(frame.table("person")), dtype=bool
            ),
        }
        return {name: values[name] for name in variables}


class _PensionCreditEngine:
    """Unit 1 is entitled to Guarantee Credit, unit 2 to Savings Credit only."""

    country = "uk"

    def materialize(self, frame, variables, period, *, dataset_period=None):
        ids = frame.table("benunit")["benunit_id"].to_numpy() % 1_000_000_000
        values = {
            PENSION_CREDIT_MIXED_AGE_SAVING: ids == 2,
            "guarantee_credit": np.where(ids == 1, 40.0, 0.0),
            "savings_credit": np.where(ids == 2, 15.0, 0.0),
            "is_pension_credit_eligible": ids != 3,
            "pension_credit_assessable_capital": np.zeros(len(ids)),
            "pension_credit_deemed_income": np.zeros(len(ids)),
        }
        return {name: values[name] for name in variables}


def _uc_take_up_frame():
    person = pd.DataFrame(
        {
            "person_id": [11, 21, 31],
            "person_benunit_id": [1, 2, 3],
            "person_household_id": [1, 2, 3],
            "age": [30, 40, 50],
            "is_uc_claimant": [True] * 3,
            "is_benunit_head": [True] * 3,
            "is_parent": [False] * 3,
            "is_claimant_or_partner": [True] * 3,
            "universal_credit_reported": [1_000.0, 0.0, 0.0],
            **{
                column: [0.0] * 3
                for column in (
                    "child_tax_credit_reported",
                    "working_tax_credit_reported",
                    "housing_benefit_reported",
                    "esa_income_reported",
                    "income_support_reported",
                    "jsa_income_reported",
                )
            },
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [1, 2, 3],
            "dependent_children": [0, 0, 0],
            "would_claim_uc": [True, True, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "region": ["WALES", "WALES", "NORTHERN_IRELAND"],
            "source_household_id": [1, 2, 3],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([10.0, 10.0, 10.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _pension_credit_frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [11, 21, 31],
                "person_benunit_id": [1, 2, 3],
                "person_household_id": [1, 2, 3],
                "age": [80, 78, 70],
                "pension_credit_reported": [0.0, 0.0, 500.0],
            }
        ),
        benunit=pd.DataFrame(
            {
                "benunit_id": [1, 2, 3],
                "would_claim_pc": [True, True, False],
                support_channel_column("benunit"): ["frs", "frs", "spi"],
            }
        ),
        household=pd.DataFrame({"household_id": [1, 2, 3], "region": ["WALES"] * 3}),
        household_weights=np.asarray([10.0, 10.0, 10.0]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _uc_targets() -> dict[tuple[str, int], float]:
    targets = {
        (name, index): 0.0
        for name, _ in UC_FAMILY_TYPES
        for index in range(len(UC_AWARD_BAND_LABELS))
    }
    targets[("SINGLE", 0)] = 10.0
    return targets


def test_the_temporary_engine_frames_keep_the_clone_index():
    """The three stages that hand the engine a frame of their own making
    (the reporter screen, the claim-everyone award and the Pension Credit
    entitlement with its stored saving) keep the household clone index on
    it, so an engine built for a spine of two clones runs each of their calls
    clone by clone, as it does the calls over the stage's own frame."""

    runs = {
        "uc_reporter_redraw": lambda frame, engine: redraw_spi_reported_uc(
            frame, engine=engine, qrf_factory=_StubQRFFactory()
        ),
        "uc_take_up": lambda frame, engine: redraw_uc_take_up(
            frame,
            engine=engine,
            population_policy=UKTakeUpPopulationPolicy(
                adult_age=18, state_pension_age=66, instant="2024-01-01", source="test"
            ),
            targets=_uc_targets(),
            closure_rates={"all": 0.8},
            minimum_pool_sources=1,
        ),
        "pension_credit_take_up": lambda frame, engine: redraw_pension_credit_take_up(
            frame, engine=engine, contract=load_uk_take_up_contract()
        ),
    }
    frames = {
        "uc_reporter_redraw": (_reporter_frame(), _ReporterEngine()),
        "uc_take_up": (_uc_take_up_frame(), _UCEngine()),
        "pension_credit_take_up": (_pension_credit_frame(), _PensionCreditEngine()),
    }
    # The temporary frame is each stage's only call but Pension Credit's,
    # whose first call reads the stage's own frame for the saving it stores.
    calls = {"uc_reporter_redraw": 1, "uc_take_up": 1, "pension_credit_take_up": 2}
    for stage, run in runs.items():
        base, stand_in = frames[stage]
        engine = _ClonedCallRecorder(stand_in)
        run(clone_first_frame(base, _CLONES), engine)
        assert engine.batches == [_CLONES] * calls[stage], stage
        # The control: a frame without the index takes one simulation a call,
        # which is what a temporary frame that dropped it would record.
        engine = _ClonedCallRecorder(stand_in)
        run(base, engine)
        assert engine.batches == [1] * calls[stage], stage
