"""A release built from a base frame should keep the base's imputed inputs.

``tools/build_us_fiscal_refresh_release.py --base-h5`` loads a frame built by
``tools/build_us_puf_support_base.py`` and, outside the validated-pool path
(``pool_frame is None``), runs several base stages again before calibration.
Each of the QRF stages tested here recomputes its PUF-half draw whenever the
frame carries support roles and the raw ASEC source columns, and the exported
base H5 carries both. Before those re-runs the release rescales household
weights to the Census population and rescales the four Social Security
component leaves to SSA totals. ``social_security`` (their sum) is one of the
eight predictors the childcare, energy-subsidy and retirement-contribution
QRFs condition on, so the re-runs overwrite base values on part of the PUF
half.

These are stage-level tests on the synthetic base in
``test_support/microcosm_build/us_release_base_input.py``. They do not execute
``_main``; they replay its stage calls, read from its source (which stages it
still calls, and with which constant keyword arguments), around the release
builder's own repair functions. A fix that stops ``_main`` re-running a stage,
or passes it a flag that consumes the base's value, changes what the replay
runs.

1. Idempotence: re-running a stage on its own unchanged output reproduces
   every output value exactly. Holds.
2. The release's population mass repair (one factor on every household
   weight) leaves every re-run output unchanged. Holds; the property module
   samples other factors.
3. Base preservation: after the release's value repairs, the stage calls
   ``_main`` makes store the base's stage outputs unchanged on every row.
   Fails today for all three stages: a strict expected failure until the
   ruling on the microcosm#1191 follow-up. Beside it, the ASEC half must stay
   untouched either way.
4. Inventory: every stage call in ``_main`` that the base builder also makes
   is pinned, with its guard and constant keyword arguments, so a new or
   changed re-run has to be classified against invariant 3. So are the two
   value repairs the release frames above apply, and their order before the
   stages.

The full-scale differential of a real base and release, and replays of these
stages on that base, are under ``docs/evidence/us-release-reimputation-1191/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_support.microcosm_build.us_release_base_input import (
    BASE_TOOL,
    PERIOD,
    RELEASE_TOOL,
    ROUTE_A_SS_FACTORS,
    STAGES,
    ReleaseScenario,
    UnmodeledReleaseCallError,
    assert_identical,
    base_builder_call_names,
    constant_binding,
    load_release_tool,
    outputs,
    release_stage,
    release_stage_calls,
)

# Stage-call statements ``_main`` runs only when it has no validated pool
# frame, in source order, exactly as ``ast.unparse`` prints them. Any change to
# a call, its arguments or its target has to be classified here.
_EXPECTED_POOL_ABSENT_STATEMENTS = (
    (
        "base_frame = with_us_weeks_unemployed(base_frame, seed=args.seed"
        ", time_period=PERIOD, asec_2023_source=weeks_unemployed_source)"
    ),
    "base_frame, base_population_repair = _with_base_population_mass_repair(base_frame)",
    "base_frame = with_us_qbi_input_reconciliation(base_frame)",
    (
        "base_frame = with_us_childcare_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD, allow_existing_without_source=True)"
    ),
    (
        "base_frame = with_us_energy_subsidy_input(base_frame, seed=args.seed"
        ", time_period=PERIOD, allow_existing_without_source=True)"
    ),
    (
        "base_frame = with_us_retirement_contribution_inputs(base_frame"
        ", seed=args.seed, time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_immigration_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_take_up_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_hours_worked_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_relationship_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_spm_independence_role(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_medicare_take_up_input(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_retirement_distribution_inputs(base_frame"
        ", seed=args.seed, time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_eligibility_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_education_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_pregnancy_inputs(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
    (
        "base_frame = with_us_wic_claim_input(base_frame, seed=args.seed"
        ", time_period=PERIOD)"
    ),
)
# The value repairs ``_main`` makes on the --base-h5 path, in source order:
# ``ReleaseScenario`` applies the first two before the stages, as here.
_EXPECTED_REPAIR_STATEMENTS = (
    "base_frame, base_population_repair = _with_base_population_mass_repair(base_frame)",
    (
        "base_frame, social_security_component_repair = "
        "_with_social_security_component_value_repair(base_frame, target_specs)"
    ),
    (
        "base_frame, non_sch_d_cgd_repair = "
        "_with_non_sch_d_cgd_value_repair(base_frame, target_specs)"
    ),
)
# The three of those the base builder does not call.
_NOT_BASE_STAGES = frozenset(
    {
        "_with_base_population_mass_repair",
        "with_us_take_up_inputs",
        "with_us_hours_worked_inputs",
    }
)


@pytest.fixture(scope="module")
def scenario() -> ReleaseScenario:
    return ReleaseScenario(load_release_tool())


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_rerun_on_unchanged_base_reproduces_every_output(
    scenario: ReleaseScenario, stage: str
) -> None:
    assert_identical(scenario.rerun("base", stage), outputs(scenario.base, stage))


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_rerun_is_invariant_to_the_population_mass_repair(
    scenario: ReleaseScenario, stage: str
) -> None:
    assert_identical(
        scenario.rerun("mass_repaired", stage), outputs(scenario.base, stage)
    )


def test_social_security_repair_changes_the_order_of_the_shared_predictor(
    scenario: ReleaseScenario,
) -> None:
    before = scenario.base.table("person")
    after = scenario.release_prefix.table("person")
    for column, factor in ROUTE_A_SS_FACTORS.items():
        np.testing.assert_allclose(
            after[column].to_numpy(dtype=np.float64),
            before[column].to_numpy(dtype=np.float64) * factor,
            rtol=1e-12,
        )
    # A tree splits on a predictor's order. Mathematically one common factor
    # would keep the order of the summed predictor; these unequal factors
    # change it here. The sum is built as the stages build it.
    columns = [
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
    ]

    def summed(table) -> np.ndarray:
        return np.sum(
            np.column_stack(
                [table[column].to_numpy(dtype=np.float64) for column in columns]
            ),
            axis=1,
        )

    summed_before = summed(before)
    summed_after = summed(after)
    positive = summed_before > 0
    assert not np.array_equal(
        np.argsort(summed_before[positive], kind="stable"),
        np.argsort(summed_after[positive], kind="stable"),
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "The release re-imputes PUF-half inputs after rescaling Social "
        "Security, a predictor of these QRFs; ruling pending on the "
        "microcosm#1191 follow-up issue."
    ),
)
@pytest.mark.parametrize("stage", sorted(STAGES))
def test_release_stage_calls_store_the_base_outputs(
    scenario: ReleaseScenario, stage: str
) -> None:
    assert_identical(
        scenario.rerun("release_prefix", stage), outputs(scenario.base, stage)
    )


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_release_stage_calls_leave_the_asec_half_untouched(
    scenario: ReleaseScenario, stage: str
) -> None:
    """Differential: base against release outputs, row by row, by channel."""

    entity = STAGES[stage][0]
    base = outputs(scenario.base, stage)
    release = scenario.rerun("release_prefix", stage)
    table = scenario.base.table(entity).set_index(f"{entity}_id").sort_index()
    native = table[f"{entity}_support_clone_index"].astype(int).to_numpy() == 0
    differs = (base.to_numpy() != release.to_numpy()).any(axis=1)

    assert native.any() and (~native).any()
    assert not differs[native].any()


def test_release_calls_to_base_stages_are_pinned() -> None:
    calls = release_stage_calls()
    base_calls = base_builder_call_names()
    pool_absent = [call for call in calls if call.pool_absent_only]

    assert tuple(call.statement for call in pool_absent) == (
        _EXPECTED_POOL_ABSENT_STATEMENTS
    )
    # Each is a plain statement under exactly ``pool_frame is None``, so it
    # runs on the --base-h5 path; a changed guard has to be classified here.
    assert all(call.plain for call in pool_absent)
    assert {call.path for call in pool_absent} == {("pool_frame is None",)}
    assert {call.on_base_h5_path for call in pool_absent} == {True}
    assert {
        call.name for call in pool_absent if call.name not in base_calls
    } == _NOT_BASE_STAGES
    # No base stage is called outside the pool-absent branches or inside a
    # construct the path model does not describe.
    assert [
        call.statement
        for call in calls
        if call.name in base_calls and not (call.pool_absent_only and call.plain)
    ] == []
    # The three stages replayed above are called once each.
    for _, _, function in STAGES.values():
        assert [call.name for call in calls].count(function.__name__) == 1


def test_release_value_repairs_precede_the_stage_calls() -> None:
    calls = release_stage_calls()
    repairs = [
        call
        for call in calls
        if call.name.endswith("_repair") and call.name.startswith("_with_")
    ]
    stage_lines = [
        call.lineno
        for call in calls
        if call.name in {function.__name__ for _, _, function in STAGES.values()}
    ]

    assert tuple(call.statement for call in repairs) == _EXPECTED_REPAIR_STATEMENTS
    assert all(call.plain and call.on_base_h5_path is True for call in repairs)
    # The mass repair runs only without a validated pool; the value repairs
    # run on every path.
    assert [call.path for call in repairs] == [("pool_frame is None",), (), ()]
    assert max(call.lineno for call in repairs) < min(stage_lines)


def test_both_builders_bind_the_replayed_period() -> None:
    # The base builder's --target-year defaults to its PERIOD.
    assert constant_binding(BASE_TOOL.read_text(), "PERIOD") == PERIOD
    assert constant_binding(RELEASE_TOOL.read_text(), "PERIOD") == PERIOD
    assert {call.period for call in release_stage_calls()} == {PERIOD}


_READER_SOURCE = """
def _main():
    if pool_frame is None:
        frame = with_us_alpha(frame, seed=args.seed, flag=True, time_period=2024)
    if pool_frame is None and args.base_h5:
        frame = with_us_beta(frame)
    if pool_frame is not None:
        frame = with_us_gamma(frame)
    else:
        if args.x:
            frame = stages.with_us_delta(frame, mode="consume")
    frame = with_us_epsilon(frame)
    if pool_frame is None or args.base_h5:
        frame = with_us_zeta(frame)
    if not pool_frame is None:
        frame = with_us_eta(frame)
    else:
        frame = with_us_theta(frame)
    if pool_frame is not None and args.y:
        frame = with_us_iota(frame)
    else:
        frame = with_us_kappa(frame)
    if pool_frame is None and args.base_h5 is None:
        frame = with_us_lambda(frame)
"""


def test_stage_call_reader_evaluates_guards() -> None:
    calls = {call.name: call for call in release_stage_calls(_READER_SOURCE)}

    # Unreachable when a validated pool is present.
    assert {name: call.pool_absent_only for name, call in calls.items()} == {
        "with_us_alpha": True,
        "with_us_beta": True,
        "with_us_gamma": False,
        "with_us_delta": True,
        "with_us_epsilon": False,
        # ``or``: runs with a pool whenever args.base_h5 is set.
        "with_us_zeta": False,
        "with_us_eta": False,
        "with_us_theta": True,
        # else of an ``and``: runs with a pool when args.y is false.
        "with_us_iota": False,
        "with_us_kappa": False,
        "with_us_lambda": True,
    }
    # Runs on the --base-h5 path (True), is skipped (False), or is undecided.
    assert calls["with_us_alpha"].on_base_h5_path is True
    assert calls["with_us_beta"].on_base_h5_path is True
    assert calls["with_us_gamma"].on_base_h5_path is False
    assert calls["with_us_delta"].on_base_h5_path is None
    assert calls["with_us_zeta"].on_base_h5_path is True
    assert calls["with_us_theta"].on_base_h5_path is True
    assert calls["with_us_kappa"].on_base_h5_path is True
    assert calls["with_us_lambda"].on_base_h5_path is False
    assert calls["with_us_lambda"].path == (
        "pool_frame is None and args.base_h5 is None",
    )
    assert calls["with_us_alpha"].plain
    assert calls["with_us_delta"].plain
    assert calls["with_us_alpha"].statement == (
        "frame = with_us_alpha(frame, seed=args.seed, flag=True, time_period=2024)"
    )


def _calls_for(body: str, prelude: str = f"PERIOD = {PERIOD}\n"):
    return release_stage_calls(prelude + "def _main():\n" + body)


def test_replay_follows_the_release_guard() -> None:
    name = STAGES["childcare"][2].__name__
    frame = object()
    call = f"        frame = {name}(frame, seed=args.seed, time_period=PERIOD"

    skipped = _calls_for(
        "    if pool_frame is None and args.base_h5 is None:\n" + call + ")\n"
    )
    assert release_stage("childcare", skipped)(frame) is frame

    removed = _calls_for("    frame = with_us_other(frame)\n")
    assert release_stage("childcare", removed)(frame) is frame

    undecided = _calls_for(
        "    if pool_frame is None and args.consume_base:\n" + call + ")\n"
    )
    with pytest.raises(UnmodeledReleaseCallError, match="not decided"):
        release_stage("childcare", undecided)

    extra_argument = _calls_for("    if pool_frame is None:\n" + call + ", mode=m)\n")
    with pytest.raises(UnmodeledReleaseCallError, match="non-constant"):
        release_stage("childcare", extra_argument)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # A conditional expression hides a guard from the if-path model.
        (
            "    if pool_frame is None:\n"
            "        frame = {name}(frame, seed=args.seed, time_period=PERIOD)"
            " if args.base_h5 is None else frame\n",
            "not a plain assignment",
        ),
        # A loop, try or with block runs its body conditionally or repeatedly.
        (
            "    for _ in range(1):\n"
            "        frame = {name}(frame, seed=args.seed, time_period=PERIOD)\n",
            "not a plain assignment",
        ),
        (
            "    try:\n"
            "        frame = {name}(frame, seed=args.seed, time_period=PERIOD)\n"
            "    except ValueError:\n"
            "        pass\n",
            "not a plain assignment",
        ),
        # A changed input or seed.
        (
            "    if pool_frame is None:\n"
            "        frame = {name}(strip(frame), seed=args.seed, time_period=PERIOD)\n",
            "only positional argument",
        ),
        (
            "    if pool_frame is None:\n"
            "        frame = {name}(frame, seed=args.seed + 1, time_period=PERIOD)\n",
            "seed=args.seed \\+ 1",
        ),
        (
            "    if pool_frame is None:\n"
            "        frame = {name}(frame, seed=args.seed, time_period=PERIOD, **extra)\n",
            "non-constant",
        ),
    ],
)
def test_replay_refuses_calls_it_cannot_model(body: str, message: str) -> None:
    name = STAGES["childcare"][2].__name__
    calls = _calls_for(body.format(name=name))

    with pytest.raises(UnmodeledReleaseCallError, match=message):
        release_stage("childcare", calls)


_CHILDCARE_CALL = (
    "    if pool_frame is None:\n"
    "        frame = {name}(frame, seed=args.seed, time_period=PERIOD)\n"
)


def test_replay_accepts_the_period_it_replays() -> None:
    name = STAGES["childcare"][2].__name__
    calls = _calls_for(_CHILDCARE_CALL.format(name=name))

    assert calls[0].period == PERIOD
    assert callable(release_stage("childcare", calls))


@pytest.mark.parametrize(
    ("prelude", "body"),
    [
        # The module's period changes.
        (f"PERIOD = {PERIOD + 1}\n", _CHILDCARE_CALL),
        # It is not a constant, or not bound at all.
        ("PERIOD = int(os.environ['PERIOD'])\n", _CHILDCARE_CALL),
        ("", _CHILDCARE_CALL),
        # Something else rebinds or shadows it.
        (f"PERIOD = {PERIOD}\nPERIOD += 1\n", _CHILDCARE_CALL),
        (f"PERIOD = {PERIOD}\nfrom periods import NEXT as PERIOD\n", _CHILDCARE_CALL),
        # A wildcard import may supply it.
        (f"PERIOD = {PERIOD}\nfrom next_period import *\n", _CHILDCARE_CALL),
        (f"PERIOD = {PERIOD}\n", f"    PERIOD = {PERIOD + 1}\n" + _CHILDCARE_CALL),
        (f"PERIOD = {PERIOD}\n", "    global PERIOD\n" + _CHILDCARE_CALL),
    ],
)
def test_replay_refuses_a_period_it_does_not_replay(prelude: str, body: str) -> None:
    name = STAGES["childcare"][2].__name__
    calls = _calls_for(body.format(name=name), prelude=prelude)

    with pytest.raises(UnmodeledReleaseCallError, match="does not bind PERIOD"):
        release_stage("childcare", calls)


@pytest.mark.parametrize("parameter", ["PERIOD", "*PERIOD", "**PERIOD"])
def test_period_check_sees_type_parameters(parameter: str) -> None:
    # Inside _main, a type parameter named PERIOD shadows the module's.
    source = f"PERIOD = {PERIOD}\ndef _main[{parameter}]():\n    return PERIOD\n"

    assert constant_binding(source, "PERIOD") is None
