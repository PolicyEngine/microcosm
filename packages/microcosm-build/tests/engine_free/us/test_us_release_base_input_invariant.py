"""A release built from a base frame must keep the base's imputed inputs.

``tools/build_us_fiscal_refresh_release.py --base-h5`` loads a frame built by
``tools/build_us_puf_support_base.py`` and, outside the validated-pool path
(``pool_frame is None``), runs several base stages again before calibration.
Each of the QRF stages tested here recomputes its PUF-half draw whenever the
frame carries support roles and the raw ASEC source columns, and the exported
base H5 carries both. Before those re-runs the release rescales household
weights to the Census population and rescales the four Social Security
component leaves to SSA totals. ``social_security`` (their sum) is one of the
eight predictors the childcare, energy-subsidy and retirement-contribution
QRFs condition on, so the re-runs fit different forests and overwrite base
values on part of the PUF half.

Invariants, on the synthetic base in
``test_support/microcosm_build/us_release_base_input.py``:

1. Idempotence: re-running a stage on its own unchanged output reproduces
   every output bit for bit. Holds.
2. Weight-scale invariance: the release's population mass repair (one factor
   on every household weight) leaves every re-run output unchanged. Holds; the
   property module draws the factor.
3. Base preservation: after the release's declared value repairs, the
   release's re-runs store the base's stage outputs unchanged on every row.
   Fails today for all three stages: a strict expected failure until the
   ruling on the microcosm#1191 follow-up. The differential beside it requires
   the ASEC half to stay untouched either way.
4. Inventory: the release re-runs exactly the pinned list of stages on a base
   H5, so a new re-run has to be classified against invariant 3.

The full-scale differential of a real base and release, and replays of these
stages on that base, are under ``docs/evidence/us-release-reimputation-1191/``.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from test_support.microcosm_build.us_release_base_input import (
    BASE_TOOL,
    RELEASE_TOOL,
    ROUTE_A_SS_FACTORS,
    STAGES,
    ReleaseScenario,
    load_release_tool,
    outputs,
)

# The release's ``pool_frame is None`` block in ``_main``, in call order. Every
# entry except the three in ``_RELEASE_ONLY_GUARDED`` re-runs a stage the base
# builder already ran.
_EXPECTED_RELEASE_RERUNS = (
    "with_us_weeks_unemployed",
    "_with_base_population_mass_repair",
    "with_us_qbi_input_reconciliation",
    "with_us_childcare_inputs",
    "with_us_energy_subsidy_input",
    "with_us_retirement_contribution_inputs",
    "with_us_immigration_inputs",
    "with_us_take_up_inputs",
    "with_us_hours_worked_inputs",
    "with_us_relationship_inputs",
    "with_us_spm_independence_role",
    "with_us_medicare_take_up_input",
    "with_us_retirement_distribution_inputs",
    "with_us_eligibility_inputs",
    "with_us_education_inputs",
    "with_us_pregnancy_inputs",
    "with_us_wic_claim_input",
)
_RELEASE_ONLY_GUARDED = frozenset(
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
    pd.testing.assert_frame_equal(
        scenario.rerun("base", stage), outputs(scenario.base, stage)
    )


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_rerun_is_invariant_to_the_population_mass_repair(
    scenario: ReleaseScenario, stage: str
) -> None:
    pd.testing.assert_frame_equal(
        scenario.rerun("mass_repaired", stage), outputs(scenario.base, stage)
    )


def test_social_security_repair_reorders_the_shared_predictor(
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
    # Unequal factors change the order of the summed predictor, not only its
    # scale, so no rescaling of split thresholds reproduces the base forests.
    columns = list(ROUTE_A_SS_FACTORS)
    summed_before = before[columns].sum(axis=1).to_numpy()
    summed_after = after[columns].sum(axis=1).to_numpy()
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
def test_release_reruns_store_the_base_outputs(
    scenario: ReleaseScenario, stage: str
) -> None:
    pd.testing.assert_frame_equal(
        scenario.rerun("release_prefix", stage), outputs(scenario.base, stage)
    )


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_release_reruns_leave_the_asec_half_untouched(
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


def test_release_rerun_inventory_is_pinned() -> None:
    tree = ast.parse(RELEASE_TOOL.read_text())
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_main"
    )

    def guarded_by_pool_none(node: ast.If) -> bool:
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "pool_frame"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )

    calls = sorted(
        (inner.lineno, inner.func.id)
        for node in ast.walk(main)
        if isinstance(node, ast.If) and guarded_by_pool_none(node)
        for statement in node.body
        for inner in ast.walk(statement)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id.startswith(("with_us_", "_with_"))
    )

    assert tuple(name for _, name in calls) == _EXPECTED_RELEASE_RERUNS
    base_source = BASE_TOOL.read_text()
    for name in _EXPECTED_RELEASE_RERUNS:
        assert (f"{name}(" in base_source) == (name not in _RELEASE_ONLY_GUARDED), name
