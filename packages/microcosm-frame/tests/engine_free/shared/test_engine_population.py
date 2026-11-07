"""The shared engine-block machinery on fake engines (microcosm#931).

A fake simulation stands in for a policyengine-core one: holders that know
which periods hold inputs, formulas run through ``_run_formula``, a
parameter tree read at a period, and a ``clone`` that rebinds the methods
bound to the original, as core's does. Everything is synthetic.
"""

from __future__ import annotations

import copy
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters.engine_population import (
    FLOAT32_UNIT_ROUNDOFF,
    MEASURE_PASS,
    SPINE_BATCHED_PASS,
    FormulaRunRecorder,
    InertRule,
    ParameterTest,
    PassPolicy,
    PoolShare,
    PoolShareError,
    PopulationDeclaration,
    PopulationDeclarationError,
    PopulationGuardError,
    PositionWindow,
    all_of,
    any_of,
    assert_groups_nest_in_households,
    check_key_sums,
    check_uniform_growth,
    derived_watched_periods,
    factor_spread,
    float32_sum_roundings,
    household_batches,
    inject_pool_shares,
    judge,
    pool_share_atol,
    pool_share_rtol,
    pool_shares,
    scatter_by_entity_ids,
    source_sha256,
    validate_declaration,
)

D, P = 2024, 2025
N = 6
FAKE_ENGINE = "fake_population_engine"


# ---------------------------------------------------------------------------
# A fake engine
# ---------------------------------------------------------------------------


def _namespace(tree):
    if isinstance(tree, dict):
        return SimpleNamespace(
            **{key: _namespace(value) for key, value in tree.items()}
        )
    return tree


class FakeParameters:
    """A parameter tree: called with a period it gives that period's values,
    and its root's children are reachable as attributes, as core's are."""

    def __init__(self, tree, overrides=None):
        self.tree = tree
        self.overrides = overrides or {}

    def __call__(self, period):
        tree = copy.deepcopy(self.tree)
        for path, value in self.overrides.get(int(str(period)[:4]), {}).items():
            *parents, leaf = path.split(".")
            node = tree
            for parent in parents:
                node = node[parent]
            node[leaf] = value
        return _namespace(tree)

    def __getattr__(self, name):
        tree = self.__dict__["tree"]
        if name in tree:
            return _namespace(tree[name])
        raise AttributeError(name)


class FakeVariable:
    def __init__(self, name, *, formula=None, adds=None, dtype=np.float32):
        self.name = name
        self.entity = SimpleNamespace(key="household")
        self.formula = formula
        self.formulas = {"0000-01-01": formula} if formula is not None else {}
        self.adds = adds
        self.subtracts = None
        self.dtype = dtype


class FakeHolder:
    def __init__(self, variable):
        self.variable = variable
        self.arrays: dict[str, np.ndarray] = {}
        self.inputs: set[str] = set()

    def get_known_periods(self):
        return list(self.arrays)

    def get_input_periods(self, branch_name="default"):
        return [period for period in self.arrays if period in self.inputs]

    def get_array(self, period, branch_name="default"):
        return self.arrays.get(str(period))

    def copy(self):
        holder = FakeHolder(self.variable)
        holder.arrays = {key: value.copy() for key, value in self.arrays.items()}
        holder.inputs = set(self.inputs)
        return holder


class FakeSimulation:
    def __init__(self, system, inputs, *, rows=N, baseline=True):
        self.tax_benefit_system = system
        self.rows = rows
        self.branch_name = "default"
        self.branches = {}
        self.holders = {
            name: FakeHolder(variable) for name, variable in system.variables.items()
        }
        for (name, period), values in inputs.items():
            self.set_input(name, period, values)
        if baseline:
            self.baseline = self.clone()

    def get_holder(self, name):
        return self.holders[name]

    def set_input(self, name, period, values):
        holder = self.holders[name]
        holder.arrays[str(period)] = np.array(values, dtype=holder.variable.dtype)
        holder.inputs.add(str(period))

    def calculate(self, name, period):
        holder = self.holders[name]
        cached = holder.arrays.get(str(period))
        if cached is not None:
            return cached
        variable = holder.variable
        result = self._run_formula(variable, None, period)
        if result is None:
            values = np.zeros(self.rows, dtype=variable.dtype)
        else:
            values = np.broadcast_to(
                np.asarray(result).astype(variable.dtype), (self.rows,)
            ).copy()
        holder.arrays[str(period)] = values
        return values

    def _run_formula(self, variable, population, period):
        if variable.formula is not None:
            return variable.formula(self, period)
        if variable.adds:
            total = 0
            for name in variable.adds:
                total = total + self.calculate(name, period)
            return total
        return None

    def clone(self):
        """Copy as policyengine-core does: every attribute but the branches,
        with the methods bound to the original bound to the copy."""

        new = object.__new__(type(self))
        for key, value in self.__dict__.items():
            if key != "branches":
                new.__dict__[key] = value
        for key, value in list(new.__dict__.items()):
            if isinstance(value, types.MethodType) and value.__self__ is self:
                new.__dict__[key] = types.MethodType(value.__func__, new)
        new.holders = {name: holder.copy() for name, holder in self.holders.items()}
        new.branches = {}
        return new

    def get_branch(self, name):
        if name not in self.branches:
            branch = self.clone()
            branch.branch_name = name
            branch.parent_branch = self
            self.branches[name] = branch
        return self.branches[name]


def _share(simulation, period):
    key = simulation.calculate("key", period).astype(np.float64)
    weight = simulation.calculate("weight", period).astype(np.float64)
    total = float(np.sum(key * weight))
    return key / total if total else np.zeros_like(key)


def _rates(simulation, period):
    return simulation.calculate("share", period) * 1_000.0


def _budget_change(simulation, period):
    budget = simulation.tax_benefit_system.parameters(period).gov.budget
    if budget == 0:
        return 0
    weight = simulation.calculate("weight", period)
    return budget / weight.sum() + 0 * weight


def _months(simulation, period):
    return np.full(simulation.rows, 6.0)


def _attendance(simulation, period):
    income = simulation.calculate("local", period)
    return income > np.median(income)


def _vat(rate_path):
    def formula(simulation, period):
        node = simulation.tax_benefit_system.parameters(period)
        for part in rate_path.split("."):
            node = getattr(node, part)
        return simulation.calculate("attendance", period).astype(np.float64) * node

    return formula


def _vat_change(simulation, period):
    return simulation.calculate("vat", period) - simulation.calculate(
        "baseline_vat", period
    )


def _tax_total(simulation, period):
    return simulation.calculate("vat", period) + simulation.calculate("rates", period)


def _decile(simulation, period):
    income = simulation.calculate("local", period)
    return np.argsort(np.argsort(income)) % 10


def _local(simulation, period):
    return simulation.calculate("key_a", period) * 2.0


def _fake_system(*, budget=0.0, vat_rate=0.2, baseline_vat_rate=0.2):
    names = {
        "weight": FakeVariable("weight"),
        "key_a": FakeVariable("key_a"),
        "key_b": FakeVariable("key_b"),
        "key": FakeVariable("key", adds=["key_a", "key_b"]),
        "share": FakeVariable("share", formula=_share),
        "rates": FakeVariable("rates", formula=_rates),
        "months": FakeVariable("months", formula=_months),
        "attendance": FakeVariable("attendance", formula=_attendance, dtype=bool),
        "vat": FakeVariable("vat", formula=_vat("gov.vat_rate")),
        "baseline_vat": FakeVariable(
            "baseline_vat", formula=_vat("baseline.gov.vat_rate")
        ),
        "vat_change": FakeVariable("vat_change", formula=_vat_change),
        "tax_total": FakeVariable("tax_total", formula=_tax_total),
        "budget_change": FakeVariable("budget_change", formula=_budget_change),
        "decile": FakeVariable("decile", formula=_decile, dtype=np.int64),
        "local": FakeVariable("local", formula=_local),
    }
    tree = {
        "gov": {"budget": budget, "vat_rate": vat_rate},
        "baseline": {"gov": {"vat_rate": baseline_vat_rate}},
    }
    return SimpleNamespace(variables=names, parameters=FakeParameters(tree))


def _quiet_reform(parameters, period):
    return None


def _loud_reform(parameters, period):
    return object()


def _late_reform(parameters, period):
    return object() if int(str(period)[:4]) == P else None


@pytest.fixture
def fake_engine(monkeypatch):
    """The engine's own module: its list, its window and its reform factories."""

    module = types.ModuleType(FAKE_ENGINE)
    module.WHOLE_POPULATION = frozenset(
        {"share", "months", "attendance", "budget_change", "decile"}
    )
    module.WINDOW = range(2026, 2033)
    module.quiet_reform = _quiet_reform
    module.loud_reform = _loud_reform
    module.late_reform = _late_reform
    monkeypatch.setitem(sys.modules, FAKE_ENGINE, module)
    return module


def _declaration(**changes):
    fields = {
        "engine": FAKE_ENGINE,
        "distribution": "fake-population-engine",
        "reviewed_version": "1.0.0",
        "list_source": f"{FAKE_ENGINE}:WHOLE_POPULATION",
        "weight_variable": "weight",
        "corrected": {"share": "key"},
        "key_adds": {"key": ("key_a", "key_b")},
        "stored": ("months", "attendance"),
        "zero_at_policy": {"budget_change": ParameterTest("gov.budget", "eq", 0)},
        "inert": {
            "attendance": InertRule(
                sink="vat_change",
                levels=("vat", "baseline_vat"),
                carriers=("tax_total",),
                predicate=all_of(
                    ParameterTest("gov.vat_rate", "eq_path", "baseline.gov.vat_rate")
                ),
            )
        },
        "refused": ("decile",),
        "window": PositionWindow(2026, f"{FAKE_ENGINE}:WINDOW", "positional draws"),
        "observed": ("rates",),
        "constructor_factories": (f"{FAKE_ENGINE}:quiet_reform",),
    }
    fields.update(changes)
    return PopulationDeclaration(**fields)


# Two clones of three households each: the first block's keys are larger.
KEY_A = np.array([10.0, 0.0, 30.0, 1.0, 2.0, 3.0])
KEY_B = np.array([5.0, 5.0, 0.0, 1.0, 0.0, 2.0])
WEIGHT = np.array([100.0, 200.0, 300.0, 150.0, 150.0, 150.0])
BLOCKS = {"0": np.array([0, 1, 2]), "1": np.array([3, 4, 5])}
GROWTH = 1.03


def _inputs(rows, *, weight_scale=1.0, store=("months", "attendance")):
    inputs = {}
    for period, factor in ((D, 1.0), (P, GROWTH)):
        inputs[("key_a", period)] = KEY_A[rows] * factor
        inputs[("key_b", period)] = KEY_B[rows] * factor
        inputs[("weight", period)] = WEIGHT[rows] * weight_scale
        if "months" in store:
            inputs[("months", period)] = np.full(len(rows), 3.0)
        if "attendance" in store:
            inputs[("attendance", period)] = np.zeros(len(rows), dtype=bool)
    return inputs


def _frame_sum(rows):
    return float(np.sum((KEY_A[rows] + KEY_B[rows]) * WEIGHT[rows]))


def _block_shares(*, scaled):
    pool_mass = float(WEIGHT.sum())
    scales = (
        {block: pool_mass / float(WEIGHT[rows].sum()) for block, rows in BLOCKS.items()}
        if scaled
        else None
    )
    shares = pool_shares(
        {block: {"share": _frame_sum(rows)} for block, rows in BLOCKS.items()},
        scales,
        key_variables={"share": "key"},
    )
    return shares, scales


def _block(block, *, scaled=True, store=("months", "attendance"), system=None):
    rows = BLOCKS[block]
    scale = 1.0
    if scaled:
        scale = float(WEIGHT.sum()) / float(WEIGHT[rows].sum())
    return FakeSimulation(
        system or _fake_system(),
        _inputs(rows, weight_scale=scale, store=store),
        rows=len(rows),
    )


def _pool():
    return FakeSimulation(_fake_system(), _inputs(np.arange(N)))


# ---------------------------------------------------------------------------
# Pool shares and the zero rules
# ---------------------------------------------------------------------------


def test_the_correction_is_the_blocks_scaled_share_of_the_pool_total():
    share = PoolShare(
        "share", "key", block_key_sum=2.0, pool_key_sum=12.0, weight_scale=3.0
    )
    assert share.factor() == pytest.approx(0.5)
    assert share.as_receipt() == {
        "formula": "share",
        "key_variable": "key",
        "correction": share.factor(),
    }


def test_block_corrections_split_the_pool_by_their_key_totals():
    shares, scales = _block_shares(scaled=True)
    total = sum(shares[block]["share"].factor() / scales[block] for block in BLOCKS)
    assert total == pytest.approx(1.0, rel=1e-15)
    assert shares["0"]["share"].pool_key_sum == pytest.approx(_frame_sum(np.arange(N)))


@pytest.mark.parametrize("scaled", [True, False])
def test_corrected_blocks_give_the_pools_share_at_every_period(scaled):
    pool = _pool()
    shares, _scales = _block_shares(scaled=scaled)
    for block, rows in BLOCKS.items():
        simulation = _block(block, scaled=scaled)
        injected = inject_pool_shares(
            simulation, shares[block], (D, P), weight_variable="weight"
        )
        assert injected.periods == (str(D), str(P))
        for period in (D, P):
            np.testing.assert_allclose(
                simulation.calculate("share", period),
                pool.calculate("share", period)[rows],
                rtol=1e-6,
            )
            np.testing.assert_array_equal(
                simulation.baseline.calculate("share", period),
                simulation.calculate("share", period),
            )
            for target in (simulation, simulation.baseline):
                assert str(period) in target.get_holder("share").inputs


def test_identical_copies_inject_their_own_values_bitwise():
    rows = np.arange(3)
    copies = {"0": rows, "1": rows}
    pool_mass = 2 * float(WEIGHT[rows].sum())
    shares = pool_shares(
        {block: {"share": _frame_sum(rows)} for block in copies},
        {block: pool_mass / float(WEIGHT[rows].sum()) for block in copies},
        key_variables={"share": "key"},
    )
    for block in copies:
        assert shares[block]["share"].factor() == 1.0
        simulation = FakeSimulation(
            _fake_system(), _inputs(rows, weight_scale=2.0), rows=3
        )
        own = simulation.clone().calculate("share", P)
        inject_pool_shares(simulation, shares[block], (P,), weight_variable="weight")
        np.testing.assert_array_equal(simulation.calculate("share", P), own)


def test_a_block_without_keys_takes_zero():
    rows = np.arange(3)
    inputs = _inputs(rows)
    for period in (D, P):
        inputs[("key_a", period)] = np.zeros(3)
        inputs[("key_b", period)] = np.zeros(3)
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    share = PoolShare("share", "key", block_key_sum=0.0, pool_key_sum=50.0)
    assert share.factor() == 0.0
    assert check_key_sums(
        simulation, {"share": share}, D, weight_variable="weight"
    ) == {"share": 0.0}
    inject_pool_shares(simulation, {"share": share}, (D, P), weight_variable="weight")
    np.testing.assert_array_equal(simulation.calculate("share", P), np.zeros(3))


def test_a_zero_block_total_over_nonzero_keys_is_refused():
    simulation = _block("0")
    share = PoolShare("share", "key", block_key_sum=0.0, pool_key_sum=50.0)
    with pytest.raises(PoolShareError, match="not every key"):
        check_key_sums(simulation, {"share": share}, D, weight_variable="weight")
    with pytest.raises(PoolShareError, match="not every key"):
        inject_pool_shares(simulation, {"share": share}, (D,), weight_variable="weight")


def test_a_zero_pool_total_under_a_nonzero_block_total_is_refused():
    share = PoolShare("share", "key", block_key_sum=3.0, pool_key_sum=0.0)
    with pytest.raises(PoolShareError, match="fallback branch"):
        share.factor()


@pytest.mark.parametrize(
    "fields",
    [
        {"block_key_sum": math.nan},
        {"pool_key_sum": math.inf},
        {"weight_scale": 0.0},
        {"weight_scale": -2.0},
    ],
)
def test_a_correction_needs_finite_totals_and_a_positive_scale(fields):
    values = {"block_key_sum": 1.0, "pool_key_sum": 2.0, "weight_scale": 1.0, **fields}
    with pytest.raises(PoolShareError):
        PoolShare("share", "key", **values)


def test_pool_shares_need_every_formula_in_every_block():
    with pytest.raises(PoolShareError, match="carries key sums"):
        pool_shares({"0": {"share": 1.0}, "1": {}}, key_variables={"share": "key"})
    with pytest.raises(PoolShareError, match="weight factor"):
        pool_shares({"0": {"share": 1.0}}, {"1": 1.0}, key_variables={"share": "key"})


# ---------------------------------------------------------------------------
# Checks (a) and (b)
# ---------------------------------------------------------------------------


def test_key_sums_check_the_engines_total_against_the_frames():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("1")
    gaps = check_key_sums(simulation, shares["1"], D, weight_variable="weight")
    assert gaps["share"] < 1e-6
    wrong = PoolShare(
        "share",
        "key",
        block_key_sum=shares["1"]["share"].block_key_sum * 1.01,
        pool_key_sum=shares["1"]["share"].pool_key_sum,
        weight_scale=shares["1"]["share"].weight_scale,
    )
    with pytest.raises(PoolShareError, match="frame total"):
        check_key_sums(simulation, {"share": wrong}, D, weight_variable="weight")


def test_uniform_growth_returns_one_factor_per_name():
    simulation = _block("0", scaled=False)
    factors = check_uniform_growth(simulation, ("key", "weight"), D, P)
    assert factors["key"] == pytest.approx(GROWTH, rel=1e-6)
    assert factors["weight"] == pytest.approx(1.0)


def test_growth_that_varies_by_row_is_refused():
    inputs = _inputs(BLOCKS["0"])
    # The third row's key grows by 1.0 where the others grow by 1.03.
    inputs[("key_a", P)] = inputs[("key_a", D)] * np.array([1.03, 1.03, 1.0])
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    with pytest.raises(PoolShareError, match="one factor per row"):
        check_uniform_growth(simulation, ("key",), D, P)


def test_the_growth_factor_is_the_ratio_of_totals():
    # Stored in float32, rows grown by one factor round apart: the factor is
    # the totals' ratio, not any one row's.
    inputs = _inputs(BLOCKS["0"])
    inputs[("key_a", D)] = np.array([10.1, 0.0, 30.7])
    inputs[("key_a", P)] = inputs[("key_a", D)] * GROWTH
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    then = simulation.calculate("key", D).astype(np.float64)
    now = simulation.calculate("key", P).astype(np.float64)
    ratios = now / then
    assert ratios.max() > ratios.min()
    factor = check_uniform_growth(simulation, ("key",), D, P)["key"]
    assert factor == pytest.approx(float(now.sum() / then.sum()), rel=1e-15)
    assert all(factor != pytest.approx(ratio, rel=1e-12) for ratio in ratios)


def test_a_row_a_few_parts_in_a_hundred_thousand_off_is_refused():
    # Every row ends 1.2e-5 or more from the totals' factor, above the 1e-5
    # tolerance.
    inputs = _inputs(BLOCKS["0"])
    inputs[("key_a", P)] = inputs[("key_a", D)] * GROWTH * np.array([1, 1, 1 + 3e-5])
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    with pytest.raises(PoolShareError, match="one factor per row"):
        check_uniform_growth(simulation, ("key",), D, P)


def test_a_row_that_appears_after_the_dataset_year_is_refused():
    inputs = _inputs(BLOCKS["0"])
    # key_a is zero on the second row at D; give it a value at P only.
    inputs[("key_a", P)] = np.array([10.3, 4.0, 30.9])
    inputs[("key_b", D)] = np.zeros(3)
    inputs[("key_b", P)] = np.zeros(3)
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    with pytest.raises(PoolShareError, match="zero at"):
        check_uniform_growth(simulation, ("key",), D, P)


def test_a_name_no_row_carries_has_no_factor():
    inputs = _inputs(BLOCKS["0"])
    for period in (D, P):
        inputs[("key_a", period)] = np.zeros(3)
        inputs[("key_b", period)] = np.zeros(3)
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    assert check_uniform_growth(simulation, ("key",), D, P) == {"key": None}


def test_factor_spread_is_relative_and_skips_blocks_without_carriers():
    spread = factor_spread(
        {"0": {"key": 1.03, "weight": 1.0}, "1": {"key": 1.0300103, "weight": None}}
    )
    assert spread["key"] == pytest.approx(1e-5, rel=1e-3)
    assert spread["weight"] == 0.0


# ---------------------------------------------------------------------------
# Injection into every simulation a block builds
# ---------------------------------------------------------------------------


def test_a_formula_already_an_input_is_not_injected_twice():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")
    with pytest.raises(PoolShareError, match="already an input"):
        inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")


def test_siblings_take_the_same_arrays_only_while_they_hold_the_same_values():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    injected = inject_pool_shares(
        simulation, shares["0"], (D, P), weight_variable="weight"
    )
    sibling = _block("0")
    injected.apply(sibling)
    for period in (D, P):
        np.testing.assert_array_equal(
            sibling.calculate("share", period), simulation.calculate("share", period)
        )
        np.testing.assert_array_equal(
            sibling.baseline.calculate("share", period),
            simulation.calculate("share", period),
        )
    assert injected.matches(simulation) and injected.matches(sibling)
    assert injected.as_receipt() == {
        "periods": [str(D), str(P)],
        "corrections": {"share": shares["0"]["share"].factor()},
    }

    substituted = _block("0")
    substituted.set_input("key_a", P, np.zeros(3))
    with pytest.raises(PoolShareError, match="differs"):
        injected.apply(substituted)

    sibling.set_input("weight", D, np.ones(3))
    assert not injected.matches(sibling)


def test_injection_reads_every_own_value_before_setting_any():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    expected = {
        period: simulation.clone().calculate("share", period) for period in (D, P)
    }
    injected = inject_pool_shares(
        simulation, shares["0"], (P, D), weight_variable="weight"
    )
    factor = injected.factors["share"]
    for period in (D, P):
        np.testing.assert_array_equal(
            simulation.calculate("share", period),
            (expected[period].astype(np.float64) * factor).astype(np.float32),
        )


def test_a_baseline_whose_own_shares_differ_is_refused():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    simulation.baseline.set_input("key_a", D, KEY_A[BLOCKS["0"]] + 1.0)
    with pytest.raises(PoolShareError, match="its baseline"):
        inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")


def test_a_zero_block_whose_keys_turn_nonzero_later_is_refused():
    inputs = _inputs(BLOCKS["0"])
    inputs[("key_a", D)] = np.zeros(3)
    inputs[("key_b", D)] = np.zeros(3)
    simulation = FakeSimulation(_fake_system(), inputs, rows=3)
    share = PoolShare("share", "key", block_key_sum=0.0, pool_key_sum=50.0)
    with pytest.raises(PoolShareError, match=f"not every key is at {P}"):
        inject_pool_shares(
            simulation, {"share": share}, (D, P), weight_variable="weight"
        )


def test_a_sibling_whose_baseline_already_holds_the_shares_is_refused():
    shares, _scales = _block_shares(scaled=True)
    injected = inject_pool_shares(
        _block("0"), shares["0"], (D, P), weight_variable="weight"
    )
    sibling = _block("0")
    # The baseline's input is its own value, so only the refusal catches it.
    own = sibling.baseline.clone().calculate("share", P)
    sibling.baseline.set_input("share", P, own)
    with pytest.raises(PoolShareError, match="already an input"):
        injected.apply(sibling)


def test_a_sibling_baseline_changed_after_apply_no_longer_matches():
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    injected = inject_pool_shares(
        simulation, shares["0"], (D, P), weight_variable="weight"
    )
    sibling = _block("0")
    injected.apply(sibling)
    assert injected.matches(sibling)
    sibling.baseline.set_input("weight", D, np.ones(3))
    assert not injected.matches(sibling)


def test_shares_go_in_before_branches_and_before_arming(fake_engine):
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    simulation.get_branch("early")
    with pytest.raises(PoolShareError, match="live branches"):
        inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")
    simulation = _block("0")
    simulation.baseline.get_branch("early")
    with pytest.raises(PoolShareError, match="live branches"):
        inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")

    simulation = _block("0")
    recorder = _recorder()
    recorder.arm(simulation)
    with pytest.raises(PoolShareError, match="before arming"):
        inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")

    injected = inject_pool_shares(
        _block("0"), shares["0"], (D,), weight_variable="weight"
    )
    sibling = _block("0")
    recorder = _recorder()
    recorder.arm(sibling)
    with pytest.raises(PoolShareError, match="before arming"):
        injected.apply(sibling)
    sibling = _block("0")
    sibling.get_branch("early")
    with pytest.raises(PoolShareError, match="live branches"):
        injected.apply(sibling)


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------


def _recorder(declaration=None):
    return FormulaRunRecorder.for_declaration(declaration or _declaration())


def test_the_recorder_binds_per_instance_and_follows_core_clones(fake_engine):
    simulation = _block("0")
    recorder = _recorder()
    method = vars(FakeSimulation)["_run_formula"]
    recorder.arm(simulation)
    assert "_run_formula" in vars(simulation)
    assert "_run_formula" in vars(simulation.baseline)
    assert vars(FakeSimulation)["_run_formula"] is method
    assert recorder.is_armed(simulation) and recorder.is_armed(simulation.baseline)

    simulation.calculate("rates", P)
    branch = simulation.get_branch("after_arming")
    assert branch._run_formula.__self__ is branch
    branch.calculate("local", D)
    tokens = {run.simulation for run in recorder.runs}
    assert recorder.token(branch) is not None
    assert recorder.token(branch) != recorder.token(simulation)
    assert recorder.token(simulation) in tokens
    names = {(run.name, run.period) for run in recorder.runs}
    assert ("share", str(P)) in names and ("rates", str(P)) in names
    assert "local" in recorder.periods_run()[str(D)]

    recorder.disarm()
    assert "_run_formula" not in vars(simulation)
    assert "_run_formula" not in vars(branch)
    before = len(recorder.runs)
    simulation.calculate("rates", D)
    assert len(recorder.runs) == before


def test_a_copy_that_inherits_a_token_gets_its_own():
    simulation = _block("0")
    recorder = _recorder()
    recorder.arm(simulation)
    simulation.calculate("share", P)
    copied = simulation.clone()
    assert vars(copied)["_microcosm_formula_run_token"] == recorder.token(simulation)
    assert recorder.token(copied) is None
    copied.holders["share"].arrays.clear()
    copied.calculate("share", P)
    assert recorder.token(copied) not in (None, recorder.token(simulation))


def test_predicates_and_zeros_are_read_at_the_run_in_that_simulation():
    simulation = _block("0")
    recorder = _recorder()
    recorder.arm(simulation)
    simulation.calculate("budget_change", D)
    branch = simulation.get_branch("reformed")
    branch.tax_benefit_system = _fake_system(budget=5.0)
    branch.calculate("budget_change", P)
    runs = {run.period: run for run in recorder.runs if run.name == "budget_change"}
    assert runs[str(D)].predicate_holds and runs[str(D)].all_zero
    assert runs[str(P)].predicate_holds is False
    assert runs[str(P)].all_zero is False


def test_arming_refuses_a_foreign_run_formula_but_replaces_a_disarmed_recorders():
    simulation = _block("0")
    simulation._run_formula = types.MethodType(
        lambda self, variable, population, period: None, simulation
    )
    with pytest.raises(ValueError, match="own _run_formula"):
        _recorder().arm(simulation)

    simulation = _block("0")
    first = _recorder()
    first.arm(simulation)
    with pytest.raises(ValueError, match="own _run_formula"):
        _recorder().arm(simulation)
    copied = simulation.clone()
    first.disarm()
    second = _recorder()
    second.arm(copied)
    assert second.is_armed(copied)


def test_a_refused_arming_binds_nothing():
    simulation = _block("0")
    simulation.baseline._run_formula = types.MethodType(
        lambda self, variable, population, period: None, simulation.baseline
    )
    recorder = _recorder()
    with pytest.raises(ValueError, match="own _run_formula"):
        recorder.arm(simulation)
    assert "_run_formula" not in vars(simulation)
    assert not recorder.is_armed(simulation)
    simulation.calculate("rates", P)
    assert recorder.runs == () and recorder.periods_run() == {}


class FakePeriod:
    """A policyengine-core-style period: a unit, a start and a stop."""

    def __init__(self, text, unit, first, last):
        self.text, self.unit = text, unit
        self.start = SimpleNamespace(year=first)
        self.stop = SimpleNamespace(year=last)

    def __str__(self):
        return self.text


@pytest.mark.parametrize(
    "period, reaches",
    [
        (2025, False),
        (2026, True),
        ("2025", False),
        ("2025-12", False),
        ("2026-01-01", True),
        ("ETERNITY", True),
        ("year:2024:2", True),  # A form not read: taken to reach the window.
        (FakePeriod("year:2025:2", "year", 2025, 2026), True),
        (FakePeriod("year:2024:2", "year", 2024, 2025), False),
        (FakePeriod("2025-12", "month", 2025, 2025), False),
        (FakePeriod("ETERNITY", "eternity", 1, math.inf), True),
    ],
)
def test_a_period_reaches_the_window_by_its_last_year(period, reaches):
    window = PositionWindow(2026, f"{FAKE_ENGINE}:WINDOW", "positional draws")
    assert window.contains(period) is reaches


def test_runs_reach_the_window_by_the_years_their_own_period_covers(fake_engine):
    simulation, recorder = _armed_block()
    across = FakePeriod("year:2025:2", "year", 2025, 2026)
    month = FakePeriod("2025-12", "month", 2025, 2025)
    simulation.calculate("local", across)
    simulation.calculate("local", month)
    assert recorder.period_span("year:2025:2") == (2025, 2026)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    window = {(v.name, v.period) for v in verdict.violations if v.kind == "window_run"}
    assert ("local", "year:2025:2") in window
    assert not any(period == "2025-12" for _name, period in window)


def test_predicates_must_be_for_watched_formulas():
    with pytest.raises(ValueError, match="unwatched"):
        FormulaRunRecorder({"share"}, predicates={"other": ParameterTest("a", "eq", 0)})


def test_the_walk_finds_values_calculated_before_arming():
    simulation = _block("0")
    simulation.calculate("decile", P)
    simulation.get_branch("subset").calculate("budget_change", D)
    assert derived_watched_periods(
        [simulation], ("decile", "budget_change", "share")
    ) == {
        "budget_change": (str(D),),
        "decile": (str(P),),
    }


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


def _armed_block(
    block="0", *, scaled=True, store=("months", "attendance"), system=None
):
    shares, _scales = _block_shares(scaled=scaled)
    simulation = _block(block, scaled=scaled, store=store, system=system)
    inject_pool_shares(simulation, shares[block], (D, P), weight_variable="weight")
    recorder = _recorder()
    recorder.arm(simulation)
    return simulation, recorder


def _kinds(verdict):
    return {(v.kind, v.name, v.period) for v in verdict.violations}


def test_a_corrected_block_with_stored_inputs_and_zero_formulas_is_clean(fake_engine):
    simulation, recorder = _armed_block()
    for name in ("tax_total", "budget_change", "local"):
        simulation.calculate(name, P)
    verdict = judge(
        _declaration(), MEASURE_PASS, recorder, [simulation], (D, P), [("tax_total", P)]
    )
    assert verdict.clean, verdict.describe()
    assert verdict.classes == {
        "corrected": ("share",),
        "stored": ("attendance", "months"),
        "zero_at_policy": ("budget_change",),
        "inert": (),
    }
    receipt = verdict.as_receipt()
    assert receipt["pass"] == "measure" and receipt["clean"] is True
    assert receipt["window"] == {"first_year": 2026, "violations": []}
    assert receipt["refused"] == [] and receipt["violations"] == []
    assert receipt["declaration_sha256"] == _declaration().sha256
    assert receipt["engine"] == {"package": "fake-population-engine", "version": None}


def test_a_share_read_at_a_period_never_injected_is_a_violation(fake_engine):
    simulation, recorder = _armed_block()
    simulation.calculate("rates", 2023)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("corrected_derived", "share", "2023") in _kinds(verdict)
    assert not verdict.clean


def test_a_share_not_injected_at_a_computed_period_is_a_violation(fake_engine):
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    inject_pool_shares(simulation, shares["0"], (D,), weight_variable="weight")
    recorder = _recorder()
    recorder.arm(simulation)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("corrected_not_input", "share", str(P)) in _kinds(verdict)


def test_a_stored_formula_the_data_lacks_is_a_violation_in_the_measure_pass(
    fake_engine,
):
    simulation, recorder = _armed_block(store=("attendance",))
    simulation.calculate("months", P)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert {
        ("stored_not_input", "months", str(D)),
        ("stored_not_input", "months", str(P)),
        ("stored_derived", "months", str(P)),
    } <= _kinds(verdict)


def test_a_zero_formula_out_of_policy_or_nonzero_is_a_violation(fake_engine):
    system = _fake_system(budget=5.0)
    simulation, recorder = _armed_block(system=system)
    simulation.calculate("budget_change", P)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("zero_at_policy", "budget_change", str(P)) in _kinds(verdict)
    assert "budget_change" not in verdict.classes["zero_at_policy"]


def test_a_zero_formula_calculated_before_arming_is_judged_from_its_held_value(
    fake_engine,
):
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", system=_fake_system(budget=5.0))
    simulation.calculate("budget_change", D)
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    recorder = _recorder()
    recorder.arm(simulation)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("zero_at_policy", "budget_change", str(D)) in _kinds(verdict)


def _reformed_branch(simulation, name, **system):
    branch = simulation.get_branch(name)
    branch.tax_benefit_system = _fake_system(**system)
    return branch


def test_a_held_zero_formula_is_judged_where_it_is_held(fake_engine):
    # A branch made before arming holds a nonzero value; the main simulation
    # then runs the formula cleanly at the same period. The clean run says
    # nothing of the branch's value.
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0")
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    _reformed_branch(simulation, "reformed", budget=5.0).calculate("budget_change", P)
    recorder = _recorder()
    recorder.arm(simulation)
    simulation.calculate("budget_change", P)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("zero_at_policy", "budget_change", str(P)) in _kinds(verdict)
    assert "budget_change" not in verdict.classes["zero_at_policy"]


@pytest.mark.parametrize(
    "predicate, budget",
    [
        # The predicate holds but the value is not zero.
        (ParameterTest("gov.vat_rate", "gt", 0), 5.0),
        # The value is zero but the predicate fails.
        (ParameterTest("gov.vat_rate", "eq", 0), 0.0),
    ],
)
def test_a_zero_formula_needs_its_predicate_and_zero_values(
    fake_engine, predicate, budget
):
    declaration = _declaration(zero_at_policy={"budget_change": predicate})
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", system=_fake_system(budget=budget))
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    recorder = FormulaRunRecorder.for_declaration(declaration)
    recorder.arm(simulation)
    simulation.calculate("budget_change", P)
    verdict = judge(declaration, MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("zero_at_policy", "budget_change", str(P)) in _kinds(verdict)

    # The same, held from before arming, is judged from the held value.
    simulation = _block("0", system=_fake_system(budget=budget))
    simulation.calculate("budget_change", D)
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    recorder = FormulaRunRecorder.for_declaration(declaration)
    recorder.arm(simulation)
    verdict = judge(declaration, MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("zero_at_policy", "budget_change", str(D)) in _kinds(verdict)


def test_values_held_from_before_arming_are_judged_as_runs(fake_engine):
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", store=())
    simulation.calculate("share", 2023)
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    for name in ("decile", "months", "tax_total"):
        simulation.calculate(name, P)
    recorder = _recorder()
    recorder.arm(simulation)
    assert recorder.runs == ()
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert {
        ("corrected_derived", "share", "2023"),
        ("refused_ran", "decile", str(P)),
        ("stored_derived", "months", str(P)),
        ("inert_carrier", "tax_total", str(P)),
    } <= _kinds(raised.value.verdict)


def test_a_baseline_without_the_injected_shares_is_a_violation(fake_engine):
    simulation, recorder = _armed_block()
    holder = simulation.baseline.get_holder("share")
    holder.inputs.discard(str(P))
    holder.arrays.pop(str(P))
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("corrected_not_input", "share", str(P)) in _kinds(verdict)


def test_attendance_may_run_in_a_spine_batch_through_a_zero_sink(fake_engine):
    simulation, recorder = _armed_block(store=("months",))
    simulation.calculate("vat_change", P)
    verdict = judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert verdict.clean, verdict.describe()
    assert verdict.classes["inert"] == ("attendance",)

    raising = PassPolicy("measure_raise", inert_allowed=False, on_violation="raise")
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), raising, recorder, [simulation], (D, P))
    assert ("stored_not_input", "attendance", str(D)) in _kinds(raised.value.verdict)
    assert "stored_not_input attendance@2024" in str(raised.value)


def test_attendance_that_reaches_a_level_or_a_nonzero_sink_raises_in_the_spine(
    fake_engine,
):
    simulation, recorder = _armed_block(store=("months",))
    simulation.calculate("tax_total", P)
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert {
        ("inert_carrier", "tax_total", str(P)),
        ("inert_without_sink", "attendance", str(P)),
    } <= _kinds(raised.value.verdict)

    reformed = _fake_system(vat_rate=0.25)
    simulation, recorder = _armed_block(store=("months",), system=reformed)
    simulation.calculate("vat_change", P)
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert {
        ("inert_sink", "vat_change", str(P)),
        ("inert_predicate", "attendance", str(P)),
    } <= _kinds(raised.value.verdict)

    simulation, recorder = _armed_block(store=("months",))
    with pytest.raises(PopulationGuardError) as raised:
        judge(
            _declaration(),
            SPINE_BATCHED_PASS,
            recorder,
            [simulation],
            (D, P),
            [("vat", P)],
        )
    assert ("inert_request", "vat", str(P)) in _kinds(raised.value.verdict)


def test_a_held_sink_is_judged_where_it_is_held(fake_engine):
    # A branch made before arming, under a reformed rate, holds a nonzero
    # sink; the main simulation's clean run of the sink at the same period
    # says nothing of it.
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", store=("months",))
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    reformed = _reformed_branch(simulation, "reformed", vat_rate=0.25)
    assert np.any(reformed.calculate("vat_change", P) != 0)
    recorder = _recorder()
    recorder.arm(simulation)
    assert np.all(simulation.calculate("vat_change", P) == 0)
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert ("inert_sink", "vat_change", str(P)) in _kinds(raised.value.verdict)


def test_every_held_sink_at_a_period_is_judged(fake_engine):
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", store=("months",))
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    simulation.calculate("vat_change", P)
    _reformed_branch(simulation, "reformed", vat_rate=0.25).calculate("vat_change", P)
    recorder = _recorder()
    recorder.arm(simulation)
    with pytest.raises(PopulationGuardError) as raised:
        judge(_declaration(), SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert ("inert_sink", "vat_change", str(P)) in _kinds(raised.value.verdict)


def _inert_declaration(predicate):
    rule = InertRule(
        sink="vat_change",
        levels=("vat", "baseline_vat"),
        carriers=("tax_total",),
        predicate=predicate,
    )
    return _declaration(inert={"attendance": rule})


@pytest.mark.parametrize(
    "predicate, vat_rate",
    [
        # The predicate holds but the sink is not zero.
        (ParameterTest("gov.budget", "eq", 0), 0.25),
        # The sink is zero but the predicate fails.
        (ParameterTest("gov.budget", "ne", 0), 0.2),
    ],
)
def test_a_sink_needs_its_predicate_and_zero_values(fake_engine, predicate, vat_rate):
    declaration = _inert_declaration(predicate)
    shares, _scales = _block_shares(scaled=True)
    simulation = _block("0", store=("months",), system=_fake_system(vat_rate=vat_rate))
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    recorder = FormulaRunRecorder.for_declaration(declaration)
    recorder.arm(simulation)
    simulation.calculate("vat_change", P)
    with pytest.raises(PopulationGuardError) as raised:
        judge(declaration, SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert ("inert_sink", "vat_change", str(P)) in _kinds(raised.value.verdict)

    # The same, held from before arming, is judged from the held value.
    simulation = _block("0", store=("months",), system=_fake_system(vat_rate=vat_rate))
    inject_pool_shares(simulation, shares["0"], (D, P), weight_variable="weight")
    simulation.calculate("vat_change", P)
    recorder = FormulaRunRecorder.for_declaration(declaration)
    recorder.arm(simulation)
    with pytest.raises(PopulationGuardError) as raised:
        judge(declaration, SPINE_BATCHED_PASS, recorder, [simulation], (D, P))
    assert ("inert_sink", "vat_change", str(P)) in _kinds(raised.value.verdict)


def test_a_refused_rank_is_a_violation_run_or_requested(fake_engine):
    simulation, recorder = _armed_block()
    simulation.calculate("decile", P)
    verdict = judge(
        _declaration(), MEASURE_PASS, recorder, [simulation], (D, P), [("decile", P)]
    )
    assert {
        ("refused_ran", "decile", str(P)),
        ("refused_request", "decile", str(P)),
    } <= _kinds(verdict)
    assert verdict.as_receipt()["refused"] == ["decile"]


def test_work_in_the_position_window_is_a_violation(fake_engine):
    simulation, recorder = _armed_block()
    simulation.calculate("local", 2026)
    verdict = judge(
        _declaration(),
        MEASURE_PASS,
        recorder,
        [simulation],
        (D, P, 2027),
        [("local", 2028)],
    )
    expected = {
        ("window_run", "local", "2026"),
        ("window_run", "key_a", "2026"),
        ("window_period", None, "2027"),
        ("window_request", "local", "2028"),
    }
    assert expected <= _kinds(verdict)
    window = verdict.as_receipt()["window"]
    assert window["first_year"] == 2026
    assert {
        (item["kind"], item["name"], item["period"]) for item in window["violations"]
    } == expected


def test_a_structural_reform_switched_on_is_a_violation(fake_engine):
    simulation, recorder = _armed_block()
    declaration = _declaration(
        constructor_factories=(
            f"{FAKE_ENGINE}:quiet_reform",
            f"{FAKE_ENGINE}:loud_reform",
        )
    )
    verdict = judge(declaration, MEASURE_PASS, recorder, [simulation], (D, P))
    assert ("constructor_reform", f"{FAKE_ENGINE}:loud_reform", str(D)) in _kinds(
        verdict
    )


def test_a_structural_reform_switched_on_at_a_later_period_is_a_violation(
    fake_engine,
):
    simulation, recorder = _armed_block()
    declaration = _declaration(constructor_factories=(f"{FAKE_ENGINE}:late_reform",))
    verdict = judge(declaration, MEASURE_PASS, recorder, [simulation], (D, P))
    assert _kinds(verdict) == {
        ("constructor_reform", f"{FAKE_ENGINE}:late_reform", str(P))
    }


def test_a_verdict_lists_at_most_fifty_violations(fake_engine):
    simulation, recorder = _armed_block()
    requested = [("decile", year) for year in range(1950, 2010)]
    verdict = judge(
        _declaration(), MEASURE_PASS, recorder, [simulation], (D, P), requested
    )
    assert verdict.truncated and len(verdict.violations) == 50
    assert verdict.as_receipt()["truncated"] is True
    assert verdict.describe().endswith("(and more).")


def test_a_simulation_the_recorder_never_armed_is_a_violation(fake_engine):
    simulation, recorder = _armed_block()
    sibling = _block("0")
    verdict = judge(
        _declaration(), MEASURE_PASS, recorder, [simulation, sibling], (D, P)
    )
    assert ("not_armed", None, None) in _kinds(verdict)


def test_a_verdict_carries_names_periods_and_booleans_only(fake_engine):
    simulation, recorder = _armed_block(store=())
    for period in (2023, 2026):
        simulation.calculate("tax_total", period)
        simulation.calculate("decile", period)
    verdict = judge(_declaration(), MEASURE_PASS, recorder, [simulation], (D, P))

    def leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from leaves(item)
        else:
            yield value

    for leaf in leaves(verdict.as_receipt()):
        assert leaf is None or isinstance(leaf, (str, bool, int)), leaf
        assert not isinstance(leaf, float)


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"refused": ("decile", "share")}, "both"),
        ({"stored": ("months",)}, "inert formula must also be declared stored"),
        ({"corrected": {"share": "other_key"}}, "name no inputs"),
        ({"refused": ("decile", "decile")}, "repeat"),
    ],
)
def test_a_declaration_refuses_overlapping_or_incomplete_classes(changes, message):
    with pytest.raises(ValueError, match=message):
        _declaration(**changes)


def test_the_declaration_digest_follows_its_content():
    assert _declaration().sha256 == _declaration().sha256
    assert _declaration().sha256 != _declaration(observed=()).sha256
    assert _declaration().classified() == {
        "share",
        "months",
        "attendance",
        "budget_change",
        "decile",
    }
    assert _declaration().watched() == _declaration().classified() | {
        "rates",
        "vat",
        "baseline_vat",
        "vat_change",
        "tax_total",
    }
    assert set(_declaration().run_predicates()) == {
        "budget_change",
        "attendance",
        "vat_change",
    }


def test_predicates_combine_parameter_tests():
    parameters = _fake_system(budget=0.0).parameters(D)
    assert ParameterTest("gov.budget", "eq", 0).holds(parameters)
    assert ParameterTest("gov.vat_rate", "gt", 0.1).holds(parameters)
    assert any_of(
        ParameterTest("gov.budget", "ne", 0), ParameterTest("gov.vat_rate", "le", 0.2)
    ).holds(parameters)
    assert not all_of(
        ParameterTest("gov.budget", "eq", 0), ParameterTest("gov.vat_rate", "lt", 0.2)
    ).holds(parameters)
    with pytest.raises(ValueError):
        ParameterTest("gov.budget", "approx", 0)
    with pytest.raises(ValueError):
        ParameterTest("gov.budget", "eq_path", 0)
    with pytest.raises(ValueError):
        all_of()


@pytest.fixture
def pinned_package(tmp_path, monkeypatch):
    package = tmp_path / "fake_pinned_engine"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "share.py").write_text(
        "def share_formula(key, total):\n    return key / total\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    importlib.invalidate_caches()
    yield package
    sys.modules.pop("fake_pinned_engine.share", None)
    sys.modules.pop("fake_pinned_engine", None)


def test_source_pins_hash_files_and_objects(pinned_package):
    import hashlib

    path = pinned_package / "share.py"
    assert source_sha256("fake_pinned_engine/share.py") == (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert source_sha256("fake_pinned_engine.share:share_formula") == (
        hashlib.sha256(path.read_text().encode()).hexdigest()
    )
    with pytest.raises(FileNotFoundError):
        source_sha256("fake_pinned_engine/missing.py")


def test_validation_passes_on_the_engine_it_describes(fake_engine, pinned_package):
    digest = source_sha256("fake_pinned_engine/share.py")
    declaration = _declaration(digests={"fake_pinned_engine/share.py": digest})
    validate_declaration(declaration, _fake_system())


def test_validation_fails_closed_on_every_kind_of_drift(fake_engine, pinned_package):
    digest = source_sha256("fake_pinned_engine/share.py")
    declaration = _declaration(
        digests={"fake_pinned_engine/share.py": digest},
        constructor_factories=(f"{FAKE_ENGINE}:WINDOW",),
        zero_at_policy={"budget_change": ParameterTest("gov.missing", "eq", 0)},
    )
    fake_engine.WHOLE_POPULATION = fake_engine.WHOLE_POPULATION | {"extra"}
    fake_engine.WINDOW = range(2027, 2033)
    (pinned_package / "share.py").write_text("def share_formula():\n    return 1\n")
    system = _fake_system()
    system.variables["key"].adds = ["key_a"]
    with pytest.raises(PopulationDeclarationError) as raised:
        validate_declaration(declaration, system)
    message = str(raised.value)
    for fragment in (
        "lists ['extra']",
        "adds ['key_a']",
        "fake_pinned_engine/share.py has changed",
        "no longer starts the window at 2026",
        "is not callable",
        "no parameter gov.missing",
    ):
        assert fragment in message, message


def test_validation_requires_formulas_over_plain_inputs(fake_engine):
    system = _fake_system()
    system.variables["share"].formulas = {}
    system.variables["key_b"].formulas = {"0000-01-01": _local}
    with pytest.raises(PopulationDeclarationError) as raised:
        validate_declaration(_declaration(), system)
    assert "share is no longer a formula" in str(raised.value)
    assert "key_b is no longer a plain household input" in str(raised.value)


@pytest.mark.parametrize(
    "change, fragment",
    [
        (
            lambda variables: setattr(variables["key"], "subtracts", ["key_b"]),
            "key is no longer a plain sum of its inputs",
        ),
        (
            lambda variables: setattr(
                variables["key"], "formulas", {"0000-01-01": _local}
            ),
            "key is no longer a plain sum of its inputs",
        ),
        (
            lambda variables: setattr(
                variables["share"], "entity", SimpleNamespace(key="person")
            ),
            "share, key and weight no longer share an entity",
        ),
        (
            lambda variables: setattr(
                variables["weight"], "entity", SimpleNamespace(key="person")
            ),
            "share, key and weight no longer share an entity",
        ),
        (
            lambda variables: setattr(
                variables["key_a"], "entity", SimpleNamespace(key="person")
            ),
            "key_a is no longer a plain household input",
        ),
    ],
)
def test_validation_requires_one_entity_and_a_plain_key(fake_engine, change, fragment):
    system = _fake_system()
    change(system.variables)
    with pytest.raises(PopulationDeclarationError) as raised:
        validate_declaration(_declaration(), system)
    assert fragment in str(raised.value)


def test_pass_policies_record_or_raise():
    assert MEASURE_PASS.on_violation == "record" and not MEASURE_PASS.inert_allowed
    assert (
        SPINE_BATCHED_PASS.on_violation == "raise" and SPINE_BATCHED_PASS.inert_allowed
    )
    with pytest.raises(ValueError):
        PassPolicy("other", inert_allowed=False, on_violation="warn")


# ---------------------------------------------------------------------------
# The float32 bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, roundings",
    [(1, 0), (7, 6), (8, 3), (15, 10), (127, 24), (128, 18), (129, 12), (255, 19)],
)
def test_sum_roundings_follow_numpys_pairwise_leaves(n, roundings):
    assert float32_sum_roundings(n) == roundings


def test_the_leaf_term_does_not_shrink_with_n():
    assert float32_sum_roundings(2**20) - float32_sum_roundings(2**10) == 10
    assert float32_sum_roundings(1024) > math.log2(1024)


@pytest.mark.parametrize("n", [5, 100, 1_000, 40_000, 300_000])
def test_numpys_float32_sums_stay_inside_the_bound(n):
    rng = np.random.default_rng(n)
    for values in (
        rng.lognormal(3.0, 2.0, n).astype(np.float32),
        np.full(n, np.float32(1.0) + np.float32(2.0**-23)),
    ):
        exact = math.fsum(values.astype(np.float64))
        error = abs(float(values.sum(dtype=np.float32)) - exact)
        assert error <= float32_sum_roundings(n) * FLOAT32_UNIT_ROUNDOFF * exact


def test_the_share_bound_grows_with_the_sums_and_the_projection():
    base = pool_share_rtol(8_000, 120_000, key_inputs=1)
    assert base == pytest.approx(
        (float32_sum_roundings(8_000) + float32_sum_roundings(120_000) + 13)
        * FLOAT32_UNIT_ROUNDOFF
    )
    assert pool_share_rtol(
        8_000, 120_000, key_inputs=1, projection_years=1
    ) == pytest.approx(base + 4 * FLOAT32_UNIT_ROUNDOFF)
    assert pool_share_rtol(80_000, 1_200_000, key_inputs=12) < 1e-5


def test_the_share_bound_counts_the_adds_that_form_the_key():
    # Each of a key's float32 adds can round in the block's sum one way and
    # in the pool's the other: two roundings per add beyond the first input.
    one = pool_share_rtol(1, 2, key_inputs=1)
    assert pool_share_rtol(1, 2, key_inputs=2) == pytest.approx(
        one + 2 * FLOAT32_UNIT_ROUNDOFF
    )
    assert pool_share_rtol(1, 2, key_inputs=12) == pytest.approx(
        one + 22 * FLOAT32_UNIT_ROUNDOFF
    )
    with pytest.raises(ValueError):
        pool_share_rtol(1, 2, key_inputs=0)


def _engine_adds(parts):
    """A key as policyengine-core's adds form it: 0 plus each float32 input
    in turn, in float32."""

    total = 0
    for part in parts:
        total = total + np.asarray(part, dtype=np.float32)
    return np.asarray(total, dtype=np.float32)


def test_a_twelve_input_key_that_rounds_apart_stays_inside_the_bound():
    # Two made-up one-person households. The block's household's eleven small
    # inputs each round down when added to its first, the rest of the pool's
    # each round up; the frame sums them in float64, as c_b is formed.
    u = 2.0**-24
    down = np.float32(u * (1 - 2.0**-23))
    up = np.float32(u * (1 + 2.0**-22))
    parts = {
        "block": [np.float32(1.0)] + [down] * 11,
        "rest": [np.float32(1.0)] + [up] * 11,
    }
    weight = {"block": 1.0, "rest": 1023.0}
    key = {name: _engine_adds([[part] for part in parts[name]]) for name in parts}
    frame = {
        name: math.fsum(float(part) for part in parts[name]) * weight[name]
        for name in parts
    }
    scale = (weight["block"] + weight["rest"]) / weight["block"]

    block_weights = np.array([weight["block"] * scale], dtype=np.float32)
    block_value = key["block"] / (key["block"] * block_weights).sum()
    pool_keys = np.concatenate([key["block"], key["rest"]])
    pool_weights = np.array([weight["block"], weight["rest"]], dtype=np.float32)
    pool_value = (pool_keys / (pool_keys * pool_weights).sum())[0]
    share = PoolShare(
        "consumption_shareholding",
        "consumption",
        block_key_sum=frame["block"],
        pool_key_sum=frame["block"] + frame["rest"],
        weight_scale=scale,
    )
    corrected = np.float32(float(block_value[0]) * share.factor())
    gap = abs(float(corrected) / float(pool_value) - 1)
    assert gap > pool_share_rtol(1, 2, key_inputs=1)
    assert gap <= pool_share_rtol(1, 2, key_inputs=12)


def test_the_per_row_bound_covers_a_value_netting_large_terms():
    # A value adding a share term to large household-local terms that
    # nearly cancel, in float32, as a household's net income does: relative
    # to itself its gap is far above the share bound, within the per-row one.
    rng = np.random.default_rng(7)
    rows = 20_000
    rtol = pool_share_rtol(5_000, 20_000, key_inputs=2)
    income = rng.uniform(1e5, 1e6, rows).astype(np.float32)
    taxes = (income * np.float32(0.999)).astype(np.float32)
    pool_share = rng.uniform(0.0, 300.0, rows).astype(np.float32)
    off = rng.uniform(-rtol, rtol, rows) * 0.5
    block_share = (pool_share.astype(np.float64) * (1 + off)).astype(np.float32)

    def net(share):
        return (income - taxes - share).astype(np.float32)

    gap = np.abs(net(block_share).astype(np.float64) - net(pool_share))
    terms = [income, taxes, pool_share]
    magnitude = sum(np.abs(term.astype(np.float64)) for term in terms)
    atol = pool_share_atol(pool_share, magnitude, share_rtol=rtol, additions=2)
    assert np.all(gap <= atol)
    relative = gap / np.maximum(np.abs(net(pool_share)), 1e-30)
    assert relative.max() > rtol
    np.testing.assert_allclose(
        pool_share_atol([2.0], [10.0], share_rtol=1e-6, additions=3),
        [2e-6 + 6 * FLOAT32_UNIT_ROUNDOFF * 10.0],
    )
    with pytest.raises(ValueError):
        pool_share_atol([1.0], [1.0], share_rtol=1e-6, additions=-1)


# ---------------------------------------------------------------------------
# Household batches and the scatter back
# ---------------------------------------------------------------------------


def _clone_frame():
    """Three clones of two households; people listed out of id order."""

    household = pd.DataFrame(
        {
            "household_id": np.array([11, 12, 21, 22, 31, 32], dtype=np.int64),
            "household_clone_index": [2, 0, 1, 2, 0, 1],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.array(
                [322, 111, 211, 121, 221, 311, 112, 321], dtype=np.int64
            ),
            "person_benunit_id": np.array(
                [32, 11, 21, 12, 22, 31, 11, 32], dtype=np.int64
            ),
            "person_household_id": np.array(
                [32, 11, 21, 12, 22, 31, 11, 32], dtype=np.int64
            ),
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": np.array([11, 12, 21, 22, 31, 32], dtype=np.int64)}
    )
    return Frame(
        {"person": person, "benunit": benunit, "household": household},
        EntitySchema(group_entities=("benunit", "household")),
        {
            "household": Weights(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), kind=WeightKind.DESIGN
            )
        },
    )


def test_household_batches_follow_the_column_in_ascending_order():
    frame = _clone_frame()
    batches = household_batches(frame, "household_clone_index")
    assert [batch.key for batch in batches] == [0, 1, 2]
    household_ids = frame.table("household")["household_id"].to_numpy()
    person_households = frame.table("person")["person_household_id"].to_numpy()
    for batch in batches:
        ids = household_ids[batch.households]
        np.testing.assert_array_equal(batch.persons, np.isin(person_households, ids))
    assert sum(int(batch.persons.sum()) for batch in batches) == frame.n("person")
    again = household_batches(frame, "household_clone_index")
    for first, second in zip(batches, again, strict=True):
        assert first.key == second.key
        np.testing.assert_array_equal(first.persons, second.persons)


def test_batches_need_the_column_and_whole_households():
    frame = _clone_frame()
    with pytest.raises(ValueError, match="carries no"):
        household_batches(frame, "missing_column")
    person = frame.table("person").copy()
    person.loc[person["person_id"] == 112, "person_benunit_id"] = 12
    split = Frame(
        {
            "person": person,
            "benunit": frame.table("benunit"),
            "household": frame.table("household"),
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(ValueError, match="must nest in households"):
        assert_groups_nest_in_households(split)
    with pytest.raises(ValueError, match="must nest in households"):
        household_batches(split, "household_clone_index")


@pytest.mark.parametrize("entity", ["person", "benunit", "household"])
def test_scatter_puts_batch_results_in_one_engines_order(entity):
    frame = _clone_frame()
    column = frame.schema.entity_id_column(entity)
    expected = frame.table(entity)[column].to_numpy() * 10 + 1
    parts = []
    for batch in reversed(household_batches(frame, "household_clone_index")):
        ids = frame.select(batch.persons).table(entity)[column].to_numpy()[::-1]
        parts.append((ids, ids * 10 + 1))
    np.testing.assert_array_equal(scatter_by_entity_ids(frame, entity, parts), expected)


def test_scatter_refuses_missing_repeated_or_mistyped_rows():
    frame = _clone_frame()
    ids = frame.table("household")["household_id"].to_numpy()
    values = np.arange(len(ids), dtype=np.float32)
    with pytest.raises(ValueError, match="different number"):
        scatter_by_entity_ids(frame, "household", [(ids[:-1], values[:-1])])
    with pytest.raises(ValueError, match="same household row"):
        repeated = ids.copy()
        repeated[-1] = repeated[0]
        scatter_by_entity_ids(frame, "household", [(repeated, values)])
    with pytest.raises(ValueError, match="does not hold"):
        unknown = ids.copy()
        unknown[0] = 99
        scatter_by_entity_ids(frame, "household", [(unknown, values)])
    with pytest.raises(ValueError, match="different types"):
        scatter_by_entity_ids(
            frame,
            "household",
            [(ids[:3], values[:3]), (ids[3:], values[3:].astype(np.float64))],
        )
