"""A fake policyengine-core engine for the shared engine-block machinery.

A fake simulation stands in for a policyengine-core one: holders that know
which periods hold inputs, formulas run through ``_run_formula``, a
parameter tree read at a period, and a ``clone`` that rebinds the methods
bound to the original, as core's does. Its declaration names a fake engine
module, which :func:`install_fake_engine` puts in ``sys.modules``. Two clones
of three households each make the blocks. The shared tests and the build
tests that hold a block's receipt to the measure step's finalisation use it
(microcosm#931). Everything is synthetic.
"""

from __future__ import annotations

import copy
import re
import sys
import types
from types import SimpleNamespace

import numpy as np

from microcosm.frame.adapters.engine_population import (
    InertRule,
    ParameterTest,
    PopulationDeclaration,
    PositionWindow,
    all_of,
    pool_shares,
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


def local_formula(simulation, period):
    return simulation.calculate("key_a", period) * 2.0


def fake_system(*, budget=0.0, vat_rate=0.2, baseline_vat_rate=0.2):
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
        "local": FakeVariable("local", formula=local_formula),
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


def install_fake_engine(monkeypatch):
    """Install the fake engine's own module, as the declaration names it: its
    list, its window and its reform factories. Returns the module."""

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


def fake_declaration(**changes):
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


def fake_inputs(rows, *, weight_scale=1.0, store=("months", "attendance")):
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


def frame_sum(rows):
    return float(np.sum((KEY_A[rows] + KEY_B[rows]) * WEIGHT[rows]))


def block_shares(*, scaled):
    pool_mass = float(WEIGHT.sum())
    scales = (
        {block: pool_mass / float(WEIGHT[rows].sum()) for block, rows in BLOCKS.items()}
        if scaled
        else None
    )
    shares = pool_shares(
        {block: {"share": frame_sum(rows)} for block, rows in BLOCKS.items()},
        scales,
        key_variables={"share": "key"},
    )
    return shares, scales


def fake_block(block, *, scaled=True, store=("months", "attendance"), system=None):
    rows = BLOCKS[block]
    scale = 1.0
    if scaled:
        scale = float(WEIGHT.sum()) / float(WEIGHT[rows].sum())
    return FakeSimulation(
        system or fake_system(),
        fake_inputs(rows, weight_scale=scale, store=store),
        rows=len(rows),
    )


def fake_pool():
    return FakeSimulation(fake_system(), fake_inputs(np.arange(N)))


def numbers_are_years(texts) -> bool:
    """Whether every number written in ``texts`` is a year, as a refusal's
    must be: it names formulas, variables and periods, never a value or a
    count."""

    numbers = [
        number for text in texts for number in re.findall(r"\d+(?:\.\d+)?", text)
    ]
    return all(re.fullmatch(r"(19|20)\d\d", number) for number in numbers)
