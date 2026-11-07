"""policyengine-uk's whole-population formulas as the UK declares them (microcosm#931).

``UK_ENGINE_POPULATION`` is held against the installed engine: its own list,
the allocation keys' inputs, the pinned sources and their coverage, the
predicates at current law and moved off it, the constructor's structural
reforms, the window, the readers the inert rule rests on, and the drift
lists of axis-free reductions and whole-array branches outside the list.
The shared machinery then runs on small synthetic datasets: a pool of two
clones that differ, corrected clone by clone, against one engine over the
pool. Every household here is made up.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import re
import subprocess
import sys
import textwrap
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame.adapters.engine_population import (
    MEASURE_PASS,
    FormulaRunRecorder,
    PopulationDeclarationError,
    check_key_sums,
    check_uniform_growth,
    inject_pool_shares,
    judge,
    pool_share_atol,
    pool_share_rtol,
    pool_shares,
    source_sha256,
    validate_declaration,
)
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
    uk_engine_population,
)

D, P = 2024, 2025
CLONES, HOUSEHOLDS = 2, 30
COICOP = UK_ENGINE_POPULATION.key_adds["consumption"]
REGIONS = ("LONDON", "NORTH_WEST", "SCOTLAND", "WALES", "SOUTH_EAST")


@pytest.fixture(scope="module")
def system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def _household_tax_variables():
    from policyengine_uk.variables.gov.hmrc.household_tax import (
        HOUSEHOLD_TAX_VARIABLES,
    )

    return HOUSEHOLD_TAX_VARIABLES


def _package_root() -> Path:
    import policyengine_uk

    return Path(policyengine_uk.__file__).parent


# ---------------------------------------------------------------------------
# The declaration against the installed engine
# ---------------------------------------------------------------------------


def test_the_modules_load_without_importing_an_engine():
    code = (
        "import sys\n"
        "import microcosm.frame.adapters.policyengine_uk_population\n"
        "loaded = [m for m in sys.modules if m.startswith('policyengine')]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_declaration_classifies_exactly_the_engines_list():
    from policyengine_uk.utils.household_subset import WHOLE_POPULATION_VARIABLES

    assert UK_ENGINE_POPULATION.classified() == WHOLE_POPULATION_VARIABLES
    assert len(WHOLE_POPULATION_VARIABLES) == 19


def test_the_declaration_validates_against_the_installed_engine(system):
    assert uk_engine_population(system) is UK_ENGINE_POPULATION


def test_the_allocation_keys_are_the_engines_adds_of_household_inputs(system):
    for formula, key in UK_ENGINE_POPULATION.corrected.items():
        variable = system.variables[formula]
        assert not variable.is_input_variable(), formula
        assert variable.entity.key == "household", formula
        key_variable = system.variables[key]
        assert tuple(key_variable.adds) == UK_ENGINE_POPULATION.key_adds[key]
        for name in key_variable.adds:
            leaf = system.variables[name]
            assert leaf.is_input_variable() and leaf.entity.key == "household", name
    assert UK_ENGINE_POPULATION.key_adds["corporate_sector_wealth"] == (
        "corporate_wealth",
        "private_pension_wealth",
    )
    assert len(COICOP) == 12


@pytest.mark.parametrize("reference", sorted(UK_ENGINE_POPULATION.digests))
def test_every_pinned_source_is_the_reviewed_one(reference):
    assert source_sha256(reference) == UK_ENGINE_POPULATION.digests[reference]


def _source_file(system, name):
    """The file a variable's class is defined in, as a digest names it."""

    path = Path(inspect.getsourcefile(type(system.variables[name]))).resolve()
    return f"policyengine_uk/{path.relative_to(_package_root().resolve()).as_posix()}"


def test_the_pins_cover_every_source_a_class_rests_on(system):
    # Every corrected, zero-at-policy and inert formula, the inert rule's sink
    # and levels, the constructor's structural reforms (adjust_budgets would
    # rescale spending by a population total) and the Universal Credit
    # modifier the window rests on; nothing else.
    declaration = UK_ENGINE_POPULATION
    names = {*declaration.corrected, *declaration.zero_at_policy}
    for name, rule in declaration.inert.items():
        names |= {name, rule.sink, *rule.levels}
    required = {_source_file(system, name) for name in names} | {
        "policyengine_uk/reforms/policyengine/adjust_budgets.py",
        "policyengine_uk/reforms/reforms.py",
        "policyengine_uk.scenarios.uc_reform:add_universal_credit_reform",
    }
    assert set(declaration.digests) == required


def test_a_source_that_moved_fails_validation(system):
    reference = "policyengine_uk/variables/household/wealth/shareholding.py"
    moved = dataclasses.replace(
        UK_ENGINE_POPULATION,
        digests={**UK_ENGINE_POPULATION.digests, reference: "0" * 64},
    )
    with pytest.raises(PopulationDeclarationError, match="has changed"):
        validate_declaration(moved, system)


@pytest.mark.parametrize("year", [D, P])
def test_every_predicate_holds_at_current_law(system, year):
    parameters = system.parameters(year)
    for name, predicate in UK_ENGINE_POPULATION.run_predicates().items():
        assert predicate.holds(parameters), (name, year)


@pytest.mark.parametrize("year", [D, P])
def test_the_inert_predicate_gives_the_engines_rate_equality(system, year):
    from policyengine_uk.variables.gov.hmrc.private_school_vat import (
        private_school_vat_rate,
    )

    parameters = system.parameters(year)
    rule = UK_ENGINE_POPULATION.inert["attends_private_school"]
    assert rule.predicate.holds(parameters)
    assert private_school_vat_rate(parameters.gov) == private_school_vat_rate(
        parameters.baseline.gov
    )
    # The rate reads exactly the three parameters the predicate compares.
    source = inspect.getsource(private_school_vat_rate)
    read = set(re.findall(r"(?:vat|gov)\.([a-z_.]+)", source))
    assert read == {
        "hmrc.vat",
        "private_school_fees_exempt",
        "standard_rate",
        "contrib.labour.private_school_vat",
    }


_LSR = "gov.simulation.labour_supply_responses"
_EMPLOYER_NI = "gov.contrib.policyengine.employer_ni"
_BUDGET = "gov.contrib.policyengine.budget"

#: For each formula with a predicate, moves of its guarded parameters off
#: current law, each of which makes the predicate false; together they move
#: every parameter it guards.
PREDICATE_MOVES = {
    "employment_income_behavioral_response": (
        {f"{_LSR}.income_elasticity": 0.1},
        {f"{_LSR}.substitution_elasticity": 0.1},
    ),
    "employer_ni_response_capital_incidence": (
        {
            f"{_EMPLOYER_NI}.capital_incidence": 0.5,
            f"{_EMPLOYER_NI}.employee_incidence": 0.5,
        },
    ),
    "employer_ni_response_consumer_incidence": (
        {
            f"{_EMPLOYER_NI}.consumer_incidence": 0.5,
            f"{_EMPLOYER_NI}.employee_incidence": 0.5,
        },
    ),
    "carbon_tax": ({"gov.contrib.ubi_center.carbon_tax.rate": 50.0},),
    "consumer_incident_tax_revenue_change": (
        {f"{_BUDGET}.consumer_incident_tax_change": 1.0},
    ),
    "education_budget_change": ({f"{_BUDGET}.education": 1.0},),
    "nhs_budget_change": ({f"{_BUDGET}.nhs": 1.0},),
    "other_public_spending_budget_change": ({f"{_BUDGET}.other_public_spending": 1.0},),
    # Each moves the private school VAT rate off its baseline.
    "attends_private_school": (
        {"gov.hmrc.vat.private_school_fees_exempt": True},
        {"gov.hmrc.vat.standard_rate": 0.25},
        {"gov.contrib.labour.private_school_vat": 0.3},
    ),
}
#: What else a formula needs, beyond its predicate failing, to have
#: something to respond to.
RESPONSE_NEEDS = {
    # A change in net income and marginal rates.
    "employment_income_behavioral_response": {
        "gov.hmrc.income_tax.rates.uk[0].rate": 0.25
    },
    # A change in employers' costs.
    "employer_ni_response_capital_incidence": {
        "gov.hmrc.national_insurance.class_1.rates.employer": 0.2
    },
    "employer_ni_response_consumer_incidence": {
        "gov.hmrc.national_insurance.class_1.rates.employer": 0.2
    },
}


def test_every_guarded_parameter_is_moved():
    declaration = UK_ENGINE_POPULATION
    predicates = declaration.run_predicates()
    assert set(PREDICATE_MOVES) == {*declaration.zero_at_policy, *declaration.inert}
    for name, moves in PREDICATE_MOVES.items():
        guarded = {
            path
            for path in predicates[name].paths()
            if not path.startswith("baseline.")
        }
        assert {path for move in moves for path in move} == guarded, name


def _control_tables():
    """One clone of the synthetic households, its children at private
    school, so the private school VAT change has pupils to charge."""

    person, benunit, household = _subset(_tables(), 0)
    person["attends_private_school"] = person["age"] < 18
    return person, benunit, household


@pytest.mark.parametrize(
    "name, move",
    [(name, move) for name, moves in PREDICATE_MOVES.items() for move in moves],
    ids=lambda value: value if isinstance(value, str) else "+".join(sorted(value)),
)
def test_a_moved_parameter_breaks_its_predicate_and_frees_its_formula(name, move):
    # The negative control for each predicate: off current law it fails, and
    # the formula it guards (an inert formula's sink) is no longer zero.
    rule = UK_ENGINE_POPULATION.inert.get(name)
    changes = {**move, **RESPONSE_NEEDS.get(name, {})}
    simulation = _simulation(
        *_control_tables(),
        reform={
            path: {"2024-01-01.2100-12-31": value} for path, value in changes.items()
        },
    )
    predicate = UK_ENGINE_POPULATION.run_predicates()[name]
    assert not predicate.holds(simulation.tax_benefit_system.parameters(P))
    target = name if rule is None else rule.sink
    assert np.any(_values(simulation, target, P) != 0), target


@pytest.mark.parametrize("year", [D, P])
def test_no_structural_reform_is_switched_on(system, year):
    from policyengine_uk.reforms import reforms

    for reference in UK_ENGINE_POPULATION.constructor_factories:
        module, _, name = reference.partition(":")
        factory = getattr(__import__(module, fromlist=[name]), name)
        assert factory(system.parameters, year) is None, reference
    tree = ast.parse(
        inspect.getsource(reforms.create_structural_reforms_from_parameters)
    )
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith(("create_", "disable_", "adjust_"))
    }
    declared = {
        reference.rpartition(":")[2]
        for reference in UK_ENGINE_POPULATION.constructor_factories
    }
    assert called == declared


def test_the_window_starts_at_the_rebalancing_years():
    from policyengine_uk.scenarios.uc_reform import (
        FIRST_REBALANCING_YEAR,
        REBALANCING_YEARS,
    )

    assert REBALANCING_YEARS.start == FIRST_REBALANCING_YEAR == 2026
    assert UK_ENGINE_POPULATION.window.first_year == REBALANCING_YEARS.start
    assert UK_SPINE_ENGINE_BATCH_COLUMN == "household_clone_index"


def _readers(names: set[str]) -> dict[str, set[str]]:
    """Where each name appears as a string constant or is called, by the
    module and outermost definition (outside the engine's own tests)."""

    root = _package_root()
    found: dict[str, set[str]] = {name: set() for name in names}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for top in tree.body:
            scope = getattr(top, "name", None)
            if scope is None and isinstance(top, ast.Assign):
                targets = [t.id for t in top.targets if isinstance(t, ast.Name)]
                scope = targets[0] if targets else None
            where = f"{relative}:{scope or '<module>'}"
            for node in ast.walk(top):
                value = None
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    value = node.func.id
                if value in found:
                    found[value].add(where)
    return found


def test_attendance_reaches_household_taxes_only_through_the_vat_change():
    readers = _readers(
        {
            "attends_private_school",
            "private_school_vat_liability",
            "private_school_vat",
            "baseline_private_school_vat",
            "private_school_vat_change",
        }
    )
    assert readers["attends_private_school"] == {
        "data/filter_dataset.py:filter_dataset",
        "utils/household_subset.py:WHOLE_POPULATION_VARIABLES",
        # A calibration script, run by hand; no formula imports it.
        "utils/solve_private_school_attendance_factor.py:<module>",
        "variables/gov/hmrc/private_school_vat.py:private_school_vat_liability",
    }
    assert readers["private_school_vat_liability"] == {
        "variables/gov/hmrc/baseline_private_school_vat.py:baseline_private_school_vat",
        "variables/gov/hmrc/private_school_vat.py:private_school_vat",
    }
    assert readers["private_school_vat"] == {
        "variables/gov/gov_tax.py:GOV_TAX_VARIABLES",
        "variables/gov/hmrc/private_school_vat_change.py:private_school_vat_change",
    }
    assert readers["baseline_private_school_vat"] == {
        "variables/gov/hmrc/private_school_vat_change.py:private_school_vat_change",
    }
    assert readers["private_school_vat_change"] == {
        "variables/gov/hmrc/household_tax.py:HOUSEHOLD_TAX_VARIABLES",
    }


#: Formulas outside the engine's list with an axis-free reduction (a sum,
#: mean, extreme or any/all over the whole array) in their own source,
#: reviewed as household-local: membership tests and shortcuts, a loop bound
#: and parameter-share cumulative sums. A new hit must be reviewed.
AXIS_FREE_REDUCTIONS = {
    "brma": [".any()"],
    "capital_gains_tax": ["np.any(", "np.max("],
    "ebr_council_tax_rebate": ["np.any("],
    "is_cared_for_by_carer_benefit_recipient": ["np.max("],
    "legacy_means_test_uncounted_income_tax_without_earnings": [".any()"],
    "pension_credit_income_tax_without_earnings": [".any()"],
    "pension_credit_income_tax_without_employment_income": [".any()"],
    "uc_award_carries_deductions": [".any()"],
    "uc_deduction_combination": [".sum()", "np.cumsum("],
    "uc_latent_deduction_rate": [".sum()", "np.cumsum("],
    "uses_property_allowance": [".all()", ".any()"],
    "uses_trading_allowance": [".any()"],
}
_REDUCTION = re.compile(
    r"\.(sum|mean|max|min|any|all|median|std|var|cumsum|argsort|sort_values|rank"
    r"|quantile|percentile)\(\s*\)"
    r"|np\.(sum|mean|max|min|any|all|median|quantile|percentile|sort|argsort"
    r"|cumsum|unique)\("
    r"|numpy\.(sum|mean|max|min|any|all|median|quantile|percentile|sort|argsort"
    r"|cumsum|unique)\("
)


def test_axis_free_reductions_outside_the_list_are_the_reviewed_ones(system):
    listed = UK_ENGINE_POPULATION.classified()
    hits: dict[str, set[str]] = {}
    for name, variable in system.variables.items():
        if name in listed:
            continue
        for formula in variable.formulas.values():
            found = {
                m.group(0) for m in _REDUCTION.finditer(inspect.getsource(formula))
            }
            if found:
                hits.setdefault(name, set()).update(found)
    assert {name: sorted(found) for name, found in hits.items()} == AXIS_FREE_REDUCTIONS


#: What marks an if, while or conditional-expression test (or a
#: comprehension filter) as reading the whole array rather than a row: a
#: reduction, any/all, or a length or size.
_WHOLE_ARRAY_TEST = (
    ".any()",
    ".all()",
    "np.any(",
    "np.all(",
    "numpy.any(",
    "numpy.all(",
    "any(",
    "all(",
    "len(",
    ".sum()",
    "np.sum(",
    ".max()",
    ".min()",
    "np.max(",
    "np.min(",
    ".mean()",
    "np.mean(",
    ".size",
    ".shape",
    "np.unique(",
    "np.median(",
    ".count",
    ".nunique(",
)


def _engine_function(function, root):
    """``path:qualname`` of a function defined in the engine (not its tests)."""

    code = getattr(function, "__code__", None)
    if code is None:
        return None
    try:
        relative = Path(code.co_filename).resolve().relative_to(root)
    except ValueError:
        return None
    if relative.parts[0] == "tests":
        return None
    return f"{relative.as_posix()}:{function.__qualname__}"


def _code_names(code):
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return names


def _unwrapped(value):
    try:
        return inspect.unwrap(value)
    except ValueError:
        return value


def _called(function, root):
    """The engine functions ``function`` names: globals, and attributes of
    the modules it names, unwrapped from their decorators."""

    namespace = getattr(function, "__globals__", {})
    names = _code_names(function.__code__)
    for name in sorted(names):
        value = namespace.get(name)
        candidates = []
        if isinstance(value, types.ModuleType):
            for other in sorted(names):
                attribute = getattr(value, other, None)
                if callable(attribute):
                    candidates.append(_unwrapped(attribute))
        elif callable(value):
            candidates.append(_unwrapped(value))
        for candidate in candidates:
            if isinstance(candidate, types.FunctionType):
                if _engine_function(candidate, root) is not None:
                    yield candidate


def _whole_array_tests(function):
    """Each such test in ``function``'s source, as written, on one line."""

    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []
    found = set()
    for node in ast.walk(tree):
        conditions = []
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            conditions.append(node.test)
        elif isinstance(node, ast.comprehension):
            conditions.extend(node.ifs)
        for condition in conditions:
            text = " ".join(ast.get_source_segment(source, condition).split())
            if any(mark in text for mark in _WHOLE_ARRAY_TEST):
                found.add(text)
    return sorted(found)


def whole_array_branches(system):
    """Every formula outside the engine's list, and every engine function
    those formulas reach, with its whole-array tests."""

    root = _package_root().resolve()
    listed = UK_ENGINE_POPULATION.classified()
    reached = {}
    for name, variable in system.variables.items():
        if name in listed:
            continue
        stack = list(variable.formulas.values())
        while stack:
            function = stack.pop()
            label = _engine_function(function, root)
            if label is None or label in reached:
                continue
            reached[label] = function
            stack.extend(_called(function, root))
    branches = {}
    for label, function in sorted(reached.items()):
        tests = _whole_array_tests(function)
        if tests:
            branches[label] = tests
    return branches


#: Formulas outside the engine's list, and the engine functions they reach,
#: whose control flow turns on the whole array. Each was reviewed as
#: household-local, so a block takes each household's branch as the pool
#: does (microcosm#931): a shortcut whose branches give every row the same
#: value, a test of parameters or of which inputs the data supplies, a
#: check that raises, or policyengine-uk's own household-subset machinery,
#: which falls back to the whole simulation. A new or changed test must be
#: reviewed the same way.
WHOLE_ARRAY_BRANCHES = {
    # A configuration check that raises.
    "data/isa_income.py:validate_contract": [
        'not all( isinstance(contract[key], str) for key in ("representation", "projection") )',
    ],
    # Where every couple is the claimant and partner, np.where gives each the
    # same award.
    "utils/benefit_unit.py:award_of_couple": [
        "np.all(same_couple)",
    ],
    # Raises where the baseline holds other people.
    "utils/capital_gains.py:measure_capital_gains_mtrs": [
        "baseline_count != person.count",
    ],
    # Without gains nobody's gains respond, as the factors would give.
    "utils/capital_gains.py:realisation_factors": [
        'not np.any(person("capital_gains_before_response", period) != 0)',
    ],
    # Without participation the working terms are false; targeted eligibility
    # counts only where a row's own terms overlap.
    "utils/childcare.py:childcare_term_value": [
        'component in ("targeted", "extended") or ( component == "universal" and any(np.any(u & t) for u, t in zip(universal_terms, targeted_terms)) )',
        "np.any(participating)",
    ],
    # Checks that raise.
    "utils/dates.py:_check_given_dates_of_birth": [
        "invalid.any()",
        "mismatch.any()",
    ],
    # Checks that raise; the instant is chosen row by row.
    "utils/dates.py:birth_instant": [
        "(given < 0).any()",
        "use_given.any()",
    ],
    # policyengine-uk's household subsets, which fall back to the whole
    # simulation.
    "utils/household_subset.py:_positions": [
        "(kept[members] != kept_people).any()",
    ],
    "utils/household_subset.py:_restrict": [
        "np.ndim(values) != 1 or len(values) != population.count",
    ],
    "utils/household_subset.py:_restricted_branch": [
        "any( baseline.populations[key].count != population.count or not np.array_equal(baseline.populations[key].ids, population.ids) for key, population in simulation.populations.items() )",
        "any( holder._disk_storage is not None for population in simulation.populations.values() for holder in population._holders.values() )",
    ],
    "utils/household_subset.py:household_subset": [
        "not households.all()",
    ],
    # A parameter scale.
    "utils/income_tax.py:first_extended_scottish_threshold": [
        "len(scale.thresholds) == 3",
    ],
    # Rows without net pay and a trading loss take net income either way (net
    # income is floored at zero).
    "utils/pensions.py:annual_allowance_net_income": [
        'not np.any((relievable_net_pay > 0) & (person("trading_loss", period) > 0))',
    ],
    # Parameter trees.
    "utils/rule_comparison.py:same_parameters": [
        'len(node.brackets) != len(other_node.brackets) or node.metadata.get( "type" ) != other_node.metadata.get("type")',
        "len(values) != len(other_values)",
    ],
    # Which inputs the data supplies.
    "variables/gov/dft/person_bus_fare_spending.py:person_bus_fare_spending.formula": [
        'all( supplied_input(person, name, period) is None for name in ("bus_in_london_trips", "other_local_bus_trips") )',
    ],
    # Where no household is over its total, np.where keeps every balance.
    "variables/gov/dwp/_person_capital_pool.py:bounded_holdings": [
        "not over.any()",
    ],
    # Parts of the year from parameters, and whether an input or reform
    # supplies the award.
    "variables/gov/dwp/carers_allowance.py:carers_allowance.formula": [
        "len(parts) == 1 or not calculated_by_model( person, period, carers_allowance_pre_overlap )",
    ],
    # Parts of the year from parameters.
    "variables/gov/dwp/carers_allowance_pre_overlap.py:carers_allowance_pre_overlap.formula": [
        "len(parts) == 1",
    ],
    # Segments of the year from parameters, and whether an input overrides the
    # income.
    "variables/gov/dwp/housing_benefit/entitlement/housing_benefit_entitlement.py:housing_benefit_entitlement.formula": [
        'len(segments) == 1 or hb_uses_annual_override( benunit, period, "housing_benefit_applicable_income", housing_benefit_applicable_income.formula, )',
    ],
    # Zeroing earnings nobody has changes nobody's tax.
    "variables/gov/dwp/legacy_means_test_uncounted_income_tax_without_earnings.py:legacy_means_test_uncounted_income_tax_without_earnings.formula": [
        "not any((person(source, period) != 0).any() for source in sources)",
    ],
    # A parameter list.
    "variables/gov/dwp/pension_credit/pension_credit_earnings_disregard.py:pension_credit_earnings_disregard.formula": [
        "len(unit_disability_benefits) > 0",
    ],
    # Zeroing earnings nobody has changes nobody's tax.
    "variables/gov/dwp/pension_credit/pension_credit_earnings_income_tax.py:pension_credit_income_tax_without_earnings.formula": [
        'not (person("self_employment_income", period) != 0).any()',
    ],
    "variables/gov/dwp/pension_credit/pension_credit_earnings_income_tax.py:pension_credit_income_tax_without_employment_income.formula": [
        "not any((person(source, period) != 0).any() for source in sources)",
    ],
    # Without a changed baseline or an award, every unit carries its
    # deductions, as the comparison gives a unit without an award.
    "variables/gov/dwp/universal_credit/deductions/uc_award_carries_deductions.py:uc_award_carries_deductions.formula": [
        'not getattr(simulation, "changed_from_baseline", False) or baseline is None or not award.any() or baseline.populations["person"].count != person.count',
    ],
    # A parameter list.
    "variables/gov/dwp/universal_credit/forgoes_universal_credit_for_tax_free_childcare.py:forgoes_universal_credit_for_tax_free_childcare.formula": [
        "not any( program in UNIVERSAL_CREDIT_AWARDS for program in p.disqualifying_benefits )",
    ],
    # With no exemption left anywhere, a further pass takes nothing.
    "variables/gov/hmrc/capital_gains_tax/capital_gains_tax.py:capital_gains_tax.formula": [
        "not np.any(aea_left > 0)",
    ],
    # Each household decides its own route, and households do not move each
    # other's tax; the tests skip work nobody needs.
    "variables/gov/hmrc/income_tax/allowances/uses_property_allowance.py:uses_property_allowance.formula": [
        "(deciding & ~unsettled).any()",
        "(unsettled & (most_charge_not_using > 0)).any()",
        "(unsettled & (most_charge_using > 0)).any()",
        "not choosing.any()",
        "not done.all()",
    ],
    # Only a claimant can take the allowance.
    "variables/gov/hmrc/income_tax/allowances/uses_trading_allowance.py:uses_trading_allowance.formula": [
        "not claimant.any()",
    ],
    # Parameter thresholds.
    "variables/gov/hmrc/pensions/private_pension_contributions_tax.py:_top_slice_charge": [
        "i + 1 < len(thresholds)",
    ],
    # A parameter list.
    "variables/gov/ofgem/warm_home_discount.py:warm_home_discount.formula": [
        "len(benefits) == 0",
    ],
    # A household that has not moved keeps its BRMA either way.
    "variables/household/BRMA.py:brma.formula": [
        "not moved.any()",
    ],
}


def test_whole_array_branches_outside_the_list_are_the_reviewed_ones(system):
    assert whole_array_branches(system) == WHOLE_ARRAY_BRANCHES


# ---------------------------------------------------------------------------
# The machinery on the engine
# ---------------------------------------------------------------------------


def _tables(*, differ: bool = True):
    """Two clones of the same households; with ``differ`` each clone draws
    its own keys, pay and weights. Household ids rise with the clone."""

    persons, benunits, households = [], [], []
    for clone in range(CLONES):
        for h in range(HOUSEHOLDS):
            household_id = (clone + 1) * 1000 + h
            draw = np.random.default_rng(17 * h + (1009 * clone if differ else 0))
            shared = np.random.default_rng(5000 + h)
            benunits.append({"benunit_id": household_id * 10})
            weight = 40_000.0 * (1 + h % 5) / CLONES
            if differ:
                weight *= float(draw.uniform(0.7, 1.3))
            row = {
                "household_id": household_id,
                "household_weight": weight,
                "region": REGIONS[h % len(REGIONS)],
                "council_tax": 1_400.0 + 90 * (h % 5),
                "rent": 9_000.0 if h % 2 else 0.0,
                "tenure_type": "RENT_PRIVATELY" if h % 2 else "OWNED_OUTRIGHT",
                "corporate_wealth": float(draw.lognormal(10, 1.6)) if h % 2 else 0.0,
                "private_pension_wealth": (
                    float(draw.lognormal(11, 1.1)) if h % 3 else 0.0
                ),
            }
            for column in COICOP:
                row[column] = float(draw.uniform(300, 6_000))
            households.append(row)
            for p in range(1 + h % 4):
                age = (41, 38, 12, 9)[p]
                pay = np.random.default_rng(31 * h + p + (997 * clone if differ else 0))
                persons.append(
                    {
                        "person_id": household_id * 10 + p,
                        "person_benunit_id": household_id * 10,
                        "person_household_id": household_id,
                        "age": age,
                        "gender": "MALE" if (h + p) % 2 else "FEMALE",
                        "employment_income": (
                            float(pay.uniform(0, 60_000))
                            if age >= 18 and h % 3
                            else 0.0
                        ),
                        "months_since_last_birthday": float(shared.integers(0, 12)),
                        "attends_private_school_random_draw": float(shared.uniform()),
                        "attends_private_school": False,
                    }
                )
    return pd.DataFrame(persons), pd.DataFrame(benunits), pd.DataFrame(households)


def _subset(tables, clone, *, weight_scale=1.0):
    person, benunit, household = tables
    ids = household["household_id"].to_numpy()
    kept = household[(ids // 1000 - 1) == clone].copy()
    kept["household_weight"] = kept["household_weight"] * weight_scale
    people = person[person["person_household_id"].isin(set(kept["household_id"]))]
    units = benunit[benunit["benunit_id"].isin(set(people["person_benunit_id"]))]
    return people.copy(), units.copy(), kept


def _simulation(person, benunit, household, reform=None):
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Microsimulation(
            dataset=UKSingleYearDataset(
                person=person.copy(),
                benunit=benunit.copy(),
                household=household.copy(),
                fiscal_year=D,
            ),
            reform=reform,
        )


def _values(simulation, name, period, map_to=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = simulation.calculate(name, period, map_to=map_to)
    return np.asarray(raw)


def _key_sum(household, formula):
    key = UK_ENGINE_POPULATION.corrected[formula]
    values = np.zeros(len(household))
    for column in UK_ENGINE_POPULATION.key_adds[key]:
        values += household[column].to_numpy(dtype=np.float64)
    return float(np.sum(values * household["household_weight"].to_numpy(np.float64)))


#: Each share result and the allocation key it divides by: business rates
#: and corporate SDLT are shareholding times a receipt.
SHARE_RESULTS = {
    "shareholding": "corporate_sector_wealth",
    "corporate_land_value": "corporate_sector_wealth",
    "consumption_shareholding": "consumption",
    "business_rates": "corporate_sector_wealth",
    "corporate_sdlt": "corporate_sector_wealth",
}
#: household_tax's share-dependent terms (the rest are zero at current law
#: or household-local).
SHARE_DEPENDENT_TAXES = ("business_rates", "corporate_sdlt")
HOUSEHOLD_LOCAL = ("income_tax", "national_insurance", "household_market_income")


def _share_rtol(name, block_rows, pool_rows, years):
    key = SHARE_RESULTS[name]
    return pool_share_rtol(
        block_rows,
        pool_rows,
        key_inputs=len(UK_ENGINE_POPULATION.key_adds[key]),
        projection_years=years,
    )


@pytest.fixture(scope="module")
def pool_engine():
    tables = _tables()
    return tables, _simulation(*tables)


def test_zero_formulas_are_zero_on_a_synthetic_dataset(pool_engine):
    _tables_, simulation = pool_engine
    for name in UK_ENGINE_POPULATION.zero_at_policy:
        for period in (D, P):
            assert np.all(_values(simulation, name, period) == 0), (name, period)


def test_corrected_blocks_reproduce_one_engine_over_the_pool(pool_engine, system):
    tables, pool = pool_engine
    household = tables[2]
    household_taxes = _household_tax_variables()
    declaration = uk_engine_population(system)
    clone_of = household["household_id"].to_numpy() // 1000 - 1
    pool_mass = float(household["household_weight"].sum())
    blocks = {clone: household[clone_of == clone] for clone in range(CLONES)}
    scales = {
        clone: pool_mass / float(rows["household_weight"].sum())
        for clone, rows in blocks.items()
    }
    shares = pool_shares(
        {
            clone: {
                formula: _key_sum(rows, formula) for formula in declaration.corrected
            }
            for clone, rows in blocks.items()
        },
        scales,
        key_variables=declaration.corrected,
    )
    pool_ids = _values(pool, "household_id", P)
    pool_people = _values(pool, "person_id", P)
    for clone, rows in blocks.items():
        block = _simulation(*_subset(tables, clone, weight_scale=scales[clone]))
        check_key_sums(
            block,
            shares[clone],
            D,
            weight_variable=declaration.weight_variable,
        )
        factors = check_uniform_growth(
            block,
            (*sorted(set(declaration.corrected.values())), declaration.weight_variable),
            D,
            P,
        )
        assert all(factor is not None for factor in factors.values())
        here = np.searchsorted(pool_ids, _values(block, "household_id", P))
        for formula in declaration.corrected:
            # Uncorrected, the block's own shares miss the pool's.
            assert not np.allclose(
                _values(block, formula, P),
                _values(pool, formula, P)[here],
                rtol=1e-3,
                atol=0,
            ), formula
        injected = inject_pool_shares(
            block, shares[clone], (D, P), weight_variable=declaration.weight_variable
        )
        recorder = FormulaRunRecorder.for_declaration(declaration)
        recorder.arm(block)
        assert recorder.is_armed(block) and recorder.is_armed(block.baseline)

        requested = [(name, P) for name in (*SHARE_RESULTS, "household_tax")]
        for name, period in requested:
            _values(block, name, period)
        built = [block]
        if clone == 0:
            # Another simulation the block builds from the same data (as the
            # resolver's annual-income sibling is) takes the same arrays.
            sibling = _simulation(*_subset(tables, clone, weight_scale=scales[clone]))
            injected.apply(sibling)
            recorder.arm(sibling)
            for name in SHARE_RESULTS:
                np.testing.assert_array_equal(
                    _values(sibling, name, P), _values(block, name, P), err_msg=name
                )
            assert injected.matches(block) and injected.matches(sibling)
            built.append(sibling)
        for period, years in ((D, 0), (P, 1)):
            for name in SHARE_RESULTS:
                np.testing.assert_allclose(
                    _values(block, name, period),
                    _values(pool, name, period)[here],
                    rtol=_share_rtol(name, len(rows), len(household), years),
                    atol=0,
                    err_msg=f"{name} at {period}",
                )
            # household_tax adds the share-dependent taxes to household-local
            # ones: bounded per row, by the share terms and its own additions.
            terms = {
                name: _values(block, name, period, map_to="household")
                for name in household_taxes
            }
            atol = pool_share_atol(
                sum(np.abs(terms[name]) for name in SHARE_DEPENDENT_TAXES),
                sum(np.abs(values.astype(np.float64)) for values in terms.values()),
                share_rtol=max(
                    _share_rtol(name, len(rows), len(household), years)
                    for name in SHARE_DEPENDENT_TAXES
                ),
                additions=len(household_taxes),
            )
            gap = np.abs(
                _values(block, "household_tax", period).astype(np.float64)
                - _values(pool, "household_tax", period)[here]
            )
            assert np.all(gap <= atol), f"household_tax at {period}"
            assert np.any(terms["business_rates"] != 0)
        people = np.searchsorted(pool_people, _values(block, "person_id", P))
        for name in HOUSEHOLD_LOCAL:
            values = _values(block, name, P)
            reference = _values(pool, name, P)
            positions = people if len(values) == len(people) else here
            np.testing.assert_array_equal(values, reference[positions], err_msg=name)

        verdict = judge(declaration, MEASURE_PASS, recorder, built, (D, P), requested)
        assert verdict.clean, verdict.describe()
        assert verdict.classes["corrected"] == tuple(sorted(declaration.corrected))
        assert verdict.classes["stored"] == (
            "attends_private_school",
            "months_since_last_birthday",
        )
        assert "carbon_tax" in verdict.classes["zero_at_policy"]

        branch = block.get_branch("after_arming")
        assert branch._run_formula.__self__ is branch
        # Nothing the block computed: the branch runs it and records.
        _values(branch, "household_net_income", D)
        assert recorder.token(branch) is not None

        # A household-local read in a rebalancing year still breaks the window.
        _values(block, "income_tax", 2026)
        assert "income_tax" in recorder.periods_run()["2026"]
        late = judge(declaration, MEASURE_PASS, recorder, built, (D, P), requested)
        assert not late.clean
        assert {(v.kind, v.period) for v in late.violations} == {("window_run", "2026")}
        recorder.disarm()
        assert all("_run_formula" not in vars(simulation) for simulation in built)
        del block, branch, built, recorder, injected


def test_a_share_read_at_a_period_never_injected_is_caught(pool_engine, system):
    tables, _pool = pool_engine
    declaration = uk_engine_population(system)
    block = _simulation(*_subset(tables, 1))
    household = _subset(tables, 1)[2]
    shares = pool_shares(
        {
            0: {
                formula: _key_sum(household, formula)
                for formula in declaration.corrected
            },
            1: {
                formula: _key_sum(household, formula)
                for formula in declaration.corrected
            },
        },
        key_variables=declaration.corrected,
    )
    inject_pool_shares(block, shares[1], (P,), weight_variable="household_weight")
    recorder = FormulaRunRecorder.for_declaration(declaration)
    recorder.arm(block)
    _values(block, "business_rates", D)
    verdict = judge(declaration, MEASURE_PASS, recorder, [block], (D, P))
    kinds = {(v.kind, v.name, v.period) for v in verdict.violations}
    assert ("corrected_derived", "shareholding", str(D)) in kinds
    assert ("corrected_not_input", "shareholding", str(D)) in kinds
    recorder.disarm()


def test_a_twelve_input_key_that_rounds_apart_stays_inside_the_bound():
    # Two made-up one-person households. The engine adds each household's
    # twelve COICOP inputs in float32: the block's eleven small ones each
    # round down when added to the first, the rest of the pool's each round
    # up. c_b comes from float64 frame sums, as the measure step forms it, so
    # the adds' roundings reach the corrected share and need not cancel.
    u = 2.0**-24
    small = {
        1000: float(np.float32(u * (1 - 2.0**-23))),
        2000: float(np.float32(u * (1 + 2.0**-22))),
    }
    weights = {1000: 1.0, 2000: 1023.0}
    rows = []
    for household_id in (1000, 2000):
        row = {
            "household_id": household_id,
            "household_weight": weights[household_id],
            "region": "LONDON",
            "council_tax": 1_400.0,
            "rent": 0.0,
            "tenure_type": "OWNED_OUTRIGHT",
            "corporate_wealth": 0.0,
            "private_pension_wealth": 0.0,
        }
        for index, column in enumerate(COICOP):
            row[column] = 1024.0 * (1.0 if index == 0 else small[household_id])
        rows.append(row)
    household = pd.DataFrame(rows)
    person = pd.DataFrame(
        [
            {
                "person_id": household_id * 10,
                "person_benunit_id": household_id * 10,
                "person_household_id": household_id,
                "age": 40,
                "gender": "FEMALE",
                "employment_income": 0.0,
                "months_since_last_birthday": 6.0,
                "attends_private_school_random_draw": 0.5,
                "attends_private_school": False,
            }
            for household_id in (1000, 2000)
        ]
    )
    benunit = pd.DataFrame({"benunit_id": [10000, 20000]})
    sums = {
        household_id: math.fsum(
            float(value)
            for value in household.loc[household["household_id"] == household_id].iloc[
                0
            ][list(COICOP)]
        )
        * weights[household_id]
        for household_id in (1000, 2000)
    }
    scale = sum(weights.values()) / weights[1000]
    shares = pool_shares(
        {block: {"consumption_shareholding": total} for block, total in sums.items()},
        {block: sum(weights.values()) / weight for block, weight in weights.items()},
        key_variables={"consumption_shareholding": "consumption"},
    )

    pool = _simulation(person, benunit, household)
    alone = household[household["household_id"] == 1000].copy()
    alone["household_weight"] = alone["household_weight"] * scale
    block = _simulation(person.iloc[:1], benunit.iloc[:1], alone)
    inject_pool_shares(block, shares[1000], (D,), weight_variable="household_weight")
    corrected = float(_values(block, "consumption_shareholding", D)[0])
    reference = float(_values(pool, "consumption_shareholding", D)[0])
    gap = abs(corrected / reference - 1)
    assert gap > pool_share_rtol(1, 2, key_inputs=1)
    assert gap <= pool_share_rtol(1, 2, key_inputs=len(COICOP))
