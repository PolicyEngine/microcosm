"""A synthetic base frame and the release's re-runs of its QRF stages.

Shared by the release base-input invariant tests. The base is built with the
production stage functions, called as ``tools/build_us_puf_support_base.py``
calls them (childcare, energy subsidy, retirement contributions, once before
the PUF support clone and once after). The release frames apply the release
builder's own repair functions and then call each stage as
``tools/build_us_fiscal_refresh_release.py::_main`` calls it: the function,
whether it is called at all, and its constant keyword arguments are read from
``_main``'s source. A change to those calls (dropping a re-run, or passing a
flag that consumes the base's value) changes what these helpers run.

The tests are stage-level: they replay ``_main``'s calls on a small frame.
They do not execute ``_main``, which needs the ledger, the engine and the
release inputs.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare import (
    US_CHILDCARE_OUTPUT_COLUMNS,
    with_us_childcare_inputs,
)
from microcosm.build.us_runtime.energy_subsidy import (
    US_ENERGY_SUBSIDY_OUTPUT_COLUMNS,
    with_us_energy_subsidy_input,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    with_us_retirement_contribution_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

TEST_PATHS = paths_for("microcosm-build")
RELEASE_TOOL = TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
BASE_TOOL = TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
SEED = 0
PERIOD = 2024
N_PEOPLE = 320
STAGE_CALL_PREFIXES = ("with_us_", "_with_", "impute_us_", "with_household_")

# The Route A release's recorded Social Security component factors
# (calibration_diagnostics.json build.support_value_repairs, release
# populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z).
ROUTE_A_SS_FACTORS = {
    "social_security_retirement": 1.1474528249656013,
    "social_security_disability": 1.1103656007127674,
    "social_security_survivors": 2.4926129688735283,
    "social_security_dependents": 0.9056065944636575,
}

# Stage -> (entity, output columns, stage function).
STAGES: dict[str, tuple[str, tuple[str, ...], Callable[..., Frame]]] = {
    "childcare": ("spm_unit", US_CHILDCARE_OUTPUT_COLUMNS, with_us_childcare_inputs),
    "energy_subsidy": (
        "spm_unit",
        US_ENERGY_SUBSIDY_OUTPUT_COLUMNS,
        with_us_energy_subsidy_input,
    ),
    "retirement_contributions": (
        "person",
        US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
        with_us_retirement_contribution_inputs,
    ),
}


@dataclass(frozen=True)
class StageCall:
    """One stage call in the release builder's ``_main``.

    ``path`` holds the conditions that must hold for the call to run, one per
    enclosing ``if`` (negated for an ``else`` branch), as source text.
    ``statement`` is the source of the statement holding the call. ``plain``
    is true when that statement is an assignment whose value is the call
    itself and nothing but ``if`` statements encloses it, so ``path`` is its
    whole execution condition. ``context`` names any other construct around
    the call (a loop, ``try``, ``with``, a conditional expression, a nested
    function, ...). ``period`` is the constant the module binds ``PERIOD`` to,
    or ``None`` unless that is its only binding (see ``constant_binding``).
    """

    name: str
    lineno: int
    path: tuple[str, ...]
    statement: str
    plain: bool
    context: tuple[str, ...]
    pool_absent_only: bool
    on_base_h5_path: bool | None
    period: object | None
    node: ast.Call


def constant_binding(source: str, name: str) -> object | None:
    """The constant a module binds ``name`` to, or ``None``.

    Only a single top-level ``name = <constant>`` counts, and only when nothing
    else in the module binds ``name``: no other assignment, augmented or
    annotated assignment, ``del``, ``global``/``nonlocal``, import alias,
    parameter, loop or ``with`` target, ``except ... as``, ``match`` capture,
    walrus, or function or class of that name.
    """

    tree = ast.parse(source)
    bindings: list[ast.AST] = []
    for node in ast.walk(tree):
        if (
            (
                isinstance(node, ast.Name)
                and node.id == name
                and not isinstance(node.ctx, ast.Load)
            )
            or (isinstance(node, ast.Global | ast.Nonlocal) and name in node.names)
            or (
                isinstance(node, ast.alias)
                and (node.asname or node.name.split(".")[0]) == name
            )
            or (isinstance(node, ast.arg) and node.arg == name)
            or (isinstance(node, ast.ExceptHandler) and node.name == name)
            or (isinstance(node, ast.MatchAs | ast.MatchStar) and node.name == name)
            or (isinstance(node, ast.MatchMapping) and node.rest == name)
            or (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node.name == name
            )
        ):
            bindings.append(node)
    top = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
        and isinstance(statement.value, ast.Constant)
    ]
    if len(top) != 1 or bindings != [top[0].targets[0]]:
        return None
    return top[0].value.value


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# Three-valued logic: True, False, or None for "not decided by the model".
def _and(values: list[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def _or(values: list[bool | None]) -> bool | None:
    if any(value is True for value in values):
        return True
    return False if all(value is False for value in values) else None


def _not(value: bool | None) -> bool | None:
    return None if value is None else not value


def _is_none_test(node: ast.expr) -> tuple[str, bool] | None:
    """``("pool_frame", True)`` for ``pool_frame is None``, ``False`` for
    ``is not None``; the same for ``args.<name>``; otherwise ``None``."""

    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is | ast.IsNot)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    ):
        return None
    return ast.unparse(node.left), isinstance(node.ops[0], ast.Is)


def evaluate(test: ast.expr, known_none: dict[str, bool]) -> bool | None:
    """Evaluate a guard given which names are known to be ``None``.

    ``known_none`` maps source names (``"pool_frame"``, ``"args.base_h5"``) to
    whether each is ``None``. ``X is None``, ``X is not None`` and a bare ``X``
    (truthy when not ``None``) are decided for known names; ``and``, ``or``
    and ``not`` combine them; anything else is undecided.
    """

    if isinstance(test, ast.BoolOp):
        values = [evaluate(value, known_none) for value in test.values]
        return _and(values) if isinstance(test.op, ast.And) else _or(values)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _not(evaluate(test.operand, known_none))
    is_none = _is_none_test(test)
    if is_none is not None:
        name, wants_none = is_none
        if name in known_none:
            return known_none[name] == wants_none
        return None
    name = ast.unparse(test)
    if isinstance(test, ast.Name | ast.Attribute) and name in known_none:
        return not known_none[name]
    return None


# A validated pool is present: the call must be unreachable to count as a
# release re-run outside the pool path.
_POOL_PRESENT = {"pool_frame": False}
# The reviewed path: ``--base-h5`` given, no validated pool.
_BASE_H5_PATH = {"pool_frame": True, "args.base_h5": False}


# Constructs whose bodies run conditionally, repeatedly or later in ways the
# ``if``-path model does not describe.
_OPAQUE = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.IfExp,
    ast.BoolOp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def release_stage_calls(source: str | None = None) -> tuple[StageCall, ...]:
    """Every stage call in ``_main``, in source order, with its path condition.

    ``pool_absent_only`` is true when the path condition is false whenever a
    validated pool is present. ``on_base_h5_path`` evaluates the path on the
    reviewed ``--base-h5`` path: true (runs), false (skipped) or ``None``
    (decided by something this model does not know). ``source`` defaults to
    the release builder's source.
    """

    if source is None:
        source = RELEASE_TOOL.read_text()
    tree = ast.parse(source)
    period = constant_binding(source, "PERIOD")
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_main"
    )
    calls: list[StageCall] = []

    def visit(
        node: ast.AST,
        path: tuple[tuple[ast.expr, bool], ...],
        statement: ast.stmt | None,
        context: tuple[str, ...],
    ) -> None:
        if isinstance(node, ast.stmt):
            statement = node
        if isinstance(node, ast.If):
            visit(node.test, path, statement, context)
            for child in node.body:
                visit(child, (*path, (node.test, True)), None, context)
            for child in node.orelse:
                visit(child, (*path, (node.test, False)), None, context)
            return
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None and name.startswith(STAGE_CALL_PREFIXES):

                def holds(known: dict[str, bool]) -> bool | None:
                    return _and(
                        [
                            evaluate(test, known)
                            if branch
                            else _not(evaluate(test, known))
                            for test, branch in path
                        ]
                    )

                calls.append(
                    StageCall(
                        name=name,
                        lineno=node.lineno,
                        path=tuple(
                            ast.unparse(test)
                            if branch
                            else f"not ({ast.unparse(test)})"
                            for test, branch in path
                        ),
                        statement=ast.unparse(statement) if statement else "",
                        plain=(
                            not context
                            and isinstance(statement, ast.Assign)
                            and statement.value is node
                        ),
                        context=context,
                        pool_absent_only=holds(_POOL_PRESENT) is False,
                        on_base_h5_path=holds(_BASE_H5_PATH),
                        period=period,
                        node=node,
                    )
                )
        inner = (
            (*context, type(node).__name__)
            if node is not main and isinstance(node, _OPAQUE)
            else context
        )
        for child in ast.iter_child_nodes(node):
            visit(child, path, statement, inner)

    visit(main, (), None, ())
    return tuple(sorted(calls, key=lambda call: call.lineno))


class UnmodeledReleaseCallError(RuntimeError):
    """A release stage call these tests cannot replay faithfully.

    Deliberately not an ``AssertionError``: the strict expected failure must
    not absorb a change to the release builder; the test errors instead, so
    the change gets classified.
    """


def base_builder_call_names() -> frozenset[str]:
    """Names of every function the base builder calls."""

    tree = ast.parse(BASE_TOOL.read_text())
    return frozenset(
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
    )


def base_stage(stage: str) -> Callable[[Frame], Frame]:
    """The stage as the base builder calls it (seed and target year only)."""

    function = STAGES[stage][2]
    return lambda frame: function(frame, seed=SEED, time_period=PERIOD)


def release_stage(
    stage: str, calls: tuple[StageCall, ...] | None = None
) -> Callable[[Frame], Frame]:
    """The stage as the release builder's ``_main`` runs it on the ``--base-h5``
    path today.

    Returns the identity when ``_main`` no longer calls the stage or its path
    condition is false on the ``--base-h5`` path. Otherwise the call must be a
    plain ``x = stage(x, seed=args.seed, time_period=PERIOD, ...)`` statement
    under ``if`` guards only, with every other keyword a constant, in a module
    that binds ``PERIOD`` once to the test's period; constant keywords are
    passed through and the seed and period take the test's values. Raises ``UnmodeledReleaseCallError`` for anything else, including
    a guard this model cannot decide or more than one call. ``calls`` defaults
    to the release builder's calls.
    """

    function = STAGES[stage][2]
    if calls is None:
        calls = release_stage_calls()
    matches = [call for call in calls if call.name == function.__name__]
    if not matches:
        return lambda frame: frame
    if len(matches) > 1:
        raise UnmodeledReleaseCallError(
            f"_main calls {function.__name__} {len(matches)} times; say which "
            "call the invariant tests should replay."
        )
    call = matches[0]
    if not call.plain:
        raise UnmodeledReleaseCallError(
            f"_main's call to {function.__name__} is not a plain assignment "
            f"under if-guards only ({call.statement!r}; context {call.context})."
        )
    if call.on_base_h5_path is None:
        raise UnmodeledReleaseCallError(
            f"_main's guard for {function.__name__} ({' and '.join(call.path)}) "
            "is not decided by the test's model of the --base-h5 path."
        )
    if call.on_base_h5_path is False:
        return lambda frame: frame
    keywords = _replay_keywords(call, function.__name__)
    return lambda frame: function(frame, **keywords)


def _replay_keywords(call: StageCall, name: str) -> dict[str, object]:
    """Keyword arguments for replaying ``call`` on one frame, or refuse.

    The call must read ``x = stage(x, seed=args.seed, time_period=PERIOD, ...)``
    with every other keyword a constant: one positional argument, the frame it
    assigns back to, and the build's seed and period. The module must bind
    ``PERIOD`` once, to the period the tests replay.
    """

    statement = ast.parse(call.statement).body[0]
    node = call.node
    target = statement.targets[0] if len(statement.targets) == 1 else None
    if not (
        isinstance(target, ast.Name)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == target.id
    ):
        raise UnmodeledReleaseCallError(
            f"_main's call to {name} does not pass the frame it assigns back to "
            f"as its only positional argument ({call.statement!r})."
        )
    keywords: dict[str, object] = {}
    expected = {"seed": "args.seed", "time_period": "PERIOD"}
    for keyword in node.keywords:
        if keyword.arg in expected:
            if ast.unparse(keyword.value) != expected[keyword.arg]:
                raise UnmodeledReleaseCallError(
                    f"_main passes {name} {keyword.arg}="
                    f"{ast.unparse(keyword.value)}; the tests replay "
                    f"{expected[keyword.arg]}."
                )
        elif keyword.arg is not None and isinstance(keyword.value, ast.Constant):
            keywords[keyword.arg] = keyword.value.value
        else:
            raise UnmodeledReleaseCallError(
                f"_main passes {name} a non-constant argument "
                f"({ast.unparse(keyword)}); the tests do not know how to supply it."
            )
    if set(expected) - {keyword.arg for keyword in node.keywords}:
        raise UnmodeledReleaseCallError(
            f"_main's call to {name} omits seed or time_period."
        )
    if call.period is None or call.period != PERIOD or type(call.period) is not int:
        raise UnmodeledReleaseCallError(
            f"_main passes {name} time_period=PERIOD, but the module does not "
            f"bind PERIOD once to the {PERIOD} the tests replay "
            f"(binding: {call.period!r})."
        )
    return {"seed": SEED, "time_period": PERIOD, **keywords}


def load_release_tool():
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_invariant", RELEASE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def asec_frame(n: int = N_PEOPLE, seed: int = 11) -> Frame:
    """One person per unit, carrying every raw source the three stages read."""

    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=np.int64)
    age = rng.integers(18, 86, n)
    wages = np.where(rng.random(n) < 0.6, np.round(rng.lognormal(10.6, 0.8, n)), 0.0)
    self_employment = np.where(
        rng.random(n) < 0.15, np.round(rng.lognormal(9.8, 1.0, n)), 0.0
    )

    def benefit(mask: np.ndarray, mean: float, sd: float) -> np.ndarray:
        return np.where(mask, np.maximum(np.round(rng.normal(mean, sd, n)), 1.0), 0.0)

    earned = (wages > 0) | (self_employment > 0)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids * 10,
            "person_tax_unit_id": ids * 100,
            "person_spm_unit_id": ids * 1_000,
            "person_family_id": ids * 10_000,
            "person_marital_unit_id": ids * 100_000,
            "age": age,
            "is_female": rng.random(n) < 0.5,
            "has_esi": (wages > 0) & (rng.random(n) < 0.7),
            "tax_unit_role_input": ["HEAD"] * n,
            "WSAL_VAL": wages,
            "SEMP_VAL": self_employment,
            "employment_income_before_lsr": wages,
            "self_employment_income_before_lsr": self_employment,
            "RETCB_VAL": np.where(
                earned & (rng.random(n) < 0.5),
                np.round((wages + self_employment) * rng.uniform(0.01, 0.12, n)),
                0.0,
            ),
            "SPM_CHILDCAREXPNS": np.where(
                (age < 50) & (rng.random(n) < 0.25),
                np.round(rng.lognormal(8.5, 0.7, n)),
                0.0,
            ),
            "SPM_ENGVAL": np.where(
                rng.random(n) < 0.12, np.round(rng.lognormal(6.0, 0.5, n)), 0.0
            ),
            "social_security_retirement": benefit(
                (age >= 62) & (rng.random(n) < 0.75), 21_000, 6_000
            ),
            "social_security_disability": benefit(
                (age < 62) & (rng.random(n) < 0.1), 15_000, 4_000
            ),
            "social_security_survivors": benefit(rng.random(n) < 0.08, 14_000, 5_000),
            "social_security_dependents": benefit(rng.random(n) < 0.06, 8_000, 3_000),
        }
    )
    tables = {
        entity: pd.DataFrame({f"{entity}_id": ids * scale})
        for entity, scale in (
            ("household", 10),
            ("tax_unit", 100),
            ("spm_unit", 1_000),
            ("family", 10_000),
            ("marital_unit", 100_000),
        )
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = np.where(
        rng.random(n) < 0.45, "JOINT", "SINGLE"
    )
    weights = rng.uniform(500.0, 3_000.0, n)
    return Frame(tables, US_SCHEMA, {"household": Weights(weights, WeightKind.DESIGN)})


def replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def with_scaled_weights(frame: Frame, factor: float) -> Frame:
    weights = frame.weights_for("household")
    return Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {"household": Weights(weights.values * factor, weights.kind)},
        frame.strata,
    )


def build_base(n: int = N_PEOPLE, seed: int = 11) -> Frame:
    """ASEC pass, PUF-support clone, PUF-half income, then the post-clone pass."""

    frame = asec_frame(n, seed)
    for stage in STAGES:
        frame = base_stage(stage)(frame)
    expanded = clone_us_frame_for_puf_support(frame)
    person = expanded.table("person").copy()
    puf = person["person_support_clone_index"].astype(int).to_numpy() > 0
    rng = np.random.default_rng(seed + 18)
    # Stand-in for the PUF tax-detail QRF: the clone half carries different
    # wages, self-employment income and Social Security from its ASEC twin.
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        *ROUTE_A_SS_FACTORS,
    ):
        values = person[column].to_numpy(dtype=np.float64).copy()
        values[puf] = np.round(values[puf] * rng.lognormal(0.0, 0.35, puf.sum()))
        person[column] = values
    frame = replace_person(expanded, person)
    for stage in STAGES:
        frame = base_stage(stage)(frame)
    return frame


def outputs(frame: Frame, stage: str) -> pd.DataFrame:
    entity, columns, _ = STAGES[stage]
    return frame.table(entity).set_index(f"{entity}_id")[list(columns)].sort_index()


def assert_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Require two output tables to hold exactly the same values.

    ``assert_frame_equal`` compares floats to a relative tolerance of 1e-5 by
    default; these invariants are about exact reproduction. ``check_exact``
    compares values, not representations: it treats ``+0.0`` and ``-0.0`` as
    equal.
    """

    pd.testing.assert_frame_equal(left, right, check_exact=True)


def ss_targets(release_tool, frame: Frame) -> list[SimpleNamespace]:
    """SSA targets that make the release repair apply the Route A factors."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    role_by_column = {
        column: role
        for role, column in release_tool.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.items()
    }
    return [
        SimpleNamespace(
            metadata={"target_role": role_by_column[column]},
            value=float(np.sum(person[column].to_numpy(dtype=np.float64) * weights))
            * factor,
        )
        for column, factor in ROUTE_A_SS_FACTORS.items()
    ]


class ReleaseScenario:
    """The synthetic base, the release's repaired frames, memoized re-runs.

    The repairs follow ``_main`` on the ``--base-h5`` path, where its order is
    pinned by ``test_release_value_repairs_precede_the_stage_calls``: the
    population mass repair, then the Social Security component repair, both
    before the three stages. Of the calls ``_main`` makes in between, the
    non-Schedule-D capital-gains repair rescales ``non_sch_d_capital_gains``,
    which is not among the three stages' eight predictors, and QBI
    reconciliation is not modeled here. Replayed on the real Route A base
    (macOS arm64), applying only the recorded Social Security factors
    reproduces the release's stored outputs of all three stages
    (``docs/evidence/us-release-reimputation-1191/``).
    """

    def __init__(self, release_tool) -> None:
        self.release_tool = release_tool
        self.base = build_base()
        self.mass_repaired, mass = release_tool._with_base_population_mass_repair(
            self.base
        )
        assert mass["applied"] and mass["factor"] != 1.0
        self.release_prefix, ss = (
            release_tool._with_social_security_component_value_repair(
                self.mass_repaired, ss_targets(release_tool, self.mass_repaired)
            )
        )
        assert ss["applied"]
        self._reruns: dict[tuple[str, str], pd.DataFrame] = {}

    def rerun(self, start: str, stage: str) -> pd.DataFrame:
        """Outputs of ``stage``, called as ``_main`` calls it, on ``base``,
        ``mass_repaired`` or ``release_prefix`` (mass repair, then Social
        Security repair)."""

        key = (start, stage)
        if key not in self._reruns:
            frame = {
                "base": self.base,
                "mass_repaired": self.mass_repaired,
                "release_prefix": self.release_prefix,
            }[start]
            self._reruns[key] = outputs(release_stage(stage)(frame), stage)
        return self._reruns[key]
