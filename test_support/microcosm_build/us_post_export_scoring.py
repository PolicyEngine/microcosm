"""Household-batched post-export scoring (route A remediation, microcosm#956).

The reform-coverage smoke, reform_validation and demographics score the
written release H5 through ``_HouseholdBatchedPostExportScorer`` instead of one
whole-pool Microsimulation each. These tests run the shipped consumers (the
41 smoke probes, the shipped reform-validation specs and baseline levels, and
the age distribution) against a fake engine on a small nested frame and pin
the plan's contract: output identical to the unbatched path, batches that
partition the pool, one live engine at a time, the SPM selection on every
construction, one reform system per reform (handed to each batch engine
without ``reform=``), one baseline pass per plan, ascending periods per engine,
engine-free recording, the ``_main`` and writer wiring, the scored-sha
binding, the nesting, order and weight premises, the additivity guards
(population aggregates, baseline-branch readers, behavioral-response
parameters, pinned against the installed engine's sources), the calibration
result's target frames dropped before the export, and a plan that cannot be
built joining the terminal batch instead of raising. Real-engine tests cover
whole-file equality for selected reforms, the additivity refusals, and all
shipped baseline plans. The MD CCS request-order premise is observed, with a
warning if the upstream error disappears. The complete reform sweep runs in
sequential worker processes through ``tools/sweep_us_post_export_scoring.py``;
fake-engine tests check its recording and worker plumbing here.
"""

# ruff: noqa: F401

from __future__ import annotations

import ast
import gc
import importlib.util
import inspect
import json
import re
import sys
import warnings
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.reform_coverage_smoke as smoke_module
import microcosm.build.us_runtime.reform_validation as reform_validation_module
from microcosm.build.us_runtime.demographics import demographics_payload
from microcosm.build.us_runtime.reform_validation import (
    US_RELEASE_SPM_SELECTION,
    ReformValidationSpec,
)
from microcosm.build.us_runtime.release_input_coverage import (
    ReformCoverageProbe,
    us_release_reform_coverage_probes,
)
from microcosm.calibrate import TargetRegistry, TargetSpec, calibrate
from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_builder_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


# ---------------------------------------------------------------------------
# A small nested US frame and a fake engine over it
# ---------------------------------------------------------------------------

# (state_fips, [ages]) per household; households 2 and 5 hold two tax units.
_HOUSEHOLDS = (
    (24, (41, 39, 6)),
    (6, (29, 31)),
    (36, (72,)),
    (24, (35, 8, 3, 1)),
    (48, (55, 52, 19)),
    (6, (24,)),
    (36, (47, 16)),
)
_SECOND_TAX_UNIT = {2: 1, 5: 2}  # household -> member index starting unit 2


def _nested_frame() -> Frame:
    rows = []
    person_id = 0
    for household_id, (_, ages) in enumerate(_HOUSEHOLDS, start=1):
        split = _SECOND_TAX_UNIT.get(household_id)
        for index, age in enumerate(ages):
            person_id += 1
            tax_unit = household_id * 10 + (
                1 if split is not None and index >= split else 0
            )
            rows.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit,
                    "person_spm_unit_id": household_id * 100,
                    "person_family_id": household_id * 1000,
                    "person_marital_unit_id": person_id * 10_000,
                    "age": float(age),
                }
            )
    person = pd.DataFrame(rows)
    household_ids = np.arange(1, len(_HOUSEHOLDS) + 1, dtype=np.int64)
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.asarray([state for state, _ in _HOUSEHOLDS]),
            }
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": np.unique(person["person_tax_unit_id"].to_numpy())}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids * 100}),
        "family": pd.DataFrame({"family_id": household_ids * 1000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    # Integer-valued weights and values keep every weighted sum exact, so the
    # batched and unbatched payloads can be compared with ==.
    weights = np.asarray([3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]) * 100.0
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


class _FakeReform:
    def __init__(self, label: str, *, parameters=None) -> None:
        self.label = label
        self.delta = float(zlib.crc32(label.encode()) % 41 + 1)
        # Parameter values this reform sets, by full parameter name.
        self.parameters = dict(parameters or {})


class _FakeParameter:
    """A policyengine-core ``Parameter``: named, valued, callable at an instant."""

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.values_list = [value]

    def __call__(self, instant):
        return self.values_list[0]

    def get_descendants(self):
        return iter(())


class _FakeParameterTree:
    """``system.parameters``: ``get_child`` returns a node over its leaves."""

    _BASELINE = {
        "gov.simulation.capital_gains_responses.elasticity": 0.0,
        "gov.simulation.labor_supply_responses.elasticities.income": 0.0,
        "gov.simulation.labor_supply_responses.elasticities.substitution.all": 0.0,
        "gov.simulation.labor_supply_responses.bounds.income_change": 0.5,
    }

    def __init__(self, overrides) -> None:
        self._leaves = tuple(
            _FakeParameter(name, value)
            for name, value in {**self._BASELINE, **overrides}.items()
        )

    def get_child(self, path: str):
        leaves = tuple(leaf for leaf in self._leaves if leaf.name.startswith(path))
        assert leaves, path
        return SimpleNamespace(
            name=path, children={}, get_descendants=lambda: iter(leaves)
        )


class _FakeSeries:
    """The slice of a policyengine MicroSeries the consumers read."""

    def __init__(self, values, weights) -> None:
        self._values = np.asarray(values)
        self.weights = np.asarray(weights, dtype=np.float64)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self):
        return self._values

    def sum(self) -> float:
        return float(pd.Series(self._values).multiply(self.weights).sum())


_HOUSEHOLD_VARIABLES = frozenset({"state_code_str", "state_fips"})
_PERSON_VARIABLES = frozenset(
    {"age", "in_poverty", "is_child", "ssi", "medicare_cost", "head_start", "wic"}
)
_SPM_UNIT_VARIABLES = frozenset({"snap", "housing_assistance", "spm_unit_benefits"})


def _native_entity(variable: str) -> str:
    """Person, SPM-unit and household measures as the engine keys them; every
    other measure (taxes, credits, SOI caps) on the tax unit."""
    if variable in _HOUSEHOLD_VARIABLES:
        return "household"
    if variable in _PERSON_VARIABLES:
        return "person"
    if variable in _SPM_UNIT_VARIABLES or variable.startswith("spm_unit"):
        return "spm_unit"
    return "tax_unit"


def _household_positions(frame: Frame, entity: str) -> np.ndarray:
    household_positions = pd.Series(
        np.arange(frame.n("household")),
        index=frame.table("household")["household_id"].to_numpy(),
    )
    person = frame.table("person")
    if entity == "household":
        return np.arange(frame.n("household"))
    if entity == "person":
        return household_positions.reindex(person["person_household_id"]).to_numpy()
    first_household = person.drop_duplicates(f"person_{entity}_id").set_index(
        f"person_{entity}_id"
    )["person_household_id"]
    unit_households = first_household.reindex(
        frame.table(entity)[f"{entity}_id"].to_numpy()
    )
    return household_positions.reindex(unit_households.to_numpy()).to_numpy()


def _native_values(frame: Frame, variable: str, period: int, delta: float):
    person = frame.table("person")
    state_fips = frame.table("household")["state_fips"].to_numpy()
    if variable == "state_fips":
        return state_fips
    if variable == "state_code_str":
        return np.asarray(
            [US_STATE_NUMERIC_FIPS_TO_POSTAL[int(fips)] for fips in state_fips],
            dtype=object,
        )
    if variable == "age":
        return person["age"].to_numpy(dtype=np.float64)
    if variable == "is_child":
        return (person["age"].to_numpy() < 18).astype(np.float64)
    if variable == "in_poverty":
        return person["person_id"].to_numpy() % 3 == 0
    entity = _native_entity(variable)
    ids = frame.table(entity)[f"{entity}_id"].to_numpy()
    base = zlib.crc32(f"{variable}@{period}".encode()) % 89
    return (base + ids % 11 + delta).astype(np.float64)


def _map_values(frame: Frame, values, native: str, target: str):
    if target == native:
        return values
    person = frame.table("person")
    if target == "person":
        if native == "household":
            return np.asarray(values)[_household_positions(frame, "person")]
        unit_positions = pd.Series(
            np.arange(frame.n(native)),
            index=frame.table(native)[f"{native}_id"].to_numpy(),
        ).reindex(person[f"person_{native}_id"].to_numpy())
        return np.asarray(values)[unit_positions.to_numpy()]
    if target == "household":
        out = np.zeros(frame.n("household"), dtype=np.float64)
        np.add.at(out, _household_positions(frame, native), values)
        return out
    raise NotImplementedError(f"{native} -> {target}")


class _EngineLog:
    def __init__(self) -> None:
        self.constructions: list = []
        self.systems: list = []
        self.max_unreleased_at_construction = 0
        self.fail_on: str | None = None
        # variable -> the watched formulas computing it leaves a value for.
        self.computes: dict[str, tuple[str, ...]] = {}
        # variable -> periods the "written H5" stores for it as an input.
        self.stored_inputs: dict[str, tuple[int, ...]] = {}


def _fake_engine(log: _EngineLog):
    class FakeSystem:
        def __init__(self, reform=None) -> None:
            self.reform = reform
            self.parameters = _FakeParameterTree(
                {} if reform is None else reform.parameters
            )
            log.systems.append(self)

    class FakeHolder:
        def __init__(self, periods) -> None:
            self._periods = periods

        def get_known_periods(self):
            return sorted(self._periods)

    class FakeMicrosimulation:
        default_tax_benefit_system = FakeSystem
        default_tax_benefit_system_instance = SimpleNamespace(
            parameters=_FakeParameterTree({})
        )

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None, spm=None):
            unreleased = sum(engine.dataset is not None for engine in log.constructions)
            log.max_unreleased_at_construction = max(
                log.max_unreleased_at_construction, unreleased
            )
            self.dataset = dataset
            # What this construction saw, kept past release_engine_simulation
            # (which severs ``dataset``).
            self.household_ids = tuple(dataset.table("household")["household_id"])
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system
            self.spm = spm
            self.periods: list[int] = []
            self.branches: dict = {}
            self.known = {
                name: set(periods) for name, periods in log.stored_inputs.items()
            }
            log.constructions.append(self)

        @property
        def policy_reform(self):
            """The reform this engine scores: passed, or carried by its system."""
            if self.reform is not None:
                return self.reform
            return getattr(self.tax_benefit_system, "reform", None)

        def get_holder(self, variable):
            return FakeHolder(self.known.get(variable, ()))

        def calculate(self, variable, period, map_to=None):
            frame = self.dataset
            assert frame is not None, "calculate on a released engine"
            if variable == log.fail_on:
                raise RuntimeError(f"fixture engine refuses {variable}")
            period = int(period)
            self.periods.append(period)
            for computed in (variable, *log.computes.get(variable, ())):
                self.known.setdefault(computed, set()).add(period)
            native = _native_entity(variable)
            reform = self.policy_reform
            delta = 0.0 if reform is None else reform.delta
            values = _native_values(frame, variable, period, delta)
            target = map_to or native
            values = _map_values(frame, values, native, target)
            weights = frame.weights_for("household").values[
                _household_positions(frame, target)
            ]
            return _FakeSeries(values, weights)

    return FakeMicrosimulation


@pytest.fixture
def fake_reforms(monkeypatch, builder):
    """Engine-free reform objects for the shipped probes and specs.

    Full garbage collections are stubbed too; these stand-ins do not create
    the engine state collected between production passes.
    """
    monkeypatch.setattr(builder, "_collect_family_garbage", lambda: None)
    monkeypatch.setattr(
        reform_validation_module, "gc", SimpleNamespace(collect=lambda *args: 0)
    )
    monkeypatch.setattr(
        smoke_module, "_build_reform", lambda probe: _FakeReform(probe.id)
    )
    monkeypatch.setattr(
        ReformValidationSpec, "build_reform", lambda spec: _FakeReform(spec.id)
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        lambda changes: _FakeReform(json.dumps(changes, sort_keys=True, default=str)),
    )


def _empty_calibration_result():
    return SimpleNamespace(diagnostics=[], problem=SimpleNamespace(targets=[]))


#: Fitted (final estimate, target) per in-sample JCT reform in
#: ``_calibration_result_with_in_sample_fit``, in spec order.
_IN_SAMPLE_FIT = ((-1_250.0, -1_000.0), (-3_500.0, -4_000.0))


def _calibration_result_with_in_sample_fit(period: int):
    """A calibration fit covering the first two in-sample JCT reforms, plus a
    target no reform reads. Reform validation takes those two rows from the
    fit instead of simulating them."""
    fitted = reform_validation_module.in_sample_reform_specs(period=period)[:2]
    fit = (*_IN_SAMPLE_FIT, (7.0, 8.0))
    names = (*(spec.id for spec in fitted), "fixture_other_target")
    return SimpleNamespace(
        diagnostics=[
            SimpleNamespace(final_estimate=estimate, target=target)
            for estimate, target in fit
        ],
        problem=SimpleNamespace(targets=[SimpleNamespace(name=name) for name in names]),
    )


def _written_h5(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "populace_us_2024.h5"
    path.write_bytes(b"the written release H5 bytes")
    return path


def _scorer(builder, frame, log, batch_size, tmp_path, *, loads=None):
    path = _written_h5(tmp_path)

    def load_frame(load_path, *, expected_sha256):
        if loads is not None:
            loads.append((Path(load_path), expected_sha256))
        return frame

    return builder._HouseholdBatchedPostExportScorer(
        path,
        maximum_microsim_batch_size=batch_size,
        microsimulation_cls=_fake_engine(log),
        dataset_from_frame=lambda batch_frame: batch_frame,
        load_frame=load_frame,
    )


def _unbatched_simulate(frame, log):
    engine_cls = _fake_engine(log)

    def simulate(reform):
        return engine_cls(
            dataset=frame, reform=reform, spm=dict(US_RELEASE_SPM_SELECTION)
        )

    return simulate


def _run_consumers(builder, simulate_for):
    """Run the three shipped consumers; ``simulate_for(name, consumer)``."""
    gate = builder.us_reform_coverage_smoke_gate(
        simulate=simulate_for(
            "reform_coverage_smoke", builder._reform_coverage_smoke_consumer
        ),
        period=builder.PERIOD,
    )
    payload_for = builder._reform_validation_consumer(
        result=_empty_calibration_result(), release_id="fixture-release"
    )
    payload = payload_for(simulate_for("reform_validation", payload_for))
    ages, weights = builder._demographics_consumer(
        simulate_for("demographics", builder._demographics_consumer)
    )
    demographics = demographics_payload(ages, weights, period=builder.PERIOD)
    return gate, payload, demographics


def _batched_run(builder, frame, batch_size, tmp_path):
    log = _EngineLog()
    scorer = _scorer(builder, frame, log, batch_size, tmp_path)
    consumers: dict[str, object] = {}

    def simulate_for(name, consumer):
        plan = builder._record_post_export_baseline_plan(consumer)
        consumers[name] = scorer.open_consumer(name, plan)
        return consumers[name].simulate

    outputs = _run_consumers(builder, simulate_for)
    return outputs, log, scorer, consumers


# ---------------------------------------------------------------------------
# (a) output invariance
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# (b)-(g) the engine contract
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Additivity: population aggregates and the missing baseline branch
# ---------------------------------------------------------------------------






_WALKER_VARIABLES = (
    "household_income_decile",
    "medicaid_slcsp_state_denominator",
    "other",
)










_POPULATION_AGGREGATE_MARKER = re.compile(
    r"\bsum_by_state\(|\bMicroSeries\(|\.decile_rank\(|\bquantile\(|\bpercentile\("
    r"|\bnp\.(?:sum|mean|median|average|nansum|nanmean|nanmedian|sort|argsort"
    r"|cumsum)\((?![^\n]*\baxis\s*=)"
)
_WEIGHT_READ = re.compile(r"[\"'](\w+_weight)[\"']")
_BASELINE_BRANCH_MARKER = re.compile(r"\.baseline\b|get_branch\(\s*[\"']baseline[\"']")


def _is_variable_class(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        getattr(base, "id", None) == "Variable" for base in node.bases
    )


def _engine_formula_modules() -> list[tuple[str, ast.Module]]:
    """Every policyengine-us module that can define a formula, parsed but never
    imported: ``variables`` and the contrib ``reforms``."""
    spec = importlib.util.find_spec("policyengine_us")
    root = Path(next(iter(spec.submodule_search_locations)))
    modules = []
    for package in ("variables", "reforms"):
        for path in sorted((root / package).rglob("*.py")):
            source = path.read_text()
            modules.append((source, ast.parse(source)))
    return modules


def _variables_reaching(modules, marker, *, weight_reads: bool = False) -> dict:
    """Variable classes whose source matches ``marker``, directly or through
    a helper function (in a module that defines no Variable) that does."""
    helpers: dict[str, str] = {}
    for source, tree in modules:
        if not any(_is_variable_class(node) for node in ast.walk(tree)):
            helpers.update(
                (node.name, ast.get_source_segment(source, node))
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            )
    reaching = {name for name, body in helpers.items() if marker.search(body)}
    grew = True
    while grew:
        grew = False
        for name, body in helpers.items():
            if name not in reaching and any(
                re.search(rf"\b{other}\(", body) for other in reaching
            ):
                reaching.add(name)
                grew = True
    found: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in ast.walk(tree):
            if not _is_variable_class(node):
                continue
            body = ast.get_source_segment(source, node)
            evidence = [match.group(0) for match in marker.finditer(body)]
            evidence += [
                f"{helper}()"
                for helper in sorted(reaching)
                if re.search(rf"\b{helper}\(", body)
            ]
            if weight_reads and not node.name.endswith("_weight"):
                # A weight variable's own formula broadcasts the household
                # weight; any other formula reading a weight aggregates.
                evidence += _WEIGHT_READ.findall(body)
            if evidence:
                found[node.name] = evidence
    return found




# ---------------------------------------------------------------------------
# (h) engine-free recording
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# (i) wiring in _main
# ---------------------------------------------------------------------------


def _function_source(builder, name: str) -> tuple[str, ast.AST]:
    source = inspect.getsource(getattr(builder, name))
    return source, ast.parse(source)


def _called_names(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.append(
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
    return names




def _plan_args(**overrides) -> SimpleNamespace:
    """The ``_main`` arguments the post-export plan reads."""
    return SimpleNamespace(
        **{
            "skip_reform_coverage_smoke": False,
            "skip_reform_validation": False,
            "skip_out_of_sample_reforms": False,
            "skip_demographics": False,
            "maximum_microsim_batch_size": 3,
            **overrides,
        }
    )


@pytest.fixture
def engine_free_loader(builder, monkeypatch):
    """Serve the default scorer construction without an engine or an H5.

    ``policyengine_us.Microsimulation`` is the fake engine, the written H5
    loads as the nested frame, and each batch frame is its own dataset. Yields
    the fake engine's log and the load calls.
    """
    log = _EngineLog()
    loads: list = []

    def load_frame(path, *, expected_sha256=None):
        loads.append((Path(path), expected_sha256))
        return _nested_frame()

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(Microsimulation=_fake_engine(log)),
    )
    monkeypatch.setattr(builder, "_load_frame", load_frame)
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda frame, **_: frame)
    return log, loads










def _small_calibration_problem(n: int = 40):
    """A one-person-per-household frame and one household-total target."""
    rng = np.random.default_rng(0)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "income": rng.uniform(0.0, 100.0, n),
        }
    )
    household = pd.DataFrame(
        {"household_id": np.arange(n), "x": rng.uniform(0.0, 10.0, n)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    target = TargetSpec(
        name="x_total@2024",
        entity="household",
        measure="x",
        value=float(household["x"].sum() * 1.1),
        source="fixture",
        family="fixture",
    )
    return frame, TargetRegistry([target], country="us").to_target_set()










# ---------------------------------------------------------------------------
# (j) the scored bytes are the manifest's bytes
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# (k) the nesting premise
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# (l) the same invariance against policyengine-us
# ---------------------------------------------------------------------------

# (state_fips, county_fips, [(age, pre-LSR wages, weekly hours), ...]). The
# first two households put North Dakota and Maryland in one batch at size 2.
# Test (l) probes the MD CCS request-order premise on that batch and warns if
# the engine no longer raises the expected error.
_ENGINE_HOUSEHOLDS = (
    (38, "38105", ((32, 47_000.0, 40.0),)),
    (
        24,
        "24043",
        (
            (38, 50_000.0, 20.0),
            (40, 75_000.0, 40.0),
            (11, 0.0, 0.0),
            (7, 0.0, 0.0),
            (5, 0.0, 0.0),
            (2, 0.0, 0.0),
        ),
    ),
    (6, "06037", ((58, 150_000.0, 40.0), (55, 0.0, 0.0))),
    (24, "24031", ((29, 31_000.0, 30.0), (2, 0.0, 0.0))),
)


#: Four low-income households with children or a senior, in California, New
#: York and Texas. The guard test checks that the combined fixture has
#: Medicaid enrollees and reaches a watched SLCSP state aggregate.
_MEDICAID_HOUSEHOLDS = (
    (6, "06001", ((26, 6_000.0, 10.0), (4, 0.0, 0.0), (1, 0.0, 0.0))),
    (36, "36047", ((67, 0.0, 0.0),)),
    (
        48,
        "48201",
        ((45, 12_000.0, 25.0), (44, 0.0, 0.0), (15, 0.0, 0.0), (9, 0.0, 0.0)),
    ),
    (6, "06073", ((31, 18_000.0, 30.0), (3, 0.0, 0.0))),
)


def _write_engine_h5(builder, directory: Path, households) -> Path:
    """Write ``households`` as a release H5 through the engine's own writer.

    One tax unit, SPM unit and family per household; household ``i`` weighs
    ``1,000 * i``.
    """
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    rows = []
    person_id = 0
    for household_id, (_, _, members) in enumerate(households, start=1):
        for age, wages, hours in members:
            person_id += 1
            rows.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": household_id * 10,
                    "person_spm_unit_id": household_id * 100,
                    "person_family_id": household_id * 1000,
                    "person_marital_unit_id": person_id * 10_000,
                    "age": age,
                    "employment_income_before_lsr": wages,
                    "weekly_hours_worked_before_lsr": hours,
                }
            )
    person = pd.DataFrame(rows)
    household_ids = np.arange(1, len(households) + 1, dtype=np.int64)
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": household_ids,
                    "state_fips": [state for state, _, _ in households],
                    "county_fips": [county for _, county, _ in households],
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": household_ids * 10}),
            "spm_unit": pd.DataFrame({"spm_unit_id": household_ids * 100}),
            "family": pd.DataFrame({"family_id": household_ids * 1000}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                household_ids.astype(np.float64) * 1_000.0, WeightKind.CALIBRATED
            )
        },
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "populace_us_2024.h5"
    PolicyEngineUSEngine().write_dataset(frame, path, period=builder.PERIOD)
    return path


def _engine_probe(probe_id, budget_measure, period, **reform) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id=probe_id,
        name=probe_id,
        parameter_changes=reform.get("parameter_changes", {}),
        neutralized_variable=reform.get("neutralized_variable"),
        budget_measure=budget_measure,
        period=period,
        binding_inputs=("employment_income_before_lsr",),
        min_abs_effect=1.0,
        effect_direction="baseline_minus_reform",
        expected_sign="either",
        reason="fixture",
        issue="PolicyEngine/microcosm#956",
    )




# ---------------------------------------------------------------------------
# The guards and the shipped request surface against policyengine-us
# ---------------------------------------------------------------------------

#: The (l) households (two in Maryland) and the four Medicaid ones.
_GUARD_HOUSEHOLDS = (*_ENGINE_HOUSEHOLDS, *_MEDICAID_HOUSEHOLDS)




def _load_sweep_module():
    path = _TEST_PATHS.repository / "tools" / "sweep_us_post_export_scoring.py"
    spec = importlib.util.spec_from_file_location("post_export_guard_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

__all__ = [name for name in globals() if not name.startswith("__")]
