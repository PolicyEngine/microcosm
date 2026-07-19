"""Bounded-memory behavior of the US release builder (populace#456).

The dense release build accumulated unbounded memory (~2.5 GB/min, OOM at
125-249 GB) through the target materializations. The measured causes, per the
profile on #456:

1. a fresh ``CountryTaxBenefitSystem`` per reform x batch — each build leaks
   ~5,600 permanent ``sys.modules`` entries (~55-60 MB RSS floor) that no
   garbage collection reclaims;
2. finished simulations stay reachable through engine backrefs
   (``system.simulation``, branch clones, ``calc``/``df`` bound-method
   self-cycles), stranding multi-hundred-MB cyclic graphs past the builder's
   generation-0-only collections.

These tests pin the fix: one reform system per target family, and an explicit
release of every finished simulation. The ``requires_us`` tests measure the
real engine on a tiny synthetic frame and fail against the pre-#456 builder
(which built one system per batch: 6 builds where the fixed code does 2).
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.engine_lifecycle import release_engine_simulation
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# release_engine_simulation unit behavior (engine-free)
# ---------------------------------------------------------------------------


class _FakeHolderMap(dict):
    pass


def _fake_population() -> SimpleNamespace:
    population = SimpleNamespace()
    population._holders = _FakeHolderMap(income=object())
    population.simulation = None
    return population


def _fake_simulation(system: SimpleNamespace | None = None) -> SimpleNamespace:
    simulation = SimpleNamespace()
    simulation.tax_benefit_system = system
    simulation.populations = {"person": _fake_population()}
    for population in simulation.populations.values():
        population.simulation = simulation
    simulation.branches = {}
    simulation.baseline = None
    simulation.parent_branch = None
    simulation.dataset = object()
    simulation.tracer = object()
    simulation._fast_cache = {"cached": object()}
    simulation.calc = lambda: None
    simulation.df = lambda: None
    if system is not None:
        system.simulation = simulation
    return simulation


def test_release_severs_backrefs_and_drops_holder_mass() -> None:
    system = SimpleNamespace(simulation=None)
    simulation = _fake_simulation(system)
    population = simulation.populations["person"]

    release_engine_simulation(simulation)

    assert system.simulation is None
    assert population._holders == {}
    assert population.simulation is None
    assert simulation.populations == {}
    assert simulation.dataset is None
    assert simulation.tracer is None
    assert simulation._fast_cache is None
    assert simulation.calc is None
    assert simulation.df is None


def test_release_walks_branch_clones_and_baseline() -> None:
    system = SimpleNamespace(simulation=None)
    simulation = _fake_simulation(system)
    branch = _fake_simulation()
    branch.parent_branch = simulation
    baseline = _fake_simulation()
    baseline.parent_branch = simulation
    simulation.branches = {"reform": branch}
    simulation.baseline = baseline
    branch_population = branch.populations["person"]
    baseline_population = baseline.populations["person"]

    release_engine_simulation(simulation)

    assert simulation.branches == {}
    assert simulation.baseline is None
    assert branch.parent_branch is None
    assert baseline.parent_branch is None
    assert branch_population._holders == {}
    assert baseline_population._holders == {}


def test_release_leaves_a_newer_simulations_shared_backref_alone() -> None:
    """The shared system instance may already belong to a newer live sim."""
    shared_system = SimpleNamespace(simulation=None)
    finished = _fake_simulation(shared_system)
    live = _fake_simulation(shared_system)
    assert shared_system.simulation is live

    release_engine_simulation(finished)

    assert shared_system.simulation is live


def test_release_is_defensive_on_stub_simulations() -> None:
    """Test doubles without engine attributes must pass through unharmed."""
    stub = SimpleNamespace(dataset={"frame": object()})
    release_engine_simulation(stub)
    assert stub.dataset is None
    release_engine_simulation(stub)  # idempotent
    release_engine_simulation(object())  # nothing to sever at all


# ---------------------------------------------------------------------------
# Thread-pool defaults (#447 ops note)
# ---------------------------------------------------------------------------


_THREAD_POOL_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


def test_builder_bounds_default_thread_pools_without_overriding_operator(
    monkeypatch,
) -> None:
    import os

    for variable in _THREAD_POOL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")

    _load_builder_module()

    expected = str(min(os.cpu_count() or 1, 16))
    assert os.environ["OMP_NUM_THREADS"] == "3"  # operator wins
    for variable in _THREAD_POOL_VARIABLES[1:]:
        assert os.environ[variable] == expected


# ---------------------------------------------------------------------------
# Real-engine memory regression + bit identity (requires the [us] extra)
# ---------------------------------------------------------------------------


def _tiny_us_frame(n_households: int = 12) -> Frame:
    """A minimal real-engine frame: one adult per household, wage income."""
    count = n_households
    ids = np.arange(1, count + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids * 10,
            "person_tax_unit_id": ids * 100,
            "person_spm_unit_id": ids * 1_000,
            "person_family_id": ids * 10_000,
            "person_marital_unit_id": ids * 100_000,
            "age": np.linspace(25, 64, count).round(),
            "employment_income_before_lsr": np.linspace(10_000.0, 250_000.0, count),
        }
    )
    tables = {
        entity: pd.DataFrame({f"{entity}_id": ids * factor})
        for entity, factor in (
            ("household", 10),
            ("tax_unit", 100),
            ("spm_unit", 1_000),
            ("family", 10_000),
            ("marital_unit", 100_000),
        )
    }
    tables["person"] = person
    tables["household"]["state_fips"] = np.asarray([6] * count, dtype="int64")
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(count, dtype=np.float64), WeightKind.DESIGN)},
    )


def _variable_module_count() -> int:
    """Count engine variable modules registered in sys.modules.

    policyengine-core names them ``{id(system)}_{path hash}_{file}``; the
    leading id makes the first segment purely numeric, which no ordinary
    package name has.
    """
    return sum(
        1
        for name in sys.modules
        if name.split("_", 1)[0].isdigit() and name.count("_") >= 2
    )


@requires_us
def test_reform_materialization_builds_one_engine_system_per_family() -> None:
    """The pre-#456 builder rebuilt the full tax-benefit system every batch.

    Each build permanently registers one set of variable modules in
    ``sys.modules`` (measured: ~5,600 entries, ~55-60 MB RSS floor, immune to
    gc). Three batches per family must therefore add ~one set, not three:
    against the old builder this assertion sees three sets per family and
    fails.
    """
    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    frame = _tiny_us_frame()
    before_metadata_system = _variable_module_count()
    system = CountryTaxBenefitSystem()
    after_metadata_system = _variable_module_count()
    single_build_modules = after_metadata_system - before_metadata_system
    assert single_build_modules > 1_000  # sanity: the leak metric is live

    # Warm the builder's one-per-process formula-owned gate adapter so the
    # measurement below isolates the per-family behavior (the adapter's own
    # single engine build is bounded and expected).
    builder._assert_no_formula_owned_columns(frame)
    after_metadata_system = _variable_module_count()

    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]
    values = builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=4,  # 12 households -> 3 batches
    )
    assert values.shape == (frame.n("household"),)
    assert np.isfinite(values).all()

    added = _variable_module_count() - after_metadata_system
    # One family build (~1 set) with headroom; the per-batch builder adds ~3
    # sets here and fails.
    assert added <= int(1.5 * single_build_modules), (
        f"reform materialization registered {added} variable modules for one "
        f"family of 3 batches (single build ~{single_build_modules}); the "
        "per-target-family system reuse of populace#456 has regressed"
    )


@requires_us
def test_reform_materialization_batching_is_bit_identical() -> None:
    """Sharing one reform system across batches must not change results.

    The family system sees ``apply_reform_set`` once per batch simulation
    (idempotent variable replacement); a value drift here would mean batch
    N+1's simulation saw different policy than batch 1's.
    """
    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    frame = _tiny_us_frame()
    system = CountryTaxBenefitSystem()
    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]

    single = builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=None,  # one batch
    )
    batched = builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=4,  # three batches sharing the family system
    )
    np.testing.assert_array_equal(single, batched)


@requires_us
def test_released_simulations_do_not_accumulate() -> None:
    """After a family completes, no engine simulations may remain alive."""
    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    def alive_microsimulations() -> int:
        gc.collect()
        return sum(
            1 for obj in gc.get_objects() if type(obj).__name__ == "Microsimulation"
        )

    frame = _tiny_us_frame()
    system = CountryTaxBenefitSystem()
    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]
    alive_before = alive_microsimulations()
    builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=4,
    )
    alive_after = alive_microsimulations()
    assert alive_after <= alive_before, (
        f"{alive_after - alive_before} finished batch simulations survived "
        "the family boundary; release_engine_simulation has regressed"
    )
