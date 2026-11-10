"""Engine-free stand-ins for the engine's per-system variable module leak."""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from microcosm.build.us_runtime.engine_lifecycle import (
    temporary_engine_variable_modules,
)
from test_support.microcosm_build.us_release_head_to_head_scorer import (
    _fixture_materialize,
    _load_head_to_head_module,
    _tiny_frame,
    _tiny_registry,
)


def _variable_module(name: str, variables_dir: Path) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(variables_dir / "income_tax.py")
    exec(
        "class IncomeTax:\n"
        "    def formula(self):\n"
        "        return 42.0\n"
        "formula = IncomeTax.formula\n",
        module.__dict__,
    )
    return module


def _register_variable_module(name: str, variables_dir: Path):
    module = _variable_module(name, variables_dir)
    sys.modules[name] = module
    return tuple(
        weakref.ref(value) for value in (module, module.IncomeTax, module.formula)
    )


def test_temporary_variable_modules_release_modules_classes_and_formulas(
    tmp_path,
) -> None:
    variables_dir = tmp_path / "variables"
    names = [f"741000{i}_-987654_income_tax" for i in range(6)]
    retained = []
    try:
        for name in names:
            with temporary_engine_variable_modules(variables_dir):
                retained.extend(_register_variable_module(name, variables_dir))
                assert sys.modules[name].IncomeTax().formula() == 42.0
            gc.collect()
            assert name not in sys.modules
            assert all(reference() is None for reference in retained)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_temporary_variable_modules_preserve_existing_and_unrelated_modules(
    monkeypatch, tmp_path
) -> None:
    variables_dir = tmp_path / "variables"
    original_name = "7420001_-123_income_tax"
    original = _variable_module(original_name, variables_dir)
    outside_name = "7420002_-123_income_tax"
    outside = _variable_module(outside_name, tmp_path / "variables-other")
    ordinary_name = "slice_memory_fixture.income_tax"
    ordinary = _variable_module(ordinary_name, variables_dir)
    missing_file_name = "7420003_-123_income_tax"
    missing_file = ModuleType(missing_file_name)
    monkeypatch.setitem(sys.modules, original_name, original)

    with temporary_engine_variable_modules(variables_dir):
        monkeypatch.setitem(sys.modules, outside_name, outside)
        monkeypatch.setitem(sys.modules, ordinary_name, ordinary)
        monkeypatch.setitem(sys.modules, missing_file_name, missing_file)
        sys.modules[original_name] = _variable_module(original_name, variables_dir)
        assert sys.modules[original_name] is not original

    assert sys.modules[original_name] is original
    assert original.IncomeTax().formula() == 42.0
    assert sys.modules[outside_name] is outside
    assert sys.modules[ordinary_name] is ordinary
    assert sys.modules[missing_file_name] is missing_file


def test_temporary_variable_modules_clean_up_when_scoring_raises(tmp_path) -> None:
    variables_dir = tmp_path / "variables"
    name = "7430001_-123_income_tax"
    retained = []
    try:
        with pytest.raises(RuntimeError, match="fixture scoring failure"):
            with temporary_engine_variable_modules(variables_dir):
                retained.extend(_register_variable_module(name, variables_dir))
                raise RuntimeError("fixture scoring failure")
        gc.collect()
        assert name not in sys.modules
        assert all(reference() is None for reference in retained)
    finally:
        sys.modules.pop(name, None)


def test_temporary_variable_modules_keep_outer_scope_live(tmp_path) -> None:
    variables_dir = tmp_path / "variables"
    outer_name = "7440001_-123_income_tax"
    inner_name = "7440002_-123_income_tax"
    try:
        with temporary_engine_variable_modules(variables_dir):
            outer_refs = _register_variable_module(outer_name, variables_dir)
            with temporary_engine_variable_modules(variables_dir):
                inner_refs = _register_variable_module(inner_name, variables_dir)
            gc.collect()
            assert outer_name in sys.modules
            assert all(reference() is not None for reference in outer_refs)
            assert inner_name not in sys.modules
            assert all(reference() is None for reference in inner_refs)
        gc.collect()
        assert outer_name not in sys.modules
        assert all(reference() is None for reference in outer_refs)
    finally:
        sys.modules.pop(outer_name, None)
        sys.modules.pop(inner_name, None)


def test_scored_slices_do_not_accumulate_engine_variable_objects(
    monkeypatch, tmp_path
) -> None:
    """Use the actual slice scorer with an engine-free module-leaking seam."""
    module = _load_head_to_head_module()
    variables_dir = tmp_path / "variables"
    engine = ModuleType("policyengine_us")
    engine.CountryTaxBenefitSystem = SimpleNamespace(variables_dir=variables_dir)
    monkeypatch.setitem(sys.modules, "policyengine_us", engine)
    frame = _tiny_frame(measure_values=(100.0, 300.0))
    specs = tuple(_tiny_registry().specs)
    monkeypatch.setattr(
        module.release, "_materialize_target_frame", _fixture_materialize
    )

    def score(slice_index):
        return module._score_household_slice(
            frame,
            specs,
            np.asarray([slice_index % 2], dtype=np.int64),
            chunk_loss_weights=np.ones(len(specs), dtype=np.float64),
            artifact_name="fixture",
            chunk_label="chunk 5/5",
            slice_index=slice_index,
            slice_count=6,
            maximum_microsim_batch_size=1,
        )

    expected = [score(index) for index in range(2)]
    names = [f"745000{i}_-123_income_tax" for i in range(6)]
    retained = []
    scored = []

    def leaking_materialize(frame, specs, **kwargs):
        retained.extend(_register_variable_module(names[len(scored)], variables_dir))
        return _fixture_materialize(frame, specs, **kwargs)

    monkeypatch.setattr(
        module.release, "_materialize_target_frame", leaking_materialize
    )
    try:
        for index, name in enumerate(names):
            scored.append(score(index))
            gc.collect()
            assert name not in sys.modules
            assert all(reference() is None for reference in retained)
            reference_score = expected[index % 2]
            for attribute in ("estimates", "targets", "scales"):
                np.testing.assert_array_equal(
                    getattr(scored[-1], attribute).view(np.uint64),
                    getattr(reference_score, attribute).view(np.uint64),
                )
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_temporary_variable_modules_clean_up_first_lazy_engine_import(
    monkeypatch, tmp_path
) -> None:
    """A fresh spawned worker imports the engine inside its first slice."""
    variables_dir = tmp_path / "variables"
    shared_system = SimpleNamespace()
    engine = ModuleType("policyengine_us")
    engine.CountryTaxBenefitSystem = SimpleNamespace(variables_dir=variables_dir)
    engine.Microsimulation = SimpleNamespace(
        default_tax_benefit_system_instance=shared_system
    )
    monkeypatch.delitem(sys.modules, "policyengine_us", raising=False)
    shared_name = f"{id(shared_system)}_-123_income_tax"
    shared_module = _variable_module(shared_name, variables_dir)
    transient_name = "7460001_-123_income_tax"
    try:
        with temporary_engine_variable_modules():
            monkeypatch.setitem(sys.modules, "policyengine_us", engine)
            monkeypatch.setitem(sys.modules, shared_name, shared_module)
            transient_refs = _register_variable_module(transient_name, variables_dir)
            assert sys.modules[transient_name].IncomeTax().formula() == 42.0

        gc.collect()
        assert sys.modules[shared_name] is shared_module
        assert shared_module.IncomeTax().formula() == 42.0
        assert transient_name not in sys.modules
        assert all(reference() is None for reference in transient_refs)
    finally:
        sys.modules.pop(transient_name, None)
