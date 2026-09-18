"""Invented controls for the real ACS owner's source-only compilation cache."""

from __future__ import annotations
import __future__

import ast
import builtins
import sys
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

import pytest

from microcosm.build.us_runtime import acs_native_coverage_binding as native


@pytest.fixture(autouse=True)
def empty_compile_cache():
    native._clear_compile_cache()
    yield
    native._clear_compile_cache()


@contextmanager
def _observed_calls():
    """Observe real compile/read calls without replacing attested functions."""
    assert sys.getprofile() is None
    counts = {"compile": 0, "read": 0}
    helper_code = native._compile_source.__code__
    read_code = Path.read_bytes.__code__

    def observe(frame, event, arg):
        if (
            event == "c_call"
            and frame.f_code is helper_code
            and arg is builtins.compile
        ):
            counts["compile"] += 1
        elif event == "call" and frame.f_code is read_code:
            counts["read"] += 1

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(None)


def _module(tmp_path, monkeypatch, source, *, name="invented_acs_compile_cache"):
    path = tmp_path / f"{name}.py"
    path.write_bytes(source)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    monkeypatch.setitem(sys.modules, name, module)
    exec(builtins.compile(source, str(path), "exec", dont_inherit=True), vars(module))
    return module, path


def _index(code):
    result = {}

    def visit(item):
        result[item.co_qualname] = item
        for value in item.co_consts:
            if isinstance(value, CodeType):
                visit(value)

    visit(code)
    return result


@pytest.mark.parametrize("optimize", [0, 1, 2])
def test_plain_helper_preserves_compiled_code_and_qualified_name_order(optimize):
    source = """# coding: utf-8
from __future__ import annotations
label = "café"
def outer(value: Missing):
    "docstring"
    assert value is not None
    def nested():
        return [item for item in range(value)]
    def nested():
        return lambda: {item: item for item in range(value)}
    return nested
class Example:
    @property
    def item(self):
        return label
    @item.setter
    def item(self, value):
        self._item = value
""".encode()
    filename = "invented/café/../module.py"
    expected = builtins.compile(
        source, filename, "exec", dont_inherit=True, optimize=optimize
    )
    with _observed_calls() as calls:
        first = native._compile_source(source, filename, optimize=optimize)
        second = native._compile_source(source, filename, optimize=optimize)
    assert type(native._compile_source) is FunctionType
    assert first is second
    assert first == expected
    assert first.co_filename == filename
    assert _index(first) == _index(expected)
    assert calls["compile"] == 1


def test_complete_inputs_separate_cache_entries():
    source = b"42"
    options = [
        ("literal/../source.py", "exec", 0, 0),
        ("source.py", "exec", 0, 0),
        ("source.py", "eval", 0, 0),
        ("source.py", "exec", __future__.annotations.compiler_flag, 0),
        ("source.py", "exec", 0, 1),
        ("source.py", "exec", 0, 2),
    ]
    with _observed_calls() as calls:
        for filename, mode, flags, optimize in options:
            expected = builtins.compile(
                source,
                filename,
                mode,
                flags=flags,
                dont_inherit=True,
                optimize=optimize,
            )
            first = native._compile_source(
                source, filename, mode, flags=flags, optimize=optimize
            )
            assert first == expected
            assert first.co_filename == filename
            assert (
                native._compile_source(
                    source, filename, mode, flags=flags, optimize=optimize
                )
                is first
            )
    assert calls["compile"] == len(options)
    assert len(native._COMPILE_CACHE) == len(options)
    with _observed_calls() as calls:
        effective = native._compile_source(
            source, "source.py", optimize=sys.flags.optimize
        )
        assert native._compile_source(source, "source.py", optimize=-1) is effective
    assert calls["compile"] == 0


def test_same_literal_path_tracks_a_b_a_source_bytes():
    first_source, second_source = b"value = 1", b"value = 2"
    assert len(first_source) == len(second_source)
    with _observed_calls() as calls:
        first = native._compile_source(first_source, "unchanged.py")
        second = native._compile_source(second_source, "unchanged.py")
        again = native._compile_source(first_source, "unchanged.py")
    assert first is again
    assert second != first
    assert calls["compile"] == 2


def test_changed_compiler_is_called_each_time_and_original_entry_remains(monkeypatch):
    source = b"value = 1"
    original = native._compile_source(source, "compiler.py")
    replacements = []

    def replacement(raw, filename, mode, **options):
        replacements.append(raw)
        return builtins.compile(
            f"value = {len(replacements) + 1}".encode(), filename, mode, **options
        )

    with monkeypatch.context() as context:
        context.setattr(native, "compile", replacement, raising=False)
        first = native._compile_source(source, "compiler.py")
        second = native._compile_source(source, "compiler.py")
    assert replacements == [source, source]
    assert first != second
    assert first != original
    assert len(native._COMPILE_CACHE) == 1
    assert native._compile_source(source, "compiler.py") is original


@pytest.mark.parametrize(
    "options", [{"dont_inherit": False}, {"flags": ast.PyCF_ONLY_AST}]
)
def test_context_dependent_or_mutable_compiler_outputs_are_not_retained(options):
    with _observed_calls() as calls:
        first = native._compile_source(b"value = 1", "bypass.py", **options)
        second = native._compile_source(b"value = 1", "bypass.py", **options)
    assert first is not second
    assert calls["compile"] == 2
    assert not native._COMPILE_CACHE


def test_failures_are_not_memoized():
    with _observed_calls() as calls:
        for _ in range(2):
            with pytest.raises(SyntaxError):
                native._compile_source(b"def invalid(:", "repair.py")
        assert not native._COMPILE_CACHE
        fixed = native._compile_source(b"value = 1", "repair.py")
    assert calls["compile"] == 3
    assert type(fixed) is CodeType
    assert len(native._COMPILE_CACHE) == 1


def test_fifo_entry_limit_and_clear(monkeypatch):
    monkeypatch.setattr(native, "_COMPILE_CACHE_MAX_ENTRIES", 2)
    source = b"value = 1"
    first = native._compile_source(source, "first.py")
    second = native._compile_source(source, "second.py")
    assert native._compile_source(source, "first.py") is first
    native._compile_source(source, "third.py")
    assert len(native._COMPILE_CACHE) == 2
    assert native._compile_source(source, "second.py") is second
    with _observed_calls() as calls:
        assert native._compile_source(source, "first.py") is not first
    assert calls["compile"] == 1
    native._clear_compile_cache()
    assert not native._COMPILE_CACHE
    with _observed_calls() as calls:
        native._compile_source(source, "first.py")
    assert calls["compile"] == 1


def test_source_byte_budget_counts_distinct_filename_keys(monkeypatch):
    source = b"value = 1"
    monkeypatch.setattr(native, "_COMPILE_CACHE_MAX_SOURCE_BYTES", len(source) * 2)
    first = native._compile_source(source, "first.py")
    native._compile_source(source, "second.py")
    native._compile_source(source, "third.py")
    assert sum(len(key[0]) for key in native._COMPILE_CACHE) == len(source) * 2
    with _observed_calls() as calls:
        assert native._compile_source(source, "first.py") is not first
    assert calls["compile"] == 1


@pytest.mark.parametrize(
    "limit_name", ["_COMPILE_CACHE_MAX_ENTRY_BYTES", "_COMPILE_CACHE_MAX_SOURCE_BYTES"]
)
def test_oversize_source_is_compiled_without_cache(limit_name, monkeypatch):
    source = b"value = 1"
    monkeypatch.setattr(native, limit_name, len(source) - 1)
    with _observed_calls() as calls:
        first = native._compile_source(source, "oversize.py")
        second = native._compile_source(source, "oversize.py")
    assert first == second
    assert first is not second
    assert calls["compile"] == 2
    assert not native._COMPILE_CACHE


def test_live_check_reads_source_twice_even_after_warming(tmp_path, monkeypatch):
    module, _ = _module(tmp_path, monkeypatch, b"def value():\n    return 1\n")
    with _observed_calls() as cold:
        first_index = {}
        native._live_code(module, first_index)
    with _observed_calls() as warm:
        second_index = {}
        native._live_code(module, second_index)
    assert cold == {"compile": 1, "read": 2}
    assert warm == {"compile": 0, "read": 2}
    assert first_index == second_index
    assert first_index is not second_index
    first_index[module.__file__]["value"] = None
    native._live_code(module, {})
    assert second_index[module.__file__]["value"] is not None


def test_real_producer_keeps_read_counts_and_full_evidence_on_warm_cache():
    # Settle unrelated import/manifest memoization, then compare only compile
    # cache cold versus warm with the real producer and unchanged source files.
    native._producer()
    native._clear_compile_cache()
    with _observed_calls() as cold:
        first = native._producer()
    with _observed_calls() as warm:
        second = native._producer()
    assert cold["compile"] > 0
    assert warm["compile"] == 0
    assert cold["read"] > 0
    assert warm["read"] == cold["read"]
    assert first == second
    assert len(native._COMPILE_CACHE) <= native._COMPILE_CACHE_MAX_ENTRIES
    assert (
        sum(len(key[0]) for key in native._COMPILE_CACHE)
        <= native._COMPILE_CACHE_MAX_SOURCE_BYTES
    )


@pytest.mark.parametrize("drift", ["code", "globals", "declaration"])
def test_warm_cache_does_not_hide_loaded_function_drift(drift, tmp_path, monkeypatch):
    module, _ = _module(tmp_path, monkeypatch, b"def value():\n    return 1\n")
    native._live_code(module, {})
    if drift == "code":
        module.value.__code__ = module.value.__code__.replace(co_consts=(None, 2))
    elif drift == "globals":
        module.value = FunctionType(module.value.__code__, dict(vars(module)))
    else:
        del module.value
    with pytest.raises(native.ACSNativeCoverageBindingError, match="^LOADED_PRODUCER$"):
        native._live_code(module, {})


def test_warm_cache_preserves_imported_alias_identity(tmp_path, monkeypatch):
    origin, _ = _module(
        tmp_path,
        monkeypatch,
        b"def value():\n    return 1\n",
        name="invented_acs_origin",
    )
    consumer, _ = _module(
        tmp_path,
        monkeypatch,
        b"from invented_acs_origin import value\n",
        name="invented_acs_consumer",
    )
    native._live_code(consumer, {})
    consumer.value = FunctionType(origin.value.__code__, vars(origin))
    assert consumer.value.__code__ is origin.value.__code__
    with pytest.raises(native.ACSNativeCoverageBindingError, match="^LOADED_PRODUCER$"):
        native._live_code(consumer, {})


def test_warm_cache_checks_distinct_functions_with_shared_code_and_closures(
    tmp_path, monkeypatch
):
    source = b"""def make():
    def target():
        return 1
    def entry():
        return target()
    return entry
first = make()
second = make()
"""
    module, _ = _module(tmp_path, monkeypatch, source)
    native._live_code(module, {})
    assert module.first.__code__ is module.second.__code__
    first_target = module.first.__closure__[0].cell_contents
    second_target = module.second.__closure__[0].cell_contents
    assert first_target is not second_target
    second_target.__code__ = second_target.__code__.replace(co_consts=(None, 2))
    with pytest.raises(native.ACSNativeCoverageBindingError, match="^LOADED_PRODUCER$"):
        native._live_code(module, {})


def test_second_source_read_can_invalidate_a_warm_entry(tmp_path, monkeypatch):
    original = b"def value():\n    return 1\n"
    changed = b"def value():\n    return 2\n"
    module, path = _module(tmp_path, monkeypatch, original)
    native._live_code(module, {})
    read_bytes = Path.read_bytes
    calls = 0

    def change_on_second_read(self):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 2:
                path.write_bytes(changed)
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", change_on_second_read)
    with pytest.raises(native.ACSNativeCoverageBindingError, match="^LOADED_PRODUCER$"):
        native._live_code(module, {})
    assert calls == 2


def test_malformed_cache_entry_is_recompiled_and_replaced():
    source = b"value = 1"
    first = native._compile_source(source, "malformed.py")
    key = next(iter(native._COMPILE_CACHE))
    native._COMPILE_CACHE[key] = None
    with _observed_calls() as calls:
        repaired = native._compile_source(source, "malformed.py")
        again = native._compile_source(source, "malformed.py")
    assert repaired == first
    assert repaired is not first
    assert again is repaired
    assert calls["compile"] == 1


def test_nested_compilation_keeps_current_fifo_bound(monkeypatch):
    monkeypatch.setattr(native, "_COMPILE_CACHE_MAX_ENTRIES", 1)
    assert sys.getprofile() is None
    nested = []
    helper_code = native._compile_source.__code__

    def after_compile(frame, event, arg):
        if (
            not nested
            and event == "c_return"
            and frame.f_code is helper_code
            and arg is builtins.compile
        ):
            nested.append(True)
            native._compile_source(b"nested = 2", "nested.py")

    sys.setprofile(after_compile)
    try:
        outer = native._compile_source(b"outer = 1", "outer.py")
    finally:
        sys.setprofile(None)
    assert nested == [True]
    assert outer == builtins.compile(
        b"outer = 1", "outer.py", "exec", dont_inherit=True
    )
    assert len(native._COMPILE_CACHE) == 1
    assert next(iter(native._COMPILE_CACHE))[1] == "outer.py"


def test_swapped_valid_entries_are_recompiled_for_their_complete_inputs():
    first_source, second_source = b"value = 1", b"value = 2"
    first = native._compile_source(first_source, "first.py")
    second = native._compile_source(second_source, "second.py")
    first_key, second_key = tuple(native._COMPILE_CACHE)
    native._COMPILE_CACHE[first_key], native._COMPILE_CACHE[second_key] = (
        native._COMPILE_CACHE[second_key],
        native._COMPILE_CACHE[first_key],
    )
    with _observed_calls() as calls:
        repaired_first = native._compile_source(first_source, "first.py")
        repaired_second = native._compile_source(second_source, "second.py")
        assert native._compile_source(first_source, "first.py") is repaired_first
        assert native._compile_source(second_source, "second.py") is repaired_second
    assert calls["compile"] == 2
    assert repaired_first == first and repaired_first is not first
    assert repaired_second == second and repaired_second is not second
    assert repaired_first.co_filename == "first.py"
    assert repaired_second.co_filename == "second.py"


def test_python_compiler_captured_as_initial_compiler_still_bypasses(monkeypatch):
    # This is the exact captured/current state after a pre-import replacement;
    # leave all issued-owner modules loaded and exercise the real cache helper.
    calls = []

    def replacement(source, filename, mode, **options):
        calls.append(source)
        return builtins.compile(
            f"value = {len(calls)}".encode(), filename, mode, **options
        )

    # Textual labels alone must not classify a Python replacement as builtin.
    replacement.__module__ = "builtins"
    replacement.__name__ = "compile"
    replacement.__self__ = builtins
    monkeypatch.setattr(native, "_COMPILE_CACHE_COMPILER", replacement)
    monkeypatch.setattr(native, "compile", replacement, raising=False)
    first = native._compile_source(b"value = 0", "captured.py")
    second = native._compile_source(b"value = 0", "captured.py")
    assert calls == [b"value = 0", b"value = 0"]
    assert first != second
    assert not native._COMPILE_CACHE
