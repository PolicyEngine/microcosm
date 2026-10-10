"""``binding_identity.describe``: the identity of a gate binding's behaviour.

Invariants (Hypothesis property tests):

- determinism: describing a binding twice gives the same canonical-JSON value;
- exactness: two field values get equal descriptions exactly when an
  independent oracle says they are the same data (type, order, content: a
  tuple is not a list, mapping order counts, booleans, integers and floats
  stay distinct, sets are unordered, bytes compare by content).

Example tests:

- every shape outside the closed vocabulary is refused: closures, partials,
  bound methods, lambdas, nested functions, decorated wrappers, builtins,
  numpy scalars, cycles, non-string mapping keys, mutable or non-dataclass
  bindings, and state kept outside a dataclass's fields;
- the hash of ``gates.battery@1`` is the same in processes with different
  string-hash seeds, so set ordering never leaks into a node key.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.gate_battery import FunctionBinding
from microcosm.build.transport.binding_identity import (
    UndescribableBindingError,
    describe,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import support_gate
from test_support.paths import paths_for


@dataclass(frozen=True)
class _Holder:
    name: str
    value: object


@dataclass
class _Mutable:
    name: str


@dataclass(frozen=True)
class _WithInitVar:
    name: str
    seed: dataclasses.InitVar[int] = 0

    def __post_init__(self, seed: int) -> None:
        object.__setattr__(self, "derived", seed * 2)


class _Plain:
    name = "plain"


class _SlotBase:
    __slots__ = ("hidden",)


@dataclass(frozen=True, slots=True)
class _Slotted(_SlotBase):
    name: str


@dataclass(frozen=True)
class _DictBinding(dict):
    name: str


@dataclass(frozen=True)
class _ListBinding(list):
    name: str


@dataclass(frozen=True)
class _SetBinding(set):
    name: str


_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=6),
    st.binary(max_size=6),
)
_hashable = st.one_of(st.integers(-50, 50), st.text(max_size=4))
_values = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.lists(children, max_size=4).map(tuple),
        st.dictionaries(st.text(max_size=4), children, max_size=4),
        st.frozensets(_hashable, max_size=4),
    ),
    max_leaves=12,
)


def _oracle(value: object) -> object:
    """Independent typed encoding of the data a description must keep."""

    if value is None:
        return ["none"]
    for kind in (bool, int, str):
        if type(value) is kind:
            return [kind.__name__, value]
    if type(value) is float:
        return ["float", math.copysign(1.0, value), abs(value)]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_oracle(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[key, _oracle(item)] for key, item in value.items()]]
    if type(value) is frozenset:
        return ["frozenset", sorted(json.dumps(_oracle(item)) for item in value)]
    raise TypeError(type(value))


def _description(value: object) -> bytes:
    description, _ = describe(_Holder("holder", value))
    return canonical_json(description)


@settings(max_examples=300, deadline=None)
@given(_values)
def test_property_description_is_deterministic(value) -> None:
    assert _description(value) == _description(value)


@settings(max_examples=400, deadline=None)
@given(_values, _values)
def test_property_equal_descriptions_iff_equal_data(first, second) -> None:
    same = json.dumps(_oracle(first)) == json.dumps(_oracle(second))
    assert (_description(first) == _description(second)) is same


def test_types_order_and_signs_are_kept() -> None:
    assert _description(0.0) != _description(-0.0)
    assert _description(1) != _description(1.0)
    assert _description(True) != _description(1)
    assert _description((1, 2)) != _description([1, 2])
    assert _description({"a": 1, "b": 2}) != _description({"b": 2, "a": 1})
    assert _description(frozenset({1, 2})) == _description(frozenset({2, 1}))
    assert _description({"a": 1}) != _description(MappingProxyType({"a": 1}))


def test_a_function_binding_reaches_its_gate_module() -> None:
    binding = FunctionBinding(
        name="support",
        gate=support_gate,
        parameter_keys=frozenset({"min_persons"}),
        frame_argument="frame",
    )
    description, sources = describe(binding)
    names = set(sources)
    assert all(len(digest) == 64 for digest in sources.values())
    assert "test_support.microcosm_build.transport_graph" in names
    assert "microcosm.build.gate_battery" in names
    assert "transport_graph.support_gate" in canonical_json(description).decode()


def _closure_gate(floor):
    def gate(*, frame, min_persons):  # pragma: no cover - never evaluated
        return support_gate(frame=frame, min_persons=max(min_persons, floor))

    return gate


def _nested_gate():
    def gate(*, frame, min_persons):  # pragma: no cover - never evaluated
        return support_gate(frame=frame, min_persons=min_persons)

    return gate


@functools.wraps(support_gate)
def _wrapped_gate(**kwargs):  # pragma: no cover - never evaluated
    return support_gate(**kwargs)


_lambda_gate = lambda **kwargs: None  # noqa: E731 - the refused shape itself


class _Bound:
    def gate(self, *, frame, min_persons):  # pragma: no cover
        return support_gate(frame=frame, min_persons=min_persons)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (_closure_gate(3), "closes over"),
        (_nested_gate(), "module-level"),
        (_wrapped_gate, "decorated wrapper"),
        (_lambda_gate, "module-level"),
        (functools.partial(support_gate, min_persons=3), "vocabulary"),
        (_Bound().gate, "vocabulary"),
        (len, "vocabulary"),
        (np.float64(1.0), "vocabulary"),
        (np.int64(1), "vocabulary"),
        (float("inf"), "non-finite"),
        ({1: "x"}, "mapping keys"),
        (_Plain(), "vocabulary"),
        (_Mutable("m"), "frozen dataclass"),
        (_WithInitVar("w", 3), "outside its fields"),
        (_Slotted("s"), "outside its fields"),
    ],
)
def test_values_outside_the_vocabulary_are_refused(value, match) -> None:
    with pytest.raises(UndescribableBindingError, match=match):
        describe(_Holder("holder", value))


@pytest.mark.parametrize(
    ("kind", "mutate", "arguments"),
    [
        pytest.param(_DictBinding, dict.__setitem__, ("ok", True), id="dict"),
        pytest.param(_ListBinding, list.append, (True,), id="list"),
        pytest.param(_SetBinding, set.add, (True,), id="set"),
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["binding", "field"])
def test_frozen_dataclasses_with_builtin_container_state_are_refused(
    kind, mutate, arguments, nested
) -> None:
    value = kind("container")
    mutate(value, *arguments)
    assert len(value) == 1
    assert vars(value) == {"name": "container"}
    binding = _Holder("holder", value) if nested else value
    with pytest.raises(UndescribableBindingError, match="builtin container"):
        describe(binding)


def test_cycles_and_non_dataclass_bindings_are_refused() -> None:
    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(UndescribableBindingError, match="reference cycle"):
        describe(_Holder("holder", cyclic))
    with pytest.raises(UndescribableBindingError, match="frozen dataclass instance"):
        describe(_Plain())
    with pytest.raises(UndescribableBindingError, match="frozen dataclass"):
        describe(_Mutable("m"))


_PROBE = """
from microcosm.build.transport.gate_kernels import GateBatteryKernel
from test_support.microcosm_build.transport_graph import TOY_BINDINGS
print(GateBatteryKernel(TOY_BINDINGS).implementation_hash())
"""


def test_the_kernel_hash_is_stable_across_string_hash_seeds() -> None:
    root = paths_for("microcosm-build").repository
    # Prefer this worktree's sources to editable installs in another checkout.
    pythonpath = os.pathsep.join(
        [
            *(str(path) for path in sorted((root / "packages").glob("*/src"))),
            str(root),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    hashes = set()
    for seed in ("1", "31337"):
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=root,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": pythonpath},
            capture_output=True,
            text=True,
            check=True,
        )
        hashes.add(result.stdout.strip())
    assert len(hashes) == 1
    (value,) = hashes
    assert len(value) == len(hashlib.sha256().hexdigest())


_GATE_MODULE = """
from g5b_probe import helper_const
from g5b_probe.helper_const import LIMIT


def gate(*, frame, min_persons):  # pragma: no cover - never evaluated
    import g5b_probe.helper_local

    return g5b_probe.helper_local.limit() + LIMIT + helper_const.OTHER
"""


def test_the_source_closure_follows_every_import_statement(
    tmp_path, monkeypatch
) -> None:
    """Helpers reached by constant import, module import and local import."""

    import microcosm.build.transport.binding_identity as identity

    package = tmp_path / "g5b_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper_const.py").write_text("LIMIT = 1\nOTHER = 2\n")
    (package / "helper_local.py").write_text("def limit():\n    return 1\n")
    (package / "unrelated.py").write_text("X = 1\n")
    (package / "gates_mod.py").write_text(_GATE_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        identity, "FIRST_PARTY_PACKAGES", (*identity.FIRST_PARTY_PACKAGES, "g5b_probe")
    )
    import importlib

    module = importlib.import_module("g5b_probe.gates_mod")
    try:
        binding = FunctionBinding(name="support", gate=module.gate)
        _, sources = describe(binding)
        assert {
            "g5b_probe",
            "g5b_probe.gates_mod",
            "g5b_probe.helper_const",
            "g5b_probe.helper_local",
        } <= set(sources)
        assert "g5b_probe.unrelated" not in sources

        before = sources["g5b_probe.helper_local"]
        (package / "helper_local.py").write_text("def limit():\n    return 22\n")
        _, changed = describe(binding)
        assert changed["g5b_probe.helper_local"] != before
    finally:
        _forget("g5b_probe")


def _write_package(root, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _probe_package(tmp_path, monkeypatch, files, *, first_party=True):
    import importlib

    import microcosm.build.transport.binding_identity as identity

    _write_package(tmp_path, files)
    monkeypatch.syspath_prepend(str(tmp_path))
    if first_party:
        monkeypatch.setattr(
            identity,
            "FIRST_PARTY_PACKAGES",
            (*identity.FIRST_PARTY_PACKAGES, "g5b_rel"),
        )
    importlib.invalidate_caches()
    return importlib, identity


def _forget(prefix: str) -> None:
    import microcosm.build.transport.binding_identity as identity

    for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[name]
    for name in [n for n in identity._RESOLVED if n.startswith(prefix)]:
        del identity._RESOLVED[name]


_REL_FILES = {
    "g5b_rel/__init__.py": "from . import top_helper\n",
    "g5b_rel/top_helper.py": "VALUE = 1\n",
    "g5b_rel/unused.py": "VALUE = 9\n",
    "g5b_rel/sub/__init__.py": "from .. import sibling\nfrom . import inner\n",
    "g5b_rel/sibling.py": "VALUE = 2\n",
    "g5b_rel/sub/inner.py": "from ..deep.leaf import LEAF\n",
    "g5b_rel/deep/__init__.py": "",
    "g5b_rel/deep/leaf.py": "LEAF = 3\n",
    "g5b_rel/sub/gates_mod.py": (
        "def gate(*, frame, min_persons):  # pragma: no cover\n"
        "    from .inner import LEAF\n"
        "    return LEAF\n"
    ),
}


def test_relative_imports_and_parent_packages_resolve_exactly(
    tmp_path, monkeypatch
) -> None:
    importlib, _ = _probe_package(tmp_path, monkeypatch, _REL_FILES)
    try:
        module = importlib.import_module("g5b_rel.sub.gates_mod")
        _, sources = describe(FunctionBinding(name="support", gate=module.gate))
        names = {name for name in sources if name.startswith("g5b_rel")}
        assert names == {
            "g5b_rel",
            "g5b_rel.top_helper",
            "g5b_rel.sub",
            "g5b_rel.sibling",
            "g5b_rel.sub.inner",
            "g5b_rel.deep",
            "g5b_rel.deep.leaf",
            "g5b_rel.sub.gates_mod",
        }
    finally:
        _forget("g5b_rel")


def test_a_defining_module_outside_first_party_is_still_hashed(
    tmp_path, monkeypatch
) -> None:
    files = {
        "g5b_rel/__init__.py": "",
        "g5b_rel/gates_mod.py": (
            "def gate(*, frame, min_persons):  # pragma: no cover\n    return 1\n"
        ),
    }
    importlib, _ = _probe_package(tmp_path, monkeypatch, files, first_party=False)
    try:
        module = importlib.import_module("g5b_rel.gates_mod")
        binding = FunctionBinding(name="support", gate=module.gate)
        _, sources = describe(binding)
        assert "g5b_rel.gates_mod" in sources
        before = sources["g5b_rel.gates_mod"]
        (tmp_path / "g5b_rel" / "gates_mod.py").write_text(
            "def gate(*, frame, min_persons):  # pragma: no cover\n    return 22\n"
        )
        assert describe(binding)[1]["g5b_rel.gates_mod"] != before
    finally:
        _forget("g5b_rel")


def test_a_module_created_after_a_failed_lookup_is_found(tmp_path, monkeypatch) -> None:
    files = {
        "g5b_rel/__init__.py": "",
        "g5b_rel/gates_mod.py": (
            "def gate(*, frame, min_persons):  # pragma: no cover\n"
            "    import g5b_rel.later\n"
            "    return g5b_rel.later.VALUE\n"
        ),
    }
    importlib, _ = _probe_package(tmp_path, monkeypatch, files)
    try:
        module = importlib.import_module("g5b_rel.gates_mod")
        binding = FunctionBinding(name="support", gate=module.gate)
        assert "g5b_rel.later" not in describe(binding)[1]
        (tmp_path / "g5b_rel" / "later.py").write_text("VALUE = 1\n")
        importlib.invalidate_caches()
        assert "g5b_rel.later" in describe(binding)[1]
    finally:
        _forget("g5b_rel")


def test_the_closure_never_resolves_an_imported_name_by_case() -> None:
    """``from microcosm.fit import QRF`` must not resolve to ``qrf.py``."""

    from microcosm.build.gate_battery import DEFAULT_REGISTRY

    for binding in DEFAULT_REGISTRY.values():
        _, sources = describe(binding)
        assert not [name for name in sources if name.split(".")[-1][:1].isupper()]
