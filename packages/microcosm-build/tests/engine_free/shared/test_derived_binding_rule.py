"""Controls for the derived binding rule's walk (microcosm#931).

``test_support.microcosm_build.derived_binding_rule`` holds the rule's
country-neutral machinery: each country declares its tables in a
``RuleTables`` and walks through a ``BindingRule`` built from them, as
test_uk_graph.py does for the UK. These controls pin, on scratch packages,
each way the walk follows what a root names, the kernel rule's choice of what
a kernel runs, the engine root, and the refusal check, each control with a
rule of its own tables; and that the packages' lazy exports are the ones the
walk reads.
"""

from __future__ import annotations

import ast
import importlib
import uuid
from pathlib import Path

import microcosm
from test_support.microcosm_build.derived_binding_rule import (
    FIRST_PARTY_SCOPE,
    LAZY_EXPORT_TABLES,
    NO_LAZY_EXPORTS,
    BindingRule,
    RuleTables,
    hashed_modules,
    module_definitions,
    writes_nothing,
)


def test_the_refusal_check_sees_each_way_a_function_writes() -> None:
    """Control for ``writes_nothing``: a function that only refuses, or
    returns its sole argument, passes; one that writes through any of these
    shapes, or returns one of several arguments, does not."""
    refusing = {
        "raise": "def f(frame):\n    if frame is None:\n        raise ValueError('x')\n",
        "identity": (
            "def f(entity):\n"
            "    if entity not in {'person'}:\n"
            "        raise ValueError('x')\n"
            "    return entity\n"
        ),
    }
    writing = {
        "copyto": "def f(values, mask):\n    np.copyto(values, 0.0, where=mask)\n",
        "out": "def f(values):\n    np.clip(values, 0, 1, out=values)\n",
        "add_at": "def f(w, roots, copies):\n    np.add.at(w, roots, w[copies])\n",
        "shuffle": "def f(rng, values):\n    rng.shuffle(values)\n",
        "reverse": "def f(items):\n    items.reverse()\n",
        "augmented": "def f(values):\n    values *= 2.0\n",
        "subscript": "def f(values):\n    values[0] = 1.0\n",
        "inplace": "def f(table):\n    table.drop(columns=['a'], inplace=True)\n",
        "pick": "def f(left, right):\n    return left\n",
        "computed": "def f(values):\n    return values + 1\n",
    }
    for label, source in {**refusing, **writing}.items():
        function = ast.parse(source).body[0]
        assert writes_nothing(function) is (label in refusing), label


def test_the_helper_rule_follows_each_way_a_stage_names_a_helper(
    tmp_path, monkeypatch
) -> None:
    """Control: a helper called by name, through a re-export, a lazy export,
    by attribute on its module, through an alias or a computed constant, or
    held only in an annotation (a record's field type included) is reached,
    as are a numeric constant, a collection of strings and a function that
    writes in place or only refuses but is not listed; a listed check, a
    single string, a column name, a record of plain fields, a listed core
    module, what only a core module names, and a helper the root never names
    are not. The walk returns the names it stopped at and the core modules it
    met, and follows a core module only when asked to."""
    scratch = f"helper_rule_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    # A package exporting its runtime module's names lazily, as
    # microcosm.graph does.
    (package / "lazily").mkdir()
    (package / "lazily" / "__init__.py").write_text(
        "from importlib import import_module\n\n"
        "_EXPORTS = {'deferred': 'runtime', 'idle': 'runtime'}\n\n"
        "def __getattr__(name):\n"
        "    if name not in _EXPORTS:\n"
        "        raise AttributeError(name)\n"
        "    return getattr(import_module(f'{__name__}.{_EXPORTS[name]}'), name)\n"
    )
    (package / "lazily" / "runtime.py").write_text(
        "def deferred(x):\n    return x * 6\n\ndef idle(x):\n    return x * 7\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "direct": "def draw(x):\n    return x + 1\n",
        "origin": "def total(x):\n    return sum(x)\n",
        "reexport": "from .origin import total\n",
        "attribute": "def share(x):\n    return x / 2\n",
        "alias": "def _scale(x):\n    return x * 3\n\nscale = _scale\n",
        "computed": "def _build():\n    return {'a': 1}\n\nTABLE = _build()\n",
        "collaborator": (
            "class Contract:\n    def rate(self, key):\n        return 0.5\n"
        ),
        "fielded": (
            "from typing import NamedTuple\n\n"
            "class Rate:\n    def value(self):\n        return 0.1\n\n"
            "class Inputs(NamedTuple):\n    rate: Rate\n"
        ),
        "numeric": "FLOOR = 10\n",
        "codes": (
            "CODES = frozenset({'A', 'B'})\n"
            "OWNER = {'OWNED': 'owner', 'RENTED': 'renter'}\n"
        ),
        "inplace": (
            "import numpy as np\n\n"
            "def fold(weights, roots, copies):\n"
            "    np.add.at(weights, roots, weights[copies])\n"
        ),
        "guard": (
            "def guard(frame):\n"
            "    if frame is None:\n"
            "        raise ValueError('no frame')\n"
        ),
        "harmless": (
            "COLUMN = 'household_weight'\n\n"
            "def check(frame):\n"
            "    if frame is None:\n"
            "        raise ValueError('no frame')\n\n"
            "def column(entity):\n"
            "    return f'{entity}_support_channel'\n\n"
            "class Record:\n"
            "    value: float\n"
        ),
        "unused": "def never(x):\n    return x - 1\n",
        "behind": "def kept(x):\n    return x * 4\n",
        "core": (
            "from .behind import kept\n\n"
            "class Container:\n"
            "    def grow(self, x):\n"
            "        return kept(x)\n"
        ),
        "stage": (
            "from . import attribute\n"
            "from .alias import scale\n"
            "from .codes import CODES, OWNER\n"
            "from .collaborator import Contract\n"
            "from .computed import TABLE\n"
            "from .core import Container\n"
            "from .direct import draw\n"
            "from .fielded import Inputs\n"
            "from .guard import guard\n"
            "from .harmless import COLUMN, Record, check, column\n"
            "from .inplace import fold\n"
            "from .lazily import deferred\n"
            "from .numeric import FLOOR\n"
            "from .reexport import total\n"
            "from .unused import never\n\n"
            "class Transform:\n"
            "    contract: Contract\n"
            "    inputs: Inputs\n\n"
            "    def __call__(self, frame):\n"
            "        check(frame)\n"
            "        guard(frame)\n"
            "        fold(frame, [0], [1])\n"
            "        record = Record(value=draw(1) + total([1]) + deferred(1))\n"
            "        rate = self.contract.rate(column('household'))\n"
            "        rate += self.inputs.rate.value()\n"
            "        owners = [OWNER[code] for code in CODES]\n"
            "        rate += Container().grow(rate)\n"
            "        return attribute.share(scale(rate)) + TABLE['a'] + FLOOR, (\n"
            "            record, COLUMN, owners\n"
            "        )\n\n"
            "def helper():\n"
            "    return never(1)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    rule = BindingRule(
        RuleTables(
            core_modules={f"{prefix}core": "control"},
            refusal_only={(f"{prefix}harmless", "check"): "control"},
            lazy_export_tables={**LAZY_EXPORT_TABLES, f"{prefix}lazily": "_EXPORTS"},
        )
    )
    reach, stops, core = rule.helper_walk((f"{prefix}stage", "Transform"), scope=prefix)
    assert {module.removeprefix(prefix): names for module, names in reach.items()} == {
        "stage": {"Transform"},
        "direct": {"draw"},
        "origin": {"total"},
        "lazily.runtime": {"deferred"},
        "attribute": {"share"},
        "alias": {"scale", "_scale"},
        "computed": {"TABLE", "_build"},
        "collaborator": {"Contract"},
        "fielded": {"Inputs", "Rate"},
        "numeric": {"FLOOR"},
        "codes": {"CODES", "OWNER"},
        "inplace": {"fold"},
        "guard": {"guard"},
    }
    assert {(module.removeprefix(prefix), name) for module, name in stops} == {
        ("harmless", "COLUMN"),
        ("harmless", "Record"),
        ("harmless", "column"),
    }
    assert core == {f"{prefix}core"}
    followed = rule.helper_walk(
        (f"{prefix}stage", "Transform"), scope=prefix, follow_core=True
    )
    assert set(followed.reach) - set(reach) == {f"{prefix}core", f"{prefix}behind"}
    assert followed.core == core


def test_the_kernel_rule_follows_what_a_kernel_runs(tmp_path, monkeypatch) -> None:
    """Control: the kernel rule follows what a kernel's methods name,
    through its bases by method resolution order (a base method it inherits,
    and one it overrides only where it calls ``super()``), and leaves out its
    implementation hash, a method only the hash calls, a listed refusal-only
    method, a base method it overrides without ``super()`` and what lies
    behind a listed kernel stop, which it reports with what named it. A
    method the hash and the run both call stays. The recorder sees each
    module the hash passes to ``source_hash``, in whichever module calls
    it."""
    scratch = f"kernel_rule_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "computed": "def computed(x):\n    return x + 1\n",
        "inherited": "def base_only(x):\n    return x * 2\n",
        "assisted": "def kept(x):\n    return x * 3\n",
        "overridden": "def replaced(x):\n    return x - 2\n",
        "refused": "def check(x):\n    return x < 0\n",
        "keyonly": "def salt():\n    return 'salt'\n",
        "behind": "def beyond(x):\n    return x * 5\n",
        "resourced": "def fingerprint(x):\n    return x * 7\n",
        "shared": "def both(x):\n    return x * 11\n",
        "table": "from .behind import beyond\n\nTABLE = {'a': beyond}\n",
        "keyed": (
            "from microcosm.graph import source_hash\n\n"
            "from . import computed, keyonly\n\n"
            "def key():\n    return source_hash(computed, keyonly)\n"
        ),
        "base": (
            "from .assisted import kept\n"
            "from .inherited import base_only\n"
            "from .overridden import replaced\n\n"
            "class Base:\n"
            "    def run(self, x):\n"
            "        return base_only(x)\n\n"
            "    def prepare(self, x):\n"
            "        return replaced(x)\n\n"
            "    def assist(self, x):\n"
            "        return kept(x)\n"
        ),
        "kernel": (
            "from .base import Base\n"
            "from .computed import computed\n"
            "from .keyed import key\n"
            "from .keyonly import salt\n"
            "from .refused import check\n"
            "from .resourced import fingerprint\n"
            "from .shared import both\n"
            "from .table import TABLE\n\n"
            "class Kernel(Base):\n"
            "    def implementation_hash(self):\n"
            "        return salt() + key() + self._resources(1) + self._both(1)\n\n"
            "    def _resources(self, x):\n"
            "        return str(fingerprint(x))\n\n"
            "    def _both(self, x):\n"
            "        return str(both(x))\n\n"
            "    def _admit(self, x):\n"
            "        if check(x):\n"
            "            raise ValueError('refused')\n\n"
            "    def prepare(self, x):\n"
            "        return x\n\n"
            "    def run(self, x):\n"
            "        self._admit(x)\n"
            "        self._both(x)\n"
            "        return computed(x) + super().run(x) + TABLE['a'](x)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    rule = BindingRule(
        RuleTables(
            core_modules={},
            refusal_only={(f"{prefix}kernel", "Kernel._admit"): "control"},
            stops={(f"{prefix}table", "TABLE"): "control"},
        )
    )
    kernel_class = importlib.import_module(f"{prefix}kernel").Kernel
    walk = rule.kernel_walk(kernel_class, scope=prefix)
    assert {
        module.removeprefix(prefix): names for module, names in walk.reach.items()
    } == {
        "kernel": {"Kernel"},
        "base": {"Base"},
        "computed": {"computed"},
        "inherited": {"base_only"},
        "assisted": {"kept"},
        "shared": {"both"},
    }
    assert walk.met == {
        (f"{prefix}table", "TABLE"): frozenset({(f"{prefix}kernel", "Kernel.run")})
    }
    assert hashed_modules(kernel_class()) == {f"{prefix}computed", f"{prefix}keyonly"}


def test_the_engine_root_follows_the_methods_a_stage_calls(
    tmp_path, monkeypatch
) -> None:
    """Control for the engine root: the methods a stage names on an engine,
    by attribute on any receiver or through ``getattr``, are followed, with
    the constructor and the methods they call on ``self``; a method no stage
    names is not, nor what only it computes through. A method a helper the
    engine hands itself to calls on that argument is followed too, to a fixed
    point (``engine_reach``), which one pass from the stage's methods
    misses."""
    scratch = f"engine_root_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "microcosm" / scratch
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sources = {
        "tables": "def tables(frame):\n    return frame\n",
        "tree": "def tree(year):\n    return year\n",
        "vocabulary": "def vocabulary():\n    return {'a': 1}\n",
        "count": "def count(x):\n    return int(x)\n",
        "engine": (
            "from .count import count\n"
            "from .tables import tables\n"
            "from .tree import tree\n"
            "from .vocabulary import vocabulary\n\n"
            "class Engine:\n"
            "    def __init__(self, clones=1):\n"
            "        self._clones = count(clones)\n\n"
            "    def materialize(self, frame, names, period):\n"
            "        return self._dataset(frame)\n\n"
            "    def _dataset(self, frame):\n"
            "        return tables(frame)\n\n"
            "    def parameters(self, period):\n"
            "        return tree(period)\n\n"
            "    def concept_mapping(self):\n"
            "        return vocabulary()\n"
        ),
        "stage": (
            "class Transform:\n"
            "    def __call__(self, frame, model):\n"
            "        reader = getattr(model, 'parameters', None)\n"
            "        return model.materialize(frame, ['x'], 2024), reader\n"
        ),
        "delegating": (
            "from .engine import Engine\n\n"
            "def _vocabulary_of(engine):\n"
            "    return engine.concept_mapping()\n\n"
            "class Delegating(Engine):\n"
            "    def materialize(self, frame, names, period):\n"
            "        return _vocabulary_of(self)\n"
        ),
    }
    for module, source in sources.items():
        (package / f"{module}.py").write_text(source)
    prefix = f"microcosm.{scratch}."
    rule = BindingRule(RuleTables(core_modules={}))

    def short(reach):
        return {module.removeprefix(prefix): names for module, names in reach.items()}

    engine_class = importlib.import_module(f"{prefix}engine").Engine
    walk = rule.helper_walk((f"{prefix}stage", "Transform"), scope=prefix)
    methods = rule.engine_methods(engine_class, walk)
    assert methods == {"materialize", "parameters"}
    reach = rule.engine_walk(engine_class, methods, scope=prefix).reach
    assert short(reach) == {
        "engine": {"Engine"},
        "count": {"count"},
        "tables": {"tables"},
        "tree": {"tree"},
    }
    assert rule.engine_reach(engine_class, walk, scope=prefix).reach == reach
    delegating = importlib.import_module(f"{prefix}delegating").Delegating
    once = rule.engine_walk(
        delegating, rule.engine_methods(delegating, walk), scope=prefix
    ).reach
    assert "vocabulary" not in short(once)
    assert short(rule.engine_reach(delegating, walk, scope=prefix).reach) == {
        "delegating": {"Delegating", "_vocabulary_of"},
        "engine": {"Engine"},
        "count": {"count"},
        "tree": {"tree"},
        "vocabulary": {"vocabulary"},
    }


def test_every_lazy_export_resolves_to_the_object_python_returns() -> None:
    """The walk follows a name a package exports lazily, through its module
    ``__getattr__``, to the module that defines it, as Python imports it
    (microcosm#931 ruling 8). Every name microcosm.graph exports that way
    resolves to a definition holding the very object the package returns,
    the table the walk reads is the one the package's ``__getattr__`` reads,
    and a name the package imports eagerly resolves as before. Every
    first-party module that defines a module ``__getattr__`` is listed, as a
    lazy exporter (``LAZY_EXPORT_TABLES``) or as one that exports nothing
    through it (``NO_LAZY_EXPORTS``), so a new one cannot hide names from the
    walk."""
    from microcosm import graph

    rule = BindingRule(RuleTables(core_modules={}))
    for module, table in LAZY_EXPORT_TABLES.items():
        getter = module_definitions(module, FIRST_PARTY_SCOPE).definitions[
            "__getattr__"
        ]
        assert table in {
            sub.id for sub in ast.walk(getter) if isinstance(sub, ast.Name)
        }, module
    exports = getattr(graph, LAZY_EXPORT_TABLES[graph.__name__])
    assert {"load_source", "KernelContext", "keyed_uniform"} <= set(exports)
    for name in exports:
        found = rule.resolve(graph.__name__, name)
        assert found is not None, name
        module, defined = found
        assert getattr(importlib.import_module(module), defined) is getattr(
            graph, name
        ), name
    assert rule.resolve(graph.__name__, "compile_graph") == (
        "microcosm.graph.decl",
        "compile_graph",
    )
    getters = set()
    for root in map(Path, microcosm.__path__):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "def __getattr__" not in text or not any(
                isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
                for node in ast.parse(text).body
            ):
                continue
            parts = path.relative_to(root.parent).with_suffix("").parts
            getters.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    listed = set(LAZY_EXPORT_TABLES) | set(NO_LAZY_EXPORTS)
    assert getters == listed, sorted(getters ^ listed)
    # The modules that export nothing that way only raise from __getattr__.
    for module in NO_LAZY_EXPORTS:
        getter = module_definitions(module, FIRST_PARTY_SCOPE).definitions[
            "__getattr__"
        ]
        assert not any(isinstance(sub, ast.Return) for sub in ast.walk(getter))
