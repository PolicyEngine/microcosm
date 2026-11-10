"""Synthetic modules for the static walk's own tests.

Each root in :data:`ROOTS` reaches its target modules (``t_*``) in one of the
ways that :mod:`test_support.microcosm_build.static_reach` lists, and its walk
must reach exactly those targets. A target defines nothing but a module
``__getattr__`` that records each read of ``X`` or ``ROWS``. So the walk
reaches a target only through code it follows, and :func:`python_reads` can
run each root for real and report which targets Python read.

Python must read exactly what the walk reaches, apart from the documented
gaps: :data:`OVER` holds the targets a walk reaches by design and a run
doesn't read, and :data:`BLIND` holds the reads a run makes that the walk's
"What it can't see" list explains.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from test_support.microcosm_build.static_reach import Reach, module_paths

#: The package that holds the synthetic modules.
PACKAGE = "microcosm.walkcases"

TARGETS = (
    "add",
    "alias",
    "alias_base",
    "alias_cycle",
    "alias_diamond",
    "alias_modules",
    "alternative",
    "alternative_first",
    "attrgetter",
    "augassign",
    "base_attr",
    "base_dunder_import",
    "base_else",
    "base_first",
    "base_getattr",
    "base_if",
    "base_import",
    "base_or",
    "base_second",
    "base_starred",
    "base_subscript",
    "base_walrus",
    "block_body",
    "block_case",
    "block_else",
    "block_finally",
    "block_handler",
    "body_annotation",
    "body_before",
    "body_conditional",
    "body_decorator",
    "body_del",
    "body_hidden",
    "body_nested",
    "class_body_update",
    "class_fill_call",
    "class_fill_del",
    "class_fill_item",
    "class_fill_method",
    "class_fill_view",
    "class_global",
    "class_global_alias",
    "class_global_class",
    "class_global_def",
    "class_global_hidden",
    "class_global_import",
    "class_header",
    "class_import",
    "class_import_body",
    "class_import_decorator",
    "class_import_default",
    "class_member_change",
    "class_nested_fill",
    "class_scope",
    "comprehension",
    "comprehension_fill",
    "comprehension_iter",
    "conditional",
    "decorator",
    "default",
    "del_hidden",
    "del_item",
    "delattr",
    "delattr_patch",
    "dotted",
    "dunder_top",
    "factory",
    "fill_call",
    "fill_global",
    "float",
    "for_store",
    "for_target",
    "getattr_alias",
    "getattr_constant",
    "global",
    "go",
    "hasattr",
    "header_for",
    "header_for_target",
    "header_guard",
    "header_if",
    "header_match",
    "header_pattern",
    "header_try",
    "header_trystar",
    "header_while",
    "header_with",
    "header_with_target",
    "import_diamond",
    "import_from",
    "import_level",
    "import_module",
    "import_name",
    "import_relative",
    "import_renamed",
    "import_renamed_alias",
    "import_spec",
    "lambda",
    "lambda_shadow",
    "lazy",
    "lazy_assigned",
    "lazy_cycle",
    "local",
    "local_alternative",
    "local_alternative_first",
    "match_capture",
    "match_diamond",
    "match_inherited",
    "match_keyword",
    "match_positional",
    "match_rest",
    "match_star",
    "metaclass",
    "methodcaller",
    "mutation_alias",
    "namespace",
    "nested_scope",
    "nonlocal",
    "override",
    "prop",
    "registrar",
    "registry_decorator",
    "root_base",
    "self_method",
    "setattr",
    "setattr_patch",
    "setdefault",
    "setitem",
    "setup_call",
    "shadow",
    "shadowed",
    "sibling",
    "star",
    "star_clique",
    "star_diamond",
    "star_lazy",
    "store_hidden",
    "subclass",
    "subscript",
    "sunder",
    "unpack",
    "update",
    "value",
    "walrus",
    "with_store",
    "with_target",
)

#: Targets a case binds so that a wrong walk would reach them. No walk
#: reaches them and no run reads them: t_shadowed behind a name the class
#: body rebinds (r_class_decorator) and behind a comprehension's target
#: (r_comprehension_decoy), and t_body_hidden behind the function's import of
#: nest_p, which the class in r_body_nested never reads.
DECOYS = frozenset({"body_hidden", "shadowed"})

#: Each target's source: reads of X or ROWS are recorded, anything else fails.
_TARGET = """\
from walkcase_reads import read


def __getattr__(name):
    return read(__name__, name)
"""

#: The recorder, a top-level module outside ``microcosm``, so the walk never
#: parses it.
_READS = """\
READS = set()


def read(module, name):
    if name not in ("X", "ROWS"):
        raise AttributeError(name)
    READS.add(module.rpartition(".t_")[2])
    return 1 if name == "X" else {}
"""

#: Each module's source, by its dotted name inside :data:`PACKAGE`.
SOURCES: dict[str, str] = {
    "__init__": "",
    **{f"t_{name}": _TARGET for name in TARGETS},
    # -- Bodies -----------------------------------------------------------------
    # A default value, which ``def`` evaluates as it runs.
    "r_default": """
        from microcosm.walkcases import t_default


        def run(value=t_default.X):
            return value
        """,
    # A class keyword: the metaclass's __init__ runs as the class is made.
    "meta_mod": """
        from microcosm.walkcases import t_metaclass


        class Meta(type):
            def __init__(cls, name, bases, namespace):
                super().__init__(name, bases, namespace)
                assert t_metaclass.X
        """,
    "r_metaclass": """
        from microcosm.walkcases.meta_mod import Meta


        class Kernel(metaclass=Meta):
            def run(self):
                return 1
        """,
    # A method's decorator that the class body defines first.
    "r_decorator": """
        from microcosm.walkcases import t_decorator


        class Kernel:
            @staticmethod
            def _deco(function):
                assert t_decorator.X
                return function

            @_deco
            def run(self):
                return 1
        """,
    # -- Bindings ---------------------------------------------------------------
    # A star import binds the names its module defines.
    "star_src": """
        from microcosm.walkcases import t_star


        def starred():
            return t_star.X
        """,
    "r_star": """
        from microcosm.walkcases.star_src import *  # noqa: F403


        def run():
            return starred()  # noqa: F405
        """,
    # A name with two bindings reaches both, whichever runs: here the live
    # import is the second, and in r_alternative_first the first.
    "slow": """
        from microcosm.walkcases import t_alternative


        def impl():
            return t_alternative.X
        """,
    "r_alternative": """
        try:
            from microcosm.walkcases.absent import impl
        except ImportError:
            from microcosm.walkcases.slow import impl


        def run():
            return impl()
        """,
    "slow_first": """
        from microcosm.walkcases import t_alternative_first


        def impl():
            return t_alternative_first.X
        """,
    "fast": """
        def impl():
            return 0
        """,
    "r_alternative_first": """
        try:
            from microcosm.walkcases.slow_first import impl
        except ImportError:
            from microcosm.walkcases.fast import impl


        def run():
            return impl()
        """,
    # The same inside a function, for a module bound two ways.
    "alt_impl": """
        from microcosm.walkcases import t_local_alternative


        def impl():
            return t_local_alternative.X
        """,
    "r_local_alternative": """
        def run():
            try:
                from microcosm.walkcases import absent as backend
            except ImportError:
                from microcosm.walkcases import alt_impl as backend
            return backend.impl()
        """,
    "alt_first": """
        from microcosm.walkcases import t_local_alternative_first


        def impl():
            return t_local_alternative_first.X
        """,
    "r_local_alternative_first": """
        def run():
            try:
                from microcosm.walkcases import alt_first as backend
            except ImportError:
                from microcosm.walkcases import absent as backend
            return backend.impl()
        """,
    # A name that a for loop, a with statement, a match capture or a walrus
    # binds at module level, imported from that module. Nothing else leads
    # the walk to the module.
    "bind_for": """
        from microcosm.walkcases import t_for_target

        for VALUE in (t_for_target.X,):
            pass
        """,
    "r_bind_for": """
        from microcosm.walkcases.bind_for import VALUE


        def run():
            return VALUE
        """,
    "bind_with": """
        from contextlib import nullcontext

        from microcosm.walkcases import t_with_target

        with nullcontext(t_with_target.X) as VALUE:
            pass
        """,
    "r_bind_with": """
        from microcosm.walkcases.bind_with import VALUE


        def run():
            return VALUE
        """,
    "bind_match": """
        from microcosm.walkcases import t_match_capture

        match t_match_capture.X:
            case VALUE:
                pass
        """,
    "r_bind_match": """
        from microcosm.walkcases.bind_match import VALUE


        def run():
            return VALUE
        """,
    "bind_star": """
        from microcosm.walkcases import t_match_star

        match (t_match_star.X,):
            case [*VALUE]:
                pass
        """,
    "r_bind_star": """
        from microcosm.walkcases.bind_star import VALUE


        def run():
            return VALUE
        """,
    "bind_rest": """
        from microcosm.walkcases import t_match_rest

        match t_match_rest.ROWS:
            case {**VALUE}:
                pass
        """,
    "r_bind_rest": """
        from microcosm.walkcases.bind_rest import VALUE


        def run():
            return VALUE
        """,
    "bind_walrus": """
        from microcosm.walkcases import t_walrus

        _FIRST = (VALUE := t_walrus.X)
        """,
    "r_bind_walrus": """
        from microcosm.walkcases.bind_walrus import VALUE


        def run():
            return VALUE
        """,
    # A name a class body declares global binds at module level: through an
    # assignment, an alias of a module, an import, a def and a class there.
    # In _Setup, a class nothing reaches, the assignment reads a name the
    # class body bound.
    "cglob_alias": """
        from microcosm.walkcases import t_class_global_alias


        def fn():
            return t_class_global_alias.X
        """,
    "cglob_import": """
        from microcosm.walkcases import t_class_global_import


        def fn():
            return t_class_global_import.X
        """,
    "r_class_global": """
        from microcosm.walkcases import (
            cglob_alias,
            t_class_global,
            t_class_global_class,
            t_class_global_def,
            t_class_global_hidden,
        )


        class Kernel:
            global _CACHE, _IMPL, _probe, Probe, imported
            _CACHE = t_class_global.X
            _IMPL = cglob_alias
            import microcosm.walkcases.cglob_import as imported

            def _probe():
                return t_class_global_def.X

            class Probe:
                VALUE = t_class_global_class.X

            def run(self):
                found = _CACHE, _IMPL.fn(), imported.fn(), _probe()
                return found, Probe.VALUE, _ROWS


        class _Setup:
            global _ROWS
            _source = t_class_global_hidden
            _ROWS = _source.ROWS
        """,
    # A package's lazy export, through its module-level __getattr__.
    "lazy.__init__": """
        def __getattr__(name):
            from importlib import import_module

            return getattr(import_module(f"{__name__}.impl"), name)
        """,
    "lazy.impl": """
        from microcosm.walkcases import t_lazy


        class Lazy:
            def go(self):
                return t_lazy.X
        """,
    "r_lazy": """
        from microcosm.walkcases.lazy import Lazy


        def run():
            return Lazy().go()
        """,
    # ... with __getattr__ assigned rather than defined.
    "lazy2.__init__": """
        def _lazy(name):
            from importlib import import_module

            return getattr(import_module(f"{__name__}.impl"), name)


        __getattr__ = _lazy
        """,
    "lazy2.impl": """
        from microcosm.walkcases import t_lazy_assigned


        class Lazy2:
            def go(self):
                return t_lazy_assigned.X
        """,
    "r_lazy_assigned": """
        from microcosm.walkcases.lazy2 import Lazy2


        def run():
            return Lazy2().go()
        """,
    # ... and through a star import, which __all__ lets reach the lazy name.
    "lazy3.__init__": """
        __all__ = ["Lazy3"]


        def __getattr__(name):
            from importlib import import_module

            return getattr(import_module(f"{__name__}.impl"), name)
        """,
    "lazy3.impl": """
        from microcosm.walkcases import t_star_lazy


        class Lazy3:
            def go(self):
                return t_star_lazy.X
        """,
    "r_star_lazy": """
        from microcosm.walkcases.lazy3 import *  # noqa: F403


        def run():
            return Lazy3().go()  # noqa: F405
        """,
    # -- Scopes -----------------------------------------------------------------
    # A nested function's local leaves the enclosing scope's import alone.
    "r_shadow": """
        from microcosm.walkcases import t_shadow as target


        def run():
            def inner():
                target = 1
                return target

            return target.X, inner()
        """,
    # A comprehension's target binds only inside the comprehension.
    "r_comprehension": """
        from microcosm.walkcases import t_comprehension as target


        def run():
            values = [target for target in range(3)]
            return target.X, values
        """,
    # ... even where the module binds the same name to another module.
    "r_comprehension_decoy": """
        from microcosm.walkcases import t_comprehension_iter
        from microcosm.walkcases import t_shadowed as row


        def run():
            return [row.X for row in (t_comprehension_iter,)]
        """,
    # A local import binds only in its own function, not in a sibling.
    "r_sibling": """
        from microcosm.walkcases import t_sibling as target


        def run():
            def first():
                from microcosm.walkcases import t_local as target

                return target.X

            def second():
                return target.X

            return first(), second()
        """,
    # A class body's names stay out of its methods.
    "r_class_scope": """
        from microcosm.walkcases import t_class_scope as target


        class Holder:
            target = None

            def run(self):
                return target.X
        """,
    # ... and out of a lambda there.
    "r_lambda_shadow": """
        from microcosm.walkcases import t_lambda_shadow as target


        class Kernel:
            target = None
            pick = staticmethod(lambda: target.X)

            def run(self):
                return self.pick()
        """,
    # ... and a nested class's names out of its methods, which read the
    # enclosing function's binding instead.
    "nest_q": """
        from microcosm.walkcases import t_body_nested


        def fn():
            return t_body_nested.X
        """,
    "nest_p": """
        from microcosm.walkcases import t_body_hidden


        def fn():
            return t_body_hidden.X
        """,
    "nest_s": """
        from microcosm.walkcases import t_nested_scope


        def fn():
            return t_nested_scope.X
        """,
    "r_nested_scope": """
        from microcosm.walkcases import nest_q as target


        def run():
            from microcosm.walkcases import nest_s as target

            class Inner:
                target = None

                def get(self):
                    return target.fn()

            return Inner().get()
        """,
    # A method's decorator reads the class body's names first: the body has
    # bound target above it, so the module's import stays unreached.
    "r_class_decorator": """
        from microcosm.walkcases import t_shadowed as target


        class Kernel:
            target = staticmethod(lambda function: function)

            @target
            def run(self):
                return 1
        """,
    # Before the body binds a name, Python reads the module's binding: a read
    # above the binding, a binding under an if that doesn't run, a bare
    # annotation (which binds nothing), a decorator above the binding, and a
    # read after a del of the binding.
    "r_body_before": """
        from microcosm.walkcases import t_body_before as target


        class Kernel:
            seen = target.X
            target = None

            def run(self):
                return self.seen
        """,
    "r_body_conditional": """
        from microcosm.walkcases import t_body_conditional as target

        ON = False


        class Kernel:
            if ON:
                target = None
            seen = target.X

            def run(self):
                return self.seen
        """,
    "r_body_annotation": """
        from microcosm.walkcases import t_body_annotation as target


        class Kernel:
            target: object
            seen = target.X

            def run(self):
                return self.seen
        """,
    "deco_body": """
        from microcosm.walkcases import t_body_decorator


        def wrap(function):
            assert t_body_decorator.X
            return function
        """,
    "r_body_decorator": """
        from microcosm.walkcases.deco_body import wrap as target


        class Kernel:
            @target
            def run(self):
                return 1

            target = None
        """,
    "r_body_del": """
        from microcosm.walkcases import t_body_del as target


        class Kernel:
            target = None
            del target
            seen = target.X

            def run(self):
                return self.seen
        """,
    # A class inside a function reads such a name from the module too, never
    # from the function, so nest_p stays unreached.
    "r_body_nested": """
        from microcosm.walkcases import nest_q as target


        def run():
            from microcosm.walkcases import nest_p as target

            class Inner:
                seen = target.fn()
                target = None

            return Inner.seen
        """,
    # ``global`` sends a name to module level, past the function's own store.
    "r_global": """
        from microcosm.walkcases import t_global

        _CACHE = t_global.X


        def run():
            global _CACHE
            _CACHE = _CACHE or 1
            return _CACHE
        """,
    # ``nonlocal`` sends a name to the enclosing function's import.
    "nl_holder": """
        from microcosm.walkcases import t_nonlocal


        def get():
            return t_nonlocal.X
        """,
    "r_nonlocal": """
        def run():
            from microcosm.walkcases import nl_holder as holder

            def inner():
                nonlocal holder
                value = holder.get()
                holder = None
                return value

            return inner()
        """,
    # -- Aliases ----------------------------------------------------------------
    # A constant that holds a module stands for it, through tuple unpacking,
    # and through either branch of a condition. (A module used as a value
    # reaches these too: see the attribute fallback.)
    "helper": """
        from microcosm.walkcases import t_alias


        def compute():
            return t_alias.X
        """,
    "r_alias": """
        import microcosm.walkcases.helper as helper_module

        _helper = helper_module


        def run():
            return _helper.compute()
        """,
    "mod_a": """
        def fn():
            return 0
        """,
    "mod_b": """
        from microcosm.walkcases import t_unpack


        def fn():
            return t_unpack.X
        """,
    "mod_c": """
        from microcosm.walkcases import t_conditional


        def fn():
            return t_conditional.X
        """,
    "r_unpack": """
        from microcosm.walkcases import mod_a, mod_b

        _first, _second = mod_a, mod_b


        def run():
            return _second.fn()
        """,
    "r_conditional": """
        from microcosm.walkcases import mod_a, mod_c

        FLAG = True
        _impl = mod_c if FLAG else mod_a


        def run():
            return _impl.fn()
        """,
    # A class base held in a constant: run() is the base's. Only the alias
    # leads there, since nothing else names run.
    "made3": """
        from microcosm.walkcases import t_alias_base


        class Base:
            def run(self):
                return t_alias_base.X
        """,
    "r_alias_base": """
        from microcosm.walkcases import made3

        _Base = made3.Base


        class Kernel(_Base):
            pass
        """,
    # ... through each branch of a conditional (the walk follows both), each
    # operand of or, a walrus, each element of a tuple, a starred list,
    # getattr, a subscript, import_module, and __import__'s top package.
    "bases": """
        from microcosm.walkcases import (
            t_base_dunder_import,
            t_base_else,
            t_base_first,
            t_base_getattr,
            t_base_if,
            t_base_import,
            t_base_or,
            t_base_second,
            t_base_starred,
            t_base_subscript,
            t_base_walrus,
        )


        class IfBase:
            def run(self):
                return t_base_if.X


        class ElseBase:
            def run(self):
                return t_base_else.X


        class FirstBase:
            def run(self):
                return t_base_first.X


        class SecondBase:
            def run(self):
                return t_base_second.X


        class AttrBase:
            def run(self):
                return t_base_getattr.X


        class SubscriptBase:
            def __class_getitem__(cls, item):
                return cls

            def run(self):
                return t_base_subscript.X


        class ImportBase:
            def run(self):
                return t_base_import.X


        class OrBase:
            def run(self):
                return t_base_or.X


        class WalrusBase:
            def run(self):
                return t_base_walrus.X


        class StarredBase:
            def run(self):
                return t_base_starred.X


        class DunderImportBase:
            def run(self):
                return t_base_dunder_import.X
        """,
    "r_base_conditional": """
        from microcosm.walkcases import bases

        FLAG = False
        _Base = bases.IfBase if FLAG else bases.ElseBase


        class Kernel(_Base):
            pass
        """,
    "r_base_tuple": """
        from microcosm.walkcases import bases

        _First, _Second = bases.FirstBase, bases.SecondBase


        class Kernel(_Second):
            pass
        """,
    "r_base_getattr": """
        from microcosm.walkcases import bases

        _Base = getattr(bases, "AttrBase")


        class Kernel(_Base):
            pass
        """,
    "r_base_subscript": """
        from microcosm.walkcases import bases

        _Base = bases.SubscriptBase[int]


        class Kernel(_Base):
            pass
        """,
    "r_base_or": """
        from microcosm.walkcases import bases

        _Base = None or bases.OrBase


        class Kernel(_Base):
            pass
        """,
    "r_base_walrus": """
        from microcosm.walkcases import bases

        _Base = (_Found := bases.WalrusBase)


        class Kernel(_Base):
            pass
        """,
    "r_base_starred": """
        from microcosm.walkcases import bases

        _Base = [*[bases.StarredBase]][0]


        class Kernel(_Base):
            pass
        """,
    "r_base_dunder_import": """
        _top = __import__("microcosm.walkcases.bases")


        class Kernel(_top.walkcases.bases.DunderImportBase):
            pass
        """,
    "r_import_alias_base": """
        import importlib

        _impl = importlib.import_module("microcosm.walkcases.bases")


        class Kernel(_impl.ImportBase):
            pass
        """,
    # A module alias through getattr with a literal name.
    "gmod": """
        from microcosm.walkcases import t_getattr_alias


        def fn():
            return t_getattr_alias.X
        """,
    "r_getattr_alias": """
        import microcosm.walkcases.gmod  # noqa: F401
        from microcosm import walkcases

        _impl = getattr(walkcases, "gmod")


        def run():
            return _impl.fn()
        """,
    # -- Inheritance ------------------------------------------------------------
    # The base's run() calls the root class's override.
    "made6": """
        class Base:
            def run(self):
                return self.step()

            def step(self):
                return 0
        """,
    "r_override": """
        from microcosm.walkcases import t_override
        from microcosm.walkcases.made6 import Base


        class Kernel(Base):
            def step(self):
                return t_override.X
        """,
    # A subscripted base in the class statement itself.
    "made9": """
        from microcosm.walkcases import t_subscript


        class Base:
            def __class_getitem__(cls, item):
                return cls

            def run(self):
                return t_subscript.X
        """,
    "r_subscript": """
        from microcosm.walkcases.made9 import Base


        class Kernel(Base[int]):
            pass
        """,
    # -- Self and cls -----------------------------------------------------------
    # A method called through self, which the class-attribute fallback
    # reaches.
    "r_self_method": """
        from microcosm.walkcases import t_self_method


        class Kernel:
            def run(self):
                return self.pick()

            def pick(self):
                return t_self_method.X
        """,
    # A module the class body imports, called through self.
    "cmod": """
        from microcosm.walkcases import t_class_import


        def fn():
            return t_class_import.X
        """,
    "r_class_import": """
        class Kernel:
            from microcosm.walkcases import cmod as mod

            def run(self):
                return self.mod.fn()
        """,
    # ... and read where the class body runs: in the body itself, in a
    # decorator and in a default value.
    "cimp_body": """
        from microcosm.walkcases import t_class_import_body


        def fn():
            return t_class_import_body.X
        """,
    "cimp_decorator": """
        from microcosm.walkcases import t_class_import_decorator


        def mark(function):
            function.mark = t_class_import_decorator.X
            return function
        """,
    "cimp_default": """
        from microcosm.walkcases import t_class_import_default


        def fn():
            return t_class_import_default.X
        """,
    "r_class_import_scope": """
        class Kernel:
            import microcosm.walkcases.cimp_body as body_mod
            import microcosm.walkcases.cimp_decorator as decorator_mod
            import microcosm.walkcases.cimp_default as default_mod

            seen = body_mod.fn()

            @decorator_mod.mark
            def run(self, value=default_mod.fn()):
                return self.seen, value
        """,
    # -- Implicit calls ---------------------------------------------------------
    # Thing's dunders count wherever Thing does.
    "made": """
        from microcosm.walkcases import t_add, t_float, t_go, t_prop


        class Thing:
            def go(self):
                return t_go.X

            def __float__(self):
                return float(t_float.X)

            def __add__(self, other):
                return t_add.X

            @property
            def prop(self):
                return t_prop.X


        def make():
            return Thing()
        """,
    "r_float": """
        from microcosm.walkcases.made import make


        def run():
            return float(make())
        """,
    "r_add": """
        from microcosm.walkcases.made import make


        def run():
            return make() + 1
        """,
    "enum_mod": """
        import enum

        from microcosm.walkcases import t_sunder


        class Kind(enum.Enum):
            A = "a"

            @classmethod
            def _missing_(cls, value):
                return cls.A if t_sunder.X else None
        """,
    "r_sunder": """
        from microcosm.walkcases.enum_mod import Kind


        def run():
            return Kind("b")
        """,
    # A base's __init_subclass__ and class attribute, for a class run() makes.
    "made2": """
        from microcosm.walkcases import t_base_attr, t_factory, t_subclass


        def factory():
            return [t_factory.X]


        class Base:
            KIND = t_base_attr.X

            def __init_subclass__(cls, **kw):
                return t_subclass.X
        """,
    "r_subclass": """
        from microcosm.walkcases.made2 import Base


        def run():
            class Sub(Base):
                pass

            return Sub
        """,
    # The root class's own base runs __init_subclass__ for it.
    "made5": """
        from microcosm.walkcases import t_root_base


        class Base:
            def __init_subclass__(cls, **kw):
                return t_root_base.X
        """,
    "r_root_base": """
        from microcosm.walkcases.made5 import Base


        class Kernel(Base):
            def run(self):
                return 1
        """,
    # A dataclass field's default factory. Importing made2 also runs Base's
    # body, which the walk doesn't reach from here (see BLIND).
    "r_factory": """
        from dataclasses import dataclass, field

        from microcosm.walkcases.made2 import factory


        @dataclass
        class Holder:
            items: list = field(default_factory=factory)


        def run():
            return Holder()
        """,
    "r_lambda": """
        from microcosm.walkcases import t_lambda


        def run():
            f = lambda: t_lambda.X  # noqa: E731
            return f()
        """,
    # -- Import-time code -------------------------------------------------------
    # Headers of compound statements at module level, and in a class body.
    "r_headers": """
        from contextlib import nullcontext

        from microcosm.walkcases import (
            t_header_for,
            t_header_for_target,
            t_header_guard,
            t_header_if,
            t_header_match,
            t_header_pattern,
            t_header_try,
            t_header_trystar,
            t_header_while,
            t_header_with,
            t_header_with_target,
        )

        if t_header_if.X:
            VALUE = 1
        else:
            VALUE = 2
        while not t_header_while.X:
            pass
        for _row in t_header_for.ROWS:
            pass
        with nullcontext(t_header_with.X):
            pass
        _SLOTS = {}
        for _SLOTS[t_header_for_target.X] in (1,):
            pass
        with nullcontext(1) as _SLOTS[t_header_with_target.X]:
            pass
        match t_header_match.ROWS:
            case {"key": found}:
                FOUND = found
            case _:
                FOUND = None
        match 0:
            case t_header_pattern.X:
                PATTERN = 1
            case _ if t_header_guard.X:
                PATTERN = 2
        try:
            raise ValueError
        except ValueError if t_header_try.X else TypeError:
            pass
        try:
            raise ExceptionGroup("group", [ValueError()])
        except* (ValueError if t_header_trystar.X else TypeError):
            pass


        def run():
            return VALUE
        """,
    # Statements that run for their effect inside each kind of block.
    "r_blocks": """
        from microcosm.walkcases import (
            t_block_body,
            t_block_case,
            t_block_else,
            t_block_finally,
            t_block_handler,
        )

        if True:
            t_block_body.X
        for _row in ():
            pass
        else:
            t_block_else.X
        try:
            pass
        finally:
            t_block_finally.X
        try:
            raise ValueError
        except ValueError:
            t_block_handler.X
        match 1:
            case _:
                t_block_case.X


        def run():
            return 1
        """,
    "r_class_header": """
        from microcosm.walkcases import t_class_header


        class Kernel:
            if t_class_header.X:
                LEVEL = 1
            else:
                LEVEL = 2

            def run(self):
                return self.LEVEL
        """,
    # Calls at module level that fill a global or a table.
    "r_fill_global": """
        from microcosm.walkcases import t_fill_global

        _CACHE = None


        def _fill():
            global _CACHE
            _CACHE = t_fill_global.X


        _fill()


        def run():
            return _CACHE
        """,
    "r_setup_call": """
        from microcosm.walkcases import t_setup_call

        _TABLE = {}


        def _setup():
            _TABLE.update({"key": t_setup_call.X})


        _setup()


        def run():
            return _TABLE
        """,
    "r_fill_call": """
        from microcosm.walkcases import t_fill_call

        _TABLE = {}


        def _fill(table):
            table["key"] = t_fill_call.X


        _fill(_TABLE)


        def run():
            return _TABLE
        """,
    # A del from a table run() reads, and a store through an alias of it.
    # (Changes by name reaches these as well.)
    "r_del_item": """
        from microcosm.walkcases import t_del_item

        _TABLE = {"a": 1, "b": 2}
        del _TABLE[t_del_item.X and "a"]


        def run():
            return _TABLE
        """,
    "r_mutation_alias": """
        from microcosm.walkcases import t_mutation_alias

        _TABLE = {}
        _VIEW = _TABLE
        _VIEW["key"] = t_mutation_alias.X


        def run():
            return _TABLE
        """,
    # The same through a call's result, which no name links to the table,
    # so only the scan of import-time code reaches the targets.
    "r_del_hidden": """
        from microcosm.walkcases import t_del_hidden

        _TABLE = {"a": 1, "b": 2}


        def _table():
            return _TABLE


        _VIEW = _table()
        del _VIEW[t_del_hidden.X and "a"]


        def run():
            return _TABLE
        """,
    "r_store_hidden": """
        from microcosm.walkcases import t_store_hidden

        _TABLE = {}


        def _table():
            return _TABLE


        _VIEW = _table()
        _VIEW["key"] = t_store_hidden.X


        def run():
            return _TABLE
        """,
    # Patches to the root class.
    "patch_helper": """
        from microcosm.walkcases import t_setattr_patch


        def helper():
            return t_setattr_patch.X
        """,
    "r_setattr_patch": """
        from microcosm.walkcases.patch_helper import helper


        class Kernel:
            def run(self):
                return self.helper()


        setattr(Kernel, "helper", staticmethod(helper))
        """,
    "r_delattr_patch": """
        from microcosm.walkcases import t_delattr_patch


        class Base:
            def pick(self):
                return 0


        class Kernel(Base):
            def pick(self):
                return 1

            def run(self):
                return self.pick()


        delattr(Kernel, "pick" if t_delattr_patch.X else "run")
        """,
    # A reached class's body changing its own attribute.
    "r_class_body_update": """
        from microcosm.walkcases import t_class_body_update


        class Kernel:
            TABLE = {}
            TABLE.update({"key": t_class_body_update.X})

            def run(self):
                return self.TABLE
        """,
    # -- Changes by name --------------------------------------------------------
    # Another module re-exports the table run() reads and changes it in place.
    # The walk reads that module but reaches nothing it defines, so only the
    # change by name leads to the target.
    "update_table": """
        TABLE = {}
        """,
    "update_fill": """
        from microcosm.walkcases import t_update
        from microcosm.walkcases.update_table import TABLE

        TABLE.update(t_update.ROWS)
        """,
    "r_update": """
        from microcosm.walkcases.update_fill import TABLE


        def run():
            return TABLE
        """,
    "setitem_table": """
        TABLE = {}
        """,
    "setitem_fill": """
        from microcosm.walkcases import t_setitem
        from microcosm.walkcases.setitem_table import TABLE

        TABLE["key"] = t_setitem.X
        """,
    "r_setitem": """
        from microcosm.walkcases.setitem_fill import TABLE


        def run():
            return TABLE
        """,
    # A method call inside an assignment.
    "r_setdefault": """
        from microcosm.walkcases import t_setdefault

        _TABLE = {}
        _ROW = _TABLE.setdefault("key", t_setdefault.X)


        def run():
            return _TABLE
        """,
    # The body of a class nothing reaches still runs at import, and each of
    # its statements changes a table run() reads.
    "r_class_fill": """
        from microcosm.walkcases import (
            t_class_fill_call,
            t_class_fill_del,
            t_class_fill_item,
            t_class_fill_method,
            t_class_fill_view,
        )

        _TABLE = {}
        _ROWS = []
        _VIEW = _TABLE


        def _fill(table):
            table["call"] = t_class_fill_call.X


        class Filler:
            _TABLE.update({"method": t_class_fill_method.X})
            _TABLE["item"] = t_class_fill_item.X
            del _ROWS[t_class_fill_del.X :]
            _fill(_TABLE)
            _VIEW["view"] = t_class_fill_view.X


        def run():
            return _TABLE, _ROWS
        """,
    # Another module re-exports the registry run() reads, and a call statement
    # there takes the registry as an argument.
    "registry": """
        REGISTRY = []
        """,
    "registrar": """
        from microcosm.walkcases import t_registrar
        from microcosm.walkcases.registry import REGISTRY


        def _register(registry, value):
            registry.append(value)


        _register(REGISTRY, t_registrar.X)
        """,
    "r_registry": """
        from microcosm.walkcases.registrar import REGISTRY


        def run():
            return REGISTRY
        """,
    # ... with a for and a with statement there that store to an item of the
    # table, and what each stores.
    "store_table": """
        TABLE = {}
        """,
    "store_fill": """
        from microcosm.walkcases import t_for_store, t_with_store
        from microcosm.walkcases.store_table import TABLE


        class _Entered:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self.value

            def __exit__(self, *exc):
                return False


        for TABLE["for"] in (t_for_store.X,):
            pass
        with _Entered(t_with_store.X) as TABLE["with"]:
            pass
        """,
    "r_for_store": """
        from microcosm.walkcases.store_fill import TABLE


        def run():
            return TABLE
        """,
    # ... with a comprehension there that changes the table.
    "comp_table": """
        TABLE = {}
        """,
    "comp_fill": """
        from microcosm.walkcases import t_comprehension_fill
        from microcosm.walkcases.comp_table import TABLE

        [TABLE.update({key: t_comprehension_fill.X}) for key in ("key",)]
        """,
    "r_comprehension_fill": """
        from microcosm.walkcases.comp_fill import TABLE


        def run():
            return TABLE
        """,
    # += on an imported list extends that list in place. The walk reaches
    # aug_fill for helper(), but the statement only binds a name there that
    # nothing reads, so it doesn't run for its effect.
    "aug_table": """
        ROWS = []
        """,
    "aug_fill": """
        from microcosm.walkcases import t_augassign
        from microcosm.walkcases.aug_table import ROWS

        ROWS += [t_augassign.X]


        def helper():
            return 1
        """,
    "r_augassign": """
        from microcosm.walkcases import aug_fill
        from microcosm.walkcases.aug_table import ROWS


        def run():
            return ROWS, aug_fill.helper()
        """,
    # A class attribute that a statement in the class body changes, where the
    # statement neither runs for its effect nor binds a class attribute.
    "r_class_member_change": """
        from microcosm.walkcases import t_class_member_change


        class Kernel:
            TABLE = {}
            assert TABLE.setdefault("key", t_class_member_change.X)

            def run(self):
                return self.TABLE
        """,
    # A class nested in one nothing reaches runs at import too, and the names
    # it binds resolve in its own body.
    "r_class_nested_fill": """
        from microcosm.walkcases import t_class_nested_fill

        _TABLE = {}


        class Outer:
            class Inner:
                row = t_class_nested_fill.X
                _TABLE.update({"row": row})


        def run():
            return _TABLE
        """,
    # -- Attribute fallback -----------------------------------------------------
    "r_prop": """
        from microcosm.walkcases.made import make


        def run():
            return make().prop
        """,
    # A module passed as an argument, by name and by a dotted chain.
    "helper2": """
        from microcosm.walkcases import t_value


        def compute2():
            return t_value.X
        """,
    "use": """
        def use(module):
            return module.compute2()
        """,
    "r_value": """
        from microcosm.walkcases import helper2
        from microcosm.walkcases.use import use


        def run():
            return use(helper2)
        """,
    "dot.__init__": "",
    "dot.helper": """
        from microcosm.walkcases import t_dotted


        def compute2():
            return t_dotted.X
        """,
    "r_dotted": """
        import microcosm.walkcases.dot.helper
        from microcosm.walkcases.use import use


        def run():
            return use(microcosm.walkcases.dot.helper)
        """,
    # A namespace sub-package (nsdir has no __init__.py) under a module used
    # as a value. The walk must still settle.
    "nspkg.__init__": "",
    "nspkg.nsdir.leaf": """
        from microcosm.walkcases import t_namespace


        def fn():
            return t_namespace.X
        """,
    "r_namespace": """
        import importlib

        import microcosm.walkcases.nspkg.nsdir.leaf  # noqa: F401


        def run():
            package = importlib.import_module("microcosm.walkcases.nspkg")
            return package.nsdir.leaf.fn()
        """,
    # -- Literal names ----------------------------------------------------------
    "r_getattr": """
        from microcosm.walkcases.made import make


        def run():
            return getattr(make(), "go")()
        """,
    "made7": """
        from microcosm.walkcases import t_setattr


        class Thing:
            @property
            def level(self):
                return 0

            @level.setter
            def level(self, value):
                assert t_setattr.X
        """,
    "r_setattr": """
        from microcosm.walkcases.made7 import Thing


        def run():
            thing = Thing()
            setattr(thing, "level", 1)
            return thing
        """,
    "made8": """
        from microcosm.walkcases import t_delattr, t_hasattr


        class Probe:
            @property
            def ready(self):
                return t_hasattr.X

            @property
            def item(self):
                return 0

            @item.deleter
            def item(self):
                assert t_delattr.X
        """,
    "r_hasattr": """
        from microcosm.walkcases.made8 import Probe


        def run():
            return hasattr(Probe(), "ready")
        """,
    "r_delattr": """
        from microcosm.walkcases.made8 import Probe


        def run():
            probe = Probe()
            delattr(probe, "item")
            return probe
        """,
    "made12": """
        from microcosm.walkcases import t_getattr_constant


        class Gauge:
            @property
            def probe(self):
                return t_getattr_constant.X
        """,
    "r_getattr_constant": """
        from microcosm.walkcases.made12 import Gauge

        _NAME = "probe"


        def run():
            return getattr(Gauge(), _NAME)
        """,
    "made4": """
        from microcosm.walkcases import t_attrgetter


        class Thing:
            @property
            def named(self):
                return t_attrgetter.X
        """,
    "r_attrgetter": """
        from operator import attrgetter

        from microcosm.walkcases.made4 import Thing


        def run():
            return attrgetter("named")(Thing())
        """,
    "made13": """
        from microcosm.walkcases import t_methodcaller


        class Pinger:
            def ping(self):
                return t_methodcaller.X
        """,
    "r_methodcaller": """
        from operator import methodcaller

        from microcosm.walkcases.made13 import Pinger


        def run():
            return methodcaller("ping")(Pinger())
        """,
    # A match class pattern reads its keyword names, and the names in an
    # explicit __match_args__ for its positional ones.
    "made10": """
        from microcosm.walkcases import t_match_keyword, t_match_positional


        class Point:
            __match_args__ = ("left",)

            @property
            def depth(self):
                return t_match_keyword.X

            @property
            def left(self):
                return t_match_positional.X
        """,
    "r_match_keyword": """
        from microcosm.walkcases.made10 import Point


        def run():
            match Point():
                case Point(depth=value):
                    return value
            return None
        """,
    "r_match_positional": """
        from microcosm.walkcases.made10 import Point


        def run():
            match Point():
                case Point(value):
                    return value
            return None
        """,
    # A class that doesn't set __match_args__ takes its base's.
    "made14": """
        from microcosm.walkcases import t_match_inherited


        class Base:
            __match_args__ = ("left",)


        class Point(Base):
            @property
            def left(self):
                return t_match_inherited.X
        """,
    "r_match_inherited": """
        from microcosm.walkcases.made14 import Point


        def run():
            match Point():
                case Point(value):
                    return value
            return None
        """,
    "imp_mod": """
        from microcosm.walkcases import t_import_module


        def fn():
            return t_import_module.X
        """,
    "r_import_module": """
        import importlib


        def run():
            return importlib.import_module("microcosm.walkcases.imp_mod").fn()
        """,
    "rel_mod": """
        from microcosm.walkcases import t_import_relative


        def fn():
            return t_import_relative.X
        """,
    "r_import_relative": """
        import importlib


        def run():
            return importlib.import_module(".rel_mod", __package__).fn()
        """,
    # __import__ with a fromlist returns the module it names.
    "imp_leaf": """
        from microcosm.walkcases import t_import_from


        def fn():
            return t_import_from.X
        """,
    "r_import_from": """
        def run():
            module = __import__("microcosm.walkcases.imp_leaf", fromlist=["fn"])
            return module.fn()
        """,
    # Without a fromlist, __import__ returns the top package, and a chain from
    # there reaches a module that the package holds.
    "dunder_named": """
        VALUE = 1
        """,
    "r_dunder_import_top": """
        import microcosm.walkcases.t_dunder_top  # so that the attribute exists


        def run():
            package = __import__("microcosm.walkcases.dunder_named")
            return package.walkcases.t_dunder_top.X
        """,
    # __import__ with a level above zero counts from the caller's package.
    "level_mod": """
        from microcosm.walkcases import t_import_level


        def fn():
            return t_import_level.X
        """,
    "r_import_level": """
        def run():
            return __import__("level_mod", globals(), None, ["fn"], 1).fn()
        """,
    # import_module relative to the package that __spec__ names.
    "spec_mod": """
        from microcosm.walkcases import t_import_spec


        def fn():
            return t_import_spec.X
        """,
    "r_import_spec": """
        import importlib


        def run():
            return importlib.import_module(".spec_mod", package=__spec__.parent).fn()
        """,
    # import_module relative to the module's own name: two dots climb from
    # the module to its package, one level short of where __package__ climbs.
    "name_mod": """
        from microcosm.walkcases import t_import_name


        def fn():
            return t_import_name.X
        """,
    "r_import_name": """
        import importlib


        def run():
            return importlib.import_module("..name_mod", __name__).fn()
        """,
    # An importer under a module-level name: an import's and an assignment's.
    "renamed_mod": """
        from microcosm.walkcases import t_import_renamed


        def fn():
            return t_import_renamed.X
        """,
    "renamed_alias_mod": """
        from microcosm.walkcases import t_import_renamed_alias


        def fn():
            return t_import_renamed_alias.X
        """,
    "r_import_renamed": """
        import importlib
        from importlib import import_module as load

        _import = importlib.import_module


        def run():
            first = load("microcosm.walkcases.renamed_mod").fn()
            return first, _import("microcosm.walkcases.renamed_alias_mod").fn()
        """,
    # -- What the walk can't see ------------------------------------------------
    # A decorator records _entry, and run() calls whatever the registry holds.
    "reg": """
        REGISTRY = []


        def register(function):
            REGISTRY.append(function)
            return function
        """,
    "r_registry_decorator": """
        from microcosm.walkcases import t_registry_decorator
        from microcosm.walkcases.reg import REGISTRY, register


        @register
        def _entry():
            return t_registry_decorator.X


        def run():
            return [entry() for entry in REGISTRY]
        """,
}

# -- Cost ----------------------------------------------------------------------
# Shapes in which a walk that ran each lookup afresh wherever it met it would
# take time exponential in a diamond's depth or factorial in a clique's size.
# The walk runs each lookup once, so each of these takes milliseconds.

#: How deep each diamond goes, and how many modules the clique holds.
_DEPTH = 22
_CLIQUE = 8
#: The lazy packages in the cycle.
_LAZY = 3

# Each level's two modules star-import both modules of the level below, and
# the bottom level imports the target. run() also calls len, which no module
# binds, so the walk looks len up through every module.
SOURCES.update(
    {
        f"diamond_{level}_{side}": (
            f"from {PACKAGE} import t_star_diamond  # noqa: F401\n"
            if level == _DEPTH - 1
            else f"from {PACKAGE}.diamond_{level + 1}_a import *\n"
            f"from {PACKAGE}.diamond_{level + 1}_b import *\n"
        )
        for level in range(_DEPTH)
        for side in "ab"
    }
)
SOURCES["r_star_diamond"] = """
    from microcosm.walkcases.diamond_0_a import *


    def run():
        return len([t_star_diamond.X])
    """
# Each module of a clique star-imports all the others; the first imports the
# target.
SOURCES.update(
    {
        f"clique_{index}": (
            f"from {PACKAGE} import t_star_clique  # noqa: F401\n" if index == 0 else ""
        )
        + "".join(
            f"from {PACKAGE}.clique_{other} import *\n"
            for other in range(_CLIQUE)
            if other != index
        )
        for index in range(_CLIQUE)
    }
)
SOURCES["r_star_clique"] = """
    from microcosm.walkcases.clique_1 import *


    def run():
        return len([t_star_clique.X])
    """
# Constants in a diamond: each of a level's two names both of the next.
SOURCES["r_alias_diamond"] = (
    f"from {PACKAGE} import t_alias_diamond\n\n"
    f"_V{_DEPTH}a = t_alias_diamond\n"
    f"_V{_DEPTH}b = t_alias_diamond\n"
    + "".join(
        f"_V{level}{side} = _V{level + 1}a or _V{level + 1}b\n"
        for level in range(_DEPTH - 1, -1, -1)
        for side in "ab"
    )
    + "\n\ndef run():\n    return len([_V0a.X])\n"
)
# Modules in a diamond, each binding V to either module's V below it.
SOURCES.update(
    {
        f"aliases_{level}_{side}": (
            f"from {PACKAGE} import t_alias_modules\n\nV = t_alias_modules\n"
            if level == _DEPTH - 1
            else f"from {PACKAGE}.aliases_{level + 1}_a import V as _a\n"
            f"from {PACKAGE}.aliases_{level + 1}_b import V as _b\n\n"
            "V = _a or _b\n"
        )
        for level in range(_DEPTH)
        for side in "ab"
    }
)
SOURCES["r_alias_modules"] = """
    from microcosm.walkcases.aliases_0_a import V


    def run():
        return len([V.X])
    """
# Classes in a diamond, and a positional class pattern on the top one: it
# takes the bottom class's __match_args__, through each path.
SOURCES["r_match_diamond"] = (
    f"from {PACKAGE} import t_match_diamond\n\n\n"
    f"class K{_DEPTH}a:\n"
    '    __match_args__ = ("go",)\n\n'
    "    def go(self):\n"
    "        return t_match_diamond.X\n\n\n"
    f"class K{_DEPTH}b:\n"
    "    pass\n"
    + "".join(
        f"\n\nclass K{level}{side}(K{level + 1}a, K{level + 1}b):\n    pass\n"
        for level in range(_DEPTH - 1, -1, -1)
        for side in "ab"
    )
    + "\n\ndef run():\n"
    "    match K0a():\n"
    "        case K0a(go):\n"
    "            return go()\n"
)
# Modules in a diamond of alternatives: each imports the target from either
# module below it.
SOURCES.update(
    {
        f"alts_{level}_{side}": (
            f"from {PACKAGE} import t_import_diamond  # noqa: F401\n"
            if level == _DEPTH - 1
            else "try:\n"
            f"    from {PACKAGE}.alts_{level + 1}_a import t_import_diamond\n"
            "except ImportError:\n"
            f"    from {PACKAGE}.alts_{level + 1}_b import t_import_diamond\n"
        )
        for level in range(_DEPTH)
        for side in "ab"
    }
)
SOURCES["r_import_diamond"] = """
    from microcosm.walkcases.alts_0_a import t_import_diamond


    def run():
        return len([t_import_diamond.X])
    """
# Lazy packages in a cycle: each one's __getattr__ reads its inner module,
# which imports the name from the next package. The first package's inner
# module imports the target, or, failing that, the name from the second.
SOURCES.update(
    {
        f"lazyc_{index}.__init__": """
            def __getattr__(name):
                from importlib import import_module

                return getattr(import_module(f"{__name__}.inner"), name)
            """
        for index in range(_LAZY)
    }
)
SOURCES.update(
    {
        f"lazyc_{index}.inner": (
            f"from {PACKAGE}.lazyc_{(index + 1) % _LAZY} import t_lazy_cycle\n"
        )
        for index in range(1, _LAZY)
    }
)
SOURCES["lazyc_0.inner"] = """
    try:
        from microcosm.walkcases import t_lazy_cycle
    except ImportError:
        from microcosm.walkcases.lazyc_1 import t_lazy_cycle
    """
SOURCES["r_lazy_cycle"] = """
    from microcosm.walkcases.lazyc_1 import t_lazy_cycle


    def run():
        return len([t_lazy_cycle.X])
    """
# Constants that name each other: _B names _A and the target, and _A, bound
# again, names _B. Python binds both to Base.
SOURCES["r_alias_cycle"] = """
    from microcosm.walkcases import t_alias_cycle


    class Base:
        X = 1


    _A = Base
    _B = _A or t_alias_cycle
    _A = _B


    def run():
        return _A.X
    """
# Constants that name only each other: each stands for both.
SOURCES["r_alias_pure_cycle"] = """
    _A = None
    _B = _A
    _A = _B


    def run():
        return getattr(_B, "X", None)
    """

# A root that makes a Thing reaches every dunder Thing defines, so it reaches
# t_add and t_float as well: the walk over-approximates there by design.
_THING = {"add", "float"}

#: (root module, qualname) -> the targets its walk reaches, exactly.
ROOTS: dict[tuple[str, str], set[str]] = {
    # Bodies
    ("r_default", "run"): {"default"},
    ("r_metaclass", "Kernel.run"): {"metaclass"},
    ("r_decorator", "Kernel.run"): {"decorator"},
    # Bindings
    ("r_star", "run"): {"star"},
    ("r_alternative", "run"): {"alternative"},
    ("r_alternative_first", "run"): {"alternative_first"},
    ("r_local_alternative", "run"): {"local_alternative"},
    ("r_local_alternative_first", "run"): {"local_alternative_first"},
    ("r_bind_for", "run"): {"for_target"},
    ("r_bind_with", "run"): {"with_target"},
    ("r_bind_match", "run"): {"match_capture"},
    ("r_bind_star", "run"): {"match_star"},
    ("r_bind_rest", "run"): {"match_rest"},
    ("r_bind_walrus", "run"): {"walrus"},
    ("r_lazy", "run"): {"lazy"},
    ("r_lazy_assigned", "run"): {"lazy_assigned"},
    ("r_star_lazy", "run"): {"star_lazy"},
    ("r_class_global", "Kernel.run"): {
        "class_global",
        "class_global_alias",
        "class_global_class",
        "class_global_def",
        "class_global_hidden",
        "class_global_import",
    },
    # Scopes
    ("r_shadow", "run"): {"shadow"},
    ("r_comprehension", "run"): {"comprehension"},
    ("r_comprehension_decoy", "run"): {"comprehension_iter"},
    ("r_sibling", "run"): {"sibling", "local"},
    ("r_class_scope", "Holder.run"): {"class_scope"},
    ("r_lambda_shadow", "Kernel.run"): {"lambda_shadow"},
    ("r_nested_scope", "run"): {"nested_scope"},
    ("r_class_decorator", "Kernel.run"): set(),
    ("r_body_before", "Kernel.run"): {"body_before"},
    ("r_body_conditional", "Kernel.run"): {"body_conditional"},
    ("r_body_annotation", "Kernel.run"): {"body_annotation"},
    ("r_body_decorator", "Kernel.run"): {"body_decorator"},
    ("r_body_del", "Kernel.run"): {"body_del"},
    ("r_body_nested", "run"): {"body_nested"},
    ("r_global", "run"): {"global"},
    ("r_nonlocal", "run"): {"nonlocal"},
    # Aliases
    ("r_alias", "run"): {"alias"},
    ("r_unpack", "run"): {"unpack"},
    ("r_conditional", "run"): {"conditional"},
    ("r_alias_base", "Kernel.run"): {"alias_base"},
    ("r_base_conditional", "Kernel.run"): {"base_if", "base_else"},
    ("r_base_tuple", "Kernel.run"): {"base_first", "base_second"},
    ("r_base_getattr", "Kernel.run"): {"base_getattr"},
    ("r_base_subscript", "Kernel.run"): {"base_subscript"},
    ("r_base_or", "Kernel.run"): {"base_or"},
    ("r_base_walrus", "Kernel.run"): {"base_walrus"},
    ("r_base_starred", "Kernel.run"): {"base_starred"},
    ("r_base_dunder_import", "Kernel.run"): {"base_dunder_import"},
    ("r_import_alias_base", "Kernel.run"): {"base_import"},
    ("r_getattr_alias", "run"): {"getattr_alias"},
    # Inheritance
    ("r_override", "Kernel.run"): {"override"},
    ("r_subscript", "Kernel.run"): {"subscript"},
    # Self and cls
    ("r_self_method", "Kernel.run"): {"self_method"},
    ("r_class_import", "Kernel.run"): {"class_import"},
    ("r_class_import_scope", "Kernel.run"): {
        "class_import_body",
        "class_import_decorator",
        "class_import_default",
    },
    # Implicit calls
    ("r_float", "run"): {"float", *_THING},
    ("r_add", "run"): {"add", *_THING},
    ("r_sunder", "run"): {"sunder"},
    ("r_subclass", "run"): {"subclass", "base_attr"},
    ("r_root_base", "Kernel.run"): {"root_base"},
    ("r_factory", "run"): {"factory"},
    ("r_lambda", "run"): {"lambda"},
    # Import-time code
    ("r_headers", "run"): {
        "header_for",
        "header_for_target",
        "header_guard",
        "header_if",
        "header_match",
        "header_pattern",
        "header_try",
        "header_trystar",
        "header_while",
        "header_with",
        "header_with_target",
    },
    ("r_blocks", "run"): {
        "block_body",
        "block_case",
        "block_else",
        "block_finally",
        "block_handler",
    },
    ("r_class_header", "Kernel.run"): {"class_header"},
    ("r_fill_global", "run"): {"fill_global"},
    ("r_setup_call", "run"): {"setup_call"},
    ("r_fill_call", "run"): {"fill_call"},
    ("r_del_item", "run"): {"del_item"},
    ("r_mutation_alias", "run"): {"mutation_alias"},
    ("r_del_hidden", "run"): {"del_hidden"},
    ("r_store_hidden", "run"): {"store_hidden"},
    ("r_setattr_patch", "Kernel.run"): {"setattr_patch"},
    ("r_delattr_patch", "Kernel.run"): {"delattr_patch"},
    ("r_class_body_update", "Kernel.run"): {"class_body_update"},
    # Changes by name
    ("r_update", "run"): {"update"},
    ("r_setitem", "run"): {"setitem"},
    ("r_setdefault", "run"): {"setdefault"},
    ("r_class_fill", "run"): {
        "class_fill_call",
        "class_fill_del",
        "class_fill_item",
        "class_fill_method",
        "class_fill_view",
    },
    ("r_registry", "run"): {"registrar"},
    ("r_for_store", "run"): {"for_store", "with_store"},
    ("r_comprehension_fill", "run"): {"comprehension_fill"},
    ("r_augassign", "run"): {"augassign"},
    ("r_class_member_change", "Kernel.run"): {"class_member_change"},
    ("r_class_nested_fill", "run"): {"class_nested_fill"},
    # Attribute fallback
    ("r_prop", "run"): {"prop", *_THING},
    ("r_value", "run"): {"value"},
    ("r_dotted", "run"): {"dotted"},
    ("r_namespace", "run"): {"namespace"},
    # Literal names
    ("r_getattr", "run"): {"go", *_THING},
    ("r_setattr", "run"): {"setattr"},
    ("r_hasattr", "run"): {"hasattr"},
    ("r_delattr", "run"): {"delattr"},
    ("r_getattr_constant", "run"): {"getattr_constant"},
    ("r_attrgetter", "run"): {"attrgetter"},
    ("r_methodcaller", "run"): {"methodcaller"},
    ("r_match_keyword", "run"): {"match_keyword"},
    ("r_match_positional", "run"): {"match_positional"},
    ("r_match_inherited", "run"): {"match_inherited"},
    ("r_import_module", "run"): {"import_module"},
    ("r_import_relative", "run"): {"import_relative"},
    ("r_import_from", "run"): {"import_from"},
    ("r_dunder_import_top", "run"): {"dunder_top"},
    ("r_import_level", "run"): {"import_level"},
    ("r_import_spec", "run"): {"import_spec"},
    ("r_import_name", "run"): {"import_name"},
    ("r_import_renamed", "run"): {"import_renamed", "import_renamed_alias"},
    # Cost
    ("r_star_diamond", "run"): {"star_diamond"},
    ("r_star_clique", "run"): {"star_clique"},
    ("r_alias_diamond", "run"): {"alias_diamond"},
    ("r_alias_modules", "run"): {"alias_modules"},
    ("r_match_diamond", "run"): {"match_diamond"},
    ("r_import_diamond", "run"): {"import_diamond"},
    ("r_lazy_cycle", "run"): {"lazy_cycle"},
    ("r_alias_cycle", "run"): {"alias_cycle"},
    ("r_alias_pure_cycle", "run"): set(),
    # What the walk can't see
    ("r_registry_decorator", "run"): set(),
}

#: Targets a root's walk reaches that its run doesn't read. The walk reaches
#: every dunder of a class it references, both branches of a condition, and
#: both operands of ``or``.
OVER: dict[tuple[str, str], set[str]] = {
    ("r_alias_cycle", "run"): {"alias_cycle"},
    ("r_float", "run"): {"add"},
    ("r_add", "run"): {"float"},
    ("r_prop", "run"): _THING,
    ("r_getattr", "run"): _THING,
    ("r_base_conditional", "Kernel.run"): {"base_if"},
    ("r_base_tuple", "Kernel.run"): {"base_first"},
}

#: Targets a root's run reads that its walk doesn't reach, each one a line of
#: the walk's "What it can't see".
BLIND: dict[tuple[str, str], set[str]] = {
    # Importing made2 runs Base's body, a class the walk doesn't reach from
    # factory. Nothing hands that value to the run.
    ("r_factory", "run"): {"base_attr"},
    # Registration: the decorator records _entry, and run() calls it.
    ("r_registry_decorator", "run"): {"registry_decorator"},
}


def expected_reads(root: tuple[str, str]) -> set[str]:
    """The targets a root's run must read: its walk's, less OVER, plus BLIND."""

    return (ROOTS[root] - OVER.get(root, set())) | BLIND.get(root, set())


def write(root: Path) -> dict[str, Path]:
    """Write :data:`SOURCES` under ``root``; return their modules' paths."""

    source = root / "packages" / "cases" / "src"
    base = source.joinpath(*PACKAGE.split("."))
    for name, text in SOURCES.items():
        path = base.joinpath(*name.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    (source / "walkcase_reads.py").write_text(_READS, encoding="utf-8")
    return module_paths(root)


def targets(walked: Reach) -> set[str]:
    """The targets a walk reached, by their names in :data:`TARGETS`."""

    prefix = f"{PACKAGE}.t_"
    return {
        name.removeprefix(prefix) for name in walked.modules if name.startswith(prefix)
    }


# Imports each root's module afresh and calls the root; prints, for each root,
# the targets Python read, or the error the root raised.
_DRIVER = """
import importlib, json, sys

source, package, roots = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, source)
import walkcase_reads

found = {}
for module, qualname in roots:
    for name in [name for name in sys.modules if name.split(".")[0] == "microcosm"]:
        del sys.modules[name]
    walkcase_reads.READS.clear()
    try:
        loaded = importlib.import_module(f"{package}.{module}")
        owner, _, name = qualname.rpartition(".")
        root = getattr(getattr(loaded, owner)(), name) if owner else getattr(loaded, name)
        root()
    except Exception as error:
        found[f"{module}:{qualname}"] = f"{type(error).__name__}: {error}"
    else:
        found[f"{module}:{qualname}"] = sorted(walkcase_reads.READS)
print(json.dumps(found))
"""


def compose(
    root: Path, roots: Iterable[tuple[str, str]]
) -> tuple[tuple[str, str], dict[str, Path]]:
    """Write a module whose ``run()`` runs each of ``roots`` in turn.

    ``root`` holds what :func:`write` wrote. Returns the new root, named for
    the roots it runs, and the tree's module paths with it.
    """

    picked = sorted(roots)
    digest = sha256(json.dumps(picked).encode()).hexdigest()[:12]
    lines = ["def run():"]
    for module, qualname in picked:
        owner, _, name = qualname.rpartition(".")
        lines.append(f"    from {PACKAGE} import {module}")
        lines.append(
            f"    {module}.{owner}().{name}()" if owner else f"    {module}.{name}()"
        )
    module = f"c_{digest}"
    base = root.joinpath("packages", "cases", "src", *PACKAGE.split("."))
    (base / f"{module}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (module, "run"), module_paths(root)


def python_reads(
    root: Path, roots: Iterable[tuple[str, str]] | None = None
) -> dict[tuple[str, str], set[str] | str]:
    """Run each root (by default each in :data:`ROOTS`) for real; the targets
    Python read.

    ``root`` holds what :func:`write` wrote. One interpreter with no site
    packages and no environment (``-I -S``) imports each root's module from
    scratch, then calls the root (for ``Class.method``, on a new instance). A
    root that raises maps to its error's text instead.
    """

    picked = sorted(ROOTS if roots is None else roots)

    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _DRIVER,
            str(root / "packages" / "cases" / "src"),
            PACKAGE,
            json.dumps(picked),
        ],
        capture_output=True,
        check=True,
        cwd=root,
        text=True,
    )
    found = json.loads(done.stdout)
    return {
        (module, qualname): (
            set(reads)
            if isinstance(reads := found[f"{module}:{qualname}"], list)
            else reads
        )
        for module, qualname in picked
    }
