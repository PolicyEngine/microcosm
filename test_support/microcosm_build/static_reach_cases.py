"""Synthetic modules for the static walk's own tests.

Each root in :data:`ROOTS` reaches its target modules (``t_*``) in one of the
ways that :mod:`test_support.microcosm_build.static_reach` lists, and its walk
must reach exactly those targets. A target holds only constants, so a root
reaches it only through code the walk follows.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from test_support.microcosm_build.static_reach import Reach, module_paths

#: The package that holds the synthetic modules.
PACKAGE = "microcosm.walkcases"

TARGETS = (
    "add",
    "alias",
    "alias_base",
    "alternative",
    "attrgetter",
    "base_attr",
    "class_scope",
    "comprehension",
    "conditional",
    "decorator",
    "factory",
    "float",
    "global",
    "go",
    "import_module",
    "lambda",
    "lazy",
    "local",
    "nonlocal",
    "override",
    "prop",
    "registrar",
    "root_base",
    "setattr",
    "setitem",
    "shadow",
    "shadowed",
    "sibling",
    "star",
    "subclass",
    "sunder",
    "unpack",
    "update",
    "value",
)

#: Each module's source, by its dotted name inside :data:`PACKAGE`.
SOURCES: dict[str, str] = {
    "__init__": "",
    **{f"t_{name}": "X = 1\nROWS = {}\n" for name in TARGETS},
    # -- Scopes ---------------------------------------------------------------
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
    # A method's decorator reads the class body's names first, so the
    # module's import of the same name stays unreached.
    "r_class_decorator": """
        from microcosm.walkcases import t_shadowed as target


        class Kernel:
            target = staticmethod(lambda function: function)

            @target
            def run(self):
                return 1
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
    # -- Resolution -----------------------------------------------------------
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
    # A name with two bindings reaches both, whichever runs.
    "fast": """
        def impl():
            return 0
        """,
    "slow": """
        from microcosm.walkcases import t_alternative


        def impl():
            return t_alternative.X
        """,
    "r_alternative": """
        try:
            from microcosm.walkcases.slow import impl
        except ImportError:
            from microcosm.walkcases.fast import impl


        def run():
            return impl()
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
    # -- Constants ------------------------------------------------------------
    # A constant that holds a module stands for it.
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
    # ... through tuple unpacking, and through either branch of a condition.
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
    # A class base held in a constant: run() is the base's.
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
    # -- Changes in place -----------------------------------------------------
    "r_update": """
        from microcosm.walkcases import t_update

        _TABLE = {}
        _TABLE.update(t_update.ROWS)


        def run():
            return _TABLE
        """,
    "r_setitem": """
        from microcosm.walkcases import t_setitem

        _TABLE = {}
        _TABLE["key"] = t_setitem.X


        def run():
            return _TABLE
        """,
    # Another module the walk reads fills the registry run() reads.
    "registry": """
        REGISTRY = []
        """,
    "registrar": """
        from microcosm.walkcases import t_registrar
        from microcosm.walkcases.registry import REGISTRY

        REGISTRY.append(t_registrar.X)


        def helper():
            return 1
        """,
    "r_registry": """
        from microcosm.walkcases import registrar
        from microcosm.walkcases.registry import REGISTRY


        def run():
            return REGISTRY, registrar.helper()
        """,
    # -- Attributes of objects the walk can't type ------------------------------
    # Thing's dunders count wherever Thing does (see Implicit calls).
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
    "r_prop": """
        from microcosm.walkcases.made import make


        def run():
            return make().prop
        """,
    # A module passed as an argument.
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
    # -- Implicit calls ---------------------------------------------------------
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
                return t_sunder.X
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
    # A method's decorator, held in a class attribute.
    "deco_mod": """
        from microcosm.walkcases import t_decorator


        def wrap(function):
            assert t_decorator.X
            return function
        """,
    "r_decorator": """
        from microcosm.walkcases import deco_mod


        class Kernel:
            _deco = deco_mod.wrap

            @_deco
            def run(self):
                return 1
        """,
}

# A root that makes a Thing reaches every dunder Thing defines, so it reaches
# t_add and t_float as well: the walk over-approximates there by design.
_THING = {"add", "float"}

#: (root module, qualname) -> the targets its walk reaches, exactly.
ROOTS: dict[tuple[str, str], set[str]] = {
    ("r_shadow", "run"): {"shadow"},
    ("r_comprehension", "run"): {"comprehension"},
    ("r_sibling", "run"): {"sibling", "local"},
    ("r_class_scope", "Holder.run"): {"class_scope"},
    ("r_class_decorator", "Kernel.run"): set(),
    ("r_global", "run"): {"global"},
    ("r_nonlocal", "run"): {"nonlocal"},
    ("r_star", "run"): {"star"},
    ("r_alternative", "run"): {"alternative"},
    ("r_lazy", "run"): {"lazy"},
    ("r_alias", "run"): {"alias"},
    ("r_unpack", "run"): {"unpack"},
    ("r_conditional", "run"): {"conditional"},
    ("r_alias_base", "Kernel.run"): {"alias_base"},
    ("r_update", "run"): {"update"},
    ("r_setitem", "run"): {"setitem"},
    ("r_registry", "run"): {"registrar"},
    ("r_prop", "run"): {"prop", *_THING},
    ("r_value", "run"): {"value"},
    ("r_override", "Kernel.run"): {"override"},
    ("r_getattr", "run"): {"go", *_THING},
    ("r_setattr", "run"): {"setattr"},
    ("r_attrgetter", "run"): {"attrgetter"},
    ("r_import_module", "run"): {"import_module"},
    ("r_float", "run"): {"float", *_THING},
    ("r_add", "run"): {"add", *_THING},
    ("r_sunder", "run"): {"sunder"},
    ("r_subclass", "run"): {"subclass", "base_attr"},
    ("r_root_base", "Kernel.run"): {"root_base"},
    ("r_factory", "run"): {"factory"},
    ("r_lambda", "run"): {"lambda"},
    ("r_decorator", "Kernel.run"): {"decorator"},
}


def write(root: Path) -> dict[str, Path]:
    """Write :data:`SOURCES` under ``root``; return their modules' paths."""

    base = root / "packages" / "cases" / "src" / "microcosm" / "walkcases"
    for name, source in SOURCES.items():
        path = base.joinpath(*name.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return module_paths(root)


def targets(walked: Reach) -> set[str]:
    """The targets a walk reached, by their names in :data:`TARGETS`."""

    prefix = f"{PACKAGE}.t_"
    return {
        name.removeprefix(prefix) for name in walked.modules if name.startswith(prefix)
    }
