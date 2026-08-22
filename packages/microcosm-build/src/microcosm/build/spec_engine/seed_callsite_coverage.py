"""Fail-closed static coverage of the US-pool stochastic source surface.

The source universe and physical-callsite manifest in this module are
deliberately independent of :mod:`microcosm.build.spec_engine.seeds`.  Tests
join the two independent inventories and require exact equality.  Adding a
Python module or a stochastic/hash call therefore fails until that physical
site is explicitly bound to a legacy-v1 ledger id or given a typed exemption.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True, slots=True)
class PhysicalCallsite:
    """A source-stable physical callsite key (line numbers are diagnostics)."""

    module: str
    qualname: str
    api: str
    occurrence: int = 0


@dataclass(frozen=True, slots=True)
class DiscoveredCallsite:
    """One discovered callsite plus its current source line."""

    callsite: PhysicalCallsite
    line: int


@dataclass(frozen=True, slots=True)
class CallsiteBinding:
    """Bind one physical callsite to one or more logical seed-ledger sites."""

    callsite: PhysicalCallsite
    site_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.site_ids or self.site_ids != tuple(sorted(set(self.site_ids))):
            raise ValueError("callsite site ids must be nonempty, sorted, and unique")


@dataclass(frozen=True, slots=True)
class CallsiteExemption:
    """A typed, reviewable exclusion from the published producer graph."""

    callsite: PhysicalCallsite
    kind: str
    reason: str

    def __post_init__(self) -> None:
        if not self.kind or not self.reason:
            raise ValueError("callsite exemptions require a kind and reason")


HASH_CLASSIFICATION_KINDS = frozenset(
    {
        "stochastic_draw",
        "content_identity",
        "source_integrity",
        "operational_subset",
    }
)


@dataclass(frozen=True, slots=True)
class HashCallsiteClassification:
    """Classify every hash use, including deterministic non-draw hashes."""

    callsite: PhysicalCallsite
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in HASH_CLASSIFICATION_KINDS:
            raise ValueError(f"unknown hash callsite classification {self.kind!r}")


@dataclass(frozen=True, slots=True)
class SourceNamespaceExemption:
    """A typed exclusion from the otherwise filesystem-wide shard census."""

    prefix: str
    kind: str
    reason: str

    def __post_init__(self) -> None:
        if not self.prefix or not self.kind or not self.reason:
            raise ValueError("source namespace exemptions must be fully described")


def _key(
    module: str,
    qualname: str,
    api: str,
    occurrence: int = 0,
) -> PhysicalCallsite:
    return PhysicalCallsite(module, qualname, api, occurrence)


# Filesystem-derived production scope.  All shared build modules and every US
# runtime module are included automatically; fit/calibrate are shared kernels.
# The UK runtime and its one shared helper are explicitly outside this US lane.
_BUILD_SOURCE_ROOT = Path("packages/microcosm-build/src/microcosm/build")
_FIT_SOURCE_ROOT = Path("packages/microcosm-fit/src/microcosm/fit")
_CALIBRATE_SOURCE_ROOT = Path("packages/microcosm-calibrate/src/microcosm/calibrate")
_DATA_SOURCE_ROOT = Path("packages/microcosm-data/src/microcosm/data")
_FRAME_SOURCE_ROOT = Path("packages/microcosm-frame/src/microcosm/frame")
_POOL_TOOL = Path("tools/build_us_multispine_pool.py")

SOURCE_NAMESPACE_EXEMPTIONS = (
    SourceNamespaceExemption(
        "microcosm.build.spec_engine.brokers",
        "rng_broker_implementation",
        "the broker is the sole attested construction boundary for RNG, file, "
        "environment, and clock primitives; its callers remain in the scan",
    ),
    SourceNamespaceExemption(
        "microcosm.build.uk_runtime",
        "uk_only_namespace",
        "UK runtime is a separate country producer graph and is not reachable "
        "from tools.build_us_multispine_pool",
    ),
    SourceNamespaceExemption(
        "microcosm.build.stochastic_assignment",
        "uk_only_shared_helper",
        "shared-location helper is imported only by microcosm.build.uk_runtime",
    ),
)
UK_ONLY_SOURCE_PREFIXES = tuple(
    row.prefix for row in SOURCE_NAMESPACE_EXEMPTIONS if row.kind.startswith("uk_")
)
_EXCLUDED_SOURCE_PREFIXES = tuple(row.prefix for row in SOURCE_NAMESPACE_EXEMPTIONS)


def _has_module_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _module_name(path: Path, source_root: Path, prefix: str) -> str:
    relative = path.relative_to(source_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((prefix, *parts))


def discover_production_source_modules(repository_root: Path) -> dict[str, Path]:
    """Return the filesystem-derived US-pool and shared-kernel source universe."""

    modules: dict[str, Path] = {}
    build_root = repository_root / _BUILD_SOURCE_ROOT
    for path in sorted(build_root.rglob("*.py")):
        module = _module_name(path, build_root, "microcosm.build")
        if any(
            _has_module_prefix(module, prefix) for prefix in _EXCLUDED_SOURCE_PREFIXES
        ):
            continue
        modules[module] = path

    for relative_root, prefix in (
        (_FIT_SOURCE_ROOT, "microcosm.fit"),
        (_CALIBRATE_SOURCE_ROOT, "microcosm.calibrate"),
        (_DATA_SOURCE_ROOT, "microcosm.data"),
        (_FRAME_SOURCE_ROOT, "microcosm.frame"),
    ):
        source_root = repository_root / relative_root
        for path in sorted(source_root.rglob("*.py")):
            modules[_module_name(path, source_root, prefix)] = path

    modules["tools.build_us_multispine_pool"] = repository_root / _POOL_TOOL
    return dict(sorted(modules.items()))


def discover_exempted_source_modules(
    repository_root: Path,
) -> dict[SourceNamespaceExemption, tuple[str, ...]]:
    """Resolve every typed country-namespace exclusion against the filesystem."""

    matches = {row: [] for row in SOURCE_NAMESPACE_EXEMPTIONS}
    build_root = repository_root / _BUILD_SOURCE_ROOT
    for path in sorted(build_root.rglob("*.py")):
        module = _module_name(path, build_root, "microcosm.build")
        owners = [
            row
            for row in SOURCE_NAMESPACE_EXEMPTIONS
            if _has_module_prefix(module, row.prefix)
        ]
        if len(owners) > 1:
            raise AssertionError(
                f"source module {module!r} has overlapping namespace exemptions"
            )
        if owners:
            matches[owners[0]].append(module)
    return {row: tuple(modules) for row, modules in matches.items()}


def _attribute_name(node: ast.expr, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return f"{_attribute_name(node.func, aliases)}()"
    return ""


_NUMPY_GENERATOR_METHODS = frozenset(
    {
        "beta",
        "binomial",
        "bytes",
        "chisquare",
        "choice",
        "dirichlet",
        "exponential",
        "f",
        "gamma",
        "geometric",
        "gumbel",
        "hypergeometric",
        "integers",
        "laplace",
        "logistic",
        "lognormal",
        "logseries",
        "multinomial",
        "multivariate_hypergeometric",
        "multivariate_normal",
        "negative_binomial",
        "noncentral_chisquare",
        "noncentral_f",
        "normal",
        "pareto",
        "permutation",
        "permuted",
        "poisson",
        "power",
        "random",
        "rayleigh",
        "shuffle",
        "spawn",
        "standard_cauchy",
        "standard_exponential",
        "standard_gamma",
        "standard_normal",
        "standard_t",
        "triangular",
        "uniform",
        "vonmises",
        "wald",
        "weibull",
        "zipf",
    }
)
_NUMPY_GENERATOR_CONTROL_METHODS: frozenset[str] = frozenset()
_NUMPY_RANDOMSTATE_CONTROL_METHODS = frozenset({"get_state", "seed", "set_state"})
_NUMPY_RANDOMSTATE_DRAW_METHODS = frozenset(
    {
        *(
            _NUMPY_GENERATOR_METHODS
            - {"integers", "multivariate_hypergeometric", "permuted", "spawn"}
        ),
        "rand",
        "randint",
        "randn",
        "random_integers",
        "random_sample",
        "tomaxint",
    }
)
_NUMPY_RANDOMSTATE_METHODS = frozenset(
    _NUMPY_RANDOMSTATE_DRAW_METHODS | _NUMPY_RANDOMSTATE_CONTROL_METHODS
)
_PYTHON_RANDOM_CONTROL_METHODS = frozenset({"getstate", "seed", "setstate"})
_PYTHON_RANDOM_DRAW_METHODS = frozenset(
    {
        "betavariate",
        "binomialvariate",
        "choice",
        "choices",
        "expovariate",
        "gammavariate",
        "gauss",
        "getrandbits",
        "lognormvariate",
        "normalvariate",
        "paretovariate",
        "randbytes",
        "randint",
        "random",
        "randrange",
        "sample",
        "shuffle",
        "triangular",
        "uniform",
        "vonmisesvariate",
        "weibullvariate",
    }
)
_PYTHON_RANDOM_METHODS = frozenset(
    _PYTHON_RANDOM_DRAW_METHODS | _PYTHON_RANDOM_CONTROL_METHODS
)
_RNG_DRAW_METHODS = frozenset(
    _NUMPY_GENERATOR_METHODS | _NUMPY_RANDOMSTATE_METHODS | _PYTHON_RANDOM_METHODS
)
_HASHLIB_APIS = frozenset(
    {
        "hashlib.blake2b",
        "hashlib.blake2s",
        "hashlib.md5",
        "hashlib.sha1",
        "hashlib.sha224",
        "hashlib.sha256",
        "hashlib.sha384",
        "hashlib.sha512",
        "hashlib.sha3_224",
        "hashlib.sha3_256",
        "hashlib.sha3_384",
        "hashlib.sha3_512",
        "hashlib.shake_128",
        "hashlib.shake_256",
        "hashlib.new",
        "hashlib.file_digest",
        "hashlib.pbkdf2_hmac",
        "hashlib.scrypt",
    }
)
_UUID_RANDOM_APIS = frozenset({"uuid.uuid4"})
_TORCH_RANDOM_APIS = frozenset(
    {
        "bernoulli",
        "manual_seed",
        "multinomial",
        "normal",
        "rand",
        "randint",
        "randn",
        "randperm",
        "seed",
    }
)
_TORCH_IN_PLACE_RANDOM_APIS = frozenset(
    {"bernoulli_", "exponential_", "normal_", "random_", "uniform_"}
)


@dataclass(slots=True)
class _LexicalFacts:
    aliases: dict[str, str]
    namespace_aliases: dict[str, str]
    callable_aliases: dict[str, str]
    random_receivers: dict[str, str]
    seed_sequence_receivers: set[str]

    @classmethod
    def empty(cls) -> _LexicalFacts:
        return cls({}, {}, {}, {}, set())

    def clone(self) -> _LexicalFacts:
        return _LexicalFacts(
            dict(self.aliases),
            dict(self.namespace_aliases),
            dict(self.callable_aliases),
            dict(self.random_receivers),
            set(self.seed_sequence_receivers),
        )


class _CallsiteVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: str,
        tree: ast.AST,
        *,
        _precollected: dict[tuple[str, ...], _LexicalFacts] | None = None,
        _collecting: bool = False,
    ) -> None:
        self.module = module
        if _precollected is None:
            previous: dict[tuple[str, ...], _LexicalFacts] = {}
            for _ in range(12):
                collector = _CallsiteVisitor(
                    module,
                    tree,
                    _precollected=previous,
                    _collecting=True,
                )
                collector.visit(tree)
                current = collector.collected_scope_facts
                if current == previous:
                    break
                previous = current
            else:  # pragma: no cover - finite symbol graph must converge quickly
                raise AssertionError("stochastic alias prepass did not converge")
            self.precollected_scope_facts = previous
        else:
            self.precollected_scope_facts = _precollected
        self.collecting = _collecting
        self.collected_scope_facts: dict[tuple[str, ...], _LexicalFacts] = {}
        self._facts_stack = [self._seed_scope(())]
        self._scope_kinds = ["module"]
        self.qualnames: list[str] = []
        self.parents: dict[ast.AST, ast.AST] = {}
        self.calls: list[DiscoveredCallsite] = []
        self._occurrences: dict[tuple[str, str], int] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent

    @property
    def _facts(self) -> _LexicalFacts:
        return self._facts_stack[-1]

    @property
    def aliases(self) -> dict[str, str]:
        return self._facts.aliases

    @property
    def callable_aliases(self) -> dict[str, str]:
        return self._facts.callable_aliases

    @property
    def namespace_aliases(self) -> dict[str, str]:
        return self._facts.namespace_aliases

    @property
    def random_receivers(self) -> dict[str, str]:
        return self._facts.random_receivers

    @property
    def seed_sequence_receivers(self) -> set[str]:
        return self._facts.seed_sequence_receivers

    def _seed_scope(self, key: tuple[str, ...]) -> _LexicalFacts:
        inherited = (
            self._facts_stack[-1].clone()
            if hasattr(self, "_facts_stack")
            else _LexicalFacts.empty()
        )
        seeded = self.precollected_scope_facts.get(key)
        if seeded is not None:
            inherited.aliases.update(seeded.aliases)
            inherited.namespace_aliases.update(seeded.namespace_aliases)
            inherited.callable_aliases.update(seeded.callable_aliases)
            inherited.random_receivers.update(seeded.random_receivers)
            inherited.seed_sequence_receivers.update(seeded.seed_sequence_receivers)
        return inherited

    def _enter_scope(self, name: str, kind: str) -> None:
        self.qualnames.append(name)
        self._facts_stack.append(self._seed_scope(tuple(self.qualnames)))
        self._scope_kinds.append(kind)

    def _leave_scope(self) -> None:
        if self._scope_kinds[-1] == "function":
            self._promote_function_class_attributes()
        self.collected_scope_facts[tuple(self.qualnames)] = self._facts.clone()
        self._facts_stack.pop()
        self._scope_kinds.pop()
        self.qualnames.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.generic_visit(node)
        self.collected_scope_facts[()] = self._facts.clone()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter_scope(node.name, "class")
        self.generic_visit(node)
        self._leave_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_scope(node.name, "function")
        self._register_random_parameters(node.args)
        self.generic_visit(node)
        self._leave_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._enter_scope(node.name, "function")
        self._register_random_parameters(node.args)
        self.generic_visit(node)
        self._leave_scope()

    def _record(self, api: str, line: int) -> None:
        if self.collecting:
            return
        qualname = ".".join(self.qualnames) or "<module>"
        key = (qualname, api)
        occurrence = self._occurrences.get(key, 0)
        self._occurrences[key] = occurrence + 1
        self.calls.append(
            DiscoveredCallsite(
                PhysicalCallsite(self.module, qualname, api, occurrence),
                line,
            )
        )

    def _resolved_name(self, node: ast.expr) -> str:
        source_name = self._source_name(node)
        if source_name in self.callable_aliases:
            return self.callable_aliases[source_name]
        if source_name in self.namespace_aliases:
            return self.namespace_aliases[source_name]
        if isinstance(node, ast.Call):
            return f"{self._resolved_name(node.func)}()"
        if isinstance(node, ast.Attribute):
            base = self._resolved_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return _attribute_name(node, {**self.aliases, **self.callable_aliases})

    @staticmethod
    def _source_name(node: ast.expr) -> str:
        return _attribute_name(node, {})

    def _annotation_family(self, annotation: ast.expr | None) -> str | None:
        if annotation is None:
            return None
        names = {
            self._resolved_name(candidate)
            for candidate in ast.walk(annotation)
            if isinstance(candidate, (ast.Name, ast.Attribute))
        }
        if any(name.endswith("RandomState") for name in names):
            return "numpy.random.RandomState"
        if any(name.endswith("Generator") for name in names):
            return "numpy.random.Generator"
        if any(name.endswith("random.Random") for name in names):
            return "random.Random"
        return None

    def _register_random_parameters(self, arguments: ast.arguments) -> None:
        positional = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        for argument in positional:
            self._kill_name(argument.arg)
            family = self._annotation_family(argument.annotation)
            if family is not None:
                self.random_receivers[argument.arg] = family
        for argument in (arguments.vararg, arguments.kwarg):
            if argument is None:
                continue
            self._kill_name(argument.arg)
            family = self._annotation_family(argument.annotation)
            if family is not None:
                self.random_receivers[argument.arg] = family

    def _rng_receiver_family(self, node: ast.expr) -> str | None:
        source_name = self._source_name(node)
        if source_name in self.random_receivers:
            return self.random_receivers[source_name]
        name = self._resolved_name(node)
        if name.startswith("numpy.random.RandomState("):
            return "numpy.random.RandomState"
        if name.startswith("random.Random("):
            return "random.Random"
        if name.startswith("numpy.random.default_rng(") or name.startswith(
            "numpy.random.Generator("
        ):
            return "numpy.random.Generator"
        return None

    def _is_seed_sequence_receiver(self, node: ast.expr) -> bool:
        source_name = self._source_name(node)
        if source_name in self.seed_sequence_receivers:
            return True
        name = self._resolved_name(node)
        terminal = name.removesuffix("()").rsplit(".", 1)[-1].lower()
        return (
            terminal == "seed"
            or terminal.endswith("_seed")
            or terminal.endswith("seed_sequence")
            or "seedsequence" in name.lower()
        )

    @staticmethod
    def _assignment_targets(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(node.elts)
        return (node,)

    def _callable_alias_api(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Call):
            return None
        name = self._resolved_name(value)
        if name in _HASHLIB_APIS or name in _UUID_RANDOM_APIS:
            return name
        if name in {"hashlib", "numpy.random", "random", "uuid"}:
            return None
        if name.startswith("stochastic.unresolved."):
            return name
        if name.startswith("numpy.random.") or name.startswith("random."):
            return name
        if isinstance(value, ast.Attribute):
            leaf = value.attr
            if leaf in _RNG_DRAW_METHODS:
                family = self._rng_receiver_family(value.value)
                # Deliberately conservative: an arbitrary receiver name cannot
                # hide a stochastic-looking method behind a bound-method alias.
                return f"{family}.{leaf}" if family else f"stochastic.unresolved.{leaf}"
        return None

    def _namespace_alias_api(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Call):
            return None
        name = self._resolved_name(value)
        if name in {"hashlib", "numpy.random", "random", "uuid"}:
            return name
        return None

    @staticmethod
    def _family_methods(family: str) -> frozenset[str]:
        if family == "numpy.random.Generator":
            return _NUMPY_GENERATOR_METHODS
        if family == "numpy.random.RandomState":
            return _NUMPY_RANDOMSTATE_METHODS
        if family == "random.Random":
            return _PYTHON_RANDOM_METHODS
        return frozenset()

    def _value_random_family(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Call):
            called = self._resolved_name(value.func).replace("()", "")
            if called in {
                "numpy.random.default_rng",
                "numpy.random.Generator",
            }:
                return "numpy.random.Generator"
            if called == "numpy.random.RandomState":
                return "numpy.random.RandomState"
            if called == "random.Random":
                return "random.Random"
        return self._rng_receiver_family(value)

    def _bind_assignment(self, target: ast.expr, value: ast.expr) -> None:
        for item in self._assignment_targets(target):
            target_name = self._source_name(item)
            if not target_name:
                continue
            family = self._value_random_family(value)
            callable_api = self._callable_alias_api(value)
            namespace_api = self._namespace_alias_api(value)
            is_seed_sequence = False
            if isinstance(value, ast.Call):
                called = self._resolved_name(value.func).replace("()", "")
                if called == "numpy.random.SeedSequence":
                    is_seed_sequence = True
            elif self._source_name(value) in self.seed_sequence_receivers:
                is_seed_sequence = True

            prior_family = self.random_receivers.get(target_name)
            prior_callable = self.callable_aliases.get(target_name)
            prior_namespace = self.namespace_aliases.get(target_name)
            prior_seed = target_name in self.seed_sequence_receivers
            self._kill_name(target_name)
            if family is not None:
                self.random_receivers[target_name] = family
            elif prior_family is not None:
                self.random_receivers[target_name] = "stochastic.unresolved"
            if callable_api is not None:
                self.callable_aliases[target_name] = callable_api
            elif prior_callable is not None and namespace_api is None:
                self.callable_aliases[target_name] = "stochastic.unresolved.bound_alias"
            if namespace_api is not None:
                self.namespace_aliases[target_name] = namespace_api
            elif prior_namespace is not None:
                self.namespace_aliases[target_name] = "stochastic.unresolved.namespace"
            if is_seed_sequence:
                self.seed_sequence_receivers.add(target_name)
            elif prior_seed:
                self.callable_aliases[target_name] = "stochastic.unresolved.seed_alias"

    def _kill_name(self, name: str) -> None:
        self.aliases.pop(name, None)
        self.namespace_aliases.pop(name, None)
        self.callable_aliases.pop(name, None)
        self.random_receivers.pop(name, None)
        self.seed_sequence_receivers.discard(name)

    @staticmethod
    def _join_promoted_value(
        destination: dict[str, str],
        name: str,
        incoming: str,
        unresolved: str,
    ) -> None:
        existing = destination.get(name)
        destination[name] = (
            incoming if existing is None or existing == incoming else unresolved
        )

    def _promote_function_class_attributes(self) -> None:
        for index in range(len(self._scope_kinds) - 1, -1, -1):
            if self._scope_kinds[index] != "class":
                continue
            class_facts = self._facts_stack[index]
            for name, family in self.random_receivers.items():
                if name.startswith("self."):
                    self._join_promoted_value(
                        class_facts.random_receivers,
                        name,
                        family,
                        "stochastic.unresolved",
                    )
            for name, api in self.callable_aliases.items():
                if name.startswith("self."):
                    self._join_promoted_value(
                        class_facts.callable_aliases,
                        name,
                        api,
                        "stochastic.unresolved.bound_alias",
                    )
            for name, namespace in self.namespace_aliases.items():
                if name.startswith("self."):
                    self._join_promoted_value(
                        class_facts.namespace_aliases,
                        name,
                        namespace,
                        "stochastic.unresolved.namespace",
                    )
            class_facts.seed_sequence_receivers.update(
                name
                for name in self.seed_sequence_receivers
                if name.startswith("self.")
            )
            return

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.generic_visit(node)
        for target in node.targets:
            self._bind_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self.generic_visit(node)
        target_name = self._source_name(node.target)
        family = self._annotation_family(node.annotation)
        if target_name and family is not None:
            self.random_receivers[target_name] = family
            if self._scope_kinds[-1] == "class" and isinstance(node.target, ast.Name):
                self.random_receivers[f"self.{target_name}"] = family
        if node.value is not None:
            self._bind_assignment(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.generic_visit(node)
        self._bind_assignment(node.target, node.value)

    @staticmethod
    def _merge_branch_facts(left: _LexicalFacts, right: _LexicalFacts) -> _LexicalFacts:
        aliases = {
            name: value
            for name, value in left.aliases.items()
            if right.aliases.get(name) == value
        }

        def merge_mapping(
            first: dict[str, str], second: dict[str, str], unresolved: str
        ) -> dict[str, str]:
            merged: dict[str, str] = {}
            for name in first.keys() | second.keys():
                first_value = first.get(name)
                second_value = second.get(name)
                merged[name] = (
                    first_value
                    if first_value is not None and first_value == second_value
                    else unresolved
                )
            return merged

        namespace_aliases = merge_mapping(
            left.namespace_aliases,
            right.namespace_aliases,
            "stochastic.unresolved.namespace",
        )
        callable_aliases = merge_mapping(
            left.callable_aliases,
            right.callable_aliases,
            "stochastic.unresolved.bound_alias",
        )
        random_receivers = merge_mapping(
            left.random_receivers,
            right.random_receivers,
            "stochastic.unresolved",
        )
        seed_receivers = left.seed_sequence_receivers & right.seed_sequence_receivers
        for name in left.seed_sequence_receivers ^ right.seed_sequence_receivers:
            callable_aliases[name] = "stochastic.unresolved.seed_alias"
        return _LexicalFacts(
            aliases,
            namespace_aliases,
            callable_aliases,
            random_receivers,
            seed_receivers,
        )

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self.visit(node.test)
        before = self._facts.clone()
        self._facts_stack[-1] = before.clone()
        for statement in node.body:
            self.visit(statement)
        body_facts = self._facts.clone()
        self._facts_stack[-1] = before.clone()
        for statement in node.orelse:
            self.visit(statement)
        else_facts = self._facts.clone()
        self._facts_stack[-1] = self._merge_branch_facts(body_facts, else_facts)

    def _is_direct_call_or_alias_value(self, node: ast.expr) -> bool:
        parent = self.parents.get(node)
        return (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            or isinstance(parent, ast.Call)
            and parent.func is node
            or isinstance(parent, ast.Assign)
            and parent.value is node
            or isinstance(parent, ast.AnnAssign)
            and parent.value is node
            or isinstance(parent, ast.NamedExpr)
            and parent.value is node
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if not self._is_direct_call_or_alias_value(node):
            name = self._resolved_name(node)
            family = self._rng_receiver_family(node.value)
            if name in _HASHLIB_APIS:
                self._record(f"{name}[callable_reference]", node.lineno)
            elif (
                name in _UUID_RANDOM_APIS
                or family is not None
                and node.attr in self._family_methods(family)
            ):
                self._record("stochastic.unresolved.bound_method_escape", node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and not self._is_direct_call_or_alias_value(
            node
        ):
            api = self.callable_aliases.get(node.id)
            if api is not None and (
                api.startswith("numpy.random.")
                or api.startswith("random.")
                or api.startswith("hashlib.")
                or api.startswith("uuid.")
                or api.startswith("stochastic.unresolved.")
            ):
                self._record("stochastic.unresolved.bound_method_escape", node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolved_name(node.func)
        leaf = name.removesuffix("()").rsplit(".", maxsplit=1)[-1]
        parent = self.parents.get(node)

        if name == "getattr" and len(node.args) >= 2:
            receiver_family = self._rng_receiver_family(node.args[0])
            attribute = node.args[1]
            attribute_name = (
                attribute.value if isinstance(attribute, ast.Constant) else None
            )
            if receiver_family is not None or attribute_name in _RNG_DRAW_METHODS:
                self._record("stochastic.unresolved.dynamic_getattr", node.lineno)
        elif name in _HASHLIB_APIS:
            self._record(name, node.lineno)
        elif name in _UUID_RANDOM_APIS:
            self._record(name, node.lineno)
        elif leaf == "hash_pandas_object":
            self._record("pandas.util.hash_pandas_object", node.lineno)
        elif name.startswith("numpy.random.default_rng()."):
            self._record(
                f"numpy.random.Generator.{name.rsplit('.', 1)[-1]}", node.lineno
            )
        elif name.startswith("numpy.random."):
            self._record(name.replace("()", ""), node.lineno)
        elif (
            leaf in {"generate_state", "spawn"}
            and isinstance(node.func, ast.Attribute)
            and self._is_seed_sequence_receiver(node.func.value)
        ):
            self._record(f"numpy.random.SeedSequence.{leaf}", node.lineno)
        elif name.startswith("random."):
            self._record(name.replace("()", ""), node.lineno)
        elif (
            isinstance(node.func, ast.Attribute)
            and ((family := self._rng_receiver_family(node.func.value)) is not None)
            and leaf in self._family_methods(family)
        ):
            self._record(f"{family}.{leaf}", node.lineno)
        elif leaf == "sample" and not name.startswith("torch."):
            mode = (
                "random_state"
                if any(keyword.arg == "random_state" for keyword in node.keywords)
                else "ambient"
            )
            self._record(f"pandas.DataFrame.sample[{mode}]", node.lineno)
        elif any(keyword.arg == "random_state" for keyword in node.keywords):
            self._record(f"{leaf}[random_state]", node.lineno)
        elif name.startswith("torch.") and leaf in _TORCH_RANDOM_APIS:
            self._record(f"torch.{leaf}", node.lineno)
        elif name.startswith("torch.") and leaf in _TORCH_IN_PLACE_RANDOM_APIS:
            self._record(f"torch.Tensor.{leaf}", node.lineno)
        elif name.startswith("torch.") and leaf == "sample":
            self._record("torch.distributions.Distribution.sample", node.lineno)
        elif name.startswith("stochastic.unresolved."):
            self._record(name, node.lineno)
        elif leaf in _RNG_DRAW_METHODS and isinstance(node.func, ast.Attribute):
            self._record(f"stochastic.unresolved.{leaf}", node.lineno)
        elif leaf in {"QRF", "RegimeGatedQRF"}:
            self._record("microcosm.fit.QRF", node.lineno)
        elif isinstance(node.func, ast.Call) and (
            _attribute_name(node.func.func, self.aliases).rsplit(".", 1)[-1] == "_qrf"
        ):
            self._record("microcosm.fit.QRF[dynamic]", node.lineno)
        elif leaf == "_qrf" and not (
            isinstance(parent, ast.Call) and parent.func is node
        ):
            self._record("microcosm.fit.QRF[dynamic]", node.lineno)
        self.generic_visit(node)


def discover_production_callsites(
    repository_root: Path,
) -> tuple[DiscoveredCallsite, ...]:
    """Discover every stochastic/hash API call in the production universe."""

    result: list[DiscoveredCallsite] = []
    for module, path in discover_production_source_modules(repository_root).items():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        visitor = _CallsiteVisitor(module, tree)
        visitor.visit(tree)
        result.extend(visitor.calls)
    return tuple(sorted(result, key=lambda row: row.callsite))


def _bind(
    module: str,
    qualname: str,
    api: str,
    *site_ids: str,
    occurrence: int = 0,
) -> CallsiteBinding:
    return CallsiteBinding(
        _key(module, qualname, api, occurrence), tuple(sorted(site_ids))
    )


def _bindings(
    module: str,
    qualname: str,
    apis: tuple[str, ...],
    *site_ids: str,
) -> tuple[CallsiteBinding, ...]:
    return tuple(_bind(module, qualname, api, *site_ids) for api in apis)


# Independent logical ids used only by the physical manifest.  These are not
# imported from seeds.py; exact equality with that ledger is asserted by tests.
_QRF_ALL_SITE_IDS = tuple(
    sorted(
        {
            "acs_qrf_fit_draw",
            "acs_rent_qrf_model",
            "child_support_puf_qrf_model",
            "childcare_puf_qrf_model",
            "disability_benefits_puf_qrf_model",
            "energy_subsidy_puf_qrf_model",
            "housing_assistance_puf_qrf_model",
            "org_wages_qrf_model",
            "other_health_insurance_puf_qrf_model",
            "primary_puf_monolithic_qrf_model",
            "primary_qrf_fit_draw",
            "prior_year_income_puf_qrf_model",
            "retirement_contributions_puf_qrf_model",
            "retirement_distributions_puf_qrf_model",
            "scf_auto_loan_qrf_model",
            "scf_financial_asset_qrf_model",
            "scf_net_worth_qrf_model",
            "sipp_financial_asset_qrf_models",
            "sipp_head_start_qrf_model",
            "sipp_tip_qrf_model",
            "sipp_vehicle_qrf_model",
            "ssi_archived_qrf_model",
            "voluntary_filing_qrf_model",
            "weeks_unemployed_puf_qrf_model",
            "workers_compensation_puf_qrf_model",
        }
    )
)
_QRF_MONOLITHIC_SITE_IDS = tuple(
    site_id for site_id in _QRF_ALL_SITE_IDS if site_id != "primary_qrf_fit_draw"
)
_QRF_CHAIN_SITE_IDS = ("acs_qrf_fit_draw", "primary_qrf_fit_draw")

LEGACY_V1_PRODUCTION_BINDINGS = (
    *_bindings(
        "microcosm.build.frame_sampling",
        "sample_frame_households",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "survey_sample_acs",
        "survey_sample_asec",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_support",
        "_attach_clone_arm_to_seeded_sample",
        ("stochastic.unresolved.choice", "numpy.random.default_rng"),
        "puf_clone_attachment",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_source_agi",
        "_assign_weights",
        ("numpy.random.Generator.choice",),
        "puf_archived_aggregate_disaggregation",
    ),
    _bind(
        "microcosm.build.us_runtime.puf_source_agi",
        "_assign_weights",
        "numpy.random.Generator.choice",
        "puf_archived_aggregate_disaggregation",
        occurrence=1,
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_source_agi",
        "_sample_bucket_donors",
        ("numpy.random.Generator.choice",),
        "puf_archived_aggregate_disaggregation",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_source_agi",
        "source_year_puf_adjusted_gross_income",
        ("numpy.random.default_rng",),
        "puf_archived_aggregate_disaggregation",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_aggregate_records",
        "disaggregate_puf_aggregate_records",
        ("numpy.random.default_rng",),
        "puf_live_aggregate_disaggregation",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_aggregate_records",
        "_assign_s006_values",
        ("numpy.random.Generator.choice",),
        "puf_live_aggregate_disaggregation",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_aggregate_records",
        "_sample_bucket_donors",
        ("numpy.random.Generator.choice",),
        "puf_live_aggregate_disaggregation",
    ),
    *_bindings(
        "microcosm.build.us_runtime.ssi_disability_criteria",
        "load_sipp_2023_ssi_disability_donor",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "ssi_weighted_replacement_training",
    ),
    *_bindings(
        "microcosm.build.us_runtime.ssi_disability_criteria",
        "_weighted_replacement_sample",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "ssi_weighted_replacement_training",
    ),
    *_bindings(
        "microcosm.build.us_runtime.sipp_vehicles",
        "_sample_rng",
        ("numpy.random.default_rng",),
        "sipp_vehicle_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_vehicles",
        "_cap_vehicle_training_sample",
        "stochastic.unresolved.choice",
        "sipp_vehicle_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_vehicles",
        "_cap_vehicle_training_sample",
        "stochastic.unresolved.choice",
        "sipp_vehicle_training_cap",
        occurrence=1,
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_vehicles",
        "impute_us_sipp_vehicles",
        "RandomForestClassifier[random_state]",
        "sipp_vehicle_count_random_forest_model",
    ),
    *_bindings(
        "microcosm.build.us_runtime.sipp_financial_assets",
        "_sample_rng",
        ("numpy.random.default_rng",),
        "sipp_financial_asset_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_financial_assets",
        "_target_balanced_cap",
        "stochastic.unresolved.choice",
        "sipp_financial_asset_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_financial_assets",
        "_target_balanced_cap",
        "stochastic.unresolved.choice",
        "sipp_financial_asset_training_cap",
        occurrence=1,
    ),
    *_bindings(
        "microcosm.build.us_runtime.sipp_financial_assets",
        "impute_us_sipp_financial_assets",
        (
            "numpy.random.SeedSequence",
            "numpy.random.SeedSequence.generate_state",
            "numpy.random.SeedSequence.spawn",
        ),
        "sipp_financial_asset_qrf_models",
    ),
    *_bindings(
        "microcosm.build.us_runtime.housing_inputs",
        "_archived_training_rng",
        ("numpy.random.default_rng",),
        "acs_rent_archived_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.housing_inputs",
        "_archived_joint_training_sample",
        "stochastic.unresolved.choice",
        "acs_rent_archived_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.housing_inputs",
        "_archived_joint_training_sample",
        "stochastic.unresolved.choice",
        "acs_rent_archived_training_cap",
        occurrence=1,
    ),
    *_bindings(
        "microcosm.build.us_runtime.sipp_tips",
        "impute_us_sipp_tips",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "sipp_tip_training_cap",
    ),
    *_bindings(
        "microcosm.build.us_runtime.scf_wealth",
        "financial_asset_source_is_scf",
        (
            "numpy.random.Generator.random",
            "numpy.random.SeedSequence",
            "numpy.random.default_rng",
        ),
        "scf_household_source_selector",
    ),
    *_bindings(
        "microcosm.build.us_runtime.adult_care",
        "derive_us_adult_care_from_manifest",
        ("numpy.random.Generator.permutation", "numpy.random.default_rng"),
        "adult_care_weighted_prefix_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.adult_care",
        "derive_us_adult_care_from_manifest",
        "stochastic.unresolved.permutation",
        "adult_care_weighted_prefix_assignment",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puf_capital_gains_tail",
        "_recipient_candidates",
        ("stochastic.unresolved.random", "numpy.random.default_rng"),
        "capital_gains_tail_random_rank",
    ),
    *_bindings(
        "microcosm.calibrate.exact_k",
        "select_exact_k",
        ("numpy.random.Generator", "numpy.random.PCG64"),
        "exact_k_pcg64_selection",
    ),
    _bind(
        "microcosm.calibrate.exact_k",
        "_categorical",
        "numpy.random.Generator.random",
        "exact_k_pcg64_selection",
    ),
    _bind(
        "microcosm.calibrate.exact_k",
        "_sampford_core",
        "numpy.random.Generator.random",
        "exact_k_pcg64_selection",
    ),
    _bind(
        "microcosm.calibrate.exact_k",
        "_sampford_core",
        "numpy.random.Generator.random",
        "exact_k_pcg64_selection",
        occurrence=1,
    ),
    _bind(
        "microcosm.calibrate.exact_k",
        "_sampford_dynamic_programming",
        "numpy.random.Generator.random",
        "exact_k_pcg64_selection",
    ),
    _bind(
        "microcosm.calibrate.gates",
        "HardConcrete.forward",
        "torch.Tensor.uniform_",
        "torch_calibration_reseed",
    ),
    _bind(
        "microcosm.calibrate.solve",
        "_search_l0_lambda_for_budget.evaluate",
        "torch.manual_seed",
        "torch_calibration_reseed",
    ),
    _bind(
        "microcosm.calibrate.solve",
        "calibrate",
        "torch.manual_seed",
        "torch_calibration_reseed",
    ),
    *_bindings(
        "microcosm.build.us_runtime.geography_ladder",
        "assign_us_geography_ladder",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "legacy_geography_ladder",
    ),
    *_bindings(
        "microcosm.build.us_runtime.puma_ladder",
        "assign_us_puma_ladder",
        ("numpy.random.default_rng",),
        "legacy_puma_ladder",
    ),
    _bind(
        "microcosm.build.us_runtime.puma_ladder",
        "_draw_puma_within_state",
        "numpy.random.Generator.choice",
        "legacy_puma_ladder",
    ),
    _bind(
        "microcosm.build.us_runtime.puma_ladder",
        "_draw_layer_values",
        "numpy.random.Generator.choice",
        "legacy_puma_ladder",
    ),
    *_bindings(
        "microcosm.build.us_runtime.congressional_district_geography",
        "assign_congressional_districts_to_households",
        ("numpy.random.Generator.choice", "numpy.random.default_rng"),
        "legacy_congressional_district_assignment",
    ),
    # Pandas MT19937 training caps.
    _bind(
        "microcosm.build.us_runtime.prior_year_income",
        "impute_us_prior_year_income_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "prior_year_income_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.childcare",
        "impute_us_childcare_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "childcare_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.retirement_contributions",
        "impute_us_retirement_contributions_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "retirement_contributions_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.disability_benefits",
        "impute_us_disability_benefits_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "disability_benefits_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.housing_inputs",
        "impute_us_housing_assistance_to_puf_support",
        "pandas.DataFrame.sample[random_state]",
        "housing_inputs_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.workers_compensation",
        "impute_us_workers_compensation_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "workers_compensation_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.retirement_distributions",
        "impute_us_retirement_distributions_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "retirement_distributions_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.child_support",
        "impute_us_child_support_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "child_support_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.energy_subsidy",
        "impute_us_energy_subsidy_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "energy_subsidy_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.other_health_insurance",
        "impute_us_other_health_insurance_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "other_health_insurance_training_cap",
    ),
    _bind(
        "microcosm.build.us_runtime.weeks_unemployed",
        "impute_us_weeks_unemployed_to_puf_support_from_manifest",
        "pandas.DataFrame.sample[random_state]",
        "weeks_unemployed_training_cap",
    ),
    # QRF public invocation sites.
    _bind(
        "microcosm.build.us_runtime.acs_transfer",
        "_fit_family_patterns",
        "microcosm.fit.QRF[dynamic]",
        "acs_qrf_fit_draw",
    ),
    _bind(
        "microcosm.build.us_runtime.acs_transfer",
        "_fit_family_patterns_banked",
        "microcosm.fit.QRF[dynamic]",
        "acs_qrf_fit_draw",
    ),
    _bind(
        "microcosm.build.us_runtime.acs_transfer",
        "_fit_family_patterns_banked",
        "microcosm.fit.QRF[dynamic]",
        "acs_qrf_fit_draw",
        occurrence=1,
    ),
    _bind(
        "microcosm.build.us_runtime.puf_qrf_chain",
        "initialize_primary_puf_qrf_chain",
        "microcosm.fit.QRF",
        "primary_qrf_fit_draw",
    ),
    _bind(
        "microcosm.build.us_runtime.puf_qrf_chain",
        "run_primary_puf_qrf_target",
        "microcosm.fit.QRF",
        "primary_qrf_fit_draw",
    ),
    _bind(
        "microcosm.build.us_runtime.puf_support",
        "impute_us_puf_tax_detail_support",
        "microcosm.fit.QRF",
        "primary_puf_monolithic_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.ssi_disability_criteria",
        "impute_us_ssi_disability_criteria",
        "microcosm.fit.QRF",
        "ssi_archived_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_vehicles",
        "impute_us_sipp_vehicles",
        "microcosm.fit.QRF",
        "sipp_vehicle_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_financial_assets",
        "impute_us_sipp_financial_assets",
        "microcosm.fit.QRF",
        "sipp_financial_asset_qrf_models",
    ),
    _bind(
        "microcosm.build.us_runtime.scf_wealth",
        "_draw_us_scf_targets",
        "microcosm.fit.QRF",
        "scf_financial_asset_qrf_model",
        "scf_net_worth_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.scf_auto_loans",
        "impute_us_scf_auto_loans",
        "microcosm.fit.QRF",
        "scf_auto_loan_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.child_support",
        "impute_us_child_support_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "child_support_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.childcare",
        "impute_us_childcare_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "childcare_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.disability_benefits",
        "impute_us_disability_benefits_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "disability_benefits_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.energy_subsidy",
        "impute_us_energy_subsidy_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "energy_subsidy_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.housing_inputs",
        "impute_us_pre_subsidy_rent",
        "microcosm.fit.QRF",
        "acs_rent_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.housing_inputs",
        "impute_us_housing_assistance_to_puf_support",
        "microcosm.fit.QRF",
        "housing_assistance_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.org_wages",
        "impute_us_org_wages",
        "microcosm.fit.QRF",
        "org_wages_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.other_health_insurance",
        "impute_us_other_health_insurance_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "other_health_insurance_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.prior_year_income",
        "impute_us_prior_year_income_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "prior_year_income_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.retirement_contributions",
        "impute_us_retirement_contributions_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "retirement_contributions_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.retirement_distributions",
        "impute_us_retirement_distributions_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "retirement_distributions_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_head_start",
        "impute_us_sipp_head_start",
        "microcosm.fit.QRF",
        "sipp_head_start_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.sipp_tips",
        "impute_us_sipp_tips",
        "microcosm.fit.QRF",
        "sipp_tip_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.voluntary_filing",
        "impute_us_voluntary_filing",
        "microcosm.fit.QRF",
        "voluntary_filing_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.weeks_unemployed",
        "impute_us_weeks_unemployed_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "weeks_unemployed_puf_qrf_model",
    ),
    _bind(
        "microcosm.build.us_runtime.workers_compensation",
        "impute_us_workers_compensation_to_puf_support_from_manifest",
        "microcosm.fit.QRF",
        "workers_compensation_puf_qrf_model",
    ),
    # Shared QRF implementation: exact stream subsets by public path.
    *_bindings(
        "microcosm.fit.qrf",
        "RegimeGatedQRF.fit",
        (
            "numpy.random.SeedSequence",
            "numpy.random.SeedSequence.spawn",
            "numpy.random.default_rng",
        ),
        *_QRF_MONOLITHIC_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "RegimeGatedQRF.start_chain",
        (
            "numpy.random.SeedSequence",
            "numpy.random.SeedSequence.spawn",
            "numpy.random.default_rng",
        ),
        *_QRF_CHAIN_SITE_IDS,
    ),
    _bind(
        "microcosm.fit.qrf",
        "RegimeGatedQRF.start_chain",
        "numpy.random.default_rng",
        *_QRF_CHAIN_SITE_IDS,
        occurrence=1,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_rng_from_state_json",
        ("numpy.random.Generator", "numpy.random.PCG64"),
        *_QRF_CHAIN_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "FittedRegimeGatedQRF.__init__",
        ("numpy.random.default_rng",),
        *_QRF_MONOLITHIC_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "RegimeGatedQRF._fit_gate",
        ("numpy.random.Generator.integers",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "RegimeGatedQRF._fit_target.forest",
        ("numpy.random.Generator.integers",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_weighted_bootstrap",
        ("numpy.random.Generator.choice",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_make_gate",
        ("HistGradientBoostingClassifier[random_state]",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_fit_forest",
        ("RandomForestQuantileRegressor[random_state]",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_qrf_row_quantiles",
        ("numpy.random.Generator.random",),
        *_QRF_ALL_SITE_IDS,
    ),
    *_bindings(
        "microcosm.fit.qrf",
        "_gate_draw_with_rng",
        ("numpy.random.Generator.random",),
        *_QRF_ALL_SITE_IDS,
    ),
    # Hash-derived draws.
    _bind(
        "microcosm.build.us_runtime.acs_transfer",
        "_family_seed",
        "hashlib.sha256",
        "acs_transfer_family_seed",
    ),
    _bind(
        "microcosm.build.us_runtime.acs_transfer",
        "_pattern_seed",
        "hashlib.sha256",
        "acs_transfer_pattern_seed",
    ),
    _bind(
        "microcosm.build.us_runtime.org_wages",
        "_assign_union",
        "pandas.util.hash_pandas_object",
        "org_union_hash_lottery",
    ),
    _bind(
        "microcosm.build.us_runtime.immigration",
        "_stable_person_draws",
        "hashlib.blake2b",
        "immigration_ead_students_assignment",
        "immigration_ead_workers_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.pregnancy",
        "_stable_person_draws",
        "hashlib.blake2b",
        "pregnancy_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.snap_discretionary_exemption",
        "_stable_person_draws",
        "hashlib.blake2b",
        "snap_discretionary_exemption_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.snap_take_up",
        "_stable_unit_draws",
        "hashlib.blake2b",
        "snap_take_up_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.source_runtime",
        "_stable_draws",
        "hashlib.blake2b",
        "source_aca_assignment",
        "source_count_calibration",
        "source_joint_count_calibration",
    ),
    _bind(
        "microcosm.build.us_runtime.ssi_take_up",
        "_stable_source_draw",
        "hashlib.blake2b",
        "ssi_take_up_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.take_up",
        "_stable_unit_draws",
        "hashlib.blake2b",
        "eitc_take_up_assignment",
        "medicaid_take_up_assignment",
        "snap_state_take_up_assignment",
        "tanf_take_up_assignment",
    ),
    _bind(
        "microcosm.build.us_runtime.wic_claim",
        "_stable_person_draws",
        "hashlib.blake2b",
        "wic_claim_assignment",
    ),
)


def _operational_uuid_exemption(module: str, qualname: str) -> CallsiteExemption:
    return CallsiteExemption(
        _key(module, qualname, "uuid.uuid4"),
        "operational_nonce",
        "UUID4 supplies a collision-resistant temporary or run identifier; it does "
        "not select a model outcome and belongs to the operational effects broker",
    )


_NON_HASH_EXEMPTIONS = (
    CallsiteExemption(
        _key(
            "microcosm.build.holdout",
            "rotated_folds",
            "numpy.random.Generator.permutation",
        ),
        "offline_model_evaluation",
        "rotated holdout diagnostics are not invoked by the US pool producer graph",
    ),
    CallsiteExemption(
        _key(
            "microcosm.build.holdout",
            "rotated_folds",
            "numpy.random.default_rng",
        ),
        "offline_model_evaluation",
        "rotated holdout diagnostics are not invoked by the US pool producer graph",
    ),
    CallsiteExemption(
        _key("microcosm.fit", "fit", "microcosm.fit.QRF"),
        "uninvoked_public_convenience",
        "the pool calls the QRF class directly; this generic convenience front door "
        "is attested but not invoked by the producer graph",
    ),
    _operational_uuid_exemption("microcosm.build.logbook", "_atomic_write_row"),
    _operational_uuid_exemption("microcosm.build.logbook", "_atomic_write_bytes"),
    _operational_uuid_exemption(
        "microcosm.build.logbook_adoption", "atomic_write_json"
    ),
    _operational_uuid_exemption(
        "microcosm.build.us_runtime.h5_io", "AuthenticatedPoolH5.copy_verified_to"
    ),
    _operational_uuid_exemption(
        "microcosm.build.us_runtime.h5_io", "write_nullable_us_h5"
    ),
    _operational_uuid_exemption(
        "tools.build_us_multispine_pool", "_new_stacked_release_id"
    ),
    _operational_uuid_exemption(
        "tools.build_us_multispine_pool", "_new_publication_run_id"
    ),
    _operational_uuid_exemption("tools.build_us_multispine_pool", "_atomic_write_json"),
    _operational_uuid_exemption(
        "tools.build_us_multispine_pool", "_new_stacked_attempt_id"
    ),
)


def _hash_classification(
    kind: str,
    module: str,
    qualname: str,
    api: str = "hashlib.sha256",
    occurrence: int = 0,
) -> HashCallsiteClassification:
    return HashCallsiteClassification(_key(module, qualname, api, occurrence), kind)


# Filled below with all non-draw hashes.  Draw hashes are listed explicitly too
# so the taxonomy itself is a closed independent manifest.
_NON_DRAW_HASH_CLASSIFICATIONS = (
    _hash_classification(
        "content_identity", "microcosm.build.code_identity", "builder_code_identity"
    ),
    _hash_classification(
        "content_identity", "microcosm.build.frame_sampling", "ids_sha256"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.gate_battery",
        "GateBatteryRun.report_payload",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.gate_battery",
        "GateBatteryRun.report_payload",
        api="hashlib.sha256[callable_reference]",
    ),
    _hash_classification(
        "content_identity", "microcosm.build.gate_battery", "_canonical_sha256"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.ledger_artifact", "_sha256_file"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.ledger_artifact",
        "load_ledger_consumer_artifact",
    ),
    _hash_classification(
        "content_identity", "microcosm.build.ledger_targets", "_registry_digest"
    ),
    _hash_classification(
        "content_identity", "microcosm.build.logbook", "compute_row_digest"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.logbook_adoption", "_sha256"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.outer_stage_runtime", "_file_sha256"
    ),
    _hash_classification(
        "content_identity", "microcosm.build.outer_stage_runtime", "_mapping_sha256"
    ),
    _hash_classification(
        "content_identity", "microcosm.build.outer_stage_runtime", "frame_identity"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.outer_stage_runtime",
        "frame_identity",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification("content_identity", "microcosm.build.trace", "_payload_spec"),
    _hash_classification(
        "content_identity", "microcosm.build.trace", "compute_composition_fingerprint"
    ),
    _hash_classification("source_integrity", "microcosm.build.trace", "sha256_file"),
    _hash_classification(
        "content_identity", "microcosm.build.spec_engine.canonical", "sha256_json"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.artifact_collection",
        "_DigestSink.__init__",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.spec_engine.artifact_collection",
        "_regular_file_identity",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.artifact_comparison",
        "ArtifactDigest.from_bytes",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.artifact_comparison",
        "_receipt_rule_row",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.artifact_comparison",
        "_receipt_rule_row",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.compiler_ir",
        "_compiler_ir_abi",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.executor",
        "_pickle_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.engine_abi",
        "_fresh_remaining_stage_input_manifest",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.engine_abi",
        "_fresh_remaining_stage_input_manifest",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.imputation_semantics",
        "_canonical_sha256",
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.spec_engine.loader", "_sha256_bytes"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.reemission",
        "_spec_with_emitted_receipts",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.resolver",
        "KernelRegistry.from_ids",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.seeds",
        "SeedProtocol.implementation_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.seeds",
        "source_inventory_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.seeds",
        "source_inventory_sha256",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.seeds",
        "validate_seed_protocol_wire",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.stacked_authority_semantics",
        "_generation_zero_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.spec_engine.vintage_authorities",
        "resolve_vintage_authorities",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.checkpoint_authority",
        "_legacy_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_income_universe",
        "_mapping_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_income_universe",
        "_source_cells_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_income_universe",
        "_source_cells_sha256",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_income_universe",
        "_values_sha256",
    ),
    _hash_classification(
        "operational_subset",
        "microcosm.build.us_runtime.acs_pums",
        "_smoke_household_selection.rank",
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.acs_sources", "_file_identity"
    ),
    _hash_classification(
        "content_identity", "microcosm.build.us_runtime.acs_transfer", "_pattern_name"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_transfer",
        "acs_transfer_execution_contract_identity",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_transfer_bank",
        "AcsTransferTargetBankStore.load_target",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_transfer_bank",
        "AcsTransferTargetBankStore.write_target",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_transfer_bank",
        "_file_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.acs_transfer_bank",
        "_mapping_sha256",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.capital_gain_distributions",
        "capital_gain_distribution_shares_asset_identity",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.congressional_district_vintage",
        "_state_total_proxy_source_district_fact",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.congressional_district_vintage",
        "_translated_fact_shell",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.education_assistance_source",
        "_sha256_stream",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.education_assistance_source",
        "fetch_asec_education_assistance_source",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.education_assistance_source",
        "fetch_asec_education_assistance_source",
        occurrence=1,
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.h5_io", "_file_sha256_and_size"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.h5_io",
        "_read_json_object_with_identity",
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.housing_inputs", "_sha256"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.l0_refit_export", "_sha256"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.late_producer_dag",
        "derive_producer_schedule",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.multispine_pool",
        "_numeric_series_byte_receipt",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.multispine_pool",
        "_pool_engine_input_projection",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.multispine_pool",
        "pool_engine_input_projection_receipt",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.multispine_pool",
        "pool_remaining_stage_input_manifest_receipt",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.multispine_pool",
        "pool_remaining_stage_input_manifest_receipt",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity", "microcosm.build.us_runtime.org_wages", "_sha256"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.public_assistance_type_source",
        "_sha256_stream",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_capital_gains_tail",
        "_canonical_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_capital_gains_tail",
        "write_puf_capital_gains_tail_manifest",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_e01000_reconciliation",
        "_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_interest_components",
        "puf_e19200_agi_bands_runtime_identity",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.puf_interest_components",
        "puf_e19200_interest_components_asset_identity",
    ),
    _hash_classification(
        "content_identity", "microcosm.build.us_runtime.puf_qrf_chain", "_file_sha256"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_load_target_checkpoint",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_mapping_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_ordered_strings_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_recipient_identity_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_recipient_identity_sha256",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_qrf_chain",
        "_write_target_checkpoint",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_support",
        "_predictor_feature_values_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_support",
        "_predictor_feature_values_sha256",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification(
        "content_identity", "microcosm.build.us_runtime.puf_support", "_receipt_sha256"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.puf_support",
        "_source_ids_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.qbi_inputs",
        "_qbi_person_values_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.qbi_inputs",
        "_qbi_person_values_sha256",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.qbi_inputs",
        "_qbi_receipt_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.qbi_inputs",
        "_qbi_table_values_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.qbi_inputs",
        "_qbi_table_values_sha256",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.scf_auto_loans",
        "_sha256_hexdigest",
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.scf_wealth", "_sha256_hexdigest"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.sipp_financial_assets",
        "_sha256_stream",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.sipp_head_start",
        "_sha256_stream",
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.sipp_tips", "_sha256_file"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.sipp_tips", "_sha256_hexdigest"
    ),
    _hash_classification(
        "source_integrity", "microcosm.build.us_runtime.sipp_vehicles", "_sha256_stream"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.sipp_vehicles",
        "fetch_sipp_2023_vehicle_donor",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.ssi_disability_criteria",
        "_sha256_file",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.stacked_spine",
        "_canonical_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.stacked_spine",
        "_integer_rows_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.stacked_spine",
        "_late_frame_content_sha256",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.stacked_spine",
        "_late_source_stage_spec_binding",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.stacked_spine",
        "_late_table_values_sha256",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.take_up_contract",
        "_canonical_resource_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.us_late_overlap_ownership",
        "_canonical_sha256",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.us_late_producer_registry",
        "us_late_producer_schedule_receipt",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.census_country_bridge",
        "load_census_country_bridge",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.census_country_bridge",
        "load_census_country_bridge",
        occurrence=1,
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.census_imports",
        "_fetch_prefix_with_cache",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.census_imports",
        "_fetch_prefix_with_cache",
        occurrence=1,
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.imdb_bulk",
        "ensure_imdb_archive",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.import_entry_facts",
        "_fact_row",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.us_trade.import_entry_facts",
        "_key",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.import_entry_facts",
        "_month_file_identities",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.us_trade.import_entry_facts",
        "_retrieval_set_digest",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.us_trade.import_entry_facts",
        "write_consumer_artifact",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.voluntary_filing",
        "_sha256_stream",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.voluntary_filing",
        "fetch_sipp_2023_voluntary_filing_donor",
    ),
    _hash_classification(
        "content_identity",
        "microcosm.build.us_runtime.warm_start_selection",
        "_identities_digest",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.warm_start_selection",
        "_sha256_file",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.weeks_unemployed",
        "_sha256_stream",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.weeks_unemployed",
        "fetch_asec_2023_weeks_unemployed_source",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.build.us_runtime.weeks_unemployed",
        "fetch_asec_2023_weeks_unemployed_source",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "microcosm.calibrate._target_loss_attribution",
        "target_loss_basis_hash",
    ),
    _hash_classification(
        "content_identity", "microcosm.calibrate.diagnostics", "_target_surface_payload"
    ),
    _hash_classification(
        "content_identity",
        "microcosm.calibrate.diagnostics",
        "_target_surface_payload",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "microcosm.calibrate.diagnostics",
        "_target_surface_payload",
        occurrence=2,
    ),
    _hash_classification(
        "content_identity", "microcosm.calibrate.registry", "TargetRegistry.version"
    ),
    _hash_classification(
        "content_identity", "microcosm.data.contract", "_canonical_sha256"
    ),
    _hash_classification(
        "source_integrity", "microcosm.data.contract", "_check_uk_gate_battery_report"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.data.contract",
        "_check_uk_gate_battery_report",
        api="hashlib.sha256[callable_reference]",
    ),
    _hash_classification(
        "source_integrity", "microcosm.data.contract", "_check_uk_terminal_gate_report"
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.data.contract",
        "_check_uk_terminal_gate_report",
        api="hashlib.sha256[callable_reference]",
    ),
    _hash_classification("content_identity", "microcosm.data.contract", "_sha256"),
    _hash_classification("source_integrity", "microcosm.data.loader", "_sha256"),
    _hash_classification("source_integrity", "microcosm.data.release", "_sha256"),
    _hash_classification("content_identity", "microcosm.fit.cache", "_digest_payload"),
    _hash_classification("content_identity", "microcosm.fit.qrf", "_index_identity"),
    _hash_classification(
        "content_identity",
        "microcosm.fit.qrf",
        "_index_identity",
        api="pandas.util.hash_pandas_object",
    ),
    _hash_classification("content_identity", "microcosm.fit.qrf", "_weight_identity"),
    _hash_classification(
        "content_identity",
        "microcosm.frame.adapters.policyengine_us",
        "PolicyEngineUSVariableMetadataIndex.variable_dependency_closure",
    ),
    _hash_classification(
        "source_integrity",
        "microcosm.frame.adapters.policyengine_us",
        "_index_policyengine_us_generated_variable_sources",
    ),
    _hash_classification(
        "content_identity",
        "tools.build_us_multispine_pool",
        "_configured_input_pins_digest",
    ),
    _hash_classification(
        "source_integrity", "tools.build_us_multispine_pool", "_file_sha256"
    ),
    _hash_classification(
        "content_identity", "tools.build_us_multispine_pool", "_input_pins_digest"
    ),
    _hash_classification(
        "content_identity", "tools.build_us_multispine_pool", "_main_stacked"
    ),
    _hash_classification(
        "content_identity",
        "tools.build_us_multispine_pool",
        "_main_stacked",
        occurrence=1,
    ),
    _hash_classification(
        "content_identity",
        "tools.build_us_multispine_pool",
        "_pool_checkpoint_identity_sha256",
    ),
    _hash_classification(
        "content_identity",
        "tools.build_us_multispine_pool",
        "_stacked_checkpoint_base_identity",
    ),
    _hash_classification(
        "content_identity", "tools.build_us_multispine_pool", "_stacked_checkpoint_root"
    ),
    _hash_classification(
        "content_identity",
        "tools.build_us_multispine_pool",
        "_stacked_manifest_payload",
    ),
)
_STOCHASTIC_HASH_CLASSIFICATIONS = tuple(
    HashCallsiteClassification(binding.callsite, "stochastic_draw")
    for binding in LEGACY_V1_PRODUCTION_BINDINGS
    if binding.callsite.api.startswith("hashlib.")
    or binding.callsite.api == "pandas.util.hash_pandas_object"
)
LEGACY_V1_HASH_CLASSIFICATIONS = (
    *_STOCHASTIC_HASH_CLASSIFICATIONS,
    *_NON_DRAW_HASH_CLASSIFICATIONS,
)
_HASH_EXEMPTION_REASONS = {
    "content_identity": (
        "deterministic content/provenance identity; it selects no stochastic outcome"
    ),
    "source_integrity": (
        "source or artifact integrity verification; it selects no stochastic outcome"
    ),
    "operational_subset": (
        "developer-only deterministic smoke subset outside the published producer graph"
    ),
}
LEGACY_V1_PRODUCTION_EXEMPTIONS = (
    *_NON_HASH_EXEMPTIONS,
    *(
        CallsiteExemption(
            row.callsite,
            row.kind,
            _HASH_EXEMPTION_REASONS[row.kind],
        )
        for row in _NON_DRAW_HASH_CLASSIFICATIONS
    ),
)


def assert_exact_production_callsite_coverage(
    repository_root: Path,
    *,
    protocol_site_ids: frozenset[str],
    kernel_source_modules_by_site: dict[str, frozenset[str]],
) -> None:
    """Reject new, stale, unbound, unclassified, or unattested physical sites."""

    discovered_rows = discover_production_callsites(repository_root)
    discovered = {row.callsite for row in discovered_rows}
    if len(discovered) != len(discovered_rows):
        raise AssertionError("discovery emitted duplicate physical callsite keys")
    included_modules = discover_production_source_modules(repository_root)
    excluded_modules = discover_exempted_source_modules(repository_root)
    for exemption, modules in excluded_modules.items():
        if not modules:
            raise AssertionError(
                f"stale source namespace exemption {exemption.prefix!r}"
            )
        overlap = set(modules) & included_modules.keys()
        if overlap:
            raise AssertionError(
                f"source namespace exemption leaked into scan: {sorted(overlap)!r}"
            )

    binding_by_callsite = {
        binding.callsite: binding for binding in LEGACY_V1_PRODUCTION_BINDINGS
    }
    exemption_by_callsite = {
        exemption.callsite: exemption for exemption in LEGACY_V1_PRODUCTION_EXEMPTIONS
    }
    if len(binding_by_callsite) != len(LEGACY_V1_PRODUCTION_BINDINGS):
        raise AssertionError("duplicate physical callsite binding")
    if len(exemption_by_callsite) != len(LEGACY_V1_PRODUCTION_EXEMPTIONS):
        raise AssertionError("duplicate physical callsite exemption")
    overlap = binding_by_callsite.keys() & exemption_by_callsite.keys()
    if overlap:
        raise AssertionError(
            f"callsites are both bound and exempt: {sorted(overlap)!r}"
        )
    declared = binding_by_callsite.keys() | exemption_by_callsite.keys()
    if discovered != declared:
        lines = {row.callsite: row.line for row in discovered_rows}
        unbound = [(site, lines[site]) for site in sorted(discovered - declared)]
        raise AssertionError(
            "production stochastic-callsite manifest differs: "
            f"unbound={unbound!r}, stale={sorted(declared - discovered)!r}"
        )

    bound_site_ids = {
        site_id
        for binding in LEGACY_V1_PRODUCTION_BINDINGS
        for site_id in binding.site_ids
    }
    if bound_site_ids != protocol_site_ids:
        raise AssertionError(
            "logical ledger/physical bindings differ: "
            f"unbound_sites={sorted(protocol_site_ids - bound_site_ids)!r}, "
            f"unknown_sites={sorted(bound_site_ids - protocol_site_ids)!r}"
        )

    hash_calls = {
        callsite
        for callsite in discovered
        if callsite.api.startswith("hashlib.")
        or callsite.api == "pandas.util.hash_pandas_object"
    }
    classification_by_callsite = {
        row.callsite: row for row in LEGACY_V1_HASH_CLASSIFICATIONS
    }
    if len(classification_by_callsite) != len(LEGACY_V1_HASH_CLASSIFICATIONS):
        raise AssertionError("duplicate hash callsite classification")
    if hash_calls != classification_by_callsite.keys():
        raise AssertionError(
            "hash classifications differ from discovered hash surface: "
            f"unclassified={sorted(hash_calls - classification_by_callsite.keys())!r}, "
            f"stale={sorted(classification_by_callsite.keys() - hash_calls)!r}"
        )
    for callsite, classification in classification_by_callsite.items():
        if classification.kind == "stochastic_draw":
            if callsite not in binding_by_callsite:
                raise AssertionError(
                    f"stochastic hash is not ledger-bound: {callsite!r}"
                )
        elif callsite not in exemption_by_callsite:
            raise AssertionError(f"non-draw hash lacks typed exemption: {callsite!r}")
        elif exemption_by_callsite[callsite].kind != classification.kind:
            raise AssertionError(
                f"hash classification/exemption disagree for {callsite!r}"
            )

    for binding in LEGACY_V1_PRODUCTION_BINDINGS:
        for site_id in binding.site_ids:
            if binding.callsite.module not in kernel_source_modules_by_site[site_id]:
                raise AssertionError(
                    f"{site_id!r} callsite module {binding.callsite.module!r} is not "
                    "covered by its kernel attestation"
                )


__all__ = [
    "CallsiteBinding",
    "CallsiteExemption",
    "DiscoveredCallsite",
    "HASH_CLASSIFICATION_KINDS",
    "HashCallsiteClassification",
    "LEGACY_V1_HASH_CLASSIFICATIONS",
    "LEGACY_V1_PRODUCTION_BINDINGS",
    "LEGACY_V1_PRODUCTION_EXEMPTIONS",
    "PhysicalCallsite",
    "SOURCE_NAMESPACE_EXEMPTIONS",
    "SourceNamespaceExemption",
    "UK_ONLY_SOURCE_PREFIXES",
    "assert_exact_production_callsite_coverage",
    "discover_production_callsites",
    "discover_production_source_modules",
    "discover_exempted_source_modules",
]
