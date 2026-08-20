"""Node-scoped ambient-access brokers for the spec-engine executor.

The compiler owns grants.  A :class:`BrokerSession` narrows those grants to one
compiled node (or one typed outer-stage owner), mints opaque RNG site tokens,
and records operational access events.  The event receipt is deliberately not
part of the bundle identity, node reuse identity, patch hash, or artifact
bytes.

The ambient guard is a defence-in-depth boundary for Python kernels.  It
refuses direct file, environment, clock, and common RNG entry points while a
kernel or trusted row classifier is running.  Broker implementations use
captured primitives and never relax that guard for kernel code.
"""

from __future__ import annotations

import builtins
import codecs
import copy
import dataclasses
import datetime as datetime_module
import functools
import hashlib
import inspect
import io
import os
import pickle
import random as python_random
import secrets
import socket
import stat
import subprocess
import sys
import threading
import time as time_module
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from string import Formatter
from types import FunctionType, MappingProxyType, MethodType, ModuleType
from typing import IO, Any, Literal
from unittest.mock import patch

import numpy as np
import pandas as pd

from .canonical import sha256_json
from .compiler_ir import CompiledNode, SeedOwnerIR, SeedSiteIR, SeedStreamMap
from .model import FrozenMap, freeze_json, thaw_json


class BrokerError(ValueError):
    """A broker contract, token, or access is invalid."""


class BrokerContractError(BrokerError):
    """A session does not match its compiler-owned authority."""


class BrokerAccessError(BrokerError):
    """An explicit broker access was refused."""


class AmbientAccessError(BrokerAccessError):
    """A kernel attempted ambient access instead of using its broker."""


_SEED_SITE_CONTRACT_FIELDS = frozenset(
    {
        "value_source",
        "default",
        "rng_family",
        "rng_version",
        "kernel",
        "seed_material",
        "consumption_order",
        "reset_boundary",
        "draw_condition",
        "derivation",
    }
)
_OWNER_KINDS = frozenset({"producer_node", "source_stage", "pipeline_operation"})
_EFFECTS = frozenset({"none", "declared_source_read", "declared_sink_write"})
_DETERMINISM = frozenset({"deterministic", "seeded", "nondeterministic"})
_RECEIPT_DOMAIN = "microcosm.spec-engine.broker-access-receipt.v1"
_RNG_BEHAVIOR_DOMAIN = "microcosm.spec-engine.rng-behavior-inputs.v1"
_SOURCE_BEHAVIOR_DOMAIN = "microcosm.spec-engine.source-behavior-inputs.v1"
_RNG_BEHAVIOR_ISSUER = object()
_SOURCE_BEHAVIOR_ISSUER = object()
_RUN_PROVENANCE_FIELDS = frozenset(
    {
        "identity_generation",
        "source_grammar_receipt",
        "spec_binding",
        "authority_versions",
        "code_inventory_digest",
        "artifact_protocol_inventory",
        "run_request",
        "execution_receipt",
    }
)
_GRAMMAR_RECEIPT_FIELDS = frozenset(
    {"schema_version", "canonicalizer_version", "migration_chain"}
)
_SPEC_BINDING_FIELDS = frozenset(
    {
        "country",
        "schema_id",
        "schema_version",
        "canonicalizer_version",
        "spec_sha256",
        "attestation",
    }
)
_DEFAULT_RNG_BOUNDARY_KEY = "default"
_PHYSICAL_OPERATION_POLICIES = frozenset({"broker-only", "legacy-v1"})
_PINNED_DEPENDENCY_ENVIRONMENT_DEFAULTS = MappingProxyType(
    {
        "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "PYARROW_IGNORE_TIMEZONE": None,
    }
)
_SAFE_GENERATOR_DRAW_METHODS = frozenset(
    {"choice", "integers", "permutation", "random"}
)
_BROKER_KINDS = frozenset({"rng", "file", "environment", "clock", "ambient"})
_DISPOSITIONS = frozenset({"allowed", "refused"})
_NUMPY_RANDOM_AMBIENT_NAMES = (
    "BitGenerator",
    "Generator",
    "MT19937",
    "PCG64",
    "PCG64DXSM",
    "Philox",
    "RandomState",
    "SFC64",
    "SeedSequence",
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "default_rng",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "permutation",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random",
    "random_sample",
    "rayleigh",
    "seed",
    "shuffle",
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
)
_OS_AMBIENT_NAMES = (
    "access",
    "chdir",
    "fstat",
    "getcwd",
    "getcwdb",
    "listdir",
    "lstat",
    "readlink",
    "scandir",
    "stat",
    "statvfs",
    "walk",
)

_ORIGINAL_BUILTINS_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_FSTAT = os.fstat
_ORIGINAL_OS_URANDOM = os.urandom
_ORIGINAL_ENVIRON = os.environ
_ORIGINAL_ENVIRONB = getattr(os, "environb", None)
_ORIGINAL_DEFAULT_RNG = np.random.default_rng
_ORIGINAL_GENERATOR = np.random.Generator
_ORIGINAL_PCG64 = np.random.PCG64
_ORIGINAL_SEED_SEQUENCE = np.random.SeedSequence
_ORIGINAL_RANDOM_STATE = np.random.RandomState
_ORIGINAL_PYTHON_RANDOM_CLASS = python_random.Random
_ORIGINAL_HASH_PANDAS_OBJECT = pd.util.hash_pandas_object
_ORIGINAL_DATAFRAME_SAMPLE = pd.DataFrame.sample
_ORIGINAL_TIME = time_module.time
_ORIGINAL_OS_CPU_COUNT = os.cpu_count
_ORIGINAL_DATETIME = datetime_module.datetime
_ORIGINAL_DATE = datetime_module.date
_ORIGINAL_PATH_OPEN = Path.open
_ORIGINAL_OS_AMBIENT = MappingProxyType(
    {name: getattr(os, name) for name in _OS_AMBIENT_NAMES if hasattr(os, name)}
)
_ORIGINAL_ENVIRONMENT_CALLS = MappingProxyType(
    {
        name: getattr(os, name)
        for name in ("getenv", "getenvb", "putenv", "unsetenv")
        if hasattr(os, name)
    }
)
_ORIGINAL_NUMPY_RANDOM = MappingProxyType(
    {
        name: getattr(np.random, name)
        for name in _NUMPY_RANDOM_AMBIENT_NAMES
        if hasattr(np.random, name)
    }
)
_ORIGINAL_SUBPROCESS = MappingProxyType(
    {
        name: getattr(subprocess, name)
        for name in ("Popen", "call", "check_call", "check_output", "run")
        if hasattr(subprocess, name)
    }
)
_ORIGINAL_OPERATIONAL_CLOCKS = MappingProxyType(
    {
        name: getattr(time_module, name)
        for name in (
            "time",
            "time_ns",
            "monotonic",
            "monotonic_ns",
            "perf_counter",
            "perf_counter_ns",
            "process_time",
            "process_time_ns",
            "thread_time",
            "thread_time_ns",
            "clock_gettime",
            "clock_gettime_ns",
            "sleep",
        )
        if hasattr(time_module, name)
    }
)
_ORIGINAL_UUID4 = uuid.uuid4


def _captured_forbidden_callables() -> Mapping[int, str]:
    values: dict[int, str] = {}

    def add(value: object, label: str) -> None:
        if callable(value):
            values[id(value)] = label

    add(_ORIGINAL_BUILTINS_OPEN, "builtins.open")
    add(_ORIGINAL_IO_OPEN, "io.open")
    add(_ORIGINAL_OS_OPEN, "os.open")
    add(Path.open, "pathlib.Path.open")
    add(_ORIGINAL_DATAFRAME_SAMPLE, "pandas.DataFrame.sample")
    for module, prefix, names in (
        (
            time_module,
            "time",
            (
                "time",
                "time_ns",
                "monotonic",
                "monotonic_ns",
                "perf_counter",
                "perf_counter_ns",
                "process_time",
                "process_time_ns",
                "thread_time",
                "thread_time_ns",
                "clock_gettime",
                "clock_gettime_ns",
                "sleep",
            ),
        ),
        (
            os,
            "os",
            (
                "getenv",
                "getenvb",
                "putenv",
                "unsetenv",
                "urandom",
                *_OS_AMBIENT_NAMES,
            ),
        ),
        (
            np.random,
            "numpy.random",
            _NUMPY_RANDOM_AMBIENT_NAMES,
        ),
        (
            python_random,
            "random",
            (
                "Random",
                "SystemRandom",
                "seed",
                "random",
                "randint",
                "randrange",
                "choice",
                "choices",
                "uniform",
                "shuffle",
                "sample",
                "getrandbits",
            ),
        ),
        (
            secrets,
            "secrets",
            (
                "token_bytes",
                "token_hex",
                "token_urlsafe",
                "choice",
                "randbelow",
                "randbits",
            ),
        ),
        (uuid, "uuid", ("uuid4",)),
        (socket, "socket", ("create_connection", "socket", "socketpair")),
        (
            subprocess,
            "subprocess",
            ("Popen", "call", "check_call", "check_output", "run"),
        ),
    ):
        for name in names:
            if hasattr(module, name):
                add(getattr(module, name), f"{prefix}.{name}")
    for cls, prefix, names in (
        (_ORIGINAL_DATETIME, "datetime.datetime", ("now", "utcnow", "today")),
        (_ORIGINAL_DATE, "datetime.date", ("today",)),
    ):
        add(cls, prefix)
        for name in names:
            add(getattr(cls, name), f"{prefix}.{name}")
    return MappingProxyType(values)


_FORBIDDEN_CALLABLES = _captured_forbidden_callables()
_DYNAMICALLY_GUARDED_MODULE_NAMES = frozenset(
    {
        "builtins",
        "datetime",
        "io",
        "numpy",
        "numpy.random",
        "os",
        "pandas",
        "pathlib",
        "random",
        "secrets",
        "socket",
        "subprocess",
        "time",
        "uuid",
    }
)


def _prebound_ambient_hits(function: object) -> tuple[str, ...]:
    """Find ambient primitives captured before runtime monkeypatching.

    This complements the dynamic guard.  It intentionally inspects only the
    submitted callable and closure-local helper functions; repository-wide
    transitive source enforcement remains the static inventory gate.
    """

    hits: set[str] = set()
    visited: set[int] = set()
    root_module = getattr(function, "__module__", None)

    def inspect_value(value: object, *, location: str, strict: bool) -> None:
        if id(value) in visited:
            return
        label = _FORBIDDEN_CALLABLES.get(id(value))
        if label is not None:
            hits.add(f"{location}={label}")
            return
        if value is _ORIGINAL_ENVIRON or value is _ORIGINAL_ENVIRONB:
            hits.add(f"{location}=os.environ")
            return
        if isinstance(value, _ORIGINAL_GENERATOR | _ORIGINAL_RANDOM_STATE):
            hits.add(f"{location}=private_numpy_rng")
            return
        if isinstance(value, python_random.Random):
            hits.add(f"{location}=private_python_rng")
            return
        if isinstance(value, FunctionType):
            # Direct forbidden primitives were matched above.  Recurse through
            # helpers owned by the submitted kernel's module, where aliases can
            # evade the dynamic monkeypatch.  Imported library implementations
            # are covered by the runtime guard and the static callsite gate;
            # walking their private globals would reject harmless sentinels and
            # implementation caches unrelated to the submitted kernel.
            inspect_function(
                value,
                prefix=location,
                strict=strict and value.__module__ == root_module,
            )
            return
        visited.add(id(value))
        if isinstance(value, functools.partial):
            inspect_value(
                value.func, location=f"{location}/partial:func", strict=strict
            )
            for index, child in enumerate(value.args):
                inspect_value(
                    child,
                    location=f"{location}/partial:arg:{index}",
                    strict=strict,
                )
            for name, child in (value.keywords or {}).items():
                inspect_value(
                    child,
                    location=f"{location}/partial:kw:{name}",
                    strict=strict,
                )
            return
        if isinstance(value, MethodType):
            inspect_value(
                value.__func__,
                location=f"{location}/method:function",
                strict=strict,
            )
            inspect_value(
                value.__self__, location=f"{location}/method:self", strict=strict
            )
            return
        if isinstance(value, Mapping):
            if not isinstance(value, (dict, FrozenMap, MappingProxyType)):
                if strict:
                    hits.add(f"{location}=uninspectable_captured_mapping")
                return
            if strict and isinstance(value, dict):
                hits.add(f"{location}=mutable_captured_mapping")
            for index, (key, child) in enumerate(value.items()):
                inspect_value(
                    key,
                    location=f"{location}/mapping:key:{index}",
                    strict=strict,
                )
                inspect_value(
                    child,
                    location=f"{location}/mapping:value:{index}",
                    strict=strict,
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            if not isinstance(value, (list, tuple)):
                if strict:
                    hits.add(f"{location}=uninspectable_captured_sequence")
                return
            if strict and isinstance(value, list):
                hits.add(f"{location}=mutable_captured_sequence")
            for index, child in enumerate(value):
                inspect_value(
                    child,
                    location=f"{location}/sequence:{index}",
                    strict=strict,
                )
            return
        if isinstance(value, set | frozenset):
            if strict and isinstance(value, set):
                hits.add(f"{location}=mutable_captured_set")
            for index, child in enumerate(value):
                inspect_value(child, location=f"{location}/set:{index}", strict=strict)
            return
        if isinstance(value, np.ndarray | pd.DataFrame | pd.Series):
            if strict:
                hits.add(f"{location}=mutable_captured_data")
            return
        if isinstance(value, Enum):
            return
        if isinstance(value, ModuleType):
            if value.__name__ in _DYNAMICALLY_GUARDED_MODULE_NAMES:
                return
            for name, child in vars(value).items():
                label = _FORBIDDEN_CALLABLES.get(id(child))
                if (
                    label is not None
                    or child is _ORIGINAL_ENVIRON
                    or (_ORIGINAL_ENVIRONB is not None and child is _ORIGINAL_ENVIRONB)
                ):
                    hits.add(f"{location}/module:{name}={label or 'os.environ'}")
            return
        if isinstance(value, type):
            for name, child in vars(value).items():
                if isinstance(child, staticmethod | classmethod):
                    child = child.__func__
                inspect_value(
                    child,
                    location=f"{location}/class:{name}",
                    strict=False,
                )
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            params = getattr(value, "__dataclass_params__", None)
            if strict and (params is None or not params.frozen):
                hits.add(f"{location}=mutable_captured_object")
            for item in dataclasses.fields(value):
                inspect_value(
                    getattr(value, item.name),
                    location=f"{location}/field:{item.name}",
                    strict=strict,
                )
            return
        if strict and (hasattr(value, "__dict__") or hasattr(type(value), "__slots__")):
            hits.add(f"{location}=unsealed_captured_object:{type(value).__qualname__}")

    def inspect_function(value: object, *, prefix: str, strict: bool) -> None:
        if not isinstance(value, FunctionType) or id(value) in visited:
            return
        visited.add(id(value))
        closure = inspect.getclosurevars(value)
        for namespace, mapping in (
            ("nonlocal", closure.nonlocals),
            ("global", closure.globals),
            ("builtin", closure.builtins),
        ):
            for name, child in mapping.items():
                inspect_value(
                    child,
                    location=f"{prefix}/{namespace}:{name}",
                    strict=strict,
                )
        for index, child in enumerate(value.__defaults__ or ()):
            inspect_value(child, location=f"{prefix}/default:{index}", strict=strict)
        for name, child in (value.__kwdefaults__ or {}).items():
            inspect_value(child, location=f"{prefix}/kwdefault:{name}", strict=strict)

    inspect_function(
        function,
        prefix=getattr(function, "__qualname__", "callable"),
        strict=True,
    )
    return tuple(sorted(hits))


def _wire(value: object) -> object:
    try:
        return thaw_json(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return copy.deepcopy(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_detail(value: object) -> object:
    """Return a receipt-safe value without exposing arbitrary Python objects."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise BrokerContractError("broker receipt values must be finite")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not np.isfinite(result):
            raise BrokerContractError("broker receipt values must be finite")
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_detail(child)
            for key, child in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonical_detail(child) for child in value]
    raise BrokerContractError(
        f"broker receipt value has unsupported type {type(value).__name__}"
    )


def _run_provenance_identity(
    value: Mapping[str, object],
) -> FrozenMap:
    """Validate and freeze the closed run identity carried by every receipt."""

    if not isinstance(value, Mapping) or set(value) != _RUN_PROVENANCE_FIELDS:
        raise BrokerContractError(
            "run provenance identity does not have the closed RFC field set"
        )
    row = _canonical_detail(value)
    assert isinstance(row, dict)
    generation = row["identity_generation"]
    if isinstance(generation, bool) or generation not in {0, 1}:
        raise BrokerContractError("run provenance identity_generation must be 0 or 1")
    digest = row["code_inventory_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise BrokerContractError("run provenance code inventory must be sha256")
    for name in (
        "authority_versions",
        "artifact_protocol_inventory",
        "run_request",
        "execution_receipt",
    ):
        if not isinstance(row[name], dict):
            raise BrokerContractError(f"run provenance {name} must be an object")
    grammar = row["source_grammar_receipt"]
    binding = row["spec_binding"]
    if generation == 0:
        if grammar is not None or binding is not None:
            raise BrokerContractError(
                "generation 0 provenance cannot carry the binding triad"
            )
    else:
        if not isinstance(grammar, dict) or set(grammar) != _GRAMMAR_RECEIPT_FIELDS:
            raise BrokerContractError("generation 1 grammar receipt is invalid")
        if not isinstance(binding, dict) or set(binding) != _SPEC_BINDING_FIELDS:
            raise BrokerContractError("generation 1 spec binding is invalid")
        for name in ("schema_version", "canonicalizer_version"):
            if (
                isinstance(grammar[name], bool)
                or not isinstance(grammar[name], int)
                or grammar[name] < 1
                or grammar[name] != binding[name]
            ):
                raise BrokerContractError(
                    f"run provenance {name} is invalid or inconsistent"
                )
        if binding["attestation"] not in {
            "mirror-attested",
            "bundle-authoritative",
        }:
            raise BrokerContractError("run provenance attestation is unknown")
        for name in ("country", "schema_id"):
            if not isinstance(binding[name], str) or not binding[name]:
                raise BrokerContractError(
                    f"run provenance spec binding {name} must be non-empty"
                )
        spec_sha256 = binding["spec_sha256"]
        if (
            not isinstance(spec_sha256, str)
            or len(spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec_sha256)
        ):
            raise BrokerContractError("run provenance spec digest must be sha256")
        migration_chain = grammar["migration_chain"]
        if not isinstance(migration_chain, list):
            raise BrokerContractError("run provenance migration chain must be an array")
        for index, migration in enumerate(migration_chain):
            if not isinstance(migration, dict) or set(migration) != {"id", "sha256"}:
                raise BrokerContractError(
                    "run provenance migration rows require exactly id and sha256"
                )
            if not isinstance(migration["id"], str) or not migration["id"]:
                raise BrokerContractError(
                    f"run provenance migration {index} id must be non-empty"
                )
            self_digest = migration["sha256"]
            if (
                not isinstance(self_digest, str)
                or len(self_digest) != 64
                or any(character not in "0123456789abcdef" for character in self_digest)
            ):
                raise BrokerContractError(
                    f"run provenance migration {index} digest must be sha256"
                )
    frozen = freeze_json(row)
    assert isinstance(frozen, FrozenMap)
    return frozen


@dataclass(frozen=True, slots=True)
class BrokerOwner:
    """The compiler-owned execution unit to which a session is bound."""

    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in _OWNER_KINDS:
            raise BrokerContractError(f"unknown broker owner kind {self.kind!r}")
        if not isinstance(self.id, str) or not self.id:
            raise BrokerContractError("broker owner id must be non-empty")

    def to_wire(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


@dataclass(frozen=True, slots=True)
class PhysicalOperation:
    """One implementation-pinned physical call owned by a node session.

    A ``broker-only`` callable receives only the kernel-visible broker context;
    its other runtime input is closed over before executor dispatch and attested
    by ``input_binding_sha256``.  An explicitly selected ``legacy-v1`` callable
    remains zero-argument for byte-compatibility with the transitional physical
    bridge.

    A kernel can request the call, but cannot replace the callable, pass it raw
    authority, or alter its sink roots.  Only the explicit legacy policy enters
    the narrowly restored legacy primitive scope; broker-only calls remain under
    the ambient guard and must use the supplied context.
    """

    function: Callable[..., object] = field(repr=False, compare=False)
    implementation_sha256: str
    input_binding_sha256: str
    policy: Literal["broker-only", "legacy-v1"]
    sink_roots: tuple[Path | str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.function, FunctionType):
            raise BrokerContractError(
                "physical operation must be a directly inspectable Python function"
            )
        if inspect.iscoroutinefunction(self.function) or inspect.isgeneratorfunction(
            self.function
        ):
            raise BrokerContractError(
                "physical operation must be a synchronous scalar call"
            )
        try:
            signature = inspect.signature(self.function)
        except (TypeError, ValueError) as error:
            raise BrokerContractError(
                "physical operation signature cannot be inspected"
            ) from error
        parameters = tuple(signature.parameters.values())
        if self.policy not in _PHYSICAL_OPERATION_POLICIES:
            raise BrokerContractError(
                f"unknown physical operation policy {self.policy!r}"
            )
        if self.policy == "legacy-v1":
            if parameters:
                raise BrokerContractError(
                    "legacy-v1 physical operation must accept no runtime arguments"
                )
        elif (
            len(parameters) != 1
            or parameters[0].kind
            not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            or parameters[0].default is not inspect.Parameter.empty
        ):
            raise BrokerContractError(
                "broker-only physical operation must accept exactly one required "
                "kernel context"
            )
        for name, digest in (
            ("implementation", self.implementation_sha256),
            ("input binding", self.input_binding_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise BrokerContractError(
                    f"physical operation {name} digest must be lowercase sha256"
                )
        captured: list[object] = list(self.function.__defaults__ or ())
        captured.extend((self.function.__kwdefaults__ or {}).values())
        for closure in self.function.__closure__ or ():
            try:
                captured.append(closure.cell_contents)
            except ValueError:  # pragma: no cover - empty cells are harmless
                continue
        if _contains_raw_generator(captured):
            raise BrokerContractError(
                "physical operation may not capture raw RNG authority"
            )
        roots: list[Path] = []
        for root in self.sink_roots:
            path = Path(root)
            if not path.is_absolute():
                raise BrokerContractError(
                    "physical operation sink roots must be absolute paths"
                )
            resolved = path.resolve(strict=False)
            if resolved not in roots:
                roots.append(resolved)
        object.__setattr__(self, "sink_roots", tuple(roots))


@dataclass(frozen=True, slots=True)
class DeclaredSource:
    """One preverified logical source binding.

    The path is operational.  The content digest and size are semantic input
    facts and are verified before bytes are returned to a kernel.
    """

    id: str
    path: Path | str
    sha256: str
    byte_size: int
    _resolved_path: Path = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise BrokerContractError("declared source id must be non-empty")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise BrokerContractError("declared source sha256 must be lowercase hex")
        if isinstance(self.byte_size, bool) or self.byte_size < 0:
            raise BrokerContractError("declared source byte_size must be non-negative")
        path = Path(self.path)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise BrokerContractError(
                f"declared source {self.id!r} cannot be resolved"
            ) from error
        if not resolved.is_file():
            raise BrokerContractError(f"declared source {self.id!r} is not a file")
        if path.is_symlink() or resolved != path.absolute():
            raise BrokerContractError(
                f"declared source {self.id!r} may not traverse a symlink"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "_resolved_path", resolved)

    @property
    def resolved_path(self) -> Path:
        return self._resolved_path


@dataclass(frozen=True, slots=True, init=False)
class SourceBehaviorIdentity:
    """Path-free content identities for all file grants in one session."""

    owner: BrokerOwner
    sources: tuple[FrozenMap, ...]
    _issuer: object = field(repr=False, compare=False)
    _session_issuer: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        owner: BrokerOwner,
        sources: tuple[FrozenMap, ...],
        _issuer: object,
        _session_issuer: object,
    ) -> None:
        if _issuer is not _SOURCE_BEHAVIOR_ISSUER:
            raise BrokerContractError(
                "source behavior identities may only be issued by a file broker"
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "_issuer", _issuer)
        object.__setattr__(self, "_session_issuer", _session_issuer)
        self.validate_issued()

    def validate_issued(self) -> None:
        if self._issuer is not _SOURCE_BEHAVIOR_ISSUER:
            raise BrokerContractError("source behavior identity issuer is invalid")
        if not isinstance(self.owner, BrokerOwner):
            raise BrokerContractError("source behavior owner must be a BrokerOwner")
        if not isinstance(self.sources, tuple) or any(
            not isinstance(source, FrozenMap) for source in self.sources
        ):
            raise BrokerContractError("source behavior rows must be frozen objects")
        rows = [thaw_json(source) for source in self.sources]
        ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "source_id",
                "content_sha256",
                "byte_size",
            }:
                raise BrokerContractError("source behavior row shape is not closed")
            source_id = row["source_id"]
            digest = row["content_sha256"]
            size = row["byte_size"]
            if not isinstance(source_id, str) or not source_id:
                raise BrokerContractError("source behavior id must be non-empty")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise BrokerContractError("source behavior digest must be sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise BrokerContractError(
                    "source behavior byte_size must be non-negative"
                )
            ids.append(source_id)
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise BrokerContractError(
                "source behavior ids must be unique and canonically ordered"
            )

    def same_session(self, rng_identity: RNGBehaviorIdentity) -> bool:
        return self._session_issuer is rng_identity._session_issuer

    def body_wire(self) -> dict[str, object]:
        return {
            "domain": _SOURCE_BEHAVIOR_DOMAIN,
            "owner": self.owner.to_wire(),
            "sources": [thaw_json(source) for source in self.sources],
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.body_wire())

    def to_wire(self) -> dict[str, object]:
        return self.body_wire()


@dataclass(frozen=True, slots=True, init=False)
class RNGInvocation:
    """One named, ordered RNG reset-boundary invocation.

    Semantic material is normalized and recursively frozen at construction so
    neither a caller nor a kernel can alter the behavior declared to the
    broker session.
    """

    boundary_key: str
    semantic_material: FrozenMap

    def __init__(
        self,
        boundary_key: str,
        semantic_material: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(boundary_key, str) or not boundary_key:
            raise BrokerContractError("RNG invocation boundary_key must be non-empty")
        normalized = _canonical_detail(
            {} if semantic_material is None else semantic_material
        )
        frozen = freeze_json(normalized)
        assert isinstance(frozen, FrozenMap)
        object.__setattr__(self, "boundary_key", boundary_key)
        object.__setattr__(self, "semantic_material", frozen)

    @property
    def semantic_material_sha256(self) -> str:
        return sha256_json(thaw_json(self.semantic_material))

    def to_wire(self) -> dict[str, object]:
        return {
            "boundary_key": self.boundary_key,
            "semantic_material": thaw_json(self.semantic_material),
            "semantic_material_sha256": self.semantic_material_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class RNGBehaviorIdentity:
    """Immutable semantic RNG behavior derived from the invocation plan."""

    protocol_id: str
    protocol_sha256: str
    owner: BrokerOwner
    sites: tuple[FrozenMap, ...]
    _issuer: object = field(repr=False, compare=False)
    _session_issuer: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        protocol_id: str,
        protocol_sha256: str,
        owner: BrokerOwner,
        sites: tuple[FrozenMap, ...],
        _issuer: object,
        _session_issuer: object,
    ) -> None:
        if _issuer is not _RNG_BEHAVIOR_ISSUER:
            raise BrokerContractError(
                "RNG behavior identities may only be issued by an RNG broker"
            )
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "protocol_sha256", protocol_sha256)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "_issuer", _issuer)
        object.__setattr__(self, "_session_issuer", _session_issuer)
        self.validate_issued()

    def validate_issued(self) -> None:
        if self._issuer is not _RNG_BEHAVIOR_ISSUER:
            raise BrokerContractError("RNG behavior identity issuer is invalid")
        if not self.protocol_id:
            raise BrokerContractError("RNG behavior protocol id must be non-empty")
        if (
            not isinstance(self.protocol_sha256, str)
            or len(self.protocol_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.protocol_sha256
            )
        ):
            raise BrokerContractError("RNG behavior protocol digest must be sha256")
        if not isinstance(self.owner, BrokerOwner):
            raise BrokerContractError("RNG behavior owner must be a BrokerOwner")
        if not isinstance(self.sites, tuple) or any(
            not isinstance(site, FrozenMap) for site in self.sites
        ):
            raise BrokerContractError("RNG behavior sites must be frozen objects")
        site_rows = [thaw_json(site) for site in self.sites]
        if any(not isinstance(site, dict) for site in site_rows):
            raise BrokerContractError("RNG behavior sites must be objects")
        common_fields = {
            "site_id",
            "stream",
            "contract_sha256",
            "invocations",
        }
        semantic_fields = {"run_input", "literal_seed", "dynamic_material"}
        for row in site_rows:
            assert isinstance(row, dict)
            selected = set(row) & semantic_fields
            if len(selected) != 1 or set(row) != common_fields | selected:
                raise BrokerContractError("RNG behavior site shape is not closed")
            contract_sha256 = row["contract_sha256"]
            if not isinstance(contract_sha256, str) or (
                len(contract_sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in contract_sha256
                )
            ):
                raise BrokerContractError(
                    "RNG behavior site contract digest is invalid"
                )
            stream = row["stream"]
            if not isinstance(stream, str) or not stream.startswith("stream:"):
                raise BrokerContractError("RNG behavior site stream is invalid")
            invocations = row["invocations"]
            if not isinstance(invocations, list):
                raise BrokerContractError("RNG behavior invocations must be an array")
            for invocation in invocations:
                if not isinstance(invocation, dict) or set(invocation) != {
                    "boundary_key",
                    "semantic_material",
                    "semantic_material_sha256",
                }:
                    raise BrokerContractError(
                        "RNG behavior invocation shape is not closed"
                    )
                if not invocation["boundary_key"]:
                    raise BrokerContractError(
                        "RNG behavior invocation boundary must be non-empty"
                    )
                if (
                    sha256_json(invocation["semantic_material"])
                    != invocation["semantic_material_sha256"]
                ):
                    raise BrokerContractError(
                        "RNG behavior invocation material digest differs"
                    )
        site_ids = [row.get("site_id") for row in site_rows]
        if any(not isinstance(site_id, str) or not site_id for site_id in site_ids):
            raise BrokerContractError("RNG behavior sites require non-empty ids")
        if len(site_ids) != len(set(site_ids)):
            raise BrokerContractError("RNG behavior site ids must be unique")

    def body_wire(self) -> dict[str, object]:
        return {
            "domain": _RNG_BEHAVIOR_DOMAIN,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "owner": self.owner.to_wire(),
            "sites": [thaw_json(site) for site in self.sites],
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.body_wire())

    def to_wire(self) -> dict[str, object]:
        return self.body_wire()


@dataclass(frozen=True, slots=True)
class RNGStreamToken:
    """An opaque, single-use grant for one exact ledger site and boundary."""

    protocol_id: str
    protocol_sha256: str
    owner: BrokerOwner
    site_id: str
    stream: str
    contract_sha256: str
    boundary_instance: int
    boundary_key: str
    semantic_material: FrozenMap
    _issuer: object = field(repr=False, compare=False)
    _activation: object = field(repr=False, compare=False)

    @property
    def semantic_material_sha256(self) -> str:
        return sha256_json(thaw_json(self.semantic_material))

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "owner": self.owner.to_wire(),
            "site_id": self.site_id,
            "stream": f"stream:{self.stream}",
            "contract_sha256": self.contract_sha256,
            "boundary_instance": self.boundary_instance,
            "boundary_key": self.boundary_key,
            "semantic_material": thaw_json(self.semantic_material),
            "semantic_material_sha256": self.semantic_material_sha256,
        }


@dataclass(frozen=True, slots=True)
class DerivedSeedHandle:
    """Opaque proof that a same-session ledger derivation was completed."""

    protocol_id: str
    protocol_sha256: str
    owner: BrokerOwner
    site_id: str
    boundary_key: str
    value_sha256: str
    _issuer: object = field(repr=False, compare=False)
    _activation: object = field(repr=False, compare=False)

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "owner": self.owner.to_wire(),
            "site_id": self.site_id,
            "boundary_key": self.boundary_key,
            "value_sha256": self.value_sha256,
        }


@dataclass(frozen=True, slots=True)
class BrokerAccessEvent:
    """One ordered operational broker decision."""

    sequence: int
    broker: Literal["rng", "file", "environment", "clock", "ambient"]
    operation: str
    resource: str
    disposition: Literal["allowed", "refused"]
    reason_code: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise BrokerContractError("broker event sequence must be non-negative")
        if self.broker not in _BROKER_KINDS:
            raise BrokerContractError(f"unknown broker event kind {self.broker!r}")
        if self.disposition not in _DISPOSITIONS:
            raise BrokerContractError(
                f"unknown broker event disposition {self.disposition!r}"
            )
        if any(
            not isinstance(value, str) or not value
            for value in (self.operation, self.resource, self.reason_code)
        ):
            raise BrokerContractError("broker events require closed descriptive fields")
        normalized = _canonical_detail(self.details)
        assert isinstance(normalized, dict)
        frozen = freeze_json(normalized)
        assert isinstance(frozen, FrozenMap)
        object.__setattr__(self, "details", frozen)

    def to_wire(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "broker": self.broker,
            "operation": self.operation,
            "resource": self.resource,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "details": thaw_json(self.details),
        }


@dataclass(frozen=True, slots=True)
class BrokerReceipt:
    """Closed operational receipt emitted when one session is sealed."""

    owner: BrokerOwner
    node_key: str | None
    attempt: int
    attempt_scope: str | None
    status: Literal["complete", "aborted"]
    run_provenance_identity: FrozenMap
    events: tuple[BrokerAccessEvent, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.owner, BrokerOwner):
            raise BrokerContractError("broker receipt owner must be a BrokerOwner")
        if self.node_key is not None and (
            not isinstance(self.node_key, str)
            or len(self.node_key) != 64
            or any(character not in "0123456789abcdef" for character in self.node_key)
        ):
            raise BrokerContractError("broker receipt node_key must be sha256 or null")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 0
        ):
            raise BrokerContractError("broker receipt attempt must be non-negative")
        if self.attempt_scope is not None and (
            not isinstance(self.attempt_scope, str) or not self.attempt_scope
        ):
            raise BrokerContractError(
                "broker receipt attempt_scope must be non-empty or null"
            )
        if self.status not in {"complete", "aborted"}:
            raise BrokerContractError(f"unknown broker receipt status {self.status!r}")
        if not isinstance(self.run_provenance_identity, FrozenMap):
            raise BrokerContractError(
                "broker receipt run provenance identity must be frozen"
            )
        _run_provenance_identity(thaw_json(self.run_provenance_identity))
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, BrokerAccessEvent) for event in self.events
        ):
            raise BrokerContractError("broker receipt events must be an event tuple")
        if self.receipt_sha256 and (
            not isinstance(self.receipt_sha256, str)
            or len(self.receipt_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.receipt_sha256
            )
        ):
            raise BrokerContractError("broker receipt digest must be sha256")

    def body_wire(self) -> dict[str, object]:
        return {
            "domain": _RECEIPT_DOMAIN,
            "schema_version": 1,
            "surface": "operational",
            "owner": self.owner.to_wire(),
            "node_key": self.node_key,
            "attempt": self.attempt,
            "attempt_scope": self.attempt_scope,
            "status": self.status,
            "run_provenance_identity": thaw_json(self.run_provenance_identity),
            "events": [event.to_wire() for event in self.events],
        }

    def to_wire(self) -> dict[str, object]:
        return {**self.body_wire(), "receipt_sha256": self.receipt_sha256}

    def validate(self) -> None:
        if not self.receipt_sha256:
            raise BrokerContractError("broker receipt digest is missing")
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise BrokerContractError("broker receipt event sequence is not canonical")
        if sha256_json(self.body_wire()) != self.receipt_sha256:
            raise BrokerContractError("broker receipt digest differs from its body")


class _AccessLog:
    def __init__(self) -> None:
        self._events: list[BrokerAccessEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        broker: Literal["rng", "file", "environment", "clock", "ambient"],
        operation: str,
        resource: str,
        disposition: Literal["allowed", "refused"],
        reason_code: str,
        details: Mapping[str, object] | None = None,
    ) -> BrokerAccessEvent:
        with self._lock:
            event = BrokerAccessEvent(
                sequence=len(self._events),
                broker=broker,
                operation=operation,
                resource=resource,
                disposition=disposition,
                reason_code=reason_code,
                details={} if details is None else details,
            )
            self._events.append(event)
            return event

    def events(self) -> tuple[BrokerAccessEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True, slots=True)
class _ActivePolicy:
    session: BrokerSession
    activation: object
    role: Literal["kernel", "row_classifier"]


_ACTIVE_POLICY: ContextVar[_ActivePolicy | None] = ContextVar(
    "microcosm_spec_engine_active_broker_policy", default=None
)
_AMBIENT_GUARD_LOCK = threading.RLock()
_GLOBAL_ACTIVE_POLICY: _ActivePolicy | None = None


def _current_policy() -> _ActivePolicy | None:
    return _ACTIVE_POLICY.get() or _GLOBAL_ACTIVE_POLICY


def _deny_ambient(kind: str, operation: str, resource: object = "ambient") -> None:
    policy = _current_policy()
    if policy is None:
        return
    rendered = str(resource)
    policy.session._log.record(
        broker="ambient",
        operation=f"{kind}.{operation}",
        resource=rendered,
        disposition="refused",
        reason_code="ambient_access_prohibited",
    )
    raise AmbientAccessError(
        f"ambient {kind} access {operation!r} is prohibited for "
        f"{policy.session.owner.kind} {policy.session.owner.id!r}"
    )


class _RefusingEnvironment(MutableMapping[str, str]):
    def __init__(self, wrapped: MutableMapping[str, str]) -> None:
        self._wrapped = wrapped

    def __getitem__(self, key: str) -> str:
        _deny_ambient("environment", "getitem", key)
        return self._wrapped[key]

    def __setitem__(self, key: str, value: str) -> None:
        _deny_ambient("environment", "setitem", key)
        self._wrapped[key] = value

    def __delitem__(self, key: str) -> None:
        _deny_ambient("environment", "delitem", key)
        del self._wrapped[key]

    def __iter__(self) -> Iterator[str]:
        _deny_ambient("environment", "iter")
        return iter(self._wrapped)

    def __len__(self) -> int:
        _deny_ambient("environment", "len")
        return len(self._wrapped)

    def get(self, key: str, default: str | None = None) -> str | None:
        _deny_ambient("environment", "get", key)
        return self._wrapped.get(key, default)

    def copy(self) -> dict[str, str]:
        _deny_ambient("environment", "copy")
        return dict(self._wrapped)


class _PinnedDependencyEnvironment(_RefusingEnvironment):
    """Expose only explicit unset defaults needed by deterministic libraries."""

    def __init__(self, wrapped: MutableMapping[str, str], session: BrokerSession):
        super().__init__(wrapped)
        self._session = session

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _PINNED_DEPENDENCY_ENVIRONMENT_DEFAULTS:
            return super().get(key, default)
        value = _PINNED_DEPENDENCY_ENVIRONMENT_DEFAULTS[key]
        self._session._log.record(
            broker="ambient",
            operation="physical_operation_dependency_environment",
            resource=key,
            disposition="allowed",
            reason_code="pinned_dependency_environment_default",
            details={"present": value is not None},
        )
        return default if value is None else value


class _RefusingDateTime(_ORIGINAL_DATETIME):
    @classmethod
    def now(cls, tz: object = None) -> Any:
        _deny_ambient("clock", "datetime.now")
        raise AssertionError("unreachable outside an active broker guard")

    @classmethod
    def utcnow(cls) -> Any:
        _deny_ambient("clock", "datetime.utcnow")
        raise AssertionError("unreachable outside an active broker guard")

    @classmethod
    def today(cls) -> Any:
        _deny_ambient("clock", "datetime.today")
        raise AssertionError("unreachable outside an active broker guard")


class _RefusingDate(_ORIGINAL_DATE):
    @classmethod
    def today(cls) -> Any:
        _deny_ambient("clock", "date.today")
        raise AssertionError("unreachable outside an active broker guard")


def _ambient_refusal(kind: str, operation: str):
    def refuse(*args: object, **_kwargs: object) -> Any:
        resource = args[0] if args else "ambient"
        _deny_ambient(kind, operation, resource)
        raise AssertionError("unreachable outside an active broker guard")

    return refuse


@contextmanager
def _ambient_guard(policy: _ActivePolicy) -> Iterator[None]:
    """Install a serialized, fail-closed guard around one Python dispatch."""

    global _GLOBAL_ACTIVE_POLICY
    with _AMBIENT_GUARD_LOCK:
        if _GLOBAL_ACTIVE_POLICY is not None:
            raise BrokerContractError("broker sessions may not overlap or nest")
        token = _ACTIVE_POLICY.set(policy)
        _GLOBAL_ACTIVE_POLICY = policy
        try:
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(os, "environ", _RefusingEnvironment(os.environ))
                )
                if hasattr(os, "environb") and os.environb is not None:
                    stack.enter_context(
                        patch.object(
                            os,
                            "environb",
                            _RefusingEnvironment(os.environb),
                        )
                    )
                for name in ("getenv", "putenv", "unsetenv"):
                    if hasattr(os, name):
                        stack.enter_context(
                            patch.object(
                                os,
                                name,
                                _ambient_refusal("environment", name),
                            )
                        )
                if hasattr(os, "getenvb"):
                    stack.enter_context(
                        patch.object(
                            os,
                            "getenvb",
                            _ambient_refusal("environment", "getenvb"),
                        )
                    )
                for name in _OS_AMBIENT_NAMES:
                    if hasattr(os, name):
                        stack.enter_context(
                            patch.object(
                                os,
                                name,
                                _ambient_refusal("file", f"os.{name}"),
                            )
                        )
                for name in (
                    "time",
                    "time_ns",
                    "monotonic",
                    "monotonic_ns",
                    "perf_counter",
                    "perf_counter_ns",
                    "process_time",
                    "process_time_ns",
                    "thread_time",
                    "thread_time_ns",
                    "clock_gettime",
                    "clock_gettime_ns",
                    "sleep",
                ):
                    if hasattr(time_module, name):
                        stack.enter_context(
                            patch.object(
                                time_module,
                                name,
                                _ambient_refusal("clock", name),
                            )
                        )
                stack.enter_context(
                    patch.object(datetime_module, "datetime", _RefusingDateTime)
                )
                stack.enter_context(
                    patch.object(datetime_module, "date", _RefusingDate)
                )
                stack.enter_context(
                    patch.object(
                        builtins,
                        "open",
                        _ambient_refusal("file", "builtins.open"),
                    )
                )
                stack.enter_context(
                    patch.object(io, "open", _ambient_refusal("file", "io.open"))
                )
                stack.enter_context(
                    patch.object(os, "open", _ambient_refusal("file", "os.open"))
                )
                for name in _NUMPY_RANDOM_AMBIENT_NAMES:
                    if hasattr(np.random, name):
                        stack.enter_context(
                            patch.object(
                                np.random,
                                name,
                                _ambient_refusal("rng", f"numpy.random.{name}"),
                            )
                        )
                stack.enter_context(
                    patch.object(
                        pd.DataFrame,
                        "sample",
                        _ambient_refusal("rng", "pandas.DataFrame.sample"),
                    )
                )
                for name in (
                    "Random",
                    "SystemRandom",
                    "seed",
                    "random",
                    "randint",
                    "randrange",
                    "choice",
                    "choices",
                    "uniform",
                    "shuffle",
                    "sample",
                    "getrandbits",
                ):
                    if hasattr(python_random, name):
                        stack.enter_context(
                            patch.object(
                                python_random,
                                name,
                                _ambient_refusal("rng", f"random.{name}"),
                            )
                        )
                for name in (
                    "token_bytes",
                    "token_hex",
                    "token_urlsafe",
                    "choice",
                    "randbelow",
                    "randbits",
                ):
                    if hasattr(secrets, name):
                        stack.enter_context(
                            patch.object(
                                secrets,
                                name,
                                _ambient_refusal("rng", f"secrets.{name}"),
                            )
                        )
                stack.enter_context(
                    patch.object(os, "urandom", _ambient_refusal("rng", "os.urandom"))
                )
                stack.enter_context(
                    patch.object(uuid, "uuid4", _ambient_refusal("rng", "uuid.uuid4"))
                )
                for module, kind, names in (
                    (
                        socket,
                        "network",
                        ("create_connection", "socket", "socketpair"),
                    ),
                    (
                        subprocess,
                        "process",
                        ("Popen", "call", "check_call", "check_output", "run"),
                    ),
                ):
                    for name in names:
                        if hasattr(module, name):
                            stack.enter_context(
                                patch.object(
                                    module,
                                    name,
                                    _ambient_refusal(kind, f"{module.__name__}.{name}"),
                                )
                            )
                torch_module = sys.modules.get("torch")
                if torch_module is not None:
                    for name in (
                        "Generator",
                        "bernoulli",
                        "initial_seed",
                        "manual_seed",
                        "multinomial",
                        "normal",
                        "poisson",
                        "rand",
                        "rand_like",
                        "randint",
                        "randint_like",
                        "randn",
                        "randn_like",
                        "random",
                        "seed",
                    ):
                        if hasattr(torch_module, name):
                            stack.enter_context(
                                patch.object(
                                    torch_module,
                                    name,
                                    _ambient_refusal("rng", f"torch.{name}"),
                                )
                            )
                    tensor_type = getattr(torch_module, "Tensor", None)
                    if tensor_type is not None and hasattr(tensor_type, "uniform_"):
                        stack.enter_context(
                            patch.object(
                                tensor_type,
                                "uniform_",
                                _ambient_refusal("rng", "torch.Tensor.uniform_"),
                            )
                        )
                yield
        finally:
            _GLOBAL_ACTIVE_POLICY = None
            _ACTIVE_POLICY.reset(token)


def _contains_raw_generator(value: object, *, _seen: set[int] | None = None) -> bool:
    """Return whether a draw result could transfer RNG authority to a kernel."""

    if isinstance(value, _ORIGINAL_GENERATOR | _ORIGINAL_RANDOM_STATE):
        return True
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, Mapping):
        return any(
            _contains_raw_generator(child, _seen=seen)
            for item in value.items()
            for child in item
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_raw_generator(child, _seen=seen) for child in value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return any(_contains_raw_generator(child, _seen=seen) for child in value.flat)
    return False


def _physical_sink_path(
    session: BrokerSession,
    operation: PhysicalOperation,
    value: object,
    *,
    primitive: str,
) -> Path:
    """Resolve one Python-level sink path without following a symlink escape."""

    if isinstance(value, int | bytes) or not isinstance(value, str | os.PathLike):
        session._log.record(
            broker="ambient",
            operation="physical_operation_sink_path",
            resource=primitive,
            disposition="refused",
            reason_code="physical_sink_path_shape_invalid",
        )
        raise BrokerAccessError(
            f"physical operation {primitive} requires a filesystem path"
        )
    with (
        patch.object(os, "lstat", _ORIGINAL_OS_AMBIENT["lstat"]),
        patch.object(os, "readlink", _ORIGINAL_OS_AMBIENT["readlink"]),
    ):
        candidate = Path(os.path.realpath(os.path.abspath(os.fspath(value))))
    containing_root = next(
        (root for root in operation.sink_roots if candidate.is_relative_to(root)),
        None,
    )
    if containing_root is None:
        session._log.record(
            broker="ambient",
            operation="physical_operation_sink_path",
            resource=primitive,
            disposition="refused",
            reason_code="physical_sink_path_outside_grant",
            details={"path_sha256": _sha256_bytes(os.fsencode(candidate))},
        )
        raise BrokerAccessError(
            f"physical operation {primitive} path is outside its sink roots"
        )
    return candidate


def _physical_sink_probe_path(
    session: BrokerSession,
    operation: PhysicalOperation,
    value: object,
    *,
    primitive: str,
) -> Path:
    """Allow read-only metadata probes inside or above a declared sink root."""

    if isinstance(value, int | bytes) or not isinstance(value, str | os.PathLike):
        session._log.record(
            broker="ambient",
            operation="physical_operation_sink_path",
            resource=primitive,
            disposition="refused",
            reason_code="physical_sink_path_shape_invalid",
        )
        raise BrokerAccessError(
            f"physical operation {primitive} requires a filesystem path"
        )
    lexical = Path(os.path.abspath(os.fspath(value)))
    with (
        patch.object(os, "lstat", _ORIGINAL_OS_AMBIENT["lstat"]),
        patch.object(os, "readlink", _ORIGINAL_OS_AMBIENT["readlink"]),
    ):
        resolved = Path(os.path.realpath(lexical))
    allowed = any(
        (
            lexical.is_relative_to(root)
            and resolved.is_relative_to(root)
        )
        or (
            root.is_relative_to(lexical)
            and root.is_relative_to(resolved)
        )
        for root in operation.sink_roots
    )
    if not allowed:
        session._log.record(
            broker="ambient",
            operation="physical_operation_sink_path",
            resource=primitive,
            disposition="refused",
            reason_code="physical_sink_path_outside_grant",
            details={"path_sha256": _sha256_bytes(os.fsencode(resolved))},
        )
        raise BrokerAccessError(
            f"physical operation {primitive} path is outside its sink roots"
        )
    return lexical


def _physical_seed_required(
    session: BrokerSession,
    operation: str,
    function: Callable[..., object],
) -> Callable[..., object]:
    """Refuse entropy-seeking legacy constructors inside a seeded scope."""

    @functools.wraps(function)
    def call(seed: object = None, *args: object, **kwargs: object) -> object:
        if seed is None:
            session._log.record(
                broker="ambient",
                operation="physical_operation_rng",
                resource=operation,
                disposition="refused",
                reason_code="physical_rng_entropy_prohibited",
            )
            raise BrokerAccessError(
                f"physical operation {operation} requires explicit seed material"
            )
        return function(seed, *args, **kwargs)

    return call


def _physical_seeded_constructor(
    session: BrokerSession,
    operation: str,
    constructor: type,
) -> type:
    """Proxy an RNG type while preserving ``isinstance`` compatibility."""

    class SeededConstructorMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, constructor)

        def __subclasscheck__(cls, subclass: type) -> bool:
            return issubclass(subclass, constructor)

    class SeededConstructor(metaclass=SeededConstructorMeta):
        def __new__(
            cls, seed: object = None, *args: object, **kwargs: object
        ) -> object:
            if seed is None:
                session._log.record(
                    broker="ambient",
                    operation="physical_operation_rng",
                    resource=operation,
                    disposition="refused",
                    reason_code="physical_rng_entropy_prohibited",
                )
                raise BrokerAccessError(
                    f"physical operation {operation} requires explicit seed material"
                )
            return constructor(seed, *args, **kwargs)

    SeededConstructor.__name__ = constructor.__name__
    SeededConstructor.__qualname__ = constructor.__qualname__
    return SeededConstructor


@contextmanager
def _physical_operation_compatibility_scope(
    session: BrokerSession,
    operation: PhysicalOperation,
    *,
    legacy_rng: bool,
) -> Iterator[None]:
    """Narrowly restore declared sinks and, only when requested, legacy RNG.

    The registered kernel remains under the ambient guard.  Only the broker's
    prebound operation runs in this nested scope, at most once. Source access
    remains denied and sink access is constrained to the declared roots.
    ``broker-only`` calls set ``legacy_rng=False`` and therefore retain the
    ambient RNG guard; their implementation-pinned third-party fits use typed
    lease methods instead.
    """

    with ExitStack() as stack:
        if not legacy_rng:
            pinned_environment = _PinnedDependencyEnvironment(
                _ORIGINAL_ENVIRON,
                session,
            )
            stack.enter_context(
                patch.object(
                    os,
                    "environ",
                    pinned_environment,
                )
            )
            stack.enter_context(patch.object(os, "getenv", pinned_environment.get))
        if legacy_rng and session.determinism == "seeded":
            for site_id in session.rng.granted_sites:
                session._log.record(
                    broker="rng",
                    operation="physical_operation_scope",
                    resource=site_id,
                    disposition="allowed",
                    reason_code="legacy_v1_physical_rng_grant",
                    details={
                        "implementation_sha256": operation.implementation_sha256,
                        "input_binding_sha256": operation.input_binding_sha256,
                        "protocol_sha256": session.rng.protocol_sha256,
                    },
                )
            for name in ("BitGenerator", "Generator"):
                original = _ORIGINAL_NUMPY_RANDOM.get(name)
                if original is not None:
                    stack.enter_context(patch.object(np.random, name, original))
            for name in ("PCG64", "RandomState", "SeedSequence"):
                original = _ORIGINAL_NUMPY_RANDOM.get(name)
                if isinstance(original, type):
                    stack.enter_context(
                        patch.object(
                            np.random,
                            name,
                            _physical_seeded_constructor(
                                session,
                                f"numpy.random.{name}",
                                original,
                            ),
                        )
                    )
            for name in ("default_rng",):
                original = _ORIGINAL_NUMPY_RANDOM.get(name)
                if original is not None:
                    stack.enter_context(
                        patch.object(
                            np.random,
                            name,
                            _physical_seed_required(
                                session,
                                f"numpy.random.{name}",
                                original,
                            ),
                        )
                    )

            @functools.wraps(_ORIGINAL_DATAFRAME_SAMPLE)
            def sample(
                frame: pd.DataFrame, *args: object, **kwargs: object
            ) -> pd.DataFrame:
                if kwargs.get("random_state") is None:
                    session._log.record(
                        broker="ambient",
                        operation="physical_operation_rng",
                        resource="pandas.DataFrame.sample",
                        disposition="refused",
                        reason_code="physical_rng_entropy_prohibited",
                    )
                    raise BrokerAccessError(
                        "physical pandas sampling requires explicit random_state"
                    )
                return _ORIGINAL_DATAFRAME_SAMPLE(frame, *args, **kwargs)

            stack.enter_context(patch.object(pd.DataFrame, "sample", sample))

        if "declared_sink_write" in session.effects:
            session._log.record(
                broker="file",
                operation="physical_operation_sink_scope",
                resource="declared_sink_roots",
                disposition="allowed",
                reason_code="legacy_v1_physical_sink_grant",
                details={
                    "implementation_sha256": operation.implementation_sha256,
                    "input_binding_sha256": operation.input_binding_sha256,
                    "root_count": len(operation.sink_roots),
                },
            )

            def open_file(file: object, *args: object, **kwargs: object) -> IO[Any]:
                path = _physical_sink_path(
                    session, operation, file, primitive="builtins.open"
                )
                return _ORIGINAL_BUILTINS_OPEN(path, *args, **kwargs)

            def io_open(file: object, *args: object, **kwargs: object) -> IO[Any]:
                path = _physical_sink_path(
                    session, operation, file, primitive="io.open"
                )
                return _ORIGINAL_IO_OPEN(path, *args, **kwargs)

            def path_open(path: Path, *args: object, **kwargs: object) -> IO[Any]:
                checked = _physical_sink_path(
                    session, operation, path, primitive="pathlib.Path.open"
                )
                return _ORIGINAL_PATH_OPEN(checked, *args, **kwargs)

            def os_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if dir_fd is not None:
                    session._log.record(
                        broker="ambient",
                        operation="physical_operation_sink_path",
                        resource="os.open",
                        disposition="refused",
                        reason_code="physical_sink_dir_fd_prohibited",
                    )
                    raise BrokerAccessError(
                        "physical operation os.open dir_fd is prohibited"
                    )
                checked = _physical_sink_path(
                    session, operation, path, primitive="os.open"
                )
                return _ORIGINAL_OS_OPEN(checked, flags, mode)

            stack.enter_context(patch.object(builtins, "open", open_file))
            stack.enter_context(patch.object(io, "open", io_open))
            stack.enter_context(patch.object(Path, "open", path_open))
            stack.enter_context(patch.object(os, "open", os_open))
            for name in ("getcwd", "getcwdb", "fstat"):
                original = _ORIGINAL_OS_AMBIENT.get(name)
                if original is not None:
                    stack.enter_context(patch.object(os, name, original))
            for name in (
                "access",
                "listdir",
                "lstat",
                "readlink",
                "scandir",
                "stat",
                "statvfs",
                "walk",
            ):
                original = _ORIGINAL_OS_AMBIENT.get(name)
                if original is None:
                    continue

                def path_call(
                    value: object = ".",
                    *args: object,
                    _name: str = name,
                    _original: Callable[..., object] = original,
                    **kwargs: object,
                ) -> object:
                    resolver = (
                        _physical_sink_probe_path
                        if _name in {"lstat", "readlink", "stat"}
                        else _physical_sink_path
                    )
                    checked = resolver(
                        session,
                        operation,
                        value,
                        primitive=f"os.{_name}",
                    )
                    return _original(checked, *args, **kwargs)

                stack.enter_context(patch.object(os, name, path_call))
            if legacy_rng:
                stack.enter_context(patch.object(os, "environ", _ORIGINAL_ENVIRON))
                if _ORIGINAL_ENVIRONB is not None and hasattr(os, "environb"):
                    stack.enter_context(
                        patch.object(os, "environb", _ORIGINAL_ENVIRONB)
                    )
                for name, original in _ORIGINAL_ENVIRONMENT_CALLS.items():
                    stack.enter_context(patch.object(os, name, original))
                for name, original in _ORIGINAL_OPERATIONAL_CLOCKS.items():
                    stack.enter_context(patch.object(time_module, name, original))
                for name, original in _ORIGINAL_SUBPROCESS.items():
                    stack.enter_context(patch.object(subprocess, name, original))
                stack.enter_context(
                    patch.object(os, "urandom", _ORIGINAL_OS_URANDOM)
                )
                stack.enter_context(patch.object(uuid, "uuid4", _ORIGINAL_UUID4))
        yield


class _GeneratorLeaseStore:
    """Session-owned serialized states; no raw generator survives a broker call."""

    __slots__ = ("_authorized_child_seeds", "_session", "_states")

    def __init__(self, session: BrokerSession) -> None:
        self._session = session
        self._states: dict[object, bytes] = {}
        self._authorized_child_seeds: dict[object, list[int]] = {}

    def register(self, generator: np.random.Generator) -> object:
        self._session._require_active()
        handle = object()
        self._states[handle] = pickle.dumps(
            copy.deepcopy(generator.bit_generator.state), protocol=5
        )
        self._authorized_child_seeds[handle] = []
        return handle

    def contains(self, handle: object) -> bool:
        return handle in self._states

    def close(self, handle: object) -> None:
        self._states.pop(handle, None)
        self._authorized_child_seeds.pop(handle, None)

    def close_all(self) -> None:
        self._states.clear()
        self._authorized_child_seeds.clear()

    def _decode_state(self, handle: object) -> dict[str, object]:
        try:
            payload = self._states[handle]
        except KeyError as error:
            raise BrokerAccessError("RNG lease is closed") from error
        state = pickle.loads(payload)  # noqa: S301 - broker-created in-memory bytes
        if not isinstance(state, dict):  # pragma: no cover - broker invariant
            raise BrokerContractError("serialized RNG state is not an object")
        return state

    def _materialize(self, handle: object) -> np.random.Generator:
        generator = _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(0))
        generator.bit_generator.state = copy.deepcopy(self._decode_state(handle))
        return generator

    def _store(self, handle: object, generator: np.random.Generator) -> None:
        if handle not in self._states:
            raise BrokerAccessError("RNG lease is closed")
        self._states[handle] = pickle.dumps(
            copy.deepcopy(generator.bit_generator.state), protocol=5
        )

    def state(
        self, handle: object, *, token: RNGStreamToken, label: str
    ) -> dict[str, object]:
        self._session._require_active()
        state = copy.deepcopy(self._decode_state(handle))
        self._session._log.record(
            broker="rng",
            operation="state_read",
            resource=token.site_id,
            disposition="allowed",
            reason_code="brokered_rng_state",
            details={
                "owner": token.owner.to_wire(),
                "lease": label,
                "state_sha256": sha256_json(state),
            },
        )
        return state

    def draw(
        self,
        handle: object,
        method: str,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        *,
        token: RNGStreamToken,
        label: str,
    ) -> Any:
        self._session._require_active()
        generator = self._materialize(handle)
        result = getattr(generator, method)(*args, **dict(kwargs))
        self._store(handle, generator)
        if method == "integers" and isinstance(result, int | np.integer) and not (
            isinstance(result, bool | np.bool_)
        ):
            self._authorized_child_seeds[handle].append(int(result))
        if _contains_raw_generator(result):
            self._session._log.record(
                broker="rng",
                operation="draw",
                resource=token.site_id,
                disposition="refused",
                reason_code="rng_authority_escape",
                details={"lease": label, "method": method},
            )
            raise BrokerAccessError("RNG draw returned generator authority")
        self._session._log.record(
            broker="rng",
            operation="draw",
            resource=token.site_id,
            disposition="allowed",
            reason_code="brokered_rng_draw",
            details={
                "owner": token.owner.to_wire(),
                "lease": label,
                "method": method,
            },
        )
        return result

    def consume_child_seed(self, handle: object, value: object) -> int:
        """Consume one scalar seed previously drawn from this exact lease."""

        self._session._require_active()
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise BrokerAccessError(
                "seeded estimator random_state must be an integer lease draw"
            )
        seed = int(value)
        try:
            authorized = self._authorized_child_seeds[handle]
            index = authorized.index(seed)
        except (KeyError, ValueError) as error:
            raise BrokerAccessError(
                "seeded estimator random_state was not drawn from this lease"
            ) from error
        authorized.pop(index)
        return seed


class GeneratorLease:
    """A session-bounded proxy containing no raw NumPy generator."""

    __slots__ = ("_session", "_handle", "_token", "_label", "_closed")

    def __init__(
        self,
        handle: object,
        *,
        session: BrokerSession,
        token: RNGStreamToken,
        label: str,
    ) -> None:
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_closed", False)

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            session = object.__getattribute__(self, "_session")
            token = object.__getattribute__(self, "_token")
            session._log.record(
                broker="ambient",
                operation="rng_lease_private_access",
                resource=token.site_id,
                disposition="refused",
                reason_code="rng_authority_internals_prohibited",
                details={"field": name},
            )
            raise BrokerAccessError("RNG lease internals are not accessible")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, _value: object) -> None:
        raise BrokerAccessError(f"RNG lease field {name!r} is immutable")

    @property
    def closed(self) -> bool:
        session = object.__getattribute__(self, "_session")
        handle = object.__getattribute__(self, "_handle")
        return object.__getattribute__(self, "_closed") or not (
            session._rng_leases.contains(handle)
        )

    def _check(self) -> None:
        session = object.__getattribute__(self, "_session")
        if self.closed or session.sealed:
            raise BrokerAccessError("RNG lease is closed")
        session._require_active()

    def __enter__(self) -> GeneratorLease:
        object.__getattribute__(self, "_check")()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        session = object.__getattribute__(self, "_session")
        handle = object.__getattribute__(self, "_handle")
        session._rng_leases.close(handle)
        object.__setattr__(self, "_closed", True)

    def bit_generator_state(self) -> dict[str, object]:
        object.__getattribute__(self, "_check")()
        session = object.__getattribute__(self, "_session")
        return session._rng_leases.state(
            object.__getattribute__(self, "_handle"),
            token=object.__getattribute__(self, "_token"),
            label=object.__getattribute__(self, "_label"),
        )

    def qrf_fit_n_jobs(self) -> int:
        """Return the constants-era unset fit-width without ambient env reads."""

        object.__getattribute__(self, "_check")()
        return -1

    def qrf_predict_workers(self) -> int:
        """Return the host width through a captured operational primitive."""

        object.__getattribute__(self, "_check")()
        return _ORIGINAL_OS_CPU_COUNT() or 1

    def draw_qrf_estimator(
        self,
        estimator: object,
        features: object,
        row_quantiles: object,
        *,
        grid: object,
        bounds: object,
        workers: int,
    ) -> object:
        """Draw a chunked grid through the pinned quantile-forest dependency."""

        object.__getattribute__(self, "_check")()
        session = object.__getattribute__(self, "_session")
        token = object.__getattribute__(self, "_token")
        estimator_type = type(estimator)
        identity = (estimator_type.__module__, estimator_type.__qualname__)
        module = sys.modules.get("quantile_forest._quantile_forest")
        installed_type = (
            None
            if module is None
            else getattr(module, "RandomForestQuantileRegressor", None)
        )
        if identity != (
            "quantile_forest._quantile_forest",
            "RandomForestQuantileRegressor",
        ) or estimator_type is not installed_type:
            session._log.record(
                broker="rng",
                operation="draw_qrf_estimator",
                resource=token.site_id,
                disposition="refused",
                reason_code="qrf_estimator_implementation_refused",
                details={"estimator": f"{identity[0]}.{identity[1]}"},
            )
            raise BrokerAccessError(
                "QRF prediction is not an implementation-pinned dependency"
            )
        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or not isinstance(row_quantiles, np.ndarray)
            or row_quantiles.ndim != 1
            or len(row_quantiles) != len(features)
            or not isinstance(grid, np.ndarray)
            or grid.ndim != 1
            or len(grid) < 2
            or isinstance(workers, bool)
            or not isinstance(workers, int)
            or workers < 1
            or not isinstance(bounds, tuple)
        ):
            raise BrokerAccessError("pinned QRF draw arguments are invalid")
        normalized_bounds: list[tuple[int, int]] = []
        cursor = 0
        for bound in bounds:
            if (
                not isinstance(bound, tuple)
                or len(bound) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bound
                )
            ):
                raise BrokerAccessError("pinned QRF draw bounds are invalid")
            start, stop = bound
            if start != cursor or stop <= start or stop > len(features):
                raise BrokerAccessError("pinned QRF draw bounds are not contiguous")
            normalized_bounds.append((start, stop))
            cursor = stop
        if cursor != len(features):
            raise BrokerAccessError("pinned QRF draw bounds do not cover the rows")
        counter = 0
        counter_lock = threading.Lock()
        operational_identity = sha256_json(token.to_wire()).encode("ascii")

        def dependency_urandom(size: int) -> bytes:
            nonlocal counter
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise BrokerAccessError("pinned dependency requested invalid bytes")
            output = bytearray()
            with counter_lock:
                while len(output) < size:
                    payload = (
                        b"microcosm-qrf-operational-id-v1\x00"
                        + operational_identity
                        + counter.to_bytes(8, "big")
                    )
                    output.extend(hashlib.sha256(payload).digest())
                    counter += 1
            return bytes(output[:size])

        output = np.empty(len(features), dtype=np.float64)

        def draw_chunk(bound: tuple[int, int]) -> None:
            start, stop = bound
            predictions = np.asarray(
                estimator_type.predict(
                    estimator,
                    features[start:stop],
                    quantiles=list(grid),
                )
            ).reshape(stop - start, len(grid))
            quantiles = row_quantiles[start:stop]
            upper = np.searchsorted(grid, quantiles, side="left")
            upper = np.clip(upper, 1, len(grid) - 1)
            lower = upper - 1
            grid_lo = grid[lower]
            grid_hi = grid[upper]
            span = grid_hi - grid_lo
            weight = np.where(span > 0, (quantiles - grid_lo) / span, 0.0)
            weight = np.clip(weight, 0.0, 1.0)
            rows = np.arange(len(quantiles))
            values_lo = predictions[rows, lower]
            values_hi = predictions[rows, upper]
            output[start:stop] = values_lo + weight * (values_hi - values_lo)

        saved_n_jobs = getattr(estimator, "n_jobs", None)
        with (
            patch.object(os, "urandom", dependency_urandom),
            patch.object(time_module, "time", _ORIGINAL_TIME),
        ):
            if workers <= 1 or len(normalized_bounds) <= 1:
                for bound in normalized_bounds:
                    draw_chunk(bound)
            else:
                estimator.n_jobs = 1
                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for _ in pool.map(draw_chunk, normalized_bounds):
                            pass
                finally:
                    estimator.n_jobs = saved_n_jobs
        if _contains_raw_generator(output):
            raise BrokerAccessError("QRF prediction returned unexpected authority")
        session._log.record(
            broker="rng",
            operation="draw_qrf_estimator",
            resource=token.site_id,
            disposition="allowed",
            reason_code="brokered_qrf_estimator_draw",
            details={
                "owner": token.owner.to_wire(),
                "estimator": f"{identity[0]}.{identity[1]}",
                "chunk_count": len(normalized_bounds),
                "workers": workers,
            },
        )
        return output

    def fit_seeded_qrf_estimator(
        self,
        estimator: object,
        features: object,
        targets: object,
        *,
        sample_weight: object | None = None,
    ) -> object:
        """Fit one exact QRF dependency using a seed drawn from this lease.

        The callback cannot supply an arbitrary callable. Only the installed
        quantile forest and sign-gate implementation classes are accepted, and
        their explicit ``random_state`` must be a scalar value already drawn
        from this lease. The dependency's module-level Python shuffle is backed
        by a call-local ``Random`` instance, never by ambient module state.
        """

        object.__getattribute__(self, "_check")()
        session = object.__getattribute__(self, "_session")
        token = object.__getattribute__(self, "_token")
        handle = object.__getattribute__(self, "_handle")
        estimator_type = type(estimator)
        identity = (estimator_type.__module__, estimator_type.__qualname__)
        allowed = {
            (
                "quantile_forest._quantile_forest",
                "RandomForestQuantileRegressor",
            ): ("quantile_forest._quantile_forest", "RandomForestQuantileRegressor"),
            (
                "sklearn.ensemble._hist_gradient_boosting.gradient_boosting",
                "HistGradientBoostingClassifier",
            ): (
                "sklearn.ensemble._hist_gradient_boosting.gradient_boosting",
                "HistGradientBoostingClassifier",
            ),
        }
        module_name, class_name = allowed.get(identity, (None, None))
        module = None if module_name is None else sys.modules.get(module_name)
        installed_type = None if module is None else getattr(module, class_name, None)
        if estimator_type is not installed_type:
            session._log.record(
                broker="rng",
                operation="fit_seeded_qrf_estimator",
                resource=token.site_id,
                disposition="refused",
                reason_code="qrf_estimator_implementation_refused",
                details={"estimator": f"{identity[0]}.{identity[1]}"},
            )
            raise BrokerAccessError(
                "seeded QRF estimator is not an implementation-pinned dependency"
            )
        seed = session._rng_leases.consume_child_seed(
            handle,
            getattr(estimator, "random_state", None),
        )
        local_python_rng = _ORIGINAL_PYTHON_RANDOM_CLASS(0)
        fit = estimator_type.fit
        kwargs = {}
        if identity[1] == "HistGradientBoostingClassifier":
            if getattr(estimator, "warm_start", None) is not False:
                raise BrokerAccessError(
                    "pinned sign-gate adapter requires warm_start=False"
                )
            kwargs["sample_weight"] = sample_weight
        elif sample_weight is not None:
            raise BrokerAccessError(
                "quantile-forest compatibility fit does not accept sample weights"
            )

        def dependency_default_rng(seed: object = None) -> np.random.Generator:
            if seed is None:
                raise BrokerAccessError(
                    "pinned QRF dependency requested an entropy-seeded generator"
                )
            return _ORIGINAL_DEFAULT_RNG(seed)

        operational_counter = 0
        operational_identity = sha256_json(token.to_wire()).encode("ascii")

        def dependency_urandom(size: int) -> bytes:
            nonlocal operational_counter
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise BrokerAccessError("pinned dependency requested invalid bytes")
            output = bytearray()
            while len(output) < size:
                payload = (
                    b"microcosm-qrf-operational-id-v1\x00"
                    + operational_identity
                    + operational_counter.to_bytes(8, "big")
                )
                output.extend(hashlib.sha256(payload).digest())
                operational_counter += 1
            return bytes(output[:size])

        with (
            patch.object(np.random, "RandomState", _ORIGINAL_RANDOM_STATE),
            patch.object(np.random, "default_rng", dependency_default_rng),
            patch.object(python_random, "seed", local_python_rng.seed),
            patch.object(python_random, "shuffle", local_python_rng.shuffle),
            patch.object(os, "urandom", dependency_urandom),
            patch.object(time_module, "time", _ORIGINAL_TIME),
        ):
            result = fit(estimator, features, targets, **kwargs)
        if identity[1] == "HistGradientBoostingClassifier":
            feature_rng = getattr(estimator, "_feature_subsample_rng", None)
            if not isinstance(feature_rng, _ORIGINAL_GENERATOR):
                raise BrokerAccessError(
                    "pinned sign-gate fit did not retain its expected generator"
                )
            # The generator is fit-only state when warm_start is disabled.
            # Strip it before returning the estimator so raw RNG authority
            # cannot escape this exact dependency adapter.
            estimator._feature_subsample_rng = None
        if result is not estimator or _contains_raw_generator(result):
            raise BrokerAccessError(
                "seeded QRF estimator fit returned unexpected authority"
            )
        session._log.record(
            broker="rng",
            operation="fit_seeded_qrf_estimator",
            resource=token.site_id,
            disposition="allowed",
            reason_code="brokered_seeded_qrf_estimator_fit",
            details={
                "owner": token.owner.to_wire(),
                "estimator": f"{identity[0]}.{identity[1]}",
                "seed_sha256": _sha256_bytes(str(seed).encode()),
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        object.__getattribute__(self, "_check")()
        session = object.__getattribute__(self, "_session")
        token = object.__getattribute__(self, "_token")
        label = object.__getattribute__(self, "_label")
        if name not in _SAFE_GENERATOR_DRAW_METHODS:
            session._log.record(
                broker="rng",
                operation="draw",
                resource=token.site_id,
                disposition="refused",
                reason_code="unsafe_rng_method",
                details={"lease": label, "method": name},
            )
            raise BrokerAccessError(
                f"RNG draw method {name!r} is not in the safe broker allowlist"
            )

        def brokered_call(*args: object, **kwargs: object) -> Any:
            object.__getattribute__(self, "_check")()
            return session._rng_leases.draw(
                object.__getattribute__(self, "_handle"),
                name,
                args,
                kwargs,
                token=token,
                label=label,
            )

        return brokered_call


class _TorchGeneratorLeaseStore:
    """Session-owned serialized CPU Torch generator states."""

    __slots__ = (
        "_empty",
        "_from_numpy",
        "_generator_type",
        "_module",
        "_session",
        "_states",
        "_uniform",
    )

    def __init__(self, session: BrokerSession) -> None:
        self._session = session
        self._states: dict[object, bytes] = {}
        self._module: object | None = None
        self._generator_type: object | None = None
        self._empty: object | None = None
        self._from_numpy: object | None = None
        self._uniform: object | None = None
        self.prepare()

    def prepare(self) -> None:
        if self._module is not None:
            return
        module = sys.modules.get("torch")
        if module is None:
            return
        generator_type = getattr(module, "Generator", None)
        tensor_type = getattr(module, "Tensor", None)
        empty = getattr(module, "empty", None)
        from_numpy = getattr(module, "from_numpy", None)
        uniform = (
            None if tensor_type is None else getattr(tensor_type, "uniform_", None)
        )
        if not all(
            callable(value) for value in (generator_type, empty, from_numpy, uniform)
        ):
            raise BrokerContractError("loaded Torch RNG surface is incomplete")
        self._module = module
        self._generator_type = generator_type
        self._empty = empty
        self._from_numpy = from_numpy
        self._uniform = uniform

    def _require_backend(self) -> None:
        self.prepare()
        if self._module is None:
            raise BrokerContractError("Torch must be loaded before broker activation")

    def _new_generator(self) -> object:
        self._require_backend()
        assert callable(self._generator_type)
        return self._generator_type(device="cpu")

    @staticmethod
    def _state_bytes(generator: object) -> bytes:
        state = generator.get_state()  # type: ignore[attr-defined]
        return state.cpu().numpy().tobytes()

    def register(self, seed: int) -> object:
        self._session._require_active()
        generator = self._new_generator()
        generator.manual_seed(seed)  # type: ignore[attr-defined]
        handle = object()
        self._states[handle] = self._state_bytes(generator)
        return handle

    def contains(self, handle: object) -> bool:
        return handle in self._states

    def close(self, handle: object) -> None:
        self._states.pop(handle, None)

    def close_all(self) -> None:
        self._states.clear()

    def _materialize(self, handle: object) -> object:
        try:
            payload = self._states[handle]
        except KeyError as error:
            raise BrokerAccessError("Torch RNG lease is closed") from error
        generator = self._new_generator()
        assert callable(self._from_numpy)
        state = self._from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
        generator.set_state(state)  # type: ignore[attr-defined]
        return generator

    def uniform(
        self,
        handle: object,
        shape: Sequence[int],
        *,
        low: float,
        high: float,
        dtype: object | None,
        token: RNGStreamToken,
    ) -> object:
        self._session._require_active()
        normalized_shape = tuple(shape)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in normalized_shape
        ):
            raise BrokerContractError("Torch uniform shape must be non-negative ints")
        if not np.isfinite(low) or not np.isfinite(high) or not low < high:
            raise BrokerContractError("Torch uniform bounds must be finite and ordered")
        generator = self._materialize(handle)
        assert callable(self._empty) and callable(self._uniform)
        kwargs: dict[str, object] = {"device": "cpu"}
        if dtype is not None:
            kwargs["dtype"] = dtype
        result = self._empty(normalized_shape, **kwargs)
        self._uniform(result, low, high, generator=generator)
        if handle not in self._states:
            raise BrokerAccessError("Torch RNG lease is closed")
        self._states[handle] = self._state_bytes(generator)
        self._session._log.record(
            broker="rng",
            operation="draw",
            resource=token.site_id,
            disposition="allowed",
            reason_code="brokered_torch_rng_draw",
            details={
                "owner": token.owner.to_wire(),
                "method": "Tensor.uniform_",
                "shape": list(normalized_shape),
                "dtype": str(getattr(result, "dtype", dtype)),
            },
        )
        return result


class TorchGeneratorLease:
    """Session-bounded Torch uniform stream with no raw generator or seed."""

    __slots__ = ("_closed", "_handle", "_session", "_token")

    def __init__(
        self,
        handle: object,
        *,
        session: BrokerSession,
        token: RNGStreamToken,
    ) -> None:
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_closed", False)

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            session = object.__getattribute__(self, "_session")
            token = object.__getattribute__(self, "_token")
            session._log.record(
                broker="ambient",
                operation="torch_rng_lease_private_access",
                resource=token.site_id,
                disposition="refused",
                reason_code="rng_authority_internals_prohibited",
                details={"field": name},
            )
            raise BrokerAccessError("Torch RNG lease internals are not accessible")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, _value: object) -> None:
        raise BrokerAccessError(f"Torch RNG lease field {name!r} is immutable")

    @property
    def closed(self) -> bool:
        session = object.__getattribute__(self, "_session")
        return object.__getattribute__(self, "_closed") or not (
            session._torch_rng_leases.contains(object.__getattribute__(self, "_handle"))
        )

    def _check(self) -> None:
        session = object.__getattribute__(self, "_session")
        if self.closed or session.sealed:
            raise BrokerAccessError("Torch RNG lease is closed")
        session._require_active()

    def __enter__(self) -> TorchGeneratorLease:
        object.__getattribute__(self, "_check")()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        session = object.__getattribute__(self, "_session")
        session._torch_rng_leases.close(object.__getattribute__(self, "_handle"))
        object.__setattr__(self, "_closed", True)

    def uniform(
        self,
        shape: Sequence[int],
        *,
        low: float = 0.0,
        high: float = 1.0,
        dtype: object | None = None,
    ) -> object:
        object.__getattribute__(self, "_check")()
        session = object.__getattribute__(self, "_session")
        return session._torch_rng_leases.uniform(
            object.__getattribute__(self, "_handle"),
            shape,
            low=low,
            high=high,
            dtype=dtype,
            token=object.__getattribute__(self, "_token"),
        )


@dataclass(frozen=True, slots=True)
class QRFGeneratorLease:
    """The exact legacy-v1 fit-child/draw-child pair for one QRF boundary."""

    fit: GeneratorLease
    draw: GeneratorLease

    def __enter__(self) -> QRFGeneratorLease:
        self.fit.__enter__()
        self.draw.__enter__()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.fit.close()
        self.draw.close()


@dataclass(frozen=True, slots=True)
class _RNGBrokerAuthority:
    protocol_id: str
    protocol_sha256: str


class RNGBroker:
    """Ledger-driven typed RNG adapter for one broker session."""

    __slots__ = (
        "_authority",
        "_behavior_identity",
        "_consumed",
        "_consumed_count",
        "_contracts",
        "_declared_sites",
        "_derived_seeds",
        "_invocation_plan",
        "_issued",
        "_issuer",
        "_owner",
        "_run_inputs",
        "_session",
        "_sites",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise BrokerContractError(f"RNG broker field {name!r} is immutable")
        if name in {"protocol_id", "protocol_sha256"}:
            raise BrokerContractError(f"RNG broker field {name!r} is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        session: BrokerSession,
        owner: BrokerOwner,
        protocol_id: str,
        protocol_sha256: str,
        sites: Sequence[SeedSiteIR],
        run_inputs: Mapping[str, int],
        invocation_plan: Mapping[str, Sequence[RNGInvocation]],
    ) -> None:
        self._session = session
        self._owner = owner
        self._authority = _RNGBrokerAuthority(protocol_id, protocol_sha256)
        self._issuer = object()
        normalized_sites = {site.id: site for site in sites}
        if len(normalized_sites) != len(sites):
            raise BrokerContractError("RNG grants contain duplicate site ids")
        self._sites = MappingProxyType(normalized_sites)
        normalized_contracts: dict[str, FrozenMap] = {}
        for site in sites:
            contract = _wire(site.contract)
            if not isinstance(contract, Mapping) or set(contract) != (
                _SEED_SITE_CONTRACT_FIELDS
            ):
                raise BrokerContractError(
                    f"RNG site {site.id!r} does not have the closed ledger contract"
                )
            if not isinstance(contract["seed_material"], list) or not isinstance(
                contract["consumption_order"], list
            ):
                raise BrokerContractError(
                    f"RNG site {site.id!r} has non-array ordering metadata"
                )
            frozen_contract = freeze_json(dict(contract))
            assert isinstance(frozen_contract, FrozenMap)
            normalized_contracts[site.id] = frozen_contract
        self._contracts = MappingProxyType(normalized_contracts)
        normalized_inputs: dict[str, int] = {}
        for key, value in run_inputs.items():
            normalized_inputs[str(key)] = self._uint64(value, location=str(key))
        self._run_inputs = MappingProxyType(normalized_inputs)
        self._declared_sites = frozenset(invocation_plan)
        self._invocation_plan = self._normalize_invocation_plan(invocation_plan)
        self._issued: dict[str, int] = {site_id: 0 for site_id in self._sites}
        self._consumed_count: dict[str, int] = {site_id: 0 for site_id in self._sites}
        self._consumed: set[tuple[str, int]] = set()
        self._derived_seeds: dict[tuple[str, str], int] = {}
        self._behavior_identity = self._build_behavior_identity()

    @property
    def protocol_id(self) -> str:
        return self._authority.protocol_id

    @property
    def protocol_sha256(self) -> str:
        return self._authority.protocol_sha256

    @staticmethod
    def _uint64(value: object, *, location: str) -> int:
        if (
            not isinstance(value, int | np.integer)
            or isinstance(value, bool)
            or not 0 <= int(value) <= 2**64 - 1
        ):
            raise BrokerContractError(f"{location} must be a uint64 integer")
        return int(value)

    @property
    def granted_sites(self) -> tuple[str, ...]:
        return tuple(self._sites)

    @property
    def granted_streams(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(site.stream for site in self._sites.values()))

    @staticmethod
    def _requires_explicit_invocation(contract: Mapping[str, object]) -> bool:
        source = contract["value_source"]
        family = contract["rng_family"]
        return bool(
            source in {"derived_acs_pattern_seed", "stable_string"}
            or "time_period" in contract["seed_material"]
            or family
            in {
                "SHA-256 derived integer",
                "hashlib.blake2b stateless uniform",
                "pandas.util.hash_pandas_object stateless uint64",
                "pandas.DataFrame.sample RandomState(MT19937)",
            }
            or any(
                "{" in value and "}" in value
                for value in contract["seed_material"]
                if isinstance(value, str)
            )
        )

    @staticmethod
    def _digest(value: object, *, location: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BrokerContractError(f"{location} must be a lowercase SHA-256")
        return value

    @staticmethod
    def _salt_fields(contract: Mapping[str, object]) -> tuple[str, ...]:
        salt_rows = [
            value.removeprefix("literal_salt=")
            for value in contract["seed_material"]
            if isinstance(value, str) and value.startswith("literal_salt=")
        ]
        if len(salt_rows) != 1:
            raise BrokerContractError("BLAKE2b RNG sites require one literal salt")
        return tuple(
            field_name
            for _literal, field_name, _format_spec, _conversion in Formatter().parse(
                salt_rows[0]
            )
            if field_name is not None
        )

    def _validate_invocation(
        self,
        *,
        site: SeedSiteIR,
        contract: Mapping[str, object],
        invocation: RNGInvocation,
    ) -> None:
        material = thaw_json(invocation.semantic_material)
        assert isinstance(material, dict)
        required: set[str] = set()
        allowed: set[str] = set()
        source = contract["value_source"]
        family = contract["rng_family"]
        derivation = str(contract["derivation"])
        if source == "derived_acs_pattern_seed":
            required.add("derived_from")
        elif source == "stable_string":
            required.add("stable_key")
        if derivation == "default_rng(SeedSequence([seed,time_period,374]))":
            required.add("time_period")
        if family == "numpy.random.Generator(PCG64)":
            allowed.add("restored_state")
        elif family == "microcosm.fit.QRF SeedSequence(PCG64)" and (
            "spawn(3)" not in derivation
        ):
            allowed.update({"restored_fit_state", "restored_draw_state"})
        elif family == "SHA-256 derived integer":
            required.update(self._sha_dynamic_fields(contract))
        elif family == "hashlib.blake2b stateless uniform":
            required.update(self._salt_fields(contract))
            required.add("stable_keys_sha256")
        elif family == "pandas.util.hash_pandas_object stateless uint64":
            required.add("frame_sha256")
        elif family == "pandas.DataFrame.sample RandomState(MT19937)":
            required.add("stage_training_cap")
        allowed.update(required)
        missing = sorted(required - set(material))
        extra = sorted(set(material) - allowed)
        if missing or extra:
            raise BrokerContractError(
                f"RNG invocation {site.id!r}/{invocation.boundary_key!r} material "
                f"differs from the ledger; missing={missing!r}, extra={extra!r}"
            )
        if "derived_from" in material:
            derived_from = material["derived_from"]
            if not isinstance(derived_from, dict) or set(derived_from) != {
                "site_id",
                "boundary_key",
            }:
                raise BrokerContractError(
                    f"RNG site {site.id!r} derived_from must name one site boundary"
                )
            expected_sources = tuple(contract["seed_material"])
            if len(expected_sources) != 1 or (
                derived_from["site_id"] != expected_sources[0]
            ):
                raise BrokerContractError(
                    f"RNG site {site.id!r} derived source differs from the ledger"
                )
            source_contract = self._contracts.get(str(derived_from["site_id"]))
            if source_contract is None or (
                source_contract["rng_family"] != "SHA-256 derived integer"
            ):
                raise BrokerContractError(
                    f"RNG site {site.id!r} derived source is not a granted SHA site"
                )
            if (
                not isinstance(derived_from["boundary_key"], str)
                or not derived_from["boundary_key"]
            ):
                raise BrokerContractError(
                    f"RNG site {site.id!r} derived boundary must be non-empty"
                )
        if "time_period" in material:
            self._uint64(material["time_period"], location=f"{site.id}/time_period")
        if "stage_training_cap" in material:
            cap = material["stage_training_cap"]
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise BrokerContractError(
                    f"RNG site {site.id!r} stage_training_cap must be positive"
                )
        if "stable_key" in material and (
            not isinstance(material["stable_key"], str) or not material["stable_key"]
        ):
            raise BrokerContractError(
                f"RNG site {site.id!r} requires non-empty stable_key material"
            )
        if family == "SHA-256 derived integer":
            for name in self._sha_dynamic_fields(contract):
                component = material[name]
                if not isinstance(component, str | int) or isinstance(component, bool):
                    raise BrokerContractError(
                        f"RNG site {site.id!r} SHA component {name!r} must be a "
                        "string or integer"
                    )
                if (
                    isinstance(component, str)
                    and not component
                    and name != ("nul_joined_ordered_optional_predictors")
                ):
                    raise BrokerContractError(
                        f"RNG site {site.id!r} SHA component {name!r} is empty"
                    )
        for name in ("stable_keys_sha256", "frame_sha256"):
            if name in material:
                self._digest(material[name], location=f"{site.id}/{name}")
        for name in (
            "restored_state",
            "restored_fit_state",
            "restored_draw_state",
        ):
            if name in material and not isinstance(material[name], Mapping):
                raise BrokerContractError(
                    f"RNG site {site.id!r} {name} must be a state object"
                )

    @staticmethod
    def _sha_dynamic_fields(contract: Mapping[str, object]) -> tuple[str, ...]:
        material = contract["seed_material"]
        source = contract["value_source"]
        if (
            not isinstance(material, Sequence)
            or isinstance(material, str | bytes)
            or not material
        ):
            raise BrokerContractError("SHA RNG sites require ordered seed material")
        if not isinstance(source, str) or not source.startswith("run_request."):
            raise BrokerContractError("SHA RNG sites require a run-request base seed")
        base_name = source.removeprefix("run_request.")
        if material[0] != base_name or any(
            not isinstance(name, str) or not name for name in material
        ):
            raise BrokerContractError(
                "SHA RNG seed material must begin with its run-request seed"
            )
        return tuple(material[1:])

    def _normalize_invocation_plan(
        self, declared: Mapping[str, Sequence[RNGInvocation]]
    ) -> Mapping[str, tuple[RNGInvocation, ...]]:
        unknown = sorted(set(declared) - set(self._sites))
        if unknown:
            raise BrokerContractError(
                f"RNG invocation plan contains ungranted sites: {unknown!r}"
            )
        normalized: dict[str, tuple[RNGInvocation, ...]] = {}
        for site_id, site in self._sites.items():
            contract = self._contracts[site_id]
            if site_id in declared:
                raw_invocations = declared[site_id]
                if isinstance(raw_invocations, str | bytes):
                    raise BrokerContractError(
                        f"RNG invocation plan for {site_id!r} must be an array"
                    )
                invocations = tuple(raw_invocations)
            elif self._requires_explicit_invocation(contract):
                invocations = ()
            else:
                invocations = (RNGInvocation(_DEFAULT_RNG_BOUNDARY_KEY),)
            if any(not isinstance(item, RNGInvocation) for item in invocations):
                raise BrokerContractError(
                    f"RNG invocation plan for {site_id!r} must contain "
                    "RNGInvocation values"
                )
            for invocation in invocations:
                self._validate_invocation(
                    site=site, contract=contract, invocation=invocation
                )
            normalized[site_id] = invocations
        return MappingProxyType(normalized)

    def _build_behavior_identity(self) -> RNGBehaviorIdentity:
        site_rows: list[FrozenMap] = []
        for site_id, site in self._sites.items():
            contract = self._contracts[site_id]
            source = contract["value_source"]
            row: dict[str, object] = {
                "site_id": site_id,
                "stream": f"stream:{site.stream}",
                "contract_sha256": sha256_json(contract),
                "invocations": [
                    invocation.to_wire()
                    for invocation in self._invocation_plan[site_id]
                ],
            }
            if isinstance(source, str) and source.startswith("run_request."):
                key = source.removeprefix("run_request.")
                row["run_input"] = {
                    "name": key,
                    "value": self._run_inputs.get(key, contract["default"]),
                }
            elif source == "literal":
                row["literal_seed"] = contract["default"]
            else:
                row["dynamic_material"] = list(contract["seed_material"])
            frozen = freeze_json(_canonical_detail(row))
            assert isinstance(frozen, FrozenMap)
            site_rows.append(frozen)
        return RNGBehaviorIdentity(
            protocol_id=self.protocol_id,
            protocol_sha256=self.protocol_sha256,
            owner=self._owner,
            sites=tuple(site_rows),
            _issuer=_RNG_BEHAVIOR_ISSUER,
            _session_issuer=self._session._behavior_issuer,
        )

    @property
    def behavior_identity(self) -> RNGBehaviorIdentity:
        return self._behavior_identity

    def _unconsumed_declared_invocations(
        self,
    ) -> tuple[tuple[str, int, int], ...]:
        """Return explicitly planned invocation counts that were not consumed."""

        return tuple(
            (
                site_id,
                self._consumed_count[site_id],
                len(self._invocation_plan[site_id]),
            )
            for site_id in self._sites
            if site_id in self._declared_sites
            and self._consumed_count[site_id] < len(self._invocation_plan[site_id])
        )

    def behavior_input_wire(self) -> dict[str, object]:
        """Compatibility wire view of the immutable behavior identity."""

        return self._behavior_identity.to_wire()

    def token(
        self,
        site_id: str,
        boundary_key: str = _DEFAULT_RNG_BOUNDARY_KEY,
    ) -> RNGStreamToken:
        self._session._require_active()
        try:
            site = self._sites[site_id]
        except KeyError as error:
            self._session._log.record(
                broker="rng",
                operation="token",
                resource=site_id,
                disposition="refused",
                reason_code="undeclared_rng_site",
            )
            raise BrokerAccessError(f"RNG site {site_id!r} is not granted") from error
        boundary = self._issued[site_id]
        invocations = self._invocation_plan[site_id]
        if boundary >= len(invocations):
            self._session._log.record(
                broker="rng",
                operation="token",
                resource=site_id,
                disposition="refused",
                reason_code="rng_invocation_plan_exhausted",
                details={"declared_invocations": len(invocations)},
            )
            raise BrokerAccessError(
                f"RNG site {site_id!r} invocation plan is exhausted"
            )
        invocation = invocations[boundary]
        if boundary_key != invocation.boundary_key:
            self._session._log.record(
                broker="rng",
                operation="token",
                resource=site_id,
                disposition="refused",
                reason_code="rng_invocation_order_mismatch",
                details={
                    "expected_boundary_key": invocation.boundary_key,
                    "requested_boundary_key": boundary_key,
                },
            )
            raise BrokerAccessError(
                f"RNG site {site_id!r} expected boundary "
                f"{invocation.boundary_key!r}, got {boundary_key!r}"
            )
        self._issued[site_id] += 1
        activation = self._session._active_activation
        assert activation is not None
        contract_sha256 = sha256_json(self._contracts[site_id])
        token = RNGStreamToken(
            protocol_id=self.protocol_id,
            protocol_sha256=self.protocol_sha256,
            owner=self._owner,
            site_id=site.id,
            stream=site.stream,
            contract_sha256=contract_sha256,
            boundary_instance=boundary,
            boundary_key=invocation.boundary_key,
            semantic_material=invocation.semantic_material,
            _issuer=self._issuer,
            _activation=activation,
        )
        self._session._log.record(
            broker="rng",
            operation="token",
            resource=site_id,
            disposition="allowed",
            reason_code="compiled_rng_grant",
            details=token.to_wire(),
        )
        return token

    def _consume(
        self,
        token: RNGStreamToken,
        *,
        family: str,
        operation: str,
    ) -> tuple[SeedSiteIR, Mapping[str, object]]:
        self._session._require_active()
        reason: str | None = None
        site = self._sites.get(token.site_id)
        contract = self._contracts.get(token.site_id)
        invocation: RNGInvocation | None = None
        if token._issuer is not self._issuer:
            reason = "foreign_rng_token"
        elif token._activation is not self._session._active_activation:
            reason = "expired_rng_token"
        elif token.protocol_id != self.protocol_id or (
            token.protocol_sha256 != self.protocol_sha256
        ):
            reason = "rng_protocol_mismatch"
        elif token.owner != self._owner:
            reason = "rng_owner_mismatch"
        elif site is None or contract is None:
            reason = "undeclared_rng_site"
        elif token.stream != site.stream:
            reason = "rng_stream_mismatch"
        elif token.contract_sha256 != sha256_json(contract):
            reason = "rng_contract_mismatch"
        elif contract["rng_family"] != family:
            reason = "rng_family_mismatch"
        elif (
            not isinstance(token.boundary_instance, int)
            or isinstance(token.boundary_instance, bool)
            or not 0
            <= token.boundary_instance
            < len(self._invocation_plan[token.site_id])
        ):
            reason = "rng_invocation_mismatch"
        else:
            invocation = self._invocation_plan[token.site_id][token.boundary_instance]
        if reason is None and invocation is not None:
            if token.boundary_key != invocation.boundary_key:
                reason = "rng_boundary_key_mismatch"
            elif token.semantic_material != invocation.semantic_material:
                reason = "rng_semantic_material_mismatch"
            elif (token.site_id, token.boundary_instance) in self._consumed:
                reason = "rng_token_already_consumed"
            elif token.boundary_instance != self._consumed_count[token.site_id]:
                reason = "rng_invocation_consumption_order"
        if reason is not None:
            self._session._log.record(
                broker="rng",
                operation=operation,
                resource=token.site_id,
                disposition="refused",
                reason_code=reason,
            )
            raise BrokerAccessError(
                f"RNG token for {token.site_id!r} refused: {reason}"
            )
        assert site is not None and contract is not None
        self._consumed.add((token.site_id, token.boundary_instance))
        self._consumed_count[token.site_id] += 1
        return site, contract

    @staticmethod
    def _material(token: RNGStreamToken) -> dict[str, object]:
        material = thaw_json(token.semantic_material)
        assert isinstance(material, dict)
        return material

    def _base_seed(
        self,
        site: SeedSiteIR,
        contract: Mapping[str, object],
        material: Mapping[str, object],
    ) -> int:
        source = contract["value_source"]
        default = contract["default"]
        if isinstance(source, str) and source.startswith("run_request."):
            key = source.removeprefix("run_request.")
            value = self._run_inputs.get(key, default)
            return self._uint64(value, location=f"{site.id}/{key}")
        if source == "literal":
            return self._uint64(default, location=f"{site.id}/literal")
        if source == "derived_acs_pattern_seed":
            reference = material.get("derived_from")
            if not isinstance(reference, Mapping):  # validated at session creation
                raise BrokerContractError(
                    f"RNG site {site.id!r} has no derived seed reference"
                )
            source_site = reference.get("site_id")
            boundary_key = reference.get("boundary_key")
            assert isinstance(source_site, str) and isinstance(boundary_key, str)
            key = (source_site, boundary_key)
            try:
                return self._derived_seeds[key]
            except KeyError as error:
                self._session._log.record(
                    broker="rng",
                    operation="resolve_derived_seed",
                    resource=site.id,
                    disposition="refused",
                    reason_code="derived_seed_not_produced",
                    details={
                        "source_site_id": source_site,
                        "source_boundary_key": boundary_key,
                    },
                )
                raise BrokerAccessError(
                    f"RNG site {site.id!r} derived seed has not been produced"
                ) from error
        if source == "stable_string":
            stable_key = material.get("stable_key")
            if not isinstance(stable_key, str) or not stable_key:
                raise BrokerContractError(
                    f"RNG site {site.id!r} requires non-empty stable_key material"
                )
            mask = 2**64 - 1
            hashed = 0
            for byte in stable_key.encode("utf-8"):
                hashed = (hashed * 31 + byte) & mask
            hashed ^= hashed >> 33
            hashed = (hashed * 0xFF51AFD7ED558CCD) & mask
            hashed ^= hashed >> 33
            return hashed % (2**63)
        raise BrokerContractError(
            f"RNG site {site.id!r} has unsupported value source {source!r}"
        )

    def _record_open(
        self,
        *,
        token: RNGStreamToken,
        contract: Mapping[str, object],
        operation: str,
        realized_seed: int | Sequence[int],
    ) -> None:
        self._session._log.record(
            broker="rng",
            operation=operation,
            resource=token.site_id,
            disposition="allowed",
            reason_code="legacy_v1_rng_lease",
            details={
                "owner": token.owner.to_wire(),
                "stream": f"stream:{token.stream}",
                "rng_family": contract["rng_family"],
                "rng_version": contract["rng_version"],
                "kernel": contract["kernel"],
                "seed_material": contract["seed_material"],
                "semantic_material_sha256": token.semantic_material_sha256,
                "realized_seed": _canonical_detail(realized_seed),
                "consumption_order": contract["consumption_order"],
                "reset_boundary": contract["reset_boundary"],
                "boundary_instance": token.boundary_instance,
                "boundary_key": token.boundary_key,
            },
        )

    def _lease(
        self,
        generator: np.random.Generator,
        *,
        token: RNGStreamToken,
        label: str,
    ) -> GeneratorLease:
        handle = self._session._rng_leases.register(generator)
        return GeneratorLease(
            handle,
            session=self._session,
            token=token,
            label=label,
        )

    def generator(
        self,
        token: RNGStreamToken,
    ) -> GeneratorLease:
        """Lease an exact legacy-v1 PCG64 generator."""

        site, contract = self._consume(
            token,
            family="numpy.random.Generator(PCG64)",
            operation="generator",
        )
        supplied = self._material(token)
        seed = self._base_seed(site, contract, supplied)
        derivation = contract["derivation"]
        if derivation == "default_rng(SeedSequence([seed,time_period,374]))":
            period = self._uint64(
                supplied.get("time_period"), location=f"{site.id}/time_period"
            )
            entropy: int | list[int] = [seed, period, 374]
        else:
            entropy = seed
        generator = _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(entropy))
        restored_state = supplied.get("restored_state")
        if restored_state is not None:
            assert isinstance(restored_state, Mapping)
            generator.bit_generator.state = copy.deepcopy(dict(restored_state))
        self._record_open(
            token=token,
            contract=contract,
            operation="generator",
            realized_seed=entropy,
        )
        return self._lease(generator, token=token, label="generator")

    def qrf_generators(
        self,
        token: RNGStreamToken,
    ) -> QRFGeneratorLease:
        """Lease QRF fit child 0 and draw child 1 with exact PCG64 state."""

        site, contract = self._consume(
            token,
            family="microcosm.fit.QRF SeedSequence(PCG64)",
            operation="qrf_generators",
        )
        supplied = self._material(token)
        seed = self._base_seed(site, contract, supplied)
        if "outer_SeedSequence" in str(contract["reset_boundary"]):
            raise BrokerContractError(
                f"RNG site {site.id!r} requires qrf_target_generators"
            )
        fit_child, draw_child = _ORIGINAL_SEED_SEQUENCE(seed).spawn(2)
        fit_generator = _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(fit_child))
        draw_generator = _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(draw_child))
        restored_fit_state = supplied.get("restored_fit_state")
        restored_draw_state = supplied.get("restored_draw_state")
        if restored_fit_state is not None:
            assert isinstance(restored_fit_state, Mapping)
            fit_generator.bit_generator.state = copy.deepcopy(dict(restored_fit_state))
        if restored_draw_state is not None:
            assert isinstance(restored_draw_state, Mapping)
            draw_generator.bit_generator.state = copy.deepcopy(
                dict(restored_draw_state)
            )
        self._record_open(
            token=token,
            contract=contract,
            operation="qrf_generators",
            realized_seed=seed,
        )
        return QRFGeneratorLease(
            fit=self._lease(
                fit_generator,
                token=token,
                label="fit_child_0",
            ),
            draw=self._lease(
                draw_generator,
                token=token,
                label="draw_child_1",
            ),
        )

    def qrf_target_generators(
        self,
        token: RNGStreamToken,
    ) -> tuple[QRFGeneratorLease, ...]:
        """Lease the legacy outer-spawn then inner-QRF generator pairs."""

        site, contract = self._consume(
            token,
            family="microcosm.fit.QRF SeedSequence(PCG64)",
            operation="qrf_target_generators",
        )
        supplied = self._material(token)
        seed = self._base_seed(site, contract, supplied)
        derivation = str(contract["derivation"])
        if "SeedSequence([base_seed,374]).spawn(3)" not in derivation:
            raise BrokerContractError(
                f"RNG site {site.id!r} has no outer target-spawn contract"
            )
        target_count = 3
        outer = _ORIGINAL_SEED_SEQUENCE([seed, 374]).spawn(target_count)
        inner_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0]) for child in outer
        ]
        leases: list[QRFGeneratorLease] = []
        for index, inner_seed in enumerate(inner_seeds):
            fit_child, draw_child = _ORIGINAL_SEED_SEQUENCE(inner_seed).spawn(2)
            leases.append(
                QRFGeneratorLease(
                    fit=self._lease(
                        _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(fit_child)),
                        token=token,
                        label=f"target_{index}_fit_child_0",
                    ),
                    draw=self._lease(
                        _ORIGINAL_GENERATOR(_ORIGINAL_PCG64(draw_child)),
                        token=token,
                        label=f"target_{index}_draw_child_1",
                    ),
                )
            )
        self._record_open(
            token=token,
            contract=contract,
            operation="qrf_target_generators",
            realized_seed=[seed, 374, *inner_seeds],
        )
        return tuple(leases)

    def pandas_sample(
        self,
        token: RNGStreamToken,
        frame: pd.DataFrame,
        *,
        n: int,
    ) -> pd.DataFrame:
        """Run the exact legacy row-sampling algorithm inside the broker."""

        site, contract = self._consume(
            token,
            family="pandas.DataFrame.sample RandomState(MT19937)",
            operation="pandas_sample",
        )
        material = self._material(token)
        cap = material["stage_training_cap"]
        if isinstance(n, bool) or not isinstance(n, int) or n != cap or n > len(frame):
            self._session._log.record(
                broker="rng",
                operation="pandas_sample",
                resource=token.site_id,
                disposition="refused",
                reason_code="training_cap_mismatch",
                details={"requested_n": n, "declared_cap": cap},
            )
            raise BrokerAccessError(
                f"RNG site {token.site_id!r} sample size differs from its plan"
            )
        seed = self._base_seed(site, contract, material)
        random_state = _ORIGINAL_RANDOM_STATE(seed)
        sampled_indices = random_state.choice(
            len(frame), size=n, replace=False, p=None
        ).astype(np.intp, copy=False)
        result = frame.take(sampled_indices, axis=0)
        if _contains_raw_generator(result):
            raise BrokerAccessError("pandas sample result contained RNG authority")
        self._record_open(
            token=token,
            contract=contract,
            operation="pandas_sample",
            realized_seed=seed,
        )
        return result

    def random_forest_classifier_predict(
        self,
        token: RNGStreamToken,
        *,
        train_x: object,
        train_y: object,
        predict_x: object,
        params: Mapping[str, object],
        sample_weight: object | None = None,
    ) -> np.ndarray:
        """Fit and predict with the ledger-owned sklearn MT19937 stream."""

        site, contract = self._consume(
            token,
            family="sklearn RandomForestClassifier check_random_state(MT19937)",
            operation="random_forest_classifier_predict",
        )
        if "random_state" in params:
            raise BrokerContractError(
                "random forest params may not supply private RNG authority"
            )
        if params.get("n_jobs") not in {None, 1}:
            raise BrokerContractError(
                "brokered random forest requires serial execution"
            )
        ensemble_module = sys.modules.get("sklearn.ensemble")
        classifier_type = (
            None
            if ensemble_module is None
            else getattr(ensemble_module, "RandomForestClassifier", None)
        )
        if classifier_type is None:
            raise BrokerContractError(
                "sklearn.ensemble must be loaded before broker activation"
            )
        material = self._material(token)
        seed = self._base_seed(site, contract, material)
        # sklearn resolves integer seeds through the public numpy namespace.
        # Restore only that constructor for the duration of this closed broker
        # operation; no caller callback runs while it is restored.
        with (
            patch.object(np.random, "RandomState", _ORIGINAL_RANDOM_STATE),
            patch.object(os, "urandom", _ORIGINAL_OS_URANDOM),
            patch.object(time_module, "time", _ORIGINAL_TIME),
        ):
            model = classifier_type(**dict(params), random_state=seed)
            model.fit(train_x, train_y, sample_weight=sample_weight)
            result = np.asarray(model.predict(predict_x))
        if _contains_raw_generator(result):  # pragma: no cover - ndarray invariant
            raise BrokerAccessError("random forest result contained RNG authority")
        self._record_open(
            token=token,
            contract=contract,
            operation="random_forest_classifier_predict",
            realized_seed=seed,
        )
        return result.copy()

    def torch_generator(self, token: RNGStreamToken) -> TorchGeneratorLease:
        """Lease the ledger-owned CPU Torch stream without exposing its seed."""

        site, contract = self._consume(
            token,
            family="torch.manual_seed + Tensor.uniform_",
            operation="torch_generator",
        )
        material = self._material(token)
        seed = self._base_seed(site, contract, material)
        handle = self._session._torch_rng_leases.register(seed)
        self._record_open(
            token=token,
            contract=contract,
            operation="torch_generator",
            realized_seed=seed,
        )
        return TorchGeneratorLease(
            handle,
            session=self._session,
            token=token,
        )

    def sha256_derived_seed(
        self,
        token: RNGStreamToken,
    ) -> DerivedSeedHandle:
        """Derive and retain the ledger seed without returning raw seed authority."""

        site, contract = self._consume(
            token, family="SHA-256 derived integer", operation="derived_seed"
        )
        material = self._material(token)
        base_seed = self._base_seed(site, contract, material)
        components: list[str | int] = [base_seed]
        components.extend(
            material[field_name] for field_name in self._sha_dynamic_fields(contract)
        )
        payload = "\0".join(str(component) for component in components).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
        key = (token.site_id, token.boundary_key)
        if key in self._derived_seeds:  # pragma: no cover - token is single use
            raise BrokerContractError("derived seed boundary was already recorded")
        self._derived_seeds[key] = seed
        self._record_open(
            token=token,
            contract=contract,
            operation="derived_seed",
            realized_seed=seed,
        )
        return DerivedSeedHandle(
            protocol_id=self.protocol_id,
            protocol_sha256=self.protocol_sha256,
            owner=self._owner,
            site_id=token.site_id,
            boundary_key=token.boundary_key,
            value_sha256=_sha256_bytes(str(seed).encode()),
            _issuer=self._issuer,
            _activation=token._activation,
        )

    def assert_derived_seed(
        self,
        handle: DerivedSeedHandle,
        claimed_value: int,
    ) -> None:
        """Verify a legacy receipt seed without disclosing retained authority."""

        self._session._require_active()
        reason: str | None = None
        if not isinstance(handle, DerivedSeedHandle) or handle._issuer is not self._issuer:
            reason = "foreign_derived_seed_handle"
        elif handle._activation is not self._session._active_activation:
            reason = "expired_derived_seed_handle"
        elif handle.protocol_id != self.protocol_id or (
            handle.protocol_sha256 != self.protocol_sha256
        ):
            reason = "derived_seed_protocol_mismatch"
        elif handle.owner != self._owner:
            reason = "derived_seed_owner_mismatch"
        elif handle.site_id not in self._sites:
            reason = "undeclared_rng_site"
        elif isinstance(claimed_value, bool) or not isinstance(
            claimed_value, int | np.integer
        ):
            reason = "derived_seed_claim_shape_invalid"
        else:
            key = (handle.site_id, handle.boundary_key)
            retained = self._derived_seeds.get(key)
            if retained is None:
                reason = "derived_seed_not_produced"
            elif handle.value_sha256 != _sha256_bytes(str(retained).encode()):
                reason = "derived_seed_handle_digest_mismatch"
            elif int(claimed_value) != retained:
                reason = "derived_seed_claim_mismatch"
        if reason is not None:
            self._session._log.record(
                broker="rng",
                operation="assert_derived_seed",
                resource=getattr(handle, "site_id", "invalid_handle"),
                disposition="refused",
                reason_code=reason,
            )
            raise BrokerAccessError(f"derived seed assertion refused: {reason}")
        self._session._log.record(
            broker="rng",
            operation="assert_derived_seed",
            resource=handle.site_id,
            disposition="allowed",
            reason_code="derived_seed_claim_verified",
            details={
                "owner": handle.owner.to_wire(),
                "boundary_key": handle.boundary_key,
                "value_sha256": handle.value_sha256,
            },
        )

    def blake2b_uniforms(
        self,
        token: RNGStreamToken,
        *,
        stable_keys: Sequence[object],
    ) -> np.ndarray:
        """Return the ledger's stateless big-endian BLAKE2b uniforms."""

        site, contract = self._consume(
            token,
            family="hashlib.blake2b stateless uniform",
            operation="stateless_uniforms",
        )
        supplied = self._material(token)
        actual_keys_sha256 = sha256_json([str(key) for key in stable_keys])
        if supplied["stable_keys_sha256"] != actual_keys_sha256:
            self._session._log.record(
                broker="rng",
                operation="stateless_uniforms",
                resource=token.site_id,
                disposition="refused",
                reason_code="stable_key_digest_mismatch",
                details={"actual_stable_keys_sha256": actual_keys_sha256},
            )
            raise BrokerAccessError(
                f"RNG site {token.site_id!r} stable-key digest differs from its plan"
            )
        resolved_seed = self._base_seed(site, contract, supplied)
        salt_rows = [
            value.removeprefix("literal_salt=")
            for value in contract["seed_material"]
            if isinstance(value, str) and value.startswith("literal_salt=")
        ]
        if len(salt_rows) != 1:
            raise BrokerContractError(
                f"RNG site {site.id!r} requires one ledger-owned literal salt"
            )
        salt_template = salt_rows[0]
        fields = self._salt_fields(contract)
        salt_material = {field: supplied[field] for field in fields}
        try:
            salt = salt_template.format(**salt_material)
        except (KeyError, ValueError) as error:
            raise BrokerContractError(
                f"RNG site {site.id!r} cannot resolve its literal salt"
            ) from error
        values = np.asarray(
            [
                int.from_bytes(
                    hashlib.blake2b(
                        f"{resolved_seed}:{salt}:{key}".encode(), digest_size=8
                    ).digest(),
                    "big",
                )
                / 2**64
                for key in stable_keys
            ],
            dtype=np.float64,
        )
        self._record_open(
            token=token,
            contract=contract,
            operation="stateless_uniforms",
            realized_seed=resolved_seed,
        )
        return values

    def pandas_hash_uniforms(
        self,
        token: RNGStreamToken,
        *,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Return the exact stateless pandas-hash lottery uniforms."""

        _site, contract = self._consume(
            token,
            family="pandas.util.hash_pandas_object stateless uint64",
            operation="pandas_hash_uniforms",
        )
        frame_sha256 = _sha256_bytes(frame.to_csv(index=False).encode())
        material = self._material(token)
        if material["frame_sha256"] != frame_sha256:
            self._session._log.record(
                broker="rng",
                operation="pandas_hash_uniforms",
                resource=token.site_id,
                disposition="refused",
                reason_code="frame_digest_mismatch",
                details={"actual_frame_sha256": frame_sha256},
            )
            raise BrokerAccessError(
                f"RNG site {token.site_id!r} frame digest differs from its plan"
            )
        hashed = _ORIGINAL_HASH_PANDAS_OBJECT(frame, index=False).to_numpy(
            dtype=np.uint64
        )
        values = (hashed.astype(np.float64) + 0.5) / float(2**64)
        values = np.clip(values, 1e-12, 1.0 - 1e-12)
        self._record_open(
            token=token,
            contract=contract,
            operation="pandas_hash_uniforms",
            realized_seed=0,
        )
        return values


class FileReadLease:
    """Session-bound file proxy; outstanding handles are closed at session seal."""

    __slots__ = (
        "_broker",
        "_closed",
        "__descriptor",
        "__opened_stat",
        "__source",
        "__stream",
    )

    def __init__(
        self,
        stream: IO[bytes] | IO[str],
        *,
        broker: FileBroker,
        descriptor: int | None = None,
        opened_stat: os.stat_result | None = None,
        source: DeclaredSource | None = None,
    ) -> None:
        object.__setattr__(self, "_FileReadLease__stream", stream)
        object.__setattr__(self, "_FileReadLease__descriptor", descriptor)
        object.__setattr__(self, "_FileReadLease__opened_stat", opened_stat)
        object.__setattr__(self, "_FileReadLease__source", source)
        object.__setattr__(self, "_broker", broker)
        object.__setattr__(self, "_closed", False)

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            broker = object.__getattribute__(self, "_broker")
            broker._session._log.record(
                broker="ambient",
                operation="file_lease_private_access",
                resource=name,
                disposition="refused",
                reason_code="file_handle_escape_prohibited",
            )
            raise BrokerAccessError("file lease internals are not accessible")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, _value: object) -> None:
        raise BrokerAccessError(f"file lease field {name!r} is immutable")

    def _verify(self) -> None:
        broker = object.__getattribute__(self, "_broker")
        stream = object.__getattribute__(self, "_FileReadLease__stream")
        closed = object.__getattribute__(self, "_closed")
        if closed or stream.closed or broker._session.sealed:
            raise BrokerAccessError("file lease is closed")
        broker._session._require_active()

    @property
    def closed(self) -> bool:
        stream = object.__getattribute__(self, "_FileReadLease__stream")
        return object.__getattribute__(self, "_closed") or stream.closed

    def __enter__(self) -> FileReadLease:
        object.__getattribute__(self, "_verify")()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        stream = object.__getattribute__(self, "_FileReadLease__stream")
        broker = object.__getattribute__(self, "_broker")
        if object.__getattribute__(self, "_closed"):
            broker._leases.discard(self)
            return
        failure: BrokerAccessError | None = None
        source = object.__getattribute__(self, "_FileReadLease__source")
        descriptor = object.__getattribute__(self, "_FileReadLease__descriptor")
        opened_stat = object.__getattribute__(self, "_FileReadLease__opened_stat")
        if source is not None:
            assert descriptor is not None
            assert opened_stat is not None
            try:
                broker._verify_snapshot_at_close(
                    source,
                    stream=stream,
                    descriptor=descriptor,
                    opened_stat=opened_stat,
                )
            except BrokerAccessError as error:
                failure = error
        try:
            stream.close()
        finally:
            object.__setattr__(self, "_closed", True)
            broker._leases.discard(self)
        if failure is not None:
            raise failure

    def read(self, size: int = -1) -> bytes | str:
        object.__getattribute__(self, "_verify")()
        return object.__getattribute__(self, "_FileReadLease__stream").read(size)

    def readinto(self, buffer: object) -> int | None:
        object.__getattribute__(self, "_verify")()
        stream = object.__getattribute__(self, "_FileReadLease__stream")
        readinto = getattr(stream, "readinto", None)
        if readinto is None:  # pragma: no cover - binary snapshots always support it
            raise BrokerAccessError("file lease does not support readinto")
        return readinto(buffer)

    def readline(self, size: int = -1) -> bytes | str:
        object.__getattribute__(self, "_verify")()
        return object.__getattribute__(self, "_FileReadLease__stream").readline(size)

    def readlines(self, hint: int = -1) -> list[bytes] | list[str]:
        object.__getattribute__(self, "_verify")()
        return object.__getattribute__(self, "_FileReadLease__stream").readlines(hint)

    def seek(self, offset: int, whence: int = 0) -> int:
        object.__getattribute__(self, "_verify")()
        return object.__getattribute__(self, "_FileReadLease__stream").seek(
            offset, whence
        )

    def tell(self) -> int:
        object.__getattribute__(self, "_verify")()
        return object.__getattribute__(self, "_FileReadLease__stream").tell()

    def readable(self) -> bool:
        object.__getattribute__(self, "_verify")()
        return bool(object.__getattribute__(self, "_FileReadLease__stream").readable())

    def seekable(self) -> bool:
        object.__getattribute__(self, "_verify")()
        return bool(object.__getattribute__(self, "_FileReadLease__stream").seekable())

    def __iter__(self) -> FileReadLease:
        object.__getattribute__(self, "_verify")()
        return self

    def __next__(self) -> bytes | str:
        object.__getattribute__(self, "_verify")()
        return next(object.__getattribute__(self, "_FileReadLease__stream"))


class FileBroker:
    """Read-only access to preverified logical source handles."""

    def __init__(
        self, *, session: BrokerSession, sources: Sequence[DeclaredSource]
    ) -> None:
        self._session = session
        normalized_sources = {source.id: source for source in sources}
        if len(normalized_sources) != len(sources):
            raise BrokerContractError("declared source ids must be unique")
        paths = [source.resolved_path for source in sources]
        if len(paths) != len(set(paths)):
            raise BrokerContractError("declared source paths must be unique")
        self._sources = MappingProxyType(normalized_sources)
        self._leases: set[FileReadLease] = set()
        rows: list[FrozenMap] = []
        for source in sorted(sources, key=lambda item: item.id):
            row = freeze_json(
                {
                    "source_id": source.id,
                    "content_sha256": source.sha256,
                    "byte_size": source.byte_size,
                }
            )
            assert isinstance(row, FrozenMap)
            rows.append(row)
        self._behavior_identity = SourceBehaviorIdentity(
            owner=session.owner,
            sources=tuple(rows),
            _issuer=_SOURCE_BEHAVIOR_ISSUER,
            _session_issuer=session._behavior_issuer,
        )

    @property
    def behavior_identity(self) -> SourceBehaviorIdentity:
        return self._behavior_identity

    def _lease(
        self,
        stream: IO[bytes] | IO[str],
        *,
        descriptor: int | None = None,
        opened_stat: os.stat_result | None = None,
        source: DeclaredSource | None = None,
    ) -> FileReadLease:
        lease = FileReadLease(
            stream,
            broker=self,
            descriptor=descriptor,
            opened_stat=opened_stat,
            source=source,
        )
        self._leases.add(lease)
        return lease

    def close_all(self) -> None:
        first_error: BrokerAccessError | None = None
        for lease in tuple(self._leases):
            try:
                lease.close()
            except BrokerAccessError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _source(self, source_id: str) -> DeclaredSource:
        self._session._require_active()
        if "declared_source_read" not in self._session.effects:
            reason = "source_read_effect_not_declared"
        elif source_id not in self._sources:
            reason = "undeclared_source"
        else:
            return self._sources[source_id]
        self._session._log.record(
            broker="file",
            operation="open_read",
            resource=source_id,
            disposition="refused",
            reason_code=reason,
        )
        raise BrokerAccessError(f"source {source_id!r} refused: {reason}")

    @contextmanager
    def _verified_stream(
        self, source_id: str, *, operation: str
    ) -> Iterator[tuple[DeclaredSource, IO[bytes]]]:
        source = self._source(source_id)
        stream = _ORIGINAL_BUILTINS_OPEN(source.resolved_path, "rb")
        digest = hashlib.sha256()
        actual_size = 0
        try:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                actual_size += len(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_size != source.byte_size or actual_sha256 != source.sha256:
                self._session._log.record(
                    broker="file",
                    operation=operation,
                    resource=source_id,
                    disposition="refused",
                    reason_code="source_identity_mismatch",
                    details={
                        "actual_sha256": actual_sha256,
                        "actual_size": actual_size,
                    },
                )
                raise BrokerAccessError(
                    f"declared source {source_id!r} differs from its verified identity"
                )
            stream.seek(0)
            self._session._log.record(
                broker="file",
                operation=operation,
                resource=source_id,
                disposition="allowed",
                reason_code="declared_source_read",
                details={
                    "resolved_path": str(source.resolved_path),
                    "content_sha256": source.sha256,
                    "byte_size": source.byte_size,
                },
            )
            yield source, stream
        finally:
            stream.close()

    def read_bytes(self, source_id: str) -> bytes:
        with self._verified_stream(source_id, operation="read_bytes") as (
            _source,
            stream,
        ):
            return stream.read()

    def read_text(self, source_id: str, *, encoding: str = "utf-8") -> str:
        payload = self.read_bytes(source_id)
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeError) as error:
            self._session._log.record(
                broker="file",
                operation="read_text",
                resource=source_id,
                disposition="refused",
                reason_code="source_text_decode_failed",
                details={"encoding": encoding},
            )
            raise BrokerAccessError(
                f"declared source {source_id!r} is not valid {encoding}"
            ) from error

    @staticmethod
    def _snapshot_stat_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(stat.S_IFMT(value.st_mode)),
            int(value.st_size),
        )

    @staticmethod
    def _snapshot_digest(stream: IO[bytes] | IO[str]) -> tuple[str, int]:
        stream.seek(0)
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(1024 * 1024):
            if not isinstance(chunk, bytes):  # pragma: no cover - binary invariant
                raise BrokerContractError("file snapshot stream must be binary")
            digest.update(chunk)
            size += len(chunk)
        stream.seek(0)
        return digest.hexdigest(), size

    def _verify_snapshot_at_close(
        self,
        source: DeclaredSource,
        *,
        stream: IO[bytes] | IO[str],
        descriptor: int,
        opened_stat: os.stat_result,
    ) -> None:
        try:
            before = _ORIGINAL_OS_FSTAT(descriptor)
            actual_sha256, actual_size = self._snapshot_digest(stream)
            after = _ORIGINAL_OS_FSTAT(descriptor)
        except (OSError, ValueError) as error:
            self._session._log.record(
                broker="file",
                operation="close_snapshot",
                resource=source.id,
                disposition="refused",
                reason_code="source_snapshot_verification_failed",
            )
            raise BrokerAccessError(
                f"declared source {source.id!r} could not be reverified at close"
            ) from error
        stable = (
            self._snapshot_stat_identity(before)
            == self._snapshot_stat_identity(opened_stat)
            == self._snapshot_stat_identity(after)
        )
        if (
            not stable
            or actual_size != source.byte_size
            or actual_sha256 != source.sha256
        ):
            self._session._log.record(
                broker="file",
                operation="close_snapshot",
                resource=source.id,
                disposition="refused",
                reason_code="source_snapshot_drift",
                details={
                    "actual_sha256": actual_sha256,
                    "actual_size": actual_size,
                    "descriptor_stable": stable,
                },
            )
            raise BrokerAccessError(
                f"declared source {source.id!r} changed while its snapshot was open"
            )

    @contextmanager
    def open_snapshot(self, source_id: str) -> Iterator[FileReadLease]:
        """Open one authenticated, descriptor-stable binary source snapshot.

        The path is resolved only for the initial ``O_NOFOLLOW`` open.  All
        authentication and parser reads use that retained regular-file
        descriptor, so replacing the directory entry cannot redirect a live
        lease.  Closing re-authenticates the same descriptor and refuses
        in-place drift.
        """

        source = self._source(source_id)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:  # pragma: no cover - supported build platforms
            raise BrokerContractError("file snapshots require O_NOFOLLOW support")
        descriptor = _ORIGINAL_OS_OPEN(source.resolved_path, flags | nofollow)
        stream: IO[bytes] | None = None
        lease: FileReadLease | None = None
        try:
            opened_stat = _ORIGINAL_OS_FSTAT(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                self._session._log.record(
                    broker="file",
                    operation="open_snapshot",
                    resource=source_id,
                    disposition="refused",
                    reason_code="source_not_regular_file",
                )
                raise BrokerAccessError(
                    f"declared source {source_id!r} is not a regular file"
                )
            stream = io.FileIO(descriptor, mode="rb", closefd=True)
            descriptor_owned_by_stream = descriptor
            descriptor = -1
            actual_sha256, actual_size = self._snapshot_digest(stream)
            authenticated_stat = _ORIGINAL_OS_FSTAT(descriptor_owned_by_stream)
            stable = self._snapshot_stat_identity(
                opened_stat
            ) == self._snapshot_stat_identity(authenticated_stat)
            if (
                not stable
                or actual_size != source.byte_size
                or actual_sha256 != source.sha256
            ):
                self._session._log.record(
                    broker="file",
                    operation="open_snapshot",
                    resource=source_id,
                    disposition="refused",
                    reason_code="source_identity_mismatch",
                    details={
                        "actual_sha256": actual_sha256,
                        "actual_size": actual_size,
                        "descriptor_stable": stable,
                    },
                )
                raise BrokerAccessError(
                    f"declared source {source_id!r} differs from its verified identity"
                )
            self._session._log.record(
                broker="file",
                operation="open_snapshot",
                resource=source_id,
                disposition="allowed",
                reason_code="declared_source_read",
                details={
                    "resolved_path": str(source.resolved_path),
                    "content_sha256": source.sha256,
                    "byte_size": source.byte_size,
                },
            )
            lease = self._lease(
                stream,
                descriptor=descriptor_owned_by_stream,
                opened_stat=authenticated_stat,
                source=source,
            )
            try:
                yield lease
            except BaseException:
                try:
                    lease.close()
                except BrokerAccessError:
                    pass
                raise
            else:
                lease.close()
        finally:
            if lease is None and stream is not None:
                stream.close()
            elif descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def open_read(
        self, source_id: str, *, binary: bool = True, encoding: str = "utf-8"
    ) -> Iterator[FileReadLease]:
        with self._verified_stream(source_id, operation="open_read") as (
            _source,
            raw_stream,
        ):
            if binary:
                lease = self._lease(raw_stream)
                try:
                    yield lease
                finally:
                    lease.close()
                return
            try:
                decoder = codecs.getincrementaldecoder(encoding)()
                while chunk := raw_stream.read(1024 * 1024):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            except (LookupError, UnicodeError) as error:
                self._session._log.record(
                    broker="file",
                    operation="open_read",
                    resource=source_id,
                    disposition="refused",
                    reason_code="source_text_decode_failed",
                    details={"encoding": encoding},
                )
                raise BrokerAccessError(
                    f"declared source {source_id!r} is not valid {encoding}"
                ) from error
            raw_stream.seek(0)
            text_stream = io.TextIOWrapper(raw_stream, encoding=encoding)
            lease = self._lease(text_stream)
            try:
                yield lease
            finally:
                lease.close()


class EnvironmentBroker:
    """Explicit environment snapshot; unavailable without source-read effects."""

    def __init__(
        self, *, session: BrokerSession, values: Mapping[str, str | None]
    ) -> None:
        self._session = session
        self._values = MappingProxyType(dict(values))
        if any(not isinstance(key, str) or not key for key in self._values):
            raise BrokerContractError("environment grants require non-empty names")
        if any(
            value is not None and not isinstance(value, str)
            for value in self._values.values()
        ):
            raise BrokerContractError(
                "environment grant values must be strings or null"
            )

    def get(self, name: str) -> str | None:
        self._session._require_active()
        if self._session.determinism in {"deterministic", "seeded"}:
            reason = "environment_forbidden_for_reproducible_kernel"
        elif "declared_source_read" not in self._session.effects:
            reason = "environment_effect_not_declared"
        elif name not in self._values:
            reason = "undeclared_environment_name"
        else:
            value = self._values[name]
            self._session._log.record(
                broker="environment",
                operation="get",
                resource=name,
                disposition="allowed",
                reason_code="declared_environment_read",
                details={
                    "present": value is not None,
                    "value_sha256": None
                    if value is None
                    else _sha256_bytes(value.encode()),
                },
            )
            return value
        self._session._log.record(
            broker="environment",
            operation="get",
            resource=name,
            disposition="refused",
            reason_code=reason,
        )
        raise BrokerAccessError(f"environment name {name!r} refused: {reason}")


class ClockBroker:
    """Explicit operational clock values; never an ambient numerical input."""

    def __init__(self, *, session: BrokerSession, values: Mapping[str, float]) -> None:
        self._session = session
        normalized: dict[str, float] = {}
        for name, value in values.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
            ):
                raise BrokerContractError("clock grants require finite named values")
            normalized[name] = float(value)
        self._values = MappingProxyType(normalized)

    def read(self, name: str) -> float:
        self._session._require_active()
        if self._session.determinism in {"deterministic", "seeded"}:
            reason = "clock_forbidden_for_reproducible_kernel"
        elif self._session.require_byte_equivalence:
            reason = "clock_forbidden_in_byte_equivalence_mode"
        elif name not in self._values:
            reason = "undeclared_clock"
        else:
            value = self._values[name]
            self._session._log.record(
                broker="clock",
                operation="read",
                resource=name,
                disposition="allowed",
                reason_code="declared_operational_clock",
                details={"value": value},
            )
            return value
        self._session._log.record(
            broker="clock",
            operation="read",
            resource=name,
            disposition="refused",
            reason_code=reason,
        )
        raise BrokerAccessError(f"clock {name!r} refused: {reason}")


class _KernelBrokerView:
    """Public-method-only projection of a broker facade passed to kernels."""

    __slots__ = ("__session",)

    def __init__(self, session: BrokerSession) -> None:
        object.__setattr__(self, "_KernelBrokerView__session", session)

    def __getattribute__(self, name: str) -> object:
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            session = object.__getattribute__(self, "_KernelBrokerView__session")
            session._log.record(
                broker="ambient",
                operation="kernel_broker_private_access",
                resource=name,
                disposition="refused",
                reason_code="broker_authority_internals_prohibited",
            )
            raise BrokerAccessError("kernel broker internals are not accessible")
        return object.__getattribute__(self, name)


class KernelRNGBroker(_KernelBrokerView):
    """Kernel-visible RNG operations without access to broker authority state."""

    __slots__ = ("__brokers_by_site", "__primary")

    def __init__(self, brokers: Sequence[RNGBroker]) -> None:
        if not brokers:
            raise BrokerContractError("kernel RNG view requires a primary broker")
        primary = brokers[0]
        _KernelBrokerView.__init__(self, primary._session)
        by_site: dict[str, RNGBroker] = {}
        for broker in brokers:
            if broker._session is not primary._session:
                raise BrokerContractError(
                    "kernel RNG brokers must share one broker session"
                )
            for site_id in broker.granted_sites:
                if site_id in by_site:
                    raise BrokerContractError(
                        f"kernel RNG route for site {site_id!r} is ambiguous"
                    )
                by_site[site_id] = broker
        object.__setattr__(
            self,
            "_KernelRNGBroker__brokers_by_site",
            MappingProxyType(by_site),
        )
        object.__setattr__(self, "_KernelRNGBroker__primary", primary)

    def _call(
        self,
        name: str,
        site_id: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        brokers = object.__getattribute__(
            self, "_KernelRNGBroker__brokers_by_site"
        )
        broker = brokers.get(site_id)
        if broker is None:
            broker = object.__getattribute__(self, "_KernelRNGBroker__primary")
        return getattr(broker, name)(*args, **kwargs)

    def token(
        self, site_id: str, boundary_key: str = _DEFAULT_RNG_BOUNDARY_KEY
    ) -> RNGStreamToken:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "token", site_id, site_id, boundary_key
        )

    def generator(self, token: RNGStreamToken) -> GeneratorLease:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "generator", token.site_id, token
        )

    def qrf_generators(self, token: RNGStreamToken) -> QRFGeneratorLease:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "qrf_generators", token.site_id, token
        )

    def qrf_target_generators(
        self, token: RNGStreamToken
    ) -> tuple[QRFGeneratorLease, ...]:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "qrf_target_generators", token.site_id, token
        )

    def sha256_derived_seed(self, token: RNGStreamToken) -> DerivedSeedHandle:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "sha256_derived_seed", token.site_id, token
        )

    def assert_derived_seed(
        self,
        handle: DerivedSeedHandle,
        claimed_value: int,
    ) -> None:
        call = object.__getattribute__(self, "_call")
        call("assert_derived_seed", handle.site_id, handle, claimed_value)

    def blake2b_uniforms(
        self, token: RNGStreamToken, *, stable_keys: Sequence[object]
    ) -> np.ndarray:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "blake2b_uniforms", token.site_id, token, stable_keys=stable_keys
        )

    def pandas_hash_uniforms(
        self, token: RNGStreamToken, *, frame: pd.DataFrame
    ) -> np.ndarray:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "pandas_hash_uniforms", token.site_id, token, frame=frame
        )

    def pandas_sample(
        self, token: RNGStreamToken, frame: pd.DataFrame, *, n: int
    ) -> pd.DataFrame:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "pandas_sample", token.site_id, token, frame, n=n
        )

    def random_forest_classifier_predict(
        self,
        token: RNGStreamToken,
        *,
        train_x: object,
        train_y: object,
        predict_x: object,
        params: Mapping[str, object],
        sample_weight: object | None = None,
    ) -> np.ndarray:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "random_forest_classifier_predict",
            token.site_id,
            token,
            train_x=train_x,
            train_y=train_y,
            predict_x=predict_x,
            params=params,
            sample_weight=sample_weight,
        )

    def torch_generator(self, token: RNGStreamToken) -> TorchGeneratorLease:
        call = object.__getattribute__(self, "_call")
        return call(  # type: ignore[return-value]
            "torch_generator", token.site_id, token
        )


class KernelFileBroker(_KernelBrokerView):
    """Kernel-visible declared-source operations."""

    __slots__ = ("__broker",)

    def __init__(self, broker: FileBroker) -> None:
        _KernelBrokerView.__init__(self, broker._session)
        object.__setattr__(self, "_KernelFileBroker__broker", broker)

    def read_bytes(self, source_id: str) -> bytes:
        broker = object.__getattribute__(self, "_KernelFileBroker__broker")
        return broker.read_bytes(source_id)

    def read_text(self, source_id: str, *, encoding: str = "utf-8") -> str:
        broker = object.__getattribute__(self, "_KernelFileBroker__broker")
        return broker.read_text(source_id, encoding=encoding)

    @contextmanager
    def open_read(
        self, source_id: str, *, binary: bool = True, encoding: str = "utf-8"
    ) -> Iterator[FileReadLease]:
        broker = object.__getattribute__(self, "_KernelFileBroker__broker")
        with broker.open_read(source_id, binary=binary, encoding=encoding) as stream:
            yield stream

    @contextmanager
    def open_snapshot(self, source_id: str) -> Iterator[FileReadLease]:
        broker = object.__getattribute__(self, "_KernelFileBroker__broker")
        with broker.open_snapshot(source_id) as stream:
            yield stream


class KernelEnvironmentBroker(_KernelBrokerView):
    __slots__ = ("__broker",)

    def __init__(self, broker: EnvironmentBroker) -> None:
        _KernelBrokerView.__init__(self, broker._session)
        object.__setattr__(self, "_KernelEnvironmentBroker__broker", broker)

    def get(self, name: str) -> str | None:
        broker = object.__getattribute__(self, "_KernelEnvironmentBroker__broker")
        return broker.get(name)


class KernelClockBroker(_KernelBrokerView):
    __slots__ = ("__broker",)

    def __init__(self, broker: ClockBroker) -> None:
        _KernelBrokerView.__init__(self, broker._session)
        object.__setattr__(self, "_KernelClockBroker__broker", broker)

    def read(self, name: str) -> float:
        broker = object.__getattribute__(self, "_KernelClockBroker__broker")
        return broker.read(name)


class KernelBrokerSession(_KernelBrokerView):
    """The capability projection kernels receive instead of the owner session."""

    __slots__ = ("__clock", "__environment", "__files", "__rng")

    def __init__(
        self,
        *,
        rng: RNGBroker,
        supplemental_rngs: Sequence[RNGBroker] = (),
        files: FileBroker,
        environment: EnvironmentBroker,
        clock: ClockBroker,
    ) -> None:
        _KernelBrokerView.__init__(self, rng._session)
        object.__setattr__(
            self,
            "_KernelBrokerSession__rng",
            KernelRNGBroker((rng, *supplemental_rngs)),
        )
        object.__setattr__(self, "_KernelBrokerSession__files", KernelFileBroker(files))
        object.__setattr__(
            self,
            "_KernelBrokerSession__environment",
            KernelEnvironmentBroker(environment),
        )
        object.__setattr__(
            self, "_KernelBrokerSession__clock", KernelClockBroker(clock)
        )

    @property
    def rng(self) -> KernelRNGBroker:
        return object.__getattribute__(self, "_KernelBrokerSession__rng")

    @property
    def files(self) -> KernelFileBroker:
        return object.__getattribute__(self, "_KernelBrokerSession__files")

    @property
    def environment(self) -> KernelEnvironmentBroker:
        return object.__getattribute__(self, "_KernelBrokerSession__environment")

    @property
    def clock(self) -> KernelClockBroker:
        return object.__getattribute__(self, "_KernelBrokerSession__clock")

    def run_physical_operation(self, *, input_binding_sha256: str) -> object:
        """Request the session's prebound physical operation.

        No callable or authority is accepted from kernel code.  The owner
        session verifies the input binding and performs the call itself.
        """

        session = object.__getattribute__(self, "_KernelBrokerView__session")
        return session._run_physical_operation(
            input_binding_sha256=input_binding_sha256
        )


@dataclass(frozen=True, slots=True)
class _BrokerAuthority:
    owner: BrokerOwner
    determinism: str
    effects: frozenset[str]
    protocol_id: str
    protocol_sha256: str
    node_key: str | None
    attempt: int
    attempt_scope: str | None
    require_byte_equivalence: bool
    run_provenance_identity: FrozenMap


class BrokerSession:
    """Aggregate broker facade bound to one owner and one attempt."""

    _PUBLIC_AUTHORITY_FIELDS = frozenset(
        {
            "attempt",
            "attempt_scope",
            "clock",
            "determinism",
            "effects",
            "environment",
            "files",
            "node_key",
            "owner",
            "require_byte_equivalence",
            "rng",
            "run_provenance_identity",
        }
    )
    __slots__ = (
        "_active_activation",
        "_authority",
        "_behavior_issuer",
        "_clock",
        "_environment",
        "_files",
        "_kernel_view",
        "_log",
        "_physical_operation",
        "_physical_operation_invoked",
        "_rng",
        "_rng_leases",
        "_sealed_receipt",
        "_supplemental_rngs",
        "_torch_rng_leases",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._PUBLIC_AUTHORITY_FIELDS or (
            name in self.__slots__ and hasattr(self, name)
        ):
            raise BrokerContractError(
                f"broker session authority field {name!r} is immutable"
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        owner: BrokerOwner,
        determinism: str,
        effects: Sequence[str],
        protocol_id: str,
        protocol_sha256: str,
        seed_sites: Sequence[SeedSiteIR] = (),
        run_inputs: Mapping[str, int] | None = None,
        rng_invocation_plan: Mapping[str, Sequence[RNGInvocation]] | None = None,
        seed_stream_map: SeedStreamMap | None = None,
        supplemental_seed_owners: Sequence[SeedOwnerIR] = (),
        rng_invocation_plans_by_owner: Mapping[
            tuple[str, str], Mapping[str, Sequence[RNGInvocation]]
        ]
        | None = None,
        sources: Sequence[DeclaredSource] = (),
        environment: Mapping[str, str | None] | None = None,
        clocks: Mapping[str, float] | None = None,
        run_provenance_identity: Mapping[str, object],
        node_key: str | None = None,
        attempt: int = 0,
        attempt_scope: str | None = None,
        require_byte_equivalence: bool = True,
    ) -> None:
        if determinism not in _DETERMINISM:
            raise BrokerContractError(f"unknown determinism {determinism!r}")
        normalized_effects = frozenset(effects)
        if not normalized_effects or not normalized_effects <= _EFFECTS:
            raise BrokerContractError("broker effects are empty or unknown")
        if "none" in normalized_effects and normalized_effects != {"none"}:
            raise BrokerContractError("broker effect 'none' is exclusive")
        if determinism == "seeded" and not seed_sites:
            raise BrokerContractError("seeded broker session has no RNG site grants")
        if determinism == "deterministic" and seed_sites:
            raise BrokerContractError(
                "deterministic broker session unexpectedly has RNG site grants"
            )
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise BrokerContractError("broker attempt must be non-negative")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise BrokerContractError("broker protocol id must be non-empty")
        if (
            not isinstance(protocol_sha256, str)
            or len(protocol_sha256) != 64
            or any(character not in "0123456789abcdef" for character in protocol_sha256)
        ):
            raise BrokerContractError("broker protocol digest must be sha256")
        if node_key is not None and (
            not isinstance(node_key, str)
            or len(node_key) != 64
            or any(character not in "0123456789abcdef" for character in node_key)
        ):
            raise BrokerContractError("broker node key must be sha256 or null")
        if attempt_scope is not None and (
            not isinstance(attempt_scope, str) or not attempt_scope
        ):
            raise BrokerContractError("broker attempt scope must be non-empty or null")
        if not isinstance(require_byte_equivalence, bool):
            raise BrokerContractError("byte-equivalence policy must be boolean")
        self._authority = _BrokerAuthority(
            owner=owner,
            determinism=determinism,
            effects=normalized_effects,
            protocol_id=protocol_id,
            protocol_sha256=protocol_sha256,
            node_key=node_key,
            attempt=attempt,
            attempt_scope=attempt_scope,
            require_byte_equivalence=require_byte_equivalence,
            run_provenance_identity=_run_provenance_identity(run_provenance_identity),
        )
        self._behavior_issuer = object()
        self._log = _AccessLog()
        self._active_activation: object | None = None
        self._sealed_receipt: BrokerReceipt | None = None
        self._physical_operation: PhysicalOperation | None = None
        self._physical_operation_invoked = False
        self._rng_leases = _GeneratorLeaseStore(self)
        self._torch_rng_leases = _TorchGeneratorLeaseStore(self)
        supplemental_grants = tuple(supplemental_seed_owners)
        if any(not isinstance(grant, SeedOwnerIR) for grant in supplemental_grants):
            raise BrokerContractError(
                "supplemental seed grants must be compiled SeedOwnerIR values"
            )
        if supplemental_grants and seed_stream_map is None:
            raise BrokerContractError(
                "supplemental seed grants require their compiled SeedStreamMap"
            )
        if seed_stream_map is not None and (
            seed_stream_map.protocol_id != protocol_id
            or seed_stream_map.implementation_sha256 != protocol_sha256
        ):
            raise BrokerContractError(
                "supplemental seed map differs from the session RNG protocol"
            )
        owner_sites: dict[tuple[str, str], tuple[SeedSiteIR, ...]] = {
            (owner.kind, owner.id): tuple(seed_sites)
        }
        supplemental_owners: list[BrokerOwner] = []
        for grant in supplemental_grants:
            supplemental_owner = BrokerOwner(grant.kind, grant.id)
            owner_key = (supplemental_owner.kind, supplemental_owner.id)
            if owner_key in owner_sites:
                raise BrokerContractError(
                    f"supplemental seed owner {owner_key!r} repeats the primary "
                    "or another supplemental owner"
                )
            assert seed_stream_map is not None
            if seed_stream_map.owner(*owner_key) != grant:
                raise BrokerContractError(
                    f"supplemental seed owner {owner_key!r} is not the compiled "
                    "SeedStreamMap grant"
                )
            site_ids = set(grant.sites)
            sites = tuple(
                site for site in seed_stream_map.sites if site.id in site_ids
            )
            if tuple(site.id for site in sites) != grant.sites:
                raise BrokerContractError(
                    f"supplemental seed owner {owner_key!r} site order differs "
                    "from the compiled map"
                )
            if tuple(dict.fromkeys(site.stream for site in sites)) != grant.streams:
                raise BrokerContractError(
                    f"supplemental seed owner {owner_key!r} streams differ from "
                    "the compiled map"
                )
            owner_sites[owner_key] = sites
            supplemental_owners.append(supplemental_owner)

        if rng_invocation_plans_by_owner is not None and (
            rng_invocation_plan is not None
        ):
            raise BrokerContractError(
                "owner-scoped RNG plans cannot be combined with the primary-plan "
                "shorthand"
            )
        normalized_owner_plans: dict[
            tuple[str, str], Mapping[str, Sequence[RNGInvocation]]
        ] = {}
        if rng_invocation_plans_by_owner is not None:
            for raw_owner_key, plan in rng_invocation_plans_by_owner.items():
                if (
                    not isinstance(raw_owner_key, tuple)
                    or len(raw_owner_key) != 2
                    or any(
                        not isinstance(component, str) or not component
                        for component in raw_owner_key
                    )
                ):
                    raise BrokerContractError(
                        "owner-scoped RNG plan keys must be (kind, id) strings"
                    )
                normalized_key = (
                    BrokerOwner(raw_owner_key[0], raw_owner_key[1]).kind,
                    raw_owner_key[1],
                )
                if not isinstance(plan, Mapping):
                    raise BrokerContractError(
                        f"RNG plan for owner {normalized_key!r} must be a mapping"
                    )
                normalized_owner_plans[normalized_key] = plan
            missing_owners = sorted(set(owner_sites) - set(normalized_owner_plans))
            extra_owners = sorted(set(normalized_owner_plans) - set(owner_sites))
            if missing_owners or extra_owners:
                raise BrokerContractError(
                    "owner-scoped RNG plans differ from the granted owners; "
                    f"missing={missing_owners!r}, extra={extra_owners!r}"
                )
            for owner_key, sites in owner_sites.items():
                planned_sites = set(normalized_owner_plans[owner_key])
                granted_sites = {site.id for site in sites}
                if planned_sites != granted_sites:
                    raise BrokerContractError(
                        f"RNG plan for owner {owner_key!r} differs from its granted "
                        f"sites; missing={sorted(granted_sites - planned_sites)!r}, "
                        f"extra={sorted(planned_sites - granted_sites)!r}"
                    )
        elif supplemental_grants:
            raise BrokerContractError(
                "supplemental seed grants require exact owner-scoped RNG plans"
            )

        primary_owner_key = (owner.kind, owner.id)
        primary_plan = (
            normalized_owner_plans[primary_owner_key]
            if rng_invocation_plans_by_owner is not None
            else ({} if rng_invocation_plan is None else rng_invocation_plan)
        )
        self._rng = RNGBroker(
            session=self,
            owner=owner,
            protocol_id=protocol_id,
            protocol_sha256=protocol_sha256,
            sites=tuple(seed_sites),
            run_inputs={} if run_inputs is None else run_inputs,
            invocation_plan=primary_plan,
        )
        self._supplemental_rngs = tuple(
            RNGBroker(
                session=self,
                owner=supplemental_owner,
                protocol_id=protocol_id,
                protocol_sha256=protocol_sha256,
                sites=owner_sites[
                    (supplemental_owner.kind, supplemental_owner.id)
                ],
                run_inputs={} if run_inputs is None else run_inputs,
                invocation_plan=normalized_owner_plans[
                    (supplemental_owner.kind, supplemental_owner.id)
                ],
            )
            for supplemental_owner in supplemental_owners
        )
        self._files = FileBroker(session=self, sources=tuple(sources))
        self._environment = EnvironmentBroker(
            session=self, values={} if environment is None else environment
        )
        self._clock = ClockBroker(session=self, values={} if clocks is None else clocks)
        self._kernel_view = KernelBrokerSession(
            rng=self._rng,
            supplemental_rngs=self._supplemental_rngs,
            files=self._files,
            environment=self._environment,
            clock=self._clock,
        )

    @property
    def owner(self) -> BrokerOwner:
        return self._authority.owner

    @property
    def determinism(self) -> str:
        return self._authority.determinism

    @property
    def effects(self) -> frozenset[str]:
        return self._authority.effects

    @property
    def node_key(self) -> str | None:
        return self._authority.node_key

    @property
    def attempt(self) -> int:
        return self._authority.attempt

    @property
    def attempt_scope(self) -> str | None:
        return self._authority.attempt_scope

    @property
    def require_byte_equivalence(self) -> bool:
        return self._authority.require_byte_equivalence

    @property
    def run_provenance_identity(self) -> FrozenMap:
        return self._authority.run_provenance_identity

    @property
    def rng(self) -> RNGBroker:
        return self._rng

    @property
    def files(self) -> FileBroker:
        return self._files

    @property
    def source_behavior_identity(self) -> SourceBehaviorIdentity:
        return self._files.behavior_identity

    @property
    def environment(self) -> EnvironmentBroker:
        return self._environment

    @property
    def clock(self) -> ClockBroker:
        return self._clock

    @property
    def kernel_view(self) -> KernelBrokerSession:
        return self._kernel_view

    @classmethod
    def for_compiled_node(
        cls,
        node: CompiledNode,
        *,
        run_provenance_identity: Mapping[str, object],
        run_inputs: Mapping[str, int] | None = None,
        rng_invocation_plan: Mapping[str, Sequence[RNGInvocation]] | None = None,
        seed_stream_map: SeedStreamMap | None = None,
        supplemental_seed_owners: Sequence[SeedOwnerIR] = (),
        rng_invocation_plans_by_owner: Mapping[
            tuple[str, str], Mapping[str, Sequence[RNGInvocation]]
        ]
        | None = None,
        sources: Sequence[DeclaredSource] = (),
        environment: Mapping[str, str | None] | None = None,
        clocks: Mapping[str, float] | None = None,
        attempt: int = 0,
        attempt_scope: str | None = None,
        require_byte_equivalence: bool = True,
        physical_operation: PhysicalOperation | None = None,
    ) -> BrokerSession:
        capabilities = _wire(node.capabilities)
        if not isinstance(capabilities, Mapping):
            raise BrokerContractError("compiled node capabilities must be an object")
        effects = capabilities.get("effects")
        if not isinstance(effects, list):
            raise BrokerContractError("compiled node effects must be an array")
        session = cls(
            owner=BrokerOwner("producer_node", node.id),
            determinism=str(capabilities.get("determinism")),
            effects=[str(effect) for effect in effects],
            protocol_id="legacy-v1",
            protocol_sha256=node.seed_protocol_sha256,
            seed_sites=node.seed_sites,
            run_inputs=run_inputs,
            rng_invocation_plan=rng_invocation_plan,
            seed_stream_map=seed_stream_map,
            supplemental_seed_owners=supplemental_seed_owners,
            rng_invocation_plans_by_owner=rng_invocation_plans_by_owner,
            sources=sources,
            environment=environment,
            clocks=clocks,
            run_provenance_identity=run_provenance_identity,
            node_key=node.node_key,
            attempt=attempt,
            attempt_scope=attempt_scope,
            require_byte_equivalence=require_byte_equivalence,
        )
        if physical_operation is not None:
            session._bind_physical_operation(node, physical_operation)
        return session

    @classmethod
    def for_seed_owner(
        cls,
        stream_map: SeedStreamMap,
        *,
        owner_kind: str,
        owner_id: str,
        run_provenance_identity: Mapping[str, object],
        determinism: str = "seeded",
        effects: Sequence[str] = ("none",),
        run_inputs: Mapping[str, int] | None = None,
        rng_invocation_plan: Mapping[str, Sequence[RNGInvocation]] | None = None,
        sources: Sequence[DeclaredSource] = (),
        environment: Mapping[str, str | None] | None = None,
        clocks: Mapping[str, float] | None = None,
        attempt: int = 0,
        attempt_scope: str | None = None,
        require_byte_equivalence: bool = True,
    ) -> BrokerSession:
        owner = BrokerOwner(owner_kind, owner_id)
        grant = stream_map.owner(owner_kind, owner_id)
        if grant is None:
            raise BrokerContractError(
                f"seed owner {owner_kind}:{owner_id} is not in the compiled map"
            )
        site_ids = set(grant.sites)
        sites = tuple(site for site in stream_map.sites if site.id in site_ids)
        if tuple(site.id for site in sites) != grant.sites:
            raise BrokerContractError("seed owner site order differs from compiled map")
        if tuple(dict.fromkeys(site.stream for site in sites)) != grant.streams:
            raise BrokerContractError("seed owner streams differ from compiled map")
        return cls(
            owner=owner,
            determinism=determinism,
            effects=effects,
            protocol_id=stream_map.protocol_id,
            protocol_sha256=stream_map.implementation_sha256,
            seed_sites=sites,
            run_inputs=run_inputs,
            rng_invocation_plan=rng_invocation_plan,
            sources=sources,
            environment=environment,
            clocks=clocks,
            run_provenance_identity=run_provenance_identity,
            attempt=attempt,
            attempt_scope=attempt_scope,
            require_byte_equivalence=require_byte_equivalence,
        )

    @property
    def sealed(self) -> bool:
        return self._sealed_receipt is not None

    @property
    def active(self) -> bool:
        return self._active_activation is not None

    def _require_active(self) -> None:
        if self.sealed:
            raise BrokerAccessError("broker session is sealed")
        policy = _ACTIVE_POLICY.get()
        if (
            policy is None
            or policy.session is not self
            or policy.activation is not self._active_activation
        ):
            raise BrokerAccessError("broker access requires the active bound session")
        if policy.role != "kernel":
            self._log.record(
                broker="ambient",
                operation="explicit_broker_access",
                resource=policy.role,
                disposition="refused",
                reason_code="row_classifier_broker_access_prohibited",
            )
            raise BrokerAccessError(
                "row classifiers may not consume kernel broker authority"
            )

    def _bind_physical_operation(
        self, node: CompiledNode, operation: PhysicalOperation
    ) -> None:
        if self._physical_operation is not None:
            raise BrokerContractError("physical operation is already bound")
        if not isinstance(operation, PhysicalOperation):
            raise BrokerContractError(
                "physical operation must be a PhysicalOperation contract"
            )
        if (
            self.owner != BrokerOwner("producer_node", node.id)
            or self.node_key != node.node_key
        ):
            raise BrokerContractError(
                "physical operation node differs from its broker session"
            )
        if operation.implementation_sha256 != node.kernel_implementation_sha256:
            raise BrokerContractError(
                "physical operation implementation digest differs from compiled node"
            )
        if self._authority.protocol_id != "legacy-v1":
            raise BrokerContractError(
                "physical operation compatibility requires protocol legacy-v1"
            )
        if self.determinism == "nondeterministic":
            raise BrokerContractError(
                "physical operation compatibility does not support nondeterminism"
            )
        if operation.policy == "legacy-v1" and self._supplemental_rngs:
            raise BrokerContractError(
                "supplemental RNG owners require broker-only physical policy"
            )
        has_sink_effect = "declared_sink_write" in self.effects
        if has_sink_effect != bool(operation.sink_roots):
            raise BrokerContractError(
                "physical operation sink roots must exactly match sink-write authority"
            )
        object.__setattr__(self, "_physical_operation", operation)

    def _run_physical_operation(self, *, input_binding_sha256: str) -> object:
        self._require_active()
        operation = self._physical_operation
        if operation is None:
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_operation_not_bound",
            )
            raise BrokerAccessError("broker session has no physical operation")
        if (
            not isinstance(input_binding_sha256, str)
            or len(input_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in input_binding_sha256
            )
        ):
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_input_binding_shape_invalid",
            )
            raise BrokerAccessError(
                "physical operation input binding must be lowercase sha256"
            )
        if input_binding_sha256 != operation.input_binding_sha256:
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_input_binding_mismatch",
                details={"actual_sha256": input_binding_sha256},
            )
            raise BrokerAccessError(
                "physical operation input binding differs from its contract"
            )
        if self._physical_operation_invoked:
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_operation_repeat_invocation",
            )
            raise BrokerAccessError("physical operation may be invoked only once")
        object.__setattr__(self, "_physical_operation_invoked", True)
        try:
            if operation.policy == "legacy-v1":
                with _physical_operation_compatibility_scope(
                    self,
                    operation,
                    legacy_rng=True,
                ):
                    result = operation.function()
            else:
                with _physical_operation_compatibility_scope(
                    self,
                    operation,
                    legacy_rng=False,
                ):
                    result = operation.function(self.kernel_view)
        except BaseException:
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_operation_failed",
            )
            raise
        if _contains_raw_generator(result):
            self._log.record(
                broker="rng",
                operation="physical_operation_result",
                resource=(
                    self.rng.granted_sites[0]
                    if self.rng.granted_sites
                    else self.owner.id
                ),
                disposition="refused",
                reason_code="physical_rng_authority_returned",
            )
            raise BrokerAccessError(
                "physical operation may not return raw RNG authority"
            )
        return result

    def validate_executor_binding(
        self,
        *,
        node: CompiledNode,
        determinism: str,
        effects: Sequence[str],
        attempt: int,
        attempt_scope: str | None,
        require_byte_equivalence: bool,
        run_provenance_identity: Mapping[str, object],
    ) -> None:
        expected_owner = BrokerOwner("producer_node", node.id)
        expected_effects = frozenset(effects)
        checks = {
            "owner": self.owner == expected_owner,
            "node_key": self.node_key == node.node_key,
            "determinism": self.determinism == determinism,
            "effects": self.effects == expected_effects,
            "attempt": self.attempt == attempt,
            "attempt_scope": self.attempt_scope == attempt_scope,
            "byte_equivalence": (
                self.require_byte_equivalence == require_byte_equivalence
            ),
            "run_provenance_identity": self.run_provenance_identity
            == _run_provenance_identity(run_provenance_identity),
            "seed_protocol": self.rng.protocol_sha256 == node.seed_protocol_sha256,
            "seed_sites": self.rng.granted_sites
            == tuple(site.id for site in node.seed_sites),
            "seed_streams": self.rng.granted_streams == node.seed_streams,
            "physical_operation": (
                self._physical_operation is None
                or self._physical_operation.implementation_sha256
                == node.kernel_implementation_sha256
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise BrokerContractError(
                f"broker session differs from compiled executor binding: {failures!r}"
            )

    def validate_callable(self, function: object, *, role: str) -> None:
        """Reject ambient primitives captured before the dynamic guard."""

        if self.sealed:
            raise BrokerAccessError("broker session is sealed")
        if not isinstance(function, FunctionType):
            self._log.record(
                broker="ambient",
                operation="callable_shape_scan",
                resource=role,
                disposition="refused",
                reason_code="uninspectable_callable_shape",
                details={"type": type(function).__qualname__},
            )
            raise AmbientAccessError(
                f"{role} must be a directly inspectable Python function"
            )
        hits = _prebound_ambient_hits(function)
        if not hits:
            return
        self._log.record(
            broker="ambient",
            operation="prebound_callable_scan",
            resource=role,
            disposition="refused",
            reason_code="prebound_ambient_access",
            details={"bindings": list(hits)},
        )
        raise AmbientAccessError(
            f"{role} captures prohibited ambient access: {list(hits)!r}"
        )

    @contextmanager
    def activate(self) -> Iterator[BrokerSession]:
        if self.sealed:
            raise BrokerAccessError("broker session is sealed")
        if self._active_activation is not None:
            raise BrokerContractError("broker session is already active")
        activation = object()
        object.__setattr__(self, "_active_activation", activation)
        self._torch_rng_leases.prepare()
        try:
            with _ambient_guard(_ActivePolicy(self, activation, "kernel")):
                yield self
        finally:
            object.__setattr__(self, "_active_activation", None)

    @contextmanager
    def classifier_scope(self) -> Iterator[None]:
        """Deny explicit broker authority while the trusted classifier runs."""

        policy = _ACTIVE_POLICY.get()
        if (
            policy is None
            or policy.session is not self
            or policy.activation is not self._active_activation
            or policy.role != "kernel"
        ):
            raise BrokerContractError(
                "row classifier scope requires the active kernel session"
            )
        token = _ACTIVE_POLICY.set(
            _ActivePolicy(self, policy.activation, "row_classifier")
        )
        try:
            yield
        finally:
            _ACTIVE_POLICY.reset(token)

    def _audit_allowed_events(self) -> None:
        """Fail closed if a broker event is incompatible with sealed authority."""

        granted_sites = frozenset(
            site_id
            for rng in (self.rng, *self._supplemental_rngs)
            for site_id in rng.granted_sites
        )
        for event in self._log.events():
            if event.disposition != "allowed":
                continue
            authorized = {
                "rng": (
                    self.determinism == "seeded" and event.resource in granted_sites
                ),
                "file": (
                    "declared_source_read" in self.effects
                    or (
                        "declared_sink_write" in self.effects
                        and event.operation == "physical_operation_sink_scope"
                        and event.reason_code == "legacy_v1_physical_sink_grant"
                    )
                ),
                "environment": (
                    self.determinism == "nondeterministic"
                    and "declared_source_read" in self.effects
                ),
                "clock": (
                    self.determinism == "nondeterministic"
                    and not self.require_byte_equivalence
                ),
                "ambient": False,
            }[event.broker]
            if event.broker == "ambient" and (
                event.operation == "physical_operation_dependency_environment"
                and event.reason_code == "pinned_dependency_environment_default"
                and event.resource in _PINNED_DEPENDENCY_ENVIRONMENT_DEFAULTS
            ):
                authorized = True
            if authorized:
                continue
            self._log.record(
                broker="ambient",
                operation="authority_integrity",
                resource=f"event:{event.sequence}",
                disposition="refused",
                reason_code="allowed_event_outside_sealed_authority",
                details={"broker": event.broker, "operation": event.operation},
            )

    def seal(
        self, *, status: Literal["complete", "aborted"] = "complete"
    ) -> BrokerReceipt:
        if self._active_activation is not None:
            raise BrokerContractError("cannot seal an active broker session")
        if self._sealed_receipt is not None:
            return self._sealed_receipt
        if (
            status == "complete"
            and self._physical_operation is not None
            and not self._physical_operation_invoked
        ):
            self._log.record(
                broker="ambient",
                operation="physical_operation_dispatch",
                resource=self.owner.id,
                disposition="refused",
                reason_code="physical_operation_not_invoked",
            )
        if status == "complete":
            for rng in (self.rng, *self._supplemental_rngs):
                for site_id, consumed, declared in (
                    rng._unconsumed_declared_invocations()
                ):
                    self._log.record(
                        broker="rng",
                        operation="invocation_plan_consumption",
                        resource=site_id,
                        disposition="refused",
                        reason_code="rng_declared_invocations_unconsumed",
                        details={
                            "owner": rng._owner.to_wire(),
                            "consumed_invocations": consumed,
                            "declared_invocations": declared,
                        },
                    )
        self._rng_leases.close_all()
        self._torch_rng_leases.close_all()
        file_close_error: BrokerAccessError | None = None
        try:
            self.files.close_all()
        except BrokerAccessError as error:
            # Snapshot close failures already record a refused event.  Finish
            # sealing every lease and issue the aborted receipt before the
            # caller receives the refusal.
            file_close_error = error
        self._audit_allowed_events()
        events = self._log.events()
        tainted = any(event.disposition == "refused" for event in events)
        requested_status = status
        if requested_status == "complete" and tainted:
            status = "aborted"
        provisional = BrokerReceipt(
            owner=self.owner,
            node_key=self.node_key,
            attempt=self.attempt,
            attempt_scope=self.attempt_scope,
            status=status,
            run_provenance_identity=self.run_provenance_identity,
            events=events,
            receipt_sha256="",
        )
        receipt = BrokerReceipt(
            owner=self.owner,
            node_key=self.node_key,
            attempt=self.attempt,
            attempt_scope=self.attempt_scope,
            status=status,
            run_provenance_identity=self.run_provenance_identity,
            events=events,
            receipt_sha256=sha256_json(provisional.body_wire()),
        )
        receipt.validate()
        object.__setattr__(self, "_sealed_receipt", receipt)
        if requested_status == "complete" and (tainted or file_close_error is not None):
            refused = [
                f"{event.broker}:{event.operation}:{event.reason_code}"
                for event in events
                if event.disposition == "refused"
            ]
            raise BrokerAccessError(
                "broker session recorded a refused access and cannot complete: "
                + ", ".join(refused)
            ) from file_close_error
        return receipt

    @property
    def receipt(self) -> BrokerReceipt:
        if self._sealed_receipt is None:
            raise BrokerAccessError("broker session has not been sealed")
        return self._sealed_receipt


def deny_all_session_for_node(
    node: CompiledNode,
    *,
    run_provenance_identity: Mapping[str, object],
    attempt: int,
    attempt_scope: str | None,
    require_byte_equivalence: bool,
) -> BrokerSession:
    """Construct the implicit empty session used by pure deterministic nodes."""

    capabilities = _wire(node.capabilities)
    if not isinstance(capabilities, Mapping):
        raise BrokerContractError("compiled node capabilities must be an object")
    if capabilities.get("determinism") != "deterministic" or capabilities.get(
        "effects"
    ) != ["none"]:
        raise BrokerContractError(
            "an explicit broker session is required outside pure deterministic nodes"
        )
    return BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=run_provenance_identity,
        attempt=attempt,
        attempt_scope=attempt_scope,
        require_byte_equivalence=require_byte_equivalence,
    )


__all__ = [
    "AmbientAccessError",
    "BrokerAccessError",
    "BrokerAccessEvent",
    "BrokerContractError",
    "BrokerError",
    "BrokerOwner",
    "BrokerReceipt",
    "BrokerSession",
    "ClockBroker",
    "DeclaredSource",
    "DerivedSeedHandle",
    "EnvironmentBroker",
    "FileBroker",
    "FileReadLease",
    "GeneratorLease",
    "KernelBrokerSession",
    "PhysicalOperation",
    "QRFGeneratorLease",
    "RNGBroker",
    "RNGBehaviorIdentity",
    "RNGInvocation",
    "RNGStreamToken",
    "SourceBehaviorIdentity",
    "TorchGeneratorLease",
    "deny_all_session_for_node",
]
