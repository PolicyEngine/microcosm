"""Reviewed, path-independent implementation identities for the US source graph.

Whole modules remain the unit of code identity. The packaged inventory is a
reviewed dependency fence, not a Python sandbox or inferred call-graph proof.
Its import/use/resource expressions prevent silently extending a reviewed route
to an unbound helper. Updating that inventory requires a scope review.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import pandas as pd

STAGE_DEPENDENCIES = MappingProxyType(
    {
        "asec_codec_v4": ("numpy", "pandas", "h5py"),
        "acs_codec_2024": ("numpy", "pandas", "microunit", "PyYAML", "pyarrow"),
        "acs_housing_universe_2024": (
            "numpy",
            "pandas",
            "microunit",
            "PyYAML",
            "pyarrow",
        ),
        # Full authenticated source catalogues, one household draw, selected
        # native construction and one allocation. This deliberately excludes
        # legacy prepared-source income observations and engine resources.
        "authenticated_survey_population_v1": (
            "numpy",
            "pandas",
            "microunit",
            "tables",
            "h5py",
            "PyYAML",
            "pyarrow",
        ),
        "assembly_prepare": (
            "numpy",
            "pandas",
            "microunit",
            "tables",
            "h5py",
            "PyYAML",
            "pyarrow",
        ),
        "assembly_harmonize": ("numpy", "pandas"),
        "geography": ("numpy", "pandas"),
        # The ASEC prepared current-money slice. PolicyEngine-US is deliberately
        # absent: its version is a normative parameter of the evaluation node and
        # a pinned field of every admitted closure contract, so engine drift
        # refuses there. Putting it here instead would make a secrets-free lane
        # without the engine extra unable to hash any stage of this graph.
        "asec_prepared_v3": (
            "numpy",
            "pandas",
            "microunit",
            "tables",
            "h5py",
            "PyYAML",
            "pyarrow",
        ),
        # The composed prepared-ASEC/native-ACS population. It runs both source
        # scopes in one CREATE, so its module inventory is their union and its
        # dependency set is theirs (identical in both). PolicyEngine-US is
        # absent for the same reason it is absent from the prepared scope: no
        # engine node lives in this graph.
        "composed_population_v1": (
            "numpy",
            "pandas",
            "microunit",
            "tables",
            "h5py",
            "PyYAML",
            "pyarrow",
        ),
        # Binding the carried prepared-ASEC evidence to the composed ASEC rows
        # and deriving the reviewed corrected leaves and reported observations
        # over them. It runs no source loader of its own, but its resolution and
        # every verifier it reuses live inside the composing scope, so its module
        # inventory is that scope's plus the two binding modules and its
        # dependency set is identical.
        "composed_asec_binding_v1": (
            "numpy",
            "pandas",
            "microunit",
            "tables",
            "h5py",
            "PyYAML",
            "pyarrow",
        ),
    }
)
_CODECS = MappingProxyType(
    {
        "us-asec-raw-stage-v4": (
            "asec_codec_v4",
            "microcosm.build.us_runtime.graph_sources:load_graph_asec",
        ),
        "us-acs-native-2024-v1": (
            "acs_codec_2024",
            "microcosm.build.us_runtime.graph_sources:load_graph_acs",
        ),
        "us-asec-prepared-current-money-v3": (
            "asec_prepared_v3",
            "microcosm.build.us_runtime.asec_prepared_source:load_graph_asec_prepared",
        ),
        "raw-bytes-v1": ("geography", "microcosm.graph.codecs:load_raw_bytes"),
        "us-acs-housing-universe-2024-v1": (
            "acs_housing_universe_2024",
            "microcosm.build.us_runtime.acs_housing_universe_source:load_graph_acs_housing_universe",
        ),
    }
)
_RESOURCE_CALLS = frozenset(
    {
        "open",
        "read_bytes",
        "read_text",
        "read_csv",
        "read_parquet",
        "read_hdf",
        "read_excel",
        "read_json",
        "read_table",
        "read_feather",
        "File",
        "loadtxt",
        "genfromtxt",
        "load",
        "safe_load",
        "files",
        "joinpath",
        "import_module",
        "__import__",
        "exec",
        "eval",
    }
)


# These local lazy entry points are dependencies even though no ImportFrom
# binds their names. Record references AND complete calls in their caller scope:
# passing the dispatcher around must not hide its use, and changing a selected
# symbol must require review. This adds tripwires; it grants no scope exemption.
_LOCAL_IMPORT_DISPATCHERS = {
    "microcosm.build/us_runtime/graph_sources.py": frozenset(
        {
            "_stacked_alias",
            "__getattr__",
            "assemble_stacked_spine",
            "prepare_stacked_spine",
            "harmonize_stacked_spine_weights",
        }
    ),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _package_roots() -> dict[str, Path]:
    # Resolve packages, never import leaf modules (including optional adapters)
    # merely to hash their files. Editable and wheel installs use the same names.
    roots = {}
    for package in (
        "microcosm.build",
        "microcosm.frame",
        "microcosm.graph",
        "microunit",
    ):
        spec = importlib.util.find_spec(package)
        if spec is None and package == "microunit":
            continue  # Only the ACS/CREATE scopes require this dependency.
        locations = () if spec is None else spec.submodule_search_locations or ()
        if len(locations) != 1:
            raise ValueError(
                f"US implementation package inventory unavailable: {package}."
            )
        roots[package] = Path(next(iter(locations))).resolve()
    return roots


@lru_cache(maxsize=256)
def _dependency_details(payload: bytes, name: str, covered: tuple[str, ...]) -> dict:
    """Static tripwire cached by exact bytes, never by path, mtime or version."""
    tree = ast.parse(payload)
    imports, aliases, unbound_imports = set(), {}, set()
    package, relative = name.split("/", 1)
    module = package + "." + relative.removesuffix(".py").replace("/", ".")
    parent = module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    raise ValueError("Unclassified wildcard dependency import.")
                target = (
                    alias.name
                    if isinstance(node, ast.Import)
                    else (
                        importlib.util.resolve_name(
                            "." * node.level + (node.module or ""), parent
                        )
                        if node.level
                        else node.module
                    )
                )
                is_stdlib = target.split(".")[0] in sys.stdlib_module_names
                if not is_stdlib:
                    imports.add(target)
                is_bound = target in covered or any(
                    target == item[1:] or target.startswith(item[1:] + ".")
                    for item in covered
                    if item.startswith("*")
                )
                if not is_stdlib and not is_bound:
                    aliases[alias.asname or alias.name.split(".")[0]] = target
                    unbound_imports.add(f"import:{target}:{alias.name}")
    resources, uses = set(), set(unbound_imports)
    dispatchers = _LOCAL_IMPORT_DISPATCHERS.get(name, ())

    class Visitor(ast.NodeVisitor):
        scope = "<module>"

        def visit_FunctionDef(self, node):
            previous = self.scope
            self.scope = f"{previous}.{node.name}"
            self.generic_visit(node)
            self.scope = previous

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 - AST visitor API
        visit_ClassDef = visit_FunctionDef  # noqa: N815 - AST visitor API

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in dispatchers:
                resources.add(
                    f"{self.scope}:{ast.dump(node, include_attributes=False)}"
                )
            if isinstance(node.ctx, ast.Load) and node.id in aliases:
                uses.add(f"{self.scope}:{aliases[node.id]}:{node.id}")

        def visit_Call(self, node):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None)
            )
            # Bind reflective selection too, including the dispatcher's
            # getattr(import_module(...), name) target expression.
            if (
                name in _RESOURCE_CALLS
                or name in dispatchers
                or (dispatchers and name == "getattr")
            ):
                resources.add(
                    f"{self.scope}:{ast.dump(node, include_attributes=False)}"
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return {
        "imports": sorted(imports),
        "unbound_uses": sorted(uses),
        "resource_accesses": sorted(resources),
    }


def _covered_imports(name: str, inventory: dict) -> tuple[str, ...]:
    """Omit a symbol tripwire only if every scope using this file binds it."""
    scopes = [spec for spec in inventory["stages"].values() if name in spec["modules"]]
    modules = set.intersection(*(set(spec["modules"]) for spec in scopes))
    dependencies = set.intersection(*(set(spec["dependencies"]) for spec in scopes))
    covered = set()
    for module in modules:
        package, relative = module.split("/", 1)
        resolved = package + "." + relative.removesuffix(".py").replace("/", ".")
        covered.add(resolved.removesuffix(".__init__"))
    covered.update(
        "*" + ("yaml" if dependency == "PyYAML" else dependency)
        for dependency in dependencies
    )
    return tuple(sorted(covered))


def _dependency_contract(payload: bytes, name: str, covered: tuple[str, ...]) -> dict:
    details = _dependency_details(payload, name, covered)
    return {
        "imports": details["imports"],
        "unbound_uses_sha256": _digest(_canonical(details["unbound_uses"])),
        "resource_accesses_sha256": _digest(_canonical(details["resource_accesses"])),
    }


def _inventory(roots: Mapping[str, Path]) -> tuple[dict, bytes]:
    payload = (
        roots["microcosm.build"] / "us_runtime/graph_implementation_inventory.json"
    ).read_bytes()
    value = json.loads(payload)
    if value["schema"] != "microcosm.us.implementation-inventory.v1":
        raise ValueError("Unsupported US implementation inventory schema.")
    if set(value["stages"]) != set(STAGE_DEPENDENCIES):
        raise ValueError("US implementation stage dependency inventory differs.")
    for name, dependencies in STAGE_DEPENDENCIES.items():
        if value["stages"][name]["dependencies"] != list(dependencies):
            raise ValueError(f"US implementation dependency inventory differs: {name}.")
    imports = {
        name for contract in value["contracts"].values() for name in contract["imports"]
    }
    if set(value["import_classifications"]) != imports:
        raise ValueError("US implementation import classification inventory differs.")
    return value, payload


def _validate_package_roster(package: str, root: Path, inventory: dict) -> None:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    expected = inventory["package_rosters"][package]
    if len(expected) != len(set(expected)) or actual != set(expected):
        raise ValueError(f"US implementation Python inventory differs: {package}.")
    if package == "microunit":
        # py.typed is packaging metadata. Everything else must be explicitly
        # bound, including an added non-Python rule/data file.
        actual_resources = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix != ".py"
            and "__pycache__" not in path.parts
            and path.name != "py.typed"
        }
        if actual_resources != set(inventory["package_resources"][package]):
            raise ValueError("US implementation resource inventory differs: microunit.")


def _projections() -> dict:
    from . import operator_boundary

    def columns(value):
        if isinstance(value, Mapping):
            return {key: columns(item) for key, item in sorted(value.items())}
        if not isinstance(value, (tuple, list, frozenset, set)) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                "US source registry projection requires string column sets."
            )
        return sorted(value)

    return {
        name: columns(getattr(operator_boundary, name))
        for name in (
            "PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES",
            "FORMULA_OWNED_SOURCE_COLUMNS",
        )
    }


def _loader_ref(loader: object, expected: str, roots: Mapping[str, Path]) -> None:
    actual = (
        f"{getattr(loader, '__module__', '')}:{getattr(loader, '__qualname__', '')}"
    )
    module, _ = expected.split(":")
    package = next(name for name in roots if module.startswith(name + "."))
    path = roots[package] / (
        module.removeprefix(package + ".").replace(".", "/") + ".py"
    )
    # A correctly named function imported from another checkout is not the
    # implementation whose file bytes this manifest promises.
    source = inspect.getsourcefile(loader) if inspect.isfunction(loader) else None
    if actual != expected or source is None or Path(source).resolve() != path.resolve():
        raise ValueError(
            f"US source codec loader differs from declared reference: {expected}."
        )


def validate_source_codecs(codecs) -> None:
    """Bind US registry names to real loaders; generic graph codec APIs stay unchanged."""
    roots = _package_roots()
    # Explicit declared selectors keep this registry inspection source-blind.
    try:
        loaders = (
            codecs.get("us-asec-raw-stage-v4"),
            codecs.get("us-acs-native-2024-v1"),
            codecs.get("us-asec-prepared-current-money-v3"),
            codecs.get("raw-bytes-v1"),
            codecs.get("us-acs-housing-universe-2024-v1"),
        )
    except Exception as error:
        raise ValueError("US source codec loader missing.") from error
    for (_, (_, expected)), loader in zip(_CODECS.items(), loaders, strict=True):
        _loader_ref(loader, expected, roots)


def implementation_manifest(stage: str) -> dict:
    """Return small auditable identities, never source data or absolute paths."""
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"Unclassified US implementation stage: {stage}.")
    roots = _package_roots()
    inventory, inventory_bytes = _inventory(roots)
    spec = inventory["stages"][stage]
    modules, resources = {}, {}
    packages = {name.split("/", 1)[0] for name in spec["modules"]}
    if not packages <= roots.keys():
        raise ValueError("US implementation dependency package inventory unavailable.")
    for package in packages - {"microcosm.build"}:
        _validate_package_roster(package, roots[package], inventory)
    if len(spec["modules"]) != len(set(spec["modules"])):
        raise ValueError("Duplicate US implementation module inventory.")
    for name in spec["modules"]:
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        if not path.is_file():
            raise ValueError(f"US implementation module inventory missing: {name}.")
        payload = path.read_bytes()
        actual = _dependency_contract(payload, name, _covered_imports(name, inventory))
        if actual != inventory["contracts"][name]:
            raise ValueError(f"Unclassified US dependency/resource contract: {name}.")
        modules[name] = _digest(payload)
    for name in spec["resources"]:
        package, relative = name.split("/", 1)
        resources[name] = _digest((roots[package] / relative).read_bytes())
    manifest = {
        "schema": "microcosm.us.implementation-manifest.v1",
        "stage": stage,
        "inventory_sha256": _digest(inventory_bytes),
        "modules": modules,
        "resources": resources,
        "dependencies": {
            name: importlib_metadata.version(name) for name in STAGE_DEPENDENCIES[stage]
        },
        "projections": _projections() if spec["source_boundary_projection"] else {},
        "runtime_options": (
            {
                "pandas.mode.string_storage": pd.get_option("mode.string_storage"),
                "pandas.future.infer_string": pd.get_option("future.infer_string"),
                "acs_reader_resolved_string_storage": pd.StringDtype().storage,
            }
            if stage
            in {
                "acs_codec_2024",
                "assembly_prepare",
                "acs_housing_universe_2024",
                "authenticated_survey_population_v1",
                "composed_population_v1",
                "composed_asec_binding_v1",
            }
            else {}
        ),
        "codecs": {},
    }
    if stage in {
        "asec_prepared_v3",
        "acs_housing_universe_2024",
        "authenticated_survey_population_v1",
        "composed_population_v1",
        "composed_asec_binding_v1",
    }:
        # These stages parse pinned official CSV members with the stdlib reader.
        # csv's field size limit is process state, so it belongs in the identity
        # of each stage that reads under it. Whole-module and inventory hashes
        # separately record code changes, including additions of new stages.
        manifest["runtime_options"]["csv.field_size_limit"] = csv.field_size_limit()
    if stage in {
        "assembly_prepare",
        "composed_population_v1",
        "composed_asec_binding_v1",
    }:
        bound = (
            ("asec_codec_v4", "acs_codec_2024")
            if stage == "assembly_prepare"
            # The composed CREATE reads the prepared directory, not the
            # raw-stage checkpoint, so it binds that codec's identity instead.
            # The binding stage reads no source, but every artifact it admits
            # was produced under those two codecs, so it binds them too rather
            # than claiming an identity narrower than its actual inputs.
            else ("asec_prepared_v3", "acs_codec_2024")
        )
        for name, (codec_stage, loader) in _CODECS.items():
            if codec_stage in bound:
                manifest["codecs"][name] = {
                    "loader": loader,
                    "implementation_sha256": implementation_hash(codec_stage),
                }
    elif stage == "authenticated_survey_population_v1":
        # The standalone factory registers one real closure over an operational
        # capture directory. Its whole module belongs to this stage. The legacy
        # five-codec registry and its strict validator retain their contract.
        manifest["codecs"] = {
            "us-survey-population-source-v1": {
                "loader": (
                    "microcosm.build.us_runtime.graph_survey_population:"
                    "survey_population_source_codecs.<locals>.load"
                )
            }
        }
    elif stage in (
        "asec_codec_v4",
        "acs_codec_2024",
        "geography",
        "asec_prepared_v3",
        "acs_housing_universe_2024",
    ):
        manifest["codecs"] = {
            name: {"loader": loader}
            for name, (codec_stage, loader) in _CODECS.items()
            if codec_stage == stage
        }
    return manifest


def implementation_hash(stage: str) -> str:
    return _digest(
        b"microcosm.us.scoped-implementation.v1\0"
        + _canonical(implementation_manifest(stage))
    )
