"""Compile atomic geography support from a precalibration target surface.

The resolved definition is spec data: atomic-area codes, each row's compiled
population target, population shares, and the existing atomic-support column
metadata. Integer population is apportioned within each target with exact
largest remainders, conserving its total. No geography mapping or population
value is inferred from country names. Empty, incomplete or ambiguous support
refuses execution, including an unharvested crosswalk.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from fractions import Fraction

import numpy as np

import microcosm.build.atomic_geography as atomic_geography_module
import microcosm.calibrate.hierarchy as hierarchy_module
import microcosm.calibrate.registry as registry_module
from microcosm.build.atomic_geography import _U53, encode_atomic_support
from microcosm.build.graph_atomic_geography import ATOMIC_SUPPORT_TYPE
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import graph_inputs, target_kernels
from .artifact_types import TARGET_SURFACE_TYPE
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    sha256_text,
    string_param,
)
from .target_kernels import TargetSurface, decode_target_surface

__all__ = [
    "ATOMIC_SUPPORT_TYPE",
    "GEOGRAPHY_SUPPORT_FROM_FACTS",
    "GeographySupportFromFactsKernel",
    "support_from_surface",
    "register_geography_kernels",
]

_DEFINITION_KEYS = frozenset(
    {
        "version",
        "level",
        "code_system",
        "vintage",
        "columns",
        "rows",
        "share_tolerance",
        "population_geography_level",
        "population_code_column",
    }
)


def support_from_surface(
    surface: TargetSurface, definition: Mapping[str, object], *, system: str
) -> bytes:
    """Resolve declared support rows through already compiled population facts.

    Each row is ``{target, codes, population_share}``; ``codes`` contains
    exactly the definition's code columns, including a unique ``area``.
    The population column is the sole weight column and follows the existing
    atomic-support contract. A target's hierarchy geographic id must equal
    the declared population code. Population targets carry
    ``precal_use=atomic_geography_support`` and ``ledger_measure_unit=count``
    and have integral values.

    Largest-remainder apportionment rounds shares to integer sampling counts,
    preserving each published target exactly; equal remainders use ascending
    atomic-area code. Sorting those codes also makes row permutation inert.
    """

    if (
        set(definition) != _DEFINITION_KEYS
        or type(definition["version"]) is not int
        or definition["version"] != 1
    ):
        raise ValueError("Atomic support definition has an unsupported shape.")
    columns = definition["columns"]
    if not isinstance(columns, Mapping):
        raise ValueError("Atomic support definition requires column metadata.")
    code_columns = {
        name
        for name, description in columns.items()
        if isinstance(description, Mapping) and description.get("kind") == "code"
    }
    weight_columns = {
        name
        for name, description in columns.items()
        if isinstance(description, Mapping) and description.get("kind") == "weight"
    }
    if (
        "area" not in code_columns
        or weight_columns != {"population"}
        or set(columns) != code_columns | weight_columns
    ):
        raise ValueError(
            "Atomic support declares code columns and one population weight column."
        )
    population_column = definition["population_code_column"]
    if population_column not in code_columns:
        raise ValueError("Atomic support population code column must be a code column.")
    tolerance = definition["share_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Atomic support share tolerance must be finite and nonnegative."
        )
    rows = definition["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Atomic support requires harvested nonempty rows.")
    groups: dict[str, list[tuple[Mapping[str, str], Fraction]]] = {}
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "target",
            "codes",
            "population_share",
        }:
            raise ValueError(
                "Atomic support row requires target, codes and population_share."
            )
        target, codes, share = row["target"], row["codes"], row["population_share"]
        if (
            not isinstance(target, str)
            or not target
            or not isinstance(codes, Mapping)
            or set(codes) != code_columns
        ):
            raise ValueError("Atomic support row target or code declaration differs.")
        if any(
            not isinstance(code, str) or not code or code.strip() != code
            for code in codes.values()
        ):
            raise ValueError(
                "Atomic support codes must be nonempty strings without whitespace."
            )
        if codes["area"] in seen:
            raise ValueError("Atomic support area codes must be unique.")
        seen.add(codes["area"])
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(share)
            or not 0 <= share <= 1
        ):
            raise ValueError(
                "Atomic support shares must be finite fractions in [0, 1]."
            )
        groups.setdefault(target, []).append((codes, Fraction(str(share))))
    specs = {spec.name: spec for spec in surface.registry.specs}
    if len(specs) != len(surface.registry.specs):
        raise ValueError(
            "Atomic support requires unambiguous target names across periods."
        )
    eligible = {
        spec.name
        for spec in surface.registry.specs
        if spec.metadata.get("precal_use") == "atomic_geography_support"
    }
    if set(groups) != eligible:
        raise ValueError(
            "Atomic support rows must cover exactly the surface's geography population targets."
        )
    apportioned = []
    for target, members in sorted(groups.items()):
        spec = specs[target]
        if spec.metadata.get("ledger_measure_unit") != "count":
            raise ValueError(
                "Atomic support population facts must declare count units."
            )
        if spec.value < 0 or not spec.value.is_integer():
            raise ValueError(
                "Atomic support population facts must be nonnegative integers."
            )
        if (
            spec.hierarchy is None
            or spec.hierarchy.geography.level
            != definition["population_geography_level"]
        ):
            raise ValueError(
                "Atomic support population fact has the wrong geographic level."
            )
        if any(
            codes[population_column] != spec.hierarchy.geography.id
            for codes, _ in members
        ):
            raise ValueError(
                "Atomic support row codes differ from the population fact geography."
            )
        members.sort(key=lambda member: member[0]["area"])
        total_share = sum((share for _, share in members), Fraction())
        if total_share <= 0 or abs(total_share - 1) > Fraction(str(tolerance)):
            raise ValueError(
                "Atomic support shares for each population target must sum to one."
            )
        population = int(spec.value)
        quotas = [population * share / total_share for _, share in members]
        counts = [quota.numerator // quota.denominator for quota in quotas]
        remainder = population - sum(counts)
        ranked = sorted(
            range(len(members)),
            key=lambda i: (-(quotas[i] - counts[i]), members[i][0]["area"]),
        )
        for index in ranked[:remainder]:
            counts[index] += 1
        apportioned.extend(
            (codes, count) for (codes, _), count in zip(members, counts, strict=True)
        )
    apportioned.sort(key=lambda item: item[0]["area"])
    # Validate the exact sampling bound before converting Python integers to
    # int64, avoiding a wrap or a numpy conversion error on excessive counts.
    if not 0 < sum(count for _, count in apportioned) <= _U53:
        raise ValueError(
            "Atomic support population total is outside the exact sampling range."
        )
    arrays = {
        name: np.asarray([codes[name] for codes, _ in apportioned], dtype=str)
        for name in sorted(code_columns)
    }
    arrays["population"] = np.asarray(
        [count for _, count in apportioned], dtype=np.int64
    )
    return encode_atomic_support(
        {
            "version": 1,
            "system": system,
            "level": definition["level"],
            "code_system": definition["code_system"],
            "vintage": definition["vintage"],
            "columns": dict(columns),
        },
        arrays,
    )


class GeographySupportFromFactsKernel(KernelBase):
    """``geography.support_from_facts@1`` over a declared target surface."""

    ref = "geography.support_from_facts@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy",),
        consumes_se=False,
    )

    def implementation_hash(self) -> str:
        return source_hash(
            sys.modules[__name__],
            atomic_geography_module,
            target_kernels,
            graph_inputs,
            registry_module,
            hierarchy_module,
            canonical_json,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset({"definition", "definition_sha256", "system"}),
        )
        digest = sha256_param(context, self.ref, "definition_sha256")
        definition = canonical_document_param(context, self.ref, "definition")
        system = string_param(context, self.ref, "system")
        declared = {
            output.name: output.type for output in context.node.artifact_outputs
        }
        if declared != {"support": ATOMIC_SUPPORT_TYPE}:
            raise ValueError(f"{self.ref} declares exactly its typed support output.")
        value = context.artifacts.get("surface")
        if (
            set(context.artifacts) != {"surface"}
            or value is None
            or value.type != TARGET_SURFACE_TYPE
        ):
            raise ValueError(
                f"{self.ref} reads one typed target surface under alias 'surface'."
            )
        if context.node.sources:
            raise ValueError(f"{self.ref} reads an artifact and declares no sources.")
        surface = decode_target_surface(value.payload)
        payload = support_from_surface(surface, definition, system=system)
        return KernelResult(
            artifacts={"support": payload},
            receipt={
                "system": system,
                "surface_sha256": surface.sha256,
                "definition_sha256": digest,
                "support_sha256": sha256_text(payload),
                "areas": len(definition["rows"]),
            },
        )


GEOGRAPHY_SUPPORT_FROM_FACTS = GeographySupportFromFactsKernel()


def register_geography_kernels(registry: KernelRegistry) -> None:
    """Register the population-facts support adapter explicitly."""

    registry.register(GEOGRAPHY_SUPPORT_FROM_FACTS)
