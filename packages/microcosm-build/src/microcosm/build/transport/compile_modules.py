"""The modules whose source decides a transport target surface.

``targets.compile@1`` and every other transport kernel built on target
compilation hash :func:`transport_compile_modules` (through
``target_kernels._COMPILE_MODULES``). ``source_hash`` hashes the files it is
given and does not follow imports, so the tuple has to name every module the
compile reads; ``ledger_targets`` labels a target's geography from
``microcosm.calibrate.geography_constants``, which no compile kernel hashed
before microcosm#1492.

The list lives here, in a module no kernel hashes, as the UK's lives in
``uk_runtime.target_compile_modules``: editing a name or a reason re-keys
only the kernels whose hashed modules the edit adds or removes, not every
kernel that hashes ``target_kernels``.

``EXCLUDED_DIRECT_IMPORTS`` names every first-party (``microcosm.*``) module
a listed module imports but the tuple leaves out, with the reason. A test
pins that the two together cover every direct import of every listed module,
so a new import must be listed or excluded. The exclusions cover only the
listed modules' own imports; a second test walks every module each compile
kernel hashes, its own module included, and records the imports no review
has classified yet (microcosm#1497).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType, ModuleType

TRANSPORT_COMPILE_MODULE_NAMES: tuple[str, ...] = (
    "microcosm.build.transport.graph_inputs",
    "microcosm.build.transport.artifact_types",
    "microcosm.build.country_spec",
    "microcosm.build.ledger_targets",
    "microcosm.build.ledger_artifact",
    "microcosm.build.chronicle_epoch",
    "microcosm.build.reference_factors",
    "microcosm.build.reference_profiles",
    "microcosm.calibrate.registry",
    "microcosm.calibrate.target",
    "microcosm.calibrate.hierarchy",
    "microcosm.calibrate.geography_constants",
    "microcosm.build.transport.reference_selection",
)

_COUNTRY_SPEC_LOADER = (
    "country_spec is listed for the reference validators target compilation "
    "calls, _validate_target_references and _carried_value_keys; it imports "
    "this module for its country-spec loader and stage plans, which those "
    "validators do not use"
)
_GRAPH_RUNTIME = (
    "the graph runtime (kernel protocol, declarations and the canonical JSON "
    "encoder node keys are computed with); no kernel in the repository "
    "hashes a microcosm.graph module"
)

EXCLUDED_DIRECT_IMPORTS: Mapping[str, str] = MappingProxyType(
    {
        "microcosm.build.plan": _COUNTRY_SPEC_LOADER,
        "microcosm.build.source_manifest": _COUNTRY_SPEC_LOADER,
        "microcosm.build.trace": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.canonical": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.engine_abi": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.loader": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.model": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.schemas": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.take_up_semantics": _COUNTRY_SPEC_LOADER,
        "microcosm.build.spec_engine.yaml12": _COUNTRY_SPEC_LOADER,
        "microcosm.build.reference_intervals": (
            "reference_profiles reaches it only when given an interval "
            "document; compile_target_surface calls validate_reference_resolution "
            "without one. calibration.prepare@1 and targets.problem@1, which "
            "apply interval tables, hash it"
        ),
        "microcosm.calibrate.artifacts": (
            "artifact_types imports PROBLEM_TYPE to declare targets.problem@1's "
            "output; target compilation checks only its own surface output"
        ),
        "microcosm.diagnostics.schema": (
            "artifact_types imports the diagnostics schema version to declare "
            "the diagnostics output type; target compilation emits no diagnostics"
        ),
        "microcosm.frame.bundle": (
            "the population container: graph_inputs.context_frame builds one, "
            "calibrate.target measures a target on one and country_spec's stage "
            "plans transform one; target compilation reads facts, not a population"
        ),
        "microcosm.frame.schema": (
            "graph_inputs.context_frame's entity schema; target compilation "
            "reads no population"
        ),
        "microcosm.graph.canonical": _GRAPH_RUNTIME,
        "microcosm.graph.decl": _GRAPH_RUNTIME,
        "microcosm.graph.kernel": _GRAPH_RUNTIME,
    }
)


def transport_compile_modules() -> tuple[ModuleType, ...]:
    """The modules of :data:`TRANSPORT_COMPILE_MODULE_NAMES`, imported."""

    return tuple(
        importlib.import_module(name) for name in TRANSPORT_COMPILE_MODULE_NAMES
    )
