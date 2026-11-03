"""The transport compile kernels hash every module target compilation reads.

``targets.compile@1`` and every other transport kernel built on target
compilation hash ``target_kernels._COMPILE_MODULES``. ``source_hash`` hashes
the files it is given and does not follow imports, so a module that a listed
module imports, and that nobody lists, can change a compiled surface without
changing a node key. ``ledger_targets`` takes a target's geography label from
``microcosm.calibrate.geography_constants`` that way; before microcosm#1492 no
compile kernel hashed it, so a resumed store published stored surfaces with
the old label.

``COMPILE_EXCLUDED_IMPORTS`` names every first-party module a listed module
imports but the tuple leaves out, with the reason. It lives here rather than
next to the tuple because ``target_kernels.py`` is hashed: editing a reason
there would give every kernel that hashes the file a new key.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

import microcosm.calibrate.geography_constants as geography_constants_module
from microcosm.build.transport import target_kernels
from microcosm.build.transport.population_kernels import (
    TRANSPORT_CREATE,
    TRANSPORT_CREATE_DECLARED,
    TRANSPORT_QUANTILE_MAP,
)
from microcosm.build.transport.quantile_kernels import TRANSPORT_QUANTILE_MAP_V2
from microcosm.build.transport.takeup_kernels import TAKEUP_GAP
from microcosm.build.transport.target_kernels import TAKEUP_COMPARE, TARGETS_COMPILE
from test_support.microcosm_build.first_party_imports import (
    first_party_imports,
    uncovered_imports,
)

#: Every kernel whose implementation hash includes ``_COMPILE_MODULES``. The
#: first four compile references with ``compile_target_surface``; the two
#: quantile maps decode the surface an upstream compile stored.
KERNELS_HASHING_THE_COMPILE_MODULES = (
    TARGETS_COMPILE,
    TAKEUP_COMPARE,
    TAKEUP_GAP,
    TRANSPORT_CREATE,
    TRANSPORT_CREATE_DECLARED,
    TRANSPORT_QUANTILE_MAP,
    TRANSPORT_QUANTILE_MAP_V2,
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

COMPILE_EXCLUDED_IMPORTS = MappingProxyType(
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


def _compile_module_names() -> tuple[str, ...]:
    return tuple(module.__name__ for module in target_kernels._COMPILE_MODULES)


@pytest.mark.parametrize(
    "kernel", KERNELS_HASHING_THE_COMPILE_MODULES, ids=lambda kernel: kernel.ref
)
def test_an_edit_to_the_geography_labels_rekeys_the_kernel(
    tmp_path, monkeypatch, kernel
):
    """ledger_targets labels a target's geography from UK_GEOGRAPHY_ID_TO_LABEL
    when the facts give no name, so an edit there must give a new key."""
    before = kernel.implementation_hash()
    source = Path(geography_constants_module.__file__)
    edited = tmp_path / source.name
    edited.write_bytes(source.read_bytes() + b"\n# an edited geography label\n")
    monkeypatch.setattr(geography_constants_module, "__file__", str(edited))
    assert kernel.implementation_hash() != before


def test_every_first_party_import_of_a_compile_module_is_listed_or_excluded():
    uncovered = uncovered_imports(
        _compile_module_names(), frozenset(COMPILE_EXCLUDED_IMPORTS)
    )
    assert not uncovered, sorted(uncovered)


def test_compile_exclusions_are_live_and_never_listed():
    imported = set().union(*map(first_party_imports, _compile_module_names()))
    assert set(COMPILE_EXCLUDED_IMPORTS) <= imported, sorted(
        set(COMPILE_EXCLUDED_IMPORTS) - imported
    )
    assert not set(COMPILE_EXCLUDED_IMPORTS) & set(_compile_module_names())


def test_the_import_scan_sees_each_import_form():
    """Control: the three forms that hide a module from a plain scan."""
    # A constant imported from its own module.
    assert "microcosm.calibrate.geography_constants" in first_party_imports(
        "microcosm.build.ledger_targets"
    )
    # A class imported from a package resolves to the module defining it.
    assert "microcosm.calibrate.registry" in first_party_imports(
        "microcosm.build.ledger_targets"
    )
    # A constant the package __init__ re-exports resolves through the __init__.
    assert "microcosm.diagnostics.schema" in first_party_imports(
        "microcosm.build.transport.artifact_types"
    )
    # An import inside a function body.
    assert "microcosm.build.reference_intervals" in first_party_imports(
        "microcosm.build.reference_profiles"
    )
