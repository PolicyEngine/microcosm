"""The typed artifact contracts the transport kernels exchange.

Each constant names the nominal type and payload version an
``ArtifactOutput``/``ArtifactInput`` declares (``microcosm.graph.decl``,
amendment 19). The graph carries the name and version only; every consumer
here validates the bytes itself before using them. The calibration problem,
solution and result types are the shared ones in
:mod:`microcosm.calibrate.artifacts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from microcosm.calibrate.artifacts import PROBLEM_TYPE
from microcosm.diagnostics import CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
from microcosm.graph import ArtifactType

__all__ = [
    "KERNEL_OUTPUTS",
    "COMPARISON_TYPE",
    "DIAGNOSTICS_TYPE",
    "EXPORT_DESCRIPTOR_TYPE",
    "EXPORT_READBACK_TYPE",
    "GATE_REPORT_TYPE",
    "PACKAGE_RECEIPT_TYPE",
    "TARGET_SURFACE_TYPE",
]

#: A compiled target registry plus its reference trace (canonical JSON).
TARGET_SURFACE_TYPE = ArtifactType("microcosm.targets.surface", 1)
#: Hold-out comparators measured on a population (canonical JSON).
COMPARISON_TYPE = ArtifactType("microcosm.targets.comparison", 1)
#: One calibration's ``microcosm-diagnostics`` model, validated, as canonical
#: JSON; the type version is the diagnostics schema version.
DIAGNOSTICS_TYPE = ArtifactType(
    "microcosm.diagnostics.calibration", CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
)
#: One gate-battery phase's outcomes and enforcement (canonical JSON).
GATE_REPORT_TYPE = ArtifactType("microcosm.gates.phase-report", 1)
#: The exact entity tables an export must write (canonical JSON).
EXPORT_DESCRIPTOR_TYPE = ArtifactType("microcosm.transport.export-descriptor", 1)
#: The written dataset compared with its descriptor (canonical JSON).
EXPORT_READBACK_TYPE = ArtifactType("microcosm.transport.export-readback", 1)
#: The single object a build's reviewers read (canonical JSON).
PACKAGE_RECEIPT_TYPE = ArtifactType("microcosm.transport.package-receipt", 1)

#: Kernel ref -> the typed outputs its node must declare, exactly. Each kernel
#: refuses a node that declares a different name or type, so a composer that
#: builds its ``ArtifactOutput`` tuples from this table cannot drift.
KERNEL_OUTPUTS: Mapping[str, Mapping[str, ArtifactType]] = MappingProxyType(
    {
        "targets.compile@1": MappingProxyType({"surface": TARGET_SURFACE_TYPE}),
        "targets.problem@1": MappingProxyType({"problem": PROBLEM_TYPE}),
        "takeup.compare@1": MappingProxyType({"comparison": COMPARISON_TYPE}),
        "diagnostics.calibration@1": MappingProxyType(
            {"diagnostics": DIAGNOSTICS_TYPE}
        ),
        "gates.battery@1": MappingProxyType({"gate_report": GATE_REPORT_TYPE}),
        "export.prepare@1": MappingProxyType(
            {"export_descriptor": EXPORT_DESCRIPTOR_TYPE}
        ),
        "export.readback@1": MappingProxyType(
            {"export_readback": EXPORT_READBACK_TYPE}
        ),
        "transport.package@1": MappingProxyType({"receipt": PACKAGE_RECEIPT_TYPE}),
    }
)
