"""The US congressional-district published-benchmark contract, as executable rules.

This package implements protocol ``us-cd-published-benchmark-v4`` (SHA-256
``f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea``) under an
approval whose scope is **invented-fixture implementation only**. Candidate
scoring and production ledger registration are explicitly not approved, and
nothing here performs either: no candidate, population, ACS or Census artifact
is read, no ledger file is created or appended, and no release, promotion or
publication decision is produced.

The approved document itself ships as a packaged resource beside
:mod:`~microcosm.build.cd_benchmark.protocol` and is hash-verified on load. Every
threshold, vocabulary and payload roster is compiled out of those bytes rather
than retyped, so the code cannot drift from the contract without a test failing.
Paths named inside the document are locators recording what was inspected; this
package never opens them.

Import cost: the modules below are pure and depend only on the standard library
and this package. The two source adapters the contract pins are reached through
:mod:`~microcosm.build.cd_benchmark.universe`, which defers those imports to call
time, so importing this package does not load ``microcosm.build.us_runtime`` and
does not load PolicyEngine.

Layout:

* :mod:`~microcosm.build.cd_benchmark.canonical` — canonical JSON and digests.
* :mod:`~microcosm.build.cd_benchmark.reasons` — the closed vocabularies.
* :mod:`~microcosm.build.cd_benchmark.protocol` — compile the approved document.
* :mod:`~microcosm.build.cd_benchmark.metrics` — point and categorical checks.
* :mod:`~microcosm.build.cd_benchmark.reference` — reference validity and precision.
* :mod:`~microcosm.build.cd_benchmark.support` — origin support and concentration.
* :mod:`~microcosm.build.cd_benchmark.origin` — typed original-source origin keys.
* :mod:`~microcosm.build.cd_benchmark.approval` — injected approval and registration.
* :mod:`~microcosm.build.cd_benchmark.ledger` — in-memory ledger records and states.
* :mod:`~microcosm.build.cd_benchmark.binding` — candidate binding slots.
* :mod:`~microcosm.build.cd_benchmark.universe` — the pinned source adapters.
* :mod:`~microcosm.build.cd_benchmark.evaluation` — the six axes and one reduction.
"""

from __future__ import annotations

from microcosm.build.cd_benchmark.protocol import (
    APPROVED_PROTOCOL_ID,
    APPROVED_PROTOCOL_SHA256,
    CompiledProtocol,
    ProtocolError,
    canonical_protocol,
    compile_protocol,
)
from microcosm.build.cd_benchmark.reasons import (
    AXES,
    BenchmarkVerdict,
    ExposureStatus,
    MappingStatus,
    PrecisionStatus,
    Reason,
    SupportVerdict,
)

__all__ = [
    "APPROVED_PROTOCOL_ID",
    "APPROVED_PROTOCOL_SHA256",
    "AXES",
    "BenchmarkVerdict",
    "CompiledProtocol",
    "ExposureStatus",
    "MappingStatus",
    "PrecisionStatus",
    "ProtocolError",
    "Reason",
    "SupportVerdict",
    "canonical_protocol",
    "compile_protocol",
]
