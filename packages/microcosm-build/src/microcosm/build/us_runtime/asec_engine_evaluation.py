"""Closure-admitted PolicyEngine-US evaluation of the corrected money leaves.

An output is admitted only when its complete static input closure is produced by
this graph. The defaults allowlist is empty by decision, so a closure leaf this
slice does not map blocks its root outright rather than being zero-filled or
split into an unreviewed surface. Each admitted root carries a pinned closure
identity; engine or closure drift refuses before any simulation runs.

Outputs are formula-owned. They are retained in an evaluation artifact and are
never written back as population cells.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph import ArtifactType
from microcosm.graph.canonical import canonical_json

from .operator_boundary import PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES

US_ASEC_ENGINE_EVALUATION_TYPE = ArtifactType("microcosm.us.asec_engine_evaluation", 1)
ENGINE_PACKAGE = "policyengine-us"
ENGINE_DEFAULTS_RESOURCE = "asec_current_money_engine_defaults_v1.json"
ENGINE_DEFAULTS_SHA256 = (
    "090dd200ecdb837127dba1d1f22903d152cef8a434085e4ead4d379350ee5b82"
)
ENGINE_DEFAULTS_ARTIFACT_KIND = "microcosm.us.asec_current_money_engine_defaults.v1"
EVALUATION_MAGIC = b"MCASECEV\x01"
EVALUATION_ARTIFACT_KIND = "microcosm.us.asec_engine_evaluation.v1"
EVALUATION_PERIOD = 2024
EVALUATION_HEADER_MAX_BYTES = 64 * 1024
EVALUATION_MAX_ROWS = 1_000_000
_PRIOR_YEAR_MARKERS = ("last_year", "previous_year", "prior_year")


class EngineAdmissionError(ValueError):
    """A candidate output is not admitted, or a pinned identity drifted."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise EngineAdmissionError(reason)


@dataclass(frozen=True)
class EngineOutputContract:
    """Pinned identity of one admitted root's static PolicyEngine-US closure."""

    root: str
    entity: str
    engine_version: str
    input_leaves: tuple[str, ...]
    formula_node_count: int
    edge_count: int
    sha256: str


ADMITTED_ENGINE_OUTPUT_CONTRACTS: tuple[EngineOutputContract, ...] = (
    EngineOutputContract(
        root="capital_gains",
        entity="person",
        engine_version="1.819.0",
        input_leaves=(
            "long_term_capital_gains_before_response",
            "short_term_capital_gains",
        ),
        formula_node_count=4,
        edge_count=6,
        sha256="1768cb833b183b3898567910b463e645a2744c2138688c50b5fc1cc17c3319df",
    ),
    EngineOutputContract(
        root="dividend_income",
        entity="person",
        engine_version="1.819.0",
        input_leaves=("non_qualified_dividend_income", "qualified_dividend_income"),
        formula_node_count=2,
        edge_count=3,
        sha256="a27f07536f4909361de7444831f7852a3f976e12d40d6afe4059c6a5fbdfbbe3",
    ),
    EngineOutputContract(
        root="employment_income",
        entity="person",
        engine_version="1.819.0",
        input_leaves=("employment_income_before_lsr",),
        formula_node_count=3,
        edge_count=4,
        sha256="bdcd8da23a79ff228a610a4d0b483f5fd2068d968f31e96c07cea8a86cc3366b",
    ),
    EngineOutputContract(
        root="social_security",
        entity="person",
        engine_version="1.819.0",
        input_leaves=(
            "social_security_dependents",
            "social_security_disability",
            "social_security_retirement",
            "social_security_survivors",
        ),
        formula_node_count=1,
        edge_count=4,
        sha256="3f5a5304ac1e8c8d785614069867590176325b212a0a080457f80594a6fd53c0",
    ),
)
BLOCKED_ENGINE_OUTPUTS: tuple[str, ...] = (
    "interest_income",
    "pension_income",
    "self_employment_income",
)
ADMITTED_ROOTS: tuple[str, ...] = tuple(
    contract.root for contract in ADMITTED_ENGINE_OUTPUT_CONTRACTS
)


def load_engine_defaults() -> dict:
    """Read and pin-verify the defaults resource; the allowlist must stay empty."""
    payload = (
        resources.files(__package__).joinpath(ENGINE_DEFAULTS_RESOURCE).read_bytes()
    )
    _require(
        sha256(payload).hexdigest() == ENGINE_DEFAULTS_SHA256,
        "ENGINE_DEFAULTS_FINGERPRINT",
    )
    import json

    document = json.loads(payload)
    _require(
        document["schema_version"] == 1
        and document["artifact_kind"] == ENGINE_DEFAULTS_ARTIFACT_KIND,
        "ENGINE_DEFAULTS_SCHEMA",
    )
    _require(document["allowlist"] == [], "ENGINE_DEFAULTS_ALLOWLIST")
    _require(
        tuple(sorted(document["admitted_roots"])) == ADMITTED_ROOTS,
        "ENGINE_DEFAULTS_ROOTS",
    )
    _require(
        tuple(sorted(document["blocked_roots"])) == BLOCKED_ENGINE_OUTPUTS,
        "ENGINE_DEFAULTS_BLOCKED",
    )
    _require(document["engine_package"] == ENGINE_PACKAGE, "ENGINE_DEFAULTS_PACKAGE")
    _require(document["release_eligible"] is False, "ENGINE_DEFAULTS_RELEASE_CLAIM")
    return document


def _runtime_package_identity(package: str, module: str) -> dict:
    """Hash installed model code and parameter files, never population data.

    Include relative names so additions, removals and relocation have explicit
    semantics. Recompute on every invocation: cache admission must see an edit
    made after a previous evaluation in the same process.
    """
    installed = distribution(package)
    root = Path(installed.locate_file(module))
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".py"
            or (
                "parameters" in path.relative_to(root).parts
                and path.suffix in (".yaml", ".yml", ".json")
            )
        )
    )
    records = [
        [path.relative_to(root).as_posix(), sha256(path.read_bytes()).hexdigest()]
        for path in files
    ]
    return {
        "version": installed.version,
        "files": len(records),
        "source_and_parameters_sha256": sha256(canonical_json(records)).hexdigest(),
    }


def _verify_runtime_origin(package: str, module: str) -> None:
    """The adapter must import the same distribution whose bytes are pinned."""
    root = Path(distribution(package).locate_file(module)).resolve()
    spec = find_spec(module)
    _require(
        spec is not None
        and spec.origin is not None
        and Path(spec.origin).resolve() == root / "__init__.py"
        and spec.submodule_search_locations is not None
        and [Path(path).resolve() for path in spec.submodule_search_locations]
        == [root],
        "ENGINE_RUNTIME_ORIGIN",
    )
    for name, loaded in tuple(sys.modules.items()):
        if name != module and not name.startswith(module + "."):
            continue
        filename = getattr(loaded, "__file__", None)
        if filename is not None:
            _require(
                Path(filename).resolve().is_relative_to(root), "ENGINE_RUNTIME_ORIGIN"
            )
        for path in getattr(loaded, "__path__", ()):
            _require(Path(path).resolve().is_relative_to(root), "ENGINE_RUNTIME_ORIGIN")


def engine_runtime_identity() -> dict:
    """Verify the pinned baseline runtime before execution or a cache lookup."""
    document = load_engine_defaults()
    for package, module in (
        ("policyengine-us", "policyengine_us"),
        ("policyengine-core", "policyengine_core"),
    ):
        _verify_runtime_origin(package, module)
    observed = {
        package: _runtime_package_identity(package, module)
        for package, module in (
            ("policyengine-us", "policyengine_us"),
            ("policyengine-core", "policyengine_core"),
        )
    }
    _require(observed == document["baseline_runtime"], "ENGINE_RUNTIME_DRIFT")
    return {"policy": "installed_baseline_no_reforms", "packages": observed}


def _prior_year_leaves() -> frozenset[str]:
    family = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["prior_year_income"]
    return frozenset(name for columns in family.values() for name in columns)


def _closure(index, root: str):
    try:
        return index.variable_dependency_closure(root)
    except ValueError as error:
        raise EngineAdmissionError(f"UNKNOWN_ENGINE_ROOT:{root}") from error


def _check_contract(closure, contract: EngineOutputContract) -> None:
    observed = {
        "engine_version": closure.engine_version,
        "root": closure.root,
        "input_leaves": tuple(closure.input_leaves),
        "formula_node_count": len(closure.formula_nodes),
        "edge_count": len(closure.edges),
        "sha256": closure.sha256,
    }
    expected = {
        "engine_version": contract.engine_version,
        "root": contract.root,
        "input_leaves": contract.input_leaves,
        "formula_node_count": contract.formula_node_count,
        "edge_count": contract.edge_count,
        "sha256": contract.sha256,
    }
    if observed != expected:
        raise EngineAdmissionError(
            "PolicyEngine-US closure drifted for "
            f"{contract.root!r}; expected={expected}, observed={observed}."
        )


def admit_engine_outputs(
    index,
    *,
    produced_leaves,
    defaults: dict | None = None,
) -> tuple[EngineOutputContract, ...]:
    """Admit only roots whose complete closure this graph produces."""
    document = load_engine_defaults() if defaults is None else defaults
    _require(document["allowlist"] == [], "ENGINE_DEFAULTS_ALLOWLIST")
    produced = frozenset(produced_leaves)
    prior_year = _prior_year_leaves()
    for contract in ADMITTED_ENGINE_OUTPUT_CONTRACTS:
        closure = _closure(index, contract.root)
        _check_contract(closure, contract)
        metadata = index.variable_metadata(contract.root)
        _require(metadata.entity == contract.entity, "ADMITTED_ROOT_ENTITY")
        missing = sorted(set(closure.input_leaves) - produced)
        if missing:
            raise EngineAdmissionError(
                f"Root {contract.root!r} has unproduced closure leaves {missing}; "
                "the engine-default allowlist is empty by decision."
            )
        reaching = sorted(set(closure.input_leaves) & prior_year)
        _require(not reaching, f"PRIOR_YEAR_LEAF_IN_CLOSURE:{contract.root}")
        _require(
            not any(
                marker in leaf
                for leaf in closure.input_leaves
                for marker in _PRIOR_YEAR_MARKERS
            ),
            f"PRIOR_YEAR_LEAF_IN_CLOSURE:{contract.root}",
        )
    for root in BLOCKED_ENGINE_OUTPUTS:
        closure = _closure(index, root)
        blocked = sorted(set(closure.input_leaves) - produced)
        _require(bool(blocked), f"BLOCKED_ROOT_NOW_COMPLETE:{root}")
    return ADMITTED_ENGINE_OUTPUT_CONTRACTS


@dataclass(frozen=True)
class EngineEvaluation:
    """Engine outputs retained as evidence; never population cells."""

    header: dict
    values: dict[str, np.ndarray]

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))


def _aggregates(values: np.ndarray) -> dict[str, object]:
    return {
        "rows": int(values.size),
        "nonzero_rows": int(np.count_nonzero(values)),
        "unweighted_total": float(values.sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def materialize_engine_outputs(
    frame: Frame,
    engine,
    contracts: tuple[EngineOutputContract, ...],
    *,
    period: int = EVALUATION_PERIOD,
    context_bindings: dict | None = None,
) -> EngineEvaluation:
    """Run the real adapter for the admitted roots and retain the arrays only."""
    _require(type(frame) is Frame and frame.schema == US_SCHEMA, "ENGINE_FRAME_SCHEMA")
    _require(getattr(engine, "country", None) == "us", "ENGINE_COUNTRY")
    _require(period == EVALUATION_PERIOD, "ENGINE_PERIOD")
    _require(bool(contracts), "ENGINE_NO_ADMITTED_OUTPUTS")
    _require(contracts == ADMITTED_ENGINE_OUTPUT_CONTRACTS, "ENGINE_CONTRACT_ROSTER")
    runtime = engine_runtime_identity()
    rows = frame.n("person")
    _require(0 < rows <= EVALUATION_MAX_ROWS, "ENGINE_ROW_BOUND")
    roots = tuple(sorted(contract.root for contract in contracts))
    _require(len(set(roots)) == len(roots), "ENGINE_DUPLICATE_ROOT")
    results = engine.materialize(frame, list(roots), period)
    _require(set(results) == set(roots), "ENGINE_RESULT_ROSTER")
    values: dict[str, np.ndarray] = {}
    for root in roots:
        array = np.asarray(results[root], dtype="float64")
        _require(array.shape == (rows,), "ENGINE_RESULT_SHAPE")
        _require(bool(np.isfinite(array).all()), "ENGINE_RESULT_NONFINITE")
        values[root] = np.ascontiguousarray(array, dtype="<f8")
    header = {
        "schema_version": 1,
        "artifact_kind": EVALUATION_ARTIFACT_KIND,
        "release_eligible": False,
        "representative": False,
        "claim": "engineering_evidence_only_no_tax_or_benefit_score",
        "engine_package": ENGINE_PACKAGE,
        "engine_version": contracts[0].engine_version,
        "period": period,
        "baseline_runtime": runtime,
        "person_rows": rows,
        "roots": list(roots),
        "closures": {
            contract.root: {
                "entity": contract.entity,
                "engine_version": contract.engine_version,
                "input_leaves": list(contract.input_leaves),
                "formula_node_count": contract.formula_node_count,
                "edge_count": contract.edge_count,
                "sha256": contract.sha256,
            }
            for contract in sorted(contracts, key=lambda item: item.root)
        },
        "recorded_defaults": [],
        "defaults_resource_sha256": ENGINE_DEFAULTS_SHA256,
        "blocked_roots": list(BLOCKED_ENGINE_OUTPUTS),
        "aggregates": {root: _aggregates(values[root]) for root in roots},
        "bindings": dict(context_bindings or {}),
    }
    return EngineEvaluation(header, values)


def _evaluation_parts(evaluation: EngineEvaluation) -> tuple[bytes, ...]:
    _require(type(evaluation) is EngineEvaluation, "TYPED_EVALUATION_REQUIRED")
    header = canonical_json(evaluation.header)
    _require(0 < len(header) <= EVALUATION_HEADER_MAX_BYTES, "HEADER_SIZE")
    roots = tuple(evaluation.header["roots"])
    _require(roots == tuple(sorted(evaluation.values)), "ENGINE_RESULT_ROSTER")
    rows = evaluation.header["person_rows"]
    parts = [EVALUATION_MAGIC, struct.pack("<I", len(header)), header]
    for root in roots:
        array = evaluation.values[root]
        _require(
            array.dtype == np.dtype("<f8") and array.shape == (rows,),
            "ENGINE_RESULT_SHAPE",
        )
        parts.append(array.tobytes())
    return tuple(parts)


def encode_engine_evaluation(evaluation: EngineEvaluation) -> bytes:
    """Encode the evaluation header and arrays with a transport checksum."""
    parts = _evaluation_parts(evaluation)
    checksum = sha256()
    for part in parts:
        checksum.update(part)
    return b"".join((*parts, checksum.digest()))


def decode_engine_evaluation(payload: bytes) -> EngineEvaluation:
    """Decode a stored evaluation, verifying framing, checksum and shapes."""
    _require(
        type(payload) is bytes and len(payload) > len(EVALUATION_MAGIC) + 4 + 32,
        "PAYLOAD_SIZE",
    )
    _require(payload.startswith(EVALUATION_MAGIC), "PAYLOAD_VERSION")
    size = struct.unpack_from("<I", payload, len(EVALUATION_MAGIC))[0]
    start = len(EVALUATION_MAGIC) + 4
    _require(
        0 < size <= EVALUATION_HEADER_MAX_BYTES and start + size <= len(payload) - 32,
        "HEADER_SIZE",
    )
    import json

    header = json.loads(payload[start : start + size])
    _require(
        canonical_json(header) == payload[start : start + size], "HEADER_CANONICAL"
    )
    _require(header["artifact_kind"] == EVALUATION_ARTIFACT_KIND, "PAYLOAD_VERSION")
    roots = tuple(header["roots"])
    rows = header["person_rows"]
    _require(type(rows) is int and 0 < rows <= EVALUATION_MAX_ROWS, "ENGINE_ROW_BOUND")
    _require(
        len(payload) == start + size + 8 * rows * len(roots) + 32, "PAYLOAD_LENGTH"
    )
    _require(
        sha256(memoryview(payload)[:-32]).digest() == payload[-32:], "PAYLOAD_CHECKSUM"
    )
    cursor = start + size
    values = {}
    for root in roots:
        width = 8 * rows
        values[root] = np.frombuffer(payload[cursor : cursor + width], dtype="<f8")
        cursor += width
    return EngineEvaluation(header, values)


__all__ = [
    "ADMITTED_ENGINE_OUTPUT_CONTRACTS",
    "ADMITTED_ROOTS",
    "BLOCKED_ENGINE_OUTPUTS",
    "ENGINE_DEFAULTS_ARTIFACT_KIND",
    "ENGINE_DEFAULTS_RESOURCE",
    "ENGINE_DEFAULTS_SHA256",
    "EVALUATION_ARTIFACT_KIND",
    "EVALUATION_MAGIC",
    "EVALUATION_PERIOD",
    "EngineAdmissionError",
    "EngineEvaluation",
    "EngineOutputContract",
    "US_ASEC_ENGINE_EVALUATION_TYPE",
    "admit_engine_outputs",
    "decode_engine_evaluation",
    "encode_engine_evaluation",
    "engine_runtime_identity",
    "load_engine_defaults",
    "materialize_engine_outputs",
]
