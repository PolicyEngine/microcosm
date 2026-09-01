"""Frozen run provenance records for the graph executor."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

from .canonical import canonical_json, sha256_domain
from .decl import StructuralDelta
from .kernel import Capabilities, Determinism, Numeric, SeedSource
from .population import MassRecord

if TYPE_CHECKING:
    from microcosm.frame import Frame

__all__ = ["Decision", "NodeReceipt", "RunManifest"]

_SCHEMA_VERSION = 1


def _freeze_json(value: object) -> object:
    """Copy JSON-like receipt data into immutable containers."""

    if isinstance(value, Enum):
        return _freeze_json(value.value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("manifest receipt mappings require string keys")
            converted[key] = _freeze_json(child)
        return MappingProxyType(converted)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(child) for child in value)
    raise TypeError(f"manifest cannot serialize {type(value).__name__}")


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


@dataclass(frozen=True)
class Decision:
    """A signed human decision carried as provenance, never as a node input."""

    owner: str
    kind: str
    text: str
    signed_at: str

    def __post_init__(self) -> None:
        for name in ("owner", "kind", "text", "signed_at"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Decision.{name} must be a string")

    def _payload(self) -> dict[str, str]:
        return {
            "owner": self.owner,
            "kind": self.kind,
            "text": self.text,
            "signed_at": self.signed_at,
        }


@dataclass(frozen=True)
class NodeReceipt:
    """The complete operational receipt for one graph node."""

    key: str
    hit: bool
    seed: int
    kernel_ref: str
    kernel_impl_hash: str
    capabilities: Capabilities
    receipt: Mapping[str, object] = field(default_factory=dict)
    artifacts: Mapping[tuple[str, str], str] = field(default_factory=dict)
    wall_time: float = 0.0
    frame_key: str | None = None
    weight_key: str | None = None
    opaque_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError("NodeReceipt.key must be a string")
        if not isinstance(self.hit, bool):
            raise TypeError("NodeReceipt.hit must be a bool")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("NodeReceipt.seed must be an int")
        if not isinstance(self.kernel_ref, str):
            raise TypeError("NodeReceipt.kernel_ref must be a string")
        if not isinstance(self.kernel_impl_hash, str):
            raise TypeError("NodeReceipt.kernel_impl_hash must be a string")
        if not isinstance(self.capabilities, Capabilities):
            raise TypeError("NodeReceipt.capabilities must be Capabilities")
        frozen_receipt = _freeze_json(self.receipt)
        if not isinstance(frozen_receipt, Mapping):
            raise TypeError("NodeReceipt.receipt must be a mapping")
        object.__setattr__(self, "receipt", frozen_receipt)

        artifacts: dict[tuple[str, str], str] = {}
        for coordinate, key in self.artifacts.items():
            if (
                not isinstance(coordinate, tuple)
                or len(coordinate) != 2
                or not all(isinstance(part, str) for part in coordinate)
            ):
                raise TypeError(
                    "NodeReceipt.artifacts keys must be (entity, column) tuples"
                )
            if not isinstance(key, str):
                raise TypeError("NodeReceipt.artifacts values must be strings")
            artifacts[coordinate] = key
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

        for name in ("frame_key", "weight_key"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"NodeReceipt.{name} must be a string or None")

        opaque_artifacts: dict[str, str] = {}
        for name, key in self.opaque_artifacts.items():
            if not isinstance(name, str):
                raise TypeError("NodeReceipt.opaque_artifacts keys must be strings")
            if not isinstance(key, str):
                raise TypeError("NodeReceipt.opaque_artifacts values must be strings")
            opaque_artifacts[name] = key
        object.__setattr__(self, "opaque_artifacts", MappingProxyType(opaque_artifacts))

        wall_time = float(self.wall_time)
        if not math.isfinite(wall_time) or wall_time < 0:
            raise ValueError("NodeReceipt.wall_time must be finite and non-negative")
        object.__setattr__(self, "wall_time", wall_time)

    @property
    def node_key(self) -> str:
        """Alias spelling used by identity-oriented callers."""

        return self.key

    @property
    def store_hit(self) -> bool:
        """Alias spelling used by store-oriented callers."""

        return self.hit

    @property
    def implementation_hash(self) -> str:
        """The implementation digest under its concise spelling."""

        return self.kernel_impl_hash

    @property
    def artifact_keys(self) -> Mapping[tuple[str, str], str]:
        """Backward-compatible descriptive alias for :attr:`artifacts`."""

        return self.artifacts

    @property
    def wall_time_s(self) -> float:
        return self.wall_time

    def _payload(self) -> dict[str, object]:
        capabilities = self.capabilities
        return {
            "key": self.key,
            "hit": self.hit,
            "seed": self.seed,
            "kernel_ref": self.kernel_ref,
            "kernel_impl_hash": self.kernel_impl_hash,
            "capabilities": {
                "determinism": _enum_value(capabilities.determinism),
                "numeric": _enum_value(capabilities.numeric),
                "seed_source": _enum_value(capabilities.seed_source),
                "structural": _enum_value(capabilities.structural),
                "consumes_se": capabilities.consumes_se,
                "dependencies": capabilities.dependencies,
            },
            "receipt": self.receipt,
            "artifacts": tuple(
                {"entity": entity, "column": column, "key": key}
                for (entity, column), key in sorted(self.artifacts.items())
            ),
            "frame_key": self.frame_key,
            "weight_key": self.weight_key,
            "opaque_artifacts": self.opaque_artifacts,
            "wall_time": self.wall_time,
        }


@dataclass(frozen=True)
class RunManifest:
    """One run's provenance plus its attached, non-serialized populations.

    Only sorted node keys and signed decisions form :attr:`key`.  Store hits,
    timings, host, timestamps, receipts, and attached ``Frame`` instances are
    run-level observations and cannot invalidate computational reuse.
    """

    country: str
    nodes: Mapping[str, NodeReceipt]
    decisions: tuple[Decision, ...] = ()
    started_at: str = ""
    finished_at: str = ""
    host: str = ""
    populations: Mapping[str, Frame] = field(
        default_factory=dict, repr=False, compare=False
    )
    mass_ledgers: Mapping[str, tuple[MassRecord, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.country, str):
            raise TypeError("RunManifest.country must be a string")
        nodes: dict[str, NodeReceipt] = {}
        for node_id, receipt in self.nodes.items():
            if not isinstance(node_id, str):
                raise TypeError("RunManifest.nodes keys must be strings")
            if not isinstance(receipt, NodeReceipt):
                raise TypeError("RunManifest.nodes values must be NodeReceipt")
            nodes[node_id] = receipt
        object.__setattr__(self, "nodes", MappingProxyType(nodes))

        decisions = tuple(self.decisions)
        if not all(isinstance(decision, Decision) for decision in decisions):
            raise TypeError("RunManifest.decisions must contain Decision values")
        object.__setattr__(self, "decisions", decisions)

        for name in ("started_at", "finished_at", "host"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"RunManifest.{name} must be a string")
        object.__setattr__(
            self, "populations", MappingProxyType(dict(self.populations))
        )
        mass_ledgers: dict[str, tuple[MassRecord, ...]] = {}
        for version_id, records in self.mass_ledgers.items():
            if not isinstance(version_id, str):
                raise TypeError("RunManifest.mass_ledgers keys must be strings")
            frozen_records = tuple(records)
            if not all(isinstance(record, MassRecord) for record in frozen_records):
                raise TypeError(
                    "RunManifest.mass_ledgers values must contain MassRecord values"
                )
            mass_ledgers[version_id] = frozen_records
        object.__setattr__(self, "mass_ledgers", MappingProxyType(mass_ledgers))

    @property
    def content_addressed(self) -> Mapping[str, object]:
        """The exact projection hashed by :attr:`key`."""

        decisions = sorted(
            (decision._payload() for decision in self.decisions),
            key=canonical_json,
        )
        return MappingProxyType(
            {
                "node_keys": tuple(sorted(item.key for item in self.nodes.values())),
                "decisions": tuple(decisions),
            }
        )

    @property
    def key(self) -> str:
        return sha256_domain("manifest", canonical_json(self.content_addressed))

    @property
    def receipts(self) -> Mapping[str, NodeReceipt]:
        return self.nodes

    def node(self, node_id: str) -> NodeReceipt:
        """Look up one node receipt with a useful missing-node error."""

        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise KeyError(f"Manifest has no receipt for node {node_id!r}.") from error

    def receipt(self, node_id: str) -> NodeReceipt:
        return self.node(node_id)

    def population(self, version_id: str) -> Frame:
        """Return an attached final population version.

        Population frames are deliberately not serialized in manifest JSON;
        a manifest restored from JSON therefore raises this explicit error
        until a loader attaches them again.
        """

        try:
            return self.populations[version_id]
        except KeyError as error:
            raise KeyError(
                f"Population {version_id!r} is not attached to this manifest."
            ) from error

    def mass_ledger(self, version_id: str) -> tuple[MassRecord, ...]:
        """Return the transient mass audit trail for one attached version."""

        try:
            return self.mass_ledgers[version_id]
        except KeyError as error:
            raise KeyError(
                f"Mass ledger {version_id!r} is not attached to this manifest."
            ) from error

    def __getitem__(self, node_id: str) -> NodeReceipt:
        return self.node(node_id)

    def to_json(self) -> str:
        """Serialize the complete portable provenance as canonical JSON."""

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "key": self.key,
            "country": self.country,
            "nodes": {
                node_id: receipt._payload() for node_id, receipt in self.nodes.items()
            },
            "decisions": tuple(decision._payload() for decision in self.decisions),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "host": self.host,
        }
        return canonical_json(payload).decode("utf-8")

    def to_json_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Restore a manifest and verify its serialized content key."""

        if isinstance(value, bytes | bytearray):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise TypeError("RunManifest.from_json expects str or bytes")
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError("manifest JSON must contain an object")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema version {raw.get('schema_version')!r}"
            )

        nodes_raw = raw.get("nodes")
        if not isinstance(nodes_raw, dict):
            raise ValueError("manifest nodes must be an object")
        nodes = {
            str(node_id): _node_receipt_from_payload(payload)
            for node_id, payload in nodes_raw.items()
        }
        decisions_raw = raw.get("decisions", [])
        if not isinstance(decisions_raw, list):
            raise ValueError("manifest decisions must be an array")
        decisions = tuple(_decision_from_payload(item) for item in decisions_raw)
        manifest = cls(
            country=_string_field(raw, "country"),
            nodes=nodes,
            decisions=decisions,
            started_at=_string_field(raw, "started_at"),
            finished_at=_string_field(raw, "finished_at"),
            host=_string_field(raw, "host"),
        )
        serialized_key = raw.get("key")
        if serialized_key != manifest.key:
            raise ValueError(
                "manifest content key mismatch: serialized provenance was altered"
            )
        return manifest


def _string_field(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"manifest field {name!r} must be a string")
    return value


def _decision_from_payload(value: object) -> Decision:
    if not isinstance(value, Mapping):
        raise ValueError("manifest decisions must contain objects")
    return Decision(
        owner=_string_field(value, "owner"),
        kind=_string_field(value, "kind"),
        text=_string_field(value, "text"),
        signed_at=_string_field(value, "signed_at"),
    )


def _capabilities_from_payload(value: object) -> Capabilities:
    if not isinstance(value, Mapping):
        raise ValueError("node capabilities must be an object")
    consumes_se = value.get("consumes_se")
    dependencies = value.get("dependencies")
    if not isinstance(consumes_se, bool):
        raise ValueError("capabilities.consumes_se must be a bool")
    if not isinstance(dependencies, list):
        raise ValueError("capabilities.dependencies must be an array")
    if not all(isinstance(item, str) for item in dependencies):
        raise ValueError("capabilities.dependencies must contain strings")
    return Capabilities(
        determinism=Determinism(_string_field(value, "determinism")),
        numeric=Numeric(_string_field(value, "numeric")),
        seed_source=SeedSource(_string_field(value, "seed_source")),
        structural=StructuralDelta(_string_field(value, "structural")),
        consumes_se=consumes_se,
        dependencies=tuple(dependencies),
    )


def _node_receipt_from_payload(value: object) -> NodeReceipt:
    if not isinstance(value, Mapping):
        raise ValueError("manifest node receipts must be objects")
    hit = value.get("hit")
    seed_value = value.get("seed")
    wall_time = value.get("wall_time")
    receipt = value.get("receipt")
    artifacts_raw = value.get("artifacts")
    frame_key = value.get("frame_key")
    weight_key = value.get("weight_key")
    opaque_artifacts = value.get("opaque_artifacts", {})
    if not isinstance(hit, bool):
        raise ValueError("node receipt hit must be a bool")
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        raise ValueError("node receipt seed must be an int")
    if not isinstance(wall_time, int | float) or isinstance(wall_time, bool):
        raise ValueError("node receipt wall_time must be numeric")
    if not isinstance(receipt, Mapping):
        raise ValueError("node receipt receipt must be an object")
    if not isinstance(artifacts_raw, list):
        raise ValueError("node receipt artifacts must be an array")
    if frame_key is not None and not isinstance(frame_key, str):
        raise ValueError("node receipt frame_key must be a string or null")
    if weight_key is not None and not isinstance(weight_key, str):
        raise ValueError("node receipt weight_key must be a string or null")
    if not isinstance(opaque_artifacts, Mapping):
        raise ValueError("node receipt opaque_artifacts must be an object")
    if not all(
        isinstance(name, str) and isinstance(key, str)
        for name, key in opaque_artifacts.items()
    ):
        raise ValueError("node receipt opaque_artifacts must map strings to strings")
    artifacts: dict[tuple[str, str], str] = {}
    for item in artifacts_raw:
        if not isinstance(item, Mapping):
            raise ValueError("node receipt artifacts must contain objects")
        coordinate = (
            _string_field(item, "entity"),
            _string_field(item, "column"),
        )
        if coordinate in artifacts:
            raise ValueError(f"node receipt repeats artifact {coordinate!r}")
        artifacts[coordinate] = _string_field(item, "key")
    return NodeReceipt(
        key=_string_field(value, "key"),
        hit=hit,
        seed=seed_value,
        kernel_ref=_string_field(value, "kernel_ref"),
        kernel_impl_hash=_string_field(value, "kernel_impl_hash"),
        capabilities=_capabilities_from_payload(value.get("capabilities")),
        receipt=receipt,
        artifacts=artifacts,
        wall_time=float(wall_time),
        frame_key=frame_key,
        weight_key=weight_key,
        opaque_artifacts=opaque_artifacts,
    )
