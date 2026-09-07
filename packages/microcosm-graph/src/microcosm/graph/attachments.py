"""Private, non-portable population attachments and execution lifetimes.

These store references never enter node records or portable manifest identity.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import MappingProxyType
from weakref import WeakValueDictionary

from microcosm.frame import Frame

from .canonical import canonical_json, sha256_domain
from .decl import CompiledGraph, StructuralDelta
from .manifest import NodeReceipt, PopulationView
from .population import MassRecord, Population
from .store import (
    _FRAME_FORMAT,
    ContentStore,
    StoreCorrupt,
    _payload_table,
    _write_frame,
)


@dataclass(frozen=True)
class _StoredPopulation:
    key: str
    node_key: str
    metadata: Mapping[str, object]
    payload_hash: str


class _LazyPopulations(Mapping[str, PopulationView]):
    """Immutable attachment index, with caller-lifetime weak view memoization."""

    def __init__(
        self, store: ContentStore, references: Mapping[str, _StoredPopulation]
    ):
        if any(
            not isinstance(k, str) or type(v) is not _StoredPopulation
            for k, v in references.items()
        ):
            raise TypeError("Lazy populations require validated stored references.")
        self._store = store
        self._references = MappingProxyType(dict(references))
        self._cache: WeakValueDictionary[str, PopulationView] = WeakValueDictionary()
        self._lock = RLock()

    def __iter__(self) -> Iterator[str]:
        return iter(self._references)

    def __contains__(self, version: object) -> bool:
        return version in self._references

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, version: str) -> PopulationView:
        with self._lock:
            view = self._cache.get(version)
            if view is None:
                reference = self._references[version]
                metadata = self._store.metadata(reference.key, kind="frame")
                if _payload_hash(metadata["payloads"]) != reference.payload_hash:
                    raise StoreCorrupt(
                        "Stored population attachment changed after execution."
                    )
                frame = self._store.load_frame(
                    reference.key, node_key=reference.node_key
                )
                # A valid same-key replacement can occur after either the
                # check above or load_frame's own metadata verification. Bind
                # the actual decoded content before exposing or memoizing it.
                if _decoded_payload_hash(self._store, frame) != reference.payload_hash:
                    raise StoreCorrupt(
                        "The decoded population attachment changed after execution."
                    )
                view = PopulationView(frame)
                # The existing Frame store codec omits metadata. Retain the
                # original Frame's already-frozen scalar/container descriptor,
                # which cannot own tables/arrays, for this live manifest only.
                object.__setattr__(view, "_metadata", reference.metadata)
                self._cache[version] = view
            return view


class _PopulationRetention:
    """Retire realized versions after their final scheduled population use."""

    def __init__(
        self, compiled: CompiledGraph, store: ContentStore, *, verify_existing: bool
    ):
        self._compiled = compiled
        self._store = store
        self._verify_existing = verify_existing
        last_use: dict[str, str] = {}
        self._versions: list[str] = []
        for node_id in compiled.order:
            node = compiled.graph.node(node_id)
            if node.structural is not StructuralDelta.NONE:
                self._versions.append(node_id)
                last_use[node_id] = node_id
            if node.structural is StructuralDelta.NONE:
                last_use[compiled.versions[node_id]] = node_id
            elif node.base is not None:
                last_use[node.base] = node_id
        self._retire_at: dict[str, list[str]] = {}
        for version, node_id in last_use.items():
            self._retire_at.setdefault(node_id, []).append(version)
        self._references: dict[str, _StoredPopulation] = {}
        self._ledgers: dict[str, tuple[MassRecord, ...]] = {}
        self._patches: dict[str, list[NodeReceipt]] = {}
        self._force_snapshot: set[str] = set()

    def finish(
        self,
        node_id: str,
        populations: dict[str, Population],
        receipts: Mapping[str, NodeReceipt],
        *,
        structural_collision: bool = False,
    ) -> None:
        node = self._compiled.graph.node(node_id)
        receipt = receipts[node_id]
        if structural_collision:
            self._force_snapshot.add(node_id)
        if node.structural is StructuralDelta.NONE and (
            receipt.artifacts or receipt.weight_key
        ):
            version = self._compiled.versions[node_id]
            self._patches.setdefault(version, []).append(receipt)
        for version in self._retire_at.get(node_id, ()):
            population = populations.pop(version, None)
            if population is None:  # Unreached structural descendants have no Frame.
                continue
            structural = receipts[version]
            key = structural.frame_key
            assert key is not None
            patches = self._patches.pop(version, ())
            # Derive the binding from the actual live Frame even when its
            # structural artifact seems reusable. Otherwise a concurrent
            # replacement could become the expected signature at retirement.
            key, payload_hash = _snapshot(
                self._store,
                population,
                structural,
                patches,
                verify_existing=self._verify_existing,
                reuse_structural=not patches and version not in self._force_snapshot,
            )
            self._force_snapshot.discard(version)
            self._references[version] = _StoredPopulation(
                key, structural.key, population.frame.metadata, payload_hash
            )
            self._ledgers[version] = population.mass_ledger

    def views(self) -> _LazyPopulations:
        return _LazyPopulations(
            self._store,
            {v: self._references[v] for v in self._versions if v in self._references},
        )

    def ledgers(self) -> Mapping[str, tuple[MassRecord, ...]]:
        return {v: self._ledgers[v] for v in self._versions if v in self._ledgers}


def _payload_hash(payloads: object) -> str:
    return sha256_domain("population-attachment-payloads/1", canonical_json(payloads))


def _decoded_payload_hash(store: ContentStore, frame: Frame) -> str:
    """Hash the actual decoded Frame, using its canonical codec representation.

    Temporary writes are bounded by one Frame and use the same per-column
    codec as snapshots. Reading metadata again would leave the same race.
    """

    with TemporaryDirectory(
        prefix="population-read-check-", dir=store.tmp
    ) as directory:
        staging = Path(directory)
        _write_frame(staging, frame)
        return _payload_hash(_payload_table(staging))


def _snapshot(
    store: ContentStore,
    population: Population,
    structural: NodeReceipt,
    patches: Sequence[NodeReceipt],
    *,
    verify_existing: bool,
    reuse_structural: bool = False,
) -> tuple[str, str]:
    """Publish one codec snapshot bound to both provenance and actual bytes.

    A tolerance-bound recomputation can emit different bytes under the same
    node key. Receipt-only attachment identities would silently reuse the
    previous Frame. Stage once, hash bounded payload files, then rename those
    files into the store's normal atomic publication path without copying.
    """

    population.frame.revalidate()
    with TemporaryDirectory(
        prefix="population-attachment-", dir=store.tmp
    ) as directory:
        staging = Path(directory)
        _write_frame(staging, population.frame)
        payload_hash = _payload_hash(_payload_table(staging))
        if reuse_structural:
            assert structural.frame_key is not None
            metadata = store.metadata(structural.frame_key, kind="frame")
            if (
                metadata.get("node_key") == structural.key
                and _payload_hash(metadata["payloads"]) == payload_hash
            ):
                return structural.frame_key, payload_hash
        key = sha256_domain(
            "population-attachment/1",
            canonical_json(
                {
                    "structural": structural._content_payload(),
                    "patches": [patch._content_payload() for patch in patches],
                    "payload_hash": payload_hash,
                }
            ),
        )

        def publish(destination: Path) -> Mapping[str, object]:
            for child in staging.iterdir():
                child.rename(destination / child.name)
            return {"frame_format": _FRAME_FORMAT, "node_key": structural.key}

        store._put(key, "frame", publish, verify_existing=verify_existing)
    return key, payload_hash
