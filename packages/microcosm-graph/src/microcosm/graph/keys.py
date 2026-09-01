"""Content identities for graph sources, nodes, and artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .canonical import canonical_json, normative, sha256_domain
from .decl import CompiledGraph, StructuralDelta

__all__ = [
    "artifact_key",
    "frame_key",
    "node_key",
    "seed",
    "source_content_key",
    "weights_key",
]


def _hash_parts(domain: str, *parts: object) -> str:
    return sha256_domain(domain, canonical_json(parts))


def _directory_identity(path: Path) -> tuple[str, int]:
    """Hash a directory as a deterministic sequence of relative file bytes.

    Both built-in source codecs consume directories.  Relative names are
    length-prefixed into this aggregate, so renaming the source directory is
    inert while renaming or changing a file inside it changes the identity.
    ``size`` is the sum of regular-file payload sizes.
    """

    digest = hashlib.sha256(b"microcosm-graph/source-directory/1\0")
    size = 0
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        content = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        size += len(content)
    return digest.hexdigest(), size


def source_content_key(name: str, path: str | Path) -> str:
    """Return the path-independent identity of one named source's content.

    A regular file follows the normative ``sha256(bytes), size`` formula.
    Directories use a deterministic packed representation for the directory-
    based ``frame-store`` and ``csv-tables`` codecs.
    """

    source_path = Path(path)
    if source_path.is_file():
        content = source_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        size = len(content)
    elif source_path.is_dir():
        content_hash, size = _directory_identity(source_path)
    else:
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    return _hash_parts("source", name, content_hash, size)


def artifact_key(node_key: str, entity: str, column: str) -> str:
    """Derive one column artifact identity from its producing node."""

    return _hash_parts("artifact", node_key, entity, column)


def frame_key(node_key: str) -> str:
    """Derive the structural frame artifact identity from its node."""

    return _hash_parts("frame", node_key)


def weights_key(node_key: str, entity: str) -> str:
    """Derive a typed-weight artifact identity outside the column namespace."""

    return _hash_parts("weights", node_key, entity)


def _required_key(keys: Mapping[str, str], node_id: str, consumer: str) -> str:
    try:
        return keys[node_id]
    except KeyError as error:
        raise KeyError(
            f"Node {consumer!r} needs the key of predecessor {node_id!r}; "
            "compute keys in CompiledGraph.order."
        ) from error


def node_key(
    compiled: CompiledGraph,
    node_id: str,
    input_keys: Mapping[str, str],
    kernel_impl_hash: str,
    source_keys: Mapping[str, str],
) -> str:
    """Derive a node key from its declaration and resolved input identities.

    ``input_keys`` maps already-visited node ids to their node keys.  The
    function resolves each declared column exactly as ``compile_graph`` does:
    a local owner supplies its artifact, while a carried column is supplied by
    the current structural version's frame.
    """

    node = compiled.graph.node(node_id)
    input_version: str | None
    if node.structural is StructuralDelta.CREATE:
        input_version = None
    elif node.structural is StructuralDelta.NONE:
        input_version = compiled.versions[node_id]
    else:
        input_version = node.base

    resolved: dict[tuple[str, str], str] = {}
    if input_version is not None:
        for slice_ in node.inputs:
            for column in slice_.columns:
                coordinate = (slice_.entity, column)
                producer = compiled.owners.get(
                    (input_version, slice_.entity, column), input_version
                )
                producer_key = _required_key(input_keys, producer, node_id)
                resolved[coordinate] = artifact_key(producer_key, slice_.entity, column)
    input_artifacts = tuple(
        (entity, column, resolved[(entity, column)])
        for entity, column in sorted(resolved)
    )

    if node.structural is StructuralDelta.NONE:
        version = compiled.versions[node_id]
        population_input = {
            "population": frame_key(_required_key(input_keys, version, node_id))
        }
    elif node.structural is StructuralDelta.CREATE:
        population_input = {}
    else:
        assert node.base is not None
        population_input = {
            "base": frame_key(_required_key(input_keys, node.base, node_id)),
            # A structural transform receives the fully patched base version,
            # not merely the original structural Frame.  compile_graph makes
            # every ordinary member of that version a predecessor; binding
            # their keys prevents an old FILTER/EXPAND/REWEIGHT frame from
            # surviving a changed base patch (including a weight-only node).
            "members": tuple(
                (predecessor, _required_key(input_keys, predecessor, node_id))
                for predecessor in compiled.predecessors[node_id]
                if predecessor != node.base
            ),
        }

    if node.sources:
        resolved_sources = {
            name: source_keys[name]
            for name in sorted(node.sources)
            if name in source_keys
        }
        missing_sources = sorted(set(node.sources) - source_keys.keys())
        if missing_sources:
            joined = ", ".join(repr(name) for name in missing_sources)
            raise KeyError(f"Node {node_id!r} has no content key for source {joined}.")
    else:
        resolved_sources = {}

    return _hash_parts(
        "node",
        normative(node),
        input_artifacts,
        population_input,
        kernel_impl_hash,
        resolved_sources,
    )


def seed(node_key: str) -> int:
    """Derive the node's unsigned 64-bit little-endian RNG seed."""

    digest = hashlib.sha256(b"seed\0" + node_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")
