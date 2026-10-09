"""Explicit codecs for donor populations, Chronicle facts and RuleSpec trees.

The graph registry supports Frame and immutable-byte loaders. The donor codec
therefore returns a donor-scale concept Frame; CREATE installs the destination
schema and mass itself. Facts are validated by the shared Chronicle loader and
returned as their original JSONL bytes. RuleSpec trees have a canonical,
root-path-independent byte envelope containing every relative file and byte.
Importing this module does not mutate the shared codec registry.
"""

from __future__ import annotations

import base64
from hashlib import file_digest
from pathlib import Path

import pandas as pd

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.transport import read_populace_us_donor
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SOURCE_CODECS, SourceCodecRegistry
from microcosm.graph.store import ContentStore

__all__ = [
    "DONOR_SOURCE_CODEC",
    "FACTS_SOURCE_CODEC",
    "RULESPEC_SOURCE_CODEC",
    "load_ledger_consumer_bytes",
    "load_populace_us_h5",
    "load_rulespec_tree_bytes",
    "register_transport_codecs",
]

DONOR_SOURCE_CODEC = "populace-us-h5-v1"
FACTS_SOURCE_CODEC = "ledger-consumer-artifact-v1"
RULESPEC_SOURCE_CODEC = "rulespec-tree-v1"


def load_populace_us_h5(path: Path, *, store: ContentStore | None = None) -> Frame:
    """Read an authenticated donor at its original scale and currency.

    SourceRef identity checks belong to the executor. This loader authenticates
    the same local file for the wrapped reader, whose mandatory size/hash pins
    are computed here; a CREATE kernel can instead supply reviewed pins to the
    reader directly. Source lineage stays donor provenance, never destination
    observations.
    """

    path = Path(path)
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    donor = read_populace_us_donor(path, sha256=digest, size=path.stat().st_size)
    person = donor.tables["person"]
    households = donor.tables["household"]
    rows = pd.Index(households["household_id"]).get_indexer(
        person["person_household_id"]
    )
    return Frame(
        donor.tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(donor.weights, WeightKind.DESIGN)},
        pd.Series(
            donor.support_strata[rows],
            index=person.index,
            dtype=object,
            name="stratum",
        ),
        metadata={
            "content_basis": donor.content_basis.value,
            "donor_country": donor.donor_country,
            "currency": donor.currency,
            "source_person_ids": donor.source_person_ids.tolist(),
            "dropped_parent_cycle_edges": donor.dropped_parent_cycle_edges,
        },
    )


def load_ledger_consumer_bytes(
    path: Path, *, store: ContentStore | None = None
) -> bytes:
    """Validate a Chronicle artifact and return its unchanged fact-file bytes."""

    path = Path(path)
    load_ledger_consumer_artifact(path)
    facts_path = path / "consumer_facts.jsonl" if path.is_dir() else path
    return facts_path.read_bytes()


def load_rulespec_tree_bytes(path: Path, *, store: ContentStore | None = None) -> bytes:
    """Pack a nonempty tree deterministically without its absolute root path.

    Every file is included, including auxiliary files. Symlinks are refused:
    an input tree must contain its own bytes rather than borrow undeclared
    resources outside the source root.
    """

    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("A RuleSpec tree source must be a regular directory.")
    entries = sorted(
        path.rglob("*"), key=lambda entry: entry.relative_to(path).as_posix()
    )
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("A RuleSpec tree source must not contain symlinks.")
    files = [
        {
            "path": entry.relative_to(path).as_posix(),
            "bytes_base64": base64.b64encode(entry.read_bytes()).decode("ascii"),
        }
        for entry in entries
        if entry.is_file()
    ]
    if not files:
        raise ValueError("A RuleSpec tree source must contain at least one file.")
    return canonical_json({"format": RULESPEC_SOURCE_CODEC, "files": files})


def register_transport_codecs(registry: SourceCodecRegistry = SOURCE_CODECS) -> None:
    """Register the three loaders explicitly and idempotently.

    The registry refuses an incumbent registered under a different loader or
    mode; registering these same functions again changes nothing.
    """

    registry.register(DONOR_SOURCE_CODEC, load_populace_us_h5)
    registry.register_bytes(FACTS_SOURCE_CODEC, load_ledger_consumer_bytes)
    registry.register_bytes(RULESPEC_SOURCE_CODEC, load_rulespec_tree_bytes)
