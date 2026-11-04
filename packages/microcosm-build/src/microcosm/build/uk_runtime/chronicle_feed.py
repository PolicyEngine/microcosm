"""Reviewed Chronicle artifact identity for the UK calibration surfaces.

One declaration governs every UK consumer of the Chronicle feed: the national
references and membership, the local census, the local references and the
local validation levels all read this pin (the target contract itself is one
file for both grains), so a re-pin is one reviewed change and the surfaces
cannot drift apart silently. Each surface still regenerates from the pinned
artifact under its own test.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files

from microcosm.build.chronicle_epoch import (
    is_accepted_consumer_artifact_schema_version,
    require_accepted_consumer_fact_schema_versions,
)
from microcosm.build.ledger_artifact import validate_pinned_row_contracts


@dataclass(frozen=True)
class UKChronicleFeed:
    version: int
    country: str
    scope: str
    source_repo: str
    source_commit: str
    build: str
    artifact_schema_version: str
    consumer_fact_schema_versions: tuple[str, ...]
    consumer_fact_schema_sha256: str
    fact_row_count: int
    facts_sha256: str
    manifest_sha256: str
    resource_sha256: str
    resource_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Include the declaration's own digest in the run provenance."""
        result = asdict(self)
        result["consumer_fact_schema_versions"] = list(
            self.consumer_fact_schema_versions
        )
        return result


class UKChronicleFeedPinError(ValueError):
    """A Ledger artifact is not the committed UK national Chronicle feed."""


def require_committed_uk_chronicle_feed_pin(
    facts_sha256: str,
    *,
    manifest_sha256: str | None,
    allow_unpinned_feed: bool,
    pin: UKChronicleFeed | None = None,
) -> UKChronicleFeed:
    """Refuse a Ledger artifact whose digests are not the committed feed pin.

    Returns the pin the artifact was checked against. ``allow_unpinned_feed``
    admits a mismatch for an explicitly reviewed diagnostic run; the caller
    records the override in the run provenance.
    """

    pin = pin or load_uk_chronicle_feed()
    mismatches = []
    for label, loaded, committed in (
        ("facts", facts_sha256, pin.facts_sha256),
        ("manifest", manifest_sha256, pin.manifest_sha256),
    ):
        if loaded != committed:
            mismatches.append(f"{label}: loaded {loaded}, committed {committed}")
    if mismatches and not allow_unpinned_feed:
        raise UKChronicleFeedPinError(
            "Chronicle artifact differs from the committed UK national feed pin: "
            + "; ".join(mismatches)
            + "; pass --allow-unpinned-feed only for an explicitly reviewed "
            "diagnostic run"
        )
    return pin


def require_uk_chronicle_feed_row_contracts(
    fact_schema_versions: Iterable[str],
    *,
    facts_sha256: str,
    pin: UKChronicleFeed | None = None,
) -> UKChronicleFeed:
    """Hold the row contracts a loaded UK feed's rows carry to the pin.

    The digest checks say the rows are the committed feed's; this says what
    they are. Rows of the committed feed (``facts_sha256`` is the pin's) must
    carry exactly the pin's ``consumer_fact_schema_versions``, which a re-pin
    types by hand. Rows of another feed, admitted by ``--allow-unpinned-feed``
    for a reviewed diagnostic run, must carry declared row contracts only.
    Returns the pin they were checked against.
    """

    pin = pin or load_uk_chronicle_feed()
    validate_pinned_row_contracts(
        fact_schema_versions,
        pinned=(
            pin.consumer_fact_schema_versions
            if facts_sha256 == pin.facts_sha256
            else None
        ),
        subject="UK Chronicle feed",
    )
    return pin


def require_uk_chronicle_feed_subset_row_contracts(
    fact_schema_versions: Iterable[str],
    *,
    pinned_versions: Sequence[str] | None = None,
) -> None:
    """Hold the row contracts of part of the pinned UK feed to the pin.

    For the tools that parse the feed's rows themselves: the reference
    generators and the compile-parity tool, which read the guide's subsets
    of the pinned artifact, and the DESNZ constituency builder, which parses
    only the rows it needs. A subset need not carry every row contract the
    pin lists, so each row's contract must be one the pin lists
    (``pinned_versions``, by default the committed pin's) and one
    :mod:`microcosm.build.chronicle_epoch` declares.
    """

    validate_pinned_row_contracts(
        fact_schema_versions,
        pinned=(
            load_uk_chronicle_feed().consumer_fact_schema_versions
            if pinned_versions is None
            else pinned_versions
        ),
        subject="UK Chronicle feed",
        subset=True,
    )


def _feed_path():
    return files("microcosm.build.uk").joinpath("chronicle_feed.json")


def load_uk_chronicle_feed() -> UKChronicleFeed:
    """Load the packaged national pin without importing Chronicle or an engine."""
    content = _feed_path().read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("UK Chronicle feed must contain a JSON object.")
    for field, expected in (
        ("version", 1),
        ("country", "uk"),
        ("scope", "uk_calibration"),
        ("source_repo", "PolicyEngine/chronicle"),
    ):
        if type(raw.get(field)) is not type(expected) or raw[field] != expected:
            raise ValueError(f"UK Chronicle feed {field} must be {expected!r}.")
    for field, length in (
        ("source_commit", 40),
        ("facts_sha256", 64),
        ("manifest_sha256", 64),
        ("consumer_fact_schema_sha256", 64),
    ):
        value = raw.get(field)
        if not isinstance(value, str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value
        ):
            raise ValueError(
                f"UK Chronicle feed {field} must be {length} lowercase hex chars."
            )
    if type(raw.get("fact_row_count")) is not int or raw["fact_row_count"] <= 0:
        raise ValueError("UK Chronicle feed fact_row_count must be positive.")
    if not isinstance(raw.get("build"), str) or not raw["build"].strip():
        raise ValueError("UK Chronicle feed build must be a nonempty string.")
    if not is_accepted_consumer_artifact_schema_version(
        raw.get("artifact_schema_version")
    ):
        raise ValueError("UK Chronicle feed artifact_schema_version is unsupported.")
    versions = raw.get("consumer_fact_schema_versions")
    if (
        not isinstance(versions, list)
        or not versions
        or any(not isinstance(v, str) or not v for v in versions)
        or len(set(versions)) != len(versions)
    ):
        raise ValueError("UK Chronicle feed consumer_fact_schema_versions is invalid.")
    require_accepted_consumer_fact_schema_versions(
        versions, subject="UK Chronicle feed"
    )
    return UKChronicleFeed(
        **{**raw, "consumer_fact_schema_versions": tuple(versions)},
        resource_sha256=hashlib.sha256(content).hexdigest(),
        resource_size_bytes=len(content),
    )
