"""Reviewed Chronicle feed identity for the US fiscal target registry.

One declaration, ``us/chronicle_feed.json``, names the Chronicle commit, the
scope file that selects the feed's (record set, period) pairs, and the digest
of the feed those two produce through ``tools/build_us_chronicle_feed.py``.
The target-parity resources (``us/target_parity_manifest.json`` and
``us/target_parity_feed_families.json``) restate the feed digest and are
drift-gated against this pin by their tests, and
``tools/build_us_target_parity_manifest.py`` refuses to regenerate them
against any other feed.

The pin may describe a bare feed (``manifest_sha256`` and
``artifact_schema_version`` null) when Chronicle's consumer-artifact validator
refuses the feed at the pinned commit; the ``--base-h5`` release arm accepts a
bare ``consumer_facts.jsonl``, the ``--exact-k`` arm needs the artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files

from microcosm.build.chronicle_epoch import (
    is_accepted_consumer_artifact_schema_version,
)

__all__ = [
    "US_CHRONICLE_FEED_RESOURCE",
    "US_CHRONICLE_FEED_SCOPE_RESOURCE",
    "USChronicleFeed",
    "load_us_chronicle_feed",
    "us_chronicle_feed_scope_sha256",
]

US_CHRONICLE_FEED_RESOURCE = "chronicle_feed.json"
US_CHRONICLE_FEED_SCOPE_RESOURCE = "chronicle_feed_scope.json"
_HEX = "[0-9a-f]"


@dataclass(frozen=True)
class USChronicleFeed:
    version: int
    country: str
    scope: str
    source_repo: str
    source_commit: str
    build: str
    artifact_schema_version: str | None
    consumer_fact_schema_versions: tuple[str, ...]
    consumer_fact_schema_sha256: str
    fact_row_count: int
    facts_sha256: str
    manifest_sha256: str | None
    scope_sha256: str
    resource_sha256: str
    resource_size_bytes: int

    @property
    def is_bare_feed(self) -> bool:
        """True when the pin names a facts file with no consumer-artifact manifest."""

        return self.manifest_sha256 is None

    def to_dict(self) -> dict[str, object]:
        """Include the declaration's own digest in the run provenance."""

        result = asdict(self)
        result["consumer_fact_schema_versions"] = list(
            self.consumer_fact_schema_versions
        )
        return result


def _feed_path():
    return files("microcosm.build.us").joinpath(US_CHRONICLE_FEED_RESOURCE)


def _scope_path():
    return files("microcosm.build.us").joinpath(US_CHRONICLE_FEED_SCOPE_RESOURCE)


def us_chronicle_feed_scope_sha256() -> str:
    """SHA-256 of the packaged scope file the pinned feed was built from."""

    return hashlib.sha256(_scope_path().read_bytes()).hexdigest()


def _require_hex(raw: dict, field: str, length: int, *, nullable: bool = False) -> None:
    value = raw.get(field)
    if value is None and nullable:
        return
    if not isinstance(value, str) or not re.fullmatch(rf"{_HEX}{{{length}}}", value):
        raise ValueError(
            f"US Chronicle feed {field} must be {length} lowercase hex chars"
            + (" or null." if nullable else ".")
        )


def load_us_chronicle_feed() -> USChronicleFeed:
    """Load the packaged US pin without importing Chronicle or an engine."""

    content = _feed_path().read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("US Chronicle feed must contain a JSON object.")
    for field, expected in (
        ("version", 1),
        ("country", "us"),
        ("scope", "us_fiscal_targets"),
        ("source_repo", "PolicyEngine/chronicle"),
    ):
        if type(raw.get(field)) is not type(expected) or raw[field] != expected:
            raise ValueError(f"US Chronicle feed {field} must be {expected!r}.")
    _require_hex(raw, "source_commit", 40)
    _require_hex(raw, "facts_sha256", 64)
    _require_hex(raw, "consumer_fact_schema_sha256", 64)
    _require_hex(raw, "scope_sha256", 64)
    _require_hex(raw, "manifest_sha256", 64, nullable=True)
    if type(raw.get("fact_row_count")) is not int or raw["fact_row_count"] <= 0:
        raise ValueError("US Chronicle feed fact_row_count must be positive.")
    if not isinstance(raw.get("build"), str) or not raw["build"].strip():
        raise ValueError("US Chronicle feed build must be a nonempty string.")
    artifact_schema_version = raw.get("artifact_schema_version")
    if (raw.get("manifest_sha256") is None) != (artifact_schema_version is None):
        raise ValueError(
            "US Chronicle feed manifest_sha256 and artifact_schema_version must "
            "both be set (artifact feed) or both be null (bare feed)."
        )
    if artifact_schema_version is not None and (
        not is_accepted_consumer_artifact_schema_version(artifact_schema_version)
    ):
        raise ValueError("US Chronicle feed artifact_schema_version is unsupported.")
    versions = raw.get("consumer_fact_schema_versions")
    if (
        not isinstance(versions, list)
        or not versions
        or any(not isinstance(v, str) or not v for v in versions)
        or len(set(versions)) != len(versions)
    ):
        raise ValueError("US Chronicle feed consumer_fact_schema_versions is invalid.")
    expected_scope_sha256 = us_chronicle_feed_scope_sha256()
    if raw["scope_sha256"] != expected_scope_sha256:
        raise ValueError(
            "US Chronicle feed scope_sha256 does not match the packaged "
            f"{US_CHRONICLE_FEED_SCOPE_RESOURCE} ({expected_scope_sha256}); a "
            "changed scope needs a rebuilt feed and a new pin."
        )
    return USChronicleFeed(
        **{**raw, "consumer_fact_schema_versions": tuple(versions)},
        resource_sha256=hashlib.sha256(content).hexdigest(),
        resource_size_bytes=len(content),
    )
