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

A second declaration, ``us/chronicle_county_facts.json`` with scope
``us/chronicle_county_facts_scope.json``, pins the county-grain facts that only
the ACS-local county rung reads (microcosm#922). It has the same fields and is
always a bare facts file; :func:`load_us_chronicle_county_facts_pin` loads it
and ``us_runtime.county_facts`` reads the file it names. The two scopes share
no (record set, period) pair, so the main feed, its scope and the parity
resources do not move when the county facts do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.resources import files

from microcosm.build.chronicle_epoch import (
    is_accepted_consumer_artifact_schema_version,
    require_accepted_consumer_fact_schema_versions,
)
from microcosm.build.ledger_artifact import validate_pinned_row_contracts

__all__ = [
    "US_CHRONICLE_COUNTY_FACTS_RESOURCE",
    "US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE",
    "US_CHRONICLE_FEED_RESOURCE",
    "US_CHRONICLE_FEED_SCOPE_RESOURCE",
    "USChronicleFeed",
    "load_us_chronicle_county_facts_pin",
    "load_us_chronicle_feed",
    "require_us_chronicle_feed_row_contracts",
    "us_chronicle_county_facts_scope_sha256",
    "us_chronicle_feed_scope_sha256",
]

US_CHRONICLE_FEED_RESOURCE = "chronicle_feed.json"
US_CHRONICLE_FEED_SCOPE_RESOURCE = "chronicle_feed_scope.json"
US_CHRONICLE_COUNTY_FACTS_RESOURCE = "chronicle_county_facts.json"
US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE = "chronicle_county_facts_scope.json"
_FEED_SUBJECT = "US Chronicle feed"
_COUNTY_FACTS_SUBJECT = "US Chronicle county facts"
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


def require_us_chronicle_feed_row_contracts(
    fact_schema_versions: Iterable[str],
    *,
    facts_sha256: str,
    pin: USChronicleFeed | None = None,
) -> USChronicleFeed:
    """Hold the row contracts a loaded US feed's rows carry to the US pin.

    The US pin describes a bare feed, so the full feed-pin check cannot run
    on it; the code that holds the feed to the pin by digest calls this
    instead. Rows of the committed feed (``facts_sha256`` is the pin's) must
    carry exactly the pin's ``consumer_fact_schema_versions``, which a re-pin
    types by hand. Rows of any other feed (a reviewed
    ``--allow-unpinned-feed`` run of the release tool) must carry declared
    row contracts only. Returns the pin they were checked against.
    """

    pin = pin or load_us_chronicle_feed()
    validate_pinned_row_contracts(
        fact_schema_versions,
        pinned=(
            pin.consumer_fact_schema_versions
            if facts_sha256 == pin.facts_sha256
            else None
        ),
        subject="US Chronicle feed",
    )
    return pin


def _feed_path():
    return files("microcosm.build.us").joinpath(US_CHRONICLE_FEED_RESOURCE)


def _scope_path():
    return files("microcosm.build.us").joinpath(US_CHRONICLE_FEED_SCOPE_RESOURCE)


def _county_facts_path():
    return files("microcosm.build.us").joinpath(US_CHRONICLE_COUNTY_FACTS_RESOURCE)


def _county_facts_scope_path():
    return files("microcosm.build.us").joinpath(
        US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE
    )


def us_chronicle_feed_scope_sha256() -> str:
    """SHA-256 of the packaged scope file the pinned feed was built from."""

    return hashlib.sha256(_scope_path().read_bytes()).hexdigest()


def us_chronicle_county_facts_scope_sha256() -> str:
    """SHA-256 of the packaged scope file the pinned county facts were built from."""

    return hashlib.sha256(_county_facts_scope_path().read_bytes()).hexdigest()


def _require_hex(
    raw: dict,
    field: str,
    length: int,
    *,
    nullable: bool = False,
    subject: str = _FEED_SUBJECT,
) -> None:
    value = raw.get(field)
    if value is None and nullable:
        return
    if not isinstance(value, str) or not re.fullmatch(rf"{_HEX}{{{length}}}", value):
        raise ValueError(
            f"{subject} {field} must be {length} lowercase hex chars"
            + (" or null." if nullable else ".")
        )


def _load_pin(
    content: bytes,
    *,
    scope: str,
    scope_resource: str,
    scope_sha256: str,
    subject: str,
) -> USChronicleFeed:
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError(f"{subject} must contain a JSON object.")
    for field, expected in (
        ("version", 1),
        ("country", "us"),
        ("scope", scope),
        ("source_repo", "PolicyEngine/chronicle"),
    ):
        if type(raw.get(field)) is not type(expected) or raw[field] != expected:
            raise ValueError(f"{subject} {field} must be {expected!r}.")
    _require_hex(raw, "source_commit", 40, subject=subject)
    _require_hex(raw, "facts_sha256", 64, subject=subject)
    _require_hex(raw, "consumer_fact_schema_sha256", 64, subject=subject)
    _require_hex(raw, "scope_sha256", 64, subject=subject)
    _require_hex(raw, "manifest_sha256", 64, nullable=True, subject=subject)
    if type(raw.get("fact_row_count")) is not int or raw["fact_row_count"] <= 0:
        raise ValueError(f"{subject} fact_row_count must be positive.")
    if not isinstance(raw.get("build"), str) or not raw["build"].strip():
        raise ValueError(f"{subject} build must be a nonempty string.")
    artifact_schema_version = raw.get("artifact_schema_version")
    if (raw.get("manifest_sha256") is None) != (artifact_schema_version is None):
        raise ValueError(
            f"{subject} manifest_sha256 and artifact_schema_version must "
            "both be set (artifact feed) or both be null (bare feed)."
        )
    if artifact_schema_version is not None and (
        not is_accepted_consumer_artifact_schema_version(artifact_schema_version)
    ):
        raise ValueError(f"{subject} artifact_schema_version is unsupported.")
    versions = raw.get("consumer_fact_schema_versions")
    if (
        not isinstance(versions, list)
        or not versions
        or any(not isinstance(v, str) or not v for v in versions)
        or len(set(versions)) != len(versions)
    ):
        raise ValueError(f"{subject} consumer_fact_schema_versions is invalid.")
    require_accepted_consumer_fact_schema_versions(versions, subject=subject)
    if raw["scope_sha256"] != scope_sha256:
        raise ValueError(
            f"{subject} scope_sha256 does not match the packaged "
            f"{scope_resource} ({scope_sha256}); a "
            "changed scope needs a rebuilt feed and a new pin."
        )
    return USChronicleFeed(
        **{**raw, "consumer_fact_schema_versions": tuple(versions)},
        resource_sha256=hashlib.sha256(content).hexdigest(),
        resource_size_bytes=len(content),
    )


def load_us_chronicle_feed() -> USChronicleFeed:
    """Load the packaged US pin without importing Chronicle or an engine."""

    return _load_pin(
        _feed_path().read_bytes(),
        scope="us_fiscal_targets",
        scope_resource=US_CHRONICLE_FEED_SCOPE_RESOURCE,
        scope_sha256=us_chronicle_feed_scope_sha256(),
        subject=_FEED_SUBJECT,
    )


def load_us_chronicle_county_facts_pin() -> USChronicleFeed:
    """Load the packaged county facts pin (microcosm#922).

    Same fields and checks as the main pin, with scope ``us_county_facts``; the
    county facts are always a bare facts file, never a consumer artifact.
    """

    pin = _load_pin(
        _county_facts_path().read_bytes(),
        scope="us_county_facts",
        scope_resource=US_CHRONICLE_COUNTY_FACTS_SCOPE_RESOURCE,
        scope_sha256=us_chronicle_county_facts_scope_sha256(),
        subject=_COUNTY_FACTS_SUBJECT,
    )
    if not pin.is_bare_feed:
        raise ValueError(
            f"{_COUNTY_FACTS_SUBJECT} must pin a bare facts file "
            "(manifest_sha256 and artifact_schema_version null)."
        )
    return pin
