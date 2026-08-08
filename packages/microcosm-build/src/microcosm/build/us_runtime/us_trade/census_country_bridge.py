"""Census Schedule C country code → ISO 3166-1 alpha-2 bridge (vendored).

The bridge is maintained fail-closed in TheAxiomFoundation/axiom-oracles at
``reference/us-tariff-panel/census_iso_bridge.csv`` (built by
``scripts/build_census_iso_bridge.py`` from the Census Schedule C country
list). Microcosm vendors the built CSV verbatim and never rebuilds it; the
pinned copy is verified against the upstream build's content hash at load
time so a silent local edit cannot ship.

Schedule C codes with no ISO 3166-1 assignment carry Census's own two-letter
conventions and are passed through as published: GZ (Gaza Strip) and WE
(West Bank), which ISO folds into PS, and KV (Kosovo, no assigned ISO code).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType

__all__ = [
    "BRIDGE_PROVENANCE_RESOURCE_NAME",
    "BRIDGE_PROVENANCE_SHA256",
    "BRIDGE_RESOURCE_NAME",
    "BRIDGE_SHA256",
    "BRIDGE_UPSTREAM",
    "BRIDGE_UPSTREAM_COMMIT",
    "CensusCountryBridge",
    "load_census_country_bridge",
]

BRIDGE_RESOURCE_NAME = "census_iso_bridge.csv"

#: The vendored companion provenance record, copied verbatim from the same
#: upstream commit as the CSV. It documents the Schedule C snapshot hash,
#: the builder script hash, the row count, and the Census-convention
#: extensions (GZ/WE/KV), and declares the CSV's ``bridge_sha256``.
BRIDGE_PROVENANCE_RESOURCE_NAME = "bridge_provenance.json"

#: Content hash of the vendored ``bridge_provenance.json`` bytes.
BRIDGE_PROVENANCE_SHA256 = (
    "d2eb46153e9b1069757fe9fd04529c6f9865b382033c09a7ca6ce733f393bac8"
)

#: Content hash of the upstream build (``bridge_sha256`` in the provenance
#: record; built 2026-08-01 from the Schedule C snapshot ``ad7d5993…``).
BRIDGE_SHA256 = "4a5a96988c92f51938326cffa16fb5cdb3dd7ce81a56a941f287c8b7a0ffe6f7"

#: The exact axiom-oracles commit both vendored files were taken from —
#: an immutable pin, unlike a branch URL.
BRIDGE_UPSTREAM_COMMIT = "9310d19b8c9649c2f8cd16dbc545ceabf15dd19c"

BRIDGE_UPSTREAM = (
    "https://github.com/TheAxiomFoundation/axiom-oracles/blob/"
    f"{BRIDGE_UPSTREAM_COMMIT}/reference/us-tariff-panel/census_iso_bridge.csv"
)


@dataclass(frozen=True)
class CensusCountryBridge:
    """Fail-closed mapping from Census Schedule C codes to ISO-2 codes."""

    iso2_by_census_code: Mapping[str, str]
    name_by_census_code: Mapping[str, str]
    sha256: str

    def iso2(self, census_code: str) -> str:
        """Return the ISO-2 code for a Schedule C country code.

        Raises:
            KeyError: If the code is not in the bridge. Unmapped codes must
                fail the ingest rather than silently drop or mislabel a
                country margin.
        """
        code = str(census_code).strip()
        try:
            return self.iso2_by_census_code[code]
        except KeyError:
            raise KeyError(
                f"Census Schedule C country code {code!r} is not in the "
                f"vendored bridge ({BRIDGE_UPSTREAM}). The bridge is "
                "fail-closed: extend it upstream, do not guess here."
            ) from None

    def __len__(self) -> int:
        return len(self.iso2_by_census_code)


def load_census_country_bridge() -> CensusCountryBridge:
    """Load and hash-verify the vendored Schedule C → ISO-2 bridge.

    Both vendored files are verified: the CSV bytes must hash to
    ``BRIDGE_SHA256``, the companion provenance record must hash to
    ``BRIDGE_PROVENANCE_SHA256``, and the provenance record's own
    ``bridge_sha256`` claim must agree with the CSV — so neither file can
    drift from the other or from the pinned upstream commit silently.
    """
    package = files("microcosm.build.us_runtime.us_trade")
    raw = package.joinpath(BRIDGE_RESOURCE_NAME).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BRIDGE_SHA256:
        raise ValueError(
            f"Vendored census ISO bridge hash mismatch: {digest} != "
            f"{BRIDGE_SHA256}. Re-vendor the upstream build verbatim "
            f"({BRIDGE_UPSTREAM}) and update BRIDGE_SHA256 with its "
            "recorded provenance hash."
        )
    provenance_raw = package.joinpath(BRIDGE_PROVENANCE_RESOURCE_NAME).read_bytes()
    provenance_digest = hashlib.sha256(provenance_raw).hexdigest()
    if provenance_digest != BRIDGE_PROVENANCE_SHA256:
        raise ValueError(
            "Vendored census ISO bridge provenance hash mismatch: "
            f"{provenance_digest} != {BRIDGE_PROVENANCE_SHA256}. Re-vendor "
            f"bridge_provenance.json from commit {BRIDGE_UPSTREAM_COMMIT}."
        )
    declared = json.loads(provenance_raw).get("bridge_sha256")
    if declared != BRIDGE_SHA256:
        raise ValueError(
            "Vendored bridge provenance declares a different CSV hash "
            f"({declared}) than the vendored CSV ({BRIDGE_SHA256}); the two "
            "files are not from the same upstream build."
        )
    iso2_by_code: dict[str, str] = {}
    name_by_code: dict[str, str] = {}
    reader = csv.DictReader(raw.decode("utf-8").splitlines())
    for row in reader:
        code = row["census_code"].strip()
        iso2 = row["iso2"].strip()
        if not code or not iso2:
            raise ValueError(f"Census ISO bridge has a blank mapping row: {row!r}.")
        if code in iso2_by_code:
            raise ValueError(f"Census ISO bridge duplicates census code {code!r}.")
        iso2_by_code[code] = iso2
        name_by_code[code] = row.get("name", "").strip()
    if not iso2_by_code:
        raise ValueError("Census ISO bridge loaded zero rows.")
    return CensusCountryBridge(
        iso2_by_census_code=MappingProxyType(iso2_by_code),
        name_by_census_code=MappingProxyType(name_by_code),
        sha256=digest,
    )
