"""Census Schedule C country code → ISO 3166-1 alpha-2 bridge (vendored).

The bridge is maintained fail-closed in TheAxiomFoundation/axiom-oracles at
``reference/us-tariff-panel/census_iso_bridge.csv`` (built by
``scripts/build_census_iso_bridge.py`` from the Census Schedule C country
list). Populace vendors the built CSV verbatim and never rebuilds it; the
pinned copy is verified against the upstream build's content hash at load
time so a silent local edit cannot ship.

Schedule C codes with no ISO 3166-1 assignment carry Census's own two-letter
conventions and are passed through as published: GZ (Gaza Strip) and WE
(West Bank), which ISO folds into PS, and KV (Kosovo, no assigned ISO code).
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType

__all__ = [
    "BRIDGE_RESOURCE_NAME",
    "BRIDGE_SHA256",
    "BRIDGE_UPSTREAM",
    "CensusCountryBridge",
    "load_census_country_bridge",
]

BRIDGE_RESOURCE_NAME = "census_iso_bridge.csv"

#: Content hash of the upstream build, recorded in axiom-oracles
#: ``reference/us-tariff-panel/bridge_provenance.json`` (``bridge_sha256``,
#: built 2026-08-01 from the Schedule C snapshot ``ad7d5993…``).
BRIDGE_SHA256 = "4a5a96988c92f51938326cffa16fb5cdb3dd7ce81a56a941f287c8b7a0ffe6f7"

BRIDGE_UPSTREAM = (
    "https://github.com/TheAxiomFoundation/axiom-oracles/blob/main/"
    "reference/us-tariff-panel/census_iso_bridge.csv"
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
    """Load and hash-verify the vendored Schedule C → ISO-2 bridge."""
    resource = files("populace.build.us_runtime.us_trade").joinpath(
        BRIDGE_RESOURCE_NAME
    )
    raw = resource.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BRIDGE_SHA256:
        raise ValueError(
            f"Vendored census ISO bridge hash mismatch: {digest} != "
            f"{BRIDGE_SHA256}. Re-vendor the upstream build verbatim "
            f"({BRIDGE_UPSTREAM}) and update BRIDGE_SHA256 with its "
            "recorded provenance hash."
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
