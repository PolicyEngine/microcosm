"""Print per-section spec-envelope digests for cross-environment diffing.

When two environments disagree about a country's ``spec_sha256``, the
aggregate hash names the symptom but not the leaking section. This tool
prints one short digest per envelope section (each normative file and
each resolved binding), so diffing two environments' outputs localizes
the divergence to a single section in one pass.

Usage: ``python tools/spec_envelope_digests.py [country ...]``
(defaults to the engine-less compile-proof countries: be, uk).
"""

from __future__ import annotations

import hashlib
import sys

from microcosm.build.spec_engine import canonical, loader
from microcosm.build.spec_engine.canonical import canonical_json_bytes


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:16]


def main(countries: list[str]) -> None:
    captured: list[object] = []
    real = loader.sha256_json

    def capture(value: object) -> str:
        captured.append(value)
        return real(value)

    loader.sha256_json = capture
    try:
        for country in countries:
            captured.clear()
            bundle = loader.load_bundle(country)
            print(country, "spec_sha256", bundle.spec_sha256)
            envelopes = [
                value
                for value in captured
                if isinstance(value, dict)
                and value.get("domain") == canonical.SPEC_DOMAIN
            ]
            for key, value in envelopes[0].items():
                if key in ("files", "resolved_bindings") and isinstance(value, dict):
                    for sub, subvalue in value.items():
                        print(f"  {country} {key}/{sub} {_digest(subvalue)}")
                else:
                    print(f"  {country} {key} {_digest(value)}")
    finally:
        loader.sha256_json = real


if __name__ == "__main__":
    main(sys.argv[1:] or ["be", "uk"])
