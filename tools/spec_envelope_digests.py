"""Print per-section spec-envelope digests for cross-environment diffing.

When two environments disagree about a country's ``spec_sha256``, the
aggregate hash names the symptom but not the leaking section. This tool
prints one short digest per envelope section (each normative file and
each resolved binding), so diffing two environments' outputs localizes
the divergence to a single section in one pass.

Usage: ``python tools/spec_envelope_digests.py [country ...]``
(defaults to the engine-less compile-proof countries: be, uk).
``--dump SECTION`` additionally prints the named resolved-binding
section as canonical JSON for the first country, so two environments'
outputs can be diffed byte-for-byte.
"""

from __future__ import annotations

import hashlib
import sys

from microcosm.build.spec_engine import canonical, loader, seeds
from microcosm.build.spec_engine.canonical import canonical_json_bytes


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:16]


def main(argv: list[str]) -> None:
    dump_section = None
    if "--dump" in argv:
        at = argv.index("--dump")
        dump_section = argv[at + 1]
        argv = argv[:at] + argv[at + 2 :]
    countries = argv or ["be", "uk"]

    captured: list[object] = []
    real = loader.sha256_json

    def capture(value: object) -> str:
        captured.append(value)
        return real(value)

    loader.sha256_json = capture
    try:
        for index, country in enumerate(countries):
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
            if dump_section is not None and index == 0:
                section = envelopes[0]["resolved_bindings"][dump_section]
                print(f"DUMP {dump_section}")
                print(canonical_json_bytes(section).decode("utf-8"))
                if dump_section == "seed_protocol":
                    for kernel in seeds.LEGACY_V1_KERNELS:
                        for module in kernel.source_modules:
                            print(
                                "MODULE",
                                module,
                                seeds.source_inventory_sha256((module,))[:16],
                            )
    finally:
        loader.sha256_json = real


if __name__ == "__main__":
    main(sys.argv[1:])
