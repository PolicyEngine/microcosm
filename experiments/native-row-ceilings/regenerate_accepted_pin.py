"""Regenerate acs_native_coverage_binding._ACCEPTED. Never hand-edit a pin.

``_ACCEPTED`` pins four ACS modules by whole-file sha256 and refuses with
``UNREVIEWED_PREPARATION`` when one differs. Its generator is exactly
``coverage._sha(Path(__file__).with_name(name).read_bytes())``, which is
sha256 of the file's bytes -- the same call the check itself makes. This
rewrites each entry with that value and prints what moved.

    python regenerate_accepted_pin.py [--write]
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
US = ROOT / "packages/microcosm-build/src/microcosm/build/us_runtime"
OWNER = US / "acs_native_coverage_binding.py"


def main() -> int:
    write = "--write" in sys.argv[1:]
    text = OWNER.read_text()
    block = re.search(r"_ACCEPTED = \{\n(.*?)\n\}\n", text, re.S)
    if block is None:
        raise SystemExit("could not locate the _ACCEPTED block")
    body = block.group(1)
    moved, replaced = [], body
    for name, old in re.findall(r'^    "([^"]+)": "([0-9a-f]{64})",$', body, re.M):
        new = hashlib.sha256((US / name).read_bytes()).hexdigest()
        if new != old:
            moved.append((name, old, new))
            replaced = replaced.replace(f'"{name}": "{old}"', f'"{name}": "{new}"')
    for name, old, new in moved:
        print(f"MOVED {name}\n    old {old}\n    new {new}")
    if not moved:
        print("no pin moved")
        return 0
    if not write:
        print("(dry run; pass --write to apply)")
        return 0
    OWNER.write_text(text.replace(body, replaced))
    print(f"rewrote {OWNER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
