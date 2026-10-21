"""Re-fetch the pinned Stats NZ pages behind MC9's Accommodation Supplement area bridge.

``tools/build_nz_as_area_crosswalk.py`` reads its inputs from a local directory
and verifies every byte against the pins in ``nz/as_area_crosswalk.json``. The
pages are public Stats NZ map-service responses (CC BY 4.0), too large to
commit, so this tool recreates that directory: it fetches each pinned page from
its recorded query URL, and each layer's metadata from the layer endpoint, and
refuses any response whose size or SHA-256 differs from its pin. A mismatch
means the publisher's service now returns different bytes; regenerate the
crosswalk deliberately rather than relaxing the pin. Example::

    python tools/fetch_nz_as_area_sources.py --out nz-as-area-sources
    python tools/build_nz_as_area_crosswalk.py --sources nz-as-area-sources \
        --out packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json

Only public URLs are requested and no credentials are sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

SPEC = Path("packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json")
USER_AGENT = "PolicyEngine-microcosm/fetch_nz_as_area_sources"


def metadata_url(query_url):
    """Return the layer metadata endpoint for a FeatureServer query URL."""
    if not query_url.endswith("/query"):
        raise ValueError(f"Expected a FeatureServer query URL: {query_url}")
    return query_url[: -len("/query")] + "?f=json"


def pinned_downloads(provenance):
    """List (file, url, sha256, bytes-or-None) for every pinned input."""
    downloads = [
        (page["file"], page["url"], page["sha256"], page.get("bytes"))
        for page in provenance["pages"]
    ]
    downloads.extend(
        (
            layer["metadata_file"],
            metadata_url(layer["query_url"]),
            layer["metadata_sha256"],
            None,
        )
        for layer in provenance["layers"]
    )
    names = [name for name, *_ in downloads]
    if len(names) != len(set(names)):
        raise ValueError("Pinned file names must be unique")
    return downloads


def verify(name, payload, sha256, size):
    """Refuse bytes that differ from their pin."""
    if size is not None and len(payload) != size:
        raise ValueError(f"{name}: {len(payload)} bytes, pinned {size}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise ValueError(f"{name}: SHA-256 {digest}, pinned {sha256}")


def _fetch(url, attempts=4):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_all(provenance, out_dir, fetch=_fetch):
    """Fetch every pinned input into out_dir; keep files that already verify."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, url, sha256, size in pinned_downloads(provenance):
        target = out_dir / name
        if target.exists():
            try:
                verify(name, target.read_bytes(), sha256, size)
                continue
            except ValueError:
                pass
        payload = fetch(url)
        verify(name, payload, sha256, size)
        target.write_bytes(payload)
        fetched.append(name)
    return fetched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    provenance = json.loads(args.spec.read_text())["provenance"]
    fetched = fetch_all(provenance, args.out)
    total = len(pinned_downloads(provenance))
    print(f"{total} pinned inputs verified in {args.out}; {len(fetched)} fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
