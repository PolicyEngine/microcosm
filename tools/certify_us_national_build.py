#!/usr/bin/env python3
"""Certify a finished US national build so it can be published.

``tools/build_us_fiscal_refresh_release.py`` builds the dataset and stops: it
writes the H5, the calibration weights, the diagnostics, every gate verdict
the build itself checks, and ``build_record.json``, then exits so the build
can be evaluated. Nothing it writes is publishable.

This command runs the release's post-export checks on that build: the
reform-coverage smoke, reform validation, demographics and the holdouts,
scored on the written H5, then the source coverage and take-up receipts. It
writes both manifests, so ``microcosm-publish-release-build`` can then
publish the build (microcosm#1585 item 11, PR 4a). It reads the build's
options from its record, never from its own command line, and runs at the
commit the build was built at::

    uv run --no-sync python tools/certify_us_national_build.py <build --out dir>

A failed certification leaves the build as it was; rerun this command once
the cause is fixed. A certified build is never certified again.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import build_us_fiscal_refresh_release as fiscal_release  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    fiscal_release.certify_main(argv)


if __name__ == "__main__":
    main()
