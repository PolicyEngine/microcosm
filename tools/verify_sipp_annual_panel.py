"""Run invented panel tests and record precommit or Git-byte verification evidence."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-git-bytes", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir()
    sys.path.insert(0, str(REPO))
    for source_root in sorted((REPO / "packages").glob("microcosm-*/src")):
        sys.path.insert(0, str(source_root))
    from tools.run_sipp_annual_panel import bytecode_flags, loaded_code

    flags = bytecode_flags()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    started = time.monotonic()
    code = pytest.main(
        [
            "-q",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "--confcutdir=" + str(REPO),
            str(REPO / "packages/microcosm-build/tests/test_us_sipp_annual_panel.py"),
            "--junitxml=" + str(args.output_dir / "tests.xml"),
        ]
    )
    elapsed = time.monotonic() - started
    git_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    loaded = loaded_code()
    comparison = {}
    for name, digest in loaded.items():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", git_head + ":" + name], capture_output=True
        )
        git_digest = (
            hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
        )
        comparison[name] = {
            "loaded_sha256": digest,
            "git_blob_sha256": git_digest,
            "match": digest == git_digest,
        }
    suites = ET.parse(args.output_dir / "tests.xml").getroot()
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    receipt = {
        "phase": "final_loaded_git_bytes" if args.require_git_bytes else "precommit",
        "git_head": git_head,
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "tests": counts,
        "pytest_exit": int(code),
        "wall_seconds": elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "bytecode": flags,
        "python": sys.version,
        "loaded_code": comparison,
        "all_loaded_code_matches_git": all(x["match"] for x in comparison.values()),
    }
    with (args.output_dir / "verification.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if code or (args.require_git_bytes and not receipt["all_loaded_code_matches_git"]):
        raise SystemExit(1)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "phase",
                    "tests",
                    "wall_seconds",
                    "peak_rss_bytes",
                    "all_loaded_code_matches_git",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
