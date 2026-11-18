"""Recover the seven candidate Chronicle packages from immutable Git objects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "paths.json").read_text())
PACKAGES = (
    "soi-historic-table-2-2023",
    "soi-historic-table-2-state-agi-2023",
    "soi-historic-table-2-state-broad-2023",
    "soi-historic-table-2-state-eitc-2023",
    "soi-state-2023",
    "soi-ira-roth-contributions-2023",
    "soi-ira-traditional-contributions-2023",
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronicle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        # These are the only pieces of the base repository the seven parsers
        # need, including Chronicle's canonical display labels and raw source.
        archive = git(
            args.chronicle,
            "archive",
            PINS["base"],
            "chronicle",
            "packages/__init__.py",
            "db/*.py",
            "db/data/irs_soi/historic_table_2",
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        for path, commit in PINS["overrides"].items():
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(git(args.chronicle, "show", f"{commit}:{path}"))
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(HERE / "integration.patch")],
            cwd=root,
            check=True,
        )
        sys.path.insert(0, str(root))
        from chronicle.consumer_contract import consumer_fact_rows
        from chronicle.source_package import load_source_package

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as stage:
            target = Path(stage) / "consumer_facts.jsonl"
            sha = hashlib.sha256()
            with target.open("wb") as stream:
                for package in PACKAGES:
                    for row in consumer_fact_rows(
                        load_source_package(package).build_facts(2023)
                    ):
                        raw = (
                            json.dumps(row, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        ).encode()
                        sha.update(raw)
                        stream.write(raw)
            if sha.hexdigest() != PINS["annual_sha256"]:
                raise ValueError(f"Unexpected rebuilt feed digest {sha.hexdigest()}")
            target.replace(args.out)


if __name__ == "__main__":
    main()
