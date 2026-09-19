"""Rebuild the pinned US Chronicle consumer feed from targeted package runs.

    uv run python tools/build_us_chronicle_feed.py \\
        --chronicle-root ~/PolicyEngine/chronicle --out <dir> [--replace] \\
        [--scope <chronicle_feed_scope.json>] [--skip-artifact]

The scope file (default: the packaged ``us/chronicle_feed_scope.json``) names
the Chronicle commit and, for every (record set, period) pair the feed keeps,
the source package that emits it and the ``--year`` to build that package with.
This tool:

1. refuses unless ``--chronicle-root`` is a clean checkout at exactly the
   scope's commit;
2. runs one ``chronicle build-bundle --year Y --source <package> ...`` per
   build year, naming only the packages the scope needs for that year;
3. reads each package's own ``consumer_facts.jsonl`` from the bundle's
   ``source_packages.json`` listing and keeps a row only when the scope maps
   its (record set, period) to that package and year, so a row can enter the
   feed from exactly one run;
4. refuses if any scoped pair produced no row, or if two kept rows share an
   ``aggregate_fact_key`` with different bytes;
5. writes ``<out>/consumer_facts.jsonl`` with Chronicle's own line bytes,
   sorted by ``aggregate_fact_key``;
6. runs ``chronicle build-consumer-artifact`` on it into ``<out>/artifact``
   unless ``--skip-artifact`` (the pinned source-authority repair now validates
   all rows; ``docs/us-chronicle-feed-repin.md`` records the previous refusal);
7. writes ``<out>/receipt.json`` with the commands, row count and digests.

Two runs at the same commit produce byte-identical feeds; the receipt's
``facts_sha256`` is what ``us/chronicle_feed.json`` pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

DEFAULT_CHRONICLE_COMMAND = ("uv", "run", "--frozen", "chronicle")
SCOPE_RESOURCE = "chronicle_feed_scope.json"

Pair = tuple[str, str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(row: dict) -> Pair:
    period = row["period"]
    return (row["layout"]["record_set_id"], period["type"], str(period["value"]))


def default_scope_path() -> Path:
    return Path(str(files("microcosm.build.us").joinpath(SCOPE_RESOURCE)))


def load_scope(path: Path) -> dict:
    """Load and check a scope file: unique pairs, consistent run listing."""

    scope = json.loads(path.read_text(encoding="utf-8"))
    if scope.get("version") != 1 or scope.get("country") != "us":
        raise ValueError(f"{path}: not a version-1 US Chronicle feed scope.")
    commit = scope.get("source_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not all(c in "0123456789abcdef" for c in commit)
    ):
        raise ValueError(f"{path}: source_commit must be a 40-hex commit.")
    pairs: dict[Pair, tuple[str, int]] = {}
    runs: dict[int, set[str]] = defaultdict(set)
    for entry in scope["pairs"]:
        pair = (
            entry["record_set_id"],
            entry["period_type"],
            str(entry["period_value"]),
        )
        if pair in pairs:
            raise ValueError(f"{path}: pair {pair} is listed twice.")
        package = entry["package"]
        if not package.startswith("packages/"):
            raise ValueError(f"{path}: package {package!r} is not a packages/ path.")
        year = int(entry["build_year"])
        pairs[pair] = (package, year)
        runs[year].add(package)
    declared = {int(y): set(p) for y, p in scope["runs"].items()}
    if declared != dict(runs):
        raise ValueError(f"{path}: runs listing disagrees with the pairs.")
    if scope.get("pair_count") != len(pairs):
        raise ValueError(f"{path}: pair_count disagrees with the pairs.")
    return {"commit": commit, "pairs": pairs, "runs": dict(runs), "raw": scope}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_checkout(root: Path, commit: str) -> None:
    """Refuse a checkout that is not clean at exactly the scope's commit."""

    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise SystemExit(
            f"Chronicle checkout {root} is at {head}, not the scope's commit {commit}."
        )
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise SystemExit(
            f"Chronicle checkout {root} has uncommitted changes; the feed must be "
            "built from the committed tree."
        )


def run_bundle(
    chronicle_command: tuple[str, ...],
    root: Path,
    year: int,
    packages: list[str],
    out_dir: Path,
) -> list[str]:
    command = [
        *chronicle_command,
        "build-bundle",
        "--year",
        str(year),
        *[arg for package in packages for arg in ("--source", package)],
        "--out",
        str(out_dir),
        "--replace",
    ]
    subprocess.run(command, cwd=str(root), check=True)
    return command


def package_fact_files(bundle_dir: Path, packages: list[str]) -> dict[str, Path]:
    """Map each requested package to its own consumer_facts.jsonl in the bundle."""

    listing = json.loads((bundle_dir / "source_packages.json").read_text())
    by_source = {entry["source"]: entry for entry in listing["source_packages"]}
    resolved: dict[str, Path] = {}
    for package in packages:
        entry = by_source.get(package)
        if entry is None:
            raise SystemExit(
                f"{bundle_dir}: Chronicle did not report a run for {package!r}."
            )
        if not entry.get("valid", False):
            raise SystemExit(f"{bundle_dir}: Chronicle marked {package!r} invalid.")
        facts = Path(entry["outputs"]["consumer_facts"])
        if not facts.is_file():
            raise SystemExit(f"{bundle_dir}: {package!r} wrote no consumer_facts.")
        resolved[package] = facts
    return resolved


def select_rows(
    pairs: dict[Pair, tuple[str, int]],
    fact_files: dict[tuple[str, int], Path],
) -> tuple[list[bytes], dict[Pair, int]]:
    """Keep each row from the one run its pair is scoped to; refuse gaps."""

    kept: dict[str, bytes] = {}
    counts: dict[Pair, int] = defaultdict(int)
    for (package, year), path in sorted(fact_files.items()):
        with path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                line = raw if raw.endswith(b"\n") else raw + b"\n"
                row = json.loads(line)
                pair = _pair(row)
                if pairs.get(pair) != (package, year):
                    continue
                key = row["aggregate_fact_key"]
                previous = kept.get(key)
                if previous is not None and previous != line:
                    raise SystemExit(
                        f"aggregate_fact_key {key} appears twice with different "
                        f"bytes (pair {pair})."
                    )
                kept[key] = line
                counts[pair] += 1
    missing = sorted(pair for pair in pairs if counts.get(pair, 0) == 0)
    if missing:
        raise SystemExit(
            f"{len(missing)} scoped pairs produced no row; first: {missing[:5]}"
        )
    return [kept[key] for key in sorted(kept)], dict(counts)


def build_artifact(
    chronicle_command: tuple[str, ...],
    root: Path,
    facts_path: Path,
    out_dir: Path,
) -> dict[str, object]:
    command = [
        *chronicle_command,
        "build-consumer-artifact",
        "--facts",
        str(facts_path),
        "--out",
        str(out_dir),
        "--replace",
    ]
    subprocess.run(command, cwd=str(root), check=True)
    manifest_path = out_dir / "manifest.json"
    artifact_facts = out_dir / "consumer_facts.jsonl"
    manifest = json.loads(manifest_path.read_text())
    return {
        "command": shlex.join(command),
        "manifest_sha256": _sha256_file(manifest_path),
        "facts_sha256": _sha256_file(artifact_facts),
        "artifact_schema_version": manifest.get("schema_version"),
        "manifest": manifest,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chronicle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-artifact", action="store_true")
    parser.add_argument(
        "--chronicle-command",
        default=shlex.join(DEFAULT_CHRONICLE_COMMAND),
        help="Command prefix that runs the chronicle CLI inside --chronicle-root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scope_path = args.scope if args.scope is not None else default_scope_path()
    scope = load_scope(scope_path)
    root = args.chronicle_root.expanduser().resolve()
    require_checkout(root, scope["commit"])
    out = args.out.expanduser().resolve()
    if out.exists():
        if not args.replace:
            raise SystemExit(f"{out} exists; pass --replace to rebuild into it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    chronicle_command = tuple(shlex.split(args.chronicle_command))

    runs: list[dict[str, object]] = []
    fact_files: dict[tuple[str, int], Path] = {}
    for year, packages in sorted(scope["runs"].items()):
        ordered = sorted(packages)
        bundle_dir = out / "bundles" / str(year)
        command = run_bundle(chronicle_command, root, year, ordered, bundle_dir)
        for package, path in package_fact_files(bundle_dir, ordered).items():
            fact_files[(package, year)] = path
        runs.append({"year": year, "packages": ordered, "command": shlex.join(command)})

    lines, counts = select_rows(scope["pairs"], fact_files)
    facts_path = out / "consumer_facts.jsonl"
    payload = b"".join(lines)
    facts_path.write_bytes(payload)
    facts_sha256 = _sha256_bytes(payload)

    artifact: dict[str, object] | None = None
    if not args.skip_artifact:
        artifact = build_artifact(chronicle_command, root, facts_path, out / "artifact")

    receipt = {
        "scope": str(scope_path),
        "scope_sha256": _sha256_file(scope_path),
        "source_commit": scope["commit"],
        "chronicle_root": str(root),
        "runs": runs,
        "pair_count": len(scope["pairs"]),
        "rows_per_pair_min": min(counts.values()),
        "rows_per_pair_max": max(counts.values()),
        "row_count": len(lines),
        "facts_path": str(facts_path),
        "facts_sha256": facts_sha256,
        "artifact": (
            None
            if artifact is None
            else {k: v for k, v in artifact.items() if k != "manifest"}
        ),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(
        f"{len(lines)} rows for {len(scope['pairs'])} pairs from "
        f"{sum(len(p) for p in scope['runs'].values())} package runs; "
        f"facts sha256 {facts_sha256}; receipt {out / 'receipt.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
