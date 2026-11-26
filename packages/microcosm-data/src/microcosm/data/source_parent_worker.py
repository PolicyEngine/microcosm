"""Fixed stdlib bootstrap for local source-parent replay, never a public worker API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    worker = Path(__file__)
    helper = worker.with_name("source_parent_replay.py")
    for path, digest in (
        (worker, args.worker_sha256),
        (helper, args.helper_sha256),
        (args.config, args.config_sha256),
    ):
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("replay bootstrap inputs must be local regular files")
        if _digest(path) != digest:
            raise ValueError("replay bootstrap input changed before execution")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("worker output must be fresh")
    spec = importlib.util.spec_from_file_location("_source_parent_fixed_helper", helper)
    if spec is None or spec.loader is None:
        raise ValueError("fixed replay helper could not be loaded")
    import sys

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._worker_main(module._read_json(args.config))
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
