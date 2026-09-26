"""Compare a run's preparation receipt against a recovered pilot artifact.

Reassembles the roster transport's segments from its own ``header.json``,
verifies each segment against its recorded digest and the whole against the
stream digest, and reports, block by block, what is byte-identical and what is
not. Reads only; writes only the named output file. Not a build, not a
certification, not release eligible.

    <venv>/bin/python receipt_parity.py <spill-dir> <pilot.json> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


def _canonical(value):
    return (
        json.JSONEncoder(
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        .encode(value)
        .encode("utf-8")
    )


def _differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += _differences(left.get(key), right.get(key), f"{path}.{key}")
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out += _differences(a, b, f"{path}[{index}]")
        return out
    if left != right:
        return [{"path": path, "run": str(left)[:80], "pilot": str(right)[:80]}]
    return []


def main():
    spill = pathlib.Path(sys.argv[1])
    pilot_path = pathlib.Path(sys.argv[2])
    out = pathlib.Path(sys.argv[3])
    header = json.loads((spill / "header.json").read_bytes())
    segments = []
    for sha256, size in header["segments"]:
        raw = (spill / (sha256 + ".segment")).read_bytes()
        segments.append(
            {
                "sha256": sha256,
                "size": size,
                "verified": hashlib.sha256(raw).hexdigest() == sha256
                and len(raw) == size,
            }
        )
    payload = b"".join(
        (spill / (sha256 + ".segment")).read_bytes()
        for sha256, _size in header["segments"]
    )
    run = json.loads(payload)
    pilot = json.loads(pilot_path.read_bytes())
    blocks = {}
    for key in sorted(set(run) | set(pilot)):
        left = _canonical(run[key]) if key in run else b""
        right = _canonical(pilot[key]) if key in pilot else b""
        blocks[key] = {
            "run_bytes": len(left),
            "pilot_bytes": len(right),
            "identical": left == right,
        }
    implementation = run["producer"]["implementation"]
    pilot_implementation = pilot["producer"]["implementation"]
    result = {
        "spill": str(spill),
        "pilot_artifact": str(pilot_path),
        "pilot_artifact_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        "header": header,
        "segments_verified": all(segment["verified"] for segment in segments),
        "reassembled_bytes": len(payload),
        "reassembled_sha256": hashlib.sha256(payload).hexdigest(),
        "reassembled_matches_header": hashlib.sha256(payload).hexdigest()
        == header["sha256"]
        and len(payload) == header["size"],
        "blocks": blocks,
        "source_file_differences": _differences(
            run["source_files"], pilot["source_files"], "source_files"
        ),
        "modules_whose_bytes_differ": sorted(
            name
            for name in set(implementation) & set(pilot_implementation)
            if implementation[name] != pilot_implementation[name]
        ),
        "modules_only_in_run": sorted(set(implementation) - set(pilot_implementation)),
        "modules_only_in_pilot": sorted(
            set(pilot_implementation) - set(implementation)
        ),
        "release_eligible": False,
        "scope": (
            "Block-by-block parity of one run's preparation receipt against one "
            "recovered pilot artifact. Not a build, not a certification, not "
            "release eligible."
        ),
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    for key, row in blocks.items():
        print(
            f"{key:<22} run {row['run_bytes']:>9}  "
            f"pilot {row['pilot_bytes']:>9}  identical {row['identical']}"
        )
    print("segments verified:", result["segments_verified"])
    print("reassembled matches header:", result["reassembled_matches_header"])


if __name__ == "__main__":
    main()
