#!/usr/bin/env python3
"""Prove the ACS record fence rewrite on the real staged archive, read-only.

The scan in ``acs_person_coverage_authentication._records`` replaced a
byte-at-a-time loop. The unit suite proves equivalence on invented bytes; this
script proves it on the actual source the native build fences, by running both
implementations over every applicable member of a staged ``csv_pus.zip`` and
comparing a rolling digest of the yielded records, the record count and the
total bytes.

It opens the archive read-only, writes only the receipt path it is given, and
executes no build. Usage:

    .venv/bin/python -I -B experiments/native-verify-once/record_fence_real_archive.py \
        <path-to-csv_pus.zip> <receipt.json> [byte-limit-per-member]
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path[:0] = sorted(
    glob.glob(str(Path(__file__).resolve().parents[2] / "packages/*/src"))
)

from microcosm.build.us_runtime import (  # noqa: E402
    acs_person_coverage_authentication as owner,
)


def reference_records(stream):
    """The exact pre-change body of ``_records``, kept verbatim."""

    record = bytearray()
    quoted, pending_cr, first, token_bytes = False, False, True, 0
    cap = owner.MAX_CSV_HEADER_BYTES
    while block := stream.read(4096):
        for byte in block:
            if pending_cr:
                if byte == 10:
                    owner._require(len(record) < cap, "CSV_RECORD_BYTES")
                    record.append(byte)
                yield bytes(record)
                record.clear()
                first, pending_cr, token_bytes = False, False, 0
                if byte == 10:
                    continue
            cap = owner.MAX_CSV_HEADER_BYTES if first else owner.MAX_RECORD_BYTES
            owner._require(len(record) < cap, "CSV_RECORD_BYTES")
            if byte == 44 and not quoted:
                token_bytes = 0
            else:
                token_bytes += 1
                owner._require(token_bytes <= owner.MAX_TOKEN_BYTES, "CSV_TOKEN_BYTES")
            record.append(byte)
            if byte == 34:
                quoted = not quoted
            if not quoted and byte in (10, 13):
                if byte == 13:
                    pending_cr = True
                else:
                    yield bytes(record)
                    record.clear()
                    first, token_bytes = False, 0
    if record:
        yield bytes(record)


class Bounded(io.RawIOBase):
    """Read-only truncating wrapper; it never writes to the archive."""

    def __init__(self, stream, limit):
        self._stream, self._left = stream, limit

    def read(self, size=-1):
        if self._left <= 0:
            return b""
        want = size if size and size > 0 else 65536
        block = self._stream.read(min(want, self._left))
        self._left -= len(block)
        return block


def fence(archive_path, member, implementation, limit):
    digest, count, total = hashlib.sha256(), 0, 0
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        started = time.process_time()
        for record in implementation(Bounded(raw, limit)):
            digest.update(len(record).to_bytes(8, "little"))
            digest.update(record)
            count += 1
            total += len(record)
        elapsed = time.process_time() - started
    return {
        "records_sha256": digest.hexdigest(),
        "records": count,
        "bytes": total,
        "cpu_seconds": round(elapsed, 3),
    }


def main():
    archive_path = Path(sys.argv[1])
    receipt_path = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 62
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info.filename
            for info in archive.infolist()
            if info.filename.casefold().startswith("psam_pus")
        ]
        sizes = {info.filename: info.file_size for info in archive.infolist()}
    rows, identical = [], True
    for member in sorted(members):
        old = fence(archive_path, member, reference_records, limit)
        new = fence(archive_path, member, owner._records, limit)
        same = {k: v for k, v in old.items() if k != "cpu_seconds"} == {
            k: v for k, v in new.items() if k != "cpu_seconds"
        }
        identical &= same
        rows.append(
            {
                "member": member,
                "member_bytes": sizes[member],
                "byte_limit": min(limit, sizes[member]),
                "identical": same,
                "byte_loop": old,
                "chunked_scan": new,
                "speedup": round(old["cpu_seconds"] / max(new["cpu_seconds"], 1e-9), 1),
            }
        )
    payload = {
        "protocol": "owned.us-native-record-fence-parity.v1",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256_of_central_directory_names": hashlib.sha256(
            "\n".join(sorted(members)).encode()
        ).hexdigest(),
        "identical": identical,
        "members": rows,
        "note": (
            "Descriptive parity measurement on a read-only staged archive. Not a "
            "build, not a certification, not release eligible."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if identical else 1


if __name__ == "__main__":
    os.environ["TZ"] = "UTC"
    raise SystemExit(main())
