"""Compare pandas with the range reader on the real ACS-local file, without holding the file.

Each byte the range reader fetches also lands at its own offset in a sparse local file. Pandas then reads the same rows from that file, and the two
must return the same frames. A read that needed any byte the range reader did not fetch sees zeros there, and shows up as a failure or a difference.
The sparse file stays under SPARSE_CAP bytes on disk and goes away at the end. A last read of one full chunk measures its time and memory, and writes nothing.
Usage: real_fixed_diff.py <url> <size> <sha256> <sparse path>
"""

import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from remote_fixed import RemoteFile, RemoteStore, check_identity

SPARSE_CAP = 10**9


def peak():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)


class Recording(RemoteFile):
    def __init__(self, url, size, path):
        super().__init__(url, size)
        self.sparse = open(path, "wb")
        self.sparse.truncate(size)
        self.recording = True

    def _fetch(self, start, view):
        super()._fetch(start, view)
        if self.recording:
            self.sparse.seek(start)
            self.sparse.write(view)
            if os.fstat(self.sparse.fileno()).st_blocks * 512 > SPARSE_CAP:
                raise RuntimeError("sparse file passed its cap")


url, size, sha, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
check_identity(url, size, sha)
print("identity: the URL reports the expected sha256 and size", flush=True)
remote = Recording(url, size, path)
frames = {}
try:
    with RemoteStore(remote) as mine:
        # PyTables reads each node's metadata as it walks the file, so touch all of it.
        def touch(_, node):
            for key in node.attrs:
                node.attrs[key]

        mine.h5.visititems(touch)
        print("metadata bytes", remote.bytes_read, flush=True)
        n_people, n_households = mine.nrows("person"), mine.nrows("household")
        cases = [
            ("family", None, None),
            ("household", 0, 20000),
            ("household", n_households // 2, n_households // 2 + 20000),
            ("person", 0, 20000),
            ("person", n_people // 2, n_people // 2 + 20000),
            ("person", n_people - 20000, n_people),
        ]
        for key, start, stop in cases:
            t0 = time.time()
            frames[(key, start, stop)] = mine.select(key, start=start, stop=stop)
            print(
                "range read",
                key,
                start,
                stop,
                frames[(key, start, stop)].shape,
                "seconds",
                round(time.time() - t0),
                "bytes so far",
                remote.bytes_read,
                "peak GiB",
                peak(),
                flush=True,
            )
        remote.sparse.flush()
        print(
            "sparse file on disk, MB",
            round(os.fstat(remote.sparse.fileno()).st_blocks * 512 / 1e6),
            flush=True,
        )
        failures = 0
        with pd.HDFStore(path, "r") as store:
            for (key, start, stop), frame in frames.items():
                theirs = store.select(key, start=start, stop=stop)
                try:
                    pd.testing.assert_frame_equal(frame, theirs, check_exact=True)
                    print(
                        "same as pandas",
                        key,
                        start,
                        stop,
                        frame.shape,
                        dict(frame.dtypes.astype(str).value_counts()),
                        flush=True,
                    )
                except AssertionError as error:
                    failures += 1
                    print("DIFFERS", key, start, stop, str(error)[:800], flush=True)
                del theirs
        print(
            "people",
            n_people,
            "households",
            n_households,
            "columns",
            frames[("person", 0, 20000)].shape[1],
            flush=True,
        )
        frames.clear()
        remote.recording = False
        remote.sparse.close()
        os.remove(path)
        print("sparse file removed", flush=True)
        t0, b0 = time.time(), remote.bytes_read
        chunk = mine.select("person", start=1_000_000, stop=1_200_000)
        print(
            "one 200,000-row chunk",
            chunk.shape,
            "seconds",
            round(time.time() - t0),
            "bytes",
            remote.bytes_read - b0,
            "peak GiB",
            peak(),
            flush=True,
        )
        print("requests", remote.requests, "bytes", remote.bytes_read, flush=True)
finally:
    if os.path.exists(path):
        remote.sparse.close()
        os.remove(path)
print("failures", failures, flush=True)
sys.exit(1 if failures else 0)
