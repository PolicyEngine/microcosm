"""Simulate cuts already on disk, smallest first. Usage: runcuts.py <tag> <workers>"""

import os
import sys
from pathlib import Path

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
if __name__ == "__main__":
    from cutrun import Pool, run

    tag, workers = sys.argv[1], int(sys.argv[2])
    d = Path(SP) / "run" / f"{tag}_cuts"
    out = Path(SP) / "run" / tag
    out.mkdir(parents=True, exist_ok=True)
    fs = sorted(d.glob("*.pkl"), key=lambda q: q.stat().st_size)
    with Pool(workers, maxtasksperchild=1) as pool:
        for done, sec in pool.imap_unordered(
            run, [(int(q.stem), str(d), str(out), (2024, 2025)) for q in fs]
        ):
            print(done, sec, flush=True)
