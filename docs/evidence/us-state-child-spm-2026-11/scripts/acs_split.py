"""Split the ACS-local file by state, one table per file, as state_baseline.py's split_by_state does.

The source is a local H5 path or an https URL. A URL reads through range requests (remote_fixed.py), so no copy of the file lands anywhere.
The person table reads in chunks, so the whole table never sits in memory, and rows keep file order within each state.
A group that spans states goes to every state it touches, so no member loses its group's row.
With SPLIT_GATE set to "<GiB> <GB> <log>", it waits at host_wait.py before each table and each chunk.
A run that stops keeps its finished chunks, and the next run with the same chunk size starts after them.
Stops itself with exit 75 if peak RSS passes RSS_CAP_GIB (default 8).
Usage: acs_split.py <h5 or url> <outdir>
A URL also needs SPLIT_SIZE and SPLIT_SHA256, which it checks against what the URL reports.
"""

import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
source, out = sys.argv[1], sys.argv[2]
GROUPS = ("tax_unit", "spm_unit", "family", "marital_unit")
CHUNK = int(os.environ.get("SPLIT_CHUNK", "400000"))
GATE = os.environ.get("SPLIT_GATE", "").split()
GZ = {"method": "gzip", "compresslevel": 1}
finished = f"{out}/parts.finished"
if os.path.exists(finished):
    shutil.rmtree(finished)
    print("an earlier run finished this split", flush=True)
    sys.exit(0)
os.makedirs(f"{out}/parts", exist_ok=True)
t0 = time.time()
CAP = float(os.environ.get("RSS_CAP_GIB", "8"))


def peak():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)


def watch():
    while True:
        time.sleep(2)
        if peak() > CAP:
            print("rss stop, peak GiB", peak(), flush=True)
            os._exit(75)


def gate():
    if GATE:
        subprocess.run(["python3", "-I", f"{HERE}/host_wait.py", *GATE], check=True)


def open_store():
    if not source.startswith("https://") and not source.startswith("http://"):
        return pd.HDFStore(source, "r")
    sys.path.insert(0, HERE)
    from remote_fixed import RemoteStore

    return RemoteStore(
        source,
        size=int(os.environ["SPLIT_SIZE"]),
        sha256=os.environ["SPLIT_SHA256"],
        rate=int(os.environ.get("SPLIT_RATE", "3000000")),
        patience=int(os.environ.get("SPLIT_PATIENCE", "7200")),
    )


def nrows(store, key):
    if not isinstance(store, pd.HDFStore):
        return store.nrows(key)
    node = store.get_node(key)
    return node.table.nrows if "table" in node else node.axis1.shape[0]


def replace_when_written(frame, path):
    frame.to_pickle(f"{path}.tmp", compression=GZ)
    os.replace(f"{path}.tmp", path)


threading.Thread(target=watch, daemon=True).start()


with open_store() as store:
    n = nrows(store, "person")
    print("source", type(store).__name__, "people", n, "chunk", CHUNK, flush=True)
    # A stopped run's finished chunks count only if the next run cuts the table the same way.
    plan, wanted = f"{out}/parts/plan.json", {"people": int(n), "chunk": CHUNK}
    if os.path.exists(plan):
        with open(plan) as f:
            held = json.load(f)
        if held != wanted:
            sys.exit(
                f"{out}/parts holds a run cut as {held}, and this run wants {wanted}; remove it to start over"
            )
    else:
        with open(f"{plan}.tmp", "w") as f:
            json.dump(wanted, f)
        os.replace(f"{plan}.tmp", plan)
    state_file = f"{out}/parts/states.pkl"
    if os.path.exists(state_file):
        states = pd.read_pickle(state_file)
    else:
        gate()
        households = store.select("household")
        print("households", households.shape, "peak GiB", peak(), flush=True)
        for state, rows in households.groupby("state_fips", sort=False):
            replace_when_written(rows, f"{out}/{int(state):02d}-household.pkl.gz")
        states = households.set_index("household_id")["state_fips"]
        del households
        states.to_pickle(f"{state_file}.tmp")
        os.replace(f"{state_file}.tmp", state_file)
    ids = {g: [] for g in GROUPS}
    fips_all, columns, seen = [], None, 0
    for i, a in enumerate(range(0, n, CHUNK)):
        marker = f"{out}/parts/chunk-{i:05d}.npz"
        if os.path.exists(marker):
            with np.load(marker) as kept:
                fips = kept["fips"]
                for g in GROUPS:
                    ids[g].append(kept[g])
            seen += len(fips)
            fips_all.append(fips)
            print("chunk", i, "already done", flush=True)
            continue
        gate()
        people = store.select("person", start=a, stop=min(a + CHUNK, n))
        columns = people.shape[1]
        seen += len(people)
        fips = people["person_household_id"].map(states)
        assert fips.notna().all()
        for state, rows in people.groupby(fips.values, sort=False):
            rows.to_pickle(
                f"{out}/parts/{int(state):02d}-person-{i:05d}.pkl.gz", compression=GZ
            )
        chunk_ids = {g: people[f"person_{g}_id"].values for g in GROUPS}
        for g in GROUPS:
            ids[g].append(chunk_ids[g])
        fips_all.append(fips.values)
        np.savez(f"{marker}.tmp.npz", fips=fips.values, **chunk_ids)
        os.replace(f"{marker}.tmp.npz", marker)
        del people, fips, chunk_ids
        print(
            "chunk",
            i,
            "seconds",
            round(time.time() - t0),
            "peak GiB",
            peak(),
            flush=True,
        )
    assert seen == n
    print("people", seen, "person columns", columns, flush=True)
    fips = np.concatenate(fips_all)
    for g in GROUPS:
        gate()
        pairs = pd.DataFrame(
            {"id": np.concatenate(ids[g]), "state": fips}
        ).drop_duplicates()
        print(
            g,
            "groups",
            pairs.id.nunique(),
            "spanning states",
            int(pairs.id.duplicated().sum()),
            flush=True,
        )
        table = store.select(g)
        for state, members in pairs.groupby("state", sort=False):
            replace_when_written(
                table[table[f"{g}_id"].isin(members.id)],
                f"{out}/{int(state):02d}-{g}.pkl.gz",
            )
        del table, pairs, members
        print(g, "peak GiB", peak(), flush=True)
for state in sorted({int(s) for s in states.unique()}):
    final, merged = (
        f"{out}/{state:02d}-person.pkl.gz",
        f"{out}/parts/merged-{state:02d}",
    )
    parts = sorted(glob.glob(f"{out}/parts/{state:02d}-person-*.pkl.gz"))
    if not os.path.exists(merged):
        replace_when_written(pd.concat([pd.read_pickle(p) for p in parts]), final)
        open(merged, "w").close()
    for p in parts:
        os.remove(p)
# The rename settles that the split is whole, so no stop can leave a chunk's marker without its parts for a later run to trust.
os.replace(f"{out}/parts", finished)
shutil.rmtree(finished)
print(
    "done files",
    len(glob.glob(f"{out}/*.pkl.gz")),
    "seconds",
    round(time.time() - t0),
    "peak GiB",
    peak(),
    flush=True,
)
