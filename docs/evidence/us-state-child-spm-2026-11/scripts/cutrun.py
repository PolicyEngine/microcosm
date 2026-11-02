"""Cut a Populace file into one flat frame per state in one process, then simulate each state in workers.

The flat frame matches state_baseline.py's state_file. Workers load only their state's frame, so a large file needs
no whole-table load in each worker. Usage: cutrun.py <tag> <h5> <workers> [years] [order]
"""

import os
import sys
import time
import warnings
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import extract_state as ex
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation

DATA_YEAR = 2024


def cut(h5, d):
    ex.load(h5)
    t = ex.tables
    print({k: v.shape for k, v in t.items()}, flush=True)
    fips = sorted(int(v) for v in t["household"]["state_fips"].unique())
    only = os.environ.get("ONLY_FIPS")
    if only:
        fips = [f for f in fips if f in {int(v) for v in only.split(",")}]
    for f in fips:
        p = d / f"{f:02d}.pkl"
        if p.exists():
            continue
        frame = ex.state_file(f)
        frame.to_pickle(p)
    return fips


def fix(frame):
    """Blank parent links stored in the ACS-local file fail integer casts from pe-us 2.30.2; -1 means no parent."""
    for c in frame.columns:
        if c.startswith("parent_") and c.endswith("_id__2024"):
            frame[c] = frame[c].fillna(-1).astype(np.int64)
    return frame


def run(a):
    fips, d, out, years = a
    target = Path(out) / f"{fips:02d}.parquet"
    if target.exists():
        return fips, "cached"
    started = time.time()
    frame = pd.read_pickle(Path(d) / f"{fips:02d}.pkl")
    frame = fix(frame)
    sim = Microsimulation(dataset=Dataset.from_dataframe(frame, DATA_YEAR))
    ids = sim.calculate("spm_unit_id", DATA_YEAR).values
    res = {"spm_unit_id": ids}
    age = sim.calculate("age", DATA_YEAR).values
    pu = sim.calculate("person_spm_unit_id", DATA_YEAR).values
    res["n_children"] = pd.Series(age < 18).groupby(pu).sum().reindex(ids).values
    res["n_people"] = pd.Series(age).groupby(pu).size().reindex(ids).values
    extra = os.environ.get("REFORM_YEAR")  # noqa: F841
    for y in years:
        for n in ex.SPM_VARS + ex.UP_VARS:
            res[f"{n}__{y}"] = np.asarray(
                sim.calculate(n, y, map_to="spm_unit").values, dtype=float
            )
    res = pd.DataFrame(res)
    res["state_fips"] = fips
    res.to_parquet(target)
    return fips, round(time.time() - started)


if __name__ == "__main__":
    tag, h5, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    years = tuple(
        int(y) for y in (sys.argv[4] if len(sys.argv) > 4 else "2024,2025").split(",")
    )
    d = Path(SP) / "run" / f"{tag}_cuts"
    d.mkdir(parents=True, exist_ok=True)
    out = Path(SP) / "run" / tag
    out.mkdir(parents=True, exist_ok=True)
    fips = cut(h5, d)
    ex.tables = None
    sizes = {f: (d / f"{f:02d}.pkl").stat().st_size for f in fips}
    order = sorted(fips, key=sizes.get)
    print(
        "policyengine-us",
        version("policyengine-us"),
        "states",
        len(order),
        "cut GB",
        round(sum(sizes.values()) / 1e9, 2),
        flush=True,
    )
    with Pool(workers, maxtasksperchild=1) as pool:
        for done, sec in pool.imap_unordered(
            run, [(f, str(d), str(out), years) for f in order]
        ):
            print(done, sec, flush=True)
