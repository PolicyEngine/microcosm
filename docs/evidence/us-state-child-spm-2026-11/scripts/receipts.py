"""Write outputs/ and receipts.json for the evidence folder."""

import glob
import hashlib
import json
import os
import shutil
import subprocess

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
E = os.path.expanduser(
    "~/PolicyEngine/microcosm/_worktrees/state-child-spm/docs/evidence/us-state-child-spm-2026-11"
)
os.makedirs(E + "/outputs", exist_ok=True)
for f in glob.glob(E + "/outputs/*"):
    if os.path.isfile(f):
        os.remove(f)
os.makedirs(E + "/outputs/logs", exist_ok=True)
for f in glob.glob(E + "/outputs/logs/*"):
    os.remove(f)
for f in sorted(glob.glob(SP + "/run/*.csv")):
    shutil.copy(f, E + "/outputs")
for f in sorted(glob.glob(SP + "/run/noise_ci*.json")):
    shutil.copy(f, E + "/outputs")
for f in sorted(glob.glob(SP + "/run/logs/*.txt")):
    shutil.copy(f, E + "/outputs/logs")
for f in ("census_child_spm.json", "state_baseline.json"):
    shutil.copy(f"{SP}/{f}", E + "/outputs")
shutil.copy(SP + "/asec/state_raw.csv", E + "/outputs/asec_state_raw.csv")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as x:
        for b in iter(lambda: x.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# A host restart wiped some inputs on 2026-11-05. Inputs this host no longer holds keep the digests already recorded;
# inputs it still holds must match them. The rerun checked the refetched ACS-local file against its recorded digest.
old = json.load(open(E + "/receipts.json"))
data = old["data"]
data["acs_local"]["rerun"] = {
    "read": "2026-11-06, by acs_split.py through HTTP range requests (remote_fixed.py), with no local copy",
    "identity": "the Hub's x-linked-etag and x-linked-size for the pinned revision matched sha256 and bytes above",
    "blank_parent_links": "fill 0, which the model reads as unknown; outputs/*_fill_minus1.* keep the earlier -1 fill's figures",
}
v = subprocess.run(
    [
        SP + "/env/.venv/bin/python",
        "-c",
        "import importlib.metadata as m;print(m.version('policyengine-us'),m.version('policyengine-core'))",
    ],
    capture_output=True,
    text=True,
).stdout.split()
ext = dict(old["external_inputs"])


def check(key, path):
    if os.path.exists(path):
        d = sha(path)
        assert ext.get(key, d) == d, key
        ext[key] = d


BR = os.path.expanduser(
    "~/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/runs/"
)
for n in sorted(pd.read_csv(SP + "/run/acs_solves.csv").run):
    check(
        "_build_artifacts/acs-local-l2-basis-20260928/runs/" + n + "/weights.npz",
        BR + n + "/weights.npz",
    )
for f in sorted(
    glob.glob(
        os.path.expanduser("~/reviews/pe30-population-2026-09-27/out/calib_resolve/*")
    )
):
    if os.path.isfile(f):
        check(
            "reviews/pe30-population-2026-09-27/out/calib_resolve/"
            + os.path.basename(f),
            f,
        )
r = {
    "purpose": "Receipts for the state child SPM note. Aggregates by state and national totals only; no household rows.",
    "date": "2026-11-07",
    "model": {
        "policyengine_us": v[0],
        "policyengine_us_commit": "469ad12cb957",
        "policyengine_core": v[1],
    },
    "benchmark": "Census P60-290 Table 18, 2023 to 2025 average, under 18",
    "data": data,
    "external_inputs": ext,
    "note": "Poverty rates stay holdouts: nothing here entered a calibration target.",
    "files": {},
}
for f in sorted(
    x
    for x in glob.glob(E + "/outputs/**", recursive=True) + glob.glob(E + "/scripts/*")
    if os.path.isfile(x)
):
    r["files"][os.path.relpath(f, E)] = {"sha256": sha(f), "bytes": os.path.getsize(f)}
json.dump(r, open(E + "/receipts.json", "w"), indent=2)
open(E + "/receipts.json", "a").write("\n")
print(
    len(r["files"]), "files", sum(x["bytes"] for x in r["files"].values()) / 1e6, "MB"
)
print(json.dumps(data, indent=0)[:600])
