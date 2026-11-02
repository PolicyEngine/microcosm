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
shutil.copy(SP + "/run/noise_ci.json", E + "/outputs")
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


H = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
digest = {
    l.split()[1].split("/snapshots/")[1].split("/")[0][:8]
    + "/"
    + os.path.basename(l.split()[1]): l.split()[0]
    for l in open(SP + "/run/input_sha256.txt")  # noqa: E741
}
data = {}
for tag, rev, name in (
    ("default", "9a814a3b", "populace_us_2024.h5"),
    ("dense", "15cefac4", None),
    ("acs_local", "725d8999", "populace_us_2024_acs_local.h5"),
):
    d = glob.glob(H + rev + "*")[0]
    p = glob.glob(d + "/*.h5")
    p = [
        x for x in p if (name and x.endswith(name)) or (not name and "calib" not in x)
    ][0]
    t = os.path.realpath(p)
    data[tag] = {
        "repo": "policyengine/populace-us",
        "revision": os.path.basename(d),
        "file": os.path.basename(p),
        "sha256": digest[os.path.basename(d)[:8] + "/" + os.path.basename(p)],
        "bytes": os.path.getsize(t),
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
ext = {}
BR = os.path.expanduser(
    "~/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/runs/"
)
for n in sorted(pd.read_csv(SP + "/run/acs_solves.csv").run):
    ext["_build_artifacts/acs-local-l2-basis-20260928/runs/" + n + "/weights.npz"] = (
        sha(BR + n + "/weights.npz")
    )
for f in sorted(
    glob.glob(
        os.path.expanduser("~/reviews/pe30-population-2026-09-27/out/calib_resolve/*")
    )
):
    if os.path.isfile(f):
        ext[
            "reviews/pe30-population-2026-09-27/out/calib_resolve/"
            + os.path.basename(f)
        ] = sha(f)
ext["asec/raw_asec_2022_2025.parquet"] = sha(SP + "/asec/raw_asec_2022_2025.parquet")
for f in sorted(glob.glob(SP + "/dash/*.json")):
    ext["dash/" + os.path.basename(f)] = sha(f)
ext["populace_us_2024_calibration.npz, revision 15cefac4"] = digest[
    "15cefac4/populace_us_2024_calibration.npz"
]
r = {
    "purpose": "Receipts for the state child SPM diagnosis. Aggregates by state and national totals only; no household rows.",
    "date": "2026-11-01",
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
