"""One ACS-local state from the per-state split: baseline and the 2021 CTC reform for 2025, keyed for clustering.

Every donor in the file leaves its parent links blank. The model reads 0 as unknown (parent_1_id.py), so blanks take 0.
Usage: acs_rerun.py <fips> <tag>. Tag acs keeps the published rent. Tag acsrent gives donors the base record's right rent
and ACS renter heads twelve times native monthly contract rent, with weights unchanged.
FILL overrides the fill for the checks in acs_rerun.sh: -1 repeats the earlier runs, and none leaves the blanks in place.
Such a check skips the reform unless WITH_REFORM=1, which runs it too: the South Dakota trace compares reform effects that way.
Stops itself with exit 75 if peak RSS passes RSS_CAP_GIB (default 8).

Texas passed 8 GiB holding the baseline and then the reform in one image. So once run/two_part_ok exists (two_part_check.sh
writes it after two steps reproduce one process exactly), a state whose person split reaches TWO_PART_MB (default 12) runs in
two steps. The baseline step writes its results to <output>.base and re-executes this process for the reform step, which reads
them back: the process holds one simulation at a time, under the same stop. A run stopped after the baseline step starts at the
reform. TWO_PART=1 or 0 forces either way, and OUT sends the output to another folder, for the check.
While run/hold exists, a run waits at its start, before any heavy import, unless IGNORE_HOLD is set: two_part_check.sh
holds the rerun between states that way.
HOUSEHOLDS names a .npy file of household ids, and the run then covers only those households: acs_parts.sh runs a state
in parts that way.
"""

import gc
import os
import resource
import sys
import threading
import time
import warnings

SP = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f"{SP}/run/hold") and not os.environ.get("IGNORE_HOLD"):
    # two_part_check.sh holds the rerun between states: wait here, before any heavy import, until it lets go.
    mark = f"{SP}/run/held.{os.getpid()}"
    open(mark, "w").close()
    while os.path.exists(f"{SP}/run/hold"):
        time.sleep(15)
    os.remove(mark)
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, SP)
fips, TAG = int(sys.argv[1]), sys.argv[2]
FILL = os.environ.get("FILL", "0")
CAP = float(os.environ.get("RSS_CAP_GIB", "8")) * 2**30


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def watch():
    while True:
        time.sleep(2)
        if peak() > CAP:
            print(f"rss stop {fips} {TAG} peak {peak() / 2**30:.2f} GiB", flush=True)
            os._exit(75)


threading.Thread(target=watch, daemon=True).start()
import extract_state as ex
from policyengine_core.data import Dataset
from policyengine_core.reforms import Reform
from policyengine_us import CountryTaxBenefitSystem, Microsimulation

Y = 2025
checking = FILL != "0"
with_reform = not checking or os.environ.get("WITH_REFORM") == "1"
folder = os.environ.get("OUT") or (
    f"{SP}/run/check_fill{FILL}" if checking else f"{SP}/run/{TAG}"
)
os.makedirs(folder, exist_ok=True)
target = f"{folder}/{fips:02d}.parquet"
if os.path.exists(target):
    sys.exit(0)
partial = f"{target}.base"
forced = os.environ.get("TWO_PART")
if checking:
    two_part = False
elif forced in ("0", "1"):
    two_part = forced == "1"
else:
    size_mb = os.path.getsize(f"{SP}/run/acs_split/{fips:02d}-person.pkl.gz") / 1e6
    two_part = os.path.exists(f"{SP}/run/two_part_ok") and size_mb >= float(
        os.environ.get("TWO_PART_MB", "12")
    )
step = ("reform" if os.path.exists(partial) else "baseline") if two_part else "both"
t0 = float(os.environ.get("TWO_PART_T0", time.time()))
ex.known = set(CountryTaxBenefitSystem().variables)
ex.tables = {
    n: pd.read_pickle(f"{SP}/run/acs_split/{fips:02d}-{n}.pkl.gz")
    for n in ("household", "person", *ex.GROUPS)
}
if TAG == "acsrent":
    h, p = ex.tables["household"], ex.tables["person"]
    right = (
        pd.read_parquet(f"{SP}/run/rent_right.parquet")
        .set_index("household_source_id")
        .rent_right
    )
    acs = h.household_support_channel.astype(str).str.startswith("acs").values
    hr = pd.Series(
        np.where(
            acs,
            12
            * pd.to_numeric(h.acs_monthly_contract_rent, errors="coerce")
            .fillna(0)
            .values,
            h.household_source_id.map(right).values,
        ),
        index=h.household_id.values,
    )
    assert hr.notna().all()
    head = p.is_household_head.astype(bool).values
    rent = np.zeros(len(p))
    rent[head] = p.person_household_id[head].map(hr).values
    p["pre_subsidy_rent"] = rent
if os.environ.get("HOUSEHOLDS"):
    # One part of the state (acs_parts.sh): only these households reach the simulation.
    households = ex.tables["household"]
    ex.tables["household"] = households[
        households.household_id.isin(np.load(os.environ["HOUSEHOLDS"]))
    ]
frame = ex.state_file(fips)
ex.tables = None
gc.collect()
if FILL != "none":
    for c in frame.columns:
        if c.startswith("parent_") and c.endswith("_id__2024"):
            frame[c] = frame[c].fillna(int(FILL)).astype(np.int64)
if "spm_unit_spm_universe_status__2024" not in frame:
    # policyengine-us 2.90.2 ignores this column, so every SPM unit stays in the run.
    frame["spm_unit_spm_universe_status__2024"] = "INCLUDED"
d = "2025-01-01.2025-12-31"
ref = Reform.from_dict(
    {
        "gov.irs.credits.ctc.amount.arpa[0].amount": {d: 3600},
        "gov.irs.credits.ctc.amount.arpa[1].amount": {d: 3000},
        "gov.irs.credits.ctc.refundable.fully_refundable": {d: True},
        "gov.irs.credits.ctc.phase_out.arpa.in_effect": {d: True},
    },
    country_id="us",
)


def unit(sim, name, period=Y):
    return np.asarray(sim.calculate(name, period, map_to="spm_unit").values, float)


if step == "reform":
    res = pd.read_parquet(partial)
else:
    base = Microsimulation(dataset=Dataset.from_dataframe(frame, 2024))
    ids = base.calculate("spm_unit_id", 2024).values
    age = base.calculate("age", 2024).values
    pu = base.calculate("person_spm_unit_id", 2024).values
    res = {
        "spm_unit_id": ids,
        "kids": pd.Series(age < 18).groupby(pu).sum().reindex(ids).values,
    }
    res["poor"] = unit(base, "spm_unit_is_in_spm_poverty")
    res["net"] = unit(base, "spm_unit_net_income")
    res["thr"] = unit(base, "spm_unit_spm_threshold")
    if TAG == "acsrent":
        res["snap"] = unit(base, "snap", 2024)
        res["house"] = unit(base, "spm_unit_capped_housing_subsidy", 2024)
    del base
    gc.collect()
if step == "baseline":
    pd.DataFrame(res).to_parquet(partial + ".tmp", compression="zstd")
    os.replace(partial + ".tmp", partial)
    print(
        fips,
        TAG,
        "baseline step seconds",
        round(time.time() - t0),
        "peak GiB",
        round(peak() / 2**30, 2),
        flush=True,
    )
    os.execve(
        sys.executable,
        sys.orig_argv,
        dict(os.environ, TWO_PART_T0=repr(t0), TWO_PART_PEAK=str(peak())),
    )
if with_reform:
    sim = Microsimulation(dataset=Dataset.from_dataframe(frame, 2024), reform=ref)
    res["poor_r"] = unit(sim, "spm_unit_is_in_spm_poverty")
    if step == "reform":
        assert np.array_equal(
            sim.calculate("spm_unit_id", 2024).values, res["spm_unit_id"].values
        )
    del sim
    gc.collect()
pd.DataFrame(res).to_parquet(target + ".tmp", compression="zstd")
os.replace(target + ".tmp", target)
if step == "reform":
    os.remove(partial)
top = max(peak(), int(os.environ.get("TWO_PART_PEAK", "0")))
print(
    fips,
    TAG,
    "fill",
    FILL,
    "seconds",
    round(time.time() - t0),
    "peak GiB",
    round(top / 2**30, 2),
    *(["two steps"] if two_part else []),
    flush=True,
)
