"""microcosm#968 plan item 4: does WHO claims UC (at a fixed claimant count) move measured poverty? v20 at 2024, calibrated weights."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from policyengine_uk import Microsimulation
PATH="/Users/mariajuaristi/Desktop/PolicyEngine/runs/uk-623-first-calibrated/spine-assessment-v20/microcosm_uk_2024.h5"; YEAR=2024
def wmedian(x,w):
    o=np.argsort(x); c=np.cumsum(w[o]); return float(x[o][np.searchsorted(c,c[-1]/2)])
with pd.HDFStore(PATH,"r") as st: person=st["person"]; h5hh=st["household"]; bu=st["benunit"]
hid=h5hh["household_id"].to_numpy(); p_idx=pd.Index(hid).get_indexer(person["person_household_id"].to_numpy())
bid=bu["benunit_id"].to_numpy(); b_of_p=pd.Index(bid).get_indexer(person["person_benunit_id"].to_numpy())
# household index of each benunit (first member)
first_p=pd.Series(np.arange(len(person))).groupby(b_of_p).first().reindex(np.arange(len(bid))).to_numpy(); b_hh=p_idx[first_p]
base_claim=bu["would_claim_uc"].to_numpy(bool)
anchor=np.zeros(len(bid),bool); anchor[np.unique(b_of_p[pd.to_numeric(person["universal_credit_reported"],errors="coerce").fillna(0).to_numpy()>0])]=True
def run(claim, label):
    sim=Microsimulation(dataset=PATH)
    sim.set_input("would_claim_uc", YEAR, claim.astype(bool))
    hh=lambda v: np.asarray(sim.calculate(v,YEAR,map_to="household").values,float)
    hw=hh("household_weight"); pw=hw[p_idx]; eq=hh("equiv_hbai_household_net_income")[p_idx]; thr=float(hh("poverty_threshold_bhc")[0])
    uc_b=np.asarray(sim.calculate("universal_credit",YEAR).values,float); bw=np.asarray(sim.calculate("benunit_weight",YEAR).values,float)
    med=wmedian(eq,pw); age=np.asarray(sim.calculate("age",YEAR).values,float); child=age<18
    r={"label":label,"claim_true_units":int(claim.sum()),"uc_families_m":float(bw[uc_b>0].sum()/1e6),"uc_bn":float((uc_b*bw).sum()/1e9),
       "rel_bhc_pct":100*float(pw[eq<0.6*med].sum()/pw.sum()),"abs_bhc_pct":100*float(pw[eq<thr].sum()/pw.sum()),"child_rel_pct":100*float(pw[child&(eq<0.6*med)].sum()/pw[child].sum()),"median_week":med/52}
    print(json.dumps(r), flush=True); return r, uc_b, eq
res=[]
r0, uc0, eq0 = run(base_claim, "baseline draw (v20 file)")
res.append(r0)
# entitled set: everyone claims
r_all, uc_all, eq_all = run(np.ones(len(bid),bool), "everyone eligible claims (full take-up)"); res.append(r_all)
entitled = uc_all > 0
pool = entitled & ~anchor            # non-anchored entitled units: the draw chooses among these
n_base_pool = int((base_claim & pool).sum())
print("entitled units", int(entitled.sum()), "anchored", int(anchor.sum()), "anchored&entitled", int((anchor&entitled).sum()), "pool", int(pool.sum()), "base draws in pool", n_base_pool, flush=True)
# pre-UC equivalised income of the unit's household under full take-up minus its UC (proxy for need ranking)
sim_hh_eq_no_uc = None
rng=np.random.default_rng(12345)
for seed in (1,2,3):
    rng=np.random.default_rng(seed); pick=np.zeros(len(bid),bool); idx=np.flatnonzero(pool); pick[rng.choice(idx,size=n_base_pool,replace=False)]=True
    res.append(run(anchor|pick|(base_claim&~pool), f"alternative random draw seed {seed} (same pool count)")[0])
# need-ordered: rank pool by household equivalised income excluding their own UC (from the full-take-up run)
hh_eq_full = eq_all  # person-level; take household-level via first member
hh_level_eq_full = np.zeros(len(hid)); hh_level_eq_full[p_idx]=eq_all
with_uc_removed = hh_level_eq_full[b_hh] - uc_all/np.maximum(1e-9,1)  # crude: subtract unit UC (not equivalised) — ordering only
order=np.argsort(with_uc_removed); pool_order=[i for i in order if pool[i]]
poorest=np.zeros(len(bid),bool); poorest[pool_order[:n_base_pool]]=True
richest=np.zeros(len(bid),bool); richest[pool_order[-n_base_pool:]]=True
res.append(run(anchor|poorest|(base_claim&~pool), "residual to the POOREST entitled units (lower bound)")[0])
res.append(run(anchor|richest|(base_claim&~pool), "residual to the RICHEST entitled units (upper bound)")[0])
res.append(run(anchor|(base_claim&~pool), "anchored reporters only (no residual draw)")[0])
json.dump(res, open(sys.argv[1],"w"), indent=1); print("DONE")
