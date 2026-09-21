"""microcosm#968: bottom-tail profile and relative-poverty breakdowns (year 2024) for HBAI comparison."""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from policyengine_uk import Microsimulation
ROOT="/Users/mariajuaristi/Desktop/PolicyEngine"; HF="/Users/mariajuaristi/.cache/huggingface/hub"
FILES={"v20_national": f"{ROOT}/runs/uk-623-first-calibrated/spine-assessment-v20/microcosm_uk_2024.h5",
       "efrs_1_57_3": f"{HF}/models--policyengine--policyengine-uk-data-private/snapshots/25af520a6651b8812fef56964a42a79a3f9f515a/enhanced_frs_2024_25.h5"}
SPINE_S=f"{ROOT}/data/ukds/acceptance/spine-s-main-rebuild/spine-s.h5"
YEARS=[int(y) for y in (sys.argv[2].split(",") if len(sys.argv)>2 else ["2024"])]
def wmedian(x,w):
    o=np.argsort(x); c=np.cumsum(w[o]); return float(x[o][np.searchsorted(c,c[-1]/2)])
out={}
for name,path in FILES.items():
    sim=Microsimulation(dataset=path)
    with pd.HDFStore(path,"r") as st: person=st["person"]; h5hh=st["household"]
    hid=h5hh["household_id"].to_numpy(); p_idx=pd.Index(hid).get_indexer(person["person_household_id"].to_numpy()); assert (p_idx>=0).all()
    tenure = h5hh["tenure_type"].astype(str).to_numpy() if "tenure_type" in h5hh else np.array(["?"]*len(hid))
    for YEAR in YEARS:
        hh=lambda v: np.asarray(sim.calculate(v,YEAR,map_to="household").values,float)
        pp=lambda v: np.asarray(sim.calculate(v,YEAR).values,float)
        hw=hh("household_weight"); assert np.allclose(hw, h5hh["household_weight"].to_numpy(float)) or YEAR!=2024
        age=pp("age"); emp=pp("employment_income")+pp("self_employment_income"); is_child=age<18; pens=age>=66; adult=age>=18
        n_child=np.bincount(p_idx,weights=is_child,minlength=len(hid)); n_adult=np.bincount(p_idx,weights=adult,minlength=len(hid))
        n_pens=np.bincount(p_idx,weights=pens,minlength=len(hid)); n_work=np.bincount(p_idx,weights=(emp>1000),minlength=len(hid))
        n_ft=np.bincount(p_idx,weights=(emp>=12000),minlength=len(hid))
        equiv=hh("equiv_hbai_household_net_income"); thr=float(hh("poverty_threshold_bhc")[0])
        uc=hh("universal_credit"); wc=np.asarray(sim.calculate("would_claim_uc",YEAR,map_to="household").values,float)
        ftype=np.full(len(hid),"other_3plus_adults",dtype=object)
        ftype[(n_adult==1)&(n_child==0)&(n_pens==0)]="single_wa_no_children"; ftype[(n_adult==2)&(n_child==0)&(n_pens==0)]="couple_wa_no_children"
        ftype[(n_adult==1)&(n_child>0)]="lone_parent"; ftype[(n_adult==2)&(n_child>0)]="couple_with_children"
        ftype[(n_adult==1)&(n_pens==1)&(n_child==0)]="single_pensioner"; ftype[(n_adult==2)&(n_pens>=1)&(n_child==0)]="pensioner_couple_or_mixed"
        weightings={"calibrated":hw}
        if name=="v20_national":
            with pd.HDFStore(SPINE_S,"r") as st: design=st["household"]["household_weight"].to_numpy(float)
            weightings["spine_s_design"]=design*hw.sum()/design.sum()
        for wname,w in weightings.items():
            pw=w[p_idx]; eqp=equiv[p_idx]; med=wmedian(eqp,pw); rel=eqp<0.6*med
            tot=pw.sum()
            r={"median_week":med/52,"rel_bhc_pct":100*pw[rel].sum()/tot,"abs_bhc_pct":100*pw[eqp<thr].sum()/tot}
            r["share_below_abs_week_pct"]={str(c):100*pw[eqp<c*52].sum()/tot for c in (10,50,100,150,200,260,325,370,390,455)}
            r["share_in_band_pct_of_median"]={f"{a}-{b}":100*pw[(eqp>=a/100*med)&(eqp<b/100*med)].sum()/tot for a,b in ((-1000,0),(0,20),(20,40),(40,50),(50,60),(60,70),(70,80))}
            def grp(mask_hh, label):
                m=mask_hh[p_idx]; return {"pop_share_pct":100*pw[m].sum()/tot, "rel_rate_pct":100*pw[m&rel].sum()/max(pw[m].sum(),1e-9), "share_of_poor_pct":100*pw[m&rel].sum()/pw[rel].sum()}
            r["by_family_type"]={t:grp(ftype==t,t) for t in sorted(set(ftype))}
            r["by_n_children"]={str(k):grp((n_child==k) if k<3 else (n_child>=3),k) for k in (0,1,2,3)}
            r["children_rel_rate_by_family_children"]={}
            for k in (1,2,3):
                m=((n_child==k) if k<3 else (n_child>=3))[p_idx]&is_child; r["children_rel_rate_by_family_children"][str(k)]={"rate":100*pw[m&rel].sum()/max(pw[m].sum(),1e-9),"children_m":pw[m].sum()/1e6}
            r["by_work"]={"workless_hh":grp(n_work==0,"w0"),"working_hh":grp(n_work>0,"w1"),"any_ft_earner":grp(n_ft>0,"ft"),"working_no_ft":grp((n_work>0)&(n_ft==0),"pt")}
            r["by_tenure"]={t:grp(tenure==t,t) for t in sorted(set(tenure))}
            r["children"]={"rel_rate_pct":100*pw[is_child&rel].sum()/pw[is_child].sum(),"in_workless_hh_rate":100*pw[is_child&rel&(n_work==0)[p_idx]].sum()/max(pw[is_child&(n_work==0)[p_idx]].sum(),1e-9),"share_children_in_workless_hh":100*pw[is_child&(n_work==0)[p_idx]].sum()/pw[is_child].sum()}
            r["uc"]={"hh_with_uc_m":w[uc>0].sum()/1e6,"persons_in_uc_hh_rel_poor_pct":100*pw[(uc>0)[p_idx]&rel].sum()/max(pw[(uc>0)[p_idx]].sum(),1e-9),"share_of_rel_poor_in_uc_hh_pct":100*pw[(uc>0)[p_idx]&rel].sum()/pw[rel].sum(),"would_claim_false_eligible_hh_m":None}
            out[f"{name}|{wname}|{YEAR}"]=r
            print(name,wname,YEAR,"rel",round(r["rel_bhc_pct"],2),"child",round(r["children"]["rel_rate_pct"],2),"bands",{k:round(v,2) for k,v in r["share_in_band_pct_of_median"].items()},flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1); print("DONE")
