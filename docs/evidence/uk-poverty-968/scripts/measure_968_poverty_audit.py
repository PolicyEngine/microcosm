"""microcosm#968: poverty concept audit + decomposition across UK files (one engine)."""
import sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from importlib.metadata import version
from policyengine_uk import Microsimulation

ROOT = "/Users/mariajuaristi/Desktop/PolicyEngine"
HF = "/Users/mariajuaristi/.cache/huggingface/hub"
FILES = {
  "v20_national": (f"{ROOT}/runs/uk-623-first-calibrated/spine-assessment-v20/microcosm_uk_2024.h5", [2024, 2025, 2026]),
  "efrs_1_57_3": (f"{HF}/models--policyengine--policyengine-uk-data-private/snapshots/25af520a6651b8812fef56964a42a79a3f9f515a/enhanced_frs_2024_25.h5", [2024, 2025, 2026]),
  "x50_55k": (f"{ROOT}/data/ukds/acceptance/spine-s-main-rebuild/l0-k20-h55000-e2000-p50-f001-s42/microcosm_uk_2025_local.h5", [2026]),
  "june_populace_2023": (f"{HF}/datasets--policyengine--populace-uk-private/snapshots/a75a9a831d6b07aaffbd09713f2a1124f5c0f08f/populace_uk_2023.h5", [2026]),
}
SPINE_S = f"{ROOT}/data/ukds/acceptance/spine-s-main-rebuild/spine-s.h5"
OUT = sys.argv[1] if len(sys.argv) > 1 else "poverty_audit.json"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None

BENEFITS = ["child_benefit","council_tax_benefit","esa_income","esa_contrib","housing_benefit","income_support","jsa_income","jsa_contrib","pension_credit","universal_credit","working_tax_credit","child_tax_credit","attendance_allowance","afcs","bsp","carers_allowance","dla","iidb","incapacity_benefit","pip","sda","state_pension","maternity_allowance","statutory_sick_pay","statutory_maternity_pay","ssmg","cost_of_living_support_payment","winter_fuel_allowance","tax_free_childcare","healthy_start_vouchers","scottish_child_payment","carer_support_payment","free_school_meals","free_school_fruit_veg","free_school_milk","free_tv_licence_value"]
MARKET = ["employment_income","self_employment_income","savings_interest_income","dividend_income","miscellaneous_income","property_income","private_pension_income","private_transfer_income","maintenance_income"]
TAXES = ["income_tax","national_insurance","council_tax","domestic_rates","student_loan_repayments","employee_pension_contributions","personal_pension_contributions","maintenance_expenses","external_child_payments"]

def wmedian(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, c[-1] / 2)])
def wshare(mask, w): return float(w[mask].sum() / w.sum())
def wq(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]) / w[o].sum(); return [float(x[o][np.searchsorted(c, q)]) for q in qs]

def measure(sim, year, hw, p_idx, hh_arrays, per_arrays):
    A = hh_arrays; P = per_arrays
    pw = hw[p_idx]
    bhc = A["equiv_bhc"][p_idx]; ahc = A["equiv_ahc"][p_idx]
    thr_b = float(A["thr_bhc"][0]); thr_a = float(A["thr_ahc"][0])
    med_b_p = wmedian(bhc, pw); med_a_p = wmedian(ahc, pw)
    med_b_h = wmedian(A["equiv_bhc"], hw); med_a_h = wmedian(A["equiv_ahc"], hw)
    child = P["age"] < 18; pens = P["age"] >= 66; wa = (~child) & (~pens)
    def block(inc, thr_abs, med_p, med_h):
        rel_p = inc < 0.6 * med_p; rel_h = inc < 0.6 * med_h; abs_ = inc < thr_abs
        d = {
          "abs_rate_pct": 100*wshare(abs_, pw), "rel_rate_person_median_pct": 100*wshare(rel_p, pw),
          "rel_rate_household_median_pct": 100*wshare(rel_h, pw),
          "median_person_weighted_week": med_p/52, "median_household_weighted_week": med_h/52,
          "abs_threshold_week": thr_abs/52, "rel_line_person_week": 0.6*med_p/52,
          "abs_threshold_over_person_median": thr_abs/med_p,
          "share_below_pct_of_person_median": {str(q): 100*wshare(inc < q/100*med_p, pw) for q in (40,50,60,70,80)},
          "share_below_abs_line_scaled": {str(q): 100*wshare(inc < q/100*thr_abs, pw) for q in (90,95,100,105,110)},
          "by_group_abs_pct": {"children": 100*wshare(abs_ & child, pw)/wshare(child,pw), "working_age": 100*wshare(abs_ & wa, pw)/wshare(wa,pw), "pensioners": 100*wshare(abs_ & pens, pw)/wshare(pens,pw)},
          "by_group_rel_pct": {"children": 100*wshare(rel_p & child, pw)/wshare(child,pw), "working_age": 100*wshare(rel_p & wa, pw)/wshare(wa,pw), "pensioners": 100*wshare(rel_p & pens, pw)/wshare(pens,pw)},
          "group_pop_shares_pct": {"children": 100*wshare(child,pw), "working_age": 100*wshare(wa,pw), "pensioners": 100*wshare(pens,pw)},
          "share_nonpositive_income_pct": 100*wshare(inc <= 0, pw),
          "person_quantiles_week": dict(zip(["p5","p10","p20","p30","p40","p50"], [v/52 for v in wq(inc, pw, [.05,.1,.2,.3,.4,.5])])),
        }
        return d
    out = {"bhc": block(bhc, thr_b, med_b_p, med_b_h), "ahc": block(ahc, thr_a, med_a_p, med_a_h)}
    # decile composition on person-weighted equivalised BHC deciles (household-level amounts, weekly per household)
    edges = wq(bhc, pw, [i/10 for i in range(1,10)])
    dec = np.searchsorted(edges, bhc, side="right")  # 0..9 per person
    comp = {}
    for d in range(10):
        m = dec == d; w = pw[m]; hh = p_idx[m]
        comp[str(d+1)] = {k: float((A[k][hh]*w).sum()/w.sum()/52) for k in ("market","benefits","taxes","hbai_net","uc","state_pension","hb","pension_credit","child_benefit","equiv_bhc")}
        comp[str(d+1)]["persons_m"] = float(w.sum()/1e6)
        comp[str(d+1)]["equiv_factor_mean"] = float((A["equiv_factor"][hh]*w).sum()/w.sum())
    out["decile_composition_person_weighted_weekly_per_household"] = comp
    out["totals_bn"] = {k: float((A[k]*hw).sum()/1e9) for k in ("market","benefits","taxes","hbai_net","uc","state_pension","hb","pension_credit","child_benefit")}
    out["population_m"] = float(pw.sum()/1e6); out["households_m"] = float(hw.sum()/1e6)
    out["engine_flags_pct"] = {"in_poverty_bhc": 100*wshare(A["flag_abs_bhc"][p_idx] > 0, pw), "in_relative_poverty_bhc": 100*wshare(A["flag_rel_bhc"][p_idx] > 0, pw), "in_poverty_ahc": 100*wshare(A["flag_abs_ahc"][p_idx] > 0, pw)}
    return out

def load_arrays(sim, year):
    hh = lambda v: np.asarray(sim.calculate(v, year, map_to="household").values, dtype=float)
    A = {"equiv_bhc": hh("equiv_hbai_household_net_income"), "equiv_ahc": hh("equiv_hbai_household_net_income_ahc"),
         "thr_bhc": hh("poverty_threshold_bhc"), "thr_ahc": hh("poverty_line_ahc")/hh("household_equivalisation_ahc"), "equiv_factor": hh("household_equivalisation_bhc"),
         "hbai_net": hh("hbai_household_net_income"), "flag_abs_bhc": hh("in_poverty_bhc"), "flag_rel_bhc": hh("in_relative_poverty_bhc"), "flag_abs_ahc": hh("in_poverty_ahc"),
         "hw": hh("household_weight")}
    A["market"] = sum(hh(v) for v in MARKET); A["benefits"] = sum(hh(v) for v in BENEFITS); A["taxes"] = sum(hh(v) for v in TAXES)
    A["uc"] = hh("universal_credit"); A["state_pension"] = hh("state_pension"); A["hb"] = hh("housing_benefit"); A["pension_credit"] = hh("pension_credit"); A["child_benefit"] = hh("child_benefit")
    P = {"age": np.asarray(sim.calculate("age", year).values, dtype=float), "pw": np.asarray(sim.calculate("person_weight", year).values, dtype=float)}
    hid = np.asarray(sim.calculate("household_id", year).values); hid_p = np.asarray(sim.calculate("household_id", year, map_to="person").values)
    p_idx = pd.Index(hid).get_indexer(hid_p)
    assert (p_idx >= 0).all(), "person->household mapping failed"
    return A, P, p_idx

res = {"engine": version("policyengine-uk"), "files": {}}
for name, (path, years) in FILES.items():
    if ONLY and name not in ONLY: continue
    t0 = time.time(); print("==", name, flush=True)
    sim = Microsimulation(dataset=path)
    entry = {"h5": path, "years": {}}
    for y in years:
        t1 = time.time()
        A, P, p_idx = load_arrays(sim, y)
        hw = A["hw"]
        assert np.allclose(P["pw"], hw[p_idx]), "person_weight != household weight mapping"
        entry["years"][str(y)] = measure(sim, y, hw, p_idx, A, P)
        print("  year", y, "done", round(time.time()-t1), "s; abs BHC", round(entry["years"][str(y)]["bhc"]["abs_rate_pct"],2), "rel(person med)", round(entry["years"][str(y)]["bhc"]["rel_rate_person_median_pct"],2), flush=True)
        if name == "v20_national":
            # decomposition against spine-s design weights (same records, same order — verified)
            with pd.HDFStore(SPINE_S, "r") as st: design = st["household"]["household_weight"].to_numpy(float)
            with pd.HDFStore(path, "r") as st: cal = st["household"]["household_weight"].to_numpy(float)
            ratio = hw.sum()/cal.sum()  # engine may scale weights by year; apply same scaling to design
            dw = design * ratio
            dm = measure(sim, y, dw, p_idx, A, P)
            entry["years"][str(y)]["spine_s_design_weights"] = dm
            # fixed-line decomposition: design line applied to calibrated weights
            pw_c = hw[p_idx]; pw_d = dw[p_idx]; bhc = A["equiv_bhc"][p_idx]
            med_d = wmedian(bhc, pw_d); med_c = wmedian(bhc, pw_c)
            entry["years"][str(y)]["decomposition_bhc_rel"] = {
                "design_weights_design_line_pct": 100*wshare(bhc < 0.6*med_d, pw_d),
                "calibrated_weights_design_line_pct": 100*wshare(bhc < 0.6*med_d, pw_c),
                "calibrated_weights_calibrated_line_pct": 100*wshare(bhc < 0.6*med_c, pw_c),
                "design_median_week": med_d/52, "calibrated_median_week": med_c/52,
                "weight_scale_ratio_engine_vs_h5": ratio,
            }
            thr = float(A["thr_bhc"][0])
            entry["years"][str(y)]["decomposition_bhc_abs"] = {"design_weights_pct": 100*wshare(bhc < thr, pw_d), "calibrated_weights_pct": 100*wshare(bhc < thr, pw_c)}
            print("  decomposition:", json.dumps(entry["years"][str(y)]["decomposition_bhc_rel"]), flush=True)
        json.dump(res | {"files": res["files"] | {name: entry}}, open(OUT, "w"), indent=1)
    res["files"][name] = entry
    json.dump(res, open(OUT, "w"), indent=1)
    print("  file done", round(time.time()-t0), "s", flush=True)
print("ALL DONE")
