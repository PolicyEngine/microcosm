# ruff: noqa: N801, N805 (PolicyEngine variable classes and formulas)
"""Real PolicyEngine-US microsimulation runs: Medicare premiums in the SE health premium attribution.

Every figure this script writes is the output of a PolicyEngine-US
``Microsimulation`` on the dataset named on the command line. A scenario changes
one person input (``health_insurance_premiums``, and ``is_self_employed`` with
``--widen``) with ``set_input`` before anything is calculated, or applies one
structural reform; nothing is scaled or interpolated between runs.

Scenarios (the attribution's own columns come from the baseline run):

- ``shipped``: the dataset as stored (the Medicare proxy withholds attribution).
- ``reported_where_not_enrolled``: the reported premium is also attributed to
  self-employed people in the Medicare proxy whom the engine does not count as
  Medicare enrollees (their modeled Part B premium is zero).
- ``reported_only``: the reported non-Part-B premium is also attributed to
  self-employed people in the Medicare proxy (outside measured ESI).
- ``reported_plus_modeled_part_b``: the reported premium plus the engine's
  ``medicare_part_b_premium`` for enrollees, as a build would freeze it.
- ``reported_plus_reported_part_b``: the reported premium plus the stored CPS
  Medicare premium (``medicare_part_b_premiums``, ASEC PEMCPREM: the monthly
  Medicare deduction from Social Security, which the survey says includes
  Medicare Advantage, Part B and Part D premiums), where the dataset stores it.
- ``reported_plus_all_modeled_medicare``: the reported premium plus modeled
  Part B, the Part A premium and the Part D income-related surcharge.
- ``reported_plus_modeled_part_b_where_reported``: as the modeled Part B
  scenario, but only for enrollees who report a premium of their own.
- ``reported_plus_modeled_part_b_sole_proprietors`` (``--widen`` only): as the
  modeled Part B scenario, but enrollees whose only profitable business is a
  partnership stay withheld (a partner's plan needs the partnership to pay or
  reimburse the premiums, which no input records).
- ``engine_side``: the stored dataset, with ``self_employed_health_insurance_premiums``
  redefined in the engine as the medical-expense premium resolver, withheld
  under measured ESI (the data stage's own exclusion).

- ``engine_side_standard_part_b``: as ``engine_side``, but the engine adds only
  the standard Part B premium net of Medicare Savings Program coverage, which
  does not read ``filing_status`` (the income-related adjustment does).

Each stage runs in its own process, so one process holds one simulation
(two simulations in one process passed the shared heavy lock's 20 GB ceiling):

    measure.py LABEL OUT_PREFIX DATASET [--widen] --stage prepare
    measure.py LABEL OUT_PREFIX DATASET [--widen] --stage scenario --name NAME [--light]
    measure.py LABEL OUT_PREFIX DATASET [--widen] --stage combine

``prepare`` reads the stored inputs and the engine's modeled Medicare premiums
from one simulation of the stored dataset and writes every scenario's premium
input; ``scenario`` runs one simulation with that input; ``combine`` compares
each scenario with ``shipped`` on the arrays the runs saved.
"""

import argparse
import gc
import json
import resource
import time
from importlib.metadata import version

import numpy as np
from policyengine_core.reforms import Reform
from policyengine_us import Microsimulation
from policyengine_us.model_api import USD, YEAR, Person, Variable, max_, where

PERIOD = 2024
OUT_YEAR = 2026
MEDICARE_AGE = 65
ALL_SCENARIOS = (
    "shipped",
    "reported_where_not_enrolled",
    "reported_only",
    "reported_plus_modeled_part_b",
    "reported_plus_reported_part_b",
    "reported_plus_all_modeled_medicare",
    "reported_plus_modeled_part_b_where_reported",
    "reported_plus_modeled_part_b_sole_proprietors",
    "engine_side",
    "engine_side_standard_part_b",
)

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("out_prefix")
parser.add_argument("dataset")
parser.add_argument("--widen", action="store_true")
parser.add_argument(
    "--stage", choices=("prepare", "scenario", "outyear", "combine"), required=True
)
parser.add_argument("--name", choices=ALL_SCENARIOS)
parser.add_argument("--light", action="store_true")
parser.add_argument(
    "--outputs",
    default="federal,state,snap",
    help="heavy output groups to compute in this process (pe-us 2.2.1 needs one per process)",
)
parser.add_argument("--tag", default="", help="file suffix for a split heavy run")
parser.add_argument(
    "--neutralize",
    action="store_true",
    help="also neutralize self_employed_health_insurance_ald and report income_tax only "
    "(the JCT tax-expenditure row: reform income_tax minus baseline income_tax)",
)
args = parser.parse_args()
assert (args.stage in ("scenario", "outyear")) == (args.name is not None), (
    "--name goes with --stage scenario or outyear"
)
OUTPUT_GROUPS = {
    "federal": (
        ("tax_unit", "adjusted_gross_income"),
        ("tax_unit", "income_tax"),
        ("tax_unit", "itemized_medical_expenses"),
        ("tax_unit", "medical_expense_deduction"),
    ),
    "state": (("tax_unit", "state_income_tax"),),
    "snap": (("spm_unit", "snap"),),
}
outputs = [group for group in args.outputs.split(",") if group]
assert set(outputs) <= set(OUTPUT_GROUPS), outputs


class self_employed_health_insurance_premiums(Variable):
    value_type = float
    entity = Person
    label = "Self-employed health insurance premiums (engine-side Medicare reading)"
    unit = USD
    definition_period = YEAR
    defined_for = "is_self_employed"

    def formula(person, period, parameters):
        premiums = person("medical_expense_health_insurance_premiums", period)
        return where(person("has_esi", period), 0, premiums)


class EngineSide(Reform):
    def apply(self) -> None:
        self.update_variable(self_employed_health_insurance_premiums)


def standard_part_b_formula(person, period, parameters):
    direct = person("health_insurance_premiums", period)
    non_medicare = person("health_insurance_premiums_without_medicare_part_b", period)
    enrolled = person("medicare_enrolled", period)
    standard = max_(
        person("base_part_b_premium", period)
        - person("msp_part_b_premium_coverage", period),
        0,
    )
    premiums = where(direct != 0, direct, non_medicare + standard * enrolled)
    return where(person("has_esi", period), 0, premiums)


class EngineSideStandardPartB(Reform):
    def apply(self) -> None:
        variable = type(
            "self_employed_health_insurance_premiums",
            (Variable,),
            dict(
                value_type=float,
                entity=Person,
                label="Self-employed health insurance premiums (standard Part B)",
                unit=USD,
                definition_period=YEAR,
                defined_for="is_self_employed",
                formula=standard_part_b_formula,
            ),
        )
        self.update_variable(variable)


class NeutralizeALD(Reform):
    def apply(self) -> None:
        self.neutralize_variable("self_employed_health_insurance_ald")


ENGINE_REFORMS = {
    "engine_side": EngineSide,
    "engine_side_standard_part_b": EngineSideStandardPartB,
}


def arr(sim, name, period=PERIOD):
    values = np.asarray(sim.calculate(name, period).values)
    # The engine stores float32; totals over 166,000 people are summed in float64.
    return values.astype(np.float64) if values.dtype.kind == "f" else values


def float32_tolerance(reference):
    """A per-element tolerance for comparing engine float32 values: one cent or
    a few float32 units of the reference, whichever is larger."""
    return np.maximum(
        0.01, 4 * np.spacing(np.abs(reference).astype(np.float32)).astype(np.float64)
    )


def total(values, weights, mask=None):
    v = values * weights
    return float(v.sum() if mask is None else v[mask].sum())


def has(sim, name):
    return name in sim.tax_benefit_system.variables


def peak_rss_mb():
    # macOS reports ru_maxrss in bytes.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)


def script_sha256():
    import hashlib

    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def save_json(path, payload):
    payload.setdefault("measure_py_sha256", script_sha256())
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


INPUTS = args.out_prefix + ".inputs.npz"
HEADER = args.out_prefix + ".header.json"


def scenario_paths(name, neutralized=False, tag=""):
    stem = f"{name}.neutralized" if neutralized else name
    if tag:
        stem += f".{tag}"
    return args.out_prefix + f".{stem}.json", args.out_prefix + f".{stem}.npz"


t_start = time.time()


# ----------------------------------------------------------------- prepare
def prepare():
    out = {
        "label": args.label,
        "policyengine_us": version("policyengine-us"),
        "policyengine_core": version("policyengine-core"),
        "dataset": args.dataset,
        "period": PERIOD,
        "widened_plan_businesses": bool(args.widen),
    }
    base = Microsimulation(dataset=args.dataset)
    p_w = arr(base, "person_weight")
    tu_w = arr(base, "tax_unit_weight")
    spm_w = arr(base, "spm_unit_weight")
    age = arr(base, "age")
    ssdi = arr(base, "social_security_disability")
    has_esi = arr(base, "has_esi").astype(bool)
    stored_flag = arr(base, "is_self_employed").astype(bool)
    stored_direct = arr(base, "health_insurance_premiums")
    reported = arr(base, "health_insurance_premiums_without_medicare_part_b")
    sched_c = arr(base, "total_self_employment_income")
    farm = arr(base, "farm_operations_income")
    partnership = arr(base, "partnership_self_employment_net_earnings")
    enrolled = arr(base, "medicare_enrolled").astype(bool)
    eligible = arr(base, "is_medicare_eligible").astype(bool)
    part_b = arr(base, "medicare_part_b_premium")
    gross_b = arr(base, "gross_medicare_part_b_premium")
    base_b = arr(base, "base_part_b_premium")
    msp_b = arr(base, "msp_part_b_premium_coverage")
    part_a = arr(base, "medicare_part_a_premium")
    part_d_irmaa = arr(base, "income_adjusted_part_d_premium_surcharge")
    social_security = arr(base, "social_security")
    takes_up = arr(base, "takes_up_medicare_if_eligible").astype(bool)
    person_tu = np.asarray(base.populations["tax_unit"].members_entity_id)
    del base
    gc.collect()

    proxy = (age >= MEDICARE_AGE) | (ssdi > 0)
    plan_business = {
        "schedule_c": sched_c > 0,
        "farm": farm > 0,
        "partnership": partnership > 0,
    }
    widened_flag = (
        plan_business["schedule_c"]
        | plan_business["farm"]
        | plan_business["partnership"]
    )
    flag = widened_flag if args.widen else stored_flag
    modeled_b = part_b * enrolled

    # The stored attribution must be the Schedule C rule; otherwise the
    # scenarios would not differ from the dataset by the Medicare proxy alone.
    reproduced = np.where(
        plan_business["schedule_c"] & ~proxy & ~has_esi, reported, 0.0
    )
    out["stored_attribution_reproduced"] = {
        "premiums_equal": bool(
            np.allclose(reproduced, stored_direct, rtol=0, atol=1e-6)
        ),
        "flag_equal": bool((plan_business["schedule_c"] == stored_flag).all()),
        "max_abs_premium_difference": float(np.abs(reproduced - stored_direct).max()),
    }
    if not (
        out["stored_attribution_reproduced"]["premiums_equal"]
        and out["stored_attribution_reproduced"]["flag_equal"]
    ):
        raise SystemExit(
            "the stored attribution is not the Schedule C rule this script assumes"
        )
    # Whether a build could tell enrollees apart from stored inputs alone: the
    # engine's own predicate against the age and take-up columns.
    enrolled_from_inputs = (age >= MEDICARE_AGE) & takes_up
    out["enrollment_from_stored_inputs"] = {
        "equal": bool((enrolled_from_inputs == enrolled).all()),
        "rows_differing": int((enrolled_from_inputs != enrolled).sum()),
        "enrolled_under_65_rows": int((enrolled & (age < MEDICARE_AGE)).sum()),
        "modeled_part_b_positive_not_enrolled_rows": int(
            ((part_b > 0) & ~enrolled).sum()
        ),
    }

    reported_part_b = None
    try:
        import pandas as pd

        stored_person = pd.read_hdf(
            args.dataset, "person", columns=["medicare_part_b_premiums"]
        )
        reported_part_b = stored_person["medicare_part_b_premiums"].to_numpy(float)
        assert len(reported_part_b) == len(p_w)
    except Exception as error:  # the column is absent from builds after microcosm#590
        out["reported_part_b_unavailable"] = repr(error)

    attributable = flag & ~has_esi
    withheld = flag & ~proxy & ~has_esi
    partner_only = (
        plan_business["partnership"]
        & ~plan_business["schedule_c"]
        & ~plan_business["farm"]
    )
    premium_inputs = {
        "shipped": np.where(withheld, reported, 0.0),
        "reported_where_not_enrolled": np.where(
            attributable & ~(proxy & enrolled), reported, 0.0
        ),
        "reported_only": np.where(attributable, reported, 0.0),
        "reported_plus_modeled_part_b": np.where(
            attributable, reported + modeled_b, 0.0
        ),
        "reported_plus_all_modeled_medicare": np.where(
            attributable, reported + modeled_b + (part_a + part_d_irmaa) * enrolled, 0.0
        ),
        "reported_plus_modeled_part_b_where_reported": np.where(
            attributable & ~(proxy & enrolled & ~(reported > 0)),
            reported + modeled_b,
            0.0,
        ),
        "engine_side": np.where(withheld, reported, 0.0),
        "engine_side_standard_part_b": np.where(withheld, reported, 0.0),
    }
    if args.widen:
        premium_inputs["reported_plus_modeled_part_b_sole_proprietors"] = np.where(
            attributable & ~(proxy & enrolled & partner_only), reported + modeled_b, 0.0
        )
    if reported_part_b is not None:
        premium_inputs["reported_plus_reported_part_b"] = np.where(
            attributable, reported + reported_part_b, 0.0
        )
    out["scenarios_available"] = sorted(premium_inputs)

    def population(mask):
        return {"people": float(p_w[mask].sum()), "rows": int(mask.sum())}

    se_proxy = flag & proxy
    out["population"] = {
        "people": population(np.ones_like(flag)),
        "self_employed_flag": population(flag),
        "medicare_proxy": population(proxy),
        "medicare_eligible": population(eligible),
        "medicare_enrolled": population(enrolled),
        "enrolled_outside_proxy": population(enrolled & ~proxy),
        "proxy_not_enrolled": population(proxy & ~enrolled),
        "self_employed_in_proxy": population(se_proxy),
        "self_employed_in_proxy_enrolled": population(se_proxy & enrolled),
        "self_employed_in_proxy_with_esi": population(se_proxy & has_esi),
        "self_employed_in_proxy_outside_esi": population(se_proxy & ~has_esi),
        "self_employed_in_proxy_outside_esi_enrolled": population(
            se_proxy & ~has_esi & enrolled
        ),
        "self_employed_in_proxy_outside_esi_not_enrolled": population(
            se_proxy & ~has_esi & ~enrolled
        ),
        "self_employed_in_proxy_outside_esi_reported_positive": population(
            se_proxy & ~has_esi & (reported > 0)
        ),
        "self_employed_in_proxy_outside_esi_part_b_positive": population(
            se_proxy & ~has_esi & (modeled_b > 0)
        ),
        "self_employed_in_proxy_outside_esi_part_b_only": population(
            se_proxy & ~has_esi & (modeled_b > 0) & ~(reported > 0)
        ),
        "self_employed_enrolled_outside_esi_without_social_security": population(
            flag & enrolled & ~has_esi & ~(social_security > 0)
        ),
        "self_employed_enrolled_outside_esi_without_social_security_reported_positive": population(
            flag & enrolled & ~has_esi & ~(social_security > 0) & (reported > 0)
        ),
        "self_employed_in_proxy_outside_esi_partner_only": population(
            se_proxy & ~has_esi & partner_only
        ),
        "by_plan_business_in_proxy": {
            k: population(m & proxy) for k, m in plan_business.items()
        },
        "by_plan_business": {k: population(m) for k, m in plan_business.items()},
    }
    group = se_proxy & ~has_esi
    out["premium_mass_self_employed_in_proxy_outside_esi"] = {
        "reported_non_part_b": total(reported, p_w, group),
        "modeled_part_b_net_of_msp": total(modeled_b, p_w, group),
        "modeled_gross_part_b_enrolled": total(gross_b * enrolled, p_w, group),
        "modeled_base_part_b_enrolled": total(base_b * enrolled, p_w, group),
        "modeled_irmaa_part_b_enrolled": total(
            (gross_b - base_b) * enrolled, p_w, group
        ),
        "msp_part_b_coverage": total(msp_b, p_w, group),
        "modeled_part_a_premium": total(part_a * enrolled, p_w, group),
        "modeled_part_d_irmaa": total(part_d_irmaa * enrolled, p_w, group),
        "stored_reported_part_b": (
            None if reported_part_b is None else total(reported_part_b, p_w, group)
        ),
    }
    out["part_b_all_enrollees"] = {
        "modeled_net": total(modeled_b, p_w),
        "modeled_gross": total(gross_b * enrolled, p_w),
        "irmaa": total((gross_b - base_b) * enrolled, p_w),
        "stored_reported": None
        if reported_part_b is None
        else total(reported_part_b, p_w),
        "enrolled_people": float(p_w[enrolled].sum()),
        "stored_reported_positive_people": (
            None if reported_part_b is None else float(p_w[reported_part_b > 0].sum())
        ),
    }
    out["prepare_seconds"] = round(time.time() - t_start, 1)
    out["prepare_peak_rss_mb"] = peak_rss_mb()
    keep = dict(
        person_weight=p_w,
        tax_unit_weight=tu_w,
        spm_unit_weight=spm_w,
        age=age,
        ssdi=ssdi,
        has_esi=has_esi,
        flag=flag,
        stored_flag=stored_flag,
        stored_direct=stored_direct,
        reported=reported,
        schedule_c=sched_c,
        farm=farm,
        partnership=partnership,
        enrolled=enrolled,
        eligible=eligible,
        modeled_part_b=modeled_b,
        gross_part_b=gross_b,
        msp_part_b=msp_b,
        part_a=part_a,
        part_d_irmaa=part_d_irmaa,
        social_security=social_security,
        takes_up_medicare=takes_up,
        proxy=proxy,
        person_tax_unit=person_tu,
    )
    if reported_part_b is not None:
        keep["reported_part_b"] = reported_part_b
    for name, values in premium_inputs.items():
        keep[f"premium_input.{name}"] = values
    np.savez_compressed(INPUTS, **keep)
    save_json(HEADER, out)
    print(args.label, "prepare", json.dumps(out, indent=1), flush=True)


# ---------------------------------------------------------------- scenario
def run_scenario(name):
    data = np.load(INPUTS)
    key = f"premium_input.{name}"
    if key not in data:
        print(args.label, name, "not available on this dataset or mode", flush=True)
        return
    p_w = data["person_weight"]
    tu_w = data["tax_unit_weight"]
    spm_w = data["spm_unit_weight"]
    proxy = data["proxy"]
    enrolled = data["enrolled"]
    flag = data["flag"]
    premium_input = data[key]
    plan_business = {
        "schedule_c": data["schedule_c"] > 0,
        "farm": data["farm"] > 0,
        "partnership": data["partnership"] > 0,
    }
    t0 = time.time()
    reforms = (ENGINE_REFORMS[name],) if name in ENGINE_REFORMS else ()
    if args.neutralize:
        reforms += (NeutralizeALD,)
    if reforms:
        sim = Microsimulation(dataset=args.dataset, reform=reforms)
    else:
        sim = Microsimulation(dataset=args.dataset)
    sim.set_input("health_insurance_premiums", PERIOD, premium_input)
    if args.widen:
        sim.set_input("is_self_employed", PERIOD, flag)
    # The engine holds inputs as float32: compare at float32 precision.
    assert np.allclose(
        arr(sim, "health_insurance_premiums"), premium_input, rtol=1e-6, atol=0.01
    )
    assert (arr(sim, "is_self_employed").astype(bool) == flag).all()

    if args.neutralize:
        income_tax = arr(sim, "income_tax")
        ald = arr(sim, "self_employed_health_insurance_ald")
        assert not ald.any(), "the neutralized deduction must be zero"
        result = {
            "name": name,
            "neutralized": "self_employed_health_insurance_ald",
            "income_tax": total(income_tax, tu_w),
            "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": peak_rss_mb(),
        }
        json_path, npz_path = scenario_paths(name, neutralized=True)
        np.savez_compressed(npz_path, income_tax=income_tax)
        save_json(json_path, result)
        print(args.label, name, "neutralized", json.dumps(result, indent=1), flush=True)
        return

    result = {"name": name, "light": bool(args.light)}
    keep = {}
    ald_p = arr(sim, "self_employed_health_insurance_ald_person")
    ald = arr(sim, "self_employed_health_insurance_ald")
    se_premiums = arr(sim, "self_employed_health_insurance_premiums")
    medical = arr(sim, "medical_expense_health_insurance_premiums")
    direct = arr(sim, "health_insurance_premiums")
    part_b_now = arr(sim, "medicare_part_b_premium")
    pos = ald_p > 0
    sc, fm, pt = (
        plan_business["schedule_c"],
        plan_business["farm"],
        plan_business["partnership"],
    )
    result["deduction"] = {
        "amount": total(ald, tu_w),
        "returns": float(tu_w[ald > 0].sum()),
        "tax_unit_rows": int((ald > 0).sum()),
        "people": float(p_w[pos].sum()),
        "amount_in_medicare_proxy": total(ald_p, p_w, proxy),
        "people_in_medicare_proxy": float(p_w[pos & proxy].sum()),
        "amount_enrolled": total(ald_p, p_w, enrolled),
        "people_enrolled": float(p_w[pos & enrolled].sum()),
        "amount_outside_proxy": total(ald_p, p_w, ~proxy),
        "by_plan_business": {
            "schedule_c": {
                "people": float(p_w[pos & sc].sum()),
                "amount": total(ald_p, p_w, sc),
            },
            "farm_without_schedule_c_profit": {
                "people": float(p_w[pos & ~sc & fm].sum()),
                "amount": total(ald_p, p_w, ~sc & fm),
            },
            "partnership_without_schedule_c_or_farm_profit": {
                "people": float(p_w[pos & ~sc & ~fm & pt].sum()),
                "amount": total(ald_p, p_w, ~sc & ~fm & pt),
            },
        },
    }
    result["premium_base"] = {
        "direct_input_total": total(direct, p_w),
        "direct_input_people": float(p_w[direct > 0].sum()),
        "self_employed_premiums_total": total(se_premiums, p_w),
        "self_employed_premiums_people": float(p_w[se_premiums > 0].sum()),
        "self_employed_premiums_in_proxy": total(se_premiums, p_w, proxy),
        "limited_by_earned_income_people": float(
            p_w[(se_premiums > ald_p + 0.005)].sum()
        ),
        "premiums_above_deduction": total(se_premiums - ald_p, p_w),
    }
    result["medical_expense_premiums_total"] = total(medical, p_w)
    result["medicare_part_b_premium_total"] = total(part_b_now, p_w)
    keep.update(
        ald_person=ald_p,
        ald_tax_unit=ald,
        se_premiums=se_premiums,
        medical_expense_premiums=medical,
        direct=direct,
        part_b=part_b_now,
    )
    if not args.light:
        result["output_groups"] = outputs
        weights = {"tax_unit": tu_w, "spm_unit": spm_w}
        for group in outputs:
            for entity, variable in OUTPUT_GROUPS[group]:
                if has(sim, variable):
                    values = arr(sim, variable)
                    result[variable] = total(values, weights[entity])
                    keep[variable] = values
    result["adjusted_gross_income_periods_computed"] = sorted(
        str(p) for p in sim.get_holder("adjusted_gross_income").get_known_periods()
    )
    if name == "shipped" and not args.tag:
        try:
            result["out_year"] = out_year(sim, data, data["stored_direct"])
        except Exception as error:  # an out-year parameter gap must not lose the run
            result["out_year_error"] = repr(error)
    result["seconds"] = round(time.time() - t0, 1)
    result["peak_rss_mb"] = peak_rss_mb()
    json_path, npz_path = scenario_paths(name, tag=args.tag)
    np.savez_compressed(npz_path, **keep)
    save_json(json_path, result)
    print(args.label, name, args.tag, json.dumps(result, indent=1), flush=True)


def out_year(sim, data, base, mask=None):
    """Whether the direct input and the decomposed premium stay equal after uprating.

    policyengine-us extends a single-year file to every later year when it
    loads it, uprating each stored input (``extend_single_year_dataset``), so
    later years hold the file's own values: ``set_input`` for 2024 changes 2024
    only. This therefore tests the attribution stored in the dataset the
    simulation loaded. ``base`` is that stored direct input; ``mask`` restricts
    the comparison (e.g. to Medicare enrollees).
    """
    p_w = data["person_weight"]
    reported = data["reported"]
    modeled_b = data["modeled_part_b"]
    mask = np.ones(len(p_w), dtype=bool) if mask is None else mask
    result = {"year": OUT_YEAR}
    for year in (PERIOD, OUT_YEAR):
        direct = arr(sim, "health_insurance_premiums", year)
        non_medicare = arr(
            sim, "health_insurance_premiums_without_medicare_part_b", year
        )
        enrolled_then = arr(sim, "medicare_enrolled", year).astype(bool)
        part_b_then = arr(sim, "medicare_part_b_premium", year) * enrolled_then
        medical = arr(sim, "medical_expense_health_insurance_premiums", year)
        decomposed = non_medicare + part_b_then
        diff = np.where(mask, medical - decomposed, 0.0)
        diff = np.where(np.abs(diff) > float32_tolerance(decomposed), diff, 0.0)
        carrier = (base > 0) & mask
        result[str(year)] = {
            "direct_total_carriers": total(direct, p_w, carrier),
            "direct_growth_factor": float(
                direct[carrier].sum() / max(base[carrier].sum(), 1e-9)
            ),
            "reported_growth_factor": float(
                non_medicare[reported > 0].sum() / reported[reported > 0].sum()
            ),
            "part_b_growth_factor": float(
                part_b_then.sum() / max(modeled_b.sum(), 1e-9)
            ),
            "resolver_minus_decomposed_total": total(diff, p_w),
            "resolver_minus_decomposed_rows": int((np.abs(diff) > 0.005).sum()),
            "resolver_minus_decomposed_people": float(p_w[np.abs(diff) > 0.005].sum()),
            "decomposed_total_in_mask": total(decomposed, p_w, mask),
            "resolver_total_in_mask": total(medical, p_w, mask),
        }
    return result


def candidate_out_year(name):
    """Write the scenario's attribution into a copy of the dataset and test it in later years."""
    import pandas as pd

    data = np.load(INPUTS)
    premium_input = data[f"premium_input.{name}"]
    candidate = args.out_prefix + f".{name}.candidate.h5"
    keys = [
        "_time_period",
        "family",
        "household",
        "marital_unit",
        "person",
        "spm_unit",
        "tax_unit",
    ]
    try:
        with pd.HDFStore(args.dataset, mode="r") as store:
            tables = {key: store[key] for key in keys}
        person = tables["person"]
        assert len(person) == len(premium_input)
        person["health_insurance_premiums"] = premium_input
        if args.widen:
            person["is_self_employed"] = data["flag"]
        with pd.HDFStore(candidate, mode="w") as store:
            for key in keys:
                store.put(key, tables[key], format="table")
        del tables, person
        gc.collect()
        t0 = time.time()
        sim = Microsimulation(dataset=candidate)
        stored = arr(sim, "health_insurance_premiums")
        assert np.allclose(stored, premium_input, rtol=1e-6, atol=0.01)
        enrolled = data["enrolled"]
        if args.light:
            # Inputs only: how the stored direct input and the reported premium
            # uprate. pe-us 2.2.1 cannot compute the 2026 Part B premium (its
            # IRMAA reads 2024 AGI, the full tax cascade) within the shared
            # memory cap, so the resolver is not computed here.
            p_w = data["person_weight"]
            carriers = premium_input > 0
            inputs = {}
            for year in (PERIOD, OUT_YEAR):
                direct = arr(sim, "health_insurance_premiums", year)
                non_medicare = arr(
                    sim, "health_insurance_premiums_without_medicare_part_b", year
                )
                inputs[str(year)] = {
                    mask_name: {
                        "direct": total(direct, p_w, mask),
                        "reported": total(non_medicare, p_w, mask),
                    }
                    for mask_name, mask in (
                        ("carriers", carriers),
                        ("non_enrollee_carriers", carriers & ~enrolled),
                        ("enrollee_carriers", carriers & enrolled),
                    )
                }
            result = {
                "name": name,
                "light": True,
                "candidate": "certified file with this scenario's health_insurance_premiums",
                "input_uprating": inputs,
                "seconds": round(time.time() - t0, 1),
                "peak_rss_mb": peak_rss_mb(),
            }
            save_json(args.out_prefix + f".{name}.outyear.json", result)
            print(args.label, name, "outyear", json.dumps(result, indent=1), flush=True)
            return
        result = {
            "name": name,
            "candidate": "certified file with this scenario's health_insurance_premiums"
            + (" and is_self_employed" if args.widen else ""),
            "all_people": out_year(sim, data, premium_input),
            "medicare_enrollee_carriers": out_year(
                sim, data, premium_input, mask=enrolled & (premium_input > 0)
            ),
            "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": peak_rss_mb(),
        }
    finally:
        import os

        if os.path.exists(candidate):
            os.remove(candidate)
    save_json(args.out_prefix + f".{name}.outyear.json", result)
    print(args.label, name, "outyear", json.dumps(result, indent=1), flush=True)


# ----------------------------------------------------------------- combine
def load_scenario(name):
    """One scenario's result and arrays, merged from its untagged and tagged parts.

    A part written before this prefix's prepare stage is stale and refused.
    """
    import os

    prepared = os.path.getmtime(INPUTS)
    result, arrays = None, {}
    parts = [""] + [f"{group}" for group in OUTPUT_GROUPS]
    for tag in parts:
        json_path, npz_path = scenario_paths(name, tag=tag)
        try:
            part = json.load(open(json_path))
        except FileNotFoundError:
            continue
        if os.path.getmtime(json_path) < prepared:
            STALE.append(json_path)
            continue
        if "error" in part:
            if result is None:
                result = part
            else:
                result.setdefault("part_errors", {})[tag or "core"] = part["error"]
            continue
        if result is None or "error" in result:
            result = part
        else:
            for key, value in part.items():
                result.setdefault(key, value)
            result.setdefault("parts", []).append(tag)
        if os.path.getmtime(npz_path) < prepared:
            STALE.append(npz_path)
            continue
        loaded = np.load(npz_path)
        for key in loaded.files:
            arrays.setdefault(key, loaded[key])
    return result, arrays


STALE = []


def combine():
    import os

    import pandas as pd

    out = json.load(open(HEADER))
    data = np.load(INPUTS)
    p_w = data["person_weight"]
    shipped, shipped_arrays = load_scenario("shipped")
    out["scenarios"] = {"shipped": shipped}
    heavy = (
        "adjusted_gross_income",
        "income_tax",
        "itemized_medical_expenses",
        "medical_expense_deduction",
        "state_income_tax",
        "snap",
    )
    for name in ALL_SCENARIOS[1:]:
        result, arrays = load_scenario(name)
        if result is None:
            continue
        if "error" in result:
            out["scenarios"][name] = result
            continue
        diff = (
            arrays["medical_expense_premiums"]
            - shipped_arrays["medical_expense_premiums"]
        )
        changed = np.abs(diff) > float32_tolerance(
            shipped_arrays["medical_expense_premiums"]
        )
        result["medical_expense_premiums_vs_shipped"] = {
            "people_changed": float(p_w[changed].sum()),
            "rows_changed": int(changed.sum()),
            "total_change": total(diff, p_w),
            "increase": total(np.maximum(diff, 0), p_w),
            "decrease": total(np.minimum(diff, 0), p_w),
            "max_abs_change": float(np.abs(diff).max()),
        }
        part_b_diff = arrays["part_b"] - shipped_arrays["part_b"]
        result["medicare_part_b_premium_vs_shipped"] = {
            "rows_changed": int(
                (
                    np.abs(part_b_diff) > float32_tolerance(shipped_arrays["part_b"])
                ).sum()
            ),
            "total_change": total(part_b_diff, p_w),
        }
        result["deduction_vs_shipped"] = {
            "amount": result["deduction"]["amount"] - shipped["deduction"]["amount"],
            "returns": result["deduction"]["returns"] - shipped["deduction"]["returns"],
        }
        for variable in heavy:
            if variable in result and variable in shipped:
                result[f"{variable}_vs_shipped"] = result[variable] - shipped[variable]
        out["scenarios"][name] = result
    for name, result in out["scenarios"].items():
        try:
            result["out_year_candidate"] = json.load(
                open(args.out_prefix + f".{name}.outyear.json")
            )
        except FileNotFoundError:
            pass
    # JCT's tax-expenditure row: income_tax with the deduction neutralized
    # minus income_tax with it, in the same scenario.
    for name, result in out["scenarios"].items():
        try:
            neutralized_path = scenario_paths(name, neutralized=True)[0]
            neutralized = json.load(open(neutralized_path))
        except FileNotFoundError:
            continue
        if os.path.getmtime(neutralized_path) < os.path.getmtime(INPUTS):
            STALE.append(neutralized_path)
            continue
        if "error" in neutralized:
            result["neutralized_error"] = neutralized["error"]
            continue
        if "income_tax" in result and "income_tax" in neutralized:
            result["revenue_loss"] = neutralized["income_tax"] - result["income_tax"]
            result["income_tax_ald_neutralized"] = neutralized["income_tax"]
    # The engine-side readings must reproduce the frozen-input readings exactly.
    differential = {}
    _, a = load_scenario("reported_plus_modeled_part_b")
    for name in ("engine_side", "engine_side_standard_part_b"):
        _, b = load_scenario(name)
        if not a or not b:
            continue
        differential[name] = {
            "premium_base_max_abs_difference": float(
                np.abs(a["se_premiums"] - b["se_premiums"]).max()
            ),
            "ald_person_max_abs_difference": float(
                np.abs(a["ald_person"] - b["ald_person"]).max()
            ),
            "medical_expense_premiums_max_abs_difference": float(
                np.abs(
                    a["medical_expense_premiums"] - b["medical_expense_premiums"]
                ).max()
            ),
        }
        for variable in ("income_tax", "state_income_tax", "snap"):
            if variable in a and variable in b:
                differential[name][f"{variable}_max_abs_difference"] = float(
                    np.abs(a[variable] - b[variable]).max()
                )
                differential[name][f"{variable}_total_difference"] = total(
                    b[variable] - a[variable],
                    data["tax_unit_weight"]
                    if variable != "snap"
                    else data["spm_unit_weight"],
                )
    out["engine_side_vs_frozen_modeled_part_b"] = differential
    out["stale_artifacts_refused"] = STALE

    # Support channel (ASEC respondent vs PUF tax-detail support) per scenario,
    # read from the dataset itself; the engine does not carry the column.
    channel = (
        pd.read_hdf(args.dataset, "person", columns=["person_support_channel"])[
            "person_support_channel"
        ]
        .astype(str)
        .to_numpy()
    )
    tu_w = data["tax_unit_weight"]
    person_tu = data["person_tax_unit"]
    for name, result in out["scenarios"].items():
        if "error" in result:
            continue
        _, arrays = load_scenario(name)
        ald_p = arrays["ald_person"]
        premiums = arrays["se_premiums"]
        result["by_support_channel"] = {
            c: {
                "deduction": total(ald_p, p_w, channel == c),
                "people_with_deduction": float(p_w[(ald_p > 0) & (channel == c)].sum()),
                "premium_base": total(premiums, p_w, channel == c),
            }
            for c in sorted(set(channel))
        }
        # Weight reconciliation: person-weighted person deductions against the
        # tax-unit-weighted tax-unit deduction.
        result["weight_reconciliation"] = {
            "tax_unit_total": result["deduction"]["amount"],
            "person_total": total(ald_p, p_w),
            # Members' person-level deductions the tax unit's total leaves out
            # (on the #10120 head a dependent's own deduction is not the filer's).
            "tax_units_where_member_sum_differs": int(
                (
                    np.abs(
                        np.bincount(person_tu, weights=ald_p, minlength=len(tu_w))
                        - arrays["ald_tax_unit"]
                    )
                    > 0.05
                ).sum()
            ),
            "member_sum_minus_tax_unit_total": total(
                np.bincount(person_tu, weights=ald_p, minlength=len(tu_w))
                - arrays["ald_tax_unit"],
                tu_w,
            ),
        }
    # Possible overlap: an enrollee with no Social Security pays Medicare
    # directly, and the CPS premium question only excludes premiums deducted
    # from Social Security, so their reported premium may already hold Part B.
    enrolled = data["enrolled"]
    reported = data["reported"]
    social_security = data["social_security"]
    group = data["flag"] & enrolled & ~data["has_esi"] & ~(social_security > 0)
    out["possible_reported_part_b_overlap"] = {
        "self_employed_enrollees_outside_esi_without_social_security": float(
            p_w[group].sum()
        ),
        "of_whom_reported_premium_positive": float(p_w[group & (reported > 0)].sum()),
        "their_reported_premium": total(reported, p_w, group & (reported > 0)),
        "their_modeled_part_b": total(
            data["modeled_part_b"], p_w, group & (reported > 0)
        ),
        "rows": int((group & (reported > 0)).sum()),
        "by_support_channel": {
            c: {
                "people": float(p_w[group & (reported > 0) & (channel == c)].sum()),
                "modeled_part_b": total(
                    data["modeled_part_b"], p_w, group & (reported > 0) & (channel == c)
                ),
            }
            for c in sorted(set(channel))
        },
    }
    save_json(args.out_prefix + ".json", out)
    print(json.dumps(out, indent=1))


if args.stage == "prepare":
    prepare()
elif args.stage == "outyear":
    candidate_out_year(args.name)
elif args.stage == "scenario":
    try:
        run_scenario(args.name)
    except Exception as error:
        import traceback

        # Record the failure (an engine-side reading can raise CycleError) so
        # combine reports it; the stage still exits nonzero.
        save_json(
            scenario_paths(args.name, neutralized=args.neutralize, tag=args.tag)[0],
            {
                "name": args.name,
                "error": repr(error)[:2000],
                "traceback": traceback.format_exc()[-3000:],
            },
        )
        raise
else:
    combine()
