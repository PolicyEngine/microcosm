"""Extract the incumbent UK local target surface (fixture B) at pin 12a1e028.

Reproduces the y-side of both local loss builders WITHOUT a Microsimulation: every
target value in those builders comes from committed CSV/XLSX sources plus declared
scalings, so the model side (which needs licensed microdata) is not required.

For each bound column we emit BOTH:
  raw_value        - the published statistic before the builder's national-consistency
                     scaling (boundary-mapped for constituencies, so it is comparable
                     row-for-row with the calibrated value)
  calibrated_value - the value the incumbent actually calibrates against

so a parity comparison can attribute a difference to the source statistic or to the
incumbent's solve-time scaling, instead of reporting the whole surface as drift.

Constituency rows are mapped 2010 -> PCON24 with the incumbent's own mapping_matrix.
LA rows are already at LAD-2021 codes and are consumed positionally by the incumbent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_uk_data.datasets.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)
from policyengine_uk_data.datasets.local_areas.constituencies.devolved_housing import (
    add_private_rent_targets,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.local_age import (
    get_constituency_age_targets,
    get_la_age_targets,
    get_uk_total_population,
)
from policyengine_uk_data.targets.sources.local_income import (
    INCOME_VARIABLES,
    get_constituency_income_targets,
    get_la_income_targets,
    get_national_income_projections,
)
from policyengine_uk_data.targets.sources.local_la_extras import (
    get_ons_income_uprating_factors,
    load_household_counts,
    load_ons_la_income,
    load_private_rents,
    load_tenure_data,
)
from policyengine_uk_data.targets.sources.local_uc import (
    get_constituency_uc_by_children_targets,
    get_constituency_uc_targets,
    get_la_uc_targets,
)

TIME_PERIOD = 2025


def _income_block(incomes, national_incomes, raw, cal, scalings):
    for income_variable in INCOME_VARIABLES:
        local_targets = incomes[f"{income_variable}_amount"].values
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_amount"].iloc[0]
        adjustment = national_target / local_targets.sum()
        scalings[f"hmrc/{income_variable}/amount"] = float(adjustment)
        scalings[f"hmrc/{income_variable}/count"] = float(adjustment)

        raw[f"hmrc/{income_variable}/amount"] = local_targets
        cal[f"hmrc/{income_variable}/amount"] = local_targets * adjustment
        counts = incomes[f"{income_variable}_count"].values
        raw[f"hmrc/{income_variable}/count"] = counts
        cal[f"hmrc/{income_variable}/count"] = counts * adjustment


def _age_block(age_targets, raw, cal, scalings):
    uk_total_population = get_uk_total_population(TIME_PERIOD)
    age_cols = [c for c in age_targets.columns if c.startswith("age/")]
    targets_total_pop = sum(age_targets[c].values.sum() for c in age_cols)
    factor = uk_total_population / targets_total_pop * 0.9
    for col in age_cols:
        raw[col] = age_targets[col].values
        cal[col] = age_targets[col].values * factor
        scalings[col] = float(factor)
    return age_cols


def constituency_surface():
    raw, cal, scalings = pd.DataFrame(), pd.DataFrame(), {}

    incomes = get_constituency_income_targets()
    _income_block(
        incomes, get_national_income_projections(TIME_PERIOD), raw, cal, scalings
    )

    age_targets = get_constituency_age_targets()
    _age_block(age_targets, raw, cal, scalings)

    uc = get_constituency_uc_targets().values
    raw["uc_households"] = uc
    cal["uc_households"] = uc
    scalings["uc_households"] = 1.0

    uc_children = get_constituency_uc_by_children_targets()
    for col in uc_children.columns:
        raw[col] = uc_children[col].values
        cal[col] = uc_children[col].values
        scalings[col] = 1.0

    # Devolved housing: the y-side needs only the age-target population shares and
    # the module's hardcoded country totals; the array arguments feed the matrix side
    # only, so zero-length-safe placeholders are passed.
    n = len(age_targets)
    placeholder = np.zeros(n)
    for frame in (raw, cal):
        add_private_rent_targets(
            pd.DataFrame(index=range(n)),
            frame,
            age_targets,
            country=np.array([""] * n),
            tenure_type=np.array([""] * n),
            rent=placeholder,
        )
    for col in cal.columns:
        scalings.setdefault(col, 1.0)

    # 2010 -> PCON24, applied to both frames so they stay row-comparable.
    codes = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv").code.tolist()
    raw_mapped = pd.DataFrame(mapping_matrix @ raw.values, columns=list(raw.columns))
    cal_mapped = pd.DataFrame(mapping_matrix @ cal.values, columns=list(cal.columns))
    return codes, raw_mapped, cal_mapped, scalings, True


def local_authority_surface():
    raw, cal, scalings = pd.DataFrame(), pd.DataFrame(), {}

    incomes = get_la_income_targets()
    _income_block(
        incomes, get_national_income_projections(TIME_PERIOD), raw, cal, scalings
    )

    age_targets = get_la_age_targets()
    _age_block(age_targets, raw, cal, scalings)

    uc = get_la_uc_targets().values
    raw["uc_households"] = uc
    cal["uc_households"] = uc
    scalings["uc_households"] = 1.0

    la_codes = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")

    # The remaining LA families are produced inline in the incumbent's loss builder
    # rather than by importable target functions, so their y-side is replayed here.
    ons_income = load_ons_la_income()
    households_by_la = load_household_counts()
    ons_merged = la_codes.merge(
        ons_income, left_on="code", right_on="la_code", how="left"
    ).merge(
        households_by_la,
        left_on="code",
        right_on="la_code",
        how="left",
        suffixes=("", "_hh"),
    )
    bhc_factor, housing_factor = get_ons_income_uprating_factors(TIME_PERIOD)
    bhc_raw = ons_merged["net_income_bhc"] * ons_merged["households"]
    housing_raw = (
        ons_merged["net_income_bhc"] * ons_merged["households"]
        - ons_merged["net_income_ahc"] * ons_merged["households"]
    )
    has_ons = (
        ons_merged["net_income_bhc"].notna() & ons_merged["households"].notna()
    ).values
    bhc_cal = bhc_raw * bhc_factor
    housing_cal = housing_raw * housing_factor
    for col, rvals, cvals, factor in (
        ("ons/equiv_net_income_bhc", bhc_raw, bhc_cal, bhc_factor),
        ("ons/equiv_housing_costs", housing_raw, housing_cal, housing_factor),
        (
            "ons/equiv_net_income_ahc",
            bhc_raw - housing_raw,
            bhc_cal - housing_cal,
            None,
        ),
    ):
        raw[col] = np.where(has_ons, rvals.values, np.nan)
        cal[col] = np.where(has_ons, cvals.values, np.nan)
        scalings[col] = float(factor) if factor is not None else None

    tenure_merged = la_codes.merge(
        load_tenure_data(), left_on="code", right_on="la_code", how="left"
    ).merge(
        households_by_la,
        left_on="code",
        right_on="la_code",
        how="left",
        suffixes=("", "_hh"),
    )
    has_tenure = (
        tenure_merged["owned_outright_pct"].notna()
        & tenure_merged["households"].notna()
    ).values
    for tenure_key, pct_col in (
        ("owned_outright", "owned_outright_pct"),
        ("owned_mortgage", "owned_mortgage_pct"),
        ("private_rent", "private_rent_pct"),
        ("social_rent", "social_rent_pct"),
    ):
        targets = tenure_merged[pct_col] / 100 * tenure_merged["households"]
        col = f"tenure/{tenure_key}"
        raw[col] = np.where(has_tenure, targets.values, np.nan)
        cal[col] = np.where(has_tenure, targets.values, np.nan)
        scalings[col] = 1.0

    tenure_merged = tenure_merged.merge(
        load_private_rents(), left_on="code", right_on="area_code", how="left"
    )
    rent_target = (
        tenure_merged["median_annual_rent"]
        * tenure_merged["private_rent_pct"]
        / 100
        * tenure_merged["households"]
    )
    has_rent = (
        tenure_merged["median_annual_rent"].notna()
        & tenure_merged["private_rent_pct"].notna()
        & tenure_merged["households"].notna()
    ).values
    raw["rent/private_rent"] = np.where(has_rent, rent_target.values, np.nan)
    cal["rent/private_rent"] = np.where(has_rent, rent_target.values, np.nan)
    scalings["rent/private_rent"] = 1.0

    # Council tax: bound only when the storage CSV is present (file-existence gate).
    ct_path = STORAGE_FOLDER / "la_council_tax.csv"
    if ct_path.exists():
        ct_data = pd.read_csv(ct_path)
        ct_columns = ["code"] + [f"count_band_{b}" for b in "ABCDEFGH"]
        if "total_council_tax_net" in ct_data.columns:
            ct_columns.append("total_council_tax_net")
        ct_merged = la_codes.merge(ct_data[ct_columns], on="code", how="left")
        for band in "ABCDEFGH":
            col = f"voa/council_tax/{band}"
            csv_col = f"count_band_{band}"
            values = np.where(
                ct_merged[csv_col].notna().values, ct_merged[csv_col].values, np.nan
            )
            raw[col] = values
            cal[col] = values
            scalings[col] = 1.0
        if "total_council_tax_net" in ct_merged.columns:
            values = np.where(
                ct_merged["total_council_tax_net"].notna().values,
                ct_merged["total_council_tax_net"].values,
                np.nan,
            )
            raw["housing/council_tax_net"] = values
            cal["housing/council_tax_net"] = values
            scalings["housing/council_tax_net"] = 1.0

    return la_codes.code.tolist(), raw, cal, scalings, False


def rows_for(area_type, codes, raw, cal, scalings, boundary_mapped):
    out = []
    for col in cal.columns:
        for i, code in enumerate(codes):
            rv = float(raw[col].values[i]) if col in raw.columns else None
            cv = float(cal[col].values[i])
            if not np.isfinite(cv):
                continue
            out.append(
                {
                    "name": f"{col}@{code}",
                    "metric": col,
                    "area_type": area_type,
                    "geography_id": code,
                    "period": TIME_PERIOD,
                    "raw_value": rv if rv is not None and np.isfinite(rv) else None,
                    "value": cv,
                    "adjustment_factor": scalings.get(col),
                    "boundary_mapped_from_2010": boundary_mapped,
                }
            )
    return out


def main():
    out_path = Path(sys.argv[1])
    rows = []
    codes, raw, cal, scalings, mapped = constituency_surface()
    rows += rows_for("constituency", codes, raw, cal, scalings, mapped)
    codes, raw, cal, scalings, mapped = local_authority_surface()
    rows += rows_for("local_authority", codes, raw, cal, scalings, mapped)

    metrics = sorted({r["metric"] for r in rows})
    payload = {
        "schema_version": 1,
        "country": "uk",
        "fixture": "incumbent_local_2025",
        "provenance": {
            "source_repository": "policyengine-uk-data",
            "pinned_ref": "12a1e028afeef08d8b2d74ee03fd9de3a78b2dd3",
            "pinned_version": "1.56.16",
            "method": (
                "y-side replay of datasets/local_areas/{constituencies,local_authorities}"
                "/loss.py: published statistics with the builders' declared national-"
                "consistency scalings; constituency rows boundary-mapped 2010->PCON24 "
                "with the incumbent's own mapping_matrix."
            ),
            "time_period": TIME_PERIOD,
        },
        "surface": {
            "row_count": len(rows),
            "metric_count": len(metrics),
            "metrics": metrics,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "metrics": len(metrics)}))


if __name__ == "__main__":
    main()
