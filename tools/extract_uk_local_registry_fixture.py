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
    _scaled_uc_children_by_country,
)
from policyengine_uk_data.utils.uc_data import uc_la_households, uc_pc_households

TIME_PERIOD = 2025


def _normalized_area_name(name):
    """Publisher name normalized for joining: bilingual Welsh labels keep the
    English half, case and stray dots are ignored."""

    return str(name).split(" / ")[0].strip().lower().replace(".", "")


def _corrected_uc_by_code(parsed, name_column, roster):
    """uk-data#468 fix-and-sign: attach UC counts to areas BY NAME, not by row
    position. The incumbent's parsed UC tables carry publisher names only and
    its loss builders consume them positionally against unrelated orderings;
    this join is the correction, applied fixture-side and signed in the
    receipt, because the incumbent getters cannot yet produce aligned values.
    Every roster area must resolve or the extraction fails."""

    by_name = {}
    raw_names = {}
    for name, count in zip(parsed[name_column], parsed["household_count"], strict=True):
        key = _normalized_area_name(name)
        if key in by_name:
            other = raw_names[key]
            if str(other) == str(name):
                raise ValueError(f"uc: duplicate publisher area name {name!r}.")
            raise ValueError(
                "uc: normalization collision - distinct publisher areas "
                f"{other!r} and {name!r} normalize to the same key {key!r}."
            )
        by_name[key] = float(count)
        raw_names[key] = name
    corrected = {}
    missing = []
    for code, roster_name in roster:
        key = _normalized_area_name(roster_name)
        if key in by_name:
            corrected[code] = by_name[key]
        else:
            missing.append(f"{code} ({roster_name})")
    if missing:
        raise ValueError(
            f"uc: {len(missing)} roster area(s) have no publisher row: {missing[:5]}"
        )
    if len(by_name) != len(roster):
        # The reverse direction: publisher rows matching no roster area would
        # otherwise drop silently, letting a superset or renamed-area extract
        # pass (re-review finding 4).
        matched = {_normalized_area_name(name) for _, name in roster}
        orphans = sorted(str(raw_names[key]) for key in by_name if key not in matched)
        raise ValueError(
            f"uc: {len(by_name)} publisher row(s) for a {len(roster)}-area "
            f"roster; unmatched publisher rows: {orphans[:5]}"
        )
    return corrected


def _assert_code_aligned(frame, codes, family):
    """Refuse a source frame that does not carry the roster, in roster order.

    The incumbent's loss builders consume these frames positionally, so the
    fixture must too — but only after proving the positional pairing IS the
    code pairing. A source whose codes disagree with the roster (a partial or
    re-sorted re-extract) fails here instead of silently permuting the
    fixture; see review finding 2 on PR #795 and chronicle#202 for the class.
    """

    if len(frame) != len(codes):
        raise ValueError(
            f"{family}: source has {len(frame)} rows for a {len(codes)}-code roster."
        )
    if "code" not in frame.columns:
        # A source that cannot be code-checked is a fact worth failing on: a
        # bare length comparison passes any permutation of the right size,
        # which is the uk-data#468 failure mode surviving inside the guard
        # written to prevent it (re-review finding 2).
        raise ValueError(
            f"{family}: source carries no 'code' column, so positional "
            "consumption cannot be proven code-aligned."
        )
    if list(frame["code"]) != list(codes):
        raise ValueError(
            f"{family}: source code order disagrees with the roster order."
        )


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

    age_targets = get_constituency_age_targets()
    grain_codes = age_targets["code"].tolist()

    incomes = get_constituency_income_targets()
    _assert_code_aligned(incomes, grain_codes, "hmrc constituency income")
    _income_block(
        incomes, get_national_income_projections(TIME_PERIOD), raw, cal, scalings
    )

    _age_block(age_targets, raw, cal, scalings)

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
    if mapping_matrix.shape != (len(codes), len(grain_codes)):
        raise ValueError(
            f"boundary mapping is {mapping_matrix.shape}; expected "
            f"({len(codes)}, {len(grain_codes)})."
        )
    if len(raw) != len(grain_codes):
        raise ValueError(
            f"pre-mapping frame has {len(raw)} rows on a {len(grain_codes)}-row grain."
        )
    raw_mapped = pd.DataFrame(mapping_matrix @ raw.values, columns=list(raw.columns))
    cal_mapped = pd.DataFrame(mapping_matrix @ cal.values, columns=list(cal.columns))

    # uk-data#468 fix-and-sign: UC is PCON24-native, so under the correction it
    # is attached AFTER the boundary mapping, by name-join to the 2024 roster --
    # never through the 2010->2024 matrix, and never by row position. The child
    # splits recompute from the corrected totals with the incumbent's own
    # country-proportion buckets, keyed by each constituency's true country.
    roster_2024 = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    corrected = _corrected_uc_by_code(
        uc_pc_households,
        "constituency_name",
        list(zip(roster_2024["code"], roster_2024["name"], strict=True)),
    )
    totals = np.array([corrected[code] for code in codes])
    for frame in (raw_mapped, cal_mapped):
        frame["uc_households"] = totals
    scalings["uc_households"] = 1.0

    gb_total = sum(
        value for code, value in corrected.items() if code[0] in ("E", "W", "S")
    )
    country_buckets = _scaled_uc_children_by_country(gb_total)
    split_cols = [
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    ]
    splits = {col: [] for col in split_cols}
    for code in codes:
        proportions = country_buckets.get(code[0], country_buckets["N"])
        shares = proportions / proportions.sum()
        # Largest-remainder allocation: independent per-bucket rounding lets
        # the four splits drift up to +/-2 households from the total, and a
        # reference fixture must prove its parts sum to its whole
        # (re-review finding 1).
        total = round(corrected[code])
        exact = [corrected[code] * share for share in shares]
        floors = [int(value) for value in exact]
        order = sorted(
            range(len(exact)), key=lambda j: exact[j] - floors[j], reverse=True
        )
        for j in order[: total - sum(floors)]:
            floors[j] += 1
        # Assert closure against the value the fixture actually stores as
        # uc_households -- not a locally-rounded copy, which cannot fail on
        # its own construction (round-3 review finding 1). A publisher total
        # that is not integral refuses here instead of drifting silently.
        if sum(floors) != corrected[code]:
            raise ValueError(
                f"uc splits: allocation for {code} sums to {sum(floors)}, "
                f"but the stored uc_households total is {corrected[code]!r}."
            )
        for j, col in enumerate(split_cols):
            splits[col].append(floors[j])
    for col in split_cols:
        for frame in (raw_mapped, cal_mapped):
            frame[col] = np.array(splits[col], dtype=float)
        scalings[col] = 1.0

    return codes, raw_mapped, cal_mapped, scalings, True


def local_authority_surface():
    raw, cal, scalings = pd.DataFrame(), pd.DataFrame(), {}

    la_codes = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
    roster = la_codes["code"].tolist()

    incomes = get_la_income_targets()
    _assert_code_aligned(incomes, roster, "hmrc la income")
    _income_block(
        incomes, get_national_income_projections(TIME_PERIOD), raw, cal, scalings
    )

    age_targets = get_la_age_targets()
    _assert_code_aligned(age_targets, roster, "ons la age")
    _age_block(age_targets, raw, cal, scalings)

    # uk-data#468 fix-and-sign: LA UC totals join by name, never by position.
    corrected_la = _corrected_uc_by_code(
        uc_la_households,
        "la_name",
        list(zip(la_codes["code"], la_codes["name"], strict=True)),
    )
    la_totals = np.array([corrected_la[code] for code in roster])
    raw["uc_households"] = la_totals
    cal["uc_households"] = la_totals
    scalings["uc_households"] = 1.0

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


_UC_METRIC_NAMES = frozenset(
    {
        "uc_households",
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    }
)


def _is_uc_metric(col):
    return col in _UC_METRIC_NAMES


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
                    # UC columns are attached post-mapping on native PCON24
                    # codes under the uk-data#468 correction, so they are
                    # never boundary-mapped even on the constituency surface.
                    "boundary_mapped_from_2010": (
                        boundary_mapped and not _is_uc_metric(col)
                    ),
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
                "with the incumbent's own mapping_matrix. EXCEPTION, fix-and-signed: "
                "the UC family corrects policyengine-uk-data#468 at extraction - "
                "totals join to their areas BY NAME (never by row position), attach "
                "post-mapping on native PCON24/LAD codes, and the child splits "
                "recompute from the corrected totals with the incumbent's own "
                "country-proportion buckets keyed by true country prefix."
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
