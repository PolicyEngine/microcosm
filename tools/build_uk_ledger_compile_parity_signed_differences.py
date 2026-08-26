"""Regenerate UK Ledger compile-parity signed-difference resources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path

from microcosm.build.gates import ledger_compile_parity_signed_differences
from microcosm.build.uk_runtime.ledger_targets import (
    LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE,
    align_uk_local_registry_parity_fixture,
    compile_uk_local_target_registry,
    compile_uk_target_registry,
)

UK_PACKAGE = "microcosm.build.uk"
UK_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk"
)


@dataclass(frozen=True)
class ParityReceiptSpec:
    fixture_resource: str
    output_resource: str
    target_period: int
    surface: str = "national"


RECEIPTS = (
    ParityReceiptSpec(
        fixture_resource="parity_fixture_production_2023.json",
        output_resource="ledger_compile_parity_production_2023_signed_differences.json",
        target_period=2023,
    ),
    ParityReceiptSpec(
        fixture_resource="registry_parity_fixture_2025.json",
        output_resource="ledger_compile_parity_incumbent_2025_signed_differences.json",
        target_period=2025,
    ),
    ParityReceiptSpec(
        fixture_resource=LOCAL_REGISTRY_PARITY_FIXTURE_RESOURCE,
        output_resource=(
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json"
        ),
        target_period=2025,
        surface="local",
    ),
)

_CGT_GAINS_TOTAL_RATIONALE = (
    "Ledger carries the HMRC 2023-24 outturn value GBP 65,937,000,000 and "
    "holds it by identity under the current doctrine; the incumbent Fixture B "
    "row is GBP 67,727,478,991.60 at 2025 because it carries a forecast/uprated "
    "value. Signed as a doctrine consequence, not a binding error."
)

_ONS_TERMINAL_BAND_RATIONALES = {
    "ons.population.female_85_89": (
        "María ruling 2026-08-21: incumbent ons/female_85_90 maps to the "
        "uniform female 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.male_85_89": (
        "María ruling 2026-08-21: incumbent ons/male_85_90 maps to the "
        "uniform male 85_89 declaration. The value gap is the single-age-90 "
        "share; ages 90+ are constrained separately."
    ),
    "ons.population.female_90_plus": (
        "María ruling 2026-08-21: new female 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
    "ons.population.male_90_plus": (
        "María ruling 2026-08-21: new male 90_plus tail constrains ages 90+; "
        "the incumbent six-year terminal band left ages 91+ unconstrained."
    ),
}

_DEVOLVED_RENT_FIXTURE_ONLY_RATIONALE = (
    "Ruled signed exclusion: the incumbent y-values for devolved private rent "
    "are hardcoded constants in devolved_housing.py with no source or year, "
    "then allocated by population share. Adopting them would violate the "
    "facts-only doctrine; a Chronicle facts request covers sourced Scottish "
    "Government and StatsWales private-rent statistics."
)

_COUNCIL_TAX_FIXTURE_ONLY_RATIONALE = (
    "Ruled signed exclusion: the local contract declares no council-tax "
    "targets and Microcosm has no council-tax local metric computations. The "
    "wave-3 VOA/MHCLG/Scottish Government/Welsh Government facts are scoped "
    "as follow-up work, including the contract targets and metric bindings."
)

_DEVOLVED_RENT_METRICS = frozenset(
    {
        "housing/wales_private_renter_households",
        "housing/wales_private_rent_amount",
        "housing/scotland_private_renter_households",
        "housing/scotland_private_rent_amount",
    }
)

_COUNCIL_TAX_METRICS = frozenset(
    {"housing/council_tax_net"} | {f"voa/council_tax/{band}" for band in "ABCDEFGH"}
)

_UC_METRICS = frozenset(
    {
        "uc_households",
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    }
)

_UC_DRIFT_RATIONALE = (
    "Incumbent-defect class, not calibration drift: policyengine-uk-data#468 — "
    "the incumbent's local UC targets are positionally misaligned (name-only "
    "spreadsheet rows joined by position to code files in a different order; "
    "the constituency vector is PCON24-native yet boundary-mapped 2010->2024 on "
    "top). Verified by name-join: the incumbent reproduces the same Stat-Xplore "
    "publication our facts carry at its uniform national rescale (~0.898, max "
    "per-area deviation 0.46%), so the per-area disagreement is the incumbent's "
    "permutation, not a value dispute. Our side is the published per-area count."
)

_UC_NI_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: DWP Stat-Xplore publishes UC for Great Britain only, so "
    "the Northern Ireland areas have no UC facts in the pinned feed. The "
    "incumbent fills NI from the DfC May-2025 supplementary tables, which are "
    "requested as a Chronicle port in chronicle#200; the child splits stay "
    "deferred regardless (no NI child-count source; the incumbent imputes them "
    "from GB proportions)."
)

_AGE_DRIFT_RATIONALE = (
    "Vintage and scaling class: ours holds the mid-2024 per-area publisher "
    "estimates at identity; the incumbent scales its age columns by "
    "uk_total_population / targets_total_pop * 0.9 (a bare literal) and, at "
    "constituency grain, boundary-maps 2010-vintage rows to PCON24."
)

_HMRC_DRIFT_RATIONALE = (
    "Scaling and operation class: both sides read the same SPI tables for tax "
    "year 2023-24; ours resolves counts at identity and amounts as the "
    "declared count_x_mean operation, while the incumbent multiplies every "
    "column by a national income-projection ratio for the target year."
)

_TENURE_DRIFT_RATIONALE = (
    "Vintage class: ours holds the per-nation census tenure counts (E&W 2021, "
    "Scotland 2022, NI 2021) at identity with uprating declared as holds; the "
    "incumbent derives counts as percentage shares times a separate "
    "household-count workbook."
)

_EQUIV_INCOME_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: ONS publishes small-area income at MSOA grain only — the "
    "feed carries no local-authority rows, and MSOA-to-LA aggregation of a "
    "mean-valued, model-output series is ruled downstream build work behind the "
    "frs_model_based_target_circularity and ons_bhc_ahc_noncomparable fences."
)

_PRIVATE_RENT_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: every PIPR fact in the pinned feed is dated 2026-06, "
    "after the 2025 target period; Scotland is published at BRMA grain and "
    "Northern Ireland is absent. chronicle#200 asks for the 2025 months, which "
    "the package's preserved source rows already carry."
)

_LEDGER_ONLY_RATIONALE = (
    "Coverage the incumbent lacks: our side compiles a published per-area fact "
    "the incumbent's surface NaN-masks or never carries (for example Northern "
    "Ireland SPI and tenure rows, and the crosswalk area absent from its "
    "360-row LA file)."
)

_LOCAL_DRIFT_RATIONALES = {
    **{metric: _UC_DRIFT_RATIONALE for metric in _UC_METRICS},
    **{f"age/{lower}_{lower + 10}": _AGE_DRIFT_RATIONALE for lower in range(0, 80, 10)},
    **{
        f"hmrc/{variable}/{measure}": _HMRC_DRIFT_RATIONALE
        for variable in ("employment_income", "self_employment_income")
        for measure in ("amount", "count")
    },
    **{
        f"tenure/{key}": _TENURE_DRIFT_RATIONALE
        for key in ("owned_outright", "owned_mortgage", "private_rent", "social_rent")
    },
}

_SPI_CELL_FIXTURE_ONLY_RATIONALE = (
    "Signed deferral: the publisher's SPI area tables lack the cell upstream — "
    "E14001416 publishes a self-employment count with no mean (so count_x_mean "
    "cannot resolve), and two local authorities carry no SPI target measures. "
    "The incumbent's fixture row for these cells is a boundary-mapped blend, "
    "not a published cell."
)

_LOCAL_FIXTURE_ONLY_RATIONALES = {
    **{metric: _UC_NI_FIXTURE_ONLY_RATIONALE for metric in _UC_METRICS},
    **{
        f"hmrc/{variable}/{measure}": _SPI_CELL_FIXTURE_ONLY_RATIONALE
        for variable in ("employment_income", "self_employment_income")
        for measure in ("amount", "count")
    },
    **{
        f"ons/{key}": _EQUIV_INCOME_FIXTURE_ONLY_RATIONALE
        for key in (
            "equiv_net_income_bhc",
            "equiv_net_income_ahc",
            "equiv_housing_costs",
        )
    },
    "rent/private_rent": _PRIVATE_RENT_FIXTURE_ONLY_RATIONALE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Chronicle consumer JSONL artifact used to compile UK references.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UK_PACKAGE_DIR,
        help="Directory receiving the signed-difference JSON resources.",
    )
    parser.add_argument(
        "--surface",
        choices=("all", "national", "local"),
        default="all",
        help="Receipt surface to regenerate. Use 'local' to avoid national receipt churn.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_fixture(resource: str) -> dict:
    return json.loads(
        importlib_resources.files(UK_PACKAGE).joinpath(resource).read_text()
    )


def _load_crosswalk() -> dict:
    return json.loads(
        importlib_resources.files(UK_PACKAGE)
        .joinpath("local_area_crosswalk.json")
        .read_text()
    )


def _local_metric_by_target_id() -> dict[str, str]:
    contract = json.loads(
        importlib_resources.files(UK_PACKAGE)
        .joinpath("uk_local_geography_targets.json")
        .read_text()
    )
    mapping: dict[str, str] = {}
    for target in contract.get("targets", ()):
        binding = target.get("bindings", {}).get("policyengine", {})
        metric = binding.get("metric_name")
        if metric:
            mapping[str(target["target_id"])] = str(metric)
    return mapping


def _aligned_fixture(fixture: dict) -> dict:
    """Apply current contract-row names to retired-fixture rows before comparison."""

    rows = []
    for row in fixture.get("rows", ()):
        if not isinstance(row, dict):
            rows.append(row)
            continue
        updated = dict(row)
        if updated.get("name") == "ons/female_85_90":
            updated["contract_target_id"] = "ons.population.female_85_89"
            updated["measure"] = "ons.population.female_85_89"
        elif updated.get("name") == "ons/male_85_90":
            updated["contract_target_id"] = "ons.population.male_85_89"
            updated["measure"] = "ons.population.male_85_89"
        rows.append(updated)
    aligned = dict(fixture)
    aligned["rows"] = rows
    return aligned


def _add_signed_rationale_notes(
    report: dict[str, object],
    *,
    fixture_resource: str,
) -> None:
    if fixture_resource != "registry_parity_fixture_2025.json":
        return
    for row in report.get("differences", ()):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if name == "hmrc.cgt.gains_total":
            row["reason"] = _CGT_GAINS_TOTAL_RATIONALE
        elif name in _ONS_TERMINAL_BAND_RATIONALES:
            row["reason"] = _ONS_TERMINAL_BAND_RATIONALES[name]


def _add_local_signed_rationale_notes(report: dict[str, object]) -> None:
    """Replace generic classifier reasons with the ruled per-family rationales.

    Every rationale names its cause (an adjudicated exclusion, a signed
    deferral, a diagnosed incumbent defect, or a declared scaling/vintage
    class). A row whose (metric, kind) has no ruled rationale keeps the
    classifier's generic reason — never invent one.
    """

    metric_by_target_id = _local_metric_by_target_id()
    for row in report.get("differences", ()):
        if not isinstance(row, dict):
            continue
        prefix = str(row.get("name", "")).split("@", 1)[0]
        metric = metric_by_target_id.get(prefix, prefix)
        kind = str(row.get("kind", ""))
        if metric in _DEVOLVED_RENT_METRICS:
            row["reason"] = _DEVOLVED_RENT_FIXTURE_ONLY_RATIONALE
        elif metric in _COUNCIL_TAX_METRICS:
            row["reason"] = _COUNCIL_TAX_FIXTURE_ONLY_RATIONALE
        elif kind == "calibration_drift" and metric in _LOCAL_DRIFT_RATIONALES:
            row["reason"] = _LOCAL_DRIFT_RATIONALES[metric]
        elif kind == "fixture_only" and metric in _LOCAL_FIXTURE_ONLY_RATIONALES:
            row["reason"] = _LOCAL_FIXTURE_ONLY_RATIONALES[metric]
        elif kind == "ledger_only":
            row["reason"] = _LEDGER_ONLY_RATIONALE


def _compile_for_receipt(spec: ParityReceiptSpec, facts: list[dict]):
    if spec.surface == "national":
        return compile_uk_target_registry(
            facts,
            target_period=spec.target_period,
        )
    if spec.surface == "local":
        return compile_uk_local_target_registry(
            facts,
            target_period=spec.target_period,
            crosswalk=_load_crosswalk(),
        )
    raise ValueError(f"unknown parity receipt surface {spec.surface!r}.")


def _fixture_for_receipt(spec: ParityReceiptSpec) -> dict:
    fixture = _aligned_fixture(_load_fixture(spec.fixture_resource))
    if spec.surface == "local":
        return align_uk_local_registry_parity_fixture(fixture)
    return fixture


def main() -> None:
    args = _parse_args()
    facts = _load_jsonl(args.ledger_facts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in RECEIPTS:
        if args.surface != "all" and spec.surface != args.surface:
            continue
        compilation = _compile_for_receipt(spec, facts)
        report = ledger_compile_parity_signed_differences(
            compilation.registry,
            _fixture_for_receipt(spec),
            unsupported=compilation.unsupported,
        )
        _add_signed_rationale_notes(report, fixture_resource=spec.fixture_resource)
        if spec.surface == "local":
            _add_local_signed_rationale_notes(report)
        output_path = args.output_dir / spec.output_resource
        output_path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "compiled_count": report["compiled_count"],
                    "counts_by_kind": report["counts_by_kind"],
                    "difference_count": report["difference_count"],
                    "fixture": spec.fixture_resource,
                    "fixture_count": report["fixture_count"],
                    "output": spec.output_resource,
                    "surface": spec.surface,
                    "target_period": spec.target_period,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
