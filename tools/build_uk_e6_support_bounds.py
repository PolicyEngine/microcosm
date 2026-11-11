"""Build disclosure-safe E6 UK support bounds from pinned donor tabs.

A bound is derived once per donor tab: while a file's tab digests are
unchanged, writing refuses a re-derivation that moves a rounded bound
(``refuse_rerounding``), since two roundings of one donor extreme narrow it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.etb_services import (
    clean_etb_services_table,
    etb_donor_uprating,
)
from microcosm.build.uk_runtime.etb_services import (
    donor_realized_ranges as etb_services_ranges,
)
from microcosm.build.uk_runtime.etb_vat import (
    clean_etb_vat_table,
    etb_vat_configuration,
)
from microcosm.build.uk_runtime.etb_vat import (
    donor_realized_ranges as etb_vat_ranges,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    clean_lcfs_consumption_table,
    donor_consumption_floor,
    floor_negative_donor_consumption,
    lcfs_donor_uprating,
    lcfs_energy_pricing,
)
from microcosm.build.uk_runtime.lcfs_consumption import (
    donor_realized_ranges as lcfs_ranges,
)
from microcosm.build.uk_runtime.nts_bus_travel import (
    CLEAN_NTS_TRAVEL_TABLES_KIND,
    NTSColumns,
    clean_nts_travel_tables,
    nts_support_ranges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"


def _committed_stage(name: str):
    """The committed UK manifest stage, so bounds use the declared donor uprating."""

    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[name]


def build_lcfs_support_bounds(
    household_tab: Path, person_tab: Path
) -> dict[str, object]:
    hh_sha = _sha256(household_tab)
    person_sha = _sha256(person_tab)
    stage = _committed_stage("lcfs_consumption")
    uprating, _receipt = lcfs_donor_uprating(stage)
    donor = clean_lcfs_consumption_table(
        pd.read_csv(person_tab, sep="\t", low_memory=False),
        pd.read_csv(household_tab, sep="\t", low_memory=False),
        uprating=uprating,
        energy=lcfs_energy_pricing(stage),
    )
    # The stage floors the donor before it reads clip ranges, so the bounds
    # start where the stage's support does.
    donor, _floor_receipt = floor_negative_donor_consumption(
        donor, floor=donor_consumption_floor(stage)
    )
    study = _lcfs_study(stage)
    return _payload(
        source={
            **study,
            "household_tab_sha256": hh_sha,
            "person_tab_sha256": person_sha,
        },
        bounds=lcfs_ranges(donor),
        label="LCFS consumption",
    )


def _lcfs_study(stage) -> dict[str, object]:
    """The LCFS study number and DOI the committed stage pins on both tabs."""

    studies = {
        (artifact.get("ukds_study_number"), artifact.get("doi"))
        for artifact in stage.artifacts
        if artifact.get("role") in ("lcfs_household_tab", "lcfs_person_tab")
    }
    if len(studies) != 1 or None in next(iter(studies)):
        raise SystemExit(
            "The LCFS stage must pin one ukds_study_number and doi on both tabs."
        )
    number, doi = next(iter(studies))
    return {"ukds_study_number": int(number), "doi": str(doi)}


def build_etb_vat_support_bounds(etb_tab: Path) -> dict[str, object]:
    """The declared [0, 1] share range, checked against the donor (uk-data#572).

    The rate is a share of expenditure after VAT and the stage declares its
    bounds, so the committed bound is that range, not a rounded donor extreme.
    """

    sha = _sha256(etb_tab)
    lower, upper = etb_vat_configuration(_committed_stage("etb_vat"))["rate_bounds"]
    donor = clean_etb_vat_table(
        pd.read_csv(etb_tab, sep="\t", low_memory=False), rate_bounds=(lower, upper)
    )
    for column, (low, high) in etb_vat_ranges(donor).items():
        if low < lower or high > upper:
            raise SystemExit(
                f"ETB VAT donor {column} leaves the declared range {[lower, upper]}."
            )
    return _payload(
        source={
            "ukds_study_number": 8856,
            "doi": "10.5255/UKDA-SN-8856-4",
            "tab_sha256": sha,
        },
        bounds={column: (lower, upper) for column in etb_vat_ranges(donor)},
        label="ETB VAT",
        sdc_treatment=(
            "The bounds are the stage's declared [0, 1] share range, which the "
            "donor is checked to lie within; no donor value is committed."
        ),
    )


def build_etb_services_support_bounds(etb_tab: Path) -> dict[str, object]:
    sha = _sha256(etb_tab)
    uprating, _receipt = etb_donor_uprating(_committed_stage("etb_services"))
    donor = clean_etb_services_table(
        pd.read_csv(etb_tab, sep="\t", low_memory=False), uprating=uprating
    )
    return _payload(
        source={
            "ukds_study_number": 8856,
            "doi": "10.5255/UKDA-SN-8856-4",
            "tab_sha256": sha,
        },
        bounds=etb_services_ranges(donor),
        label="ETB services",
    )


def build_nts_support_bounds(
    household_tab: Path,
    individual_tab: Path,
    trip_tab: Path,
    stage_tab: Path,
    ticket_tab: Path,
) -> dict[str, object]:
    """Bounds of the nts_bus_travel person outputs from the pinned NTS tabs (#930)."""

    stage = _committed_stage("nts_bus_travel")
    derive = next(
        dict(op.parameters)
        for op in stage.operations
        if op.kind == CLEAN_NTS_TRAVEL_TABLES_KIND
    )
    columns = NTSColumns.from_parameters(derive)
    cars_clip = tuple(int(v) for v in derive.get("cars_clip", (0, 5)))
    donor = clean_nts_travel_tables(
        pd.read_csv(household_tab, sep="\t", low_memory=False),
        pd.read_csv(individual_tab, sep="\t", low_memory=False),
        pd.read_csv(trip_tab, sep="\t", low_memory=False),
        pd.read_csv(stage_tab, sep="\t", low_memory=False),
        pd.read_csv(ticket_tab, sep="\t", low_memory=False),
        columns=columns,
        cars_clip=(cars_clip[0], cars_clip[1]),
    )
    return _payload(
        source={
            "ukds_study_number": 5340,
            "doi": "10.5255/UKDA-SN-5340-19",
            "household_tab_sha256": _sha256(household_tab),
            "individual_tab_sha256": _sha256(individual_tab),
            "trip_tab_sha256": _sha256(trip_tab),
            "stage_tab_sha256": _sha256(stage_tab),
            "ticket_tab_sha256": _sha256(ticket_tab),
        },
        bounds=nts_support_ranges(donor.person),
        label="NTS bus travel",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcfs-hh-tab", type=Path)
    parser.add_argument("--lcfs-person-tab", type=Path)
    parser.add_argument("--etb-tab", type=Path)
    parser.add_argument("--nts-household-tab", type=Path)
    parser.add_argument("--nts-individual-tab", type=Path)
    parser.add_argument("--nts-trip-tab", type=Path)
    parser.add_argument("--nts-stage-tab", type=Path)
    parser.add_argument("--nts-ticket-tab", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    jobs: list[tuple[Path, dict[str, object]]] = []
    nts = (
        args.nts_household_tab,
        args.nts_individual_tab,
        args.nts_trip_tab,
        args.nts_stage_tab,
        args.nts_ticket_tab,
    )
    if any(nts):
        if not all(nts):
            raise SystemExit("NTS support bounds require all five NTS tabs.")
        jobs.append(
            (
                UK_PACKAGE / "nts_bus_travel_support_bounds.json",
                build_nts_support_bounds(*nts),
            )
        )
    if args.lcfs_hh_tab or args.lcfs_person_tab:
        if not args.lcfs_hh_tab or not args.lcfs_person_tab:
            raise SystemExit("LCFS support bounds require both LCFS tabs.")
        jobs.append(
            (
                UK_PACKAGE / "lcfs_consumption_support_bounds.json",
                build_lcfs_support_bounds(args.lcfs_hh_tab, args.lcfs_person_tab),
            )
        )
    if args.etb_tab:
        jobs.extend(
            [
                (
                    UK_PACKAGE / "etb_vat_support_bounds.json",
                    build_etb_vat_support_bounds(args.etb_tab),
                ),
                (
                    UK_PACKAGE / "etb_services_support_bounds.json",
                    build_etb_services_support_bounds(args.etb_tab),
                ),
            ]
        )
    if not jobs:
        raise SystemExit("No support-bound inputs supplied.")
    for path, payload in jobs:
        rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        if args.check:
            if path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"{path} is stale.")
        else:
            refuse_rerounding(path, payload)
            path.write_text(rendered, encoding="utf-8")
    return 0


def refuse_rerounding(path: Path, payload: dict[str, object]) -> None:
    """Refuse to move a committed rounded bound while its donor tabs are unchanged.

    Each committed bound is the donor's extreme rounded outward. Re-derived
    from the same tabs after a change in the declared uprating, the cleaning
    or the floor, a bound that rounds to a different value tells the reader,
    with the first, that the extreme lies between the two thresholds, each
    divided by its own factor: a narrow interval around a unit record (UKDS
    EUL disclosure control; microcosm#1164). A lower bound may move to zero,
    the floor, which reads no donor value; any other move needs new donor tabs
    or a bound that reads no donor value either.
    """

    if not path.exists():
        return
    committed = json.loads(path.read_text(encoding="utf-8"))

    def digests(source: object) -> dict[str, object]:
        if not isinstance(source, dict):
            return {}
        return {k: v for k, v in source.items() if str(k).endswith("sha256")}

    if not digests(committed.get("source")) or digests(
        committed.get("source")
    ) != digests(payload.get("source")):
        return
    old = committed.get("bounds") or {}
    new = payload.get("bounds") or {}
    moved = sorted(
        column
        for column in set(old) & set(new)
        if list(old[column]) != list(new[column])
        and not (list(old[column])[1] == list(new[column])[1] and new[column][0] == 0)
    )
    if moved:
        raise SystemExit(
            f"{path.name}: the donor tabs are unchanged but the rounded bounds of "
            f"{moved} would move. Committing a second rounding of the same donor "
            "extremes narrows them (UKDS disclosure control); keep the committed "
            "bounds or replace them with ones that read no donor value."
        )


def _payload(
    *,
    source: dict[str, object],
    bounds: dict[str, tuple[float, float]],
    label: str,
    sdc_treatment: str | None = None,
) -> dict[str, object]:
    return {
        "version": 1,
        "country": "uk",
        "policy": (
            f"Disclosure-safe outward-rounded {label} support bounds generated "
            "from pinned licensed donor tabs. Values are rounded outward to "
            "one significant figure; exact donor min/max values are not committed."
        ),
        "source": {
            **source,
            "sdc_treatment": sdc_treatment
            or (
                "Exact donor min/max values are rounded outward to one "
                "significant figure before commit."
            ),
        },
        "bounds": {
            column: list(_outward_round(pair))
            for column, pair in sorted(bounds.items())
        },
        "chronicle": [f"Support bounds generated for {label} from source SHA pins."],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outward_round(bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = bounds
    return (_round_down(lo), _round_up(hi))


def _round_down(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    return math.floor(value / magnitude) * magnitude


def _round_up(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    return math.ceil(value / magnitude) * magnitude


if __name__ == "__main__":
    raise SystemExit(main())
