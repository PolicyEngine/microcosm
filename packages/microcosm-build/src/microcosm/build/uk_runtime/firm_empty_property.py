"""UK firm empty property: published MHCLG figures, read but not assigned.

policyengine-uk reads two business rates inputs for empty property in
England:

- ``firm_property_occupied`` (bool, default true): whether the firm's
  property is occupied throughout the year. An unoccupied property is
  treated as empty all year and gets the empty property exemption but no
  other relief;
- ``firm_months_empty`` (int, default 0): months the property had already
  been continuously empty at the start of the year. The exemption is gross
  rates times ``max(3 - months, 0) / 12`` (6 months for
  ``firm_is_industrial_property``), or the whole year when rateable value is
  below the £2,900 small property threshold.

Both stay unset (policyengine-uk's defaults: every property occupied). This
module only reads MHCLG's *National non-domestic rates collected by councils
in England* figures, extracted by
``tools/extract_uk_firm_empty_property_tables.py`` to
``tests/fixtures/uk/firm_empty_property_2024_25/``, and reports them through
``uk_firm_empty_property_published_figures``. The figures do not support an
assignment:

- they are a point-in-time stock (hereditaments in receipt of empty premises
  relief on 31 December), while policyengine-uk's flag makes a property
  empty for the whole year; empties past their exemption pay full rates and
  appear in no published count;
- an empty hereditament is not a trading firm: its ratepayer is the owner,
  who may hold it outside any business the generator draws;
- no source gives empty durations, so ``firm_months_empty`` could only be
  tuned to a relief total, not estimated;
- no source gives the rateable value of empty property or empty shares by
  property type.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

UK_FIRM_EMPTY_HEREDITAMENTS_FILE = "mhclg_nndr1_empty_hereditaments.csv"
UK_FIRM_EMPTY_RELIEF_FILE = "mhclg_nndr1_empty_relief_value.csv"
UK_FIRM_EMPTY_NNDR3_FILE = "mhclg_nndr3_2024_25.csv"
UK_FIRM_EMPTY_FILES = (
    UK_FIRM_EMPTY_HEREDITAMENTS_FILE,
    UK_FIRM_EMPTY_RELIEF_FILE,
    UK_FIRM_EMPTY_NNDR3_FILE,
)
UK_FIRM_EMPTY_SNAPSHOT = "2024-12-31"
UK_FIRM_EMPTY_RELIEF_YEAR = "2024-25"
#: The relief categories policyengine-uk's exemption could represent: empty
#: property above the exemption threshold, industrial and non-industrial.
UK_FIRM_EMPTY_MODELLED_HEREDITAMENTS = ("hdit_mr_emptyind", "hdit_mr_emptyabove")
UK_FIRM_EMPTY_MODELLED_RELIEF = ("value_emptyind", "value_emptyabove")
#: Status of ``firm_property_occupied`` and ``firm_months_empty``.
UK_FIRM_EMPTY_STATUS = "not_set"
UK_FIRM_EMPTY_GAP = (
    "firm_property_occupied and firm_months_empty stay unset: MHCLG counts "
    "hereditaments in receipt of empty relief on one date, policyengine-uk's "
    "flag is for the whole year, empty hereditaments are not trading firms, "
    "and months empty could only be calibrated to a relief total"
)


@dataclass(frozen=True)
class UKFirmEmptyPropertySource:
    #: snapshot date -> variable -> England hereditaments.
    hereditaments: Mapping[str, Mapping[str, float]]
    #: forecast year -> variable -> England relief (£, positive).
    relief_gbp: Mapping[str, Mapping[str, float]]
    #: NNDR3 2024-25 item -> value in its published unit.
    nndr3: Mapping[str, float]
    sha256: Mapping[str, str]


def _csv(directory: Path, name: str) -> tuple[pd.DataFrame, str]:
    path = directory / name
    payload = path.read_bytes()
    return pd.read_csv(path, dtype=str, keep_default_na=False), hashlib.sha256(
        payload
    ).hexdigest()


def _float(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"MHCLG {label} {value!r} is not a number.") from None
    if not np.isfinite(number):
        raise ValueError(f"MHCLG {label} must be finite.")
    return number


def read_uk_firm_empty_property_source(
    path: str | Path,
) -> UKFirmEmptyPropertySource:
    """Read the extracted MHCLG CSVs from a fixture directory."""

    directory = Path(path)
    counts, counts_sha = _csv(directory, UK_FIRM_EMPTY_HEREDITAMENTS_FILE)
    relief, relief_sha = _csv(directory, UK_FIRM_EMPTY_RELIEF_FILE)
    nndr3, nndr3_sha = _csv(directory, UK_FIRM_EMPTY_NNDR3_FILE)
    hereditaments: dict[str, dict[str, float]] = {}
    for record in counts.to_dict("records"):
        value = _float(record["england_hereditaments"], record["variable"])
        if value < 0:
            raise ValueError(f"MHCLG {record['variable']} must be non-negative.")
        hereditaments.setdefault(record["snapshot_date"], {})[record["variable"]] = (
            value
        )
    values: dict[str, dict[str, float]] = {}
    for record in relief.to_dict("records"):
        # The 2025-26 edition prints relief with a minus sign.
        value = abs(_float(record["england_value_gbp"], record["variable"]))
        values.setdefault(record["forecast_year"], {})[record["variable"]] = value
    items = {
        record["item"]: _float(record["value"], record["item"])
        for record in nndr3.to_dict("records")
    }
    if UK_FIRM_EMPTY_SNAPSHOT not in hereditaments:
        raise ValueError(f"MHCLG counts lack the {UK_FIRM_EMPTY_SNAPSHOT} snapshot.")
    if UK_FIRM_EMPTY_RELIEF_YEAR not in values:
        raise ValueError(f"MHCLG relief lacks {UK_FIRM_EMPTY_RELIEF_YEAR}.")
    for snapshot, row in hereditaments.items():
        parts = sum(v for k, v in row.items() if k != "hdit_mr_empty")
        if parts != row.get("hdit_mr_empty"):
            raise ValueError(f"MHCLG {snapshot} empty categories do not sum.")
    return UKFirmEmptyPropertySource(
        hereditaments=MappingProxyType(
            {k: MappingProxyType(v) for k, v in hereditaments.items()}
        ),
        relief_gbp=MappingProxyType(
            {k: MappingProxyType(v) for k, v in values.items()}
        ),
        nndr3=MappingProxyType(items),
        sha256=MappingProxyType(
            {
                UK_FIRM_EMPTY_HEREDITAMENTS_FILE: counts_sha,
                UK_FIRM_EMPTY_RELIEF_FILE: relief_sha,
                UK_FIRM_EMPTY_NNDR3_FILE: nndr3_sha,
            }
        ),
    )


def uk_firm_empty_property_published_figures(
    source: UKFirmEmptyPropertySource,
) -> dict[str, Any]:
    """The published England stock and relief figures, with the gap stated.

    Hereditaments in receipt of empty relief on 31 December 2024 (total and
    the two above-threshold categories) and their share of the list at 30
    September; empty relief in 2024-25 (NNDR1 forecast, total and the two
    categories, and NNDR3 outturn) and its share of gross rates payable.
    """

    counts = source.hereditaments[UK_FIRM_EMPTY_SNAPSHOT]
    relief = source.relief_gbp[UK_FIRM_EMPTY_RELIEF_YEAR]
    on_list = 1000.0 * source.nndr3["hereditaments_30_sept"]
    gross = 1e6 * source.nndr3["gross_rates_payable"]
    outturn = 1e6 * source.nndr3["empty_premises_relief"]
    above = float(sum(counts[v] for v in UK_FIRM_EMPTY_MODELLED_HEREDITAMENTS))
    return {
        "status": UK_FIRM_EMPTY_STATUS,
        "gap": UK_FIRM_EMPTY_GAP,
        "snapshot": UK_FIRM_EMPTY_SNAPSHOT,
        "empty_relief_hereditaments": counts["hdit_mr_empty"],
        "empty_relief_hereditaments_above_threshold": above,
        "hereditaments_on_list": on_list,
        "empty_relief_share_of_hereditaments": counts["hdit_mr_empty"] / on_list,
        "above_threshold_share_of_hereditaments": above / on_list,
        "aggregate_rateable_value_gbp": 1e6
        * source.nndr3["aggregate_rateable_value_30_sept"],
        "relief_year": UK_FIRM_EMPTY_RELIEF_YEAR,
        "empty_relief_forecast_gbp": relief["value_empty"],
        "empty_relief_above_threshold_forecast_gbp": float(
            sum(relief[v] for v in UK_FIRM_EMPTY_MODELLED_RELIEF)
        ),
        "empty_relief_outturn_gbp": outturn,
        "gross_rates_payable_gbp": gross,
        "empty_relief_share_of_gross_rates": outturn / gross,
        "sha256": dict(source.sha256),
    }


__all__ = [
    "UK_FIRM_EMPTY_FILES",
    "UK_FIRM_EMPTY_GAP",
    "UK_FIRM_EMPTY_HEREDITAMENTS_FILE",
    "UK_FIRM_EMPTY_MODELLED_HEREDITAMENTS",
    "UK_FIRM_EMPTY_MODELLED_RELIEF",
    "UK_FIRM_EMPTY_NNDR3_FILE",
    "UK_FIRM_EMPTY_RELIEF_FILE",
    "UK_FIRM_EMPTY_RELIEF_YEAR",
    "UK_FIRM_EMPTY_SNAPSHOT",
    "UK_FIRM_EMPTY_STATUS",
    "UKFirmEmptyPropertySource",
    "read_uk_firm_empty_property_source",
    "uk_firm_empty_property_published_figures",
]
