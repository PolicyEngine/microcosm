"""A local UK firm build: generated firms to a policyengine-uk table on disk.

``build_uk_firm_dataset`` runs the experimental firm pipeline end to end:

1. ``read_uk_firm_source_data`` reads the processed ONS/HMRC tables;
2. ``generate_uk_firm_population`` draws and calibrates the firms, with VAT
   liability from the ``rules`` it is given;
3. ``assign_uk_firm_attributes`` draws ONS region and legal status;
4. with a ``paybill_dir``, ``assign_uk_firm_paybill`` sets ``firm_paybill``
   to employees times the ONS ASHE section mean annual gross pay;
   with ``employment_allowance``, ``assign_uk_firm_employment_allowance``
   sets the assumed Employment Allowance flags;
5. ``to_policyengine_uk_firm_table`` exports policyengine-uk firm columns;
6. ``validate_uk_firm_table`` checks the table.

It writes ``uk_firms.parquet`` and ``uk_firms.receipt.json`` to the output
directory and nothing else: there is no publication, staging or upload path.
The receipt holds the inputs' hashes keyed by role and file name, the
configuration's JSON-safe fields, the output's hash and shape, the
calibration's accuracy figures, the attribute fallback counts and the
provenance the rules reported; with a ``paybill_dir``, also the paybill CSV's
hash (``inputs.paybill``), the paybill fallback counts, the weighted
paybill, its in-frame split (``uk_firm_paybill_in_frame``) and its status
(``uk_firm_paybill``). Without one, the receipt has no paybill key at all.
With ``employment_allowance``, it also holds the flags' status, method and
weighted counts of firms where each flag is true (``uk_firm_employment_allowance``).
It holds no timestamp, hostname or absolute path, so two builds from the same inputs write the same receipt.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_attributes import (
    assign_uk_firm_attributes,
    read_uk_firm_attribute_source_data,
    uk_firm_attribute_fallback_counts,
)
from microcosm.build.uk_runtime.firm_employment_allowance import (
    UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
    UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD,
    UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS,
    assign_uk_firm_employment_allowance,
)
from microcosm.build.uk_runtime.firm_generation import (
    UKFirmGenerationConfig,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_pay_distribution import (
    UK_FIRM_PAY_DISTRIBUTION_METHOD,
    UK_FIRM_PAY_DISTRIBUTION_STATUS,
    assign_uk_firm_pay_below_secondary_threshold,
    read_uk_firm_pay_distribution_source,
)
from microcosm.build.uk_runtime.firm_paybill import (
    UK_FIRM_PAYBILL_METHOD,
    UK_FIRM_PAYBILL_STATUS,
    UK_FIRM_PAYBILL_TAX_YEAR,
    assign_uk_firm_paybill,
    read_uk_firm_paybill_source,
    uk_firm_paybill_fallback_counts,
)
from microcosm.build.uk_runtime.firm_profit import (
    UK_FIRM_PROFIT_METHOD,
    UK_FIRM_PROFIT_STATUS,
    assign_uk_firm_taxable_profit,
    read_uk_firm_profit_source,
    uk_firm_profit_diagnostics,
)
from microcosm.build.uk_runtime.firm_public_sector import (
    UK_FIRM_PUBLIC_SECTOR_STATUS,
    read_uk_firm_public_sector_source,
    scale_uk_firm_public_sector_employment,
    uk_firm_public_sector_diagnostics,
)
from microcosm.build.uk_runtime.firm_rateable_value import (
    UK_FIRM_RATEABLE_VALUE_METHOD,
    UK_FIRM_RATEABLE_VALUE_STATUS,
    assign_uk_firm_rateable_value,
    read_uk_firm_rateable_value_source,
)
from microcosm.build.uk_runtime.firm_rules import (
    UKFirmRuleOutputs,
    UKFirmRules,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_ENGLISH_REGION_CODES,
    to_policyengine_uk_firm_table,
    validate_uk_firm_table,
)

#: policyengine-uk's 2024-25 annual secondary threshold: £175 a week
#: (``gov.hmrc.national_insurance.class_1.employer_scale``, bracket 1) times
#: 52.
UK_FIRM_SECONDARY_THRESHOLD_GBP = 175.0 * 52

#: Version of the ``uk_firms.receipt.json`` layout.
UK_FIRM_BUILD_RECEIPT_SCHEMA_VERSION = 1
UK_FIRM_TABLE_FILE = "uk_firms.parquet"
UK_FIRM_RECEIPT_FILE = "uk_firms.receipt.json"

#: Configuration fields holding rule objects, which the receipt leaves out.
_RULE_CONFIG_FIELDS = frozenset({"firm_rules", "vat_rule_evaluator"})


@dataclass
class _RecordingRules:
    """``UKFirmRules`` that passes every call through and keeps its provenance."""

    rules: UKFirmRules
    provenance: list[dict[str, str]] = field(default_factory=list)

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        outputs = self.rules.evaluate(firms, period=period)
        self.provenance.append(
            {str(key): str(value) for key, value in outputs.provenance.items()}
        )
        return outputs


def build_uk_firm_dataset(
    *,
    processed_dir: str | Path,
    attributes_dir: str | Path,
    output_dir: str | Path,
    rules: UKFirmRules,
    config: UKFirmGenerationConfig,
    paybill_dir: str | Path | None = None,
    employment_allowance: bool = False,
    public_sector_dir: str | Path | None = None,
    profit_dir: str | Path | None = None,
    rateable_value_dir: str | Path | None = None,
    pay_distribution_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the UK firm table and its receipt into ``output_dir``.

    ``config`` is run with ``firm_rules`` set to ``rules`` (wrapped so the
    receipt can record the provenance it reports); a ``config`` that already
    carries other firm rules raises ``ValueError``. ``paybill_dir``, when
    given, holds the ONS ASHE section pay CSV
    (``tests/fixtures/uk/firm_paybill_2024``) and adds ``firm_paybill`` to
    the table; without it the table and receipt are as before. The paybill
    table is for tax year ``UK_FIRM_PAYBILL_TAX_YEAR``; a ``paybill_dir``
    with any other ``config.data_vintage`` raises ``ValueError``.
    ``employment_allowance`` adds the assumed Employment Allowance flags
    (``firm_employment_allowance.py``); they read no random numbers.
    ``public_sector_dir`` (``tests/fixtures/uk/firm_public_sector_2025``)
    scales the headcount of central government and local authority rows to
    ONS public sector employment before the paybill is set;
    ``profit_dir`` (``tests/fixtures/uk/firm_ct_2024``) adds the assumed
    ``firm_taxable_trading_profit``; ``rateable_value_dir``
    (``tests/fixtures/uk/firm_rateable_value_2024``) the assumed
    ``firm_rateable_value``. Each is opt-in, reads its own fixed-seed
    generator if any, and leaves the table and receipt unchanged when
    omitted. ``pay_distribution_dir``
    (``tests/fixtures/uk/firm_pay_distribution_2024``), which needs a
    ``paybill_dir``, adds the assumed count and pay of employees at or below
    the secondary threshold (``UK_FIRM_SECONDARY_THRESHOLD_GBP``). Returns ``table`` (the
    validated policyengine-uk firm table), ``receipt`` (the dict written as
    JSON), ``result`` (the ``UKFirmGenerationResult``), ``table_path`` and
    ``receipt_path``.
    """

    if config.firm_rules is not None and config.firm_rules is not rules:
        raise ValueError(
            "config.firm_rules differs from rules; pass the rules once, as rules."
        )
    if paybill_dir is not None and config.data_vintage != UK_FIRM_PAYBILL_TAX_YEAR:
        raise ValueError(
            f"The UK firm paybill table is for tax year {UK_FIRM_PAYBILL_TAX_YEAR}; "
            f"config.data_vintage is {config.data_vintage}."
        )
    for name, directory in (
        ("public_sector_dir", public_sector_dir),
        ("profit_dir", profit_dir),
        ("rateable_value_dir", rateable_value_dir),
        ("pay_distribution_dir", pay_distribution_dir),
    ):
        if directory is not None and config.data_vintage != "2024-25":
            raise ValueError(
                f"The UK firm {name} source is for tax year 2024-25; "
                f"config.data_vintage is {config.data_vintage}."
            )
    if pay_distribution_dir is not None and paybill_dir is None:
        raise ValueError("pay_distribution_dir needs paybill_dir.")
    processed = Path(processed_dir)
    attributes = Path(attributes_dir)
    output = Path(output_dir)

    recorder = _RecordingRules(rules)
    run_config = dataclasses.replace(config, firm_rules=recorder)
    source = read_uk_firm_source_data(processed, input_files=run_config.input_files)
    result = generate_uk_firm_population(source, run_config)
    if not recorder.provenance:
        raise RuntimeError("The UK firm rules were never evaluated.")

    attribute_source = read_uk_firm_attribute_source_data(attributes)
    firms = assign_uk_firm_attributes(
        result.firms, attribute_source, seed=result.config.seed
    )
    public_sector = None
    if public_sector_dir is not None:
        public_sector_source = read_uk_firm_public_sector_source(public_sector_dir)
        public_sector = uk_firm_public_sector_diagnostics(firms, public_sector_source)
        firms = scale_uk_firm_public_sector_employment(firms, public_sector_source)
    paybill_source = None
    if paybill_dir is not None:
        paybill_source = read_uk_firm_paybill_source(paybill_dir)
        firms = firms.assign(firm_paybill=assign_uk_firm_paybill(firms, paybill_source))
    pay_distribution_source = None
    if pay_distribution_dir is not None:
        pay_distribution_source = read_uk_firm_pay_distribution_source(
            pay_distribution_dir
        )
        firms = firms.assign(
            **assign_uk_firm_pay_below_secondary_threshold(
                firms,
                pay_distribution_source,
                threshold_gbp=UK_FIRM_SECONDARY_THRESHOLD_GBP,
            )
        )
    if employment_allowance:
        firms = assign_uk_firm_employment_allowance(firms)
    profit_source = None
    if profit_dir is not None:
        profit_source = read_uk_firm_profit_source(profit_dir)
        firms = firms.assign(
            firm_taxable_trading_profit=assign_uk_firm_taxable_profit(
                firms, profit_source
            )
        )
    rateable_value_source = None
    if rateable_value_dir is not None:
        rateable_value_source = read_uk_firm_rateable_value_source(rateable_value_dir)
        in_england = firms.assign(
            firm_in_england=firms["ons_region_code"].isin(
                list(UK_FIRM_ENGLISH_REGION_CODES)
            )
        )
        firms = firms.assign(
            firm_rateable_value=assign_uk_firm_rateable_value(
                in_england, rateable_value_source
            )
        )
    table = to_policyengine_uk_firm_table(
        firms, threshold_k=float(result.config.vat_threshold)
    )
    validate_uk_firm_table(table)

    output.mkdir(parents=True, exist_ok=True)
    table_path = output / UK_FIRM_TABLE_FILE
    table.to_parquet(table_path, index=False, engine="pyarrow")
    written = pd.read_parquet(table_path, engine="pyarrow")
    validate_uk_firm_table(written)

    fallback = uk_firm_attribute_fallback_counts(firms, attribute_source)
    validation = result.validation
    receipt: dict[str, Any] = {
        "schema_version": UK_FIRM_BUILD_RECEIPT_SCHEMA_VERSION,
        "data_vintage": result.config.data_vintage,
        "seed": int(result.config.seed),
        "rule_period": rule_period_for_vintage(result.config.data_vintage),
        "config": _config_record(result.config),
        "inputs": {
            "processed": _csv_hashes(processed),
            "attributes": _csv_hashes(attributes),
        },
        "output": {
            "file": UK_FIRM_TABLE_FILE,
            "sha256": _sha256(table_path),
            "rows": len(table),
            "columns": [str(column) for column in table.columns],
            "weighted_firms": float(table["firm_weight"].sum()),
        },
        "validation": {
            **{
                key: float(value)
                for key, value in dataclasses.asdict(validation).items()
            },
            "overall": float(validation.overall),
        },
        "calibration": {
            "initial_loss": float(result.calibration.initial_loss),
            "final_loss": float(result.calibration.final_loss),
        },
        "target_diagnostics": _records(result.target_diagnostics),
        "uk_firm_attribute_fallback_counts": _records(fallback),
        "rules_provenance": recorder.provenance,
    }
    if paybill_source is not None:
        receipt["inputs"]["paybill"] = {paybill_source.file: paybill_source.sha256}
        receipt["output"]["weighted_paybill_gbp"] = float(
            (table["firm_weight"] * table["firm_paybill"]).sum()
        )
        receipt["uk_firm_paybill_fallback_counts"] = _records(
            uk_firm_paybill_fallback_counts(table, paybill_source)
        )
        in_frame = table["in_frame"].to_numpy(bool)
        weight = table["firm_weight"].to_numpy(np.float64)
        receipt["uk_firm_paybill_in_frame"] = [
            {
                "in_frame": flag,
                "weighted_employees": float(
                    (weight * table["firm_employees"].to_numpy(np.float64))[
                        in_frame == flag
                    ].sum()
                ),
                "weighted_paybill_gbp": float(
                    (weight * table["firm_paybill"].to_numpy(np.float64))[
                        in_frame == flag
                    ].sum()
                ),
            }
            for flag in (True, False)
        ]
        receipt["uk_firm_paybill"] = {
            "status": UK_FIRM_PAYBILL_STATUS,
            "method": UK_FIRM_PAYBILL_METHOD,
            "tax_year": UK_FIRM_PAYBILL_TAX_YEAR,
        }
    if employment_allowance:
        weight = table["firm_weight"].to_numpy(np.float64)
        receipt["uk_firm_employment_allowance"] = {
            "status": UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS,
            "method": UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD,
            "weighted_firms_where_true": {
                column: float(weight[table[column].to_numpy(bool)].sum())
                for column in UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS
            },
        }
    weight = table["firm_weight"].to_numpy(np.float64)
    if public_sector is not None:
        receipt["inputs"]["public_sector"] = {
            public_sector_source.file: public_sector_source.sha256
        }
        receipt["uk_firm_public_sector"] = {
            "status": UK_FIRM_PUBLIC_SECTOR_STATUS,
            **public_sector,
        }
    if pay_distribution_source is not None:
        receipt["inputs"]["pay_distribution"] = {
            pay_distribution_source.file: pay_distribution_source.sha256
        }
        receipt["uk_firm_pay_distribution"] = {
            "status": UK_FIRM_PAY_DISTRIBUTION_STATUS,
            "method": UK_FIRM_PAY_DISTRIBUTION_METHOD,
            "threshold_gbp": UK_FIRM_SECONDARY_THRESHOLD_GBP,
            "weighted_employees_below": float(
                (weight * table["firm_employees_below_secondary_threshold"]).sum()
            ),
            "weighted_paybill_below_gbp": float(
                (weight * table["firm_paybill_below_secondary_threshold"]).sum()
            ),
        }
    if profit_source is not None:
        receipt["inputs"]["profit"] = {profit_source.file: profit_source.sha256}
        receipt["uk_firm_profit"] = {
            "status": UK_FIRM_PROFIT_STATUS,
            "method": UK_FIRM_PROFIT_METHOD,
            "weighted_profit_gbp": float(
                (weight * table["firm_taxable_trading_profit"]).sum()
            ),
            "bands": _records(uk_firm_profit_diagnostics(firms, profit_source)),
        }
    if rateable_value_source is not None:
        receipt["inputs"]["rateable_value"] = {
            rateable_value_source.file: rateable_value_source.sha256
        }
        receipt["uk_firm_rateable_value"] = {
            "status": UK_FIRM_RATEABLE_VALUE_STATUS,
            "method": UK_FIRM_RATEABLE_VALUE_METHOD,
            "weighted_rateable_value_gbp": float(
                (weight * table["firm_rateable_value"]).sum()
            ),
        }
    receipt = _json_safe(receipt)
    receipt_path = output / UK_FIRM_RECEIPT_FILE
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "table": table,
        "receipt": receipt,
        "result": result,
        "table_path": table_path,
        "receipt_path": receipt_path,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_hashes(directory: Path) -> dict[str, str]:
    """Every CSV directly in ``directory``, by file name, to its sha256."""

    return {path.name: _sha256(path) for path in sorted(directory.glob("*.csv"))}


def _config_record(config: UKFirmGenerationConfig) -> dict[str, Any]:
    record = {}
    for item in dataclasses.fields(config):
        if item.name in _RULE_CONFIG_FIELDS:
            continue
        value = getattr(config, item.name)
        if isinstance(value, Mapping):
            value = {str(key): str(entry) for key, entry in value.items()}
        record[item.name] = value
    record["thinned"] = config.thinned
    return record


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    """``value`` with numpy scalars made native and non-finite floats None."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"UK firm receipt value {value!r} is not JSON-safe.")
