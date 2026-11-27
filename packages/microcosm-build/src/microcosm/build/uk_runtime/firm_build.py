"""A local UK firm build: generated firms to a policyengine-uk table on disk.

``build_uk_firm_dataset`` runs the experimental firm pipeline end to end:

1. ``read_uk_firm_source_data`` reads the processed ONS/HMRC tables;
2. ``generate_uk_firm_population`` draws and calibrates the firms, with VAT
   liability from the ``rules`` it is given;
3. ``assign_uk_firm_attributes`` draws ONS region and legal status;
4. ``to_policyengine_uk_firm_table`` exports policyengine-uk firm columns;
5. ``validate_uk_firm_table`` checks the table.

It writes ``uk_firms.parquet`` and ``uk_firms.receipt.json`` to the output
directory and nothing else: there is no publication, staging or upload path.
The receipt holds the inputs' hashes keyed by role and file name, the rule
period with a note on how it treats mid-year changes, the years the source
data describe (``data_years``, which can lag the rule period: the frame's
sources for a checked vintage, then one entry per input group read, from
``UK_FIRM_INPUT_DATA_YEARS``), the
configuration's JSON-safe fields, the output's hash and shape, the
calibration's accuracy figures, the attribute fallback counts and the
provenance the rules reported.
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
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_attributes import (
    assign_uk_firm_attributes,
    read_uk_firm_attribute_source_data,
    uk_firm_attribute_fallback_counts,
)
from microcosm.build.uk_runtime.firm_generation import (
    UKFirmGenerationConfig,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_rules import (
    UK_FIRM_RULE_PERIOD_NOTE,
    UKFirmRuleOutputs,
    UKFirmRules,
    data_years_for_vintage,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    to_policyengine_uk_firm_table,
    validate_uk_firm_table,
)

#: Version of the ``uk_firms.receipt.json`` layout.
UK_FIRM_BUILD_RECEIPT_SCHEMA_VERSION = 1
UK_FIRM_TABLE_FILE = "uk_firms.parquet"
UK_FIRM_RECEIPT_FILE = "uk_firms.receipt.json"

#: The years each input group's source describes, recorded in the receipt's
#: ``data_years`` beside the frame's and apart from ``rule_period``. A key
#: appears only when its group's source is read. Each value is the source's
#: own reference period, from its fixture provenance; the build reads the
#: 2024-25 fixtures only.
UK_FIRM_INPUT_DATA_YEARS: Mapping[str, str] = MappingProxyType(
    {
        # ONS UK Business 2025 Tables 8, 12 and 13: the 2025 frame, whose
        # turnover bands are mostly from VAT returns for 2023-24 (note 4).
        "ons_attributes": "2025 frame (turnover mostly 2023-24)",
    }
)


def _input_data_years(groups: Mapping[str, bool]) -> dict[str, str]:
    """``UK_FIRM_INPUT_DATA_YEARS`` entries of the groups switched on."""

    unknown = sorted(set(groups) - set(UK_FIRM_INPUT_DATA_YEARS))
    if unknown:
        raise ValueError(f"No UK firm data year for input groups {unknown}.")
    return {key: UK_FIRM_INPUT_DATA_YEARS[key] for key, on in groups.items() if on}


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
) -> dict[str, Any]:
    """Build the UK firm table and its receipt into ``output_dir``.

    ``config`` is run with ``firm_rules`` set to ``rules`` (wrapped so the
    receipt can record the provenance it reports); a ``config`` that already
    carries other firm rules raises ``ValueError``. Returns ``table`` (the
    validated policyengine-uk firm table), ``receipt`` (the dict written as
    JSON), ``result`` (the ``UKFirmGenerationResult``), ``table_path`` and
    ``receipt_path``.
    """

    if config.firm_rules is not None and config.firm_rules is not rules:
        raise ValueError(
            "config.firm_rules differs from rules; pass the rules once, as rules."
        )
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
        "rule_period_note": UK_FIRM_RULE_PERIOD_NOTE,
        "data_years": {
            **(data_years_for_vintage(result.config.data_vintage) or {}),
            **_input_data_years(
                {
                    "ons_attributes": True,
                }
            ),
        },
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
