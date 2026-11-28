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
The receipt holds the inputs' hashes keyed by role and file name, the rule
period with a note on how it treats mid-year changes, the years the source
data describe (``data_years``, which can lag the rule period: the frame's
sources for a checked vintage, then one entry per input group read, from
``UK_FIRM_INPUT_DATA_YEARS``), the
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
from types import MappingProxyType
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
from microcosm.build.uk_runtime.firm_employment_allowance_takeup import (
    UK_FIRM_EA_TAKEUP_METHOD,
    UK_FIRM_EA_TAKEUP_STATUS,
    assign_uk_firm_employment_allowance_takeup,
    read_uk_firm_employment_allowance_takeup,
    uk_firm_employment_allowance_takeup_held_out,
)
from microcosm.build.uk_runtime.firm_generation import (
    UKFirmGenerationConfig,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_headcount import (
    UK_FIRM_HEADCOUNT_METHOD,
    UK_FIRM_HEADCOUNT_STATUS,
    UK_FIRM_LARGE_HEADCOUNT_METHOD,
    UK_FIRM_MID_HEADCOUNT_METHOD,
    read_uk_firm_bpe_source,
    redraw_uk_firm_small_headcount,
    remap_uk_firm_large_headcount,
    remap_uk_firm_mid_headcount,
    uk_firm_bpe_size_band_reconciliation,
    uk_firm_employees,
    uk_firm_small_headcount_diagnostics,
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
from microcosm.build.uk_runtime.firm_rd_tax_credits import (
    UK_FIRM_RD_METHOD,
    UK_FIRM_RD_STATUS,
    draw_uk_firm_rd_tax_credits,
    read_uk_firm_rd_source,
)
from microcosm.build.uk_runtime.firm_rules import (
    UK_FIRM_RULE_PERIOD_NOTE,
    UKFirmRuleOutputs,
    UKFirmRules,
    data_years_for_vintage,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_EA_RULE_CLAIMS_COLUMN,
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
        # ONS ASHE 2025 Table 4.7a: the tax year ending 5 April 2025.
        "ashe_paybill": "2024-25",
        "ashe_pay_distribution": "2024-25",
        # ONS public sector employment, headcount at March 2025.
        "ons_public_sector_employment": "Mar 2025",
        # DBT Business Population Estimates 2025: start of 2025.
        "dbt_bpe": "start of 2025",
        # HMRC Employment Allowance take-up statistics, tax year 2024-25.
        "hmrc_employment_allowance_takeup": "2024-25",
        # HMRC Corporation Tax statistics 2026, Tables 9A/9B: FY 2024-25.
        "hmrc_ct_statistics": "2024-25",
        # HMRC R&D Tax Credits statistics, September 2026: 2024-25 provisional.
        "hmrc_rd_statistics": "2024-25",
        # VOA non-domestic rating stock at 31 March 2026, 2023 rating lists.
        "voa_rating_list": "31 Mar 2026 (2023 list)",
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
    paybill_dir: str | Path | None = None,
    employment_allowance: bool = False,
    public_sector_dir: str | Path | None = None,
    profit_dir: str | Path | None = None,
    rateable_value_dir: str | Path | None = None,
    pay_distribution_dir: str | Path | None = None,
    bpe_dir: str | Path | None = None,
    employment_allowance_takeup_dir: str | Path | None = None,
    bpe_mid_headcount: bool = False,
    out_of_frame_no_employees: bool = False,
    bpe_large_headcount: bool = False,
    rd_tax_credits_dir: str | Path | None = None,
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
    the secondary threshold (``UK_FIRM_SECONDARY_THRESHOLD_GBP``).
    ``bpe_dir`` (``tests/fixtures/uk/firm_bpe_2025``), which needs a
    ``paybill_dir``, redraws headcount inside the ONS 0-4 employment band by
    legal status from DBT Business Population Estimates 2025 Table 3
    (``firm_headcount.py``) before the public sector rescale, and switches on
    the labour link group: ``firm_employees`` becomes employment less the
    assumed working proprietors, and the paybill and pay split are set on
    employees. ``employment_allowance_takeup_dir``
    (``tests/fixtures/uk/firm_employment_allowance_takeup_2024_25``), which
    needs ``bpe_dir`` and ``employment_allowance``, draws each rule claimant's
    claim at HMRC's 2024-25 take-up by employee size band
    (``firm_employment_allowance_takeup.py``).
    ``bpe_mid_headcount`` (needs ``bpe_dir``) remaps headcount inside the
    ONS bands 5-9 to 100-249 to BPE 2025 Table 1 shapes by weighted rank;
    ``out_of_frame_no_employees`` (needs ``bpe_dir``) gives the rows outside
    the ONS frame no headcount, an assumption with no published pin.
    ``bpe_large_headcount`` (needs ``bpe_dir``) remaps 250+ headcounts to
    BPE 2025 Table 1's large rows. ``rd_tax_credits_dir``
    (``tests/fixtures/uk/firm_rd_tax_credits_2024_25``) adds the assumed
    ``firm_qualifying_rd_expenditure`` and ``firm_is_sme`` from HMRC R&D tax
    credit statistics (``firm_rd_tax_credits.py``).
    Returns ``table`` (the
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
        ("bpe_dir", bpe_dir),
        ("employment_allowance_takeup_dir", employment_allowance_takeup_dir),
        ("rd_tax_credits_dir", rd_tax_credits_dir),
    ):
        if directory is not None and config.data_vintage != "2024-25":
            raise ValueError(
                f"The UK firm {name} source is for tax year 2024-25; "
                f"config.data_vintage is {config.data_vintage}."
            )
    if pay_distribution_dir is not None and paybill_dir is None:
        raise ValueError("pay_distribution_dir needs paybill_dir.")
    if bpe_dir is not None and paybill_dir is None:
        raise ValueError("bpe_dir needs paybill_dir.")
    if (
        bpe_mid_headcount or out_of_frame_no_employees or bpe_large_headcount
    ) and bpe_dir is None:
        raise ValueError(
            "bpe_mid_headcount, out_of_frame_no_employees and "
            "bpe_large_headcount need bpe_dir."
        )
    if employment_allowance_takeup_dir is not None and (
        bpe_dir is None or not employment_allowance
    ):
        raise ValueError(
            "employment_allowance_takeup_dir needs bpe_dir and employment_allowance."
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
    bpe_source = None
    headcount = None
    if bpe_dir is not None:
        bpe_source = read_uk_firm_bpe_source(bpe_dir)
        redrawn = redraw_uk_firm_small_headcount(
            firms, bpe_source, seed=result.config.seed
        )
        headcount = uk_firm_small_headcount_diagnostics(firms, redrawn, bpe_source)
        firms = redrawn
    out_of_frame_employment = None
    if out_of_frame_no_employees:
        outside = ~firms["in_frame"].to_numpy(bool)
        out_of_frame_employment = float(
            (firms["firm_weight"].to_numpy(np.float64) * firms["employment"])[
                outside
            ].sum()
        )
        employment = firms["employment"].to_numpy().copy()
        employment[outside] = 0
        firms = firms.assign(employment=employment)
    mid_headcount = None
    if bpe_mid_headcount:
        remapped = remap_uk_firm_mid_headcount(
            firms, bpe_source, seed=result.config.seed
        )
        mid_headcount = _mid_headcount_record(firms, remapped)
        firms = remapped
    large_headcount = None
    if bpe_large_headcount:
        remapped = remap_uk_firm_large_headcount(
            firms, bpe_source, seed=result.config.seed
        )
        large_headcount = _mid_headcount_record(firms, remapped)
        large_headcount["method"] = UK_FIRM_LARGE_HEADCOUNT_METHOD
        firms = remapped
    public_sector = None
    if public_sector_dir is not None:
        public_sector_source = read_uk_firm_public_sector_source(public_sector_dir)
        public_sector = uk_firm_public_sector_diagnostics(firms, public_sector_source)
        firms = scale_uk_firm_public_sector_employment(firms, public_sector_source)
    # The rows the paybill and pay split read: with bpe_dir their headcount
    # is employees (employment less the assumed working proprietors).
    pay_rows = firms
    if bpe_source is not None:
        firms = firms.assign(
            employees=uk_firm_employees(
                firms["ons_legal_status"].to_numpy(), firms["employment"].to_numpy()
            )
        )
        pay_rows = firms.assign(employment=firms["employees"])
    paybill_source = None
    if paybill_dir is not None:
        paybill_source = read_uk_firm_paybill_source(paybill_dir)
        firms = firms.assign(
            firm_paybill=assign_uk_firm_paybill(pay_rows, paybill_source)
        )
    pay_distribution_source = None
    if pay_distribution_dir is not None:
        pay_distribution_source = read_uk_firm_pay_distribution_source(
            pay_distribution_dir
        )
        firms = firms.assign(
            **assign_uk_firm_pay_below_secondary_threshold(
                pay_rows.assign(firm_paybill=firms["firm_paybill"]),
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
    rd_source = None
    rd_draw = None
    if rd_tax_credits_dir is not None:
        rd_source = read_uk_firm_rd_source(rd_tax_credits_dir)
        rd_draw = draw_uk_firm_rd_tax_credits(firms, rd_source, seed=result.config.seed)
        firms = firms.assign(
            firm_qualifying_rd_expenditure=rd_draw.qualifying_rd_expenditure,
            firm_is_sme=rd_draw.is_sme,
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
    if bpe_source is not None:
        firms = firms.assign(
            firm_sic_division=pd.to_numeric(firms["sic_code"])
            .to_numpy()
            .astype(np.int64)
        )
    takeup_source = None
    takeup = None
    if employment_allowance_takeup_dir is not None:
        takeup_source = read_uk_firm_employment_allowance_takeup(
            employment_allowance_takeup_dir
        )
        rule_table = to_policyengine_uk_firm_table(
            firms, threshold_k=float(result.config.vat_threshold)
        )
        drawn, takeup_diagnostics = assign_uk_firm_employment_allowance_takeup(
            rule_table, takeup_source, seed=result.config.seed
        )
        claims = pd.Series(
            drawn["firm_claims_employment_allowance"].to_numpy(bool),
            index=drawn["firm_id"].to_numpy(),
        )
        takeup = {
            **takeup_diagnostics,
            "held_out": uk_firm_employment_allowance_takeup_held_out(
                drawn,
                drawn["firm_claims_employment_allowance"].to_numpy(bool),
                takeup_source,
            ),
        }
        drawn_claims = claims.reindex(firms["firm_id"].to_numpy())
        if drawn_claims.isna().any():
            raise RuntimeError("The take-up draw missed some UK firm ids.")
        firms = firms.assign(
            **{
                UK_FIRM_EA_RULE_CLAIMS_COLUMN: firms[
                    "firm_claims_employment_allowance"
                ].to_numpy(bool),
                "firm_claims_employment_allowance": drawn_claims.to_numpy(bool),
            }
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
                    "ashe_paybill": paybill_dir is not None,
                    "ashe_pay_distribution": pay_distribution_dir is not None,
                    "ons_public_sector_employment": public_sector_dir is not None,
                    "dbt_bpe": bpe_dir is not None,
                    "hmrc_employment_allowance_takeup": (
                        employment_allowance_takeup_dir is not None
                    ),
                    "hmrc_ct_statistics": profit_dir is not None,
                    "hmrc_rd_statistics": rd_tax_credits_dir is not None,
                    "voa_rating_list": rateable_value_dir is not None,
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
    if bpe_source is not None:
        receipt["inputs"]["bpe"] = dict(bpe_source.sha256)
        receipt["uk_firm_headcount"] = {
            "status": UK_FIRM_HEADCOUNT_STATUS,
            "method": UK_FIRM_HEADCOUNT_METHOD,
            "small_band_redraw": _records(headcount),
            # Report-only: paybill over BPE 2025 whole-economy employees times
            # ASHE's all-employee mean pay. ASHE annual jobs (the usual
            # check's base) cover only jobs held a full year.
            "paybill_vs_bpe_employees_x_ashe_mean": float(
                (weight * table["firm_paybill"]).sum()
                / (
                    _bpe_whole_economy_employees(bpe_source)
                    * paybill_source.all_employees_mean
                )
            ),
            "weighted_employees": float((weight * table["firm_employees"]).sum()),
            "weighted_ons_employment": float((weight * table["ons_employment"]).sum()),
            "bpe_reconciliation": _records(
                uk_firm_bpe_size_band_reconciliation(
                    table.assign(employment=table["ons_employment"]), bpe_source
                )
            ),
        }
    if bpe_source is not None:
        # Opt-in switches appear only when set.
        for key, value in (
            ("mid_band_remap", mid_headcount),
            ("large_band_remap", large_headcount),
            ("out_of_frame_employment_removed", out_of_frame_employment),
        ):
            if value is not None:
                receipt["uk_firm_headcount"][key] = value
    if takeup_source is not None:
        receipt["inputs"]["employment_allowance_takeup"] = dict(takeup_source.sha256)
        receipt["uk_firm_employment_allowance_takeup"] = {
            "status": UK_FIRM_EA_TAKEUP_STATUS,
            "method": UK_FIRM_EA_TAKEUP_METHOD,
            **takeup,
        }
    if rd_source is not None:
        receipt["inputs"]["rd_tax_credits"] = dict(rd_source.sha256)
        receipt["uk_firm_rd_tax_credits"] = {
            "status": UK_FIRM_RD_STATUS,
            "method": UK_FIRM_RD_METHOD,
            "weighted_expenditure_gbp": float(
                (weight * table["firm_qualifying_rd_expenditure"]).sum()
            ),
            **dict(rd_draw.diagnostics),
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


def _bpe_whole_economy_employees(source: Any) -> float:
    rows = source.size_bands
    row = rows[
        (rows["table"] == "Table 2")
        & (rows["sector"] == "whole_economy")
        & (rows["size_band"] == "All")
    ]
    return float(row["employees_thousand"].iloc[0]) * 1_000.0


def _mid_headcount_record(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    weight = before["firm_weight"].to_numpy(np.float64)
    return {
        "method": UK_FIRM_MID_HEADCOUNT_METHOD,
        "weighted_employment_before": float(
            (weight * before["employment"].to_numpy(np.float64)).sum()
        ),
        "weighted_employment_after": float(
            (weight * after["employment"].to_numpy(np.float64)).sum()
        ),
        "weighted_rows_changed": float(
            weight[
                before["employment"].to_numpy() != after["employment"].to_numpy()
            ].sum()
        ),
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
