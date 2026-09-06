"""Actual pinned-engine probes of the housing charge, assistance and tenant rent.

Every household here is explicitly invented and tiny. The probes run the real
installed policyengine-us through Microcosm's production export and simulation
seams -- the Frame adapter's ``write_dataset`` gate and
``reform_validation.default_simulate_factory`` -- so the recorded numbers are
engine behaviour, not a restatement of the formulas.

They establish four consumer-relevant facts and one blocking defect:

* the full charge input is independent of assistance policy,
* assistance policy moves modeled HAP and tenant rent,
* a utility-reimbursing HAP cannot drive tenant rent negative,
* two SPM units in one household each draw the household's whole payment
  standard, so a household receipt answer must not be replicated to both, and
* the bundled FMR table has no 2024 fiscal year and falls forward to FY2025.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

PROBE_PERIOD = 2024

#: Reported per SPM unit unless the entity is named otherwise.
SPM_MEASURES = (
    "housing_assistance",
    "hud_hap",
    "hud_ttp",
    "hud_gross_rent",
    "hud_max_subsidy",
    "is_eligible_for_housing_assistance",
    "spm_unit_capped_housing_subsidy",
)
PERSON_MEASURES = ("pre_subsidy_rent", "rent", "housing_cost")
HOUSEHOLD_MEASURES = (
    "hud_utility_allowance",
    "pha_payment_standard",
    "hud_fair_market_rent",
)

#: Formula-owned names an export must never persist as dataset inputs.
FORMULA_OWNED_HOUSING_OUTPUTS = (
    "rent",
    "housing_assistance",
    "hud_hap",
    "spm_unit_capped_housing_subsidy",
)

HUD_ABOLITION_REFORM = {"gov.hud.abolition": {"2024-01-01.2100-12-31": True}}

#: household_key -> (persons, spm units, what the household demonstrates)
PROBE_HOUSEHOLDS = {
    "sole_spm_unit_assisted": "one renter SPM unit with an unmet full charge",
    "two_spm_units_one_household": (
        "two SPM units sharing one housing unit; the first holds two people"
    ),
    "utility_reimbursing_hap": "tenant-paid utilities lift HAP above the charge",
    "owner_without_receipt_flag": "modelled ineligibility with the flag False",
    "owner_with_receipt_flag": "the same unit with the receipt flag True",
}


def invented_probe_frame() -> Frame:
    """Five invented households; identical apart from the property being probed.

    ``owner_without_receipt_flag`` and ``owner_with_receipt_flag`` differ only in
    ``receives_housing_assistance``, which is the one engine input this source
    slice could ever supply. Their gap is what coercing an unknown household to
    an observed ``False`` would cost.
    """
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6, 7],
            "person_household_id": [1, 2, 2, 2, 3, 4, 5],
            "person_tax_unit_id": [1, 2, 2, 3, 4, 5, 6],
            "person_spm_unit_id": [1, 2, 2, 3, 4, 5, 6],
            "person_family_id": [1, 2, 2, 3, 4, 5, 6],
            "person_marital_unit_id": [1, 2, 2, 3, 4, 5, 6],
            "age": [40.0, 41.0, 39.0, 42.0, 43.0, 44.0, 45.0],
            "employment_income_before_lsr": [
                12_000.0,
                7_000.0,
                5_000.0,
                9_000.0,
                8_000.0,
                12_000.0,
                12_000.0,
            ],
            "pre_subsidy_rent": [
                12_000.0,
                7_000.0,
                5_000.0,
                9_600.0,
                3_600.0,
                12_000.0,
                12_000.0,
            ],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "state_fips": [6, 6, 6, 6, 6],
            "county_fips": [6037, 6037, 6037, 6037, 6037],
            "bedrooms": [2, 2, 2, 2, 2],
            "is_sro": [False, False, False, False, False],
            "tenant_pays_utilities": [False, False, True, False, False],
        }
    )
    spm_unit = pd.DataFrame(
        {
            "spm_unit_id": [1, 2, 3, 4, 5, 6],
            "spm_unit_tenure_type": [
                "RENTER",
                "RENTER",
                "RENTER",
                "RENTER",
                "OWNER_WITH_MORTGAGE",
                "OWNER_WITH_MORTGAGE",
            ],
            "receives_housing_assistance": [
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )
    tables = {
        "person": person,
        "household": household,
        "spm_unit": spm_unit,
        "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2, 3, 4, 5, 6]}),
        "family": pd.DataFrame({"family_id": [1, 2, 3, 4, 5, 6]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2, 3, 4, 5, 6]}),
    }
    weights = {
        "household": Weights(values=np.ones(5, dtype=float), kind=WeightKind.DESIGN)
    }
    return Frame(tables, US_SCHEMA, weights)


def _values(simulation: Any, measure: str) -> list[float]:
    values = np.asarray(simulation.calculate(measure, PROBE_PERIOD))
    if values.dtype == np.dtype("bool"):
        return [bool(value) for value in values]
    return [round(float(value), 6) for value in values]


def _measure_block(simulation: Any) -> dict[str, list[float]]:
    return {
        measure: _values(simulation, measure)
        for measure in (*PERSON_MEASURES, *SPM_MEASURES, *HOUSEHOLD_MEASURES)
    }


def export_probe_dataset(directory: str | Path) -> Path:
    """Write the invented Frame through the real export gate."""
    path = Path(directory) / "us_housing_charge_probe.h5"
    PolicyEngineUSEngine().write_dataset(invented_probe_frame(), path, PROBE_PERIOD)
    return path


def formula_owned_export_refusals(directory: str | Path) -> dict[str, str]:
    """Confirm each formula-owned housing output is refused as a dataset input."""
    adapter = PolicyEngineUSEngine()
    refusals: dict[str, str] = {}
    for name in FORMULA_OWNED_HOUSING_OUTPUTS:
        bundle = invented_probe_frame()
        entity = adapter.variable_metadata(name).entity
        table = bundle.table(entity)
        table[name] = np.zeros(len(table), dtype=float)
        rebuilt = Frame(
            {name_: bundle.table(name_) for name_ in bundle.entities},
            US_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        path = Path(directory) / f"refused_{name}.h5"
        try:
            adapter.write_dataset(rebuilt, path, PROBE_PERIOD)
        except ValueError as error:
            refusals[name] = str(error)
        else:  # pragma: no cover - a passed export is the probe's failure
            refusals[name] = "EXPORTED"
        if path.exists():  # pragma: no cover - the gate must not write
            refusals[name] += " (FILE WRITTEN)"
    return refusals


def installed_fmr_year_resolution() -> dict[str, Any]:
    """Record the bundled FMR fiscal years and what a 2024 request resolves to."""
    from policyengine_us.parameters.gov.hud.fmr import (
        available_fmr_years,
        nearest_fmr_year,
    )

    resolved = int(nearest_fmr_year(PROBE_PERIOD))
    return {
        "available_fmr_years": [int(year) for year in available_fmr_years],
        "requested_year": PROBE_PERIOD,
        "resolved_year": resolved,
        "falls_forward": resolved != PROBE_PERIOD,
    }


def run_housing_charge_probes(directory: str | Path) -> dict[str, Any]:
    """Run the baseline and HUD-abolition simulations and assemble the receipt."""
    from importlib import metadata

    from policyengine_core.reforms import Reform

    from .reform_validation import default_simulate_factory

    path = export_probe_dataset(directory)
    simulate = default_simulate_factory(path)
    baseline = _measure_block(simulate(None))
    reform = _measure_block(
        simulate(Reform.from_dict(HUD_ABOLITION_REFORM, country_id="us"))
    )
    return {
        "schema_version": 1,
        "probe_kind": "microcosm.us_housing_charge_engine_probe.v1",
        "release_eligible": False,
        "inputs_are_invented": True,
        "period": PROBE_PERIOD,
        "engine": {
            "policyengine-us": metadata.version("policyengine-us"),
            "policyengine-core": metadata.version("policyengine-core"),
        },
        "households": PROBE_HOUSEHOLDS,
        "reform": HUD_ABOLITION_REFORM,
        "baseline": baseline,
        "hud_abolition": reform,
        "formula_owned_export_refusals": formula_owned_export_refusals(directory),
        "fmr_year_resolution": installed_fmr_year_resolution(),
    }
