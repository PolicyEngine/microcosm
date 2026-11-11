"""The area-to-BRMA layer names only BRMAs the engine prices (microcosm#1262).

A BRMA the engine's enum or LHA rate table did not know would leave a placed
household without a Local Housing Allowance rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import policyengine_uk
from policyengine_uk.variables.household.demographic.locations import BRMAName

from microcosm.build.uk_runtime import brma_area_layer as bal
from test_support.paths import paths_for

_UK = paths_for("microcosm-build").package / "src/microcosm/build/uk"
_RATES = (
    Path(policyengine_uk.__file__).parent
    / "parameters/gov/dwp/LHA/lha_published_rates.csv.gz"
)


def _layer_names() -> set[str]:
    layer = bal.load_brma_area_layer((_UK / "brma_area_layer.json").read_bytes())
    return set(layer.brma_by_area["ew_lsoa21"].values()) | set(
        layer.brma_by_area["scotland_oa22"].values()
    )


def test_layer_names_are_the_engine_brmas_outside_northern_ireland() -> None:
    table = json.loads((_UK / "brma_private_rented_households.json").read_text("utf-8"))
    northern_ireland = {
        brma for cells in table["cells"]["NORTHERN_IRELAND"].values() for brma in cells
    }
    assert _layer_names() == set(BRMAName.__members__) - northern_ireland


def test_every_layer_brma_has_published_lha_rates() -> None:
    rates = pd.read_csv(_RATES, usecols=["brma"])
    assert _layer_names() <= set(rates["brma"].astype(str))


def test_every_rent_table_brma_is_an_engine_brma_with_rates() -> None:
    from microcosm.build.uk_runtime import private_rent_localisation as prl

    table = prl.load_rent_quantiles((_UK / "brma_rent_quantiles.json").read_bytes())
    names = {brma for brma, _ in table.knots}
    assert names == _layer_names()
    assert names <= set(BRMAName.__members__)
