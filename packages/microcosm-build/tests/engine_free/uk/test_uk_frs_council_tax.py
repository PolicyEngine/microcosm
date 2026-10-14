from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.frs_council_tax import (
    SCOTTISH_CHARGES_IN_RECIPIENT_BILL,
    assert_frs_council_tax_stage_parameters,
    derive_council_tax,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from test_support.paths import paths_for

UK_PACKAGE = (
    paths_for("microcosm-build").repository
    / "packages/microcosm-build/src/microcosm/build/uk"
)


def _raw(**columns) -> pd.DataFrame:
    rows = len(columns["ctannual"])
    defaults = {
        "household_id": list(range(1, rows + 1)),
        "ctreb": [2] * rows,
        "ctrebamt": [0.0] * rows,
        "gvtregno": [1] * rows,
        "ctband": [4] * rows,
        "adulth": [2] * rows,
        "cwatamt1": [0.0] * rows,
        "csewamt1": [0.0] * rows,
        "cwatamtd": [0.0] * rows,
    }
    return pd.DataFrame({**defaults, **columns})


def _derive(raw: pd.DataFrame) -> pd.Series:
    return derive_council_tax(pd.DataFrame({"household_id": raw["household_id"]}), raw)


def test_council_tax_imputes_raw_missing_from_raw_cells() -> None:
    raw = _raw(
        gvtregno=[12, 12, 1, 1, 2, 2],
        ctband=[1, 1, np.nan, np.nan, 2, 2],
        adulth=[1, 1, 1, 1, 2, 2],
        ctannual=[1000.0, -1.0, 600.0, np.nan, -1.0, 0.0],
        # FRS 2024-25 shape: the gross Scottish cells CWATAMT1/CSEWAMT1.
        cwatamtd=[3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        cwatamt1=[4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        csewamt1=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = _derive(raw)

    # A non-recipient's bill carries the gross charges in full.
    scottish_tax = 1000.0 - (4.0 + 5.0) * WEEKS_IN_YEAR
    assert np.isclose(result.loc[1], scottish_tax)
    assert np.isclose(result.loc[2], scottish_tax)
    assert result.loc[4] == 600.0
    # No positive non-recipient bill in the cell: nothing to impute from.
    assert result.loc[5] == 0.0
    # A non-recipient's zero bill (an exempt dwelling) stays zero.
    assert result.loc[6] == 0.0


def test_reported_reduction_is_added_back_to_the_bill() -> None:
    raw = _raw(
        ctannual=[0.0, 500.0, 1300.0],
        ctreb=[1, 1, 2],
        ctrebamt=[20.0, 10.0, 0.0],
    )

    result = _derive(raw)

    # A full reduction leaves CTANNUAL at zero; the liability is the reduction.
    assert result.loc[1] == pytest.approx(20.0 * WEEKS_IN_YEAR)
    assert result.loc[2] == pytest.approx(500.0 + 10.0 * WEEKS_IN_YEAR)
    assert result.loc[3] == pytest.approx(1300.0)


def test_unknown_reduction_takes_the_larger_of_bill_and_cell_mean() -> None:
    raw = _raw(
        ctannual=[1200.0, 1400.0, 300.0, 1500.0],
        ctreb=[2, 2, 1, 1],
        ctrebamt=[0.0, 0.0, 0.0, np.nan],
    )

    result = _derive(raw)

    assert result.loc[3] == pytest.approx(1300.0)
    assert result.loc[4] == pytest.approx(1500.0)


def test_cell_means_pool_non_recipients_only() -> None:
    raw = _raw(
        ctannual=[1200.0, 100.0, np.nan],
        ctreb=[2, 1, 2],
        ctrebamt=[0.0, 15.0, 0.0],
    )

    result = _derive(raw)

    # The recipient's net bill (100) would pull the cell mean down.
    assert result.loc[3] == pytest.approx(1200.0)
    assert result.loc[2] == pytest.approx(100.0 + 15.0 * WEEKS_IN_YEAR)


def test_scottish_recipients_net_the_reduced_share_of_the_gross_charges() -> None:
    raw = _raw(
        gvtregno=[12, 12],
        ctannual=[800.0, 1600.0],
        ctreb=[1, 2],
        ctrebamt=[5.0, 0.0],
        cwatamt1=[4.0, 4.0],
        csewamt1=[5.0, 5.0],
        cwatamtd=[3.0, 3.0],
    )

    result = _derive(raw)

    charges = (4.0 + 5.0) * WEEKS_IN_YEAR
    assert result.loc[1] == pytest.approx(
        800.0 - SCOTTISH_CHARGES_IN_RECIPIENT_BILL * charges + 5.0 * WEEKS_IN_YEAR
    )
    assert result.loc[2] == pytest.approx(1600.0 - charges)


def test_scottish_household_without_a_gross_cell_nets_its_recorded_charge() -> None:
    raw = _raw(
        gvtregno=[12],
        ctannual=[900.0],
        cwatamtd=[3.0],
    )

    assert _derive(raw).loc[1] == pytest.approx(900.0 - 3.0 * WEEKS_IN_YEAR)


def test_bills_outside_scotland_keep_ctannual_or_add_the_reduction() -> None:
    rng = np.random.default_rng(1095)
    rows = 400
    raw = _raw(
        gvtregno=rng.integers(1, 12, rows),
        ctband=rng.integers(1, 9, rows),
        adulth=rng.integers(1, 3, rows),
        ctannual=rng.choice([0.0, 250.0, 900.0, 1800.0, 2600.0], rows),
        ctreb=rng.choice([1, 2], rows),
        ctrebamt=rng.choice([0.0, 4.0, 18.0, 30.0], rows),
    )

    result = _derive(raw).to_numpy()
    ctannual = raw["ctannual"].to_numpy()
    recipient = raw["ctreb"].to_numpy() == 1
    reduction = raw["ctrebamt"].to_numpy() * WEEKS_IN_YEAR

    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result[~recipient], ctannual[~recipient])
    known = recipient & (reduction > 0)
    np.testing.assert_allclose(result[known], ctannual[known] + reduction[known])
    assert (result[recipient] >= ctannual[recipient] - 1e-9).all()


def test_missing_raw_cells_refuse() -> None:
    raw = _raw(ctannual=[100.0]).drop(columns="ctreb")

    with pytest.raises(KeyError, match="ctreb"):
        _derive(raw)


def _committed_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads((UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8"))
    )
    return next(stage for stage in manifest.stages if stage.stage == "frs_council_tax")


def test_committed_manifest_declares_the_implemented_rule() -> None:
    assert_frs_council_tax_stage_parameters(_committed_stage())


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("donor_filter", "raw ctannual > 0"),
        ("value", "Scottish-water-netted CTANNUAL"),
        ("cells", ["gvtregno", "ctband"]),
    ],
)
def test_manifest_drift_from_the_implemented_rule_refuses(parameter, value) -> None:
    stage = _committed_stage()
    operations = [
        {"kind": operation.kind, **dict(operation.parameters)}
        for operation in stage.operations
    ]
    operations[1][parameter] = value
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match=parameter):
        assert_frs_council_tax_stage_parameters(drifted)
