"""US PUF support-channel expansion tests."""

# ruff: noqa: F401

import importlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.puf_support as puf_support_module
from microcosm.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    CPS_CARRIED_FORMULA_OWNED_COLUMNS,
    CPS_CARRIED_PERSON_INPUTS,
    CPS_CARRIED_SPM_UNIT_INPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING,
    clone_us_frame_for_puf_support,
    derive_us_cps_carried_inputs,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    split_us_puf_e19200_by_agi_band,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.puf_support import (
    _PUF_TAX_DETAIL_SIGNED_MASS_CALIBRATED_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS,
    assert_formula_owned_blocklist_current,
    resolve_formula_owned_outputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)


def _installed_variable_metadata_index() -> PolicyEngineUSVariableMetadataIndex:
    try:
        return PolicyEngineUSVariableMetadataIndex()
    except ImportError:
        pytest.skip("requires the policyengine-us [us] extra")


def _legacy_snap_to_observed_values(
    values: list[object] | np.ndarray,
    observed: list[object] | np.ndarray,
) -> np.ndarray:
    """Reference the pre-OOM-fix recipient-by-donor implementation."""

    value_array = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    observed_values = pd.to_numeric(pd.Series(observed), errors="coerce").fillna(0.0)
    observed_array = np.unique(
        np.rint(observed_values.to_numpy(dtype=np.float64)).clip(min=0.0)
    )
    if len(observed_array) == 0:
        return value_array.to_numpy(dtype=np.float64)
    positions = np.abs(
        value_array.to_numpy(dtype=np.float64)[:, None] - observed_array[None, :]
    ).argmin(axis=1)
    return observed_array[positions]


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "age": np.asarray([42, 40, 51], dtype="int64"),
            "employment_income_before_lsr": np.asarray(
                [50_000, 20_000, 125_000],
                dtype="int64",
            ),
            "partnership_income": [1_000.0, 2_000.0, 3_000.0],
            "s_corp_income": [4_000.0, 5_000.0, 6_000.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    strata = pd.Series(
        ["asec_2024", "asec_2024", "asec_2023"],
        name="stratum",
    )
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    return Frame(tables, US_SCHEMA, weights, strata)


def _raw_asec_predictor_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2, 3], dtype="int64"),
                "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
                "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                "person_marital_unit_id": np.asarray(
                    [10000, 10000, 20000], dtype="int64"
                ),
                "A_AGE": [65, 40, 61],
                "A_SEX": [1, 2, 2],
                "WSAL_VAL": [100.0, 200.0, 0.0],
                "SEMP_VAL": [10.0, 0.0, 30.0],
                "INT_VAL": [1_000.0, 0.0, 200.0],
                "DIV_VAL": [100.0, 0.0, 50.0],
                "CAP_VAL": [50.0, 0.0, 25.0],
                "SS_VAL": [1_000.0, 900.0, 800.0],
                "RESNSS1": [1, 2, 0],
                "RESNSS2": [0, 0, 0],
                "PNSN_VAL": [1_000.0, 0.0, 0.0],
                "ANN_VAL": [100.0, 0.0, 0.0],
                "DST_SC1": [4, 3, 0],
                "DST_VAL1": [500.0, 200.0, 0.0],
                "NOW_MRK": [1, 2, 1],
                "NOW_NONM": [2, 1, 2],
                "NOW_MCAID": [1, 2, 2],
                "NOW_GRP": [2, 1, 1],
                "NOW_CHAMPVA": [2, 2, 1],
                "NOW_MIL": [1, 2, 2],
                "NOW_VACARE": [2, 1, 2],
                "NOW_OTHMT": [2, 1, 2],
                "NOW_IHSFLG": [1, 2, 2],
                "RNT_VAL": [20.0, 0.0, 0.0],
                "FRSE_VAL": [5.0, 0.0, 0.0],
                "UC_VAL": [0.0, 70.0, 0.0],
                "OI_OFF": [19, 0, 0],
                "OI_VAL": [3.0, 0.0, 0.0],
                "PHIP_VAL": [400.0, 0.0, 50.0],
                "PEMCPREM": [100.0, 0.0, 25.0],
                "PMED_VAL": [200.0, 0.0, 40.0],
                "POTC_VAL": [30.0, 0.0, 10.0],
                # Unit 100 is TANF-enrolled through the PAW_TYP 3 ("both")
                # member; unit 200's positive amount is PAW_TYP 2 (other
                # cash welfare) and must NOT mark TANF enrollment.
                "PAW_VAL": [0.0, 125.0, 80.0],
                "PAW_TYP": [0, 3, 2],
                "SPM_SNAPSUB": [0.0, 0.0, 900.0],
                "WICYN": [0, 1, 2],
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
    )


class _FakeFormulaOwnedEngine:
    """Minimal metadata source for the formula-owned guard (issue #301).

    Reports exactly the names in ``formula_owned`` as formula-owned, restricted
    to the requested set — the contract
    :func:`resolve_formula_owned_outputs` and
    :func:`assert_formula_owned_blocklist_current` depend on. Injecting it keeps
    these tests deterministic whether or not ``policyengine_us`` happens to be
    installed in the test environment.
    """

    def __init__(self, formula_owned: set[str]) -> None:
        self._formula_owned = set(formula_owned)

    def formula_owned_outputs(self, names) -> set[str]:
        return set(names) & self._formula_owned


class _ImportErrorEngine:
    """An adapter whose lazy policyengine_us import is missing at call time."""

    def formula_owned_outputs(self, names):
        raise ImportError("No module named 'policyengine_us'")


def _tanf_gate_person(
    amounts: Sequence[float],
    types: Sequence[int] | None,
) -> pd.DataFrame:
    person = pd.DataFrame(
        {
            "person_spm_unit_id": np.arange(len(amounts), dtype=np.int64) + 100,
            "PAW_VAL": np.asarray(amounts, dtype=np.float64),
        }
    )
    if types is not None:
        person["PAW_TYP"] = np.asarray(types, dtype=np.int64)
    return person


# --------------------------------------------------------------------------- #
# Signed-mass calibration: pin the imputed net signed mass of a sparse,
# sign-mixed, heavy-tailed person output (farm_operations_income, microcosm
# farm-chain-sign-structure; partnership_self_employment_net_earnings, #432) to
# the donor instrument, so the regime-gated QRF's regression toward balance
# cannot flip or cancel the source's net sign.
# --------------------------------------------------------------------------- #


def _puf_recipient_frame(
    employment_income: Sequence[float],
    self_employment_income: Sequence[float],
    household_weights: Sequence[float],
) -> Frame:
    """One person per household/tax-unit, so person and tax-unit weights align."""

    n = len(household_weights)
    ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "employment_income": np.asarray(employment_income, dtype=np.float64),
            "self_employment_income": np.asarray(
                self_employment_income, dtype=np.float64
            ),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.full(n, 6, dtype="int64")}
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": ids, "filing_status_input": ["SINGLE"] * n}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    weights = {
        "household": Weights(
            np.asarray(household_weights, dtype=np.float64), WeightKind.DESIGN
        )
    }
    return Frame(tables, US_SCHEMA, weights)


def _per_weight_legs(
    values: Sequence[float], weights: Sequence[float]
) -> tuple[float, float]:
    """Return (positive, negative) per-unit-weight weighted leg masses."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    positive = float((np.maximum(values, 0.0) * weights).sum() / total)
    negative = float((np.minimum(values, 0.0) * weights).sum() / total)
    return positive, negative


def _puf_channel_person_legs(frame: Frame, column: str) -> tuple[float, float]:
    """Per-unit-weight positive/negative leg masses on the PUF person channel."""

    person = frame.table("person")
    household_weight = pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"],
    )
    person_weight = person["person_household_id"].map(household_weight).to_numpy()
    puf = (
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ).to_numpy()
    values = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
    return _per_weight_legs(values[puf], person_weight[puf])


__all__ = [name for name in globals() if not name.startswith("__")]
