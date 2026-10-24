# ruff: noqa: F401

"""US PUF support-channel expansion tests."""

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


def _raw_asec_predictor_frame(*, with_tax_unit_roles: bool = False) -> Frame:
    """Raw ASEC person fields, optionally with the constructed tax-unit roles.

    The roles, age and sex are what the tax-unit construction stage hands the
    PUF imputation; the PUF demographic predictors read them (microcosm#982).
    """

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
    if with_tax_unit_roles:
        tables["person"] = tables["person"].assign(
            age=[65, 40, 61],
            is_female=[False, True, True],
            tax_unit_role_input=["HEAD", "SPOUSE", "HEAD"],
        )
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


class TestPufSupportWeightsAuditWiring:
    """The PUF-support production fit emits a weights-audit record.

    This is what makes the build-level weights audit (microcosm #300) real rather
    than dead code: the actual production imputation records the weight kind it
    resolved, so a release manifest carries it and a ``"none"`` fit fails the
    release. The fit runs on a synthetic frame with no ``policyengine_us``, so
    this proves the wiring end to end in CI's engine-less environment.
    """

    def _donor(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filing_status_code": [1.0, 2.0, 4.0, 1.0],
                "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
                "employment_income_before_lsr": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def _impute(self, fit_records):
        return impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            self._donor(),
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
            fit_records=fit_records,
        )

    def test_production_fit_records_design_weight_kind(self) -> None:
        from microcosm.build import FitWeightRecord
        from microcosm.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        fit_records: list[FitWeightRecord] = []
        self._impute(fit_records)

        assert fit_records == [FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "design")]

    def test_recorded_fit_passes_the_weights_audit_gate(self) -> None:
        from microcosm.build import weights_audit_gate

        fit_records = []
        self._impute(fit_records)

        result = weights_audit_gate(fit_records)
        assert result.passed
        from microcosm.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        assert result.details["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_wired_gate_would_fail_a_none_fit(self) -> None:
        # Prove the wired gate can actually find something: swap the resolved
        # kind to "none" and the release-blocking gate fails, naming the fit.
        from microcosm.build import FitWeightRecord, weights_audit_gate
        from microcosm.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        result = weights_audit_gate([FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "none")])
        assert not result.passed
        assert US_PUF_SUPPORT_FIT_NAME in result.failures[0]
        assert "unweighted" in result.failures[0]

    def test_records_are_only_emitted_when_a_sink_is_provided(self) -> None:
        # The out-parameter is opt-in: existing callers that pass nothing get
        # the same Frame return and are unaffected.
        imputed = impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            self._donor(),
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
        )
        assert imputed.table("person") is not None

    def test_production_fit_records_design_kind_under_installed_metadata(self) -> None:
        # The engine-less tests above run the imputation with a trivial output
        # that never trips the formula-owned guard. This gated test runs the same
        # audited seam with the installed PolicyEngine-US source guard active
        # (assert_formula_owned_blocklist_current + resolve_formula_owned_outputs
        # both read pinned engine metadata), over real leaf-input outputs, so the
        # seam is proven end to end on the production code path an actual build
        # takes: the guard passes on genuine leaves, the DESIGN-weighted fit
        # records "design", and the release-blocking gate passes carrying it.
        _installed_variable_metadata_index()
        from microcosm.build import FitWeightRecord, weights_audit_gate
        from microcosm.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        donor = pd.DataFrame(
            {
                "filing_status_code": [1.0, 2.0, 4.0, 1.0],
                "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
                "employment_income_before_lsr": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
                "qualified_dividend_income": [10.0, 20.0, 30.0, 40.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        fit_records: list = []
        impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            donor,
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=(
                "employment_income_before_lsr",
                "qualified_dividend_income",
            ),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
            fit_records=fit_records,
        )

        assert fit_records == [FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "design")]
        result = weights_audit_gate(fit_records)
        assert result.passed
        assert result.details["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }


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
