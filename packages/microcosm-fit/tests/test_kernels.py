"""Byte-level parity for the QRF graph adapter and its public direct call."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.fit.model as fit_model_module
import microcosm.fit.qrf as qrf_module
from microcosm.fit import FittedRegimeGatedQRF
from microcosm.fit import fit as fit_qrf
from microcosm.fit.kernels import (
    FIT_QRF_DEPENDENCIES,
    FIT_QRF_TOLERANCE,
    QRF_EXECUTOR_KERNEL,
    QRF_EXECUTOR_SEED_HIGH,
    QRF_PARAM_KERNEL,
    QRFKernel,
)
from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    Determinism,
    Kernel,
    KernelContext,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    source_hash,
)

FIXTURES = Path(__file__).parent / "fixtures" / "graph_parity"
PREDICTORS = ("age", "score")
PARAMS = {
    "donor_target": "observed_y",
    "n_estimators": 9,
    "min_samples_leaf": 1,
    "zero_atol": 1e-6,
    "max_samples_leaf": 8,
    "seed": 947,
}


@pytest.fixture(autouse=True)
def _pin_qrf_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture independent of host CPU count."""
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def _fixture() -> tuple[pd.DataFrame, Weights]:
    donors = pd.read_csv(FIXTURES / "qrf_donors.csv", float_precision="round_trip")
    recipients = pd.read_csv(
        FIXTURES / "qrf_recipients.csv", float_precision="round_trip"
    )
    donors = donors.assign(is_donor=True, is_recipient=False)
    recipients = recipients.assign(
        observed_y=np.nan,
        is_donor=False,
        is_recipient=True,
    )
    weights = Weights(
        np.concatenate(
            [
                donors.pop("weight").to_numpy(dtype=np.float64),
                recipients.pop("weight").to_numpy(dtype=np.float64),
            ]
        ),
        WeightKind.DESIGN,
    )
    table = pd.concat([donors, recipients], ignore_index=True).loc[
        :,
        [
            "person_id",
            *PREDICTORS,
            "observed_y",
            "is_donor",
            "is_recipient",
        ],
    ]
    return table, weights


def _context(
    *,
    params: dict[str, object] | None = None,
    table: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
) -> KernelContext:
    fixture_table, weights = _fixture()
    if table is None:
        table = fixture_table
    actual_params = dict(PARAMS if params is None else params)
    node = Node(
        "impute_y",
        "fit.qrf@1",
        inputs=(
            Slice(
                "person",
                (*PREDICTORS, "observed_y", "is_donor", "is_recipient"),
            ),
        ),
        outputs=(Owned("person", "y", "float64", rows="is_recipient"),),
        params=actual_params,
    )
    return KernelContext(
        node=node,
        tables={"person": table},
        weights={"person": weights},
        strata=pd.Series("all", index=table["person_id"], dtype="string"),
        params=node.params,
        rng=np.random.default_rng(0) if rng is None else rng,
    )


def _direct(
    context: KernelContext, *, seed: int
) -> tuple[FittedRegimeGatedQRF, pd.Series, bytes]:
    table = context.tables["person"]
    donor_mask = table["is_donor"].to_numpy(dtype=bool)
    recipient_mask = ~donor_mask
    donor = table.loc[donor_mask, [*PREDICTORS, "observed_y"]].rename(
        columns={"observed_y": "y"}
    )
    recipient = table.loc[recipient_mask, list(PREDICTORS)]
    params = context.params
    fitted = fit_qrf(
        donor,
        list(PREDICTORS),
        ["y"],
        weights=context.weights["person"].values[donor_mask],
        n_estimators=params["n_estimators"],
        zero_atol=params["zero_atol"],
        max_samples_leaf=params["max_samples_leaf"],
        seed=seed,
    )
    model_bytes = pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)
    drawn = fitted.predict(recipient)["y"]
    expected = pd.Series(
        drawn.to_numpy(dtype=np.float64),
        index=pd.Index(
            table.loc[recipient_mask, "person_id"].to_numpy(), name="person_id"
        ),
        name="y",
        dtype=np.float64,
    )
    return fitted, expected, model_bytes


def _hex_fixture() -> pd.Series:
    expected = pd.read_csv(FIXTURES / "qrf_expected_hex.csv", dtype="string")
    return pd.Series(
        np.array([float.fromhex(value) for value in expected["value_hex"]]),
        index=pd.Index(expected["person_id"].astype("int64"), name="person_id"),
        name="y",
        dtype=np.float64,
    )


def _assert_series_bytes(actual: pd.Series, expected: pd.Series) -> None:
    assert actual.name == expected.name
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert actual.index.equals(expected.index)
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()


def test_param_seed_kernel_matches_direct_call_byte_for_byte() -> None:
    context = _context()
    table_before = context.tables["person"].copy(deep=True)
    weights_before = context.weights["person"].values.tobytes()

    result = QRF_PARAM_KERNEL.run(context)
    direct_model, direct, direct_model_bytes = _direct(context, seed=PARAMS["seed"])
    actual = result.columns[("person", "y")]

    _assert_series_bytes(actual, direct)
    _assert_series_bytes(actual, _hex_fixture())
    assert result.artifacts["model"] == direct_model_bytes
    assert isinstance(direct_model, FittedRegimeGatedQRF)
    artifact_model = pickle.loads(result.artifacts["model"])  # noqa: S301 - trusted
    assert isinstance(artifact_model, FittedRegimeGatedQRF)
    artifact_draw = artifact_model.predict(
        context.tables["person"].loc[
            context.tables["person"]["is_recipient"], list(PREDICTORS)
        ]
    )["y"]
    assert artifact_draw.to_numpy().tobytes() == actual.to_numpy().tobytes()
    assert artifact_model.regimes() == {"y": "positive_only"}
    assert artifact_model.weight_kind == "explicit"

    assert result.receipt["seed"] == PARAMS["seed"]
    assert result.receipt["seed_source"] == "param"
    assert result.receipt["weight_kind"] == "design"
    assert result.receipt["fit_weight_kind"] == "explicit"
    assert result.receipt["donor_rows"] == 40
    assert result.receipt["recipient_rows"] == 12
    assert result.receipt["regime"] == "positive_only"
    pd.testing.assert_frame_equal(context.tables["person"], table_before)
    assert context.weights["person"].values.tobytes() == weights_before


def test_executor_seed_kernel_matches_direct_call_and_consumes_one_seed() -> None:
    params = {name: value for name, value in PARAMS.items() if name != "seed"}
    kernel_rng = np.random.default_rng(20260901)
    direct_rng = np.random.default_rng(20260901)
    direct_seed = int(direct_rng.integers(0, QRF_EXECUTOR_SEED_HIGH))
    context = _context(params=params, rng=kernel_rng)

    result = QRF_EXECUTOR_KERNEL.run(context)
    _, direct, direct_model_bytes = _direct(context, seed=direct_seed)

    _assert_series_bytes(result.columns[("person", "y")], direct)
    assert result.artifacts["model"] == direct_model_bytes
    assert result.receipt["seed"] == direct_seed
    assert result.receipt["seed_source"] == "executor"
    np.testing.assert_array_equal(
        kernel_rng.integers(0, 2**31 - 1, size=8),
        direct_rng.integers(0, 2**31 - 1, size=8),
    )


def test_capabilities_protocol_and_wrapped_source_hash() -> None:
    assert isinstance(QRF_PARAM_KERNEL, Kernel)
    assert isinstance(QRF_EXECUTOR_KERNEL, Kernel)
    assert QRF_PARAM_KERNEL.ref == QRF_EXECUTOR_KERNEL.ref == "fit.qrf@1"
    assert QRF_PARAM_KERNEL.capabilities == Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.TOLERANCE_BOUND,
        seed_source=SeedSource.PARAM,
        dependencies=FIT_QRF_DEPENDENCIES,
        tolerance=FIT_QRF_TOLERANCE,
    )
    assert QRF_EXECUTOR_KERNEL.capabilities.seed_source is SeedSource.EXECUTOR
    assert QRF_PARAM_KERNEL.implementation_hash() == source_hash(
        QRFKernel,
        fit_qrf,
        fit_model_module,
        qrf_module,
        dependencies=FIT_QRF_DEPENDENCIES,
    )


@pytest.mark.parametrize(
    ("kernel", "mutate_params", "match"),
    [
        (
            QRFKernel(SeedSource.PARAM),
            lambda params: params.pop("seed"),
            "requires params\\['seed'\\]",
        ),
        (
            QRFKernel(SeedSource.EXECUTOR),
            lambda params: None,
            "must omit params\\['seed'\\]",
        ),
        (
            QRFKernel(SeedSource.PARAM),
            lambda params: params.update(min_samples_leaf=2),
            "supports only.*1",
        ),
    ],
    ids=["param-needs-seed", "executor-forbids-seed", "leaf-default-only"],
)
def test_parameter_contract_rejects_mismatches(
    kernel: QRFKernel, mutate_params, match: str
) -> None:
    params = dict(PARAMS)
    mutate_params(params)
    with pytest.raises(ValueError, match=match):
        kernel.run(_context(params=params))


def test_masks_must_be_boolean_non_null_complements() -> None:
    table, _ = _fixture()
    not_complement = table.copy()
    not_complement.loc[0, "is_recipient"] = True
    with pytest.raises(ValueError, match="exact complement"):
        QRF_PARAM_KERNEL.run(_context(table=not_complement))

    not_boolean = table.copy()
    not_boolean["is_donor"] = not_boolean["is_donor"].astype("int64")
    with pytest.raises(TypeError, match="boolean dtype"):
        QRF_PARAM_KERNEL.run(_context(table=not_boolean))

    nullable = table.copy()
    nullable["is_donor"] = nullable["is_donor"].astype("boolean")
    nullable.loc[0, "is_donor"] = pd.NA
    with pytest.raises(ValueError, match="contains null"):
        QRF_PARAM_KERNEL.run(_context(table=nullable))
