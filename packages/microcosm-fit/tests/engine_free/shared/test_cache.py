from __future__ import annotations

import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from microcosm.fit import cache as cache_module
from microcosm.fit.cache import (
    QRFCacheKey,
    load_fitted_qrf_cache,
    save_fitted_qrf_cache,
)
from microcosm.fit.qrf import _Forest


def _key(seed: int = 0) -> QRFCacheKey:
    return QRFCacheKey(
        donor_sha256="a" * 64,
        predictors=("x",),
        targets=("y",),
        seed=seed,
        weight_kind="explicit",
        package_versions={"microcosm-fit": "test"},
    )


def test_qrf_cache_save_load_and_miss(tmp_path) -> None:
    model = {"fitted": True}
    path = save_fitted_qrf_cache(model, tmp_path, _key())

    assert path.exists()
    assert load_fitted_qrf_cache(tmp_path, _key()) == model
    assert load_fitted_qrf_cache(tmp_path, _key(seed=1)) is None


def test_entry_holding_a_library_class_forest_is_a_cache_miss(
    tmp_path, monkeypatch
) -> None:
    """An entry from schema 1 is ignored rather than unpickled and refused."""
    predictors = np.arange(40, dtype=np.float64).reshape(-1, 1)
    target = predictors[:, 0] + 1.0
    library_model = RandomForestQuantileRegressor(
        n_estimators=2, random_state=0, n_jobs=1
    )
    library_model.fit(predictors, target)
    schema_one_entry = {"forest": _Forest(model=library_model, columns=("x",))}
    with monkeypatch.context() as schema_one:
        schema_one.setattr(cache_module, "SCHEMA_VERSION", 1)
        schema_one_path = save_fitted_qrf_cache(schema_one_entry, tmp_path, _key())

    assert schema_one_path.exists()
    assert load_fitted_qrf_cache(tmp_path, _key()) is None
