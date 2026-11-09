from __future__ import annotations

from microcosm.fit.cache import (
    QRFCacheKey,
    load_fitted_qrf_cache,
    save_fitted_qrf_cache,
)


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
