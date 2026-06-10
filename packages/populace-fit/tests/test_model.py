"""The weight contract and entity resolution of the conditional-model protocol.

These tests pin the operator's defining rule (DESIGN.md, "populace-fit"): a fit
is weighted by construction, ``weights="none"`` is the *only* unweighted path,
and a misspelled or mismatched weights spec fails loudly rather than silently
falling back to an unweighted fit. They also pin that predictors and targets
must resolve to a single entity, and that the protocols are satisfied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.fit import (
    NO_WEIGHTS,
    QRF,
    ConditionalModel,
    FittedModel,
    FittedRegimeGatedQRF,
    RegimeGatedQRF,
    fit,
)
from populace.fit.model import predictors_targets_entity, resolve_fit_weights
from populace.frame import (
    EntitySchema,
    Frame,
    WeightKind,
    Weights,
)

# ----------------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------------


def test_qrf_is_a_conditional_model() -> None:
    """The canonical model satisfies the ConditionalModel protocol."""
    assert isinstance(RegimeGatedQRF(), ConditionalModel)
    assert QRF is RegimeGatedQRF


def test_fitted_model_satisfies_the_fitted_protocol(
    weight_correlated_frame,
) -> None:
    """A fitted model satisfies the FittedModel protocol."""
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    fitted = fit(frame, ["age", "is_male"], ["target"], n_estimators=10, seed=0)
    assert isinstance(fitted, FittedModel)
    assert isinstance(fitted, FittedRegimeGatedQRF)


# ----------------------------------------------------------------------------
# weights="none" is the only unweighted path
# ----------------------------------------------------------------------------


def test_none_is_the_only_unweighted_path_typo_raises(
    weight_correlated_frame,
) -> None:
    """A misspelled weights kind raises — it never silently fits unweighted.

    This is the core safety property. ``"nonе"`` (or any unknown spelling) must
    not degrade into an unweighted fit the way a forgotten ``weight_col`` did in
    microimpute; it must raise and point the caller at ``weights="none"``.
    """
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    with pytest.raises(ValueError, match="Unknown weights spec"):
        fit(frame, ["age", "is_male"], ["target"], weights="non")
    with pytest.raises(ValueError, match="Unknown weights spec"):
        fit(frame, ["age", "is_male"], ["target"], weights="unweighted")
    # The error names the explicit escape hatch.
    with pytest.raises(ValueError, match="weights='none'"):
        fit(frame, ["age", "is_male"], ["target"], weights="off")


def test_non_string_non_kind_weights_raises(weight_correlated_frame) -> None:
    """A weights spec of the wrong type raises a TypeError, not a stray fit."""
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    with pytest.raises(TypeError, match="WeightKind"):
        fit(frame, ["age", "is_male"], ["target"], weights=0)
    with pytest.raises(TypeError, match="weights='none'"):
        fit(frame, ["age", "is_male"], ["target"], weights=[1.0, 2.0])


def test_weights_string_and_kind_are_equivalent(weight_correlated_frame) -> None:
    """``weights="design"`` and ``weights=WeightKind.DESIGN`` resolve the same."""
    frame, _, _ = weight_correlated_frame(seed=0, n=300)
    entity = "person"
    by_string = resolve_fit_weights(frame, entity, "design")
    by_kind = resolve_fit_weights(frame, entity, WeightKind.DESIGN)
    np.testing.assert_array_equal(by_string, by_kind)
    # And "none" yields no vector (the unweighted path).
    assert resolve_fit_weights(frame, entity, NO_WEIGHTS) is None


def test_requesting_a_kind_the_entity_lacks_raises(
    weight_correlated_frame,
) -> None:
    """Asking for calibrated weights on a design-weighted entity is refused.

    The frame stores design weights; requesting ``"calibrated"`` must not
    silently fall through to design (or to unweighted). It raises and names the
    stored kind and the fix.
    """
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    with pytest.raises(ValueError, match="stored weights are 'design'"):
        fit(frame, ["age", "is_male"], ["target"], weights="calibrated")
    with pytest.raises(ValueError, match="stored weights are 'design'"):
        resolve_fit_weights(frame, "person", WeightKind.IMPORTANCE)


def test_none_path_returns_no_vector_even_with_weights_present(
    weight_correlated_frame,
) -> None:
    """``weights="none"`` ignores the stored weights deliberately."""
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    assert resolve_fit_weights(frame, "person", NO_WEIGHTS) is None


# ----------------------------------------------------------------------------
# Entity resolution: predictors and targets must share one entity
# ----------------------------------------------------------------------------


def _two_entity_frame() -> Frame:
    """A frame with a person column and a household column at different grains."""
    person = pd.DataFrame(
        {
            "person_id": [0, 1, 2],
            "person_household_id": [1, 1, 2],
            "age": [40.0, 8.0, 33.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2], "state_income": [50.0, 70.0]})
    schema = EntitySchema(group_entities=("household",))
    return Frame(
        {"person": person, "household": household},
        schema,
        {"household": Weights(values=np.array([10.0, 20.0]), kind=WeightKind.DESIGN)},
    )


def test_fit_refuses_predictors_and_targets_spanning_entities() -> None:
    """A person predictor and a household target span grains and are refused.

    Silently cross-joining a person column with a household column would fit at
    a mismatched grain with mismatched weights. The resolver refuses and names
    each column's entity.
    """
    frame = _two_entity_frame()
    with pytest.raises(ValueError, match="must all live on one entity"):
        fit(frame, ["age"], ["state_income"], weights="none")
    # The message names the entities involved.
    with pytest.raises(ValueError, match="household"):
        predictors_targets_entity(frame, ["age"], ["state_income"])


def test_fit_refuses_unknown_columns() -> None:
    """An unknown predictor or target names the missing column."""
    frame = _two_entity_frame()
    with pytest.raises(ValueError, match="not found on any entity table"):
        fit(frame, ["age", "nonexistent"], ["age"], weights="none")


def test_fit_refuses_empty_predictors_or_targets() -> None:
    """A fit needs at least one predictor and one target."""
    frame = _two_entity_frame()
    with pytest.raises(ValueError, match="at least one predictor"):
        predictors_targets_entity(frame, [], ["age"])
    with pytest.raises(ValueError, match="at least one target"):
        predictors_targets_entity(frame, ["age"], [])


def test_predict_missing_predictor_column_raises(weight_correlated_frame) -> None:
    """Predicting on a DataFrame that lacks a predictor names the gap."""
    frame, _, _ = weight_correlated_frame(seed=0, n=200)
    fitted = fit(frame, ["age", "is_male"], ["target"], n_estimators=10, seed=0)
    bad = frame.table("person").drop(columns=["is_male"])
    with pytest.raises(ValueError, match="missing predictor column"):
        fitted.predict(bad)
