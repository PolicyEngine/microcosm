"""The DataFrame front door: fitting without a Frame, weights stated explicitly.

A plain :class:`pandas.DataFrame` has no typed weight vectors, so the operator's
defining rule — no silent unweighted fit — cannot ride on a default. The
DataFrame path therefore *requires* the caller to state weights: a weight
column name, a weight vector, or ``weights="none"`` and meaning it. These tests
pin that contract, the validation around it, and bit-for-bit parity with the
Frame path (same data + same weights + same seed => identical draws).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import fit as fit_convenience
from microcosm.fit.model import FittedModel
from microcosm.fit.qrf import RegimeGatedQRF
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _toy_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A tiny numeric DataFrame for validation-error tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "y": np.abs(rng.normal(100.0, 10.0, n)),
            "weight": np.full(n, 2.0),
        }
    )


def _small_model(**overrides) -> RegimeGatedQRF:
    """A fast model for tests that assert plumbing, not statistics."""
    kwargs = {"n_estimators": 5, "seed": 0}
    kwargs.update(overrides)
    return RegimeGatedQRF(**kwargs)


def test_dataframe_fit_requires_explicit_weights() -> None:
    """Omitting ``weights`` on a DataFrame fit is an error, not a default.

    The Frame path defaults to design weights because the frame carries them; a
    DataFrame carries nothing, so the default ("design") must fail loudly
    rather than silently fitting unweighted (the eCPS landmine failure mode).
    """
    df = _toy_df()
    with pytest.raises(ValueError, match="no typed weights"):
        _small_model().fit(df, ["x"], ["y"])


@pytest.mark.parametrize(
    "spec",
    ["design", "importance", "calibrated", WeightKind.DESIGN],
    ids=["design", "importance", "calibrated", "WeightKind"],
)
def test_typed_kind_specs_are_frame_concepts(spec) -> None:
    """Typed weight kinds cannot be requested from a bare DataFrame."""
    df = _toy_df()
    with pytest.raises(ValueError, match="typed weight"):
        _small_model().fit(df, ["x"], ["y"], weights=spec)


def test_weight_column_vector_and_series_are_equivalent() -> None:
    """A weight column name, an array, and a Series fit identically.

    All three spell the same vector, so with one seed the three fitted models
    must produce bit-identical draws.
    """
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "y": np.abs(rng.normal(50_000.0, 9_000.0, n)),
            "w": rng.integers(1, 60, n).astype(float),
        }
    )
    new_rows = df.loc[:, ["x"]].iloc[:200]

    draws = []
    for weights in ("w", df["w"].to_numpy(), df["w"]):
        fitted = _small_model(n_estimators=25).fit(df, ["x"], ["y"], weights=weights)
        draws.append(fitted.predict(new_rows))
    pd.testing.assert_frame_equal(draws[0], draws[1])
    pd.testing.assert_frame_equal(draws[0], draws[2])


@pytest.mark.parametrize("unweighted", [False, True], ids=["design", "none"])
def test_dataframe_fit_matches_frame_fit_bit_for_bit(
    weight_correlated_frame, unweighted
) -> None:
    """Same rows, same weights, same seed: the two front doors are one model.

    The DataFrame path must be the Frame path minus the typed-weight
    resolution, so fitting on ``frame.table("person")`` with the frame's own
    weight vector (or ``"none"`` on both sides) and one seed has to reproduce
    the Frame fit's draws exactly.
    """
    frame, _, weights = weight_correlated_frame(seed=11, n=2000)
    table = frame.table("person")
    predictors, targets = ["age", "is_male"], ["target"]
    new_rows = table.loc[:, predictors].iloc[:500]

    frame_spec = "none" if unweighted else "design"
    df_spec = "none" if unweighted else weights

    from_frame = _small_model(n_estimators=50).fit(
        frame, predictors, targets, weights=frame_spec
    )
    from_df = _small_model(n_estimators=50).fit(
        table, predictors, targets, weights=df_spec
    )
    pd.testing.assert_frame_equal(
        from_frame.predict(new_rows), from_df.predict(new_rows)
    )


def test_none_is_reserved_even_if_a_column_is_named_none() -> None:
    """``weights="none"`` always means unweighted, never a column lookup."""
    df = _toy_df().assign(none=np.full(60, 9.0))
    fitted = _small_model().fit(df, ["x"], ["y"], weights="none")
    assert isinstance(fitted, FittedModel)


@pytest.mark.parametrize(
    ("mutate", "weights", "match"),
    [
        # Column-form errors.
        (None, "no_such_column", r"not a column"),
        (None, "x", r"also a predictor"),
        (None, "y", r"also a target"),
        ("stringy_weight", "weight", r"numeric"),
        # Vector-form errors.
        (None, np.full(59, 1.0), r"length"),
        (None, np.full((60, 1), 1.0), r"1-D"),
        (None, np.r_[np.full(59, 1.0), -1.0], r"negative"),
        (None, np.r_[np.full(59, 1.0), np.nan], r"finite"),
        (None, np.zeros(60), r"zero"),
        (None, object(), r"weight column name"),
    ],
    ids=[
        "unknown-column",
        "column-is-predictor",
        "column-is-target",
        "non-numeric-column",
        "wrong-length",
        "not-1d",
        "negative",
        "nan",
        "all-zero",
        "not-a-spec",
    ],
)
def test_dataframe_weight_validation(mutate, weights, match) -> None:
    """Bad weight specs raise naming the problem, before any fitting."""
    df = _toy_df()
    if mutate == "stringy_weight":
        df["weight"] = "heavy"
    with pytest.raises((ValueError, TypeError), match=match):
        _small_model().fit(df, ["x"], ["y"], weights=weights)


def test_dataframe_fit_refuses_missing_and_malformed_columns() -> None:
    """Column validation mirrors the Frame path: missing names are named."""
    df = _toy_df()
    with pytest.raises(ValueError, match="missing"):
        _small_model().fit(df, ["x", "ghost"], ["y"], weights="weight")
    with pytest.raises(ValueError, match="missing"):
        _small_model().fit(df, ["x"], ["phantom"], weights="weight")
    with pytest.raises(ValueError, match="at least one predictor"):
        _small_model().fit(df, [], ["y"], weights="weight")
    with pytest.raises(ValueError, match="both predictor and target"):
        _small_model().fit(df, ["x"], ["x"], weights="weight")


def test_df_fitted_model_draws_dataframes_but_refuses_frames(
    make_person_frame,
) -> None:
    """A DataFrame-fitted model has no entity, so Frame prediction is refused.

    Drawing for a DataFrame keeps working and preserves the caller's index.
    """
    df = _toy_df(n=80)
    df.index = pd.RangeIndex(1000, 1080)
    fitted = _small_model().fit(df, ["x"], ["y"], weights="weight")

    drawn = fitted.predict(df.loc[:, ["x"]])
    assert list(drawn.columns) == ["y"]
    assert drawn.index.equals(df.index)

    frame = make_person_frame(
        {"x": np.zeros(16), "y": np.ones(16)}, weights=np.full(16, 2.0)
    )
    with pytest.raises(ValueError, match="fit on a plain DataFrame"):
        fitted.predict(frame)


def test_top_level_fit_accepts_dataframes() -> None:
    """The convenience ``microcosm.fit.fit`` exposes the same front door."""
    df = _toy_df()
    fitted = fit_convenience(df, ["x"], ["y"], weights="weight", n_estimators=5, seed=1)
    assert isinstance(fitted, FittedModel)
    assert len(fitted.predict(df.loc[:, ["x"]])) == len(df)


class TestFittedModelRecordsResolvedWeightKind:
    """A fitted model exposes the weight kind it *resolved* at fit time.

    The build-level weights audit (microcosm #300) records this per production
    fit and blocks a release on an unlisted ``"none"``. Recording the *resolved*
    kind — not the spec the caller passed — is what makes the audit trustworthy:
    on a Frame the resolved kind reflects inherited weights, and an unweighted
    fit reads back ``"none"`` no matter how it was spelled.
    """

    def test_frame_design_weights_read_back_as_design(self, make_person_frame) -> None:
        frame = make_person_frame(
            {"x": np.zeros(16), "y": np.arange(1.0, 17.0)},
            weights=np.full(16, 3.0),
        )
        fitted = _small_model().fit(frame, ["x"], ["y"], weights="design")
        assert fitted.weight_kind == "design"

    def test_frame_calibrated_weights_read_back_as_calibrated(self) -> None:
        # A frame whose person weights are calibrated: the resolved kind must be
        # what the frame carries, not the spec's spelling.
        n = 16
        person = pd.DataFrame(
            {
                "person_id": np.arange(n, dtype="int64"),
                "person_household_id": np.arange(n, dtype="int64"),
                "x": np.zeros(n),
                "y": np.arange(1.0, n + 1.0),
            }
        )
        household = pd.DataFrame({"household_id": np.arange(n, dtype="int64")})
        frame = Frame(
            {"person": person, "household": household},
            EntitySchema(group_entities=("household",)),
            {"person": Weights(np.full(n, 4.0), WeightKind.CALIBRATED)},
        )
        fitted = _small_model().fit(frame, ["x"], ["y"], weights="calibrated")
        assert fitted.weight_kind == "calibrated"

    def test_frame_unweighted_reads_back_as_none(self, make_person_frame) -> None:
        frame = make_person_frame(
            {"x": np.zeros(16), "y": np.arange(1.0, 17.0)},
            weights=np.full(16, 3.0),
        )
        fitted = _small_model().fit(frame, ["x"], ["y"], weights="none")
        assert fitted.weight_kind == "none"

    def test_dataframe_vector_reads_back_as_explicit(self) -> None:
        # A bare DataFrame has no typed kind; an explicit weight vector is
        # weighted-but-untyped, recorded as "explicit" (never "none").
        df = _toy_df()
        fitted = _small_model().fit(df, ["x"], ["y"], weights=df["weight"].to_numpy())
        assert fitted.weight_kind == "explicit"

    def test_dataframe_column_reads_back_as_explicit(self) -> None:
        df = _toy_df()
        fitted = _small_model().fit(df, ["x"], ["y"], weights="weight")
        assert fitted.weight_kind == "explicit"

    def test_dataframe_unweighted_reads_back_as_none(self) -> None:
        df = _toy_df()
        fitted = _small_model().fit(df, ["x"], ["y"], weights="none")
        assert fitted.weight_kind == "none"

    def test_weight_kind_is_read_only(self, make_person_frame) -> None:
        frame = make_person_frame(
            {"x": np.zeros(16), "y": np.arange(1.0, 17.0)},
            weights=np.full(16, 3.0),
        )
        fitted = _small_model().fit(frame, ["x"], ["y"], weights="design")
        with pytest.raises(AttributeError):
            fitted.weight_kind = "none"  # type: ignore[misc]
