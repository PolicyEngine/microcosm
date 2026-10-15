"""The twin-conditional transport of the SPI stage-1 draw (microcosm#840)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import spi_rank_transport as transport
from microcosm.build.uk_runtime.spi_rank_transport import (
    SPI_STAGE1_ANALOGUE_COLUMNS,
    analogue_source_columns,
    analogue_targets,
    assert_analogue_map_covers,
    identity_uniforms,
    observed_analogues,
    transport_uniforms,
)
from microcosm.build.uk_runtime.spi_support import SPI_INCOME_QRF_OUTPUT_COLUMNS
from microcosm.fit import fit


def test_analogue_map_names_every_stage1_output_and_refuses_drift() -> None:
    assert_analogue_map_covers(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert set(SPI_STAGE1_ANALOGUE_COLUMNS) == set(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert analogue_targets() == tuple(
        target
        for target in SPI_INCOME_QRF_OUTPUT_COLUMNS
        if SPI_STAGE1_ANALOGUE_COLUMNS[target] is not None
    )
    with pytest.raises(ValueError, match=r"unmapped \['hmrc_spi_new_leaf'\]"):
        assert_analogue_map_covers(
            (*SPI_INCOME_QRF_OUTPUT_COLUMNS, "hmrc_spi_new_leaf")
        )
    with pytest.raises(ValueError, match="stale"):
        assert_analogue_map_covers(SPI_INCOME_QRF_OUTPUT_COLUMNS[1:])


def test_observed_analogues_read_the_twin_and_derive_taxable_interest() -> None:
    person = pd.DataFrame(
        {column: [0.0, 5.0] for column in analogue_source_columns()},
        index=[10, 11],
    )
    person["savings_interest_income"] = [100.0, 20.0]
    person["tax_free_savings_income"] = [30.0, 50.0]
    person["dividend_income"] = [7.0, 0.0]
    observed = observed_analogues(person)
    assert list(observed.columns) == list(analogue_targets())
    assert observed.index.tolist() == [10, 11]
    assert observed["savings_interest_income"].tolist() == [70.0, 0.0]
    assert observed["dividend_income"].tolist() == [7.0, 0.0]
    with pytest.raises(ValueError, match="missing column"):
        observed_analogues(person.drop(columns=["miscellaneous_income"]))
    person.loc[10, "property_income"] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        observed_analogues(person)


def _synthetic_frs(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encoded predictors and observed analogues of ``n`` FRS-like adults."""

    rng = np.random.default_rng(seed)
    age = rng.integers(20, 80, n).astype(float)
    has_pay = (rng.random(n) < 0.6).astype(float)
    predictors = pd.DataFrame({"age": age, "has_pay": has_pay})
    observed = pd.DataFrame(index=predictors.index)
    for target in analogue_targets():
        observed[target] = 0.0
    observed["hmrc_spi_pay"] = np.where(has_pay > 0, 15_000.0 + 400.0 * age, 0.0)
    observed["hmrc_spi_pay"] += np.where(has_pay > 0, rng.normal(0, 2_000, n), 0.0)
    observed["self_employment_income"] = np.where(
        rng.random(n) < 0.2, rng.gamma(2.0, 4_000.0, n), 0.0
    )
    observed["savings_interest_income"] = np.where(
        rng.random(n) < 0.5, rng.gamma(1.5, 300.0, n), 0.0
    )
    observed["dividend_income"] = np.where(
        rng.random(n) < 0.15, rng.gamma(2.0, 1_000.0, n), 0.0
    )
    observed["private_pension_income"] = np.where(
        age >= 60, rng.gamma(3.0, 3_000.0, n), 0.0
    )
    observed["property_income"] = np.where(
        rng.random(n) < 0.1, rng.gamma(2.0, 2_500.0, n), 0.0
    )
    observed["hmrc_spi_state_pension_income"] = np.where(age >= 66, 10_000.0, 0.0)
    observed["hmrc_spi_unemployment_benefit_income"] = np.where(
        rng.random(n) < 0.05, 2_000.0, 0.0
    )
    observed["hmrc_spi_other_social_security_income"] = np.where(
        rng.random(n) < 0.05, 1_500.0, 0.0
    )
    observed["hmrc_spi_other_income"] = np.where(rng.random(n) < 0.05, 800.0, 0.0)
    # Incapacity benefit is all zero on the FRS rows: a degenerate analogue.
    return predictors, observed


@pytest.fixture(scope="module")
def reference():
    predictors, observed = _synthetic_frs(400, seed=3)
    donor = pd.concat([predictors, observed], axis=1)
    return fit(
        donor,
        list(predictors.columns),
        list(analogue_targets()),
        weights="none",
        n_estimators=4,
        seed=5,
    )


def test_transport_locates_twins_and_keys_the_rest_on_identity(reference) -> None:
    predictors, observed = _synthetic_frs(120, seed=4)
    predictors.index = observed.index = pd.RangeIndex(500, 620)
    identities = np.arange(9_000, 9_120)
    groups = np.where(predictors["has_pay"].to_numpy() > 0, "pay", "other")
    groups[:5] = "trade"  # a group too small to report
    result = transport_uniforms(
        reference,
        encoded_recipients=predictors,
        observed=observed,
        identities=identities,
        groups=groups,
        seed=42,
    )
    outputs = set(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert set(result.quantiles) == outputs == set(result.sign_uniforms)
    for mapping in (result.quantiles, result.sign_uniforms):
        for values in mapping.values():
            assert values.shape == (120,)
            assert ((values >= 0.0) & (values < 1.0)).all()
    # Leaves without an FRS analogue draw at the identity-keyed uniforms.
    for target in result.receipt["identity_targets"]:
        assert SPI_STAGE1_ANALOGUE_COLUMNS[target] is None
        np.testing.assert_array_equal(
            result.quantiles[target],
            identity_uniforms(identities, target=target, part="quantile", seed=42),
        )
        np.testing.assert_array_equal(
            result.sign_uniforms[target],
            identity_uniforms(identities, target=target, part="sign", seed=42),
        )
    assert "gift_aid" in result.receipt["identity_targets"]
    # The all-zero FRS analogue is degenerate and falls back to identity.
    incapacity = "hmrc_spi_incapacity_benefit_income"
    assert incapacity in result.receipt["degenerate_targets"]
    assert result.receipt["targets"][incapacity]["source"] == (
        "identity_degenerate_analogue"
    )
    assert result.receipt["targets"]["hmrc_spi_pay"]["source"] == "twin_conditional"
    assert result.receipt["targets"]["hmrc_spi_pay"]["reference_regime"] == (
        "zero_inflated_positive"
    )
    # Drawing the reference at the located uniforms gives the twins back: the
    # transport carries each twin's own standing, not a random one.
    located = {target: result.quantiles[target] for target in reference.targets}
    signs = {target: result.sign_uniforms[target] for target in reference.targets}
    drawn = reference.predict_from_uniforms(
        predictors, quantiles=located, sign_uniforms=signs
    )
    # In-sample atoms are exact; held-out values sit between atoms, so the
    # draw at the located quantile is the twin's value where the conditional
    # is strictly increasing and the nearest atom where it is flat.
    pay = observed["hmrc_spi_pay"].to_numpy()
    assert ((drawn["hmrc_spi_pay"].to_numpy() > 0) == (pay > 0)).all()
    assert np.corrcoef(drawn["hmrc_spi_pay"][pay > 0], pay[pay > 0])[0, 1] > 0.95
    # The receipt reports uniformity overall and by group, never on a group
    # under ten rows.
    kolmogorov = result.receipt["targets"]["hmrc_spi_pay"]["kolmogorov"]
    assert 0.0 <= kolmogorov["sign"] <= 1.0 and 0.0 <= kolmogorov["quantile"] <= 1.0
    assert kolmogorov["positive_rows"] == int((pay > 0).sum())
    by_group = result.receipt["targets"]["hmrc_spi_pay"]["by_group"]
    assert by_group["trade"] is None
    assert by_group["pay"]["rows"] == int((groups == "pay").sum())
    assert result.receipt["draw"] == "twin_conditional_quantiles"
    assert result.receipt["recipients_sharing_a_twin"] == 0


def test_transport_is_row_order_invariant_and_shares_twins(reference) -> None:
    predictors, observed = _synthetic_frs(40, seed=6)
    identities = np.arange(100, 140)
    first = transport_uniforms(
        reference,
        encoded_recipients=predictors,
        observed=observed,
        identities=identities,
        groups=None,
        seed=1,
    )
    order = np.random.default_rng(2).permutation(40)
    shuffled = transport_uniforms(
        reference,
        encoded_recipients=predictors.iloc[order],
        observed=observed.iloc[order],
        identities=identities[order],
        groups=None,
        seed=1,
    )
    for target in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(
            first.quantiles[target][order], shuffled.quantiles[target]
        )
        np.testing.assert_array_equal(
            first.sign_uniforms[target][order], shuffled.sign_uniforms[target]
        )
    assert first.receipt["targets"]["hmrc_spi_pay"]["by_group"] is None
    # Two copies of one twin share their uniforms, and the receipt says so.
    doubled = transport_uniforms(
        reference,
        encoded_recipients=pd.concat([predictors, predictors], ignore_index=True),
        observed=pd.concat([observed, observed], ignore_index=True),
        identities=np.r_[identities, identities],
        groups=None,
        seed=1,
    )
    assert doubled.receipt["recipients_sharing_a_twin"] == 40
    np.testing.assert_array_equal(
        doubled.quantiles["hmrc_spi_pay"][:40], doubled.quantiles["hmrc_spi_pay"][40:]
    )


def test_transport_refuses_misaligned_inputs(reference) -> None:
    predictors, observed = _synthetic_frs(12, seed=7)
    identities = np.arange(12)
    with pytest.raises(ValueError, match="one entry per recipient"):
        transport_uniforms(
            reference,
            encoded_recipients=predictors,
            observed=observed,
            identities=identities[:-1],
            groups=None,
            seed=1,
        )
    with pytest.raises(ValueError, match="align to recipients"):
        transport_uniforms(
            reference,
            encoded_recipients=predictors,
            observed=observed.iloc[::-1],
            identities=identities,
            groups=None,
            seed=1,
        )

    class Drifted:
        targets = ("hmrc_spi_pay",)

    with pytest.raises(ValueError, match="targets drifted"):
        transport_uniforms(
            Drifted(),
            encoded_recipients=predictors,
            observed=observed,
            identities=identities,
            groups=None,
            seed=1,
        )
    with pytest.raises(ValueError, match="part must be"):
        identity_uniforms(identities, target="hmrc_spi_pay", part="sign_", seed=1)
    assert transport.SPI_TRANSPORT_SEED_OFFSET == 3


def test_cell_folds_share_a_cell_between_twins_and_copies() -> None:
    from microcosm.build.uk_runtime.spi_rank_transport import cell_fold_ids

    predictors = pd.DataFrame(
        {
            "gender": ["MALE", "MALE", "FEMALE", "FEMALE", "MALE"],
            "region": ["LONDON"] * 5,
            "age": [30.5, 30.5, 40.5, 40.5, 50.5],
            "main_source": ["pay", "pay", "pension", "pension", "trade"],
            "has_pay": [1.0, 1.0, 0.0, 0.0, 0.0],
            "has_trade": [0.0, 0.0, 0.0, 0.0, 1.0],
            "at_or_over_spa": [0.0] * 5,
        }
    )
    folds, used = cell_fold_ids(predictors, folds=5, seed=42)
    assert used == 3  # three distinct cells
    assert folds[0] == folds[1] and folds[2] == folds[3]
    assert ((folds >= 0) & (folds < used)).all()
    # A copy with the twin's predictors lands in the twin's fold, whatever
    # frame it sits in and however many cells that frame has.
    copies, _ = cell_fold_ids(predictors.iloc[[4, 0]], folds=5, seed=42, used=used)
    assert copies.tolist() == [folds[4], folds[0]]
    with pytest.raises(ValueError, match="fold cells need"):
        cell_fold_ids(predictors.drop(columns=["main_source"]), folds=5, seed=42)
    with pytest.raises(ValueError, match="positive integer"):
        cell_fold_ids(predictors, folds=0, seed=42)


def test_cross_fit_locates_a_copy_on_a_forest_that_never_saw_its_twin() -> None:
    """Cross-fitting by cell: the twin is held out, and ranks survive within a cell.

    Rows of one predictor cell share a fold, so a copy is located by a forest
    fitted without its twin; the forest never holds the twin's own value as an
    atom, and within a cell the located quantiles are monotone in the twin's
    value. The fit callable records which rows each forest saw.
    """
    from microcosm.build.uk_runtime.spi_rank_transport import (
        cell_fold_ids,
        cross_fit_transport,
    )

    rng = np.random.default_rng(21)
    n = 600
    ages = rng.integers(25, 60, n).astype(float) + 0.5
    raw = pd.DataFrame(
        {
            "gender": np.where(rng.random(n) < 0.5, "MALE", "FEMALE"),
            "region": "LONDON",
            "age": ages,
            "main_source": "pay",
            "has_pay": 1.0,
            "has_trade": 0.0,
            "at_or_over_spa": 0.0,
        }
    )
    encoded = pd.DataFrame(
        {
            "age": ages,
            "gender_MALE": (raw.gender == "MALE").astype(float),
            "gender_FEMALE": (raw.gender == "FEMALE").astype(float),
        }
    )
    observed = pd.DataFrame(index=encoded.index)
    for target in analogue_targets():
        observed[target] = 0.0
    observed["hmrc_spi_pay"] = 12_000.0 + 500.0 * (ages - 25) + rng.normal(0, 4_000, n)
    observed["hmrc_spi_pay"] = observed["hmrc_spi_pay"].clip(lower=1_000.0)
    # Pay is chained on the earlier leaves, so they stay at zero here: the
    # monotonicity below is within a cell *and* the same earlier leaves.
    seen: list[set[int]] = []

    def fit_reference(encoded_train, observed_train):
        seen.append(set(encoded_train.index.tolist()))
        donor = pd.concat([encoded_train, observed_train], axis=1)
        return fit(
            donor,
            list(encoded_train.columns),
            list(analogue_targets()),
            weights="none",
            n_estimators=30,
            seed=3,
        )

    folds, used = cell_fold_ids(raw, folds=5, seed=42)
    assert used == 5
    # The recipients are copies of the first 200 rows: same cells, same values.
    copies = np.arange(200)
    training_identities = np.arange(10_000, 10_000 + len(raw))
    result = cross_fit_transport(
        fit_reference,
        training_predictors=raw,
        encoded_training=encoded,
        training_observed=observed,
        training_folds=folds,
        training_identities=training_identities,
        used=used,
        recipient_predictors=raw.iloc[copies].reset_index(drop=True),
        encoded_recipients=encoded.iloc[copies].reset_index(drop=True),
        observed=observed.iloc[copies].reset_index(drop=True),
        identities=np.arange(10_000, 10_200),
        groups=None,
        seed=42,
        folds=5,
    )
    receipt = result.receipt
    assert receipt["folds"] == 5 and receipt["cross_fitted"] is True
    assert receipt["in_sample_folds"] == []
    assert sum(r["scored_recipients"] for r in receipt["fold_receipts"].values()) == 200
    # Every forest was fitted without the rows it located.
    for fold, rows_seen in enumerate(seen):
        located = copies[folds[copies] == fold]
        assert not (set(located.tolist()) & rows_seen)
        assert len(rows_seen) == int((folds != fold).sum())
    # Within a cell (same whole-year age and gender, and here the same
    # earlier leaves) the located pay quantiles are monotone in the twin's
    # pay: one forest locates the whole cell and holds no twin's own value.
    pay = observed["hmrc_spi_pay"].to_numpy()[copies]
    located = result.quantiles["hmrc_spi_pay"]
    cells = pd.DataFrame(
        {
            "cell": (
                np.floor(ages[copies]).astype(int).astype(str)
                + raw.gender.to_numpy()[copies]
            ),
            "pay": pay,
            "u": located,
        }
    )
    # A value beyond the ends of its conditional is placed inside the end
    # interval by the placement uniform, a tie-break rather than a rank, so
    # the check runs over the rows the conditional covers.
    interior = (cells.u > 1e-4) & (cells.u < 1 - 1e-4)
    assert interior.mean() > 0.8
    checked = 0
    for _, group in cells[interior].groupby("cell"):
        if len(group) >= 3:
            order = np.argsort(group.pay.to_numpy())
            assert (np.diff(group.u.to_numpy()[order]) >= 0).all()
            checked += 1
    assert checked > 0
    assert receipt["targets"]["hmrc_spi_pay"]["source"] == "twin_conditional"
    assert receipt["targets"]["hmrc_spi_pay"]["reference_regime"] == ["positive_only"]
    # The folds agree on the sign classes, which the receipt reports for the
    # tape-side check.
    assert receipt["targets"]["hmrc_spi_pay"]["reference_classes"] == [1]
    assert receipt["targets"]["gift_aid"]["reference_classes"] is None
    # A copy whose fold-key cell moved away from its twin's is refused: the
    # forest that located it could have seen the twin.
    moved = raw.iloc[copies].reset_index(drop=True)
    moved["age"] = moved["age"] + 7.0
    with pytest.raises(ValueError, match="differs from its twin's"):
        cross_fit_transport(
            fit_reference,
            training_predictors=raw,
            encoded_training=encoded,
            training_observed=observed,
            training_folds=folds,
            training_identities=training_identities,
            used=used,
            recipient_predictors=moved,
            encoded_recipients=encoded.iloc[copies].reset_index(drop=True),
            observed=observed.iloc[copies].reset_index(drop=True),
            identities=np.arange(10_000, 10_200),
            groups=None,
            seed=42,
            folds=5,
        )
    # A frame of one cell cannot hold anything out: refused unless the caller
    # is a sampled rung or a synthetic fixture, and then the receipt says so.
    one_cell = raw.iloc[:10].assign(age=30.5, gender="MALE")
    one_encoded = encoded.iloc[:10].assign(age=30.5, gender_MALE=1.0, gender_FEMALE=0.0)
    folds_one, used_one = cell_fold_ids(one_cell, folds=5, seed=42)
    assert used_one == 1
    one_cell_call = dict(
        training_predictors=one_cell,
        encoded_training=one_encoded,
        training_observed=observed.iloc[:10],
        training_folds=folds_one,
        training_identities=training_identities[:10],
        used=used_one,
        recipient_predictors=one_cell.reset_index(drop=True),
        encoded_recipients=one_encoded.reset_index(drop=True),
        observed=observed.iloc[:10].reset_index(drop=True),
        identities=np.arange(10_000, 10_010),
        groups=None,
        seed=42,
        folds=5,
    )
    with pytest.raises(ValueError, match="cannot be cross-fitted"):
        cross_fit_transport(fit_reference, **one_cell_call)
    single = cross_fit_transport(fit_reference, allow_in_sample=True, **one_cell_call)
    assert single.receipt["folds"] == 1 and single.receipt["cross_fitted"] is False
    # A positive-row count under the group floor is withheld.
    assert single.receipt["targets"]["dividend_income"]["kolmogorov"][
        "positive_rows"
    ] in (
        None,
        *range(10, 11),
    )


class _ClassReference:
    """A fitted reference whose sign classes depend on the fold it was fitted on."""

    def __init__(self, classes_by_call: list[tuple[int, ...]], targets) -> None:
        self.targets = tuple(targets)
        self._classes = classes_by_call
        self._fits = 0

    def regimes(self) -> dict[str, str]:
        return dict.fromkeys(self.targets, "zero_inflated_positive")

    def sign_classes(self, target: str) -> tuple[int, ...]:
        return self._classes[min(self._fits - 1, len(self._classes) - 1)]

    def conditional_uniforms(self, frame, *, sign_placement, quantile_placement):
        from microcosm.fit import ConditionalUniforms

        return ConditionalUniforms(
            quantiles={t: np.asarray(quantile_placement[t]) for t in self.targets},
            sign_uniforms={t: np.asarray(sign_placement[t]) for t in self.targets},
            degenerate=frozenset(),
            ungated=frozenset(),
        )


def test_cross_fit_refuses_folds_that_disagree_on_sign_classes() -> None:
    """One forest per fold locates a cell; folds whose gates order different
    sign classes would read the same uniform two ways, so they are refused."""
    from microcosm.build.uk_runtime.spi_rank_transport import (
        cell_fold_ids,
        cross_fit_transport,
    )

    rng = np.random.default_rng(5)
    n = 400
    ages = rng.integers(25, 60, n).astype(float) + 0.5
    raw = pd.DataFrame(
        {
            "gender": np.where(rng.random(n) < 0.5, "MALE", "FEMALE"),
            "region": "LONDON",
            "age": ages,
            "main_source": "pay",
            "has_pay": 1.0,
            "has_trade": 0.0,
            "at_or_over_spa": 0.0,
        }
    )
    encoded = pd.DataFrame(
        {
            "age": ages,
            "gender_MALE": (raw.gender == "MALE").astype(float),
            "gender_FEMALE": (raw.gender == "FEMALE").astype(float),
        }
    )
    observed = pd.DataFrame(index=encoded.index)
    for target in analogue_targets():
        observed[target] = 1_000.0
    folds, used = cell_fold_ids(raw, folds=5, seed=42)
    assert used == 5
    reference = _ClassReference([(0, 1), (1,)], analogue_targets())

    def fit_reference(encoded_train, observed_train):
        reference._fits += 1
        return reference

    copies = np.arange(100)
    with pytest.raises(ValueError, match="disagree on the sign classes"):
        cross_fit_transport(
            fit_reference,
            training_predictors=raw,
            encoded_training=encoded,
            training_observed=observed,
            training_folds=folds,
            training_identities=np.arange(1, n + 1),
            used=used,
            recipient_predictors=raw.iloc[copies].reset_index(drop=True),
            encoded_recipients=encoded.iloc[copies].reset_index(drop=True),
            observed=observed.iloc[copies].reset_index(drop=True),
            identities=np.arange(1, 101),
            groups=None,
            seed=42,
            folds=5,
        )
