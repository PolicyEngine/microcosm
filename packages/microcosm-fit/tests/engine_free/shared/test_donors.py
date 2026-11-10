"""Keyed nearest-donor choice: cells, fallbacks, weights and the caller's uniforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import donor_distance_features, keyed_nearest_donors


def _donors(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cell": rng.choice(["a", "b"], n),
            "age": rng.integers(18, 80, n).astype(float),
            "pay": np.where(rng.random(n) < 0.3, 0.0, rng.lognormal(10, 1, n)),
            "area": rng.choice(["x", "y", "z"], n),
            "weight": rng.uniform(0.5, 2.0, n),
        }
    )


def _features(recipients: pd.DataFrame, donors: pd.DataFrame):
    return donor_distance_features(
        recipients,
        donors,
        rank_columns=["age"],
        presence_rank_columns=["pay"],
        scaled_categoricals={"area": 0.3},
    )


def test_features_rank_against_the_donors() -> None:
    donors = pd.DataFrame(
        {
            "age": [20.0, 40.0, 60.0, 80.0],
            "pay": [0.0, 1.0, 2.0, 3.0],
            "area": list("xxyy"),
        }
    )
    recipients = pd.DataFrame({"age": [40.0], "pay": [0.0], "area": ["y"]})
    r, d = _features(recipients, donors)
    # age rank 2/4; pay absent; area one-hot (x, y) times 0.3.
    assert r.tolist() == [[0.5, 0.0, 0.0, 0.0, 0.3]]
    assert d.shape == (4, 5)


def test_each_recipient_draws_from_its_own_cell() -> None:
    donors = _donors()
    recipients = _donors(100, seed=1)
    r, d = _features(recipients, donors)
    positions, levels = keyed_nearest_donors(
        r,
        d,
        recipient_cells=[recipients["cell"].to_numpy(), np.zeros(len(recipients))],
        donor_cells=[donors["cell"].to_numpy(), np.zeros(len(donors))],
        donor_weights=donors["weight"].to_numpy(),
        uniforms=np.random.default_rng(2).random(len(recipients)),
        k=20,
    )
    assert (levels == 0).all()
    assert (donors["cell"].to_numpy()[positions] == recipients["cell"].to_numpy()).all()


def test_a_thin_cell_falls_back_to_the_coarser_level() -> None:
    donors = _donors()
    donors.loc[donors.index[:3], "cell"] = "thin"
    recipients = _donors(10, seed=3).assign(cell="thin")
    r, d = _features(recipients, donors)
    positions, levels = keyed_nearest_donors(
        r,
        d,
        recipient_cells=[recipients["cell"].to_numpy(), np.zeros(10)],
        donor_cells=[donors["cell"].to_numpy(), np.zeros(len(donors))],
        donor_weights=donors["weight"].to_numpy(),
        uniforms=np.full(10, 0.5),
        k=20,
    )
    assert (levels == 1).all()
    assert (positions >= 0).all()


def test_a_recipients_donor_depends_on_its_own_uniform_only() -> None:
    donors = _donors()
    recipients = _donors(60, seed=4)
    u = np.random.default_rng(5).random(60)
    r, d = _features(recipients, donors)
    cells = ([recipients["cell"].to_numpy()], [donors["cell"].to_numpy()])
    positions, _ = keyed_nearest_donors(
        r,
        d,
        recipient_cells=cells[0],
        donor_cells=cells[1],
        donor_weights=donors["weight"].to_numpy(),
        uniforms=u,
        k=10,
    )
    order = np.random.default_rng(6).permutation(60)
    shuffled, _ = keyed_nearest_donors(
        r[order],
        d,
        recipient_cells=[cells[0][0][order]],
        donor_cells=cells[1],
        donor_weights=donors["weight"].to_numpy(),
        uniforms=u[order],
        k=10,
    )
    assert np.array_equal(shuffled, positions[order])


def test_choice_follows_the_donor_weights() -> None:
    donors = pd.DataFrame({"age": [30.0, 30.0], "weight": [1.0, 3.0]})
    recipients = pd.DataFrame({"age": np.full(4000, 30.0)})
    r, d = donor_distance_features(recipients, donors, rank_columns=["age"])
    positions, _ = keyed_nearest_donors(
        r,
        d,
        recipient_cells=[np.zeros(4000)],
        donor_cells=[np.zeros(2)],
        donor_weights=donors["weight"].to_numpy(),
        uniforms=np.random.default_rng(7).random(4000),
        k=2,
    )
    assert (positions == 1).mean() == pytest.approx(0.75, abs=0.03)


def test_no_donor_at_any_level_is_refused() -> None:
    r = np.zeros((1, 1))
    with pytest.raises(ValueError, match="no donor"):
        keyed_nearest_donors(
            r,
            np.zeros((2, 1)),
            recipient_cells=[np.array(["a"])],
            donor_cells=[np.array(["b", "b"])],
            donor_weights=np.ones(2),
            uniforms=np.array([0.5]),
            k=1,
        )


def test_bad_uniforms_are_refused() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        keyed_nearest_donors(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            recipient_cells=[np.zeros(1)],
            donor_cells=[np.zeros(1)],
            donor_weights=np.ones(1),
            uniforms=np.array([1.0]),
            k=1,
        )
