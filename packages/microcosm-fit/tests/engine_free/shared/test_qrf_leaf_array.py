"""The in-place leaf array: byte-identical to quantile-forest's own.

After growing its trees, quantile-forest 1.4.2 maps every leaf to the training
rows it holds in one zero-padded int64 array of shape (trees x nodes x outputs
x width). The library builds one array per tree and then copies the list into
the final array with ``np.array(list)``; the copy writes every byte, padding
included, so all of it becomes resident memory, and the two copies coexist.
``microcosm.fit.qrf._InPlaceLeafQRF`` allocates the final array once and fills
each tree's slice directly.

These tests pin that the subclass's array equals the library's byte for byte
(dtype, shape, C order and ``tobytes()``), that it leaves Python's global
``random`` state where the library leaves it, that every prediction and draw is
the same, and that the override refuses any quantile-forest version other than
the one it mirrors.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import math
import os
import pickle
import random
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import quantile_forest
import sklearn
from quantile_forest import RandomForestQuantileRegressor
from quantile_forest import _quantile_forest as quantile_forest_module
from sklearn.base import clone
from sklearn.utils.fixes import parse_version

from microcosm.fit import fit, qrf
from test_support.paths import paths_for

# ----------------------------------------------------------------------------
# Synthetic donors
# ----------------------------------------------------------------------------


def _continuous_donors(seed: int, rows: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Positive continuous target over coarse predictors.

    The predictors take a few integer values each, so many donor rows share a
    predictor vector; a tree cannot split them apart, and leaves hold several
    distinct target values. That makes the uncapped width larger than one.
    """
    generator = np.random.default_rng(seed)
    predictors = generator.integers(0, 8, size=(rows, 3)).astype(np.float64)
    target = np.exp(generator.normal(2.5, 0.6, rows)) + predictors[:, 0]
    return predictors, target


def _near_discrete_donors(
    seed: int, rows: int = 3000, predictor_levels: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Target with four distinct values, dominated by zero.

    Two predictors with ``predictor_levels`` values each give at most
    ``predictor_levels**2`` distinct predictor vectors, so leaves hold tens to hundreds
    of rows: far more than the small caps the tests set.
    """
    generator = np.random.default_rng(seed)
    predictors = generator.integers(0, predictor_levels, size=(rows, 2)).astype(
        np.float64
    )
    target = generator.choice(
        [0.0, 0.25, 0.5, 1.0], size=rows, p=[0.85, 0.05, 0.05, 0.05]
    )
    return predictors, target


def _sample_weights(seed: int, rows: int) -> np.ndarray:
    """Positive weights with a fifth of the rows at exactly zero.

    The library's weight filter runs on every leaf. With a fifth of the
    weights at zero no leaf empties;
    ``test_weighted_leaves_that_empty_get_the_library_placeholder`` covers that.
    """
    generator = np.random.default_rng(seed)
    weights = generator.uniform(0.1, 3.0, rows)
    weights[generator.random(rows) < 0.2] = 0.0
    return weights


# ----------------------------------------------------------------------------
# Capturing the leaf array each class builds
# ----------------------------------------------------------------------------


def _fit_capturing_leaf_array(
    model_class: type[RandomForestQuantileRegressor],
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    **parameters,
) -> tuple[RandomForestQuantileRegressor, np.ndarray, object]:
    """Fit ``model_class`` and return the array its ``_get_y_train_leaves`` built.

    Also returns Python's global ``random`` state after the fit. The global
    state is seeded to the same value before every fit, so two fits that make
    the same ``random.seed``/``random.shuffle`` calls end in the same state.
    """
    captured: dict[str, np.ndarray] = {}

    class _Capturing(model_class):
        def _get_y_train_leaves(self, *args, **kwargs):
            leaves = super()._get_y_train_leaves(*args, **kwargs)
            captured["leaves"] = leaves
            return leaves

    random.seed(20261117)
    model = _Capturing(**parameters)
    model.fit(predictors, target, sample_weight=sample_weight)
    return model, captured["leaves"], random.getstate()


def _assert_byte_identical(in_place: np.ndarray, library: np.ndarray) -> None:
    assert isinstance(in_place, np.ndarray)
    assert in_place.dtype == library.dtype == np.int64
    assert in_place.shape == library.shape
    assert in_place.flags.c_contiguous and library.flags.c_contiguous
    assert in_place.tobytes() == library.tobytes()


def _assert_same_fit(
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    **parameters,
) -> np.ndarray:
    """Fit both classes on the same inputs and compare everything they store."""
    parameters = {"n_estimators": 6, "random_state": 11, "n_jobs": 1, **parameters}
    library_model, library_leaves, library_random_state = _fit_capturing_leaf_array(
        RandomForestQuantileRegressor,
        predictors,
        target,
        sample_weight=sample_weight,
        **parameters,
    )
    in_place_model, in_place_leaves, in_place_random_state = _fit_capturing_leaf_array(
        qrf._InPlaceLeafQRF,
        predictors,
        target,
        sample_weight=sample_weight,
        **parameters,
    )
    _assert_byte_identical(in_place_leaves, library_leaves)
    # The forest object keeps the array without copying it.
    assert np.shares_memory(
        np.asarray(in_place_model.forest_.y_train_leaves), in_place_leaves
    )
    _assert_byte_identical(
        np.asarray(in_place_model.forest_.y_train_leaves),
        np.asarray(library_model.forest_.y_train_leaves),
    )
    assert (
        np.asarray(in_place_model.forest_.y_train).tobytes()
        == np.asarray(library_model.forest_.y_train).tobytes()
    )
    assert in_place_random_state == library_random_state
    quantiles = list(qrf._QUANTILE_GRID)
    in_place_predictions = np.asarray(
        in_place_model.predict(predictors[:200], quantiles=quantiles)
    )
    library_predictions = np.asarray(
        library_model.predict(predictors[:200], quantiles=quantiles)
    )
    assert in_place_predictions.tobytes() == library_predictions.tobytes()
    return in_place_leaves


# ----------------------------------------------------------------------------
# Byte identity of the leaf array
# ----------------------------------------------------------------------------


def test_continuous_target_without_a_cap_matches_the_library() -> None:
    predictors, target = _continuous_donors(seed=1)
    leaves = _assert_same_fit(predictors, target, max_samples_leaf=None)
    # The uncapped width is the widest leaf, so it exceeds one here.
    assert leaves.shape[-1] > 1


@pytest.mark.parametrize("cap", [16, np.int64(16), 0.0047])
def test_near_discrete_target_over_the_cap_matches_the_library(
    monkeypatch, cap
) -> None:
    """Leaves larger than the cap are shuffled and truncated identically.

    The fractional cap is 0.0047 of 3,000 rows, 14.1, which the library rounds
    up to 15. A numpy integer cap counts as an integer, as in the library.
    """
    shuffle_calls: list[int] = []
    original_shuffle = random.shuffle

    def counting_shuffle(sequence):
        shuffle_calls.append(len(sequence))
        return original_shuffle(sequence)

    monkeypatch.setattr(random, "shuffle", counting_shuffle)
    predictors, target = _near_discrete_donors(seed=2)
    leaves = _assert_same_fit(predictors, target, max_samples_leaf=cap)
    integer_cap = isinstance(cap, int | np.integer)
    width = cap if integer_cap else math.ceil(cap * len(target))
    assert leaves.shape[-1] == width == (16 if integer_cap else 15)
    # Both fits shuffled the same leaves, each larger than the cap, before
    # truncating them; the library fit ran first.
    half = len(shuffle_calls) // 2
    assert half > 0
    assert shuffle_calls[:half] == shuffle_calls[half:]
    assert min(shuffle_calls) > width


@pytest.mark.parametrize("max_samples_leaf", [None, 5, 1])
def test_several_outputs_match_the_library(max_samples_leaf) -> None:
    predictors, first = _continuous_donors(seed=3)
    generator = np.random.default_rng(3)
    target = np.column_stack(
        [first, first * generator.uniform(0.5, 2.0, len(first)), predictors[:, 1]]
    )
    leaves = _assert_same_fit(predictors, target, max_samples_leaf=max_samples_leaf)
    assert leaves.shape[2] == 3


@pytest.mark.parametrize("max_samples_leaf", [None, 7, 1])
def test_weighted_samples_match_the_library(max_samples_leaf) -> None:
    predictors, target = _near_discrete_donors(seed=4, rows=800, predictor_levels=6)
    target = target + predictors[:, 0]
    weights = _sample_weights(seed=4, rows=len(target))
    _assert_same_fit(
        predictors,
        target,
        sample_weight=weights,
        max_samples_leaf=max_samples_leaf,
    )


@pytest.mark.parametrize("max_samples_leaf", [None, 3, 1])
def test_weighted_leaves_that_empty_get_the_library_placeholder(
    monkeypatch, max_samples_leaf
) -> None:
    """A leaf whose entries the weight filter removes holds ``[0]``.

    Nine in ten weights are zero, so the filter, which reads another row's
    weight for each entry, empties some leaves. Both classes hand the
    filtered leaves to quantile-forest's compiled helper, which records how
    many hold the placeholder. With a cap of one entry the helper is not
    used, and only the arrays are compared.
    """
    placeholders: list[int] = []
    original_map = quantile_forest_module.map_indices_to_leaves

    def recording_map(**arguments):
        placeholders.append(
            sum(
                isinstance(values, list) and values == [0]
                for values in arguments["leaf_values_list"]
            )
        )
        return original_map(**arguments)

    monkeypatch.setattr(quantile_forest_module, "map_indices_to_leaves", recording_map)
    predictors, target = _continuous_donors(seed=15, rows=500)
    generator = np.random.default_rng(15)
    weights = generator.uniform(0.1, 3.0, len(target))
    weights[generator.random(len(target)) < 0.9] = 0.0
    _assert_same_fit(
        predictors,
        target,
        sample_weight=weights,
        max_samples_leaf=max_samples_leaf,
    )
    if max_samples_leaf == 1:
        assert placeholders == []
        return
    # Six trees per class, each handed to the helper once.
    half = len(placeholders) // 2
    assert half == 6
    assert placeholders[:half] == placeholders[half:]
    assert sum(placeholders[half:]) > 0


@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("max_samples_leaf", [None, 5])
def test_weight_filter_matches_the_library_for_tiny_zero_and_negative_weights(
    max_samples_leaf, bootstrap
) -> None:
    """quantile-forest 1.4.2's weight filter is copied as it is.

    It looks up each entry's weight by the draw's position minus one, not by
    the drawn row, so it keeps or drops entries by another row's weight. Tiny,
    zero and negative weights all reach the comparison it makes. From
    scikit-learn 1.9 the forest's weighted bootstrap draw refuses negative
    weights, so with bootstrapping on they are left out there; with it off
    they reach the filter on every version.
    """
    predictors, target = _continuous_donors(seed=18, rows=600)
    generator = np.random.default_rng(18)
    levels = [-1.0, 0.0, 1e-3, 0.04, 0.5, 2.0, 3.0]
    if bootstrap and parse_version(sklearn.__version__) >= parse_version("1.9.0"):
        levels.remove(-1.0)
    weights = generator.choice(levels, len(target))
    _assert_same_fit(
        predictors,
        target,
        sample_weight=weights,
        max_samples_leaf=max_samples_leaf,
        bootstrap=bootstrap,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({}, id="library-default-one-sample-per-leaf"),
        # 0.021 of 400 rows is 8.4, which the library rounds up to 9.
        pytest.param({"max_samples_leaf": 0.021}, id="fractional-cap"),
        pytest.param({"max_samples_leaf": None, "bootstrap": False}, id="no-bootstrap"),
        pytest.param({"max_samples_leaf": 9, "max_samples": 0.5}, id="half-bootstrap"),
        pytest.param(
            {"max_samples_leaf": None, "max_samples": 150}, id="row-bootstrap"
        ),
    ],
)
def test_other_forest_settings_match_the_library(parameters) -> None:
    """Other forest settings still give the library's array.

    ``_fit_forest`` passes a fractional cap when a chain configuration asks
    for one; the bootstrap settings are ones microcosm leaves at the default.
    """
    predictors, target = _continuous_donors(seed=5, rows=400)
    _assert_same_fit(predictors, target, **parameters)


def test_parallel_tree_fitting_matches_the_library() -> None:
    predictors, target = _continuous_donors(seed=6)
    _assert_same_fit(predictors, target, max_samples_leaf=None, n_jobs=-1)


def test_fill_refuses_a_helper_that_returns_a_copy(monkeypatch) -> None:
    """If the compiled helper returned a copy, fitting fails.

    The fill checks that the array the helper returns shares memory with the
    slice it was given. A copy would leave the preallocated slice at zero, so
    every leaf would read as empty.
    """
    original_map = quantile_forest_module.map_indices_to_leaves

    def copying_map(**arguments):
        arguments["y_train_leaves_slice"] = arguments["y_train_leaves_slice"].copy()
        return original_map(**arguments)

    monkeypatch.setattr(quantile_forest_module, "map_indices_to_leaves", copying_map)
    predictors, target = _continuous_donors(seed=16, rows=100)
    model = qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=0)
    with pytest.raises(RuntimeError, match="returned a copy"):
        model.fit(predictors, target)


def test_forest_keeps_the_array_the_fill_wrote(monkeypatch) -> None:
    """The stored leaf array is the one preallocated buffer, not a copy of it.

    A copy would hold the same bytes but make every padding page resident,
    which is the memory this subclass exists to save.
    """
    written: list[np.ndarray] = []
    original_fill = qrf._fill_tree_leaves

    def recording_fill(tree_leaves, *arguments):
        written.append(tree_leaves)
        return original_fill(tree_leaves, *arguments)

    monkeypatch.setattr(qrf, "_fill_tree_leaves", recording_fill)
    predictors, target = _continuous_donors(seed=19, rows=300)
    model = qrf._InPlaceLeafQRF(n_estimators=4, max_samples_leaf=None, random_state=1)
    model.fit(predictors, target)
    stored = np.asarray(model.forest_.y_train_leaves)
    assert len(written) == 4
    for tree, tree_leaves in enumerate(written):
        assert np.shares_memory(stored[tree], tree_leaves)
    # One buffer for the whole forest: each slice is a view of the same base.
    assert {id(tree_leaves.base) for tree_leaves in written} == {id(written[0].base)}


def test_fit_returns_the_model_and_keeps_the_default_pickle_format() -> None:
    predictors, target = _continuous_donors(seed=20, rows=150)
    model = qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=4)
    assert model.fit(predictors, target) is model
    assert not model.forest_.sparse_pickle


def test_dataframe_predictors_give_the_library_warnings_and_array() -> None:
    """quantile-forest silences ``apply``'s feature-name warning; so does the copy."""
    predictors, target = _continuous_donors(seed=21, rows=300)
    frame = pd.DataFrame(predictors, columns=["first", "second", "third"])
    warning_texts = {}
    leaves = {}
    for name, model_class in (
        ("library", RandomForestQuantileRegressor),
        ("in_place", qrf._InPlaceLeafQRF),
    ):
        model = model_class(n_estimators=3, max_samples_leaf=None, random_state=6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(frame, target)
        warning_texts[name] = [
            (type(warning.message), str(warning.message)) for warning in caught
        ]
        leaves[name] = np.asarray(model.forest_.y_train_leaves)
    assert warning_texts["in_place"] == warning_texts["library"]
    _assert_byte_identical(leaves["in_place"], leaves["library"])


def test_scikit_learn_1_9_bootstrap_arguments_match_the_library(monkeypatch) -> None:
    """Under scikit-learn 1.9 both classes pass the helpers ``sample_weight_``.

    scikit-learn 1.9 adds a required ``sample_weight`` argument to its private
    bootstrap helpers, and quantile-forest 1.4.2 passes the fit's
    ``sample_weight_`` when it finds 1.9 or later. The test reports 1.9 and
    wraps the helpers to record whether they received that exact array. Under
    an older scikit-learn the wrappers then drop it before calling the
    installed helpers; under 1.9 or later they pass it on.
    """
    recorded: list[tuple[str, bool]] = []
    original_count = quantile_forest_module._get_n_samples_bootstrap
    original_indices = quantile_forest_module._generate_sample_indices
    installed_takes_weights = parse_version(sklearn.__version__) >= parse_version(
        "1.9.0"
    )
    predictors, target = _continuous_donors(seed=22, rows=300)
    weights = np.random.default_rng(22).uniform(0.5, 2.0, len(target))

    def count_with_weights(n_samples, max_samples, sample_weight):
        recorded.append(("count", sample_weight is weights))
        if installed_takes_weights:
            return original_count(n_samples, max_samples, sample_weight)
        return original_count(n_samples, max_samples)

    def indices_with_weights(
        random_state, n_samples, n_samples_bootstrap, sample_weight
    ):
        recorded.append(("indices", sample_weight is weights))
        if installed_takes_weights:
            return original_indices(
                random_state, n_samples, n_samples_bootstrap, sample_weight
            )
        return original_indices(random_state, n_samples, n_samples_bootstrap)

    monkeypatch.setattr(
        quantile_forest_module, "sklearn_version", parse_version("1.9.0")
    )
    monkeypatch.setattr(
        quantile_forest_module, "_get_n_samples_bootstrap", count_with_weights
    )
    monkeypatch.setattr(
        quantile_forest_module, "_generate_sample_indices", indices_with_weights
    )
    _assert_same_fit(predictors, target, sample_weight=weights, max_samples_leaf=None)
    # One count call, then one index call per tree (six trees), for each class.
    expected = [("count", True)] + [("indices", True)] * 6
    assert recorded == expected + expected


def test_fit_leaves_the_warning_filters_as_they_were() -> None:
    """The copy silences ``apply``'s warning only inside ``catch_warnings``.

    With context-aware warnings (the default on free-threaded builds) the
    active filters live in a context variable, which ``_get_filters`` reads.
    """
    active_filters = getattr(warnings, "_get_filters", lambda: warnings.filters)
    predictors, target = _continuous_donors(seed=24, rows=150)
    before = list(active_filters())
    qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=5).fit(
        predictors, target
    )
    assert list(active_filters()) == before


def _fit_outcome(model_class, predictors, target, weights) -> tuple[object, object]:
    """Fit and return the leaf array's bytes, or the exception it raised."""
    model = model_class(n_estimators=3, max_samples_leaf=None, random_state=0)
    try:
        model.fit(predictors, target, sample_weight=weights)
    except Exception as error:  # the exception itself is the outcome compared
        return type(error), str(error).replace(model_class.__name__, "<forest>")
    return "fitted", np.asarray(model.forest_.y_train_leaves).tobytes()


@pytest.mark.parametrize("case", ["several-outputs", "one-row"])
def test_weighted_fits_the_library_refuses_are_refused_alike(case) -> None:
    """quantile-forest 1.4.2 fails on these weighted fits; so does the copy."""
    predictors, first = _continuous_donors(seed=25, rows=200)
    if case == "several-outputs":
        target = np.column_stack([first, first * 2.0])
    else:
        predictors, target = predictors[:1], first[:1]
    weights = np.linspace(0.5, 2.0, len(predictors))
    library = _fit_outcome(RandomForestQuantileRegressor, predictors, target, weights)
    in_place = _fit_outcome(qrf._InPlaceLeafQRF, predictors, target, weights)
    assert library[0] is IndexError
    assert in_place == library


#: Run in a fresh Python process, so the forest's array gets pages no earlier
#: allocation in the test session has written. A large allocation can reuse
#: memory freed earlier in the process, which ``calloc`` then zeroes, writing
#: every page. Prints the share of the array's pages that are mapped, or
#: "unavailable" when ``mincore`` cannot be called.
_RESIDENT_SHARE_CHILD = """
import ctypes
import ctypes.util
import os

import numpy as np
from numpy._core import multiarray

from microcosm.fit import qrf

# A transparent huge page makes 2 MiB resident at the first write.
multiarray._set_madvise_hugepage(False)
generator = np.random.default_rng(26)
predictors = generator.integers(0, 12, size=(3000, 2)).astype(np.float64)
target = generator.choice([0.0, 0.25, 0.5, 1.0], size=3000, p=[0.85, 0.05, 0.05, 0.05])
model = qrf._InPlaceLeafQRF(
    n_estimators=8, max_samples_leaf=4096, random_state=3, n_jobs=1
)
model.fit(predictors, target)
leaves = np.asarray(model.forest_.y_train_leaves)
assert leaves.nbytes > 40 * 2**20, leaves.nbytes
library_path = ctypes.util.find_library("c")
if library_path is None:
    print("unavailable")
    raise SystemExit(0)
libc = ctypes.CDLL(library_path, use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.mincore.restype = ctypes.c_int
page = os.sysconf("SC_PAGE_SIZE")
start = leaves.ctypes.data
aligned = start - start % page
length = start + leaves.nbytes - aligned
pages = -(-length // page)
status = (ctypes.c_ubyte * pages)()
if libc.mincore(aligned, length, status) != 0:
    print("unavailable")
    raise SystemExit(0)
print(float(np.mean(np.frombuffer(status, dtype=np.uint8) & 1)))
"""


def test_in_place_fill_leaves_unwritten_pages_unmapped() -> None:
    """Most pages of a capped forest's padded array are never touched.

    Each node's row is 4,096 entries (32 KiB) wide, and a leaf's few entries
    sit at the start of its row, so rows of internal nodes and most of each
    leaf row are never written, and ``mincore`` reports those pages as not in
    memory. The library's ``np.array(list)`` copy writes them all (a share of
    1.0). The array, over 40 MiB, is larger than glibc's largest mmap
    threshold, so it gets fresh pages; numpy's request for transparent huge
    pages is turned off in the child, and glibc's ``MALLOC_PERTURB_`` and
    ``GLIBC_TUNABLES`` (which can fill allocations or request huge pages) are
    unset. About a quarter of the pages hold entries
    with 16 KiB pages and a sixteenth with 4 KiB pages. With pages over
    16 KiB, or huge pages always on at any size, most pages hold an entry,
    so the test skips.
    """
    if not hasattr(os, "sysconf"):
        pytest.skip("os.sysconf is unavailable")
    if os.sysconf("SC_PAGE_SIZE") > 16 * 2**10:
        pytest.skip("memory pages are larger than 16 KiB")
    huge_page_settings = Path("/sys/kernel/mm/transparent_hugepage")
    for setting in [
        huge_page_settings / "enabled",
        *huge_page_settings.glob("hugepages-*/enabled"),
    ]:
        if setting.exists() and "[always]" in setting.read_text():
            pytest.skip(f"transparent huge pages are always on ({setting})")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"MALLOC_PERTURB_", "GLIBC_TUNABLES"}
    }
    completed = subprocess.run(
        [sys.executable, "-c", _RESIDENT_SHARE_CHILD],
        capture_output=True,
        text=True,
        check=False,
        env={**environment, "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    reading = completed.stdout.strip().splitlines()[-1]
    if reading == "unavailable":
        pytest.skip("mincore is unavailable")
    assert float(reading) < 0.6


def test_sparse_pickle_reaches_the_library() -> None:
    predictors, target = _continuous_donors(seed=17, rows=200)
    model = qrf._InPlaceLeafQRF(n_estimators=3, max_samples_leaf=None, random_state=2)
    model.fit(predictors, target, sparse_pickle=True)
    assert model.forest_.sparse_pickle
    restored = pickle.loads(pickle.dumps(model))
    _assert_byte_identical(
        np.asarray(restored.forest_.y_train_leaves),
        np.asarray(model.forest_.y_train_leaves),
    )


# ----------------------------------------------------------------------------
# microcosm's forest construction uses the subclass
# ----------------------------------------------------------------------------


def _fit_forest_with(
    model_class: type[RandomForestQuantileRegressor],
    predictors: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    *,
    seed: int,
    n_estimators: int,
    max_samples_leaf: int | float | None,
) -> qrf._Forest:
    """Run microcosm's ``_fit_forest`` with ``model_class`` as its forest."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qrf, "_InPlaceLeafQRF", model_class)
        return qrf._fit_forest(
            predictors,
            target,
            tuple(f"x{index}" for index in range(predictors.shape[1])),
            weights,
            seed=seed,
            n_estimators=n_estimators,
            max_samples_leaf=max_samples_leaf,
            rng=np.random.default_rng(seed),
        )


@pytest.mark.parametrize(
    ("donors", "weighted"),
    [
        pytest.param("continuous", True, id="continuous-weighted"),
        pytest.param("near_discrete", False, id="near-discrete-capped"),
    ],
)
def test_fit_forest_draws_match_the_library(donors, weighted) -> None:
    """``_fit_forest`` builds the subclass, and its grid and draws are unchanged."""
    if donors == "continuous":
        predictors, target = _continuous_donors(seed=7, rows=900)
    else:
        predictors, target = _near_discrete_donors(seed=7, rows=1500)
    weights = _sample_weights(seed=7, rows=len(target)) + 0.05 if weighted else None
    arguments = {"seed": 13, "n_estimators": 8, "max_samples_leaf": None}
    in_place = _fit_forest_with(
        qrf._InPlaceLeafQRF, predictors, target, weights, **arguments
    )
    library = _fit_forest_with(
        RandomForestQuantileRegressor, predictors, target, weights, **arguments
    )
    assert type(in_place.model) is qrf._InPlaceLeafQRF
    assert type(library.model) is RandomForestQuantileRegressor
    if donors == "near_discrete":
        assert in_place.model.max_samples_leaf == qrf._DISCRETE_Y_LEAF_BOUND
    _assert_byte_identical(
        np.asarray(in_place.model.forest_.y_train_leaves),
        np.asarray(library.model.forest_.y_train_leaves),
    )
    recipients = pd.DataFrame(predictors[:400], columns=list(in_place.columns))
    grid = list(qrf._QUANTILE_GRID)
    assert (
        np.asarray(
            in_place.model.predict(recipients.to_numpy(), quantiles=grid)
        ).tobytes()
        == np.asarray(
            library.model.predict(recipients.to_numpy(), quantiles=grid)
        ).tobytes()
    )
    uniforms = np.random.default_rng(99).random(len(recipients))
    assert (
        in_place.draw(recipients, uniforms).tobytes()
        == library.draw(recipients, uniforms).tobytes()
    )


def test_fit_forest_truncates_leaves_over_the_production_cap(monkeypatch) -> None:
    """A near-discrete target whose leaves exceed the 4,096 cap is unchanged.

    One binary predictor leaves two predictor vectors, so each leaf keeps about
    half of the 12,000 bootstrap rows, well over :data:`_DISCRETE_Y_LEAF_BOUND`.
    """
    generator = np.random.default_rng(8)
    rows = 12_000
    predictors = generator.integers(0, 2, size=(rows, 1)).astype(np.float64)
    target = generator.choice([0.0, 1.0, 2.0], size=rows, p=[0.9, 0.05, 0.05])
    shuffled: list[int] = []
    original_shuffle = random.shuffle

    def counting_shuffle(sequence):
        shuffled.append(len(sequence))
        return original_shuffle(sequence)

    arguments = {"seed": 17, "n_estimators": 3, "max_samples_leaf": None}
    monkeypatch.setattr(random, "shuffle", counting_shuffle)
    random.seed(20261117)
    library = _fit_forest_with(
        RandomForestQuantileRegressor, predictors, target, None, **arguments
    )
    library_random_state = random.getstate()
    library_shuffles = list(shuffled)
    shuffled.clear()
    random.seed(20261117)
    in_place = _fit_forest_with(
        qrf._InPlaceLeafQRF, predictors, target, None, **arguments
    )
    assert type(in_place.model) is qrf._InPlaceLeafQRF
    assert shuffled == library_shuffles
    assert min(shuffled) > qrf._DISCRETE_Y_LEAF_BOUND
    assert random.getstate() == library_random_state
    in_place_leaves = np.asarray(in_place.model.forest_.y_train_leaves)
    assert in_place_leaves.shape[-1] == qrf._DISCRETE_Y_LEAF_BOUND
    _assert_byte_identical(
        in_place_leaves, np.asarray(library.model.forest_.y_train_leaves)
    )
    recipients = pd.DataFrame(predictors[:50], columns=list(in_place.columns))
    uniforms = np.random.default_rng(5).random(len(recipients))
    assert (
        in_place.draw(recipients, uniforms).tobytes()
        == library.draw(recipients, uniforms).tobytes()
    )


def test_public_fit_and_predict_match_the_library(monkeypatch) -> None:
    """A chained public fit draws the same values with either class.

    The targets cover a positive continuous amount, an amount of either sign
    (a classifier picks the sign, then one forest gives positive and another
    negative amounts) and a near-discrete ratio (the 4,096 cap), each fitted
    on the predictors and the targets before it.
    """
    generator = np.random.default_rng(9)
    rows = 1200
    age = generator.integers(18, 80, rows).astype(np.float64)
    region = generator.integers(0, 5, rows).astype(np.float64)
    donor = pd.DataFrame(
        {
            "age": age,
            "region": region,
            "wage": np.round(np.exp(generator.normal(3.0, 0.5, rows)), 2),
            "gain": np.where(
                generator.random(rows) < 0.5,
                -np.exp(generator.normal(5.0, 1.0, rows)),
                np.exp(generator.normal(6.0, 1.0, rows)),
            ),
            "ratio": generator.choice([0.0, 0.2, 0.5, 1.0], rows),
        }
    )
    weights = generator.uniform(0.5, 4.0, rows)
    recipient = pd.DataFrame(
        {
            "age": generator.integers(18, 80, 500).astype(np.float64),
            "region": generator.integers(0, 5, 500).astype(np.float64),
        }
    )
    targets = ["wage", "gain", "ratio"]

    def fit_and_draw() -> pd.DataFrame:
        model = fit(
            donor,
            ["age", "region"],
            targets,
            weights=weights,
            n_estimators=6,
            seed=21,
        )
        return model.predict(recipient)

    in_place_draws = fit_and_draw()
    monkeypatch.setattr(qrf, "_InPlaceLeafQRF", RandomForestQuantileRegressor)
    library_draws = fit_and_draw()
    pd.testing.assert_frame_equal(in_place_draws, library_draws, check_exact=True)
    for target in targets:
        assert (
            in_place_draws[target].to_numpy().tobytes()
            == library_draws[target].to_numpy().tobytes()
        )


# ----------------------------------------------------------------------------
# The subclass behaves as an ordinary estimator
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["fit", "_get_y_train_leaves"])
def test_overrides_keep_the_library_signatures(method) -> None:
    """Callers that pass the library's argument names by keyword still work."""
    assert inspect.signature(getattr(qrf._InPlaceLeafQRF, method)) == (
        inspect.signature(getattr(RandomForestQuantileRegressor, method))
    )


def test_subclass_pickles_and_clones_like_the_library_class() -> None:
    predictors, target = _continuous_donors(seed=10, rows=300)
    model = qrf._InPlaceLeafQRF(
        n_estimators=4, max_samples_leaf=None, random_state=3, n_jobs=1
    )
    model.fit(predictors, target)
    forest = qrf._Forest(model=model, columns=("a", "b", "c"))
    restored = pickle.loads(pickle.dumps(forest))
    assert type(restored.model) is qrf._InPlaceLeafQRF
    _assert_byte_identical(
        np.asarray(restored.model.forest_.y_train_leaves),
        np.asarray(model.forest_.y_train_leaves),
    )
    recipients = pd.DataFrame(predictors[:40], columns=["a", "b", "c"])
    uniforms = np.linspace(0.0, 1.0, len(recipients))
    assert (
        restored.draw(recipients, uniforms).tobytes()
        == forest.draw(recipients, uniforms).tobytes()
    )
    cloned = clone(model)
    assert type(cloned) is qrf._InPlaceLeafQRF
    assert cloned.get_params() == model.get_params()


# ----------------------------------------------------------------------------
# The exact quantile-forest version the override mirrors
# ----------------------------------------------------------------------------


def test_installed_quantile_forest_is_the_version_the_override_mirrors() -> None:
    supported = qrf._SUPPORTED_QUANTILE_FOREST_VERSION
    assert quantile_forest.__version__ == supported
    assert importlib.metadata.version("quantile-forest") == supported


def test_dependency_and_lock_pin_the_mirrored_version_exactly() -> None:
    supported = qrf._SUPPORTED_QUANTILE_FOREST_VERSION
    paths = paths_for("microcosm-fit")
    project = tomllib.loads((paths.package / "pyproject.toml").read_text())
    declared = [
        requirement.replace(" ", "")
        for requirement in project["project"]["dependencies"]
        if requirement.replace(" ", "").startswith("quantile-forest")
    ]
    assert declared == [f"quantile-forest=={supported}"]

    lock = tomllib.loads((paths.repository / "uv.lock").read_text())
    locked = [
        package["version"]
        for package in lock["package"]
        if package["name"] == "quantile-forest"
    ]
    assert locked == [supported]
    (microcosm_fit,) = [
        package for package in lock["package"] if package["name"] == "microcosm-fit"
    ]
    locked_specifiers = [
        requirement["specifier"]
        for requirement in microcosm_fit["metadata"]["requires-dist"]
        if requirement["name"] == "quantile-forest"
    ]
    assert locked_specifiers == [f"=={supported}"]


@pytest.mark.parametrize("installed", ["1.4.3", "1.4.1", "1.5.0", "1.4.2.post1"])
def test_fit_refuses_another_quantile_forest_version(monkeypatch, installed) -> None:
    monkeypatch.setattr(quantile_forest, "__version__", installed)
    predictors, target = _continuous_donors(seed=12, rows=60)
    model = qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=0)
    with pytest.raises(RuntimeError) as refusal:
        model.fit(predictors, target)
    message = str(refusal.value)
    assert installed in message
    assert qrf._SUPPORTED_QUANTILE_FOREST_VERSION in message
    # The refusal comes before any tree is grown.
    assert not hasattr(model, "estimators_")


def test_library_fit_on_the_subclass_refuses_another_version(monkeypatch) -> None:
    """Calling quantile-forest's own ``fit`` still reaches the version check.

    The copied leaf-array code runs only inside ``_get_y_train_leaves``, so the
    check there covers callers that skip the subclass's ``fit``.
    """
    monkeypatch.setattr(quantile_forest, "__version__", "1.4.3")
    predictors, target = _continuous_donors(seed=14, rows=60)
    model = qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=0)
    with pytest.raises(RuntimeError, match="1.4.3"):
        RandomForestQuantileRegressor.fit(model, predictors, target)


def test_fit_refuses_a_quantile_forest_without_a_version(monkeypatch) -> None:
    monkeypatch.delattr(quantile_forest, "__version__")
    predictors, target = _continuous_donors(seed=23, rows=60)
    model = qrf._InPlaceLeafQRF(n_estimators=2, max_samples_leaf=None, random_state=0)
    with pytest.raises(RuntimeError, match="None is installed"):
        model.fit(predictors, target)


def test_fit_forest_refuses_another_quantile_forest_version(monkeypatch) -> None:
    monkeypatch.setattr(quantile_forest, "__version__", "1.4.3")
    predictors, target = _continuous_donors(seed=13, rows=60)
    with pytest.raises(RuntimeError, match="1.4.3"):
        qrf._fit_forest(
            predictors,
            target,
            ("a", "b", "c"),
            None,
            seed=1,
            n_estimators=2,
            max_samples_leaf=None,
            rng=np.random.default_rng(1),
        )
