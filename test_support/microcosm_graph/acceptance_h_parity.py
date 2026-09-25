"""Charter group H: parity (migration acceptance).

These three properties are the only ones in the charter whose subject is not
the toy country. Each compares a graph node's output against a pinned artifact
produced by the lane that wraps the legacy kernel or migrates the country
spine, so each waits on a fixture this lane cannot manufacture: inventing one
would prove that the suite agrees with itself, which is exactly what parity
must not mean.

Every test names the fixture path it expects and reads it and nothing else, so
the producing lane can drop its artifact in and delete the marker. Until then
the fixture is absent, the test fails, and the ``xfail`` reason says whose
fixture it is waiting for.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from microcosm.graph import platform_fingerprint
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-graph")

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", (_TEST_PATHS.tests / "_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: Where the parity lanes drop their pinned fixtures.
PARITY = _TEST_PATHS.tests / "fixtures" / "parity"

#: H1: one directory per wrapped legacy kernel, each holding ``graph.json``
#: (the node declaration), ``inputs.csv``, ``direct.csv`` (the direct call's
#: output at the pinned seed), and ``pins.json`` (seed, kernel ref, kernel
#: implementation hash, target node key, and the dependency versions the pin
#: was taken under).
KERNEL_PARITY = PARITY / "kernels"

#: H2: ``uk_spine.json`` — the 33-stage FRS spine expressed as a graph — plus
#: ``sources/``, the data-only bundle both the graph and the legacy oracle
#: rebuild their transforms from. The root transform's weights differ at the
#: last bit between machines, so both sides recompute the root from the raw
#: tables in the test's own process and nothing is pinned.
UK_SPINE_PARITY = PARITY / "uk_spine"

#: H3: ``us_post_transfer.json`` — the derive/seed/simulate subgraph of the
#: stacked spine — plus ``expected.csv``, its pinned output.
US_POST_TRANSFER_PARITY = PARITY / "us_post_transfer"

#: The wrapped kernels H1 covers, in the order the charter names them.
#: The three wrapped kernels the kernel lane shipped: ``fit.qrf@1`` fits on
#: donors and draws on recipients in one node, so there is no separate draw.
WRAPPED_KERNELS = ("fit.qrf", "calibrate", "simulate")

#: What each wrapper honestly claims about its numbers. The forest stack does
#: not promise cross-platform bit stability, so ``fit.qrf@1`` says so; parity
#: in the locked environment is still asserted byte for byte below.
NUMERIC_CLAIMS = {
    "fit.qrf": "platform_bitwise",
    "calibrate": "bitwise",
    "simulate": "bitwise",
}


def _assert_same_bytes(actual, expected) -> None:
    assert actual.dtype == expected.dtype
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert np.array_equal(actual.isna().to_numpy(), expected.isna().to_numpy())


def _frame_differences(actual, expected) -> str:
    """Name the cells two frames disagree on; two identities alone say nothing."""
    import pandas as pd

    lines: list[str] = []
    for entity in sorted(set(actual.entities) | set(expected.entities)):
        if entity not in actual.entities or entity not in expected.entities:
            lines.append(f"{entity}: present in only one frame")
            continue
        left, right = actual.table(entity), expected.table(entity)
        if list(left.columns) != list(right.columns):
            symmetric = sorted(set(left.columns) ^ set(right.columns))
            lines.append(f"{entity}: column order or set differs ({symmetric})")
        if len(left) != len(right):
            lines.append(f"{entity}: {len(left)} vs {len(right)} rows")
            continue
        for column in left.columns:
            if column not in right.columns:
                continue
            x, y = left[column], right[column]
            if str(x.dtype) != str(y.dtype):
                lines.append(f"{entity}.{column}: dtype {x.dtype} vs {y.dtype}")
            if x.equals(y):
                continue
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                xv = x.to_numpy(dtype="float64", na_value=np.nan)
                yv = y.to_numpy(dtype="float64", na_value=np.nan)
                unequal = ~np.isclose(xv, yv, rtol=0.0, atol=0.0, equal_nan=True)
                count = int(unequal.sum())
                largest = float(np.nanmax(np.abs(xv - yv)[unequal])) if count else 0.0
                lines.append(
                    f"{entity}.{column}: {count} of {len(x)} cells differ, "
                    f"max |difference| {largest:.3e}"
                )
            else:
                count = int((x.astype("string") != y.astype("string")).sum())
                lines.append(f"{entity}.{column}: {count} of {len(x)} cells differ")
    for entity in sorted(
        set(actual.weighted_entities) | set(expected.weighted_entities)
    ):
        if (
            entity not in actual.weighted_entities
            or entity not in expected.weighted_entities
        ):
            lines.append(f"{entity}: weighted in only one frame")
            continue
        xv = np.asarray(actual.weights_for(entity).values, dtype="float64")
        yv = np.asarray(expected.weights_for(entity).values, dtype="float64")
        if xv.shape != yv.shape:
            lines.append(f"{entity} weights: {xv.shape} vs {yv.shape}")
        elif xv.tobytes() != yv.tobytes():
            unequal = ~np.isclose(xv, yv, rtol=0.0, atol=0.0, equal_nan=True)
            lines.append(
                f"{entity} weights: {int(unequal.sum())} of {len(xv)} differ, "
                f"max |difference| {float(np.nanmax(np.abs(xv - yv))):.3e}"
            )
    return "\n".join(lines) or (
        "no table cell or weight differs; strata, mass log, or metadata differ"
    )


def _require(path: Path, produced_by: str) -> Path:
    """Fail with the fixture's path and its owner, never with a bare error."""
    assert path.exists(), (
        f"missing parity fixture {path}; it is produced by {produced_by}, not "
        "by the acceptance lane — inventing it here would make the test agree "
        "with itself instead of with the legacy kernel."
    )
    return path


def _direct_table(case: Path, name: str = "direct.csv"):
    import pandas as pd

    return pd.read_csv(case / name, float_precision="round_trip")


__all__ = [name for name in globals() if not name.startswith("__")]
