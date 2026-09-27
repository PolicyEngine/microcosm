"""A seeded sweep for any surviving disagreement between the two paths.

The battery drives 116 hand-built comparisons and a 121-case sweep of the
measured object-axis equality classes. This is the wider net: a deterministic
pseudo-random walk over axis kinds, axis values, column kinds and column
mutations, driving every pair through ``same_replayed_frame`` and through the
seal and recording any pair where the two verdicts differ.

An earlier adversarial pass found three real divergences this way, two of them
in the accept/refuse directions, so the sweep is committed rather than
described: a disagreement it reports is a finding, and the receipt names it.

    python experiments/native-retention-seal/agreement_fuzz.py OUT.json [ROUNDS]
"""

import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in sorted((ROOT / "packages").glob("*/src"))]
sys.path.insert(0, str(ROOT / "packages/microcosm-build/tests"))

import test_us_survey_population_replay as battery  # noqa: E402

from microcosm.build.us_runtime.survey_population_replay import (  # noqa: E402
    replayed_frame_seal,
    same_replayed_frame,
    same_replayed_frame_seals,
)

# Values the store codec admits, plus the null sentinels, plus the two shapes
# `==` holds equal that the codec spells apart.
_POOL = [
    None,
    float("nan"),
    pd.NA,
    pd.NaT,
    True,
    False,
    0,
    1,
    -0.0,
    0.0,
    1.0,
    2**53,
    2**53 + 1,
    float("inf"),
    float("-inf"),
    "a",
    "",
    b"a",
    np.int64(1),
    np.float64(1.0),
    np.bool_(True),
]
_AXES = ("object", "int64", "Int64", "string", "boolean", "float64", "datetime")


def _axis(rng, length):
    kind = rng.choice(_AXES)
    if kind == "object":
        return pd.Index([rng.choice(_POOL) for _ in range(length)], dtype=object)
    if kind == "int64":
        return pd.Index(
            [rng.randrange(-(2**60), 2**60) for _ in range(length)], dtype="int64"
        )
    if kind == "Int64":
        return pd.Index(
            pd.array(
                [
                    None if rng.random() < 0.3 else rng.randrange(0, 2**60)
                    for _ in range(length)
                ],
                dtype="Int64",
            )
        )
    if kind == "string":
        return pd.Index(
            pd.array(
                [
                    None if rng.random() < 0.3 else rng.choice("abc")
                    for _ in range(length)
                ],
                dtype="string",
            )
        )
    if kind == "boolean":
        return pd.Index(
            pd.array(
                [
                    None if rng.random() < 0.3 else rng.random() < 0.5
                    for _ in range(length)
                ],
                dtype="boolean",
            )
        )
    if kind == "float64":
        return pd.Index(
            [float(rng.choice([0.0, -0.0, 1.5, float("nan")])) for _ in range(length)],
            dtype="float64",
        )
    return pd.DatetimeIndex(
        [f"2020-01-{1 + rng.randrange(0, 28):02d}" for _ in range(length)]
    )


def _zero_hidden_backing(frame, rng):
    """What a store round trip does: zero the bytes beneath a null mask."""
    column = rng.choice(["nullable_integer", "nullable_boolean"])
    frame.person[column].array._data[frame.person[column].array._mask] = 0
    return frame


def _flip_hidden_backing(frame, rng):
    column = rng.choice(["nullable_integer", "nullable_boolean"])
    data = frame.person[column].array._data
    data[frame.person[column].array._mask] = rng.randrange(1, 90)
    return frame


def _flip_present_value(frame, rng):
    data = frame.person["nullable_integer"].array._data
    data[~frame.person["nullable_integer"].array._mask] = rng.randrange(1, 90)
    return frame


def _flip_mask(frame, rng):
    mask = frame.person["nullable_integer"].array._mask
    mask[rng.randrange(0, len(mask))] ^= True
    return frame


def _retype_native(frame, rng):
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["native_float"] = tables["person"]["native_float"].astype(
        rng.choice(["float32", "int64", "float64"])
    )
    return battery._rebuild(frame, tables=tables)


def _retype_text(frame, rng):
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["text"] = pd.array(
        [rng.choice([None, "a", "b", ""]) for _ in range(len(tables["person"]))],
        dtype="string",
    )
    return battery._rebuild(frame, tables=tables)


def _retype_object(frame, rng):
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["object_value"] = pd.Series(
        [rng.choice(_POOL) for _ in range(len(tables["person"]))],
        index=tables["person"].index,
        dtype=object,
    )
    return battery._rebuild(frame, tables=tables)


def _untouched(frame, rng):
    return frame


_COLUMN_MUTATIONS = (
    _untouched,
    _untouched,
    _zero_hidden_backing,
    _flip_hidden_backing,
    _flip_present_value,
    _flip_mask,
    _retype_native,
    _retype_text,
    _retype_object,
)


def _verdict(call, *arguments):
    try:
        call(*arguments)
    except ValueError as error:
        return str(error)
    return None


def _sealed(expected, actual):
    try:
        one, other = replayed_frame_seal(expected), replayed_frame_seal(actual)
    except ValueError as error:
        return str(error)
    return _verdict(same_replayed_frame_seals, one, other)


def main():
    out = pathlib.Path(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    rng = random.Random(20260918)
    base = battery._frame()
    length = len(base.person)
    disagreements, verdicts = [], {}
    for index in range(rounds):
        left, right = _axis(rng, length), _axis(rng, length)
        if rng.random() < 0.25:
            right = left  # equal axes, so a column mutation is what differs
        try:
            expected = battery._with_person_axis(base, left)
            actual = battery._with_person_axis(base, right)
        except Exception:  # an axis pandas will not accept on this table
            continue
        # Reach the series arm as well as the axis arm: mutate one column of
        # the actual side the way the store or a defect would.
        mutation = rng.choice(_COLUMN_MUTATIONS)
        try:
            actual = mutation(actual, rng)
        except Exception:
            continue
        direct = _verdict(same_replayed_frame, expected, actual)
        sealed = _sealed(expected, actual)
        verdicts[str(direct)] = verdicts.get(str(direct), 0) + 1
        if direct != sealed:
            disagreements.append(
                {
                    "round": index,
                    "left": repr(list(left))[:200],
                    "right": repr(list(right))[:200],
                    "object_path": direct,
                    "seal_path": sealed,
                }
            )
    out.write_text(
        json.dumps(
            {
                "scope": (
                    "diagnostic agreement sweep; not a build, certification or "
                    "release artifact"
                ),
                "release_eligible": False,
                "seed": 20260918,
                "rounds_requested": rounds,
                "pairs_compared": sum(verdicts.values()),
                "disagreements": disagreements,
                "verdicts_reached": dict(
                    sorted(verdicts.items(), key=lambda item: -item[1])
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"pairs={sum(verdicts.values())} disagreements={len(disagreements)}")
    for code, count in sorted(verdicts.items(), key=lambda item: -item[1]):
        print(f"  {count:5d}  {code}")
    for row in disagreements[:10]:
        print("  DISAGREE", json.dumps(row)[:220])


if __name__ == "__main__":
    main()
