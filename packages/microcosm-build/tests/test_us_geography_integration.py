"""Ordinary invented geography controls for the combined graph contract.

All NPZ/CSV bytes are created in the test. The national-sized fixture satisfies
the shape validator with invented equal cell masses and district allocations;
it is not a Census crosswalk or evidence of genuine source admission.
"""

import hashlib
import json
from collections import Counter
from dataclasses import fields
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_geography as geography
from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
)
from microcosm.build.us_runtime.puma_ladder import (
    assign_us_puma_ladder,
    decode_us_puma_ladder,
    load_us_puma_ladder,
)
from microcosm.frame import WeightKind, Weights
from microcosm.graph import ArtifactValue, KernelContext, Owned
from microcosm.graph.kernel import NumericScope


def _ordinary_lookup_bytes(**overrides):
    # The production validator requires 436 districts across 51 jurisdictions.
    # Allocate invented counts: 7 at-large + 33*10 + 11*9 = 436.
    states = [i for i in range(1, 57) if i not in {3, 7, 14, 43, 52}]
    at_large = {2, 10, 11, 38, 46, 50, 56}
    multi = [i for i in states if i not in at_large]
    rows, crosswalk = [], []
    for state in states:
        districts = (
            [0]
            if state in at_large
            else range(1, 11 if multi.index(state) < 33 else 10)
        )
        for district in districts:
            county = max(1, district)
            if state == 36:
                county = 61 if district == 1 else (1 if district == 2 else district)
            puma = state * 100_000 + 100
            tract = (state * 1000 + county) * 1_000_000 + 100
            cd = state * 100 + district
            rows.append((puma, tract, cd))
            crosswalk.append(
                {
                    "source_geography_id": f"{SOURCE_CONGRESSIONAL_DISTRICT_PREFIX}{cd:04d}",
                    "target_geography_id": f"{CURRENT_CONGRESSIONAL_DISTRICT_PREFIX}{cd:04d}",
                    "weight": 1.0,
                    "pair_population": 1,
                }
            )
    rows.sort()
    pumas = Counter(row[0] for row in rows)
    arrays = {
        "puma": np.asarray(sorted(pumas), dtype=np.int64),
        "puma_population": np.asarray([pumas[p] for p in sorted(pumas)]),
        "joint_overlap_puma": np.asarray([r[0] for r in rows]),
        "joint_overlap_tract": np.asarray([r[1] for r in rows]),
        "joint_overlap_cd": np.asarray([r[2] for r in rows]),
        "joint_overlap_population": np.ones(len(rows), dtype=np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "us_puma_ladder",
                    "puma_vintage": "2020_puma",
                    "sampling_basis": "population",
                    "layers": {
                        name: {
                            "vintage": vintage,
                            "source": "invented ordinary fixture",
                        }
                        for name, vintage in (
                            ("congressional_district", "119th_congress"),
                            ("county", "2020_census"),
                            ("tract", "2020_census"),
                        )
                    },
                }
            )
        ),
    }
    for layer, coordinate in (("cd", 2), ("county", 1), ("tract", 1)):
        counts = Counter(
            (
                row[0],
                row[coordinate] // 1_000_000 if layer == "county" else row[coordinate],
            )
            for row in rows
        )
        keys = sorted(counts)
        arrays[f"{layer}_overlap_puma"] = np.asarray([key[0] for key in keys])
        arrays[f"{layer}_overlap_{layer}"] = np.asarray([key[1] for key in keys])
        arrays[f"{layer}_overlap_population"] = np.asarray(
            [counts[key] for key in keys]
        )
    arrays.update(overrides)
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue(), pd.DataFrame(crosswalk).to_csv(index=False).encode()


def _nodes():
    return geography.us_geography_nodes(
        (Owned("person", "age", "int64"), Owned("household", "state_fips", "int64")),
        base="ordinary_base",
        context_producer="ordinary_context",
    )


def _context(household, weights, ladder_bytes, crosswalk_bytes):
    artifacts = {}
    for alias, payload, type_ in (
        ("ladder", ladder_bytes, geography.US_PUMA_LOOKUP_TYPE),
        ("crosswalk", crosswalk_bytes, geography.US_CD_CROSSWALK_TYPE),
    ):
        artifacts[alias] = ArtifactValue(
            payload,
            type_,
            hashlib.sha256(payload).hexdigest(),
            "a" * 64,
            NumericScope(),
        )
    node = _nodes()[-1]
    return KernelContext(
        node=node,
        tables={"household": household},
        weights={"household": Weights(np.asarray(weights), WeightKind.DESIGN)},
        strata=pd.Series(dtype="int64"),
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


def _households():
    # Same ordinary mass control as test_us_puma_ladder._gated_household:
    # NYC = 2.6% nationally and 2.6/6.2 of New York's mass.
    return pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "state_fips": [36, 36, 6, 48, 1],
            "puma": ["3600100", "3600100", "0600100", "4800100", "0100100"],
            "county_fips": ["36061", "36001", "06001", "48001", "01001"],
            "congressional_district_geoid": [3601, 3602, 601, 4801, 101],
        }
    )


def test_byte_and_path_joint_decode_and_assignments_match(tmp_path):
    payload, _ = _ordinary_lookup_bytes()
    path = tmp_path / "invented-ladder.npz"
    path.write_bytes(payload)
    path_ladder = load_us_puma_ladder(path)
    byte_ladder = decode_us_puma_ladder(payload)
    for field in fields(path_ladder):
        left, right = getattr(path_ladder, field.name), getattr(byte_ladder, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
            assert left.dtype == right.dtype
        else:
            assert left == right
    for assign_tract in (False, True):
        pd.testing.assert_frame_equal(
            assign_us_puma_ladder(
                _households(), path_ladder, seed=17, assign_tract=assign_tract
            ),
            assign_us_puma_ladder(
                _households(), byte_ladder, seed=17, assign_tract=assign_tract
            ),
        )


@pytest.mark.parametrize("defect", ["schema_v1", "joint_population"])
def test_byte_and_path_reject_the_same_joint_defect(tmp_path, defect):
    payload, _ = _ordinary_lookup_bytes()
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if defect == "schema_v1":
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["schema_version"] = 1
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    else:
        arrays["joint_overlap_population"][0] += 1
    path = tmp_path / "invalid-invented-ladder.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError) as path_error:
        load_us_puma_ladder(path)
    with pytest.raises(ValueError) as byte_error:
        decode_us_puma_ladder(path.read_bytes())
    assert str(path_error.value) == str(byte_error.value)


@pytest.mark.parametrize("wrapper", [bytearray, memoryview])
def test_byte_decoder_requires_immutable_bytes(wrapper):
    with pytest.raises(TypeError, match="immutable bytes"):
        decode_us_puma_ladder(wrapper(b"not an archive"))


@pytest.mark.parametrize("unsupported_weight", [0.0, 1.0])
def test_graph_gate_rejects_real_unsupported_pair_even_at_zero_weight(
    unsupported_weight,
):
    ladder_bytes, crosswalk_bytes = _ordinary_lookup_bytes()
    household = _households()
    weights = [2.6, 3.6, 73.8, 20.0, unsupported_weight]
    kernel = geography.USGeographyGateKernel()
    assert (
        kernel.run(_context(household, weights, ladder_bytes, crosswalk_bytes)).receipt[
            "outcome"
        ]
        == "pass"
    )
    # County 01001 and CD 102 are individually present in PUMA 0100100,
    # but the only supported pairs are county 01001/CD 101, county 01002/CD 102.
    household.loc[4, "congressional_district_geoid"] = 102
    result = kernel.run(_context(household, weights, ladder_bytes, crosswalk_bytes))
    assert result.receipt["outcome"] == "fail"
    expected_mass = geography.us_puma_ladder_gate(household, np.asarray(weights))
    ladder = decode_us_puma_ladder(ladder_bytes)
    assert (100100, 1001) in set(
        zip(ladder.county_overlap_puma, ladder.county_overlap_county, strict=True)
    )
    assert (100100, 102) in set(
        zip(ladder.cd_overlap_puma, ladder.cd_overlap_cd, strict=True)
    )
    assert (100100, 1001, 102) not in set(
        zip(
            ladder.joint_overlap_puma,
            ladder.joint_overlap_tract // 1_000_000,
            ladder.joint_overlap_cd,
            strict=True,
        )
    )
    expected_joint = geography.us_puma_ladder_joint_support_gate(household, ladder)
    assert expected_mass.passed
    assert not expected_joint.passed
    assert expected_joint.details["unsupported_rows"] == 1
    assert (
        result.receipt["evidence"]
        == geography.GateReport((expected_mass, expected_joint)).to_manifest()
    )


def test_graph_gate_retains_independent_legacy_mass_failure():
    ladder_bytes, crosswalk_bytes = _ordinary_lookup_bytes()
    household = _households()
    weights = np.asarray([0.0, 6.2, 73.8, 20.0, 0.0])
    ladder = decode_us_puma_ladder(ladder_bytes)
    assert geography.us_puma_ladder_joint_support_gate(household, ladder).passed
    assert not geography.us_puma_ladder_gate(household, weights).passed
    result = geography.USGeographyGateKernel().run(
        _context(household, weights, ladder_bytes, crosswalk_bytes)
    )
    assert result.receipt["outcome"] == "fail"


def test_graph_gate_declares_both_lookup_producer_dependencies():
    lookup, _, _, gate = _nodes()
    assert {
        (item.name, item.producer, item.artifact, item.type)
        for item in gate.artifact_inputs
    } == {
        ("ladder", lookup.id, "ladder", geography.US_PUMA_LOOKUP_TYPE),
        ("crosswalk", lookup.id, "crosswalk", geography.US_CD_CROSSWALK_TYPE),
    }
