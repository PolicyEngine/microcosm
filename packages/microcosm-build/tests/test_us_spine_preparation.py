"""Separate actual US source stacking from importance-weight harmonization."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.spine_assembly as assembly
import microcosm.build.us_runtime.stacked_spine as stacked
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import ContentStore
from microcosm.graph.canonical import canonical_json


def _fixture_module(name):
    spec = importlib.util.spec_from_file_location(
        f"_preparation_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


raw = _fixture_module("test_us_spine_assembly")
surveys = _fixture_module("test_us_stacked_spine")


def _fingerprint(frame, directory):
    path = ContentStore(directory).put_frame("a" * 64, frame, node_key="b" * 64)
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            relative, content = (
                file.relative_to(path).as_posix().encode(),
                file.read_bytes(),
            )
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "little"))
            digest.update(content)
    digest.update(
        json.dumps(
            stacked._json_ready(frame.metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    return digest.hexdigest()


def _interleaved():
    return {
        "asec": raw._with_structural_ids(raw._asec_frame(), [5, 95]),
        "acs": raw._with_structural_ids(raw._asec_frame(), [1, 2]),
    }


def _equal_frames(left, right):
    assert left.metadata == right.metadata
    assert left.mass_log == right.mass_log  # Ordered operator history is normative.
    pd.testing.assert_series_equal(left.strata, right.strata, check_exact=True)
    for entity in left.entities:
        pd.testing.assert_frame_equal(
            left.table(entity), right.table(entity), check_exact=True
        )
    assert left.weights_for("household").kind == right.weights_for("household").kind
    assert (
        left.weights_for("household").values.tobytes()
        == right.weights_for("household").values.tobytes()
    )


@pytest.mark.parametrize(
    "case",
    [
        "generic_mixed",
        "generic_interleaved",
        "production_1.0",
        "production_0.5",
        "pilot_0.5",
    ],
)
def test_legacy_outputs_match_pre_refactor_pinned_codec_and_metadata(case, tmp_path):
    # Captured at aad0defb by reviewdir/us-stack-legacy-baseline.py using these
    # exact existing synthetic fixtures; includes codec bytes and ordered logs.
    expected = BASELINE_HASHES[case]
    if case == "generic_mixed":
        frame = assembly.assemble_spines(
            {"asec": raw._asec_frame(), "acs": raw._acs_frame()},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )
    elif case == "generic_interleaved":
        frame = assembly.assemble_spines(
            _interleaved(), household_mass_shares={"asec": 0.3, "acs": 0.7}
        )
    else:
        controls = (
            {"sample_fraction": float(case.split("_")[1]), "sample_seed": 578}
            if case.startswith("production")
            else {"acs_sample_fraction": 0.5, "acs_sample_seed": 578}
        )
        frame = stacked.assemble_stacked_spine(
            surveys._asec_source(), surveys._acs_source(), **controls
        ).frame
    assert _fingerprint(frame, tmp_path / "cache") == expected


def test_prepare_spines_keeps_design_then_reproduces_interleaved_harmonization():
    sources, shares = _interleaved(), {"asec": 0.3, "acs": 0.7}
    prepared = assembly.prepare_spines(sources, household_mass_shares=shares)
    assert prepared.frame.weights_for("household").kind is WeightKind.DESIGN
    assert prepared.context["household_order"] is not None
    assert stacked.STACKED_SPINE_MANIFEST_KEY not in prepared.frame.metadata
    result = assembly.harmonize_spine_weights(
        household=prepared.frame.table("household"),
        weights=prepared.frame.weights_for("household"),
        context=prepared.context,
    )
    legacy = assembly.assemble_spines(sources, household_mass_shares=shares)
    assert result.weights.kind is WeightKind.IMPORTANCE
    assert (
        result.weights.values.tobytes()
        == legacy.weights_for("household").values.tobytes()
    )
    assert result.mass_log == legacy.mass_log
    for entity in legacy.entities:
        pd.testing.assert_frame_equal(
            prepared.frame.table(entity), legacy.table(entity), check_exact=True
        )


def test_graph_preparation_rejects_mixed_kinds_while_legacy_still_accepts():
    sources = {"asec": raw._asec_frame(), "acs": raw._acs_frame()}
    assert sources["acs"].weights_for("household").kind is WeightKind.CALIBRATED
    with pytest.raises(ValueError, match="DESIGN"):
        assembly.prepare_spines(
            sources, household_mass_shares={"asec": 0.5, "acs": 0.5}
        )
    assert (
        assembly.assemble_spines(
            sources, household_mass_shares={"asec": 0.5, "acs": 0.5}
        )
        .weights_for("household")
        .kind
        is WeightKind.IMPORTANCE
    )


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_stacked_preparation_and_view_only_harmonization_match_legacy(fraction):
    asec, acs = surveys._asec_source(), surveys._acs_source()
    prepared = stacked.prepare_stacked_spine(
        asec, acs, sample_fraction=fraction, sample_seed=578
    )
    assert prepared.frame.weights_for("household").kind is WeightKind.DESIGN
    assert stacked.STACKED_SPINE_MANIFEST_KEY not in prepared.frame.metadata
    assert "weight_harmonization" not in prepared.receipt
    with pytest.raises(ValueError, match="is absent"):
        stacked.validate_stacked_spine_frame(prepared.frame, boundary="preparation")
    restored = json.loads(canonical_json(prepared.receipt))
    result = stacked.harmonize_stacked_spine_weights(
        household=prepared.frame.table("household"),
        weights=prepared.frame.weights_for("household"),
        preparation=restored,
    )
    finished = stacked.finish_stacked_spine(prepared)
    legacy = stacked.assemble_stacked_spine(
        asec, acs, sample_fraction=fraction, sample_seed=578
    )
    _equal_frames(finished.frame, legacy.frame)
    assert (
        result.weights.values.tobytes()
        == legacy.frame.weights_for("household").values.tobytes()
    )
    assert stacked._json_ready(result.metadata) == stacked._json_ready(
        legacy.frame.metadata
    )
    assert stacked._json_ready(result.receipt) == legacy.receipt
    assert result.legacy_mass_log == legacy.frame.mass_log
    if fraction == 1.0:
        source_weights = np.concatenate(
            [asec.weights_for("household").values, acs.weights_for("household").values]
        )
        assert (
            prepared.frame.weights_for("household").values.tobytes()
            == source_weights.tobytes()
        )
    else:
        sample = prepared.receipt["sampling"]["survey_samples"]["asec"]
        assert (
            sample["normalization_factor"]
            == sample["incoming_household_mass"] / sample["sampled_household_mass"]
        )
        assert (
            sample["normalization_factor"]
            != sample["eligible_household_count"] / sample["realized_household_count"]
        )


@pytest.mark.parametrize("field", ["weights", "ids", "channels", "permutation"])
def test_harmonization_refuses_changed_input_binding(field):
    prepared = assembly.prepare_spines(
        _interleaved(), household_mass_shares={"asec": 0.5, "acs": 0.5}
    )
    household = prepared.frame.table("household").copy()
    weights = prepared.frame.weights_for("household")
    context = stacked._json_ready(prepared.context)
    if field == "weights":
        weights = Weights(weights.values * 2, WeightKind.DESIGN)
    elif field == "ids":
        household["household_id"] += 1
    elif field == "channels":
        household["household_support_channel"] = "asec"
    else:
        context["household_order"] = [0] * len(household)
    with pytest.raises(ValueError):
        assembly.harmonize_spine_weights(
            household=household, weights=weights, context=context
        )


def test_preparation_never_performs_importance_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Importance allocation ran during source preparation")

    monkeypatch.setattr(assembly, "_values_to_total", forbidden)
    prepared = stacked.prepare_stacked_spine(
        surveys._asec_source(),
        surveys._acs_source(),
        sample_fraction=0.5,
        sample_seed=578,
    )
    assert prepared.frame.weights_for("household").total == 950.0
    assert all("allocated" not in record.reason for record in prepared.frame.mass_log)
    assert prepared.receipt["assembly_context"]["household_order"] is None


def test_harmonization_needs_only_bound_household_views_and_serialized_receipt(
    monkeypatch,
):
    prepared = stacked.prepare_stacked_spine(
        surveys._asec_source(),
        surveys._acs_source(),
        sample_fraction=0.5,
        sample_seed=578,
    )
    expected = stacked.finish_stacked_spine(prepared)
    receipt = json.loads(canonical_json(prepared.receipt))

    def forbidden(*args, **kwargs):
        raise AssertionError("Harmonization attempted to restack source tables")

    monkeypatch.setattr(assembly, "_stack_spine_tables", forbidden)
    monkeypatch.setattr(stacked, "_sample_stacked_inputs", forbidden)
    result = stacked.harmonize_stacked_spine_weights(
        household=prepared.frame.table("household")[
            ["household_id", "household_support_channel"]
        ],
        weights=prepared.frame.weights_for("household"),
        preparation=receipt,
    )
    assert (
        result.weights.values.tobytes()
        == expected.frame.weights_for("household").values.tobytes()
    )
    assert result.legacy_mass_log == expected.frame.mass_log


def test_valid_permutation_cannot_change_within_source_correction_order():
    prepared = assembly.prepare_spines(
        _interleaved(), household_mass_shares={"asec": 0.5, "acs": 0.5}
    )
    context = stacked._json_ready(prepared.context)
    # Swap the two ASEC source positions while retaining a valid bijection and
    # contiguous channel blocks. Source-ID binding must reject this change.
    context["household_order"] = [
        1 if position == 0 else 0 if position == 1 else position
        for position in context["household_order"]
    ]
    with pytest.raises(ValueError, match="original source ID order"):
        assembly.harmonize_spine_weights(
            household=prepared.frame.table("household"),
            weights=prepared.frame.weights_for("household"),
            context=context,
        )


@pytest.mark.parametrize("anchor", ["asec", "acs"])
def test_pilot_preparation_preserves_selected_anchor_share_order_and_clone_ancestry(
    anchor,
):
    controls = dict(
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
        mass_anchor_channel=anchor,
        household_mass_shares={"acs": 0.7, "asec": 0.3},
    )
    asec, acs = surveys._asec_gap_source(), surveys._acs_gap_source()
    prepared = stacked.prepare_stacked_spine(asec, acs, **controls)
    finished = stacked.finish_stacked_spine(prepared)
    legacy = stacked.assemble_stacked_spine(asec, acs, **controls)
    _equal_frames(finished.frame, legacy.frame)
    assert list(finished.receipt["household_mass_shares"]) == ["acs", "asec"]
    assert prepared.receipt["assembly_context"]["ordered_channels"][0] == anchor
    clone_controls = dict(
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
    )
    _equal_frames(
        raw.clone_us_frame_for_puf_support(finished.frame, **clone_controls),
        raw.clone_us_frame_for_puf_support(legacy.frame, **clone_controls),
    )


def test_legacy_stacked_mixed_kind_path_remains_supported():
    asec, acs = surveys._asec_source(), surveys._acs_source()
    calibrated = Frame(
        {entity: acs.table(entity) for entity in acs.entities},
        acs.schema,
        {
            "household": Weights(
                acs.weights_for("household").values, WeightKind.CALIBRATED
            )
        },
        acs.strata,
    )
    controls = dict(acs_sample_fraction=1.0, acs_sample_seed=578)
    _equal_frames(
        stacked.assemble_stacked_spine(asec, calibrated, **controls).frame,
        stacked.assemble_stacked_spine(asec, acs, **controls).frame,
    )
    with pytest.raises(ValueError, match="DESIGN"):
        stacked.prepare_stacked_spine(asec, calibrated, **controls)


def test_preparation_receipt_is_frozen_and_cannot_authorize_mutated_gq_lineage():
    prepared = stacked.prepare_stacked_spine(
        surveys._asec_gap_source(),
        surveys._acs_gap_source(),
        sample_fraction=1.0,
        sample_seed=578,
    )
    with pytest.raises(TypeError):
        prepared.receipt["assembly_context"]["incoming_masses"]["acs"] = 0.0
    household = prepared.frame.table("household").copy()
    gq = household["household_support_channel"].eq("acs") & household["TYPEHUGQ"].eq(2)
    assert gq.sum() == 1
    household.loc[gq, "TYPEHUGQ"] = 1
    household.loc[gq, "tenure_type"] = "RENTED"
    tables = {
        entity: prepared.frame.table(entity) for entity in prepared.frame.entities
    }
    tables["household"] = household
    mutated = Frame(
        tables,
        prepared.frame.schema,
        {"household": prepared.frame.weights_for("household")},
        prepared.frame.strata,
        mass_log=prepared.frame.mass_log,
        metadata=prepared.frame.metadata,
    )
    with pytest.raises(ValueError, match="group-quarters household lineage differs"):
        stacked.StackedSpinePreparation(mutated, prepared.receipt)


@pytest.mark.parametrize("context", [None, {}, {"schema": "unknown"}])
def test_harmonization_refuses_invalid_context_by_name(context):
    prepared = assembly.prepare_spines(
        _interleaved(), household_mass_shares={"asec": 0.5, "acs": 0.5}
    )
    with pytest.raises(ValueError, match="preparation context"):
        assembly.harmonize_spine_weights(
            household=prepared.frame.table("household"),
            weights=prepared.frame.weights_for("household"),
            context=context,
        )


BASELINE_HASHES = {
    "generic_mixed": "9623fb7266c16e512a243297916362119f2635dc43561a726501d84b63f45707",
    "generic_interleaved": "496dc1882fca59fe604b2e49cb0f46e4d384fcf6f434af747288d94b7a246ab2",
    "production_1.0": "ff018a811791f7a3f0df0a210c7a827512a890fdeb52b6a704bdfadd51cada61",
    "production_0.5": "ba49067a0275888dfd8515b7ac1b10e60e48ad53754bee13a3341904e0f80d50",
    "pilot_0.5": "c8aeb1eb6abef3b633b4a5d20f1810727e1db379675c524a8f56372fd3c10cd3",
}
