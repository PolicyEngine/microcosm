"""The actual rent fit has donor-only preparation and explicit finalization."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import housing_inputs as rent
from microcosm.fit import QRF
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFApplyKernel,
    LegacyQRFTrainKernel,
    legacy_qrf_apply_nodes,
    legacy_qrf_train_nodes,
    read_raw_target,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)
from microcosm.graph.canonical import canonical_json


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "_early_rent_fixtures", Path(__file__).with_name("test_us_housing_inputs.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()


@pytest.fixture(autouse=True)
def serial_workers(monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def test_donor_preparation_does_not_read_recipients_or_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Donor preparation touched a recipient or fitted a model")

    monkeypatch.setattr(rent, "_recipient_head_features", forbidden)
    monkeypatch.setattr(rent, "QRF", forbidden)
    donor = fixtures._donor()
    prepared = rent.prepare_acs_rent_donor(donor)
    assert prepared.context["schema"] == "microcosm.us.acs_rent_encoding.v1"
    assert prepared.training.index.equals(donor.index)
    assert tuple(prepared.training.columns) == (
        *prepared.predictors,
        "rent",
        "household_weight",
    )
    np.testing.assert_array_equal(
        prepared.training["household_weight"], donor["household_weight"]
    )
    with pytest.raises(TypeError):
        prepared.context["category_levels"]["tenure_type"] = ("changed",)


def test_preparation_preserves_archived_joint_sample_and_reset_order():
    donor = fixtures._donor(12_000)
    donor.index = pd.Index(np.arange(len(donor)) + 1_000, name="external_row")
    donor["rent_is_allocated"] = np.arange(len(donor)) >= 6_000
    donor["real_estate_taxes_is_allocated"] = np.arange(len(donor)) < 6_000
    sampled, masks = rent._archived_joint_training_sample(donor)
    expected = sampled.loc[masks["rent"]]
    prepared = rent.prepare_acs_rent_donor(donor)
    assert len(prepared.training) == 5_000
    assert prepared.training.index.equals(expected.index)
    for column in ("age", "rent", "household_weight"):
        pd.testing.assert_series_equal(prepared.training[column], expected[column])


def test_recipient_context_roundtrip_and_finalizer_preserve_indices_and_head_placement():
    base = fixtures._frame()
    tables = {entity: base.table(entity).copy() for entity in base.entities}
    nonhead = tables["person"].iloc[[0]].copy()
    nonhead["person_id"] = 999
    nonhead["is_household_head"] = False
    tables["person"] = pd.concat([tables["person"], nonhead], ignore_index=True)
    tables["person"].index = pd.Index(np.arange(21) * 3 + 101, name="original_person")
    frame = rent.derive_us_housing_inputs(
        Frame(tables, base.schema, {"household": base.weights_for("household")})
    )
    donor = rent.prepare_acs_rent_donor(fixtures._donor())
    prepared = rent.prepare_acs_rent_recipient(
        frame, donor_context=json.loads(canonical_json(donor.context))
    )
    assert prepared.person_index.equals(frame.person.index)
    assert prepared.predictors.index.equals(
        pd.Index(frame.table("household")["household_id"].to_numpy())
    )
    np.testing.assert_array_equal(prepared.head_positions, np.arange(20))
    assert not prepared.head_positions.flags.writeable
    raw = pd.Series(
        np.arange(20, dtype=float) - 3, index=prepared.predictors.index, name="rent"
    )
    actual = rent.finalize_acs_rent(prepared, raw)
    np.testing.assert_array_equal(actual, np.r_[np.maximum(raw.to_numpy(), 0), 0.0])
    with pytest.raises(ValueError, match="index"):
        rent.finalize_acs_rent(prepared, raw.set_axis(pd.RangeIndex(20)))
    with pytest.raises(ValueError, match="nonfinite"):
        rent.finalize_acs_rent(prepared, raw.mask(raw.index == 1))


def test_recipient_unknown_category_refuses_against_donor_encoding():
    frame = rent.derive_us_housing_inputs(fixtures._frame())
    frame.table("household").loc[0, "state_fips"] = 99
    donor = rent.prepare_acs_rent_donor(fixtures._donor())
    with pytest.raises(ValueError, match="donor-unsupported"):
        rent.prepare_acs_rent_recipient(frame, donor_context=donor.context)


def test_rent_is_scattered_by_household_identity_when_head_order_differs(monkeypatch):
    base = fixtures._frame()
    tables = {entity: base.table(entity).copy() for entity in base.entities}
    # Person IDs stay ordered and unique; household membership need not follow
    # that order. A non-head row also proves placement uses original positions.
    tables["person"]["person_household_id"] = np.roll(np.arange(1, 21), 1)
    nonhead = tables["person"].iloc[[0]].copy()
    nonhead["person_id"] = 999
    nonhead["is_household_head"] = False
    tables["person"] = pd.concat([tables["person"], nonhead], ignore_index=True)
    tables["person"].index = pd.Index(np.arange(21) * 3 + 101, name="original_person")
    frame = rent.derive_us_housing_inputs(
        Frame(tables, base.schema, {"household": base.weights_for("household")})
    )
    donor = rent.prepare_acs_rent_donor(fixtures._donor())
    recipient = rent.prepare_acs_rent_recipient(frame, donor_context=donor.context)
    raw = pd.Series(
        recipient.predictors.index.to_numpy(dtype=float) * 100,
        index=recipient.predictors.index,
        name="rent",
    )
    expected = np.r_[frame.person["person_household_id"].to_numpy()[:20] * 100, 0.0]
    # The old person_rent[head_mask] = raw placed household 1's 100 on household
    # 20's head. Distinct draws expose that error without any model uncertainty.
    old_placement = np.r_[raw.to_numpy(), 0.0]
    assert old_placement[0] == 100.0 and expected[0] == 2_000.0
    np.testing.assert_array_equal(rent.finalize_acs_rent(recipient, raw), expected)
    np.testing.assert_array_equal(recipient.head_positions, np.r_[np.arange(1, 20), 0])
    assert recipient.person_index.equals(frame.person.index)
    assert recipient.predictors.index.tolist() == list(range(1, 21))

    class DistinctRent:
        def __init__(self, **kwargs):
            pass

        def fit(self, training, **kwargs):
            return self

        def predict(self, features):
            assert features.index.equals(raw.index)
            return raw.to_frame()

    monkeypatch.setattr(rent, "QRF", DistinctRent)
    np.testing.assert_array_equal(
        rent.impute_us_pre_subsidy_rent(frame, fixtures._donor(), seed=31), expected
    )


def test_actual_two_tree_composition_matches_pinned_legacy_raw_bits(monkeypatch):
    frame = rent.derive_us_housing_inputs(fixtures._frame())
    source = fixtures._donor()
    donor = rent.prepare_acs_rent_donor(source)
    recipient = rent.prepare_acs_rent_recipient(frame, donor_context=donor.context)
    fitted = QRF(seed=31, n_estimators=2).fit(
        donor.training,
        predictors=list(donor.predictors),
        targets=["rent"],
        weights="household_weight",
    )
    raw = fitted.predict(recipient.predictors)["rent"]
    expected = rent.finalize_acs_rent(recipient, raw)
    monkeypatch.setattr(rent, "QRF", QRF)
    direct = rent.impute_us_pre_subsidy_rent(frame, source, seed=31, n_estimators=2)
    assert direct.tobytes() == expected.tobytes()
    # Filled from the pre-edit db0c8418 baseline; never regenerate to pass.
    assert (
        hashlib.sha256(donor.training.to_numpy(dtype=np.float64).tobytes()).hexdigest()
        == "28f3f1c3387f3db870bcabdafbf8d1fe42b92c533cdb72be329c08d7641f10a5"
    )
    assert (
        hashlib.sha256(
            recipient.predictors.to_numpy(dtype=np.float64).tobytes()
        ).hexdigest()
        == "9be7415b377bf13c9db3fdb99d2c4e1851360bbcedcfb744b1d200e4e4f2d87c"
    )
    assert (
        hashlib.sha256(raw.to_numpy(dtype=np.float64).tobytes()).hexdigest()
        == LEGACY_RAW_SHA256
    )


LEGACY_RAW_SHA256 = "7d20c12e9807b1423ad87beaa5e512006f1142b2124930b2d2f5fefe5c2491d0"


class _PreparedSource(KernelBase):
    """A real tiny source population, not a derived-recipient graph adapter."""

    ref = "test.acs_rent.prepared_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.sources[0]])
        )


def _prepared_population(values, weights):
    values = values.copy()
    ids = np.arange(len(values), dtype=np.int64) + 1
    values.insert(0, "household_id", ids)
    return Frame(
        {
            "household": values,
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _prepared_source_node(name, frame):
    return Node(
        name,
        _PreparedSource.ref,
        sources=(name,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("household", column, str(frame.table("household")[column].dtype))
            for column in frame.table("household")
            if column != "household_id"
        ),
    )


def _store_prepared_source(store, frame):
    digest = hashlib.sha256(
        frame.table("household").to_json(orient="table").encode()
        + frame.weights_for("household").values.tobytes()
    ).hexdigest()
    return store.put_frame(digest, frame)


@pytest.mark.parametrize("regime", ("positive", "hurdle", "zero"))
def test_real_qrf_dataframe_fit_matches_typed_weight_graph_and_reuses_fit(
    tmp_path, monkeypatch, regime
):
    import microcosm.fit.graph_legacy_train as train_module

    donor_source = fixtures._donor()
    values = 1_000.125 + (np.arange(len(donor_source)) % 11) * 823.25
    donor_source["rent"] = (
        values
        if regime == "positive"
        else np.where(np.arange(len(values)) % 3 == 0, values, 0.0)
        if regime == "hurdle"
        else np.zeros(len(values))
    )
    donor = rent.prepare_acs_rent_donor(donor_source)
    recipient = rent.prepare_acs_rent_recipient(
        rent.derive_us_housing_inputs(fixtures._frame()), donor_context=donor.context
    )
    # Ordinary preexisting DataFrame fit/predict is the comparison authority.
    expected = (
        QRF(seed=31, n_estimators=2)
        .fit(
            donor.training,
            predictors=list(donor.predictors),
            targets=["rent"],
            weights="household_weight",
        )
        .predict(recipient.predictors)["rent"]
    )
    donor_population = _prepared_population(
        donor.training.drop(columns="household_weight"),
        donor.training["household_weight"].to_numpy(),
    )
    recipient_population = _prepared_population(
        recipient.predictors, np.ones(len(recipient.predictors))
    )
    fits = legacy_qrf_train_nodes(
        "rent.fit",
        population="donor",
        entity="household",
        predictors=donor.predictors,
        targets=("rent",),
        seed=31,
        n_estimators=2,
        phase="early_gap_fill",
    )
    applies = legacy_qrf_apply_nodes(
        "rent.apply",
        population="recipient",
        fit_nodes=fits,
        seed=31,
        phase="early_gap_fill",
    )
    compiled = compile_graph(
        Graph(
            country="us",
            sources=(
                SourceRef("donor", "frame-store"),
                SourceRef("recipient", "frame-store"),
            ),
            nodes=(
                _prepared_source_node("donor", donor_population),
                _prepared_source_node("recipient", recipient_population),
                *fits,
                *applies,
            ),
        )
    )
    kernels = KernelRegistry()
    for kernel in (_PreparedSource(), LegacyQRFTrainKernel(), LegacyQRFApplyKernel()):
        kernels.register(kernel)
    store = ContentStore(tmp_path)
    sources = {
        "donor": _store_prepared_source(store, donor_population),
        "recipient": _store_prepared_source(store, recipient_population),
    }
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    actual = read_raw_target(
        store.load_bytes(manifest.node(applies[0].id).opaque_artifacts["raw_draw"]),
        target="rent",
        index=recipient.predictors.index,
    )
    assert actual.tobytes() == expected.to_numpy().tobytes()
    assert (
        rent.finalize_acs_rent(
            recipient, pd.Series(actual, index=expected.index, name="rent")
        ).tobytes()
        == rent.finalize_acs_rent(recipient, expected).tobytes()
    )
    again = run_graph(
        compiled,
        sources=sources,
        store=ContentStore(tmp_path),
        kernels=kernels,
        resume="require",
    )
    assert again.key == manifest.key
    assert all(node.hit for node in again.nodes.values())

    def cannot_refit(*args, **kwargs):
        raise AssertionError("A recipient-only source change refitted the donor")

    monkeypatch.setattr(train_module, "fit_target", cannot_refit)
    changed_values = recipient.predictors.copy()
    changed_values["age"] += 1.0
    changed = _prepared_population(changed_values, np.ones(len(changed_values)))
    sources["recipient"] = _store_prepared_source(store, changed)
    rerun = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert rerun.node(fits[0].id).key == manifest.node(fits[0].id).key
    assert rerun.node(fits[0].id).hit
    assert not rerun.node(applies[0].id).hit
