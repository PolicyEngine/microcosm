"""Opt-in disability continuation on genuine, wholly invented source owners.

The fitting fixture exercises the default enrichment terminal, not original-arm
placement, SPM policy choices, actual inputs, or scientific/release acceptance.
All literal changes precede source issuance. No owner or model is replaced.
"""

import hashlib
import inspect
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_asec_coverage_authentication import _changed_parent
from test_us_graph_atomic_survey_population import _support_payload
from test_us_graph_current_survey_other_disability_completion import (
    RealDiskFixturePatch,
)
from test_us_graph_puf55_canonical_donor import _original_sources
from test_us_graph_us_survey_enrichment import enrichment_source_arguments

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import current_asec_demographics as demographics
from microcosm.build.us_runtime import graph_us_other_disability_host as continuation
from microcosm.build.us_runtime import graph_us_survey_enrichment as host

fragment, values = continuation.fragment, continuation.values


def test_continuation_installs_frame_codec_without_mutating_predecessor(tmp_path):
    codecs = continuation.codecs
    previous_codecs = codecs.SourceCodecRegistry()
    previous_codecs.register_bytes("raw-bytes-v1", codecs.load_raw_bytes)
    previous = continuation.ContentStore(tmp_path / "store", codecs=previous_codecs)
    before = previous_codecs.as_mapping(), previous_codecs.as_bytes_mapping()
    store = continuation._continuation_store(previous)
    assert store is not previous and store.root == previous.root
    assert store.codecs is not previous_codecs
    assert store.codecs.get("frame-store") is codecs.load_frame_store
    assert store.codecs.get("raw-bytes-v1") is codecs.load_raw_bytes
    assert (previous_codecs.as_mapping(), previous_codecs.as_bytes_mapping()) == before
    assert "frame-store" not in previous_codecs.names()


@pytest.mark.parametrize(
    "cls",
    (
        continuation.parent.LegacyQRFTrainKernel,
        continuation.parent.LegacyQRFApplyMatrixKernel,
    ),
)
def test_only_known_stateless_kernel_collisions_are_reusable(cls):
    continuation._require_reusable_kernel(cls(), cls())


def test_stateful_same_class_and_hash_collision_refuses():
    class StatefulKernel:
        def __init__(self, owner):
            self.owner = owner

        def implementation_hash(self):
            return "same-code-different-owner"

    first, second = StatefulKernel(object()), StatefulKernel(object())
    assert type(first) is type(second)
    assert first.implementation_hash() == second.implementation_hash()
    with pytest.raises(ValueError, match="HOST_KERNEL_COLLISION"):
        continuation._require_reusable_kernel(first, second)


def disability_host_arguments(root, monkeypatch):
    """Add four explicit source cases to the maintained genuine host fixture."""
    patch = RealDiskFixturePatch(monkeypatch)
    arguments = enrichment_source_arguments(root, patch)
    folder = arguments["source_dir"] / "asec"
    parent, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent).frame.person
    literal = {
        105: (1, 6, 200),  # Affirmed eligible receipt, known dollars.
        106: (2, 0, 0),  # Fixture's age15 respondent, known nonreceipt.
        107: (1, 0, 0),  # Affirmed receipt without a resolved source/amount.
        108: (0, 0, 0),  # Under15: outside the question universe, not zero.
    }
    changed = {}
    for name in values.observed.AMOUNT_FIELDS:
        data = people[name].to_numpy(copy=True)
        for pid, (_, _, amount) in literal.items():
            data[people.person_id.eq(pid)] = amount if name == "DIS_VAL1" else 0
        changed[name] = data
    _changed_parent(parent, attachment, patch, changed)
    people = load_frame_checkpoint(parent).frame.person.set_index("PERIDNUM")
    coverage = values.observed.routing.coverage
    restoration = values.source.asec_native.restoration
    paths, pins = {}, []
    for year, member, archive, *_ in coverage._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if year == 2024:
            for name in values.observed.detail.READ_COLUMNS:
                if name not in raw:
                    raw[name] = "0"
            for i, record in raw.iterrows():
                pid = int(people.loc[record.PERIDNUM, "person_id"])
                receipt, code, amount = literal[pid]
                # Preserve every actual invented parent literal, including ANN
                # printed -1; do not turn a declared NIU source into raw zero.
                for name in values.observed.detail.RETAINED_MONEY_FIELDS:
                    value = people.loc[record.PERIDNUM, name]
                    assert np.isfinite(value) and value == np.floor(value)
                    raw.loc[i, name] = str(int(value))
                for name, value in (
                    ("DIS_YN", receipt),
                    ("DIS_SC1", code),
                    ("DIS_VAL1", amount),
                    ("DIS_SC2", 0),
                    ("DIS_VAL2", 0),
                ):
                    raw.loc[i, name] = str(value)
                for name in (
                    *values.observed.ALLOCATION_FIELDS,
                    *values.observed.TOPCODE_FIELDS,
                ):
                    raw.loc[i, name] = "0"
            raw.iloc[::-1].to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(raw),
                len(payload),
            )
        )
        paths[year] = path
    for owner in (coverage, restoration, demographics.demographic):
        patch.setattr(owner, "_MEMBER_PINS", tuple(pins))
    restored = root / "host-disability-restored-money"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=restored
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    return arguments


@pytest.mark.parametrize("enabled", [1, 0, None, "yes", np.bool_(True)])
def test_invalid_option_before_parent_access(enabled):
    with pytest.raises(ValueError, match="OTHER_DISABILITY_OPTION"):
        host.run_us_survey_enrichment(object(), other_disability_completion=enabled)


@pytest.mark.parametrize("seed", [None, False, -1, 1.5, "29", np.int64(29)])
def test_invalid_seed_before_parent_access(seed):
    with pytest.raises(ValueError, match="HOST_SEED"):
        host.run_us_survey_enrichment(
            object(), other_disability_completion=True, other_disability_seed=seed
        )


@pytest.mark.parametrize("trees", [False, 0, -1, 1.5])
def test_invalid_estimator_count_before_parent_access(trees):
    with pytest.raises(ValueError, match="ESTIMATORS"):
        host.run_us_survey_enrichment(
            object(),
            other_disability_completion=True,
            other_disability_seed=29,
            n_estimators=trees,
        )


def test_disabled_seed_refused_and_default_api_preserved(monkeypatch):
    parameters = inspect.signature(host.run_us_survey_enrichment).parameters
    assert parameters["other_disability_completion"].default is False
    assert parameters["other_disability_seed"].default is None
    with pytest.raises(ValueError, match="OTHER_DISABILITY_DISABLED_SEED"):
        host.run_us_survey_enrichment(object(), other_disability_seed=29)
    seen = []

    def stop(run, **options):
        seen.append(options)
        raise RuntimeError("stopped before source admission")

    monkeypatch.setattr(host, "_construct", stop)
    with pytest.raises(RuntimeError, match="stopped before source admission"):
        host.run_us_survey_enrichment(object(), original_application_seed=73)
    assert seen[0]["original_application_seed"] == 73
    assert "other_disability_seed" not in seen[0]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("other-disability-host")
    with pytest.MonkeyPatch.context() as patch:
        arguments = disability_host_arguments(root, patch)
        payload, source_ids = _support_payload()
        support = root / "invented-block-support.npz"
        support.write_bytes(payload)
        config = host.parent.financial.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        financial = host.parent.financial.run_atomic_survey_financial(
            **arguments,
            geography_config=config,
            store_root=root / "store",
            demographic_conditioning=True,
            n_estimators=2,
            return_values=True,
        )
        definition, paths, _ = _original_sources(
            root / "original-puf", full_finalization_support=True
        )
        parent = host.parent.run_survey_puf55(
            financial,
            donor_sources=paths,
            fixture_definition=definition,
            seed=578,
            n_estimators=2,
            zero_atol=0,
        )
        cold = host.run_us_survey_enrichment(
            parent,
            n_estimators=2,
            full_original_amount_donors=False,
            other_disability_completion=True,
            other_disability_seed=29,
        )
        yield parent, cold, host._ISSUED[id(cold)][1]


def test_genuine_cold_preserves_population_and_source_knownness(completed):
    parent, cold, boundary = completed
    assert type(cold) is host.SurveyEnrichmentRun and cold.parent_run is parent
    host.check_survey_enrichment_run(cold)
    assert boundary.receiving_terminal.id == fragment.ATTACH_NODE
    assert boundary.receiving_terminal.population is not None
    assert set(dict(boundary.paths)) - set(dict(boundary.predecessor.sources)) == {
        fragment.SOURCE_NAME
    }
    for source in cold.compiled.graph.sources:
        assert callable(cold.store.codecs.get(source.codec))
    before, after = boundary.predecessor.population.frame, cold.population.frame
    assert all(cold.manifest.node(n).hit for n in boundary.predecessor.compiled.order)
    assert not any(cold.manifest.node(n.id).hit for n in boundary.nodes)
    for entity in before.entities:
        columns = before.table(entity).columns.difference([values.observed.OUTPUT])
        pd.testing.assert_frame_equal(
            before.table(entity).loc[:, columns],
            after.table(entity).loc[:, columns],
            check_exact=True,
        )
    for entity in before.weighted_entities:
        assert before.weights_for(entity).kind is after.weights_for(entity).kind
        np.testing.assert_array_equal(
            before.weights_for(entity).values, after.weights_for(entity).values
        )
    pd.testing.assert_series_equal(before.strata, after.strata)
    assert before.mass_log == after.mass_log
    prefix = boundary.predecessor.population.mass_ledger
    assert cold.population.mass_ledger[: len(prefix)] == prefix
    assert len(cold.population.mass_ledger) == len(prefix) + 1
    record = cold.population.mass_ledger[-1]
    assert record.node_id == fragment.VERSION_NODE
    assert record.before_total == record.after_total
    assert record.before_by_stratum == record.after_by_stratum
    q = boundary.fragment.qualified
    assert q.evidence["full_original_asec_rows"] == 4
    assert q.evidence["eligible_donors"] == 2
    assert q.evidence["excluded_affirmed_receipt_unknown_amount_rows"] == 1
    source = q.selected_source.person.set_index("native_person_id")
    assert source.loc[105, values.observed.AMOUNT_COLUMN] == 200
    assert source.loc[106, values.observed.AMOUNT_COLUMN] == 0
    assert source.loc[[107, 108], values.observed.AMOUNT_COLUMN].isna().all()
    asec = after.person[values.provenance.support_channel_column("person")].eq("asec")
    originals = after.person[values.provenance.support_source_id_column("person")]
    selected = q.selected_source.person.reindex(originals[asec].to_numpy())
    np.testing.assert_array_equal(
        after.person.loc[asec, values.observed.OUTPUT].to_numpy(),
        selected[values.observed.AMOUNT_COLUMN].to_numpy(
            dtype="float64", na_value=np.nan
        ),
    )
    acs_applicable = ~asec & after.person.age.ge(15)
    assert after.person.loc[acs_applicable, values.CANONICAL_KNOWN].all()
    assert after.person.loc[acs_applicable, values.VALUE_ORIGIN].eq("modeled").all()
    assert not after.person.loc[
        ~asec, values.observed.attached_name(values.observed.KNOWN_COLUMN)
    ].any()
    assert (
        after.person.loc[after.person.age.lt(15), values.observed.OUTPUT].isna().all()
    )
    receipt = json.loads(cold.receipt)
    assert receipt["other_disability_completion"]["full_original_donors"] is True
    assert (
        receipt["other_disability_completion"]["reporting_age_minimum"]
        == values.observed.REPORTING_AGE
    )
    assert (
        receipt["other_disability_completion"]["scientific_qualification"] == "pending"
    )
    assert receipt["release_eligible"] is False
    for node_id, node in boundary.predecessor.manifest.nodes.items():
        for name, key in node.opaque_artifacts.items():
            assert cold.store.load_bytes(
                cold.manifest.node(node_id).opaque_artifacts[name]
            ) == boundary.predecessor.store.load_bytes(key)


def test_required_replay_has_no_qrf_fit_or_apply(completed):
    parent, cold, _ = completed
    previous, entered = sys.getprofile(), []

    def no_qrf_fit_or_apply(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == "call" and frame.f_code in (
            fragment.qrf_target.fit_target.__code__,
            fragment.qrf_target.apply_target.__code__,
        ):
            entered.append(frame.f_code.co_name)
            raise AssertionError(
                "required host replay called QRF fit_target/apply_target"
            )

    try:
        sys.setprofile(no_qrf_fit_or_apply)
        warm = host.run_us_survey_enrichment(
            parent,
            n_estimators=2,
            resume="require",
            other_disability_completion=True,
            other_disability_seed=29,
        )
    finally:
        sys.setprofile(previous)
    assert not entered and all(n.hit for n in warm.manifest.nodes.values())
    assert cold.manifest.key == warm.manifest.key
    host.physical.replay.same_replayed_population(cold.population, warm.population)
    host.check_survey_enrichment_run(warm)


def test_retained_predecessor_and_terminal_binding_mutations_refused(completed):
    _, cold, boundary = completed
    person = boundary.predecessor.population.frame.person
    old = person.iloc[0]["age"]
    try:
        person.loc[person.index[0], "age"] = old + 1
        with pytest.raises(ValueError):
            boundary.pure()
    finally:
        person.loc[person.index[0], "age"] = old
    original = boundary.terminal_binding
    try:
        boundary.terminal_binding = (*original[:3], original[3] + b" ")
        with pytest.raises(ValueError, match="HOST_TERMINAL_CHANGED"):
            boundary.pure()
    finally:
        boundary.terminal_binding = original
    host.check_survey_enrichment_run(cold)


def test_inherited_dispatch_and_wrapper_mutation_refused(completed):
    _, cold, boundary = completed
    node = boundary.predecessor.compiled.graph.nodes[0]
    wrapper = cold.kernels.get(node.kernel)
    assert wrapper.kernel is boundary.predecessor.kernels.get(node.kernel)
    with pytest.raises(ValueError, match="INHERITED_COLD_DISPATCH"):
        wrapper.run(SimpleNamespace(node=node))
    inherited = wrapper.inherited
    try:
        wrapper.inherited = frozenset()
        with pytest.raises(ValueError, match="HOST_KERNEL_CHANGED"):
            wrapper.implementation_hash()
        with pytest.raises(ValueError, match="HOST_BOUND_REGISTRY_CHANGED"):
            boundary.pure()
    finally:
        wrapper.inherited = inherited
    host.check_survey_enrichment_run(cold)
