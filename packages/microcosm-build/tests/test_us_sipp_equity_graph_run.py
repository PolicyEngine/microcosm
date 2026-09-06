"""Executor-level checks for the SIPP nominal December-2022 equity graph.

INVENTED data only. Real ``compile_graph``/``run_graph``, real ContentStore,
real QRF train and apply kernels, real source codecs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import us_sipp_equity_fixture as fx

import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_equity_graph as equity
import microcosm.graph as mgraph

TARGET = contract_module.TARGET_NAME
PREDICTOR = contract_module.PREDICTOR_NAME
ROSTER = contract_module.ROSTER_PREDICTOR_NAME


class _CountingDonor(equity.SippDonorCreateKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def run(self, context):
        self.calls += 1
        return super().run(context)


class _CountingRoster(equity.SippRecipientRosterKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def run(self, context):
        self.calls += 1
        return super().run(context)


class _CountingSupport(equity.SippSupportValidationKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def run(self, context):
        self.calls += 1
        return super().run(context)


def _counting(base):
    class Counting(base):
        def __init__(self):
            self.calls = 0

        def run(self, context):
            self.calls += 1
            return super().run(context)

    return Counting


class _Harness:
    """One invented accepted store, one recipient population, one graph."""

    def __init__(
        self, monkeypatch, tmp_path, sizes=(1, 2, 3, 4, 2), recipient_dir=None, **kwargs
    ):
        import microcosm.fit.graph_models as graph_models

        self.tmp_path = Path(tmp_path)
        root, self.contract, _, _ = fx.accepted_store(self.tmp_path)
        self.source_store = equity.prepare_source_store(
            root, self.tmp_path / "source-store", contract=self.contract
        )
        self.recipient = fx.recipient_source(
            recipient_dir or self.tmp_path / "recipient", sizes, **kwargs
        )
        # Explicit test-only construction route: the packaged loader is
        # replaced by the invented contract for the duration of the test. The
        # production codec registration and pins are untouched.
        monkeypatch.setattr(contract_module, "load_contract", lambda: self.contract)
        self.donor = _CountingDonor(contract=self.contract)
        self.roster = _CountingRoster(contract=self.contract)
        self.support = _CountingSupport(contract=self.contract)
        self.train = _counting(graph_models.QRFTrainKernel)()
        self.apply = _counting(graph_models.QRFApplyKernel)()
        self.sources = {
            equity.DONOR_POPULATION: self.source_store,
            equity.RECIPIENT_POPULATION: self.recipient,
        }

    def compiled(self, *, concept=TARGET, period=2022):
        return mgraph.compile_graph(
            equity.build_sipp_equity_graph(
                concept=concept, period=period, contract=self.contract
            )
        )

    def kernels(self):
        registry = mgraph.KernelRegistry()
        for kernel in (self.donor, self.roster, self.support, self.train, self.apply):
            registry.register(kernel)
        return registry

    def run(self, *, store_name="store", resume="auto", compiled=None):
        return mgraph.run_graph(
            compiled or self.compiled(),
            sources=self.sources,
            store=mgraph.ContentStore(self.tmp_path / store_name),
            kernels=self.kernels(),
            resume=resume,
        )

    def calls(self):
        return {
            "create_donor": self.donor.calls,
            "create_recipient": self.roster.calls,
            "support": self.support.calls,
            "fit": self.train.calls,
            "draw": self.apply.calls,
        }


def test_the_validator_is_a_real_compiler_derived_predecessor_of_draw(
    tmp_path, monkeypatch
):
    compiled = _Harness(monkeypatch, tmp_path).compiled()
    assert equity.VALIDATE_NODE in compiled.predecessors[equity.DRAW_NODE]
    assert compiled.owners[(equity.RECIPIENT_POPULATION, "household", PREDICTOR)] == (
        equity.VALIDATE_NODE
    )
    assert compiled.owners[(equity.RECIPIENT_POPULATION, "household", ROSTER)] == (
        equity.ROSTER_NODE
    )


def test_a_cold_run_fits_the_admitted_donors_and_draws_the_recipients(
    tmp_path, monkeypatch
):
    harness = _Harness(monkeypatch, tmp_path)
    manifest = harness.run()
    assert harness.calls() == {
        "create_donor": 1,
        "create_recipient": 1,
        "support": 1,
        "fit": 1,
        "draw": 1,
    }
    fit = manifest.nodes[equity.FIT_NODE].receipt
    assert fit["donor_rows"] == fx.ADMITTED_RESIDENCES
    # Zero-weight residences are removed before CREATE, so FIT excludes none.
    assert fit["excluded_zero_weight_rows"] == 0
    assert fit["source_weight_kind"] == "design"
    assert fit["seed"] == 0 and fit["n_estimators"] == 2
    assert fit["predictors"] == (PREDICTOR,) or list(fit["predictors"]) == [PREDICTOR]
    donor = manifest.nodes[equity.DONOR_POPULATION].receipt
    assert donor["admitted_residences"] == fx.ADMITTED_RESIDENCES
    assert donor["excluded_residences"] == 15 - fx.ADMITTED_RESIDENCES
    assert donor["weight_is_calibrated"] is False
    drawn = manifest.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    assert len(drawn) == 5
    assert np.isfinite(drawn.to_numpy()).all()


def test_a_warm_rerun_hits_every_node_and_executes_no_kernel(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    first = harness.run()
    second = _Harness.run.__get__(harness)(store_name="store")
    assert all(node.hit for node in second.nodes.values())
    assert harness.calls()["draw"] == 1
    assert second.key == first.key
    a = first.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    b = second.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_a_fresh_store_reproduces_the_same_node_keys_and_draws(tmp_path, monkeypatch):
    warm = _Harness(monkeypatch, tmp_path / "a").run()
    fresh = _Harness(monkeypatch, tmp_path / "b").run()
    assert {n: r.key for n, r in warm.nodes.items()} == {
        n: r.key for n, r in fresh.nodes.items()
    }
    a = warm.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    b = fresh.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_model_bytes_equality_is_reported_apart_from_semantic_equality(
    tmp_path, monkeypatch
):
    """Byte equality is observed platform-bitwise serialization behavior.

    The QRF header carries the pickle digest and the loader enforces it, so
    artifact byte equality implies pickle equality. It is reported separately
    from the semantic header and deterministic-draw assertions below.
    """
    import microcosm.fit._graph_qrf as graph_qrf

    warm = _Harness(monkeypatch, tmp_path / "a").run()
    fresh = _Harness(monkeypatch, tmp_path / "b").run()
    artifacts = [
        dict(manifest.nodes[equity.FIT_NODE].opaque_artifacts)
        for manifest in (warm, fresh)
    ]
    assert set(artifacts[0]) == {"model"}
    assert artifacts[0] == artifacts[1]  # content-addressed artifact identity
    payloads = []
    for entry, root in zip(
        artifacts, (tmp_path / "a" / "store", tmp_path / "b" / "store"), strict=True
    ):
        payloads.append(mgraph.ContentStore(root).load_bytes(entry["model"]))
    # Serialization byte equality, including the pickle digest in the header.
    assert payloads[0] == payloads[1]
    headers = [_model_header(graph_qrf, payload) for payload in payloads]
    # Semantic header equality, asserted independently of the bytes.
    for header in headers:
        assert header["predictors"] == [PREDICTOR]
        assert header["targets"] == [TARGET]
        assert header["source_weight_kind"] == "design"
        assert header["fit_weight_kind"] == "explicit"
        assert header["schema_version"] == 1
    # The pickle digest is serialization evidence, reported separately.
    assert headers[0]["pickle_sha256"] == headers[1]["pickle_sha256"]
    # Deterministic draw equality, asserted independently again.
    a = warm.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    b = fresh.populations[equity.RECIPIENT_POPULATION].table("household")[TARGET]
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def _model_header(graph_qrf, payload):
    """Read the model envelope header without weakening the real loader."""
    offset = len(graph_qrf._MAGIC)
    assert payload.startswith(graph_qrf._MAGIC)
    size = int.from_bytes(payload[offset : offset + 8], "big")
    return json.loads(payload[offset + 8 : offset + 8 + size])


def test_an_unsupported_recipient_size_refuses_and_leaves_draw_unexecuted(
    tmp_path, monkeypatch
):
    harness = _Harness(monkeypatch, tmp_path, sizes=(1, 2, 9))
    with pytest.raises(Exception, match="UNSUPPORTED_RECIPIENT_HOUSEHOLD_SIZE"):
        harness.run()
    assert harness.calls()["draw"] == 0
    assert harness.calls()["support"] == 1


def test_the_refused_validator_leaves_draw_unexecuted_on_a_warm_rerun(
    tmp_path, monkeypatch
):
    harness = _Harness(monkeypatch, tmp_path, sizes=(1, 2, 9))
    for _ in range(2):
        with pytest.raises(Exception, match="UNSUPPORTED_RECIPIENT_HOUSEHOLD_SIZE"):
            harness.run()
    assert harness.calls()["draw"] == 0
    assert harness.calls()["support"] == 2


def test_a_recipient_supplying_a_conflicting_household_size_refuses(
    tmp_path, monkeypatch
):
    directory = tmp_path / "recipient-conflict"
    with pytest.raises(
        Exception, match="RECIPIENT_ROSTER_COLUMN_SUPPLIED|SIZE_CONFLICT"
    ):
        _Harness(
            monkeypatch,
            tmp_path,
            sizes=(1, 2, 3),
            recipient_dir=directory,
            extra_column=(PREDICTOR, [5, 5, 5]),
        ).run()


def test_a_recipient_supplying_an_agreeing_household_size_is_accepted(
    tmp_path, monkeypatch
):
    harness = _Harness(
        monkeypatch,
        tmp_path,
        sizes=(1, 2, 3),
        recipient_dir=tmp_path / "recipient-agree",
        extra_column=(PREDICTOR, [1, 2, 3]),
    )
    manifest = harness.run()
    roster = manifest.nodes[equity.ROSTER_NODE].receipt
    assert list(roster["roster_sizes"]) == [1, 2, 3]
    assert roster["roster_column"] == ROSTER


def test_permuting_recipient_household_ids_invalidates_the_draw_cache(
    tmp_path, monkeypatch
):
    """Same length, same size multiset, different recipient identity."""
    base = _Harness(monkeypatch, tmp_path / "a", sizes=(1, 2, 3, 4, 2))
    first = base.run()
    permuted = _Harness(
        monkeypatch,
        tmp_path / "b",
        sizes=(2, 4, 3, 2, 1),
        household_ids=[900001, 900002, 900003, 900004, 900005],
    )
    second = permuted.run()
    assert first.nodes[equity.DRAW_NODE].key != second.nodes[equity.DRAW_NODE].key
    assert first.nodes[equity.ROSTER_NODE].key != second.nodes[equity.ROSTER_NODE].key
    assert permuted.calls()["draw"] == 1
    assert not second.nodes[equity.DRAW_NODE].hit


def test_a_changed_source_store_changes_the_donor_and_descendant_keys(
    tmp_path, monkeypatch
):
    base = _Harness(monkeypatch, tmp_path / "a")
    base.run()
    changed = _Harness(monkeypatch, tmp_path / "b")
    projection_key = changed.contract.source["projection_key"]
    obj = mgraph.ContentStore(changed.source_store).object_path(projection_key)
    (obj / "payload.bin").write_bytes(b"different bytes entirely")
    with pytest.raises(mgraph.NodeRejectedError):
        changed.run()
    assert changed.calls()["fit"] == 0


def test_a_changed_model_configuration_changes_every_node_key(tmp_path, monkeypatch):
    """The model section lives in the contract, so it moves the whole graph.

    Both runs use the same kernel classes, so the implementation hash is held
    fixed and the only moving part is the declared contract identity.
    """
    harness = _Harness(monkeypatch, tmp_path)
    first = harness.run()
    document = json.loads(json.dumps(dict(_plain(harness.contract.document))))
    document["model"]["n_estimators"] = 3
    other = contract_module._contract_from_document(document)
    compiled = mgraph.compile_graph(
        equity.build_sipp_equity_graph(concept=TARGET, period=2022, contract=other)
    )
    registry = mgraph.KernelRegistry()
    for kernel in (
        _CountingDonor(contract=other),
        _CountingRoster(contract=other),
        _CountingSupport(contract=other),
        harness.train,
        harness.apply,
    ):
        registry.register(kernel)
    second = mgraph.run_graph(
        compiled,
        sources=harness.sources,
        store=mgraph.ContentStore(harness.tmp_path / "store"),
        kernels=registry,
        resume="auto",
    )
    for node in (
        equity.DONOR_POPULATION,
        equity.ROSTER_NODE,
        equity.VALIDATE_NODE,
        equity.FIT_NODE,
        equity.DRAW_NODE,
    ):
        assert first.nodes[node].key != second.nodes[node].key
    # The warm store from the first contract satisfies none of them.
    assert not any(record.hit for record in second.nodes.values())


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def test_a_hand_built_node_cannot_bypass_the_contract_check(tmp_path, monkeypatch):
    """Manual node construction hits the same refusal inside the kernel."""
    harness = _Harness(monkeypatch, tmp_path)
    graph = equity.build_sipp_equity_graph(
        concept=TARGET, period=2022, contract=harness.contract
    )
    nodes = []
    for node in graph.nodes:
        if node.id == equity.DONOR_POPULATION:
            node = mgraph.Node(
                node.id,
                node.kernel,
                sources=node.sources,
                structural=node.structural,
                outputs=node.outputs,
                artifact_outputs=node.artifact_outputs,
                params={
                    "concept": TARGET,
                    "period": 2024,
                    "contract_identity": harness.contract.identity,
                },
                description=node.description,
            )
        nodes.append(node)
    hand_built = mgraph.Graph(
        country=graph.country, sources=graph.sources, nodes=tuple(nodes)
    )
    with pytest.raises(Exception, match="YEAR_BRIDGE_REQUIRED|UNSUPPORTED_PERIOD"):
        harness.run(compiled=mgraph.compile_graph(hand_built))
    assert harness.calls()["fit"] == 0
