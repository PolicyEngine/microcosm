"""Regressions for the root review of the SIPP nominal-equity consumer.

INVENTED data only. These exercise the two REQUEST_CHANGES findings and the
concretely-assigned close issues. Unlike the counting subclasses in
``test_us_sipp_equity_graph_run``, the identity closure below runs the ACTUAL
production kernel classes: a subclass defined in a test module would make
``source_hash(type(self))`` hash this file, which is not evidence about the
implementation identity of the shipped kernels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import microcosm.build.sipp_december_readback as readback
import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_equity_graph as equity
import microcosm.graph as mgraph
import us_sipp_equity_fixture as fx

TARGET = contract_module.TARGET_NAME
PREDICTOR = contract_module.PREDICTOR_NAME


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _production_kernels():
    """The shipped classes, not test subclasses: real implementation hashes."""
    import microcosm.fit.graph_models as graph_models

    registry = mgraph.KernelRegistry()
    for kernel in (
        equity.SippDonorCreateKernel(),
        equity.SippRecipientRosterKernel(),
        equity.SippSupportValidationKernel(),
        graph_models.QRFTrainKernel(),
        graph_models.QRFApplyKernel(),
    ):
        registry.register(kernel)
    return registry


class _Bench:
    """One invented accepted store and recipient population, production kernels."""

    def __init__(self, monkeypatch, tmp_path, sizes=(1, 2, 3, 4, 2), **kwargs):
        self.tmp_path = Path(tmp_path)
        root, self.contract, _, _ = fx.accepted_store(self.tmp_path / "accepted")
        self.accepted_root = root
        self.source_store = equity.prepare_source_store(
            root, self.tmp_path / "source-store", contract=self.contract
        )
        self.recipient = fx.recipient_source(
            self.tmp_path / "recipient", sizes, **kwargs
        )
        self.active = [self.contract]
        monkeypatch.setattr(contract_module, "load_contract", lambda: self.active[0])
        self.sources = {
            equity.DONOR_POPULATION: self.source_store,
            equity.RECIPIENT_POPULATION: self.recipient,
        }

    def compiled(self):
        return mgraph.compile_graph(
            equity.build_sipp_equity_graph(
                concept=TARGET, period=2022, contract=self.active[0]
            )
        )

    def run(self, *, store_name="store", resume="auto", sources=None):
        return mgraph.run_graph(
            self.compiled(),
            sources=sources or self.sources,
            store=mgraph.ContentStore(self.tmp_path / store_name),
            kernels=_production_kernels(),
            resume=resume,
        )

    def variant(self, mutate):
        """A contract document with one field changed, through the test route."""
        document = json.loads(json.dumps(_plain(self.contract.document)))
        mutate(document)
        return contract_module._contract_from_document(document)


# --------------------------------------------------------------------------
# Finding 1: the closed contract identity must be in graph node identity
# --------------------------------------------------------------------------
def test_the_contract_identity_is_declared_by_every_node_that_interprets_it():
    graph = equity.build_sipp_equity_graph(concept=TARGET, period=2022)
    identity = contract_module.load_contract().identity
    declared = {
        node.id: node.params.get("contract_identity")
        for node in graph.nodes
        if node.id
        in (equity.DONOR_POPULATION, equity.ROSTER_NODE, equity.VALIDATE_NODE)
    }
    assert declared == {
        equity.DONOR_POPULATION: identity,
        equity.ROSTER_NODE: identity,
        equity.VALIDATE_NODE: identity,
    }


def test_a_changed_source_pin_moves_every_key_and_refuses_the_warm_store(
    tmp_path, monkeypatch
):
    """The root counterexample, inverted.

    Changing only the contract's ``source.measurement_digest`` used to leave a
    warm store serving all five nodes under the unchanged donor key. It must
    now move every key that declares the identity, so nothing is a hit.
    """
    bench = _Bench(monkeypatch, tmp_path)
    cold = bench.run()
    assert all(record.hit is False for record in cold.nodes.values())

    other = bench.variant(
        lambda doc: doc["source"].__setitem__("measurement_digest", "0" * 64)
    )
    assert other.identity != bench.contract.identity
    bench.active[0] = other
    with pytest.raises(Exception) as raised:
        bench.run(store_name="store", resume="auto")
    chain, error = [], raised.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    # The refusal is the readback pin, reached by execution: no warm hit stood in.
    assert any("MEASUREMENT_DIGEST_PIN" in text for text in chain)

    # The declaration itself moved, which is what carries into every node key.
    declared = {
        node.id: node.params.get("contract_identity")
        for node in bench.compiled().graph.nodes
    }
    assert declared[equity.DONOR_POPULATION] == other.identity
    assert declared[equity.DONOR_POPULATION] != bench.contract.identity


def test_a_changed_admission_policy_moves_every_executed_node_key(
    tmp_path, monkeypatch
):
    """A contract change that readback still accepts, so both runs complete.

    The admission rule id is consumer policy, not a source pin, so this run
    reaches the executor and the moved keys are observed on real manifests.
    """
    bench = _Bench(monkeypatch, tmp_path)
    first = bench.run()
    bench.active[0] = bench.variant(
        lambda doc: doc["admission"].__setitem__(
            "rule_id", "sipp-december-equity-admission/2"
        )
    )
    second = bench.run(store_name="store", resume="auto")
    for node in (
        equity.DONOR_POPULATION,
        equity.ROSTER_NODE,
        equity.VALIDATE_NODE,
        equity.FIT_NODE,
        equity.DRAW_NODE,
    ):
        assert first.nodes[node].key != second.nodes[node].key
    # The warm store written under the old admission policy serves none of them.
    assert not any(record.hit for record in second.nodes.values())


def test_a_node_declaring_another_contract_is_refused_on_execution(
    tmp_path, monkeypatch
):
    """A stale declared identity is caught against the contract actually loaded."""
    bench = _Bench(monkeypatch, tmp_path)
    graph = equity.build_sipp_equity_graph(
        concept=TARGET, period=2022, contract=bench.contract
    )
    stale = bench.variant(
        lambda doc: doc["source"].__setitem__("measurement_digest", "1" * 64)
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
                params=dict(node.params) | {"contract_identity": stale.identity},
                description=node.description,
            )
        nodes.append(node)
    hand_built = mgraph.Graph(
        country=graph.country, sources=graph.sources, nodes=tuple(nodes)
    )
    with pytest.raises(Exception, match="CONTRACT_IDENTITY_MISMATCH"):
        mgraph.run_graph(
            mgraph.compile_graph(hand_built),
            sources=bench.sources,
            store=mgraph.ContentStore(bench.tmp_path / "hand-built"),
            kernels=_production_kernels(),
            resume="auto",
        )


def test_a_node_missing_the_contract_identity_param_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    graph = equity.build_sipp_equity_graph(
        concept=TARGET, period=2022, contract=bench.contract
    )
    nodes = []
    for node in graph.nodes:
        if node.id == equity.VALIDATE_NODE:
            node = mgraph.Node(
                node.id,
                node.kernel,
                population=node.population,
                inputs=node.inputs,
                outputs=node.outputs,
                artifact_inputs=node.artifact_inputs,
                params={},
                description=node.description,
            )
        nodes.append(node)
    with pytest.raises(Exception, match="SUPPORT_NODE_PARAMS"):
        mgraph.run_graph(
            mgraph.compile_graph(
                mgraph.Graph(
                    country=graph.country, sources=graph.sources, nodes=tuple(nodes)
                )
            ),
            sources=bench.sources,
            store=mgraph.ContentStore(bench.tmp_path / "no-identity"),
            kernels=_production_kernels(),
            resume="auto",
        )


def test_the_new_kernels_declare_the_numeric_dependencies_they_use():
    """numpy and pandas carry behavior for all three kernels, so they are pinned."""
    for kernel in (
        equity.SippDonorCreateKernel,
        equity.SippRecipientRosterKernel,
        equity.SippSupportValidationKernel,
    ):
        assert kernel.capabilities.dependencies == ("numpy", "pandas")
    first = equity.SippDonorCreateKernel().implementation_hash()
    assert isinstance(first, str) and len(first) == 64


def test_production_kernels_run_direct_cold_and_fresh_process_warm(
    tmp_path, monkeypatch
):
    """Direct, cold and a warm rerun in a NEW process: all hits, zero kernels.

    The warm arm runs in a subprocess so no in-memory contract cache, codec
    registration or kernel instance is shared with the cold arm.
    """
    bench = _Bench(monkeypatch, tmp_path)
    direct = equity.load_sipp_equity_donor_frame(
        bench.source_store, contract=bench.contract
    )
    cold = bench.run()
    assert len(direct.table("household")) == fx.ADMITTED_RESIDENCES

    script = Path(__file__).with_name("us_sipp_equity_fixture.py").parent
    program = f"""
import json, sys
sys.path[:0] = {json.dumps(sys.path)}
import microcosm.build.sipp_equity_contract as cm
import microcosm.build.sipp_equity_graph as equity
import microcosm.graph as mgraph
import microcosm.fit.graph_models as gm
import us_sipp_equity_fixture as fx
from pathlib import Path as _Path
root, contract, _, _ = fx.accepted_store(_Path({json.dumps(str(tmp_path / "accepted"))}))
cm.load_contract = lambda: contract
calls = {{"n": 0}}
registry = mgraph.KernelRegistry()
for kernel in (
    equity.SippDonorCreateKernel(),
    equity.SippRecipientRosterKernel(),
    equity.SippSupportValidationKernel(),
    gm.QRFTrainKernel(),
    gm.QRFApplyKernel(),
):
    original = kernel.run
    def counted(context, _original=original):
        calls["n"] += 1
        return _original(context)
    kernel.run = counted
    registry.register(kernel)
manifest = mgraph.run_graph(
    mgraph.compile_graph(
        equity.build_sipp_equity_graph(concept=cm.TARGET_NAME, period=2022,
                                       contract=contract)
    ),
    sources={{
        equity.DONOR_POPULATION: {json.dumps(str(bench.source_store))},
        equity.RECIPIENT_POPULATION: {json.dumps(str(bench.recipient))},
    }},
    store=mgraph.ContentStore({json.dumps(str(tmp_path / "store"))}),
    kernels=registry,
    resume="require",
)
print(json.dumps({{
    "all_hits": all(r.hit for r in manifest.nodes.values()),
    "kernel_calls": calls["n"],
    "keys": {{n: r.key for n, r in manifest.nodes.items()}},
}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(script), *sys.path, environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    warm = json.loads(completed.stdout.strip().splitlines()[-1])
    assert warm["all_hits"] is True
    assert warm["kernel_calls"] == 0
    assert warm["keys"] == {n: r.key for n, r in cold.nodes.items()}


# --------------------------------------------------------------------------
# Finding 2: supplied recipient sizes are compared exactly, before any cast
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "supplied,code",
    [
        ((1.5, 2.5, 3.5, 4.5, 2.5), "RECIPIENT_SIZE_NOT_INTEGER"),
        ((1.0, 2.0, 3.0, 4.0, 2.9), "RECIPIENT_SIZE_NOT_INTEGER"),
        (("nan", 2, 3, 4, 2), "RECIPIENT_SIZE_NOT_FINITE"),
        (("inf", 2, 3, 4, 2), "RECIPIENT_SIZE_NOT_FINITE"),
        (("True", "False", "True", "False", "True"), "RECIPIENT_SIZE_TYPE:bool"),
        (("one", "two", "three", "four", "two"), "RECIPIENT_SIZE_TYPE:"),
    ],
)
def test_a_supplied_size_that_is_not_an_exact_integer_count_is_refused(
    tmp_path, monkeypatch, supplied, code
):
    """The root's fractional counterexample and its neighbours.

    ``astype(np.int64)`` truncated 1.5 to 1 before comparison, so a fractional
    claim silently agreed with a one-person roster. Every one of these must be
    a clean refusal instead.
    """
    bench = _Bench(
        monkeypatch,
        tmp_path,
        sizes=(1, 2, 3, 4, 2),
        extra_column=(PREDICTOR, supplied),
    )
    with pytest.raises(Exception, match=code):
        bench.run()


def test_an_agreeing_supplied_size_is_still_accepted_checked_and_dropped(
    tmp_path, monkeypatch
):
    """A valid agreeing claim stays admissible; the derived roster stays the identity."""
    bench = _Bench(
        monkeypatch, tmp_path, sizes=(1, 2, 3, 4, 2), extra_column=(PREDICTOR, (1, 2, 3, 4, 2))
    )
    manifest = bench.run()
    household = manifest.populations[equity.RECIPIENT_POPULATION].table("household")
    roster = manifest.nodes[equity.ROSTER_NODE].receipt
    assert list(roster["roster_sizes"]) == [1, 2, 3, 4]
    assert np.array_equal(
        household[contract_module.ROSTER_PREDICTOR_NAME].to_numpy(),
        np.array([1, 2, 3, 4, 2], dtype=np.int64),
    )
    assert np.array_equal(
        household[PREDICTOR].to_numpy(), np.array([1, 2, 3, 4, 2], dtype=np.int64)
    )


def test_an_exactly_agreeing_float_size_claim_is_accepted(tmp_path, monkeypatch):
    """A float column whose values are exact integers is a valid agreeing claim."""
    bench = _Bench(
        monkeypatch,
        tmp_path,
        sizes=(1, 2, 3, 4, 2),
        extra_column=(PREDICTOR, (1.0, 2.0, 3.0, 4.0, 2.0)),
    )
    manifest = bench.run()
    household = manifest.populations[equity.RECIPIENT_POPULATION].table("household")
    assert household[PREDICTOR].dtype == np.dtype("int64")


def test_a_truncating_conflict_is_refused_rather_than_silently_agreed(
    tmp_path, monkeypatch
):
    """2.9 would truncate to 2 and agree with a two-person roster. It must not."""
    bench = _Bench(
        monkeypatch,
        tmp_path,
        sizes=(2, 2, 3, 4, 2),
        extra_column=(PREDICTOR, (2.9, 2, 3, 4, 2)),
    )
    with pytest.raises(Exception, match="RECIPIENT_SIZE_NOT_INTEGER"):
        bench.run()


# --------------------------------------------------------------------------
# Concretely-assigned close issues
# --------------------------------------------------------------------------
def test_scratch_equal_to_the_source_store_root_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    store = mgraph.ContentStore(bench.source_store)
    with pytest.raises(readback.ReadbackError, match="SCRATCH_INSIDE_SOURCE_STORE"):
        readback.read_sipp_december_projection(
            store, Path(bench.source_store), contract=bench.contract
        )


def test_a_symlinked_objects_directory_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    decoy = tmp_path / "decoy"
    (decoy / "objects").mkdir(parents=True)
    (decoy / "tmp").mkdir()
    for shard in (bench.source_store / "objects").iterdir():
        os.link if False else None
        target = decoy / "objects" / shard.name
        target.mkdir()
        for obj in shard.iterdir():
            (target / obj.name).mkdir()
            for entry in obj.iterdir():
                (target / obj.name / entry.name).write_bytes(entry.read_bytes())
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    (swapped / "tmp").mkdir()
    (swapped / "objects").symlink_to(decoy / "objects", target_is_directory=True)
    with pytest.raises(
        equity.EquityGraphError, match="SOURCE_STORE_OBJECTS_NOT_A_DIRECTORY"
    ):
        equity.load_sipp_equity_donor_frame(swapped, contract=bench.contract)


def test_an_object_parked_under_the_wrong_shard_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    objects = bench.source_store / "objects"
    key = bench.contract.source["receipt_key"]
    original = objects / key[:2] / key
    wrong = objects / ("zz" if key[:2] != "zz" else "yy")
    wrong.mkdir()
    original.rename(wrong / key)
    with pytest.raises(equity.EquityGraphError, match="SOURCE_STORE_SHARD_PLACEMENT"):
        equity.load_sipp_equity_donor_frame(
            bench.source_store, contract=bench.contract
        )


def test_a_duplicated_object_under_a_second_shard_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    objects = bench.source_store / "objects"
    key = bench.contract.source["receipt_key"]
    source_dir = objects / key[:2] / key
    duplicate = objects / "zz" / key
    duplicate.mkdir(parents=True)
    for entry in source_dir.iterdir():
        (duplicate / entry.name).write_bytes(entry.read_bytes())
    with pytest.raises(equity.EquityGraphError, match="SOURCE_STORE_SHARD_PLACEMENT"):
        equity.load_sipp_equity_donor_frame(
            bench.source_store, contract=bench.contract
        )


def test_the_prepared_copy_is_bounded_by_the_pinned_object_length(
    tmp_path, monkeypatch
):
    """A larger-than-pinned object refuses at the copy, not after materializing it."""
    bench = _Bench(monkeypatch, tmp_path)
    key = bench.contract.source["projection_key"]
    payload = bench.accepted_root / "objects" / key[:2] / key / "payload.bin"
    payload.write_bytes(payload.read_bytes() + b"\x00")
    with pytest.raises(equity.EquityGraphError, match="SOURCE_OBJECT_TOO_LARGE"):
        equity.prepare_source_store(
            bench.accepted_root, tmp_path / "bounded", contract=bench.contract
        )


def test_an_oversized_object_meta_refuses(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    key = bench.contract.source["receipt_key"]
    meta = bench.accepted_root / "objects" / key[:2] / key / "meta.json"
    meta.write_bytes(b" " * (equity._META_MAX_BYTES + 1))
    with pytest.raises(equity.EquityGraphError, match="SOURCE_OBJECT_TOO_LARGE"):
        equity.prepare_source_store(
            bench.accepted_root, tmp_path / "bounded-meta", contract=bench.contract
        )


def test_the_signed_target_conversion_accepts_zero_and_negative_values():
    """The signed target must not reuse the strictly-positive weight predicate."""
    from decimal import Decimal

    assert equity._checked_target_float(Decimal("0"), "TARGET_CONVERSION") == 0.0
    assert equity._checked_target_float(Decimal("-12345"), "TARGET_CONVERSION") == -12345.0
    with pytest.raises(equity.EquityGraphError, match="TARGET_CONVERSION"):
        equity._checked_target_float(Decimal("1e400"), "TARGET_CONVERSION")
    # The weight predicate is unchanged and still refuses a nonpositive weight.
    with pytest.raises(equity.EquityGraphError, match="WEIGHT_CONVERSION"):
        equity._checked_float(Decimal("0"), "WEIGHT_CONVERSION")


def test_the_signed_targets_reach_the_donor_frame_unchanged(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    frame = equity.load_sipp_equity_donor_frame(
        bench.source_store, contract=bench.contract
    )
    values = frame.table("household")[TARGET].to_numpy()
    assert np.isfinite(values).all()
    assert (values < 0).any() and (values == 0).any() and (values > 0).any()
