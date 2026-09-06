"""Evidence persistence for the SIPP nominal-equity consumer.

INVENTED data only. These close the Fable review's first finding: coverage,
the exact ``Decimal`` evidence and the household/SSUID lineage were reachable
only through ``Frame.metadata``, which ``ContentStore`` replay drops, so a
warm-hit donor population carried none of it.

Like ``test_us_sipp_equity_review_fixes``, and unlike the counting subclasses
in ``test_us_sipp_equity_graph_run``, the identity and replay arms here run the
ACTUAL production kernel classes: a subclass defined in a test module makes
``source_hash(type(self))`` hash this file, which is not evidence about the
shipped implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import us_sipp_equity_fixture as fx

import microcosm.build.sipp_december_readback as readback
import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_equity_graph as equity
import microcosm.graph as mgraph
from microcosm.graph.canonical import canonical_json, sha256_domain

#: This test file lives at ``<worktree>/packages/microcosm-build/tests/``, so
#: the checkout the warm subprocess must have loaded its modules from is four
#: levels up. Nothing here reads the repository through Git.
_WORKTREE = Path(__file__).resolve().parents[3]

TARGET = contract_module.TARGET_NAME
PREDICTOR = contract_module.PREDICTOR_NAME
ROSTER = contract_module.ROSTER_PREDICTOR_NAME
DONOR = equity.DONOR_POPULATION


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _json(value):
    """A plain, JSON-normal copy: receipts come back as frozen mappings."""
    return json.loads(canonical_json(_plain(value)))


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
        store_kwargs = {
            name: kwargs.pop(name)
            for name in ("rows", "counts", "expected", "ssuid_digest", "genuine_shaped")
            if name in kwargs
        }
        root, self.contract, _, _ = fx.accepted_store(
            self.tmp_path / "accepted", **store_kwargs
        )
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
            DONOR: self.source_store,
            equity.RECIPIENT_POPULATION: self.recipient,
        }

    def compiled(self):
        return mgraph.compile_graph(
            equity.build_sipp_equity_graph(
                concept=TARGET, period=2022, contract=self.active[0]
            )
        )

    def run(self, *, store_name="store", resume="auto", kernels=None):
        return mgraph.run_graph(
            self.compiled(),
            sources=self.sources,
            store=mgraph.ContentStore(self.tmp_path / store_name),
            kernels=kernels or _production_kernels(),
            resume=resume,
        )

    def donor_frame(self):
        return equity.load_sipp_equity_donor_frame(
            self.source_store, contract=self.contract
        )

    def variant(self, mutate):
        document = json.loads(json.dumps(_plain(self.contract.document)))
        mutate(document)
        return contract_module._contract_from_document(document)


def _strings_and_numbers(value, strings, numbers):
    """Collect every string and every integer anywhere in a JSON-shaped value."""
    if isinstance(value, dict):
        for key, item in value.items():
            strings.add(key)
            _strings_and_numbers(item, strings, numbers)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _strings_and_numbers(item, strings, numbers)
    elif isinstance(value, str):
        strings.add(value)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, int):
        numbers.add(value)


# --------------------------------------------------------------------------
# The premise Fable identified
# --------------------------------------------------------------------------
def test_content_store_replay_drops_frame_metadata(tmp_path, monkeypatch):
    """Why the evidence artifact has to exist at all.

    The store's frame writer persists tables, weights and strata. It does not
    persist ``Frame.metadata``, so everything the codec attached to the donor
    Frame is gone the moment a run is served from cache.
    """
    bench = _Bench(monkeypatch, tmp_path)
    frame = bench.donor_frame()
    assert frame.metadata["coverage"] and frame.metadata["lineage"]["households"]

    store = mgraph.ContentStore(tmp_path / "replay")
    store.put_frame("a" * 64, frame, node_key="b" * 64)
    replayed = store.load_frame("a" * 64)
    assert dict(replayed.metadata) == {}
    assert len(replayed.table("household")) == len(frame.table("household"))


# --------------------------------------------------------------------------
# Coverage and admission travel in the receipt, without identifiers
# --------------------------------------------------------------------------
def test_the_donor_receipt_carries_the_full_coverage_report(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    receipt = bench.run().nodes[DONOR].receipt
    metadata = bench.donor_frame().metadata
    coverage = _json(receipt["coverage"])
    assert coverage == _json(metadata["coverage"])
    assert coverage["roster"]["december_residences"] == fx.DENOMINATOR_RESIDENCES
    assert coverage["household_weights"]["unknown_count"] == fx.EXCLUDED_UNKNOWN_WEIGHT
    assert coverage["household_weights"]["zero_count"] == fx.EXCLUDED_ZERO_WEIGHT
    assert (
        coverage["target"]["quality_observations"][
            "component_derived_allocation_unresolved"
        ]
        > 0
    )
    assert "TVEH_NUM" in coverage["diagnostics"]
    assert coverage["vehicle_count_is_not_an_admission_requirement"] is True


def test_the_donor_receipt_carries_admission_without_the_excluded_keys(
    tmp_path, monkeypatch
):
    bench = _Bench(monkeypatch, tmp_path)
    receipt = bench.run().nodes[DONOR].receipt
    admission = receipt["admission"]
    assert set(admission) == {
        "rule_id",
        "denominator_residences",
        "admitted_residences",
        "excluded_residences",
        "excluded_reasons_exclusive",
        "reconciliation",
    }
    assert "excluded" not in admission
    assert admission["denominator_residences"] == fx.DENOMINATOR_RESIDENCES
    assert admission["admitted_residences"] == fx.ADMITTED_RESIDENCES
    assert dict(admission["excluded_reasons_exclusive"]) == {
        "target_unknown": fx.EXCLUDED_UNKNOWN_TARGET,
        "weight_not_positive": fx.EXCLUDED_ZERO_WEIGHT,
        "weight_unknown": fx.EXCLUDED_UNKNOWN_WEIGHT,
    }
    # The full admission, with its per-residence keys, is still built.
    full = bench.donor_frame().metadata["admission"]
    assert len(full["excluded"]) == fx.DENOMINATOR_RESIDENCES - fx.ADMITTED_RESIDENCES


def test_no_source_identifier_reaches_any_node_receipt(tmp_path, monkeypatch):
    """Receipts are rendered verbatim into the explain report, so they stay clean."""
    bench = _Bench(monkeypatch, tmp_path)
    manifest = bench.run()
    identifiers = fx.source_identifiers()
    strings: set[str] = set()
    numbers: set[int] = set()
    for record in manifest.nodes.values():
        _strings_and_numbers(_plain(record.receipt), strings, numbers)
    assert not strings & identifiers["ssuids"]
    assert not numbers & identifiers["residences"]


def test_no_source_identifier_reaches_the_public_explain_report(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    compiled = bench.compiled()
    manifest = bench.run()
    page = mgraph.explain_html(compiled, manifest)
    for ssuid in fx.source_identifiers()["ssuids"]:
        assert ssuid not in page
    document = json.loads(_evidence(manifest, bench.tmp_path / "store"))
    for donor in document["donors"]:
        assert donor["exact_source_weight"] not in page
    # The report names and binds the evidence; it never carries its payload.
    assert manifest.nodes[DONOR].receipt["evidence"]["sha256"] in page
    assert "donor_evidence" in page
    assert '"donors"' not in page


# --------------------------------------------------------------------------
# The typed evidence artifact, bound in the production graph path
# --------------------------------------------------------------------------
def test_the_production_graph_declares_the_evidence_artifact():
    """The packaged contract's graph, not a test-only construction."""
    graph = equity.build_sipp_equity_graph(concept=TARGET, period=2022)
    donor = graph.node(DONOR)
    assert donor.artifact_outputs == (
        mgraph.ArtifactOutput("donor_support", equity.DONOR_SUPPORT_TYPE),
        mgraph.ArtifactOutput("donor_evidence", equity.DONOR_EVIDENCE_TYPE),
    )
    assert equity.DONOR_EVIDENCE_TYPE.name == "microcosm.build.sipp_equity_evidence"


def _evidence(manifest, store_root):
    key = dict(manifest.nodes[DONOR].opaque_artifacts)["donor_evidence"]
    return mgraph.ContentStore(store_root).load_bytes(key)


def test_the_evidence_artifact_carries_the_exact_decimals_and_the_lineage(
    tmp_path, monkeypatch
):
    bench = _Bench(monkeypatch, tmp_path)
    manifest = bench.run()
    document = json.loads(_evidence(manifest, bench.tmp_path / "store"))
    metadata = bench.donor_frame().metadata

    assert document["format"] == equity.EVIDENCE_FORMAT
    assert document["consumer_contract_identity"] == bench.contract.identity
    assert len(document["donors"]) == fx.ADMITTED_RESIDENCES
    assert [d["exact_source_weight"] for d in document["donors"]] == list(
        metadata["exact_source_weights"]
    )
    assert [d["exact_source_target"] for d in document["donors"]] == list(
        metadata["exact_source_targets"]
    )
    # The exact strings are the source Decimals, not a float round trip.
    weights = manifest.populations[DONOR].weights_for("household").values
    assert [float(d["exact_source_weight"]) for d in document["donors"]] == list(
        weights
    )
    assert any("." in d["exact_source_weight"] for d in document["donors"])
    assert any(d["exact_source_target"].startswith("-") for d in document["donors"])

    identifiers = fx.source_identifiers()
    assert {d["ssuid"] for d in document["donors"]} <= identifiers["ssuids"]
    assert {d["residence"] for d in document["donors"]} <= identifiers["residences"]
    assert len(document["people"]) == len(manifest.populations[DONOR].entity("person"))
    assert {p["pnum"] for p in document["people"]} <= identifiers["pnums"]
    # One SSUID with two December residences stays two donors.
    repeated = [d for d in document["donors"] if d["ssuid"] == "00000010"]
    assert {d["residence"] for d in repeated} == {100010, 100011}


def test_the_evidence_lineage_digest_is_recomputable_from_the_artifact(
    tmp_path, monkeypatch
):
    """The digest in the receipt is derivable from the private artifact alone."""
    bench = _Bench(monkeypatch, tmp_path)
    manifest = bench.run()
    document = json.loads(_evidence(manifest, bench.tmp_path / "store"))
    lineage = {
        "grain": document["grain"],
        "households": [
            {
                "household_id": d["household_id"],
                "ssuid": d["ssuid"],
                "residence": d["residence"],
            }
            for d in document["donors"]
        ],
        "people": document["people"],
    }
    recomputed = sha256_domain(equity.LINEAGE_DOMAIN, canonical_json(lineage))
    assert recomputed == document["lineage_digest"]
    assert recomputed == manifest.nodes[DONOR].receipt["lineage_digest"]


def test_the_receipt_binds_the_evidence_it_does_not_contain(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    manifest = bench.run()
    payload = _evidence(manifest, bench.tmp_path / "store")
    evidence = manifest.nodes[DONOR].receipt["evidence"]
    assert evidence["sha256"] == hashlib.sha256(payload).hexdigest()
    assert evidence["bytes"] == len(payload)
    assert evidence["format"] == equity.EVIDENCE_FORMAT
    assert evidence["type"] == equity.DONOR_EVIDENCE_TYPE.name
    assert evidence["donor_records"] == fx.ADMITTED_RESIDENCES
    # The validator's own receipt carries the same digest, through the support
    # artifact it actually reads.
    support = manifest.nodes[equity.VALIDATE_NODE].receipt
    assert support["donor_evidence_sha256"] == evidence["sha256"]


def test_the_evidence_document_refuses_to_exceed_its_cap(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    document = equity.evidence_document(bench.donor_frame().metadata, bench.contract)
    monkeypatch.setattr(equity, "_EVIDENCE_MAX_BYTES", 16)
    with pytest.raises(equity.EquityGraphError, match="EVIDENCE_ARTIFACT_TOO_LARGE"):
        equity._encode_evidence(document)


# --------------------------------------------------------------------------
# Cold versus fresh-process warm
# --------------------------------------------------------------------------
_WARM_PROGRAM = """
import hashlib, json, sys
sys.path[:0] = {syspath}
import microcosm.build.sipp_equity_contract as cm
import microcosm.build.sipp_equity_graph as equity
import microcosm.graph as mgraph
import microcosm.fit.graph_models as gm
import us_sipp_equity_fixture as fx
from pathlib import Path

root, contract, _, _ = fx.accepted_store(Path({accepted}))
cm.load_contract = lambda: contract
calls = {{"n": 0}}


def plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {{k: plain(v) for k, v in value.items()}}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


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

store = mgraph.ContentStore({store})
manifest = mgraph.run_graph(
    mgraph.compile_graph(
        equity.build_sipp_equity_graph(
            concept=cm.TARGET_NAME, period=2022, contract=contract
        )
    ),
    sources={{
        equity.DONOR_POPULATION: {source_store},
        equity.RECIPIENT_POPULATION: {recipient},
    }},
    store=store,
    kernels=registry,
    resume="require",
)
donor = manifest.nodes[equity.DONOR_POPULATION]
evidence = store.load_bytes(dict(donor.opaque_artifacts)["donor_evidence"])
support = store.load_bytes(dict(donor.opaque_artifacts)["donor_support"])
worktree = Path({worktree}).resolve()
modules = {{}}
for name, module in sorted(sys.modules.items()):
    if not (name == "microcosm" or name.startswith("microcosm.")):
        continue
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    resolved = Path(filename).resolve()
    inside = resolved.is_relative_to(worktree)
    modules[name] = {{
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "path": str(resolved),
        "inside_worktree": inside,
        "relative_path": str(resolved.relative_to(worktree)) if inside else None,
    }}
print(json.dumps({{
    "all_hits": all(r.hit for r in manifest.nodes.values()),
    "kernel_calls": calls["n"],
    "keys": {{n: r.key for n, r in manifest.nodes.items()}},
    "donor_receipt": plain(donor.receipt),
    "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
    "evidence_bytes": len(evidence),
    "evidence": json.loads(evidence),
    "support": json.loads(support),
    "draws": list(
        manifest.populations[equity.RECIPIENT_POPULATION]
        .table("household")[cm.TARGET_NAME]
        .to_numpy()
    ),
    "modules": modules,
    "module_count": len(modules),
    "worktree": str(worktree),
}}))
"""


def _fresh_process_warm(bench, tmp_path):
    tests = Path(__file__).parent
    program = _WARM_PROGRAM.format(
        syspath=json.dumps(sys.path),
        accepted=json.dumps(str(bench.tmp_path / "accepted")),
        store=json.dumps(str(bench.tmp_path / "store")),
        source_store=json.dumps(str(bench.source_store)),
        recipient=json.dumps(str(bench.recipient)),
        worktree=json.dumps(str(_WORKTREE)),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tests), *sys.path, environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_cold_and_fresh_process_warm_evidence_and_receipts_are_identical(
    tmp_path, monkeypatch
):
    """The whole point: a warm hit in a NEW process still has the evidence.

    Nothing is shared with the cold arm — no contract cache, no codec
    registration, no kernel instance, no interpreter. Every node is a store
    hit, no kernel body runs, and the persisted coverage, admission, exact
    Decimal strings and lineage come back byte for byte.
    """
    bench = _Bench(monkeypatch, tmp_path)
    cold = bench.run()
    assert not any(record.hit for record in cold.nodes.values())
    cold_evidence = _evidence(cold, bench.tmp_path / "store")
    cold_support = mgraph.ContentStore(bench.tmp_path / "store").load_bytes(
        dict(cold.nodes[DONOR].opaque_artifacts)["donor_support"]
    )

    warm = _fresh_process_warm(bench, tmp_path)

    assert warm["all_hits"] is True
    assert warm["kernel_calls"] == 0
    assert warm["keys"] == {n: r.key for n, r in cold.nodes.items()}
    assert warm["evidence_sha256"] == hashlib.sha256(cold_evidence).hexdigest()
    assert warm["evidence_bytes"] == len(cold_evidence)
    assert canonical_json(warm["evidence"]) == cold_evidence
    assert canonical_json(warm["support"]) == cold_support
    assert warm["donor_receipt"] == _json(cold.nodes[DONOR].receipt)
    # The values Fable said were unreachable on a warm hit, reached.
    assert warm["donor_receipt"]["coverage"] == _json(
        cold.nodes[DONOR].receipt["coverage"]
    )
    assert warm["donor_receipt"]["admission"]["reconciliation"]["agrees"] is True
    assert len(warm["evidence"]["donors"]) == fx.ADMITTED_RESIDENCES
    assert warm["evidence"]["coverage"]["roster"]["december_residences"] == (
        fx.DENOMINATOR_RESIDENCES
    )
    cold_draws = cold.populations[equity.RECIPIENT_POPULATION].table("household")[
        TARGET
    ]
    assert np.array_equal(np.array(warm["draws"]), cold_draws.to_numpy())
    # Every microcosm module the warm process loaded came from this checkout.
    # The subprocess reports each module's resolved path, so "from this
    # checkout" is asserted rather than implied by a digest of the right
    # length: the path must be inside this worktree, and the bytes at that
    # path must still hash here to the digest the subprocess reported.
    assert warm["module_count"] > 0
    assert warm["worktree"] == str(_WORKTREE)
    assert all(entry["inside_worktree"] for entry in warm["modules"].values())
    for entry in warm["modules"].values():
        loaded = _WORKTREE / entry["relative_path"]
        assert loaded.is_file()
        assert hashlib.sha256(loaded.read_bytes()).hexdigest() == entry["sha256"]


def test_direct_cold_and_warm_donor_evidence_agree(tmp_path, monkeypatch):
    """The codec called directly and the graph's own CREATE produce one document."""
    bench = _Bench(monkeypatch, tmp_path)
    direct = equity.evidence_document(bench.donor_frame().metadata, bench.contract)
    cold = bench.run()
    stored = json.loads(_evidence(cold, bench.tmp_path / "store"))
    assert stored == json.loads(canonical_json(direct))
    warm = bench.run()
    assert all(record.hit for record in warm.nodes.values())
    assert _evidence(warm, bench.tmp_path / "store") == _evidence(
        cold, bench.tmp_path / "store"
    )


# --------------------------------------------------------------------------
# selected_ssuid_digest: recomputed, not merely pinned
# --------------------------------------------------------------------------
def test_the_selected_ssuid_digest_uses_the_producers_own_selector(
    tmp_path, monkeypatch
):
    import microcosm.build.sipp_december_source as source_module

    bench = _Bench(monkeypatch, tmp_path)
    result = readback.read_sipp_december_projection(
        mgraph.ContentStore(bench.source_store),
        _scratch(tmp_path),
        contract=bench.contract,
    )
    expected = source_module._key(
        "sipp-december-ssuid-selection/1",
        source_module._selected_ssuids(
            tuple(h.key for h in result.households), "all_ssuids"
        ),
    )
    assert result.selected_ssuid_digest == expected
    assert expected == bench.contract.source["selected_ssuid_digest"]
    # The stored receipt's own field, read straight off the closed object.
    stored = json.loads(
        (
            Path(
                mgraph.ContentStore(bench.source_store).object_path(
                    bench.contract.source["receipt_key"]
                )
            )
            / "payload.json"
        ).read_bytes()
    )
    assert stored["selected_ssuid_digest"] == expected
    # It reaches the receipt, and through it the manifest.
    assert bench.run().nodes[DONOR].receipt["selected_ssuid_digest"] == expected


def _scratch(tmp_path):
    path = Path(tmp_path) / "ssuid-scratch"
    path.mkdir(exist_ok=True)
    return path


def test_a_contract_pinning_another_ssuid_digest_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    other = bench.variant(
        lambda doc: doc["source"].__setitem__("selected_ssuid_digest", "0" * 64)
    )
    with pytest.raises(readback.ReadbackError, match="SELECTED_SSUID_DIGEST_PIN"):
        readback.read_sipp_december_projection(
            mgraph.ContentStore(bench.source_store),
            _scratch(tmp_path),
            contract=other,
        )


def test_a_receipt_disagreeing_on_the_ssuid_digest_is_refused(tmp_path, monkeypatch):
    """The receipt's own field must agree with the recomputation, not only the pin."""
    bench = _Bench(monkeypatch, tmp_path, ssuid_digest="1" * 64)
    with pytest.raises(readback.ReadbackError, match="SELECTED_SSUID_DIGEST"):
        readback.read_sipp_december_projection(
            mgraph.ContentStore(bench.source_store),
            _scratch(tmp_path),
            contract=bench.contract,
        )


def test_a_narrower_selection_is_refused_rather_than_compared(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    other = bench.variant(
        lambda doc: doc["source"].__setitem__("selection", "first_32_ssuids")
    )
    with pytest.raises(readback.ReadbackError, match="SELECTION_NOT_ALL_SSUIDS"):
        readback.read_sipp_december_projection(
            mgraph.ContentStore(bench.source_store),
            _scratch(tmp_path),
            contract=other,
        )


# --------------------------------------------------------------------------
# Expected versus observed admission
# --------------------------------------------------------------------------
def test_the_packaged_contract_reconciles_against_the_root_plans_counts():
    """The genuine reconciliation is wired to the genuine numbers, not to zero."""
    contract = contract_module.load_contract()
    admitted = contract.admission["expected_admitted_residences"]
    zero_weight = contract.admission["expected_excluded_zero_weight"]
    unknown_target = contract.admission["expected_excluded_unknown_target"]
    residences = contract.source["counts"]["december_residences"]
    assert (admitted, zero_weight, unknown_target, residences) == (
        17588,
        20,
        38,
        17646,
    )
    # Exhaustiveness is asserted on the contract's own fields rather than on
    # the literals above. ``reconcile_admission`` compares four numbers and
    # concludes there is no room for an unpinned exclusion reason; that
    # conclusion holds only while the three expectations sum to the pinned
    # roster, so a future contract edit that broke it must fail here.
    assert admitted + zero_weight + unknown_target == residences


def test_admission_records_expected_versus_observed(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    reconciliation = bench.run().nodes[DONOR].receipt["admission"]["reconciliation"]
    assert reconciliation["agrees"] is True
    assert reconciliation["expected"] == reconciliation["observed"]
    assert reconciliation["expected"] == {
        "denominator_residences": fx.DENOMINATOR_RESIDENCES,
        "admitted_residences": fx.ADMITTED_RESIDENCES,
        "excluded_zero_weight": fx.EXCLUDED_ZERO_WEIGHT,
        "excluded_unknown_target": fx.EXCLUDED_UNKNOWN_TARGET,
    }
    assert dict(reconciliation["observed_excluded_reasons_exclusive"]) == {
        "target_unknown": fx.EXCLUDED_UNKNOWN_TARGET,
        "weight_not_positive": fx.EXCLUDED_ZERO_WEIGHT,
        "weight_unknown": fx.EXCLUDED_UNKNOWN_WEIGHT,
    }


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("expected_admitted_residences", 17588, "admitted_residences"),
        ("expected_excluded_zero_weight", 20, "excluded_zero_weight"),
        ("expected_excluded_unknown_target", 38, "excluded_unknown_target"),
    ],
)
def test_a_contract_expecting_other_counts_refuses_the_donor_frame(
    tmp_path, monkeypatch, field, value, code
):
    """A genuine expectation applied to this artifact must refuse, not proceed."""
    bench = _Bench(monkeypatch, tmp_path)
    other = bench.variant(lambda doc: doc["admission"].__setitem__(field, value))
    with pytest.raises(
        equity.EquityGraphError, match=f"ADMISSION_RECONCILIATION:.*{code}"
    ):
        equity.load_sipp_equity_donor_frame(bench.source_store, contract=other)


def test_the_reconciliation_refusal_reaches_the_graph_and_leaves_fit_unexecuted(
    tmp_path, monkeypatch
):
    bench = _Bench(monkeypatch, tmp_path)
    bench.active[0] = bench.variant(
        lambda doc: doc["admission"].__setitem__("expected_admitted_residences", 17588)
    )
    with pytest.raises(Exception) as raised:
        bench.run()
    chain, error = [], raised.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert any("ADMISSION_RECONCILIATION" in text for text in chain)


def test_a_source_admitting_nothing_refuses_after_a_matching_expectation(
    tmp_path, monkeypatch
):
    """No admitted donors is its own refusal, not an empty fit."""
    rows = [
        row
        for row in fx.invented_rows()
        if row["SSUID"] in {"00000020", "00000021", "00000022", "00000023"}
    ]
    bench = _Bench(
        monkeypatch,
        tmp_path,
        rows=rows,
        expected={
            "expected_admitted_residences": 0,
            "expected_excluded_zero_weight": 1,
            "expected_excluded_unknown_target": 2,
        },
    )
    with pytest.raises(equity.EquityGraphError, match="NO_ADMITTED_DONORS"):
        equity.load_sipp_equity_donor_frame(bench.source_store, contract=bench.contract)


# --------------------------------------------------------------------------
# The support artifact's identity and evidence binding
# --------------------------------------------------------------------------
class _DonorEmittingSupport(equity.SippDonorCreateKernel):
    """Rewrites only the support payload, to exercise the validator's checks."""

    def __init__(self, mutate, **kwargs):
        super().__init__(**kwargs)
        self._mutate = mutate

    def run(self, context):
        result = super().run(context)
        support = json.loads(result.artifacts["donor_support"])
        self._mutate(support)
        return type(result)(
            frame=result.frame,
            artifacts={
                "donor_support": canonical_json(support),
                "donor_evidence": result.artifacts["donor_evidence"],
            },
            receipt=result.receipt,
        )


def _kernels_with_donor(donor):
    import microcosm.fit.graph_models as graph_models

    registry = mgraph.KernelRegistry()
    for kernel in (
        donor,
        equity.SippRecipientRosterKernel(),
        equity.SippSupportValidationKernel(),
        graph_models.QRFTrainKernel(),
        graph_models.QRFApplyKernel(),
    ):
        registry.register(kernel)
    return registry


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda s: s.pop("evidence_sha256"), "SUPPORT_ARTIFACT_EVIDENCE_BINDING"),
        (
            lambda s: s.__setitem__("evidence_sha256", "not-a-digest"),
            "SUPPORT_ARTIFACT_EVIDENCE_BINDING",
        ),
        (
            lambda s: s.__setitem__("evidence_format", "something-else/1"),
            "SUPPORT_ARTIFACT_EVIDENCE_BINDING",
        ),
        (
            lambda s: s.__setitem__("target", "another_target"),
            "SUPPORT_ARTIFACT_IDENTITY",
        ),
        (
            lambda s: s.__setitem__("consumer_contract_identity", "0" * 64),
            "SUPPORT_ARTIFACT_IDENTITY",
        ),
        (
            lambda s: s.__setitem__("supported_household_sizes", []),
            "SUPPORT_ARTIFACT_SIZES",
        ),
    ],
)
def test_a_support_artifact_that_fails_its_checks_is_refused(
    tmp_path, monkeypatch, mutate, code
):
    bench = _Bench(monkeypatch, tmp_path)
    donor = _DonorEmittingSupport(mutate, contract=bench.contract)
    with pytest.raises(Exception, match=code):
        bench.run(kernels=_kernels_with_donor(donor))


# --------------------------------------------------------------------------
# The narrow recipient-column constraint, and the readback's remaining gaps
# --------------------------------------------------------------------------
def test_a_recipient_carrying_any_other_column_is_refused_by_create(
    tmp_path, monkeypatch
):
    """CREATE must load exactly its declared columns, so nothing else may ride along.

    Recipient CREATE declares only ``roster_household_size``. A recipient
    population may therefore carry ids, membership and an exactly agreeing
    ``household_size`` claim — which is checked and dropped — and nothing else.
    """
    bench = _Bench(
        monkeypatch,
        tmp_path,
        sizes=(1, 2, 3, 4, 2),
        extra_column=("region", ["a", "b", "c", "d", "e"]),
    )
    with pytest.raises(Exception, match="data columns do not exactly equal"):
        bench.run()


def test_a_recipient_household_with_no_members_is_refused(tmp_path, monkeypatch):
    """The Frame refuses it first; the kernel's own guard sits behind that.

    ``csv-tables`` builds a Frame before the recipient kernel sees anything,
    and the Frame requires the group ids to be exactly the distinct membership
    values. So an empty household never reaches
    ``RECIPIENT_INCOMPLETE_MEMBERSHIP``; it is refused earlier, by name.
    """
    bench = _Bench(monkeypatch, tmp_path, sizes=(1, 2, 0, 4, 2))
    with pytest.raises(
        Exception, match="referenced by no person|RECIPIENT_INCOMPLETE_MEMBERSHIP"
    ):
        bench.run()


def test_a_wrong_producer_binding_is_refused(tmp_path, monkeypatch):
    bench = _Bench(monkeypatch, tmp_path)
    other = bench.variant(
        lambda doc: doc["source"].__setitem__("producer_binding", "3" * 64)
    )
    with pytest.raises(Exception, match="STORE_BINDING|RECEIPT_BINDING|RECEIPT_PIN"):
        equity.load_sipp_equity_donor_frame(bench.source_store, contract=other)


def test_a_receipt_contract_field_disagreeing_with_a_pin_is_refused(
    tmp_path, monkeypatch
):
    bench = _Bench(monkeypatch, tmp_path)
    other = bench.variant(lambda doc: doc["source"].__setitem__("parent_bytes", 999999))
    with pytest.raises(readback.ReadbackError, match="RECEIPT_CONTRACT_PIN"):
        readback.read_sipp_december_projection(
            mgraph.ContentStore(bench.source_store),
            _scratch(tmp_path),
            contract=other,
        )


def test_a_receipt_shaped_like_the_genuine_one_is_accepted(tmp_path, monkeypatch):
    """The accepted producer also writes ``parser`` and ``aggregates``."""
    bench = _Bench(monkeypatch, tmp_path, genuine_shaped=True)
    _, _, _, receipt = fx.accepted_store(tmp_path / "shape-only", genuine_shaped=True)
    assert {"parser", "aggregates"} <= set(json.loads(receipt))
    manifest = bench.run()
    assert manifest.nodes[DONOR].receipt["admitted_residences"] == (
        fx.ADMITTED_RESIDENCES
    )
    document = json.loads(_evidence(manifest, bench.tmp_path / "store"))
    assert len(document["donors"]) == fx.ADMITTED_RESIDENCES


def test_the_codec_writes_no_scratch_beside_the_source_store(tmp_path, monkeypatch):
    """The bounded-copy namespace is a private temporary directory, not a sibling."""
    bench = _Bench(monkeypatch, tmp_path)
    before = sorted(p.name for p in bench.tmp_path.iterdir())
    bench.donor_frame()
    assert sorted(p.name for p in bench.tmp_path.iterdir()) == before
    assert not (bench.tmp_path / "readback-scratch").exists()
