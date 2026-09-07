"""Invented metadata only: no candidate/reference values or scientific scoring."""
# ruff: noqa: E402 - the explicit isolated subprocess entry installs source roots.

from __future__ import annotations

import hashlib
import os
from pathlib import Path

if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[3]
    for shard in (root / "packages").iterdir():
        if (shard / "src").is_dir():
            sys.path.insert(0, str(shard / "src"))

import pytest

from microcosm.build.cd_benchmark_io import BoundaryError, FilePin, read_pinned


def test_mutation_is_refused_before_decode(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_bytes(b'{"a":1}')
    pin = FilePin(path, 7, hashlib.sha256(path.read_bytes()).hexdigest())
    assert read_pinned(pin) == b'{"a":1}'
    path.write_bytes(b'{"a":2}')
    with pytest.raises(BoundaryError, match="artifact_digest"):
        read_pinned(pin)


def test_symbolic_metadata_refuses(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_bytes(b"{}")
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(BoundaryError, match="path"):
        read_pinned(FilePin(link, 2, hashlib.sha256(b"{}").hexdigest()))


def test_nonregular_and_bounded_metadata(tmp_path):
    with pytest.raises(BoundaryError):
        FilePin(tmp_path, True, "a" * 64)
    with pytest.raises(BoundaryError):
        FilePin(tmp_path, 9_000_000, "a" * 64)


from microcosm.build.cd_benchmark.approval import (
    ApprovalAuthority,
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
    ReviewerRole,
)
from microcosm.build.cd_benchmark.canonical import canonical_bytes, digest, strict_load
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark_io import (
    DurableLedger,
    ProducerEvidence,
    durable,
    load_authority,
    load_candidate,
)


def pin_document(path, data):
    raw = canonical_bytes(data)
    path.write_bytes(raw)
    return FilePin(path, len(raw), digest(raw))


def approval_files(tmp_path, protocol, scopes=None):
    if scopes is None:
        scopes = frozenset(
            (
                ApprovalScope.CANDIDATE_SCORING,
                ApprovalScope.PRODUCTION_LEDGER_REGISTRATION,
            )
        )
    pins = []
    for role in ReviewerRole:
        record = ApprovalRecord(
            protocol.sha256,
            ApprovalVerdict.APPROVE,
            role,
            "invented:" + role.value,
            scopes,
            digest(role.value.encode()),
        )
        pins.append(
            pin_document(tmp_path / (role.value + ".json"), record.as_payload())
        )
    return pins


def make_ledger(tmp_path):
    protocol = canonical_protocol()
    root, peer = approval_files(tmp_path, protocol)
    authority = load_authority(
        protocol, root=root, peer=peer, root_id="invented:root", peer_id="invented:peer"
    )
    directory = tmp_path / "ledger"
    directory.mkdir()
    ledger = DurableLedger(
        directory, "invented:ledger", "invented:actor", protocol, authority
    )
    with ledger.locked():
        snapshot = ledger.register(expected_snapshot=None)
    return ledger, snapshot


def fixture_adapter(slot, doc, artifacts):
    """An invented codec, deliberately no genuine schema/producer accepted."""
    assert doc["schema"] == "invented.cd_final_population_proof.v1"
    assert doc["grain"] == "final_cloned_population"
    assert doc["covers_zero_weight_rows"] is True
    assert doc["source_grain_digest"] == artifacts["source"]["original_grain_digest"]
    assert doc["dependencies"]["source"] == digest(canonical_bytes(artifacts["source"]))
    return ProducerEvidence(doc["identity"], doc["dependencies"])


def candidate(
    tmp_path,
    ledger,
    snapshot,
    *,
    name="invented:candidate",
    population="p",
    roots=("invented:tuning",),
    ancestors=(),
    external=(),
    mutate=None,
    missing=None,
    adapters=True,
):
    directory = tmp_path / ("candidate-" + str(len(list(tmp_path.glob("candidate-*")))))
    directory.mkdir()
    identity = dict(
        candidate_id=name,
        population_artifact_id="invented:population:" + population,
        population_content_sha256=digest(population.encode()),
        population_semantic_snapshot_sha256=digest((population + "semantic").encode()),
        producer_run_id="invented:run",
        producer_git_head="a" * 40,
        producer_implementation_sha256="b" * 64,
        model_identity_or_not_invoked="not_invoked",
    )
    history = dict(
        schema="microcosm.cd_exposure_audit_io.v1",
        identity=identity,
        reservation_complete=True,
        ancestry_complete=True,
        prior_history_complete=True,
        tuning_roots=list(roots),
        ancestor_populations=list(ancestors),
        external_exposures=list(external),
        expected_snapshot=snapshot,
    )
    source = dict(
        schema="invented.original_source.v1",
        original_grain_digest=digest(b"original-source-grain"),
    )
    source_pin = pin_document(directory / "source.json", source)
    history_pin = pin_document(directory / "history.json", history)
    proof = dict(
        schema="invented.cd_final_population_proof.v1",
        identity=identity,
        grain="final_cloned_population",
        covers_zero_weight_rows=True,
        source_grain_digest=source["original_grain_digest"],
        dependencies={"source": source_pin.sha256, "history": history_pin.sha256},
    )
    if mutate:
        mutate(proof, history, source)
        history_pin = pin_document(directory / "history.json", history)
        source_pin = pin_document(directory / "source.json", source)
        proof["dependencies"] = {
            "source": source_pin.sha256,
            "history": history_pin.sha256,
        }
    proof_pin = pin_document(directory / "proof.json", proof)
    files = dict(proof=proof_pin, source=source_pin, history=history_pin)
    slots = {
        s: {"artifact_id": "proof", "sha256": proof_pin.sha256}
        for s in ledger.protocol.required_binding_slots
    }
    if missing:
        del slots[missing]
    doc = dict(
        schema="microcosm.cd_candidate_io.v1",
        protocol_sha256=ledger.protocol.sha256,
        identity=identity,
        slots=slots,
        income_claimed=True,
        history_artifact_id="history",
        artifacts={
            k: {
                "sha256": p.sha256,
                "bytes": p.size,
                "dependencies": proof["dependencies"] if k == "proof" else {},
            }
            for k, p in files.items()
        },
    )
    manifest = pin_document(directory / "manifest.json", doc)
    return load_candidate(
        ledger.protocol,
        manifest=manifest,
        expected_identity=identity,
        files=files,
        adapters={s: fixture_adapter for s in slots} if adapters else {},
    )


def uncomputed(ledger):
    return {
        b.id: {"verdict": "uncomputed", "reasons": []} for b in ledger.protocol.blocks
    }


def finish_failed(ledger, item):
    with (
        ledger.locked(),
        ledger.attempt(item, planned_scope="invented uncomputed metadata") as permit,
    ):
        return permit.finish(uncomputed(ledger), terminal_state="failed")


def snapshot(ledger):
    with ledger.locked():
        return durable._snapshot(ledger.read())


def test_complete_lifecycle_and_idempotent_attachment(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    result = finish_failed(ledger, item)
    assert result["ledger_snapshot_before_attempt"] == before
    assert result["ledger_snapshot_through_completion"]["record_count"] == 3
    assert (
        result["exposure_status_per_block"]["income_distribution"]
        == "fresh_same_source_holdout_eligible"
    )
    with ledger.locked():
        data = ledger.read()
        evaluation_id = next(iter(data.attempts))
        assert ledger.attach(evaluation_id) == result
        assert ledger.read().raw == data.raw
        assert ledger.register(
            expected_snapshot=durable._snapshot(data)
        ) == durable._snapshot(data)
    assert not list(ledger.directory.glob("*.tmp"))


def test_attempt_is_fsynced_before_permit_and_directories_synced(tmp_path, monkeypatch):
    events = []
    original = os.fsync

    def traced(fd):
        events.append((os.fstat(fd).st_mode, os.fstat(fd).st_size))
        original(fd)

    monkeypatch.setattr(os, "fsync", traced)
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    with ledger.locked(), ledger.attempt(item, planned_scope="invented") as permit:
        raw = (ledger.directory / "ledger.ndjson").read_bytes()
        assert strict_load(raw.splitlines()[-1])["event_type"] == "evaluation_attempt"
        assert any(size == len(raw) for _, size in events)
        permit.finish(uncomputed(ledger), terminal_state="failed")
    import stat

    assert sum(stat.S_ISDIR(mode) for mode, _ in events) >= 5


@pytest.mark.parametrize(
    "slot",
    [
        "candidate_id",
        "source_lineage_mapping_artifact_id_and_sha256",
        "geography_assignment_artifact_id_and_sha256",
        "exposure_ledger_snapshot_sha256",
    ],
)
def test_missing_slots_refuse(tmp_path, slot):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="candidate_binding_incomplete"):
        candidate(tmp_path, ledger, before, missing=slot)
    assert snapshot(ledger) == before


def test_missing_scientific_adapters_refuse(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="producer_adapters"):
        candidate(tmp_path, ledger, before, adapters=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p, h, s: p.update(grain="original_source"),
        lambda p, h, s: p.update(covers_zero_weight_rows=False),
        lambda p, h, s: s.update(original_grain_digest="d" * 64),
    ],
)
def test_actual_fixture_source_grain_and_zero_coverage_refuse(tmp_path, mutate):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(AssertionError):
        candidate(tmp_path, ledger, before, mutate=mutate)


@pytest.mark.parametrize(
    "key,value",
    [("producer_git_head", "c" * 40), ("population_content_sha256", "d" * 64)],
)
def test_producer_wrong_head_or_population_refuses(tmp_path, key, value):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="producer_population_identity"):
        candidate(
            tmp_path,
            ledger,
            before,
            mutate=lambda p, h, s: p.update(identity={**p["identity"], key: value}),
        )


@pytest.mark.parametrize(
    "scopes",
    [
        frozenset((ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY,)),
        frozenset((ApprovalScope.CANDIDATE_SCORING,)),
    ],
)
def test_wrong_action_approvals_refuse(tmp_path, scopes):
    p = canonical_protocol()
    root, peer = approval_files(tmp_path, p, scopes)
    with pytest.raises(BoundaryError, match="approval_action"):
        load_authority(
            p, root=root, peer=peer, root_id="invented:root", peer_id="invented:peer"
        )


def test_constructed_authority_is_not_loaded_authority(tmp_path):
    with pytest.raises(BoundaryError, match="approval_loader_required"):
        DurableLedger(
            tmp_path, "l", "a", canonical_protocol(), ApprovalAuthority.empty()
        )


def test_candidate_bytes_mutate_after_admission_refuses_before_attempt(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    path = item.artifact_pins["source"].path
    path.write_bytes(path.read_bytes().replace(b"source", b"SOURCE"))
    with ledger.locked(), pytest.raises(BoundaryError, match="artifact_digest"):
        with ledger.attempt(item, planned_scope="invented"):
            pytest.fail("comparison reached")
    assert snapshot(ledger) == before


def test_new_candidate_id_same_population_inherits_exposure(tmp_path):
    ledger, before = make_ledger(tmp_path)
    finish_failed(ledger, candidate(tmp_path, ledger, before))
    second = candidate(
        tmp_path,
        ledger,
        snapshot(ledger),
        name="invented:renamed",
        roots=("different-root",),
    )
    result = finish_failed(ledger, second)
    assert (
        result["exposure_status_per_block"]["income_distribution"]
        == "post_exposure_conformity"
    )
    assert (
        result["exposure_status_per_block"]["tenure_distribution"]
        == "post_exposure_conformity"
    )


def test_parent_population_and_tuning_root_inheritance(tmp_path):
    ledger, before = make_ledger(tmp_path)
    finish_failed(ledger, candidate(tmp_path, ledger, before, population="parent"))
    descendant = candidate(
        tmp_path,
        ledger,
        snapshot(ledger),
        population="child",
        roots=("new",),
        ancestors=(digest(b"parent"),),
    )
    assert (
        finish_failed(ledger, descendant)["exposure_status_per_block"][
            "income_distribution"
        ]
        == "post_exposure_conformity"
    )
    sibling = candidate(
        tmp_path,
        ledger,
        snapshot(ledger),
        population="unrelated-data",
        roots=("invented:tuning",),
    )
    assert (
        finish_failed(ledger, sibling)["exposure_status_per_block"][
            "tenure_distribution"
        ]
        == "post_exposure_conformity"
    )


@pytest.mark.parametrize("family", ["B19001", "B25003"])
def test_reserved_alias_external_exposure_marks_shared_total(tmp_path, family):
    ledger, before = make_ledger(tmp_path)
    external = [
        {
            "families": [family],
            "tuning_roots": ["invented:tuning"],
            "populations": [digest(b"p")],
        }
    ]
    item = candidate(tmp_path, ledger, before, external=external)
    assert (
        finish_failed(ledger, item)["exposure_status_per_block"][
            "occupied_household_total"
        ]
        == "post_exposure_conformity"
    )


def test_renamed_derivative_cannot_hide_its_root(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="history_family_root_required"):
        candidate(
            tmp_path,
            ledger,
            before,
            external=[
                {
                    "families": ["renamed-B19001-share"],
                    "tuning_roots": ["invented:tuning"],
                    "populations": [digest(b"p")],
                }
            ],
        )


@pytest.mark.parametrize(
    "key", ["reservation_complete", "ancestry_complete", "prior_history_complete"]
)
def test_unknown_history_refuses(tmp_path, key):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="exposure_history_unknown"):
        candidate(
            tmp_path, ledger, before, mutate=lambda p, h, s: h.update({key: False})
        )


def test_missing_or_recreated_ledger_refuses_expected_history(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    (ledger.directory / "ledger.ndjson").unlink()
    with ledger.locked(), pytest.raises(BoundaryError, match="history_missing"):
        with ledger.attempt(item, planned_scope="invented"):
            pytest.fail("comparison reached")
    (ledger.directory / "ledger.ndjson").write_bytes(b"")
    with ledger.locked(), pytest.raises(BoundaryError):
        with ledger.attempt(item, planned_scope="invented"):
            pytest.fail("comparison reached")
    assert (ledger.directory / "ledger.ndjson").read_bytes() == b""


def test_stale_snapshot_and_partial_tail_are_preserved(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    finish_failed(ledger, item)
    with ledger.locked(), pytest.raises(BoundaryError, match="snapshot_mismatch"):
        with ledger.attempt(item, planned_scope="invented"):
            pytest.fail("comparison reached")
    path = ledger.directory / "ledger.ndjson"
    raw = path.read_bytes() + b'{"torn":'
    path.write_bytes(raw)
    with ledger.locked(), pytest.raises(BoundaryError, match="integrity_unknown"):
        ledger.read()
    assert path.read_bytes() == raw


def test_exception_interrupts_without_masking_primary(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    primary = RuntimeError("invented primary")
    with ledger.locked(), pytest.raises(RuntimeError) as exc:
        with ledger.attempt(item, planned_scope="invented"):
            raise primary
    assert exc.value is primary
    with ledger.locked():
        state = next(iter(ledger.read().attempts.values()))
        assert state.terminal_state == "interrupted"
        with pytest.raises(BoundaryError, match="not_completed"):
            ledger.attach(state.evaluation_id)


def test_secondary_interruption_failure_preserves_primary(tmp_path, monkeypatch):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    primary = KeyboardInterrupt("invented primary")

    def fail(*args):
        raise OSError("invented full disk")

    with ledger.locked(), pytest.raises(KeyboardInterrupt) as exc:
        with ledger.attempt(item, planned_scope="invented"):
            monkeypatch.setattr(ledger, "_interrupt", fail)
            raise primary
    assert exc.value is primary
    assert "unfinished exposure" in primary.__notes__[0]


def test_leaving_permit_and_duplicate_finish_refuse(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    with ledger.locked(), ledger.attempt(item, planned_scope="invented") as permit:
        permit.finish(uncomputed(ledger), terminal_state="failed")
        with pytest.raises(BoundaryError, match="already_terminal"):
            permit.finish(uncomputed(ledger), terminal_state="failed")
    with pytest.raises(BoundaryError, match="inactive"):
        _ = permit.identity


def test_quality_write_failure_reattaches_same_completed_attempt(tmp_path, monkeypatch):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    real = ledger._immutable

    def failed_quality(name, data):
        if name.endswith(".quality.json"):
            raise OSError("invented envelope failure")
        return real(name, data)

    with ledger.locked(), pytest.raises(OSError):
        with ledger.attempt(item, planned_scope="invented") as permit:
            monkeypatch.setattr(ledger, "_immutable", failed_quality)
            permit.finish(uncomputed(ledger), terminal_state="failed")
    monkeypatch.setattr(ledger, "_immutable", real)
    with ledger.locked():
        before_attach = ledger.read()
        evaluation_id = next(iter(before_attach.attempts))
        ledger.attach(evaluation_id)
        assert ledger.read().raw == before_attach.raw


def test_result_mutation_refuses_attachment(tmp_path):
    ledger, before = make_ledger(tmp_path)
    finish_failed(ledger, candidate(tmp_path, ledger, before))
    path = next(ledger.directory.glob("*.result.json"))
    path.write_bytes(path.read_bytes().replace(b"uncomputed", b"Uncomputed"))
    with ledger.locked():
        evaluation_id = next(iter(ledger.read().attempts))
        with pytest.raises(BoundaryError, match="artifact_digest"):
            ledger.attach(evaluation_id)


def process_probe(tmp_path, ledger, mode):
    import subprocess
    import sys
    import uuid

    prefix = tmp_path / ("absent-child-" + str(uuid.uuid4()))
    assert not prefix.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            "pycache_prefix=" + str(prefix),
            str(Path(__file__).resolve()),
            mode,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert not prefix.exists()
    proof = strict_load((tmp_path / (mode + ".proof.json")).read_bytes())
    assert proof["isolated"] == 1 and proof["dont_write_bytecode"] is True
    assert proof["pycache_prefix"] == str(prefix)
    return result


def test_actual_second_process_cannot_take_lock(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with ledger.locked():
        result = process_probe(tmp_path, ledger, "lock")
        assert result.returncode == 19, result.stderr
        assert result.stdout.strip() == "ledger_busy"
    assert snapshot(ledger) == before


def test_actual_hard_exit_leaves_exposure_and_recovers(tmp_path):
    ledger, before = make_ledger(tmp_path)
    result = process_probe(tmp_path, ledger, "crash")
    assert result.returncode == 17, result.stderr
    after_crash = snapshot(ledger)
    second = candidate(tmp_path, ledger, after_crash, name="invented:new-name")
    result = finish_failed(ledger, second)
    assert (
        result["exposure_status_per_block"]["income_distribution"]
        == "post_exposure_conformity"
    )
    with ledger.locked():
        states = list(ledger.read().attempts.values())
        assert [s.terminal_state for s in states] == ["interrupted", "failed"]


def test_metadata_path_swap_during_read_refuses(tmp_path, monkeypatch):
    path = tmp_path / "metadata.json"
    path.write_bytes(b"{}")
    pin = FilePin(path, 2, digest(b"{}"))
    original = os.read
    swapped = False

    def swapping(fd, size):
        nonlocal swapped
        raw = original(fd, size)
        if not swapped:
            replacement = tmp_path / "new.json"
            replacement.write_bytes(b"{}")
            replacement.replace(path)
            swapped = True
        return raw

    monkeypatch.setattr(os, "read", swapping)
    with pytest.raises(BoundaryError, match="path_changed"):
        read_pinned(pin)


def test_append_io_failure_never_yields_permit_and_preserves_torn_tail(
    tmp_path, monkeypatch
):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)

    def torn(fd, raw):
        os.write(fd, raw[:9])
        raise OSError("invented append failure")

    with ledger.locked(), pytest.raises(OSError):
        monkeypatch.setattr(durable, "_write_all", torn)
        with ledger.attempt(item, planned_scope="invented"):
            pytest.fail("comparison reached")
    raw = (ledger.directory / "ledger.ndjson").read_bytes()
    assert not raw.endswith(b"\n")
    with ledger.locked(), pytest.raises(BoundaryError, match="integrity_unknown"):
        ledger.read()
    assert (ledger.directory / "ledger.ndjson").read_bytes() == raw


def _child_probe():
    import subprocess
    import sys

    assert sys.flags.isolated and sys.dont_write_bytecode
    assert not Path(sys.pycache_prefix).exists()
    mode, base = sys.argv[1], Path(sys.argv[2])
    protocol = canonical_protocol()
    pins = [
        FilePin(
            base / (role + ".json"),
            (base / (role + ".json")).stat().st_size,
            digest((base / (role + ".json")).read_bytes()),
        )
        for role in ("root", "peer")
    ]
    authority = load_authority(
        protocol,
        root=pins[0],
        peer=pins[1],
        root_id="invented:root",
        peer_id="invented:peer",
    )
    ledger = DurableLedger(
        base / "ledger", "invented:ledger", "invented:actor", protocol, authority
    )
    loaded = []
    root = Path(__file__).resolve().parents[3]
    for _name, module in sys.modules.copy().items():
        file = getattr(module, "__file__", None)
        if file and Path(file).is_absolute():
            path = Path(file).resolve()
            if path.is_relative_to(root) and path.suffix == ".py":
                rel = path.relative_to(root).as_posix()
                raw = path.read_bytes()
                try:
                    committed = subprocess.check_output(
                        ["git", "-C", str(root), "show", "HEAD:" + rel],
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
                    committed = None
                loaded.append(
                    {
                        "path": rel,
                        "sha256": digest(raw),
                        "matches_HEAD": raw == committed,
                    }
                )
    (base / (mode + ".proof.json")).write_bytes(
        canonical_bytes(
            {
                "isolated": sys.flags.isolated,
                "dont_write_bytecode": sys.dont_write_bytecode,
                "pycache_prefix": sys.pycache_prefix,
                "python": sys.version,
                "executable_sha256": digest(Path(sys.executable).read_bytes()),
                "loaded_source": loaded,
            }
        )
    )
    if mode == "lock":
        try:
            with ledger.locked():
                os._exit(20)
        except BoundaryError as exc:
            print(str(exc), flush=True)
            os._exit(19)
    with ledger.locked():
        before = durable._snapshot(ledger.read())
        item = candidate(base, ledger, before)
        with ledger.attempt(item, planned_scope="invented hard-exit probe"):
            os._exit(17)


if __name__ == "__main__":
    _child_probe()


def test_replaced_loaded_candidate_identity_and_history_refuse(tmp_path):
    from dataclasses import replace

    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    wrong_identity = canonical_bytes(
        {**item.identity, "candidate_id": "invented:replacement"}
    )
    wrong_history = canonical_bytes(
        {**item.history, "tuning_roots": ["invented:replacement"]}
    )
    for forged in (
        replace(item, identity_bytes=wrong_identity),
        replace(item, history_bytes=wrong_history),
    ):
        with ledger.locked(), pytest.raises(BoundaryError, match="candidate_loaded"):
            with ledger.attempt(forged, planned_scope="invented"):
                pytest.fail("comparison reached")
    assert snapshot(ledger) == before


def test_mutated_approval_refuses_before_ledger_read(tmp_path):
    ledger, before = make_ledger(tmp_path)
    root = tmp_path / "root.json"
    root.write_bytes(root.read_bytes().replace(b"APPROVE", b"REJECT!"))
    with ledger.locked(), pytest.raises(BoundaryError, match="artifact_digest"):
        ledger.read()


def test_wrong_external_expected_identity_refuses(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    files = {k: v for k, v in item.artifact_pins.items() if k != "__manifest__"}
    with pytest.raises(BoundaryError, match="candidate_expected_identity"):
        load_candidate(
            ledger.protocol,
            manifest=item.manifest_pin,
            expected_identity={**item.identity, "producer_git_head": "e" * 40},
            files=files,
            adapters={s: fixture_adapter for s in item.binding.slots},
        )


def test_missing_dependency_and_wrong_slot_pins_refuse(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    files = {k: v for k, v in item.artifact_pins.items() if k != "__manifest__"}
    for variant in ("dependency", "slot"):
        doc = strict_load(read_pinned(item.manifest_pin))
        if variant == "dependency":
            doc["artifacts"]["proof"]["dependencies"]["source"] = "e" * 64
        else:
            doc["slots"]["candidate_id"]["sha256"] = "e" * 64
        manifest = pin_document(tmp_path / (variant + ".json"), doc)
        with pytest.raises(BoundaryError, match="candidate_.*pin"):
            load_candidate(
                ledger.protocol,
                manifest=manifest,
                expected_identity=item.identity,
                files=files,
                adapters={s: fixture_adapter for s in item.binding.slots},
            )


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{ "a":1}', b'{"a":NaN}'])
def test_duplicate_noncanonical_and_nonfinite_metadata_refuse(tmp_path, raw):
    from microcosm.build.cd_benchmark_io.artifacts import document

    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        document(FilePin(path, len(raw), digest(raw)))


def test_ledger_replacement_while_locked_refuses(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with ledger.locked():
        old = ledger.read().raw
        path = ledger.directory / "ledger.ndjson"
        replacement = ledger.directory / "replacement"
        replacement.write_bytes(old)
        replacement.replace(path)
        with pytest.raises(BoundaryError, match="ledger_path_changed"):
            ledger.read()


def test_repeated_none_registration_does_not_reset_ledger(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with ledger.locked(), pytest.raises(FileExistsError):
        ledger.register(expected_snapshot=None)
    assert snapshot(ledger) == before


@pytest.mark.parametrize("suffix", ["result.json", "snapshot.ndjson", "quality.json"])
def test_retry_repairs_retained_file_and_directory_durability(
    tmp_path, monkeypatch, suffix
):
    ledger, before = make_ledger(tmp_path)
    target = ledger.directory / ("invented." + suffix)
    original = os.fsync
    failed = False
    observed = []

    def witness(fd):
        nonlocal failed
        st = os.fstat(fd)
        if target.exists() and st.st_ino == target.stat().st_ino:
            if not failed:
                failed = True
                raise OSError("invented first file fsync")
            observed.append("file")
        import stat

        if stat.S_ISDIR(st.st_mode):
            observed.append("directory")
        original(fd)

    with ledger.locked():
        monkeypatch.setattr(os, "fsync", witness)
        with pytest.raises(OSError, match="first file fsync"):
            ledger._immutable(target.name, b"{}")
        observed.clear()
        ledger._immutable(target.name, b"{}")
        assert observed == ["file", "directory"]


def test_retry_repairs_retained_entry_after_directory_fsync_failure(
    tmp_path, monkeypatch
):
    ledger, before = make_ledger(tmp_path)
    real = durable._sync_directory
    calls = []

    def fail_once(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("invented directory fsync")
        real(path)

    with ledger.locked():
        monkeypatch.setattr(durable, "_sync_directory", fail_once)
        with pytest.raises(OSError):
            ledger._immutable("invented.result.json", b"{}")
        ledger._immutable("invented.result.json", b"{}")
        assert len(calls) == 2


@pytest.mark.parametrize("existing", ["registered", "empty"])
def test_registration_none_after_read_never_reuses_an_open_file(tmp_path, existing):
    ledger, before = make_ledger(tmp_path)
    path = ledger.directory / "ledger.ndjson"
    if existing == "empty":
        path.write_bytes(b"")
    original = path.read_bytes()
    with ledger.locked():
        ledger.read()
        with pytest.raises(BoundaryError, match="registration_new_file_required"):
            ledger.register(expected_snapshot=None)
    assert path.read_bytes() == original


def test_reserved_artifact_cannot_replace_internal_manifest_pin(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    doc = strict_load(read_pinned(item.manifest_pin))
    doc["artifacts"]["__manifest__"] = doc["artifacts"].pop("source")
    doc["artifacts"]["proof"]["dependencies"]["__manifest__"] = doc["artifacts"][
        "proof"
    ]["dependencies"].pop("source")
    files = dict(item.artifact_pins)
    files["__manifest__"] = files.pop("source")
    pin = pin_document(tmp_path / "reserved.json", doc)
    with pytest.raises(BoundaryError, match="candidate_reserved_artifact_id"):
        load_candidate(
            ledger.protocol,
            manifest=pin,
            expected_identity=item.identity,
            files=files,
            adapters={s: fixture_adapter for s in item.binding.slots},
        )
    assert item.manifest_pin not in item.artifact_pins.values()


def test_loaded_dependency_roster_cannot_be_dropped(tmp_path):
    from dataclasses import replace

    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    pins = dict(item.artifact_pins)
    del pins["source"]
    forged = replace(item, artifact_pins=pins)
    with (
        ledger.locked(),
        pytest.raises(BoundaryError, match="candidate_loaded_roster_changed"),
    ):
        with ledger.attempt(forged, planned_scope="invented"):
            pytest.fail("comparison reached")


@pytest.mark.parametrize(
    "change", [{"prefix_bytes": 1}, {"record_count": True}, {"record_count": 1.0}]
)
def test_attachment_checks_declared_pre_snapshot_against_actual_prefix(
    tmp_path, change
):
    from dataclasses import replace

    from microcosm.build.cd_benchmark.ledger import build_records, verify_ledger

    ledger, before = make_ledger(tmp_path)
    finish_failed(ledger, candidate(tmp_path, ledger, before))
    with ledger.locked():
        verified = ledger.read()
        records = list(verified.records)
        attempt = records[1]
        payload = dict(attempt.payload)
        payload["ledger_snapshot_before_attempt"] = {**before, **change}
        records[1] = replace(attempt, payload=payload)
        completion_payload = dict(records[2].payload)
        completion_payload["attempt_record_sha256"] = records[1].sha256
        records[2] = replace(
            records[2],
            payload=completion_payload,
            previous_record_sha256=records[1].sha256,
        )
        rewritten = build_records(records)
        assert verify_ledger(rewritten, protocol=ledger.protocol).ok
    (ledger.directory / "ledger.ndjson").write_bytes(rewritten)
    with (
        ledger.locked(),
        pytest.raises(BoundaryError, match="attempt_snapshot_mismatch"),
    ):
        ledger.attach(payload["evaluation_id"])


@pytest.mark.parametrize("value", [True, 1.0])
@pytest.mark.parametrize("boundary", ["registration", "attempt"])
def test_snapshot_counts_require_exact_json_types(tmp_path, value, boundary):
    ledger, before = make_ledger(tmp_path)
    wrong = {**before, "record_count": value}
    assert wrong == before  # The Python equality trap this regression closes.
    with ledger.locked(), pytest.raises(BoundaryError, match="snapshot_mismatch"):
        if boundary == "registration":
            ledger.register(expected_snapshot=wrong)
        else:
            item = candidate(tmp_path, ledger, wrong)
            with ledger.attempt(item, planned_scope="invented"):
                pass


def test_external_population_history_requires_content_digest(tmp_path):
    ledger, before = make_ledger(tmp_path)
    with pytest.raises(BoundaryError, match="history_external_population_digest"):
        candidate(
            tmp_path,
            ledger,
            before,
            external=[
                {
                    "families": ["B19001"],
                    "tuning_roots": ["invented:unrelated"],
                    "populations": ["not-a-content-digest"],
                }
            ],
        )


def test_declared_artifact_byte_count_is_an_integer(tmp_path):
    ledger, before = make_ledger(tmp_path)
    item = candidate(tmp_path, ledger, before)
    doc = strict_load(read_pinned(item.manifest_pin))
    doc["artifacts"]["source"]["bytes"] = float(doc["artifacts"]["source"]["bytes"])
    pin = pin_document(tmp_path / "float-byte-count.json", doc)
    with pytest.raises(BoundaryError, match="candidate_artifact_pin"):
        load_candidate(
            ledger.protocol,
            manifest=pin,
            expected_identity=item.identity,
            files=item.artifact_pins,
            adapters={s: fixture_adapter for s in item.binding.slots},
        )
