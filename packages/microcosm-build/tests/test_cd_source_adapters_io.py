"""Invented metadata controls only; no scientific approval or scoring."""

import copy

from microcosm.build.cd_benchmark.canonical import canonical_bytes, digest
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark_io import source_adapters as io
from microcosm.build.cd_benchmark_io.artifacts import FilePin, document, load_candidate


def envelope_result():
    """Only an envelope test fixture, never claimed as source-verifier output."""
    identity = {
        "candidate_id": "invented:évidence",
        "population_artifact_id": "invented:population",
        "population_content_sha256": "a" * 64,
        "population_semantic_snapshot_sha256": "b" * 64,
        "producer_run_id": "invented:run",
        "producer_git_head": "c" * 40,
        "producer_implementation_sha256": "d" * 64,
        "model_identity_or_not_invoked": "not_invoked",
    }
    return {
        "identity": identity,
        "slots": {
            slot: {
                "dependencies": {},
                "evidence": {"row_count": 2, "actual_verifier_sha256": "e" * 64},
            }
            for slot in io.SLOTS
        },
    }


def exercise(root, result=None):
    result = envelope_result() if result is None else result
    records = []

    def refused(name, action, reason=None):
        try:
            action()
        except (ValueError, TypeError) as error:
            if reason:
                assert reason in str(error), (name, str(error))
            records.append(
                {
                    "name": name,
                    "status": "REFUSED",
                    "reason": type(error).__name__ if reason is None else reason,
                }
            )
        else:
            raise AssertionError(name + " unexpectedly passed")

    control, completion, seal = io.write_verifier_control(
        root / "external",
        result=result,
        run_id="invented",
        proposal_sha256="1" * 64,
        verifier_implementation_sha256="2" * 64,
    )
    pins = io.write_slot_summaries(root / "bundle", control_pin=control)
    args = dict(
        control_pin=control,
        completion_pin=completion,
        seal_pin=seal,
        expected_run_id="invented",
        expected_proposal_sha256="1" * 64,
        expected_verifier_implementation_sha256="2" * 64,
        candidate_files=pins,
    )
    adapters = io.producer_adapters(**args)
    docs = {slot: document(pin) for slot, pin in pins.items()}
    for slot, doc in docs.items():
        evidence = adapters[slot](slot, doc, docs)
        assert dict(evidence.identity) == result["identity"]
        assert pins[slot].path.read_bytes() == canonical_bytes(doc)
        records.append({"name": slot + ":exact_control", "status": "PASS"})
        for label, mutate in (
            ("missing_evidence", lambda d: d.pop("evidence")),
            ("other_evidence", lambda d: d.update(evidence={"wrong": True})),
            (
                "wrong_control_digest",
                lambda d: d.update(external_control_sha256="0" * 64),
            ),
            (
                "wrong_identity",
                lambda d: d["identity"].update(population_content_sha256="0" * 64),
            ),
        ):
            changed = copy.deepcopy(doc)
            mutate(changed)
            refused(
                slot + ":" + label,
                lambda changed=changed, slot=slot: adapters[slot](slot, changed, docs),
                "source_actual_verifier_mismatch",
            )
    refused(
        "no_external_control",
        lambda: io.producer_adapters(candidate_files=pins),
        "source_external_control_required",
    )
    refused(
        "dict_control",
        lambda: io.producer_adapters(**{**args, "control_pin": document(control)}),
        "source_external_control_required",
    )
    refused(
        "bundle_only_control",
        lambda: io.producer_adapters(**{**args, "control_pin": pins[io.UNIVERSE]}),
        "source_control_in_candidate_bundle",
    )
    refused(
        "control_in_bundle",
        lambda: io.producer_adapters(
            **{**args, "candidate_files": {**pins, "pretend-proof": control}}
        ),
        "source_control_in_candidate_bundle",
    )
    copy_path = root / "bundle" / "renamed-control.json"
    copy_path.write_bytes(control.path.read_bytes())
    copied = FilePin(copy_path, control.size, control.sha256)
    refused(
        "renamed_identical_control",
        lambda: io.producer_adapters(
            **{**args, "candidate_files": {**pins, "different-name": copied}}
        ),
        "source_control_in_candidate_bundle",
    )
    refused(
        "bundle_echo_at_invocation",
        lambda: adapters[io.UNIVERSE](
            io.UNIVERSE,
            docs[io.UNIVERSE],
            {**docs, "untrusted-control": document(control)},
        ),
        "source_control_echo",
    )
    refused(
        "wrong_expected_run",
        lambda: io.producer_adapters(**{**args, "expected_run_id": "other"}),
        "source_control_execution",
    )
    refused(
        "wrong_expected_verifier",
        lambda: io.producer_adapters(
            **{**args, "expected_verifier_implementation_sha256": "0" * 64}
        ),
        "source_verifier_implementation",
    )

    protocol = canonical_protocol()
    bundle = {"universe": pins[io.UNIVERSE], "geography": pins[io.GEOGRAPHY]}

    def candidate(files, *, all_slots=False, adapter_roster=adapters):
        slots = {
            s: {
                "artifact_id": "geography" if s == io.GEOGRAPHY else "universe",
                "sha256": files[
                    "geography" if s == io.GEOGRAPHY else "universe"
                ].sha256,
            }
            for s in (protocol.required_binding_slots if all_slots else io.SLOTS)
        }
        doc = {
            "schema": "microcosm.cd_candidate_io.v1",
            "protocol_sha256": protocol.sha256,
            "identity": result["identity"],
            "slots": slots,
            "income_claimed": False,
            "history_artifact_id": "not_supplied_in_this_partial_slice",
            "artifacts": {
                name: {"sha256": p.sha256, "bytes": p.size, "dependencies": {}}
                for name, p in files.items()
            },
        }
        path = root / ("manifest-" + str(len(records)) + ".json")
        raw = canonical_bytes(doc)
        path.write_bytes(raw)
        return load_candidate(
            protocol,
            manifest=FilePin(path, len(raw), digest(raw)),
            expected_identity=result["identity"],
            files=files,
            adapters=adapter_roster,
        )

    refused(
        "all_slots_two_adapters",
        lambda: candidate(bundle, all_slots=True),
        "producer_adapters_missing_or_extra",
    )
    refused(
        "two_slots_two_adapters",
        lambda: candidate(bundle),
        "candidate_binding_incomplete",
    )
    touched = []

    def fatal(*args):
        touched.append(True)
        raise AssertionError("malformed document reached adapter")

    for name, raw in (
        ("binary_graph_payload", b"MCCDACS\x01\x00\x00"),
        ("non_dict", b"[]"),
        ("trailing_lf", canonical_bytes(docs[io.UNIVERSE]) + b"\n"),
        ("whitespace", b'{ "x": 1 }'),
        ("key_order", b'{"z":1,"a":2}'),
        ("escaped_unicode", b'{"name":"\\u00e9"}'),
        ("duplicate", b'{"x":1,"x":2}'),
        ("bom", b"\xef\xbb\xbf{}"),
        ("nonfinite", b'{"x":NaN}'),
    ):
        path = root / "bundle" / (name + ".json")
        path.write_bytes(raw)
        bad = FilePin(path, len(raw), digest(raw))
        refused(
            name,
            lambda bad=bad: candidate(
                {**bundle, "universe": bad}, adapter_roster={s: fatal for s in io.SLOTS}
            ),
        )
    assert not touched
    refused(
        "over_8MiB_pin_before_read",
        lambda: FilePin(root / "never_created", 8 * 1024**2 + 1, "a" * 64),
        "artifact_size",
    )
    raw = control.path.read_bytes()
    control.path.write_bytes(raw[:-1] + b" ")
    refused(
        "changed_control_at_use",
        lambda: adapters[io.UNIVERSE](io.UNIVERSE, docs[io.UNIVERSE], docs),
        "artifact_digest",
    )
    return records


def test_real_821_factories_and_partial_loader(tmp_path):
    assert len(exercise(tmp_path)) >= 30
