"""TRACE Transparent Research Object emission for populace builds.

These tests pin the parts a third-party verifier (or the policyengine.py
ecosystem) must agree with byte-for-byte:

- the canonical-JSON encoding and the composition fingerprint algorithm are
  recomputed by hand here, so a drift in either definition fails loudly;
- ``serialize_tro`` is deterministic (same inputs -> identical bytes);
- the release-manifest converter lands the real ``release_manifest.json``
  fields (build id, gate verdict, artifact hashes) in the right TRO nodes;
- a restricted input keeps its hash in the composition but is flagged
  ``pop:accessRestricted`` at its location — the TRACE point per Vilhuber;
- an artifact without a sha256 is refused (no hash, no provenance);
- the emission context reflects ``GITHUB_ACTIONS`` when set.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from populace.build import trace

# A trimmed copy of the real
# packages/populace-data/build/us/release_manifest.json — same field names and
# shapes, smaller smoke/score sections. The converter must work off these
# actual keys, not a paraphrase.
RELEASE_MANIFEST_FIXTURE: dict = {
    "build_id": "populace-us-2024-9f1260b-20260611",
    "builder": "populace",
    "build_sha": "9f1260b",
    "build_date": "2026-06-11",
    "dataset": {
        "filename": "populace_us_2024.h5",
        "sha256": ("dc75c0d4fdedd57946db84a8d838dbc5b61a284365c3ce6eb6508b8e81111a4b"),
    },
    "calibration": {
        "filename": "populace_us_2024_calibration.npz",
        "sha256": ("a3da2f59085c45f0e16b06337818e3513c2635911dc0d16fa7deb5006263c12a"),
    },
    "construction": (
        "eCPS-free: every layer from primary sources (CPS ASEC, IRS PUF 2015 "
        "uprated, Fed SCF 2022, Census SIPP, CPS-ORG, MEPS-IC parameters, "
        "Census ACS 2022); enhanced CPS used only as the scoring benchmark"
    ),
    "gates": {
        "parity_gaps": 0,
        "exported_nonzero": {"passed": True, "stored_columns": 309},
        "calibration": {
            "within_10pct_share": 0.9509,
            "loss": 0.022,
            "max_weight": 297651,
            "weights_above_500k": 0,
            "max_weight_ratio": 50,
        },
        "smoke": {
            "people_m": 332.3,
            "snap_b": 97.8,
            "net_worth_t": 176.5,
            "net_stcg_b": -77.5,
        },
    },
    "score_vs_enhanced_cps": {
        "protocol": (
            "matched 41,314 households, symmetric refit, 739-target holdout "
            "(seed 20260529)"
        ),
        "train_loss": {"populace": 0.18957, "enhanced_cps": 1.08879},
        "holdout_loss": {"populace": 0.03837, "enhanced_cps": 0.3167},
        "full_loss": {"populace": 0.22794, "enhanced_cps": 1.40549},
        "per_target_wins": {"populace": 1040, "enhanced_cps": 2613, "ties": 51},
    },
}

# The IRS PUF: a restricted input the build consumes but a re-runner cannot
# access. Hash is illustrative (sha256 of a fixed byte string), not a real PUF
# digest.
PUF_INPUT = {
    "id": "irs_puf_2015",
    "name": "IRS SOI Public Use File 2015 (uprated)",
    "sha256": hashlib.sha256(b"irs-puf-2015-uprated").hexdigest(),
    "location": (
        "https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-"
        "microdata-files"
    ),
    "access_restricted": True,
}


def _graph_node(tro: dict) -> dict:
    """The single TRO node in the @graph."""
    graph = tro["@graph"]
    assert len(graph) == 1
    return graph[0]


def _artifact_hashes(node: dict) -> set[str]:
    composition = node["trov:hasComposition"]
    return {a["trov:sha256"] for a in composition["trov:hasArtifact"]}


def _locations(node: dict) -> list[dict]:
    arrangement = node["trov:hasArrangement"][0]
    return arrangement["trov:hasArtifactLocation"]


class TestCanonicalBytesAndFingerprint:
    def test_canonical_json_bytes_matches_documented_algorithm(self) -> None:
        value = {"b": 1, "a": [2, 3]}
        # Documented: json.dumps(indent=2, sort_keys=True) + "\n", utf-8.
        expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        assert trace.canonical_json_bytes(value) == expected

    def test_canonical_json_bytes_is_sorted_and_newline_terminated(self) -> None:
        out = trace.canonical_json_bytes({"z": 1, "a": 2})
        text = out.decode("utf-8")
        assert text.endswith("\n")
        # sort_keys puts "a" before "z"
        assert text.index('"a"') < text.index('"z"')

    def test_composition_fingerprint_matches_documented_algorithm(self) -> None:
        hashes = ["ccc", "aaa", "bbb"]
        # Documented: sha256 of the *sorted* hashes joined by "\n".
        expected = hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()
        assert trace.compute_composition_fingerprint(hashes) == expected

    def test_fingerprint_is_order_independent(self) -> None:
        a = trace.compute_composition_fingerprint(["x", "y", "z"])
        b = trace.compute_composition_fingerprint(["z", "x", "y"])
        assert a == b

    def test_sha256_file_streams_correct_digest(self, tmp_path) -> None:
        payload = b"populace dataset bytes" * 1000
        path = tmp_path / "artifact.h5"
        path.write_bytes(payload)
        assert trace.sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestBuildBuildTroDeterminism:
    def _minimal_tro(self) -> dict:
        return trace.build_build_tro(
            build_id="populace-us-2024-test",
            builder="populace",
            build_sha="deadbee",
            build_date="2026-06-11",
            config_manifest={"year": 2024, "seed": 0},
            stage_records=[{"stage": "export", "produced": ["snap"]}],
            gate_manifest={"passed": True, "gates": {"parity": {"passed": True}}},
            input_artifacts=[PUF_INPUT],
            output_artifacts=[
                {
                    "id": "dataset",
                    "name": "populace_us_2024.h5",
                    "sha256": "a" * 64,
                    "location": "hf://policyengine/populace-us/populace_us_2024.h5",
                    "mime_type": "application/x-hdf5",
                }
            ],
            created_at="2026-06-11T00:00:00Z",
        )

    def test_serialize_is_byte_identical_across_calls(self) -> None:
        a = trace.serialize_tro(self._minimal_tro())
        b = trace.serialize_tro(self._minimal_tro())
        assert a == b

    def test_serialize_uses_canonical_json(self) -> None:
        tro = self._minimal_tro()
        assert trace.serialize_tro(tro) == trace.canonical_json_bytes(tro)

    def test_context_maps_all_required_namespaces(self) -> None:
        tro = self._minimal_tro()
        ctx = tro["@context"][0]
        for prefix in ("rdf", "rdfs", "trov", "schema", "pop"):
            assert prefix in ctx
        assert ctx["trov"] == "https://w3id.org/trace/trov/0.1#"
        assert ctx["pop"] == "https://populace.dev/trace/0.1#"

    def test_tro_node_shape(self) -> None:
        node = _graph_node(self._minimal_tro())
        assert node["@type"] == "trov:TransparentResearchObject"
        assert node["schema:creator"]["schema:name"] == "PolicyEngine"
        assert node["schema:creator"]["schema:url"] == "https://policyengine.org"
        assert node["trov:createdWith"]["schema:name"] == "populace-build"
        # populace-build version is pinned from the package
        assert (
            node["trov:createdWith"]["schema:softwareVersion"]
            == trace._populace_build_version()
        )
        assert node["trov:wasAssembledBy"]["@type"] == (
            "trov:TransparentResearchSystem"
        )


class TestCompositionContents:
    def _tro(self) -> dict:
        return trace.build_build_tro(
            build_id="b",
            builder="populace",
            build_sha="deadbee",
            build_date="2026-06-11",
            config_manifest={"year": 2024},
            stage_records=[{"stage": "export"}],
            gate_manifest={"passed": True},
            input_artifacts=[PUF_INPUT],
            output_artifacts=[
                {
                    "id": "dataset",
                    "name": "populace_us_2024.h5",
                    "sha256": "a" * 64,
                    "location": "hf://x/y.h5",
                },
                {
                    "id": "calibration",
                    "name": "populace_us_2024_calibration.npz",
                    "sha256": "b" * 64,
                    "location": "hf://x/y.npz",
                },
            ],
            created_at="2026-06-11T00:00:00Z",
        )

    def test_output_and_input_hashes_in_composition(self) -> None:
        node = _graph_node(self._tro())
        hashes = _artifact_hashes(node)
        assert "a" * 64 in hashes  # dataset
        assert "b" * 64 in hashes  # calibration
        assert PUF_INPUT["sha256"] in hashes  # restricted input

    def test_payload_artifacts_hash_their_canonical_bytes(self) -> None:
        node = _graph_node(self._tro())
        hashes = _artifact_hashes(node)
        config_hash = hashlib.sha256(
            trace.canonical_json_bytes({"year": 2024})
        ).hexdigest()
        gate_hash = hashlib.sha256(
            trace.canonical_json_bytes({"passed": True})
        ).hexdigest()
        stage_hash = hashlib.sha256(
            trace.canonical_json_bytes([{"stage": "export"}])
        ).hexdigest()
        assert config_hash in hashes
        assert gate_hash in hashes
        assert stage_hash in hashes

    def test_payload_artifacts_named_for_the_payload(self) -> None:
        node = _graph_node(self._tro())
        names = {
            a.get("schema:name")
            for a in node["trov:hasComposition"]["trov:hasArtifact"]
        }
        assert "build_config" in names
        assert "gate_report" in names
        assert "stage_records" in names

    def test_fingerprint_is_over_all_artifact_hashes(self) -> None:
        node = _graph_node(self._tro())
        composition = node["trov:hasComposition"]
        recorded = composition["trov:hasFingerprint"]["trov:sha256"]
        all_hashes = [a["trov:sha256"] for a in composition["trov:hasArtifact"]]
        assert recorded == trace.compute_composition_fingerprint(all_hashes)

    def test_optional_payloads_omitted_when_none(self) -> None:
        tro = trace.build_build_tro(
            build_id="b",
            builder="populace",
            build_sha="deadbee",
            build_date="2026-06-11",
            config_manifest=None,
            stage_records=None,
            gate_manifest=None,
            input_artifacts=(),
            output_artifacts=[
                {
                    "id": "dataset",
                    "name": "d.h5",
                    "sha256": "a" * 64,
                    "location": "hf://x/y.h5",
                }
            ],
            created_at="2026-06-11T00:00:00Z",
        )
        node = _graph_node(tro)
        names = {
            a.get("schema:name")
            for a in node["trov:hasComposition"]["trov:hasArtifact"]
        }
        assert "build_config" not in names
        assert "gate_report" not in names
        assert "stage_records" not in names


class TestRestrictedInputArrangement:
    def _tro(self) -> dict:
        return trace.build_build_tro(
            build_id="b",
            builder="populace",
            build_sha="deadbee",
            build_date="2026-06-11",
            config_manifest=None,
            stage_records=None,
            gate_manifest=None,
            input_artifacts=[PUF_INPUT],
            output_artifacts=[
                {
                    "id": "dataset",
                    "name": "d.h5",
                    "sha256": "a" * 64,
                    "location": "hf://x/y.h5",
                }
            ],
            created_at="2026-06-11T00:00:00Z",
        )

    def test_restricted_input_flagged_at_its_location(self) -> None:
        node = _graph_node(self._tro())
        locations = _locations(node)
        puf_locations = [
            loc for loc in locations if loc["trov:hasLocation"] == PUF_INPUT["location"]
        ]
        assert len(puf_locations) == 1
        assert puf_locations[0]["pop:accessRestricted"] is True

    def test_restricted_input_still_in_composition(self) -> None:
        # The whole TRACE point: the hash is bound even when the bytes are
        # unreachable to a re-runner.
        node = _graph_node(self._tro())
        assert PUF_INPUT["sha256"] in _artifact_hashes(node)

    def test_output_location_is_not_flagged_restricted(self) -> None:
        node = _graph_node(self._tro())
        for loc in _locations(node):
            if loc["trov:hasLocation"] == "hf://x/y.h5":
                assert "pop:accessRestricted" not in loc


class TestValidation:
    def test_input_without_sha256_raises(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            trace.build_build_tro(
                build_id="b",
                builder="populace",
                build_sha="deadbee",
                build_date="2026-06-11",
                input_artifacts=[
                    {"id": "x", "name": "x", "location": "https://example.com"}
                ],
                output_artifacts=[
                    {
                        "id": "dataset",
                        "name": "d.h5",
                        "sha256": "a" * 64,
                        "location": "hf://x/y.h5",
                    }
                ],
                created_at="2026-06-11T00:00:00Z",
            )

    def test_output_without_sha256_raises(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            trace.build_build_tro(
                build_id="b",
                builder="populace",
                build_sha="deadbee",
                build_date="2026-06-11",
                input_artifacts=(),
                output_artifacts=[
                    {"id": "dataset", "name": "d.h5", "location": "hf://x/y.h5"}
                ],
                created_at="2026-06-11T00:00:00Z",
            )


class TestReleaseManifestConverter:
    def _tro(self) -> dict:
        return trace.tro_from_release_manifest(
            RELEASE_MANIFEST_FIXTURE,
            input_artifacts=[PUF_INPUT],
            self_url="https://populace.dev/trace/us-2024.jsonld",
        )

    def test_build_identity_lands_on_performance(self) -> None:
        node = _graph_node(self._tro())
        perf = node["trov:hasPerformance"]
        assert perf["pop:buildId"] == "populace-us-2024-9f1260b-20260611"
        assert perf["pop:buildSha"] == "9f1260b"
        assert perf["pop:builder"] == "populace"

    def test_dataset_and_calibration_hashes_are_outputs(self) -> None:
        node = _graph_node(self._tro())
        hashes = _artifact_hashes(node)
        assert RELEASE_MANIFEST_FIXTURE["dataset"]["sha256"] in hashes
        assert RELEASE_MANIFEST_FIXTURE["calibration"]["sha256"] in hashes

    def test_output_locations_point_to_hf(self) -> None:
        node = _graph_node(self._tro())
        hf_locations = [
            loc["trov:hasLocation"]
            for loc in _locations(node)
            if "huggingface.co" in loc["trov:hasLocation"]
        ]
        # both the dataset and calibration files live on the Hub
        assert any("populace_us_2024.h5" in loc for loc in hf_locations)
        assert any("populace_us_2024_calibration.npz" in loc for loc in hf_locations)

    def test_gates_section_becomes_gate_report_payload(self) -> None:
        node = _graph_node(self._tro())
        gate_hash = hashlib.sha256(
            trace.canonical_json_bytes(RELEASE_MANIFEST_FIXTURE["gates"])
        ).hexdigest()
        assert gate_hash in _artifact_hashes(node)

    def test_gates_passed_and_names_recorded(self) -> None:
        node = _graph_node(self._tro())
        perf = node["trov:hasPerformance"]
        # The flat manifest records verdicts as parity_gaps == 0 and
        # exported_nonzero.passed; the derived overall verdict is their
        # conjunction.
        assert perf["pop:gatesPassed"] is True
        names = perf["pop:gateNames"]
        assert names == sorted(names)
        assert "parity_gaps" in names
        assert "smoke" in names

    def test_self_url_recorded(self) -> None:
        node = _graph_node(self._tro())
        assert node["pop:selfUrl"] == ("https://populace.dev/trace/us-2024.jsonld")

    def test_restricted_input_carried_through(self) -> None:
        node = _graph_node(self._tro())
        assert PUF_INPUT["sha256"] in _artifact_hashes(node)
        restricted = [
            loc for loc in _locations(node) if loc.get("pop:accessRestricted") is True
        ]
        assert len(restricted) == 1

    def test_converter_output_is_deterministic(self) -> None:
        assert trace.serialize_tro(self._tro()) == trace.serialize_tro(self._tro())

    def test_hf_revision_pins_artifact_locations(self) -> None:
        tro = trace.tro_from_release_manifest(
            RELEASE_MANIFEST_FIXTURE, hf_revision="4a8e7d39eb9e"
        )
        node = _graph_node(tro)
        dataset_locations = [
            loc["trov:hasLocation"]
            for loc in _locations(node)
            if "populace_us_2024.h5" in loc["trov:hasLocation"]
        ]
        assert dataset_locations == [
            "https://huggingface.co/datasets/policyengine/populace-us/"
            "resolve/4a8e7d39eb9e/populace_us_2024.h5"
        ]


class TestGatesPassedDerivation:
    def _tro_with_gates(self, gates: dict) -> dict:
        manifest = dict(RELEASE_MANIFEST_FIXTURE, gates=gates)
        return trace.tro_from_release_manifest(manifest)

    def test_explicit_top_level_passed_wins(self) -> None:
        perf = _graph_node(self._tro_with_gates({"passed": False, "gates": {}}))[
            "trov:hasPerformance"
        ]
        assert perf["pop:gatesPassed"] is False

    def test_failing_gap_counter_fails_overall(self) -> None:
        gates = dict(RELEASE_MANIFEST_FIXTURE["gates"], parity_gaps=3)
        perf = _graph_node(self._tro_with_gates(gates))["trov:hasPerformance"]
        assert perf["pop:gatesPassed"] is False

    def test_failing_subgate_fails_overall(self) -> None:
        gates = dict(
            RELEASE_MANIFEST_FIXTURE["gates"],
            exported_nonzero={"passed": False, "stored_columns": 4},
        )
        perf = _graph_node(self._tro_with_gates(gates))["trov:hasPerformance"]
        assert perf["pop:gatesPassed"] is False

    def test_evidence_only_gates_make_no_claim(self) -> None:
        # No recorded verdicts at all: the gate block stays bound in the
        # composition as evidence, but no overall pass/fail is claimed.
        perf = _graph_node(self._tro_with_gates({"smoke": {"people_m": 332.3}}))[
            "trov:hasPerformance"
        ]
        assert "pop:gatesPassed" not in perf
        assert perf["pop:gateNames"] == ["smoke"]


class TestEmissionContext:
    def test_github_actions_context(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "PolicyEngine/populace")
        monkeypatch.setenv("GITHUB_RUN_ID", "12345")
        monkeypatch.setenv("GITHUB_SHA", "abc123")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        node = _graph_node(trace.tro_from_release_manifest(RELEASE_MANIFEST_FIXTURE))
        perf = node["trov:hasPerformance"]
        assert perf["pop:emittedIn"] == "github-actions"
        assert perf["pop:ciRunUrl"] == (
            "https://github.com/PolicyEngine/populace/actions/runs/12345"
        )
        assert perf["pop:ciGitSha"] == "abc123"
        assert perf["pop:ciGitRef"] == "refs/heads/main"

    def test_local_context(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        node = _graph_node(trace.tro_from_release_manifest(RELEASE_MANIFEST_FIXTURE))
        assert node["trov:hasPerformance"]["pop:emittedIn"] == "local"
