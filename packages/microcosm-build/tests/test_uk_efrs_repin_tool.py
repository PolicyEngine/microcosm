"""The UK eFRS parity re-pin tool and the lockstep it enforces.

The reference artifact's identity is mirrored across the build shard, the data
shard, the committed UK spec and four test modules, and attested by the totals
digest and the gate-battery digests. These tests hold every mirror equal to
the committed parity reference (the tool's definition of "current"), prove the
tool's recompute path reproduces the committed digests before it is allowed
to mint new ones, and exercise the literal-move machinery hermetically. No
licensed bytes are touched: the extraction path itself is covered by
``test_uk_efrs_weighted_totals.py`` and the committed tools' own tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPIN = _module(
    "tools/repin_uk_efrs_parity_reference.py", "repin_uk_efrs_parity_reference"
)
PARITY = _module(
    "tools/build_uk_efrs_parity_reference.py",
    "build_uk_efrs_parity_reference_for_repin_test",
)


class TestCommittedLockstep:
    def test_every_identity_mirror_carries_the_committed_identity(self) -> None:
        identity = REPIN.committed_identity()
        assert (identity.revision, identity.sha256, identity.size_bytes) == (
            PARITY.SOURCE_REVISION,
            PARITY.SOURCE_SHA256,
            PARITY.SOURCE_SIZE_BYTES,
        )
        # Once the parity tool records the release tag (#747), the committed
        # reference must name the same tag.
        assert identity.version == getattr(PARITY, "SOURCE_VERSION", None)
        for path in REPIN.IDENTITY_MIRRORS:
            text = path.read_text(encoding="utf-8")
            assert identity.revision in text, path
            assert identity.sha256 in text, path

    def test_identity_mirrors_agree_with_the_gate_registry_and_the_data_shard(
        self,
    ) -> None:
        from microcosm.build.uk_runtime.weighted_integrity import (
            UK_INPUT_MASS_REFERENCE_REGISTRY,
        )
        from microcosm.data import contract

        identity = REPIN.committed_identity()
        active, totals_digest = REPIN.committed_totals_digest()
        descriptor = UK_INPUT_MASS_REFERENCE_REGISTRY[active]
        assert descriptor.revision == identity.revision
        assert descriptor.sha256 == identity.sha256
        assert descriptor.totals_sha256 == totals_digest
        assert (
            contract._UK_INPUT_MASS_REFERENCE_IDENTITY["revision"] == identity.revision
        )
        assert contract._UK_INPUT_MASS_REFERENCE_IDENTITY["sha256"] == identity.sha256
        assert contract._UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256 == totals_digest
        gates = json.loads(REPIN.GATES_PATH.read_text(encoding="utf-8"))
        registry_identity = REPIN._input_mass_entry(gates)["parameters"][
            "reference_registry"
        ][active]["identity"]
        assert registry_identity["revision"] == identity.revision
        assert registry_identity["sha256"] == identity.sha256
        manifest = json.loads(REPIN.MANIFEST_PATH.read_text(encoding="utf-8"))
        assert manifest["reference"]["revision"] == identity.revision
        assert manifest["reference"]["sha256"] == identity.sha256

    def test_totals_digest_mirrors_carry_the_committed_digest(self) -> None:
        _, totals_digest = REPIN.committed_totals_digest()
        for path in REPIN.TOTALS_DIGEST_MIRRORS:
            assert totals_digest in path.read_text(encoding="utf-8"), path

    def test_recompute_path_reproduces_the_committed_battery_digests(self) -> None:
        from microcosm.data import contract

        pinned = REPIN.committed_battery_digests()
        assert pinned == {
            constant: getattr(contract, constant)
            for constant in REPIN.BATTERY_DIGEST_CONSTANTS
        }
        gates = json.loads(REPIN.GATES_PATH.read_text(encoding="utf-8"))
        _, totals_digest = REPIN.committed_totals_digest()
        recomputed = REPIN.recut_battery_digests(gates, totals_digest)
        assert {
            key: recomputed[key] for key in REPIN.BATTERY_DIGEST_CONSTANTS.values()
        } == {
            key: pinned[constant]
            for constant, key in REPIN.BATTERY_DIGEST_CONSTANTS.items()
        }
        for path in REPIN.BATTERY_DIGEST_MIRRORS:
            text = path.read_text(encoding="utf-8")
            for digest in pinned.values():
                assert digest in text, (path, digest)

    def test_moving_the_registry_identity_moves_policy_manifest_and_evidence(
        self,
    ) -> None:
        gates = json.loads(REPIN.GATES_PATH.read_text(encoding="utf-8"))
        _, totals_digest = REPIN.committed_totals_digest()
        before = REPIN.recut_battery_digests(gates, totals_digest)
        moved = REPIN._gates_with_identity(
            gates,
            new=REPIN.ArtifactIdentity("b" * 40, "c" * 64, 1),
            totals_digest="d" * 64,
        )
        after = REPIN.recut_battery_digests(moved, "d" * 64)
        assert after["policy_sha256"] != before["policy_sha256"]
        assert after["gates_manifest_sha256"] != before["gates_manifest_sha256"]
        assert after["spec_fingerprint"] != before["spec_fingerprint"]
        assert (
            after["input_mass_evidence_sha256"] != before["input_mass_evidence_sha256"]
        )


class TestLiteralMoves:
    def test_identity_replacements_render_both_size_literal_forms(self) -> None:
        old = REPIN.ArtifactIdentity("a" * 40, "b" * 64, 126_579_434)
        new = REPIN.ArtifactIdentity("c" * 40, "d" * 64, 126_553_300)
        replacements = REPIN.identity_replacements(old, new)
        assert replacements["126_579_434"] == "126_553_300"
        assert replacements["126579434"] == "126553300"
        assert replacements["a" * 40] == "c" * 40
        assert replacements["b" * 64] == "d" * 64

    def test_move_literals_rewrites_every_file_and_reports_counts(
        self, tmp_path
    ) -> None:
        first = tmp_path / "first.py"
        second = tmp_path / "second.json"
        first.write_text('REV = "old-rev"\nSHA = "old-sha"\nSIZE = 1_000\n')
        second.write_text(
            '{"revision": "old-rev", "sha256": "old-sha", "size": 1000}\n'
        )
        counts = REPIN.move_literals(
            (first, second),
            {
                "old-rev": "new-rev",
                "old-sha": "new-sha",
                "1_000": "2_000",
                "1000": "2000",
            },
            label="test",
            write=True,
        )
        assert first.read_text() == 'REV = "new-rev"\nSHA = "new-sha"\nSIZE = 2_000\n'
        assert (
            second.read_text()
            == '{"revision": "new-rev", "sha256": "new-sha", "size": 2000}\n'
        )
        assert counts == {
            str(first): {"old-rev": 1, "old-sha": 1, "1_000": 1},
            str(second): {"old-rev": 1, "old-sha": 1, "1000": 1},
        }

    def test_move_literals_refuses_a_drifted_mirror(self, tmp_path) -> None:
        drifted = tmp_path / "drifted.py"
        drifted.write_text('REV = "something-else"\n')
        with pytest.raises(SystemExit, match="carries none of the old literals"):
            REPIN.move_literals(
                (drifted,), {"old-rev": "new-rev"}, label="test", write=False
            )

    def test_move_literals_dry_pass_leaves_files_untouched(self, tmp_path) -> None:
        path = tmp_path / "pinned.py"
        path.write_text('REV = "old-rev"\n')
        REPIN.move_literals((path,), {"old-rev": "new-rev"}, label="test", write=False)
        assert path.read_text() == 'REV = "old-rev"\n'


class TestSourceVersion:
    def test_move_source_version_is_anchored_to_the_assignment(self, tmp_path) -> None:
        tool = tmp_path / "tool.py"
        tool.write_text(
            'PINNED = "1.56.16"\nSOURCE_VERSION = "1.56.16"\nOTHER = "x1.56.16"\n'
        )
        previous = REPIN.move_source_version(tool, "1.56.17", write=True)
        assert previous == "1.56.16"
        assert tool.read_text() == (
            'PINNED = "1.56.16"\nSOURCE_VERSION = "1.56.17"\nOTHER = "x1.56.16"\n'
        )

    def test_move_source_version_reports_absence(self, tmp_path) -> None:
        tool = tmp_path / "tool.py"
        tool.write_text('SOURCE_REVISION = "a" * 40\n')
        assert REPIN.move_source_version(tool, "1.56.17", write=True) is None
        assert tool.read_text() == 'SOURCE_REVISION = "a" * 40\n'

    def test_parity_tool_patch_requires_a_tag_when_the_tool_records_one(
        self, monkeypatch
    ) -> None:
        class Tool:
            SOURCE_REPO_ID = "repo"
            SOURCE_FILENAME = "f.h5"
            SOURCE_VERSION = "1.56.16"

        monkeypatch.setattr(REPIN, "_load_module", lambda path, name: Tool())
        identity = REPIN.ArtifactIdentity("b" * 40, "c" * 64, 1)
        with pytest.raises(SystemExit, match="SOURCE_VERSION"):
            REPIN._parity_tool(identity)
        tagged = REPIN.ArtifactIdentity("b" * 40, "c" * 64, 1, version="1.56.17")
        patched = REPIN._parity_tool(tagged)
        assert patched.SOURCE_VERSION == "1.56.17"
        assert patched.SOURCE_REVISION == "b" * 40


class TestCanonicalTotalsDigest:
    def test_matches_the_gate_loader_digest(self) -> None:
        from microcosm.build.uk_runtime.weighted_integrity import (
            UKInputMassReference,
            _input_mass_reference_evidence_sha256,
        )

        payload = {
            "schema_version": 1,
            "identity": {
                "filename": "enhanced_frs_2024_25.h5",
                "revision": "a" * 40,
                "sha256": "b" * 64,
                "vintage": "2024_25",
            },
            "totals": {"employment_income": 10.5, "council_tax": 3.0},
        }
        expected = _input_mass_reference_evidence_sha256(
            UKInputMassReference(
                totals=payload["totals"],
                filename="enhanced_frs_2024_25.h5",
                revision="a" * 40,
                sha256="b" * 64,
                vintage="2024_25",
            )
        )
        assert REPIN.canonical_totals_digest(payload) == expected


class TestIdentityValidation:
    @pytest.mark.parametrize(
        ("revision", "sha", "size"),
        [
            ("short", "b" * 64, 1),
            ("a" * 40, "B" * 64, 1),
            ("a" * 40, "b" * 64, 0),
        ],
    )
    def test_rejects_malformed_identities(self, revision, sha, size) -> None:
        with pytest.raises(ValueError):
            REPIN.ArtifactIdentity(revision, sha, size)
