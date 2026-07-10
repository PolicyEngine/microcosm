"""The pinned eCPS parity reference and the reason'd exemption register.

These tests defend the two checked-in facts the parity gate depends on:
the reference JSON's integrity (its recorded sha256 matches the real incumbent
artifact bytes) and the register's schema (every exemption is documented and
issue-linked, and names only a real reference layer).
"""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from populace.build.us_runtime.eligibility_inputs import (
    US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS,
)
from populace.build.us_runtime.hours_worked import US_HOURS_WORKED_OUTPUT_COLUMNS
from populace.build.us_runtime.immigration import US_IMMIGRATION_OUTPUT_COLUMNS
from populace.build.us_runtime.parity_reference import (
    ECPS_PARITY_REFERENCE_RESOURCE,
    load_ecps_parity_known_gaps,
    load_ecps_parity_reference,
)
from populace.build.us_runtime.sipp_tips import US_SIPP_TIPS_OUTPUT_COLUMNS
from populace.build.us_runtime.snap_take_up import US_SNAP_TAKE_UP_OUTPUT_COLUMN
from populace.build.us_runtime.take_up_contract import load_take_up_contract

_US_PACKAGE = "populace.build.us"


def _resource(name: str) -> Path:
    return Path(str(files(_US_PACKAGE).joinpath(name)))


def _cached_incumbent_path(source) -> str | None:
    """Cache-only resolution of the pinned incumbent artifact (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    resolved = try_to_load_from_cache(
        repo_id=source.repo_id,
        filename=source.filename,
        revision=source.revision,
        repo_type=source.repo_type,
    )
    return resolved if isinstance(resolved, str) else None


class TestReference:
    def test_reference_loads_with_populated_layers(self) -> None:
        reference = load_ecps_parity_reference()
        assert reference.nonzero_shares
        assert reference.populated_layers
        # Every stored share is a populated layer (the writer drops zeros).
        assert all(share > 0.0 for share in reference.nonzero_shares.values())

    def test_source_provenance_is_complete(self) -> None:
        source = load_ecps_parity_reference().source
        assert source.repo_id == "policyengine/policyengine-us-data"
        assert source.filename == "enhanced_cps_2024.h5"
        assert source.repo_type in {"model", "dataset"}
        assert len(source.sha256) == 64
        assert source.revision
        assert source.vintage

    def test_recorded_sha256_is_a_real_hex_digest(self) -> None:
        # The recorded sha256 must be a 64-char lowercase hex string — the
        # form of a real content digest, not a placeholder.
        sha = load_ecps_parity_reference().source.sha256
        assert len(sha) == 64
        int(sha, 16)  # raises if not hex
        assert sha == sha.lower()

    def test_reference_json_content_matches_loader(self) -> None:
        # Integrity: the raw JSON and the loaded object agree on the shares,
        # so a hand-edit that corrupts the file is caught.
        raw = json.loads(_resource(ECPS_PARITY_REFERENCE_RESOURCE).read_text())
        reference = load_ecps_parity_reference()
        assert set(raw["nonzero_shares"]) == set(reference.nonzero_shares)
        assert raw["source"]["sha256"] == reference.source.sha256

    def test_empty_reference_is_refused(self, tmp_path) -> None:
        bad = tmp_path / "bad_reference.json"
        bad.write_text(
            json.dumps(
                {
                    "source": json.loads(
                        _resource(ECPS_PARITY_REFERENCE_RESOURCE).read_text()
                    )["source"],
                    "nonzero_shares": {},
                }
            )
        )
        # A silently-empty reference would make the parity gate vacuous.
        with pytest.raises(ValueError, match="vacuous"):
            load_ecps_parity_reference(str(bad))

    def test_load_accepts_an_explicit_resource_path(self, tmp_path) -> None:
        payload = json.loads(_resource(ECPS_PARITY_REFERENCE_RESOURCE).read_text())
        alt = tmp_path / "alt.json"
        alt.write_text(json.dumps(payload))
        assert load_ecps_parity_reference(str(alt)).source.sha256 == (
            load_ecps_parity_reference().source.sha256
        )

    def test_recorded_sha256_matches_the_cached_incumbent_bytes(self) -> None:
        # When the pinned incumbent artifact is already in the local Hugging
        # Face cache, the recorded sha256 must equal its real content digest.
        # Cache-only (no network) so CI stays hermetic; skipped when absent.
        source = load_ecps_parity_reference().source
        cached = _cached_incumbent_path(source)
        if cached is None:
            pytest.skip("pinned incumbent artifact is not in the local HF cache")
        digest = hashlib.sha256(Path(cached).read_bytes()).hexdigest()
        assert digest == source.sha256


class TestKnownGapsRegister:
    def test_every_entry_has_a_reason_and_issue(self) -> None:
        register = load_ecps_parity_known_gaps()
        assert register
        for gap in register:
            assert gap.reason.strip(), gap.variable
            assert gap.issue.strip(), gap.variable
            # Issue refs point at a real repo#number.
            assert "#" in gap.issue, gap.variable

    def test_register_names_only_reference_layers(self) -> None:
        # A gap must be a layer the incumbent actually populates — exempting a
        # variable the reference never had would be register rot.
        reference = load_ecps_parity_reference()
        populated = set(reference.populated_layers)
        stray = sorted(
            gap.variable
            for gap in load_ecps_parity_known_gaps()
            if gap.variable not in populated
        )
        assert stray == [], f"register exempts non-reference layers: {stray}"

    def test_register_exempts_no_release_produced_layer(self) -> None:
        # The release pipeline deterministically materializes these inputs on
        # the base frame before the parity gate runs (immigration, hours,
        # SNAP take-up, and eligibility derivations, plus take-up flags the
        # contract marks as populace-produced), so an exemption for any of
        # them is stale on arrival. parity_gate enforces this at build time
        # from live candidate shares; this pins the statically-knowable
        # subset so the PR that wires a producing stage or flips a program's
        # contract treatment must remove its register line here, not first on
        # a release build (the #293 hours stage merged with its two register
        # lines intact; only this test's absence let that through). Contract
        # treatments outside this set (out_of_scope, rate_unsourced) are
        # produced elsewhere or not at all, and their producing PR owns the
        # register line (the #294 SNAP pattern).
        produced_treatments = {"seed", "count_calibrated"}
        produced = (
            set(US_IMMIGRATION_OUTPUT_COLUMNS)
            | set(US_HOURS_WORKED_OUTPUT_COLUMNS)
            | set(US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS)
            | set(US_SIPP_TIPS_OUTPUT_COLUMNS)
            | {US_SNAP_TAKE_UP_OUTPUT_COLUMN}
            | {
                program.variable
                for program in load_take_up_contract().programs
                if program.populace_treatment in produced_treatments
            }
        )
        stale = sorted(
            gap.variable
            for gap in load_ecps_parity_known_gaps()
            if gap.variable in produced
        )
        assert stale == [], f"register exempts release-produced layers: {stale}"

    def test_register_entries_are_unique(self) -> None:
        register = load_ecps_parity_known_gaps()
        names = [gap.variable for gap in register]
        assert len(names) == len(set(names))

    def test_missing_reason_is_refused(self, tmp_path) -> None:
        bad = tmp_path / "bad_gaps.json"
        bad.write_text(
            json.dumps(
                {"known_gaps": {"some_var": {"reason": "", "issue": "org/repo#1"}}}
            )
        )
        with pytest.raises(ValueError, match="reason"):
            load_ecps_parity_known_gaps(str(bad))

    def test_missing_issue_is_refused(self, tmp_path) -> None:
        bad = tmp_path / "bad_gaps.json"
        bad.write_text(json.dumps({"known_gaps": {"some_var": {"reason": "because"}}}))
        with pytest.raises(ValueError, match="issue"):
            load_ecps_parity_known_gaps(str(bad))
