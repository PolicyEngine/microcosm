"""Reference-pinned, recorded, reconstructable parity (issue #19).

The bug these tests pin down: parity ``gaps=0`` was certified against an
*unpinned* enhanced-CPS reference, the release manifest recorded no gate
result, and the runner that produced the verdict was later deleted — so the
verdict was neither reproducible nor auditable. The fixes under test:

- a :class:`~populace.build.parity_reference.ReferenceSpec` that refuses to
  identify a reference without a sha256 (an unpinned reference is the bug);
- :func:`~populace.build.parity_reference.judge_parity`, a pure function that
  reuses the surviving :func:`populace.build.parity_gate` and **records the
  reference identity** (repo/revision/sha256 or path/sha256) plus the
  gap/skip/populated-layer counts in the :class:`GateResult` details — so the
  verdict carries what it was judged against;
- :func:`~populace.build.parity_reference.gather_candidate_shares`, the
  simulation-level share gathering ported from the deleted ``check_parity.py``,
  with the ``Microsimulation`` injected as a plain callable so the skip logic
  and structural-weight popping are testable **without** ``policyengine_us`` or
  a 355 MB dataset;
- a round-trip: a judged :class:`GateResult` serializes into a manifest-style
  dict that carries the reference identity, closing manifest-gap #2.

Every test here runs with no network and no large files.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from populace.build import GateResult
from populace.build.parity_reference import (
    ReferenceSpec,
    gather_candidate_shares,
    judge_parity,
    reference_layers,
)

# A real eCPS sha256 is 64 hex chars; use a fixed illustrative digest so the
# recorded identity is checkable byte-for-byte.
_SHA = hashlib.sha256(b"enhanced_cps_2024.h5 reference bytes").hexdigest()


class TestReferenceSpec:
    def test_local_spec_records_path_and_sha(self) -> None:
        spec = ReferenceSpec(path="storage/enhanced_cps_2024.h5", sha256=_SHA)
        assert spec.sha256 == _SHA
        identity = spec.to_dict()
        assert identity["path"] == "storage/enhanced_cps_2024.h5"
        assert identity["sha256"] == _SHA
        assert identity["kind"] == "local"

    def test_hf_spec_records_repo_revision_and_sha(self) -> None:
        spec = ReferenceSpec(
            repo="policyengine/policyengine-us-data",
            revision="4a8e7d39eb9e",
            sha256=_SHA,
        )
        identity = spec.to_dict()
        assert identity["repo"] == "policyengine/policyengine-us-data"
        assert identity["revision"] == "4a8e7d39eb9e"
        assert identity["sha256"] == _SHA
        assert identity["kind"] == "huggingface"

    def test_missing_sha256_is_refused(self) -> None:
        # The whole point of #19: an unpinned reference is the bug.
        with pytest.raises(ValueError, match="sha256"):
            ReferenceSpec(path="storage/enhanced_cps_2024.h5", sha256="")

    def test_no_sha256_argument_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            ReferenceSpec(repo="r", revision="rev", sha256=None)  # type: ignore[arg-type]

    def test_hf_without_repo_or_revision_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repo.*revision|revision.*repo"):
            ReferenceSpec(repo="policyengine/x", sha256=_SHA)  # revision missing

    def test_local_without_path_or_hf_without_repo_is_refused(self) -> None:
        with pytest.raises(ValueError, match="path|repo"):
            ReferenceSpec(sha256=_SHA)  # neither local path nor HF repo

    def test_from_local_file_hashes_the_bytes(self, tmp_path) -> None:
        payload = b"enhanced cps reference bytes" * 1000
        ref = tmp_path / "enhanced_cps_2024.h5"
        ref.write_bytes(payload)
        spec = ReferenceSpec.from_local_file(ref)
        assert spec.sha256 == hashlib.sha256(payload).hexdigest()
        assert spec.path == str(ref)
        assert spec.kind == "local"


class TestJudgeParityRecordsReference:
    def _spec(self) -> ReferenceSpec:
        return ReferenceSpec(
            repo="policyengine/policyengine-us-data",
            revision="4a8e7d39eb9e",
            sha256=_SHA,
        )

    def test_gap_when_reference_populates_a_layer_candidate_is_zero(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1, "general_business_credit": 0.0},
            reference_shares={"snap": 0.12, "general_business_credit": 0.054},
            reference_spec=self._spec(),
        )
        assert not result.passed
        assert "general_business_credit" in result.failures[0]

    def test_pass_when_candidate_populates_every_reference_layer(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1, "net_worth": 0.5},
            reference_shares={"snap": 0.12, "net_worth": 0.9},
            reference_spec=self._spec(),
        )
        assert result.passed
        assert result.details["gaps"] == 0

    def test_known_gap_exemption_is_honored_and_recorded(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1, "amt_foreign_tax_credit": 0.0},
            reference_shares={"snap": 0.12, "amt_foreign_tax_credit": 0.093},
            reference_spec=self._spec(),
            known_gaps=["amt_foreign_tax_credit"],
        )
        assert result.passed
        assert result.details["exempted"] == ["amt_foreign_tax_credit"]

    def test_reference_identity_is_recorded_in_details(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1},
            reference_shares={"snap": 0.12},
            reference_spec=self._spec(),
        )
        ref = result.details["reference"]
        assert ref["sha256"] == _SHA
        assert ref["repo"] == "policyengine/policyengine-us-data"
        assert ref["revision"] == "4a8e7d39eb9e"
        assert ref["kind"] == "huggingface"

    def test_local_reference_identity_is_recorded(self) -> None:
        spec = ReferenceSpec(path="storage/enhanced_cps_2024.h5", sha256=_SHA)
        result = judge_parity(
            candidate_shares={"snap": 0.1},
            reference_shares={"snap": 0.12},
            reference_spec=spec,
        )
        ref = result.details["reference"]
        assert ref["path"] == "storage/enhanced_cps_2024.h5"
        assert ref["sha256"] == _SHA

    def test_layer_and_skipped_counts_are_recorded(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1, "net_worth": 0.5},
            reference_shares={"snap": 0.12, "net_worth": 0.9, "vacuous": 0.0},
            reference_spec=self._spec(),
            skipped=("non_annual_var", "unregistered_var"),
        )
        # populated reference layers: snap + net_worth (vacuous is 0%)
        assert result.details["reference_populated_layers"] == 2
        # candidate layers we actually measured a non-zero share for
        assert result.details["candidate_populated_layers"] == 2
        assert result.details["skipped_layers"] == 2
        assert list(result.details["skipped"]) == [
            "non_annual_var",
            "unregistered_var",
        ]

    def test_judge_returns_a_gateresult_named_parity(self) -> None:
        result = judge_parity(
            candidate_shares={"snap": 0.1},
            reference_shares={"snap": 0.12},
            reference_spec=self._spec(),
        )
        assert isinstance(result, GateResult)
        assert result.name == "parity"

    def test_judge_preserves_parity_gate_failure_text(self) -> None:
        # The recorded result must not invent its own caveats: failures come
        # verbatim from the surviving parity_gate.
        from populace.build import parity_gate

        cand = {"snap": 0.0}
        ref = {"snap": 0.12}
        bare = parity_gate(cand, ref)
        recorded = judge_parity(
            candidate_shares=cand, reference_shares=ref, reference_spec=self._spec()
        )
        assert recorded.failures == bare.failures


class TestGatherCandidateShares:
    """Port of check_parity.py's simulation loop, sim injected as a callable."""

    class _FakeVariable:
        def __init__(self, definition_period: str) -> None:
            self.definition_period = definition_period

    def _tbs(self):
        """A stub tax-benefit system: a few annual vars, one monthly, and
        deliberately *no* entry for the unregistered reference layer."""

        class _TBS:
            variables = {
                "snap": TestGatherCandidateShares._FakeVariable("year"),
                "net_worth": TestGatherCandidateShares._FakeVariable("year"),
                "household_weight": TestGatherCandidateShares._FakeVariable("year"),
                "person_weight": TestGatherCandidateShares._FakeVariable("year"),
                "monthly_thing": TestGatherCandidateShares._FakeVariable("month"),
            }

        return _TBS()

    def _calculate(self, shares_by_var):
        """A fake sim: returns arrays whose non-zero share matches the spec."""

        def calculate(var: str, year: int) -> np.ndarray:
            n = 10
            nonzero = round(shares_by_var[var] * n)
            return np.asarray([1.0] * nonzero + [0.0] * (n - nonzero))

        return calculate

    def test_unregistered_and_non_annual_layers_are_skipped(self) -> None:
        reference = {
            "snap": 0.12,
            "net_worth": 0.9,
            "monthly_thing": 0.5,  # non-annual -> skipped
            "amt_foreign_tax_credit": 0.093,  # not in tbs.variables -> skipped
        }
        candidate, ref_out, skipped = gather_candidate_shares(
            reference,
            year=2024,
            tax_benefit_system=self._tbs(),
            calculate=self._calculate({"snap": 0.1, "net_worth": 0.5}),
        )
        assert set(skipped) == {"monthly_thing", "amt_foreign_tax_credit"}
        assert set(candidate) == {"snap", "net_worth"}
        # the reference shares returned are aligned to the kept (non-skipped,
        # non-structural) layers only
        assert set(ref_out) == {"snap", "net_worth"}
        assert candidate["snap"] == pytest.approx(0.1)
        assert candidate["net_worth"] == pytest.approx(0.5)
        assert ref_out["snap"] == pytest.approx(0.12)

    def test_structural_weights_are_popped(self) -> None:
        reference = {
            "snap": 0.12,
            "household_weight": 1.0,
            "person_weight": 1.0,
        }
        candidate, ref_out, skipped = gather_candidate_shares(
            reference,
            year=2024,
            tax_benefit_system=self._tbs(),
            calculate=self._calculate(
                {"snap": 0.1, "household_weight": 1.0, "person_weight": 1.0}
            ),
        )
        assert "household_weight" not in candidate
        assert "person_weight" not in candidate
        assert "household_weight" not in ref_out
        assert "person_weight" not in ref_out
        assert "snap" in candidate

    def test_calc_failure_records_zero_share_not_a_crash(self) -> None:
        # check_parity.py reported a failing sim.calculate as candidate 0.0
        # (a real gap) rather than masking it; the port keeps that contract.
        def calculate(var: str, year: int) -> np.ndarray:
            if var == "net_worth":
                raise RuntimeError("formula blew up")
            return np.asarray([1.0, 0.0])

        candidate, ref_out, skipped = gather_candidate_shares(
            {"snap": 0.12, "net_worth": 0.9},
            year=2024,
            tax_benefit_system=self._tbs(),
            calculate=calculate,
        )
        assert candidate["net_worth"] == 0.0
        assert ref_out["net_worth"] == 0.9
        # and it is NOT counted as skipped — it is a measured (failed) layer
        assert "net_worth" not in skipped

    def test_gather_then_judge_finds_the_gap(self) -> None:
        # End-to-end of the pure half: gather over a fake sim, judge with a
        # pinned spec, get a recorded gap.
        reference = {"snap": 0.12, "net_worth": 0.9}
        candidate, ref_out, skipped = gather_candidate_shares(
            reference,
            year=2024,
            tax_benefit_system=self._tbs(),
            calculate=self._calculate({"snap": 0.1, "net_worth": 0.0}),
        )
        result = judge_parity(
            candidate_shares=candidate,
            reference_shares=ref_out,
            reference_spec=ReferenceSpec(path="x.h5", sha256=_SHA),
            skipped=skipped,
        )
        assert not result.passed
        assert "net_worth" in result.failures[0]
        assert result.details["reference"]["sha256"] == _SHA


class TestReferenceLayers:
    """Port of check_parity.py's stored_layers: flat var/YEAR in an H5 file."""

    def test_reads_flat_var_year_layout(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "ref.h5"
        with h5py.File(path, "w") as f:
            # var/YEAR layout, the real eCPS shape
            f.create_dataset("snap/2024", data=np.asarray([0.0, 1.0, 2.0, 0.0]))
            f.create_dataset("net_worth/2024", data=np.asarray([10.0, 20.0, 0.0, 5.0]))
            f.create_dataset("all_zero/2024", data=np.zeros(4))
        layers = reference_layers(path, year=2024)
        assert layers["snap"] == pytest.approx(0.5)  # 2 of 4 non-zero
        assert layers["net_worth"] == pytest.approx(0.75)  # 3 of 4
        assert layers["all_zero"] == 0.0

    def test_string_columns_are_skipped(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "ref.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("snap/2024", data=np.asarray([1.0, 0.0]))
            f.create_dataset(
                "county_fips/2024",
                data=np.asarray([b"06037", b"36061"], dtype="S5"),
            )
        layers = reference_layers(path, year=2024)
        assert "snap" in layers
        assert "county_fips" not in layers  # |S5 string column dropped

    def test_only_the_requested_year_is_read(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "ref.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("snap/2024", data=np.asarray([1.0, 0.0]))
            f.create_dataset("snap/2023", data=np.asarray([1.0, 1.0]))
        layers = reference_layers(path, year=2024)
        assert layers["snap"] == pytest.approx(0.5)  # the 2024 layer, not 2023
        # The off-year layer must not leak a spurious variable: a multi-year
        # file must not record the wrong year, nor a year token as a var name.
        assert set(layers) == {"snap"}
        assert "2023" not in layers

    def test_entity_variable_single_year_layout(self, tmp_path) -> None:
        # The other layout the deleted runner accepted: entity/variable with
        # no year token. The trailing component is the variable.
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "ref.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "person/employment_income", data=np.asarray([1.0, 0.0, 2.0, 0.0])
            )
        layers = reference_layers(path, year=2024)
        assert layers["employment_income"] == pytest.approx(0.5)


class TestPackageReexports:
    def test_symbols_are_importable_from_populace_build(self) -> None:
        # gates/holdout/plan are re-exported at the package root; the parity
        # reference surface joins them so callers import one place.
        import populace.build as build

        for name in (
            "ReferenceSpec",
            "judge_parity",
            "gather_candidate_shares",
            "reference_layers",
            "run_parity_against_reference",
        ):
            assert hasattr(build, name)
            assert name in build.__all__


class TestManifestRoundTrip:
    def test_recorded_result_serializes_into_a_manifest_dict(self) -> None:
        from populace.build import GateReport

        spec = ReferenceSpec(
            repo="policyengine/policyengine-us-data",
            revision="4a8e7d39eb9e",
            sha256=_SHA,
        )
        result = judge_parity(
            candidate_shares={"snap": 0.1, "net_worth": 0.5},
            reference_shares={"snap": 0.12, "net_worth": 0.9},
            reference_spec=spec,
            skipped=("monthly_thing",),
        )
        manifest = GateReport((result,)).to_manifest()
        parity = manifest["gates"]["parity"]
        assert parity["passed"] is True
        # The reference identity survives the round-trip into the manifest:
        # this is exactly the gap #19 calls out (the certified manifest
        # recorded no reference identity).
        assert parity["details"]["reference"]["sha256"] == _SHA
        assert parity["details"]["reference"]["revision"] == "4a8e7d39eb9e"
        assert parity["details"]["skipped_layers"] == 1
        # JSON-roundtrippable (no non-serializable objects leaked into details)
        import json

        json.loads(json.dumps(manifest))
