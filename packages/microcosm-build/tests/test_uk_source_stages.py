from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.source_manifest import (
    FORBIDDEN_SOURCE_DEPENDENCIES,
    SourceManifest,
)
from microcosm.frame import Frame

ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
FROZEN_SOURCE_STAGES = UK_PACKAGE / "hmrc_income_source_stages.json"
CANONICAL_SOURCE_STAGES = UK_PACKAGE / "source_stages.json"
FROZEN_SOURCE_STAGES_SHA256 = (
    "c0341af7166ae3a85a3c1164e7d9e880c4b4aec122f1a8fa90c73b46c596e1ea"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: Frame) -> Frame:
    return frame


def _assert_no_forbidden_dependency(value: object) -> None:
    text = json.dumps(value, sort_keys=True).lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        assert dependency not in text


def _expected_reviewed_source() -> str:
    return (
        "PolicyEngine licensed UKDS mirror (private Hugging Face repository), "
        "spi_2022_23.zip"
    )


def _rephrase_stage2_predictor_note(value: str) -> str:
    return value.replace(
        "policyengine-" + "uk-data frs_only.py",
        "the incumbent UK data build's frs_only.py",
    )


class TestUKSourceStagesManifest:
    def test_source_stages_json_loads_as_shared_manifest(self) -> None:
        manifest = SourceManifest.from_mapping(_load_json(CANONICAL_SOURCE_STAGES))

        assert manifest.country == "uk"
        assert manifest.version == 1
        assert [stage.stage for stage in manifest.stages] == [
            "frs_spine",
            "frs_hmrc_retained_leaves",
            "hmrc_spi_income",
        ]

    def test_country_spec_declares_three_uk_source_stages(self) -> None:
        spec = load_country_spec("uk")

        assert spec.sources is not None
        assert [stage.stage for stage in spec.sources.stages] == [
            "frs_spine",
            "frs_hmrc_retained_leaves",
            "hmrc_spi_income",
        ]

    def test_copy_is_lockstep_with_frozen_original_except_citation_rewrites(
        self,
    ) -> None:
        frozen = _load_json(FROZEN_SOURCE_STAGES)
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        frozen_stage = frozen["stages"][0]
        _, stage1, stage2 = canonical["stages"]

        expected_operations = copy.deepcopy(frozen_stage["operations"])
        predictor_note = expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ]
        expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ] = _rephrase_stage2_predictor_note(predictor_note)

        assert stage1["operations"] + stage2["operations"] == expected_operations
        _assert_no_forbidden_dependency(
            stage2["operations"][4]["reviewed_absent_predictors"][
                "other_investment_income"
            ]
        )

        expected_artifacts = copy.deepcopy(frozen_stage["artifacts"])
        expected_artifacts[0]["reviewed_source"] = _expected_reviewed_source()
        # Declared output-name correction (licensed-data acceptance finding):
        # the frozen original listed the SPI concept "state_pension", but the
        # stage writes the auxiliary column SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        # ("hmrc_spi_state_pension_income") — the model input state_pension is
        # formula-owned and never a frame column here. Outputs became
        # load-bearing when country_stage_plan compiled them into
        # StagePlan.produces, so the copy declares the persisted truth. The
        # operation payloads keep the concept name unchanged.
        expected_outputs = [
            "hmrc_spi_state_pension_income" if name == "state_pension" else name
            for name in frozen_stage["outputs"]
        ]
        assert stage2["outputs"] == expected_outputs
        assert stage2["grain"] == frozen_stage["grain"]
        assert stage2["artifacts"] == expected_artifacts
        _assert_no_forbidden_dependency(stage2["artifacts"])
        _assert_no_forbidden_dependency(stage2["notes"])

    def test_frozen_original_bytes_are_pinned(self) -> None:
        digest = hashlib.sha256(FROZEN_SOURCE_STAGES.read_bytes()).hexdigest()

        assert digest == FROZEN_SOURCE_STAGES_SHA256

    def test_country_stage_plan_assembles_two_uk_national_stages(self) -> None:
        spec = load_country_spec("uk")
        plan = country_stage_plan(
            spec,
            {
                "frs_hmrc_retained_leaves": _identity,
                "hmrc_spi_income": _identity,
            },
            stage_names=("frs_hmrc_retained_leaves", "hmrc_spi_income"),
        )

        assert [stage.name for stage in plan.stages] == [
            "frs_hmrc_retained_leaves",
            "hmrc_spi_income",
        ]

    @pytest.mark.parametrize(
        "implementations, match",
        [
            ({"frs_hmrc_retained_leaves": _identity}, "missing"),
            (
                {
                    "frs_spine": _identity,
                    "frs_hmrc_retained_leaves": _identity,
                    "hmrc_spi_income": _identity,
                    "hmrc_spi_income_fallback": _identity,
                },
                "Unknown stage implementation",
            ),
        ],
    )
    def test_country_stage_plan_refuses_missing_or_unknown_uk_stage(
        self,
        implementations,
        match: str,
    ) -> None:
        spec = load_country_spec("uk")

        with pytest.raises(ValueError, match=match):
            country_stage_plan(spec, implementations)


class TestDeclaredOutputsAreWrittenColumns:
    """Declared outputs must name columns the stages actually write.

    Outputs are load-bearing (``country_stage_plan`` compiles them into
    ``StagePlan.produces``), and the licensed-data acceptance for this
    migration caught a declared output that was an SPI *concept* rather
    than a persisted column — harmless while nothing read the field,
    refused at full rung once it did. This pins every declared output to
    a named runtime written-column constant so the class cannot recur
    without a licensed build to find it (microcosm#690 review).
    """

    def test_stage1_outputs_are_exactly_the_retained_leaf_columns(self) -> None:
        from microcosm.build.uk_runtime.frs_hmrc_leaves import (
            FRS_HMRC_RETAINED_LEAF_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage1 = stages["frs_hmrc_retained_leaves"]
        assert stage1.outputs == tuple(FRS_HMRC_RETAINED_LEAF_COLUMNS)

    def test_stage2_outputs_are_backed_by_runtime_written_columns(self) -> None:
        from microcosm.build.uk_runtime.spi_support import (
            SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
            SPI_HMRC_QRF_AUXILIARY_COLUMNS,
            SPI_INCOME_IMPUTATION_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage2 = stages["hmrc_spi_income"]
        written = (
            set(SPI_INCOME_IMPUTATION_COLUMNS)
            | set(SPI_HMRC_QRF_AUXILIARY_COLUMNS)
            | set(SPI_HMRC_DERIVED_AUXILIARY_COLUMNS)
        )
        # The narrow PAY+EPB+TAXTERM employment input is written on SPI rows
        # by the stage even though the QRF output surface excludes it.
        written.add("employment_income")
        unbacked = [name for name in stage2.outputs if name not in written]
        assert unbacked == [], (
            "Declared outputs with no named runtime written-column constant "
            f"backing them: {unbacked}. Either the manifest declares a "
            "concept instead of a persisted column, or the runtime constant "
            "moved without the manifest following."
        )
