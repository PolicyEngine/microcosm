from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime import US_SOURCE_MANIFEST
from populace.build.us_runtime.capital_gain_distributions import (
    load_capital_gain_distribution_shares,
    split_us_component_by_share_from_manifest,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_SOURCE_COLUMN = "long_term_capital_gains_before_response"
_EXCLUSIVE_COLUMN = "non_sch_d_capital_gains"
_OUTPUT_COLUMN = "schedule_d_capital_gain_distributions"


def _toy_tax_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": [1, 2, 3, 4, 5],
            "tax_unit_weight": [100.0, 100.0, 100.0, 100.0, 100.0],
            "long_term_capital_gains_before_response": [
                10_000.0,
                0.0,
                -4_000.0,
                50_000.0,
                8_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 0.0, 0.0, 0.0, 3_000.0],
        }
    )


def _toy_us_frame() -> Frame:
    """Place the existing tax-unit fixture's inputs on PUF-style people."""

    toy = _toy_tax_units()
    tax_unit_ids = toy["tax_unit_id"].to_numpy(dtype="int64")
    person_tax_unit_ids = np.asarray([1, 1, 2, 3, 4, 5], dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype="int64"),
            "person_household_id": person_tax_unit_ids,
            "person_tax_unit_id": person_tax_unit_ids,
            "person_spm_unit_id": person_tax_unit_ids,
            "person_family_id": person_tax_unit_ids,
            "person_marital_unit_id": person_tax_unit_ids,
            # Split tax unit 1 across two people so the test exercises the
            # person -> tax-unit -> first-person placement seam.
            _SOURCE_COLUMN: np.asarray(
                [6_000.0, 4_000.0, 0.0, -4_000.0, 50_000.0, 8_000.0]
            ),
            _EXCLUSIVE_COLUMN: np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 3_000.0]),
        }
    )
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": tax_unit_ids})
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                toy["tax_unit_weight"].to_numpy(dtype=float),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["puf_tax_detail"] * len(person), name="stratum"),
    )


def _load_support_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capital_gain_stage_args(builder, checkpoint_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        stage=builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
        checkpoint_dir=checkpoint_dir,
        seed=0,
        target_year=2024,
    )


def _prime_capital_gain_stage_predecessor(
    builder,
    checkpoint_dir: Path,
    frame: Frame,
):
    """Commit the exact outer-stage prefix ending at CG tail transfer."""

    runtime = builder.StageRuntime(
        checkpoint_dir,
        builder.OUTER_STAGE_PIPELINE,
        run_config={},
    )
    runtime.complete("source_construction", frame)
    runtime.complete("pre_clone_enrichment", frame)
    runtime.complete("clone_feature_extraction", frame)
    runtime.complete_without_frame("primary_qrf_chain")
    runtime.complete("qrf_finalization", frame)
    runtime.complete(builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME, frame)
    return runtime


def _split_operation(**overrides) -> SourceOperationSpec:
    raw = {
        "kind": "split_component_by_share",
        "resource": "soca_capital_gain_distribution_shares",
        "share_field": "schedule_d_cgd_share_of_lt_net_gains",
        "source_column": "long_term_capital_gains_before_response",
        "output": "schedule_d_capital_gain_distributions",
        "exclusive_with": ["non_sch_d_capital_gains"],
    }
    raw.update(overrides)
    return SourceOperationSpec.from_mapping(raw)


def _context() -> SourceRuntimeContext:
    return SourceRuntimeContext(config=SourceRuntimeConfig(), tables={})


class TestSharesResource:
    def test_packaged_resource_rederives_its_own_anchor(self) -> None:
        shares = load_capital_gain_distribution_shares()
        share = shares.schedule_d_cgd_share_of_lt_net_gains
        anchor = shares.anchor
        residual = (
            anchor["soca_all_route_capital_gain_distributions_k"]
            - anchor["pub1304_direct_1040_route_k"]
        )
        assert np.isclose(residual, anchor["schedule_d_route_residual_k"])
        assert np.isclose(
            share,
            residual
            / (
                anchor["soca_long_term_total_net_gain_k"]
                - anchor["pub1304_direct_1040_route_k"]
            ),
            rtol=1e-6,
        )
        # TY2015: $68.1B Schedule-D route over $690.7B of Schedule-D LT
        # net gains — the split is a bit under ten percent.
        assert 0.09 < share < 0.11


class TestCapitalGainDistributionsStage:
    def test_manifest_stage_splits_the_schedule_d_route(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()["capital_gain_distributions"]
        toy = _toy_tax_units()

        result = run_source_stage(
            stage,
            tables={"tax_unit": toy},
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=0, target_year=2024),
        )

        share = (
            load_capital_gain_distribution_shares().schedule_d_cgd_share_of_lt_net_gains
        )
        out = result["schedule_d_capital_gain_distributions"]
        assert np.isclose(out.iloc[0], 10_000.0 * share)
        # Zero and negative long-term gains get nothing.
        assert out.iloc[1] == 0.0
        assert out.iloc[2] == 0.0
        assert np.isclose(out.iloc[3], 50_000.0 * share)
        # The two reporting routes are mutually exclusive on a real return.
        assert out.iloc[4] == 0.0
        # The source column is untouched and the split never exceeds it.
        pd.testing.assert_series_equal(
            result["long_term_capital_gains_before_response"],
            toy["long_term_capital_gains_before_response"],
        )
        assert (
            out
            <= result["long_term_capital_gains_before_response"].clip(lower=0.0) + 1e-9
        ).all()
        # National reconstruction: the split is exactly proportional.
        eligible_lt = toy.loc[
            (toy["long_term_capital_gains_before_response"] > 0)
            & ~(toy["non_sch_d_capital_gains"] > 0),
            "long_term_capital_gains_before_response",
        ].sum()
        assert np.isclose(out.sum(), share * eligible_lt)

    def test_base_outer_stage_records_and_materializes_the_declared_split(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        builder = _load_support_builder_module()
        frame = _toy_us_frame()
        args = _capital_gain_stage_args(builder, tmp_path)
        runtime = _prime_capital_gain_stage_predecessor(builder, tmp_path, frame)
        monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
        monkeypatch.setattr(
            builder,
            "profile_stage",
            lambda *_args, **_kwargs: nullcontext(),
        )

        builder._run_outer_stage(args)

        completed = runtime.load(builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME)
        actual_frame = completed.frame
        # The base-builder grain adapter must preserve both PUF-produced person
        # inputs exactly; only the new memo output is added.
        pd.testing.assert_series_equal(
            actual_frame.person[_SOURCE_COLUMN],
            frame.person[_SOURCE_COLUMN],
        )
        pd.testing.assert_series_equal(
            actual_frame.person[_EXCLUSIVE_COLUMN],
            frame.person[_EXCLUSIVE_COLUMN],
        )
        assert _OUTPUT_COLUMN in actual_frame.person
        assert _OUTPUT_COLUMN not in actual_frame.table("tax_unit")

        toy = _toy_tax_units()
        expected = run_source_stage(
            US_SOURCE_MANIFEST.stage_map()["capital_gain_distributions"],
            tables={"tax_unit": toy},
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=0, target_year=2024),
        )
        actual_tax_units = actual_frame.place(
            _OUTPUT_COLUMN,
            "tax_unit",
            how="sum",
        ).table("tax_unit")
        np.testing.assert_allclose(
            actual_tax_units[_OUTPUT_COLUMN].to_numpy(),
            expected[_OUTPUT_COLUMN].to_numpy(),
        )

        # PolicyEngine owns the output on Person and aggregates it to the tax
        # unit, so the neighboring-stage convention is a deterministic
        # first-person carry rather than an invented filer/spouse allocation.
        memberships = actual_frame.person["person_tax_unit_id"]
        first_person = ~memberships.duplicated()
        expected_by_id = pd.Series(
            expected[_OUTPUT_COLUMN].to_numpy(),
            index=toy["tax_unit_id"].to_numpy(),
        )
        expected_person = np.zeros(len(actual_frame.person), dtype=float)
        expected_person[first_person.to_numpy()] = memberships[first_person].map(
            expected_by_id
        )
        np.testing.assert_allclose(
            actual_frame.person[_OUTPUT_COLUMN].to_numpy(),
            expected_person,
        )

        context = json.loads(
            (tmp_path / "stage_run_context.json").read_text(encoding="utf-8")
        )
        pipeline_names = [entry["name"] for entry in context["pipeline"]["stages"]]
        stage_index = pipeline_names.index(
            builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME
        )
        assert pipeline_names[stage_index - 1 : stage_index + 2] == [
            builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
            builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
            "qbi_reconciliation",
        ]
        assert context["completed"] == pipeline_names[: stage_index + 1]
        stage_record = context["stage_records"][stage_index]
        assert stage_record["stage"] == builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME
        assert (
            stage_record["checkpoint_stage"]
            == builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME
        )
        assert stage_record["checkpoint_filename"] == (
            f"{stage_index:03d}_{builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME}"
            ".frame.h5"
        )
        assert stage_record["wrote_frame"] is True

    def test_base_outer_stage_fails_loudly_when_predecessor_has_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        builder = _load_support_builder_module()
        args = _capital_gain_stage_args(builder, tmp_path)
        populated, _metadata = builder._capital_gain_distributions_stage(
            args,
            _toy_us_frame(),
        )
        runtime = _prime_capital_gain_stage_predecessor(
            builder,
            tmp_path,
            populated,
        )
        monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
        monkeypatch.setattr(
            builder,
            "profile_stage",
            lambda *_args, **_kwargs: nullcontext(),
        )

        with pytest.raises(SourceRuntimeError, match="already exists"):
            builder._run_outer_stage(args)

        # This is an uncompleted CGD stage whose predecessor already carries
        # the output, not a normal resume of an already-committed stage.
        assert (
            runtime.context.completed[-1] == builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME
        )
        assert builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME not in (
            runtime.context.completed
        )
        context = json.loads(
            (tmp_path / "stage_run_context.json").read_text(encoding="utf-8")
        )
        assert all(
            record["stage"] != builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME
            for record in context["stage_records"]
        )

    def test_split_requires_a_frame(self) -> None:
        with pytest.raises(SourceRuntimeError, match="must follow read_table"):
            split_us_component_by_share_from_manifest(
                None, _split_operation(), _context()
            )

    def test_split_rejects_unknown_parameters(self) -> None:
        with pytest.raises(SourceRuntimeError, match="unexpected parameter"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(), _split_operation(bogus=1), _context()
            )

    def test_split_rejects_unknown_resource_and_share_field(self) -> None:
        with pytest.raises(SourceRuntimeError, match="knows only"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(),
                _split_operation(resource="somewhere_else"),
                _context(),
            )
        with pytest.raises(SourceRuntimeError, match="share_field"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(),
                _split_operation(share_field="not_a_field"),
                _context(),
            )

    def test_split_refuses_to_overwrite_an_existing_output(self) -> None:
        frame = _toy_tax_units()
        frame["schedule_d_capital_gain_distributions"] = 1.0
        with pytest.raises(SourceRuntimeError, match="already exists"):
            split_us_component_by_share_from_manifest(
                frame, _split_operation(), _context()
            )

    def test_split_requires_source_and_exclusive_columns(self) -> None:
        missing_source = _toy_tax_units().drop(
            columns=["long_term_capital_gains_before_response"]
        )
        with pytest.raises(SourceRuntimeError, match="source column"):
            split_us_component_by_share_from_manifest(
                missing_source, _split_operation(), _context()
            )
        missing_exclusive = _toy_tax_units().drop(columns=["non_sch_d_capital_gains"])
        with pytest.raises(SourceRuntimeError, match="exclusive_with"):
            split_us_component_by_share_from_manifest(
                missing_exclusive, _split_operation(), _context()
            )

    def test_split_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["split_component_by_share"]
            is split_us_component_by_share_from_manifest
        )
