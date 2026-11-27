"""A base step frees its input frame before it writes its checkpoint.

``_run_outer_stage`` loads the previous step's checkpoint, and
``StageRuntime.load`` computes the loaded frame's identity (ordered IDs,
memberships, clone metadata and provenance) and checks it against the run
context. The step keeps that identity, drops every reference to the input
frame once the step function has returned (in ``clone_feature_extraction``,
once the clone check has read it), and compares the step's result with the
kept identity before the step is recorded.
"""

from __future__ import annotations

import gc
import importlib.util
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import outer_stage_runtime
from microcosm.build.us_runtime.gate_shadow import current_gate_shadow
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_gate_shadow import arm_us_tool_gate_shadows
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_PASSING_GATE = SimpleNamespace(passed=True, failures=(), details={})

# One step for each branch of _run_outer_stage that loads the previous
# checkpoint and writes a frame of its own, plus a second post-QRF step that
# runs signal checks. final_export is left out: it writes no frame.
_FRAME_WRITING_STEPS = (
    "pre_clone_enrichment",
    "clone_feature_extraction",
    "qrf_finalization",
    "capital_gains_tail_transfer",
    "capital_gain_distributions",
    "qbi_reconciliation",
    "wic_post_clone",
    "household_location_assignment",
)
# One step for each branch whose result must keep the input's identity and
# row order.
_IDENTITY_KEEPING_STEPS = (
    "pre_clone_enrichment",
    "qrf_finalization",
    "capital_gain_distributions",
    "qbi_reconciliation",
    "household_location_assignment",
)


@pytest.fixture(scope="module")
def builder():
    path = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_us_frame() -> Frame:
    tax_unit_ids = np.arange(1, 6, dtype="int64")
    person_tax_unit_ids = np.asarray([1, 1, 2, 3, 4, 5], dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype="int64"),
            "person_household_id": person_tax_unit_ids,
            "person_tax_unit_id": person_tax_unit_ids,
            "person_spm_unit_id": person_tax_unit_ids,
            "person_family_id": person_tax_unit_ids,
            "person_marital_unit_id": person_tax_unit_ids,
            "long_term_capital_gains_before_response": np.asarray(
                [6_000.0, 4_000.0, 0.0, -4_000.0, 50_000.0, 8_000.0]
            ),
            "non_sch_d_capital_gains": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 3_000.0]),
        }
    )
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": tax_unit_ids})
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(5, 100.0), WeightKind.DESIGN)},
        pd.Series(["puf_tax_detail"] * len(person), name="stratum"),
    )


def _with_person_column(frame: Frame, name: str) -> Frame:
    """A new frame equal to ``frame`` plus one person column."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][name] = np.arange(len(tables["person"]), dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _with_person_ids_offset(frame: Frame) -> Frame:
    """A new frame whose person IDs differ from ``frame``'s."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["person_id"] = tables["person"]["person_id"] + 1_000
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _rewrite_input_person_ids_in_place(frame: Frame) -> Frame:
    """Change the input's person IDs in its own table, then build the result.

    The result carries the changed IDs, and its identity equals the changed
    input's, so a check that hashes the input again after the step finds
    nothing; only the identity recorded when the input was loaded shows the
    change.
    """

    person = frame.table("person")
    person["person_id"] = person["person_id"] + 1_000
    return _with_person_column(frame, "after_in_place_rewrite")


def _pipeline_and_steps(builder, stage: str):
    if stage == builder.HOUSEHOLD_LOCATION_STAGE_NAME:
        return builder.BLOCK_V1_OUTER_STAGE_PIPELINE, builder.BLOCK_V1_PIPELINE_STEPS
    return builder.OUTER_STAGE_PIPELINE, builder.PIPELINE_STEPS


def _prime_predecessors(builder, checkpoint_dir: Path, stage: str) -> Path:
    """Complete every step before ``stage``; return ``stage``'s checkpoint path."""

    pipeline, steps = _pipeline_and_steps(builder, stage)
    runtime = builder.StageRuntime(checkpoint_dir, pipeline, run_config={})
    base = _toy_us_frame()
    cloned = builder.clone_us_frame_for_puf_support(base)
    clone_index = steps.index("clone_feature_extraction")
    for index, name in enumerate(steps):
        if name == stage:
            return checkpoint_dir / f"{index:03d}_{stage}.frame.h5"
        if name == "primary_qrf_chain":
            runtime.complete_without_frame(name)
        else:
            runtime.complete(name, cloned if index >= clone_index else base)
    raise AssertionError(f"{stage!r} is not a step of the pipeline")


def _stage_args(builder, checkpoint_dir: Path, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        stage=stage,
        checkpoint_dir=checkpoint_dir,
        seed=0,
        target_year=2024,
        out=checkpoint_dir / "out",
        location_rule=(
            "block_v1" if stage == builder.HOUSEHOLD_LOCATION_STAGE_NAME else "legacy"
        ),
    )


def _patch_step(
    monkeypatch: pytest.MonkeyPatch,
    builder,
    stage: str,
    transform: Callable[[Frame], Frame] | None,
) -> None:
    """Make ``stage``'s step apply ``transform`` to its input frame.

    ``transform=None`` keeps the real step function where it runs without
    real inputs: the capital gain distribution split, and the post-QRF step
    dispatch with its input function and signal checks replaced.
    """

    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(builder, "profile_stage", lambda *_a, **_k: nullcontext())
    if stage == "pre_clone_enrichment":
        monkeypatch.setattr(
            builder,
            "_pre_clone_enrichment_stage",
            lambda _args, frame, _source_metadata: (transform(frame), {}),
        )
    elif stage == "clone_feature_extraction":
        monkeypatch.setattr(
            builder,
            "_clone_feature_extraction_stage",
            lambda _args, frame: (transform(frame), {}),
        )
    elif stage == "qrf_finalization":
        monkeypatch.setattr(
            builder,
            "_qrf_finalization_stage",
            lambda _args, frame: (transform(frame), {}),
        )
    elif stage == builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME:
        monkeypatch.setattr(
            builder,
            "_capital_gains_tail_transfer_stage",
            lambda _args, frame: (transform(frame), {}),
        )
    elif stage == builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME:
        if transform is not None:
            monkeypatch.setattr(
                builder,
                "_capital_gain_distributions_stage",
                lambda _args, frame: (transform(frame), {}),
            )
    elif stage == "qbi_reconciliation":
        monkeypatch.setattr(
            builder,
            "with_us_qbi_input_reconciliation",
            transform or (lambda frame: _with_person_column(frame, "qbi")),
        )
    elif stage == "wic_post_clone":
        step = transform or (lambda frame: _with_person_column(frame, "wic"))
        monkeypatch.setattr(
            builder, "with_us_wic_claim_input", lambda frame, **_kwargs: step(frame)
        )
        monkeypatch.setattr(
            builder, "us_wic_claim_signal_gate", lambda _frame: _PASSING_GATE
        )
        monkeypatch.setattr(
            builder, "us_medicare_take_up_signal_gate", lambda _frame: _PASSING_GATE
        )
    elif stage == builder.HOUSEHOLD_LOCATION_STAGE_NAME:
        monkeypatch.setattr(
            builder,
            "_block_v1_household_location",
            lambda _args, frame: (transform(frame), {"location_rule": "block_v1"}),
        )
    else:
        raise AssertionError(f"no step patch for {stage!r}")


def _completed_steps(builder, checkpoint_dir: Path, stage: str) -> tuple[str, ...]:
    pipeline, _steps = _pipeline_and_steps(builder, stage)
    return builder.StageRuntime(checkpoint_dir, pipeline).context.completed


def _stored_objects(frame: Frame) -> list[object]:
    """The tables and strata ``frame`` stores (it holds each of them).

    ``Frame`` defines ``__slots__`` without ``__weakref__``, so a weak
    reference cannot point at the frame itself. Once every object it stores
    is gone, the frame is gone too.
    """

    return [
        *(frame.table(entity) for entity in frame.entities),
        *(frame.link(name) for name in frame.links),
        frame.strata,
    ]


def _column_buffers(frame: Frame) -> list[np.ndarray]:
    """The NumPy arrays that own the memory of ``frame``'s NumPy-typed columns.

    A new DataFrame or Series over the same columns (a shallow copy, a column
    selection) would keep these arrays alive after the frame's own tables
    are gone. Text columns are left out: their buffers are immutable, and a
    deep copy of the table shares them with the step's result.
    """

    buffers: list[np.ndarray] = []
    tables = (
        *(frame.table(entity) for entity in frame.entities),
        *(frame.link(name) for name in frame.links),
    )
    for table in tables:
        for column in table.columns:
            series = table[column]
            if not isinstance(series.dtype, np.dtype):
                continue
            owner = series.to_numpy()
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            buffers.append(owner)
    return buffers


@pytest.mark.parametrize("stage", _FRAME_WRITING_STEPS)
def test_each_step_frees_its_input_frame_before_its_checkpoint_write(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    """At the checkpoint write, nothing holds the frame the step was given.

    The steps run their real ``_run_outer_stage`` branch with the shadow
    check run on (the second evaluation of the ``us/gates.json`` validation
    checks, which the package conftest turns off for every test); the capital
    gain distribution split and the post-QRF step dispatch also run their
    real step functions. Garbage collection is off
    while the step runs, so only reference counting can free the input.
    Column arrays the result itself holds and weights the step passes
    through to its result stay alive with the result, as they should.
    """

    checkpoint_path = _prime_predecessors(builder, tmp_path, stage)
    if stage == "clone_feature_extraction":
        transform = builder.clone_us_frame_for_puf_support
    elif stage in (
        builder.CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
        "qbi_reconciliation",
        "wic_post_clone",
    ):
        transform = None
    else:
        transform = lambda frame: _with_person_column(frame, stage)  # noqa: E731
    _patch_step(monkeypatch, builder, stage, transform)
    arm_us_tool_gate_shadows(monkeypatch)

    input_objects: list[weakref.ref] = []
    input_buffers: list[weakref.ref] = []
    load_predecessor = builder.StageRuntime.load_predecessor

    def recording_load_predecessor(runtime, stage_name, **kwargs):
        loaded = load_predecessor(runtime, stage_name, **kwargs)
        input_objects.extend(
            weakref.ref(item) for item in _stored_objects(loaded.frame)
        )
        input_buffers.extend(
            weakref.ref(buffer) for buffer in _column_buffers(loaded.frame)
        )
        return loaded

    written: list[dict[str, object]] = []
    write_frame_checkpoint = outer_stage_runtime.write_frame_checkpoint

    def recording_write(path, frame, **kwargs):
        result_buffers = {id(buffer) for buffer in _column_buffers(frame)}
        live_buffers = [reference() for reference in input_buffers]
        written.append(
            {
                "path": Path(path),
                "live_objects": sum(
                    reference() is not None for reference in input_objects
                ),
                "live_buffers_not_in_result": sum(
                    buffer is not None and id(buffer) not in result_buffers
                    for buffer in live_buffers
                ),
                "shadow_active": current_gate_shadow() is not None,
            }
        )
        del live_buffers
        return write_frame_checkpoint(path, frame, **kwargs)

    monkeypatch.setattr(
        builder.StageRuntime, "load_predecessor", recording_load_predecessor
    )
    monkeypatch.setattr(outer_stage_runtime, "write_frame_checkpoint", recording_write)

    args = _stage_args(builder, tmp_path, stage)
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        builder._run_outer_stage(args)
    finally:
        if collection_was_enabled:
            gc.enable()

    assert input_objects and input_buffers, "the step loaded no input frame"
    assert [record["path"] for record in written] == [checkpoint_path]
    (record,) = written
    assert record["live_objects"] == 0, (
        f"{stage}: {record['live_objects']} of the input frame's "
        f"{len(input_objects)} stored objects were alive at the checkpoint write"
    )
    assert record["live_buffers_not_in_result"] == 0, (
        f"{stage}: {record['live_buffers_not_in_result']} of the input frame's "
        f"{len(input_buffers)} column arrays were alive at the checkpoint write "
        "without being part of the result"
    )
    assert record["shadow_active"]
    assert builder._stage_gate_shadow_observations(args, stage).is_file()
    assert _completed_steps(builder, tmp_path, stage)[-1] == stage


@pytest.mark.parametrize("stage", _IDENTITY_KEEPING_STEPS)
def test_a_step_that_rewrites_its_input_ids_in_place_is_refused_before_it_is_recorded(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    """The result is compared with the identity recorded when the input loaded.

    Hashing the input again after the step would hash the rewritten IDs and
    find them equal to the result's.
    """

    checkpoint_path = _prime_predecessors(builder, tmp_path, stage)
    _patch_step(monkeypatch, builder, stage, _rewrite_input_person_ids_in_place)

    with pytest.raises(
        AssertionError,
        match=rf"stage {stage!r} changed Frame identity or row order",
    ):
        builder._run_outer_stage(_stage_args(builder, tmp_path, stage))

    assert stage not in _completed_steps(builder, tmp_path, stage)
    assert not checkpoint_path.exists()


@pytest.mark.parametrize("stage", _IDENTITY_KEEPING_STEPS)
def test_a_step_that_changes_identity_is_still_refused_before_it_is_recorded(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    checkpoint_path = _prime_predecessors(builder, tmp_path, stage)
    _patch_step(monkeypatch, builder, stage, _with_person_ids_offset)

    with pytest.raises(
        AssertionError,
        match=rf"stage {stage!r} changed Frame identity or row order",
    ):
        builder._run_outer_stage(_stage_args(builder, tmp_path, stage))

    assert stage not in _completed_steps(builder, tmp_path, stage)
    assert not checkpoint_path.exists()


def test_the_capital_gains_tail_step_still_has_no_identity_comparison(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The tail transfer had no identity check before this change and gains none.

    This records today's behaviour; adding a check to this step is a separate
    decision.
    """

    stage = builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME
    checkpoint_path = _prime_predecessors(builder, tmp_path, stage)
    _patch_step(monkeypatch, builder, stage, _with_person_ids_offset)

    builder._run_outer_stage(_stage_args(builder, tmp_path, stage))

    assert _completed_steps(builder, tmp_path, stage)[-1] == stage
    assert checkpoint_path.exists()


def test_clone_feature_extraction_still_checks_the_clone_against_its_input(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The clone check reads the input frame, so it runs before the drop."""

    stage = "clone_feature_extraction"
    checkpoint_path = _prime_predecessors(builder, tmp_path, stage)
    _patch_step(
        monkeypatch,
        builder,
        stage,
        lambda frame: _with_person_column(frame, "not_cloned"),
    )

    with pytest.raises(AssertionError, match="clone expansion must double"):
        builder._run_outer_stage(_stage_args(builder, tmp_path, stage))

    assert stage not in _completed_steps(builder, tmp_path, stage)
    assert not checkpoint_path.exists()
