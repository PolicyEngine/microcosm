"""Stage plans: declared inputs/outputs enforced, failures abort, no fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build import DonorSpec, Stage, StagePlan
from microcosm.frame import Frame


def _add_column(name: str, values) -> callable:
    def transform(frame: Frame) -> Frame:
        person = frame.person.copy()
        person[name] = values
        tables = {"person": person}
        for group in frame.schema.group_entities:
            tables[group] = frame.table(group)
        return Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
        )

    return transform


SCF = DonorSpec(
    survey="Fed SCF 2022",
    source="https://www.federalreserve.gov/econres/scfindex.htm",
)


class TestPlanValidation:
    def test_duplicate_stage_names_refused(self) -> None:
        stage = Stage(name="a", transform=lambda frame: frame)
        with pytest.raises(ValueError, match="Duplicate stage name"):
            StagePlan([stage, Stage(name="a", transform=lambda frame: frame)])

    def test_two_producers_of_one_column_refused(self) -> None:
        with pytest.raises(ValueError, match="one canonical producer"):
            StagePlan(
                [
                    Stage(
                        name="a",
                        transform=lambda frame: frame,
                        produces=("net_worth",),
                    ),
                    Stage(
                        name="b",
                        transform=lambda frame: frame,
                        produces=("net_worth",),
                    ),
                ]
            )

    def test_empty_donor_fields_refused(self) -> None:
        with pytest.raises(ValueError, match="source citation is required"):
            DonorSpec(survey="SCF", source="")


class TestExecution:
    def test_run_produces_columns_and_records(self, small_frame) -> None:
        plan = StagePlan(
            [
                Stage(
                    name="wealth",
                    transform=_add_column(
                        "net_worth", np.asarray([1e5, 0.0, 2e5, 0.0])
                    ),
                    produces=("net_worth",),
                    donor=SCF,
                ),
                Stage(
                    name="derive",
                    transform=_add_column(
                        "has_wealth", np.asarray([True, False, True, False])
                    ),
                    produces=("has_wealth",),
                    consumes=("net_worth",),
                ),
            ]
        )
        lines: list[str] = []
        result, records = plan.run(small_frame, log=lines.append)
        assert "net_worth" in result.person.columns
        assert records[0].donor_survey == "Fed SCF 2022"
        assert records[0].nonzero_share["net_worth"] == pytest.approx(0.5)
        assert records[1].donor_survey is None
        assert records[1].nonzero_share["has_wealth"] == pytest.approx(0.5)
        assert len(lines) == 2 and "Fed SCF 2022" in lines[0]
        assert plan.donors() == (("wealth", SCF),)

    def test_missing_consumed_column_aborts_before_running(self, small_frame) -> None:
        ran = []

        def transform(frame: Frame) -> Frame:
            ran.append(True)
            return frame

        plan = StagePlan([Stage(name="x", transform=transform, consumes=("absent",))])
        with pytest.raises(ValueError, match="consumes 'absent'"):
            plan.run(small_frame)
        assert not ran

    def test_undelivered_produce_fails_after_running(self, small_frame) -> None:
        plan = StagePlan(
            [
                Stage(
                    name="x",
                    transform=lambda frame: frame,
                    produces=("promised",),
                )
            ]
        )
        with pytest.raises(ValueError, match="'promised' but the column is absent"):
            plan.run(small_frame)

    def test_stage_exception_aborts_the_build(self, small_frame) -> None:
        """No fallback path exists: a failing donor stage kills the run."""

        def org_loader_fails(frame: Frame) -> Frame:
            raise FileNotFoundError("cps_org_2024.parquet not found")

        downstream_ran = []

        def downstream(frame: Frame) -> Frame:
            downstream_ran.append(True)
            return frame

        plan = StagePlan(
            [
                Stage(name="org_wages", transform=org_loader_fails),
                Stage(name="after", transform=downstream),
            ]
        )
        with pytest.raises(FileNotFoundError):
            plan.run(small_frame)
        assert not downstream_ran

    def test_non_frame_return_is_refused(self, small_frame) -> None:
        plan = StagePlan([Stage(name="x", transform=lambda frame: frame.person)])
        with pytest.raises(TypeError, match="must return a Frame"):
            plan.run(small_frame)
