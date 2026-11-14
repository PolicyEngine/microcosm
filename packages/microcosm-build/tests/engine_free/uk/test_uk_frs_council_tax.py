from __future__ import annotations

import json
import math
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.frs_council_tax import (
    FRS_COUNCIL_TAX_BILL_MISSING,
    FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
    FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES,
    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
    NORTHERN_IRELAND_GVTREGNO,
    SCOTLAND_GVTREGNO,
    SCOTTISH_CHARGES_IN_RECIPIENT_BILL,
    assert_frs_council_tax_stage_parameters,
    council_tax_provenance_evidence,
    derive_council_tax,
    derive_council_tax_provenance,
    frs_council_tax_provenance_operation_parameters,
)
from microcosm.build.uk_runtime.frs_spine import REGION_MAP, WEEKS_IN_YEAR
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import (
    SOURCE_HOUSEHOLD_COLUMN,
    rounded_count,
)
from test_support.paths import paths_for

OBSERVED = FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS
MISSING = FRS_COUNCIL_TAX_BILL_MISSING
UNKNOWN = FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION
NOT_LEVIED = FRS_COUNCIL_TAX_BILL_NOT_LEVIED

UK_PACKAGE = (
    paths_for("microcosm-build").repository
    / "packages/microcosm-build/src/microcosm/build/uk"
)


def _raw(**columns) -> pd.DataFrame:
    rows = len(columns["ctannual"])
    defaults = {
        "household_id": list(range(1, rows + 1)),
        # CTPAID 3, the household does not have to pay: a zero bill stays zero.
        "ctpaid": [3] * rows,
        "ctreb": [2] * rows,
        "ctrebamt": [0.0] * rows,
        "gvtregno": [1] * rows,
        "ctband": [4] * rows,
        "adulth": [2] * rows,
        "cwatamt1": [0.0] * rows,
        "csewamt1": [0.0] * rows,
        "cwatamtd": [0.0] * rows,
    }
    return pd.DataFrame({**defaults, **columns})


def _derive(raw: pd.DataFrame) -> pd.Series:
    return derive_council_tax(pd.DataFrame({"household_id": raw["household_id"]}), raw)


def _provenance(raw: pd.DataFrame) -> pd.Series:
    return derive_council_tax_provenance(
        pd.DataFrame({"household_id": raw["household_id"]}), raw
    )


def test_council_tax_imputes_raw_missing_from_raw_cells() -> None:
    raw = _raw(
        gvtregno=[12, 12, 1, 1, 2, 2],
        ctband=[1, 1, np.nan, np.nan, 2, 2],
        adulth=[1, 1, 1, 1, 2, 2],
        ctannual=[1000.0, -1.0, 600.0, np.nan, -1.0, 0.0],
        # FRS 2024-25 shape: the gross Scottish cells CWATAMT1/CSEWAMT1.
        cwatamtd=[3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        cwatamt1=[4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        csewamt1=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = _derive(raw)

    # A non-recipient's bill carries the gross charges in full.
    scottish_tax = 1000.0 - (4.0 + 5.0) * WEEKS_IN_YEAR
    assert np.isclose(result.loc[1], scottish_tax)
    assert np.isclose(result.loc[2], scottish_tax)
    assert result.loc[4] == 600.0
    # No positive non-recipient bill in the cell: nothing to impute from.
    assert result.loc[5] == 0.0
    # A non-recipient's zero bill (an exempt dwelling) stays zero.
    assert result.loc[6] == 0.0


def test_paying_household_with_a_zero_bill_takes_the_cell_mean() -> None:
    raw = _raw(
        gvtregno=[1, 1, 1, 1, 1, 1, 1, 13, 2],
        ctannual=[1200.0, 1400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctpaid=[1, 1, 1, 1, 2, 3, np.nan, np.nan, 1],
        ctreb=[2, 2, 2, 1, 2, 2, 2, 2, 2],
        ctrebamt=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = _derive(raw)

    # Someone pays and no reduction is reported, so the household owes a bill.
    assert result.loc[3] == pytest.approx(1300.0)
    # A reported reduction is added back, as for any recipient.
    assert result.loc[4] == pytest.approx(20.0 * WEEKS_IN_YEAR)
    # Council-paid, exempt, blank and unasked (Northern Ireland) zeros stay zero.
    assert result.loc[5] == 0.0
    assert result.loc[6] == 0.0
    assert result.loc[7] == 0.0
    assert result.loc[8] == 0.0
    # No positive non-recipient bill in the cell: nothing to impute from.
    assert result.loc[9] == 0.0


def test_reported_reduction_is_added_back_to_the_bill() -> None:
    raw = _raw(
        ctannual=[0.0, 500.0, 1300.0],
        ctreb=[1, 1, 2],
        ctrebamt=[20.0, 10.0, 0.0],
    )

    result = _derive(raw)

    # A full reduction leaves CTANNUAL at zero; the liability is the reduction.
    assert result.loc[1] == pytest.approx(20.0 * WEEKS_IN_YEAR)
    assert result.loc[2] == pytest.approx(500.0 + 10.0 * WEEKS_IN_YEAR)
    assert result.loc[3] == pytest.approx(1300.0)


def test_unknown_reduction_takes_the_larger_of_bill_and_cell_mean() -> None:
    raw = _raw(
        ctannual=[1200.0, 1400.0, 300.0, 1500.0],
        ctreb=[2, 2, 1, 1],
        ctrebamt=[0.0, 0.0, 0.0, np.nan],
    )

    result = _derive(raw)

    assert result.loc[3] == pytest.approx(1300.0)
    assert result.loc[4] == pytest.approx(1500.0)


def test_cell_means_pool_non_recipients_only() -> None:
    raw = _raw(
        ctannual=[1200.0, 100.0, np.nan],
        ctreb=[2, 1, 2],
        ctrebamt=[0.0, 15.0, 0.0],
    )

    result = _derive(raw)

    # The recipient's net bill (100) would pull the cell mean down.
    assert result.loc[3] == pytest.approx(1200.0)
    assert result.loc[2] == pytest.approx(100.0 + 15.0 * WEEKS_IN_YEAR)


def test_scottish_recipients_net_the_reduced_share_of_the_gross_charges() -> None:
    raw = _raw(
        gvtregno=[12, 12],
        ctannual=[800.0, 1600.0],
        ctreb=[1, 2],
        ctrebamt=[5.0, 0.0],
        cwatamt1=[4.0, 4.0],
        csewamt1=[5.0, 5.0],
        cwatamtd=[3.0, 3.0],
    )

    result = _derive(raw)

    charges = (4.0 + 5.0) * WEEKS_IN_YEAR
    assert result.loc[1] == pytest.approx(
        800.0 - SCOTTISH_CHARGES_IN_RECIPIENT_BILL * charges + 5.0 * WEEKS_IN_YEAR
    )
    assert result.loc[2] == pytest.approx(1600.0 - charges)


def test_scottish_household_without_a_gross_cell_nets_its_recorded_charge() -> None:
    raw = _raw(
        gvtregno=[12],
        ctannual=[900.0],
        cwatamtd=[3.0],
    )

    assert _derive(raw).loc[1] == pytest.approx(900.0 - 3.0 * WEEKS_IN_YEAR)


def test_bills_outside_scotland_keep_ctannual_or_add_the_reduction() -> None:
    rng = np.random.default_rng(1095)
    rows = 400
    raw = _raw(
        gvtregno=rng.integers(1, 12, rows),
        ctband=rng.integers(1, 9, rows),
        adulth=rng.integers(1, 3, rows),
        ctannual=rng.choice([0.0, 250.0, 900.0, 1800.0, 2600.0], rows),
        ctreb=rng.choice([1, 2], rows),
        ctrebamt=rng.choice([0.0, 4.0, 18.0, 30.0], rows),
    )

    result = _derive(raw).to_numpy()
    ctannual = raw["ctannual"].to_numpy()
    recipient = raw["ctreb"].to_numpy() == 1
    reduction = raw["ctrebamt"].to_numpy() * WEEKS_IN_YEAR

    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result[~recipient], ctannual[~recipient])
    known = recipient & (reduction > 0)
    np.testing.assert_allclose(result[known], ctannual[known] + reduction[known])
    assert (result[recipient] >= ctannual[recipient] - 1e-9).all()


@pytest.mark.parametrize("column", ["ctreb", "ctpaid"])
def test_missing_raw_cells_refuse(column) -> None:
    raw = _raw(ctannual=[100.0]).drop(columns=column)

    with pytest.raises(KeyError, match=column):
        _derive(raw)


def _committed_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads((UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8"))
    )
    return next(stage for stage in manifest.stages if stage.stage == "frs_council_tax")


def test_committed_manifest_declares_the_implemented_rule() -> None:
    assert_frs_council_tax_stage_parameters(_committed_stage())


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("donor_filter", "raw ctannual > 0"),
        ("missing", "raw ctannual < 0 or NaN"),
        ("value", "Scottish-water-netted CTANNUAL"),
        ("cells", ["gvtregno", "ctband"]),
    ],
)
def test_manifest_drift_from_the_implemented_rule_refuses(parameter, value) -> None:
    stage = _committed_stage()
    operations = [
        {"kind": operation.kind, **dict(operation.parameters)}
        for operation in stage.operations
    ]
    operations[1][parameter] = value
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match=parameter):
        assert_frs_council_tax_stage_parameters(drifted)


class _Case(NamedTuple):
    """One invented household, with its bill and provenance worked by hand."""

    gvtregno: int
    ctannual: float
    ctpaid: float
    ctreb: float
    ctrebamt: float
    bill: float
    provenance: str
    adulth: int = 2
    cwatamt1: float = 0.0
    csewamt1: float = 0.0


_CASES = (
    # England's cell (GVTREGNO 1, band 4, two adults). Its donors, households
    # without a reported reduction and with a positive bill, are the first
    # two: the cell mean is (1200 + 1400) / 2 = 1300.
    _Case(1, 1200.0, 1, 2, 0.0, 1200.0, OBSERVED),
    _Case(1, 1400.0, 1, 2, 0.0, 1400.0, OBSERVED),
    # A blank or negative bill is missing, though a reduction is reported,
    # with an amount or without.
    _Case(1, np.nan, 1, 1, 10.0, 1300.0, MISSING),
    _Case(1, -1.0, 1, 1, 0.0, 1300.0, MISSING),
    # Someone pays, no reduction is reported, and the bill is zero: missing.
    _Case(1, 0.0, 1, 2, 0.0, 1300.0, MISSING),
    # The council pays, the household need not pay, or CTPAID is blank: the
    # zero bill is kept.
    _Case(1, 0.0, 2, 2, 0.0, 0.0, OBSERVED),
    _Case(1, 0.0, 3, 2, 0.0, 0.0, OBSERVED),
    _Case(1, 0.0, np.nan, 2, 0.0, 0.0, OBSERVED),
    # A reported reduction is added back: 500 + 10 x 52 = 1020.
    _Case(1, 500.0, 1, 1, 10.0, 1020.0, OBSERVED),
    # A reduction without a positive amount: the larger of the bill and 1300,
    # a zero bill where someone pays included.
    _Case(1, 300.0, 1, 1, 0.0, 1300.0, UNKNOWN),
    _Case(1, 1500.0, 1, 1, np.nan, 1500.0, UNKNOWN),
    _Case(1, 0.0, 1, 1, 0.0, 1300.0, UNKNOWN),
    # Northern Ireland is not levied, whatever branch its bill takes: its cell
    # has no donor, so a missing bill is zero, and a reported reduction of 5 a
    # week is added back, 5 x 52 = 260.
    _Case(13, 0.0, np.nan, 2, 0.0, 0.0, NOT_LEVIED),
    _Case(13, np.nan, np.nan, 2, 0.0, 0.0, NOT_LEVIED),
    _Case(13, 0.0, np.nan, 1, 5.0, 260.0, NOT_LEVIED),
    # Scotland's cell nets (4 + 5) x 52 = 468 of water and sewerage from
    # 1000, so its mean is 532, which its paying zero takes.
    _Case(12, 1000.0, 1, 2, 0.0, 532.0, OBSERVED, cwatamt1=4.0, csewamt1=5.0),
    _Case(12, 0.0, 1, 2, 0.0, 532.0, MISSING, cwatamt1=4.0, csewamt1=5.0),
    # A single adult in Scotland pays and reports no reduction; its CTANNUAL of
    # 400 is below the 468 of water and sewerage, so the netted bill is zero and
    # kept, since CTANNUAL itself is not zero. It is its cell's only donor.
    _Case(12, 400.0, 1, 2, 0.0, 0.0, OBSERVED, adulth=1, cwatamt1=4.0, csewamt1=5.0),
    # Single adults are a cell of their own.
    _Case(1, 900.0, 1, 2, 0.0, 900.0, OBSERVED, adulth=1),
    _Case(1, np.nan, 1, 2, 0.0, 900.0, MISSING, adulth=1),
)


def _case_rows(cases: tuple[_Case, ...] = _CASES) -> pd.DataFrame:
    return _raw(
        **{
            column: [getattr(case, column) for case in cases]
            for column in (
                "gvtregno",
                "ctannual",
                "ctpaid",
                "ctreb",
                "ctrebamt",
                "adulth",
                "cwatamt1",
                "csewamt1",
            )
        }
    )


def test_each_bill_and_its_provenance_match_the_hand_worked_cases() -> None:
    raw = _case_rows()

    assert _derive(raw).to_list() == [case.bill for case in _CASES]
    assert _provenance(raw).to_list() == [case.provenance for case in _CASES]


def _cell_means(raw: pd.DataFrame) -> list[float]:
    """Each row's cell mean outside Scotland, worked one row at a time: the mean
    CTANNUAL of its (GVTREGNO, band, single adult) cell's rows without a
    reported reduction and with a positive bill, zero where there is none."""

    cells = list(zip(raw["gvtregno"], raw["ctband"], raw["adulth"] == 1, strict=True))
    donors = [
        (cell, bill)
        for cell, bill, ctreb in zip(cells, raw["ctannual"], raw["ctreb"], strict=True)
        if ctreb != 1 and bill > 0
    ]
    means = []
    for cell in cells:
        bills = [bill for donor, bill in donors if donor == cell]
        means.append(sum(bills) / len(bills) if bills else 0.0)
    return means


def test_outside_northern_ireland_the_provenance_is_the_branch_the_bill_took() -> None:
    rng = np.random.default_rng(931)
    rows = 600
    raw = _raw(
        gvtregno=rng.choice([1, 2, 11, NORTHERN_IRELAND_GVTREGNO], rows),
        ctband=rng.integers(1, 4, rows),
        adulth=rng.integers(1, 3, rows),
        ctannual=rng.choice([np.nan, -1.0, 0.0, 250.0, 900.0, 1800.0], rows),
        ctpaid=rng.choice([1.0, 2.0, 3.0, np.nan], rows),
        ctreb=rng.choice([1.0, 2.0, np.nan], rows),
        ctrebamt=rng.choice([np.nan, 0.0, 4.0, 18.0], rows),
    )

    bills = _derive(raw).to_list()
    flags = _provenance(raw).to_list()

    assert set(flags) == set(FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES)
    for row, bill, flag, mean in zip(
        raw.itertuples(), bills, flags, _cell_means(raw), strict=True
    ):
        if row.gvtregno == NORTHERN_IRELAND_GVTREGNO:
            assert flag == NOT_LEVIED
            continue
        recipient = row.ctreb == 1
        if (
            math.isnan(row.ctannual)
            or row.ctannual < 0
            or (row.ctannual == 0 and row.ctpaid == 1 and not recipient)
        ):
            assert (flag, bill) == (MISSING, pytest.approx(mean, rel=1e-12))
        elif recipient and row.ctrebamt > 0:
            reduction = row.ctrebamt * WEEKS_IN_YEAR
            assert (flag, bill) == (OBSERVED, row.ctannual + reduction)
        elif recipient:
            expected = pytest.approx(max(row.ctannual, mean), rel=1e-12)
            assert (flag, bill) == (UNKNOWN, expected)
        else:
            assert (flag, bill) == (OBSERVED, row.ctannual)


def test_the_provenance_is_a_python_string_column_on_the_bill_index() -> None:
    raw = _case_rows()
    # The households in reverse, and one the raw tab lacks, whose bill reads
    # zero and whose provenance reads missing.
    household = pd.DataFrame(
        {"household_id": [*raw["household_id"].to_list()[::-1], len(_CASES) + 1]}
    )

    provenance = derive_council_tax_provenance(household, raw)
    bill = derive_council_tax(household, raw)

    assert provenance.dtype == pd.StringDtype(storage="python")
    assert provenance.index.equals(bill.index)
    assert provenance.index.to_list() == household["household_id"].to_list()
    assert provenance.to_list() == [
        *(case.provenance for case in reversed(_CASES)),
        MISSING,
    ]
    assert bill.iloc[-1] == 0.0


def test_the_provenance_declaration_names_the_values_in_their_precedence() -> None:
    declared = frs_council_tax_provenance_operation_parameters()

    assert sorted(declared) == ["formula", "kind", "output"]
    assert declared["kind"] == "derive"
    assert declared["output"] == FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN
    assert FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN == "council_tax_bill_provenance"
    formula = declared["formula"]
    firsts = [formula.find(value) for value in (NOT_LEVIED, MISSING, OBSERVED, UNKNOWN)]
    assert -1 not in firsts and firsts == sorted(firsts)
    # The default, after every condition.
    assert formula.rfind(OBSERVED) > formula.find(UNKNOWN)


def test_the_region_codes_are_the_spine_region_map_codes() -> None:
    assert REGION_MAP[SCOTLAND_GVTREGNO] == "SCOTLAND"
    assert REGION_MAP[NORTHERN_IRELAND_GVTREGNO] == "NORTHERN_IRELAND"


def _frame(household_ids: np.ndarray):
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": household_ids,
                "person_benunit_id": household_ids,
                "person_household_id": household_ids,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": household_ids}),
        household=pd.DataFrame(
            {"household_id": household_ids, "household_weight": 1.0}
        ),
        time_period="2024",
    )


def _is_a_rounded_count(count: object) -> bool:
    """Whether ``count`` is a value ``rounded_count`` gives."""

    if count in ("<3", "3-9"):
        return True
    return isinstance(count, int) and rounded_count(count) == count


def test_the_evidence_receipts_each_value_in_rounded_distinct_sources() -> None:
    # Invented counts: 1234 rounds to 1200, 7 reads "3-9" and 2 "<3".
    values = [OBSERVED] * 1234 + [UNKNOWN] * 7 + [MISSING] * 2
    frame = _frame(np.arange(1, len(values) + 1))

    evidence = council_tax_provenance_evidence(
        frame, pd.Series(values, dtype=pd.StringDtype(storage="python"))
    )

    assert evidence == {
        "stage": "frs_council_tax",
        "bill_provenance_sources": {
            OBSERVED: 1200,
            MISSING: "<3",
            UNKNOWN: "3-9",
            NOT_LEVIED: 0,
        },
    }
    assert list(evidence["bill_provenance_sources"]) == list(
        FRS_COUNCIL_TAX_BILL_PROVENANCE_VALUES
    )
    assert all(map(_is_a_rounded_count, evidence["bill_provenance_sources"].values()))


def test_the_evidence_counts_a_source_and_its_copies_once() -> None:
    # Twelve rows from four sources, three rows each. Six rows from two sources
    # read "<3", where six rows would read "3-9"; all twelve, four sources, read
    # "3-9", where twelve rows would round to 10.
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 13),
            SOURCE_HOUSEHOLD_COLUMN: np.repeat([1, 2, 3, 4], 3),
        }
    )

    evidence = council_tax_provenance_evidence(
        {"household": household}, [MISSING] * 6 + [OBSERVED] * 6
    )

    assert evidence["bill_provenance_sources"] == {
        OBSERVED: "<3",
        MISSING: "<3",
        UNKNOWN: 0,
        NOT_LEVIED: 0,
    }
    evidence = council_tax_provenance_evidence({"household": household}, [MISSING] * 12)
    assert evidence["bill_provenance_sources"][MISSING] == "3-9"


@pytest.mark.parametrize(
    "values",
    [
        [OBSERVED],
        [OBSERVED, "drawn"],
        [OBSERVED, None],
        pd.array([OBSERVED, pd.NA], dtype=pd.StringDtype(storage="python")),
        [[OBSERVED, MISSING]],
    ],
)
def test_the_evidence_refuses_what_it_cannot_receipt(values) -> None:
    household = pd.DataFrame({"household_id": [1, 2]})

    with pytest.raises(
        ValueError, match=FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN
    ) as refused:
        council_tax_provenance_evidence({"household": household}, values)
    assert not any(character.isdigit() for character in str(refused.value))
