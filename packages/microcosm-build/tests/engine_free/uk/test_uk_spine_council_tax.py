"""The clone-first spine's council-tax localisation, G6 (microcosm#931).

G6 (``council_tax_localise``) first keeps each household's paid bill from
before it rewrites anything, as ``council_tax_frs_bill``. Then it prices each
Great Britain household's council tax at the local authority the spine drew
for it. The FRS bill is multiplied by the authority's 2024-25 Band D over the
expected Band D of the household's own draw cell. Where no gross bill was
observed and the band was, the bill is the 2024-25 statute. The rebate and
the reported council tax reduction are scaled by the same ratio and capped
at the new bill. Northern Ireland's rows are left as they are. sources.yaml
declares G6 between private_rent and was_wealth; the transform here runs as a
stage built from the reviewed declaration spine_council_tax states, which the
packaged stage equals.

The frames are test_support's invented council-tax cases
(uk_council_tax_fixtures) on the toy conditioned supports, at K = 1 and
K = 3. The spine's own steps clone and key them, and its area draw places
them. A stand-in for the support channel copies them, and the copies are
re-placed as the SPI-copy area stage re-places them. Each toy authority has
an invented Band D. Each expected value here is worked again from the toy
supports, the statute's proportions restated and the invented Band D,
never from the module's own references.

They hold:

- each level is the area draw's, and every SPI copy and band donor is drawn
  at L3;
- each reference is the exact expected Band D of the row's cell, and the
  ratio averages one over every Great Britain cell of the toys;
- each row takes its path; ratio rows scale the bill, rebate and reduction
  by the ratio, bit for bit; statute rows price their band at their own
  authority by nation, with the single-adult discount, and scale the rebate
  and reduction by the ratio, in a cell of two authorities too;
- the cap holds every Great Britain reduction and rebate at the new bill;
- Northern Ireland is left bit for bit, a reduction above its zero bill
  included, which is counted;
- the kept bill is WAS's paid bill from the columns before the rewrite, and
  nets the rebate where it differs from the reduction;
- clone 0 at K = 3 is K = 1's, and the result does not depend on row order;
- the evidence counts true sources through ``receipt_counts``, each count
  its own where every FRS path holds a different number of sources: the
  support channel's in clone 0 even where later clones copy other
  households, and the capped households and Northern Ireland's reductions
  above the bill, and whether they lie within it, in clone 0 even where
  later clones differ; it is byte-equal at K = 1 and K = 3 but for the
  clone count;
- every refusal names columns only, the draw-cell guard among them;
- the transform runs only as its reviewed stage, and the packaged stage is
  that stage.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib.resources import files
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from microcosm.build.atomic_geography import validate_assignment_spec
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    _reject_executable_parameter_keys,
    _reject_incumbent_dependencies,
)
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    IDENTITY_COLUMN,
    UK_SPINE_LADDER_COLUMNS,
)
from microcosm.build.uk_runtime.council_tax_band_d import (
    SCOTLAND_BAND_D_RESOURCE,
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
    UKCouncilTaxBandD,
    check_scotland_band_d_declaration,
    load_uk_council_tax_band_d,
)
from microcosm.build.uk_runtime.frs_council_tax import (
    FRS_COUNCIL_TAX_BILL_MISSING,
    FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
    FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS,
    FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
    FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
)
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import extreme_count, rounded_count
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.spine_council_tax import (
    COUNCIL_TAX_FRS_BILL_COLUMN,
    COUNCIL_TAX_LOCALISE_METHOD,
    COUNCIL_TAX_LOCALISE_OPERATION_KIND,
    COUNCIL_TAX_LOCALISE_STAGE_NAME,
    UK_COUNCIL_TAX_LOCALISE_CHECKS,
    UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS,
    UK_COUNCIL_TAX_LOCALISE_FRS_PATHS,
    UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_INPUTS,
    UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON,
    UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS,
    UK_COUNCIL_TAX_LOCALISE_REWRITES,
    UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS,
    UKCouncilTaxLocaliseResult,
    UKCouncilTaxLocaliseStageTransform,
    UKSpineCouncilTax,
    council_tax_localise_operation_parameters,
    localise_uk_council_tax,
    localise_uk_spine_council_tax,
    uk_council_tax_band_ratio_text,
    uk_council_tax_references,
)
from microcosm.build.uk_runtime.spine_geography import (
    atomic_area_assign_operation_parameters,
    draw_uk_spine_areas,
)
from microcosm.build.uk_runtime.spine_private_rent import (
    private_rent_operation_parameters,
)
from microcosm.frame import Frame
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    SCOT,
    ToyHousehold,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
)
from test_support.microcosm_build.uk_council_tax_fixtures import (
    COUNCIL_TAX_CASES,
    council_tax_cell_sweep,
    council_tax_spine_frame,
    invented_band_d,
    toy_area_draw_definition,
)

_CLONE = ladder_clone_index_column("household")
_CHANNEL = support_channel_column("household")
_ORDINAL = support_clone_index_column("household")
_REDUCTION = "council_tax_benefit_reported"
_NORTHERN_IRELAND = "NORTHERN_IRELAND"
_STRING = pd.StringDtype("python")
_RATIO_PATHS = (
    "ratio_observed_gross",
    "ratio_missing_unbanded",
    "ratio_unknown_reduction_unbanded",
    "ratio_support_channel",
)
_FORMULA_PATHS = ("formula_missing", "formula_unknown_reduction")
#: The statute's band proportions to Band D, restated: England's and Wales's
#: ninths (Wales's band I at 21/9) and Scotland's three-hundred-and-sixtieths.
_NINTHS = dict(zip("ABCDEFGH", (6, 7, 8, 9, 11, 13, 15, 18), strict=True))
_STATUTE = MappingProxyType(
    {
        "ENGLAND": {band: Fraction(ninths, 9) for band, ninths in _NINTHS.items()},
        "WALES": {
            **{band: Fraction(ninths, 9) for band, ninths in _NINTHS.items()},
            "I": Fraction(21, 9),
        },
        "SCOTLAND": {
            band: Fraction(parts, 360)
            for band, parts in zip(
                "ABCDEFGH", (240, 280, 320, 360, 473, 585, 705, 882), strict=True
            )
        },
    }
)
_SINGLE_ADULT = Fraction(3, 4)


def _nation(region: str) -> str:
    return region if region in ("WALES", "SCOTLAND") else "ENGLAND"


# --- The stage and the frames -------------------------------------------------------


def _stage(**overrides: object) -> SourceStageSpec:
    """G6's stage built from the reviewed declaration spine_council_tax
    states, with ``overrides`` replacing fields."""

    stage = SourceStageSpec(
        stage=COUNCIL_TAX_LOCALISE_STAGE_NAME,
        survey="Family Resources Survey 2024-25",
        source="Microcosm structural operator (microcosm#931)",
        grain="household",
        artifacts=tuple(
            {
                "role": role,
                "kind": "public_aggregate_reference",
                "resource": resource,
                "format": "json",
                "runtime_sha256_required": True,
            }
            for role, resource in (
                ("council_tax_band_d", UK_BAND_D_ENGLAND_WALES_RESOURCE),
                ("scotland_band_d_facts", SCOTLAND_BAND_D_RESOURCE),
            )
        ),
        operations=tuple(
            SourceOperationSpec(kind=kind, parameters=dict(parameters))
            for kind, parameters in council_tax_localise_operation_parameters()
        ),
        outputs=UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS,
        nonnegative_outputs=UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS,
        rewrites=UK_COUNCIL_TAX_LOCALISE_REWRITES,
    )
    return replace(stage, **overrides)


@functools.cache
def _supports() -> dict:
    return toy_conditioned_supports()


@functools.cache
def _definition() -> dict:
    return toy_area_draw_definition()


@functools.cache
def _band_d() -> UKCouncilTaxBandD:
    return invented_band_d()


@functools.cache
def _frame(clones: int) -> Frame:
    return council_tax_spine_frame(clones=clones)


def _localise(frame: Frame, band_d: UKCouncilTaxBandD | None = None):
    return localise_uk_spine_council_tax(
        frame,
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d() if band_d is None else band_d,
    )


@functools.cache
def _run(clones: int) -> UKCouncilTaxLocaliseResult:
    return _localise(_frame(clones))


@functools.cache
def _local(clones: int) -> UKSpineCouncilTax:
    frame = _frame(clones)
    return localise_uk_council_tax(
        frame.table("household"),
        frame.table("person"),
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d(),
    )


def _with(
    frame: Frame,
    *,
    household: pd.DataFrame | None = None,
    person: pd.DataFrame | None = None,
    time_period: str | None = None,
) -> Frame:
    """``frame`` with its household or person table replaced."""

    return uk_national_frame(
        person=frame.table("person") if person is None else person,
        benunit=frame.table("benunit"),
        household=frame.table("household") if household is None else household,
        time_period=uk_time_period(frame) if time_period is None else time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


def _carried(household: pd.DataFrame, person: pd.DataFrame) -> np.ndarray:
    """Each household's reported reduction, summed over its persons."""

    summed = person.groupby("person_household_id")[_REDUCTION].sum()
    return (
        summed.reindex(household["household_id"].to_numpy())
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


# --- The oracle ---------------------------------------------------------------------


@functools.cache
def _support_areas(system: str) -> tuple[tuple[str, str, str, dict[str, int]], ...]:
    """Each toy support area as (region, decile, authority, weights)."""

    arrays = _supports()[system].arrays
    weights = [
        column
        for column in arrays
        if column == "households" or column.startswith("households_band_")
    ]
    return tuple(
        (
            str(arrays["frs_region"][area]),
            str(arrays["imd_decile"][area]),
            str(arrays["local_authority_code"][area]),
            {column: int(arrays[column][area]) for column in weights},
        )
        for area in range(len(arrays["area"]))
    )


def _cell_weights(
    system: str, region: str, decile: str | None, band: str | None
) -> list[tuple[int, str]]:
    """Each toy support area of the cell with mass, as (weight, authority):
    the region's areas, of the decile where the level keeps it, weighed by
    the band's households where the level keeps the band and by households
    otherwise."""

    weight = "households" if band is None else f"households_band_{band.lower()}"
    return [
        (weights[weight], authority)
        for area_region, area_decile, authority, weights in _support_areas(system)
        if area_region == region
        and (decile is None or area_decile == decile)
        and weights[weight]
    ]


@dataclass(frozen=True)
class _Expected:
    """One household row's localisation, worked again: its cell's exact
    expected Band D, its authority's Band D, its ratio and path, its new
    bill, rebate and summed reduction, and whether the cap fired."""

    cell: tuple[str, str | None, str | None] | None
    expected_band_d: Fraction | None
    authority_band_d: Fraction | None
    ratio: float
    path: str
    bill: float
    rebate: float
    reduction: float
    capped: bool


def _expected_rows(frame: Frame) -> list[_Expected]:
    """Every household row's localisation from the toy supports, the
    invented Band D and the statute restated, in row order. A row's level is
    its case's stated one (each copy's L3), which the level test holds equal
    to the area draw's."""

    household = frame.table("household")
    reductions = _carried(household, frame.table("person"))
    band_d = _band_d().band_d
    rows = []
    for position in range(len(household)):
        row = household.iloc[position]
        case = COUNCIL_TAX_CASES[int(row["source_household_id"]) - 1]
        copy = row[_CHANNEL] != BASE_FRS_SUPPORT_CHANNEL
        region = str(row["region"])
        bill, rebate = float(row["council_tax"]), float(row["council_tax_rebate"])
        reduction = float(reductions[position])
        if region == _NORTHERN_IRELAND:
            rows.append(
                _Expected(
                    None,
                    None,
                    None,
                    float("nan"),
                    "unchanged_northern_ireland",
                    bill,
                    rebate,
                    reduction,
                    False,
                )
            )
            continue
        level = "L3" if copy else case.household.level
        band = str(row["council_tax_band"])
        observed = bool(row["council_tax_band_observed"])
        cell = (
            region,
            str(int(row["imd_decile"])) if level in ("L0", "L1") else None,
            band if level in ("L0", "L2") else None,
        )
        weights = _cell_weights(str(row["atomic_area_system"]), *cell)
        expected = Fraction(
            sum(mass * band_d[code] for mass, code in weights),
            sum(mass for mass, _ in weights),
        )
        own = band_d[str(row["local_authority_code"])]
        ratio = float(own / expected)
        provenance = str(row[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN])
        if copy:
            path = "ratio_support_channel"
        elif provenance == FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS:
            path = "ratio_observed_gross"
        else:
            kind = {
                FRS_COUNCIL_TAX_BILL_MISSING: "missing",
                FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION: "unknown_reduction",
            }[provenance]
            path = f"formula_{kind}" if observed else f"ratio_{kind}_unbanded"
        if path in _FORMULA_PATHS:
            statute = own * _STATUTE[_nation(region)][band]
            if float(row["council_tax_single_adult_raw"]) == 1.0:
                statute *= _SINGLE_ADULT
            new_bill = float(statute)
        else:
            new_bill = ratio * bill
        scaled_rebate, scaled_reduction = ratio * rebate, ratio * reduction
        rows.append(
            _Expected(
                cell,
                expected,
                own,
                ratio,
                path,
                new_bill,
                min(scaled_rebate, new_bill),
                min(scaled_reduction, new_bill),
                scaled_rebate > new_bill or scaled_reduction > new_bill,
            )
        )
    return rows


@functools.cache
def _expected(clones: int) -> tuple[_Expected, ...]:
    return tuple(_expected_rows(_frame(clones)))


def _sources(household: pd.DataFrame) -> np.ndarray:
    return household["source_household_id"].to_numpy(dtype=np.int64)


def _frs(household: pd.DataFrame) -> np.ndarray:
    return household[_CHANNEL].to_numpy(dtype=object) == BASE_FRS_SUPPORT_CHANNEL


def _great_britain(household: pd.DataFrame) -> np.ndarray:
    return household["region"].to_numpy(dtype=object) != _NORTHERN_IRELAND


# --- The reviewed declaration -------------------------------------------------------


def test_the_reviewed_operations_are_declarative_and_state_the_statute() -> None:
    operations = council_tax_localise_operation_parameters()
    for kind, parameters in operations:
        _reject_executable_parameter_keys(parameters, context=kind)
        _reject_incumbent_dependencies(parameters, context=kind)
        assert json.loads(json.dumps(parameters)) == parameters
    assert [kind for kind, _ in operations] == [
        "verify_vendored_fact_resource",
        "derive",
        COUNCIL_TAX_LOCALISE_OPERATION_KIND,
        "record_mass_conservation_receipt",
    ]
    (_, facts), (_, derive), (_, localise), (_, receipt) = operations
    # The Scottish resource's feed check is the declaration its runtime arm
    # accepts at the build period.
    check_scotland_band_d_declaration(
        facts, stage=COUNCIL_TAX_LOCALISE_STAGE_NAME, period=2024
    )
    assert facts["runtime_sha256_required"] is True
    assert facts["require_before_source_read"] is True
    assert derive["output"] == COUNCIL_TAX_FRS_BILL_COLUMN == "council_tax_frs_bill"
    assert "max(council_tax - council_tax_rebate, 0)" in derive["formula"]
    assert COUNCIL_TAX_LOCALISE_OPERATION_KIND == "localise_council_tax"
    assert localise["method"] == COUNCIL_TAX_LOCALISE_METHOD
    assert localise["band_d_resources"] == {
        "england_wales": UK_BAND_D_ENGLAND_WALES_RESOURCE,
        "scotland": SCOTLAND_BAND_D_RESOURCE,
    }
    assert localise["band_d_year"] == "2024-25"
    assert localise["authority_column"] == "local_authority_code"
    assert localise["reference"] == "expected_band_d_of_draw_cell"
    assert (localise["draw_stage"], localise["copy_draw_stage"]) == (
        "atomic_area_assign",
        "spi_copy_areas",
    )
    # The supports are the area draw's own.
    ((_, draw),) = atomic_area_assign_operation_parameters()
    assert localise["support_sources"] == draw["sources"]
    assert localise["provenance_column"] == FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN
    assert localise["ratio_classes"] == [FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS]
    assert localise["formula_classes"] == [
        FRS_COUNCIL_TAX_BILL_MISSING,
        FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION,
    ]
    assert localise["formula_needs"] == "council_tax_band_observed"
    # The statute, restated here.
    assert localise["band_ratios"] == uk_council_tax_band_ratio_text()
    for nation, proportions in _STATUTE.items():
        text = localise["band_ratios"][nation]
        assert list(text) == list(proportions), nation
        assert {
            band: Fraction(*map(int, ratio.split("/"))) for band, ratio in text.items()
        } == proportions, nation
    assert {
        text.split("/")[1] for text in localise["band_ratios"]["WALES"].values()
    } == {"9"}
    assert {
        text.split("/")[1] for text in localise["band_ratios"]["SCOTLAND"].values()
    } == {"360"}
    assert localise["single_adult_column"] == "council_tax_single_adult_raw"
    assert localise["single_adult_discount"] == 0.25
    assert localise["scaled_columns"] == {
        "household": ["council_tax_rebate"],
        "person": [_REDUCTION],
    }
    assert localise["cap"] == "reduction_and_rebate_at_bill"
    assert localise["excluded_regions"] == [_NORTHERN_IRELAND]
    assert UK_COUNCIL_TAX_LOCALISE_EXCLUDED_REGIONS == (_NORTHERN_IRELAND,)
    assert localise["support_channel_rows"] == "ratio_at_copy_draw_level"
    # The mass receipt couples to the terminal family gate as the private
    # rents' does.
    rents_receipt = dict(private_rent_operation_parameters())[
        "record_mass_conservation_receipt"
    ]
    assert receipt == {
        "entity": "household",
        "reason": UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON,
        "declared_factor": 1.0,
        "gate_coupling": rents_receipt["gate_coupling"],
    }
    assert UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS == (COUNCIL_TAX_FRS_BILL_COLUMN,)
    assert UK_COUNCIL_TAX_LOCALISE_REWRITES == (
        "council_tax",
        "council_tax_rebate",
        _REDUCTION,
    )


def test_the_transform_runs_only_as_its_reviewed_stage() -> None:
    payloads = toy_conditioned_support_payloads()
    stage = _stage()
    UKCouncilTaxLocaliseStageTransform(stage=stage, supports=payloads, band_d=_band_d())
    facts, derive, localise, receipt = stage.operations
    drifted = SourceOperationSpec(
        kind=localise.kind, parameters={**localise.parameters, "cap": "none"}
    )
    england_wales, scotland = stage.artifacts
    for overrides, message in (
        ({"stage": "private_rent"}, "Expected stage"),
        ({"grain": "person"}, "grain must be 'household'"),
        ({"outputs": ()}, "writes exactly its own columns"),
        ({"nonnegative_outputs": ()}, "writes exactly its own columns"),
        ({"rewrites": ("council_tax", "council_tax_rebate")}, "its own columns"),
        ({"operations": (facts, derive, drifted, receipt)}, "drifted"),
        ({"operations": (derive, facts, localise, receipt)}, "order drifted"),
        ({"operations": (facts, derive, localise)}, "order drifted"),
        ({"artifacts": (england_wales,)}, "and no other"),
        (
            {
                "artifacts": (
                    england_wales,
                    {**scotland, "runtime_sha256_required": False},
                )
            },
            "runtime_sha256_required",
        ),
        (
            {
                "artifacts": (
                    england_wales,
                    scotland,
                    {
                        "role": "other",
                        "resource": "x.json",
                        "runtime_sha256_required": True,
                    },
                )
            },
            "and no other",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            UKCouncilTaxLocaliseStageTransform(
                stage=_stage(**overrides), supports=payloads, band_d=_band_d()
            )
    with pytest.raises(TypeError, match="SourceStageSpec"):
        UKCouncilTaxLocaliseStageTransform(stage=object(), supports=payloads)
    with pytest.raises(TypeError, match="UKCouncilTaxBandD"):
        UKCouncilTaxLocaliseStageTransform(
            stage=stage, supports=payloads, band_d=dict(_band_d().band_d)
        )


def test_the_packaged_stage_is_the_reviewed_stage() -> None:
    """sources.yaml declares G6 as the reviewed declaration states it, so the
    spine runs the transform as the packaged stage (microcosm#931)."""
    from microcosm.build.country_spec import load_country_spec

    spec = load_country_spec("uk")
    assert spec.sources is not None
    packaged = spec.sources.stage_map()[COUNCIL_TAX_LOCALISE_STAGE_NAME]
    reviewed = _stage()
    assert packaged.grain == reviewed.grain == "household"
    assert [dict(artifact) for artifact in packaged.artifacts] == [
        dict(artifact) for artifact in reviewed.artifacts
    ]
    assert [(op.kind, dict(op.parameters)) for op in packaged.operations] == [
        (kind, parameters)
        for kind, parameters in council_tax_localise_operation_parameters()
    ]
    assert tuple(packaged.outputs) == UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS
    assert tuple(packaged.nonnegative_outputs) == UK_COUNCIL_TAX_LOCALISE_OUTPUT_COLUMNS
    assert tuple(packaged.rewrites) == UK_COUNCIL_TAX_LOCALISE_REWRITES
    UKCouncilTaxLocaliseStageTransform(
        stage=packaged, supports=toy_conditioned_support_payloads(), band_d=_band_d()
    )


def test_the_transform_needs_a_run_and_draws_with_the_area_draws_definition() -> None:
    transform = UKCouncilTaxLocaliseStageTransform(
        stage=_stage(), supports=toy_conditioned_support_payloads(), band_d=_band_d()
    )
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    assert UKCouncilTaxLocaliseStageTransform.output_columns() == (
        COUNCIL_TAX_FRS_BILL_COLUMN,
    )
    # A level reads no stream, and the transform builds G2's definition at
    # seed 0, the area draw's own.
    assert validate_assignment_spec(transform.definition)["stream"][3] == 0
    assert transform.definition == _definition()


# --- The references -----------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_each_level_is_the_area_draws_and_every_copy_draws_at_l3(clones) -> None:
    household = _frame(clones).table("household")
    references = _local(clones).references
    frs = _frs(household)
    # The area draw (G2) on the FRS rows' keys and inputs gives their areas
    # and levels; on the stage's whole frame, the copies' inputs are missing
    # by rule, so it gives them L3, the level the SPI-copy area stage drew
    # them at.
    base = draw_uk_spine_areas(
        household.loc[frs], definition=_definition(), supports=_supports()
    )
    assert np.array_equal(references.levels[frs], np.asarray(base.levels))
    assert (
        base.columns["atomic_area_code"].to_numpy(dtype=object)
        == household.loc[frs, "atomic_area_code"].to_numpy(dtype=object)
    ).all()
    whole = draw_uk_spine_areas(
        household, definition=_definition(), supports=_supports()
    )
    assert np.array_equal(references.levels, np.asarray(whole.levels))
    assert (references.levels[~frs] == "L3").all()
    # Both copy kinds, SPI copies and band donors, are among them.
    assert set(household.loc[~frs, _ORDINAL]) == {1, 2}
    stated = np.asarray(
        [
            COUNCIL_TAX_CASES[source - 1].household.level
            for source in _sources(household)
        ]
    )
    assert np.array_equal(references.levels[frs], stated[frs])
    assert set(references.levels[frs]) == {"L0", "L1", "L2", "L3"}


@pytest.mark.parametrize("clones", [1, 3])
def test_each_reference_is_the_exact_expected_band_d_of_its_draw_cell(clones) -> None:
    references = _local(clones).references
    expected = _expected(clones)
    household = _frame(clones).table("household")
    great_britain = _great_britain(household)
    assert np.array_equal(references.great_britain, great_britain)
    for position, row in enumerate(expected):
        if not great_britain[position]:
            assert references.expected_band_d[position] is None
            assert references.authority_band_d[position] is None
            assert np.isnan(references.ratios[position])
            continue
        assert isinstance(references.expected_band_d[position], Fraction)
        assert references.expected_band_d[position] == row.expected_band_d, position
        assert references.authority_band_d[position] == row.authority_band_d
        # One correctly rounded division of the two exact rationals.
        assert _bits(references.ratios[position]) == _bits(row.ratio), position
    # Region-level cells of more than one toy authority give ratios other
    # than one, FRS rows and copies alike.
    ratios = references.ratios
    frs = _frs(household)
    assert (ratios[great_britain & frs] != 1.0).any()
    assert (ratios[great_britain & ~frs] != 1.0).any()


def _cell_sum(cell, system: str, expected: Fraction) -> Fraction:
    """The sum over the cell's areas of each area's draw probability times
    its authority's Band D over the cell's expected Band D."""

    weights = _cell_weights(system, *cell)
    total = sum(mass for mass, _ in weights)
    band_d = _band_d().band_d
    return sum(
        (Fraction(mass, total) * band_d[code] / expected for mass, code in weights),
        Fraction(0),
    )


def test_the_ratio_averages_one_over_every_great_britain_cell() -> None:
    """Over every Great Britain draw cell of the toy supports, at every level,
    the ratio averages exactly one under the draw's own probabilities, so a
    cell's expected bill is its FRS bill."""

    sweep = council_tax_cell_sweep()
    frame = council_tax_spine_frame(
        clones=1,
        cases=[
            replace(COUNCIL_TAX_CASES[0], household=household) for household in sweep
        ],
    )
    household = frame.table("household")
    references = uk_council_tax_references(
        household, definition=_definition(), supports=_supports(), band_d=_band_d()
    )
    frs = _frs(household)
    stated = np.asarray([case.level for case in sweep])[_sources(household) - 1]
    assert np.array_equal(references.levels[frs], stated[frs])
    cells = set()
    for position in np.flatnonzero(frs).tolist():
        row = household.iloc[position]
        level = str(references.levels[position])
        cell = (
            str(row["region"]),
            str(int(row["imd_decile"])) if level in ("L0", "L1") else None,
            str(row["council_tax_band"]) if level in ("L0", "L2") else None,
        )
        system = str(row["atomic_area_system"])
        expected = references.expected_band_d[position]
        assert _cell_sum(cell, system, expected) == 1, cell
        cells.add((system, cell))
    # One cell per sweep household: every level of every Great Britain
    # region's cells.
    assert len(cells) == len(sweep)
    # The stage's own frames, copies included.
    for clones in (1, 3):
        stage_household = _frame(clones).table("household")
        for position, row in enumerate(_expected(clones)):
            if row.cell is not None:
                system = str(stage_household["atomic_area_system"].iloc[position])
                assert _cell_sum(row.cell, system, row.expected_band_d) == 1


# --- The paths and the amounts ------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_each_row_takes_the_path_its_case_names(clones) -> None:
    household = _frame(clones).table("household")
    local = _local(clones)
    frs = _frs(household)
    stated = np.asarray(
        [COUNCIL_TAX_CASES[source - 1].path for source in _sources(household)]
    )
    assert np.array_equal(local.paths[frs], stated[frs])
    assert np.array_equal(local.paths, [row.path for row in _expected(clones)])
    assert np.array_equal(local.support_channel, ~frs)
    # Every FRS path and both support-channel paths are taken.
    assert set(local.paths[frs]) == set(UK_COUNCIL_TAX_LOCALISE_FRS_PATHS)
    assert set(local.paths[~frs]) == set(UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS)


@pytest.mark.parametrize("clones", [1, 3])
def test_every_amount_is_the_ratio_the_statute_or_the_cap_bit_for_bit(clones) -> None:
    frame = _frame(clones)
    after = _run(clones).frame
    household = after.table("household")
    expected = _expected(clones)
    assert np.array_equal(
        _bits(household["council_tax"]), _bits([row.bill for row in expected])
    )
    assert np.array_equal(
        _bits(household["council_tax_rebate"]), _bits([row.rebate for row in expected])
    )
    assert np.array_equal(
        _bits(_carried(household, after.table("person"))),
        _bits([row.reduction for row in expected]),
    )
    assert np.array_equal(_local(clones).capped, [row.capped for row in expected])
    # Where the cap does not fire, a ratio row's bill, rebate and reduction
    # are its ratio times the stage's input, bit for bit.
    before = frame.table("household")
    carried = _carried(before, frame.table("person"))
    after_carried = _carried(household, after.table("person"))
    for position, row in enumerate(expected):
        if row.path not in _RATIO_PATHS:
            continue
        ratio = row.ratio
        assert _bits(household["council_tax"].iloc[position]) == _bits(
            ratio * float(before["council_tax"].iloc[position])
        )
        if not row.capped:
            assert _bits(household["council_tax_rebate"].iloc[position]) == _bits(
                ratio * float(before["council_tax_rebate"].iloc[position])
            )
            assert _bits(after_carried[position]) == _bits(ratio * carried[position])
    assert household["council_tax"].dtype == np.float64
    assert household["council_tax_rebate"].dtype == np.float64
    assert after.table("person")[_REDUCTION].dtype == np.float64


@pytest.mark.parametrize("clones", [1, 3])
def test_statute_rows_price_their_band_at_their_authority_by_nation(clones) -> None:
    frame = _frame(clones)
    before = frame.table("household")
    after = _run(clones).frame
    household = after.table("household")
    expected = _expected(clones)
    carried = _carried(before, frame.table("person"))
    after_carried = _carried(household, after.table("person"))
    band_d = _band_d().band_d
    priced = set()
    scaled_in_wide_cells = 0
    for position in np.flatnonzero(np.isin(_local(clones).paths, _FORMULA_PATHS)):
        row = before.iloc[position]
        nation = _nation(str(row["region"]))
        band = str(row["council_tax_band"])
        single = float(row["council_tax_single_adult_raw"]) == 1.0
        bill = band_d[str(row["local_authority_code"])] * _STATUTE[nation][band]
        if single:
            bill *= _SINGLE_ADULT
        assert bool(row["council_tax_band_observed"])
        assert _bits(household["council_tax"].iloc[position]) == _bits(float(bill))
        priced.add((nation, band, single))
        # Below the cap, the rebate and the reduction take the row's ratio, as
        # on a ratio row: the statute prices the bill alone.
        ratio = expected[position].ratio
        if not expected[position].capped:
            assert _bits(household["council_tax_rebate"].iloc[position]) == _bits(
                ratio * float(row["council_tax_rebate"])
            )
            assert _bits(after_carried[position]) == _bits(ratio * carried[position])
            if ratio != 1.0 and carried[position] > 0.0:
                scaled_in_wide_cells += 1
    # Wales's band I with and without the discount, Scotland's bands E to H,
    # and England's.
    assert {("WALES", "I", True), ("WALES", "I", False)} <= priced
    assert {band for nation, band, _ in priced if nation == "SCOTLAND"} >= set("EFGH")
    assert {nation for nation, _, _ in priced} == {"ENGLAND", "WALES", "SCOTLAND"}
    assert {single for _, _, single in priced} == {True, False}
    # A statute row in a cell of two authorities, its ratio not one, carries a
    # reduction the cap leaves alone: the statute prices its own authority's
    # Band D, not the cell's expected one, and its reduction takes the ratio.
    assert scaled_in_wide_cells


@pytest.mark.parametrize("clones", [1, 3])
def test_a_missing_bill_or_unknown_reduction_takes_the_statute_only_with_a_band(
    clones,
) -> None:
    before = _frame(clones).table("household")
    paths = _local(clones).paths
    frs = _frs(before) & _great_britain(before)
    provenance = before[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN].to_numpy(dtype=object)
    observed = before["council_tax_band_observed"].to_numpy(dtype=bool)
    for value in (FRS_COUNCIL_TAX_BILL_MISSING, FRS_COUNCIL_TAX_BILL_UNKNOWN_REDUCTION):
        rows = frs & (provenance == value)
        assert (rows & observed).any() and (rows & ~observed).any(), value
        assert np.isin(paths[rows & observed], _FORMULA_PATHS).all(), value
        assert np.isin(paths[rows & ~observed], _RATIO_PATHS).all(), value
    assert (
        paths[frs & (provenance == FRS_COUNCIL_TAX_BILL_OBSERVED_GROSS)]
        == ("ratio_observed_gross")
    ).all()


@pytest.mark.parametrize("clones", [1, 3])
def test_support_channel_rows_take_the_ratio_whatever_they_inherit(clones) -> None:
    before = _frame(clones).table("household")
    household = _run(clones).frame.table("household")
    references = _local(clones).references
    copies = ~_frs(before) & _great_britain(before)
    # Some copies inherit a provenance the statute would price, with an
    # observed band: they take the ratio at their L3 cell all the same.
    inherited = before[FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN].to_numpy(dtype=object)
    observed = before["council_tax_band_observed"].to_numpy(dtype=bool)
    assert (
        copies & np.isin(inherited, [FRS_COUNCIL_TAX_BILL_MISSING]) & observed
    ).any()
    assert (_local(clones).paths[copies] == "ratio_support_channel").all()
    assert (references.levels[copies] == "L3").all()
    assert np.array_equal(
        _bits(household.loc[copies, "council_tax"]),
        _bits(
            references.ratios[copies]
            * before.loc[copies, "council_tax"].to_numpy(dtype=float)
        ),
    )


@pytest.mark.parametrize("clones", [1, 3])
def test_the_cap_holds_reductions_at_the_new_bill(clones) -> None:
    before = _frame(clones).table("household")
    after = _run(clones).frame
    household = after.table("household")
    local = _local(clones)
    paths = local.paths
    sources = _sources(before)
    frs = _frs(before)
    great_britain = _great_britain(before)
    carried = _carried(household, after.table("person"))
    bill = household["council_tax"].to_numpy(dtype=float)
    rebate = household["council_tax_rebate"].to_numpy(dtype=float)
    assert (carried[great_britain] <= bill[great_britain]).all()
    assert (rebate[great_britain] <= bill[great_britain]).all()
    # The rebate still equals the reduction on every row.
    assert np.array_equal(_bits(rebate), _bits(carried))
    # A missing bill below its reduction, and a zero bill with one: capped
    # on ratio paths. A statutory bill below its scaled reduction: capped on
    # the statute's path. A reduction equal to its bill: kept.
    for source, path, capped in (
        (4, "ratio_missing_unbanded", True),
        (7, "ratio_observed_gross", True),
        (29, "formula_missing", True),
        (8, "ratio_observed_gross", False),
    ):
        rows = frs & (sources == source)
        assert (paths[rows] == path).all(), source
        assert (local.capped[rows] == capped).all(), source
        if capped:
            assert np.array_equal(_bits(carried[rows]), _bits(bill[rows])), source
    assert (bill[frs & (sources == 7)] == 0.0).all()
    assert (carried[frs & (sources == 7)] == 0.0).all()
    assert local.capped[np.isin(paths, _RATIO_PATHS)].any()
    assert local.capped[np.isin(paths, _FORMULA_PATHS)].any()


def test_a_rebate_and_a_reduction_that_differ_are_scaled_and_capped_apart() -> None:
    """The root and the housing shell write the rebate equal to the
    reduction, but the stage scales and caps each on its own. A household
    whose reduction alone exceeds its bill and one whose rebate alone does
    both count as capped, and the rebate is held equal to the reduction only
    where it was."""

    frame = _frame(1)
    household = frame.table("household").copy()
    person = frame.table("person").copy()
    sources = _sources(household)
    frs = _frs(household)
    # Source 1, a ratio row without a reduction, takes a reduction of twice
    # its bill, its rebate left at zero; source 2 a rebate of twice its bill,
    # its reduction left at zero.
    reduction_only = frs & (sources == 1)
    rebate_only = frs & (sources == 2)
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)
    carrier = first & person["person_household_id"].isin(
        household.loc[reduction_only, "household_id"]
    ).to_numpy(dtype=bool)
    assert carrier.sum() == 1
    person.loc[carrier, _REDUCTION] = 2.0 * float(
        household.loc[reduction_only, "council_tax"].iloc[0]
    )
    household.loc[rebate_only, "council_tax_rebate"] = (
        2.0 * household.loc[rebate_only, "council_tax"]
    )
    edited = _with(frame, household=household, person=person)
    local = localise_uk_council_tax(
        edited.table("household"),
        edited.table("person"),
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d(),
    )
    assert local.capped[reduction_only].all()
    assert local.capped[rebate_only].all()
    result = _localise(edited)
    after = result.frame.table("household")
    carried = _carried(after, result.frame.table("person"))
    bill = after["council_tax"].to_numpy(dtype=float)
    assert np.array_equal(_bits(carried[reduction_only]), _bits(bill[reduction_only]))
    assert (after.loc[reduction_only, "council_tax_rebate"] == 0.0).all()
    assert np.array_equal(
        _bits(after.loc[rebate_only, "council_tax_rebate"]), _bits(bill[rebate_only])
    )
    assert (carried[rebate_only] == 0.0).all()
    assert result.evidence()["checks"]["rebate_equals_reduction"] is True


@pytest.mark.parametrize("clones", [1, 3])
def test_northern_ireland_is_left_bit_for_bit_and_never_capped(clones) -> None:
    frame = _frame(clones)
    before = frame.table("household")
    result = _run(clones)
    after = result.frame.table("household")
    ni = ~_great_britain(before)
    assert ni.any() and (ni & _frs(before)).any() and (ni & ~_frs(before)).any()
    for column in ("council_tax", "council_tax_rebate"):
        assert np.array_equal(
            _bits(after.loc[ni, column]), _bits(before.loc[ni, column])
        )
    ni_people = (
        frame.table("person")["person_household_id"]
        .isin(before.loc[ni, "household_id"])
        .to_numpy(dtype=bool)
    )
    assert np.array_equal(
        _bits(result.frame.table("person")[_REDUCTION].to_numpy()[ni_people]),
        _bits(frame.table("person")[_REDUCTION].to_numpy()[ni_people]),
    )
    assert not _local(clones).capped[ni].any()
    # Source 25's FRS rows hold a reduction above their zero bill, and keep it.
    above = ni & _frs(before) & (_sources(before) == 25)
    assert (after.loc[above, "council_tax"] == 0.0).all()
    assert (after.loc[above, "council_tax_rebate"] > 0.0).all()
    evidence = result.evidence()
    assert evidence["checks"]["northern_ireland_reductions_within_bill"] is False
    assert evidence["checks"]["northern_ireland_unchanged"] is True
    assert evidence["northern_ireland_reduction_above_bill_sources"] == extreme_count(1)
    # Without it, Northern Ireland's reductions lie within their bills.
    household = before.copy()
    person = frame.table("person").copy()
    within = household["source_household_id"].to_numpy() == 25
    household.loc[within, ["council_tax_rebate"]] = 0.0
    person.loc[
        person["person_household_id"].isin(household.loc[within, "household_id"]),
        _REDUCTION,
    ] = 0.0
    clean = _localise(_with(frame, household=household, person=person)).evidence()
    assert clean["checks"]["northern_ireland_reductions_within_bill"] is True
    assert clean["northern_ireland_reduction_above_bill_sources"] == 0


# --- The kept bill ------------------------------------------------------------------


def _was_paid_bill(household: pd.DataFrame) -> np.ndarray:
    """The paid bill was_wealth.recipient_predictors computed its council-tax
    predictor as before it read the kept bill, spelled as it spelled it."""

    return np.maximum(
        household["council_tax"].to_numpy(dtype=float)
        - household["council_tax_rebate"].to_numpy(dtype=float),
        0.0,
    )


@pytest.mark.parametrize("clones", [1, 3])
def test_the_kept_bill_is_the_paid_bill_from_before_the_rewrite(clones) -> None:
    before = _frame(clones).table("household")
    after = _run(clones).frame.table("household")
    kept = after[COUNCIL_TAX_FRS_BILL_COLUMN]
    assert kept.dtype == np.float64
    assert (kept.to_numpy() >= 0.0).all()
    assert np.array_equal(_bits(kept), _bits(_was_paid_bill(before)))
    # Not the paid bill after the rewrite, which carries the area.
    assert not np.array_equal(_bits(kept), _bits(_was_paid_bill(after)))
    # The same on every clone of an FRS source.
    frs = _frs(before)
    by_source = pd.Series(kept.to_numpy()[frs]).groupby(_sources(before)[frs])
    assert (by_source.nunique() == 1).all()


def test_the_kept_bill_nets_the_rebate_where_it_differs_from_the_reduction() -> None:
    """WAS's paid bill nets the household's rebate, not its persons' reported
    reduction. Here source 5 keeps its reduction with no rebate, and source 1,
    which carries no reduction, takes a rebate."""

    frame = _frame(1)
    household = frame.table("household").copy()
    sources, frs = _sources(household), _frs(household)
    carried = _carried(household, frame.table("person"))
    with_reduction = frs & (sources == 5)
    without_reduction = frs & (sources == 1)
    assert (carried[with_reduction] > 0.0).all()
    assert (carried[without_reduction] == 0.0).all()
    household.loc[with_reduction, "council_tax_rebate"] = 0.0
    household.loc[without_reduction, "council_tax_rebate"] = 100.0
    kept = _localise(_with(frame, household=household)).frame.table("household")[
        COUNCIL_TAX_FRS_BILL_COLUMN
    ]
    assert np.array_equal(_bits(kept), _bits(_was_paid_bill(household)))


def test_the_kept_bill_is_wass_expression_on_edge_amounts() -> None:
    frame = _frame(1)
    household = frame.table("household").copy()
    person = frame.table("person").copy()
    edges = (
        (0.0, 0.0),
        (0.0, 120.0),
        (100.0, 100.0),
        (100.0, 250.5),
        (0.1 + 0.2, 0.3),
        (0.3, 0.1 + 0.2),
        (2.0**53 + 2.0, 1.0),
        (1e300, 1e299),
        (123_456_789.123, 0.001),
        (5e-324, 0.0),
        (987_654_321_987.65, 987_654_321_987.64),
    )
    pairs = [edges[position % len(edges)] for position in range(len(household))]
    household["council_tax"] = np.asarray([bill for bill, _ in pairs], dtype=np.float64)
    household["council_tax_rebate"] = np.asarray(
        [rebate for _, rebate in pairs], dtype=np.float64
    )
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)
    rebates = pd.Series(
        household["council_tax_rebate"].to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    person[_REDUCTION] = np.where(
        first, person["person_household_id"].map(rebates).to_numpy(dtype=float), 0.0
    )
    edited = _with(frame, household=household, person=person)
    kept = _localise(edited).frame.table("household")[COUNCIL_TAX_FRS_BILL_COLUMN]
    assert np.array_equal(_bits(kept), _bits(_was_paid_bill(household)))


# --- The frame ----------------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_the_stage_rewrites_its_three_columns_and_adds_the_kept_bill_alone(
    clones,
) -> None:
    frame = _frame(clones)
    result = _run(clones).frame
    before, after = frame.table("household"), result.table("household")
    assert list(after.columns) == [*before.columns, COUNCIL_TAX_FRS_BILL_COLUMN]
    kept = [
        column
        for column in before.columns
        if column not in ("council_tax", "council_tax_rebate")
    ]
    assert after[kept].equals(before[kept])
    assert after.index.equals(before.index)
    people = [
        column for column in frame.table("person").columns if column != _REDUCTION
    ]
    assert result.table("person")[people].equals(frame.table("person")[people])
    assert result.table("benunit").equals(frame.table("benunit"))
    assert np.array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    assert result.metadata == frame.metadata


@pytest.mark.parametrize("clones", [1, 3])
def test_the_mass_log_ends_with_the_stages_conserving_record(clones) -> None:
    frame = _frame(clones)
    log = _run(clones).frame.mass_log
    assert log[: len(frame.mass_log)] == frame.mass_log
    assert len(log) == len(frame.mass_log) + 1
    record = log[-1]
    total = float(frame.weights_for("household").total)
    assert record.entity == "household"
    assert record.reason == UK_COUNCIL_TAX_LOCALISE_MASS_CONSERVATION_REASON
    assert record.declared_factor == 1.0
    assert record.old_total == record.new_total == total


def _clone_zero_columns(clones: int) -> pd.DataFrame:
    """Clone 0's localised columns by source and support-channel ordinal."""

    result = _run(clones).frame
    household = result.table("household")
    table = pd.DataFrame(
        {
            "source": _sources(household),
            "ordinal": household[_ORDINAL].to_numpy(),
            **{
                column: _bits(household[column])
                for column in (
                    "council_tax",
                    "council_tax_rebate",
                    COUNCIL_TAX_FRS_BILL_COLUMN,
                )
            },
            _REDUCTION: _bits(_carried(household, result.table("person"))),
        }
    )
    zero = (household[_CLONE] == 0).to_numpy()
    return table.loc[zero].set_index(["source", "ordinal"]).sort_index()


def test_clone_zero_at_k_3_is_k_1_bit_for_bit_and_later_clones_differ() -> None:
    assert _clone_zero_columns(3).equals(_clone_zero_columns(1))
    household = _run(3).frame.table("household")
    frs = _frs(household)
    zero = household.loc[frs & (household[_CLONE] == 0).to_numpy()].set_index(
        "source_household_id"
    )["council_tax"]
    for clone in (1, 2):
        rows = household.loc[frs & (household[_CLONE] == clone).to_numpy()]
        assert (
            rows["council_tax"].to_numpy()
            != zero.loc[rows["source_household_id"]].to_numpy()
        ).any(), clone


def test_the_localisation_does_not_depend_on_row_order() -> None:
    frame = _frame(3)
    household, person = frame.table("household"), frame.table("person")
    shuffled = household.sample(frac=1.0, random_state=931)
    people = person.sample(frac=1.0, random_state=1931)
    ordered = _local(3)
    permuted = localise_uk_council_tax(
        shuffled,
        people,
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d(),
    )
    for column in ("council_tax", "council_tax_rebate", COUNCIL_TAX_FRS_BILL_COLUMN):
        mine, theirs = getattr(permuted, column), getattr(ordered, column)
        assert mine.index.equals(shuffled.index)
        assert np.array_equal(_bits(mine.loc[household.index]), _bits(theirs))
    reduction = permuted.council_tax_benefit_reported
    assert reduction.index.equals(people.index)
    assert np.array_equal(
        _bits(reduction.loc[person.index]), _bits(ordered.council_tax_benefit_reported)
    )
    position = pd.Series(np.arange(len(shuffled)), index=shuffled.index)
    back = position.loc[household.index].to_numpy()
    assert np.array_equal(permuted.paths[back], ordered.paths)
    assert np.array_equal(permuted.capped[back], ordered.capped)
    assert np.array_equal(
        _bits(permuted.references.ratios[back]), _bits(ordered.references.ratios)
    )


# --- The evidence -------------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_the_evidence_counts_true_sources_through_receipt_counts(clones) -> None:
    household = _frame(clones).table("household")
    sources = _sources(household)
    frs = _frs(household)
    zero = (household[_CLONE] == 0).to_numpy()
    great_britain = _great_britain(household)
    expected = _expected(clones)
    paths = np.asarray([row.path for row in expected])
    capped = np.asarray([row.capped for row in expected])

    def among(mask: np.ndarray) -> int:
        return len(set(sources[mask].tolist()))

    carried = _carried(household, _frame(clones).table("person"))
    bill = household["council_tax"].to_numpy(dtype=float)
    above = (household["council_tax_rebate"].to_numpy(dtype=float) > bill) | (
        carried > bill
    )
    evidence = _run(clones).evidence()
    assert list(evidence) == [
        "stage",
        "method",
        "clone_count",
        "band_d",
        "support_sha256",
        "sources_by_path",
        "support_channel_sources_by_path",
        "capped_sources",
        "northern_ireland_reduction_above_bill_sources",
        "checks",
        "outputs",
        "rewrites",
    ]
    assert evidence["stage"] == COUNCIL_TAX_LOCALISE_STAGE_NAME
    assert evidence["method"] == COUNCIL_TAX_LOCALISE_METHOD
    assert evidence["clone_count"] == clones
    assert evidence["band_d"] == {
        "england_wales": {
            "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
            "sha256": "supplied",
        },
        "scotland": {
            "resource": SCOTLAND_BAND_D_RESOURCE,
            "sha256": "supplied",
            "source_fact_feed": {},
        },
    }
    assert evidence["support_sha256"] == {
        system: support.sha256 for system, support in sorted(_supports().items())
    }
    # Each FRS path over every clone; the support channel's paths, the
    # capped and Northern Ireland's reductions above the bill in clone 0.
    assert evidence["sources_by_path"] == {
        path: rounded_count(among(frs & (paths == path)))
        for path in UK_COUNCIL_TAX_LOCALISE_FRS_PATHS
    }
    assert evidence["support_channel_sources_by_path"] == {
        path: rounded_count(among(~frs & zero & (paths == path)))
        for path in UK_COUNCIL_TAX_LOCALISE_SUPPORT_CHANNEL_PATHS
    }
    assert evidence["capped_sources"] == {
        "ratio": extreme_count(among(zero & capped & np.isin(paths, _RATIO_PATHS))),
        "formula": extreme_count(among(zero & capped & np.isin(paths, _FORMULA_PATHS))),
    }
    assert evidence["northern_ireland_reduction_above_bill_sources"] == extreme_count(
        among(zero & ~great_britain & above)
    )
    assert evidence["checks"] == {
        check: check != "northern_ireland_reductions_within_bill"
        for check in UK_COUNCIL_TAX_LOCALISE_CHECKS
    }
    assert evidence["outputs"] == [COUNCIL_TAX_FRS_BILL_COLUMN]
    assert evidence["rewrites"] == list(UK_COUNCIL_TAX_LOCALISE_REWRITES)


#: Each FRS path, the case that takes it (by position in COUNCIL_TAX_CASES)
#: and how many sources hold that case: one to six, a different number for
#: each path.
_PATH_MULTIPLICITIES = (
    ("ratio_observed_gross", 0, 1),
    # A reduction above its missing bill: both sources are capped.
    ("ratio_missing_unbanded", 3, 2),
    ("ratio_unknown_reduction_unbanded", 5, 3),
    ("formula_missing", 2, 4),
    ("formula_unknown_reduction", 11, 5),
    ("unchanged_northern_ireland", 26, 6),
)


def test_each_evidence_count_is_its_own_on_a_frame_of_distinct_multiplicities() -> None:
    """Every FRS path holds a different number of sources, so a count read
    from another path, the two capped kinds swapped, or a Northern Ireland
    count that reads the reduction alone cannot agree with the exact counts.
    The first Northern Ireland source's rebate alone sits above its bill."""

    cases = []
    for path, position, count in _PATH_MULTIPLICITIES:
        assert COUNCIL_TAX_CASES[position].path == path
        cases += [COUNCIL_TAX_CASES[position]] * count
    first_northern_ireland = 1 + sum(
        count
        for path, _, count in _PATH_MULTIPLICITIES
        if path != "unchanged_northern_ireland"
    )
    for clones in (1, 3):
        frame = council_tax_spine_frame(clones=clones, cases=cases)
        household = frame.table("household").copy()
        rebate_only = _frs(household) & (_sources(household) == first_northern_ireland)
        assert (household.loc[rebate_only, "region"] == _NORTHERN_IRELAND).all()
        bill = household.loc[rebate_only, "council_tax"].to_numpy(dtype=float)
        reduction = _carried(household, frame.table("person"))[rebate_only]
        assert (reduction <= bill).all() and (bill < 600.0).all()
        household.loc[rebate_only, "council_tax_rebate"] = 600.0
        result = _localise(_with(frame, household=household))
        assert dict(result.sources_by_path) == {
            path: count for path, _, count in _PATH_MULTIPLICITIES
        }
        assert dict(result.capped_sources) == {"ratio": 2, "formula": 0}
        assert result.northern_ireland_reduction_above_bill_sources == 1
        assert result.evidence()["sources_by_path"] == {
            path: rounded_count(count) for path, _, count in _PATH_MULTIPLICITIES
        }


def test_every_evidence_field_but_the_clone_count_is_byte_equal_at_k_1_and_k_3() -> (
    None
):
    single, triple = ({**_run(clones).evidence()} for clones in (1, 3))
    assert (single.pop("clone_count"), triple.pop("clone_count")) == (1, 3)
    assert json.dumps(single) == json.dumps(triple)


def _first_seven_copied_in_clone_zero(source: int, clone: int) -> tuple[int, ...]:
    """The support channel's stand-in copies sources 1 to 7 in clone 0, and
    every other source in each later clone."""

    return (1,) if (clone == 0) == (source <= 7) else ()


def test_the_support_channel_is_counted_in_clone_zero_where_later_clones_differ() -> (
    None
):
    evidence = {}
    for clones in (1, 3):
        frame = council_tax_spine_frame(
            clones=clones, picks=_first_seven_copied_in_clone_zero
        )
        household = frame.table("household")
        sources = _sources(household)
        copies = ~_frs(household)
        zero = (household[_CLONE] == 0).to_numpy()
        great_britain = _great_britain(household)
        if clones == 3:
            # Later clones copy households clone 0 does not, in both nations.
            assert set(sources[copies & ~zero & great_britain]) - set(
                sources[copies & zero]
            )
            assert (copies & ~zero & ~great_britain).any()
        evidence[clones] = {**_localise(frame).evidence()}
        assert evidence[clones]["support_channel_sources_by_path"] == {
            "ratio_support_channel": rounded_count(
                len(set(sources[copies & zero & great_britain].tolist()))
            ),
            "unchanged_northern_ireland": rounded_count(
                len(set(sources[copies & zero & ~great_britain].tolist()))
            ),
        }
    single, triple = evidence[1], evidence[3]
    assert (single.pop("clone_count"), triple.pop("clone_count")) == (1, 3)
    assert json.dumps(single) == json.dumps(triple)


def test_the_capped_and_northern_ireland_counts_read_clone_zero_alone() -> None:
    """The capped households and Northern Ireland's reductions above the bill
    are counted among clone-0 rows. Here clone 1 alone gives five sources a
    reduction above the bill: the evidence stays the unedited frame's, while
    each count taken over every clone would fall in another band."""

    frame = _frame(3)
    household = frame.table("household").copy()
    person = frame.table("person").copy()
    sources = _sources(household)
    clone_one = _frs(household) & (household[_CLONE] == 1).to_numpy()
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)
    # An invented rebate and reduction above every bill these rows take: a
    # ratio row (source 1), two statute rows (sources 3 and 19), and two
    # Northern Ireland rows whose bill is zero (sources 26 and 28). None
    # carries a reduction before.
    edited = (1, 3, 19, 26, 28)
    for source in edited:
        row = clone_one & (sources == source)
        assert row.sum() == 1, source
        members = person["person_household_id"].isin(household.loc[row, "household_id"])
        assert (person.loc[members, _REDUCTION] == 0.0).all(), source
        household.loc[row, "council_tax_rebate"] = 100_000.0
        person.loc[first & members.to_numpy(dtype=bool), _REDUCTION] = 100_000.0
    changed = _with(frame, household=household, person=person)
    local = localise_uk_council_tax(
        changed.table("household"),
        changed.table("person"),
        definition=_definition(),
        supports=_supports(),
        band_d=_band_d(),
    )
    paths = local.paths
    assert local.capped[clone_one & np.isin(sources, (1, 3, 19))].all()
    evidence = _localise(changed).evidence()
    assert json.dumps(evidence) == json.dumps(_run(3).evidence())
    zero = (household[_CLONE] == 0).to_numpy()

    def among(mask: np.ndarray) -> int:
        return len(set(sources[mask].tolist()))

    for kind, kind_paths in (("ratio", _RATIO_PATHS), ("formula", _FORMULA_PATHS)):
        capped = local.capped & np.isin(paths, kind_paths)
        assert evidence["capped_sources"][kind] == extreme_count(among(zero & capped))
        assert extreme_count(among(capped)) != extreme_count(among(zero & capped))
    bill = household["council_tax"].to_numpy(dtype=float)
    above = ~_great_britain(household) & (
        (household["council_tax_rebate"].to_numpy(dtype=float) > bill)
        | (_carried(household, person) > bill)
    )
    assert evidence["northern_ireland_reduction_above_bill_sources"] == (
        extreme_count(among(zero & above))
    )
    assert extreme_count(among(above)) != extreme_count(among(zero & above))


def test_northern_irelands_within_bill_report_reads_clone_zero_alone() -> None:
    """Whether Northern Ireland's reductions lie within their bills is read
    among clone-0 rows, as their count is. Without source 25's reduction
    above its zero bill, every Northern Ireland reduction lies within its
    bill; then clone 1 alone gives source 26 a rebate and a reduction above
    its zero bill, and the evidence stays as it was."""

    frame = _frame(3)
    household = frame.table("household").copy()
    person = frame.table("person").copy()
    sources = _sources(household)
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)

    def members(rows: np.ndarray) -> np.ndarray:
        return (
            person["person_household_id"]
            .isin(household.loc[rows, "household_id"])
            .to_numpy(dtype=bool)
        )

    within = sources == 25
    household.loc[within, "council_tax_rebate"] = 0.0
    person.loc[members(within), _REDUCTION] = 0.0
    clean = _localise(_with(frame, household=household, person=person)).evidence()
    assert clean["checks"]["northern_ireland_reductions_within_bill"] is True
    assert clean["northern_ireland_reduction_above_bill_sources"] == 0
    late = _frs(household) & (household[_CLONE] == 1).to_numpy() & (sources == 26)
    assert late.sum() == 1
    assert (household.loc[late, "council_tax"] == 0.0).all()
    assert (household.loc[late, "region"] == _NORTHERN_IRELAND).all()
    household.loc[late, "council_tax_rebate"] = 100.0
    person.loc[first & members(late), _REDUCTION] = 100.0
    edited = _localise(_with(frame, household=household, person=person))
    assert json.dumps(edited.evidence()) == json.dumps(clean)
    # Clone 1's row kept its rebate and reduction, above its zero bill.
    after = edited.frame.table("household")
    assert (after.loc[late, "council_tax_rebate"] == 100.0).all()
    assert (after.loc[late, "council_tax"] == 0.0).all()


def _leaves(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def test_the_evidence_prints_no_key_code_or_amount() -> None:
    household = _frame(3).table("household")
    evidence = _run(3).evidence()
    text = json.dumps(evidence, sort_keys=True)
    for column in (IDENTITY_COLUMN, "atomic_area_code", *UK_SPINE_LADDER_COLUMNS):
        values = {str(value) for value in household[column]}
        assert not any(value in text for value in values), column
    # No float anywhere: every number is a rounded or banded count, or the
    # clone count.
    assert not any(isinstance(leaf, float) for leaf in _leaves(evidence))


# --- The refusals -------------------------------------------------------------------

#: The digits a refusal may print: the Band D's fiscal year and build
#: period, the issue, and dtype names.
_DECLARED_DIGITS = ("microcosm#931", "2024-25", "2024", "float64", "int64")


def _assert_names_no_household(message: str, household: pd.DataFrame) -> None:
    text = message
    for token in _DECLARED_DIGITS:
        text = text.replace(token, "")
    assert not re.search(r"[0-9]", text), message
    for column in (IDENTITY_COLUMN, "atomic_area_code", "local_authority_code"):
        values = {
            value for value in household[column] if isinstance(value, str) and value
        }
        assert not any(value in message for value in values), column


def _refusal(frame: Frame, band_d: UKCouncilTaxBandD | None = None) -> str:
    with pytest.raises(ValueError) as caught:
        _localise(frame, band_d)
    message = str(caught.value)
    _assert_names_no_household(message, frame.table("household"))
    return message


def _edit_household(column: str, edit: Callable[[pd.DataFrame], object]) -> Frame:
    frame = _frame(1)
    household = frame.table("household").copy()
    household[column] = edit(household)
    return _with(frame, household=household)


def _first(household: pd.DataFrame, mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def _set_at(column: str, mask: Callable[[pd.DataFrame], np.ndarray], value: object):
    def edit(household: pd.DataFrame) -> pd.Series:
        values = household[column].copy()
        values.iloc[_first(household, mask(household))] = value
        return values

    return edit


def _frs_gb(household: pd.DataFrame) -> np.ndarray:
    return _frs(household) & _great_britain(household)


def _frs_ni(household: pd.DataFrame) -> np.ndarray:
    return _frs(household) & ~_great_britain(household)


@pytest.mark.parametrize(
    ("column", "edit", "message"),
    [
        (
            FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
            _set_at(
                FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
                _frs_gb,
                FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
            ),
            "a household there reads 'not_levied'",
        ),
        (
            # An SPI-channel row in Great Britain too, though its inherited
            # provenance chooses no path.
            FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
            _set_at(
                FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
                lambda household: ~_frs(household) & _great_britain(household),
                FRS_COUNCIL_TAX_BILL_NOT_LEVIED,
            ),
            "a household there reads 'not_levied'",
        ),
        (
            FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
            _set_at(FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN, _frs_ni, "observed_gross"),
            "does not read 'not_levied'",
        ),
        (
            FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
            _set_at(FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN, _frs_gb, "drawn"),
            "some household holds another value or none",
        ),
        (
            FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN,
            _set_at(FRS_COUNCIL_TAX_BILL_PROVENANCE_COLUMN, _frs_gb, pd.NA),
            "some household holds another value or none",
        ),
        (
            "council_tax",
            _set_at("council_tax", _frs_gb, np.nan),
            "'council_tax' as a finite nonnegative amount",
        ),
        (
            "council_tax",
            _set_at("council_tax", _frs_ni, np.inf),
            "'council_tax' as a finite nonnegative amount",
        ),
        (
            "council_tax_rebate",
            _set_at("council_tax_rebate", _frs_gb, -1.0),
            "'council_tax_rebate' as a finite nonnegative amount",
        ),
        (
            "council_tax",
            lambda household: household["council_tax"].round().astype(np.int64),
            "the column holds int64",
        ),
        (
            "council_tax_single_adult_raw",
            lambda household: household["council_tax_single_adult_raw"].astype(str),
            "reads a numeric 'council_tax_single_adult_raw'",
        ),
        (
            "local_authority_code",
            _set_at("local_authority_code", _frs_gb, ""),
            "some household lacks one",
        ),
    ],
)
def test_each_refusal_of_a_household_value_names_its_column_only(
    column, edit, message
) -> None:
    assert message in _refusal(_edit_household(column, edit))


@pytest.mark.parametrize("column", UK_COUNCIL_TAX_LOCALISE_HOUSEHOLD_INPUTS)
def test_a_frame_without_an_input_column_is_refused_by_name(column) -> None:
    frame = _frame(1)
    household = frame.table("household").drop(columns=[column])
    with pytest.raises(ValueError, match=re.escape(column)) as caught:
        localise_uk_council_tax(
            household,
            frame.table("person"),
            definition=_definition(),
            supports=_supports(),
            band_d=_band_d(),
        )
    _assert_names_no_household(str(caught.value), frame.table("household"))


def test_the_persons_reduction_is_read_once_per_household_and_never_negative() -> None:
    frame = _frame(1)
    person = frame.table("person").copy()
    shared = person["person_household_id"].duplicated(keep=False).to_numpy()
    person.loc[shared, _REDUCTION] = 5.0
    message = _refusal(_with(frame, person=person))
    assert "some household holds more than one positive value" in message
    negative = frame.table("person").copy()
    negative.loc[negative.index[0], _REDUCTION] = -1.0
    assert "'council_tax_benefit_reported' as a finite nonnegative" in _refusal(
        _with(frame, person=negative)
    )
    without = frame.table("person").drop(columns=[_REDUCTION])
    with pytest.raises(ValueError, match=_REDUCTION):
        localise_uk_council_tax(
            frame.table("household"),
            without,
            definition=_definition(),
            supports=_supports(),
            band_d=_band_d(),
        )
    stray = frame.table("person").copy()
    stray.loc[stray.index[0], "person_household_id"] = -1
    with pytest.raises(ValueError, match="some person names none") as caught:
        localise_uk_council_tax(
            frame.table("household"),
            stray,
            definition=_definition(),
            supports=_supports(),
            band_d=_band_d(),
        )
    _assert_names_no_household(str(caught.value), frame.table("household"))


def _without_band_d(*codes: str) -> UKCouncilTaxBandD:
    band_d = _band_d()
    return replace(
        band_d,
        band_d=MappingProxyType(
            {code: value for code, value in band_d.band_d.items() if code not in codes}
        ),
    )


def test_an_authority_without_band_d_is_refused_by_name() -> None:
    frame = _frame(1)
    household = frame.table("household")
    drawn = household.loc[_great_britain(household), "local_authority_code"]
    message = _refusal(frame, _without_band_d(str(drawn.iloc[0])))
    assert "some household's 'local_authority_code' has none" in message
    # An authority with mass in a household's cell, where no household is
    # drawn: London's region-level cell, on a frame of one London household.
    single = council_tax_spine_frame(clones=1, cases=[COUNCIL_TAX_CASES[3]])
    rows = single.table("household")
    undrawn = sorted(
        {"E09000001", "E09000030", "E09000033"} - set(rows["local_authority_code"])
    )
    assert undrawn
    message = _refusal(single, _without_band_d(undrawn[0]))
    assert "a code with mass in a household's cell has no value" in message
    with pytest.raises(TypeError, match="UKCouncilTaxBandD"):
        localise_uk_spine_council_tax(
            frame,
            definition=_definition(),
            supports=_supports(),
            band_d=dict(_band_d().band_d),
        )


def test_a_statute_band_its_nation_does_not_price_is_refused() -> None:
    # An English household whose observed band is I, which only Wales
    # prices: its missing bill would take the statute.
    case = replace(
        COUNCIL_TAX_CASES[2],
        household=ToyHousehold("LONDON", 1, "I", True, "L1"),
        provenance=FRS_COUNCIL_TAX_BILL_MISSING,
    )
    frame = council_tax_spine_frame(clones=1, cases=[case])
    assert "some such band is not one its region's nation prices" in _refusal(frame)


def test_the_draw_cell_guard_refuses_an_input_changed_after_the_draw() -> None:
    household = _frame(1).table("household")
    # Source 1 is London's band-B household at L0 in decile 1, where only
    # one toy area holds band B. Band F has households in that decile, so the
    # household stays at L0, but its drawn area holds no band F.
    one = _frs(household) & (_sources(household) == 1)
    assert set(_local(1).references.levels[one]) == {"L0"}
    guard = "is not an area of that cell with mass in the weight it samples by"
    assert guard in _refusal(
        _edit_household(
            "council_tax_band", _set_at("council_tax_band", lambda h: one, "F")
        )
    )
    # A decile whose cell holds another area; an authority or system that is
    # not the drawn area's.
    assert guard in _refusal(
        _edit_household("imd_decile", _set_at("imd_decile", lambda h: one, 4))
    )
    other = next(
        code
        for code in sorted(_band_d().band_d)
        if code != household["local_authority_code"].to_numpy()[_first(household, one)]
    )
    assert guard in _refusal(
        _edit_household(
            "local_authority_code",
            _set_at("local_authority_code", lambda h: one, other),
        )
    )
    assert guard in _refusal(
        _edit_household(
            "atomic_area_system", _set_at("atomic_area_system", lambda h: one, SCOT)
        )
    )


def test_the_frame_function_refuses_a_written_kept_bill_or_another_period() -> None:
    frame = _frame(1)
    household = frame.table("household").copy()
    household[COUNCIL_TAX_FRS_BILL_COLUMN] = 0.0
    assert "already carries 'council_tax_frs_bill'" in _refusal(
        _with(frame, household=household)
    )
    later = _with(frame, time_period="2025")
    with pytest.raises(ValueError, match="the frame's period is another"):
        _localise(later)
    with pytest.raises(TypeError, match="Frame"):
        _localise(frame.table("household"))


# --- The transform ------------------------------------------------------------------


def test_the_transform_gives_the_functions_frame_and_evidence() -> None:
    transform = UKCouncilTaxLocaliseStageTransform(
        stage=_stage(), supports=toy_conditioned_support_payloads(), band_d=_band_d()
    )
    for clones in (1, 3):
        frame = transform(_frame(clones))
        expected = _run(clones)
        for entity in ("household", "person", "benunit"):
            assert frame.table(entity).equals(expected.frame.table(entity)), entity
        assert frame.mass_log == expected.frame.mass_log
        assert transform.checkpoint_metadata() == {"evidence": expected.evidence()}


def _sha256(resource: str) -> str:
    return hashlib.sha256(
        files("microcosm.build.uk").joinpath(resource).read_bytes()
    ).hexdigest()


def test_the_transform_prices_at_the_committed_band_d_by_default() -> None:
    transform = UKCouncilTaxLocaliseStageTransform(
        stage=_stage(), supports=toy_conditioned_support_payloads()
    )
    transform(_frame(1))
    evidence = transform.checkpoint_metadata()["evidence"]
    committed = load_uk_council_tax_band_d()
    assert evidence["band_d"] == {
        "england_wales": {
            "resource": UK_BAND_D_ENGLAND_WALES_RESOURCE,
            "sha256": _sha256(UK_BAND_D_ENGLAND_WALES_RESOURCE),
        },
        "scotland": {
            "resource": SCOTLAND_BAND_D_RESOURCE,
            "sha256": _sha256(SCOTLAND_BAND_D_RESOURCE),
            "source_fact_feed": dict(committed.scotland_source_fact_feed),
        },
    }
    assert all(
        evidence["checks"][check] for check in UK_COUNCIL_TAX_LOCALISE_CHECKS[:4]
    )


def test_the_transform_runs_the_scottish_arm_with_the_frames_period() -> None:
    transform = UKCouncilTaxLocaliseStageTransform(
        stage=_stage(), supports=toy_conditioned_support_payloads(), band_d=_band_d()
    )
    with pytest.raises(ValueError, match="maps Scotland's Band D to build period"):
        transform(_with(_frame(1), time_period="2025"))
    assert transform.last_result is None
