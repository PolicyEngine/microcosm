"""The geography identity, G1b of the clone-first spine (microcosm#931).

G1b keys every household of the geographic expansion's frame by its source,
FRS vintage, true source and copy path, as the dense line's
``uk.full.identity`` keys it. These tests hold clone 0's key, and clone k's,
to the dense identity kernel itself on the same households, and check the
clone path, uniqueness, nesting across K, the dtype, the vintage read from
frs_release.json, the evidence and every refusal. Invented frames only; no
refusal prints a key, an id or a year of a row.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.atomic_area_support import IDENTITY_COLUMN
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.geographic_expand import (
    SOURCE_HOUSEHOLD_KEY_COLUMN,
    SOURCE_YEAR_COLUMN,
    expand_uk_geographic_clones,
)
from microcosm.build.uk_runtime.geographic_identity import (
    GEOGRAPHIC_IDENTITY_INPUTS,
    GEOGRAPHIC_IDENTITY_SOURCE,
    GEOGRAPHIC_IDENTITY_STAGE_NAME,
    UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS,
    UKGeographicIdentityStageTransform,
    frs_vintages_by_source_year,
    key_uk_households_by_clone_path,
)
from microcosm.build.uk_runtime.graph_population import (
    UKGeographyIdentityKernel,
    append_uk_population_nodes,
)
from microcosm.build.uk_runtime.household_lineage import GEOGRAPHIC_SUPPORT_BRANCH
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import rounded_count
from microcosm.frame import Frame
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from microcosm.graph.population import dtype_for_token
from test_support.microcosm_build.uk_clone_first_frames import base_id_tables

_RELEASE = load_uk_frs_release()
_PERIOD = str(_RELEASE.survey_year)
#: Years the release pins no vintage for: before its survey year and after
#: its base year.
_UNPINNED_EARLIER = min(_RELEASE.survey_year, _RELEASE.base_year) - 1
_UNPINNED_LATER = max(_RELEASE.survey_year, _RELEASE.base_year) + 1
_CLONE = ladder_clone_index_column("household")
#: The dense identity's lineage columns at their values on a base household:
#: no SPI support copy, CGT copy, incidence clone or residential arm.
_BASE_LINEAGE = {
    "household_support_channel": "frs",
    "household_support_clone_index": 0,
    "household_is_cgt_support_copy": False,
    "cgt_support_copy_index": 0,
    "household_is_capital_gains_clone": False,
    "household_is_cgt_residential_clone": False,
    "cgt_residential_clone_index": 0,
}


def _base_frame(households: int = 9, *, time_period: str = _PERIOD) -> Frame:
    """Invented FRS-shaped ids (test_support's ``base_id_tables``) without any
    lineage, at invented design weights."""

    tables = base_id_tables(households)
    household = tables.household[["household_id"]].copy()
    household["region"] = "WALES"
    return uk_national_frame(
        person=tables.person,
        benunit=tables.benunit,
        household=household,
        household_weights=np.random.default_rng(1931).uniform(
            400.0, 3000.0, households
        ),
        time_period=time_period,
    )


def _expanded(clones: int, households: int = 9, **kwargs: str) -> Frame:
    return expand_uk_geographic_clones(
        _base_frame(households, **kwargs), clones=clones
    ).frame


def _with_household(frame: Frame, household: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _stage(**overrides: object) -> SourceStageSpec:
    """A stand-in stage for G1b: no stage declares it until the spine wires it."""

    fields: dict[str, object] = {
        "stage": GEOGRAPHIC_IDENTITY_STAGE_NAME,
        "survey": "Invented test stage",
        "source": "Invented test stage",
        "grain": "household",
        "artifacts": (),
        "operations": (),
        "outputs": UK_GEOGRAPHIC_IDENTITY_OUTPUT_COLUMNS,
    }
    fields.update(overrides)
    return SourceStageSpec(**fields)


def _dense_keys(household: pd.DataFrame) -> list[str]:
    """The dense line's own identity kernel on these households, given the
    lineage a base household carries there and the full build's params."""

    dense = household.assign(**_BASE_LINEAGE)
    context = SimpleNamespace(
        tables={"household": dense},
        params={"source": "frs", "source_vintage": _RELEASE.vintage},
    )
    result = UKGeographyIdentityKernel().run(context)
    (keys,) = result.columns.values()
    return keys.tolist()


def _keys(frame: Frame) -> list[str]:
    return frame.table("household")[IDENTITY_COLUMN].tolist()


# --- the keys ----------------------------------------------------------------------


@pytest.mark.parametrize("clones", [1, 3])
def test_every_clone_keys_as_the_dense_identity_keys_that_clone(clones: int) -> None:
    expanded = _expanded(clones)
    keyed = key_uk_households_by_clone_path(expanded).frame
    assert _keys(keyed) == _dense_keys(expanded.table("household"))


def test_clone_zero_keys_on_its_source_and_clone_k_on_its_geographic_branch() -> None:
    expanded = _expanded(3)
    household = expanded.table("household")
    keys = _keys(key_uk_households_by_clone_path(expanded).frame)
    for key, source, clone in zip(
        keys,
        household[SOURCE_HOUSEHOLD_ID_COLUMN].tolist(),
        household[_CLONE].tolist(),
        strict=True,
    ):
        path = ((GEOGRAPHIC_SUPPORT_BRANCH, clone),) if clone else ()
        assert key == household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=_RELEASE.vintage,
            source_household_id=source,
            clone_path=path,
        )


def test_keys_are_unique_and_a_larger_k_keeps_every_key() -> None:
    small = _keys(key_uk_households_by_clone_path(_expanded(3)).frame)
    large = _keys(key_uk_households_by_clone_path(_expanded(7)).frame)
    assert len(set(large)) == len(large)
    # Clone-major: the first three clones of seven are the three clones.
    assert large[: len(small)] == small


def test_the_key_is_the_graphs_string_dtype_and_the_source_key_stays_object() -> None:
    keyed = key_uk_households_by_clone_path(_expanded(3)).frame
    household = keyed.table("household")
    assert household[IDENTITY_COLUMN].dtype == dtype_for_token("string")
    assert household[IDENTITY_COLUMN].dtype == pd.StringDtype(storage="python")
    # The legacy audit's rule: object columns holding text. The source key
    # is one, as the legacy plan's SPI support channel writes it; the
    # geography key is not.
    object_strings = {
        column
        for column in household.columns
        if household[column].dtype == object
        and household[column].map(lambda value: isinstance(value, str)).all()
    }
    assert SOURCE_HOUSEHOLD_KEY_COLUMN in object_strings
    assert IDENTITY_COLUMN not in object_strings


def test_the_step_adds_the_key_alone() -> None:
    expanded = _expanded(3)
    keyed = key_uk_households_by_clone_path(expanded).frame
    for entity in expanded.entities:
        before = expanded.table(entity)
        pd.testing.assert_frame_equal(
            keyed.table(entity)[list(before.columns)], before, check_exact=True
        )
    assert list(keyed.table("household").columns) == [
        *expanded.table("household").columns,
        IDENTITY_COLUMN,
    ]
    assert (
        keyed.weights_for("household").values.tobytes()
        == expanded.weights_for("household").values.tobytes()
    )
    assert keyed.weights_for("household").kind is expanded.weights_for("household").kind
    assert keyed.mass_log == expanded.mass_log
    assert dict(keyed.metadata) == dict(expanded.metadata)


def test_the_vintage_is_frs_release_json_s_for_the_source_year() -> None:
    assert dict(frs_vintages_by_source_year()) == {
        _RELEASE.survey_year: _RELEASE.vintage,
        _RELEASE.base_year: _RELEASE.vintage,
    }
    # Another pin, for the year before, keys a frame of that year with its
    # own vintage.
    earlier = replace(
        _RELEASE,
        survey_year=_RELEASE.survey_year - 1,
        vintage="invented_vintage",
    )
    expanded = _expanded(2, time_period=str(_RELEASE.survey_year - 1))
    result = key_uk_households_by_clone_path(expanded, release=earlier)
    household = expanded.table("household")
    assert _keys(result.frame) == [
        household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage="invented_vintage",
            source_household_id=source,
            clone_path=((GEOGRAPHIC_SUPPORT_BRANCH, clone),) if clone else (),
        )
        for source, clone in zip(
            household[SOURCE_HOUSEHOLD_ID_COLUMN].tolist(),
            household[_CLONE].tolist(),
            strict=True,
        )
    ]
    assert result.evidence()["source_vintages"] == ["invented_vintage"]


def test_a_base_year_after_the_survey_year_keys_with_the_release_s_vintage() -> None:
    """G1 writes source_year from the frame's time period, the release's base
    year, so a pin whose base year follows its survey year keys the base
    year's households with its vintage, as the dense identity does."""

    later = replace(
        _RELEASE,
        base_year=_RELEASE.survey_year + 1,
        time_period=str(_RELEASE.survey_year + 1),
    )
    assert dict(frs_vintages_by_source_year(later)) == {
        _RELEASE.survey_year: _RELEASE.vintage,
        _RELEASE.survey_year + 1: _RELEASE.vintage,
    }
    expanded = _expanded(3, time_period=later.time_period)
    result = key_uk_households_by_clone_path(expanded, release=later)
    assert _keys(result.frame) == _dense_keys(expanded.table("household"))
    assert result.evidence()["source_vintages"] == [_RELEASE.vintage]


def test_the_source_is_the_dense_identity_s() -> None:
    parameters = inspect.signature(append_uk_population_nodes).parameters
    assert parameters["identity_source"].default == GEOGRAPHIC_IDENTITY_SOURCE


def test_the_evidence_gives_rounded_sources_uniqueness_source_and_vintage() -> None:
    result = key_uk_households_by_clone_path(_expanded(3, households=9))
    assert result.evidence() == {
        "stage": GEOGRAPHIC_IDENTITY_STAGE_NAME,
        # Distinct source households, a household's clones counted once:
        # the same at every K.
        "source_households": rounded_count(9),
        "unique": True,
        "source": GEOGRAPHIC_IDENTITY_SOURCE,
        "source_vintages": [_RELEASE.vintage],
        "inputs": [
            SOURCE_HOUSEHOLD_ID_COLUMN,
            SOURCE_YEAR_COLUMN,
            ladder_clone_index_column("household"),
        ],
    }
    assert list(GEOGRAPHIC_IDENTITY_INPUTS) == result.evidence()["inputs"]
    assert (
        key_uk_households_by_clone_path(_expanded(1, households=9)).evidence()
        == result.evidence()
    )
    small = key_uk_households_by_clone_path(_expanded(3, households=2))
    assert small.evidence()["source_households"] == "<3"
    # The result's repr prints no count and no key.
    assert repr(small) == (
        f"UKGeographicIdentityResult(source_vintages=('{_RELEASE.vintage}',))"
    )


def test_the_transform_runs_the_step_and_records_its_evidence() -> None:
    expanded = _expanded(3)
    transform = UKGeographicIdentityStageTransform(_stage())
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    keyed = transform(expanded)
    direct = key_uk_households_by_clone_path(expanded)
    pd.testing.assert_frame_equal(
        keyed.table("household"), direct.frame.table("household"), check_exact=True
    )
    assert transform.checkpoint_metadata() == {"evidence": direct.evidence()}
    assert transform.output_columns() == (IDENTITY_COLUMN,)


# --- refusals ---------------------------------------------------------------------


def _assert_no_key_or_id(message: str, frame: Frame) -> None:
    assert "uk-household-v1" not in message
    for household_id in frame.table("household")["household_id"].tolist():
        if household_id > 1000:
            assert str(household_id) not in message


def test_a_year_the_release_pins_no_vintage_for_is_refused_by_count() -> None:
    other = _expanded(2, time_period=str(_UNPINNED_EARLIER))
    with pytest.raises(
        ValueError, match="^18 household\\(s\\) carry a source_year that frs_release"
    ):
        key_uk_households_by_clone_path(other)
    expanded = _expanded(3)
    household = expanded.table("household").copy()
    household.loc[household.index[[1, 4, 7, 20]], SOURCE_YEAR_COLUMN] = _UNPINNED_LATER
    with pytest.raises(
        ValueError, match="^4 household\\(s\\) carry a source_year"
    ) as error:
        key_uk_households_by_clone_path(_with_household(expanded, household))
    _assert_no_key_or_id(str(error.value), expanded)


def test_repeated_keys_are_refused_by_count() -> None:
    expanded = _expanded(2)
    household = expanded.table("household").copy()
    # Clone 1 loses its clone index: its keys repeat clone 0's.
    household[_CLONE] = np.zeros(len(household), dtype=np.int64)
    with pytest.raises(
        ValueError, match="^9 household\\(s\\) repeat an earlier household's geography"
    ) as error:
        key_uk_households_by_clone_path(_with_household(expanded, household))
    _assert_no_key_or_id(str(error.value), expanded)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (_CLONE, r"lacks the 'geographic_support' step's column\(s\)"),
        (SOURCE_YEAR_COLUMN, "reads household 'source_year', which the geographic"),
        (
            SOURCE_HOUSEHOLD_ID_COLUMN,
            "reads household 'source_household_id', which the geographic",
        ),
    ],
)
def test_a_frame_without_g1_s_columns_is_refused(column: str, message: str) -> None:
    expanded = _expanded(2)
    household = expanded.table("household").drop(columns=[column])
    with pytest.raises(ValueError, match=message) as error:
        key_uk_households_by_clone_path(_with_household(expanded, household))
    _assert_no_key_or_id(str(error.value), expanded)


@pytest.mark.parametrize("column", [SOURCE_HOUSEHOLD_ID_COLUMN, SOURCE_YEAR_COLUMN])
def test_a_source_column_that_is_not_int64_is_refused(column: str) -> None:
    expanded = _expanded(2)
    household = expanded.table("household").copy()
    household[column] = household[column].astype(np.int32)
    with pytest.raises(ValueError, match=f"reads an int64 '{column}'; it holds int32"):
        key_uk_households_by_clone_path(_with_household(expanded, household))


def test_a_frame_already_keyed_is_refused() -> None:
    keyed = key_uk_households_by_clone_path(_expanded(2)).frame
    with pytest.raises(ValueError, match="already carries 'geography_household_key'"):
        key_uk_households_by_clone_path(keyed)


def test_a_non_frame_is_refused() -> None:
    with pytest.raises(TypeError, match="keys a Frame"):
        key_uk_households_by_clone_path(_expanded(2).table("household"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage": "geographic_expand"}, "Expected stage 'geographic_identity'"),
        ({"grain": "person"}, "grain must be 'household'"),
        ({"outputs": ()}, "must declare outputs"),
        ({"rewrites": ("region",)}, "and no rewrites"),
    ],
)
def test_a_stage_that_is_not_g1b_s_is_refused(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        UKGeographicIdentityStageTransform(_stage(**overrides))
