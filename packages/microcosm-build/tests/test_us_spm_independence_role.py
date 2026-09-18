"""The SPM independence role as a build-stage input leaf.

The stage's derivation body is a call into the certified
``derive_spm_role_source``; these tests pin the stage's manifest and plan
wiring, every refusal the brief names (a missing source column, a unit with
more or fewer than one ``SPM_HEAD``, a unit left without a classified adult, a
person with no ASEC origin, a Census count that does not reconcile), the frame
integration and idempotence, the signal gate, the release coverage contract,
and — beside the hand-built battery — a seeded agreement sweep showing the
stage's role equals the derivation's and the raw rule's on random populations.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime import (
    US_DONORS,
    US_PUF_SUPPORT_STAGE_NAME,
    US_RELATIONSHIP_INPUTS_STAGE_NAME,
    US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_STAGE_NAME,
    US_STAGE_NAMES,
    derive_us_spm_independence_role_from_manifest,
    load_release_input_coverage_manifest,
    resolve_asec_spm_role_source_paths,
    us_spm_independence_role_signal_gate,
    us_spm_independence_role_stage_spec,
    us_spm_independence_role_summary,
    with_us_spm_independence_role,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    POST_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_independence_role import (
    US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND,
    US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
    US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY,
    US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY,
)
from microcosm.build.us_runtime.spm_role_source import (
    _OPTIONAL_RAW_CHECKS,
    _REQUIRED_RAW_CHECKS,
    ASEC_SPM_ROLE_SOURCES,
    NATIVE_SPM_ROLE,
    AsecSpmRoleSource,
    derive_spm_role_source,
    independent_minor_role,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

pytest.importorskip("tables")

_ROLE = NATIVE_SPM_ROLE
_INCOME_YEAR = 2024
_SOURCE_COLUMNS = (
    "PERIDNUM",
    "SPM_ID",
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
    "SPM_HAGE",
    "SPM_HEAD",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "SPM_NUMPER",
    "A_FAMTYP",
    "A_FAMREL",
    "A_SPOUSE",
    "PECOHAB",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic Census person files and the frames built from them
# ---------------------------------------------------------------------------


def _finish_units(rows: list[dict]) -> pd.DataFrame:
    """Fill the Census unit counts from the rows' own ages and roles."""

    source = pd.DataFrame(rows)
    role = independent_minor_role(source)
    adult = source.A_AGE.ge(18) | (source.A_AGE.ge(15) & role)
    units = source.assign(_adult=adult).groupby("SPM_ID")
    source["SPM_NUMADULTS"] = units["_adult"].transform("sum").astype(int)
    source["SPM_NUMPER"] = units["_adult"].transform("size").astype(int)
    source["SPM_NUMKIDS"] = source["SPM_NUMPER"] - source["SPM_NUMADULTS"]
    head_age = source.loc[source.SPM_HEAD.eq(1)].set_index("SPM_ID")["A_AGE"]
    source["SPM_HAGE"] = source["SPM_ID"].map(head_age).astype(int)
    source["PERIDNUM"] = [f"{index + 1:022}" for index in range(len(source))]
    return source[list(_SOURCE_COLUMNS)]


def _hand_built_source() -> pd.DataFrame:
    """Three Census units: a minor spouse is independent, a same-age child is not.

    Unit 100: a 17-year-old head, a 16-year-old spouse, a 10-year-old child.
    Unit 200: a 40-year-old head and a 16-year-old child.
    Unit 300: a 15-year-old living alone (an SPM head).
    """

    rows = [
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=1, A_LINENO=1, A_AGE=17, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=2, PECOHAB=0),
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=2, A_LINENO=2, A_AGE=16, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=2, A_SPOUSE=1, PECOHAB=0),
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=3, A_LINENO=3, A_AGE=10, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=3, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=200, PH_SEQ=2, P_SEQ=1, A_LINENO=1, A_AGE=40, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=200, PH_SEQ=2, P_SEQ=2, A_LINENO=2, A_AGE=16, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=3, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=300, PH_SEQ=3, P_SEQ=1, A_LINENO=1, A_AGE=15, SPM_HEAD=1, A_FAMTYP=4, A_FAMREL=1, A_SPOUSE=0, PECOHAB=0),
    ]  # fmt: skip
    return _finish_units(rows)


def _random_source(seed: int, n_units: int) -> pd.DataFrame:
    """A random Census person file with a realistic minority of teen heads."""

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for unit in range(1, n_units + 1):
        teen_headed = rng.random() < 0.02
        head_age = (
            int(rng.integers(15, 18)) if teen_headed else int(rng.integers(25, 71))
        )
        members = [
            dict(A_AGE=head_age, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=0),
        ]
        if not teen_headed and rng.random() < 0.6:
            members.append(
                dict(
                    A_AGE=int(np.clip(head_age + rng.integers(-5, 6), 18, 90)),
                    SPM_HEAD=0,
                    A_FAMTYP=1,
                    A_FAMREL=2,
                    A_SPOUSE=1,
                )
            )
            members[0]["A_SPOUSE"] = 2
        if not teen_headed:
            for _child in range(int(rng.integers(0, 4))):
                members.append(
                    dict(
                        A_AGE=int(rng.integers(0, 18)),
                        SPM_HEAD=0,
                        A_FAMTYP=1,
                        A_FAMREL=3,
                        A_SPOUSE=0,
                    )
                )
        for line, member in enumerate(members, start=1):
            rows.append(
                dict(
                    SPM_ID=unit,
                    PH_SEQ=unit,
                    P_SEQ=line,
                    A_LINENO=line,
                    PECOHAB=0,
                    **member,
                )
            )
    return _finish_units(rows)


def _write_source(
    tmp_path: Path, source: pd.DataFrame
) -> tuple[Path, AsecSpmRoleSource]:
    path = tmp_path / "pppub25.csv"
    source.to_csv(path, index=False)
    pin = AsecSpmRoleSource(
        income_year=_INCOME_YEAR,
        survey_year=_INCOME_YEAR + 1,
        csv_sha256=_digest(path),
        csv_size_bytes=path.stat().st_size,
        persons=int(len(source)),
        units=int(source.SPM_ID.nunique()),
        official_archive_url="https://example.invalid/fixture.zip",
        archive_sha256="a" * 64,
        member="pppub25.csv",
    )
    return path, pin


def _person_table(
    source: pd.DataFrame, *, clone_units: tuple[int, ...] = ()
) -> pd.DataFrame:
    """The pooled person table a base carries for ``source``, plus optional clones.

    Mirrors what ``asec_pool`` and cloning leave on the frame: frozen-vintage
    identity (``source_year``, ``source_household_id``, ``source_person_id``,
    ``source_row_id``), the raw age/count fields, no ``SPM_HEAD`` (no frozen
    vintage carries it) and ``A_FAMTYP``/``A_FAMREL`` null for some rows.
    """

    native = source.copy()
    native["source_row_id"] = np.arange(len(native), dtype=np.int64)
    native["clone"] = 0
    pieces = [native]
    for index, unit in enumerate(clone_units, start=1):
        clone = native.loc[native.SPM_ID.eq(unit)].copy()
        clone["clone"] = index
        pieces.append(clone)
    person = pd.concat(pieces, ignore_index=True)
    person["source_year"] = _INCOME_YEAR
    person["source_person_id"] = person["PERIDNUM"]
    person["source_household_id"] = person["PH_SEQ"]
    person["person_id"] = np.arange(1001, 1001 + len(person), dtype=np.int64)
    unit_codes = pd.factorize(
        pd.MultiIndex.from_arrays([person["clone"], person["SPM_ID"]]), sort=True
    )[0]
    person["person_spm_unit_id"] = (unit_codes + 10).astype(np.int64)
    household_codes = pd.factorize(
        pd.MultiIndex.from_arrays([person["clone"], person["PH_SEQ"]]), sort=True
    )[0]
    person["person_household_id"] = (household_codes + 1).astype(np.int64)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(len(person), dtype=np.int64) + 4_000
    person["age"] = person["A_AGE"].astype(float)
    person["SPM_ID"] = pd.factorize(person["SPM_ID"], sort=True)[0] + 1
    person = person.drop(columns=["SPM_HEAD", "clone"])
    person["A_FAMREL"] = person["A_FAMREL"].astype(float)
    person.loc[person.index[:2], "A_FAMREL"] = np.nan
    person["A_FAMTYP"] = person["A_FAMTYP"].astype(float)
    person.loc[person.index[:2], "A_FAMTYP"] = np.nan
    return person


def _frame(person: pd.DataFrame, weights: np.ndarray | None = None) -> Frame:
    households = np.sort(person["person_household_id"].unique())
    tables = {
        "person": person.reset_index(drop=True),
        "household": pd.DataFrame({"household_id": households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": households + 1_000}),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.sort(person["person_spm_unit_id"].unique())}
        ),
        "family": pd.DataFrame({"family_id": households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    if weights is None:
        weights = np.full(len(households), 100.0)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _run(frame: Frame, path: Path, pin: AsecSpmRoleSource) -> Frame:
    return with_us_spm_independence_role(
        frame,
        seed=0,
        time_period=_INCOME_YEAR,
        asec_spm_role_source_paths={_INCOME_YEAR: path},
        source_pins={_INCOME_YEAR: pin},
    )


def _operation():
    return next(
        operation
        for operation in us_spm_independence_role_stage_spec().operations
        if operation.kind == US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND
    )


def _context(frame: Frame, path: Path, pin: AsecSpmRoleSource) -> SourceRuntimeContext:
    return SourceRuntimeContext(
        config=SourceRuntimeConfig(
            seed=0,
            target_year=_INCOME_YEAR,
            extra={
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY: {_INCOME_YEAR: path},
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY: {_INCOME_YEAR: pin},
            },
        ),
        tables={"spm_unit": frame.table("spm_unit")},
    )


@pytest.fixture
def population(tmp_path: Path):
    source = _hand_built_source()
    path, pin = _write_source(tmp_path, source)
    frame = _frame(_person_table(source, clone_units=(100,)))
    return source, path, pin, frame


# ---------------------------------------------------------------------------
# Manifest, plan and shared constants
# ---------------------------------------------------------------------------


class TestManifestAndPlan:
    def test_stage_spec_pins_the_source_rule(self) -> None:
        spec = us_spm_independence_role_stage_spec()

        assert spec.stage == US_SPM_INDEPENDENCE_ROLE_STAGE_NAME
        assert (
            tuple(spec.outputs) == US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS == (_ROLE,)
        )
        assert US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS == (_ROLE,)
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND,
        ]
        assert (
            "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})" in spec.notes
        )
        assert "derive_spm_role_source" in spec.notes
        assert "exactly one SPM_HEAD" in spec.notes
        assert "no SPM unit is left without a classified adult" in spec.notes

    def test_handler_and_plan_are_wired_between_relationships_and_puf_support(
        self,
    ) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers[US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND]
            is derive_us_spm_independence_role_from_manifest
        )
        assert US_SPM_INDEPENDENCE_ROLE_STAGE_NAME in US_DONORS
        assert (
            US_DONORS[US_SPM_INDEPENDENCE_ROLE_STAGE_NAME].survey == "Census CPS ASEC"
        )
        order = US_STAGE_NAMES.index
        assert (
            order(US_RELATIONSHIP_INPUTS_STAGE_NAME)
            < order(US_SPM_INDEPENDENCE_ROLE_STAGE_NAME)
            < order(US_PUF_SUPPORT_STAGE_NAME)
        )

    def test_required_columns_are_the_derivations_own(self) -> None:
        """The stage restates only the frame identity; the raw tail is imported."""

        assert US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS == (
            "person_id",
            "person_spm_unit_id",
            "source_year",
            "PERIDNUM",
            "age",
            "source_household_id",
            "source_person_id",
            "source_row_id",
            *_REQUIRED_RAW_CHECKS,
        )
        assert US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS == _OPTIONAL_RAW_CHECKS

    @pytest.mark.parametrize("column", US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS)
    def test_the_derivation_itself_requires_each_column(
        self, tmp_path: Path, column: str
    ) -> None:
        """Consistency: every column the stage names, derive_spm_role_source refuses."""

        source = _hand_built_source()
        path, pin = _write_source(tmp_path, source)
        person = _person_table(source).drop(columns=[column])
        parent = tmp_path / "parent.h5"
        person.to_hdf(parent, key="person")
        pd.DataFrame(
            {
                "spm_unit_id": np.sort(
                    _person_table(source)["person_spm_unit_id"].unique()
                )
            }
        ).to_hdf(parent, key="spm_unit")
        with pytest.raises(ValueError, match=f"missing required columns.*{column}"):
            derive_spm_role_source(
                parent,
                {_INCOME_YEAR: path},
                expected_parent_sha256=_digest(parent),
                source_pins={_INCOME_YEAR: pin},
            )

    def test_resolver_refuses_unpinned_years_and_keeps_explicit_paths(
        self, tmp_path: Path
    ) -> None:
        explicit = tmp_path / "pppub25.csv"
        resolved = resolve_asec_spm_role_source_paths(
            {_INCOME_YEAR: explicit}, income_years=(_INCOME_YEAR,)
        )
        assert resolved == {_INCOME_YEAR: explicit}
        with pytest.raises(ValueError, match="pinned income years"):
            resolve_asec_spm_role_source_paths({1999: explicit}, income_years=(1999,))
        with pytest.raises(ValueError, match="have no pinned ASEC SPM role source"):
            resolve_asec_spm_role_source_paths(None, income_years=(1999,))
        assert set(ASEC_SPM_ROLE_SOURCES) == {2022, 2023, 2024}


# ---------------------------------------------------------------------------
# The derivation through the stage: agreement and refusals
# ---------------------------------------------------------------------------


class TestDerivation:
    def test_roles_equal_the_certified_derivation_and_the_raw_rule(
        self, population
    ) -> None:
        source, path, pin, frame = population

        result = _run(frame, path, pin)

        person = result.table("person")
        role = person[_ROLE].to_numpy()
        assert person[_ROLE].dtype == bool
        # The raw rule on the source rows, joined by PERIDNUM (clones repeat it).
        expected = (
            person["PERIDNUM"]
            .map(
                dict(zip(source.PERIDNUM, independent_minor_role(source), strict=True))
            )
            .to_numpy(dtype=bool)
        )
        assert np.array_equal(role, expected)
        assert role.tolist() == [
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ]
        # Unit 100's clone (native unit 40) carries the same roles as the native.
        assert person.loc[person.person_spm_unit_id.eq(13), _ROLE].tolist() == [
            True,
            True,
            False,
        ]
        # The stage is the certified derivation, run on a projection of the frame.
        parent = path.parent / "direct.h5"
        frame.table("person").to_hdf(parent, key="person")
        frame.table("spm_unit").to_hdf(parent, key="spm_unit")
        direct = derive_spm_role_source(
            parent,
            {_INCOME_YEAR: path},
            expected_parent_sha256=_digest(parent),
            source_pins={_INCOME_YEAR: pin},
        )
        assert np.array_equal(direct.role, role)

    def test_provenance_rides_on_the_frame(self, population) -> None:
        _source, path, pin, frame = population

        result = _run(frame, path, pin)

        provenance = result.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY]
        assert provenance["unmatched_persons"] == 0
        assert provenance["adult_child_person_count_mismatch_units"] == 0
        assert provenance["native_spm_units"] == 4
        assert provenance["total_source_units"] == 3
        assert provenance["minor_only_units_resolved"] == 3
        assert provenance["independent_minor_persons"] == 5
        assert provenance["weights_used"] is False
        assert provenance["ages_changed"] is False
        assert len(provenance["frame_projection_sha256"]) == 64
        assert provenance["dataset_sha256"] == provenance["frame_projection_sha256"]
        assert set(provenance["frame_projection_columns"]) == set(
            US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS
        ) | {"A_FAMTYP", "A_FAMREL", "A_SPOUSE", "PECOHAB"}

    def test_refuses_a_unit_with_two_heads(self, tmp_path: Path) -> None:
        source = _hand_built_source()
        source.loc[source.index[1], "SPM_HEAD"] = 1
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="exactly one SPM head per unit"):
            _run(frame, path, pin)

    def test_refuses_a_unit_with_no_head(self, tmp_path: Path) -> None:
        source = _hand_built_source()
        source.loc[source.index[3], "SPM_HEAD"] = 0
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="exactly one SPM head per unit"):
            _run(frame, path, pin)

    def test_refuses_a_unit_left_without_a_classified_adult(
        self, tmp_path: Path
    ) -> None:
        """A 14-year-old head: the role is True but the age gate leaves no adult."""

        source = _hand_built_source()
        source.loc[source.index[5], "A_AGE"] = 14
        source.loc[source.index[5], "SPM_HAGE"] = 14
        source.loc[source.index[5], ["SPM_NUMADULTS", "SPM_NUMKIDS"]] = [0, 1]
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="unresolved zero-adult SPM unit"):
            _run(frame, path, pin)

    def test_refuses_a_person_with_no_asec_origin(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").copy()
        person.loc[person.index[0], ["PERIDNUM", "source_person_id"]] = "9" * 22
        with pytest.raises(SourceRuntimeError, match="unmatched parent persons"):
            _run(_frame(person), path, pin)

    def test_refuses_a_census_count_that_does_not_reconcile(
        self, tmp_path: Path
    ) -> None:
        source = _hand_built_source()
        source.loc[source.SPM_ID.eq(200), "SPM_NUMADULTS"] = 2
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(
            SourceRuntimeError, match="adult/child/person count reconciliation"
        ):
            _run(frame, path, pin)

    def test_refuses_a_native_unit_mixing_two_source_units(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").copy()
        # Move unit 200's child into native unit 10 (source unit 100).
        person.loc[person.index[4], "person_spm_unit_id"] = 10
        with pytest.raises(SourceRuntimeError, match="combines distinct source units"):
            _run(_frame(person), path, pin)

    @pytest.mark.parametrize("column", US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS)
    def test_missing_source_column_is_named(self, population, column: str) -> None:
        """The handler names the column; the frame wrapper refuses it first when
        the frame cannot even be built or its income years cannot be read."""

        _source, path, pin, frame = population
        person = frame.table("person").drop(columns=[column])
        with pytest.raises(SourceRuntimeError, match=column):
            derive_us_spm_independence_role_from_manifest(
                person, _operation(), _context(frame, path, pin)
            )
        if column in ("person_id", "person_spm_unit_id"):
            return  # the Frame schema refuses a person table without these
        if column == "source_year":
            with pytest.raises(ValueError, match=column):
                _run(_frame(person), path, pin)
            return
        with pytest.raises(SourceRuntimeError, match=column):
            _run(_frame(person), path, pin)

    def test_handler_refuses_wrong_operation_missing_table_context_and_paths(
        self, population
    ) -> None:
        _source, path, pin, frame = population
        person = frame.table("person")
        wrong = SourceStageSpec.from_mapping(
            {
                "stage": "test",
                "survey": "test",
                "source": "https://example.com",
                "grain": "person",
                "operations": [{"kind": "derive"}],
                "outputs": [_ROLE],
            }
        ).operations[0]
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_spm_independence_role_from_manifest(
                person, wrong, _context(frame, path, pin)
            )
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_spm_independence_role_from_manifest(
                None, _operation(), _context(frame, path, pin)
            )
        with pytest.raises(SourceRuntimeError, match="runtime context"):
            derive_us_spm_independence_role_from_manifest(person, _operation(), None)
        no_spm = SourceRuntimeContext(
            config=_context(frame, path, pin).config, tables={}
        )
        with pytest.raises(SourceRuntimeError, match="spm_unit"):
            derive_us_spm_independence_role_from_manifest(person, _operation(), no_spm)
        no_paths = SourceRuntimeContext(
            config=SourceRuntimeConfig(seed=0, target_year=_INCOME_YEAR),
            tables={"spm_unit": frame.table("spm_unit")},
        )
        with pytest.raises(SourceRuntimeError, match="CSV paths"):
            derive_us_spm_independence_role_from_manifest(
                person, _operation(), no_paths
            )

    def test_refuses_an_unpinned_csv(self, population) -> None:
        _source, path, pin, frame = population
        with pytest.raises(SourceRuntimeError, match="CSV SHA-256"):
            _run(frame, path, replace(pin, csv_sha256="0" * 64))


# ---------------------------------------------------------------------------
# Frame integration, summary and gate
# ---------------------------------------------------------------------------


class TestFrameAndGate:
    def test_idempotent_and_leaves_everything_else_untouched(self, population) -> None:
        _source, path, pin, frame = population

        result = _run(frame, path, pin)

        assert _ROLE not in frame.table("person")
        assert set(result.table("person").columns) == set(
            frame.table("person").columns
        ) | {_ROLE}
        for entity in frame.entities:
            if entity == "person":
                continue
            assert result.table(entity).equals(frame.table(entity))
        assert np.array_equal(
            result.table("person")["person_spm_unit_id"].to_numpy(),
            frame.table("person")["person_spm_unit_id"].to_numpy(),
        )
        assert with_us_spm_independence_role(result, seed=0, time_period=2024) is result

    def test_requires_the_us_schema_and_source_year(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").drop(columns=["source_year"])
        with pytest.raises(ValueError, match="source_year"):
            with_us_spm_independence_role(
                _frame(person),
                seed=0,
                time_period=2024,
                asec_spm_role_source_paths={_INCOME_YEAR: path},
                source_pins={_INCOME_YEAR: pin},
            )
        with pytest.raises(ValueError, match="explicit CSV paths"):
            with_us_spm_independence_role(
                frame, seed=0, time_period=2024, source_pins={_INCOME_YEAR: pin}
            )

    def test_gate_passes_on_a_realistic_population_and_reports_the_composition(
        self, tmp_path: Path
    ) -> None:
        source = _random_source(seed=7, n_units=300)
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))

        result = _run(frame, path, pin)
        gate = us_spm_independence_role_signal_gate(result)

        assert gate.passed, gate.failures
        summary = us_spm_independence_role_summary(result)
        assert summary["spm_composition"]["status"] == "PASS"
        assert summary["spm_composition"]["role_source"] == "source_column"
        assert summary["spm_composition"]["n_units_without_classified_adult"] == 0
        assert summary["independent_minor_persons"] == int(
            (source.A_AGE.between(15, 17) & source.SPM_HEAD.eq(1)).sum()
        )
        assert 0.40 <= summary["role_share"] <= 0.75
        assert 0.003 <= summary["minor_role_share"] <= 0.06
        assert summary["derivation"]["unmatched_persons"] == 0
        assert check_spm_composition(result).status == "PASS"

    def test_gate_names_a_missing_column(self, population) -> None:
        _source, _path, _pin, frame = population
        gate = us_spm_independence_role_signal_gate(frame)
        assert not gate.passed
        assert gate.details["missing"] == [_ROLE]

    def test_gate_fails_a_frame_whose_role_leaves_a_unit_unclassified(
        self, population
    ) -> None:
        """A stored False on the only teen of a minor-only unit: the engine refuses."""

        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        person = result.table("person").copy()
        person.loc[person.person_spm_unit_id.eq(12), _ROLE] = False
        gate = us_spm_independence_role_signal_gate(_frame(person))

        assert not gate.passed
        assert any("no classified adult" in failure for failure in gate.failures)
        assert any("SPM_COMPOSITION_REQUIRED" in failure for failure in gate.failures)

    def test_gate_fails_a_degenerate_or_missing_valued_role(self, population) -> None:
        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        person = result.table("person").copy()
        person[_ROLE] = True
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert any("degenerate" in failure for failure in gate.failures)

        person = result.table("person").copy()
        person[_ROLE] = person[_ROLE].astype(object)
        person.loc[person.index[0], _ROLE] = None
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert any("missing for 1 person" in failure for failure in gate.failures)


# ---------------------------------------------------------------------------
# Seeded agreement sweep beside the hand-built battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_seeded_agreement_with_the_raw_rule_and_the_derivation(
    tmp_path: Path, seed: int
) -> None:
    """On random populations the stage, the raw rule and the derivation agree.

    The stage adds nothing to the rule: for every person its role equals
    ``independent_minor_role`` evaluated on that person's Census row, and
    equals ``derive_spm_role_source`` run directly on an H5 of the same frame;
    afterwards no unit is left without a classified adult, and the counts the
    derivation reconciled are the frame's own.
    """

    rng = np.random.default_rng(seed)
    source = _random_source(seed=seed, n_units=int(rng.integers(40, 160)))
    clone_units = tuple(
        int(unit)
        for unit in rng.choice(
            source.SPM_ID.unique(), size=int(rng.integers(0, 6)), replace=False
        )
    )
    path, pin = _write_source(tmp_path, source)
    person = _person_table(source, clone_units=clone_units)
    frame = _frame(
        person,
        weights=rng.uniform(
            10.0, 500.0, size=int(person["person_household_id"].nunique())
        ),
    )

    result = _run(frame, path, pin)

    person = result.table("person")
    role = person[_ROLE].to_numpy(dtype=bool)
    by_source = dict(zip(source.PERIDNUM, independent_minor_role(source), strict=True))
    assert np.array_equal(role, person["PERIDNUM"].map(by_source).to_numpy(dtype=bool))

    parent = tmp_path / f"direct-{seed}.h5"
    frame.table("person").to_hdf(parent, key="person")
    frame.table("spm_unit").to_hdf(parent, key="spm_unit")
    direct = derive_spm_role_source(
        parent,
        {_INCOME_YEAR: path},
        expected_parent_sha256=_digest(parent),
        source_pins={_INCOME_YEAR: pin},
    )
    assert np.array_equal(direct.role, role)

    composition = check_spm_composition(result)
    assert composition.status == "PASS", composition.failures
    provenance = result.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY]
    assert provenance["persons_joined"] == len(person)
    assert provenance["native_spm_units"] == result.n("spm_unit")
    assert provenance["unmatched_persons"] == 0
    assert provenance["adult_child_person_count_mismatch_units"] == 0
    teens = person["age"].between(15, 17).to_numpy()
    assert provenance["independent_minor_persons"] == int((role & teens).sum())


# ---------------------------------------------------------------------------
# Release coverage and export contract
# ---------------------------------------------------------------------------


class TestCoverageAndExport:
    def test_role_is_a_hard_release_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert _ROLE in POST_REFERENCE_ECPS_REQUIRED_INPUTS
        assert _ROLE in manifest.required_columns
        assert _ROLE not in manifest.reviewed_exclusions
        assert _ROLE in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS

    @pytest.mark.requires_us
    def test_engine_adapter_persists_the_role_as_an_input_leaf(
        self, population, tmp_path: Path
    ) -> None:
        """Both adapter paths classify the role by the engine's own declaration."""

        import policyengine_us.spm as spm
        from policyengine_us.data import USSingleYearDataset

        from microcosm.frame.adapters.policyengine_us import (
            PolicyEngineUSEngine,
            PolicyEngineUSVariableMetadataIndex,
        )

        assert spm.DATASET_SOURCE_INPUTS == frozenset({_ROLE})
        engine = PolicyEngineUSEngine()
        index = PolicyEngineUSVariableMetadataIndex()
        assert _ROLE in engine.variables()
        assert _ROLE in index.variables()
        assert engine.formula_owned_outputs([_ROLE, "spm_measurement_adults"]) == {
            "spm_measurement_adults"
        }
        assert index.formula_owned_outputs([_ROLE, "spm_measurement_adults"]) == {
            "spm_measurement_adults"
        }
        assert engine.default_values([_ROLE]) == {_ROLE: False}

        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        export = result.table("person")[
            [
                "person_id",
                "person_household_id",
                "person_tax_unit_id",
                "person_spm_unit_id",
                "person_family_id",
                "person_marital_unit_id",
                "age",
                _ROLE,
            ]
        ]
        tables = {entity: result.table(entity) for entity in result.entities}
        tables["person"] = export
        exportable = Frame(
            tables,
            US_SCHEMA,
            {"household": result.weights_for("household")},
        )
        destination = tmp_path / "role.h5"
        engine.write_dataset(exportable, destination, period=_INCOME_YEAR)
        written = USSingleYearDataset(file_path=str(destination)).person
        assert written[_ROLE].tolist() == export[_ROLE].tolist()
