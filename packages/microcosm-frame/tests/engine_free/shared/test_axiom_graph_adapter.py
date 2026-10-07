"""Axiom adapter surfaces for graph nodes: periods, nesting, dtypes, engine refs.

Everything above the ``needs_engine`` classes runs without
``axiom_rules_engine``: period and nesting checks run before the engine is
touched, the graph dtype cast is a pure function, ``axiom_engine_ref`` reads
only bytes on disk, and materialization is exercised through a recording
stand-in for the dense surface that returns exactly the array types the native
extension returns (int8 judgment codes, float64 decimals, int64 integers, bool,
and Python string lists for text and dates).

The ``needs_engine`` classes run the real engine on the ``rulespec-zz``
fixture (the existing skip pattern): a graph-typed ``simulate.rules@1`` node
in a FILTER-free graph, the ``tax_year`` period label, family-level judgment
codes, and ``simulate.rules_by_ref@1`` with two Axiom bindings in one run.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame import ExportContract, Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import (
    BE_SCHEMA,
    NZ_NESTING,
    NZ_SCHEMA,
    AxiomEngine,
    AxiomPeriod,
    _graph_output,
    assert_no_relations,
    axiom_engine_ref,
    rulespec_tree_digest,
)
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.frame.rules_kernels import SimulateRulesByRefKernel
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.keys import _directory_identity, source_content_key
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-frame")

_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False

needs_engine = pytest.mark.skipif(
    not _DENSE_AVAILABLE,
    reason="axiom_rules_engine (with the dense native extension) is not installed",
)
needs_tables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="pytables (microcosm-frame[axiom]) is not installed",
)

FIXTURE_RULESPEC_ROOT = _TEST_PATHS.tests / "fixtures" / "rulespec-zz"
FIXTURE_MODULE = FIXTURE_RULESPEC_ROOT / "zz/policies/tests/axiom_toy_country.yaml"
FAMILY_FIXTURE_MODULE = (
    FIXTURE_RULESPEC_ROOT / "zz/policies/tests/axiom_toy_family.yaml"
)
FIXTURE_RULESPEC_ROOTS = (FIXTURE_RULESPEC_ROOT,)

TAX_YEAR = AxiomPeriod(start="2026-04-01", end="2027-03-31", kind="tax_year")
ENGINE_COMMIT = "a" * 40
WHEEL_SHA256 = "b" * 64
RULESPEC_COMMIT = "c" * 40

# A full override of every git setting a test repository depends on, so the
# developer's global configuration (hooks, signing) never reaches it.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=microcosm-test",
            "-c",
            "user.email=microcosm-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=_GIT_ENV,
    )
    return completed.stdout.strip()


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------


def _nz_frame(
    person_households: list[int],
    person_families: list[int],
    household_weights: list[float],
    *,
    family_columns: Mapping[str, list[object]] | None = None,
    person_columns: Mapping[str, list[object]] | None = None,
) -> Frame:
    """An NZ_SCHEMA frame with household-only design weights."""

    count = len(person_households)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype=np.int64),
            "person_household_id": np.asarray(person_households, dtype=np.int64),
            "person_family_id": np.asarray(person_families, dtype=np.int64),
            **(person_columns or {}),
        }
    )
    household = pd.DataFrame(
        {"household_id": np.unique(np.asarray(person_households, dtype=np.int64))}
    )
    family = pd.DataFrame(
        {
            "family_id": np.unique(np.asarray(person_families, dtype=np.int64)),
            **(family_columns or {}),
        }
    )
    return Frame(
        {"person": person, "household": household, "family": family},
        NZ_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(household_weights, dtype=np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


@st.composite
def nz_populations(draw: st.DrawFn) -> dict[str, object]:
    """A nested NZ population, optionally with one family split across households.

    Households hold one to three families and families one to three persons,
    so every family nests in exactly one household. When ``split`` is drawn,
    one person of a family with at least two members moves to another
    household while keeping the family, and that family then spans two
    households. Weights are drawn equal or distinct so a test can show weight
    agreement never hides the split.
    """

    household_count = draw(st.integers(min_value=2, max_value=5))
    person_households: list[int] = []
    person_families: list[int] = []
    family_household: dict[int, int] = {}
    next_family = 10
    for household in range(1, household_count + 1):
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            family = next_family
            next_family += 1
            family_household[family] = household
            for _ in range(draw(st.integers(min_value=1, max_value=3))):
                person_households.append(household)
                person_families.append(family)
    splittable = [
        family for family in family_household if person_families.count(family) >= 2
    ]
    split = bool(splittable) and draw(st.booleans())
    if split:
        family = draw(st.sampled_from(splittable))
        position = person_families.index(family)
        others = [
            household
            for household in range(1, household_count + 1)
            if household != family_household[family]
        ]
        person_households[position] = draw(st.sampled_from(others))
    equal_weights = draw(st.booleans())
    weights = [
        250.0 if equal_weights else float(100 + 37 * household)
        for household in range(1, household_count + 1)
    ]
    return {
        "person_households": person_households,
        "person_families": person_families,
        "household_weights": weights,
        "split": split,
    }


# ----------------------------------------------------------------------
# A recording stand-in for the dense surface
# ----------------------------------------------------------------------


class _Metadata:
    """The authoring metadata fields the adapter reads."""

    def __init__(self, name: str, entity: str, dtype: str, period: str = "Year"):
        self.name = name
        self.entity = entity
        self.dtype = dtype
        self.period = period


class _Relation:
    def __init__(self, name: str) -> None:
        self.name = name


_FAKE_METADATA = (
    _Metadata("person_benefit", "Person", "decimal"),
    _Metadata("family_assets_ok", "Family", "judgment"),
    _Metadata("family_assistance", "Family", "decimal"),
    _Metadata("family_band", "Family", "integer"),
    _Metadata("family_is_large", "Family", "bool"),
    _Metadata("family_category", "Family", "text"),
    _Metadata("family_review_date", "Family", "date"),
)


class _FakeProgram:
    """One entity's dense program; outputs mirror the native return types."""

    def __init__(self, entity: str, relations: tuple[_Relation, ...] = ()) -> None:
        self.entity = entity
        self.root_inputs = ["family_rent", "person_income"]
        self.relations = list(relations)
        self.derived_metadata = list(_FAKE_METADATA)
        self.calls: list[dict[str, object]] = []

    def execute(self, *, period_kind, start, end, inputs, outputs):
        self.calls.append(
            {
                "period_kind": period_kind,
                "start": start,
                "end": end,
                "inputs": sorted(inputs),
                "outputs": list(outputs),
            }
        )
        rows = len(next(iter(inputs.values())))
        codes = np.resize(np.asarray([1, -1, 0], dtype=np.int8), rows)
        available = {
            "person_benefit": np.full(rows, 12.5),
            "family_assets_ok": codes,
            "family_assistance": np.linspace(0.0, 70.25, rows),
            "family_band": np.arange(rows, dtype=np.int64),
            "family_is_large": codes > 0,
            "family_category": ["small"] * rows,
            "family_review_date": ["2026-04-01"] * rows,
        }
        return {
            "row_count": rows,
            "outputs": {name: available[name] for name in outputs},
        }

    execute_f64 = execute


def _fake_engine(
    adapter: AxiomEngine,
    monkeypatch: pytest.MonkeyPatch,
    *,
    relations: tuple[_Relation, ...] = (),
) -> dict[str, _FakeProgram]:
    """Swap the adapter's engine import for the recording stand-in."""

    programs: dict[str, _FakeProgram] = {}

    class _Compiled:
        @classmethod
        def from_file(cls, path, *, rulespec_roots, entity):
            if entity not in {"Person", "Family"}:
                raise ValueError(
                    "dense compilation could not find derived outputs for entity "
                    f"`{entity}`"
                )
            program = _FakeProgram(entity, relations)
            programs[entity] = program
            return program

    class _Engine:
        CompiledDenseProgram = _Compiled

    monkeypatch.setattr(adapter, "_import_engine", lambda: _Engine)
    return programs


def _fake_frame() -> Frame:
    return _nz_frame(
        [1, 1, 2, 2, 2],
        [10, 10, 20, 21, 21],
        [300.0, 500.0],
        family_columns={"family_rent": [320.0, 0.0, 180.0]},
        person_columns={"person_income": [1.0, 2.0, 3.0, 4.0, 5.0]},
    )


def _nz_adapter(**kwargs) -> AxiomEngine:
    return AxiomEngine(
        FIXTURE_MODULE,
        schema=NZ_SCHEMA,
        rulespec_roots=FIXTURE_RULESPEC_ROOTS,
        **kwargs,
    )


# ----------------------------------------------------------------------
# NZ schema
# ----------------------------------------------------------------------


class TestNzSchema:
    def test_nz_schema_has_households_and_families(self) -> None:
        assert NZ_SCHEMA.entities == ("person", "household", "family")
        assert dict(NZ_NESTING) == {"family": "household"}

    def test_nz_nesting_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            NZ_NESTING["family"] = "person"  # type: ignore[index]

    @settings(max_examples=60, deadline=None)
    @given(population=nz_populations())
    def test_household_weights_resolve_onto_families(self, population) -> None:
        """Invariant (layout half): families inherit their household's weight.

        With household-only weights, every family's effective weight is its
        one household's weight, and the adapter's export tables persist
        ``household_weight`` alone. The enforcing half, an export refusing any
        other explicit weight, is
        ``test_property_only_household_weights_reach_an_export``.
        """

        if population["split"]:
            return
        frame = _nz_frame(
            population["person_households"],
            population["person_families"],
            population["household_weights"],
        )
        assert frame.weighted_entities == ("household",)
        household_weight = dict(
            zip(
                frame.table("household")["household_id"],
                population["household_weights"],
                strict=True,
            )
        )
        person = frame.table("person")
        family_household = person.groupby("person_family_id")[
            "person_household_id"
        ].first()
        expected = [
            household_weight[family_household[family]]
            for family in frame.table("family")["family_id"]
        ]
        np.testing.assert_array_equal(
            frame.resolve_weights("family").values, np.asarray(expected)
        )
        tables = _nz_adapter(nesting=NZ_NESTING)._engine_tables(frame)
        assert "household_weight" in tables["household"].columns
        assert "family_weight" not in tables["family"].columns
        assert "person_weight" not in tables["person"].columns


#: The New Zealand export rule #821 carried: weights persist on households
#: only, so the person and family weight columns are forbidden.
_NZ_WEIGHT_CONTRACT = ExportContract(
    required=(),
    forbidden=("person_weight", "family_weight"),
    optional=(),
    formula_owned_excluded=(),
)


class TestExportWeights:
    @settings(max_examples=60, deadline=None)
    @given(
        population=nz_populations(),
        extra=st.sets(st.sampled_from(["person", "family"]), min_size=1),
    )
    def test_property_only_household_weights_reach_an_export(
        self, tmp_path_factory, population, extra
    ) -> None:
        """Invariant: only households carry explicit weights in an NZ export.

        The executor hands a kernel resolved weights for every entity it
        projects (``G/executor.py`` builds ``KernelContext.weights`` for each
        one), so ``materialize`` cannot refuse explicit family weights. The
        export can: with the NZ contract, any explicit person or family weight
        blocks the write, and nothing is written.
        """

        if population["split"]:
            return
        base = _nz_frame(
            population["person_households"],
            population["person_families"],
            population["household_weights"],
        )
        weights = {"household": base.weights_for("household")}
        for entity in extra:
            weights[entity] = Weights(
                values=np.full(base.n(entity), 7.0), kind=WeightKind.DESIGN
            )
        frame = Frame(
            {entity: base.table(entity) for entity in NZ_SCHEMA.entities},
            NZ_SCHEMA,
            weights,
        )
        adapter = _nz_adapter(nesting=NZ_NESTING, contract=_NZ_WEIGHT_CONTRACT)
        forbidden = sorted(f"{entity}_weight" for entity in extra)
        with pytest.MonkeyPatch.context() as monkeypatch:
            _fake_engine(adapter, monkeypatch)
            path = tmp_path_factory.mktemp("export") / "weighted.h5"
            with pytest.raises(
                ValueError,
                match=f"forbidden column\\(s\\) present: {re.escape(repr(forbidden))}",
            ):
                adapter.write_dataset(frame, path, 2026)
            assert not path.exists()

    @needs_tables
    def test_a_household_weighted_frame_exports_household_weight_alone(
        self, tmp_path, monkeypatch
    ) -> None:
        from microcosm.frame.adapters.axiom import AxiomEntityTableDataset

        adapter = _nz_adapter(nesting=NZ_NESTING, contract=_NZ_WEIGHT_CONTRACT)
        _fake_engine(adapter, monkeypatch)
        path = tmp_path / "household_weighted.h5"
        adapter.write_dataset(_nz_frame([1, 1, 2], [1, 1, 2], [2.0, 3.0]), path, 2026)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.household["household_weight"].tolist() == [2.0, 3.0]
        assert "family_weight" not in reloaded.family.columns
        assert "person_weight" not in reloaded.person.columns


# ----------------------------------------------------------------------
# Periods
# ----------------------------------------------------------------------


class TestAxiomPeriod:
    def test_bounds_are_the_dense_executor_tuple(self) -> None:
        assert TAX_YEAR.bounds() == ("2026-04-01", "2027-03-31", "tax_year")

    def test_refuses_reversed_dates(self) -> None:
        with pytest.raises(ValueError, match="start.*end"):
            AxiomPeriod(start="2027-03-31", end="2026-04-01", kind="tax_year")

    @pytest.mark.parametrize(
        "start", ["2026-02-30", "20260401", "2026-4-1", "", None, 20260401]
    )
    def test_refuses_dates_that_are_not_canonical_iso(self, start) -> None:
        with pytest.raises(ValueError, match="ISO"):
            AxiomPeriod(start=start, end="2027-03-31", kind="tax_year")

    @pytest.mark.parametrize("kind", ["", "   ", " tax_year", "tax_year\n", None])
    def test_refuses_an_empty_or_padded_kind(self, kind) -> None:
        with pytest.raises(ValueError, match="kind"):
            AxiomPeriod(start="2026-04-01", end="2027-03-31", kind=kind)

    def test_a_single_day_period_is_valid(self) -> None:
        assert AxiomPeriod("2026-04-01", "2026-04-01", "day").bounds()[0] == (
            "2026-04-01"
        )


class TestPeriodLabels:
    def test_a_mapped_label_resolves_to_its_explicit_bounds(self) -> None:
        adapter = _nz_adapter(periods={"2026-27": TAX_YEAR})
        assert adapter._materialization_period("2026-27") == TAX_YEAR.bounds()

    def test_int_and_string_labels_are_one_label(self) -> None:
        adapter = _nz_adapter(periods={2026: TAX_YEAR})
        assert adapter._materialization_period(2026) == TAX_YEAR.bounds()
        assert adapter._materialization_period("2026") == TAX_YEAR.bounds()

    @pytest.mark.parametrize("label", ["2026", 2026, "2026-04", "2027-28", "2026-27 "])
    def test_an_unmapped_label_fails_closed(self, label) -> None:
        adapter = _nz_adapter(periods={"2026-27": TAX_YEAR})
        with pytest.raises(ValueError, match="No explicit Axiom period bounds"):
            adapter._materialization_period(label)

    def test_explicit_bounds_pass_through_a_mapping(self) -> None:
        other = AxiomPeriod("2025-04-01", "2026-03-31", "tax_year")
        adapter = _nz_adapter(periods={"2026-27": TAX_YEAR})
        assert adapter._materialization_period(other) == other.bounds()

    def test_without_a_mapping_calendar_labels_are_unchanged(self) -> None:
        adapter = _nz_adapter()
        assert adapter._materialization_period(2025) == (
            "2025-01-01",
            "2025-12-31",
            "calendar_year",
        )
        # "2026-27" reads as month 27 of 2026 and fails; it never becomes
        # the April-to-March tax year without an explicit mapping.
        with pytest.raises(ValueError, match="Invalid month"):
            adapter._materialization_period("2026-27")

    def test_duplicate_string_forms_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            _nz_adapter(periods={2026: TAX_YEAR, "2026": TAX_YEAR})

    @pytest.mark.parametrize("label", [True, "", 2026.0, None])
    def test_labels_must_be_ints_or_non_empty_strings(self, label) -> None:
        with pytest.raises(TypeError, match="labels"):
            _nz_adapter(periods={label: TAX_YEAR})

    def test_values_must_be_axiom_periods(self) -> None:
        with pytest.raises(TypeError, match="AxiomPeriod"):
            _nz_adapter(periods={"2026-27": ("2026-04-01", "2027-03-31", "x")})

    def test_an_empty_mapping_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one label"):
            _nz_adapter(periods={})

    @settings(max_examples=100, deadline=None)
    @given(
        labels=st.sets(
            st.text(alphabet="0123456789-ab", min_size=1, max_size=9),
            min_size=1,
            max_size=6,
        ),
        query=st.text(alphabet="0123456789-ab", min_size=1, max_size=9),
    )
    def test_property_a_label_resolves_iff_it_is_mapped(self, labels, query) -> None:
        """Invariant: an unmapped period label fails closed, never defaults."""

        periods = {
            label: AxiomPeriod("2026-04-01", f"2027-03-{10 + index:02d}", "tax_year")
            for index, label in enumerate(sorted(labels))
        }
        adapter = _nz_adapter(periods=periods)
        if query in periods:
            assert adapter._materialization_period(query) == periods[query].bounds()
        else:
            with pytest.raises(ValueError, match="No explicit Axiom period bounds"):
                adapter._materialization_period(query)

    def test_materialize_hands_the_mapped_bounds_to_the_engine(
        self, monkeypatch
    ) -> None:
        adapter = _nz_adapter(periods={"2026-27": TAX_YEAR})
        programs = _fake_engine(adapter, monkeypatch)
        adapter.materialize(_fake_frame(), ["family_assistance"], "2026-27")
        assert programs["Family"].calls == [
            {
                "period_kind": "tax_year",
                "start": "2026-04-01",
                "end": "2027-03-31",
                "inputs": ["family_rent"],
                "outputs": ["family_assistance"],
            }
        ]

    def test_materialize_refuses_an_unmapped_label_before_the_engine(
        self, monkeypatch
    ) -> None:
        adapter = _nz_adapter(periods={"2026-27": TAX_YEAR})
        programs = _fake_engine(adapter, monkeypatch)
        with pytest.raises(ValueError, match="No explicit Axiom period bounds"):
            adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        assert programs == {}


# ----------------------------------------------------------------------
# Nesting
# ----------------------------------------------------------------------


class TestNesting:
    @pytest.mark.parametrize(
        ("nesting", "message"),
        [
            ({"family": "tax_unit"}, "not a group entity"),
            ({"person": "household"}, "not a group entity"),
            ({"family": "family"}, "to itself"),
            ({"family": "household", "household": "family"}, "cyclic"),
        ],
    )
    def test_declarations_are_validated_against_the_schema(
        self, nesting, message
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _nz_adapter(nesting=nesting)

    def test_a_non_mapping_is_refused(self) -> None:
        with pytest.raises(TypeError, match="nesting"):
            _nz_adapter(nesting=[("family", "household")])

    def test_cross_household_family_is_refused_from_memberships_alone(
        self,
    ) -> None:
        # Equal household weights: the family weight still resolves, so weight
        # agreement cannot be what detects the split.
        frame = _nz_frame([1, 2], [1, 1], [2.0, 2.0])
        assert frame.resolve_weights("family").values.tolist() == [2.0]
        with pytest.raises(ValueError, match="exactly one 'household'.*\\[1\\]"):
            _nz_adapter(nesting=NZ_NESTING).materialize(frame, [], 2026)

    def test_without_a_declaration_the_split_is_not_checked(self) -> None:
        frame = _nz_frame([1, 2], [1, 1], [2.0, 2.0])
        assert _nz_adapter().materialize(frame, [], 2026) == {}

    def test_a_nested_population_passes(self) -> None:
        frame = _nz_frame([1, 1, 2, 2], [1, 1, 2, 3], [2.0, 3.0])
        assert _nz_adapter(nesting=NZ_NESTING).materialize(frame, [], 2026) == {}

    def test_the_split_blocks_an_export_before_anything_is_written(
        self, tmp_path
    ) -> None:
        frame = _nz_frame([1, 2], [1, 1], [2.0, 2.0])
        path = tmp_path / "split.h5"
        with pytest.raises(ValueError, match="exactly one 'household'"):
            _nz_adapter(nesting=NZ_NESTING).write_dataset(frame, path, 2026)
        assert not path.exists()

    def test_explicit_relation_column_disagreeing_with_membership_is_refused(
        self,
    ) -> None:
        frame = _nz_frame(
            [1, 2], [1, 1], [2.0, 2.0], family_columns={"family_household_id": [1]}
        )
        with pytest.raises(ValueError, match="family_household_id.*membership"):
            _nz_adapter().materialize(frame, [], 2026)

    def test_explicit_relation_column_agreeing_with_membership_passes(
        self,
    ) -> None:
        frame = _nz_frame(
            [1, 1, 2],
            [1, 1, 2],
            [2.0, 3.0],
            family_columns={"family_household_id": [1, 2]},
        )
        assert _nz_adapter().materialize(frame, [], 2026) == {}

    def test_relation_column_on_the_wrong_entity_blocks_materialize_and_export(
        self, tmp_path
    ) -> None:
        frame = _nz_frame(
            [1, 2], [1, 1], [2.0, 2.0], person_columns={"family_household_id": [1, 2]}
        )
        adapter = _nz_adapter()
        with pytest.raises(ValueError, match="family_household_id.*family.*person"):
            adapter.materialize(frame, [], 2026)
        path = tmp_path / "wrong_owner.h5"
        with pytest.raises(ValueError, match="family_household_id.*family.*person"):
            adapter.write_dataset(frame, path, 2026)
        assert not path.exists()

    def test_a_required_relation_column_is_an_export_rule(self, tmp_path) -> None:
        # A graph node need not slice an export-only column to materialize;
        # the export itself refuses the frame and writes nothing.
        contract = ExportContract(
            required=("family_household_id",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        frame = _nz_frame([1, 1, 2], [1, 1, 2], [2.0, 3.0])
        adapter = _nz_adapter(contract=contract)
        assert adapter.materialize(frame, [], 2026) == {}
        path = tmp_path / "missing_relation.h5"
        with pytest.raises(ValueError, match="Required relation column"):
            adapter.write_dataset(frame, path, 2026)
        assert not path.exists()

    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(population=nz_populations())
    def test_property_refused_iff_a_family_spans_households(self, population) -> None:
        """Invariant: a family nests in exactly one household.

        The declared nesting refuses a population exactly when some family
        has members in two households, whatever the weights.
        """

        frame = _nz_frame(
            population["person_households"],
            population["person_families"],
            population["household_weights"],
        )
        person = frame.table("person")
        spans = (
            person.groupby("person_family_id")["person_household_id"].nunique() > 1
        ).any()
        assert bool(spans) == population["split"]
        adapter = _nz_adapter(nesting=NZ_NESTING)
        if spans:
            with pytest.raises(ValueError, match="exactly one 'household'"):
                adapter.materialize(frame, [], 2026)
        else:
            assert adapter.materialize(frame, [], 2026) == {}


# ----------------------------------------------------------------------
# Graph output dtypes
# ----------------------------------------------------------------------


class TestGraphOutputCast:
    def test_judgment_int8_codes_become_int64(self) -> None:
        codes = np.asarray([1, -1, 0, 1], dtype=np.int8)
        cast = _graph_output("j", "judgment", codes)
        assert cast.dtype == np.dtype(np.int64)
        assert cast.tolist() == [1, -1, 0, 1]

    @settings(max_examples=200, deadline=None)
    @given(
        codes=st.lists(st.sampled_from([-1, 0, 1]), min_size=0, max_size=64),
        source=st.sampled_from([np.int8, np.int16, np.int32, np.int64]),
    )
    def test_property_judgment_codes_survive_the_cast_losslessly(
        self, codes, source
    ) -> None:
        """Invariant: graph-typed judgments are lossless (codes -1/0/1)."""

        native = np.asarray(codes, dtype=source)
        cast = _graph_output("j", "judgment", native)
        assert cast.dtype == np.dtype(np.int64)
        assert cast.astype(source).tobytes() == native.tobytes()

    @pytest.mark.parametrize("bad", [2, -2, 127])
    def test_judgment_codes_outside_the_tri_state_are_refused(self, bad) -> None:
        with pytest.raises(ValueError, match="outside -1/0/1"):
            _graph_output("j", "judgment", np.asarray([0, bad], dtype=np.int8))

    def test_unsigned_judgment_codes_cast_when_in_range(self) -> None:
        cast = _graph_output("j", "judgment", np.asarray([0, 1], dtype=np.uint8))
        assert cast.dtype == np.dtype(np.int64) and cast.tolist() == [0, 1]

    @pytest.mark.parametrize(
        ("engine_dtype", "values", "expected"),
        [
            ("integer", np.asarray([3, -4], dtype=np.int32), np.int64),
            ("integer", np.asarray([3, 4], dtype=np.uint16), np.int64),
            ("decimal", np.asarray([1.25, -0.5], dtype=np.float64), np.float64),
            ("decimal", np.asarray([1.25, -0.5], dtype=np.float32), np.float64),
            ("bool", np.asarray([True, False]), np.bool_),
        ],
    )
    def test_each_supported_dtype_casts_to_its_graph_dtype(
        self, engine_dtype, values, expected
    ) -> None:
        cast = _graph_output("v", engine_dtype, values)
        assert cast.dtype == np.dtype(expected)
        np.testing.assert_array_equal(cast, values)

    @pytest.mark.parametrize(
        ("engine_dtype", "values"),
        [
            ("integer", np.asarray([1.5, 2.0])),
            ("integer", np.asarray([True, False])),
            ("decimal", np.asarray([1, 2], dtype=np.int64)),
            ("decimal", np.asarray(["1.5"])),
            ("bool", np.asarray([0, 1], dtype=np.int8)),
            ("judgment", np.asarray([0.0, 1.0])),
            ("integer", np.asarray([2**64 - 1], dtype=np.uint64)),
        ],
    )
    def test_a_lossy_or_mismatched_array_is_refused(self, engine_dtype, values) -> None:
        with pytest.raises(ValueError, match="'v'"):
            _graph_output("v", engine_dtype, values)

    @pytest.mark.parametrize("engine_dtype", ["text", "date", "unknown"])
    def test_text_and_date_have_no_graph_dtype(self, engine_dtype) -> None:
        with pytest.raises(ValueError, match="no graph"):
            _graph_output("v", engine_dtype, np.asarray(["a"]))


class TestGraphTypedMaterialize:
    def test_output_dtypes_must_be_native_or_graph(self) -> None:
        with pytest.raises(ValueError, match="output_dtypes"):
            _nz_adapter(output_dtypes="pandas")

    def test_native_default_returns_the_engine_arrays_unchanged(
        self, monkeypatch
    ) -> None:
        adapter = _nz_adapter()
        _fake_engine(adapter, monkeypatch)
        results = adapter.materialize(
            _fake_frame(), ["family_assets_ok", "family_category"], 2026
        )
        assert results["family_assets_ok"].dtype == np.dtype(np.int8)
        assert results["family_category"].tolist() == ["small"] * 3

    def test_graph_mode_returns_graph_ownable_dtypes(self, monkeypatch) -> None:
        adapter = _nz_adapter(output_dtypes="graph", periods={"2026-27": TAX_YEAR})
        _fake_engine(adapter, monkeypatch)
        variables = [
            "person_benefit",
            "family_assets_ok",
            "family_assistance",
            "family_band",
            "family_is_large",
        ]
        results = adapter.materialize(_fake_frame(), variables, "2026-27")
        assert {name: results[name].dtype.str for name in variables} == {
            "person_benefit": np.dtype(np.float64).str,
            "family_assets_ok": np.dtype(np.int64).str,
            "family_assistance": np.dtype(np.float64).str,
            "family_band": np.dtype(np.int64).str,
            "family_is_large": np.dtype(np.bool_).str,
        }
        assert results["family_assets_ok"].tolist() == [1, -1, 0]
        assert results["person_benefit"].shape == (5,)

    @pytest.mark.parametrize("variable", ["family_category", "family_review_date"])
    def test_graph_mode_refuses_text_and_date_before_the_engine_runs(
        self, monkeypatch, variable
    ) -> None:
        adapter = _nz_adapter(output_dtypes="graph")
        programs = _fake_engine(adapter, monkeypatch)
        with pytest.raises(
            ValueError, match=f"{variable}.*refused|refused.*{variable}"
        ):
            adapter.materialize(_fake_frame(), ["family_assistance", variable], 2026)
        assert all(not program.calls for program in programs.values())

    def test_graph_dtype_refuses_a_native_adapter(self, monkeypatch) -> None:
        adapter = _nz_adapter()
        _fake_engine(adapter, monkeypatch)
        with pytest.raises(ValueError, match="output_dtypes='graph' adapters only"):
            adapter.graph_dtype("family_assets_ok")

    def test_graph_dtype_names_the_owned_column_token(self, monkeypatch) -> None:
        adapter = _nz_adapter(output_dtypes="graph")
        _fake_engine(adapter, monkeypatch)
        assert {
            name: adapter.graph_dtype(name)
            for name in (
                "person_benefit",
                "family_assets_ok",
                "family_band",
                "family_is_large",
            )
        } == {
            "person_benefit": "float64",
            "family_assets_ok": "int64",
            "family_band": "int64",
            "family_is_large": "bool",
        }
        with pytest.raises(ValueError, match="no graph column"):
            adapter.graph_dtype("family_category")
        with pytest.raises(ValueError, match="Unknown Axiom variable"):
            adapter.graph_dtype("not_a_variable")


# ----------------------------------------------------------------------
# Relations
# ----------------------------------------------------------------------


class TestAssertNoRelations:
    def test_a_relation_free_program_passes(self, monkeypatch) -> None:
        adapter = _nz_adapter()
        _fake_engine(adapter, monkeypatch)
        assert assert_no_relations(adapter, "family") is None

    def test_relations_are_refused_naming_the_module_path(self, monkeypatch) -> None:
        adapter = _nz_adapter()
        _fake_engine(
            adapter, monkeypatch, relations=(_Relation("member_of_household"),)
        )
        with pytest.raises(
            NotImplementedError,
            match="zz/policies/tests/axiom_toy_country.yaml.*member_of_household",
        ):
            assert_no_relations(adapter, "family")

    def test_an_entity_without_derived_rules_is_named(self, monkeypatch) -> None:
        adapter = _nz_adapter()
        _fake_engine(adapter, monkeypatch)
        with pytest.raises(ValueError, match="no derived rules.*Household"):
            assert_no_relations(adapter, "household")

    def test_only_axiom_engines_are_accepted(self) -> None:
        with pytest.raises(TypeError, match="AxiomEngine"):
            assert_no_relations(object(), "family")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Engine references
# ----------------------------------------------------------------------

_MODULE = "zz/policies/tests/toy.yaml"
_BASE_TREE = {
    _MODULE: b"format: rulespec/v1\nrules: []\n",
    "zz/policies/shared/rates.yaml": b"format: rulespec/v1\nrules:\n  - name: r\n",
    ".axiom/toolchain.toml": b'engine = "pinned"\n',
    "README.md": b"RuleSpec fixture tree.\n",
}


def _write_tree(root: Path, files: Mapping[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _ref(root: Path, **engine_kwargs) -> str:
    engine = AxiomEngine(
        root / _MODULE, schema=NZ_SCHEMA, rulespec_roots=(root,), **engine_kwargs
    )
    return axiom_engine_ref(
        engine,
        engine_commit=ENGINE_COMMIT,
        wheel_sha256=WHEEL_SHA256,
        rulespec_root=root,
        rulespec_commit=RULESPEC_COMMIT,
    )


_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=6
)


@st.composite
def rulespec_trees(draw: st.DrawFn) -> dict[str, bytes]:
    """A random RuleSpec-like tree that always holds the module.

    Directory components carry a ``d_`` prefix and files a ``.yaml`` suffix,
    so no path is both a file and a directory; names are lower case, so a
    case-insensitive file system cannot fold two of them together.
    """

    paths = draw(
        st.sets(
            st.builds(
                lambda directories, name: "/".join(
                    [*(f"d_{part}" for part in directories), f"{name}.yaml"]
                ),
                st.lists(_NAME, max_size=3),
                _NAME,
            ),
            max_size=6,
        )
    )
    files = {path: draw(st.binary(min_size=1, max_size=48)) for path in sorted(paths)}
    files[_MODULE] = draw(st.binary(min_size=1, max_size=48))
    return files


class TestRulespecTreeDigest:
    @settings(max_examples=60, deadline=None)
    @given(files=rulespec_trees())
    def test_differential_digest_equals_the_graph_directory_identity(
        self, tmp_path_factory, files
    ) -> None:
        """Two implementations of one formula: adapter and graph agree."""

        root = _write_tree(tmp_path_factory.mktemp("tree"), files)
        digest, size = rulespec_tree_digest(root)
        assert (digest, size) == _directory_identity(root)
        assert size == sum(len(content) for content in files.values())
        # The public source key wraps exactly this digest and size.
        assert source_content_key("rulespec_nz", root) == sha256_domain(
            "source", canonical_json(("rulespec_nz", digest, size))
        )

    def test_a_file_is_not_a_root(self, tmp_path) -> None:
        (tmp_path / "file.yaml").write_bytes(b"x")
        with pytest.raises(ValueError, match="not a directory"):
            rulespec_tree_digest(tmp_path / "file.yaml")


class TestAxiomEngineRef:
    def test_the_reference_is_canonical_json_of_every_pin(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec-zz", _BASE_TREE)
        reference = _ref(
            root, periods={"2026-27": TAX_YEAR}, nesting=NZ_NESTING, arithmetic="f64"
        )
        document = json.loads(reference)
        assert reference == canonical_json(document).decode("utf-8")
        digest, size = rulespec_tree_digest(root)
        assert document == {
            "format": "microcosm.frame.axiom-engine-ref/1",
            "engine": "axiom-rules-engine",
            "engine_commit": ENGINE_COMMIT,
            "engine_wheel_sha256": WHEEL_SHA256,
            "rulespec_commit": RULESPEC_COMMIT,
            "rulespec_tree_sha256": digest,
            "rulespec_tree_bytes": size,
            "module": _MODULE,
            "module_sha256": hashlib.sha256(_BASE_TREE[_MODULE]).hexdigest(),
            "arithmetic": "f64",
            "output_dtypes": "native",
            "person_entity": "person",
            "group_entities": ["household", "family"],
            "entity_names": {
                "family": "Family",
                "household": "Household",
                "person": "Person",
            },
            "periods": {
                "2026-27": {
                    "start": "2026-04-01",
                    "end": "2027-03-31",
                    "kind": "tax_year",
                }
            },
            "nesting": {"family": "household"},
        }
        assert str(tmp_path) not in reference

    def test_determinism_relocation_and_timestamps_leave_it_unchanged(
        self, tmp_path
    ) -> None:
        root = _write_tree(tmp_path / "one", _BASE_TREE)
        first = _ref(root)
        assert _ref(root) == first
        relocated = tmp_path / "elsewhere" / "two"
        shutil.copytree(root, relocated)
        os.utime(relocated / _MODULE, (0, 0))
        assert _ref(relocated) == first

    @pytest.mark.parametrize(
        "change",
        [
            {"arithmetic": "f64"},
            {"output_dtypes": "graph"},
            {"periods": {"2026-27": TAX_YEAR}},
            {"nesting": NZ_NESTING},
            {
                "entity_names": {
                    "person": "Person",
                    "household": "Household",
                    "family": "BenefitUnit",
                }
            },
        ],
    )
    def test_every_configuration_field_moves_it(self, tmp_path, change) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        assert _ref(root, **change) != _ref(root)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("engine_commit", "d" * 40),
            ("wheel_sha256", "e" * 64),
            ("rulespec_commit", "f" * 40),
        ],
    )
    def test_every_declared_pin_moves_it(self, tmp_path, field, value) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        engine = AxiomEngine(root / _MODULE, rulespec_roots=(root,))
        pins = {
            "engine_commit": ENGINE_COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "rulespec_commit": RULESPEC_COMMIT,
        }
        base = axiom_engine_ref(engine, rulespec_root=root, **pins)
        moved = axiom_engine_ref(engine, rulespec_root=root, **{**pins, field: value})
        assert moved != base

    @settings(max_examples=60, deadline=None)
    @given(files=rulespec_trees(), data=st.data())
    def test_property_it_moves_iff_a_pinned_byte_moves(
        self, tmp_path_factory, files, data
    ) -> None:
        """Invariant: ``engine_ref`` changes iff a pinned byte changes.

        The same bytes in another place give the same reference; editing any
        one byte of any file under the root gives a different one.
        """

        root = _write_tree(tmp_path_factory.mktemp("base"), files)
        reference = _ref(root)
        copy = tmp_path_factory.mktemp("copy")
        shutil.copytree(root, copy, dirs_exist_ok=True)
        assert _ref(copy) == reference
        name = data.draw(st.sampled_from(sorted(files)))
        content = files[name]
        offset = data.draw(st.integers(min_value=0, max_value=len(content) - 1))
        replacement = data.draw(
            st.integers(min_value=0, max_value=255).filter(
                lambda value: value != content[offset]
            )
        )
        edited = bytearray(content)
        edited[offset] = replacement
        (copy / name).write_bytes(bytes(edited))
        assert _ref(copy) != reference

    def test_adding_or_renaming_a_file_moves_it(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        reference = _ref(root)
        (root / "zz/policies/shared/new.yaml").write_bytes(b"x")
        added = _ref(root)
        assert added != reference
        (root / "zz/policies/shared/new.yaml").rename(
            root / "zz/policies/shared/renamed.yaml"
        )
        assert _ref(root) not in {reference, added}

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("engine_commit", "a" * 39, "engine_commit"),
            ("engine_commit", "A" * 40, "engine_commit"),
            ("engine_commit", "a" * 12, "engine_commit"),
            ("rulespec_commit", "main", "rulespec_commit"),
            ("rulespec_commit", None, "rulespec_commit"),
            ("wheel_sha256", "b" * 63, "wheel_sha256"),
            ("wheel_sha256", "B" * 64, "wheel_sha256"),
        ],
    )
    def test_malformed_pins_are_refused(self, tmp_path, field, value, message) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        engine = AxiomEngine(root / _MODULE, rulespec_roots=(root,))
        pins = {
            "engine_commit": ENGINE_COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "rulespec_commit": RULESPEC_COMMIT,
            field: value,
        }
        with pytest.raises(ValueError, match=message):
            axiom_engine_ref(engine, rulespec_root=root, **pins)

    def test_a_64_hex_commit_is_accepted(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        engine = AxiomEngine(root / _MODULE, rulespec_roots=(root,))
        assert axiom_engine_ref(
            engine,
            engine_commit="a" * 64,
            wheel_sha256=WHEEL_SHA256,
            rulespec_root=root,
            rulespec_commit="c" * 64,
        )

    def test_the_root_must_be_the_adapters_only_root(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        other = _write_tree(tmp_path / "other", {"x.yaml": b"x"})
        pins = {
            "engine_commit": ENGINE_COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "rulespec_commit": RULESPEC_COMMIT,
        }
        two_roots = AxiomEngine(root / _MODULE, rulespec_roots=(root, other))
        with pytest.raises(ValueError, match="only root"):
            axiom_engine_ref(two_roots, rulespec_root=root, **pins)
        one_root = AxiomEngine(root / _MODULE, rulespec_roots=(root,))
        with pytest.raises(ValueError, match="only root"):
            axiom_engine_ref(one_root, rulespec_root=other, **pins)

    def test_the_module_must_be_a_file_under_the_root(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        outside = _write_tree(tmp_path / "outside", {"m.yaml": b"x"}) / "m.yaml"
        pins = {
            "engine_commit": ENGINE_COMMIT,
            "wheel_sha256": WHEEL_SHA256,
            "rulespec_commit": RULESPEC_COMMIT,
        }
        for module in (outside, root / "zz/missing.yaml"):
            engine = AxiomEngine(module, rulespec_roots=(root,))
            with pytest.raises(ValueError, match="not a file under"):
                axiom_engine_ref(engine, rulespec_root=root, **pins)

    def test_a_symbolic_link_under_the_root_is_refused(self, tmp_path) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        target = _write_tree(tmp_path / "linked", {"hidden.yaml": b"x"})
        (root / "zz/policies/linked").symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError, match="symbolic link zz/policies/linked"):
            _ref(root)

    def test_only_axiom_engines_are_accepted(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="AxiomEngine"):
            axiom_engine_ref(
                object(),  # type: ignore[arg-type]
                engine_commit=ENGINE_COMMIT,
                wheel_sha256=WHEEL_SHA256,
                rulespec_root=tmp_path,
                rulespec_commit=RULESPEC_COMMIT,
            )


class TestEngineRefPin:
    """The adapter computes only from the bytes its reference names."""

    def _adapter(self, root: Path) -> AxiomEngine:
        return AxiomEngine(
            root / _MODULE,
            schema=NZ_SCHEMA,
            rulespec_roots=(root,),
            output_dtypes="graph",
        )

    def _reference(self, adapter: AxiomEngine, root: Path) -> str:
        return axiom_engine_ref(
            adapter,
            engine_commit=ENGINE_COMMIT,
            wheel_sha256=WHEEL_SHA256,
            rulespec_root=root,
            rulespec_commit=RULESPEC_COMMIT,
        )

    def test_a_reference_then_unchanged_compiles_succeed(
        self, tmp_path, monkeypatch
    ) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        adapter = self._adapter(root)
        _fake_engine(adapter, monkeypatch)
        reference = self._reference(adapter, root)
        adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        adapter.materialize(_fake_frame(), ["person_benefit"], 2026)
        assert self._reference(adapter, root) == reference

    def test_an_edit_after_the_reference_blocks_the_next_compile(
        self, tmp_path, monkeypatch
    ) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        adapter = self._adapter(root)
        programs = _fake_engine(adapter, monkeypatch)
        self._reference(adapter, root)
        (root / "zz/policies/shared/rates.yaml").write_bytes(b"edited")
        with pytest.raises(ValueError, match="changed after this adapter"):
            adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        assert programs == {}

    def test_an_edit_after_the_reference_blocks_a_second_reference(
        self, tmp_path, monkeypatch
    ) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        adapter = self._adapter(root)
        _fake_engine(adapter, monkeypatch)
        self._reference(adapter, root)
        adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        (root / _MODULE).write_bytes(b"edited")
        with pytest.raises(ValueError, match="changed after this adapter"):
            self._reference(adapter, root)

    def test_a_symbolic_link_added_after_the_reference_blocks_the_next_compile(
        self, tmp_path, monkeypatch
    ) -> None:
        # The digest does not descend into a linked directory, so it cannot
        # see the link; the compile-time check must refuse it directly.
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        adapter = self._adapter(root)
        programs = _fake_engine(adapter, monkeypatch)
        self._reference(adapter, root)
        before = rulespec_tree_digest(root)
        target = _write_tree(tmp_path / "linked", {"hidden.yaml": b"x"})
        (root / "zz/policies/linked").symlink_to(target, target_is_directory=True)
        assert rulespec_tree_digest(root) == before
        with pytest.raises(ValueError, match="symbolic link zz/policies/linked"):
            adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        assert programs == {}

    def test_a_reference_after_compiling_is_refused(
        self, tmp_path, monkeypatch
    ) -> None:
        root = _write_tree(tmp_path / "rulespec", _BASE_TREE)
        adapter = self._adapter(root)
        _fake_engine(adapter, monkeypatch)
        adapter.materialize(_fake_frame(), ["family_assistance"], 2026)
        with pytest.raises(ValueError, match="before the adapter compiles"):
            self._reference(adapter, root)

    def test_an_adapter_without_a_reference_never_hashes_its_root(
        self, monkeypatch
    ) -> None:
        # The Belgian path is unchanged: no reference, no digest on compile.
        adapter = _nz_adapter()
        _fake_engine(adapter, monkeypatch)

        def refuse(root):
            raise AssertionError("an unreferenced adapter hashed its root")

        monkeypatch.setattr(
            "microcosm.frame.adapters.axiom.rulespec_tree_digest", refuse
        )
        adapter.materialize(_fake_frame(), ["family_assistance"], 2026)


class TestAxiomEngineRefGitCheckout:
    @pytest.fixture
    def checkout(self, tmp_path) -> tuple[Path, str]:
        root = _write_tree(
            tmp_path / "rulespec-git",
            {**_BASE_TREE, ".gitignore": b"*.local.yaml\n"},
        )
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "fixture")
        return root, _git(root, "rev-parse", "HEAD")

    def _checkout_ref(self, root: Path, commit: str) -> str:
        engine = AxiomEngine(root / _MODULE, rulespec_roots=(root,))
        return axiom_engine_ref(
            engine,
            engine_commit=ENGINE_COMMIT,
            wheel_sha256=WHEEL_SHA256,
            rulespec_root=root,
            rulespec_commit=commit,
        )

    def test_a_clean_checkout_at_the_declared_commit_is_accepted(
        self, checkout
    ) -> None:
        root, head = checkout
        first = self._checkout_ref(root, head)
        # The status check takes no optional locks, so it cannot rewrite the
        # index and move the digest between two calls.
        assert self._checkout_ref(root, head) == first
        assert json.loads(first)["rulespec_commit"] == head

    def test_a_modified_file_is_refused(self, checkout) -> None:
        root, head = checkout
        (root / _MODULE).write_bytes(b"format: rulespec/v1\nrules: [edited]\n")
        with pytest.raises(ValueError, match="modified, untracked, or ignored"):
            self._checkout_ref(root, head)

    def test_an_untracked_file_is_refused(self, checkout) -> None:
        root, head = checkout
        (root / "zz/untracked.yaml").write_bytes(b"x")
        with pytest.raises(ValueError, match="modified, untracked, or ignored"):
            self._checkout_ref(root, head)

    def test_an_ignored_file_is_refused(self, checkout) -> None:
        # The digest would hash it, but the declared commit does not hold it.
        root, head = checkout
        (root / "zz/override.local.yaml").write_bytes(b"x")
        assert _git(root, "status", "--porcelain") == ""
        with pytest.raises(ValueError, match="ignored"):
            self._checkout_ref(root, head)

    @pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
    def test_an_edit_hidden_by_an_index_flag_is_refused(self, checkout, flag) -> None:
        # git status trusts the flag and reports nothing, so the checkout would
        # pass as clean while the module no longer holds the commit's bytes.
        root, head = checkout
        _git(root, "update-index", flag, _MODULE)
        (root / _MODULE).write_bytes(b"format: rulespec/v1\nrules: [hidden]\n")
        assert _git(root, "status", "--porcelain", "--ignored") == ""
        with pytest.raises(ValueError, match=f"{_MODULE} is flagged"):
            self._checkout_ref(root, head)

    @pytest.mark.parametrize(
        ("flag", "unflag"),
        [
            ("--skip-worktree", "--no-skip-worktree"),
            ("--assume-unchanged", "--no-assume-unchanged"),
        ],
    )
    def test_a_clean_checkout_with_its_index_flags_cleared_is_accepted(
        self, checkout, flag, unflag
    ) -> None:
        root, head = checkout
        _git(root, "update-index", flag, _MODULE)
        with pytest.raises(ValueError, match="flagged"):
            self._checkout_ref(root, head)
        _git(root, "update-index", unflag, _MODULE)
        assert json.loads(self._checkout_ref(root, head))["rulespec_commit"] == head

    def test_a_checkout_reference_is_clone_specific_and_an_export_is_not(
        self, checkout, tmp_path
    ) -> None:
        # Documented: a checkout's .git is hashed, so its reference names that
        # clone; exports of the same commit agree wherever they land.
        root, head = checkout
        archive = tmp_path / "export.tar"
        _git(root, "archive", "--format=tar", "-o", str(archive), "HEAD")
        exports = []
        for name in ("export-one", "export-two"):
            destination = tmp_path / name
            with tarfile.open(archive) as bundle:
                bundle.extractall(destination, filter="data")
            exports.append(self._checkout_ref(destination, head))
        assert exports[0] == exports[1]
        assert self._checkout_ref(root, head) != exports[0]

    def test_a_checkout_at_another_commit_is_refused(self, checkout) -> None:
        root, head = checkout
        other = "0" * 40 if head != "0" * 40 else "1" * 40
        with pytest.raises(ValueError, match="not the declared rulespec_commit"):
            self._checkout_ref(root, other)


# ----------------------------------------------------------------------
# Real engine (skipped without axiom_rules_engine)
# ----------------------------------------------------------------------


_FIXTURE_COMMIT = "0" * 40


def _fixture_ref(engine: AxiomEngine) -> str:
    """A reference for the in-repo fixture; the commit is a test placeholder."""

    return axiom_engine_ref(
        engine,
        engine_commit=_FIXTURE_COMMIT,
        wheel_sha256=WHEEL_SHA256,
        rulespec_root=FIXTURE_RULESPEC_ROOT,
        rulespec_commit=_FIXTURE_COMMIT,
    )


class _ToyPopulation(KernelBase):
    """CREATE kernel loading the toy population from a JSON source."""

    ref = "test.toy_population@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.CREATE,
    )

    def run(self, context: KernelContext) -> KernelResult:
        document = json.loads(context.sources["population"].read_text())
        schema = NZ_SCHEMA if "family" in document else BE_SCHEMA
        tables = {
            entity: pd.DataFrame(
                {
                    column: np.asarray(values, dtype=document["dtypes"][column])
                    for column, values in document[entity].items()
                }
            )
            for entity in schema.entities
        }
        frame = Frame(
            tables,
            schema,
            {
                "household": Weights(
                    np.asarray(document["household_weights"], dtype=np.float64),
                    WeightKind.DESIGN,
                )
            },
        )
        return KernelResult(frame=frame, receipt={"persons": len(tables["person"])})


_TOY_COUNTRY_DOCUMENT = {
    "person": {
        "person_id": [1, 2, 3],
        "person_household_id": [1, 1, 2],
        "toy_taxable_income": [5_000.0, 10_000.0, 20_000.0],
        "toy_is_exempt": [False, False, False],
        "toy_child_count": [0, 2, 1],
    },
    "household": {"household_id": [1, 2], "toy_household_rent": [7_200.0, 4_800.0]},
    "household_weights": [1500.0, 900.0],
    "dtypes": {
        "person_id": "int64",
        "person_household_id": "int64",
        "toy_taxable_income": "float64",
        "toy_is_exempt": "bool",
        "toy_child_count": "int64",
        "household_id": "int64",
        "toy_household_rent": "float64",
    },
}

_TOY_FAMILY_DOCUMENT = {
    "person": {
        "person_id": [1, 2, 3, 4, 5],
        "person_household_id": [1, 1, 2, 2, 2],
        "person_family_id": [10, 10, 20, 21, 21],
        "toy_taxable_income": [5_000.0, 10_000.0, 20_000.0, 0.0, 0.0],
        "toy_is_exempt": [False, False, False, False, False],
        "toy_child_count": [0, 0, 0, 1, 0],
    },
    "household": {"household_id": [1, 2]},
    "family": {
        "family_id": [10, 20, 21],
        "toy_family_weekly_rent": [300.0, 80.0, 180.0],
        "toy_family_cash_assets": [9_000.0, 1_000.0, 8_000.0],
        "toy_family_size": [2, 1, 3],
    },
    "household_weights": [1200.0, 800.0],
    "dtypes": {
        "person_id": "int64",
        "person_household_id": "int64",
        "person_family_id": "int64",
        "toy_taxable_income": "float64",
        "toy_is_exempt": "bool",
        "toy_child_count": "int64",
        "household_id": "int64",
        "family_id": "int64",
        "toy_family_weekly_rent": "float64",
        "toy_family_cash_assets": "float64",
        "toy_family_size": "int64",
    },
}


def _source_node(document: Mapping[str, object]) -> Node:
    outputs = tuple(
        Owned(entity, column, document["dtypes"][column])
        for entity in ("person", "household", "family")
        if entity in document
        for column in document[entity]
        if not column.endswith("_id")
    )
    return Node(
        "population",
        _ToyPopulation.ref,
        outputs=outputs,
        structural=StructuralDelta.CREATE,
        sources=("population",),
    )


def _run(
    tmp_path: Path,
    document: Mapping[str, object],
    nodes: tuple[Node, ...],
    kernels: tuple[object, ...],
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "population.json"
    source.write_text(json.dumps(document))
    registry = KernelRegistry()
    registry.register(_ToyPopulation())
    for kernel in kernels:
        registry.register(kernel)
    graph = Graph(
        "axiom-graph-adapter",
        (SourceRef("population", "raw-bytes-v1"),),
        (_source_node(document), *nodes),
    )
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(graph),
        sources={"population": source},
        store=store,
        kernels=registry,
        resume="forbid",
        decisions=(),
    )
    return manifest, store


def _column(manifest, store, node_id: str, entity: str, column: str) -> pd.Series:
    return store.load_column(manifest.nodes[node_id].artifacts[(entity, column)])


@needs_engine
class TestRealEngineOnTheGraph:
    def test_graph_typed_simulate_rules_runs_in_a_filter_free_graph(
        self, tmp_path
    ) -> None:
        engine = AxiomEngine(
            FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS, output_dtypes="graph"
        )
        engine_ref = _fixture_ref(engine)
        variables = ("toy_income_tax", "toy_housing_allowance")
        node = Node(
            "toy_rules",
            SimulateRulesKernel.ref,
            inputs=(
                Slice(
                    "person",
                    ("toy_taxable_income", "toy_is_exempt", "toy_child_count"),
                ),
                Slice("household", ("toy_household_rent",)),
            ),
            outputs=tuple(
                Owned(
                    engine.variable_metadata(name).entity,
                    name,
                    engine.graph_dtype(name),
                )
                for name in variables
            ),
            params={"engine_ref": engine_ref, "variables": variables, "period": 2025},
        )
        manifest, store = _run(
            tmp_path,
            _TOY_COUNTRY_DOCUMENT,
            (node,),
            (SimulateRulesKernel(engine_ref, engine),),
        )
        tax = _column(manifest, store, "toy_rules", "person", "toy_income_tax")
        allowance = _column(
            manifest, store, "toy_rules", "household", "toy_housing_allowance"
        )
        assert tax.dtype == np.dtype(np.float64)
        np.testing.assert_allclose(tax.to_numpy(), [500.0, 1_000.0, 3_500.0])
        np.testing.assert_allclose(allowance.to_numpy(), [1_200.0, 0.0])

    def test_the_tax_year_label_reaches_the_engine_as_explicit_bounds(
        self, monkeypatch
    ) -> None:
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            periods={"2026-27": TAX_YEAR},
            output_dtypes="graph",
        )
        program = adapter._program("person")
        calls: list[tuple[str, str, str]] = []
        original = program.execute

        def recording_execute(*, period_kind, start, end, **kwargs):
            calls.append((start, end, period_kind))
            return original(period_kind=period_kind, start=start, end=end, **kwargs)

        monkeypatch.setattr(program, "execute", recording_execute)
        frame = Frame(
            {
                "person": pd.DataFrame(
                    {
                        "person_id": [1, 2],
                        "person_household_id": [1, 2],
                        "toy_taxable_income": [5_000.0, 20_000.0],
                        "toy_is_exempt": [False, False],
                        "toy_child_count": [0, 1],
                    }
                ),
                "household": pd.DataFrame({"household_id": [1, 2]}),
            },
            BE_SCHEMA,
            {"household": Weights(np.asarray([1.0, 1.0]), WeightKind.DESIGN)},
        )
        results = adapter.materialize(frame, ["toy_income_tax"], "2026-27")
        assert calls == [("2026-04-01", "2027-03-31", "tax_year")]
        np.testing.assert_allclose(results["toy_income_tax"], [500.0, 3_500.0])
        with pytest.raises(ValueError, match="No explicit Axiom period bounds"):
            adapter.materialize(frame, ["toy_income_tax"], 2026)

    def test_family_judgments_are_lossless_graph_codes(self) -> None:
        adapter = AxiomEngine(
            FAMILY_FIXTURE_MODULE,
            schema=NZ_SCHEMA,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            nesting=NZ_NESTING,
        )
        typed = AxiomEngine(
            FAMILY_FIXTURE_MODULE,
            schema=NZ_SCHEMA,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            nesting=NZ_NESTING,
            output_dtypes="graph",
        )
        assert_no_relations(adapter, "family")
        frame = _nz_frame(
            [1, 1, 2, 2, 2],
            [10, 10, 20, 21, 21],
            [1200.0, 800.0],
            family_columns={
                "toy_family_weekly_rent": [300.0, 80.0, 180.0],
                "toy_family_cash_assets": [9_000.0, 1_000.0, 8_000.0],
                "toy_family_size": np.asarray([2, 1, 3], dtype=np.int64),
            },
        )
        variables = [
            "toy_family_assets_within_limit",
            "toy_family_rent_assistance",
            "toy_family_size_band",
            "toy_family_is_large",
        ]
        native = adapter.materialize(frame, variables, 2026)
        graph = typed.materialize(frame, variables, 2026)
        judgment = "toy_family_assets_within_limit"
        assert native[judgment].dtype == np.dtype(np.int8)
        assert graph[judgment].dtype == np.dtype(np.int64)
        assert graph[judgment].tolist() == [-1, 1, 1]
        assert graph[judgment].astype(np.int8).tobytes() == native[judgment].tobytes()
        np.testing.assert_allclose(
            graph["toy_family_rent_assistance"], [100.0, 0.0, 40.0]
        )
        assert graph["toy_family_size_band"].dtype == np.dtype(np.int64)
        assert graph["toy_family_size_band"].tolist() == [1, 1, 2]
        assert graph["toy_family_is_large"].dtype == np.dtype(np.bool_)
        assert graph["toy_family_is_large"].tolist() == [False, False, True]
        assert {name: typed.graph_dtype(name) for name in variables} == {
            "toy_family_assets_within_limit": "int64",
            "toy_family_rent_assistance": "float64",
            "toy_family_size_band": "int64",
            "toy_family_is_large": "bool",
        }

    def test_rules_by_ref_runs_two_axiom_engines_in_one_graph(self, tmp_path) -> None:
        person_engine = AxiomEngine(
            FIXTURE_MODULE,
            schema=NZ_SCHEMA,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            nesting=NZ_NESTING,
            output_dtypes="graph",
            periods={"2026-27": TAX_YEAR},
        )
        family_engine = AxiomEngine(
            FAMILY_FIXTURE_MODULE,
            schema=NZ_SCHEMA,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            nesting=NZ_NESTING,
            output_dtypes="graph",
            periods={"2026-27": TAX_YEAR},
        )
        person_ref = _fixture_ref(person_engine)
        family_ref = _fixture_ref(family_engine)
        assert person_ref != family_ref
        person_variables = ("toy_income_tax",)
        family_variables = (
            "toy_family_assets_within_limit",
            "toy_family_rent_assistance",
        )

        def node(node_id, kernel_ref, engine, engine_ref, variables, inputs):
            return Node(
                node_id,
                kernel_ref,
                inputs=inputs,
                outputs=tuple(
                    Owned(
                        engine.variable_metadata(name).entity,
                        name,
                        engine.graph_dtype(name),
                    )
                    for name in variables
                ),
                params={
                    "engine_ref": engine_ref,
                    "variables": variables,
                    "period": "2026-27",
                },
            )

        person_inputs = (
            Slice(
                "person",
                ("toy_taxable_income", "toy_is_exempt", "toy_child_count"),
            ),
        )
        # A family-only node still slices one person data column: the
        # kernel rebuilds group tables from the person table it is given.
        family_inputs = (
            Slice(
                "family",
                (
                    "toy_family_weekly_rent",
                    "toy_family_cash_assets",
                    "toy_family_size",
                ),
            ),
            Slice("person", ("toy_taxable_income",)),
        )
        by_ref = SimulateRulesByRefKernel(
            {person_ref: person_engine, family_ref: family_engine}
        )
        manifest, store = _run(
            tmp_path / "by-ref",
            _TOY_FAMILY_DOCUMENT,
            (
                node(
                    "rules.person",
                    by_ref.ref,
                    person_engine,
                    person_ref,
                    person_variables,
                    person_inputs,
                ),
                node(
                    "rules.family",
                    by_ref.ref,
                    family_engine,
                    family_ref,
                    family_variables,
                    family_inputs,
                ),
            ),
            (by_ref,),
        )
        np.testing.assert_allclose(
            _column(
                manifest, store, "rules.person", "person", "toy_income_tax"
            ).to_numpy(),
            [500.0, 1_000.0, 3_500.0, 0.0, 0.0],
        )
        codes = _column(
            manifest, store, "rules.family", "family", "toy_family_assets_within_limit"
        )
        assert codes.dtype == np.dtype(np.int64)
        assert codes.tolist() == [-1, 1, 1]

        # Differential: each node equals a single-engine simulate.rules@1 run.
        for node_id, engine, engine_ref, variables, inputs in (
            (
                "rules.person",
                person_engine,
                person_ref,
                person_variables,
                person_inputs,
            ),
            (
                "rules.family",
                family_engine,
                family_ref,
                family_variables,
                family_inputs,
            ),
        ):
            single, single_store = _run(
                tmp_path / f"single-{node_id}",
                _TOY_FAMILY_DOCUMENT,
                (
                    node(
                        node_id,
                        SimulateRulesKernel.ref,
                        engine,
                        engine_ref,
                        variables,
                        inputs,
                    ),
                ),
                (SimulateRulesKernel(engine_ref, engine),),
            )
            assert single.nodes[node_id].receipt == manifest.nodes[node_id].receipt
            for name in variables:
                entity = engine.variable_metadata(name).entity
                expected = _column(single, single_store, node_id, entity, name)
                actual = _column(manifest, store, node_id, entity, name)
                pd.testing.assert_series_equal(actual, expected)
                assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
