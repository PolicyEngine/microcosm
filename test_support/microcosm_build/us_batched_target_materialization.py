"""Household-batched base target materialization.

Fake-engine tests cover household partitions, column placement, engine
release and group nesting. Real-engine tests compare household-local targets
across batch sizes and demonstrate why Medicaid-cost targets require one
whole-pool batch. Multi-batch passes refuse known population aggregates.
"""

# ruff: noqa: F401

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_builder_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake engine: every value is a function of one record's own id
# ---------------------------------------------------------------------------

_VARIABLE_ENTITIES = {
    "income_tax": "tax_unit",
    "taxable_income": "tax_unit",
    "adjusted_gross_income": "tax_unit",
    "filing_status": "tax_unit",
    "state_income_tax": "tax_unit",
    "eitc": "tax_unit",
    "eitc_child_count": "tax_unit",
    "tax_unit_itemizes": "tax_unit",
    "itemized_taxable_income_deductions": "tax_unit",
    "ctc": "tax_unit",
    "ctc_limiting_tax_liability": "tax_unit",
    "tax_unit_size": "tax_unit",
    "mock_credit": "tax_unit",
    "age": "person",
    "ssi": "person",
    "employment_income": "person",
    "rental_income": "person",
    "farm_rent_income": "person",
    "snap": "spm_unit",
    "tanf": "spm_unit",
    "spm_unit_net_income": "spm_unit",
    "household_benefits": "household",
    "family_mock_income": "family",
    "marital_unit_mock_income": "marital_unit",
    # A per-record value whose formula also reads a population aggregate
    # when a test asks it to (``aggregate_reads``).
    "aggregate_probe": "person",
}
_FILING_STATUSES = ("SINGLE", "JOINT", "HEAD_OF_HOUSEHOLD", "SEPARATE")


def _record_value(variable: str, record_id: int):
    """One record's value; it reads nothing but the record's own id."""

    if variable == "filing_status":
        return _FILING_STATUSES[(record_id + record_id // 10) % len(_FILING_STATUSES)]
    if variable == "tax_unit_itemizes":
        return record_id % 3 == 0
    if variable == "eitc_child_count":
        return float(record_id % 4)
    if variable == "age":
        return float((record_id * 13) % 90)
    salt = sum(ord(character) for character in variable)
    # Signed, with zeros, so positive-part, loss-part and indicator slices
    # all see non-trivial rows.
    value = float(((record_id * 7919 + salt * 104729) % 2_000) - 500)
    return 0.0 if (record_id * 31 + salt) % 6 == 0 else value


class _FakeEngineLedger:
    """Records every engine the materializer builds, in order."""

    def __init__(self) -> None:
        self.simulations: list[object] = []
        self.constructions: list[dict[str, object]] = []
        self.reform_systems: list[object] = []


class _FakeHolders:
    """policyengine-core's holder view: the periods each variable is known for."""

    def __init__(self, known: dict[str, tuple[str, ...]] | None = None) -> None:
        self.known = {name: list(periods) for name, periods in (known or {}).items()}
        self.branches: dict[str, _FakeHolders] = {}
        self.clone_branches = False

    def get_branch(self, name: str):
        if name not in self.branches:
            branch = copy.copy(self) if self.clone_branches else _FakeHolders()
            branch.branches = {}
            branch.known = {}
            self.branches[name] = branch
        return self.branches[name]

    def get_holder(self, variable: str):
        periods = tuple(self.known.get(variable, ()))
        return SimpleNamespace(get_known_periods=lambda: list(periods))

    def record(self, variable: str, period) -> None:
        self.known.setdefault(variable, []).append(str(period))


def _install_fake_engine(
    builder,
    monkeypatch,
    *,
    reform_specs,
    aggregate_reads: dict[str, tuple[str, ...]] | None = None,
    aggregate_branch: str | None = None,
    aggregate_delete_branch: bool = False,
    aggregate_household: int | None = None,
    aggregate_reform_only: bool = False,
    aggregate_in_baseline: bool = False,
    clone_branches: bool = False,
    stored_inputs: dict[str, tuple[str, ...]] | None = None,
):
    """``aggregate_reads`` maps a variable to the population aggregates its
    formula computes (in the slash-delimited ``aggregate_branch`` if given);
    ``stored_inputs`` are known from construction, as dataset inputs are."""

    ledger = _FakeEngineLedger()
    aggregate_reads = dict(aggregate_reads or {})

    class FakeVariable:
        def __init__(self, entity: str) -> None:
            self.entity = SimpleNamespace(key=entity)

    class FakeSystem:
        variables = {
            name: FakeVariable(entity) for name, entity in _VARIABLE_ENTITIES.items()
        }

        def __init__(self, reform=None) -> None:
            self.reform = reform
            ledger.reform_systems.append(self)

    class FakeMicrosimulation(_FakeHolders):
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            super().__init__(stored_inputs)
            alive = [
                simulation
                for simulation in ledger.simulations
                if simulation.dataset is not None
            ]
            frame = dataset["frame"]
            ledger.constructions.append(
                {
                    "reform": reform,
                    "households": tuple(
                        int(value) for value in frame.table("household")["household_id"]
                    ),
                    "alive_before": len(alive),
                }
            )
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system
            self.branches: dict[str, _FakeHolders] = {}
            self.clone_branches = clone_branches
            self.baseline = _FakeHolders() if reform is not None else None
            if self.baseline is not None:
                self.baseline.clone_branches = clone_branches
            ledger.simulations.append(self)

        def _ids(self, entity: str) -> np.ndarray:
            return self.dataset["frame"].table(entity)[f"{entity}_id"].to_numpy()

        def _values(self, variable: str) -> np.ndarray:
            entity = _VARIABLE_ENTITIES[variable]
            ids = self._ids(entity)
            if self.reform is not None:
                assert variable == "income_tax"
                return np.asarray(
                    [
                        _record_value(variable, int(id_)) + 10.0 * (id_ % 7)
                        for id_ in ids
                    ]
                )
            return np.asarray([_record_value(variable, int(id_)) for id_ in ids])

        def calculate(self, variable, *, period, map_to=None):
            assert self.dataset is not None, "calculate on a released engine"
            self.record(variable, period)
            for aggregate in aggregate_reads.get(variable, ()):
                if aggregate_reform_only and self.reform is None:
                    continue
                if (
                    aggregate_household is not None
                    and aggregate_household not in self._ids("household")
                ):
                    continue
                holders = self.baseline if aggregate_in_baseline else self
                branch_parent = holders
                if aggregate_branch is not None:
                    for branch_name in aggregate_branch.split("/"):
                        holders = holders.get_branch(branch_name)
                holders.record(aggregate, period)
                if aggregate_delete_branch:
                    del branch_parent.branches[aggregate_branch.split("/")[0]]
            values = self._values(variable)
            entity = _VARIABLE_ENTITIES[variable]
            if map_to is None or map_to == entity:
                return values
            # Project the record values onto persons, then sum persons into
            # the requested entity: still within-household by construction.
            frame = self.dataset["frame"]
            person = frame.table("person")
            if entity == "person":
                person_values = values.astype(np.float64)
            else:
                positions = pd.Series(
                    np.arange(len(values)), index=self._ids(entity)
                ).reindex(person[f"person_{entity}_id"].to_numpy())
                person_values = values.astype(np.float64)[positions.to_numpy()]
            if map_to == "person":
                return person_values
            target_ids = self._ids(map_to)
            target_positions = pd.Series(
                np.arange(len(target_ids)), index=target_ids
            ).reindex(person[f"person_{map_to}_id"].to_numpy())
            out = np.zeros(len(target_ids), dtype=np.float64)
            np.add.at(out, target_positions.to_numpy(), person_values)
            return out

    def fake_dataset_from_frame(
        frame_arg,
        *,
        zero_variables=(),
        system=None,
        assert_no_formula_owned_columns=True,
        period_only=False,
    ):
        return {"frame": frame_arg, "zero_variables": tuple(zero_variables)}

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", lambda frame: None)
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder,
        "_make_zero_variable_reform",
        lambda system, variable_name: variable_name,
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", reform_specs)
    monkeypatch.setattr(
        builder,
        "SOI_VARIABLE_MAP",
        {
            "adjusted_gross_income": "adjusted_gross_income",
            "eitc": "eitc",
            "ctc": "ctc",
            "employment_income": "employment_income",
            "itemized_taxable_income_deductions": (
                "itemized_taxable_income_deductions"
            ),
            "rent_and_royalty_net_income": "rent_and_royalty_net_income",
            "rent_and_royalty_net_losses": "rent_and_royalty_net_income",
            "tax_filer_individual_count": "tax_unit_size",
        },
    )
    return ledger


# ---------------------------------------------------------------------------
# A small nested frame: five households, multi-member units, three states
# ---------------------------------------------------------------------------

# (household_id, state_fips, congressional_district_geoid) and, per person:
# (person_id, household, tax_unit, spm_unit, family, marital_unit).
_HOUSEHOLDS = (
    (1, 6, "0601"),
    (2, 36, "3601"),
    (3, 6, "0602"),
    (4, 24, "2401"),
    (5, 36, "3602"),
)
_PERSONS = (
    (1, 1, 10, 100, 1000, 10000),
    (2, 1, 10, 100, 1000, 10000),
    (3, 1, 11, 100, 1000, 10001),
    (4, 2, 20, 200, 2000, 20000),
    (5, 3, 30, 300, 3000, 30000),
    (6, 3, 30, 300, 3000, 30001),
    (7, 4, 40, 400, 4000, 40000),
    (8, 4, 40, 400, 4000, 40000),
    (9, 4, 41, 400, 4000, 40001),
    (10, 4, 42, 401, 4001, 40002),
    (11, 5, 50, 500, 5000, 50000),
)


def _nested_frame(persons=_PERSONS) -> Frame:
    person = pd.DataFrame(
        persons,
        columns=[
            "person_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        ],
    ).astype("int64")
    tables = {"person": person}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([row[0] for row in _HOUSEHOLDS], "int64"),
            "state_fips": np.asarray([row[1] for row in _HOUSEHOLDS], "int64"),
            "congressional_district_geoid": np.asarray(
                [row[2] for row in _HOUSEHOLDS], dtype=object
            ),
            "household_input": np.asarray([3.0, 1.0, 4.0, 1.0, 5.0]),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), WeightKind.DESIGN
            )
        },
    )


def _spec(name, family, metadata=None, *, measure=None, signed=True):
    return TargetSpec(
        name=name,
        entity="household",
        measure=measure or name,
        value=1.0,
        source="fixture",
        family=family,
        signed=signed,
        metadata=dict(metadata or {}),
    )


def _soi(name, variable, **metadata):
    return _spec(
        name,
        "irs_soi",
        {
            "variable": variable,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": name,
            **metadata,
        },
    )


def _variable(name, **metadata):
    return _spec(name, "fixture", {"materializer": "policyengine_variable", **metadata})


def _population_age(name, lower, upper, **metadata):
    return _spec(
        name,
        "census_population",
        {
            "materializer": "population_age",
            "measure_mode": "indicator_sum",
            "age_lower_bound": str(lower),
            "age_upper_bound": str(upper),
            **metadata,
        },
    )


# One target per branch of the base simulation, plus one JCT reform.
_TARGETS = (
    _soi("agi_amount", "adjusted_gross_income"),
    _soi(
        "agi_single_ca_bracket_count",
        "count",
        agi_lower_bound="-100",
        agi_upper_bound="900",
        filing_status="Single",
        state_fips="06",
    ),
    _soi(
        "agi_joint_count",
        "count",
        filing_status="Married Filing Jointly/Surviving Spouse",
    ),
    _soi("taxable_returns", "count", taxable_only="true"),
    _soi("eitc_two_children_amount", "eitc"),
    _soi(
        "eitc_returns_with_credit",
        "count",
        ledger_domain="individual_income_tax_returns_with_earned_income_credit",
    ),
    _soi("itemized_amount", "itemized_taxable_income_deductions", itemized_only="true"),
    _soi("ctc_amount", "ctc"),
    _soi("ctc_claims", "ctc", measure_mode="indicator_sum"),
    _soi("rent_royalty_net", "rent_and_royalty_net_income"),
    _soi("rent_royalty_losses", "rent_and_royalty_net_losses"),
    _soi(
        "cd_0601_wages",
        "employment_income",
        congressional_district_geoid="0601",
    ),
    _soi("filer_individuals", "tax_filer_individual_count"),
    _population_age("pop_under_18", 0, 18),
    _population_age("ca_pop_18_to_65", 18, 65, state_fips="06"),
    _population_age("cd_0601_pop", 0, "inf", congressional_district_geoid="0601"),
    _variable("ny_snap", base_variable="snap", state_fips="36"),
    _variable("ssi_recipients", base_variable="ssi", measure_mode="indicator_sum"),
    _variable(
        "low_net_income_snap_units",
        base_variable="spm_unit_net_income",
        measure_mode="less_than_indicator_sum",
        indicator_less_than="100",
        indicator_filter_variable="snap",
    ),
    _variable("rent_combined", base_variables="rental_income,farm_rent_income"),
    _variable(
        "ssi_household_indicator",
        base_variable="ssi",
        measure_mode="indicator_sum",
        indicator_map_to="household",
    ),
    _variable(
        "child_ssi_recipients",
        base_variable="ssi",
        measure_mode="indicator_sum",
        age_lower_bound="0",
        age_upper_bound="18",
    ),
    _variable(
        "cd_3601_household_benefits",
        base_variable="household_benefits",
        congressional_district_geoid="3601",
    ),
    _variable("family_income", base_variable="family_mock_income"),
    _variable("marital_income", base_variable="marital_unit_mock_income"),
    _spec("tanf", "hhs_tanf"),
    _spec("household_input", "fixture"),
    _spec("ca_state_income_tax", "state_income_tax", {"state_fips": "06"}),
    _spec("ny_state_income_tax", "state_income_tax", {"state_fips": "36"}),
    _spec("jct_mock_credit", "jct"),
)
_REFORMS = (
    SimpleNamespace(measure="jct_mock_credit", neutralized_variable="mock_credit"),
)


def _materialize(builder, monkeypatch, frame, batch_size):
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    target_frame, registry, compilation = builder._materialize_target_frame(
        frame,
        _TARGETS,
        maximum_microsim_batch_size=batch_size,
    )
    return target_frame, registry, compilation, ledger


_AGGREGATE_PROBE_TARGETS = (
    _variable("aggregate_probe_total", base_variable="aggregate_probe"),
)


def _second_batch_drops_a_column(real):
    calls = [0]

    def columns(*args, **kwargs):
        calls[0] += 1
        result = real(*args, **kwargs)
        if calls[0] == 2:
            result.pop("income_tax")
        return result

    return columns


def _one_value_short(real):
    def columns(*args, **kwargs):
        result = real(*args, **kwargs)
        result["income_tax"] = result["income_tax"][:-1]
        return result

    return columns


# ---------------------------------------------------------------------------
# Real engine (requires the [us] extra)
# ---------------------------------------------------------------------------


def _real_engine_frame() -> Frame:
    """Four households over three states; two share a household across tax
    units, and every group nests in its household."""

    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 8, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 3, 3, 3, 4], "int64"),
            "person_tax_unit_id": np.asarray([10, 11, 20, 30, 31, 32, 40], "int64"),
            "person_spm_unit_id": np.asarray(
                [100, 100, 200, 300, 300, 301, 400], "int64"
            ),
            "person_family_id": np.asarray(
                [1000, 1000, 2000, 3000, 3000, 3001, 4000], "int64"
            ),
            "person_marital_unit_id": np.asarray(
                [10000, 10001, 20000, 30000, 30001, 30002, 40000], "int64"
            ),
            "age": np.asarray([34.0, 71.0, 45.0, 29.0, 67.0, 19.0, 52.0]),
            "employment_income_before_lsr": np.asarray(
                [48_000.0, 0.0, 180_000.0, 9_000.0, 0.0, 14_000.0, 72_000.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([1, 2, 3, 4], "int64"),
            "state_fips": np.asarray([6, 36, 24, 6], "int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(4, dtype=np.float64), WeightKind.DESIGN)},
    )


def _medicaid_frame() -> Frame:
    """Four California parents with children; the first parent's earnings
    exceed the other three parents' earnings."""

    n_households = 4
    household_ids = np.arange(1, n_households + 1, dtype="int64")
    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n_households + 1, dtype="int64"),
            "person_household_id": np.repeat(household_ids, 2),
            "person_tax_unit_id": np.repeat(household_ids * 10, 2),
            "person_spm_unit_id": np.repeat(household_ids * 100, 2),
            "person_family_id": np.repeat(household_ids * 1000, 2),
            "person_marital_unit_id": np.arange(1, 2 * n_households + 1, dtype="int64")
            * 10_000,
            "age": np.tile([30.0, 5.0], n_households),
            "employment_income_before_lsr": np.asarray(
                [400_000.0, 0.0, 12_000.0, 0.0, 15_000.0, 0.0, 5_000.0, 0.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": household_ids,
            "state_fips": np.full(n_households, 6, dtype="int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([1.0, 2.0, 3.0, 4.0]), WeightKind.DESIGN)},
    )


_POPULATION_AGGREGATE_MARKER = re.compile(
    r"\bsum_by_state\(|\bMicroSeries\(|\.decile_rank\(|\bquantile\(|\bpercentile\("
    r"|\bnp\.(?:sum|mean|median|average|nansum|nanmean|nanmedian|sort|argsort"
    r"|cumsum|isin|in1d|unique|bincount|searchsorted)\((?![^\n]*\baxis\s*=)"
    r"|\.(?:sum|mean|max|min|median)\(\s*\)"
)
_WEIGHT_READ = re.compile(r"[\"'](\w+_weight)[\"']")


def _is_variable_class(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        getattr(base, "id", None) == "Variable" for base in node.bases
    )


def _engine_population_aggregate_sources(
    root: Path | None = None,
    *,
    source_digests: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Scan package Variable classes and module helpers for aggregation markers.

    Pattern matches identify formulas to inspect; they do not prove household
    locality or recognize every possible cross-record computation.
    """
    if root is None:
        spec = importlib.util.find_spec("policyengine_us")
        root = Path(next(iter(spec.submodule_search_locations)))
    modules = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or path.name.startswith("test_"):
            continue
        source = path.read_text()
        modules.append((source, ast.parse(source)))

    helpers: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                helpers.setdefault(node.name, []).append(
                    ast.get_source_segment(source, node)
                )
    reaching = {
        name
        for name, bodies in helpers.items()
        if any(_POPULATION_AGGREGATE_MARKER.search(body) for body in bodies)
    }
    grew = True
    while grew:
        grew = False
        for name, bodies in helpers.items():
            if name not in reaching and any(
                re.search(rf"\b{other}\(", body)
                for body in bodies
                for other in reaching
            ):
                reaching.add(name)
                grew = True

    found: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in ast.walk(tree):
            if not _is_variable_class(node):
                continue
            body = ast.get_source_segment(source, node)
            evidence = [
                match.group(0) for match in _POPULATION_AGGREGATE_MARKER.finditer(body)
            ]
            evidence += [
                f"{helper}()"
                for helper in sorted(reaching)
                if re.search(rf"\b{helper}\(", body)
            ]
            if not node.name.endswith("_weight"):
                evidence += _WEIGHT_READ.findall(body)
            if evidence:
                found[node.name] = evidence
                if source_digests is not None:
                    sources = [body]
                    pending = [
                        helper
                        for helper in reaching
                        if re.search(rf"\b{helper}\(", body)
                    ]
                    seen = set()
                    while pending:
                        helper = pending.pop()
                        if helper in seen:
                            continue
                        seen.add(helper)
                        sources.extend(helpers[helper])
                        pending.extend(
                            other
                            for helper_body in helpers[helper]
                            for other in reaching - seen
                            if re.search(rf"\b{other}\(", helper_body)
                        )
                    source_digests[node.name] = hashlib.sha256(
                        "\n".join(sorted(sources)).encode()
                    ).hexdigest()
    return found


# A period offset whose amount is not negative (a later month or year), and a
# holder lookup, which can read any stored period.
_FORWARD_PERIOD_READ = re.compile(r"\.offset\(\s*(?!-)[\w.]+")
_HOLDER_LOOKUP = re.compile(
    r"\b(?:get_known_periods|get_holder|get_array)\(|\b_holders\b"
)


def _engine_forward_read_sources(
    root: Path | None = None,
    *,
    source_digests: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Scan package Variable classes and module helpers for later-period reads.

    A forward offset or a holder lookup marks a formula to inspect: it may read
    a year after the one asked, which a dataset cut at that year does not hold.
    A match proves nothing either way, and a formula can build a later period
    in other ways.
    """
    if root is None:
        spec = importlib.util.find_spec("policyengine_us")
        root = Path(next(iter(spec.submodule_search_locations)))
    modules = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or path.name.startswith("test_"):
            continue
        source = path.read_text()
        modules.append((source, ast.parse(source)))

    def markers(body: str) -> list[str]:
        return [
            match.group(0)
            for pattern in (_FORWARD_PERIOD_READ, _HOLDER_LOOKUP)
            for match in pattern.finditer(body)
        ]

    helpers: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                helpers.setdefault(node.name, []).append(
                    ast.get_source_segment(source, node)
                )
    reaching = {name for name, bodies in helpers.items() if any(map(markers, bodies))}
    grew = True
    while grew:
        grew = False
        for name, bodies in helpers.items():
            if name not in reaching and any(
                re.search(rf"\b{other}\(", body)
                for body in bodies
                for other in reaching
            ):
                reaching.add(name)
                grew = True

    found: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in ast.walk(tree):
            if not _is_variable_class(node):
                continue
            body = ast.get_source_segment(source, node)
            called = sorted(
                helper for helper in reaching if re.search(rf"\b{helper}\(", body)
            )
            evidence = markers(body) + [f"{helper}()" for helper in called]
            if not evidence:
                continue
            found[node.name] = evidence
            if source_digests is not None:
                sources = [body] + [
                    text for helper in called for text in helpers[helper]
                ]
                source_digests[node.name] = hashlib.sha256(
                    "\n".join(sorted(sources)).encode()
                ).hexdigest()
    return found


# These inspected formulas (read on policyengine-us 2.2.1 and again on 2.145.3)
# compare a record's status or geography with parameter lists, literal lists or
# enum members, rather than another record.
_CONSTANT_MEMBERSHIP_FORMULAS = {
    "additional_senior_deduction_eligible_person",
    "ak_ccap_rate_region",
    "ak_permanent_fund_dividend_eligible",
    "al_ccsp_region",
    "american_worker_tax_rebate_eligible",
    "ar_sra_zone",
    "ca_ala_general_assistance_immigration_status_eligible",
    "ca_calworks_child_care_immigration_status_eligible_person",
    "ca_cc_general_assistance_immigration_status_eligible",
    "ca_liheap_covered_liquid_heating",
    "ca_liheap_energy_burden_group",
    "ca_liheap_utility_eligible",
    "ca_liheap_utility_heating",
    "ca_marin_general_relief_immigration_status_eligible_person",
    "ca_oc_general_relief_immigration_status_eligible",
    "ca_riv_general_relief_immigration_status_eligible",
    "ca_sbd_general_relief_immigration_status_eligible",
    "ca_scc_general_assistance_immigration_status_eligible",
    "ca_sf_caap_immigration_status_eligible",
    "ca_smc_general_assistance_immigration_status_eligible_person",
    "ca_snap_immigration_status_eligible",
    "ca_tanf_immigration_status_eligible_person",
    "ca_tanf_region1",
    "co_ccap_fpg_eligible",
    "ct_c4k_region",
    "dc_ccsp_immigration_status_eligible_person",
    "dc_ccsp_is_full_time",
    "dc_tanf_immigration_status_eligible_person",
    "ga_caps_zone",
    "id_iccp_county_cluster",
    "id_liheap_heating_area",
    "il_aabd_area",
    "il_aabd_immigration_status_eligible_person",
    "il_ccap_county_group",
    "il_dhs_csfp_county_eligible",
    "il_hfs_immigration_status_eligible",
    "il_tanf_county_group",
    "il_tanf_immigration_status_eligible_person",
    "in_eap",
    "in_ny_mctd_zone_2",
    "in_nyc",
    "is_aca_ptc_immigration_status_eligible",
    "is_basic_health_program_eligible",
    "is_basic_health_program_immigration_status_eligible",
    "is_ca_medicaid_immigration_status_eligible",
    "is_ccdf_immigration_eligible_child",
    "is_chip_immigration_status_eligible",
    "is_citizen_or_legal_immigrant",
    "is_federal_medicaid_immigration_status_eligible",
    "is_medicaid_lawfully_residing_option_eligible",
    "is_medicare_eligible",
    "is_snap_gross_test_full_income_count_alien",
    "is_snap_immigration_status_eligible",
    "is_snap_prorated_income_member",
    "is_snap_state_discretion_ineligible_alien",
    "is_ssi_qualified_noncitizen",
    "ks_ccap_rate_group",
    "ks_dcf_csfp_county_eligible",
    "ks_tanf_county_group",
    "ky_ccap_rate_region",
    "local_sales_tax",
    "local_sales_tax_table",
    "ma_ccfa_immigration_status_eligible",
    "ma_ccfa_region",
    "ma_dese_csfp_county_eligible",
    "md_ccs_region",
    "md_meap",
    "me_ccap_region",
    "me_heap_points",
    "meets_ctc_child_identification_requirements",
    "meets_ctc_identification_requirements",
    "meets_snap_qualified_alien_waiting_period",
    "mo_ccs_region",
    "mo_ssp_immigration_status_eligible",
    "ms_ccpp_facility_location",
    "mt_tanf_immigration_status_eligible_person",
    "ne_child_care_subsidy_location",
    "nh_ccap_immigration_status_eligible_person",
    "nh_fap_heating_cost",
    "nh_ssp_eligible",
    "nj_liheap_region",
    "nv_liheap_immigration_eligible",
    "ny_ccap_county_group",
    "ny_ssp",
    "oh_ccap_county_rate_category",
    "or_healthier_oregon_immigration_status_eligible",
    "or_liheap_region",
    "overtime_income_deduction_eligible_person",
    "pa_ccw_region",
    "pa_ccw_stepparent_county_group",
    "pa_liheap_matrix_amount",
    "pa_tanf_county_group",
    "sc_ccap_geography",
    "sc_liheap",
    "sd_cca_hourly_rate",
    "sd_cca_region",
    "sd_liheap_region",
    "snap_allotment_before_eligibility",
    "state_group",
    "state_itemized_deductions",
    "state_standard_deduction",
    "taxsim_state_agi",
    "tip_income_deduction_eligible_person",
    "tn_ccap_county_tier",
    "trump_dividend_eligible",
    "tx_ccs_workforce_board_region",
    "ut_heat",
    "ut_heat_eligible_person",
    "va_ag_maximum_care_rate",
    "va_ccsp_income_eligible",
    "va_ccsp_locality_group",
    "va_ccsp_ready_region",
    "va_eap_climate_zone",
    "va_eap_eligible",
    "va_medicaid_lifc_locality_group",
    "va_tanf",
    "va_tanf_grant_standard",
    "va_tanf_need_standard",
    "va_tanf_up_grant_standard",
    "wa_rca_immigration_status_eligible",
    "wa_tanf_immigration_status_eligible",
    "wa_wccc_center_region",
    "wa_wccc_region",
    "wi_cts_area_i",
    "wi_cts_child_eligible",
}

# policyengine-us 2.145.3 (US engine refresh, 17 Nov 2026). Each class below
# was read in the installed sources; none reads another household's records.

#: The fpg() guideline helper checks each record's state group against the
#: guideline table's areas (np.isin on parameter keys).
_FPG_GUIDELINE_FORMULAS = frozenset(
    {
        "ak_hap_income_factor",
        "ar_liheap_income_limit",
        "az_liheap_income_limit",
        "ca_liheap_income_eligible",
        "ca_liheap_income_group",
        "ca_riv_liheap_eligible",
        "co_ccap_fpg",
        "co_leap_eligible",
        "co_leap_income_contribution_rate",
        "co_leap_income_limit",
        "ct_ceap_benefit_level",
        "ct_ceap_income_eligible",
        "dc_liheap_eligible",
        "de_liheap_income_eligible",
        "fl_liheap",
        "fl_liheap_eligible",
        "fl_liheap_income_limit",
        "fl_medicaid_parent_income_limit",
        "ga_liheap_income_eligible",
        "hi_liheap_income_limit",
        "ia_liheap_fpg",
        "id_liheap_income_limit",
        "il_fpp_income_level",
        "il_ihwap_income_eligible",
        "in_eap_eligible",
        "ks_liheap_income_limit",
        "ks_tanf_gross_income_eligible",
        "ky_liheap_fpg",
        "la_liheap_income_eligible",
        "ma_liheap_fpg",
        "md_meap_fpg",
        "md_poverty_line_credit_income_level",
        "me_heap_income_limit",
        "me_heap_income_multiplier",
        "medicaid_income_level",
        "medicaid_medically_needy_income_limit",
        "medicaid_optional_senior_or_disabled_income_limit",
        "mn_eap_income_limit",
        "mo_chip_premium",
        "mo_liheap_eligible",
        "mo_mhf_parent_income_limit",
        "ms_liheap_eligible",
        "ms_medicaid_parent_income_limit",
        "mt_liheap_income_limit",
        "nc_lieap_income_limit",
        "nd_liheap_share",
        "ne_liheap_fpg",
        "ne_liheap_payment_fpg_ratio",
        "nh_fap",
        "nh_fap_eligible",
        "nj_liheap_eligible",
        "nm_liheap",
        "nm_liheap_income_limit",
        "nv_liheap",
        "nv_liheap_income_limit",
        "ny_heap_fpg",
        "oh_liheap_eligible",
        "oh_liheap_income_limit",
        "ok_liheap_income_limit",
        "or_erdc_income_eligible",
        "or_liheap_eligible",
        "or_liheap_income_limit",
        "pa_liheap_income_limit",
        "ri_liheap_eligible",
        "ri_liheap_income_band",
        "sc_ccap_copay",
        "sc_liheap_eligible",
        "sd_liheap_income_limit",
        "sd_liheap_income_threshold",
        "spm_unit_fpg",
        "tax_unit_fpg",
        "tax_unit_medicaid_income_level",
        "tn_liheap",
        "tn_liheap_eligible",
        "tn_liheap_income_limit",
        "tn_medicaid_parent_income_limit",
        "tx_medicaid_parent_income_limit",
        "ut_heat_fpg",
        "ut_medicaid_parent_income_limit",
        "va_medicaid_lifc_income_limit",
        "vt_liheap_fpg",
        "vt_liheap_income_limit",
        "wa_liheap_income_limit",
        "wi_wheap",
        "wi_wheap_eligible",
        "wv_lieap_income_limit",
        "wy_lieap_fpg",
        "wy_medicaid_parent_income_limit",
    }
)

#: Formulas that reach fpg() or a constant list through a program helper
#: (Medicaid income levels, LIHEAP and SNAP income standards, the SNAP waiver
#: areas, Alaska HAP places), with their evidence.
_GUIDELINE_HELPER_FORMULAS: dict[str, frozenset[str]] = {
    "aca_fpg": frozenset({"fpg()", "np.isin("}),
    "ak_hap": frozenset({"ak_hap_award()"}),
    "ak_hap_base_benefit": frozenset({"ak_hap_award()"}),
    "al_liheap_cooling_income_limit": frozenset({"al_liheap_income_limit_for_lag()"}),
    "al_liheap_income_limit": frozenset({"al_liheap_income_limit_for_lag()"}),
    "ca_cchip_eligible": frozenset({"medicaid_income_eligible()", "np.isin("}),
    "hi_liheap_points": frozenset({"fpg()", "np.isin("}),
    "is_adult_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_chip_eligible_child": frozenset({"medicaid_income_eligible()"}),
    "is_chip_eligible_standard_pregnant_person": frozenset(
        {"medicaid_income_eligible()"}
    ),
    "is_chip_fcep_eligible_person": frozenset(
        {"medicaid_income_eligible()", "np.isin("}
    ),
    "is_in_snap_abawd_waived_area": frozenset({"in_listed_abawd_waiver_area()"}),
    "is_infant_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_older_child_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_parent_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_pregnant_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_young_adult_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "is_young_child_for_medicaid_fc": frozenset({"medicaid_income_eligible()"}),
    "ky_liheap_cooling_eligible": frozenset({"ky_liheap_meets_income_limit()"}),
    "ky_liheap_income_eligible": frozenset({"ky_liheap_meets_income_limit()"}),
    "meets_snap_gross_income_test": frozenset({"snap_monthly_income_standard()"}),
    "meets_snap_net_income_test": frozenset({"snap_monthly_income_standard()"}),
    "meets_tanf_non_cash_net_income_test": frozenset(
        {"snap_monthly_income_standard()"}
    ),
    "mt_liheap_multiplier": frozenset({"fpg()", "np.isin("}),
    "oh_liheap": frozenset({"fpg()", "np.isin("}),
    "oh_owf_fpg": frozenset({"fpg()", "np.isin("}),
    "ri_medicaid_state_funded_postpartum_eligible": frozenset(
        {"medicaid_income_eligible()"}
    ),
    "snap_fpg": frozenset(
        {"_snap_fpg_context()", "fpg()", "np.isin(", "snap_monthly_fpg_amounts()"}
    ),
    "va_eap_cooling_income_limit": frozenset({"va_eap_annual_income_limit()"}),
    "va_eap_income_limit": frozenset({"va_eap_annual_income_limit()"}),
    "wic_income_limit": frozenset({"fpg()", "np.isin("}),
}

#: Social Security helpers (policyengine_us.tools.social_security*): each
#: record's own arithmetic. np.unique groups records by eligibility year, and
#: a COLA override is checked against the constants -1, 0 and 1. A worker link
#: needs a *_worker_id input, which no Microcosm frame stores; an id outside
#: the batch raises rather than reading another household.
_SOCIAL_SECURITY_HELPER_FORMULAS: dict[str, frozenset[str]] = {
    "social_security_dependents_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_dependents_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_dependents_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_disability_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_disability_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_retirement_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_retirement_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_survivors_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_survivors_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "ss_calculated_aime_lower": frozenset({"calculator_output()"}),
    "ss_calculated_aime_upper": frozenset({"calculator_output()"}),
    "ss_calculated_annual_entitlement": frozenset({"calculator_output()"}),
    "ss_calculated_annual_payment": frozenset({"calculator_output()"}),
    "ss_calculated_benefit": frozenset({"calculator_output()"}),
    "ss_calculated_deferred_earnings_test_payment": frozenset({"calculator_output()"}),
    "ss_calculated_earnings_test_reduction": frozenset({"calculator_output()"}),
    "ss_calculated_eligibility_pia": frozenset({"calculator_output()"}),
    "ss_calculated_history_aime": frozenset({"calculator_output()"}),
    "ss_calculated_history_aime_plus": frozenset({"calculator_output()"}),
    "ss_calculated_history_conventional_pia": frozenset({"calculator_output()"}),
    "ss_calculated_history_ledger": frozenset({"calculator_output()"}),
    "ss_calculated_history_mini_pia": frozenset({"calculator_output()"}),
    "ss_calculated_payment": frozenset({"calculator_output()"}),
    "ss_calculated_pia_lower": frozenset({"calculator_output()"}),
    "ss_calculated_pia_upper": frozenset({"calculator_output()"}),
    "ss_calculated_retirement": frozenset({"calculator_output()"}),
    "ss_calculated_survivors": frozenset({"calculator_output()"}),
}

#: Parent links (policyengine-us _parent_links.py) resolve parent_1_id and
#: parent_2_id among the person's household (or tax unit) members only, so
#: np.bincount counts co-residents; SSI deeming pools and Wisconsin CTS units
#: are keyed by co-resident rows.
_PARENT_LINK_FORMULAS: dict[str, frozenset[str]] = {
    "ca_capi_countable_income": frozenset({"ssi_parental_deemed_income()"}),
    "co_state_supplement_deemed_income": frozenset({"unlinked_parent()"}),
    "ia_ssa_income_deemed_from_parents": frozenset(
        {"_ssi_parental_resource_pool()", "np.bincount("}
    ),
    "ia_ssa_resources_deemed_from_parents": frozenset(
        {"_ssi_parental_resource_pool()", "np.bincount("}
    ),
    "is_optional_senior_or_disabled_asset_eligible": frozenset(
        {"_ssi_parental_resource_pool()", "unlinked_parent()"}
    ),
    "is_parent": frozenset({"linked_child_count()"}),
    "is_ssi_ineligible_parent": frozenset({"is_parent_of_any()"}),
    "medicaid_optional_senior_or_disabled_unearned_income_deemed_from_ineligible_parent_monthly": frozenset(
        {"is_parent_of_any()", "ssi_parental_child_allocation_total()"}
    ),
    "medicaid_parental_medical_expense_share": frozenset({"is_parent_of_any()"}),
    "medicaid_tax_dependent_exception_living_with_both_parents": frozenset(
        {"unlinked_parent()"}
    ),
    "mo_tanf_is_parent_of_dependent_child": frozenset({"np.bincount("}),
    "nh_ssp_resources_deemed_from_parents": frozenset(
        {"_ssi_parental_resource_pool()"}
    ),
    "nh_ssp_unearned_income_deemed_from_ineligible_parent": frozenset(
        {"ssi_parental_deemed_income()"}
    ),
    "ok_ssp_unearned_income_deemed_from_ineligible_parent": frozenset(
        {"ssi_parental_deemed_income()"}
    ),
    "ssi_resources_deemed_from_ineligible_parent": frozenset(
        {"_ssi_parental_resource_pool()", "np.bincount("}
    ),
    "ssi_spousal_deeming_child_allocations": frozenset({"unlinked_parent()"}),
    "ssi_unearned_income_deemed_from_ineligible_parent_monthly": frozenset(
        {"ssi_parental_deemed_income()"}
    ),
    "wi_cts": frozenset({"cts_filing_unit()", "cts_financial_test()"}),
    "wi_cts_buening_member": frozenset({"cts_filing_unit()"}),
    "wi_cts_caretaker_children": frozenset({"np.bincount("}),
    "wi_cts_sfu_member": frozenset({"cts_filing_unit()"}),
    "wi_works_custodial_parent": frozenset({"np.bincount("}),
}

#: Population-weighted sales-tax rates by locality: group sums over the
#: packaged rate schedule, not over records.
_PACKAGE_DATA_FORMULAS: dict[str, frozenset[str]] = {
    "combined_sales_tax_rate": frozenset({"locality_sales_tax_rates()"}),
    "local_sales_tax_rate": frozenset({"state_general_sales_tax_rates()"}),
}

#: Same-period reads: whether the period asked holds a supplied input
#: (has_input_for_period; annual_input for the year holding the month asked),
#: branch bookkeeping that drops calculated arrays and sets overrides at the
#: period asked, and reads of supplied inputs at that period.
_SAME_PERIOD_READ_FORMULAS: dict[str, frozenset[str]] = {
    "aca_ptc_coverage_family_premium": frozenset(
        {"benchmark_premium()", "has_input_for_period()"}
    ),
    "aca_ptc_slcsp": frozenset({"benchmark_premium()"}),
    "aca_ptc_slcsp_family_tier_category": frozenset({"family_tier_category()"}),
    "aca_ptc_slcsp_family_tier_multiplier": frozenset({"family_tier_multiplier()"}),
    "al_federal_income_tax_deduction": frozenset({"get_branch_for_period()"}),
    "ca_capi_countable_income": frozenset({"annual_input()"}),
    "ca_wdp_income_eligible": frozenset({"ca_wdp_annual_spousal_deeming_applies()"}),
    "ca_wdp_ssi_ssp_income_eligible": frozenset(
        {"ca_wdp_annual_spousal_deeming_applies()"}
    ),
    "capital_gains_behavioral_response": frozenset(
        {"get_behavioral_response_measurements()"}
    ),
    "co_adult_financial_medicaid_enrolled": frozenset({"get_override_branch()"}),
    "combined_sales_tax_rate": frozenset({"sales_tax_location()"}),
    "de_income_tax_if_claiming_non_refundable_eitc": frozenset(
        {"get_override_branch()"}
    ),
    "de_income_tax_if_claiming_refundable_eitc": frozenset({"get_override_branch()"}),
    "dependent_traditional_ira_deduction": frozenset(
        {"traditional_ira_contributions_without_spousal_rule()"}
    ),
    "federal_marginal_tax_rate": frozenset({"compute_component_mtr()"}),
    "fica_marginal_tax_rate": frozenset({"compute_component_mtr()"}),
    "hi_income_tax_if_claiming_claim_of_right_credit": frozenset(
        {"get_override_branch()"}
    ),
    "hi_income_tax_if_claiming_claim_of_right_deduction": frozenset(
        {"get_override_branch()"}
    ),
    "id_health_insurance_premiums_medical_deduction": frozenset(
        {"has_input_for_period()"}
    ),
    "id_income_tax_if_receiving_aged_or_disabled_credit": frozenset(
        {"get_override_branch()"}
    ),
    "id_income_tax_if_receiving_aged_or_disabled_deduction": frozenset(
        {"get_override_branch()"}
    ),
    "id_itemized_deductions": frozenset({"has_input_for_period()"}),
    "income_elasticity_lsr": frozenset({"calculate_income_lsr_effect()"}),
    "income_tax_if_claiming_claim_of_right_credit": frozenset(
        {"get_override_branch()"}
    ),
    "income_tax_if_claiming_claim_of_right_deduction": frozenset(
        {"get_override_branch()"}
    ),
    "income_tax_with_pre_1993_social_security": frozenset(
        {"income_tax_with_social_security_override()"}
    ),
    "income_tax_without_social_security": frozenset(
        {"income_tax_with_social_security_override()"}
    ),
    "ira_contribution_limit": frozenset({"supplied_ira_contributions()"}),
    "ira_contribution_scale": frozenset({"supplied_ira_contributions()"}),
    "is_aca_filer": frozenset({"has_input_for_period()"}),
    "is_aca_marketplace_enrollee": frozenset({"has_input_for_period()"}),
    "is_ssi_ineligible_parent": frozenset(
        {
            "_ssi_parent_supplement_level()",
            "ssi_budget_spousal_deeming_applies()",
            "ssi_income_deemed_from_ineligible_spouse_without_ism()",
        }
    ),
    "is_ssi_spousal_deeming_applies_monthly": frozenset({"annual_input()"}),
    "is_tax_unit_head": frozenset({"head_or_spouse_candidates()"}),
    "is_tax_unit_spouse": frozenset({"head_or_spouse_candidates()"}),
    "itemized_medical_expenses_applied_exclusion": frozenset(
        {"has_input_for_period()"}
    ),
    "itemized_medical_expenses_long_term_care_premiums": frozenset(
        {"has_input_for_period()"}
    ),
    "labor_supply_behavioral_response": frozenset(
        {
            "calculate_income_lsr_effect()",
            "calculate_substitution_lsr_effect()",
            "get_behavioral_response_measurements()",
        }
    ),
    "local_sales_tax_table": frozenset({"sales_tax_location()"}),
    "marginal_tax_rate": frozenset({"marginal_earnings_change()"}),
    "marginal_tax_rate_including_health_benefits": frozenset(
        {"marginal_earnings_change()"}
    ),
    "marginal_tax_rate_on_capital_gains": frozenset(
        {"create_perturbed_branch()", "has_input()"}
    ),
    "medicaid_enrolled_for_ssi_state_supplement": frozenset(
        {"get_array(", "get_override_branch()"}
    ),
    "medicaid_medically_needy_uses_couple_standard": frozenset(
        {"uses_couple_standard()"}
    ),
    "medicaid_optional_senior_or_disabled_countable_income_monthly": frozenset(
        {"annual_input()"}
    ),
    "medicaid_optional_senior_or_disabled_income_deemed_from_ineligible_spouse_monthly": frozenset(
        {"annual_input()"}
    ),
    "medicaid_optional_senior_or_disabled_unearned_income_deemed_from_ineligible_parent_monthly": frozenset(
        {"annual_input()"}
    ),
    "medicaid_optional_senior_or_disabled_uses_couple_standard": frozenset(
        {"uses_couple_standard()"}
    ),
    "mo_tanf_if_non_parent_caretaker_excluded": frozenset({"get_override_branch()"}),
    "mo_tanf_if_non_parent_caretaker_included": frozenset({"get_override_branch()"}),
    "mt_medical_expense_deduction_joint": frozenset({"has_input_for_period()"}),
    "nd_renters_refund_income": frozenset({"has_input_for_period()"}),
    "nh_ssp_unearned_income_deemed_from_ineligible_parent": frozenset(
        {"annual_input()"}
    ),
    "nj_medical_expense_deduction": frozenset({"has_input_for_period()"}),
    "nm_medical_expense_credit": frozenset({"has_input_for_period()"}),
    "nm_medical_expense_exemption": frozenset({"has_input_for_period()"}),
    "ny_ctc_pre_2024": frozenset(
        {"get_branch_for_period()", "get_ny_recomputed_branch()"}
    ),
    "ny_ctc_pre_2024_eligible": frozenset({"get_branch_for_period()"}),
    "ny_eitc": frozenset({"get_branch_for_period()", "get_ny_recomputed_branch()"}),
    "nyc_eitc": frozenset({"get_ny_recomputed_branch()"}),
    "ok_ssp_unearned_income_deemed_from_ineligible_parent": frozenset(
        {"annual_input()"}
    ),
    "owners_draw_active_income": frozenset({"_draw()"}),
    "owners_draw_credited_earnings": frozenset({"_draw()"}),
    "owners_draw_gross": frozenset({"_draw()"}),
    "owners_draw_hi_tax": frozenset({"_draw()"}),
    "owners_draw_net_earnings": frozenset({"_draw()"}),
    "owners_draw_net_earnings_factor": frozenset({"_draw()"}),
    "owners_draw_niit_exclusion": frozenset({"_draw()"}),
    "owners_draw_oasdi_base": frozenset({"_draw()"}),
    "owners_draw_oasdi_room": frozenset({"_draw()"}),
    "owners_draw_oasdi_tax": frozenset({"_draw()"}),
    "owners_draw_planned_gross": frozenset({"_draw()"}),
    "relative_capital_gains_mtr_change": frozenset(
        {"get_behavioral_response_measurements()"}
    ),
    "relative_income_change": frozenset({"get_behavioral_response_measurements()"}),
    "relative_wage_change": frozenset({"get_behavioral_response_measurements()"}),
    "slcsp_age_curve_amount_person_uncapped": frozenset({"has_input_for_period()"}),
    "slcsp_family_tier_category": frozenset({"family_tier_category()"}),
    "slcsp_family_tier_multiplier": frozenset({"family_tier_multiplier()"}),
    "ssi_countable_income_monthly": frozenset({"annual_input()"}),
    "ssi_earned_income_deemed_from_ineligible_spouse_monthly": frozenset(
        {"annual_input()"}
    ),
    "ssi_in_kind_support_and_maintenance_monthly": frozenset({"annual_input()"}),
    "ssi_income_deemed_from_ineligible_spouse_monthly": frozenset({"annual_input()"}),
    "ssi_ineligible_child_allocation_monthly": frozenset({"annual_input()"}),
    "ssi_ineligible_parent_allocation": frozenset({"_ssi_parent_supplement_level()"}),
    "ssi_pmv_amount_monthly": frozenset({"annual_input()"}),
    "ssi_unearned_income_deemed_from_ineligible_parent_monthly": frozenset(
        {"annual_input()"}
    ),
    "ssi_unearned_income_deemed_from_ineligible_spouse_monthly": frozenset(
        {"annual_input()"}
    ),
    "state_marginal_tax_rate": frozenset({"compute_component_mtr()"}),
    "substitution_elasticity_lsr": frozenset({"calculate_substitution_lsr_effect()"}),
    "tax_liability_if_itemizing": frozenset({"get_itemization_branch()"}),
    "tax_liability_if_not_itemizing": frozenset({"get_itemization_branch()"}),
    "tax_unit_itemizes": frozenset({"get_itemization_branch()"}),
    "va_income_tax_if_claiming_non_refundable_eitc": frozenset(
        {"get_override_branch()"}
    ),
    "va_income_tax_if_claiming_refundable_eitc": frozenset({"get_override_branch()"}),
}

#: Social Security reported and calculator helpers: same-period input checks.
#: The calculator may also check a supplied ss_calculator_* record for the
#: next payment year; no Microcosm frame stores one.
_SOCIAL_SECURITY_PERIOD_FORMULAS: dict[str, frozenset[str]] = {
    "social_security_dependents_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_dependents_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_dependents_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_dependents_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_disability_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_disability_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_disability_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_retirement_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_retirement_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_retirement_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_cola_means_test_withheld": frozenset(
        {"component_withholding()"}
    ),
    "social_security_survivors_inferred_aime": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_inferred_aime_lower_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_inferred_aime_upper_bound": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_joint_cola_factor": frozenset(
        {"reported_joint_cola_factor()"}
    ),
    "social_security_survivors_pia_ratio_claiming_adjustment": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_eligibility_year": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_factor": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_measurement_uncertainty": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_missing_claim_age": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_own_record_retired_worker": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_pre_1983": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_proxy": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "social_security_survivors_pia_ratio_unmodeled": frozenset(
        {"reported_pia_ratio_output()"}
    ),
    "ss_calculated_aime_lower": frozenset({"calculator_output()"}),
    "ss_calculated_aime_upper": frozenset({"calculator_output()"}),
    "ss_calculated_annual_entitlement": frozenset({"calculator_output()"}),
    "ss_calculated_annual_payment": frozenset({"calculator_output()"}),
    "ss_calculated_benefit": frozenset({"calculator_output()"}),
    "ss_calculated_deferred_earnings_test_payment": frozenset({"calculator_output()"}),
    "ss_calculated_earnings_test_reduction": frozenset({"calculator_output()"}),
    "ss_calculated_eligibility_pia": frozenset({"calculator_output()"}),
    "ss_calculated_history_aime": frozenset({"calculator_output()"}),
    "ss_calculated_history_aime_plus": frozenset({"calculator_output()"}),
    "ss_calculated_history_conventional_pia": frozenset({"calculator_output()"}),
    "ss_calculated_history_ledger": frozenset({"calculator_output()"}),
    "ss_calculated_history_mini_pia": frozenset({"calculator_output()"}),
    "ss_calculated_payment": frozenset({"calculator_output()"}),
    "ss_calculated_pia_lower": frozenset({"calculator_output()"}),
    "ss_calculated_pia_upper": frozenset({"calculator_output()"}),
    "ss_calculated_retirement": frozenset({"calculator_output()"}),
    "ss_calculated_survivors": frozenset({"calculator_output()"}),
}

__all__ = [name for name in globals() if not name.startswith("__")]
