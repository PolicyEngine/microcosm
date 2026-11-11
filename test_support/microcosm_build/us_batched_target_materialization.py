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




# These inspected 2.2.1 formulas compare a record's status or geography with
# parameter lists, literal lists or enum members, rather than another record.
_CONSTANT_MEMBERSHIP_FORMULAS = {
    "additional_senior_deduction_eligible_person",
    "ak_ccap_rate_region",
    "al_ccsp_region",
    "american_worker_tax_rebate_eligible",
    "ar_sra_zone",
    "ca_ala_general_assistance_immigration_status_eligible",
    "ca_calworks_child_care_immigration_status_eligible_person",
    "ca_cc_general_assistance_immigration_status_eligible",
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
    "il_aabd_area",
    "il_aabd_immigration_status_eligible_person",
    "il_ccap_county_group",
    "il_dhs_csfp_county_eligible",
    "il_hfs_immigration_status_eligible",
    "il_tanf_county_group",
    "il_tanf_immigration_status_eligible_person",
    "in_ny_mctd_zone_2",
    "in_nyc",
    "is_aca_ptc_immigration_status_eligible",
    "is_basic_health_program_eligible",
    "is_basic_health_program_immigration_status_eligible",
    "is_ca_medicaid_immigration_status_eligible",
    "is_ccdf_immigration_eligible_child",
    "is_chip_fcep_eligible_person",
    "is_citizen_or_legal_immigrant",
    "is_in_snap_abawd_waived_area",
    "is_medicaid_immigration_status_eligible",
    "is_snap_gross_test_full_income_count_alien",
    "is_snap_immigration_status_eligible",
    "is_snap_prorated_income_member",
    "is_snap_state_discretion_ineligible_alien",
    "is_ssi_qualified_noncitizen",
    "ks_ccap_rate_group",
    "ks_dcf_csfp_county_eligible",
    "ks_tanf_county_group",
    "ky_ccap_rate_region",
    "ma_ccfa_immigration_status_eligible",
    "ma_ccfa_region",
    "ma_dese_csfp_county_eligible",
    "md_ccs_region",
    "me_ccap_region",
    "meets_ctc_child_identification_requirements",
    "meets_ctc_identification_requirements",
    "mo_ccs_region",
    "ms_ccpp_facility_location",
    "mt_tanf_immigration_status_eligible_person",
    "ne_child_care_subsidy_location",
    "nh_ccap_immigration_status_eligible_person",
    "ny_ccap_county_group",
    "oh_ccap_county_rate_category",
    "or_healthier_oregon_immigration_status_eligible",
    "overtime_income_deduction_ssn_requirement_met",
    "pa_ccw_region",
    "pa_ccw_stepparent_county_group",
    "pa_tanf_county_group",
    "sc_ccap_geography",
    "sd_cca_region",
    "state_group",
    "state_itemized_deductions",
    "state_standard_deduction",
    "taxsim_state_agi",
    "tip_income_deduction_ssn_requirement_met",
    "tn_ccap_county_tier",
    "trump_dividend_eligible",
    "tx_ccs_workforce_board_region",
    "va_ccsp_income_eligible",
    "va_ccsp_locality_group",
    "va_ccsp_ready_region",
    "va_medicaid_lifc_locality_group",
    "va_tanf",
    "va_tanf_grant_standard",
    "va_tanf_need_standard",
    "va_tanf_up_grant_standard",
    "wa_rca_immigration_status_eligible",
    "wa_tanf_immigration_status_eligible",
    "wa_wccc_center_region",
    "wa_wccc_region",
    "wic_income_limit",
}

__all__ = [name for name in globals() if not name.startswith("__")]
