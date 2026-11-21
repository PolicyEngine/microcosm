"""d2283 PR 2: the one-population contract library.

Phase 0 (the foundation) pins the module's surface and engine-free import,
the injected engine variable index (the default adapter's rules over a fake
inner index, and the stand-in built from the engine ABI lock), PR 1's public
key digest, and the generated population every later clause test corrupts:
uncorrupted, it passes the SPM composition check, the SPM role signal gate
(bands included) with a record computed by the gate's rule, and the assembly
validator with a clone-0, three-channel record.

Phase 1 covers the provenance rewrite (``to_assembled_provenance``: X3, X4,
X6 and one refusal per P5, P6, P6b and P7 rule) and ``cps_unit_ids``.

Phase 2 covers ``require_one_population``: the valid generated population
passes in both forms; X1 as a table of single-cell corruptions (a few add or
drop a column, where no single cell can break the clause), one or more per
clause, each refusing with its own clause's message; a Hypothesis property
over null cells of the checked columns; the clause order; the exempt list
pinned to its stages' declared outputs; and a ``weeks_worked``-like column
(no formula at 2024) refused under R6.

Phase 3 covers ``combined_spm_role_record`` (the record passes the real SPM
role gate on the generated population in both forms; one refusal per check,
a digest-only mismatch included), ``absences`` and ``write_absences`` (both
directions, rows and shares by channel, the six kinds, the raw columns'
readers pinned to the predictor join, the fill manifest checked against the
frame, deterministic JSON), X2 (N-2's kinds on a generated population with
an injected register) and X7 (group quarters add nothing to household-type
targets, skipped until PolicyEngine/microcosm#1610's materializer imports).

The lead's 11-18 calls: (a) the six ACS-native ``acs_*`` amounts CPS rows
lack are "origin-native", with the names pinned to the operator boundary's
contract by reading its source; (b) one declarations mapping passes
``require_one_population`` and ``absences``, which records stale and
needless declarations instead of refusing them; (c) the refusals phase 3
left untested each have a case.

Phase 4 covers X5, the whole-frame differential: on an ASEC-only population
(Route A's labels: ``asec`` clone 0, ``puf_tax_detail`` clones), each
provenance-reading gate returns the same verdict and statistics before and
after ``to_assembled_provenance`` with no ACS row, one parametrized case per
reader. The four readers Route A's clone label moves (the weeks UC-rule
scope, Medicare take-up and both retirement families) are strict expected
failures naming PR 3, with a companion check that each moves only its named
fields; ``acs_local_hours`` is left out, its donor prep refusing the
assembled population. Lead call L7 (11-18) adds two more strict expected
failures on verdicts: one clone-only defect each (a clone's ``MCARE``
mismatch, a clone's weeks with no unemployment compensation) fails Route
A's gate and passes the assembled one, so PR 3 owns six.

The PR #1685 review (round 1) adds: R6's refusal of ``pd.NA`` in an exempt
column, with None and NaN still passing (blocking 1); staging's lineage
keyed by native ids, so a ``SERIALNO`` or ``SPORDER`` swap that keeps the
set of pairs refuses (blocking 2); P6c, every group unit inside one
household (each of the four group entities on CPS rows) and each ACS
household one SPM unit (blocking 3); each ACS role
re-derived from ``RELSHIPP`` (should-fix 1); nullable lineage values
refused before they are compared (should-fix 2); R2's ids and clone indices
parsed as R1 parses them (should-fix 3); an index whose inputs omit a
known column refused (should-fix 4); a negative per-spine fill count
refused (should-fix 5); written zeros counted on a nullable numeric column
(should-fix 6); a spine assembly record count the validator would coerce
refused at R1 (nit 1); and tests that pin R11 between R1 and R2 in the
clause order, with a real location clone and a repeated ``SERIALNO``
reporting their earlier clause (nit 2).

Round 2 adds: complex and temporal ids, lineage values and ``RELSHIPP``
codes refused before any conversion to real numbers, and a timedelta key
in staging's lineage refused (should-fix 1); R6's release-written nulls
limited to None and a float NaN, a float32 column and a numpy float32 NaN
included, with ``pd.NaT`` and numpy's ``NaT`` refused (nit 1); the lineage
diagnostic ahead of a missing weight (nit 2); an ACS SPM unit merger on
both paths and a CPS crossing at R1 (nit 3); and a lone CPS ``pd.NA`` at
R6 and an ACS ``pd.NA`` beside written zeros (nit 4).

Round 3 adds: the shared guard as an allowlist, which reads the value type
behind a categorical or an Arrow dictionary and refuses an Arrow dictionary
of times in lineage values and ``RELSHIPP``, with 27 cases that pin dtype
forms it must keep accepting (should-fix 1); numpy timedelta64 refused
wherever the module tests for a number (donor ids, counts, ``TYPEHUGQ``,
the tail flag) and complex or temporal household weights refused
(should-fix 2); membership links of ACS and CPS persons through the same
guard (nit 2); and ``pd.NA`` reported ahead of another null at R6 (nit 3).
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import json
import math
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.gates import GateResult
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime import one_population_contract as contract
from microcosm.build.us_runtime import one_population_selection as selection
from microcosm.build.us_runtime.acs_local_hours import prepare_acs_local_hours_donor
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.base_pool import ASEC_PUF_SPINE
from microcosm.build.us_runtime.block_location import (
    LOCATION_CLONE_INDEX_COLUMN,
    clone_frame_for_location,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
)
from microcosm.build.us_runtime.one_population_contract import (
    ABSENCE_KINDS,
    ABSENCES_FILENAME,
    ACS_SUPPORT_CHANNEL,
    ASSEMBLED_CHANNELS,
    CPS_SPINE_TAG,
    ENGINE_PERIOD,
    PUF_TAIL_TAX_UNIT_COLUMNS,
    RAW_COLUMN_READERS,
    RELEASE_WRITTEN_INPUTS,
    STRUCTURAL_COLUMNS,
    EngineVariableIndex,
    OnePopulationContractError,
    PolicyEngineUSVariableIndex,
    _engine_columns,
    absences,
    combined_spm_role_record,
    cps_unit_ids,
    require_one_population,
    to_assembled_provenance,
    write_absences,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
)
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.retirement_distributions import (
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.spine_agreement import spine_agreement_gate
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_independence_role import (
    _role_binding_sha256,
    us_spm_independence_role_signal_gate,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    require_assembled_support_provenance,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_copy_rank_series,
    support_gate_source_channel_series,
    support_role_series,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    undefined_stored_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind
from test_support.microcosm_build.us_one_population_contract import (
    ENGINE_ABI_LOCK,
    READER_CLONE_DEFECTS,
    READER_GATES,
    READER_SOURCE_COLUMNS,
    RELEASE_WRITTEN_ON_ACS,
    SPM_ROLE,
    STAND_IN_COMPAT_AGGREGATES,
    STAND_IN_COMPUTED_AT_2024,
    STAND_IN_FORMULA_AFTER_2024,
    STAND_IN_STRUCTURAL_FORMULAS,
    TAIL_PROVENANCE_DEFAULTS,
    WIC_SOURCE_TRIPLE,
    StandInVariableIndex,
    acs_household_weight_series,
    assembled_frame,
    clone_zero_assembly_record,
    cps_household_weight_series,
    cps_role_record,
    fill_manifest,
    generate_population,
    joined_frame,
    lock_engine_inputs,
    mixed_origin_reader_population,
    reader_population,
    role_shares,
    selection_manifest,
    spm_role_gate_record,
    staging_lineage,
    staging_role_receipt,
    with_spm_role_record,
    with_staging_union_columns,
)
from test_support.paths import REPOSITORY_ROOT

_SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def population():
    return generate_population(0)


@pytest.fixture(scope="module")
def frame(population):
    return joined_frame(population)


@pytest.fixture(scope="module")
def stand_in():
    return StandInVariableIndex.from_lock()


# ----------------------------------------------------------- the surface


class TestModuleSurface:
    def test_public_names_resolve_and_are_sorted(self) -> None:
        assert contract.__all__ == sorted(contract.__all__, key=_all_sort_key)
        for name in contract.__all__:
            assert hasattr(contract, name), name

    def test_signatures_take_the_injected_inputs_by_keyword(self) -> None:
        def keyword_only(function) -> set[str]:
            return {
                name
                for name, parameter in inspect.signature(function).parameters.items()
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            }

        parameters = inspect.signature(contract.to_assembled_provenance).parameters
        assert list(parameters)[:2] == ["cps_tables", "acs_tables"]
        assert keyword_only(contract.to_assembled_provenance) == {
            "cps_household_weights",
            "acs_household_weights",
            "staging_lineage",
        }
        assert keyword_only(contract.require_one_population) == {
            "parent_tail_donor_ids",
            "declared_absences",
            "staging_lineage",
            "variable_index",
            "register",
        }
        assert list(
            inspect.signature(contract.combined_spm_role_record).parameters
        ) == ["cps_record", "staging_receipt", "frame"]
        assert keyword_only(contract.absences) == {
            "fill_manifest",
            "zero_written",
            "declared",
        }
        assert list(inspect.signature(contract.cps_unit_ids).parameters) == ["person"]

    def test_constants_match_staging_and_provenance(self) -> None:
        # Staging's ACS channel and spine tag, and its donor-row spine tag.
        assert ACS_SUPPORT_CHANNEL == ACS_2024_1YR_SPINE
        assert CPS_SPINE_TAG == ASEC_PUF_SPINE
        assert ASSEMBLED_CHANNELS == (
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ACS_2024_1YR_SPINE,
        )
        assert ENGINE_PERIOD == 2024
        assert STRUCTURAL_COLUMNS == {
            "person_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        }

    def test_module_imports_no_engine_at_import_time(self) -> None:
        tree = ast.parse(inspect.getsource(contract))
        top_level = [
            node.module if isinstance(node, ast.ImportFrom) else alias.name
            for node in tree.body
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in (node.names if isinstance(node, ast.Import) else [None])
        ]
        assert not [name for name in top_level if "policyengine" in str(name)]

    def test_error_is_a_value_error(self) -> None:
        assert issubclass(OnePopulationContractError, ValueError)


def _all_sort_key(name: str) -> tuple[int, str]:
    # The repo's __all__ order: constants, then classes, then functions.
    if name.isupper():
        return (0, name)
    return (1, name) if name[0].isupper() else (2, name)


# ------------------------------------------- the default adapter (M2)


class _FakeMetadataIndex:
    """The three methods the adapter calls, over fixed name sets."""

    def __init__(self) -> None:
        self.inputs = {"age", "household_id", "person_household_id", "tip_income"}
        self.formula_owned = {"person_id", "adjusted_gross_income", "weeks_worked"}
        self.computed_2024 = {"adjusted_gross_income"}
        self.compat = {"social_security"}
        self.periods: list[object] = []

    def variables(self) -> list[str]:
        return sorted(self.inputs)

    def formula_owned_outputs(self, names) -> set[str]:
        names = set(names)
        return (names & self.compat) | (names & self.formula_owned)

    def _engine_computed_columns(self, tables, *, period) -> set[str]:
        self.periods.append(period)
        present = {column for table in tables.values() for column in table.columns}
        return (present & self.compat) | (
            (present & self.computed_2024) - STRUCTURAL_COLUMNS
        )


class TestDefaultAdapter:
    def test_builds_no_inner_index_until_asked(self, monkeypatch) -> None:
        import microcosm.frame.adapters.policyengine_us as adapter_module

        built: list[str] = []

        class Sentinel:
            def __init__(self) -> None:
                built.append("built")
                raise RuntimeError("inner index built")

        monkeypatch.setattr(
            adapter_module, "PolicyEngineUSVariableMetadataIndex", Sentinel
        )
        index = PolicyEngineUSVariableIndex()
        assert built == []
        assert isinstance(index, EngineVariableIndex)
        with pytest.raises(RuntimeError, match="inner index built"):
            index.known_variables(["age"])
        assert built == ["built"]

    def test_classes_follow_the_release_guard(self) -> None:
        fake = _FakeMetadataIndex()
        index = PolicyEngineUSVariableIndex(fake)
        tables = {
            "person": pd.DataFrame(
                columns=[
                    "person_id",
                    "person_household_id",
                    "age",
                    "tip_income",
                    "weeks_worked",
                    "adjusted_gross_income",
                    "social_security",
                    "not_a_variable",
                ]
            ),
            "household": pd.DataFrame(columns=["household_id"]),
        }
        present = {column for table in tables.values() for column in table}

        assert index.known_variables(present) == present - {"not_a_variable"}
        assert index.computed_at_2024(tables) == {
            "adjusted_gross_income",
            "social_security",
        }
        # weeks_worked: known, no formula at 2024, so an input at 2024 although
        # the year-blind variables() leaves it out. Structural ids never count.
        assert index.inputs_at_2024(tables) == {"age", "tip_income", "weeks_worked"}
        assert set(fake.periods) == {ENGINE_PERIOD}

    def test_coherent_answers_pass_the_contract_check(self) -> None:
        tables = {"person": pd.DataFrame(columns=["person_id", "age", "ssi"])}
        classes = _engine_columns(tables, StandInVariableIndex.from_lock())
        assert classes.inputs == {"age"}
        assert classes.computed == {"ssi"}
        assert classes.known == {"person_id", "age", "ssi"}

    @pytest.mark.parametrize(
        ("broken", "message"),
        [
            ("computed", "inputs also computed"),
            ("structural", "inputs include structural columns"),
            ("unknown", "inputs not known"),
            ("absent", "names columns no table holds"),
            # PR #1685 review, should-fix 4: the other direction.
            ("omitted", "inputs omit known columns that are neither computed"),
        ],
    )
    def test_incoherent_answers_are_refused(self, broken, message) -> None:
        class Broken(StandInVariableIndex):
            def inputs_at_2024(self, tables):
                inputs = set(super().inputs_at_2024(tables))
                if broken == "omitted":
                    return frozenset(inputs - {"age"})
                extra = {
                    "computed": "ssi",
                    "structural": "person_household_id",
                    "unknown": "not_a_variable",
                    "absent": "tip_income",
                }[broken]
                return frozenset(inputs | {extra})

        tables = {
            "person": pd.DataFrame(
                columns=["person_household_id", "age", "ssi", "not_a_variable"]
            )
        }
        index = Broken(inputs=frozenset(lock_engine_inputs()))
        with pytest.raises(OnePopulationContractError, match=message):
            _engine_columns(tables, index)


# --------------------------------------------------- the stand-in (M2)


class TestStandInIndex:
    def test_lock_projection_is_the_pool_contract(self) -> None:
        # lock_engine_inputs checks the lock's own receipt; this pins the
        # same names to the pool's installed-engine contract.
        inputs = lock_engine_inputs()
        payload = [
            {"entity": entity, "variable": variable}
            for variable, entity in sorted(inputs.items())
        ]
        digest = hashlib.sha256(
            json.dumps(
                payload, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        assert len(inputs) == POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count == 926
        assert digest == POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256

    def test_explicit_sets_are_disjoint_from_the_lock(self, stand_in) -> None:
        for names in (
            STAND_IN_COMPUTED_AT_2024,
            STAND_IN_COMPAT_AGGREGATES,
            STAND_IN_FORMULA_AFTER_2024,
            STAND_IN_STRUCTURAL_FORMULAS,
        ):
            assert not names & stand_in.inputs

    def test_a_post_2024_formula_is_an_input_at_2024(self, stand_in) -> None:
        # weeks_worked has only formula_2025 at policyengine-us 2.2.1: absent
        # from the year-blind 926, not computed at 2024, so an input at 2024.
        tables = {"person": pd.DataFrame(columns=["weeks_worked"])}
        assert "weeks_worked" not in lock_engine_inputs()
        assert stand_in.computed_at_2024(tables) == frozenset()
        assert stand_in.inputs_at_2024(tables) == {"weeks_worked"}

    def test_compat_aggregates_are_computed_and_never_inputs(self, stand_in) -> None:
        tables = {"person": pd.DataFrame(columns=sorted(STAND_IN_COMPAT_AGGREGATES))}
        assert stand_in.computed_at_2024(tables) == STAND_IN_COMPAT_AGGREGATES
        assert stand_in.inputs_at_2024(tables) == frozenset()

    def test_structural_columns_are_known_but_neither_class(self, stand_in) -> None:
        tables = {"person": pd.DataFrame(columns=sorted(STRUCTURAL_COLUMNS))}
        assert stand_in.known_variables(STRUCTURAL_COLUMNS) == STRUCTURAL_COLUMNS
        assert stand_in.computed_at_2024(tables) == frozenset()
        assert stand_in.inputs_at_2024(tables) == frozenset()

    def test_lock_inputs_the_clauses_read_are_inputs(self, stand_in) -> None:
        names = [
            "age",
            "employer_sponsored_insurance_premiums",
            SPM_ROLE,
            "takes_up_ssi_if_eligible",
            "congressional_district_geoid",
            "tip_income",
        ]
        tables = {"person": pd.DataFrame(columns=names)}
        assert stand_in.inputs_at_2024(tables) == set(names)


# ------------------------------------------------ PR 1's key digest (Q7)


class TestKeysDigest:
    def test_keys_sha256_is_public_and_is_the_manifest_rule(self) -> None:
        assert "keys_sha256" in selection.__all__
        assert not hasattr(selection, "_keys_sha256")
        keys = ["2024|cps:2", "2023|cps:1", "acs:9"]
        assert (
            selection.keys_sha256(keys)
            == hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()
        )


# ---------------------------------------------- the generated population


class TestGeneratedPopulation:
    def test_is_deterministic_in_its_seed(self) -> None:
        first, again, other = (generate_population(seed) for seed in (3, 3, 4))
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(
                first.cps_tables[entity], again.cps_tables[entity]
            )
            pd.testing.assert_frame_equal(
                first.acs_tables[entity], again.acs_tables[entity]
            )
        assert first.cps_keys == again.cps_keys
        assert first.cps_keys != other.cps_keys

    def test_cps_rows_carry_route_a_labels(self, population) -> None:
        multiplier = population.clone_multiplier
        for entity in US_SCHEMA.entities:
            table = population.cps_tables[entity]
            ids = table[US_SCHEMA.entity_id_column(entity)].to_numpy()
            channel = table[support_channel_column(entity)].to_numpy()
            clone = table[support_clone_index_column(entity)].to_numpy()
            source = table[support_source_id_column(entity)].to_numpy()
            assert set(clone) == {0, 1, 2}
            assert ((channel == BASE_ASEC_SUPPORT_CHANNEL) == (clone == 0)).all()
            assert set(channel[clone > 0]) == {PUF_TAX_DETAIL_SUPPORT_CHANNEL}
            # puf_support._remap_ids, then the tail's one more multiplier.
            assert (ids == source + clone * multiplier).all()
            assert set(source[clone > 0]) <= set(ids[clone == 0])
            assert spine_source_id_column(entity) not in table
            assert f"{entity}_spine" not in table

    def test_acs_rows_carry_staging_labels(self, population) -> None:
        for entity in US_SCHEMA.entities:
            table = population.acs_tables[entity]
            id_column = US_SCHEMA.entity_id_column(entity)
            cps_ids = population.cps_tables[entity][id_column]
            assert (table[support_channel_column(entity)] == ACS_2024_1YR_SPINE).all()
            assert (table[f"{entity}_spine"] == ACS_2024_1YR_SPINE).all()
            assert (table[support_clone_index_column(entity)] == 0).all()
            shift = table[id_column] - table[support_source_id_column(entity)]
            assert set(shift) == {population.acs_id_offsets[entity]}
            assert table[id_column].min() == cps_ids.max() + 1
            assert spine_source_id_column(entity) not in table
        person = population.acs_tables["person"]
        for group in US_SCHEMA.group_entities:
            membership = person[US_SCHEMA.membership_column(group)]
            assert set(membership) == set(
                population.acs_tables[group][US_SCHEMA.id_column(group)]
            )

    def test_raw_lineage_columns(self, population) -> None:
        cps = population.cps_tables["person"]
        native = cps[cps[support_clone_index_column("person")] == 0]
        peridnum = native["PERIDNUM"]
        assert peridnum.str.fullmatch(r"[1-9][0-9]{21}").all()
        assert peridnum.is_unique
        assert (native["source_person_id"] == peridnum).all()
        assert (
            peridnum.str[:20].groupby(native["person_household_id"]).nunique() == 1
        ).all()
        assert set(native["source_year"]) == {2023, 2024}
        assert (
            len(population.cps_keys)
            == len(set(population.cps_keys))
            == len(
                population.cps_tables["household"].query(
                    "household_support_clone_index == 0"
                )
            )
        )

        household = population.acs_tables["household"]
        person = population.acs_tables["person"]
        assert household["SERIALNO"].is_unique
        assert set(household["TYPEHUGQ"]) == {1, 2, 3}
        serial = person["person_household_id"].map(
            household.set_index("household_id")["SERIALNO"]
        )
        assert (
            not pd.DataFrame({"s": serial, "o": person["SPORDER"]}).duplicated().any()
        )
        assert (person["source_person_id"] == person["SPORDER"].astype(str)).all()
        assert (person["source_year"] == 2024).all()
        native_household = person["person_household_id"].map(
            household.set_index("household_id")["household_source_id"]
        )
        assert (person["source_household_id"] == native_household).all()
        assert population.acs_keys == tuple(
            f"acs:{value}" for value in household["SERIALNO"]
        )

    def test_districts_carry_their_state_prefix(self, frame) -> None:
        household = frame.table("household")
        cd = household["congressional_district_geoid"]
        assert (cd > 0).all()
        assert (cd // 100 == household["state_fips"]).all()

    def test_tail_columns_on_every_tax_unit(self, population, frame) -> None:
        tax_unit = frame.table("tax_unit")
        for column in TAIL_PROVENANCE_DEFAULTS:
            assert tax_unit[column].notna().all(), column
        applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy()
        clone = tax_unit[support_clone_index_column("tax_unit")].to_numpy()
        assert applied.any()
        assert (clone[applied] == 2).all()
        donors = tax_unit.loc[applied, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN]
        assert set(donors) == population.tail_donor_ids
        assert (
            tax_unit.loc[~applied, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN] == -1
        ).all()

    def test_every_person_resolves_to_one_row_per_group(self, frame) -> None:
        person = frame.table("person")
        assert person["person_id"].is_unique
        for group in US_SCHEMA.group_entities:
            table = frame.table(group)
            assert table[US_SCHEMA.id_column(group)].is_unique
            joined = person[US_SCHEMA.membership_column(group)].map(
                table.set_index(US_SCHEMA.id_column(group))[
                    support_channel_column(group)
                ]
            )
            assert (joined == person[support_channel_column("person")]).all()

    def test_shared_columns_keep_one_dtype(self, population, frame) -> None:
        for entity in US_SCHEMA.entities:
            cps = population.cps_tables[entity]
            acs = population.acs_tables[entity]
            for column in set(cps.columns) & set(acs.columns):
                assert cps[column].dtype == acs[column].dtype, (entity, column)
        assert frame.table("person")[SPM_ROLE].dtype == np.dtype(bool)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_spm_composition_passes(self, seed) -> None:
        result = check_spm_composition(joined_frame(generate_population(seed)))
        assert result.status == "PASS", result.failures
        assert result.details["role_source"] == "source_column"

    def test_a_role_classifies_an_adult_on_its_own(self, frame) -> None:
        # The ACS household headed by a 17-year-old passes only through the role.
        person = frame.table("person")
        minor_heads = person[(person["age"] == 17) & person[SPM_ROLE]]
        assert len(minor_heads) == 1
        unit = minor_heads["person_spm_unit_id"].iloc[0]
        assert (person.loc[person["person_spm_unit_id"] == unit, "age"] < 18).all()

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_role_signal_gate_passes_with_a_record_by_its_rule(self, seed) -> None:
        frame = joined_frame(generate_population(seed))
        gated = with_spm_role_record(frame, spm_role_gate_record(frame))
        result = us_spm_independence_role_signal_gate(gated)
        assert result.passed, result.failures
        share, minor_share = role_shares(frame)
        assert result.details["role_share"] == pytest.approx(share)
        assert result.details["minor_role_share"] == pytest.approx(minor_share)
        lower, upper = result.details["minor_role_share_band"]
        assert lower < minor_share < upper

    def test_role_signal_gate_fails_without_the_record(self, frame) -> None:
        result = us_spm_independence_role_signal_gate(frame)
        assert not result.passed
        assert "no source derivation provenance" in " ".join(result.failures)

    def test_assembly_validator_passes_on_a_clone_zero_record(self, frame) -> None:
        record = clone_zero_assembly_record(frame)
        assembled = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata={**frame.metadata, **record},
        )
        manifest = validate_assembly_provenance(assembled, boundary="phase 0 test")
        assert tuple(manifest["channels"]) == ASSEMBLED_CHANNELS
        counts = manifest["native_row_counts"]
        assert all(
            counts[entity][PUF_TAX_DETAIL_SUPPORT_CHANNEL] == 0
            for entity in US_SCHEMA.entities
        )

    def test_stored_columns_are_engine_variables_or_registered(
        self, frame, stand_in
    ) -> None:
        columns = {
            column for entity in frame.entities for column in frame.table(entity)
        }
        assert (
            undefined_stored_inputs(
                columns,
                engine_variables=stand_in.known_variables(columns),
                register=US_STORED_NON_VARIABLE_COLUMNS,
            )
            == ()
        )

    def test_no_column_is_computed_at_2024(self, frame, stand_in) -> None:
        tables = {entity: frame.table(entity) for entity in frame.entities}
        assert stand_in.computed_at_2024(tables) == frozenset()

    def test_inputs_are_filled_except_the_release_written(
        self, frame, stand_in
    ) -> None:
        tables = {entity: frame.table(entity) for entity in frame.entities}
        inputs = stand_in.inputs_at_2024(tables)
        assert {"age", SPM_ROLE, "employer_sponsored_insurance_premiums"} <= inputs
        nulls = {
            column
            for table in tables.values()
            for column in inputs & set(table.columns)
            if table[column].isna().any()
        }
        assert nulls == RELEASE_WRITTEN_ON_ACS
        person = frame.table("person")
        acs = person[support_channel_column("person")] == ACS_2024_1YR_SPINE
        for column in RELEASE_WRITTEN_ON_ACS:
            assert person.loc[acs, column].isna().all()
            assert person.loc[~acs, column].notna().all()

    def test_raw_upper_case_columns_sit_on_one_origin(self, frame) -> None:
        person = frame.table("person")
        household = frame.table("household")
        acs = person[support_channel_column("person")] == ACS_2024_1YR_SPINE
        acs_household = (
            household[support_channel_column("household")] == ACS_2024_1YR_SPINE
        )
        for column in ("SSI_VAL", "PERIDNUM"):
            assert person.loc[acs, column].isna().all()
            assert person.loc[~acs, column].notna().all()
        assert person.loc[acs, "SPORDER"].notna().all()
        assert person.loc[~acs, "SPORDER"].isna().all()
        for column in ("SERIALNO", "TYPEHUGQ"):
            assert household.loc[acs_household, column].notna().all()
            assert household.loc[~acs_household, column].isna().all()


class TestStagingUnionForm:
    """The ACS tables in staging's pooled form: every CPS column, all null."""

    def test_acs_rows_gain_every_cps_column_all_null(self, population) -> None:
        union = with_staging_union_columns(population)
        for entity in US_SCHEMA.entities:
            trimmed = population.acs_tables[entity]
            pooled = union.acs_tables[entity]
            cps_only = set(population.cps_tables[entity]) - set(trimmed)
            assert set(pooled) == set(trimmed) | set(population.cps_tables[entity])
            for column in cps_only:
                assert pooled[column].isna().all(), (entity, column)
            for column in trimmed:
                assert (
                    pooled[column].astype(object).tolist()
                    == trimmed[column].astype(object).tolist()
                ), (entity, column)
        person = union.acs_tables["person"]
        assert person["takes_up_ssi_if_eligible"].isna().all()
        # An ACS-only integer column comes back as float64 from staging.
        assert person["SPORDER"].dtype == np.dtype("float64")
        assert union.acs_tables["household"]["TYPEHUGQ"].dtype == np.dtype("float64")

    def test_union_form_passes_the_same_checks(self, population) -> None:
        frame = joined_frame(with_staging_union_columns(population))
        assert check_spm_composition(frame).status == "PASS"
        gated = with_spm_role_record(frame, spm_role_gate_record(frame))
        assert us_spm_independence_role_signal_gate(gated).passed
        assembled = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata={**frame.metadata, **clone_zero_assembly_record(frame)},
        )
        validate_assembly_provenance(assembled, boundary="phase 0 union test")


# --------------------------------- phase 1: the provenance rewrite (P1-P8)

_FORMS = ("trimmed", "union")


def _form(population, form: str):
    return with_staging_union_columns(population) if form == "union" else population


@pytest.fixture(scope="module", params=_FORMS)
def rewritten(request, population):
    """(input population, rewritten frame) in the trimmed and union forms."""

    source = _form(population, request.param)
    return source, assembled_frame(source)


def _by_id(table: pd.DataFrame, entity: str) -> pd.DataFrame:
    return table.set_index(US_SCHEMA.entity_id_column(entity), drop=False)


def _acs_rows(table: pd.DataFrame, entity: str) -> pd.DataFrame:
    return table[table[support_channel_column(entity)] == ACS_SUPPORT_CHANNEL]


def _person_lineage(frame: Frame) -> dict[tuple[int, int], tuple[str, int]]:
    """The frame's ACS lineage in staging's keyed form, read off the frame.

    Each ACS person's native (household id, person id), its household's and
    its own ``<entity>_spine_source_id`` after the rewrite, to its
    (``SERIALNO``, ``SPORDER``).
    """

    household = frame.table("household").set_index("household_id")
    person = _acs_rows(frame.table("person"), "person")
    link = person["person_household_id"]
    return {
        (int(native_household), int(native_person)): (serial, int(sporder))
        for native_household, native_person, serial, sporder in zip(
            link.map(household[spine_source_id_column("household")]).tolist(),
            person[spine_source_id_column("person")].tolist(),
            link.map(household["SERIALNO"]).tolist(),
            person["SPORDER"].tolist(),
            strict=True,
        )
    }


class TestProvenanceRewrite:
    def test_returns_a_frame_with_only_the_clone_zero_record(self, rewritten) -> None:
        _, frame = rewritten
        assert isinstance(frame, Frame)
        assert set(frame.metadata) == {SPINE_ASSEMBLY_MANIFEST_KEY}
        assert frame.mass_log == ()
        record = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
        expected = clone_zero_assembly_record(frame)[SPINE_ASSEMBLY_MANIFEST_KEY]
        assert tuple(record["channels"]) == ASSEMBLED_CHANNELS
        assert {
            entity: dict(counts)
            for entity, counts in record["native_row_counts"].items()
        } == expected["native_row_counts"]

    def test_cps_rows_keep_route_a_labels_and_gain_the_spine_id(
        self, rewritten
    ) -> None:
        population, frame = rewritten
        for entity in US_SCHEMA.entities:
            before = _by_id(population.cps_tables[entity], entity)
            after = _by_id(frame.table(entity), entity).loc[before.index]
            for column in (
                support_source_id_column(entity),
                support_channel_column(entity),
                support_clone_index_column(entity),
            ):
                assert after[column].tolist() == before[column].tolist(), column
            assert (
                after[spine_source_id_column(entity)].tolist()
                == before[support_source_id_column(entity)].tolist()
            )
            assert (after[f"{entity}_spine"] == CPS_SPINE_TAG).all()

    def test_no_row_is_labelled_acs_and_every_row_has_a_spine_tag(
        self, rewritten
    ) -> None:
        # P3: the predictor join's "acs" channel is never written.
        _, frame = rewritten
        for entity in US_SCHEMA.entities:
            table = frame.table(entity)
            assert "acs" not in set(table[support_channel_column(entity)])
            assert table[f"{entity}_spine"].notna().all()
            acs = table[support_channel_column(entity)] == ACS_SUPPORT_CHANNEL
            assert (table.loc[acs, f"{entity}_spine"] == ACS_SUPPORT_CHANNEL).all()
            assert (table.loc[~acs, f"{entity}_spine"] == CPS_SPINE_TAG).all()

    def test_household_weights_follow_their_households(self, rewritten) -> None:
        population, frame = rewritten
        expected = {
            **dict(
                zip(
                    population.cps_tables["household"]["household_id"].tolist(),
                    population.cps_household_weights.tolist(),
                    strict=True,
                )
            ),
            **dict(
                zip(
                    population.acs_tables["household"]["household_id"].tolist(),
                    population.acs_household_weights.tolist(),
                    strict=True,
                )
            ),
        }
        weights = frame.weights_for("household")
        assert weights.kind is WeightKind.IMPORTANCE
        actual = dict(
            zip(
                frame.table("household")["household_id"].tolist(),
                weights.values.tolist(),
                strict=True,
            )
        )
        assert actual == expected
        ids = frame.table("household")["household_id"].to_numpy()
        assert (np.diff(ids) > 0).all()

    def test_weights_follow_households_through_a_real_sort(self, population) -> None:
        # The generated tables arrive sorted, so the sort above is the
        # identity. Here both origins' household rows (with their weights)
        # and one other group table come in shuffled.
        rng = np.random.default_rng(7)
        cps = {entity: table.copy() for entity, table in population.cps_tables.items()}
        acs = {entity: table.copy() for entity, table in population.acs_tables.items()}
        cps_order = rng.permutation(len(cps["household"]))
        acs_order = rng.permutation(len(acs["household"]))
        cps["household"] = cps["household"].iloc[cps_order].reset_index(drop=True)
        acs["household"] = acs["household"].iloc[acs_order].reset_index(drop=True)
        cps["tax_unit"] = cps["tax_unit"].iloc[::-1].reset_index(drop=True)
        cps_weights = population.cps_household_weights[cps_order]
        acs_weights = population.acs_household_weights[acs_order]
        frame = to_assembled_provenance(
            cps,
            acs,
            cps_household_weights=cps_household_weight_series(
                cps["household"], cps_weights
            ),
            acs_household_weights=acs_household_weight_series(
                acs["household"], acs_weights
            ),
            staging_lineage=staging_lineage(population),
        )
        given = np.concatenate(
            [cps["household"]["household_id"], acs["household"]["household_id"]]
        )
        expected = dict(
            zip(
                given.tolist(),
                np.concatenate([cps_weights, acs_weights]).tolist(),
                strict=True,
            )
        )
        household = frame.table("household")["household_id"].to_numpy()
        assert household.tolist() != given.tolist()
        assert (np.diff(household) > 0).all()
        assert (np.diff(frame.table("tax_unit")["tax_unit_id"].to_numpy()) > 0).all()
        actual = dict(
            zip(
                household.tolist(),
                frame.weights_for("household").values.tolist(),
                strict=True,
            )
        )
        assert actual == expected
        # Same frame as the sorted inputs give, row for row.
        reference = assembled_frame(population)
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(
                frame.table(entity), reference.table(entity), check_like=True
            )
        np.testing.assert_array_equal(
            frame.weights_for("household").values,
            reference.weights_for("household").values,
        )

    @pytest.mark.parametrize("form", ("series", "mapping"))
    def test_weights_are_tied_to_households_by_key(self, population, form) -> None:
        # d2283 lead call L3's hazard: PR 1's ACS starting weights come in
        # selection order and staging's households in id order. Keyed
        # weights in another order than the tables' still give each
        # household its own weight; by position they would not.
        rng = np.random.default_rng(11)
        cps = population.keyed_cps_household_weights
        acs = population.keyed_acs_household_weights
        cps = cps.iloc[rng.permutation(len(cps))]
        acs = acs.iloc[::-1]
        assert not np.array_equal(acs.to_numpy(), population.acs_household_weights)
        assert not np.array_equal(cps.to_numpy(), population.cps_household_weights)
        if form == "mapping":
            cps, acs = dict(cps.items()), dict(acs.items())
        frame = to_assembled_provenance(
            population.cps_tables,
            population.acs_tables,
            cps_household_weights=cps,
            acs_household_weights=acs,
            staging_lineage=staging_lineage(population),
        )
        np.testing.assert_array_equal(
            frame.weights_for("household").values,
            assembled_frame(population).weights_for("household").values,
        )

    def test_a_serialno_swap_cannot_trade_weights(self, population) -> None:
        # PR #1685 review, blocking 2: two households of equal size, weighted
        # 10 and 90, trade SERIALNO. The persons' (SERIALNO, SPORDER) set is
        # unchanged, which is all the old set-of-pairs check compared, and
        # the weights stay keyed by SERIALNO, so each household would have
        # taken the other's weight. Staging's lineage, keyed by native ids,
        # refuses the swap before any weight is looked up.
        acs = _serialno_swaps_between_households(population)["acs"]
        before = population.acs_tables["household"]["SERIALNO"]
        swapped = acs["household"]["SERIALNO"].ne(before).to_numpy()
        assert int(swapped.sum()) == 2
        weights = population.keyed_acs_household_weights.copy()
        low, high = (selection.acs_unit_key(serial) for serial in before[swapped])
        weights.loc[low], weights.loc[high] = 10.0, 90.0
        moved = staging_lineage(dataclasses.replace(population, acs_tables=acs))
        assert set(moved.values()) == set(staging_lineage(population).values())
        assert moved != staging_lineage(population)
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("0 extra (such as []), 4 moved (such as ["),
        ):
            to_assembled_provenance(
                population.cps_tables,
                acs,
                cps_household_weights=population.keyed_cps_household_weights,
                acs_household_weights=weights,
                staging_lineage=staging_lineage(population),
            )

    def test_the_lineage_refusal_precedes_a_missing_weight(self, population) -> None:
        # PR #1685 review r2, nit 2: the serial swap plus a keyed weight map
        # missing one household. P7 compares the lineage before any weight is
        # looked up, so its moved-pair diagnostic comes first; a keyed
        # comparison delayed past the weight lookup would report the missing
        # weight instead.
        acs = _serialno_swaps_between_households(population)["acs"]
        before = population.acs_tables["household"]["SERIALNO"]
        swapped = acs["household"]["SERIALNO"].ne(before).to_numpy()
        traded = {selection.acs_unit_key(serial) for serial in before[swapped]}
        weights = population.keyed_acs_household_weights
        dropped = next(key for key in weights.index if key not in traded)
        weights = weights.drop(dropped)
        assert len(weights) == len(acs["household"]) - 1
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("0 extra (such as []), 4 moved (such as ["),
        ):
            _rewrite(population, {"acs": acs, "acs_weights": weights})

    def test_inputs_are_not_mutated(self, population) -> None:
        cps = {entity: table.copy() for entity, table in population.cps_tables.items()}
        acs = {entity: table.copy() for entity, table in population.acs_tables.items()}
        assembled_frame(population)
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(population.cps_tables[entity], cps[entity])
            pd.testing.assert_frame_equal(population.acs_tables[entity], acs[entity])


class TestRewriteInvariants:
    """X3: after the rewrite, across both origins."""

    def test_source_id_and_rank_are_unique_across_origins(self, rewritten) -> None:
        _, frame = rewritten
        for entity in US_SCHEMA.entities:
            table = frame.table(entity)
            pairs = pd.DataFrame(
                {
                    "source": table[support_source_id_column(entity)].to_numpy(),
                    "rank": support_copy_rank_series(table, entity=entity).to_numpy(),
                }
            )
            assert not pairs.duplicated().any(), entity
            # Both origins are present, so uniqueness is across them.
            assert set(table[support_channel_column(entity)]) == set(ASSEMBLED_CHANNELS)

    def test_every_acs_spine_id_is_its_native_id(self, rewritten) -> None:
        population, frame = rewritten
        for entity in US_SCHEMA.entities:
            native = _by_id(population.acs_tables[entity], entity)
            after = _by_id(_acs_rows(frame.table(entity), entity), entity)
            assert sorted(after.index) == sorted(native.index)
            after = after.loc[native.index]
            # Native id: staging's source id before the rewrite.
            assert (
                after[spine_source_id_column(entity)].tolist()
                == native[support_source_id_column(entity)].tolist()
            )
            # The new source id is the pooled id.
            assert (
                after[support_source_id_column(entity)].tolist()
                == after[US_SCHEMA.entity_id_column(entity)].tolist()
            )
            assert (after[support_clone_index_column(entity)] == 0).all()

    def test_every_acs_serialno_and_sporder_survives(self, rewritten) -> None:
        population, frame = rewritten
        assert _person_lineage(frame) == staging_lineage(population)
        native = _by_id(population.acs_tables["household"], "household")
        after = _by_id(frame.table("household"), "household").loc[native.index]
        assert after["SERIALNO"].tolist() == native["SERIALNO"].tolist()
        assert after["TYPEHUGQ"].tolist() == native["TYPEHUGQ"].tolist()


class TestAssembledValidators:
    """X6: the readers that check assembled provenance accept the frame."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("form", _FORMS)
    def test_validator_passes_on_three_channels_with_clones(self, seed, form) -> None:
        population = _form(generate_population(seed), form)
        frame = assembled_frame(population)
        manifest = validate_assembly_provenance(frame, boundary="phase 1 test")
        assert tuple(manifest["channels"]) == ASSEMBLED_CHANNELS
        counts = manifest["native_row_counts"]
        for entity in US_SCHEMA.entities:
            cps = population.cps_tables[entity]
            clone = cps[support_clone_index_column(entity)]
            assert dict(counts[entity]) == {
                BASE_ASEC_SUPPORT_CHANNEL: int((clone == 0).sum()),
                PUF_TAX_DETAIL_SUPPORT_CHANNEL: 0,
                ACS_SUPPORT_CHANNEL: len(population.acs_tables[entity]),
            }
            assert set(clone) == {0, 1, 2}
            require_assembled_support_provenance(frame.table(entity), entity=entity)

    def test_spine_agreement_gate_does_not_raise_with_the_record(
        self, rewritten
    ) -> None:
        _, frame = rewritten
        result = spine_agreement_gate(frame)
        assert isinstance(result, GateResult)
        # It fails, but as a result: puf_tax_detail is a clone role to it.
        assert not result.passed
        assert any("clone role" in failure for failure in result.failures)

    def test_spine_agreement_gate_raises_without_the_record(self, rewritten) -> None:
        _, frame = rewritten
        bare = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
        )
        with pytest.raises(ValueError, match="has no assembly manifest"):
            spine_agreement_gate(bare)


# ------------------------------------------------ X4: a Route A fixture


@pytest.fixture(scope="module")
def route_a():
    """A Route A base made by the production operators, not by hand.

    ``clone_us_frame_for_puf_support`` clones the release-base ASEC fixture
    (``us_release_base_input.asec_frame``) into native rows (clone 0,
    ``asec``) and PUF tax-detail copies (clone 1, ``puf_tax_detail``) on all
    six entities, and ``with_capital_gains_tail_copies`` adds tail copies
    (clone 2) through ``puf_capital_gains_tail._clone_and_transfer``. The
    generator's CPS half restates the same rules by hand; this fixture is
    used so the differential does not rest on that restatement.
    """

    from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
    from test_support.microcosm_build.us_release_base_input import asec_frame
    from test_support.microcosm_build.us_tail_clone_fixtures import (
        with_capital_gains_tail_copies,
    )

    cloned = clone_us_frame_for_puf_support(asec_frame(40, seed=5))
    household = cloned.table("household")
    sources = household.loc[
        household["household_support_clone_index"] == 1, "household_source_id"
    ].tolist()[::9]
    return with_capital_gains_tail_copies(cloned, sources)


def test_role_readers_match_route_a_with_no_acs_row(route_a, population) -> None:
    """X4: with no ACS row, the role readers return Route A's series.

    The Route A fixture is built by the production clone and tail operators
    (see ``route_a``), not from the generator's hand-written CPS half, so the
    differential does not rest on the generator restating Route A's labels.
    The ACS input is the generator's ACS tables cut to zero rows.
    """

    tables = {entity: route_a.table(entity) for entity in route_a.entities}
    empty_acs = {
        entity: population.acs_tables[entity].iloc[:0] for entity in US_SCHEMA.entities
    }
    frame = to_assembled_provenance(
        tables,
        empty_acs,
        cps_household_weights=cps_household_weight_series(
            route_a.table("household"), route_a.weights_for("household").values
        ),
        acs_household_weights={},
        staging_lineage={},
    )
    for entity in US_SCHEMA.entities:
        before, after = tables[entity], frame.table(entity)
        id_column = US_SCHEMA.entity_id_column(entity)
        assert spine_source_id_column(entity) not in before
        assert spine_source_id_column(entity) in after
        assert set(before[support_clone_index_column(entity)]) == {0, 1, 2}
        for column in before:
            assert after[column].dtype == before[column].dtype, (entity, column)
        for reader in (
            support_role_series,
            support_gate_source_channel_series,
            support_copy_rank_series,
        ):
            expected = pd.Series(
                reader(before, entity=entity).to_numpy(),
                index=before[id_column].to_numpy(),
            ).sort_index()
            actual = pd.Series(
                reader(after, entity=entity).to_numpy(),
                index=after[id_column].to_numpy(),
            ).sort_index()
            pd.testing.assert_series_equal(actual, expected, check_names=False)
    validate_assembly_provenance(frame, boundary="phase 1 X4")
    assert dict(
        zip(
            frame.table("household")["household_id"].tolist(),
            frame.weights_for("household").values.tolist(),
            strict=True,
        )
    ) == dict(
        zip(
            tables["household"]["household_id"].tolist(),
            route_a.weights_for("household").values.tolist(),
            strict=True,
        )
    )


# ------------------------------------ refusals: one per P5, P6, P6b, P7 rule


def _copies(population, origin: str) -> dict[str, pd.DataFrame]:
    tables = population.cps_tables if origin == "cps" else population.acs_tables
    return {entity: table.copy() for entity, table in tables.items()}


def _set(table: pd.DataFrame, row: int, column: str, value: object) -> None:
    """Set one cell, upcasting the column to object when the dtype refuses."""

    try:
        table.loc[table.index[row], column] = value
    except (TypeError, ValueError):
        table[column] = table[column].astype(object)
        table.loc[table.index[row], column] = value


def _first(table: pd.DataFrame, entity: str, clone: int) -> int:
    rows = np.flatnonzero(table[support_clone_index_column(entity)].to_numpy() == clone)
    return int(rows[0])


def _cps_missing_column(p):
    cps = _copies(p, "cps")
    cps["household"] = cps["household"].drop(columns=["household_support_channel"])
    return {"cps": cps}


def _cps_null_source(p):
    cps = _copies(p, "cps")
    _set(cps["person"], 0, "person_source_id", None)
    return {"cps": cps}


def _cps_non_integer_source(p):
    cps = _copies(p, "cps")
    _set(cps["person"], 0, "person_source_id", 1.5)
    return {"cps": cps}


def _cps_unknown_channel(p):
    cps = _copies(p, "cps")
    table = cps["tax_unit"]
    _set(table, _first(table, "tax_unit", 1), "tax_unit_support_channel", "acs")
    return {"cps": cps}


def _cps_asec_on_a_clone(p):
    cps = _copies(p, "cps")
    table = cps["spm_unit"]
    _set(table, _first(table, "spm_unit", 1), "spm_unit_support_channel", "asec")
    return {"cps": cps}


def _cps_clone_index_three(p):
    cps = _copies(p, "cps")
    table = cps["family"]
    _set(table, _first(table, "family", 2), "family_support_clone_index", 3)
    return {"cps": cps}


def _cps_spine_source_present(p):
    cps = _copies(p, "cps")
    table = cps["marital_unit"]
    table["marital_unit_spine_source_id"] = table["marital_unit_source_id"]
    return {"cps": cps}


def _cps_wrong_spine_tag(p):
    cps = _copies(p, "cps")
    cps["household"]["household_spine"] = ACS_SUPPORT_CHANNEL
    return {"cps": cps}


def _acs_missing_column(p):
    acs = _copies(p, "acs")
    acs["tax_unit"] = acs["tax_unit"].drop(columns=["tax_unit_support_clone_index"])
    return {"acs": acs}


def _acs_null_channel(p):
    acs = _copies(p, "acs")
    _set(acs["spm_unit"], 0, "spm_unit_support_channel", None)
    return {"acs": acs}


def _acs_channel_acs(p):
    acs = _copies(p, "acs")
    acs["person"]["person_support_channel"] = "acs"
    return {"acs": acs}


def _acs_clone_index(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 0, "household_support_clone_index", 1)
    return {"acs": acs}


def _acs_offset_varies(p):
    acs = _copies(p, "acs")
    _set(
        acs["family"],
        0,
        "family_source_id",
        int(acs["family"]["family_source_id"].iloc[0]) + 1,
    )
    return {"acs": acs}


def _acs_spine_source_present(p):
    acs = _copies(p, "acs")
    table = acs["household"]
    table["household_spine_source_id"] = table["household_source_id"]
    return {"acs": acs}


def _acs_wrong_spine_tag(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "person_spine", CPS_SPINE_TAG)
    return {"acs": acs}


def _acs_null_spine_tag(p):
    acs = _copies(p, "acs")
    _set(acs["tax_unit"], 0, "tax_unit_spine", None)
    return {"acs": acs}


def _acs_spine_tag_absent(p):
    acs = _copies(p, "acs")
    acs["household"] = acs["household"].drop(columns=["household_spine"])
    return {"acs": acs}


def _acs_weights_by_position(p):
    # PR 1's StartingWeights.acs as it comes: an array in selection order.
    return {"acs_weights": p.acs_household_weights}


def _cps_weights_by_position(p):
    return {"cps_weights": p.cps_household_weights.tolist()}


def _acs_weight_key_repeats(p):
    weights = p.keyed_acs_household_weights
    return {"acs_weights": pd.concat([weights, weights.iloc[:1]])}


def _acs_household_without_a_weight(p):
    return {"acs_weights": p.keyed_acs_household_weights.iloc[1:]}


def _serialno_not_a_clean_key(p):
    # P7 takes any non-empty, unique SERIALNO; the weight key needs a clean
    # one, so staging's own lineage is made to agree with it.
    acs = _copies(p, "acs")
    serial = acs["household"]["SERIALNO"].iloc[0]
    _set(acs["household"], 0, "SERIALNO", f"{serial}|x")
    lineage = staging_lineage(dataclasses.replace(p, acs_tables=acs))
    return {"acs": acs, "lineage": lineage}


def _cps_weight_names_no_household(p):
    weights = p.keyed_cps_household_weights
    extra = pd.Series([1.0], index=[int(weights.index.max()) + 1])
    return {"cps_weights": pd.concat([weights, extra])}


def _cps_weight_negative(p):
    weights = p.keyed_cps_household_weights.copy()
    weights.iloc[0] = -1.0
    return {"cps_weights": weights}


def _acs_weight_not_finite(p):
    weights = dict(p.keyed_acs_household_weights.items())
    weights[next(iter(weights))] = np.nan
    return {"acs_weights": weights}


def _acs_weight_complex(p):
    # Review r3, should-fix 2: the float64 conversion dropped the imaginary
    # part, with only a ComplexWarning, and kept the real weight.
    weights = p.keyed_acs_household_weights.astype(np.complex128)
    weights.iloc[0] += 1j
    return {"acs_weights": weights}


def _cps_weight_numpy_complex(p):
    weights = dict(p.keyed_cps_household_weights.items())
    first = next(iter(weights))
    weights[first] = np.complex128(weights[first] + 1j)
    return {"cps_weights": weights}


def _cps_weight_timedelta(p):
    # The float64 conversion read each weight as its count of nanoseconds.
    weights = p.keyed_cps_household_weights
    return {
        "cps_weights": pd.Series(
            pd.to_timedelta(weights.to_numpy(), unit="ns"), index=weights.index
        )
    }


def _ids_collide_across_origins(p):
    # Staged on another donor: staging gave the ACS tax units no offset, so
    # their native ids sit on the export's.
    acs = _copies(p, "acs")
    offset = p.acs_id_offsets["tax_unit"]
    acs["tax_unit"]["tax_unit_id"] -= offset
    acs["person"]["person_tax_unit_id"] -= offset
    return {"acs": acs}


def _membership_crosses_origins(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    sizes = person["person_spm_unit_id"].map(
        person["person_spm_unit_id"].value_counts()
    )
    row = int(np.flatnonzero(sizes.to_numpy() > 1)[0])
    cps_unit = int(p.cps_tables["spm_unit"]["spm_unit_id"].iloc[0])
    _set(person, row, "person_spm_unit_id", cps_unit)
    return {"acs": acs}


def _group_without_a_person(p):
    acs = _copies(p, "acs")
    family = acs["family"]
    extra = family.iloc[[-1]].copy()
    extra["family_id"] += 1
    extra["family_source_id"] += 1
    acs["family"] = pd.concat([family, extra], ignore_index=True)
    return {"acs": acs}


def _rank_repeats_within_cps(p):
    cps = _copies(p, "cps")
    table = cps["person"]
    _set(table, _first(table, "person", 2), "person_support_clone_index", 1)
    return {"cps": cps}


def _rank_repeats_across_origins(p):
    cps = _copies(p, "cps")
    acs_id = int(p.acs_tables["household"]["household_id"].iloc[0])
    table = cps["household"]
    _set(table, _first(table, "household", 0), "household_source_id", acs_id)
    return {"cps": cps}


def _serialno_column_missing(p):
    acs = _copies(p, "acs")
    acs["household"] = acs["household"].drop(columns=["SERIALNO"])
    return {"acs": acs}


def _serialno_null(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 0, "SERIALNO", None)
    return {"acs": acs}


def _serialno_repeats(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 1, "SERIALNO", acs["household"]["SERIALNO"].iloc[0])
    return {"acs": acs}


def _sporder_null(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "SPORDER", None)
    return {"acs": acs}


def _sporder_repeats(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    second = int(np.flatnonzero(person["SPORDER"].to_numpy() == 2)[0])
    _set(person, second, "SPORDER", 1)
    return {"acs": acs}


def _source_household_id_moves(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    _set(
        person,
        0,
        "source_household_id",
        int(person["source_household_id"].iloc[0]) + 1000,
    )
    return {"acs": acs}


def _source_person_id_moves(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "source_person_id", "99")
    return {"acs": acs}


def _source_year_moves(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "source_year", 2023)
    return {"acs": acs}


def _with_pd_na(table: pd.DataFrame, column: str, dtype: str) -> None:
    """``column`` in pandas' nullable ``dtype``, with ``pd.NA`` in its first row."""

    values = table[column].astype(dtype)
    values.iloc[0] = pd.NA
    table[column] = values


def _source_year_pd_na(p):
    # PR #1685 review, should-fix 2: an Int64 pd.NA, which the old year
    # check's mismatch sum skipped.
    acs = _copies(p, "acs")
    _with_pd_na(acs["person"], "source_year", "Int64")
    return {"acs": acs}


def _source_person_id_pd_na(p):
    # Should-fix 2: a nullable string's pd.NA, which the old comparison met
    # with a bare TypeError.
    acs = _copies(p, "acs")
    _with_pd_na(acs["person"], "source_person_id", "string")
    return {"acs": acs}


def _source_household_id_pd_na(p):
    acs = _copies(p, "acs")
    _with_pd_na(acs["person"], "source_household_id", "Int64")
    return {"acs": acs}


def _minor_head_household(tables) -> object:
    """The id of the generated ACS household headed by a 17-year-old (two persons)."""

    person = tables["person"]
    heads = person.loc[
        person["person_support_channel"].eq(ACS_SUPPORT_CHANNEL).to_numpy()
        & person["is_household_head"].to_numpy(dtype=bool)
        & person["age"].eq(17).to_numpy(),
        "person_household_id",
    ]
    assert len(heads) == 1
    household = heads.iloc[0]
    assert int(person["person_household_id"].eq(household).sum()) == 2
    return household


def _swap_serialno(household: pd.DataFrame, first: int, second: int) -> None:
    serials = household["SERIALNO"].tolist()
    _set(household, first, "SERIALNO", serials[second])
    _set(household, second, "SERIALNO", serials[first])


def _two_person_acs_households(tables) -> tuple[int, int]:
    """Rows of the first two ACS households with two persons each.

    Their persons hold SPORDER 1 and 2, so a SERIALNO trade between them
    keeps the set of (SERIALNO, SPORDER) pairs.
    """

    person = tables["person"]
    acs = person["person_support_channel"].eq(ACS_SUPPORT_CHANNEL).to_numpy()
    sizes = person.loc[acs, "person_household_id"].value_counts(sort=False)
    pairs = sizes.index[sizes.eq(2).to_numpy()]
    rows = np.flatnonzero(tables["household"]["household_id"].isin(pairs).to_numpy())
    first, second = (int(row) for row in rows[:2])
    return first, second


def _serialno_swaps_between_households(p):
    # PR #1685 review, blocking 2: two households trade SERIALNO. Their
    # persons' (SERIALNO, SPORDER) set holds, so the old set check passed
    # the frame and each household took the other's weight (weights are
    # keyed by SERIALNO). Keyed by native ids, all four persons' pairs
    # moved. The review's households hold one person each; two persons each
    # keep this case's count apart from the SPORDER swap's.
    acs = _copies(p, "acs")
    _swap_serialno(acs["household"], *_two_person_acs_households(acs))
    return {"acs": acs}


def _sporder_swaps_within_a_household(p):
    # Blocking 2's second shape: two persons of one household trade SPORDER
    # and, to keep source_person_id = str(SPORDER), source_person_id.
    acs = _copies(p, "acs")
    person = acs["person"]
    first, second = (
        int(row)
        for row in np.flatnonzero(
            person["person_household_id"].eq(_minor_head_household(acs)).to_numpy()
        )
    )
    for column in ("SPORDER", "source_person_id"):
        values = person[column].tolist()
        _set(person, first, column, values[second])
        _set(person, second, column, values[first])
    return {"acs": acs}


def _staging_has_a_person_the_selection_lost(p):
    return {"lineage": {**staging_lineage(p), (999_999, 1): ("2024HU9999999", 1)}}


def _staging_lacks_a_selected_person(p):
    lineage = staging_lineage(p)
    first = min(lineage)
    return {"lineage": {key: pair for key, pair in lineage.items() if key != first}}


def _staging_pair_malformed(p):
    return {"lineage": {**staging_lineage(p), (999_999, 1): ("2024HU0000001",)}}


def _staging_key_malformed(p):
    # A key that is not native ids (here a SERIALNO) cannot tie a pair to
    # its person.
    return {"lineage": {**staging_lineage(p), "2024HU0000001": ("2024HU0000001", 1)}}


def _staging_key_timedelta(p):
    # PR #1685 review r2, should-fix 1, on staging's lineage: numpy derives
    # timedelta64 from signedinteger, so int() read this key as the native
    # ids it replaces and the lineage matched.
    lineage = staging_lineage(p)
    first = min(lineage)
    pair = lineage.pop(first)
    lineage[tuple(np.timedelta64(part, "ns") for part in first)] = pair
    return {"lineage": lineage}


def _acs_tax_unit_spans_households(p):
    # PR #1685 review, blocking 3: one ACS person joins a tax unit of another
    # household. Its own tax unit keeps a member, so every id resolves (P6b)
    # and every channel matches (the validator).
    acs = _copies(p, "acs")
    person = acs["person"]
    row = int(np.flatnonzero(_household_sizes(person, "person_tax_unit_id") > 1)[0])
    household = person["person_household_id"].iloc[row]
    elsewhere = person.loc[
        person["person_household_id"].ne(household), "person_tax_unit_id"
    ]
    _set(person, row, "person_tax_unit_id", elsewhere.iloc[0])
    return {"acs": acs}


def _cps_unit_spans_households(p, group: str):
    # Blocking 3 on CPS rows: a native person joins another native
    # household's ``group`` unit. Its own unit keeps a member, so every id
    # resolves (P6b) and every channel matches (the validator).
    cps = _copies(p, "cps")
    person = cps["person"]
    column = f"person_{group}_id"
    native = person["person_support_clone_index"].eq(0).to_numpy()
    sizes = _household_sizes(person, column)
    row = int(np.flatnonzero(native & (sizes > 1))[0])
    household = person["person_household_id"].iloc[row]
    elsewhere = native & person["person_household_id"].ne(household).to_numpy()
    _set(person, row, column, person[column].iloc[int(np.flatnonzero(elsewhere)[0])])
    return {"cps": cps}


def _cps_family_spans_households(p):
    return _cps_unit_spans_households(p, "family")


def _acs_household_holds_two_spm_units(p):
    # Blocking 3's partition: one person of a two-or-more-person ACS
    # household moves to an SPM unit of its own, inside the household.
    # Staging's ACS role rule is defined only on one SPM unit per household.
    acs = _copies(p, "acs")
    person, spm_unit = acs["person"], acs["spm_unit"]
    row = int(np.flatnonzero(_household_sizes(person, "person_household_id") > 1)[0])
    extra = spm_unit.iloc[[-1]].copy()
    shift = int(spm_unit["spm_unit_id"].max()) + 1 - int(extra["spm_unit_id"].iloc[0])
    extra["spm_unit_id"] += shift
    extra["spm_unit_source_id"] += shift
    acs["spm_unit"] = pd.concat([spm_unit, extra], ignore_index=True)
    _set(person, row, "person_spm_unit_id", int(extra["spm_unit_id"].iloc[0]))
    return {"acs": acs}


def _merge_acs_spm_units(
    person: pd.DataFrame, spm_unit: pd.DataFrame, acs: np.ndarray
) -> pd.DataFrame:
    """Move every person of the first ACS household into the second's SPM unit.

    Edits ``person`` in place and returns ``spm_unit`` without the emptied
    unit's row, so every membership still resolves and every unit holds a
    person. Each household still holds one SPM unit, so only the forward
    check (no unit holds persons of two households) sees the merger.
    """

    households = person.loc[acs, "person_household_id"].unique()
    moved = acs & person["person_household_id"].eq(households[0]).to_numpy()
    kept = acs & person["person_household_id"].eq(households[1]).to_numpy()
    emptied = person.loc[moved, "person_spm_unit_id"].unique()
    assert len(emptied) == 1
    person.loc[moved, "person_spm_unit_id"] = person.loc[
        kept, "person_spm_unit_id"
    ].iloc[0]
    return spm_unit.loc[spm_unit["spm_unit_id"].ne(emptied[0]).to_numpy()].reset_index(
        drop=True
    )


def _acs_spm_units_merge(p):
    # PR #1685 review r2, nit 3: one ACS household's persons join another
    # household's SPM unit; the reverse check (one SPM unit per household)
    # passes it.
    acs = _copies(p, "acs")
    person = acs["person"]
    acs["spm_unit"] = _merge_acs_spm_units(
        person, acs["spm_unit"], np.ones(len(person), dtype=bool)
    )
    return {"acs": acs}


_REFUSALS = [
    # P5 on CPS rows (Route A's labels, complete).
    (
        _cps_missing_column,
        "CPS rows lack complete support metadata: 'household' is missing",
    ),
    (
        _cps_null_source,
        "CPS rows lack complete support metadata: 'person_source_id' has 1 null",
    ),
    (
        _cps_non_integer_source,
        "CPS 'person_source_id' must hold source ids: 1 value(s) are not integers",
    ),
    (_cps_unknown_channel, "CPS rows must carry channel 'asec' or 'puf_tax_detail'"),
    (
        _cps_asec_on_a_clone,
        "CPS rows must carry channel 'asec' exactly on clone index 0",
    ),
    (_cps_clone_index_three, "CPS clone indices must be 0, 1 or 2"),
    (
        _cps_spine_source_present,
        "CPS rows already carry 'marital_unit_spine_source_id'",
    ),
    (_cps_wrong_spine_tag, "CPS rows carry a spine tag other than 'asec_puf'"),
    # P5 on ACS rows (staging's labels, complete, one offset).
    (
        _acs_missing_column,
        "ACS rows lack complete support metadata: 'tax_unit' is missing",
    ),
    (
        _acs_null_channel,
        "ACS rows lack complete support metadata: 'spm_unit_support_channel' has 1 null",
    ),
    (_acs_channel_acs, "ACS rows must carry channel 'acs_2024_1yr'"),
    (_acs_clone_index, "ACS rows must carry clone index 0"),
    (_acs_offset_varies, "ACS 'family' ids must sit one constant offset"),
    (_acs_spine_source_present, "ACS rows already carry 'household_spine_source_id'"),
    (_acs_wrong_spine_tag, "ACS rows carry a spine tag other than 'acs_2024_1yr'"),
    # Q3: a null spine tag after the rewrite.
    (_acs_null_spine_tag, "A null spine tag: 1 ACS 'tax_unit' row(s)"),
    (_acs_spine_tag_absent, "A null spine tag: 24 ACS 'household' row(s)"),
    # Household weights keyed by household (d2283 lead call L3's check).
    (
        _acs_weights_by_position,
        "ACS household weights must be keyed by acs_unit_key(SERIALNO) (a pandas "
        "Series or a mapping), not a ndarray",
    ),
    (
        _cps_weights_by_position,
        "CPS household weights must be keyed by household_id (a pandas Series or "
        "a mapping), not a list",
    ),
    (_acs_weight_key_repeats, "ACS household weights repeat a key: ['acs:"),
    (_cps_weight_negative, "CPS household weights must be finite and non-negative."),
    (_acs_weight_not_finite, "ACS household weights must be finite and non-negative."),
    (
        _acs_weight_complex,
        "ACS household weights must be real numbers; they hold complex128 values",
    ),
    (
        _cps_weight_numpy_complex,
        "CPS household weights must be real numbers; they hold 1 value(s) that are "
        "not real numbers or strings, such as [np.complex128(",
    ),
    (
        _cps_weight_timedelta,
        "CPS household weights must be real numbers; they hold timedelta64[ns] values",
    ),
    # P6b: frame ids and memberships across origins.
    (
        _ids_collide_across_origins,
        "Frame ids collide: 'tax_unit_id' repeats across origins",
    ),
    (
        _membership_crosses_origins,
        "Memberships do not resolve: 1 ACS person(s)' 'person_spm_unit_id' match no ACS spm_unit row",
    ),
    (
        _group_without_a_person,
        "Memberships do not resolve: 1 ACS family row(s) hold no ACS person",
    ),
    # P6: a repeated (source id, rank) pair.
    (_rank_repeats_within_cps, "A (source id, rank) pair repeats in 'person'"),
    (_rank_repeats_across_origins, "A (source id, rank) pair repeats in 'household'"),
    # P7: SERIALNO/SPORDER lineage.
    (
        _serialno_column_missing,
        "ACS lineage does not survive: the ACS household table lacks ['SERIALNO']",
    ),
    (
        _serialno_null,
        "ACS lineage does not survive: every ACS household needs a non-empty string SERIALNO",
    ),
    (
        _serialno_repeats,
        "ACS lineage does not survive: SERIALNO repeats across ACS households",
    ),
    (
        _sporder_null,
        "ACS lineage does not survive: every ACS person needs an integral SPORDER: 1 value(s) are null",
    ),
    (
        _sporder_repeats,
        "ACS lineage does not survive: (SERIALNO, SPORDER) repeats on 2 ACS person(s)",
    ),
    (
        _source_household_id_moves,
        "ACS lineage does not survive: source_household_id differs from the linked household's native id",
    ),
    (
        _source_person_id_moves,
        "ACS lineage does not survive: source_person_id differs from str(SPORDER) on 1",
    ),
    (
        _source_year_moves,
        "ACS lineage does not survive: source_year must be 2024 on every ACS person; 1 differ",
    ),
    # A pandas nullable dtype's pd.NA in a lineage column (review should-fix 2).
    (
        _source_year_pd_na,
        "ACS lineage does not survive: every ACS person needs an integral "
        "source_year: 1 value(s) are null",
    ),
    (
        _source_person_id_pd_na,
        "ACS lineage does not survive: source_person_id is null on 1 ACS person(s)",
    ),
    (
        _source_household_id_pd_na,
        "ACS lineage does not survive: every ACS person needs an integral "
        "source_household_id: 1 value(s) are null",
    ),
    # Staging's lineage, person by person under native ids (review blocking 2).
    (
        _serialno_swaps_between_households,
        "keyed by native (household id, person id); 0 missing (such as []), 0 "
        "extra (such as []), 4 moved",
    ),
    (
        _sporder_swaps_within_a_household,
        "keyed by native (household id, person id); 0 missing (such as []), 0 "
        "extra (such as []), 2 moved",
    ),
    (
        _staging_has_a_person_the_selection_lost,
        "keyed by native (household id, person id); 1 missing (such as [(999999, 1)])",
    ),
    (
        _staging_lacks_a_selected_person,
        "keyed by native (household id, person id); 0 missing (such as []), 1 extra",
    ),
    (
        _staging_pair_malformed,
        "ACS lineage does not survive: staging's lineage maps (999999, 1) to "
        "('2024HU0000001',), not a (SERIALNO, SPORDER) pair",
    ),
    (
        _staging_key_malformed,
        "ACS lineage does not survive: staging's lineage has the key "
        "'2024HU0000001', not a native (household id, person id) pair of integers",
    ),
    (
        _staging_key_timedelta,
        "ACS lineage does not survive: staging's lineage has the key (np.timedelta64(",
    ),
    # P6c: group units nest in households; ACS SPM units are the households
    # (review blocking 3).
    (
        _acs_tax_unit_spans_households,
        "Group units do not nest in households: 1 ACS tax_unit unit(s) hold "
        "persons of more than one household",
    ),
    (
        _cps_family_spans_households,
        "Group units do not nest in households: 1 CPS family unit(s) hold "
        "persons of more than one household",
    ),
    (
        _acs_household_holds_two_spm_units,
        "ACS SPM units must be the households (staging's one-SPM-unit-per-"
        "household partition): 1 ACS household(s) hold more than one SPM unit",
    ),
    (
        _acs_spm_units_merge,
        "Group units do not nest in households: 1 ACS spm_unit unit(s) hold "
        "persons of more than one household",
    ),
    # Each household's weight found by its key, after P7 checked SERIALNO.
    (
        _acs_household_without_a_weight,
        "ACS household weights must name each ACS household by its "
        "acs_unit_key(SERIALNO) exactly once: 1 household(s) have no weight",
    ),
    (
        _serialno_not_a_clean_key,
        "ACS household weights need each household's SERIALNO key",
    ),
    (
        _cps_weight_names_no_household,
        "CPS household weights must name each CPS household by its household_id "
        "exactly once: 0 household(s) have no weight (such as []) and 1 weight(s) "
        "name no household",
    ),
]


def _rewrite(population, corruption: dict) -> Frame:
    return to_assembled_provenance(
        corruption.get("cps", population.cps_tables),
        corruption.get("acs", population.acs_tables),
        cps_household_weights=corruption.get(
            "cps_weights", population.keyed_cps_household_weights
        ),
        acs_household_weights=corruption.get(
            "acs_weights", population.keyed_acs_household_weights
        ),
        staging_lineage=corruption.get("lineage", staging_lineage(population)),
    )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    _REFUSALS,
    ids=[corrupt.__name__.lstrip("_") for corrupt, _ in _REFUSALS],
)
def test_each_rule_refuses_with_its_own_message(population, corrupt, message) -> None:
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        _rewrite(population, corrupt(population))


def test_refusal_messages_are_distinct() -> None:
    messages = [message for _, message in _REFUSALS]
    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize("group", ["tax_unit", "spm_unit", "family", "marital_unit"])
def test_a_cps_group_unit_across_households_refuses(population, group) -> None:
    # P6c reads every group entity on CPS rows, not only the family; on ACS
    # rows the one-SPM-unit partition would also catch a split SPM unit.
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Group units do not nest in households: 1 CPS {group} unit(s) hold "
            "persons of more than one household"
        ),
    ):
        _rewrite(population, _cps_unit_spans_households(population, group))


def _complex_membership(person: pd.DataFrame, row: int, column: str) -> None:
    """``row``'s ``column`` as a numpy complex, its real part the same id."""

    _set(person, row, column, np.complex128(int(person[column].iloc[row]) + 0j))
    assert person[column].dtype == np.dtype(object)


_GROUPS = ("household", "spm_unit", "tax_unit", "family", "marital_unit")


@pytest.mark.parametrize("group", _GROUPS)
@pytest.mark.parametrize("origin", ["acs", "cps"])
def test_a_complex_membership_refuses(population, origin, group) -> None:
    # PR #1685 review r3, nit 2: isin matches a complex 5+0j with the id 5,
    # so a person's link held that way resolved and the frame passed.
    # P6b, where the five person links resolve, reads each through the
    # shared guard first, on both origins' persons.
    tables = _copies(population, origin)
    column = US_SCHEMA.membership_column(group)
    _complex_membership(tables["person"], 0, column)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Memberships do not resolve: {origin.upper()} persons' {column!r} "
            "holds 1 value(s) that are not real numbers or strings, such as "
            "[np.complex128("
        ),
    ):
        _rewrite(population, {origin: tables})


def test_union_form_refuses_a_fractional_sporder(population) -> None:
    union = with_staging_union_columns(population)
    acs = {entity: table.copy() for entity, table in union.acs_tables.items()}
    assert acs["person"]["SPORDER"].dtype == np.dtype("float64")
    acs["person"].loc[acs["person"].index[0], "SPORDER"] = 1.5
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "every ACS person needs an integral SPORDER: 1 value(s) are not integers"
        ),
    ):
        _rewrite(union, {"acs": acs, "lineage": staging_lineage(population)})


def _with_source_year(person: pd.DataFrame, form: str) -> None:
    """``source_year`` held as complex or temporal data, each value's real part 2024."""

    years = person["source_year"]
    if form == "complex128 column":
        values = years.astype(np.complex128)
        values.iloc[0] = complex(int(years.iloc[0]), 1)
        person["source_year"] = values
    elif form == "complex in object":
        _set(person, 0, "source_year", complex(int(years.iloc[0]), 1))
    elif form == "timedelta64 column":
        person["source_year"] = pd.to_timedelta(years.to_numpy(), unit="ns")
    elif form.startswith("Arrow dictionary"):
        # Review r3, should-fix 1: the dtype's kind is "O" but it is not an
        # object dtype, so the r2 guard skipped it and pd.to_numeric read
        # each time as its count of nanoseconds.
        unit = pa.timestamp("ns") if "timestamps" in form else pa.duration("ns")
        person["source_year"] = _arrow_dictionary(years, unit)
    else:
        # The last person, not the first: pd.to_numeric reads a leading
        # numpy timedelta64 in an object column as NaN but a later one as
        # its count of units.
        _set(person, -1, "source_year", np.timedelta64(int(years.iloc[-1]), "ns"))


def _arrow_dictionary(values: pd.Series, arrow_type) -> pd.Series:
    """``values`` (integers, or null) as an Arrow dictionary of ``arrow_type``.

    A timestamp or duration value holds the integer as its count of units,
    the encoding review r3's should-fix 1 found read as that integer.
    """

    array = pa.array(
        [None if pd.isna(value) else int(value) for value in values],
        type=arrow_type,
    ).dictionary_encode()
    return pd.Series(pd.arrays.ArrowExtensionArray(array), index=values.index)


_COMPLEX_OR_TEMPORAL_FORMS = {
    "complex128 column": "it holds complex128 values, not integers",
    "complex in object": (
        "it holds 1 value(s) that are not real numbers or strings, such as [(2024+1j)]"
    ),
    "timedelta64 column": "it holds timedelta64[ns] values, not integers",
    "numpy timedelta64 in object": (
        "it holds 1 value(s) that are not real numbers or strings, such as "
        "[np.timedelta64(2024,"
    ),
    "Arrow dictionary of timestamps": (
        "it holds dictionary<values=timestamp[ns], indices=int32, ordered=0>"
        "[pyarrow] values, not integers"
    ),
    "Arrow dictionary of durations": (
        "it holds dictionary<values=duration[ns], indices=int32, ordered=0>"
        "[pyarrow] values, not integers"
    ),
}


@pytest.mark.parametrize("form", list(_COMPLEX_OR_TEMPORAL_FORMS))
def test_a_complex_or_temporal_source_year_refuses(population, form) -> None:
    # PR #1685 review r2, should-fix 1: pd.to_numeric keeps 2024+1j complex
    # and the float64 conversion after it drops the imaginary part, and it
    # reads a timedelta64 of 2024 units as 2024, as a column or as a value
    # on a later row of an object column, so P7 saw 2024 and passed the
    # frame. On the first row of an object column it read as NaN and
    # refused as not an integer, by its position alone.
    acs = _copies(population, "acs")
    _with_source_year(acs["person"], form)
    if "object" in form:
        assert acs["person"]["source_year"].dtype == np.dtype(object)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "ACS lineage does not survive: every ACS person needs an integral "
            f"source_year: {_COMPLEX_OR_TEMPORAL_FORMS[form]}"
        ),
    ):
        _rewrite(population, {"acs": acs})


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        "UInt16",
        CANONICAL_STRING_DTYPE,
        "int64[pyarrow]",
        "Arrow dictionary of int64",
        "category",
    ],
    ids=str,
)
def test_a_source_year_in_a_real_dtype_passes(population, dtype) -> None:
    # Review r3, should-fix 1: the allowlist reads these as the integer
    # 2024, at the public entry as in _integral_ids.
    acs = _copies(population, "acs")
    years = acs["person"]["source_year"]
    if dtype == "Arrow dictionary of int64":
        acs["person"]["source_year"] = _arrow_dictionary(years, pa.int64())
    elif dtype is CANONICAL_STRING_DTYPE:
        acs["person"]["source_year"] = years.astype(str).astype(dtype)
    else:
        acs["person"]["source_year"] = years.astype(dtype)
    _rewrite(population, {"acs": acs})


@pytest.mark.parametrize(
    ("values", "found"),
    [
        (pd.Series(np.array([7 + 0j, 7 + 1j])), "complex128 values"),
        (pd.Series(np.array([7 + 0j], dtype=np.complex64)), "complex64 values"),
        (
            pd.Series([7, complex(7, 0)], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, np.complex64(7)], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (pd.Series(pd.to_datetime(["2024-01-01"])), "datetime64["),
        (pd.Series(pd.to_datetime(["2024-01-01"]).tz_localize("UTC")), "datetime64["),
        (pd.Series(pd.to_timedelta([7], unit="ns")), "timedelta64[ns] values"),
        (
            pd.Series(pd.period_range("2024", periods=1, freq="Y")),
            "period[Y-DEC] values",
        ),
        (
            pd.Series([7, pd.Timestamp(7)], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, pd.Timedelta(7)], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, np.datetime64(7, "ns")], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, np.timedelta64(7, "ns")], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, pd.Period("2024", freq="Y")], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (
            pd.Series([7, datetime.date(2024, 1, 1)], dtype=object),
            "1 value(s) that are not real numbers or strings",
        ),
        (pd.Series([7 + 1j], dtype="category"), "categories of complex128 values"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_integral_ids_refuse_complex_and_temporal_data(values, found) -> None:
    # Review r2, should-fix 1: the shared id parser refuses each before
    # pd.to_numeric and the float64 conversion read it. Before the fix the
    # complex cases, the datetime64 and timedelta64 columns, the object
    # column holding np.timedelta64 and the category passed as numbers; the
    # period column and the object columns holding a Timestamp, Timedelta,
    # np.datetime64, Period or date refused as "not integers".
    with pytest.raises(
        OnePopulationContractError, match=re.escape(f"ids: it holds {found}")
    ):
        contract._integral_ids(values, refusal="ids")


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([7, 8]),
        pd.Series([7.0, 8.0]),
        pd.Series([7, 8], dtype="Int64"),
        pd.Series([7, "8"], dtype=object),
        pd.Series([7, 8], dtype="category"),
    ],
    ids=["int64", "integral float64", "Int64", "object", "category"],
)
def test_integral_ids_still_read_real_ids(values) -> None:
    np.testing.assert_array_equal(
        contract._integral_ids(values, refusal="ids"), np.array([7, 8])
    )


def _arrow(values: list, arrow_type) -> pd.Series:
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(values, type=arrow_type)))


@pytest.mark.parametrize(
    ("values", "found"),
    [
        (
            _arrow_dictionary(pd.Series([7, 8]), pa.timestamp("ns")),
            "dictionary<values=timestamp[ns], indices=int32, ordered=0>[pyarrow] "
            "values",
        ),
        (
            _arrow_dictionary(pd.Series([7, 8]), pa.duration("ns")),
            "dictionary<values=duration[ns], indices=int32, ordered=0>[pyarrow] values",
        ),
        (
            pd.Series(pd.Categorical(_arrow([7, 8], pa.duration("ns")))),
            "categories of duration[ns][pyarrow] values",
        ),
        (
            _arrow([decimal.Decimal(7), decimal.Decimal(8)], pa.decimal128(10, 0)),
            "decimal128(10, 0)[pyarrow] values",
        ),
        (_arrow(["7", "8"], pa.string_view()), "string_view[pyarrow] values"),
        (pd.Series(pd.interval_range(7, 9)), "interval[int64, right] values"),
        (
            pd.Series([7, np.array(8, dtype="m8[ns]")], dtype=object),
            "1 value(s) that are not real numbers or strings, such as "
            "[array(8, dtype='timedelta64[ns]')]",
        ),
        (
            pd.Series([7, np.array(8 + 0j)], dtype=object),
            "1 value(s) that are not real numbers or strings, such as [array(8.+0.j)]",
        ),
    ],
    ids=[
        "Arrow dictionary of timestamps",
        "Arrow dictionary of durations",
        "category of Arrow durations",
        "Arrow decimal",
        "Arrow string_view",
        "interval",
        "zero-dimensional timedelta64 array in object",
        "zero-dimensional complex array in object",
    ],
)
def test_integral_ids_refuse_any_data_but_integers_reals_and_strings(
    values, found
) -> None:
    # Review r3, should-fix 1: the guard is an allowlist. An Arrow
    # dictionary's dtype has kind "O" but is not an object dtype, so the r2
    # guard skipped it and pd.to_numeric read each time as its count; it
    # also passed an Arrow decimal, raised NotImplementedError on an Arrow
    # string_view, and passed a zero-dimensional numpy array in an object
    # column, which pd.to_numeric reads as its count or its real part.
    with pytest.raises(
        OnePopulationContractError, match=re.escape(f"ids: it holds {found}")
    ):
        contract._integral_ids(values, refusal="ids")


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([7, 8], dtype=np.int32),
        pd.Series([7, 8], dtype=np.uint64),
        pd.Series([7.0, 8.0], dtype=np.float32),
        pd.Series([7, 8], dtype="UInt32"),
        pd.Series([7.0, 8.0], dtype="Float64"),
        pd.Series(["7", "8"], dtype=CANONICAL_STRING_DTYPE),
        pd.Series(["7", "8"]),
        pd.Series(["7", "8"], dtype="string[pyarrow]"),
        pd.Series(["7", "8"], dtype="category"),
        pd.Series([7, np.int64(8)], dtype=object),
        _arrow([7, 8], pa.int64()),
        _arrow([7, 8], pa.uint8()),
        _arrow([7.0, 8.0], pa.float64()),
        _arrow(["7", "8"], pa.string()),
        _arrow(["7", "8"], pa.large_string()),
        _arrow_dictionary(pd.Series([7, 8]), pa.int64()),
        pd.Series(
            pd.arrays.ArrowExtensionArray(pa.array(["7", "8"]).dictionary_encode())
        ),
    ],
    ids=[
        "int32",
        "uint64",
        "float32",
        "UInt32",
        "Float64",
        "canonical string",
        "pandas default string",
        "string[pyarrow]",
        "category of strings",
        "object of numpy ints",
        "Arrow int64",
        "Arrow uint8",
        "Arrow float64",
        "Arrow string",
        "Arrow large_string",
        "Arrow dictionary of int64",
        "Arrow dictionary of strings",
    ],
)
def test_integral_ids_read_the_pinned_id_dtypes(values) -> None:
    # Review r3, should-fix 1, the lead's call: the allowlist must not refuse
    # a real input. The generator writes int64 ids, staging's pooled form
    # float64 (both in the test above); the parent export's H5 load and
    # staging's PUMS parse store strings in CANONICAL_STRING_DTYPE
    # (serialization_dtypes); the review's probes kept unsigned, nullable,
    # categorical and Arrow numeric ids valid.
    np.testing.assert_array_equal(
        contract._integral_ids(values, refusal="ids"), np.array([7, 8])
    )


def test_a_cross_grain_channel_mismatch_fails_the_validator(population) -> None:
    # No P5-P7 rule reads a person's channel against its groups'; the
    # assembly validator does, and its ValueError comes back as the
    # contract's error. A native CPS person moves into its household's
    # PUF tax-detail copy (the native household keeps another member).
    cps = _copies(population, "cps")
    person = cps["person"]
    native = person["person_support_clone_index"] == 0
    sizes = person["person_household_id"].map(
        person.loc[native, "person_household_id"].value_counts()
    )
    row = int(np.flatnonzero(native.to_numpy() & (sizes.to_numpy() > 1))[0])
    household = int(person["person_household_id"].iloc[row])
    _set(person, row, "person_household_id", household + population.clone_multiplier)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape("The joined one-population frame is not valid")
        + ".*cross-grain provenance disagrees",
    ):
        _rewrite(population, {"cps": cps})


# ------------------------------------------------------------ cps_unit_ids


class TestCpsUnitIds:
    def test_keys_are_the_selection_keys(self, population) -> None:
        person = population.cps_tables["person"]
        household = population.cps_tables["household"].set_index("household_id")
        ids = cps_unit_ids(person)
        assert ids.index.name == "household_id"
        assert ids.index.tolist() == sorted(household.index)
        assert ids["income_year"].dtype == np.dtype("int64")
        native = household["household_support_clone_index"] == 0
        native_keys = ids.loc[native[native].index, "key"].tolist()
        assert sorted(native_keys) == sorted(population.cps_keys)
        assert selection.keys_sha256(native_keys) == selection.keys_sha256(
            population.cps_keys
        )
        # A clone household shares its source household's key.
        source = household["household_source_id"]
        assert (ids["key"] == ids["key"].loc[source.loc[ids.index]].to_numpy()).all()
        assert (ids["unit_id"].str.len() == 20).all()
        assert (
            ids["key"]
            == [
                selection.cps_unit_key(year, unit)
                for year, unit in zip(ids["income_year"], ids["unit_id"], strict=True)
            ]
        ).all()

    def test_reads_source_person_id_when_peridnum_is_absent(self, population) -> None:
        person = population.cps_tables["person"]
        pd.testing.assert_frame_equal(
            cps_unit_ids(person.drop(columns=["PERIDNUM"])), cps_unit_ids(person)
        )

    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            (
                lambda person: person.drop(columns=["PERIDNUM"]).assign(
                    source_person_id=np.arange(len(person))
                ),
                "need source_person_id as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(PERIDNUM=person["PERIDNUM"].astype(float)),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(PERIDNUM=person["PERIDNUM"].str[:19]),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(
                    PERIDNUM=person["PERIDNUM"].where(person.index != 0, None)
                ),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.drop(columns=["PERIDNUM", "source_person_id"]),
                "need PERIDNUM or source_person_id",
            ),
            (
                lambda person: person.assign(
                    source_year=person["source_year"]
                    .astype(float)
                    .where(person.index != 0, np.nan)
                ),
                "need an integral source_year",
            ),
        ],
        ids=["integer_fallback", "float", "short", "null", "absent", "null_year"],
    )
    def test_refuses_a_malformed_id(self, population, corrupt, message) -> None:
        person = population.cps_tables["person"].copy()
        with pytest.raises(OnePopulationContractError, match=re.escape(message)):
            cps_unit_ids(corrupt(person))

    def test_refuses_acs_rows(self, frame) -> None:
        with pytest.raises(
            OnePopulationContractError, match="as a string of at least 20 digits"
        ):
            cps_unit_ids(frame.table("person"))

    @pytest.mark.parametrize(
        ("column", "message"),
        [
            ("PERIDNUM", "disagree on PERIDNUM[:20] (H_IDNUM)"),
            ("source_year", "disagree on source_year"),
        ],
    )
    def test_refuses_a_household_whose_persons_disagree(
        self, population, column, message
    ) -> None:
        person = population.cps_tables["person"].copy()
        sizes = person["person_household_id"].map(
            person["person_household_id"].value_counts()
        )
        row = person.index[int(np.flatnonzero(sizes.to_numpy() > 1)[0])]
        if column == "PERIDNUM":
            value = person.loc[row, "PERIDNUM"]
            person.loc[row, "PERIDNUM"] = ("1" if value[0] != "1" else "2") + value[1:]
        else:
            person.loc[row, "source_year"] = person.loc[row, "source_year"] - 1
        with pytest.raises(OnePopulationContractError, match=re.escape(message)):
            cps_unit_ids(person)


# ------------------------------- phase 2: require_one_population (R1-R12, T1)


@pytest.fixture(scope="module")
def assembled(population):
    return assembled_frame(population)


def _require(population, frame: Frame, **overrides) -> None:
    kwargs = {
        "parent_tail_donor_ids": population.tail_donor_ids,
        "variable_index": StandInVariableIndex.from_lock(),
    }
    kwargs.update(overrides)
    manifest = kwargs.pop("manifest", None) or selection_manifest(population)
    require_one_population(frame, manifest, **kwargs)


def _tables(frame: Frame) -> dict[str, pd.DataFrame]:
    return {entity: frame.table(entity).copy() for entity in frame.entities}


def _rebuild(
    frame: Frame,
    tables: dict[str, pd.DataFrame],
    *,
    metadata=None,
) -> Frame:
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata if metadata is None else metadata,
    )


def _origin_mask(table: pd.DataFrame, entity: str, origin: str) -> np.ndarray:
    channel = table[support_channel_column(entity)]
    if origin == "acs":
        return channel.eq(ACS_2024_1YR_SPINE).to_numpy()
    return channel.ne(ACS_2024_1YR_SPINE).to_numpy()


def _row(table: pd.DataFrame, entity: str, origin: str, *, clone=None, where=None):
    """Position of the first row of ``origin`` (and clone index, predicate)."""

    mask = np.array(_origin_mask(table, entity, origin), dtype=bool)
    if clone is not None:
        mask &= table[support_clone_index_column(entity)].to_numpy() == clone
    if where is not None:
        mask &= np.asarray(where, dtype=bool)
    return int(np.flatnonzero(mask)[0])


def _cell(frame: Frame, entity: str, column: str, value, *, origin, **row):
    tables = _tables(frame)
    table = tables[entity]
    _set(table, _row(table, entity, origin, **row), column, value)
    return _rebuild(frame, tables)


def _with_column(frame: Frame, entity: str, column: str, values) -> Frame:
    tables = _tables(frame)
    tables[entity][column] = values
    return _rebuild(frame, tables)


def _without_column(frame: Frame, entity: str, column: str) -> Frame:
    tables = _tables(frame)
    tables[entity] = tables[entity].drop(columns=[column])
    return _rebuild(frame, tables)


def _household_sizes(person: pd.DataFrame, column: str) -> np.ndarray:
    return person[column].map(person[column].value_counts()).to_numpy()


def _acs_null_on_cps(frame: Frame, entity: str, column: str, value) -> Frame:
    """``column`` filled with ``value`` on CPS rows and null on ACS rows."""

    table = frame.table(entity)
    acs = _origin_mask(table, entity, "acs")
    return _with_column(
        frame,
        entity,
        column,
        pd.Series(np.where(acs, None, value), index=table.index, dtype=object),
    )


#: The engine inputs d2283 lead call L1 (11-18) moved off the exempt list:
#: the release's first engine pass refuses a null in each float or integer
#: one at load, and later passes compute from is_self_employed, before their
#: producers write them, so prepare fills each on ACS rows and R6 refuses its
#: null. The
#: three integer recodes are the ones staging writes natively. The ACA pass
#: computes from takes_up_medicaid_if_eligible before Medicaid writes it,
#: masked today; it moved with the lead-call review (fail closed).
_MOVED_OFF_THE_EXEMPT_LIST = (
    "cps_race",
    "hourly_wage",
    "fsla_overtime_premium",
    "selected_marketplace_plan_benchmark_ratio",
    "other_health_insurance_premiums",
    "health_insurance_premiums",
    "is_self_employed",
    "detailed_occupation_recode",
    "detailed_industry_recode",
    "major_industry_recode",
    "takes_up_medicaid_if_eligible",
)
#: The moved inputs the engine holds as booleans.
_MOVED_BOOLEANS = frozenset({"is_self_employed", "takes_up_medicaid_if_eligible"})


class TestRequireOnePopulation:
    @pytest.mark.parametrize("form", _FORMS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_a_valid_population_passes(self, seed, form) -> None:
        population = _form(generate_population(seed), form)
        frame = assembled_frame(population)
        _require(population, frame)
        _require(population, frame, staging_lineage=staging_lineage(population))

    def test_a_frame_with_no_acs_row_passes(self, population) -> None:
        # T1 and P7 are vacuous, R2's ACS half is the empty key set and
        # R10 has no ACS row to read.
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        manifest = dataclasses.replace(
            selection_manifest(population),
            n_acs=0,
            selected_acs_keys_sha256=selection.keys_sha256(()),
        )
        assert "SERIALNO" not in frame.table("household")
        _require(population, frame, manifest=manifest, staging_lineage={})

    def test_a_frame_with_no_acs_row_still_checks_staging_lineage(
        self, population
    ) -> None:
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        manifest = dataclasses.replace(
            selection_manifest(population),
            n_acs=0,
            selected_acs_keys_sha256=selection.keys_sha256(()),
        )
        lineage = staging_lineage(population)
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "ACS lineage does not survive: the selected ACS persons' (SERIALNO, "
                "SPORDER) lineage differs from staging's, keyed by native "
                f"(household id, person id); {len(lineage)} missing"
            ),
        ):
            _require(population, frame, manifest=manifest, staging_lineage=lineage)

    def test_the_default_index_is_built_at_call_time(
        self, population, assembled, monkeypatch
    ) -> None:
        built: list[object] = []

        def fake_default():
            built.append(None)
            return StandInVariableIndex.from_lock()

        monkeypatch.setattr(contract, "PolicyEngineUSVariableIndex", fake_default)
        assert built == []
        _require(population, assembled, variable_index=None)
        assert len(built) == 1

    def test_the_default_register_is_the_stored_inputs_register(
        self, population, assembled
    ) -> None:
        frame = _with_column(
            assembled, "person", "cps_employment_income_last_year", 0.0
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("['cps_employment_income_last_year']"),
        ):
            _require(population, frame)
        register = {
            **US_STORED_NON_VARIABLE_COLUMNS,
            "cps_employment_income_last_year": "test auxiliary column",
        }
        _require(population, frame, register=register)

    def test_tail_donors_are_compared_without_weights(
        self, population, assembled
    ) -> None:
        tables = _tables(assembled)
        tables["tax_unit"]["puf_capital_gains_tail_transfer_weight"] *= 3.0
        weights = assembled.weights_for("household")
        frame = Frame(
            tables,
            assembled.schema,
            {"household": type(weights)(weights.values * 2.0, weights.kind)},
            assembled.strata,
            mass_log=assembled.mass_log,
            metadata=assembled.metadata,
        )
        _require(population, frame)

    def test_a_post_2024_formula_column_with_nulls_refuses_under_r6(
        self, population, assembled, stand_in
    ) -> None:
        # weeks_worked has only a 2025 formula at policyengine-us 2.2.1: the
        # year-blind input list (the lock) lacks it and R3 does not refuse
        # it, so R6's year-aware input rule is what catches a null.
        assert "weeks_worked" not in lock_engine_inputs()
        frame = _acs_null_on_cps(assembled, "person", "weeks_worked", 52.0)
        tables = {entity: frame.table(entity) for entity in frame.entities}
        assert "weeks_worked" not in stand_in.computed_at_2024(tables)
        assert "weeks_worked" in stand_in.inputs_at_2024(tables)
        n_acs = int(_origin_mask(frame.table("person"), "person", "acs").sum())
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                f"A null in an engine input at 2024: {{'person.weeks_worked': {n_acs}}}"
            ),
        ):
            _require(population, frame)
        _require(population, _with_column(assembled, "person", "weeks_worked", 52.0))

    def test_a_register_column_absent_on_acs_rows_needs_a_declaration(
        self, population, assembled
    ) -> None:
        register = {
            **US_STORED_NON_VARIABLE_COLUMNS,
            "cps_employment_income_last_year": "test auxiliary column",
        }
        frame = _acs_null_on_cps(
            assembled, "person", "cps_employment_income_last_year", 1.0
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("'person.cps_employment_income_last_year'"),
        ):
            _require(population, frame, register=register)
        _require(
            population,
            frame,
            register=register,
            declared_absences={
                "cps_employment_income_last_year": "owner: ACS has no prior year"
            },
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("A declared absence needs its owner's decision"),
        ):
            _require(
                population,
                frame,
                register=register,
                declared_absences={"cps_employment_income_last_year": "  "},
            )

    def test_a_stage_output_outside_the_exempt_list_is_not_exempt(
        self, population, assembled
    ) -> None:
        # scf_wealth declares the education-debt columns, but only an opt-in
        # stage writes them: R10 does not exempt a stage's outputs whole.
        assert "federal_student_loan_balance" in US_STORED_NON_VARIABLE_COLUMNS
        assert "federal_student_loan_balance" not in RELEASE_WRITTEN_INPUTS
        frame = _acs_null_on_cps(
            assembled, "person", "federal_student_loan_balance", 0.0
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A model-named column held on some CPS row is null on ACS rows "
                "without a declared absence: {'person.federal_student_loan_balance'"
            ),
        ):
            _require(population, frame)

    def test_a_release_written_input_may_be_null_on_acs_rows(
        self, population, assembled
    ) -> None:
        frame = _acs_null_on_cps(
            assembled, "person", "takes_up_head_start_if_eligible", True
        )
        _require(population, frame)
        not_exempt = _acs_null_on_cps(
            assembled, "spm_unit", "takes_up_snap_if_eligible", True
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("'spm_unit.takes_up_snap_if_eligible'"),
        ):
            _require(population, not_exempt)

    @pytest.mark.parametrize(
        ("null", "dtype", "filled"),
        [
            (None, object, True),
            (np.nan, object, True),
            (np.nan, "float64", 1.0),
            (np.nan, "float32", 1.0),
            (np.float32("nan"), object, True),
        ],
        ids=[
            "none",
            "nan in object",
            "nan in float64",
            "nan in float32",
            "numpy float32 nan in object",
        ],
    )
    def test_a_release_written_null_may_be_none_or_nan(
        self, population, assembled, null, dtype, filled
    ) -> None:
        # PR #1685 review, blocking 1: the release's first engine pass loads
        # a boolean None as False and a NaN as True (numpy's casts), so both
        # stay allowed in an exempt column; pd.NA and every other null refuse
        # (the tests below). Review r2: a float32 column's NaN passes by its
        # dtype, and a numpy float32 NaN in an object column, which is not a
        # Python float, passes as a float NaN.
        person = assembled.table("person")
        acs = _origin_mask(person, "person", "acs")
        values = pd.Series(
            [null if row else filled for row in acs], index=person.index, dtype=dtype
        )
        if dtype is object:
            assert type(values[acs].iloc[0]) is type(null)
        _require(
            population,
            _with_column(assembled, "person", "takes_up_ssi_if_eligible", values),
        )

    @pytest.mark.parametrize("dtype", ["Float32", "Float64"])
    def test_a_nullable_float_nan_that_is_not_pd_na_passes(
        self, population, assembled, dtype
    ) -> None:
        # Review r2, the lead's call on nit 1: a nullable Float column can
        # hold a NaN unmasked, which pandas does not count as null, so R6
        # passes it; only the column's masked null, pd.NA, refuses.
        person = assembled.table("person")
        acs = _origin_mask(person, "person", "acs")
        values = pd.Series(
            pd.arrays.FloatingArray(
                np.where(acs, np.nan, 1.0).astype(dtype.lower()),
                np.zeros(len(acs), dtype=bool),
            ),
            index=person.index,
        )
        assert str(values.dtype) == dtype
        assert not values.isna().any() and np.isnan(values.to_numpy()[acs]).all()
        _require(
            population,
            _with_column(assembled, "person", "takes_up_ssi_if_eligible", values),
        )

    @pytest.mark.parametrize(
        ("null", "origin", "dtype", "filled"),
        [
            (pd.NaT, "acs", object, True),
            (np.datetime64("NaT"), "cps", object, True),
            (np.timedelta64("NaT"), "acs", object, True),
            (
                np.datetime64("NaT", "ns"),
                "acs",
                "datetime64[ns]",
                np.datetime64(1, "ns"),
            ),
            (
                np.timedelta64("NaT", "ns"),
                "acs",
                "timedelta64[ns]",
                np.timedelta64(1, "ns"),
            ),
        ],
        ids=[
            "pd.NaT on ACS rows",
            "numpy NaT on a CPS row",
            "timedelta NaT on ACS rows",
            "datetime64 column",
            "timedelta64 column",
        ],
    )
    def test_a_null_other_than_none_or_nan_refuses(
        self, population, assembled, null, origin, dtype, filled
    ) -> None:
        # Review r2, nit 1: R6 lets a release-written column be null only as
        # None or a float NaN. Pandas also counts pd.NaT and numpy's NaT as
        # null; R6 used to refuse only pd.NA, so they passed. They refuse on
        # either origin, in an object column or in a datetime64 or
        # timedelta64 column, whose only null is NaT.
        person = assembled.table("person")
        rows = _origin_mask(person, "person", origin)
        if origin == "cps":
            rows = np.arange(len(person)) == int(np.flatnonzero(rows)[0])
        values = pd.Series(
            [null if row else filled for row in rows], index=person.index, dtype=dtype
        )
        assert values.dtype == np.dtype(dtype)
        assert int(values.isna().sum()) == int(rows.sum()) > 0
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A null other than None or NaN in a release-written input: "
                f"{{'person.takes_up_ssi_if_eligible': {int(rows.sum())}}}"
            ),
        ):
            _require(
                population,
                _with_column(assembled, "person", "takes_up_ssi_if_eligible", values),
            )

    def test_a_lone_pd_na_on_a_cps_row_refuses(self, population, assembled) -> None:
        # Review r2, nit 4: R6 reads every row of a release-written column,
        # not only ACS rows; an ACS-only scan would pass this frame.
        person = assembled.table("person")
        row = _row(person, "person", "cps")
        values = pd.array([True] * len(person), dtype="boolean")
        values[row] = pd.NA
        frame = _with_column(
            assembled,
            "person",
            "takes_up_ssi_if_eligible",
            pd.Series(values, index=person.index),
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A pd.NA in a release-written input: "
                "{'person.takes_up_ssi_if_eligible': 1}"
            ),
        ):
            _require(population, frame)

    def test_a_pd_na_is_reported_before_another_null(
        self, population, assembled
    ) -> None:
        # Review r3, nit 3: R6 refuses a pd.NA in any release-written column
        # before any other null, so a pd.NaT on an ACS row of one column
        # cannot take the diagnostic from a pd.NA on a CPS row of another.
        person = assembled.table("person")
        nat_row = _row(person, "person", "acs")
        frame = _with_column(
            assembled,
            "person",
            "takes_up_ssi_if_eligible",
            pd.Series(
                [pd.NaT if row == nat_row else True for row in range(len(person))],
                index=person.index,
                dtype=object,
            ),
        )
        column = "would_file_taxes_voluntarily"
        entity = lock_engine_inputs()[column]
        table = frame.table(entity)
        values = pd.array([True] * len(table), dtype="boolean")
        values[_row(table, entity, "cps")] = pd.NA
        frame = _with_column(
            frame, entity, column, pd.Series(values, index=table.index)
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                f"A pd.NA in a release-written input: {{'{entity}.{column}': 1}}"
            ),
        ):
            _require(population, frame)

    @pytest.mark.parametrize(
        ("column", "dtype", "filled"),
        [
            ("takes_up_ssi_if_eligible", "boolean", True),
            ("would_file_taxes_voluntarily", "boolean", True),
            ("bank_account_assets", "Float64", 1_000.0),
        ],
    )
    def test_a_pd_na_in_a_release_written_input_refuses(
        self, population, assembled, column, dtype, filled
    ) -> None:
        # Blocking 1: a pandas nullable dtype holds its null as pd.NA, which
        # the H5 codec keeps and the first engine pass cannot cast to bool
        # (bool(pd.NA) raises TypeError). The SSI December pass loads the
        # twelve booleans before their producers write them; R6 refuses a
        # pd.NA in every exempt column, the SCF wealth amounts included,
        # fail closed. Before the fix R6 left exempt columns unread.
        entity = lock_engine_inputs()[column]
        table = assembled.table(entity)
        acs = _origin_mask(table, entity, "acs")
        values = pd.array([pd.NA if row else filled for row in acs], dtype=dtype)
        frame = _with_column(
            assembled, entity, column, pd.Series(values, index=table.index)
        )
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A pd.NA in a release-written input: "
                f"{{'{entity}.{column}': {int(acs.sum())}}}"
            ),
        ):
            _require(population, frame)

    def test_r2_parses_cps_clone_indices_as_r1_does(
        self, population, assembled
    ) -> None:
        # PR #1685 review, should-fix 3: R1 and the assembly validator take
        # digit-string clone indices through the integral parser (lead call
        # L5's leniency). R2 compared the raw values with 0, found no clone-0
        # household and refused every CPS key as orphaned.
        tables = _tables(assembled)
        household = tables["household"]
        column = support_clone_index_column("household")
        household[column] = [str(int(value)) for value in household[column]]
        assert not pd.api.types.is_numeric_dtype(household[column].dtype)
        _require(population, _rebuild(assembled, tables))

    def test_r2_parses_string_household_ids(self, population) -> None:
        # Should-fix 3's second shape, on R2 itself: string household ids
        # with matching string memberships on a frame with no ACS row.
        # cps_unit_ids keys households by int64, which a raw string index
        # joined to nothing.
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        manifest = dataclasses.replace(
            selection_manifest(population),
            n_acs=0,
            selected_acs_keys_sha256=selection.keys_sha256(()),
        )
        tables = _tables(frame)
        for entity, column in (
            ("household", "household_id"),
            ("person", "person_household_id"),
        ):
            tables[entity][column] = [
                str(int(value)) for value in tables[entity][column]
            ]
        contract._require_selected_keys(contract._split_origins(tables), manifest)

    def test_an_index_that_omits_an_input_cannot_hide_its_null(
        self, population, assembled
    ) -> None:
        # PR #1685 review, should-fix 4: an index that leaves a known,
        # non-computed column out of its inputs hid that column's null from
        # R6 (which reads only the inputs) and R10 (which skips every known
        # column). The coherence check now refuses the index itself.
        class OmitsTipIncome(StandInVariableIndex):
            def inputs_at_2024(self, tables):
                return frozenset(super().inputs_at_2024(tables) - {"tip_income"})

        frame = _cell(assembled, "person", "tip_income", None, origin="acs")
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "The engine variable index answers do not cohere (the inputs must "
                "be exactly the known columns not computed at 2024 and not "
                "structural): inputs omit known columns that are neither computed "
                "at 2024 nor structural: ['tip_income']"
            ),
        ):
            _require(
                population,
                frame,
                variable_index=OmitsTipIncome(inputs=frozenset(lock_engine_inputs())),
            )

    @pytest.mark.parametrize("column", _MOVED_OFF_THE_EXEMPT_LIST)
    def test_a_moved_input_null_on_an_acs_row_refuses(
        self, population, assembled, column
    ) -> None:
        # d2283 lead call L1: each moved input is R6's on every row. The
        # generator carries none of them, so each is added on its entity.
        entity = lock_engine_inputs()[column]
        value = True if column in _MOVED_BOOLEANS else 1.0
        filled = _with_column(assembled, entity, column, value)
        _require(population, filled)
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                f"A null in an engine input at 2024: {{'{entity}.{column}': 1}}"
            ),
        ):
            _require(population, _cell(filled, entity, column, None, origin="acs"))

    def test_arguments_are_type_checked(self, population, assembled) -> None:
        with pytest.raises(TypeError, match="frame must be a Frame"):
            _require(population, assembled.table("person"))
        with pytest.raises(TypeError, match="manifest must be a SelectionManifest"):
            require_one_population(
                assembled, {}, parent_tail_donor_ids=population.tail_donor_ids
            )
        with pytest.raises(TypeError, match="parent_tail_donor_ids must hold int"):
            _require(population, assembled, parent_tail_donor_ids={True})
        with pytest.raises(TypeError, match="declared_absences must map"):
            _require(population, assembled, declared_absences=["x"])

    def test_timedelta_parent_donor_ids_refuse(self, population, assembled) -> None:
        # PR #1685 review r3, should-fix 2: numpy registers timedelta64 as
        # numbers.Integral and int() reads a nanosecond one as its count, so
        # these ids read as the parent's own and R8 passed the frame.
        donors = {
            np.timedelta64(int(donor), "ns") for donor in population.tail_donor_ids
        }
        with pytest.raises(
            TypeError,
            match=re.escape("parent_tail_donor_ids must hold int donor ids only."),
        ):
            _require(population, assembled, parent_tail_donor_ids=donors)


class TestExemptList:
    """RELEASE_WRITTEN_INPUTS: d2283 Q1 (a), re-read at the module's commit."""

    @staticmethod
    def _stage_outputs() -> dict[str, list[str]]:
        from importlib.resources import files

        payload = json.loads(
            files("microcosm.build.us")
            .joinpath("source_stages.json")
            .read_text(encoding="utf-8")
        )
        return {stage["stage"]: stage["outputs"] for stage in payload["stages"]}

    def test_each_entry_names_a_stage_that_declares_it(self) -> None:
        outputs = self._stage_outputs()
        for column, evidence in RELEASE_WRITTEN_INPUTS.items():
            match = re.match(r"(\w+) stage: ", evidence)
            assert match, (column, evidence)
            stage = match.group(1)
            assert stage in outputs, (column, stage)
            assert column in outputs[stage], (column, stage)

    def test_each_entry_cites_its_lines(self) -> None:
        for column, evidence in RELEASE_WRITTEN_INPUTS.items():
            assert re.search(r"\w+\.py:\d+", evidence), column
            # d2283 lead call L1: each entry states its earlier readers.
            assert "Earlier readers: " in evidence, column

    def test_the_list_is_the_decided_list(self) -> None:
        outputs = self._stage_outputs()
        org_bools = {
            "is_hispanic",
            "worked_last_year",
            "has_never_worked",
            "is_military",
            "is_computer_scientist",
            "is_executive_administrative_professional",
            "is_farmer_fisher",
            "is_paid_hourly",
            "is_union_member_or_covered",
        }
        assert set(RELEASE_WRITTEN_INPUTS) == {
            "bank_account_assets",
            "bond_assets",
            "stock_assets",
            "net_worth",
            "meets_ssi_disability_criteria",
            "takes_up_head_start_if_eligible",
            "takes_up_ssi_if_eligible",
            "would_file_taxes_voluntarily",
            *org_bools,
            "takes_up_aca_if_eligible",
        }
        assert len(RELEASE_WRITTEN_INPUTS) == 18
        assert len(outputs["org_wages"]) == 15
        moved = set(_MOVED_OFF_THE_EXEMPT_LIST)
        assert set(outputs["org_wages"]) == org_bools | (
            moved & set(outputs["org_wages"])
        )
        assert len(moved & set(outputs["org_wages"])) == 6
        assert moved - set(outputs["org_wages"]) <= set(
            outputs["aca_marketplace_inputs"]
        ) | set(outputs["other_health_insurance_premiums"]) | set(
            outputs["medicaid_take_up"]
        )
        not_exempt = {
            # d2283 lead call L1: the first engine pass refuses each null
            # float or integer at load, and later passes compute from
            # is_self_employed, before the producers write them.
            *_MOVED_OFF_THE_EXEMPT_LIST,
            # SNAP take-up passes ACS nulls to a gate that misgrades them.
            "takes_up_snap_if_eligible",
            # Pass-through producers.
            "is_snap_abawd_discretionary_exempt",
            "auto_loan_balance",
            "auto_loan_interest",
            "qualified_passenger_vehicle_loan_interest",
            *outputs["vehicle_assets"],
            *outputs["sipp_tips"],
            # A stage's source column.
            "health_insurance_premiums_without_medicare_part_b",
            # The hours stage is skipped on the pool branch.
            *outputs["hours_worked"],
        }
        assert not_exempt.isdisjoint(RELEASE_WRITTEN_INPUTS)
        assert not any(
            column.startswith("scf_") or "student_loan" in column
            for column in RELEASE_WRITTEN_INPUTS
        )

    def test_every_entry_is_an_engine_input_in_the_lock(self) -> None:
        # So R10 (register entries only) never sees one: each is R6's.
        assert set(RELEASE_WRITTEN_INPUTS) <= set(lock_engine_inputs())

    def test_module_constants_match_their_sources(self) -> None:
        assert contract._LOCATION_CLONE_INDEX_SUFFIX == (
            "_" + LOCATION_CLONE_INDEX_COLUMN
        )
        assert set(PUF_TAIL_TAX_UNIT_COLUMNS) == set(TAIL_PROVENANCE_DEFAULTS)
        assert "unrecaptured_section_1250_gain" not in PUF_TAIL_TAX_UNIT_COLUMNS


# ---- what the exempt list's verdicts rest on (d2283 lead-call review, 11-19)

#: The release tool whose pool arm d2283 lead call L1's trace read.
_RELEASE_TOOL = REPOSITORY_ROOT / "tools" / "build_us_fiscal_refresh_release.py"
#: The calls whose order the exempt list's verdicts read, in the release's
#: order: the three producers no engine pass precedes, the first engine
#: pass (the SSI December pass), the producers whose outputs only that pass
#: loads first, and the passes that compute from the moved premiums and
#: self-employment flag before their producer, other health insurance.
_TRACED_STAGE_ORDER = (
    "with_us_scf_wealth_inputs",
    "with_us_ssi_disability_criteria",
    "with_us_sipp_head_start_input",
    "_ssi_person_uncapped_amount",
    "with_us_ssi_take_up",
    "with_us_voluntary_filing_input",
    "with_us_org_wages_inputs",
    "_with_aca_marketplace_source_outputs",
    "_with_medicaid_take_up_outputs",
    "_with_snap_state_take_up_outputs",
    "with_us_other_health_insurance_inputs",
)
#: The release's engine passes built from the frame, up to and including
#: the other-health-insurance stage's own (L1's trace: E1 to E5).
_TRACED_FRAME_PASSES = (
    "_ssi_person_uncapped_amount",
    "_with_aca_marketplace_source_outputs",
    "_with_medicaid_take_up_outputs",
    "_with_snap_state_take_up_outputs",
    "with_us_other_health_insurance_inputs",
)
#: The calls that build an engine pass from a frame. An engine made only to
#: read variable metadata (``PolicyEngineUSEngine()``) and a dataset read
#: from a file (``_load_frame``'s ``USSingleYearDataset``) are not passes.
_FRAME_PASS_CALLS = frozenset({"_dataset_from_frame", "Microsimulation", "materialize"})
#: The engine releases L1's trace read statically: policyengine-us's
#: variables and policyengine-core's null handling at load.
_L1_TRACED_ENGINE = {"policyengine-us": "2.2.1", "policyengine-core": "3.32.5"}
_RETRACE = (
    "re-run d2283 lead call L1's earlier-reader trace over "
    "RELEASE_WRITTEN_INPUTS and the inputs moved off it before this changes, "
    "then update their evidence and this test"
)


class _SourceCalls:
    """Which functions build a frame engine pass, read from source text.

    Nothing is imported. A function builds a pass when it calls a name in
    :data:`_FRAME_PASS_CALLS`, or a function that does, followed by name
    through its own module (methods and nested functions included) and its
    ``microcosm`` imports. Following by name over-approximates, which errs
    toward a failure; a call through a variable or a class is not followed.
    """

    def __init__(self) -> None:
        self._parsed: dict[Path, tuple[dict, dict]] = {}
        self._builds: dict[tuple[Path, str], bool] = {}

    @staticmethod
    def calls(node: ast.AST) -> list[tuple[int, str]]:
        found = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else getattr(func, "attr", None)
                )
                if name:
                    found.append((sub.lineno, name))
        return sorted(found)

    @staticmethod
    def module_path(module: str) -> Path | None:
        if not module.startswith("microcosm"):
            return None
        for src in sorted((REPOSITORY_ROOT / "packages").glob("*/src")):
            base = src.joinpath(*module.split("."))
            for candidate in (base / "__init__.py", base.with_suffix(".py")):
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _absolute(path: Path, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        package = path.parent
        for _ in range(node.level - 1):
            package = package.parent
        src = next(parent for parent in package.parents if parent.name == "src")
        parts = package.relative_to(src).parts
        return ".".join((*parts, *([node.module] if node.module else [])))

    def parsed(self, path: Path) -> tuple[dict, dict]:
        if path not in self._parsed:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions: dict[str, list[ast.AST]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    functions.setdefault(node.name, []).append(node)
            imports = {
                alias.asname or alias.name: (self._absolute(path, node), alias.name)
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            self._parsed[path] = (functions, imports)
        return self._parsed[path]

    def resolve(self, module: str, name: str, depth: int = 0):
        path = self.module_path(module)
        if path is None or depth > 8:
            return None
        functions, imports = self.parsed(path)
        if name in functions:
            return path, name
        if name in imports:
            return self.resolve(*imports[name], depth + 1)
        return None

    def builds(self, path: Path, name: str, *, seen=frozenset()) -> bool:
        key = (path, name)
        if key in self._builds:
            return self._builds[key]
        if key in seen:
            return False
        functions, imports = self.parsed(path)
        result = False
        for node in functions.get(name, ()):
            for _, called in self.calls(node):
                if called in _FRAME_PASS_CALLS:
                    result = True
                elif called in functions:
                    result = self.builds(path, called, seen=seen | {key})
                elif called in imports:
                    target = self.resolve(*imports[called])
                    result = target is not None and self.builds(
                        *target, seen=seen | {key}
                    )
                if result:
                    break
            if result:
                break
        # A False found under a cut cycle may be wrong from elsewhere: keep
        # only what holds whatever the path.
        if result or not seen:
            self._builds[key] = result
        return result

    def main_calls(self, path: Path) -> list[tuple[int, str, bool]]:
        """``_main``'s calls in source order, each marked if it builds a pass."""

        functions, imports = self.parsed(path)
        (main,) = functions["_main"]
        marked = []
        for line, name in self.calls(main):
            if name in _FRAME_PASS_CALLS:
                builds = True
            elif name in functions:
                builds = self.builds(path, name)
            elif name in imports:
                target = self.resolve(*imports[name])
                builds = target is not None and self.builds(*target)
            else:
                builds = False
            marked.append((line, name, builds))
        return marked


class TestWhatTheExemptListRestsOn:
    """The release order and engine pins L1's verdicts read (lead-call review).

    Every exempt entry's "Earlier readers" verdict, and the moved inputs'
    fill notes, are static reads of the release tool's stage order and of
    the pinned engine. These fail when either moves, so the trace is re-run.
    """

    @pytest.fixture(scope="class")
    def release_calls(self) -> list[tuple[int, str, bool]]:
        return _SourceCalls().main_calls(_RELEASE_TOOL)

    def test_the_stages_run_in_the_traced_order(self, release_calls) -> None:
        first: dict[str, int] = {}
        for line, name, _ in release_calls:
            first.setdefault(name, line)
        missing = [name for name in _TRACED_STAGE_ORDER if name not in first]
        assert not missing, f"The release no longer calls {missing}: {_RETRACE}."
        lines = {name: first[name] for name in _TRACED_STAGE_ORDER}
        assert list(lines.values()) == sorted(lines.values()), (
            f"The release's stage order moved: {lines}; {_RETRACE}."
        )

    def test_the_frame_engine_passes_are_the_traced_ones(self, release_calls) -> None:
        last = next(
            line for line, name, _ in release_calls if name == _TRACED_FRAME_PASSES[-1]
        )
        passes = [
            name for line, name, builds in release_calls if builds and line <= last
        ]
        assert passes == list(_TRACED_FRAME_PASSES), (
            "The release's engine passes through other health insurance are "
            f"{passes}; {_RETRACE}."
        )

    def test_an_untraced_pass_would_be_seen(self, tmp_path) -> None:
        # The reader, fed a tool as data: a tool-local helper that builds a
        # dataset from the frame, a call it cannot resolve, and a stage
        # imported through us_runtime's package that materializes a pass.
        tool = tmp_path / "tool.py"
        tool.write_text(
            "from microcosm.build.us_runtime import (\n"
            "    with_us_other_health_insurance_inputs,\n"
            ")\n"
            "def _early(frame):\n"
            "    return _dataset_from_frame(frame)\n"
            "def _main(frame):\n"
            "    frame = _early(frame)\n"
            "    frame = with_us_scf_wealth_inputs(frame)\n"
            "    return with_us_other_health_insurance_inputs(frame)\n",
            encoding="utf-8",
        )
        marked = [(name, builds) for _, name, builds in _SourceCalls().main_calls(tool)]
        assert marked == [
            ("_early", True),
            ("with_us_scf_wealth_inputs", False),
            ("with_us_other_health_insurance_inputs", True),
        ]

    def test_the_engine_pins_are_the_ones_l1_traced(self) -> None:
        engine = json.loads(ENGINE_ABI_LOCK.read_text(encoding="utf-8"))["engine"]
        lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
        build = next(p for p in lock["package"] if p["name"] == "microcosm-build")
        core = [
            requirement["specifier"]
            for requirement in build["metadata"]["requires-dist"]
            if requirement["name"] == "policyengine-core"
            and requirement.get("marker") == "extra == 'us'"
        ]
        locked_us = {
            p["version"] for p in lock["package"] if p["name"] == "policyengine-us"
        }
        traced_us = _L1_TRACED_ENGINE["policyengine-us"]
        pins = {
            "policyengine-us": (engine["package"], engine["version"], locked_us),
            "policyengine-core": core,
        }
        assert pins == {
            "policyengine-us": ("policyengine-us", traced_us, {traced_us}),
            "policyengine-core": [f"=={_L1_TRACED_ENGINE['policyengine-core']}"],
        }, f"The engine pins moved: {pins}; {_RETRACE}."
        for package, version in _L1_TRACED_ENGINE.items():
            assert f"{package} {version}" in contract._LOADED_NOT_COMPUTED


# X1: one or more corruptions per clause. Each takes the population and its
# valid assembled frame and returns the corrupted frame and any argument
# overrides. All but a few change one cell; the exceptions add or drop a
# column or the record, or change a household's key on every row, where no
# single cell can break the clause (R1's assembled form, R2's digest and
# orphaned clone key, R3, R9, R11, T1's presence, R4's presence, R5's age,
# R6's moved input, R7's presence, R8's twice-applied and moved donors,
# R10's column beside a CPS null).


def _r1_not_assembled(p, f):
    return joined_frame(p, metadata=f.metadata), {}


def _r1_mixed(p, f):
    return _without_column(f, "tax_unit", "tax_unit_spine_source_id"), {}


def _r1_null_source(p, f):
    return _cell(f, "person", "person_source_id", None, origin="cps", clone=1), {}


def _r1_null_clone(p, f):
    return _cell(
        f, "household", "household_support_clone_index", None, origin="acs"
    ), {}


def _r1_null_spine_source(p, f):
    return _cell(f, "spm_unit", "spm_unit_spine_source_id", None, origin="cps"), {}


def _r1_unknown_channel(p, f):
    return _cell(f, "family", "family_support_channel", "acs", origin="acs"), {}


def _r1_record_missing(p, f):
    return _rebuild(f, _tables(f), metadata={}), {}


def _r1_record_two_channels(p, f):
    record = clone_zero_assembly_record(f)[SPINE_ASSEMBLY_MANIFEST_KEY]
    counts = {
        entity: {
            channel: count
            for channel, count in by_channel.items()
            if channel != PUF_TAX_DETAIL_SUPPORT_CHANNEL
        }
        for entity, by_channel in record["native_row_counts"].items()
    }
    metadata = {
        SPINE_ASSEMBLY_MANIFEST_KEY: {
            **record,
            "channels": (BASE_ASEC_SUPPORT_CHANNEL, ACS_2024_1YR_SPINE),
            "native_row_counts": counts,
        }
    }
    return _rebuild(f, _tables(f), metadata=metadata), {}


def _record_with_count(f, value_of):
    """``f`` whose record's ACS household count is ``value_of(count)``."""

    record = clone_zero_assembly_record(f)[SPINE_ASSEMBLY_MANIFEST_KEY]
    counts = {
        entity: dict(by_channel)
        for entity, by_channel in record["native_row_counts"].items()
    }
    counts["household"][ACS_2024_1YR_SPINE] = value_of(
        counts["household"][ACS_2024_1YR_SPINE]
    )
    metadata = {SPINE_ASSEMBLY_MANIFEST_KEY: {**record, "native_row_counts": counts}}
    return _rebuild(f, _tables(f), metadata=metadata)


def _r1_record_fractional_count(p, f):
    # Review nit 1: the validator reads a count through int(), so N + 0.5
    # passed for N live rows.
    return _record_with_count(f, lambda count: count + 0.5), {}


def _r1_cross_grain(p, f):
    # A clone-1 person moves into its clone-0 household (the clone-1
    # household keeps another member): only the validator's cross-grain
    # check reads a person's channel against its household's.
    tables = _tables(f)
    person = tables["person"]
    where = _household_sizes(person, "person_household_id") > 1
    row = _row(person, "person", "cps", clone=1, where=where)
    household = int(person["person_household_id"].iloc[row])
    _set(person, row, "person_household_id", household - p.clone_multiplier)
    return _rebuild(f, tables), {}


def _p5_cps_clone_three(p, f):
    return _cell(
        f, "household", "household_support_clone_index", 3, origin="cps", clone=1
    ), {}


def _p5_cps_asec_on_a_clone(p, f):
    return _cell(
        f, "tax_unit", "tax_unit_support_channel", "asec", origin="cps", clone=1
    ), {}


def _p5_cps_spine_source_moves(p, f):
    table = f.table("spm_unit")
    value = int(table["spm_unit_spine_source_id"].iloc[_row(table, "spm_unit", "cps")])
    return _cell(f, "spm_unit", "spm_unit_spine_source_id", value + 1, origin="cps"), {}


def _p5_cps_wrong_spine_tag(p, f):
    return _cell(f, "family", "family_spine", ACS_2024_1YR_SPINE, origin="cps"), {}


def _p5_cps_non_integer_source(p, f):
    return _cell(f, "person", "person_source_id", 1.5, origin="cps", clone=0), {}


def _p5_acs_clone_index(p, f):
    return _cell(
        f, "marital_unit", "marital_unit_support_clone_index", 1, origin="acs"
    ), {}


def _p5_acs_source_not_pooled(p, f):
    table = f.table("person")
    value = int(table["person_source_id"].iloc[_row(table, "person", "acs")])
    return _cell(f, "person", "person_source_id", value + 10**6, origin="acs"), {}


def _p5_acs_offset_varies(p, f):
    table = f.table("family")
    value = int(table["family_spine_source_id"].iloc[_row(table, "family", "acs")])
    return _cell(f, "family", "family_spine_source_id", value + 1000, origin="acs"), {}


def _p5_acs_wrong_spine_tag(p, f):
    return _cell(f, "household", "household_spine", ASEC_PUF_SPINE, origin="acs"), {}


def _q3_null_spine_tag(p, f):
    return _cell(f, "tax_unit", "tax_unit_spine", None, origin="acs"), {}


def _p6b_acs_person_in_a_cps_group(p, f):
    tables = _tables(f)
    person = tables["person"]
    where = _household_sizes(person, "person_spm_unit_id") > 1
    row = _row(person, "person", "acs", where=where)
    spm_unit = tables["spm_unit"]
    cps_unit = int(spm_unit["spm_unit_id"].iloc[_row(spm_unit, "spm_unit", "cps")])
    _set(person, row, "person_spm_unit_id", cps_unit)
    return _rebuild(f, tables), {}


def _p6_tail_copy_relabelled(p, f):
    return _cell(
        f, "household", "household_support_clone_index", 1, origin="cps", clone=2
    ), {}


def _acs_person_in_a_household_of_two(f) -> tuple[dict, int, int]:
    tables = _tables(f)
    person = tables["person"]
    where = _household_sizes(person, "person_household_id") > 1
    row = _row(person, "person", "acs", where=where)
    household = person["person_household_id"].iloc[row]
    other = int(
        np.flatnonzero(
            (person["person_household_id"] == household).to_numpy()
            & (np.arange(len(person)) != row)
        )[0]
    )
    return tables, row, other


def _p7_sporder_repeats(p, f):
    tables, row, other = _acs_person_in_a_household_of_two(f)
    person = tables["person"]
    _set(person, row, "SPORDER", person["SPORDER"].iloc[other])
    return _rebuild(f, tables), {}


def _p7_source_person_id_moves(p, f):
    return _cell(f, "person", "source_person_id", "99", origin="acs"), {}


def _p7_serialno_null(p, f):
    return _cell(f, "household", "SERIALNO", None, origin="acs"), {}


def _p7_staging_lacks_a_person(p, f):
    lineage = staging_lineage(p)
    first = min(lineage)
    return f, {
        "staging_lineage": {key: pair for key, pair in lineage.items() if key != first}
    }


def _p7_serialno_swapped(p, f):
    # Review blocking 2 on the frame R1 reads: see
    # _serialno_swaps_between_households.
    tables = _tables(f)
    household = tables["household"]
    _swap_serialno(household, *_two_person_acs_households(tables))
    return _rebuild(f, tables), {"staging_lineage": staging_lineage(p)}


def _p7_source_year_pd_na(p, f):
    # Review should-fix 2 on the frame R1 reads: an Int64 pd.NA on one ACS
    # person, which R10 would otherwise let a declaration cover.
    tables = _tables(f)
    person = tables["person"]
    values = person["source_year"].astype("Int64")
    values.iloc[_row(person, "person", "acs")] = pd.NA
    person["source_year"] = values
    return _rebuild(f, tables), {}


def _source_year_as_arrow_times(f, arrow_type) -> Frame:
    """Review r3, should-fix 1, on the frame R1 reads: every ``source_year``
    as an Arrow dictionary of times, each holding the year as its count."""

    tables = _tables(f)
    person = tables["person"]
    person["source_year"] = _arrow_dictionary(person["source_year"], arrow_type)
    return _rebuild(f, tables)


def _p7_source_year_arrow_timestamps(p, f):
    return _source_year_as_arrow_times(f, pa.timestamp("ns")), {}


def _p7_source_year_arrow_durations(p, f):
    return _source_year_as_arrow_times(f, pa.duration("ns")), {}


def _p6c_tax_unit_spans_households(p, f):
    # Review blocking 3 on the frame R1 reads: see
    # _acs_tax_unit_spans_households.
    tables = _tables(f)
    person = tables["person"]
    acs = _origin_mask(person, "person", "acs")
    row = int(
        np.flatnonzero(acs & (_household_sizes(person, "person_tax_unit_id") > 1))[0]
    )
    household = person["person_household_id"].iloc[row]
    elsewhere = acs & person["person_household_id"].ne(household).to_numpy()
    _set(
        person,
        row,
        "person_tax_unit_id",
        person["person_tax_unit_id"].iloc[int(np.flatnonzero(elsewhere)[0])],
    )
    return _rebuild(f, tables), {}


def _p6c_acs_household_holds_two_spm_units(p, f):
    # Review blocking 3's partition on the frame R1 reads, with the clone-0
    # record recounted so the validator passes the extra unit.
    tables = _tables(f)
    person, spm_unit = tables["person"], tables["spm_unit"]
    acs = _origin_mask(person, "person", "acs")
    row = int(
        np.flatnonzero(acs & (_household_sizes(person, "person_household_id") > 1))[0]
    )
    extra = spm_unit.iloc[[-1]].copy()
    assert extra["spm_unit_support_channel"].iloc[0] == ACS_SUPPORT_CHANNEL
    for column in ("spm_unit_id", "spm_unit_source_id", "spm_unit_spine_source_id"):
        extra[column] += 1
    tables["spm_unit"] = pd.concat([spm_unit, extra], ignore_index=True)
    _set(person, row, "person_spm_unit_id", int(extra["spm_unit_id"].iloc[0]))
    frame = _rebuild(f, tables)
    return (
        _rebuild(frame, _tables(frame), metadata=clone_zero_assembly_record(frame)),
        {},
    )


def _p6c_acs_spm_units_merge(p, f):
    # Review r2, nit 3, on the frame R1 reads: see _acs_spm_units_merge, with
    # the clone-0 record recounted so the validator passes the lost unit.
    tables = _tables(f)
    person = tables["person"]
    tables["spm_unit"] = _merge_acs_spm_units(
        person, tables["spm_unit"], _origin_mask(person, "person", "acs")
    )
    frame = _rebuild(f, tables)
    return (
        _rebuild(frame, _tables(frame), metadata=clone_zero_assembly_record(frame)),
        {},
    )


def _r11_location_clone_column(p, f):
    return _with_column(f, "household", "household_location_clone_index", 0), {}


def _r2_peridnum_disagrees(p, f):
    tables = _tables(f)
    person = tables["person"]
    where = _household_sizes(person, "person_household_id") > 1
    row = _row(person, "person", "cps", clone=0, where=where)
    value = person["PERIDNUM"].iloc[row]
    _set(person, row, "PERIDNUM", ("1" if value[0] != "1" else "2") + value[1:])
    return _rebuild(f, tables), {}


def _r2_two_native_rows_share_a_key(p, f):
    # A one-person native household takes another native household's
    # H_IDNUM (its own clone keeps the old one).
    tables = _tables(f)
    person = tables["person"]
    sizes = _household_sizes(person, "person_household_id")
    row = _row(person, "person", "cps", clone=0, where=sizes == 1)
    other = _row(
        person,
        "person",
        "cps",
        clone=0,
        where=person["person_household_id"].to_numpy()
        != person["person_household_id"].iloc[row],
    )
    _set(person, row, "PERIDNUM", person["PERIDNUM"].iloc[other][:20] + "01")
    return _rebuild(f, tables), {}


def _r2_cps_key_substituted(p, f):
    # A native household and its clones take a new, unused H_IDNUM: the
    # counts by income year hold, only the digest moves.
    tables = _tables(f)
    person = tables["person"]
    old = person["PERIDNUM"].iloc[_row(person, "person", "cps", clone=0)][:20]
    new = ("1" if old[0] != "1" else "2") + old[1:]
    assert not person["PERIDNUM"].str.startswith(new, na=False).any()
    hit = person["PERIDNUM"].str.startswith(old, na=False).to_numpy(dtype=bool)
    person.loc[hit, "PERIDNUM"] = [new + v[20:] for v in person.loc[hit, "PERIDNUM"]]
    return _rebuild(f, tables), {}


def _r2_orphaned_clone_key(p, f):
    # A clone-1 household's persons take a key no clone-0 household has.
    tables = _tables(f)
    person = tables["person"]
    row = _row(person, "person", "cps", clone=1)
    hit = (
        person["person_household_id"] == person["person_household_id"].iloc[row]
    ).to_numpy()
    old = person["PERIDNUM"].iloc[row][:20]
    new = ("1" if old[0] != "1" else "2") + old[1:]
    person.loc[hit, "PERIDNUM"] = [new + v[20:] for v in person.loc[hit, "PERIDNUM"]]
    return _rebuild(f, tables), {}


def _r2_acs_household_not_selected(p, f):
    return _cell(f, "household", "SERIALNO", "2024HU9999999", origin="acs"), {}


def _r2_manifest_counts_differ(p, f):
    manifest = dataclasses.replace(selection_manifest(p), n_cps=len(p.cps_keys) - 1)
    return f, {"manifest": manifest}


def _r12_null_district(p, f):
    return _cell(f, "household", "congressional_district_geoid", None, origin="cps"), {}


def _r12_wrong_state_prefix(p, f):
    table = f.table("household")
    state = int(table["state_fips"].iloc[_row(table, "household", "acs")])
    return _cell(
        f,
        "household",
        "congressional_district_geoid",
        (state + 1) * 100 + 1,
        origin="acs",
    ), {}


def _r12_non_positive_district(p, f):
    return _cell(f, "household", "congressional_district_geoid", -601, origin="cps"), {}


def _t1_typehugq_out_of_domain(p, f):
    return _cell(f, "household", "TYPEHUGQ", 4, origin="acs"), {}


def _t1_typehugq_null(p, f):
    return _cell(f, "household", "TYPEHUGQ", None, origin="acs"), {}


def _t1_typehugq_missing(p, f):
    return _without_column(f, "household", "TYPEHUGQ"), {}


def _t1_typehugq_boolean(p, f):
    return _cell(f, "household", "TYPEHUGQ", True, origin="acs"), {}


def _t1_typehugq_timedelta(p, f):
    # Review r3, should-fix 2: numpy registers timedelta64 as a number, and
    # np.timedelta64(1, "ns") in (1, 2, 3) holds, so T1 passed this code.
    table = f.table("household")
    code = int(table["TYPEHUGQ"].iloc[_row(table, "household", "acs")])
    return _cell(
        f, "household", "TYPEHUGQ", np.timedelta64(code, "ns"), origin="acs"
    ), {}


def _r3_computed_column(p, f):
    return _with_column(f, "spm_unit", "snap", 0.0), {}


def _r3_compat_aggregate(p, f):
    return _with_column(f, "person", "social_security", 0.0), {}


def _r9_unregistered_column(p, f):
    return _with_column(f, "person", "would_claim_wic", True), {}


def _r4_null_premium(p, f):
    return _cell(
        f, "person", "employer_sponsored_insurance_premiums", None, origin="acs"
    ), {}


def _r4_premium_missing(p, f):
    return _without_column(f, "person", "employer_sponsored_insurance_premiums"), {}


def _r5_minor_head_loses_role(p, f):
    person = f.table("person")
    where = (person["age"] == 17).to_numpy() & person["is_household_head"].to_numpy(
        dtype=bool
    )
    return _cell(f, "person", SPM_ROLE, False, origin="acs", where=where), {}


def _r5_no_age(p, f):
    return _without_column(f, "person", "age"), {}


def _r6_null_tip_income(p, f):
    return _cell(f, "person", "tip_income", None, origin="acs"), {}


def _r6_null_is_female_on_a_tail_copy(p, f):
    return _cell(f, "person", "is_female", None, origin="cps", clone=2), {}


def _r6_null_moved_premiums_on_acs(p, f):
    # d2283 lead call L1 moved health_insurance_premiums off the exempt list
    # (the ACA, Medicaid and SNAP-state passes read it before its producer).
    filled = _with_column(f, "person", "health_insurance_premiums", 1.0)
    return _cell(filled, "person", "health_insurance_premiums", None, origin="acs"), {}


def _r7_null_tail_column(p, f):
    return _cell(
        f,
        "tax_unit",
        "puf_capital_gains_tail_donor_agi_band_index",
        None,
        origin="acs",
    ), {}


def _r7_tail_column_missing(p, f):
    return _without_column(f, "tax_unit", "puf_capital_gains_tail_transfer_weight"), {}


def _r8_donor_moves(p, f):
    applied = f.table("tax_unit")[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(
        dtype=bool
    )
    return _cell(
        f,
        "tax_unit",
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
        1,
        origin="cps",
        where=applied,
    ), {}


def _r8_unapplied_unit_applies(p, f):
    applied = f.table("tax_unit")[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(
        dtype=bool
    )
    return _cell(
        f,
        "tax_unit",
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        True,
        origin="cps",
        where=~applied,
    ), {}


def _r8_parent_donor_unapplied(p, f):
    applied = f.table("tax_unit")[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(
        dtype=bool
    )
    return _cell(
        f,
        "tax_unit",
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        False,
        origin="cps",
        where=applied,
    ), {}


def _r8_donor_applied_twice(p, f):
    # A second CPS tax unit applies a parent donor: the id set holds.
    tables = _tables(f)
    tax_unit = tables["tax_unit"]
    applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(dtype=bool)
    donor = tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN].iloc[
        int(np.flatnonzero(applied)[0])
    ]
    row = _row(tax_unit, "tax_unit", "cps", where=~applied)
    _set(tax_unit, row, PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN, True)
    _set(tax_unit, row, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN, donor)
    return _rebuild(f, tables), {}


def _r8_applied_not_a_flag(p, f):
    return _cell(
        f, "tax_unit", PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN, 2, origin="acs"
    ), {}


def _r8_applied_flag_timedelta(p, f):
    # Review r3, should-fix 2: an applied CPS unit's flag held as
    # np.timedelta64(1, "ns"), which numbers.Real accepts and which equals 1,
    # so R8 read it as True and passed the frame.
    applied = f.table("tax_unit")[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(
        dtype=bool
    )
    return _cell(
        f,
        "tax_unit",
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        np.timedelta64(1, "ns"),
        origin="cps",
        where=applied,
    ), {}


def _r8_donor_moved_to_an_acs_unit(p, f):
    # d2283 lead call L4: a parent donor moves from its CPS tax unit to an
    # ACS one, so the id set and the one-to-one rule both still hold.
    tables = _tables(f)
    tax_unit = tables["tax_unit"]
    applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy(dtype=bool)
    cps_row = int(np.flatnonzero(applied)[0])
    donor = tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN].iloc[cps_row]
    _set(tax_unit, cps_row, PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN, False)
    _set(tax_unit, cps_row, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN, -1)
    acs_row = _row(tax_unit, "tax_unit", "acs")
    _set(tax_unit, acs_row, PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN, True)
    _set(tax_unit, acs_row, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN, donor)
    return _rebuild(f, tables), {}


def _r10_register_column_null_on_acs(p, f):
    return _cell(f, "person", "source_row_id", None, origin="acs"), {}


def _r10_register_column_null_beside_a_cps_null(p, f):
    # d2283 lead call L2: R10 shares absences' gap predicate, so one null on a
    # CPS row no longer switches it off for a column ACS rows lack.
    column = "cps_employment_income_last_year"
    frame = _acs_null_on_cps(f, "person", column, 1.0)
    frame = _cell(frame, "person", column, None, origin="cps")
    register = {**US_STORED_NON_VARIABLE_COLUMNS, column: "test auxiliary column"}
    return frame, {"register": register}


_FRAME_REFUSALS = [
    # R1: incomplete assembled provenance (form, completeness, record).
    (_r1_not_assembled, "Incomplete assembled provenance: the frame is not assembled"),
    (_r1_mixed, "Mixed assembled and unassembled provenance: ['tax_unit'] lack"),
    (
        _r1_null_source,
        "Incomplete assembled provenance: assembled support metadata requires "
        "complete non-null 'person_source_id'",
    ),
    (
        _r1_null_clone,
        "Incomplete assembled provenance: 'household_support_clone_index' has 1 null",
    ),
    (
        _r1_null_spine_source,
        "Incomplete assembled provenance: 'spm_unit_spine_source_id' has 1 null",
    ),
    (
        _r1_unknown_channel,
        "Incomplete assembled provenance: rows must carry one of the assembled "
        "channels",
    ),
    (
        _r1_record_missing,
        "Incomplete assembled provenance: one-population provenance: "
        "receipt-validated support provenance has no assembly manifest",
    ),
    (
        _r1_record_two_channels,
        "Incomplete assembled provenance: the spine assembly record must declare",
    ),
    (
        _r1_record_fractional_count,
        "Incomplete assembled provenance: the spine assembly record's native row "
        "counts must be nonnegative integers; ['household.acs_2024_1yr=",
    ),
    (_r1_cross_grain, "cross-grain provenance disagrees with the assembly manifest"),
    # P5: each origin's value rules on the assembled frame.
    (_p5_cps_clone_three, "CPS clone indices must be 0, 1 or 2"),
    (
        _p5_cps_asec_on_a_clone,
        "CPS rows must carry channel 'asec' exactly on clone index 0; 1 'tax_unit'",
    ),
    (
        _p5_cps_spine_source_moves,
        "CPS rows' spine source id must equal their source id",
    ),
    (_p5_cps_wrong_spine_tag, "CPS rows carry a spine tag other than 'asec_puf'"),
    (
        _p5_cps_non_integer_source,
        "CPS 'person_source_id' must hold source ids: 1 value(s) are not integers",
    ),
    (_p5_acs_clone_index, "ACS rows must carry clone index 0"),
    (_p5_acs_source_not_pooled, "ACS rows' source id must equal their pooled id"),
    (_p5_acs_offset_varies, "ACS 'family' ids must sit one constant offset"),
    (
        _p5_acs_wrong_spine_tag,
        "ACS rows carry a spine tag other than 'acs_2024_1yr'",
    ),
    (_q3_null_spine_tag, "A null spine tag: 1 ACS 'tax_unit' row(s)"),
    # P6b and P6.
    (
        _p6b_acs_person_in_a_cps_group,
        "Memberships do not resolve: 1 ACS person(s)' 'person_spm_unit_id' match "
        "no ACS spm_unit row",
    ),
    (
        _p6_tail_copy_relabelled,
        "A (source id, rank) pair repeats in 'household'",
    ),
    # P7: SERIALNO/SPORDER lineage.
    (
        _p7_sporder_repeats,
        "ACS lineage does not survive: (SERIALNO, SPORDER) repeats on 2 ACS",
    ),
    (
        _p7_source_person_id_moves,
        "ACS lineage does not survive: source_person_id differs from str(SPORDER)",
    ),
    (
        _p7_serialno_null,
        "ACS lineage does not survive: every ACS household needs a non-empty "
        "string SERIALNO",
    ),
    (
        _p7_staging_lacks_a_person,
        "keyed by native (household id, person id); 0 missing (such as []), 1 extra",
    ),
    (
        _p7_serialno_swapped,
        "keyed by native (household id, person id); 0 missing (such as []), 0 "
        "extra (such as []), 4 moved",
    ),
    (
        _p7_source_year_pd_na,
        "ACS lineage does not survive: every ACS person needs an integral "
        "source_year: 1 value(s) are null",
    ),
    (
        _p7_source_year_arrow_timestamps,
        "ACS lineage does not survive: every ACS person needs an integral "
        "source_year: it holds dictionary<values=timestamp[ns], indices=int32, "
        "ordered=0>[pyarrow] values, not integers",
    ),
    (
        _p7_source_year_arrow_durations,
        "ACS lineage does not survive: every ACS person needs an integral "
        "source_year: it holds dictionary<values=duration[ns], indices=int32, "
        "ordered=0>[pyarrow] values, not integers",
    ),
    # P6c, after the assembly validator: groups nest in households, and ACS
    # SPM units are the households.
    (
        _p6c_tax_unit_spans_households,
        "Group units do not nest in households: 1 ACS tax_unit unit(s) hold "
        "persons of more than one household",
    ),
    (
        _p6c_acs_household_holds_two_spm_units,
        "ACS SPM units must be the households (staging's one-SPM-unit-per-"
        "household partition): 1 ACS household(s) hold more than one SPM unit",
    ),
    (
        _p6c_acs_spm_units_merge,
        "Group units do not nest in households: 1 ACS spm_unit unit(s) hold "
        "persons of more than one household",
    ),
    # R11.
    (_r11_location_clone_column, "A household with more than one location"),
    # R2.
    (
        _r2_peridnum_disagrees,
        "A CPS key without exactly one native row: CPS unit ids: persons of one "
        "household disagree",
    ),
    (
        _r2_two_native_rows_share_a_key,
        "A CPS key without exactly one native row: 1 key(s) have more than one "
        "clone-0 household",
    ),
    (
        _r2_orphaned_clone_key,
        "A CPS key without exactly one native row: 1 clone household key(s) have "
        "no clone-0 household",
    ),
    (
        _r2_cps_key_substituted,
        "CPS keys do not match the selection manifest: the frame's clone-0 "
        "households give (count, by income year, digest) (24, ((2023, 12), "
        "(2024, 12)), '",
    ),
    (
        _r2_acs_household_not_selected,
        "ACS keys do not match the selection manifest",
    ),
    (_r2_manifest_counts_differ, "CPS keys do not match the selection manifest"),
    # R12.
    (
        _r12_null_district,
        "A household without a congressional district: "
        "'congressional_district_geoid' must be an integral geoid on every "
        "household: 1 value(s) are null",
    ),
    (
        _r12_wrong_state_prefix,
        "A household without a congressional district: 1 household(s) hold a "
        "'congressional_district_geoid' whose state prefix",
    ),
    (
        _r12_non_positive_district,
        "A household without a congressional district: 1 household(s) hold a "
        "non-positive",
    ),
    # T1.
    (
        _t1_typehugq_out_of_domain,
        "TYPEHUGQ must be 1, 2 or 3 on every ACS household",
    ),
    (_t1_typehugq_null, "1 ACS household(s) hold [nan]"),
    (
        _t1_typehugq_missing,
        "on every ACS household (#1610's household materializer "
        "reads it to leave group quarters out); the household table lacks it.",
    ),
    (_t1_typehugq_boolean, "1 ACS household(s) hold [True]"),
    (_t1_typehugq_timedelta, "1 ACS household(s) hold [np.timedelta64("),
    # R3.
    (_r3_computed_column, "A column the engine computes at 2024 is stored: ['snap']"),
    (
        _r3_compat_aggregate,
        "A column the engine computes at 2024 is stored: ['social_security']",
    ),
    # R9.
    (
        _r9_unregistered_column,
        "A stored lower-case column is neither an engine variable nor registered: "
        "['would_claim_wic']",
    ),
    # R4.
    (_r4_null_premium, "A null ESI premium: d2283 one-population contract: ESI"),
    (_r4_premium_missing, "A null ESI premium: the person table lacks"),
    # R5.
    (_r5_minor_head_loses_role, "An SPM unit has no classified adult: 1 of"),
    (
        _r5_no_age,
        "An SPM unit has no classified adult: the composition check cannot run",
    ),
    # R6.
    (
        _r6_null_tip_income,
        "A null in an engine input at 2024: {'person.tip_income': 1}",
    ),
    (
        _r6_null_is_female_on_a_tail_copy,
        "A null in an engine input at 2024: {'person.is_female': 1}",
    ),
    (
        _r6_null_moved_premiums_on_acs,
        "A null in an engine input at 2024: {'person.health_insurance_premiums': 1}",
    ),
    # R7.
    (
        _r7_null_tail_column,
        "A null PUF tail column on a tax unit: "
        "{'tax_unit.puf_capital_gains_tail_donor_agi_band_index': 1}",
    ),
    (
        _r7_tail_column_missing,
        "A null PUF tail column on a tax unit: the tax-unit table lacks",
    ),
    # R8.
    (
        _r8_donor_moves,
        "The tail donor set differs from the parent's: 1 parent donor(s) missing",
    ),
    (
        _r8_unapplied_unit_applies,
        "The tail donor set differs from the parent's: 0 parent donor(s) missing "
        "(such as []) and 1 donor(s) the parent never applied (such as [-1])",
    ),
    (
        _r8_parent_donor_unapplied,
        "The tail donor set differs from the parent's: 1 parent donor(s) missing "
        "(such as [7000000]) and 0 donor(s) the parent never applied",
    ),
    (
        _r8_donor_applied_twice,
        "The tail donor set differs from the parent's: donor id(s) [7000000] are "
        "applied on more than one tax unit",
    ),
    (
        _r8_applied_not_a_flag,
        "The tail donor set differs from the parent's: "
        "'puf_capital_gains_tail_transfer_applied' holds non-flag values [2]",
    ),
    (
        _r8_applied_flag_timedelta,
        "The tail donor set differs from the parent's: "
        "'puf_capital_gains_tail_transfer_applied' holds non-flag values "
        "[np.timedelta64(1,'ns')]",
    ),
    (
        _r8_donor_moved_to_an_acs_unit,
        "The tail donor set differs from the parent's: 1 ACS tax unit(s) apply a "
        "tail donor",
    ),
    # R10.
    (
        _r10_register_column_null_on_acs,
        "A model-named column held on some CPS row is null on ACS rows without a "
        "declared absence: {'person.source_row_id': '1 of",
    ),
    (
        _r10_register_column_null_beside_a_cps_null,
        "A model-named column held on some CPS row is null on ACS rows without a "
        "declared absence: {'person.cps_employment_income_last_year': '",
    ),
]


@pytest.mark.parametrize(
    ("corrupt", "message"),
    _FRAME_REFUSALS,
    ids=[corrupt.__name__.lstrip("_") for corrupt, _ in _FRAME_REFUSALS],
)
def test_each_clause_refuses_with_its_own_message(
    population, assembled, corrupt, message
) -> None:
    frame, overrides = corrupt(population, assembled)
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        _require(population, frame, **overrides)


def test_frame_refusal_messages_are_distinct() -> None:
    messages = [message for _, message in _FRAME_REFUSALS]
    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize("group", ["tax_unit", "spm_unit", "family", "marital_unit"])
def test_a_cps_group_unit_across_households_refuses_at_r1(
    population, assembled, group
) -> None:
    # PR #1685 review r2, nit 3: R1 runs P6c on both origins. Its other P6c
    # fixtures are all ACS, so a call on the ACS rows alone would pass a CPS
    # crossing made after assembly. As _cps_unit_spans_households, on the
    # frame R1 reads.
    tables = _tables(assembled)
    person = tables["person"]
    column = f"person_{group}_id"
    native = _origin_mask(person, "person", "cps") & (
        person["person_support_clone_index"].to_numpy() == 0
    )
    row = int(np.flatnonzero(native & (_household_sizes(person, column) > 1))[0])
    household = person["person_household_id"].iloc[row]
    elsewhere = native & person["person_household_id"].ne(household).to_numpy()
    _set(person, row, column, person[column].iloc[int(np.flatnonzero(elsewhere)[0])])
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Group units do not nest in households: 1 CPS {group} unit(s) hold "
            "persons of more than one household"
        ),
    ):
        _require(population, _rebuild(assembled, tables))


@pytest.mark.parametrize("group", _GROUPS)
@pytest.mark.parametrize("origin", ["acs", "cps"])
def test_a_complex_membership_refuses_at_r1(
    population, assembled, origin, group
) -> None:
    # Review r3, nit 2, on the frame R1 reads: as test_a_complex_membership_
    # refuses, one person's link held as a numpy complex, on either origin.
    tables = _tables(assembled)
    person = tables["person"]
    column = US_SCHEMA.membership_column(group)
    _complex_membership(person, _row(person, "person", origin), column)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            f"Memberships do not resolve: {origin.upper()} persons' {column!r} "
            "holds 1 value(s) that are not real numbers or strings, such as "
            "[np.complex128("
        ),
    ):
        _require(population, _rebuild(assembled, tables))


def test_every_clause_has_a_corruption() -> None:
    prefixes = {corrupt.__name__.split("_")[1] for corrupt, _ in _FRAME_REFUSALS}
    assert prefixes >= {
        "p5",
        "p6",
        "p6b",
        "p7",
        "q3",
        *(f"r{number}" for number in range(1, 13)),
        "t1",
    }


@pytest.mark.parametrize(
    ("corruptions", "message"),
    [
        # Within R1, the assembly validator before P6c (nesting): a person
        # moved across households is a cross-grain move first.
        (
            (_p6c_tax_unit_spans_households, _r1_cross_grain),
            "cross-grain provenance disagrees",
        ),
        # R1 (P5) before R2, R6 and R12.
        (
            (_p5_acs_clone_index, _r2_peridnum_disagrees, _r12_null_district),
            "ACS rows must carry clone index 0",
        ),
        # R1 before R11, and R11 before R2 (review nit 2: no case held R11).
        (
            (_r11_location_clone_column, _p5_acs_clone_index),
            "ACS rows must carry clone index 0",
        ),
        (
            (_r2_peridnum_disagrees, _r11_location_clone_column),
            "A household with more than one location",
        ),
        # R2 before R12 and T1.
        (
            (_r12_null_district, _t1_typehugq_null, _r2_peridnum_disagrees),
            "CPS unit ids: persons of one household disagree",
        ),
        # R3 before R9.
        ((_r9_unregistered_column, _r3_computed_column), "computes at 2024"),
        # R4 before R5 and R6.
        ((_r6_null_tip_income, _r5_minor_head_loses_role, _r4_null_premium), "ESI"),
        # R6 and R7 before R8 and R10.
        (
            (_r10_register_column_null_on_acs, _r8_donor_moves, _r7_null_tail_column),
            "A null PUF tail column",
        ),
        (
            (_r10_register_column_null_on_acs, _r6_null_tip_income),
            "A null in an engine input at 2024",
        ),
    ],
)
def test_clauses_run_in_their_fixed_order(
    population, assembled, corruptions, message
) -> None:
    frame = assembled
    for corrupt in corruptions:
        frame, overrides = corrupt(population, frame)
        assert overrides == {}
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        _require(population, frame)


def _acs_serialno_repeats(frame: Frame) -> Frame:
    """The second ACS household takes the first's ``SERIALNO``."""

    tables = _tables(frame)
    household = tables["household"]
    acs = np.flatnonzero(_origin_mask(household, "household", "acs"))
    _set(household, int(acs[1]), "SERIALNO", household["SERIALNO"].iloc[int(acs[0])])
    return _rebuild(frame, tables)


def test_a_real_location_clone_refuses_at_r1_first(population, assembled) -> None:
    # Review nit 2: block_location's own cloning remaps every pooled id and
    # keeps the provenance ids, so P5 (R1) reads the clone before R11 reads
    # the location-clone columns it also adds. The documented precedence.
    frame = clone_frame_for_location(assembled, 2)
    marked = [
        entity
        for entity in US_SCHEMA.entities
        if f"{entity}_location_clone_index" in frame.table(entity)
    ]
    assert marked == list(US_SCHEMA.entities)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape("ACS rows' source id must equal their pooled id"),
    ):
        _require(population, frame)


def test_a_repeated_serialno_refuses_at_r1_before_r2(population, assembled) -> None:
    # Review nit 2: P7 (R1) reads a repeated SERIALNO before R2's
    # repeated-key branch, which stays as a guard; called on its own, R2
    # refuses the same households.
    frame = _acs_serialno_repeats(assembled)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "ACS lineage does not survive: SERIALNO repeats across ACS households"
        ),
    ):
        _require(population, frame)
    tables = _tables(frame)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape("An ACS household with more than one row"),
    ):
        contract._require_selected_keys(
            contract._split_origins(tables), selection_manifest(population)
        )


@pytest.mark.parametrize(
    "value_of",
    [str, float, lambda count: -1],
    ids=["digit string", "integral float", "negative"],
)
def test_r1_refuses_a_record_count_the_validator_would_coerce(
    population, assembled, value_of
) -> None:
    # Review nit 1, the other forms: the validator's int() reads "N" and
    # N.0 as N; a negative count it refuses itself, but R1's check reads
    # every count first, so the message is one.
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "the spine assembly record's native row counts must be nonnegative integers"
        ),
    ):
        _require(population, _record_with_count(assembled, value_of))


# Property: a null in any checked cell refuses. "Checked" means some clause
# reads every value of the column on those rows; an exempt column, or a
# register column on a CPS row (R10 reads it only on ACS rows), is not.
def _checked_cells(population, frame: Frame) -> list[tuple[str, str, str]]:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    inputs = StandInVariableIndex.from_lock().inputs_at_2024(tables)
    cells = [
        (entity, column, "any")
        for entity in US_SCHEMA.entities
        for column in (
            support_source_id_column(entity),
            support_channel_column(entity),
            support_clone_index_column(entity),
            spine_source_id_column(entity),
            f"{entity}_spine",
        )
    ]
    cells += [
        ("household", "congressional_district_geoid", "any"),
        ("household", "state_fips", "any"),
        ("household", "SERIALNO", "acs"),
        ("household", "TYPEHUGQ", "acs"),
        ("person", "SPORDER", "acs"),
        ("person", "source_household_id", "acs"),
        ("person", "source_person_id", "acs"),
        ("person", "source_row_id", "acs"),
        ("person", "source_year", "any"),
        ("person", "PERIDNUM", "cps"),
        *(("tax_unit", column, "any") for column in PUF_TAIL_TAX_UNIT_COLUMNS),
    ]
    cells += [
        (entity, column, "any")
        for entity in US_SCHEMA.entities
        for column in sorted(inputs - set(RELEASE_WRITTEN_INPUTS))
        if column in tables[entity]
    ]
    return sorted(set(cells))


@pytest.fixture(scope="module")
def checked_cells(population, assembled):
    return _checked_cells(population, assembled)


def test_the_checked_cells_cover_the_engine_inputs(checked_cells) -> None:
    columns = {column for _, column, _ in checked_cells}
    assert {"age", SPM_ROLE, "tip_income", "employer_sponsored_insurance_premiums"} <= (
        columns
    )
    assert "takes_up_ssi_if_eligible" not in columns


@settings(max_examples=60, deadline=None, database=None)
@given(data=st.data())
def test_a_null_in_any_checked_cell_refuses(
    population, assembled, checked_cells, data
) -> None:
    entity, column, origin = data.draw(st.sampled_from(checked_cells))
    table = assembled.table(entity)
    if origin == "any":
        rows = np.arange(len(table))
    else:
        rows = np.flatnonzero(_origin_mask(table, entity, origin))
    row = int(data.draw(st.sampled_from(rows.tolist())))
    tables = _tables(assembled)
    _set(tables[entity], row, column, None)
    frame = _rebuild(assembled, tables)
    with pytest.raises(OnePopulationContractError):
        _require(population, frame)


# --------------------------- phase 3: the combined SPM role record (S1, Q6)


def _role_inputs(frame: Frame) -> tuple[dict, dict]:
    return cps_role_record(frame), staging_role_receipt(frame)


def _swap_cps_roles(frame: Frame) -> Frame:
    """Two CPS persons trade roles: every count holds, the digest moves."""

    tables = _tables(frame)
    person = tables["person"]
    cps = _origin_mask(person, "person", "cps")
    role = person[SPM_ROLE].to_numpy(dtype=bool)
    holder = int(np.flatnonzero(cps & role)[0])
    other = int(np.flatnonzero(cps & ~role)[0])
    values = role.copy()
    values[[holder, other]] = values[[other, holder]]
    person[SPM_ROLE] = values
    return _rebuild(frame, tables)


def _swap_acs_roles(frame: Frame) -> Frame:
    """An adult ACS role holder and an adult of its household without it trade.

    PR #1685 review, should-fix 1 (its example: a 40-year-old reference
    person and a 25-year-old adult child): every count holds, so staging's
    receipt still covers the rows and the role gate's counts are unchanged;
    only the persons' ``RELSHIPP`` says which of them holds the role.
    """

    tables = _tables(frame)
    person = tables["person"]
    adult_acs = _origin_mask(person, "person", "acs") & person["age"].ge(18).to_numpy()
    role = person[SPM_ROLE].to_numpy(dtype=bool)
    household = person["person_household_id"].to_numpy()
    holder = next(
        int(row)
        for row in np.flatnonzero(adult_acs & role)
        if (adult_acs & ~role & (household == household[row])).any()
    )
    other = int(np.flatnonzero(adult_acs & ~role & (household == household[holder]))[0])
    values = role.copy()
    values[[holder, other]] = values[[other, holder]]
    person[SPM_ROLE] = values
    return _rebuild(frame, tables)


def _relshipp_as(frame: Frame, form: str) -> Frame:
    """The first ACS person's ``RELSHIPP`` as complex or temporal data.

    Each value keeps its code as the real part (or the count of time units),
    so a conversion to real numbers reads the person's own code and role
    (PR #1685 review r2, should-fix 1).
    """

    tables = _tables(frame)
    person = tables["person"]
    row = _row(person, "person", "acs")
    code = int(person["RELSHIPP"].iloc[row])
    if form == "complex128 column":
        values = person["RELSHIPP"].astype(np.complex128)
        values.iloc[row] = complex(code, 1)
        person["RELSHIPP"] = values
    elif form == "complex in object":
        _set(person, row, "RELSHIPP", complex(code, 1))
    elif form == "timedelta64 column":
        person["RELSHIPP"] = pd.to_timedelta(person["RELSHIPP"].to_numpy(), unit="ns")
    elif form.startswith("Arrow dictionary"):
        # Review r3, should-fix 1: every code as an Arrow dictionary of
        # times (CPS rows null), each holding the code as its count.
        unit = pa.timestamp("ns") if "timestamps" in form else pa.duration("ns")
        person["RELSHIPP"] = _arrow_dictionary(person["RELSHIPP"], unit)
    else:
        _set(person, row, "RELSHIPP", np.timedelta64(code, "ns"))
    return _rebuild(frame, tables)


class TestCombinedSpmRoleRecord:
    @pytest.mark.parametrize("form", _FORMS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_the_record_passes_the_real_role_gate(self, seed, form) -> None:
        frame = assembled_frame(_form(generate_population(seed), form))
        record = combined_spm_role_record(*_role_inputs(frame), frame)
        result = us_spm_independence_role_signal_gate(
            with_spm_role_record(frame, record)
        )
        assert result.passed, result.failures
        # Plain builtins: the record survives canonical JSON as it is.
        assert json.loads(json.dumps(record, allow_nan=False)) == record

    def test_the_gate_keys_recount_the_whole_frame(self, assembled) -> None:
        record = combined_spm_role_record(*_role_inputs(assembled), assembled)
        expected = spm_role_gate_record(assembled)
        assert {key: record[key] for key in expected} == expected
        assert record["role_source"] == "source_column"
        assert record["column"] == SPM_ROLE

    def test_the_digest_is_the_gates_own_rule(self, assembled) -> None:
        person = assembled.table("person")
        record = combined_spm_role_record(*_role_inputs(assembled), assembled)
        assert record["person_role_binding_sha256"] == _role_binding_sha256(person)
        cps = _origin_mask(person, "person", "cps")
        assert record["origins"]["CPS"][
            "person_role_binding_sha256"
        ] == _role_binding_sha256(person.loc[cps])

    def test_counts_by_channel_and_origin(self, assembled) -> None:
        record = combined_spm_role_record(*_role_inputs(assembled), assembled)
        person = assembled.table("person")
        spm_unit = assembled.table("spm_unit")
        for channel in ASSEMBLED_CHANNELS:
            rows = person["person_support_channel"].eq(channel)
            assert record["counts_by_channel"][channel] == {
                "persons": int(rows.sum()),
                "spm_units": int(
                    spm_unit["spm_unit_support_channel"].eq(channel).sum()
                ),
                "true_role_persons": int(person.loc[rows, SPM_ROLE].sum()),
            }
        assert all(
            record["counts_by_channel"][channel]["persons"] > 0
            for channel in ASSEMBLED_CHANNELS
        )
        cps, acs = record["origins"]["CPS"], record["origins"]["ACS"]
        assert cps["channels"] == [
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ]
        assert acs["channels"] == [ACS_2024_1YR_SPINE]
        for key, gate_key in (
            ("persons", "persons_joined"),
            ("spm_units", "native_spm_units"),
            ("true_role_persons", "true_role_persons"),
        ):
            assert cps[key] + acs[key] == record[gate_key]
        assert cps["persons"] == cps_role_record(assembled)["persons_joined"]

    def test_both_inputs_are_cited(self, assembled) -> None:
        cps_record, receipt = _role_inputs(assembled)
        record = combined_spm_role_record(cps_record, receipt, assembled)
        assert record["evidence"] == {
            "cps_record": cps_record,
            "staging_receipt": receipt,
        }

    def test_numpy_scalars_in_the_inputs_become_builtins(self, assembled) -> None:
        cps_record, receipt = _role_inputs(assembled)
        cps_record = {
            **cps_record,
            "persons_joined": np.int64(cps_record["persons_joined"]),
        }
        receipt = {**receipt, "weighted_role_share": np.float64(0.6)}
        record = combined_spm_role_record(cps_record, receipt, assembled)
        assert type(record["evidence"]["cps_record"]["persons_joined"]) is int
        assert type(record["evidence"]["staging_receipt"]["weighted_role_share"]) is (
            float
        )

    def test_the_inputs_are_not_mutated(self, assembled) -> None:
        cps_record, receipt = _role_inputs(assembled)
        before = (json.dumps(cps_record), json.dumps(receipt))
        person = assembled.table("person").copy()
        combined_spm_role_record(cps_record, receipt, assembled)
        assert (json.dumps(cps_record), json.dumps(receipt)) == before
        pd.testing.assert_frame_equal(assembled.table("person"), person)

    def test_a_frame_with_no_acs_row(self, population) -> None:
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        record = combined_spm_role_record(*_role_inputs(frame), frame)
        assert record["origins"]["ACS"]["persons"] == 0
        assert record["persons_joined"] == record["origins"]["CPS"]["persons"]

    def test_the_receipt_rule_and_partition_are_staging_constants(self) -> None:
        # Read from acs_inputs' source: importing it loads the ACS transfer.
        source = _U / "acs_inputs.py"
        for constant, key in (
            ("ACS_SPM_ROLE_RULE", "rule"),
            ("ACS_SPM_ROLE_PARTITION", "partition"),
        ):
            value = ast.literal_eval(_module_assignment(source, constant))
            assert contract._ACS_ROLE_RECEIPT_IDENTITY[key] == value, constant

    def test_the_relshipp_codes_are_staging_constants(self) -> None:
        # The codes the row-level ACS role check reads (review should-fix 1)
        # are copies of the rule's own constants (d2283 lead call L8).
        from microcosm.build.us_runtime import acs_inputs

        assert contract._ACS_ROLE_RELSHIPP == {
            acs_inputs._ACS_REFERENCE_PERSON,
            *acs_inputs._ACS_SPOUSE_OF_REFERENCE,
            *acs_inputs._ACS_GROUP_QUARTERS_PERSON,
        }
        assert contract._ACS_RELSHIPP_CODES == acs_inputs._ACS_RELSHIPP_CODES

    def test_a_swap_of_acs_roles_keeps_every_count(self, assembled) -> None:
        # Why the receipt and the counts cannot catch should-fix 1's swap:
        # the gate's counts and the receipt's coverage are unchanged.
        swapped = _swap_acs_roles(assembled)
        before, after = (spm_role_gate_record(frame) for frame in (assembled, swapped))
        assert {
            key: before[key] for key in before if key != "person_role_binding_sha256"
        } == {key: after[key] for key in after if key != "person_role_binding_sha256"}
        assert staging_role_receipt(swapped) == staging_role_receipt(assembled)

    def test_the_receipt_keys_are_staging_receipts(self) -> None:
        # The dict acs_inputs.with_acs_spm_independence_role assigns to
        # ``provenance`` (U/acs_inputs.py:478-511).
        from microcosm.build.us_runtime import acs_inputs

        tree = ast.parse(
            inspect.getsource(acs_inputs.with_acs_spm_independence_role).strip()
        )
        receipts = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and [getattr(target, "id", None) for target in node.targets]
            == ["provenance"]
            and isinstance(node.value, ast.Dict)
        ]
        assert len(receipts) == 1
        keys = {key.value for key in receipts[0].keys}
        assert keys == contract._ACS_ROLE_RECEIPT_KEYS
        assert set(staging_role_receipt(joined_frame(generate_population(0)))) == keys


_ROLE_RECORD_REFUSALS = [
    (
        "persons",
        lambda f: (
            {**cps_role_record(f), "persons_joined": 1},
            staging_role_receipt(f),
            f,
        ),
        "The CPS role record does not match the frame's CPS rows: {'persons_joined': "
        "{'record': 1, 'frame': ",
    ),
    (
        "units",
        lambda f: (
            {**cps_role_record(f), "native_spm_units": 0},
            staging_role_receipt(f),
            f,
        ),
        "The CPS role record does not match the frame's CPS rows: {'native_spm_units'",
    ),
    (
        "digest only",
        lambda f: (cps_role_record(_swap_cps_roles(f)), staging_role_receipt(f), f),
        "The CPS role record does not match the frame's CPS rows: "
        "{'person_role_binding_sha256'",
    ),
    (
        "whole-frame record",
        lambda f: (spm_role_gate_record(f), staging_role_receipt(f), f),
        "The CPS role record does not match the frame's CPS rows: {'persons_joined'",
    ),
    (
        "missing key",
        lambda f: (
            {
                key: value
                for key, value in cps_role_record(f).items()
                if key != "true_role_persons"
            },
            staging_role_receipt(f),
            f,
        ),
        "The CPS role record does not match the frame's CPS rows: it lacks the "
        "role gate's keys ['true_role_persons']",
    ),
    (
        "boolean count",
        lambda f: (
            {**cps_role_record(f), "table_rows": True},
            staging_role_receipt(f),
            f,
        ),
        "The CPS role record does not match the frame's CPS rows: ['table_rows'] "
        "must be integer counts",
    ),
    (
        # Review r3, should-fix 2: numpy registers timedelta64 as an integer,
        # and it compares equal to its count, so the record matched.
        "timedelta count",
        lambda f: (
            {
                **cps_role_record(f),
                "persons_joined": np.timedelta64(
                    cps_role_record(f)["persons_joined"], "ns"
                ),
            },
            staging_role_receipt(f),
            f,
        ),
        "The CPS role record does not match the frame's CPS rows: "
        "['persons_joined'] must be integer counts",
    ),
    (
        "role source",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _without_column(f, "person", SPM_ROLE),
        ),
        "The SPM role must come from the source column (role_source "
        "'source_column'); the frame's role source is 'household_structure_fallback'",
    ),
    (
        "null role",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _cell(f, "person", SPM_ROLE, None, origin="acs"),
        ),
        "The SPM role record needs a non-null boolean 'is_spm_independent_minor_role'",
    ),
    (
        "missing age",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _without_column(f, "person", "age"),
        ),
        "The SPM role record needs 'age' on the person table",
    ),
    (
        "integer role",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _with_column(
                f, "person", SPM_ROLE, f.table("person")[SPM_ROLE].astype(np.int64)
            ),
        ),
        "The SPM role record needs a non-null boolean 'is_spm_independent_minor_role' "
        "on every person; 0 null(s), inferred type 'integer'",
    ),
    (
        "unknown person channel",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _cell(f, "person", "person_support_channel", "acs", origin="acs"),
        ),
        "The SPM role record needs every row's origin: 'person_support_channel' "
        "must hold one of ['asec', 'puf_tax_detail', 'acs_2024_1yr'] on every row; "
        "it holds [\"'acs'\"]",
    ),
    (
        "null spm unit channel",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _cell(f, "spm_unit", "spm_unit_support_channel", None, origin="acs"),
        ),
        "The SPM role record needs every row's origin: 'spm_unit_support_channel' "
        "must hold one of",
    ),
    (
        "channel column missing",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _without_column(f, "spm_unit", "spm_unit_support_channel"),
        ),
        "The SPM role record needs every row's origin: the 'spm_unit' table lacks "
        "'spm_unit_support_channel'",
    ),
    (
        "receipt keys",
        lambda f: (
            cps_role_record(f),
            {k: v for k, v in staging_role_receipt(f).items() if k != "rule"},
            f,
        ),
        "Staging's ACS role receipt lacks keys ['rule']",
    ),
    (
        "receipt identity",
        lambda f: (
            cps_role_record(f),
            {**staging_role_receipt(f), "provenance": "asec"},
            f,
        ),
        "Staging's ACS role receipt is not the ACS RELSHIPP role rule's: "
        "{'provenance': 'asec'}",
    ),
    (
        "receipt rule",
        lambda f: (
            cps_role_record(f),
            {
                **staging_role_receipt(f),
                "rule": staging_role_receipt(f)["rule"] + " OR RELSHIPP == 22",
            },
            f,
        ),
        "Staging's ACS role receipt is not the ACS RELSHIPP role rule's: {'rule': ",
    ),
    (
        "receipt partition",
        lambda f: (
            cps_role_record(f),
            {**staging_role_receipt(f), "partition": "spm_unit"},
            f,
        ),
        "Staging's ACS role receipt is not the ACS RELSHIPP role rule's: "
        "{'partition': 'spm_unit'}",
    ),
    (
        "receipt coverage",
        lambda f: (cps_role_record(f), staging_role_receipt(f, staged_beyond=-1), f),
        "Staging's ACS role receipt covers fewer rows than the frame's ACS rows: "
        "{'persons'",
    ),
    (
        # Review r3, should-fix 2: a numpy timedelta64 count passed as an
        # integer at least the selected one.
        "receipt timedelta count",
        lambda f: (
            cps_role_record(f),
            {
                **staging_role_receipt(f),
                "spm_units": np.timedelta64(staging_role_receipt(f)["spm_units"], "ns"),
            },
            f,
        ),
        "Staging's ACS role receipt covers fewer rows than the frame's ACS rows: "
        "{'spm_units'",
    ),
    # Each ACS person's role is the one its RELSHIPP gives (review
    # should-fix 1): the receipt states counts only.
    (
        "acs roles traded",
        lambda f: (cps_role_record(f), staging_role_receipt(f), _swap_acs_roles(f)),
        "The ACS persons' SPM roles are not staging's RELSHIPP rule's: 2 of ",
    ),
    (
        "relshipp missing",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _without_column(f, "person", "RELSHIPP"),
        ),
        "The ACS persons' SPM roles are not staging's RELSHIPP rule's: the person "
        "table lacks 'RELSHIPP'",
    ),
    (
        "relshipp off-domain",
        lambda f: (
            cps_role_record(f),
            staging_role_receipt(f),
            _cell(f, "person", "RELSHIPP", 19, origin="acs"),
        ),
        "The ACS persons' SPM roles are not staging's RELSHIPP rule's: 1 ACS "
        "person(s) hold a blank or off-domain RELSHIPP",
    ),
    # Review r2, should-fix 1: each keeps the person's code as its real part
    # or count of units. The complex forms and the timedelta64 column passed,
    # read as the person's own code; one numpy timedelta64 in an object
    # column already refused as off-domain, and now gets this message.
    # Review r3, should-fix 1: the Arrow dictionaries of times passed too.
    *(
        (
            f"relshipp {form}",
            lambda f, form=form: (
                cps_role_record(f),
                staging_role_receipt(f),
                _relshipp_as(f, form),
            ),
            "The ACS persons' SPM roles are not staging's RELSHIPP rule's: RELSHIPP "
            f"on ACS rows holds {found}",
        )
        for form, found in (
            ("complex128 column", "complex128 values, not integer codes"),
            (
                "complex in object",
                "1 value(s) that are not real numbers or strings, such as [(",
            ),
            ("timedelta64 column", "timedelta64[ns] values, not integer codes"),
            (
                "numpy timedelta64 in object",
                "1 value(s) that are not real numbers or strings, such as "
                "[np.timedelta64(",
            ),
            (
                "Arrow dictionary of durations",
                "dictionary<values=duration[ns], indices=int32, ordered=0>"
                "[pyarrow] values, not integer codes",
            ),
            (
                "Arrow dictionary of timestamps",
                "dictionary<values=timestamp[ns], indices=int32, ordered=0>"
                "[pyarrow] values, not integer codes",
            ),
        )
    ),
]


@pytest.mark.parametrize(
    ("case", "inputs", "message"),
    _ROLE_RECORD_REFUSALS,
    ids=[case for case, _, _ in _ROLE_RECORD_REFUSALS],
)
def test_the_role_record_refuses_with_its_own_message(
    assembled, case, inputs, message
) -> None:
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        combined_spm_role_record(*inputs(assembled))


def test_the_role_record_refusal_messages_are_distinct() -> None:
    messages = [message for _, _, message in _ROLE_RECORD_REFUSALS]
    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize(
    "form", ["Int64", "object", "int64[pyarrow]", "Arrow dictionary of int64"]
)
def test_relshipp_in_a_real_dtype_passes(assembled, form) -> None:
    # Review r3, should-fix 1: the allowlist reads these codes. The frame
    # holds RELSHIPP as float64, null on CPS rows (test_the_record_passes_
    # the_real_role_gate covers it); staging's PUMS parse reads it as a
    # number, and the review's probes kept nullable and object codes valid.
    tables = _tables(assembled)
    person = tables["person"]
    codes = person["RELSHIPP"]
    if form == "Arrow dictionary of int64":
        person["RELSHIPP"] = _arrow_dictionary(codes, pa.int64())
    elif form == "int64[pyarrow]":
        person["RELSHIPP"] = pd.Series(
            pd.arrays.ArrowExtensionArray(
                pa.array([None if pd.isna(code) else int(code) for code in codes])
            ),
            index=codes.index,
        )
        assert str(person["RELSHIPP"].dtype) == form
    elif form == "object":
        person["RELSHIPP"] = pd.Series(
            [None if pd.isna(code) else int(code) for code in codes],
            index=codes.index,
            dtype=object,
        )
    else:
        person["RELSHIPP"] = codes.astype(form)
    frame = _rebuild(assembled, tables)
    combined_spm_role_record(*_role_inputs(frame), frame)


def test_the_role_records_refusals_are_live_against_the_gate(assembled) -> None:
    # The digest-only corruption is one the gate itself would catch on a
    # frame whose roles moved: the record of the swapped frame fails the
    # gate on the original frame.
    swapped = _swap_cps_roles(assembled)
    stale = spm_role_gate_record(swapped)
    result = us_spm_independence_role_signal_gate(
        with_spm_role_record(assembled, stale)
    )
    assert not result.passed
    assert stale["persons_joined"] == spm_role_gate_record(assembled)["persons_joined"]


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_a_cps_record_derived_before_the_join_is_accepted(seed, form) -> None:
    # PR 4 derives the CPS record on the selected CPS rows before they meet
    # the ACS rows: the join must keep their order and the digest's dtypes.
    population = _form(generate_population(seed), form)
    frame = assembled_frame(population)
    person = population.cps_tables["person"]
    record = {
        "persons_joined": len(person),
        "table_rows": len(person),
        "native_spm_units": len(population.cps_tables["spm_unit"]),
        "true_role_persons": int(person[SPM_ROLE].sum()),
        "person_role_binding_sha256": _role_binding_sha256(person),
    }
    combined = combined_spm_role_record(record, staging_role_receipt(frame), frame)
    assert (
        combined["origins"]["CPS"]["person_role_binding_sha256"]
        == (record["person_role_binding_sha256"])
    )


# -------------------------------------------- phase 3: absences (A1, Q2)


def _absences(frame: Frame, *, fills=(), zero_written=None, declared=None) -> dict:
    return absences(
        frame,
        fill_manifest=fill_manifest(fills),
        zero_written=zero_written or {},
        declared=declared or {},
    )


def _keyed(record: dict) -> dict[tuple[str, str, str, str], dict]:
    keyed = {
        (entry["entity"], entry["column"], entry["kind"], entry["origin"]): entry
        for entry in record["entries"]
    }
    assert len(keyed) == len(record["entries"])
    return keyed


def _share(frame: Frame, entity: str, mask: np.ndarray) -> float:
    weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
    return float(weights[mask].sum() / weights.sum())


_GENERATED_ABSENCES = {
    ("household", "SERIALNO", "raw source column", "CPS"),
    ("household", "TYPEHUGQ", "raw source column", "CPS"),
    ("person", "PERIDNUM", "raw source column", "ACS"),
    # The ACS role rule's source column, kept on ACS rows (review should-fix 1).
    ("person", "RELSHIPP", "raw source column", "CPS"),
    ("person", "SPORDER", "raw source column", "CPS"),
    ("person", "SSI_VAL", "raw source column", "ACS"),
    ("person", "takes_up_ssi_if_eligible", "release-written", "ACS"),
}


def _acs_fill(frame: Frame, column: str, *, rows: int | None = None) -> dict:
    table = frame.table("person")
    acs = int(_origin_mask(table, "person", "acs").sum())
    filled = acs if rows is None else rows
    return {
        "entity": "person",
        "column": column,
        "filled_rows": filled,
        "rows": len(table),
        "fill_value": "0.0",
        "fill_kind": "variable default_value",
        "register_missing_rows": filled,
        "missing_rows_by_spine": {ACS_2024_1YR_SPINE: filled},
    }


class TestAbsences:
    @pytest.mark.parametrize("form", _FORMS)
    def test_both_directions_on_the_generated_population(self, form) -> None:
        frame = assembled_frame(_form(generate_population(0), form))
        record = _absences(frame)
        assert set(_keyed(record)) == _GENERATED_ABSENCES

    def test_rows_and_shares_are_keyed_by_channel(self, assembled) -> None:
        keyed = _keyed(_absences(assembled))
        person = assembled.table("person")
        acs = _origin_mask(person, "person", "acs")
        ssi = keyed[("person", "takes_up_ssi_if_eligible", "release-written", "ACS")]
        assert ssi["rows"] == ssi["origin_rows"] == int(acs.sum())
        assert ssi["rows_by_channel"] == {ACS_2024_1YR_SPINE: int(acs.sum())}
        assert ssi["entity_weight_share"] == pytest.approx(
            _share(assembled, "person", acs)
        )
        assert ssi["evidence"] == RELEASE_WRITTEN_INPUTS["takes_up_ssi_if_eligible"]
        sporder = keyed[("person", "SPORDER", "raw source column", "CPS")]
        channel = person["person_support_channel"]
        assert sporder["rows_by_channel"] == {
            BASE_ASEC_SUPPORT_CHANNEL: int(channel.eq(BASE_ASEC_SUPPORT_CHANNEL).sum()),
            PUF_TAX_DETAIL_SUPPORT_CHANNEL: int(
                channel.eq(PUF_TAX_DETAIL_SUPPORT_CHANNEL).sum()
            ),
        }
        by_channel = sporder["entity_weight_share_by_channel"]
        assert by_channel[BASE_ASEC_SUPPORT_CHANNEL] == pytest.approx(
            _share(
                assembled, "person", channel.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
            )
        )
        assert sum(by_channel.values()) == pytest.approx(sporder["entity_weight_share"])
        assert sporder["entity_weight_share"] + ssi["entity_weight_share"] == (
            pytest.approx(1.0)
        )

    def test_every_entity_resolves_weights(self, assembled) -> None:
        record = _absences(assembled)
        for entity in US_SCHEMA.entities:
            rows = record["rows_by_channel"][entity]
            assert sum(rows.values()) == assembled.n(entity)
            weights = assembled.resolve_weights(entity).values
            assert len(weights) == assembled.n(entity)
            assert np.isfinite(weights).all() and weights.sum() > 0

    def test_a_column_null_on_every_row_is_no_absence(self, assembled) -> None:
        # Neither origin holds a value, so neither lacks it relative to the
        # other: no entry and no refusal, although parent_1_id is model-named.
        frame = _with_column(
            assembled,
            "person",
            "parent_1_id",
            pd.Series(None, index=assembled.table("person").index, dtype=object),
        )
        assert set(_keyed(_absences(frame))) == _GENERATED_ABSENCES

    def test_typehugq_is_not_filed_as_harmless(self, assembled) -> None:
        typehugq = _keyed(_absences(assembled))[
            ("household", "TYPEHUGQ", "raw source column", "CPS")
        ]
        assert typehugq["readers"] == list(RAW_COLUMN_READERS["TYPEHUGQ"])
        assert "#1610" in typehugq["readers"][0]
        assert "the release reads it" in typehugq["evidence"]

    def test_the_household_type_and_predictor_columns_name_their_readers(
        self, assembled
    ) -> None:
        frame = _acs_null_on_cps(assembled, "household", "HRHTYPE", 1)
        for column, value in (
            ("A_EXPRRP", 1),
            ("A_MARITL", 7),
            ("WKSWORK", 52),
            ("PRDTRACE", 1),
            ("PEDISDRS", 2),
        ):
            frame = _acs_null_on_cps(frame, "person", column, value)
        keyed = _keyed(_absences(frame))
        for entity, column in (
            ("household", "HRHTYPE"),
            ("person", "A_EXPRRP"),
            ("person", "A_MARITL"),
        ):
            entry = keyed[(entity, column, "raw source column", "ACS")]
            assert entry["readers"] == [contract._HOUSEHOLD_TYPE_READER]
        assert keyed[("person", "WKSWORK", "raw source column", "ACS")]["readers"] == [
            "org_wages"
        ]
        assert keyed[("person", "PRDTRACE", "raw source column", "ACS")]["readers"] == [
            "org_wages",
            "scf_auto_loans",
            "scf_wealth",
        ]
        assert keyed[("person", "PEDISDRS", "raw source column", "ACS")]["readers"] == [
            "ssi_disability_criteria"
        ]
        # A raw column no release reader is listed for says so.
        ssi_val = keyed[("person", "SSI_VAL", "raw source column", "ACS")]
        assert ssi_val["readers"] == []
        assert "no release reader is listed" in ssi_val["evidence"]

    def test_raw_readers_are_the_predictor_joins_producers(self) -> None:
        from microcosm.build.us_runtime import acs_release_predictors

        inverted: dict[str, set[str]] = {}
        for producer, columns in acs_release_predictors._MODEL_PREDICTORS.items():
            for column in columns:
                inverted.setdefault(column, set()).add(producer)
        household_type = {"TYPEHUGQ", "HRHTYPE", "A_EXPRRP", "A_MARITL"}
        assert set(RAW_COLUMN_READERS) == set(inverted) | household_type
        assert set(acs_release_predictors._OUTPUT_COLUMNS) == set(inverted)
        for column, producers in inverted.items():
            assert RAW_COLUMN_READERS[column] == tuple(sorted(producers)), column
        for column in household_type:
            assert RAW_COLUMN_READERS[column] == (contract._HOUSEHOLD_TYPE_READER,)
        assert ABSENCE_KINDS == (
            "owner-declared",
            "release-written",
            "raw source column",
            "reviewed default fill",
            "zero written",
            "origin-native",
        )

    def test_a_reviewed_default_fill_over_every_acs_row(self, assembled) -> None:
        record = _absences(assembled, fills=[_acs_fill(assembled, "tip_income")])
        entry = _keyed(record)[("person", "tip_income", "reviewed default fill", "ACS")]
        acs = _origin_mask(assembled.table("person"), "person", "acs")
        assert entry["rows_by_channel"] == {ACS_2024_1YR_SPINE: int(acs.sum())}
        assert entry["entity_weight_share"] == pytest.approx(
            _share(assembled, "person", acs)
        )
        assert entry["detail"] == {
            "fill_value": "0.0",
            "fill_kind": "variable default_value",
            "register_missing_rows": int(acs.sum()),
        }
        assert "reviewed_engine_input_nulls" in entry["evidence"]

    def test_a_partial_fill_names_no_rows_or_share(self, assembled) -> None:
        cps_fill = {
            **_acs_fill(assembled, "employer_sponsored_insurance_premiums", rows=3),
            "missing_rows_by_spine": {CPS_SPINE_TAG: 2, ACS_2024_1YR_SPINE: 1},
        }
        keyed = _keyed(_absences(assembled, fills=[cps_fill]))
        for origin, rows in (("CPS", 2), ("ACS", 1)):
            entry = keyed[
                (
                    "person",
                    "employer_sponsored_insurance_premiums",
                    "reviewed default fill",
                    origin,
                )
            ]
            assert entry["rows"] == rows
            assert entry["rows_by_channel"] is None
            assert entry["entity_weight_share"] is None
            assert "names no rows" in entry["rows_rule"]

    def test_a_zero_spine_count_files_no_entry(self, assembled) -> None:
        # Review should-fix 5's bound is nonnegative, not positive: a spine
        # the fill did not touch may count zero, and files nothing.
        fill = {
            **_acs_fill(assembled, "tip_income", rows=3),
            "missing_rows_by_spine": {CPS_SPINE_TAG: 0, ACS_2024_1YR_SPINE: 3},
        }
        keyed = _keyed(_absences(assembled, fills=[fill]))
        assert (
            keyed[("person", "tip_income", "reviewed default fill", "ACS")]["rows"] == 3
        )
        assert ("person", "tip_income", "reviewed default fill", "CPS") not in keyed

    def test_zero_written_on_a_nullable_numeric_column(self, assembled) -> None:
        # Review should-fix 6: a Float64 origin-native amount, pd.NA on CPS
        # rows and zeros on two ACS rows. eq(0) on it is a nullable boolean,
        # which used to reach the weight lookup as an object array. Review r2,
        # nit 4: a third ACS row holds pd.NA, so a mask that read a null as a
        # zero would count three ACS rows, not two.
        column = "acs_social_security_income"
        frame = _acs_only(assembled, "person", column, 120.0)
        tables = _tables(frame)
        person = tables["person"]
        acs = _origin_mask(person, "person", "acs")
        values = person[column].astype("Float64")
        zeros, null = np.flatnonzero(acs)[:2], np.flatnonzero(acs)[2]
        values.iloc[zeros] = 0.0
        values.iloc[null] = pd.NA
        person[column] = values
        frame = _rebuild(frame, tables)
        assert str(frame.table("person")[column].dtype) == "Float64"
        assert int(frame.table("person")[column].isna().sum()) == int((~acs).sum()) + 1
        record = _absences(frame, zero_written={column: "receipt"})
        entry = _keyed(record)[("person", column, "zero written", "ACS")]
        mask = np.zeros(len(person), dtype=bool)
        mask[zeros] = True
        assert entry["rows"] == 2
        assert entry["rows_by_channel"] == {ACS_2024_1YR_SPINE: 2}
        assert entry["entity_weight_share"] == pytest.approx(
            _share(frame, "person", mask)
        )

    def test_zero_written_counts_acs_zeros_as_an_upper_bound(self, assembled) -> None:
        receipt = "acs_transfer cells_kept_at_zero (U/acs_transfer.py:2440-2468)"
        record = _absences(assembled, zero_written={"tip_income": receipt})
        entry = _keyed(record)[("person", "tip_income", "zero written", "ACS")]
        person = assembled.table("person")
        zeros = _origin_mask(person, "person", "acs") & person["tip_income"].eq(0)
        assert entry["rows"] == int(zeros.sum()) > 0
        assert entry["entity_weight_share"] == pytest.approx(
            _share(assembled, "person", zeros.to_numpy())
        )
        assert entry["detail"] == {"receipt": receipt}
        assert "upper bound" in entry["rows_rule"]

    def test_an_owner_declared_absence(self, assembled) -> None:
        frame = _acs_null_on_cps(assembled, "person", "parent_1_id", 3)
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A column one origin lacks has no absence kind: "
                "{'person.parent_1_id': 'null on"
            ),
        ):
            _absences(frame)
        decision = "US data owner: ACS rows carry no parent link (d2283 Q2)"
        entry = _keyed(_absences(frame, declared={"parent_1_id": decision}))[
            ("person", "parent_1_id", "owner-declared", "ACS")
        ]
        assert entry["detail"] == {"decision": decision}
        assert entry["evidence"] == decision

    def test_a_frame_with_no_acs_row(self, population) -> None:
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        record = _absences(frame)
        assert record["entries"] == []
        assert record["rows_by_channel"]["person"][ACS_2024_1YR_SPINE] == 0
        contract._absences_json(record)  # allow_nan=False holds

    def test_the_json_is_deterministic(self, population, tmp_path) -> None:
        records = [
            _absences(
                assembled_frame(_form(population, form)),
                zero_written={"tip_income": "receipt"},
            )
            for form in _FORMS
        ]
        texts = [contract._absences_json(record) for record in records]
        assert texts[0] == texts[1]
        path = write_absences(records[0], tmp_path)
        assert path == tmp_path / ABSENCES_FILENAME
        assert path.read_text(encoding="ascii") == texts[0]
        assert json.loads(texts[0]) == records[0]
        assert (
            texts[0]
            == json.dumps(json.loads(texts[0]), sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        with pytest.raises(ValueError):
            contract._absences_json({**records[0], "bad": float("nan")})


_ABSENCE_REFUSALS = [
    (
        "fill on a missing column",
        lambda f: {"fills": [{**_acs_fill(f, "tip_income"), "column": "no_such"}]},
        "The fill manifest does not describe this frame: it fills 'person'.'no_such'",
    ),
    (
        "fill rows",
        lambda f: {"fills": [{**_acs_fill(f, "tip_income"), "rows": 1}]},
        "The fill manifest does not describe this frame: its 'person' table had 1 rows",
    ),
    (
        # Review r3, should-fix 2: a numpy timedelta64 count equals its count
        # of units, so it passed as the table's row count.
        "fill rows timedelta",
        lambda f: {
            "fills": [
                {
                    **_acs_fill(f, "tip_income"),
                    "rows": np.timedelta64(len(f.table("person")), "ns"),
                }
            ]
        },
        "The fill manifest does not describe this frame: its 'person' table had "
        "np.timedelta64(",
    ),
    (
        "fill spine tag",
        lambda f: {
            "fills": [
                {
                    **_acs_fill(f, "tip_income", rows=2),
                    "missing_rows_by_spine": {"acs": 2},
                }
            ]
        },
        "The fill manifest does not describe this frame: 'person'.'tip_income''s rows "
        "by spine {'acs': 2}",
    ),
    (
        "fill count",
        lambda f: {
            "fills": [
                {
                    **_acs_fill(f, "tip_income", rows=2),
                    "missing_rows_by_spine": {ACS_2024_1YR_SPINE: 3},
                }
            ]
        },
        "The fill manifest does not describe this frame: 'person'.'tip_income''s rows "
        "by spine {'acs_2024_1yr': 3}",
    ),
    (
        # Review should-fix 5: the sum matches, and the origin-capacity check
        # rejects only counts above capacity, so -1 became a CPS entry's rows.
        "fill negative count",
        lambda f: {
            "fills": [
                {
                    **_acs_fill(f, "tip_income", rows=3),
                    "missing_rows_by_spine": {
                        CPS_SPINE_TAG: -1,
                        ACS_2024_1YR_SPINE: 4,
                    },
                }
            ]
        },
        "The fill manifest does not describe this frame: 'person'.'tip_income''s rows "
        "by spine {'asec_puf': -1, 'acs_2024_1yr': 4} must count its 3 filled rows, "
        "in nonnegative integers",
    ),
    (
        "fill beyond the origin",
        lambda f: {
            "fills": [
                {
                    **_acs_fill(f, "tip_income", rows=10_000),
                    "rows": len(f.table("person")),
                }
            ]
        },
        "The fill manifest does not describe this frame: it fills 10000 ACS rows",
    ),
    (
        "fill leaves a null",
        lambda f: {
            "frame": _cell(f, "person", "tip_income", None, origin="acs"),
            "fills": [_acs_fill(f, "tip_income")],
        },
        "The fill manifest does not describe this frame: 'person'.'tip_income' still "
        "holds 1 null(s)",
    ),
    (
        "fill twice",
        lambda f: {"fills": [_acs_fill(f, "tip_income")] * 2},
        "The fill manifest does not describe this frame: it fills 'person'.'tip_income' "
        "twice",
    ),
    (
        "manifest totals",
        lambda f: {
            "manifest": {
                **fill_manifest([_acs_fill(f, "tip_income")]),
                "columns_filled": 2,
            }
        },
        "The fill manifest does not describe this frame: it says 2 columns",
    ),
    (
        "manifest keys",
        lambda f: {"manifest": {"fills": []}},
        "The fill manifest does not describe this frame: it lacks keys",
    ),
    (
        "manifest violations",
        lambda f: {
            "manifest": {
                **fill_manifest(),
                "unregistered_violations": [{"column": "tip_income"}],
            }
        },
        "The fill manifest does not describe this frame: it records unregistered "
        "violations",
    ),
    (
        "zero written absent",
        lambda f: {"zero_written": {"no_such": "receipt"}},
        "A zero-written column the frame lacks: 'no_such'",
    ),
    (
        "zero written text",
        lambda f: {"zero_written": {"SERIALNO": "receipt"}},
        "A zero-written column must be numeric: 'household'.'SERIALNO'",
    ),
    (
        "zero written receipt",
        lambda f: {"zero_written": {"tip_income": " "}},
        "A zero-written column needs its receipt; 'tip_income' has none",
    ),
    (
        "two kinds",
        lambda f: {
            "fills": [_acs_fill(f, "tip_income")],
            "zero_written": {"tip_income": "receipt"},
        },
        "An absence has one kind: ['tip_income'] are named by both fill_manifest and "
        "zero_written",
    ),
    (
        "declared empty",
        lambda f: {"declared": {"parent_1_id": " "}},
        "A declared absence needs its owner's decision; 'parent_1_id' has none",
    ),
    (
        "unknown channel",
        lambda f: {
            "frame": _cell(f, "family", "family_support_channel", "acs", origin="acs")
        },
        "The absences file needs every row's origin: 'family_support_channel' must "
        "hold one of",
    ),
    (
        "null channel",
        lambda f: {
            "frame": _cell(f, "person", "person_support_channel", None, origin="cps")
        },
        "The absences file needs every row's origin: 'person_support_channel' must "
        "hold one of ['asec', 'puf_tax_detail', 'acs_2024_1yr'] on every row; it "
        "holds ['None']",
    ),
    (
        "channel column missing",
        lambda f: {"frame": _without_column(f, "tax_unit", "tax_unit_support_channel")},
        "The absences file needs every row's origin: the 'tax_unit' table lacks "
        "'tax_unit_support_channel'",
    ),
    (
        "undeclared",
        lambda f: {"frame": _acs_null_on_cps(f, "tax_unit", "filing_status_input", 1)},
        "A column one origin lacks has no absence kind: "
        "{'tax_unit.filing_status_input': 'null on",
    ),
    (
        # ACS rows carry a PUMA, CPS rows none: not one of the six
        # origin-native amounts, so it refuses until filled or declared.
        "undeclared reverse",
        lambda f: {"frame": _acs_only(f, "household", "puma_geoid", "0600101")},
        "A column one origin lacks has no absence kind: "
        "{'household.puma_geoid': 'null on",
    ),
]


@pytest.mark.parametrize(
    ("case", "corrupt", "message"),
    _ABSENCE_REFUSALS,
    ids=[case for case, _, _ in _ABSENCE_REFUSALS],
)
def test_absences_refuse_with_their_own_message(
    assembled, case, corrupt, message
) -> None:
    inputs = corrupt(assembled)
    frame = inputs.get("frame", assembled)
    manifest = inputs.get("manifest") or fill_manifest(inputs.get("fills", ()))
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        absences(
            frame,
            fill_manifest=manifest,
            zero_written=inputs.get("zero_written", {}),
            declared=inputs.get("declared", {}),
        )


def test_absence_refusal_messages_are_distinct() -> None:
    messages = [message for _, _, message in _ABSENCE_REFUSALS]
    assert len(set(messages)) == len(messages)


def test_every_kindless_column_is_listed_in_one_refusal(assembled) -> None:
    # One refusal names every gap, in both directions, so a first live run
    # lists them all at once.
    frame = _acs_null_on_cps(assembled, "person", "parent_1_id", 3)
    acs = _origin_mask(frame.table("person"), "person", "acs")
    frame = _acs_only(frame, "household", "puma_geoid", "0600101")
    cps_households = int(
        (~_origin_mask(frame.table("household"), "household", "acs")).sum()
    )
    with pytest.raises(OnePopulationContractError) as raised:
        _absences(frame)
    message = str(raised.value)
    assert f"'person.parent_1_id': 'null on {int(acs.sum())} ACS row(s)'" in message
    assert (f"'household.puma_geoid': 'null on {cps_households} CPS row(s)'") in message


# ---- the moved inputs on a whole population (lead-call review, 11-19)


def _with_moved_inputs(frame: Frame, *, acs_null: bool) -> Frame:
    """Every moved input on its entity: on every row, or null on ACS rows.

    PR 4's genuine input carries each on CPS rows (the parent export's
    release values) and fills each on ACS rows; the generator carries none.
    """

    entities = lock_engine_inputs()
    tables = _tables(frame)
    for column in _MOVED_OFF_THE_EXEMPT_LIST:
        entity = entities[column]
        table = tables[entity]
        value = True if column in _MOVED_BOOLEANS else 1.0
        if acs_null:
            acs = _origin_mask(table, entity, "acs")
            table[column] = pd.Series(
                np.where(acs, None, value), index=table.index, dtype=object
            )
        else:
            table[column] = value
    return _rebuild(frame, tables)


class TestMovedInputs:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_filled_on_every_row_they_pass_and_add_no_absence(self, seed) -> None:
        population = generate_population(seed)
        frame = assembled_frame(population)
        filled = _with_moved_inputs(frame, acs_null=False)
        _require(population, filled, staging_lineage=staging_lineage(population))
        assert set(_keyed(_absences(filled))) == set(_keyed(_absences(frame)))

    def test_null_on_acs_rows_both_checks_name_every_one(
        self, population, assembled
    ) -> None:
        frame = _with_moved_inputs(assembled, acs_null=True)
        with pytest.raises(OnePopulationContractError) as refused:
            _require(population, frame)
        message = str(refused.value)
        assert message.startswith("A null in an engine input at 2024: "), message
        missing = [c for c in _MOVED_OFF_THE_EXEMPT_LIST if f".{c}'" not in message]
        assert not missing, message
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape("A column one origin lacks has no absence kind"),
        ) as gap:
            _absences(frame)
        missing = [
            c for c in _MOVED_OFF_THE_EXEMPT_LIST if f".{c}'" not in str(gap.value)
        ]
        assert not missing, str(gap.value)


# ------- lead call (a), 11-18: ACS-native amounts CPS rows lack by origin


def _acs_only(frame: Frame, entity: str, column: str, value) -> Frame:
    """``column`` holding ``value`` on ACS rows and null on CPS rows."""

    table = frame.table(entity)
    acs = _origin_mask(table, entity, "acs")
    return _with_column(
        frame,
        entity,
        column,
        pd.Series(np.where(acs, value, None), index=table.index, dtype=object),
    )


def _with_acs_native_amounts(frame: Frame) -> Frame:
    for column, (entity, _) in contract.ACS_NATIVE_AMOUNTS.items():
        frame = _acs_only(frame, entity, column, 120.0)
    return frame


def _module_assignment(path: Path, name: str) -> ast.expr:
    """The value assigned to ``name`` at the top level of ``path``'s source."""

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else node.targets
            if isinstance(node, ast.Assign)
            else []
        )
        if [getattr(target, "id", None) for target in targets] == [name]:
            return node.value
    raise AssertionError(f"{path.name} assigns no {name}.")


_U = Path(contract.__file__).parent


class TestOriginNativeAmounts:
    """The six ACS-native ``acs_*`` amounts CPS rows lack need no decision."""

    def test_they_are_origin_native_on_cps_rows(self, population, assembled) -> None:
        frame = _with_acs_native_amounts(assembled)
        record = _absences(frame)
        keyed = _keyed(record)
        assert "origin-native" in record["kinds"]
        for column, (entity, readers) in contract.ACS_NATIVE_AMOUNTS.items():
            entry = keyed[(entity, column, "origin-native", "CPS")]
            cps = ~_origin_mask(frame.table(entity), entity, "acs")
            assert entry["rows"] == entry["origin_rows"] == int(cps.sum())
            assert entry["entity_weight_share"] == pytest.approx(
                _share(frame, entity, cps)
            )
            assert "_ACS_NATIVE_INPUT_CONTRACTS" in entry["evidence"]
            assert entry["readers"] == list(readers)
            assert entry["detail"] == {}
        # No declaration is needed, and the contract passes the same frame.
        assert record["needless_declarations"] == []
        _require(population, frame)

    def test_the_transfer_reads_the_three_combined_amounts(self) -> None:
        # acs_transfer._RECIPIENT_COMBINED_SOURCES, read from its source.
        sources = _module_assignment(
            _U / "acs_transfer.py", "_RECIPIENT_COMBINED_SOURCES"
        )
        assert isinstance(sources, ast.Dict)
        read = {value.value for value in sources.values}
        assert read == {
            column
            for column, (_, readers) in contract.ACS_NATIVE_AMOUNTS.items()
            if readers
        }
        assert all(
            "acs_transfer" in reader
            for _, readers in contract.ACS_NATIVE_AMOUNTS.values()
            for reader in readers
        )

    def test_the_names_are_the_operator_boundary_contracts(self) -> None:
        # Read from operator_boundary's source: importing it loads the ACS
        # transfer, and with it microcosm.fit.
        contracts = _module_assignment(
            _U / "operator_boundary.py", "_ACS_NATIVE_INPUT_CONTRACTS"
        )
        assert isinstance(contracts, ast.Dict)
        literal = {
            key.value: value.elts[0].value
            for key, value in zip(contracts.keys, contracts.values, strict=True)
            if key is not None
        }
        spreads = [
            value
            for key, value in zip(contracts.keys, contracts.values, strict=True)
            if key is None
        ]
        spread_sources = {
            generator.iter.func.value.id
            if isinstance(generator.iter, ast.Call)
            else generator.iter.id
            for spread in spreads
            for generator in spread.generators
        }
        assert spread_sources == {
            "ACS_WORK_EXPERIENCE_NATIVE_MAPPINGS",
            "ACS_UNRESOLVED_PARENT_ID_MAPPINGS",
        }
        work = ast.literal_eval(
            _module_assignment(
                _U / "acs_inputs.py", "ACS_WORK_EXPERIENCE_NATIVE_MAPPINGS"
            )
        )
        parents = ast.literal_eval(
            _module_assignment(
                _U / "acs_inputs.py", "ACS_UNRESOLVED_PARENT_ID_MAPPINGS"
            )
        )
        spread_outputs = set(work) | {output for output, _ in parents}
        assert not {name for name in spread_outputs if name.startswith("acs_")}
        amounts = {
            column: entity
            for column, entity in literal.items()
            if column.startswith("acs_")
        }
        assert len(amounts) == 6
        assert {
            column: entity
            for column, (entity, _) in contract.ACS_NATIVE_AMOUNTS.items()
        } == amounts
        # The register files each as an ACS-native amount, not a model input.
        for column in amounts:
            assert "not a model input" in US_STORED_NON_VARIABLE_COLUMNS[column]

    def test_the_reverse_direction_still_refuses(self, assembled) -> None:
        frame = _acs_null_on_cps(assembled, "person", "acs_social_security_income", 1.0)
        n_acs = int(_origin_mask(frame.table("person"), "person", "acs").sum())
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A column one origin lacks has no absence kind: "
                f"{{'person.acs_social_security_income': 'null on {n_acs} ACS row(s)'}}"
            ),
        ):
            _absences(frame)
        decision = "US data owner: decided for this test"
        record = _absences(frame, declared={"acs_social_security_income": decision})
        entry = _keyed(record)[
            ("person", "acs_social_security_income", "owner-declared", "ACS")
        ]
        assert entry["detail"] == {"decision": decision}

    @pytest.mark.parametrize(
        ("entity", "column"),
        [
            # A model-named column outside the six.
            ("person", "parent_1_id"),
            # One of the six, on an entity its contract does not name.
            ("person", "acs_monthly_gross_rent"),
        ],
    )
    def test_any_other_column_cps_rows_lack_refuses(
        self, assembled, entity, column
    ) -> None:
        frame = _acs_only(assembled, entity, column, 3.0)
        n_cps = int((~_origin_mask(frame.table(entity), entity, "acs")).sum())
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A column one origin lacks has no absence kind: "
                f"{{'{entity}.{column}': 'null on {n_cps} CPS row(s)'}}"
            ),
        ):
            _absences(frame)


# --------- lead call (b), 11-18: one declarations mapping for R10 and absences


_DECISION = "US data owner: decided for this test"


class TestOneDeclarationsMapping:
    def test_stale_and_needless_declarations_are_recorded(
        self, population, assembled
    ) -> None:
        register = {**US_STORED_NON_VARIABLE_COLUMNS, _AUXILIARY: "auxiliary"}
        frame = _acs_null_on_cps(assembled, "person", _AUXILIARY, 41_000.0)
        declared = {
            _AUXILIARY: f"{_DECISION} (used)",
            # Stale: no origin lacks it (staging filled it, say).
            "parent_1_id": f"{_DECISION} (stale)",
            # Needless: release-written, and a raw source column.
            "takes_up_ssi_if_eligible": f"{_DECISION} (release-written)",
            "SSI_VAL": f"{_DECISION} (raw)",
        }
        _require(population, frame, register=register, declared_absences=declared)
        record = _absences(frame, declared=declared)
        assert record["unused_declarations"] == [
            {"column": "parent_1_id", "decision": declared["parent_1_id"]}
        ]
        assert record["needless_declarations"] == [
            {"column": "SSI_VAL", "decision": declared["SSI_VAL"]},
            {
                "column": "takes_up_ssi_if_eligible",
                "decision": declared["takes_up_ssi_if_eligible"],
            },
        ]
        keyed = _keyed(record)
        assert ("person", _AUXILIARY, "owner-declared", "ACS") in keyed
        # The needless ones keep the kinds they need no decision for.
        assert ("person", "SSI_VAL", "raw source column", "ACS") in keyed
        assert ("person", "takes_up_ssi_if_eligible", "release-written", "ACS") in (
            keyed
        )
        assert "owner-declared" not in {
            kind for (_, column, kind, _) in keyed if column != _AUXILIARY
        }

    def test_a_declaration_an_origin_native_gap_needs_not_is_needless(
        self, assembled
    ) -> None:
        frame = _with_acs_native_amounts(assembled)
        declared = {"acs_social_security_income": _DECISION}
        record = _absences(frame, declared=declared)
        assert record["needless_declarations"] == [
            {"column": "acs_social_security_income", "decision": _DECISION}
        ]
        assert record["unused_declarations"] == []
        assert (
            "person",
            "acs_social_security_income",
            "origin-native",
            "CPS",
        ) in _keyed(record)

    @pytest.mark.parametrize("receipt", ["fill_manifest", "zero_written"])
    def test_a_declaration_a_receipt_covers_is_needless(
        self, population, assembled, receipt
    ) -> None:
        # The owner declared a gap that prepare later filled (or staging wrote
        # as zero): the receipt files the entry and the declaration is
        # recorded as needless, on the mapping R10 also accepts.
        declared = {"tip_income": f"{_DECISION} (before the fill)"}
        inputs = (
            {"fills": [_acs_fill(assembled, "tip_income")]}
            if receipt == "fill_manifest"
            else {"zero_written": {"tip_income": "staging receipt"}}
        )
        _require(population, assembled, declared_absences=declared)
        record = _absences(assembled, declared=declared, **inputs)
        assert record["needless_declarations"] == [
            {"column": "tip_income", "decision": declared["tip_income"]}
        ]
        assert record["unused_declarations"] == []
        kinds = {
            kind for (_, column, kind, _) in _keyed(record) if column == "tip_income"
        }
        assert kinds == {
            "reviewed default fill" if receipt == "fill_manifest" else "zero written"
        }

    def test_one_declaration_answers_r10_and_absences(
        self, population, assembled
    ) -> None:
        # d2283 lead call L2: R10 and absences share one gap predicate, so a
        # register column null on every ACS row and on one CPS row refuses
        # both undeclared, and one declaration passes both.
        register = {**US_STORED_NON_VARIABLE_COLUMNS, _AUXILIARY: "auxiliary"}
        frame = _acs_null_on_cps(assembled, "person", _AUXILIARY, 41_000.0)
        frame = _cell(frame, "person", _AUXILIARY, None, origin="cps")
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A model-named column held on some CPS row is null on ACS rows "
                f"without a declared absence: {{'person.{_AUXILIARY}': "
            ),
        ):
            _require(population, frame, register=register)
        with pytest.raises(
            OnePopulationContractError,
            match=re.escape(
                "A column one origin lacks has no absence kind: "
                f"{{'person.{_AUXILIARY}': "
            ),
        ):
            _absences(frame)
        declared = {_AUXILIARY: _DECISION}
        _require(population, frame, register=register, declared_absences=declared)
        record = _absences(frame, declared=declared)
        keyed = _keyed(record)
        assert ("person", _AUXILIARY, "owner-declared", "ACS") in keyed
        # The CPS null is no gap: no ACS row holds a value.
        assert ("person", _AUXILIARY, "owner-declared", "CPS") not in keyed
        assert record["unused_declarations"] == []
        assert record["needless_declarations"] == []

    def test_a_register_column_null_on_every_row_is_no_gap(
        self, population, assembled
    ) -> None:
        # The shared predicate's other half: with no value on a CPS row there
        # is no gap, so R10 and absences both pass it undeclared.
        register = {**US_STORED_NON_VARIABLE_COLUMNS, _AUXILIARY: "auxiliary"}
        index = assembled.table("person").index
        frame = _with_column(
            assembled, "person", _AUXILIARY, pd.Series(None, index=index, dtype=object)
        )
        _require(population, frame, register=register)
        record = _absences(frame)
        assert _AUXILIARY not in {column for (_, column, _, _) in _keyed(record)}

    def test_a_frame_with_no_acs_row_lists_every_declaration_as_unused(
        self, population
    ) -> None:
        empty_acs = {
            entity: population.acs_tables[entity].iloc[:0]
            for entity in US_SCHEMA.entities
        }
        frame = to_assembled_provenance(
            population.cps_tables,
            empty_acs,
            cps_household_weights=population.keyed_cps_household_weights,
            acs_household_weights={},
            staging_lineage={},
        )
        manifest = dataclasses.replace(
            selection_manifest(population),
            n_acs=0,
            selected_acs_keys_sha256=selection.keys_sha256(()),
        )
        declared = {
            column: f"{_DECISION} ({column})"
            for column in ("parent_2_id", "parent_1_id", _AUXILIARY)
        }
        _require(
            population,
            frame,
            manifest=manifest,
            staging_lineage={},
            declared_absences=declared,
        )
        record = _absences(frame, declared=declared)
        assert record["unused_declarations"] == [
            {"column": column, "decision": declared[column]}
            for column in sorted(declared)
        ]
        assert record["needless_declarations"] == []
        assert record["entries"] == []


# ------ lead call (c), 11-18: each public function's type and schema refusals


def _foreign_schema_frame() -> Frame:
    """A valid frame in a schema other than the US one."""

    from microcosm.frame import EntitySchema, Weights

    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [0, 1], "person_household_id": [1, 1]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
    )


def _rewrite_with(p, **overrides) -> Frame:
    kwargs = {
        "cps_tables": p.cps_tables,
        "acs_tables": p.acs_tables,
        "cps_household_weights": p.keyed_cps_household_weights,
        "acs_household_weights": p.keyed_acs_household_weights,
        "staging_lineage": staging_lineage(p),
        **overrides,
    }
    return to_assembled_provenance(
        kwargs.pop("cps_tables"), kwargs.pop("acs_tables"), **kwargs
    )


def _absences_with(f, **overrides) -> dict:
    kwargs = {
        "fill_manifest": fill_manifest(),
        "zero_written": {},
        "declared": {},
        **overrides,
    }
    return absences(kwargs.pop("frame", f), **kwargs)


_TYPE_REFUSALS = [
    (
        "rewrite cps tables",
        lambda p, f: _rewrite_with(p, cps_tables=list(p.cps_tables.values())),
        "CPS tables must be a mapping of entity to table, got list.",
    ),
    (
        "rewrite acs table",
        lambda p, f: _rewrite_with(
            p, acs_tables={**p.acs_tables, "person": p.acs_tables["person"].to_dict()}
        ),
        "ACS 'person' table must be a DataFrame, got dict.",
    ),
    (
        "rewrite staging lineage",
        lambda p, f: _rewrite_with(p, staging_lineage="2024HU0000001"),
        "staging_lineage must map each staged ACS person's native (household id, "
        "person id) to its (SERIALNO, SPORDER), got str.",
    ),
    (
        "require staging lineage",
        lambda p, f: _require(p, f, staging_lineage=5),
        "staging_lineage must map each staged ACS person's native (household id, "
        "person id) to its (SERIALNO, SPORDER), got int.",
    ),
    (
        # PR #1685 review, blocking 2: the old set of (SERIALNO, SPORDER)
        # pairs cannot tie a SERIALNO to its household, so it is refused.
        "rewrite staging lineage as a set of pairs",
        lambda p, f: _rewrite_with(
            p, staging_lineage=frozenset(staging_lineage(p).values())
        ),
        "staging_lineage must map each staged ACS person's native (household id, "
        "person id) to its (SERIALNO, SPORDER), got frozenset.",
    ),
    (
        "role record frame",
        lambda p, f: combined_spm_role_record(
            cps_role_record(f), staging_role_receipt(f), f.table("person")
        ),
        "frame must be a Frame, got DataFrame.",
    ),
    (
        "role record cps record",
        lambda p, f: combined_spm_role_record(
            list(cps_role_record(f)), staging_role_receipt(f), f
        ),
        "cps_record must be a mapping, got list.",
    ),
    (
        "role record receipt",
        lambda p, f: combined_spm_role_record(cps_role_record(f), None, f),
        "staging_receipt must be a mapping, got NoneType.",
    ),
    (
        "absences frame",
        lambda p, f: _absences_with(f, frame=f.table("person")),
        "frame must be a Frame, got DataFrame.",
    ),
    (
        "absences fill manifest",
        lambda p, f: _absences_with(f, fill_manifest=[]),
        "fill_manifest must be the reviewed default fill's manifest mapping, got list.",
    ),
    (
        "absences zero written",
        lambda p, f: _absences_with(f, zero_written=["tip_income"]),
        "zero_written must map each column to the receipt that records its written "
        "zeros, got list.",
    ),
    (
        "absences zero written receipt",
        lambda p, f: _absences_with(f, zero_written={"tip_income": 1}),
        "zero_written must map str columns to str receipts, got 'tip_income': 1.",
    ),
    (
        "absences declared",
        lambda p, f: _absences_with(f, declared=["parent_1_id"]),
        "declared must map each column to its owner's decision, got list.",
    ),
    (
        "absences declared decision",
        lambda p, f: _absences_with(f, declared={"parent_1_id": 1}),
        "declared must map str columns to str decisions, got 'parent_1_id': 1.",
    ),
    (
        "write absences",
        lambda p, f: write_absences(list(_absences(f)), Path("no-such-directory")),
        "record must be a mapping, got list.",
    ),
    (
        "cps unit ids",
        lambda p, f: cps_unit_ids(p.cps_tables["person"].to_dict()),
        "person must be a DataFrame, got dict.",
    ),
]


@pytest.mark.parametrize(
    ("case", "call", "message"),
    _TYPE_REFUSALS,
    ids=[case for case, _, _ in _TYPE_REFUSALS],
)
def test_each_public_function_refuses_a_wrong_type(
    population, assembled, case, call, message
) -> None:
    with pytest.raises(TypeError, match=re.escape(message)):
        call(population, assembled)


@pytest.mark.parametrize(
    ("case", "call", "message"),
    [
        (
            "require",
            lambda p: require_one_population(
                _foreign_schema_frame(),
                selection_manifest(p),
                parent_tail_donor_ids=frozenset(),
            ),
            "Incomplete assembled provenance: the frame must use the US schema.",
        ),
        (
            "role record",
            lambda p: combined_spm_role_record({}, {}, _foreign_schema_frame()),
            "The SPM role record needs a frame in the US schema.",
        ),
        (
            "absences",
            lambda p: _absences_with(None, frame=_foreign_schema_frame()),
            "The absences file needs a frame in the US schema.",
        ),
    ],
    ids=["require", "role record", "absences"],
)
def test_a_frame_in_another_schema_refuses(population, case, call, message) -> None:
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        call(population)


# ------------------------- phase 3: X2, N-2's kinds on a generated population


_AUXILIARY = "cps_employment_income_last_year"


def test_n2_kinds_on_a_generated_population(population, assembled) -> None:
    """A registered auxiliary null on every ACS row refuses until declared.

    The SSI take-up column (a Q1-exempt producer output) and ``SSI_VAL`` (an
    upper-case raw column), both null on every ACS row, pass the contract
    and appear in the absences file with their kinds; so do a reviewed
    default fill, a written zero and an ACS-native amount CPS rows lack
    (origin-native).
    """

    register = {
        **US_STORED_NON_VARIABLE_COLUMNS,
        _AUXILIARY: "registered auxiliary person column (#1575's shape)",
    }
    frame = _acs_null_on_cps(assembled, "person", _AUXILIARY, 41_000.0)
    person = frame.table("person")
    acs = _origin_mask(person, "person", "acs")
    for column in (_AUXILIARY, "takes_up_ssi_if_eligible", "SSI_VAL"):
        assert person.loc[acs, column].isna().all()
        assert person.loc[~acs, column].notna().all()

    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "A model-named column held on some CPS row is null on ACS rows without "
            f"a declared absence: {{'person.{_AUXILIARY}'"
        ),
    ):
        _require(population, frame, register=register)
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape("A column one origin lacks has no absence kind"),
    ):
        _absences(frame)

    declared = {_AUXILIARY: "US data owner: ACS rows carry no prior-year wage"}
    frame = _acs_only(frame, "person", "acs_social_security_income", 900.0)
    _require(population, frame, register=register, declared_absences=declared)
    record = _absences(
        frame,
        fills=[_acs_fill(frame, "tip_income")],
        zero_written={"employer_sponsored_insurance_premiums": "staging receipt"},
        declared=declared,
    )
    keyed = _keyed(record)
    kinds = {(column, kind) for (_, column, kind, origin) in keyed if origin == "ACS"}
    assert {
        (_AUXILIARY, "owner-declared"),
        ("takes_up_ssi_if_eligible", "release-written"),
        ("SSI_VAL", "raw source column"),
        ("tip_income", "reviewed default fill"),
        ("employer_sponsored_insurance_premiums", "zero written"),
    } <= kinds
    assert ("person", "acs_social_security_income", "origin-native", "CPS") in keyed
    assert {kind for (_, _, kind, _) in keyed} == set(ABSENCE_KINDS)
    for column in (_AUXILIARY, "takes_up_ssi_if_eligible", "SSI_VAL"):
        entry = next(
            entry
            for key, entry in keyed.items()
            if key[1] == column and key[3] == "ACS"
        )
        assert entry["rows"] == entry["origin_rows"] == int(acs.sum())
        assert entry["entity_weight_share"] == pytest.approx(
            _share(frame, "person", acs)
        )
    owner = keyed[("person", _AUXILIARY, "owner-declared", "ACS")]
    assert owner["detail"] == {"decision": declared[_AUXILIARY]}


# ------------- phase 3: X7, group quarters add nothing to household types


def _with_household_type_codes(frame: Frame) -> Frame:
    """CPS ``A_EXPRRP`` and ``A_MARITL`` on every person, from the roster.

    Household heads are 1 (with relatives) or 2; a role holder sharing the
    head's tax unit is the spouse (3); other role holders are housemates
    (13); persons without the role are children (5). Group-quarters persons
    get 14, as the ACS loader writes. Couples are married (1), everyone else
    never married (7).
    """

    tables = _tables(frame)
    person = tables["person"]
    household = tables["household"].set_index("household_id")
    kind = person["person_household_id"].map(household["TYPEHUGQ"])
    head = person["is_household_head"].to_numpy(dtype=bool)
    role = person[SPM_ROLE].to_numpy(dtype=bool)
    head_unit = person.loc[head].set_index("person_household_id")["person_tax_unit_id"]
    shares_unit = (
        person["person_household_id"].map(head_unit).to_numpy()
        == person["person_tax_unit_id"].to_numpy()
    )
    spouse = role & ~head & shares_unit
    child = ~role
    relatives = (
        pd.Series(spouse | child)
        .groupby(person["person_household_id"].to_numpy())
        .transform("any")
        .to_numpy()
    )
    relationship = np.select(
        [head & relatives, head, spouse, child], [1, 2, 3, 5], default=13
    )
    group_quarters = kind.isin([2, 3]).to_numpy()
    relationship[group_quarters] = 14
    couple = (
        pd.Series(spouse)
        .groupby(person["person_household_id"].to_numpy())
        .transform("any")
        .to_numpy()
    )
    person["A_EXPRRP"] = relationship.astype(np.int64)
    person["A_MARITL"] = np.where((head & couple) | spouse, 1, 7).astype(np.int64)
    return _rebuild(frame, tables)


_HOUSEHOLD_TYPES = (
    "all",
    "family",
    "married_couple_family",
    "other_family",
    "male_householder_no_spouse",
    "female_householder_no_spouse",
    "nonfamily",
    "householder_living_alone",
    "householder_not_living_alone",
)


def test_the_household_type_codes_give_one_householder_per_housing_unit(
    assembled,
) -> None:
    # Runs today, so X7's input is checked before #1610 can run on it.
    frame = _with_household_type_codes(assembled)
    person = frame.table("person")
    kind = person["person_household_id"].map(
        frame.table("household").set_index("household_id")["TYPEHUGQ"]
    )
    housing = ~kind.isin([2, 3]).to_numpy()
    relationship = person["A_EXPRRP"].to_numpy()
    householders = (
        pd.Series(np.isin(relationship, [1, 2]) & housing)
        .groupby(person["person_household_id"].to_numpy())
        .sum()
    )
    housing_households = np.unique(person["person_household_id"].to_numpy()[housing])
    assert (householders.loc[housing_households] == 1).all()
    assert (relationship[~housing] == 14).all()
    assert set(relationship.tolist()) <= {1, 2, 3, 5, 13, 14}
    assert set(person["A_MARITL"].tolist()) <= {1, 7}
    assert (person.loc[relationship == 3, "A_MARITL"] == 1).all()


def test_group_quarters_add_nothing_to_household_type_targets(assembled) -> None:
    """Unexecuted until PolicyEngine/microcosm#1610 lands its materializer."""

    targets = pytest.importorskip(
        "microcosm.build.us_runtime.household_type_targets",
        reason=(
            "PolicyEngine/microcosm#1610's household-type materializer "
            "(household_type_household_values) is not on main yet"
        ),
    )
    import warnings

    frame = _with_household_type_codes(assembled)
    household = frame.table("household")
    group_quarters = household["TYPEHUGQ"].isin([2, 3]).to_numpy()
    assert group_quarters.sum() == 2
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", targets.HouseholdUniverseFallbackWarning)
        for household_type in _HOUSEHOLD_TYPES:
            for own_children in (
                "all",
                "with_own_children_under_18",
                "no_own_children_under_18",
            ):
                values = targets.household_type_household_values(
                    frame,
                    {"household_type": household_type, "own_children": own_children},
                )
                assert len(values) == len(household)
                assert (values[group_quarters] == 0).all()
                assert float((weights * values)[group_quarters].sum()) == 0.0
        housing = targets.household_type_household_values(
            frame, {"household_type": "all"}
        )
    assert (housing[~group_quarters] == 1).all()


# ------------- phase 4: X5, the whole-frame differential (BP, K-21, Q4)

#: The fields each moved reader's statistics change in (d2283 Q4: CPS clones
#: keep Route A's ``puf_tax_detail`` label once assembled), keyed by reader,
#: and, keyed by defect, the detail fields each clone-only defect of
#: :data:`READER_CLONE_DEFECTS` moves besides the verdict (lead call L7).
#: PR 3 owns every entry.
_X5_MOVED_FIELDS: dict[str, frozenset[str]] = {
    "retirement_contributions": frozenset(
        {
            "allocation_mismatch_rows",
            "source_positive_rows",
            "source_reconciliation_rows",
            "source_rows",
            "source_total",
        }
    ),
}

_X5_PR3_REASONS: dict[str, str] = {
    "retirement_contributions": (
        "d2283 PR 3 flips this. Retirement contributions: once assembled, the "
        "source and reconciliation masks keep only stored-asec rows "
        "(retirement_contributions.py:172-193) where Route A takes every row, "
        "so five source fields move; the verdict does not."
    ),
}

#: Runtime modules that test the assembled discriminator only as writers or
#: provenance owners, not as gate readers.
_X5_WRITERS = frozenset(
    {
        "__init__",
        "one_population_contract",
        "puf_support",
        "spine_agreement",
        "spine_assembly",
        "stacked_spine",
        "support_provenance",
    }
)


@pytest.fixture(scope="module")
def readers():
    # pytest counts a fixture's error as a marked test's expected failure
    # when its type matches ``raises``; the builder's self-checks raise
    # AssertionError, so re-raise them as an error the strict xfails
    # (raises=AssertionError) report rather than absorb.
    try:
        return reader_population()
    except AssertionError as exc:
        raise RuntimeError(f"The X5 reader frame is broken: {exc}") from exc


@pytest.fixture(scope="module")
def defect_readers():
    # As ``readers``: a broken defect frame is an error, never an xfail.
    try:
        return {
            defect: reader_population(clone_defect=defect)
            for defect in READER_CLONE_DEFECTS
        }
    except AssertionError as exc:
        raise RuntimeError(f"An X5 clone-defect frame is broken: {exc}") from exc


def _x5_statistics(result: object) -> object:
    """A reader's verdict and summary statistics, as plain data."""

    if isinstance(result, GateResult):
        return {
            "name": result.name,
            "passed": result.passed,
            "failures": list(result.failures),
            "details": result.details,
        }
    if isinstance(result, frozenset):
        return sorted(result)
    raise TypeError(f"Unexpected reader result {type(result).__name__}.")


def _x5_differences(route_a: object, assembled: object, path=()) -> list[tuple]:
    """Every path at which two readers' statistics differ.

    Exact: both frames hold the same person rows in the same order with the
    same weights (see the fixture test), so equal statistics are the same
    sums, bit for bit. NaN equals NaN.
    """

    if isinstance(route_a, dict) and isinstance(assembled, dict):
        differences: list[tuple] = []
        for key in sorted(set(route_a) | set(assembled), key=str):
            if key not in route_a or key not in assembled:
                differences.append((*path, key))
            else:
                differences.extend(
                    _x5_differences(route_a[key], assembled[key], (*path, key))
                )
        return differences
    if isinstance(route_a, (list, tuple)) and isinstance(assembled, (list, tuple)):
        if len(route_a) != len(assembled):
            return [path]
        differences = []
        for index, (left, right) in enumerate(zip(route_a, assembled, strict=True)):
            differences.extend(_x5_differences(left, right, (*path, index)))
        return differences
    if (
        isinstance(route_a, float)
        and isinstance(assembled, float)
        and math.isnan(route_a)
        and math.isnan(assembled)
    ):
        return []
    return [] if route_a == assembled else [path]


def _x5_run(readers, reader: str) -> tuple[object, object]:
    gate = READER_GATES[reader]
    return (
        _x5_statistics(gate(readers.route_a)),
        _x5_statistics(gate(readers.assembled)),
    )


def _x5_runtime_sources() -> dict[str, str]:
    """Map each top-level ``us_runtime`` module's stem to its source."""

    runtime = Path(inspect.getfile(contract)).parent
    return {
        path.stem: path.read_text(encoding="utf-8") for path in runtime.glob("*.py")
    }


def test_x5_covers_every_assembled_form_reader() -> None:
    """The readers are F-19's: every runtime module that tests the assembled
    discriminator itself, less writers and owners, plus every caller of the
    physical-channel accessor.

    ``acs_local_hours`` is the one such module with no release gate; its
    case is the donor-prep refusal below.
    """

    sources = _x5_runtime_sources()
    discriminator = re.compile(
        r"has_assembled_support_metadata|spine_source_id_column\("
    )
    testing = {name for name, text in sources.items() if discriminator.search(text)}
    callers = {
        name
        for name, text in sources.items()
        if "support_gate_source_channel_series(" in text
    } - {"support_provenance"}
    modules = {gate.__module__.rsplit(".", 1)[1] for gate in READER_GATES.values()}
    assert testing - _X5_WRITERS == (modules - {"ssi_take_up"}) | {"acs_local_hours"}
    assert callers <= modules
    assert set(READER_SOURCE_COLUMNS) == set(READER_GATES)
    assert set(_X5_MOVED_FIELDS) == set(_X5_PR3_REASONS)
    assert set(READER_GATES).isdisjoint(READER_CLONE_DEFECTS)
    assert set(_X5_MOVED_FIELDS) <= set(READER_GATES) | set(READER_CLONE_DEFECTS)
    assert {reader for reader, _ in READER_CLONE_DEFECTS.values()} <= set(READER_GATES)


#: A name at the start of a line, an optional annotation, ``=`` and an ACS
#: support label alone: ``acs`` (the stacked pool's) or ``acs_2024_1yr``
#: (staging's, which a d2283 child's ACS rows carry). Every part of the
#: match stays on one line.
_X5_ACS_LABEL_LITERAL = re.compile(
    r'^_?[A-Za-z_0-9]+[ \t]*(:[^=\n]+)?=[ \t]*"(acs|acs_2024_1yr)"[ \t]*$',
    re.MULTILINE,
)

#: A call of the ACS PUMS earnings-universe resolver, which keeps rows whose
#: stored channel is ``acs``.
_X5_ACS_UNIVERSE_CALL = re.compile(
    r"resolve_acs_pums_earnings_universe\(|apply_acs_pums_earnings_universe_zeros\("
)

#: Every runtime module that names an ACS support label or calls the ACS
#: earnings-universe resolver, with what it does with the label (d2283 lead
#: call P3-L7). A module that keeps rows whose stored channel is ``acs``
#: sees no ACS row of a d2283 child, whose ACS rows carry ``acs_2024_1yr``.
_X5_ACS_LABEL_MODULES: dict[str, str] = {
    "acs_income_universe": (
        "Owns the ACS earnings universe: _ACS_SUPPORT_CHANNEL is acs and the "
        "resolver keeps stored-acs rows only, so a d2283 child's ACS rows fall "
        "outside it; the QBI gate reads it through qbi_inputs."
    ),
    "acs_pums": (
        "Staging's spine tag, acs_2024_1yr, which a d2283 child's ACS rows "
        "carry; test_constants_match_staging_and_provenance pins it to the "
        "contract's."
    ),
    "acs_release_predictors": (
        "Fills CPS-named predictors on the stacked pool's stored-acs rows at "
        "release time. It passes a frame with no such row through unchanged, "
        "or refuses it when the caller passes any archive option."
    ),
    "one_population_contract": (
        "The contract's ACS_SUPPORT_CHANNEL, acs_2024_1yr, the label it gives ACS rows."
    ),
    "puf_support": (
        "Calls the resolver to scope its PUF recipient predictors' source universe."
    ),
    "qbi_inputs": (
        "The QBI gate's reconciliation scope calls the resolver, whose universe "
        "holds stored-acs rows only, so a d2283 child's ACS rows are outside it."
    ),
    "spm_universe_source": (
        "ACS_ARM, acs, names the producer's ACS universe arm, not a stored "
        "channel; SPM_UNIVERSE_CHANNEL_ARMS maps both acs_2024_1yr and acs "
        "to it."
    ),
    "stacked_spine": (
        "The stacked pool's ACS channel, acs, and a resolver caller on the "
        "stacked pool."
    ),
    "weeks_unemployed": (
        "_ACS_NATIVE_CHANNEL, acs_2024_1yr, names the channel the weeks gate "
        "fails closed with one line until a reviewed rule lands; "
        "test_weeks_acs_channel_label_matches_the_contract pins it to the "
        "contract's."
    ),
}


def test_x5_names_every_acs_label_reader() -> None:
    """This test pins every runtime module that names an ACS support label,
    with a reason for each (d2283 lead call P3-L7). It reads the sources
    X5's reader census reads.

    A module is in scope if one whole line of it is a name at the line's
    start, an optional type annotation, ``=`` and the literal ``"acs"`` or
    ``"acs_2024_1yr"``, or if it calls ``resolve_acs_pums_earnings_universe``
    or ``apply_acs_pums_earnings_universe_zeros``. Add a new label constant
    or resolver caller here with its reason, and remove the row of a module
    that drops its label.

    The rule sees a module-level assignment of one literal and a resolver
    call, nothing else. It misses a label in a tuple or a dict, an
    assignment inside a class or function body, an inline comparison, and
    an assignment with a comment after it on the same line. Modules of that
    shape today, none of them a gate reader: ``h5_io``
    (``_STACKED_SURVEY_CHANNELS``), ``spm_universe_source``
    (``SPM_UNIVERSE_CHANNEL_ARMS``), ``us_late_overlap_ownership`` (an
    ``origin`` loop) and ``multispine_pool`` (``POOL_HOUSEHOLD_MASS_SHARES``
    and the spines it assembles). The scan reads no subpackage of
    ``us_runtime``.
    """

    sources = _x5_runtime_sources()
    labelled = {
        name
        for name, text in sources.items()
        if _X5_ACS_LABEL_LITERAL.search(text) or _X5_ACS_UNIVERSE_CALL.search(text)
    }
    assert labelled == set(_X5_ACS_LABEL_MODULES)
    assert all(reason.strip() for reason in _X5_ACS_LABEL_MODULES.values())


def test_the_reader_frame_exercises_every_reader(readers) -> None:
    """X5's preconditions, outside every expected failure.

    A strict xfail cannot pass for want of data, and an equal pair of
    statistics is not two failing gates: the frame has clone rows (indices
    0, 1 and 2) on every entity and each reader's raw source columns; every
    Route A gate passes and takes its source branch; clone rows differ from
    the source wherever Route A does not reconcile them, so a reader whose
    assembled mask took clone rows would show; and the WIC gate's two
    clone keys (the source triple on Route A, ``person_source_id`` once
    assembled) partition the rows alike.
    """

    route_a, assembled = readers.route_a, readers.assembled
    person = route_a.table("person")
    for entity in US_SCHEMA.entities:
        table = route_a.table(entity)
        assert set(table[support_clone_index_column(entity)]) == {0, 1, 2}, entity
        assert spine_source_id_column(entity) not in table
        assert spine_source_id_column(entity) in assembled.table(entity)
    for reader, columns in READER_SOURCE_COLUMNS.items():
        assert set(columns) <= set(person.columns), reader
        assert set(columns) <= set(assembled.table("person").columns), reader
    assert (
        person["person_id"].tolist() == assembled.table("person")["person_id"].tolist()
    )
    np.testing.assert_array_equal(
        route_a.resolve_weights("person").values,
        assembled.resolve_weights("person").values,
    )

    results = {name: gate(route_a) for name, gate in READER_GATES.items()}
    for name, result in results.items():
        if isinstance(result, GateResult):
            assert result.passed, (name, result.failures)
    assert results["ssi_take_up_reporters"] == readers.reporters
    assert readers.clone_only_ssi not in results["ssi_take_up_reporters"]
    clone = person["person_support_clone_index"].to_numpy() > 0
    assert int(results["alimony"].details["asec_source_rows"]) == int((~clone).sum())
    assert "source_rows" in results["workers_compensation"].details
    for name in (
        "medicare_take_up",
        "retirement_contributions",
        "retirement_distributions",
    ):
        assert int(results[name].details["source_rows"]) == len(person), name
    contributions = results["retirement_contributions"].details
    assert int(contributions["source_reconciliation_rows"]) == len(person)
    contributing = person["RETCB_VAL"].to_numpy() > 0
    assert int(contributions["source_positive_rows"]) == int(contributing.sum())
    assert int(contributions["allocation_mismatch_rows"]) > 0
    weeks = results["weeks_unemployed"].details
    assert not weeks["source_missing"]
    assert int(weeks["source_reconciliation_rows"]) == int((~clone).sum())
    assert int(weeks["uc_constraint_rows"]) == int(clone.sum())
    assert int(results["wic_claim"].details["clone_group_count"]) == int((~clone).sum())

    def differs(column: str, expected) -> bool:
        return bool((clone & (person[column].to_numpy() != expected)).any())

    oi_off = person["OI_OFF"].to_numpy()
    oi_val = person["OI_VAL"].to_numpy(dtype=np.float64)
    assert differs("workers_compensation", person["WC_VAL"].to_numpy())
    assert differs("alimony_income", np.where(oi_off == 20, oi_val, 0.0))
    lkweeks = person["LKWEEKS"].to_numpy()
    assert differs("weeks_unemployed", np.where(lkweeks == -1, 0, lkweeks))
    weeks_positive = person["weeks_unemployed"].to_numpy() > 0
    uc_positive = person["unemployment_compensation"].to_numpy() > 0
    assert (clone & weeks_positive).any()
    assert not (clone & weeks_positive & ~uc_positive).any()
    native_values = person.loc[~clone].set_index("person_source_id")
    for column in (
        "traditional_401k_contributions_desired",
        "taxable_401k_distributions",
    ):
        expected = person["person_source_id"].map(native_values[column]).to_numpy()
        assert differs(column, expected), column
    ssi_clone = clone & (
        person["person_source_id"].astype(str).eq(readers.clone_only_ssi).to_numpy()
    )
    assert (person.loc[ssi_clone, "SSI_VAL"] > 0).all() and ssi_clone.any()

    triple = person[list(WIC_SOURCE_TRIPLE)].astype(str).agg(":".join, axis=1)
    source = person["person_source_id"]
    assert triple.groupby(source).nunique().max() == 1
    assert source.groupby(triple).nunique().max() == 1


def test_the_mixed_origin_reader_frame_holds_both_origins(readers) -> None:
    """The mixed-origin reader frame, d2283 PR 3's fixture for the four gates
    it changes, keeps the reader frame's CPS rows and adds ACS rows.

    Its builder shares the Route A half with ``reader_population``. Every
    entity stores the channels ``asec``, ``puf_tax_detail`` and
    ``acs_2024_1yr`` and no other, every ACS row has clone index 0, and the
    assembly record counts each entity's ACS rows. Each entity's CPS rows
    equal the ASEC-only assembled frame's, in order, over that frame's
    columns, and keep their weights; each ACS household keeps the weight the
    generator drew for its ``SERIALNO``. The join adds ACS-only columns,
    null on CPS rows, and widens a column the ACS rows lack (an integer
    ``MCARE`` becomes float), so the test compares values and ignores
    dtypes. On ACS
    rows the four gates' raw ASEC columns hold only nulls,
    ``weeks_unemployed`` holds 0, unemployment compensation stays finite and
    turns positive on some rows, Medicare take-up keeps a boolean dtype, and
    every retirement output holds 0.
    """

    population = mixed_origin_reader_population()
    for form in ("route_a", "assembled"):
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(
                getattr(population, form).table(entity),
                getattr(readers, form).table(entity),
            )
    mixed_origin, assembled = population.mixed_origin, population.assembled
    record = mixed_origin.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]["native_row_counts"]
    acs_rows: dict[str, np.ndarray] = {}
    for entity in US_SCHEMA.entities:
        table = mixed_origin.table(entity)
        channel = table[support_channel_column(entity)]
        assert set(channel) == {
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ACS_SUPPORT_CHANNEL,
        }, entity
        acs = channel.eq(ACS_SUPPORT_CHANNEL).to_numpy()
        acs_rows[entity] = acs
        assert set(table.loc[acs, support_clone_index_column(entity)]) == {0}, entity
        assert record[entity][ACS_SUPPORT_CHANNEL] == int(acs.sum()), entity
        expected = assembled.table(entity)
        pd.testing.assert_frame_equal(
            table.loc[~acs, list(expected.columns)].reset_index(drop=True),
            expected,
            check_dtype=False,
            check_exact=True,
        )
    assert int(acs_rows["household"].sum()) == 12
    np.testing.assert_array_equal(
        mixed_origin.resolve_weights("person").values[~acs_rows["person"]],
        assembled.resolve_weights("person").values,
    )
    household = mixed_origin.table("household")
    acs_households = acs_rows["household"]
    assert {
        str(serial): float(weight)
        for serial, weight in zip(
            household.loc[acs_households, "SERIALNO"],
            mixed_origin.weights_for("household").values[acs_households],
            strict=True,
        )
    } == population.acs_household_weights

    person = mixed_origin.table("person")
    acs_person = person.loc[acs_rows["person"]]
    raw = {
        column
        for reader in (
            "medicare_take_up",
            "retirement_contributions",
            "retirement_distributions",
            "weeks_unemployed",
        )
        for column in READER_SOURCE_COLUMNS[reader]
    } - {"unemployment_compensation"}
    assert {"LKWEEKS", "MCARE", "RETCB_VAL", "WSAL_VAL", "SEMP_VAL"} <= raw
    for column in sorted(raw):
        assert acs_person[column].isna().all(), column
    assert (acs_person["weeks_unemployed"] == 0.0).all()
    uc = acs_person["unemployment_compensation"].to_numpy(dtype=np.float64)
    assert np.isfinite(uc).all() and (uc > 0.0).any() and (uc == 0.0).any()
    assert pd.api.types.is_bool_dtype(person["takes_up_medicare_if_eligible"])
    for column in (
        *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
        *US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    ):
        assert (acs_person[column] == 0.0).all(), column


def test_the_mixed_origin_builder_follows_its_seed_and_household_count() -> None:
    """``seed`` draws the mixed-origin frame's ACS households and
    ``n_acs_households`` sets how many it joins; a count below 1 raises."""

    with pytest.raises(ValueError, match="at least 1"):
        mixed_origin_reader_population(n_acs_households=0)
    acs_people: dict[int, pd.DataFrame] = {}
    for seed in (0, 1):
        frame = mixed_origin_reader_population(seed, n_acs_households=3).mixed_origin
        channel = frame.table("household")[support_channel_column("household")]
        assert int(channel.eq(ACS_SUPPORT_CHANNEL).sum()) == 3, seed
        person = frame.table("person")
        acs = person[support_channel_column("person")].eq(ACS_SUPPORT_CHANNEL)
        acs_people[seed] = person.loc[acs].reset_index(drop=True)
    assert not acs_people[0].equals(acs_people[1])


_X5_CASES = [
    pytest.param(
        name,
        id=name,
        marks=pytest.mark.xfail(
            strict=True, raises=AssertionError, reason=_X5_PR3_REASONS[name]
        ),
    )
    if name in _X5_PR3_REASONS
    else pytest.param(name, id=name)
    for name in sorted(READER_GATES)
]


@pytest.mark.parametrize("reader", _X5_CASES)
def test_assembled_readers_return_route_a_statistics(readers, reader) -> None:
    """X5 (BP): on an ASEC-only population every provenance-reading gate
    returns Route A's verdict and statistics after the rewrite.

    Run once on the Route A frame and once on its
    ``to_assembled_provenance`` with no ACS row. Each reader, or clone-only
    defect below, whose assembled branch moves under Route A's clone label
    carries a strict expected failure naming PR 3; the WIC frame's two
    clone partitions agree
    (see the fixture test), so the WIC case compares one partition.
    """

    route_a, assembled = _x5_run(readers, reader)
    assert _x5_differences(route_a, assembled) == []


@pytest.mark.parametrize("reader", sorted(set(_X5_MOVED_FIELDS) & set(READER_GATES)))
def test_a_moved_reader_moves_only_its_named_fields(readers, reader) -> None:
    """Outside the expected failure, a moved reader keeps its verdict and
    every statistic but the fields named for it. The fixture test checks
    that Route A's ``source_rows`` (contributions and distributions),
    ``source_reconciliation_rows`` (contributions) and
    ``uc_constraint_rows`` (weeks) take the clone rows."""

    route_a, assembled = _x5_run(readers, reader)
    moved = {("details", field) for field in _X5_MOVED_FIELDS[reader]}
    assert set(_x5_differences(route_a, assembled)) <= moved
    assert route_a["passed"] and assembled["passed"]


def test_each_clone_defect_changes_one_clone_cell(readers, defect_readers) -> None:
    """L7's preconditions, outside the expected failures: each defect frame
    differs from the reader frame in one cell, on a clone row, on both
    forms; the builder has already checked that Route A's gate for the
    defect's reader fails with exactly its one failure and every other
    reader passes. Route A's gate counts that one mismatch, over every clone
    row (weeks) or every row (Medicare)."""

    for defect, population in defect_readers.items():
        reader, failure = READER_CLONE_DEFECTS[defect]
        for form in ("route_a", "assembled"):
            base = getattr(readers, form).table("person")
            changed = getattr(population, form).table("person")
            assert list(base.columns) == list(changed.columns), (defect, form)
            differs = [
                (row, column)
                for column in base.columns
                for row in np.flatnonzero(
                    (base[column] != changed[column]).to_numpy()
                    & ~(base[column].isna() & changed[column].isna()).to_numpy()
                )
            ]
            assert len(differs) == 1, (defect, form, differs)
            row, column = differs[0]
            assert base["person_support_clone_index"].iloc[row] > 0, (defect, form)
            assert column in READER_SOURCE_COLUMNS[reader], (defect, column)
        result = READER_GATES[reader](population.route_a)
        assert list(result.failures) == [failure], defect
        person = population.route_a.table("person")
        clones = int((person["person_support_clone_index"] > 0).sum())
        if reader == "weeks_unemployed":
            assert result.details["uc_constraint_mismatch_count"] == 1, defect
            assert result.details["uc_constraint_rows"] == clones, defect
        else:
            assert result.details["source_mismatch_count"] == 1, defect
            assert result.details["source_rows"] == len(person), defect


_X5_DEFECT_CASES = [
    pytest.param(
        defect,
        id=defect,
        marks=pytest.mark.xfail(
            strict=True, raises=AssertionError, reason=_X5_PR3_REASONS[defect]
        ),
    )
    if defect in _X5_PR3_REASONS
    else pytest.param(defect, id=defect)
    for defect in sorted(READER_CLONE_DEFECTS)
]


@pytest.mark.parametrize("defect", _X5_DEFECT_CASES)
def test_a_clone_only_defect_keeps_route_a_verdict(defect_readers, defect) -> None:
    """X5 on verdicts (lead call L7): a defect on one clone row fails Route
    A's gate, so the assembled gate must fail with the same failures.

    The defect frame's builder and
    ``test_each_clone_defect_changes_one_clone_cell`` check that Route A's
    gate fails with the defect's one failure, so a defect that did nothing
    fails there rather than passing here.
    """

    route_a, assembled = _x5_run(
        defect_readers[defect], READER_CLONE_DEFECTS[defect][0]
    )
    assert (assembled["passed"], assembled["failures"]) == (
        route_a["passed"],
        route_a["failures"],
    )


@pytest.mark.parametrize(
    "defect", sorted(set(_X5_MOVED_FIELDS) & set(READER_CLONE_DEFECTS))
)
def test_a_clone_only_defect_moves_only_its_named_fields(
    defect_readers, defect
) -> None:
    """Outside the expected failure, a defect moves only the verdict and its
    named fields. ``test_each_clone_defect_changes_one_clone_cell`` checks
    Route A's count of the one clone mismatch."""

    reader, failure = READER_CLONE_DEFECTS[defect]
    route_a, assembled = _x5_run(defect_readers[defect], reader)
    moved = {("passed",), ("failures",)} | {
        ("details", field) for field in _X5_MOVED_FIELDS[defect]
    }
    assert set(_x5_differences(route_a, assembled)) <= moved
    assert not route_a["passed"]
    assert route_a["failures"] == [failure]
    # The assembled gate fails for no other reason (true before and after).
    assert set(assembled["failures"]) <= {failure}


def test_the_local_hours_donor_refuses_the_assembled_population(readers) -> None:
    """X5 leaves ``acs_local_hours`` out: it has no release gate, and its
    donor prep refuses an assembled frame (``acs_local_hours.py:227-237``).

    The Route A frame passes that check and stops later, at the raw hours
    columns this fixture does not carry.
    """

    with pytest.raises(ValueError, match="explicit legacy ASEC observation roles"):
        prepare_acs_local_hours_donor(readers.assembled, seed=0, period=2024)
    with pytest.raises(ValueError, match="requires complete raw ASEC"):
        prepare_acs_local_hours_donor(readers.route_a, seed=0, period=2024)
