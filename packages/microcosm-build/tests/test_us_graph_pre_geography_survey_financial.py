"""Genuine source/clone and declaration checks before financial fitting.

These tests do not execute the new financial runner successfully, fit models,
transfer PUF values, assign geography, or establish release acceptance. The only
successful graph executions here are the existing four-node survey clone prefix.
"""

import inspect
import shutil
from dataclasses import replace

import numpy as np
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_current_property_income_sources import source_arguments

from microcosm.build.us_runtime import graph_atomic_survey_financial as host
from microcosm.build.us_runtime import graph_current_survey_property as property_graph
from microcosm.build.us_runtime import survey_financial_successor as budget_bridge
from microcosm.frame import WeightKind
from microcosm.graph import compile_graph


class _RealDiskPatch:
    """Retain fixture literals while refusing its obsolete fake resource probe."""

    def __init__(self, patch):
        self.patch = patch
        self.real = shutil.disk_usage
        self.suppressed = 0

    def setattr(self, target, name, value, *args, **kwargs):
        if target is shutil and name == "disk_usage":
            assert shutil.disk_usage is self.real
            self.suppressed += 1
            return
        self.patch.setattr(target, name, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.patch, name)


def _pins(prefix):
    result = {}
    for edge in host._prefix_host_edges(host._PRE_GEOGRAPHY):
        record = prefix.manifest.node(edge.producer)
        key = record.opaque_artifacts[edge.artifact]
        result[edge.name] = {
            "producer_key": record.key,
            "artifact_key": key,
            "payload_sha256": host.codec.sha(prefix.store.load_bytes(key)),
        }
    return result


def _check_prefix(prefix):
    expected, receipts = host._pre_geography_prefix_expectations(prefix)
    clone, claim = (
        host.atomic.clone.COMBINED_CLONE_NODE,
        host.atomic.clone.COMBINED_CLONE_CLAIM_NODE,
    )
    assert set(expected) == {
        host.survey.CREATE_NODE,
        host.survey.ALLOCATION_NODE,
        clone,
        claim,
    }
    assert set(receipts) == {clone, claim}
    assert expected[clone].version == expected[claim].version == clone
    assert expected[clone].owners != expected[claim].owners
    for output in prefix.compiled.graph.node(claim).outputs:
        cell = output.entity, output.column
        assert expected[clone].owners[cell] == clone
        assert expected[claim].owners[cell] == claim
    # Match independently reconstructed pre/post-claim states and clone receipts
    # against the genuine executor manifest, without fitting a financial model.
    _, source_keys = host._source_paths_and_keys(
        prefix.compiled, prefix.sources, prefix.store
    )
    states = host.atomic._states(
        prefix.compiled,
        prefix.kernels,
        source_keys,
        expected,
        {
            **{name: row.receipt for name, row in prefix.manifest.nodes.items()},
            **receipts,
        },
    )
    host.survey._check_node_states(prefix.manifest, states)
    host.atomic.same_replayed_population(expected[claim], prefix.clone_population)
    assert prefix.clone_population.mass_ledger == expected[claim].mass_ledger
    assert len(prefix.clone_population.mass_ledger) == 2
    assert not hasattr(prefix, "geography_config")
    assert host.financial._geography_edge().producer not in prefix.compiled.order
    assert set(prefix.sources) == {host.survey.SOURCE_NAME}
    assert "census_block_geoid" not in prefix.clone_population.frame.table("household")
    assert host._prefix_names(host._PRE_GEOGRAPHY) == (
        "allocated_population",
        "clone_population",
    )
    return expected


def test_genuine_pre_geography_clone_predictor_property_boundary(tmp_path, monkeypatch):
    genuine = _RealDiskPatch(monkeypatch)
    arguments = source_arguments(tmp_path, genuine)
    assert genuine.suppressed > 0 and shutil.disk_usage is genuine.real
    with host.values.source.verification_epoch() as epoch:
        cold = host.survey.run_authenticated_survey_population(
            **arguments, clones=True, return_values=True, resume="auto"
        )
        warm = host.survey.run_authenticated_survey_population(
            **arguments, clones=True, return_values=True, resume="require"
        )
        assert len(cold.compiled.order) == len(warm.compiled.order) == 4
        assert cold.manifest.key == warm.manifest.key
        assert all(record.hit for record in warm.manifest.nodes.values())
        expected = _check_prefix(cold)
        _check_prefix(warm)
        host.atomic.same_replayed_population(
            cold.clone_population, warm.clone_population
        )
        retained = host.reconstruction._population_stamp(cold.clone_population)
        for entity in cold.allocated_population.frame.entities:
            assert cold.clone_population.frame.n(
                entity
            ) == 2 * cold.allocated_population.frame.n(entity)
        # Existing independent clone checking verifies lineage, groups, exact
        # weight halves, every inherited cell, DESIGN anchors and both owners.
        predictors = host.values.qualify_current_survey_predictors(
            cold.preparation,
            cold.allocated_population,
            cold.clone_population,
            geography_config=None,
        )
        properties = property_graph.sources.qualify_current_property_income_sources(
            cold.preparation,
            cold.allocated_population,
            cold.clone_population,
            geography_config=None,
        )
        assert (
            predictors.geography_config_payload
            is predictors.geography_validation
            is None
        )
        assert "atomic_geography" not in predictors.evidence
        assert (
            predictors.donor_frame.resolve_weights("person").kind is WeightKind.DESIGN
        )
        assert properties.evidence["donor_weight_kind"] == "original_household_design"
        assert properties.evidence["model_fitted"] is False
        assert properties.evidence["clone_attachment_performed"] is False
        assert properties.shared_predictors.projection == predictors.projection
        source = cold.preparation._checked()[2].frame
        channel = host.values.provenance.support_channel_column("person")
        donor = source.select(source.person[channel].eq("asec").to_numpy())
        np.testing.assert_array_equal(
            predictors.donor_frame.resolve_weights("person").values,
            donor.resolve_weights("person").values,
        )
        pins = _pins(cold)
        options = property_graph.PropertyIncomeOptions(
            scales=(1.0, 1.0, 1.0, 1.0), atol=1e-8, rtol=1e-8, n_estimators=2
        )
        predictors_nodes = host.financial.current_survey_predictor_nodes(
            predictors, cold.clone_population.frame, host_pins=pins, n_estimators=2
        )
        property_nodes = property_graph.current_survey_property_nodes(
            properties, cold.clone_population.frame, host_pins=pins, options=options
        )
        compiled = compile_graph(
            replace(
                cold.compiled.graph,
                nodes=(
                    *cold.compiled.graph.nodes,
                    *predictors_nodes,
                    *property_nodes,
                ),
            )
        )
        assert len(compiled.order) == 30  # 4 prefix + 10 predictors + 16 property.
        gate = host.financial._geography_edge()
        assert gate.producer not in compiled.order
        assert all(gate not in node.artifact_inputs for node in compiled.graph.nodes)
        assert set(pins) == {"preparation", "allocation", "frame_context"}
        # The descriptive prefix cannot smuggle later nodes into its four-node
        # declaration, even when those nodes are independently valid financial
        # declarations. Nor can it omit the live clone or substitute allocation.
        with pytest.raises(ValueError, match="PRE_GEOGRAPHY_PREFIX_DECLARATION$"):
            host._pre_geography_prefix_expectations(replace(cold, compiled=compiled))
        with pytest.raises(ValueError, match="LIVE_POPULATION_REQUIRED$"):
            host._pre_geography_prefix_expectations(
                replace(cold, clone_population=None)
            )
        changed_allocation = host.reconstruction._copy_population(
            cold.allocated_population
        )
        changed_allocation.frame.person.loc[
            changed_allocation.frame.person.index[0], "age"
        ] += 1
        with pytest.raises(ValueError, match="MATERIALIZED_FRAME_VALUES$"):
            host._pre_geography_prefix_expectations(
                replace(cold, allocated_population=changed_allocation)
            )
        assert all(
            node.params.get("release_eligible") is not True
            for node in compiled.graph.nodes
        )
        assert host.reconstruction._population_stamp(cold.clone_population) == retained

        # Negative inputs are detached from the genuine retained source. A
        # post-geography/altered population must not pass as the raw clone.
        for defect, reason in (
            ("amount", "CLONE_FRAME_VALUES$"),
            ("weight", "CLONE_WEIGHTS$"),
            ("clone_index", "CLONE_FRAME_VALUES$"),
            ("assigned_column", "CLONE_FRAME_VALUES$"),
            ("owners", "POPULATION_OWNERS$"),
        ):
            changed = host.reconstruction._copy_population(cold.clone_population)
            if defect == "amount":
                changed.frame.person.loc[changed.frame.person.index[0], "age"] += 1
            elif defect == "weight":
                changed.frame.weights_for("household").values.setflags(write=True)
                changed.frame.weights_for("household").values[0] += 1
            elif defect == "clone_index":
                name = host.values.provenance.support_clone_index_column("person")
                changed.frame.person.loc[changed.frame.person.index[0], name] = 2
            elif defect == "assigned_column":
                changed.frame.table("household")["census_block_geoid"] = (
                    "060010001001001"
                )
            else:
                owners = dict(changed.owners)
                owners[next(iter(owners))] = "invented.wrong-owner"
                changed = replace(changed, owners=owners)
            with pytest.raises(ValueError, match=reason):
                host._pre_geography_prefix_expectations(
                    replace(cold, clone_population=changed)
                )
        copied = object.__new__(type(cold.preparation))
        object.__setattr__(copied, "payload", cold.preparation.payload)
        with pytest.raises(ValueError, match="UNISSUED_OR_CHANGED"):
            host._pre_geography_prefix_expectations(replace(cold, preparation=copied))
        assert host.reconstruction._population_stamp(cold.clone_population) == retained
        host.atomic.same_replayed_population(
            expected[host.atomic.clone.COMBINED_CLONE_CLAIM_NODE], cold.clone_population
        )
    assert host.values.source.epoch_record() is None
    assert epoch["misses"] and epoch["hits"] and epoch["final_validations"]
    cold.preparation._checked()
    warm.preparation._checked()
    assert host.reconstruction._population_stamp(cold.clone_population) == retained
    assert shutil.disk_usage is genuine.real


def test_no_geography_demographics_use_observed_source_state(tmp_path, monkeypatch):
    genuine = _RealDiskPatch(monkeypatch)
    arguments = _demographic_arguments(tmp_path, genuine, unknown=False, zero=False)
    assert genuine.suppressed > 0 and shutil.disk_usage is genuine.real
    with host.values.source.verification_epoch() as epoch:
        prefix = host.survey.run_authenticated_survey_population(
            **arguments,
            store_root=tmp_path / "survey-store",
            clones=True,
            return_values=True,
        )
        _check_prefix(prefix)
        qualified = host.values.qualify_current_survey_predictors(
            prefix.preparation,
            prefix.allocated_population,
            prefix.clone_population,
            demographic_conditioning=True,
            geography_config=None,
        )
        state = qualified.donor_columns["survey_predictor_state_fips"]
        assert len(state) and set(state) <= {6.0, 36.0}
        assert (
            qualified.donor_columns["survey_predictor_is_female"].isin((0.0, 1.0)).all()
        )
        matrix = host.financial.model_input.decode_recipient_matrix(qualified.matrix)
        assert tuple(matrix.features.columns) == host.values.DEMOGRAPHIC_FEATURES
        assert (
            qualified.geography_config_payload is qualified.geography_validation is None
        )
        assert "census_block_geoid" not in prefix.clone_population.frame.table(
            "household"
        )
        assert "assigned_state_fips" not in prefix.clone_population.frame.table(
            "household"
        )
        assert qualified.evidence["model_judgments"]["demographic_conditioning"]
    assert host.values.source.epoch_record() is None and epoch["final_validations"]
    prefix.preparation._checked()


@pytest.mark.parametrize(
    "run_type",
    (host.AtomicSurveyFinancialRunValues, host.PreGeographySurveyFinancialRunValues),
)
def test_public_result_construction_does_not_issue_financial_authority(run_type):
    run = run_type(*([None] * 9))
    for check in (
        run.checked_view,
        lambda: host.check_survey_financial_run(run),
        lambda: host.financial_host_edges(run),
    ):
        with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
            check()
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        budget_bridge.admit_survey_financial_population(object(), financial_run=run)


def test_prefix_host_edges_preserve_exact_order_and_atomic_contract():
    raw = host.financial.host.current_survey_host_edges()
    assert tuple(edge.name for edge in raw) == (
        "preparation",
        "allocation",
        "frame_context",
    )
    assert host._prefix_host_edges(host._PRE_GEOGRAPHY) == raw
    assert host._prefix_host_edges(host._ATOMIC) == (
        *raw,
        host.financial._geography_edge(),
    )


def test_named_variants_and_atomic_only_checker_preserve_scope(tmp_path):
    assert (
        "geography_config"
        in inspect.signature(host.run_atomic_survey_financial).parameters
    )
    assert (
        "geography_config"
        not in inspect.signature(host.run_pre_geography_survey_financial).parameters
    )
    pre = host.PreGeographySurveyFinancialRunValues(*([None] * 9))
    atomic = host.AtomicSurveyFinancialRunValues(*([None] * 9))
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        host.check_atomic_survey_financial_run(pre)
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        host.check_pre_geography_survey_financial_run(atomic)
    with pytest.raises(ValueError, match="FINANCIAL_PREFIX_TYPE"):
        host._prefix_config(object(), host._PRE_GEOGRAPHY)
    with pytest.raises(ValueError, match="FINANCIAL_PREFIX_MODE"):
        host._prefix_names("invented_other_mode")
    missing = tmp_path / "not-created"
    with pytest.raises(ValueError, match="ATOMIC_GEOGRAPHY_REQUIRED"):
        host.run_atomic_survey_financial(
            missing,
            snapshot_root=missing,
            store_root=missing,
            fraction=None,
            seed=0,
            geography_config=None,
        )
    assert not missing.exists()


@pytest.mark.parametrize(
    "kwargs,reason",
    (
        ({"person_status": 1}, "PERSON_STATUS_FLAG"),
        ({"household_roles": True}, "ROLES_REQUIRE_CHILD"),
        ({"rebase_property_taxes": True}, "PROPERTY_TAX_REQUIRES_PROPERTY"),
        ({"child_property": object()}, "CHILD_REQUIRES_PROPERTY"),
    ),
)
def test_pre_geography_invalid_options_refuse_before_source_work(
    tmp_path, kwargs, reason
):
    missing = tmp_path / "not-created"
    with pytest.raises(ValueError, match=reason):
        host.run_pre_geography_survey_financial(
            missing,
            snapshot_root=missing,
            store_root=missing,
            fraction=None,
            seed=0,
            **kwargs,
        )
    assert not missing.exists()
