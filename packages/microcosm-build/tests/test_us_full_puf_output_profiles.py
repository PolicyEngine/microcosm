"""Ordinary profile controls with invented donors and real graph/QRF execution.

The helper fixtures retain their explicit lack of source admission. These tests
run ordinary maintained fit/draw/finalizer and Population/store/replay paths;
they do not replace source qualification or an independent production gate.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_full_puf_enrichment import (
    InventedMatrix,
    InventedSource,
    _donor_columns,
    _known,
)
from test_us_graph_full_puf_enrichment import (
    InventedCompleteBoundary,
    _complete_frame,
    _copy_population,
    _load_artifact,
)
from test_us_graph_full_puf_enrichment import (
    attached_full65 as original_attached_full65,
)

from microcosm.build.us_runtime import full_puf_enrichment as full
from microcosm.build.us_runtime import graph_full_puf_enrichment as placement
from microcosm.build.us_runtime import puf_support as support
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph import population as population_ops
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.population import dtype_for_token

SCF_MORTGAGE_OUTPUTS = (
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
)
PUF59_TAX_UNIT_OUTPUTS = (
    "domestic_production_ald",
    "unrecaptured_section_1250_gain",
    "health_savings_account_ald",
)

# Register the existing ordinary fixture unchanged, without invoking its body.
attached_full65 = original_attached_full65


def _puf59_columns():
    person, tax_unit, person_known, tax_unit_known = _donor_columns()
    # Invented upstream leaves, not a source measurement implementation or receipt.
    tax_unit[full.PUF59.predictors[0]] = [1.0, 2.0, 3.0, 4.0]
    tax_unit[full.PUF59.predictors[1]] = [3.0, 5.0, 2.0, 4.0]
    for name in full.PUF59.predictors[:2]:
        tax_unit_known[name] = True
    return (
        person,
        tax_unit.drop(columns=list(SCF_MORTGAGE_OUTPUTS)),
        person_known,
        tax_unit_known.drop(columns=list(SCF_MORTGAGE_OUTPUTS)),
    )


def _puf59_donor():
    person, tax_unit, person_known, tax_unit_known = _puf59_columns()
    return full.canonical_full_puf_donor(
        person,
        tax_unit,
        person_known=person_known,
        tax_unit_known=tax_unit_known,
        profile=full.PUF59,
    )


def _puf59_return_only_columns():
    """Invented return incidence bridge; no physical people or source admission."""
    reduced = _puf59_donor()
    _, returns, _, _ = _puf59_columns()
    for name in full.PUF59.person_outputs:
        returns[name] = (
            np.array([0.0, 1.0, 1.0, 0.0])
            if name in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
            else reduced[name]
        )
    returns["puf_person_incidence_capacity"] = 1.0
    known = pd.DataFrame(
        True,
        index=returns.index,
        columns=(
            "weight",
            "filing_status_code",
            "puf_person_incidence_capacity",
            *full.PUF59.person_outputs,
            *full.PUF59.tax_unit_outputs,
            *full.PUF59.predictors[:2],
        ),
    )
    return returns, known


def _puf59_return_only_donor():
    returns, known = _puf59_return_only_columns()
    return full.canonical_full_puf_donor(
        None,
        returns,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=full.PUF59.person_outputs,
        profile=full.PUF59,
    )


def _puf59_frame():
    frame = _complete_frame()
    tax_unit = frame.table("tax_unit")
    tax_unit[full.PUF59.predictors[0]] = np.resize(
        np.array([2.0, 4.0, 1.0, 3.0]), len(tax_unit)
    )
    tax_unit[full.PUF59.predictors[1]] = np.resize(
        np.array([5.0, 4.0, 3.0, 2.0]), len(tax_unit)
    )
    return frame


def _puf59_known(frame):
    return _known(frame).set_axis(full.PUF59.predictors, axis="columns")


class InventedProfileMatrix(InventedMatrix):
    """Real ordinary projection of the declared selected predictor columns."""

    ref = "test.puf_profile.matrix@1"

    def run(self, context):
        table = context.tables["tax_unit"]
        ids = table.tax_unit_id.to_numpy()
        features = table.loc[:, list(context.node.inputs[0].columns)].copy()
        features.index = pd.Index(ids, name="tax_unit_id")
        return KernelResult(
            artifacts={
                "matrix": model_input.encode_recipient_matrix(
                    features, entity="tax_unit", entity_ids=ids
                )
            }
        )


def _finalizer_arguments(case, result):
    artifacts = result.artifacts
    return dict(
        predictor_known=case.binding.predictor_known,
        matrix=artifacts["matrix"].payload,
        matrix_producer_key=artifacts["matrix"].producer_key,
        raw_draws={
            target: artifacts[f"raw_{index:03d}"].payload
            for index, target in enumerate(case.binding.profile.targets)
        },
        apply_state=artifacts["apply_state"].payload,
        training_state=artifacts["training_state"].payload,
        last_model=artifacts["last_model"].payload,
        seed=578,
    )


def _independent_puf59_population(binding, nodes, artifacts):
    """Actual finalizer and graph patch, independent of attachment result code."""
    case = SimpleNamespace(binding=binding)
    arguments = _finalizer_arguments(case, SimpleNamespace(artifacts=artifacts))
    candidate, receipt = full.finalize_full_puf(
        binding.expected_population.frame,
        binding.donor,
        **arguments,
        profile=full.PUF59,
    )
    expected = binding.expected_population
    masks = {}
    for entity in ("person", "tax_unit"):
        table = expected.frame.table(entity)
        id_column = expected.frame.schema.entity_id_column(entity)
        masks[entity] = pd.Series(
            support.puf_tax_detail_clone_mask(table, entity=entity),
            index=pd.Index(table[id_column], name=id_column),
            dtype=bool,
        )
    expected = population_ops.patch(
        expected,
        nodes[0],
        KernelResult(
            columns={
                (entity, placement.MASKS[entity]): mask
                for entity, mask in masks.items()
            }
        ),
    )
    columns = {}
    for owned in nodes[1].outputs:
        table = candidate.table(owned.entity)
        mask = masks[owned.entity].to_numpy()
        id_column = candidate.schema.entity_id_column(owned.entity)
        columns[(owned.entity, owned.column)] = pd.Series(
            table.loc[mask, owned.column].array,
            index=pd.Index(table.loc[mask, id_column], name=id_column),
            dtype=dtype_for_token(owned.dtype),
        )
    return population_ops.patch(
        expected, nodes[1], KernelResult(columns=columns)
    ), receipt


@pytest.fixture(scope="module", params=("absent", "incumbent"))
def attached_puf59(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("puf59_" + request.param)
    frame = _puf59_frame()
    donor = _puf59_return_only_donor() if request.param == "absent" else _puf59_donor()
    if request.param == "incumbent":
        # The actual Frame contract requires global column-name uniqueness.
        # These six details belong to tax units; the separate observed person
        # home_mortgage_interest destination remains part of the PUF59 profile.
        table = frame.table("tax_unit")
        for index, name in enumerate(SCF_MORTGAGE_OUTPUTS):
            # Independent incumbent bytes, including nulls and signed zero,
            # survive on native AND clone rows without entering the PUF model.
            table[name] = np.resize(
                np.array([-0.0, np.nan, -12.5, 1800.5 + index]), len(table)
            )
            assert name not in frame.person
    frame.revalidate()
    known = _puf59_known(frame)
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=known, profile=full.PUF59
    )
    decoded = model_input.decode_recipient_matrix(prepared.matrix)
    table = decoded.features.copy()
    table.insert(0, "tax_unit_id", decoded.entity_ids)
    matrix_frame = Frame(
        {
            "tax_unit": table,
            "person": pd.DataFrame(
                {
                    "person_id": decoded.entity_ids,
                    "person_tax_unit_id": decoded.entity_ids,
                }
            ),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(np.ones(len(table), dtype="float64"), WeightKind.DESIGN)},
    )
    retained_store = ContentStore(
        root / "retained", codecs={"frame-store": load_frame_store}
    )
    sources, nodes = {}, []
    for name, value in (
        ("survey", frame),
        ("donor", prepared.donor_frame),
        ("matrix_input", matrix_frame),
    ):
        source = "invented." + name
        sources[source] = retained_store.put_frame(
            full.codec.sha(("ordinary-puf59:" + request.param + ":" + name).encode()),
            value,
        )
        outputs = (
            frame_column_declarations(value)
            if name == "survey"
            else tuple(
                Owned("tax_unit", name, "float64")
                for name in value.table("tax_unit")
                if name != "tax_unit_id"
            )
        )
        nodes.append(
            Node(
                name,
                InventedSource.ref,
                sources=(source,),
                structural=StructuralDelta.CREATE,
                outputs=outputs,
            )
        )
    boundary = Node(
        "complete_upstream",
        InventedCompleteBoundary.ref,
        base="survey",
        structural=StructuralDelta.FILTER,
        inputs=(Slice("person", ("age",)),),
    )
    matrix_node = Node(
        "matrix",
        InventedProfileMatrix.ref,
        population="matrix_input",
        inputs=(Slice("tax_unit", full.PUF59.predictors),),
        artifact_outputs=(ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),),
    )
    base_graph = Graph(
        "us",
        tuple(SourceRef(name, "frame-store") for name in sources),
        (*nodes, boundary, matrix_node),
    )
    kernels = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedCompleteBoundary(),
        InventedProfileMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        kernels.register(kernel)
    observed = {}
    retained_manifest = run_graph(
        compile_graph(base_graph),
        sources=sources,
        store=retained_store,
        kernels=kernels,
        _population_observer=lambda name, value: observed.__setitem__(name, value),
    )
    upstream = observed[boundary.id]
    matrix = _load_artifact(
        retained_manifest,
        retained_store,
        ArtifactInput(
            "matrix", matrix_node.id, "matrix", model_input.RECIPIENT_MATRIX_TYPE
        ),
    )
    assert matrix.payload == prepared.matrix
    fits, applies = full.full_puf_train_apply_nodes(
        donor_population="donor",
        recipient_population=boundary.id,
        matrix_producer=matrix_node.id,
        seed=578,
        n_estimators=2,
        zero_atol=0,
        profile=full.PUF59,
    )
    retention_arguments = dict(
        population=upstream,
        expected_population=_copy_population(upstream),
        population_node=boundary,
        input_owners=dict(upstream.owners),
        donor=donor,
        predictor_known=known,
        matrix=matrix,
        fit_nodes=fits,
        apply_nodes=applies,
        profile=full.PUF59,
    )
    binding = placement.retain_full_puf_attachment(**retention_arguments)
    attachment_nodes = placement.full_puf_attachment_nodes(binding)
    kernels.register(placement.FullPufMaskKernel(binding))
    kernels.register(placement.FullPufAttachKernel(binding))
    compiled = compile_graph(
        replace(
            base_graph, nodes=(*base_graph.nodes, *fits, *applies, *attachment_nodes)
        )
    )
    store = ContentStore(root / "application", codecs={"frame-store": load_frame_store})
    results = []
    for resume in ("auto", "require"):
        observed = {}
        manifest = run_graph(
            compiled,
            sources=sources,
            store=store,
            kernels=kernels,
            resume=resume,
            _population_observer=lambda name, value, target=observed: (
                target.__setitem__(name, value)
            ),
        )
        assert all(
            record.hit is (resume == "require") for record in manifest.nodes.values()
        )
        artifacts, producer_keys = placement.load_full_puf_attachment_artifacts(
            binding, compiled=compiled, manifest=manifest, store=store
        )
        actual = observed[attachment_nodes[-1].id]
        evidence = placement.verify_materialized_full_puf_attachment(
            binding,
            upstream_population=observed[boundary.id],
            population=actual,
            artifacts=artifacts,
            producer_keys=producer_keys,
        )
        expected, receipt = _independent_puf59_population(
            binding, attachment_nodes, artifacts
        )
        replay.same_replayed_population(expected, actual)
        frame_key = full.codec.sha(
            (
                "puf59-materialized:" + manifest.node(attachment_nodes[-1].id).key
            ).encode()
        )
        store.put_frame(frame_key, actual.frame)
        replay.same_replayed_frame(expected.frame, store.load_frame(frame_key))
        results.append(
            SimpleNamespace(
                manifest=manifest,
                population=_copy_population(actual),
                upstream=_copy_population(observed[boundary.id]),
                artifacts=artifacts,
                producer_keys=producer_keys,
                evidence=evidence,
                finalizer_receipt=receipt,
            )
        )
    replay.same_replayed_population(results[0].population, results[1].population)
    assert results[0].artifacts == results[1].artifacts
    assert results[0].manifest.key == results[1].manifest.key
    return SimpleNamespace(
        binding=binding,
        retention_arguments=retention_arguments,
        nodes=attachment_nodes,
        fits=fits,
        applies=applies,
        results=results,
        compiled=compiled,
        sources=sources,
        store=store,
        kernels=kernels,
        mortgage_mode=request.param,
    )


def test_closed_profiles_keep_observed_mortgage_and_original_conditioning_order():
    assert (
        full.FULL65.person_outputs == full.PUF59.person_outputs == full.PERSON_OUTPUTS
    )
    assert full.FULL65.tax_unit_outputs == full.TAX_UNIT_OUTPUTS
    assert full.FULL65.targets == full.TARGETS
    assert full.PUF59.tax_unit_outputs == PUF59_TAX_UNIT_OUTPUTS
    assert len(full.PUF59.person_outputs) == 56 and len(full.PUF59.targets) == 59
    assert full.PUF59.targets == tuple(
        t for t in full.TARGETS if t not in SCF_MORTGAGE_OUTPUTS
    )
    assert "home_mortgage_interest" in full.PUF59.person_outputs
    assert not {"prior_year_wages", "employment_income_last_year"} & set(
        full.PUF59.targets
    )
    assert full.FULL65.predictors == full.PREDICTORS
    assert full.FULL65.donor_auxiliary_columns == ()
    assert full.PUF59.donor_auxiliary_columns == ("puf_person_incidence_capacity",)
    assert full.PUF59.predictors == (
        "puf_2015_filing_status_code",
        "puf_2015_capped_return_size",
        *full.PREDICTORS[2:],
    )


@pytest.mark.parametrize("invalid", (None, "puf59", "full65", 59, True, ("puf59",)))
def test_profiles_require_explicit_enum_members(invalid):
    with pytest.raises(ValueError, match="PUF_OUTPUT_PROFILE"):
        full.require_puf_output_profile(invalid)


def test_puf59_canonical_donor_accepts_truly_absent_scf_columns_and_keeps_observed_interest():
    person, tax_unit, pk, tk = _puf59_columns()
    # Actual canonical arithmetic after an invented QBI allocation: entirely
    # SSTB positive, ordinary positive, entirely SSTB loss, ordinary positive.
    person["self_employment_income_before_lsr"] = [
        0.0,
        0.0,
        50.0,
        25.0,
        0.0,
        0.0,
        700.0,
        300.0,
    ]
    person["sstb_self_employment_income_before_lsr"] = [
        900.0,
        100.0,
        0.0,
        0.0,
        -250.0,
        -50.0,
        0.0,
        0.0,
    ]
    before = tuple(value.copy(deep=True) for value in (person, tax_unit, pk, tk))
    donor = full.canonical_full_puf_donor(
        person, tax_unit, person_known=pk, tax_unit_known=tk, profile=full.PUF59
    )
    assert tuple(donor) == (
        *full.PUF59.predictors,
        *full.PUF59.targets,
        "weight",
        "puf_person_incidence_capacity",
    )
    assert not set(SCF_MORTGAGE_OUTPUTS) & set(donor)
    np.testing.assert_array_equal(
        donor.home_mortgage_interest,
        person.home_mortgage_interest.groupby(person.person_tax_unit_id).sum(),
    )
    assert donor.weight.tolist() == [1.0, 0.0, 2.0, 3.0]
    np.testing.assert_array_equal(
        donor[full.PUF59.predictors[3]], [1000.0, 75.0, -300.0, 1000.0]
    )
    np.testing.assert_array_equal(
        donor.self_employment_income_before_lsr, [0.0, 75.0, 0.0, 1000.0]
    )
    np.testing.assert_array_equal(
        donor.sstb_self_employment_income_before_lsr, [1000.0, 0.0, -300.0, 0.0]
    )
    for actual, expected in zip((person, tax_unit, pk, tk), before, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(ValueError, match="PUF_DONOR_COLUMNS:tax_unit"):
        full.canonical_full_puf_donor(
            person, tax_unit, person_known=pk, tax_unit_known=tk
        )


def test_puf59_donor_does_not_validate_excluded_years_or_consume_scf_payloads():
    person, tax_unit, pk, tk = _puf59_columns()
    expected = _puf59_donor()
    for name in SCF_MORTGAGE_OUTPUTS:
        tax_unit[name] = ["independent SCF", None, False, 1800.5]
    actual = full.canonical_full_puf_donor(
        person, tax_unit, person_known=pk, tax_unit_known=tk, profile=full.PUF59
    )
    pd.testing.assert_frame_equal(actual, expected)
    person["business_is_sstb"] = person.business_is_sstb.astype("float64")
    with pytest.raises(ValueError, match="PUF_PHYSICAL_TYPE:person.business_is_sstb"):
        full.canonical_full_puf_donor(
            person, tax_unit, person_known=pk, tax_unit_known=tk, profile=full.PUF59
        )


def test_puf59_return_only_donor_uses_explicit_leaves_without_inventing_persons():
    expected = _puf59_donor()
    returns, known = _puf59_return_only_columns()
    before = returns.copy(deep=True), known.copy(deep=True)
    assert "tax_unit_person_count" not in returns
    assert "tax_unit_person_count" not in known
    for name in full.PUF59.person_outputs:
        if name in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS:
            assert returns[name].dtype == np.dtype("float64")
            assert set(returns[name]) == {0.0, 1.0}
            expected[name] = returns[name]
    expected["puf_person_incidence_capacity"] = 1.0
    actual = full.canonical_full_puf_donor(
        None,
        returns,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=full.PUF59.person_outputs,
        profile=full.PUF59,
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert bool((returns.sstb_self_employment_income_before_lsr != 0).any())
    np.testing.assert_array_equal(
        actual[full.PUF59.predictors[3]],
        returns.self_employment_income_before_lsr
        + returns.sstb_self_employment_income_before_lsr,
    )
    pd.testing.assert_frame_equal(returns, before[0])
    pd.testing.assert_frame_equal(known, before[1])


@pytest.mark.parametrize(
    ("failure", "error"),
    (
        ("absent", "PUF_DONOR_COLUMNS:tax_unit"),
        ("physical_count_alias", "PUF_DONOR_COLUMNS:tax_unit"),
        ("unknown", "PUF_UNKNOWN:tax_unit.puf_person_incidence_capacity"),
        ("missing_knownness", "PUF_KNOWNNESS_AXIS:tax_unit"),
    ),
)
def test_puf59_return_only_requires_exact_known_incidence_capacity(failure, error):
    returns, known = _puf59_return_only_columns()
    name = "puf_person_incidence_capacity"
    if failure == "absent":
        returns.drop(columns=name, inplace=True)
    elif failure == "physical_count_alias":
        returns.rename(columns={name: "tax_unit_person_count"}, inplace=True)
        known.rename(columns={name: "tax_unit_person_count"}, inplace=True)
    elif failure == "unknown":
        known.loc[0, name] = False
    else:
        known.drop(columns=name, inplace=True)
    with pytest.raises(ValueError, match=error):
        full.canonical_full_puf_donor(
            None,
            returns,
            person_known=None,
            tax_unit_known=known,
            person_targets_at_tax_unit=full.PUF59.person_outputs,
            profile=full.PUF59,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (0.0, "PUF_PERSON_INCIDENCE_CAPACITY_DOMAIN"),
        (-1.0, "PUF_PERSON_INCIDENCE_CAPACITY_DOMAIN"),
        (0.5, "PUF_PERSON_INCIDENCE_CAPACITY_DOMAIN"),
        ("1", "PUF_PHYSICAL_TYPE:puf_person_incidence_capacity"),
        (True, "PUF_PHYSICAL_TYPE:puf_person_incidence_capacity"),
        (None, "PUF_UNKNOWN:puf_person_incidence_capacity"),
        (np.inf, "PUF_NONFINITE:puf_person_incidence_capacity"),
    ),
)
def test_puf59_return_only_incidence_capacity_requires_physical_positive_integer(
    value, error
):
    returns, known = _puf59_return_only_columns()
    name = "puf_person_incidence_capacity"
    returns[name] = returns[name].astype(object)
    returns.loc[0, name] = value
    with pytest.raises(ValueError, match=error):
        full.canonical_full_puf_donor(
            None,
            returns,
            person_known=None,
            tax_unit_known=known,
            person_targets_at_tax_unit=full.PUF59.person_outputs,
            profile=full.PUF59,
        )


def test_puf59_return_only_rejects_boolean_incidence_above_source_capacity():
    returns, known = _puf59_return_only_columns()
    returns.loc[0, "business_is_sstb"] = 2.0
    with pytest.raises(ValueError, match="PUF_BOOLEAN_COUNT_DOMAIN:business_is_sstb"):
        full.canonical_full_puf_donor(
            None,
            returns,
            person_known=None,
            tax_unit_known=known,
            person_targets_at_tax_unit=full.PUF59.person_outputs,
            profile=full.PUF59,
        )
    donor = _puf59_return_only_donor()
    donor.loc[0, "business_is_sstb"] = 2.0
    frame = _puf59_frame()
    with pytest.raises(ValueError, match="PUF_BOOLEAN_COUNT_DOMAIN:business_is_sstb"):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=_puf59_known(frame), profile=full.PUF59
        )


@pytest.mark.parametrize("predictor_index", range(8))
def test_puf59_requires_all_eight_known_predictors(predictor_index):
    frame, donor = _puf59_frame(), _puf59_donor()
    known = _puf59_known(frame)
    known.iloc[0, predictor_index] = False
    with pytest.raises(ValueError, match="PUF_UNKNOWN:recipient_predictor"):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=known, profile=full.PUF59
        )
    known.iloc[0, predictor_index] = True
    with pytest.raises(ValueError, match="PUF_KNOWNNESS_AXIS:recipient_predictor"):
        full.prepare_full_puf_inputs(
            frame,
            donor,
            predictor_known=known.drop(columns=known.columns[predictor_index]),
            profile=full.PUF59,
        )


@pytest.mark.parametrize("predictor_index", (0, 1))
def test_puf59_requires_new_leaves_and_never_falls_back_to_generic_measurements(
    predictor_index,
):
    frame, donor = _puf59_frame(), _puf59_donor()
    known = _puf59_known(frame)
    before = frame.table("tax_unit").filing_status_input.copy(deep=True)
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=known, profile=full.PUF59
    )
    assert prepared.profile is full.PUF59
    decoded = model_input.decode_recipient_matrix(prepared.matrix)
    assert tuple(decoded.features) == full.PUF59.predictors
    tax_unit = frame.table("tax_unit")
    mask = support.puf_tax_detail_clone_mask(tax_unit, entity="tax_unit")
    for name in full.PUF59.predictors[:2]:
        np.testing.assert_array_equal(decoded.features[name], tax_unit.loc[mask, name])
    pd.testing.assert_series_equal(tax_unit.filing_status_input, before)
    name = full.PUF59.predictors[predictor_index]
    tax_unit.drop(columns=name, inplace=True)
    with pytest.raises(ValueError, match="PUF_PROFILE_PREDICTOR_SOURCE:" + name):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=known, profile=full.PUF59
        )
    # A same-named person column is also insufficient for the explicit return-grain leaf.
    frame.table("person")[name] = 1.0
    with pytest.raises(ValueError, match="PUF_PROFILE_PREDICTOR_SOURCE:" + name):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=known, profile=full.PUF59
        )
    person, returns, pk, tk = _puf59_columns()
    with pytest.raises(ValueError, match="PUF_DONOR_COLUMNS:tax_unit"):
        full.canonical_full_puf_donor(
            person,
            returns.drop(columns=name),
            person_known=pk,
            tax_unit_known=tk.drop(columns=name),
            profile=full.PUF59,
        )


@pytest.mark.parametrize("value", (0.5, 6.0))
def test_puf59_prepare_still_rejects_invalid_selected_boolean_counts(value):
    frame, donor = _puf59_frame(), _puf59_donor()
    donor.loc[0, "business_is_sstb"] = value
    with pytest.raises(ValueError, match="PUF_BOOLEAN_COUNT_DOMAIN:business_is_sstb"):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=_puf59_known(frame), profile=full.PUF59
        )


@pytest.mark.parametrize(
    ("predictor_index", "value", "error"),
    (
        (0, "1", "PUF_PHYSICAL_TYPE"),
        (1, "2", "PUF_PHYSICAL_TYPE"),
        (0, True, "PUF_PHYSICAL_TYPE"),
        (1, False, "PUF_PHYSICAL_TYPE"),
        (0, None, "missing values before coercion"),
        (1, np.inf, "PUF_NONFINITE"),
        (0, 5.0, "PUF_PROFILE_FILING_STATUS_DOMAIN"),
        (1, 1.5, "PUF_PROFILE_RETURN_SIZE_DOMAIN"),
    ),
)
def test_puf59_measured_predictor_leaves_require_physical_values_and_domains(
    predictor_index, value, error
):
    frame, donor = _puf59_frame(), _puf59_donor()
    tax_unit = frame.table("tax_unit")
    selected = support.puf_tax_detail_clone_mask(tax_unit, entity="tax_unit")
    name = full.PUF59.predictors[predictor_index]
    tax_unit[name] = tax_unit[name].astype(object)
    tax_unit.loc[tax_unit.index[selected][0], name] = value
    with pytest.raises(ValueError, match=error):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=_puf59_known(frame), profile=full.PUF59
        )


def test_puf59_person_donor_derives_incidence_capacity_from_actual_membership():
    person, returns, pk, tk = _puf59_columns()
    person.loc[:1, "business_is_sstb"] = True
    returns.loc[0, full.PUF59.predictors[1]] = 1.0
    # The bound is derived from membership here, not read as a return source fact.
    returns["puf_person_incidence_capacity"] = "not a source fact"
    returns["tax_unit_person_count"] = "not a source fact"
    donor = full.canonical_full_puf_donor(
        person, returns, person_known=pk, tax_unit_known=tk, profile=full.PUF59
    )
    assert (
        donor.loc[0, "business_is_sstb"]
        == donor.loc[0, "puf_person_incidence_capacity"]
        == 2
    )
    assert donor.loc[0, full.PUF59.predictors[1]] == 1
    frame = _puf59_frame()
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=_puf59_known(frame), profile=full.PUF59
    )
    for auxiliary in ("puf_person_incidence_capacity", "tax_unit_person_count"):
        assert auxiliary not in prepared.donor
        assert auxiliary not in prepared.donor_frame.table("tax_unit")
    donor.loc[0, "puf_person_incidence_capacity"] = 1.0
    with pytest.raises(ValueError, match="PUF_BOOLEAN_COUNT_DOMAIN:business_is_sstb"):
        full.prepare_full_puf_inputs(
            frame, donor, predictor_known=_puf59_known(frame), profile=full.PUF59
        )


@pytest.mark.parametrize(
    "year",
    ("first_home_mortgage_origination_year", "second_home_mortgage_origination_year"),
)
def test_default_full65_retains_selected_discrete_year_checks(year):
    person, tax_unit, pk, tk = _donor_columns()
    tax_unit.loc[0, year] = 2010.5
    with pytest.raises(ValueError, match="PUF_YEAR_DOMAIN:" + year):
        full.canonical_full_puf_donor(
            person, tax_unit, person_known=pk, tax_unit_known=tk
        )


def test_default_full65_return_only_retains_legacy_person_count_contract():
    person, returns, person_known, tax_unit_known = _donor_columns()
    for name in full.FULL65.person_outputs:
        if name in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS:
            person.loc[:1, name] = True
    expected = full.canonical_full_puf_donor(
        person,
        returns,
        person_known=person_known,
        tax_unit_known=tax_unit_known,
    )
    returns["tax_unit_person_count"] = 2.0
    for name in full.FULL65.person_outputs:
        returns[name] = expected[name]
    known = pd.DataFrame(
        True,
        index=returns.index,
        columns=(
            "weight",
            "filing_status_code",
            "tax_unit_person_count",
            *full.FULL65.person_outputs,
            *full.FULL65.tax_unit_outputs,
        ),
    )
    assert "puf_person_incidence_capacity" not in returns
    for arguments in ({}, {"profile": full.FULL65}):
        actual = full.canonical_full_puf_donor(
            None,
            returns,
            person_known=None,
            tax_unit_known=known,
            person_targets_at_tax_unit=full.FULL65.person_outputs,
            **arguments,
        )
        pd.testing.assert_frame_equal(actual, expected)
        assert actual.loc[0, "business_is_sstb"] == 2.0
        # FULL65 compatibility is explicit: its historical alias excludes the
        # nonzero SSTB component rather than silently changing old evidence.
        assert bool((returns.sstb_self_employment_income_before_lsr != 0).any())
        np.testing.assert_array_equal(
            actual[full.FULL65.predictors[3]], returns.self_employment_income_before_lsr
        )
    returns["puf_person_incidence_capacity"] = 1.0
    actual = full.canonical_full_puf_donor(
        None,
        returns,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=full.FULL65.person_outputs,
    )
    pd.testing.assert_frame_equal(actual, expected)
    returns.drop(columns="tax_unit_person_count", inplace=True)
    with pytest.raises(ValueError, match="PUF_DONOR_COLUMNS:tax_unit"):
        full.canonical_full_puf_donor(
            None,
            returns,
            person_known=None,
            tax_unit_known=known,
            person_targets_at_tax_unit=full.FULL65.person_outputs,
        )


def test_real_puf59_chain_finalization_attachment_and_required_replay(attached_puf59):
    case = attached_puf59
    assert case.binding.profile is full.PUF59
    assert len(case.fits) == len(case.applies) == 59
    assert [
        edge.name
        for edge in case.applies[-1].artifact_inputs
        if edge.name.startswith("prior_")
    ] == [f"prior_{i:03d}" for i in range(58)]
    assert tuple(owned.column for owned in case.nodes[-1].outputs) == full.PUF59.targets
    for node in (*case.fits, *case.applies):
        assert node.params["phase"] == full.PUF59.phase
        assert not {
            *SCF_MORTGAGE_OUTPUTS,
            "puf_person_incidence_capacity",
            "tax_unit_person_count",
        } & {column for item in node.inputs for column in item.columns}
    assert all(node.params["predictors"] == full.PUF59.predictors for node in case.fits)
    assert all(node.params["targets"] == full.PUF59.targets for node in case.fits)
    assert "raw_058" in case.results[0].artifacts
    assert "raw_059" not in case.results[0].artifacts
    if case.mortgage_mode == "absent":
        assert case.binding.donor.puf_person_incidence_capacity.eq(1.0).all()
        for name in full.PUF59.person_outputs:
            if name in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS:
                assert set(case.binding.donor[name]) == {0.0, 1.0}
    for result in case.results:
        assert result.finalizer_receipt["output_profile"] == "puf59"
        assert result.finalizer_receipt["person_target_count"] == 56
        assert result.finalizer_receipt["tax_unit_target_count"] == 3
        assert result.finalizer_receipt["target_count"] == 59
        assert result.finalizer_receipt["predictor_order"] == list(
            full.PUF59.predictors
        )
        assert result.finalizer_receipt["donor_auxiliary_columns"] == [
            "puf_person_incidence_capacity"
        ]
        assert result.finalizer_receipt["target_order"] == list(full.PUF59.targets)
        assert result.finalizer_receipt["person_target_count"] == 56
        assert result.finalizer_receipt["tax_unit_target_count"] == 3
        assert result.finalizer_receipt["target_count"] == 59
        assert result.evidence["output_profile"] == "puf59"
        assert result.evidence["source_admission_issued"] is False
        assert result.evidence["release_eligible"] is False
        for entity in ("person", "tax_unit"):
            before, after = (
                result.upstream.frame.table(entity),
                result.population.frame.table(entity),
            )
            for name in SCF_MORTGAGE_OUTPUTS:
                if case.mortgage_mode == "absent" or entity != "tax_unit":
                    assert name not in before and name not in after
                else:
                    assert population_ops.storage_equal(
                        before[name], after[name], np.ones(len(before), dtype=bool)
                    )
                    assert (
                        result.population.owners[(entity, name)]
                        == result.upstream.owners[(entity, name)]
                    )
        for owned in case.nodes[-1].outputs:
            before = result.upstream.frame.table(owned.entity)
            after = result.population.frame.table(owned.entity)
            mask = support.puf_tax_detail_clone_mask(before, entity=owned.entity)
            assert after.loc[mask, owned.column].notna().all()
            if owned.column in before:
                assert population_ops.storage_equal(
                    before[owned.column], after[owned.column], ~mask
                )
            else:
                assert after.loc[~mask, owned.column].isna().all()


def test_puf59_incidence_capacity_remains_bound_to_attachment(attached_puf59):
    case = attached_puf59
    donor = case.binding.donor.copy(deep=True)
    donor["puf_person_incidence_capacity"] += 1.0
    stale = replace(case.binding, donor=donor)
    with pytest.raises(ValueError, match="FULL_PUF_DONOR_CHANGED"):
        placement.full_puf_attachment_nodes(stale)
    rebound = placement.retain_full_puf_attachment(
        **{**case.retention_arguments, "donor": donor}
    )
    assert rebound.fit_nodes == case.binding.fit_nodes
    assert rebound.apply_nodes == case.binding.apply_nodes
    assert rebound.matrix == case.binding.matrix
    nodes = placement.full_puf_attachment_nodes(rebound)
    for original, changed in zip(case.nodes, nodes, strict=True):
        assert original.params["donor_values"] != changed.params["donor_values"]


def test_default_full65_matches_explicit_full65_and_runs_actual_complete_chain(
    attached_full65,
):
    case = attached_full65
    kwargs = dict(
        donor_population="donor",
        recipient_population="complete_upstream",
        matrix_producer="matrix",
        seed=578,
        n_estimators=2,
        zero_atol=0,
    )
    default = full.full_puf_train_apply_nodes(**kwargs)
    explicit = full.full_puf_train_apply_nodes(**kwargs, profile=full.FULL65)
    assert default == explicit == (case.fits, case.applies)
    assert len(case.fits) == len(case.applies) == 65
    assert tuple(owned.column for owned in case.nodes[-1].outputs) == full.TARGETS
    assert case.binding.profile is full.FULL65
    assert set(SCF_MORTGAGE_OUTPUTS) <= set(full.FULL65.tax_unit_outputs)
    for result in case.results:
        assert "raw_064" in result.artifacts
        assert result.finalizer_receipt["output_profile"] == "full65"
        assert result.finalizer_receipt["target_order"] == list(full.TARGETS)


def test_actual_wrong_profile_chains_and_manifests_are_refused(
    attached_puf59, attached_full65
):
    for correct, wrong in (
        (attached_puf59, attached_full65),
        (attached_full65, attached_puf59),
    ):
        with pytest.raises(ValueError, match="FULL_PUF_CHAIN_ROSTER"):
            placement.retain_full_puf_attachment(
                **{**correct.retention_arguments, "profile": wrong.binding.profile}
            )
        result = correct.results[-1]
        arguments = _finalizer_arguments(correct, result)
        arguments.pop("predictor_known")
        arguments.pop("last_model")
        with pytest.raises(ValueError, match="PUF_"):
            full.decode_full_puf_draws(**arguments, profile=wrong.binding.profile)
        # Feed the real other-profile compiled graph, manifest and ContentStore;
        # no fabricated manifest, source issuer or artifact readiness is involved.
        with pytest.raises(ValueError, match="FULL_PUF_"):
            placement.load_full_puf_attachment_artifacts(
                correct.binding,
                compiled=wrong.compiled,
                manifest=wrong.results[-1].manifest,
                store=wrong.store,
            )


def test_puf59_wrong_profile_model_and_trimmed_full65_history_are_refused(
    attached_puf59, attached_full65
):
    case, result = attached_puf59, attached_puf59.results[-1]
    arguments = _finalizer_arguments(case, result)
    arguments["last_model"] = (
        attached_full65.results[-1].artifacts["last_model"].payload
    )
    with pytest.raises(ValueError, match="Legacy QRF artifact content digest mismatch"):
        full.finalize_full_puf(
            case.binding.expected_population.frame,
            case.binding.donor,
            **arguments,
            profile=full.PUF59,
        )
    full_arguments = _finalizer_arguments(attached_full65, attached_full65.results[-1])
    full_arguments.pop("predictor_known")
    full_arguments.pop("last_model")
    full_arguments["raw_draws"] = {
        target: full_arguments["raw_draws"][target] for target in full.PUF59.targets
    }
    with pytest.raises(ValueError, match="PUF_"):
        full.decode_full_puf_draws(**full_arguments, profile=full.PUF59)


def test_puf59_rebuilt_binding_survives_nullable_frame_store_normalization(
    attached_puf59, tmp_path
):
    case = attached_puf59
    retained = _copy_population(case.binding.expected_population)
    nullable = retained.frame.table("person").unrelated_nullable.array
    missing = nullable._mask.copy()
    assert missing.any()
    nullable._data[missing] = True
    arguments = {**case.retention_arguments, "expected_population": retained}
    physical_binding = placement.retain_full_puf_attachment(**arguments)
    assert placement.full_puf_attachment_nodes(physical_binding) == case.nodes

    directory = tmp_path / "puf59-new-session"
    key = full.codec.sha(b"invented-puf59-retained-upstream")
    ContentStore(directory).put_frame(key, retained.frame)
    # Fresh ContentStore decoding performs the actual v2 masked-byte normalization.
    restored_frame = ContentStore(directory).load_frame(key)
    assert (
        not restored_frame.table("person").unrelated_nullable.array._data[missing].any()
    )
    replay.same_replayed_frame(retained.frame, restored_frame)
    restored = replace(_copy_population(retained), frame=restored_frame)
    upstream = _copy_population(case.results[-1].upstream)
    rebuilt = placement.retain_full_puf_attachment(
        **{
            **arguments,
            "population": upstream,
            "expected_population": restored,
            "input_owners": dict(upstream.owners),
        }
    )
    assert physical_binding.expected_stamp != rebuilt.expected_stamp
    nodes = placement.full_puf_attachment_nodes(rebuilt)
    assert nodes == case.nodes
    replacements = {node.id: node for node in nodes}
    compiled = compile_graph(
        replace(
            case.compiled.graph,
            nodes=tuple(
                replacements.get(node.id, node) for node in case.compiled.graph.nodes
            ),
        )
    )
    registry = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedCompleteBoundary(),
        InventedProfileMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        placement.FullPufMaskKernel(rebuilt),
        placement.FullPufAttachKernel(rebuilt),
    ):
        registry.register(kernel)
    observed = {}
    manifest = run_graph(
        compiled,
        sources=case.sources,
        store=case.store,
        kernels=registry,
        resume="require",
        _population_observer=lambda name, value: observed.__setitem__(name, value),
    )
    assert all(record.hit for record in manifest.nodes.values())
    assert manifest.key == case.results[-1].manifest.key
    artifacts, producer_keys = placement.load_full_puf_attachment_artifacts(
        rebuilt, compiled=compiled, manifest=manifest, store=case.store
    )
    evidence = placement.verify_materialized_full_puf_attachment(
        rebuilt,
        upstream_population=observed[rebuilt.population_node.id],
        population=observed[nodes[-1].id],
        artifacts=artifacts,
        producer_keys=producer_keys,
    )
    assert evidence["output_profile"] == "puf59"
    replay.same_replayed_population(case.results[-1].population, observed[nodes[-1].id])
