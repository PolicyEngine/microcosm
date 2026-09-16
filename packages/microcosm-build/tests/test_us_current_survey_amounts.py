"""Invented reporting bases and clone joins; no native data or country engine."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_unemployment_source as uc
from microcosm.build.us_runtime import current_survey_amounts as values
from microcosm.fit import model_input
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


@pytest.mark.parametrize(
    "amount,age,token,label,canonical",
    [
        (100, 40, "1", "known_receipt", 100),
        (0, 40, "2", "known_nonreceipt", 0),
        (0, 40, "1", "ambiguous_recipient_zero", None),
        (0, 40, "0", "niu", None),
        (0, 14, "0", "outside_reporting_universe", None),
        (0, 14, "2", "contradictory_outside_reporting_universe", None),
        (100, 14, "1", "contradictory_outside_reporting_universe", None),
        (100, 40, "2", "contradictory_no_positive", None),
        (100, 40, "0", "contradictory_niu_positive", None),
        (0, 40, "", "missing_receipt_literal", None),
        (100, 40, "9", "unrecognized_receipt_literal", None),
        (100, 40, "yes", "unrecognized_receipt_literal", None),
        (100, 40, "01", "unrecognized_receipt_literal", None),
        (np.nan, 40, "1", "missing_amount", None),
    ],
)
def test_uc_raw_amount_and_reporting_knownness_are_separate(
    amount, age, token, label, canonical
):
    basis = uc.reporting_basis(
        np.array([amount], dtype="float64"), np.array([age], dtype="float64"), [token]
    )
    row = basis.iloc[0]
    assert row.reporting_status == label
    assert row.receipt_literal == token
    assert bool(row.source_reporting_universe) == (age >= 15)
    assert bool(row.canonical_amount_known) == (canonical is not None)
    if canonical is None:
        assert np.isnan(row.canonical_amount)
    else:
        assert row.canonical_amount == canonical
    assert (
        np.isnan(row.source_amount) if np.isnan(amount) else row.source_amount == amount
    )


@pytest.mark.parametrize(
    "amount,age", [(-1, 40), (100000, 40), (np.inf, 40), (0, -1), (0, 40.5)]
)
def test_uc_refuses_out_of_dictionary_numeric_domain(amount, age):
    with pytest.raises(ValueError):
        uc.reporting_basis(
            np.array([amount], dtype="float64"), np.array([age], dtype="float64"), ["2"]
        )


def test_uc_bounded_literal_reader_and_duplicate_join_refusal(tmp_path):
    path = tmp_path / "invented.csv"
    header = ",".join(uc.READ_COLUMNS) + "\n"
    row = "0000000000000000000001,1,1,40,0,2\n"
    path.write_text(header + row)
    result = uc._read_capture(path, rows=1)
    assert result.iloc[0].UC_YN == "2"
    path.write_text(header + row + row)
    with pytest.raises(ValueError, match="DUPLICATE"):
        uc._read_capture(path, rows=2)
    path.write_text(header + row.replace(",0,2", ",0," + "9" * 65))
    with pytest.raises(ValueError, match="TOKEN_BOUND"):
        uc._read_capture(path, rows=1)


def test_money_literal_join_retains_selected_and_unselected_missingness():
    field = uc.money.MoneyField(
        "UC_VAL",
        np.array([0.0, 0.0, 40.0], dtype="<f8").tobytes(),
        bytes([1, 1, 2]),
        bytes([0, 0, 1]),
        bytes([9, 9, 0]),
    )
    values_ = uc._amount_observations(
        field, np.array([0, 1, 2], dtype="int64"), ["", "", "40"]
    )
    assert np.isnan(values_[:2]).all()
    assert values_[2] == 40
    # Downstream source selection cannot turn either missing source row into 0.
    np.testing.assert_array_equal(values_[[0, 2]], [np.nan, 40.0])
    assert np.isnan(values_[[1]][0])
    assert field.amounts.tolist() == [0.0, 0.0, 40.0]
    assert field.zero_origin.tolist() == [9, 9, 0]
    with pytest.raises(ValueError, match="SOURCE_VALIDITY"):
        uc._amount_observations(
            field, np.array([0, 1, 2], dtype="int64"), ["0", "", "40"]
        )


def _invented_family():
    ids = np.array([0, 1, 2, 3], dtype="int64")
    p = pd.DataFrame({"person_id": ids, "age": [14, 40, 14, 40]})
    for entity in US_SCHEMA.group_entities:
        p[US_SCHEMA.membership_column(entity)] = ids
    p[values.provenance.spine_source_id_column("person")] = [10, 11, 20, 21]
    p[values.provenance.support_channel_column("person")] = pd.array(
        ["acs", "acs", "asec", "asec"], dtype="string"
    )
    tables = {
        "person": p,
        **{
            e: pd.DataFrame(
                {e + "_id": ids, "fixture_source_" + e: np.ones(4, dtype="int64")}
            )
            for e in US_SCHEMA.group_entities
        },
    }
    source = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 3.0, 4.0, 5.0]), WeightKind.DESIGN)},
        pd.Series(["fixture"] * 4, dtype="string"),
    )
    clones = {}
    for entity, table in tables.items():
        parts = []
        for clone in (0, 1):
            part = table.copy(deep=True)
            part[values.provenance.support_source_id_column(entity)] = ids
            part[values.provenance.support_clone_index_column(entity)] = clone
            part[entity + "_id"] += clone * 10
            if entity == "person":
                for group in US_SCHEMA.group_entities:
                    part[US_SCHEMA.membership_column(group)] += clone * 10
            parts.append(part)
        clones[entity] = pd.concat(parts, ignore_index=True)
    receiving = Frame(
        clones,
        US_SCHEMA,
        {"household": Weights(np.ones(8), WeightKind.IMPORTANCE)},
        pd.Series(["fixture"] * 8, dtype="string"),
    )
    index = pd.Index(ids, name="person_id")
    origins = pd.DataFrame(
        {
            "person_id": ids,
            "native_person_id": [10, 11, 20, 21],
            "source": ["acs", "acs", "asec", "asec"],
        },
        index=index,
    )
    native = pd.DataFrame(
        {"unemployment_compensation": [np.nan, np.nan, np.nan, 0.0]}, index=index
    )
    reports = pd.DataFrame(
        {
            "survey_current_UC_VAL_origin": pd.array(
                [
                    "unresolved",
                    "modeled_from_current_asec",
                    "outside_reporting_universe",
                    "known_nonreceipt",
                ],
                dtype="string",
            )
        },
        index=index,
    )
    spec = values.GROUPS[0]
    features = pd.DataFrame(
        {"survey_predictor_age": [40.0]}, index=pd.Index([1], name="person_id")
    )
    matrix = model_input.encode_recipient_matrix(
        features, entity="person", entity_ids=np.array([1], dtype="<i8")
    )
    keep = np.array([False, False, False, True])
    group = values.GroupValues(spec, source.select(keep), pd.DataFrame(), matrix, keep)
    qualified = values.QualifiedSurveyAmounts(
        b"{}", {}, source, origins, native, reports, ("survey_predictor_age",), (group,)
    )
    draws = {spec.key: pd.DataFrame({spec.targets[0]: [12.0]}, index=features.index)}
    return qualified, receiving, draws


def test_clone_draw_sharing_keeps_source_unknown_and_source_zero_distinct():
    qualified, receiving, draws = _invented_family()
    columns = values.attach_columns(qualified, receiving, draws)
    amounts = columns["person", "unemployment_compensation"]
    np.testing.assert_array_equal(
        amounts.to_numpy(), [np.nan, 12, np.nan, 0, np.nan, 12, np.nan, 0]
    )
    assert (
        columns["person", "survey_current_UC_VAL_origin"].loc[2]
        == "outside_reporting_universe"
    )
    # Reordering receiving records cannot change the source-keyed assignment.
    # This fixture is owned here; no Frame copy/pickle lifecycle is needed.
    reordered = receiving
    reordered.person.sort_values("person_id", ascending=False, inplace=True)
    changed = values.attach_columns(qualified, reordered, draws)
    pd.testing.assert_series_equal(
        amounts.sort_index(),
        changed["person", "unemployment_compensation"].sort_index(),
    )


@pytest.mark.parametrize(
    "mutation",
    ["clone", "source", "native", "channel", "collision", "draw_axis", "draw_unknown"],
)
def test_clone_attachment_refuses_misalignment_or_ownership_collision(mutation):
    qualified, receiving, draws = _invented_family()
    if mutation in ("clone", "source", "native", "channel"):
        columns = {
            "clone": values.provenance.support_clone_index_column("person"),
            "source": values.provenance.support_source_id_column("person"),
            "native": values.provenance.spine_source_id_column("person"),
            "channel": values.provenance.support_channel_column("person"),
        }
        receiving.person.loc[0, columns[mutation]] = (
            "asec" if mutation == "channel" else 99
        )
    elif mutation == "collision":
        receiving.person["unemployment_compensation"] = np.nan
    elif mutation == "draw_axis":
        draws["unemployment"].index = pd.Index([0], name="person_id")
    else:
        draws["unemployment"].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="CURRENT_SURVEY_AMOUNTS"):
        values.attach_columns(qualified, receiving, draws)


def test_closed_groups_do_not_introduce_legacy_ss_or_puf_owned_amounts():
    assert values.selected_groups(("unemployment",)) == values.GROUPS[:1]
    assert tuple(raw for raw, _ in values.GROUPS[1].fields) == (
        "PHIP_VAL",
        "PMED_VAL",
        "POTC_VAL",
    )
    with pytest.raises(ValueError, match="GROUP"):
        values.selected_groups(("legacy_social_security",))
    with pytest.raises(ValueError, match="GROUP"):
        values.selected_groups(("health_costs", "unemployment"))


def test_amount_fragment_declarations_compile_with_explicit_parent_dependency():
    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph
    from microcosm.graph import (
        ArtifactOutput,
        Graph,
        Node,
        Owned,
        SourceRef,
        StructuralDelta,
        compile_graph,
    )

    qualified, receiving, _ = _invented_family()
    nodes = graph.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    create_id = values.predictors.host.survey_graph.CREATE_NODE
    prefix = (
        Node(
            create_id,
            "fixture.create@1",
            structural=StructuralDelta.CREATE,
            sources=("fixture",),
            outputs=tuple(
                Owned(e, c, str(receiving.table(e)[c].dtype))
                for e in receiving.entities
                for c in receiving.table(e)
            ),
        ),
        Node(
            graph.parent.attach.FILTER_NODE,
            "fixture.filter@1",
            base=create_id,
            structural=StructuralDelta.FILTER,
            mass="free",
        ),
        Node(
            graph.parent.attach.ATTACH_NODE,
            "fixture.attach@1",
            population=graph.parent.attach.FILTER_NODE,
            artifact_outputs=(
                ArtifactOutput("finalization", graph.parent.attach.FINALIZATION_TYPE),
            ),
        ),
    )
    compiled = compile_graph(
        Graph("us", (SourceRef("fixture", "raw-bytes-v1"),), (*prefix, *nodes))
    )
    assert (
        graph.parent.attach.ATTACH_NODE in compiled.predecessors[graph.PROJECTION_NODE]
    )
    assert (
        graph.PROJECTION_NODE
        in compiled.predecessors["survey_amounts.unemployment.donor"]
    )
    assert len(compiled.order) == 9


def test_origin_digest_preserves_overlapping_person_id_column_and_axis():
    table = pd.DataFrame(
        {"person_id": np.array([10, 11], dtype="int64"), "source": ["acs", "asec"]}
    ).set_index("person_id", drop=False)
    before = values.origin_digest(table)
    assert values.origin_digest(table.copy(deep=True)) == before
    changed_axis = table.copy(deep=True)
    changed_axis.index = pd.Index(np.array([12, 13], dtype="int64"), name="person_id")
    assert values.origin_digest(changed_axis) != before
    changed_column = table.copy(deep=True)
    changed_column.loc[10, "person_id"] = 99
    assert values.origin_digest(changed_column) != before
