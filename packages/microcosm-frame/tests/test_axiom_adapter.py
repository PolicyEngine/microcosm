"""Axiom adapter behavior: lazy import, metadata, materialize, export.

The no-engine tests run everywhere (the load-bearing claim is that the
adapter module imports and constructs without ``axiom_rules_engine``, and
fails with a precise error only when an engine-backed method is called).
Engine-backed tests skip unless the package *and* its native dense extension
are importable — the engine is not on PyPI, so CI runs without it; build it
locally from an axiom-rules-engine checkout to exercise them.

``TestBelgianPilotSlice`` additionally needs a rulespec-be checkout, pointed
at by the ``POPULACE_RULESPEC_BE`` environment variable — it materializes
the real CIR 1992 article 130 rate scale and checks hand-computed 2025
liabilities.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    LinkSpec,
    RulesEngine,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.adapters.axiom import (
    BE_SCHEMA,
    AxiomEngine,
    AxiomEntityTableDataset,
    AxiomRelationBinding,
    _canonical_digest,
    _period_bounds,
    verify_axiom_materialization_receipt,
)

_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False
_TABLES_INSTALLED = importlib.util.find_spec("tables") is not None

needs_engine = pytest.mark.skipif(
    not _DENSE_AVAILABLE,
    reason="axiom_rules_engine (with the dense native extension) is not installed",
)
needs_tables = pytest.mark.skipif(
    not _TABLES_INSTALLED,
    reason="pytables (microcosm-frame[axiom]) is not installed",
)

FIXTURE_RULESPEC_ROOT = Path(__file__).parent / "fixtures" / "rulespec-zz"
FIXTURE_MODULE = FIXTURE_RULESPEC_ROOT / "zz/policies/tests/axiom_toy_country.yaml"
FIXTURE_RELATION_MODULE = (
    FIXTURE_RULESPEC_ROOT / "zz/policies/tests/axiom_toy_relation.yaml"
)
FIXTURE_DUPLICATE_RELATION_MODULE = (
    FIXTURE_RULESPEC_ROOT / "zz/policies/tests/axiom_toy_duplicate_relation.yaml"
)
FIXTURE_DUPLICATE_RELATION_KEY = (
    "zz:policies/tests/axiom_toy_duplicate_relation#relation.member_of_household:1:0"
)
FIXTURE_RULESPEC_ROOTS = (FIXTURE_RULESPEC_ROOT,)
RULESPEC_BE = os.environ.get("POPULACE_RULESPEC_BE")


def _toy_bundle(
    incomes=(5_000.0, 10_000.0, 20_000.0),
    exempt=(False, False, False),
    children=(0, 2, 1),
    rents=(7_200.0, 4_800.0),
) -> Frame:
    """Three persons in two households carrying the fixture's inputs."""
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [1, 1, 2],
            "toy_taxable_income": list(incomes),
            "toy_is_exempt": list(exempt),
            "toy_child_count": list(children),
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2], "toy_household_rent": list(rents)}
    )
    weights = {
        "household": Weights(values=np.array([1500.0, 900.0]), kind=WeightKind.DESIGN)
    }
    return Frame({"person": person, "household": household}, BE_SCHEMA, weights)


class TestLazyImport:
    def test_adapter_constructs_without_the_engine(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        assert isinstance(adapter, RulesEngine)

    def test_entity_schema_needs_no_engine(self) -> None:
        assert (
            AxiomEngine(
                FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS
            ).entity_schema()
            == BE_SCHEMA
        )

    def test_export_contract_needs_no_engine(self) -> None:
        contract = ExportContract(
            required=("person_id",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        assert adapter.export_contract() is contract

    @pytest.mark.skipif(_ENGINE_INSTALLED, reason="axiom engine is installed here")
    def test_engine_methods_describe_installation_when_missing(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        with pytest.raises(ImportError, match="axiom-rules-engine"):
            adapter.variable_metadata("toy_income_tax")


class TestConstruction:
    def test_requires_explicit_rulespec_roots(self) -> None:
        with pytest.raises(TypeError, match="rulespec_roots"):
            AxiomEngine(FIXTURE_MODULE)  # type: ignore[call-arg]

    def test_rejects_empty_rulespec_roots(self) -> None:
        with pytest.raises(ValueError, match="at least one explicit rulespec"):
            AxiomEngine(FIXTURE_MODULE, rulespec_roots=())

    @pytest.mark.parametrize(
        "roots", [FIXTURE_RULESPEC_ROOT, str(FIXTURE_RULESPEC_ROOT)]
    )
    def test_rejects_a_scalar_rulespec_root(self, roots) -> None:
        with pytest.raises(TypeError, match="non-empty sequence"):
            AxiomEngine(FIXTURE_MODULE, rulespec_roots=roots)

    def test_rejects_unknown_arithmetic(self) -> None:
        with pytest.raises(ValueError, match="arithmetic"):
            AxiomEngine(
                FIXTURE_MODULE,
                rulespec_roots=FIXTURE_RULESPEC_ROOTS,
                arithmetic="float32",
            )

    def test_rejects_entity_names_outside_the_schema(self) -> None:
        with pytest.raises(ValueError, match="undeclared frame entit"):
            AxiomEngine(
                FIXTURE_MODULE,
                rulespec_roots=FIXTURE_RULESPEC_ROOTS,
                entity_names={"person": "Person", "tax_unit": "TaxUnit"},
            )

    def test_default_entity_names_capitalize(self) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        assert adapter._entity_names == {"person": "Person", "household": "Household"}

    def test_relation_bindings_require_declared_entities(self) -> None:
        with pytest.raises(ValueError, match="undeclared frame entity"):
            AxiomEngine(
                FIXTURE_MODULE,
                rulespec_roots=FIXTURE_RULESPEC_ROOTS,
                relation_bindings={
                    "member_of_household:1:0": AxiomRelationBinding(
                        current_entity="tax_unit",
                        related_entity="person",
                        edge_table="person",
                        edge_current_id_column="person_tax_unit_id",
                        edge_related_id_column="person_id",
                    )
                },
            )

    def test_relation_bindings_require_typed_values(self) -> None:
        with pytest.raises(TypeError, match="AxiomRelationBinding"):
            AxiomEngine(
                FIXTURE_MODULE,
                rulespec_roots=FIXTURE_RULESPEC_ROOTS,
                relation_bindings={
                    "member_of_household:1:0": {
                        "current_entity": "household",
                        "related_entity": "person",
                    }
                },  # type: ignore[dict-item]
            )

    def test_forwards_exact_roots_and_entity_to_the_dense_loader(
        self, monkeypatch
    ) -> None:
        calls: list[dict[str, object]] = []

        class RecordingProgram:
            derived_metadata: tuple[object, ...] = ()

            @classmethod
            def from_file(cls, path, *, rulespec_roots, entity):
                calls.append(
                    {
                        "path": path,
                        "rulespec_roots": rulespec_roots,
                        "entity": entity,
                    }
                )
                return cls()

        class RecordingEngine:
            CompiledDenseProgram = RecordingProgram

        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        monkeypatch.setattr(adapter, "_import_engine", lambda: RecordingEngine)

        assert adapter._program("person").derived_metadata == ()
        assert calls == [
            {
                "path": FIXTURE_MODULE,
                "rulespec_roots": FIXTURE_RULESPEC_ROOTS,
                "entity": "Person",
            }
        ]

    def test_does_not_mask_rulespec_root_validation_errors(self, monkeypatch) -> None:
        class RejectingProgram:
            @classmethod
            def from_file(cls, path, *, rulespec_roots, entity):
                raise ValueError("repository root error: root must be canonical")

        class RejectingEngine:
            CompiledDenseProgram = RejectingProgram

        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        monkeypatch.setattr(adapter, "_import_engine", lambda: RejectingEngine)

        with pytest.raises(ValueError, match="root must be canonical"):
            adapter.variables()
        assert adapter._programs == {}


class _FakeDenseRelationBatch:
    def __init__(self, *, offsets, inputs) -> None:
        self.offsets = offsets
        self.inputs = inputs


class _RecordingRelationProgram:
    root_entity = "Household"
    root_inputs: tuple[str, ...] = ()

    def __init__(self, *, related_inputs=("toy_taxable_income",)) -> None:
        self.relations = [
            SimpleNamespace(
                key="member_of_household:1:0",
                name="member_of_household",
                current_slot=1,
                related_slot=0,
                related_inputs=tuple(related_inputs),
            )
        ]
        self.last_relations = None

    def execute(self, *, relations, outputs, **_kwargs):
        self.last_relations = relations
        relation = relations["member_of_household:1:0"]
        missing = {
            name
            for declaration in self.relations
            for name in declaration.related_inputs
        } - set(relation.inputs)
        if missing:
            raise ValueError(f"missing dense relation input(s): {sorted(missing)}")
        incomes = relation.inputs["toy_taxable_income"]
        totals = np.array(
            [
                incomes[relation.offsets[i] : relation.offsets[i + 1]].sum()
                for i in range(len(relation.offsets) - 1)
            ]
        )
        assert outputs == ["toy_household_income"]
        return {"outputs": {"toy_household_income": totals}}

    execute_f64 = execute


class _RecordingLookupProgram:
    root_entity = "Person"
    root_inputs: tuple[str, ...] = ()
    relations = [
        SimpleNamespace(
            key="member_of_household:0:1",
            name="member_of_household",
            current_slot=0,
            related_slot=1,
            related_inputs=("toy_household_rent",),
        )
    ]

    def execute(self, *, relations, outputs, **_kwargs):
        relation = relations["member_of_household:0:1"]
        assert relation.offsets.tolist() == [0, 1, 2, 3]
        assert outputs == ["toy_person_household_rent"]
        return {
            "outputs": {
                "toy_person_household_rent": relation.inputs["toy_household_rent"]
            }
        }

    execute_f64 = execute


def _relation_bundle(*, alternate_membership=None) -> Frame:
    # Interleave households so relation construction must perform a stable
    # group without assuming person-table order already matches the root.
    person = pd.DataFrame(
        {
            "person_id": [3, 1, 2],
            "person_household_id": [2, 1, 1],
            "toy_taxable_income": [20_000.0, 5_000.0, 10_000.0],
            "toy_is_eligible": [True, False, True],
        }
    )
    if alternate_membership is not None:
        person["explicit_relation_household_id"] = alternate_membership
    household = pd.DataFrame(
        {"household_id": [1, 2], "toy_household_rent": [100.0, 200.0]}
    )
    return Frame(
        {"person": person, "household": household},
        BE_SCHEMA,
        {"household": Weights(values=np.array([1.0, 1.0]), kind=WeightKind.DESIGN)},
    )


def _string_relation_bundle() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": ["p3", "p1", "p2"],
            "person_household_id": ["h2", "h1", "h1"],
            "toy_taxable_income": [20_000.0, 5_000.0, 10_000.0],
            "toy_is_eligible": [True, False, True],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ["h1", "h2"],
            "toy_household_rent": [100.0, 200.0],
        }
    )
    return Frame(
        {"person": person, "household": household},
        BE_SCHEMA,
        {"household": Weights(values=np.ones(2), kind=WeightKind.DESIGN)},
    )


def _string_toy_bundle() -> Frame:
    source = _toy_bundle()
    person = source.table("person").copy()
    person["person_id"] = ["p1", "p2", "p3"]
    person["person_household_id"] = ["h1", "h1", "h2"]
    household = source.table("household").copy()
    household["household_id"] = ["h1", "h2"]
    return Frame(
        {"person": person, "household": household},
        BE_SCHEMA,
        {"household": source.weights_for("household")},
    )


def _empty_link_bundle() -> tuple[Frame, EntitySchema]:
    schema = EntitySchema(
        group_entities=("household",),
        links=(
            LinkSpec(
                name="household_members",
                left_entity="household",
                right_entity="person",
            ),
        ),
    )
    source = _relation_bundle()
    frame = Frame(
        {
            "person": source.table("person"),
            "household": source.table("household"),
            # This is pandas' natural construction for an empty link table:
            # both ID columns infer object rather than the entity ID dtype.
            "household_members": pd.DataFrame(columns=["household_id", "person_id"]),
        },
        schema,
        {"household": source.weights_for("household")},
    )
    return frame, schema


def _relation_adapter(monkeypatch, *, binding=None, program=None) -> AxiomEngine:
    if binding is None:
        binding = AxiomRelationBinding(
            current_entity="household",
            related_entity="person",
            edge_table="person",
            edge_current_id_column="person_household_id",
            edge_related_id_column="person_id",
        )
    adapter = AxiomEngine(
        FIXTURE_MODULE,
        rulespec_roots=FIXTURE_RULESPEC_ROOTS,
        relation_bindings={"member_of_household:1:0": binding},
    )
    relation_program = program or _RecordingRelationProgram()
    monkeypatch.setattr(adapter, "_program", lambda _entity: relation_program)
    monkeypatch.setattr(
        adapter,
        "variable_metadata",
        lambda name: VariableMetadata(
            name=name, entity="household", dtype="float", period="year"
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_import_engine",
        lambda: SimpleNamespace(DenseRelationBatch=_FakeDenseRelationBatch),
    )
    return adapter


class TestRelationBindings:
    def test_ordinary_materialize_preserves_empty_and_duplicate_requests(
        self, monkeypatch
    ) -> None:
        program = _RecordingRelationProgram()
        adapter = _relation_adapter(monkeypatch, program=program)

        assert adapter.materialize(_relation_bundle(), [], period=2025) == {}
        outputs = adapter.materialize(
            _relation_bundle(),
            ["toy_household_income", "toy_household_income"],
            period=2025,
        )
        assert list(outputs) == ["toy_household_income"]
        np.testing.assert_allclose(outputs["toy_household_income"], [15_000, 20_000])

    def test_materializes_stable_dense_batch_and_receipts_exact_order(
        self, monkeypatch
    ) -> None:
        program = _RecordingRelationProgram()
        adapter = _relation_adapter(monkeypatch, program=program)
        outputs, receipt = adapter.materialize_with_receipt(
            _relation_bundle(), ["toy_household_income"], period=2025
        )

        np.testing.assert_allclose(outputs["toy_household_income"], [15_000, 20_000])
        relation = program.last_relations["member_of_household:1:0"]
        assert relation.offsets.tolist() == [0, 2, 3]
        assert relation.inputs["toy_taxable_income"].tolist() == [
            5_000.0,
            10_000.0,
            20_000.0,
        ]
        entity = receipt["entities"]["household"]
        evidence = entity["relations"]["member_of_household:1:0"]
        assert evidence["binding"]["current_entity"] == "household"
        assert evidence["binding"]["related_entity"] == "person"
        assert evidence["binding"]["edge_table"] == "person"
        assert evidence["binding"]["edge_current_id_column"] == "person_household_id"
        assert evidence["offsets"]["shape"] == [3]
        assert len(evidence["receipt_sha256"]) == 64
        assert len(receipt["input_frame_sha256"]) == 64
        assert len(receipt["receipt_sha256"]) == 64
        verify_axiom_materialization_receipt(_relation_bundle(), receipt, outputs)

        _, repeated = adapter.materialize_with_receipt(
            _relation_bundle(), ["toy_household_income"], period=2025
        )
        assert repeated == receipt

    def test_declared_relation_without_binding_fails_closed(self, monkeypatch) -> None:
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
        )
        program = _RecordingRelationProgram()
        monkeypatch.setattr(adapter, "_program", lambda _entity: program)
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="household", dtype="float", period="year"
            ),
        )
        with pytest.raises(ValueError, match="missing=.*member_of_household"):
            adapter.materialize(
                _relation_bundle(), ["toy_household_income"], period=2025
            )

    def test_binding_for_undeclared_program_relation_fails_closed(
        self, monkeypatch
    ) -> None:
        program = _RecordingRelationProgram()
        program.relations = []
        adapter = _relation_adapter(monkeypatch, program=program)
        with pytest.raises(ValueError, match="extra=.*member_of_household"):
            adapter.materialize(
                _relation_bundle(), ["toy_household_income"], period=2025
            )

    def test_duplicate_runtime_key_fails_closed_before_the_native_boundary(
        self, monkeypatch
    ) -> None:
        program = _RecordingRelationProgram()
        program.relations.append(
            SimpleNamespace(
                key="member_of_household:1:0",
                name="member_of_household",
                current_slot=1,
                related_slot=0,
                related_inputs=("toy_is_eligible",),
            )
        )
        adapter = _relation_adapter(monkeypatch, program=program)
        with pytest.raises(ValueError, match="cannot safely bind repeated relation"):
            adapter.materialize_with_receipt(
                _relation_bundle(), ["toy_household_income"], period=2025
            )
        assert program.last_relations is None

    def test_current_to_related_lookup_uses_explicit_edge(self, monkeypatch) -> None:
        program = _RecordingLookupProgram()
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:0:1": AxiomRelationBinding(
                    current_entity="person",
                    related_entity="household",
                    edge_table="person",
                    edge_current_id_column="person_id",
                    edge_related_id_column="person_household_id",
                )
            },
        )
        monkeypatch.setattr(adapter, "_program", lambda _entity: program)
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="person", dtype="float", period="year"
            ),
        )
        monkeypatch.setattr(
            adapter,
            "_import_engine",
            lambda: SimpleNamespace(DenseRelationBatch=_FakeDenseRelationBatch),
        )
        outputs, receipt = adapter.materialize_with_receipt(
            _relation_bundle(), ["toy_person_household_rent"], period=2025
        )
        np.testing.assert_allclose(
            outputs["toy_person_household_rent"], [200.0, 100.0, 100.0]
        )
        verify_axiom_materialization_receipt(_relation_bundle(), receipt, outputs)

    def test_string_entity_ids_project_by_position_and_receipt_as_text(
        self, monkeypatch
    ) -> None:
        program = _RecordingRelationProgram()
        adapter = _relation_adapter(monkeypatch, program=program)
        frame = _string_relation_bundle()

        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )

        np.testing.assert_allclose(outputs["toy_household_income"], [15_000, 20_000])
        entity = receipt["entities"]["household"]
        relation = entity["relations"]["member_of_household:1:0"]
        assert entity["current_ids"]["dtype"] == "string"
        assert relation["source_related_entity_ids"]["dtype"] == "string"
        assert relation["source_edge_current_ids"]["dtype"] == "string"
        verify_axiom_materialization_receipt(frame, receipt, outputs)

    def test_declared_link_table_can_supply_explicit_edges(self, monkeypatch) -> None:
        schema = EntitySchema(
            group_entities=("household",),
            links=(
                LinkSpec(
                    name="household_members",
                    left_entity="household",
                    right_entity="person",
                ),
            ),
        )
        source = _relation_bundle()
        frame = Frame(
            {
                "person": source.table("person"),
                "household": source.table("household"),
                "household_members": pd.DataFrame(
                    {
                        "household_id": [1, 1, 2],
                        "person_id": [1, 2, 3],
                    }
                ),
            },
            schema,
            {"household": source.weights_for("household")},
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            schema=schema,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="household_members",
                    edge_current_id_column="household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        program = _RecordingRelationProgram()
        monkeypatch.setattr(adapter, "_program", lambda _entity: program)
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="household", dtype="float", period="year"
            ),
        )
        monkeypatch.setattr(
            adapter,
            "_import_engine",
            lambda: SimpleNamespace(DenseRelationBatch=_FakeDenseRelationBatch),
        )

        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )
        np.testing.assert_allclose(outputs["toy_household_income"], [15_000, 20_000])
        relation = receipt["entities"]["household"]["relations"][
            "member_of_household:1:0"
        ]
        assert relation["binding"]["edge_table"] == "household_members"
        verify_axiom_materialization_receipt(frame, receipt, outputs)

        frame.link("household_members").loc[0, "person_id"] = 3
        with pytest.raises(ValueError, match="relation receipt"):
            verify_axiom_materialization_receipt(frame, receipt, outputs)

    def test_natural_empty_object_link_table_projects_zero_edges(
        self, monkeypatch
    ) -> None:
        frame, schema = _empty_link_bundle()
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            schema=schema,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="household_members",
                    edge_current_id_column="household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        program = _RecordingRelationProgram()
        monkeypatch.setattr(adapter, "_program", lambda _entity: program)
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="household", dtype="float", period="year"
            ),
        )
        monkeypatch.setattr(
            adapter,
            "_import_engine",
            lambda: SimpleNamespace(DenseRelationBatch=_FakeDenseRelationBatch),
        )

        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )

        np.testing.assert_allclose(outputs["toy_household_income"], [0, 0])
        relation = receipt["entities"]["household"]["relations"][
            "member_of_household:1:0"
        ]
        assert relation["source_edge_current_ids"]["dtype"] == "int64"
        assert relation["source_edge_current_ids"]["shape"] == [0]
        verify_axiom_materialization_receipt(frame, receipt, outputs)

    @pytest.mark.parametrize(
        ("memberships", "message"),
        [
            ([1, 1, 999], "current ids absent from 'household'"),
            ([1.0, 1.0, 2.0], "integer or string dtype"),
        ],
    )
    def test_invalid_explicit_membership_fails_closed(
        self, monkeypatch, memberships, message
    ) -> None:
        binding = AxiomRelationBinding(
            current_entity="household",
            related_entity="person",
            edge_table="person",
            edge_current_id_column="explicit_relation_household_id",
            edge_related_id_column="person_id",
        )
        adapter = _relation_adapter(monkeypatch, binding=binding)
        with pytest.raises(ValueError, match=message):
            adapter.materialize(
                _relation_bundle(alternate_membership=memberships),
                ["toy_household_income"],
                period=2025,
            )

    def test_current_row_with_no_related_edge_is_valid(self, monkeypatch) -> None:
        binding = AxiomRelationBinding(
            current_entity="household",
            related_entity="person",
            edge_table="person",
            edge_current_id_column="explicit_relation_household_id",
            edge_related_id_column="person_id",
        )
        adapter = _relation_adapter(monkeypatch, binding=binding)
        outputs, receipt = adapter.materialize_with_receipt(
            _relation_bundle(alternate_membership=[1, 1, 1]),
            ["toy_household_income"],
            period=2025,
        )
        np.testing.assert_allclose(outputs["toy_household_income"], [35_000, 0])
        relation = receipt["entities"]["household"]["relations"][
            "member_of_household:1:0"
        ]
        assert relation["offsets"]["shape"] == [3]

    def test_missing_related_input_fails_closed(self, monkeypatch) -> None:
        adapter = _relation_adapter(
            monkeypatch,
            program=_RecordingRelationProgram(related_inputs=("missing_income",)),
        )
        with pytest.raises(ValueError, match="missing_income"):
            adapter.materialize(
                _relation_bundle(), ["toy_household_income"], period=2025
            )

    def test_live_verifier_detects_input_and_receipt_tampering(
        self, monkeypatch
    ) -> None:
        adapter = _relation_adapter(monkeypatch)
        frame = _relation_bundle()
        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )
        verify_axiom_materialization_receipt(frame, receipt, outputs)

        frame.table("person").loc[0, "toy_taxable_income"] = 99_999.0
        with pytest.raises(ValueError, match="provided inputs|relation receipt"):
            verify_axiom_materialization_receipt(frame, receipt, outputs)

        clean_frame = _relation_bundle()
        forged = json.loads(json.dumps(receipt))
        forged["entities"]["household"]["requested_outputs"]["toy_household_income"][
            "values"
        ]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="receipt digest differs"):
            verify_axiom_materialization_receipt(clean_frame, forged, outputs)

        changed_outputs = {
            "toy_household_income": outputs["toy_household_income"].copy()
        }
        changed_outputs["toy_household_income"][0] += 1
        with pytest.raises(ValueError, match="live output.*differs"):
            verify_axiom_materialization_receipt(clean_frame, receipt, changed_outputs)

        forged = json.loads(json.dumps(receipt))
        forged["entities"]["household"]["requested_outputs"]["toy_household_income"][
            "values"
        ]["shape"] = [999]
        unsigned = {
            key: value for key, value in forged.items() if key != "receipt_sha256"
        }
        forged["receipt_sha256"] = _canonical_digest(unsigned)
        with pytest.raises(ValueError, match="cardinality differs"):
            verify_axiom_materialization_receipt(clean_frame, forged, outputs)

        forged = json.loads(json.dumps(receipt))
        forged["period"]["end"] = "2025-12-30"
        with pytest.raises(ValueError, match="receipt digest differs"):
            verify_axiom_materialization_receipt(clean_frame, forged, outputs)

    def test_executor_cannot_mutate_owned_root_snapshot(self, monkeypatch) -> None:
        class MutatingProgram:
            root_entity = "Person"
            root_inputs = ("toy_taxable_income",)
            relations = ()

            @staticmethod
            def execute(*, inputs, **_kwargs):
                inputs["toy_taxable_income"][0] = 999_999.0
                return {"outputs": {"toy_income_tax": np.zeros(3)}}

            execute_f64 = execute

        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        monkeypatch.setattr(adapter, "_program", lambda _entity: MutatingProgram())
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="person", dtype="float", period="year"
            ),
        )

        frame = _toy_bundle()
        with pytest.raises(ValueError, match="mutated an owned root-input"):
            adapter.materialize_with_receipt(frame, ["toy_income_tax"], period=2025)
        assert frame.table("person").loc[0, "toy_taxable_income"] == 5_000.0

    def test_executor_cannot_mutate_owned_relation_snapshot(self, monkeypatch) -> None:
        class MutatingRelationProgram(_RecordingRelationProgram):
            def execute(self, *, relations, **_kwargs):
                relation = relations["member_of_household:1:0"]
                relation.inputs["toy_taxable_income"][0] = 999_999.0
                return {"outputs": {"toy_household_income": np.zeros(2)}}

            execute_f64 = execute

        adapter = _relation_adapter(monkeypatch, program=MutatingRelationProgram())
        frame = _relation_bundle()
        with pytest.raises(ValueError, match="mutated relation.*input snapshot"):
            adapter.materialize_with_receipt(
                frame, ["toy_household_income"], period=2025
            )
        assert frame.table("person").loc[0, "toy_taxable_income"] == 20_000.0

    def test_live_frame_mutation_during_execution_is_refused(self, monkeypatch) -> None:
        frame = _toy_bundle()

        class AliasingProgram:
            root_entity = "Person"
            root_inputs = ("toy_taxable_income",)
            relations = ()

            @staticmethod
            def execute(*, inputs, **_kwargs):
                result = inputs["toy_taxable_income"] * 0.1
                frame.table("person").loc[0, "toy_taxable_income"] = 999_999.0
                return {"outputs": {"toy_income_tax": result}}

            execute_f64 = execute

        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        monkeypatch.setattr(adapter, "_program", lambda _entity: AliasingProgram())
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="person", dtype="float", period="year"
            ),
        )

        with pytest.raises(ValueError, match="provided inputs.*differ"):
            adapter.materialize_with_receipt(frame, ["toy_income_tax"], period=2025)

    def test_materialize_revalidates_mutated_frame(self, monkeypatch) -> None:
        adapter = _relation_adapter(monkeypatch)
        frame = _relation_bundle()
        frame.table("household").iloc[:] = (
            frame.table("household").iloc[::-1].to_numpy()
        )
        with pytest.raises(ValueError, match="must be sorted ascending"):
            adapter.materialize(frame, ["toy_household_income"], period=2025)

    def test_text_output_receipt_uses_canonical_values(self, monkeypatch) -> None:
        program = SimpleNamespace(
            root_entity="Person",
            root_inputs=(),
            relations=[],
            execute=lambda **_kwargs: {
                "outputs": {"toy_status": ["eligible", "ineligible", "eligible"]}
            },
        )
        program.execute_f64 = program.execute
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
        )
        monkeypatch.setattr(adapter, "_program", lambda _entity: program)
        monkeypatch.setattr(
            adapter,
            "variable_metadata",
            lambda name: VariableMetadata(
                name=name, entity="person", dtype="str", period="point"
            ),
        )
        outputs, receipt = adapter.materialize_with_receipt(
            _relation_bundle(), ["toy_status"], period=2025
        )
        assert outputs["toy_status"].tolist() == [
            "eligible",
            "ineligible",
            "eligible",
        ]
        identity = receipt["entities"]["person"]["requested_outputs"]["toy_status"][
            "values"
        ]
        assert identity["encoding"] == "canonical_json_utf8_v1"
        assert identity["dtype"] == "string"
        verify_axiom_materialization_receipt(_relation_bundle(), receipt, outputs)


@needs_engine
class TestRelationBindingsWithRealAxiom:
    def test_native_duplicate_relation_key_fails_before_pyo3_conversion(
        self,
    ) -> None:
        adapter = AxiomEngine(
            FIXTURE_DUPLICATE_RELATION_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                FIXTURE_DUPLICATE_RELATION_KEY: AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="person",
                    edge_current_id_column="person_household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        program = adapter._program("household")
        assert [item.key for item in program.relations] == [
            FIXTURE_DUPLICATE_RELATION_KEY,
            FIXTURE_DUPLICATE_RELATION_KEY,
        ]

        with pytest.raises(ValueError, match="cannot safely bind repeated relation"):
            adapter.materialize(
                _relation_bundle(), ["toy_household_income"], period=2025
            )

    def test_root_input_and_output_change_materialization_receipt(self) -> None:
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
        )
        baseline = _toy_bundle()
        changed = _toy_bundle(incomes=(9_000.0, 10_000.0, 20_000.0))
        baseline_outputs, baseline_receipt = adapter.materialize_with_receipt(
            baseline, ["toy_income_tax"], period=2025
        )
        changed_outputs, changed_receipt = adapter.materialize_with_receipt(
            changed, ["toy_income_tax"], period=2025
        )
        assert baseline_outputs["toy_income_tax"][0] == 500.0
        assert changed_outputs["toy_income_tax"][0] == 900.0
        assert (
            baseline_receipt["input_frame_sha256"]
            != changed_receipt["input_frame_sha256"]
        )
        assert baseline_receipt["receipt_sha256"] != changed_receipt["receipt_sha256"]
        verify_axiom_materialization_receipt(
            baseline, baseline_receipt, baseline_outputs
        )
        verify_axiom_materialization_receipt(changed, changed_receipt, changed_outputs)

    def test_household_sum_executes_in_the_real_dense_runtime(self) -> None:
        adapter = AxiomEngine(
            FIXTURE_RELATION_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="person",
                    edge_current_id_column="person_household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        outputs, receipt = adapter.materialize_with_receipt(
            _relation_bundle(), ["toy_household_income"], period=2025
        )
        np.testing.assert_allclose(outputs["toy_household_income"], [15_000, 20_000])
        assert receipt["entities"]["household"]["relations"]
        verify_axiom_materialization_receipt(_relation_bundle(), receipt, outputs)

    def test_real_dense_runtime_accepts_zero_cardinality_current_row(self) -> None:
        adapter = AxiomEngine(
            FIXTURE_RELATION_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="person",
                    edge_current_id_column="explicit_relation_household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        frame = _relation_bundle(alternate_membership=[1, 1, 1])
        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )
        np.testing.assert_allclose(outputs["toy_household_income"], [35_000, 0])
        verify_axiom_materialization_receipt(frame, receipt, outputs)

    def test_real_dense_runtime_accepts_natural_empty_link_table(self) -> None:
        frame, schema = _empty_link_bundle()
        adapter = AxiomEngine(
            FIXTURE_RELATION_MODULE,
            schema=schema,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="household_members",
                    edge_current_id_column="household_id",
                    edge_related_id_column="person_id",
                )
            },
        )

        outputs, receipt = adapter.materialize_with_receipt(
            frame, ["toy_household_income"], period=2025
        )

        np.testing.assert_allclose(outputs["toy_household_income"], [0, 0])
        verify_axiom_materialization_receipt(frame, receipt, outputs)

    def test_relation_side_input_is_part_of_dataset_input_surface(self) -> None:
        adapter = AxiomEngine(
            FIXTURE_RELATION_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            relation_bindings={
                "member_of_household:1:0": AxiomRelationBinding(
                    current_entity="household",
                    related_entity="person",
                    edge_table="person",
                    edge_current_id_column="person_household_id",
                    edge_related_id_column="person_id",
                )
            },
        )
        assert adapter.variables() == ["toy_taxable_income"]


class TestPeriodBounds:
    def test_year_as_int_and_str(self) -> None:
        assert _period_bounds(2025) == ("2025-01-01", "2025-12-31", "calendar_year")
        assert _period_bounds("2025") == ("2025-01-01", "2025-12-31", "calendar_year")

    def test_month(self) -> None:
        assert _period_bounds("2025-02") == ("2025-02-01", "2025-02-28", "month")

    def test_rejects_other_shapes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported period"):
            _period_bounds("2025-Q1")
        with pytest.raises(ValueError, match="Invalid month"):
            _period_bounds("2025-13")


@needs_engine
class TestVariableMetadata:
    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        return AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)

    def test_person_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("toy_income_tax")
        assert meta.entity == "person"
        assert meta.dtype == "float"
        assert meta.period == "year"

    def test_household_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("toy_housing_allowance")
        assert meta.entity == "household"
        assert meta.period == "year"

    def test_month_period_variable(self, adapter) -> None:
        assert adapter.variable_metadata("toy_monthly_benefit").period == "month"

    def test_unknown_variable_is_named(self, adapter) -> None:
        with pytest.raises(ValueError, match="not_a_variable"):
            adapter.variable_metadata("not_a_variable")

    def test_input_variable_refuses_fabricated_metadata(self, adapter) -> None:
        with pytest.raises(ValueError, match="input variable"):
            adapter.variable_metadata("toy_taxable_income")

    def test_variables_lists_inputs_not_outputs(self, adapter) -> None:
        names = adapter.variables()
        assert "toy_taxable_income" in names
        assert "toy_is_exempt" in names
        assert "toy_household_rent" in names  # household-scope input
        assert "toy_income_tax" not in names  # derived output
        assert names == sorted(names)


@needs_engine
class TestMaterialize:
    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        return AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)

    def test_person_values_row_aligned_and_hand_computed(self, adapter) -> None:
        bundle = _toy_bundle()
        results = adapter.materialize(bundle, ["toy_income_tax"], period=2025)
        # 5,000 * 10% = 500; 10,000 * 10% = 1,000;
        # 10,000 * 10% + 10,000 * 25% = 3,500.
        np.testing.assert_allclose(results["toy_income_tax"], [500.0, 1_000.0, 3_500.0])

    def test_relation_free_string_ids_materialize_and_verify(self, adapter) -> None:
        bundle = _string_toy_bundle()
        outputs, receipt = adapter.materialize_with_receipt(
            bundle, ["toy_income_tax"], period=2025
        )

        np.testing.assert_allclose(outputs["toy_income_tax"], [500.0, 1_000.0, 3_500.0])
        assert receipt["entities"]["person"]["current_ids"]["dtype"] == "string"
        verify_axiom_materialization_receipt(bundle, receipt, outputs)

    def test_bool_column_drives_the_exemption_predicate(self, adapter) -> None:
        bundle = _toy_bundle(exempt=(True, False, True))
        results = adapter.materialize(bundle, ["toy_income_tax"], period=2025)
        np.testing.assert_allclose(results["toy_income_tax"], [0.0, 1_000.0, 0.0])

    def test_household_values_align_to_the_household_table(self, adapter) -> None:
        bundle = _toy_bundle()
        results = adapter.materialize(
            bundle, ["toy_income_tax", "toy_housing_allowance"], period=2025
        )
        assert results["toy_income_tax"].shape == (bundle.n("person"),)
        assert results["toy_housing_allowance"].shape == (bundle.n("household"),)
        np.testing.assert_allclose(results["toy_housing_allowance"], [1_200.0, 0.0])

    def test_integer_column_feeds_count_inputs(self, adapter) -> None:
        bundle = _toy_bundle(children=(0, 2, 1))
        results = adapter.materialize(bundle, ["toy_monthly_benefit"], period=2025)
        np.testing.assert_allclose(results["toy_monthly_benefit"], [0.0, 200.0, 100.0])

    def test_f64_arithmetic_matches_decimal(self) -> None:
        fast = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            arithmetic="f64",
        )
        bundle = _toy_bundle()
        results = fast.materialize(bundle, ["toy_income_tax"], period=2025)
        np.testing.assert_allclose(results["toy_income_tax"], [500.0, 1_000.0, 3_500.0])

    def test_wrong_bundle_entities_are_refused(self, adapter) -> None:
        person = pd.DataFrame(
            {"person_id": [1], "person_family_id": [1], "toy_taxable_income": [1.0]}
        )
        family = pd.DataFrame({"family_id": [1]})
        bundle = Frame(
            {"person": person, "family": family},
            EntitySchema(group_entities=("family",)),
            {"family": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="requires the schema entities"):
            adapter.materialize(bundle, ["toy_income_tax"], period=2025)

    def test_object_column_is_refused_with_the_column_named(self, adapter) -> None:
        bundle = _toy_bundle()
        person = bundle.table("person").copy()
        person["toy_taxable_income"] = person["toy_taxable_income"].astype(object)
        broken = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        with pytest.raises(ValueError, match="toy_taxable_income"):
            adapter.materialize(broken, ["toy_income_tax"], period=2025)


@needs_engine
@needs_tables
class TestWriteDataset:
    def test_round_trips_and_carries_household_weight(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        bundle = _toy_bundle()
        path = tmp_path / "toy.h5"
        adapter.write_dataset(bundle, path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.time_period == 2025
        assert reloaded.household["household_weight"].tolist() == [1500.0, 900.0]
        assert reloaded.person["toy_taxable_income"].tolist() == [
            5_000.0,
            10_000.0,
            20_000.0,
        ]

    def test_typed_weights_overwrite_a_stale_weight_column(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        bundle = _toy_bundle()
        household = bundle.table("household").copy()
        household["household_weight"] = [999.0, 999.0]
        stale = Frame(
            {"person": bundle.table("person"), "household": household},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        path = tmp_path / "stale.h5"
        adapter.write_dataset(stale, path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.household["household_weight"].tolist() == [1500.0, 900.0]

    def test_formula_owned_column_blocks_the_write(self, tmp_path) -> None:
        adapter = AxiomEngine(FIXTURE_MODULE, rulespec_roots=FIXTURE_RULESPEC_ROOTS)
        bundle = _toy_bundle()
        person = bundle.table("person").copy()
        person["toy_income_tax"] = [0.0, 0.0, 0.0]  # persisted engine output
        poisoned = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        path = tmp_path / "poisoned.h5"
        with pytest.raises(ValueError, match="toy_income_tax"):
            adapter.write_dataset(poisoned, path, period=2025)
        assert not path.exists()

    def test_missing_required_column_blocks_the_write(self, tmp_path) -> None:
        contract = ExportContract(
            required=("definitely_absent",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        path = tmp_path / "missing.h5"
        with pytest.raises(ValueError, match="definitely_absent"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_defaults_broadcast_onto_the_owning_table(self, tmp_path) -> None:
        contract = ExportContract(
            required=("toy_default_flag",),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
            defaults={"toy_default_flag": 0.0},
        )
        path = tmp_path / "defaulted.h5"
        adapter.write_dataset(_toy_bundle(), path, period=2025)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.person["toy_default_flag"].tolist() == [0.0, 0.0, 0.0]

    def test_closed_contract_rejects_unexpected_columns(self, tmp_path) -> None:
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
            closed=True,
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        path = tmp_path / "closed.h5"
        with pytest.raises(ValueError, match="unexpected"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_forbidden_column_blocks_the_write(self, tmp_path) -> None:
        contract = ExportContract(
            required=(),
            forbidden=("toy_child_count",),
            optional=(),
            formula_owned_excluded=(),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        path = tmp_path / "forbidden.h5"
        with pytest.raises(ValueError, match="toy_child_count"):
            adapter.write_dataset(_toy_bundle(), path, period=2025)
        assert not path.exists()

    def test_contract_formula_owned_exclusion_blocks_non_engine_column(
        self, tmp_path
    ) -> None:
        # legacy_output is not a derived rule of the module, so only the
        # contract's formula_owned_excluded list can catch it — the exact
        # case the shared contract tests never exercise.
        bundle = _toy_bundle()
        person = bundle.person.copy()
        person["legacy_output"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(),
            formula_owned_excluded=("legacy_output",),
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        path = tmp_path / "contract_formula_blocked.h5"
        with pytest.raises(ValueError, match="legacy_output"):
            adapter.write_dataset(rebuilt, path, period=2025)
        assert not path.exists()

    def test_formula_owned_exclusion_applies_under_a_closed_contract(
        self, tmp_path
    ) -> None:
        # Listing the column as optional in a closed contract must not
        # readmit it: formula_owned_excluded wins.
        bundle = _toy_bundle()
        person = bundle.person.copy()
        person["legacy_output"] = [1.0, 0.0, 1.0]
        rebuilt = Frame(
            {"person": person, "household": bundle.table("household")},
            BE_SCHEMA,
            {"household": bundle.weights_for("household")},
        )
        contract = ExportContract(
            required=(),
            forbidden=(),
            optional=(
                "toy_taxable_income",
                "toy_is_exempt",
                "toy_child_count",
                "toy_household_rent",
                "legacy_output",
            ),
            formula_owned_excluded=("legacy_output",),
            closed=True,
        )
        adapter = AxiomEngine(
            FIXTURE_MODULE,
            rulespec_roots=FIXTURE_RULESPEC_ROOTS,
            contract=contract,
        )
        path = tmp_path / "closed_formula_blocked.h5"
        with pytest.raises(ValueError, match="legacy_output"):
            adapter.write_dataset(rebuilt, path, period=2025)
        assert not path.exists()


class TestAxiomEntityTableDataset:
    def test_requires_exactly_one_construction_mode(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="tables and time_period"):
            AxiomEntityTableDataset()
        with pytest.raises(ValueError, match="not both"):
            AxiomEntityTableDataset(
                tables={"person": pd.DataFrame({"person_id": [1]})},
                time_period=2025,
                file_path=tmp_path / "x.h5",
            )

    def test_missing_file_is_named(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="nothing.h5"):
            AxiomEntityTableDataset(file_path=tmp_path / "nothing.h5")

    @needs_tables
    def test_save_and_reload_without_an_engine(self, tmp_path) -> None:
        tables = {
            "person": pd.DataFrame({"person_id": [1, 2], "age": [40.0, 8.0]}),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        path = tmp_path / "plain.h5"
        AxiomEntityTableDataset(tables=tables, time_period=2025).save(path)
        reloaded = AxiomEntityTableDataset(file_path=path)
        assert reloaded.time_period == 2025
        assert reloaded.person["age"].tolist() == [40.0, 8.0]
        assert reloaded.table("household")["household_id"].tolist() == [1]

    def test_unknown_attribute_is_an_attribute_error(self, tmp_path) -> None:
        dataset = AxiomEntityTableDataset(
            tables={"person": pd.DataFrame({"person_id": [1]})}, time_period=2025
        )
        with pytest.raises(AttributeError, match="tax_unit"):
            _ = dataset.tax_unit


@needs_engine
@pytest.mark.skipif(
    RULESPEC_BE is None,
    reason="set POPULACE_RULESPEC_BE to a rulespec-be checkout to run",
)
class TestBelgianPilotSlice:
    """The microcosm#260 smoke test: real CIR 1992 rules, hand-computed values.

    2025 article 130 brackets: 25% to 16,320; 40% to 28,800; 45% to 49,840;
    50% above. Hand computation for taxable incomes 10,000 / 30,000 / 60,000:

    - 10,000 * 25% = 2,500
    - 16,320 * 25% + (30,000 - 16,320) * 40% = 4,080 + 5,472 ... exceeds the
      28,800 threshold, so: 4,080 + 12,480 * 40% + 1,200 * 45% = 9,612
    - 4,080 + 4,992 + 21,040 * 45% + 10,160 * 50% = 23,620
    """

    @pytest.fixture(scope="class")
    def adapter(self) -> AxiomEngine:
        module = Path(RULESPEC_BE) / "be/statutes/income_tax/individual/rate_scale.yaml"
        return AxiomEngine(
            module,
            rulespec_roots=(Path(RULESPEC_BE),),
        )

    def _be_bundle(self) -> Frame:
        person = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [1, 1, 2],
                "belgium_pit_taxable_income": [10_000.0, 30_000.0, 60_000.0],
            }
        )
        household = pd.DataFrame({"household_id": [1, 2]})
        weights = {
            "household": Weights(
                values=np.array([1200.0, 800.0]), kind=WeightKind.DESIGN
            )
        }
        return Frame({"person": person, "household": household}, BE_SCHEMA, weights)

    def test_article_130_base_tax_reproduces_hand_computed_values(
        self, adapter
    ) -> None:
        results = adapter.materialize(
            self._be_bundle(), ["belgium_pit_article_130_base_tax"], period=2025
        )
        np.testing.assert_allclose(
            results["belgium_pit_article_130_base_tax"],
            [2_500.0, 9_612.0, 23_620.0],
        )

    def test_metadata_resolves_the_statutory_variable(self, adapter) -> None:
        meta = adapter.variable_metadata("belgium_pit_article_130_base_tax")
        assert meta.entity == "person"
        assert meta.dtype == "float"
        assert meta.period == "year"

    def test_taxable_income_is_an_input(self, adapter) -> None:
        assert "belgium_pit_taxable_income" in adapter.variables()
