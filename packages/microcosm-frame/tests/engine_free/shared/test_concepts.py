"""The engine-neutral concept schema: declared contracts and their invariants.

Example tests pin what the schema declares; Hypothesis properties check the
invariants every valid concept frame, alignment id and take-up seed must
satisfy, and that the frame validator catches each planted violation.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.concepts import (
    CONCEPT_BY_ID,
    CONCEPT_ENTITIES,
    CONCEPT_SCHEMA_VERSION,
    CONCEPTS,
    AlignmentRelation,
    CanonicalConceptKind,
    ConceptAlignment,
    ConceptFrameDeclaration,
    ContentBasis,
    IndexFamily,
    MonetaryHandling,
    ProvenanceClass,
    TemporalBasis,
    TransportRule,
    Unit,
    canonical_concept_kind,
    concept,
    concept_for_column,
    concept_schema_sha256,
    concepts_for_entity,
    derive_take_up_draws,
    split_for_transport,
    validate_concept_tables,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.paths import paths_for

PROPERTY = settings(max_examples=150, deadline=None)

#: chronicle/core.py ALLOWED_CONCEPT_RELATIONS at chronicle origin/main 505e0e7.
CHRONICLE_RELATIONS = {
    "exact",
    "broad_match",
    "narrow_match",
    "approximate",
    "source_label",
}
#: chronicle/concepts.py ConceptAlignment fields (origin/main 505e0e7), plus the
#: content key Chronicle's feeds carry on each concept_alignment record.
CHRONICLE_ALIGNMENT_FIELDS = {
    "canonical_concept",
    "source_concept",
    "relation",
    "fact_key",
    "source_record_id",
    "authority",
    "evidence_url",
    "evidence_notes",
    "legal_vintage",
    "concept_alignment_key",
}
#: A committed Chronicle consumer-feed fixture with real concept_alignment rows.
CHRONICLE_FEED = (
    paths_for("microcosm-build").tests
    / "fixtures"
    / "uk_target_reference_feed_rows.jsonl"
)

#: Pinned schema digest. A change to any concept's declared contract moves it:
#: bump CONCEPT_SCHEMA_VERSION when the change is deliberate, then re-pin.
SCHEMA_SHA256 = "d824f5e97d58292d62f65ffae60bc8654408711344b638d4e3caa95d6c2e8d1e"


class TestDeclaredSchema:
    def test_the_requested_topics_are_all_covered(self) -> None:
        required = {
            "fact:person.age",
            "fact:person.sex",
            "fact:person.partner_person_id",
            "fact:person.parent_1_person_id",
            "fact:person.parent_2_person_id",
            "fact:person.employment_income",
            "fact:person.nonfarm_self_employment_income",
            "fact:person.farm_self_employment_income",
            "fact:person.interest_income",
            "fact:person.dividend_income",
            "fact:person.rental_income",
            "fact:person.realized_capital_gains",
            "fact:person.liquid_financial_assets",
            "fact:person.private_pension_income",
            "fact:person.public_pension_income",
            "fact:person.usual_weekly_hours",
            "fact:person.weeks_worked",
            "fact:household.tenure",
            "fact:household.rent",
            "fact:household.mortgage_interest",
            "fact:household.mortgage_principal",
            "fact:household.property_tax",
            "fact:person.has_disability",
            "fact:person.educational_attainment",
            "fact:person.education_enrollment",
            "fact:person.take_up_seed",
        }
        assert required <= set(CONCEPT_BY_ID)

    def test_ids_are_unique_and_follow_the_primitive_grammar(self) -> None:
        ids = [item.id for item in CONCEPTS]
        assert len(ids) == len(set(ids))
        for item in CONCEPTS:
            assert canonical_concept_kind(item.id) is CanonicalConceptKind.PRIMITIVE
            assert item.id == f"fact:{item.entity}.{item.name}"
            assert item.entity in CONCEPT_ENTITIES

    def test_every_concept_declares_its_semantics(self) -> None:
        for item in CONCEPTS:
            assert item.label and item.definition
            assert item.dtype in ("float", "int", "bool", "str")
            assert item.period in ("year", "month", "point")
            assert isinstance(item.unit, Unit)
            assert isinstance(item.provenance, ProvenanceClass)
            assert isinstance(item.transport, TransportRule)

    def test_amounts_carry_monetary_handling_and_nothing_else_does(self) -> None:
        for item in CONCEPTS:
            is_money = item.unit is Unit.BASE_CURRENCY
            assert is_money == (item.monetary is not None), item.id
            if is_money:
                assert item.monetary.price_basis == "nominal"
                assert item.dtype == "float"
                assert (item.lower is None) == item.monetary.signed

    def test_categories_carry_a_closed_domain(self) -> None:
        for item in CONCEPTS:
            assert (item.unit is Unit.CATEGORY) == bool(item.domain), item.id

    def test_only_relationship_pointers_are_nullable(self) -> None:
        nullable = {item.id for item in CONCEPTS if item.nullable}
        pointers = {item.id for item in CONCEPTS if item.unit is Unit.PERSON_ID}
        assert nullable == pointers - {"fact:household.reference_person_id"}
        assert all(
            item.name.endswith("_person_id") for item in CONCEPTS if item.nullable
        )

    def test_transport_rules(self) -> None:
        dropped = {item.id for item in CONCEPTS if item.transport is TransportRule.DROP}
        assert dropped == {"fact:person.public_pension_income"}
        for item in CONCEPTS:
            if item.transport is TransportRule.QUANTILE_MAP:
                assert item.monetary is not None
            if item.provenance is ProvenanceClass.GENERATED:
                assert item.transport is TransportRule.CARRY

    def test_liquid_financial_assets_are_a_quantile_mapped_person_stock(
        self,
    ) -> None:
        item = concept("fact:person.liquid_financial_assets")
        assert (item.entity, item.dtype, item.unit) == (
            "person",
            "float",
            Unit.BASE_CURRENCY,
        )
        assert (item.period, item.temporal_basis) == (
            "point",
            TemporalBasis.REFERENCE_STATE,
        )
        assert item.provenance is ProvenanceClass.OBSERVED
        assert item.transport is TransportRule.QUANTILE_MAP
        assert item.monetary == MonetaryHandling(
            index_family=IndexFamily.CONSUMER_PRICES
        )
        assert (item.lower, item.upper, item.nullable) == (0.0, None, False)

    def test_liquid_financial_assets_are_the_only_amount_held_as_a_stock(
        self,
    ) -> None:
        stocks = {
            item.id
            for item in CONCEPTS
            if item.monetary is not None
            and item.temporal_basis is not TemporalBasis.ANNUAL_FLOW
        }
        assert stocks == {"fact:person.liquid_financial_assets"}

    def test_the_take_up_seed_is_generated_persistent_state(self) -> None:
        seed = concept("fact:person.take_up_seed")
        assert seed.provenance is ProvenanceClass.GENERATED
        assert (seed.lower, seed.upper, seed.upper_inclusive) == (0.0, 1.0, False)

    def test_lookups(self) -> None:
        assert concept_for_column("person", "age") is concept("fact:person.age")
        assert concept_for_column("person", "tenure") is None
        assert concept_for_column("household", "not_a_concept") is None
        assert {item.entity for item in concepts_for_entity("household")} == {
            "household"
        }
        with pytest.raises(KeyError, match="fact:person.nope"):
            concept("fact:person.nope")
        with pytest.raises(ValueError, match="tax_unit"):
            concepts_for_entity("tax_unit")

    def test_schema_digest_is_pinned(self) -> None:
        assert CONCEPT_SCHEMA_VERSION == 2
        assert concept_schema_sha256() == SCHEMA_SHA256

    def test_schema_digest_ignores_alignments(self) -> None:
        aligned = list(CONCEPTS)
        aligned[0] = replace(aligned[0], alignments=(_legal(),))
        assert concept_schema_sha256(aligned) == concept_schema_sha256()

    def test_schema_digest_moves_with_any_declared_field(self) -> None:
        base = concept_schema_sha256()
        assert concept_schema_sha256(CONCEPTS) == base
        changed = list(CONCEPTS)
        changed[0] = replace(changed[0], definition=changed[0].definition + " ")
        assert concept_schema_sha256(changed) != base
        assert concept_schema_sha256(reversed(CONCEPTS)) != base


class TestConceptValidation:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"id": "person.age"}, "fact:"),
            ({"id": "fact:tax_unit.age"}, "fact:"),
            ({"dtype": "double"}, "dtype"),
            ({"period": "eternity"}, "period"),
            ({"unit": Unit.BASE_CURRENCY}, "monetary"),
            ({"nullable": True}, "nullable"),
            ({"unit": Unit.PERSON_ID}, "int named"),
            ({"lower": 200.0}, "empty"),
            ({"transport": TransportRule.QUANTILE_MAP}, "quantile"),
        ],
    )
    def test_bad_declarations_are_refused(self, change, message) -> None:
        with pytest.raises(ValueError, match=message):
            replace(concept("fact:person.age"), **change)

    def test_generated_concepts_must_carry(self) -> None:
        with pytest.raises(ValueError, match="persistence"):
            replace(concept("fact:person.take_up_seed"), transport=TransportRule.DROP)

    def test_signed_amounts_are_unbounded_below(self) -> None:
        with pytest.raises(ValueError, match="Signed"):
            replace(concept("fact:person.rental_income"), lower=0.0)
        with pytest.raises(ValueError, match="lower bound 0"):
            replace(concept("fact:household.rent"), lower=None)

    def test_categories_need_snake_case_unique_values(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            replace(concept("fact:person.sex"), domain=("female", "female"))
        with pytest.raises(ValueError, match="snake_case"):
            replace(concept("fact:person.sex"), domain=("Female", "male"))


class TestCanonicalConceptIds:
    @pytest.mark.parametrize(
        ("concept_id", "kind"),
        [
            ("fact:person.age", CanonicalConceptKind.PRIMITIVE),
            ("us:statutes/26/32/c/2#earned_income", CanonicalConceptKind.LEGAL),
            ("us:statutes/26/62#input.wages", CanonicalConceptKind.LEGAL),
            (
                "nz:statutes/income_tax/core/taxable_income"
                "#input.income_tax_employment_income",
                CanonicalConceptKind.LEGAL,
            ),
            ("us-ca:statutes/rtc/17041#tax", CanonicalConceptKind.LEGAL),
            ("census_pep.resident_population", CanonicalConceptKind.STATISTICAL),
            ("irs_soi.adjusted_gross_income", CanonicalConceptKind.STATISTICAL),
        ],
    )
    def test_kinds(self, concept_id, kind) -> None:
        assert canonical_concept_kind(concept_id) is kind

    @pytest.mark.parametrize(
        "concept_id", ["", "fact:person", "fact:firm.size", "age", "US:x#y", "a b"]
    )
    def test_malformed_ids_are_refused(self, concept_id) -> None:
        with pytest.raises(ValueError):
            canonical_concept_kind(concept_id)

    @PROPERTY
    @given(
        jurisdiction=st.from_regex(r"[a-z]{2}(-[a-z0-9]{2,4})?", fullmatch=True),
        path=st.from_regex(r"[a-z0-9_]+(/[a-z0-9_]+){0,5}", fullmatch=True),
        name=st.from_regex(r"[a-z_][a-z0-9_]*(\.[a-z0-9_]+)?", fullmatch=True),
    )
    def test_every_rulespec_id_is_legal(self, jurisdiction, path, name) -> None:
        concept_id = f"{jurisdiction}:{path}#{name}"
        assert canonical_concept_kind(concept_id) is CanonicalConceptKind.LEGAL


def _legal(**change) -> ConceptAlignment:
    base = {
        "source_concept": "fact:person.age",
        "canonical_concept": "nz:statutes/nzs#age",
        "relation": AlignmentRelation.APPROXIMATE,
        "authority": "microcosm",
        "legal_vintage": "2026",
        "evidence_notes": "Age at the entitlement date, read as survey age.",
    }
    return ConceptAlignment(**{**base, **change})


class TestConceptAlignment:
    def test_relation_vocabulary_is_chronicles(self) -> None:
        assert {relation.value for relation in AlignmentRelation} == CHRONICLE_RELATIONS

    def test_record_uses_chronicles_field_names(self) -> None:
        alignment = _legal(fact_key="f1", concept_alignment_key="k1")
        record = alignment.to_dict()
        assert set(record) <= CHRONICLE_ALIGNMENT_FIELDS
        assert record["relation"] == "approximate"
        assert alignment.jurisdiction == "nz"
        assert ConceptAlignment.from_dict(record) == alignment

    def test_every_alignment_in_a_committed_chronicle_feed_loads(self) -> None:
        rows = [
            json.loads(line)
            for line in CHRONICLE_FEED.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [
            row["concept_alignment"] for row in rows if row.get("concept_alignment")
        ]
        assert len(records) > 100
        for record in records:
            alignment = ConceptAlignment.from_dict(record)
            assert alignment.to_dict() == record
            assert alignment.issues() == ()
        # Chronicle's source_label records may repeat the source id.
        assert any(
            record["source_concept"] == record["canonical_concept"]
            for record in records
        )

    def test_chronicles_own_checks_are_reported_not_raised(self) -> None:
        bare = ConceptAlignment(
            source_concept="sfpd.legal_pension.beneficiaries",
            canonical_concept="legal_pension_recipients",
            relation=AlignmentRelation.EXACT,
        )
        assert bare.canonical_kind is None
        assert bare.issues() == (
            "Exact source-to-canonical concept alignments need evidence.",
        )

    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"source_concept": "a", "relation": "exact"}, "lacks"),
            (
                {"source_concept": "a", "canonical_concept": "b", "relation": "x"},
                "is not a valid",
            ),
            (
                {
                    "source_concept": "a",
                    "canonical_concept": "b",
                    "relation": "exact",
                    "colour": "red",
                },
                "Unknown",
            ),
            (
                {"source_concept": "", "canonical_concept": "b", "relation": "exact"},
                "names",
            ),
        ],
    )
    def test_malformed_records_are_refused(self, record, message) -> None:
        with pytest.raises(ValueError, match=message):
            ConceptAlignment.from_dict(record)

    def test_a_concept_accepts_a_well_evidenced_legal_alignment(self) -> None:
        aligned = replace(concept("fact:person.age"), alignments=(_legal(),))
        assert aligned.contract()["alignments"] == [_legal().to_dict()]
        statistical = _legal(canonical_concept="census_pep.age", legal_vintage=None)
        replace(concept("fact:person.age"), alignments=(statistical,))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"source_concept": "fact:person.sex"}, "has source"),
            (
                {"canonical_concept": "fact:person.sex", "legal_vintage": None},
                "primitives",
            ),
            ({"canonical_concept": "legal_pension_recipients"}, "primitives"),
            ({"relation": AlignmentRelation.SOURCE_LABEL}, "no semantic relation"),
            ({"authority": None}, "no authority"),
            ({"evidence_notes": None}, "no evidence"),
            ({"legal_vintage": None}, "no legal vintage"),
        ],
    )
    def test_a_concepts_alignments_meet_stricter_rules(self, change, message) -> None:
        with pytest.raises(ValueError, match=message):
            replace(concept("fact:person.age"), alignments=(_legal(**change),))

    def test_a_concept_carries_each_alignment_once(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            replace(concept("fact:person.age"), alignments=(_legal(), _legal()))

    def test_no_concept_claims_an_alignment_yet(self) -> None:
        assert all(item.alignments == () for item in CONCEPTS)


class TestConceptFrameDeclaration:
    def test_transport_names_a_different_donor(self) -> None:
        declaration = ConceptFrameDeclaration(
            country="nz",
            currency="NZD",
            reference_year=2026,
            content_basis=ContentBasis.TRANSPORT,
            donor_country="us",
        )
        assert declaration.content_basis is ContentBasis.TRANSPORT
        with pytest.raises(ValueError, match="other than"):
            replace(declaration, donor_country="nz")
        with pytest.raises(ValueError, match="exactly for transport"):
            replace(declaration, donor_country=None)
        with pytest.raises(ValueError, match="exactly for transport"):
            replace(declaration, content_basis=ContentBasis.OWN_DATA)

    @pytest.mark.parametrize(
        "change", [{"currency": "nzd"}, {"currency": "NZ"}, {"country": "NZ"}]
    )
    def test_codes_are_validated(self, change) -> None:
        with pytest.raises(ValueError):
            ConceptFrameDeclaration(
                **{
                    "country": "uk",
                    "currency": "GBP",
                    "reference_year": 2025,
                    "content_basis": ContentBasis.OWN_DATA,
                    **change,
                }
            )


def _violations(tables) -> set[tuple[str, str]]:
    return {(item.column, item.code) for item in validate_concept_tables(tables)}


def _person_row(tables, row: int = 0) -> int:
    return tables["person"].index[row]


class TestFrameValidation:
    @PROPERTY
    @given(tables=concept_frames())
    def test_generated_frames_are_valid(self, tables) -> None:
        assert validate_concept_tables(tables) == ()

    @PROPERTY
    @given(tables=concept_frames(), data=st.data())
    def test_out_of_domain_categories_are_caught(self, tables, data) -> None:
        column = data.draw(
            st.sampled_from(["sex", "legal_marital_status", "education_enrollment"])
        )
        tables["person"].loc[_person_row(tables), column] = "not_a_category"
        assert (column, "domain") in _violations(tables)

    @PROPERTY
    @given(tables=concept_frames(), data=st.data())
    def test_out_of_bound_values_are_caught(self, tables, data) -> None:
        column, value = data.draw(
            st.sampled_from(
                [
                    ("age", 131),
                    ("age", -1),
                    ("usual_weekly_hours", 168.5),
                    ("weeks_worked", 54),
                    ("take_up_seed", 1.0),
                    ("employment_income", -0.01),
                    ("liquid_financial_assets", -0.01),
                ]
            )
        )
        person = tables["person"]
        if column == "weeks_worked":
            person.loc[_person_row(tables), "usual_weekly_hours"] = 10.0
        person.loc[_person_row(tables), column] = value
        assert (column, "bounds") in _violations(tables)

    @PROPERTY
    @given(tables=concept_frames())
    def test_nulls_are_caught_outside_pointers(self, tables) -> None:
        person = tables["person"].astype({"employment_income": "float64"})
        person.loc[person.index[0], "employment_income"] = np.nan
        tables["person"] = person
        assert ("employment_income", "null") in _violations(tables)

    def test_wrong_dtypes_are_caught(self) -> None:
        tables = _two_person_frame()
        tables["person"]["age"] = ["40", "38"]
        tables["person"]["has_disability"] = [0, 1]
        found = _violations(tables)
        assert ("age", "dtype") in found
        assert ("has_disability", "dtype") in found

    def test_pointer_violations_are_caught(self) -> None:
        tables = _two_person_frame()
        person = tables["person"]
        person["partner_person_id"] = pd.array([20, pd.NA], dtype="Int64")
        assert ("partner_person_id", "asymmetric") in _violations(tables)
        person["partner_person_id"] = pd.array([10, pd.NA], dtype="Int64")
        assert ("partner_person_id", "self") in _violations(tables)
        person["partner_person_id"] = pd.array([99, pd.NA], dtype="Int64")
        assert ("partner_person_id", "dangling") in _violations(tables)

    def test_cross_household_pointers_are_caught(self) -> None:
        tables = _two_person_frame(same_household=False)
        tables["person"]["parent_1_person_id"] = pd.array([pd.NA, 10], dtype="Int64")
        assert ("parent_1_person_id", "cross_household") in _violations(tables)

    def test_parent_violations_are_caught(self) -> None:
        tables = _two_person_frame()
        person = tables["person"]
        person["parent_2_person_id"] = pd.array([pd.NA, 10], dtype="Int64")
        assert ("parent_2_person_id", "parent_order") in _violations(tables)
        person["parent_1_person_id"] = pd.array([pd.NA, 10], dtype="Int64")
        assert ("parent_2_person_id", "same_parent") in _violations(tables)
        person["parent_2_person_id"] = pd.array([pd.NA, pd.NA], dtype="Int64")
        person["partner_person_id"] = pd.array([20, 10], dtype="Int64")
        assert ("partner_person_id", "partner_is_parent") in _violations(tables)
        person["partner_person_id"] = pd.array([pd.NA, pd.NA], dtype="Int64")
        person["parent_1_person_id"] = pd.array([20, 10], dtype="Int64")
        assert ("parent_1_person_id", "parent_cycle") in _violations(tables)

    def test_reference_person_must_be_a_member(self) -> None:
        tables = _two_person_frame(same_household=False)
        tables["household"]["reference_person_id"] = pd.array([20, 20], dtype="Int64")
        assert ("reference_person_id", "not_member") in _violations(tables)
        tables["household"]["reference_person_id"] = pd.array(
            [10, pd.NA], dtype="Int64"
        )
        assert ("reference_person_id", "null") in _violations(tables)

    def test_consistency_rules_are_caught(self) -> None:
        tables = _two_person_frame()
        person, household = tables["person"], tables["household"]
        person["education_enrollment"] = ["not_enrolled", "tertiary"]
        person["enrolled_full_time"] = [True, True]
        person["usual_weekly_hours"] = [40.0, 0.0]
        person["weeks_worked"] = [0, 0]
        household["tenure"] = ["owned_outright"]
        household["rent"] = [1200.0]
        household["mortgage_interest"] = [10.0]
        found = _violations(tables)
        assert ("enrolled_full_time", "inconsistent") in found
        assert ("usual_weekly_hours", "inconsistent") in found
        assert ("rent", "inconsistent") in found
        assert ("mortgage_interest", "inconsistent") in found

    @pytest.mark.parametrize(
        ("tenure", "column", "value"),
        [
            *[
                (tenure, "rent", 10.0)
                for tenure in ("owned_outright", "owned_with_mortgage", "rent_free")
            ],
            *[
                (tenure, column, 10.0)
                for tenure in (
                    "owned_outright",
                    "rented_public_authority",
                    "rented_nonprofit_social",
                    "rented_private",
                    "rent_free",
                )
                for column in ("mortgage_interest", "mortgage_principal")
            ],
        ],
    )
    def test_housing_costs_follow_tenure(self, tenure, column, value) -> None:
        tables = _two_person_frame()
        tables["household"]["tenure"] = [tenure]
        tables["household"][column] = [value]
        assert (column, "inconsistent") in _violations(tables)

    @pytest.mark.parametrize(
        ("tenure", "column"),
        [
            ("rented_private", "rent"),
            ("rented_public_authority", "rent"),
            ("owned_with_mortgage", "mortgage_interest"),
            ("owned_with_mortgage", "mortgage_principal"),
        ],
    )
    def test_housing_costs_that_fit_tenure_pass(self, tenure, column) -> None:
        tables = _two_person_frame()
        tables["household"]["tenure"] = [tenure]
        tables["household"][column] = [100.0]
        assert _violations(tables) == set()

    @pytest.mark.parametrize(("hours", "weeks"), [(40.0, 0), (0.0, 10)])
    def test_hours_and_weeks_are_zero_together(self, hours, weeks) -> None:
        tables = _two_person_frame()
        tables["person"]["usual_weekly_hours"] = [hours, 0.0]
        tables["person"]["weeks_worked"] = np.array([weeks, 0], dtype=np.int64)
        assert ("usual_weekly_hours", "inconsistent") in _violations(tables)

    def test_a_single_category_categorical_column_validates(self) -> None:
        tables = _two_person_frame()
        tables["person"]["sex"] = pd.Categorical(["female", "female"])
        tables["household"]["tenure"] = pd.Categorical(["owned_outright"])
        assert _violations(tables) == set()

    def test_unreadable_columns_are_reported_not_raised(self) -> None:
        tables = _two_person_frame()
        person, household = tables["person"], tables["household"]
        person["enrolled_full_time"] = pd.array([True, pd.NA], dtype="boolean")
        person["education_enrollment"] = pd.array(["tertiary", pd.NA], dtype="string")
        person["usual_weekly_hours"] = ["forty", "none"]
        person["weeks_worked"] = np.array([10, 0], dtype=np.int64)
        household["tenure"] = pd.array([pd.NA], dtype="string")
        household["rent"] = [100.0]
        found = _violations(tables)
        assert ("usual_weekly_hours", "dtype") in found
        assert ("enrolled_full_time", "null") in found
        assert ("tenure", "null") in found

    @pytest.mark.parametrize(
        ("column", "values"),
        [("person_id", [1.5, 2.0]), ("person_household_id", [1, None])],
    )
    def test_ids_must_be_non_null_integers(self, column, values) -> None:
        tables = _two_person_frame()
        tables["person"][column] = values
        with pytest.raises(ValueError, match="non-null integers"):
            validate_concept_tables(tables)

    def test_the_seed_must_be_stored_as_float64(self) -> None:
        tables = _two_person_frame()
        tables["person"]["take_up_seed"] = np.array([0.25, 0.5], dtype=np.float32)
        assert ("take_up_seed", "dtype") in _violations(tables)

    def test_structure_errors_raise(self) -> None:
        tables = _two_person_frame()
        with pytest.raises(ValueError, match="tables"):
            validate_concept_tables({"person": tables["person"]})
        tables["person"]["person_household_id"] = [1, 5]
        with pytest.raises(ValueError, match="unknown household"):
            validate_concept_tables(tables)

    def test_extension_columns_pass_unchecked(self) -> None:
        tables = _two_person_frame()
        tables["person"]["receives_snap"] = ["yes", object()]
        assert validate_concept_tables(tables) == ()


def _two_person_frame(same_household: bool = True) -> dict[str, pd.DataFrame]:
    person = pd.DataFrame(
        {
            "person_id": [10, 20],
            "person_household_id": [1, 1] if same_household else [1, 2],
            "age": np.array([40, 38], dtype=np.int64),
            "has_disability": [False, True],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1] if same_household else [1, 2],
            "reference_person_id": pd.array(
                [10] if same_household else [10, 20], dtype="Int64"
            ),
        }
    )
    return {"person": person, "household": household}


def _cycle_reference(parents: dict[int, list[int]], n: int) -> int:
    """Persons on a cycle or on a path between cycles, by reachability."""

    reach = [[False] * n for _ in range(n)]
    for child, targets in parents.items():
        for parent in targets:
            reach[child][parent] = True
    for middle in range(n):
        for start in range(n):
            if reach[start][middle]:
                for end in range(n):
                    if reach[middle][end]:
                        reach[start][end] = True
    on_cycle = {node for node in range(n) if reach[node][node]}
    return sum(
        1
        for node in range(n)
        if any(reach[source][node] or source == node for source in on_cycle)
        and any(reach[node][target] or target == node for target in on_cycle)
    )


class TestParentCycles:
    def test_only_the_cycle_is_counted_not_its_ancestors_or_descendants(self) -> None:
        # 0 and 1 name each other; 2 is their parent; 3 is their child.
        person = pd.DataFrame(
            {
                "person_id": [0, 1, 2, 3],
                "person_household_id": [1, 1, 1, 1],
                "parent_1_person_id": pd.array([1, 0, pd.NA, 0], dtype="Int64"),
                "parent_2_person_id": pd.array([2, pd.NA, pd.NA, pd.NA], dtype="Int64"),
            }
        )
        tables = {"person": person, "household": pd.DataFrame({"household_id": [1]})}
        (cycle,) = [
            item
            for item in validate_concept_tables(tables)
            if item.code == "parent_cycle"
        ]
        assert cycle.rows == 2

    @PROPERTY
    @given(
        n=st.integers(1, 7),
        data=st.data(),
    )
    def test_cycle_count_matches_a_reachability_reference(self, n, data) -> None:
        parents = {
            child: data.draw(
                st.lists(
                    st.integers(0, n - 1).filter(lambda value, c=child: value != c),
                    max_size=2,
                    unique=True,
                ),
                label=f"parents of {child}",
            )
            for child in range(n)
        }
        first = [values[0] if values else None for values in parents.values()]
        second = [values[1] if len(values) > 1 else None for values in parents.values()]
        person = pd.DataFrame(
            {
                "person_id": list(range(n)),
                "person_household_id": [1] * n,
                "parent_1_person_id": pd.array(first, dtype="Int64"),
                "parent_2_person_id": pd.array(second, dtype="Int64"),
            }
        )
        tables = {"person": person, "household": pd.DataFrame({"household_id": [1]})}
        rows = sum(
            item.rows
            for item in validate_concept_tables(tables)
            if item.code == "parent_cycle"
        )
        assert rows == _cycle_reference(parents, n)


class TestTransportSplit:
    @PROPERTY
    @given(tables=concept_frames())
    def test_split_drops_receipts_and_extensions_and_keeps_the_rest(
        self, tables
    ) -> None:
        tables["person"]["receives_snap"] = True
        tables["household"]["state_fips"] = 6
        kept, dropped = split_for_transport(tables)
        assert "public_pension_income" in dropped["person"]
        assert "receives_snap" in dropped["person"]
        assert "liquid_financial_assets" in kept["person"].columns
        assert dropped["household"] == ("state_fips",)
        for entity in CONCEPT_ENTITIES:
            for column in kept[entity].columns:
                item = concept_for_column(entity, column)
                assert item is None or item.transport is not TransportRule.DROP
                pd.testing.assert_series_equal(
                    kept[entity][column], tables[entity][column]
                )
        assert validate_concept_tables(kept) == ()

    @PROPERTY
    @given(tables=concept_frames())
    def test_split_is_idempotent(self, tables) -> None:
        once, _ = split_for_transport(tables)
        twice, dropped = split_for_transport(once)
        assert dropped == {"person": (), "household": ()}
        for entity in CONCEPT_ENTITIES:
            pd.testing.assert_frame_equal(once[entity], twice[entity])


seeds = st.lists(
    st.floats(min_value=0.0, max_value=1.0, exclude_max=True, allow_nan=False),
    max_size=50,
)
programs = st.text(min_size=1, max_size=20)


class TestTakeUpDraws:
    @PROPERTY
    @given(values=seeds, program=programs)
    def test_draws_lie_on_the_unit_interval_and_are_deterministic(
        self, values, program
    ) -> None:
        draws = derive_take_up_draws(values, program)
        assert draws.shape == (len(values),)
        assert np.all((draws >= 0.0) & (draws < 1.0))
        np.testing.assert_array_equal(draws, derive_take_up_draws(values, program))

    @PROPERTY
    @given(first=seeds, second=seeds, program=programs)
    def test_a_records_draw_does_not_depend_on_other_records(
        self, first, second, program
    ) -> None:
        together = derive_take_up_draws(first + second, program)
        np.testing.assert_array_equal(
            together,
            np.concatenate(
                [
                    derive_take_up_draws(first, program),
                    derive_take_up_draws(second, program),
                ]
            ),
        )

    @PROPERTY
    @given(value=st.floats(min_value=0.0, max_value=1.0, exclude_max=True))
    def test_program_keys_are_namespaced(self, value) -> None:
        assert (
            derive_take_up_draws([value], "us.snap")[0]
            != derive_take_up_draws([value], "nz.snap")[0]
        )

    def test_negative_zero_draws_like_zero(self) -> None:
        assert (
            derive_take_up_draws([-0.0], "us.snap")[0]
            == derive_take_up_draws([0.0], "us.snap")[0]
        )

    def test_a_float32_round_trip_changes_draws(self) -> None:
        seed = np.asarray([0.1234567891234])
        narrowed = seed.astype(np.float32).astype(np.float64)
        assert (
            derive_take_up_draws(seed, "us.snap")[0]
            != derive_take_up_draws(narrowed, "us.snap")[0]
        )

    def test_draws_are_uniform_and_programs_independent(self) -> None:
        seeds_ = np.random.default_rng(20260928).random(40_000)
        snap = derive_take_up_draws(seeds_, "us.snap")
        tanf = derive_take_up_draws(seeds_, "us.tanf")
        # Decile counts within 5 standard errors of 4,000.
        counts, _ = np.histogram(snap, bins=10, range=(0.0, 1.0))
        assert np.all(np.abs(counts - 4_000) < 5 * np.sqrt(4_000 * 0.9))
        assert abs(np.corrcoef(snap, tanf)[0, 1]) < 0.03
        assert abs(np.corrcoef(snap, seeds_)[0, 1]) < 0.03

    @pytest.mark.parametrize(
        ("values", "program", "message"),
        [
            ([0.5], "", "non-empty"),
            ([1.0], "us.snap", r"\[0, 1\)"),
            ([-0.1], "x", r"\[0, 1\)"),
        ],
    )
    def test_bad_input_is_refused(self, values, program, message) -> None:
        with pytest.raises(ValueError, match=message):
            derive_take_up_draws(values, program)


def test_concept_is_frozen() -> None:
    with pytest.raises(AttributeError):
        concept("fact:person.age").label = "x"  # type: ignore[misc]
