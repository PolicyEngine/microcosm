"""UK weighted-integrity gates: input-mass parity + QRF tail concentration.

Increment 4 of the UK parity plan (#609): both gates reuse the shared
implementations in ``populace.build.gates`` and add the national-table
evidence plumbing plus the universal reviewed-exclusion discipline
(mandatory reason, dormant reported, stale fails).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
)
from populace.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    load_uk_input_mass_reference,
    load_uk_reviewed_exclusion_register,
    uk_dataset_input_mass_totals,
    uk_input_mass_parity_gate,
    uk_qrf_tail_concentration_columns,
    uk_qrf_tail_concentration_gate,
)


def _dataset(
    *,
    n: int = 4,
    weights: list[float] | np.ndarray | None = None,
    person_columns: dict[str, object] | None = None,
):
    if weights is None:
        weights = np.ones(n, dtype=float)
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    person = {
        "person_id": np.arange(101, 101 + n, dtype=np.int64),
        "person_household_id": household_ids,
        "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
        "employment_income": np.arange(1, n + 1, dtype=float),
    }
    person.update(person_columns or {})
    return SimpleNamespace(
        person=pd.DataFrame(person),
        benunit=pd.DataFrame({"benunit_id": np.arange(201, 201 + n, dtype=np.int64)}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": np.asarray(weights, dtype=float),
                "council_tax": np.full(n, 10.0),
            }
        ),
    )


def _reference(totals, **overrides) -> UKInputMassReference:
    fields = {
        "filename": "enhanced_frs_2023_24.h5",
        "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
        "sha256": "a" * 64,
        "vintage": "2023_24",
    }
    fields.update(overrides)
    return UKInputMassReference(totals=totals, **fields)


def _policy(**overrides) -> UKInputMassParityPolicy:
    fields = {"relative_tolerance": 0.5, "minimum_reference_total": 0.0}
    fields.update(overrides)
    return UKInputMassParityPolicy(**fields)


def test_dataset_totals_broadcast_household_weights_by_membership() -> None:
    dataset = _dataset(weights=[2.0, 1.0, 1.0, 1.0])

    totals = uk_dataset_input_mass_totals(dataset)

    # person values 1..4 under weights [2,1,1,1] -> 2 + 2 + 3 + 4.
    assert totals["person.employment_income"] == 11.0
    assert totals["household.council_tax"] == 50.0
    assert "household.household_weight" not in totals
    assert "person.person_household_id" not in totals


def test_dataset_totals_fail_closed_on_memberless_benunits() -> None:
    dataset = _dataset()
    dataset.benunit = pd.DataFrame({"benunit_id": [201, 202, 203, 204, 999]})

    with pytest.raises(ValueError, match="no member persons"):
        uk_dataset_input_mass_totals(dataset)


def test_dataset_totals_fail_closed_on_benunit_split_across_households() -> None:
    """A split benunit has no single household weight to inherit."""

    dataset = _dataset()
    # Person 4 keeps household 4 but joins benunit 201, which sits in
    # household 1: benunit 201 would silently take whichever came first.
    dataset.person.loc[3, "person_benunit_id"] = 201

    with pytest.raises(ValueError, match="exactly one household"):
        uk_dataset_input_mass_totals(dataset)


def test_zeroed_input_column_fails_by_name_at_any_tolerance() -> None:
    dataset = _dataset(person_columns={"employment_income": [0.0, 0.0, 0.0, 0.0]})
    reference = _reference({"person.employment_income": 10.0})

    gate = uk_input_mass_parity_gate(
        uk_dataset_input_mass_totals(dataset),
        reference,
        policy=_policy(relative_tolerance=1e9),
    )

    assert gate.name == "input_mass_parity"
    assert not gate.passed
    assert "person.employment_income" in gate.failures[0]
    assert "mass is zero" in gate.failures[0]


def test_material_mass_loss_fails_and_within_tolerance_passes() -> None:
    reference = _reference({"person.employment_income": 10.0})
    lost = uk_input_mass_parity_gate(
        {"person.employment_income": 0.01},
        reference,
        policy=_policy(),
    )
    kept = uk_input_mass_parity_gate(
        {"person.employment_income": 9.0, "person.pension_income": 5.0},
        reference,
        policy=_policy(),
    )

    assert not lost.passed
    assert "-99.9%" in lost.failures[0]
    assert kept.passed
    # Candidate-only columns are reported, never failed.
    assert kept.details["candidate_only_columns"] == ["person.pension_income"]


def test_input_mass_reference_identity_is_recorded() -> None:
    reference = _reference({"person.employment_income": 10.0})

    gate = uk_input_mass_parity_gate(
        {"person.employment_income": 10.0},
        reference,
        policy=_policy(),
    )

    assert gate.details["reference_identity"] == {
        "filename": "enhanced_frs_2023_24.h5",
        "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
        "sha256": "a" * 64,
        "vintage": "2023_24",
    }


def test_input_mass_exclusion_discipline_live_stale_dormant() -> None:
    reason = "Seeded reviewed loss for the fixture."
    reference = _reference({"person.employment_income": 10.0, "person.tiny_layer": 0.5})
    policy = _policy(
        minimum_reference_total=1.0,
        reviewed_exclusions={
            "person.employment_income": reason,
            "person.tiny_layer": reason,
            "person.never_shipped": reason,
        },
    )

    live = uk_input_mass_parity_gate(
        {"person.employment_income": 0.0},
        reference,
        policy=policy,
    )
    stale = uk_input_mass_parity_gate(
        {"person.employment_income": 10.0},
        reference,
        policy=policy,
    )

    # A live exclusion suppresses the zeroed-column failure and is recorded.
    assert live.passed
    assert live.details["reviewed_exclusions"]["person.employment_income"] == reason
    # Below-floor and absent-from-reference entries are dormant, not failing.
    assert live.details["dormant_exclusions"] == [
        "person.never_shipped",
        "person.tiny_layer",
    ]
    # A column now within tolerance is a rotted register entry and fails.
    assert not stale.passed
    assert stale.details["stale_exclusions"] == ["person.employment_income"]
    assert "Stale reviewed input-mass exclusions" in stale.failures[0]


def test_input_mass_policy_and_reference_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _policy(relative_tolerance=-0.1)
    with pytest.raises(ValueError, match="need reasons"):
        _policy(reviewed_exclusions={"person.employment_income": "  "})
    with pytest.raises(ValueError, match="lowercase sha256"):
        _reference({"person.employment_income": 1.0}, sha256="nope")
    with pytest.raises(ValueError, match="non-empty mapping"):
        _reference({})
    with pytest.raises(ValueError, match="finite"):
        _reference({"person.employment_income": float("nan")})


def test_qrf_surface_is_derived_from_the_source_manifest() -> None:
    from populace.build.uk_runtime.hmrc_source_contract import (
        uk_hmrc_weighted_qrf_output_columns,
    )

    declared = uk_hmrc_weighted_qrf_output_columns()

    # Stage-1 outputs come first and match the executable runtime surface;
    # stage-2 declares the FRS-only fill columns minus reviewed absences.
    assert declared[: len(SPI_INCOME_QRF_OUTPUT_COLUMNS)] == (
        SPI_INCOME_QRF_OUTPUT_COLUMNS
    )
    assert set(declared[len(SPI_INCOME_QRF_OUTPUT_COLUMNS) :]) <= set(
        FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    )
    assert len(declared) == len(set(declared))


def test_qrf_columns_check_every_declared_output_regardless_of_density() -> None:
    n = 6
    dense = np.full(n, 5.0)  # 100% nonzero: the US 5% cutoff would skip it.
    dataset = _dataset(n=n, person_columns={"self_employment_income": dense})

    values, weights, surface = uk_qrf_tail_concentration_columns(dataset)

    assert "self_employment_income" in values
    assert values["self_employment_income"].shape == (n,)
    assert weights["self_employment_income"].shape == (n,)
    assert surface["checked_columns"] == ["self_employment_income"]
    assert "density_filter" in surface
    # Declared outputs the fixture does not carry are reported, not invented.
    assert "dividend_income" in surface["absent_columns"]
    assert surface["declared_qrf_outputs"] >= 47


def test_declared_absent_qrf_output_is_a_named_gate_failure() -> None:
    values, weights, surface = uk_qrf_tail_concentration_columns(
        _dataset(),
        output_columns=("declared_but_absent",),
    )

    assert set(values) == {"declared_but_absent"}
    assert set(weights) == {"declared_but_absent"}
    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert not gate.passed
    assert gate.details["columns_checked"] == 0
    assert gate.details["surface"]["absent_columns"] == ["declared_but_absent"]
    assert "declared_but_absent" in gate.failures[0]
    assert "absent" in gate.failures[0]


def test_declared_nonnumeric_qrf_output_is_a_named_gate_failure() -> None:
    values, weights, surface = uk_qrf_tail_concentration_columns(
        _dataset(person_columns={"declared_nonnumeric": ["x"] * 4}),
        output_columns=("declared_nonnumeric",),
    )

    assert set(values) == {"declared_nonnumeric"}
    assert set(weights) == {"declared_nonnumeric"}
    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert not gate.passed
    assert gate.details["columns_checked"] == 0
    assert gate.details["surface"]["non_numeric_columns"] == ["declared_nonnumeric"]
    assert "declared_nonnumeric" in gate.failures[0]
    assert "not numeric" in gate.failures[0]


def test_concentrated_qrf_column_fails_by_name() -> None:
    n = 10
    concentrated = np.ones(n)
    concentrated[0] = 1_000.0
    dataset = _dataset(n=n, person_columns={"self_employment_income": concentrated})
    values, weights, surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert gate.name == "qrf_tail_concentration"
    assert not gate.passed
    assert "self_employment_income" in gate.failures[0]
    assert gate.details["surface"]["declared_qrf_outputs"] >= 47


def test_thin_qrf_column_is_reported_not_checked() -> None:
    n = 4
    dataset = _dataset(n=n, person_columns={"self_employment_income": np.ones(n)})
    values, weights, _surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=10,
            max_top_share=0.5,
            min_nonzero_records=100,
        ),
    )

    assert not gate.passed
    assert gate.details["thin_columns"]["self_employment_income"] == 4
    assert "No declared QRF output" in gate.failures[0]


def test_thin_qrf_exclusion_is_classified_as_dormant() -> None:
    values = {"person.x": np.ones(4)}
    weights = {"person.x": np.ones(4)}

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=10,
            max_top_share=0.5,
            min_nonzero_records=100,
            reviewed_exclusions={"person.x": "Seeded thin entry."},
        ),
    )

    assert not gate.passed
    assert gate.details["thin_columns"] == {"person.x": 4}
    assert gate.details["reviewed_exclusions"] == {}
    assert gate.details["stale_exclusions"] == []
    assert gate.details["dormant_exclusions"] == ["person.x"]


def test_qrf_stale_exclusion_fails_and_dormant_is_reported() -> None:
    n = 10
    dataset = _dataset(n=n, person_columns={"self_employment_income": np.ones(n)})
    values, weights, _surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions={
                "self_employment_income": "Seeded stale entry.",
                "dividend_income": "Seeded dormant entry.",
            },
        ),
    )

    assert not gate.passed
    assert "Stale reviewed exclusions" in gate.failures[0]
    assert gate.details["stale_exclusions"] == ["self_employment_income"]
    assert gate.details["dormant_exclusions"] == ["dividend_income"]


def test_qrf_policy_validation() -> None:
    with pytest.raises(ValueError, match="strict subset"):
        UKQRFTailConcentrationPolicy(
            top_k=10, max_top_share=0.5, min_nonzero_records=10
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        UKQRFTailConcentrationPolicy(top_k=1, max_top_share=1.0, min_nonzero_records=2)
    with pytest.raises(ValueError, match="need reasons"):
        UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions={"self_employment_income": ""},
        )


def test_committed_exclusion_registers_load_and_are_empty() -> None:
    input_mass = load_uk_reviewed_exclusion_register(
        None,
        resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    )
    qrf_tail = load_uk_reviewed_exclusion_register(
        None,
        resource=UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    )

    assert input_mass == {}
    assert qrf_tail == {}


def test_register_loader_rejects_missing_reasons_and_bad_schema(tmp_path) -> None:
    bad_reason = tmp_path / "register.json"
    bad_reason.write_text(
        json.dumps({"schema_version": 1, "exclusions": {"person.x": ""}})
    )
    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(json.dumps({"schema_version": 2, "exclusions": {}}))

    with pytest.raises(ValueError, match="need reasons"):
        load_uk_reviewed_exclusion_register(
            bad_reason,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )
    with pytest.raises(ValueError, match="schema_version"):
        load_uk_reviewed_exclusion_register(
            bad_schema,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )


def test_uk_totals_match_the_shared_frame_helper_on_equivalent_data() -> None:
    """The UK table-layout helper must not reinvent the US numeric semantics.

    Same records, same weights, one expressed as a populace Frame (the US
    path) and one as UK national tables: per-column weighted totals must be
    identical, differing only in the ``entity.`` namespace the UK layout
    needs because its tables do not enforce globally unique column names.
    """

    from populace.build.input_mass import input_mass_totals
    from populace.frame import EntitySchema, Frame, WeightKind, Weights

    person = pd.DataFrame(
        {
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "employment_income": [30_000.0, np.nan, 12_000.0],
            "is_disabled": pd.array([True, False, pd.NA], dtype="boolean"),
            "occupation": ["a", "b", "c"],  # strings are skipped on both paths
        }
    )
    household = pd.DataFrame({"household_id": [1, 2], "council_tax": [900.0, 1_500.0]})
    weights = np.array([2.0, 5.0])

    frame_totals = input_mass_totals(
        Frame(
            {"person": person, "household": household},
            EntitySchema(group_entities=("household",)),
            {"household": Weights(weights, WeightKind.DESIGN)},
        )
    )
    uk_totals = uk_dataset_input_mass_totals(
        SimpleNamespace(
            person=person.assign(person_benunit_id=[201, 201, 202]),
            benunit=pd.DataFrame({"benunit_id": [201, 202]}),
            household=household.assign(household_weight=weights),
        )
    )

    assert frame_totals == {
        "employment_income": uk_totals["person.employment_income"],
        "is_disabled": uk_totals["person.is_disabled"],
        "council_tax": uk_totals["household.council_tax"],
    }
    # NaN fills to 0, booleans total weighted True mass, weights broadcast
    # through membership — asserted against hand computation once, so both
    # paths are pinned to the same semantics rather than merely to each other.
    assert frame_totals["employment_income"] == 30_000.0 * 2.0 + 12_000.0 * 5.0
    assert frame_totals["is_disabled"] == 2.0
    assert "occupation" not in frame_totals
    assert "person.occupation" not in uk_totals


def test_uk_input_mass_gate_is_the_shared_gate_plus_recorded_identity() -> None:
    """Without exclusions the UK wrapper must reproduce the US gate verbatim.

    Same failure lines, same verdict, same details — the UK result may only
    add the reference identity and the (empty) stale/dormant register fields
    the #609 discipline requires on top of the shared gate.
    """

    from populace.build.gates import input_mass_parity_gate

    candidate = {
        "person.employment_income": 0.0,  # the #278 signature
        "person.pension_income": 4.0,  # -60% drift
        "person.new_layer": 7.0,  # candidate-only
    }
    reference = _reference(
        {
            "person.employment_income": 10.0,
            "person.pension_income": 10.0,
            "person.tiny": 0.5,  # below the floor
        }
    )
    policy = _policy(relative_tolerance=0.5, minimum_reference_total=1.0)

    shared = input_mass_parity_gate(
        candidate,
        reference.totals,
        candidate_name="uk_release_candidate",
        reference_name=reference.filename,
        relative_tolerance=policy.relative_tolerance,
        minimum_reference_total=policy.minimum_reference_total,
    )
    ported = uk_input_mass_parity_gate(candidate, reference, policy=policy)

    assert ported.name == shared.name == "input_mass_parity"
    assert ported.passed == shared.passed
    assert ported.failures == shared.failures
    shared_details = dict(shared.details)
    assert {
        key: value for key, value in ported.details.items() if key in shared_details
    } == shared_details
    assert set(ported.details) - set(shared_details) == {
        "stale_exclusions",
        "dormant_exclusions",
        "reference_identity",
    }


def test_uk_tail_gate_is_the_shared_gate_under_the_uk_name() -> None:
    """Exclusion discipline included: the shared US gate already carries it."""

    from populace.build.gates import tail_concentration_gate

    n = 12
    concentrated = np.ones(n)
    concentrated[0] = 500.0
    values = {"self_employment_income": concentrated, "dividend_income": np.ones(n)}
    weights = {name: np.ones(n) for name in values}
    exclusions = {"dividend_income": "Seeded stale entry."}

    shared = tail_concentration_gate(
        values,
        weights,
        top_k=1,
        max_top_share=0.5,
        min_nonzero_records=2,
        reviewed_exclusions=exclusions,
    )
    ported = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions=exclusions,
        ),
    )

    assert shared.name == "tail_concentration"
    assert ported.name == "qrf_tail_concentration"
    assert ported.passed == shared.passed
    assert ported.failures == shared.failures
    assert dict(ported.details) == dict(shared.details)


def test_input_mass_reference_round_trips_the_measurement_schema(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "filename": "enhanced_frs_2023_24.h5",
                    "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
                    "sha256": "a" * 64,
                    "vintage": "2023_24",
                },
                "totals": {"person.employment_income": 10.5},
            }
        )
    )

    reference = load_uk_input_mass_reference(path)

    assert reference.filename == "enhanced_frs_2023_24.h5"
    assert dict(reference.totals) == {"person.employment_income": 10.5}
    with pytest.raises(ValueError, match="schema_version"):
        path.write_text(json.dumps({"schema_version": 9}))
        load_uk_input_mass_reference(path)
