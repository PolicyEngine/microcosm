"""Actual bounded source owners on invented inputs; tests authored, not native."""

import json
import shutil
import sys
from fractions import Fraction
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import current_child_property_income_source as child
from microcosm.fit.joint_empirical import (
    SupportRequirements,
    SupportThreshold,
    fit_joint_empirical,
)


def source_arguments(tmp_path, monkeypatch):
    """Place the sole eligible teenager in the unselected household before issuance."""
    import test_us_current_asec_income_routing as routing_fixture
    import test_us_current_property_income_sources as property_fixture
    import test_us_survey_population_preparation as preparation_fixture

    fraction, seed = Fraction(2, 3), 41
    domain = child.domains.Domain.SHARED_HOUSING.value
    chosen = child.source.survey_domain_sample.select_domain_households(
        row_ids=("00007", "00008"),
        source_channels=("asec", "asec"),
        domain_keys=(domain, domain),
        cells=((domain, "asec"),),
        fraction=fraction,
        seed=seed,
    )[0]
    assert len(chosen.positions) == 1
    selected_adult = (105, 107)[chosen.positions[0]]
    donor = 107 if selected_adult == 105 else 105
    ages = {105: 55, 106: 14, 107: 55, 108: 14}
    ages[donor] = 15
    monkeypatch.setattr(routing_fixture, "AGES", ages)
    original_fixture = preparation_fixture.fixture

    def positive_households(path, patch):
        return original_fixture(path, patch, zero=False)

    monkeypatch.setattr(routing_fixture, "fixture", positive_households)
    arguments = property_fixture.source_arguments(tmp_path, monkeypatch)
    # Both preparation requests are created before either owner is issued.
    source_dir = arguments["source_dir"]
    partial = tmp_path / "partial-source"
    shutil.copytree(source_dir, partial)
    request = json.loads((partial / "selection-request.json").read_bytes())
    request["fraction"] = [fraction.numerator, fraction.denominator]
    request["seed"] = seed
    (partial / "selection-request.json").write_bytes(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    )
    captures = tmp_path / "partial-captures"
    captures.mkdir()
    full = {
        name: arguments[name]
        for name in ("source_dir", "snapshot_root", "fraction", "seed")
    }
    smaller = dict(
        source_dir=partial, snapshot_root=captures, fraction=fraction, seed=seed
    )
    return full, smaller, donor


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    root = tmp_path_factory.mktemp("child-property-source-owner")
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, donor = source_arguments(root, patch)
        full = child.source.prepare_authenticated_survey_population(**full_args)
        partial = child.source.prepare_authenticated_survey_population(**partial_args)
        full_values = child.qualify_child_property_sources(full)
        partial_values = child.qualify_child_property_sources(partial)
        yield SimpleNamespace(
            full=full,
            partial=partial,
            values=full_values,
            partial_values=partial_values,
            donor=donor,
        )
        full._checked()
        partial._checked()


def test_full_source_donor_is_available_when_target_sample_has_no_teenagers(actual):
    selected = actual.partial._checked()[2].frame.person
    asec = selected.loc[
        selected[child.routing.support_channel_column("person")].eq("asec")
    ]
    assert not asec.A_AGE.between(15, 17).any()
    left, right = actual.values.donor_projection, actual.partial_values.donor_projection
    assert left.donors.index.tolist() == right.donors.index.tolist() == [actual.donor]
    pd.testing.assert_frame_equal(left.donors, right.donors)
    pd.testing.assert_frame_equal(left.diagnostics, right.diagnostics)
    one = SupportThreshold(1, 1, 1.0, 1.0)
    support = SupportRequirements(
        "invented-one-source-donor", "test", one, one, one, (3,)
    )

    def fit(projection):
        frame = projection.donors
        return fit_joint_empirical(
            frame,
            targets=child.TARGETS,
            donor_keys=frame.donor_key.tolist(),
            household_keys=frame.household_key.tolist(),
            support=support,
            weights="original_household_design_weight",
        )

    assert fit(left).to_bytes() == fit(right).to_bytes()
    assert (
        actual.values.evidence["preparation_sha256"]
        != actual.partial_values.evidence["preparation_sha256"]
    )


def test_actual_source_knownness_and_complete_preparation_are_preserved(actual):
    before = child.source._frame_identity(actual.full._checked()[2].frame)
    value = child.qualify_child_property_sources(actual.full)
    assert before == child.source._frame_identity(actual.full._checked()[2].frame)
    eligible = value.recipients.loc[value.recipients.eligible]
    assert len(eligible) >= 2
    assert (
        not eligible.ordinary_source_known.any()
        and not eligible.dividend_source_known.any()
    )
    assert value.evidence["full_source_rows"] == 4
    assert value.evidence["donor_sampling_fraction_applied"] is False
    assert value.evidence["source_admission_issued"] is False


def test_copied_preparation_cannot_supply_full_donor_authority(actual):
    forged = object.__new__(type(actual.full))
    object.__setattr__(forged, "payload", actual.full.payload)
    with pytest.raises(ValueError):
        child.qualify_child_property_sources(forged)
    actual.full._checked()


@pytest.mark.parametrize("when", ["first", "final"])
@pytest.mark.parametrize(
    "mutation", ["constant", "function", "defaults", "closure", "source_value"]
)
def test_initial_and_final_owner_callback_mutations_refuse_current_borrow(
    actual, when, mutation
):
    target = 1 if when == "first" else 2
    count = 0
    mutated = False
    original_profile = sys.getprofile()
    original_ages = child.DONOR_AGES
    original_function = child._bad_status
    original_defaults = child._age.__defaults__
    closure_cell = child.QualifiedChildPropertySources.__repr__.__closure__[0]
    original_closure = closure_cell.cell_contents
    source_people = actual.full._checked()[2].source_frames[1].person
    source_values = source_people.A_AGE.copy(deep=True)

    def profile(frame, event, argument):
        nonlocal count, mutated
        if (
            event == "return"
            and frame.f_code
            is child.source.AuthenticatedSurveyPopulationPreparation._checked.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is child.qualify_child_property_sources.__code__
        ):
            count += 1
            if count == target:
                mutated = True
                if mutation == "constant":
                    child.DONOR_AGES = (16, 17)
                elif mutation == "function":
                    child._bad_status = lambda value: False
                elif mutation == "defaults":
                    child._age.__defaults__ = ("15",)
                elif mutation == "closure":
                    closure_cell.cell_contents = object()
                else:
                    source_people.loc[:, "A_AGE"] = source_people.A_AGE + 1

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError):
            child.qualify_child_property_sources(actual.full)
    finally:
        sys.setprofile(original_profile)
        child.DONOR_AGES = original_ages
        child._bad_status = original_function
        child._age.__defaults__ = original_defaults
        closure_cell.cell_contents = original_closure
        source_people.loc[:, "A_AGE"] = source_values
    assert mutated
    actual.full._checked()


def test_detached_donor_mutation_does_not_change_retained_source(actual):
    values = child.qualify_child_property_sources(actual.full)
    old = child.child_property_sources_seal(values)
    values.donor_projection.donors.loc[:, child.TARGETS[0]] += 11
    assert child.child_property_sources_seal(values) != old
    new = child.qualify_child_property_sources(actual.full)
    assert child.child_property_sources_seal(new) == old
    actual.full._checked()
