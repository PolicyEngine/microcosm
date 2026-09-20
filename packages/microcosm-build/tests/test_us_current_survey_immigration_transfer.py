"""Original allocation authority and paired transfer on invented sources."""

import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_immigration_transfer as owner


def _phase(name):
    if os.environ.get("MICROCOSM_INVENTED_PHASES") == "1":
        print(
            "IMMIGRATION_PHASE "
            + json.dumps(
                {
                    "phase": name,
                    "cpu_s": time.process_time(),
                    "wall_s": time.monotonic(),
                }
            ),
            flush=True,
        )


def test_fit_implementation_uses_canonical_class_and_refuses_rebound_cache(monkeypatch):
    owner._require_fit_implementation()
    with monkeypatch.context() as patch:
        patch.setattr(owner.transfer, "QRF", object())
        with pytest.raises(ValueError, match="FIT_IMPLEMENTATION_CHANGED"):
            owner._require_fit_implementation()


def test_temporary_lineage_text_preserves_exact_draw_keys_and_serialization():
    from microcosm.build.serialization_dtypes import canonicalize_table_string_dtypes

    original = pd.DataFrame(
        {
            "source_year": [2024, 2024],
            "source_household_id": pd.Series(
                [2**100 + 1, "2024HU0000001"], dtype=object
            ),
            "source_person_id": pd.Series(
                ["0001234567890123456789", "001"], dtype=object
            ),
        }
    )
    projected = original.copy(deep=True)
    for name in ("source_household_id", "source_person_id"):
        projected[name] = owner._lineage_text(original[name])
    assert projected.source_household_id.tolist() == [str(2**100 + 1), "2024HU0000001"]
    assert projected.source_person_id.tolist() == original.source_person_id.tolist()
    np.testing.assert_array_equal(
        owner.rules._stable_person_draws(original, seed=42, salt="lineage-contract"),
        owner.rules._stable_person_draws(projected, seed=42, salt="lineage-contract"),
    )
    canonical = canonicalize_table_string_dtypes(
        projected, boundary="invented lineage projection", table_name="person"
    )
    # The serializer may normalize the missing-value string dtype convention;
    # it must accept the declared string columns and preserve every key value.
    pd.testing.assert_frame_equal(canonical, projected, check_dtype=False)
    assert original.source_household_id.iloc[0] == 2**100 + 1
    assert type(original.source_household_id.iloc[0]) is int
    raw = {"PH_SEQ": "01234", "PERIDNUM": "0" * 19 + "123", "A_LINENO": "01"}
    retained = dict(raw)
    native_key = owner.assignment_owner.donor_owner.literals.original._key(raw, "asec")
    assert native_key == (1234, retained["PERIDNUM"], 1)
    assert owner._lineage_text([native_key[0]])[0] == "1234"
    assert owner._lineage_text([native_key[1]])[0] == retained["PERIDNUM"]
    assert raw == retained
    assert (
        owner.assignment_owner.donor_owner.literals.original._key(raw, "asec")
        == native_key
    )


@pytest.mark.parametrize("value", [1.0, True, pd.NA, ""])
def test_temporary_lineage_text_refuses_ambiguous_scalars(value):
    with pytest.raises(ValueError, match="LINEAGE_SCALAR"):
        owner._lineage_text([value])


def test_unissued_parent_refuses_before_sources_or_fit():
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        owner.transfer_current_survey_immigration(
            object(), object(), object(), seed=42, n_estimators=2
        )


@pytest.mark.parametrize("seed", [-1, True, 2**64, 1.0])
def test_invalid_seed_refuses_before_sources_or_fit(seed):
    with pytest.raises(ValueError, match="SEED"):
        owner.transfer_current_survey_immigration(
            object(), object(), object(), seed=seed, n_estimators=2
        )


@pytest.mark.parametrize("trees", [0, True, -1, 1.5])
def test_invalid_forest_refuses_before_sources_or_fit(trees):
    with pytest.raises(ValueError, match="TREES"):
        owner.transfer_current_survey_immigration(
            object(), object(), object(), seed=42, n_estimators=trees
        )


def test_copied_owner_is_not_issued():
    value = owner.CurrentSurveyImmigrationTransfer(None, None, b"{}")
    with pytest.raises(ValueError, match="ISSUED_OWNER_REQUIRED"):
        replace(value).validate()


def source_arguments(root, patch):
    """Compose literal sources before all real graph/source issuers run."""
    import test_us_graph_atomic_completion_host as completion
    import test_us_survey_population_preparation as preparation

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographics
    from microcosm.build.us_runtime import current_survey_person_status_source as status

    original_person = preparation._person

    def person(*args, **kwargs):
        row = original_person(*args, **kwargs)
        row.update(
            CIT="1", POBP="001", YOEP="", SEX="1" if int(row["SPORDER"]) == 1 else "2"
        )
        if row["SERIALNO"] == "2024HU0000001" and int(row["SPORDER"]) == 1:
            row.update(CIT="5", POBP="303", YOEP="1980")
        return row

    patch.setattr(preparation, "_person", person)
    call, _ = completion._source_arguments(root, patch)
    call["fraction"] = Fraction(1)
    request = call["source_dir"] / "selection-request.json"
    document = json.loads(request.read_bytes())
    document["fraction"] = [1, 1]
    request.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    )
    folder = call["source_dir"] / "asec"
    paths, pins = {}, []
    for (
        year,
        member,
        archive,
        *_,
    ) in owner.assignment_owner.donor_owner.literals.original.asec._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name in owner.assignment_owner.donor_owner.literals.ASEC_VALUE_COLUMNS:
            if name not in raw:
                raw[name] = "0"
        raw["PRCITSHP"] = "1"
        raw["PENATVTY"] = "057"
        raw["A_MARITL"] = "7"
        raw["A_LFSR"] = raw.A_AGE.map(lambda age: "7" if int(age) >= 16 else "0")
        raw["PEAFEVER"] = raw.A_AGE.map(lambda age: "2" if int(age) >= 17 else "-1")
        raw["SPM_CAPHOUSESUB"] = "0"
        for name in ("MCARE", "CAID", "IHSFLG", "CHAMPVA", "MIL"):
            raw[name] = "2"
        if year == 2024:
            adult = raw.A_AGE.eq("55")
            assert adult.sum() == 1
            raw.loc[adult, ["PRCITSHP", "PENATVTY", "PEINUSYR"]] = ["5", "303", "1"]
        raw.iloc[::-1].to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(raw),
                len(data),
            )
        )
        paths[year] = path
    for module in (
        owner.assignment_owner.donor_owner.literals.original.asec,
        restoration,
        demographics.demographic,
        status.student,
    ):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = root / "immigration-restored"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=restored,
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    return call


def zero_fixture_controls(patch):
    """Fixture-only stage bytes; original parser/issuers/guards stay active.

    Five invented recipients cannot represent national humanitarian stocks. Set
    only those fixture targets to zero, retain citations and undocumented
    controls, and intercept every resource read consistently through validation.
    These bytes never change the maintained resource or production defaults.
    """
    read = Path.read_text

    def replace_targets(value):
        if isinstance(value, dict):
            return {
                k: 0 if k == "target" else replace_targets(v) for k, v in value.items()
            }
        return value

    def fixture_text(path, *args, **kwargs):
        text = read(path, *args, **kwargs)
        if path.name == "source_stages.json":
            doc = json.loads(text)

            def visit(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "humanitarian_status_stocks":
                            value[key] = replace_targets(child)
                        else:
                            visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(doc)
            return json.dumps(doc)
        return text

    patch.setattr(Path, "read_text", fixture_text)


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        root = tmp_path_factory.mktemp("original-immigration")
        _phase("sources.before")
        call = source_arguments(root, patch)
        _phase("sources.after")
        _phase("full51.before")
        run = owner.host.run_atomic_survey_financial(**call)
        _phase("full51.after")
        original_stamp = owner.allocation._population_stamp(
            run.prefix.allocated_population
        )
        zero_fixture_controls(patch)
        _phase("asec_donor.before")
        donor = owner.assignment_owner.donor_owner.borrow_full_asec_immigration_donor(
            run.prefix.preparation
        )
        _phase("asec_donor.after")
        _phase("asec_assignment.before")
        assigned = owner.assignment_owner.assign_full_asec_immigration(donor, seed=42)
        _phase("asec_assignment.after")
        _phase("acs_projection.before")
        acs = owner.acs_owner.borrow_current_acs_immigration_projection(
            run.prefix.preparation
        )
        _phase("acs_projection.after")
        bank_root = root / "paired-draw-bank"
        _phase("original_transfer.before")
        result = owner.transfer_current_survey_immigration(
            run, assigned, acs, seed=42, n_estimators=2, bank_root=bank_root
        )
        _phase("original_transfer.after")
        yield SimpleNamespace(
            root=root,
            run=run,
            donor=donor,
            assigned=assigned,
            acs=acs,
            result=result,
            call=call,
            original_stamp=original_stamp,
            bank_root=bank_root,
        )
        _phase("postchecks.before")
        assert (
            owner.allocation._population_stamp(run.prefix.allocated_population)
            == original_stamp
        )
        _phase("postchecks.after")


def test_genuine_original_transfer_preserves_weights_sources_and_asec(actual):
    from microcosm.frame import WeightKind

    result, original = actual.result, actual.run.prefix.allocated_population.frame
    _phase("positive_validate.before")
    result.validate()
    _phase("positive_validate.after")
    receipt = json.loads(result.receipt)
    assert actual.donor.frame.weights_for("household").kind is WeightKind.DESIGN
    assert result.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    assert result.frame.weights_for("household") is original.weights_for("household")
    assert result.frame.metadata == original.metadata
    assert result.frame.mass_log == original.mass_log
    for entity in original.entities:
        table = result.frame.table(entity)
        if entity == "person":
            table = table.drop(columns=list(owner.OUTPUTS))
        pd.testing.assert_frame_equal(table, original.table(entity))
    assert set(result.pairs.columns) == set(owner.OUTPUTS)
    assert result.pairs.notna().all().all()
    origins = owner.acs_owner._RETAINED[id(actual.acs)][2].qualified.origins
    assert receipt["asec_immutable_rows"] == int(origins.source.eq("asec").sum()) == 2
    assert receipt["acs_imputed_rows"] == int(origins.source.eq("acs").sum()) == 3
    # The full donor includes rows outside this receiving native spine.
    assert len(actual.donor.frame.person) == 4
    assert receipt["national_stock_alignment_qualified"] is False
    assert receipt["release_eligible"] is False
    assert all(r["reconciliation"] is not None for r in receipt["imputed_inputs"])
    for pid, origin in origins.loc[origins.source.eq("asec")].iterrows():
        np.testing.assert_array_equal(
            result.pairs.loc[pid].to_numpy(),
            actual.assigned.pairs.loc[
                origin.native_person_id, list(owner.OUTPUTS)
            ].to_numpy(),
        )


def test_exact_row_years_masks_predictors(actual):
    origins = owner.acs_owner._RETAINED[id(actual.acs)][2].qualified.origins
    donor, recipient, mutable = owner._project_inputs(
        actual.run.prefix.allocated_population.frame,
        actual.assigned,
        actual.acs,
        origins,
    )
    assert donor.person[owner.YEAR].eq(2025).all()
    np.testing.assert_array_equal(
        recipient.person[owner.YEAR], np.where(mutable, 2024, 2025)
    )
    assert recipient.person.loc[mutable, "A_LINENO"].isna().all()
    assert recipient.person.loc[mutable, "PRCITSHP"].isna().all()
    assert recipient.person.loc[~mutable, "CIT"].isna().all()
    for name in owner.OUTPUTS:
        np.testing.assert_array_equal(recipient.person[name].isna(), mutable)
    surface = owner.transfer._transfer_feature_surface(
        donor,
        recipient,
        entity="person",
        targets=owner.OUTPUTS,
        observation_year_column=owner.YEAR,
    )
    assert surface.required == (
        *owner.transfer.ACS_PERSON_TRANSFER_PREDICTORS,
        *owner.transfer._IMMIGRATION_REQUIRED_FEATURES,
    )
    assert set(surface.optional) <= {owner.transfer._HEAD_FEATURE}


def test_bank_replay(actual):
    _phase("warm_transfer.before")
    repeated = owner.transfer_current_survey_immigration(
        actual.run,
        actual.assigned,
        actual.acs,
        seed=42,
        n_estimators=2,
        bank_root=actual.bank_root,
    )
    _phase("warm_transfer.after")
    pd.testing.assert_frame_equal(repeated.pairs, actual.result.pairs)
    assert (
        json.loads(repeated.receipt)["bank_identity_sha256"]
        == json.loads(actual.result.receipt)["bank_identity_sha256"]
    )


@pytest.mark.parametrize("which", ["run", "assigned", "acs", "result"])
def test_copied_authority_refuses(actual, which):
    if which == "result":
        with pytest.raises(ValueError, match="ISSUED_OWNER_REQUIRED"):
            replace(actual.result).validate()
    else:
        inputs = [actual.run, actual.assigned, actual.acs]
        inputs[("run", "assigned", "acs").index(which)] = replace(
            getattr(actual, which)
        )
        with pytest.raises(
            ValueError,
            match="UNISSUED_FINANCIAL_RUN|ISSUED_OWNER_REQUIRED|RETAINED_VIEW",
        ):
            owner.transfer_current_survey_immigration(*inputs, seed=42, n_estimators=2)


@pytest.mark.parametrize("field", ["frame", "pairs", "receipt"])
def test_equal_detached_result_values_refuse(actual, field):
    value = actual.result
    old = getattr(value, field)
    replacement = (
        owner._frame_with_person(old, old.person.copy(deep=True))
        if field == "frame"
        else old.copy(deep=True)
        if field == "pairs"
        else b"{}"
    )
    try:
        object.__setattr__(value, field, replacement)
        with pytest.raises(ValueError, match="RETAINED_CHANGED"):
            value.validate()
    finally:
        object.__setattr__(value, field, old)


@pytest.mark.parametrize(
    "surface",
    [
        "cell",
        "metadata",
        "weight",
        "kind",
        "anchor",
        "owners",
        "version",
        "ledger",
    ],
)
def test_original_population_authority_is_not_replaceable(actual, surface):
    # Detached pure reconstruction controls never alter the issued parent.
    # They prove allocation equality checks, not issuance of these copies.
    population = owner.allocation._copy_population(
        actual.run.prefix.allocated_population
    )
    view = actual.run.prefix.preparation.checked_view()
    restore = []

    def set_attr(obj, name, value):
        previous = getattr(obj, name)
        restore.append(lambda: object.__setattr__(obj, name, previous))
        object.__setattr__(obj, name, value)

    if surface == "cell":
        table = population.frame.person
        old = table.at[table.index[0], "person_household_id"]
        restore.append(
            lambda: table.__setitem__(
                "person_household_id",
                table.person_household_id.where(table.index != table.index[0], old),
            )
        )
        table.at[table.index[0], "person_household_id"] += 1
    elif surface == "metadata":
        set_attr(population.frame, "_metadata", {"foreign": True})
    elif surface == "weight":
        weights = population.frame.weights_for("household")
        changed = weights.values.copy()
        changed[0] += 1
        changed[1] -= 1
        set_attr(weights, "values", changed)
    elif surface == "kind":
        set_attr(
            population.frame.weights_for("household"), "kind", owner.WeightKind.DESIGN
        )
    elif surface == "anchor":
        changed = dict(population.design_weights)
        changed["household"] = changed["household"].copy() + 1
        set_attr(population, "design_weights", changed)
    elif surface == "owners":
        set_attr(population, "owners", {})
    elif surface == "version":
        set_attr(population, "version", "foreign")
    elif surface == "ledger":
        set_attr(population, "mass_ledger", ())
    try:
        with pytest.raises(ValueError):
            owner.allocation._raw_allocation(view, population)
    finally:
        for operation in reversed(restore):
            operation()


def test_real_nonzero_controls_refuse_invented_stock_shortfall(actual):
    origins = owner.acs_owner._RETAINED[id(actual.acs)][2].qualified.origins
    donor, recipient, mutable = owner._project_inputs(
        actual.run.prefix.allocated_population.frame,
        actual.assigned,
        actual.acs,
        origins,
    )
    controls = owner._ISSUED[id(actual.result)][2].controls
    draw = next(d for d in controls.humanitarian if d.origin == "afghanistan")
    nonzero = replace(controls, humanitarian=(replace(draw, target=73566),))
    person = recipient.person.copy(deep=True)
    for name in owner.OUTPUTS:
        person[name] = actual.result.pairs[name].to_numpy()
    with pytest.raises(ValueError, match="shortfall|candidate|available"):
        owner.rules.reconcile_us_immigration_humanitarian_transfer(
            person,
            weights=recipient.resolve_weights("person").values,
            mutable_rows=mutable,
            seed=42,
            controls=nonzero,
            observation_year_column=owner.YEAR,
        )


@pytest.mark.parametrize(
    "module,name",
    [
        (owner.allocation, "_raw_allocation"),
        (owner.allocation, "_population_stamp"),
        (owner.host, "_pure_run"),
        (owner.source, "_pure_final"),
        (owner.allocation.graph, "_check_population_state"),
        (owner.allocation.replay, "same_replayed_frame"),
    ],
)
def test_rebound_authority_helpers_refuse_before_invocation(
    actual, monkeypatch, module, name
):
    def changed(*args, **kwargs):
        raise AssertionError("Rebound authority must never be called")

    with monkeypatch.context() as patch:
        patch.setattr(module, name, changed)
        with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
            actual.result.validate()


def test_final_control_io_mutation_permanently_revokes_parent_last(actual, monkeypatch):
    # Last test: this poisons the one genuine host. Restore only borrowed bytes,
    # never reset the permanent revocation flag or claim the owner is usable.
    population = actual.run.prefix.allocated_population
    original = population.owners
    reader = Path.read_text
    changed = []

    def late(path, *args, **kwargs):
        text = reader(path, *args, **kwargs)
        if path.name == "source_stages.json" and not changed:
            changed.append(True)
            object.__setattr__(population, "owners", {})
        return text

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", late)
            with pytest.raises(ValueError):
                actual.result.validate()
    finally:
        object.__setattr__(population, "owners", original)
    assert changed
    assert owner.host._run_entry(actual.run)[2].completion_boundary.revoked
    with pytest.raises(ValueError, match="REVOKED"):
        actual.run.checked_view()
