"""Qualified transport for one development-only PUF conditional diagnostic.

Callers supply independently reviewed content pins. Receipt labels or fixture
definitions cannot replace those pins. No function opens a file or fits a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input

from . import asec_current_money as current_money
from . import graph_asec_income as asec_income
from . import native_household_origin as origin
from . import puf_detail_transfer as detail
from . import puf_monetary_agi_projection as agi
from . import puf_raw_source as raw
from . import support_provenance as provenance
from .graph_context import _row_identity

require = detail.require
RECIPE_ID = "us.puf.schedule_c_wage_mars_diagnostic.v1"
SCOPE = "development_conditional_support_not_tax_inputs"
SOURCE_HEAD = "c5c34a048fc38c01840b2da5454385e0a51d1c49"
HELPER_HEAD = "dea036278cc3f7c29c99da67bcadd968822c1d7a"
MAX_DONORS = 250_000
ROLES = (
    "acceptance",
    "proposal",
    "config",
    "completion",
    "cold",
    "required",
    "definition",
    "projection_definition",
    "status",
    "projection",
    "arrays",
)


def recipe_document():
    """A diagnostic modeling choice, never a modification of source admission."""
    return {
        "id": RECIPE_ID,
        "scope": SCOPE,
        "target": detail.TARGET,
        "features": list(detail.FEATURES),
        "output": detail.OUTPUT,
        "trees": 2,
        "seed": 578,
        "zero_atol": 0.0,
        "donor_weight": "delivered_S006_integer_hundredths_DESIGN",
        "wage_proxy": "head_and_actual_JOINT_spouse_only_dependents_excluded",
        "host_role_source": "microunit_tax_unit_role_input",
        "filing_mapping": detail.MARS,
        "missing_wage": "refuse_selected_earner_no_age_only_zero",
        "prices": "2015_file_CPI_U_to_2024_development_prices",
        "source_period": "unresolved_raw_FLPDYR",
        "source_fit_admission": detail.SOURCE_ADMISSION,
        "population_meaning": "potential_filing_unit_support_not_observed_filers",
        "tax_input_admission": False,
        "scientific_launch_admission": False,
        "release_eligible": False,
    }


def parse_document(payload):
    """Strict JSON, allowing the reviewed helper's formatted receipt encoding."""
    require(
        type(payload) is bytes and 0 < len(payload) <= 16_000_000,
        "DIAGNOSTIC_DOCUMENT_BOUND",
    )

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DIAGNOSTIC_DUPLICATE_KEY")
            result[key] = value
        return result

    def bad(_):
        raise ValueError("DIAGNOSTIC_NONFINITE_JSON")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=bad)
    require(isinstance(value, dict), "DIAGNOSTIC_DOCUMENT_TYPE")
    return value


def content_pin(payload):
    return {"bytes": len(payload), "sha256": codec.sha(payload)}


def check_pin(payload, pin, reason):
    require(
        isinstance(pin, Mapping)
        and type(pin.get("bytes")) is int
        and codec._hash(pin.get("sha256"))
        and content_pin(payload) == {k: pin[k] for k in ("bytes", "sha256")},
        reason,
    )


def qualify_price_transport(payloads, expected_pins, *, route, recipe):
    """Validate DEA completion → config/source → grown-array transport.

    Expected pins are execution inputs established outside this parser. The
    production route accepts only the packaged original source definitions;
    tests use a visibly different acceptance/mode with actual fixture codecs.
    """
    require(
        codec.encode_json(recipe) == codec.encode_json(recipe_document()),
        "DIAGNOSTIC_RECIPE",
    )
    require(route in ("packaged", "test_fixture"), "DIAGNOSTIC_SOURCE_ROUTE")
    require(set(payloads) == set(expected_pins) == set(ROLES), "PRICE_TRANSPORT_ROSTER")
    for name in ROLES:
        check_pin(payloads[name], expected_pins[name], "PRICE_TRANSPORT_PIN:" + name)
    documents = {n: parse_document(payloads[n]) for n in ROLES[:6]}
    acceptance, proposal, config, completed, cold, warm = (
        documents[n] for n in ROLES[:6]
    )
    genuine = route == "packaged"
    require(
        acceptance["status"]
        == (
            "ACCEPTED_GENUINE_DEVELOPMENTAL_PRICE_RESTATEMENT"
            if genuine
            else "ACCEPTED_INVENTED_DEVELOPMENTAL_PRICE_RESTATEMENT"
        )
        and acceptance["release_eligible"] is False
        and acceptance["fit_admission"] == detail.SOURCE_ADMISSION
        and config["mode"] == ("genuine" if genuine else "bound_invented")
        and completed["genuine_input_bound"] is genuine
        and completed["status"]
        == (
            "GENUINE_DEVELOPMENTAL_PRICE_RESTATEMENT_COMPLETED"
            if genuine
            else "INVENTED_PRICE_RESTATEMENT_PREPARATION_PASSED"
        ),
        "PRICE_TRANSPORT_AUTHORITY_SCOPE",
    )
    for name in (
        "config",
        "completion",
        "cold",
        "required",
        "arrays",
        "status",
        "projection",
        "definition",
        "projection_definition",
    ):
        require(
            any(
                content_pin(payloads[name]) == {k: p[k] for k in ("bytes", "sha256")}
                for p in acceptance["saved_file_pins_checked"]
            ),
            "PRICE_ACCEPTANCE_PIN:" + name,
        )
    check_pin(
        payloads["proposal"], acceptance["reviewed_proposal"], "PRICE_PROPOSAL_PIN"
    )
    for name, receipt in (("cold", cold), ("required", warm)):
        require(
            completed[name]["sha256"] == codec.sha(payloads[name]), "PRICE_CHILD_PIN"
        )
        require(receipt["phase"] == name, "PRICE_CHILD_PHASE")
    check_pin(payloads["config"], completed["config_pin"], "PRICE_CONFIG_PIN")
    require(
        {k: v for k, v in config.items() if k != "execution"} == proposal["config"]
        and proposal["price_execution_authorized"] is False
        and proposal["release_eligible"] is False,
        "PRICE_REVIEWED_CONFIGURATION",
    )
    require(
        config["inputs"] == cold["input_binding"] == warm["input_binding"],
        "PRICE_SOURCE_INPUTS",
    )
    require(
        set(config["inputs"])
        == {"definition", "projection_definition", "status", "projection"},
        "PRICE_SOURCE_ROSTER",
    )
    for name, pin in config["inputs"].items():
        check_pin(payloads[name], pin, "PRICE_SOURCE_PIN:" + name)
    source = config["bound_source"]
    require(
        source["route"] == route and source["invented"] is (not genuine),
        "PRICE_BOUND_ROUTE",
    )
    source_sha = codec.sha(codec.encode_json(source))
    require(
        source_sha
        == config["binding_sha256"]
        == cold["bound_source_identity"]
        == warm["bound_source_identity"]
        == completed["bound_source_identity"],
        "PRICE_BOUND_SOURCE_SHA",
    )
    runtime_keys = (
        "source_head",
        "helper_head",
        "python",
        "executable",
        "executable_sha256",
        "versions",
    )
    runtime = {k: cold["runtime"][k] for k in runtime_keys}
    require(
        all(
            {k: item["runtime"][k] for k in runtime_keys} == runtime
            for item in (warm, completed)
        ),
        "PRICE_RUNTIME_BINDING",
    )
    require(
        runtime["source_head"] == SOURCE_HEAD
        and runtime["helper_head"] == HELPER_HEAD
        and runtime["executable_sha256"]
        == "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6"
        and runtime["versions"] == {"numpy": "2.4.6", "pandas": "3.0.3"},
        "PRICE_REVIEWED_RUNTIME",
    )
    for item in (cold, warm, completed):
        isolation = item["runtime"]["bytecode_isolation"]
        require(
            isolation["isolated"] is True
            and isolation["dont_write_bytecode"] is True
            and isolation["absent_absolute_prefix"].startswith("/"),
            "PRICE_BYTECODE_ISOLATION",
        )
    require(
        cold["runtime"]["bytecode_isolation"]["absent_absolute_prefix"]
        != warm["runtime"]["bytecode_isolation"]["absent_absolute_prefix"],
        "PRICE_FRESH_PROCESS_PREFIX",
    )
    require(
        cold["pid"] != warm["pid"]
        and warm["all_hits"]
        and warm["kernel_calls"] == []
        and warm["raising_replay_stubs"] is True
        and all(v == 0 for v in warm["transform_calls"].values()),
        "PRICE_REQUIRED_PROOF",
    )
    for name in (
        "contract",
        "array_identity",
        "graph_artifacts",
        "grown_artifact",
        "excluded_lexical_artifact",
    ):
        require(cold[name] == warm[name], "PRICE_COLD_WARM_IDENTITY:" + name)
    contract = cold["contract"]
    require(source["contract"] == contract, "PRICE_SOURCE_CONTRACT")
    require(
        source["runtime"]["executable_sha256"] == runtime["executable_sha256"]
        and source["runtime"]["package_versions"] == runtime["versions"],
        "PRICE_SOURCE_RUNTIME",
    )
    require(
        contract["ratio"] == [313689, 237017]
        and contract["ratio_bits"] == "0aa9e810012df53f"
        and contract["release_eligible"] is False
        and contract["per_record_preliminary_inflation"] is False
        and contract["nominal_income_growth_claim"] is False,
        "PRICE_BASIS",
    )
    require(
        acceptance["source_head"] == SOURCE_HEAD
        and acceptance["helper_head"] == HELPER_HEAD
        and config["helper_head"] == HELPER_HEAD,
        "PRICE_REVIEWED_CODE",
    )
    if genuine:
        definition, projection = (
            raw.packaged_definition(),
            agi.packaged_agi_projection(),
        )
    else:
        definition = raw.fixture_definition(parse_document(payloads["definition"]))
        projection = agi.fixture_agi_projection_document(
            parse_document(payloads["projection_definition"]), definition
        )
    # Source files may use formatted JSON; their exact input pins were checked above.
    require(
        parse_document(payloads["definition"])
        == raw.definition_document_json(definition),
        "PRICE_DEFINITION",
    )
    require(
        parse_document(payloads["projection_definition"])
        == json.loads(projection.canonical),
        "PRICE_PROJECTION_DEFINITION",
    )
    require(
        cold["source_definition_sha256"] == definition.sha256
        and cold["projection_definition_sha256"] == projection.sha256,
        "PRICE_DEFINITION_IDENTITY",
    )
    require(
        source["source"]["head"] == SOURCE_HEAD
        and source["source"]["definition_canonical_sha256"] == definition.sha256
        and source["source"]["projection_canonical_sha256"] == projection.sha256,
        "PRICE_DECLARED_SOURCE_IDENTITY",
    )
    metadata = source["source_column_metadata"]
    require(
        set(metadata) == set(detail.MONEY_FIELDS)
        and codec.sha(codec.encode_json(metadata)) == source["source_basis_sha256"]
        and cold["source_column_metadata_preserved"]
        == warm["source_column_metadata_preserved"]
        == metadata,
        "PRICE_SOURCE_METADATA",
    )
    for value in metadata.values():
        require(
            value["fit_admission"] == detail.SOURCE_ADMISSION
            and value["period_semantics"] == "unresolved_raw_FLPDYR"
            and value["amount_units"] == "whole_usd_source_year"
            and value["grain"] == "source_return"
            and value["projection_sha256"] == projection.sha256,
            "PRICE_SOURCE_MEANING",
        )
    rows = config["ordinary_rows"]
    require(
        type(rows) is int
        and 0 < rows <= MAX_DONORS
        and rows
        == source["ordinary_rows"]
        == acceptance["ordinary_rows"]
        == completed["ordinary_rows"]
        and acceptance["excluded_aggregate_rows"] == completed["aggregate_rows"] == 4,
        "PRICE_ORDINARY_ROSTER",
    )
    check_pin(payloads["arrays"], cold["grown_artifact"], "PRICE_EXPORT_PIN")
    header = {
        "schema": "local.puf.price_restatement.arrays.v1",
        "input_binding_sha256": codec.sha(codec.encode_json(config["inputs"])),
        "contract_sha256": contract["sha256"],
        "source_head": SOURCE_HEAD,
        "helper_head": HELPER_HEAD,
        "release_eligible": False,
    }
    arrays = detail.decode_price_arrays(
        payloads["arrays"],
        expected=header,
        expected_sha256=codec.sha(payloads["arrays"]),
        max_rows=rows,
    )
    require(len(arrays["RECID"]) == rows, "PRICE_EXACT_ROWS")
    for name, values in arrays.items():
        require(
            cold["array_identity"][name]
            == {
                "dtype": values.dtype.str,
                "rows": len(values),
                "sha256": codec.sha(values.tobytes()),
            },
            "PRICE_ARRAY_RECEIPT",
        )
    status = raw.decode_return_status(payloads["status"], definition)
    decoded = agi.decode_agi_projection(
        payloads["projection"],
        projection,
        expected_status_artifact_sha256=codec.sha(payloads["status"]),
    )
    require(
        metadata == agi._thawed(decoded.column_metadata),
        "PRICE_ORIGINAL_METADATA_BINDING",
    )
    frame = detail._donor_frame(arrays, status, decoded, projection, scope=SCOPE)
    binding = {
        "schema": "microcosm.us.puf_diagnostic_price_transport.v1",
        "route": route,
        "recipe": recipe,
        "input_pins": {n: content_pin(payloads[n]) for n in ROLES},
        "source_column_metadata": metadata,
        "source_projection_document": projection.canonical.decode(),
        "original_source_admission": detail.SOURCE_ADMISSION,
        "contract": contract,
        "source_binding_sha256": source_sha,
        "ordinary_rows": rows,
        "donor_model_input_sha256": donor_model_input_sha256(frame),
        "release_eligible": False,
    }
    return frame, codec.encode_json(binding)


def donor_model_input_sha256(frame):
    """Exact donor IDs/features/target and DESIGN resampling weights."""
    table = frame.table("tax_unit").loc[:, [*detail.FEATURES, detail.TARGET]].copy()
    table.index = pd.Index(
        frame.table("tax_unit").tax_unit_id.to_numpy(dtype="int64"), name="tax_unit_id"
    )
    payload = model_input.encode_recipient_matrix(
        table.loc[:, [*detail.FEATURES, detail.TARGET]],
        entity="tax_unit",
        entity_ids=table.index.to_numpy(dtype="int64"),
    )
    weight = frame.weights_for("tax_unit")
    return codec.sha(
        codec.encode_json(
            {
                "matrix_sha256": codec.sha(payload),
                "weight_kind": weight.kind.value,
                "weight_sha256": codec.sha(weight.values.astype("<f8").tobytes()),
            }
        )
    )


def _numeric(series, reason, *, nullable=False):
    require(
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and not pd.api.types.is_complex_dtype(series),
        reason,
    )
    require(nullable or not series.isna().any(), reason)
    return series.to_numpy(dtype="float64", na_value=np.nan)


def acs_wage_evidence(person):
    """Verify carried native ACS arithmetic and classify its earnings universe.

    No missing amount is converted to zero, including under-15 blanks. The
    caller binds these rows to the authenticated full source/clone population.
    """
    required = (
        "person_id",
        "AGEP",
        "WAGP",
        "ADJINC",
        "age",
        "source_year",
        "employment_income_before_lsr",
    )
    require(all(n in person for n in required), "ACS_WAGE_SOURCE_COLUMNS")
    age = _numeric(person.AGEP, "ACS_WAGE_AGE")
    require(
        np.isfinite(age).all()
        and ((age >= 0) & (age <= 99) & (age == np.floor(age))).all()
        and np.array_equal(age, _numeric(person.age, "ACS_WAGE_AGE")),
        "ACS_WAGE_AGE",
    )
    require(person.source_year.astype(str).eq("2024").all(), "ACS_WAGE_COHORT")
    wage = _numeric(person.WAGP, "ACS_WAGE_RAW_TYPE", nullable=True)
    adjustment = _numeric(person.ADJINC, "ACS_WAGE_ADJINC_TYPE", nullable=True)
    carried = _numeric(
        person.employment_income_before_lsr, "ACS_WAGE_CARRIED_TYPE", nullable=True
    )
    observed = ~np.isnan(wage)
    require(
        np.isfinite(wage[observed]).all()
        and (wage[observed] >= 0).all()
        and np.isfinite(adjustment[observed]).all()
        and (adjustment[observed] > 0).all(),
        "ACS_WAGE_DOMAIN",
    )
    require(not (observed & (age < 15)).any(), "ACS_WAGE_OUTSIDE_UNIVERSE_OBSERVED")
    expected = wage * (adjustment / 1_000_000.0)
    require(
        np.array_equal(np.isnan(carried), ~observed)
        and np.array_equal(
            expected[observed].view("uint64"), carried[observed].view("uint64")
        ),
        "ACS_WAGE_ADJUSTED_BINDING",
    )
    return {
        "ids": person.person_id.to_numpy(dtype="int64").tolist(),
        "age_sha256": codec.sha(age.astype("<f8").tobytes()),
        "raw_wage_sha256": codec.sha(wage.astype("<f8").tobytes()),
        "adjustment_sha256": codec.sha(adjustment.astype("<f8").tobytes()),
        "carried_sha256": codec.sha(carried.astype("<f8").tobytes()),
        "universe": "ACS_2024_age_15_plus_rolling_prior_12_month_income",
        "observed": observed.tolist(),
        "in_universe": (age >= 15).tolist(),
        "missing_treatment": "preserved_no_zero_substitution",
        "arithmetic": "WAGP*(ADJINC/1000000)",
        "source_period_equivalence_claim": False,
    }


def asec_wage_evidence(person, values):
    """Reconstruct the accepted income artifact, then join its native person axis.

    ``values`` are executor-checked typed artifacts. H and its T1 descendant
    have legitimate differing source ancestry; the accepted readers bind the
    prepared receipt, current-money buffers, cohort and exact income payload.
    """
    from . import graph_composed_asec_binding as asec_binding
    from . import graph_composed_asec_measures as measures

    binding = measures._binding(values)
    prepared = measures._prepared(values["prepared_receipt"])
    document = measures._document(values["frame_context"])
    measures._bound_population(document, binding)
    selected = measures._selected_money(binding, prepared, values)
    rows = asec_binding.bind_composed_asec_arm_rows(
        values["arm_rows"].payload, binding_document=binding
    )
    original = rows.array("person", "original_ids")

    def integer(name):
        return asec_binding._int64_view(person[name], reason="ASEC_HOST:" + name)

    require(
        np.array_equal(integer(provenance.spine_source_id_column("person")), original),
        "ASEC_HOST_NATIVE_PERSON_ROSTER",
    )
    composed = integer(provenance.support_source_id_column("person"))
    require(
        _row_identity(pd.DataFrame({"person_id": composed}), "person")[
            "ordered_ids_sha256"
        ]
        == binding["arm"]["person"]["composed_ordered_ids_sha256"],
        "ASEC_HOST_COMPOSED_ROSTER",
    )
    text_year = person.source_year.astype(str)
    require(text_year.isin(("2022", "2023", "2024")).all(), "ASEC_HOST_COHORT")
    years = text_year.to_numpy().astype("int64")
    income = asec_income.bind_income_observations(
        values["income_observations"].payload, prepared_receipt=prepared
    )
    result = asec_income.read_reported_income(
        values["reported_income"].payload,
        selected,
        income,
        person_ids=original,
        income_years=years,
        person_positions_sha256=binding["arm"]["person"]["source_positions_sha256"],
        prepared_receipt_sha256=codec.sha(values["prepared_receipt"].payload),
        selection_sha256=codec.sha(values["asec_binding"].payload),
        selected_money_sha256=codec.sha(values["selected_current_money"].payload),
        income_payload_sha256=codec.sha(values["income_observations"].payload),
        source_producer_key=values["prepared_receipt"].producer_key,
        selection_producer_key=values["asec_binding"].producer_key,
        frame_context_sha256=codec.sha(values["frame_context"].payload),
    )
    require(
        np.array_equal(integer("A_AGE"), result.array("A_AGE"))
        and np.array_equal(
            _numeric(person.age, "ASEC_HOST_AGE"), result.array("A_AGE")
        ),
        "ASEC_HOST_NATIVE_AGE",
    )
    name = "asec_reported_wage_income_2024_price"
    actual = _numeric(person[name], "ASEC_HOST_WAGE_TYPE")
    require(
        np.isfinite(actual).all()
        and (actual >= 0).all()
        and np.array_equal(actual.view("uint64"), result.array(name).view("uint64")),
        "ASEC_HOST_WAGE_BINDING",
    )
    require((result.array("WSAL_VAL.validity") == 1).all(), "ASEC_HOST_WAGE_VALIDITY")
    return {
        "ids": person.person_id.to_numpy(dtype="int64").tolist(),
        "source_native_person_sha256": codec.sha(original.tobytes()),
        "income_year_sha256": codec.sha(years.tobytes()),
        "reported_income_sha256": codec.sha(values["reported_income"].payload),
        "wage_sha256": codec.sha(actual.tobytes()),
        "evidence_axes": {
            n: result.array("WSAL_VAL." + n).tolist()
            for n in ("status", "validity", "zero_origin")
        },
        "monetary_basis": json.loads(result.header)["monetary_basis"],
        "missing_treatment": "refuse_no_evidence_upgrade",
        "source_period_equivalence_claim": False,
    }


def recipient_matrix(frame):
    """Read the actual microunit role carrier; do not calculate policy roles."""
    return detail.recipient_matrix(frame, role_column="tax_unit_role_input")


def qualify_host_population(frame, values):
    """Bind the complete untouched clone pool before selecting model features.

    Reconstruct native membership from typed original-source projections and
    native-parent binding. Exact content comparison also catches a caller's
    omitted declared column; the materialized verifier remains mandatory.
    """
    sources = []
    for name in ("acs", "asec"):
        value = values[name]
        require(
            origin._source_document(value.payload)["arm"] == name, "HOST_ORIGIN_ARM"
        )
        sources.append(
            origin.AuthenticatedNativeOriginSource(value.payload, _token=origin._TOKEN)
        )
    parent_doc = codec.decode_json(values["parent_binding"].payload)
    require(parent_doc["schema"] == origin.BINDING_SCHEMA, "HOST_PARENT_SCHEMA")
    require(
        parent_doc["graph_parent_edges"]
        == {
            n: {
                "producer_key": values[n].producer_key,
                "artifact_key": values[n].key,
                "payload_sha256": codec.sha(values[n].payload),
            }
            for n in ("acs", "asec")
        },
        "HOST_NATIVE_ORIGIN_PRODUCERS",
    )
    parent = origin.PopulationOriginBinding(
        values["parent_binding"].payload, _token=origin._BOUND_TOKEN
    )
    expected = origin.bind_population_origins(
        frame, sources=sources, parent=parent
    ).document
    recorded = codec.decode_json(values["clone_binding"].payload)
    require(
        {
            k: v
            for k, v in recorded.items()
            if k not in ("graph_parent_edges", "population_node")
        }
        == expected,
        "HOST_FULL_ORIGIN_BINDING",
    )
    require(
        recorded["graph_parent_edges"]
        == {
            n: {
                "producer_key": values[n].producer_key,
                "artifact_key": values[n].key,
                "payload_sha256": codec.sha(values[n].payload),
            }
            for n in ("acs", "asec", "parent_binding")
        },
        "HOST_ORIGIN_PRODUCERS",
    )
    # Exact pair checks include every carried cell, linked entity and weight,
    # including zero-weight rows. This also refuses incomplete selected wages.
    matrix, mask = recipient_matrix(frame)
    native = frame.person.loc[
        frame.person[provenance.support_clone_index_column("person")].eq(0)
    ]
    channel = native[provenance.support_channel_column("person")]
    acs_person = native.loc[channel.eq("acs")].reset_index(drop=True)
    asec_person = native.loc[channel.eq("asec")].reset_index(drop=True)
    require(len(acs_person) > 0 and len(asec_person) > 0, "HOST_BOTH_ARMS")
    evidence = {
        "acs": acs_wage_evidence(acs_person),
        "asec": asec_wage_evidence(asec_person, values),
    }
    return {
        "schema": "microcosm.us.puf_diagnostic_host_features.v1",
        "recipe": recipe_document(),
        "full_population_content_sha256": detail.population_content(frame),
        "origin_binding_sha256": codec.sha(values["clone_binding"].payload),
        "wage_evidence": evidence,
        "matrix_sha256": codec.sha(matrix),
        "recipients": int(mask.sum()),
        "all_clone_rows_including_zero_weights": True,
        "release_eligible": False,
    }


CURRENT_SURVEY_ROUTE = "authenticated_current_survey_v1"
CURRENT_SURVEY_PROJECTION_PROTOCOL = "microcosm.us.survey-puf-host-projection.v1"
CURRENT_SURVEY_MAX_BYTES = 64 * 1024**2


def current_survey_recipient_matrix(frame, projection):
    """Pure typed-transport consumer; live owner verification remains mandatory.

    Both this parser and the existing matrix validate every clone pair. The
    projection retains current ASEC evidence bytes. ACS arithmetic/missingness
    remains the existing WAGP/ADJINC contract, including under-15 refusal for a
    missing selected earner. No serialized field confers source admission.
    """
    require(
        type(projection) is bytes and 0 < len(projection) <= CURRENT_SURVEY_MAX_BYTES,
        "SURVEY_HOST_PROJECTION_BOUND",
    )
    document = codec.decode_json(projection)
    require(
        set(document)
        == {
            "protocol",
            "route",
            "preparation_sha256",
            "allocation_sha256",
            "clone_sha256",
            "asec",
            "release_eligible",
        },
        "SURVEY_HOST_PROJECTION_FIELDS",
    )
    require(
        document["protocol"] == CURRENT_SURVEY_PROJECTION_PROTOCOL
        and document["route"] == CURRENT_SURVEY_ROUTE
        and document["release_eligible"] is False
        and all(
            codec._hash(document[k])
            for k in ("preparation_sha256", "allocation_sha256", "clone_sha256")
        ),
        "SURVEY_HOST_PROJECTION_CONTRACT",
    )
    asec = document["asec"]
    require(
        type(asec) is dict
        and set(asec)
        == {
            "preparation_sha256",
            "asec_native_sha256",
            "money_header",
            "money_header_sha256",
            "domain",
            "zero_origin_code",
            "columns",
            "rows",
        },
        "SURVEY_WAGE_FIELDS",
    )
    require(
        asec["preparation_sha256"] == document["preparation_sha256"]
        and codec._hash(asec["asec_native_sha256"])
        and codec.sha(codec.encode_json(asec["money_header"]))
        == asec["money_header_sha256"]
        and asec["money_header"]["target_year"] == 2024
        and asec["money_header"]["semantic"] == "annual_current_money"
        and asec["columns"]
        == [
            "stacked_person_id",
            "native_person_id",
            "income_year",
            "amount_f64le_hex",
            "status",
            "validity",
            "zero_origin",
        ],
        "SURVEY_WAGE_CONTRACT",
    )
    people = frame.person
    native_role = people[provenance.support_clone_index_column("person")].eq(0)
    native_acs = people.loc[
        native_role & people[provenance.support_channel_column("person")].eq("acs")
    ]
    native_asec = people.loc[
        native_role & people[provenance.support_channel_column("person")].eq("asec")
    ]
    acs_evidence = acs_wage_evidence(native_acs)
    rows = asec["rows"]
    require(
        type(rows) is list and len(rows) == len(native_asec) > 0, "SURVEY_WAGE_ROSTER"
    )
    require(
        type(asec["domain"]) is dict
        and set(asec["domain"])
        == {"name", "entity", "column", "grain", "minimum", "maximum", "zero_semantics"}
        and asec["domain"]["name"] == "WSAL_VAL"
        and asec["domain"]["entity"] == "person"
        and type(asec["zero_origin_code"]) is int,
        "SURVEY_WAGE_DOMAIN",
    )
    amounts = {}
    for row in rows:
        require(
            type(row) is list
            and len(row) == 7
            and all(type(row[i]) is int for i in (0, 1, 2, 4, 5, 6))
            and row[2] == 2024
            and row[5] == 1
            and all(0 <= row[i] <= 255 for i in (4, 5, 6))
            and type(row[3]) is str
            and len(row[3]) == 16
            and set(row[3]) <= set("0123456789abcdef"),
            "SURVEY_WAGE_ROW",
        )
        require(row[0] not in amounts, "SURVEY_WAGE_DUPLICATE")
        amount = np.frombuffer(bytes.fromhex(row[3]), dtype="<f8")[0]
        require(np.isfinite(amount) and amount >= 0, "SURVEY_WAGE_AMOUNT")
        amounts[row[0]] = (row[1], amount)
    # Reuse the maintained money owner's complete buffer/evidence/domain check.
    # These value objects do not issue money/source authority. The live verifier
    # compares the complete projection, including this source-owned domain.
    current_money._validate_field(
        current_money.MoneyField(
            "WSAL_VAL",
            b"".join(bytes.fromhex(row[3]) for row in rows),
            bytes(row[4] for row in rows),
            bytes(row[5] for row in rows),
            bytes(row[6] for row in rows),
        ),
        current_money.MoneyDomain(**asec["domain"]),
        len(rows),
        nominal=True,
        zero_origin_code=current_money.ZeroOrigin(asec["zero_origin_code"]),
    )
    source_id = provenance.support_source_id_column("person")
    original_id = provenance.spine_source_id_column("person")
    require(
        [r[0] for r in rows] == native_asec[source_id].tolist()
        and [r[1] for r in rows] == native_asec[original_id].tolist()
        and native_asec.source_year.astype(str).eq("2024").all(),
        "SURVEY_WAGE_ORIGIN",
    )
    wages = []
    columns = [
        provenance.support_channel_column("person"),
        source_id,
        original_id,
        "employment_income_before_lsr",
    ]
    for channel, stacked, native, value in people[columns].itertuples(
        index=False, name=None
    ):
        if channel == "asec":
            require(stacked in amounts, "SURVEY_WAGE_COVERAGE")
            original, amount = amounts[stacked]
            require(original == native, "SURVEY_WAGE_ORIGIN")
            wages.append(amount)
        else:
            require(channel == "acs", "SURVEY_WAGE_CHANNEL")
            wages.append(np.nan if pd.isna(value) else value)
    projected = pd.Series(
        wages, index=pd.Index(people.person_id.to_numpy()), dtype="float64"
    )
    matrix, mask = detail.recipient_matrix(
        frame, role_column="tax_unit_role_input", person_wages=projected
    )
    require(len(matrix) <= CURRENT_SURVEY_MAX_BYTES, "SURVEY_MATRIX_BOUND")
    return (
        matrix,
        mask,
        {
            "scope": SCOPE,
            "host_route": CURRENT_SURVEY_ROUTE,
            "projection_sha256": codec.sha(projection),
            "acs_wage_evidence": acs_evidence,
            "asec_money_header_sha256": asec["money_header_sha256"],
            "selected_tax_units": int(mask.sum()),
            "release_eligible": False,
            "source_period_equivalence_claim": False,
        },
    )
