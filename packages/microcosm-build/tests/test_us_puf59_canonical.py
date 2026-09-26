"""Invented source/control coverage; no native files or tax engine."""


def test_puf59_canonical_source_and_artifact_controls():
    import copy
    import hashlib
    import json
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from microcosm.build.us_runtime import full_puf_enrichment as enrichment
    from microcosm.build.us_runtime import puf55_canonical_donor as donor55
    from microcosm.build.us_runtime import puf59_canonical as canonical
    from microcosm.build.us_runtime import puf59_canonical_artifact as artifact
    from microcosm.build.us_runtime import puf_full_source as full
    from microcosm.build.us_runtime import puf_interest_components as interest
    from microcosm.build.us_runtime import puf_raw_source as raw
    from microcosm.build.us_runtime import puf_target2024_growth as growth

    checks = []

    def check(name, value):
        assert bool(value), name
        checks.append(name)

    def refuses(name, fn, code):
        try:
            fn()
        except ValueError as e:
            check(name, str(e).startswith(code))
        else:
            raise AssertionError(name + " accepted")

    def fixture(n=2048, override=None):
        definition = raw.packaged_definition()
        rows = []
        for i in range(n + 4):
            agg = i >= n
            recid = 999996 + i - n if agg else 501000 + i
            row = {k: "0" for k in definition.main.delivered_header}
            mars = 0 if agg else 1 + i % 4
            secondary = int(mars == 2)
            xfpt = int(not agg and i % 13 != 0)
            row.update(
                RECID=str(recid),
                FLPDYR="2014" if i % 7 == 0 else "2015",
                FLPDMO="12",
                MARS=str(mars),
                DSI="0",
                S006=str(125 + i % 200),
                XFPT=str(xfpt),
                XFST=str(secondary),
                XTOT=str(xfpt + secondary),
            )
            if not agg:
                row.update(
                    E00200=str(40000 + i),
                    E00900=str(-300 if i % 5 == 0 else 2000 + i),
                    E00600="120",
                    E00650="30",
                    E25940="700",
                    E25980="500",
                    E25920="100",
                    E25960="50",
                    E26110="20",
                    E26170="200",
                    E26190="300",
                    E26160="10",
                    E26180="20",
                    E26100="5",
                    E26270="1495",
                    E00100=str(500 * i - 10000),
                    E19200=str(1000 + i),
                    E02400="50",
                    E03230="100",
                    E87530="300",
                )
            else:
                row["E19200"] = ""
            if override and i == 0:
                row.update(override)
            rows.append(row)
        mh = definition.main.delivered_header
        dh = definition.demographic.delivered_header
        main = (
            ",".join(mh)
            + "\n"
            + "\n".join(",".join(row[k] for k in mh) for row in rows)
            + "\n"
        ).encode()
        d = dict(
            RECID="501000",
            AGEDP1="0",
            AGEDP2="0",
            AGEDP3="0",
            AGERANGE="2",
            EARNSPLIT="0",
            GENDER="1",
        )
        demo = (",".join(dh) + "\n" + ",".join(d[k] for k in dh) + "\n").encode()
        doc = raw.definition_document_json(definition)
        doc.update(route="test_fixture", authority="invented_fixture_nonauthority")
        for name, payload, count in (("main", main, n + 4), ("demographic", demo, 1)):
            doc["sources"][name].update(
                bytes=len(payload),
                data_records=count,
                sha256=hashlib.sha256(payload).hexdigest(),
                git_blob_sha1=hashlib.sha1(
                    b"blob " + str(len(payload)).encode() + b"\0" + payload
                ).hexdigest(),
            )
        return full.decode_full_puf_source(main, demo, raw.fixture_definition(doc))

    asset = interest.puf_e19200_interest_components_asset_identity()
    check(
        "exact_public_interest_asset",
        asset["asset_sha256"] == canonical.INTEREST_ASSET_SHA256,
    )

    def build(source, **kwargs):
        return canonical.construct_canonical_puf59(
            source,
            interest_bands=interest.US_PUF_E19200_AGI_BANDS,
            interest_asset_sha256=asset["asset_sha256"],
            **kwargs,
        )

    small = fixture(64)
    small_result = build(small)
    check(
        "small64_actual_raw_to59",
        len(small_result.recids) == 64 and len(small_result.columns) == 59,
    )
    source = fixture()
    before = full.encode_full_puf_source(source)
    result = build(source)
    check(
        "larger2048_actual_raw_to59",
        len(result.recids) == 2048 and len(result.columns) == 59,
    )
    check("all_source_bytes_unchanged", full.encode_full_puf_source(source) == before)
    check(
        "all_ordinary_ids_and_missing_demographics_retained",
        np.array_equal(result.recids, source.status["RECID"][source.ordinary])
        and len(result.recids) == 2048,
    )
    check(
        "source_s006_divided_once",
        np.array_equal(
            result.design_weight,
            source.status["S006"][source.ordinary].astype(float) / 100,
        ),
    )
    check(
        "capacity_is_constant_one_not_return_size",
        np.all(result.person_incidence_capacity == 1)
        and np.any(result.source_predictors["puf_2015_capped_return_size"] > 1),
    )
    check(
        "no_six_mortgage_destinations_or_invented_persons",
        not any(
            k.startswith(("first_home_mortgage_", "second_home_mortgage_"))
            for k in result.columns
        )
        and not hasattr(result, "person"),
    )
    check(
        "raw_non2015_fields_not_repriced_again",
        result.receipt["source_statistical_year"] == 2015
        and result.receipt["growth"]["no_extra_raw_row_year_cpi"],
    )
    selected = tuple(int(x) for x in result.recids[:64])
    replay = build(
        source,
        selected_recids=selected,
        qbi_employment_calibration=result.qbi_calibration,
    )
    check(
        "full_cohort_fit_prefix_replays_all59",
        all(
            np.array_equal(replay.columns[k], v[:64]) for k, v in result.columns.items()
        ),
    )
    cpi = build(
        source,
        qbi_employment_calibration=result.qbi_calibration,
        growth_scheme="cpi_only",
    )
    check(
        "cpi_sensitivity_preserves_ids_weight_capacity",
        np.array_equal(cpi.recids, result.recids)
        and np.array_equal(cpi.design_weight, result.design_weight)
        and np.array_equal(
            cpi.person_incidence_capacity, result.person_incidence_capacity
        ),
    )
    out, aux, status = full.observed_and_derived_return_columns(source)
    modeled, receipt = canonical.baseline_modeled_columns(
        aux,
        n=2048,
        interest_bands=interest.US_PUF_E19200_AGI_BANDS,
        interest_asset_sha256=asset["asset_sha256"],
    )
    check(
        "all11_baseline_modeled_outputs_explicit",
        len(modeled) == 11 and receipt["origin"] == "modeled_not_observed",
    )
    check(
        "interest_decomposition_bit_conserves_e19200",
        np.array_equal(
            modeled["home_mortgage_interest"] + modeled["investment_interest_expense"],
            aux["raw_total_interest_deduction"].astype(float),
        ),
    )
    check(
        "ss_carrier_preserves_observed_total",
        np.array_equal(
            sum(
                modeled[k]
                for k in (
                    "social_security_retirement",
                    "social_security_disability",
                    "social_security_dependents",
                    "social_security_survivors",
                )
            ),
            aux["raw_total_social_security"],
        ),
    )
    check(
        "tuition_proxy_is_explicit_maximum",
        np.all(modeled["qualified_tuition_expenses"] == 300),
    )
    refuses(
        "negative_interest_is_not_clipped",
        lambda: build(fixture(64, {"E19200": "-1"})),
        "PUF59_MODEL_NEGATIVE_SOURCE:",
    )
    refuses(
        "wrong_interest_asset_identity_refused",
        lambda: canonical.baseline_modeled_columns(
            aux,
            n=2048,
            interest_bands=interest.US_PUF_E19200_AGI_BANDS,
            interest_asset_sha256="0" * 64,
        ),
        "PUF59_INTEREST_ASSET_IDENTITY",
    )
    bad = list(interest.US_PUF_E19200_AGI_BANDS)
    bad[3] = replace(bad[3], lower_bound=bad[3].lower_bound + 1)
    refuses(
        "band_gap_refused",
        lambda: canonical.baseline_modeled_columns(
            aux, n=2048, interest_bands=bad, interest_asset_sha256=asset["asset_sha256"]
        ),
        "PUF59_INTEREST_CONTIGUOUS",
    )
    refuses(
        "aggregate_selection_refused",
        lambda: build(source, selected_recids=(999996,)),
        "FULL_SELECTION_UNIVERSE",
    )
    payload = artifact.encode_canonical_puf59(result)
    arrays, receipt = artifact.decode_canonical_puf59(payload)
    check(
        "canonical59_typed_envelope_byte_replay",
        artifact.reencode_canonical_puf59(arrays, receipt) == payload,
    )
    check(
        "canonical59_typed_envelope_all_values_replay",
        all(np.array_equal(arrays[k], v) for k, v in result.columns.items()),
    )
    changed = payload[:-1] + bytes([payload[-1] ^ 1])
    refuses(
        "canonical59_body_corruption_refused",
        lambda: artifact.decode_canonical_puf59(changed),
        "PUF59_ARTIFACT_BODY_IDENTITY",
    )
    refuses(
        "canonical59_truncation_refused",
        lambda: artifact.decode_canonical_puf59(payload[:-8]),
        "PUF59_ARTIFACT_BODY_IDENTITY",
    )
    refuses(
        "canonical59_unknown_magic_refused",
        lambda: artifact.decode_canonical_puf59(b"BADMAGIC" + payload[8:]),
        "PUF59_ARTIFACT_MAGIC",
    )
    bad_arrays = {
        **arrays,
        "puf_person_incidence_capacity": np.full(2048, 2, dtype=np.int64),
    }
    refuses(
        "canonical59_incidence_capacity_not_person_count",
        lambda: artifact.reencode_canonical_puf59(bad_arrays, receipt),
        "PUF59_ARTIFACT_CAPACITY",
    )
    try:
        arrays["employment_income_before_lsr"][0] = 2
    except ValueError:
        checks.append("canonical59_decoded_arrays_immutable")
    else:
        raise AssertionError("decoded arrays mutable")
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": arrays["RECID"],
            "weight": arrays["weight"],
            "filing_status_code": arrays["puf_2015_filing_status_code"],
            "puf_person_incidence_capacity": arrays["puf_person_incidence_capacity"],
            **{k: arrays[k] for k in enrichment.PUF59.targets},
            **{k: arrays[k] for k in enrichment.PUF59.source_predictors},
        }
    )
    known = pd.DataFrame(
        True,
        index=tax_unit.index,
        columns=(
            "weight",
            "filing_status_code",
            "puf_person_incidence_capacity",
            *enrichment.PUF59.person_outputs,
            *enrichment.PUF59.tax_unit_outputs,
            *enrichment.PUF59.source_predictors,
        ),
    )
    donor = enrichment.canonical_full_puf_donor(
        None,
        tax_unit,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=enrichment.PUF59.person_outputs,
        profile=enrichment.PUF59,
    )
    check(
        "actual_profile59_canonical_donor_ordered_contract",
        tuple(donor.columns)
        == (
            *enrichment.PUF59.predictors,
            *enrichment.PUF59.targets,
            "weight",
            "puf_person_incidence_capacity",
        ),
    )
    check(
        "actual_profile59_conditioning_uses_total_schedule_c",
        np.array_equal(
            donor[enrichment.PUF59.predictors[3]].to_numpy(),
            arrays["self_employment_income_before_lsr"]
            + arrays["sstb_self_employment_income_before_lsr"],
        ),
    )
    check(
        "actual_profile59_retains_exact_source_size_measurement",
        np.array_equal(
            donor.puf_2015_capped_return_size.to_numpy(),
            result.source_predictors["puf_2015_capped_return_size"],
        ),
    )
    check(
        "actual_profile59_weight_unchanged",
        np.array_equal(donor.weight.to_numpy(), result.design_weight),
    )

    # Reuse actual source decoding, canonical construction and envelope checks.
    # The old four SS carriers represent one total, never observed components.
    for candidate in (small_result, result):
        packed = artifact.encode_canonical_puf59(candidate)
        digest = hashlib.sha256(packed).hexdigest()
        projected, metadata = donor55.canonical_puf55_donor_from_artifact(
            packed, expected_artifact_sha256=digest
        )
        profile = enrichment.PUF55_SURVEY_SS
        check(
            "puf55_exact_order_" + str(len(candidate.recids)),
            tuple(projected.columns)
            == (
                *profile.predictors,
                *profile.targets,
                "weight",
                "puf_person_incidence_capacity",
            ),
        )
        check(
            "puf55_identity_axis_" + str(len(candidate.recids)),
            projected.index.name == "tax_unit_id"
            and np.array_equal(projected.index.to_numpy(), candidate.recids),
        )
        check(
            "puf55_total_predictor_is_grown_source_total",
            np.array_equal(
                projected[enrichment.SURVEY_SS_TOTAL_PREDICTOR].to_numpy(),
                candidate.columns["social_security_retirement"],
            ),
        )
        check(
            "puf55_all_retained_targets_and_weight_unchanged",
            all(
                np.array_equal(projected[k].to_numpy(), candidate.columns[k])
                for k in profile.targets
            )
            and np.array_equal(projected.weight.to_numpy(), candidate.design_weight)
            and (projected.puf_person_incidence_capacity == 1).all(),
        )
        check(
            "puf55_no_component_carriers_or_source_authority",
            not set(enrichment.SURVEY_SS_COMPONENTS) & set(projected.columns)
            and metadata["artifact_sha256"] == digest
            and metadata["source_receipt_sha256"] == candidate.receipt["sha256"]
            and metadata["social_security_total"]["origin"] == "modeled_transport"
            and metadata["release_eligible"] is False
            and metadata["source_authority_granted"] is False,
        )
        check(
            "puf55_source_bytes_unchanged",
            artifact.encode_canonical_puf59(candidate) == packed,
        )
        explicit, explicit_metadata = donor55.canonical_puf55_donor_from_artifact(
            packed,
            expected_artifact_sha256=digest,
            profile=enrichment.PUF55_SURVEY_SS,
        )
        check(
            "puf55_default_nine_profile_unchanged",
            explicit.equals(projected) and explicit_metadata == metadata,
        )
        eight, eight_metadata = donor55.canonical_puf55_donor_from_artifact(
            packed,
            expected_artifact_sha256=digest,
            profile=enrichment.PUF55_SURVEY_SS_NO_TOTAL,
        )
        check(
            "puf55_eight_exact_projection_without_source_or_target_change",
            eight.equals(projected.drop(columns=[enrichment.SURVEY_SS_TOTAL_PREDICTOR]))
            and tuple(eight.columns)
            == (
                *enrichment.PUF55_SURVEY_SS_NO_TOTAL.predictors,
                *profile.targets,
                "weight",
                "puf_person_incidence_capacity",
            )
            and artifact.encode_canonical_puf59(candidate) == packed,
        )
        check(
            "puf55_eight_omission_is_explicit_and_grants_no_authority",
            eight_metadata["profile"] == enrichment.PUF55_SURVEY_SS_NO_TOTAL.value
            and eight_metadata["ordered_columns"] == list(eight.columns)
            and eight_metadata["social_security_total"]["predictor"] is None
            and eight_metadata["social_security_total"]["raw_field"] == "E02400"
            and eight_metadata["artifact_sha256"] == digest
            and eight_metadata["source_receipt_sha256"] == candidate.receipt["sha256"]
            and eight_metadata["source_authority_granted"] is False
            and eight_metadata["release_eligible"] is False,
        )
        refuses(
            "puf55_expected_artifact_identity_required",
            lambda packed=packed: donor55.canonical_puf55_donor_from_artifact(
                packed, expected_artifact_sha256="0" * 64
            ),
            "PUF55_DONOR_ARTIFACT_IDENTITY",
        )

    for invalid_profile in (
        enrichment.PUF59,
        enrichment.FULL65,
        enrichment.PUF55_SURVEY_SS_NO_TOTAL.value,
        None,
    ):
        refuses(
            "puf55_explicit_closed_profile_required",
            lambda invalid_profile=invalid_profile: (
                donor55.canonical_puf55_donor_from_artifact(
                    packed,
                    expected_artifact_sha256=digest,
                    profile=invalid_profile,
                )
            ),
            "PUF55_DONOR_PROFILE",
        )

    # Correctly rehashed, internally consistent artifacts still cannot change
    # the upstream SS carrier semantics accepted by this projection.
    def repacked_ss_change(*, column=None, method=None):
        changed_arrays = dict(arrays)
        changed_receipt = copy.deepcopy(receipt)
        if column is not None:
            changed_arrays[column] = np.ones(len(arrays["RECID"]))
            changed_receipt["growth"]["output_values_sha256"] = growth._digest(
                changed_arrays
            )
        if method is not None:
            changed_receipt["model_assumptions"]["ss_method"] = method
        changed_receipt.pop("sha256")
        changed_receipt["sha256"] = hashlib.sha256(
            json.dumps(
                changed_receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        return artifact.reencode_canonical_puf59(changed_arrays, changed_receipt)

    for altered in (
        repacked_ss_change(column="social_security_disability"),
        repacked_ss_change(method="Individual component observations"),
    ):
        refuses(
            "puf55_changed_carrier_contract_refused",
            lambda altered=altered: donor55.canonical_puf55_donor_from_artifact(
                altered, expected_artifact_sha256=hashlib.sha256(altered).hexdigest()
            ),
            "PUF55_DONOR_SS_CARRIER",
        )

        refuses(
            "puf55_eight_cannot_bypass_carrier_contract",
            lambda altered=altered: donor55.canonical_puf55_donor_from_artifact(
                altered,
                expected_artifact_sha256=hashlib.sha256(altered).hexdigest(),
                profile=enrichment.PUF55_SURVEY_SS_NO_TOTAL,
            ),
            "PUF55_DONOR_SS_CARRIER",
        )

    # Integrity successor: domain-valid mutations cannot inherit old provenance.
    import struct

    check(
        "canonical_v2_and_exact_band_fingerprint",
        result.receipt["schema"] == canonical.VERSION
        and result.receipt["model_assumptions"]["interest_band_facts_sha256"]
        == canonical.INTEREST_BAND_FACTS_SHA256,
    )
    for field, replacement in (
        ("RECID", arrays["RECID"][::-1]),
        ("weight", arrays["weight"] * 2),
        ("puf_2015_filing_status_code", arrays["puf_2015_filing_status_code"] % 4 + 1),
        ("puf_2015_capped_return_size", np.ones(2048, dtype=np.int64)),
    ):
        refuses(
            "valid_domain_prefix_mutation_refused_" + field,
            lambda field=field, replacement=replacement: (
                artifact.reencode_canonical_puf59(
                    {**arrays, field: replacement}, receipt
                )
            ),
            "PUF59_ARTIFACT_PREFIX_BINDING",
        )
    for field in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "business_is_sstb",
    ):
        changed_array = arrays[field].copy()
        if field == "business_is_sstb":
            changed_array = 1 - changed_array
        else:
            changed_array = changed_array[::-1]
        refuses(
            "valid_domain_output_mutation_refused_" + field,
            lambda field=field, changed_array=changed_array: (
                artifact.reencode_canonical_puf59(
                    {**arrays, field: changed_array}, receipt
                )
            ),
            "PUF59_ARTIFACT_GROWTH_BINDING",
        )
    prefix = len(artifact.MAGIC) + 4
    header_length = struct.unpack("<I", payload[len(artifact.MAGIC) : prefix])[0]
    header = json.loads(payload[prefix : prefix + header_length])
    body = payload[prefix + header_length :]

    def wrap(header, body, *, pretty=False):
        text = (
            json.dumps(header, sort_keys=True, indent=2).encode()
            if pretty
            else json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        return artifact.MAGIC + struct.pack("<I", len(text)) + text + body

    altered = bytearray(body)
    offset = artifact.NAMES.index("employment_income_before_lsr") * 2048 * 8
    old_value = struct.unpack_from("<d", altered, offset)[0]
    struct.pack_into("<d", altered, offset, old_value + 1)
    altered_header = {**header, "body_sha256": hashlib.sha256(altered).hexdigest()}
    refuses(
        "recomputed_body_hash_cannot_reuse_old_receipt",
        lambda: artifact.decode_canonical_puf59(wrap(altered_header, bytes(altered))),
        "PUF59_ARTIFACT_GROWTH_BINDING",
    )
    refuses(
        "noncanonical_json_header_refused",
        lambda: artifact.decode_canonical_puf59(wrap(header, body, pretty=True)),
        "PUF59_ARTIFACT_HEADER_CANONICAL",
    )
    utf16 = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-16")
    refuses(
        "utf16_json_header_refused",
        lambda: artifact.decode_canonical_puf59(
            artifact.MAGIC + struct.pack("<I", len(utf16)) + utf16 + body
        ),
        "PUF59_ARTIFACT_HEADER_CANONICAL",
    )
    refuses(
        "legacy_v1_magic_not_silently_promoted",
        lambda: artifact.decode_canonical_puf59(b"MCPUF59\x00" + payload[8:]),
        "PUF59_ARTIFACT_MAGIC",
    )

    def rehash_receipt(changed):
        changed = dict(changed)
        changed.pop("sha256", None)
        changed["sha256"] = hashlib.sha256(
            json.dumps(
                changed, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        return changed

    for field, value in (
        ("rows", True),
        ("input_money_year", True),
        ("output_money_year", True),
    ):
        invalid = rehash_receipt({**receipt, field: value})
        refuses(
            "boolean_receipt_integer_refused_" + field,
            lambda invalid=invalid: artifact.reencode_canonical_puf59(arrays, invalid),
            "PUF59_ARTIFACT_RECEIPT_CONTRACT",
        )
    missing = rehash_receipt({k: v for k, v in receipt.items() if k != "rows"})
    refuses(
        "missing_receipt_rows_valueerror",
        lambda: artifact.reencode_canonical_puf59(arrays, missing),
        "PUF59_ARTIFACT_RECEIPT_CONTRACT",
    )
    cpipayload = artifact.encode_canonical_puf59(cpi, expected_growth_scheme="cpi_only")
    refuses(
        "cpi_export_requires_explicit_scheme",
        lambda: artifact.encode_canonical_puf59(cpi),
        "PUF59_ARTIFACT_GROWTH_SCHEME",
    )
    refuses(
        "cpi_import_requires_explicit_scheme",
        lambda: artifact.decode_canonical_puf59(cpipayload),
        "PUF59_ARTIFACT_GROWTH_SCHEME",
    )
    ca, cr = artifact.decode_canonical_puf59(
        cpipayload, expected_growth_scheme="cpi_only"
    )
    check(
        "explicit_cpi_schema_roundtrip",
        artifact.reencode_canonical_puf59(ca, cr, expected_growth_scheme="cpi_only")
        == cpipayload,
    )
    altered_bands = list(interest.US_PUF_E19200_AGI_BANDS)
    last = altered_bands[-1]
    altered_bands[-1] = replace(
        last,
        investment_interest_amount=last.investment_interest_amount + 1,
        total_interest_paid_amount=last.total_interest_paid_amount + 1,
    )
    refuses(
        "unrepresented_band_tampering_refused",
        lambda: canonical.baseline_modeled_columns(
            aux,
            n=2048,
            interest_bands=altered_bands,
            interest_asset_sha256=canonical.INTEREST_ASSET_SHA256,
        ),
        "PUF59_INTEREST_BAND_FACTS_IDENTITY",
    )
    check(
        "embedded_recipe_hash_verified",
        hashlib.sha256(growth._RECIPE_JSON.encode()).hexdigest()
        == growth.RECIPE_SHA256,
    )
    globals()["INTEGRITY_BEHAVIORAL_CHECKS"] = len(checks)
