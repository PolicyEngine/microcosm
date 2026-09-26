"""Invented S0101-only bytes exercise activation mechanics, never real evidence."""

import hashlib
import json
from dataclasses import replace

import pytest

from microcosm.build.us_runtime import survey_age_activation as activation
from microcosm.build.us_runtime import survey_age_sources as source


def invented_documents():
    """Obvious arithmetic fixture values, independent of all measured inputs."""
    row = {"GEO_ID": "0100000US", "NAME": "United States", "us": "1"}
    variables = {}
    bands = activation.age.NATIONAL_AGE_ACTIVATION.bands
    for ordinal, (variable, label) in enumerate(
        [("S0101_C01_001", "Estimate!!Total!!Total population")]
        + [(band.variable, band.label) for band in bands]
    ):
        count = sum(range(1, 19)) if ordinal == 0 else ordinal
        row.update(
            {
                variable + "E": str(count),
                variable + "EA": None,
                variable + "M": "16.45",
                variable + "MA": None,
            }
        )
        variables[variable + "E"] = {
            "group": "S0101",
            "concept": "Age and Sex",
            "predicateType": "int",
            "label": label,
        }
    return {"variables": variables}, [list(row), list(row.values())]


def capture_fixture(root, *, metadata=None, data=None):
    default_metadata, default_data = invented_documents()
    documents = {
        "metadata": default_metadata if metadata is None else metadata,
        "data": default_data if data is None else data,
    }
    digests = {}
    for request in source.survey_age_requests():
        payload = json.dumps(documents[request["kind"]], separators=(",", ":")).encode()
        source.capture_survey_age_response(root, request, payload)
        digests[request["kind"]] = hashlib.sha256(payload).hexdigest()
    return activation.SurveyAgeActivation(
        metadata_sha256=digests["metadata"], data_sha256=digests["data"]
    )


def test_invented_capture_yields_eighteen_ordered_terms_with_declared_source_semantics(
    tmp_path,
):
    declaration = capture_fixture(tmp_path)
    registry = activation.activate_survey_age_targets(tmp_path, declaration=declaration)
    assert [s.value for s in registry] == list(range(1, 19))
    assert [s.name for s in registry] == [f"S0101_C01_{i:03}" for i in range(2, 20)]
    assert all(
        s.metadata["reference_sha256"] == activation.activation_digest(declaration)
        for s in registry
    )
    assert all(s.se == pytest.approx(10) and s.period == "2024" for s in registry)
    assert all(
        "ASEC 2025" in s.notes and "income year 2024" in s.notes for s in registry
    )
    # This is a source-documentation protocol exercised with invented values,
    # not an independently authenticated real Census capture.
    assert all(s.metadata["evidence_scope"] == "source_documented" for s in registry)


@pytest.mark.parametrize(
    "change", ["estimate", "total", "duplicate", "geography", "foreign", "annotation"]
)
def test_malformed_data_refuses_instead_of_partial_activation(tmp_path, change):
    metadata, data = invented_documents()
    header, values = data
    if change == "estimate":
        values[header.index("S0101_C01_002E")] = "1.5"
    elif change == "total":
        values[header.index("S0101_C01_001E")] = "1"
    elif change == "duplicate":
        header.append(header[-1])
        values.append(values[-1])
    elif change == "geography":
        values[header.index("us")] = "2"
    elif change == "foreign":
        header.append("UNSUPPORTED_001E")
        values.append("invented-unconsumed")
    else:
        values[header.index("S0101_C01_002EA")] = "(X)"
    declaration = capture_fixture(tmp_path, metadata=metadata, data=data)
    with pytest.raises(ValueError):
        activation.activate_survey_age_targets(tmp_path, declaration=declaration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("group", "UNSUPPORTED"),
        ("label", "Wrong age band"),
        ("concept", "Other"),
        ("predicateType", "float"),
    ],
)
def test_publisher_definition_is_part_of_activation(tmp_path, field, value):
    metadata, data = invented_documents()
    metadata["variables"]["S0101_C01_002E"][field] = value
    declaration = capture_fixture(tmp_path, metadata=metadata, data=data)
    with pytest.raises(ValueError):
        activation.activate_survey_age_targets(tmp_path, declaration=declaration)


def test_controlled_margin_keeps_unknown_standard_error(tmp_path):
    metadata, data = invented_documents()
    header, values = data
    values[header.index("S0101_C01_002M")] = "-555555555"
    values[header.index("S0101_C01_002MA")] = "*****"
    declaration = capture_fixture(tmp_path, metadata=metadata, data=data)
    registry = activation.activate_survey_age_targets(tmp_path, declaration=declaration)
    assert registry.specs[0].se is None


def test_changed_pin_or_declaration_cannot_reuse_capture(tmp_path):
    declaration = capture_fixture(tmp_path)
    with pytest.raises(ValueError, match="PIN"):
        activation.activate_survey_age_targets(
            tmp_path, declaration=replace(declaration, data_sha256="0" * 64)
        )
    binding = activation.activation_binding(declaration)
    binding["period"] = "2025"
    with pytest.raises(ValueError, match="DECLARATION"):
        activation.declaration_from_binding(binding)


def test_registry_relabel_or_value_change_fails_exact_activation_binding(tmp_path):
    from microcosm.calibrate.registry import TargetRegistry

    declaration = capture_fixture(tmp_path)
    registry = activation.activate_survey_age_targets(tmp_path, declaration=declaration)
    altered = TargetRegistry(
        [replace(s, value=s.value + 1) for s in registry], country="us"
    )
    with pytest.raises(ValueError, match="REGISTRY"):
        activation.verify_survey_age_targets(
            tmp_path, declaration=declaration, registry=altered
        )
