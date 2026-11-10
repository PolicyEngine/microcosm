"""Diagnostics schema 9: hierarchy labels declared once, in catalogs.

Schema 8 copies every provider, category, geography and dimension label onto
every target row, so one geography or one dimension value could be labelled
two ways in one file. Schema 9 declares each label once in
``hierarchy_catalogs`` and rows refer to it by ID. The writer builds the
catalogs from the rows and refuses a document whose rows give one identifier
two labels, unless an explicit label choice resolves it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from microcosm.diagnostics import (
    CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION,
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    HIERARCHY_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS,
    SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS,
    CalibrationDiagnosticsV8,
    CalibrationDiagnosticsV9,
    HierarchyLabelChoices,
    HierarchyLabelConflictError,
    catalog_calibration_diagnostics,
    expand_calibration_diagnostics,
    parse_calibration_diagnostics,
    write_calibration_diagnostics,
)
from test_support.microcosm_diagnostics.payloads import schema_8_payload

AGI = "us:statutes/26/62#adjusted_gross_income"


def _target(
    name: str,
    *,
    geography: dict,
    dimensions: list[dict],
    provider: dict | None = None,
    category: dict | None = None,
) -> dict:
    row = copy.deepcopy(schema_8_payload()["targets"][0])
    provider = provider or {"id": "irs_soi", "label": "IRS Statistics of Income"}
    category = category or {
        "id": "irs_soi.eitc",
        "label": "Earned income tax credit",
        "provider_id": provider["id"],
    }
    row.update(name=f"{name}@2024", target_name=name)
    row["hierarchy"] = {
        "provider": provider,
        "category": category,
        "geography": geography,
        "dimensions": dimensions,
        "target": {"id": name, "label": name.replace(".", " ")},
    }
    return row


def _payload(targets: list[dict]) -> dict:
    payload = schema_8_payload()
    payload["targets"] = targets
    payload["target_surface"]["n_targets"] = len(targets)
    payload["target_surface"]["constraint_matrix"]["rows"] = len(targets)
    payload["target_registry"]["n_specs"] = len(targets)
    return payload


UNITED_STATES = {"id": "0100000US", "label": "United States", "level": "country"}
MAINE = {"id": "0400000US23", "label": "Maine", "level": "state"}


def _band(value_id: str, value_label: str) -> dict:
    return {
        "id": AGI,
        "label": "Adjusted gross income",
        "value_id": value_id,
        "value_label": value_label,
    }


def _status(value_id: str, value_label: str) -> dict:
    return {
        "id": "filing_status",
        "label": "Filing status",
        "value_id": value_id,
        "value_label": value_label,
    }


def _us_payload() -> dict:
    """Three rows; both dimensions have a value ``all``."""

    return _payload(
        [
            _target(
                "irs_soi.eitc.under_1.all",
                geography=UNITED_STATES,
                dimensions=[
                    _band("under_1", "No adjusted gross income"),
                    _status("all", "All filing statuses"),
                ],
            ),
            _target(
                "irs_soi.eitc.all.joint",
                geography=UNITED_STATES,
                dimensions=[
                    _band("all", "All returns"),
                    _status("joint", "Married filing jointly"),
                ],
            ),
            _target("irs_soi.eitc.maine", geography=MAINE, dimensions=[]),
        ]
    )


def _model(payload: dict) -> CalibrationDiagnosticsV8:
    return CalibrationDiagnosticsV8.model_validate(payload)


def test_the_published_schema_is_9_and_producers_still_assemble_schema_8() -> None:
    assert CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION == 9
    assert CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION == 8
    assert HIERARCHY_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS == frozenset({8, 9})
    assert SUPPORTED_CALIBRATION_DIAGNOSTICS_SCHEMA_VERSIONS == frozenset({6, 7, 8, 9})


def test_catalogs_declare_each_label_once_and_rows_refer_by_id() -> None:
    catalogued = catalog_calibration_diagnostics(_model(_us_payload()))
    document = json.loads(catalogued.model_dump_json(exclude_defaults=True))

    assert document["schema_version"] == 9
    # Catalog entries are listed in the order the rows first use them.
    assert document["hierarchy_catalogs"] == {
        "providers": [{"id": "irs_soi", "label": "IRS Statistics of Income"}],
        "categories": [
            {
                "id": "irs_soi.eitc",
                "label": "Earned income tax credit",
                "provider_id": "irs_soi",
            }
        ],
        "geographies": [UNITED_STATES, MAINE],
        "dimensions": [
            {
                "id": AGI,
                "label": "Adjusted gross income",
                "values": [
                    {"id": "under_1", "label": "No adjusted gross income"},
                    {"id": "all", "label": "All returns"},
                ],
            },
            {
                "id": "filing_status",
                "label": "Filing status",
                "values": [
                    {"id": "all", "label": "All filing statuses"},
                    {"id": "joint", "label": "Married filing jointly"},
                ],
            },
        ],
    }
    assert [row["hierarchy"] for row in document["targets"]] == [
        {
            "category_id": "irs_soi.eitc",
            "geography_id": "0100000US",
            "dimensions": [
                {"id": AGI, "value_id": "under_1"},
                {"id": "filing_status", "value_id": "all"},
            ],
            "target": {
                "id": "irs_soi.eitc.under_1.all",
                "label": "irs_soi eitc under_1 all",
            },
        },
        {
            "category_id": "irs_soi.eitc",
            "geography_id": "0100000US",
            "dimensions": [
                {"id": AGI, "value_id": "all"},
                {"id": "filing_status", "value_id": "joint"},
            ],
            "target": {
                "id": "irs_soi.eitc.all.joint",
                "label": "irs_soi eitc all joint",
            },
        },
        {
            "category_id": "irs_soi.eitc",
            "geography_id": "0400000US23",
            "dimensions": [],
            "target": {"id": "irs_soi.eitc.maine", "label": "irs_soi eitc maine"},
        },
    ]
    # Every other field is schema 8's.
    for written, assembled in zip(
        document["targets"], _us_payload()["targets"], strict=True
    ):
        assert {key: value for key, value in written.items() if key != "hierarchy"} == {
            key: value for key, value in assembled.items() if key != "hierarchy"
        }


def test_expanding_schema_9_gives_back_the_schema_8_document() -> None:
    model = _model(_us_payload())

    assert expand_calibration_diagnostics(catalog_calibration_diagnostics(model)) == (
        model
    )


def test_the_writer_writes_schema_9_and_reports_it(tmp_path: Path) -> None:
    model = _model(_us_payload())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "available"
    assert outcome.schema_version == 9
    written = json.loads(path.read_text())
    assert written == json.loads(
        catalog_calibration_diagnostics(model).model_dump_json(exclude_defaults=True)
    )
    parsed = parse_calibration_diagnostics(written)
    assert isinstance(parsed, CalibrationDiagnosticsV9)
    assert expand_calibration_diagnostics(parsed) == model


def _conflict_payload(kind: str) -> dict:
    payload = _us_payload()
    first, second, third = payload["targets"]
    if kind == "geography label":
        third["hierarchy"]["geography"] = {**UNITED_STATES, "label": "USA"}
    elif kind == "geography level":
        third["hierarchy"]["geography"] = {**UNITED_STATES, "level": "nation"}
    elif kind == "dimension value label":
        second["hierarchy"]["dimensions"][0] = _band(
            "under_1", "No adjusted gross income and deficit"
        )
    elif kind == "dimension label":
        second["hierarchy"]["dimensions"][1]["label"] = "Filing status of return"
    elif kind == "provider label":
        third["hierarchy"]["provider"] = {"id": "irs_soi", "label": "IRS SOI"}
        third["hierarchy"]["category"]["provider_id"] = "irs_soi"
    elif kind == "category label":
        third["hierarchy"]["category"] = {
            **first["hierarchy"]["category"],
            "label": "EITC",
        }
    elif kind == "category provider":
        third["hierarchy"]["provider"] = {"id": "irs", "label": "IRS"}
        third["hierarchy"]["category"] = {
            **first["hierarchy"]["category"],
            "provider_id": "irs",
        }
    return payload


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (
            "geography label",
            "geography 0100000US: 'United States' (first target "
            "irs_soi.eitc.under_1.all@2024), 'USA' (first target "
            "irs_soi.eitc.maine@2024)",
        ),
        (
            "geography level",
            "geography 0100000US level: 'country' (first target "
            "irs_soi.eitc.under_1.all@2024), 'nation' (first target "
            "irs_soi.eitc.maine@2024)",
        ),
        (
            "dimension value label",
            f"dimension {AGI} value under_1: 'No adjusted gross income' (first "
            "target irs_soi.eitc.under_1.all@2024), 'No adjusted gross income and "
            "deficit' (first target irs_soi.eitc.all.joint@2024)",
        ),
        (
            "dimension label",
            "dimension filing_status: 'Filing status' (first target "
            "irs_soi.eitc.under_1.all@2024), 'Filing status of return' (first "
            "target irs_soi.eitc.all.joint@2024)",
        ),
        (
            "provider label",
            "provider irs_soi: 'IRS Statistics of Income' (first target "
            "irs_soi.eitc.under_1.all@2024), 'IRS SOI' (first target "
            "irs_soi.eitc.maine@2024)",
        ),
        (
            "category label",
            "category irs_soi.eitc: 'Earned income tax credit' (first target "
            "irs_soi.eitc.under_1.all@2024), 'EITC' (first target "
            "irs_soi.eitc.maine@2024)",
        ),
        (
            "category provider",
            "category irs_soi.eitc provider: 'irs_soi' (first target "
            "irs_soi.eitc.under_1.all@2024), 'irs' (first target "
            "irs_soi.eitc.maine@2024)",
        ),
    ],
)
def test_two_labels_for_one_identifier_are_refused(kind: str, message: str) -> None:
    with pytest.raises(HierarchyLabelConflictError) as raised:
        catalog_calibration_diagnostics(_model(_conflict_payload(kind)))

    assert message in str(raised.value)
    assert len(raised.value.conflicts) == 1


def test_the_writer_refuses_a_conflict_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")

    outcome = write_calibration_diagnostics(
        _model(_conflict_payload("dimension value label")), path
    )

    assert outcome.status == "failed"
    assert outcome.error_code == "label_conflict"
    assert outcome.expected_schema_version == 9
    assert "No adjusted gross income and deficit" in outcome.message
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_a_label_choice_resolves_a_geography_conflict_only() -> None:
    payload = _us_payload()
    payload["targets"][1]["hierarchy"]["geography"] = {**UNITED_STATES, "label": "USA"}
    # A chosen label applies only where the rows disagree: Maine has one label
    # among the rows, so it keeps it although the choices name it otherwise.
    choices = HierarchyLabelChoices(
        geographies={"0100000US": "United States", "0400000US23": "State of Maine"}
    )

    catalogued = catalog_calibration_diagnostics(_model(payload), label_choices=choices)

    assert [
        (entry.id, entry.label) for entry in catalogued.hierarchy_catalogs.geographies
    ] == [("0100000US", "United States"), ("0400000US23", "Maine")]


def test_a_label_choice_resolves_a_dimension_value_conflict() -> None:
    choices = HierarchyLabelChoices(
        dimension_values={(AGI, "under_1"): "No adjusted gross income"}
    )

    catalogued = catalog_calibration_diagnostics(
        _model(_conflict_payload("dimension value label")), label_choices=choices
    )

    # The second row now names band ``under_1`` too, so ``all`` is unused.
    (band, _status_dimension) = catalogued.hierarchy_catalogs.dimensions
    assert [(value.id, value.label) for value in band.values] == [
        ("under_1", "No adjusted gross income"),
    ]


def test_a_dimension_value_choice_covers_only_its_reviewed_spellings() -> None:
    # Like Chronicle's REVIEWED_VALUE_LABEL_VARIANTS (chronicle#377): a choice
    # resolves the spellings it was reviewed for, and a new one is refused.
    reviewed = HierarchyLabelChoices(
        dimension_values={(AGI, "under_1"): "No adjusted gross income"},
        dimension_value_spellings={
            (AGI, "under_1"): frozenset(
                {"No adjusted gross income", "No adjusted gross income and deficit"}
            )
        },
    )
    catalogued = catalog_calibration_diagnostics(
        _model(_conflict_payload("dimension value label")), label_choices=reviewed
    )
    (band, _status_dimension) = catalogued.hierarchy_catalogs.dimensions
    assert [(value.id, value.label) for value in band.values] == [
        ("under_1", "No adjusted gross income")
    ]

    payload = _conflict_payload("dimension value label")
    payload["targets"][1]["hierarchy"]["dimensions"][0]["value_label"] = "No AGI"
    with pytest.raises(HierarchyLabelConflictError) as raised:
        catalog_calibration_diagnostics(_model(payload), label_choices=reviewed)
    assert (
        f"dimension {AGI} value under_1 (a spelling its reviewed list does not "
        "include): 'No adjusted gross income' (first target "
        "irs_soi.eitc.under_1.all@2024), 'No AGI'"
    ) in str(raised.value)


def test_every_disagreement_a_choice_resolves_is_logged_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A choice never hides a disagreement: the build log names the identifier,
    # each spelling with its first target, and the label written.
    choices = HierarchyLabelChoices(
        dimension_values={(AGI, "under_1"): "No adjusted gross income"}
    )

    with caplog.at_level("WARNING", logger="microcosm.diagnostics.catalogs"):
        catalog_calibration_diagnostics(
            _model(_conflict_payload("dimension value label")), label_choices=choices
        )

    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "Calibration diagnostics: the target rows label 1 identifier(s) more "
        "than one way; the chosen label is written for each: "
        f"dimension {AGI} value under_1: 'No adjusted gross income' (first "
        "target irs_soi.eitc.under_1.all@2024), 'No adjusted gross income and "
        "deficit' (first target irs_soi.eitc.all.joint@2024) -> 'No adjusted "
        "gross income'"
    )


def test_a_payload_whose_rows_agree_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="microcosm.diagnostics.catalogs"):
        catalog_calibration_diagnostics(_model(_us_payload()))

    assert caplog.records == []


def test_a_label_choice_does_not_resolve_a_level_conflict() -> None:
    choices = HierarchyLabelChoices(geographies={"0100000US": "United States"})

    with pytest.raises(HierarchyLabelConflictError, match="0100000US level"):
        catalog_calibration_diagnostics(
            _model(_conflict_payload("geography level")), label_choices=choices
        )


# The UK pattern found in the published UK trees: Chronicle's local facts
# carry the area as a dimension named ``geography`` whose value ID is the
# lower-case GSS code, labelled with each publisher's own name ("Hartlepool
# UA" from HMRC, "Hartlepool" from ONS), while the row's geography already has
# one label.
HARTLEPOOL = {"id": "E06000001", "label": "Hartlepool", "level": "local_authority"}


def _area(value_label: str, value_id: str = "e06000001") -> dict:
    return {
        "id": "geography",
        "label": "Geography",
        "value_id": value_id,
        "value_label": value_label,
    }


def _uk_area_payload() -> dict:
    return _payload(
        [
            _target(
                "hmrc.income.e06000001",
                geography=HARTLEPOOL,
                dimensions=[_area("Hartlepool UA")],
            ),
            _target(
                "ons.population.e06000001",
                geography=HARTLEPOOL,
                dimensions=[_area("Hartlepool")],
            ),
        ]
    )


def test_a_geography_dimension_value_takes_its_area_label_from_the_payload() -> None:
    model = _model(_uk_area_payload())
    with pytest.raises(HierarchyLabelConflictError, match="value e06000001"):
        catalog_calibration_diagnostics(model)

    catalogued = catalog_calibration_diagnostics(
        model,
        label_choices=HierarchyLabelChoices(
            geography_dimensions=frozenset({"geography"})
        ),
    )

    (area,) = catalogued.hierarchy_catalogs.dimensions
    assert [(value.id, value.label) for value in area.values] == [
        ("e06000001", "Hartlepool")
    ]


def test_a_geography_dimension_value_outside_the_payload_uses_the_chosen_label() -> (
    None
):
    payload = _uk_area_payload()
    # Neither row is at Middlesbrough, so the payload's geographies do not
    # name it; the chosen geography labels do, by upper-case GSS code.
    for row, label in zip(
        payload["targets"], ("Middlesbrough UA", "Middlesbrough"), strict=True
    ):
        row["hierarchy"]["dimensions"] = [_area(label, "e06000002")]
    choices = HierarchyLabelChoices(
        geographies={"E06000002": "Middlesbrough"},
        geography_dimensions=frozenset({"geography"}),
    )

    catalogued = catalog_calibration_diagnostics(_model(payload), label_choices=choices)

    (area,) = catalogued.hierarchy_catalogs.dimensions
    assert [(value.id, value.label) for value in area.values] == [
        ("e06000002", "Middlesbrough")
    ]


def _schema_9_document() -> dict:
    return json.loads(
        catalog_calibration_diagnostics(_model(_us_payload())).model_dump_json(
            exclude_defaults=True
        )
    )


def _corrupt(document: dict, change: str) -> None:
    catalogs = document["hierarchy_catalogs"]
    first_row = document["targets"][0]["hierarchy"]
    if change == "unknown category":
        first_row["category_id"] = "irs_soi.ctc"
    elif change == "unknown geography":
        first_row["geography_id"] = "0400000US06"
    elif change == "unknown dimension":
        first_row["dimensions"][0]["id"] = "income_range"
    elif change == "value under the wrong dimension":
        first_row["dimensions"][1]["value_id"] = "under_1"
    elif change == "duplicate geography":
        catalogs["geographies"].append({**catalogs["geographies"][0], "label": "USA"})
    elif change == "duplicate dimension value":
        catalogs["dimensions"][0]["values"].append({"id": "all", "label": "All"})
    elif change == "category with an unknown provider":
        catalogs["categories"][0]["provider_id"] = "irs"
    elif change == "dimension listed twice on a row":
        first_row["dimensions"][1] = dict(first_row["dimensions"][0])
    elif change == "hierarchy target of another row":
        first_row["target"]["id"] = "irs_soi.eitc.all.joint"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unknown category", "category irs_soi.ctc is not in hierarchy_catalogs"),
        ("unknown geography", "geography 0400000US06 is not in hierarchy_catalogs"),
        ("unknown dimension", "dimension income_range is not in hierarchy_catalogs"),
        (
            "value under the wrong dimension",
            "value under_1 is not listed under dimension filing_status",
        ),
        ("duplicate geography", "hierarchy_catalogs.geographies lists 0100000US twice"),
        (
            "duplicate dimension value",
            f"hierarchy_catalogs dimension {AGI} lists value all twice",
        ),
        (
            "category with an unknown provider",
            "category irs_soi.eitc refers to provider irs, which "
            "hierarchy_catalogs.providers does not list",
        ),
        ("dimension listed twice on a row", "hierarchy dimension ids must be unique"),
        ("hierarchy target of another row", "hierarchy target id must match"),
    ],
)
def test_schema_9_refuses_a_structurally_inconsistent_document(
    change: str, message: str
) -> None:
    document = _schema_9_document()
    _corrupt(document, change)

    with pytest.raises(ValidationError, match=message):
        parse_calibration_diagnostics(document)


def test_both_hierarchy_schemas_parse_to_their_own_models() -> None:
    assert isinstance(
        parse_calibration_diagnostics(_us_payload()), CalibrationDiagnosticsV8
    )
    assert isinstance(
        parse_calibration_diagnostics(_schema_9_document()), CalibrationDiagnosticsV9
    )
