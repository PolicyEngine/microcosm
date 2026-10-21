# Calibration target hierarchy

Microcosm calibration diagnostics (schema 8, and schema 9 which current
builds publish) carry a fixed sequence of hierarchy tier types for every
registry-backed calibration target:

1. provider
2. category
3. geography
4. zero or more dimensions
5. target

The number of dimension entries is variable. The hierarchy is represented as
one ordered array of dimension records rather than recursively nesting an
arbitrary number of dimension objects in the authoring contract.

## Authoring contract

A country target contract uses `schema_version: 2`. It declares provider and
category labels once in normalized catalogs and gives every flat target a
`category_id`:

```json
{
  "schema_version": 2,
  "hierarchy": {
    "providers": {
      "obr": {"label": "Office for Budget Responsibility"}
    },
    "categories": {
      "obr.efo_receipts": {
        "provider_id": "obr",
        "label": "Economic and fiscal outlook receipts"
      }
    }
  },
  "targets": [
    {
      "target_id": "obr.income_tax",
      "label": "Income tax",
      "category_id": "obr.efo_receipts"
    }
  ]
}
```

Provider is therefore level 1 and the referenced category is level 2. The
category owns the provider relationship, so a target cannot independently
declare an inconsistent provider. Contract validation rejects missing provider
or category labels, unknown category references, and categories that reference
unknown providers.

The optional target `label` is Microcosm-owned. It is required whenever the
compiled target combines facts, transforms a fact value, or uses a different
period from its source fact. The UK contract declares it for every target so a
future operation or period change cannot introduce another metadata location.

Generated UK target-reference resources keep the same normalized catalogs and
add `hierarchy.target_categories` and `hierarchy.target_labels` lookups keyed by
`contract_target_id`.
Individual scalar rows already carry `metadata.contract_target_id`, so the
generated local rows do not repeat labels. Smaller country resources may carry a
direct `category_id` on a reference. Loading either form resolves the immutable
provider/category seed on `LedgerTargetReference`.

The UK release assembler also derives `release_manifest.publisher_labels` from
this provider catalog. Do not add a second UK provider or category label map in
the calibration package or release tooling.

## Chronicle inheritance

Geography, dimensions, and direct-target labels come from the Chronicle
consumer fact selected by a reference. These fields are required source
metadata, not hints for constructing display text.

- Geography uses Chronicle's `geography.level`, `geography.id`, and
  `geography.name`. The level and identifier must be explicit and constant.
  When `name` is absent, Microcosm may resolve only a known identifier through
  its shared authoritative geography catalog. An unknown identifier or missing
  identity fails compilation; neither condition is interpreted as UK.
- Every selected Chronicle fact must have a non-empty `label`. A same-period,
  identity-operation, single-fact target uses that label. A multi-fact,
  transformed, or period-adjusted target uses its explicit Microcosm target
  label because an individual member-fact label does not describe the compiled
  target.
- `layout.groupby_dimension` is first when it has a value. Its categorical
  value label comes from `layout.groupby_value_label`.
- Remaining Chronicle `dimensions` ids are sorted. Chronicle must supply
  exactly one non-empty `dimension_labels` entry and one non-empty
  `dimension_value_labels` entry for each selected dimension and value.
  Missing or conflicting labels fail compilation.
- A layout dimension and a fact dimension merge only when their identifiers
  are exactly equal. Similar labels do not establish identity.
- A multi-fact aggregate inherits only dimensions present with the same value
  on every member fact. Varying dimension ids are serialized in
  `metadata.ledger_aggregation_varying_dimensions` as provenance.

Dimensions are therefore automatic for new Chronicle facts, while their labels
remain Chronicle-authored. Authors define a provider/category once, select a
category for a new target, and define the target label once in the target
declaration. They do not maintain a separate visualization path.

Every declared calibration target must provide a non-empty `ledger_selector`.
Reference authoring rejects both missing selectors and target declarations that
contain `materialization`. If Chronicle does not yet publish the required fact,
add the source fact to Chronicle before declaring the Microcosm target. Do not
construct an observed target value from a Microcosm runtime input.

## One label per geography

One geography id has exactly one label across a target registry, so a
calibration diagnostics file names each geography once (microcosm#1335).
`TargetRegistry` checks this whenever a build assembles targets into a
registry, which is before calibration. A US registry refuses a second label
with `GeographyLabelConflictError`, naming the id, the labels and the first
target carrying each. Every other country's registry reports the conflict as a
`GeographyLabelConflictWarning`: UK publishers name some regions, nations and
local authorities differently ("Yorkshire and The Humber" and "Yorkshire and
the Humber", "St Helens" and "St. Helens"), and one label per UK code has not
been chosen yet.

A target Microcosm declares itself takes the label its other sources give the
same id. The US ACS local release's Census population targets label a state
with `US_STATE_FIPS_TO_NAME` ("Maine", as Chronicle's state facts name it) and
a congressional district with the Census Bureau's district name
("Congressional District 1 (119th Congress), Alabama"), from the same function
that labels the IRS SOI district rows translated to the 119th Congress plan.
That release's materialize stage assembles its admin and population targets
into one registry before the engine pass, so a conflict stops it at the start.

## Diagnostics schema 9: labels declared once

Schema 8 copies every provider, category, geography and dimension label onto
every target row. The registry check above covers geography ids only, and
Chronicle reports a value two packages word differently only as a warning, so
a schema 8 file could still label one identifier two ways. Published examples:
the US national release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z`
labels the AGI band `under_1` "No adjusted gross income" (106 targets) and "No
adjusted gross income and deficit" (2), and the October 2026 UK dense release
gives 88 areas two names, because Chronicle's local facts carry the area as a
dimension named `geography` whose value is the lower-case GSS code and each
publisher names it its own way (HMRC "Hartlepool UA", ONS "Hartlepool").

Schema 9 (`CalibrationDiagnosticsV9`) declares each of those labels once, in a
top-level `hierarchy_catalogs` object, and each row's `hierarchy` refers to
them by id:

```json
{
  "schema_version": 9,
  "hierarchy_catalogs": {
    "providers": [{"id": "obr", "label": "Office for Budget Responsibility"}],
    "categories": [
      {"id": "obr.efo_receipts", "label": "Economic and fiscal outlook receipts",
       "provider_id": "obr"}
    ],
    "geographies": [
      {"id": "K02000001", "label": "United Kingdom", "level": "country"}
    ],
    "dimensions": [
      {"id": "obr.efo_line", "label": "Economic and fiscal outlook line",
       "values": [{"id": "income_tax", "label": "Income tax"}]}
    ]
  },
  "targets": [
    {
      "name": "obr.income_tax@2025",
      "target_name": "obr.income_tax",
      "hierarchy": {
        "category_id": "obr.efo_receipts",
        "geography_id": "K02000001",
        "dimensions": [{"id": "obr.efo_line", "value_id": "income_tax"}],
        "target": {"id": "obr.income_tax", "label": "Income tax receipts"}
      }
    }
  ]
}
```

The provider is the category's; providers and categories move into catalogs
with geographies and dimensions, mirroring the authoring contract's catalogs,
so a category cannot appear under two providers either. Each value is listed
under its own dimension. A geography id has one label and one level. The
target's own label stays on the row, and every other field is schema 8's.

Producers still assemble schema 8 (`CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION`);
the writer publishes schema 9 (`catalog_calibration_diagnostics`). It refuses,
with the failure code `label_conflict` and no file, rows that give one id more
than one label (or one geography more than one level) unless a reviewed label
choice resolves it (`microcosm.calibrate.hierarchy_labels`):

- UK: a `geography` dimension value takes its area's label: the payload's own
  geography label for that code, else `UK_GEOGRAPHY_ID_TO_LABEL`, else the ONS
  April 2023 local-authority names in `local_authority_names.json` (the UK
  build's writer adds those). This resolves all 88 dense-release areas, two of
  them constituencies that only the payload's geography names, and the
  national releases' "Yorkshire And The Humber".
- US: `under_1` is "No adjusted gross income", `all` is "All returns, total",
  and `age` `85_plus` is "Age 85 plus", the labels chosen in review on
  2026-10-21. A US feed re-pin brings the same labels from Chronicle
  (chronicle#377 for `under_1`), so these choices matter only for feeds and
  releases built before it. Each choice covers only the two spellings
  Chronicle registers as reviewed variants; a third spelling is refused.

Every disagreement a choice settles is logged as a warning naming the id, each
spelling with its first target, and the label written, so a choice never hides
a variant.

A choice applies only where the rows disagree, so an id every row labels one
way keeps its label. Any other disagreement stops the writer until the source
is corrected or a reviewed choice is added. The same catalogs are written by
the transport diagnostics kernel and the UK graph's final validation kernel,
which raise instead of writing.

Releases published before schema 9 keep their schema-8 files;
`parse_calibration_diagnostics` reads both, and
`expand_calibration_diagnostics` rebuilds schema 8 from schema 9.

## Propagation stages

The expected data flow is:

1. The country contract validates normalized provider/category ownership.
2. Reference generation selects one or more Chronicle facts and rejects
   missing or conflicting source-owned labels.
3. A scalar reference retains its category relationship and the declared
   Microcosm target label.
4. `LedgerTargetReference` resolves the catalog relationship to a typed seed.
5. Chronicle geography and dimensions complete `CalibrationHierarchy` while
   compiling a Chronicle-backed `TargetSpec`.
6. Registry format 3 serializes and reloads that hierarchy. Format 2 remains
   readable for historical artifacts.
7. Solver compilation carries the hierarchy unchanged from `TargetSpec` to
   `Target`.
8. Diagnostics assembly (schema 8) serializes `targets[].hierarchy` without
   constructing provider, category, geography, dimension, or target labels;
   the writer publishes schema 9, declaring each label once in
   `hierarchy_catalogs`.
9. Release validation requires every tier and label, in schema 8 or 9.
10. The calibration dashboard selects its parser from top-level
    `schema_version` and builds navigation from the ordered hierarchy.

## Three representative UK targets

### OBR income tax

`obr.income_tax` references `obr.efo_receipts`. Chronicle contributes the UK
geography, the `obr.efo_line` dimension, the `income_tax` value and label, and
the fact label. If the source fact already represents the 2025 target without a
value transformation, its fact label names the target. Otherwise the declared
Microcosm label `Income tax` names the compiled target.

### HMRC employment income band

The HMRC SPI declaration references `hmrc.survey_of_personal_incomes`.
Chronicle's `hmrc.total_income_band` layout dimension is ordered first and uses
its authored band label. A distinct raw dimension such as
`total_income_lower_bound` remains a separate dimension because the ids are not
equal.

### DWP Universal Credit payment distribution

All fan-out scalar rows from the Universal Credit payment-distribution
declarations reference `dwp.universal_credit`; no row-specific classification
table is required. Each selected Chronicle cell supplies the relevant household
type and payment-band dimensions. The target declaration supplies one label for
the transformed target, while future cells inherit their dimensions and labels
automatically.

## AI assistant checks

When changing calibration targets, verify the declaration, generated scalar
reference, compiled `TargetSpec`, registry round trip, diagnostics row, release
validation, dashboard parser, and resulting tree in that order. Do not add a
dashboard label fallback for schema 8. Do not infer category membership from a
target name, Chronicle measure id, or display label. If a new target fails
generation because `category_id` is absent, add or reuse a normalized category
in the country declaration. If compilation reports missing Chronicle labels,
correct the Chronicle source adapter and refresh the pinned consumer artifact;
do not add identifier-humanization logic in Microcosm.
