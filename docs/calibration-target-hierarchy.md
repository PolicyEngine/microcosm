# Calibration target hierarchy

Microcosm diagnostics schema 8 publishes a fixed sequence of hierarchy tier
types for every registry-backed calibration target:

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
8. Diagnostics schema 8 serializes `targets[].hierarchy` without constructing
   provider, category, geography, dimension, or target labels.
9. Release validation requires every schema-8 tier and label.
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
