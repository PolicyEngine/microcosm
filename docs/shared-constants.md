# Shared constants

This guidance applies to human contributors and AI assistants editing
Microcosm.

Before adding a module-local literal mapping, enumeration, identifier, or
display label, search the repository for an existing definition. Use `rg` with
the proposed constant name, representative keys, and representative values.

When the same static data is needed by more than one module:

1. Define it once in a domain-specific constants module located in the lowest
   existing package dependency shared by all consumers.
2. Import that definition directly everywhere it is used. Do not preserve a
   second copy under a local alias.
3. If consumers require different representations, derive those views beside
   the canonical definition. Do not rebuild inverse or integer-keyed mappings
   independently in each consumer.
4. Prefer immutable mappings so runtime code cannot modify process-wide
   reference data.
5. Add a unit test confirming that derived views remain consistent with the
   canonical definition.

Static code-system data belongs in a constants module. Values supplied by a
country specification, Chronicle fact, build artifact, or other runtime input
must continue to come from that input; do not copy runtime data into Python
constants.

## Geography constants

[`microcosm.calibrate.geography_constants`](../packages/microcosm-calibrate/src/microcosm/calibrate/geography_constants.py)
is the shared source for geography code mappings used by both the calibration
and build packages. It currently defines:

- `UK_GEOGRAPHY_ID_TO_LABEL`
- `UK_REGION_TIER`, `UK_REGION_TIER_ENUM` and `UK_LADDER_NATION_REGION_CODES` (the
  twelve-area region tier UK national references fan out over, its spine
  `region` enum names, and the ladder's nation pseudo-codes; microcosm#905)
- `US_STATE_FIPS_TO_POSTAL`
- `US_STATE_NUMERIC_FIPS_TO_POSTAL`
- `US_STATE_POSTAL_TO_NUMERIC_FIPS`

Import the representation required by the caller:

```python
from microcosm.calibrate.geography_constants import (
    US_STATE_FIPS_TO_POSTAL,
)
```

Do not add another state FIPS/postal mapping to a country runtime or diagnostics
module. Extend the shared definition and its consistency test when supported
geographies change.

## US target comparison views

[`microcosm.build.us_runtime.target_geography_view`](../packages/microcosm-build/src/microcosm/build/us_runtime/target_geography_view.py)
owns the national/state/congressional-district vocabulary that US
candidate-versus-incumbent target comparisons roll up on
(`US_TARGET_GEOGRAPHY_VIEW_LEVELS`, `US_TARGET_GEOGRAPHY_VIEW_ORDER`,
`UNRESOLVED_GEOGRAPHY_VIEW_LEVEL`, `GEOGRAPHY_VIEW_LEVEL_ALIASES`) and the
`us_target_geography_view` / `us_target_spec_geography_view` resolvers that
read it out of a compiled target's declared evidence.

The same three scopes are advertised by the frozen native measurement kernel
`graph_fiscal_measurement` as its household `FiscalGeographyScope` levels.
That kernel is not edited to import the shared tuple, because doing so would
move a live graph node identity;
`test_us_release_head_to_head_scorer.py::test_view_levels_match_the_native_measurement_kernel_scopes`
asserts the two vocabularies equal instead, and is the consistency test
required by item 5 above. Congressional-district membership itself is not
restated here: it comes from the shared
`microcosm.data.us_critical_targets.is_congressional_district_target`.

The same module also owns which metadata key identifies an *area* at each
level (`_LEVEL_OWNED_ID_METADATA_KEYS`: `state` → `state_fips`,
`congressional_district` → `congressional_district_geoid`) and which two
sources carry the ledger fact's geography id verbatim
(`_CANONICAL_GEOGRAPHY_ID_SOURCES`). Both matter because the US target
compiler derives a district's parent `state_fips` from its geoid
(`fiscal_targets.py:2825-2830`), so an identifier read without regard to level
would name the wrong area. Read an identifier only through
`us_target_geography_view`; do not re-derive a level-to-key mapping in a
consumer.

The source groups do not establish an identifier's encoding: hierarchy and
ledger fields accept bare or malformed text too. The view validates canonical
values at their resolved level with the existing fiscal target compiler's
state/CD parsers and `national_age_activation.NATIONAL_GEOGRAPHY_ID`. Bare
values are recognized separately, with the shared state FIPS mapping; other
nonempty values remain unrecognized. Neither validation rewrites values nor
assigns a missing congressional-district vintage.

The repo has no shared
constant for the state prefix (`"0400000US"` is an unnamed literal in
`fiscal_targets.py`, `congressional_district_vintage.py`,
`congressional_district_geography.py` and `medicaid_take_up.py`). Until one
exists, `target_geography_view` declares no equivalence between them:
`TargetGeographyView.geography_id_is_canonical` is how a consumer counting
distinct areas keeps the encodings apart. `geography_id_is_bare` distinguishes
recognized bare values from unrecognized ones. Scorecard schema 6 reports both
remainders and counts resolved rows without IDs separately from unresolved
scopes; no consumer should convert between encodings on its own.

If every explicit scope declaration is unsupported, the row stays unresolved
even when its name or other metadata provides CD evidence. The fallback is
reserved for rows without any explicit scope declaration.

Do not add another geographic-level enumeration to a US comparison, scorer, or
diagnostics module. Extend this definition and both its consistency tests when
a new view is supported.
