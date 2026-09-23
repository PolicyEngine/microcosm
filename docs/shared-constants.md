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

The engine's `local_authority` member names are not a constants table: the 361
April 2023 authority names are published ONS data, held in the sha-pinned
resource `microcosm/build/uk/local_authority_names.json` (regenerate with
`tools/generate_uk_local_authority_names.py`). Only the mechanical key rule and
its declared aliases are code, in
[`microcosm.build.uk_runtime.local_authority_input`](../packages/microcosm-build/src/microcosm/build/uk_runtime/local_authority_input.py);
derive an engine key through `local_authority_engine_key` rather than
re-implementing the rule, and resolve codes through
`resolve_local_authority_engine_keys`, which fails closed (microcosm#953).
