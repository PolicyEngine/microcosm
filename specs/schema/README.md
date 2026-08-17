# Bundle schemas (RFC v3)

Closed-world JSON Schemas (draft 2020-12, `additionalProperties: false`)
for every bundle file kind, per docs/spec-engine.md v3. Pulled forward
from F0 so sign-off is on real schemas, not prose.

- `defs.schema.json` — shared: refs (kernel/source/stream/vintage), the
  row-scope predicate algebra, the executable gate record, surfaces.
- `resource_manifest.schema.json` — the typed rows country_package.json
  adopts (`{path, kind, schema_id}`); the SINGLE file inventory.
- One schema per authored kind: bundle, sources, spine, geography,
  imputation (blocks/models/chaining/concepts/families/producer_graph),
  take_up (ownership × typed steps), battery (gate records), calibration
  (full math contract), selection (exact-k + post-selection weights),
  publication (attempt events, promotion, audit≠release graphs),
  vintages (typed index refs; engine pin lives once in its generated lock), catalogs (contract + docs;
  lineage NEVER authored).
- `locks.schema.json` — EMITTED artifacts: bundle.lock, plan.lock
  (node keys), engine_abi.lock. Reproducible-or-rejected.

These bind through the F0 `CountrySpec` seam. The complete US resources
ship under `microcosm/build/us/spec/`; `specs/us/` is only the pointer left
behind when the drafting location was retired.
