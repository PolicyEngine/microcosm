# QBI v3 evidence progress

## State

- Active branch: `qbi-v3-evidence`.
- Dedicated worktree:
  `.claude/worktrees/populace-wt-530-v3`.
- Base: local `repeal-validation-298` at `e45f797`.
- Offline, local commits only. This lane produces provisional evidence
  resources and reproducible estimation code; simulation wiring remains out
  of scope.

## Done

- Created the requested isolated worktree and branch.
- Read the GitNexus exploration skill; its MCP tools are unavailable in this
  session, so codebase exploration will use direct repository searches.
- Read/inventoried the SCF, SOI, and Section 199A factsheets plus the restricted
  SCF archive and six SOI workbooks in place.
- Mapped the package-resource, spec-only, restricted-builder, and schema-test
  conventions.
- Declared the offline-cached `xlrd` reader needed for the two genuine BIFF
  SOI workbooks and regenerated its lock entry without network access.
- Implemented the pure SCF business-record stack, X42001/5 implicate pooling,
  owned-net bands, legal-form groups, employer proxy, independent presence/size
  collapse, weighted margin quantiles, JCT comparison, and strict resource
  validator.
- Added synthetic tests for implicate arithmetic, `-1` economic-zero handling,
  ownership scaling, thin-cell fallback, inverse-CDF margins, and schema
  refusal. Four focused tests and Ruff pass.
- Implemented legacy-XLS and XLSX SOI parsers with merged-header resolution,
  number-format disclosure/caution flags, finest-leaf classification,
  form-specific wage/capital arithmetic, source-cell provenance, conservative
  SCF-bin hints, and a strict resource validator.
- Added synthetic partnership/S-corporation workbook fixtures and form-arithmetic,
  suppression, hint, and schema-refusal tests. All 11 focused tests and Ruff
  pass. A real-file dry run reproduces the audited industry counts and
  wage/capital ranges.
- Added the configurable deterministic builder CLI, strict JSON writer, SCF
  DTA/ZIP reader, SHA-256 provenance, and all-corporation review-only seam.
  A synthetic DTA-versus-ZIP test brings the focused total to 12 passing tests.
- Ran the builder on the restricted SCF archive and all six SOI workbooks.
  Both emitted resources pass their strict validators and contain reproducible
  commands plus SHA-256 digests without embedding absolute source paths.
- The SCF estimate implies a 35.4919% headcount-at-most-one share among
  positive-QBI business interests (34.5661% when ambiguous legal-form code 40
  is excluded), versus the JCT firm-level zero-W-2-employee anchor of 84.2%.
- Finest-industry wage-share ranges are 0.4568%–34.1482% for sole
  proprietorships, 2.3615%–42.2970% for partnerships, and
  4.1531%–78.4294% for S corporations.
- Declared both JSON files in the spec-only US country package and added
  packaged-resource validation, provisional/declaration contracts, and the
  changelog fragment.
- The focused QBI, packaged-resource, country-spec, spec-only, and retired-name
  guards pass (45 tests); focused Ruff formatting/checking is clean.
- Two independent read-only audits reproduced both JSON files byte-for-byte
  and found no numerical, digest, disclosure, proxy, or determinism defect.
- Resolved their metadata findings by exposing the exact SCF active-management
  selection and comparison denominators; labeling partnership rows as sector
  totals; excluding sole-proprietor unclassified establishments from
  finest-industry summaries; and adding source cells for partnership/S-corp
  depreciation diagnostics. The rebuilt estimates and reported ranges are
  unchanged; the sole-proprietor finest universe is now 126 classified rows.
- The pre-review full workspace baseline completed without failure. The rebuilt
  focused QBI/spec-only selection passes 19 tests and focused Ruff is clean.

## Next

- Tighten nested resource-schema validation and add unequal-weight/fallback
  regression coverage.
- Run Ruff and the final full workspace suite, then write the final report.
