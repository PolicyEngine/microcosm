# QBI v2 engine progress

## State

- Branch: `qbi-v2-engine`
- Base: `qbi-port-530` at `d1a6428`
- Status: version-gated QRF target selection and builder wiring in progress

## Done

- Created the task branch in the dedicated `populace-wt-530` worktree.
- Recorded the required deliverables, offline constraints, and full-suite finish gate.
- Traced the v1 donor simulation, QRF placement, post-QRF reconciliation,
  invariant gate, checkpoint target-order lock, and production builder seams.
- Confirmed the frozen `census_cps` person-column declaration includes `AGI`
  and detailed occupation `PEIOOCC`, but no detailed-industry field. V2 will
  therefore declare `occupation_column: "PEIOOCC"` and
  `industry_column: null`.
- Confirmed all country-package resources must be JSON/JSONLD, declared in
  `country_package.json`, and free of executable-looking strings.
- Added and declared `qbi_assumptions_v2.json` with strict derived/prior
  qualification contracts, occupation-first host SSTB configuration, complete
  AGI-band coverage, unchanged v1 W-2/UBIA parameter blocks, and five
  independently seeded RNG families.
- Added the empty `sstb_crosswalk_placeholder.json` resource and a strict
  crosswalk loader that rejects placeholder status.
- Added strict v2 full-schema parsing, unknown-key/mode rejection, public
  runtime exports, and focused loader/crosswalk tests. The untouched v1 golden
  stream test remains green.
- Implemented v2 donor execution: deterministic `source != 0` derivations do
  not consume qualification RNG, residual-prior sources retain seeded draws,
  W-2 and UBIA have separate family generators, and the donor emits a neutral
  preliminary SSTB route for authoritative post-QRF host classification.
- Made public v2 simulation fail closed before drawing when the packaged
  crosswalk remains placeholder-status.
- Added byte-level stream-independence tests: changing the final derived
  qualification source to an equivalent prior leaves W-2, UBIA, investment,
  REIT/PTP, and BDC family outputs byte-identical; changing W-2 or UBIA seeds
  leaves the other family byte-identical.
- Added the pure post-QRF `with_host_sstb_classification` transform. It derives
  law-determined flags from host record structure, preserves residual-prior
  flags, applies industry-primary/occupation-secondary crosswalk lookup,
  assigns ambiguous and passive-only records from the SSTB family stream, and
  reuses the exact v1 reconciliation router for Schedule C, W-2/UBIA pools,
  mutually exclusive routes, and exposure caps.
- Added a ready synthetic crosswalk fixture covering clear SSTB, non-SSTB,
  ambiguous, passive-only AGI bands, source-eligibility fail-closed behavior,
  and Schedule C precedence. The transform is deterministic, does not mutate
  its source frame, satisfies every summary invariant, and passes the QBI
  signal gate.

## Next

- Add version-gated QRF target selection.
- Thread the explicit version through monolithic and checkpointed builder
  paths while retaining v1 as the default.
- Add declarative source-stage target metadata and version-seam tests.
