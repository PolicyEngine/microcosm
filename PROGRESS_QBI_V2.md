# QBI v2 engine progress

## State

- Branch: `qbi-v2-engine`
- Base: `qbi-port-530` at `d1a6428`
- Status: v2 assumptions schema and resource implementation in progress

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

## Next

- Add strictly validated v2 assumptions and placeholder crosswalk resources.
- Implement v2 qualification derivations and independently seeded families
  without changing the v1 golden path.
- Add version-gated QRF target selection and the post-QRF host SSTB transform.
