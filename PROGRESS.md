# PROGRESS — eCPS parity_gate wiring (issue #313)

Branch: `ecps-parity-gate` (worktree from origin/main @ 7e9a32d)
PR: (not yet opened)

## Goal
Wire `parity_gate` into the US release with a pinned incumbent-eCPS reference
(committed JSON of per-variable nonzero shares + source sha256/vintage) and a
reason'd, issue-linked exemption register. Failure = nonzero exit; GateResult
details land in the release manifest.

## Architecture findings (before writing code)
- `parity_gate(candidate_nonzero, reference_nonzero, *, known_gaps=())` in
  `packages/populace-build/src/populace/build/gates.py:1347`. Two share dicts +
  `known_gaps` = iterable of NAMES only (no reason mapping). Details:
  `reference_populated_layers`, `gaps`, `exempted`.
- `GateReport.to_manifest()` (gates.py:126) already serializes gate details.
- Declarative `gates.json` / `GatesManifest` (`country_spec.py`) is a
  **Belgium-only** greenfield construct. BE `gates.json` policy note explicitly
  says parity/export_surface/target_surface are "deliberately not selected"
  (no incumbent). `ALLOWED_GATE_FUNCTIONS` (country_spec.py:72) DOES list
  "parity" — so parity is a known gate id, just never selected by any spec.
- **US build (`tools/build_us_fiscal_refresh_release.py`) does NOT load
  country_spec/GatesManifest at all.** US gates run as DIRECT function calls
  (e.g. `input_mass_parity_gate` at lines 3237/3261). => The seam for wiring
  parity into US is the **direct call** in the release tool, alongside the
  geography-ladder and weights-audit gates. (Chosen seam; rationale recorded.)
- JSON reference idiom: `packages/populace-build/src/populace/build/us/*.json`,
  loaded via `from importlib.resources import files` →
  `files("populace.build.us").joinpath("X.json").read_text()` (see
  `us_runtime/fiscal_targets.py:715`).

## Reference artifact (incumbent eCPS) — PINNED
- HF: `policyengine/policyengine-us-data` (repo_type=model), file
  `enhanced_cps_2024.h5`, revision `21280dca5995e978d706740a8a4b9b7860cfd7b6`
  (refs/main at compute time), sha256
  `0a6b961ad363a421bde99f2c8e5d8f20370bcba45fd303050537a25bdd805b14`
  (verified: HF LFS blob name == content sha256). vintage/period 2024.
- Native layout is the LEGACY flat `variable/2024` HDF5 (244 var keys), NOT the
  `USSingleYearDataset` entity-table layout Populace publishes — so it cannot be
  loaded with `load_us_frame`; read each variable array directly.
- Reference scope: PolicyEngineUSEngine().variables() = 841 engine INPUT vars
  (formula-owned excluded), intersected with eCPS keys, minus 10 schema-
  structural entity-id/membership columns. => 179 present, **158 populated**.
- Candidate measured: `policyengine/populace-us` `populace_us_2024.h5` (sha
  c2065b64…), the small national support frame — populates 51 cols, 48 of them
  reference-populated (parity OK on those).
- Committed: `us/ecps_parity_reference.json` (158 shares + source sha/vintage),
  declared in `us/country_package.json` resources.

## FIRST-RUN GAP LIST — 110 honest gaps (SURPRISING: far exceeds the 3 families)
The issue anticipated take-up (#312) + SCF wealth (#49) + SPM (#32). Those
cover only 36 of 110. Mapped the remaining 74 to EXISTING open trackers (no new
issues filed — lead files). Register `us/ecps_parity_known_gaps.json`, 110
entries, all reason'd + issue-linked. Distribution:
  #38  (42) remaining US tax-input & reported-observation & demographic/geo
       layers — the explicit catch-all tracker for exactly this residue
  #298 (16) QBI / passthrough qualification (`*_would_be_qualified`, sstb_*,
       REIT/PTP/BDC, UBIA, W2-QBI) — QBI base already off vs targets
  #32  (15) SPM inputs (housing, child support, workers comp, MOOP/expenses)
  #312 (13) take-up flags (`takes_up_*`, WIC, voluntary-filing)
  #49  ( 8) wealth/assets/vehicles (net worth, SCF assets, auto-loan; see #252)
  #253 ( 7) education-credit inputs (AOC / tuition)
  #274 ( 4) capital-gains detail (collectibles / 1250 / 4952 / inv-interest)
  #242 ( 5) hours-worked / labor inputs
NOTE for lead: #38 is doing heavy lifting (42). Its scope ("remaining US input-
layer work … release diagnostics expose reviewed exclusions") legitimately
covers these, but the lead may want dedicated children for: retirement
distributions (401k/403b/sep/keogh/IRA), occupation/labor-status descriptors,
fine geography codes (block/county/tract), and demographic flags. household_weight
is a genuine type-mismatch (incumbent stored a weight column; Populace carries a
typed Frame weight) — flagged under #38 as reported-obs-vs-exclusion.

## Steps
- [ ] 1. Pin reference: compute incumbent per-variable nonzero shares ONCE,
      check in JSON with source sha256 + vintage.
- [ ] 2. Wire gate into US release path (direct call seam), fail-loud + manifest.
- [ ] 3. Seed exemption register (checked-in file, reason'd + issue-linked).
- [ ] 4. Tests: plant-a-layer (prove-it-can-find-something), register schema,
      reference-JSON integrity (sha match).

## Coordination
- Open PRs #308 (weights-audit) + #311 (validation-input gate) both touch
  gates.py exports + release tool gate assembly. Rebase on origin/main before PR.

## Log
- Set up worktree, read gates.py + country_spec.py, mapped the seam.
