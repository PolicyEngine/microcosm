HOLD — exact blockers: live head `76e39f9b` has no acceptance evidence for its 25-stage candidate and its final `age_tail` stage reproducibly breaks E6's stored NHS identities; separately, strict signed parity still accepts unadjudicated directions, magnitudes, and entity counts.

# Defensive correctness and completeness audit — microcosm PR #747

## Round 2 — live-head verdict

I audited live PR head `76e39f9b77256766d7608c1f03e12db2f8cef118`
against live base `5abda6145b06d55b255db87ecead2438d5d8fde7`: 28 PR
commits, 52 files, 10,327 insertions, and 3,922 deletions. The checked tree is
`ca00954e78a752a2a4ba492fad6e08120add572a`. I read the full diff and every
added receipt, register, fixture, resource, gate pin, and generated manifest.

GitHub reported during this audit that the PR had already merged at
2026-08-25 11:45:37 UTC as
`2807e99cc11d1a8c69a0886992ae21a533719958`. The HOLD therefore means: do not
rely on or promote the spine swap from that merge; land an immediate corrective
follow-up or revert until the two blocking conditions below are closed.

### Round-1 finding disposition

| Round-1 finding | Round-2 disposition on `76e39f9b` |
|---|---|
| 1. Final-head acceptance is stale; age-tail breaks E6 | **SURVIVES — BLOCKING.** No receipt or relevant implementation changed. |
| 2. Signed entries are column amnesties | **SURVIVES — BLOCKING.** Commit `76e39f9b` resolves the H5 entity-table/student-loan subpart only; magnitude, direction, counts, and candidate-only entity scope remain open. |
| 3. Strict mode does not pin the acceptance band | **SURVIVES — HIGH.** No resolving change. |
| 4. Weighted totals are not closed-world or identity-bound | **SURVIVES — HIGH.** No resolving change. |
| 5. Re-pinned evidence quotes a stale incumbent figure | **SURVIVES — MEDIUM.** No resolving change. |

The only PR-authored commit after the stale round-1 head was `76e39f9b`
(`fix(build): scope signed differences to their owning entity`). It is a real
but narrow repair. The two experiment documents and the age/NHS/E6 path are
byte-identical to round 1.

## Ranked findings

### 1. BLOCKING — SURVIVES: acceptance certifies 24 stages, while the live 25-stage spine fails E6

The committed L0 receipt still says the accepted candidate ran “all 24 stages”
(`experiments/686-uk-spine-swap-receipts.md:330-342`), yet the live driver lists
25, with `etb_services` at stage 16 and `age_tail` last
(`tools/build_uk_frs_spine.py:93-119`), and installs that final transform in the
production plan (`tools/build_uk_frs_spine.py:845-852`). The ledger nevertheless
declares L0, E4–E8, and signed L2 complete
(`experiments/686-uk-spine-comparison-ledger.md:60-67`).

This is a demonstrated correctness break, not a stale stage-count label.
`etb_services` allocates and persists six NHS columns from each person's
stage-time age (`packages/microcosm-build/src/microcosm/build/uk_runtime/etb_services.py:117-139`,
`:426-436`). `age_tail` defines the 80–97 tail and later rewrites top-coded age
in place (`packages/microcosm-build/src/microcosm/build/uk_runtime/age_tail.py:62-77`,
`:151-196`).
E6 then recomputes the NHS values from final age and requires equality with the
stored columns (`tools/verify_uk_identity_stability.py:378-386`, `:428-464`).

Evidence run on the live tree: a synthetic production-function probe started
with 400 age-80 people, persisted the production NHS allocation, ran the
committed age-tail transform, and invoked the production E6 receipt. It moved
188 people to age 85+ and 67 to age 90+. Permutation identity remained true,
but `matches_stored_columns` was false and all six NHS columns were reported as
mismatches. The probe preserved nonzero status for age and all six NHS columns,
so the L2 nonzero-share comparison is blind to this value corruption; a fresh
L2-only receipt would not close it.

The only named R5 JSON remains an uncommitted local path
(`experiments/686-uk-spine-swap-receipts.md:552-576`); `git ls-files` found no
tracked `data/ukds/acceptance/686-spine-swap/*`. There is no committed candidate
SHA, stage-roster sidecar, twin receipt, parity JSON, or E4–E8 JSON. The same
receipt also predates the persisted-zero SPI changes at
`packages/microcosm-build/src/microcosm/build/uk_runtime/spi_income.py:464-475`,
`:597-621`, and `:864-881`. No age-tail × NHS/E6 integration test exists.

### 2. BLOCKING — SURVIVES: strict signed parity does not enforce the facts the register signs

The register says direction is part of the exact observed divergence
(`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:3`),
including water at about +0.10 (`:6-21`) and exact person +32 / benunit -12
counts (`:203-221`). The model stores magnitude only as prose
(`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:109-123`).
`covers()` checks expectation, surface, column, and sometimes entity, but never
sign, bounds, magnitude, or an expected count (`:124-184`); the count/share
comparators pass names rather than the attested values
(`tools/verify_uk_spine_parity.py:124-188`).

Live strict-mode probes using individual committed entries all returned
`signed_parity`, `strict_failure=false`, and no unsigned differences when:

- both candidate person and benunit counts were omitted;
- both candidate counts were set to 1; and
- water moved from reference 0.776937 to candidate 0.0, an opposite-direction
  delta of -0.776937 from the registered positive movement.

Commit `76e39f9b` does resolve one round-1 subfinding: the H5 comparator now
passes the table entity (`tools/compare_uk_h5_payload.py:349-368`), the register
correctly scopes `student_loan_balance` to person
(`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:148-163`),
and adversarial table-scope tests exist
(`packages/microcosm-build/tests/test_uk_h5_payload_compare.py:564-655`).

The entity fence is still incomplete for candidate-only share columns. The
extractor emits candidate `input_entities`
(`tools/build_uk_efrs_parity_reference.py:433-437`), but verification ignores
that map and supplies only reference entities
(`tools/verify_uk_spine_parity.py:318-326`). For a candidate-only column this
becomes `entity=None` (`:190-225`), which deliberately skips filtering
(`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:177-183`).
A live probe declared candidate-only `num_bedrooms` as person-owned; its
household-scoped committed entry (`spine_swap_signed_differences.json:224-239`)
still signed it.

### 3. HIGH — newly surfaced: acceptance receipts do not bind the register actually used

The parity CLI accepts an arbitrary `--register` path even in `--strict` mode
(`tools/verify_uk_spine_parity.py:446-453`, `:526-539`), but its receipt always
claims `spine_swap_signed_differences.json` and records no path or digest
(`:408-415`). The structure-only H5 CLI has the same override and hard-coded
resource identity (`tools/compare_uk_h5_payload.py:386-396`, `:403-416`,
`:456-464`).

Evidence run: an end-to-end strict invocation used a temporary one-entry fake
register to sign the opposite water regression. It exited 0 as
`signed_parity`, emitted the fake ID `pretend-water-signature`, but claimed the
packaged register resource and included no register SHA. R5's Markdown shows a
default command, but its raw receipt is uncommitted, so no machine evidence
binds the claimed run to the 13 committed adjudications.

### 4. HIGH — SURVIVES: `--strict` permits an arbitrary acceptance band

`--share-band` accepts any value in `[0, 1)`
(`tools/verify_uk_spine_parity.py:468-504`). In-band differences never become
unsigned (`:150-188`), while strict mode only fails unused register entries
(`:390-425`, `:552-566`) despite being documented as the swap-acceptance
posture (`:479-485`).

Evidence run: with an empty register and water delta -0.776937,
`--strict --share-band 0.9` exited 0 with `verdict=parity`,
`strict_failure=false`, and `unsigned_differences=[]`. R5 records the default
0.02 band, so this does not allege that its prose command used 0.9; it proves
the acceptance mode itself is fail-open.

### 5. HIGH — SURVIVES: weighted totals are neither closed-world nor identity-bound

The comparator examines only intersecting total keys and merely reports
one-sided keys (`tools/verify_uk_spine_parity.py:230-274`). The caller copies
optional sidecar identities into output without validating them and extends
only the incomplete unsigned list (`:365-383`). The producer is capable of
emitting SHA-bound identities (`tools/build_uk_efrs_parity_reference.py:520-557`).

Evidence run: strict CLI comparison of reference totals
`{kept: 1, omitted: 2}` and candidate totals `{kept: 1}` accepted reference
identity `{filename: anonymous-left}` and a missing candidate identity. It
reported `only_in_reference=[omitted]` but exited 0 with `verdict=parity` and no
unsigned differences. R5 explicitly leaves this surface dormant
(`experiments/686-uk-spine-swap-receipts.md:656-663`), so this does not change
its incidence-only output; it blocks safe later use of the advertised
water-level surface.

### 6. MEDIUM — SURVIVES: the 1.56.16 evidence still quotes a 1.56.14 value

The register claims every incumbent figure was measured on pinned 1.56.16
(`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:3`),
and the ledger repeats that guarantee
(`experiments/686-uk-spine-comparison-ledger.md:43-58`). The pinned reference is
`main_residence_value = 0.674658`
(`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:313`),
but both the register (`spine_swap_signed_differences.json:142`) and ledger
(`experiments/686-uk-spine-comparison-ledger.md:184-190`) quote 0.6761. The live
diff shows that 0.6761 is the rounded old 1.56.14 value, 0.676078, rather than
rounding of the new pin. Direction and band classification do not change, so
this is not independently blocking.

### 7. MEDIUM — newly surfaced: a columnless `root_attr` entry blanket-signs all root attributes

The loader's new fence requires columns only for entity-scoped surfaces and
excludes `root_attr`
(`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:98-106`,
`:340-373`). `covers()` then treats empty columns as a wildcard (`:151-184`),
while the H5 comparator asks that register lookup for every differing root
attribute (`tools/compare_uk_h5_payload.py:370-400`). A live probe loaded a
columnless `root_attr` entry and thereby signed an arbitrary future attribute,
`completely_unadjudicated_future_attr`, with `structure_only_ok=true`.

The committed register currently has no `root_attr` entry, so this is a latent
fence defect, not a claim that R5 used such an amnesty. It also predates the
latest commit; round 2 surfaced that `76e39f9b` incompletely resolved the wider
scope problem rather than introducing this route.

## Verified clean / no finding

- No terminal gate threshold or criticality was weakened. `target_surface`
  remains release-blocking and parameter-free
  (`packages/microcosm-build/src/microcosm/build/uk/gates.json:293-307`), and its
  binding accepts no parameter keys
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/battery_bindings.py:887-892`).
- No target-membership runtime option, reviewed waiver, Logbook-rule change,
  staging change, publication/promotion path, or remote side effect was added.
  The governing rules remain `DESIGN.md:136-144` and `CLAUDE.md:37-49`.
- All 13 committed register entries have nonempty adjudicator, adjudication
  date, magnitude evidence, and evidence fields; their anchors resolve. No
  expiry or ownerless waiver was added.
- The six age-tail resource values match the live membership facts
  (`packages/microcosm-build/src/microcosm/build/uk/ons_age_tail_band_populations.json:7-14`;
  `packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:6611-6652`,
  `:6896-6937`). The source-stage manifest independently recomputes to the
  committed `84d1c1c85f0172a9a396ef69d445ee33cf1340f62ff73c8653c399af40087bf4`
  digest in all 11 coverage mirrors.
- Incumbent identity/totals pins and the UK specification digest are repinned
  consistently. The latest entity-scope commit moves no seed or attested
  identity surface.

## Verification log

- Verified the supplied/live commit object and worktree byte-for-byte against
  tree `ca00954e...`; the supplied full diff has SHA-256
  `d6aba8939a655d50416f68f00b5f80b3f1c30104f0288a54a1459c51329594e2`,
  matches the 52-file live diff stat, and passes `git diff --check`.
- The requested `uv sync --all-packages --extra us` resolved all 123 locked
  packages but could not download `ndindex==1.10.1` because PyPI DNS was
  unavailable in the managed sandbox. Tests used a compatible sibling locked
  environment with this worktree's five package source roots first on
  `PYTHONPATH`.
- Six focused test files covering parity, signed differences, H5 payload,
  age-tail, identity receipts, and NHS allocation: **115 tests, exit 0**.
- The four supplemental UK parity-reference, terminal-gate,
  weighted-integrity, and source-stage files completed: **104 passed, 1
  skipped**. The
  following unrelated US country-package test stalled in the fallback
  environment and was interrupted; it is not counted as evidence for this
  verdict.
- Production-function and full-CLI probes reproduced the E6 failure, signed
  magnitude/count/candidate-entity failures, loose-band bypass, unbound-register
  receipt, missing-total/identity bypass, and root-attribute wildcard described
  above.
- No candidate or repository build, publication, promotion, push, or commit to
  the PR branch occurred. Review-only commits are on
  `review/pr-747-r2-audit`.

## Minimum closure

1. Make final ages and stored age-derived NHS values obey E6, and add an
   age-tail × NHS/E6 integration regression.
2. Rebuild the exact final 25-stage candidate and commit a disclosure-safe
   receipt binding candidate SHA, live composition/stage roster, twin payload,
   E4–E8, and strict parity results.
3. Make strict parity enforce registered direction/magnitude/exact counts and
   both reference and candidate entity scopes.
4. Pin the strict acceptance band; refuse register overrides in acceptance
   mode or bind the exact register bytes by digest in every receipt.
5. Make weighted totals closed-world and identity-bound; close the root-attribute
   wildcard; correct the stale 1.56.16 magnitude claim.

## Round-1 historical reference

Round 1 audited cached head `86f55741081fa3fb5e3c55234e3c8dc7ff77c777`
and recorded HOLD in audit commit `eb50a753`. Its five substantive findings are
preserved above with explicit round-2 dispositions; citations and probes in
this section are all refreshed against live `76e39f9b`.
