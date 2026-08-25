HOLD — the committed swap-acceptance receipt certifies a 24-stage candidate and predates PR head's value-changing 25th stage and SPI persisted-value changes; no final-head acceptance evidence exists.

# Defensive correctness and completeness audit — microcosm PR #747

Audited cached PR head `86f55741081fa3fb5e3c55234e3c8dc7ff77c777` against merge base `7b90bb1882b0248d751a64bf817ec127e5c42a47`: 27 commits, 51 files, 9,965 insertions, and 3,922 deletions. I read the full diff and both experiment records, and checked every changed committed receipt, register, resource, fixture, gate pin, and generated manifest relevant to the swap claim.

GitHub and PyPI DNS were unavailable, so `git fetch origin pull/747/head:pr-747`, `gh pr view 747`, and the requested `uv sync --all-packages --extra us` could not complete. The audited cached remote head matches the María-authored #686/#747 sequence available in the shared repository. I did not run a candidate/repository build, publish, push, or commit to `pr-747`; review artifacts are on `review/pr-747-audit`.

## Ranked findings

### 1. BLOCKING — final-head acceptance evidence is stale, and the new last stage reproduces an E6 failure

The committed L0 receipt says the accepted candidate ran “all 24 stages” (`experiments/686-uk-spine-swap-receipts.md:330-342`), and the ledger consequently declares L0 through L2 complete (`experiments/686-uk-spine-comparison-ledger.md:60-67`). The current driver instead has 25 spine stages: `etb_services` is stage 16 and `age_tail` is the new final stage 25 (`tools/build_uk_frs_spine.py:93-119`), with the latter installed in the actual plan at `tools/build_uk_frs_spine.py:845-848`.

That order changes more than the roster hash. `etb_services` derives and persists all six NHS inputs from each person's then-current age (`packages/microcosm-build/src/microcosm/build/uk_runtime/etb_services.py:117-139`, `packages/microcosm-build/src/microcosm/build/uk_runtime/etb_services.py:426-436`). `age_tail` later rewrites every selected top-coded `person.age` in place (`packages/microcosm-build/src/microcosm/build/uk_runtime/age_tail.py:150-196`). E6 recomputes those six NHS inputs from the *final* ages and requires them to equal the stored columns (`tools/verify_uk_identity_stability.py:378-386`, `tools/verify_uk_identity_stability.py:431-464`). Whenever the tail moves a person across the NHS age-85 boundary, the stored and recomputed values can therefore disagree.

Evidence run:

- A synthetic frame used the production NHS allocator, copied its six outputs as the stage does, loaded the committed age-band resource, ran the production age-tail transform, then called the production `e6_identity_receipt`. Of 400 top-coded people, 188 moved to age 85+ (67 to 90+), and the result was `matches_stored_columns=False`; all six NHS columns were listed in `stored_column_mismatches`.
- The 82 focused parity/register/age/identity tests all passed, but repository search found no age-tail × NHS/E6 integration test. `test_uk_age_tail.py` tests the tail alone, and the new `test_uk_identity_stability_receipts.py` covers E7 only.
- `git log` shows the experiment records end at `a7336d31` (2026-08-24 11:48). Candidate-changing commit `19799f37` then added `age_tail` and changed absent FRS-channel SPI values from NaN to persisted zero (`packages/microcosm-build/src/microcosm/build/uk_runtime/spi_income.py:464-475`, `:608-621`, `:864-881`) at 13:13; final `86f55741` only re-pinned a roster-derived manifest and test. Neither experiment record was updated.
- R5 names an uncommitted local extraction and receipt (`experiments/686-uk-spine-swap-receipts.md:552-576`); `git ls-files data/ukds/acceptance/686-spine-swap/*` found no committed receipt. More importantly, the Markdown rendition itself contains no post-`19799f37` run.

This alone blocks the swap. Fix the age/NHS stage-time inconsistency (or provide and enforce a coherent alternative contract), add the missing integration regression, then rebuild the exact 25-stage head and commit a disclosure-safe receipt binding the candidate content identity, stage roster, E4–E8 results, twin-payload result, and strict parity result.

### 2. BLOCKING — signed entries are column amnesties, not the scoped facts the register claims

The register explicitly says direction is part of what is signed and that entries cover the exact observed divergence (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:3`). For example, the water entry attests an approximately `+0.10` candidate-minus-incumbent share (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:6-21`), while the entity-count entry attests exactly person `+32` and benunit `-12` (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:203-221`).

The implementation never enforces those facts. `UKSignedDifference.covers()` matches only surface, column, and expectation; it does not receive or check entity, sign, direction, magnitude, or an expected count (`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:114-142`). The count and share comparators make that name-only lookup at `tools/verify_uk_spine_parity.py:124-182`. The payload bridge extends the same entry to arbitrary row-value mismatches on that column (`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:126-141`; `tools/compare_uk_h5_payload.py:349-361`).

Evidence run against the **committed** register:

- Candidate person/benunit counts omitted entirely: both candidate values became `None`, both matched `donor-selection-rng-entity-counts`, and `unsigned=[]`.
- Candidate person/benunit counts set to `1` instead of 113,649/61,211: the same signed ID matched and `unsigned=[]`.
- Water changed from reference `0.776937` to candidate `0.0`, an opposite-direction `-0.776937` regression: it matched `scottish-water-incumbent-nan-zeroing` and `unsigned=[]` even though the entry signs about `+0.10`.
- The register declares `student_loan_balance` as household-owned (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:148-160`), while the authoritative parity fixture says person (`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:212`). The entry still matches because `entities` is parsed but ignored; the existing surface test checks column existence only (`packages/microcosm-build/tests/test_uk_signed_differences.py:117-149`).

Thus a `signed_parity --strict` verdict does not establish the adjudicated facts reported in R5. Replace prose-only magnitude claims with machine-checked direction/bounds or exact expected counts, pass entity into matching, validate entity scopes against the reference, and add adversarial strict-mode tests before relying on this register for swap acceptance.

### 3. HIGH — `--strict` does not pin the acceptance band

An operator can supply any `--share-band` in `[0, 1)` (`tools/verify_uk_spine_parity.py:453-490`). Differences inside that band never enter `unsigned` (`tools/verify_uk_spine_parity.py:147-182`), while `--strict` only changes unused-register handling (`tools/verify_uk_spine_parity.py:376-412`) despite being documented as the swap-acceptance posture (`tools/verify_uk_spine_parity.py:464-470`).

Evidence run: the default comparator marked an unregistered 0.70 share delta unsigned. A full CLI probe then changed the committed water reference share from `0.776937` to `0.0` and ran an empty register with `--strict --share-band 0.9`; it exited **0** with `verdict='parity'`, `strict_failure=False`, and `unsigned=[]` while reporting the `-0.776937` delta inside the band.

R5 records the default 0.02 band, so this is not an allegation that the recorded command used 0.9. It is a fail-open in the newly introduced acceptance instrument. Strict mode should refuse a non-contract band (or require a separately named, non-acceptance diagnostic mode).

### 4. HIGH — weighted-total evidence is neither closed-world nor identity-bound

The weighted comparator examines only intersecting keys. One-sided keys are placed in `only_in_reference` / `only_in_candidate` but never added to `unsigned` (`tools/verify_uk_spine_parity.py:221-260`). The caller extends only that incomplete `totals_unsigned` list and merely echoes both sidecars' optional `identity` values without validating either against the pinned reference or candidate extraction (`tools/verify_uk_spine_parity.py:352-374`). The authored passing test supplies filename-only identities (`packages/microcosm-build/tests/test_uk_spine_parity_instrument.py:433-472`).

Evidence run: a full `--strict` CLI invocation compared reference totals `{'kept': 1, 'omitted': 2}` with candidate totals `{'kept': 1}`. It accepted identities `{'filename': 'anonymous-left'}` and `{}`, reported `only_in_reference=['omitted']`, and exited **0** with `verdict='parity'`, `strict_failure=False`, and `unsigned=[]`.

R5 did not supply weighted sidecars and correctly calls that surface dormant (`experiments/686-uk-spine-swap-receipts.md:656-663`), so this does not alter its incidence-only output. It does mean the dormant water-level adjudication and any later calibrated level proof cannot be closed safely with this tool. Require the expected total-key set and bind both sidecars to content identities before using that path.

### 5. MEDIUM — the re-pinned incumbent evidence still contains a stale 1.56.14 figure

The register says every incumbent figure was measured on pinned 1.56.16 (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:3`), and the ledger repeats that guarantee (`experiments/686-uk-spine-comparison-ledger.md:45-58`). But the committed 1.56.16 reference has `main_residence_value = 0.674658` (`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:313`), while both the signed magnitude evidence and ledger still quote `0.6761` (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:142`; `experiments/686-uk-spine-comparison-ledger.md:184-190`).

The corrected value does not flip direction or move the column inside the 0.02 band, so this is not independently blocking. It does disprove the claimed cross-check completeness and should be corrected when the final-head evidence is regenerated.

## Verified clean / no finding

- No terminal gate threshold or criticality was weakened. The one-target-surface gate remains release-blocking and parameter-free (`packages/microcosm-build/src/microcosm/build/uk/gates.json:293-307`); the `gates.json` diff only re-pins incumbent identity and totals digest.
- No reviewed-exclusion register, target-membership surface, Logbook rule, publication path, promotion code, or production pointer changed. The governing doctrine still requires a durable Logbook row for every outcome (`DESIGN.md:136-144`) and keeps publication/promotion a separate human action (`CLAUDE.md:37-49`); the PR adds no membership option or publication side effect.
- All 13 signed entries contain adjudicator, date, magnitude-evidence, and evidence fields; their anchors resolve, and no expiry/waiver was added. The problem is enforcement and two incorrect facts, not missing owner fields.
- The six committed age-tail population values (`packages/microcosm-build/src/microcosm/build/uk/ons_age_tail_band_populations.json:5-16`) match the current target-membership facts (`packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:6581-6623`, `packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json:6866-6907`). The generated source-stage digest independently recomputed to `84d1c1c85f0172a9a396ef69d445ee33cf1340f62ff73c8653c399af40087bf4`; release coverage remains 145 required inputs and zero exclusions.

## Verification log

- `pytest -q` over `test_uk_spine_parity_instrument.py`, `test_uk_signed_differences.py`, `test_uk_age_tail.py`, and `test_uk_identity_stability_receipts.py`: **82 passed**.
- `pytest -q` over `test_uk_parity_reference.py`, `test_uk_terminal_gates.py`, `test_uk_weighted_integrity.py`, `test_uk_source_stages.py`, and `test_country_spec.py`: **167 passed, 1 skipped**.
- Both runs used the byte-compatible sibling environment with this worktree's five package source roots first on `PYTHONPATH`; no dataset or candidate build was run.
- Independent production-function and full-CLI probes reproduced the E6 mismatch, signed magnitude/count/entity failures, loose-band bypass, and missing-total/identity bypass described above.
- `git diff --check 7b90bb1882b0248d751a64bf817ec127e5c42a47..pr-747`: clean.

## Minimum closure before merge

1. Make final ages and stored age-derived NHS columns satisfy the E6 contract, with an integration test.
2. Run and record the complete acceptance ladder on the exact final 25-stage candidate, binding every receipt to candidate content identity and stage roster.
3. Make strict parity enforce registered direction/magnitude/count/entity scope and the contract share band; close and identity-bind weighted-total key coverage.
4. Correct the stale incumbent magnitude and student-loan entity facts, then rerun the targeted/adversarial tests.
