# Retirement model/data: blocked serial-owner f025 charter

## Disposition

This file is a run charter, not authority to execute and not an artifact
receipt. No f025 pool, refit, or Phase-P gate was run by this lane.

**Cold f025 execution is blocked.** A historical f025 build reached
`81,026,367,488` bytes RSS and a historical f001 stacked stage reached
`84,729,479,168` bytes. Both are far above the binding `<15 GiB` limit. The
owner may unlock a run only after either:

1. a reviewed memory-safe implementation has demonstrated the complete cold
   path below the fixed 14 GiB fail-closed stop; or
2. an exact candidate resume bundle has been authenticated against the
   configured namespace, complete checkpoint identity, bytes, revision,
   inputs, sampling, clone controls, and authority described below.

An old baseline checkpoint, a renamed checkpoint directory, an RSS stop, or a
partly matching identity is not a resume. The configured identity is only the
pre-input routing identity (`tools/build_us_multispine_pool.py:1192-1208`); the
complete checkpoint identity additionally binds verified input `{sha256,
size_bytes}` pairs, the realized stack manifest, model configuration, operator
order, and authority (`tools/build_us_multispine_pool.py:1088-1189`). Resume
discovery recomputes that complete identity and accepts exact equality only
(`tools/build_us_multispine_pool.py:1223-1286`).

Two other blockers are independent of memory:

- `REFIT_SHA` does not exist until the five-leg dense-refit implementation is
  reviewed, suite-green, and committed.
- The two pension and four Social Security legs are concept mismatches. They
  are not refit work and remain non-runnable until the owner adjudicates the
  exact equations in `AUDIT.md`. No exclusion may be created by this lane.

## Exact scope: five refits, six concept decisions

The reviewed `REFIT_SHA` must change only the following five physical legs:

| Leg | Required model/data change |
|---|---|
| `taxable_ira_distributions` | A post-compatibility single-target exact-ASEC-clone-0 two-part overlay: participation from the exact clone-0 zero/positive support, then amount conditional on a positive value. Fit independently for every realized early optional-predictor pattern and recompute the QRF regime from that pattern's donor rows. Do not reuse the pension model or alter the existing early chain. |
| `keogh_distributions` | Late model trained from the exact ASEC clone-0 target, not ASEC-origin clone 1. Preserve its realized degenerate/positive support as observed at f025. |
| `taxable_401k_distributions` | Same exact clone-0 late donor rule. |
| `taxable_403b_distributions` | Same exact clone-0 late donor rule. |
| `taxable_sep_distributions` | Same exact clone-0 late donor rule. |

The four late targets may share feature preparation, but their targets,
support counts, regimes, fits, draws, and receipts remain target-specific. For
every target/pattern the receipt must persist ordered predictors, donor-index
SHA-256, donor count, negative/zero/positive carrier counts, derived pattern
seed, and the regime recomputed from that exact support. Regime detection is
unweighted and distinguishes degenerate-zero, positive-only, and
zero-inflated-positive support (`packages/microcosm-fit/src/microcosm/fit/qrf.py:92-150`);
the gated fit uses the sign classifier and sign-conditional forests with
sample weights (`packages/microcosm-fit/src/microcosm/fit/qrf.py:1333-1442`).
No f001 regime label may be copied into an f025 receipt.

The taxable-IRA implementation must preserve the current early chain. IRA is
target 7 immediately before Social Security retirement in the first bounded
eight-target batch; moving or refitting it inside that batch would change the
owner-blocked target's donor/recipient prefix and RNG path
(`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:371-514`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2338-2379`;
`packages/microcosm-fit/src/microcosm/fit/qrf.py:649-662,1151-1230`). The
candidate must:

1. run and bank the current early family unchanged and snapshot every early
   terminal output;
2. derive a temporary strict code-4 label on exact ASEC clone-0 donor rows and
   fit a single-target overlay on the unchanged predictor/availability surface;
3. use the literal family string
   `puf_tax_itemization__clone0_taxable_ira_overlay` and overlay registry ID
   `early/asec_survey_to_acs/person/puf_tax_itemization__clone0_taxable_ira_overlay`,
   with only the unchanged hash-derived family/pattern seeds; and
4. replace only the original IRA recipient complement, preserve all producer
   cells, and require byte identity for the entire compatibility bank and every
   non-IRA early terminal column.

The strict code-4 derivation and existing early target-specific merge are
code-defined
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:994-1043,1247-1408,2902-2916`).
Its receipt must bind the literal IDs and derived seeds, strict-label/source-ID
digest, donor/pattern support and regime, compatibility and overlay bank
identities, original IRA complement hash/count, producer-byte identity, the
single terminal diff mask, and byte identity for all other early targets.

The late implementation must not refit the passing
`tax_exempt_ira_distributions` terminal output. It must:

1. run the current clone-1 late family and snapshot the complete terminal
   tax-exempt IRA column;
2. require a unique, complete join between clone 1 and clone 0 on the preserved
   person support source ID, start a hybrid donor from clone 1, and replace
   **only** the four red late target columns with exact clone-0 labels;
3. form the union of all four overlay recipient complements; fit/draw Keogh
   standalone over that entire union, including rows where terminal Keogh is
   producer-owned but a downstream leaf is not; then fit the ordered
   `[401(k), 403(b), SEP]` chain over the union with two fixed prefixes: exact
   clone-0 Keogh plus unchanged clone-1 tax-exempt IRA on the donor, and the
   refit Keogh raw draw plus the compatibility-pass tax-exempt IRA raw draw on
   recipients; use the exact family strings
   `source_operator_retirement_distributions__clone0_keogh_overlay` and
   `source_operator_retirement_distributions__clone0_accounts_overlay`, with
   overlay registry IDs formed by prefixing each with `late/person/`, and only the
   normal hash-derived seeds; and
4. replace only the four red targets on their already-authorized recipient
   complements, then require byte identity for the snapshotted terminal
   tax-exempt IRA column and for every producer-owned cell.

The strict source stage derives tax-exempt IRA on both clones, while its
internal PUF-role QRF overwrites exactly Keogh, 401(k), 403(b), and SEP
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:127-135,257-333,336-473`).
This is therefore a four-output overlay, not a five-output family replacement.
A single hybrid five-target chain is forbidden: changing preceding Keogh
evidence would necessarily change the tax-exempt IRA model and raw draw because QRF reads the
observed donor prefix and exact raw recipient prefix target by target
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:649-662,1151-1230`). Clone
construction preserves the support source ID; the current late path projects
clone 1, consumes one ordered target chain, and snapshots producer-owned cells
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:2121-2143`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8405-8440,8477-8483,8518-8538,8631-8729`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1411-1671`).
Those literal family identifiers derive their seeds through the unchanged hash
rule
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2902-2916`).
The existing compatibility path establishes the desired union-draw and
target-specific-merge semantics, but cannot itself execute this overlay
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:994-1001,1030-1043,1285-1304,1461-1480,1561-1671`).
The refit receipt must bind the joined source-ID digest, prove both sides unique
and complete, list exactly the four overwritten donor columns, and for each one
hash clone-1-before, source-ID-aligned clone-0-source, and hybrid-after bytes,
with bit equality `hybrid_after == clone0_source`. It must hash every retained
donor column before and after, bind the compatibility tax-exempt IRA
raw-prior hash, the refit Keogh raw-prior hash, the overlay dependency DAG, and
the distinct compatibility/overlay target-bank identities, and record the
overlay-union mask hash/count, full finite Keogh raw-draw coverage on that
union, both literal family/registry IDs and every derived family/pattern seed,
and the exact four terminal replacement masks. Any other row or column change
aborts the run; existing banks already bind family, target order, and chain
state
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1561-1663,1706-1788`).

### Required overlay execution seam in `REFIT_SHA`

No current public call can execute either overlay. `transfer_acs_inputs`
derives draw eligibility only as the OR of missing masks for targets in that
call and writes only nulls. A standalone Keogh call therefore cannot draw on
the four-target union, and IRA cannot replace its authenticated non-null
compatibility cells. Ordinary normalization also rejects a target declared in
two families
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1043,1285-1304,1461-1480,2283-2334`).
QRF starts with an empty completed prefix and accepts raw priors only for
completed chain targets
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1519-1523`;
`packages/microcosm-fit/src/microcosm/fit/qrf.py:1110-1150,1151-1230`).

The reviewed `REFIT_SHA` must add one typed post-compatibility overlay executor
whose bound input has exactly these semantic fields:

```text
family_id
draw_scope
per_target_write_masks
fixed_prefix_order
donor_prefix_frame
recipient_raw_prefix_frame
separate_bank
```

It must enforce the following contract:

- `draw_scope` and every write mask share the recipient index/order; every
  write mask is a subset of the draw scope and bit-equal to its saved,
  authenticated original complement. A non-null write is allowed only when the
  pre-write bytes equal the compatibility snapshot. Any out-of-mask change is
  an error.
- IRA binds `draw_scope == per_target_write_masks[taxable_ira_distributions]`
  to the saved IRA complement. Late overlays bind `draw_scope` to the OR of the
  four saved complements while retaining four distinct write masks. Thus the
  Keogh raw draw covers a downstream-only row without changing producer-owned
  terminal Keogh on that row.
- The accounts overlay appends fixed prefix columns exactly after
  `(required_predictors, realized_optional_predictors)` in order
  `[keogh_distributions, tax_exempt_ira_distributions]`. Donor prefix bytes are
  exact clone-0 Keogh and unchanged clone-1 tax-exempt IRA; recipient prefix
  bytes are the refit Keogh and compatibility tax-exempt IRA raw draws. Within
  each frame, both columns must be float64 and share that frame's exact index
  and order. The donor frame index/order must equal the authenticated hybrid
  donor index and be finite there; the recipient frame index/order must equal
  the `draw_scope` recipient index/order and be finite there. Bind separate
  donor and recipient index digests. Pass both frames as ordinary fixed
  predictors—not fabricated completed targets. The ordinary QRF target chain
  remains exactly `[taxable_401k_distributions,
  taxable_403b_distributions, taxable_sep_distributions]`.
- Compatibility declarations stay in their current `TargetFamilies` call.
  Overlay declarations live only in a separate registry
  `retirement_dense_refit_overlays_v1`, invoked afterward in this fixed order:
  `early/asec_survey_to_acs/person/puf_tax_itemization__clone0_taxable_ira_overlay`,
  `late/person/source_operator_retirement_distributions__clone0_keogh_overlay`,
  then
  `late/person/source_operator_retirement_distributions__clone0_accounts_overlay`.
  They must never be co-declared with compatibility families, whose target-bank
  identities remain byte-identical.
- The separate overlay bank identity binds registry/version/order, family ID,
  draw-scope hash/count, every target write-mask hash/count, fixed-prefix
  names/order/dtype, separate donor/recipient index digests and byte hashes,
  compatibility bank identity, predictor order, support/regime, seed
  derivation, and raw chain state.

Before `REFIT_SHA` can be reviewed, regressions must prove: compatibility
terminal/bank byte identity; IRA overwrite only on its saved complement; raw
Keogh coverage for a row with producer-owned Keogh and downstream write
authority; fixed-prefix dtype/index/order/hash; fail-closed rejection of every
mask/prefix/bank mismatch; zero output diffs outside authenticated masks; and
bit-identical uninterrupted versus target-banked/resumed overlay draws and
state. These are implementation tests, not new scientific gates.

The other six legs are deliberately outside the refit:

- `taxable_private_pension_income` and
  `tax_exempt_private_pension_income` require one owner-approved retirement
  total/taxability/public-private role equation. ASEC pension source roles and
  annuities cannot be represented by independently refitting the two current
  private leaves. The redesign must conserve the approved total and must
  state how public pension, private pension, annuity, railroad, other, mixed,
  and unclassified records are represented or declared absent.
- `social_security_retirement`, `social_security_disability`,
  `social_security_dependents`, and `social_security_survivors` require the
  owner-approved ambiguity/declared-absence equation in `AUDIT.md`. A row whose
  combined amount cannot be uniquely assigned is not training data for an
  arbitrary component.

Thus a future dense build refits exactly five legs. It may measure all eleven
physical battery comparisons, but it must not describe the six concept legs as
refit, fixed, eligible, or excludable.

## Immutable run order and revisions

One owner supervisor must hold one lock continuously across this sequence:

1. five-leg dense refit, revision **BLOCKED: reviewed `REFIT_SHA`**;
2. removal-only factorial, revision
   `1a8ad451c6eff17d405ef75cbdd014de72447153`;
3. broad-addition/direct-predictors-retained factorial, revision
   `539e415defb27bf103a40081239f123ce9d76c6d`.

The two factorial commits are frozen surgeries, not substitutes for the dense
refit. Removal changes only the early person/`puf_tax_itemization` optional
surface by dropping the direct Social Security and retirement analogues
(`1a8ad451:packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:151-167,374-401`).
Broad retains both analogues and selects the frozen ordered 20-predictor early
surface
(`539e415:packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:156-212,438-467`).

The frozen arm authority bindings are:

| Arm | Authority SHA-256 | Gap-fill-plan SHA-256 |
|---|---|---|
| removal | `16892c623844c21955ddcd4b39b829613db749b195922cbe2b5337aeb8c4cbdd` | `6ad7b75b193950cfb273daa1660df7e94440cc24e0806963bff5c333a94b0fdd` |
| broad | `7f6eab189fa51627c4e78c3cfcd2517f723a909d9197e538fbfec888261d564a` | `642646981e96fe67cc29e687a8a8362691bcc571c5c491426243e836f16202d6` |

The production authority receipt live-digests the canonical bundle and its
components (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3317-3385`),
and production emission rejects component, direction, registry, profile, or
canonical-identity drift
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3600-3695`).

## Fixed source pins

| Role | Absolute path | SHA-256 |
|---|---|---|
| ASEC raw stage | `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5` | `51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe` |
| ACS household | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip` | `8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0` |
| ACS person | `/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip` | `afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894` |
| ACS rent donor | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5` | `0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4` |
| processed PUF | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5` | `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df` |
| source-year PUF | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv` | `0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df` |

The configured digest `9506a2097a34731cf14314154a382f99de0c2fde3c8276ea794a18d6dfb196ba`
hashes the six `role -> expected SHA-256` values. The verified digest
`59587db52a879614fa55989c47b30bfe689e4c035296c306bf788e781670b682`
does **not** hash paths: it hashes sorted
`role -> {sha256, size_bytes}` after byte verification; paths remain separate
provenance fields (`tools/build_us_multispine_pool.py:377-390,1056-1073`).

The retained baseline bindings are:

| Binding | Value |
|---|---|
| artifact directory | `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1` |
| code provenance pin | `f71731202b6ff1f0dfa757b6b52aa87ff2a34161` |
| configured stacked namespace | `9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6` |
| gates | `685cad63d4dc62234da72501c5a3ce9ec5a81fcd3f7b412b61474a9c1d8b423b` |
| simulated checkpoint | `5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a` |
| operational receipts | `cb9ed4abb7586b6128f6aeefc9a4923e7e5934844bc3bd56e458299ac28e8c27` |

The code pin is provenance supplied by the frozen baseline owner; it must not
be inferred from an H5 filename or a moving worktree.

## Literal enclosing supervisor and atomic status prerequisite

Darwin's nonblocking lock invocation is exactly:

```bash
/usr/bin/lockf -t 0 -k \
  /private/tmp/microcosm-retirement-f025.serial.lock \
  /ABSOLUTE/REVIEWED/retirement-f025-supervisor \
  /ABSOLUTE/REVIEWED/run-authority.json
```

The literal `retirement-f025-supervisor` process is the enclosing boundary: it
acquires the lock before any checkout/sync/test/build/gate action and retains
the inherited lock file descriptor until all three run directories are closed
or the sequence aborts. Running `lockf` separately around individual payloads
does not meet the serial contract.

No such reviewed supervisor or run-authority receipt exists today, so launch
remains blocked. Before authorization, the supervisor must be tested to write
`build.status.json` atomically (`build.status.json.tmp` in the same directory,
`fsync`, then rename) on normal exit, exit 1, every other exit, every caught
signal, guard stop, and supervisor exception. The status schema must bind:

- schema/kind, arm name, exact revision, clean-tree and full-suite receipts;
- literal argv and environment allowlist;
- the six input hashes and verified pin digest;
- configured namespace and complete checkpoint identity;
- memory-path authority (reviewed cold implementation or exact resume bundle);
- start/end timestamps, child exit, guard disposition and both RSS peaks;
- artifact hashes, terminal disposition, abort reason, and closed state.

Because the current lane does not supply that wrapper, this charter does not
claim that a nonexistent status file will appear and contains no executable
cold-build command.

## Non-executable standard build argv

The following is the exact payload the future supervisor must construct after
it has authenticated `WORKTREE`, `BUILD_SHA`, `OUT_DIR`, and one of the two
memory-path authorities above. It is deliberately shown as an argv record,
not as a shell command:

```text
ENV unset: POPULACE_LOGBOOK_PREV_ROW_DIGEST POPULACE_LEDGER_URL
           POPULACE_LEDGER_KEY POPULACE_LEDGER_API_KEY
ENV fixed: PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
           OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
           NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
ARGV:
  WORKTREE/.venv/bin/python
  WORKTREE/tools/build_us_multispine_pool.py
  --sample-fraction 0.25
  --sample-seed 578
  --clone-attachment-fraction 1.0
  --clone-attachment-seed 578
  --asec-raw-stage-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5
  --asec-raw-stage-h5-sha256 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe
  --acs-household-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip
  --acs-household-zip-sha256 8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0
  --acs-person-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip
  --acs-person-zip-sha256 afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894
  --acs-rent-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5
  --acs-rent-h5-sha256 0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4
  --puf-h5 /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5
  --puf-h5-sha256 7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df
  --puf-source-year-csv /Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv
  --puf-source-year-csv-sha256 0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df
  --checkpoint-root OUT_DIR/populace_us_2024_stacked_pool.checkpoints
  --out OUT_DIR/populace_us_2024_stacked_pool.h5
```

No extra argument is permitted. In particular, do not pass a Logbook
predecessor, do not read or write `logbook-pending-chain.txt`, and do not tune
any gate, band, ceiling, seed, fold, estimator count, batch size, or target
order. A new attempt gets a new `rN` directory. A closed or aborted directory
is preserved, never edited or reused.

## Mandatory preflight and artifact authentication

Before the still-blocked launch is unlocked, the supervisor must require a
detached exact revision, an empty worktree, `uv sync --all-packages --extra
us`, every `packages/*/tests` shard, and repository Ruff under the fixed
14 GiB no-descendant-escape guard. Every nonzero result aborts.

The run-authority JSON must be reviewed and immutable. At minimum, these
checks must pass before a child is spawned:

```bash
jq -e --arg revision "$BUILD_SHA" --arg out "$OUT_DIR" '
  .schema_version == 1 and
  .artifact_kind == "retirement_f025_execution_authority" and
  .authorized == true and
  .revision == $revision and .output_directory == $out and
  (.memory_path.kind == "reviewed_memory_safe_cold" or
   .memory_path.kind == "authenticated_exact_resume") and
  .sampling == {sample_fraction:0.25, fraction_token:"f025", sample_seed:578} and
  .clone_attachment == {fraction:1, seed:578} and
  .verified_input_pins_digest ==
    "59587db52a879614fa55989c47b30bfe689e4c035296c306bf788e781670b682" and
  (.expected_authority_sha256 | test("^[0-9a-f]{64}$")) and
  (.expected_gap_fill_plan_sha256 | test("^[0-9a-f]{64}$")) and
  (.expected_execution_contract_sha256 | test("^[0-9a-f]{64}$")) and
  (.expected_early_target_bank_identity_sha256 | keys | sort) ==
    ["asec_housing_to_acs","asec_survey_to_acs"]
' "$RUN_AUTHORITY_JSON"
```

For removal and broad, the authority and gap-fill values must equal the frozen
table above. For the refit they must equal a separately reviewed static receipt
from `REFIT_SHA`; an arbitrary value that merely matches a produced artifact
is circular evidence.

After a technically completed build, set `MANIFEST`, `GATES`, and `RECEIPTS`
from the output manifest and require the following representative fail-closed
checks (the supervisor also hashes every referenced file):

```bash
jq -e --arg pins 59587db52a879614fa55989c47b30bfe689e4c035296c306bf788e781670b682 '
  .sampling.sample_fraction == 0.25 and
  .sampling.fraction_token == "f025" and .sampling.sample_seed == 578 and
  .clone_attachment == {fraction:1, seed:578} and .random_seed == 0 and
  .input_pins_digest == $pins and
  .terminal_gates == .agreement_gate
' "$MANIFEST"

jq -e --arg authority "$EXPECTED_AUTHORITY_SHA256" \
      --arg gap "$EXPECTED_GAP_FILL_PLAN_SHA256" '
  .terminal_gates == .agreement_gate and
  .terminal_gates.gates.us_by_origin_battery.details.authority as $a |
  $a.sha256 == $authority and $a.declared_sha256 == $authority and
  $a.canonical == true and $a.canonical_content == true and
  $a.canonical_identity == true and $a.digest_matches_declared == true and
  $a.integrity_valid == true and $a.production_manifest_permitted == true and
  $a.components.gap_fill_plan.sha256 == $gap and
  $a.components.gap_fill_plan.declared_sha256 == $gap and
  $a.components.gap_fill_plan.digest_matches_declared == true
' "$GATES"

jq -e --arg survey "$EXPECTED_SURVEY_BANK_SHA256" \
      --arg housing "$EXPECTED_HOUSING_BANK_SHA256" '
  .artifact_kind ==
    "populace_us_multispine_pool_stage_checkpoint_operational_receipts" and
  .stage == "simulated" and
  (.operational_stage_receipts.impute.acs_qrf_transfer.target_bank.directions
   | keys | sort) == ["asec_housing_to_acs","asec_survey_to_acs"] and
  .operational_stage_receipts.impute.acs_qrf_transfer.target_bank.directions
    .asec_survey_to_acs.identity_sha256 == $survey and
  .operational_stage_receipts.impute.acs_qrf_transfer.target_bank.directions
    .asec_housing_to_acs.identity_sha256 == $housing and
  .operational_stage_receipts.impute.acs_qrf_transfer.target_bank
    .identity_routing.identity_mismatches == []
' "$RECEIPTS"
```

The owner must additionally recompute each target-bank identity SHA from its
canonical identity object; verify every target H5 byte hash/size and descriptor;
and require its transfer execution contract to equal the commit's static
`acs_transfer_execution_contract_identity`. The frozen Phase-P implementation
performs those exact authority, gap-fill, execution-surface, target-bank,
checkpoint, and target-descriptor checks
(`9d6eecb4:tools/predictor_set_oos_gate.py:639-705,930-1146`). A path string or
receipt-declared digest alone is insufficient.

Exit 1 is acceptable only as `completed-red` after all artifact, receipt,
identity, guard, and status checks pass. Exit 1 alone is never evidence. A
guard stop, exception, error receipt, missing status, incomplete checkpoint,
or binding mismatch is `aborted`.

## Refitted and factorial 16-row ledger contracts

The five-leg refit run must first emit a baseline-versus-refit ledger with the
same 16 adjudicated rows and the same physical-record checks below. Each row
has `baseline` and `refit` cells; the refit cell binds the refit artifact key,
unrounded criterion, official status, ASEC/ACS incidence and carrier counts,
QED when supported, distance to the frozen pass region, delta, and
`to_green|toward_green|unchanged|away_from_green`. This is the direct evidence
for the five proposed model changes; Phase P does not replace it.

The ledger bundle must also contain the late negative-control receipt described
above: the compatibility tax-exempt IRA raw bank and terminal bytes must remain
identical, its overlay/replacement-row count must be zero, all four corrected
late diff masks must be subsets of their declared recipient complements, and
every other late target must have an empty diff mask. This check is outside the
16 adjudicated rows and cannot be dropped merely because tax-exempt IRA is not
red.

It must likewise contain the early negative-control receipt: the current early
compatibility bank and every non-IRA early terminal column—including all four
Social Security components—must remain byte-identical; the overlay diff mask
must be a subset of the original taxable-IRA recipient complement; every other
early diff mask and the producer-row diff mask must be empty. This receipt is
also outside the 16 adjudicated rows and is mandatory.

The owner must separately create one factorial ledger only after authenticating
all three factorial inputs: the frozen baseline, completed removal, and
completed broad artifacts. It must bind this top-level artifact receipt for
each arm:

```json
{
  "revision": "40 lowercase hex characters",
  "gates_path": "absolute path",
  "gates_sha256": "64 lowercase hex characters",
  "manifest_sha256": "64 lowercase hex characters",
  "stacked_configured_namespace": "64 lowercase hex characters",
  "simulated_checkpoint_sha256": "64 lowercase hex characters",
  "operational_receipts_sha256": "64 lowercase hex characters",
  "authority_sha256": "64 lowercase hex characters",
  "gap_fill_plan_sha256": "64 lowercase hex characters",
  "early_target_bank_identity_sha256": {
    "asec_housing_to_acs": "64 lowercase hex characters",
    "asec_survey_to_acs": "64 lowercase hex characters"
  }
}
```

Each of the 16 check rows must then contain `baseline`, `removal`, and `broad`
cells. Every cell binds its artifact key and records the unrounded criterion,
official status, ASEC incidence, ACS incidence, ASEC/ACS carrier counts, QED
when supported, and distance to the frozen pass region. The candidate cells
also record delta and `to_green|toward_green|unchanged|away_from_green`.

The extractor must assert exactly the adjudicated 9 incidence plus 7 QED rows,
with no duplicates. It must validate the underlying physical record before
selecting a criterion: `metric == monetary_sign_separated`, `status == tested`,
a positive leg with finite incidence/counts, and a negative leg exactly
`status == absent_on_both_origins`, zero ASEC/ACS incidence, and zero ASEC/ACS
carriers. This negative-leg assertion is required for all eleven physical legs,
including Keogh, whose positive QED is legitimately unsupported. It must also
assert the frozen baseline values from `adjudication.json`; a candidate must
not silently become the baseline.

## Frozen standalone Phase-P gate

Recover the reference runner from the exact Git object, not from a moving
worktree:

```bash
git -C /Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split \
  show 9d6eecb4:tools/predictor_set_oos_gate.py \
  > "$PHASE_P_REFERENCE_RUNNER"
test "$(/sbin/sha256sum "$PHASE_P_REFERENCE_RUNNER" | awk '{print $1}')" = \
  319cd4292577c5c88322fcfe6205e176a354094950b98ca45047f7f2bb320ca1
```

Derive a separately reviewed candidate-bound copy. The only allowed AST/config
change is
`.execution_surface.artifact_role_bindings.candidate`: exact candidate
authority, gap-fill plan, and the two early target-bank identities. Comparing
the two parsed modules after replacing the whole `PHASE_P_CONFIG` assignment,
then comparing the two configs after deleting only that candidate member,
must show equality. Record the candidate-runner SHA-256. The frozen design and
receipt validation live at
`9d6eecb4:tools/predictor_set_oos_gate.py:70-343,639-705,930-1146,3094-3212`.

The runner interface is exactly four paths:

```bash
"$WORKTREE/.venv/bin/python" "$PHASE_P_CANDIDATE_RUNNER" \
  --baseline /Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1 \
  --candidate "$OUT_DIR" \
  --output "$OUT_DIR/phase_p/verdict.json" \
  --markdown-output "$OUT_DIR/phase_p/report.md"
```

This command remains blocked with the build. When authorized, it runs inside
the same enclosing `lockf` supervisor and 14 GiB guard. The frozen controls
are fold seed `20260816`, 5 folds, held-out fold `0`; quantiles
`0.10/0.25/0.50/0.75/0.90`; forest seed `260816`, 32 estimators; bootstrap seed
`8162026`, 256 clusters, 2,000 replicates, 95% confidence; incidence band
`[0.8,1.25]`; QED ceiling `0.25`; and unchanged teacher forcing, target
allowlist, comparator, and clauses A/B/C.

After deleting only
`.config.execution_surface.artifact_role_bindings.candidate`, the audited
normalizer must report
`10f29427d83c340973869fb735188967aff9a4eea770b066f604e7a28a0b5e5a`.
The completion check must also require:

```bash
jq -e --arg runner "$PHASE_P_CANDIDATE_RUNNER_SHA256" \
      --arg baseline_gates 685cad63d4dc62234da72501c5a3ce9ec5a81fcd3f7b412b61474a9c1d8b423b \
      --arg baseline_identity 9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6 \
      --arg baseline_checkpoint 5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a \
      --arg baseline_receipts cb9ed4abb7586b6128f6aeefc9a4923e7e5934844bc3bd56e458299ac28e8c27 \
      --arg candidate_gates "$CANDIDATE_GATES_SHA256" \
      --arg candidate_identity "$CANDIDATE_STACKED_NAMESPACE" \
      --arg candidate_checkpoint "$CANDIDATE_SIMULATED_CHECKPOINT_SHA256" \
      --arg candidate_receipts "$CANDIDATE_OPERATIONAL_RECEIPTS_SHA256" \
      --arg authority "$EXPECTED_AUTHORITY_SHA256" \
      --arg gap "$EXPECTED_GAP_FILL_PLAN_SHA256" \
      --arg survey "$EXPECTED_SURVEY_BANK_SHA256" \
      --arg housing "$EXPECTED_HOUSING_BANK_SHA256" '
  .schema_version == 1 and
  .artifact_kind == "phase_p_predictor_set_oos_acceptance_verdict" and
  .measurement_complete == true and
  .family_count == 32 and (.families | length) == 32 and
  .target_count == 118 and (.oos_pinball_loss | length) == 118 and
  ([.oos_pinball_loss[] | select(.is_money_amount == true)] | length) == 74 and
  .by_origin_battery.counts.total_legs == 211 and
  (.by_origin_battery.legs | length) == 211 and
  (.clauses | keys | sort) ==
    ["a_champva","b_money_oos","c_no_battery_leg_leaves_band"] and
  all(.clauses[]; (.passed | type) == "boolean") and
  .artifacts.baseline.gates_json_sha256 == $baseline_gates and
  .artifacts.baseline.stacked_identity == $baseline_identity and
  .artifacts.baseline.operational_receipts.checkpoint_binding.sha256 ==
    $baseline_checkpoint and
  .artifacts.baseline.operational_receipts.sha256 == $baseline_receipts and
  .artifacts.candidate.gates_json_sha256 == $candidate_gates and
  .artifacts.candidate.stacked_identity == $candidate_identity and
  .artifacts.candidate.operational_receipts.checkpoint_binding.sha256 ==
    $candidate_checkpoint and
  .artifacts.candidate.operational_receipts.sha256 == $candidate_receipts and
  .artifacts.candidate.run_identity.stacked_authority.sha256 == $authority and
  .artifacts.candidate.run_identity.stacked_authority.components.gap_fill_plan
    .sha256 == $gap and
  .config.execution_surface.artifact_role_bindings.candidate
    .stacked_authority_sha256 == $authority and
  .config.execution_surface.artifact_role_bindings.candidate
    .gap_fill_plan_sha256 == $gap and
  .config.execution_surface.artifact_role_bindings.candidate
    .early_target_bank_identity_sha256 == {
      asec_housing_to_acs:$housing, asec_survey_to_acs:$survey
    } and
  .gate_implementation.sha256 == $runner and
  (([.clauses | to_entries[] | select(.value.passed == false) | .key] | sort)
   == (.failing_clauses | sort)) and
  .passed == (all(.clauses[]; .passed == true)) and
  .verdict == (if .passed then "PASS" else "FAIL" end)
' "$OUT_DIR/phase_p/verdict.json"
```

The executed runner is the separately hashed candidate-bound copy. Its
permitted AST diff, its candidate bindings, and its own SHA must already have
been authenticated against the exact reference. Independently check from the
verdict and candidate receipts that
the candidate authority, gap-fill, execution contract, target-bank identities,
gates hash, simulated-checkpoint hash, and operational-receipt hash all match.
Likewise re-hash the frozen baseline gates/checkpoint/receipts and require the
pins listed above. Require the normalized-config hash separately; the jq sweep
check is not a substitute.

A clause failure may be valid measured evidence, so Phase-P exit 1 can be
retained only when every authentication and completeness assertion above
passes. `measurement_complete: false`, an unexpected clause schema, any count
other than 32/118/74/211, an integrity failure, or merely receiving exit 1 is
an abort.

## Closure and prohibitions

For each run, hash the H5, manifest, gates, configured and complete checkpoint
receipts, simulated checkpoint, operational receipts, status, guard trace,
16-row ledger, Phase-P runner, verdict, and report before atomically closing
the directory. Finish and close removal before switching to broad. Preserve
all aborted evidence under its original `rN` name.

No push, pool build from this lane, gate/band/ceiling/fold/seed tuning,
exclusion, Logbook predecessor, or pending-chain access is authorized. A green
assigned row is diagnostic evidence only; it does not make either factorial
arm landable without the complete frozen Phase-P decision and owner
adjudication.
