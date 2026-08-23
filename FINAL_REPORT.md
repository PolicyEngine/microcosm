# Final report: battery residual FIX packages

## Outcome

The only new generating repair is committed at `eaba1eab`: late
ASEC-to-ACS `weeks_unemployed` calibration no longer treats positive
unemployment compensation as a carrier prerequisite. That relationship is
specific to the PUF QRF leg, not the direct ASEC `LKWEEKS` source leg
(`weeks_unemployed.py:791-830,833-840,959-983,1266-1276`).

The SHA-pinned no-build replay predicts that this repair changes the weeks
incidence ratio from `0.031371146` to `1.0006685424`, while retaining integer
positive donor support, QED `0`, and immutable bytes. The serial host owner's
guarded 1% rerun remains required to observe the new pool result.

The other owned checks did not justify another local change:

- both assigned adult-care expense mechanisms ran under pkg3 and its incidence
  check is green;
- unemployment-compensation positive-amount QED is already exactly `0`;
- Schedule D's assigned joint-parent incidence is green; and
- SSI was traced past take-up and eligibility to upstream countable-income
  distributions, for which this lane owns no calibrated surface.

No pool build, push, gate/band/ceiling/fold/seed tuning, exclusion change,
certification, publication, or release mutation occurred in this lane.

## Frozen evidence and residual universe

- Baseline gate SHA-256:
  `1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a`.
- Pkg3 gate SHA-256:
  `3ace0af0fd9e2ed6cb37cb110280f0c5cade182118c62737635c7ad177050ac3`.
- Mirror-deduplicated physical failures: `127 -> 114`.
- Pkg3's terminal and agreement gate objects are JSON-equivalent, so the 114
  count uses one canonical failure array rather than double-counting mirrors.

The 114-check list was recomputed before choosing a change. The named residual
slice, exact artifact values, and ownership decisions are recorded in
`_LANE-NOTES.md`; neither gate artifact was modified.

## Per-check disposition

### Adult-care post-reconciliation: both assigned mechanisms resolved

Mechanism: person-grain transfer output can create an expense on a
nonqualifying person or multiple carriers in a tax unit. The section-21 mask
distinguishes qualifying dependents and married heads/spouses; reconciliation
clears only mutable invalid/duplicate carriers, and an immutable positive
blocks additions within its unit (`acs_transfer.py:660-739`).

Existing pkg3 fix: before reconciliation, the late owner limits mutable
carriers to qualifying rows and additions to one stable candidate per empty
tax unit (`stacked_spine.py:8698-8716,8930-8987`). The shared amount owner
maps only mutable positive recipients from positive ASEC donor support and
leaves immutable values untouched (`post_transfer_calibration.py:573-720`). The owner
then requires reconciliation to be byte-identical and records
`verified_no_op` (`stacked_spine.py:9019-9041`).

Regression: route/exclusivity and immutable-blocking coverage is in
`test_us_acs_transfer.py:2263-2333,2513-2574`; donor-support mapping and
immutable-byte coverage is in `test_us_post_transfer_calibration.py:354-436`; terminal live
validation is in `test_us_multispine_pool_tool.py:2302-2341`.

Expected and observed effect: the assigned positive-incidence ratio is green
at `1.024210809`. The pkg3 execution receipt records 27 mapped rows and amount
QED `1.738865343 -> 0`; the independent battery reports insufficient QED
support because its ASEC leg has only four positive rows. No residual adult-
care change is warranted. The separately red incapacity flag is not either of
the two assigned expense checks.

### Model-required targeted calibration: unemployment amount QED

Mechanism and existing fix: unemployment compensation is declared
`preserve_recipient`, so carrier membership is frozen while the shared amount
owner rewrites only mutable positive values from positive ASEC donor support
(`post_transfer_calibration.py:190-197,573-720,930-940,963-993`).

Regression: the preserve-mode test requires identical carriers, all five exact
quantile anchors, QED `0`, donor support, and the preserve-carriers invariant
(`test_us_post_transfer_calibration.py:354-390`).

Expected and observed effect: the assigned positive-amount QED is already
`0.0`; no code change was made. Its `0.131859569` incidence ratio is a distinct,
unassigned criterion and remains intentionally unchanged.

### Schedule D joint-parent reconciliation

Mechanism and existing fix: Schedule D distributions are not independently
fitted. They are the packaged share of positive
`long_term_capital_gains_before_response` only when the mutually exclusive
`non_sch_d_capital_gains` route is absent
(`acs_transfer.py:611-657`). At pool grain, both parents are first summed by
tax unit; the derived value is placed on the first missing person and remaining
missing members receive zero, while observed rows are preserved
(`multispine_pool.py:2868-2946`).

Regression: the derivation tests require the share identity, route
exclusivity, and complete parents (`test_us_acs_transfer.py:2461-2510`).

Expected and observed effect: the assigned positive-incidence ratio is green
at `1.092661185`. The residual Schedule D QED `0.352941176` follows still-red
PUF-tax parent amount shapes; independently remapping this child would break
its deterministic parent identity and cross into the PUF-tax lane. No local
change was made.

### SSI incidence residual

Mechanism: SSI take-up does not explain the difference. The pool owner fills
unresolved take-up inputs from the installed engine default and records that
provenance (`multispine_pool.py:2970-3091`); the contract declares the SSI
default `true` (`take_up_contract.json:121-145`). The pkg3 receipt records all
80,395 rows as defaulted, with no seeded or preserved rows.

The installed formula floors and caps `uncapped_ssi`, then multiplies by
take-up (`policyengine_us/variables/gov/ssa/ssi/ssi.py:13-40`). Eligibility is
the aged/blind/disabled, resource, and immigration conjunction
(`policyengine_us/variables/gov/ssa/ssi/is_ssi_eligible.py:10-18`). The income
test subtracts `ssi_countable_income` from the eligible amount
(`policyengine_us/variables/gov/ssa/ssi/uncapped_ssi.py:11-16`). Countable
income sends earned, unearned, parentally deemed, and in-kind support through
the exclusions, then adds spousal deemed income afterward
(`policyengine_us/variables/gov/ssa/ssi/eligibility/income/ssi_countable_income.py:28-87`;
`policyengine_us/variables/gov/ssa/ssi/eligibility/income/_apply_ssi_exclusions.py:21-43`).

The read-only pkg3 decomposition exactly reproduces the `1.3354825449451269`
incidence ratio. Weighted eligibility is slightly lower on ACS
(`0.1888374135`) than ASEC (`0.1970392720`), but conditional
`uncapped_ssi > 0` is higher on ACS (`0.1440777296` versus `0.1033936553`).
Eligible-person countable-income q10 is correspondingly lower on ACS
(`6,884.259765625` versus `8,880.0`). The generating divergence is therefore
upstream income shape, not take-up, eligibility, or an SSI-local carrier
selector.

Fix/regression/effect: this repository keeps SSI formula-owned, materializes it
only on an ephemeral gate view, and rejects persistence
(`multispine_pool.py:3094-3111`). No frozen adjudication row grants this lane a
mutable surface over the upstream incomes. Changing the default or formula
would substitute an unauthorized mechanism, so no SSI code or local-fix
regression was added; the expected local effect is exactly zero.

### Weeks-unemployed incidence residual

Mechanism: ASEC `LKWEEKS` is directly validated as integer `-1` or `0..52`,
with `-1` mapped to zero; it is independent of unemployment compensation
(`weeks_unemployed.py:791-830`). UC-based zeroing is inside the PUF-only QRF
postprocessing (`weeks_unemployed.py:833-840,959-983`), and the signal audit
checks that relationship only on PUF rows
(`weeks_unemployed.py:1266-1276,1333-1337`). Pkg3 had incorrectly promoted
that restriction into later ASEC-to-ACS calibration
(`33bf52fe:post_transfer_calibration.py:122-180,237-244`;
`33bf52fe:stacked_spine.py:8863-8876`). Only 32 of 34,293 mutable ACS rows had
positive UC, so carrier capacity saturated and the ACS weeks share collapsed
to the UC share, producing ratio `0.031371146`
(`reproduce_us_post_transfer_weeks_checkpoint.py:28-62,132-230,278-320`).

Fix: weeks is now ordinary late `match_reference` with no special constraint
(`post_transfer_calibration.py:122-178,190-240`). The stacked owner consequently
uses every transferred-null then nonnull ACS clone-0 row as mutable, allowed,
and addition-candidate support (`stacked_spine.py:8930-8987`). Carrier
selection still targets the weighted ASEC share, and positive values still map
only from positive ASEC donors while all nonmutable bytes remain protected
(`post_transfer_calibration.py:573-720,782-817,819-943,963-993`). Only the
authored policy content hash changes
(`post_transfer_calibration.py:297-344`;
`test_imputation_lineage_spec.py:103-105`); no schema, comparator, or gate
changes.

Regression: `test_weeks_late_calibration_does_not_require_unemployment_carriers`
sets every UC value to zero and still requires positive integer donor-supported
ACS weeks, default mutable masks, exact reference share, no capacity limit,
and immutable-byte preservation (`test_us_stacked_spine.py:6367-6438`). Policy
identity coverage requires weeks' `special_constraint` to remain `none`
(`test_us_post_transfer_calibration.py:85-104`).

Expected effect from the SHA-pinned no-build replay:

- allowed/addition rows: `32 -> 34,293`;
- positive ACS rows: `32 -> 2,174`;
- incidence ratio: `0.0313711455 -> 1.0006685424`;
- positive support: integer ASEC donor values `1..46`, with zero violations;
- capacity-limited: `false`;
- QED: remains `0`; and
- immutable bytes: preserved.

The replay pins all assembled, UC, and weeks checkpoint digests and asserts
those exact current-scope outcomes
(`reproduce_us_post_transfer_weeks_checkpoint.py:1-6,28-62,91-129,132-320,323-378`).
It performs no fit, late-producer execution, artifact write, or pool build.

```bash
UV_CACHE_DIR=/private/tmp/microcosm-residual-uv-cache \
  uv run --no-sync python tools/reproduce_us_post_transfer_weeks_checkpoint.py \
  --checkpoint-stage-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/pkg3/pool.checkpoints/stacked/8f5077d6a1d5440b241f22fe4d20ad1d889924a27d094cb669e1035f9306546b \
  --carrier-scope current-mutable \
  --expect valid
```

## Verification

The required `uv sync --all-packages --extra us` was attempted first. The
sandbox blocks the default cache and network resolution, so verification used
the complete lock-compatible pkg3 environment with its five editable package
links repointed to this worktree.

On the exact implementation tree committed as `eaba1eab`, each package root
exited zero:

```text
uv run --no-sync pytest -q packages/microcosm-frame/tests
uv run --no-sync pytest -q packages/microcosm-fit/tests
uv run --no-sync pytest -q packages/microcosm-calibrate/tests
uv run --no-sync pytest -q packages/microcosm-data/tests
uv run --no-sync pytest -q packages/microcosm-build/tests
uv run --no-sync ruff check .
git diff --check
```

The complete build shard reached 100% with one expected skip and only the
established warning set. The current-scope checkpoint replay also exited zero
and matched every pinned value above. The final documentation-only tree was
rerun through the same five package roots, Ruff, and whitespace checks before
its commit.

## Serial-host 1% rerun handoff

Pkg3's `assembled` stage is semantically before late calibration and the
`transferred` checkpoint is after it
(`multispine_pool.py:200-201`; `stacked_spine.py:10009-10037`;
`tools/build_us_multispine_pool.py:3121-3175`). That assembled stage is not
mechanically reusable through the current CLI. The complete post-transfer
policy is content-hashed into stacked authority and configured checkpoint
identity (`post_transfer_calibration.py:297-344`;
`stacked_spine.py:2463-2492`;
`tools/build_us_multispine_pool.py:1150-1178,4198-4227`). The parser exposes no
audited stage-import override, and loading requires current manifest identity
before selecting a valid stage (`tools/build_us_multispine_pool.py:406-532,1181-1250,1407-1428,1716-1937`).

The serial host owner should therefore queue this exact cold 1% build in a
fresh namespace under the existing one-build-at-a-time and `<15 GiB RSS`
guard. Do not copy, edit, rebind, or reuse pkg3 manifests. Unsetting the
predecessor variable prevents ambient logbook state from entering the build
(`tools/build_us_multispine_pool.py:3971-3979`). This lane did not run this
command.

```bash
WT=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-residual-fixes
OUT=/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/residual-fixes
DATA_REPO=/Users/maxghenis/PolicyEngine/policyengine-"us"-data
DATA_PACKAGE=policyengine_"us"_data

mkdir -p "$OUT"
cd "$WT" || exit 1

env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  PATH="/Users/maxghenis/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME=/Users/maxghenis \
  PYTHONUNBUFFERED=1 \
  "$WT/.venv/bin/python" tools/build_us_multispine_pool.py \
  --sample-fraction 0.01 \
  --sample-seed 578 \
  --clone-attachment-fraction 1.0 \
  --clone-attachment-seed 578 \
  --asec-raw-stage-h5 /Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5 \
  --asec-raw-stage-h5-sha256 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe \
  --acs-household-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip \
  --acs-household-zip-sha256 8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0 \
  --acs-person-zip /Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip \
  --acs-person-zip-sha256 afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894 \
  --acs-rent-h5 "$DATA_REPO/$DATA_PACKAGE/storage/acs_2022.h5" \
  --acs-rent-h5-sha256 0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4 \
  --puf-h5 "$DATA_REPO/$DATA_PACKAGE/storage/puf_2024.h5" \
  --puf-h5-sha256 7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df \
  --puf-source-year-csv "$DATA_REPO/$DATA_PACKAGE/storage/puf_2015.csv" \
  --puf-source-year-csv-sha256 0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df \
  --checkpoint-root "$OUT/pool.checkpoints" \
  --out "$OUT/pool.h5" >>"$OUT/build.log" 2>&1

rc=$?
echo "residual-fixes exit: $rc" >>"$OUT/build.log"
exit "$rc"
```

## Commit lineage

- `f150b6da` — opened and committed the residual lane journal;
- `eaba1eab` — committed the weeks generating repair, regressions, SHA-pinned
  replay, complete scoped adjudication, exact host handoff, and green suite
  record; and
- this documentation-only closeout commit — publishes this final report and
  closes `PROGRESS.md` without changing the tested executable tree.

The next authorized action is the serial host owner's guarded 1% rerun above.
Certification, publication, pushing, and release-chain mutation remain outside
this lane.
