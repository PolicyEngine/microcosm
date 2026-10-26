# UK dense line release assembly runbook (#762)

The dense line `microcosm-uk-2024-25-dense` is the spine cloned K=15 times
through the atomic output-area assignment and calibrated to the national and local
target surfaces in one solve (`microcosm-build-uk --release-role dense`, the
graph driver `tools/build_uk_full.py`; `tools/build_uk_rowwise_candidate.py`
is a stub over it). It ships
on the **inspect lane only**: a constant release id, an immutable per-cut tag,
`dataset_role: non_default_local_area`, an empty `default_datasets` map, and
`--no-latest` at publication, so it can never displace the default artifact.
It is registered as `("uk", 2025, "dense")` in the private repo
`policyengine/populace-uk-private`. Publication is a separate human step.
Since microcosm#1124 (ruling of 2026-10-13, amended for microcosm#1301) the
line may ship the 150,000-household exact-k selection from that pool instead
of the whole pool, under the same requirements; see
[Sized release](#sized-release-microcosm1124).

The R16/R17 release verdicts recorded in the historical receipts used the
previous gate policy. They do not satisfy the current contract: four quality
gates now block release. No historical run was re-signed or recalibrated by the
PR #870 review fixes.

Since microcosm#1301 the rotated holdout and the incumbent-surface comparison
are opt-in diagnostics, not publication requirements. A release that skips
them says so: its manifest carries the `no_out_of_sample_claim` and
`no_incumbent_comparison` limitations, beside `engine_measurement` (how the
engine measured the file) and the names and reasons of the reviewed exclusions
the gates stood on. See [Optional diagnostics](#optional-diagnostics-microcosm1301).

## Prerequisites

- The four pins the run stood on (`spine`, `ladder`, `facts`, `manifest`) and
  the signing key in `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` (base64, 32 bytes).
- The Ledger consumer artifact, the spine H5 with its sidecar, the OA ladder.
- Optionally, the incumbent extraction
  (`tools/extract_uk_local_incumbent_surface.py`), only for the opt-in
  incumbent comparison.
- The licensed input-mass reference the national line keeps with its runs
  (`--input-mass-reference`). `uk_input_mass_parity` is release-blocking on
  this line too and records `evidence_absent` without it. The sidecar's totals
  digest must match the registry's `totals_sha256`; the totals depend on the
  installed policyengine-uk, so re-emitting them under a newer engine does not
  reproduce the pinned file.

## 1. Pre-flight the environment

```bash
uv run --no-sync python tools/preflight_uk_local_release_candidate.py --env \
  --pins <pins.txt> --spine-h5 <spine.h5> --ladder build/uk/uk_oa_ladder_2021.npz \
  --ledger-facts <chronicle-uk-artifact-dir>
```

Fails closed and by name on a missing or malformed key, a missing pin, a
digest mismatch, or doctrine constants that are not the ruled ones.

## 2. Run the release candidate

```bash
uv run --no-sync python tools/build_uk_full.py --release-role dense \
  --release-candidate --input-h5 <spine.h5> --input-sha256 <spine> \
  --ladder build/uk/uk_oa_ladder_2021.npz --ladder-sha256 <ladder> \
  --ledger-facts <chronicle-uk-artifact-dir> --ledger-facts-sha256 <facts> \
  --ledger-manifest-sha256 <manifest> --seed 42 \
  --input-mass-reference <licensed-input-mass-reference> \
  --review-date <YYYY-MM-DD> --graph-store <graph-store-dir> \
  --logbook-prev-row-digest <previous row> --out <candidate-dir>
```

Pass `--review-date` and `--graph-store` explicitly, even though both have
defaults. If this build's gate-only failures are signed (below), a second
build must repeat both to reuse this build's solve, and neither default
carries over: `--review-date` defaults to the day the command runs, which keys
a different solve on a later day, and `--graph-store` defaults to
`<out>/.graph-store`, which is empty under the second build's new `--out`.

**Retrying build 1.** If build 1 fails or is stopped, run it again into a new
`--out`, never the failed attempt's. Each attempt spools its own Logbook row
under `<out>/logbook-spool/`, and nothing refuses an occupied `--out` for
build 1 (only build 2's `--require-solve-reuse` does), so a retry into the
same directory leaves two attempts' rows there. The drafter and `--verify`
refuse a run directory whose spool names two attempts, and the assembler reads
the last row by file name, which need not be the retry's, and then refuses
the mismatch with the gate report. Do not clear the failed attempt's row or
`failure.json` to reuse its directory either: the row is that attempt's
Logbook record. Repeat every other argument, with the same `--review-date`
and `--graph-store`, so that the retry reuses the nodes the failed attempt
finished; a failed attempt that took the default store is reused with
`--graph-store <failed-out>/.graph-store`. If the failed attempt spooled a
row, `--logbook-prev-row-digest` names that row's digest (its `row_digest`,
which is also its file name), the new tail of the `uk/local` chain, as for
build 2 below.

`--release-role dense` is required (microcosm#823): the role supplies the
solve defaults (K=15, seed 42, 2,000 epochs, learning rate 0.15,
`nation_grain_family_equal_sqrt_count`), names the outputs `microcosm_uk_2024_25_local.h5` and
`microcosm_uk_2024_25_local.local_gates.json`, records itself in the
manifest, and refuses the national role's flags. `--release-candidate` pins
the doctrine (bound 10, `nation_grain_family_equal_sqrt_count`, K=15, 2000
epochs). The rotated holdout runs only with `--rotated-holdout` (an opt-in,
report-only diagnostic; `--skip-holdout` no longer exists, and the national
role refuses `--rotated-holdout`). The engine may resolve in a single block or per clone
(`--engine-blocks K`): each block's engine weights are scaled to the pool so
weight-share formulas see the pool's denominator, and the measures receipt
records whether the blocks were identical copies (an exact representation);
a per-block run is releasable only when it was. Best-effort staging
telemetry uploads every 300 s by default on this driver (the Hub allows about
128 commits per hour per repository).
For scale: at K=25 with `--engine-blocks 25` the graph driver's dense solve took
about 3 hours (2,000 epochs over 1.6 million households, M1 Max), each of three
size-search probes about as long, and the build peaked at 24.7 GiB (the
2026-10-11 100,000-household candidate). At K=15 the pool is 60 % of that size,
and the rowwise tool measured about 3.5 hours and 10 GB for an unsized build.
A run with `--rotated-holdout` also runs the holdout, which repeats the dense
solve in each of its five folds, and in a sized build the size search and
refit too. Since microcosm#901 the
dense role runs through the [full-build graph](uk-full-build-graph.md): the
candidate directory also holds `candidate.json`, `certification.json`,
`operations.json` and the `build.json` completion marker, the graph store sits
at `<candidate-dir>/.graph-store` unless `--graph-store` says otherwise, and
`--input-h5` must be a spine whose gate manifest matches the branch's gate
declarations.

**Every row's figures ship (María's ruling of 2026-10-26).** The build no
longer withholds the model figures of rows resting on fewer than ten source
households, nor of the rows that would give them back by difference
(microcosm#1262, #1383). Every row is judged by the fit gates and carries its
figures in the diagnostics, the targets table and the gate reports, so a
release goes only to the licensed private repository. The graph's evidence
summaries carry them too, which is why a build publishes to the public Orrery
only on `--publish-orrery`.

### Sized release (microcosm#1124)

A release candidate may add `--dataset-households 150000` to ship the exact-k
selection instead of the whole pool. It must carry the local size doctrine
(`local_doctrine.uk_local_size_doctrine`): 150,000 households, an exact-count
draw at π_hi 0.5, the refit baseline floored at 0.5 and a chi-square refit L2
penalty toward uniform weights at λ 0.0216, drawn with the solve seed. The role
fills those values when the request leaves them unset and refuses any other.
The microcosm#1301 amendment moved the size from 100,000, which K=15 could not
reach under the solver's L0 floor of 1e-7 (the search's open mass there was
about 94,500).

The size search runs in its own L0 bracket, from
`local_doctrine.UK_LOCAL_L0_SEARCH_FLOOR` (1e-10) up to the solver's ceiling
of 10; the solver's default bracket, [1e-7, 10], is unchanged for every other
caller. `--selection-initial-lambda` warm-starts the search and must lie in
that bracket: a value outside it, or one given without `--dataset-households`,
is refused when the command is parsed, never clamped. The microcosm#1301
release builds warm-start at 2.5e-8, and a second build repeats the first
build's warm start, not its selected penalty: the warm start keys the search,
draw, refit and holdout nodes. The search never probes a
penalty twice, and it stops with a named reason
(`search_floor_reached`, `search_ceiling_reached` or `bracket_exhausted`) when
a probe on a bound of the bracket asks to go past it; the size receipt records
the bracket as `selection_search_bracket` and every probe under
`selection_budget_search`. Every holdout fold's search (when
`--rotated-holdout` asks for one) starts from the same warm start in the same
bracket. The requirements are the dense line's (no matched-size scorecard):
every release-blocking gate passed and the pre-flight, which also checks the
size receipt and its search bracket. The release records the size, the
selected penalty and the bracket in its coverage (`size`) and a
`size_doctrine` limitation. The release identity is unchanged.

## Sign gate-only failures between builds (microcosm#1301)

A release candidate blocked only by failures that can be signed without
touching the solve is released by a second build at the sign-off's head.
Three gate-only registers take those sign-offs:

- national target rows (`uk_target_fit`, keys `name@2025`) in
  `target_fit_reviewed_exclusions.json`;
- local target cells (`uk_local_target_fit`, keys `family/area_code/metric`) in
  `local_target_fit_reviewed_exclusions.json`;
- areas below the support floor (`uk_local_area_support`) in the `exclusions`
  block of `local_area_support_exclusions.json`.

Any other release-blocking failure needs a solve change and a third build.
That covers per-family fit, the weight gates and the geography ladder.
Input-mass entries are the exception: they are signed by hand in the same PR,
with the contract's `_UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256` re-pin.

Draft the sign-off from build 1's run directory. Use a checkout whose
`gates.json` matches build 1, because the tool decodes the full gate report
against it and refuses one that moved:

```bash
uv run --no-sync python tools/draft_uk_dense_signoff.py --run-dir <build-1> \
  --approved-by <reviewer> --approved-on <YYYY-MM-DD> \
  --gate-clock <planned build-2 gate clock> --out <outside-every-git-work-tree>/signoff
```

**What the tool checks.** It reads the full gate report, the signed
`*.local_gates.json`, `*.targets.csv`, `*.area_support.csv`, the Logbook row
and the measure exclusions build 1's solve ran under. Those come from
`rowwise_candidate_manifest.json`, or, when build 1's gates stopped it before
the export, from `uk.full.target_compilation.surface.json`, which every build
that reached the calibrated gates leaves. It refuses any of these:

- inputs that disagree with one another;
- a report bound to no attempt;
- a development build (`--allow-dev-build` makes a dry run that signs nothing);
- a signable gate that failed closed;
- a window that leaves an entry premature or expired at the gate clock, or is
  longer than 92 days;
- a gate clock earlier than build 1's;
- a gate clock later than the earliest expiry among those measure exclusions
  and the local line's degenerate-register entries, which no sign-off can
  renew without changing the solve;
- national entries that break the national graph test's review invariant.

**What it drafts.** Every drafted entry carries `release_lines: ["local"]`:

- an entry for every signable failure;
- a renewal for a failing row whose entry is out of its window at the gate
  clock;
- the removal of every entry build 2 would fail as stale, unknown or out of
  its window, and of local entries that no row earns.

The national target-fit register is shared with the national line, which may
still stand on its entries, so an entry there that applies on every line is
never removed or replaced. When build 2 would fail it on the local line
(stale, premature or expired at the gate clock), it is rescoped to
`release_lines: ["national"]` and otherwise left as it is:

- An expired one also goes to review, because only the national line can
  renew or retire it. Kept beside approvals made after its expiry, it breaks
  the national graph test's review invariant, so the draft is then unsignable
  until the national line renews or retires it by hand and the draft is redone.
- A row build 1 still breaches keeps such an entry for the national line too.
  A register key holds one entry, so no local-only entry can be drafted beside
  it, and the row needs a hand sign-off. For a premature entry, a gate clock
  on or after its approval also works, since the entry is then in force.

Fit entries expire after 28 days and area entries after 3 months. Each reason
is templated: it names the key, the gate's bound or the floors the area
missed (policy constants), the build-1 attempt id and the full gate report's
sha256, plus any `--class-note` sentence and, for an area, its name. What the
tool cannot template it checks in three ways, and only these three:

- `--approved-by` must be a GitHub login (letters, digits and hyphens, at most
  39 characters) that starts with a letter. Only a login on the tool's
  `KNOWN_DIGIT_LOGINS` (`juaristi22`) may hold digits, since no rule on their
  shape tells a login from `ess7` or `rows-7`; a reviewer whose login has
  digits is added there by a reviewed change. `4521`, `rows-7` and `w312` are
  refused.
- `--class-note` text, `--area-name` names and `--adjudication` may carry
  digits only inside a citation of a repository on the tool's
  `KNOWN_REPOSITORIES` (this project's repositories and those it cites, in any
  case) and in an ISO date within 366 days of the day the tool runs, and never
  a percent sign in any form. A citation is an issue reference
  (`microcosm#1301`, `chronicle#359`, `uk-data#521` or `PolicyEngine/<repo>#N`),
  a GitHub URL of an issue or pull request
  (`https://github.com/PolicyEngine/<repo>/issues/<N>` or `.../pull/<N>`,
  optionally anchored `#issuecomment-`, `#pullrequestreview-` or
  `#discussion_r` with the comment's ten-digit id) or a GitHub URL of a file
  (`.../blob/<ref>/<path>`, whose path may hold digits). A citation right next
  to a noun a figure counts or belongs to (`uk-data#7 rows`, `household
  microcosm#4521`) is refused, as are a bare `#N`, an anchor outside a URL
  (cite the comment by its URL), another repository (`PolicyEngine/rows#7`) or
  another repository's short form (write `PolicyEngine/policyengine.py#553`), a
  comment cited as "issue comment <id>", a date far from today (`2045-12-31`)
  and any other number (`K=15`, `150,000 households`, a plan code such as
  `c9`, a bare year), before the build is read.
- The drafted entries then go through the release contract's disclosure walk,
  and a finding refuses the draft, though on a release directory the walk only
  warns. On text, the walk catches only an area's support worded with its
  measured value, such as `rows 7 < 50`.

None of these checks catches a figure written in words, so scan the outputs as
described under "The sign-off PR" below. The degenerate register is never
touched.

The local reference generator keeps its own support-floor constants,
lockstep-tested against the area register. So the tool never edits the
`bound_despite_support_floor` block, and it never retires or renews the
exclusions of the areas whose cells the generator defers
(`SUPPORT_FLOOR_DEFERRED_AREA_IDS`: the Isles of Scilly and the City of
London). If build 2 would fail one of those entries, the area-support gate is
unsignable:

- a stale entry needs the generator to move, which changes the solve;
- an entry out of its window, or a `bound_despite_support_floor` approval out
  of its window, is renewed by hand, keeping its reviewed wording and lines.
  Then draft again.

Likewise, an input-mass entry on the local line that is out of its window at
the gate clock makes the draft unsignable: the tool never drafts input-mass
entries, so renew it by hand and draft again.

**Outputs**, in `--out`, which must lie outside every git work tree (the tool
refuses one inside this checkout or any other, and `.gitignore` also covers
`PRIVATE_review.csv` and `signoff-draft/`):

- `signoff.patch`: the registers, their key-set pins (the two target-fit
  registers' in `test_uk_terminal_gates.py`, the area exclusions' in
  `test_uk_weighted_integrity.py`) and a changelog fragment;
- `signoff_draft.json`: the receipt;
- `PR_BODY.md`: gate ids, build 1's attempt id and report digest, register
  entry counts, the national register's stale-risk and near-bound counts, the
  national entries for review, the reviewer, the adjudication, policy dates,
  the names of the entries that set the assembly deadline and the tests to
  run;
- `PRIVATE_review.csv`: mode 0600, holding per-row errors, every fit row's
  stale-risk and near-bound flags and per-area support figures. Never commit
  or post it.

The draft also counts stale-risk rows (signed rows within one point above the
25% bound, which build 2 fails as stale unless it reproduces build 1 exactly)
and near-bound rows (unsigned rows within one point below it). Only the
national register's counts reach `PR_BODY.md` and the receipt. A local cell's
key is in the patch, so a public count of signed cells next to the bound would
bracket a named sub-GOR estimate: the local registers' counts are printed on
the tool's standard output and kept as flags in `PRIVATE_review.csv`.

The tool also prints the range of gate clocks build 2 may use, and the receipt
and `PR_BODY.md` give the **assembly deadline**, where that range ends: the
earliest expiry among the measure exclusions build 1's solve ran under, the
local line's degenerate-register entries and every gate-only entry build 2's
gates hold to its window after the sign-off: those on the local line, every
`bound_despite_support_floor` family whatever its line, and the local line's
input-mass entries. The range starts at the latest of build 1's clock and
those entries' approvals, and build 2 refuses a `--gate-review-date` outside
it before anything runs. Build 2, its verification and the release
assembly must all run by that date. Assembly and upload check the measure
exclusions on the day they run, and renewing a measure exclusion or a
degenerate-register entry changes the solve, so it would need a third build.

**The sign-off PR.** Branch from build 1's commit and `git apply` the patch.
Run the tests `PR_BODY.md` lists. Before committing the patch or posting the
body, scan both `signoff.patch` (every drafted reason, the adjudication and
the reviewer) and `PR_BODY.md` for disclosure: no estimate, error, weight or
support figure, in digits or in words. Merge with a merge commit, so that
build 2's commit stays in main's history.

### Second build: reuse the first build's solve (microcosm#1301)

When build 1 is blocked only by failures signed in gate-only registers, build
2 runs at the sign-off's head. It repeats build 1's command with every
argument, the sized-release ones included, and the same `--review-date` and
`--graph-store`. It changes only the output directory and the Logbook
predecessor, and adds two flags:

```bash
uv run --no-sync python tools/build_uk_full.py <build 1's other arguments> \
  --review-date <build 1's review date> --graph-store <build 1's graph store> \
  --logbook-prev-row-digest <build 1's row digest> --out <new-candidate-dir> \
  --gate-review-date <inside the range the draft printed> --require-solve-reuse
```

- `--out` must be a new directory: one holding `build.json`, `failure.json` or
  any `logbook-spool/*.json` row is refused before the attempt opens, so a
  retry after a failed build 2 also needs a new `--out`. Build 1's would lose
  its bundle, which `--verify` and the draft receipt still need; a failed
  attempt's would end up with two Logbook rows, which `--verify` refuses.
- `--logbook-prev-row-digest` must name the tail of the `uk/local` Logbook
  chain when build 2 starts. That is build 1's row digest (the `row_digest` of
  the one row in `<build-1>/logbook-spool/`, which is also the file's name),
  unless another dense attempt has been recorded since, in which case it is
  that attempt's. Build 1's own value would fork the chain, and leaving both
  the flag and `POPULACE_LOGBOOK_PREV_ROW_DIGEST` unset would start a second
  genesis. The Logbook store refuses either row and the archive export refuses
  the chain, but the build itself does not check: the remote insert is
  best-effort and the digest keys no graph node, so it never affects solve
  reuse.

The calibrated gates evaluate their reviewed exclusions on the gate clock,
which may not precede `--review-date` or be later than today. With
`--require-solve-reuse` it must also lie in the range the draft printed:
before anything runs, the build refuses a gate clock before the approval of an
in-line gate-only or degenerate-register entry, or after the expiry of one of
them or of a measure exclusion the solve applied, and names those entries and
the usable range. An expired gate-only entry is renewed by a new sign-off;
renewing an expired degenerate-register entry or measure exclusion changes the
solve and needs a third build. Every node
before the calibrated gates is required from the store, and a miss stops the
build with `SolveReuseError` naming the missing nodes before anything
computes. A miss means something that keys the solve changed between the
builds; [the graph doc](uk-full-build-graph.md#a-second-build-that-reuses-the-solve)
lists what. Expect about 1–2 hours instead of a full build. Without those two
flags, build 1 must run with `--review-date` set to the planned approval date,
and build 2 re-solves.

### Verify build 2

Verify build 2 from a checkout at the sign-off's head, with the signing
key exported:

```bash
uv run --no-sync python tools/draft_uk_dense_signoff.py --verify <build-2> \
  --build-1 <build-1> \
  --draft-receipt <outside-every-git-work-tree>/signoff/signoff_draft.json
```

Verification fails unless all of these hold:

- build 2 stood on every in-line entry at the checkout;
- no entry was stale, premature, expired or unknown, and no local-only entry
  was dormant;
- every receipt matches its register entry;
- no release-blocking gate failed;
- the signed local report passes the dense release contract;
- with `--draft-receipt`, every drafted entry is committed, no retired one is,
  and every rescoped one is committed for the national line only and otherwise
  unchanged (the receipt keeps it as committed before the sign-off);
- the assembly deadline, computed as the draft computes it from build 2's
  measure exclusions and the checkout's registers (every bound family and the
  local line's input-mass entries included), has not passed.

It warns when build 2 did not reuse build 1's solve, when its per-row errors
differ, when the disclosure walk a release directory gets finds a per-area
support figure in the signed local report, and when the assembly deadline is
two days away or less. The entries
expire: retire or renew them before `expires_on`, because
engine-free tests evaluate the committed registers at today's date.

## 3. Pre-flight the finished run

```bash
uv run --no-sync python tools/preflight_uk_local_release_candidate.py --candidate-dir <candidate-dir>
```

The pre-flight checks the manifest and the signed gate report for everything
the contract will demand: release posture attested, shippable, every
release-blocking gate passed, an exact engine population (single block, or
per-block with identical copies scaled to the pool), the doctrine values (for
a sized run, the size doctrine, a realized count equal to the request and the
size search's L0 bracket), the
A15 census household uprating and A17 tenure application, the measure
exclusions and their windows, the holdout (recorded as not requested, or, when
`--rotated-holdout` asked for it, measured with five folds and a finite mean,
and in agreement with the request), the Logbook row, the artifact digest.

It also runs the release's disclosure walk over the files assembly ships byte
for byte: the candidate manifest, the original diagnostics, the signed local
report and, when the run directory holds them, `score_vs_incumbent.json` and
`incumbent_surface_evaluation.json`. Add `--incumbent-manifest
<incumbent-dir>/incumbent_local_surface_manifest.json` to walk the extraction
manifest you will attach too. Each file is read as the contract reads it, so
one carrying NaN, or nested too deep to walk, is reported as a file the walk
cannot rule out. A finding prints a `WARN` line naming the file and the paths,
never a value, and never fails the pre-flight.

## Optional diagnostics (microcosm#1301)

Neither of these is needed to assemble or publish. Run them when there is
time; attach the incumbent comparison only when its result should ship with
the release.

**Rotated holdout.** `--rotated-holdout` on the build. The release then
records a `rotated_holdout` limitation with `claim: report_only` and the
coverage's `holdout` block (`run: true`); without it, `no_out_of_sample_claim`
and `holdout: {run: false, claim: none}`.

**Incumbent comparison.** Score the candidate, then measure the pinned
incumbent surface:

```bash
uv run --no-sync python tools/score_uk_local_candidate.py ... --output-json <candidate-dir>/score_vs_incumbent.json
```

The scorer's `--registry-json` is the candidate's target selection,
`<candidate-dir>/microcosm_uk_2024_25_local.target_selection.json` (the
manifest's `outputs.target_registry`); it scores that selection's local rows,
which must be the committed active local references (the count is read from
`uk/local_target_reference_membership.json`, `active_reference_count`) and
must all appear in the candidate's diagnostics, so scoring needs the full rung.
A candidate that did not request the holdout scores with no holdout
(`holdout_basis: none_requested`).

```bash
uv run --no-sync python tools/evaluate_uk_incumbent_surface.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024_25_local.h5 \
  --candidate-manifest <candidate-dir>/rowwise_candidate_manifest.json \
  --ledger-facts <chronicle-uk-artifact-dir> --ledger-facts-sha256 <facts> \
  --ledger-manifest-sha256 <manifest> --engine-blocks 1 \
  --incumbent-manifest <incumbent-dir>/incumbent_local_surface_manifest.json \
  --incumbent-metrics-csv <incumbent-dir>/household_metrics.csv \
  --incumbent-weights-csv <incumbent-dir>/wide_weights.csv \
  --out-json <candidate-dir>/incumbent_surface_evaluation.json \
  --out-md <candidate-dir>/incumbent_surface_evaluation.md
```

`tools/evaluate_uk_dataset_size.py` runs both as its steps 30 and 40 only
with `--incumbent-diagnostic` (or when `--steps` names them).

The build's release-blocking fit gates read the measures it resolved, per
clone when `--engine-blocks K`; the build records that representation as exact
only for the weight-share scaling
(`engine_population_representation.exact_scope`), and engine draws that depend
on a record's position can differ between per-clone and single-block runs. The
release states this in its `engine_measurement` limitation; nothing re-measures
the released file on one engine unless this evaluation is run. Its assessment
marks any national or local row more than 25% from its target, or any family
of five or more rows with fewer than half within 25%. Signed deferrals and
measure exclusions do not exempt a row.

Use the actual metrics and weights filenames from the extraction manifest.
Only one engine block is accepted. The evaluation records the pinned row
inventory check (`inventory_failures`) beside its summary. Attached at
assembly (`--incumbent-manifest`), the score, the evaluation and the
extraction manifest ship together or not at all: their identities (the
candidate and extraction bytes, the pinned row inventory, the recomputed
summary) refuse, while the measured limits and the single-block check are
recorded in the `incumbent_surface_diagnostic` limitation (`passed`,
`failure_count`) rather than enforced. National comparisons use the pinned
incumbent targets; they do not claim realized incumbent fit.

The same existing absolute quality limits apply to this surface: every row
within 25%, and at least half each family's rows within 25% when the family
has at least five rows. The within-10% family share remains diagnostic.
The candidate's fitted score uses uniform rows on its active local surface.
Its holdout uses the separately recorded weighting rule over held local
grains. Their shared cap does not make the losses directly comparable; no
ranking of fitted versus holdout losses is reported.

## 4. Assemble the release directory

```bash
uv run --no-sync python tools/assemble_uk_dense_release_dir.py \
  --candidate-dir <candidate-dir> --spine-h5 <spine.h5> --out-dir releases
```

Add `--incumbent-manifest <incumbent-dir>/incumbent_local_surface_manifest.json`
only to attach the optional incumbent comparison; without it, any score or
evaluation in the candidate directory is ignored (with a note on stderr).

Assembly requires `release_role: "dense"` in the candidate manifest (as does
the pre-flight; a candidate built before the role existed is rebuilt, never
grandfathered), verifies the hash join (every manifest output against its
bytes, the spine against its pin, the gate report against the Logbook build
id, and the full gate report `outputs.full_gate_report` against its digest),
requires the manifest's `staging_delivery` receipt (the run's version 2 staging
telemetry evidence, copied into `build_manifest.json` as `staging` so
publication can apply its undelivered-staging refusal, as on the national
lane), re-runs the candidate pre-flight, mints the cut tag
`microcosm-uk-2024-25-dense-<YYYYMMDDTHHMMSSZ>-<uuid8>` from the run's attempt
id, clones the H5 beside itself as `microcosm_uk_2024_25_dense.h5`, stages
`build_manifest.json`, `release_manifest.json`, `calibration_diagnostics.json`,
`gate_summary.json`, `uk_source_coverage.json`, the signed `uk_local_gates.json`,
the original `rowwise_candidate_manifest.json`,
`source_calibration_diagnostics.json` and `sha256sums.txt` (plus
`score_vs_incumbent.json`, `incumbent_surface_evaluation.json` and
`incumbent_manifest.json` when the comparison is attached), validates the
directory with
`microcosm.data.contract.validate_release_dir`, and only then renames it into
`releases/microcosm-uk-2024-25-dense/`. Re-assembling requires removing the
previous directory first. The JSON summary prints the publication command.

The validator's disclosure walk warns rather than refuses (microcosm#1301). A
shipped JSON document carrying an area's rows, ESS or distinct sources, or a
minimum, maximum or median of one, prints a `warning:` line naming the file and
the first paths to the figures, never their values, and the summary lists it
under `warnings`; so does a shipped JSON document the walk cannot read. Weigh
each warning before publishing. A figure in the signed `uk_local_gates.json`
cannot be edited out without failing its signature, nor one in
`source_calibration_diagnostics.json` without failing the candidate manifest's
hash of it; a figure in a file the assembler writes is fixed at its source and
the directory re-assembled. The pre-flight's `--candidate-dir` runs the same
walk over the files the assembler ships byte for byte (step 3), so a finding
shows right after each build. The walk reads key names and the shortfall
wording only: a figure under another name, in other prose or outside the
top-level JSON files is not found, so the manual scan stays.

The manifest's `reviewed_limitations` record what the release does not claim,
and the contract checks them against the coverage: `engine_measurement`
always; `no_out_of_sample_claim` or `rotated_holdout` (`claim: report_only`);
`no_incumbent_comparison` or `incumbent_surface_diagnostic`; and
`size_doctrine` for a sized release. The coverage's holdout and size records,
and the `engine_measurement` and `size_doctrine` limitations, must restate the
shipped candidate manifest: an edited coverage cannot switch between
`no_out_of_sample_claim` and `rotated_holdout`, report a holdout loss the
candidate did not measure, or change the size.

The manifest also names every reviewed exclusion the gates stood on, as
`reviewed_exclusion:<gate>:<name>` for `uk_target_fit`, `uk_local_target_fit`
and `uk_input_mass_parity`, and as `area_support_exclusion:<level>/<code>`
beside the `area_support` block's `below_floor` codes. The assembler reads the
first from the full gate report and records the names per gate in the
coverage's `signed_deferrals.reviewed_exclusions`. The contract requires
exactly those limitations; it checks the `uk_local_target_fit` names and the
`below_floor` areas against the signed `uk_local_gates.json`, and requires the
`area_support` block. The full gate report is not shipped, so the build records
the names each of the three gates stood on in the candidate manifest
(`reviewed_exclusions_stood_on`), which the release ships and hash-joins, and
the contract holds the coverage's names to that record: a hand edit that drops
a `uk_target_fit` or `uk_input_mass_parity` name from both the coverage and the
limitations is refused. Each limitation carries the exclusion's name and its
register reason, copied verbatim; the gate's measured error for the row is not
copied, but a reason can cite figures of its own (the input-mass register's
reasons cite national totals).

Assembly and every later directory validation require measured clean code
(`code.git_dirty` exactly `false`) and full measure-exclusion provenance.
Approval and expiry dates must be valid ISO dates and in force on the current
validation date; expiry-day validation is allowed, the following day is not.
The upload path invokes this validator again before uploading bytes, and
repeats its warnings. Separate support
and binding adjudications keep their own policies. The gate thresholds and
existing approvals have not been widened or renewed.

## 5. Publish for inspection (human step)

Run the printed command. Its shape is:

```bash
uv run python -m microcosm.data.publish_cli releases/microcosm-uk-2024-25-dense \
  --repo-id policyengine/populace-uk-private --artifact-root <candidate-dir> \
  --no-latest --tag-only --tag-name microcosm-uk-2024-25-dense-<timestamp>-<uuid8>
```

`--no-latest` is mandatory and enforced: publication refuses to move
`latest.json` for a non-default role. `--tag-only` publishes the immutable tag
revision without a main-branch commit. The artifact is reachable by its tag and
by the registry key `("uk", 2025, "dense")` only.

The disclosure walk's warnings reach the publisher's JSON output as well as
stderr: `--preflight-only` adds them to its verdict under `contract_warnings`,
and publication lists them the same way beside the pointer it prints. Neither
refuses on them. The immutable tag's message records that the release was
published past them and how many there were, never what they say, since they
name paths in the shipped files.

## Promotion is a separate change

Making the dense line (or a sparse successor via the L0 penalty, #762 I10) a
default dataset is a registry and contract change with its own review; nothing
in this runbook promotes anything.
