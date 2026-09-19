# Validation cost and borrow boundaries

The US graph must detect changed sources, substituted artifacts and changes to retained Populations without repeating the same expensive work at every nested call. This note records the current decision and the proposals still under review. It does not change dataset release requirements.

## Adopted: reuse source compilation

The ACS native coverage owner keeps a bounded process-local cache of compiled code. Its key includes exact source bytes, filename, mode, flags, inheritance setting, effective optimization level and compiler identity. Every source read and loaded-function, global, alias and closure check remains fresh. Cached code is neither source authority nor a portable receipt.

The change passes 25 focused controls and the original invented financial fixture, including cold execution, required replay and final checks. One paired profile took 216.82 wall seconds versus 308.26 seconds before the change, a 29.66% reduction. CPU time fell from 303.90 to 214.69 seconds. These instrumented observations establish an improvement for that fixture; they do not establish native-data performance. See [the experiment](../experiments/us-acs-compilation-cache-adoption-20260910.json) for exact source, test and verification identities.

The native pilot already in progress retains its original frozen source snapshot.

## Next: remove duplicate work within a check

A source review confirmed nine direct Frame storage traversals in a healthy `verify_acs_native_coverage` call: three calls to `_verify_frame`, each performing three storage hashes. Some repetitions occur across supplemental axis and resolved-weight checks, so collapsing all nine into three is not automatically equivalent under the existing mutation checks.

The next candidate will share the first storage calculation with the supplemental fingerprint calculation, while preserving the independent final prepared-Frame storage pass and all three outer verification points. That would reduce nine direct traversals to six. It needs a regression that mutates a cell during the supplemental check, unchanged receipt and fingerprint comparisons, and a paired measurement before adoption.

## Keep separate: code, files and retained objects

Three different facts currently appear together in nested validation:

- Loaded-code identity: the interpreter's functions, code, globals, aliases and closures.
- Agreement with files: source bytes, archive snapshots and stored graph artifacts still match their recorded identities.
- Retained-object identity: the actual Frame, Population, weights and ancestry still match their issued state.

An independent Fable 5.1 review recommended making these checks explicit and performing expensive file verification primarily at public entry and emission boundaries. Moving file checks changes when an altered archive is detected, so it needed a documented entry contract and tests at the boundaries before implementation.

**Adopted 2026-09-16, as a scoped epoch rather than a boundary rule.** `survey_population_preparation.verification_epoch()` opens a verification epoch around a graph run. Outside one, nothing changes: every borrow re-authenticates in full. Inside one, each borrow still pays the cheap tier — loaded-code identity through `_encode(_producer())`, the attached owner payloads and the live authority — and the expensive file and retained-object tier is skipped only while a signature is unchanged: the stat identity of every path the four foreign verifications and `_source_files` re-read, including `_file_stats` over the whole source roster, the identity and length of every borrowed payload, and a read-free witness of each live Frame's storage. The roster's stat identities are read on every borrow, but into that signature rather than as a comparison of their own: a signature that moved is a memo miss, not a refusal, so the complete validation the miss runs is what refuses, with the code it raises today. Leaving the epoch, at every nesting level, re-validates every memoised capsule in full with the memo bypassed, so a change no signature can see still refuses before the run returns anything; a close whose signature moved while it was validating records no memo answer, and at the outermost close, where no borrow follows, it pays that validation itself. The contract, the residuals it accepts and the argument for each are in [verifying native sources once per run](us-native-verification-once.md); the boundary tests are `packages/microcosm-build/tests/test_us_native_verify_once_epoch.py`.

**Adopted 2026-09-19: cache immutable declaration summaries.** The ACS owner now
retains ordered tuples of top-level definition names and imported aliases, keyed
by exact source bytes. It does not retain mutable ASTs. Every check still reads
the current source and compares its declarations with the current loaded objects;
the existing loaded-code checks also remain. Source changes therefore select a
different entry, and a warm entry cannot hide a changed live alias.

The cache accepts only the original parser function, code, compiler, exact typed
defaults and AST flag values. Altered parser context bypasses it. Up to 128 source
keys of at most 128 KiB bound retained source bytes to 16 MiB; this is not a total
memory bound. Failed parsing is not cached. Like the existing code check, this
operates within a trusted Python process, not against arbitrary private-state
tampering.

The candidate passes 210 focused source, coverage, membership and row-ceiling
tests, including warm-cache parser drift, changed source and alias refusals.
Eight repeated code-only producer checks took 2.033 CPU seconds with declaration
caching disabled and 0.385 seconds with it enabled. Their complete producer
documents matched within this candidate. This does not establish a native-run
speedup or equality with producer identities from earlier source versions. The
active native retention experiment keeps its separate frozen source. See
[the declaration-cache experiment](../experiments/us-acs-declaration-cache-adoption-20260919.json)
for reviewed source and verification identities.

## Composition review

The same review raised kernel-registry sharing, registration order and recipient ordering. These require owner-level checks before being treated as bugs. For example, each financial run currently constructs a fresh survey prefix internally; it does not accept an existing prefix for a second extension. The PUF raw-route merger separately derives and verifies the complete receiving tax-unit axis before restoring row order. Neither observation alone proves every composition boundary correct.

The budget returned-view correction is now adopted. Detached documents and grouped bounds are constructed after the last borrowed I/O, then checked against the issued payload and retained owners. Four targeted controls passed, including nine callback branches; the post-run verification rechecked 501 source/owned files, 5,983 model-code files and 12 resources. The earlier eight financial-successor controls remain evidence for their original source revision. These are separate component runs, not twelve tests of one integrated revision. See [the correction experiment](../experiments/us-budget-detached-view-correction-20260910.json).

A second Fable source review found no confirmed correctness or ownership bug in the public PUF recipient interface. It recommended explicit cold-cache reuse assertions, exact comparisons with the original issuers, and clearer documentation of the original-source reads required by the recipient kernels. The frozen two-positive pilot remains unchanged; those additions belong to subsequent controls.

The immediate implementation priorities remain actual public PUF recipient and graph checks, authenticated donor/model binding, one whole-cohort PUF finalization and attachment, then native calibration and release verification. Performance work proceeds alongside those tasks.

## Review scope

Fable 5.1 completed a read-only source review on September 10, 2026, using the saved Hivesight login. Its requested scratch-cache read was denied and that limitation was disclosed in its result. It supplied architectural advice, not runtime approval or a release verdict. The recommendations above are distinguished from independently verified source findings and adopted changes.
