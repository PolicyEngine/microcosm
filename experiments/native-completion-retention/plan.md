# Completion-compatible native observation retention

Status: source-only plan for root review, 2026-09-19. No implementation or
new native run has started. The intended change serves the actual completion
path; a compact prefix that refuses completion is not the deliverable.

## Branch and evidence

- Fresh isolated branch: `native-completion-retention-20260919`, initially at
  `9af56aa8a7a210e18674d484b11d36f417de6759`.
- Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-completion-retention-20260919`.
- This continues the authorized native lane. The measured detached checkout
  and its original receipts remain untouched. Main reconciliation/#893 is
  separate work; no generic main primitive is introduced just to satisfy a
  fresh-main preference.
- Live `origin/main` was verified with `git ls-remote` as
  `16c8e78d2f60d629da5d70294f643bce5c4597e2`. It lacks the replay comparator,
  financial runner and completion host. Its #951 option is not used: the
  shared executor/store remain unchanged and observers always receive the
  existing detached snapshot.
- Completed attribution evidence is under
  `/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/lanes/native-attribution-9af56aa8a`,
  commit `d30cd778f76f199c77ecf7c81035e5b6941b818c`. The 1/1000 cold/required
  replay succeeded; final aggregate SHA is
  `4e7be8adfe472a684cd8ff86a7139cffb42e8a3699d8072d02775fdd9554976d`.
- Root's design review is
  `/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/NATIVE-RETENTION-DESIGN-REVIEW-20260919.md`,
  SHA `8cf77dbd7a9415b8c8eed37e18e89478d973f0983f9d61340fd763a787576ff9`.

## Source consumer audit

All line numbers refer to the unchanged 9af source under
`packages/microcosm-build/src/microcosm/build/us_runtime/` unless specified.

- `graph_atomic_survey_financial.py:1642` retains each detached observation.
  Comparison occurs at `:1796` and again at `:1948`; expected populations and
  stamps remain independently reconstructed. Live outputs consumed later are
  the financial attach boundary, optional property boundary, optional tax
  gate, the actual final output and existing prefix boundaries (`:1813-1824`).
- `_node_population_seals` (`:500`) and `_pure_run` (`:675`) assume an exact
  all-node roster. Both need a private compact branch with an explicit retained
  roster; the existing all-retained branch remains unchanged.
- The `child_property` recursion at `:1320` must forward the selected profile
  to the base and completion extension. Silent fallback to `all` is forbidden.
- `graph_survey_completion_host.py:814` consumes every base snapshot as an
  expected population. Its `_states` call (`:831`) is a second consumer, not
  merely another comparison. `graph_atomic_survey_population.py:79` reads
  structural column rosters, the last mass record and `weight_cap_receipt`,
  in addition to rederiving graph/key/implementation/typed-artifact and
  tolerance-writer state.
- Completion's actual live consumers are child VERIFY (`:648`, `:806`,
  `:836`), child DONOR/RECIPIENT (`:657`), receiving/role observations for
  `boundary.expected` comparisons (`:663`), and final tax gate (`:842`).
  Completion also seals all retained observations through `attestation` and
  `pure` (`:535`, `:551`). Retain every extension observation initially: this
  covers those consumers and keeps the special support custody intact.
- The private support DONOR/RECIPIENT/FIT/DRAW populations must continue to
  use completion's declaration-bound `_population_stamp` (`:233`), not the
  ordinary US-only stamp. Existing custody tests assert the distinction.
- No other production consumer of `state.node_populations` was found outside
  these two modules. Tests also inspect its legacy all-retained format.
- Completion requires a property-enabled base (`:311`). The real 45/49-node
  fixtures have 35/39-node bases plus 10 extension nodes; the existing roles
  fixture has 51 total nodes. A 19-node-only measurement does not demonstrate
  compact full-completion compatibility.

## Narrow implementation scope

Only private US runtime helpers, the four owning runtime modules below, and
their flat synthetic tests may change. No executor/store, generic API,
launcher, source parser, source authority, `_run_document` schema or unrelated
country code changes. The separate CSV-scanner lane owns its modules.

1. Add a private witness helper next to `survey_population_replay.py`. Leave
   the original `_series`, `same_replayed_frame` and
   `same_replayed_population` byte-identical as the differential oracle.
2. Add private `_population_retention="all" | "compact"` to the financial
   entry point and carry it explicitly through child recursion. Default
   behavior, all-node observations, mutation checks and issued ancestry stay
   unchanged. No profile-dependent normative key or receipt field is added.
3. In compact financial callbacks, make an immutable witness for every
   actual detached observation, retaining actual objects only for the exact
   boundary roster derived from the selected options. Compare independent
   expected populations to those observations at both existing checkpoints;
   retain the existing full comparisons and physical seals for retained
   objects. Final source/artifact/manifest/expected-object checks stay in place.
4. Issue a private capsule bound to the actual owner entry, compiled order,
   node/version/key identity, source/implementation closure and retention
   roster. Its contents are immutable canonical bytes and contain no arrays,
   Frames, Series, arbitrary caller objects or mutable aliases. No public
   witness deserializer or caller-supplied bytes grant authority.
5. Adapt completion in the same change: do not retain any base-intermediate
   observations from its union callback. Compare base observations against
   the internally issued base witness, save immutable observation witnesses,
   and retain all extension populations/stamps. Real support, receiving,
   child verification and tax consumers continue to receive actual objects.
   The all-retained completion path is untouched.
6. Adapt completion's node-state derivation rather than passing absent base
   populations to `_states`. The base issuer freezes only the independently
   checked population-derived inputs needed for each node: structural cell
   roster, mass receipt projection and design-cap receipt fields. When the
   completion union is compiled, rederive ALL current keys, implementation
   identities, kernel capabilities, typed contracts, seeds, artifact keys,
   frame/weight keys and tolerance-writer obligations in union order. Compare
   the resulting base entries with the issued base state and current union
   manifest. An unchanged base key or an old compiled-scope state alone is
   not proof of a valid union. Do not use union output as its own oracle.
7. `_CompletionHost.attestation`, `pure`, final comparisons and `_issue_run`
   bind both the exact immutable base-observation roster and the live retained
   extension roster. Parent `checked_view`, requalification, actual store
   artifacts, manifest Frames and output identity remain required. Revocation
   on any refused observation or later mutation remains intact.

The smallest private `_states` refactor shares one derivation loop while
selecting population-derived inputs from either live independently expected
populations or an issuer-owned frozen base capsule. It must rederive the
current union obligations for both cases; avoid importing old finished
states as if they were automatically valid in the new scope.

## Witness contract and admission

This is an observation comparison, not source authority or a mutation seal
for a discarded object. Non-retention must prove that discarded snapshots
are no longer reachable. Surviving actual boundaries retain their old live
identity and mutation guarantees. Hashes are compared modulo the existing
SHA-256 collision assumption and constructed incrementally by column/chunk;
do not serialize an entire population into another large buffer.

Use a closed registry of exact reviewed Python type objects and dtype values,
serialized as versioned protocol tags. Qualname plus `str(dtype)` is not type
identity. Admit ordinary NumPy primitive numeric/bool dtypes without objects,
fields or subdtypes; the exact pandas masked integer/bool implementations;
and explicitly enumerated `StringDtype` storage/NA policies. Object cells use
the existing typed scalar codec. Unsupported extensions, subclasses or
storage implementations receive an explicit compact-profile refusal.

Start axes with exact plain `pd.Index` in the reviewed dtype profile and
canonical `pd.RangeIndex`; admit only explicitly modeled name values and
metadata. Refuse DatetimeIndex, TimedeltaIndex, PeriodIndex, CategoricalIndex,
MultiIndex and unreviewed subclasses until their complete `identical`
semantics are represented. Do not silently discard frequency, categories,
ordering or class identity. A profile refusal is allowed even where the
general oracle would accept a type outside this declared compact profile.

Masked values carry mask, present bits, full backing bits and the actual-side
zero-under-mask predicate. Comparison remains directional: in completion the
issued base observation is expected and the new union observation is actual.
The actual union's zero-null flag grants the exception. Witness-to-witness
comparison must implement this relation, never plain equality, normalized
hash equality, swapped arguments or a transitive acceptance shortcut.

For supported, well-formed values, acceptance and semantic refusal behavior
must match the untouched oracle. Eager witness admission may refuse malformed
or unsupported actual state earlier, and its code precedence may differ when
multiple defects coexist. Document that earlier fail-closed boundary; do not
claim universal exact error-code equivalence. Retained-object mutation and
source/custody failures retain their current checks and named codes.

## Synthetic acceptance matrix

- Differential Series/Frame/Population cases: every admitted dtype, endian
  and signed-zero/NaN bits, masked present/null backing, string storage/NA,
  scalar object codec, schema/context, flags, strata, weights/design weights,
  mass log/ledger, owners/version and ordering. The directional masked triad
  tests each pair directly; no equality/transitivity assumption.
- Exact type/class and axis cases: dtype subclasses with identical names,
  categorical ordering, datetime/timedelta frequency, RangeIndex forms,
  MultiIndex, unusual names and non-reflexive names. Unsupported compact
  cases must refuse explicitly; supported cases match the oracle.
- Error precedence: malformed mixed-defect examples establish the documented
  eager admission refusal rather than falsely asserting universal code parity.
- Weak-reference/finalizer and alias tests prove base intermediates die after
  callbacks in both the compact base and completion union. Witness bytes stay
  unchanged after deliberate mutation of the former snapshot, and no witness
  field shares buffers. Retained boundaries still fail on mutation/identity
  substitution, including equal-content replacement objects.
- Exact callback roster, duplicates, missing/reordered observations, profile
  forwarding through `child_property`, private-capsule replacement, foreign
  issuer, source/producer changes and all/compact ancestry equality.
- Forged/stale union scopes: changed typed edges, capabilities, weight/mass
  declarations, tolerance writers, source keys, implementations or artifact
  contracts refuse even when a supplied old base state looks internally valid.
- Default 19-node, property/status variants, and existing completion 45/49/51
  tests stay unchanged. Add compact cold + required pairs for actual 45 and
  49 synthetic unions and the existing 51-node role case. Assert all base nodes
  remain hits, support custody uses its special stamps, every final output and
  lineage is correct, and no base-intermediate observations are retained.
- Hostile stored values for a discarded base node, source mutation during
  execution, expected-object mutation between the two checkpoints, retained
  support mutation, and final output substitution all refuse. No test may
  weaken store admission, child verification or revocation to obtain parity.

New test files stay flat in `packages/microcosm-build/tests/`. Run targeted
replay, financial, successor, completion/support-custody tests and the
unchanged executor tests, then Ruff and `tools/ci_test_groups.py --verify`.
Synthetic fixtures only: no private source loading or actual native launch.

## Review and later measurement gates

Root must read this plan before implementation. The implementation receives
independent source/test review before a numerical launch. At a single new
source pin, compare cold-all, required-all, cold-compact on a fresh store and
required-compact: exact keys, canonical receipts, artifact hashes, final
Frames, ancestry and source checks must agree across retention profiles.
Producer-bound keys may change between untouched 9af and adapted source;
cross-source validation therefore records changed provenance and checks
explicit semantic/output equivalence, not invented key identity.

A future actual 1/1000 pair must separately prove full completion compatibility
and record retained population counts and RSS. The completed 9af attribution
is the preserved reference, not permission to launch again. No scale increase,
1/15 repeat, default switch, publication or full-build admission follows from
this plan or from synthetic success alone. Other live frames, expected maps,
manifest Frames and source preparation remain; no memory saving is yet claimed.
