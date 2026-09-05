# US relationship/eligibility person-preparation helpers

`microcosm.build.us_runtime.relationship_inputs` and
`microcosm.build.us_runtime.eligibility_inputs` each split their existing
Frame-based `with_us_*_inputs` stage into a deterministic table-helper and a
thin Frame wrapper that composes it. The wrapper's schema check, idempotence
pass-through, and person-table reattachment are unchanged; only the "run the
source-manifest stage and align the result" core moved into a reusable
function.

1. `prepare_us_relationship_person(person, weights, *, seed, time_period)` and
   `prepare_us_eligibility_person(person, weights, *, seed, time_period)`
   take the actual person `DataFrame` and an explicit weight vector aligned
   1:1 with its rows (not necessarily reindexed by `person_id`), run the
   existing `derive_us_*_inputs_from_manifest` handler through
   `run_source_stage` unchanged, align the output back onto `person`'s
   original index and row order by `person_id`, and return a copy of
   `person` with the stage's output columns attached at their legacy dtypes
   (`bool` for the boolean leaves; `float64` for
   `own_children_in_household`/`veterans_benefits`). Neither helper decides
   whether the surface already carries signal — that pass-through decision
   stays in the Frame wrapper. Both validate that `weights` is a 1-D array
   aligned in length to `person`, finite, and nonnegative; every existing
   call site sources its weights from `Frame.resolve_weights`, which already
   guarantees this, so the checks are a no-op there and only guard new
   callers that construct a weight vector by hand.
2. `with_us_relationship_inputs` and `with_us_eligibility_inputs` are now
   thin: schema check, the existing idempotence/pass-through test, then
   `prepare_us_*_person(frame.table("person"), frame.resolve_weights("person").values, ...)`,
   then reattach the returned person table to a copied `Frame`. They invoke
   no other logic than before.
3. `us_*_inputs_person_summary(person, weights)` is the real-person-table
   counterpart of `us_*_inputs_summary(frame)`: the same weighted-share and
   invariant computation, taking `person`/`weights` directly. The
   Frame-based summary is now a one-line call into it.
4. `us_*_inputs_gate_from_summary(summary)` is the pure band/invariant check
   — bands, order, and messages unchanged — factored out of
   `us_*_inputs_signal_gate` so it can run against a summary computed
   off the real person table.
5. `us_*_inputs_person_gate(person, weights)` composes the missing-column
   check, `us_*_inputs_person_summary`, and `us_*_inputs_gate_from_summary`
   exactly as `us_*_inputs_signal_gate(frame)` composes the Frame-based
   equivalents — same order, so a frame missing its output columns never
   reaches weight resolution in either path.

None of this changes what ships: the manifest-driven derivation
(`derive_us_relationship_inputs_from_manifest` /
`derive_us_eligibility_inputs_from_manifest`), its raw-column requirements,
its fail-closed behavior on missing/malformed ASEC columns, and the
legacy-schema dtype/index/row-order contract are exactly what they were
before this split — the helpers call the same handler through the same
`run_source_stage` plumbing, they just no longer require a `Frame` to do it.
A byte-for-byte comparison against the pre-split implementation (base commit
`cf7a73bc`), run in a separate process over fixtures with a shuffled/relabeled
person index, unequal per-household weights, an already-signal-carrying
pass-through frame, and a partial-null `is_disabled` column on both sides of
the recompute threshold, produced identical person tables, summaries, gates,
and error messages.

These real-person-table APIs exist so a future graph-native adapter can
reuse the incumbent checks against whatever person table and weight vector
it assembles, without constructing a `Frame`. Wiring that adapter — and any
source projection/merge it needs — is out of scope here.
