> Historical review of the initial proposal, preserved verbatim. The findings
> were subsequently resolved or adjudicated. See the
> [13 September acceptance record](experiments/uk-shared-graph-contract-acceptance-20260913.md)
> for the tested revision and remaining scope.

<!-- Filed verbatim by the fix lane from the reviewer's tool result:
     subfleet run 20260913-085330-fable-review-md (dispatched 2026-09-13 08:53,
     returned 09:01). The reviewer reported it could not write this file itself
     (plan mode, no write tool), so root files the output unedited below.
     Reviewed HEAD 6f4ba4ec989eba93786b5d88033631ad253fdc5c, base 15ebde806. -->

**Verdict: REQUEST_CHANGES** for the source proposal. Two findings are actionable design defects in the new contracts; the rest are hardening and test-strength items. I could not save FABLE-REVIEW.md: this session is in plan mode with no write tool exposed, so the full review is below for root to file verbatim.

## Reviewed identity

| Item | Value |
| --- | --- |
| HEAD | `6f4ba4ec989eba93786b5d88033631ad253fdc5c` (resolved from `.git/refs/heads/uk-shared-graph-contracts-20260913`) |
| Base | `15ebde806cd1a262363f7217fe535c7234ff757f` (as given; not independently resolved) |
| Method | Read-only. No git diff was available without a shell, so I reviewed the current contents of every file in the receipts' identity table plus both new test files, the B2 edit, the kernel-contract unit test, and the amendment 25/26 text. No imports, pytest, engine, or data. |

## Ranked findings

**1. HIGH. `frame_mass_log` leaks non-declared same-version sibling output into kernel inputs with no key binding.**
`_project_context` passes `population.frame.mass_log` at `packages/microcosm-graph/src/microcosm/graph/executor.py:678`. For an ordinary node that population is the cumulative version state, updated after every ordinary node at `executor.py:2569-2570`, and `_append_frame_mass_log` runs for ordinary nodes too at `population.py:1157`. So node X sees records appended by any earlier ordinary node A in the same version, whether or not A is an ancestor. X's key binds only `frame_key(version)` plus declared input owners (`keys.py:187-209`), never A. Result: adding, removing, or re-parameterising A changes X's visible input while X's key is unchanged, so a cache hit replays output computed against a different log. This is a new charter-A hole; before amendment 26 kernels could not see the log at all. The doc sentence "incidental node order is not authority" describes the hazard but nothing enforces it.
Exact fix, minimal: in `run_graph` record `boundary_logs[node.id] = updated.frame.mass_log` when a structural node is admitted, and pass `frame_mass_log=boundary_logs[compiled.versions[node_id]]` for ordinary nodes; structural nodes keep the full incumbent log, which their key already binds through `members` (`keys.py:221-225`). Alternative that keeps sibling visibility: track which node appended each record and reject projection when a record's author is not in `_transitive_ancestors(compiled, node_id)`. Either way `test_mass_log_is_the_incoming_log` at `test_graph_frame_context.py:346-354` must change: the successor is an ordinary node reading the appender's column, and the fix makes it see `()` unless the ancestor rule is used.

**2. HIGH. A `WeightUpdate` on kind `design` leaves design anchors stale and, after an EXPAND, mixed.**
Anchors are captured once at CREATE (`population.py:333-338`) and only carried afterwards (`population.py:1159`, `_carry_design_weights` at `2125-2181`). `_apply_weight_update` replaces the frame's design-kind values (`population.py:2038`) without re-anchoring. Consequences: the calibrated cap at `population.py:2323-2351` and `realized_max_weight_ratio` at `2354-2375` compare against pre-update design weights, so a normalisation by factor k makes every later cap ratio off by k; an EXPAND after the update anchors entrants from the updated values (`2166-2179`) while retained rows keep original anchors. The #901 consumer, `uk.full.normalize`, is exactly a design-weight normalisation. The decl docstring at `decl.py:381-382` states "ancestry is untouched" as if deliberate, but the mixed-anchor case is not a coherent contract.
Exact fix: in `patch` after line 1159, when `isinstance(node.weights, WeightUpdate)` and the kind is `design`, set `design_weights[entity] = frame.weights_for(entity).values`, and say so in amendment 25. If root prefers the current semantics, refuse `kind="design"` in `WeightUpdate.__post_init__` instead; leaving it silent is the one option I would not accept.

**3. MEDIUM. `_context_digest` omission is defensible but the fields share live objects.**
Actual behaviour checked: `frame_metadata` is a `MappingProxyType` over a copy whose leaves are `Frame`-frozen tuples, frozensets and `_FrozenMapping` (`bundle.py:1346-1362`); `frame_column_order` values are `tuple[str]`; `frame_mass_log` is the population's own tuple of frozen `MassChangeRecord` objects, shared by reference (`executor.py:678`). Through the public API nothing is writable, so the omission at `executor.py:459-488` does not produce false passes. But `object.__setattr__` on a shared record silently rewrites the live version's log, and unlike `tables` the executor would not notice. This matches the existing treatment of `context.node`, so it is not blocking. Cheap hardening: digest `canonical_json(column_order)`, the mass-log record fields, and `store._encode_frame_metadata(frame_metadata)` inside `_context_digest`.

**4. MEDIUM. Two new tests are weaker than their names.**
`test_cold_then_required_replay_revalidates_the_axis` (`test_graph_weight_update.py:311-337`) only proves a hit succeeds; it never shows the axis check executes on replay. `test_a_retained_mutating_observer_changes_nothing` (`test_graph_frame_context.py:410-435`) says it rewrites the metadata view but only rewrites tables; the three new fields are untouched.

**5. LOW. The motivating "re-solve an existing calibration" case is unreachable with the shared kernel.** `calibrate.adam@1` returns no `receipt['weight_update']` and its error text at `packages/microcosm-calibrate/src/microcosm/calibrate/kernels.py:205-209` still demands a `WeightTransition`. Declaring it as a `WeightUpdate` rejects at `population.py:2025-2030`. Not this lane's file, but amendment 25's text should not claim the case is covered.

**6. LOW. Cosmetic.** `decl.py:568-572` says "weight transition's" for an update mismatch. `test_graph_kernel_contract.py:339-345` was also edited, so the receipts' "isolated to one test file" claim applies to the acceptance suite only; that is acceptable under the charter.

## Reproducing tests to invent

- **T1 leak:** probe kernel writes `float(len(context.frame_mass_log))` into `person.x`, reading only `person.age`. Run `{survey, X}` cold: x = 0. Run `{survey, A, X}` into the same store where A appends a record: X hits and reports 0 while a cold run gives 1. Same key, different truth.
- **T2 anchors:** `WeightUpdate("household","design",…)` with factor 2, then a `calibrated` transition returning weights equal to the updated design weights with `max_weight_ratio=1.5`. Currently rejected as ratio 2.0; the receipt reports realized ratio 2.0 against a frame whose own design weights it equals.
- **T3 replay:** monkeypatch `microcosm.graph.population.weight_update_receipt` to return a foreign digest during the `resume="require"` run and assert `NodeRejectedError` matching "different .household. axis".
- **T4 tamper:** a kernel that calls `object.__setattr__(context.frame_mass_log[0], "reason", "x")`; assert the next node sees the original reason. Currently the live log changes and no rejection fires.

## Residual runtime risks

- Nothing here was executed. Pandas copy-on-write behaviour in the observer test and the fixture column order at `fixtures/toy_country/person.csv` line 1 (age before income) are the two places a source-only read can be wrong.
- In `resume="require"`, a mismatched axis receipt fails mid-run at apply time rather than in `_preflight_require`, since preflight validates record shape only.
- Column order is bound to the key by construction (owners are ancestors, order follows depth then id, rewrites keep position); I found no leak there.
