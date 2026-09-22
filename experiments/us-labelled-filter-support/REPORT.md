# Restated Ledger filter constraints — build lane report

Branch `us-labelled-filter-support`, stacked on PolicyEngine/microcosm#955
(`us-chronicle-feed-repin`). Draft PR: PolicyEngine/microcosm#969.
(2026-09-22: #955 merged into `main` at `6710d0f5f`; this branch is now
re-levelled on `main` and #969 targets `main`. See section 8.)

## 1. What I read

Producing the filter metadata:

- `packages/microcosm-build/src/microcosm/build/ledger_targets.py:3257` — the
  **only** producer of `ledger_filter_*` metadata anywhere in the tree:
  `metadata[f"ledger_filter_{key}"] = str(value)` over `_dimensions(fact)`.
- `ledger_targets.py:3118` — `_dimensions(fact)` returns the fact's `filters`
  mapping, else its `dimensions` mapping, else `{}`. It never reads
  `universe_constraints`.
- `ledger_targets.py:3132` — `_constraint_rows(fact)`; the constraint rows are
  stamped only as the count `ledger_universe_constraint_count`
  (`ledger_targets.py:3253`).

Producing the compiled constraint the materializer slices on:

- `us_runtime/fiscal_targets.py:3178` — `_agi_bounds(fact)` folds every
  `adjusted_gross_income` constraint row into one pair, `>`/`>=` into the
  lower edge and `<`/`<=` into the upper, formatted `str(float(value))` with
  `-inf`/`inf` for open ends.
- `us_runtime/fiscal_targets.py:2426-2440` — the SOI target reference's
  metadata: `agi_lower_bound`, `agi_upper_bound`, `filing_status`,
  `materializer: irs_soi_slice`, plus `_soi_layout_filter_metadata`.
- `us_runtime/fiscal_targets.py:2520-2532` — `_soi_layout_filter_metadata`,
  the only place `ledger_filter_eitc_child_count` is emitted by the US
  compiler (from the `eitc_child_count` dimension, else the layout groupby).

Applying it, and the guards (line numbers at this head, after my change):

- `tools/build_us_fiscal_refresh_release.py:622` —
  `SUPPORTED_LEDGER_FILTER_METADATA_KEYS`; `:643` —
  `IDENTITY_LEDGER_FILTER_METADATA_KEYS`; `:825` —
  `SUPPORTED_SOI_LEDGER_FILTERS` (only `income_range`, `filing_status`,
  `eitc_child_count`).
- `:4335` — `_soi_eitc_child_count_filter`; `:4395` —
  `_eitc_child_count_mask` (resolves every value to `== 0`, `== 1`, `== 2` or
  `>= 3`); `:4418` — `_as_bound`.
- `:4371` — `_unsupported_soi_ledger_filters`, consulted at `:5072` where a
  non-empty result `continue`s, dropping the spec from SOI materialization
  **with no error**.
- `:4601` — `_unsupported_ledger_filter_metadata`; `:4630` —
  `_assert_supported_ledger_filter_metadata`, called once at `:4820`, the top
  of `_materialize_target_frame`, i.e. before the SOI loop at `:5068`.
- `:5075-5077` — the applied AGI band:
  `mask = (agi_tax_unit >= lower) & (agi_tax_unit < upper)`, half-open.
- `:5085-5089` — the applied child-count filter.
- `tools/build_us_acs_local_release.py:138` — `state_admin_specs`; `:472` —
  `materialize_chunked` calls the production
  `release_tool._materialize_target_frame` per household chunk, so the ACS
  local release runs through exactly these guards.

Consumers of the three names, exhaustively (`grep` over `packages/*/src`,
`tools`, `experiments`): `SUPPORTED_LEDGER_FILTER_METADATA_KEYS` and
`IDENTITY_LEDGER_FILTER_METADATA_KEYS` are read only inside
`_unsupported_ledger_filter_metadata`; `_unsupported_ledger_filter_metadata`
only from `_assert_supported_ledger_filter_metadata` and the tests;
`_unsupported_soi_ledger_filters` only from the SOI loop and the tests.

## 2. The rule as implemented

`_restated_ledger_filter_refusal(key, value, metadata) -> str | None` returns
the refusal entry for one `ledger_filter_*` key, or `None` when the key merely
restates a constraint the materializer already applies to that spec.

1. `_restated_ledger_filter_concept` splits the key into concept and bound
   side using `RESTATED_LEDGER_FILTER_BOUND_SIDES`. A key whose concept is not
   in `RESTATED_LEDGER_FILTER_CONCEPTS` returns its bare name — the pre-change
   refusal entry, unchanged.
2. `agi_band`: the side selects `agi_lower_bound` or `agi_upper_bound` through
   `RESTATED_AGI_BAND_COMPILED_KEYS`; both sides go through `_as_bound`, so
   `100000` and `100000.0` agree and `-inf`/`inf` agree with the compiler's own
   spelling. Missing compiled key, unparseable value, or inequality refuses and
   names both values. A suffix-free (exact-value) restatement always refuses:
   the materializer applies a half-open band and no exact-AGI filter.
3. `eitc_child_count`: the compiled constraint is `_soi_eitc_child_count_filter`
   — the function the SOI loop calls. Both the restatement (`>= n`, `< n`, or
   `== n`) and the applied filter (through `_eitc_child_count_mask`) are
   evaluated over counts `0..16` and must produce the identical mask. That is
   population equality, not string equality: `>= 3` agrees with `3plus`,
   `< 1` agrees with `0`, `== 2` agrees with `2`, and nothing else agrees.
   A value outside `0..16` or non-integral refuses rather than being compared.
   (2026-09-22, superseded for upper bounds by Max's ruling on question 6.4:
   a restated qualifying-child `_upper_bound` is no longer read as `< n` or
   compared at all; it is refused outright, naming the unconfirmed operator.
   Lower and exact restatements are compared as described. See section 6,
   "Rulings".)
4. Both call sites use it. `_unsupported_ledger_filter_metadata` collects the
   returned entries per spec, so the existing refusal text gains the
   disagreeing values. `_unsupported_soi_ledger_filters` drops a key from its
   result when the rule accepts it — necessary, because leaving it there would
   have converted an accepted restatement from a refusal into a silent drop.

Why the probe rather than a value table: `_eitc_child_count_mask` already owns
the mapping from every accepted spelling (`3`, `3+`, `3plus`, `three or more
qualifying children`, …) to a predicate. Comparing masks reuses that mapping
instead of duplicating it, so the two can never drift.

Why metadata-only and family-blind: `_unsupported_soi_ledger_filters` receives
only a metadata mapping, and running one rule in both places is worth more than
a family gate. A spec of another family carries neither `agi_*_bound` nor a
resolvable child-count filter, so it refuses on the "does not compile" arm.
The guard is already family-blind about `ledger_filter_eitc_child_count`
itself, which is a blanket supported key.

## 3. Tests

Unit, on invented specs (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py`):

| Test | Accepted | Refused (negative controls) |
|---|---|---|
| `test_restated_agi_bounds_pass_the_guard_only_where_they_agree` | both bounds; lower only; upper only; `-inf`/`inf` ends | — |
| `test_disagreeing_restated_agi_bounds_are_refused_by_value` | — | lower `50000` vs `100000.0`; upper `250000` vs `200000.0`; labelled lower with **no** `agi_lower_bound`; exact-value AGI restatement; unknown `ledger_filter_novel_dimension` still refused by bare name |
| `test_restated_eitc_child_bounds_pass_the_guard_only_where_they_agree` | lower `3` with `3plus`; exact `2` with `2` (the upper `1` with `0` case moved to the next row, 2026-09-22) | — |
| `test_restated_eitc_child_upper_bounds_are_refused_outright` (added 2026-09-22) | exact `0` and AGI upper `200000` on the same spec, still | upper `1` with `0` (agrees if read `<`); upper `0` with `0` (agrees if read `<=`); upper with no child-count filter; `_upper_bound_inclusive` by bare key; fatal guard raises and SOI skip keeps the key |
| `test_restated_eitc_child_comparison_never_guesses_an_upper_reading` (added 2026-09-22) | — | with `RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES` emptied, upper `0` with `0` raises `ValueError` instead of being compared as an exact count |
| `test_disagreeing_restated_eitc_child_bounds_are_refused_by_value` | — | `3` with `2`; `3` with no child-count filter; `three` (not a count) |
| `test_legacy_agi_usd_dimensions_stay_refused_by_bare_key` (added 2026-09-22) | — | `ledger_filter_agi_{lower,upper}_usd` equal to the compiled band, refused by bare name at both call sites; fatal guard raises |
| `test_restated_filters_clear_the_soi_skip_only_where_they_agree` | agreeing restatement clears the SOI skip | disagreeing stays listed **and** the fatal guard refuses it first; unknown SOI filter unchanged |
| `test_restated_concept_rules_all_have_a_comparison` | the two rules that exist | a concept with no rule would raise |

Feed-level, engine-free:

- `test_pinned_chronicle_feed_compiles_no_unsupported_ledger_filters` — the
  committed fixture of real compiled metadata (50 targets: one per
  (family, `target_role`) plus four SOI rows with a band or a child count)
  clears both guards, and every `ledger_filter_*` key in the registry-wide
  census that is neither supported nor a reviewed identity qualifier is
  noop-valued on every target carrying it.
- `test_restated_bounds_track_each_compiled_band_in_the_fixture` — injects the
  restatement a labelled vocabulary would add onto each real banded row: its
  own edge is accepted, the same edge moved by one is refused. Asserts at
  least four banded rows so it cannot pass vacuously.
- `test_pinned_chronicle_feed_state_surface_compiles_no_unsupported_filters` —
  compiles the real state surface when `MICROCOSM_US_CHRONICLE_FACTS` points
  at the 164 MB feed, skips otherwise.

## 3b. Verbatim summaries, at `5b1326cad`

```
$ uv run python -m pytest packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py
232 passed, 1 skipped in 54.95s
PYTEST_EXIT=0

$ MICROCOSM_US_CHRONICLE_FACTS=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/\
    consumer_facts_us_c5e5bf8.jsonl \
  uv run python -m pytest packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py \
    -k pinned_chronicle_feed_state_surface
1 passed, 232 deselected in 768.45s (0:12:48)
exit 0                       # the arm skipped in the run above

$ uv run python tools/ci_test_groups.py --verify
VERIFY_EXIT=0                # test_us_fiscal_refresh_builder.py in the us lane
                             # group, never under [defaulted]

$ uv run ruff check .
All checks passed!
RUFF_CHECK_EXIT=0

$ uv run ruff format --check tools/build_us_fiscal_refresh_release.py \
    packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py \
    experiments/us-labelled-filter-support/census_compiled_ledger_filters.py
3 files already formatted
RUFF_FORMAT_EXIT=0
```

No attested spec-engine module was touched, so no identity pin moves.

(Historical, 2026-09-22: true while #969 was based on
`us-chronicle-feed-repin`; it now targets `main`, so CI runs. See section 8.)
**PR CI does not run on this stack.** `.github/workflows/test.yml` and
`.github/workflows/integration-tests.yml` are both
`pull_request: branches: [main]`, so a PR based on `us-chronicle-feed-repin`
gets no checks at all; `gh pr checks 969` reports none because nothing is
triggered, not because a run is pending. CI first runs when #955 merges and
#969 retargets to `main`. `gh pr view 969` reports
`mergeable: MERGEABLE`, `mergeStateStatus: CLEAN` against its current base.

## 4. Counts, from the pinned feed

`compile_us_fiscal_target_registry`, `age_targets=True`, CD vintage crosswalk,
`target_period=2024`, feed `consumer_facts_us_c5e5bf8.jsonl` (Chronicle
`c5e5bf8`, 39,158 facts), measured with the real guard functions.

| Surface | Targets | Refused | SOI silent skips |
|---|---:|---:|---:|
| Whole compiled registry | 32,866 | 0 | 0 |
| State surface, `soi_mode="full"` | 31,066 | 0 | 0 |
| State surface, `soi_mode="totals"` | 760 | 0 | 0 |

State surface by family and `target_role`:

| Family \| `target_role` | `full` | `totals` |
|---|---:|---:|
| `irs_soi` \| `soi_fiscal_distribution` | 30,306 | 0 |
| `irs_soi` \| `aca_ptc_returns` | 547 | 547 |
| `irs_soi` \| `aca_spending` | 60 | 60 |
| `usda_snap` \| `snap_households` | 51 | 51 |
| `usda_snap` \| `snap_total` | 51 | 51 |
| `cms_medicaid` \| `medicaid_enrollment` | 51 | 51 |
| **Total** | **31,066** | **760** |

Dense float32 measure matrix, 4 bytes × 1,588,854 households × specs, beside
the historical ACS local release (3,972 admin specs, 487 population measures,
4,459 fitted targets; `_buildp-runtime/logs/acs-local/release_chain.log`,
2026-08-14, feed v9.3):

| Surface | Specs | Bytes | GiB |
|---|---:|---:|---:|
| Historical admin specs | 3,972 | 25,243,712,352 | 23.5 |
| Historical fitted targets | 4,459 | 28,338,799,944 | 26.4 |
| Pinned feed, `totals` | 760 | 4,830,116,160 | 4.5 |
| Pinned feed, `totals` + 487 population | 1,247 | 7,925,203,752 | 7.4 |
| Pinned feed, `full` | 31,066 | 197,437,353,456 | 183.9 |
| Pinned feed, `full` + 487 population | 31,553 | 200,532,441,048 | 186.8 |
| Whole registry (all geographies) | 32,866 | 208,877,102,256 | 194.5 |

No selection or default was changed.

## 5. The brief's measurement does not reproduce at this head

The brief reports 1,988 of 31,066 state-surface specs carrying
`ledger_filter_us:statutes/26/62#adjusted_gross_income_{lower,upper}_bound`
and `ledger_filter_us.tax.earned_income_credit_qualifying_children_lower_bound`.
The spec counts match exactly (31,066 full, 760 totals); the unsupported count
does not — it is 0 on every surface. What I measured:

1. The pinned feed's facts carry 22 distinct `dimensions` keys
   (`income_range` 35,570, `filing_status` 35,064, `eitc_child_count` 4,248,
   `agi_lower_usd`/`agi_upper_usd` 2,232 each, …). None contains `#` or a
   concept URI. The string `adjusted_gross_income_lower_bound` appears zero
   times in the 164 MB file. The concept vocabulary appears only in
   `universe_constraints[].variable`, `dimension_labels` and
   `layout.groupby_dimension`.
2. `ledger_targets.py:3257` is the only `ledger_filter_*` producer in the tree
   and it stamps `_dimensions(fact)` alone. Nothing synthesises a
   `*_lower_bound` filter key from a constraint row.
3. The compiled registry carries 13 distinct `ledger_filter_*` keys over all
   32,866 targets, all supported, identity, or always noop-valued.

The same holds for the other labelled export on this machine
(`inputs/chronicle_us_b571381/artifact/consumer_facts.jsonl`, same 39,158
facts, same 22 dimension keys).

So the rule lands as a forward guard on the *next* labelled export, not as an
unblock of a current stop. It is still the right shape: the day a Chronicle
export promotes those constraint rows into dimensions, the guard will refuse
them, and this rule is what lets the equal ones through without ever letting a
disagreeing one pass.

One adjacent thing worth noting: 2,356 facts already carry `agi_lower_usd`
and/or `agi_upper_usd` *dimensions* that restate their AGI constraint exactly
— 2,232 carry both edges and 124 top-band facts carry only the lower edge, and
every one of the 2,356 agrees with its own constraint rows. Those facts belong to
`irs_soi.table_1_4.total_wages.v1` and `irs_soi.table_1_1.v1` and do not
compile into targets in this registry, so `ledger_filter_agi_lower_usd` never
reaches the guard today. If a future compile does pick them up, they are the
same pattern in the legacy vocabulary.

## 6. Questions for Max

1. **The premise.** The 1,988-spec refusal does not reproduce at
   `us-chronicle-feed-repin` + this WIP with either feed here. Options:
   (a) merge the guard as a forward-compatibility rule, as written;
   (b) hold it until the labelled export that produces those keys exists, so
   the rule can be tested against real keys rather than invented ones;
   (c) point me at the exact feed/checkout the 2026-09-21 pilot used and I
   will re-measure against it.
2. **`agi_lower_usd` / `agi_upper_usd`.** Add them to
   `RESTATED_LEDGER_FILTER_CONCEPTS` now, under the identical agreement rule?
   Options: (a) yes — same pattern, legacy vocabulary, cheap, closes the hole
   before the facts compile; (b) no — no target carries them today, so adding
   them names a constraint nothing applies; (c) add with a test that fails the
   day such a target appears.
3. **Full mode.** 31,066 admin specs is 7.8× the historical 3,972, and a dense
   float32 matrix over 1,588,854 households goes from 23.5 GiB to 183.9 GiB.
   `soi_mode="totals"` is 760 specs and 4.5 GiB. I changed no default.
   Options: (a) keep `full` and size the box for a 184 GiB matrix or a chunked
   / sparse representation; (b) make `totals` the ACS local default and treat
   the AGI-band distribution as an opt-in; (c) select a middle surface (e.g.
   drop `soi_fiscal_distribution` below some state/band granularity) — say the
   rule and I will measure it.
4. **EITC upper bounds.** No feed carries one; I read `_upper_bound` as
   exclusive to match `_agi_bounds`, which folds both `<` and `<=` into one
   edge the materializer applies as `<`. If the Ledger means `<=` for count
   dimensions, one line changes and one test flips.

### Rulings (Max, 2026-09-22)

Max ruled on all four questions on 2026-09-22. The questions above are kept as
asked; these are the answers, and what this branch did about each.

1. **The premise: merge.** Answer (a), with the premise no longer in doubt.
   Section 7 found the refusal on `origin/main`, and section 8's committed
   receipt regenerates it on this tree: on the state surface with
   `soi_mode="full"`, 1,988 specs are refused with the rule reverted (which is
   `main`'s behaviour) and 0 with it. The rule merges as the fix for a
   refusal `main` makes today, not as forward-compatibility scaffolding.
2. **`agi_lower_usd` / `agi_upper_usd`: do not add them.** Answer (b). No
   concept is added for them. Nothing slips through meanwhile: a concept not
   in `RESTATED_LEDGER_FILTER_CONCEPTS` comes back from
   `_restated_ledger_filter_refusal` as its bare key
   (`tools/build_us_fiscal_refresh_release.py:4634-4635`), so a
   `ledger_filter_agi_lower_usd` that reaches the guard is refused by name and
   `_assert_supported_ledger_filter_metadata` raises (`:4675-4687`). The new
   `test_legacy_agi_usd_dimensions_stay_refused_by_bare_key` pins that at both
   call sites, including where the legacy value equals the compiled edge.
3. **Full mode: `totals` becomes the ACS local default, and the AGI-band
   distribution becomes an opt-in.** Answer (b). That change is made in a
   separate PR, not here; this branch still changes no default or selection.
4. **EITC upper bounds: refuse them until the Ledger confirms its operator.**
   Neither reading is adopted. Done on this branch, in the commit that adds
   this block: `_restated_eitc_child_count_refusal` now refuses any key on a
   side in `RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES` (`{"upper"}`) before it
   reads the compiled filter or parses the value, with
   `<key>=<value> restates a qualifying-child upper bound, refused outright:
   the Ledger's operator for it (< or <=) is unconfirmed, and the two readings
   select different returns`. Lower and exact restatements stay under the
   agreement rule. The comparison raises if any other side reaches it, so
   taking `"upper"` out of the set cannot quietly fall back to an exact-count
   comparison. `test_restated_eitc_child_upper_bounds_are_refused_outright`
   and `test_restated_eitc_child_comparison_never_guesses_an_upper_reading`
   are new; the upper-`1`-with-`0` case left the acceptance test.

   Why the key name alone does not settle the operator, read at this head:
   `_constraint_bound_filters` stamps a `<` constraint row as `_upper_bound`
   and a `<=` row as `_upper_bound_inclusive`
   (`packages/microcosm-build/src/microcosm/build/ledger_targets.py:3470-3478`),
   and `_upper_bound_inclusive` names no restated concept, so it is refused by
   bare key. But the dimension stamp runs first (`:3459-3461`) and wins through
   `setdefault` (`:3463-3464`), and a dimension of the same name carries no
   operator at all.

   **The receipt's counts do not change.** This is deduced from the committed
   receipt, not re-measured. In `restated_filter_arms_receipt.json`'s reverted
   arm, `_unsupported_ledger_filter_metadata` lists every otherwise-unsupported,
   non-noop `ledger_filter_*` key by bare name. On every surface of both feeds,
   the only qualifying-child key it lists is
   `…earned_income_credit_qualifying_children_lower_bound` (1,137 entries over
   the whole registry, 1,076 on the state surface); no `…_upper_bound` or
   `…_upper_bound_inclusive` key appears. Noop values are skipped before the
   rule in both arms, and the new refusal sits inside the rule. The compile
   those specs come from is in `packages/*/src`, which this change does not
   touch. So no spec in these compiles carries a key the new refusal can fire
   on, and every count in section 8's table stands. The full measurement was
   not re-run.
5. **The 939 age-band refusals** (section 7, "Still open") get a follow-up PR
   of their own. This PR adds no `age` concept.

Tests at the commit that adds this block (its parent is `235dc701c`):

```
$ MICROCOSM_US_CHRONICLE_FACTS=<consumer_facts_us_c5e5bf8.jsonl> \
    .venv/bin/python -m pytest packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py -rs
248 passed in 80.37s, no skips, exit 0   # the pinned-feed state-surface arm ran

$ .venv/bin/python tools/ci_test_groups.py --verify          verification=ok, exit 0
$ ruff check .                                              All checks passed!
$ ruff format --check <the two changed .py files>           2 files already formatted
```

The other modules that import `tools/build_us_fiscal_refresh_release.py` were
not re-run for this change: it touches only `_restated_eitc_child_count_refusal`
and its constants. That function is called only from
`_restated_ledger_filter_refusal` (`tools/build_us_fiscal_refresh_release.py:4639`),
which outside the tests is called only from the two guards (`:4424`, `:4665`)
and swapped out by `measure_restated_filter_arms.py` (grep over `tools`,
`packages`, `experiments`, 2026-09-22).

## 7. 2026-09-22: re-levelled on `origin/main` — the premise does reproduce

Sections 1–6 above were written at `369dedf1f`, before this branch was
re-levelled. **Section 5 is superseded: the refusal is real.** What changed is
the tree, not the feed.

### Why the first measurement found nothing

`origin/main` gained
`microcosm.build.ledger_targets._constraint_bound_filters` after this stack's
merge base `8c44daa52` (the `parameter_gated_threshold` filter-aware work in
the UK CGT lane, changelog `725-filter-aware-gated-provider.changed.md`). It
runs inside `_ledger_metadata` right after the dimension stamp:

```python
for key, value in sorted(_constraint_bound_filters(fact, dimensions).items()):
    metadata.setdefault(f"ledger_filter_{key}", value)
```

and turns every `role: filter` universe constraint whose operator is one of
`>= > < <=`, whose value is a finite number, and whose variable is **not**
already a dimension, into `ledger_filter_<variable>_lower_bound` /
`_lower_bound_exclusive` / `_upper_bound` / `_upper_bound_inclusive`.

Section 1's reading — "`ledger_targets.py:3257` is the **only** producer of
`ledger_filter_*` metadata and it stamps `_dimensions(fact)` alone; it never
reads `universe_constraints`" — was correct for the tree it was read in, and
is exactly what stopped being true. Section 5's observation that the concept
vocabulary "appears only in `universe_constraints[].variable`" was the
mechanism, seen one commit too early: those constraint rows are now filter
keys.

### The three counts, same feed, same surface

`tools/build_us_acs_local_release.py::state_admin_specs(feed,
["snap","medicaid","soi"], soi_mode="full")` over
`consumer_facts_us_c5e5bf8.jsonl`; 31,066 state specs in every run.

| Tree | `_unsupported_ledger_filter_metadata` | `_unsupported_soi_ledger_filters` |
|---|---:|---:|
| `us-chronicle-feed-repin` merged with `origin/main` (no rule) | **1,988** | 1,988 |
| this head (rule at both call sites) | **0** | 0 |
| this head, both call sites reverted | **1,988** | 1,988 |

The 1,988 refused specs carry 2,702 refusal entries:
`…earned_income_credit_qualifying_children_lower_bound` 1,076,
`…adjusted_gross_income_lower_bound` 816,
`…adjusted_gross_income_upper_bound` 810 — the three keys section 2's rule was
written for, in the vocabulary it was written for.

So the rule is not forward-compatibility scaffolding. It is what keeps the ACS
local state surface compiling once this branch sits on `main`, and the
answer to question 6.1 is (a) with the premise now evidenced.

### Still open: `ledger_filter_age_{lower,upper}_bound`, 939 targets

`_constraint_bound_filters` also stamps the **bare** vocabulary. Over the whole
compiled registry (32,866 targets, not just the state surface) this head still
refuses 939:

| Key | Targets carrying it |
|---|---:|
| `ledger_filter_age_lower_bound` | 939 |
| `ledger_filter_age_upper_bound` | 886 |

936 are `census_population` / `population_age` and 3 are
`ssa` / `ssa_ssi_age_band_recipients`, and every one restates the
`age_lower_bound` / `age_upper_bound` the materializer already slices on in
`_population_age_household_values`. It is section 2's pattern exactly, one
concept further out, and it is **not** this branch's regression. Measured
against `origin/main` at `18c6d39a0` with no part of this stack applied
(main's own `packages/*/src` on `PYTHONPATH`, `ledger_targets.__file__`
asserted to be under that checkout), same feed,
`compile_us_fiscal_target_registry(..., age_targets=True)`:

| Tree | Compiled targets | Refused |
|---|---:|---:|
| `origin/main` `18c6d39a0` alone | 32,866 | **3,208** |
| this head | 32,866 | **939** |

Main's 3,208 split `irs_soi|soi_fiscal_distribution` 2,269,
`census_population|population_age` 936, `ssa|ssa_ssi_age_band_recipients` 3.
This branch clears the 2,269; the 939 age-band refusals are main's and survive
it. The ACS local state surface never selects `census_population` or `ssa`,
which is why the state-surface table above reads 0.

Not fixed here, deliberately: adding `age` to
`RESTATED_LEDGER_FILTER_CONCEPTS` under a third `age_band` rule (compare
`ledger_filter_age_lower_bound` against `age_lower_bound` and
`ledger_filter_age_upper_bound` against `age_upper_bound`, both through
`_as_bound`) is a one-rule change of the same shape, but it widens what this
PR accepts beyond the surface it was reviewed against. It wants its own
decision.

### What the committed fixture now is

`tests/fixtures/us_compiled_ledger_filter_specs.json` was captured at
`369dedf1f`, so its key census predates `_constraint_bound_filters` and lists
none of the restated keys. It is kept as-is: it still pins the compile the
supported/identity classification was reviewed against, and the fixture
`description` and the two tests reading it now say so.
`test_pinned_chronicle_feed_state_surface_compiles_no_unsupported_filters` is
the arm that meets the restated keys — before the re-level it passed over a
feed carrying none, and now it passes over 31,066 real specs of which 1,988
carry one.

## 8. 2026-09-22: re-levelled on `main` after #955 merged

#955 merged into `main` at `6710d0f5f`. `origin/main` was merged into this
branch as `b10e117a0` with no conflicts
(`tools/build_us_fiscal_refresh_release.py` and
`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py` changed on
both sides and auto-merged), and #969 was retargeted to `main`. Against
`origin/main` the branch then differed in six files (eight after the
measurement script and its receipt below were added, and still eight after
`origin/main` `2b85b7b22` was merged in as `b6e0b04cd`): this report, the census
script, the fixture, the changelog fragment, the test module, and
`tools/build_us_fiscal_refresh_release.py`, where the difference is the rule,
its constants, one docstring and the two call sites. No `packages/*/src` file
differs from `main`, so the compile measured below is `main`'s.

### The counts on this tree

Every count in this subsection is in
`experiments/us-labelled-filter-support/restated_filter_arms_receipt.json`,
written by the committed
`experiments/us-labelled-filter-support/measure_restated_filter_arms.py`
(the receipt's `script_sha256`, `00d09505…`, is the committed script's; it ran
at `fb1a09646`, which is `b10e117a0` plus a `REPORT.md` edit, so the code it
measured is `b10e117a0`'s; one process for both feeds, 3.0 GB peak RSS per
the receipt's `max_rss_bytes`; the receipt records compile-and-select time
per feed, 39.2 s and 41.2 s, but no total wall time — the 113 s total was
read off the terminal and is not recorded). Regenerate with:

```
uv run python experiments/us-labelled-filter-support/measure_restated_filter_arms.py \
    <consumer_facts_us_c5e5bf8.jsonl> <chronicle_us_b571381/artifact/consumer_facts.jsonl> \
    > experiments/us-labelled-filter-support/restated_filter_arms_receipt.json
```

The script uses the same functions as section 7:
`compile_us_fiscal_target_registry(..., age_targets=True)` for the whole
registry, `state_admin_specs(feed, ["snap","medicaid","soi"], soi_mode=...)`
for the state surface, refusals from `_unsupported_ledger_filter_metadata`,
silent skips from `_unsupported_soi_ledger_filters` over `irs_soi` specs. The
reverted arm replaces `_restated_ledger_filter_refusal` with a function that
returns the bare key. That is the pre-rule behaviour of both call sites, and
so `main`'s: on `main` (`2b85b7b22`) both guards list every
otherwise-unsupported, non-noop `ledger_filter_*` key by name, with no
restatement check (`tools/build_us_fiscal_refresh_release.py` on `main`,
`_unsupported_soi_ledger_filters` and `_unsupported_ledger_filter_metadata`,
read 2026-09-22).

Both US exports on this machine gave identical counts: the pinned
`consumer_facts_us_c5e5bf8.jsonl` (receipt `feed_sha256` `b8543739…`, equal to
the `facts_sha256` in `packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json`)
and `chronicle_us_b571381/artifact/consumer_facts.jsonl` (`4d1dba8c…`).

| Surface | Targets | Refused, rule | SOI skips, rule | Refused, reverted (`main`) | SOI skips, reverted |
|---|---:|---:|---:|---:|---:|
| Whole compiled registry | 32,866 | 939 | 0 | 3,208 | 2,269 |
| State surface, `soi_mode="full"` | 31,066 | **0** | 0 | **1,988** | 1,988 |
| State surface, `soi_mode="totals"` | 760 | 0 | 0 | 0 | 0 |

With the rule reverted, the 1,988 state-surface refusals carry 2,702 entries
(`…earned_income_credit_qualifying_children_lower_bound` 1,076,
`…adjusted_gross_income_lower_bound` 816,
`…adjusted_gross_income_upper_bound` 810), all on
`irs_soi|soi_fiscal_distribution`. The whole registry's 3,208 split
`irs_soi|soi_fiscal_distribution` 2,269, `census_population|population_age`
936 and `ssa|ssa_ssi_age_band_recipients` 3. With the rule, the 939 left are
section 7's age-band restatements (`ledger_filter_age_lower_bound` on 939
targets, `ledger_filter_age_upper_bound` on 886).

Every number equals section 7's, which stands unchanged. The only new
observation is that the `b571381` export gives the pinned feed's counts on this
tree too; section 5 had compared the two exports only on the pre-merge tree.

### Provenance of the numbers in sections 3b–7

Sections 3b–7 record runs made before this merge, at the heads they name
(section 7: this branch re-levelled on `origin/main` before #955 merged), with
`census_compiled_ledger_filters.py`, the test module and ad hoc calls to the
same guard functions. Their outputs were not committed, so those sections are
records of those runs rather than numbers regenerable from files on this
branch. What the receipt above does regenerate on this tree: the target counts
(32,866 / 31,066 / 760), section 7's state-surface counts with the rule (0)
and reverted (1,988, carrying 2,702 entries split 1,076 / 816 / 810), and
section 7's age-band entries (939 / 886). Section 4's whole-registry
"Refused 0" is superseded by this tree's 939 (rule) and 3,208 (reverted).
Section 4's family-by-role table and section 5's fact-level counts are not in
the receipt and were not re-measured after the merge. Section 4's byte table
is arithmetic (4 bytes × 1,588,854 households × specs), not a measurement; its
historical rows cite `_buildp-runtime/logs/acs-local/release_chain.log`.

### Tests

Run at `b6e0b04cd`, the merge of `origin/main` `2b85b7b22` (#976, which
touches only `reform_validation.py`, its test module and a changelog
fragment) into this branch; no conflicts.

```
$ MICROCOSM_US_CHRONICLE_FACTS=<consumer_facts_us_c5e5bf8.jsonl> \
    .venv/bin/python -m pytest packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py
245 dots, no skips, exit 0     # the pinned-feed state-surface arm ran

$ .venv/bin/python -m pytest <the 29 other test modules that import
    tools/build_us_fiscal_refresh_release.py or ledger_targets>
11 failed, 1405 passed, 13 skipped, 4 errors in 1681.84s
```

Every one of the 15 failures and errors was in
`test_us_multispine_pool_tool.py` (a module this branch does not change) and
every traceback ends in `OSError: [Errno 28] No space left on device` or
pytest failing to create its temp directory: the machine's disk filled during
the run. Re-running those tests with the disk clear passed:
`-k` over the seven affected test functions, 22 passed, exit 0.

```
$ .venv/bin/python tools/ci_test_groups.py --verify     verification=ok, exit 0
$ ruff check / ruff format --check on the four changed .py files
All checks passed! / 4 files already formatted
```

## 9. 2026-09-22: the age-band follow-up (Max's ruling 2)

Section 7's "still open" item is closed on branch
`us-restated-age-band-filters`, stacked on this branch at `01dc806f4`. It
adds `age` to `RESTATED_LEDGER_FILTER_CONCEPTS` under a third rule,
`age_band`, in `tools/build_us_fiscal_refresh_release.py`.

**What the materializer applies, read at `01dc806f4`.** Both age paths in
`_materialize_target_frame` read `age_lower_bound` / `age_upper_bound`
through `_as_bound` into `lower <= age < upper`: `population_age` in
`_population_age_household_values`, and the `policyengine_variable` branch
entered whenever either key is present (the SSA SSI by-age counts). The
compiled bounds come from `us_runtime.fiscal_targets._age_bounds`, which
takes the value of any `>`/`>=` row as the lower edge and any `<`/`<=` row as
the upper edge, **dropping the operator**. The restated key keeps it:
`ledger_targets._constraint_bound_filters` stamps `age_lower_bound` only for
`>=` and `age_upper_bound` only for `<`, and `>`/`<=` as
`_lower_bound_exclusive` / `_upper_bound_inclusive`.

**The rule.** A restated `age_{lower,upper}_bound` is accepted only when it
parses to the same number as the compiled edge on its own side and the spec's
materializer is `population_age` or `policyengine_variable`. Refused, naming
the values: a disagreeing edge, no compiled counterpart, an exact-age key
(`ledger_filter_age=<v>`), any other materializer, and a spec carrying
`ledger_filter_age` at all (age is then a dimension, so
`_constraint_bound_filters` stamped none of the age rows and the bound came
from the operator-less dimension stamp). `_exclusive` / `_inclusive` keys
are not restated concepts and keep the bare-key refusal whatever their value.
A test that runs real facts through `_dynamic_us_fiscal_target_references`
and `compile_ledger_target_references` shows why that matters: an
`age <= 9` row compiles to `age_upper_bound=9`, which the materializer reads
as `age < 9`. The AGI and age rules share one edge comparison,
`_restated_band_edge_refusal`.

**The pinned feeds, measured.** Both exports carry 1,493 facts with an age
constraint. Every row is `role: filter`, an integer value, `>=` or `<`, at
most one per side, and no fact has an age-like dimension key. Counted by a
throwaway scan of the two JSONL files, not committed.

**Counts.** From `age_band_rule_receipt.json`, written by the committed
`measure_age_band_rule.py` (receipt `script_sha256` `9a7a77ac…`). It ran at
`b43aec644` with only the receipt untracked, took 35.4 s and 32.6 s to
compile, and peaked at 2.9 GB RSS. The whole compiled registry
(`compile_us_fiscal_target_registry(..., age_targets=True)`) has 32,866
targets on each feed:

| Arm | Refused targets | Refusal entries | SOI silent skips |
|---|---:|---:|---:|
| `rule` (this follow-up) | **0** | 0 | 0 |
| `without_age_band` (#969 alone) | **939** | 1,825 | 0 |
| `reverted` (`main`'s behaviour) | 3,208 | 5,116 | 2,269 |

The two feeds (`b8543739…` and `4d1dba8c…`) give identical counts. The 939
are the targets carrying a restated age key: 936 `census_population` /
`population_age` / `population_age` and 3 `ssa` /
`ssa_ssi_age_band_recipients` / `policyengine_variable`. The only
`ledger_filter_age*` keys present are `_lower_bound` (939) and
`_upper_bound` (886). The `without_age_band` and `reverted` rows equal the
section 8 receipt's `rule` and `reverted` rows.
