# Informed-L0 / warm-start selection — design (populace#328)

Reconstructs the informed-L0 / warm-start selection behind the certified US
default as **committed, spec-declared machinery**, closing the #159 discipline gap
(no uncommitted certified steps). Grounded in the Phase 1 forensics
(`informed-l0-warm-start-forensics.md`), which established that the frozen
57,240-record support is recoverable from the published H5's source identities and
joins 100% cleanly onto a freshly-rebuilt base.

## Problem restated

The live default is a 1,500-epoch dense polish (`method = "dense_no_l0"`) of a
**frozen 57,240-record support** warm-started from an ad-hoc run-dir `.npz`. The
selection that produced that support was never committed. Consequently:

- The **sparse** default cannot be reproduced from `main`: cold L0 on the rebuilt
  337,704 pool gives federal income tax **+19.9%**, loss 0.119, within-10% 50.2%
  (vs certified 0.044 / 94.7%).
- The **dense** parent overshoots the JCT mortgage tax-expenditure **+59.7%** —
  the same cold-vs-informed-selection gap.

Both are fixed by making the informed selection a first-class, reproducible build
input.

## Two modes

**(a) Frozen-support mode** — select *exactly* the named record set, then calibrate
weights on that fixed support. This is the certified pattern: reduce the candidate
frame to the frozen 57,240 support, then run the dense polish (or an L0-refit) on
it. Requires no change to `populace-calibrate`.

**(b) Informed-init mode** — initialize the L0 selection *probabilities* from the
named artifact's selection, then let L0 optimize. More robust to base drift (a
selected source record that no longer exists just starts cold instead of failing),
but requires a new per-record selection-probability-init parameter in
`populace-calibrate.calibrate_l0_refit` (which today exposes only a scalar
`init_mean`).

### Which first, and why

**Frozen-support mode ships first.** Phase 1 proved the join onto the current
reconstructed base is exact (57,240/57,240, 0 unmapped, 0 ambiguous), so there is
no base drift to tolerate today — mode (b)'s robustness buys nothing yet, and its
cost (a cross-package solver change) is real. Mode (a) reproduces the certified
artifact's construction faithfully and is fully testable now. Mode (b) is specified
here so the seam, spec schema, and manifest fields are forward-compatible, and is
implemented when a future base rebuild first produces a non-empty unmapped set.

## The seam

### Spec/CLI input (#159 discipline)

A release declares a **selection source** naming an artifact plus the **join
contract**. Two surfaces, same underlying object:

1. **CLI** on `tools/build_us_fiscal_refresh_release.py`:
   - `--selection-source-h5 PATH` — a populace US H5 whose record set defines the
     frozen support (e.g. the certified live default).
   - `--selection-join-key KEY` — the identity key; default
     `source_year,source_household_id,household_support_channel,household_support_clone_index`
     (the Phase-1-verified unique key). Named, not hard-coded, so the contract is
     auditable and overridable.
   - `--selection-mode {frozen_support,informed_init}` — default `frozen_support`.
   - `--selection-source-manifest PATH` (alternative to `--selection-source-h5`):
     a committed JSON listing the selected identities directly (see below), for a
     laptop reproduction that does not depend on downloading the 354 MB H5.

2. **Release spec field** (forward-looking, per #159): the same three values as a
   `selection_source` block the country release spec can carry, so a spec-compiled
   build passes them without ad-hoc flags. The CLI flags remain the authoritative
   runtime surface; the spec field is the declarative mirror.

The join key vocabulary is restricted to a **whitelist of stable identity
columns** (`source_year`, `source_household_id`, `source_person_id`,
`household_support_channel`, `household_support_clone_index`, `household_source_id`)
— never an assigned row id (`household_id`), which is order-dependent and would
silently misjoin across a rebuild.

### Selection-source manifest (committed, laptop-reproducible)

To make the sparse default reproducible from `main` on a laptop **without**
re-downloading the source H5 every time, the selection can be frozen once into a
small committed JSON:

```json
{
  "schema_version": 1,
  "join_key": ["source_year", "source_household_id",
               "household_support_channel", "household_support_clone_index"],
  "source": {"repo_id": "policyengine/populace-us",
             "revision": "populace-us-2024-sparse-l0-refit-57k-...-20260701",
             "path": "populace_us_2024.h5",
             "sha256": "c2065b64..."},
  "n_selected": 57240,
  "identities_sha256": "<sha256 of the canonical-sorted identity rows>",
  "identities": [[2024, 13, "asec", 0], ...]
}
```

`identities_sha256` binds the exact set; the `source` block records provenance back
to the HF artifact it was distilled from. A tool
(`tools/build_us_selection_source_manifest.py`) produces this from any source H5.
This manifest is the committed, spec-declared embodiment of the previously
uncommitted `a7a77c4` selection.

### Runtime support code (`us_runtime/`, never `build/us/`)

New module `packages/populace-build/src/populace/build/us_runtime/warm_start_selection.py`:

- `SelectionSource` — a frozen dataclass: the join key (validated against the
  whitelist), the identity rows, and provenance (source sha / repo / revision, or
  the source H5 path).
- `load_selection_source_from_h5(path, join_key)` — read the identity rows from a
  source H5's person + household tables, building the per-household key exactly as
  Phase 1 did (household-level components are constant within a household; the
  channel/clone come from the household table). Returns a `SelectionSource`.
- `load_selection_source_from_manifest(path)` — read + validate the committed JSON,
  including the `identities_sha256` integrity check.
- `select_frozen_support(base_frame, source) -> (Frame, SelectionReport)` — the
  core join. Builds the base's per-household key, intersects with the source's
  identity set, expands to a person mask, and calls `Frame.select`. Returns the
  reduced frame and a report (`n_source`, `n_base_candidates`, `n_selected`,
  `n_unmapped`, `n_ambiguous`, the join key, source provenance).

**Refusal contract (no fabricated selections):** `select_frozen_support` raises if
- the join key is not fully unique in either frame (would make selection
  order-dependent), or
- any source identity fails to map onto the base (`n_unmapped > 0`) in
  `frozen_support` mode — a silently-unmappable artifact must **fail loudly**, not
  select a smaller-than-declared or wrong support, or
- a source identity maps to more than one base row (`n_ambiguous > 0`).

Informed-init mode relaxes only the unmapped rule (unmapped source records are
dropped from the init set, recorded in the report, and the L0 selector starts those
records cold) — but still refuses on a non-unique key or ambiguous match.

### Where it slots into the build

Immediately after the base frame is loaded and **before** target materialization
(`_materialize_target_frame`), when a selection source is supplied and the mode is
`frozen_support`:

```
base_frame = _load_frame(base_h5)
if selection_source is not None and mode == "frozen_support":
    base_frame, selection_report = select_frozen_support(base_frame, selection_source)
# ... existing immigration/take-up input attach, then materialize on the reduced frame
target_frame, registry, compilation = _materialize_target_frame(base_frame, ...)
```

Reducing *before* materialization matches the certified run (`n_candidate = 57,240`,
`compiled_candidate_targets = 32,637`) and materializes PolicyEngine over 57k rows
instead of 337k. The existing calibration call then runs unchanged on the reduced
frame — dense (`--dense-default-dataset`, the certified polish) or L0-refit.

For informed-init mode the base frame is *not* reduced; instead the per-record init
probabilities are passed into `calibrate_l0_refit` (new parameter) so L0 optimizes
from the informed start over the full pool.

### Manifest provenance

A `selection_source` block is added to both `build_manifest.json` and
`release_manifest.json` (alongside the existing `warm_start_calibration` and
`default_dataset` blocks):

```json
"selection_source": {
  "mode": "frozen_support",
  "join_key": ["source_year", "source_household_id",
               "household_support_channel", "household_support_clone_index"],
  "source": {"kind": "h5"|"manifest", "path": "...", "sha256": "...",
             "repo_id": "...", "revision": "..."},
  "n_source": 57240, "n_base_candidates": 337704,
  "n_selected": 57240, "n_unmapped": 0, "n_ambiguous": 0
}
```

This is the recorded provenance #328 requires — the selection is reproducible from
`main` given the (committed manifest or sha-pinned HF) source, and the build
records exactly which source it descended from and how the join resolved.

### Relationship to the existing warm-start weights seam

`--warm-start-calibration-npz` (positional weight start) is **complementary and
unchanged**. After frozen-support reduction, the reduced frame's household order is
deterministic (base `household_id` ascending), so an aligned warm-start weight
vector for the *reduced* frame can still be supplied the existing way. The
certified pattern is: (1) frozen-support reduction → 57,240 frame, (2) optional
positional warm-start of those 57,240 weights, (3) dense polish. This design adds
step (1) as committed machinery; steps (2)–(3) already exist. The `solver_config`
identity omission (#326) is orthogonal and not blocking here.

## Testing (synthetic fixtures + refusal)

Unit tests in `packages/populace-build/tests/test_us_warm_start_selection.py`, all
on tiny synthetic frames (no network, no large H5):

1. **Join maps a clone-aware support** — a base with 2-year × 2-clone records; a
   source naming a subset; assert the reduced frame is exactly the named
   households, in the right order, with the source's identities.
2. **Order-independence** — shuffle the base row order; assert the same support is
   recovered (proves identity-join, not positional).
3. **Refuse silently-unmappable (frozen)** — a source identity absent from the base
   raises with a message naming the unmapped identity; assert **no** frame is
   returned (no fabricated / truncated selection).
4. **Refuse non-unique key** — a key missing the channel/clone component (so a
   source household's two clones collide) raises before selecting.
5. **Refuse ambiguous match** — a source identity mapping to >1 base row raises.
6. **Manifest round-trip + integrity** — write a selection-source manifest, reload,
   assert identities match; corrupt `identities_sha256` and assert it raises.
7. **Informed-init drops-and-records** — in `informed_init` mode an unmapped source
   record is dropped from the init set and appears in the report, without raising.

A build-level test asserts the manifest carries the `selection_source` block with
the resolved counts when a selection source is passed.

## Validation (Phase 4)

Because Phase 1 recovered a usable selection, run the sparse path against the Build
F base (`18833fb6…`) with the recovered frozen support and report income tax / SS /
loss / within-10% versus the certified reference (0.044 / 94.7%) and cold-L0
(0.119 / 50.2%). Detached compute (nohup + pidfile). If the run reproduces certified
quality, the sparse default is once again buildable from `main`.

## Out of scope / deferred

- Informed-init mode's calibrate-library parameter (implemented when a rebuild
  first yields `n_unmapped > 0`).
- `solver_config_sha256` (#326) — orthogonal identity hardening.
- The export mass-parity gate reference fix (#327) — separate one-line reference
  change; the mortgage residual it exposes is resolved by *this* work (the frozen
  support fits the mortgage tax-expenditure tighter than cold dense).
