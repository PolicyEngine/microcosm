# Rare signed-tail battery lane notes

## 2026-08-21 — initialization

- Branch: `battery-rare-signed-tails`, starting at `2c7a7218` (`origin/main`).
- Binding scope: diagnose the 48 QED reds and the one-sided Keogh leg from the
  frozen arm-split adjudication; repair generating mechanisms only.
- Forbidden changes: gates, bands, ceilings, floors, folds, seeds, and owner-only
  exclusions/register entries.
- Build discipline: at most a 1% sample; check for a live
  `build_us_multispine_pool` process before every build; remain below 15 GiB RSS;
  run off-chain without `--logbook-prev-row-digest`; never touch
  `logbook-pending-chain.txt`.
- Environment receipt: `uv sync --all-packages --extra us` failed first on the
  sandboxed default UV cache, then on network DNS while fetching the locked
  pandas wheel from a writable cache. The primary-checkout venv lacks the build
  shard's `jsonschema`; the compatible `microcosm-f1` venv is used read-only for
  the full suite with lane-local sources and `UV_NO_SYNC=1`.
- GitNexus receipt: a fresh local index exists, but the CLI cannot query it
  because this worktree is absent from the read-only global registry. Direct
  source/call-site tracing and regression tests will substantiate every claim.
- No build has run and no chain-bearing file has been read or written as an
  input to a build.

## Evidence ledger

Mechanism classifications, realized-regime recomputations, code citations,
test receipts, build commands, RSS/queue receipts, and failure-line diffs will
be appended here as they are established.

### Frozen regime recomputation

- All 48 baseline-red QED checks use four receipted availability patterns. Each
  pattern at a given entity uses the same complete donor identity (person
  108,073; tax unit 57,630; SPM unit 43,961).
- Recomputing signs at `zero_atol=1e-6` gives 35
  `zero_inflated_positive` targets and 13 `three_sign` targets. There are no
  degenerate or single-sign QED targets, so a regime substitution cannot explain
  or honestly repair any of the 48 QED failures.
- The route partition is 17 early ASEC-to-ACS gap-fill QED checks and 31 late
  producer-complement checks (20 PUF-only, 10 ASEC-source-only, one overlapping
  producer target).
- Rare-tail actual-donor sign support is: collectibles +18 (late PUF; native
  ASEC +82), alimony +61 (early), casualty +27 (late PUF; native ASEC +79), farm
  operations -89 (early), and prior-year self-employment -48 (early). All five
  remain gated and sign-capable; the adjudication therefore still requires
  dense/additional evidence before a target-specific refit.

### Keogh one-sided leg

- Native ASEC clone 0 contains exactly two positive Keogh values: `2,040` and
  `30,000`. The late transfer is hard-wired to ASEC-origin clone 1, whose Keogh
  support is entirely zero, so its recomputed regime is `degenerate_zero`.
- The frozen bank contains 1,736,840 finite Keogh recipient draws and every draw
  is exactly zero. This proves loss before transfer fitting, not loss during the
  recipient merge.
- The retirement producer uniformly caps its 108,073-row ASEC training donor at
  5,000 rows and explicitly permits a rare output to remain zero when that draw
  contains no carrier. The smallest honest repair must preserve sign support in
  that capped training donor (with sampling-weight correction), rather than
  declare ACS absence or alter a terminal gate.

### Process visibility receipt

- Both `ps` and `pgrep -af build_us_multispine_pool` are denied by the managed
  macOS sandbox (`operation not permitted` / no process list). No build has been
  started: the binding pre-build queue check cannot yet be satisfied.
