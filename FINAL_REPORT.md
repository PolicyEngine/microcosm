# Final report: US release replacement scorecard

## Outcome

The replacement yardstick is complete and the live US incumbent is scored.
`tools/score_us_release_head_to_head.py` accepts the incumbent H5 and an
optional candidate H5 or authenticated pool, normalizes each through one
role-neutral loader API, and sends both through the same scoring function
(`tools/score_us_release_head_to_head.py:495-515,1398-1536,1670-1721`). Its
deterministic JSON contains every compiled fiscal target and every nominal
terminal-battery scalar leg; its Markdown twin is the readable scorecard.

The scorer compiles the sole US fiscal registry once, then materializes and
scores fixed registry chunks across fixed household slices
(`tools/score_us_release_head_to_head.py:518-579,675-852`). Every slice must
match the target, scale, diagnostic-name, and scored-column contracts; the two
artifacts must also have identical final scored-column contracts, so a missing
candidate measure cannot disappear silently
(`tools/score_us_release_head_to_head.py:858-938,1539-1553`).

No gate, threshold, tolerance, or band decides the replacement. When a
candidate is present, the output reports its weighted-loss delta, the exact
per-target balance of lower/equal/higher absolute relative errors, and each
side's battery evidence, then leaves the flip to the owner
(`tools/score_us_release_head_to_head.py:1571-1667`).

## Frozen incumbent and yardstick

The current PolicyEngine.py `5.0.3` bundle resolves the US default dataset as:

- repository: `policyengine/populace-us` (Hugging Face dataset);
- revision/build:
  `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`;
- filename: `populace_us_2024.h5`;
- resolved Hugging Face commit:
  `26dcad66867687f15735dc4926523e3741920836`;
- artifact SHA-256:
  `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`;
- PolicyEngine.py source commit:
  `cfdd128fc316e07ef54c182f2149fac217e8706f`, certified for
  `policyengine-us==1.764.6`.

That identity comes from PolicyEngine.py `5.0.3`'s bundle manifest
(`src/policyengine/data/bundle/manifest.json:113-140,156-160,181-189`), with
resolution at
`src/policyengine/provenance/manifest.py:180-187,270-299,301-318,540-560`,
Hugging Face handling at
`src/policyengine/provenance/dataset_sources.py:57-74,77-117`, and the US model
selection at `src/policyengine/tax_benefit_models/us/model.py:423-462`. The
charter's enhanced-CPS assumption is historical; it is not the dataset the
current package resolves. The scorer independently hash-matched the cached
bytes before attaching this identity
(`tools/score_us_release_head_to_head.py:120-138,399-405,448-492`).

The frozen yardstick is registry `c4ac617743f2`: 32,842 targets compiled from
Ledger facts SHA-256
`b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
The production loss weights use square-root target magnitude, semantic-concept
budgets, equal amount/count budgets, and final mean normalization; the
aggregate is the weighted mean of capped target-scaled absolute errors, with
no family multipliers
(`tools/build_us_fiscal_refresh_release.py:344-348,481-516,5781-5814,6214-6290`;
`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:471-537,576-600`).

## Incumbent evidence

The committed results are:

- `experiments/replacement_scorecard/incumbent_48b9d479.json` — complete
  32,842-row machine-readable evidence, SHA-256
  `b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`;
- `experiments/replacement_scorecard/incumbent_48b9d479.md` — human scorecard,
  SHA-256
  `3f9171b8f63fcef61518a4af1c18a8555c4f449ac62e9283e41ac2fe9c779021`.

The incumbent weighted loss is `0.11462448275649702`; its fraction of targets
within 10% is `0.2669143170330674`; all 57,240 household weights are nonzero.
The Markdown summary records these values and the exact identity at
`experiments/replacement_scorecard/incumbent_48b9d479.md:5-18`.

The terminal battery is entirely by-origin: its canonical surface is 131
single-column comparisons plus one joint comparison, normalized to 369 scalar
legs (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3011-3025,11644-11709,11824-11832,11948-12154`). The incumbent has 120,261
positive-weight clone-0 ASEC rows and zero ACS rows, so all 132 comparisons and
all 369 scalar legs are explicitly **inapplicable**. No zero, pass, or failure
was synthesized for the absent side; the human receipt is at
`experiments/replacement_scorecard/incumbent_48b9d479.md:97-104`.

A finished candidate H5 with both origins computes the same canonical formulas
while explicitly marking the production assembly/tail receipt unauthenticated
(`tools/score_us_release_head_to_head.py:1077-1358`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11474-11492,11530-11898`). A pool candidate instead consumes its authenticated
terminal receipt. The scoring-only pool loader may authenticate the exact
current `gate_failed`/`simulation_ready=false` publication pair without
weakening or reusing the separate production-ready loader
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:371-588,691-869`;
`tools/score_us_release_head_to_head.py:1361-1395`).

## Owner handoff

`_LANE-NOTES.md`, under “Owner command when the 25% candidate exists,” contains
the exact dense-pool and sparse-57k commands, including the required host-side
builder-process check and manifest hash pin. “Better than the incumbent” means
comparing each candidate view against the exact incumbent weighted loss and
the target-by-target lower/equal/higher error balance, while separately
inspecting every candidate battery leg that is computable. There is no invented
threshold or automatic conjunction; the owner decides whether the dense and
sparse evidence warrants the flip.

The candidate does not yet exist, so the incumbent-only JSON correctly leaves
`artifacts.candidate` and `comparison` null. This is the only external work
remaining.

## Verification

- Environment sync completed with
  `UV_CACHE_DIR=/tmp/microcosm-scorecard-uv-cache uv sync --all-packages --extra us`.
- Full workspace suite:
  `UV_CACHE_DIR=/tmp/microcosm-scorecard-uv-cache uv run python -m pytest` —
  **7,028 passed, 76 skipped, 0 failed** in 1:39:56. The `python -m` form avoids
  a copied virtualenv console-script shebang that pointed at a sibling
  worktree.
- Repository-wide `ruff check .`: green. All six Python files changed by this
  branch pass `ruff format --check`; the whole-tree format audit names 69
  pre-existing mainline files and was not used to rewrite unrelated code.
- `py_compile` for the scorer and `git diff --check`: green.
- Contract and loader-symmetry tests are at
  `packages/microcosm-build/tests/test_us_release_head_to_head_scorer.py:415-499`;
  deterministic fixture end-to-end coverage at `:502-569`; chunked/one-shot
  equivalence at `:575-678`; scalar-leg completeness at `:285-378`; failed-pool
  authentication at
  `packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1017-1085`;
  and finished-H5 canonical battery equivalence at
  `packages/microcosm-build/tests/test_us_stacked_spine.py:6929-6966`.
- The real incumbent run exited zero at 18.666 GiB peak RSS, below the binding
  20 GiB limit. Its five registry chunks used twelve household slices apiece;
  the scorer enforces the RSS ceiling after loading, after every chunk, and
  after releasing each side
  (`tools/score_us_release_head_to_head.py:332-352,653-852,1398-1463,1670-1715`).

The sandbox denied `ps`, `pgrep`, and `top`. Before scoring, the permitted
`lsof` process and open-file audits found no build-runtime process or open pool,
checkpoint, manifest, or build file. This lane started no pool build. It also
made no push and changed no gate, threshold, tolerance, or band.

## Commits

- `fd7d5515` — start the replacement-scorecard lane and committed progress log.
- `de5ca6aa` — document the yardstick audit.
- `256165e2` — correct the live incumbent identity.
- `90fe2364` — add the common head-to-head scorer.
- `758aa0c4` — preserve H5 snapshot identity across cache symlinks.
- `eebd1d6f` — make scoring chunked and memory-bounded.
- `8226f376` — complete artifact battery evidence and pool authentication.
- `34d93846` — merge the current `origin/main` before scoring.
- `e834baad` — record the merge and pre-score queue audit.
- `d876c971` — commit the live incumbent scorecard.
- `8d2ba34c` — document the exact candidate handoff.
- `6847d245` — keep the journal within the source-hygiene contract.
- Final progress, validation, and this report — the commit containing this file.

Nothing was pushed, and no pool build or publication was performed.
