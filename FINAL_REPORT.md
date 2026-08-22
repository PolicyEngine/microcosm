# Final report: microcosm #722 passive pass-through NIIT input

## Outcome and base

Completed on `passive-pass-through-722`, based on
`origin/main@9c20c1d2`.

I inspected the verified #530 chain through `qbi-v3-wiring` but did not stack
on it. Its tip diverges before the repository rename and still imports the
retired `popu` + `lace.*` namespace and uses the retired package layout.
Stacking it would have reintroduced stale names into the current
`microcosm.*` tree, so I ported only the required evidence, calibration, RNG,
and wiring concepts onto current main.

The result emits `passive_partnership_s_corp_income` as a person-level subset
of positive partnership plus S-corporation income. It does not add that subset
to gross income because the parent income is already present there.

## Evidence and SCF cells

The new provisional, spec-only resource
`us/qbi_passive_passthrough_v1.json` is declared in
`country_package.json`. It pools the five SCF 2022 implicates with weight
`X42001 / 5`, defines active businesses from `X3103/X3104/X3105`, and uses
the SCFP `NONACTBUS` value components for holdings without an active
management role. Bands use Schedule-E income `X5714`.

The prevalence denominator is households with an actively managed business.
All prevalence cells clear effective n = 30. Conditional-share cells use the
band estimate when holder effective n is at least 30; the three thin middle
bands use the pooled all-band holder cell (effective n 236.0; 1,180 pooled
records).

| X5714 band | Active effective n | Non-active holding prevalence | Holder effective n | Selected non-active value share | Share cell |
|---|---:|---:|---:|---:|---|
| Nonpositive | 454.4 | 3.3493% | 63.0 | 60.2080% | Exact |
| $0–$25k | 102.6 | 4.1324% | 13.8 | 41.0605% | Pooled fallback |
| $25k–$100k | 121.2 | 6.4533% | 11.4 | 41.0605% | Pooled fallback |
| $100k–$250k | 82.6 | 9.6543% | 17.2 | 41.0605% | Pooled fallback |
| $250k–$1m | 117.2 | 21.2269% | 35.2 | 14.3741% | Exact |
| Over $1m | 229.0 | 38.9806% | 95.4 | 32.9077% | Exact |

The administrative anchor is IRS Publication 4801, Form 8960, TY2023:

- Line 4a: $1,185,607,258,000 across 4,988,033 returns.
- Line 4b: −$1,076,350,273,000 across 4,038,235 returns.
- Reported line 4c: $109,256,984,000 across 2,260,296 returns.
- Reported line 4c / line 4a survival ratio: 9.215276%.

The published thousand-dollar rows differ by $1,000 when line 4a and line 4b
are recombined. The resource therefore preserves the independently reported
line 4c value rather than replacing it with derived arithmetic.

No passive split exists in the checked entity-side partnership tables
23pa01/04/06/10/23 or IRS Table 1.4. Form 8960 line 4 is the only
administrative level available on disk, and the resource says so directly.

## Assignment and calibration

I implemented a version-1 sibling stage before QBI reconciliation rather than
`qbi_simulation_version=4`. Current main consumes 15 archived QBI leaves and
does not contain the old opt-in simulator, so a sibling preserves that contract
without advancing an old stream. The stage order is Schedule-D completion,
passive assignment, then QBI reconciliation.

The pure assignment uses a new PCG64 seed family with entropy 4722 and separate
presence/share children. It draws full-length arrays before support masks. Its
Schedule-E proxy is:

`partnership_income + s_corp_income + rental_income + estate_income`.

The array API accepts a v3 latent entity form when available and uses it only
to route partnership/S-corporation eligibility. The current main frame has no
such latent field. SCF probabilities and shares remain band-only because SCF
active and non-active businesses are not entity-linked.

The assumptions-build step solved and persisted one shift; runtime performs no
calibration and never reads the restricted artifact:

| Calibration quantity | Result |
|---|---:|
| Lower bound | $0 |
| Upper bound | $109,256,984,000 |
| Provisional midpoint target | $54,628,492,000 |
| Persisted log-odds shift | −1.157105426398319 |
| Expected weighted aggregate | $54,628,491,999.999985 |
| Seed-0 replay aggregate | $55,021,131,518.061035 |
| Replay error versus target | +$392,639,518.061035 (+0.718745%) |
| Positive assigned rows | 6,207 |

The achieved replay is inside the required ±5% tolerance and inside the
documented $0–$109.257 billion bounds.

The pinned replay artifact is 241,045,964 bytes with SHA-256
`8182579ddfecaf5e5b872e2307b88f03e8e8def993171b648f701a19a847f37b`.
It contains 484,015 persons and 207,692 tax units; 83,820 rows have positive
partnership/S-corporation income, whose weighted aggregate is
$1,632,692,702,130.5078.

## Wiring and diagnostics

The column is wired through multispine, direct PUF-support, fiscal-refresh,
operator ownership, L0 export, release coverage, and checkpoint identities.
Existing assigned values survive fiscal rebuilds; legacy inputs missing the
column are assigned once. The 997-row remaining-stage manifest now has SHA-256
`4ec692e3262f396ebacd6144c900b31ef2ae3eabc1062f38e6db6d3ab6f433fa`.

The #535 standing surface now contains two distinct, provisional,
diagnostics-only, non-gating rows with per-row error containment:

- `net_investment_income` versus Form 8960 line 12:
  $1,197,238,417,000.
- `passive_partnership_s_corp_income` versus reported line 4c as an upper
  bound: $0–$109,256,984,000.

The locked PolicyEngine-US 1.764.6 predates PR #9306. An exact temporary
engine-registry exception permits the new emitted input while that lock is in
place and becomes a no-op once the engine recognizes it.

## Open questions and explicit limitations

- Form 8960 line 4c combines rental, royalty, partnership, S-corporation,
  trust, and other flow-through income. There is no source-backed rental versus
  passive-pass-through decomposition. The lower bound assumes all line 4c is
  rental/royalty; the upper bound assumes all is passive pass-through; the
  midpoint is deliberately provisional.
- The engine includes `rental_income` in NIIT in full, while Schedule-E
  rental is frequently passive. Rental handling is unchanged here. The overlap
  must be resolved before promoting the calibration target.
- The passive leaf is a subset of an existing gross-income parent, not a new
  gross-income leg. The change affects only NIIT source routing once the engine
  update lands.
- TY2023 administrative anchors are applied to the pinned 2024-shaped replay
  without a population-vintage backcast. Line 12 also differs from the modeled
  surface in deductions/modifications and estate/trust-return coverage.
- Exact historical v1/v2/v3 seeded simulator goldens could not be executed:
  that simulator and its tests are absent from current main, and stacking the
  stale branches would violate the rename decision above. The available
  replacement test proves the independent passive family does not call or
  advance old streams and preserves all 15 current archived QBI leaves
  byte-for-byte under multiple passive seeds. This is not a claim that the
  absent historical golden suite was run.
- Current main exposes no v3 latent entity form to the frame wrapper. The pure
  API's latent-form routing is covered synthetically; live assignment therefore
  uses the SCF Schedule-E-band cells.
- Until PolicyEngine-US #9306 is present in the lock, the emitted column cannot
  change live NIIT calculations. The diagnostic remains contained and
  non-gating under the old engine.

## Verification

All commands ran offline with the existing synced environment.

- Full workspace, exact final code tree: `6,394 passed, 73 skipped` in
  42m45s; exit code 0. Pytest reported 1,890 non-failing warnings.
- Restricted replay with the specified `POPULACE_PUF_2024_H5` artifact:
  `12 passed`.
- Evidence, passive assignment, and reform diagnostics focused set:
  `64 passed, 1 expected gated skip`.
- Spec-only country-package, executable-entrypoint, and ordinary incumbent
  guard set: `51 passed`.
- Repository-wide `uv run ruff check .`: clean.
- All 26 changed Python files: Ruff format check clean.
- `git diff --check origin/main...HEAD`: clean.
- Manual incumbent package-name sweep over every changed Python, TOML,
  Markdown, and JSON file: no hits. This was run explicitly because the
  ordinary guard skips paths under `.claude/`.
- Deterministic resource regeneration, strict content hashes, checkpoint
  identities, release ownership/coverage, replay tolerance, and QBI
  byte-preservation contracts pass.

The towncrier fragment is `changelog.d/722-passive-pass-through.added.md`.
No network access, push, or pull request was used.

## Commits

- `ce08745a` — start the progress journal.
- `d1830f00` — record the sibling-stage design.
- `bb390fac` — add SCF/Form 8960 evidence.
- `5dc6a874` — add assignment, assumptions, calibration, and tests.
- `365c7428` — record calibrated replay progress.
- `3eb1c2e6` — add Form 8960 diagnostics.
- `d671b80e` — wire assignment through build surfaces.
- `218b6cb9` — clarify provisional contracts.
- `0842acef` and `3f6ecc73` — update stage-order/sparse fixtures.
- `cc7fe11f` — format branch-owned passive files.
