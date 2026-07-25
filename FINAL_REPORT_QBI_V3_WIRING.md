# Final report: QBI v3 evidence wiring (populace #530)

## Outcome

QBI simulation version 3 is complete on local branch `qbi-v3-wiring` in
`.claude/worktrees/populace-wt-530`. The work remained offline: no network
access, push, or pull request was attempted.

Version 1 remains the production default. Versions 2 and 3 remain explicit
opt-ins. Version 3 preserves version 2's qualification, host SSTB, and
investment machinery while replacing the invented wage/capital distributions
with parameters built from the committed SCF and SOI evidence resources.

The simulation is a pure function of the persisted assumptions and seeded
inputs. The restricted replay and calibration solve occur only in the
versioned assumptions-build command.

## Persisted calibration

The builder uses net qualified QBI and full-artifact person weights. A positive
qualified `partnership_s_corp_income` component takes record-level precedence
and receives the required JCT-renormalized 17/53 partnership/S-corporation
split. Other positive-QBI records use the sole-proprietorship evidence proxy.

The replay's weighted positive-QBI form distribution is:

| Latent form | Weighted records | Share |
|---|---:|---:|
| Sole proprietorship | 31,822,380.35712497 | 0.7709150741123305 |
| Partnership | 2,333,832.7958239755 | 0.05653841298379155 |
| S corporation | 7,122,497.59708642 | 0.17254651290387787 |
| Total | 41,278,710.75003537 | 1.0 |

Using those replay form weights, the persisted S-corporation residual target is

```text
[0.842 * total - 0.95 * sole - 0.80 * partnership] / S corporation
= 0.37323240048397105
```

The solved per-form log-odds shifts are:

| Latent form | Zero-employee target | Solved shift |
|---|---:|---:|
| Sole proprietorship | 0.95 | -2.598762342032285 |
| Partnership | 0.80 | -2.3076956692827935 |
| S corporation | 0.37323240048397105 | -0.9906098787763655 |

The persisted expected shares equal their targets to the strict resource
tolerance, and their weighted overall share is exactly 0.842. The runtime only
reads these solved shifts; it does not recalibrate.

## Restricted replay diagnostics

The committed diagnostic verifies the exact restricted H5 filename, byte size,
and SHA-256 before replay:

```text
filename: puf_2024.h5
bytes:    241,045,964
sha256:   8182579ddfecaf5e5b872e2307b88f03e8e8def993171b648f701a19a847f37b
```

Realized employer-gate results are:

| Population | Target zero-employee share | Realized share | Difference |
|---|---:|---:|---:|
| Sole proprietorship | 0.95 | 0.9510065290364592 | +0.100653 pp |
| Partnership | 0.80 | 0.791442063158404 | -0.855794 pp |
| S corporation | 0.37323240048397105 | 0.3774158749014317 | +0.418347 pp |
| All positive-QBI records | 0.842 | 0.8430139401618544 | +0.101394 pp |

Every result is within the required absolute tolerance of two percentage
points.

### W-2 wages

The persisted SOI all-industry wage bills are:

| Form | Wage bill |
|---|---:|
| Sole proprietorship | $175,013,634,000 |
| Partnership | $1,308,541,653,000 |
| S corporation, including officer compensation | $1,514,251,813,000 |

The documented broad magnitude envelope is:

```text
lower = (0.28 * sole + 0.17 * partnership + 0.53 * S corporation) / 0.98
      = $1,095,927,917,775.5103

upper = sole + partnership + S corporation
      = $2,997,807,100,000
```

This is a plausibility envelope derived from the persisted SOI tables, not a
confidence interval or calibration target. The version-3 replay aggregate is
`$1,472,347,345,828.11`, inside the band.

The requested input-signal comparison is:

| Simulation | Weighted W-2 nonzero share |
|---|---:|
| Version 2 | 0.000995818675737822 (0.0995819%) |
| Version 3 | 0.020430544809711643 (2.0430545%) |
| Version 2 to 3 delta | 0.01943472613397382 (+1.943473 pp) |

The version-3 share remains inside the unchanged `qbi_inputs` plausibility gate
of `[0.001, 0.35]`; the gate was not edited or flagged.

Non-employer records receive exactly zero W-2 wages. Employer wages equal the
latent industry's SOI wage share times SCF-margin-implied receipts.
S-corporation wage shares include officer compensation and salaries.
Partnership shares include cost of labor and salaries but exclude guaranteed
payments. The v3 path does not add or modify a SECA input, so the existing
non-SECA treatment of S-corporation income remains unchanged.

### REIT/PTP anchor

Version 3 retains the exact version-2 investment family and seed. The
restricted replay's version-3 REIT/PTP array is byte-identical to version 2:

```text
weighted aggregate: $20,943,037,788.761116
published anchor:   $21,070,000,000
anchor factor:      0.993974266197
allowed band:       [0.3, 3.0]
```

## Evidence machinery

- Industry is never assigned as an observed PUF attribute. Each latent form
  draws once from a joint, receipts-weighted distribution over the finest
  classified, nonaggregate SOI rows, and the chosen component supplies both
  its wage share and UBIA intensity.
- Employer shape marginalizes the SCF industry-bin dimension into
  income-band-by-form probabilities and applies one persisted log-odds shift
  per form.
- Receipts use a separate seeded draw through each form's SCF empirical
  profit-margin inverse CDF. Linear interpolation joins q05, q25, q50, q75,
  and q95, with endpoint clamping.
- UBIA equals latent SOI intensity times receipts times a mean-one lognormal
  residual. The sole-proprietor flow-ratio evidence remains marked
  `proxy: true`.

The allowed UBIA residual is justified entirely from the persisted SOI
cross-industry dispersion. For each form:

```text
sigma = receipts-weighted SD(log industry intensity)
        / sqrt(receipts-weight effective industry count)
```

This leaves the latent industry mixture carrying the observed cross-industry
heterogeneity while adding only a modest residual around the selected point
value:

| Form | Log-intensity SD | Effective count | Sigma |
|---|---:|---:|---:|
| Sole proprietorship | 0.6163001042648324 | 23.53513031065801 | 0.12703808473210823 |
| Partnership | 1.232972002301773 | 10.925132136575865 | 0.3730266549334975 |
| S corporation | 0.7167581982380723 | 19.533677794780417 | 0.16217378734400031 |

The new independent PCG64 families are:

```text
entity_split:     3041
latent_industry:  3042
employer_gate:    3043
margin_quantile:  3044
ubia_dispersion:  3045
```

Version 3 retains version 2's qualification, SSTB, and investment seeds
`2041`, `2064`, and `2043`. Stream-independence tests mutate each new family
individually. Existing v1/v2 full-artifact goldens pass unchanged, and the
host SSTB result is byte-identical between v2 and v3.

## Resource reproduction

The build command is:

```text
uv run python tools/build_us_qbi_v3_assumptions.py \
  --puf-h5 "$POPULACE_PUF_2024_H5"
```

Rebuilding from the pinned H5 produced a byte-for-byte match with the committed
resource. The committed `qbi_assumptions_v3.json` SHA-256 is:

```text
22a190ab57015415c054aca77fd150af6fc4b6d535b846179b9f099a0a7e9bf2
```

The assumptions resource also pins and tests the current v2 assumptions, SCF
employer, and SOI wage/capital resource digests. It is declared in
`country_package.json`, contains only spec data, and passes the repository's
executable-key, entrypoint-shaped-string, and incumbent-reference guards.

## Validation

The required full-workspace command was:

```text
UV_CACHE_DIR=/private/tmp/populace-wt-530-uv-cache \
UV_OFFLINE=1 UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=.venv \
POPULACE_PUF_2024_H5=/Users/maxghenis/ops/populace-qbi-port/assets/puf_2024.h5 \
POPULACE_RAW_PUF_2015_CSV=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv \
uv run pytest
```

Result: **3,293 passed, 132 skipped, 0 failed** in 121.40 seconds.

Also green:

- full-workspace `uv run ruff check .`;
- `ruff format --check` on all nine changed Python files;
- `git diff --check`;
- focused QBI assumptions, evidence, engine, host, synthetic, and restricted
  replay tests;
- spec-only package, source-manifest, incumbent-reference, and
  entrypoint-heuristic contracts;
- byte-for-byte assumptions rebuild;
- offline no-isolation `populace-build` wheel build, with the new assumptions
  and both evidence JSON resources present in the wheel.

## Merge conflicts

The prescribed sibling merge produced one conflict, in the shared
`PROGRESS.md` journal. It was resolved by making the v3 wiring ledger current
while retaining both sibling histories. The adjacent country-package manifest
merged cleanly. No stash was created or used.

## Binding-design deviations

**None.**

The following implementation choices are explicitly persisted and tested but
are not deviations:

- the record-level legal-form precedence rule for a positive qualified
  partnership/S-corporation component;
- the replay-weighted S-corporation residual needed to satisfy the 84.2%
  overall target simultaneously with the binding 95%/80% form targets;
- exclusion of aggregate, unallocable, nonpositive-receipt, or incomplete SOI
  rows from the latent-industry mixture;
- form-level SCF margin curves with endpoint clamping;
- the allowed, SOI-derived modest UBIA dispersion described above;
- the broad SOI/JCT W-2 plausibility envelope;
- use of the JCT 2% trust share in the supported-form denominator and metadata,
  without inventing a fourth form that has no binding wage/capital table.

## Local handoff

The implementation is committed locally on `qbi-v3-wiring`. Nothing was
pushed; the supervisor owns the push and pull-request steps.
