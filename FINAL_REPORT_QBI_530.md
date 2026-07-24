# Final report: QBI v2 engine scaffold (populace #530)

## Outcome

All five v2 deliverables landed on `qbi-v2-engine`, based on
`qbi-port-530` commit `d1a6428`. The work was completed offline, with local
commits only.

Version 1 remains the production default. Version 2 is selectable in the base
builder but intentionally fails closed while its packaged SSTB crosswalk has
`status: "placeholder"`.

## What landed

### Strict v2 assumptions and resources

- `us/qbi_assumptions_v2.json` adds a strict schema for:
  - five distinct PCG64 family seeds: qualification `2041`, SSTB `2064`,
    W-2 `2042`, UBIA `2044`, and investment `2043`;
  - per-source `derived` or `prior` qualification modes, parameters, and
    nonempty rationales;
  - crosswalk-driven SSTB classification, host columns, an ambiguous-code
    prior, and exhaustive lower-inclusive/upper-exclusive AGI prior bands;
  - the unchanged v1 W-2 and UBIA model structures and parameters.
- The parser rejects unknown keys and modes, malformed bands, negative or
  reused family seeds, invalid probabilities, and incomplete nested blocks.
- `us/sstb_crosswalk_placeholder.json` is declared in
  `country_package.json`, has placeholder status and an empty mapping, and
  cannot run v2.
- Ready crosswalks are strictly parsed. Caller-constructed crosswalk objects
  are revalidated too, so they cannot bypass the fail-closed category checks.

### Qualification derivations and independent streams

The packaged v2 rules are:

- `self_employment_income`: derived as source != 0;
- `farm_operations_income`: derived as source != 0;
- `partnership_s_corp_income`: derived as source != 0;
- `farm_rent_income`: residual prior `0.8`;
- `rental_income`: residual prior `0.7`;
- `estate_income`: residual prior `0.6`.

Derived sources consume no qualification RNG. Prior sources retain seeded
draws. W-2, UBIA, investment, and post-QRF SSTB each construct their own
generator from their own seed.

The donor has no host occupation or industry, so v2 emits a neutral preliminary
SSTB route. The authoritative classification runs only after QRF placement on
the host record.

### Host-conditioned SSTB reconciliation

`with_host_sstb_classification(...)` is a pure post-QRF transform. It:

- derives the law-determined flags and preserves QRF-placed residual-prior
  flags;
- classifies positive combined Schedule C income from industry when configured,
  otherwise from occupation;
- maps `clear_sstb` to true, `non_sstb` to false, and `ambiguous` or unmapped
  codes through the seeded ambiguous prior;
- applies the AGI-band prior to qualified partnership/S-corporation or estate
  income when there is no positive qualified Schedule C route;
- gives an observed Schedule C route precedence over the passive prior;
- preserves total signed Schedule C income and the base W-2/UBIA pools;
- reroutes the SSTB Schedule C, W-2, and UBIA leaves;
- recreates mutually exclusive ordinary/SSTB qualification routes;
- applies the existing BDC and REIT/PTP exposure caps; and
- checks every invariant reported by `us_qbi_inputs_summary` before returning.

The v1 reconciliation function now shares the exact routing helper, preserving
its existing behavior while preventing v1/v2 identity drift.

### Version-gated QRF and builder seam

- V1 retains its locked target surface: 55 person targets plus 9 tax-unit
  targets.
- V2 retains 51 person targets plus 9 tax-unit targets. It excludes exactly:
  - `farm_operations_income_would_be_qualified`;
  - `partnership_s_corp_income_would_be_qualified`;
  - `self_employment_income_would_be_qualified`;
  - `sstb_self_employment_income_would_be_qualified`.
- The exclusions are declared in `source_stages.json` and checked against the
  runtime selector.
- Both monolithic and checkpointed base-builder paths use the same
  version-selected donor/QRF target tuple.
- `qbi_simulation_version` is part of child-stage CLI reconstruction,
  checkpoint run identity, donor construction, post-QRF dispatch, and build
  summaries.
- The CLI and runtime default remain version 1. Version 2 dispatches the
  existing `qbi_reconciliation` stage boundary to the host-conditioned
  transform.

## Census CPS industry-column finding

The frozen union of the `census_cps_2022`, `census_cps_2023`, and
`census_cps_2024` person columns is pinned in
`packages/populace-build/tests/test_us_plan.py`.

- `AGI` is declared at line 1011.
- Detailed occupation `PEIOOCC` is declared at line 1092.
- No detailed industry field is declared: neither `PEIOIND` nor `A_MJIND`
  occurs in the frozen set or the country source declarations.

Accordingly, v2 is occupation-first today:

```json
{
  "occupation_column": "PEIOOCC",
  "industry_column": null,
  "agi_column": "AGI"
}
```

The transform already implements and tests the future
industry-primary/occupation-fallback branch, but no undeclared industry column
is read.

## Stream-independence proof

The tests prove independence at the byte level:

1. Changing the partnership/S-corporation qualification mode from derived to
   an equivalent prior changes qualification-flag bytes but leaves W-2, UBIA,
   REIT/PTP, and BDC output bytes identical.
2. The equivalent mode change on a host frame leaves `business_is_sstb` and
   routed SSTB Schedule C, W-2, and UBIA bytes identical.
3. Changing only the W-2 seed changes W-2 output while UBIA remains
   byte-identical.
4. Changing only the UBIA seed changes UBIA output while W-2 remains
   byte-identical.

These checks live in:

- `test_v2_qualification_mode_change_preserves_other_family_bytes`;
- `test_host_sstb_stream_is_independent_of_qualification_mode`; and
- `test_v2_w2_and_ubia_family_seeds_are_independent`.

The untouched v1 15-output golden-stream test still passes, including the
restricted full-artifact replay.

## Exact remaining research and activation seams

V2 must not be enabled in production until these are resolved:

1. Add a reviewed, versioned ready crosswalk resource with Census occupation
   codes mapped to `clear_sstb`, `non_sstb`, or `ambiguous`; declare it in
   `country_package.json` and point `crosswalk_resource` to it.
2. Replace the placeholder `ambiguous_prior: 0.0` and
   `placeholder_pending_evidence` status with a reviewed empirical prior and
   provenance.
3. Replace all-zero passive pass-through AGI-band values and
   `placeholder_pending_published_priors` with published probabilities,
   reviewed band boundaries, and provenance.
4. Review or replace the carried residual qualification priors for farm rent,
   rental, and estate income when source evidence becomes available. Their
   current rationales explicitly identify the missing host facts.
5. If detailed industry is later frozen into the Census CPS input contract,
   declare the exact column and code vintage, set `industry_column`, and add
   the industry mapping/code system to the ready crosswalk. Industry will then
   become primary automatically, with occupation fallback.
6. Run a full v2 base build with the ready inputs, review incidence bands, and
   only then consider changing the production default from version 1.
7. Propagate the version selector into the downstream fiscal-refresh builder
   when that workflow is authorized to consume a v2 base; it currently retains
   the production v1 reconciliation path.

The v1 W-2 and UBIA blocks remain intentionally carried forward. Replacing
those models belongs to the separately planned v3 work, not this scaffold.

## Verification

Final checks were run from the dedicated worktree with:

```text
POPULACE_PUF_2024_H5=$HOME/ops/populace-qbi-port/assets/puf_2024.h5 \
  UV_CACHE_DIR=/private/tmp/populace-qbi-v2-uv-cache uv run pytest
```

Result: **3,246 passed, 132 skipped, 0 failed** in 124.63 seconds.

Also green:

- `ruff format` and `ruff check` on every touched Python file;
- JSON parsing for all touched country resources and manifests;
- `test_spec_only_country_packages.py`;
- `test_us_plan.py`;
- QBI assumptions, host-routing, QRF-chain, and builder suites; and
- `git diff --check`.

The first full run exposed two unrelated target-parity regeneration failures:
merged JCT references had outgrown the old v9.2 Ledger feed pin. A separate
commit refreshed the generator and generated parity artifacts to the locally
available v9.4 digest under the unchanged congressional-district-off regime.
That repair changes no QBI or calibration behavior, and the final full run is
green.
