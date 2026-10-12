# New Zealand take-up engine: the graph build

Working plan for [epic #343](https://github.com/PolicyEngine/microcosm/issues/343).
Companion inventory: [nz-calibration-targets.md](./nz-calibration-targets.md).
Rules: [rulespec-nz](https://github.com/TheAxiomFoundation/rulespec-nz) through
the Frame `RulesEngine` protocol and the Axiom adapter.

## Product definition

For each major New Zealand transfer, estimate **predicted entitlement dollars**
(rules × calibrated population, full take-up) against **actual dollars paid**
(administrative expenditure), and present the gap by income percentile and
family type. The first cut is the **Accommodation Supplement** (AS). Survey
evidence puts receipt among the potentially eligible at about 44% (MSD Income
Support Survey 2022, findings pack 7); an IDI-based estimate puts Working for
Families (WFF) take-up at about 87% in 2020 (McLeod and Wilson 2022, targets
inventory M5).

## Status: 7 October 2026

The spec-only [NZ country package](../packages/microcosm-build/src/microcosm/build/nz/country_package.json)
compiles through the shared country seam. It is a declared layout, not a
calibrated population or a release: every target reference is an unactivated
placeholder, the threshold-bearing gates read "awaiting D1", and no NZ graph
node exists yet. Donor records are US support records and are never presented
as New Zealand microdata.

The package replaces the stage-era scaffold of
[#821](https://github.com/PolicyEngine/microcosm/pull/821). It keeps #821's
donor pin, the region-layer geography contract, the release contract and the
gate set. It leaves out the WFF official-Budget transport stage and every pin
to rulespec-nz `3b663b3e` (rulespec-nz#117, unmerged); the #821 branch
`codex/nz-transport-microcosm` remains the salvage source for that WFF work.
The two region-geography kernel ids, `assign_nz_regions` and
`nz_region_geography_gate`, join the F0 contract-only vocabulary in
`spec_engine/resolver.py`; they close references and are never executable.

## How the build runs

The whole v0 build, from the donor file to the AS gap bands, is planned as
one `microcosm.graph.Graph` run with `run_graph` over a content store. None
of its nodes exists yet. Most of its kernels are still to be written; the
geography kernels `geography.assign_atomic@1`, `geography.derive@1` and
`geography.gate@1` already exist and are to be reused unchanged. The plan
has no `nz_runtime/` package: kernels are to be country-neutral
(`transport.*`, `concepts.*`, `geography.*`, `takeup.*`, `targets.*`,
`calibrate.*`, `gates.*`, `export.*`, `diagnostics.*`, `simulate.*`), and
the New Zealand specifics are to be data: the spec in `build/nz/` and the
Frame concept mapping `adapters/axiom_concept_mappings/nz.json`. In
package source (`packages/*/src`), the New Zealand-specific identifiers are
to be the two contract-only resolver ids above (`assign_nz_regions`,
`nz_region_geography_gate`) and the Axiom adapter's `NZ_SCHEMA` entity
schema, which package G1 is to add. Graph node ids (`nz.create`, …) also
name New Zealand.

Version layout:

```
nz.create (CREATE) ── nz.open (FILTER keep-all: transport, geography, receipt layer, targets)
                         ├── nz.calibrate (REWEIGHT; no members)
                         │      ├── nz.as (FILTER: AS inputs + reg 17/18 bridge)
                         │      │      └── nz.scn.S0 … nz.scn.S5 (FILTER each)
                         │      ├── nz.validate (FILTER: WFF tripwire, parked)
                         │      └── nz.terminal (FILTER: bands, diagnostics, gates, export)
                         └── nz.v.V1 … nz.v.V3 (FILTER each: variant → calibrate → AS → gap)
```

- **CREATE is to build the benefit units.** Entity ids and person
  memberships are structural columns, which the graph executor does not let
  a non-structural node own, so the NZ `family` benefit unit is to be built
  inside CREATE, with design weights installed at the Stats NZ population
  scale.
- **`nz.calibrate` has no members.** Scenarios, validation and terminal
  nodes hang off FILTER branches, so adding one never re-keys calibration.
- **Hold-out by ancestry.** No ancestor of a calibration node may read an AS
  output, the hold-out references, or the hold-out Chronicle artifact. The
  composer package (G6) is to add a test over the compiled graph's
  predecessors that checks this. Until then, `test_nz_spec_package.py`
  checks the hold-out only as data: no calibration or pre-calibration
  reference shares a hold-out reference's name, Ledger identifier or family,
  or has a selector that could match the same Ledger fact. Without the
  Ledger feed it cannot match an identifier on one side against a selector
  on the other.
- **Spec values enter node params.** Each node is to carry the resolved
  values it reads and a digest of where they came from; no kernel's
  implementation hash is to bind the whole NZ spec fingerprint, which would
  re-key every spec-reading node on any spec edit.

## Spec resources

| Resource | Holds | Read by (planned nodes) | Method card |
|---|---|---|---|
| `source_stages.json`, `spec/sources.yaml` | the populace-us Build P donor pin (revision, SHA-256, byte size) | `nz.create` | MC4 |
| `benefit_unit_rule.json` | adult + partner + dependent children; other adults form their own unit; refusals | `nz.create` | MC7 |
| `currency_bridge.json` | USD→NZD 1.654 (IRS 2024 yearly average), one multiplication, 2 dp | `nz.transport.currency` | MC4 |
| `precal_references.json` | quantile-map destinations (IRD wage bands, MBIE rents, Stats NZ financial net worth), TA population, MSD M1 and O4 for take-up draws | `nz.facts.precal` | MC4, MC6, MC8–MC10 |
| `as_area_crosswalk.json` | Work and Income area definitions; the TA → AS area population shares are a placeholder (`rows: []`, `status: requires_harvest`) until harvested | `nz.geo.support` | MC9 |
| `axiom_rules_bindings.json` | rulespec-nz pin, five module bindings, variables, the `2026-27` tax-year period | every Axiom-bound node | MC1 |
| `target_references.json` | the calibration set (MC5 list) | `nz.targets.compile` | MC5 |
| `as_rate_bridge.json` | reg 17 base rate and reg 18 cutout composition | `nz.as.bridge.*` | MC11 |
| `scenarios.json` | S0–S5, V1–V3, knobs, gap conventions (×52) | scenario and variant branches | MC3, MC12 |
| `holdout_references.json` | AS comparators and the WFF tripwire | gap and validation nodes only | MC2, MC5 |
| `gates.json` | the gate battery, evaluated with the shared transport bindings (`transport/gate_bindings.py`); thresholds await D1; support and release-input coverage remain applicable and block builds until their producers and evidence checks exist | `nz.gates.calibrated` | MC15 |
| `export_contract.json` | the closed export contract | export nodes | — |

The three reference sets are separate files on purpose. Each planned node is
to carry the digest of the resource it reads, so a hold-out edit cannot move
a calibration-side key, and a new calibration target does not re-run
transport.

### Reference sets

- **Calibration (MC5):** region × age × sex population, census household
  composition, census family type, tenure × region, IRD taxable-income bands
  (counts and totals), MSD main-benefit recipients, NZ Super and Veterans
  Pension recipients.
- **Pre-calibration:** IRD wage and salary bands, MBIE bond rents by TA, Stats
  NZ household financial net worth, TA population, and the same MSD recipient
  facts as the calibration set (the take-up draws calibrate to them).
- **Hold-out:** AS recipients by Work and Income region, MSD "Accommodation
  Assistance" expenditure 2024/25, Treasury AN 24/01's AS fiscal total, and
  the IRD WFF recipient-family counts and income distribution.

Three #821 references are outside the D1 default set and were not carried
over: national single-year age, student-loan borrowers, and census ethnicity
by age and region. D1 can add them back from the #821
branch.

### The reg 17/18 bridge

The AS module takes the reg 17 base rate and the reg 18 non-beneficiary
cutout as inputs. Until rulespec-nz encodes those regulations,
`as_rate_bridge.json` declares how the bridge nodes (package G7, not yet
written) are to compose them from engine evaluations of the bound
main-benefit and family-tax-credit modules, porting the statutory branch of
the IncomeExplorer conformance harness. Under method card amendment MC11a,
which is proposed (awaiting D1), this would be the one sanctioned exception
to "real Axiom runtime only"; it needs Max's MC11 ruling.

The sole-parent cutout cannot be a zero-solve on the unit's own Jobseeker
schedule. The engine applies Income Test 1 to a single person with dependent
children, while the harness pairs the single-with-children rate with the
Income Test 3 slope. The bridge declaration recovers the threshold, the slope
and the gross rate from three engine evaluations instead. Golden-08's base rate
(673.846923…) and cutout (905.028571…) are the differential expectations.

### What is still open

- **Crosswalk shares.** The Work and Income area definitions list 2017
  Statistical Area Units, not TAs. Shares need an area-unit to TA concordance
  and area-unit populations; the crosswalk ships with `rows: []` and
  `status: requires_harvest`, and the geography support must refuse it until
  rows exist.
- **Dependent-child rule.** rulespec-nz encodes no general dependent-child
  definition at the pin, so the age limit is spec data awaiting D1 (null
  until ruled).
- **Engine pin.** The hub pins the Axiom engine at M0 (MC1).
- **Thresholds.** MC15's per-family fit, effective-sample-size floor and
  weight-ratio threshold land with G8 after D1.
- **Superannuitants.** The harness has no NZ Super or Veteran's Pension
  base-rate branch; the central run excludes and counts superannuitant units
  unless reg 17 can be implemented from its text, and S5 carries the
  alternative.

## Validation

1. **Per case.** rulespec-nz against Treasury IncomeExplorer on the
   conformance cases; golden-08 is the AS differential.
2. **WFF tripwire.** The IRD WFF recipient-family income distribution (I3,
   2020 onward) is the one published conditional slice of the joint-income
   distribution. It stays held out, and its comparison is parked until the
   WFF modules are bound.
3. **Comparators.** The AS gap compares modelled full-entitlement dollars
   with MSD's Accommodation Assistance line (after its scope is confirmed)
   and AN 24/01. Treasury DistributionExplorer is TAWA-modelled; AS values
   assigned by TAWA's take-up model; an independent comparison, not full
   entitlement.

## Risks and honesty constraints

- Donor provenance: every output carries the US-support-stratum provenance;
  nothing is presented as New Zealand microdata.
- Joint income is imputed from donor structure; the WFF slice is the
  tripwire.
- Housing costs: MBIE bond rents may describe new tenancies rather than the
  sitting stock (scenario S4); boarders' costs and owners' rates are absent.
- Regional dollar splits assume a uniform average payment within a
  programme.
- legislation.govt.nz blocks scripted fetches; rules provenance rides on
  rulespec-nz's corpus pins, not live fetches.
