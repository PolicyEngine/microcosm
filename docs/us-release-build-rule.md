# US release: main builds the certified default from raw sources

## The rule

Main must demonstrably build the certified US default from raw sources and pass
its preflight and certification gates (decided 15 September 2026).

The last from-scratch build is Build P (28 July 2026, policyengine-us 1.764.6).
Every later default is a supplied-parent enrichment of that file. A pull request
large enough to change how the dataset is built carries a from-scratch build and
certification receipt from its own tree before it merges. Main carries the same
build as a standing check afterwards.

A dataset earns default status separately, by beating the incumbent. The rule
main enforces is microcosm#578's: the candidate beats the incumbent on the
frozen comparison register, both rescored on it, with strict improvement
(`_exact_k_frozen_register_fit_gate`,
`tools/build_us_fiscal_refresh_release.py:6897`). That gate runs on the
`--exact-k` arm only (`:11149`), and that arm takes a pool manifest, which is
mutually exclusive with `--base-h5`. A `--base-h5` release runs no improvement
gate. Main has no US held-out test: the rotated-fold holdout in
`microcosm.build.holdout` is imported by the UK runtime and UK tools only. So a
US release says it beats the incumbent on the frozen register, and says it beats
it on held-out cells only once a US holdout exists and has run.

Publication stays a human step: `tools/publish_release.sh` wraps
`microcosm-publish-release`, which uploads `latest.json` last
(`microcosm.data.release.publish_release`), and only that pointer makes a
release the certified default.

## What stands between main and that build

A from-scratch attempt ran on main at `51c314382` on 15 and 16 September 2026
and reached no release. The causes below were re-read on main at `d1196af10`.
The engine lock that also blocked it is fixed: main resolves policyengine-us
2.2.1, policyengine-core 3.32.5 and spm-calculator 1.0.0.

### 1. No Chronicle feed on the build machine carries dimension labels

`_validate_chronicle_hierarchy_labels`
(`packages/microcosm-build/src/microcosm/build/ledger_targets.py:1025`) requires
exactly one Chronicle-owned label for every dimension a target selects on, and
refuses to substitute the identifier. The pinned US feed
(`consumer_facts_buildn_v9_4.jsonl`, `b3c08356…`, named in
`us/target_parity_manifest.json` and `us/target_parity_feed_families.json`)
predates those labels, so target compilation refuses. The refusal was measured
on the `--base-h5` arm, 49.59 seconds in. The pool arm never reached target
compilation in that attempt, and it cannot take the bare pinned feed at all,
because `--exact-k` needs an artifact directory with a manifest; the compile
call is shared, so the same validator stands in front of it. Every other stage
waits behind this one.

Chronicle main writes the labels (`dimension_labels`, `dimension_value_labels`,
`layout.groupby_dimension_label`), and the UK feed on main is already pinned to
such an export (`uk/chronicle_feed.json`). The re-pin was approved on 18
September 2026. The US needs a fresh export, the two
parity resources regenerated together with
`tools/build_us_target_parity_manifest.py`, and a `us/chronicle_feed.json` that
records the export the way the UK file does. `chronicle build-bundle` takes one
`--year`, and the pinned US feed carries many periods: tax years 2020 to 2026,
fiscal years 2023, 2024 and 2026 to 2029, calendar years 2018, 2023 and 2024,
and the months 2024-12 and 2025-12 (counted from the feed's own rows). One
`--year 2023` export at Chronicle `c5e5bf8` reproduces 555 of the feed's 586
record sets, every row labelled; the rest, including the JCT tax expenditures
and the CBO revenue projections that age dollar targets, sit in other years'
bundles. So the export's scope is settled, and its coverage compared with the
pinned feed record set by record set, before anything is re-pinned.

### 2. Nothing in the build emits the SPM independence role

In policyengine-us 2.2.1 one SPM unit with no classified adult (age 18 or over,
or age 15 or over with `is_spm_independent_minor_role`) refuses the SPM
measurement for the whole population. The first whole-dataset simulation of the
written release file is the reform-coverage smoke gate
(`tools/build_us_fiscal_refresh_release.py:11657`), which runs after the export
write (`:11641`) and before the calibration NPZ (`:11723`); its LIHEAP probe
reaches SPM composition through `spm_unit_benefits`
(`tools/build_us_release_input_coverage_manifest.py:513-520`), so a population
with such a unit refuses there. The reform-validation diagnostics that follow
the NPZ (`_write_reform_validation`, near `:11725`; 104 `in_poverty` rows from
`us/state_spm_poverty_levels.json` on one whole-dataset simulation) would
refuse the same way; they record, they do not gate.

The certified default passes because `tools/build_us_spm_role_enrichment.py`
added the role afterwards, and that tool accepts only Build P's exact bytes
(`:117-118`). The pins record what the role does there: 28 SPM units have no
member aged 18 or over, none lacks a classified adult once the role is present
(`packages/microcosm-data/src/microcosm/data/source_enrichment.py`,
`EXPECTED_COUNTS`). A dataset built from raw sources today carries no role.

Decided 17 September 2026:

- A preflight check counts SPM units with no classified adult before the engine
  runs, and names the remedy.
- The role is delivered for a base that is not Build P.
- SPM units are never re-grouped to make the count zero. That changes the
  poverty measurement the 104 rows exist to check.
- No adult is invented where the source delivers none.

Decided 18 September 2026: the role reaches a dataset built from raw sources as
a build-stage input leaf that carries the source role. Build P's enrichment
lane stays pinned to Build P.

### 3. The July selection does not map onto a rebuilt base

`check_selection_carryover` (`us_runtime/release_gate_preflight.py:313`) failed
on the rebuilt base: 15,228 capital-gains own-tail donors sit in households the
frozen July identity list does not name (the list names households, and the
selection mask is household-level, so those donors' tax units are dropped)
(`assert_puf_capital_gains_tail_survives_selection`,
`us_runtime/puf_capital_gains_tail.py:465`). The release tool does not require a
selection source (`--selection-source-manifest` is optional, and the reduction
at `tools/build_us_fiscal_refresh_release.py:9019` runs only when one is given).
Without one, calibration runs on the full base of about 353,000 households. The
only measured comparator is July's run on about 57,000 households: 2 hours 42
minutes at 85 GB. Nothing has been measured at full-base size.

A dataset built this way is a new lineage, not a replay of Build P: no frozen
support, a different capital-gains tail stratum, a different congressional
district assignment and a different feed identity.

Decided 18 September 2026: the next certified default may be a new lineage. The
`--exact-k` arm refuses a frozen selection source outright, so the July
selection is out of scope there, not merely inconvenient.

`--dense-default-dataset` is diagnostic only. A release build leaves it unset,
so the default is the sparse dataset that runs on standard machines.

### 4. Three raw inputs exist in one untracked directory

The base stage reads six processed inputs. The licensed IRS public use file pair
already lives in a token-required Hugging Face repository, and the 2022 ACS rent
donor is already on the Hub. The three processed CPS ASEC files
(`census_cps_2022.h5`, `census_cps_2023.h5`, `census_cps_2024.h5`) exist only in
an untracked directory on the build machine. They derive from public Census
files. The base stage records their digests and compares them to nothing.

Decided 18 September 2026: the three files are mirrored, unchanged, to a public
PolicyEngine dataset repository on Hugging Face; microcosm pins revision, digest
and size and fetches them as it fetches the SIPP donor; the base stage verifies
the digest it is given. `us/spec/sources.yaml` is generated from a sealed
six-role receipt and does not change until a real build re-cuts it.

## Measured stage times

From the September attempt, on one 128 GiB machine, under policyengine-us
1.819.0:

| Stage | Wall time | Peak memory |
|---|---|---|
| Base from raw sources (`tools/build_us_puf_support_base.py --stage all`), machine mostly free | 46 min 28 s | 72.47 GB |
| The same under contention | 1 h 24 min 18 s | 65.10 GB |
| Preflight on the fresh base | 32.58 s | 9.76 GB |
| Target compilation, to its refusal | 49.59 s | 3.40 GB |
