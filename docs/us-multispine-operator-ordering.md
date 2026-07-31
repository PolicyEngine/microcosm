# US multispine operator ordering

This note records the executable pool ordering in
`tools/build_us_multispine_pool.py` and the serial lineage it replaces. The
assembly and agreement contracts originated in populace#581; populace#578
increment 2 wires them into a build path. This documentation and its fixture
tests do not certify a full-data output artifact.

## Retired serial ordering

The earlier lineage used two serial builds. The first build produced an
operated ASEC-by-PUF-detail donor. The second build created ACS records,
transferred inputs from that donor, and only then appended ACS.

### `build_us_puf_support_base.py`

`PIPELINE_STEPS` and `STAGE_BOUNDARIES` are executable ordering
configuration. The monolithic and checkpointed paths implement the same
sequence.

| Boundary | Operators | State consumed |
|---|---|---|
| `source_construction` | `_load_base_frame_from_args` | Either an existing US base H5 or pooled ASEC unit frames built from the declared source years. |
| `pre_clone_enrichment` | `derive_us_cps_carried_inputs`; prior-year income, relationships, Medicare take-up, housing, eligibility, pregnancy, WIC, child support, disability benefits, workers compensation, weeks unemployed, childcare, energy subsidy, retirement contribution/distribution, and immigration operators | The ASEC-only frame, build year and seed; some operators also consume pinned external source tables. Housing can consume the pinned ACS 2022 rent donor when its input gate is not already satisfied. |
| `clone_feature_extraction` | `clone_us_frame_for_puf_support`; PUF donor extraction; primary-QRF initialization | The already enriched ASEC frame and processed PUF donor arrays. Cloning creates an ASEC copy and a PUF-tax-detail copy and splits weights across them. |
| `primary_qrf_chain` and `qrf_finalization` | Chained weighted QRF fits and finalization | Predictors on the cloned frame, PUF donor targets, design weights, fit seed, and estimator count. Predictions are assigned to the PUF-detail copy. |
| PUF tail and derived-detail stages | Capital-gains tail transfer, capital-gain distributions, QBI reconciliation | The cloned, QRF-imputed frame plus PUF donor detail and declared deterministic reconciliation rules. |
| post-clone input stages | WIC, housing assistance, prior-year income, child support, disability benefits, workers compensation, weeks unemployed, childcare, adult care, energy subsidy, retirement contributions/distributions, and education inputs | The operated ASEC/PUF-detail frame, source columns, build year, seed, and operator-specific external tables. Signal gates run between these mutations. |
| geography and export | Congressional-district assignment, block-ladder assignment, H5 export | The operated frame, geography artifacts and seeds, followed by the final frame writer. |

Thus many derivation, imputation, and seeded-assignment operators run before
any ACS peer spine exists. The exported H5 is not a raw donor: it contains the
results of the pre-clone, PUF-transfer, post-clone, and optional geography
stages.

### `build_us_acs_multispine_base.py`

The former ACS builder took that exported H5 as `--base-h5`. Its runtime call
graph was:

1. Load and validate the dense ASEC-by-PUF-detail base and declared transfer
   coverage.
2. Fetch the hash-pinned ACS archives and build an ACS unit frame.
3. Map ACS-native inputs.
4. Preflight pooled geography when a PUMA ladder is supplied.
5. Run `transfer_acs_inputs(mapped_acs, operated_base, ...)`. The operated
   base is the QRF donor; declared input leaves and deterministic
   post-transfer structure are added to ACS.
6. Run `with_optional_acs_spine(operated_base, transferred_acs, ...)`.
   This is the first point at which the two household spines share one
   frame. It rescales their household mass, remaps colliding IDs, aligns
   nullable columns, and can assign the pooled PUMA geography ladder.
7. Audit fits and nullable inputs and write a pre-calibration staging H5.

Calibration is downstream of this tool. Simulation is not run by either
builder in this call graph.

The important ordering flaw is structural: ACS receives model-input
transfers from a donor after the donor has crossed the ASEC-only operator
sequence. Appending the transferred ACS records later does not cause those
operators to run over the combined population.

`build_us_acs_multispine_base.py` is now a deprecated compatibility shim for
shared legacy H5 helpers. It does not expose a second executable pool builder.

## Executable increment-2 pool build

`build_us_multispine_pool.py` consumes only explicit local files and their
declared SHA-256 values:

- the input-complete ASEC checkpoint from the
  `pre_clone_enrichment` outer-stage boundary;
- the ACS household and person PUMS archives, whose caller-supplied hashes
  must also match the checked-in ACS source manifest; and
- the processed PUF H5 and source-year PUF CSV used by the existing donor
  loader.

The tool does not download any source. It verifies all file pins before
loading frames, maps measured ACS fields without overwriting them, and then
runs this fixed sequence:

1. `assemble_spines({"asec": ..., "acs": ...})` creates the first shared
   population state and binds the immutable assembly receipt.
2. `clone_us_frame_for_puf_support(...)` applies the PUF-detail clone to the
   whole assembled pool. Clone-index provenance, not source-spine identity,
   controls later PUF-detail routing.
3. The primary PUF QRF chain and capital-gains tail transfer run over the
   combined frame, followed by the declared ACS input-family QRF transfers.
   Existing measured target cells remain unchanged, and transfer receipts
   record fitted and imputed rows.
4. Deterministic input reconciliation runs over that same pool.
5. The seed stage preserves existing take-up values, applies the sourced
   TANF and EITC mechanisms, and explicitly receipts live engine defaults
   used for unresolved, non-transfer-owned take-up inputs. Those defaults
   are not described as fitted or administrative mechanisms.
6. SSI is materialized only on an ephemeral agreement view in fixed
   household batches. Any engine defaults required solely for that
   calculation are separately receipted; formula output is not written into
   the input pool.
7. The unchanged spine-agreement gate is terminal. It uses its checked-in
   registry and fixed tolerances, batches all failures, and controls the
   manifest's simulation-ready status.

The output H5 is a nullable, input-only, pre-calibration pool. Its companion
manifest carries input pins, the assembly receipt, per-source and per-clone
counts, operator receipts, and the complete agreement result. A failed gate
writes diagnostics and a non-ready manifest and exits nonzero. Calibration
is deliberately absent; the downstream k-ladder may consume only a pool
whose terminal agreement result passed.

## Provenance axes

The retired lineage used two related metadata schemes:

- PUF support cloning adds, on every entity,
  `*_source_id`, `*_support_channel`, and
  `*_support_clone_index`. The default support-channel values are `asec` and
  `puf_tax_detail`; clone index zero keeps the original IDs and later clones
  receive remapped IDs.
- Late ACS pooling adds `*_spine`, with `asec_puf` on the dense donor and
  `acs_2024_1yr` on ACS. When the donor already has complete support metadata,
  pooling synthesizes ACS support metadata with its native ID, channel
  `acs_2024_1yr`, and clone index zero.
- `transfer_acs_inputs` keeps fit and transfer provenance outside the frame,
  including the target family, donor spine/channel, predictors, seeds,
  weight kind, recipient patterns, and unmodeled-row count.

Those fields mixed two concepts: the population source that carried a record
and the PUF-detail copy created by an operator. The assembly seam keeps those
concepts separate.

## Canonical executable ordering

The US multispine build order is:

```text
source ingestion and faithful schema harmonization
    -> assemble peer spines
    -> clone PUF detail
    -> impute
    -> derive
    -> seed take-up and other stochastic inputs
    -> simulate
    -> spine-agreement gate
    -> emit input-only pool and receipts
```

Calibration is a downstream consumer boundary, not a stage in this tool.

`assemble_spines(...)` is the boundary between source preparation and
population operators. It receives nullable, schema-compatible peer frames
and produces one combined frame before cloning, fitted transfer, derivation,
seeded assignment, simulation, or calibration. Downstream operator
entrypoints receive that combined frame and operate on measured
characteristics without selecting behavior by source spine.

ASEC and ACS are peer household spines. A future household source can join
the same assembly contract. PUF tax detail is not a peer spine: it remains a
clone operator applied after assembly, so every assembled household source
is subject to the same clone and downstream operator sequence.

The new provenance contract is:

- `*_support_channel` records source-spine provenance. Its vocabulary is the
  set of source names declared at assembly, such as `asec`, `acs`, and future
  peer channels.
- `*_spine_source_id` is the entity ID in the source frame before assembly
  remaps colliding ID spaces.
- `*_source_id` is the assembly-unique structural ID before cloning. Operator
  clones retain it so a source record and its copies remain one lineage even
  when two peer spines reused the same local ID.
- `*_support_clone_index` records operator-created copies. Index zero is the
  assembled source record; the PUF-detail clone is identified by its clone
  index rather than by changing the source channel.

`Frame.table()` exposes mutable pandas tables, so the provenance columns are
not made read-only by the dataframe API. Enforcement instead uses a private,
deeply frozen frame-metadata receipt. At assembly it records the declared
channel set and the native row count for every entity/channel pair. Assembly
output, PUF clone entry/output, and the spine-agreement gate validate live
channels and clone-index-zero row counts against that receipt. They also
require every person's channel to agree with each linked group row, including
its household. An unknown/forged channel, a drifted native count, a missing
receipt on an assembled frame, or a cross-grain mismatch raises `ValueError`
that names the assembly manifest. US runtime frame rebuilds carry the receipt
whenever they carry the source frame's mass log, and a structural test rejects
a mass-log-preserving rebuild that drops it.

Assembly, provenance reporting, and the spine-agreement gate may read source
channels. Population operators must not. In particular, an operator may
route PUF-detail behavior using clone provenance, but it may not make a fit,
draw, transformation, or overwrite conditional on `asec`, `acs`, or another
source channel. The current unassembled lineage remains compatible: when the
raw-spine ID field is absent, its historical `asec`/`puf_tax_detail` channel
labels are validated and translated to clone roles centrally.

The structural guard covers person, household, tax-unit, SPM-unit, family,
marital-unit, and benefit-unit naming variants. It recognizes direct,
aliased-helper, dynamic-subscript, `getattr`, and `*_spine_source_id` reads.
Every US runtime module is classified as a reviewed population operator or an
explicit non-operator/provenance owner, so a new unclassified module fails the
guard. A second fail-closed scan starts at `build_us_multispine_pool.py` and
covers the tool plus its transitive US-runtime import graph under the same
owner registry. `transfer_acs_inputs` selects fit donors by the centrally
derived clone role; assembled source-channel names never determine donor
eligibility.

Assembly accepts only integer-typed, nonnegative structural source IDs and
names the source spine and offending IDs on failure. PUF cloning revalidates
integrality without truncating fractional values and accepts only the
canonical native/PUF-detail role pair.

## Source harmonization and geography boundary

“Assemble before operators” starts after the minimum work needed to represent
each source as a valid US `Frame`. File decoding, unit construction, ID
normalization, column naming, categorical decoding, and faithful mappings of
observed source fields are source ingestion or schema harmonization. They may
occur before assembly and may be source-specific.

That exception is narrow:

- Measured values remain untouched. A native mapping can carry an observed
  value into its canonical column, including its source provenance; it is
  not permission to fill, smooth, reconcile, or overwrite the measurement.
- A donor transfer, fitted imputation, deterministic synthetic allocation,
  seeded take-up assignment, or model simulation is a population operator
  and belongs after assembly.
- Observed geography such as an ACS PUMA can be normalized and preserved
  before assembly. Drawing missing PUMAs, congressional districts, counties,
  tracts, or other geography is an operator. It belongs after assembly and
  must condition on the available measured geography, not on the source
  channel.

This distinction leaves the exact source-to-schema adapters to their own
contracts while making the first shared mutable population state explicit.

## Gate and calibration boundary

The spine-agreement gate runs after simulation and before calibration. Its
registry covers each transferred or imputed input family, every variable in
the checked-in take-up contract, and the downstream `ssi` simulation output
measured by the multispine QA contract. This includes
`takes_up_ssi_if_eligible`. The registry declares the statistics and
tolerances used to compare source-conditional distributions. Failures are
collected and reported as one batch. Calibration must not consume a frame
whose agreement gate failed.

The increment-1 registry is generated from the declared ACS transfer families,
with transfer batch suffixes normalized and deterministic transfer outputs
included, then extended from the take-up inventory and the declared simulated
output surface. Each registered numeric or boolean distribution uses the same
fixed contract, with no family-specific override:

- every pair of source spines is compared using positive record weights;
- the weighted nonzero-incidence ratio must be in `[0.8, 1.25]`; and
- among nonzero records, the largest symmetric relative distance at weighted
  q10, q25, q50, q75, or q90 must not exceed `0.25`.

The union of source channels observed across registered entity grains defines
the required pair set, so a missing third-spine entity comparison is explicit
and fails. A one-sided zero incidence is a disagreement. A both-zero pair is
recorded as untestable in gate details and also fails; registered all-zero
surfaces cannot pass silently.

The gate returns one `GateResult` containing every malformed-input and
distribution failure rather than stopping at the first disagreement.

This ordering makes the gate diagnostic of the shared operator surface:
calibration cannot hide a disagreement, and no per-spine target, loss term,
seed, or tolerance may be introduced to shape a passing result.

## Increment-2 validation boundary

This increment makes the assembly seam executable and covers it with small,
synthetic two-spine fixtures. It does not execute or certify a full-data
build, download a source dataset, calibrate the pool, select k, or change the
current sparse/dense release artifacts. Those data-scale and release steps
remain downstream of this code increment.
