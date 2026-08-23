# QBI amount-margin ownership evidence

This package answers workstream 5 of the battery adjudication. It uses the
SHA-bound failed-attempt checkpoint named by the adjudication, rather than a
new lane build. The canonical machine receipt is `evidence.json`; the
extractor refuses to write it when any artifact binding, comparator value,
ownership assertion, realized regime, or terminal invariant disagrees
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-995,999-1413,1990-2003`).

## Artifact and comparison contract

The replay authenticates the publication manifest and gates, all three stage
checkpoints, 13 target-bank files, and their target identities. The simulated
checkpoint SHA-256 is
`5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a`;
the transferred checkpoint SHA-256 is
`bdc9355d92659bb28d58b1ddcd647ec303f2ad217661e17d5b4b0984e04532e8`.
The bound publication is `gate_failed` and `simulation_ready=false`; this is
diagnostic evidence, not release certification.

`transferred` contains the `impute` phase, while `simulated` also contains
`derive`, `seed`, and `simulate` (`tools/build_us_multispine_pool.py:2139-2141`).
QBI reconciliation runs during derive after the transfer
(`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2851-2908`).
That makes clone 1 in `transferred` the upstream PUF-producer measurement,
clone 0 in `transferred` the late-transfer/pre-reconciliation measurement,
and clone 0 in `simulated` the terminal battery measurement.

The replay uses the production comparator contract: targets are grouped by
their registered clone role and then scoped to positive-weight rows on that
role (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:14006-14391`).
The incidence leg divides weighted ACS carrier prevalence by weighted ASEC
carrier prevalence, and the magnitude leg compares weighted conditional
quantiles (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13891-13893,14441-14555,14598-14643`).

## Ownership result

Terminal provenance and the first stage at which a criterion becomes red are
different facts. The exact replay values are recorded at
`evidence.json:4-713`, and the eight check-level attributions at
`evidence.json:121397-121862`.

| amount | clone-1 producer ratio / QED | transfer ratio / QED | terminal ratio / QED | incidence first red | QED first red |
|---|---:|---:|---:|---|---|
| `qualified_bdc_income` | 1.068456 / 0.584598 | 0.329192 / 0.827384 | 0.328928 / 0.751361 | transfer | producer |
| `qualified_reit_and_ptp_income` | 1.047306 / 0.597071 | 0.385159 / 1.162639 | 0.446585 / 1.157625 | transfer | producer |
| `unadjusted_basis_qualified_property` | 1.107017 / 0.187807 | 0.774548 / 0.590155 | 0.774548 / 0.590155 | transfer | transfer |
| `w2_wages_from_qualified_business` | 1.172296 / 0.451897 | 0.760975 / 1.337128 | 0.760975 / 1.337128 | transfer | producer |

Thus all four incidence checks and UBIA QED are transfer-first. BDC,
REIT/PTP, and W2 QED are already over the unchanged 0.25 ceiling on the
clone-1 producer and worsen after transfer. All eight terminal values have
the `qrf_transfer` origin channel.

The failed artifact predates the explicit origin field, so that last statement
is a code-plus-receipt inference for the historical checkpoint. The extractor
byte-verifies each exporting bank target and verifies that the old late receipt
names `puf_clone` as producer, 964,699 authorized/imputed complement rows, and
zero unmodeled/residual rows
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-995,1087-1195,1668-1717`).
The production route selects ASEC clone 1 as donor, treats clone>0 as the PUF
producer surface, fills only its complement, and proves producer byte identity
and null-flow accounting
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:10920-10972,11025-11041,11076-11191,11194-11313`).

Going forward, that inference is explicit. Each target leaf records producer
roles, row counts, origin channel, model target, availability-pattern catalog,
and realized regime map; the aggregate and group copies are validated against
the canonical donor/recipient route and the live producer mask
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:5072-5198,5201-5754,11286-11308,11635-11759`).

## Realized QRF regimes

The adjudication requires the regime to be recomputed from frozen donor
support for every availability pattern before choosing a regime-specific
remedy. The replay reconstructs the exact 108,073-row ASEC clone-1 donor
support, matches every banked donor identity, and calls `detect_regime` with
the banked `zero_atol` for all 52 cells (13 chained targets × four patterns)
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:1857-1972`).
The 16 cells belonging to the four red amount targets are all
`zero_inflated_positive`. The historical bank did not persist those regimes;
that missing field is the adjudication's receipt gap.

This conclusion comes from donor replay, not from trusting a receipt. In the
current runtime, both monolithic and banked fitting independently recompute
the donor-support regime and reject disagreement from the fitted/drawn QRF
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1631-1651,1705-1885,1908-2206`).
Current origin receipts persist the full pattern/regime map
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-607,634-723`).
Receipt validation proves schema, canonical predictor membership, catalog
agreement, model-target binding, and agreement among sibling copies
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4423,4426-4687,5364-5416,5676-5742`).
It does not substitute for replaying donor values when establishing regime
truth.

The regime is mechanically important: the QRF detects support regime before
fitting its zero/sign gate and conditional amount forests
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:104-150,950-1003,1333-1429`).
The historical artifacts do not retain recipient-level gate scores/outcomes
or channel-by-pattern gate cross-tabs, so they locate ownership without
identifying which gate feature causes the transfer collapse.

## Coupled reconciliation result

The transferred checkpoint has the following nine pre-reconciliation
violations; the simulated checkpoint has zero for all nine
(`evidence.json:121373-121396`):

| invariant | transferred | terminal |
|---|---:|---:|
| non-SSTB qualification route | 35,121 | 0 |
| non-SSTB rows with SSTB income | 34,184 | 0 |
| BDC exposure cap | 2,996 | 0 |
| REIT/PTP exposure cap | 22,350 | 0 |
| qualification overlap | 34,817 | 0 |
| SSTB qualification route | 810 | 0 |
| SSTB rows with non-SSTB income | 13,433 | 0 |
| SSTB UBIA split | 26,636 | 0 |
| SSTB W2 split | 4,917 | 0 |

The reconciliation kernel makes the SSTB/non-SSTB splits and applies the BDC
and REIT/PTP exposure caps
(`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1266-1359`).
Its summary recomputes the same nine identities
(`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1377-1487`).
For UBIA and W2, transfer and terminal measurements are identical; for BDC and
REIT/PTP, reconciliation is secondary and does not cure the red transfer
margin.

## Separate SSTB terminal-role finding

The historical terminal clone-0 values of
`sstb_self_employment_income_would_be_qualified` and `business_is_sstb` are
dead by construction, while clone 1 is live and in band: ratios 1.06307 and
1.06573. The reconciliation explicitly clears both SSTB flags on the ASEC-role
surface (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1242-1264`).

The implemented terminal authority therefore assigns exactly those two
physical checks to PUF-detail clone 1 and defaults every other target to clone
0 (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2133-2160`).
It rejects any missing, extra, malformed, or duplicate role declaration
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3030-3131`),
and the canonical spec carries the same exact assignments
(`packages/microcosm-build/src/microcosm/build/us/spec/battery.yaml:459-503`).
These are live signal checks, not reviewed exclusions.

## Reproduction

```bash
uv run python experiments/qbi_ownership/extract_qbi_ownership_evidence.py
```

Debug runs may skip the expensive artifact digest only with a noncanonical
output path; the extractor forbids `--skip-sha` from overwriting canonical
evidence
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:1432-1436,1990-2003`).
Two fully verified runs produced byte-identical output at SHA-256
`38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`
with zero adjudication mismatches.
