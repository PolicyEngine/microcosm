---
license: mit
pretty_name: populace-us
tags:
  - policyengine
  - microsimulation
  - synthetic-population
  - tax-benefit
  - united-states
---

# populace-us

The **populace-built US population**: a calibrated synthetic microdataset for
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us), built by the
[`populace`](https://github.com/PolicyEngine/populace) stack **entirely from
primary sources** — the enhanced CPS appears only as the benchmark this file is
scored against, never as a build input. It loads anywhere the enhanced CPS
loads (an API-compatible alternative population), with its own calibrated
weights — and its own strengths and gaps, both documented below.

## Load it

```bash
pip install 'populace-data[us]'
```

```python
from policyengine_us import Microsimulation
from populace.data import load

sim = Microsimulation(dataset=load("us", 2024))
sim.calculate("household_net_income", 2024).sum()
```

Or grab the H5 directly:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="policyengine/populace-us",
    filename="populace_us_2024.h5",
    repo_type="dataset",
)
```

## How it is built

One HDF5 `USSingleYearDataset` per year. Every layer comes from a primary
survey or administrative source:

| source | provides |
| --- | --- |
| Census CPS ASEC | household structure, demographics, incomes, benefits, tenure, hours, occupation flags, health coverage at interview, retirement distributions (DST codes), childcare, prior-year income (longitudinal PERIDNUM join) |
| IRS SOI Public Use File 2015 (uprated) | tax detail: capital gains, dividends, interest, itemized-deduction inputs, QBI/SSTB components, partnership self-employment, estates, tuition |
| Fed SCF 2022 | wealth: bank/stock/bond assets, net worth, mortgage balance hints |
| Census SIPP | tip income for tipped occupations; household vehicles (count and value) |
| CPS-ORG | hourly wage, paid-hourly status, union coverage |
| MEPS-IC parameters | employer-sponsored insurance premiums |
| Census ACS 2022 | rent for renter households |

Imputations use weight-aware quantile-forest models fit on each donor's own
records, and every imputed value is clipped to **that donor's** realized range
(the support guard) — nothing is anchored to the enhanced CPS. The result is
calibrated to PolicyEngine's administrative target surface (3,704 IRS/Census/
program targets, **plus a signed net short-term capital gains target** so the
optimizer cannot silently drive a net-negative aggregate to extremes) with a
hard per-record weight bound (`max_weight_ratio=50`), so no aggregate leans on
a handful of super-weighted records.

## Acceptance gates

The build refuses to publish unless every gate passes; this file passed all
of them:

- **Parity 0**: every PolicyEngine input layer the enhanced CPS populates
  non-degenerately, this file's simulation populates (169 reference layers
  checked at simulation level).
- **Exported-nonzero**: all 309 stored columns carry signal — no all-zero
  scaffolding that would silently mask engine formulas or defaults.
- **Calibration**: 95.09% of 3,704 targets within 10% (loss 0.022); max
  household weight 297,651 with **zero records above 500k** (the enhanced CPS
  ships 21, max 1.05M).
- **Smoke aggregates** through `Microsimulation`: 332.3M people, $97.8B SNAP,
  $176.5T net worth (Fed Z.1 ≈ $169T), net short-term capital gains
  **−$77.5B** against the −$76.8B PUF-anchored target, tips $52.9B, rent
  $757.5B.

## Validation

Scored by the sound comparison — matched samples (41,314 households),
symmetric weight refit on the full administrative target surface, held-out
targets never seen by either side's refit. Lower is better.

| metric | populace-us | enhanced CPS |
| --- | --- | --- |
| training loss (2,965 targets) | **0.190** | 1.089 |
| held-out loss (739 unseen targets) | **0.038** | 0.317 |
| full-surface loss (3,704 targets) | **0.228** | 1.406 |

Per individual target the incumbent still wins more often (2,613 of 3,704 to our 1,040, 51 ties): populace wins big where it wins and loses narrowly where it loses. Both facts are the story.

## Known gaps

We publish the misses with the hits:

- **Net worth runs ~4% above Fed Z.1** ($176.5T vs ≈ $169T): the calibration
  target ($160T) sits below Z.1 and the achieved total lands between them.
- **Investment interest expense is thin** ($5.1B against IRS SOI ≈ $24B):
  the PUF-residual rule populates the layer conservatively; a dedicated SOI
  calibration target is the roadmap item.
- **Per-target wins vs the incumbent**: see Validation — aggregate losses
  are what the comparison gates on, but per-target patterns differ between
  the two populations. Results are not interchangeable.

The dashboard at [populace.dev/dashboard](https://populace.dev/dashboard)
shows the full per-family calibration fit, the worst-fit targets by name, the
weight distribution, and a live strip while a build chain runs. Methodology
and evidence: [populace.dev](https://populace.dev); loader and registry:
[github.com/PolicyEngine/populace](https://github.com/PolicyEngine/populace)
(`packages/populace-data`).
