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
[`populace`](https://github.com/PolicyEngine/populace) stack. It loads anywhere
the enhanced CPS loads (an API-compatible alternative population), with its own
calibrated weights — and its own strengths and gaps, both documented below.

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

## What it is

One HDF5 `USSingleYearDataset` per year. The population is generated from a
declarative spec: Current Population Survey ASEC provides household structure,
demographics, benefits, and tenure (~half the records are PUF-derived clones,
flagged per record); tax detail is imputed from the IRS Public Use File with
weight-aware quantile-forest models; the wealth, mortgage, vehicle, insurance
-premium, and prior-year-income layers are imputed with the same models using
the published enhanced CPS as the donor (those layers are survey-imputed in
the incumbent itself); every imputed value is clipped to the incumbent's
realized per-record range (the support guard); and the result is calibrated to
PolicyEngine's administrative target surface (3,704 IRS/Census/program
targets) with a hard per-record weight bound (`max_weight_ratio=50`), so no
aggregate leans on a handful of super-weighted records. Same source classes
and hosting precedent as PolicyEngine's published enhanced CPS.

## Validation

Scored by the sound comparison — matched samples (41,314 households),
symmetric weight refit on the full administrative target surface, held-out
targets never seen by either side's refit. Lower is better.

| metric | populace-us | enhanced CPS |
| --- | --- | --- |
| training loss (2,965 targets) | **0.132** | 1.089 |
| held-out loss (739 unseen targets) | **0.032** | 0.317 |
| full-surface loss (3,704 targets) | **0.164** | 1.406 |

Per individual target the incumbent still wins more often (2,484 of 3,704 to
our 1,168, 52 ties): populace wins big where it wins and loses narrowly where
it loses. Both facts are the story.

Shipped-file properties: **0 parity gaps** (every PolicyEngine input the
enhanced CPS populates non-degenerately, this file populates), **95.55% of
3,704 calibration targets within 10%** (calibration loss 0.022), max household
weight 382,478 with **zero records above 500k** (the enhanced CPS ships 21,
max 1.05M). End-to-end through `Microsimulation`: 332.7M people, $93.2B SNAP,
$338.4B traditional 401(k) contributions, $163.6T net worth (incumbent:
$163.4T).

## Known gaps

We publish the misses with the hits:

- **Short-term capital gains over-weight large losses**: the aggregate is
  −$0.9T against the donor's weighted −$77B. The pool's records are faithful
  at design weights (−$164B); calibration amplifies loss-heavy records to hit
  the targets it can see, and net STCG is not on the target surface. A
  net-STCG calibration target is on the roadmap.
- **Donor-imputed layers inherit the incumbent's model error.** Wealth,
  mortgages, vehicles, premiums, and prior-year income are drawn from models
  trained on the enhanced CPS, whose own values for those layers are
  themselves survey-imputed (SCF/SIPP/ACS).
- **Aggregate household net income is $13.7T** vs the incumbent's $22.2T —
  most of the gap is the STCG item above plus thinner tail capital income;
  several program totals land closer to administrative actuals than the
  incumbent (SNAP $93.2B, SSI near the ~$60B outlay). Results from the two
  populations are not interchangeable.
- **Per-target wins still favor the incumbent** (see Validation): the
  aggregate losses are far lower, but on a majority of individual targets the
  enhanced CPS sits closer.

The dashboard at [populace.dev/dashboard](https://populace.dev/dashboard)
shows the full per-family calibration fit, the worst-fit targets by name, and
the weight distribution. Methodology and evidence:
[populace.dev](https://populace.dev); loader and registry:
[github.com/PolicyEngine/populace](https://github.com/PolicyEngine/populace)
(`packages/populace-data`).
