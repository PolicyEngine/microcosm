# Released-frame probe and measurement status

A full release build is unnecessary for this diagnostic. The public
`reseed_us_ssi_take_up` function accepts the existing frame, its delivered
weights, and an aligned December `uncapped_ssi` engine probe. It returns a new
frame and stage diagnostics. Pass the original seed and registry SSA targets;
optional explicit reporter source IDs preserve anchors when pruning removed
their ASEC support rows. Capture canonical source holdings once with
`us_ssi_take_up_source_liquid_assets` and pass the unchanged map to both
reseeds and later diagnostics; a map from the original pre-pruning frame can
be supplied when available. The function draws for every source person, including
people who fail the current asset test, and changes only the take-up flag.

These local source files were inspected read-only and their complete SHA-256
digests verified on 2026-10-09:

| Frame | Cached Hugging Face snapshot | H5 SHA-256 | People | Households |
|---|---|---|---:|---:|
| July Build P | `f09f2f3b9fa8409642dc0c7fc9c8f7516ae0e3c5` | `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e` | 166,321 | 57,240 |
| Certified SPM enrichment, `populace-us-2024-spm-20260915` | `8ab57ffc2ca41d8af631ff62ebbce95e966299f3` | `6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84` | 166,321 | 57,240 |

For either row the local filename is
`/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/<snapshot>/populace_us_2024.h5`.
Both contain `age`, `SSI_VAL`, `person_source_id`, `bank_account_assets`,
`stock_assets`, `bond_assets`, `takes_up_ssi_if_eligible`, and support provenance.
The second hash agrees with the cached SPM release manifest. That manifest
certifies policyengine-us `>=2.0.0,<2.3` and policyengine-core `==3.32.5`.
This inspection does not itself recertify either artifact.

A subsequent bounded attribute audit read only `person_source_id`, `age`, and
the three liquid-asset inputs. Both frames have 135,833 source IDs, of which
30,488 have two rows. Age agrees within every source ID, but total liquid
assets disagree for **9,446 source IDs (18,892 rows)**; the maximum within-source
spread is $60,547,861.95822643. All inspected ages are finite and all asset
components are finite and nonnegative.

A follow-up channel audit found exactly one physical `asec` row for every
conflicting ID; none of those IDs lacks its ASEC row or has multiple ASEC rows.
Across all IDs, 66,001 retain one ASEC row and 69,832 retain only PUF-role rows.
The row counts are 66,001 ASEC and 100,320 `puf_tax_detail` in each frame.

The runtime now makes source-attribute ownership explicit: the physical ASEC
row supplies a source person's holdings when present; contradictory physical
ASEC rows fail. Without a physical row, surviving copies must agree, with a
sole survivor permitted. Every observed conflict has an unambiguous physical
ASEC owner, so both full cached frames satisfy this holdings rule. Propensities,
asset-band diagnostics and intercept-basis asset mass use that canonical source
holding. The diagnostics expose the 9,446 support disagreements. This is a
static attribute-compatibility finding; the helper and engine probe were not
executed on these frames.

The existing financial-asset source contract explains why differing support
holdings can arise: `scf_wealth.py` draws a vector for each actual recipient
household, with its seeded SIPP/SCF selector keyed to the actual household ID;
its SCF predictors and `sipp_financial_assets.py` predictors use that row's
income components and household head status. They do not deduplicate or fan
holdings by `person_source_id`. `puf_support.py` preserves the baseline ASEC
record while its copy receives PUF tax detail. This is policy-invariant support
imputation, rather than evidence for averaging the two holdings. Canonicalizing
the take-up attribute does not rewrite any row's bank, stock or bond inputs.
PUF engine resource inputs can therefore still differ from the physical source
holding, and the candidate mask remains the independent probe on those actual
inputs. That distinction is an open resource-measurement risk (#424), not a
claiming correction. The builder captures a canonical holdings map before
assignment and carries it unchanged into final diagnostics, preserving the
frozen Bernoulli law if pruning removes a physical owner. A supplied map must
cover every retained source ID with finite nonnegative values and agree with
still-present physical ASEC rows; it may contain IDs removed by pruning.
Without that original map, capture on an already pruned frame uses its remaining
ownership evidence and cannot recover discarded physical holdings. That is a
new reseed basis, not a replay of the original frozen assignment.

The minimal reseed recipe reuses the fiscal builder's existing batched candidate
probe. Run it through `heavy --mem 16 -- ...` in a compatible engine environment
with the workspace packages and `tools` on `PYTHONPATH`:

```python
import json
from pathlib import Path

from build_us_fiscal_refresh_release import _ssi_person_uncapped_amount
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.build.us_runtime.ssi_take_up import (
    reseed_us_ssi_take_up,
    us_ssi_take_up_source_liquid_assets,
)

# Supply the original build seed and the same ledger-fed targets used by
# the builder; these are caller inputs, never reform-derived quantities.
frame = load_legacy_calibrated_us_h5(export_path)
# If available, use the original pre-pruning map instead of this capture.
source_liquid_assets = us_ssi_take_up_source_liquid_assets(frame)
candidate_probe = _ssi_person_uncapped_amount(
    frame, maximum_microsim_batch_size=2000
)
gradient_frame, gradient_diagnostics = reseed_us_ssi_take_up(
    frame,
    uncapped_ssi=candidate_probe,
    seed=original_seed,
    targets=targets,
    source_liquid_assets=source_liquid_assets,
)
constant_frame, constant_diagnostics = reseed_us_ssi_take_up(
    frame,
    uncapped_ssi=candidate_probe,
    seed=original_seed,
    targets=targets,
    source_liquid_assets=source_liquid_assets,
    asset_slopes=dict.fromkeys(targets, 0.0),
)
# Write aggregate diagnostics to an explicitly chosen workspace output.
Path(output_path).write_text(json.dumps({
    "gradient": gradient_diagnostics,
    "zero_slope": constant_diagnostics,
}, indent=2) + "\n")
```

This recipe is not a release, calibration, publication, or tested scoring
driver. The source frame and its weights stay fixed. Score current law,
$10k/$20k, and full elimination independently on each returned frame through
the existing compatible engine harness, keeping household batches and matching
person IDs. Classify the outcome age bands using the harness's outcome-period
age variable. Never pass a reform eligibility mask or reform outputs back into
the reseed helper.

Keep three results: the incumbent's persisted flags, the zero-slope reseed and
the estimated-gradient reseed. Recomputing priors on delivered weights can
change the zero-slope result relative to the incumbent. For elderly full
elimination, call the three new-recipient counts `I`, `Z`, and `G`. Report
`(Z - G) / (I - 115000)` as the gradient-only share of the measured published
CBPP gap and `(I - G) / (I - 115000)` as the total reseed share. Handle a zero
denominator explicitly. The historical rounded July gap in the task is
`694000 - 115000 = 579000`; using that denominator should be labelled historical
if the engine replay does not reproduce 694,000. Report the adult $10k/$20k
count beside 153,211 (the task's incumbent) and CBPP's published 154,000, without
tuning the slope to either. The elderly $10k/$20k cell is unpublished and has
no numerical target.

The heavy wrapper could not start even its harmless permission probe:

```text
heavy --mem 8 -- python3 -c 'print("SSI diagnostic permission probe")'
...
PermissionError: [Errno 1] Operation not permitted: 'ps'
```

The wrapper fails while reading its process registry, before starting the
requested program. No engine probe was run outside that wrapper. Consequently
the July gap share, adult non-regression and reform counts are **not evaluated**;
no percentage of the elderly gap is claimed. The exact status is recorded in
[released_frame_probe_status.json](released_frame_probe_status.json).
