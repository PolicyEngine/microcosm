# Evidence for the Lifetime ISA stage (microcosm#1003)

These are aggregate-only extracts of the licensed builds behind the `was_lisa` spine stage.
[`experiments/1003-uk-lisa-receipts.md`](../../../experiments/1003-uk-lisa-receipts.md) is the
receipt that reads them, and [`docs/uk-lisa-1003.md`](../../uk-lisa-1003.md) is the topic doc.

The licensed inputs and outputs live outside the tree under `data/ukds/acceptance/1003-lisa/`:

- the WAS round-8 household and person tabs;
- the spine H5 files and their build sidecars;
- the full receipts, which include per-record detail.

## The builds

All four were built on the #930 lane's SPI-first spine recipe with `--was-person-tab`, seed 42 and
the same pinned inputs:

- `spine-ctl`: main `5187fce25`.
- `spine-lisa`: branch head `360e7846c`.
- `spine-lisa-2`: a rebuild of `spine-lisa`, for determinism.
- `spine-lisa-b`: the head without the four financial predictors. This is measurement only and was
  never committed.

Every tree carries the measurement-only PLAN_5 student-loan realisation relaxation (1.0 to 2.0)
while #1049 is open. It is never committed.

## Files

- `donor_audit.csv`: the donor audit of receipt Part A.
  - The person-to-household pin check, the non-dependent adult population and the released and
    credible holders.
  - The response classes and the credibility rule's counts and weighted shares.
- `ownership_model_comparison.csv`: receipt Part B.
  - Held-out ownership shares by age, tenure and income for the stage logistic, the house regime
    gate, the gate tuned for rare events and the age-group rates.
  - Each model's held-out weighted log loss. The folds are five, by household.
- `ownership_realised.csv`: receipt Part C.
  - Weighted ownership shares of adults on the donor, the candidate spine and arm B, from the
    stage evidence, plus the model's expected share.
  - Co-holding within households.
- `balance_quantiles.csv`: weighted owner balance quantiles and means for four populations:
  - the donor's credible holders;
  - the held-out forest draws;
  - the spine;
  - arm B.
- `financial_draws_by_age.csv`: receipt Part F, by age group.
  - The spine's WAS financial draws against the donor's, overall and by support channel.
  - Weighted medians of gross financial wealth, and weighted means of log(1 + x) for the
    financial and earnings predictors.
  - The private-renting share.
- `checks.csv`: receipt Parts D and E.
  - The coherence cap, the support clip and the release-side column checks on the written H5.
  - The spine gate counts and the twin and determinism verdicts.

## Scripts

`scripts/` holds the measurement scripts, formatted but otherwise as run. Rerunning them from the
committed copies reproduces the lane's receipts byte for byte:

- `lisa_receipt.py` reads a build's `was_lisa` stage evidence, recomputes the realised figures
  from its H5 and runs the release-side checks.
- `model_comparison.py` is the five-fold held-out comparison on the donor.
- `fw_probe.py` compares the stage-time spine's financial draws with the donor's.
- `twin_diff_1003.sh` runs the twin payload compare, the classification and the adjudication.
- `determinism_1003.sh` runs the rebuild and its payload compare.
- `extract_lisa_1003_evidence.py` writes the CSVs here from the lane's receipts.

## Redaction

- Donor counts under 10 are withheld, and so is any count that would reveal one by subtraction:
  - holders above the £40,000 ceiling;
  - the credible holders in the balance fit;
  - the banded and ownership-imputed classes apart.
- The donor's lowest household-income tertile has fewer than 10 holders, so income is published
  as tertiles 1–2 against tertile 3.
- Balance quantiles and means are rounded to two significant figures, and weighted medians of
  financial wealth to three.
- No single respondent's value is written, neither a donor maximum nor the spine's largest draw.
