# ACS rent preparation

`microcosm.build.us_runtime.housing_inputs` exposes the existing external ACS
rent calculation as three operations. The public `impute_us_pre_subsidy_rent`
entry point composes them around the ordinary weighted QRF fit and prediction.

1. `prepare_acs_rent_donor(donor)` returns `AcsRentDonorPreparation`: the actual
   sampled numeric training table, predictor order, and versioned categorical
   encoding context. It preserves the joint rent/real-estate-tax sample cap,
   target allocation masks, reset-index and subsequent rent-filter order,
   observed rent values, and household design weights. It neither reads a
   recipient nor fits a model.
2. `prepare_acs_rent_recipient(frame, donor_context=...)` returns
   `AcsRentRecipientPreparation`: the encoded household-head predictors with
   their household-ID index, corresponding head positions in the original
   person table, and original person index. `head_positions[i]` is the head
   for household `predictors.index[i]`; this position vector may be a
   permutation. Unsupported recipient categories fail against the donor's levels.
3. `finalize_acs_rent(prepared, raw_rent)` checks the named raw prediction and
   exact household index, clips negative predictions to zero, and places the
   each household's result on its own head. Non-head values remain zero.

The encoding context has schema `microcosm.us.acs_rent_encoding.v1`. It records
the sorted donor levels for `tenure_type` and `state_code_str`, and the exact
numeric-plus-dummy predictor order. In-memory mappings and level tuples are
immutable; ordinary JSON round trips are accepted and validated. Preparation
tables are local staging values. They are not immutable graph payloads or fitted
models, and the context is not a provenance substitute for its authenticated
donor source.

The external rent fit depends on its ACS donor, sampling recipe and fit
parameters. It can be reused when recipient source or sample ancestry changes.
This does not remove the real rung dependence of later ASEC-to-ACS gap-fill
donors: those donors are prepared from the actual sampled ASEC support.

The two-tree synthetic acceptance test compares the ordinary DataFrame fit,
using `weights="household_weight"`, with the existing per-target graph fit and
apply kernels using typed DESIGN weights. Raw draw bytes agree for positive,
hurdle and zero rent regimes; a recipient-only source change reuses the fitted
model. These tests start from honest prepared source populations. Production
derived-recipient graph wiring needs a declared model-input layout that retains
the household-ID index; it must not substitute the population table's index.

An identity-scatter correction follows the pure refactor: the legacy code
assigned household-ordered draws to person-ordered heads, which could place rent
on another household's head. Preparation now records the actual household-to-head
association without changing predictor order, model fitting, sampling or random
draws. A shuffled membership fixture with distinct draws proves the correction;
the pinned aligned-order fixture retains its exact encoded training, prediction
and final draw bytes against the pre-refactor implementation.
