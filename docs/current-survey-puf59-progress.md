# PUF59 and current survey predictor implementation status

The PUF59 profile fits and transfers 56 person outcomes and three tax-unit outcomes. Six detailed mortgage fields remain owned by the separate SCF producer. The historical full65 default remains available. PUF59 conditions on the source-specific PUF2015 filing-status class and capped return-size proxy, plus six monetary predictors. Its self-employment predictor includes ordinary and SSTB Schedule C income. Return-level Boolean outputs use an explicit incidence capacity of one; that is not a physical person count.

The local PUF59/full65 regression run passed 128 ordinary tests. Those tests execute real weighted QRF fits, sequential target conditioning, draws, finalization, whole-Population preservation and required cache replay. They include excluded mortgage incumbent preservation and source-specific predictor checks. The run took 93.9 seconds and peaked at 622.7 MB RSS. These are invented mechanism fixtures, not held-out fit-quality or launch acceptance.

Current survey financial completion passed 26 ordinary tests, including 15 ASEC demographic projection controls. The financial graph uses original ASEC design weights to fit interest, dividends and capital-gains outcomes for native ACS people. It preserves ACS observed wage/self-employment amounts and source-owned age-15 universe zeros. A draw is shared with its clone by source identity. The graph retains typed preparation, allocation and context evidence while separating the donor branch from survey allocation, avoiding a structural dependency cycle. Cold execution and required replay passed with the actual fitted model artifacts. The combined run took 121.4 seconds and peaked at 524.0 MB RSS.

The source projection records the distinct survey observation, income and price periods. ASEC sex and state come from the retained original person and household sources, with allocation flags and unknownness preserved. Attaching those demographic projections to the graph is a separate pending integration step. The current financial model conditions on age and two earnings fields; sex/state conditioning and held-out fit quality remain explicit modeling work.

Social Security handling remains a launch blocker. The historical PUF59 donor transports its total through the retirement component with modeled zero carrier columns for the other components. Those zeros do not establish absent recipient benefits. Its reconciliation code can treat unknown recipient components as zero and use equal component shares. The explicit `PUF55_SURVEY_SS` profile now excludes these four outcomes from PUF enrichment and conditions the remaining 55 outcomes on an additional Social Security total predictor. It preserves existing survey component fields, including unknownness. This profile is not yet wired into the native build.

The new profile requires a known, finite, nonnegative `tax_unit.puf_conditioning_social_security_total` supplied by an upstream source owner. It does not infer that total from unknown components or move it between entity grains. All eight historical PUF59 predictors retain their order, with the total appended as predictor nine. The profile has a distinct train/apply phase and retains return-level incidence-capacity validation. Historical full65/PUF59 defaults and their 65/59 outcome contracts remain unchanged.

Eleven new controls pass with real 55-target QRF fits, finalization, raw-draw validation and required cache replay. They check exact Social Security incumbent bit patterns, including signed zero and NaN, and refuse changes to the conditioning total after fitting. The run took 28.5 seconds and peaked at 631.6 MB RSS; receipt SHA256 is `8d5beacf15d8eb7bf568705d4e12bd71d810f3e748571b669a6ec44921ebf0cf`. All 128 historical profile cases also pass against the identical production module. An earlier combined run retained a failure in the new test's comparison of Boolean storage types; the corrected eleven-case run preserves exact values and missingness while treating the requested PUF Boolean output's nullable storage explicitly. Native recipient total qualification, beneficiary completion and whole-population PUF55 attachment remain separate pending checks. The native donor projection now passes as described below.

The additive current-source qualifier now verifies original ASEC total, reason and allocation literals against the live survey preparation and retained original member. It also binds the ACS 2024 issuance and SSP price adjustment. Both surveys preserve under-15 question-universe unknownness. ASEC may record combined family payments on one person's report; a reason code is not a separately observed component amount or an assignment to individual beneficiaries. See [Social Security source semantics](survey-social-security-source.md) for the source contract, judgment boundaries and unresolved model work.

The qualifier and numerical basis pass 52 controls: 50 numerical/literal cases and two actual maintained source issuances on invented original inputs. They include reordered source identities, ambiguous reasons, allocation flags, exact requalification, unknownness and no input mutation. The corrected run took 67.6 seconds and peaked at 428.9 MB RSS; the receipt SHA256 is `2b9d8e45cec11a6ebda2ccc01c5f12374e794ec0c006ba6525f082fd6454049b`. An earlier run's two source tests exposed an incorrect capture-budget argument; the corrected caller uses the source owner's existing decoded-body bound while retaining exact CSV byte and digest checks. No native Social Security completion or beneficiary allocation has been accepted.

The combined current source also passes all **225 component integration controls**: 34 shared geography cases, 52 survey Social Security cases, 11 PUF55 cases and all 128 historical full65/PUF59 cases. The exact combined run completed in 205.6 seconds at 687.8 MB peak RSS with no skips or unexpected access refusals. Before/after/current source, model-source and code-resource checks all agree; the maintained checkout matches the tested source map. Receipt SHA256: `63ef0ad0602a66d40f0736490d4c1e29f57723ba938c58d317802b387f947a87`. This remains an invented-input component integration, not a native population certification.

The consolidated source passes a 314-case selected integration and identity suite, including the 128 profile cases, 26 financial/demographic cases, two canonical/growth wrappers, five metadata regressions and 153 identity controls. A subsequent diagnostic-option reconstruction correction passes all 16 current seven-node age-development tests, including late-mutation refusals. All source, model-source and resource pins survived external postchecks for each run. See the [review guide](us-launch-review.md) for exact revisions, receipts and exclusions.

The new canonical59-to-PUF55 donor adapter also passes the maintained invented-source construction and artifact wrapper on 64 and 2,048 returns. It verifies the supplied artifact identity and the exact Social Security carrier convention before projecting the grown total into predictor nine. All 55 retained outputs, source RECIDs and weights survive unchanged. Internally rehashed artifacts with altered carrier semantics refuse. The corrected wrapper passed in 9.4 seconds at 486.9 MB peak RSS; receipt SHA256: `db3e9d8aa91863973de5d3201efcbabe31cbc1067c8c3f7fd72be79d887ffc27`. The initial knownness-column mismatch is preserved as a failed control; the fix supplies the owner's exact nonstructural column roster. This adapter grants no source authority. A subsequent separately guarded native projection passes for all 207,692 returns, retaining every target, source RECID, weight and incidence capacity and appending the modeled Social Security total as predictor nine. It took 11.05 seconds at 1.43 GB peak RSS. The original canonical artifact remained unchanged; source/control/model/code checks agree, with no unexpected access refusals. Receipt SHA256: `ce09455947f0efbca16014a23baa19037ac6450876c639568af6584c81bfc45c`. The earlier attempt passed its value assertions but lacked collection accounting and was refused by the guard; that failure is preserved. This native projection ran in memory and produced metadata only. It did not fit a recipient model or attach results to the survey population.

The genuine seven-node survey age-development run now passes at the unchanged 1/1000 sampling fraction. It prepares 1,584 households and 3,464 people (ACS: 1,529 households/3,324 people; ASEC: 55/140), then retains 3,168 household records and 6,928 person records after the support clone. Export/readback, final source-owner and target checks completed under the frozen `a9895d8a5` source; the final manifest was written only after those checks. All 473 source/control/runtime and 12 code-resource postchecks agree, with no unexpected access refusals. The run took 4,268.8 seconds and peaked at 8.26 GB RSS. Receipt SHA256: `2aa76b5daca7bb0df1ac85d4db23d6b5482073d31d437418d46442ef01b002ca`. This run does not include the subsequent Social Security, PUF55 or atomic-geography integration.

This is accepted native age-development evidence, not a national or congressional-district release, native enrichment fit-quality result or country-engine result. Complete enrichment, geography, calibration and release-export acceptance remain pending.

The PUF55 profile now also passes three invented whole-Population attachment
controls. They execute all 55 fitted outcomes through the actual graph, typed
artifact loader and materialized population verifier, followed by Frame storage
and required replay. Every non-owned field, Social Security bit pattern and
unknown cell, membership, design anchor and household weight remains intact.
Two negative cases reject changed Social Security values and unknown-to-zero
conversions even when predictor matrices remain identical. The run took 21.03
seconds at 631.3 MB peak RSS; receipt SHA256:
`ad8519d867279ebce9ee78bf8f495f3799afb45ec1b3fcda1be649f8a129af30`.
All 482 source/control, 5,983 model-source and 12 code-resource postchecks agree,
with no skips or unexpected access refusals. A subsequent independent review
required a direct finalizer comparison, because the attachment and its verifier
share a column-construction helper. The strengthened three-case run also passes:
all 55 attached outputs agree with the direct finalizer after independent entity
ID alignment, on cold and required replay. It took 20.73 seconds at 630.5 MB;
receipt SHA256 is
`e3a4d548f0209cfacd7d9e3beddb63c1b46204a3168eb9869d2b2eafe9cf2291`.
All 486 source/control, 5,983 model-source and 12 code-resource postchecks agree.
Native recipient qualification, measurement comparability and actual survey
attachment remain pending.

The observed-geography source qualifier passes 11 invented original-source
controls. It binds current ASEC state and ACS state/PUMA to literal source
household keys, preserves unknown locations, and keeps draw keys stable across
sampling fractions. The initial ten-case run correctly refused a smaller sample
with no positive ASEC support; its corrected fixture supplies positive support
before issuance. Review also added a receipt-digest check against mutation of
the detached ASEC projection. The passing run took 182.50 seconds at 430.3 MB;
receipt SHA256 is
`1900801b1bbcea89a3747f656410e500b1328519694c3351100562bba61cab08`.
The 484 source/control, 5,983 model-source and 12 code-resource postchecks agree,
with no skips or unexpected refusals.

The observed-geography graph node separately passes 14 controls over the same
kind of invented original sources. Actual three-node execution and required
replay preserve inputs, weights and unknown geography. Declaration, source
receipt, typed-artifact and final-mutation cases refuse invalid output. The run
took 319.44 seconds at 439.1 MB; receipt SHA256 is
`28a6b5c58a258c7b3129a48054a3360647c788f8b76d781cc3bc8fc35a73f997`.
All 486 source/control, 5,983 model-source and 12 code-resource postchecks agree.
Full geography-before-clone composition, native block-source integration and
complete calibration remain separate work.
