# PUF 2015 canonical donor and 2024 transport

These source owners turn the statistical 2015 PUF into an explicit return-level donor with 59 canonical outputs. They preserve raw fields, source status and documentary decisions separately from modeled values. The six detailed mortgage balance, interest and origination-year outputs belong to the downstream SCF stage.

```mermaid
flowchart LR
    A[Authenticated PUF typed source] --> B[Observed and derived return fields]
    B --> C[Eleven explicit baseline models]
    C --> D[QBI model in 2015 money]
    D --> E[Source-specific 2024 monetary transport]
    E --> F[Canonical donor with 59 outputs typed donor]
    F --> G[Eight-feature PUF59 donor interface]
```

The source codec retains all ordinary returns, including those missing the demographic supplement. Disclosure aggregates retain their source lexemes but are excluded from individual donors. Neither a demographic gap nor zero reported exemptions means that a source return is dropped.

## Count measurements

PUF 2015 MARS 2 combines joint and qualifying widow(er) returns. The matching predictor maps the survey's separate widow category into that combined class. It is named `puf_2015_filing_status_code`; generic model filing status is unchanged.

The corresponding `puf_2015_capped_return_size` equals one plus an extra unit for MARS 2 plus the sum of the four reported dependent-category counts. The source owner validates their disclosure caps by filing class. On the survey side, the matching measurement caps the reported dependent total before adding the filing-class units. The result is a statistical predictor, not a count of physical people or co-residents. Actual survey membership remains unchanged.

The separate `puf_person_incidence_capacity` is one for return-level modeled zero/one QBI incidence. It is used only to validate the canonical outcome representation and is excluded from the eight fitted predictors.

## Observed fields and explicit models

The decoder emits 33 directly mapped or algebraically derived baseline fields. Eleven additional baseline outputs explicitly model source gaps: a normalized interest split, a component-neutral Social Security carrier, desired contributions proxied by realized deductions, incomplete tuition and employee-expense proxies, and active-partnership earnings. These transformations retain their assumptions in the receipt.

The interest model uses the existing [IRS 2015 Table 2.1 workbook](https://www.irs.gov/pub/irs-soi/15in21id.xls). It normalizes the four published component amounts within the source AGI band. Mortgage interest, points and mortgage insurance share the available home-interest leaf; the investment component alone enters investment expense. Complementary rounding preserves the source total exactly. This does not create a mortgage balance, origination year or observed loan structure.

The QBI owner reproduces the archived data model's assumptions with RECID-keyed random draws, retaining a reusable full-cohort employee calibration. Its 15 added leaves and Schedule C replacement are modeled inputs. The statutory engine owns deductions and limits. Ordinary and SSTB Schedule C branches are summed for the donor's total self-employment predictor, matching the survey measurement.

## Source-specific monetary transport

The publisher's statistical 2015 convention is the input basis. Raw filing-year/month fields remain intact; the transform does not apply another row-year CPI adjustment.

Wages and modeled W2 amounts use the [SSA average wage index](https://www.ssa.gov/oact/cola/AWI.html). Social Security carriers use [SSA COLAs](https://www.ssa.gov/oact/cola/colaseries.html) received from January 2016 through January 2024. This transports fixed entitlements and does not predict changes in recipient composition.

Other income families use the observed 2015 and 2023 national amount/count cells of [IRS Table 1.4 and 1.4A](https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income), with positive and loss magnitudes kept separate. The 2023→2024 step uses the [BLS annual CPI-U series](https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202412.pdf), explicitly assuming stable real conditional amounts for that year. Expenses, UBIA and selected mixed model outputs use an explicitly named CPI proxy. Every monetary field has a declared rule; incidence fields are unchanged.

Partnership and S-corporation reporting counts overlap, so their dollar components are summed with the common all-return denominator. Dividend residuals use the ordinary-dividend reporting universe and do not subtract overlapping recipient counts.

The PUF short- and long-term capital fields remove the effect of carried losses. Published net-loss means that include carryovers cannot be substituted directly. The model uses current-year gross gain/loss components per all return as a declared proxy for signed net flows. The evidence retains both the carryover-adjusted reported net and component-difference net, including their unresolved residual. It does not attribute that difference to rounding without evidence.

The observed series are source facts. Their application to individual donor amounts, the related-family substitutions, and the last-year bridge are modeling decisions. They do not establish forecast accuracy or current reporting incidence. A same-cohort CPI-only sensitivity is available. Source design weights are divided from S006 hundredths once and are never grown.

## Validation status

The ordinary verification executes real source decoding, the QBI model, growth, the deterministic typed envelope and the PUF59 donor interface at 64 and 2,048 invented records. It checks all 59 outputs, sign/zero preservation, physical type and knownness refusal, exact interest conservation, SSTB/base identities, source immutability, stable full-fit subset replay, artifact byte/value replay and malformed-input refusal.

The 59 invented controls passed. The separately reviewed genuine construction then produced all 59 canonical outcomes for 207,692 ordinary returns in 23.07 seconds. The 64-return subset reproduced the full-cohort outputs when it reused the full QBI employee fit, and the typed donor passed byte/value replay through the canonical donor interface.

Independent review subsequently found that the version 1 envelope could attach
an old descriptive receipt to altered, domain-valid values. The version 2 codec
now checks the row-ordered identity/weight/source-feature digest and all 59 output
values against that receipt on encode and decode. It also pins the actual
interest-band facts, verifies the embedded growth recipe, requires canonical
JSON and physical integer row metadata, and makes CPI-sensitivity acceptance
explicit. The corrected source passed 80 invented controls, including the 21
new integrity cases. Formatting followed that run; the composed integration
checkout still needs its own execution checks.

The original genuine donor and evidence remain on version 1. A version 2 genuine
construction or repacking requires separate verification, and the new decoder
refuses old magic. Serialized receipt consistency does not authenticate an
issuer, establish execution or confer release status.

This result verifies source donor construction. Social Security component reconciliation, downstream SCF mortgage semantics, conditional-fit quality, recipient placement and calibration remain open release requirements. It does not certify a calibrated population or policy-analysis file.

## Evidence bundled with the source

The [national extract](evidence/puf2015-target2024/NATIONAL-GROWTH-EXTRACT.json) records the exact public workbook cells, units, hashes, family calculations and unresolved capital residuals. The [index transcript](evidence/puf2015-target2024/INDEX-VALUES.json) identifies the SSA and BLS values and retrieval sources; it is a transcription of retrieved values, not a hash of the original web pages. The [growth recipe](evidence/puf2015-target2024/GROWTH-RECIPE.json) assigns each monetary field a factor and interpretation.

[Public workbook locators](evidence/puf2015-target2024/PUBLIC-WORKBOOK-SOURCES.json) provide the four official URLs and exact sizes and hashes without local cache paths. The recipe also preserves hashes of the original local acquisition receipt and extraction script. Those two local files are outside this publication bundle; their recorded hashes do not imply that a reader received them.
