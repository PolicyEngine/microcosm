# SSI asset-gradient validation evidence

These rows are reported diagnostics. They do not choose the slope, intercept,
asset transform, or SSA targets. The machine-readable register is
[validation.json](validation.json). Its `model_result` fields stay null when the
available SSI fixture cannot reproduce the study's population, program and
follow-up period. Null means not evaluated, never a passing result.

The repository already has a reform-validation suite in
`packages/microcosm-build/src/microcosm/build/us_runtime/reform_validation.py`.
It scores weighted policy budget changes against published fiscal estimates,
with additional baseline total and rate comparisons. Its released-frame
simulation contract does not directly replay MSP's 24-month interrupted time
series, SNAP's state-year BBCE study, or an unidentified causal SSI 1985–89
enrollment response. These evidence-register rows therefore remain reported
diagnostics, pending matching population, program and study-period bridges;
they are not added as invented annual budget gates.

## CBPP coefficient cross-check

Kathleen Romig, Luis Nuñez and Arloc Sherman (2023), *The Case for Updating SSI
Asset Limits*, updated September 20, Appendix p. 14, report a **−0.04** adult
(18–64) probit coefficient on the natural logarithm of countable assets. The
outcome combines SSI and SSDI participation among income-eligible respondents
with observed key data; standard errors and the elderly equation are not
published. [Original report](https://www.cbpp.org/sites/default/files/6-26-23socsec.pdf#page=14)
and [HTML appendix](https://www.cbpp.org/research/social-security/the-case-for-updating-ssi-asset-limits).

Compare the SIPP estimate's sign and probability derivatives, with each link and
asset definition stated. A logistic coefficient on `log1p(liquid_assets)` cannot
be compared one-for-one with this probit coefficient. A constant link conversion
does not account for differing covariates, outcomes or samples. This cross-check
therefore has no numerical acceptance tolerance.

The completed primary SIPP adult fit is **+0.0333019549** (SE **0.0334195443**),
so its point-estimate sign disagrees with CBPP. At participation probability
0.5, the illustrative local derivative is **+0.0083254887** for the SIPP
logistic fit and **−0.0159576912** for CBPP's probit equation, with respect to
each model's own asset transform. These derivatives show the comparison;
they do not establish numerical equivalence or supply an acceptance tolerance.

CBPP Table 1, p. 11, publishes 154,000 new adult recipients for $10k/$20k and
115,000 elderly recipients for complete elimination, rounded to the nearest
1,000. It withholds the $10k/$20k elderly cell and warns that totals need not
sum. Its implied residual is excluded from the register. The published count
rows are comparisons, never estimation targets. [Table 1](https://www.cbpp.org/sites/default/files/6-26-23socsec.pdf#page=11).

## Medicare Savings Program natural experiments

Vicki Fung et al. (2026), *Health Services Research* 61(3):e70116, Table 1,
estimate interrupted time series with 24 months before and after expansion.
New York removed its MSP asset test in April 2008; Oregon did so in January
2016. Enrollment at month 24 exceeds the projected counterfactual by **3.9%**
and **3.7%**, respectively. Table 1 also gives the level and monthly-trend
estimates with their 95% confidence intervals, transcribed in the register.
[Paper and Table 1](https://onlinelibrary.wiley.com/doi/10.1111/1475-6773.70116).

The study includes Medicare beneficiaries entitled through age or disability,
including Traditional Medicare and Medicare Advantage. It lacks individual
assets and incomes; the enrollment response cannot be separated into newly
eligible people, previously eligible people or transfers from other assistance.
It is not a senior-only SSI experiment. Connecticut simultaneously expanded
income eligibility; its much larger response is excluded as an asset-only
validation row. A future MSP replay can report deviations against the stated
intervals for matching outcomes; no confidence interval is published for the
24-month percentage itself. [Methods §2.2 and Discussion](https://onlinelibrary.wiley.com/doi/10.1111/1475-6773.70116).

The present SSI gradient has **not been quantitatively evaluated on these MSP
experiments**. Reusing an SSI propensity as an MSP enrollment model would require
an independently justified program bridge. The paper supports retaining a
modest-response comparison, not importing a new eligibility-conditioned claiming
factor into the data layer.

## Additional evidence and limits

Xingguo Wang, Pourya Valizadeh, Rodolfo M. Nayga Jr., Henry L. Bryant and Bart L.
Fischer (2026; first published November 11, 2025), *Journal of Policy Analysis and
Management* 45(1):e70063, Table 3, column 2, estimate a **15.34%** BBCE effect on
per-capita SNAP participation (SE **2.07 percentage points**). This is the
heterogeneity-robust Callaway–Sant'Anna specification with covariates, 1,071
state-year observations for 1996–2016. BBCE changes income rules as well as asset
rules, and the sample covers all ages. The row is contextual, has no SSI tolerance,
and is not evaluated by an SSI-only fixture.
[Primary paper, Table 3](https://onlinelibrary.wiley.com/doi/10.1002/pam.70063).

SSA's POMS SI 01110.003 §A.2 supplies the exact 1985–89 resource steps, but no
causal enrollment estimate was sourced for them. Charles G. Scott (1989),
*Resources of Supplemental Security Income Recipients*, *Social Security
Bulletin* 52(8):2–9, p. 3, confirms the steps; it analyzes 15,093 recipient QA
responses from October 1986–September 1987, not enrollment responses to the
increases. Neither a before/after caseload difference nor a cross-section of
recipients identifies a claiming slope.
[POMS limits](https://secure.ssa.gov/apps10/poms.nsf/lnx/0501110003)
and [Scott (1989)](https://www.ssa.gov/policy/docs/ssb/v52n8/v52n8p2.pdf#page=2).

Numerical SSI count comparisons require a released-frame candidate probe and a
reform score using the same weights and preserved per-source draws. Neither is
provided by these cross-program rows. A small candidate basis can force the
SSA-count-solving intercept upward (#644), increasing extrapolated propensities.
Remaining discrepancies may be resource imputation (#424) or eligibility
measurement (#644), rather than claiming. No slope should be changed to close an
elderly CBPP count gap.
