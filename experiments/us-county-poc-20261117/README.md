# US County Files proof of concept: the report (2026-11-17 run)

Tracking issue: microcosm#922, the LWI County Files. Plan: microcosm#1647.

`lwi-poc-report.md` is the first draft (v0.1, 18 November 2026) of the proof-of-concept report for the Living Wage Institute. It states the fit and the limits of the 2024 county file that the 17–18 November run built: build `populace-us-2024-buildo-acs-local-f8b1039f00-20261118T011728Z`, SHA-256 `82314a3b5a6a95762faf6eb4f5b499ce56608ad6a39a27b75e11c75b4945bd3c`. The draft is under review and has not been sent.

- **Format.** The draft is `pe-doc` Markdown (PolicyEngine/document-templates). It has the front matter, the three-line title block, `[[placeholders]]` that `pe-doc check` refuses, and source notes after the last rule. `pe-doc` has no report kind, so the draft uses `kind: proposal` with `psl_foundation: false`.
- **Sources.** The numbers come from the run report on microcosm#1647 (comments 6982547034 and 6982549060) and from the release's own reports: `county_reliability.json` and `.csv`, `calibration_diagnostics.json` and `gate_summary.json`. The loader result comes from a test on 18 November with policyengine-us 2.2.1.
- **Delivery.** The file is shared through the private Hugging Face dataset `policyengine/lwi-county-files`.
- **Still open.** The link to the methods documentation, the schema documentation, and the review.
