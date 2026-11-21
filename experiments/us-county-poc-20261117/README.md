# US County Files proof of concept: the documents for LWI (2026-11-17 run)

Tracking issue: microcosm#922, the LWI County Files. Plan: microcosm#1647.

`lwi-poc-report.md` is draft v0.2 (19 November 2026) of the proof-of-concept report for the Living Wage Institute. It states the fit and the limits of the 2024 county file that the 17–18 November run built: build `populace-us-2024-buildo-acs-local-f8b1039f00-20261118T011728Z`, SHA-256 `82314a3b5a6a95762faf6eb4f5b499ce56608ad6a39a27b75e11c75b4945bd3c`. The draft is under review and has not been sent. v0.2 takes the corrections that part 2 of the methods draft lists (microcosm#1647 comment 7005741386) and describes the 23 release checks as `gate_summary.json` records them.

`lwi-poc-methods.md` is draft v0.1 (19 November 2026) of the methods documentation. It is part 1 of the methods draft that Max's agents posted on microcosm#1647 (comment 7005740491), edited for LWI: an appendix of build identifiers, settings and input digests replaces the links to GitHub comments, and flags, run-file names and imputation family names are out of the text. Its account of how donor households got their geography and their tax-detail records comes from the national build's code at `cae8640`; Max's agents are checking it against that build's run logs.

`schema_manifest.md` and `schema_manifest.json` are the schema documentation: `tools/us_release_schema_manifest.py`'s output for the file, with a notes section at the top of the Markdown (microcosm#1647 comment 7022836029).

- **Format.** The report and the methods documentation are `pe-doc` Markdown (PolicyEngine/document-templates). Each has the front matter, the three-line title block and source notes after the last rule. `pe-doc` has no report kind, so both use `kind: proposal` with `psl_foundation: false`.
- **Sources.** The numbers come from the run report on microcosm#1647 (comments 6982547034 and 6982549060) and from the release's own reports: `county_reliability.json` and `.csv`, `calibration_diagnostics.json` and `gate_summary.json`. The loader result comes from a test on 18 November with policyengine-us 2.2.1.
- **Delivery.** The file, this report, the methods documentation and the schema documentation are shared through the private Hugging Face dataset `policyengine/lwi-county-files`. The report refers to the other two documents by name, because a link into a private repository does not open for a reader who has only a download token.
- **Still open.** Max's agents' note on the 28 July national build (the donor geography, the tax-detail records and the targets the donor weights fit), the review, the PDF builds, and Max's go for the upload.
