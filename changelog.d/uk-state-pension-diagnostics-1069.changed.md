Move the OBR State Pension line out of the UK fit and keep it, with DWP's forecast-table lines, as diagnostics beside the bound resident level (microcosm#1069 c4, rulings R2 and R4).

- **Why:** the OBR FY2025-26 forecast (£146.19bn) counts Great Britain plus pensioners paid abroad and leaves out Northern Ireland. It is a different quantity from the resident level that the DWP Stat-Xplore and DfC rows now bind.
- **What is attached:** `dwp.state_pension.amount` carries the exact OBR figure and DWP Spring 2026's Great Britain total (£146.07bn) and paid-abroad line (£5.63bn) as metadata only. `dwp.state_pension.recipients` carries DWP's caseload lines (13.204m, of which 1.088m are abroad).
- **How:** the diagnostic attachment, until now written for the CGT cash forecast alone, is driven by the contract's `diagnostic_references`. Each declaration names its metadata prefix and value unit.
- **Fail-closed:** `UK_REQUIRED_TARGET_DIAGNOSTICS` names the declarations each target requires, so a missing or misdirected one still refuses the target. The CGT diagnostic's metadata is unchanged.
- **Knock-on changes:** the incumbent `obr/state_pension` row moves to `registry_parity.excluded`, and the compile-parity receipts give it and the State Pension rows their ruled rationales. The UC support probes protect `dwp.state_pension.amount` in its place.
