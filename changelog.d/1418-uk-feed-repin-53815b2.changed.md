The UK Chronicle feed is re-pinned to `53815b2`, the merge of PolicyEngine/chronicle#394 (#1418). It adds SPI 2023-24 Table 3.10 and the OBR's October 2026 EFO beside March's, and it sits on the `0b721a4` merge of PolicyEngine/chronicle#392 (HMRC's ISA statistics). That is 1,511,063 rows, up 7,748, and the schema digest is unchanged.

A new vendored resource, `obr_savings_income_growth_facts.json`, carries the eight October 2026 EFO Table 4.5 savings income growth rates. The other vendored resources carry the same rows under the new feed identity.

On the national surface:
- the 13 SPI Table 3.7 interest rows grow by the OBR's savings income and move up 13.4%;
- Table 3.7's other income adds 13 rows, and Table 3.10 adds 16 (eight bands each of counts and amounts);
- `obr.winter_fuel_allowance` is pinned to March's expenditure table. October's table gives FY2025-26 as outturn, 0.02% above March's projection, and the selector would otherwise match both;
- no other compiled value moves. The other OBR targets window FY2024-25 and FY2025-26 into calendar 2025, and only March's edition carries FY2024-25.
