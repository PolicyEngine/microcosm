PAYE RTI's payrolled employees aged 18 to 24 now bind, counted as the model's employees at the FRS interview (María's go of 18 November 2026, PolicyEngine/microcosm#1663).

- **The measure.** Both RTI rows now count people whose `employment_status` is FT_EMPLOYED or PT_EMPLOYED: a point-in-time stock, like RTI's daily-average monthly count. They used to count anyone with employment income in the year.
- **Why it binds.** The rows were held out because the annual count should read high by job turnover. On the November builds, though, the count at interview read at least as high as the annual one, and both sat below RTI.
- **Dependent children with pay.** The FRS keeps dependent children (under 16, or aged 16 to 19 in full-time non-advanced education) on its child table, as CHILD. RTI counts any of them on a payroll, so the rows now count them too where they are in work, from the pay the spine now carries for them (`CHILD.CHEARNS`).
  - The measure is: employment status FT_EMPLOYED, PT_EMPLOYED or CHILD, and `in_work` (pay or hours above zero).
  - At 18 to 24 this keeps the employees at interview, and adds the dependents aged 18 and 19 with a job, about half the row's shortfall on the November builds (Vahid's review of PolicyEngine/microcosm#1663; María's call of 18 November 2026).
  - The pay check sums their pay over the same people.
- **Under-18s** stay held out as the check on the employee base of the employer NICs relief for employees under 21. The FRS records few jobs held by under-18s, and no reweighting of its few records could close the gap.
