The UK spine now sets policyengine-uk's `uc_is_in_gainful_self_employment` (uk-data#525, microcosm#1095). Nothing set it before, so the engine's own rule applied: any nonzero `self_employment_income` counted as gainful self-employment. Because the spine floors losses at zero, a trader at a loss or breaking even never met the Universal Credit minimum income floor, while an employee with a small side trade did. The input's only reader is `uc_mif_applies`.

The rule follows UC Regs 2013 reg 64(a), which asks whether the person carries on a trade as their main employment, and DWP's Advice for Decision Making. ADM starts from hours (H4031) but lets earnings outweigh them (H4034). A person is in gainful self-employment when either holds:

- their main job is self-employed (`EMPSTATI` 3 or 4), whatever its profit, since a trade at a loss or breaking even is still carried on (H4013, H4054, H4503);
- their side trade's profit (`SEINCAM2`) is above zero and above their employment income (`INEARNS`).

SPI-channel rows keep their donor's employment status but take new imputed pay and profit, and the draw does not read the status. So after the redraw they take the earnings route alone: a profit above zero and above pay. The main-job route returns when the SPI draw follows employment status (uk-data#529, microcosm#840).

The flag is fixed at the survey year's incomes. Uprating reprices pay and profit by different indices, so in a projected year a flagged side trade can earn less than the job, or the reverse, and the flag does not follow.
