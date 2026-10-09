The UK spine now sets policyengine-uk's `uc_is_in_startup_period` (uk-data#527, microcosm#1095). Nothing set it before, so the engine's default of False applied to everyone, and the Universal Credit minimum income floor applied from the first assessment period of every self-employed claim. policyengine-uk 2.100.0 already reads the input in `uc_mif_applies`.

The rule follows UC Regs 2013 reg 63(1) as substituted from 23 September 2020 (SI 2019/1152). DWP starts a 12-month start-up period when it finds a claimant in gainful self-employment, and the period is no longer limited to new trades.

A person is in the period when they are self-employed and either of these began less than 12 calendar months before interview (`INTDATE`):

- their benefit unit's UC claim (BENEFIT 95, `UCSTART`, written month/day/year);
- their first self-employed job (`SEJBLONG`, main job first).

Self-employed means any of these:

- `EMPSTATI` 3 or 4;
- a held self-employed `JOB` row (`ETYPE` 2-7, no `SEEND`);
- nonzero `SEINCAM2`.

One self-employed-job answer does not date the trade. When someone was self-employed all year (`SAMESIT` 2, or every `SDEMP` month 3 or 4) and does not describe a business (`JOBBUS` 2), a job under a year old is a new engagement in the same trade (ADM H4102 example 4).

The FRS cannot see two things, both inherited from uk-data's rule:

- **Earlier awards.** A re-claim after the floor applied, or a second period within five years, reads as a start-up period.
- **A move into the all-work-related-requirements group on an old claim.** This starts a period the flag misses.

**UC records with no claim date.** In FRS 2024-25, 2,114 of 2,402 UC benefit units (88%) carry `UCSTART`. The other 288 have no linked administrative record. For these, the claim recency is drawn, keyed on the benefit unit (`stable_identity_uniforms`, seed 0, salt `uc_is_in_startup_period`). The draw rate is the `GROSS4`-weighted share of self-employed people with a linked claim whose claim began inside the window, 0.324 (uk-data measured 0.320).

The draw is declared on the `frs_spine` stage as `assign_binary_from_rate`. Its notes now name it as the one identity-keyed draw at the root, because only this stage reads the raw claim, interview and job dates.

**Measured on the raw FRS 2024-25 spine at design weights.** 222 person records (488.2k) are in a start-up period, and no child is. Of these:

- 96 records (208.3k) are in benefit units reporting UC, including 17 records (39.6k) in units without a linked claim date;
- the rest are self-employed people outside UC whose trade is under a year old. The flag matters to them only if they claim.

**SPI-redrawn rows.** These keep their donor's flag only where their employment status or their imputed self-employment income still shows self-employment (`spi_income._refresh_uc_start_up_period`, beside the carer flag refresh). The graph declares the column as a root boolean and as a hidden SPI rewrite.

`person.uc_is_in_startup_period` joins the export-surface allow-list. The following are regenerated:

- `source_stages.json`;
- the H2 spine fixture, whose synthetic tabs now carry `INTDATE`, `UCSTART`, the self-employment job columns, `SAMESIT` and `SDEMP`, plus linked-recent, linked-old and unlinked UC rows;
- the coverage manifest's source-manifest pins;
- the mirrored gate digests.
