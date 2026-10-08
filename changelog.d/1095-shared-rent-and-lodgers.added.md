The UK spine marks the benefit units that share their household's rent, and records the rent boarders and lodgers pay the householder (pe-uk#2006, uk-data#511/#512/#506, microcosm#1095).

`liable_for_share_of_household_rent` holds for a later benefit unit in a shared household (`HHSTAT` 2) with any of:

- a rent share on its adults' records (`SRENTAMT`);
- Housing Benefit (`HBOTHAMT`);
- a linked UC record carrying a housing element (`UCHOUSEL`).

Conventional households are left out, because the survey does not say whether their later units owe the landlord or pay the householder.

`CVPAY`, the rent a boarder or lodger pays the householder, is recorded on the payer. It no longer counts as the payer's property income, the same line microcosm#1081 removes. It now feeds `rent_paid_as_boarder` (`CONVBL` 1, board and lodging) or `rent_paid_as_lodger` (anything else).

In FRS 2024-25 at `gross4` design weights:

- 103 shared households (319.9k weighted) hold 178 liable later benefit units (615.9k).
- 43 adult records (138.4k) pay the householder, fewer than 10 of them as boarders.
- £806.7m a year of `CVPAY` leaves property income.
