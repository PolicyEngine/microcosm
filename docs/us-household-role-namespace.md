# Separate carried headship from source-qualified roles

Native preparation preserves a carried `is_household_head` column as
`legacy_prepared_is_household_head` on its detached source copy. Every value,
missing bit and physical dtype is preserved. The original source Frame,
structural IDs, membership, weights, strata and mass history remain unchanged.
The preparation receipt records the versioned namespace rule and which sources
supplied the carried column. A preexisting reserved diagnostic name refuses.

This separates two meanings that previously occupied the same column. Legacy
ASEC preparation can derive headship from a within-household sequence. ACS
preparation supplies a dense relationship-derived boolean, including false for
group quarters. Neither establishes a source-qualified housing reference-person
role for every row. In particular, the source role is unresolved for ACS group
quarters, rather than an observed false answer.

The existing role operation now owns a fresh nullable canonical
`is_household_head`, using the original ASEC `A_EXPRRP` and ACS `RELSHIPP`
observations and their existing universe rules. It still refuses conflicting or
unsupported known canonical incumbents. It does not use the renamed diagnostic
to fill an unknown. `household_role_reconciliation` separately reports legacy
agreement, disagreement and known legacy values without qualified counterparts,
including counts by survey. Relocating a column must not hide these differences.

The rename happens at the existing authenticated preparation boundary, before
spine assembly and support cloning. The graph's same-dtype rewrite constraint
is unchanged. All preparation and descendant keys change with the source code
and receipt; an old checkpoint cannot confer a new live source qualification.

Tests preserve both dense and nullable legacy storage, reject collisions and
modified copies, retain disagreements in diagnostics, and pass the maintained
genuine source issuers over invented housing and group-quarters records. Existing
canonical-conflict refusals remain tested separately. These controls do not
certify actual-source counts, a full graph run or a published file. The final
model/export boundary must still handle any out-of-universe nullable role under
an explicit, checked applicability convention.
