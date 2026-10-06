The UK spine supplies policyengine-uk's person types (pe-uk#1896, uk-data#524/#486, microcosm#1095). Every FRS person is on the adult table or the child table:

- `is_claimant_or_partner` is adult-table membership, the benefit unit's head and any partner.
- `is_hbai_dependent_child` is child-table membership.

The engine stops inferring both from ages. The existing claimant derivation already refuses a benefit unit without one or two adult records.
