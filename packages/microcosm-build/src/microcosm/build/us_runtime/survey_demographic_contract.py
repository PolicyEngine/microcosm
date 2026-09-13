"""Immutable shared declarations for currently unbound survey demographics.

These are the historical composed ASEC binding's exact recorded requirements.
That stage still refuses their canonical writes; a later source-qualified
producer must meet its own reviewed contract. Importing these declarations
loads no source reader, population operator, model, graph or country engine.
"""

#: Demographic columns a later calibration wants on both arms and which this
#: stage deliberately does not bind: each names the exact source column the
#: reviewed ASEC mapping needs, the mapping itself, and what the native ACS arm
#: uses instead. Nothing is imputed, aged or defaulted.
#:
#: Whether the prepared arm carries the source column is an **observed property
#: of the population in hand**, recorded per run rather than asserted: the
#: invented fixture parent has neither column, and the genuine prepared arm has
#: both. Presence is not a licence — binding either column still needs a
#: reviewed cross-arm mapping decision that this stage may not make — so what
#: fails closed here is an attempted **write** of one of these columns
#: (``ASEC_DEMOGRAPHIC_COLUMN_BOUND``), not the source column's presence. Keying
#: the refusal on presence would have refused every genuine run while proving
#: nothing about what this stage binds.
#: Each entry is (column, required source columns, diagnostic-only source
#: columns, the explicit source contract that would supply it, the mappings
#: deliberately not adopted as proof, the caveats on reading presence as
#: observation, the ACS arm's own observed mapping).
#: The required columns are the ones an *explicit* binding needs, which is not
#: the same as the columns some existing mapping happens to read: the root
#: adjudication of 2026-09-06 refused ``P_SEQ == 1`` as semantic proof of
#: headship and required sex to carry its allocation provenance, so those
#: mappings are recorded here as not adopted rather than as the requirement.
_UNBOUND_DEMOGRAPHICS = (
    (
        "is_female",
        ("A_SEX", "AXSEX"),
        (),
        "asec_demographic_source: A_SEX printed codes 1 = Male and 2 = Female, "
        "admitted only with AXSEX printed codes 0 = No change or 4 = Allocated; "
        "any other token leaves the person unbound",
        (
            (
                "cps_carried.derive_us_cps_carried_inputs (is_female = A_SEX == 2)",
                "maps every token other than 2 onto male, so an unprinted code "
                "would be admitted as male, and it reads no allocation flag",
            ),
        ),
        (
            "A_SEX reaches the prepared person roster as a carried source "
            "column, but its presence attests the column, not that every "
            "delivered token is one of the two printed codes",
        ),
        "SEX == 2 on the native ACS arm, whose pinned dictionary prints the "
        "same 1 = Male / 2 = Female convention",
    ),
    (
        "is_household_head",
        ("A_EXPRRP",),
        ("P_SEQ",),
        "asec_demographic_source: A_EXPRRP printed codes 1 = Reference person "
        "with relatives and 2 = Reference person without relatives, with "
        "exactly one such person per household and every household relationship "
        "token inside the printed named codes",
        (
            (
                "relationship_inputs (is_household_head = P_SEQ == 1)",
                "P_SEQ's dictionary entry labels no code as reference person, "
                "and the ordering statement behind it is expressly limited to "
                "the ASCII file while this arm is restored from the CSV member",
            ),
            (
                "A_FAMREL == 1",
                "a family relationship scoped to primary-family membership, "
                "which a subfamily reference person does not carry; household "
                "and family roles stay separate",
            ),
            (
                "asec_pool relationship recode (A_LINENO == 1)",
                "a separately assigned Basic-CPS roster line number with no "
                "source guarantee of agreement with any ASEC sequence",
            ),
        ),
        (
            "asec_pool._with_relationship_recode derives A_EXPRRP from "
            "A_LINENO == 1 whenever the locked input omits it, so the column's "
            "presence on a prepared arm does not certify an observed source "
            "value for that cohort",
            "the native ACS arm's A_EXPRRP is likewise derived by acs_pums "
            "from RELSHIPP, and its group-quarters households carry the "
            "nonrelative code 14 rather than any reference-person code",
        ),
        "RELSHIPP == 20 observed on the native ACS arm's housing units only; "
        "RELSHIPP 37 and 38 are the group-quarters populations and carry no "
        "observed reference person",
    ),
)
#: The population columns this stage records as unbound, and therefore may not
#: own on any node. Public because the measurement module's own declarations are
#: checked against it.
UNBOUND_DEMOGRAPHIC_COLUMNS = tuple(column for column, *_rest in _UNBOUND_DEMOGRAPHICS)
