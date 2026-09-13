# ASEC amount-domain contracts

The current income-routing, interest and child-support qualifiers read monetary
definitions from the pinned `asec_current_money_domains_v1.json` artifact.
Their public amount-entry functions return detached mappings. Routing and
interest cache immutable tuples privately; child support caches immutable JSON
bytes and reconstructs its nested evidence on each call. Replacing an entry,
clearing a returned mapping or editing nested child-support metadata cannot
change a later caller's definitions. The public mapping and evidence shapes
remain unchanged.

Each relevant field must occur exactly once and contain exactly one vintage for
the current income year. Its dictionary spelling, URL and digest must match the
qualifier's source pin. Missing or duplicate entries refuse qualification.

The interest total and child-support parsers accept unsigned integer literals.
They explicitly require the corresponding artifact domains to describe
nonnegative integer dollars, with the supported zero and NIU meanings, no
additional missing-value codes or excluded dollar values, and consistent
encoded, dollar and printed ranges. Bounds remain artifact-derived. Re-pinning
an artifact with an unsupported domain requires a parser change and its own
review; it cannot silently change the interpretation of those literals.

Before capturing the original source member, each qualifier compares its
relevant amount definitions with the actual retained current-money owner's
`MoneyDomain` entries. Names must be unique; entity, grain, native column,
bounds and zero semantics must agree. Interest checks the retained `INT_VAL`
total. Its additional ordinary and retirement-account components retain their
separate published definitions. Child support checks both `CSP_VAL` and
`CHSP_VAL`; routing checks its existing nine fields.

These checks preserve the existing source meanings. In particular, a
`CHSP_YN` answer describes an obligation, and neither a no-obligation answer nor
an NIU paid amount establishes zero voluntary payments. Receipt, reporting,
allocation and top-code classifications are unchanged. The changes add no
model, canonical engine input, source issuer or native-data admission.

Validation used invented mutations of the public domain artifact and actual
invented source-owner fixtures: 73 new domain/cache controls, 88 routing tests,
40 interest tests, 10 allocation tests and 40 child-support tests passed under
the existing bounded source guard. The new controls first reproduced 65
failures against the prior implementation. These results certify the tested
source contracts, not a native dataset or release.
