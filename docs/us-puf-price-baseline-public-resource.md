# Authenticate the PUF price baseline’s public resource

The developmental PUF price baseline requires retained BLS bytes at every
baseline entry point. The factory verifies the whole body against its named
SHA-256 and reads exactly one `CUUR0000SA0` / `M13` record for each of 2015 and
2024. After removing BLS tab-field padding, the value text must be exactly
`237.017` and `313.689`. Numerically equivalent spellings, duplicate records,
missing records and malformed matching rows refuse. Empty and non-byte inputs
refuse, and the parser bounds a body to 16 MiB before copying or splitting it.

`require_price_baseline_contract`, `adapt_projection_to_money_view`, and
`price_baseline_invariants` also require the body. They check it against the
compiled contract’s own resource digest. A contract assembled through the
generic growth compiler therefore cannot bypass resource verification on its
way into baseline adaptations or measurement receipts.

```python
body = retained_bls_path.read_bytes()
contract = compile_price_baseline(
    packaged_agi_projection().fields,
    source_url=retained_retrieval["source_url"],
    resource_sha256=retained_retrieval["sha256"],
    retrieved_utc=retained_retrieval["retrieved_utc"],
    resource_bytes=body,
)
require_price_baseline_contract(contract, resource_bytes=body)
decoded = decode_agi_projection(
    payload,
    packaged_agi_projection(),
    expected_status_artifact_sha256=status_payload_sha256,
)
adaptation = adapt_projection_to_money_view(
    decoded, status, contract, resource_bytes=body
)
grown = apply_puf_growth(adaptation.money_view, contract)
weights = apply_puf_design_weight_growth(
    adaptation.design_weight.astype("float64"), contract
)
measured = price_baseline_invariants(
    adaptation, grown, contract, resource_bytes=body,
    design_weight_after=weights,
)
```

The source body is supplied explicitly; the library performs no filesystem or
network read. `verify_cpi_resource_bytes` returns the measured digest, byte
length and selected record texts for a runner’s receipt. A runner must retain
the body alongside its retrieval evidence, because the public endpoint changes
as BLS releases new data. URL and retrieval time remain declared metadata;
hashing proves consistency with the named bytes, not where or when they were
retrieved. The development authority remains ineligible for release.

Unit tests use fabricated BLS-shaped bodies and their computed digests. They do
not represent authentic downloads. The local invented end-to-end run instead
verifies the actual retained 2,685,811-byte body with SHA-256
`9baac9eb2fd78f4510a7edeb2c2d23abfbe0c4d4482febe54f415852e19dddf7`;
all PUF records in that run remain invented. These checks do not execute or
approve a genuine PUF projection, a donor fit, or a release.
