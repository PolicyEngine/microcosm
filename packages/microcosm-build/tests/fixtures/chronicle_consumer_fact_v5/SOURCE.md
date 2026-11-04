# Chronicle consumer_fact.v5 fixtures

Both files come from PolicyEngine/chronicle at `44eebe0378cd4aa708e283837e5a4ed1f7b18ef0`,
the merge of PolicyEngine/chronicle#377, which introduced `chronicle.consumer_fact.v5`.
Neither file differs at Chronicle `main` `28064d037562b47377193a5b75f4de120d3768a0`
(2026-11-04).

- `consumer_fact.v5.schema.json`: `policyengine_chronicle/schemas/consumer_fact.v5.schema.json`,
  byte for byte (sha256
  `acd33a091cd8d192be5d264e3e94c13e00afb6fd1972e57d4576ac55a54f6970`). Chronicle ships a
  byte-identical copy at `docs/schemas/consumer_fact.v5.schema.json`. This digest is the
  `consumer_fact_schema_sha256` a Chronicle consumer artifact built from v5 rows declares.
- `consumer_facts.jsonl`: Chronicle's sample consumer rows, `chronicle/fixtures/consumer_facts.jsonl`
  (sha256 `450a22c689b7a086cfc03522aa4492d491efc6875a4661dbd851a2015506897a`): IRS SOI
  Publication 1304 Table 1.1, tax year 2023, all returns, for the return count, adjusted gross
  income and total income tax. Each row carries Chronicle's label for the adjusted gross income
  value `all` ("All returns, total") in `dimension_value_labels` and
  `layout.groupby_value_label`, and the package's own wording ("All returns") in
  `package_dimension_value_labels`. Two changes from Chronicle's file: the raw-storage bucket
  name in `source.raw_r2_bucket` and `source.raw_r2_uri` is replaced by the placeholder
  `example-raw-bucket`, and each row is written with sorted keys. Every row still validates
  against the schema above.
