Require exact integer identity for entity IDs and memberships even when the
selected PolicyEngine variable stores floats. Metadata validation rejects lossy
float32/float64 conversions without changing source IDs or ordinary amount rounding.
