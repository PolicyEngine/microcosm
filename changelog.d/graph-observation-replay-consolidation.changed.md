Add explicit raw-byte source codecs, preserve complete Frame metadata in the content store, represent unavailable gate artifacts and unreached consumers independently of cache hits, and isolate population observers from execution and persistence.

Frame store v1 objects are unavailable under the new metadata contract. Rebuild
them with `resume="forbid"`; automatic and required replay refuse them rather
than silently dropping metadata.
