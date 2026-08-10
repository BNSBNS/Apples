# Archived — not comparable with current runs

These four files predate two metric changes and carry no `metrics_version` field:

- **`retrieval_recall` over-counted.** It regex-scanned the whole `search_policy`
  payload for `POL-\d{3}`, so a policy merely *cited inside another document's
  text* counted as retrieved. The corpus cross-references itself heavily, so the
  number reads high — and `hallucinated_citation` was structurally unreachable.
  `schema-fixed.json`'s `recall=1.0` is the figure quoted in README.md, and this
  is why that quote carries a caveat.
- **`expected_amount` was never scored.** Labelled on three cases, read by
  nothing, so a full-$300 credit on D-1008 passed.

They are also 1–2 case debug runs, not measurements. Kept as a record of what
the harness reported at the time; do not diff them against anything current.
