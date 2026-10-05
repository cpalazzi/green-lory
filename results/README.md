# Green Lory results

The historical top-level scenario directories are preserved as legacy evidence.
They contain outputs from several model generations and must not be merged with
new reconciled runs.

New work uses immutable campaign paths:

```text
results/campaigns/<campaign>/<phase>/<scenario>/runs/<run-id>/<stage>/
```

Each run has its own manifest, explicit shard files, merged output, QA report,
and logs. The campaign launcher refuses an existing run directory and the merge
step never selects files by a "latest" wildcard.

No legacy directory should be moved or deleted until a corrected replacement
has passed coordinate, schema, accounting, and scientific QA.
