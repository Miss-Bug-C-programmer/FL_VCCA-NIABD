# Result schema v3

The machine-readable contract is emitted as
`result_schema_v3.json`. Every non-empty round and summary table carries the
schema version, VCAA/NIABD/aggregation algorithm versions, run class, attack
condition, transaction identity, student pre-update snapshot identity, teacher
counts, NIABD state flags, numeric-failure count, rollback reason, and optional
checkpoint identity.

Use:

```text
python scripts/validate_v3_results.py --indir <output-directory>
```

Validation is fail-closed for missing columns, mixed lineage, empty run IDs,
non-contiguous rounds, and a modified manifest hash. Strict attack-window AUC
is exported only when measured attack rounds are contiguous; otherwise it is
`NaN` and the continuity flag is `0`.
