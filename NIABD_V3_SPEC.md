# NIABD v2.1 transactional specification

NIABD remains a server-side prediction-layer defense. Clients upload only
serialized proxy logits and lineage metadata. The Server owns the proxy labels,
the persistent memory, purification, probability aggregation, and student
distillation. No attack truth, triggered test loader, client model, parameter,
gradient, or raw private sample enters VCAA or NIABD.

The persistent memory is `prototype_mean[P,C]`, `prototype_variance[P,C]`, and
`thresholds[C]`. Each round reads the old state, computes teacher-level robust
deviations, purifies admitted logits, and then performs a transactional memory
update. Thresholds are updated only when the normal eligible set reaches
`minimum_consensus_teachers`; drift-only or frozen rounds do not change them.

Warmup and drift recovery use the same deterministic gate:

1. require `K >= minimum_consensus_teachers`;
2. form a robust median/MAD compact candidate set;
3. require the configured consensus fraction;
4. freeze on an unsafe or ambiguous split;
5. update only from the compact candidate set.

`proxy_chunk_size=0` selects the unchunked path. A positive value chunks the
teacher axis for scalar teacher metrics before global robust scoring, preserving
the same result while bounding temporary metric storage. Proxy shape and
`proxy_version` are fixed after initialization; a change requires an explicit
reset or a fail-closed error.

Every result row records `niabd_defense_available`,
`niabd_purification_applied`, `niabd_memory_updated`, and an explicit update
reason. Unmeasured values are `NaN`, `None`, or an empty lineage value rather
than fabricated zeros.
