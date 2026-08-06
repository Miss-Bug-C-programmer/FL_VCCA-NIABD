# CI and environment

The repository has no hidden dependency on a GPU for its protocol tests.
Install the project requirements, then run:

```text
python -m compileall -q .
python -m pytest -q
python experiment_example.py
```

The v3 checks additionally cover serialized packet lineage, non-finite
fail-closed behavior, robust probability-space aggregation, NIABD warmup
consensus, chunk equivalence, heterogeneous model construction, checkpoint
integrity, and schema validation. Dataset files, model weights, and generated
experiment outputs remain local and are not committed.
