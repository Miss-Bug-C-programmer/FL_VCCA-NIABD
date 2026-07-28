# Local validation report

This report records only checks actually executed while building this package.
It is not a paper-results report.

## Source preservation

Command:

```bash
python scripts/verify_preserved_main.py
```

Observed result:

```text
PASS: all 54 original main files are present; all 11 critical mechanism files are byte-identical to a9d9d58129ef460e85f9c70f424f8dcbd8494a70.
```

The byte-identical critical files include `vcaa.py` and `niabd.py`.

## Compilation and full regression tests

Commands:

```bash
python -m compileall -q .
python -m pytest -q
```

Observed final test result:

```text
58 passed
```

Before the additive backdoor changes, the downloaded main snapshot was also
compiled and its original suite was run successfully:

```text
46 passed
```

No original test file was deleted. New tests cover attack construction,
Dirichlet partitioning, new datasets, sync backdoor flow, process TCP/RPC
backdoor flow, BASR target-class exclusion, and the 240-job matrix.

## Formal matrix cardinality

Command:

```bash
python scripts/run_main_backdoor_matrix.py \
  --dataset-roots '{"cifar10":"/data/cifar10","cinic10":"/data/cinic10","tiny-imagenet-200":"/data/tiny-imagenet-200"}' \
  --dry-run > main_matrix.txt
wc -l main_matrix.txt
```

Observed result:

```text
240
```

The first job is CIFAR-10 / BadNets / Baseline / seed 0. The final job is
Tiny-ImageNet-200 / Dynamic / VCAA+NIABD / seed 4.

## End-to-end smoke execution

A small locally generated CINIC-10-shaped ImageFolder dataset was used only to
exercise the real code path. It contains random images and therefore cannot be
used as evidence of attack or defense effectiveness.

Actually executed paths included:

```text
Baseline + BadNets
Baseline + DBA
Baseline + Blend
NIABD-only + BadNets
VCAA+NIABD + Dynamic
```

The runs produced nonzero `poisoned_samples`, real serialized client logits,
student updates, BASR fields, AttackPlan JSON, and the expected defense CSVs.
For example, the three-round NIABD + BadNets smoke run recorded zero poisoned
samples before the configured attack start and 13 poisoned samples in each of
the next two rounds.

The random smoke data produced BASR 0 in these runs. This is expected evidence
that the implementation does not fabricate a successful attack. Formal
attack-viability and NIABD-effectiveness claims still require the real CIFAR-10,
CINIC-10, and Tiny-ImageNet-200 datasets on the target machine.

## process-semi-async attack path

The added integration test `tests/test_process_backdoor_runtime.py` was executed
as part of the 58-test suite. It uses the existing persistent spawned Client
processes and real localhost TCP/RPC path. It verifies client-local poisoning,
serialized logits, no Server client-model references, per-round BASR, and
nonzero poisoned sample counts.

## Not claimed as validated

This package does not claim that the final 240 real-dataset experiments have
already been run. In particular, the following remain target-machine work:

```text
real CIFAR-10 attack viability
real CINIC-10 attack viability
real Tiny-ImageNet-200 attack viability
five-seed final ACC/BASR statistics
A100 throughput/runtime results
statistical significance of NIABD improvements
```

Use `CODEX_LOCAL_ITERATION_PROMPT.md` for the required local continuation
workflow and the constraints against simplifying the system or weakening tests.
