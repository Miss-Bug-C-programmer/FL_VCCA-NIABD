# v3 experiment runbook

The legacy formal configuration remains unchanged and still expands to
`3 datasets × 4 attacks × 4 methods × 5 seeds = 240` runs. Use
`scripts/run_v3_matrix.py --config configs/main_backdoor_experiment.json` to
inspect that factorization without launching training.

For a short production-path check:

```text
python experiment_runner.py --dataset <path> --dataset-name cifar10 --rounds 2 --epochs 1 --num-clients-list 3 --seeds 0 --method vcaa-niabd --enable-niabd --enable-vcaa --run-class smoke --strict-numeric-checks --outdir experiment_results_smoke
python scripts/validate_v3_results.py --indir experiment_results_smoke
python scripts/compute_statistics.py --summary experiment_results_smoke/fedagg_run_summary_cifar10.csv
```

For process-semi-async, provide a runtime profile and keep server/client
devices explicit. Late packets remain eligible for VCAA; their source round,
receive time, consumed round, version lag, and knowledge age are exported.
Round consumption uses reserve → commit, with abort and client checkpoint
restore on a numeric rollback.

GPU validation is intentionally separate: the local baseline environment is
CPU-only, so CUDA claims must be verified on the target machine with the same
command and an exported manifest.
