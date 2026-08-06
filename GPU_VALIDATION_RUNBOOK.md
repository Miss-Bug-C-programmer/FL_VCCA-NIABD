# GPU/CoreX Validation Runbook

This runbook is for a real CUDA/CoreX host. It is not evidence that the
commands were executed on this CPU-only workstation. Use a new output root and
never overwrite `results`, `experiment_results_main_backdoor`, or
`cifar10_claim_validation`.

Set the following variables to local paths before running. The formal JSON is
read-only and must have the same SHA-256 before and after the matrix.

## PowerShell

```powershell
$py = "D:\conda_envs\receiversync-viz\python.exe"
$out = "experiment_results_main_backdoor_niabd_v2"
$datasets = "dataset_roots.json"
$config = "configs\main_backdoor_experiment.json"

& $py -c "import sys, torch; print(sys.version); print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda=', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
if ($LASTEXITCODE -ne 0) { throw "device check failed" }
Get-Content $datasets
Get-FileHash $config -Algorithm SHA256
& $py -m compileall -q .
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
& $py scripts\verify_preserved_main.py
if ($LASTEXITCODE -ne 0) { throw "preservation failed" }
& $py -m pytest -q --basetemp .pytest_tmp_niabd_fix
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
& $py -m pytest -q tests\test_niabd_v2.py --basetemp .pytest_tmp_niabd_fix
if ($LASTEXITCODE -ne 0) { throw "NIABD mechanism tests failed" }
```

Run a bounded CPU/synthetic check on any host before spending GPU time:

```powershell
& $py -m pytest -q tests\test_niabd_v2.py --basetemp .pytest_tmp_niabd_fix
& $py scripts\run_main_backdoor_matrix.py --dataset-roots $datasets --config $config --outdir $out --dry-run
```

Run clean and attack controls with the normal production CLI. Replace each
`<DATASET_ROOT>` and keep every output directory versioned:

```powershell
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method baseline --attack none --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\clean\baseline\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method niabd --attack none --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\clean\niabd\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method baseline --attack badnets --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\badnets\baseline\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method niabd --attack badnets --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\badnets\niabd\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method baseline --attack dba --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\dba\baseline\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method niabd --attack dba --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\dba\niabd\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method baseline --attack blend --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\blend\baseline\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method niabd --attack blend --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\blend\niabd\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method baseline --attack dynamic --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\dynamic\baseline\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method niabd --attack dynamic --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --outdir "$out\cifar10\dynamic\niabd\seed_0"
& $py experiment_runner.py --dataset <DATASET_ROOT> --dataset-name cifar10 --method vcaa-niabd --attack badnets --runtime process-semi-async --rounds 50 --epochs 1 --batch-size 64 --num-clients-list 20 --seeds 0 --partition-schemes dirichlet --device cuda --server-device cuda --client-device cpu --outdir "$out\cifar10\badnets\vcaa-niabd-process\seed_0"
```

The triggered-no-poison control must use the existing attack/data path and a
matching output identity; do not alter attack code or labels. Compare it with
attacked Baseline and classify each attack as `attack established`, `attack
weak`, or `attack not established` before making a defense claim.

## Bash

```bash
PYTHON=/opt/conda/bin/python
OUT=experiment_results_main_backdoor_niabd_v2
DATASETS=dataset_roots.json
CONFIG=configs/main_backdoor_experiment.json

"$PYTHON" -c 'import sys, torch; print(sys.version); print(torch.__version__); print("cuda_available=", torch.cuda.is_available()); print("cuda=", torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU")'
cat "$DATASETS"
sha256sum "$CONFIG"
"$PYTHON" -m compileall -q .
"$PYTHON" scripts/verify_preserved_main.py
"$PYTHON" -m pytest -q --basetemp .pytest_tmp_niabd_fix
"$PYTHON" scripts/run_main_backdoor_matrix.py --dataset-roots "$DATASETS" --config "$CONFIG" --outdir "$OUT" --dry-run
```

## Formal matrix and completion checks

The dry-run must print exactly `3 x 4 x 4 x 5 = 240` unique jobs and end with
`tiny-imagenet-200 + dynamic + vcaa-niabd + seed 4`. Only after the dry-run
passes, run the single-card formal matrix with `--resume`, redirecting stdout
and stderr to a versioned log and checking the process exit code. Preserve the
pre-run and post-run formal-config hashes.

```powershell
& $py scripts\run_main_backdoor_matrix.py --dataset-roots $datasets --config $config --outdir $out --resume *> "$out\matrix.log"
if ($LASTEXITCODE -ne 0) { throw "formal matrix failed" }
& $py scripts\check_result_completeness.py --indir $out --config $config
if ($LASTEXITCODE -ne 0) { throw "result completeness failed" }
& $py scripts\collect_main_backdoor_results.py --indir $out --out "$out\main_backdoor_mean_std.csv" --expected-runs 240
if ($LASTEXITCODE -ne 0) { throw "result collection failed" }
Get-FileHash $config -Algorithm SHA256
```

The formal run must also report final clean ACC, best clean ACC, final BASR,
attack-window mean/peak/AUC, peak round, post-attack recovery, clean-control
drop, teacher utilization, memory update rate, freeze rate, suppression and
95% confidence intervals. Old and repaired result roots must never be joined.

## Status on the current workstation

No CUDA/CoreX device is available here. All GPU checks, formal CIFAR-10,
CINIC-10, Tiny-ImageNet training, attack controls, process GPU execution, and
the 240-run execution are **未验证** until a real target device runs the
commands above.
