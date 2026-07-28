# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python research codebase focused on FedAgg-CVV experiments.

- Core experiment pipeline: `experiment_runner.py`, `simulate.py`, `trainer.py`, `aggregator.py`
- Data/model utilities: `data_utils.py`, `models.py`, `network_model.py`, `cache_system.py`, `distillation.py`
- Analysis/reporting: `summarize_results.py`, `experiment_result` (historical outputs)
- Quick behavior check: `experiment_example.py`
- Local data directory: `dataset/` (ignored by Git)

Keep new modules at repo root unless a clear subpackage boundary is introduced.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install PyTorch CUDA 12.1 stack.
- `pip install -r requrements2.txt`: install analysis/runtime extras (`numpy`, `pandas`, `matplotlib`, `tqdm`).
- `python experiment_runner.py --dataset <path> --dataset-name cifar10 --rounds 2 --epochs 1 --batch-size 64 --seeds 0 --scales 2x6 --outdir experiment_results_smoke`: run a smoke experiment.
- `python summarize_results.py --indir experiment_results_smoke --latex-ready`: aggregate seed runs and emit summary CSVs.
- `python experiment_example.py`: fast cache/gating sanity run.

## Coding Style & Naming Conventions
Use Python 3 with PEP 8 defaults: 4-space indentation, snake_case for functions/variables, PascalCase for classes, and explicit type hints for new public functions. Keep CLI flags kebab-case (for example `--delta-v-list`) and match existing metric key naming when adding CSV fields.

No formatter/linter is configured in-repo; keep style consistent with surrounding files and avoid large unrelated rewrites.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with:

1. `experiment_example.py` for cache/sync behavior.
2. A short `experiment_runner.py` smoke run (`--rounds 2 --epochs 1 --seeds 0`).
3. `summarize_results.py` on produced CSV output.

When fixing bugs, include the exact command used to reproduce and verify.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects (English or Chinese), e.g. `fix bugs`, `update mac parameters`, `clean path`. Follow that pattern:

- One focused change per commit.
- Subject line should describe behavior change directly.
- In PRs, include: purpose, key commands run, affected datasets/strategies, and before/after metrics (plus plot screenshot if visualization changed).

Do not commit `dataset/`, model weights, or generated experiment artifacts.
