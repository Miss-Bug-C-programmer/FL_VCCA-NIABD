param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
Set-Location $repoRoot

python -m pip install -r requirements-dev.txt
if ($CpuOnly) {
    Write-Host "CPU-only validation environment selected."
}
$env:PYTHONPYCACHEPREFIX = Join-Path $repoRoot ".pycache_v3_setup"
python -B -m compileall -q .
python -B scripts/run_v3_matrix.py --config configs/main_backdoor_experiment.json
python -B scripts/verify_preserved_main.py
