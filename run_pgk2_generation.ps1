# ============================================================================
# run_pgk2_generation.ps1
# Run REINVENT4 PGK2-selective ligand generation with the surrogate model.
# Run from:  C:\Users\aelsamma\Desktop\Projects\
# Usage:
#   .\run_pgk2_generation.ps1 -Mode test    # validate plugin on known SMILES
#   .\run_pgk2_generation.ps1 -Mode rl      # full 2-stage RL run
#   .\run_pgk2_generation.ps1 -Mode tb      # launch TensorBoard only
# ============================================================================

param(
    [ValidateSet("test", "rl", "tb")]
    [string]$Mode = "rl"
)

$ErrorActionPreference = "Stop"
$ProjectsDir = $PSScriptRoot
if (-not $ProjectsDir) { $ProjectsDir = (Get-Location).Path }

# ── Environment variables ─────────────────────────────────────────────────────
$EnvName     = "reinvent4"
$ScoringDir  = Join-Path $ProjectsDir "reinvent4_scoring"
$OutputDir   = Join-Path $ProjectsDir "reinvent4_outputs"
$ConfigDir   = Join-Path $ProjectsDir "reinvent4_configs"

# Add our scoring plugin root to PYTHONPATH
$env:PYTHONPATH = "$ScoringDir;$env:PYTHONPATH"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  REINVENT4 PGK2 Selectivity — Mode: $Mode"                  -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Projects dir  : $ProjectsDir"
Write-Host "Scoring plugin: $ScoringDir"
Write-Host "Output dir    : $OutputDir"
Write-Host "PYTHONPATH    : $env:PYTHONPATH"
Write-Host ""

switch ($Mode) {

    "test" {
        Write-Host "Validating PGK2Selectivity plugin on known compounds..." -ForegroundColor Green
        $LogFile    = Join-Path $OutputDir "scoring_test.log"
        $ConfigFile = Join-Path $ConfigDir "pgk2_scoring_test.toml"
        Write-Host "Config : $ConfigFile"
        Write-Host "Log    : $LogFile"
        Write-Host ""
        conda run -n $EnvName --no-capture-output `
            reinvent -l $LogFile $ConfigFile
        Write-Host ""
        Write-Host "Results: $OutputDir\scoring_test.csv" -ForegroundColor Green
        Write-Host "Open CSV and confirm PGK2-selective compounds score > 0.7" -ForegroundColor Yellow
    }

    "rl" {
        Write-Host "Starting 2-stage RL for PGK2-selective ligand generation..." -ForegroundColor Green
        Write-Host ""
        Write-Host "Estimated runtime: ~4h (CPU, 100 mol/step, 1000 steps total)" -ForegroundColor Yellow
        Write-Host "Run 'tensorboard --logdir reinvent4_outputs\tb_pgk2' to monitor" -ForegroundColor Yellow
        Write-Host ""
        $LogFile    = Join-Path $OutputDir "pgk2_rl.log"
        $ConfigFile = Join-Path $ConfigDir "pgk2_selective_rl.toml"
        Write-Host "Config : $ConfigFile"
        Write-Host "Log    : $LogFile"
        Write-Host ""
        conda run -n $EnvName --no-capture-output `
            reinvent -l $LogFile $ConfigFile
        Write-Host ""
        Write-Host "RL run complete." -ForegroundColor Green
        Write-Host "Stage 1 CSV : $OutputDir\pgk2_stage1*.csv"
        Write-Host "Stage 2 CSV : $OutputDir\pgk2_stage2*.csv"
        Write-Host "Final agent : $OutputDir\pgk2_stage2.chkpt"
        Write-Host ""
        Write-Host "Post-process: filter rows where PGK2_selectivity > 0.7" -ForegroundColor Yellow
        Write-Host "Then re-score with: python screen_enamine.py --config <your_config>" -ForegroundColor Yellow
    }

    "tb" {
        Write-Host "Launching TensorBoard for REINVENT4 run..." -ForegroundColor Green
        $TBDir = Join-Path $OutputDir "tb_pgk2"
        if (-not (Test-Path $TBDir)) {
            Write-Host "WARNING: TensorBoard log dir not found: $TBDir" -ForegroundColor Yellow
            Write-Host "Start an RL run first (-Mode rl)" -ForegroundColor Yellow
        } else {
            conda run -n $EnvName tensorboard --logdir $TBDir
        }
    }
}
