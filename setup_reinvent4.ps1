# ============================================================================
# setup_reinvent4.ps1
# One-shot setup: clone REINVENT4, create conda env, install all deps.
# Run from:  C:\Users\aelsamma\Desktop\Projects\
# Usage:     .\setup_reinvent4.ps1              (CPU-only)
#            .\setup_reinvent4.ps1 -Processor cu126   (CUDA 12.6 GPU)
# ============================================================================

param(
    [string]$Processor = "cpu"   # cpu | cu126 | cu118 | xpu
)

$ErrorActionPreference = "Stop"
$ProjectsDir = $PSScriptRoot
if (-not $ProjectsDir) { $ProjectsDir = (Get-Location).Path }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  REINVENT4 Setup for PGK2-Selective Ligand Generation"      -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Projects dir : $ProjectsDir"
Write-Host "Processor    : $Processor"
Write-Host ""

# ── Step 1: Clone REINVENT4 ──────────────────────────────────────────────────
$R4Dir = Join-Path $ProjectsDir "reinvent4"
if (Test-Path $R4Dir) {
    Write-Host "[1/6] reinvent4/ already exists — skipping clone." -ForegroundColor Yellow
} else {
    Write-Host "[1/6] Cloning REINVENT4 (shallow clone)..." -ForegroundColor Green
    git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git $R4Dir
    Write-Host "      Done." -ForegroundColor Green
}

# ── Step 2: Create conda environment ─────────────────────────────────────────
$EnvName = "reinvent4"
$EnvExists = conda env list | Select-String "^$EnvName\s"
if ($EnvExists) {
    Write-Host "[2/6] Conda env '$EnvName' already exists — skipping create." -ForegroundColor Yellow
} else {
    Write-Host "[2/6] Creating conda env: $EnvName (Python 3.10)..." -ForegroundColor Green
    conda create --name $EnvName python=3.10 -y
    Write-Host "      Done." -ForegroundColor Green
}

# ── Step 3: Install REINVENT4 + dependencies ─────────────────────────────────
Write-Host "[3/6] Installing REINVENT4 into env '$EnvName' (processor=$Processor)..." -ForegroundColor Green
Push-Location $R4Dir
conda run -n $EnvName python install.py $Processor
Pop-Location
Write-Host "      Done." -ForegroundColor Green

# ── Step 4: Install unimol-tools into reinvent4 env ──────────────────────────
Write-Host "[4/6] Installing unimol-tools into env '$EnvName'..." -ForegroundColor Green
conda run -n $EnvName pip install unimol-tools
Write-Host "      Done." -ForegroundColor Green

# ── Step 5: Locate prior model ───────────────────────────────────────────────
Write-Host "[5/6] Locating REINVENT4 prior model..." -ForegroundColor Green
$PriorPath = conda run -n $EnvName python -c @"
import reinvent, os
p = os.path.abspath(os.path.join(reinvent.__path__[0], '..', 'priors', 'reinvent.prior'))
print(p)
"@
Write-Host "      Prior model path: $PriorPath"

# Patch the TOML config with the correct prior path
$TomlPath = Join-Path $ProjectsDir "reinvent4_configs\pgk2_selective_rl.toml"
if (Test-Path $PriorPath) {
    $PriorFwd  = $PriorPath -replace '\\', '/'
    $TomlContent = Get-Content $TomlPath -Raw
    $TomlContent = $TomlContent -replace 'reinvent4/priors/reinvent\.prior', $PriorFwd
    Set-Content $TomlPath $TomlContent -NoNewline
    Write-Host "      TOML patched with actual prior path." -ForegroundColor Green
} else {
    Write-Host "      WARNING: prior not found at $PriorPath" -ForegroundColor Yellow
    Write-Host "      Download from: https://doi.org/10.5281/zenodo.15641296" -ForegroundColor Yellow
    Write-Host "      Then update prior_file / agent_file in reinvent4_configs\pgk2_selective_rl.toml" -ForegroundColor Yellow
}

# ── Step 6: Create output directory ──────────────────────────────────────────
Write-Host "[6/6] Creating reinvent4_outputs/ directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectsDir "reinvent4_outputs") | Out-Null
Write-Host "      Done." -ForegroundColor Green

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!"                                             -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Validate plugin:    .\run_pgk2_generation.ps1 -Mode test"
Write-Host "  2. Start RL run:       .\run_pgk2_generation.ps1 -Mode rl"
Write-Host ""
