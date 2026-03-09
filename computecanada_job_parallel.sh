#!/bin/bash
# =============================================================================
# computecanada_job_v2.sh — SLURM job for screen_enamine_v2.py
#
# Resources (tuned for 100k-600k compound partitions):
#   1 GPU  → UniMol embedding + SelectivityModel inference
#   12 CPU → parallel RDKit conformer generation (pool of 12 workers)
#   48 GB  → embedding dict + GPU buffers + multiprocessing overhead
#   2 hrs  → generous upper bound (~35 min expected for 500k)
#
# Usage:
#   sbatch computecanada_job_v2.sh
#   sbatch --export=CONFIG=configs/config_kinase_screen.yaml computecanada_job_v2.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f screening_v2_JOBID.log
# =============================================================================

#SBATCH --job-name=pgk_screen_v2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/screening_v2_%j.log
#SBATCH --error=logs/screening_v2_%j.err

set -euo pipefail

echo "=============================================="
echo "  screen_enamine_v2 — ComputeCanada GPU Job"
echo "=============================================="
echo ""
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "Memory   : $SLURM_MEM_PER_NODE MB"
echo "Started  : $(date)"
echo ""

# ── Load modules ──────────────────────────────────────────────────────────────
echo "Loading modules..."
module load cuda/12.6
module load python/3.10
echo "✓ Modules loaded"
echo ""

# ── Navigate to project directory ─────────────────────────────────────────────
cd "$HOME/enamine_screen" || {
    echo "ERROR: Cannot cd to ~/enamine_screen"
    exit 1
}
echo "Working directory: $(pwd)"

# ── Create logs directory if missing ─────────────────────────────────────────
mkdir -p logs

# ── Point UniMol at pre-downloaded weights (no internet on compute nodes) ─────
export UNIMOL_WEIGHT_DIR="$HOME/enamine_screen/weights"
echo "UniMol weights : $UNIMOL_WEIGHT_DIR"
echo ""

# ── Activate virtualenv ───────────────────────────────────────────────────────
echo "Activating virtualenv ~/envs/unimol..."
source "$HOME/envs/unimol/bin/activate" || {
    echo "ERROR: Cannot activate ~/envs/unimol"
    exit 1
}
echo "✓ Virtualenv active: $(which python)"
echo "  Python version : $(python --version)"
echo ""

# ── Select config (override via: sbatch --export=CONFIG=...) ─────────────────
CONFIG="${CONFIG:-config.yaml}"
echo "Config: $CONFIG"
echo ""

# ── Run screening ─────────────────────────────────────────────────────────────
echo "Starting screen_enamine_v2.py..."
echo "----------------------------------------------"

python -u screen_enamine_v2.py \
    --config "$CONFIG" \
    --n_workers "$SLURM_CPUS_PER_TASK"

RESULT=$?

echo ""
echo "=============================================="
echo "  Finished: $(date)"
if [ $RESULT -eq 0 ]; then
    echo "  Status  : SUCCESS ✓"
else
    echo "  Status  : FAILED (exit code $RESULT) ✗"
fi
echo "=============================================="

exit $RESULT
