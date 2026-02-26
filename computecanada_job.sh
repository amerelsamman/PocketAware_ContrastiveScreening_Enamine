#!/bin/bash

#SBATCH --job-name=pgk_screen_100k
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=screening_%j.log
#SBATCH --error=screening_%j.err

# ──────────────────────────────────────────────────────────────────────────────
# ComputeCanada SLURM Job Script for Parallelized Screening
# ──────────────────────────────────────────────────────────────────────────────
# 
# Resources:
#   - 1 GPU (for UniMol embeddings & model inference, batch_size=256)
#   - 12 CPU cores (for parallel RDKit conformer generation)
#   - 32GB memory (sufficient for GPU + CPU multiprocessing)
#   - 2 hours (max 2h for 100k compounds with chunking)
#
# Usage:
#   sbatch computecanada_job.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f screening_JOBID.log
#
# ──────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "ComputeCanada GPU Screening Job Starting"
echo "=========================================="
echo ""

# Load modules
echo "Loading environment modules..."
module load cuda/12.6
module load python/3.10

echo "Environment loaded."
echo ""

# Navigate to project directory
cd "$HOME/enamine_screen" || {
    echo "ERROR: Could not navigate to project directory"
    exit 1
}

echo "Project directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Allocated CPU cores: $SLURM_CPUS_PER_TASK"
echo ""

# Point UniMol to local weights (no internet needed on compute nodes)
export UNIMOL_WEIGHT_DIR="$HOME/enamine_screen/weights"

# Activate virtualenv
echo "Activating virtualenv 'unimol'..."
source "$HOME/envs/unimol/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtualenv at ~/envs/unimol"
    echo "Set it up first: module load python/3.10 && python -m venv ~/envs/unimol"
    exit 1
fi

echo "✓ Conda environment activated"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Run parallelized screening
# ──────────────────────────────────────────────────────────────────────────────

echo "Starting parallelized screening..."
echo "  - Config: config_diversity_screen.yaml"
echo "  - Compound chunks: 20,000 each"
echo "  - CPU workers: $SLURM_CPUS_PER_TASK"
echo "  - GPU batch size: 256"
echo ""

python -u parallelized_screen_enamine.py \
    --config config_diversity_screen.yaml \
    --num_workers $SLURM_CPUS_PER_TASK \
    --chunk_size 20000

RESULT=$?

echo ""
echo "=========================================="
if [ $RESULT -eq 0 ]; then
    echo "✓ Screening completed successfully"
else
    echo "✗ Screening failed with exit code $RESULT"
fi
echo "=========================================="

exit $RESULT
