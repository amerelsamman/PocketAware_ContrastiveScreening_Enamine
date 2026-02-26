# PGK1/PGK2 Selectivity Training Pipeline

This project trains a pocket-conditioned model to predict binding selectivity between PGK1 and PGK2.

## Quick Start

### 1. Prepare Ligand Features

Pre-compute Uni-Mol embeddings for all ligands:

```bash
conda activate unimol
python prepare_ligand_features.py --config config_prepare_features.yaml
```

**Config:** `config_prepare_features.yaml`
- Input: CSV paths with SMILES
- Output: PKL and CSV with embeddings
- Uni-Mol parameters: data_type, remove_hs
- RDKit conformer generation settings

### 2. Train Stage 1 (Binding Classification)

Train on DEL hits + decoys + PDB ligands:

```bash
python stage1_train.py --config config_stage1.yaml
```

**Config:** `config_stage1.yaml`
- Input: ligand CSV, pre-computed features, pocket embeddings
- Output: checkpoint, training history, predictions, metrics, plots
- Training: epochs, batch_size, learning_rate, val_split
- Evaluation: per-source confusion matrices, training curves

**Outputs:**
- `models/v0/stage1_best.pt` - Best checkpoint (by val AUC)
- `models/v0/stage1_best_history.json` - Epoch-wise metrics
- `models/v0/results/` - Predictions, metrics, confusion matrices

### 3. Evaluate Stage 2 (LOO on PDB Ligands)

Leave-one-ligand-out evaluation for selectivity:

```bash
python stage2_train.py --config config_stage2.yaml
```

**Config:** `config_stage2.yaml`
- Input: Stage 1 checkpoint path (configurable!), ligand CSV, features
- Output: per-fold checkpoints, holdout/non-holdout predictions
- Training: epochs, batch_size, learning_rate for fine-tuning
- Evaluation: verbose logging, per-fold metrics

**Outputs:**
- `models/v0/loo/fold_X/` - Per-fold checkpoints and results
  - `fold_X_best.pt` - Best checkpoint for this fold
  - `fold_X_predictions_holdout.csv` - Held-out ligand predictions
  - `fold_X_predictions_non_holdout.csv` - Training ligands predictions
  - `fold_X_metrics.json` - Metrics for both splits
- `models/v0/results/` - Aggregated LOO results

## Configuration Files

### `config_prepare_features.yaml`

Control ligand featurization:
- **Input:** CSV paths, max_smiles limit
- **Output:** Cache directory, filenames
- **Uni-Mol:** data_type, remove_hs
- **RDKit:** Conformer generation parameters

Example:
```yaml
input:
  csv_paths:
    - "data/ligands/smiles_binding.csv"
  max_smiles: null  # Process all

output:
  cache_dir: "data/ligand_features_cache"
  pkl_filename: "ligand_features_unimol.pkl"

unimol:
  data_type: "molecule"
  remove_hs: true
```

### `config_stage1.yaml`

Control Stage 1 training:
- **Version:** Model version for output directory naming
- **Input:** Ligand CSV, feature paths, pocket embeddings
- **Output:** Checkpoint directory, results directory, filenames
- **Training:** epochs, batch_size, learning_rate, val_split, seed
- **Evaluation:** Toggle train/val results, per-source confusion, plots

Example:
```yaml
version: "v0"

training:
  epochs: 50
  batch_size: 64
  learning_rate: 3.0e-4
  val_split: 0.2
  stratify: true

evaluation:
  save_train_results: true
  save_val_results: true
  per_source_confusion: true
```

### `config_stage2.yaml`

Control Stage 2 LOO evaluation:
- **Version:** Model version for output directory naming
- **Input:** **Stage 1 checkpoint path** (choose any checkpoint!), data paths
- **Output:** LOO directory, results directory, per-fold naming templates
- **Training:** epochs, batch_size, learning_rate for fine-tuning per fold
- **Evaluation:** Toggle holdout/non-holdout evaluation, verbose logging

Example:
```yaml
version: "v0"

input:
  checkpoint_stage1: "models/v0/stage1_best.pt"  # <-- Point to any checkpoint!

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4

evaluation:
  evaluate_holdout: true
  evaluate_non_holdout: true
  save_fold_checkpoints: true
```

## Key Features

### Transparency & Control

All scripts use YAML configs for full control over:
- ✅ Input/output paths
- ✅ Hyperparameters
- ✅ Evaluation strategy
- ✅ Which results to save

### Stratified Training

Stage 1 uses stratified 80/20 split by source+label:
- Ensures balanced DEL/DECOY/PDB in train/val
- Prevents source-specific overfitting

### LOO Evaluation

Stage 2 uses leave-one-ligand-out:
- 15 folds (one per PDB ligand)
- Saves per-fold checkpoints
- Evaluates on both holdout AND non-holdout (detects overfitting)

### Per-Source Metrics

Both stages track metrics per data source:
- DEL: PGK2-selective hits
- DECOY: Non-binders
- PDB: Co-crystal ligands with known targets

## File Structure

```
.
├── config_prepare_features.yaml    # Featurization config
├── config_stage1.yaml               # Stage 1 training config
├── config_stage2.yaml               # Stage 2 LOO config
├── prepare_ligand_features.py       # Pre-compute Uni-Mol embeddings
├── stage1_train.py                  # Stage 1 training script
├── stage2_train.py                  # Stage 2 LOO evaluation script
├── model.py                         # SelectivityModel architecture
├── dataset.py                       # SelectivityDataset loader
├── DNA.txt                          # Project design documentation
├── data/
│   ├── ligands/
│   │   └── smiles_binding.csv      # All ligand SMILES
│   ├── ligand_features_cache/
│   │   └── ligand_features_unimol.pkl
│   └── pocket_embeddings_cache/
│       └── pocket_embeddings_unimol_cutoff8A.pkl
└── models/
    └── v0/
        ├── stage1_best.pt           # Best Stage 1 checkpoint
        ├── stage1_best_history.json # Training history
        ├── loo/
        │   └── fold_X/              # Per-fold results
        └── results/                 # All metrics & plots
```

## Advanced Usage

### Train Multiple Versions

Create different configs for ablation studies:

```bash
# High learning rate experiment
python stage1_train.py --config config_stage1_highlr.yaml

# Different validation split
python stage1_train.py --config config_stage1_valplit30.yaml
```

### Point Stage 2 at Different Checkpoints

Test LOO with different Stage 1 models:

```bash
# Evaluate fold 10 checkpoint
python stage2_train.py --config config_stage2_from_fold10.yaml
```

Where `config_stage2_from_fold10.yaml` has:
```yaml
input:
  checkpoint_stage1: "models/v0-ablation/stage1_epoch10.pt"
```

### Custom Output Directories

Organize experiments with version naming:

```yaml
version: "v1-no-decoys"  # Creates models/v1-no-decoys/
```

## Tips

1. **Always check configs before running** - Paths are explicit and transparent
2. **Use version naming for experiments** - Keeps models/results organized
3. **Stage 2 `checkpoint_stage1` is flexible** - Point to any trained model
4. **Per-fold results show overfitting** - Compare holdout vs non-holdout AUC
5. **Configs are self-documenting** - YAML comments explain each parameter

## Citation

See `DNA.txt` for detailed project design and methodology.
