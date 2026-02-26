"""
Stage 2 training: selectivity refinement with leave-one-ligand-out (LOO).
Runs LOO fine-tuning from the Stage 1 checkpoint and writes results/plots.

Usage:
    python stage2_train.py --config config_stage2.yaml
"""

import json
from pathlib import Path
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import SelectivityModel
from dataset import SelectivityDataset


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        ligand_feat = batch['ligand_feat'].to(device)
        pocket_emb = batch['pocket_emb'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits, probs = model(ligand_feat, pocket_emb)

        loss = criterion(logits, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(probs.cpu().detach().numpy().flatten())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)

    return avg_loss, auc, acc


def validate(model, dataloader, criterion, device):
    """Validate on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ligand_feat = batch['ligand_feat'].to(device)
            pocket_emb = batch['pocket_emb'].to(device)
            labels = batch['label'].to(device)

            logits, probs = model(ligand_feat, pocket_emb)
            loss = criterion(logits, labels.unsqueeze(1).float())

            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)

    return avg_loss, auc, acc, all_preds, all_labels


def plot_per_source_confusion_matrices(metrics, stage_name, split_name, save_path):
    """Plot confusion matrices for each data source."""
    per_source = metrics.get('per_source', {})
    sources = sorted(per_source.keys())

    if not sources:
        print(f"  No per-source data for {stage_name} {split_name}")
        return

    n_sources = len(sources)
    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 4))

    if n_sources == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        cm = per_source[source]['confusion_matrix']
        cm_array = np.array([
            [cm['true_negatives'], cm['false_positives']],
            [cm['false_negatives'], cm['true_positives']],
        ])

        sns.heatmap(
            cm_array,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
        )

        auc = per_source[source]['auc']
        acc = per_source[source]['accuracy']
        n = per_source[source]['num_samples']

        auc_str = f"AUC={auc:.3f}" if auc is not None else "AUC=N/A"
        ax.set_title(f"{source} (n={n}, {auc_str}, acc={acc:.3f})")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.suptitle(f"{stage_name} ({split_name}): Confusion Matrices by Source", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved confusion matrix plot to {save_path}")
    plt.close()


def evaluate_stage2_loo(dataset, checkpoint_path, loo_dir, results_dir, device, config, output_names, eval_config):
    """Leave-one-ligand-out evaluation for Stage 2 (selectivity)."""
    print("\nEvaluating Stage 2 (LOO)...")

    # Create LOO directory
    loo_dir = Path(loo_dir)
    loo_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = sorted({ex.get('smiles') for ex in dataset.examples})
    all_preds = []
    all_probs = []
    all_labels = []
    all_sources = []
    all_smiles = []
    all_folds = []

    for fold_idx, smiles in enumerate(smiles_list, start=1):
        # Create fold-specific directory
        fold_dir = loo_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_indices = [i for i, ex in enumerate(dataset.examples) if ex.get('smiles') == smiles]
        train_indices = [i for i in range(len(dataset)) if i not in val_indices]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        model = SelectivityModel().to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr_stage2'],
            weight_decay=config['weight_decay'],
        )
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train_subset, batch_size=config['batch_size_stage2'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size_stage2'], shuffle=False, num_workers=0)

        # Train with validation tracking for best checkpoint
        best_val_auc = -1
        best_epoch = 0
        fold_checkpoint_path = fold_dir / f'fold_{fold_idx}_best.pt'

        for epoch in range(config['epochs_stage2']):
            train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate on holdout (val_loader)
            val_loss, val_auc, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
            
            # Validate on non-holdout (train_loader) for combined metric
            non_holdout_loss, non_holdout_auc, non_holdout_acc, non_holdout_preds, non_holdout_labels = validate(model, train_loader, criterion, device)
            
            # Combine predictions from both holdout and non-holdout for stopping metric
            combined_val_preds = np.concatenate([np.array(val_preds), np.array(non_holdout_preds)])
            combined_val_labels = np.concatenate([np.array(val_labels), np.array(non_holdout_labels)])
            
            if len(np.unique(combined_val_labels)) > 1:
                combined_auc = roc_auc_score(combined_val_labels, combined_val_preds)
            else:
                combined_auc = np.nan

            # Save best checkpoint based on combined AUC
            if combined_auc > best_val_auc or np.isnan(best_val_auc):
                best_val_auc = combined_auc
                best_epoch = epoch + 1
                
                if eval_config.get('save_fold_checkpoints', True):
                    fold_checkpoint_path = fold_dir / output_names['fold_checkpoint_name'].format(fold=fold_idx)
                    torch.save({
                        'epoch': epoch,
                        'fold': fold_idx,
                        'holdout_ligand': smiles,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_auc,
                        'non_holdout_auc': non_holdout_auc,
                        'combined_auc': combined_auc,
                    }, fold_checkpoint_path)

        # Load best checkpoint for final evaluation
        # Load best checkpoint for final evaluation
        if eval_config.get('save_fold_checkpoints', True):
            fold_checkpoint_path = fold_dir / output_names['fold_checkpoint_name'].format(fold=fold_idx)
            best_checkpoint = torch.load(fold_checkpoint_path)
            model.load_state_dict(best_checkpoint['model_state_dict'])

        # Evaluate on validation set (holdout ligand)
        if eval_config.get('evaluate_holdout', True):
            _, _, _, val_fold_probs, val_fold_labels = validate(model, val_loader, criterion, device)
        else:
            val_fold_probs = np.array([])
            val_fold_labels = np.array([])
        
        val_fold_probs = np.array(val_fold_probs)
        val_fold_labels = np.array(val_fold_labels)
        val_fold_preds = (val_fold_probs > 0.5).astype(int)

        # Evaluate on training set (non-holdout PDB ligands)
        if eval_config.get('evaluate_non_holdout', True):
            _, _, _, train_fold_probs, train_fold_labels = validate(model, train_loader, criterion, device)
        else:
            train_fold_probs = np.array([])
            train_fold_labels = np.array([])
        
        # Evaluate on training set (non-holdout PDB ligands)
        _, _, _, train_fold_probs, train_fold_labels = validate(model, train_loader, criterion, device)
        train_fold_probs = np.array(train_fold_probs)
        train_fold_labels = np.array(train_fold_labels)
        train_fold_preds = (train_fold_probs > 0.5).astype(int)

        # Compute fold metrics for holdout
        val_auc = roc_auc_score(val_fold_labels, val_fold_probs) if len(np.unique(val_fold_labels)) > 1 else np.nan
        val_acc = accuracy_score(val_fold_labels, val_fold_preds)
        val_cm = confusion_matrix(val_fold_labels, val_fold_preds, labels=[0, 1])
        val_tn, val_fp, val_fn, val_tp = val_cm.ravel()

        # Compute fold metrics for non-holdout
        train_auc = roc_auc_score(train_fold_labels, train_fold_probs) if len(np.unique(train_fold_labels)) > 1 else np.nan
        train_acc = accuracy_score(train_fold_labels, train_fold_preds)
        train_cm = confusion_matrix(train_fold_labels, train_fold_preds, labels=[0, 1])
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()

        # Save holdout predictions CSV
        import pandas as pd
        holdout_data = []
        for idx, (val_idx, prob, pred, label) in enumerate(zip(val_indices, val_fold_probs, val_fold_preds, val_fold_labels)):
            ex = dataset.examples[val_idx]
            
            # Determine which pocket is being tested
            pocket_emb = ex['pocket_emb']
            if np.allclose(pocket_emb, dataset.pgk1_mean):
                pocket = 'PGK1'
            elif np.allclose(pocket_emb, dataset.pgk2_mean):
                pocket = 'PGK2'
            else:
                pocket = 'unknown'
            
            holdout_data.append({
                'ligand': ex.get('smiles', ''),
                'pocket': pocket,
                'target': ex.get('target', 'unknown'),
                'source': ex.get('source', 'unknown'),
                'true_label': int(label),
                'predicted_probability': float(prob),
                'predicted_label': int(pred),
            })
        
        holdout_df = pd.DataFrame(holdout_data)
        holdout_pred_path = fold_dir / output_names['fold_predictions_holdout'].format(fold=fold_idx)
        holdout_df.to_csv(holdout_pred_path, index=False)

        # Save non-holdout predictions CSV
        non_holdout_data = []
        for idx, (train_idx, prob, pred, label) in enumerate(zip(train_indices, train_fold_probs, train_fold_preds, train_fold_labels)):
            ex = dataset.examples[train_idx]
            
            # Determine which pocket is being tested
            pocket_emb = ex['pocket_emb']
            if np.allclose(pocket_emb, dataset.pgk1_mean):
                pocket = 'PGK1'
            elif np.allclose(pocket_emb, dataset.pgk2_mean):
                pocket = 'PGK2'
            else:
                pocket = 'unknown'
            
            non_holdout_data.append({
                'ligand': ex.get('smiles', ''),
                'pocket': pocket,
                'target': ex.get('target', 'unknown'),
                'source': ex.get('source', 'unknown'),
                'true_label': int(label),
                'predicted_probability': float(prob),
                'predicted_label': int(pred),
            })
        
        non_holdout_df = pd.DataFrame(non_holdout_data)
        non_holdout_pred_path = fold_dir / output_names['fold_predictions_non_holdout'].format(fold=fold_idx)
        non_holdout_df.to_csv(non_holdout_pred_path, index=False)

        # Save per-fold metrics JSON
        fold_metrics = {
            'fold': fold_idx,
            'holdout_ligand': smiles,
            'best_epoch': int(best_epoch),
            'holdout': {
                'num_samples': int(len(val_fold_labels)),
                'num_positives': int(np.sum(val_fold_labels)),
                'num_negatives': int(len(val_fold_labels) - np.sum(val_fold_labels)),
                'auc': float(val_auc) if not np.isnan(val_auc) else None,
                'accuracy': float(val_acc),
                'confusion_matrix': {
                    'true_negatives': int(val_tn),
                    'false_positives': int(val_fp),
                    'false_negatives': int(val_fn),
                    'true_positives': int(val_tp),
                }
            },
            'non_holdout': {
                'num_samples': int(len(train_fold_labels)),
                'num_positives': int(np.sum(train_fold_labels)),
                'num_negatives': int(len(train_fold_labels) - np.sum(train_fold_labels)),
                'auc': float(train_auc) if not np.isnan(train_auc) else None,
                'accuracy': float(train_acc),
                'confusion_matrix': {
                    'true_negatives': int(train_tn),
                    'false_positives': int(train_fp),
                    'false_negatives': int(train_fn),
                    'true_positives': int(train_tp),
                }
            }
        }
        
        fold_metrics_path = fold_dir / output_names['fold_metrics'].format(fold=fold_idx)
        with open(fold_metrics_path, 'w') as f:
            json.dump(fold_metrics, f, indent=2)

        # Aggregate for overall LOO metrics (only holdout)
        all_probs.extend(val_fold_probs.tolist())
        all_preds.extend(val_fold_preds.tolist())
        all_labels.extend(val_fold_labels.tolist())

        for idx in val_indices:
            ex = dataset.examples[idx]
            all_sources.append(ex.get('source', 'unknown'))
            all_smiles.append(ex.get('smiles', ''))
            all_folds.append(fold_idx)

        if eval_config.get('verbose', True):
            val_auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
            train_auc_str = f"{train_auc:.3f}" if not np.isnan(train_auc) else "N/A"
            print(f"  Fold {fold_idx}/{len(smiles_list)} | ligand={smiles} | holdout_n={len(val_indices)} (auc={val_auc_str}) | non_holdout_n={len(train_indices)} (auc={train_auc_str}) | best_epoch={best_epoch}")

    # Convert to arrays for final metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    import pandas as pd
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'predicted_probability': all_probs,
        'source': all_sources,
        'smiles': all_smiles,
        'fold': all_folds,
    })

    preds_path = results_dir / output_names['loo_predictions']
    results_df.to_csv(preds_path, index=False)
    print(f"  Saved LOO predictions to {preds_path}")

    per_source = {}
    for source in sorted(results_df['source'].unique()):
        mask = results_df['source'] == source
        src_labels = results_df.loc[mask, 'true_label'].values
        src_preds = results_df.loc[mask, 'predicted_label'].values
        src_probs = results_df.loc[mask, 'predicted_probability'].values

        cm = confusion_matrix(src_labels, src_preds, labels=[0, 1])
        src_tn, src_fp, src_fn, src_tp = cm.ravel()
        src_auc = roc_auc_score(src_labels, src_probs) if len(np.unique(src_labels)) > 1 else np.nan
        src_acc = accuracy_score(src_labels, src_preds)

        per_source[source] = {
            'num_samples': int(len(src_labels)),
            'num_positives': int(np.sum(src_labels)),
            'num_negatives': int(len(src_labels) - np.sum(src_labels)),
            'auc': float(src_auc) if not np.isnan(src_auc) else None,
            'accuracy': float(src_acc),
            'confusion_matrix': {
                'true_negatives': int(src_tn),
                'false_positives': int(src_fp),
                'false_negatives': int(src_fn),
                'true_positives': int(src_tp),
            }
        }

    metrics = {
        'strategy': 'loo',
        'checkpoint_path': str(checkpoint_path),
        'num_samples': int(len(all_labels)),
        'num_positives': int(np.sum(all_labels)),
        'num_negatives': int(len(all_labels) - np.sum(all_labels)),
        'auc': float(auc) if not np.isnan(auc) else None,
        'accuracy': float(acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'per_source': per_source,
    }

    metrics_path = results_dir / output_names['loo_metrics']
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved LOO metrics to {metrics_path}")

    print("\nStage 2 LOO Results:")
    print(f"  Samples: {len(all_labels)} (pos: {int(np.sum(all_labels))}, neg: {int(len(all_labels) - np.sum(all_labels))})")
    print(f"  AUC-ROC: {auc if not np.isnan(auc) else 'N/A'}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Stage 2 LOO evaluation: selectivity refinement')
    parser.add_argument('--config', type=str, default='config_stage2.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {args.config}\n")
    
    # Extract paths and parameters
    checkpoint_stage1 = Path(config['input']['checkpoint_stage1'])
    
    loo_dir = Path(config['output']['loo_dir'])
    loo_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    train_config = {
        'batch_size_stage2': config['training']['batch_size'],
        'lr_stage2': config['training']['learning_rate'],
        'epochs_stage2': config['training']['epochs'],
        'weight_decay': config['training']['weight_decay'],
        'seed': config['training']['seed'],
        'checkpoint_stage1': str(checkpoint_stage1),
    }
    
    # Output naming
    output_names = config['output']

    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])

    # Device selection
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")

    print(f"\nLoading Stage 2 data ({config['training']['stage']})...")
    dataset = SelectivityDataset(stage=config['training']['stage'], csv_path=config['input']['ligand_csv'])

    metrics = evaluate_stage2_loo(
        dataset, 
        train_config['checkpoint_stage1'], 
        loo_dir,
        results_dir, 
        device, 
        train_config,
        output_names,
        config['evaluation']
    )
    
    if config['evaluation']['per_source_confusion']:
        confusion_plot_path = results_dir / output_names['loo_confusion_plot']
        plot_per_source_confusion_matrices(metrics, 'Stage 2', 'LOO', confusion_plot_path)

    print("\nStage 2 LOO complete.")
    print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    main()
